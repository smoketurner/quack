//! Who is calling, and what they may do in a workspace.
//!
//! [`Identity`] is an extractor: `--local` yields the implicit owner; a
//! bearer that is a login session or an API token, or the session cookie,
//! yields a user. [`Access`] then checks membership, role, and token scope
//! for one workspace and writes the denied audit row itself, so a handler
//! that gets an `Access` back has already been authorized.
//!
//! [`password_login`] is the one place a password is checked, for both the
//! JSON API and the browser form (issue #73).

use axum::extract::{ConnectInfo, FromRequestParts};
use axum::http::request::Parts;
use axum::http::{HeaderMap, StatusCode, header};
use axum_extra::extract::CookieJar;
use axum_extra::extract::cookie::{Cookie, SameSite};
use quack_core::config::ServerConfig;
use quack_core::crypto::sha256_hex;
use quack_core::error::Error as CoreError;
use quack_core::ids::{AuditId, UserId, WorkspaceId};
use quack_core::llm::egress::Egress;
use quack_core::storage::audit::AuditDetail;
use quack_core::storage::control::{
    AuditAction, AuditEntry, AuditResource, Channel, Origin, Outcome, Role, Scope, TokenRow,
    UserKind, UserRow, WorkspaceRow,
};
use quack_core::storage::sessions::SessionViewer;
use quack_core::web_sessions::{SessionLookup, SessionToken};
use std::convert::Infallible;
use std::net::SocketAddr;

use super::error::{ApiError, ApiResult};
use super::oidc::Oidc;
use super::state::{App, ServeMode};

pub(crate) const SESSION_COOKIE: &str = "quack_session";
pub(crate) const REQUEST_ID_HEADER: &str = "x-request-id";

/// The implicit user in `--local` mode.
pub(crate) const LOCAL_USER_ID: &str = "local";

/// The audit action both login paths record, under either outcome.
const LOGIN_ACTION: AuditAction = AuditAction::Login;

/// How the caller authenticated.
#[derive(Debug, Clone)]
pub(crate) enum Credential {
    Local,
    /// A browser or API-login session.
    Session(SessionToken),
    /// An API token; the row carries its workspace and scopes.
    Token(TokenRow),
    /// An access token from `[server.oidc]`'s issuer, presented as a bearer
    /// (RFC 9728); it carries the user's own access, like a session.
    IdentityProvider,
}

#[derive(Debug, Clone)]
pub(crate) struct Identity {
    pub user_id: UserId,
    pub username: String,
    pub kind: UserKind,
    pub credential: Credential,
    /// Where the request came from; its channel is the credential's (a
    /// session or local mode is `Web`, a token `Api`) unless the request
    /// arrived over a transport of its own (MCP).
    pub origin: Origin,
}

impl Identity {
    fn token_hash(&self) -> Option<String> {
        match &self.credential {
            Credential::Token(t) => Some(t.token_hash.clone()),
            Credential::Local | Credential::Session(_) | Credential::IdentityProvider => None,
        }
    }

    /// An audit entry attributed to this caller.
    pub(crate) fn audit(&self, action: AuditAction, outcome: Outcome) -> AuditEntry {
        let mut entry = AuditEntry::new(action, outcome, self.origin.clone());
        entry.user_id = Some(self.user_id.clone());
        entry.token_hash = self.token_hash();
        entry
    }

    /// Whether an API token restricts this caller below `scope`. Sessions
    /// and local mode carry every scope.
    pub(crate) fn lacks_scope(&self, scope: Scope) -> bool {
        match &self.credential {
            Credential::Token(t) => !t.has_scope(scope) && !t.has_scope(Scope::Admin),
            Credential::Local | Credential::Session(_) | Credential::IdentityProvider => false,
        }
    }
}

/// The address the request came from, when the server recorded one. In
/// production `into_make_service_with_connect_info` always does; the
/// `oneshot` tests never do.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Peer(pub Option<SocketAddr>);

impl<S: Send + Sync> FromRequestParts<S> for Peer {
    type Rejection = Infallible;

    fn from_request_parts(
        parts: &mut Parts,
        _: &S,
    ) -> impl Future<Output = Result<Self, Self::Rejection>> {
        std::future::ready(Ok(Self::of(parts)))
    }
}

impl Peer {
    fn of(parts: &Parts) -> Self {
        Self(
            parts
                .extensions
                .get::<ConnectInfo<SocketAddr>>()
                .map(|info| info.0),
        )
    }

    /// Whether a cookie set on this response must carry `Secure`. Anything
    /// that did not come from loopback may have crossed a network, including
    /// the hop in front of a TLS-terminating proxy, so the cookie must never
    /// go back in the clear. Loopback (and an unknown peer, treated the same
    /// way) is the plain-HTTP local case, unless the server knows its public
    /// URL is https or was told to always set it: a proxy on the same host
    /// also arrives on loopback (issue #246).
    pub(crate) fn needs_secure(self, server: &ServerConfig) -> bool {
        self.0.is_some_and(|addr| !addr.ip().is_loopback()) || server.secure_cookies_on_loopback()
    }

    pub(crate) fn ip(self) -> Option<String> {
        self.0.map(|addr| addr.ip().to_string())
    }
}

/// The id the request-id layer put on the request, for audit rows: the
/// login paths have no [`Identity`] yet to carry it.
#[derive(Debug, Clone, Default)]
pub(crate) struct RequestId(pub Option<String>);

impl RequestId {
    fn of(headers: &HeaderMap) -> Self {
        Self(
            headers
                .get(REQUEST_ID_HEADER)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned),
        )
    }
}

impl<S: Send + Sync> FromRequestParts<S> for RequestId {
    type Rejection = Infallible;

    fn from_request_parts(
        parts: &mut Parts,
        _: &S,
    ) -> impl Future<Output = Result<Self, Self::Rejection>> {
        std::future::ready(Ok(Self::of(&parts.headers)))
    }
}

/// The browser's login session cookie, `quack_session`.
pub(crate) struct SessionCookie;

impl SessionCookie {
    /// The cookie for a freshly opened session. `Max-Age` matches the
    /// session's absolute lifetime, so the browser drops it when the server
    /// would rather than holding a token that can only be refused.
    pub(crate) fn issue(app: &App, peer: Peer, token: SessionToken) -> Cookie<'static> {
        let cookie = Cookie::build((SESSION_COOKIE, token.into_string()))
            .path("/")
            .http_only(true)
            .same_site(SameSite::Lax)
            .secure(peer.needs_secure(&app.config.server));
        match app.config.server.session_max_age().try_into() {
            Ok(max_age) => cookie.max_age(max_age).build(),
            // Out of range only for a lifetime no operator can configure. A
            // cookie without `Max-Age` still dies with the browser session,
            // and the server expires the session itself regardless.
            Err(_) => cookie.build(),
        }
    }

    /// The cookie that removes it, at logout.
    pub(crate) fn clear() -> Cookie<'static> {
        Cookie::build(SESSION_COOKIE).path("/").build()
    }

    /// The session token a request carries in it.
    fn read(headers: &HeaderMap) -> Option<String> {
        CookieJar::from_headers(headers)
            .get(SESSION_COOKIE)
            .map(|c| c.value().to_owned())
    }
}

/// Check a password and open a session, for both login paths. Either
/// outcome is audited with the address and request id the caller came in
/// with; the rate limiter on the two login routes is what makes guessing
/// expensive.
pub(crate) async fn password_login(
    app: &App,
    peer: Peer,
    RequestId(request_id): RequestId,
    username: &str,
    password: &str,
) -> ApiResult<Login> {
    let verified = app.control.verify_password(username, password).await?;
    let origin = Origin {
        channel: Channel::Web,
        client_addr: peer.ip(),
        request_id,
    };
    let mut entry = AuditEntry::new(
        LOGIN_ACTION,
        if verified.is_some() {
            Outcome::Allowed
        } else {
            Outcome::Denied
        },
        origin,
    );
    entry.user_id = match &verified {
        Some(user) => Some(user.id.clone()),
        // Name the account a wrong password was aimed at, when there is one.
        None => app
            .control
            .find_user_by_username(username)
            .await?
            .map(|u| u.id),
    };
    app.control.record_audit(&entry).await?;

    let Some(user) = verified else {
        return Err(ApiError::unauthorized("wrong username or password"));
    };
    let token = app.sessions.open(&user.id, None)?;
    Ok(Login { user, token })
}

/// A password login that succeeded: who, and their new session.
pub(crate) struct Login {
    pub(crate) user: UserRow,
    pub(crate) token: SessionToken,
}

impl FromRequestParts<App> for Identity {
    type Rejection = ApiError;

    /// Who is calling. When quack is a protected resource, a 401 says where
    /// to get a token, and `invalid_token` when one was presented.
    async fn from_request_parts(parts: &mut Parts, state: &App) -> Result<Self, Self::Rejection> {
        let identity = Self::resolve(parts, state)
            .await
            .map_err(|e| match &state.resource {
                Some(resource) if e.status == StatusCode::UNAUTHORIZED => {
                    let refused = parts.headers.contains_key(header::AUTHORIZATION)
                        || SessionCookie::read(&parts.headers).is_some();
                    let challenge = resource.challenge(parts.uri.path(), refused);
                    e.with_challenge(challenge)
                }
                _ => e,
            })?;
        // Model requests this request makes, and jobs it submits, act for
        // this person at an on-behalf-of provider. Local mode is nobody's.
        if state.mode != ServeMode::Local
            && let Some(oidc) = &state.oidc
        {
            oidc.acting(&identity.user_id, identity.origin.clone())
                .enter();
        }
        Ok(identity)
    }
}

impl Identity {
    async fn resolve(parts: &Parts, app: &App) -> ApiResult<Self> {
        let RequestId(request_id) = RequestId::of(&parts.headers);
        let origin = Origin {
            channel: Channel::Web,
            client_addr: Peer::of(parts).ip(),
            request_id,
        };
        if app.mode == ServeMode::Local {
            return Ok(Self {
                user_id: UserId::from(LOCAL_USER_ID),
                username: String::from(LOCAL_USER_ID),
                kind: UserKind::Admin,
                credential: Credential::Local,
                origin,
            });
        }

        let bearer = parts
            .headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(|t| t.trim().to_owned());
        let presented = bearer.or_else(|| SessionCookie::read(&parts.headers));
        let Some(presented) = presented else {
            return Err(ApiError::unauthorized("log in or send a bearer token"));
        };

        match app.sessions.lookup(&presented) {
            SessionLookup::Active {
                user_id,
                renewal_due,
            } => {
                if renewal_due && let Some(oidc) = &app.oidc {
                    oidc.require_current(&app.control, &user_id, &presented, &origin)
                        .await?;
                }
                let user = app
                    .control
                    .get_user(&user_id)
                    .await?
                    .ok_or_else(|| ApiError::unauthorized("session user no longer exists"))?;
                return Ok(Self {
                    user_id: user.id,
                    username: user.username,
                    kind: user.kind,
                    credential: Credential::Session(SessionToken::presented(presented)),
                    origin,
                });
            }
            // Saying so, rather than falling through to "unknown token",
            // is what lets a browser tell an expired login from a bad one.
            SessionLookup::Expired => {
                let entry = AuditEntry::new(AuditAction::Session, Outcome::Denied, origin);
                app.control.record_audit(&entry).await?;
                return Err(ApiError::unauthorized("session expired; log in again"));
            }
            SessionLookup::Unknown => {}
        }

        let origin = Origin {
            channel: Channel::Api,
            ..origin
        };
        if let Some(oidc) = app.oidc.as_ref().filter(|o| o.accepts_bearers())
            && presented.split('.').count() == 3
        {
            return Self::from_access_token(app, oidc, &presented, origin).await;
        }

        Self::from_api_token(app, &presented, origin).await
    }

    /// A bearer that is one of quack's API tokens.
    async fn from_api_token(app: &App, presented: &str, origin: Origin) -> ApiResult<Self> {
        let hash = sha256_hex(presented.as_bytes());
        let Some(token) = app.control.find_token(&hash).await? else {
            let entry = AuditEntry::new(AuditAction::Token, Outcome::Denied, origin);
            app.control.record_audit(&entry).await?;
            return Err(ApiError::unauthorized("unknown token"));
        };
        if token.is_expired(jiff::Timestamp::now()) {
            let mut entry = AuditEntry::new(AuditAction::Token, Outcome::Denied, origin);
            entry.user_id = Some(token.user_id.clone());
            entry.token_hash = Some(token.token_hash.clone());
            entry = entry.in_workspace(&token.workspace_id);
            app.control.record_audit(&entry).await?;
            return Err(ApiError::unauthorized("token expired"));
        }
        app.control.touch_token(&token.token_hash).await?;
        let user = app
            .control
            .get_user(&token.user_id)
            .await?
            .ok_or_else(|| ApiError::unauthorized("token user no longer exists"))?;
        Ok(Self {
            user_id: user.id,
            username: user.username,
            kind: user.kind,
            credential: Credential::Token(token),
            origin,
        })
    }

    /// A bearer that is an access token from the issuer: verified, and the
    /// user it names found or created with no access, as a sign-in would.
    async fn from_access_token(
        app: &App,
        oidc: &Oidc,
        token: &str,
        origin: Origin,
    ) -> ApiResult<Self> {
        match oidc.bearer_user(&app.control, token).await {
            Ok(user) => Ok(Self {
                user_id: user.id,
                username: user.username,
                kind: user.kind,
                credential: Credential::IdentityProvider,
                origin,
            }),
            Err(CoreError::Bearer(reason)) => {
                tracing::info!(%reason, "access token refused");
                let entry = AuditEntry::new(AuditAction::Token, Outcome::Denied, origin);
                app.control.record_audit(&entry).await?;
                Err(ApiError::unauthorized(format!(
                    "access token refused: {reason}"
                )))
            }
            Err(e) => Err(e.into()),
        }
    }
}

/// What a handler needs in a workspace.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Need {
    pub role: Role,
    pub scope: Scope,
    /// Whether a server admin without membership is allowed (settings and
    /// members, never content).
    pub admin_ok: bool,
}

impl Need {
    pub(crate) const READ: Self = Self {
        role: Role::Viewer,
        scope: Scope::Read,
        admin_ok: false,
    };
    /// Reading, or a server admin without membership: settings and
    /// members, never content.
    pub(crate) const READ_OR_ADMIN: Self = Self {
        admin_ok: true,
        ..Self::READ
    };
    pub(crate) const WRITE: Self = Self {
        role: Role::Member,
        scope: Scope::Write,
        admin_ok: false,
    };
    pub(crate) const OWN: Self = Self {
        role: Role::Owner,
        scope: Scope::Admin,
        admin_ok: true,
    };
}

/// An authorized view of one workspace for one caller.
#[derive(Debug, Clone)]
pub(crate) struct Access {
    pub identity: Identity,
    pub workspace: WorkspaceRow,
    /// `None` for an admin acting without membership.
    pub role: Option<Role>,
}

impl Access {
    /// Whether the caller may act on something `owner` created: their own,
    /// or anyone's as a workspace owner or an admin.
    pub(crate) fn owns(&self, owner: Option<&UserId>) -> bool {
        owner == Some(&self.identity.user_id) || self.sees_all_sessions()
    }

    /// Owners and admins see every session; others see their own.
    fn sees_all_sessions(&self) -> bool {
        self.identity.kind == UserKind::Admin || self.role == Some(Role::Owner)
    }

    /// Which sessions the caller may read.
    pub(crate) fn session_viewer(&self) -> SessionViewer {
        if self.sees_all_sessions() {
            SessionViewer::All
        } else {
            SessionViewer::User(self.identity.user_id.clone())
        }
    }

    /// Whether the caller meets `need`, for a second check inside a handler
    /// (a write statement on the SQL endpoint, `allow_write` on a query).
    pub(crate) fn permits(&self, need: Need) -> bool {
        if self.identity.lacks_scope(need.scope) {
            return false;
        }
        match self.role {
            Some(role) => role >= need.role,
            None => need.admin_ok && self.identity.kind == UserKind::Admin,
        }
    }

    /// Record an access-audit row for this workspace and its content
    /// detail inside the workspace under the same id (issue #54: both
    /// halves, every time; a failed write fails the request). `resource`
    /// carries an opaque id only, never a table name or content; those
    /// go in `detail`, which stays inside the workspace.
    pub(crate) async fn audit(
        &self,
        app: &App,
        action: AuditAction,
        resource: Option<AuditResource<'_>>,
        outcome: Outcome,
        detail: Option<serde_json::Value>,
    ) -> ApiResult<AuditId> {
        let mut entry = self.entry(action, outcome);
        if let Some(resource) = resource {
            entry = entry.on(resource);
        }
        app.control.record_audit(&entry).await?;
        self.record_detail(app, &entry, detail).await?;
        Ok(entry.id)
    }

    /// An `audit_log` entry for this caller in this workspace, for a change
    /// that commits it itself.
    pub(crate) fn entry(&self, action: AuditAction, outcome: Outcome) -> AuditEntry {
        self.identity
            .audit(action, outcome)
            .in_workspace(&self.workspace.id)
    }

    /// The `_quack_audit` half of an `audit_log` row already written.
    pub(crate) async fn record_detail(
        &self,
        app: &App,
        entry: &AuditEntry,
        detail: Option<serde_json::Value>,
    ) -> ApiResult<()> {
        // Its own connection: a request never waits for a write in
        // progress on the writer just to record that it happened.
        app.audit_log(&self.workspace.id)
            .await?
            .record(AuditDetail {
                id: entry.id.clone(),
                user_id: Some(self.identity.user_id.clone()),
                action: entry.action.to_string(),
                detail: detail.unwrap_or_else(|| serde_json::json!({})),
            })
            .await?;
        Ok(())
    }

    /// The allowed row for a read that returns a listing or a page:
    /// action `list` or `page`, what was read in the detail.
    pub(crate) async fn audit_read(
        &self,
        app: &App,
        action: AuditAction,
        what: &str,
    ) -> ApiResult<()> {
        self.audit(
            app,
            action,
            None,
            Outcome::Allowed,
            Some(serde_json::json!({ "what": what })),
        )
        .await?;
        Ok(())
    }
}

impl Access {
    /// Resolve the workspace and check the caller against `need`, writing a
    /// denied audit row and returning 403/404 when they fall short.
    pub(crate) async fn resolve(
        app: &App,
        identity: Identity,
        workspace_id: &WorkspaceId,
        need: Need,
    ) -> ApiResult<Self> {
        let Some(workspace) = app.control.get_workspace(workspace_id).await? else {
            let mut entry = identity.audit(AuditAction::Open, Outcome::Denied);
            entry = entry.in_workspace(workspace_id);
            app.control.record_audit(&entry).await?;
            return Err(ApiError::not_found("no such workspace"));
        };
        let role = if app.mode == ServeMode::Local {
            Some(Role::Owner)
        } else {
            app.control
                .member_role(&workspace.id, &identity.user_id)
                .await?
        };
        if let Credential::Token(token) = &identity.credential
            && token.workspace_id != workspace.id
        {
            identity
                .deny(app, &workspace, "token is scoped to another workspace")
                .await?;
        }
        let access = Self {
            identity,
            workspace,
            role,
        };
        if !access.permits(need) {
            let reason = match access.role {
                None if access.identity.kind == UserKind::Admin => {
                    "admins read workspace content only as members"
                }
                None => "not a member of this workspace",
                Some(_) if access.identity.lacks_scope(need.scope) => "token lacks the scope",
                Some(_) => "role does not allow this",
            };
            access.identity.deny(app, &access.workspace, reason).await?;
        }
        // From here on the request, and every job it submits, sends only
        // to the model providers the workspace allows.
        Egress::Workspace(access.workspace.allowed_providers.clone()).enter();
        Ok(access)
    }

    /// The model `built` for work on this workspace, or why not: a provider
    /// the workspace's allow-list refuses is 403 and a denied row for
    /// `action`, any other failure an error row, and the detail says which.
    pub(crate) async fn model<T>(
        &self,
        app: &App,
        action: AuditAction,
        built: Result<T, CoreError>,
    ) -> ApiResult<T> {
        let error = match built {
            Ok(model) => return Ok(model),
            Err(error) => error,
        };
        self.audit(
            app,
            action,
            None,
            Outcome::of_failure(&error),
            Some(serde_json::json!({ "error": error.to_string() })),
        )
        .await?;
        Err(ApiError::from(error))
    }
}

impl Identity {
    /// Record a refused attempt on `workspace` and return 403 with `reason`.
    async fn deny(&self, app: &App, workspace: &WorkspaceRow, reason: &str) -> ApiResult<()> {
        let mut entry = self.audit(AuditAction::Open, Outcome::Denied);
        entry = entry.in_workspace(&workspace.id);
        app.control.record_audit(&entry).await?;
        Err(ApiError::forbidden(reason))
    }

    /// Server admins only; everything else is 403.
    pub(crate) fn require_admin(&self) -> ApiResult<()> {
        if self.kind == UserKind::Admin {
            Ok(())
        } else {
            Err(ApiError::forbidden("admin only"))
        }
    }
}
