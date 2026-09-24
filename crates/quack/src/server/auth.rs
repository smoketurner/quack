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
use axum::http::{HeaderMap, header};
use axum_extra::extract::CookieJar;
use axum_extra::extract::cookie::{Cookie, SameSite};
use quack_core::ids::{AuditId, UserId, WorkspaceId};
use quack_core::storage::audit::AuditDetail;
use quack_core::storage::control::{
    AuditAction, AuditEntry, AuditResource, Channel, Outcome, Role, Scope, TokenRow, UserRow,
    WorkspaceRow, sha256_hex,
};
use quack_core::storage::sessions::SessionViewer;
use std::convert::Infallible;
use std::net::SocketAddr;

use super::error::{ApiError, ApiResult};
use super::state::{App, SessionLookup, SessionToken};

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
}

#[derive(Debug, Clone)]
pub(crate) struct Identity {
    pub user_id: UserId,
    pub username: String,
    pub is_admin: bool,
    pub credential: Credential,
    pub client_addr: Option<String>,
    pub request_id: Option<String>,
    /// Set for requests that arrived over a transport of their own (MCP),
    /// so audit rows name that channel rather than the credential's.
    pub channel: Option<Channel>,
}

impl Identity {
    pub(crate) fn channel(&self) -> Channel {
        self.channel.unwrap_or(match self.credential {
            Credential::Token(_) => Channel::Api,
            Credential::Local | Credential::Session(_) => Channel::Web,
        })
    }

    fn token_hash(&self) -> Option<String> {
        match &self.credential {
            Credential::Token(t) => Some(t.token_hash.clone()),
            Credential::Local | Credential::Session(_) => None,
        }
    }

    /// An audit entry attributed to this caller.
    pub(crate) fn audit(&self, action: AuditAction, outcome: Outcome) -> AuditEntry {
        let mut entry = AuditEntry::new(action, outcome, self.channel());
        entry.user_id = Some(self.user_id.clone());
        entry.token_hash = self.token_hash();
        entry.client_addr.clone_from(&self.client_addr);
        entry.request_id.clone_from(&self.request_id);
        entry
    }

    /// Whether an API token restricts this caller below `scope`. Sessions
    /// and local mode carry every scope.
    pub(crate) fn lacks_scope(&self, scope: Scope) -> bool {
        match &self.credential {
            Credential::Token(t) => !t.has_scope(scope) && !t.has_scope(Scope::Admin),
            Credential::Local | Credential::Session(_) => false,
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
    /// go back in the clear. Loopback is the plain-HTTP local case, and an
    /// unknown peer is treated the same way.
    fn needs_secure(self) -> bool {
        self.0.is_some_and(|addr| !addr.ip().is_loopback())
    }

    fn ip(self) -> Option<String> {
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
            .secure(peer.needs_secure());
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
) -> ApiResult<(UserRow, SessionToken)> {
    let verified = app.control.verify_password(username, password).await?;
    let mut entry = AuditEntry::new(
        LOGIN_ACTION,
        if verified.is_some() {
            Outcome::Allowed
        } else {
            Outcome::Denied
        },
        Channel::Web,
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
    entry.client_addr = peer.ip();
    entry.request_id = request_id;
    app.control.record_audit(&entry).await?;

    let Some(user) = verified else {
        return Err(ApiError::unauthorized("wrong username or password"));
    };
    let token = app.sessions.open(&user.id)?;
    Ok((user, token))
}

impl FromRequestParts<App> for Identity {
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, app: &App) -> Result<Self, Self::Rejection> {
        let client_addr = Peer::of(parts).ip();
        let RequestId(request_id) = RequestId::of(&parts.headers);
        if app.local {
            return Ok(Self {
                user_id: UserId::from(LOCAL_USER_ID),
                username: String::from(LOCAL_USER_ID),
                is_admin: true,
                credential: Credential::Local,
                client_addr,
                request_id,
                channel: None,
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
            SessionLookup::Active(user_id) => {
                let user = app
                    .control
                    .get_user(&user_id)
                    .await?
                    .ok_or_else(|| ApiError::unauthorized("session user no longer exists"))?;
                return Ok(Self {
                    user_id: user.id,
                    username: user.username,
                    is_admin: user.is_admin,
                    credential: Credential::Session(SessionToken::presented(presented)),
                    client_addr,
                    request_id,
                    channel: None,
                });
            }
            // Saying so, rather than falling through to "unknown token",
            // is what lets a browser tell an expired login from a bad one.
            SessionLookup::Expired => {
                let mut entry =
                    AuditEntry::new(AuditAction::Session, Outcome::Denied, Channel::Web);
                entry.client_addr = client_addr;
                entry.request_id = request_id;
                app.control.record_audit(&entry).await?;
                return Err(ApiError::unauthorized("session expired; log in again"));
            }
            SessionLookup::Unknown => {}
        }

        let hash = sha256_hex(presented.as_bytes());
        let Some(token) = app.control.find_token(&hash).await? else {
            let mut entry = AuditEntry::new(AuditAction::Token, Outcome::Denied, Channel::Api);
            entry.client_addr = client_addr;
            entry.request_id = request_id;
            app.control.record_audit(&entry).await?;
            return Err(ApiError::unauthorized("unknown token"));
        };
        if token.is_expired(jiff::Timestamp::now()) {
            let mut entry = AuditEntry::new(AuditAction::Token, Outcome::Denied, Channel::Api);
            entry.user_id = Some(token.user_id.clone());
            entry.token_hash = Some(token.token_hash.clone());
            entry = entry.in_workspace(&token.workspace_id);
            entry.client_addr = client_addr;
            entry.request_id = request_id;
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
            is_admin: user.is_admin,
            credential: Credential::Token(token),
            client_addr,
            request_id,
            channel: None,
        })
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
        self.identity.is_admin || self.role == Some(Role::Owner)
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
            None => need.admin_ok && self.identity.is_admin,
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
        let mut entry = self.identity.audit(action, outcome);
        entry = entry.in_workspace(&self.workspace.id);
        if let Some(resource) = resource {
            entry = entry.on(resource);
        }
        app.control.record_audit(&entry).await?;
        let detail = detail.unwrap_or_else(|| serde_json::json!({}));
        // Its own connection: a request never waits for a write in
        // progress on the writer just to record that it happened.
        app.audit_log(&self.workspace.id)
            .await?
            .record(AuditDetail {
                id: entry.id.clone(),
                user_id: Some(self.identity.user_id.clone()),
                action: action.to_string(),
                detail,
            })
            .await?;
        Ok(entry.id)
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
        let role = if app.local {
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
                None if access.identity.is_admin => "admins read workspace content only as members",
                None => "not a member of this workspace",
                Some(_) if access.identity.lacks_scope(need.scope) => "token lacks the scope",
                Some(_) => "role does not allow this",
            };
            access.identity.deny(app, &access.workspace, reason).await?;
        }
        Ok(access)
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
        if self.is_admin {
            Ok(())
        } else {
            Err(ApiError::forbidden("admin only"))
        }
    }
}
