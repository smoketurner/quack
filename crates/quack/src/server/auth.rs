//! Who is calling, and what they may do in a workspace.
//!
//! [`Identity`] is an extractor: `--local` yields the implicit owner; a
//! bearer that is a login session or an API token, or the session cookie,
//! yields a user. [`Access`] then checks membership, role, and token scope
//! for one workspace and writes the denied audit row itself, so a handler
//! that gets an `Access` back has already been authorized.

use axum::extract::{ConnectInfo, FromRequestParts};
use axum::http::request::Parts;
use axum::http::{StatusCode, header};
use axum_extra::extract::CookieJar;
use quack_core::storage::control::{
    AuditEntry, Channel, Outcome, Role, Scope, TokenRow, WorkspaceRow, sha256_hex,
};
use std::net::SocketAddr;

use super::error::{ApiError, ApiResult};
use super::state::App;

pub(crate) const SESSION_COOKIE: &str = "quack_session";
pub(crate) const REQUEST_ID_HEADER: &str = "x-request-id";

/// The implicit user in `--local` mode.
pub(crate) const LOCAL_USER_ID: &str = "local";

/// How the caller authenticated.
#[derive(Debug, Clone)]
pub(crate) enum Credential {
    Local,
    /// A browser or API-login session.
    Session(String),
    /// An API token; the row carries its workspace and scopes.
    Token(TokenRow),
}

#[derive(Debug, Clone)]
pub(crate) struct Identity {
    pub user_id: String,
    pub username: String,
    pub is_admin: bool,
    pub credential: Credential,
    pub client_addr: Option<String>,
    pub request_id: Option<String>,
}

impl Identity {
    pub(crate) fn channel(&self) -> Channel {
        match self.credential {
            Credential::Token(_) => Channel::Api,
            Credential::Local | Credential::Session(_) => Channel::Web,
        }
    }

    fn token_hash(&self) -> Option<String> {
        match &self.credential {
            Credential::Token(t) => Some(t.token_hash.clone()),
            Credential::Local | Credential::Session(_) => None,
        }
    }

    /// An audit entry attributed to this caller.
    pub(crate) fn audit(&self, action: &str, outcome: Outcome) -> AuditEntry {
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

fn bearer(parts: &Parts) -> Option<String> {
    parts
        .headers
        .get(header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .map(|t| t.trim().to_owned())
}

fn request_meta(parts: &Parts) -> (Option<String>, Option<String>) {
    let addr = parts
        .extensions
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0.ip().to_string());
    let request_id = parts
        .headers
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    (addr, request_id)
}

impl FromRequestParts<App> for Identity {
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, app: &App) -> Result<Self, Self::Rejection> {
        let (client_addr, request_id) = request_meta(parts);
        if app.local {
            return Ok(Self {
                user_id: String::from(LOCAL_USER_ID),
                username: String::from(LOCAL_USER_ID),
                is_admin: true,
                credential: Credential::Local,
                client_addr,
                request_id,
            });
        }

        let presented = match bearer(parts) {
            Some(token) => Some(token),
            None => CookieJar::from_headers(&parts.headers)
                .get(SESSION_COOKIE)
                .map(|c| c.value().to_owned()),
        };
        let Some(presented) = presented else {
            return Err(ApiError::unauthorized("log in or send a bearer token"));
        };

        if let Some(user_id) = app.web_session_user(&presented) {
            let user = app
                .control
                .get_user(&user_id)
                .await?
                .ok_or_else(|| ApiError::unauthorized("session user no longer exists"))?;
            return Ok(Self {
                user_id: user.id,
                username: user.username,
                is_admin: user.is_admin,
                credential: Credential::Session(presented),
                client_addr,
                request_id,
            });
        }

        let hash = sha256_hex(presented.as_bytes());
        let Some(token) = app.control.find_token(&hash).await? else {
            let mut entry = AuditEntry::new("token", Outcome::Denied, Channel::Api);
            entry.client_addr = client_addr;
            entry.request_id = request_id;
            app.control.record_audit(&entry).await?;
            return Err(ApiError::unauthorized("unknown token"));
        };
        let now = jiff::Timestamp::now()
            .strftime("%Y-%m-%d %H:%M:%S")
            .to_string();
        if token.is_expired(&now) {
            let mut entry = AuditEntry::new("token", Outcome::Denied, Channel::Api);
            entry.user_id = Some(token.user_id.clone());
            entry.token_hash = Some(token.token_hash.clone());
            entry.workspace_id = Some(token.workspace_id.clone());
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
    /// Owners and admins see every session; others see their own.
    pub(crate) fn sees_all_sessions(&self) -> bool {
        self.identity.is_admin || self.role == Some(Role::Owner)
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

    /// Record an access-audit row for this workspace, and its content
    /// detail inside the workspace when `detail` is given.
    pub(crate) async fn audit(
        &self,
        app: &App,
        action: &str,
        resource: Option<(&str, &str)>,
        outcome: Outcome,
        detail: Option<serde_json::Value>,
    ) -> ApiResult<String> {
        let mut entry = self.identity.audit(action, outcome);
        entry.workspace_id = Some(self.workspace.id.clone());
        if let Some((kind, id)) = resource {
            entry.resource_type = Some(kind.to_owned());
            entry.resource_id = Some(id.to_owned());
        }
        app.control.record_audit(&entry).await?;
        if let Some(detail) = detail {
            let db = app.workspace_db(&self.workspace.id).await?;
            let id = entry.id.clone();
            let user = self.identity.user_id.clone();
            let action = action.to_owned();
            super::state::with_db(db, move |db| {
                quack_core::storage::audit::record(db, &id, Some(&user), &action, &detail)
            })
            .await?;
        }
        Ok(entry.id)
    }
}

/// Resolve the workspace and check the caller against `need`, writing a
/// denied audit row and returning 403/404 when they fall short.
pub(crate) async fn access(
    app: &App,
    identity: Identity,
    workspace_id: &str,
    need: Need,
) -> ApiResult<Access> {
    let Some(workspace) = app.control.get_workspace(workspace_id).await? else {
        let mut entry = identity.audit("open", Outcome::Denied);
        entry.workspace_id = Some(workspace_id.to_owned());
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
        deny(
            app,
            &identity,
            &workspace,
            "token is scoped to another workspace",
        )
        .await?;
    }
    let access = Access {
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
        deny(app, &access.identity, &access.workspace, reason).await?;
    }
    Ok(access)
}

async fn deny(
    app: &App,
    identity: &Identity,
    workspace: &WorkspaceRow,
    reason: &str,
) -> ApiResult<()> {
    let mut entry = identity.audit("open", Outcome::Denied);
    entry.workspace_id = Some(workspace.id.clone());
    app.control.record_audit(&entry).await?;
    Err(ApiError::forbidden(reason))
}

/// Server admins only; everything else is 403.
pub(crate) fn require_admin(identity: &Identity) -> ApiResult<()> {
    if identity.is_admin {
        Ok(())
    } else {
        Err(ApiError::new(StatusCode::FORBIDDEN, "admin only"))
    }
}
