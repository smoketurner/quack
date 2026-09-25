//! Password login for browsers and scripts: a session token, also set as
//! the `quack_session` cookie.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum_extra::extract::CookieJar;
use quack_core::storage::control::{AuditAction, Outcome};
use serde::Deserialize;

use crate::server::auth::{
    Credential, Identity, Login, Peer, RequestId, SessionCookie, password_login,
};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::{App, ServeMode};

#[derive(Deserialize)]
pub(crate) struct LoginRequest {
    pub username: String,
    pub password: String,
}

pub(crate) async fn login(
    State(app): State<App>,
    peer: Peer,
    jar: CookieJar,
    request_id: RequestId,
    Json(body): Json<LoginRequest>,
) -> ApiResult<impl IntoResponse> {
    if app.mode == ServeMode::Local {
        return Err(ApiError::bad_request("local mode has no login"));
    }
    let Login { user, token } =
        password_login(&app, peer, request_id, &body.username, &body.password).await?;
    Ok((
        jar.add(SessionCookie::issue(&app, peer, token.clone())),
        Json(serde_json::json!({ "token": token.as_str(), "user": user })),
    ))
}

pub(crate) async fn logout(
    State(app): State<App>,
    identity: Identity,
    jar: CookieJar,
) -> ApiResult<impl IntoResponse> {
    let jar = identity.log_out(&app, jar).await?;
    Ok((jar, StatusCode::NO_CONTENT))
}

impl Identity {
    /// End this login, from the API or the web console: the session is
    /// closed, the logout audited, and the session cookie cleared.
    pub(crate) async fn log_out(&self, app: &App, jar: CookieJar) -> ApiResult<CookieJar> {
        if let Credential::Session(token) = &self.credential {
            app.sessions.close(token.as_str());
        }
        // A signed-in user's token is kept for their sessions; the last one
        // closing is the end of quack's use for it.
        if let Some(oidc) = &app.oidc
            && !app.sessions.has_sessions(&self.user_id)
        {
            oidc.forget(&app.control, &self.user_id).await?;
        }
        let entry = self.audit(AuditAction::Logout, Outcome::Allowed);
        app.control.record_audit(&entry).await?;
        Ok(jar.remove(SessionCookie::clear()))
    }
}

pub(crate) async fn me(identity: Identity) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "id": identity.user_id,
        "username": identity.username,
        "is_admin": identity.is_admin,
        "via": match identity.credential {
            Credential::Local => "local",
            Credential::Session(_) => "session",
            Credential::Token(_) => "token",
        },
    }))
}
