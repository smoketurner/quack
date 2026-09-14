//! Password login for browsers and scripts: a session token, also set as
//! the `quack_session` cookie.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum_extra::extract::CookieJar;
use axum_extra::extract::cookie::{Cookie, SameSite};
use quack_core::storage::control::{AuditEntry, Channel, Outcome};
use serde::Deserialize;

use crate::server::auth::{Credential, Identity, SESSION_COOKIE};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::App;

#[derive(Deserialize)]
pub(crate) struct LoginRequest {
    pub username: String,
    pub password: String,
}

pub(crate) async fn login(
    State(app): State<App>,
    jar: CookieJar,
    Json(body): Json<LoginRequest>,
) -> ApiResult<impl IntoResponse> {
    if app.local {
        return Err(ApiError::bad_request("local mode has no login"));
    }
    let user = app
        .control
        .verify_password(&body.username, &body.password)
        .await?;
    let Some(user) = user else {
        let mut entry = AuditEntry::new("login", Outcome::Denied, Channel::Web);
        entry.user_id = app
            .control
            .find_user_by_username(&body.username)
            .await?
            .map(|u| u.id);
        app.control.record_audit(&entry).await?;
        return Err(ApiError::unauthorized("wrong username or password"));
    };
    let token = app.open_web_session(&user.id)?;
    let mut entry = AuditEntry::new("login", Outcome::Allowed, Channel::Web);
    entry.user_id = Some(user.id.clone());
    app.control.record_audit(&entry).await?;
    let cookie = Cookie::build((SESSION_COOKIE, token.clone()))
        .path("/")
        .http_only(true)
        .same_site(SameSite::Lax)
        .build();
    Ok((
        jar.add(cookie),
        Json(serde_json::json!({ "token": token, "user": user })),
    ))
}

pub(crate) async fn logout(
    State(app): State<App>,
    identity: Identity,
    jar: CookieJar,
) -> ApiResult<impl IntoResponse> {
    if let Credential::Session(token) = &identity.credential {
        app.close_web_session(token);
    }
    let mut entry = identity.audit("logout", Outcome::Allowed);
    entry.workspace_id = None;
    app.control.record_audit(&entry).await?;
    Ok((
        jar.remove(Cookie::build(SESSION_COOKIE).path("/").build()),
        StatusCode::NO_CONTENT,
    ))
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
