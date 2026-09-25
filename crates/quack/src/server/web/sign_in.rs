//! `GET /login/oidc` sends the browser to the issuer; the issuer sends it
//! back to `GET /login/oidc/callback`, which opens a session.

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Redirect, Response};
use axum_extra::extract::CookieJar;
use axum_extra::extract::cookie::{Cookie, SameSite};
use quack_core::config::OidcConfig;
use quack_core::error::Error as CoreError;
use quack_core::oidc::SignedIn;
use quack_core::storage::control::{AuditAction, AuditEntry, Channel, Outcome};
use serde::Deserialize;

use super::WebResult;
use super::flash::Flash;
use crate::server::auth::{Peer, RequestId, SessionCookie};
use crate::server::error::ApiError;
use crate::server::oidc::{Oidc, PENDING_TTL, STATE_COOKIE};
use crate::server::state::App;

/// What the issuer's redirect carries.
#[derive(Deserialize)]
pub(super) struct Callback {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

fn configured(app: &App) -> WebResult<&Oidc> {
    app.oidc.as_ref().ok_or_else(|| {
        ApiError::not_found("sign-in through an identity provider is not configured").into()
    })
}

/// The cookie that removes the state cookie, however the callback ends.
fn cleared_state() -> Cookie<'static> {
    Cookie::build(STATE_COOKIE)
        .path(OidcConfig::CALLBACK_PATH)
        .build()
}

pub(super) async fn begin(
    State(app): State<App>,
    peer: Peer,
    jar: CookieJar,
) -> WebResult<Response> {
    let oidc = configured(&app)?;
    let (url, state) = oidc.start().await?;
    let cookie = Cookie::build((STATE_COOKIE, state))
        .path(OidcConfig::CALLBACK_PATH)
        .http_only(true)
        // The issuer's redirect back is a top-level GET from another site,
        // which `Lax` still sends the cookie with.
        .same_site(SameSite::Lax)
        .secure(peer.needs_secure());
    let cookie = match PENDING_TTL.try_into() {
        Ok(max_age) => cookie.max_age(max_age).build(),
        Err(_) => cookie.build(),
    };
    Ok((jar.add(cookie), Redirect::to(&url)).into_response())
}

/// Why a callback signed nobody in.
enum Refusal {
    /// Something to tell the person on the login page.
    Shown(String),
    /// A failure on quack's side, shown as an error page.
    Failed(CoreError),
}

impl Callback {
    /// Match the callback to the sign-in this browser started, then
    /// exchange its code.
    async fn complete(self, oidc: &Oidc, started_here: Option<&str>) -> Result<SignedIn, Refusal> {
        let (Some(state), Some(cookie)) = (self.state.as_deref(), started_here) else {
            return Err(Refusal::Shown(String::from(
                "the sign-in could not be matched to this browser; try again",
            )));
        };
        if state != cookie {
            return Err(Refusal::Shown(String::from(
                "the sign-in could not be matched to this browser; try again",
            )));
        }
        let Some(pending) = oidc.take(state) else {
            return Err(Refusal::Shown(String::from(
                "the sign-in took too long or was already used; try again",
            )));
        };
        if let Some(error) = &self.error {
            let detail = self
                .error_description
                .as_deref()
                .map_or(String::new(), |d| format!(" ({d})"));
            return Err(Refusal::Shown(format!(
                "the identity provider refused the sign-in: {error}{detail}"
            )));
        }
        let Some(code) = &self.code else {
            return Err(Refusal::Shown(String::from(
                "the identity provider sent no code",
            )));
        };
        oidc.finish(code, pending).await.map_err(|e| match e {
            CoreError::SignIn(reason) => Refusal::Shown(reason),
            other => Refusal::Failed(other),
        })
    }
}

pub(super) async fn finish(
    State(app): State<App>,
    peer: Peer,
    RequestId(request_id): RequestId,
    jar: CookieJar,
    Query(callback): Query<Callback>,
) -> WebResult<Response> {
    let oidc = configured(&app)?;
    let started_here = jar.get(STATE_COOKIE).map(|c| c.value().to_owned());
    let jar = jar.remove(cleared_state());
    let mut entry = AuditEntry::new(AuditAction::Login, Outcome::Denied, Channel::Web);
    entry.client_addr = peer.ip();
    entry.request_id = request_id;

    let signed_in = match callback.complete(oidc, started_here.as_deref()).await {
        Ok(signed_in) => signed_in,
        Err(Refusal::Shown(reason)) => {
            tracing::warn!(%reason, "sign-in refused");
            app.control.record_audit(&entry).await?;
            return Ok((jar, Flash::error("/login", reason)).into_response());
        }
        Err(Refusal::Failed(e)) => {
            app.control.record_audit(&entry).await?;
            return Err(e.into());
        }
    };

    let user = app
        .control
        .oidc_user(&signed_in.subject, &signed_in.username)
        .await?;
    let renew_at = oidc.keep(&app.control, &user.id, &signed_in.token).await?;
    entry.outcome = Outcome::Allowed;
    entry.user_id = Some(user.id.clone());
    app.control.record_audit(&entry).await?;
    let token = app.sessions.open(&user.id, renew_at)?;
    Ok((
        jar.add(SessionCookie::issue(&app, peer, token)),
        Redirect::to("/workspaces"),
    )
        .into_response())
}
