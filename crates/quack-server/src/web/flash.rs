//! Where a web form lands once it is handled: back to a page, with at most
//! one message for that page's flash slot, and the table to open when it is
//! the Tables page. Both can name workspace content, so they never travel in
//! the URL, which request logs, proxies, and browser history keep: they wait
//! in this process's memory under a random id, and the browser carries only
//! the id, in a short-lived cookie the landing page reads and spends.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use axum::extract::{FromRequestParts, Request, State};
use axum::http::request::Parts;
use axum::middleware::Next;
use axum::response::{IntoResponse, Redirect, Response};
use axum_extra::extract::CookieJar;
use axum_extra::extract::cookie::{Cookie, SameSite};
use quack_core::storage::control;

use crate::auth::Peer;
use crate::error::ApiError;
use crate::state::App;

const FLASH_COOKIE: &str = "quack_flash";

/// A redirect after a form, carrying at most one message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Flash {
    path: String,
    stash: Stash,
}

/// What a redirect hands the page it lands on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Stash {
    message: Option<(Level, String)>,
    /// The table the Tables page opens.
    table: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Level {
    Notice,
    Error,
}

impl Flash {
    /// Back to `path` with nothing to say.
    pub(crate) fn to(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            stash: Stash::default(),
        }
    }

    /// Back to `path` with a success message.
    pub(crate) fn notice(path: impl Into<String>, text: impl Into<String>) -> Self {
        Self::to(path).with(Level::Notice, text)
    }

    /// Back to `path` with what went wrong.
    pub(crate) fn error(path: impl Into<String>, text: impl Into<String>) -> Self {
        Self::to(path).with(Level::Error, text)
    }

    /// Back to `path`: with `success`'s message when the operation worked,
    /// else with its error's.
    pub(crate) fn after<T>(
        path: impl Into<String>,
        result: Result<T, ApiError>,
        success: impl FnOnce(T) -> Option<String>,
    ) -> Self {
        match result {
            Ok(value) => match success(value) {
                Some(text) => Self::notice(path, text),
                None => Self::to(path),
            },
            Err(e) => Self::error(path, e.message),
        }
    }

    /// Open `table` on the page it lands on.
    pub(crate) fn opening(mut self, table: impl Into<String>) -> Self {
        self.stash.table = Some(table.into());
        self
    }

    fn with(mut self, level: Level, text: impl Into<String>) -> Self {
        self.stash.message = Some((level, text.into()));
        self
    }
}

impl IntoResponse for Flash {
    /// The redirect, with the stash for [`keep`] to move into the store.
    fn into_response(self) -> Response {
        let mut response = Redirect::to(&self.path).into_response();
        if self.stash != Stash::default() {
            response.extensions_mut().insert(self.stash);
        }
        response
    }
}

/// Stashes between a redirect and the page it lands on, in memory only.
#[derive(Debug, Default)]
pub(crate) struct Flashes {
    waiting: Mutex<HashMap<String, (Instant, Stash)>>,
}

impl Flashes {
    /// How long a stash waits for its page: a redirect is followed at once,
    /// so anything older was abandoned.
    const TTL: Duration = Duration::from_secs(60);

    /// Keep `stash` under a fresh random id, dropping any that expired.
    fn put(&self, stash: Stash) -> Option<String> {
        let mut bytes = [0u8; 16];
        control::random_bytes(&mut bytes).ok()?;
        let id = base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, bytes);
        let mut waiting = self.waiting.lock().unwrap_or_else(PoisonError::into_inner);
        waiting.retain(|_, (at, _)| at.elapsed() < Self::TTL);
        waiting.insert(id.clone(), (Instant::now(), stash));
        Some(id)
    }

    /// The stash under `id`, removed: a flash shows once.
    fn take(&self, id: &str) -> Option<Stash> {
        let mut waiting = self.waiting.lock().unwrap_or_else(PoisonError::into_inner);
        waiting
            .remove(id)
            .filter(|(at, _)| at.elapsed() < Self::TTL)
            .map(|(_, stash)| stash)
    }
}

/// Middleware: move a redirect's stash into the store and hand the browser
/// its id in a cookie that lives as long as the stash does.
pub(crate) async fn keep(
    State(app): State<App>,
    peer: Peer,
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    let Some(stash) = response.extensions_mut().remove::<Stash>() else {
        return response;
    };
    let Some(id) = app.flashes.put(stash) else {
        tracing::warn!("no random id for a flash message; it is dropped");
        return response;
    };
    let cookie = Cookie::build((FLASH_COOKIE, id))
        .path("/")
        .http_only(true)
        .same_site(SameSite::Lax)
        .secure(peer.needs_secure(&app.config.server));
    let cookie = match Flashes::TTL.try_into() {
        Ok(max_age) => cookie.max_age(max_age).build(),
        // Unreachable for a one-minute constant; without `Max-Age` the
        // cookie still dies with the browser session.
        Err(_) => cookie.build(),
    };
    (CookieJar::new().add(cookie), response).into_response()
}

/// The stash a request's flash cookie names, taken from the store.
#[derive(Debug, Default)]
pub(crate) struct Flashed(Stash);

impl Flashed {
    pub(crate) fn error(&self) -> Option<String> {
        self.text(Level::Error)
    }

    pub(crate) fn notice(&self) -> Option<String> {
        self.text(Level::Notice)
    }

    /// The table a redirect asked the Tables page to open.
    pub(crate) fn table(&self) -> Option<String> {
        self.0.table.clone()
    }

    fn text(&self, level: Level) -> Option<String> {
        self.0
            .message
            .as_ref()
            .filter(|(l, _)| *l == level)
            .map(|(_, text)| text.clone())
    }
}

impl FromRequestParts<App> for Flashed {
    type Rejection = std::convert::Infallible;

    fn from_request_parts(
        parts: &mut Parts,
        state: &App,
    ) -> impl Future<Output = Result<Self, Self::Rejection>> {
        let stash = CookieJar::from_headers(&parts.headers)
            .get(FLASH_COOKIE)
            .and_then(|cookie| state.flashes.take(cookie.value()))
            .unwrap_or_default();
        std::future::ready(Ok(Self(stash)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flash_redirects_to_its_path_and_keeps_its_message_out_of_the_url() {
        let response = Flash::error("/w/x/tables", "'secret.csv' is empty").into_response();
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::LOCATION)
                .and_then(|v| v.to_str().ok()),
            Some("/w/x/tables")
        );
        assert_eq!(
            response.extensions().get::<Stash>(),
            Some(&Stash {
                message: Some((Level::Error, String::from("'secret.csv' is empty"))),
                table: None,
            })
        );
        assert!(
            Flash::to("/w/x/ontology")
                .into_response()
                .extensions()
                .get::<Stash>()
                .is_none()
        );
    }

    #[test]
    fn a_stash_is_taken_once() {
        let flashes = Flashes::default();
        let id = flashes
            .put(Flash::notice("/p", "3 merged").opening("orders").stash)
            .unwrap_or_default();
        assert_eq!(id.len(), 22, "16 random bytes, base64url: {id}");
        let taken = Flashed(flashes.take(&id).unwrap_or_default());
        assert_eq!(taken.notice().as_deref(), Some("3 merged"));
        assert_eq!(taken.error(), None);
        assert_eq!(taken.table().as_deref(), Some("orders"));
        assert!(flashes.take(&id).is_none());
        assert!(flashes.take("not-an-id").is_none());
    }

    #[test]
    fn after_picks_the_message_by_outcome() {
        let failed: Result<(), ApiError> = Err(ApiError::bad_request("no rows"));
        assert_eq!(
            Flash::after("/t", failed, |()| None),
            Flash::error("/t", "no rows")
        );
        assert_eq!(
            Flash::after("/t", Ok(2), |n| Some(format!("{n} loaded"))),
            Flash::notice("/t", "2 loaded")
        );
        assert_eq!(Flash::after("/t", Ok(()), |()| None), Flash::to("/t"));
    }
}
