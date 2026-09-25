//! Sign-in through the organization's `OpenID` Connect issuer: the sign-ins
//! in flight, and each signed-in user's token.
//!
//! A user's token is kept by core's `oidc::SubjectTokens`, sealed in
//! `control.db`. Its refresh token is what ties a quack session to the
//! issuer: when the token runs out, the first request that finds it renews
//! it, and a refusal ends every session the user has.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jiff::{SignedDuration, Timestamp};
use quack_core::config::OidcConfig;
use quack_core::error::Result as CoreResult;
use quack_core::ids::UserId;
use quack_core::llm::acting::Acting;
use quack_core::llm::oauth::CachedToken;
use quack_core::oidc::{
    Pending, RENEW_MARGIN, SignIn, SignedIn, Stored, SubjectTokens, UserTokens,
};
use quack_core::storage::control::{AuditEntry, ControlPlane, UserRow};
use quack_core::vault::Vault;

use super::error::{ApiError, ApiResult};
use super::state::{AppState, WebSessions};

/// The cookie that ties a callback to the browser that started the sign-in,
/// so a callback link someone else started cannot sign this browser in.
pub(crate) const STATE_COOKIE: &str = "quack_oidc_state";

/// How long a sign-in may take at the issuer.
pub(crate) const PENDING_TTL: Duration = Duration::from_secs(600);

/// Sign-ins in flight at once. Each is a few hundred bytes, and starting one
/// costs nothing but a request, so the count is bounded.
const MAX_PENDING: usize = 10_000;

/// After the issuer could not be reached, how long until it is asked again.
const RETRY_AFTER: SignedDuration = SignedDuration::from_secs(60);

/// Whether a session's sign-in still stands after a renewal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Standing {
    Current,
    /// The issuer refused, or the user's token is gone: every session of the
    /// user has been closed.
    Ended,
}

/// The server's sign-in client and what it keeps.
pub(crate) struct Oidc {
    sign_in: Arc<SignIn>,
    /// By `state`: the sign-in and when it began.
    pending: Mutex<HashMap<String, (Pending, Instant)>>,
    /// Each signed-in user's own token (the on-behalf-of subject).
    subjects: Arc<SubjectTokens>,
}

impl Oidc {
    /// # Errors
    ///
    /// Returns an error when the HTTP client cannot be built.
    pub(crate) fn new(
        config: &OidcConfig,
        vault: Vault,
        control: ControlPlane,
    ) -> CoreResult<Self> {
        let sign_in = Arc::new(SignIn::new(config.clone())?);
        Ok(Self {
            subjects: Arc::new(SubjectTokens::new(
                Arc::clone(&sign_in),
                UserTokens::new(vault),
                control,
            )),
            sign_in,
            pending: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn issuer_host(&self) -> String {
        self.sign_in.issuer_host()
    }

    /// The acting person for a request by `user`: model requests to an
    /// on-behalf-of provider exchange their own token.
    pub(crate) fn acting(&self, user: &UserId) -> Acting {
        Acting::new(user.clone(), Arc::clone(&self.subjects))
    }

    /// Begin a sign-in: the issuer URL to send the browser to, and the
    /// `state` its callback will carry.
    pub(crate) async fn start(&self) -> ApiResult<(String, String)> {
        let (url, pending) = self.sign_in.begin().await?;
        let state = pending.state.clone();
        let mut in_flight = self
            .pending
            .lock()
            .map_err(|e| ApiError::internal(format!("sign-in store poisoned: {e}")))?;
        in_flight.retain(|_, (_, began)| began.elapsed() < PENDING_TTL);
        if in_flight.len() >= MAX_PENDING {
            return Err(ApiError::busy(
                "too many sign-ins in progress; try again shortly",
                5,
            ));
        }
        in_flight.insert(state.clone(), (pending, Instant::now()));
        Ok((url, state))
    }

    /// The sign-in `state` names, once: a replayed callback finds nothing.
    pub(crate) fn take(&self, state: &str) -> Option<Pending> {
        let (pending, began) = self.pending.lock().ok()?.remove(state)?;
        (began.elapsed() < PENDING_TTL).then_some(pending)
    }

    /// Check a callback's `iss` against the issuer (RFC 9207).
    pub(crate) async fn check_response_issuer(&self, iss: Option<&str>) -> CoreResult<()> {
        self.sign_in.check_response_issuer(iss).await
    }

    /// Complete a sign-in with the callback's code.
    pub(crate) async fn finish(&self, code: &str, pending: Pending) -> CoreResult<SignedIn> {
        self.sign_in.finish(code, pending).await
    }

    /// Keep a signed-in user's token, replacing the one from an earlier
    /// sign-in, and return when the session opened on it must renew it.
    pub(crate) async fn keep(
        &self,
        user: &UserId,
        token: &CachedToken,
    ) -> ApiResult<Option<Timestamp>> {
        self.subjects.keep(user, token).await?;
        Ok(Self::renewal_time(token))
    }

    /// When a session on `token` renews it: shortly before it expires, or
    /// never when there is no refresh token to renew it with.
    fn renewal_time(token: &CachedToken) -> Option<Timestamp> {
        token.refresh_token.as_ref().map(|_| {
            token
                .expires_at
                .checked_sub(RENEW_MARGIN)
                .unwrap_or(token.expires_at)
        })
    }

    /// Renew the sign-in behind `session`, a session of `user` whose renewal
    /// time has come. A token another session already renewed is reused; an
    /// issuer that cannot be reached is asked again a minute later, and the
    /// session goes on meanwhile; a refusal ends every session of the user.
    pub(crate) async fn renew(
        &self,
        sessions: &WebSessions,
        user: &UserId,
        session: &str,
    ) -> ApiResult<Standing> {
        Ok(match self.subjects.refreshed(user).await? {
            Stored::Current(token) => {
                sessions.renew_at(session, Self::renewal_time(&token));
                Standing::Current
            }
            Stored::Unrenewable(_) => {
                sessions.renew_at(session, None);
                Standing::Current
            }
            Stored::Unreachable(_) => {
                sessions.renew_at(session, Timestamp::now().checked_add(RETRY_AFTER).ok());
                Standing::Current
            }
            Stored::Missing | Stored::Revoked => {
                tracing::info!(user = %user, "no current stored sign-in for the session; ending every session of the user");
                sessions.close_user(user);
                Standing::Ended
            }
        })
    }

    /// Renew the sign-in behind a session that is due, and refuse the
    /// request when the issuer has ended it, recording `denied` (the
    /// caller's address and request id already on it) against the user.
    pub(crate) async fn require_current(
        &self,
        app: &AppState,
        user: &UserId,
        session: &str,
        mut denied: AuditEntry,
    ) -> ApiResult<()> {
        match self.renew(&app.sessions, user, session).await? {
            Standing::Current => Ok(()),
            Standing::Ended => {
                denied.user_id = Some(user.clone());
                app.control.record_audit(&denied).await?;
                Err(ApiError::unauthorized(
                    "the identity provider ended this sign-in; sign in again",
                ))
            }
        }
    }

    /// Whether access tokens from the issuer are accepted as bearers.
    pub(crate) fn accepts_bearers(&self) -> bool {
        self.sign_in.accepts_bearers()
    }

    /// The user an access token presented as a bearer names, created with no
    /// access on first sight, as a sign-in would. The token is remembered
    /// for exchanges made for them while they have no stored sign-in.
    pub(crate) async fn bearer_user(
        &self,
        control: &ControlPlane,
        token: &str,
    ) -> CoreResult<UserRow> {
        let bearer = self.sign_in.verify_bearer(token).await?;
        let user = control.oidc_user(&bearer.subject, &bearer.username).await?;
        self.subjects
            .remember_presented(&user.id, token, bearer.expires_at);
        Ok(user)
    }

    /// Drop a user's stored token, once they have no session left to use it.
    pub(crate) async fn forget(&self, user: &UserId) -> ApiResult<()> {
        self.subjects.forget(user).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
