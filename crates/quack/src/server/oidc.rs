//! Sign-in through the organization's `OpenID` Connect issuer: the sign-ins
//! in flight, and each signed-in user's token.
//!
//! A user's token is kept in `control.db`, sealed with HPKE under the
//! server's key (`oidc::UserTokens` over `vault::Vault`). Its refresh token is what ties a quack
//! session to the issuer: when the token runs out, the first request that
//! finds it renews it, and a refusal ends every session the user has.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jiff::{SignedDuration, Timestamp};
use quack_core::config::OidcConfig;
use quack_core::error::Result as CoreResult;
use quack_core::ids::UserId;
use quack_core::llm::oauth::CachedToken;
use quack_core::oidc::{Pending, Renewal, SignIn, SignedIn, UserTokens};
use quack_core::storage::control::{AuditEntry, ControlPlane};
use quack_core::vault::Vault;

use super::error::{ApiError, ApiResult};
use super::state::AppState;

/// The cookie that ties a callback to the browser that started the sign-in,
/// so a callback link someone else started cannot sign this browser in.
pub(crate) const STATE_COOKIE: &str = "quack_oidc_state";

/// How long a sign-in may take at the issuer.
pub(crate) const PENDING_TTL: Duration = Duration::from_secs(600);

/// Sign-ins in flight at once. Each is a few hundred bytes, and starting one
/// costs nothing but a request, so the count is bounded.
const MAX_PENDING: usize = 10_000;

/// A token is renewed this long before it expires.
const RENEW_MARGIN: SignedDuration = SignedDuration::from_secs(60);

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
    sign_in: SignIn,
    /// By `state`: the sign-in and when it began.
    pending: Mutex<HashMap<String, (Pending, Instant)>>,
    /// One renewal per user at a time, so two requests that both find the
    /// token expired do not both spend the refresh token.
    renewing: Mutex<HashMap<UserId, Arc<tokio::sync::Mutex<()>>>>,
    tokens: UserTokens,
}

impl Oidc {
    /// # Errors
    ///
    /// Returns an error when the HTTP client cannot be built.
    pub(crate) fn new(config: &OidcConfig, vault: Vault) -> CoreResult<Self> {
        Ok(Self {
            sign_in: SignIn::new(config.clone())?,
            pending: Mutex::new(HashMap::new()),
            renewing: Mutex::new(HashMap::new()),
            tokens: UserTokens::new(vault),
        })
    }

    pub(crate) fn issuer_host(&self) -> String {
        self.sign_in.issuer_host()
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

    /// Complete a sign-in with the callback's code.
    pub(crate) async fn finish(&self, code: &str, pending: Pending) -> CoreResult<SignedIn> {
        self.sign_in.finish(code, pending).await
    }

    fn lock_for(&self, user: &UserId) -> ApiResult<Arc<tokio::sync::Mutex<()>>> {
        let mut locks = self
            .renewing
            .lock()
            .map_err(|e| ApiError::internal(format!("renewal locks poisoned: {e}")))?;
        Ok(Arc::clone(locks.entry(user.clone()).or_default()))
    }

    /// Keep a signed-in user's token, replacing the one from an earlier
    /// sign-in, and return when the session opened on it must renew it.
    pub(crate) async fn keep(
        &self,
        control: &ControlPlane,
        user: &UserId,
        token: &CachedToken,
    ) -> ApiResult<Option<Timestamp>> {
        let lock = self.lock_for(user)?;
        let _renewing = lock.lock().await;
        self.tokens.store(control, user, token).await?;
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
    /// session goes on meanwhile.
    pub(crate) async fn renew(
        &self,
        app: &AppState,
        user: &UserId,
        session: &str,
    ) -> ApiResult<Standing> {
        let (sessions, control) = (&app.sessions, &app.control);
        let lock = self.lock_for(user)?;
        let _renewing = lock.lock().await;
        let Some(token) = self.tokens.load(control, user).await? else {
            tracing::info!(user = %user, "no stored sign-in for the session; ending it");
            sessions.close_user(user);
            return Ok(Standing::Ended);
        };
        let now = Timestamp::now();
        if token.is_fresh(now, RENEW_MARGIN) {
            sessions.renew_at(session, Self::renewal_time(&token));
            return Ok(Standing::Current);
        }
        let Some(refresh) = &token.refresh_token else {
            sessions.renew_at(session, None);
            return Ok(Standing::Current);
        };
        match self.sign_in.renew(refresh).await {
            Ok(Renewal::Renewed(renewed)) => {
                self.tokens.store(control, user, &renewed).await?;
                sessions.renew_at(session, Self::renewal_time(&renewed));
                Ok(Standing::Current)
            }
            Ok(Renewal::Revoked(reason)) => {
                tracing::info!(user = %user, %reason, "the issuer ended the sign-in");
                self.tokens.clear(control, user).await?;
                sessions.close_user(user);
                Ok(Standing::Ended)
            }
            Err(e) => {
                tracing::warn!(user = %user, error = %e, "could not renew the sign-in; trying again shortly");
                sessions.renew_at(session, now.checked_add(RETRY_AFTER).ok());
                Ok(Standing::Current)
            }
        }
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
        match self.renew(app, user, session).await? {
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

    /// Drop a user's stored token, once they have no session left to use it.
    pub(crate) async fn forget(&self, control: &ControlPlane, user: &UserId) -> ApiResult<()> {
        let lock = self.lock_for(user)?;
        let _renewing = lock.lock().await;
        self.tokens.clear(control, user).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
