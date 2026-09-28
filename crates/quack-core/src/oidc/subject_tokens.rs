//! Each signed-in user's own access token for quack: the `subject_token` an
//! on-behalf-of exchange trades (RFC 8693). It is their stored sign-in,
//! renewed when it is due, or else the access token they last presented as
//! a bearer. Session renewal reads the stored sign-in here too, under the
//! same lock per user, so two renewals never spend one refresh token, and an
//! issuer's refusal is handled once, wherever it is seen first.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use jiff::{SignedDuration, Timestamp};
use secrecy::SecretString;

use super::{Renewal, SignIn, UserTokens};
use crate::error::Result;
use crate::ids::UserId;
use crate::llm::oauth::CachedToken;
use crate::storage::control::{AuditAction, AuditEntry, Channel, ControlPlane, Outcome};
use crate::web_sessions::WebSessions;

/// A token is renewed this long before it expires.
pub const RENEW_MARGIN: SignedDuration = SignedDuration::from_secs(60);

/// A person's stored sign-in, after renewing it when it was due.
#[derive(Debug)]
pub enum Stored {
    /// Fresh, or renewed just now.
    Current(CachedToken),
    /// No refresh token to renew it with: as stored.
    Unrenewable(CachedToken),
    /// The issuer could not be reached: as stored.
    Unreachable(CachedToken),
    /// Nothing stored.
    Missing,
    /// The issuer refused the renewal, and the stored token is gone.
    Revoked,
}

/// Where the request that meets an issuer's refusal came from, for the
/// denied `session` row it records.
#[derive(Debug, Clone)]
pub struct Origin {
    pub channel: Channel,
    pub client_addr: Option<String>,
    pub request_id: Option<String>,
}

impl Origin {
    /// The denied `session` row for `user`, when their sign-in has ended.
    #[must_use]
    pub fn denied_session(&self, user: &UserId) -> AuditEntry {
        let mut entry = AuditEntry::new(AuditAction::Session, Outcome::Denied, self.channel);
        entry.user_id = Some(user.clone());
        entry.client_addr.clone_from(&self.client_addr);
        entry.request_id.clone_from(&self.request_id);
        entry
    }
}

/// How many times the issuer has ended a person's sign-in: a provider token
/// exchanged under an earlier count is not reused.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Revocations(u64);

/// Every signed-in user's own token, for `quack serve`.
pub struct SubjectTokens {
    sign_in: Arc<SignIn>,
    tokens: UserTokens,
    control: ControlPlane,
    sessions: Arc<WebSessions>,
    /// One renewal per person at a time.
    renewing: Mutex<HashMap<UserId, Arc<tokio::sync::Mutex<()>>>>,
    /// The access token each person last presented as a bearer.
    presented: Mutex<HashMap<UserId, CachedToken>>,
    revocations: Mutex<HashMap<UserId, Revocations>>,
}

impl std::fmt::Debug for SubjectTokens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubjectTokens").finish_non_exhaustive()
    }
}

impl SubjectTokens {
    #[must_use]
    pub fn new(
        sign_in: Arc<SignIn>,
        tokens: UserTokens,
        control: ControlPlane,
        sessions: Arc<WebSessions>,
    ) -> Self {
        Self {
            sign_in,
            tokens,
            control,
            sessions,
            renewing: Mutex::new(HashMap::new()),
            presented: Mutex::new(HashMap::new()),
            revocations: Mutex::new(HashMap::new()),
        }
    }

    /// How many times the issuer has ended this person's sign-in.
    #[must_use]
    pub fn revocations(&self, user: &UserId) -> Revocations {
        self.revocations
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(user)
            .copied()
            .unwrap_or_default()
    }

    fn lock_for(&self, user: &UserId) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.renewing.lock().unwrap_or_else(PoisonError::into_inner);
        Arc::clone(locks.entry(user.clone()).or_default())
    }

    /// Keep a person's token from a sign-in, replacing the one before, and
    /// run `then` (opening the session that uses it) under the same per-user
    /// lock. A [`Self::forget_unless`] therefore sees neither or both: it
    /// never deletes a token whose session is about to open.
    ///
    /// # Errors
    ///
    /// Returns an error when sealing or the write fails; `then` does not run.
    pub async fn keep_then<T>(
        &self,
        user: &UserId,
        token: &CachedToken,
        then: impl FnOnce() -> T,
    ) -> Result<T> {
        let lock = self.lock_for(user);
        let _renewing = lock.lock().await;
        self.tokens.store(&self.control, user, token).await?;
        Ok(then())
    }

    /// Forget a person's stored token unless `in_use` (asked under the
    /// per-user lock, after any sign-in in progress has stored its token and
    /// opened its session) says something still needs it. Returns whether
    /// it was forgotten.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails.
    pub async fn forget_unless(
        &self,
        user: &UserId,
        in_use: impl FnOnce() -> bool,
    ) -> Result<bool> {
        let lock = self.lock_for(user);
        let _renewing = lock.lock().await;
        if in_use() {
            return Ok(false);
        }
        self.tokens.clear(&self.control, user).await?;
        Ok(true)
    }

    /// The person's stored sign-in, renewed first when it is due. When the
    /// issuer refuses, everything that carried the sign-in goes with it
    /// before the lock is released: the stored and presented tokens, every
    /// session, and exchanged provider tokens; and the refusal is audited as
    /// a denied `session` from `origin`.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored token cannot be read or written, or
    /// the refusal cannot be audited; an issuer that refuses or cannot be
    /// reached is a [`Stored`] outcome.
    pub async fn refreshed(&self, user: &UserId, origin: &Origin) -> Result<Stored> {
        let lock = self.lock_for(user);
        let _renewing = lock.lock().await;
        let Some(token) = self.tokens.load(&self.control, user).await? else {
            return Ok(Stored::Missing);
        };
        if token.is_fresh(Timestamp::now(), RENEW_MARGIN) {
            return Ok(Stored::Current(token));
        }
        let Some(refresh) = &token.refresh_token else {
            return Ok(Stored::Unrenewable(token));
        };
        match self.sign_in.renew(refresh).await {
            Ok(Renewal::Renewed(renewed)) => {
                self.tokens.store(&self.control, user, &renewed).await?;
                Ok(Stored::Current(renewed))
            }
            Ok(Renewal::Revoked(reason)) => {
                tracing::info!(user = %user, %reason, "the issuer ended the sign-in");
                self.tokens.clear(&self.control, user).await?;
                self.presented
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .remove(user);
                {
                    let mut revocations = self
                        .revocations
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner);
                    let Revocations(count) = revocations.entry(user.clone()).or_default();
                    *count = count.saturating_add(1);
                }
                self.sessions.close_user(user);
                self.control
                    .record_audit(&origin.denied_session(user))
                    .await?;
                Ok(Stored::Revoked)
            }
            Err(e) => {
                tracing::warn!(user = %user, error = %e, "could not renew the sign-in; trying again shortly");
                Ok(Stored::Unreachable(token))
            }
        }
    }

    /// Remember the access token a person presented as a bearer, until it
    /// expires, for exchanges made for them.
    pub fn remember_presented(&self, user: &UserId, token: &str, expires_at: Timestamp) {
        self.presented
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(
                user.clone(),
                CachedToken {
                    access_token: SecretString::from(token.to_owned()),
                    expires_at,
                    refresh_token: None,
                },
            );
    }

    /// The person's own access token for quack, for an on-behalf-of
    /// exchange: their stored sign-in (renewed when due), else the token they
    /// last presented, while either is current.
    ///
    /// # Errors
    ///
    /// Returns why, said to the person, when neither is current.
    pub async fn subject_token(
        &self,
        user: &UserId,
        origin: &Origin,
    ) -> std::result::Result<SecretString, String> {
        let now = Timestamp::now();
        let current = |token: &CachedToken| token.is_fresh(now, RENEW_MARGIN);
        match self
            .refreshed(user, origin)
            .await
            .map_err(|e| e.to_string())?
        {
            Stored::Current(token) | Stored::Unrenewable(token) | Stored::Unreachable(token)
                if current(&token) =>
            {
                return Ok(token.access_token);
            }
            Stored::Revoked => {
                return Err(format!(
                    "{} ended this person's sign-in; they sign in again",
                    self.sign_in.issuer_host()
                ));
            }
            Stored::Current(_)
            | Stored::Unrenewable(_)
            | Stored::Unreachable(_)
            | Stored::Missing => {}
        }
        self.presented
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(user)
            .filter(|t| current(t))
            .map(|t| t.access_token.clone())
            .ok_or_else(|| {
                format!(
                    "this person has no current sign-in through {}; sign in with it to use this provider",
                    self.sign_in.issuer_host()
                )
            })
    }
}
