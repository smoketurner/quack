//! Each signed-in user's own access token for quack: the `subject_token` an
//! on-behalf-of exchange trades (RFC 8693). It is their stored sign-in,
//! renewed when it is due, or else the access token they last presented as
//! a bearer. Session renewal reads the stored sign-in here too, under the
//! same lock per user, so two renewals never spend one refresh token.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use jiff::{SignedDuration, Timestamp};
use secrecy::SecretString;

use super::{Renewal, SignIn, UserTokens};
use crate::error::Result;
use crate::ids::UserId;
use crate::llm::oauth::CachedToken;
use crate::storage::control::ControlPlane;

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

/// Every signed-in user's own token, for `quack serve`.
pub struct SubjectTokens {
    sign_in: Arc<SignIn>,
    tokens: UserTokens,
    control: ControlPlane,
    /// One renewal per person at a time.
    renewing: Mutex<HashMap<UserId, Arc<tokio::sync::Mutex<()>>>>,
    /// The access token each person last presented as a bearer.
    presented: Mutex<HashMap<UserId, CachedToken>>,
}

impl std::fmt::Debug for SubjectTokens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubjectTokens").finish_non_exhaustive()
    }
}

impl SubjectTokens {
    #[must_use]
    pub fn new(sign_in: Arc<SignIn>, tokens: UserTokens, control: ControlPlane) -> Self {
        Self {
            sign_in,
            tokens,
            control,
            renewing: Mutex::new(HashMap::new()),
            presented: Mutex::new(HashMap::new()),
        }
    }

    fn lock_for(&self, user: &UserId) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.renewing.lock().unwrap_or_else(PoisonError::into_inner);
        Arc::clone(locks.entry(user.clone()).or_default())
    }

    /// Keep a person's token from a sign-in, replacing the one before.
    ///
    /// # Errors
    ///
    /// Returns an error when sealing or the write fails.
    pub async fn keep(&self, user: &UserId, token: &CachedToken) -> Result<()> {
        let lock = self.lock_for(user);
        let _renewing = lock.lock().await;
        self.tokens.store(&self.control, user, token).await
    }

    /// Forget a person's stored token.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails.
    pub async fn forget(&self, user: &UserId) -> Result<()> {
        let lock = self.lock_for(user);
        let _renewing = lock.lock().await;
        self.tokens.clear(&self.control, user).await
    }

    /// The person's stored sign-in, renewed first when it is due.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored token cannot be read or written; an
    /// issuer that refuses or cannot be reached is a [`Stored`] outcome.
    pub async fn refreshed(&self, user: &UserId) -> Result<Stored> {
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
    pub async fn subject_token(&self, user: &UserId) -> std::result::Result<SecretString, String> {
        let now = Timestamp::now();
        let current = |token: &CachedToken| token.is_fresh(now, RENEW_MARGIN);
        match self.refreshed(user).await.map_err(|e| e.to_string())? {
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
