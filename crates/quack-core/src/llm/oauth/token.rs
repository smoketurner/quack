//! An OAuth access token with its lifetime and refresh token, and the form
//! it is serialized in before the vault seals it.

use jiff::Timestamp;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

/// A token as quack keeps it.
#[derive(Clone)]
pub struct CachedToken {
    pub access_token: SecretString,
    pub expires_at: Timestamp,
    pub refresh_token: Option<SecretString>,
}

impl CachedToken {
    /// Whether more than `margin` remains before expiry.
    #[must_use]
    pub fn is_fresh(&self, now: Timestamp, margin: jiff::SignedDuration) -> bool {
        self.expires_at.duration_since(now) > margin
    }
}

impl std::fmt::Debug for CachedToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedToken")
            .field("expires_at", &self.expires_at)
            .field("has_refresh_token", &self.refresh_token.is_some())
            .finish_non_exhaustive()
    }
}

/// A token as it is serialized before sealing.
#[derive(Serialize, Deserialize)]
pub(crate) struct Plaintext {
    access_token: String,
    expires_at: Timestamp,
    refresh_token: Option<String>,
}

impl From<Plaintext> for CachedToken {
    fn from(plain: Plaintext) -> Self {
        Self {
            access_token: SecretString::from(plain.access_token),
            expires_at: plain.expires_at,
            refresh_token: plain.refresh_token.map(SecretString::from),
        }
    }
}

impl From<&CachedToken> for Plaintext {
    fn from(token: &CachedToken) -> Self {
        Self {
            access_token: token.access_token.expose_secret().to_owned(),
            expires_at: token.expires_at,
            refresh_token: token
                .refresh_token
                .as_ref()
                .map(|t| t.expose_secret().to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freshness_uses_the_margin() {
        let t = CachedToken {
            access_token: SecretString::from(String::from("a")),
            expires_at: Timestamp::UNIX_EPOCH
                .checked_add(jiff::SignedDuration::from_hours(1))
                .unwrap_or(Timestamp::MAX),
            refresh_token: None,
        };
        let margin = jiff::SignedDuration::from_secs(60);
        assert!(t.is_fresh(Timestamp::UNIX_EPOCH, margin));
        let late = t
            .expires_at
            .checked_sub(jiff::SignedDuration::from_secs(30))
            .unwrap_or(Timestamp::MIN);
        assert!(!t.is_fresh(late, margin));
    }
}
