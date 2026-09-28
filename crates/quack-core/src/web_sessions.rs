//! `quack serve`'s browser and API-login sessions. They live in core so an
//! issuer's refusal seen by an on-behalf-of exchange can end them where it
//! is seen, under the same per-user lock as a session renewal.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use jiff::Timestamp;

use crate::config::ServerConfig;
use crate::error::Result;
use crate::ids::UserId;
use crate::storage::control::random_bytes;

/// A live browser session. Both bounds are measured with [`Instant`], so a
/// clock the operator moves cannot extend or shorten one.
struct WebSession {
    user_id: UserId,
    /// When the session was opened, against `session_max_age`.
    started: Instant,
    /// The last request that presented it, against `session_idle`.
    last_seen: Instant,
    /// When the identity provider's token behind a sign-in must be renewed,
    /// which is when the issuer is next asked whether the person may still
    /// be signed in. `None` for a password login, or a sign-in the issuer
    /// gave no refresh token for. Wall-clock, since the issuer's expiry is.
    renew_at: Option<Timestamp>,
}

/// What a presented session token resolved to.
pub enum SessionLookup {
    /// A live session, belonging to this user id; `renewal_due` when its
    /// sign-in must be renewed before the request goes on.
    Active { user_id: UserId, renewal_due: bool },
    /// The token named a session that had outlived one of its bounds. It is
    /// gone now; the caller must log in again.
    Expired,
    /// No session by that name — it may still be an API token.
    Unknown,
}

/// A login session's token: `qs_` and 32 random bytes, base64url. It
/// grants the user's access, so `Debug` shows only its prefix.
#[derive(Clone, PartialEq, Eq)]
pub struct SessionToken(String);

impl std::fmt::Debug for SessionToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SessionToken").field(&"qs_…").finish()
    }
}

impl SessionToken {
    /// A fresh token from the control plane's random source (aws-lc-rs).
    fn generate() -> Result<Self> {
        let mut bytes = [0u8; 32];
        random_bytes(&mut bytes)?;
        Ok(Self(format!(
            "qs_{}",
            base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, bytes)
        )))
    }

    /// A token a request presented, to be looked up.
    #[must_use]
    pub const fn presented(token: String) -> Self {
        Self(token)
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

/// Browser and API-login sessions, by token. Cleared on restart, and one
/// by one once either `[server]` lifetime runs out.
pub struct WebSessions {
    /// `session_max_age`: from opening.
    max_age: Duration,
    /// `session_idle`: since the last request that presented it.
    idle: Duration,
    live: Mutex<HashMap<String, WebSession>>,
}

impl std::fmt::Debug for WebSessions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebSessions").finish_non_exhaustive()
    }
}

impl WebSessions {
    #[must_use]
    pub fn new(server: &ServerConfig) -> Self {
        Self {
            max_age: server.session_max_age(),
            idle: server.session_idle(),
            live: Mutex::new(HashMap::new()),
        }
    }

    fn live(&self) -> std::sync::MutexGuard<'_, HashMap<String, WebSession>> {
        self.live.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Start a session for the user and return its token; `renew_at` is when
    /// a sign-in's token must first be renewed.
    ///
    /// # Errors
    ///
    /// Returns an error when the random source fails.
    pub fn open(&self, user_id: &UserId, renew_at: Option<Timestamp>) -> Result<SessionToken> {
        let token = SessionToken::generate()?;
        let now = Instant::now();
        let mut live = self.live();
        // A login is the natural moment to drop whatever died since the last
        // one: nothing else walks the map, and a session nobody presents
        // again would otherwise sit here until the process ends.
        live.retain(|_, session| !self.expired(session, now));
        live.insert(
            token.as_str().to_owned(),
            WebSession {
                user_id: user_id.clone(),
                started: now,
                last_seen: now,
                renew_at,
            },
        );
        Ok(token)
    }

    /// Whether `session` has outlived either bound as of `now`.
    fn expired(&self, session: &WebSession, now: Instant) -> bool {
        now.duration_since(session.started) >= self.max_age
            || now.duration_since(session.last_seen) >= self.idle
    }

    /// Resolve a presented token, dropping it if it has expired and marking
    /// it used if it has not.
    pub fn lookup(&self, token: &str) -> SessionLookup {
        let mut live = self.live();
        let now = Instant::now();
        // Read the bounds first and let that borrow end, so the expired
        // branch is free to take the mutable one `remove` needs.
        let expired = match live.get(token) {
            Some(session) => self.expired(session, now),
            None => return SessionLookup::Unknown,
        };
        if expired {
            live.remove(token);
            return SessionLookup::Expired;
        }
        let Some(session) = live.get_mut(token) else {
            return SessionLookup::Unknown;
        };
        session.last_seen = now;
        SessionLookup::Active {
            user_id: session.user_id.clone(),
            renewal_due: session.renew_at.is_some_and(|at| at <= Timestamp::now()),
        }
    }

    /// Set when a session's sign-in is next renewed.
    pub fn renew_at(&self, token: &str, at: Option<Timestamp>) {
        if let Some(session) = self.live().get_mut(token) {
            session.renew_at = at;
        }
    }

    /// Whether the user still has a session that has not expired.
    pub fn has_sessions(&self, user_id: &UserId) -> bool {
        let now = Instant::now();
        self.live()
            .values()
            .any(|session| &session.user_id == user_id && !self.expired(session, now))
    }

    /// End every session of a user, once the issuer no longer vouches for
    /// them.
    pub fn close_user(&self, user_id: &UserId) {
        self.live().retain(|_, session| &session.user_id != user_id);
    }

    /// End every session of a user if `token` is still one of them, and say
    /// whether it was: a session another path already ended is not ended,
    /// or reported, twice.
    pub fn close_user_of(&self, token: &str, user_id: &UserId) -> bool {
        let mut live = self.live();
        if !live.contains_key(token) {
            return false;
        }
        live.retain(|_, session| &session.user_id != user_id);
        true
    }

    /// End a session; an unknown token is already ended.
    pub fn close(&self, token: &str) {
        self.live().remove(token);
    }
}
