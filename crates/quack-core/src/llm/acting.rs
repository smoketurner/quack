//! The person a model request is made for, when a provider acts on their
//! behalf (`grant = "on-behalf-of"`).
//!
//! Carried by a Tokio task-local, like [`crate::priority::Priority`], so the
//! token manager finds it without an argument threaded through every layer.
//! `quack serve` scopes an empty slot around each request and fills it once
//! it knows the caller; the job queue carries the submitter's into the job;
//! nothing else sets one, so the CLI and the terminal act for nobody.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};

use secrecy::SecretString;

use crate::ids::UserId;
use crate::oidc::{SubjectRefusal, SubjectTokens};

/// The person a model request is made for.
#[derive(Clone)]
pub struct Acting {
    user: UserId,
    tokens: Subject,
    /// What `subject_token` invokes the moment the issuer ends the person's
    /// sign-in through an on-behalf-of exchange, before it returns the
    /// refusal: the server ends the person's sessions and records a denied
    /// `session` audit row, the same bookkeeping the session-renewal refusal
    /// does in `Oidc::require_current`. `None` for an [`Acting`] the server
    /// did not build (a test fixed token), or in local mode.
    on_revoked: Option<Arc<dyn OnRevoked + Send + Sync>>,
}

/// Where the person's own token comes from.
#[derive(Clone)]
enum Subject {
    /// `quack serve`'s signed-in users.
    Signed(Arc<SubjectTokens>),
    /// A token fixed by a test, or the reason there is none.
    #[cfg(test)]
    Fixed(std::result::Result<&'static str, &'static str>),
}

/// The bookkeeping the server does the moment an on-behalf-of exchange finds
/// the issuer has ended the person's sign-in — end every session of the
/// person, and record a denied `session` audit row — exactly as the
/// session-renewal refusal does in `Oidc::require_current`.
///
/// `SubjectTokens` in `quack-core` cannot itself end sessions or write a
/// session-audit row: it owns neither `WebSessions` nor the caller's audit
/// context, and the on-behalf-of exchange runs through `Acting::subject_token`
/// deep in core, outside the server layer. The server therefore installs one
/// of these on each `Acting` it builds, so the refusal's bookkeeping happens
/// at the moment the issuer is refused, rather than at the next
/// session-authenticated request's convenience (which a churned person never
/// makes; see `docs/authentication.md`). `Acting::subject_token` awaits it
/// before it returns the refusal, so the rows are written before the person
/// sees the error.
pub trait OnRevoked: Send + Sync {
    /// End `user`'s sessions and record the denied `session` row, returning a
    /// future the caller awaits before it surfaces the refusal.
    fn on_revoked<'user>(
        &self,
        user: &'user UserId,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'user>>;
}

impl std::fmt::Debug for Acting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Acting")
            .field("user", &self.user)
            .finish_non_exhaustive()
    }
}

/// The current work's acting person, set at most once.
#[derive(Clone, Default)]
struct Slot(Arc<OnceLock<Acting>>);

tokio::task_local! {
    static SLOT: Slot;
}

impl Acting {
    /// `user`, whose own token `tokens` keeps, with the `on_revoked` hook the
    /// server installs to end the person's sessions and record the denied
    /// `session` audit row at the moment an on-behalf-of exchange finds the
    /// issuer has ended their sign-in (`None` for an `Acting` the server did
    /// not build — a test fixed token).
    #[must_use]
    pub fn new(
        user: UserId,
        tokens: Arc<SubjectTokens>,
        on_revoked: Option<Arc<dyn OnRevoked + Send + Sync>>,
    ) -> Self {
        Self {
            user,
            tokens: Subject::Signed(tokens),
            on_revoked,
        }
    }

    /// `user` with a fixed token (`Ok`) or none (`Err`, the reason).
    #[cfg(test)]
    pub(crate) const fn fixed(
        user: UserId,
        token: std::result::Result<&'static str, &'static str>,
    ) -> Self {
        Self {
            user,
            tokens: Subject::Fixed(token),
            on_revoked: None,
        }
    }

    #[must_use]
    pub const fn user(&self) -> &UserId {
        &self.user
    }

    /// The person's own token for quack.
    ///
    /// # Errors
    ///
    /// Returns why, when there is no current token for them. If the reason
    /// is that the issuer refused the renewal, the `on_revoked` hook (if
    /// any) has already ended the person's sessions and recorded the denied
    /// `session` audit row by the time this returns, so the caller surfaces
    /// the refusal after the bookkeeping for it is done.
    pub async fn subject_token(&self) -> std::result::Result<SecretString, String> {
        match &self.tokens {
            Subject::Signed(tokens) => match tokens.subject_token(&self.user).await {
                Ok(token) => Ok(token),
                Err(SubjectRefusal::Revoked(message)) => {
                    if let Some(on_revoked) = &self.on_revoked {
                        on_revoked.on_revoked(&self.user).await;
                    }
                    Err(message)
                }
                Err(SubjectRefusal::Other(message)) => Err(message),
            },
            #[cfg(test)]
            Subject::Fixed(token) => token
                .map(|t| SecretString::from(t.to_owned()))
                .map_err(str::to_owned),
        }
    }

    /// Run `work` with an empty slot that [`Acting::enter`] fills once the
    /// caller is known: a server request.
    pub async fn request<F: Future>(work: F) -> F::Output {
        SLOT.scope(Slot::default(), work).await
    }

    /// Run `work` acting for `acting`, or for nobody.
    pub async fn scope<F: Future>(acting: Option<Self>, work: F) -> F::Output {
        let slot = Slot::default();
        if let Some(acting) = acting {
            drop(slot.0.set(acting));
        }
        SLOT.scope(slot, work).await
    }

    /// Make this person the one the current request acts for. The first
    /// caller wins; outside [`Acting::request`] it does nothing.
    pub fn enter(self) {
        drop(SLOT.try_with(|slot| slot.0.set(self)));
    }

    /// Who the current work acts for, if anyone.
    #[must_use]
    pub fn current() -> Option<Self> {
        SLOT.try_with(|slot| slot.0.get().cloned()).ok().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acting(user: &str) -> Acting {
        Acting::fixed(UserId::from(user), Ok("t"))
    }

    #[tokio::test]
    async fn a_request_acts_for_whoever_it_enters_first_and_nothing_leaks_out() {
        assert!(Acting::current().is_none());
        let seen = Acting::request(async {
            assert!(Acting::current().is_none());
            acting("ada").enter();
            acting("bob").enter();
            Acting::current().map(|a| a.user().clone())
        })
        .await;
        assert_eq!(seen, Some(UserId::from("ada")));
        assert!(Acting::current().is_none());
        // Entering outside a request scope does nothing.
        acting("cy").enter();
        assert!(Acting::current().is_none());
    }

    #[tokio::test]
    async fn a_scope_carries_its_person_into_the_work() {
        let token = Acting::scope(Some(acting("ada")), async {
            match Acting::current() {
                Some(a) => a.subject_token().await.ok(),
                None => None,
            }
        })
        .await;
        assert!(token.is_some());
        assert!(
            Acting::scope(None, async { Acting::current() })
                .await
                .is_none()
        );
    }
}
