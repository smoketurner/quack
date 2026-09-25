//! The person a model request is made for, when a provider acts on their
//! behalf (`grant = "on-behalf-of"`).
//!
//! Carried by a Tokio task-local, like [`crate::priority::Priority`], so the
//! token manager finds it without an argument threaded through every layer.
//! `quack serve` scopes an empty slot around each request and fills it once
//! it knows the caller; the job queue carries the submitter's into the job;
//! nothing else sets one, so the CLI and the terminal act for nobody.

use std::future::Future;
use std::sync::{Arc, OnceLock};

use secrecy::SecretString;

use crate::ids::UserId;
use crate::oidc::PersonTokens;

/// The person model requests are made for.
#[derive(Clone)]
pub struct Acting {
    user: UserId,
    tokens: Subject,
}

/// Where the person's own token comes from.
#[derive(Clone)]
enum Subject {
    /// `quack serve`'s signed-in people.
    People(Arc<PersonTokens>),
    /// A token fixed by a test, or the reason there is none.
    #[cfg(test)]
    Fixed(std::result::Result<&'static str, &'static str>),
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
    /// `user`, whose own token `tokens` keeps.
    #[must_use]
    pub const fn new(user: UserId, tokens: Arc<PersonTokens>) -> Self {
        Self {
            user,
            tokens: Subject::People(tokens),
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
    /// Returns why, when there is no current token for them.
    pub async fn subject_token(&self) -> std::result::Result<SecretString, String> {
        match &self.tokens {
            Subject::People(people) => people.subject_token(&self.user).await,
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
