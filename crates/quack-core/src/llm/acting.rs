//! The person a model request is made for, when a provider acts on their
//! behalf (`grant = "on-behalf-of"`).
//!
//! Carried by a Tokio task-local, like [`crate::priority::Priority`], so the
//! token manager finds it without an argument threaded through every layer.
//! `quack serve` scopes an empty slot around each request and fills it once
//! it knows the caller; the job queue carries the submitter's into the job;
//! nothing else sets one, so the CLI and the terminal act for nobody.

use std::future::Future;
use std::sync::Arc;

use secrecy::SecretString;

use super::slot::Slot;
use crate::ids::UserId;
use crate::oidc::{Revocations, SubjectTokens};
use crate::storage::control::Origin;

/// The person model requests are made for.
#[derive(Clone)]
pub struct Acting {
    user: UserId,
    tokens: Subject,
}

/// Where the person's own token comes from.
#[derive(Clone)]
enum Subject {
    /// `quack serve`'s signed-in users, and where the request came from.
    Signed(Arc<SubjectTokens>, Origin),
    /// A token fixed by a test, or the reason there is none.
    #[cfg(test)]
    Fixed(Result<&'static str, &'static str>),
}

impl std::fmt::Debug for Acting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Acting")
            .field("user", &self.user)
            .finish_non_exhaustive()
    }
}

tokio::task_local! {
    static SLOT: Slot<Acting>;
}

impl Acting {
    /// `user`, whose own token `tokens` keeps, acting for a request from
    /// `origin`.
    #[must_use]
    pub const fn new(user: UserId, tokens: Arc<SubjectTokens>, origin: Origin) -> Self {
        Self {
            user,
            tokens: Subject::Signed(tokens, origin),
        }
    }

    /// `user` with a fixed token (`Ok`) or none (`Err`, the reason).
    #[cfg(test)]
    pub(crate) const fn fixed(user: UserId, token: Result<&'static str, &'static str>) -> Self {
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
    pub async fn subject_token(&self) -> Result<SecretString, String> {
        match &self.tokens {
            Subject::Signed(tokens, origin) => tokens.subject_token(&self.user, origin).await,
            #[cfg(test)]
            Subject::Fixed(token) => token
                .map(|t| SecretString::from(t.to_owned()))
                .map_err(str::to_owned),
        }
    }

    /// How many times the issuer has ended the person's sign-in.
    #[must_use]
    pub fn revocations(&self) -> Revocations {
        match &self.tokens {
            Subject::Signed(tokens, _) => tokens.revocations(&self.user),
            #[cfg(test)]
            Subject::Fixed(_) => Revocations::default(),
        }
    }

    /// Run `work` with an empty slot that [`Acting::enter`] fills once the
    /// caller is known: a server request.
    pub fn request<F: Future>(work: F) -> impl Future<Output = F::Output> {
        Slot::request(&SLOT, work)
    }

    /// Run `work` acting for `acting`, or for nobody.
    pub fn scope<F: Future>(acting: Option<Self>, work: F) -> impl Future<Output = F::Output> {
        Slot::scope(&SLOT, acting, work)
    }

    /// Make this person the one the current request acts for. The first
    /// caller wins; outside [`Acting::request`] it does nothing.
    pub fn enter(self) {
        Slot::enter(&SLOT, self);
    }

    /// Who the current work acts for, if anyone.
    #[must_use]
    pub fn current() -> Option<Self> {
        Slot::current(&SLOT)
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
