//! A value the current work carries in a Tokio task-local, set at most
//! once: whom it acts for ([`super::acting::Acting`]) and where it may send
//! ([`super::egress::Egress`]). Each keeps a key of its own and shares these
//! four operations.

use std::future::Future;
use std::sync::{Arc, OnceLock};

use tokio::task::LocalKey;

/// The current work's value, set at most once.
#[derive(Clone)]
pub(crate) struct Slot<T>(Arc<OnceLock<T>>);

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self(Arc::default())
    }
}

impl<T: Clone + 'static> Slot<T> {
    /// Run `work` with an empty slot that [`Self::enter`] fills once the
    /// value is known.
    pub(crate) async fn request<F: Future>(key: &'static LocalKey<Self>, work: F) -> F::Output {
        key.scope(Self::default(), work).await
    }

    /// Run `work` holding `value`, or nothing.
    pub(crate) async fn scope<F: Future>(
        key: &'static LocalKey<Self>,
        value: Option<T>,
        work: F,
    ) -> F::Output {
        let slot = Self::default();
        if let Some(value) = value {
            drop(slot.0.set(value));
        }
        key.scope(slot, work).await
    }

    /// Fill the current request's slot. The first caller wins; outside
    /// [`Self::request`] it does nothing.
    pub(crate) fn enter(key: &'static LocalKey<Self>, value: T) {
        drop(key.try_with(|slot| slot.0.set(value)));
    }

    /// What the current work holds, if anything.
    pub(crate) fn current(key: &'static LocalKey<Self>) -> Option<T> {
        key.try_with(|slot| slot.0.get().cloned()).ok().flatten()
    }
}
