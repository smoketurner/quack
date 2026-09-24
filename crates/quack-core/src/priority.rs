//! Who is waiting on a scarce resource (design doc 4.1): a person, or
//! background work. Carried by a Tokio task-local so every model request
//! and every workspace write a piece of work makes inherits it without
//! threading an argument through every layer. Anything not scoped is
//! interactive (a request handler, a terminal command, the CLI); the job
//! queue scopes background kinds (ingest, import, extraction, proposals,
//! exports) as background, and a bridge onto another thread re-scopes the
//! priority it was called with.

use std::future::Future;

/// Who is waiting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    /// A turn, a statement, or a command someone is watching; served first.
    Interactive,
    /// Ingest embeddings, extraction, proposals, imports.
    Background,
}

tokio::task_local! {
    static PRIORITY: Priority;
}

impl Priority {
    /// Run `work` at this priority (everything it awaits on this task).
    pub async fn scope<F: Future>(self, work: F) -> F::Output {
        PRIORITY.scope(self, work).await
    }

    /// The calling task's priority: interactive unless scoped otherwise.
    #[must_use]
    pub fn current() -> Self {
        PRIORITY.try_with(|p| *p).unwrap_or(Self::Interactive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_priority_follows_its_scope() {
        assert_eq!(Priority::current(), Priority::Interactive);
        let inside = Priority::Background
            .scope(async {
                // Nested scopes win.
                let nested = Priority::Interactive
                    .scope(async { Priority::current() })
                    .await;
                (Priority::current(), nested)
            })
            .await;
        assert_eq!(inside, (Priority::Background, Priority::Interactive));
    }
}
