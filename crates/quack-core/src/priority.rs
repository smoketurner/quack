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

/// Run `work` at `priority` (everything it awaits on this task).
pub async fn with_priority<F: Future>(priority: Priority, work: F) -> F::Output {
    PRIORITY.scope(priority, work).await
}

/// The calling task's priority: interactive unless scoped otherwise.
#[must_use]
pub fn current_priority() -> Priority {
    PRIORITY.try_with(|p| *p).unwrap_or(Priority::Interactive)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_priority_follows_its_scope() {
        assert_eq!(current_priority(), Priority::Interactive);
        let inside = with_priority(Priority::Background, async {
            // Nested scopes win.
            let nested = with_priority(Priority::Interactive, async { current_priority() }).await;
            (current_priority(), nested)
        })
        .await;
        assert_eq!(inside, (Priority::Background, Priority::Interactive));
    }
}
