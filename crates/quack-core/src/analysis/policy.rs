//! What the agent may do to a workspace.
//!
//! Reads are always allowed. Writes (anything the `DuckDB` parser does not
//! recognize as a `SELECT`-shaped statement) are decided by the policy the
//! interface supplies; a refusal is recorded so the caller can report it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// How mutating statements from the agent are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WritePolicy {
    /// Execute writes without asking (`--allow-write`).
    Allow,
    /// Refuse every write; the tool result tells the model what to say.
    Deny,
    /// Emit a permission event and wait for the interface's answer.
    Ask,
}

/// Shared flag set whenever a write was refused during a turn, so the
/// interface can surface it (print mode exits 3).
#[derive(Debug, Clone, Default)]
pub struct RefusalFlag(Arc<AtomicBool>);

impl RefusalFlag {
    pub fn set(&self) {
        self.0.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn was_refused(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusal_flag_is_shared_across_clones() {
        let flag = RefusalFlag::default();
        let other = flag.clone();
        assert!(!flag.was_refused());
        other.set();
        assert!(flag.was_refused());
    }
}
