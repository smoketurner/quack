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

impl WritePolicy {
    /// The system prompt's permissions paragraph for this policy.
    #[must_use]
    pub const fn prompt_paragraph(self) -> &'static str {
        match self {
            Self::Allow => {
                "Permissions: SELECT queries always run. The user has permitted statements that \
                 modify the workspace for this session, so when asked to change data, run the \
                 statement with run_sql rather than asking for confirmation.\n"
            }
            Self::Ask => {
                "Permissions: SELECT queries always run. When you run a statement that modifies \
                 the workspace, the user is asked to approve it before it executes, so when asked \
                 to change data, run the statement with run_sql rather than asking for \
                 confirmation yourself. If the tool reports it was refused, do not retry it; tell \
                 the user.\n"
            }
            Self::Deny => {
                "Permissions: SELECT queries always run. Statements that modify the workspace are \
                 not permitted in this session; if the user asks for one, still attempt it once \
                 with run_sql so the refusal is recorded, then tell the user it needs write \
                 permission (--allow-write). Do not retry.\n"
            }
        }
    }
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
