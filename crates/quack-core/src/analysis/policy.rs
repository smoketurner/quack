//! What the agent may do to a workspace.
//!
//! Reads are always allowed. Writes (anything the `DuckDB` parser does not
//! recognize as a `SELECT`-shaped statement) are decided by the policy the
//! interface supplies; the decision is recorded so the caller can report it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// How mutating statements from the agent are handled.
#[derive(Clone)]
pub enum WritePolicy {
    /// Execute writes without asking (`--allow-write`).
    Allow,
    /// Refuse every write; the tool result tells the model what to say.
    Deny,
    /// Ask the interface; `true` means run it. The callback receives the SQL.
    Ask(Arc<dyn Fn(&str) -> bool + Send + Sync>),
}

impl std::fmt::Debug for WritePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allow => f.write_str("Allow"),
            Self::Deny => f.write_str("Deny"),
            Self::Ask(_) => f.write_str("Ask(..)"),
        }
    }
}

/// Outcome of applying a [`WritePolicy`] to one statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Run,
    Refused,
}

impl WritePolicy {
    /// Decide whether a write statement may run.
    #[must_use]
    pub fn decide(&self, sql: &str) -> Decision {
        match self {
            Self::Allow => Decision::Run,
            Self::Deny => Decision::Refused,
            Self::Ask(ask) => {
                if ask(sql) {
                    Decision::Run
                } else {
                    Decision::Refused
                }
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
    fn allow_runs_and_deny_refuses() {
        assert_eq!(WritePolicy::Allow.decide("DROP TABLE t"), Decision::Run);
        assert_eq!(WritePolicy::Deny.decide("DROP TABLE t"), Decision::Refused);
    }

    #[test]
    fn ask_consults_the_callback_with_the_sql() {
        let seen = Arc::new(std::sync::Mutex::new(String::new()));
        let seen_in = Arc::clone(&seen);
        let policy = WritePolicy::Ask(Arc::new(move |sql: &str| {
            if let Ok(mut s) = seen_in.lock() {
                s.push_str(sql);
            }
            sql.contains("ok")
        }));
        assert_eq!(policy.decide("CREATE TABLE ok(x INT)"), Decision::Run);
        assert_eq!(policy.decide("DROP TABLE t"), Decision::Refused);
        assert!(seen.lock().is_ok_and(|s| s.contains("CREATE TABLE ok")));
    }

    #[test]
    fn refusal_flag_is_shared_across_clones() {
        let flag = RefusalFlag::default();
        let other = flag.clone();
        assert!(!flag.was_refused());
        other.set();
        assert!(flag.was_refused());
    }
}
