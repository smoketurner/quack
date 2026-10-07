//! What the agent may do to a workspace.
//!
//! Reads are always allowed. Writes (anything the `DuckDB` parser does not
//! recognize as a `SELECT`-shaped statement) are decided by the policy the
//! interface supplies and by what the turn has read: once a turn has
//! retrieved document or graph text, no write runs without a person's
//! approval. A refusal is recorded so the caller can report it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// How mutating statements from the agent are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WritePolicy {
    /// Execute writes without asking (`--allow-write`), until the turn has
    /// read document text: from then on the approver decides each write.
    Allow(Approver),
    /// Refuse every write; the tool result tells the model what to say.
    Deny,
    /// Emit a permission event and wait for the interface's answer.
    Ask,
}

/// Who can approve a write while a turn runs under [`WritePolicy::Allow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Approver {
    /// The interface cannot ask anyone (print mode, a non-streamed request,
    /// MCP).
    Nobody,
    /// The interface answers permission events (the terminal, a streamed
    /// turn).
    Person,
}

/// What a turn has read that neither the person nor the workspace owner
/// wrote.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Exposure {
    #[default]
    None,
    /// Document or graph text a tool retrieved.
    Documents,
}

/// Why a write does not simply run. Ordered: a person who granted the rest
/// of a turn's writes under one reason has granted the reasons before it.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, utoipa::ToSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Hold {
    /// Writes were not permitted up front.
    NotPermitted,
    /// The turn read document text, which may have dictated the write.
    ReadDocuments,
}

text_enum!(Hold, "hold", {
    NotPermitted => "not_permitted",
    ReadDocuments => "read_documents",
});

impl Hold {
    /// What the model is told when the write did not run.
    #[must_use]
    pub const fn refusal(self) -> &'static str {
        match self {
            Self::NotPermitted => {
                "This statement would modify the workspace and was not permitted. Do not retry \
                 it. Tell the user it needs write permission (re-run with --allow-write)."
            }
            Self::ReadDocuments => {
                "This statement would modify the workspace and was not run. This turn has read \
                 document or graph text, so a write needs the user's own approval, and it was \
                 not given. Do not retry it. Tell the user which statement was not run and why."
            }
        }
    }

    /// The refused step's one-line result.
    #[must_use]
    pub const fn summary(self) -> &'static str {
        match self {
            Self::NotPermitted => "refused",
            Self::ReadDocuments => "refused: this turn read document text",
        }
    }

    /// What a person asked to approve the write is told beyond the
    /// statement, when there is more to say than "this writes".
    #[must_use]
    pub const fn notice(self) -> Option<&'static str> {
        match self {
            Self::NotPermitted => None,
            Self::ReadDocuments => Some(
                "This turn read document text, which may have asked for this statement. Check \
                 that it is what you asked for.",
            ),
        }
    }
}

/// What the policy decides for one write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteDecision {
    Run,
    /// Ask the interface, giving the reason.
    Ask(Hold),
    Refuse(Hold),
}

impl WritePolicy {
    /// `Allow` when writes were permitted up front (`--allow-write`,
    /// `allow_write` in a request, a token with the write scope), else this
    /// policy: `Deny` where nobody can be asked, `Ask` where someone can.
    /// The `Allow` remembers which, for a turn that reads document text.
    #[must_use]
    pub const fn allowed_if(self, allowed: bool) -> Self {
        match (allowed, self) {
            (false, _) | (true, Self::Allow(_)) => self,
            (true, Self::Deny) => Self::Allow(Approver::Nobody),
            (true, Self::Ask) => Self::Allow(Approver::Person),
        }
    }

    /// Whether a write the person makes directly (no agent turn) runs
    /// without asking.
    #[must_use]
    pub const fn allows_unasked(self) -> bool {
        match self {
            Self::Allow(_) => true,
            Self::Deny | Self::Ask => false,
        }
    }

    /// What happens to a write in a turn that has read `exposure`. Once a
    /// turn has read document text, no write runs without a person's
    /// approval, whatever was permitted up front.
    #[must_use]
    pub const fn decide(self, exposure: Exposure) -> WriteDecision {
        match (self, exposure) {
            (Self::Deny, Exposure::None | Exposure::Documents) => {
                WriteDecision::Refuse(Hold::NotPermitted)
            }
            (Self::Ask, Exposure::None) => WriteDecision::Ask(Hold::NotPermitted),
            (Self::Allow(_), Exposure::None) => WriteDecision::Run,
            (Self::Ask | Self::Allow(Approver::Person), Exposure::Documents) => {
                WriteDecision::Ask(Hold::ReadDocuments)
            }
            (Self::Allow(Approver::Nobody), Exposure::Documents) => {
                WriteDecision::Refuse(Hold::ReadDocuments)
            }
        }
    }

    /// The system prompt's permissions paragraph for this policy.
    #[must_use]
    pub const fn prompt_paragraph(self) -> &'static str {
        match self {
            Self::Allow(Approver::Person) => {
                "Permissions: SELECT queries always run. The user has permitted statements that \
                 modify the workspace for this session, so when asked to change data, run the \
                 statement with run_sql rather than asking for confirmation. Once this turn has \
                 searched the documents or the graph, the user is asked to approve each such \
                 statement before it executes. If the tool reports it was refused, do not retry \
                 it; tell the user.\n"
            }
            Self::Allow(Approver::Nobody) => {
                "Permissions: SELECT queries always run. The user has permitted statements that \
                 modify the workspace for this session, so when asked to change data, run the \
                 statement with run_sql rather than asking for confirmation. Once this turn has \
                 searched the documents or the graph, such a statement is refused, because \
                 nobody can approve it here. If the tool reports it was refused, do not retry \
                 it; tell the user which statement was not run.\n"
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

    /// A write permitted up front is allowed, and remembers whether the
    /// interface can ask; otherwise each interface keeps its own fallback.
    #[test]
    fn allowed_if_overrides_the_fallback_only_when_permitted() {
        assert_eq!(
            WritePolicy::Deny.allowed_if(true),
            WritePolicy::Allow(Approver::Nobody)
        );
        assert_eq!(
            WritePolicy::Ask.allowed_if(true),
            WritePolicy::Allow(Approver::Person)
        );
        assert_eq!(WritePolicy::Deny.allowed_if(false), WritePolicy::Deny);
        assert_eq!(WritePolicy::Ask.allowed_if(false), WritePolicy::Ask);
        assert!(WritePolicy::Ask.allowed_if(true).allows_unasked());
        assert!(!WritePolicy::Ask.allows_unasked() && !WritePolicy::Deny.allows_unasked());
    }

    /// Before a turn reads document text each policy decides as it always
    /// has; after, nothing runs unasked.
    #[test]
    fn a_turn_that_read_documents_runs_no_write_unasked() {
        use WriteDecision::{Ask, Refuse, Run};
        let decide = |policy: WritePolicy| {
            (
                policy.decide(Exposure::None),
                policy.decide(Exposure::Documents),
            )
        };
        assert_eq!(
            decide(WritePolicy::Allow(Approver::Nobody)),
            (Run, Refuse(Hold::ReadDocuments))
        );
        assert_eq!(
            decide(WritePolicy::Allow(Approver::Person)),
            (Run, Ask(Hold::ReadDocuments))
        );
        assert_eq!(
            decide(WritePolicy::Ask),
            (Ask(Hold::NotPermitted), Ask(Hold::ReadDocuments))
        );
        assert_eq!(
            decide(WritePolicy::Deny),
            (Refuse(Hold::NotPermitted), Refuse(Hold::NotPermitted))
        );
    }

    /// Only the document hold tells the person more than the statement, and
    /// its refusal does not send the model to --allow-write.
    #[test]
    fn a_hold_says_why_to_the_model_the_step_and_the_person() {
        assert!(Hold::NotPermitted.notice().is_none());
        assert!(Hold::ReadDocuments.notice().is_some());
        assert_eq!(Hold::NotPermitted.summary(), "refused");
        assert!(Hold::ReadDocuments.summary().contains("read document text"));
        assert!(Hold::NotPermitted.refusal().contains("--allow-write"));
        assert!(!Hold::ReadDocuments.refusal().contains("--allow-write"));
        assert_eq!(Hold::ReadDocuments.as_str(), "read_documents");
        assert!(Hold::NotPermitted < Hold::ReadDocuments);
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
