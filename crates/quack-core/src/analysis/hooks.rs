//! Recovery hooks on the agent loop: a tool call the model got wrong, a
//! model turn that came back empty, and an answer that states figures
//! without having looked at the workspace get another try instead of ending
//! the turn. Local models do all three.

use std::future;

use rig::agent::{
    AgentHook, HookContext, InvalidToolCallAction, InvalidToolCallContext, InvalidToolCallReason,
    ModelTurnAction, ModelTurnFinished,
};
use rig::completion::FinishReason;
use rig::message::AssistantContent;

use super::policy::Exposure;
use super::tools::Turn;

/// How many times one turn asks the model again after an invalid tool call.
/// Each retry also counts against `[analysis].max_turns`.
pub(crate) const INVALID_TOOL_CALL_RETRIES: usize = 2;

/// A call to a tool the turn cannot run: the name is repaired when it is a
/// registered tool spelled with other case or stray whitespace, and
/// otherwise the model is told which tools exist. rig answers a call whose
/// arguments are not JSON itself, with an error result the model reads.
pub(crate) struct InvalidToolCalls;

impl InvalidToolCalls {
    fn action(call: &InvalidToolCallContext) -> Option<InvalidToolCallAction> {
        match call.reason {
            InvalidToolCallReason::UnknownTool | InvalidToolCallReason::DisallowedByToolChoice => {}
            _ => return None,
        }
        let wanted = call.tool_name.trim().to_lowercase();
        Some(
            match call.allowed_tools.iter().find(|tool| **tool == wanted) {
                Some(tool) => InvalidToolCallAction::repair(tool.clone()),
                None => InvalidToolCallAction::retry(format!(
                    "There is no tool named `{}`. The tools you can call are: {}. Call one of them \
                 by its exact name, or answer without a tool.",
                    call.tool_name,
                    call.allowed_tools.join(", ")
                )),
            },
        )
    }
}

impl AgentHook for InvalidToolCalls {
    fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> impl Future<Output = Option<InvalidToolCallAction>> + Send {
        let action = Self::action(event);
        tracing::warn!(tool = %event.tool_name, reason = ?event.reason, ?action, "the model's tool call cannot run as written");
        future::ready(action)
    }
}

/// A model turn with no text and no tool call is asked for once more; a
/// second empty turn is accepted and the answer says the model returned
/// nothing. A turn the output limit or a content filter cut off is not
/// retried: the same request would stop the same way, and the turn's note
/// says why.
pub(crate) struct EmptyAnswer;

/// The run's count of empty turns already retried, kept in the hook
/// scratchpad.
#[derive(Clone, Copy, Default)]
struct EmptyRetries(u8);

impl EmptyAnswer {
    const RETRIES: u8 = 1;
    const FEEDBACK: &str = "Your last reply was empty. Answer the question, or call a tool if \
                            you need data to answer it.";

    fn is_empty(turn: &ModelTurnFinished<'_>) -> bool {
        let cut_off = matches!(
            turn.finish_reason,
            Some(FinishReason::Length | FinishReason::ContentFilter)
        );
        !cut_off
            && turn.content.iter().all(|part| match part {
                AssistantContent::Text(text) => text.text.trim().is_empty(),
                AssistantContent::ToolCall(_) | AssistantContent::Image(_) => false,
                // Reasoning, a provider's own item, or anything rig adds later
                // alone answers nothing.
                _ => true,
            })
    }
}

impl AgentHook for EmptyAnswer {
    fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> impl Future<Output = ModelTurnAction> + Send {
        future::ready(Self::decide(ctx, &event))
    }
}

impl EmptyAnswer {
    fn decide(ctx: &HookContext, event: &ModelTurnFinished<'_>) -> ModelTurnAction {
        if !Self::is_empty(event) {
            return ModelTurnAction::continue_run();
        }
        let retry = ctx
            .scratchpad()
            .update(|EmptyRetries(spent): &mut EmptyRetries| {
                let retry = *spent < Self::RETRIES;
                if retry {
                    *spent = spent.saturating_add(1);
                }
                retry
            });
        if retry {
            tracing::warn!("the model returned an empty turn; asking again");
            ModelTurnAction::retry_with_feedback(Self::FEEDBACK)
        } else {
            ModelTurnAction::continue_run()
        }
    }
}

/// An answer that states figures (any digit) while the turn has run no tool
/// and read no document is asked for once more, with the tools named: small
/// models answer from memory instead of the workspace and sound just as
/// sure. A second such answer is accepted, and the turn notes that it does
/// not come from the workspace ([`Ungrounded::NOTE`]). Registered only in a
/// workspace that holds tables or documents, on a turn with no history,
/// since a follow-up may rightly restate figures an earlier turn found.
pub(crate) struct Ungrounded {
    turn: Turn,
}

/// The run's count of ungrounded answers already retried.
#[derive(Clone, Copy, Default)]
struct UngroundedRetries(u8);

impl Ungrounded {
    const RETRIES: u8 = 1;
    const FEEDBACK: &str = "Your answer states figures, but this turn ran no query and read no \
                            document, so they do not come from this workspace. Use the tools \
                            (run_sql for tables, search_documents for documents) and answer from \
                            what they return. If the workspace does not cover the question, say so.";
    /// The note an answer that stayed ungrounded carries.
    pub(crate) const NOTE: &str = "No query ran and no document was read this turn, so this answer \
                                   does not come from the workspace's data.";

    pub(crate) const fn new(turn: Turn) -> Self {
        Self { turn }
    }

    /// Whether `text` states figures the workspace could be the source of.
    pub(crate) fn states_figures(text: &str) -> bool {
        text.chars().any(|c| c.is_ascii_digit())
    }

    /// Whether the turn has looked at the workspace: a tool ran, or
    /// `always_retrieve` put document text in the prompt.
    pub(crate) fn grounded(turn: &Turn) -> bool {
        !turn.recorder.steps().is_empty() || turn.exposure() == Exposure::Documents
    }

    /// A final answer (text, no tool call) that states figures unseen.
    fn is_ungrounded(&self, event: &ModelTurnFinished<'_>) -> bool {
        let mut text = String::new();
        for part in event.content {
            match part {
                AssistantContent::Text(part) => text.push_str(&part.text),
                AssistantContent::ToolCall(_) => return false,
                _ => {}
            }
        }
        Self::states_figures(&text) && !Self::grounded(&self.turn)
    }

    fn decide(&self, ctx: &HookContext, event: &ModelTurnFinished<'_>) -> ModelTurnAction {
        if !self.is_ungrounded(event) {
            return ModelTurnAction::continue_run();
        }
        let retry = ctx
            .scratchpad()
            .update(|UngroundedRetries(spent): &mut UngroundedRetries| {
                let retry = *spent < Self::RETRIES;
                if retry {
                    *spent = spent.saturating_add(1);
                }
                retry
            });
        if retry {
            tracing::warn!("the model answered with figures without using a tool; asking again");
            ModelTurnAction::retry_with_feedback(Self::FEEDBACK)
        } else {
            ModelTurnAction::continue_run()
        }
    }
}

impl AgentHook for Ungrounded {
    fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> impl Future<Output = ModelTurnAction> + Send {
        future::ready(self.decide(ctx, &event))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    /// The context rig hands the hook; it has no constructor outside rig, so
    /// it is read from its serialized form.
    fn call(name: &str, reason: serde_json::Value) -> InvalidToolCallContext {
        let mut context = serde_json::json!({
            "tool_name": name,
            "tool_call_id": null,
            "args": null,
            "available_tools": ["list_tables", "run_sql"],
            "allowed_tools": ["list_tables", "run_sql"],
            "tool_choice": null,
            "chat_history": [],
            "is_streaming": true,
        });
        if let Some(fields) = context.as_object_mut() {
            fields.insert(String::from("reason"), reason);
        }
        serde_json::from_value(context).unwrap_or_else(|e| fail(&e.to_string()))
    }

    fn unknown(name: &str) -> InvalidToolCallContext {
        call(name, serde_json::json!({ "reason": "unknown_tool" }))
    }

    /// The feedback a retry sends, or what the action was instead.
    fn retry_feedback(action: Option<InvalidToolCallAction>) -> String {
        match action {
            Some(InvalidToolCallAction::Retry { feedback }) => feedback,
            other => format!("not a retry: {other:?}"),
        }
    }

    #[test]
    fn a_misspelled_case_is_repaired_and_anything_else_retried() {
        assert_eq!(
            InvalidToolCalls::action(&unknown(" Run_SQL ")),
            Some(InvalidToolCallAction::repair("run_sql"))
        );
        let feedback = retry_feedback(InvalidToolCalls::action(&unknown("default_api")));
        assert!(feedback.contains("`default_api`"), "{feedback}");
        assert!(feedback.contains("list_tables, run_sql"), "{feedback}");
    }

    /// Renaming cannot fix arguments, and rig refuses a repair for them: the
    /// hook leaves malformed arguments to rig's own error result.
    #[test]
    fn malformed_arguments_are_left_to_rig() {
        let malformed = call(
            "run_sql",
            serde_json::json!({ "reason": "malformed_arguments", "error": "EOF while parsing" }),
        );
        assert_eq!(InvalidToolCalls::action(&malformed), None);
    }
}
