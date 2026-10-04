//! Recovery hooks on the agent loop: a tool call the model got wrong, and a
//! model turn that came back empty, get another try instead of ending the
//! turn. Local models do both.

use std::future;

use rig::agent::{
    AgentHook, HookContext, InvalidToolCallAction, InvalidToolCallContext, ModelTurnAction,
    ModelTurnFinished,
};
use rig::completion::FinishReason;
use rig::message::AssistantContent;

/// How many times one turn asks the model again after an invalid tool call.
/// Each retry also counts against `[analysis].max_turns`.
pub(crate) const INVALID_TOOL_CALL_RETRIES: usize = 2;

/// A call to a tool that does not exist: the name is repaired when it is a
/// registered tool spelled with other case or stray whitespace, and
/// otherwise the model is told which tools exist. rig answers a call whose
/// arguments are not JSON itself, with an error result the model reads.
pub(crate) struct InvalidToolCalls;

impl InvalidToolCalls {
    fn action(call: &InvalidToolCallContext) -> InvalidToolCallAction {
        let wanted = call.tool_name.trim().to_lowercase();
        match call.allowed_tools.iter().find(|tool| **tool == wanted) {
            Some(tool) => InvalidToolCallAction::repair(tool.clone()),
            None => InvalidToolCallAction::retry(format!(
                "There is no tool named `{}`. The tools you can call are: {}. Call one of them \
                 by its exact name, or answer without a tool.",
                call.tool_name,
                call.allowed_tools.join(", ")
            )),
        }
    }
}

impl AgentHook for InvalidToolCalls {
    fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> impl Future<Output = Option<InvalidToolCallAction>> + Send {
        let action = Self::action(event);
        tracing::warn!(tool = %event.tool_name, ?action, "the model's tool call cannot run as written");
        future::ready(Some(action))
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
        let cut_off = match turn.finish_reason {
            Some(FinishReason::Length | FinishReason::ContentFilter) => true,
            Some(FinishReason::Stop | FinishReason::ToolCalls | FinishReason::Other(_)) | None => {
                false
            }
        };
        !cut_off
            && turn.content.iter().all(|part| match part {
                AssistantContent::Text(text) => text.text.trim().is_empty(),
                AssistantContent::ToolCall(_) | AssistantContent::Image(_) => false,
                // Reasoning or a provider's own item alone answers nothing.
                AssistantContent::Reasoning(_) | AssistantContent::Opaque(_) => true,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str) -> InvalidToolCallContext {
        let tools = vec![String::from("list_tables"), String::from("run_sql")];
        InvalidToolCallContext {
            tool_name: name.to_owned(),
            tool_call_id: None,
            args: None,
            available_tools: tools.clone(),
            allowed_tools: tools,
            tool_choice: None,
            chat_history: Vec::new(),
            is_streaming: true,
        }
    }

    /// The feedback a retry sends, or what the action was instead.
    fn retry_feedback(action: InvalidToolCallAction) -> String {
        match action {
            InvalidToolCallAction::Retry { feedback } => feedback,
            other @ (InvalidToolCallAction::Fail
            | InvalidToolCallAction::Repair { .. }
            | InvalidToolCallAction::Skip { .. }
            | InvalidToolCallAction::Stop { .. }) => format!("not a retry: {other:?}"),
        }
    }

    #[test]
    fn a_misspelled_case_is_repaired_and_anything_else_retried() {
        assert_eq!(
            InvalidToolCalls::action(&call(" Run_SQL ")),
            InvalidToolCallAction::repair("run_sql")
        );
        let feedback = retry_feedback(InvalidToolCalls::action(&call("default_api")));
        assert!(feedback.contains("`default_api`"), "{feedback}");
        assert!(feedback.contains("list_tables, run_sql"), "{feedback}");
    }
}
