//! Recovery hooks on the agent loop: a tool call the model got wrong, a
//! model turn that came back empty, and an answer that states figures the
//! workspace did not give it get another try instead of ending the turn.
//! Local models do all three.

use std::future;

use rig::agent::{
    AgentHook, DispatchAction, DispatchEvent, HookContext, InvalidToolCallAction,
    InvalidToolCallContext, InvalidToolCallReason, ModelTurnAction, ModelTurnFinished,
    OutcomeAction, OutcomeEvent,
};
use rig::completion::FinishReason;
use rig::message::AssistantContent;

use super::grounding;
use super::policy::Exposure;
use super::think;
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

/// A tool call that repeats one the turn already made, the same tool with
/// the same arguments, is not run again: the model is told its result is
/// already in the conversation. A small model offered a tool it cannot use
/// well calls it over and over until the turn limit (the evaluation saw a
/// 3B model spend all 15 calls on one `search_graph`).
pub(crate) struct RepeatedCalls;

/// The calls the run has made, as tool name and canonical arguments.
#[derive(Clone, Default)]
struct CallsMade(Vec<(String, String)>);

impl RepeatedCalls {
    /// The arguments with their keys in one order and no whitespace, so two
    /// calls that differ only in spelling compare equal.
    fn canonical(args: &str) -> String {
        serde_json::from_str::<serde_json::Value>(args)
            .map_or_else(|_| args.trim().to_owned(), |value| value.to_string())
    }

    fn decide(ctx: &HookContext, event: &DispatchEvent<'_>) -> DispatchAction {
        let (Some(name), Some(args)) = (event.tool_name(), event.tool_args()) else {
            return DispatchAction::proceed();
        };
        let call = (name.to_owned(), Self::canonical(args));
        let repeated = ctx.scratchpad().update(|CallsMade(made): &mut CallsMade| {
            if made.contains(&call) {
                true
            } else {
                made.push(call.clone());
                false
            }
        });
        if repeated {
            tracing::warn!(
                tool = name,
                "the model repeated a tool call; not running it again"
            );
            DispatchAction::skip(format!(
                "You already called {name} with these exact arguments this turn; its result is \
                 above. Answer from it, or call a tool with different arguments."
            ))
        } else {
            DispatchAction::proceed()
        }
    }
}

impl AgentHook for RepeatedCalls {
    fn on_dispatch(
        &self,
        ctx: &HookContext,
        event: DispatchEvent<'_>,
    ) -> impl Future<Output = DispatchAction> + Send {
        future::ready(Self::decide(ctx, &event))
    }
}

/// Every figure an answer states must be one the turn showed the model
/// (`grounding`): `DuckDB` provides the answers, and the model only writes the
/// SQL and the tool calls that fetch them. The hook records every tool
/// result as the model receives it, and a final answer (text, no tool call)
/// stating a figure nothing the turn showed holds is asked for once more,
/// with those figures named. A second such answer is not shown: the turn
/// answers with the query results instead (`grounding::fallback`).
/// Registered in a workspace that holds tables or documents.
pub(crate) struct Unsourced {
    turn: Turn,
}

/// The run's count of answers already asked for again.
#[derive(Clone, Copy, Default)]
struct UnsourcedRetries(u8);

impl Unsourced {
    const RETRIES: u8 = 1;

    pub(crate) const fn new(turn: Turn) -> Self {
        Self { turn }
    }

    /// Whether the turn has looked at the workspace: a tool ran, or
    /// `always_retrieve` put document text in the prompt.
    fn looked(&self) -> bool {
        !self.turn.recorder.steps().is_empty() || self.turn.exposure() == Exposure::Documents
    }

    /// A final answer's text; `None` for a turn that calls a tool.
    fn final_text(event: &ModelTurnFinished<'_>) -> Option<String> {
        let mut text = String::new();
        for part in event.content {
            match part {
                AssistantContent::Text(part) => text.push_str(&part.text),
                AssistantContent::ToolCall(_) => return None,
                _ => {}
            }
        }
        Some(text)
    }

    fn decide(&self, ctx: &HookContext, event: &ModelTurnFinished<'_>) -> ModelTurnAction {
        let Some(text) = Self::final_text(event) else {
            return ModelTurnAction::continue_run();
        };
        let unsourced = self.turn.unsourced(&think::strip(&text));
        if unsourced.is_empty() {
            return ModelTurnAction::continue_run();
        }
        let retry = ctx
            .scratchpad()
            .update(|UnsourcedRetries(spent): &mut UnsourcedRetries| {
                let retry = *spent < Self::RETRIES;
                if retry {
                    *spent = spent.saturating_add(1);
                }
                retry
            });
        if retry {
            tracing::warn!(
                ?unsourced,
                "the answer states figures no result holds; asking again"
            );
            ModelTurnAction::retry_with_feedback(grounding::feedback(&unsourced, self.looked()))
        } else {
            ModelTurnAction::continue_run()
        }
    }
}

impl AgentHook for Unsourced {
    fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> impl Future<Output = ModelTurnAction> + Send {
        future::ready(self.decide(ctx, &event))
    }

    /// Record what each tool returned, as the model reads it.
    fn on_outcome(
        &self,
        _ctx: &HookContext,
        event: OutcomeEvent<'_>,
    ) -> impl Future<Output = OutcomeAction> + Send {
        if let Some(result) = event.tool_result() {
            let output = result.output();
            match output.as_text() {
                Some(text) => self.turn.saw(text),
                None => self
                    .turn
                    .saw(&serde_json::to_string(output).unwrap_or_default()),
            }
        }
        future::ready(OutcomeAction::Proceed)
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
