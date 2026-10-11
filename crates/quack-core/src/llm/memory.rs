//! The history a turn replays: the session's turns ([`SessionMemory`])
//! under the token window ([`TranscriptWindow`]), and with
//! `[analysis].compact_history` the stored summary of the turns the window
//! leaves out leading it. The summary is written after a turn, through
//! rig's compacting memory ([`SessionCompactor`]), so a turn never waits
//! for a model call before its own.

use std::sync::Arc;
use std::time::Duration;

use rig::id::ConversationId;
use rig::memory::{CompactingMemory, Compactor, ConversationMemory, MemoryError, MemoryPolicy};
use rig::message::Message;
use rig::wasm_compat::WasmBoxedFuture;

use schemars::{JsonSchema, schema_for};
use serde::Deserialize;

use super::after_turn::AfterTurn;
use super::{ChatClient, SchemaCall, Task};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::ids::SessionId;
use crate::storage::sessions::{self, SessionMemory, SpokenText, TranscriptWindow};
use crate::storage::writer::Writer;
use crate::text::Tokens;

/// What the summarizing model is told.
const SUMMARY_PROMPT: &str = "You summarize the earlier part of a conversation between a person \
    and a data assistant, so the conversation can go on without it. Keep the questions asked, \
    the names of tables, documents, and entities, the numbers found, and the conclusions. \
    Answer with the summary, in plain prose, in `summary`.";

/// The summarizing model's answer.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, JsonSchema)]
#[schemars(title = "history_summary")]
struct SummaryAnswer {
    summary: String,
}

/// How long one summary call may run.
const SUMMARY_TIMEOUT: Duration = Duration::from_secs(120);

/// A session's history as the next turn replays it.
pub struct History {
    db: Arc<Writer>,
    budget: Tokens,
    compactor: Option<SessionCompactor>,
}

impl History {
    /// The history under `[analysis]`: the window always, and the
    /// summarizing compactor when `compact_history` is on and the chat
    /// model takes its summary call.
    ///
    /// # Errors
    ///
    /// Returns an error if the chat model's client cannot be built.
    pub async fn from_config(config: &Config, db: Arc<Writer>) -> Result<Self> {
        let budget = config.analysis.history_token_budget;
        let compactor = if config.analysis.compact_history {
            let chat = config.background_model_ref()?;
            ChatClient::build(config, &chat)
                .await?
                .optional_schema_call(
                    chat.model,
                    config.model_settings(chat),
                    Task {
                        preamble: SUMMARY_PROMPT,
                        timeout: SUMMARY_TIMEOUT,
                        label: "history summary",
                    },
                    schema_for!(SummaryAnswer),
                )
                .map(|summarizer| SessionCompactor {
                    db: Arc::clone(&db),
                    summarizer: Arc::new(summarizer),
                    max_tokens: SessionCompactor::cap(budget),
                })
        } else {
            None
        };
        Ok(Self {
            db,
            budget,
            compactor,
        })
    }

    /// The messages to replay for `session`, oldest first: the stored
    /// summary of the earlier turns first, when compaction is on and the
    /// window leaves turns out. No model is called: the summary was written
    /// after an earlier turn ([`Self::follow_turn`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be read.
    pub async fn load(&self, session: &SessionId) -> Result<Vec<Message>> {
        let id = ConversationId::from(session.as_str());
        let memory = SessionMemory::new(Arc::clone(&self.db));
        let window = TranscriptWindow::new(self.budget);
        let failed = |e| Error::Analysis(format!("could not load the session history: {e}"));
        let messages = memory.load(&id).await.map_err(failed)?;
        let (mut kept, left_out) = window.apply_with_demoted(messages).map_err(failed)?;
        if self.compactor.is_none() || left_out.is_empty() {
            return Ok(kept);
        }
        let read = session.clone();
        let stored = self
            .db
            .run(move |db| sessions::latest_summary(db, &read))
            .await?;
        // A summary covering more than the window leaves out describes turns
        // it now keeps verbatim (the budget grew): the next follow-up writes
        // a new one.
        if let Some(stored) = stored
            && stored.covers <= left_out.len()
            && !stored.text.is_empty()
        {
            kept.insert(0, Message::from(Summary(stored.text)));
        }
        Ok(kept)
    }

    /// Summarize what the window leaves out of `session` that the stored
    /// summary does not cover yet, and store the result; nothing without a
    /// compactor.
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be read or the summary fails.
    pub async fn compact(self, session: &SessionId) -> Result<()> {
        let Some(compactor) = self.compactor else {
            return Ok(());
        };
        let id = ConversationId::from(session.as_str());
        let memory = SessionMemory::new(Arc::clone(&self.db));
        // Built for this one load, so rig hands the compactor every message
        // the window leaves out and no carry-over: the summary stored in the
        // workspace is the carry-over.
        CompactingMemory::new(memory, TranscriptWindow::new(self.budget), compactor)
            .load(&id)
            .await
            .map(drop)
            .map_err(|e| Error::Analysis(format!("could not summarize the session history: {e}")))
    }

    /// After a turn is recorded, bring `session`'s summary up to date on its
    /// own task (`AfterTurn::spawn`) when `compact_history` is on, so the
    /// next turn's [`Self::load`] finds it. A failure is logged; the next
    /// turn replays the window with the summary it has.
    pub fn follow_turn(config: &Config, db: &Arc<Writer>, session: &SessionId) {
        if !config.analysis.compact_history {
            return;
        }
        let (config, db, session) = (config.clone(), Arc::clone(db), session.clone());
        AfterTurn::spawn(async move {
            let compacted = match Self::from_config(&config, db).await {
                Ok(history) => history.compact(&session).await,
                Err(e) => Err(e),
            };
            if let Err(e) = compacted {
                tracing::warn!(session = %session, error = %e, "the session history was not summarized");
            }
        });
    }
}

/// Summarizes the turns the window leaves out with the chat model, and
/// keeps each summary in the workspace (`_quack_session_summaries`) with
/// how many messages it covers, so a later turn reuses it and only the
/// messages evicted since are summarized again, with it.
pub struct SessionCompactor {
    db: Arc<Writer>,
    summarizer: Arc<SchemaCall<SummaryAnswer>>,
    /// The most a summary may hold, outside the window's own budget.
    max_tokens: Tokens,
}

impl SessionCompactor {
    /// A quarter of the history budget.
    fn cap(budget: Tokens) -> Tokens {
        Tokens::new(budget.get().div_ceil(4))
    }

    /// The request for one summary: the summary so far, then the messages
    /// it does not cover yet.
    fn request(previous: Option<&str>, messages: &[Message]) -> String {
        let mut text = String::new();
        if let Some(previous) = previous {
            text.push_str("The summary so far:\n");
            text.push_str(previous);
            text.push_str("\n\nWhat came after it:\n");
        }
        for message in messages {
            let (who, said) = match message {
                Message::User { .. } => ("Person", message.spoken_text()),
                Message::Assistant { .. } => ("Assistant", message.spoken_text()),
                Message::System { content } => ("Summary", content.clone()),
            };
            text.push_str(who);
            text.push_str(": ");
            text.push_str(&said);
            text.push('\n');
        }
        text
    }
}

/// A summary leading the replayed history, as a system message: it is
/// context about the conversation, not a turn in it.
#[derive(Debug, Clone)]
pub struct Summary(String);

impl From<Summary> for Message {
    fn from(summary: Summary) -> Self {
        Self::System {
            content: format!("Summary of the earlier conversation: {}", summary.0),
        }
    }
}

impl Compactor for SessionCompactor {
    type Artifact = Summary;

    fn compact<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
        evicted: &'a [Message],
        _carry_over: Option<&'a Summary>,
    ) -> WasmBoxedFuture<'a, std::result::Result<Summary, MemoryError>> {
        Box::pin(async move {
            let session = SessionId::from(conversation_id.as_str());
            let read = session.clone();
            let stored = self
                .db
                .run(move |db| sessions::latest_summary(db, &read))
                .await
                .map_err(MemoryError::backend)?;
            let covered = stored.as_ref().map_or(0, |s| s.covers);
            // A budget that grew across a restart evicts fewer messages than a
            // stored summary claims to cover: its `covers` is no longer a
            // valid index into `evicted`, and the kept window already holds the
            // turns it described, so summarize again from scratch.
            let over_covered = covered > evicted.len();
            let to_summarize: &[Message] = if over_covered {
                evicted
            } else {
                evicted.get(covered..).unwrap_or_default()
            };
            if to_summarize.is_empty() {
                // Nothing left the window since the stored summary.
                return Ok(Summary(stored.map(|s| s.text).unwrap_or_default()));
            }
            // Re-summarizing from scratch drops the old, over-covering text:
            // it described turns that are now kept verbatim, not "so far".
            let prev = if over_covered {
                None
            } else {
                stored.as_ref().map(|s| s.text.as_str())
            };
            let request = Self::request(prev, to_summarize);
            tracing::info!(session = %session, messages = to_summarize.len(), "summarizing earlier turns");
            let answer = self
                .summarizer
                .answer(&request)
                .await
                .map_err(MemoryError::backend)?;
            let text: String = answer
                .summary
                .trim()
                .chars()
                .take(self.max_tokens.chars())
                .collect();
            let (saved, covers) = (text.clone(), evicted.len());
            self.db
                .run(move |db| sessions::save_summary(db, &session, covers, &saved))
                .await
                .map_err(MemoryError::backend)?;
            Ok(Summary(text))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::agent::AgentResponse;
    use crate::embedding::Dimension;
    use crate::llm::egress::Egress;
    use crate::storage::sessions::ChatMode;
    use crate::storage::workspace::WorkspaceDb;
    use jiff::Timestamp;
    use rig::completion::Usage;
    use rig::test_utils::{MockCompletionModel, MockStreamEvent};

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    /// A session of five turns in a fresh workspace, behind a writer.
    async fn session() -> (Arc<Writer>, SessionId) {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        let db = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
        let id = db
            .run(|db| {
                let session = sessions::create_session(db, "m", ChatMode::Chat, None)?;
                for i in 1..=5 {
                    let answer = AgentResponse {
                        content: format!("answer {i}"),
                        ..AgentResponse::default()
                    };
                    sessions::record_turn(
                        db,
                        &session.id,
                        &format!("question {i}"),
                        Timestamp::now(),
                        &answer,
                    )?;
                }
                Ok(session.id)
            })
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        (db, id)
    }

    fn compacting(db: &Arc<Writer>, model: &MockCompletionModel, budget: u32) -> History {
        History {
            db: Arc::clone(db),
            budget: Tokens::new(budget),
            compactor: Some(SessionCompactor {
                db: Arc::clone(db),
                summarizer: Arc::new(SchemaCall::new(
                    model.clone().erase().into(),
                    Task {
                        preamble: SUMMARY_PROMPT,
                        timeout: SUMMARY_TIMEOUT,
                        label: "history summary",
                    },
                    schema_for!(SummaryAnswer),
                )),
                max_tokens: Tokens::new(100),
            }),
        }
    }

    /// A turn's follow-up under `budget`, then the next turn's load.
    async fn compacted(
        db: &Arc<Writer>,
        model: &MockCompletionModel,
        budget: u32,
        id: &SessionId,
    ) -> Vec<Message> {
        compacting(db, model, budget)
            .compact(id)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        compacting(db, model, budget)
            .load(id)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn evicted_turns_are_summarized_once_and_lead_the_window() {
        let (db, id) = session().await;
        // One scripted summary: a second model call would fail the test.
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text(r#"{"summary": "They asked questions 1 to 3."}"#),
            MockStreamEvent::final_response(Usage::default()),
        ]]);
        // Each turn is 3 + 2 tokens, so 10 keeps the last two. Loading
        // calls no model: there is no summary until a turn's follow-up.
        let window = compacting(&db, &model, 10)
            .load(&id)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(window.len(), 4);
        assert_eq!(model.request_count(), 0);
        let history = compacted(&db, &model, 10, &id).await;
        let shown: Vec<String> = history.iter().map(SpokenText::spoken_text).collect();
        assert_eq!(
            shown,
            [
                "Summary of the earlier conversation: They asked questions 1 to 3.",
                "question 4",
                "answer 4",
                "question 5",
                "answer 5"
            ]
        );
        assert!(matches!(history.first(), Some(Message::System { .. })));
        let request = serde_json::to_string(&model.requests()).unwrap_or_default();
        assert!(
            request.contains("Person: question 1") && request.contains("Assistant: answer 3"),
            "{request}"
        );

        // A follow-up with nothing newly left out makes no model call.
        let again = compacted(&db, &model, 10, &id).await;
        assert_eq!(again, history);
        let rows = db
            .run(|db| {
                Ok(db.connection().query_row(
                    "SELECT count(*), max(covers) FROM _quack_session_summaries",
                    [],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                )?)
            })
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(rows, (1, 6), "one summary of the first six messages");

        // Deleting the session takes its summaries with it.
        let left = db
            .run(move |db| {
                sessions::delete_session(db, &id)?;
                Ok(db.connection().query_row(
                    "SELECT count(*) FROM _quack_session_summaries",
                    [],
                    |row| row.get::<_, i64>(0),
                )?)
            })
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(left, 0);
    }

    /// Raising `history_token_budget` across a resume evicts fewer messages
    /// than a stored summary claims to cover, so the compactor summarizes
    /// again from scratch over the smaller evicted prefix instead of
    /// short-circuiting and replaying a stale summary over turns the kept
    /// window already holds verbatim.
    #[tokio::test(flavor = "multi_thread")]
    async fn growing_the_window_re_summarizes_instead_of_replaying_a_stale_summary() {
        let (db, id) = session().await;
        // Two scripted summaries: the first covers turns 1-3; the second
        // re-summarizes turn one after the window grows. A third call fails.
        let model = MockCompletionModel::from_stream_turns([
            [
                MockStreamEvent::text(r#"{"summary": "They asked questions 1 to 3."}"#),
                MockStreamEvent::final_response(Usage::default()),
            ],
            [
                MockStreamEvent::text(r#"{"summary": "They asked question 1."}"#),
                MockStreamEvent::final_response(Usage::default()),
            ],
        ]);
        // Each turn is 3 + 2 tokens, so budget 10 evicts turns 1-3.
        let first = compacted(&db, &model, 10, &id).await;
        let shown: Vec<String> = first.iter().map(SpokenText::spoken_text).collect();
        assert_eq!(
            shown,
            [
                "Summary of the earlier conversation: They asked questions 1 to 3.",
                "question 4",
                "answer 4",
                "question 5",
                "answer 5"
            ]
        );
        assert_eq!(model.request_count(), 1);

        // Budget 20 keeps turns 2-5 and evicts only turn one: the stored
        // `covers` (6) now exceeds `evicted.len()` (2), so the compactor
        // summarizes again from scratch rather than replay the stale summary.
        // Until it has, the load leaves the stale summary out.
        let unsummarized = compacting(&db, &model, 20)
            .load(&id)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            !matches!(unsummarized.first(), Some(Message::System { .. })),
            "{unsummarized:?}"
        );
        let second = compacted(&db, &model, 20, &id).await;
        let shown: Vec<String> = second.iter().map(SpokenText::spoken_text).collect();
        assert_eq!(
            shown,
            [
                "Summary of the earlier conversation: They asked question 1.",
                "question 2",
                "answer 2",
                "question 3",
                "answer 3",
                "question 4",
                "answer 4",
                "question 5",
                "answer 5"
            ]
        );
        assert_eq!(model.request_count(), 2);
        // The re-summarize covers turn one alone, without the old summary as
        // "the summary so far".
        let second_request = serde_json::to_string(
            model
                .requests()
                .get(1)
                .unwrap_or_else(|| fail("the re-summarize request was not made")),
        )
        .unwrap_or_default();
        assert!(
            second_request.contains("Person: question 1")
                && second_request.contains("Assistant: answer 1"),
            "{second_request}"
        );
        assert!(
            !second_request.contains("The summary so far")
                && !second_request.contains("Assistant: answer 3"),
            "the stale summary must not lead the re-summarize: {second_request}"
        );
        // The next load reuses the newest summary without another model call:
        // `latest_summary` returns the new lower-`covers` row, not the old
        // higher-`covers` one, so the compactor does not loop on every load.
        let again = compacted(&db, &model, 20, &id).await;
        assert_eq!(again, second);
        assert_eq!(model.request_count(), 2);
        let covers = db
            .run(move |db| {
                Ok(db.connection().query_row(
                    "SELECT covers FROM _quack_session_summaries ORDER BY id DESC LIMIT 1",
                    [],
                    |row| row.get::<_, i64>(0),
                )?)
            })
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            covers, 2,
            "the newest summary covers the new evicted prefix"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn without_compaction_the_window_alone_is_replayed() {
        let (db, id) = session().await;
        let history = History {
            db: Arc::clone(&db),
            budget: Tokens::new(10),
            compactor: None,
        }
        .load(&id)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
        let shown: Vec<String> = history.iter().map(SpokenText::spoken_text).collect();
        assert_eq!(shown, ["question 4", "answer 4", "question 5", "answer 5"]);
    }

    /// A `background_effort` the model refuses leaves the window alone to
    /// replay; the turn is not refused for its summary call.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_summary_call_the_model_refuses_leaves_the_window_alone() {
        Egress::scope(Some(Egress::NoWorkspace), async {
            let (db, _id) = session().await;
            for (background_effort, compacts) in [("none", true), ("minimal", false)] {
                let config = Config::parse(&format!(
                    "[general]\nchat_model = \"p/gpt-5.6-sol\"\n\
                 [analysis]\ncompact_history = true\neffort = \"none\"\n\
                 background_effort = \"{background_effort}\"\n\
                 [providers.p]\ntype = \"openai\"\napi = \"chat-completions\"\n\
                 auth = \"api-key\"\napi_key_env = \"CARGO_PKG_NAME\"\n\
                 base_url = \"http://127.0.0.1:9\"\n"
                ))
                .unwrap_or_else(|e| fail(&e.to_string()));
                let history = History::from_config(&config, Arc::clone(&db))
                    .await
                    .unwrap_or_else(|e| fail(&e.to_string()));
                assert_eq!(history.compactor.is_some(), compacts, "{background_effort}");
            }
        })
        .await;
    }
}
