//! The history a turn replays, loaded through rig's conversation memory:
//! the session's turns ([`SessionMemory`]) under the token window
//! ([`TranscriptWindow`]), and with `[analysis].compact_history` the turns
//! the window leaves out folded into a summary ([`SessionCompactor`])
//! that leads the history.

use std::sync::Arc;
use std::time::Duration;

use rig::id::ConversationId;
use rig::memory::{CompactingMemory, Compactor, ConversationMemory, MemoryError, PolicyMemory};
use rig::message::Message;
use rig::wasm_compat::WasmBoxedFuture;

use schemars::{JsonSchema, schema_for};
use serde::Deserialize;

use super::{ChatClient, SchemaCall, Task};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::ids::SessionId;
use crate::storage::sessions::{self, SessionMemory, TranscriptWindow};
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
    /// summarizing compactor when `compact_history` is on and a chat model
    /// is configured.
    ///
    /// # Errors
    ///
    /// Returns an error if the chat model's client cannot be built.
    pub async fn from_config(config: &Config, db: Arc<Writer>) -> Result<Self> {
        let budget = config.analysis.history_token_budget;
        let compactor = if config.analysis.compact_history {
            let chat = config.chat_model_ref()?;
            let summarizer = ChatClient::build(config, &chat).await?.schema_call(
                chat.model,
                config.model_settings(chat),
                Task {
                    preamble: SUMMARY_PROMPT,
                    timeout: SUMMARY_TIMEOUT,
                    label: "history summary",
                },
                schema_for!(SummaryAnswer),
            )?;
            Some(SessionCompactor {
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

    /// The messages to replay for `session`, oldest first: a summary of the
    /// earlier turns first when there is one.
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be read or a summary fails.
    pub async fn load(self, session: &SessionId) -> Result<Vec<Message>> {
        let id = ConversationId::from(session.as_str());
        let memory = SessionMemory::new(Arc::clone(&self.db));
        let window = TranscriptWindow::new(self.budget);
        let loaded = match self.compactor {
            Some(compactor) => {
                // Built for this one load, so rig hands the compactor every
                // message the window leaves out and no carry-over: the
                // summary stored in the workspace is the carry-over.
                CompactingMemory::new(memory, window, compactor)
                    .load(&id)
                    .await
            }
            None => PolicyMemory::new(memory, window).load(&id).await,
        };
        loaded.map_err(|e| Error::Analysis(format!("could not load the session history: {e}")))
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
                Message::User { .. } => ("Person", message_text(message)),
                Message::Assistant { .. } => ("Assistant", message_text(message)),
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

/// The text parts of a message, joined.
fn message_text(message: &Message) -> String {
    use rig::message::{AssistantContent, UserContent};
    match message {
        Message::User { content } => content
            .iter()
            .filter_map(|part| match part {
                UserContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect(),
        Message::Assistant { content, .. } => content
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect(),
        Message::System { content } => content.clone(),
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
            let Some(new) = evicted.get(covered..).filter(|new| !new.is_empty()) else {
                // Nothing left the window since the stored summary.
                return Ok(Summary(stored.map(|s| s.text).unwrap_or_default()));
            };
            let request = Self::request(stored.as_ref().map(|s| s.text.as_str()), new);
            tracing::info!(session = %session, messages = new.len(), "summarizing earlier turns");
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
                    model.clone().erase(),
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

    #[tokio::test(flavor = "multi_thread")]
    async fn evicted_turns_are_summarized_once_and_lead_the_window() {
        let (db, id) = session().await;
        // One scripted summary: a second model call would fail the test.
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text(r#"{"summary": "They asked questions 1 to 3."}"#),
            MockStreamEvent::final_response(Usage::default()),
        ]]);
        // Each turn is 3 + 2 tokens, so 10 keeps the last two.
        let history = compacting(&db, &model, 10)
            .load(&id)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let shown: Vec<String> = history.iter().map(message_text).collect();
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

        // The next turn reuses the stored summary without a model call.
        let again = compacting(&db, &model, 10)
            .load(&id)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
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
        let shown: Vec<String> = history.iter().map(message_text).collect();
        assert_eq!(shown, ["question 4", "answer 4", "question 5", "answer 5"]);
    }
}
