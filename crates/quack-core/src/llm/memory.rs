//! The history a turn replays: the session's turns ([`SessionMemory`])
//! under the token window ([`TranscriptWindow`]). The turns the window
//! leaves out are not replayed; nothing stands in for them.

use std::sync::Arc;

use rig::id::ConversationId;
use rig::memory::{ConversationMemory, MemoryPolicy};
use rig::message::Message;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::ids::SessionId;
use crate::storage::sessions::{SessionMemory, TranscriptWindow};
use crate::storage::writer::Writer;
use crate::text::Tokens;

/// A session's history as the next turn replays it.
pub struct History {
    db: Arc<Writer>,
    budget: Tokens,
}

impl History {
    /// The history under `[analysis].history_token_budget`.
    #[must_use]
    pub fn from_config(config: &Config, db: Arc<Writer>) -> Self {
        Self {
            db,
            budget: config.analysis.history_token_budget,
        }
    }

    /// The messages to replay for `session`, oldest first, within the
    /// window. No model is called.
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
        window.apply(messages).map_err(failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::agent::AgentResponse;
    use crate::embedding::Dimension;
    use crate::storage::sessions::{self, ChatMode, SpokenText};
    use crate::storage::workspace::WorkspaceDb;
    use jiff::Timestamp;

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

    #[tokio::test(flavor = "multi_thread")]
    async fn the_window_alone_is_replayed() {
        let (db, id) = session().await;
        let history = History {
            db: Arc::clone(&db),
            budget: Tokens::new(10),
        }
        .load(&id)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
        let shown: Vec<String> = history.iter().map(SpokenText::spoken_text).collect();
        assert_eq!(shown, ["question 4", "answer 4", "question 5", "answer 5"]);
    }
}
