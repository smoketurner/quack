//! The history a turn replays: the session's turns
//! ([`sessions::session_turns`]) under the token window
//! ([`TranscriptWindow`]). The turns the window leaves out are not replayed;
//! nothing stands in for them.

use std::sync::Arc;

use rig::message::Message;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::ids::SessionId;
use crate::storage::sessions::{self, TranscriptWindow};
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
        let session = session.clone();
        let messages = self
            .db
            .run(move |db| sessions::session_turns(db, &session))
            .await
            .map_err(|e| Error::Analysis(format!("could not load the session history: {e}")))?;
        Ok(TranscriptWindow::new(self.budget).apply(messages))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::agent::AgentResponse;
    use crate::embedding::Dimension;
    use crate::storage::sessions::{ChatMode, SpokenText};
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
