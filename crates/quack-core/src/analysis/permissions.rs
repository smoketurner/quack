//! Writes a turn waits on a person to decide, held for an interface that
//! answers later and elsewhere: the web chat's `POST
//! .../sessions/{sid}/permissions/{request}`. The turn holds its
//! `PermissionRequest` open in memory; the person answers it by its id;
//! nobody answering within the timeout refuses the write. A restart ends
//! the waiting turn with it. (The terminal needs none of this for its own
//! turns: its prompt is on the same screen as the turn. A turn another
//! process asks of it over the workspace socket reaches that prompt
//! through [`Approvals`].)

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use jiff::{SignedDuration, Timestamp};
use serde_json::Value;

use tokio::sync::mpsc;

use crate::analysis::events::{Decision, Delivery, PermissionRequest};
use crate::ids::{PermissionId, SessionId, UserId, WorkspaceId};
use crate::jobs::JobId;

/// A write a turn run for another process (`quack mcp` over the workspace
/// socket) asks the person at this process's keyboard to decide: the
/// terminal session listens and puts it in its own `y`/`n`/`a` prompt.
/// Where nobody listens, the write is refused at once.
#[derive(Clone, Default)]
pub struct Approvals(Arc<Mutex<Option<mpsc::UnboundedSender<Asked>>>>);

/// One write waiting for the person: the job running it (a turn, or a
/// statement another process sent over the workspace socket), the turn's
/// session when it is a turn, and the request to answer.
#[derive(Debug)]
pub struct Asked {
    pub job: JobId,
    pub session: Option<SessionId>,
    pub request: PermissionRequest,
}

impl Approvals {
    /// Nobody listens yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Take the requests from here on: the one interface with a person at
    /// it. A later call replaces an earlier listener.
    #[must_use]
    pub fn listen(&self) -> mpsc::UnboundedReceiver<Asked> {
        let (sender, receiver) = mpsc::unbounded_channel();
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(sender);
        receiver
    }

    /// Hand `asked` to the listener; with none, or one that has gone, the
    /// write is refused.
    pub fn ask(&self, asked: Asked) {
        let sender = self
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let Some(sender) = sender else {
            asked.request.deny();
            return;
        };
        if let Err(mpsc::error::SendError(asked)) = sender.send(asked) {
            asked.request.deny();
        }
    }
}

/// The writes waiting for a decision, by request.
#[derive(Clone, Default)]
pub struct Permissions(Arc<Mutex<HashMap<PermissionId, Waiting>>>);

/// Whose turn asked: only that person, in that session of that workspace,
/// may answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asker {
    pub workspace: WorkspaceId,
    pub session: SessionId,
    pub user: UserId,
}

/// One write: whose turn asked, the statement, and where it stands.
struct Waiting {
    asker: Asker,
    sql: String,
    state: State,
}

enum State {
    Open(PermissionRequest),
    /// Answered; kept until it expires so a second answer is a conflict,
    /// not an unknown request.
    Decided,
}

/// A write now waiting, as the interface announces it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Held {
    pub request: PermissionId,
    pub expires_at: Timestamp,
}

/// A write nobody answered in time: refused, and handed to the caller to
/// audit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expired {
    pub request: PermissionId,
    pub sql: String,
}

impl Expired {
    /// The audit detail: the statement and that the time ran out.
    #[must_use]
    pub fn detail(&self) -> Value {
        serde_json::json!({
            "request": self.request, "sql": self.sql, "decision": "expired",
        })
    }
}

/// Why an answer was not taken.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// No such request in this session, or it expired.
    Unknown,
    /// Someone already answered it.
    Decided,
    /// Only the person whose turn asked may answer.
    NotYours,
    /// The turn that asked had stopped waiting before the answer came
    /// (its stream disconnected, or it was cancelled), so the statement
    /// did not run whatever the answer was.
    Gone { sql: String },
}

impl Refusal {
    /// The audit detail: the statement and the answer that reached no
    /// turn, recorded like an expiry; the other refusals name nothing.
    #[must_use]
    pub fn detail(&self, request: &PermissionId, answer: Decision) -> Option<Value> {
        match self {
            Self::Gone { sql } => Some(serde_json::json!({
                "request": request, "sql": sql, "decision": "gone", "answer": answer.as_str(),
            })),
            Self::Unknown | Self::Decided | Self::NotYours => None,
        }
    }

    /// What every interface says for this refusal.
    #[must_use]
    pub fn message(&self) -> &'static str {
        match self {
            Self::Unknown => "no write is waiting under that request",
            Self::Decided => "that write was already decided",
            Self::NotYours => "only the person whose question asked for the write may decide it",
            Self::Gone { .. } => Delivery::TURN_GONE,
        }
    }
}

impl Permissions {
    /// Hold `request` for `asker`'s turn. When nobody has answered after
    /// `timeout`, the write is refused and `on_expired` is called with it,
    /// to audit the expiry.
    pub fn hold<F, Fut>(
        &self,
        asker: Asker,
        request: PermissionRequest,
        timeout: Duration,
        on_expired: F,
    ) -> Held
    where
        F: FnOnce(Expired) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let id = PermissionId::generate();
        let waiting = Waiting {
            asker,
            sql: request.sql.clone(),
            state: State::Open(request),
        };
        self.lock().insert(id.clone(), waiting);
        let expires_at = Timestamp::now()
            .checked_add(SignedDuration::try_from(timeout).unwrap_or(SignedDuration::MAX))
            .unwrap_or(Timestamp::MAX);
        let (permissions, expired) = (self.clone(), id.clone());
        tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            if let Some(sql) = permissions.expire(&expired) {
                on_expired(Expired {
                    request: expired,
                    sql,
                })
                .await;
            }
        });
        Held {
            request: id,
            expires_at,
        }
    }

    /// Give `answer` to the write `request` waits on, for `asker`. Returns
    /// the statement once the turn has the answer.
    ///
    /// # Errors
    ///
    /// A [`Refusal`] when no such write waits in that session, it was
    /// already decided, it belongs to another person's turn, or the turn
    /// that asked had stopped waiting (`Gone`: the answer is recorded,
    /// nothing ran).
    pub fn decide(
        &self,
        request: &PermissionId,
        asker: &Asker,
        answer: Decision,
    ) -> Result<String, Refusal> {
        let mut waiting = self.lock();
        let entry = waiting
            .get_mut(request)
            .filter(|w| w.asker.workspace == asker.workspace && w.asker.session == asker.session)
            .ok_or(Refusal::Unknown)?;
        if entry.asker.user != asker.user {
            return Err(Refusal::NotYours);
        }
        match std::mem::replace(&mut entry.state, State::Decided) {
            // A refusal needs no turn to take it: an unanswered write is
            // refused either way.
            State::Open(open) => match (answer, open.answer(answer)) {
                (Decision::Deny, _) | (_, Delivery::Delivered) => Ok(entry.sql.clone()),
                (Decision::Allow | Decision::AllowTurn, Delivery::TurnGone) => Err(Refusal::Gone {
                    sql: entry.sql.clone(),
                }),
            },
            State::Decided => Err(Refusal::Decided),
        }
    }

    /// Forget `request` once its time is up; when still open, refuse the
    /// write and return the statement.
    fn expire(&self, request: &PermissionId) -> Option<String> {
        let waiting = self.lock().remove(request)?;
        match waiting.state {
            State::Open(open) => {
                open.deny();
                Some(waiting.sql)
            }
            State::Decided => None,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<PermissionId, Waiting>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
