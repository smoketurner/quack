//! Writes a streamed turn waits on a person to decide. The turn holds its
//! `PermissionRequest` open in memory; the person answers through `POST
//! .../sessions/{sid}/permissions/{request}`; nobody answering within
//! `[server].permission_timeout_seconds` refuses the write. A server
//! restart ends the waiting turn with it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use jiff::{SignedDuration, Timestamp};
use quack_core::analysis::events::{Decision, Delivery, PermissionRequest};
use quack_core::ids::{PermissionId, SessionId, UserId, WorkspaceId};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};

use super::auth::Access;
use super::error::ApiError;
use super::state::App;

/// The writes waiting for a decision, by request.
#[derive(Clone, Default)]
pub(crate) struct Permissions(Arc<Mutex<HashMap<PermissionId, Waiting>>>);

/// One write: whose turn asked, the statement, and where it stands.
struct Waiting {
    workspace: WorkspaceId,
    session: SessionId,
    user: UserId,
    sql: String,
    state: State,
}

enum State {
    Open(PermissionRequest),
    /// Answered; kept until it expires so a second answer is a conflict,
    /// not an unknown request.
    Decided,
}

/// A write now waiting, as the stream announces it.
pub(crate) struct Held {
    pub request: PermissionId,
    pub expires_at: Timestamp,
}

/// Why an answer was not taken.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
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
    pub(crate) fn detail(
        &self,
        request: &PermissionId,
        answer: Decision,
    ) -> Option<serde_json::Value> {
        match self {
            Self::Gone { sql } => Some(serde_json::json!({
                "request": request, "sql": sql, "decision": "gone", "answer": answer.as_str(),
            })),
            Self::Unknown | Self::Decided | Self::NotYours => None,
        }
    }
}

impl From<Refusal> for ApiError {
    fn from(refusal: Refusal) -> Self {
        match refusal {
            Refusal::Unknown => Self::not_found("no write is waiting under that request"),
            Refusal::Decided => Self::conflict("that write was already decided"),
            Refusal::NotYours => {
                Self::forbidden("only the person whose question asked for the write may decide it")
            }
            Refusal::Gone { .. } => {
                Self::gone("the question that asked for this write has already ended; nothing ran")
            }
        }
    }
}

impl Permissions {
    /// Hold `request` for the turn `access` started in `session`. When
    /// nobody has answered after `[server].permission_timeout_seconds`, the
    /// write is refused and that is audited.
    pub(crate) fn hold(
        &self,
        app: &App,
        access: &Access,
        session: &SessionId,
        request: PermissionRequest,
    ) -> Held {
        let timeout = app.config.server.permission_timeout();
        let id = PermissionId::generate();
        let waiting = Waiting {
            workspace: access.membership.workspace.id.clone(),
            session: session.clone(),
            user: access.identity.user_id.clone(),
            sql: request.sql.clone(),
            state: State::Open(request),
        };
        self.lock().insert(id.clone(), waiting);
        let expires_at = Timestamp::now()
            .checked_add(SignedDuration::try_from(timeout).unwrap_or(SignedDuration::MAX))
            .unwrap_or(Timestamp::MAX);
        let (permissions, app, access, session, expired) = (
            self.clone(),
            Arc::clone(app),
            access.clone(),
            session.clone(),
            id.clone(),
        );
        tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            if let Some(sql) = permissions.expire(&expired) {
                let detail = serde_json::json!({
                    "request": expired, "sql": sql, "decision": "expired",
                });
                if let Err(e) = access
                    .audit(
                        &app,
                        AuditAction::Permission,
                        Some(ResourceKind::Session.id(&session)),
                        Outcome::Denied,
                        Some(detail),
                    )
                    .await
                {
                    tracing::error!(error = %e.message, "audit write failed");
                }
            }
        });
        Held {
            request: id,
            expires_at,
        }
    }

    /// Give `answer` to the write `request` waits on in `session`, for the
    /// person `access` names. Returns the statement once the turn has the
    /// answer.
    ///
    /// # Errors
    ///
    /// A [`Refusal`] when no such write waits there, it was already
    /// decided, it belongs to another person's turn, or the turn that
    /// asked had stopped waiting (`Gone`: the answer is recorded, nothing
    /// ran).
    pub(crate) fn decide(
        &self,
        request: &PermissionId,
        access: &Access,
        session: &SessionId,
        answer: Decision,
    ) -> Result<String, Refusal> {
        let mut waiting = self.lock();
        let entry = waiting
            .get_mut(request)
            .filter(|w| w.workspace == access.membership.workspace.id && w.session == *session)
            .ok_or(Refusal::Unknown)?;
        if entry.user != access.identity.user_id {
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
