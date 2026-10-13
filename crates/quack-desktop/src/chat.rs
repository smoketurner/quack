//! Sessions and turns (design 3.3). A turn runs as a job in the session's
//! serial lane, exactly as the terminal submits one, and its events reach
//! the frontend over a `Channel` as [`TurnEvent`]s; a write that asks is
//! answered in the native dialog (`permission`).

use std::sync::Arc;

use quack_core::analysis::agent::AgentResponseBody;
use quack_core::analysis::events::{self, AgentEvent, ToolName, ToolStep};
use quack_core::analysis::policy::{Hold, WritePolicy};
use quack_core::classify::LabelJobs;
use quack_core::ids::{SessionId, WorkspaceId};
use quack_core::jobs::{JobInfo, JobKind, JobSpec, Lane, LaneKey};
use quack_core::llm::TurnRequest;
use quack_core::llm::egress::Egress;
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use quack_core::storage::sessions::{self, ChatMode, MessageRow, SessionRow};
use quack_core::text::OneLine;
use serde::Serialize;
use tauri::ipc::Channel;
use tauri::{AppHandle, Manager, State};

use crate::app::{AppState, OpenWorkspace};
use crate::error::{CommandError, CommandResult, ErrorKind, kind_of};
use crate::permission;

/// Sessions a listing shows unless the caller asks for fewer or more.
const SESSIONS_SHOWN: u32 = 50;

/// What a turn sends the frontend, in order: the agent's events as the
/// transcript renders them, and the write prompts' two moments (asked,
/// answered) around the native dialog.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum TurnEvent {
    /// What the turn waits on before any output (a model loading).
    Status {
        text: String,
    },
    /// The model began reasoning before its next output.
    Reasoning,
    /// A piece of the answer, as it streams.
    Text {
        delta: String,
    },
    ToolStarted {
        tool: ToolName,
        detail: String,
    },
    ToolFinished {
        step: ToolStep,
    },
    /// A write waits on the dialog: the transcript shows the statement under
    /// "Answer in the dialog".
    PermissionRequired {
        sql: String,
        reason: Hold,
        notice: Option<&'static str>,
        heading: &'static str,
    },
    /// The dialog was answered: `Decision::reply`, or that the turn had
    /// stopped waiting.
    PermissionAnswered {
        decision: &'static str,
        reply: &'static str,
    },
    /// The turn finished; the same body every interface returns.
    Complete {
        response: AgentResponseBody,
    },
    /// The turn failed after possibly sending some of the above.
    Failed {
        message: String,
        kind: ErrorKind,
    },
}

impl TurnEvent {
    /// The frontend's form of an agent event other than a permission
    /// request, which the forwarder answers before it reports.
    fn of(event: AgentEvent, session: &SessionId) -> Option<Self> {
        Some(match event {
            AgentEvent::Status(text) => Self::Status { text },
            AgentEvent::Reasoning => Self::Reasoning,
            AgentEvent::TextDelta(delta) => Self::Text { delta },
            AgentEvent::ToolStarted { tool, detail } => Self::ToolStarted { tool, detail },
            AgentEvent::ToolFinished(step) => Self::ToolFinished { step },
            AgentEvent::TurnComplete(response) => Self::Complete {
                response: response.body(session),
            },
            AgentEvent::Failed(failure) => Self::Failed {
                message: failure.message,
                kind: kind_of(failure.kind),
            },
            AgentEvent::PermissionRequired(_) => return None,
        })
    }
}

/// The sessions of `workspace`, newest first.
#[tauri::command]
pub(crate) async fn list_sessions(
    state: State<'_, AppState>,
    workspace: WorkspaceId,
    limit: Option<u32>,
) -> CommandResult<Vec<SessionRow>> {
    let ws = state.workspace(&workspace).await?;
    let limit = limit.unwrap_or(SESSIONS_SHOWN).max(1);
    let rows = ws
        .reader
        .with_db(move |db| sessions::list_sessions(db, limit))
        .await?;
    state
        .auditor
        .record(
            &ws,
            AuditAction::List,
            None,
            Outcome::Allowed,
            Some(serde_json::json!({ "what": "sessions", "count": rows.len() })),
        )
        .await?;
    Ok(rows)
}

/// A new session in `mode` (chat unless given), on the configured model.
#[tauri::command]
pub(crate) async fn new_session(
    state: State<'_, AppState>,
    workspace: WorkspaceId,
    mode: Option<ChatMode>,
) -> CommandResult<SessionRow> {
    let ws = state.workspace(&workspace).await?;
    let model = state.config.chat_model_label();
    let mode = mode.unwrap_or_default();
    let row = ws
        .db
        .run(move |db| sessions::create_session(db, &model, mode, None))
        .await?;
    state
        .auditor
        .record(
            &ws,
            AuditAction::Session,
            Some(ResourceKind::Session.id(&row.id)),
            Outcome::Allowed,
            Some(serde_json::json!({ "mode": mode })),
        )
        .await?;
    Ok(row)
}

/// Every message of `session`, in order: the transcript.
#[tauri::command]
pub(crate) async fn session_messages(
    state: State<'_, AppState>,
    workspace: WorkspaceId,
    session: SessionId,
) -> CommandResult<Vec<MessageRow>> {
    let ws = state.workspace(&workspace).await?;
    let id = session.clone();
    let rows = ws
        .reader
        .with_db(move |db| {
            sessions::get_session(db, &id)?
                .ok_or_else(|| ResourceKind::Session.missing(id.as_str()))?;
            sessions::messages(db, &id)
        })
        .await?;
    state
        .auditor
        .record(
            &ws,
            AuditAction::SessionRead,
            Some(ResourceKind::Session.id(&session)),
            Outcome::Allowed,
            Some(serde_json::json!({ "messages": rows.len() })),
        )
        .await?;
    Ok(rows)
}

/// Set `session`'s mode for its next questions.
#[tauri::command]
pub(crate) async fn set_session_mode(
    state: State<'_, AppState>,
    workspace: WorkspaceId,
    session: SessionId,
    mode: ChatMode,
) -> CommandResult<()> {
    let ws = state.workspace(&workspace).await?;
    let id = session.clone();
    ws.db
        .run(move |db| sessions::set_session_mode(db, &id, mode))
        .await?;
    state
        .auditor
        .record(
            &ws,
            AuditAction::Mode,
            Some(ResourceKind::Session.id(&session)),
            Outcome::Allowed,
            Some(serde_json::json!({ "mode": mode })),
        )
        .await?;
    Ok(())
}

/// "Run changes without asking" for `workspace`: on, the agent's writes
/// run unasked until the turn has read document text; off again at launch.
#[tauri::command]
#[expect(
    clippy::needless_pass_by_value,
    reason = "a command's arguments arrive by value from the frontend"
)]
pub(crate) fn set_allow_writes(state: State<'_, AppState>, workspace: WorkspaceId, allow: bool) {
    state.set_allow_writes(&workspace, allow);
}

/// Ask `message` in `session`, limited to `documents` when given (each by
/// id, id prefix, file name, or title). The turn's events arrive on
/// `on_event`; the job is returned at once.
#[tauri::command]
pub(crate) async fn start_turn(
    app: AppHandle,
    state: State<'_, AppState>,
    workspace: WorkspaceId,
    session: SessionId,
    message: String,
    documents: Vec<String>,
    on_event: Channel<TurnEvent>,
) -> CommandResult<JobInfo> {
    if message.trim().is_empty() {
        return Err(CommandError::invalid("the question is empty"));
    }
    let ws = state.workspace(&workspace).await?;
    let id = session.clone();
    ws.reader
        .with_db(move |db| {
            sessions::get_session(db, &id)?
                .ok_or_else(|| ResourceKind::Session.missing(id.as_str()))
        })
        .await?;
    state
        .auditor
        .record(
            &ws,
            AuditAction::Query,
            Some(ResourceKind::Session.id(&session)),
            Outcome::Allowed,
            Some(serde_json::json!({ "prompt": message, "documents": documents })),
        )
        .await?;
    let (sink, mut events) = events::channel();
    let policy = WritePolicy::Ask.allowed_if(state.allows_writes(&workspace));
    let job = submit(&state, &ws, &session, message, documents, policy, sink);
    state.turn_started(&session, job.id);
    let (job_id, ended_session) = (job.id, session.clone());
    tauri::async_runtime::spawn(async move {
        while let Some(event) = events.recv().await {
            let sent = match event {
                AgentEvent::PermissionRequired(request) => {
                    let asked = TurnEvent::PermissionRequired {
                        sql: request.sql.clone(),
                        reason: request.hold,
                        notice: request.hold.notice(),
                        heading: events::Decision::HEADING,
                    };
                    if on_event.send(asked).is_err() {
                        // Nobody is watching; the dialog still decides.
                        tracing::debug!("the turn's channel is gone before a write prompt");
                    }
                    let answer = permission::ask(&app, request).await;
                    on_event.send(TurnEvent::PermissionAnswered {
                        decision: answer.decision.as_str(),
                        reply: answer.reply(),
                    })
                }
                other => match TurnEvent::of(other, &session) {
                    Some(event) => on_event.send(event),
                    None => Ok(()),
                },
            };
            if let Err(e) = sent {
                tracing::debug!(error = %e, "a turn event could not reach the frontend");
            }
        }
        app.state::<AppState>().turn_ended(&ended_session, job_id);
    });
    Ok(job)
}

/// The turn as a job in the session's serial lane, under the workspace's
/// provider allow-list.
fn submit(
    state: &AppState,
    ws: &Arc<OpenWorkspace>,
    session: &SessionId,
    message: String,
    documents: Vec<String>,
    policy: WritePolicy,
    sink: events::EventSink,
) -> JobInfo {
    let spec = JobSpec::new(JobKind::Chat, OneLine(&message).to_string())
        .workspace(ws.row.id.clone())
        .lane(Lane::serial(&LaneKey::Session(session.clone())));
    let labelling = LabelJobs::new(state.jobs.clone()).workspace(ws.row.id.clone());
    let (config, db, reader_db, egress) = (
        state.config.clone(),
        Arc::clone(&ws.db),
        ws.reader.clone(),
        ws.egress(),
    );
    let session_id = session.clone();
    state.jobs.submit(spec, move |ctx| {
        Egress::scope(Some(egress), async move {
            let request = TurnRequest {
                db,
                reader_db,
                session_id: &session_id,
                policy,
                message: &message,
                documents: &documents,
                user: None,
                labelling,
                sink,
                cancel: ctx.cancel_token(),
            };
            // The turn emits TurnComplete or Failed itself; the job keeps
            // its one-line outcome.
            match request.run(&config).await {
                Ok(response) if response.cancelled => Err(String::from("cancelled")),
                Ok(response) => Ok(format!(
                    "answered: {} steps, {} sources",
                    response.steps.len(),
                    response.citations.len()
                )),
                Err(e) => Err(e.to_string()),
            }
        })
    })
}

/// Stop `session`'s running turn. `false` when none runs.
#[tauri::command]
pub(crate) async fn cancel_turn(
    state: State<'_, AppState>,
    workspace: WorkspaceId,
    session: SessionId,
) -> CommandResult<bool> {
    let ws = state.workspace(&workspace).await?;
    let Some(job) = state.running_turn(&session) else {
        return Ok(false);
    };
    let cancelled = state.jobs.cancel(job);
    state
        .auditor
        .record(
            &ws,
            AuditAction::Cancel,
            Some(ResourceKind::Session.id(&session)),
            Outcome::Allowed,
            Some(serde_json::json!({ "job": job, "cancelled": cancelled })),
        )
        .await?;
    Ok(cancelled)
}
