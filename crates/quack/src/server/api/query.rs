//! The agent turn (`query`, and its SSE form), direct SQL, and hybrid
//! search without the model.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use quack_core::analysis::agent::AgentResponse;
use quack_core::analysis::events::{self, AgentEvent};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::embedding::{Input, Vector};
use quack_core::error::Record;
use quack_core::ids::WorkspaceId;
use quack_core::jobs::{JobId, JobKind, JobQueue, JobSpec, Lane, LaneKey};
use quack_core::llm::{self, Embeddings};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use quack_core::storage::sessions::{self, ChatMode};
use quack_core::storage::workspace::{
    ChunkScope, HybridLimits, TEMP_OBJECT_REFUSED, creates_temp_object,
};
use serde::{Deserialize, Serialize};

use super::StreamEvent;
use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::{App, with_db};
use crate::server::web::markdown::to_html;

#[derive(Deserialize)]
pub(crate) struct QueryRequest {
    pub prompt: String,
    pub session_id: Option<String>,
    pub mode: Option<ChatMode>,
    #[serde(default)]
    pub allow_write: bool,
}

/// A turn that has passed its checks and has its session: ready to run.
struct PreparedTurn {
    access: Access,
    db: SharedDb,
    reader: ReaderDb,
    session_id: String,
    policy: WritePolicy,
    prompt: String,
}

impl PreparedTurn {
    /// Everything a turn needs before the model is called: the
    /// authorization, the provider check, and the session (existing or new).
    async fn prepare(
        app: &App,
        identity: Identity,
        workspace_id: &WorkspaceId,
        body: &QueryRequest,
    ) -> ApiResult<Self> {
        let access = Access::resolve(app, identity, workspace_id, Need::READ).await?;
        if body.allow_write && !access.permits(Need::WRITE) {
            access
                .audit(app, AuditAction::Query, None, Outcome::Denied, None)
                .await?;
            return Err(ApiError::forbidden(
                "allow_write needs the member role and the write scope",
            ));
        }
        if body.prompt.trim().is_empty() {
            return Err(ApiError::bad_request("prompt must not be empty"));
        }
        let chat = app.config.chat_model_ref()?;
        if !access
            .workspace
            .allowed_providers
            .permits(chat.provider_name.as_str())
        {
            return Err(ApiError::forbidden(format!(
                "provider '{}' is not allowed in this workspace",
                chat.provider_name
            )));
        }
        let mode = body.mode;
        let db = app.workspace_db(workspace_id).await?;
        let reader = app.reader_db(workspace_id).await?;
        let requested = body.session_id.clone();
        let model = chat.to_string();
        let user = access.identity.user_id.clone();
        let viewer = access.session_viewer();
        let session_id = with_db(Arc::clone(&db), move |db| {
            if let Some(id) = requested {
                // A session the caller may not see reads as missing, not forbidden.
                sessions::get_session(db, &id)?
                    .filter(|s| s.visible_to(&viewer))
                    .ok_or_else(|| Record::Session.missing(id.as_str()))?;
                // A session's mode is set when it is created; `mode` on a
                // later turn is ignored, and PATCH .../sessions/{sid} changes
                // it explicitly (issue #57).
                Ok(id)
            } else {
                Ok(sessions::create_session(db, &model, mode.unwrap_or_default(), Some(&user))?.id)
            }
        })
        .await?;
        let policy = if body.allow_write {
            WritePolicy::Allow
        } else {
            WritePolicy::Deny
        };
        Ok(Self {
            access,
            db,
            reader,
            session_id,
            policy,
            prompt: body.prompt.clone(),
        })
    }

    /// Submit the turn as a job in its session's lane (one turn at a time
    /// per session, in order). When an earlier turn of the session is still
    /// going, the events open with a `status` saying so.
    fn start(self, app: &App) -> Turn {
        let Self {
            access,
            db,
            reader,
            session_id,
            policy,
            prompt,
        } = self;
        let (sink, events) = events::channel();
        let lane = LaneKey::Session(session_id.clone());
        if app.jobs.lane_active(&lane) > 0 {
            drop(sink.send(AgentEvent::Status(String::from(
                "Queued: this runs when the session's previous question has been answered.",
            ))));
        }
        let config = app.config.clone();
        let (session, text) = (session_id.clone(), prompt.clone());
        let spec = JobSpec::new(JobKind::Chat, prompt.chars().take(80).collect::<String>())
            .workspace(access.workspace.id.clone())
            .owner(Some(access.identity.user_id.clone()))
            .lane(Lane::serial(&lane));
        let job = app
            .jobs
            .submit(spec, move |ctx| async move {
                // The turn emits TurnComplete or Failed itself.
                match (llm::TurnRequest {
                    db,
                    reader_db: reader,
                    session_id: &session,
                    policy,
                    message: &text,
                    sink,
                    cancel: ctx.cancel_token(),
                })
                .run(&config)
                .await
                {
                    Ok(response) if response.cancelled => Err(String::from("cancelled")),
                    Ok(response) => Ok(format!(
                        "answered: {} steps, {} sources",
                        response.steps.len(),
                        response.citations.len()
                    )),
                    Err(e) => Err(e.to_string()),
                }
            })
            .id;
        Turn {
            events,
            guard: CancelOnDrop {
                jobs: app.jobs.clone(),
                job,
            },
            access,
            session_id,
            prompt,
        }
    }
}

/// Cancels the turn's job when dropped: the SSE stream holds one, so a
/// client that goes away stops the model (or takes a queued turn out of
/// the queue) instead of leaving the turn to finish unwatched (issue #45).
struct CancelOnDrop {
    jobs: JobQueue,
    job: JobId,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.jobs.cancel(self.job);
    }
}

/// A running turn: its events, the guard that cancels it, and what its end
/// is recorded against.
struct Turn {
    events: events::EventStream,
    /// Held for its drop, which cancels the turn.
    #[expect(dead_code, reason = "held only so dropping the turn cancels its job")]
    guard: CancelOnDrop,
    access: Access,
    session_id: String,
    prompt: String,
}

/// How a turn ended, for its audit row.
enum TurnEnd<'a> {
    Answered(&'a AgentResponse),
    Failed,
}

impl Turn {
    /// Audit the turn. A failed turn that left the session without any
    /// message (a fresh session whose first turn never reached the model)
    /// is removed, as print mode does, so the session list shows no empty
    /// entries.
    async fn record(&self, app: &App, end: TurnEnd<'_>) {
        let (outcome, steps) = match end {
            TurnEnd::Answered(response) => (Outcome::Allowed, response.steps.clone()),
            TurnEnd::Failed => (Outcome::Error, Vec::new()),
        };
        if outcome == Outcome::Error
            && let Ok(db) = app.workspace_db(&self.access.workspace.id).await
        {
            let sid = self.session_id.clone();
            if let Err(e) = with_db(db, move |db| sessions::delete_if_empty(db, &sid)).await {
                tracing::warn!(error = %e.message, "could not remove the empty session");
            }
        }
        let detail = serde_json::json!({ "prompt": self.prompt, "steps": steps });
        if let Err(e) = self
            .access
            .audit(
                app,
                AuditAction::Query,
                Some(ResourceKind::Session.id(&self.session_id)),
                outcome,
                Some(detail),
            )
            .await
        {
            tracing::error!(error = %e.message, "audit write failed");
        }
    }
}

pub(crate) async fn query(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<QueryRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    // A client that disconnects drops this future, and the turn's guard
    // with it, which cancels the turn.
    let mut turn = PreparedTurn::prepare(&app, identity, &id, &body)
        .await?
        .start(&app);
    let mut failure = None;
    let mut complete = None;
    while let Some(event) = turn.events.recv().await {
        match event {
            AgentEvent::PermissionRequired(request) => request.deny(),
            AgentEvent::TurnComplete(response) => complete = Some(response),
            AgentEvent::Failed(message) => failure = Some(message),
            AgentEvent::Status(_)
            | AgentEvent::TextDelta(_)
            | AgentEvent::ToolStarted { .. }
            | AgentEvent::ToolFinished(_) => {}
        }
    }
    if let Some(response) = complete {
        turn.record(&app, TurnEnd::Answered(&response)).await;
        return Ok(Json(response.to_json(&turn.session_id)));
    }
    turn.record(&app, TurnEnd::Failed).await;
    Err(failure.map_or_else(
        || ApiError::internal("the turn ended without an answer"),
        ApiError::from,
    ))
}

/// The same turn as SSE: `text`, `tool_started`, `tool_finished`, then
/// `complete` with the full response object, or `error`.
pub(crate) async fn stream(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<QueryRequest>,
) -> ApiResult<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
    let turn = PreparedTurn::prepare(&app, identity, &id, &body)
        .await?
        .start(&app);
    let stream = futures::stream::unfold((turn, app), |(mut turn, app)| async move {
        let event = turn.events.recv().await?;
        let out = match event {
            AgentEvent::Status(status) => StreamEvent::Status.event().data(status),
            AgentEvent::TextDelta(text) => StreamEvent::Text.event().data(text),
            AgentEvent::ToolStarted { tool, detail } => StreamEvent::ToolStarted
                .event()
                .json_data(serde_json::json!({ "tool": tool, "detail": detail }))
                .unwrap_or_default(),
            AgentEvent::ToolFinished(step) => StreamEvent::ToolFinished
                .event()
                .json_data(&step)
                .unwrap_or_default(),
            AgentEvent::PermissionRequired(request) => {
                request.deny();
                StreamEvent::WriteRefused
                    .event()
                    .data("writes are off for this request")
            }
            AgentEvent::TurnComplete(response) => {
                turn.record(&app, TurnEnd::Answered(&response)).await;
                // The web page swaps this in for the streamed plain text.
                let mut payload = response.to_json(&turn.session_id);
                if let serde_json::Value::Object(map) = &mut payload {
                    map.insert(
                        String::from("answer_html"),
                        serde_json::Value::String(to_html(&response.content)),
                    );
                }
                StreamEvent::Complete
                    .event()
                    .json_data(payload)
                    .unwrap_or_default()
            }
            AgentEvent::Failed(failure) => {
                turn.record(&app, TurnEnd::Failed).await;
                StreamEvent::Error.event().data(failure.message)
            }
        };
        Some((Ok(out), (turn, app)))
    });
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

#[derive(Deserialize)]
pub(crate) struct SqlRequest {
    pub sql: String,
}

/// Direct SQL. Reads need the viewer role; anything that mutates needs the
/// member role and the write scope. `_quack_` tables are never reachable.
/// A capped result set for the API and the web grid.
#[derive(Debug, Serialize)]
pub(crate) struct SqlOutcome {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
    pub row_count: usize,
    pub truncated: bool,
}

/// Direct SQL. Reads need the viewer role; anything that mutates needs the
/// member role and the write scope. `_quack_` tables are never reachable.
pub(crate) async fn sql(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<SqlRequest>,
) -> ApiResult<Json<SqlOutcome>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    Ok(Json(access.execute_sql(&app, &body.sql).await?))
}

impl Access {
    /// Classify, authorize, run, and audit one statement: the SQL the API
    /// and the web grid share.
    pub(crate) async fn execute_sql(&self, app: &App, statement: &str) -> ApiResult<SqlOutcome> {
        let access = self;
        let sql = statement.to_owned();
        // Classifying parses the statement: a read, never in the writer's line.
        let kind = app
            .read(&access.workspace.id, move |db| {
                db.classify_user_statement(&sql)
            })
            .await
            .map_err(|e| ApiError::forbidden(e.message))?;
        let is_write = kind.writes().map_err(ApiError::bad_request)?;
        let detail = serde_json::json!({ "sql": statement });
        if is_write && creates_temp_object(statement) {
            access
                .audit(app, AuditAction::Sql, None, Outcome::Denied, Some(detail))
                .await?;
            return Err(ApiError::bad_request(TEMP_OBJECT_REFUSED));
        }
        if is_write && !access.permits(Need::WRITE) {
            access
                .audit(app, AuditAction::Sql, None, Outcome::Denied, Some(detail))
                .await?;
            return Err(ApiError::forbidden(
                "writes need the member role and the write scope",
            ));
        }
        let reader_db = app.reader_db(&access.workspace.id).await?;
        let sql = statement.to_owned();
        let max_rows = app.config.analysis.max_query_rows;
        let result = if is_write {
            let db = app.workspace_db(&access.workspace.id).await?;
            let result = with_db(db, move |db| db.execute_query_capped(&sql, max_rows)).await;
            // Whatever ran might have created a temp object the check above
            // did not catch (a leading comment, a multi-statement batch);
            // check the writer's catalog regardless of whether the statement
            // itself errored, since an earlier statement in a batch can have
            // already run.
            reader_db.observe_write().await;
            result
        } else {
            // A read never queues behind a write: run it on the workspace's
            // reader connection instead of the writer.
            reader_db
                .with_db(move |db| db.execute_query_capped(&sql, max_rows))
                .await
                .map_err(ApiError::from)
        };
        let outcome = Outcome::of(&result);
        access
            .audit(app, AuditAction::Sql, None, outcome, Some(detail))
            .await?;
        let capped = result.map_err(|e| ApiError::unprocessable(e.message))?;
        Ok(SqlOutcome {
            truncated: capped.truncated(),
            columns: capped.results.columns,
            rows: capped.results.rows,
            row_count: capped.total_rows,
        })
    }
}

#[derive(Deserialize)]
pub(crate) struct SearchQuery {
    pub query: String,
    pub top_k: Option<u32>,
}

/// Hybrid retrieval with no model call: the embedding provider when one is
/// configured, else keyword search alone.
pub(crate) async fn search(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Query(q): Query<SearchQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let query = q.query.trim().to_owned();
    if query.is_empty() {
        return Err(ApiError::bad_request("query must not be empty"));
    }
    let top_k = q.top_k.unwrap_or(app.config.retrieval.top_k).clamp(1, 100);
    let rrf_k = app.config.retrieval.rrf_k;
    let embedding: Option<Vector> = match Embeddings::from_config(&app.config).await? {
        Some(model) => Some(
            model
                .embed_interactive(&Input::Query(query.clone()))
                .await?,
        ),
        None => None,
    };
    let reader_db = app.reader_db(&id).await?;
    let text = query.clone();
    let hits = reader_db
        .with_db(move |db| {
            let scope = ChunkScope::all();
            match embedding.as_deref() {
                Some(vector) => {
                    db.search_hybrid_chunks(&text, vector, HybridLimits { top_k, rrf_k }, &scope)
                }
                None => db.search_keyword_chunks(&text, top_k, &scope),
            }
        })
        .await
        .map_err(ApiError::from)?;
    access
        .audit(
            &app,
            AuditAction::Search,
            None,
            Outcome::Allowed,
            Some(serde_json::json!({ "q": query })),
        )
        .await?;
    Ok(Json(serde_json::json!({ "chunks": hits })))
}
