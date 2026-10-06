//! The agent turn (`query`, and its SSE form), direct SQL, and hybrid
//! search without the model.

use std::convert::Infallible;
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::Stream;
use quack_core::analysis::agent::AgentResponse;
use quack_core::analysis::events::{self, AgentEvent, FailureKind, TurnFailure};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::embedding::{Input, Vector};
use quack_core::ids::{SessionId, WorkspaceId};
use quack_core::jobs::{JobId, JobKind, JobQueue, JobSpec, Lane, LaneKey};
use quack_core::llm::{self, Embeddings};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use quack_core::storage::sessions::{self, ChatMode};
use quack_core::storage::workspace::{
    ChunkScope, ExportFormat, HybridLimits, TEMP_OBJECT_REFUSED, WorkspaceDb, creates_temp_object,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use super::StreamEvent;
use super::okf::{BodyWriter, CHUNKS_IN_FLIGHT};
use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::{App, with_db};
use crate::server::web::markdown::to_html;

#[derive(Deserialize)]
pub(crate) struct QueryRequest {
    pub prompt: String,
    pub session_id: Option<SessionId>,
    pub mode: Option<ChatMode>,
    #[serde(default)]
    pub allow_write: bool,
}

/// A turn that has passed its checks and has its session: ready to run.
struct PreparedTurn {
    access: Access,
    db: SharedDb,
    reader: ReaderDb,
    session_id: SessionId,
    policy: WritePolicy,
    prompt: String,
    /// The chat model's provider and id.
    model: (String, String),
}

impl PreparedTurn {
    /// Everything a turn needs before the model is called: the
    /// authorization and the session (existing or new). The turn itself
    /// refuses a model the workspace's provider allow-list does not allow.
    /// `unasked` is the write policy without `allow_write`: `Ask` when the
    /// turn streams, so the person can answer, else `Deny`. A caller who
    /// may not write is never asked.
    async fn prepare(
        app: &App,
        identity: Identity,
        workspace_id: &WorkspaceId,
        body: &QueryRequest,
        unasked: WritePolicy,
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
        let model_names = (chat.provider_name.to_string(), chat.model.to_owned());
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
                    .ok_or_else(|| ResourceKind::Session.missing(id.as_str()))?;
                // A session's mode is set when it is created; `mode` on a
                // later turn is ignored, and PATCH .../sessions/{sid} changes
                // it explicitly (issue #57).
                Ok(id)
            } else {
                Ok(sessions::create_session(db, &model, mode.unwrap_or_default(), Some(&user))?.id)
            }
        })
        .await?;
        let unasked = if access.permits(Need::WRITE) {
            unasked
        } else {
            WritePolicy::Deny
        };
        let policy = unasked.allowed_if(body.allow_write);
        Ok(Self {
            access,
            db,
            reader,
            session_id,
            policy,
            prompt: body.prompt.clone(),
            model: model_names,
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
            model,
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
            .workspace(access.membership.workspace.id.clone())
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
            model,
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
    session_id: SessionId,
    prompt: String,
    /// The chat model's provider and id, for the audit detail.
    model: (String, String),
}

/// Whether a turn's stream has sent its `complete` or `error` event.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TurnStream {
    Open,
    Ended,
}

/// How a turn ended, for its audit row.
enum TurnEnd<'a> {
    Answered(&'a AgentResponse),
    /// With the turn's own failure, or none when its events just closed.
    Failed(Option<&'a TurnFailure>),
}

impl Turn {
    /// What the caller is told when the events closed with neither an
    /// answer nor a failure: 503 when a stopping server refused the turn
    /// (the request is fine; another server can take it), 500 when its job
    /// was cancelled while queued.
    fn unanswered(app: &App) -> ApiError {
        if app.stopping.is_cancelled() {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "the server is shutting down; the turn ended without an answer",
            )
        } else {
            ApiError::internal("the turn ended without an answer")
        }
    }

    /// Audit the turn. A failed turn that left the session without any
    /// message (a fresh session whose first turn never reached the model)
    /// is removed, as print mode does, so the session list shows no empty
    /// entries.
    async fn record(&self, app: &App, end: TurnEnd<'_>) {
        let (outcome, steps, citations) = match end {
            TurnEnd::Answered(response) => (
                Outcome::Allowed,
                response.steps.clone(),
                response.citations.clone(),
            ),
            TurnEnd::Failed(Some(TurnFailure {
                kind: FailureKind::ProviderNotAllowed,
                ..
            })) => (Outcome::Denied, Vec::new(), Vec::new()),
            TurnEnd::Failed(_) => (Outcome::Error, Vec::new(), Vec::new()),
        };
        if outcome != Outcome::Allowed
            && let Ok(db) = app.workspace_db(&self.access.membership.workspace.id).await
        {
            let sid = self.session_id.clone();
            if let Err(e) = with_db(db, move |db| sessions::delete_if_empty(db, &sid)).await {
                tracing::warn!(error = %e.message, "could not remove the empty session");
            }
        }
        // The model and the cited chunks beside the prompt and the steps:
        // what the workspace's OCSF export renders as the `ai_operation`
        // profile. Step rows are not kept here; the message has them.
        let steps: Vec<serde_json::Value> = steps
            .iter()
            .map(|s| {
                serde_json::json!({
                    "tool": s.tool, "detail": s.detail, "summary": s.summary,
                    "rows": s.rows, "duration_ms": s.duration_ms,
                })
            })
            .collect();
        let citations: Vec<serde_json::Value> = citations
            .iter()
            .map(|c| {
                serde_json::json!({
                    "document_id": c.document_id, "chunk_id": c.chunk_id, "chunk_index": c.chunk_index,
                })
            })
            .collect();
        let detail = serde_json::json!({
            "prompt": self.prompt,
            "model": { "provider": self.model.0, "name": self.model.1 },
            "steps": steps,
            "citations": citations,
        });
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
    let mut turn = PreparedTurn::prepare(&app, identity, &id, &body, WritePolicy::Deny)
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
            | AgentEvent::Reasoning
            | AgentEvent::TextDelta(_)
            | AgentEvent::ToolStarted { .. }
            | AgentEvent::ToolFinished(_) => {}
        }
    }
    if let Some(response) = complete {
        turn.record(&app, TurnEnd::Answered(&response)).await;
        return Ok(Json(response.to_json(&turn.session_id)));
    }
    turn.record(&app, TurnEnd::Failed(failure.as_ref())).await;
    Err(failure.map_or_else(|| Turn::unanswered(&app), ApiError::from))
}

/// The same turn as SSE: `text`, `tool_started`, `tool_finished`, then
/// `complete` with the full response object, or `error`. A stopping server
/// cancels the turn's job, so the stream ends as any cancelled turn does:
/// `complete` with `cancelled: true`, or `error` when the turn never ran.
pub(crate) async fn stream(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<QueryRequest>,
) -> ApiResult<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
    let turn = PreparedTurn::prepare(&app, identity, &id, &body, WritePolicy::Ask)
        .await?
        .start(&app);
    let state = (turn, app, TurnStream::Open);
    let stream = futures::stream::unfold(state, |(mut turn, app, state)| async move {
        let Some(event) = turn.events.recv().await else {
            if state == TurnStream::Ended {
                return None;
            }
            turn.record(&app, TurnEnd::Failed(None)).await;
            let out = StreamEvent::Error
                .event()
                .data(Turn::unanswered(&app).message);
            return Some((Ok(out), (turn, app, TurnStream::Ended)));
        };
        let state = match event {
            AgentEvent::TurnComplete(_) | AgentEvent::Failed(_) => TurnStream::Ended,
            AgentEvent::Status(_)
            | AgentEvent::Reasoning
            | AgentEvent::TextDelta(_)
            | AgentEvent::ToolStarted { .. }
            | AgentEvent::ToolFinished(_)
            | AgentEvent::PermissionRequired(_) => state,
        };
        let out = match event {
            // A comment line: clients skip it, and the stream stays in step.
            AgentEvent::Reasoning => Event::default().comment("reasoning"),
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
                let (sql, hold) = (request.sql.clone(), request.hold);
                let held = app
                    .permissions
                    .hold(&app, &turn.access, &turn.session_id, request);
                StreamEvent::PermissionRequired
                    .event()
                    .json_data(serde_json::json!({
                        "request": held.request,
                        "session_id": turn.session_id,
                        "sql": sql,
                        "reason": hold,
                        "notice": hold.notice(),
                        "expires_at": held.expires_at.to_string(),
                    }))
                    .unwrap_or_default()
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
                turn.record(&app, TurnEnd::Failed(Some(&failure))).await;
                StreamEvent::Error.event().data(failure.message)
            }
        };
        Some((Ok(out), (turn, app, state)))
    });
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

#[derive(Deserialize)]
pub(crate) struct SqlRequest {
    pub sql: String,
}

#[derive(Deserialize)]
pub(crate) struct ExportRequest {
    pub sql: String,
    #[serde(default)]
    pub format: ExportFormat,
}

/// `POST .../sql/export`: every row of a read statement, streamed in
/// `format` with no row cap, for the viewer role; a statement that
/// writes is refused. Audited as `export` with the statement, the format,
/// and the row count once the stream ends.
pub(crate) async fn export(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<ExportRequest>,
) -> ApiResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access.export_sql(&app, body.sql, body.format).await
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
    /// How long the statement ran, timed on its connection's thread, so a
    /// wait for the writer is not counted.
    pub duration_ms: u64,
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
    /// Stream every row of a read statement to the response in `format`
    /// (the pattern `okf::export` uses): the statement runs on a reader
    /// connection inside a read-only transaction, written a chunk at a
    /// time into a bounded channel the body drains. A statement that
    /// writes, or references internal tables, is refused before anything
    /// runs. The `export` audit row carries the statement, the format,
    /// and the rows written, or the error.
    pub(crate) async fn export_sql(
        self,
        app: &App,
        statement: String,
        format: ExportFormat,
    ) -> ApiResult<Response> {
        let workspace_id = self.membership.workspace.id.clone();
        let sql = statement.clone();
        let kind = app
            .read(&workspace_id, move |db| db.classify_user_statement(&sql))
            .await
            .map_err(|e| ApiError::forbidden(e.message))?;
        if kind.writes().map_err(ApiError::bad_request)? {
            self.audit(
                app,
                AuditAction::Export,
                None,
                Outcome::Denied,
                Some(serde_json::json!({ "sql": statement, "format": format })),
            )
            .await?;
            return Err(ApiError::bad_request(
                "only a read statement can be exported",
            ));
        }
        let reader = app.reader_db(&workspace_id).await?;
        let (tx, rx) = mpsc::channel::<io::Result<Bytes>>(CHUNKS_IN_FLIGHT);
        let failed = tx.clone();
        let audit_app = Arc::clone(app);
        let sql = statement.clone();
        tokio::spawn(async move {
            // `with_db` already runs the closure inside a read-only
            // transaction on a reader connection.
            let written = reader
                .with_db(move |db| {
                    let mut out = BodyWriter::new(tx);
                    let rows = db.stream_query(&sql, format, &mut out)?;
                    out.flush()?;
                    Ok(rows)
                })
                .await;
            let (outcome, detail) = match written {
                Ok(rows) => (
                    Outcome::Allowed,
                    serde_json::json!({ "sql": statement, "format": format, "rows": rows }),
                ),
                Err(e) => {
                    tracing::warn!(workspace = %workspace_id, error = %e, "sql export failed partway");
                    drop(failed.send(Err(io::Error::other(e.to_string()))).await);
                    (
                        Outcome::Error,
                        serde_json::json!({ "sql": statement, "format": format, "error": e.to_string() }),
                    )
                }
            };
            drop(failed);
            if let Err(e) = self
                .audit(&audit_app, AuditAction::Export, None, outcome, Some(detail))
                .await
            {
                tracing::error!(workspace = %workspace_id, error = %e.message, "could not audit a sql export");
            }
        });
        let body = Body::from_stream(futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|chunk| (chunk, rx))
        }));
        Ok((
            [
                (header::CONTENT_TYPE, String::from(format.content_type())),
                (
                    header::CONTENT_DISPOSITION,
                    format!("attachment; filename=\"query.{format}\""),
                ),
            ],
            body,
        )
            .into_response())
    }

    /// Classify, authorize, run, and audit one statement: the SQL the API
    /// and the web grid share.
    pub(crate) async fn execute_sql(&self, app: &App, statement: &str) -> ApiResult<SqlOutcome> {
        let access = self;
        let sql = statement.to_owned();
        // Classifying parses the statement: a read, never in the writer's line.
        let kind = app
            .read(&access.membership.workspace.id, move |db| {
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
        let reader_db = app.reader_db(&access.membership.workspace.id).await?;
        let sql = statement.to_owned();
        let max_rows = app.config.analysis.max_query_rows;
        let timed = move |db: &WorkspaceDb| {
            let began = Instant::now();
            db.execute_query_capped(&sql, max_rows)
                .map(|capped| (capped, began.elapsed()))
        };
        let result = if is_write {
            let db = app.workspace_db(&access.membership.workspace.id).await?;
            let result = with_db(db, timed).await;
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
            reader_db.with_db(timed).await.map_err(ApiError::from)
        };
        let outcome = Outcome::of(&result);
        access
            .audit(app, AuditAction::Sql, None, outcome, Some(detail))
            .await?;
        let (capped, took) = result.map_err(|e| ApiError::unprocessable(e.message))?;
        Ok(SqlOutcome {
            truncated: capped.truncated(),
            columns: capped.results.columns,
            rows: capped.results.rows,
            row_count: capped.total_rows,
            duration_ms: u64::try_from(took.as_millis()).unwrap_or(u64::MAX),
        })
    }
}

#[derive(Deserialize)]
pub(crate) struct SearchQuery {
    pub query: String,
    pub top_k: Option<u32>,
}

/// Hybrid retrieval with no model call: the embedding provider when one is
/// configured, else keyword search alone. The query is in the body: search
/// text is workspace content, and a URL ends up in logs.
pub(crate) async fn search(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(q): Json<SearchQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let query = q.query.trim().to_owned();
    if query.is_empty() {
        return Err(ApiError::bad_request("query must not be empty"));
    }
    let top_k = q.top_k.unwrap_or(app.config.retrieval.top_k).clamp(1, 100);
    let rrf_k = app.config.retrieval.rrf_k;
    let model = access
        .model(
            &app,
            AuditAction::Search,
            Embeddings::from_config(&app.config).await,
        )
        .await?;
    // Run the search, audit its outcome, then propagate — the way `execute_sql`
    // does, so a post-authorization failure is recorded instead of dropped.
    let search_result: ApiResult<Vec<_>> = async {
        let embedding: Option<Vector> = match model {
            Some(model) => Some(
                model
                    .embed_interactive(&Input::Query(query.clone()))
                    .await?,
            ),
            None => None,
        };
        let reader_db = app.reader_db(&id).await?;
        let text = query.clone();
        reader_db
            .with_db(move |db| {
                let scope = ChunkScope::all();
                match embedding.as_ref() {
                    Some(vector) => db.search_hybrid_chunks(
                        &text,
                        vector,
                        HybridLimits { top_k, rrf_k },
                        &scope,
                    ),
                    None => db.search_keyword_chunks(&text, top_k, &scope),
                }
            })
            .await
            .map_err(ApiError::from)
    }
    .await;
    let outcome = Outcome::of(&search_result);
    access
        .audit(
            &app,
            AuditAction::Search,
            None,
            outcome,
            Some(serde_json::json!({ "q": query })),
        )
        .await?;
    let hits = search_result?;
    Ok(Json(serde_json::json!({ "chunks": hits })))
}
