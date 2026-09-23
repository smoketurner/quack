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
use quack_core::jobs::{JobId, JobKind, JobQueue, JobSpec, Lane, LaneKey};
use quack_core::llm;
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use quack_core::storage::sessions::{self, ChatMode};
use quack_core::storage::workspace::{ChunkScope, StatementKind};
use serde::Deserialize;

use crate::server::auth::{Access, Identity, Need, access};
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

/// The response object shared with print mode (design doc 11.2).
pub(crate) fn response_json(response: &AgentResponse, session_id: &str) -> serde_json::Value {
    response.to_json(session_id)
}

/// Everything a turn needs before the model is called: the authorization,
/// the provider check, and the session (existing or new).
async fn prepare(
    app: &App,
    identity: Identity,
    workspace_id: &str,
    body: &QueryRequest,
) -> ApiResult<(Access, SharedDb, ReaderDb, String, WritePolicy)> {
    let access = access(app, identity, workspace_id, Need::READ).await?;
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
    if let Some(allowed) = access.workspace.allowed_providers.as_deref() {
        let names: Vec<String> = serde_json::from_str(allowed).unwrap_or_default();
        if !names.iter().any(|n| n == chat.provider_name) {
            return Err(ApiError::forbidden(format!(
                "provider '{}' is not allowed in this workspace",
                chat.provider_name
            )));
        }
    }
    let mode = body.mode;
    let db = app.workspace_db(workspace_id).await?;
    let reader_db = app.reader_db(workspace_id).await?;
    let requested = body.session_id.clone();
    let model = chat.to_string();
    let user = access.identity.user_id.clone();
    let sees_all = access.sees_all_sessions();
    let session_id = with_db(Arc::clone(&db), move |db| {
        if let Some(id) = requested {
            // A session the caller may not see reads as missing, not forbidden.
            sessions::get_session(db, &id)?
                .filter(|s| sessions::visible_to(s, &user, sees_all))
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
    Ok((access, db, reader_db, session_id, policy))
}

/// Cancels the turn's job when dropped: the SSE stream holds one, so a
/// client that goes away stops the model (or takes a queued turn out of
/// the queue) instead of leaving the turn to finish unwatched (issue #45).
struct CancelOnDrop(JobQueue, JobId);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel(self.1);
    }
}

/// Submit the turn as a job in its session's lane (one turn at a time per
/// session, in order) and return its events. When an earlier turn of the
/// session is still going, the stream opens with a `status` saying so.
fn start_turn(
    app: &App,
    access: &Access,
    db: SharedDb,
    reader_db: ReaderDb,
    session_id: &str,
    policy: WritePolicy,
    prompt: &str,
) -> (events::EventStream, CancelOnDrop) {
    let (sink, stream) = events::channel();
    let ahead = app
        .jobs
        .lane_active(&LaneKey::Session(session_id.to_owned()));
    if ahead > 0 {
        drop(sink.send(AgentEvent::Status(String::from(
            "Queued: this runs when the session's previous question has been answered.",
        ))));
    }
    let config = app.config.clone();
    let session = session_id.to_owned();
    let text = prompt.to_owned();
    let spec = JobSpec::new(JobKind::Chat, prompt.chars().take(80).collect::<String>())
        .workspace(access.workspace.id.clone())
        .owner(Some(access.identity.user_id.clone()))
        .lane(Lane::serial(&LaneKey::Session(session_id.to_owned())));
    let job = app.jobs.submit(spec, move |ctx| async move {
        // run_turn emits TurnComplete or Failed itself.
        match llm::run_turn(
            &config,
            db,
            reader_db,
            &session,
            policy,
            &text,
            sink,
            ctx.cancel_token(),
        )
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
    });
    (stream, CancelOnDrop(app.jobs.clone(), job))
}

/// Audit the turn. A failed turn that left the session without any message
/// (a fresh session whose first turn never reached the model) is removed,
/// as print mode does, so the session list shows no empty entries.
async fn record_turn(
    app: &App,
    access: &Access,
    session_id: &str,
    prompt: &str,
    outcome: Outcome,
    response: Option<&AgentResponse>,
) {
    if outcome == Outcome::Error
        && let Ok(db) = app.workspace_db(&access.workspace.id).await
    {
        let sid = session_id.to_owned();
        if let Err(e) = with_db(db, move |db| sessions::delete_if_empty(db, &sid)).await {
            tracing::warn!(error = %e.message, "could not remove the empty session");
        }
    }
    let detail = serde_json::json!({
        "prompt": prompt,
        "steps": response.map(|r| r.steps.clone()).unwrap_or_default(),
    });
    if let Err(e) = access
        .audit(
            app,
            AuditAction::Query,
            Some(ResourceKind::Session.id(session_id)),
            outcome,
            Some(detail),
        )
        .await
    {
        tracing::error!(error = %e.message, "audit write failed");
    }
}

pub(crate) async fn query(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Json(body): Json<QueryRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    let (access, db, reader_db, session_id, policy) = prepare(&app, identity, &id, &body).await?;
    // A client that disconnects drops this future, and the guard with it,
    // which cancels the turn.
    let (mut stream, _guard) = start_turn(
        &app,
        &access,
        db,
        reader_db,
        &session_id,
        policy,
        &body.prompt,
    );
    let mut failure = None;
    let mut complete = None;
    while let Some(event) = stream.recv().await {
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
    match (complete, failure) {
        (Some(response), _) => {
            record_turn(
                &app,
                &access,
                &session_id,
                &body.prompt,
                Outcome::Allowed,
                Some(&response),
            )
            .await;
            Ok(Json(response_json(&response, &session_id)))
        }
        (None, failure) => {
            record_turn(
                &app,
                &access,
                &session_id,
                &body.prompt,
                Outcome::Error,
                None,
            )
            .await;
            Err(failure.map_or_else(
                || ApiError::internal("the turn ended without an answer"),
                ApiError::from,
            ))
        }
    }
}

/// The same turn as SSE: `text`, `tool_started`, `tool_finished`, then
/// `complete` with the full response object, or `error`.
pub(crate) async fn stream(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Json(body): Json<QueryRequest>,
) -> ApiResult<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
    let (access, db, reader_db, session_id, policy) = prepare(&app, identity, &id, &body).await?;
    let (events, guard) = start_turn(
        &app,
        &access,
        db,
        reader_db,
        &session_id,
        policy,
        &body.prompt,
    );
    let prompt = body.prompt.clone();
    let state = (events, app, access, session_id, prompt, guard);
    let stream = futures::stream::unfold(
        state,
        |(mut events, app, access, session_id, prompt, guard)| async move {
            let event = events.recv().await?;
            let out = match event {
                AgentEvent::Status(status) => super::StreamEvent::Status.event().data(status),
                AgentEvent::TextDelta(text) => super::StreamEvent::Text.event().data(text),
                AgentEvent::ToolStarted { tool, detail } => super::StreamEvent::ToolStarted
                    .event()
                    .json_data(serde_json::json!({ "tool": tool, "detail": detail }))
                    .unwrap_or_default(),
                AgentEvent::ToolFinished(step) => super::StreamEvent::ToolFinished
                    .event()
                    .json_data(&step)
                    .unwrap_or_default(),
                AgentEvent::PermissionRequired(request) => {
                    request.deny();
                    super::StreamEvent::WriteRefused
                        .event()
                        .data("writes are off for this request")
                }
                AgentEvent::TurnComplete(response) => {
                    record_turn(
                        &app,
                        &access,
                        &session_id,
                        &prompt,
                        Outcome::Allowed,
                        Some(&response),
                    )
                    .await;
                    // The web page swaps this in for the streamed plain text.
                    let mut payload = response_json(&response, &session_id);
                    if let serde_json::Value::Object(map) = &mut payload {
                        map.insert(
                            String::from("answer_html"),
                            serde_json::Value::String(to_html(&response.content)),
                        );
                    }
                    super::StreamEvent::Complete
                        .event()
                        .json_data(payload)
                        .unwrap_or_default()
                }
                AgentEvent::Failed(failure) => {
                    record_turn(&app, &access, &session_id, &prompt, Outcome::Error, None).await;
                    super::StreamEvent::Error.event().data(failure.message)
                }
            };
            Some((Ok(out), (events, app, access, session_id, prompt, guard)))
        },
    );
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

#[derive(Deserialize)]
pub(crate) struct SqlRequest {
    pub sql: String,
}

/// Direct SQL. Reads need the viewer role; anything that mutates needs the
/// member role and the write scope. `_quack_` tables are never reachable.
/// A capped result set for the API and the web grid.
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
    Path(id): Path<String>,
    Json(body): Json<SqlRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let outcome = execute_sql(&app, &access, &body.sql).await?;
    Ok(Json(serde_json::json!({
        "columns": outcome.columns,
        "rows": outcome.rows,
        "row_count": outcome.row_count,
        "truncated": outcome.truncated,
    })))
}

/// Classify, authorize, run, and audit one statement for an `Access`.
pub(crate) async fn execute_sql(
    app: &App,
    access: &Access,
    statement: &str,
) -> ApiResult<SqlOutcome> {
    let sql = statement.to_owned();
    // Classifying parses the statement: a read, never in the writer's line.
    let kind = app
        .read(&access.workspace.id, move |db| {
            db.classify_user_statement(&sql)
        })
        .await
        .map_err(|e| ApiError::forbidden(e.message))?;
    let is_write = match kind {
        StatementKind::Read => false,
        StatementKind::Write => true,
        StatementKind::Invalid(message) => return Err(ApiError::bad_request(message)),
    };
    let detail = serde_json::json!({ "sql": statement });
    if is_write && quack_core::analysis::tools::creates_temp_object(statement) {
        access
            .audit(app, AuditAction::Sql, None, Outcome::Denied, Some(detail))
            .await?;
        return Err(ApiError::bad_request(
            quack_core::analysis::tools::TEMP_OBJECT_REFUSED,
        ));
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
    let outcome = if result.is_ok() {
        Outcome::Allowed
    } else {
        Outcome::Error
    };
    access
        .audit(app, AuditAction::Sql, None, outcome, Some(detail))
        .await?;
    let capped = result
        .map_err(|e| ApiError::new(axum::http::StatusCode::UNPROCESSABLE_ENTITY, e.message))?;
    Ok(SqlOutcome {
        truncated: capped.truncated(),
        columns: capped.results.columns,
        rows: capped.results.rows,
        row_count: capped.total_rows,
    })
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
    Path(id): Path<String>,
    Query(q): Query<SearchQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let query = q.query.trim().to_owned();
    if query.is_empty() {
        return Err(ApiError::bad_request("query must not be empty"));
    }
    let top_k = q.top_k.unwrap_or(app.config.retrieval.top_k).clamp(1, 100);
    let rrf_k = app.config.retrieval.rrf_k;
    let embedding: Option<Vector> = match llm::optional_embedding_model(&app.config).await? {
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
                Some(vector) => db.search_hybrid_chunks(&text, vector, top_k, rrf_k, &scope),
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
