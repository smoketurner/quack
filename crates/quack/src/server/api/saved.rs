//! Saved questions (design doc 8.1): an answer's SQL kept under a name
//! and run again without the model. Anyone who may read the workspace
//! lists, shows, and runs them; saving needs the source session to be
//! visible; removing follows a session's rule, its creator or an owner.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use quack_core::error::Record;
use quack_core::ids::{SavedId, SessionId, WorkspaceId};
use quack_core::saved::{self, Answer, RunStatus, SavedQuestion};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use serde::Deserialize;

use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::{App, with_db};

pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "saved questions")
        .await?;
    let questions = app.read(&id, saved::list).await?;
    Ok(Json(serde_json::json!({ "saved": questions })))
}

#[derive(Deserialize)]
pub(crate) struct NewSaved {
    pub name: String,
    /// The session the answer is in.
    pub session_id: SessionId,
    /// The answer's message sequence number; the session's last answer
    /// otherwise.
    pub message: Option<i64>,
}

/// Save an answer the caller can see. Audited as `save`; a refused save
/// (no SQL, a write, a name in use) is an error row.
pub(crate) async fn create(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<NewSaved>,
) -> ApiResult<(StatusCode, Json<serde_json::Value>)> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let session = access.visible_session(&app, &body.session_id).await?;
    let db = app.workspace_db(&id).await?;
    let answer = body.message.map_or(Answer::Last, Answer::Seq);
    let creator = access.identity.user_id.clone();
    let name = body.name;
    let session_id = session.id.clone();
    let result = with_db(db, move |db| {
        saved::save(db, &name, &session_id, answer, Some(&creator))
    })
    .await;
    let resource = result.as_ref().ok().map(|q| q.id.clone());
    access
        .audit(
            &app,
            AuditAction::Save,
            resource.as_ref().map(|r| ResourceKind::SavedQuestion.id(r)),
            Outcome::of(&result),
            Some(serde_json::json!({ "session": session.id, "message": body.message })),
        )
        .await?;
    Ok((StatusCode::CREATED, Json(serde_json::to_value(result?)?)))
}

impl Access {
    /// The saved question, or 404.
    async fn saved_question(&self, app: &App, saved: &SavedId) -> ApiResult<SavedQuestion> {
        let wanted = saved.clone();
        let found = app
            .read(&self.workspace.id, move |db| saved::by_id(db, &wanted))
            .await?;
        found.ok_or_else(|| Record::SavedQuestion.missing(saved.as_str()).into())
    }
}

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path((id, saved)): Path<(WorkspaceId, SavedId)>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let question = access.saved_question(&app, &saved).await?;
    let wanted = saved.clone();
    let last_run = app
        .read(&id, move |db| saved::runs(db, &wanted, 1))
        .await?
        .into_iter()
        .next();
    access
        .audit(
            &app,
            AuditAction::Open,
            Some(ResourceKind::SavedQuestion.id(&saved)),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(Json(
        serde_json::json!({ "question": question, "last_run": last_run }),
    ))
}

/// Remove a saved question and its runs: its creator, or an owner, may.
/// Audited as `delete`.
pub(crate) async fn remove(
    State(app): State<App>,
    identity: Identity,
    Path((id, saved)): Path<(WorkspaceId, SavedId)>,
) -> ApiResult<StatusCode> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let question = access.saved_question(&app, &saved).await?;
    let resource = Some(ResourceKind::SavedQuestion.id(&saved));
    if !access.owns(question.created_by.as_ref()) {
        access
            .audit(&app, AuditAction::Delete, resource, Outcome::Denied, None)
            .await?;
        return Err(ApiError::forbidden(
            "only the saved question's creator or an owner may remove it",
        ));
    }
    let db = app.workspace_db(&id).await?;
    with_db(db, move |db| saved::remove(db, &question.id)).await?;
    access
        .audit(&app, AuditAction::Delete, resource, Outcome::Allowed, None)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Run the saved SQL now, no model, and answer with the run: it runs on
/// the writer like the agent's `run_sql`, so no job is needed. Audited as
/// `saved_run` with the run id, its status, and `changed`.
pub(crate) async fn run(
    State(app): State<App>,
    identity: Identity,
    Path((id, saved)): Path<(WorkspaceId, SavedId)>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let question = access.saved_question(&app, &saved).await?;
    let db = app.workspace_db(&id).await?;
    let max_rows = app.config.analysis.max_query_rows;
    let run = with_db(db, move |db| saved::run(db, &question, max_rows)).await?;
    let outcome = match run.status {
        RunStatus::Ok => Outcome::Allowed,
        RunStatus::Failed => Outcome::Error,
    };
    access
        .audit(
            &app,
            AuditAction::SavedRun,
            Some(ResourceKind::SavedQuestion.id(&saved)),
            outcome,
            Some(serde_json::json!({
                "run": run.id,
                "status": run.status,
                "changed": run.changed,
            })),
        )
        .await?;
    Ok(Json(serde_json::to_value(run)?))
}

#[derive(Deserialize)]
pub(crate) struct RunsQuery {
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    20
}

pub(crate) async fn runs(
    State(app): State<App>,
    identity: Identity,
    Path((id, saved)): Path<(WorkspaceId, SavedId)>,
    Query(q): Query<RunsQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access.saved_question(&app, &saved).await?;
    access
        .audit(
            &app,
            AuditAction::List,
            Some(ResourceKind::SavedQuestion.id(&saved)),
            Outcome::Allowed,
            Some(serde_json::json!({ "what": "saved runs" })),
        )
        .await?;
    let limit = q.limit;
    let runs = app
        .read(&id, move |db| saved::runs(db, &saved, limit))
        .await?;
    Ok(Json(serde_json::json!({ "runs": runs })))
}
