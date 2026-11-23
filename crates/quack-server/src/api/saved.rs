//! Saved questions (design doc 8.1): an answer's SQL kept under a name
//! and run again without the model. Anyone who may read the workspace
//! lists, shows, and runs them; saving needs the source session to be
//! visible; removing follows a session's rule, its creator or an owner.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use quack_core::ids::{SavedId, SessionId, WorkspaceId};
use quack_core::saved::{self, Answer, RunStatus, SavedQuestion, SavedRun};
use quack_core::storage::control::{AuditAction, ResourceKind};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::auth::{Access, Audit, Identity, Need};
use crate::error::{ApiError, ApiResult};
use crate::state::{App, with_db};

/// The workspace's saved questions.
#[derive(Serialize, ToSchema)]
pub(crate) struct SavedList {
    pub saved: Vec<SavedQuestion>,
}

/// The workspace's saved questions.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/saved",
    tag = "saved",
    params(WorkspaceId),
    responses((status = 200, description = "The saved questions", body = SavedList)),
)]
pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<SavedList>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let questions = access.list_saved(&app).await?;
    Ok(Json(SavedList { saved: questions }))
}

#[derive(Deserialize, ToSchema)]
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
#[utoipa::path(
    post,
    path = "/workspaces/{id}/saved",
    tag = "saved",
    request_body = NewSaved,
    params(WorkspaceId),
    responses((status = 201, description = "Saved", body = SavedQuestion)),
)]
pub(crate) async fn create(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<NewSaved>,
) -> ApiResult<(StatusCode, Json<SavedQuestion>)> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let saved = access
        .save_answer(&app, body.name, &body.session_id, body.message)
        .await?;
    Ok((StatusCode::CREATED, Json(saved)))
}

impl Access {
    /// The workspace's saved questions, audited as a `list`.
    pub(crate) async fn list_saved(&self, app: &App) -> ApiResult<Vec<SavedQuestion>> {
        self.audited(
            app,
            Audit::read(AuditAction::List, "saved questions"),
            app.read(&self.membership.workspace.id, saved::list),
        )
        .await
    }

    /// Save an answer in a session the caller can see under `name`: the
    /// message numbered `message`, else the session's last answer. Audited
    /// as `save`; a refused save (no SQL, a write, a name in use) is an
    /// error row.
    pub(crate) async fn save_answer(
        &self,
        app: &App,
        name: String,
        session: &SessionId,
        message: Option<i64>,
    ) -> ApiResult<SavedQuestion> {
        let audit = Audit::of(AuditAction::Save)
            .detail(serde_json::json!({ "session": session, "message": message }));
        self.audited_with(app, audit, async |audit| {
            let session = self.visible_session(app, session).await?;
            let db = app.workspace(&self.membership.workspace.id).await?.writer;
            let answer = message.map_or(Answer::Last, Answer::Seq);
            let creator = self.identity.user_id.clone();
            let session_id = session.id.clone();
            let question = with_db(db, move |db| {
                saved::save(db, &name, &session_id, answer, Some(&creator))
            })
            .await?;
            audit.set_resource(ResourceKind::SavedQuestion, question.id.as_str());
            Ok(question)
        })
        .await
    }

    /// The saved question, or 404.
    async fn saved_question(&self, app: &App, saved: &SavedId) -> ApiResult<SavedQuestion> {
        let wanted = saved.clone();
        let found = app
            .read(&self.membership.workspace.id, move |db| {
                saved::by_id(db, &wanted)
            })
            .await?;
        found.ok_or_else(|| ResourceKind::SavedQuestion.missing(saved.as_str()).into())
    }
}

/// A saved question and its newest run.
#[derive(Serialize, ToSchema)]
pub(crate) struct SavedDetail {
    pub question: SavedQuestion,
    pub last_run: Option<SavedRun>,
}

/// A saved question and its newest run.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/saved/{saved}",
    tag = "saved",
    responses((status = 200, description = "The saved question", body = SavedDetail)),
)]
pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path((id, saved)): Path<(WorkspaceId, SavedId)>,
) -> ApiResult<Json<SavedDetail>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let audit = Audit::of(AuditAction::Open).resource(ResourceKind::SavedQuestion, saved.as_str());
    let detail = access
        .audited(&app, audit, async {
            let question = access.saved_question(&app, &saved).await?;
            let wanted = saved.clone();
            let last_run = app
                .read(&id, move |db| saved::runs(db, &wanted, 1))
                .await?
                .into_iter()
                .next();
            Ok(SavedDetail { question, last_run })
        })
        .await?;
    Ok(Json(detail))
}

/// Remove a saved question and its runs: its creator, or an owner, may.
/// Audited as `delete`.
#[utoipa::path(
    delete,
    path = "/workspaces/{id}/saved/{saved}",
    tag = "saved",
    responses((status = 204, description = "Removed")),
)]
pub(crate) async fn remove(
    State(app): State<App>,
    identity: Identity,
    Path((id, saved)): Path<(WorkspaceId, SavedId)>,
) -> ApiResult<StatusCode> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access.remove_saved(&app, &saved).await?;
    Ok(StatusCode::NO_CONTENT)
}

impl Access {
    /// Remove a saved question and its runs: its creator, or an owner, may.
    /// Audited as `delete`.
    pub(crate) async fn remove_saved(&self, app: &App, saved: &SavedId) -> ApiResult<()> {
        let audit =
            Audit::of(AuditAction::Delete).resource(ResourceKind::SavedQuestion, saved.as_str());
        self.audited(app, audit, async {
            let question = self.saved_question(app, saved).await?;
            if !self.owns(question.created_by.as_ref()) {
                return Err(ApiError::forbidden(
                    "only the saved question's creator or an owner may remove it",
                ));
            }
            let db = app.workspace(&self.membership.workspace.id).await?.writer;
            with_db(db, move |db| saved::remove(db, &question.id)).await?;
            Ok(())
        })
        .await
    }

    /// Run the saved SQL now, without the model, audited as `saved_run`
    /// with the run id, its status, and `changed`.
    pub(crate) async fn run_saved(&self, app: &App, saved: &SavedId) -> ApiResult<RunOf> {
        let audit =
            Audit::of(AuditAction::SavedRun).resource(ResourceKind::SavedQuestion, saved.as_str());
        self.audited_with(app, audit, async |audit| {
            let question = self.saved_question(app, saved).await?;
            let db = app.workspace(&self.membership.workspace.id).await?.writer;
            let max_rows = app.engine.config.analysis.max_query_rows;
            let ran = question.clone();
            let run = with_db(db, move |db| saved::run(db, &ran, max_rows)).await?;
            if run.status == RunStatus::Failed {
                audit.ended_in_failure();
            }
            audit.set_detail(serde_json::json!({
                "run": run.id,
                "status": run.status,
                "changed": run.changed,
            }));
            Ok(RunOf { question, run })
        })
        .await
    }
}

/// A run, and the saved question it ran.
pub(crate) struct RunOf {
    pub question: SavedQuestion,
    pub run: SavedRun,
}

/// Run the saved SQL now, no model, and answer with the run: it runs on
/// the writer like the agent's `run_sql`, so no job is needed. Audited as
/// `saved_run` with the run id, its status, and `changed`.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/saved/{saved}/run",
    tag = "saved",
    responses((status = 200, description = "The run, with its result sets", body = SavedRun)),
)]
pub(crate) async fn run(
    State(app): State<App>,
    identity: Identity,
    Path((id, saved)): Path<(WorkspaceId, SavedId)>,
) -> ApiResult<Json<SavedRun>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    Ok(Json(access.run_saved(&app, &saved).await?.run))
}

#[derive(Deserialize, utoipa::IntoParams)]
pub(crate) struct RunsQuery {
    /// Runs at most, newest first.
    #[serde(default = "default_limit")]
    #[param(default = 20)]
    pub limit: u32,
}

fn default_limit() -> u32 {
    20
}

/// A saved question's runs, newest first.
#[derive(Serialize, ToSchema)]
pub(crate) struct SavedRuns {
    pub runs: Vec<SavedRun>,
}

/// A saved question's runs, newest first.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/saved/{saved}/runs",
    tag = "saved",
    params(RunsQuery),
    responses((status = 200, description = "The runs", body = SavedRuns)),
)]
pub(crate) async fn runs(
    State(app): State<App>,
    identity: Identity,
    Path((id, saved)): Path<(WorkspaceId, SavedId)>,
    Query(q): Query<RunsQuery>,
) -> ApiResult<Json<SavedRuns>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let audit = Audit::read(AuditAction::List, "saved runs")
        .resource(ResourceKind::SavedQuestion, saved.as_str());
    let limit = q.limit;
    let runs = access
        .audited(&app, audit, async {
            access.saved_question(&app, &saved).await?;
            app.read(&id, move |db| saved::runs(db, &saved, limit))
                .await
        })
        .await?;
    Ok(Json(SavedRuns { runs }))
}
