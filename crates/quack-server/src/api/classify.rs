//! Labelling a table's text with a decision model (issue #472): `POST
//! .../tables/classify` starts a run as a background job (202), or with
//! `?preview=N` labels the first rows and answers with them (200, nothing
//! written); `GET` lists the runs. A run is audited twice under one run id
//! like the other long runs; a preview is one `classify` row.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use quack_core::classify::{
    Classification, ClassificationOutline, ClassificationPreview, ClassificationRun,
    ClassificationRuns, Labelling, OnEnd, RunEnded, Waiting,
};
use quack_core::error::Error as CoreError;
use quack_core::ids::{RunId, SessionId, WorkspaceId};
use quack_core::jobs::JobId;
use quack_core::llm::decision::DecisionModel;
use quack_core::progress::{ChunkDone, RunControl};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::auth::{Access, Identity, Need};
use crate::error::{ApiError, ApiResult};
use crate::run::{BackgroundRun, RunKind};
use crate::state::App;

/// Runs the listing shows at most.
const LISTED_RUNS: u32 = 100;

/// `?preview=N` on `POST .../tables/classify`.
#[derive(Debug, Clone, Copy, Deserialize, utoipa::IntoParams)]
pub(crate) struct PreviewQuery {
    /// Label only the first N rows (1 to 100), answer with them, and write
    /// nothing.
    pub preview: Option<u32>,
}

/// The run a request started.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ClassifyStarted {
    /// The run's id, in the closing audit row and in `GET .../tables/classify`.
    pub run: RunId,
    /// The job that labels the rows.
    pub job: JobId,
    /// What the run does: how many rows, into which table, with which
    /// questions.
    pub outline: ClassificationOutline,
}

/// Label a table's rows with the decision model: a background job (202), or
/// with `?preview=N` the first rows' labels (200), writing nothing.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/tables/classify",
    tag = "tables",
    request_body = Classification,
    params(WorkspaceId, PreviewQuery),
    responses(
        (status = 202, description = "A job labels the rows", body = ClassifyStarted),
        (status = 200, description = "The first rows, labelled", body = ClassificationPreview),
    ),
)]
pub(crate) async fn classify(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Query(query): Query<PreviewQuery>,
    Json(request): Json<Classification>,
) -> ApiResult<Response> {
    let need = if query.preview.is_some() {
        Need::READ
    } else {
        Need::WRITE
    };
    let access = Access::resolve(&app, identity, &id, need).await?;
    Ok(match query.preview {
        Some(rows) => {
            Json(access.preview_classification(&app, &request, rows).await?).into_response()
        }
        None => (
            StatusCode::ACCEPTED,
            Json(access.classify(&app, request).await?),
        )
            .into_response(),
    })
}

/// The runs that labelled tables, newest first.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/tables/classify",
    tag = "tables",
    params(WorkspaceId),
    responses((status = 200, description = "The runs", body = ClassificationRuns)),
)]
pub(crate) async fn runs(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<ClassificationRuns>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "classifications")
        .await?;
    Ok(Json(
        app.read(&id, |db| ClassificationRun::list(db, LISTED_RUNS))
            .await?,
    ))
}

impl Access {
    /// The hook that audits a run an agent turn or an MCP call started, when
    /// it ends and under the run's id, so a cancelled turn or a client that
    /// left still leaves its row. `session` is the turn's, when there is one.
    pub(crate) fn run_audit(&self, app: &App, session: Option<SessionId>) -> OnEnd {
        let (access, app) = (self.clone(), App::clone(app));
        OnEnd::new(move |ended: RunEnded| {
            let (access, app, session) = (access.clone(), App::clone(&app), session.clone());
            async move {
                let RunEnded { run, outcome } = ended;
                let detail = serde_json::json!({
                    "table": run.source_table,
                    "set": run.question_set.name,
                    "output_table": run.output_table,
                    "summary": run.to_string(),
                    "session_id": session,
                });
                let resource = ResourceKind::ClassificationRun.id(&run.id);
                if let Err(e) = access
                    .audit(
                        &app,
                        AuditAction::Classify,
                        Some(resource),
                        outcome,
                        Some(detail),
                    )
                    .await
                {
                    tracing::error!(error = %e.message, "audit write failed");
                }
            }
        })
    }

    /// The decision model for this workspace, or why not: a provider the
    /// allow-list refuses is 403, and either way the attempt is audited.
    async fn decision_model(&self, app: &App) -> ApiResult<DecisionModel> {
        let built = DecisionModel::from_config(&app.config)
            .await
            .and_then(|model| model.ok_or(CoreError::NoDecisionModel));
        self.model(app, AuditAction::Classify, built).await
    }

    /// Audit a refused or failed labelling and give the API's error for it.
    async fn classify_failed(&self, app: &App, error: CoreError) -> ApiError {
        let detail = serde_json::json!({ "error": error.to_string() });
        match self
            .audit(
                app,
                AuditAction::Classify,
                None,
                Outcome::of_failure(&error),
                Some(detail),
            )
            .await
        {
            Ok(_) => ApiError::from(error),
            Err(audit_failed) => audit_failed,
        }
    }

    /// Label the first rows of a request and hand them back, from the API
    /// or the web console; nothing is written. Audited as one `classify`
    /// row.
    pub(crate) async fn preview_classification(
        &self,
        app: &App,
        request: &Classification,
        rows: u32,
    ) -> ApiResult<ClassificationPreview> {
        let decision = self.decision_model(app).await?;
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let waiting = Waiting::Caller {
            budget: app.config.decision.interactive_budget,
        };
        let preview = request
            .preview(&db, &decision, rows, waiting, RunControl::unobserved())
            .await;
        match preview {
            Ok(preview) => {
                let detail = serde_json::json!({
                    "preview": rows,
                    "table": preview.output_table,
                    "labelled": preview.labelled,
                });
                self.audit(
                    app,
                    AuditAction::Classify,
                    None,
                    Outcome::Allowed,
                    Some(detail),
                )
                .await?;
                Ok(preview)
            }
            Err(error) => Err(self.classify_failed(app, error).await),
        }
    }

    /// Start labelling a request's rows as a background job, from the API
    /// or the web console. What can be refused now is: no decision model,
    /// an unusable table, key, or questions, a table of labels made under
    /// other questions or model, or another run into the same table.
    pub(crate) async fn classify(
        &self,
        app: &App,
        request: Classification,
    ) -> ApiResult<ClassifyStarted> {
        let decision = self.decision_model(app).await?;
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let outline = match request.outline(&db, &decision).await {
            Ok(outline) => outline,
            Err(error) => return Err(self.classify_failed(app, error).await),
        };
        let run = BackgroundRun::start(
            app,
            self,
            RunKind::CLASSIFY,
            serde_json::json!({ "outline": outline, "model": decision.label() }),
        )
        .await?;
        let run_id = run.id().clone();
        let started = run_id.clone();
        let started_by = self.identity.user_id.to_string();
        let job = run.submit(move |ctx| async move {
            let progress = |done: ChunkDone| ctx.progress(done.done, done.total);
            let cancel = ctx.cancel_token();
            request
                .run(Labelling {
                    db: &db,
                    decision: &decision,
                    started_by: Some(&started_by),
                    run_id: run_id.clone(),
                    waiting: Waiting::Job,
                    control: RunControl {
                        progress: &progress,
                        cancel: Some(&cancel),
                    },
                })
                .await
                .map_err(|e| e.to_string())
        });
        Ok(ClassifyStarted {
            run: started,
            job,
            outline,
        })
    }
}
