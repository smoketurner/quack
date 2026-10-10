//! Labelling a table's text with a decision model (issue #472): `POST
//! .../tables/classify` takes a table and a sentence (or the questions a
//! preview returned, or neither, for the last approved ones). With
//! `?preview=N` it labels the first rows and answers with the questions and
//! the labels (200, nothing written, nothing kept); without it a run starts
//! as a background job (202) and its questions become the approved ones.
//! `GET` lists the runs. A run is audited twice under one run id like the
//! other long runs; a preview is one `classify` row.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use quack_core::classify::{
    self, Draft, DraftContext, Drafter, Labelling, OnEnd, Rows, RunEnded, Waiting,
};
use quack_core::error::Error as CoreError;
use quack_core::ids::{RunId, SessionId, WorkspaceId};
use quack_core::jobs::JobId;
use quack_core::llm::ChatDrafter;
use quack_core::llm::decision::DecisionModel;
use quack_core::progress::{ChunkDone, RunControl};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use quack_core::storage::writer::Writer;
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
pub(crate) struct Started {
    /// The run's id, in the closing audit row and in `GET .../tables/classify`.
    pub run: RunId,
    /// The job that labels the rows.
    pub job: JobId,
    /// The questions the run asks, now recorded as approved.
    pub draft: Draft,
    /// What the run does: how many rows, into which table, with which
    /// questions.
    pub outline: classify::Outline,
}

/// Label a table's rows with the decision model: a background job (202), or
/// with `?preview=N` the questions and the labels of the first rows (200),
/// writing and keeping nothing. With a `sentence` and no `set` the request
/// waits for the chat model's draft, typically 30 to 120 seconds; set
/// client and proxy timeouts to match, or preview first and send the
/// returned `set` back to run exactly it.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/tables/classify",
    tag = "tables",
    request_body = classify::Request,
    params(WorkspaceId, PreviewQuery),
    responses(
        (status = 202, description = "A job labels the rows", body = Started),
        (status = 200, description = "The questions and the labels of the first rows", body = classify::Report),
    ),
)]
pub(crate) async fn label(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Query(query): Query<PreviewQuery>,
    Json(request): Json<classify::Request>,
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
    responses((status = 200, description = "The runs", body = classify::Runs)),
)]
pub(crate) async fn runs(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<classify::Runs>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "classifications")
        .await?;
    Ok(Json(
        app.read(&id, |db| classify::Run::list(db, LISTED_RUNS))
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
                    "output_table": run.output_table,
                    "sentence": run.sentence,
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

    /// The draft a request names: the `set` sent back, checked; else the
    /// chat model's questions for the `sentence`; else the last approved
    /// questions. Nothing is stored.
    async fn draft_of(
        &self,
        app: &App,
        db: &Writer,
        decision: &DecisionModel,
        request: &classify::Request,
    ) -> Result<Draft, CoreError> {
        let drafter = ChatDrafter::from_config(&app.config);
        let context = DraftContext {
            db,
            decision,
            drafter: drafter.as_ref().map(|d| -> &dyn Drafter { d }),
        };
        match &request.set {
            Some(set) => Draft::given(&context, &request.table, set.clone()).await,
            None => Draft::prepare(&context, &request.table, request.sentence.as_deref()).await,
        }
    }

    /// Draft the questions for a table and a sentence (or the last approved
    /// ones), as the web's Draft button asks. Audited as one `classify`
    /// row; nothing is stored.
    pub(crate) async fn draft_questions(
        &self,
        app: &App,
        table: &str,
        sentence: Option<&str>,
    ) -> ApiResult<Draft> {
        let decision = self.decision_model(app).await?;
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let request = classify::Request {
            table: table.to_owned(),
            sentence: sentence.map(str::to_owned),
            set: None,
            rows: Rows::Missing,
        };
        match self.draft_of(app, &db, &decision, &request).await {
            Ok(draft) => {
                let detail = serde_json::json!({
                    "draft": true,
                    "table": draft.table,
                    "origin": draft.origin,
                    "sentence": draft.set.sentence,
                });
                self.audit(
                    app,
                    AuditAction::Classify,
                    None,
                    Outcome::Allowed,
                    Some(detail),
                )
                .await?;
                Ok(draft)
            }
            Err(error) => Err(self.classify_failed(app, error).await),
        }
    }

    /// Label the first rows of a request and hand back the questions and
    /// the labels, from the API or the web console; nothing is written or
    /// kept. Audited as one `classify` row.
    pub(crate) async fn preview_classification(
        &self,
        app: &App,
        request: &classify::Request,
        rows: u32,
    ) -> ApiResult<classify::Report> {
        let decision = self.decision_model(app).await?;
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let waiting = Waiting::Caller {
            budget: app.config.decision.interactive_budget,
        };
        let report = async {
            let draft = self.draft_of(app, &db, &decision, request).await?;
            let preview = draft
                .preview(
                    &db,
                    &decision,
                    rows,
                    request.rows,
                    waiting,
                    RunControl::unobserved(),
                )
                .await?;
            let outline = draft
                .outline(&db, &decision, request.rows)
                .await?
                .estimated_from(&preview);
            Ok::<_, CoreError>(classify::Report {
                draft,
                outline,
                preview: Some(preview),
                run: None,
            })
        }
        .await;
        match report {
            Ok(report) => {
                let detail = serde_json::json!({
                    "preview": rows,
                    "table": report.draft.table,
                    "origin": report.draft.origin,
                    "sentence": report.draft.set.sentence,
                });
                self.audit(
                    app,
                    AuditAction::Classify,
                    None,
                    Outcome::Allowed,
                    Some(detail),
                )
                .await?;
                Ok(report)
            }
            Err(error) => Err(self.classify_failed(app, error).await),
        }
    }

    /// Start labelling a request's rows as a background job, from the API
    /// or the web console. What can be refused now is: no decision model,
    /// no questions, an unusable table, key, or questions, or another run
    /// into the same table.
    pub(crate) async fn classify(
        &self,
        app: &App,
        request: classify::Request,
    ) -> ApiResult<Started> {
        let decision = self.decision_model(app).await?;
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let planned = async {
            let draft = self.draft_of(app, &db, &decision, &request).await?;
            let outline = draft.outline(&db, &decision, request.rows).await?;
            Ok::<_, CoreError>((draft, outline))
        }
        .await;
        let (draft, outline) = match planned {
            Ok(planned) => planned,
            Err(error) => return Err(self.classify_failed(app, error).await),
        };
        let run = BackgroundRun::start(
            app,
            self,
            RunKind::CLASSIFY,
            serde_json::json!({
                "outline": outline,
                "model": decision.label(),
                "origin": draft.origin,
                "sentence": draft.set.sentence,
            }),
        )
        .await?;
        let run_id = run.id().clone();
        let started = run_id.clone();
        let started_by = self.identity.user_id.to_string();
        let rows = request.rows;
        let shown = draft.clone();
        let job = run.submit(move |ctx| async move {
            let progress = |done: ChunkDone| ctx.progress(done.done, done.total);
            let cancel = ctx.cancel_token();
            draft
                .run(
                    Labelling {
                        db: &db,
                        decision: &decision,
                        started_by: Some(&started_by),
                        run_id: run_id.clone(),
                        waiting: Waiting::Job,
                        control: RunControl {
                            progress: &progress,
                            cancel: Some(&cancel),
                        },
                    },
                    rows,
                )
                .await
                .map_err(|e| e.to_string())
        });
        Ok(Started {
            run: started,
            job,
            draft: shown,
            outline,
        })
    }
}
