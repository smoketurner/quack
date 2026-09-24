//! The workspace's vectors against the configured embedding profile, and
//! the refresh that brings stale ones up to date (202, background, in a
//! lane of its own per workspace).

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use quack_core::embedding::refresh::{self, Plan};
use quack_core::jobs::JobId;
use quack_core::llm::{self, Embeddings};
use quack_core::progress::{ChunkDone, RunControl};
use quack_core::storage::control::{AuditAction, Outcome};
use quack_core::storage::workspace::WorkspaceDb;

use crate::server::auth::{Access, Identity, Need, access};
use crate::server::error::ApiResult;
use crate::server::run::{BackgroundRun, RunKind};
use crate::server::state::App;
use serde::Serialize;

/// `GET .../embeddings`: how many vectors are current, stale (made under
/// another profile), or missing, and what a refresh would do.
pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let status = app.read(&id, WorkspaceDb::embedding_status).await?;
    access
        .audit_read(&app, AuditAction::EmbeddingsStatus, "embedding status")
        .await?;
    Ok(Json(serde_json::json!({
        "status": status,
        "stale_chunks": status.stale_chunks(),
        "plan": Plan::from_status(&status),
        "note": status.note(),
    })))
}

/// `POST .../embeddings/refresh`: 200 with nothing to do, else 202 with
/// the job embedding every stale or missing vector again.
pub(crate) async fn refresh(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<(StatusCode, Json<serde_json::Value>)> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let started = access.refresh_embeddings(&app).await?;
    Ok((started.status_code(), Json(serde_json::to_value(started)?)))
}

/// What asking for a refresh did.
#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum RefreshStarted {
    /// Every vector was already made with the current profile.
    Current { plan: Plan },
    /// A run embeds the stale and missing ones in the background.
    Running { plan: Plan, run: String, job: JobId },
}

impl RefreshStarted {
    /// 200 with nothing to do, 202 when a run goes on.
    pub(crate) fn status_code(&self) -> StatusCode {
        match self {
            Self::Current { .. } => StatusCode::OK,
            Self::Running { .. } => StatusCode::ACCEPTED,
        }
    }
}

impl Access {
    /// The refresh the API and the web page share.
    pub(crate) async fn refresh_embeddings(&self, app: &App) -> ApiResult<RefreshStarted> {
        // Fail now, not in the background, when no model can be built.
        let embedder = llm::required_embedding_model(&app.config).await?;
        let status = app
            .read(&self.workspace.id, WorkspaceDb::embedding_status)
            .await?;
        let plan = Plan::from_status(&status);
        if plan.is_empty() {
            self.audit(
                app,
                AuditAction::EmbeddingsRefresh,
                None,
                Outcome::Allowed,
                Some(serde_json::json!({ "plan": plan, "finished": true })),
            )
            .await?;
            return Ok(RefreshStarted::Current { plan });
        }
        let run = BackgroundRun::start(
            app,
            self,
            RunKind::Embeddings,
            serde_json::json!({ "plan": plan, "profile": embedder.profile() }),
        )
        .await?;
        let run_id = run.id().to_owned();
        let job = refresh_in_background(run, Arc::clone(app), self.workspace.id.clone(), embedder);
        Ok(RefreshStarted::Running {
            plan,
            run: run_id,
            job,
        })
    }
}

/// Run the refresh as `run`'s job.
fn refresh_in_background(
    run: BackgroundRun,
    app: App,
    workspace_id: String,
    embedder: Embeddings,
) -> JobId {
    run.submit(move |ctx| async move {
        let progress = |done: ChunkDone| ctx.progress(done.done, done.total);
        let cancel = ctx.cancel_token();
        let control = RunControl {
            progress: &progress,
            cancel: Some(&cancel),
        };
        match app.workspace_db(&workspace_id).await {
            Ok(db) => refresh::run(
                &db,
                &embedder,
                app.config.ingestion.embedding_batch_size,
                control,
            )
            .await
            .map_err(|e| e.to_string()),
            Err(e) => Err(e.message),
        }
    })
}
