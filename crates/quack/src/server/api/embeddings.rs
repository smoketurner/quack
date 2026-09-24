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
    let (code, body) = start(&app, &access, &id).await?;
    Ok((code, Json(body)))
}

/// The refresh the API and the web page share: `{status: "current"}`
/// with nothing to do, else `{plan, run, job, status: "running"}`.
pub(crate) async fn start(
    app: &App,
    access: &Access,
    id: &str,
) -> ApiResult<(StatusCode, serde_json::Value)> {
    // Fail now, not in the background, when no model can be built.
    let embedder = llm::required_embedding_model(&app.config).await?;
    let status = app.read(id, WorkspaceDb::embedding_status).await?;
    let plan = Plan::from_status(&status);
    if plan.is_empty() {
        access
            .audit(
                app,
                AuditAction::EmbeddingsRefresh,
                None,
                Outcome::Allowed,
                Some(serde_json::json!({ "plan": plan, "finished": true })),
            )
            .await?;
        return Ok((
            StatusCode::OK,
            serde_json::json!({ "plan": plan, "status": "current" }),
        ));
    }
    let run = BackgroundRun::start(
        app,
        access,
        RunKind::Embeddings,
        serde_json::json!({ "plan": plan, "profile": embedder.profile() }),
    )
    .await?;
    let run_id = run.id().to_owned();
    let job = refresh_in_background(run, Arc::clone(app), access.workspace.id.clone(), embedder);
    Ok((
        StatusCode::ACCEPTED,
        serde_json::json!({ "plan": plan, "run": run_id, "job": job, "status": "running" }),
    ))
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
