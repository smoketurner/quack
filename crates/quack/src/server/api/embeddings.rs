//! The workspace's vectors against the configured embedding profile, and
//! the re-embed that brings stale ones up to date (202, background, in a
//! lane of its own per workspace).

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use quack_core::embedding::reembed::{self, Plan, Progress};
use quack_core::jobs::{JobId, JobKind, JobSpec, Lane};
use quack_core::llm::{self, Embeddings};
use quack_core::storage::control::Outcome;
use quack_core::storage::workspace::{EmbeddingStatus, WorkspaceDb};

use crate::server::api::graph::audit_cancelled;
use crate::server::auth::{Access, Identity, Need, access};
use crate::server::error::ApiResult;
use crate::server::queue::when_cancelled_unstarted;
use crate::server::state::App;

/// What the status and the re-embed report.
fn describe(status: &EmbeddingStatus) -> serde_json::Value {
    serde_json::json!({
        "status": status,
        "stale_chunks": status.stale_chunks(),
        "plan": Plan::from_status(status),
        "note": status.note(),
    })
}

/// `GET .../embeddings`: how many vectors are current, stale (made under
/// another profile), or missing, and what a re-embed would do.
pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let status = app.read(&id, WorkspaceDb::embedding_status).await?;
    access
        .audit_read(&app, "embeddings_status", "embedding status")
        .await?;
    Ok(Json(describe(&status)))
}

/// `POST .../embeddings/reembed`: 200 with nothing to do, else 202 with
/// the job re-embedding every stale or missing vector.
pub(crate) async fn reembed(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<(StatusCode, Json<serde_json::Value>)> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let (code, body) = start(&app, &access, &id).await?;
    Ok((code, Json(body)))
}

/// The re-embed the API and the web page share: `{status: "current"}`
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
                "reembed",
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
    let run = uuid::Uuid::now_v7().to_string();
    access
        .audit(
            app,
            "reembed",
            Some(("reembed_run", &run)),
            Outcome::Allowed,
            Some(serde_json::json!({ "plan": plan, "profile": embedder.profile() })),
        )
        .await?;
    let job = spawn(Arc::clone(app), access.clone(), run.clone(), embedder);
    Ok((
        StatusCode::ACCEPTED,
        serde_json::json!({ "plan": plan, "run": run, "job": job, "status": "running" }),
    ))
}

/// Run the re-embed as a job and audit its end under `run_id`.
fn spawn(app: App, access: Access, run_id: String, embedder: Embeddings) -> JobId {
    let workspace_id = access.workspace.id.clone();
    let spec = JobSpec::new(JobKind::Reembed, "re-embed")
        .workspace(workspace_id.clone())
        .owner(Some(access.identity.user_id.clone()))
        .lane(Lane::serial(format!("reembed:{workspace_id}")));
    let jobs = app.jobs.clone();
    let (cancel_app, cancel_access, cancel_run) =
        (Arc::clone(&app), access.clone(), run_id.clone());
    let id = jobs.submit(spec, move |ctx| async move {
        let progress = |p: Progress| ctx.progress(p.done, p.total);
        let cancel = ctx.cancel_token();
        let outcome = match app.workspace_db(&workspace_id).await {
            Ok(db) => reembed::run(
                &db,
                &embedder,
                app.config.ingestion.embedding_batch_size,
                &progress,
                Some(&cancel),
            )
            .await
            .map_err(|e| e.to_string()),
            Err(e) => Err(e.message),
        };
        let (audit_outcome, detail) = match &outcome {
            Ok(summary) => (
                Outcome::Allowed,
                serde_json::json!({ "finished": true, "summary": summary }),
            ),
            Err(e) => (
                Outcome::Error,
                serde_json::json!({ "finished": true, "error": e }),
            ),
        };
        if let Err(e) = access
            .audit(
                &app,
                "reembed",
                Some(("reembed_run", &run_id)),
                audit_outcome,
                Some(detail),
            )
            .await
        {
            tracing::error!(error = %e.message, "audit write failed after re-embedding");
        }
        outcome.map(|summary| {
            format!(
                "{} chunks and {} graph node labels re-embedded",
                summary.chunks, summary.nodes
            )
        })
    });
    when_cancelled_unstarted(&jobs, id, move || async move {
        audit_cancelled(
            &cancel_app,
            &cancel_access,
            "reembed",
            "reembed_run",
            &cancel_run,
        )
        .await;
    });
    id
}
