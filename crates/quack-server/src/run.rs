//! The server's end of a background run (`quack_core::jobs::runs`): the
//! person whose request started it audits its start and its end through
//! their `Recorder`, and the graph follow-up after an ingest runs as one
//! such run in the workspace's graph lane.

use std::sync::Arc;
use std::time::Duration;

use quack_core::analysis::tools::SharedDb;
use quack_core::graph::follow_up::{FollowUp, FollowUpSummary};
use quack_core::ids::DocumentId;
use quack_core::jobs::JobId;
use quack_core::jobs::runs::RunOwner;
use quack_core::llm::Embeddings;
use quack_core::ontology::store as ontology_store;
use quack_core::progress::{ChunkDone, RunControl};
use serde_json::Value;

use crate::auth::Access;
use crate::error::ApiResult;
use crate::state::App;

pub(crate) use quack_core::jobs::runs::{GraphReport, RunKind};

pub(crate) use quack_core::jobs::runs::BackgroundRun;

/// Mint the run id and write the start audit row with `detail` as
/// `access`; the request fails if that write does.
pub(crate) async fn start(
    app: &App,
    access: &Access,
    kind: RunKind,
    detail: Value,
) -> ApiResult<BackgroundRun> {
    let owner = RunOwner {
        workspace: access.membership.workspace.id.clone(),
        user: Some(access.identity.user_id.clone()),
    };
    let recorder = access.recorder(app).await?;
    Ok(BackgroundRun::start(&app.engine.jobs, recorder, owner, kind, detail).await?)
}

/// How many seconds a follow-up waits for the workspace's extraction slot
/// before giving up: a request-time table extraction holds it briefly.
const SLOT_WAIT_SECONDS: u32 = 60;

/// Queue the graph extraction that follows `documents` becoming ready,
/// when `[graph].follow_ingest` asks for one: an audited background run
/// in the workspace's graph lane, so it waits behind an extraction in
/// progress. `None` when nothing follows.
pub(crate) async fn follow_ingest(
    app: &App,
    access: &Access,
    db: SharedDb,
    embedder: Option<Embeddings>,
    documents: Vec<DocumentId>,
) -> ApiResult<Option<JobId>> {
    if app.engine.config.graph.follow_ingest.is_off() || documents.is_empty() {
        return Ok(None);
    }
    // Nothing to extract into without an ontology: no run, no audit rows.
    let workspace_id = access.membership.workspace.id.clone();
    if app
        .read(&workspace_id, ontology_store::latest_version)
        .await?
        .is_none()
    {
        return Ok(None);
    }
    let run = start(
        app,
        access,
        RunKind::FOLLOW_UP,
        serde_json::json!({
            "documents": documents,
            "follow_ingest": app.engine.config.graph.follow_ingest.as_str(),
        }),
    )
    .await?;
    let app = Arc::clone(app);
    let job = run.submit(move |ctx| async move {
        let mut slot = None;
        for _ in 0..SLOT_WAIT_SECONDS {
            slot = app.begin_extraction(&workspace_id);
            if slot.is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        let Some(slot) = slot else {
            return Err(String::from(
                "a graph extraction is still running for this workspace; run `graph extract` later",
            ));
        };
        let progress = |done: ChunkDone| ctx.progress(done.done, done.total);
        let cancel = ctx.cancel_token();
        let control = RunControl {
            progress: &progress,
            cancel: Some(&cancel),
        };
        let outcome = FollowUp {
            db: &db,
            config: &app.engine.config,
            embeddings: embedder.as_ref(),
        }
        .run(&documents, control)
        .await;
        drop(slot);
        match outcome {
            Ok(Some(summary)) => Ok(summary),
            Ok(None) => Ok(FollowUpSummary::default()),
            Err(e) => Err(e.to_string()),
        }
    });
    Ok(Some(job))
}
