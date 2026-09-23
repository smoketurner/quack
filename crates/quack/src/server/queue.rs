//! Background work in `quack serve` goes through the process's one
//! [`JobQueue`](quack_core::jobs::JobQueue) (design doc 4.1): uploads,
//! graph extraction, the ontology document pass, and agent turns. Uploads
//! return 202 with the document id and the job id; each workspace's uploads
//! share a lane of `[server].workers_per_workspace`. Clients poll the
//! document, or the job (`GET .../jobs/{job}`), or follow the job stream.

use std::future::Future;
use std::sync::Arc;

use quack_core::analysis::tools::SharedDb;
use quack_core::config::Config;
use quack_core::ingestion;
use quack_core::jobs::{JobId, JobKind, JobQueue, JobResult, JobSpec, JobState, Lane};
use quack_core::llm;

use super::state::App;

/// Uploads a workspace may have queued or processing before another is
/// turned away with 503 and `Retry-After` (each holds its file's bytes in
/// memory until it runs).
pub(crate) const MAX_WAITING_UPLOADS: usize = 64;

/// How long a turned-away uploader is told to wait.
pub(crate) const UPLOAD_RETRY_SECONDS: u32 = 30;

/// The workspace's upload lane key.
pub(crate) fn upload_lane(workspace_id: &str) -> String {
    format!("ingest:{workspace_id}")
}

pub(crate) struct UploadJob {
    pub document_id: String,
    pub filename: String,
    pub data: Vec<u8>,
}

/// Queue an upload for processing and return its job id. The document is
/// already registered as `queued`; whatever happens to the job, it ends
/// `ready` or `error`, never stuck.
pub(crate) fn submit_upload(
    app: &App,
    workspace_id: &str,
    owner: Option<String>,
    db: SharedDb,
    job: UploadJob,
) -> JobId {
    let spec = JobSpec::new(JobKind::Ingest, job.filename.clone())
        .workspace(workspace_id)
        .owner(owner)
        .lane(Lane::new(
            upload_lane(workspace_id),
            app.config.server.workers_per_workspace,
        ));
    let config = app.config.clone();
    let workspace = workspace_id.to_owned();
    let document_id = job.document_id.clone();
    let worker_db = Arc::clone(&db);
    let id = app.jobs.submit(spec, move |ctx| async move {
        let cancel = ctx.cancel_token();
        let handle = tokio::runtime::Handle::current();
        // The blocking thread has no task-local: keep the job's priority.
        let priority = quack_core::priority::current_priority();
        let document_id = job.document_id.clone();
        // Parsing and DuckDB writes are blocking work; the embedding calls
        // inside need the runtime, so block on it from a blocking thread.
        let outcome = tokio::task::spawn_blocking({
            let db = Arc::clone(&worker_db);
            move || {
                handle.block_on(quack_core::priority::with_priority(
                    priority,
                    process(&config, &workspace, &db, job, &cancel),
                ))
            }
        })
        .await;
        match outcome {
            Ok(result) => result,
            Err(e) => {
                // The task died before `process` could record an outcome;
                // the document must not stay `processing` forever.
                tracing::error!(error = %e, document = %document_id, "upload worker task failed");
                let message = "the ingestion worker failed before finishing this file";
                mark_error(&worker_db, &document_id, message).await;
                Err(String::from(message))
            }
        }
    });
    when_cancelled_unstarted(&app.jobs, id, move || async move {
        mark_error(&db, &document_id, "cancelled before processing started").await;
    });
    id
}

/// Run `record` if job `id` is cancelled while still queued: its work
/// never ran, so whatever that work would have recorded at its end (a
/// document's error, a run's closing audit row) is recorded here.
pub(crate) fn when_cancelled_unstarted<F, Fut>(jobs: &JobQueue, id: JobId, record: F)
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let jobs = jobs.clone();
    tokio::spawn(async move {
        if let Some(job) = jobs.wait(id).await
            && job.state == JobState::Cancelled
            && job.started_at.is_none()
        {
            record().await;
        }
    });
}

/// Process the upload; a cancel stops it between steps or mid-embedding and
/// leaves the document `error: cancelled`.
async fn process(
    config: &Config,
    workspace_id: &str,
    db: &SharedDb,
    job: UploadJob,
    cancel: &llm::CancellationToken,
) -> JobResult {
    let model = match llm::optional_embedding_model(config).await {
        Ok(model) => model,
        Err(e) => {
            tracing::warn!(error = %e, document = %job.document_id, "upload fails: no embedding model");
            mark_error(db, &job.document_id, &e.to_string()).await;
            return Err(e.to_string());
        }
    };
    let result = ingestion::process_document(
        config,
        db,
        workspace_id,
        &job.document_id,
        &job.filename,
        &job.data,
        model.as_ref(),
        Some(cancel),
    )
    .await;
    match result {
        Ok(r) => {
            tracing::info!(document = %r.document_id, file = %r.filename, chunks = r.chunks_stored, "upload processed");
            Ok(match r.tables.as_slice() {
                [] => format!("{} chunks", r.chunks_stored),
                tables => format!("tables {}", tables.join(", ")),
            })
        }
        Err(e) => {
            tracing::warn!(document = %job.document_id, error = %e, "upload failed");
            Err(e.to_string())
        }
    }
}

/// Record `message` as the document's error, so a client polling it sees
/// `error` rather than `processing` without end.
async fn mark_error(db: &SharedDb, document_id: &str, message: &str) {
    let (id, text) = (document_id.to_owned(), message.to_owned());
    if let Err(mark) = db.run(move |db| db.mark_document_error(&id, &text)).await {
        tracing::error!(error = %mark, document = %document_id, "could not record the upload failure");
    }
}
