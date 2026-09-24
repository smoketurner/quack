//! Background work in `quack serve` goes through the process's one
//! [`JobQueue`](quack_core::jobs::JobQueue) (design doc 4.1): uploads,
//! graph extraction, the ontology document pass, and agent turns. Uploads
//! return 202 with the document id and the job id; each workspace's uploads
//! share a lane of `[server].workers_per_workspace`. Clients poll the
//! document, or the job (`GET .../jobs/{job}`), or follow the job stream.

use std::sync::Arc;

use quack_core::analysis::tools::SharedDb;
use quack_core::config::Config;
use quack_core::ids::{UserId, WorkspaceId};
use quack_core::ingestion::{NewFile, Processing};
use quack_core::jobs::{JobId, JobKind, JobResult, JobSpec, JobState, Lane, LaneKey};
use quack_core::llm::Embeddings;
use quack_core::progress::{ChunkDone, RunControl};

use super::state::App;

/// Uploads a workspace may have queued or processing before another is
/// turned away with 503 and `Retry-After` (each holds its file's bytes in
/// memory until it runs).
pub(crate) const MAX_WAITING_UPLOADS: usize = 64;

/// How long a turned-away uploader is told to wait.
pub(crate) const UPLOAD_RETRY_SECONDS: u32 = 30;

pub(crate) struct UploadJob {
    pub document_id: String,
    pub filename: String,
    pub data: Vec<u8>,
}

impl UploadJob {
    /// Queue the upload for processing and return its job id. The document
    /// is already registered as `queued`; whatever happens to the job, it
    /// ends `ready` or `error`, never stuck.
    pub(crate) fn submit(
        self,
        app: &App,
        workspace_id: &WorkspaceId,
        owner: Option<UserId>,
        db: SharedDb,
    ) -> JobId {
        let spec = JobSpec::new(JobKind::Ingest, self.filename.clone())
            .workspace(workspace_id.clone())
            .owner(owner)
            .lane(Lane::new(
                &LaneKey::Ingest(workspace_id.clone()),
                app.config.server.workers_per_workspace,
            ));
        let config = app.config.clone();
        let workspace = workspace_id.to_string();
        let document_id = self.document_id.clone();
        let worker_db = Arc::clone(&db);
        let id = app
            .jobs
            .submit(spec, move |ctx| async move {
                let progress = |done: ChunkDone| ctx.progress(done.done, done.total);
                let cancel = ctx.cancel_token();
                let control = RunControl {
                    progress: &progress,
                    cancel: Some(&cancel),
                };
                self.process(&config, &workspace, &worker_db, control).await
            })
            .id;
        // The work records its own outcome; a job that ends without running
        // it (cancelled while queued) or that died mid-way (a panic) must not
        // leave the document `queued` or `processing` forever.
        app.jobs.when_ended(id, move |ended| async move {
            let message = match ended.state {
                JobState::Cancelled if ended.never_started() => {
                    "cancelled before processing started"
                }
                JobState::Cancelled => "cancelled",
                JobState::Failed => "the ingestion worker failed before finishing this file",
                JobState::Queued | JobState::Running | JobState::Succeeded => return,
            };
            Self::mark_unfinished(&db, &document_id, message).await;
        });
        id
    }

    /// Process the upload, reporting each embedded batch; a cancel stops it
    /// between steps or mid-embedding and leaves the document
    /// `error: cancelled`.
    async fn process(
        self,
        config: &Config,
        workspace_id: &str,
        db: &SharedDb,
        control: RunControl<'_>,
    ) -> JobResult {
        let model = match Embeddings::from_config(config).await {
            Ok(model) => model,
            Err(e) => {
                tracing::warn!(error = %e, document = %self.document_id, "upload fails: no embedding model");
                Self::mark_error(db, &self.document_id, &e.to_string()).await;
                return Err(e.to_string());
            }
        };
        let file = NewFile::new(&self.filename, &self.data).control(control);
        let result = Processing {
            config,
            db,
            workspace_id,
            document_id: &self.document_id,
            file: &file,
            embedder: model.as_ref(),
        }
        .run()
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
                tracing::warn!(document = %self.document_id, error = %e, "upload failed");
                Err(e.to_string())
            }
        }
    }

    /// Record `message` as the document's error, so a client polling it
    /// sees `error` rather than `processing` without end.
    async fn mark_error(db: &SharedDb, document_id: &str, message: &str) {
        let (id, text) = (document_id.to_owned(), message.to_owned());
        if let Err(mark) = db.run(move |db| db.mark_document_error(&id, &text)).await {
            tracing::error!(error = %mark, document = %document_id, "could not record the upload failure");
        }
    }

    /// [`Self::mark_error`] for a document the work left `queued` or
    /// `processing`; one it finished (ready, or failed with its own
    /// message) is left alone.
    async fn mark_unfinished(db: &SharedDb, document_id: &str, message: &str) {
        let (id, text) = (document_id.to_owned(), message.to_owned());
        let marked = db
            .run(move |db| {
                let unfinished = db
                    .document(&id)?
                    .is_some_and(|doc| doc.status.is_in_flight());
                if unfinished {
                    db.mark_document_error(&id, &text)?;
                }
                Ok(unfinished)
            })
            .await;
        match marked {
            Ok(true) => {
                tracing::warn!(document = %document_id, reason = message, "upload ended unfinished");
            }
            Ok(false) => {}
            Err(e) => {
                tracing::error!(error = %e, document = %document_id, "could not record the upload failure");
            }
        }
    }
}
