//! Background work in `quack serve` goes through the process's one
//! [`JobQueue`](quack_core::jobs::JobQueue) (design doc 4.1): uploads,
//! graph extraction, the ontology document pass, and agent turns. Uploads
//! return 202 with the document id and the job id; each workspace's uploads
//! share a lane of `[server].workers_per_workspace`. Clients poll the
//! document, or the job (`GET .../jobs/{job}`), or follow the job stream.

use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::Arc;

use quack_core::analysis::tools::SharedDb;
use quack_core::config::Config;
use quack_core::ids::{DocumentId, UserId, WorkspaceId};
use quack_core::ingestion::parser::PageCounts;
use quack_core::ingestion::{NewFile, Processing};
use quack_core::jobs::{JobId, JobKind, JobResult, JobSpec, JobState, Lane, LaneKey};
use quack_core::llm::Embeddings;
use quack_core::progress::{ChunkDone, RunControl};

use super::error::{ApiError, ApiResult};
use super::state::App;

/// A queued upload's bytes, kept in its workspace's `uploads/` directory
/// from the request until its job ends, so a queue of any depth costs disk
/// rather than the server's memory.
#[derive(Clone)]
struct Spooled(PathBuf);

impl Spooled {
    async fn write(
        config: &Config,
        workspace_id: &WorkspaceId,
        document_id: &DocumentId,
        data: &[u8],
    ) -> std::io::Result<Self> {
        let dir = config.workspace_uploads_dir(workspace_id.as_str());
        tokio::fs::create_dir_all(&dir).await?;
        let path = dir.join(document_id.as_str());
        tokio::fs::write(&path, data).await?;
        Ok(Self(path))
    }

    async fn read(&self) -> std::io::Result<Vec<u8>> {
        tokio::fs::read(&self.0).await
    }

    async fn remove(&self) {
        if let Err(e) = tokio::fs::remove_file(&self.0).await
            && e.kind() != ErrorKind::NotFound
        {
            tracing::warn!(path = %self.0.display(), error = %e, "could not remove a spooled upload");
        }
    }
}

/// One registered upload, its bytes spooled, ready to queue.
pub(crate) struct UploadJob {
    document_id: DocumentId,
    filename: String,
    spool: Spooled,
}

impl UploadJob {
    /// Delete what an earlier process spooled and never processed: the
    /// workspace marks those documents failed as it opens
    /// (`WorkspaceDb::fail_stale_uploads`), so nothing will read them.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory exists and cannot be removed.
    pub(crate) fn clear_stale(config: &Config, workspace_id: &WorkspaceId) -> std::io::Result<()> {
        match std::fs::remove_dir_all(config.workspace_uploads_dir(workspace_id.as_str())) {
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
            other => other,
        }
    }

    /// Spool `data` for the document just registered as `queued`. A spool
    /// that cannot be written fails the document, so it never sits
    /// `queued` with no job to finish it.
    ///
    /// # Errors
    ///
    /// Returns the write's error after recording it on the document.
    pub(crate) async fn spool(
        app: &App,
        workspace_id: &WorkspaceId,
        db: &SharedDb,
        document_id: DocumentId,
        filename: String,
        data: &[u8],
    ) -> ApiResult<Self> {
        match Spooled::write(&app.config, workspace_id, &document_id, data).await {
            Ok(spool) => Ok(Self {
                document_id,
                filename,
                spool,
            }),
            Err(e) => {
                let message = format!("the upload could not be stored for processing: {e}");
                Self::mark_error(db, &document_id, &message).await;
                Err(ApiError::internal(message))
            }
        }
    }

    /// Queue the upload for processing and return its job id. The document
    /// is already registered as `queued`; whatever happens to the job, it
    /// ends `ready` or `error`, never stuck.
    pub(crate) fn submit(
        self,
        app: &App,
        workspace_id: &WorkspaceId,
        owner: Option<UserId>,
        db: SharedDb,
        embedder: Option<Embeddings>,
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
        let spool = self.spool.clone();
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
                self.process(&config, &workspace, &worker_db, embedder.as_ref(), control)
                    .await
            })
            .id;
        // The work records its own outcome; a job that ends without running
        // it (cancelled while queued) or that died mid-way (a panic) must not
        // leave the document `queued` or `processing` forever.
        app.jobs.when_ended(id, move |ended| async move {
            spool.remove().await;
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
        embedder: Option<&Embeddings>,
        control: RunControl<'_>,
    ) -> JobResult {
        let data = match self.spool.read().await {
            Ok(data) => data,
            Err(e) => {
                let message = format!("the spooled upload could not be read: {e}");
                tracing::warn!(document = %self.document_id, error = %e, "upload fails: spool unreadable");
                Self::mark_error(db, &self.document_id, &message).await;
                return Err(message);
            }
        };
        let file = NewFile::new(&self.filename, &data).control(control);
        let result = Processing {
            config,
            db,
            workspace_id,
            document_id: &self.document_id,
            file: &file,
            embedder,
        }
        .run()
        .await;
        match result {
            Ok(r) => {
                tracing::info!(document = %r.document_id, file = %r.filename, chunks = r.chunks_stored, "upload processed");
                let pages = PageCounts::suffix(r.pages);
                Ok(match r.tables.as_slice() {
                    [] => format!("{} chunks{pages}", r.chunks_stored),
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
    async fn mark_error(db: &SharedDb, document_id: &DocumentId, message: &str) {
        let (id, text) = (document_id.clone(), message.to_owned());
        if let Err(mark) = db.run(move |db| db.mark_document_error(&id, &text)).await {
            tracing::error!(error = %mark, document = %document_id, "could not record the upload failure");
        }
    }

    /// [`Self::mark_error`] for a document the work left `queued` or
    /// `processing`; one it finished (ready, or failed with its own
    /// message) is left alone.
    async fn mark_unfinished(db: &SharedDb, document_id: &DocumentId, message: &str) {
        let (id, text) = (document_id.clone(), message.to_owned());
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
