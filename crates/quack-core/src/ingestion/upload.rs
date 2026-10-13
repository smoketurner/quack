//! Uploads queued for processing (design doc 4.1). An upload is registered
//! as `queued`, its bytes are spooled into the workspace's `uploads/`
//! directory, and a job in the workspace's ingest lane (at most
//! `[server].workers_per_workspace` at a time) processes it from there, so
//! a queue of any depth costs disk rather than the process's memory. The
//! spooled file goes when the job ends, however it ends; whatever happens,
//! the document ends `ready` or `error`, never stuck.

use std::fmt;
use std::future::Future;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::Arc;

use crate::analysis::tools::SharedDb;
use crate::config::Config;
use crate::error::Result;
use crate::ids::{DocumentId, UserId, WorkspaceId};
use crate::ingestion::parser::PageCounts;
use crate::ingestion::{NewFile, Processing};
use crate::jobs::{JobId, JobKind, JobQueue, JobResult, JobSpec, JobState, Lane, LaneKey};
use crate::llm::Embeddings;
use crate::progress::{ChunkDone, RunControl};

/// A queued upload's bytes, kept in its workspace's `uploads/` directory
/// from the request until its job ends.
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

    /// Whether the spooled file is still there to be read.
    async fn check(&self) -> std::io::Result<()> {
        tokio::fs::File::open(&self.0).await.map(drop)
    }

    async fn remove(&self) {
        if let Err(e) = tokio::fs::remove_file(&self.0).await
            && e.kind() != ErrorKind::NotFound
        {
            tracing::warn!(path = %self.0.display(), error = %e, "could not remove a spooled upload");
        }
    }
}

/// Where a workspace's uploads queue: its job queue and lane width (from
/// the configuration), and whose uploads they are.
#[derive(Clone)]
pub struct Uploads {
    pub config: Config,
    pub jobs: JobQueue,
    pub workspace: WorkspaceId,
    /// The person the job list shows as the owner.
    pub owner: Option<UserId>,
}

/// One registered upload, its bytes spooled, ready to queue.
pub struct UploadJob {
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
    pub fn clear_stale(config: &Config, workspace_id: &WorkspaceId) -> std::io::Result<()> {
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
    pub async fn spool(
        config: &Config,
        workspace_id: &WorkspaceId,
        db: &SharedDb,
        document_id: DocumentId,
        filename: String,
        data: &[u8],
    ) -> Result<Self> {
        match Spooled::write(config, workspace_id, &document_id, data).await {
            Ok(spool) => Ok(Self {
                document_id,
                filename,
                spool,
            }),
            Err(e) => {
                let message = format!("the upload could not be stored for processing: {e}");
                Self::mark_error(db, &document_id, &message).await;
                Err(std::io::Error::new(e.kind(), message).into())
            }
        }
    }

    /// Queue the upload for processing and return its job id. The document
    /// is already registered as `queued`; whatever happens to the job, it
    /// ends `ready` or `error`, never stuck. A document that becomes ready
    /// is handed to `follow`, which queues the graph follow-up
    /// `[graph].follow_ingest` asks for (or nothing, `Ok(None)`); the job's
    /// message notes the follow-up job or why none was queued.
    pub fn submit<F, Fut, E>(
        self,
        uploads: &Uploads,
        db: SharedDb,
        embedder: Option<Embeddings>,
        follow: F,
    ) -> JobId
    where
        F: FnOnce(Vec<DocumentId>) -> Fut + Send + 'static,
        Fut: Future<Output = std::result::Result<Option<JobId>, E>> + Send,
        E: fmt::Display,
    {
        let workspace_id = &uploads.workspace;
        let spec = JobSpec::new(JobKind::Ingest, self.filename.clone())
            .workspace(workspace_id.clone())
            .owner(uploads.owner.clone())
            .lane(Lane::new(
                &LaneKey::Workspace(JobKind::Ingest, workspace_id.clone()),
                uploads.config.server.workers_per_workspace,
            ));
        let config = uploads.config.clone();
        let workspace = workspace_id.to_string();
        let document_id = self.document_id.clone();
        let spool = self.spool.clone();
        let worker_db = Arc::clone(&db);
        let id = uploads
            .jobs
            .submit(spec, move |ctx| async move {
                let progress = |done: ChunkDone| ctx.progress(done.done, done.total);
                let cancel = ctx.cancel_token();
                let control = RunControl {
                    progress: &progress,
                    cancel: Some(&cancel),
                };
                let ready = self.document_id.clone();
                let mut message = self
                    .process(&config, &workspace, &worker_db, embedder.as_ref(), control)
                    .await?;
                let note = match follow(vec![ready]).await {
                    Ok(Some(job)) => format!("; graph follow-up queued as job {job}"),
                    Ok(None) => String::new(),
                    Err(e) => {
                        tracing::warn!(error = %e, "the graph follow-up could not be queued");
                        format!("; graph follow-up not queued: {e}")
                    }
                };
                message.push_str(&note);
                Ok(message)
            })
            .id;
        // The work records its own outcome; a job that ends without running
        // it (cancelled while queued) or that died mid-way (a panic) must not
        // leave the document `queued` or `processing` forever.
        uploads.jobs.when_ended(id, move |ended| async move {
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
        if let Err(e) = self.spool.check().await {
            let message = format!("the spooled upload could not be read: {e}");
            tracing::warn!(document = %self.document_id, error = %e, "upload fails: spool unreadable");
            Self::mark_error(db, &self.document_id, &message).await;
            return Err(message);
        }
        let file = NewFile::at_path(&self.filename, &self.spool.0).control(control);
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
