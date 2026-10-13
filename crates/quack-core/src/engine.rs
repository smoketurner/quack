//! The process: its one work queue, its open workspaces, and its stop.
//!
//! `DuckDB`'s file lock is advisory and per process: a second open of one
//! workspace file inside the process succeeds and yields an independent
//! database whose writes silently overwrite the first's. [`Engine::workspace`]
//! is where one open per file is enforced (design doc section 7.4): a
//! [`WorkspaceHandle`] is built once behind a cell and every interface in the
//! process shares it. `quack serve`, the terminal session, print mode, and
//! the stdio MCP server each build one engine and hand it to what they run.

use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::Duration;

use tokio::sync::{Mutex, OnceCell, broadcast};
use tokio_util::sync::CancellationToken;

use crate::analysis::tools::{ReaderDb, SharedDb};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::forward::Holder;
use crate::ids::WorkspaceId;
use crate::ingestion::upload::UploadJob;
use crate::jobs::store as jobs_store;
use crate::jobs::{JobId, JobInfo, JobQueue, JobState};
use crate::priority::Priority;
use crate::storage::audit::AuditLog;
use crate::storage::workspace::WorkspaceDb;
use crate::storage::writer::Writer;
use crate::telemetry;

/// What one `quack` process shares between its interfaces.
pub struct Engine {
    pub config: Config,
    /// Every background job: uploads, extraction and proposal runs, agent
    /// turns, session titles. Its registry is in memory, so workspace
    /// content in a job's label never reaches `control.db`; each
    /// workspace's jobs are kept in its own file by the engine's recorder.
    pub jobs: JobQueue,
    holder: Holder,
    /// Cancelled when the process begins to stop: long-lived streams end on
    /// it, so a connection left open cannot hold the process up.
    stopping: CancellationToken,
    /// The task writing each job's state into its workspace's file.
    recorder: Recorder,
    /// A workspace file is opened once per process; every interface shares
    /// it. The cell is what enforces "once"; `None` once [`Engine::close`]
    /// ran, so nothing opens a file again beside a handle a straggler
    /// still holds.
    workspaces: Mutex<Option<HashMap<WorkspaceId, Arc<OnceCell<WorkspaceHandle>>>>>,
}

/// A workspace's writer connection plus its reader pool and its audit
/// connection, opened together so each is built once per workspace rather
/// than once per request (a request must never wait behind a slow write on
/// the writer to read or to record its audit detail).
#[derive(Clone)]
pub struct WorkspaceHandle {
    pub writer: SharedDb,
    pub reader: ReaderDb,
    /// The insert-only `_quack_audit` connection; only the server writes
    /// through it.
    pub audit: Arc<AuditLog>,
}

/// The task that keeps `_quack_jobs` current: it reads every snapshot the
/// queue broadcasts and writes the job's row into its workspace's file
/// (`jobs::store`) when the job's state changed, at background priority on
/// that workspace's writer. A workspace that is not open gets no row (it
/// is never opened for one: a delete's close races a last snapshot).
/// [`Engine::close`] tells it to drain and waits for it.
struct Recorder {
    closing: CancellationToken,
    task: std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

/// How long [`Engine::close`] waits for the recorder to write what is
/// pending.
const RECORDER_DRAIN: Duration = Duration::from_secs(5);

/// The most jobs one workspace's file puts back into the queue at open:
/// the queue's history bound is per process, and a server opens many
/// workspaces. The file keeps `[jobs].history` of them.
const RESTORED_PER_WORKSPACE: u32 = 20;

impl Recorder {
    /// Run over `events` until the queue closes or `closing` fires, then
    /// write what is still pending.
    async fn run(
        engine: Weak<Engine>,
        mut events: broadcast::Receiver<JobInfo>,
        closing: CancellationToken,
    ) {
        // What was last written per active job: a snapshot that changed
        // only progress or a status line is not written again.
        let mut written: HashMap<JobId, (JobState, bool)> = HashMap::new();
        loop {
            let next = tokio::select! {
                biased;
                () = closing.cancelled() => break,
                next = events.recv() => next,
            };
            match next {
                Ok(job) => Self::record(&engine, &mut written, job).await,
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    let Some(engine) = engine.upgrade() else {
                        return;
                    };
                    written.clear();
                    for job in engine.jobs.list() {
                        Self::record(&Arc::downgrade(&engine), &mut written, job).await;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => return,
            }
        }
        while let Ok(job) = events.try_recv() {
            Self::record(&engine, &mut written, job).await;
        }
    }

    async fn record(
        engine: &Weak<Engine>,
        written: &mut HashMap<JobId, (JobState, bool)>,
        job: JobInfo,
    ) {
        let Some(engine) = engine.upgrade() else {
            return;
        };
        let Some(workspace_id) = job.workspace_id.as_ref() else {
            return;
        };
        let now = (job.state, job.cancel_requested);
        if written.get(&job.id) == Some(&now) {
            return;
        }
        let Some(handle) = engine.open_handle(workspace_id).await else {
            return;
        };
        if job.state.is_finished() {
            written.remove(&job.id);
        } else {
            written.insert(job.id, now);
        }
        let holder = engine.holder.to_string();
        let (id, workspace) = (job.id, workspace_id.clone());
        let wrote = Priority::Background
            .scope(
                handle
                    .writer
                    .run(move |db| jobs_store::upsert(db, &job, &holder)),
            )
            .await;
        if let Err(e) = wrote {
            tracing::debug!(job = %id, workspace = %workspace, error = %e, "the job's row was not written");
        }
    }
}

/// What follows work done with a workspace's file closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AfterClose {
    /// Open the file again for the handle's connections (a snapshot).
    Reopen,
    /// Leave it closed: the workspace is going (a delete).
    StayClosed,
}

/// Whether a handle's connections are open after [`WorkspaceHandle::with_file_closed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileState {
    Open,
    Closed,
}

impl WorkspaceHandle {
    /// Run `work` with every connection to the workspace file closed (the
    /// writer, the reader pool, the audit connection), since Windows lets no
    /// other handle open a `DuckDB` file in use (#448); then, for
    /// [`AfterClose::Reopen`], open it again through `WorkspaceDb::open` in
    /// place, so whoever holds this handle carries on with the new
    /// connections. Meanwhile writes and audit rows wait on their threads
    /// and reads on their locked connections.
    async fn with_file_closed<T, F>(
        &self,
        config: &Config,
        workspace_id: &WorkspaceId,
        after: AfterClose,
        work: F,
    ) -> (Result<T>, FileState)
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        // Always writer, then audit, then readers, so two of these never
        // wait on each other.
        let mut writer = match self.writer.lend().await {
            Ok(lease) => lease,
            Err(e) => return (Err(e), FileState::Open),
        };
        let mut audit = match self.audit.lend().await {
            Ok(lease) => lease,
            Err(e) => return (Err(e), FileState::Open),
        };
        let reader = self.reader.clone();
        let config = config.clone();
        let id = workspace_id.clone();
        let closed = tokio::task::spawn_blocking(move || {
            let mut readers = reader.lend();
            // A lease dropped unchanged gives its connection back.
            if let Err(e) = writer.db().and_then(WorkspaceDb::checkpoint) {
                return (Err(e), FileState::Open);
            }
            readers.close();
            audit.close();
            writer.close();
            let worked = work();
            if after == AfterClose::StayClosed {
                return (worked, FileState::Closed);
            }
            // On a failure everything stays closed, so a fresh open is the
            // only one in the process.
            let reopened = WorkspaceDb::open(&config, id.as_str())
                .and_then(|db| db.try_clone_reader().map(|clone| (db, clone)));
            let state = match reopened {
                Ok((db, clone)) => {
                    readers.restore(&db);
                    audit.restore(clone);
                    writer.restore(db);
                    FileState::Open
                }
                Err(e) => {
                    tracing::error!(workspace = %id, error = %e, "the workspace did not reopen");
                    FileState::Closed
                }
            };
            (worked, state)
        })
        .await;
        // Release builds abort on a panic, so a failed join is the runtime
        // shutting down, and each lease's drop hands back what it held.
        closed.unwrap_or_else(|e| {
            let failed = std::io::Error::other(format!("the closed-file task failed: {e}"));
            (Err(Error::Io(failed)), FileState::Open)
        })
    }
}

impl Engine {
    /// The process's engine, with the queue `[jobs]` configures. The job
    /// recorder starts here, on the Tokio runtime the caller is inside;
    /// outside one (a test that never submits a job) nothing is recorded.
    #[must_use]
    pub fn new(config: Config, holder: Holder) -> Arc<Self> {
        let jobs = JobQueue::from_config(&config.jobs);
        let events = jobs.subscribe();
        let closing = CancellationToken::new();
        let engine = Arc::new(Self {
            jobs,
            config,
            holder,
            stopping: CancellationToken::new(),
            recorder: Recorder {
                closing: closing.clone(),
                task: std::sync::Mutex::new(None),
            },
            workspaces: Mutex::new(Some(HashMap::new())),
        });
        let task = tokio::runtime::Handle::try_current()
            .ok()
            .map(|runtime| runtime.spawn(Recorder::run(Arc::downgrade(&engine), events, closing)));
        if let Ok(mut slot) = engine.recorder.task.lock() {
            *slot = task;
        }
        engine
    }

    /// The handle of a workspace that is open now; `None` opens nothing.
    async fn open_handle(&self, workspace_id: &WorkspaceId) -> Option<WorkspaceHandle> {
        let workspaces = self.workspaces.lock().await;
        workspaces
            .as_ref()
            .and_then(|map| map.get(workspace_id))
            .and_then(|cell| cell.get())
            .cloned()
    }

    /// The one workspace this process holds, or `None` for a server that
    /// opens any.
    #[must_use]
    pub fn held(&self) -> Option<&WorkspaceId> {
        self.holder.held()
    }

    /// What this process is to other `quack` processes.
    #[must_use]
    pub fn holder(&self) -> &Holder {
        &self.holder
    }

    /// A token cancelled when the process begins to stop: watch it, since a
    /// child cannot cancel the process.
    #[must_use]
    pub fn stopping(&self) -> CancellationToken {
        self.stopping.child_token()
    }

    /// Whether the process has begun to stop.
    #[must_use]
    pub fn is_stopping(&self) -> bool {
        self.stopping.is_cancelled()
    }

    /// The writer, reader pool, and audit connection for a workspace,
    /// opening the file on first use; a holder of one workspace is refused
    /// any other.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WorkspaceNotHeld`] for a workspace this holder did
    /// not open, [`Error::EngineClosed`] after [`Engine::close`], or the
    /// open's own error (another process holds the file, the file is too
    /// new, ...).
    pub async fn workspace(&self, workspace_id: &WorkspaceId) -> Result<WorkspaceHandle> {
        if let Some(held) = self.holder.held()
            && held != workspace_id
        {
            return Err(Error::WorkspaceNotHeld {
                holder: self.holder.to_string(),
            });
        }
        let cell = {
            let mut workspaces = self.workspaces.lock().await;
            let Some(workspaces) = workspaces.as_mut() else {
                return Err(Error::EngineClosed);
            };
            Arc::clone(
                workspaces
                    .entry(workspace_id.clone())
                    .or_insert_with(|| Arc::new(OnceCell::new())),
            )
        };
        let config = self.config.clone();
        let id = workspace_id.clone();
        let pool_size = self.config.analysis.reader_pool_size;
        let jobs = self.jobs.clone();
        cell.get_or_try_init(|| async move {
            let history = config.jobs.history;
            let (db, audit, earlier) = tokio::task::spawn_blocking(move || {
                let db = WorkspaceDb::open(&config, id.as_str())?;
                // Uploads a previous process took but never finished are
                // failed, and their spooled bytes deleted: the person who
                // sent them is told to send them again.
                let stale = db.fail_stale_uploads()?;
                if stale > 0 {
                    tracing::warn!(workspace = %id, stale, "failed uploads left queued by an earlier process");
                }
                if let Err(e) = UploadJob::clear_stale(&config, &id) {
                    tracing::warn!(workspace = %id, error = %e, "could not remove the spooled uploads an earlier process left");
                }
                // Jobs an earlier process left active are failed, and the
                // newest ones come back into this process's list.
                let interrupted = jobs_store::fail_interrupted(&db)?;
                if interrupted > 0 {
                    tracing::warn!(workspace = %id, interrupted, "failed jobs left active by an earlier process");
                }
                jobs_store::prune(&db, history)?;
                let earlier =
                    jobs_store::recent(&db, &id, history.min(RESTORED_PER_WORKSPACE))?;
                let audit = AuditLog::open(&db)?;
                Ok::<_, Error>((db, audit, earlier))
            })
            .await
            .map_err(|e| {
                Error::Io(std::io::Error::other(format!(
                    "workspace open task failed: {e}"
                )))
            })??;
            jobs.restore(earlier);
            let writer: SharedDb = Arc::new(Writer::spawn(db)?);
            let reader = ReaderDb::open(&writer, pool_size).await;
            Ok(WorkspaceHandle {
                writer,
                reader,
                audit: Arc::new(audit),
            })
        })
        .await
        .cloned()
    }

    /// Run `work` with the workspace's file closed, then reopen it (the
    /// writer, the reader pool, and the audit connection all let go, since
    /// Windows lets no other handle open a `DuckDB` file in use): this
    /// workspace's work waits for it, other workspaces do not. A file that
    /// does not reopen is forgotten, so the next use opens it afresh.
    ///
    /// # Errors
    ///
    /// Returns the open's error, a lease that could not be taken, or
    /// `work`'s own error.
    pub async fn with_workspace_closed<T, F>(
        &self,
        workspace_id: &WorkspaceId,
        work: F,
    ) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let handle = self.workspace(workspace_id).await?;
        let (done, state) = handle
            .with_file_closed(&self.config, workspace_id, AfterClose::Reopen, work)
            .await;
        if state == FileState::Closed {
            let cell = self.forget(workspace_id).await;
            drop(tokio::task::spawn_blocking(move || drop(cell)));
        }
        done
    }

    /// Take a workspace's cell out of the map, so the next use opens afresh.
    async fn forget(&self, workspace_id: &WorkspaceId) -> Option<Arc<OnceCell<WorkspaceHandle>>> {
        self.workspaces
            .lock()
            .await
            .as_mut()
            .and_then(|workspaces| workspaces.remove(workspace_id))
    }

    /// Whether a workspace may be closed now: not the holder's own, and no
    /// job of it queued or running. [`Engine::close_workspace`] asks this
    /// first; a caller with something to let go of before the close (the
    /// server's MCP transports) asks it before letting go.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WorkspaceHeld`] for the holder's own workspace or
    /// [`Error::WorkspaceBusy`] while a job of it is active.
    pub fn closable(&self, workspace_id: &WorkspaceId) -> Result<()> {
        if self.holder.held() == Some(workspace_id) {
            return Err(Error::WorkspaceHeld {
                holder: self.holder.to_string(),
            });
        }
        let active = self.jobs.counts(Some(workspace_id)).active();
        if active > 0 {
            return Err(Error::WorkspaceBusy { active });
        }
        Ok(())
    }

    /// Let go of one workspace before it is deleted: its open file, once no
    /// job of it is queued or running. Whoever still holds the handle
    /// finishes on the unlinked file.
    ///
    /// # Errors
    ///
    /// Returns what [`Engine::closable`] refuses, or the close's own error.
    pub async fn close_workspace(&self, workspace_id: &WorkspaceId) -> Result<()> {
        self.closable(workspace_id)?;
        let cell = self.forget(workspace_id).await;
        // Work that still holds the handle fails from here on rather than
        // keep the file open under the delete.
        if let Some(handle) = cell.as_ref().and_then(|cell| cell.get()) {
            let (closed, _) = handle
                .with_file_closed(
                    &self.config,
                    workspace_id,
                    AfterClose::StayClosed,
                    || Ok(()),
                )
                .await;
            closed?;
        }
        tokio::task::spawn_blocking(move || drop(cell))
            .await
            .map_err(|e| {
                Error::Io(std::io::Error::other(format!(
                    "closing the workspace failed: {e}"
                )))
            })
    }

    /// Read the gauges `GET /metrics` reports: jobs by kind and state, the
    /// writers' queues, and the open workspaces.
    pub async fn refresh_gauges(&self) {
        for (kind, state, count) in self.jobs.tally() {
            telemetry::set_jobs(kind.as_str(), state.as_str(), count);
        }
        let (mut interactive, mut background) = (0_usize, 0_usize);
        let open = {
            let workspaces = self.workspaces.lock().await;
            let mut open = 0_usize;
            for cell in workspaces.iter().flat_map(HashMap::values) {
                if let Some(handle) = cell.get() {
                    open = open.saturating_add(1);
                    let (i, b) = handle.writer.waiting();
                    interactive = interactive.saturating_add(i);
                    background = background.saturating_add(b);
                }
            }
            open
        };
        telemetry::set_writer_waiting("interactive", interactive);
        telemetry::set_writer_waiting("background", background);
        telemetry::set_open_workspaces(open);
    }

    /// Begin to stop: everything watching [`Self::stopping`] ends (a
    /// server's drain starts, its streams end). Whoever owns the queue's
    /// shutdown calls [`Self::stop_jobs`] next.
    pub fn begin_stop(&self) {
        self.stopping.cancel();
    }

    /// Begin to stop, and shut the queue down under `grace` (a later submit
    /// is recorded as cancelled and never runs). Returns the jobs still
    /// active when the grace ran out. A second call returns at once with
    /// what is still active.
    pub async fn stop_jobs(&self, grace: Duration) -> Vec<JobInfo> {
        self.begin_stop();
        self.jobs.shutdown(grace).await
    }

    /// Begin to stop, and let the queue's jobs end on their own under
    /// `grace` (a later submit is refused; nothing running is cancelled):
    /// what a command that answered one question does before it exits, so
    /// the title and labelling its turn started finish.
    pub async fn drain_jobs(&self, grace: Duration) -> Vec<JobInfo> {
        self.begin_stop();
        self.jobs.drain(grace).await
    }

    /// Drop every workspace handle, so each writer finishes its queued
    /// closures and checkpoints; the last thing a process does. Begins the
    /// stop first, so no path leaves the token unfired, and refuses every
    /// later [`Self::workspace`] (`Error::EngineClosed`).
    pub async fn close(&self) {
        self.begin_stop();
        // The last snapshots reach the files before they close.
        self.recorder.closing.cancel();
        let task = self
            .recorder
            .task
            .lock()
            .ok()
            .and_then(|mut slot| slot.take());
        if let Some(task) = task
            && tokio::time::timeout(RECORDER_DRAIN, task).await.is_err()
        {
            tracing::warn!("the job recorder did not finish writing before the workspaces closed");
        }
        let workspaces = self.workspaces.lock().await.take();
        // A writer's drop joins its thread.
        let closed = tokio::task::spawn_blocking(move || drop(workspaces));
        if let Err(e) = closed.await {
            tracing::error!(error = %e, "closing the workspaces failed");
        }
    }
}

#[cfg(test)]
mod tests;
