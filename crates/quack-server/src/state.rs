//! What every handler shares: the config, the control plane, one open
//! `DuckDB` handle per workspace, the browser sessions, and the work queue.

use quack_core::ids::{UserId, WorkspaceId};
use quack_core::storage::writer::Writer;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::config::Config;
use quack_core::jobs::JobQueue;
use quack_core::llm::oauth::KeySource;
use quack_core::storage::audit::AuditLog;
use quack_core::storage::workspace::WorkspaceDb;
use quack_core::telemetry;
use quack_core::vault::Vault;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::{StreamableHttpServerConfig, StreamableHttpService};

use super::error::{ApiError, ApiResult};
use super::oidc::Oidc;
use super::permissions::Permissions;
use super::queue::UploadJob;
use super::resource::ProtectedResource;
use super::web::flash::Flashes;
use crate::mcp::McpServer;
use quack_core::error::{Error as CoreError, Result as CoreResult};
use quack_core::storage::control::ControlPlane;
use quack_core::web_sessions::WebSessions;
use tokio_util::sync::CancellationToken;

/// A workspace's writer connection plus its reader pool and its audit
/// connection, opened together so each is built once per workspace handle
/// rather than once per request (a request must never wait behind a slow
/// write on the writer to read or to record its audit detail).
#[derive(Clone)]
struct WorkspaceHandle {
    writer: SharedDb,
    reader: ReaderDb,
    audit: Arc<AuditLog>,
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
    ) -> (CoreResult<T>, FileState)
    where
        T: Send + 'static,
        F: FnOnce() -> CoreResult<T> + Send + 'static,
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
            (Err(CoreError::Io(failed)), FileState::Open)
        })
    }
}

/// How the server knows who is asking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeMode {
    /// Users log in with a password or send an API token.
    Login,
    /// `--local` or `[server].local`: no authentication, one implicit owner
    /// of everything; loopback only.
    Local,
}

pub(crate) struct AppState {
    pub config: Config,
    pub control: ControlPlane,
    pub mode: ServeMode,
    /// A workspace file is opened once per process; every request shares it.
    /// The cell is what enforces "once": `DuckDB`'s file lock is advisory
    /// and per-process, so two concurrent opens of one file both succeed
    /// and yield independent databases whose writes overwrite each other.
    workspaces:
        tokio::sync::Mutex<HashMap<WorkspaceId, Arc<tokio::sync::OnceCell<WorkspaceHandle>>>>,
    /// Browser and API-login sessions.
    pub sessions: Arc<WebSessions>,
    /// Sign-in through `[server.oidc]`'s issuer, when configured and not in
    /// local mode.
    pub oidc: Option<Oidc>,
    /// What quack publishes as a protected resource, when it accepts the
    /// issuer's access tokens (`[server.oidc].audience`).
    pub resource: Option<ProtectedResource>,
    /// Every background job: uploads, extraction and proposal runs, agent
    /// turns. Its registry is in memory, so workspace content in a job's
    /// label never reaches `control.db`.
    pub jobs: JobQueue,
    /// One MCP transport per workspace, user, and write permission; each
    /// carries its own MCP sessions. See `mcp_http`.
    mcp: tokio::sync::Mutex<HashMap<McpKey, McpTransport>>,
    /// Workspaces with a graph extraction in flight: one at a time each,
    /// so a reset cannot clear a run part way (issue #48).
    extractions: Mutex<HashSet<WorkspaceId>>,
    /// Messages between a form's redirect and the page it lands on.
    pub flashes: Flashes,
    /// Writes streamed turns are waiting on a person to decide.
    pub permissions: Permissions,
    /// Cancelled when the server begins to stop: long-lived streams end on
    /// it, so a connection left open cannot hold the process up.
    pub stopping: CancellationToken,
    /// The last readiness answer and when it was worked out: `/readyz` is
    /// open to anyone and outside the limiter, so its checks run at most
    /// once per [`READINESS_FRESH`].
    readiness: tokio::sync::Mutex<Option<(Instant, Readiness)>>,
}

/// How long a readiness answer is served again before it is checked anew.
const READINESS_FRESH: Duration = Duration::from_secs(2);

/// What `GET /readyz` answers: each component `ok` or `fail`; why one
/// failed goes to the server's log, not to whoever asked.
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct Readiness {
    pub control_db: Probe,
    pub data_dir: Probe,
    pub vault_key: Probe,
}

impl Readiness {
    pub(crate) fn ready(&self) -> bool {
        [&self.control_db, &self.data_dir, &self.vault_key]
            .into_iter()
            .all(|p| matches!(p, Probe::Ok))
    }
}

/// One readiness component.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub(crate) enum Probe {
    Ok,
    Fail,
}

impl Probe {
    fn of(component: &str, outcome: Result<(), String>) -> Self {
        match outcome {
            Ok(()) => Self::Ok,
            Err(error) => {
                tracing::warn!(component, error, "not ready");
                Self::Fail
            }
        }
    }
}

/// Holds a workspace's extraction slot; dropping it frees the slot.
pub(crate) struct ExtractionSlot {
    app: App,
    workspace_id: WorkspaceId,
}

impl Drop for ExtractionSlot {
    fn drop(&mut self) {
        if let Ok(mut running) = self.app.extractions.lock() {
            running.remove(&self.workspace_id);
        }
    }
}

pub(crate) type McpTransport = StreamableHttpService<McpServer, LocalSessionManager>;

/// Which MCP transport a request belongs to: one per workspace, user, and
/// write permission, so its audit rows and its writes are that caller's.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct McpKey {
    pub workspace_id: WorkspaceId,
    pub user_id: UserId,
    pub policy: WritePolicy,
}

pub(crate) type App = Arc<AppState>;

impl AppState {
    pub(crate) fn new(
        config: Config,
        control: ControlPlane,
        mode: ServeMode,
        sessions: Arc<WebSessions>,
        oidc: Option<Oidc>,
    ) -> Self {
        let resource = ProtectedResource::of(&config, mode);
        // The recorder lives for the process; a second state in one
        // process (tests) keeps the first.
        telemetry::install();
        Self {
            jobs: JobQueue::from_config(&config.jobs),
            config,
            control,
            mode,
            workspaces: tokio::sync::Mutex::new(HashMap::new()),
            sessions,
            resource,
            oidc,
            mcp: tokio::sync::Mutex::new(HashMap::new()),
            extractions: Mutex::new(HashSet::new()),
            flashes: Flashes::default(),
            permissions: Permissions::default(),
            stopping: CancellationToken::new(),
            readiness: tokio::sync::Mutex::new(None),
        }
    }

    /// Stop the server whose HTTP drain is `server`: streams end and the
    /// job queue shuts down beside the drain, under one grace of
    /// `[server].shutdown_grace_seconds`, and the workspaces close last.
    ///
    /// # Errors
    ///
    /// Returns the drain's own error.
    pub(crate) async fn stop<S>(self: Arc<Self>, mut server: S) -> std::io::Result<()>
    where
        S: Future<Output = std::io::Result<()>> + Unpin,
    {
        self.stopping.cancel();
        let grace = self.config.server.shutdown_grace();
        let deadline = Instant::now().checked_add(grace);
        let drain = async {
            let (served, _) = tokio::join!(&mut server, self.jobs.shutdown(grace));
            served
        };
        let served = tokio::time::timeout(grace, drain)
            .await
            .unwrap_or_else(|_| {
                tracing::warn!(
                    grace_seconds = grace.as_secs(),
                    "requests were still open when the shutdown grace ran out"
                );
                Ok(())
            });
        drop(server);
        // A request that ended during the drain may have been refused a job;
        // what that job's end records is awaited in what is left of the grace.
        let left = deadline.map_or(Duration::ZERO, |at| {
            at.saturating_duration_since(Instant::now())
        });
        for job in self.jobs.shutdown(left).await {
            tracing::warn!(job = %job.id, kind = %job.kind, state = %job.state, "job still active at exit");
        }
        self.close().await;
        served
    }

    /// Let go of every MCP transport and workspace, once requests and jobs
    /// have ended: each workspace's writer finishes its queued closures and
    /// checkpoints as its last handle drops. The transports go explicitly
    /// because their servers hold this state, which would otherwise never
    /// drop.
    pub(crate) async fn close(&self) {
        let transports = std::mem::take(&mut *self.mcp.lock().await);
        let workspaces = std::mem::take(&mut *self.workspaces.lock().await);
        // A writer's drop joins its thread.
        let closed = tokio::task::spawn_blocking(move || {
            drop(transports);
            drop(workspaces);
        });
        if let Err(e) = closed.await {
            tracing::error!(error = %e, "closing the workspaces failed");
        }
    }

    /// Let go of one workspace before it is deleted: its MCP transports
    /// and its open file, once no job of it is queued or running. A
    /// request that still holds a handle finishes on the unlinked file.
    pub(crate) async fn close_workspace(&self, workspace_id: &WorkspaceId) -> ApiResult<()> {
        let active = self.jobs.counts(Some(workspace_id)).active();
        if active > 0 {
            return Err(ApiError::conflict(format!(
                "{active} job(s) of this workspace are queued or running; cancel them first"
            )));
        }
        let transports = {
            let mut mcp = self.mcp.lock().await;
            let keys: Vec<McpKey> = mcp
                .keys()
                .filter(|key| key.workspace_id == *workspace_id)
                .cloned()
                .collect();
            keys.iter()
                .filter_map(|key| mcp.remove(key))
                .collect::<Vec<_>>()
        };
        let cell = self.workspaces.lock().await.remove(workspace_id);
        // A request that still holds the handle fails from here on rather
        // than keep the file open under the delete.
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
        let closed = tokio::task::spawn_blocking(move || {
            drop(transports);
            drop(cell);
        });
        closed
            .await
            .map_err(|e| ApiError::internal(format!("closing the workspace failed: {e}")))
    }

    /// Run `work` with the workspace's file closed, then reopen it (see
    /// [`WorkspaceHandle::with_file_closed`]): this workspace's requests
    /// wait for it, other workspaces do not. A file that does not reopen
    /// is forgotten, so the next request opens it afresh.
    pub(crate) async fn with_workspace_closed<T, F>(
        &self,
        workspace_id: &WorkspaceId,
        work: F,
    ) -> ApiResult<T>
    where
        T: Send + 'static,
        F: FnOnce() -> CoreResult<T> + Send + 'static,
    {
        let handle = self.workspace_handle(workspace_id).await?;
        let (done, state) = handle
            .with_file_closed(&self.config, workspace_id, AfterClose::Reopen, work)
            .await;
        if state == FileState::Closed {
            let cell = self.workspaces.lock().await.remove(workspace_id);
            drop(tokio::task::spawn_blocking(move || drop(cell)));
        }
        Ok(done?)
    }

    /// Whether this server can serve: `control.db` answers, the data
    /// directory takes a write, and the vault key is where it should be.
    pub(crate) async fn readiness(&self) -> Readiness {
        let mut last = self.readiness.lock().await;
        if let Some((at, answer)) = last.as_ref()
            && at.elapsed() < READINESS_FRESH
        {
            return answer.clone();
        }
        let answer = self.check_readiness().await;
        *last = Some((Instant::now(), answer.clone()));
        answer
    }

    async fn check_readiness(&self) -> Readiness {
        let control_db = self.control.ping().await.map_err(|e| e.to_string());
        let data_dir = {
            let probe = self
                .config
                .data_dir()
                .join(format!(".readyz-{}", uuid::Uuid::now_v7()));
            tokio::fs::write(&probe, b"ok")
                .await
                .and_then(|()| std::fs::remove_file(&probe))
                .map_err(|e| e.to_string())
        };
        let vault_key = Vault::new(self.config.data_dir(), KeySource::Keychain)
            .key_location()
            .await
            .map(|_| ())
            .map_err(|e| e.to_string());
        Readiness {
            control_db: Probe::of("control_db", control_db),
            data_dir: Probe::of("data_dir", data_dir),
            vault_key: Probe::of("vault_key", vault_key),
        }
    }

    /// Read the gauges `GET /metrics` reports: jobs by kind and state, the
    /// writers' queues, and the open workspaces.
    pub(crate) async fn refresh_gauges(&self) {
        for (kind, state, count) in self.jobs.tally() {
            telemetry::set_jobs(kind.as_str(), state.as_str(), count);
        }
        let (mut interactive, mut background) = (0_usize, 0_usize);
        let open = {
            let workspaces = self.workspaces.lock().await;
            let mut open = 0_usize;
            for cell in workspaces.values() {
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

    /// Claim the workspace's extraction slot, or `None` while another
    /// extraction runs there.
    pub(crate) fn begin_extraction(
        self: &Arc<Self>,
        workspace_id: &WorkspaceId,
    ) -> Option<ExtractionSlot> {
        let mut running = self.extractions.lock().ok()?;
        if !running.insert(workspace_id.clone()) {
            return None;
        }
        Some(ExtractionSlot {
            app: Arc::clone(self),
            workspace_id: workspace_id.clone(),
        })
    }

    /// The MCP transport for `key`, built with `make` on first use.
    pub(crate) async fn mcp_transport(
        &self,
        key: McpKey,
        make: impl FnOnce() -> McpServer,
    ) -> McpTransport {
        let mut open = self.mcp.lock().await;
        if let Some(transport) = open.get(&key) {
            return transport.clone();
        }
        let server = make();
        let transport = StreamableHttpService::new(
            move || Ok(server.clone()),
            Arc::new(LocalSessionManager::default()),
            StreamableHttpServerConfig::default()
                // The server may sit behind any host name; bearer auth,
                // not the Host header, is what guards it.
                .with_allowed_hosts(Vec::<String>::new())
                .with_json_response(true)
                // Its standalone event streams end when the server stops.
                .with_cancellation_token(self.stopping.child_token()),
        );
        open.insert(key, transport.clone());
        transport
    }

    /// The writer and reader for a workspace, opening the file and building
    /// the reader once on first use.
    ///
    /// The map lock is held only long enough to hand out the workspace's
    /// cell; opening the file and cloning the reader pool happen outside
    /// it, so one workspace's first access never blocks another's. The cell
    /// is what makes the open happen exactly once — concurrent first-time
    /// callers await the same initialization instead of each opening the
    /// file. De-duplicating afterwards would not do: two opens would both
    /// have already run, and both would have written.
    ///
    /// A failed open is never remembered. `get_or_try_init` leaves the cell
    /// empty on error, so the next request retries rather than inheriting a
    /// permanent failure; nothing here may cache the error alongside it.
    async fn workspace_handle(&self, workspace_id: &WorkspaceId) -> ApiResult<WorkspaceHandle> {
        let cell = {
            let mut open = self.workspaces.lock().await;
            Arc::clone(
                open.entry(workspace_id.clone())
                    .or_insert_with(|| Arc::new(tokio::sync::OnceCell::new())),
            )
        };
        let config = self.config.clone();
        let id = workspace_id.clone();
        let pool_size = self.config.analysis.reader_pool_size;
        cell.get_or_try_init(|| async move {
            let (db, audit) = tokio::task::spawn_blocking(move || {
                let db = WorkspaceDb::open(&config, id.as_str())?;
                // Uploads a previous process took but never finished are
                // failed, and their spooled bytes deleted: the person who
                // sent them is told to send them again.
                let stale = db.fail_stale_uploads()?;
                if stale > 0 {
                    tracing::warn!(workspace = %id, stale, "failed uploads left queued by an earlier process");
                }
                UploadJob::clear_stale(&config, &id)?;
                let audit = AuditLog::open(&db)?;
                Ok::<_, CoreError>((db, audit))
            })
            .await
            .map_err(|e| ApiError::internal(format!("workspace open task failed: {e}")))??;
            let writer: SharedDb = Arc::new(
                Writer::spawn(db).map_err(|e| ApiError::internal(e.to_string()))?,
            );
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

    /// The writer handle for a workspace, opening the file on first use.
    pub(crate) async fn workspace_db(&self, workspace_id: &WorkspaceId) -> ApiResult<SharedDb> {
        Ok(self.workspace_handle(workspace_id).await?.writer)
    }

    /// The reader handle for a workspace (built once, alongside the
    /// writer, on first use): every read-only handler should prefer this
    /// over [`Self::workspace_db`] so it never queues behind a write.
    pub(crate) async fn reader_db(&self, workspace_id: &WorkspaceId) -> ApiResult<ReaderDb> {
        Ok(self.workspace_handle(workspace_id).await?.reader)
    }

    /// The workspace's insert-only audit connection, opened with its writer.
    pub(crate) async fn audit_log(&self, workspace_id: &WorkspaceId) -> ApiResult<Arc<AuditLog>> {
        Ok(self.workspace_handle(workspace_id).await?.audit)
    }

    /// Run a read on one of the workspace's reader connections, inside a
    /// read-only transaction: it never waits for a write in progress, and
    /// anything that would write is refused. Every read-only handler reads
    /// through this; writes go to the writer through [`with_db`].
    pub(crate) async fn read<T, F>(&self, workspace_id: &WorkspaceId, f: F) -> ApiResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&WorkspaceDb) -> CoreResult<T> + Send + 'static,
    {
        self.reader_db(workspace_id)
            .await?
            .with_db(f)
            .await
            .map_err(ApiError::from)
    }
}

/// Run a closure on the workspace's writer, at the calling task's priority,
/// and await it: no runtime worker ever waits on the connection.
pub(crate) async fn with_db<T, F>(db: SharedDb, f: F) -> ApiResult<T>
where
    T: Send + 'static,
    F: FnOnce(&WorkspaceDb) -> CoreResult<T> + Send + 'static,
{
    db.run(f).await.map_err(ApiError::from)
}
