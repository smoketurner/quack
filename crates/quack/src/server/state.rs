//! What every handler shares: the config, the control plane, one open
//! `DuckDB` handle per workspace, the browser sessions, and the work queue.

use quack_core::ids::{UserId, WorkspaceId};
use quack_core::storage::writer::Writer;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::config::Config;
use quack_core::jobs::JobQueue;
use quack_core::storage::audit::AuditLog;
use quack_core::storage::workspace::WorkspaceDb;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::{StreamableHttpServerConfig, StreamableHttpService};

use super::error::{ApiError, ApiResult};
use super::oidc::Oidc;
use super::queue::UploadJob;
use super::resource::ProtectedResource;
use super::web::flash::Flashes;
use crate::mcp::McpServer;
use quack_core::error::{Error as CoreError, Result as CoreResult};
use quack_core::storage::control::ControlPlane;
use quack_core::web_sessions::WebSessions;

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

/// How the server knows who is asking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ServeMode {
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
    /// carries its own MCP sessions. See `server::mcp_http`.
    mcp: tokio::sync::Mutex<HashMap<McpKey, McpTransport>>,
    /// Workspaces with a graph extraction in flight: one at a time each,
    /// so a reset cannot clear a run part way (issue #48).
    extractions: Mutex<HashSet<WorkspaceId>>,
    /// Messages between a form's redirect and the page it lands on.
    pub flashes: Flashes,
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
        }
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
                .with_json_response(true),
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
