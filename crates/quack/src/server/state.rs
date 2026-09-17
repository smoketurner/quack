//! What every handler shares: the config, the control plane, one open
//! `DuckDB` handle per workspace, the browser sessions, and the upload queue.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use quack_core::analysis::tools::SharedDb;
use quack_core::config::Config;
use quack_core::storage::workspace::WorkspaceDb;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::{StreamableHttpServerConfig, StreamableHttpService};

use super::error::{ApiError, ApiResult};
use super::queue::UploadQueue;
use quack_core::storage::control::{ControlPlane, random_bytes};

pub(crate) struct AppState {
    pub config: Config,
    pub control: ControlPlane,
    /// `--local`: no authentication, one implicit owner of everything.
    pub local: bool,
    /// A workspace file is opened once per process; every request shares it.
    workspaces: tokio::sync::Mutex<HashMap<String, SharedDb>>,
    /// Browser and API-login sessions: token -> user id. Cleared on restart.
    web_sessions: Mutex<HashMap<String, String>>,
    pub queue: UploadQueue,
    /// One MCP transport per workspace, user, and write permission; each
    /// carries its own MCP sessions. See `server::mcp_http`.
    mcp: tokio::sync::Mutex<HashMap<String, (McpTransport, crate::mcp::McpServer)>>,
    /// Workspaces with a graph extraction in flight: one at a time each,
    /// so a reset cannot clear a run part way (issue #48).
    extractions: Mutex<HashSet<String>>,
}

/// Holds a workspace's extraction slot; dropping it frees the slot.
pub(crate) struct ExtractionSlot {
    app: App,
    workspace_id: String,
}

impl Drop for ExtractionSlot {
    fn drop(&mut self) {
        if let Ok(mut running) = self.app.extractions.lock() {
            running.remove(&self.workspace_id);
        }
    }
}

pub(crate) type McpTransport = StreamableHttpService<crate::mcp::McpServer, LocalSessionManager>;

pub(crate) type App = Arc<AppState>;

impl AppState {
    pub(crate) fn new(config: Config, control: ControlPlane, local: bool) -> Self {
        Self {
            queue: UploadQueue::new(config.server.workers_per_workspace),
            config,
            control,
            local,
            workspaces: tokio::sync::Mutex::new(HashMap::new()),
            web_sessions: Mutex::new(HashMap::new()),
            mcp: tokio::sync::Mutex::new(HashMap::new()),
            extractions: Mutex::new(HashSet::new()),
        }
    }

    /// Claim the workspace's extraction slot, or `None` while another
    /// extraction runs there.
    pub(crate) fn begin_extraction(self: &Arc<Self>, workspace_id: &str) -> Option<ExtractionSlot> {
        let mut running = self.extractions.lock().ok()?;
        if !running.insert(workspace_id.to_owned()) {
            return None;
        }
        Some(ExtractionSlot {
            app: Arc::clone(self),
            workspace_id: workspace_id.to_owned(),
        })
    }

    /// The MCP transport for `key`, built with `make` on first use.
    pub(crate) async fn mcp_transport(
        &self,
        key: &str,
        make: impl FnOnce() -> crate::mcp::McpServer,
    ) -> (McpTransport, crate::mcp::McpServer) {
        let mut open = self.mcp.lock().await;
        if let Some(entry) = open.get(key) {
            return entry.clone();
        }
        let server = make();
        let factory = server.clone();
        let transport = StreamableHttpService::new(
            move || Ok(factory.clone()),
            Arc::new(LocalSessionManager::default()),
            StreamableHttpServerConfig::default()
                // The server may sit behind any host name; bearer auth,
                // not the Host header, is what guards it.
                .with_allowed_hosts(Vec::<String>::new())
                .with_json_response(true),
        );
        open.insert(key.to_owned(), (transport.clone(), server.clone()));
        (transport, server)
    }

    /// The shared handle for a workspace, opening the file on first use.
    pub(crate) async fn workspace_db(&self, workspace_id: &str) -> ApiResult<SharedDb> {
        let mut open = self.workspaces.lock().await;
        if let Some(db) = open.get(workspace_id) {
            return Ok(Arc::clone(db));
        }
        let config = self.config.clone();
        let id = workspace_id.to_owned();
        let db = tokio::task::spawn_blocking(move || {
            let db = WorkspaceDb::open(&config, &id)?;
            // Uploads a previous process took but never finished cannot
            // be resumed: their bytes are gone with it.
            let stale = db.fail_stale_uploads()?;
            if stale > 0 {
                tracing::warn!(workspace = %id, stale, "failed uploads left queued by an earlier process");
            }
            Ok::<_, quack_core::error::Error>(db)
        })
        .await
        .map_err(|e| ApiError::internal(format!("workspace open task failed: {e}")))??;
        let shared: SharedDb = Arc::new(Mutex::new(db));
        open.insert(workspace_id.to_owned(), Arc::clone(&shared));
        Ok(shared)
    }

    /// Start a browser session for the user and return its token.
    pub(crate) fn open_web_session(&self, user_id: &str) -> ApiResult<String> {
        let mut bytes = [0u8; 32];
        aws_lc_rs_fill(&mut bytes)?;
        let token = format!(
            "qs_{}",
            base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, bytes)
        );
        self.web_sessions
            .lock()
            .map_err(|e| ApiError::internal(format!("session store poisoned: {e}")))?
            .insert(token.clone(), user_id.to_owned());
        Ok(token)
    }

    pub(crate) fn web_session_user(&self, token: &str) -> Option<String> {
        self.web_sessions.lock().ok()?.get(token).cloned()
    }

    pub(crate) fn close_web_session(&self, token: &str) {
        if let Ok(mut sessions) = self.web_sessions.lock() {
            sessions.remove(token);
        }
    }
}

fn aws_lc_rs_fill(bytes: &mut [u8]) -> ApiResult<()> {
    // The control plane exposes the random source it uses for tokens.
    random_bytes(bytes).map_err(|e| ApiError::internal(e.to_string()))
}

/// Run a closure against the workspace handle on the blocking pool, so a
/// slow query never stalls the runtime.
pub(crate) async fn with_db<T, F>(db: SharedDb, f: F) -> ApiResult<T>
where
    T: Send + 'static,
    F: FnOnce(&WorkspaceDb) -> quack_core::error::Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let guard = db
            .lock()
            .map_err(|e| quack_core::error::Error::Analysis(format!("mutex poisoned: {e}")))?;
        f(&guard)
    })
    .await
    .map_err(|e| ApiError::internal(format!("database task failed: {e}")))?
    .map_err(ApiError::from)
}
