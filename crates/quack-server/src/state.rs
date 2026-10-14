//! What every handler shares: the engine (its work queue and open
//! workspaces), the control plane, the browser sessions, and the MCP
//! transports.

use axum::extract::DefaultBodyLimit;
use quack_core::analysis::permissions::Permissions;
use quack_core::ids::{UserId, WorkspaceId};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::engine::{Engine, WorkspaceHandle};
use quack_core::error::Error as CoreError;
use quack_core::llm::oauth::KeySource;
use quack_core::storage::audit::AuditLog;
use quack_core::storage::workspace::WorkspaceDb;
use quack_core::telemetry;
use quack_core::vault::Vault;

use super::error::{ApiError, ApiResult};
use super::oidc::Oidc;
use super::resource::ProtectedResource;
use super::web::flash::Flashes;
use quack_core::error::Result as CoreResult;
use quack_core::storage::control::ControlPlane;
use quack_core::web_sessions::WebSessions;
use quack_mcp::client::HeldElsewhere;
use quack_mcp::{HttpTransport, McpServer};

/// How the server knows who is asking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeMode {
    /// Users log in with a password or send an API token.
    Login,
    /// `--local` or `[server].local`: no authentication, one implicit owner
    /// of everything; loopback only.
    Local,
}

/// The `Host` values a local-mode server answers: local mode makes every
/// request the implicit owner, and the cross-site check only compares
/// `Origin` with `Host`, so a page on a name that resolves to loopback
/// (DNS rebinding) would otherwise pass. Loopback names on the bound port
/// only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LocalHosts(Vec<String>);

impl LocalHosts {
    /// The loopback names on `port`; on port 80 the bare names too, since a
    /// browser omits the default port.
    pub(crate) fn on(port: u16) -> Self {
        let names = ["127.0.0.1", "localhost", "[::1]"];
        let mut hosts: Vec<String> = names.iter().map(|n| format!("{n}:{port}")).collect();
        if port == 80 {
            hosts.extend(names.iter().map(|n| String::from(*n)));
        }
        Self(hosts)
    }

    /// Whether a request's `Host` header is one of them.
    pub(crate) fn allows(&self, host: &str) -> bool {
        self.0.iter().any(|h| h.eq_ignore_ascii_case(host))
    }

    pub(crate) fn names(&self) -> Vec<String> {
        self.0.clone()
    }
}

pub(crate) struct AppState {
    /// The process: its queue, its open workspaces (one handle per file),
    /// and its stopping token.
    pub engine: Arc<Engine>,
    pub control: ControlPlane,
    pub mode: ServeMode,
    /// In local mode, the `Host` values answered; `None` with logins, where
    /// the credential, not the name, guards the server.
    pub local_hosts: Option<LocalHosts>,
    /// Browser and API-login sessions.
    pub sessions: Arc<WebSessions>,
    /// Sign-in through `[server.oidc]`'s issuer, when configured and not in
    /// local mode.
    pub oidc: Option<Oidc>,
    /// What quack publishes as a protected resource, when it accepts the
    /// issuer's access tokens (`[server.oidc].audience`).
    pub resource: Option<ProtectedResource>,
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

pub(crate) type McpTransport = HttpTransport;

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
        engine: Arc<Engine>,
        control: ControlPlane,
        mode: ServeMode,
        local_hosts: Option<LocalHosts>,
        sessions: Arc<WebSessions>,
        oidc: Option<Oidc>,
    ) -> Self {
        let resource = ProtectedResource::of(&engine.config, mode);
        // The recorder lives for the process, and only `/metrics` reads
        // it; a second state in one process (tests) keeps the first.
        telemetry::install();
        Self {
            engine,
            control,
            mode,
            local_hosts,
            sessions,
            resource,
            oidc,
            mcp: tokio::sync::Mutex::new(HashMap::new()),
            extractions: Mutex::new(HashSet::new()),
            flashes: Flashes::default(),
            permissions: Permissions::default(),
            readiness: tokio::sync::Mutex::new(None),
        }
    }

    /// Stop the server whose HTTP drain is `server`: streams end and the
    /// job queue shuts down beside the drain, under one `grace`, and the
    /// workspaces close last.
    ///
    /// # Errors
    ///
    /// Returns the drain's own error.
    pub(crate) async fn stop<S>(
        self: Arc<Self>,
        mut server: Option<S>,
        grace: Duration,
    ) -> std::io::Result<()>
    where
        S: Future<Output = std::io::Result<()>> + Unpin,
    {
        let deadline = Instant::now().checked_add(grace);
        let drain = async {
            // `None`: the server already ended on its own; nothing to drain.
            let drained = async {
                match server.as_mut() {
                    Some(server) => server.await,
                    None => Ok(()),
                }
            };
            let (served, _) = tokio::join!(drained, self.engine.stop_jobs(grace));
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
        for job in self.engine.stop_jobs(left).await {
            tracing::warn!(job = %job.id, kind = %job.kind, state = %job.state, "job still active at exit");
        }
        self.close().await;
        served
    }

    /// Drop every MCP transport, the last thing the server does; whoever
    /// owns the engine closes it after (`Engine::close`), so a terminal
    /// session serving beside itself keeps its workspace until it quits.
    pub(crate) async fn close(&self) {
        let transports = std::mem::take(&mut *self.mcp.lock().await);
        let closed = tokio::task::spawn_blocking(move || drop(transports));
        if let Err(e) = closed.await {
            tracing::error!(error = %e, "closing the MCP transports failed");
        }
    }

    /// Whether this server serves `workspace`: every one for `quack serve`,
    /// only the session's own when a terminal serves beside itself.
    pub(crate) fn serves(&self, workspace_id: &WorkspaceId) -> bool {
        self.engine.held().is_none_or(|held| held == workspace_id)
    }

    /// Let go of one workspace before it is deleted: its MCP transports
    /// and its open file, once no job of it is queued or running. A
    /// request that still holds a handle finishes on the unlinked file.
    pub(crate) async fn close_workspace(&self, workspace_id: &WorkspaceId) -> ApiResult<()> {
        // Refused (held, or a job still running): its MCP sessions live on.
        self.engine.closable(workspace_id)?;
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
        let closed = self.engine.close_workspace(workspace_id).await;
        let dropped = tokio::task::spawn_blocking(move || drop(transports));
        closed?;
        dropped
            .await
            .map_err(|e| ApiError::internal(format!("closing the workspace failed: {e}")))
    }

    /// Run `work` with the workspace's file closed, then reopen it: this
    /// workspace's requests wait for it, other workspaces do not.
    pub(crate) async fn with_workspace_closed<T, F>(
        &self,
        workspace_id: &WorkspaceId,
        work: F,
    ) -> ApiResult<T>
    where
        T: Send + 'static,
        F: FnOnce() -> CoreResult<T> + Send + 'static,
    {
        Ok(self
            .engine
            .with_workspace_closed(workspace_id, work)
            .await?)
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
                .engine
                .config
                .data_dir()
                .join(format!(".readyz-{}", uuid::Uuid::now_v7()));
            tokio::fs::write(&probe, b"ok")
                .await
                .and_then(|()| std::fs::remove_file(&probe))
                .map_err(|e| e.to_string())
        };
        let vault_key = Vault::new(self.engine.config.data_dir(), KeySource::Keychain)
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
        // With logins the server may sit behind any host name, since
        // bearer auth, not the Host header, guards it; local mode
        // answers loopback names only.
        let transport = make().http_transport(
            self.local_hosts
                .as_ref()
                .map_or_else(Vec::new, LocalHosts::names),
        );
        open.insert(key, transport.clone());
        transport
    }

    /// The writer handle for a workspace, opening the file on first use
    /// (`Engine::workspace`, which is where one open per file is enforced).
    /// The body limit of a route that takes a file (`[ingestion].upload_max_mb`):
    /// a document upload, a snapshot to restore. Every other route keeps
    /// the router's small default (`JSON_BODY_BYTES`), so a statement or a
    /// question cannot make the server buffer half a gigabyte.
    pub(crate) fn upload_body_limit(&self) -> DefaultBodyLimit {
        let bytes = usize::try_from(self.engine.config.ingestion.upload_max_mb)
            .unwrap_or(usize::MAX)
            .saturating_mul(1024 * 1024);
        DefaultBodyLimit::max(bytes)
    }

    pub(crate) async fn workspace_db(&self, workspace_id: &WorkspaceId) -> ApiResult<SharedDb> {
        Ok(self.open_workspace(workspace_id).await?.writer)
    }

    /// The workspace's handle (`Engine::workspace`). A file another
    /// process holds is 503 `workspace_locked` on every route, naming the
    /// holder the way the command line does when its socket answers
    /// (`HeldElsewhere`), since the server never reads through another
    /// process.
    async fn open_workspace(&self, workspace_id: &WorkspaceId) -> ApiResult<WorkspaceHandle> {
        match self.engine.workspace(workspace_id).await {
            Ok(handle) => Ok(handle),
            Err(locked @ CoreError::WorkspaceLocked { .. }) => {
                let config = &self.engine.config;
                let held = HeldElsewhere::find(
                    &config.workspace_dir(workspace_id.as_str()),
                    workspace_id,
                    format!("the file of workspace {workspace_id}"),
                    self.engine.holder().clone(),
                    locked,
                )
                .await;
                Err(ApiError::workspace_locked(held.to_string()))
            }
            Err(e) => Err(e.into()),
        }
    }

    /// The reader handle for a workspace (built once, alongside the
    /// writer, on first use): every read-only handler should prefer this
    /// over [`Self::workspace_db`] so it never queues behind a write.
    pub(crate) async fn reader_db(&self, workspace_id: &WorkspaceId) -> ApiResult<ReaderDb> {
        Ok(self.open_workspace(workspace_id).await?.reader)
    }

    /// The workspace's insert-only audit connection, opened with its writer.
    pub(crate) async fn audit_log(&self, workspace_id: &WorkspaceId) -> ApiResult<Arc<AuditLog>> {
        Ok(self.open_workspace(workspace_id).await?.audit)
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
