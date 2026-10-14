//! `quack mcp`: the server on stdio that holds nothing between calls.
//! Each call runs where the workspace is: in the process that holds it,
//! over its socket (the terminal session, a command, another `quack mcp`
//! mid-call), or in this process, which opens the workspace for that one
//! call and closes it after. So Claude Code's entry works whether the
//! session was started before or after it, and a session started while
//! `quack mcp` is idle takes the file. While it holds the file it serves
//! the workspace socket like every other holder, so a second `quack mcp`
//! on the same workspace (another editor, another Claude Code session)
//! and `quack -q` in a shell reach it instead of failing on the lock.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use quack_core::analysis::policy::WritePolicy;
use quack_core::engine::{Engine, WorkspaceHandle};
use quack_core::error::Error as CoreError;
use quack_core::forward::Listener;
use quack_core::storage::control::WorkspaceRow;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, ClientConfig, Implementation,
    ListResourceTemplatesResult, ListResourcesResult, ListToolsResult, PaginatedRequestParams,
    ProgressNotificationParam, ReadResourceRequestParams, ReadResourceResponse, ServerCapabilities,
    ServerConfig,
};
use rmcp::service::{NotificationContext, Peer, RequestContext};
use rmcp::{ClientHandler, ErrorData as McpError, RoleClient, RoleServer, ServerHandler};

use crate::client::{self, HolderClient, HolderError};
use crate::socket::SocketServer;
use crate::{McpServer, McpSetup, Transport, instructions};

/// How long a call waits on a lock no socket answers for before it is
/// reported: another `quack mcp` is between opening the file and binding
/// its socket, or between closing them, for a moment at most.
const LOCK_WAIT: Duration = Duration::from_secs(3);

/// How often the lock is tried again within [`LOCK_WAIT`].
const LOCK_RETRY: Duration = Duration::from_millis(50);

/// The socket this process serves while it holds the file, shared with
/// [`serve_stdio`](crate::serve_stdio) so the exit closes it.
pub(crate) type Socket = Arc<tokio::sync::Mutex<Option<SocketServer>>>;

/// The stdio server of `quack mcp`.
#[derive(Clone)]
pub struct PerCallServer {
    engine: Arc<Engine>,
    workspace: WorkspaceRow,
    policy: WritePolicy,
    /// The workspace socket, served from the first call that opens the
    /// file here until the file is let go.
    socket: Socket,
    /// Whether the log has said that `--allow-write` has no say through a
    /// holder; once per process.
    warned: Arc<AtomicBool>,
}

/// The holder's progress notifications, passed on to this server's own
/// client under the token it sent.
#[derive(Clone)]
struct Bridge {
    peer: Peer<RoleServer>,
}

impl ClientHandler for Bridge {
    fn get_info(&self) -> ClientConfig {
        ClientConfig::default()
    }

    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        if let Err(e) = self.peer.notify_progress(params).await {
            tracing::debug!(error = %e, "could not pass a progress notification on");
        }
    }
}

/// Where one call runs.
enum Backend {
    /// The process holding the workspace, over its socket.
    Holder(HolderClient<Bridge>),
    /// This process, with the workspace open for this call.
    Local(McpServer),
}

impl PerCallServer {
    #[must_use]
    pub fn new(engine: Arc<Engine>, workspace: WorkspaceRow, policy: WritePolicy) -> Self {
        Self {
            engine,
            workspace,
            policy,
            socket: Socket::default(),
            warned: Arc::default(),
        }
    }

    /// The socket this process serves while it holds the file, for the
    /// exit to close.
    pub(crate) fn socket(&self) -> Socket {
        Arc::clone(&self.socket)
    }

    /// The backend for one call: the holder when one answers on the
    /// socket, else the workspace opened here. A lock no socket answers
    /// for is tried again for [`LOCK_WAIT`] (another `quack mcp` is
    /// between opening the file and binding its socket, or letting both
    /// go), then reported: the holder is `quack serve`, which serves no
    /// socket, or a process starting or stopping.
    async fn backend(&self, peer: &Peer<RoleServer>) -> Result<Backend, McpError> {
        let dir = self.engine.config.workspace_dir(self.workspace.id.as_str());
        let deadline = tokio::time::Instant::now()
            .checked_add(LOCK_WAIT)
            .unwrap_or_else(tokio::time::Instant::now);
        loop {
            let bridge = Bridge { peer: peer.clone() };
            match HolderClient::connect_with(&dir, &self.workspace.id, bridge).await {
                Ok(Some(client)) => {
                    if self.policy.allows_unasked() && !self.warned.swap(true, Ordering::Relaxed) {
                        tracing::warn!(
                            holder = client.holder(),
                            "--allow-write has no effect while another process holds the workspace: a write asks at the terminal session's prompt, or is refused by a command"
                        );
                    }
                    return Ok(Backend::Holder(client));
                }
                Ok(None) => {}
                Err(e) => return Err(internal(e.to_string())),
            }
            match self.engine.workspace(&self.workspace.id).await {
                Ok(handle) => return Ok(Backend::Local(self.local(handle).await)),
                Err(CoreError::WorkspaceLocked { .. })
                    if tokio::time::Instant::now() < deadline =>
                {
                    tokio::time::sleep(LOCK_RETRY).await;
                }
                Err(CoreError::WorkspaceLocked { path }) => {
                    return Err(internal(client::locked_without_socket(&path)));
                }
                Err(e) => return Err(internal(e.to_string())),
            }
        }
    }

    /// The server for a call that runs here, and the workspace socket
    /// served from it while this process holds the file: another `quack
    /// mcp` or `quack -q` reaches this one there, under this process's own
    /// write policy, like any holder. A socket that cannot be bound is
    /// logged; the call runs without it.
    async fn local(&self, handle: WorkspaceHandle) -> McpServer {
        let server = McpServer::new(McpSetup::local(
            &self.engine,
            handle,
            self.workspace.clone(),
            self.policy,
            Transport::Stdio,
            None,
        ));
        let mut socket = self.socket.lock().await;
        if socket.is_none() {
            let dir = self.engine.config.workspace_dir(self.workspace.id.as_str());
            match Listener::bind(&dir, &self.workspace.id).await {
                Ok(listener) => *socket = Some(SocketServer::serve(listener, server.clone())),
                Err(e) => {
                    tracing::warn!(workspace = %self.workspace.id, error = %e, "other quack processes cannot reach this workspace through this one");
                }
            }
        }
        server
    }

    /// After a call ran here: let the file go, with the socket first (so
    /// a call relayed through it ends on an open workspace), unless a job
    /// of it (a title, a labelling) still runs or another process is
    /// mid-call through the socket, in which case the next call or the
    /// exit closes them.
    async fn release(&self, backend: Backend) {
        let Backend::Local(server) = backend else {
            return;
        };
        drop(server);
        if let Err(e) = self.engine.closable(&self.workspace.id) {
            tracing::debug!(error = %e, "the workspace stays open after the call");
            return;
        }
        {
            let mut socket = self.socket.lock().await;
            if socket.as_ref().is_some_and(SocketServer::has_sessions) {
                return;
            }
            if let Some(socket) = socket.take() {
                socket.close(Duration::ZERO).await;
            }
        }
        match self.engine.close_workspace(&self.workspace.id).await {
            Ok(()) | Err(CoreError::WorkspaceBusy { .. }) => {}
            Err(e) => tracing::debug!(error = %e, "the workspace was not closed after the call"),
        }
    }
}

fn internal(message: String) -> McpError {
    McpError::internal_error(message, None)
}

/// The holder's failure, as the client sees it: the holder's own error
/// passes through with its code and message (a client can tell a tool it
/// has no name for, -32602, from a fault); only an exchange that failed
/// (the socket closed, `initialize` refused) is this server's internal
/// error.
fn forwarded(error: HolderError) -> McpError {
    match error {
        HolderError::Answered { error, .. } => *error,
        HolderError::Unreachable(error) => internal(error.to_string()),
    }
}

impl ServerHandler for PerCallServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_server_info(
            Implementation::new("quack", env!("CARGO_PKG_VERSION"))
                .with_title("quack")
                .with_description("quack mcp"),
        )
        .with_instructions(instructions(&self.workspace.name))
    }

    async fn call_tool(
        &self,
        mut request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let backend = self.backend(&context.peer).await?;
        let result: Result<CallToolResponse, McpError> = match &backend {
            Backend::Holder(client) => {
                // The client's progress token goes with the call, so the
                // holder reports to it through the bridge.
                request.meta = Some(context.meta.clone());
                let ct = context.ct.clone();
                tokio::select! {
                    () = ct.cancelled() => Err(internal(String::from("the call was cancelled"))),
                    answered = client.call_tool(request) => {
                        answered.map(CallToolResponse::from).map_err(forwarded)
                    }
                }
            }
            Backend::Local(server) => server.call_tool_now(request, context).await,
        };
        self.release(backend).await;
        result
    }

    async fn list_tools(
        &self,
        request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let backend = self.backend(&context.peer).await?;
        let result = match &backend {
            Backend::Holder(client) => client.list_tools(request).await.map_err(forwarded),
            Backend::Local(_) => Ok(ListToolsResult::with_all_items(McpServer::tools())),
        };
        self.release(backend).await;
        result
    }

    async fn list_resources(
        &self,
        request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        let backend = self.backend(&context.peer).await?;
        let result = match &backend {
            Backend::Holder(client) => client.list_resources(request).await.map_err(forwarded),
            Backend::Local(server) => server.list_resources(request, context).await,
        };
        self.release(backend).await;
        result
    }

    async fn list_resource_templates(
        &self,
        request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        let backend = self.backend(&context.peer).await?;
        let result = match &backend {
            Backend::Holder(client) => client
                .list_resource_templates(request)
                .await
                .map_err(forwarded),
            Backend::Local(server) => server.list_resource_templates(request, context).await,
        };
        self.release(backend).await;
        result
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, McpError> {
        let backend = self.backend(&context.peer).await?;
        let result: Result<ReadResourceResponse, McpError> = match &backend {
            Backend::Holder(client) => client
                .read_resource(request)
                .await
                .map(ReadResourceResponse::from)
                .map_err(forwarded),
            Backend::Local(server) => server.read_resource(request, context).await,
        };
        self.release(backend).await;
        result
    }
}
