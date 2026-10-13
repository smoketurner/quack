//! `quack mcp`: the server on stdio that holds nothing between calls.
//! Each call runs where the workspace is: in the process that holds it,
//! over its socket (the terminal session, a command), or in this process,
//! which opens the workspace for that one call and closes it after. So
//! Claude Code's entry works whether the session was started before or
//! after it, and a session started while `quack mcp` is idle takes the
//! file.

use std::sync::Arc;

use quack_core::analysis::permissions::Approvals;
use quack_core::analysis::policy::WritePolicy;
use quack_core::engine::Engine;
use quack_core::error::Error as CoreError;
use quack_core::storage::control::WorkspaceRow;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, ClientConfig, Implementation,
    ListResourceTemplatesResult, ListResourcesResult, ListToolsResult, PaginatedRequestParams,
    ProgressNotificationParam, ReadResourceRequestParams, ReadResourceResponse, ServerCapabilities,
    ServerConfig,
};
use rmcp::service::{NotificationContext, Peer, RequestContext};
use rmcp::{ClientHandler, ErrorData as McpError, RoleClient, RoleServer, ServerHandler};

use crate::client::HolderClient;
use crate::{McpServer, McpSetup, Transport, instructions};

/// The stdio server of `quack mcp`.
#[derive(Clone)]
pub struct PerCallServer {
    engine: Arc<Engine>,
    workspace: WorkspaceRow,
    policy: WritePolicy,
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
        }
    }

    /// The backend for one call: the holder when one answers on the
    /// socket, else the workspace opened here. A workspace another process
    /// holds without a socket (`quack serve`) cannot be reached.
    async fn backend(&self, peer: &Peer<RoleServer>) -> Result<Backend, McpError> {
        let dir = self.engine.config.workspace_dir(self.workspace.id.as_str());
        let bridge = Bridge { peer: peer.clone() };
        match HolderClient::connect_with(&dir, &self.workspace.id, bridge).await {
            Ok(Some(client)) => return Ok(Backend::Holder(client)),
            Ok(None) => {}
            Err(e) => return Err(internal(e.to_string())),
        }
        let handle = match self.engine.workspace(&self.workspace.id).await {
            Ok(handle) => handle,
            Err(CoreError::WorkspaceLocked { path }) => {
                return Err(internal(format!(
                    "the workspace file {} is open in another quack process that serves no \
                     socket (quack serve); reach it through that server's /mcp/v1 endpoint, or \
                     stop it",
                    path.display()
                )));
            }
            Err(e) => return Err(internal(e.to_string())),
        };
        Ok(Backend::Local(McpServer::new(McpSetup {
            config: self.engine.config.clone(),
            db: handle.writer,
            reader: handle.reader,
            workspace: self.workspace.clone(),
            policy: self.policy,
            user_id: None,
            transport: Transport::Stdio,
            jobs: self.engine.jobs.clone(),
            stopping: self.engine.stopping(),
            holder: String::from("quack mcp"),
            approvals: Approvals::new(),
        })))
    }

    /// After a call ran here: let the file go, unless a job of it (a
    /// title, a labelling) still runs, in which case the next call or the
    /// exit closes it.
    async fn release(&self, backend: Backend) {
        let Backend::Local(server) = backend else {
            return;
        };
        drop(server);
        match self.engine.close_workspace(&self.workspace.id).await {
            Ok(()) | Err(CoreError::WorkspaceBusy { .. }) => {}
            Err(e) => tracing::debug!(error = %e, "the workspace was not closed after the call"),
        }
    }
}

fn internal(message: String) -> McpError {
    McpError::internal_error(message, None)
}

/// The holder's failure, as the client sees it.
fn forwarded(error: &CoreError) -> McpError {
    internal(error.to_string())
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
                        answered.map(CallToolResponse::from).map_err(|e| forwarded(&e))
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
            Backend::Holder(client) => client.list_tools(request).await.map_err(|e| forwarded(&e)),
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
            Backend::Holder(client) => client
                .list_resources(request)
                .await
                .map_err(|e| forwarded(&e)),
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
                .map_err(|e| forwarded(&e)),
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
                .map_err(|e| forwarded(&e)),
            Backend::Local(server) => server.read_resource(request, context).await,
        };
        self.release(backend).await;
        result
    }
}
