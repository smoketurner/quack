//! A client of the workspace socket, for a command that found the
//! workspace open in another `quack` process: `quack -q` runs its read
//! through the holder's `sql` tool, `quack jobs` reads the holder's
//! `quack://workspace/jobs`.

use std::path::Path;

use quack_core::error::{Error, Result};
use quack_core::forward;
use quack_core::ids::WorkspaceId;
use quack_core::jobs::JobInfo;
use quack_core::storage::workspace::QueryResults;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ListResourceTemplatesResult, ListResourcesResult,
    ListToolsResult, PaginatedRequestParams, ReadResourceRequestParams, ReadResourceResult,
};
use rmcp::service::{RunningService, ServiceError};
use rmcp::{ClientHandler, ErrorData as McpError, RoleClient, ServiceExt};
use serde::Deserialize;

/// What a refusal names as the holder before `initialize` named it.
const SOMEONE: &str = "another process";

/// Who holds the workspace in `workspace_dir`, as its socket's
/// `initialize` names it ("the quack terminal session", "quack ingest"):
/// one connection, closed again. `None` when nothing answers on the
/// socket, which a command refused by the file's lock turns into
/// [`locked_without_socket`].
///
/// # Errors
///
/// Returns `forward::connect`'s refusals, or an `initialize` that failed.
pub async fn holder_of(workspace_dir: &Path, workspace_id: &WorkspaceId) -> Result<Option<String>> {
    let Some(client) = HolderClient::connect(workspace_dir, workspace_id).await? else {
        return Ok(None);
    };
    let holder = client.holder().to_owned();
    client.close().await;
    Ok(Some(holder))
}

/// What the lock on the workspace file at `path` means when no socket
/// answers for it: every holder but `quack serve` serves the socket while
/// it holds the file, so the holder is the server (whose `/mcp/v1`
/// endpoint is the way in) or a process between opening the file and
/// binding the socket, or between closing them.
#[must_use]
pub fn locked_without_socket(path: &Path) -> String {
    format!(
        "the workspace file {} is open in another quack process that answers on no socket: \
         quack serve (reach the workspace through its /mcp/v1 endpoint), or a process that is \
         starting or stopping; try again, or stop it",
        path.display()
    )
}

/// An MCP session with the process holding a workspace. `H` hears the
/// holder's notifications: `()` ignores them; the per-call server passes
/// progress on to its own client.
pub struct HolderClient<H: ClientHandler = ()> {
    service: RunningService<RoleClient, H>,
    holder: String,
}

/// How an exchange with the holder failed: the holder answered the request
/// with a JSON-RPC error of its own (no tool by that name, no resource at
/// that URI, a refused argument), which keeps its code and message so a
/// relay can pass it on as it is; or the exchange itself failed (the
/// socket closed, the request timed out), which is this process's own
/// error, naming the holder.
#[derive(Debug)]
pub enum HolderError {
    /// The holder's own error, as it sent it (boxed: the error data is
    /// wide, and every exchange carries this type in its `Result`).
    Answered {
        /// Who answered, as `initialize` named it.
        holder: String,
        error: Box<McpError>,
    },
    /// The exchange failed before the holder answered.
    Unreachable(Error),
}

impl From<HolderError> for Error {
    fn from(failure: HolderError) -> Self {
        match failure {
            HolderError::Answered { holder, error } => Self::Forward(format!(
                "{holder} answered an error: {}: {}",
                error.code.0, error.message
            )),
            HolderError::Unreachable(error) => error,
        }
    }
}

/// What `quack://workspace/jobs` holds.
#[derive(Deserialize)]
struct Jobs {
    jobs: Vec<JobInfo>,
}

/// A read statement's rows as the holder's `sql` tool returned them.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct HolderRows {
    #[serde(flatten)]
    pub results: QueryResults,
    /// Whether the holder cut the rows at its `[analysis].max_query_rows`
    /// or at the structured answer's byte limit.
    pub truncated: bool,
    /// Values the holder cut at `STRUCTURED_CELL_CHARS` characters.
    #[serde(default)]
    pub cells_cut: usize,
}

impl HolderClient {
    /// Connect to the holder of the workspace in `workspace_dir`, when one
    /// listens on its socket (`forward::connect`); `None` when none does.
    ///
    /// # Errors
    ///
    /// Returns `forward::connect`'s refusals, or an `initialize` that
    /// failed.
    pub async fn connect(workspace_dir: &Path, workspace_id: &WorkspaceId) -> Result<Option<Self>> {
        Self::connect_with(workspace_dir, workspace_id, ()).await
    }
}

impl<H: ClientHandler> HolderClient<H> {
    /// [`HolderClient::connect`] with `handler` hearing the holder's
    /// notifications.
    ///
    /// # Errors
    ///
    /// Returns `forward::connect`'s refusals, or an `initialize` that
    /// failed.
    pub async fn connect_with(
        workspace_dir: &Path,
        workspace_id: &WorkspaceId,
        handler: H,
    ) -> Result<Option<Self>> {
        let Some(stream) = forward::connect(workspace_dir, workspace_id).await? else {
            return Ok(None);
        };
        let service = handler.serve(stream).await.map_err(|e| {
            Error::Forward(format!(
                "the workspace is open in {SOMEONE}, which did not answer the MCP initialize: {e}"
            ))
        })?;
        let holder = service
            .peer_info()
            .and_then(|info| {
                info.server_info
                    .as_ref()
                    .and_then(|server| server.description.clone())
            })
            .unwrap_or_else(|| String::from(SOMEONE));
        Ok(Some(Self { service, holder }))
    }

    /// A tool call, as the holder answers it.
    ///
    /// # Errors
    ///
    /// Returns the holder's own error, or the exchange's failure
    /// ([`HolderError`]).
    pub async fn call_tool(
        &self,
        params: CallToolRequestParams,
    ) -> std::result::Result<CallToolResult, HolderError> {
        self.service
            .call_tool(params)
            .await
            .map_err(|e| self.exchange("the tool call failed", e))
    }

    /// `tools/list`, as the holder answers it.
    ///
    /// # Errors
    ///
    /// Returns the holder's own error, or the exchange's failure
    /// ([`HolderError`]).
    pub async fn list_tools(
        &self,
        params: Option<PaginatedRequestParams>,
    ) -> std::result::Result<ListToolsResult, HolderError> {
        self.service
            .list_tools(params)
            .await
            .map_err(|e| self.exchange("listing the tools failed", e))
    }

    /// `resources/list`, as the holder answers it.
    ///
    /// # Errors
    ///
    /// Returns the holder's own error, or the exchange's failure
    /// ([`HolderError`]).
    pub async fn list_resources(
        &self,
        params: Option<PaginatedRequestParams>,
    ) -> std::result::Result<ListResourcesResult, HolderError> {
        self.service
            .list_resources(params)
            .await
            .map_err(|e| self.exchange("listing the resources failed", e))
    }

    /// `resources/templates/list`, as the holder answers it.
    ///
    /// # Errors
    ///
    /// Returns the holder's own error, or the exchange's failure
    /// ([`HolderError`]).
    pub async fn list_resource_templates(
        &self,
        params: Option<PaginatedRequestParams>,
    ) -> std::result::Result<ListResourceTemplatesResult, HolderError> {
        self.service
            .list_resource_templates(params)
            .await
            .map_err(|e| self.exchange("listing the resource templates failed", e))
    }

    /// `resources/read`, as the holder answers it.
    ///
    /// # Errors
    ///
    /// Returns the holder's own error, or the exchange's failure
    /// ([`HolderError`]).
    pub async fn read_resource(
        &self,
        params: ReadResourceRequestParams,
    ) -> std::result::Result<ReadResourceResult, HolderError> {
        self.service
            .read_resource(params)
            .await
            .map_err(|e| self.exchange("reading the resource failed", e))
    }

    /// Who holds the workspace, as its `initialize` reply named it.
    #[must_use]
    pub fn holder(&self) -> &str {
        &self.holder
    }

    /// Run a statement through the holder's `sql` tool: a read gives its
    /// rows; a write, or a statement the holder refuses, is the holder's
    /// own refusal, naming it.
    ///
    /// # Errors
    ///
    /// Returns the holder's refusal or error as [`Error::Forward`].
    pub async fn sql(&self, statement: &str) -> Result<HolderRows> {
        let mut arguments = serde_json::Map::new();
        arguments.insert(String::from("sql"), serde_json::Value::from(statement));
        let params = CallToolRequestParams::new("sql").with_arguments(arguments);
        let result = self.call_tool(params).await?;
        if result.is_error.unwrap_or(false) {
            let text: String = result
                .content
                .iter()
                .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
                .collect();
            return Err(Error::Forward(format!(
                "the workspace is open in {}, which refused the statement: {text}",
                self.holder
            )));
        }
        let Some(structured) = result.structured_content else {
            return Err(self.failed("the sql tool answered without rows"));
        };
        serde_json::from_value(structured)
            .map_err(|e| self.failed(&format!("the sql tool's rows did not parse: {e}")))
    }

    /// The holder's jobs of the workspace (`quack://workspace/jobs`).
    ///
    /// # Errors
    ///
    /// Returns the holder's error as [`Error::Forward`].
    pub async fn jobs(&self) -> Result<Vec<JobInfo>> {
        let read = self
            .read_resource(ReadResourceRequestParams::new("quack://workspace/jobs"))
            .await?;
        let mut text = String::new();
        for content in &read.contents {
            if let rmcp::model::ResourceContents::TextResourceContents { text: part, .. } = content
            {
                text.push_str(part);
            }
        }
        let listed: Jobs = serde_json::from_str(&text)
            .map_err(|e| self.failed(&format!("the jobs did not parse: {e}")))?;
        Ok(listed.jobs)
    }

    /// End the session.
    pub async fn close(self) {
        if let Err(e) = self.service.cancel().await {
            tracing::debug!(error = %e, "the session with the holder did not end cleanly");
        }
    }

    fn failed(&self, message: &str) -> Error {
        Error::Forward(format!("reaching {} failed: {message}", self.holder))
    }

    /// Sort one exchange's failure: the holder's own answer keeps its code;
    /// anything else (`what` failed) is this process's error.
    fn exchange(&self, what: &str, error: ServiceError) -> HolderError {
        match error {
            ServiceError::McpError(error) => HolderError::Answered {
                holder: self.holder.clone(),
                error: Box::new(error),
            },
            other => HolderError::Unreachable(self.failed(&format!("{what}: {other}"))),
        }
    }
}
