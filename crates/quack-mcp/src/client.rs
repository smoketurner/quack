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
use rmcp::model::{CallToolRequestParams, ReadResourceRequestParams};
use rmcp::service::RunningService;
use rmcp::{RoleClient, ServiceExt};
use serde::Deserialize;

/// What a refusal names as the holder before `initialize` named it.
const SOMEONE: &str = "another process";

/// An MCP session with the process holding a workspace.
pub struct HolderClient {
    service: RunningService<RoleClient, ()>,
    holder: String,
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
    /// Whether the holder cut the rows at its `[analysis].max_query_rows`.
    pub truncated: bool,
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
        let Some(stream) = forward::connect(workspace_dir, workspace_id).await? else {
            return Ok(None);
        };
        let service = ().serve(stream).await.map_err(|e| {
            Error::Forward(format!(
                "the workspace is open in {SOMEONE}, which did not answer the MCP initialize: {e}"
            ))
        })?;
        let holder = service
            .peer_info()
            .and_then(|info| info.server_info.as_ref().and_then(|s| s.title.clone()))
            .unwrap_or_else(|| String::from(SOMEONE));
        Ok(Some(Self { service, holder }))
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
        let result = self
            .service
            .call_tool(params)
            .await
            .map_err(|e| self.failed(&format!("the sql call failed: {e}")))?;
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
            .service
            .read_resource(ReadResourceRequestParams::new("quack://workspace/jobs"))
            .await
            .map_err(|e| self.failed(&format!("reading the jobs failed: {e}")))?;
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
}
