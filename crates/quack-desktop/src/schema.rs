//! The frontend's types come from Rust: every type a command returns or
//! an event carries, as the JSON Schema `utoipa` derives for the REST
//! API's document, written to `ui/src/schema.json` and turned into
//! TypeScript by `openapi-typescript` (`make ui-types`). A test fails when
//! the committed file is stale, as the OpenAPI tests do.

use quack_core::analysis::agent::AgentResponseBody;
use quack_core::analysis::events::{ToolName, ToolStep};
use quack_core::analysis::policy::Hold;
use quack_core::jobs::{JobInfo, JobKind, JobProgress, JobState};
use quack_core::storage::control::WorkspaceRow;
use quack_core::storage::sessions::{ChatMode, MessageRow, SessionRow};
use quack_core::storage::workspace::{DocumentFilter, DocumentInfo, DocumentPage};
use utoipa::OpenApi;

use crate::chat::TurnEvent;
use crate::error::{CommandError, ErrorKind};
use crate::settings::{EnvView, SettingView, SettingsView};
use crate::workspace::WorkspaceSummary;

/// The components the frontend's bindings are generated from.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "quack-desktop commands",
        description = "The types the desktop app's commands return and its events carry."
    ),
    components(schemas(
        WorkspaceRow,
        WorkspaceSummary,
        SessionRow,
        MessageRow,
        ChatMode,
        TurnEvent,
        AgentResponseBody,
        ToolStep,
        ToolName,
        Hold,
        DocumentPage,
        DocumentInfo,
        DocumentFilter,
        JobInfo,
        JobKind,
        JobState,
        JobProgress,
        SettingsView,
        SettingView,
        EnvView,
        CommandError,
        ErrorKind,
    ))
)]
struct Schema;

/// The document as JSON, pretty-printed and ending in a newline.
pub(crate) fn json() -> String {
    let mut text = Schema::openapi()
        .to_pretty_json()
        .unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"));
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    /// `ui/src/schema.json` is what the current commands generate.
    #[test]
    fn the_committed_schema_is_current() {
        let committed = include_str!("../ui/src/schema.json");
        assert!(
            committed == super::json(),
            "ui/src/schema.json is stale: run `make ui-types` and commit the result"
        );
    }
}
