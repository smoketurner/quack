//! The Workspaces view's commands over the control plane: list, create,
//! rename, delete, and the app's hold on a workspace (open, close).

use quack_core::ids::WorkspaceId;
use quack_core::storage::control::{AuditAction, Outcome, WorkspaceName, WorkspaceRow};
use serde::Serialize;
use tauri::State;

use crate::app::AppState;
use crate::error::{CommandError, CommandResult};

/// A workspace and whether the app holds it open.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub(crate) struct WorkspaceSummary {
    #[serde(flatten)]
    pub(crate) workspace: WorkspaceRow,
    pub(crate) open: bool,
}

/// Every workspace, by name.
#[tauri::command]
pub(crate) async fn list_workspaces(
    state: State<'_, AppState>,
) -> CommandResult<Vec<WorkspaceSummary>> {
    let open = state.open_ids().await;
    let rows = state.control.list_workspaces().await?;
    Ok(rows
        .into_iter()
        .map(|workspace| WorkspaceSummary {
            open: open.contains(&workspace.id),
            workspace,
        })
        .collect())
}

/// Create the workspace `name` (trimmed, non-empty, no `/`, `\`, or `.`).
#[tauri::command]
pub(crate) async fn create_workspace(
    state: State<'_, AppState>,
    name: String,
) -> CommandResult<WorkspaceRow> {
    let name: WorkspaceName = name.parse()?;
    let entry = state
        .auditor
        .entry(AuditAction::Workspace, Outcome::Allowed);
    Ok(state.control.create_workspace(&name, None, entry).await?)
}

/// Rename workspace `id` to `name`.
#[tauri::command]
pub(crate) async fn rename_workspace(
    state: State<'_, AppState>,
    id: WorkspaceId,
    name: String,
) -> CommandResult<WorkspaceRow> {
    let name: WorkspaceName = name.parse()?;
    let entry = state
        .auditor
        .entry(AuditAction::Workspace, Outcome::Allowed);
    Ok(state.control.rename_workspace(&id, &name, entry).await?)
}

/// Delete workspace `id`: its row, its members and tokens, and its files.
/// The frontend confirms first, naming the workspace.
#[tauri::command]
pub(crate) async fn delete_workspace(
    state: State<'_, AppState>,
    id: WorkspaceId,
) -> CommandResult<()> {
    let row = state
        .control
        .get_workspace(&id)
        .await?
        .ok_or_else(|| CommandError::not_found(format!("no workspace has the id {id}")))?;
    // Let go of the file before deleting it.
    state.close_workspace(&id).await;
    let entry = state.auditor.entry(AuditAction::Delete, Outcome::Allowed);
    state.control.delete_workspace(&row.id, entry).await?;
    let dir = state.config.workspace_dir(row.id.as_str());
    if dir.exists() {
        tokio::fs::remove_dir_all(&dir)
            .await
            .map_err(|e| anyhow::anyhow!("failed to remove {}: {e}", dir.display()))?;
    }
    Ok(())
}

/// Hold workspace `id` open, so its file stays open while the window shows it.
#[tauri::command]
pub(crate) async fn open_workspace(
    state: State<'_, AppState>,
    id: WorkspaceId,
) -> CommandResult<WorkspaceRow> {
    let ws = state.workspace(&id).await?;
    state
        .auditor
        .record(&ws, AuditAction::Open, None, Outcome::Allowed, None)
        .await?;
    Ok(ws.row.clone())
}

/// Let go of workspace `id`. `false` when it was not open.
#[tauri::command]
pub(crate) async fn close_workspace(
    state: State<'_, AppState>,
    id: WorkspaceId,
) -> CommandResult<bool> {
    Ok(state.close_workspace(&id).await)
}
