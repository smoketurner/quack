//! Tables: names, and one table's schema with sample rows.

use axum::Json;
use axum::extract::{Path, State};
use quack_core::ids::WorkspaceId;
use quack_core::storage::control::{AuditAction, Outcome};

use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::App;
use quack_core::storage::workspace::{INTERNAL_PREFIX, TableDescription, WorkspaceDb};

pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, AuditAction::List, "tables").await?;
    let tables = app.read(&id, WorkspaceDb::list_tables).await?;
    Ok(Json(serde_json::json!({ "tables": tables })))
}

pub(crate) async fn describe(
    State(app): State<App>,
    identity: Identity,
    Path((id, name)): Path<(WorkspaceId, String)>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let described = access.describe_table(&app, &name).await?;
    let columns: Vec<serde_json::Value> = described
        .columns
        .iter()
        .map(|c| serde_json::json!({ "name": c.name, "type": c.column_type }))
        .collect();
    Ok(Json(serde_json::json!({
        "table": described.table_name,
        "columns": columns,
        "sample": { "columns": described.sample_rows.columns, "rows": described.sample_rows.rows },
    })))
}

impl Access {
    /// One user table's columns and sample rows, from the API or the web
    /// console; an internal or missing table is not found.
    pub(crate) async fn describe_table(
        &self,
        app: &App,
        name: &str,
    ) -> ApiResult<TableDescription> {
        if name.starts_with(INTERNAL_PREFIX) {
            return Err(ApiError::not_found("no such table"));
        }
        let table = name.to_owned();
        let described = app
            .read(&self.workspace.id, move |db| {
                if !db.list_tables()?.contains(&table) {
                    return Ok(None);
                }
                db.describe_table(&table).map(Some)
            })
            .await?
            .ok_or_else(|| ApiError::not_found("no such table"))?;
        // The table name is user content: it goes in the workspace detail,
        // not in control.db.
        self.audit(
            app,
            AuditAction::Open,
            None,
            Outcome::Allowed,
            Some(serde_json::json!({ "table": name })),
        )
        .await?;
        Ok(described)
    }
}
