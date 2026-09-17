//! Tables: names, and one table's schema with sample rows.

use axum::Json;
use axum::extract::{Path, State};
use quack_core::storage::control::Outcome;

use crate::server::auth::{Identity, Need, access};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::{App, with_db};
use quack_core::storage::workspace::WorkspaceDb;

pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, "list", "tables").await?;
    let db = app.workspace_db(&id).await?;
    let tables = with_db(db, WorkspaceDb::list_tables).await?;
    Ok(Json(serde_json::json!({ "tables": tables })))
}

pub(crate) async fn describe(
    State(app): State<App>,
    identity: Identity,
    Path((id, name)): Path<(String, String)>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    if name.starts_with("_quack_") {
        return Err(ApiError::not_found("no such table"));
    }
    let db = app.workspace_db(&id).await?;
    let table = name.clone();
    let described = with_db(db, move |db| {
        if !db.list_tables()?.contains(&table) {
            return Ok(None);
        }
        db.describe_table(&table).map(Some)
    })
    .await?;
    let described = described.ok_or_else(|| ApiError::not_found("no such table"))?;
    // The table name is user content: it goes in the workspace detail,
    // not in control.db.
    access
        .audit(
            &app,
            "open",
            None,
            Outcome::Allowed,
            Some(serde_json::json!({ "table": name })),
        )
        .await?;
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
