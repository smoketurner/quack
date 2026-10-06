//! Tables: names, and one table's schema with sample rows.

use axum::Json;
use axum::extract::{Path, State};
use quack_core::ids::WorkspaceId;
use quack_core::storage::control::{AuditAction, Outcome};

use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::{App, with_db};
use quack_core::analysis::table_search;
use quack_core::graph::views;
use quack_core::storage::profile::{ColumnType, Retype, TableNote};
use quack_core::storage::workspace::{INTERNAL_PREFIX, SqlSchema, TableDescription, WorkspaceDb};
use serde::Deserialize;

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

/// Every user table with its columns, each name as a statement writes it,
/// for the SQL editor's completion. Capped (`SqlSchema::MAX_TABLES` and
/// `MAX_COLUMNS`), with `truncated` set when something was left out.
pub(crate) async fn schema(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<SqlSchema>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, AuditAction::List, "schema").await?;
    Ok(Json(app.read(&id, WorkspaceDb::sql_schema).await?))
}

/// The table to describe. In the body, not the path: a table's name is
/// workspace content, and a URL ends up in logs.
#[derive(Deserialize)]
pub(crate) struct DescribeTable {
    pub name: String,
}

pub(crate) async fn describe(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<DescribeTable>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let described = access.describe_table(&app, &body.name).await?;
    Ok(Json(described.to_json()))
}

/// A table's note: blank removes it.
#[derive(Deserialize)]
pub(crate) struct SetNote {
    pub name: String,
    pub note: String,
}

pub(crate) async fn note(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<SetNote>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    access.set_table_note(&app, &body.name, &body.note).await?;
    let described = access.describe_table(&app, &body.name).await?;
    Ok(Json(described.to_json()))
}

/// A column to give another type.
#[derive(Deserialize)]
pub(crate) struct RetypeColumn {
    pub name: String,
    pub column: String,
    #[serde(rename = "type")]
    pub to: String,
}

pub(crate) async fn retype(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<RetypeColumn>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    access
        .retype_column(&app, &body.name, &body.column, &body.to)
        .await?;
    let described = access.describe_table(&app, &body.name).await?;
    Ok(Json(described.to_json()))
}

impl Access {
    /// Set or remove a table's note, from the API or the web console, and
    /// audit it; the note itself goes in the workspace detail.
    pub(crate) async fn set_table_note(&self, app: &App, name: &str, note: &str) -> ApiResult<()> {
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let (table, text, editor) = (
            name.to_owned(),
            note.to_owned(),
            self.identity.username.clone(),
        );
        let result = with_db(db, move |db| {
            TableNote::set(db, &table, &text, Some(&editor))
        })
        .await;
        self.audit(
            app,
            AuditAction::TableNote,
            None,
            Outcome::of(&result),
            Some(serde_json::json!({ "table": name, "chars": note.trim().chars().count() })),
        )
        .await?;
        result
    }

    /// Give one column of a user table another type, every value
    /// converting, and audit it.
    pub(crate) async fn retype_column(
        &self,
        app: &App,
        name: &str,
        column: &str,
        to: &str,
    ) -> ApiResult<()> {
        let to: ColumnType = to.parse()?;
        if name.starts_with(INTERNAL_PREFIX) || views::is_reserved(name) {
            return Err(ApiError::not_found("no such table"));
        }
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let (table, col) = (name.to_owned(), column.to_owned());
        let result = with_db(db, move |db| {
            if !table_search::user_tables(db)?.contains(&table) {
                return Ok(None);
            }
            Retype {
                table: &table,
                column: &col,
                to,
            }
            .run(db)
            .map(Some)
        })
        .await;
        let detail = serde_json::json!({ "table": name, "column": column, "type": to });
        self.audit(
            app,
            AuditAction::Retype,
            None,
            Outcome::of(&result),
            Some(detail),
        )
        .await?;
        result?.ok_or_else(|| ApiError::not_found("no such table"))
    }

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
            .read(&self.membership.workspace.id, move |db| {
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
