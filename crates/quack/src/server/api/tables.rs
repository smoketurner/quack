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
use quack_core::storage::workspace::{
    INTERNAL_PREFIX, SqlSchema, TableDescription, TableDescriptionBody, WorkspaceDb,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// The workspace's user tables.
#[derive(Serialize, ToSchema)]
pub(crate) struct TableList {
    pub tables: Vec<String>,
}

/// The user tables' names.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/tables",
    tag = "tables",
    params(WorkspaceId),
    responses((status = 200, description = "The tables", body = TableList)),
)]
pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<TableList>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, AuditAction::List, "tables").await?;
    let tables = app.read(&id, WorkspaceDb::list_tables).await?;
    Ok(Json(TableList { tables }))
}

/// Every user table with its columns, each name as a statement writes it,
/// for the SQL editor's completion. Capped (`SqlSchema::MAX_TABLES` and
/// `MAX_COLUMNS`), with `truncated` set when something was left out.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/tables/schema",
    tag = "tables",
    params(WorkspaceId),
    responses((status = 200, description = "Tables and columns", body = SqlSchema)),
)]
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
#[derive(Deserialize, ToSchema)]
pub(crate) struct DescribeTable {
    pub name: String,
}

/// One table: columns, note, profile, warnings, measures, and sample rows.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/tables/describe",
    tag = "tables",
    request_body = DescribeTable,
    params(WorkspaceId),
    responses((status = 200, description = "The table", body = TableDescriptionBody)),
)]
pub(crate) async fn describe(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<DescribeTable>,
) -> ApiResult<Json<TableDescriptionBody>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let described = access.describe_table(&app, &body.name).await?;
    Ok(Json(described.body()))
}

/// A table's note: blank removes it.
#[derive(Deserialize, ToSchema)]
pub(crate) struct SetNote {
    pub name: String,
    pub note: String,
}

/// Set a table's note; a blank one removes it.
#[utoipa::path(
    put,
    path = "/workspaces/{id}/tables/note",
    tag = "tables",
    request_body = SetNote,
    params(WorkspaceId),
    responses((status = 200, description = "The table as it now is", body = TableDescriptionBody)),
)]
pub(crate) async fn note(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<SetNote>,
) -> ApiResult<Json<TableDescriptionBody>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    access.set_table_note(&app, &body.name, &body.note).await?;
    let described = access.describe_table(&app, &body.name).await?;
    Ok(Json(described.body()))
}

/// A column to give another type.
#[derive(Deserialize, ToSchema)]
pub(crate) struct RetypeColumn {
    pub name: String,
    pub column: String,
    /// `VARCHAR`, `BIGINT`, `DOUBLE`, `DATE`, `TIMESTAMP`, or `BOOLEAN`.
    #[serde(rename = "type")]
    pub to: String,
}

/// Give one column another type; every value must convert.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/tables/retype",
    tag = "tables",
    request_body = RetypeColumn,
    params(WorkspaceId),
    responses((status = 200, description = "The table as it now is", body = TableDescriptionBody)),
)]
pub(crate) async fn retype(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<RetypeColumn>,
) -> ApiResult<Json<TableDescriptionBody>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    access
        .retype_column(&app, &body.name, &body.column, &body.to)
        .await?;
    let described = access.describe_table(&app, &body.name).await?;
    Ok(Json(described.body()))
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
