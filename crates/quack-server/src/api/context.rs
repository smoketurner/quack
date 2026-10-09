//! The workspace context: read, replace (a new version), and history.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, header};
use axum::response::{IntoResponse, Response};
use quack_core::ids::WorkspaceId;
use quack_core::storage::context::{self, ContextVersion};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::auth::{Access, Identity, Need};
use crate::error::ApiResult;
use crate::state::{App, with_db};

/// The workspace context, if any version was saved.
#[derive(Serialize, ToSchema)]
pub(crate) struct CurrentContext {
    pub context: Option<ContextVersion>,
}

/// The current context, as JSON or, with `Accept: text/markdown`, as its
/// text.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/context",
    tag = "context",
    params(WorkspaceId),
    responses((status = 200, description = "The current context", content(
        (CurrentContext = "application/json"),
        (String = "text/markdown"),
    ))),
)]
pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    headers: HeaderMap,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "context")
        .await?;
    let current = app.read(&id, context::current).await?;
    let wants_markdown = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|a| a.contains("text/markdown"));
    if wants_markdown {
        let body = current.map(|c| c.content).unwrap_or_default();
        return Ok((
            [(header::CONTENT_TYPE, "text/markdown; charset=utf-8")],
            body,
        )
            .into_response());
    }
    Ok(Json(CurrentContext { context: current }).into_response())
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct ReplaceContext {
    pub content: String,
}

/// The saved context.
#[derive(Serialize, ToSchema)]
pub(crate) struct SavedContext {
    pub context: ContextVersion,
}

/// Save `content` as the context's next version.
#[utoipa::path(
    put,
    path = "/workspaces/{id}/context",
    tag = "context",
    request_body = ReplaceContext,
    params(WorkspaceId),
    responses((status = 200, description = "The new version", body = SavedContext)),
)]
pub(crate) async fn replace(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<ReplaceContext>,
) -> ApiResult<Json<SavedContext>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let stored = access.save_context(&app, body.content).await?;
    Ok(Json(SavedContext { context: stored }))
}

impl Access {
    /// Store `content` as the next version of the workspace context, from
    /// the API or the web console, and audit it.
    pub(crate) async fn save_context(
        &self,
        app: &App,
        content: String,
    ) -> ApiResult<ContextVersion> {
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let editor = self.identity.username.clone();
        let stored = with_db(db, move |db| context::set(db, &content, Some(&editor))).await?;
        self.audit(
            app,
            AuditAction::Context,
            Some(ResourceKind::Context.id(&stored.version.to_string())),
            Outcome::Allowed,
            Some(serde_json::json!({ "version": stored.version, "chars": stored.content.chars().count() })),
        )
        .await?;
        Ok(stored)
    }
}

#[derive(Deserialize, utoipa::IntoParams)]
pub(crate) struct VersionsQuery {
    /// Versions at most, newest first.
    #[serde(default = "default_limit")]
    #[param(default = 20)]
    pub limit: u32,
}

fn default_limit() -> u32 {
    20
}

/// The context's versions, newest first.
#[derive(Serialize, ToSchema)]
pub(crate) struct ContextHistory {
    pub versions: Vec<ContextVersion>,
}

/// The context's saved versions, newest first.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/context/versions",
    tag = "context",
    params(WorkspaceId, VersionsQuery),
    responses((status = 200, description = "The versions", body = ContextHistory)),
)]
pub(crate) async fn versions(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Query(q): Query<VersionsQuery>,
) -> ApiResult<Json<ContextHistory>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "context_versions")
        .await?;
    let limit = q.limit;
    let history = app.read(&id, move |db| context::history(db, limit)).await?;
    Ok(Json(ContextHistory { versions: history }))
}
