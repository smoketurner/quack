//! The workspace context: read, replace (a new version), and history.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, header};
use axum::response::{IntoResponse, Response};
use quack_core::storage::context::{self, ContextVersion};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use serde::Deserialize;

use crate::server::auth::{Access, Identity, Need, access};
use crate::server::error::ApiResult;
use crate::server::state::{App, with_db};

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
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
    Ok(Json(serde_json::json!({ "context": current })).into_response())
}

#[derive(Deserialize)]
pub(crate) struct ReplaceContext {
    pub content: String,
}

pub(crate) async fn replace(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Json(body): Json<ReplaceContext>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let stored = access.save_context(&app, body.content).await?;
    Ok(Json(serde_json::json!({ "context": stored })))
}

impl Access {
    /// Store `content` as the next version of the workspace context, from
    /// the API or the web console, and audit it.
    pub(crate) async fn save_context(
        &self,
        app: &App,
        content: String,
    ) -> ApiResult<ContextVersion> {
        let db = app.workspace_db(&self.workspace.id).await?;
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

#[derive(Deserialize)]
pub(crate) struct VersionsQuery {
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    20
}

pub(crate) async fn versions(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Query(q): Query<VersionsQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "context_versions")
        .await?;
    let limit = q.limit;
    let history = app.read(&id, move |db| context::history(db, limit)).await?;
    Ok(Json(serde_json::json!({ "versions": history })))
}
