//! Workspaces: listing by membership, creation by admins, settings by
//! owners, and the content half of the audit for members.

use std::collections::BTreeSet;

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use quack_core::storage::audit;
use quack_core::storage::control::{
    AuditAction, Outcome, ProviderAllowList, ResourceKind, Role, WorkspaceChanges, WorkspaceRow,
};
use serde::{Deserialize, Serialize};

use crate::server::auth::{Access, Credential, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::App;

/// A workspace as the API shows it: its row, and the caller's role in it.
#[derive(Debug, Serialize)]
pub(crate) struct WorkspaceView {
    #[serde(flatten)]
    workspace: WorkspaceRow,
    role: Option<Role>,
}

impl WorkspaceView {
    fn new(workspace: WorkspaceRow, role: Option<Role>) -> Self {
        Self { workspace, role }
    }
}

pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
) -> ApiResult<Json<serde_json::Value>> {
    let rows = if app.local {
        app.control
            .list_workspaces()
            .await?
            .into_iter()
            .map(|w| WorkspaceView::new(w, Some(Role::Owner)))
            .collect::<Vec<_>>()
    } else if let Credential::Token(token) = &identity.credential {
        let ws = app.control.get_workspace(&token.workspace_id).await?;
        let role = app
            .control
            .member_role(&token.workspace_id, &identity.user_id)
            .await?;
        ws.into_iter()
            .map(|w| WorkspaceView::new(w, role))
            .collect()
    } else if identity.is_admin {
        let mine = app.control.workspaces_for_user(&identity.user_id).await?;
        let all = app.control.list_workspaces().await?;
        all.into_iter()
            .map(|w| {
                let role = mine.iter().find(|(m, _)| m.id == w.id).map(|(_, r)| *r);
                WorkspaceView::new(w, role)
            })
            .collect()
    } else {
        app.control
            .workspaces_for_user(&identity.user_id)
            .await?
            .into_iter()
            .map(|(w, r)| WorkspaceView::new(w, Some(r)))
            .collect()
    };
    Ok(Json(serde_json::json!({ "workspaces": rows })))
}

#[derive(Deserialize)]
pub(crate) struct CreateWorkspace {
    pub name: String,
}

pub(crate) async fn create(
    State(app): State<App>,
    identity: Identity,
    Json(body): Json<CreateWorkspace>,
) -> ApiResult<impl IntoResponse> {
    let ws = identity.create_workspace(&app, &body.name).await?;
    Ok((
        StatusCode::CREATED,
        Json(WorkspaceView::new(ws, Some(Role::Owner))),
    ))
}

impl Identity {
    /// Create a workspace, from the API or the web console: admins only, a
    /// name without slashes or dots that is not taken; the creator becomes
    /// its owner (in local mode everyone already is).
    pub(crate) async fn create_workspace(&self, app: &App, name: &str) -> ApiResult<WorkspaceRow> {
        self.require_admin()?;
        let name = name.trim();
        if name.is_empty() || name.contains(['/', '\\', '.']) {
            return Err(ApiError::bad_request(
                "workspace name must be non-empty and contain no slashes or dots",
            ));
        }
        if app.control.find_workspace_by_name(name).await?.is_some() {
            return Err(ApiError::conflict("workspace exists"));
        }
        let ws = app.control.create_workspace(name).await?;
        if !app.local {
            app.control
                .set_member(&ws.id, &self.user_id, Role::Owner)
                .await?;
        }
        let mut entry = self.audit(AuditAction::Workspace, Outcome::Allowed);
        entry.workspace_id = Some(ws.id.clone());
        entry = entry.on(ResourceKind::Workspace.id(&ws.id));
        app.control.record_audit(&entry).await?;
        Ok(ws)
    }
}

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<WorkspaceView>> {
    let access = Access::resolve(&app, identity, &id, Need::READ_OR_ADMIN).await?;
    access
        .audit(&app, AuditAction::Open, None, Outcome::Allowed, None)
        .await?;
    Ok(Json(WorkspaceView::new(access.workspace, access.role)))
}

#[derive(Deserialize)]
pub(crate) struct UpdateWorkspace {
    pub classification: Option<String>,
    /// Absent keeps the list; an empty list allows every provider.
    pub allowed_providers: Option<BTreeSet<String>>,
}

pub(crate) async fn update(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Json(body): Json<UpdateWorkspace>,
) -> ApiResult<Json<WorkspaceView>> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    let changes = WorkspaceChanges {
        classification: body.classification,
        allowed_providers: match body.allowed_providers {
            None => ProviderAllowList::Keep,
            Some(names) if names.is_empty() => ProviderAllowList::All,
            Some(names) => ProviderAllowList::Only(names),
        },
    };
    let ws = update_settings(&app, &access, changes).await?;
    Ok(Json(WorkspaceView::new(ws, access.role)))
}

/// Change a workspace's settings for its owner, from the API or the web
/// console: every allowed provider must be configured, the classification
/// is trimmed, and the change is audited.
pub(crate) async fn update_settings(
    app: &App,
    access: &Access,
    changes: WorkspaceChanges,
) -> ApiResult<WorkspaceRow> {
    if let ProviderAllowList::Only(names) = &changes.allowed_providers
        && let Some(unknown) = names
            .iter()
            .find(|name| !app.config.providers.contains_key(name.as_str()))
    {
        return Err(ApiError::bad_request(format!(
            "'{unknown}' is not a configured provider"
        )));
    }
    let changes = WorkspaceChanges {
        classification: changes.classification.map(|c| c.trim().to_owned()),
        ..changes
    };
    let ws = app
        .control
        .update_workspace(&access.workspace.id, &changes)
        .await?;
    access
        .audit(
            app,
            AuditAction::Workspace,
            Some(ResourceKind::Workspace.id(&ws.id)),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(ws)
}

#[derive(Deserialize)]
pub(crate) struct AuditQuery {
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    100
}

/// The content half of the audit, for members only (design doc 12).
pub(crate) async fn audit_detail(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Query(q): Query<AuditQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let limit = q.limit;
    let rows = app.read(&id, move |db| audit::list(db, limit)).await?;
    access
        .audit(
            &app,
            AuditAction::SessionRead,
            Some(ResourceKind::Audit.id("detail")),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(Json(serde_json::json!({ "audit": rows })))
}
