//! Workspaces: listing by membership, creation by admins, settings by
//! owners, and the content half of the audit for members.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::{Json, response::IntoResponse};
use quack_core::storage::audit;
use quack_core::storage::control::{Outcome, ProviderAllowList, Role, WorkspaceChanges};
use serde::Deserialize;

use crate::server::auth::{Credential, Identity, Need, access, require_admin};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::{App, with_db};

fn workspace_json(
    ws: &quack_core::storage::control::WorkspaceRow,
    role: Option<Role>,
) -> serde_json::Value {
    let allowed: Option<serde_json::Value> = ws
        .allowed_providers
        .as_deref()
        .and_then(|p| serde_json::from_str(p).ok());
    serde_json::json!({
        "id": ws.id,
        "name": ws.name,
        "classification": ws.classification,
        "allowed_providers": allowed,
        "role": role,
    })
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
            .map(|w| workspace_json(&w, Some(Role::Owner)))
            .collect::<Vec<_>>()
    } else if let Credential::Token(token) = &identity.credential {
        let ws = app.control.get_workspace(&token.workspace_id).await?;
        let role = app
            .control
            .member_role(&token.workspace_id, &identity.user_id)
            .await?;
        ws.into_iter().map(|w| workspace_json(&w, role)).collect()
    } else if identity.is_admin {
        let mine = app.control.workspaces_for_user(&identity.user_id).await?;
        let all = app.control.list_workspaces().await?;
        all.into_iter()
            .map(|w| {
                let role = mine.iter().find(|(m, _)| m.id == w.id).map(|(_, r)| *r);
                workspace_json(&w, role)
            })
            .collect()
    } else {
        app.control
            .workspaces_for_user(&identity.user_id)
            .await?
            .into_iter()
            .map(|(w, r)| workspace_json(&w, Some(r)))
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
    require_admin(&identity)?;
    let name = body.name.trim();
    if name.is_empty() || name.contains(['/', '\\', '.']) {
        return Err(ApiError::bad_request(
            "workspace name must be non-empty and contain no slashes or dots",
        ));
    }
    if app.control.find_workspace_by_name(name).await?.is_some() {
        return Err(ApiError::new(StatusCode::CONFLICT, "workspace exists"));
    }
    let ws = app.control.create_workspace(name).await?;
    let role = if app.local {
        Some(Role::Owner)
    } else {
        app.control
            .set_member(&ws.id, &identity.user_id, Role::Owner)
            .await?;
        Some(Role::Owner)
    };
    let mut entry = identity.audit("workspace", Outcome::Allowed);
    entry.workspace_id = Some(ws.id.clone());
    entry.resource_type = Some(String::from("workspace"));
    entry.resource_id = Some(ws.id.clone());
    app.control.record_audit(&entry).await?;
    Ok((StatusCode::CREATED, Json(workspace_json(&ws, role))))
}

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let need = Need {
        admin_ok: true,
        ..Need::READ
    };
    let access = access(&app, identity, &id, need).await?;
    access
        .audit(&app, "open", None, Outcome::Allowed, None)
        .await?;
    Ok(Json(workspace_json(&access.workspace, access.role)))
}

#[derive(Deserialize)]
pub(crate) struct UpdateWorkspace {
    pub classification: Option<String>,
    /// Absent keeps the list; an empty list allows every provider.
    pub allowed_providers: Option<Vec<String>>,
}

pub(crate) async fn update(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Json(body): Json<UpdateWorkspace>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::OWN).await?;
    if let Some(names) = &body.allowed_providers {
        for name in names {
            if !app.config.providers.contains_key(name) {
                return Err(ApiError::bad_request(format!(
                    "'{name}' is not a configured provider"
                )));
            }
        }
    }
    let changes = WorkspaceChanges {
        classification: body.classification.map(|c| c.trim().to_owned()),
        allowed_providers: match body.allowed_providers {
            None => ProviderAllowList::Keep,
            Some(names) if names.is_empty() => ProviderAllowList::All,
            Some(names) => ProviderAllowList::Only(names),
        },
    };
    let ws = app.control.update_workspace(&id, &changes).await?;
    access
        .audit(
            &app,
            "workspace",
            Some(("workspace", &ws.id)),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(Json(workspace_json(&ws, access.role)))
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
    let access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let limit = q.limit;
    let rows = with_db(db, move |db| audit::list(db, limit)).await?;
    access
        .audit(
            &app,
            "session_read",
            Some(("audit", "detail")),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(Json(serde_json::json!({ "audit": rows })))
}
