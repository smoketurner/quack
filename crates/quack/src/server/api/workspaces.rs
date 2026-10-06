//! Workspaces: listing by membership, creation by admins, settings by
//! owners, and the content half of the audit for members.

use std::collections::{BTreeSet, HashMap};

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use quack_core::ids::{AuditId, WorkspaceId};
use quack_core::ocsf::PromptText;
use quack_core::storage::audit::{self, AuditDetailRow};
use quack_core::storage::control::{
    AuditAction, AuditRow, ControlPlane, Membership, Outcome, ProviderAllowList, ResourceKind,
    Role, Standing, UserKind, WorkspaceChanges, WorkspaceName, WorkspaceRow,
};
use serde::Deserialize;

use crate::server::auth::{Access, Credential, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::{App, ServeMode};

pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
) -> ApiResult<Json<serde_json::Value>> {
    let rows = if app.mode == ServeMode::Local {
        app.control
            .list_workspaces()
            .await?
            .into_iter()
            .map(|w| Membership {
                workspace: w,
                standing: Standing::Member(Role::Owner),
            })
            .collect::<Vec<_>>()
    } else if let Credential::Token(token) = &identity.credential {
        let ws = app.control.get_workspace(&token.workspace_id).await?;
        let role = app
            .control
            .member_role(&token.workspace_id, &identity.user_id)
            .await?;
        ws.into_iter()
            .filter(|_| role.is_some() || identity.kind == UserKind::Admin)
            .map(|w| Membership {
                workspace: w,
                standing: Standing::of(role),
            })
            .collect()
    } else if identity.kind == UserKind::Admin {
        let mine = app.control.workspaces_for_user(&identity.user_id).await?;
        let all = app.control.list_workspaces().await?;
        all.into_iter()
            .map(|w| {
                let standing = mine
                    .iter()
                    .find(|m| m.workspace.id == w.id)
                    .map_or(Standing::Admin, |m| m.standing);
                Membership {
                    workspace: w,
                    standing,
                }
            })
            .collect()
    } else {
        app.control.workspaces_for_user(&identity.user_id).await?
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
        Json(Membership {
            workspace: ws,
            standing: Standing::Member(Role::Owner),
        }),
    ))
}

impl Identity {
    /// Create a workspace, from the API or the web console: admins only, a
    /// [`WorkspaceName`] that is not taken; the creator becomes its owner
    /// (in local mode everyone already is).
    pub(crate) async fn create_workspace(&self, app: &App, name: &str) -> ApiResult<WorkspaceRow> {
        self.require_admin()?;
        let name: WorkspaceName = name.parse()?;
        let owner = (app.mode == ServeMode::Login).then_some(&self.user_id);
        let entry = self.audit(AuditAction::Workspace, Outcome::Allowed);
        Ok(app.control.create_workspace(&name, owner, entry).await?)
    }
}

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<Membership>> {
    let access = Access::resolve(&app, identity, &id, Need::READ_OR_ADMIN).await?;
    access
        .audit(&app, AuditAction::Open, None, Outcome::Allowed, None)
        .await?;
    Ok(Json(access.membership))
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
    Path(id): Path<WorkspaceId>,
    Json(body): Json<UpdateWorkspace>,
) -> ApiResult<Json<Membership>> {
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
    Ok(Json(Membership {
        workspace: ws,
        standing: access.membership.standing,
    }))
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
        .update_workspace(&access.membership.workspace.id, &changes)
        .await?;
    access
        .audit(
            app,
            AuditAction::Workspace,
            Some(ResourceKind::Workspace.id(ws.id.as_str())),
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
    /// `ocsf`: each detail row joined to its access row as an OCSF event
    /// (the `ai_operation` profile on queries); absent, the detail rows.
    pub format: Option<String>,
    /// With `format=ocsf`, carry the question's text on query events.
    #[serde(default)]
    pub prompt: bool,
}

fn default_limit() -> u32 {
    100
}

/// The content half of the audit, for members only (design doc 12).
pub(crate) async fn audit_detail(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
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
    match q.format.as_deref() {
        None => Ok(Json(serde_json::json!({ "audit": rows }))),
        Some("ocsf") => {
            let prompt = if q.prompt {
                PromptText::Include
            } else {
                PromptText::Omit
            };
            let events = ocsf_events(&app.control, &rows, prompt).await?;
            Ok(Json(serde_json::json!({ "audit": events })))
        }
        Some(other) => Err(ApiError::bad_request(format!(
            "format must be ocsf, not {other}"
        ))),
    }
}

/// A workspace's detail rows joined to their access rows by the shared
/// id, rendered as OCSF events, newest first; a detail row whose access
/// row is gone is left out.
pub(crate) async fn ocsf_events(
    control: &ControlPlane,
    details: &[AuditDetailRow],
    prompt: PromptText,
) -> ApiResult<Vec<serde_json::Value>> {
    let ids: Vec<AuditId> = details.iter().map(|d| d.id.clone()).collect();
    let access_rows = control.audit_rows_by_ids(&ids).await?;
    let by_id: HashMap<&AuditId, &AuditRow> =
        access_rows.iter().map(|r| (&r.entry.id, r)).collect();
    let mut events = Vec::with_capacity(details.len());
    for detail in details {
        if let Some(row) = by_id.get(&detail.id) {
            events.push(row.to_ocsf_with_detail(Some(detail), prompt)?);
        }
    }
    Ok(events)
}
