//! Workspaces: listing by membership, creation, restoring, and deletion by
//! admins, settings and snapshots by owners, and the content half of the
//! audit for members.

use std::collections::BTreeSet;
use std::io::{self, Write};
use std::sync::Arc;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use quack_core::ids::WorkspaceId;
use quack_core::ocsf::PromptText;
use quack_core::okf;
use quack_core::storage::audit;
use quack_core::storage::backup::{Described, Manifest, RestoreRequest};
use quack_core::storage::control::{
    AuditAction, Membership, Outcome, ProviderAllowList, ResourceKind, Role, Standing, UserKind,
    WorkspaceChanges, WorkspaceName, WorkspaceRow,
};
use serde::Deserialize;
use tokio::sync::mpsc;

use crate::server::api::okf::{BodyWriter, CHUNKS_IN_FLIGHT};
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
    /// A new name, which `-w` and the URL bar then use.
    pub name: Option<String>,
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
    let ws = access.update_settings(&app, changes).await?;
    let ws = match body.name {
        Some(name) => access.rename_workspace(&app, &name).await?,
        None => ws,
    };
    Ok(Json(Membership {
        workspace: ws,
        standing: access.membership.standing,
    }))
}

/// `GET .../snapshot`: the workspace as a tar for its owner, written on
/// the writer's thread after a checkpoint and streamed to the body as
/// the OKF export is. Audited as `snapshot`, since it moves the whole
/// workspace across the boundary.
pub(crate) async fn snapshot(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    let writer = app.workspace_db(&id).await?;
    let described = Described::of(&app.control, &access.membership.workspace).await?;
    let filename = format!("{}.snapshot.tar", okf::slug(&described.name));
    let dir = app.config.workspace_dir(id.as_str());
    let (tx, rx) = mpsc::channel::<io::Result<Bytes>>(CHUNKS_IN_FLIGHT);
    let failed = tx.clone();
    let audit_app = Arc::clone(&app);
    tokio::spawn(async move {
        let written = writer
            .run(move |db| {
                let manifest = Manifest::of(db, described)?;
                manifest.write(db, &dir, BodyWriter::new(tx))?.flush()?;
                Ok(())
            })
            .await;
        let (outcome, detail) = match written {
            Ok(()) => (
                Outcome::Allowed,
                serde_json::json!({ "format": "snapshot" }),
            ),
            Err(e) => {
                tracing::warn!(workspace = %id, error = %e, "snapshot failed partway");
                drop(failed.send(Err(io::Error::other(e.to_string()))).await);
                (
                    Outcome::Error,
                    serde_json::json!({ "format": "snapshot", "error": e.to_string() }),
                )
            }
        };
        drop(failed);
        let recorded = access
            .audit(
                &audit_app,
                AuditAction::Snapshot,
                Some(ResourceKind::Workspace.id(id.as_str())),
                outcome,
                Some(detail),
            )
            .await;
        if let Err(e) = recorded {
            tracing::error!(workspace = %id, error = %e.message, "could not audit a snapshot");
        }
    });
    let body = Body::from_stream(futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (chunk, rx))
    }));
    Ok((
        [
            (header::CONTENT_TYPE, String::from("application/x-tar")),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{filename}\""),
            ),
        ],
        body,
    )
        .into_response())
}

#[derive(Deserialize)]
pub(crate) struct RestoreQuery {
    /// The new workspace's name; the snapshot's own when absent.
    pub name: Option<String>,
}

/// `POST /workspaces/restore`: a snapshot's tar as the body becomes a new
/// workspace, for admins; in login mode the admin owns it beside the
/// snapshot's members. The body is bounded by `[server].max_upload_mb`.
pub(crate) async fn restore(
    State(app): State<App>,
    identity: Identity,
    Query(query): Query<RestoreQuery>,
    body: Bytes,
) -> ApiResult<impl IntoResponse> {
    identity.require_admin()?;
    let name = query.name.map(|n| n.parse::<WorkspaceName>()).transpose()?;
    let owner = (app.mode == ServeMode::Login).then(|| identity.user_id.clone());
    let audit = |action| identity.audit(action, Outcome::Allowed);
    let restored = RestoreRequest {
        name,
        owner: owner.as_ref(),
        audit: &audit,
    }
    .run(&app.control, &app.config, move || {
        Ok(io::Cursor::new(body.clone()))
    })
    .await?;
    Ok((StatusCode::CREATED, Json(restored)))
}

/// `DELETE /workspaces/{id}`: the workspace's row (its members and
/// tokens with it) and its directory, for an owner, once no job of it is
/// active. The access row is committed with the row; no detail row can
/// follow it into a file that no longer exists.
pub(crate) async fn delete(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<StatusCode> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    access.delete_workspace(&app).await?;
    Ok(StatusCode::NO_CONTENT)
}

impl Access {
    /// Change a workspace's settings for its owner, from the API or the web
    /// console: every allowed provider must be configured, the classification
    /// is trimmed, and the change is audited.
    pub(crate) async fn update_settings(
        &self,
        app: &App,
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
            .update_workspace(&self.membership.workspace.id, &changes)
            .await?;
        self.audit(
            app,
            AuditAction::Workspace,
            Some(ResourceKind::Workspace.id(ws.id.as_str())),
            Outcome::Allowed,
            None,
        )
        .await?;
        Ok(ws)
    }

    /// Rename the workspace for its owner, audited as a settings change; a
    /// name that is already the workspace's own changes nothing.
    pub(crate) async fn rename_workspace(&self, app: &App, name: &str) -> ApiResult<WorkspaceRow> {
        let name: WorkspaceName = name.parse()?;
        if name.as_str() == self.membership.workspace.name {
            return Ok(self.membership.workspace.clone());
        }
        let entry = self.entry(AuditAction::Workspace, Outcome::Allowed);
        let ws = app
            .control
            .rename_workspace(&self.membership.workspace.id, &name, entry.clone())
            .await?;
        self.record_detail(
            app,
            &entry,
            Some(serde_json::json!({ "renamed_to": name.as_str() })),
        )
        .await?;
        Ok(ws)
    }

    /// Delete the workspace an `Access` names, from the API or the web
    /// console; see [`delete`].
    pub(crate) async fn delete_workspace(&self, app: &App) -> ApiResult<()> {
        let id = &self.membership.workspace.id;
        app.close_workspace(id).await?;
        let entry = self.entry(AuditAction::Delete, Outcome::Allowed);
        app.control.delete_workspace(id, entry).await?;
        let dir = app.config.workspace_dir(id.as_str());
        if dir.exists()
            && let Err(e) = tokio::fs::remove_dir_all(&dir).await
        {
            tracing::error!(workspace = %id, error = %e, "the deleted workspace's directory remains");
            return Err(ApiError::internal(format!(
                "the workspace is gone from control.db, but its directory could not be removed: {e}"
            )));
        }
        Ok(())
    }
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
            let events = app
                .control
                .ocsf_events(&rows, PromptText::from(q.prompt))
                .await?;
            Ok(Json(serde_json::json!({ "audit": events })))
        }
        Some(other) => Err(ApiError::bad_request(format!(
            "format must be ocsf, not {other}"
        ))),
    }
}
