//! Membership: owners and admins add, change, and remove members.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use quack_core::ids::{UserId, WorkspaceId};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind, Role};
use serde::{Deserialize, Serialize};

use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::App;

pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ_OR_ADMIN).await?;
    access
        .audit_read(&app, AuditAction::List, "members")
        .await?;
    let members = app.control.list_members(&id).await?;
    Ok(Json(serde_json::json!({ "members": members })))
}

#[derive(Deserialize)]
pub(crate) struct AddMember {
    pub username: String,
    #[serde(default = "default_role")]
    pub role: Role,
}

fn default_role() -> Role {
    Role::Member
}

pub(crate) async fn add(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<AddMember>,
) -> ApiResult<Json<NewMember>> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    Ok(Json(access.add_member(&app, &body).await?))
}

pub(crate) async fn remove(
    State(app): State<App>,
    identity: Identity,
    Path((id, user_id)): Path<(WorkspaceId, UserId)>,
) -> ApiResult<StatusCode> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    access.remove_member(&app, &user_id).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// A member as added.
#[derive(Debug, Serialize)]
pub(crate) struct NewMember {
    pub user_id: UserId,
    pub username: String,
    pub role: Role,
}

impl Access {
    /// Give a user a role in this workspace, from the API or the web
    /// console; an existing member's role changes.
    pub(crate) async fn add_member(&self, app: &App, member: &AddMember) -> ApiResult<NewMember> {
        let user = app
            .control
            .find_user_by_username(&member.username)
            .await?
            .ok_or_else(|| ApiError::not_found("no such user"))?;
        app.control
            .set_member(&self.workspace.id, &user.id, member.role)
            .await?;
        self.audit(
            app,
            AuditAction::Member,
            Some(ResourceKind::User.id(user.id.as_str())),
            Outcome::Allowed,
            None,
        )
        .await?;
        Ok(NewMember {
            user_id: user.id,
            username: user.username,
            role: member.role,
        })
    }

    /// Take a user out of this workspace; one who was not a member is an
    /// error, after the failed attempt is audited as such. `Allowed` is
    /// for work that succeeded (`Outcome::of`): a no-op removal that
    /// answers `404` is `Error`.
    pub(crate) async fn remove_member(&self, app: &App, user_id: &UserId) -> ApiResult<()> {
        let removed = app
            .control
            .remove_member(&self.workspace.id, user_id)
            .await?;
        let (outcome, detail) = if removed {
            (Outcome::Allowed, None)
        } else {
            (
                Outcome::Error,
                Some(serde_json::json!({ "reason": "not a member" })),
            )
        };
        self.audit(
            app,
            AuditAction::Member,
            Some(ResourceKind::User.id(user_id.as_str())),
            outcome,
            detail,
        )
        .await?;
        if removed {
            Ok(())
        } else {
            Err(ApiError::not_found("not a member"))
        }
    }
}
