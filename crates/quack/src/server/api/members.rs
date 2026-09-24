//! Membership: owners and admins add, change, and remove members.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind, Role};
use serde::{Deserialize, Serialize};

use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::App;

pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let need = Need {
        admin_ok: true,
        ..Need::READ
    };
    let access = Access::resolve(&app, identity, &id, need).await?;
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
    Path(id): Path<String>,
    Json(body): Json<AddMember>,
) -> ApiResult<Json<NewMember>> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    Ok(Json(access.add_member(&app, &body).await?))
}

pub(crate) async fn remove(
    State(app): State<App>,
    identity: Identity,
    Path((id, user_id)): Path<(String, String)>,
) -> ApiResult<StatusCode> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    access.remove_member(&app, &user_id).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// A member as added.
#[derive(Debug, Serialize)]
pub(crate) struct NewMember {
    pub user_id: String,
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
            Some(ResourceKind::User.id(&user.id)),
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
    /// error, after the attempt is audited.
    pub(crate) async fn remove_member(&self, app: &App, user_id: &str) -> ApiResult<()> {
        let removed = app
            .control
            .remove_member(&self.workspace.id, user_id)
            .await?;
        self.audit(
            app,
            AuditAction::Member,
            Some(ResourceKind::User.id(user_id)),
            Outcome::Allowed,
            None,
        )
        .await?;
        if removed {
            Ok(())
        } else {
            Err(ApiError::not_found("not a member"))
        }
    }
}
