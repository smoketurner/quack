//! Membership: owners and admins add, change, and remove members.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use quack_core::ids::{UserId, WorkspaceId};
use quack_core::storage::control::{
    AuditAction, GroupRoleRow, MemberRow, Outcome, ResourceKind, Role, Standing, UserKind,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::auth::{Access, Identity, Need};
use crate::error::{ApiError, ApiResult};
use crate::state::App;

/// The workspace's members.
#[derive(Serialize, ToSchema)]
pub(crate) struct MemberList {
    pub members: Vec<MemberRow>,
}

/// The workspace's members and their roles.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/members",
    tag = "members",
    params(WorkspaceId),
    responses((status = 200, description = "The members", body = MemberList)),
)]
pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<MemberList>> {
    let access = Access::resolve(&app, identity, &id, Need::READ_OR_ADMIN).await?;
    access
        .audit_read(&app, AuditAction::List, "members")
        .await?;
    let members = app.control.list_members(&id).await?;
    Ok(Json(MemberList { members }))
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct AddMember {
    pub username: String,
    #[serde(default = "default_role")]
    #[schema(default = "member")]
    pub role: Role,
    /// Why, when an admin grants themself a role in a workspace they are
    /// not a member of; required then, ignored otherwise.
    #[serde(default)]
    pub reason: Option<String>,
}

fn default_role() -> Role {
    Role::Member
}

/// Give a user a role here; an existing member's role changes.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/members",
    tag = "members",
    request_body = AddMember,
    params(WorkspaceId),
    responses((status = 200, description = "The member", body = NewMember)),
)]
pub(crate) async fn add(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<AddMember>,
) -> ApiResult<Json<NewMember>> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    Ok(Json(access.add_member(&app, &body).await?))
}

/// Remove a member.
#[utoipa::path(
    delete,
    path = "/workspaces/{id}/members/{user}",
    tag = "members",
    responses((status = 204, description = "Removed")),
)]
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
#[derive(Debug, Serialize, ToSchema)]
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
        // An admin with no membership giving themself a role: membership by
        // admin right, marked as such and never without a reason.
        let self_grant = self.membership.standing == Standing::Admin
            && self.identity.kind == UserKind::Admin
            && user.id == self.identity.user_id;
        let reason = member
            .reason
            .as_deref()
            .map(str::trim)
            .filter(|r| !r.is_empty());
        let (action, detail) = if self_grant {
            let Some(reason) = reason else {
                return Err(ApiError::bad_request(
                    "granting yourself access to a workspace you are not a member of needs a reason",
                ));
            };
            (
                AuditAction::BreakGlass,
                serde_json::json!({ "role": member.role, "reason": reason, "acting_as": "admin" }),
            )
        } else {
            (
                AuditAction::Member,
                serde_json::json!({ "role": member.role }),
            )
        };
        let entry = self.entry(action, Outcome::Allowed);
        app.control
            .set_member(
                &self.membership.workspace.id,
                &user.id,
                member.role,
                entry.clone(),
            )
            .await?;
        self.record_detail(app, &entry, Some(detail)).await?;
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
        let entry = self.entry(AuditAction::Member, Outcome::Allowed);
        let removed = app
            .control
            .remove_member(&self.membership.workspace.id, user_id, entry.clone())
            .await?;
        let detail = (!removed).then(|| serde_json::json!({ "reason": "not a member" }));
        self.record_detail(app, &entry, detail).await?;
        if removed {
            Ok(())
        } else {
            Err(ApiError::not_found("not a member"))
        }
    }
}

/// The identity provider's groups with a role here.
#[derive(Serialize, ToSchema)]
pub(crate) struct GroupList {
    pub groups: Vec<GroupRoleRow>,
}

/// `GET .../groups`: the identity provider's groups with a role here.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/groups",
    tag = "members",
    params(WorkspaceId),
    responses((status = 200, description = "The groups", body = GroupList)),
)]
pub(crate) async fn groups(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<GroupList>> {
    let access = Access::resolve(&app, identity, &id, Need::READ_OR_ADMIN).await?;
    access.audit_read(&app, AuditAction::List, "groups").await?;
    let groups = app.control.list_group_roles(&id).await?;
    Ok(Json(GroupList { groups }))
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct GroupRole {
    pub group: String,
    #[serde(default = "default_role")]
    #[schema(default = "member")]
    pub role: Role,
}

/// Give a group a role here, or change it; members already signed in get
/// it at their next sign-in.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/groups",
    tag = "members",
    request_body = GroupRole,
    params(WorkspaceId),
    responses((status = 200, description = "The group's role", body = GroupRoleRow)),
)]
pub(crate) async fn set_group(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<GroupRole>,
) -> ApiResult<Json<GroupRoleRow>> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    Ok(Json(access.set_group_role(&app, &body).await?))
}

/// Take a group's role here away.
#[utoipa::path(
    delete,
    path = "/workspaces/{id}/groups/{group}",
    tag = "members",
    responses((status = 204, description = "Removed")),
)]
pub(crate) async fn remove_group(
    State(app): State<App>,
    identity: Identity,
    Path((id, group)): Path<(WorkspaceId, String)>,
) -> ApiResult<StatusCode> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    access.remove_group_role(&app, &group).await?;
    Ok(StatusCode::NO_CONTENT)
}

impl Access {
    /// Give a group a role here, or change it; members already signed in
    /// get it at their next sign-in.
    pub(crate) async fn set_group_role(
        &self,
        app: &App,
        body: &GroupRole,
    ) -> ApiResult<GroupRoleRow> {
        let entry = self.entry(AuditAction::Member, Outcome::Allowed);
        let row = app
            .control
            .set_group_role(
                &self.membership.workspace.id,
                &body.group,
                body.role,
                entry.clone(),
            )
            .await?;
        self.record_detail(
            app,
            &entry,
            Some(serde_json::json!({ "group": row.group_name, "role": row.role })),
        )
        .await?;
        Ok(row)
    }

    /// Take a group's role away; one that had none is an error, after the
    /// attempt is audited as such.
    pub(crate) async fn remove_group_role(&self, app: &App, group: &str) -> ApiResult<()> {
        let entry = self
            .entry(AuditAction::Member, Outcome::Allowed)
            .on(ResourceKind::Group.id(group));
        let removed = app
            .control
            .remove_group_role(&self.membership.workspace.id, group, entry.clone())
            .await?;
        let detail = serde_json::json!({
            "group": group,
            "reason": (!removed).then_some("no role"),
        });
        self.record_detail(app, &entry, Some(detail)).await?;
        if removed {
            Ok(())
        } else {
            Err(ApiError::not_found("that group has no role here"))
        }
    }
}
