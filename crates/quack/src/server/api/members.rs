//! Membership: owners and admins add, change, and remove members.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind, Role};
use serde::Deserialize;

use crate::server::auth::{Identity, Need, access};
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
    let access = access(&app, identity, &id, need).await?;
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
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::OWN).await?;
    let user = app
        .control
        .find_user_by_username(&body.username)
        .await?
        .ok_or_else(|| ApiError::not_found("no such user"))?;
    app.control.set_member(&id, &user.id, body.role).await?;
    access
        .audit(
            &app,
            AuditAction::Member,
            Some(ResourceKind::User.id(&user.id)),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(Json(
        serde_json::json!({ "user_id": user.id, "username": user.username, "role": body.role }),
    ))
}

pub(crate) async fn remove(
    State(app): State<App>,
    identity: Identity,
    Path((id, user_id)): Path<(String, String)>,
) -> ApiResult<StatusCode> {
    let access = access(&app, identity, &id, Need::OWN).await?;
    let removed = app.control.remove_member(&id, &user_id).await?;
    access
        .audit(
            &app,
            AuditAction::Member,
            Some(ResourceKind::User.id(&user_id)),
            Outcome::Allowed,
            None,
        )
        .await?;
    if removed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::not_found("not a member"))
    }
}
