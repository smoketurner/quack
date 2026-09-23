//! Server administration: users and the skeletal access audit.

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::{Json, response::IntoResponse};
use quack_core::storage::control::{AuditFilter, Outcome};
use serde::Deserialize;

use crate::server::auth::{Identity, require_admin};
use crate::server::error::ApiResult;
use crate::server::state::App;

pub(crate) async fn users(
    State(app): State<App>,
    identity: Identity,
) -> ApiResult<Json<serde_json::Value>> {
    require_admin(&identity)?;
    let users = app.control.list_users().await?;
    Ok(Json(serde_json::json!({ "users": users })))
}

#[derive(Deserialize)]
pub(crate) struct CreateUser {
    pub username: String,
    pub password: String,
    #[serde(default)]
    pub is_admin: bool,
}

pub(crate) async fn create_user(
    State(app): State<App>,
    identity: Identity,
    Json(body): Json<CreateUser>,
) -> ApiResult<impl IntoResponse> {
    require_admin(&identity)?;
    let user = app
        .control
        .create_user(&body.username, &body.password, body.is_admin)
        .await?;
    let mut entry = identity.audit("admin", Outcome::Allowed);
    entry.resource_type = Some(String::from("user"));
    entry.resource_id = Some(user.id.clone());
    app.control.record_audit(&entry).await?;
    Ok((StatusCode::CREATED, Json(serde_json::to_value(user)?)))
}

#[derive(Deserialize)]
pub(crate) struct AuditQuery {
    pub user_id: Option<String>,
    pub workspace_id: Option<String>,
    pub action: Option<String>,
    pub outcome: Option<Outcome>,
    pub since: Option<String>,
    pub until: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    100
}

pub(crate) async fn audit(
    State(app): State<App>,
    identity: Identity,
    Query(q): Query<AuditQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    require_admin(&identity)?;
    let rows = app
        .control
        .query_audit(&AuditFilter {
            user_id: q.user_id,
            workspace_id: q.workspace_id,
            action: q.action,
            outcome: q.outcome,
            since: q.since,
            until: q.until,
            limit: q.limit.min(1000),
        })
        .await?;
    Ok(Json(serde_json::json!({ "audit": rows })))
}
