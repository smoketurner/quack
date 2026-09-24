//! Server administration: users and the skeletal access audit.

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use quack_core::ids::{UserId, WorkspaceId};
use quack_core::storage::control::{
    AuditAction, AuditCursor, AuditFilter, Outcome, ResourceKind, UserKind, UserRow,
};
use serde::Deserialize;

use crate::server::auth::Identity;
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::{App, ServeMode};

pub(crate) async fn users(
    State(app): State<App>,
    identity: Identity,
) -> ApiResult<Json<serde_json::Value>> {
    identity.require_admin()?;
    let users = app.control.list_users().await?;
    Ok(Json(serde_json::json!({ "users": users })))
}

#[derive(Deserialize)]
pub(crate) struct CreateUser {
    pub username: String,
    pub password: String,
    #[serde(default, rename = "is_admin")]
    pub kind: UserKind,
}

pub(crate) async fn create_user(
    State(app): State<App>,
    identity: Identity,
    Json(body): Json<CreateUser>,
) -> ApiResult<impl IntoResponse> {
    let user = identity.create_user(&app, &body).await?;
    Ok((StatusCode::CREATED, Json(serde_json::to_value(user)?)))
}

impl Identity {
    /// Add a server user, from the API or the web console: admins only,
    /// and not in local mode, which has no logins.
    pub(crate) async fn create_user(&self, app: &App, user: &CreateUser) -> ApiResult<UserRow> {
        self.require_admin()?;
        if app.mode == ServeMode::Local {
            return Err(ApiError::bad_request("local mode has no users"));
        }
        let created = app
            .control
            .create_user(&user.username, &user.password, user.kind)
            .await?;
        let mut entry = self.audit(AuditAction::Admin, Outcome::Allowed);
        entry = entry.on(ResourceKind::User.id(created.id.as_str()));
        app.control.record_audit(&entry).await?;
        Ok(created)
    }
}

#[derive(Deserialize)]
pub(crate) struct AuditQuery {
    pub user_id: Option<UserId>,
    pub workspace_id: Option<WorkspaceId>,
    pub action: Option<String>,
    pub outcome: Option<Outcome>,
    pub since: Option<String>,
    pub until: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: u32,
    pub cursor: Option<AuditCursor>,
}

fn default_limit() -> u32 {
    100
}

/// At most 1000 rows per page.
impl From<AuditQuery> for AuditFilter {
    fn from(q: AuditQuery) -> Self {
        Self {
            user_id: q.user_id,
            workspace_id: q.workspace_id,
            action: q.action,
            outcome: q.outcome,
            since: q.since,
            until: q.until,
            limit: q.limit.min(1000),
            after: q.cursor,
        }
    }
}

pub(crate) async fn audit(
    State(app): State<App>,
    identity: Identity,
    Query(q): Query<AuditQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    identity.require_admin()?;
    let page = app.control.query_audit(&AuditFilter::from(q)).await?;
    Ok(Json(
        serde_json::json!({ "audit": page.rows, "next_cursor": page.next }),
    ))
}
