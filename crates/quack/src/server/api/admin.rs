//! Server administration: users and the skeletal access audit.

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use quack_core::error::Result as CoreResult;
use quack_core::storage::control::{
    AuditAction, AuditFilter, AuditRow, Outcome, UserKind, UserRow,
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
        let entry = self.audit(AuditAction::Admin, Outcome::Allowed);
        Ok(app
            .control
            .create_user(&user.username, &user.password, user.kind, entry)
            .await?)
    }
}

/// How each row of an audit page is shaped.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum AuditShape {
    /// The stored row's own fields.
    #[default]
    Quack,
    /// An OCSF 1.9.0 event.
    Ocsf,
}

/// Which shape `GET /audit` answers with.
#[derive(Deserialize)]
pub(crate) struct AuditShapeQuery {
    #[serde(default)]
    pub format: AuditShape,
}

pub(crate) async fn audit(
    State(app): State<App>,
    identity: Identity,
    Query(filter): Query<AuditFilter>,
    Query(shape): Query<AuditShapeQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    identity.require_admin()?;
    let page = app.control.query_audit(&filter).await?;
    let audit = match shape.format {
        AuditShape::Quack => serde_json::to_value(&page.rows)?,
        AuditShape::Ocsf => serde_json::Value::Array(
            page.rows
                .iter()
                .map(AuditRow::to_ocsf)
                .collect::<CoreResult<_>>()?,
        ),
    };
    Ok(Json(
        serde_json::json!({ "audit": audit, "next_cursor": page.next }),
    ))
}
