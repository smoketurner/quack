//! Server administration: users, their lifecycle, and the access audit.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use quack_core::error::Result as CoreResult;
use quack_core::ids::UserId;
use quack_core::storage::control::{
    AuditAction, AuditFilter, AuditRow, Outcome, ResourceKind, UserKind, UserRow,
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

/// `PATCH /admin/users/{user}`: each field given is applied; absent ones
/// keep their value.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct UpdateUser {
    pub disabled: Option<bool>,
    #[serde(rename = "is_admin")]
    pub kind: Option<UserKind>,
    pub password: Option<String>,
}

pub(crate) async fn update_user(
    State(app): State<App>,
    identity: Identity,
    Path(user): Path<UserId>,
    Json(body): Json<UpdateUser>,
) -> ApiResult<Json<UserRow>> {
    Ok(Json(identity.update_user(&app, &user, body).await?))
}

pub(crate) async fn delete_user(
    State(app): State<App>,
    identity: Identity,
    Path(user): Path<UserId>,
) -> ApiResult<Json<serde_json::Value>> {
    let forgotten = identity.delete_user(&app, &user).await?;
    Ok(Json(serde_json::json!({ "rows_forgotten": forgotten })))
}

impl Identity {
    /// Add a server user, from the API or the web console: admins only,
    /// and not in local mode, which has no logins.
    pub(crate) async fn create_user(&self, app: &App, user: &CreateUser) -> ApiResult<UserRow> {
        self.admin_of_users(app)?;
        let entry = self.audit(AuditAction::Admin, Outcome::Allowed);
        Ok(app
            .control
            .create_user(&user.username, &user.password, user.kind, entry)
            .await?)
    }

    /// Disable, enable, promote, demote, or reset the password of a user,
    /// from the API or the web console. Disabling and a new password end
    /// the user's sessions. An admin cannot disable or demote themselves.
    pub(crate) async fn update_user(
        &self,
        app: &App,
        user: &UserId,
        change: UpdateUser,
    ) -> ApiResult<UserRow> {
        self.admin_of_users(app)?;
        let myself = *user == self.user_id;
        if myself && (change.disabled == Some(true) || change.kind == Some(UserKind::Standard)) {
            return Err(ApiError::bad_request(
                "an admin cannot disable or demote their own account",
            ));
        }
        let entry = || self.audit(AuditAction::Admin, Outcome::Allowed);
        match change.disabled {
            Some(true) => {
                app.control.disable_user(user, entry()).await?;
                app.sessions.close_user(user);
            }
            Some(false) => app.control.enable_user(user, entry()).await?,
            None => {}
        }
        if let Some(kind) = change.kind {
            app.control.set_admin(user, kind, entry()).await?;
        }
        if let Some(password) = change.password {
            app.control.set_password(user, &password, entry()).await?;
            app.sessions.close_user(user);
        }
        app.control
            .get_user(user)
            .await?
            .ok_or_else(|| ApiError::not_found("no such user"))
    }

    /// Delete a user: their sessions end, their row goes with its
    /// memberships, tokens, and stored sign-in, and every workspace file
    /// replaces their id and name with a fixed marker. The access audit
    /// keeps every row that names them. Returns how many workspace rows
    /// were rewritten.
    pub(crate) async fn delete_user(&self, app: &App, user: &UserId) -> ApiResult<usize> {
        self.admin_of_users(app)?;
        if *user == self.user_id {
            return Err(ApiError::bad_request(
                "an admin cannot delete their own account",
            ));
        }
        let row = app
            .control
            .get_user(user)
            .await?
            .ok_or_else(|| ApiError::not_found("no such user"))?;
        app.sessions.close_user(user);
        let entry = self
            .audit(AuditAction::Admin, Outcome::Allowed)
            .on(ResourceKind::User.id(user.as_str()));
        app.control.delete_user(user, entry).await?;
        let mut forgotten = 0_usize;
        for workspace in app.control.list_workspaces().await? {
            let (id, name) = (user.clone(), row.username.clone());
            let changed = app
                .workspace_db(&workspace.id)
                .await?
                .run(move |db| db.forget_user(&id, &name))
                .await?;
            forgotten = forgotten.saturating_add(changed);
        }
        Ok(forgotten)
    }

    /// Admins only, and not in local mode, which has no logins.
    fn admin_of_users(&self, app: &App) -> ApiResult<()> {
        self.require_admin()?;
        if app.mode == ServeMode::Local {
            return Err(ApiError::bad_request("local mode has no users"));
        }
        Ok(())
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
