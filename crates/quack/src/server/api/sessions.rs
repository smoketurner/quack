//! Sessions: the caller's own (all of them for owners), their messages,
//! and export as SQL or Markdown, which is audited as a boundary crossing.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use quack_core::analysis::events::Decision;
use quack_core::ids::{PermissionId, SessionId, WorkspaceId};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use quack_core::storage::sessions::{
    self, ChatMode, ExportFormat, MessageRow, SessionRow, Sharing, Transcript,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::{App, with_db};

#[derive(Deserialize, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub(crate) struct ListQuery {
    /// Sessions at most, newest first.
    #[serde(default = "default_limit")]
    #[param(default = 50)]
    pub limit: u32,
}

fn default_limit() -> u32 {
    50
}

/// The sessions the caller may read.
#[derive(Serialize, ToSchema)]
pub(crate) struct SessionList {
    pub sessions: Vec<SessionRow>,
}

/// The caller's sessions and the shared ones; an owner's sees every one.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/sessions",
    tag = "sessions",
    params(ListQuery),
    responses((status = 200, description = "The sessions", body = SessionList)),
)]
pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Query(q): Query<ListQuery>,
) -> ApiResult<Json<SessionList>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "sessions")
        .await?;
    let viewer = access.session_viewer();
    let limit = q.limit;
    let rows = app
        .read(&id, move |db| {
            sessions::list_sessions_for(db, limit, &viewer)
        })
        .await?;
    Ok(Json(SessionList { sessions: rows }))
}

impl Access {
    /// The session, if the caller may see it; one they may not reads as
    /// missing.
    pub(crate) async fn visible_session(
        &self,
        app: &App,
        session_id: &SessionId,
    ) -> ApiResult<SessionRow> {
        let sid = session_id.to_owned();
        let viewer = self.session_viewer();
        let found = app
            .read(&self.membership.workspace.id, move |db| {
                Ok(sessions::get_session(db, &sid)?.filter(|s| s.visible_to(&viewer)))
            })
            .await?;
        found.ok_or_else(|| ResourceKind::Session.missing(session_id.as_str()).into())
    }
}

/// A session and every message in it.
#[derive(Serialize, ToSchema)]
pub(crate) struct SessionDetail {
    pub session: SessionRow,
    pub messages: Vec<MessageRow>,
}

/// A session and every message in it.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/sessions/{sid}",
    tag = "sessions",
    responses((status = 200, description = "The session", body = SessionDetail)),
)]
pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path((id, sid)): Path<(WorkspaceId, SessionId)>,
) -> ApiResult<Json<SessionDetail>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let session = access.visible_session(&app, &sid).await?;
    let session_id = session.id.clone();
    let messages = app
        .read(&id, move |db| sessions::messages(db, &session_id))
        .await?;
    access
        .audit(
            &app,
            AuditAction::SessionRead,
            Some(ResourceKind::Session.id(&sid)),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(Json(SessionDetail { session, messages }))
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct UpdateSession {
    pub shared: Option<Sharing>,
    /// `chat` or `query`: the explicit way to change a session's mode.
    pub mode: Option<ChatMode>,
}

/// Share a session with every member or take it back, or change its
/// mode. Its creator, or an owner, may. Audited as `share` and `mode`.
#[utoipa::path(
    patch,
    path = "/workspaces/{id}/sessions/{sid}",
    tag = "sessions",
    request_body = UpdateSession,
    responses((status = 200, description = "The session as it now is", body = SessionRow)),
)]
pub(crate) async fn update(
    State(app): State<App>,
    identity: Identity,
    Path((id, sid)): Path<(WorkspaceId, SessionId)>,
    Json(body): Json<UpdateSession>,
) -> ApiResult<Json<SessionRow>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    if body.shared.is_none() && body.mode.is_none() {
        return Err(ApiError::bad_request("give shared or mode"));
    }
    let shared = match body.shared {
        Some(sharing) => Some(access.set_session_sharing(&app, &sid, sharing).await?),
        None => None,
    };
    let session = match body.mode {
        Some(mode) => Some(access.set_session_mode(&app, &sid, mode).await?),
        None => shared,
    };
    let session =
        session.ok_or_else(|| ApiError::from(ResourceKind::Session.missing(sid.as_str())))?;
    Ok(Json(session))
}

/// Delete a session: its creator, or an owner, may. Audited as `delete`.
#[utoipa::path(
    delete,
    path = "/workspaces/{id}/sessions/{sid}",
    tag = "sessions",
    responses((status = 204, description = "Deleted")),
)]
pub(crate) async fn remove(
    State(app): State<App>,
    identity: Identity,
    Path((id, sid)): Path<(WorkspaceId, SessionId)>,
) -> ApiResult<axum::http::StatusCode> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access.delete_session(&app, &sid).await?;
    Ok(axum::http::StatusCode::NO_CONTENT)
}

/// What a session's creator, or an owner or admin, may do to it; the API
/// and the web console share these.
impl Access {
    /// The session, when the caller may change it; otherwise the refusal
    /// is audited as `action` and answered 403 with `refusal`.
    async fn own_session(
        &self,
        app: &App,
        sid: &SessionId,
        action: AuditAction,
        refusal: &'static str,
    ) -> ApiResult<SessionRow> {
        let session = self.visible_session(app, sid).await?;
        if !self.owns(session.created_by.as_ref()) {
            self.audit(
                app,
                action,
                Some(ResourceKind::Session.id(sid)),
                Outcome::Denied,
                None,
            )
            .await?;
            return Err(ApiError::forbidden(refusal));
        }
        Ok(session)
    }

    /// Change a session's mode.
    pub(crate) async fn set_session_mode(
        &self,
        app: &App,
        sid: &SessionId,
        mode: ChatMode,
    ) -> ApiResult<SessionRow> {
        let session = self
            .own_session(
                app,
                sid,
                AuditAction::Mode,
                "only the session's creator or an owner may change its mode",
            )
            .await?;
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let session_id = session.id;
        let updated = with_db(db, move |db| {
            sessions::set_session_mode(db, &session_id, mode)?;
            sessions::get_session(db, &session_id)?
                .ok_or_else(|| ResourceKind::Session.missing(session_id.as_str()))
        })
        .await?;
        self.audit(
            app,
            AuditAction::Mode,
            Some(ResourceKind::Session.id(sid)),
            Outcome::Allowed,
            Some(serde_json::json!({ "mode": mode.as_str() })),
        )
        .await?;
        Ok(updated)
    }

    /// Share a session with every member, or take it back.
    pub(crate) async fn set_session_sharing(
        &self,
        app: &App,
        sid: &SessionId,
        sharing: Sharing,
    ) -> ApiResult<SessionRow> {
        let session = self
            .own_session(
                app,
                sid,
                AuditAction::Share,
                "only the session's creator or an owner may share it",
            )
            .await?;
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let session_id = session.id;
        let updated = with_db(db, move |db| {
            sessions::set_session_sharing(db, &session_id, sharing)?;
            sessions::get_session(db, &session_id)?
                .ok_or_else(|| ResourceKind::Session.missing(session_id.as_str()))
        })
        .await?;
        self.audit(
            app,
            AuditAction::Share,
            Some(ResourceKind::Session.id(sid)),
            Outcome::Allowed,
            Some(serde_json::json!({ "shared": bool::from(sharing) })),
        )
        .await?;
        Ok(updated)
    }

    /// Delete a session, audited as `delete`.
    pub(crate) async fn delete_session(&self, app: &App, sid: &SessionId) -> ApiResult<()> {
        let session = self
            .own_session(
                app,
                sid,
                AuditAction::Delete,
                "only the session's creator or an owner may delete it",
            )
            .await?;
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let session_id = session.id;
        with_db(db, move |db| sessions::delete_session(db, &session_id)).await?;
        self.audit(
            app,
            AuditAction::Delete,
            Some(ResourceKind::Session.id(sid)),
            Outcome::Allowed,
            None,
        )
        .await?;
        Ok(())
    }
}
#[derive(Deserialize, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub(crate) struct ExportQuery {
    #[serde(default)]
    pub format: ExportFormat,
}

/// The session as Markdown (questions, steps, answers) or as a runnable
/// `.sql` file of its statements.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/sessions/{sid}/export",
    tag = "sessions",
    params(ExportQuery),
    responses((status = 200, description = "The transcript", content(
        (String = "text/markdown"),
        (String = "application/sql"),
    ))),
)]
pub(crate) async fn export(
    State(app): State<App>,
    identity: Identity,
    Path((id, sid)): Path<(WorkspaceId, SessionId)>,
    Query(q): Query<ExportQuery>,
) -> ApiResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let session = access.visible_session(&app, &sid).await?;
    let format = q.format;
    let text = app
        .read(&id, move |db| Transcript::load(db, session)?.render(format))
        .await?;
    access
        .audit(
            &app,
            AuditAction::Export,
            Some(ResourceKind::Session.id(&sid)),
            Outcome::Allowed,
            Some(serde_json::json!({ "format": q.format })),
        )
        .await?;
    let content_type = match format {
        ExportFormat::Sql => "application/sql; charset=utf-8",
        ExportFormat::Markdown => "text/markdown; charset=utf-8",
    };
    Ok(([(header::CONTENT_TYPE, content_type)], text).into_response())
}

/// A person's answer to a write their streamed turn is waiting on.
#[derive(Deserialize, ToSchema)]
pub(crate) struct PermissionAnswer {
    pub decision: Decision,
}

/// Answer the write `request` that the caller's turn in session `sid` is
/// waiting on. Only the person whose question asked may answer, and only
/// with write access; every answer is audited with the statement.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/sessions/{sid}/permissions/{request}",
    tag = "sessions",
    request_body = PermissionAnswer,
    responses((status = 204, description = "The turn has the answer")),
)]
pub(crate) async fn decide(
    State(app): State<App>,
    identity: Identity,
    Path((id, sid, request)): Path<(WorkspaceId, SessionId, PermissionId)>,
    Json(body): Json<PermissionAnswer>,
) -> ApiResult<axum::http::StatusCode> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let answer = body.decision;
    let resource = Some(ResourceKind::Session.id(&sid));
    match app.permissions.decide(&request, &access, &sid, answer) {
        Ok(sql) => {
            access
                .audit(
                    &app,
                    AuditAction::Permission,
                    resource,
                    answer.outcome(),
                    Some(serde_json::json!({
                        "request": request, "sql": sql, "decision": answer.as_str(),
                    })),
                )
                .await?;
            Ok(axum::http::StatusCode::NO_CONTENT)
        }
        Err(refusal) => {
            access
                .audit(
                    &app,
                    AuditAction::Permission,
                    resource,
                    Outcome::Denied,
                    refusal.detail(&request, answer),
                )
                .await?;
            Err(refusal.into())
        }
    }
}
