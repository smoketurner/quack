//! Sessions: the caller's own (all of them for owners), their messages,
//! and export as SQL or Markdown, which is audited as a boundary crossing.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use quack_core::error::Record;
use quack_core::ids::WorkspaceId;
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use quack_core::storage::sessions::{self, ChatMode, ExportFormat, Transcript};
use serde::Deserialize;

use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::{App, with_db};

#[derive(Deserialize)]
pub(crate) struct ListQuery {
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    50
}

pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Query(q): Query<ListQuery>,
) -> ApiResult<Json<serde_json::Value>> {
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
    Ok(Json(serde_json::json!({ "sessions": rows })))
}

impl Access {
    /// The session, if the caller may see it; one they may not reads as
    /// missing.
    async fn visible_session(
        &self,
        app: &App,
        session_id: &str,
    ) -> ApiResult<sessions::SessionRow> {
        let sid = session_id.to_owned();
        let viewer = self.session_viewer();
        let found = app
            .read(&self.workspace.id, move |db| {
                Ok(sessions::get_session(db, &sid)?.filter(|s| s.visible_to(&viewer)))
            })
            .await?;
        found.ok_or_else(|| Record::Session.missing(session_id).into())
    }
}

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path((id, sid)): Path<(WorkspaceId, String)>,
) -> ApiResult<Json<serde_json::Value>> {
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
    Ok(Json(
        serde_json::json!({ "session": session, "messages": messages }),
    ))
}

#[derive(Deserialize)]
pub(crate) struct UpdateSession {
    pub shared: Option<bool>,
    /// `chat` or `query`: the explicit way to change a session's mode.
    pub mode: Option<ChatMode>,
}

/// Share a session with every member or take it back, or change its
/// mode. Its creator, or an owner, may. Audited as `share` and `mode`.
pub(crate) async fn update(
    State(app): State<App>,
    identity: Identity,
    Path((id, sid)): Path<(WorkspaceId, String)>,
    Json(body): Json<UpdateSession>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    if body.shared.is_none() && body.mode.is_none() {
        return Err(ApiError::bad_request("give shared or mode"));
    }
    let mut session = None;
    if let Some(shared) = body.shared {
        session = Some(access.set_session_shared(&app, &sid, shared).await?);
    }
    if let Some(mode) = body.mode {
        session = Some(access.set_session_mode(&app, &sid, mode).await?);
    }
    let session = session.ok_or_else(|| ApiError::from(Record::Session.missing(sid.as_str())))?;
    Ok(Json(serde_json::to_value(session)?))
}

/// Delete a session: its creator, or an owner, may. Audited as `delete`.
pub(crate) async fn remove(
    State(app): State<App>,
    identity: Identity,
    Path((id, sid)): Path<(WorkspaceId, String)>,
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
        sid: &str,
        action: AuditAction,
        refusal: &'static str,
    ) -> ApiResult<sessions::SessionRow> {
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
        sid: &str,
        mode: ChatMode,
    ) -> ApiResult<sessions::SessionRow> {
        let session = self
            .own_session(
                app,
                sid,
                AuditAction::Mode,
                "only the session's creator or an owner may change its mode",
            )
            .await?;
        let db = app.workspace_db(&self.workspace.id).await?;
        let session_id = session.id;
        let updated = with_db(db, move |db| {
            sessions::set_session_mode(db, &session_id, mode)?;
            sessions::get_session(db, &session_id)?
                .ok_or_else(|| Record::Session.missing(session_id.as_str()))
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
    pub(crate) async fn set_session_shared(
        &self,
        app: &App,
        sid: &str,
        shared: bool,
    ) -> ApiResult<sessions::SessionRow> {
        let session = self
            .own_session(
                app,
                sid,
                AuditAction::Share,
                "only the session's creator or an owner may share it",
            )
            .await?;
        let db = app.workspace_db(&self.workspace.id).await?;
        let session_id = session.id;
        let updated = with_db(db, move |db| {
            sessions::set_session_shared(db, &session_id, shared)?;
            sessions::get_session(db, &session_id)?
                .ok_or_else(|| Record::Session.missing(session_id.as_str()))
        })
        .await?;
        self.audit(
            app,
            AuditAction::Share,
            Some(ResourceKind::Session.id(sid)),
            Outcome::Allowed,
            Some(serde_json::json!({ "shared": shared })),
        )
        .await?;
        Ok(updated)
    }

    /// Delete a session, audited as `delete`.
    pub(crate) async fn delete_session(&self, app: &App, sid: &str) -> ApiResult<()> {
        let session = self
            .own_session(
                app,
                sid,
                AuditAction::Delete,
                "only the session's creator or an owner may delete it",
            )
            .await?;
        let db = app.workspace_db(&self.workspace.id).await?;
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
#[derive(Deserialize)]
pub(crate) struct ExportQuery {
    #[serde(default)]
    pub format: ExportFormat,
}

pub(crate) async fn export(
    State(app): State<App>,
    identity: Identity,
    Path((id, sid)): Path<(WorkspaceId, String)>,
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
