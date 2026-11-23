//! Sessions: the caller's own (all of them for owners), their messages,
//! and export as SQL or Markdown, which is audited as a boundary crossing.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use quack_core::analysis::events::Decision;
use quack_core::ids::{PermissionId, SessionId, WorkspaceId};
use quack_core::storage::control::{AuditAction, ResourceKind};
use quack_core::storage::sessions::{
    self, ChatMode, ExportFormat, MessageHit, MessageRow, SessionRow, Sharing, Transcript,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::auth::{Access, Audit, Identity, Need};
use crate::error::{ApiError, ApiResult};
use crate::permissions;
use crate::state::{App, with_db};

#[derive(Deserialize, utoipa::IntoParams)]
pub(crate) struct ListQuery {
    /// Sessions at most, newest first.
    #[serde(default = "default_limit")]
    #[param(default = 50)]
    pub limit: u32,
}

fn default_limit() -> u32 {
    50
}

#[derive(Deserialize, utoipa::IntoParams)]
pub(crate) struct SearchQuery {
    /// The text to find in questions and answers, case-insensitive.
    pub q: String,
    /// Hits at most, newest first.
    #[serde(default = "default_limit")]
    #[param(default = 50)]
    pub limit: u32,
}

/// Questions and answers that matched, in sessions the caller may read.
#[derive(Serialize, ToSchema)]
pub(crate) struct MessageHits {
    pub hits: Vec<MessageHit>,
}

/// Find text in the questions and answers of the sessions the caller may
/// read; a session they may not read never counts. Audited as `search`,
/// the text in the detail row.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/sessions/search",
    tag = "sessions",
    params(WorkspaceId, SearchQuery),
    responses((status = 200, description = "The matches", body = MessageHits)),
)]
pub(crate) async fn search(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Query(q): Query<SearchQuery>,
) -> ApiResult<Json<MessageHits>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let hits = access.search_sessions(&app, &q.q, q.limit.min(200)).await?;
    Ok(Json(MessageHits { hits }))
}

impl Access {
    /// The caller's matches for `query`, audited as a `search`.
    pub(crate) async fn search_sessions(
        &self,
        app: &App,
        query: &str,
        limit: u32,
    ) -> ApiResult<Vec<MessageHit>> {
        let (viewer, text) = (self.session_viewer(), query.to_owned());
        let audit = Audit::of(AuditAction::Search).detail(serde_json::json!({ "sessions": query }));
        self.audited_with(app, audit, async |audit| {
            let hits = app
                .read(&self.membership.workspace.id, move |db| {
                    sessions::search_messages(db, &text, &viewer, limit)
                })
                .await?;
            audit.set_detail(serde_json::json!({ "sessions": query, "hits": hits.len() }));
            Ok(hits)
        })
        .await
    }
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
    params(WorkspaceId, ListQuery),
    responses((status = 200, description = "The sessions", body = SessionList)),
)]
pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Query(q): Query<ListQuery>,
) -> ApiResult<Json<SessionList>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let viewer = access.session_viewer();
    let limit = q.limit;
    let rows = access
        .audited(
            &app,
            Audit::read(AuditAction::List, "sessions"),
            app.read(&id, move |db| {
                sessions::list_sessions_for(db, limit, &viewer)
            }),
        )
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
    let audit = Audit::of(AuditAction::SessionRead).resource(ResourceKind::Session, sid.as_str());
    let detail = access
        .audited(&app, audit, async {
            let session = access.visible_session(&app, &sid).await?;
            let session_id = session.id.clone();
            let messages = app
                .read(&id, move |db| sessions::messages(db, &session_id))
                .await?;
            Ok(SessionDetail { session, messages })
        })
        .await?;
    Ok(Json(detail))
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct UpdateSession {
    pub shared: Option<Sharing>,
    /// `chat` or `query`: the explicit way to change a session's mode.
    pub mode: Option<ChatMode>,
    /// A new title; an empty one gives back the title the first question
    /// derives.
    pub title: Option<String>,
}

/// Share a session with every member or take it back, change its mode, or
/// rename it. Its creator, or an owner, may. Audited as `share`, `mode`,
/// and `rename`.
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
    if body.shared.is_none() && body.mode.is_none() && body.title.is_none() {
        return Err(ApiError::bad_request("give shared, mode, or title"));
    }
    let shared = match body.shared {
        Some(sharing) => Some(access.set_session_sharing(&app, &sid, sharing).await?),
        None => None,
    };
    let moded = match body.mode {
        Some(mode) => Some(access.set_session_mode(&app, &sid, mode).await?),
        None => shared,
    };
    let session = match body.title {
        Some(title) => Some(access.rename_session(&app, &sid, &title).await?),
        None => moded,
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
    /// The session, when the caller may change it; otherwise 403 with
    /// `refusal`, which the audit of the work records as a denial.
    async fn own_session(
        &self,
        app: &App,
        sid: &SessionId,
        refusal: &'static str,
    ) -> ApiResult<SessionRow> {
        let session = self.visible_session(app, sid).await?;
        if !self.owns(session.created_by.as_ref()) {
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
        let audit = Audit::of(AuditAction::Mode)
            .resource(ResourceKind::Session, sid.as_str())
            .detail(serde_json::json!({ "mode": mode.as_str() }));
        self.audited(app, audit, async {
            let session = self
                .own_session(
                    app,
                    sid,
                    "only the session's creator or an owner may change its mode",
                )
                .await?;
            let db = app.workspace(&self.membership.workspace.id).await?.writer;
            let session_id = session.id;
            with_db(db, move |db| {
                sessions::set_session_mode(db, &session_id, mode)?;
                sessions::get_session(db, &session_id)?
                    .ok_or_else(|| ResourceKind::Session.missing(session_id.as_str()))
            })
            .await
        })
        .await
    }

    /// Share a session with every member, or take it back.
    pub(crate) async fn set_session_sharing(
        &self,
        app: &App,
        sid: &SessionId,
        sharing: Sharing,
    ) -> ApiResult<SessionRow> {
        let audit = Audit::of(AuditAction::Share)
            .resource(ResourceKind::Session, sid.as_str())
            .detail(serde_json::json!({ "shared": bool::from(sharing) }));
        self.audited(app, audit, async {
            let session = self
                .own_session(
                    app,
                    sid,
                    "only the session's creator or an owner may share it",
                )
                .await?;
            let db = app.workspace(&self.membership.workspace.id).await?.writer;
            let session_id = session.id;
            with_db(db, move |db| {
                sessions::set_session_sharing(db, &session_id, sharing)?;
                sessions::get_session(db, &session_id)?
                    .ok_or_else(|| ResourceKind::Session.missing(session_id.as_str()))
            })
            .await
        })
        .await
    }

    /// Rename a session; a blank title gives back the derived one. The
    /// title is workspace content, so it goes in the detail row only.
    pub(crate) async fn rename_session(
        &self,
        app: &App,
        sid: &SessionId,
        title: &str,
    ) -> ApiResult<SessionRow> {
        let audit = Audit::of(AuditAction::Rename).resource(ResourceKind::Session, sid.as_str());
        self.audited_with(app, audit, async |audit| {
            let session = self
                .own_session(
                    app,
                    sid,
                    "only the session's creator or an owner may rename it",
                )
                .await?;
            let db = app.workspace(&self.membership.workspace.id).await?.writer;
            let (session_id, title) = (session.id, title.to_owned());
            let updated = with_db(db, move |db| {
                sessions::set_session_title(db, &session_id, &title)
            })
            .await?;
            audit.set_detail(
                serde_json::json!({ "title": updated.title, "title_by": updated.title_by }),
            );
            Ok(updated)
        })
        .await
    }

    /// Delete a session, audited as `delete`.
    pub(crate) async fn delete_session(&self, app: &App, sid: &SessionId) -> ApiResult<()> {
        let audit = Audit::of(AuditAction::Delete).resource(ResourceKind::Session, sid.as_str());
        self.audited(app, audit, async {
            let session = self
                .own_session(
                    app,
                    sid,
                    "only the session's creator or an owner may delete it",
                )
                .await?;
            let db = app.workspace(&self.membership.workspace.id).await?.writer;
            let session_id = session.id;
            with_db(db, move |db| sessions::delete_session(db, &session_id)).await?;
            // The file's rows went with the session; the live listing follows.
            app.engine.jobs.forget_session(sid);
            Ok(())
        })
        .await
    }
}

#[derive(Deserialize, utoipa::IntoParams)]
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
    let format = q.format;
    let audit = Audit::of(AuditAction::Export)
        .resource(ResourceKind::Session, sid.as_str())
        .detail(serde_json::json!({ "format": format }));
    let text = access
        .audited(&app, audit, async {
            let session = access.visible_session(&app, &sid).await?;
            app.read(&id, move |db| Transcript::load(db, session)?.render(format))
                .await
        })
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
    let audit = Audit::of(AuditAction::Permission).resource(ResourceKind::Session, sid.as_str());
    match permissions::decide(&app, &access, &sid, &request, answer) {
        Ok(sql) => {
            let detail = serde_json::json!({
                "request": request, "sql": sql, "decision": answer.as_str(),
            });
            access.answered(&app, audit.detail(detail), answer).await?;
            Ok(axum::http::StatusCode::NO_CONTENT)
        }
        Err(refusal) => {
            let audit = match refusal.detail(&request, answer) {
                Some(detail) => audit.detail(detail),
                None => audit,
            };
            access.denied(&app, audit).await?;
            Err(refusal.into())
        }
    }
}
