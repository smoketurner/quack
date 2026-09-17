//! Sessions: the caller's own (all of them for owners), their messages,
//! and export as SQL or Markdown, which is audited as a boundary crossing.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use quack_core::storage::control::Outcome;
use quack_core::storage::sessions;
use serde::Deserialize;

use crate::server::auth::{Access, Identity, Need, access};
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
    Path(id): Path<String>,
    Query(q): Query<ListQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let user = access.identity.user_id.clone();
    let sees_all = access.sees_all_sessions();
    let limit = q.limit;
    let rows = with_db(db, move |db| {
        sessions::list_sessions_for(db, limit, &user, sees_all)
    })
    .await?;
    Ok(Json(serde_json::json!({ "sessions": rows })))
}

async fn visible_session(
    app: &App,
    access: &Access,
    workspace_id: &str,
    session_id: &str,
) -> ApiResult<sessions::SessionRow> {
    let db = app.workspace_db(workspace_id).await?;
    let sid = session_id.to_owned();
    let user = access.identity.user_id.clone();
    let sees_all = access.sees_all_sessions();
    let found = with_db(db, move |db| {
        Ok(sessions::get_session(db, &sid)?.filter(|s| sessions::visible_to(s, &user, sees_all)))
    })
    .await?;
    found.ok_or_else(|| ApiError::not_found("no such session"))
}

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path((id, sid)): Path<(String, String)>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let session = visible_session(&app, &access, &id, &sid).await?;
    let db = app.workspace_db(&id).await?;
    let session_id = session.id.clone();
    let messages = with_db(db, move |db| sessions::messages(db, &session_id)).await?;
    access
        .audit(
            &app,
            "session_read",
            Some(("session", &sid)),
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
    pub shared: bool,
}

/// Share a session with every member, or take it back. Its creator, or an
/// owner, may. Audited as `share`.
pub(crate) async fn update(
    State(app): State<App>,
    identity: Identity,
    Path((id, sid)): Path<(String, String)>,
    Json(body): Json<UpdateSession>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let session = set_shared(&app, &access, &sid, body.shared).await?;
    Ok(Json(serde_json::to_value(session)?))
}

/// The shared toggle behind the API and the web button.
pub(crate) async fn set_shared(
    app: &App,
    access: &Access,
    sid: &str,
    shared: bool,
) -> ApiResult<sessions::SessionRow> {
    let session = visible_session(app, access, &access.workspace.id, sid).await?;
    let mine = session.created_by.as_deref() == Some(access.identity.user_id.as_str());
    if !mine && !access.sees_all_sessions() {
        access
            .audit(app, "share", Some(("session", sid)), Outcome::Denied, None)
            .await?;
        return Err(ApiError::forbidden(
            "only the session's creator or an owner may share it",
        ));
    }
    let db = app.workspace_db(&access.workspace.id).await?;
    let session_id = session.id.clone();
    let updated = with_db(db, move |db| {
        sessions::set_session_shared(db, &session_id, shared)?;
        sessions::get_session(db, &session_id)
    })
    .await?
    .ok_or_else(|| ApiError::not_found("no such session"))?;
    access
        .audit(
            app,
            "share",
            Some(("session", sid)),
            Outcome::Allowed,
            Some(serde_json::json!({ "shared": shared })),
        )
        .await?;
    Ok(updated)
}

/// Delete a session: its creator, or an owner, may. Audited as `delete`.
pub(crate) async fn remove(
    State(app): State<App>,
    identity: Identity,
    Path((id, sid)): Path<(String, String)>,
) -> ApiResult<axum::http::StatusCode> {
    let access = access(&app, identity, &id, Need::READ).await?;
    delete_session(&app, &access, &sid).await?;
    Ok(axum::http::StatusCode::NO_CONTENT)
}

/// The shared delete behind the API and the web button.
pub(crate) async fn delete_session(app: &App, access: &Access, sid: &str) -> ApiResult<()> {
    let session = visible_session(app, access, &access.workspace.id, sid).await?;
    let mine = session.created_by.as_deref() == Some(access.identity.user_id.as_str());
    if !mine && !access.sees_all_sessions() {
        access
            .audit(app, "delete", Some(("session", sid)), Outcome::Denied, None)
            .await?;
        return Err(ApiError::forbidden(
            "only the session's creator or an owner may delete it",
        ));
    }
    let db = app.workspace_db(&access.workspace.id).await?;
    let session_id = session.id.clone();
    with_db(db, move |db| sessions::delete_session(db, &session_id)).await?;
    access
        .audit(
            app,
            "delete",
            Some(("session", sid)),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(())
}

#[derive(Deserialize)]
pub(crate) struct ExportQuery {
    #[serde(default = "default_format")]
    pub format: String,
}

fn default_format() -> String {
    String::from("markdown")
}

pub(crate) async fn export(
    State(app): State<App>,
    identity: Identity,
    Path((id, sid)): Path<(String, String)>,
    Query(q): Query<ExportQuery>,
) -> ApiResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let session = visible_session(&app, &access, &id, &sid).await?;
    let as_sql = match q.format.as_str() {
        "sql" => true,
        "markdown" => false,
        _ => return Err(ApiError::bad_request("format must be sql or markdown")),
    };
    let db = app.workspace_db(&id).await?;
    let text = with_db(db, move |db| {
        let rows = sessions::messages(db, &session.id)?;
        if as_sql {
            sessions::export_sql(&rows)
        } else {
            sessions::export_markdown(&session, &rows)
        }
    })
    .await?;
    access
        .audit(
            &app,
            "export",
            Some(("session", &sid)),
            Outcome::Allowed,
            Some(serde_json::json!({ "format": q.format })),
        )
        .await?;
    let content_type = if as_sql {
        "application/sql; charset=utf-8"
    } else {
        "text/markdown; charset=utf-8"
    };
    Ok(([(header::CONTENT_TYPE, content_type)], text).into_response())
}
