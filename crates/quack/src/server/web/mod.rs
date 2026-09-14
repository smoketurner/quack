//! The web UI: server-rendered askama pages over the same access checks as
//! the API, htmx for the document list and SQL grid, and a small script for
//! the streamed chat (design doc 11.1). Everything a page does, the API can
//! do; the handlers here only shape the response as HTML.

use askama::Template;
use axum::Form;
use axum::Router;
use axum::extract::{FromRequestParts, Multipart, Path, Query, State};
use axum::http::request::Parts;
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum_extra::extract::CookieJar;
// Multi-valued fields (checkboxes) need serde_html_form, which axum's own
// Form extractor does not use.
use axum_extra::extract::Form as MultiForm;
use axum_extra::extract::cookie::{Cookie, SameSite};
use quack_core::storage::context;
use quack_core::storage::control::{
    AuditEntry, AuditFilter, AuditRow, Channel, MemberRow, Outcome, ProviderAllowList, Role, Scope,
    TokenRow, UserRow, WorkspaceChanges,
};
use quack_core::storage::sessions::{self, MessageRole, SessionRow};
use quack_core::storage::workspace::DocumentInfo;
use rust_embed::Embed;
use serde::Deserialize;

use super::api::documents as docs_api;
use super::api::query as query_api;
use super::auth::{Access, Credential, Identity, Need, SESSION_COOKIE, access, require_admin};
use super::error::ApiError;
use super::state::{App, with_db};

#[derive(Embed)]
#[folder = "static/"]
struct Assets;

/// The identity for a page: an unauthenticated browser goes to the login
/// form instead of getting a JSON 401.
pub(crate) struct WebUser(pub Identity);

impl FromRequestParts<App> for WebUser {
    type Rejection = Redirect;

    async fn from_request_parts(parts: &mut Parts, app: &App) -> Result<Self, Self::Rejection> {
        Identity::from_request_parts(parts, app)
            .await
            .map(WebUser)
            .map_err(|_| Redirect::to("/login"))
    }
}

/// An error as a page.
pub(crate) struct HtmlError(ApiError);

impl From<ApiError> for HtmlError {
    fn from(err: ApiError) -> Self {
        Self(err)
    }
}

impl From<quack_core::error::Error> for HtmlError {
    fn from(err: quack_core::error::Error) -> Self {
        Self(ApiError::from(err))
    }
}

impl From<askama::Error> for HtmlError {
    fn from(err: askama::Error) -> Self {
        Self(ApiError::internal(format!("template error: {err}")))
    }
}

impl IntoResponse for HtmlError {
    fn into_response(self) -> Response {
        let page = ErrorPage {
            status: self.0.status.as_u16(),
            message: self.0.message.clone(),
        };
        match page.render() {
            Ok(html) => (self.0.status, Html(html)).into_response(),
            Err(_) => self.0.into_response(),
        }
    }
}

type WebResult<T> = Result<T, HtmlError>;

/// What the layout needs on every page.
struct Page {
    title: String,
    username: String,
    is_admin: bool,
    local: bool,
    workspace: Option<WsNav>,
}

struct WsNav {
    id: String,
    name: String,
    role: String,
    can_write: bool,
    can_manage: bool,
}

fn page(app: &App, identity: &Identity, title: &str, access: Option<&Access>) -> Page {
    Page {
        title: title.to_owned(),
        username: identity.username.clone(),
        is_admin: identity.is_admin,
        local: app.local,
        workspace: access.map(|a| WsNav {
            id: a.workspace.id.clone(),
            name: a.workspace.name.clone(),
            role: a
                .role
                .map_or_else(|| String::from("admin"), |r| r.to_string()),
            can_write: a.permits(Need::WRITE),
            can_manage: a.permits(Need::OWN),
        }),
    }
}

fn html<T: Template>(template: &T) -> WebResult<Response> {
    Ok(Html(template.render()?).into_response())
}

// --- templates ---------------------------------------------------------------

#[derive(Template)]
#[template(path = "error.html")]
struct ErrorPage {
    status: u16,
    message: String,
}

#[derive(Template)]
#[template(path = "login.html")]
struct LoginPage {
    error: Option<String>,
}

struct WsItem {
    id: String,
    name: String,
    classification: String,
    role: String,
}

#[derive(Template)]
#[template(path = "workspaces.html")]
struct WorkspacesPage {
    page: Page,
    workspaces: Vec<WsItem>,
    can_create: bool,
    error: Option<String>,
}

struct MessageView {
    role: String,
    content: String,
    steps: Vec<StepView>,
    citations: Vec<CitationView>,
    chart_json: Option<String>,
}

struct StepView {
    tool: String,
    detail: String,
    summary: String,
    duration_ms: u64,
}

struct CitationView {
    n: u64,
    label: String,
    document_id: String,
}

#[derive(Template)]
#[template(path = "chat.html")]
struct ChatPage {
    page: Page,
    sessions: Vec<SessionRow>,
    current: Option<SessionRow>,
    messages: Vec<MessageView>,
}

#[derive(Template)]
#[template(path = "documents.html")]
struct DocumentsPage {
    page: Page,
    rows: String,
    error: Option<String>,
}

#[derive(Template)]
#[template(path = "documents_rows.html")]
struct DocumentRows {
    ws_id: String,
    can_write: bool,
    documents: Vec<DocumentInfo>,
    pending: bool,
}

struct TableView {
    name: String,
    columns: Vec<(String, String)>,
    sample_columns: Vec<String>,
    sample_rows: Vec<Vec<String>>,
}

#[derive(Template)]
#[template(path = "tables.html")]
struct TablesPage {
    page: Page,
    tables: Vec<String>,
    selected: Option<TableView>,
}

#[derive(Template)]
#[template(path = "sql.html")]
struct SqlPage {
    page: Page,
    sql: String,
    result: String,
}

#[derive(Template)]
#[template(path = "sql_result.html")]
struct SqlResult {
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
    row_count: usize,
    truncated: bool,
    error: Option<String>,
    csv_href: String,
}

#[derive(Template)]
#[template(path = "context.html")]
struct ContextPage {
    page: Page,
    content: String,
    current: Option<context::ContextVersion>,
    versions: Vec<context::ContextVersion>,
}

#[derive(Template)]
#[template(path = "settings.html")]
struct SettingsPage {
    page: Page,
    classification: String,
    providers: Vec<(String, bool)>,
    members: Vec<MemberRow>,
    tokens: Vec<TokenRow>,
    new_token: Option<String>,
    error: Option<String>,
}

#[derive(Template)]
#[template(path = "admin_users.html")]
struct AdminUsersPage {
    page: Page,
    users: Vec<UserRow>,
    error: Option<String>,
}

#[derive(Template)]
#[template(path = "admin_audit.html")]
struct AdminAuditPage {
    page: Page,
    rows: Vec<AuditRow>,
    action: String,
    outcome: String,
    workspace_id: String,
}

// --- routes ------------------------------------------------------------------

pub(crate) fn router() -> Router<App> {
    Router::new()
        .route("/", get(index))
        .route("/static/{*path}", get(static_asset))
        .route("/login", get(login_page).post(login_submit))
        .route("/logout", post(logout))
        .route("/workspaces", get(workspaces).post(create_workspace))
        .route("/w/{id}", get(workspace_index))
        .route("/w/{id}/chat", get(chat))
        .route("/w/{id}/chat/{sid}/delete", post(delete_session))
        .route("/w/{id}/documents", get(documents).post(upload))
        .route("/w/{id}/documents/rows", get(document_rows))
        .route("/w/{id}/documents/{doc}/pin", post(pin))
        .route("/w/{id}/documents/{doc}/unpin", post(unpin))
        .route("/w/{id}/documents/{doc}/delete", post(delete_doc))
        .route("/w/{id}/tables", get(tables))
        .route("/w/{id}/tables/{name}", get(table))
        .route("/w/{id}/sql", get(sql_page).post(sql_run))
        .route("/w/{id}/sql.csv", get(sql_csv))
        .route("/w/{id}/context", get(context_page).post(context_save))
        .route("/w/{id}/settings", get(settings).post(settings_save))
        .route("/w/{id}/members", post(member_add))
        .route("/w/{id}/members/{user}/remove", post(member_remove))
        .route("/w/{id}/tokens", post(token_create))
        .route("/w/{id}/tokens/{hash}/revoke", post(token_revoke))
        .route("/admin/users", get(admin_users).post(admin_user_add))
        .route("/admin/audit", get(admin_audit))
}

async fn static_asset(Path(path): Path<String>) -> Response {
    let Some(file) = Assets::get(&path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let mime = mime_guess::from_path(&path).first_or_octet_stream();
    (
        [
            (header::CONTENT_TYPE, mime.as_ref().to_owned()),
            (header::CACHE_CONTROL, String::from("public, max-age=86400")),
        ],
        file.data.into_owned(),
    )
        .into_response()
}

async fn index() -> Redirect {
    Redirect::to("/workspaces")
}

#[derive(Deserialize)]
struct LoginQuery {
    error: Option<String>,
}

async fn login_page(State(app): State<App>, Query(q): Query<LoginQuery>) -> WebResult<Response> {
    if app.local {
        return Ok(Redirect::to("/workspaces").into_response());
    }
    html(&LoginPage { error: q.error })
}

#[derive(Deserialize)]
struct LoginForm {
    username: String,
    password: String,
}

async fn login_submit(
    State(app): State<App>,
    jar: CookieJar,
    Form(form): Form<LoginForm>,
) -> WebResult<Response> {
    if app.local {
        return Ok(Redirect::to("/workspaces").into_response());
    }
    let user = app
        .control
        .verify_password(&form.username, &form.password)
        .await?;
    let Some(user) = user else {
        let mut entry = AuditEntry::new("login", Outcome::Denied, Channel::Web);
        entry.user_id = app
            .control
            .find_user_by_username(&form.username)
            .await?
            .map(|u| u.id);
        app.control.record_audit(&entry).await?;
        return Ok(Redirect::to("/login?error=wrong+username+or+password").into_response());
    };
    let token = app.open_web_session(&user.id)?;
    let mut entry = AuditEntry::new("login", Outcome::Allowed, Channel::Web);
    entry.user_id = Some(user.id);
    app.control.record_audit(&entry).await?;
    let cookie = Cookie::build((SESSION_COOKIE, token))
        .path("/")
        .http_only(true)
        .same_site(SameSite::Lax)
        .build();
    Ok((jar.add(cookie), Redirect::to("/workspaces")).into_response())
}

async fn logout(
    State(app): State<App>,
    WebUser(identity): WebUser,
    jar: CookieJar,
) -> WebResult<Response> {
    if let Credential::Session(token) = &identity.credential {
        app.close_web_session(token);
    }
    app.control
        .record_audit(&identity.audit("logout", Outcome::Allowed))
        .await?;
    Ok((
        jar.remove(Cookie::build(SESSION_COOKIE).path("/").build()),
        Redirect::to("/login"),
    )
        .into_response())
}

#[derive(Deserialize)]
struct FlashQuery {
    error: Option<String>,
}

async fn workspaces(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Query(q): Query<FlashQuery>,
) -> WebResult<Response> {
    let items: Vec<WsItem> = if app.local || identity.is_admin {
        let mine = if app.local {
            Vec::new()
        } else {
            app.control.workspaces_for_user(&identity.user_id).await?
        };
        app.control
            .list_workspaces()
            .await?
            .into_iter()
            .map(|w| WsItem {
                role: if app.local {
                    String::from("owner")
                } else {
                    mine.iter()
                        .find(|(m, _)| m.id == w.id)
                        .map_or_else(|| String::from("admin"), |(_, r)| r.to_string())
                },
                id: w.id,
                name: w.name,
                classification: w.classification,
            })
            .collect()
    } else {
        app.control
            .workspaces_for_user(&identity.user_id)
            .await?
            .into_iter()
            .map(|(w, r)| WsItem {
                id: w.id,
                name: w.name,
                classification: w.classification,
                role: r.to_string(),
            })
            .collect()
    };
    html(&WorkspacesPage {
        can_create: identity.is_admin,
        page: page(&app, &identity, "Workspaces", None),
        workspaces: items,
        error: q.error,
    })
}

#[derive(Deserialize)]
struct NameForm {
    name: String,
}

async fn create_workspace(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Form(form): Form<NameForm>,
) -> WebResult<Response> {
    require_admin(&identity)?;
    let name = form.name.trim();
    if name.is_empty() || name.contains(['/', '\\', '.']) {
        return Ok(Redirect::to("/workspaces?error=bad+workspace+name").into_response());
    }
    if app.control.find_workspace_by_name(name).await?.is_some() {
        return Ok(Redirect::to("/workspaces?error=workspace+exists").into_response());
    }
    let ws = app.control.create_workspace(name).await?;
    if !app.local {
        app.control
            .set_member(&ws.id, &identity.user_id, Role::Owner)
            .await?;
    }
    let mut entry = identity.audit("workspace", Outcome::Allowed);
    entry.workspace_id = Some(ws.id.clone());
    entry.resource_type = Some(String::from("workspace"));
    entry.resource_id = Some(ws.id.clone());
    app.control.record_audit(&entry).await?;
    Ok(Redirect::to(&format!("/w/{}/chat", ws.id)).into_response())
}

async fn workspace_index(Path(id): Path<String>) -> Redirect {
    Redirect::to(&format!("/w/{id}/chat"))
}

#[derive(Deserialize)]
struct ChatQuery {
    session: Option<String>,
}

fn message_view(row: &sessions::MessageRow) -> Option<MessageView> {
    let role = match row.role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => return None,
    };
    let meta = row.metadata.clone().unwrap_or(serde_json::Value::Null);
    let steps = meta
        .get("steps")
        .and_then(|s| s.as_array())
        .map(|steps| {
            steps
                .iter()
                .map(|s| StepView {
                    tool: s
                        .get("tool")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned(),
                    detail: s
                        .get("detail")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned(),
                    summary: s
                        .get("summary")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned(),
                    duration_ms: s
                        .get("duration_ms")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0),
                })
                .collect()
        })
        .unwrap_or_default();
    let citations = meta
        .get("citations")
        .and_then(|c| c.as_array())
        .map(|cs| {
            cs.iter()
                .map(|c| {
                    let filename = c.get("filename").and_then(|v| v.as_str()).unwrap_or("");
                    let page_no = c.get("page").and_then(serde_json::Value::as_u64);
                    let heading = c.get("heading").and_then(|v| v.as_str());
                    let page_part = page_no.map_or(String::new(), |p| format!(", page {p}"));
                    let heading_part =
                        heading.map_or(String::new(), |h| format!(", under \"{h}\""));
                    let label = format!("{filename}{page_part}{heading_part}");
                    CitationView {
                        n: c.get("n").and_then(serde_json::Value::as_u64).unwrap_or(0),
                        label,
                        document_id: c
                            .get("document_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_owned(),
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let chart_json = meta
        .get("chart")
        .filter(|c| !c.is_null())
        .map(ToString::to_string);
    Some(MessageView {
        role: role.to_owned(),
        content: row.content.clone(),
        steps,
        citations,
        chart_json,
    })
}

async fn chat(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Query(q): Query<ChatQuery>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let user = access.identity.user_id.clone();
    let sees_all = access.sees_all_sessions();
    let wanted = q.session.clone();
    let (sessions_list, current, messages) = with_db(db, move |db| {
        let list = sessions::list_sessions_for(db, 50, &user, sees_all)?;
        let current = match wanted {
            Some(id) => {
                sessions::get_session(db, &id)?.filter(|s| sessions::visible_to(s, &user, sees_all))
            }
            None => None,
        };
        let messages = match &current {
            Some(s) => sessions::messages(db, &s.id)?,
            None => Vec::new(),
        };
        Ok((list, current, messages))
    })
    .await?;
    if let Some(current) = &current {
        access
            .audit(
                &app,
                "session_read",
                Some(("session", &current.id)),
                Outcome::Allowed,
                None,
            )
            .await?;
    }
    html(&ChatPage {
        page: page(&app, &access.identity, "Chat", Some(&access)),
        sessions: sessions_list,
        current,
        messages: messages.iter().filter_map(message_view).collect(),
    })
}

async fn delete_session(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, sid)): Path<(String, String)>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    super::api::sessions::delete_session(&app, &access, &sid).await?;
    Ok(Redirect::to(&format!("/w/{id}/chat")).into_response())
}

async fn render_rows(app: &App, access: &Access) -> WebResult<String> {
    let db = app.workspace_db(&access.workspace.id).await?;
    let documents = with_db(
        db,
        quack_core::storage::workspace::WorkspaceDb::list_documents,
    )
    .await?;
    let pending = documents
        .iter()
        .any(|d| d.status == "queued" || d.status == "processing");
    Ok(DocumentRows {
        ws_id: access.workspace.id.clone(),
        can_write: access.permits(Need::WRITE),
        documents,
        pending,
    }
    .render()?)
}

async fn documents(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Query(q): Query<FlashQuery>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let rows = render_rows(&app, &access).await?;
    html(&DocumentsPage {
        page: page(&app, &access.identity, "Documents", Some(&access)),
        rows,
        error: q.error,
    })
}

async fn document_rows(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    Ok(Html(render_rows(&app, &access).await?).into_response())
}

async fn upload(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    multipart: Multipart,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let mut files = Vec::new();
    let mut text = String::new();
    let mut title = String::new();
    let mut multipart = multipart;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::bad_request(e.to_string()))?
    {
        let name = field.name().unwrap_or("").to_owned();
        if let Some(filename) = field.file_name().map(str::to_owned) {
            let data = field
                .bytes()
                .await
                .map_err(|e| ApiError::bad_request(e.to_string()))?;
            if !filename.is_empty() && !data.is_empty() {
                files.push((filename, data.to_vec()));
            }
        } else {
            let value = field
                .text()
                .await
                .map_err(|e| ApiError::bad_request(e.to_string()))?;
            match name.as_str() {
                "text" => text = value,
                "title" => title = value,
                _ => {}
            }
        }
    }
    if !text.trim().is_empty() {
        files.push(docs_api::pasted_file(&text, Some(&title))?);
    }
    let outcome = docs_api::enqueue(&app, &access, files).await;
    let target = match outcome {
        Ok(_) => format!("/w/{id}/documents"),
        Err(e) => format!("/w/{id}/documents?error={}", urlencoded(&e.message)),
    };
    Ok(Redirect::to(&target).into_response())
}

fn urlencoded(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' => {
                char::from(b).to_string()
            }
            b' ' => String::from("+"),
            other => format!("%{other:02X}"),
        })
        .collect()
}

async fn pin(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, doc)): Path<(String, String)>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    docs_api::set_pinned(&app, &access, &doc, true).await?;
    Ok(Html(render_rows(&app, &access).await?).into_response())
}

async fn unpin(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, doc)): Path<(String, String)>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    docs_api::set_pinned(&app, &access, &doc, false).await?;
    Ok(Html(render_rows(&app, &access).await?).into_response())
}

async fn delete_doc(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, doc)): Path<(String, String)>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    docs_api::delete_document(&app, &access, &doc).await?;
    Ok(Html(render_rows(&app, &access).await?).into_response())
}

async fn tables(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let list = with_db(db, quack_core::storage::workspace::WorkspaceDb::list_tables).await?;
    html(&TablesPage {
        page: page(&app, &access.identity, "Tables", Some(&access)),
        tables: list,
        selected: None,
    })
}

fn cell(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

async fn table(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, name)): Path<(String, String)>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let wanted = name.clone();
    let (list, described) = with_db(db, move |db| {
        let list = db.list_tables()?;
        let described = if list.contains(&wanted) && !wanted.starts_with("_quack_") {
            Some(db.describe_table(&wanted)?)
        } else {
            None
        };
        Ok((list, described))
    })
    .await?;
    let described = described.ok_or_else(|| ApiError::not_found("no such table"))?;
    access
        .audit(&app, "open", Some(("table", &name)), Outcome::Allowed, None)
        .await?;
    html(&TablesPage {
        page: page(&app, &access.identity, &name, Some(&access)),
        tables: list,
        selected: Some(TableView {
            name: described.table_name,
            columns: described
                .columns
                .into_iter()
                .map(|c| (c.name, c.column_type))
                .collect(),
            sample_columns: described.sample_rows.columns,
            sample_rows: described
                .sample_rows
                .rows
                .iter()
                .map(|r| r.iter().map(cell).collect())
                .collect(),
        }),
    })
}

async fn sql_page(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    html(&SqlPage {
        page: page(&app, &access.identity, "SQL", Some(&access)),
        sql: String::new(),
        result: String::new(),
    })
}

#[derive(Deserialize)]
struct SqlForm {
    sql: String,
}

async fn render_sql(app: &App, access: &Access, sql: &str) -> WebResult<String> {
    let csv_href = format!("/w/{}/sql.csv?sql={}", access.workspace.id, urlencoded(sql));
    let result = match query_api::execute_sql(app, access, sql).await {
        Ok(outcome) => SqlResult {
            columns: outcome.columns,
            rows: outcome
                .rows
                .iter()
                .map(|r| r.iter().map(cell).collect())
                .collect(),
            row_count: outcome.row_count,
            truncated: outcome.truncated,
            error: None,
            csv_href,
        },
        Err(e) => SqlResult {
            columns: Vec::new(),
            rows: Vec::new(),
            row_count: 0,
            truncated: false,
            error: Some(e.message),
            csv_href,
        },
    };
    Ok(result.render()?)
}

async fn sql_run(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Form(form): Form<SqlForm>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    Ok(Html(render_sql(&app, &access, &form.sql).await?).into_response())
}

#[derive(Deserialize)]
struct SqlQuery {
    sql: String,
}

async fn sql_csv(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Query(q): Query<SqlQuery>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let outcome = query_api::execute_sql(&app, &access, &q.sql).await?;
    let mut csv = String::new();
    csv.push_str(
        &outcome
            .columns
            .iter()
            .map(|c| csv_field(c))
            .collect::<Vec<_>>()
            .join(","),
    );
    csv.push('\n');
    for row in &outcome.rows {
        csv.push_str(
            &row.iter()
                .map(|v| csv_field(&cell(v)))
                .collect::<Vec<_>>()
                .join(","),
        );
        csv.push('\n');
    }
    Ok((
        [
            (header::CONTENT_TYPE, "text/csv; charset=utf-8"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"query.csv\"",
            ),
        ],
        csv,
    )
        .into_response())
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

async fn context_page(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let (current, versions) = with_db(db, |db| {
        Ok((context::current(db)?, context::history(db, 20)?))
    })
    .await?;
    html(&ContextPage {
        page: page(&app, &access.identity, "Context", Some(&access)),
        content: current
            .as_ref()
            .map(|c| c.content.clone())
            .unwrap_or_default(),
        current,
        versions,
    })
}

#[derive(Deserialize)]
struct ContextForm {
    content: String,
}

async fn context_save(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Form(form): Form<ContextForm>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let db = app.workspace_db(&id).await?;
    let editor = access.identity.username.clone();
    let stored = with_db(db, move |db| context::set(db, &form.content, Some(&editor))).await?;
    access
        .audit(
            &app,
            "context",
            Some(("context", &stored.version.to_string())),
            Outcome::Allowed,
            Some(serde_json::json!({ "version": stored.version })),
        )
        .await?;
    Ok(Redirect::to(&format!("/w/{id}/context")).into_response())
}

#[derive(Deserialize)]
struct SettingsQuery {
    token: Option<String>,
    error: Option<String>,
}

async fn settings_view(
    app: &App,
    access: &Access,
    new_token: Option<String>,
    error: Option<String>,
) -> WebResult<Response> {
    let allowed: Vec<String> = access
        .workspace
        .allowed_providers
        .as_deref()
        .and_then(|p| serde_json::from_str(p).ok())
        .unwrap_or_default();
    let providers = app
        .config
        .providers
        .keys()
        .map(|name| (name.clone(), allowed.is_empty() || allowed.contains(name)))
        .collect();
    let (members, tokens) = if access.permits(Need::OWN) {
        (
            app.control.list_members(&access.workspace.id).await?,
            app.control.list_tokens(&access.workspace.id).await?,
        )
    } else {
        (Vec::new(), Vec::new())
    };
    html(&SettingsPage {
        page: page(app, &access.identity, "Settings", Some(access)),
        classification: access.workspace.classification.clone(),
        providers,
        members,
        tokens,
        new_token,
        error,
    })
}

async fn settings(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Query(q): Query<SettingsQuery>,
) -> WebResult<Response> {
    let need = Need {
        admin_ok: true,
        ..Need::READ
    };
    let access = access(&app, identity, &id, need).await?;
    settings_view(&app, &access, q.token, q.error).await
}

#[derive(Deserialize)]
struct SettingsForm {
    classification: String,
    #[serde(default)]
    providers: Vec<String>,
}

async fn settings_save(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    MultiForm(form): MultiForm<SettingsForm>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::OWN).await?;
    let all = app.config.providers.len();
    let allowed_providers = if form.providers.is_empty() || form.providers.len() == all {
        ProviderAllowList::All
    } else {
        ProviderAllowList::Only(form.providers)
    };
    app.control
        .update_workspace(
            &id,
            &WorkspaceChanges {
                classification: Some(form.classification.trim().to_owned()),
                allowed_providers,
            },
        )
        .await?;
    access
        .audit(
            &app,
            "workspace",
            Some(("workspace", &id)),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(Redirect::to(&format!("/w/{id}/settings")).into_response())
}

#[derive(Deserialize)]
struct MemberForm {
    username: String,
    role: String,
}

async fn member_add(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Form(form): Form<MemberForm>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::OWN).await?;
    let Some(user) = app.control.find_user_by_username(&form.username).await? else {
        return Ok(Redirect::to(&format!("/w/{id}/settings?error=no+such+user")).into_response());
    };
    let role = Role::parse(&form.role)?;
    app.control.set_member(&id, &user.id, role).await?;
    access
        .audit(
            &app,
            "member",
            Some(("user", &user.id)),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(Redirect::to(&format!("/w/{id}/settings")).into_response())
}

async fn member_remove(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, user_id)): Path<(String, String)>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::OWN).await?;
    app.control.remove_member(&id, &user_id).await?;
    access
        .audit(
            &app,
            "member",
            Some(("user", &user_id)),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(Redirect::to(&format!("/w/{id}/settings")).into_response())
}

#[derive(Deserialize)]
struct TokenForm {
    name: String,
    #[serde(default)]
    scopes: Vec<String>,
    expires_days: Option<u32>,
}

async fn token_create(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    MultiForm(form): MultiForm<TokenForm>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::OWN).await?;
    if app.local {
        return Ok(
            Redirect::to(&format!("/w/{id}/settings?error=local+mode+has+no+users"))
                .into_response(),
        );
    }
    let scopes = form
        .scopes
        .iter()
        .map(|s| Scope::parse(s))
        .collect::<quack_core::error::Result<Vec<_>>>()?;
    let expires_at = form.expires_days.filter(|d| *d > 0).and_then(|days| {
        jiff::Timestamp::now()
            .checked_add(jiff::SignedDuration::from_hours(
                i64::from(days).saturating_mul(24),
            ))
            .ok()
            .map(|t| t.strftime("%Y-%m-%d %H:%M:%S").to_string())
    });
    let (token, row) = app
        .control
        .create_token(
            &id,
            &access.identity.user_id,
            form.name.trim(),
            &scopes,
            expires_at.as_deref(),
        )
        .await?;
    access
        .audit(
            &app,
            "token",
            Some(("token", &row.token_hash)),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(Redirect::to(&format!("/w/{id}/settings?token={token}")).into_response())
}

async fn token_revoke(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, hash)): Path<(String, String)>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::OWN).await?;
    let owned = app
        .control
        .list_tokens(&id)
        .await?
        .into_iter()
        .any(|t| t.token_hash == hash);
    if !owned {
        return Err(ApiError::not_found("no such token").into());
    }
    app.control.delete_token(&hash).await?;
    access
        .audit(
            &app,
            "token",
            Some(("token", &hash)),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(Redirect::to(&format!("/w/{id}/settings")).into_response())
}

async fn admin_users(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Query(q): Query<FlashQuery>,
) -> WebResult<Response> {
    require_admin(&identity)?;
    html(&AdminUsersPage {
        page: page(&app, &identity, "Users", None),
        users: app.control.list_users().await?,
        error: q.error,
    })
}

#[derive(Deserialize)]
struct UserForm {
    username: String,
    password: String,
    #[serde(default)]
    is_admin: bool,
}

async fn admin_user_add(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Form(form): Form<UserForm>,
) -> WebResult<Response> {
    require_admin(&identity)?;
    if app.local {
        return Ok(Redirect::to("/admin/users?error=local+mode+has+no+users").into_response());
    }
    match app
        .control
        .create_user(&form.username, &form.password, form.is_admin)
        .await
    {
        Ok(user) => {
            let mut entry = identity.audit("admin", Outcome::Allowed);
            entry.resource_type = Some(String::from("user"));
            entry.resource_id = Some(user.id);
            app.control.record_audit(&entry).await?;
            Ok(Redirect::to("/admin/users").into_response())
        }
        Err(e) => Ok(Redirect::to(&format!(
            "/admin/users?error={}",
            urlencoded(&e.to_string())
        ))
        .into_response()),
    }
}

#[derive(Deserialize)]
struct AuditQuery {
    action: Option<String>,
    outcome: Option<String>,
    workspace_id: Option<String>,
}

async fn admin_audit(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Query(q): Query<AuditQuery>,
) -> WebResult<Response> {
    require_admin(&identity)?;
    let clean = |v: Option<String>| v.map(|s| s.trim().to_owned()).filter(|s| !s.is_empty());
    let (action, outcome, workspace_id) =
        (clean(q.action), clean(q.outcome), clean(q.workspace_id));
    let rows = app
        .control
        .query_audit(&AuditFilter {
            action: action.clone(),
            outcome: outcome.clone(),
            workspace_id: workspace_id.clone(),
            limit: 200,
            ..AuditFilter::default()
        })
        .await?;
    html(&AdminAuditPage {
        page: page(&app, &identity, "Audit", None),
        rows,
        action: action.unwrap_or_default(),
        outcome: outcome.unwrap_or_default(),
        workspace_id: workspace_id.unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urlencoded_escapes_reserved_bytes() {
        assert_eq!(urlencoded("a b&c=d/é"), "a+b%26c%3Dd%2F%C3%A9");
        assert_eq!(urlencoded("plain-text_1.2"), "plain-text_1.2");
    }

    #[test]
    fn csv_fields_quote_when_needed() {
        assert_eq!(csv_field("x"), "x");
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("q\"q"), "\"q\"\"q\"");
    }

    #[test]
    fn cells_render_strings_bare_and_null_empty() {
        assert_eq!(cell(&serde_json::json!("s")), "s");
        assert_eq!(cell(&serde_json::Value::Null), "");
        assert_eq!(cell(&serde_json::json!(4.5)), "4.5");
    }
}
