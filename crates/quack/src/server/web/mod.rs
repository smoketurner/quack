//! The web UI: server-rendered askama pages over the same access checks as
//! the API, htmx for the document list and SQL grid, and a small script for
//! the streamed chat (design doc 11.1). Everything a page does, the API can
//! do; the handlers here only shape the response as HTML.

pub(crate) mod markdown;

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
use axum_extra::extract::cookie::Cookie;
use quack_core::ontology::candidates::{CandidateAction, Queue};
use quack_core::ontology::induction::{Decision, ItemKind, Proposal, propose_from_tables};
use quack_core::ontology::store as ontology_store;
use quack_core::ontology::{Ontology, OntologyDiff, candidates};
use quack_core::storage::context;
use quack_core::storage::control::{
    AuditFilter, AuditRow, MemberRow, Outcome, ProviderAllowList, Role, Scope, TokenRow, UserRow,
    WorkspaceChanges,
};
use quack_core::storage::sessions::{self, MessageRole, SessionRow};
use quack_core::storage::workspace::{DocumentInfo, DocumentSource};
use rust_embed::Embed;
use serde::Deserialize;

use super::api::documents as docs_api;
use super::api::embeddings as embeddings_api;
use super::api::graph as graph_api;
use super::api::query as query_api;
use super::auth::{
    Access, Credential, Identity, Need, Peer, SESSION_COOKIE, access, password_login, request_id,
    require_admin, session_cookie,
};
use super::error::ApiError;
use super::state::{App, with_db};
use quack_core::embedding::Vector;
use quack_core::error::{Error as CoreError, Result as CoreResult};
use quack_core::graph::store as graph_store;
use quack_core::graph::{
    ExtractSource, GraphOptions, GraphResult, GraphStatus, extract, resolve, traverse,
};
use quack_core::import::ImportRequest;
use quack_core::ontology::ROOT_CLASS;
use quack_core::storage::workspace::WorkspaceDb;

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

impl From<CoreError> for HtmlError {
    fn from(err: CoreError) -> Self {
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
    /// Rendered HTML for assistant answers; escaped text for user messages.
    content_html: String,
    steps: Vec<StepView>,
    citations: Vec<CitationView>,
    chart_json: Option<String>,
    /// One JSON `GraphResult` per graph tool call the turn made.
    graphs: Vec<String>,
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
    /// What the workspace holds, for the empty state before any session.
    tables: Vec<String>,
    documents: Vec<DocumentInfo>,
}

#[derive(Template)]
#[template(path = "documents.html")]
struct DocumentsPage {
    page: Page,
    rows: String,
    error: Option<String>,
    notice: Option<String>,
    /// Chunks found by keyword only until a refresh, when there are any.
    embeddings_note: Option<String>,
}

#[derive(Template)]
#[template(path = "jobs.html")]
struct JobsPage {
    page: Page,
    rows: String,
}

/// One job as the Jobs page shows it.
struct JobView {
    id: String,
    number: u64,
    kind: String,
    label: String,
    /// `queued`, `running`, `cancelling`, or a final state.
    state: String,
    active: bool,
    can_cancel: bool,
    progress: String,
    outcome: Option<String>,
    queued_at: String,
}

#[derive(Template)]
#[template(path = "jobs_rows.html")]
struct JobRows {
    ws_id: String,
    jobs: Vec<JobView>,
    pending: bool,
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
    error: Option<String>,
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

struct ClassRow {
    depth: usize,
    id: String,
    key: Option<String>,
    properties: String,
}

struct CandidateView {
    id: String,
    kind: ItemKind,
    proposal_id: String,
    confidence: String,
    evidence: String,
    detail: String,
}

#[derive(Template)]
#[template(path = "ontology.html")]
struct OntologyPage {
    page: Page,
    ontology: Option<Ontology>,
    classes: Vec<ClassRow>,
    json: String,
    versions: Vec<ontology_store::VersionRow>,
    diff: Option<OntologyDiff>,
    /// The page of the queue being shown.
    queue: Vec<CandidateView>,
    /// Which queue `queue` shows.
    queue_status: Queue,
    queue_page: usize,
    queue_pages: usize,
    pending_total: usize,
    low_support_total: usize,
    has_tables: bool,
    error: Option<String>,
    notice: Option<String>,
}

/// Candidates shown per review page.
const CANDIDATES_PER_PAGE: usize = 50;

#[derive(Deserialize, Default)]
struct OntologyQuery {
    error: Option<String>,
    notice: Option<String>,
    /// The main queue unless `low_support` is asked for.
    #[serde(default, deserialize_with = "blank_as_none")]
    status: Option<Queue>,
    page: Option<usize>,
}

#[derive(Deserialize)]
struct BulkDecideForm {
    /// `accept` or `reject`.
    bulk: CandidateAction,
    #[serde(default)]
    ids: Vec<String>,
    /// The queue to return to.
    #[serde(default)]
    status: Queue,
}

/// One node as the graph page's inspector shows it.
struct GraphNodeView {
    id: String,
    label: String,
    class_id: String,
    provisional: bool,
    properties: String,
    sources: String,
}

struct GraphEdgeView {
    source: String,
    relation: String,
    target: String,
    sources: String,
}

struct GraphResultView {
    title: String,
    json: String,
    nodes: Vec<GraphNodeView>,
    edges: Vec<GraphEdgeView>,
}

#[derive(Default)]
struct GraphQueryView {
    entity: String,
    class: String,
    relation: String,
    hops: u32,
    from: String,
    to: String,
    max_hops: u32,
}

#[derive(Template)]
#[template(path = "graph.html")]
struct GraphPage {
    page: Page,
    status: GraphStatus,
    drift: Vec<String>,
    has_ontology: bool,
    chunk_count: usize,
    merges: Vec<resolve::MergeProposal>,
    query: GraphQueryView,
    result: Option<GraphResultView>,
    error: Option<String>,
    notice: Option<String>,
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
    outcome: Option<Outcome>,
    outcomes: &'static [Outcome],
    workspace_id: String,
}

// --- routes ------------------------------------------------------------------

pub(crate) fn router() -> Router<App> {
    Router::new()
        .route("/", get(index))
        .route("/static/{*path}", get(static_asset))
        .route(
            "/login",
            get(login_page).merge(super::throttled_login(post(login_submit))),
        )
        .route("/logout", post(logout))
        .route("/workspaces", get(workspaces).post(create_workspace))
        .route("/w/{id}", get(workspace_index))
        .route("/w/{id}/chat", get(chat))
        .route("/w/{id}/chat/{sid}/delete", post(delete_session))
        .route("/w/{id}/chat/{sid}/share", post(share_session))
        .route("/w/{id}/chat/{sid}/unshare", post(unshare_session))
        .route("/w/{id}/documents", get(documents).post(upload))
        .route("/w/{id}/documents/rows", get(document_rows))
        .route("/w/{id}/documents/{doc}/pin", post(pin))
        .route("/w/{id}/documents/{doc}/unpin", post(unpin))
        .route("/w/{id}/documents/{doc}/delete", post(delete_doc))
        .route("/w/{id}/embeddings/refresh", post(refresh_embeddings))
        .route("/w/{id}/jobs", get(jobs_page))
        .route("/w/{id}/jobs/rows", get(job_rows))
        .route("/w/{id}/jobs/{job}/cancel", post(job_cancel))
        .route("/w/{id}/tables", get(tables))
        .route("/w/{id}/import", post(import_submit))
        .route("/w/{id}/tables/{name}", get(table))
        .route("/w/{id}/sql", get(sql_page).post(sql_run))
        .route("/w/{id}/sql.csv", get(sql_csv))
        .route("/w/{id}/context", get(context_page).post(context_save))
        .route("/w/{id}/ontology", get(ontology_page).post(ontology_import))
        .route("/w/{id}/ontology/init", post(ontology_init))
        .route("/w/{id}/ontology/propose", post(ontology_propose))
        .route("/w/{id}/ontology/candidates", post(ontology_decide_many))
        .route("/w/{id}/ontology/candidates/{cid}", post(ontology_decide))
        .route("/w/{id}/ontology/{v}/restore", post(ontology_restore))
        .route("/w/{id}/graph", get(graph_page))
        .route("/w/{id}/graph/extract", post(graph_extract))
        .route("/w/{id}/graph/revalidate", post(graph_revalidate))
        .route("/w/{id}/graph/review", post(graph_review))
        .route("/w/{id}/graph/merges/{mid}", post(graph_merge_decide))
        .route("/w/{id}/settings", get(settings).post(settings_save))
        .route("/w/{id}/members", post(member_add))
        .route("/w/{id}/members/{user}/remove", post(member_remove))
        .route("/w/{id}/tokens", post(token_create))
        .route("/w/{id}/tokens/{hash}/revoke", post(token_revoke))
        .route("/admin/users", get(admin_users).post(admin_user_add))
        .route("/admin/audit", get(admin_audit))
}

/// Embedded assets with an `ETag` from the content hash and `no-cache`, so a
/// browser revalidates on every load and picks up a rebuilt stylesheet or
/// script immediately, at the cost of one cheap 304 per asset.
async fn static_asset(Path(path): Path<String>, headers: axum::http::HeaderMap) -> Response {
    let Some(file) = Assets::get(&path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let mut etag = String::from("\"");
    for b in file.metadata.sha256_hash() {
        etag.push(char::from_digit(u32::from(b >> 4), 16).unwrap_or('0'));
        etag.push(char::from_digit(u32::from(b & 0x0f), 16).unwrap_or('0'));
    }
    etag.push('"');
    if headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').any(|candidate| candidate.trim() == etag))
    {
        return (
            StatusCode::NOT_MODIFIED,
            [
                (header::ETAG, etag),
                (header::CACHE_CONTROL, String::from("no-cache")),
            ],
        )
            .into_response();
    }
    let mime = mime_guess::from_path(&path).first_or_octet_stream();
    (
        [
            (header::CONTENT_TYPE, mime.as_ref().to_owned()),
            (header::CACHE_CONTROL, String::from("no-cache")),
            (header::ETAG, etag),
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
    peer: Peer,
    jar: CookieJar,
    headers: axum::http::HeaderMap,
    Form(form): Form<LoginForm>,
) -> WebResult<Response> {
    if app.local {
        return Ok(Redirect::to("/workspaces").into_response());
    }
    // A wrong password is the form again with a message, not a 401; any
    // other failure is still an error page.
    let token = match password_login(
        &app,
        peer,
        request_id(&headers),
        &form.username,
        &form.password,
    )
    .await
    {
        Ok((_, token)) => token,
        Err(e) if e.status == StatusCode::UNAUTHORIZED => {
            return Ok(Redirect::to("/login?error=wrong+username+or+password").into_response());
        }
        Err(e) => return Err(e.into()),
    };
    Ok((
        jar.add(session_cookie(&app, peer, token)),
        Redirect::to("/workspaces"),
    )
        .into_response())
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
    /// A success message: the same flash slot, green.
    notice: Option<String>,
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
    let graphs = meta
        .get("graph")
        .and_then(|g| g.as_array())
        .map(|gs| gs.iter().map(ToString::to_string).collect())
        .unwrap_or_default();
    let content_html = if row.role == MessageRole::Assistant {
        markdown::to_html(&row.content)
    } else {
        askama::filters::escape(&row.content, askama::filters::Html)
            .map(|e| e.to_string())
            .unwrap_or_default()
    };
    Some(MessageView {
        role: role.to_owned(),
        content_html,
        steps,
        citations,
        chart_json,
        graphs,
    })
}

async fn chat(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Query(q): Query<ChatQuery>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let user = access.identity.user_id.clone();
    let sees_all = access.sees_all_sessions();
    let wanted = q.session.clone();
    let (sessions_list, current, messages) = app
        .read(&id, move |db| {
            let list = sessions::list_sessions_for(db, 50, &user, sees_all)?;
            let current = match wanted {
                Some(id) => sessions::get_session(db, &id)?
                    .filter(|s| sessions::visible_to(s, &user, sees_all)),
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
    // The empty state says what there is to ask about.
    let (tables, documents) = if messages.is_empty() {
        app.read(&id, |db| Ok((db.list_tables()?, db.list_documents()?)))
            .await?
    } else {
        (Vec::new(), Vec::new())
    };
    html(&ChatPage {
        page: page(&app, &access.identity, "Chat", Some(&access)),
        sessions: sessions_list,
        current,
        messages: messages.iter().filter_map(message_view).collect(),
        tables,
        documents,
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

async fn share_session(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, sid)): Path<(String, String)>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    super::api::sessions::set_shared(&app, &access, &sid, true).await?;
    Ok(Redirect::to(&format!("/w/{id}/chat?session={sid}")).into_response())
}

async fn unshare_session(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, sid)): Path<(String, String)>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    super::api::sessions::set_shared(&app, &access, &sid, false).await?;
    Ok(Redirect::to(&format!("/w/{id}/chat?session={sid}")).into_response())
}

async fn render_rows(app: &App, access: &Access) -> WebResult<String> {
    let documents = app
        .read(&access.workspace.id, WorkspaceDb::list_documents)
        .await?;
    let pending = documents.iter().any(|d| d.status.is_in_flight());
    Ok(DocumentRows {
        ws_id: access.workspace.id.clone(),
        can_write: access.permits(Need::WRITE),
        documents,
        pending,
    }
    .render()?)
}

fn render_jobs(app: &App, access: &Access) -> WebResult<String> {
    let jobs: Vec<JobView> = super::api::jobs::visible_jobs(app, access)
        .into_iter()
        .map(|j| JobView {
            id: j.id.to_string(),
            number: j.number,
            kind: j.kind.to_string(),
            can_cancel: !j.state.is_finished() && super::api::jobs::may_cancel(access, &j),
            label: j.label,
            state: if j.cancel_requested && !j.state.is_finished() {
                String::from("cancelling")
            } else {
                j.state.to_string()
            },
            active: !j.state.is_finished(),
            progress: j.progress.map(|p| p.to_string()).unwrap_or_default(),
            outcome: j.outcome.or(j.status),
            queued_at: j.queued_at.strftime("%Y-%m-%d %H:%M:%S").to_string(),
        })
        .collect();
    let pending = jobs.iter().any(|j| j.active);
    Ok(JobRows {
        ws_id: access.workspace.id.clone(),
        jobs,
        pending,
    }
    .render()?)
}

async fn jobs_page(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, "page", "jobs").await?;
    let rows = render_jobs(&app, &access)?;
    html(&JobsPage {
        page: page(&app, &access.identity, "Jobs", Some(&access)),
        rows,
    })
}

async fn job_rows(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, "page", "job_rows").await?;
    Ok(Html(render_jobs(&app, &access)?).into_response())
}

async fn job_cancel(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, job)): Path<(String, String)>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let cancelled = super::api::jobs::cancel_job(&app, &access, &job).await?;
    tracing::debug!(job = %cancelled.id, state = %cancelled.state, "cancel requested from the web");
    Ok(Html(render_jobs(&app, &access)?).into_response())
}

async fn documents(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Query(q): Query<FlashQuery>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, "page", "documents").await?;
    let rows = render_rows(&app, &access).await?;
    let embeddings_note = app.read(&id, WorkspaceDb::embedding_status).await?.note();
    html(&DocumentsPage {
        page: page(&app, &access.identity, "Documents", Some(&access)),
        rows,
        error: q.error,
        notice: q.notice,
        embeddings_note,
    })
}

/// The Documents page's refresh button: the API's refresh, then back
/// to the page with what it started.
async fn refresh_embeddings(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let target = match embeddings_api::start(&app, &access, &id).await {
        Ok((_, body)) if body.get("status").and_then(|s| s.as_str()) == Some("running") => {
            format!(
                "/w/{id}/documents?notice={}",
                urlencoded(
                    "refreshing embeddings in the background; the Jobs page shows its progress"
                )
            )
        }
        Ok(_) => format!(
            "/w/{id}/documents?notice={}",
            urlencoded("every vector is already current")
        ),
        Err(e) => format!("/w/{id}/documents?error={}", urlencoded(&e.message)),
    };
    Ok(Redirect::to(&target).into_response())
}

async fn document_rows(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, "page", "document_rows").await?;
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
    let pasted = if text.trim().is_empty() {
        Vec::new()
    } else {
        vec![docs_api::pasted_file(&text, Some(&title))?]
    };
    let outcome = enqueue_web(&app, &access, files, pasted).await;
    let target = match outcome {
        Ok(skipped) if skipped.is_empty() => format!("/w/{id}/documents"),
        Ok(skipped) => format!(
            "/w/{id}/documents?error={}",
            urlencoded(&format!("Already in the workspace: {}", skipped.join(", ")))
        ),
        Err(e) => format!("/w/{id}/documents?error={}", urlencoded(&e.message)),
    };
    Ok(Redirect::to(&target).into_response())
}

/// Queue uploaded files and pasted text under their own sources; returns
/// the names of files skipped as duplicates.
async fn enqueue_web(
    app: &App,
    access: &Access,
    files: Vec<(String, Vec<u8>)>,
    pasted: Vec<(String, Vec<u8>)>,
) -> Result<Vec<String>, ApiError> {
    if files.is_empty() && pasted.is_empty() {
        return Err(ApiError::bad_request("no file or text in the request"));
    }
    let mut queued = Vec::new();
    if !files.is_empty() {
        queued.extend(docs_api::enqueue(app, access, DocumentSource::Upload, files).await?);
    }
    if !pasted.is_empty() {
        queued.extend(docs_api::enqueue(app, access, DocumentSource::Paste, pasted).await?);
    }
    let mut skipped = Vec::new();
    for entry in &queued {
        if entry.get("status").and_then(serde_json::Value::as_str) == Some("duplicate")
            && let Some(name) = entry.get("filename").and_then(serde_json::Value::as_str)
        {
            skipped.push(name.to_owned());
        }
    }
    Ok(skipped)
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
    Query(q): Query<FlashQuery>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, "page", "tables").await?;
    let list = app.read(&id, WorkspaceDb::list_tables).await?;
    html(&TablesPage {
        page: page(&app, &access.identity, "Tables", Some(&access)),
        tables: list,
        selected: None,
        error: q.error,
    })
}

#[derive(Deserialize)]
struct ImportForm {
    url: String,
    table: String,
    #[serde(default)]
    query: String,
    #[serde(default)]
    source_table: String,
}

async fn import_submit(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Form(form): Form<ImportForm>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let request = ImportRequest {
        url: form.url,
        table: form.table,
        query: (!form.query.trim().is_empty()).then(|| form.query.clone()),
        source_table: (!form.source_table.trim().is_empty()).then(|| form.source_table.clone()),
        limit: None,
    };
    let target = match super::api::import::run_import(&app, &access, &request).await {
        Ok(summary) => format!("/w/{id}/tables/{}", summary.table),
        Err(e) => format!("/w/{id}/tables?error={}", urlencoded(&e.message)),
    };
    Ok(Redirect::to(&target).into_response())
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
    let wanted = name.clone();
    let (list, described) = app
        .read(&id, move |db| {
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
        .audit(
            &app,
            "open",
            None,
            Outcome::Allowed,
            Some(serde_json::json!({ "table": name })),
        )
        .await?;
    html(&TablesPage {
        page: page(&app, &access.identity, &name, Some(&access)),
        tables: list,
        error: None,
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
    access.audit_read(&app, "page", "sql").await?;
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

fn class_rows(ontology: &Ontology) -> Vec<ClassRow> {
    fn walk(ontology: &Ontology, parent: &str, depth: usize, out: &mut Vec<ClassRow>) {
        for class in ontology.classes.iter().filter(|c| c.parent == parent) {
            out.push(ClassRow {
                depth,
                id: class.id.clone(),
                key: class.key.clone(),
                properties: ontology
                    .class_properties(&class.id)
                    .into_iter()
                    .collect::<Vec<_>>()
                    .join(", "),
            });
            walk(ontology, &class.id, depth.saturating_add(1), out);
        }
    }
    let mut out = Vec::new();
    walk(ontology, ROOT_CLASS, 0, &mut out);
    out
}

async fn ontology_page(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Query(q): Query<OntologyQuery>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, "page", "ontology").await?;
    let queue_status = q.status.unwrap_or_default();
    let (ontology, versions, diff, pending, low_support, has_tables) = app
        .read(&id, |db| {
            let current = ontology_store::current(db)?;
            let versions = ontology_store::versions(db, 20)?;
            let diff = match &current {
                Some(c) if c.version > 1 => {
                    ontology_store::version(db, c.version.saturating_sub(1))?
                        .map(|older| c.diff(&older))
                }
                _ => None,
            };
            let pending = candidates::queue(db, Queue::Pending)?;
            let low_support = candidates::queue(db, Queue::LowSupport)?;
            let has_tables = !db.list_tables()?.is_empty();
            Ok((current, versions, diff, pending, low_support, has_tables))
        })
        .await?;
    let (pending_total, low_support_total) = (pending.len(), low_support.len());
    let rows = match queue_status {
        Queue::Pending => pending,
        Queue::LowSupport => low_support,
    };
    // The queue is paged (issue #55): 221 candidates from one document
    // pass are not one wall of rows.
    let queue_pages = rows.len().div_ceil(CANDIDATES_PER_PAGE).max(1);
    let queue_page = q.page.unwrap_or(1).clamp(1, queue_pages);
    let queue = rows
        .into_iter()
        .skip(
            queue_page
                .saturating_sub(1)
                .saturating_mul(CANDIDATES_PER_PAGE),
        )
        .take(CANDIDATES_PER_PAGE)
        .map(candidate_view)
        .collect();
    let json = match &ontology {
        Some(o) => o.to_json()?,
        None => String::new(),
    };
    html(&OntologyPage {
        page: page(&app, &access.identity, "Ontology", Some(&access)),
        classes: ontology.as_ref().map(class_rows).unwrap_or_default(),
        ontology,
        json,
        versions,
        diff,
        queue,
        queue_status,
        queue_page,
        queue_pages,
        pending_total,
        low_support_total,
        has_tables,
        error: q.error,
        notice: q.notice,
    })
}

/// Accept or reject every ticked candidate at once (issue #55).
async fn ontology_decide_many(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    MultiForm(form): MultiForm<BulkDecideForm>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let back = match form.status {
        Queue::LowSupport => format!("/w/{id}/ontology?status={}", Queue::LowSupport),
        Queue::Pending => format!("/w/{id}/ontology"),
    };
    if form.ids.is_empty() {
        return Ok(Redirect::to(&format!(
            "{back}{}error=tick+at+least+one+candidate",
            if back.contains('?') { "&" } else { "?" }
        ))
        .into_response());
    }
    let accept = match form.bulk {
        CandidateAction::Accept => true,
        CandidateAction::Reject => false,
        CandidateAction::Rename | CandidateAction::MergeInto | CandidateAction::Reparent => {
            return Ok(Redirect::to(&format!(
                "{back}{}error=unknown+bulk+action",
                if back.contains('?') { "&" } else { "?" }
            ))
            .into_response());
        }
    };
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let ids = form.ids.clone();
    let outcome = with_db(db, move |db| {
        if accept {
            let decisions: Vec<(String, Decision)> =
                ids.into_iter().map(|id| (id, Decision::Accept)).collect();
            Ok(Some(
                candidates::accept(db, &decisions, Some(&author))?.version,
            ))
        } else {
            candidates::reject(db, &ids, Some(&author))?;
            Ok(None)
        }
    })
    .await;
    let target = match outcome {
        Ok(version) => {
            access
                .audit(
                    &app,
                    "ontology",
                    None,
                    Outcome::Allowed,
                    Some(serde_json::json!({
                        "bulk": form.bulk,
                        "ids": form.ids,
                        "version": version,
                    })),
                )
                .await?;
            back
        }
        Err(e) => format!(
            "{back}{}error={}",
            if back.contains('?') { "&" } else { "?" },
            urlencoded(&e.message)
        ),
    };
    Ok(Redirect::to(&target).into_response())
}

/// The one-line detail of a model- or bundle-sourced proposal.
fn proposal_detail(proposal: &Proposal) -> String {
    match proposal {
        Proposal::Relation(r) => format!("{} → {}", r.domain, r.range),
        Proposal::Class(cl) => format!("parent {}", cl.parent),
        Proposal::Property { class, property } => {
            format!("{}: {}", class, property.kind.as_str())
        }
        Proposal::Mapping(_) => String::new(),
    }
}

/// "N bundle files, e.g. ..." for an OKF-bundle candidate.
fn bundle_evidence(e: &serde_json::Value) -> String {
    let examples = e
        .get("examples")
        .and_then(|x| x.as_array())
        .map(|xs| {
            xs.iter()
                .filter_map(|x| x.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    format!(
        "{} bundle files · e.g. {examples}",
        e.get("files").map(ToString::to_string).unwrap_or_default()
    )
}

/// "N mentions in M documents, e.g. ..." for a document-evidence candidate.
fn document_evidence(e: &serde_json::Value) -> String {
    let get = |k: &str| e.get(k).map(ToString::to_string).unwrap_or_default();
    let examples = e
        .get("examples")
        .and_then(|x| x.as_array())
        .map(|xs| {
            xs.iter()
                .filter_map(|x| {
                    x.get("mention")
                        .or_else(|| x.get("subject"))
                        .or_else(|| x.get("value"))
                        .and_then(|v| v.as_str())
                })
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    format!(
        "{} mentions in {} documents · e.g. {examples}",
        get("occurrences"),
        get("documents")
    )
}

fn candidate_view(c: candidates::CandidateRow) -> CandidateView {
    let e = &c.evidence;
    let get = |k: &str| {
        e.get(k)
            .map(|v| match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .unwrap_or_default()
    };
    let source = e.get("source").and_then(|v| v.as_str());
    let from_documents = source == Some("documents");
    let from_bundle = source == Some("okf");
    let (evidence, detail) = match c.kind {
        _ if from_bundle => (bundle_evidence(e), proposal_detail(&c.proposal)),
        _ if from_documents => (document_evidence(e), proposal_detail(&c.proposal)),
        ItemKind::Class => (
            format!(
                "table {} · {} rows · key {}",
                get("table"),
                get("rows"),
                get("key_column")
            ),
            String::new(),
        ),
        ItemKind::Property => (
            format!(
                "{}.{} · {} · {} distinct of {} · e.g. {}",
                get("table"),
                get("column"),
                get("duckdb_type"),
                get("distinct"),
                get("rows"),
                get("samples")
            ),
            match &c.proposal {
                Proposal::Property { class, property } => {
                    let values = if property.values.is_empty() {
                        String::new()
                    } else {
                        format!(" [{}]", property.values.join(", "))
                    };
                    format!("{}: {}{values}", class, property.kind.as_str())
                }
                _ => String::new(),
            },
        ),
        ItemKind::Relation => (
            format!(
                "{}.{} matches {}.{} for {} of values",
                get("table"),
                get("column"),
                get("target_table"),
                get("target_key"),
                get("overlap")
            ),
            match &c.proposal {
                Proposal::Relation(r) => format!("{} → {}", r.domain, r.range),
                _ => String::new(),
            },
        ),
        ItemKind::Mapping => (
            format!("table {}", get("table")),
            match &c.proposal {
                Proposal::Mapping(m) => format!(
                    "{} → {} (key {}, {} relations)",
                    m.table,
                    m.class,
                    m.key,
                    m.relations.len()
                ),
                _ => String::new(),
            },
        ),
    };
    CandidateView {
        id: c.id,
        kind: c.kind,
        proposal_id: c.proposal.id().to_owned(),
        confidence: format!("{:.2}", c.confidence),
        evidence,
        detail,
    }
}

#[derive(Deserialize)]
struct ProposeForm {
    #[serde(default)]
    auto_accept: bool,
    #[serde(default)]
    documents: bool,
}

async fn ontology_propose(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Form(form): Form<ProposeForm>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    if form.documents {
        let target = match super::api::ontology::start_document_run(&app, &access, &id, None).await
        {
            Ok(_) => format!(
                "/w/{id}/ontology?notice=document+pass+started%3B+candidates+appear+here+when+it+finishes"
            ),
            Err(e) => format!("/w/{id}/ontology?error={}", urlencoded(&e.message)),
        };
        return Ok(Redirect::to(&target).into_response());
    }
    let options = app.config.ontology.table_evidence();
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let outcome = with_db(db, move |db| {
        let current = ontology_store::current(db)?;
        let proposals = propose_from_tables(db, current.as_ref(), &options)?;
        if proposals.is_empty() {
            return Ok((0, None));
        }
        let run = candidates::store_run(db, &proposals)?;
        if form.auto_accept {
            candidates::accept_all(db, Some(&author))?;
        }
        Ok((proposals.len(), Some(run)))
    })
    .await;
    let target = match outcome {
        Ok((0, _)) => {
            format!("/w/{id}/ontology?notice=nothing+to+propose%3A+the+tables+are+already+covered")
        }
        Ok((count, run)) => {
            access
                .audit(
                    &app,
                    "propose",
                    run.as_deref().map(|r| ("induction_run", r)),
                    Outcome::Allowed,
                    Some(serde_json::json!({ "candidates": count })),
                )
                .await?;
            format!("/w/{id}/ontology")
        }
        Err(e) => format!("/w/{id}/ontology?error={}", urlencoded(&e.message)),
    };
    Ok(Redirect::to(&target).into_response())
}

#[derive(Deserialize)]
struct DecideForm {
    action: CandidateAction,
    #[serde(default)]
    target: String,
}

async fn ontology_decide(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, cid)): Path<(String, String)>,
    Form(form): Form<DecideForm>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let decision = match form.action.decision(Some(&form.target)) {
        Ok(decision) => decision,
        Err(e) => {
            let target = format!("/w/{id}/ontology?error={}", urlencoded(&e.to_string()));
            return Ok(Redirect::to(&target).into_response());
        }
    };
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let candidate_id = cid.clone();
    let outcome = with_db(db, move |db| {
        if let Some(decision) = decision {
            let stored = candidates::accept(db, &[(candidate_id, decision)], Some(&author))?;
            return Ok(Some(stored.version));
        }
        candidates::reject(db, &[candidate_id], Some(&author))?;
        Ok(None)
    })
    .await;
    let target = match outcome {
        Ok(version) => {
            access
                .audit(
                    &app,
                    "ontology",
                    Some(("candidate", &cid)),
                    Outcome::Allowed,
                    Some(serde_json::json!({ "action": form.action, "version": version })),
                )
                .await?;
            format!("/w/{id}/ontology")
        }
        Err(e) => format!("/w/{id}/ontology?error={}", urlencoded(&e.message)),
    };
    Ok(Redirect::to(&target).into_response())
}

#[derive(Deserialize)]
struct OntologyForm {
    json: String,
}

async fn ontology_import(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Form(form): Form<OntologyForm>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let ontology = match Ontology::from_json(&form.json) {
        Ok(o) => o,
        Err(e) => {
            let target = format!("/w/{id}/ontology?error={}", urlencoded(&e.to_string()));
            return Ok(Redirect::to(&target).into_response());
        }
    };
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let saved = with_db(db, move |db| {
        ontology_store::save(db, &ontology, Some(&author), Some("edited in the web UI"))
    })
    .await;
    match saved {
        Ok(stored) => {
            access
                .audit(
                    &app,
                    "ontology",
                    Some(("ontology_version", &stored.version.to_string())),
                    Outcome::Allowed,
                    Some(serde_json::json!({ "version": stored.version })),
                )
                .await?;
            Ok(Redirect::to(&format!("/w/{id}/ontology")).into_response())
        }
        Err(e) => {
            let target = format!("/w/{id}/ontology?error={}", urlencoded(&e.message));
            Ok(Redirect::to(&target).into_response())
        }
    }
}

async fn ontology_init(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let saved = with_db(db, move |db| {
        if ontology_store::latest_version(db)? > 0 {
            return Err(CoreError::Ontology(String::from(
                "an ontology already exists",
            )));
        }
        ontology_store::save(
            db,
            &Ontology::builtin_default(),
            Some(&author),
            Some("built-in default"),
        )
    })
    .await;
    if let Ok(stored) = &saved {
        access
            .audit(
                &app,
                "ontology",
                Some(("ontology_version", &stored.version.to_string())),
                Outcome::Allowed,
                None,
            )
            .await?;
    }
    let target = match saved {
        Ok(_) => format!("/w/{id}/ontology"),
        Err(e) => format!("/w/{id}/ontology?error={}", urlencoded(&e.message)),
    };
    Ok(Redirect::to(&target).into_response())
}

async fn ontology_restore(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, v)): Path<(String, u32)>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let stored = with_db(db, move |db| ontology_store::restore(db, v, Some(&author))).await?;
    access
        .audit(
            &app,
            "ontology",
            Some(("ontology_version", &stored.version.to_string())),
            Outcome::Allowed,
            Some(serde_json::json!({ "restored": v })),
        )
        .await?;
    Ok(Redirect::to(&format!("/w/{id}/ontology")).into_response())
}

async fn context_page(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, "page", "context").await?;
    let (current, versions) = app
        .read(&id, |db| {
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
    access.audit_read(&app, "page", "settings").await?;
    settings_view(&app, &access, None, q.error).await
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
    role: Role,
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
    app.control.set_member(&id, &user.id, form.role).await?;
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
    scopes: Vec<Scope>,
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
    let scopes = form.scopes;
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
    // The secret is shown once in this response body, never in a URL where
    // browser history, proxy logs, or a Referer would keep it.
    settings_view(&app, &access, Some(token), None).await
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
    #[serde(default, deserialize_with = "blank_as_none")]
    outcome: Option<Outcome>,
    workspace_id: Option<String>,
}

/// A query or form value where blank means "not given", as the filter's
/// "any" option sends it; anything else must parse.
fn blank_as_none<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match Option::<String>::deserialize(deserializer)?
        .as_deref()
        .map(str::trim)
    {
        None | Some("") => Ok(None),
        Some(text) => text.parse().map(Some).map_err(serde::de::Error::custom),
    }
}

async fn admin_audit(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Query(q): Query<AuditQuery>,
) -> WebResult<Response> {
    require_admin(&identity)?;
    let clean = |v: Option<String>| v.map(|s| s.trim().to_owned()).filter(|s| !s.is_empty());
    let (action, outcome, workspace_id) = (clean(q.action), q.outcome, clean(q.workspace_id));
    let rows = app
        .control
        .query_audit(&AuditFilter {
            action: action.clone(),
            outcome,
            workspace_id: workspace_id.clone(),
            limit: 200,
            ..AuditFilter::default()
        })
        .await?;
    html(&AdminAuditPage {
        page: page(&app, &identity, "Audit", None),
        rows,
        action: action.unwrap_or_default(),
        outcome,
        outcomes: Outcome::ALL,
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

// --- graph ----------------------------------------------------------------------

#[derive(Deserialize, Default)]
struct GraphPageQuery {
    entity: Option<String>,
    class: Option<String>,
    relation: Option<String>,
    hops: Option<u32>,
    from: Option<String>,
    to: Option<String>,
    max_hops: Option<u32>,
    error: Option<String>,
    notice: Option<String>,
}

fn non_empty(value: Option<&String>) -> Option<String> {
    value.map(|v| v.trim().to_owned()).filter(|v| !v.is_empty())
}

async fn graph_page(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Query(q): Query<GraphPageQuery>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, "page", "graph").await?;
    let options = app.config.graph.options();
    let query = GraphQueryView {
        entity: non_empty(q.entity.as_ref()).unwrap_or_default(),
        class: non_empty(q.class.as_ref()).unwrap_or_default(),
        relation: non_empty(q.relation.as_ref()).unwrap_or_default(),
        hops: q.hops.unwrap_or(2).clamp(1, 6),
        from: non_empty(q.from.as_ref()).unwrap_or_default(),
        to: non_empty(q.to.as_ref()).unwrap_or_default(),
        max_hops: q.max_hops.unwrap_or(4).clamp(1, 8),
    };
    let embedding = if query.entity.is_empty() {
        None
    } else {
        graph_api::entity_embedding(&app, &query.entity).await?
    };
    let path_embeddings: Option<EndEmbeddings> = if query.from.is_empty() || query.to.is_empty() {
        None
    } else {
        Some((
            graph_api::entity_embedding(&app, &query.from).await?,
            graph_api::entity_embedding(&app, &query.to).await?,
        ))
    };
    let wanted = GraphQueryView {
        entity: query.entity.clone(),
        class: query.class.clone(),
        relation: query.relation.clone(),
        hops: query.hops,
        from: query.from.clone(),
        to: query.to.clone(),
        max_hops: query.max_hops,
    };
    let (status, has_ontology, chunk_count, merges, result) = app
        .read(&id, move |db| {
            graph_page_data(
                db,
                &wanted,
                embedding.as_deref(),
                path_embeddings.as_ref(),
                options,
            )
        })
        .await?;
    let result = match result {
        Some((title, found)) => Some(graph_result_view(title, &found)?),
        None => None,
    };
    let mut drift: Vec<String> = status
        .drift
        .classes
        .keys()
        .map(|c| format!("class {c}"))
        .chain(
            status
                .drift
                .relations
                .keys()
                .map(|r| format!("relation {r}")),
        )
        .collect();
    drift.sort();
    html(&GraphPage {
        page: page(&app, &access.identity, "Graph", Some(&access)),
        status,
        drift,
        has_ontology,
        chunk_count,
        merges,
        query,
        result,
        error: q.error,
        notice: q.notice,
    })
}

type PageData = (
    GraphStatus,
    bool,
    usize,
    Vec<resolve::MergeProposal>,
    Option<(String, GraphResult)>,
);

/// Status, ontology presence, chunk count, merge queue, and the result of
/// whatever the query asked for.
/// The embeddings of a path query's two ends, when a model exists.
type EndEmbeddings = (Option<Vector>, Option<Vector>);

fn graph_page_data(
    db: &WorkspaceDb,
    wanted: &GraphQueryView,
    embedding: Option<&[f32]>,
    path_embeddings: Option<&EndEmbeddings>,
    options: GraphOptions,
) -> CoreResult<PageData> {
    let status = graph_store::status(db)?;
    let ontology = ontology_store::current(db)?;
    let chunk_count = usize::try_from(extract::pending_chunk_count(db)?).unwrap_or(0);
    let merges = resolve::pending(db)?;
    let result = if let Some((a, b)) = path_embeddings {
        let from = traverse::resolve_entry(db, &wanted.from, None, a.as_deref())?;
        let to = traverse::resolve_entry(db, &wanted.to, None, b.as_deref())?;
        let found = match (from.first(), to.first()) {
            (Some(a), Some(b)) => traverse::path(db, a, b, wanted.max_hops, &options)?,
            _ => GraphResult::default(),
        };
        Some((format!("Path from {} to {}", wanted.from, wanted.to), found))
    } else if !wanted.entity.is_empty() {
        let class = (!wanted.class.is_empty()).then_some(wanted.class.as_str());
        let relation = (!wanted.relation.is_empty()).then_some(wanted.relation.as_str());
        let roots = traverse::resolve_entry(db, &wanted.entity, class, embedding)?;
        let found = traverse::neighborhood(db, &roots, wanted.hops, relation, &options)?;
        Some((format!("Around {}", wanted.entity), found))
    } else if !wanted.class.is_empty() {
        let found = traverse::by_class(
            db,
            ontology.as_ref(),
            &wanted.class,
            options.max_nodes,
            &options,
        )?;
        Some((format!("Entities of class {}", wanted.class), found))
    } else {
        None
    };
    Ok((status, ontology.is_some(), chunk_count, merges, result))
}

fn graph_result_view(title: String, result: &GraphResult) -> Result<GraphResultView, ApiError> {
    let sources_of = |subject: &str| -> String {
        let mut items: Vec<String> = result
            .provenance
            .iter()
            .filter(|p| p.subject_id == subject)
            .map(|p| match (&p.table_name, &p.document_id) {
                (Some(table), _) => format!("{table} row {}", p.row_key.as_deref().unwrap_or("?")),
                (None, Some(document)) => format!("document {}", short_id(document)),
                (None, None) => String::from("unknown"),
            })
            .collect();
        items.sort();
        items.dedup();
        items.join(", ")
    };
    let label_of = |id: &str| -> String {
        result
            .nodes
            .iter()
            .find(|n| n.id == id)
            .map_or_else(|| short_id(id), |n| n.label.clone())
    };
    let nodes = result
        .nodes
        .iter()
        .map(|n| GraphNodeView {
            id: n.id.clone(),
            label: n.label.clone(),
            class_id: n.class_id.clone(),
            provisional: n.provisional,
            properties: match &n.properties {
                serde_json::Value::Object(map) if !map.is_empty() => map
                    .iter()
                    .map(|(k, v)| format!("{k}: {}", display_json(v)))
                    .collect::<Vec<_>>()
                    .join(" · "),
                _ => String::new(),
            },
            sources: sources_of(&n.id),
        })
        .collect();
    let edges = result
        .edges
        .iter()
        .map(|e| GraphEdgeView {
            source: label_of(&e.source_node_id),
            relation: e.relation_id.clone(),
            target: label_of(&e.target_node_id),
            sources: sources_of(&e.id),
        })
        .collect();
    Ok(GraphResultView {
        title,
        json: serde_json::to_string(result)?,
        nodes,
        edges,
    })
}

fn display_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}

#[derive(Deserialize)]
struct ExtractForm {
    #[serde(default, deserialize_with = "blank_as_none")]
    source: Option<ExtractSource>,
    sample: Option<String>,
    #[serde(default)]
    reset: bool,
}

async fn graph_extract(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
    Form(form): Form<ExtractForm>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let sample = form
        .sample
        .as_deref()
        .and_then(|s| s.trim().parse::<u32>().ok());
    let outcome = graph_api::start_extraction(
        &app,
        &access,
        &id,
        &graph_api::ExtractionPlan {
            source: form.source.unwrap_or_default(),
            sample,
            reset: form.reset,
        },
    )
    .await;
    let target = match outcome {
        Ok(body) if body.get("status").and_then(|s| s.as_str()) == Some("running") => format!(
            "/w/{id}/graph?notice={}",
            urlencoded(
                "document extraction started in the background; this page shows the graph as it grows"
            )
        ),
        Ok(_) => format!("/w/{id}/graph"),
        Err(e) => format!("/w/{id}/graph?error={}", urlencoded(&e.message)),
    };
    Ok(Redirect::to(&target).into_response())
}

async fn graph_revalidate(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let db = app.workspace_db(&id).await?;
    let outcome = with_db(db, graph_store::revalidate).await;
    let target = match outcome {
        Ok(r) => {
            access
                .audit(
                    &app,
                    "graph_revalidate",
                    None,
                    Outcome::Allowed,
                    Some(serde_json::json!({
                        "dropped_nodes": r.dropped_nodes,
                        "dropped_edges": r.dropped_edges,
                        "version": r.version,
                    })),
                )
                .await?;
            format!(
                "/w/{id}/graph?notice={}",
                urlencoded(&format!(
                    "dropped {} nodes and {} edges",
                    r.dropped_nodes, r.dropped_edges
                ))
            )
        }
        Err(e) => format!("/w/{id}/graph?error={}", urlencoded(&e.message)),
    };
    Ok(Redirect::to(&target).into_response())
}

async fn graph_review(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let db = app.workspace_db(&id).await?;
    with_db(db, graph_store::mark_reviewed).await?;
    access
        .audit(&app, "graph_review", None, Outcome::Allowed, None)
        .await?;
    Ok(Redirect::to(&format!("/w/{id}/graph")).into_response())
}

#[derive(Deserialize)]
struct MergeForm {
    action: String,
}

async fn graph_merge_decide(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, mid)): Path<(String, String)>,
    Form(form): Form<MergeForm>,
) -> WebResult<Response> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    // Parsed rather than extracted, so a bad value comes back as a notice
    // on the page instead of an error page.
    let decision = match form.action.parse::<resolve::MergeDecision>() {
        Ok(decision) => decision,
        Err(e) => {
            let target = format!("/w/{id}/graph?error={}", urlencoded(&e.to_string()));
            return Ok(Redirect::to(&target).into_response());
        }
    };
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let merge_id = mid.clone();
    let outcome = with_db(db, move |db| {
        resolve::decide(db, &merge_id, decision, Some(&author))
    })
    .await;
    let target = match outcome {
        Ok(proposal) => {
            access
                .audit(
                    &app,
                    "graph_merge",
                    Some(("graph_merge", &mid)),
                    Outcome::Allowed,
                    Some(serde_json::json!({ "accept": decision == resolve::MergeDecision::Accept, "keep": proposal.keep.label, "drop": proposal.drop.label })),
                )
                .await?;
            format!("/w/{id}/graph")
        }
        Err(e) => format!("/w/{id}/graph?error={}", urlencoded(&e.message)),
    };
    Ok(Redirect::to(&target).into_response())
}
