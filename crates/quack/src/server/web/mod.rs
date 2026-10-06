//! The web UI: server-rendered askama pages over the same access checks as
//! the API, htmx for the document list and SQL grid, and a small script for
//! the streamed chat (design doc 11.1). Everything a page does, the API can
//! do; the handlers here only shape the response as HTML.

pub(crate) mod flash;
pub(crate) mod markdown;
mod sign_in;

use std::collections::BTreeSet;
use std::fmt;
use std::num::NonZeroUsize;

use askama::Template;
use axum::extract::{FromRequestParts, Multipart, Path, Query, State};
use axum::http::request::Parts;
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Form, Router};
use axum_extra::extract::CookieJar;
use jiff::Timestamp;
use jiff::civil::DateTime;
use jiff::tz::TimeZone;
// Multi-valued fields (checkboxes) need serde_html_form, which axum's own
// Form extractor does not use.
use axum_extra::extract::Form as MultiForm;
use quack_core::analysis::events::ToolStep;
use quack_core::ids::{
    CandidateId, ClassId, DocumentId, EdgeId, NodeId, RelationId, SessionId, UserId, WorkspaceId,
};
use quack_core::ontology::candidates::{CandidateAction, Queue};
use quack_core::ontology::induction::{ItemKind, Proposal};
use quack_core::ontology::{
    Ontology, OntologyDiff, OntologyVersion, candidates, store as ontology_store,
};
use quack_core::storage::context;
use quack_core::storage::control::{
    AuditAction, AuditFilter, AuditPage, AuditRow, Expiry, IssuedToken, MemberRow, Membership,
    Outcome, ProviderAllowList, ResourceKind, Role, Scope, Standing, TokenRow, UserKind, UserRow,
    WorkspaceChanges, WorkspaceTimes,
};
use quack_core::storage::sessions::{self, MessageRole, MessageRow, SessionRow, Sharing};
use quack_core::storage::workspace::{
    ChunkSearchResult, DocumentInfo, DocumentSource, DocumentStatus, ExportFormat, Pinning,
    ResultSort, SortDirection, TableDescription,
};
use rust_embed::Embed;
use serde::Deserialize;

use self::flash::{Flash, Flashed};
use super::api::admin::CreateUser;
use super::api::auth::LoginRequest;
use super::api::context::ReplaceContext;
use super::api::documents::{Enqueued, IncomingFile, UploadForm};
use super::api::embeddings::RefreshStarted;
use super::api::graph::DropApproval;
use super::api::graph::ExtractionStarted;
use super::api::import::ImportBody;
use super::api::members::AddMember;
use super::api::ontology::{DecideRequest, RenameRequest};
use super::api::workspaces::CreateWorkspace;
use super::api::{
    documents as docs_api, graph as graph_api, import as import_api, workspaces as workspaces_api,
};
use super::auth::{Access, Identity, Need, Peer, RequestId, SessionCookie, password_login};
use super::error::ApiError;
use super::oidc::Oidc;
use super::state::{App, ServeMode};
use crate::graph_cli;
use quack_core::analysis::tools::{FindPathArgs, NonBlank, SearchGraphArgs};
use quack_core::config::{GraphConfig, OidcConfig};
use quack_core::embedding::Vector;
use quack_core::error::{Error as CoreError, Result as CoreResult};
use quack_core::graph::query::{GraphQuery, PathEnds, PathQuery};
use quack_core::graph::resolve::MergeDecision;
use quack_core::graph::store::{Keep, NodeEdit, Revalidation};
use quack_core::graph::traverse::Hops;
use quack_core::graph::{
    ExtractSource, GraphResult, GraphStatus, Origin, Properties, Standing as GraphStanding,
    resolve, store as graph_store,
};
use quack_core::import::ImportRequest;
use quack_core::jobs::JobNumber;
use quack_core::llm::Embeddings;
use quack_core::ontology::ROOT_CLASS;
use quack_core::storage::workspace::WorkspaceDb;
use quack_core::text::blank_as_none;

#[derive(Embed)]
#[folder = "static/"]
struct Assets;

/// The identity for a page: an unauthenticated browser goes to the login
/// form instead of getting a JSON 401.
pub(crate) struct WebUser(pub Identity);

impl FromRequestParts<App> for WebUser {
    type Rejection = Redirect;

    async fn from_request_parts(parts: &mut Parts, state: &App) -> Result<Self, Self::Rejection> {
        Identity::from_request_parts(parts, state)
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
    /// The header link to mark as current.
    tab: Tab,
    username: String,
    kind: UserKind,
    local: bool,
    workspace: Option<WsNav>,
}

/// A header link: the workspace tabs, the admin pages, then About.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Workspaces,
    Chat,
    Documents,
    Tables,
    Sql,
    Context,
    Ontology,
    Graph,
    Jobs,
    Settings,
    Users,
    Audit,
    About,
}

impl Tab {
    /// The workspace tabs, in header order.
    const WORKSPACE: [Self; 9] = [
        Self::Chat,
        Self::Documents,
        Self::Tables,
        Self::Sql,
        Self::Context,
        Self::Ontology,
        Self::Graph,
        Self::Jobs,
        Self::Settings,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::Workspaces => "Workspaces",
            Self::Chat => "Chat",
            Self::Documents => "Documents",
            Self::Tables => "Tables",
            Self::Sql => "SQL",
            Self::Context => "Context",
            Self::Ontology => "Ontology",
            Self::Graph => "Graph",
            Self::Jobs => "Jobs",
            Self::Settings => "Settings",
            Self::Users => "Users",
            Self::Audit => "Audit",
            Self::About => "About",
        }
    }

    /// The path under `/w/{id}/` for a workspace tab.
    const fn path(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Documents => "documents",
            Self::Tables => "tables",
            Self::Sql => "sql",
            Self::Context => "context",
            Self::Ontology => "ontology",
            Self::Graph => "graph",
            Self::Jobs => "jobs",
            Self::Settings => "settings",
            Self::Workspaces | Self::Users | Self::Audit | Self::About => "",
        }
    }
}

struct WsNav {
    membership: Membership,
    can_write: bool,
    can_manage: bool,
}

impl Page {
    /// A page outside any workspace.
    fn new(app: &App, identity: &Identity, tab: Tab) -> Self {
        Self {
            title: tab.label().to_owned(),
            tab,
            username: identity.username.clone(),
            kind: identity.kind,
            local: app.mode == ServeMode::Local,
            workspace: None,
        }
    }

    /// A page inside `access`'s workspace, with its navigation.
    fn in_workspace(app: &App, tab: Tab, access: &Access) -> Self {
        Self {
            workspace: Some(WsNav {
                membership: access.membership.clone(),
                can_write: access.permits(Need::WRITE),
                can_manage: access.permits(Need::OWN),
            }),
            ..Self::new(app, &access.identity, tab)
        }
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
    /// The issuer's host, when sign-in through it is configured.
    sign_in: Option<String>,
}

struct WsItem {
    membership: Membership,
    times: Option<WorkspaceTimes>,
}

#[derive(Template)]
#[template(path = "workspaces.html")]
struct WorkspacesPage {
    page: Page,
    workspaces: Vec<WsItem>,
    can_create: bool,
    error: Option<String>,
}

/// A point in time as a page shows it: `app.js` rewrites every
/// `time[data-when]` in the viewer's time zone, and the UTC text stands
/// for a page without scripts.
struct Moment(Timestamp);

impl Moment {
    /// Stored UTC text: an RFC 3339 instant, or a `DuckDB` or `SQLite`
    /// timestamp with no zone, which both hold UTC.
    fn from_utc_text(text: &str) -> Option<Self> {
        if let Ok(at) = text.parse::<Timestamp>() {
            return Some(Self(at));
        }
        let civil: DateTime = text.parse().ok()?;
        civil
            .to_zoned(TimeZone::UTC)
            .ok()
            .map(|z| Self(z.timestamp()))
    }

    fn iso(&self) -> String {
        self.0.to_string()
    }

    fn utc(&self) -> String {
        self.0.strftime("%Y-%m-%d %H:%M UTC").to_string()
    }

    /// How long ago, as of `now`: minutes or hours within a day, else the
    /// date. Written by the server so the page never rewrites it.
    fn ago(&self, now: Timestamp) -> String {
        let minutes = now
            .duration_since(self.0)
            .as_secs()
            .checked_div(60)
            .unwrap_or(0);
        match minutes {
            ..1 => String::from("just now"),
            1..60 => format!("{minutes} min ago"),
            60..1440 => format!("{} h ago", minutes.checked_div(60).unwrap_or(0)),
            _ => self.0.strftime("%Y-%m-%d").to_string(),
        }
    }
}

/// How `app.js` words a time: the time of day (with the date when not
/// today), or how long ago for today and the date before that.
#[derive(Debug, Clone, Copy)]
enum When {
    Clock,
    Relative,
}

impl When {
    const fn attr(self) -> &'static str {
        match self {
            Self::Clock => "clock",
            Self::Relative => "relative",
        }
    }

    /// The `<time>` element for `at`. A relative time is final as written;
    /// `app.js` words a clock time in the reader's zone.
    fn element(self, at: &Moment) -> String {
        let text = match self {
            Self::Clock => at.utc(),
            Self::Relative => at.ago(Timestamp::now()),
        };
        format!(
            "<time datetime=\"{}\" data-when=\"{}\">{text}</time>",
            at.iso(),
            self.attr(),
        )
    }

    /// The `<time>` element for stored UTC text, or the text itself,
    /// escaped, when it is not a time.
    fn html(self, text: &str) -> String {
        Moment::from_utc_text(text).map_or_else(
            || {
                askama::filters::escape(text, askama::filters::Html)
                    .map(|e| e.to_string())
                    .unwrap_or_default()
            },
            |at| self.element(&at),
        )
    }
}

struct MessageView {
    role: String,
    /// When the question was asked or the answer finished.
    at: Option<Moment>,
    /// How long the answer took; `None` for questions and older answers.
    duration_ms: Option<u64>,
    /// Rendered HTML for assistant answers; escaped text for user messages.
    content_html: String,
    steps: Vec<StepView>,
    citations: Vec<CitationView>,
    chart_json: Option<String>,
    /// One JSON `GraphResult` per graph tool call the turn made.
    graphs: Vec<String>,
}

/// A step as the chat page lists it: the rows it kept, as text cells.
struct StepView {
    tool: String,
    detail: String,
    summary: String,
    duration_ms: u64,
    rows: Option<u64>,
    columns: Vec<String>,
    cells: Vec<Vec<String>>,
}

impl From<&ToolStep> for StepView {
    fn from(step: &ToolStep) -> Self {
        let (columns, cells) = step.result.as_ref().map_or((Vec::new(), Vec::new()), |r| {
            (
                r.columns.clone(),
                r.rows
                    .iter()
                    .map(|row| row.iter().map(|v| JsonText(v).to_string()).collect())
                    .collect(),
            )
        });
        Self {
            tool: step.tool.to_string(),
            detail: step.detail.clone(),
            summary: step.summary.clone(),
            duration_ms: step.duration_ms,
            rows: step.rows,
            columns,
            cells,
        }
    }
}

struct CitationView {
    n: u64,
    label: String,
    document_id: DocumentId,
    chunk_index: u32,
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
    /// Replaced documents are listed too (`?all=true`).
    show_all: bool,
}

/// One chunk of a document, where a citation link lands.
#[derive(Template)]
#[template(path = "passage.html")]
struct PassagePage {
    page: Page,
    document: DocumentInfo,
    chunk: ChunkSearchResult,
    /// Chunks the document holds, when recorded.
    total: Option<i64>,
    /// Positions of the chunks before and after, when they exist.
    previous: Option<u32>,
    next: Option<u32>,
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
    number: JobNumber,
    kind: String,
    label: String,
    /// `queued`, `running`, `cancelling`, or a final state.
    state: String,
    active: bool,
    can_cancel: bool,
    progress: String,
    outcome: Option<String>,
    queued_at: Moment,
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

/// What the Documents page polls while something processes: each row's
/// status and the note, nothing else, so the rest of the table holds still.
#[derive(Template)]
#[template(path = "documents_status.html")]
struct DocumentStatuses {
    documents: Vec<DocumentInfo>,
    pending: bool,
}

struct TableView {
    name: String,
    columns: Vec<(String, String)>,
    sample_columns: Vec<String>,
    sample_rows: Vec<Vec<String>>,
}

impl TableView {
    fn of(described: TableDescription) -> Self {
        Self {
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
                .map(|r| r.iter().map(|v| JsonText(v).to_string()).collect())
                .collect(),
        }
    }
}

#[derive(Template)]
#[template(path = "tables.html")]
struct TablesPage {
    page: Page,
    tables: Vec<String>,
    selected: Option<TableView>,
    error: Option<String>,
}

impl TablesPage {
    /// The table list, with `open` described beside it when one is asked
    /// for (a click, or an import that just made it).
    async fn render(
        app: &App,
        identity: Identity,
        id: &WorkspaceId,
        open: Option<String>,
        error: Option<String>,
    ) -> WebResult<Response> {
        let access = Access::resolve(app, identity, id, Need::READ).await?;
        let selected = if let Some(name) = open {
            Some(TableView::of(access.describe_table(app, &name).await?))
        } else {
            access.audit_read(app, AuditAction::Page, "tables").await?;
            None
        };
        let list = app.read(id, WorkspaceDb::list_tables).await?;
        let mut page = Page::in_workspace(app, Tab::Tables, &access);
        if let Some(table) = &selected {
            page.title.clone_from(&table.name);
        }
        html(&Self {
            page,
            tables: list,
            selected,
            error,
        })
    }
}

#[derive(Template)]
#[template(path = "sql.html")]
struct SqlPage {
    page: Page,
    sql: String,
    result: String,
    /// The editor is rendered in place here, never swapped in.
    editor_swap: bool,
}

/// A SQL page run: the statement as typed, and the column sort a header
/// click asked for. With no sort, the rows come back in the statement's own
/// order; nothing is imposed.
#[derive(Deserialize)]
struct SqlRun {
    sql: String,
    /// The 1-based column position to sort by.
    sort: Option<NonZeroUsize>,
    dir: Option<SortDirection>,
}

impl SqlRun {
    fn result_sort(&self) -> Option<ResultSort> {
        self.sort.map(|column| ResultSort {
            column,
            direction: self.dir.unwrap_or(SortDirection::Asc),
        })
    }

    /// The statement to run and whether its results can be sorted. A sort
    /// rewrites the statement's own `ORDER BY` (replacing any it had), and
    /// the rewritten SQL is what runs and what the editor then shows; a
    /// statement that is not one `SELECT` runs as typed.
    async fn statement(&self, app: &App, access: &Access) -> Statement {
        let (sql, sort) = (self.sql.clone(), self.result_sort());
        let planned = app
            .read(&access.membership.workspace.id, move |db| {
                let Some(sortable) = db.sortable(&sql)? else {
                    return Ok(None);
                };
                sort.map(|sort| db.sort_statement(&sortable, sort))
                    .transpose()
                    .map(Some)
            })
            .await;
        let as_typed = |sortable| Statement {
            sql: self.sql.clone(),
            sortable,
            rewritten: false,
        };
        match planned {
            Ok(Some(Some(rewritten))) => Statement {
                sql: rewritten,
                sortable: true,
                rewritten: true,
            },
            Ok(Some(None)) => as_typed(true),
            // Not sortable, or the rewrite failed: run it as typed, and
            // let the run report whatever is wrong with it.
            Ok(None) | Err(_) => as_typed(false),
        }
    }
}

/// What a SQL page run executes.
struct Statement {
    sql: String,
    sortable: bool,
    /// A header's sort rewrote the SQL; the editor takes the new text.
    rewritten: bool,
}

/// A result column's header: a button that sorts by it, ascending first,
/// then the other way.
struct SortHeader {
    name: String,
    position: NonZeroUsize,
    sorted: Option<SortDirection>,
}

impl SortHeader {
    const fn param(direction: SortDirection) -> &'static str {
        match direction {
            SortDirection::Asc => "asc",
            SortDirection::Desc => "desc",
        }
    }

    /// The direction a click asks for.
    fn next(&self) -> &'static str {
        Self::param(
            self.sorted
                .map_or(SortDirection::Asc, SortDirection::flipped),
        )
    }

    fn aria_sort(&self) -> &'static str {
        match self.sorted {
            Some(SortDirection::Asc) => "ascending",
            Some(SortDirection::Desc) => "descending",
            None => "none",
        }
    }

    fn arrow(&self) -> &'static str {
        match self.sorted {
            Some(SortDirection::Asc) => "▲",
            Some(SortDirection::Desc) => "▼",
            None => "",
        }
    }
}

#[derive(Template)]
#[template(path = "sql_result.html")]
struct SqlResult {
    ws_id: String,
    /// The statement that ran, which a header click sends back with its
    /// sort.
    sql: String,
    sortable: bool,
    /// Swap the rewritten statement into the editor.
    editor_swap: bool,
    headers: Vec<SortHeader>,
    rows: Vec<Vec<String>>,
    row_count: usize,
    truncated: bool,
    duration_ms: u64,
    error: Option<String>,
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
    id: ClassId,
    key: Option<String>,
    properties: String,
}

struct CandidateView {
    id: CandidateId,
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
    id: NodeId,
    label: String,
    class_id: ClassId,
    standing: GraphStanding,
    properties: String,
    sources: String,
}

struct GraphEdgeView {
    id: EdgeId,
    source: String,
    relation: RelationId,
    target: String,
    sources: String,
}

struct GraphResultView {
    title: String,
    json: String,
    nodes: Vec<GraphNodeView>,
    edges: Vec<GraphEdgeView>,
}

/// The graph page's search and path query, as its forms show it.
#[derive(Clone, Default)]
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
    /// What revalidating a stale graph would drop, or why that could not
    /// be counted.
    revalidation: Option<Result<Revalidation, String>>,
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
    /// This page follows an earlier one.
    continued: bool,
    next_cursor: Option<String>,
    action: String,
    outcome: Option<Outcome>,
    outcomes: &'static [Outcome],
    workspace_id: String,
}

#[derive(Template)]
#[template(path = "about.html")]
struct AboutPage {
    page: Page,
    version: &'static str,
}

// --- routes ------------------------------------------------------------------

/// The embedded stylesheet, scripts, and icon. They sit outside the rate
/// limiter: every page load fetches four of them, so counting them spent a
/// person's request budget four times faster than their clicks did, and a
/// refused stylesheet left the page unstyled.
pub(crate) fn assets() -> Router<App> {
    Router::new().route("/static/{*path}", get(static_asset))
}

pub(crate) fn router() -> Router<App> {
    Router::new()
        .route("/", get(index))
        .route(
            "/login",
            get(login_page).merge(super::throttled_login(post(login_submit))),
        )
        .route("/logout", post(logout))
        .route(
            OidcConfig::START_PATH,
            super::throttled_login(get(sign_in::begin)),
        )
        .route(
            OidcConfig::CALLBACK_PATH,
            super::throttled_login(get(sign_in::finish)),
        )
        .route("/workspaces", get(workspaces).post(create_workspace))
        .route("/w/{id}", get(workspace_index))
        .route("/w/{id}/chat", get(chat))
        .route("/w/{id}/chat/{sid}/delete", post(delete_session))
        .route("/w/{id}/chat/{sid}/share", post(share_session))
        .route("/w/{id}/chat/{sid}/unshare", post(unshare_session))
        .route("/w/{id}/documents", get(documents).post(upload))
        .route("/w/{id}/documents/rows", get(document_rows))
        .route("/w/{id}/documents/status", get(document_status))
        .route("/w/{id}/documents/{doc}/chunks/{n}", get(passage))
        .route("/w/{id}/documents/{doc}/pin", post(pin))
        .route("/w/{id}/documents/{doc}/unpin", post(unpin))
        .route("/w/{id}/documents/{doc}/delete", post(delete_doc))
        .route("/w/{id}/documents/{doc}/replace", post(replace_doc))
        .route("/w/{id}/embeddings/refresh", post(refresh_embeddings))
        .route("/w/{id}/jobs", get(jobs_page))
        .route("/w/{id}/jobs/rows", get(job_rows))
        .route("/w/{id}/jobs/{job}/cancel", post(job_cancel))
        .route("/w/{id}/tables", get(tables).post(table))
        .route("/w/{id}/import", post(import_submit))
        .route("/w/{id}/sql", get(sql_page).post(sql_run))
        .route("/w/{id}/sql.csv", post(sql_csv))
        .route("/w/{id}/context", get(context_page).post(context_save))
        .route("/w/{id}/ontology", get(ontology_page).post(ontology_import))
        .route("/w/{id}/ontology/init", post(ontology_init))
        .route("/w/{id}/ontology/rename", post(ontology_rename))
        .route("/w/{id}/ontology/propose", post(ontology_propose))
        .route("/w/{id}/ontology/candidates", post(ontology_decide_many))
        .route("/w/{id}/ontology/candidates/{cid}", post(ontology_decide))
        .route("/w/{id}/ontology/{v}/restore", post(ontology_restore))
        .route("/w/{id}/graph", get(graph_page).post(graph_search))
        .route("/w/{id}/graph/extract", post(graph_extract))
        .route("/w/{id}/graph/revalidate", post(graph_revalidate))
        .route("/w/{id}/graph/review", post(graph_review))
        .route("/w/{id}/graph/merges/{mid}", post(graph_merge_decide))
        .route("/w/{id}/graph/nodes", post(graph_node_add))
        .route("/w/{id}/graph/nodes/{nid}", post(graph_node_edit))
        .route("/w/{id}/graph/nodes/{nid}/delete", post(graph_node_delete))
        .route("/w/{id}/graph/edges", post(graph_edge_add))
        .route("/w/{id}/graph/edges/{eid}/delete", post(graph_edge_delete))
        .route("/w/{id}/settings", get(settings).post(settings_save))
        .route("/w/{id}/members", post(member_add))
        .route("/w/{id}/members/{user}/remove", post(member_remove))
        .route("/w/{id}/tokens", post(token_create))
        .route("/w/{id}/tokens/{hash}/revoke", post(token_revoke))
        .route("/admin/users", get(admin_users).post(admin_user_add))
        .route("/admin/audit", get(admin_audit))
        .route("/about", get(about))
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

async fn login_page(State(app): State<App>, flash: Flashed) -> WebResult<Response> {
    if app.mode == ServeMode::Local {
        return Ok(Redirect::to("/workspaces").into_response());
    }
    html(&LoginPage {
        error: flash.error(),
        sign_in: app.oidc.as_ref().map(Oidc::issuer_host),
    })
}

async fn login_submit(
    State(app): State<App>,
    peer: Peer,
    jar: CookieJar,
    request_id: RequestId,
    Form(form): Form<LoginRequest>,
) -> WebResult<Response> {
    if app.mode == ServeMode::Local {
        return Ok(Redirect::to("/workspaces").into_response());
    }
    // A wrong password is the form again with a message, not a 401; any
    // other failure is still an error page.
    let token = match password_login(&app, peer, request_id, &form.username, &form.password).await {
        Ok(login) => login.token,
        Err(e) if e.status == StatusCode::UNAUTHORIZED => {
            return Ok(Flash::error("/login", "wrong username or password").into_response());
        }
        Err(e) => return Err(e.into()),
    };
    Ok((
        jar.add(SessionCookie::issue(&app, peer, token)),
        Redirect::to("/workspaces"),
    )
        .into_response())
}

async fn logout(
    State(app): State<App>,
    WebUser(identity): WebUser,
    jar: CookieJar,
) -> WebResult<Response> {
    let jar = identity.log_out(&app, jar).await?;
    Ok((jar, Redirect::to("/login")).into_response())
}

async fn workspaces(
    State(app): State<App>,
    WebUser(identity): WebUser,
    flash: Flashed,
) -> WebResult<Response> {
    let mut times = app.control.workspace_times().await?;
    let items: Vec<WsItem> = if app.mode == ServeMode::Local || identity.kind == UserKind::Admin {
        let mine = if app.mode == ServeMode::Local {
            Vec::new()
        } else {
            app.control.workspaces_for_user(&identity.user_id).await?
        };
        app.control
            .list_workspaces()
            .await?
            .into_iter()
            .map(|w| WsItem {
                times: times.remove(&w.id),
                membership: Membership {
                    standing: if app.mode == ServeMode::Local {
                        Standing::Member(Role::Owner)
                    } else {
                        mine.iter()
                            .find(|m| m.workspace.id == w.id)
                            .map_or(Standing::Admin, |m| m.standing)
                    },
                    workspace: w,
                },
            })
            .collect()
    } else {
        app.control
            .workspaces_for_user(&identity.user_id)
            .await?
            .into_iter()
            .map(|membership| WsItem {
                times: times.remove(&membership.workspace.id),
                membership,
            })
            .collect()
    };
    html(&WorkspacesPage {
        can_create: identity.kind == UserKind::Admin,
        page: Page::new(&app, &identity, Tab::Workspaces),
        workspaces: items,
        error: flash.error(),
    })
}

async fn create_workspace(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Form(form): Form<CreateWorkspace>,
) -> WebResult<Response> {
    Ok(match identity.create_workspace(&app, &form.name).await {
        Ok(ws) => Redirect::to(&format!("/w/{}/chat", ws.id)).into_response(),
        Err(e) => Flash::error("/workspaces", e.message).into_response(),
    })
}

async fn workspace_index(Path(id): Path<WorkspaceId>) -> Redirect {
    Redirect::to(&format!("/w/{id}/chat"))
}

#[derive(Deserialize)]
struct ChatQuery {
    session: Option<String>,
}

impl MessageView {
    /// The chat page's messages: user questions, and each answer with the
    /// tool steps recorded before it folded in.
    fn transcript(rows: &[MessageRow]) -> Vec<Self> {
        let mut out = Vec::new();
        let mut steps = Vec::new();
        for row in rows {
            match row.role {
                MessageRole::Tool => steps.extend(row.tool().map(|m| m.step(row.content.clone()))),
                MessageRole::User => out.push(Self::question(row)),
                MessageRole::Assistant => out.push(Self::answer(row, &std::mem::take(&mut steps))),
            }
        }
        out
    }

    fn question(row: &MessageRow) -> Self {
        Self {
            role: String::from("user"),
            at: Moment::from_utc_text(&row.created_at),
            duration_ms: None,
            content_html: askama::filters::escape(&row.content, askama::filters::Html)
                .map(|e| e.to_string())
                .unwrap_or_default(),
            steps: Vec::new(),
            citations: Vec::new(),
            chart_json: None,
            graphs: Vec::new(),
        }
    }

    fn answer(row: &MessageRow, steps: &[ToolStep]) -> Self {
        let meta = row.assistant().cloned().unwrap_or_default();
        Self {
            role: String::from("assistant"),
            at: Moment::from_utc_text(&row.created_at),
            duration_ms: meta.duration_ms,
            content_html: markdown::to_html(&row.content),
            steps: steps.iter().map(StepView::from).collect(),
            citations: meta
                .citations
                .iter()
                .map(|c| CitationView {
                    n: u64::from(c.n),
                    label: c.label(),
                    document_id: c.document_id.clone(),
                    chunk_index: c.chunk_index,
                })
                .collect(),
            chart_json: meta
                .chart
                .as_ref()
                .and_then(|c| serde_json::to_string(c).ok()),
            graphs: meta
                .graph
                .iter()
                .filter_map(|g| serde_json::to_string(g).ok())
                .collect(),
        }
    }
}

async fn chat(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Query(q): Query<ChatQuery>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let viewer = access.session_viewer();
    let wanted = q.session.clone().map(SessionId::from);
    let (sessions_list, current, messages) = app
        .read(&id, move |db| {
            let list = sessions::list_sessions_for(db, 50, &viewer)?;
            let current = match wanted {
                Some(id) => sessions::get_session(db, &id)?.filter(|s| s.visible_to(&viewer)),
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
                AuditAction::SessionRead,
                Some(ResourceKind::Session.id(&current.id)),
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
        page: Page::in_workspace(&app, Tab::Chat, &access),
        sessions: sessions_list,
        current,
        messages: MessageView::transcript(&messages),
        tables,
        documents,
    })
}

async fn delete_session(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, sid)): Path<(WorkspaceId, SessionId)>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access.delete_session(&app, &sid).await?;
    Ok(Redirect::to(&format!("/w/{id}/chat")).into_response())
}

async fn share_session(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, sid)): Path<(WorkspaceId, SessionId)>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .set_session_sharing(&app, &sid, Sharing::Shared)
        .await?;
    Ok(Redirect::to(&format!("/w/{id}/chat?session={sid}")).into_response())
}

async fn unshare_session(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, sid)): Path<(WorkspaceId, SessionId)>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .set_session_sharing(&app, &sid, Sharing::Private)
        .await?;
    Ok(Redirect::to(&format!("/w/{id}/chat?session={sid}")).into_response())
}

/// Which documents the Documents page lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shown {
    /// What the workspace holds now.
    Live,
    /// Replaced documents too, each naming its replacement.
    All,
}

/// `?all=true` on the Documents page.
#[derive(Debug, Default, Deserialize)]
struct DocumentsQuery {
    #[serde(default)]
    all: bool,
}

impl DocumentsQuery {
    const fn shown(&self) -> Shown {
        if self.all { Shown::All } else { Shown::Live }
    }
}

impl DocumentRows {
    /// The workspace's documents as the caller may act on them.
    async fn load(app: &App, access: &Access, shown: Shown) -> WebResult<Self> {
        let documents = app
            .read(
                &access.membership.workspace.id,
                match shown {
                    Shown::Live => WorkspaceDb::list_documents,
                    Shown::All => WorkspaceDb::list_all_documents,
                },
            )
            .await?;
        let pending = documents.iter().any(|d| d.status.is_in_flight());
        Ok(Self {
            ws_id: access.membership.workspace.id.to_string(),
            can_write: access.permits(Need::WRITE),
            documents,
            pending,
        })
    }
}

impl JobRows {
    /// The workspace's jobs as the caller may see and cancel them.
    fn of(app: &App, access: &Access) -> Self {
        let jobs: Vec<JobView> = access
            .visible_jobs(app)
            .into_iter()
            .map(|j| JobView {
                id: j.id.to_string(),
                number: j.number,
                kind: j.kind.to_string(),
                can_cancel: !j.state.is_finished() && access.may_cancel(&j),
                label: j.label,
                state: if j.cancel_requested && !j.state.is_finished() {
                    String::from("cancelling")
                } else {
                    j.state.to_string()
                },
                active: !j.state.is_finished(),
                progress: j.progress.map(|p| p.to_string()).unwrap_or_default(),
                outcome: j.outcome.or(j.status),
                queued_at: Moment(j.queued_at),
            })
            .collect();
        let pending = jobs.iter().any(|j| j.active);
        Self {
            ws_id: access.membership.workspace.id.to_string(),
            jobs,
            pending,
        }
    }
}

async fn jobs_page(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, AuditAction::Page, "jobs").await?;
    let rows = JobRows::of(&app, &access).render()?;
    html(&JobsPage {
        page: Page::in_workspace(&app, Tab::Jobs, &access),
        rows,
    })
}

async fn job_rows(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::Page, "job_rows")
        .await?;
    Ok(Html(JobRows::of(&app, &access).render()?).into_response())
}

async fn job_cancel(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, job)): Path<(WorkspaceId, String)>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let cancelled = access.cancel_job(&app, &job).await?;
    tracing::debug!(job = %cancelled.id, state = %cancelled.state, "cancel requested from the web");
    Ok(Html(JobRows::of(&app, &access).render()?).into_response())
}

async fn documents(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Query(query): Query<DocumentsQuery>,
    flash: Flashed,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::Page, "documents")
        .await?;
    let rows = DocumentRows::load(&app, &access, query.shown())
        .await?
        .render()?;
    let embeddings_note = app.read(&id, WorkspaceDb::embedding_status).await?.note();
    html(&DocumentsPage {
        page: Page::in_workspace(&app, Tab::Documents, &access),
        rows,
        error: flash.error(),
        notice: flash.notice(),
        embeddings_note,
        show_all: query.all,
    })
}

/// The row's Replace control: the one uploaded file takes `doc`'s place
/// once it is ready; `doc` serves until then.
async fn replace_doc(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, doc)): Path<(WorkspaceId, DocumentId)>,
    multipart: Multipart,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let form = UploadForm::read(multipart).await?;
    let back = format!("/w/{id}/documents");
    let queued =
        match docs_api::enqueue(&app, &access, DocumentSource::Upload, form.files, Some(doc)).await
        {
            Ok(queued) => queued,
            Err(e) => return Ok(Flash::error(back, e.message).into_response()),
        };
    Ok(match queued.first() {
        Some(Enqueued::Duplicate { filename, .. }) => Flash::error(
            back,
            format!("{filename} is identical to the document it would replace"),
        ),
        Some(Enqueued::Queued { .. }) | None => Flash::to(back),
    }
    .into_response())
}

/// The Documents page's refresh button: the API's refresh, then back
/// to the page with what it started.
async fn refresh_embeddings(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let started = access.refresh_embeddings(&app).await;
    Ok(
        Flash::after(format!("/w/{id}/documents"), started, |started| {
            Some(String::from(match started {
                RefreshStarted::Running { .. } => {
                    "refreshing embeddings in the background; the Jobs page shows its progress"
                }
                RefreshStarted::Current { .. } => "every vector is already current",
            }))
        })
        .into_response(),
    )
}

async fn document_rows(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::Page, "document_rows")
        .await?;
    Ok(Html(
        DocumentRows::load(&app, &access, Shown::Live)
            .await?
            .render()?,
    )
    .into_response())
}

async fn document_status(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::Page, "document_status")
        .await?;
    let DocumentRows {
        documents, pending, ..
    } = DocumentRows::load(&app, &access, Shown::Live).await?;
    Ok(Html(DocumentStatuses { documents, pending }.render()?).into_response())
}

async fn upload(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    multipart: Multipart,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let form = UploadForm::read(multipart).await?;
    let text = form.fields.get("text").map_or("", String::as_str);
    let pasted = if text.trim().is_empty() {
        Vec::new()
    } else {
        let title = form.fields.get("title").map(String::as_str);
        vec![IncomingFile::pasted(text, title)?]
    };
    let back = format!("/w/{id}/documents");
    let skipped = match enqueue_web(&app, &access, form.files, pasted).await {
        Ok(skipped) => skipped,
        Err(e) => return Ok(Flash::error(back, e.message).into_response()),
    };
    Ok(if skipped.is_empty() {
        Flash::to(back)
    } else {
        Flash::error(
            back,
            format!("Already in the workspace: {}", skipped.join(", ")),
        )
    }
    .into_response())
}

/// Queue uploaded files and pasted text under their own sources; returns
/// the names of files skipped as duplicates.
async fn enqueue_web(
    app: &App,
    access: &Access,
    files: Vec<IncomingFile>,
    pasted: Vec<IncomingFile>,
) -> Result<Vec<String>, ApiError> {
    if files.is_empty() && pasted.is_empty() {
        return Err(ApiError::bad_request("no file or text in the request"));
    }
    let mut queued = Vec::new();
    if !files.is_empty() {
        queued.extend(docs_api::enqueue(app, access, DocumentSource::Upload, files, None).await?);
    }
    if !pasted.is_empty() {
        queued.extend(docs_api::enqueue(app, access, DocumentSource::Paste, pasted, None).await?);
    }
    let mut skipped = Vec::new();
    for entry in queued {
        match entry {
            Enqueued::Duplicate { filename, .. } => skipped.push(filename),
            Enqueued::Queued { .. } => {}
        }
    }
    Ok(skipped)
}

/// The passage a citation links to: chunk `n` of `doc`, with links to
/// its neighbours. A position past the end is a 404 page.
async fn passage(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, doc, n)): Path<(WorkspaceId, DocumentId, u32)>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let docs_api::Chunks { document, chunks } =
        docs_api::read_chunks(&app, &access, &doc, docs_api::ChunkPage::around(n)).await?;
    let at = |position: u32| chunks.iter().find(|c| c.chunk_index == position);
    let Some(chunk) = at(n).cloned() else {
        return Err(ResourceKind::Chunk
            .missing(format!("{doc} chunk {n}"))
            .into());
    };
    let previous = n.checked_sub(1).filter(|p| at(*p).is_some());
    let next = n.checked_add(1).filter(|p| at(*p).is_some());
    html(&PassagePage {
        page: Page::in_workspace(&app, Tab::Documents, &access),
        total: document.chunk_count,
        document,
        chunk,
        previous,
        next,
    })
}

async fn pin(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, doc)): Path<(WorkspaceId, DocumentId)>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    docs_api::set_pinned(&app, &access, &doc, Pinning::Pinned).await?;
    Ok(Html(
        DocumentRows::load(&app, &access, Shown::Live)
            .await?
            .render()?,
    )
    .into_response())
}

async fn unpin(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, doc)): Path<(WorkspaceId, DocumentId)>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    docs_api::set_pinned(&app, &access, &doc, Pinning::Unpinned).await?;
    Ok(Html(
        DocumentRows::load(&app, &access, Shown::Live)
            .await?
            .render()?,
    )
    .into_response())
}

async fn delete_doc(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, doc)): Path<(WorkspaceId, DocumentId)>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    docs_api::delete_document(&app, &access, &doc).await?;
    Ok(Html(
        DocumentRows::load(&app, &access, Shown::Live)
            .await?
            .render()?,
    )
    .into_response())
}

async fn tables(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    flash: Flashed,
) -> WebResult<Response> {
    TablesPage::render(&app, identity, &id, flash.table(), flash.error()).await
}

/// The Tables page's choice: the table to open. Posted, never in the URL:
/// a table's name is workspace content, and a URL ends up in logs.
#[derive(Deserialize)]
struct TableChoice {
    name: String,
}

async fn table(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(choice): Form<TableChoice>,
) -> WebResult<Response> {
    TablesPage::render(&app, identity, &id, Some(choice.name), None).await
}

async fn import_submit(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(form): Form<ImportBody>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let request = ImportRequest::from(form);
    Ok(
        match import_api::run_import(&app, &access, &request).await {
            Ok(imported) => match imported.graph_job {
                Some(job) => Flash::notice(
                    format!("/w/{id}/tables"),
                    format!("imported; graph follow-up queued as job {job}"),
                )
                .opening(imported.summary.table),
                None => Flash::to(format!("/w/{id}/tables")).opening(imported.summary.table),
            },
            Err(e) => Flash::error(format!("/w/{id}/tables"), e.message),
        }
        .into_response(),
    )
}

/// A JSON value shown as text: a string bare, null as nothing, anything
/// else as JSON.
struct JsonText<'a>(&'a serde_json::Value);

impl fmt::Display for JsonText<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            serde_json::Value::Null => Ok(()),
            serde_json::Value::String(s) => f.write_str(s),
            other => write!(f, "{other}"),
        }
    }
}

async fn sql_page(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, AuditAction::Page, "sql").await?;
    html(&SqlPage {
        page: Page::in_workspace(&app, Tab::Sql, &access),
        sql: String::new(),
        result: String::new(),
        editor_swap: false,
    })
}

impl SqlResult {
    /// Run the statement for the caller: its rows, or why it could not run.
    async fn run(app: &App, access: &Access, run: &SqlRun) -> Self {
        let Statement {
            sql,
            sortable,
            rewritten,
        } = run.statement(app, access).await;
        let sort = run.result_sort().filter(|_| rewritten);
        let ws_id = access.membership.workspace.id.to_string();
        match access.execute_sql(app, &sql).await {
            Ok(outcome) => Self {
                ws_id,
                sql,
                sortable,
                editor_swap: rewritten,
                truncated: outcome.truncated,
                headers: outcome
                    .columns
                    .into_iter()
                    .zip((1..).filter_map(NonZeroUsize::new))
                    .map(|(name, position)| SortHeader {
                        name,
                        position,
                        sorted: sort.filter(|s| s.column == position).map(|s| s.direction),
                    })
                    .collect(),
                rows: outcome
                    .rows
                    .iter()
                    .map(|r| r.iter().map(|v| JsonText(v).to_string()).collect())
                    .collect(),
                row_count: outcome.row_count,
                duration_ms: outcome.duration_ms,
                error: None,
            },
            Err(e) => Self {
                ws_id,
                sql,
                sortable,
                editor_swap: false,
                headers: Vec::new(),
                rows: Vec::new(),
                row_count: 0,
                truncated: false,
                duration_ms: 0,
                error: Some(e.message),
            },
        }
    }
}

async fn sql_run(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(form): Form<SqlRun>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    Ok(Html(SqlResult::run(&app, &access, &form).await.render()?).into_response())
}

/// The rows as a CSV download. A POST, so the statement travels in the
/// body: in a URL it would land in request logs, proxies, and browser
/// history, and a long one would not fit. A result the row cap cut says so
/// in its filename; a marker inside the file would break its parsers.
async fn sql_csv(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(q): Form<SqlRun>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let statement = q.statement(&app, &access).await;
    // Every row, streamed: the grid's cap does not apply to the file.
    Ok(access
        .export_sql(&app, statement.sql, ExportFormat::Csv)
        .await?)
}

impl ClassRow {
    /// The ontology's classes depth-first from the root, each with its depth.
    fn tree(ontology: &Ontology) -> Vec<Self> {
        fn walk(ontology: &Ontology, parent: &str, depth: usize, out: &mut Vec<ClassRow>) {
            for class in ontology.classes.iter().filter(|c| c.parent == parent) {
                out.push(ClassRow {
                    depth,
                    id: class.id.clone(),
                    key: class.key.clone(),
                    properties: ontology
                        .class_properties(class.id.as_str())
                        .into_iter()
                        .collect::<Vec<_>>()
                        .join(", "),
                });
                walk(ontology, class.id.as_str(), depth.saturating_add(1), out);
            }
        }
        let mut out = Vec::new();
        walk(ontology, ROOT_CLASS, 0, &mut out);
        out
    }
}

async fn ontology_page(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Query(q): Query<OntologyQuery>,
    flash: Flashed,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::Page, "ontology")
        .await?;
    let queue_status = q.status.unwrap_or_default();
    let (ontology, versions, diff, pending, low_support, has_tables) = app
        .read(&id, |db| {
            let current = ontology_store::current(db)?;
            let versions = ontology_store::versions(db, 20)?;
            let previous = current
                .as_ref()
                .and_then(|c| c.version)
                .and_then(OntologyVersion::previous);
            let diff = match (&current, previous) {
                (Some(c), Some(previous)) => {
                    ontology_store::version(db, previous)?.map(|older| c.diff(&older))
                }
                (Some(_) | None, None) | (None, Some(_)) => None,
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
        .map(CandidateView::from_row)
        .collect();
    let json = match &ontology {
        Some(o) => o.to_json()?,
        None => String::new(),
    };
    html(&OntologyPage {
        page: Page::in_workspace(&app, Tab::Ontology, &access),
        classes: ontology.as_ref().map(ClassRow::tree).unwrap_or_default(),
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
        error: flash.error(),
        notice: flash.notice(),
    })
}

/// Accept or reject every selected candidate at once (issue #55).
async fn ontology_decide_many(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    MultiForm(form): MultiForm<BulkDecideForm>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let back = match form.status {
        Queue::LowSupport => format!("/w/{id}/ontology?status={}", Queue::LowSupport),
        Queue::Pending => format!("/w/{id}/ontology"),
    };
    let (accept, reject) = match form.bulk {
        CandidateAction::Accept => (form.ids, Vec::new()),
        CandidateAction::Reject => (Vec::new(), form.ids),
        CandidateAction::Rename | CandidateAction::MergeInto | CandidateAction::Reparent => {
            return Ok(Flash::error(back, "the bulk action is accept or reject").into_response());
        }
    };
    let decided = access.decide_candidates(&app, accept, reject).await;
    Ok(Flash::after(back, decided, |_| None).into_response())
}

/// A review-queue row: what the candidate is and the evidence for it.
impl CandidateView {
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

    fn from_row(c: candidates::CandidateRow) -> Self {
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
            _ if from_bundle => (Self::bundle_evidence(e), Self::proposal_detail(&c.proposal)),
            _ if from_documents => (
                Self::document_evidence(e),
                Self::proposal_detail(&c.proposal),
            ),
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
        Self {
            id: c.id,
            kind: c.kind,
            proposal_id: c.proposal.id().to_owned(),
            confidence: format!("{:.2}", c.confidence),
            evidence,
            detail,
        }
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
    Path(id): Path<WorkspaceId>,
    Form(form): Form<ProposeForm>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let back = format!("/w/{id}/ontology");
    if form.documents {
        let started = access.start_document_run(&app, None).await;
        return Ok(Flash::after(back, started, |_| {
            Some(String::from(
                "document pass started; candidates appear here when it finishes",
            ))
        })
        .into_response());
    }
    let proposed = access.propose_from_tables(&app, form.auto_accept).await;
    Ok(Flash::after(back, proposed, |proposed| {
        (proposed.candidates == 0)
            .then(|| String::from("nothing to propose: the tables are already covered"))
    })
    .into_response())
}

async fn ontology_decide(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, cid)): Path<(WorkspaceId, String)>,
    Form(form): Form<DecideRequest>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let decided = access
        .decide_candidate(&app, &cid, form.action, form.target.as_deref())
        .await;
    Ok(Flash::after(format!("/w/{id}/ontology"), decided, |_| None).into_response())
}

#[derive(Deserialize)]
struct OntologyForm {
    json: String,
}

async fn ontology_import(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(form): Form<OntologyForm>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let stored = access
        .replace_ontology(&app, &form.json, "edited in the web UI")
        .await;
    Ok(Flash::after(format!("/w/{id}/ontology"), stored, |_| None).into_response())
}

async fn ontology_init(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let stored = access.init_ontology(&app).await;
    Ok(Flash::after(format!("/w/{id}/ontology"), stored, |_| None).into_response())
}

async fn ontology_rename(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(form): Form<RenameRequest>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let stored = access.rename_ontology_id(&app, &form).await;
    Ok(Flash::after(format!("/w/{id}/ontology"), stored, |_| {
        Some(format!(
            "renamed {} {} to {}; the graph's nodes and edges moved with it",
            form.kind, form.from, form.to
        ))
    })
    .into_response())
}

async fn ontology_restore(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, v)): Path<(WorkspaceId, OntologyVersion)>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let stored = access.restore_ontology(&app, v).await;
    Ok(Flash::after(format!("/w/{id}/ontology"), stored, |_| None).into_response())
}

async fn context_page(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::Page, "context")
        .await?;
    let (current, versions) = app
        .read(&id, |db| {
            Ok((context::current(db)?, context::history(db, 20)?))
        })
        .await?;
    html(&ContextPage {
        page: Page::in_workspace(&app, Tab::Context, &access),
        content: current
            .as_ref()
            .map(|c| c.content.clone())
            .unwrap_or_default(),
        current,
        versions,
    })
}

async fn context_save(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(form): Form<ReplaceContext>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    access.save_context(&app, form.content).await?;
    Ok(Flash::to(format!("/w/{id}/context")).into_response())
}

impl SettingsPage {
    /// The settings page: providers, and for owners the members and tokens;
    /// `new_token` is a token just created, shown once.
    async fn load(
        app: &App,
        access: &Access,
        new_token: Option<String>,
        error: Option<String>,
    ) -> WebResult<Self> {
        let allowed = &access.membership.workspace.allowed_providers;
        let providers = app
            .config
            .providers
            .keys()
            .map(|name| (name.to_string(), allowed.permits(name.as_str())))
            .collect();
        let (members, tokens) = if access.permits(Need::OWN) {
            (
                app.control
                    .list_members(&access.membership.workspace.id)
                    .await?,
                app.control
                    .list_tokens(&access.membership.workspace.id)
                    .await?,
            )
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(Self {
            page: Page::in_workspace(app, Tab::Settings, access),
            classification: access.membership.workspace.classification.clone(),
            providers,
            members,
            tokens,
            new_token,
            error,
        })
    }
}

async fn settings(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    flash: Flashed,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ_OR_ADMIN).await?;
    access
        .audit_read(&app, AuditAction::Page, "settings")
        .await?;
    html(&SettingsPage::load(&app, &access, None, flash.error()).await?)
}

#[derive(Deserialize)]
struct SettingsForm {
    classification: String,
    #[serde(default)]
    providers: BTreeSet<String>,
}

async fn settings_save(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    MultiForm(form): MultiForm<SettingsForm>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    // The form is one checkbox per configured provider, so it cannot say
    // "every provider, including ones added later" other than by ticking
    // all of them or none.
    let configured = &app.config.providers;
    let every = form.providers.len() == configured.len()
        && configured
            .keys()
            .all(|name| form.providers.contains(name.as_str()));
    let allowed_providers = if form.providers.is_empty() || every {
        ProviderAllowList::All
    } else {
        ProviderAllowList::Only(form.providers)
    };
    let changes = WorkspaceChanges {
        classification: Some(form.classification),
        allowed_providers,
    };
    let saved = workspaces_api::update_settings(&app, &access, changes).await;
    Ok(Flash::after(format!("/w/{id}/settings"), saved, |_| None).into_response())
}

async fn member_add(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(form): Form<AddMember>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    let added = access.add_member(&app, &form).await;
    Ok(Flash::after(format!("/w/{id}/settings"), added, |_| None).into_response())
}

async fn member_remove(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, user_id)): Path<(WorkspaceId, UserId)>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    let removed = access.remove_member(&app, &user_id).await;
    Ok(Flash::after(format!("/w/{id}/settings"), removed, |()| None).into_response())
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
    Path(id): Path<WorkspaceId>,
    MultiForm(form): MultiForm<TokenForm>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    if app.mode == ServeMode::Local {
        return Ok(
            Flash::error(format!("/w/{id}/settings"), "local mode has no users").into_response(),
        );
    }
    let scopes = form.scopes;
    let expires_at = form
        .expires_days
        .filter(|d| *d > 0)
        .map(Expiry::after_days)
        .transpose()?;
    let entry = access.entry(AuditAction::Token, Outcome::Allowed);
    let IssuedToken { secret, .. } = app
        .control
        .create_token(
            &id,
            &access.identity.user_id,
            form.name.trim(),
            &scopes,
            expires_at,
            entry.clone(),
        )
        .await?;
    access.record_detail(&app, &entry, None).await?;
    // The secret is shown once in this response body, never in a URL where
    // browser history, proxy logs, or a Referer would keep it.
    html(&SettingsPage::load(&app, &access, Some(secret.expose().to_owned()), None).await?)
}

async fn token_revoke(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, hash)): Path<(WorkspaceId, String)>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::OWN).await?;
    let owned = app
        .control
        .list_tokens(&id)
        .await?
        .into_iter()
        .any(|t| t.token_hash == hash);
    if !owned {
        return Err(ApiError::not_found("no such token").into());
    }
    let entry = access.entry(AuditAction::Token, Outcome::Allowed);
    app.control.delete_token(&hash, entry.clone()).await?;
    access.record_detail(&app, &entry, None).await?;
    Ok(Redirect::to(&format!("/w/{id}/settings")).into_response())
}

async fn admin_users(
    State(app): State<App>,
    WebUser(identity): WebUser,
    flash: Flashed,
) -> WebResult<Response> {
    identity.require_admin()?;
    html(&AdminUsersPage {
        page: Page::new(&app, &identity, Tab::Users),
        users: app.control.list_users().await?,
        error: flash.error(),
    })
}

async fn admin_user_add(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Form(form): Form<CreateUser>,
) -> WebResult<Response> {
    let created = identity.create_user(&app, &form).await;
    Ok(Flash::after("/admin/users", created, |_| None).into_response())
}

async fn admin_audit(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Query(filter): Query<AuditFilter>,
) -> WebResult<Response> {
    identity.require_admin()?;
    // The page's own size; the form never sends one.
    let filter = AuditFilter {
        limit: 200,
        ..filter
    };
    let AuditPage { rows, next } = app.control.query_audit(&filter).await?;
    html(&AdminAuditPage {
        page: Page::new(&app, &identity, Tab::Audit),
        rows,
        continued: filter.cursor.is_some(),
        next_cursor: next.map(|c| c.to_string()),
        action: filter.action.unwrap_or_default(),
        outcome: filter.outcome,
        outcomes: Outcome::ALL,
        workspace_id: filter
            .workspace_id
            .map(WorkspaceId::into_string)
            .unwrap_or_default(),
    })
}

/// The running version and the projects quack is built with. It reads no
/// workspace, so it writes no audit row.
async fn about(State(app): State<App>, WebUser(identity): WebUser) -> WebResult<Response> {
    html(&AboutPage {
        page: Page::new(&app, &identity, Tab::About),
        version: env!("CARGO_PKG_VERSION"),
    })
}

#[cfg(test)]
mod tests;

// --- graph ----------------------------------------------------------------------

/// The graph page's explore or path form. Posted, never a query string:
/// entity names are workspace content, and a URL ends up in logs.
#[derive(Deserialize, Default)]
#[serde(default)]
struct GraphSearch {
    entity: NonBlank,
    class: NonBlank,
    relation: NonBlank,
    hops: Option<u32>,
    from: NonBlank,
    to: NonBlank,
    max_hops: Option<u32>,
}

impl GraphSearch {
    /// The search half of the form, as the graph takes it.
    fn search(&self) -> SearchGraphArgs {
        SearchGraphArgs {
            entity: self.entity.clone(),
            class: self.class.clone(),
            relation: self.relation.clone(),
            hops: self.hops,
        }
    }

    /// The path half of the form, as the graph takes it; a blank end is
    /// an empty string, which the query refuses.
    fn path(&self) -> FindPathArgs {
        FindPathArgs {
            from: self.from.get().unwrap_or_default().to_owned(),
            to: self.to.get().unwrap_or_default().to_owned(),
            max_hops: self.max_hops,
        }
    }
}

impl GraphQueryView {
    /// The page's query: blank fields are empty strings, which the form
    /// shows as they are; hops at their defaults when not given.
    fn from_query(q: &GraphSearch) -> Self {
        let given = |value: &NonBlank| value.get().unwrap_or_default().to_owned();
        Self {
            entity: given(&q.entity),
            class: given(&q.class),
            relation: given(&q.relation),
            hops: Hops::neighborhood(q.hops).get(),
            from: given(&q.from),
            to: given(&q.to),
            max_hops: Hops::path(q.max_hops).get(),
        }
    }
}

async fn graph_page(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    flash: Flashed,
) -> WebResult<Response> {
    render_graph(&app, identity, &id, &GraphSearch::default(), &flash).await
}

async fn graph_search(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    flash: Flashed,
    Form(q): Form<GraphSearch>,
) -> WebResult<Response> {
    render_graph(&app, identity, &id, &q, &flash).await
}

/// The graph page, with `q`'s search or path result when it asks for one.
async fn render_graph(
    app: &App,
    identity: Identity,
    id: &WorkspaceId,
    q: &GraphSearch,
    flash: &Flashed,
) -> WebResult<Response> {
    let access = Access::resolve(app, identity, id, Need::READ).await?;
    access.audit_read(app, AuditAction::Page, "graph").await?;
    let options = app.config.graph;
    let query = GraphQueryView::from_query(q);
    let ask = GraphAsk::of(q, app, &access).await?;
    let data = app
        .read(id, move |db| GraphPageData::read(db, &ask, &options))
        .await?;
    // An unknown class or entity is shown on the page, not as a failed page.
    let mut error = flash.error();
    let result = match data.result {
        Some(GraphAnswer {
            title,
            result: Ok(found),
        }) => Some(GraphResultView::of(title, &found)?),
        Some(GraphAnswer { result: Err(e), .. }) => {
            error = Some(e.to_string());
            None
        }
        None => None,
    };
    let status = data.status;
    let mut drift: Vec<String> = status
        .drift
        .classes
        .names()
        .map(|c| format!("class {c}"))
        .chain(
            status
                .drift
                .relations
                .names()
                .map(|r| format!("relation {r}")),
        )
        .collect();
    drift.sort();
    html(&GraphPage {
        page: Page::in_workspace(app, Tab::Graph, &access),
        status,
        drift,
        has_ontology: data.has_ontology,
        revalidation: data
            .revalidation
            .map(|preview| preview.map_err(|e| e.to_string())),
        merges: data.merges,
        query,
        result,
        error,
        notice: flash.notice(),
    })
}

/// What the graph page was asked, with the embeddings its entry points
/// need.
enum GraphAsk {
    Nothing,
    Search(GraphQuery, Option<Vector>),
    Path(PathQuery, PathEnds),
}

impl GraphAsk {
    /// A path when both ends are given, else a search when an entity or a
    /// class is.
    async fn of(q: &GraphSearch, app: &App, access: &Access) -> WebResult<Self> {
        let path = q.path().query();
        let search = q.search().query();
        if path.is_err() && search.is_err() {
            return Ok(Self::Nothing);
        }
        let model = access
            .model(
                app,
                AuditAction::Graph,
                Embeddings::from_config(&app.config).await,
            )
            .await?;
        Ok(match (path, search) {
            (Ok(path), _) => {
                let ends = path.embeddings(model.as_ref()).await?;
                Self::Path(path, ends)
            }
            (Err(_), Ok(search)) => {
                let embedding = search.embedding(model.as_ref()).await?;
                Self::Search(search, embedding)
            }
            (Err(_), Err(_)) => Self::Nothing,
        })
    }

    /// Its title and result, or why it could not run.
    fn run(&self, db: &WorkspaceDb, options: &GraphConfig) -> Option<GraphAnswer> {
        match self {
            Self::Nothing => None,
            Self::Path(path, ends) => Some(GraphAnswer {
                title: format!("Path from {} to {}", path.from, path.to),
                result: path.run(db, ends, options),
            }),
            Self::Search(search, embedding) => {
                let title = match (&search.entity, &search.class) {
                    (Some(entity), _) => format!("Around {entity}"),
                    (None, Some(class)) => format!("Entities of class {class}"),
                    (None, None) => String::new(),
                };
                Some(GraphAnswer {
                    title,
                    result: search.run(db, embedding.as_ref(), options),
                })
            }
        }
    }
}

/// What the graph page reads from the workspace in one go.
struct GraphPageData {
    status: GraphStatus,
    has_ontology: bool,
    /// What revalidating would drop, or why that could not be counted;
    /// read only while the graph is stale.
    revalidation: Option<CoreResult<Revalidation>>,
    merges: Vec<resolve::MergeProposal>,
    /// The query's answer, when it asked for anything.
    result: Option<GraphAnswer>,
}

/// A graph page query's title and its result, or why it could not run.
struct GraphAnswer {
    title: String,
    result: CoreResult<GraphResult>,
}

impl GraphPageData {
    fn read(db: &WorkspaceDb, ask: &GraphAsk, options: &GraphConfig) -> CoreResult<Self> {
        let status = graph_store::status(db)?;
        let ontology = ontology_store::current(db)?;
        let revalidation = status.stale.then(|| Revalidation::preview(db));
        let merges = resolve::pending(db)?;
        let result = ask.run(db, options);
        Ok(Self {
            status,
            has_ontology: ontology.is_some(),
            revalidation,
            merges,
            result,
        })
    }
}

impl GraphResultView {
    /// `result` for the inspector: node and edge rows with their sources.
    fn of(title: String, result: &GraphResult) -> Result<Self, ApiError> {
        let sources_of = |subject: &str| -> String {
            let mut items: Vec<String> = result
                .provenance
                .iter()
                .filter(|p| p.subject_id == subject)
                .map(|p| match &p.origin {
                    Origin::Row {
                        table_name,
                        row_key,
                    } => format!("{table_name} row {row_key}"),
                    Origin::Chunk {
                        document_id: Some(document),
                        ..
                    } => format!("document {}", document.short()),
                    Origin::Chunk {
                        document_id: None, ..
                    } => String::from("unknown"),
                    Origin::Manual { .. } => p.origin.assertion().unwrap_or_default(),
                })
                .collect();
            items.sort();
            items.dedup();
            items.join(", ")
        };
        let label_of = |id: &NodeId| -> String {
            result
                .nodes
                .iter()
                .find(|n| &n.id == id)
                .map_or_else(|| id.short().to_owned(), |n| n.label.clone())
        };
        let nodes = result
            .nodes
            .iter()
            .map(|n| GraphNodeView {
                id: n.id.clone(),
                label: n.label.clone(),
                class_id: n.class_id.clone(),
                standing: n.standing,
                properties: n
                    .properties
                    .iter()
                    .map(|(k, v)| format!("{k}: {}", JsonText(v)))
                    .collect::<Vec<_>>()
                    .join(" · "),
                sources: sources_of(n.id.as_str()),
            })
            .collect();
        let edges = result
            .edges
            .iter()
            .map(|e| GraphEdgeView {
                id: e.id.clone(),
                source: label_of(&e.source_node_id),
                relation: e.relation_id.clone(),
                target: label_of(&e.target_node_id),
                sources: sources_of(e.id.as_str()),
            })
            .collect();
        Ok(Self {
            title,
            json: serde_json::to_string(result)?,
            nodes,
            edges,
        })
    }
}

#[derive(Deserialize)]
struct ExtractForm {
    #[serde(default, deserialize_with = "blank_as_none")]
    source: Option<ExtractSource>,
    sample: Option<String>,
    #[serde(default)]
    reset: bool,
    #[serde(default)]
    all: bool,
}

async fn graph_extract(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(form): Form<ExtractForm>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let sample = form
        .sample
        .as_deref()
        .and_then(|s| s.trim().parse::<u32>().ok());
    let started = access
        .start_extraction(
            &app,
            &graph_api::ExtractionPlan {
                source: form.source.unwrap_or_default(),
                sample,
                reset: form.reset.then_some(if form.all {
                    Keep::Nothing
                } else {
                    Keep::Asserted
                }),
            },
        )
        .await;
    Ok(
        Flash::after(format!("/w/{id}/graph"), started, |started| match started {
            ExtractionStarted::Running { .. } => Some(String::from(
                "document extraction started in the background; this page shows the graph as it grows",
            )),
            ExtractionStarted::Done { .. } => None,
        })
        .into_response(),
    )
}

/// The counts the graph page showed beside its revalidate button; a
/// button shown with nothing to drop sends none.
async fn graph_revalidate(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(approval): Form<DropApproval>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let revalidated = access.revalidate_graph(&app, approval).await;
    Ok(Flash::after(format!("/w/{id}/graph"), revalidated, |r| {
        Some(format!(
            "dropped {} nodes and {} edges",
            r.dropped_nodes, r.dropped_edges
        ))
    })
    .into_response())
}

async fn graph_review(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    access.review_graph(&app).await?;
    Ok(Flash::to(format!("/w/{id}/graph")).into_response())
}

#[derive(Deserialize)]
struct MergeForm {
    action: String,
}

async fn graph_merge_decide(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, mid)): Path<(WorkspaceId, String)>,
    Form(form): Form<MergeForm>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    // Parsed rather than extracted, so a bad value comes back as a notice
    // on the page instead of an error page.
    let back = format!("/w/{id}/graph");
    let decision = match form.action.parse::<MergeDecision>() {
        Ok(decision) => decision,
        Err(e) => return Ok(Flash::error(back, e.to_string()).into_response()),
    };
    let decided = access.decide_merge(&app, &mid, decision).await;
    Ok(Flash::after(back, decided, |_| None).into_response())
}

/// The graph page's add-node form.
#[derive(Deserialize)]
struct NodeForm {
    label: String,
    class: String,
    /// `KEY=VALUE` lines.
    #[serde(default)]
    properties: String,
    #[serde(default, deserialize_with = "blank_as_none")]
    note: Option<String>,
}

/// The graph page's edit form for one node: blank fields stay as they
/// are; a `KEY=` line with no value removes that property.
#[derive(Deserialize)]
struct NodeEditForm {
    #[serde(default, deserialize_with = "blank_as_none")]
    label: Option<String>,
    #[serde(default, deserialize_with = "blank_as_none")]
    class: Option<String>,
    #[serde(default)]
    properties: String,
    #[serde(default, deserialize_with = "blank_as_none")]
    note: Option<String>,
}

/// The graph page's add-edge form: nodes by id, as the inspector lists them.
#[derive(Deserialize)]
struct EdgeForm {
    source: String,
    target: String,
    relation: String,
    #[serde(default, deserialize_with = "blank_as_none")]
    note: Option<String>,
}

/// `KEY=VALUE` lines as a property patch: an empty value is `null`, which
/// removes the key on an edit.
fn property_lines(text: &str) -> Result<serde_json::Map<String, serde_json::Value>, ApiError> {
    let lines: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect();
    let mut patch =
        graph_cli::parse_properties(&lines).map_err(|e| ApiError::bad_request(e.to_string()))?;
    for value in patch.values_mut() {
        if value.as_str().is_some_and(str::is_empty) {
            *value = serde_json::Value::Null;
        }
    }
    Ok(patch)
}

async fn graph_node_add(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(form): Form<NodeForm>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let back = format!("/w/{id}/graph");
    let properties = match property_lines(&form.properties) {
        Ok(patch) => Properties::from(patch),
        Err(e) => return Ok(Flash::error(back, e.message).into_response()),
    };
    let added = access
        .create_node(
            &app,
            graph_api::CreateNode {
                label: form.label,
                class: ClassId::from(form.class),
                properties,
                note: form.note,
            },
        )
        .await;
    Ok(Flash::after(back, added, |added| {
        Some(if added.created {
            format!("added {}", added.subject)
        } else {
            format!("{} was already in the graph; asserted", added.subject)
        })
    })
    .into_response())
}

async fn graph_node_edit(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, nid)): Path<(WorkspaceId, NodeId)>,
    Form(form): Form<NodeEditForm>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let back = format!("/w/{id}/graph");
    let patch = match property_lines(&form.properties) {
        Ok(patch) => patch,
        Err(e) => return Ok(Flash::error(back, e.message).into_response()),
    };
    let edit = NodeEdit {
        label: form.label,
        class: form.class.map(ClassId::from),
        properties: (!patch.is_empty()).then_some(patch),
    };
    let updated = access
        .update_node(
            &app,
            &nid,
            graph_api::UpdateNode {
                edit,
                note: form.note,
            },
        )
        .await;
    Ok(Flash::after(back, updated, |node| Some(format!("updated {node}"))).into_response())
}

async fn graph_node_delete(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, nid)): Path<(WorkspaceId, NodeId)>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let deleted = access.delete_node(&app, &nid).await;
    Ok(Flash::after(format!("/w/{id}/graph"), deleted, |node| {
        Some(format!("deleted {node} and its edges"))
    })
    .into_response())
}

async fn graph_edge_add(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(form): Form<EdgeForm>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let added = access
        .create_edge(
            &app,
            graph_api::CreateEdge {
                source: NodeId::from(form.source.trim()),
                target: NodeId::from(form.target.trim()),
                relation: RelationId::from(form.relation.trim()),
                properties: Properties::default(),
                note: form.note,
            },
        )
        .await;
    Ok(Flash::after(format!("/w/{id}/graph"), added, |added| {
        Some(if added.created {
            format!("added the {} edge", added.subject.relation_id)
        } else {
            format!(
                "the {} edge was already there; asserted",
                added.subject.relation_id
            )
        })
    })
    .into_response())
}

async fn graph_edge_delete(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, eid)): Path<(WorkspaceId, EdgeId)>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let deleted = access.delete_edge(&app, &eid).await;
    Ok(Flash::after(format!("/w/{id}/graph"), deleted, |edge| {
        Some(format!("deleted the {} edge", edge.relation_id))
    })
    .into_response())
}
