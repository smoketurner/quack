//! The REST API's contract (issue #419): an `OpenAPI` 3.1 document generated
//! from the handlers' `#[utoipa::path]` annotations and the request and
//! response types' schemas, served at `/api/v1/openapi.json`, and a page that
//! renders it with the vendored Redoc at `/api/v1/docs`. Neither reveals
//! workspace content, so both sit beside `/healthz`: no sign-in, no audit
//! row, no limiter.

use std::borrow::Cow;
use std::sync::LazyLock;

use askama::Template;
use axum::Json;
use axum::http::header;
use axum::response::{Html, IntoResponse, Response};
use quack_core::analysis::events::ToolStep;
use quack_core::graph::export::GraphFormat;
use quack_core::jobs::JobInfo;
use quack_core::ontology::candidates::Queue;
use quack_core::storage::sessions::ExportFormat as TranscriptFormat;
use serde_json::json;
use utoipa::openapi::path::{Operation, PathItem};
use utoipa::openapi::security::{ApiKey, ApiKeyValue, HttpAuthScheme, HttpBuilder, SecurityScheme};
use utoipa::openapi::{ContentBuilder, Ref, RefOr, ResponseBuilder, extensions::ExtensionsBuilder};
use utoipa::{Modify, OpenApi, ToSchema};

use super::StreamEvent;
use super::{
    admin, auth, classify, context, documents, embeddings, graph, import, jobs, members, okf,
    ontology, query, saved, sessions, tables, workspaces,
};
use crate::auth::SESSION_COOKIE;
use crate::error::{ApiError, ApiResult, ErrorBody, ErrorCode};

/// Where the document is served.
pub(crate) const DOCUMENT_PATH: &str = "/api/v1/openapi.json";

/// Where the page that renders it is served.
pub(crate) const PAGE_PATH: &str = "/api/v1/docs";

/// The two security schemes: a bearer (an API token, a login's session
/// token, or the identity provider's access token) and the browser's
/// session cookie. The derive's `security` spells them as literals.
const BEARER: &str = "bearer";
const COOKIE: &str = "session";

#[derive(OpenApi)]
#[openapi(
    info(
        title = "quack",
        description = "The REST API of `quack serve`: workspaces of documents, tables, and a knowledge graph, and the agent that answers across them. Every error response is `{\"error\": \"...\", \"code\": \"...\"}`: branch on `code`, which is stable; `error` is for a person and may change in any release.",
        license(name = "MIT OR Apache-2.0", identifier = "MIT OR Apache-2.0"),
    ),
    servers((url = "/api/v1")),
    security(("bearer" = []), ("session" = [])),
    tags(
        (name = "auth", description = "Signing in and out"),
        (name = "workspaces", description = "Workspaces, their settings, snapshots, and audit detail"),
        (name = "query", description = "The agent, SQL, and search"),
        (name = "documents", description = "Documents and their chunks"),
        (name = "embeddings", description = "Stored vectors and their refresh"),
        (name = "tables", description = "User tables"),
        (name = "context", description = "The workspace context the agent reads"),
        (name = "ontology", description = "The ontology, its candidates, and its versions"),
        (name = "graph", description = "The knowledge graph"),
        (name = "import", description = "External data as tables"),
        (name = "okf", description = "Open Knowledge Format bundles"),
        (name = "jobs", description = "Background work"),
        (name = "saved", description = "Saved questions"),
        (name = "sessions", description = "Chat sessions"),
        (name = "members", description = "Members and group roles"),
        (name = "admin", description = "Server users and the access audit"),
    ),
    paths(
        auth::login,
        auth::logout,
        auth::logout_all,
        auth::change_password,
        auth::me,
        workspaces::list,
        workspaces::create,
        workspaces::restore,
        workspaces::show,
        workspaces::update,
        workspaces::delete,
        workspaces::snapshot,
        workspaces::audit_detail,
        query::query,
        query::stream,
        query::sql,
        query::export,
        query::search,
        documents::list,
        documents::upload,
        documents::show,
        documents::update,
        documents::remove,
        documents::chunks,
        documents::image,
        embeddings::show,
        embeddings::refresh,
        tables::list,
        tables::describe,
        tables::schema,
        tables::note,
        tables::retype,
        classify::classify,
        classify::runs,
        context::show,
        context::replace,
        context::versions,
        ontology::show,
        ontology::replace,
        ontology::schema,
        ontology::init,
        ontology::rename,
        ontology::propose,
        ontology::list_candidates,
        ontology::decide_many,
        ontology::decide,
        ontology::versions,
        ontology::version,
        ontology::restore,
        okf::export,
        import::import,
        import::list_saved,
        import::save,
        import::refresh,
        import::remove,
        graph::search,
        graph::path,
        graph::status,
        graph::export,
        graph::extract,
        graph::revalidation_preview,
        graph::revalidate,
        graph::review,
        graph::merges,
        graph::decide_merge,
        graph::create_node,
        graph::update_node,
        graph::delete_node,
        graph::create_edge,
        graph::delete_edge,
        jobs::list,
        jobs::stream,
        jobs::show,
        jobs::cancel,
        saved::list,
        saved::create,
        saved::show,
        saved::remove,
        saved::run,
        saved::runs,
        sessions::list,
        sessions::search,
        sessions::show,
        sessions::update,
        sessions::remove,
        sessions::export,
        sessions::decide,
        members::list,
        members::add,
        members::remove,
        members::groups,
        members::set_group,
        members::remove_group,
        admin::users,
        admin::create_user,
        admin::update_user,
        admin::delete_user,
        admin::audit,
    ),
    // Besides the bodies the paths name: the SSE events' data, and the
    // enums only query strings carry, which `IntoParams` does not collect.
    components(schemas(
        ErrorBody,
        ErrorCode,
        query::CompleteEvent,
        query::ToolStartedEvent,
        query::PermissionEvent,
        query::Choice,
        workspaces::AuditFormat,
        admin::AuditShape,
        GraphFormat,
        Queue,
        TranscriptFormat,
    )),
    modifiers(&Conventions),
)]
struct ApiDoc;

/// Built once, on the first request.
static DOCUMENT: LazyLock<utoipa::openapi::OpenApi> = LazyLock::new(ApiDoc::openapi);

/// What every operation shares, added once here rather than on each
/// annotation: the security schemes, a typed parameter for each `{name}`
/// in its path, the coded error response, and the SSE event list.
struct Conventions;

impl Modify for Conventions {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        let components = openapi.components.get_or_insert_with(Default::default);
        components.add_security_scheme(
            BEARER,
            SecurityScheme::Http(
                HttpBuilder::new()
                    .scheme(HttpAuthScheme::Bearer)
                    .description(Some(
                        "An API token (`quack token create`), the token `POST /auth/login` returns, or, with `[server.oidc].audience` set, the identity provider's access token",
                    ))
                    .build(),
            ),
        );
        components.add_security_scheme(
            COOKIE,
            SecurityScheme::ApiKey(ApiKey::Cookie(ApiKeyValue::with_description(
                SESSION_COOKIE,
                "The browser session `POST /auth/login` sets",
            ))),
        );
        let error = ResponseBuilder::new()
            .description("The request failed; `code` says how")
            .content(
                "application/json",
                ContentBuilder::new()
                    .schema(Some(Ref::from_schema_name(ErrorBody::name())))
                    .build(),
            )
            .build();
        for (path, item) in &mut openapi.paths.paths {
            let stream = EventStream::at(path);
            for operation in Endpoint::operations_mut(item) {
                // Handler names repeat across modules (`list`, `show`); the
                // tag is the module, so the pair is unique.
                if let (Some(id), Some(tag)) = (
                    operation.operation_id.as_mut(),
                    operation.tags.as_ref().and_then(|tags| tags.first()),
                ) {
                    *id = format!("{tag}_{id}");
                }
                operation
                    .responses
                    .responses
                    .insert(String::from("default"), RefOr::T(error.clone()));
                if let Some(stream) = stream {
                    let extensions = ExtensionsBuilder::new()
                        .add("x-sse-events", stream.events())
                        .build();
                    operation
                        .extensions
                        .get_or_insert_with(Default::default)
                        .merge(extensions);
                }
            }
        }
    }
}

/// The two Server-Sent Events streams, and the events each sends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventStream {
    Query,
    Jobs,
}

impl EventStream {
    const fn path(self) -> &'static str {
        match self {
            Self::Query => "/workspaces/{id}/query/stream",
            Self::Jobs => "/workspaces/{id}/jobs/stream",
        }
    }

    fn at(path: &str) -> Option<Self> {
        [Self::Query, Self::Jobs]
            .into_iter()
            .find(|stream| stream.path() == path)
    }

    /// `x-sse-events`: each event name with what it means and its data.
    fn events(self) -> serde_json::Value {
        let mut events = serde_json::Map::new();
        for event in StreamEvent::ALL {
            if event.stream() == self {
                events.insert(
                    event.as_str().to_owned(),
                    json!({ "description": event.description(), "data": event.data() }),
                );
            }
        }
        serde_json::Value::Object(events)
    }
}

/// What each event means and carries, for the document.
impl StreamEvent {
    const fn stream(self) -> EventStream {
        match self {
            Self::Status
            | Self::Text
            | Self::ToolStarted
            | Self::ToolFinished
            | Self::PermissionRequired
            | Self::Complete
            | Self::Error => EventStream::Query,
            Self::Jobs | Self::Job => EventStream::Jobs,
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::Status => "A note on the turn's progress, as plain text",
            Self::Text => "The next piece of the answer, as plain text",
            Self::ToolStarted => "A tool call began",
            Self::ToolFinished => "A tool call ended",
            Self::PermissionRequired => {
                "A write waits for an answer at `POST .../sessions/{sid}/permissions/{request}`"
            }
            Self::Complete => "The turn's response object; the stream ends",
            Self::Error => "The turn failed; the stream ends",
            Self::Jobs => {
                "Every job of the workspace, newest first: the first event, and after a gap"
            }
            Self::Job => "One job that changed",
        }
    }

    fn data(self) -> serde_json::Value {
        let schema =
            |name: Cow<'static, str>| json!({ "$ref": format!("#/components/schemas/{name}") });
        match self {
            Self::Status | Self::Text => json!({ "type": "string" }),
            Self::ToolStarted => schema(query::ToolStartedEvent::name()),
            Self::ToolFinished => schema(ToolStep::name()),
            Self::PermissionRequired => schema(query::PermissionEvent::name()),
            Self::Complete => schema(query::CompleteEvent::name()),
            Self::Error => schema(ErrorBody::name()),
            Self::Jobs => {
                json!({ "type": "array", "items": schema(JobInfo::name()) })
            }
            Self::Job => schema(JobInfo::name()),
        }
    }
}

/// The operations of one path item, by method.
pub(crate) struct Endpoint;

impl Endpoint {
    /// Each method the path item documents, with its operation.
    #[cfg(test)]
    pub(crate) fn operations(item: &PathItem) -> Vec<(&'static str, &Operation)> {
        [
            ("GET", item.get.as_ref()),
            ("PUT", item.put.as_ref()),
            ("POST", item.post.as_ref()),
            ("DELETE", item.delete.as_ref()),
            ("PATCH", item.patch.as_ref()),
            ("HEAD", item.head.as_ref()),
            ("OPTIONS", item.options.as_ref()),
            ("TRACE", item.trace.as_ref()),
        ]
        .into_iter()
        .filter_map(|(method, operation)| Some((method, operation?)))
        .collect()
    }

    fn operations_mut(item: &mut PathItem) -> impl Iterator<Item = &mut Operation> {
        [
            item.get.as_mut(),
            item.put.as_mut(),
            item.post.as_mut(),
            item.delete.as_mut(),
            item.patch.as_mut(),
            item.head.as_mut(),
            item.options.as_mut(),
            item.trace.as_mut(),
        ]
        .into_iter()
        .flatten()
    }
}

/// The document, as built at startup.
pub(crate) fn openapi() -> &'static utoipa::openapi::OpenApi {
    &DOCUMENT
}

/// `GET /api/v1/openapi.json`. It changes only with the binary, so a
/// client revalidates rather than refetches; `no_store` does not reach it.
pub(crate) async fn document() -> Response {
    ([(header::CACHE_CONTROL, "no-cache")], Json(openapi())).into_response()
}

#[derive(Template)]
#[template(path = "api_docs.html")]
struct ApiDocsPage {
    document: &'static str,
}

/// `GET /api/v1/docs`: the document, rendered by the vendored Redoc.
pub(crate) async fn page() -> ApiResult<Response> {
    let html = ApiDocsPage {
        document: DOCUMENT_PATH,
    }
    .render()
    .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok((
        [
            (header::CACHE_CONTROL, "no-cache"),
            (header::CONTENT_SECURITY_POLICY, DOCS_PAGE_POLICY),
        ],
        Html(html),
    )
        .into_response())
}

/// The server's page policy with one allowance: Redoc validates schemas
/// with Ajv, which compiles them with `new Function`. The page shows only
/// the API's own description, never workspace content.
const DOCS_PAGE_POLICY: &str = "default-src 'self'; script-src 'self' 'unsafe-eval'; \
     style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; worker-src 'self' blob:; \
     connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; \
     frame-ancestors 'none'";

#[cfg(test)]
mod tests;
