//! `/api/v1`: JSON over bearer or cookie auth (design doc 11.2). Every
//! handler resolves an [`Access`](super::auth::Access) first, so the role
//! and scope checks, and the denied audit rows, live in one place.

pub(crate) mod admin;
pub(crate) mod auth;
pub(crate) mod context;
pub(crate) mod documents;
pub(crate) mod embeddings;
pub(crate) mod graph;
pub(crate) mod import;
mod jobs;
pub(crate) mod members;
pub(crate) mod okf;
pub(crate) mod ontology;
pub(crate) mod openapi;
pub(crate) mod query;
mod saved;
mod sessions;
mod tables;
pub(crate) mod workspaces;

use axum::Router;
use axum::response::sse::Event;
use axum::routing::{MethodRouter, delete, get, patch, post, put};

use super::state::App;

/// The event names the query and job streams send. Clients subscribe by
/// these names, so they are a contract, written once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamEvent {
    Status,
    Text,
    ToolStarted,
    ToolFinished,
    PermissionRequired,
    Complete,
    Error,
    Jobs,
    Job,
}

impl StreamEvent {
    /// Every event, for the API document's `x-sse-events`.
    pub(crate) const ALL: [Self; 9] = [
        Self::Status,
        Self::Text,
        Self::ToolStarted,
        Self::ToolFinished,
        Self::PermissionRequired,
        Self::Complete,
        Self::Error,
        Self::Jobs,
        Self::Job,
    ];

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Status => "status",
            Self::Text => "text",
            Self::ToolStarted => "tool_started",
            Self::ToolFinished => "tool_finished",
            Self::PermissionRequired => "permission_required",
            Self::Complete => "complete",
            Self::Error => "error",
            Self::Jobs => "jobs",
            Self::Job => "job",
        }
    }

    /// An SSE event carrying this name.
    pub(crate) fn event(self) -> Event {
        Event::default().event(self.as_str())
    }
}

/// The API's router, and every path it registers, so a test can hold the
/// API document to the routes that exist.
pub(crate) struct ApiRoutes {
    router: Router<App>,
    paths: Vec<&'static str>,
}

impl ApiRoutes {
    fn new() -> Self {
        Self {
            router: Router::new(),
            paths: Vec::new(),
        }
    }

    fn route(self, path: &'static str, methods: MethodRouter<App>) -> Self {
        let Self { router, mut paths } = self;
        paths.push(path);
        Self {
            router: router.route(path, methods),
            paths,
        }
    }

    fn merge(self, other: Self) -> Self {
        let Self { router, mut paths } = self;
        paths.extend(other.paths);
        Self {
            router: router.merge(other.router),
            paths,
        }
    }

    /// Every path, as the router and the document spell it.
    #[cfg(test)]
    pub(crate) fn paths(&self) -> &[&'static str] {
        &self.paths
    }

    pub(crate) fn into_router(self) -> Router<App> {
        self.router
    }
}

pub(crate) fn router(app: &App) -> ApiRoutes {
    ApiRoutes::new()
        .merge(control_routes(app))
        .route("/workspaces/{id}/audit", get(workspaces::audit_detail))
        .route("/workspaces/{id}/query", post(query::query))
        .route("/workspaces/{id}/query/stream", post(query::stream))
        .route("/workspaces/{id}/sql", post(query::sql))
        .route("/workspaces/{id}/sql/export", post(query::export))
        .route("/workspaces/{id}/search", post(query::search))
        .route(
            "/workspaces/{id}/documents",
            get(documents::list).post(documents::upload),
        )
        .route(
            "/workspaces/{id}/documents/{doc}",
            get(documents::show)
                .patch(documents::update)
                .delete(documents::remove),
        )
        .route(
            "/workspaces/{id}/documents/{doc}/chunks",
            get(documents::chunks),
        )
        .route(
            "/workspaces/{id}/documents/{doc}/image",
            get(documents::image),
        )
        .route("/workspaces/{id}/embeddings", get(embeddings::show))
        .route(
            "/workspaces/{id}/embeddings/refresh",
            post(embeddings::refresh),
        )
        .route("/workspaces/{id}/tables", get(tables::list))
        .route("/workspaces/{id}/tables/describe", post(tables::describe))
        .route("/workspaces/{id}/tables/schema", get(tables::schema))
        .route("/workspaces/{id}/tables/note", put(tables::note))
        .route("/workspaces/{id}/tables/retype", post(tables::retype))
        .route(
            "/workspaces/{id}/context",
            get(context::show).put(context::replace),
        )
        .route("/workspaces/{id}/context/versions", get(context::versions))
        .merge(ontology_routes())
        .route("/workspaces/{id}/okf", get(okf::export))
        .route("/workspaces/{id}/import", post(import::import))
        .route(
            "/workspaces/{id}/imports",
            get(import::list_saved).post(import::save),
        )
        .route(
            "/workspaces/{id}/imports/{import}/refresh",
            post(import::refresh),
        )
        .route("/workspaces/{id}/imports/{import}", delete(import::remove))
        .merge(graph_routes())
        .route("/workspaces/{id}/jobs", get(jobs::list))
        .route("/workspaces/{id}/jobs/stream", get(jobs::stream))
        .route("/workspaces/{id}/jobs/{job}", get(jobs::show))
        .route("/workspaces/{id}/jobs/{job}/cancel", post(jobs::cancel))
        .route(
            "/workspaces/{id}/saved",
            get(saved::list).post(saved::create),
        )
        .route(
            "/workspaces/{id}/saved/{saved}",
            get(saved::show).delete(saved::remove),
        )
        .route("/workspaces/{id}/saved/{saved}/run", post(saved::run))
        .route("/workspaces/{id}/saved/{saved}/runs", get(saved::runs))
        .route("/workspaces/{id}/sessions", get(sessions::list))
        .route("/workspaces/{id}/sessions/search", get(sessions::search))
        .route(
            "/workspaces/{id}/sessions/{sid}",
            get(sessions::show)
                .patch(sessions::update)
                .delete(sessions::remove),
        )
        .route(
            "/workspaces/{id}/sessions/{sid}/export",
            get(sessions::export),
        )
        .route(
            "/workspaces/{id}/sessions/{sid}/permissions/{request}",
            post(sessions::decide),
        )
        .merge(membership_routes())
        .route("/admin/users", get(admin::users).post(admin::create_user))
        .route(
            "/admin/users/{user}",
            patch(admin::update_user).delete(admin::delete_user),
        )
        .route("/admin/audit", get(admin::audit))
}

/// Signing in, and the workspaces themselves: listing, creating,
/// restoring, settings, snapshots, deletion.
fn control_routes(app: &App) -> ApiRoutes {
    ApiRoutes::new()
        .route(
            "/auth/login",
            super::throttled_login(app, post(auth::login)),
        )
        .route("/auth/logout", post(auth::logout))
        .route("/auth/logout-all", post(auth::logout_all))
        .route("/auth/password", post(auth::change_password))
        .route("/auth/me", get(auth::me))
        .route(
            "/workspaces",
            get(workspaces::list).post(workspaces::create),
        )
        .route("/workspaces/restore", post(workspaces::restore))
        .route(
            "/workspaces/{id}",
            get(workspaces::show)
                .patch(workspaces::update)
                .delete(workspaces::delete),
        )
        .route("/workspaces/{id}/snapshot", get(workspaces::snapshot))
}

/// The knowledge graph's routes: search and path, status, the builds, the
/// merge queue, and a person's node and edge edits.
fn graph_routes() -> ApiRoutes {
    ApiRoutes::new()
        .route("/workspaces/{id}/graph/search", post(graph::search))
        .route("/workspaces/{id}/graph/path", post(graph::path))
        .route("/workspaces/{id}/graph/status", get(graph::status))
        .route("/workspaces/{id}/graph/export", get(graph::export))
        .route("/workspaces/{id}/graph/extract", post(graph::extract))
        .route(
            "/workspaces/{id}/graph/revalidate",
            get(graph::revalidation_preview).post(graph::revalidate),
        )
        .route("/workspaces/{id}/graph/review", post(graph::review))
        .route("/workspaces/{id}/graph/merges", get(graph::merges))
        .route(
            "/workspaces/{id}/graph/merges/{mid}",
            put(graph::decide_merge),
        )
        .route("/workspaces/{id}/graph/nodes", post(graph::create_node))
        .route(
            "/workspaces/{id}/graph/nodes/{nid}",
            patch(graph::update_node).delete(graph::delete_node),
        )
        .route("/workspaces/{id}/graph/edges", post(graph::create_edge))
        .route(
            "/workspaces/{id}/graph/edges/{eid}",
            delete(graph::delete_edge),
        )
}

/// Members by hand, and the roles the identity provider's groups carry.
fn membership_routes() -> ApiRoutes {
    ApiRoutes::new()
        .route(
            "/workspaces/{id}/members",
            get(members::list).post(members::add),
        )
        .route("/workspaces/{id}/members/{user}", delete(members::remove))
        .route(
            "/workspaces/{id}/groups",
            get(members::groups).post(members::set_group),
        )
        .route(
            "/workspaces/{id}/groups/{group}",
            delete(members::remove_group),
        )
}

/// The ontology's routes: the current one, its candidates, and its versions.
fn ontology_routes() -> ApiRoutes {
    ApiRoutes::new()
        .route(
            "/workspaces/{id}/ontology",
            get(ontology::show).put(ontology::replace),
        )
        .route("/workspaces/{id}/ontology/schema", get(ontology::schema))
        .route("/workspaces/{id}/ontology/init", post(ontology::init))
        .route("/workspaces/{id}/ontology/rename", post(ontology::rename))
        .route("/workspaces/{id}/ontology/propose", post(ontology::propose))
        .route(
            "/workspaces/{id}/ontology/candidates",
            get(ontology::list_candidates).post(ontology::decide_many),
        )
        .route(
            "/workspaces/{id}/ontology/candidates/{cid}",
            put(ontology::decide),
        )
        .route(
            "/workspaces/{id}/ontology/versions",
            get(ontology::versions),
        )
        .route(
            "/workspaces/{id}/ontology/versions/{v}",
            get(ontology::version),
        )
        .route(
            "/workspaces/{id}/ontology/versions/{v}/restore",
            post(ontology::restore),
        )
}
