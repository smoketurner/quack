//! `/api/v1`: JSON over bearer or cookie auth (design doc 11.2). Every
//! handler resolves an [`Access`](super::auth::Access) first, so the role
//! and scope checks, and the denied audit rows, live in one place.

mod admin;
mod auth;
mod context;
pub(crate) mod documents;
pub(crate) mod embeddings;
pub(crate) mod graph;
pub(crate) mod import;
pub(crate) mod jobs;
mod members;
pub(crate) mod okf;
pub(crate) mod ontology;
pub(crate) mod query;
pub(crate) mod sessions;
mod tables;
mod workspaces;

use axum::Router;
use axum::response::sse::Event;
use axum::routing::{delete, get, post};

use super::state::App;

/// The event names the query and job streams send. Clients subscribe by
/// these names, so they are a contract, written once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamEvent {
    Status,
    Text,
    ToolStarted,
    ToolFinished,
    WriteRefused,
    Complete,
    Error,
    Jobs,
    Job,
}

impl StreamEvent {
    fn as_str(self) -> &'static str {
        match self {
            Self::Status => "status",
            Self::Text => "text",
            Self::ToolStarted => "tool_started",
            Self::ToolFinished => "tool_finished",
            Self::WriteRefused => "write_refused",
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

pub(crate) fn router() -> Router<App> {
    Router::new()
        .route("/auth/login", super::throttled_login(post(auth::login)))
        .route("/auth/logout", post(auth::logout))
        .route("/auth/me", get(auth::me))
        .route(
            "/workspaces",
            get(workspaces::list).post(workspaces::create),
        )
        .route(
            "/workspaces/{id}",
            get(workspaces::show).patch(workspaces::update),
        )
        .route("/workspaces/{id}/audit", get(workspaces::audit_detail))
        .route("/workspaces/{id}/query", post(query::query))
        .route("/workspaces/{id}/query/stream", post(query::stream))
        .route("/workspaces/{id}/sql", post(query::sql))
        .route("/workspaces/{id}/search", get(query::search))
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
        .route("/workspaces/{id}/embeddings", get(embeddings::show))
        .route(
            "/workspaces/{id}/embeddings/refresh",
            post(embeddings::refresh),
        )
        .route("/workspaces/{id}/tables", get(tables::list))
        .route("/workspaces/{id}/tables/{name}", get(tables::describe))
        .route(
            "/workspaces/{id}/context",
            get(context::show).put(context::replace),
        )
        .route("/workspaces/{id}/context/versions", get(context::versions))
        .route(
            "/workspaces/{id}/ontology",
            get(ontology::show).put(ontology::replace),
        )
        .route("/workspaces/{id}/ontology/init", post(ontology::init))
        .route("/workspaces/{id}/ontology/propose", post(ontology::propose))
        .route(
            "/workspaces/{id}/ontology/candidates",
            get(ontology::list_candidates).post(ontology::decide_many),
        )
        .route(
            "/workspaces/{id}/ontology/candidates/{cid}",
            axum::routing::put(ontology::decide),
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
        .route("/workspaces/{id}/okf", get(okf::export))
        .route("/workspaces/{id}/import", post(import::import))
        .route("/workspaces/{id}/graph/search", get(graph::search))
        .route("/workspaces/{id}/graph/path", get(graph::path))
        .route("/workspaces/{id}/graph/status", get(graph::status))
        .route("/workspaces/{id}/graph/extract", post(graph::extract))
        .route("/workspaces/{id}/graph/revalidate", post(graph::revalidate))
        .route("/workspaces/{id}/graph/review", post(graph::review))
        .route("/workspaces/{id}/graph/merges", get(graph::merges))
        .route(
            "/workspaces/{id}/graph/merges/{mid}",
            axum::routing::put(graph::decide_merge),
        )
        .route("/workspaces/{id}/jobs", get(jobs::list))
        .route("/workspaces/{id}/jobs/stream", get(jobs::stream))
        .route("/workspaces/{id}/jobs/{job}", get(jobs::show))
        .route("/workspaces/{id}/jobs/{job}/cancel", post(jobs::cancel))
        .route("/workspaces/{id}/sessions", get(sessions::list))
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
            "/workspaces/{id}/members",
            get(members::list).post(members::add),
        )
        .route("/workspaces/{id}/members/{user}", delete(members::remove))
        .route("/admin/users", get(admin::users).post(admin::create_user))
        .route("/admin/audit", get(admin::audit))
}
