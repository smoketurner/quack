//! The document against the router: every route and method documented,
//! nothing documented that is not routed, and the routes that serve it.

#![expect(
    clippy::indexing_slicing,
    reason = "serde_json::Value indexing yields Null for a missing key, never a panic"
)]

use std::collections::BTreeSet;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use quack_core::config::Config;
use quack_core::storage::control::{AuditFilter, ControlPlane};
use quack_core::web_sessions::WebSessions;
use tower::ServiceExt;

use super::{DOCUMENT_PATH, Endpoint, PAGE_PATH, StreamEvent, openapi};
use crate::server::state::{App, AppState, ServeMode};
use crate::server::{self, api};
use utoipa::openapi::RefOr;
use utoipa::openapi::path::ParameterIn;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

/// A login-mode server with nobody signed in.
async fn app() -> (tempfile::TempDir, App) {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = Config::default();
    config.general.data_dir = dir.path().to_path_buf();
    let control = ControlPlane::open(&config)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let sessions = Arc::new(WebSessions::new(&config.server));
    let app = Arc::new(AppState::new(
        config,
        control,
        ServeMode::Login,
        sessions,
        None,
    ));
    (dir, app)
}

/// The routed paths the document leaves out.
fn undocumented<'a>(routes: &[&'a str], doc: &utoipa::openapi::OpenApi) -> Vec<&'a str> {
    routes
        .iter()
        .copied()
        .filter(|path| !doc.paths.paths.contains_key(*path))
        .collect()
}

/// A path with each `{name}` filled in, so the router matches it.
fn concrete(path: &str) -> String {
    path.split('/')
        .map(|segment| {
            if segment.starts_with('{') {
                if segment == "{v}" {
                    String::from("1")
                } else {
                    uuid::Uuid::now_v7().to_string()
                }
            } else {
                segment.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("/")
}

#[tokio::test(flavor = "multi_thread")]
async fn every_routed_path_is_documented_and_nothing_else() {
    let (_dir, app) = app().await;
    let routes = api::router(&app);
    let doc = openapi();
    assert_eq!(
        undocumented(routes.paths(), doc),
        Vec::<&str>::new(),
        "routes missing from the OpenAPI document"
    );
    let routed: BTreeSet<&str> = routes.paths().iter().copied().collect();
    let stale: Vec<&String> = doc
        .paths
        .paths
        .keys()
        .filter(|path| !routed.contains(path.as_str()))
        .collect();
    assert!(stale.is_empty(), "documented but not routed: {stale:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_route_left_out_of_the_document_is_caught() {
    let (_dir, app) = app().await;
    let routes = api::router(&app);
    let mut doc = openapi().clone();
    doc.paths.paths.remove("/workspaces/{id}/sql");
    assert_eq!(undocumented(routes.paths(), &doc), ["/workspaces/{id}/sql"]);
}

/// The router answers every documented method and only those: anything
/// else is 405 before any extractor runs, so no request here reaches a
/// handler (nobody is signed in).
#[tokio::test(flavor = "multi_thread")]
async fn every_method_the_router_answers_is_documented() {
    let (_dir, app) = app().await;
    let routes = api::router(&app);
    let paths: Vec<&str> = routes.paths().to_vec();
    let router = routes.into_router().with_state(Arc::clone(&app));
    let doc = openapi();
    for path in paths {
        let documented: BTreeSet<&str> = doc
            .paths
            .paths
            .get(path)
            .map(|item| {
                Endpoint::operations(item)
                    .into_iter()
                    .map(|(method, _)| method)
                    .collect()
            })
            .unwrap_or_default();
        for method in [
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
        ] {
            let request = Request::builder()
                .method(method.clone())
                .uri(concrete(path))
                .body(Body::empty())
                .unwrap_or_else(|e| fail(&e.to_string()));
            let status = router
                .clone()
                .oneshot(request)
                .await
                .unwrap_or_else(|e| fail(&e.to_string()))
                .status();
            let routed = status != StatusCode::METHOD_NOT_ALLOWED;
            assert_eq!(
                routed,
                documented.contains(method.as_str()),
                "{method} {path}: routed {routed}, documented {documented:?}"
            );
        }
    }
}

/// `axum_extras` documents a path's parameters from the handler's `Path`
/// extractor, so a handler that reads one some other way leaves it out.
#[test]
fn every_path_parameter_comes_from_its_handler() {
    let mut missing = Vec::new();
    for (path, item) in &openapi().paths.paths {
        let named: Vec<&str> = path
            .split('/')
            .filter_map(|segment| segment.strip_prefix('{')?.strip_suffix('}'))
            .collect();
        for (method, operation) in Endpoint::operations(item) {
            let documented: Vec<&str> = operation
                .parameters
                .iter()
                .flatten()
                .filter_map(|p| match p {
                    RefOr::T(parameter) => Some(parameter),
                    RefOr::Ref(_) => None,
                })
                .filter(|p| p.parameter_in == ParameterIn::Path)
                .map(|p| p.name.as_str())
                .collect();
            if documented != named {
                missing.push(format!("{method} {path}: {documented:?}"));
            }
        }
    }
    assert!(missing.is_empty(), "{missing:#?}");
}

/// A parameter's location as the document spells it; `ParameterIn` has
/// no `Debug` without utoipa's `debug` feature.
fn location(parameter_in: &ParameterIn) -> &'static str {
    match parameter_in {
        ParameterIn::Path => "path",
        ParameterIn::Query => "query",
        ParameterIn::QueryString => "querystring",
        ParameterIn::Header => "header",
        ParameterIn::Cookie => "cookie",
    }
}

/// Query structs list no `parameter_in`; `axum_extras` takes it from the
/// handler's `Query<T>`, so their fields must land in the query, beside the
/// path parameters.
#[test]
fn query_structs_are_documented_as_query_parameters() {
    let document = openapi();
    for (path, method, expected) in [
        (
            "/workspaces/{id}/documents/{doc}/chunks",
            "GET",
            vec![
                ("id", "path"),
                ("doc", "path"),
                ("from", "query"),
                ("limit", "query"),
            ],
        ),
        ("/admin/audit", "GET", vec![("format", "query")]),
    ] {
        let operation = document
            .paths
            .paths
            .get(path)
            .map(Endpoint::operations)
            .into_iter()
            .flatten()
            .find(|(m, _)| *m == method)
            .map(|(_, operation)| operation);
        let listed: Vec<(&str, &str)> = operation
            .and_then(|op| op.parameters.as_ref())
            .into_iter()
            .flatten()
            .filter_map(|p| match p {
                RefOr::T(parameter) => {
                    Some((parameter.name.as_str(), location(&parameter.parameter_in)))
                }
                RefOr::Ref(_) => None,
            })
            .collect();
        for wanted in &expected {
            assert!(
                listed.contains(wanted),
                "{method} {path} lacks {wanted:?}: {listed:?}"
            );
        }
    }
}

/// Every operation answers errors with the coded body, and the code enum
/// is a component.
#[test]
fn every_operation_documents_the_error_body() {
    let doc = serde_json::to_value(openapi()).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(doc["openapi"], "3.1.0");
    assert_eq!(doc["servers"][0]["url"], "/api/v1");
    let codes = &doc["components"]["schemas"]["ErrorCode"]["enum"];
    for code in [
        "auth_required",
        "not_found",
        "query_timeout",
        "table_taken",
        "busy",
    ] {
        assert!(
            codes
                .as_array()
                .is_some_and(|all| all.iter().any(|c| c == code)),
            "{code} missing from {codes}"
        );
    }
    let mut operations = 0_usize;
    let mut ids = BTreeSet::new();
    for (path, item) in &openapi().paths.paths {
        for (method, operation) in Endpoint::operations(item) {
            operations = operations.saturating_add(1);
            assert!(
                operation.responses.responses.contains_key("default"),
                "{method} {path} has no error response"
            );
            let id = operation.operation_id.clone().unwrap_or_default();
            assert!(
                ids.insert(id.clone()),
                "{method} {path}: operationId {id} repeats"
            );
        }
    }
    assert!(ids.contains("documents_list") && ids.contains("graph_search"));
    assert!(operations >= 90, "{operations} operations");
    for scheme in [super::BEARER, super::COOKIE] {
        assert!(
            doc["components"]["securitySchemes"][scheme].is_object(),
            "{scheme}"
        );
        assert!(
            doc["security"]
                .as_array()
                .is_some_and(|all| all.iter().any(|s| s[scheme].is_array())),
            "{scheme} is not required by default"
        );
    }
    assert_eq!(
        doc["paths"]["/auth/login"]["post"]["security"],
        serde_json::json!([{}])
    );
}

/// Every `$ref` in `value`, depth first.
fn refs<'a>(value: &'a serde_json::Value, found: &mut Vec<&'a str>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, inner) in map {
                match (key.as_str(), inner.as_str()) {
                    ("$ref", Some(target)) => found.push(target),
                    _ => refs(inner, found),
                }
            }
        }
        serde_json::Value::Array(items) => {
            for inner in items {
                refs(inner, found);
            }
        }
        serde_json::Value::Null
        | serde_json::Value::Bool(_)
        | serde_json::Value::Number(_)
        | serde_json::Value::String(_) => {}
    }
}

/// Every schema a reference names is in the components, the SSE events'
/// included.
#[test]
fn every_reference_resolves() {
    let doc = serde_json::to_value(openapi()).unwrap_or_else(|e| fail(&e.to_string()));
    let mut found = Vec::new();
    refs(&doc, &mut found);
    assert!(found.len() > 100, "{} references", found.len());
    let schemas = &doc["components"]["schemas"];
    let dangling: BTreeSet<&str> = found
        .into_iter()
        .filter(|target| {
            target
                .strip_prefix("#/components/schemas/")
                .is_none_or(|name| !schemas[name].is_object())
        })
        .collect();
    assert!(dangling.is_empty(), "{dangling:?}");
}

/// The two streams list their events, named by `StreamEvent` and nowhere
/// else.
#[test]
fn the_streams_list_their_events() {
    let doc = serde_json::to_value(openapi()).unwrap_or_else(|e| fail(&e.to_string()));
    let query = &doc["paths"]["/workspaces/{id}/query/stream"]["post"]["x-sse-events"];
    let jobs = &doc["paths"]["/workspaces/{id}/jobs/stream"]["get"]["x-sse-events"];
    let mut listed = 0_usize;
    for event in StreamEvent::ALL {
        let name = event.as_str();
        let in_query = query[name].is_object();
        let in_jobs = jobs[name].is_object();
        assert!(in_query != in_jobs, "{name} must be in exactly one stream");
        listed = listed.saturating_add(1);
    }
    assert_eq!(listed, StreamEvent::ALL.len());
    assert_eq!(
        query["complete"]["data"]["$ref"],
        "#/components/schemas/CompleteEvent"
    );
    assert_eq!(
        query["error"]["data"]["$ref"],
        "#/components/schemas/ErrorBody"
    );
    assert_eq!(jobs["jobs"]["data"]["type"], "array");
    let schemas = &doc["components"]["schemas"];
    for name in [
        "CompleteEvent",
        "ErrorBody",
        "JobInfo",
        "ToolStep",
        "PermissionEvent",
    ] {
        assert!(schemas[name].is_object(), "{name} is not a component");
    }
}

/// The document and its page need no sign-in, write no audit row, and set
/// their own cache policy.
#[tokio::test(flavor = "multi_thread")]
async fn the_document_and_its_page_are_served_to_anyone() {
    let (_dir, app) = app().await;
    let router = server::router(Arc::clone(&app));
    for (path, kind) in [
        (DOCUMENT_PATH, "application/json"),
        (PAGE_PATH, "text/html"),
    ] {
        let response = router
            .clone()
            .oneshot(
                Request::get(path)
                    .body(Body::empty())
                    .unwrap_or_else(|e| fail(&e.to_string())),
            )
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(response.status(), StatusCode::OK, "{path}");
        let headers = response.headers();
        assert_eq!(
            headers
                .get(header::CACHE_CONTROL)
                .and_then(|v| v.to_str().ok()),
            Some("no-cache"),
            "{path}"
        );
        assert!(
            headers
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.starts_with(kind)),
            "{path}"
        );
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let text = String::from_utf8_lossy(&body);
        if path == PAGE_PATH {
            assert!(text.contains("/static/js/redoc.standalone.js"), "{text}");
            assert!(text.contains(DOCUMENT_PATH), "{text}");
        } else {
            assert!(text.contains("\"openapi\":\"3.1.0\""));
        }
    }
    let page = app
        .control
        .query_audit(&AuditFilter::default())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(page.rows.is_empty(), "{:?}", page.rows);
    let viewer = router
        .oneshot(
            Request::get("/static/js/redoc.standalone.js")
                .body(Body::empty())
                .unwrap_or_else(|e| fail(&e.to_string())),
        )
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(viewer.status(), StatusCode::OK);
}
