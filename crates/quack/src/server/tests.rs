//! The API through `tower::ServiceExt::oneshot`: auth, roles, scopes,
//! audit rows, SQL, uploads through the queue, and local mode. No model is
//! configured, so the agent endpoints are exercised up to the point where
//! one would be called.

#![expect(
    clippy::indexing_slicing,
    reason = "serde_json::Value indexing yields Null for a missing key, never a panic"
)]
#![expect(
    clippy::too_many_lines,
    reason = "each test is one end-to-end scenario and reads best as one flow"
)]

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use quack_core::config::{BaseUrl, Config, ProviderConfig, ProviderName, ProviderType};
use quack_core::embedding::{Dimension, Vector};
use quack_core::ids::{ChunkId, DocumentId, RunId, SessionId, UserId, WorkspaceId};
use std::sync::Arc;
use tower::ServiceExt;

use super::state::{App, AppState, ServeMode, with_db};
use crate::server::auth::{Access, Credential, Identity};
use crate::server::queue::MAX_WAITING_UPLOADS;
use crate::server::run::{BackgroundRun, RunKind, RunReport};
use quack_core::jobs::LaneKey;
use quack_core::llm::CancellationToken;
use quack_core::okf::{Bundle, BundleFile};
use quack_core::storage::audit;
use quack_core::storage::control::{
    AuditFilter, AuditRow, Channel, ControlPlane, IssuedToken, Outcome, Role, Scope, UserKind,
};
use quack_core::storage::workspace::{DocumentStatus, NewChunk, NewDocument};

struct Harness {
    _dir: tempfile::TempDir,
    app: App,
    router: Router,
}

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

async fn harness(mode: ServeMode) -> Harness {
    harness_with(mode, Config::default()).await
}

async fn harness_with(mode: ServeMode, mut config: Config) -> Harness {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    config.general.data_dir = dir.path().to_path_buf();
    let control = ControlPlane::open(&config)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let app = Arc::new(AppState::new(config, control, mode));
    let router = super::router(Arc::clone(&app));
    Harness {
        _dir: dir,
        app,
        router,
    }
}

impl Harness {
    async fn send(
        &self,
        request: Request<Body>,
    ) -> (StatusCode, serde_json::Value, axum::http::HeaderMap) {
        let response = self
            .router
            .clone()
            .oneshot(request)
            .await
            .unwrap_or_else(|e| fail(&format!("request failed: {e}")));
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let value = serde_json::from_slice(&bytes).unwrap_or_else(|_| {
            serde_json::Value::String(String::from_utf8_lossy(&bytes).into_owned())
        });
        (status, value, headers)
    }

    async fn call(
        &self,
        method: Method,
        path: &str,
        bearer: Option<&str>,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, serde_json::Value) {
        let mut builder = Request::builder().method(method).uri(path);
        if let Some(token) = bearer {
            builder = builder.header(header::AUTHORIZATION, format!("Bearer {token}"));
        }
        let request = match body {
            Some(json) => builder
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json.to_string())),
            None => builder.body(Body::empty()),
        }
        .unwrap_or_else(|e| fail(&e.to_string()));
        let (status, value, _) = self.send(request).await;
        (status, value)
    }

    async fn get(&self, path: &str, bearer: &str) -> (StatusCode, serde_json::Value) {
        self.call(Method::GET, path, Some(bearer), None).await
    }

    async fn post(
        &self,
        path: &str,
        bearer: &str,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        self.call(Method::POST, path, Some(bearer), Some(body))
            .await
    }

    async fn user(&self, name: &str, kind: UserKind) -> UserId {
        self.app
            .control
            .create_user(name, "pw", kind)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
            .id
    }

    async fn login(&self, name: &str) -> String {
        let (status, body) = self
            .call(
                Method::POST,
                "/api/v1/auth/login",
                None,
                Some(serde_json::json!({ "username": name, "password": "pw" })),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body["token"].as_str().unwrap_or_default().to_owned()
    }

    async fn workspace(&self, name: &str, owner: &UserId) -> WorkspaceId {
        let ws = self
            .app
            .control
            .create_workspace(name)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        self.app
            .control
            .set_member(&ws.id, owner, Role::Owner)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        ws.id
    }

    async fn audit(&self, filter: AuditFilter) -> Vec<AuditRow> {
        self.app
            .control
            .query_audit(&AuditFilter {
                limit: 100,
                ..filter
            })
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
    }

    async fn wait_ready(&self, ws: &WorkspaceId, doc: &str, bearer: &str) -> serde_json::Value {
        for _ in 0..100 {
            let (status, body) = self
                .get(&format!("/api/v1/workspaces/{ws}/documents/{doc}"), bearer)
                .await;
            assert_eq!(status, StatusCode::OK, "{body}");
            match body["status"].as_str() {
                Some("ready" | "error") => return body,
                _ => tokio::time::sleep(std::time::Duration::from_millis(50)).await,
            }
        }
        fail("document never left the queue")
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn unauthenticated_requests_are_rejected() {
    let h = harness(ServeMode::Login).await;
    let (status, _) = h.call(Method::GET, "/api/v1/workspaces", None, None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, body) = h.call(Method::GET, "/healthz", None, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, "ok");
    let (status, _) = h.get("/api/v1/workspaces", "qk_not_a_token").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let denied = h
        .audit(AuditFilter {
            action: Some(String::from("token")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(denied.len(), 1);
    assert_eq!(denied.first().map(|r| r.outcome.as_str()), Some("denied"));
}

#[tokio::test(flavor = "multi_thread")]
async fn login_sets_a_cookie_and_audits_both_outcomes() {
    let h = harness(ServeMode::Login).await;
    let alice = h.user("alice", UserKind::Standard).await;
    let (status, _) = h
        .call(
            Method::POST,
            "/api/v1/auth/login",
            None,
            Some(serde_json::json!({ "username": "alice", "password": "nope" })),
        )
        .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let request = Request::builder()
        .method(Method::POST)
        .uri("/api/v1/auth/login")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({ "username": "alice", "password": "pw" }).to_string(),
        ))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, body, headers) = h.send(request).await;
    assert_eq!(status, StatusCode::OK);
    let cookie = headers
        .get(header::SET_COOKIE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    assert!(
        cookie.starts_with("quack_session=qs_") && cookie.contains("HttpOnly"),
        "{cookie}"
    );
    let token = body["token"].as_str().unwrap_or_default().to_owned();
    let (status, me) = h.get("/api/v1/auth/me", &token).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(me["username"], "alice");
    assert_eq!(me["via"], "session");
    let via_cookie = Request::builder()
        .uri("/api/v1/auth/me")
        .header(header::COOKIE, format!("quack_session={token}"))
        .body(Body::empty())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, _, _) = h.send(via_cookie).await;
    assert_eq!(status, StatusCode::OK);
    let logins = h
        .audit(AuditFilter {
            action: Some(String::from("login")),
            ..AuditFilter::default()
        })
        .await;
    let outcomes: Vec<&str> = logins.iter().map(|r| r.outcome.as_str()).collect();
    assert_eq!(outcomes, ["allowed", "denied"]);
    assert!(logins.iter().all(|r| r.user_id.as_ref() == Some(&alice)));
    let (status, _) = h
        .call(Method::POST, "/api/v1/auth/logout", Some(&token), None)
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = h.get("/api/v1/auth/me", &token).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test(flavor = "multi_thread")]
async fn workspaces_follow_membership_roles_and_admin_limits() {
    let h = harness(ServeMode::Login).await;
    h.user("root", UserKind::Admin).await;
    let bob = h.user("bob", UserKind::Standard).await;
    let carol = h.user("carol", UserKind::Standard).await;
    let root = h.login("root").await;
    let bob_token = h.login("bob").await;
    let carol_token = h.login("carol").await;

    let (status, _) = h
        .post(
            "/api/v1/workspaces",
            &bob_token,
            serde_json::json!({ "name": "team" }),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, body) = h
        .post(
            "/api/v1/workspaces",
            &root,
            serde_json::json!({ "name": "team" }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let ws = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    assert_eq!(body["role"], "owner");
    let (status, _) = h
        .post(
            "/api/v1/workspaces",
            &root,
            serde_json::json!({ "name": "team" }),
        )
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    let (status, _) = h
        .post(
            "/api/v1/workspaces",
            &root,
            serde_json::json!({ "name": "../x" }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Not a member: denied, and the denial is audited against the workspace.
    let (status, _) = h.get(&format!("/api/v1/workspaces/{ws}"), &bob_token).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let denied = h
        .audit(AuditFilter {
            user_id: Some(bob.clone()),
            outcome: Some(Outcome::Denied),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        denied
            .iter()
            .any(|r| r.workspace_id.as_ref() == Some(&ws) && r.channel == Channel::Web)
    );
    let (status, _) = h.get("/api/v1/workspaces/nope", &bob_token).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // The owner adds bob as viewer and carol as member.
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/members"),
            &root,
            serde_json::json!({ "username": "bob", "role": "viewer" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, _) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/members"),
            &root,
            serde_json::json!({ "username": "carol" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/members"),
            &bob_token,
            serde_json::json!({ "username": "carol", "role": "owner" }),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, body) = h.get("/api/v1/workspaces", &bob_token).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["workspaces"][0]["role"], "viewer");

    // Context: viewers read, members write; the editor is recorded.
    let (status, _) = h
        .call(
            Method::PUT,
            &format!("/api/v1/workspaces/{ws}/context"),
            Some(&bob_token),
            Some(serde_json::json!({ "content": "x" })),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, body) = h
        .call(
            Method::PUT,
            &format!("/api/v1/workspaces/{ws}/context"),
            Some(&carol_token),
            Some(serde_json::json!({ "content": "# Rules\nBe brief." })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["context"]["edited_by"], "carol");
    assert_eq!(body["context"]["version"], 1);
    let (status, body) = h
        .get(
            &format!("/api/v1/workspaces/{ws}/context/versions"),
            &bob_token,
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["versions"].as_array().map(Vec::len), Some(1));
    let markdown = Request::builder()
        .uri(format!("/api/v1/workspaces/{ws}/context"))
        .header(header::AUTHORIZATION, format!("Bearer {bob_token}"))
        .header(header::ACCEPT, "text/markdown")
        .body(Body::empty())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, body, headers) = h.send(markdown).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers
            .get(header::CONTENT_TYPE)
            .is_some_and(|c| c.to_str().unwrap_or_default().starts_with("text/markdown"))
    );
    assert_eq!(body, "# Rules\nBe brief.");

    // Settings: owner only; an admin without membership may too, but may not read content.
    let (status, body) = h
        .call(
            Method::PATCH,
            &format!("/api/v1/workspaces/{ws}"),
            Some(&carol_token),
            Some(serde_json::json!({ "classification": "secret" })),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
    let (status, body) = h
        .call(
            Method::PATCH,
            &format!("/api/v1/workspaces/{ws}"),
            Some(&root),
            Some(serde_json::json!({ "classification": "secret", "allowed_providers": ["nope"] })),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, body) = h
        .call(
            Method::PATCH,
            &format!("/api/v1/workspaces/{ws}"),
            Some(&root),
            Some(serde_json::json!({ "classification": "secret" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["classification"], "secret");
    let dave_root = h.user("dave", UserKind::Admin).await;
    let dave = h.login("dave").await;
    let (status, _) = h.get(&format!("/api/v1/workspaces/{ws}"), &dave).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = h
        .get(&format!("/api/v1/workspaces/{ws}/documents"), &dave)
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = h
        .get(&format!("/api/v1/workspaces/{ws}/members"), &dave)
        .await;
    assert_eq!(status, StatusCode::OK);
    let denied = h
        .audit(AuditFilter {
            user_id: Some(dave_root),
            outcome: Some(Outcome::Denied),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(denied.len(), 1);

    // Removing a member.
    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("/api/v1/workspaces/{ws}/members/{carol}"),
            Some(&root),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("/api/v1/workspaces/{ws}/members/{carol}"),
            Some(&root),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = h
        .get(&format!("/api/v1/workspaces/{ws}"), &carol_token)
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_respects_roles_hides_internal_tables_and_records_detail() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let viewer = h.user("viewer", UserKind::Standard).await;
    let ws = h.workspace("data", &owner).await;
    h.app
        .control
        .set_member(&ws, &viewer, Role::Viewer)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let owner_token = h.login("owner").await;
    let viewer_token = h.login("viewer").await;
    let sql = |s: &str| serde_json::json!({ "sql": s });

    let (status, _) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &viewer_token,
            sql("CREATE TABLE t AS SELECT 1 AS a"),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &owner_token,
            sql("CREATE TABLE t AS SELECT 1 AS a, 'x' AS b"),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &viewer_token,
            sql("SELECT a, b FROM t"),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["columns"], serde_json::json!(["a", "b"]));
    assert_eq!(body["rows"][0][0], 1);
    assert_eq!(body["row_count"], 1);
    assert_eq!(body["truncated"], false);
    let (status, _) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &viewer_token,
            sql("SELECT * FROM _quack_documents"),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &viewer_token,
            sql("SELEC nope"),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &viewer_token,
            sql("SELECT * FROM missing"),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

    let (status, body) = h
        .get(&format!("/api/v1/workspaces/{ws}/tables"), &viewer_token)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["tables"], serde_json::json!(["t"]));
    let (status, body) = h
        .get(&format!("/api/v1/workspaces/{ws}/tables/t"), &viewer_token)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["columns"][0]["name"], "a");
    assert_eq!(body["sample"]["rows"].as_array().map(Vec::len), Some(1));
    let (status, _) = h
        .get(
            &format!("/api/v1/workspaces/{ws}/tables/_quack_documents"),
            &viewer_token,
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Detail lives in the workspace; the access row in control.db carries no SQL.
    let (status, body) = h
        .get(&format!("/api/v1/workspaces/{ws}/audit"), &viewer_token)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let details = body["audit"].as_array().cloned().unwrap_or_default();
    assert!(
        details
            .iter()
            .any(|d| d["action"] == "sql" && d["detail"]["sql"] == "SELECT a, b FROM t"),
        "{body}"
    );
    let access_rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("sql")),
            ..AuditFilter::default()
        })
        .await;
    assert!(access_rows.len() >= 4);
    assert!(access_rows.iter().all(|r| r.resource_id.is_none()));
    assert!(
        access_rows
            .iter()
            .any(|r| r.outcome == Outcome::Denied && r.user_id.as_ref() == Some(&viewer))
    );
    assert!(access_rows.iter().any(|r| r.outcome == Outcome::Error));
    assert!(access_rows.iter().all(|r| r.request_id.is_some()));
}

/// The obvious `CREATE TEMP TABLE` case never reaches the writer: `POST
/// /sql` refuses it up front, the same as `run_sql`.
#[tokio::test]
async fn sql_refuses_to_create_a_temp_table() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("data", &owner).await;
    let owner_token = h.login("owner").await;
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &owner_token,
            serde_json::json!({ "sql": "CREATE TEMP TABLE scratch AS SELECT 1 AS a" }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

/// A leading comment defeats the text-based guard, so the statement runs
/// and creates a temp table on the writer — the sticky degrade
/// (`ReaderDb::observe_write`) is what keeps a later read from 422ing with
/// a Catalog Error, proven end to end through the real HTTP routes.
#[tokio::test]
async fn sql_bypass_temp_tables_are_still_visible_after_the_reader_degrades() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("data", &owner).await;
    let owner_token = h.login("owner").await;
    let sql = |s: &str| serde_json::json!({ "sql": s });

    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &owner_token,
            sql("-- scratch\nCREATE TEMP TABLE scratch AS SELECT 1 AS a"),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // Reads route through the reader; without the sticky degrade this
    // would 422 with a Catalog Error instead of seeing the row.
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &owner_token,
            sql("SELECT * FROM scratch"),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["rows"][0][0], 1);
}

/// A read handler's database work runs on a reader connection and its
/// audit detail on the audit connection: the request answers while the
/// writer is held by a long write, and a write through a read is refused
/// by the read-only transaction.
#[tokio::test(flavor = "multi_thread")]
async fn read_requests_never_wait_for_the_writer() {
    use quack_core::storage::workspace::WorkspaceDb;

    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("data", &owner).await;
    let writer = h
        .app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    writer
        .run(|db| db.execute_statement("CREATE TABLE t AS SELECT 1 AS a"))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));

    // Hold the writer busy, as a large load would.
    let (release, hold) = std::sync::mpsc::channel::<()>();
    let busy = tokio::spawn(async move {
        writer
            .run(move |_| {
                hold.recv().ok();
                Ok(())
            })
            .await
    });
    let tables = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        h.app.read(&ws, WorkspaceDb::list_tables),
    )
    .await;
    assert!(
        tables.is_ok_and(|r| r.is_ok_and(|t| t == vec![String::from("t")])),
        "a read waited for the writer"
    );
    // The whole request too: its audit detail goes to the audit
    // connection, not the busy writer.
    let token = h.login("owner").await;
    let listed = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        h.get(&format!("/api/v1/workspaces/{ws}/tables"), &token),
    )
    .await;
    assert!(
        listed.is_ok_and(|(status, body)| status == StatusCode::OK && body["tables"][0] == "t"),
        "the request waited for the writer"
    );
    let refused = h
        .app
        .read(&ws, |db| db.execute_statement("CREATE TABLE u (a INTEGER)"))
        .await;
    assert!(refused.is_err(), "a write ran inside a read");
    assert!(release.send(()).is_ok());
    assert!(busy.await.is_ok_and(|r| r.is_ok()));
}

fn multipart(filename: &str, content_type: &str, data: &str) -> (String, Vec<u8>) {
    let boundary = "quackboundary";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n{data}\r\n--{boundary}--\r\n"
    );
    (
        format!("multipart/form-data; boundary={boundary}"),
        body.into_bytes(),
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn uploads_are_queued_processed_pinned_and_deleted() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("docs", &owner).await;
    let token = h.login("owner").await;
    let base = format!("/api/v1/workspaces/{ws}/documents");

    let (status, body) = h
        .post(
            &base,
            &token,
            serde_json::json!({ "text": "Flood damage is excluded.", "title": "policy" }),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let text_id = body["documents"][0]["id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert_eq!(body["documents"][0]["filename"], "policy.md");
    let ready = h.wait_ready(&ws, &text_id, &token).await;
    assert_eq!(ready["status"], "ready", "{ready}");
    assert_eq!(ready["source"], "paste");
    assert_eq!(ready["title"], "policy");
    assert_eq!(
        ready["ingested_by"].as_str().map(str::is_empty),
        Some(false)
    );
    assert_eq!(ready["sha256"].as_str().map(str::len), Some(64));

    // The same text again is not queued a second time.
    let (status, body) = h
        .post(
            &base,
            &token,
            serde_json::json!({ "text": "Flood damage is excluded.", "title": "again" }),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(body["documents"][0]["status"], "duplicate");
    assert_eq!(body["documents"][0]["id"], text_id);
    assert_eq!(body["documents"][0]["existing_filename"], "policy.md");

    let (content_type, bytes) = multipart(
        "sales.csv",
        "text/csv",
        "region,total\nnorth,10\nsouth,20\n",
    );
    let request = Request::builder()
        .method(Method::POST)
        .uri(&base)
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from(bytes))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, body, _) = h.send(request).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let csv_id = body["documents"][0]["id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let ready = h.wait_ready(&ws, &csv_id, &token).await;
    assert_eq!(ready["status"], "ready", "{ready}");
    let (_, body) = h
        .get(&format!("/api/v1/workspaces/{ws}/tables"), &token)
        .await;
    assert_eq!(body["tables"], serde_json::json!(["sales"]));

    let (status, body) = h.get(&base, &token).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["documents"].as_array().map(Vec::len), Some(2));
    assert_eq!(ready["source"], "upload");
    assert_eq!(ready["title"], serde_json::Value::Null);
    let (status, body) = h
        .call(
            Method::PATCH,
            &format!("{base}/{text_id}"),
            Some(&token),
            Some(serde_json::json!({ "pinned": true })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["pinned"], true);

    let (status, body) = h
        .get(
            &format!("/api/v1/workspaces/{ws}/search?query=flood"),
            &token,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["chunks"][0]["filename"], "policy.md");

    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("{base}/{csv_id}"),
            Some(&token),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("{base}/{csv_id}"),
            Some(&token),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (_, body) = h
        .get(&format!("/api/v1/workspaces/{ws}/tables"), &token)
        .await;
    assert_eq!(body["tables"], serde_json::json!([]));

    let (status, _) = h
        .post(&base, &token, serde_json::json!({ "text": "   " }))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (content_type, bytes) = multipart("blob.xyz", "application/octet-stream", "??");
    let request = Request::builder()
        .method(Method::POST)
        .uri(&base)
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from(bytes))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, _, _) = h.send(request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let ingests = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("ingest")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(ingests.len(), 3, "the duplicate is audited too");
    let deletes = h
        .audit(AuditFilter {
            workspace_id: Some(ws),
            action: Some(String::from("delete")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(
        deletes.first().and_then(|r| r.resource_id.clone()),
        Some(csv_id)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn api_tokens_are_scoped_to_one_workspace_and_expire() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("a", &owner).await;
    let other = h.workspace("b", &owner).await;
    let read_token = h
        .app
        .control
        .create_token(&ws, &owner, "ro", &[Scope::Read], None)
        .await
        .map_or_else(
            |e| fail(&e.to_string()),
            |issued| issued.secret.expose().to_owned(),
        );
    let (status, body) = h
        .get(&format!("/api/v1/workspaces/{ws}/documents"), &read_token)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = h.get("/api/v1/auth/me", &read_token).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["via"], "token");
    let (status, body) = h.get("/api/v1/workspaces", &read_token).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["workspaces"].as_array().map(Vec::len), Some(1));
    let (status, _) = h
        .call(
            Method::PUT,
            &format!("/api/v1/workspaces/{ws}/context"),
            Some(&read_token),
            Some(serde_json::json!({ "content": "x" })),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = h
        .get(
            &format!("/api/v1/workspaces/{other}/documents"),
            &read_token,
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = h.get("/api/v1/admin/users", &read_token).await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    let IssuedToken {
        secret: write_token,
        row,
    } = h
        .app
        .control
        .create_token(&ws, &owner, "rw", &[Scope::Write], None)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let write_token = write_token.expose();
    let (status, _) = h
        .call(
            Method::PUT,
            &format!("/api/v1/workspaces/{ws}/context"),
            Some(write_token),
            Some(serde_json::json!({ "content": "x" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let touched = h
        .app
        .control
        .find_token(&row.token_hash)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(touched.is_some_and(|t| t.last_used_at.is_some()));
    let api_rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("context")),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        api_rows
            .iter()
            .all(|r| r.channel == Channel::Api && r.token_hash.is_some())
    );

    let expired = h
        .app
        .control
        .create_token(
            &ws,
            &owner,
            "old",
            &[Scope::Read],
            "2000-01-01 00:00:00".parse().ok(),
        )
        .await
        .map_or_else(
            |e| fail(&e.to_string()),
            |issued| issued.secret.expose().to_owned(),
        );
    let (status, _) = h
        .get(&format!("/api/v1/workspaces/{ws}/documents"), &expired)
        .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let denied = h
        .audit(AuditFilter {
            action: Some(String::from("token")),
            outcome: Some(Outcome::Denied),
            ..AuditFilter::default()
        })
        .await;
    assert!(denied.iter().any(|r| r.user_id.as_ref() == Some(&owner)));
}

#[tokio::test(flavor = "multi_thread")]
async fn query_endpoints_fail_cleanly_without_a_chat_model() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let viewer = h.user("viewer", UserKind::Standard).await;
    let ws = h.workspace("q", &owner).await;
    h.app
        .control
        .set_member(&ws, &viewer, Role::Viewer)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let owner_token = h.login("owner").await;
    let viewer_token = h.login("viewer").await;
    let path = format!("/api/v1/workspaces/{ws}/query");
    let (status, body) = h
        .post(
            &path,
            &viewer_token,
            serde_json::json!({ "prompt": "hi", "allow_write": true }),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
    let (status, body) = h
        .post(&path, &owner_token, serde_json::json!({ "prompt": "   " }))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, body) = h
        .post(&path, &owner_token, serde_json::json!({ "prompt": "hi" }))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("chat_model")
    );
    let (status, body) = h
        .post(
            &format!("{path}/stream"),
            &owner_token,
            serde_json::json!({ "prompt": "hi", "mode": "sideways" }),
        )
        .await;
    // An unknown mode is refused while the body is read, with the modes it
    // accepts.
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert!(
        body.to_string().contains("expected `chat` or `query`"),
        "{body}"
    );
    let (status, body) = h
        .get(&format!("/api/v1/workspaces/{ws}/sessions"), &owner_token)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["sessions"], serde_json::json!([]));
    let (status, _) = h
        .get(
            &format!("/api/v1/workspaces/{ws}/sessions/nope"),
            &owner_token,
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = h
        .get(
            &format!("/api/v1/workspaces/{ws}/search?query="),
            &owner_token,
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failed_first_turn_leaves_no_empty_session_behind() {
    // A chat model whose provider is unreachable: the turn fails after the
    // session was created.
    let config = Config::parse(
        "[general]\nchat_model = \"o/m\"\n[providers.o]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\n",
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    let h = harness_with(ServeMode::Local, config).await;
    let (status, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "w" })),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let ws = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    let (status, body) = h
        .call(
            Method::POST,
            &format!("/api/v1/workspaces/{ws}/query"),
            None,
            Some(serde_json::json!({ "prompt": "hi" })),
        )
        .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    let (_, body) = h
        .call(
            Method::GET,
            &format!("/api/v1/workspaces/{ws}/sessions"),
            None,
            None,
        )
        .await;
    assert_eq!(body["sessions"], serde_json::json!([]), "{body}");
    let errors = h
        .audit(AuditFilter {
            workspace_id: Some(ws),
            action: Some(String::from("query")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(errors.first().map(|r| r.outcome.as_str()), Some("error"));
}

#[tokio::test(flavor = "multi_thread")]
async fn sessions_are_deleted_by_their_creator_or_an_owner() {
    use quack_core::storage::sessions::{ChatMode, create_session};
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let viewer = h.user("viewer", UserKind::Standard).await;
    let other = h.user("other", UserKind::Standard).await;
    let ws = h.workspace("s", &owner).await;
    for u in [&viewer, &other] {
        h.app
            .control
            .set_member(&ws, u, Role::Viewer)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    let db = h
        .app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    let (mine, theirs) = {
        let (viewer, other) = (viewer.clone(), other.clone());
        db.run(move |db| {
            let mine = create_session(db, "m", ChatMode::Chat, Some(&viewer))?;
            let theirs = create_session(db, "m", ChatMode::Chat, Some(&other))?;
            Ok((mine.id, theirs.id))
        })
        .await
        .unwrap_or_else(|e| fail(&e.to_string()))
    };
    let viewer_token = h.login("viewer").await;
    let owner_token = h.login("owner").await;
    let other_token = h.login("other").await;
    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("/api/v1/workspaces/{ws}/sessions/{theirs}"),
            Some(&viewer_token),
            None,
        )
        .await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "another user's session is invisible"
    );

    // Shared by its creator, the session becomes visible to other members
    // but stays theirs to delete; unshared, it disappears again.
    let share = format!("/api/v1/workspaces/{ws}/sessions/{theirs}");
    let (status, _) = h
        .call(
            Method::PATCH,
            &share,
            Some(&viewer_token),
            Some(serde_json::json!({ "shared": true })),
        )
        .await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "cannot share what one cannot see"
    );
    let (status, body) = h
        .call(
            Method::PATCH,
            &share,
            Some(&other_token),
            Some(serde_json::json!({ "shared": true })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["shared"], true);
    let (status, body) = h.get(&share, &viewer_token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["session"]["shared"], true);
    let (status, _) = h
        .call(
            Method::PATCH,
            &share,
            Some(&viewer_token),
            Some(serde_json::json!({ "shared": false })),
        )
        .await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "a viewer of a shared session cannot unshare it"
    );
    // The mode changes only through an explicit PATCH by the creator or
    // an owner (issue #57); a viewer of a shared session cannot.
    let (status, body) = h
        .call(
            Method::PATCH,
            &share,
            Some(&other_token),
            Some(serde_json::json!({ "mode": "query" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["mode"], "query");
    let (status, _) = h
        .call(
            Method::PATCH,
            &share,
            Some(&viewer_token),
            Some(serde_json::json!({ "mode": "chat" })),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = h
        .call(
            Method::PATCH,
            &share,
            Some(&other_token),
            Some(serde_json::json!({ "mode": "loud" })),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (status, _) = h
        .call(
            Method::PATCH,
            &share,
            Some(&other_token),
            Some(serde_json::json!({})),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = h
        .call(Method::DELETE, &share, Some(&viewer_token), None)
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = h
        .call(
            Method::PATCH,
            &share,
            Some(&other_token),
            Some(serde_json::json!({ "shared": false })),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = h.get(&share, &viewer_token).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let shares = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("share")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(shares.len(), 3, "two allowed and one denied");
    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("/api/v1/workspaces/{ws}/sessions/{mine}"),
            Some(&viewer_token),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("/api/v1/workspaces/{ws}/sessions/{theirs}"),
            Some(&owner_token),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("/api/v1/workspaces/{ws}/sessions/{theirs}"),
            Some(&owner_token),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (_, body) = h
        .get(&format!("/api/v1/workspaces/{ws}/sessions"), &owner_token)
        .await;
    assert_eq!(body["sessions"], serde_json::json!([]));
    let deletes = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("delete")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(deletes.len(), 3, "two allowed and the viewer's denied one");
    assert_eq!(
        deletes
            .iter()
            .filter(|r| r.outcome == Outcome::Denied)
            .count(),
        1
    );
    assert!(
        deletes
            .iter()
            .all(|r| r.resource_type.as_deref() == Some("session"))
    );

    // The web button redirects back to the chat page; a missing session is a 404 page.
    let (_, _, headers) = h.form("/login", None, "username=owner&password=pw").await;
    let cookie = headers
        .get(header::SET_COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|c| c.split(';').next())
        .and_then(|c| c.strip_prefix("quack_session="))
        .unwrap_or_default()
        .to_owned();
    let (status, _, _) = h
        .form(&format!("/w/{ws}/chat/nope/delete"), Some(&cookie), "")
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let fresh = {
        let owner = owner.clone();
        db.run(move |db| create_session(db, "m", ChatMode::Chat, Some(&owner)))
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
            .id
    };
    let (status, _, headers) = h
        .form(&format!("/w/{ws}/chat/{fresh}/delete"), Some(&cookie), "")
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(location(&headers), format!("/w/{ws}/chat"));
}

/// A record named by id that does not exist is a 404 wherever it is
/// named, and a missing ontology version or merge proposal no longer
/// borrows that status for every other failure.
#[tokio::test(flavor = "multi_thread")]
async fn missing_records_answer_404() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("m", &owner).await;
    let token = h.login("owner").await;
    let base = format!("/api/v1/workspaces/{ws}");
    for (method, path, body) in [
        (
            Method::PATCH,
            format!("{base}/documents/nope"),
            serde_json::json!({ "pinned": true }),
        ),
        (
            Method::PATCH,
            format!("{base}/sessions/nope"),
            serde_json::json!({ "shared": true }),
        ),
        (
            Method::POST,
            format!("{base}/ontology/versions/99/restore"),
            serde_json::json!({}),
        ),
        (
            Method::PUT,
            format!("{base}/graph/merges/nope"),
            serde_json::json!({ "action": "reject" }),
        ),
        (
            Method::PUT,
            format!("{base}/ontology/candidates/nope"),
            serde_json::json!({ "action": "reject" }),
        ),
    ] {
        let (status, body) = h.call(method, &path, Some(&token), Some(body)).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{path}: {body}");
        assert!(
            body.to_string().contains("does not exist"),
            "{path}: {body}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn ontology_is_versioned_over_the_api_and_the_web_page() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let viewer = h.user("viewer", UserKind::Standard).await;
    let ws = h.workspace("o", &owner).await;
    h.app
        .control
        .set_member(&ws, &viewer, Role::Viewer)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let owner_token = h.login("owner").await;
    let viewer_token = h.login("viewer").await;
    let base = format!("/api/v1/workspaces/{ws}/ontology");

    let (status, _) = h.get(&base, &viewer_token).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = h
        .post(
            &format!("{base}/init"),
            &viewer_token,
            serde_json::json!({}),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, body) = h
        .post(&format!("{base}/init"), &owner_token, serde_json::json!({}))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["version"], 1);
    assert_eq!(body["classes"].as_array().map(Vec::len), Some(7));
    let (status, _) = h
        .post(&format!("{base}/init"), &owner_token, serde_json::json!({}))
        .await;
    assert_eq!(status, StatusCode::CONFLICT);

    let mut edited = body.clone();
    if let Some(classes) = edited["classes"].as_array_mut() {
        classes.push(serde_json::json!({ "id": "vendor", "parent": "organization" }));
    }
    let (status, body) = h
        .call(Method::PUT, &base, Some(&owner_token), Some(edited.clone()))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["version"], 2);
    let mut broken = edited.clone();
    if let Some(classes) = broken["classes"].as_array_mut() {
        classes.push(serde_json::json!({ "id": "Bad Id" }));
    }
    let (status, body) = h
        .call(Method::PUT, &base, Some(&owner_token), Some(broken))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("snake_case")
    );
    let (status, _) = h
        .call(Method::PUT, &base, Some(&viewer_token), Some(edited))
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    let (status, body) = h.get(&format!("{base}/versions"), &viewer_token).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["versions"].as_array().map(Vec::len), Some(2));
    assert_eq!(body["versions"][0]["author"], "owner");
    let (status, body) = h.get(&format!("{base}/versions/2"), &viewer_token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["diff"]["classes"]["added"],
        serde_json::json!(["vendor"])
    );
    let (status, _) = h.get(&format!("{base}/versions/9"), &viewer_token).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, body) = h
        .post(
            &format!("{base}/versions/1/restore"),
            &owner_token,
            serde_json::json!({}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["version"], 3);
    assert!(
        body["classes"]
            .as_array()
            .is_some_and(|c| !c.iter().any(|x| x["id"] == "vendor"))
    );
    let writes = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("ontology")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(writes.len(), 3);

    // The web page renders the tree and the version list.
    let (_, _, headers) = h.form("/login", None, "username=owner&password=pw").await;
    let cookie = headers
        .get(header::SET_COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|c| c.split(';').next())
        .and_then(|c| c.strip_prefix("quack_session="))
        .unwrap_or_default()
        .to_owned();
    let (status, html, _) = h.page(&format!("/w/{ws}/ontology"), Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains("works_at: person → organization") && html.contains("v3"),
        "{html}"
    );
    assert!(html.contains("restored version 1"), "{html}");
    let (status, _, headers) = h
        .form(
            &format!("/w/{ws}/ontology"),
            Some(&cookie),
            "json=%7B%22classes%22%3A%5B%7B%22id%22%3A%22Bad%22%7D%5D%7D",
        )
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert!(
        location(&headers).contains("error="),
        "{}",
        location(&headers)
    );
    let (status, _, headers) = h
        .form(&format!("/w/{ws}/ontology/2/restore"), Some(&cookie), "")
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(location(&headers), format!("/w/{ws}/ontology"));
    let (status, html, _) = h.page(&format!("/w/{ws}/ontology"), Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("vendor") && html.contains("v4"), "{html}");
}

#[tokio::test(flavor = "multi_thread")]
async fn ontology_proposals_are_reviewed_over_the_api_and_the_page() {
    let h = harness(ServeMode::Local).await;
    let (_, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "p" })),
        )
        .await;
    let ws = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    for sql in [
        "CREATE TABLE vendors (vendor_id INTEGER, name TEXT)",
        "INSERT INTO vendors SELECT i, 'V' || i FROM range(30) t(i)",
        "CREATE TABLE orders (order_id INTEGER, vendor_id INTEGER, mode TEXT)",
        "INSERT INTO orders SELECT i, i % 30, CASE WHEN i % 2 = 0 THEN 'air' ELSE 'sea' END FROM range(60) t(i)",
    ] {
        let (status, body) = h
            .call(
                Method::POST,
                &format!("/api/v1/workspaces/{ws}/sql"),
                None,
                Some(serde_json::json!({ "sql": sql })),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    let base = format!("/api/v1/workspaces/{ws}/ontology");
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{base}/propose"),
            None,
            Some(serde_json::json!({})),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["candidates"], 10, "{body}");
    let (status, body) = h
        .call(Method::GET, &format!("{base}/candidates"), None, None)
        .await;
    assert_eq!(status, StatusCode::OK);
    let queue = body["candidates"].as_array().cloned().unwrap_or_default();
    assert_eq!(queue.len(), 10);
    let relation = queue
        .iter()
        .find(|c| c["kind"] == "relation")
        .unwrap_or_else(|| fail("no relation"));
    assert_eq!(relation["proposal"]["domain"], "order");
    let mode = queue
        .iter()
        .find(|c| c["kind"] == "property" && c["proposal"]["property"]["id"] == "mode")
        .unwrap_or_else(|| fail("no mode"));
    assert_eq!(mode["proposal"]["property"]["type"], "enum");

    let rid = relation["id"].as_str().unwrap_or_default().to_owned();
    let (status, body) = h
        .call(
            Method::PUT,
            &format!("{base}/candidates/{rid}"),
            None,
            Some(serde_json::json!({ "action": "rename", "target": "placed_with" })),
        )
        .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "a relation alone cannot validate before its classes: {body}"
    );
    let (status, _) = h
        .call(
            Method::PUT,
            &format!(
                "{base}/candidates/{}",
                mode["id"].as_str().unwrap_or_default()
            ),
            None,
            Some(serde_json::json!({ "action": "reject" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = h
        .call(
            Method::PUT,
            &format!("{base}/candidates/{rid}"),
            None,
            Some(serde_json::json!({ "action": "rename" })),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "rename needs a target");
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{base}/propose"),
            None,
            Some(serde_json::json!({ "mode": "sideways" })),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{base}/propose"),
            None,
            Some(serde_json::json!({ "documents": true })),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "no chat model: {body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("chat_model")
    );

    // Accept the rest one by one in queue order (classes, properties,
    // relations, mappings), each write a version; the rejected one stays out.
    let (_, body) = h
        .call(Method::GET, &format!("{base}/candidates"), None, None)
        .await;
    let remaining = body["candidates"].as_array().cloned().unwrap_or_default();
    assert_eq!(remaining.len(), 9);
    for c in &remaining {
        let cid = c["id"].as_str().unwrap_or_default();
        let (status, body) = h
            .call(
                Method::PUT,
                &format!("{base}/candidates/{cid}"),
                None,
                Some(serde_json::json!({ "action": "accept" })),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    let (status, body) = h.call(Method::GET, &base, None, None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["version"], 9);
    let classes: Vec<&str> = body["classes"]
        .as_array()
        .map(|c| c.iter().filter_map(|x| x["id"].as_str()).collect())
        .unwrap_or_default();
    assert_eq!(classes, ["order", "vendor"]);
    assert!(
        body["properties"]
            .as_array()
            .is_some_and(|p| !p.iter().any(|x| x["id"] == "mode")),
        "rejected stays out: {body}"
    );
    assert_eq!(body["mappings"].as_array().map(Vec::len), Some(2));
    let (status, body) = h
        .call(Method::GET, &format!("{base}/candidates"), None, None)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["candidates"], serde_json::json!([]));
    // Propose has one behavior: what the current ontology lacks. A caller
    // asking for the old "full" mode is told so rather than silently served.
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{base}/propose"),
            None,
            Some(serde_json::json!({ "mode": "full" })),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body.to_string()
            .contains("only what the current ontology lacks"),
        "{body}"
    );
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{base}/propose"),
            None,
            Some(serde_json::json!({})),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["candidates"], 1,
        "only the rejected property is missing now: {body}"
    );
    let (status, html, _) = h.page(&format!("/w/{ws}/ontology"), None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains("1 pending")
            && html.contains("orders.mode")
            && html.contains("Accept selected"),
        "{html}"
    );
    // Bulk decisions (issue #55): the API rejects the candidate by id in
    // one call, the page's bulk form accepts the re-proposed one.
    let (_, body) = h
        .call(Method::GET, &format!("{base}/candidates"), None, None)
        .await;
    let pending_id = body["candidates"][0]["id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{base}/candidates"),
            None,
            Some(serde_json::json!({ "reject": [pending_id] })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["rejected"], 1);
    let (status, _) = h
        .call(
            Method::POST,
            &format!("{base}/candidates"),
            None,
            Some(serde_json::json!({})),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (_, body) = h
        .call(
            Method::POST,
            &format!("{base}/propose"),
            None,
            Some(serde_json::json!({ "mode": "extend" })),
        )
        .await;
    assert_eq!(body["candidates"], 1, "{body}");
    let (_, body) = h
        .call(Method::GET, &format!("{base}/candidates"), None, None)
        .await;
    let pending_id = body["candidates"][0]["id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let (status, _, headers) = h
        .form(
            &format!("/w/{ws}/ontology/candidates"),
            None,
            &format!("bulk=accept&status=pending&ids={pending_id}"),
        )
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(location(&headers), format!("/w/{ws}/ontology"));
    let (_, body) = h.call(Method::GET, &base, None, None).await;
    assert_eq!(body["version"], 10);
    assert!(
        body["properties"]
            .as_array()
            .is_some_and(|p| p.iter().any(|x| x["id"] == "mode"))
    );
    let proposes = h
        .audit(AuditFilter {
            workspace_id: Some(ws),
            action: Some(String::from("propose")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(proposes.len(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn admin_endpoints_manage_users_and_read_the_audit() {
    let h = harness(ServeMode::Login).await;
    h.user("root", UserKind::Admin).await;
    h.user("bob", UserKind::Standard).await;
    let root = h.login("root").await;
    let bob = h.login("bob").await;
    let (status, _) = h.get("/api/v1/admin/users", &bob).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, body) = h
        .post(
            "/api/v1/admin/users",
            &root,
            serde_json::json!({ "username": "eve", "password": "pw" }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let (status, _) = h
        .post(
            "/api/v1/admin/users",
            &root,
            serde_json::json!({ "username": "eve", "password": "pw" }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, body) = h.get("/api/v1/admin/users", &root).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["users"].as_array().map(Vec::len), Some(3));
    let (status, body) = h.get("/api/v1/admin/audit?action=admin", &root).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["audit"][0]["resource_type"], "user");
    let (status, _) = h.get("/api/v1/admin/audit", &bob).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}

#[tokio::test(flavor = "multi_thread")]
async fn local_mode_needs_no_login_and_owns_everything() {
    let h = harness(ServeMode::Local).await;
    let (status, _) = h.call(Method::GET, "/api/v1/workspaces", None, None).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = h
        .call(
            Method::POST,
            "/api/v1/auth/login",
            None,
            Some(serde_json::json!({ "username": "a", "password": "b" })),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "mine" })),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let ws = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    let (status, body) = h
        .call(
            Method::POST,
            &format!("/api/v1/workspaces/{ws}/sql"),
            None,
            Some(serde_json::json!({ "sql": "CREATE TABLE t AS SELECT 1 AS a" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = h.call(Method::GET, "/api/v1/auth/me", None, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["via"], "local");
    let rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        rows.iter()
            .all(|r| r.user_id == Some(UserId::from("local")))
    );
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn responses_are_not_cached_unless_the_handler_sets_a_policy() {
    fn assert_no_store(what: &str, headers: &axum::http::HeaderMap) {
        let get = |name| headers.get(name).and_then(|v| v.to_str().ok());
        assert_eq!(
            get(header::CACHE_CONTROL),
            Some("no-cache, no-store, must-revalidate"),
            "{what}"
        );
        assert_eq!(get(header::EXPIRES), Some("0"), "{what}");
        assert_eq!(get(header::PRAGMA), Some("no-cache"), "{what}");
    }
    let h = harness(ServeMode::Login).await;
    let alice = h.user("alice", UserKind::Standard).await;
    let ws = h.workspace("docs", &alice).await;

    // Login sets the session cookie; its response is not kept either.
    let login = Request::builder()
        .method(Method::POST)
        .uri("/api/v1/auth/login")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(r#"{"username":"alice","password":"pw"}"#))
        .unwrap();
    let (status, body, headers) = h.send(login).await;
    assert_eq!(status, StatusCode::OK);
    assert_no_store("login", &headers);
    let token = body["token"].as_str().unwrap_or_default().to_owned();

    for (what, path, bearer) in [
        (
            "api answer",
            format!("/api/v1/workspaces/{ws}"),
            Some(&token),
        ),
        ("api denial", format!("/api/v1/workspaces/{ws}"), None),
        ("web page", format!("/w/{ws}/settings"), None),
        ("mcp", format!("/mcp/v1/{ws}"), Some(&token)),
    ] {
        let mut request = Request::builder().uri(path);
        if let Some(token) = bearer {
            request = request.header(header::AUTHORIZATION, format!("Bearer {token}"));
        }
        let (_, _, headers) = h.send(request.body(Body::empty()).unwrap()).await;
        assert_no_store(what, &headers);
    }

    // Public assets keep their own revalidate-by-ETag policy, and the health
    // check sits outside the layer.
    let (status, _, headers) = h.page("/static/css/output.css", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        headers.get(header::CACHE_CONTROL).unwrap(),
        "no-cache",
        "{headers:?}"
    );
    assert!(headers.get(header::EXPIRES).is_none());
    assert!(headers.get(header::PRAGMA).is_none());
    let (status, _, headers) = h.page("/healthz", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(headers.get(header::CACHE_CONTROL).is_none());
}

// --- web UI ------------------------------------------------------------------

impl Harness {
    async fn page(
        &self,
        path: &str,
        cookie: Option<&str>,
    ) -> (StatusCode, String, axum::http::HeaderMap) {
        let mut builder = Request::builder().uri(path);
        if let Some(token) = cookie {
            builder = builder.header(header::COOKIE, format!("quack_session={token}"));
        }
        let request = builder
            .body(Body::empty())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let (status, body, headers) = self.send(request).await;
        (
            status,
            body.as_str().unwrap_or_default().to_owned(),
            headers,
        )
    }

    /// A form POST that arrives from `peer`, so the handler sees the same
    /// `ConnectInfo` the real server sets.
    async fn form_from(
        &self,
        path: &str,
        peer: &str,
        form: &str,
    ) -> (StatusCode, String, axum::http::HeaderMap) {
        let addr: std::net::SocketAddr = peer.parse().unwrap_or_else(|e| fail(&format!("{e}")));
        let mut request = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(Body::from(form.to_owned()))
            .unwrap_or_else(|e| fail(&e.to_string()));
        request
            .extensions_mut()
            .insert(axum::extract::ConnectInfo(addr));
        let (status, body, headers) = self.send(request).await;
        (
            status,
            body.as_str().unwrap_or_default().to_owned(),
            headers,
        )
    }

    async fn form(
        &self,
        path: &str,
        cookie: Option<&str>,
        form: &str,
    ) -> (StatusCode, String, axum::http::HeaderMap) {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded");
        if let Some(token) = cookie {
            builder = builder.header(header::COOKIE, format!("quack_session={token}"));
        }
        let request = builder
            .body(Body::from(form.to_owned()))
            .unwrap_or_else(|e| fail(&e.to_string()));
        let (status, body, headers) = self.send(request).await;
        (
            status,
            body.as_str().unwrap_or_default().to_owned(),
            headers,
        )
    }
}

/// Every `Set-Cookie` on a response, joined, so one assertion can look for
/// an attribute or its absence.
fn set_cookie(headers: &axum::http::HeaderMap) -> String {
    headers
        .get_all(header::SET_COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .collect::<Vec<_>>()
        .join("; ")
}

fn location(headers: &axum::http::HeaderMap) -> String {
    headers
        .get(header::LOCATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned()
}

#[tokio::test(flavor = "multi_thread")]
async fn web_pages_redirect_to_login_and_render_after_the_form_login() {
    let h = harness(ServeMode::Login).await;
    h.user("root", UserKind::Admin).await;
    let (status, _, headers) = h.page("/workspaces", None).await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(location(&headers), "/login");
    let (status, html, _) = h.page("/login", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains("<form method=\"post\" action=\"/login\""),
        "{html}"
    );
    let (status, _, headers) = h.form("/login", None, "username=root&password=wrong").await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert!(location(&headers).starts_with("/login?error="));
    let (status, _, headers) = h.form("/login", None, "username=root&password=pw").await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(location(&headers), "/workspaces");
    let cookie = headers
        .get(header::SET_COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|c| c.split(';').next())
        .and_then(|c| c.strip_prefix("quack_session="))
        .unwrap_or_default()
        .to_owned();
    assert!(cookie.starts_with("qs_"));

    let (status, _, headers) = h.form("/workspaces", Some(&cookie), "name=team").await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    let chat_url = location(&headers);
    assert!(chat_url.ends_with("/chat"), "{chat_url}");
    let ws = WorkspaceId::from(chat_url.trim_start_matches("/w/").trim_end_matches("/chat"));

    let (status, html, _) = h.page("/workspaces", Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("team") && html.contains("owner"), "{html}");
    let (status, html, _) = h.page(&chat_url, Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains("id=\"chat\"") && html.contains(&format!("data-workspace=\"{ws}\"")),
        "{html}"
    );
    assert!(html.contains("Allow the agent to change tables"));
    // The empty state names what there is to ask about (issue #59): nothing
    // yet, then the pasted document below.
    assert!(
        html.contains("id=\"empty\"") && html.contains("upload documents or tables"),
        "{html}"
    );

    // Documents: paste through the form, then the polled rows fragment.
    let boundary = "webform";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"title\"\r\n\r\nnotes\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\nhello from the web\r\n--{boundary}--\r\n"
    );
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("/w/{ws}/documents"))
        .header(header::COOKIE, format!("quack_session={cookie}"))
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, _, headers) = h.send(request).await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(location(&headers), format!("/w/{ws}/documents"));
    let (status, html, _) = h.page(&format!("/w/{ws}/documents"), Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("notes.md"), "{html}");
    let (status, html, _) = h
        .page(&format!("/w/{ws}/documents/rows"), Some(&cookie))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains("hx-post=\"/w/") && html.contains("/pin\""),
        "{html}"
    );

    // SQL grid and CSV download.
    let (status, html, _) = h
        .form(
            &format!("/w/{ws}/sql"),
            Some(&cookie),
            "sql=SELECT+1+AS+n%2C+%27a%27+AS+s",
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains("<th class=\"px-3 py-2 font-mono\">n</th>") && html.contains("Download CSV"),
        "{html}"
    );
    let (status, csv, headers) = h
        .page(&format!("/w/{ws}/sql.csv?sql=SELECT+1+AS+n"), Some(&cookie))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers
            .get(header::CONTENT_TYPE)
            .is_some_and(|c| c.to_str().unwrap_or_default().starts_with("text/csv"))
    );
    assert_eq!(csv, "n\n1\n");
    let (status, html, _) = h
        .form(&format!("/w/{ws}/sql"), Some(&cookie), "sql=SELEC+broken")
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("text-red-800"), "{html}");

    // Context, settings, tables, admin pages all render.
    let (status, _, headers) = h
        .form(
            &format!("/w/{ws}/context"),
            Some(&cookie),
            "content=Be+brief.",
        )
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(location(&headers), format!("/w/{ws}/context"));
    let (status, html, _) = h.page(&format!("/w/{ws}/context"), Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains("Be brief.") && html.contains("by root"),
        "{html}"
    );
    let (status, html, _) = h.page(&format!("/w/{ws}/settings"), Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains("API tokens") && html.contains("Members"),
        "{html}"
    );
    // The settings form validates providers as the API does: an unknown one
    // is refused with the reason, and nothing is saved.
    let (status, _, headers) = h
        .form(
            &format!("/w/{ws}/settings"),
            Some(&cookie),
            "classification=secret&providers=nope",
        )
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    let refused = location(&headers);
    assert!(
        refused.starts_with(&format!("/w/{ws}/settings?error=")),
        "{refused}"
    );
    let (status, html, _) = h.page(&refused, Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("not a configured provider"), "{html}");
    let unchanged = h.app.control.get_workspace(&ws).await;
    assert!(
        unchanged.is_ok_and(|w| w
            .is_some_and(|w| w.classification != "secret" && w.allowed_providers.permits("any"))),
        "a refused form saves nothing"
    );
    // A new token is shown once in the response body, never in a URL.
    let (status, html, headers) = h
        .form(
            &format!("/w/{ws}/tokens"),
            Some(&cookie),
            "name=ci&scopes=read&scopes=write",
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(headers.get(header::LOCATION).is_none(), "{headers:?}");
    assert_eq!(
        headers
            .get(header::CACHE_CONTROL)
            .and_then(|v| v.to_str().ok()),
        Some("no-cache, no-store, must-revalidate")
    );
    assert!(html.contains("New token, shown once"), "{html}");
    let Some((_, rest)) = html.split_once("qk_") else {
        fail(&html)
    };
    let secret: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    let token = format!("qk_{secret}");
    let (status, _) = h.get(&format!("/api/v1/workspaces/{ws}"), &token).await;
    assert_eq!(status, StatusCode::OK, "the shown token authenticates");
    let (status, html, _) = h
        .page(&format!("/w/{ws}/settings?token={token}"), Some(&cookie))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        !html.contains("shown once") && !html.contains(&token),
        "the settings page never echoes a token from the URL"
    );
    let (status, html, _) = h.page(&format!("/w/{ws}/tables"), Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("No tables"), "{html}");
    let (status, html, _) = h.page("/admin/users", Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("root"), "{html}");
    let (status, html, _) = h.page("/admin/audit?outcome=denied", Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("Access audit"), "{html}");
    assert!(
        html.contains(r#"<option value="denied" selected>"#),
        "the filter keeps its choice: {html}"
    );
    // The filter's "any outcome" choice sends a blank value.
    let (status, html, _) = h
        .page("/admin/audit?action=&outcome=&workspace_id=", Some(&cookie))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("Access audit"), "{html}");

    // Static assets and the error page.
    let (status, css, headers) = h.page("/static/css/output.css", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers
            .get(header::CONTENT_TYPE)
            .is_some_and(|c| c.to_str().unwrap_or_default().starts_with("text/css"))
    );
    assert!(css.contains("tailwindcss"));
    assert!(
        css.contains(".answer"),
        "answer styles must be in the built CSS"
    );
    let etag = headers
        .get(header::ETAG)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    assert!(
        etag.starts_with('"')
            && headers
                .get(header::CACHE_CONTROL)
                .is_some_and(|c| c == "no-cache")
    );
    let revalidate = Request::builder()
        .uri("/static/css/output.css")
        .header(header::IF_NONE_MATCH, &etag)
        .body(Body::empty())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, _, _) = h.send(revalidate).await;
    assert_eq!(status, StatusCode::NOT_MODIFIED);
    let (status, _, _) = h.page("/static/nope.js", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, html, _) = h.page("/w/nope/chat", Some(&cookie)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(html.contains("no such workspace"), "{html}");

    // A non-member sees the 403 page, not the content.
    h.user("bob", UserKind::Standard).await;
    let (_, _, headers) = h.form("/login", None, "username=bob&password=pw").await;
    let bob = headers
        .get(header::SET_COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|c| c.split(';').next())
        .and_then(|c| c.strip_prefix("quack_session="))
        .unwrap_or_default()
        .to_owned();
    let (status, html, _) = h.page(&format!("/w/{ws}/documents"), Some(&bob)).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert!(html.contains("not a member"), "{html}");
    let (status, _, _) = h.page("/admin/users", Some(&bob)).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}

#[tokio::test(flavor = "multi_thread")]
async fn local_mode_web_skips_login() {
    let h = harness(ServeMode::Local).await;
    let (status, _, headers) = h.page("/login", None).await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(location(&headers), "/workspaces");
    let (status, html, _) = h.page("/workspaces", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains("New workspace") && !html.contains("Log out"),
        "{html}"
    );
}

#[test]
fn banner_names_the_address_mode_and_models() {
    let mut config = Config::default();
    config.general.chat_model = Some(
        "ollama/llama3"
            .parse()
            .unwrap_or_else(|e: quack_core::error::Error| fail(&e.to_string())),
    );
    config.providers.insert(
        "ollama"
            .parse::<ProviderName>()
            .unwrap_or_else(|e| fail(&e.to_string())),
        ProviderConfig::new(ProviderType::Ollama),
    );
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 8080));
    let banner = |mode, users, workspaces| {
        super::Banner {
            config: &config,
            addr,
            mode,
            users,
            workspaces,
        }
        .to_string()
    };
    let text = banner(ServeMode::Local, 0, 2);
    assert!(text.contains("http://127.0.0.1:8080/"));
    assert!(text.contains("local: no login"));
    assert!(text.contains("chat model     ollama/llama3"));
    assert!(text.contains("none (documents stored without vectors)"));
    assert!(text.contains("ollama (ollama, auth none)"));
    assert!(text.contains("workspaces     2"));
    let auth = banner(ServeMode::Login, 3, 0);
    assert!(auth.contains("3 user(s)"));
}

/// One JSON-RPC request to the MCP endpoint; returns the status, the
/// parsed body, and the `Mcp-Session-Id` the server assigned.
async fn mcp_call(
    h: &Harness,
    ws: &WorkspaceId,
    token: Option<&str>,
    session: Option<&str>,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value, Option<String>) {
    let mut request = Request::builder()
        .method(Method::POST)
        .uri(format!("/mcp/v1/{ws}"))
        .header(header::HOST, "localhost")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream");
    if let Some(token) = token {
        request = request.header(header::AUTHORIZATION, format!("Bearer {token}"));
    }
    if let Some(session) = session {
        request = request.header("mcp-session-id", session);
    }
    let request = request
        .body(Body::from(body.to_string()))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, body, headers) = h.send(request).await;
    let session = headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    // The transport answers a POST as an SSE stream; the JSON-RPC result
    // is the last `data:` line that carries a JSON object.
    let body = match body.as_str() {
        Some(text) => text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
            .next_back()
            .unwrap_or(serde_json::Value::String(text.to_owned())),
        None => body,
    };
    (status, body, session)
}

fn rpc(id: u32, method: &str, params: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
}

async fn mcp_session(h: &Harness, ws: &WorkspaceId, token: &str) -> String {
    let (status, body, session) = mcp_call(
        h,
        ws,
        Some(token),
        None,
        rpc(
            1,
            "initialize",
            &serde_json::json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": { "name": "test", "version": "0" }
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["serverInfo"]["name"], "quack", "{body}");
    let session = session.unwrap_or_else(|| fail("no session id"));
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("/mcp/v1/{ws}"))
        .header(header::HOST, "localhost")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .header("mcp-session-id", &session)
        .body(Body::from(
            serde_json::json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })
                .to_string(),
        ))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, _, _) = h.send(request).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    session
}

#[tokio::test(flavor = "multi_thread")]
async fn mcp_over_http_lists_tools_runs_sql_reads_resources_and_audits() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("mcp", &owner).await;
    let read_token = h
        .app
        .control
        .create_token(&ws, &owner, "ro", &[Scope::Read], None)
        .await
        .map_or_else(
            |e| fail(&e.to_string()),
            |issued| issued.secret.expose().to_owned(),
        );
    let write_token = h
        .app
        .control
        .create_token(&ws, &owner, "rw", &[Scope::Read, Scope::Write], None)
        .await
        .map_or_else(
            |e| fail(&e.to_string()),
            |issued| issued.secret.expose().to_owned(),
        );

    // No bearer: refused before any MCP handling.
    let (status, _, _) = mcp_call(
        &h,
        &ws,
        None,
        None,
        rpc(1, "initialize", &serde_json::json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let session = mcp_session(&h, &ws, &read_token).await;
    let (status, body, _) = mcp_call(
        &h,
        &ws,
        Some(&read_token),
        Some(&session),
        rpc(2, "tools/list", &serde_json::json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let mut names: Vec<&str> = body["result"]["tools"]
        .as_array()
        .map(|tools| tools.iter().filter_map(|t| t["name"].as_str()).collect())
        .unwrap_or_default();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "describe_table",
            "find_path",
            "list_documents",
            "list_tables",
            "query",
            "search",
            "search_graph",
            "sql"
        ]
    );

    // A read-only token cannot write through `sql`; the refusal is a tool
    // error the client model can read, and it is audited as denied.
    let (status, body, _) = mcp_call(
        &h,
        &ws,
        Some(&read_token),
        Some(&session),
        rpc(3, "tools/call", &serde_json::json!({ "name": "sql", "arguments": { "sql": "CREATE TABLE t (n INTEGER)" } }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["isError"], true, "{body}");
    assert!(
        body["result"]["content"][0]["text"]
            .as_str()
            .is_some_and(|t| t.contains("cannot write")),
        "{body}"
    );

    // A write token gets its own transport and may create the table.
    let rw_session = mcp_session(&h, &ws, &write_token).await;
    let (status, body, _) = mcp_call(
        &h,
        &ws,
        Some(&write_token),
        Some(&rw_session),
        rpc(4, "tools/call", &serde_json::json!({ "name": "sql", "arguments": { "sql": "CREATE TABLE t AS SELECT 20.5 AS n" } }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["isError"], false, "{body}");

    let (_, body, _) = mcp_call(
        &h,
        &ws,
        Some(&read_token),
        Some(&session),
        rpc(
            5,
            "tools/call",
            &serde_json::json!({ "name": "list_tables", "arguments": {} }),
        ),
    )
    .await;
    assert_eq!(
        body["result"]["structuredContent"]["tables"],
        serde_json::json!(["t"]),
        "{body}"
    );
    let (_, body, _) = mcp_call(
        &h,
        &ws,
        Some(&read_token),
        Some(&session),
        rpc(
            6,
            "tools/call",
            &serde_json::json!({ "name": "sql", "arguments": { "sql": "SELECT n FROM t" } }),
        ),
    )
    .await;
    assert_eq!(
        body["result"]["structuredContent"]["rows"],
        serde_json::json!([[20.5]]),
        "{body}"
    );
    let (_, body, _) = mcp_call(
        &h,
        &ws,
        Some(&read_token),
        Some(&session),
        rpc(
            7,
            "tools/call",
            &serde_json::json!({ "name": "describe_table", "arguments": { "table": "nope" } }),
        ),
    )
    .await;
    assert_eq!(body["result"]["isError"], true, "{body}");

    let (_, body, _) = mcp_call(
        &h,
        &ws,
        Some(&read_token),
        Some(&session),
        rpc(8, "resources/list", &serde_json::json!({})),
    )
    .await;
    let uris: Vec<&str> = body["result"]["resources"]
        .as_array()
        .map(|r| r.iter().filter_map(|r| r["uri"].as_str()).collect())
        .unwrap_or_default();
    assert!(
        uris.contains(&"quack://workspace/tables/t/schema"),
        "{body}"
    );
    assert!(uris.contains(&"quack://workspace/context"), "{body}");
    let (_, body, _) = mcp_call(
        &h,
        &ws,
        Some(&read_token),
        Some(&session),
        rpc(
            9,
            "resources/read",
            &serde_json::json!({ "uri": "quack://workspace/tables/t/schema" }),
        ),
    )
    .await;
    let text = body["result"]["contents"][0]["text"]
        .as_str()
        .unwrap_or_default();
    assert!(text.contains("\"row_count\":1"), "{body}");
    let (_, body, _) = mcp_call(
        &h,
        &ws,
        Some(&read_token),
        Some(&session),
        rpc(
            10,
            "resources/read",
            &serde_json::json!({ "uri": "quack://workspace/nothing" }),
        ),
    )
    .await;
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("no resource")),
        "{body}"
    );

    let sql_rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("sql")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(
        sql_rows.len(),
        3,
        "denied write, allowed write, allowed read"
    );
    assert!(
        sql_rows.iter().all(|r| r.channel == Channel::Mcp),
        "{sql_rows:?}"
    );
    assert_eq!(
        sql_rows
            .iter()
            .filter(|r| r.outcome == Outcome::Denied)
            .count(),
        1
    );
}

/// Every allowed workspace read writes its access row and its detail row
/// under one id, and a table name never reaches control.db (issue #54).
#[tokio::test(flavor = "multi_thread")]
async fn allowed_reads_are_audited_and_table_names_stay_in_the_workspace() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("data", &owner).await;
    let token = h.login("owner").await;
    let (status, _) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &token,
            serde_json::json!({ "sql": "CREATE TABLE customer_secrets AS SELECT 1 AS a" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    for path in [
        "tables",
        "tables/customer_secrets",
        "documents",
        "sessions",
        "ontology/versions",
        "graph/status",
        "context",
        "members",
    ] {
        let (status, body) = h
            .get(&format!("/api/v1/workspaces/{ws}/{path}"), &token)
            .await;
        assert_eq!(status, StatusCode::OK, "{path}: {body}");
    }
    let (_, _, headers) = h.form("/login", None, "username=owner&password=pw").await;
    let cookie = headers
        .get(header::SET_COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|c| c.split(';').next())
        .and_then(|c| c.strip_prefix("quack_session="))
        .unwrap_or_default()
        .to_owned();
    let (status, _, _) = h
        .page(&format!("/w/{ws}/tables/customer_secrets"), Some(&cookie))
        .await;
    assert_eq!(status, StatusCode::OK);

    let rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            outcome: Some(Outcome::Allowed),
            ..AuditFilter::default()
        })
        .await;
    let lists = rows.iter().filter(|r| r.action == "list").count();
    assert!(lists >= 7, "{rows:?}");
    // The API's describe and the web table page: two opens, neither
    // naming the table in control.db.
    assert_eq!(
        rows.iter().filter(|r| r.action == "open").count(),
        2,
        "{rows:?}"
    );
    assert!(
        rows.iter()
            .all(|r| r.resource_id.as_deref() != Some("customer_secrets")),
        "{rows:?}"
    );
    // Both halves, under the same ids: the workspace detail names the
    // table the control row does not.
    let db = h
        .app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    let details = with_db(db, |db| audit::list(db, 100))
        .await
        .unwrap_or_else(|e| fail(&e.message));
    let ids: std::collections::BTreeSet<&str> = details.iter().map(|d| d.id.as_str()).collect();
    assert!(rows.iter().all(|r| ids.contains(r.id.as_str())), "{rows:?}");
    assert!(
        details.iter().any(|d| {
            d.action == "open"
                && d.detail
                    .as_ref()
                    .and_then(|v| v.get("table"))
                    .and_then(|t| t.as_str())
                    == Some("customer_secrets")
        }),
        "{details:?}"
    );
}

/// One graph extraction per workspace at a time (issue #48): the slot
/// is held until the run ends and freed when it drops.
#[tokio::test]
async fn extraction_slots_are_exclusive_per_workspace() {
    let h = harness(ServeMode::Local).await;
    let first = h.app.begin_extraction(&WorkspaceId::from("ws-a"));
    assert!(first.is_some());
    assert!(h.app.begin_extraction(&WorkspaceId::from("ws-a")).is_none());
    assert!(h.app.begin_extraction(&WorkspaceId::from("ws-b")).is_some());
    drop(first);
    assert!(h.app.begin_extraction(&WorkspaceId::from("ws-a")).is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn graph_is_built_from_mapped_tables_and_explored_over_the_api_and_the_page() {
    let h = harness(ServeMode::Local).await;
    let (_, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "g" })),
        )
        .await;
    let ws = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    for sql in [
        "CREATE TABLE shipments (po TEXT, vendor TEXT, country TEXT)",
        "INSERT INTO shipments VALUES ('PO-1', 'Orgenics', 'Kenya'), ('PO-2', 'Orgenics', 'Uganda'), ('PO-3', 'Aurobindo', 'Kenya')",
    ] {
        let (status, body) = h
            .call(
                Method::POST,
                &format!("/api/v1/workspaces/{ws}/sql"),
                None,
                Some(serde_json::json!({ "sql": sql })),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    let base = format!("/api/v1/workspaces/{ws}/graph");

    // No ontology yet: nothing to build, and the status says so.
    let (status, body) = h.get(&format!("{base}/status"), "").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["nodes"], 0);
    let (status, _) = h
        .call(
            Method::POST,
            &format!("{base}/extract"),
            None,
            Some(serde_json::json!({ "source": "tables" })),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let ontology = serde_json::json!({
        "classes": [
            { "id": "vendor", "key": "name", "properties": ["name"] },
            { "id": "country", "key": "name", "properties": ["name"] },
            { "id": "shipment", "key": "po", "properties": ["po"] }
        ],
        "relations": [
            { "id": "supplied_by", "domain": "shipment", "range": "vendor" },
            { "id": "delivered_to", "domain": "shipment", "range": "country" }
        ],
        "properties": [
            { "id": "name", "type": "string" },
            { "id": "po", "type": "string" }
        ],
        "mappings": [{
            "table": "shipments", "class": "shipment", "key": "po",
            "relations": [
                { "relation": "supplied_by", "column": "vendor", "target_class": "vendor", "target_key": "name" },
                { "relation": "delivered_to", "column": "country", "target_class": "country", "target_key": "name" }
            ]
        }]
    });
    let (status, body) = h
        .call(
            Method::PUT,
            &format!("/api/v1/workspaces/{ws}/ontology"),
            None,
            Some(ontology),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // Tables only: deterministic, answers 200 with the summary.
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{base}/extract"),
            None,
            Some(serde_json::json!({ "source": "tables" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "done");
    assert_eq!(body["tables"][0]["nodes"], 3, "{body}");
    let (_, body) = h.get(&format!("{base}/status"), "").await;
    assert_eq!(body["nodes"], 7, "{body}");
    assert_eq!(body["edges"], 6);
    assert_eq!(body["stale"], false);
    assert_eq!(body["provisional_nodes"], 0);

    // Search, class listing, and path.
    let (status, body) = h
        .get(&format!("{base}/search?entity=Kenya&hops=1"), "")
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let labels: Vec<&str> = body["nodes"]
        .as_array()
        .map(|n| n.iter().filter_map(|n| n["label"].as_str()).collect())
        .unwrap_or_default();
    assert!(
        labels.contains(&"Kenya") && labels.contains(&"PO-1") && labels.contains(&"PO-3"),
        "{labels:?}"
    );
    assert!(!labels.contains(&"Orgenics"));
    assert!(body["provenance"].as_array().is_some_and(|p| !p.is_empty()));
    let (status, body) = h.get(&format!("{base}/search?class=vendor"), "").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["nodes"].as_array().map(Vec::len), Some(2));
    let (status, _) = h.get(&format!("{base}/search"), "").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, body) = h
        .get(
            &format!("{base}/path?from=Uganda&to=Aurobindo&max_hops=6"),
            "",
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["edges"].as_array().map(Vec::len), Some(6), "{body}");
    let (_, body) = h
        .get(
            &format!("{base}/path?from=Uganda&to=Aurobindo&max_hops=2"),
            "",
        )
        .await;
    assert_eq!(body["nodes"].as_array().map(Vec::len), Some(0));

    // The ontology loses a class: the graph is stale, revalidate drops it.
    let (_, current) = h
        .get(&format!("/api/v1/workspaces/{ws}/ontology"), "")
        .await;
    let mut edited = current.clone();
    if let Some(classes) = edited["classes"].as_array_mut() {
        classes.retain(|c| c["id"] != "country");
    }
    if let Some(relations) = edited["relations"].as_array_mut() {
        relations.retain(|r| r["id"] != "delivered_to");
    }
    if let Some(mapping_relations) = edited["mappings"][0]["relations"].as_array_mut() {
        mapping_relations.retain(|r| r["target_class"] != "country");
    }
    let (status, body) = h
        .call(
            Method::PUT,
            &format!("/api/v1/workspaces/{ws}/ontology"),
            None,
            Some(edited),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, body) = h.get(&format!("{base}/status"), "").await;
    assert_eq!(body["stale"], true, "{body}");
    let (status, body) = h
        .call(Method::POST, &format!("{base}/revalidate"), None, None)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["dropped_nodes"], 2);
    let (_, body) = h.get(&format!("{base}/status"), "").await;
    assert_eq!(body["stale"], false);
    assert_eq!(body["nodes"], 5);
    let (_, body) = h.get(&format!("{base}/merges"), "").await;
    assert_eq!(body["merges"], serde_json::json!([]));
    let (status, _) = h
        .call(
            Method::PUT,
            &format!("{base}/merges/nope"),
            None,
            Some(serde_json::json!({ "action": "accept" })),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    // An unknown action is refused by the API and the page alike, before
    // anything is decided: a rejection cannot be undone.
    let (status, body) = h
        .call(
            Method::PUT,
            &format!("{base}/merges/nope"),
            None,
            Some(serde_json::json!({ "action": "acept" })),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    let (status, _, headers) = h
        .form(&format!("/w/{ws}/graph/merges/nope"), None, "action=acept")
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert!(
        location(&headers).contains("unknown+merge+decision"),
        "{}",
        location(&headers)
    );

    // The web page renders the status and a search result.
    let (status, html, _) = h.page(&format!("/w/{ws}/graph?entity=Kenya"), None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("Knowledge graph"), "{html}");
    assert!(html.contains("5 nodes, 3 edges"), "{html}");
    assert!(
        html.contains("Nothing matched"),
        "Kenya was dropped: {html}"
    );
    let (status, html, _) = h.page(&format!("/w/{ws}/graph?class=vendor"), None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains("Orgenics") && html.contains("Aurobindo") && html.contains("data-graph="),
        "{html}"
    );
    let (status, _, headers) = h
        .form(&format!("/w/{ws}/graph/extract"), None, "source=tables")
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(location(&headers), format!("/w/{ws}/graph"));

    let audits = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("graph_extract")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(audits.len(), 2, "API and web builds");
    let searches = h
        .audit(AuditFilter {
            workspace_id: Some(ws),
            action: Some(String::from("graph")),
            ..AuditFilter::default()
        })
        .await;
    assert!(searches.len() >= 4, "{searches:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn okf_bundles_export_as_tar_and_import_as_documents_and_candidates() {
    let h = harness(ServeMode::Local).await;
    let (_, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "okf" })),
        )
        .await;
    let ws = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    let (status, _) = h
        .call(
            Method::POST,
            &format!("/api/v1/workspaces/{ws}/sql"),
            None,
            Some(serde_json::json!({ "sql": "CREATE TABLE sales AS SELECT 'north' AS region, 10 AS total" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = h
        .call(
            Method::PUT,
            &format!("/api/v1/workspaces/{ws}/context"),
            None,
            Some(serde_json::json!({ "content": "Totals are in USD." })),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = h
        .call(
            Method::POST,
            &format!("/api/v1/workspaces/{ws}/ontology/init"),
            None,
            Some(serde_json::json!({})),
        )
        .await;
    assert_eq!(status, StatusCode::OK);

    // Export: a tar of Markdown files with front matter.
    let request = Request::builder()
        .uri(format!("/api/v1/workspaces/{ws}/okf"))
        .body(Body::empty())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let response = h
        .router
        .clone()
        .oneshot(request)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("application/x-tar")
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let bundle = Bundle::from_tar(&bytes).unwrap_or_else(|e| fail(&e.to_string()));
    let paths: Vec<&str> = bundle.files.iter().map(|f| f.path.as_str()).collect();
    assert!(
        paths.contains(&"index.md")
            && paths.contains(&"tables/sales.md")
            && paths.contains(&"log.md"),
        "{paths:?}"
    );
    // The export streams, so it is audited when the stream ends.
    let mut exported = false;
    for _ in 0..100 {
        exported = h
            .audit(AuditFilter {
                workspace_id: Some(ws.clone()),
                action: Some(String::from("export")),
                ..AuditFilter::default()
            })
            .await
            .iter()
            .any(|r| r.outcome == Outcome::Allowed);
        if exported {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(exported, "the finished export is audited as allowed");
    assert!(
        paths.contains(&"ontology/classes/organization.md"),
        "{paths:?}"
    );
    assert!(
        paths.contains(&"ontology/relations/works-at.md"),
        "{paths:?}"
    );
    let index = bundle.index().unwrap_or_else(|| fail("no index"));
    assert!(
        index.content.starts_with("---\ntype: index\n"),
        "{}",
        index.content
    );
    assert!(index.content.contains("Totals are in USD."));
    let table = bundle
        .files
        .iter()
        .find(|f| f.path == "tables/sales.md")
        .unwrap_or_else(|| fail("no table file"));
    assert!(
        table.content.contains("type: DuckDB Table")
            && table.content.contains("| region | VARCHAR |"),
        "{}",
        table.content
    );
    let exports = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("export")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(exports.len(), 1);

    assert!(paths.contains(&"ontology/ontology.md"), "{paths:?}");
    let log = bundle
        .files
        .iter()
        .find(|f| f.path == "log.md")
        .map(|f| f.content.clone())
        .unwrap_or_default();
    assert!(
        !log.contains("Activity"),
        "no audit detail in a bundle: {log}"
    );

    // quack's own bundle back into a fresh workspace: the ontology is
    // restored exactly, and none of the stubs become documents.
    let (_, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "okf3" })),
        )
        .await;
    let ws3 = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("/api/v1/workspaces/{ws3}/documents"))
        .header(header::CONTENT_TYPE, "application/x-tar")
        .body(Body::from(bytes.clone()))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, body, _) = h.send(request).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(
        body["documents"].as_array().map(Vec::len),
        Some(0),
        "{body}"
    );
    assert_eq!(body["ontology_version"], 1, "{body}");
    assert_eq!(body["candidates"], 0, "{body}");
    let (_, restored) = h
        .get(&format!("/api/v1/workspaces/{ws3}/ontology"), "")
        .await;
    assert!(
        restored["classes"]
            .as_array()
            .is_some_and(|c| c.iter().any(|x| x["id"] == "organization")),
        "{restored}"
    );

    // Import into a fresh workspace: concept files become documents, types
    // and links become candidates, index.md comes back as context.
    let (_, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "okf2" })),
        )
        .await;
    let ws2 = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    let mut incoming = Bundle::default();
    incoming.files.push(BundleFile {
        path: String::from("index.md"),
        content: String::from("---\ntype: index\n---\n# Shipping\n\nAll weights in kg.\n"),
    });
    incoming.files.push(BundleFile {
        path: String::from("vendors/orgenics.md"),
        content: String::from(
            "---\ntype: Vendor\ntitle: Orgenics\n---\nShips to [Kenya](../countries/kenya.md).\n",
        ),
    });
    incoming.files.push(BundleFile {
        path: String::from("countries/kenya.md"),
        content: String::from("---\ntype: Country\ntitle: Kenya\n---\nEast Africa.\n"),
    });
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("/api/v1/workspaces/{ws2}/documents"))
        .header(header::CONTENT_TYPE, "application/x-tar")
        .body(Body::from(
            incoming.to_tar().unwrap_or_else(|e| fail(&e.to_string())),
        ))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, body, _) = h.send(request).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(
        body["documents"].as_array().map(Vec::len),
        Some(2),
        "{body}"
    );
    assert_eq!(
        body["candidates"], 3,
        "vendor, country, vendor_links_country: {body}"
    );
    assert_eq!(body["context"], "# Shipping\n\nAll weights in kg.");
    let first = body["documents"][0]["id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let ready = h.wait_ready(&ws2, &first, "").await;
    assert_eq!(ready["status"], "ready", "{ready}");
    assert_eq!(ready["title"], "Orgenics");
    let (_, body) = h
        .get(&format!("/api/v1/workspaces/{ws2}/ontology/candidates"), "")
        .await;
    let ids: Vec<&str> = body["candidates"]
        .as_array()
        .map(|c| {
            c.iter()
                .filter_map(|c| c["proposal"]["id"].as_str())
                .collect()
        })
        .unwrap_or_default();
    assert!(
        ids.contains(&"vendor") && ids.contains(&"vendor_links_country"),
        "{ids:?}"
    );

    // A body that is not a tar is a 400.
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("/api/v1/workspaces/{ws2}/documents"))
        .header(header::CONTENT_TYPE, "application/x-tar")
        .body(Body::from("nope"))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, _, _) = h.send(request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test(flavor = "multi_thread")]
async fn external_rows_import_over_the_api_and_the_web_form_with_the_source_redacted() {
    let h = harness(ServeMode::Local).await;
    let (_, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "imp" })),
        )
        .await;
    let ws = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    // Outside the data directory: quack's own files are refused (below).
    let source_dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let source_path = source_dir.path().join("source.db");
    {
        use sqlx::{Connection as _, Executor as _};
        let url = format!("sqlite://{}?mode=rwc", source_path.display());
        let mut conn = sqlx::SqliteConnection::connect(&url)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        conn.execute("CREATE TABLE vendors (id INTEGER, name TEXT)")
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        conn.execute("INSERT INTO vendors VALUES (1, 'Orgenics'), (2, 'Aurobindo')")
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    let url = format!("sqlite://{}", source_path.display());
    let (status, body) = h
        .call(
            Method::POST,
            &format!("/api/v1/workspaces/{ws}/import"),
            None,
            Some(serde_json::json!({ "url": url, "table": "vendors", "source_table": "vendors" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["rows"], 2);
    assert_eq!(body["table"], "vendors");
    let (_, body) = h.get(&format!("/api/v1/workspaces/{ws}/tables"), "").await;
    assert_eq!(body["tables"], serde_json::json!(["vendors"]));

    // The server's own control database stays out, even for the owner
    // (issue #69).
    let control = h.app.config.general.data_dir.join("control.db");
    let (status, body) = h
        .call(
            Method::POST,
            &format!("/api/v1/workspaces/{ws}/import"),
            None,
            Some(serde_json::json!({
                "url": format!("sqlite:{}", control.display()),
                "table": "x",
                "source_table": "users"
            })),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("own data directory"),
        "{body}"
    );

    let (status, body) = h
        .call(
            Method::POST,
            &format!("/api/v1/workspaces/{ws}/import"),
            None,
            Some(serde_json::json!({ "url": "ftp://x/y", "table": "t", "source_table": "t" })),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, body) = h
        .call(
            Method::POST,
            &format!("/api/v1/workspaces/{ws}/import"),
            None,
            Some(serde_json::json!({ "url": url, "table": "t", "query": "SELECT * FROM nope" })),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");

    // The web form lands on the new table; a failure comes back as a flash.
    let (status, _, headers) = h
        .form(
            &format!("/w/{ws}/import"),
            None,
            &format!(
                "url={}&table=vendors2&query={}",
                urlencode(&url),
                urlencode("SELECT * FROM vendors WHERE id = 1")
            ),
        )
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(location(&headers), format!("/w/{ws}/tables/vendors2"));
    let (status, _, headers) = h
        .form(
            &format!("/w/{ws}/import"),
            None,
            "url=nope&table=t&source_table=t",
        )
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert!(location(&headers).contains("/tables?error="));

    let imports = h
        .audit(AuditFilter {
            workspace_id: Some(ws),
            action: Some(String::from("import")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(
        imports.len(),
        4,
        "two allowed, the refused control.db, and the failed query; the bad URL never reaches the audit"
    );
    assert_eq!(
        imports
            .iter()
            .filter(|r| r.outcome == Outcome::Error)
            .count(),
        2
    );
}

fn urlencode(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' => {
                char::from(b).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}

/// Issue #73: a session dies once it has sat unused for
/// `[server].session_idle_minutes`, and the token is then refused and
/// forgotten rather than quietly treated as an unknown API token.
#[tokio::test(flavor = "multi_thread")]
async fn an_idle_session_expires_and_is_audited() {
    let mut config = Config::default();
    // Zero is "already idle", so the very next request is past the bound.
    config.server.session_idle_minutes = 0;
    let h = harness_with(ServeMode::Login, config).await;
    h.user("root", UserKind::Admin).await;
    let token = h.login("root").await;

    let (status, body) = h.get("/api/v1/workspaces", &token).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("session expired"),
        "{body}"
    );
    let denied = h
        .audit(AuditFilter {
            action: Some(String::from("session")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(denied.len(), 1);
    assert_eq!(denied.first().map(|r| r.outcome.as_str()), Some("denied"));
    // The second attempt finds nothing left to expire, so it falls through
    // to the token path: the entry really was dropped.
    let (status, body) = h.get("/api/v1/workspaces", &token).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("unknown token"),
        "{body}"
    );
}

/// The absolute bound is separate from the idle one: a session in constant
/// use still dies at `[server].session_max_age_hours`.
#[tokio::test(flavor = "multi_thread")]
async fn a_session_expires_at_its_absolute_age_however_busy() {
    let mut config = Config::default();
    config.server.session_max_age_hours = 0;
    // Generous idle bound, so only the absolute one can fire.
    config.server.session_idle_minutes = 600;
    let h = harness_with(ServeMode::Login, config).await;
    h.user("root", UserKind::Admin).await;
    let token = h.login("root").await;

    let (status, body) = h.get("/api/v1/workspaces", &token).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("session expired"),
        "{body}"
    );
}

/// The cookie carries `Secure` for anything that did not arrive on
/// loopback, and a `Max-Age` matching the session's absolute lifetime, so a
/// browser drops it when the server would.
#[tokio::test(flavor = "multi_thread")]
async fn the_session_cookie_is_secure_off_loopback_and_carries_max_age() {
    let h = harness(ServeMode::Login).await;
    h.user("root", UserKind::Admin).await;

    let (status, _, headers) = h
        .form_from("/login", "127.0.0.1:51000", "username=root&password=pw")
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    let local = set_cookie(&headers);
    assert!(local.contains("quack_session="), "{local}");
    assert!(local.contains("HttpOnly"), "{local}");
    assert!(local.contains("SameSite=Lax"), "{local}");
    // 12 hours, the default absolute lifetime.
    assert!(local.contains("Max-Age=43200"), "{local}");
    assert!(
        !local.contains("Secure"),
        "plain HTTP on loopback is the normal local case: {local}"
    );

    let (status, _, headers) = h
        .form_from("/login", "203.0.113.7:51000", "username=root&password=pw")
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    let remote = set_cookie(&headers);
    assert!(
        remote.contains("Secure"),
        "a cookie minted off-loopback must never go back in the clear: {remote}"
    );
    assert!(remote.contains("Max-Age=43200"), "{remote}");
}

/// The limiters' per-key state is swept on a loop rather than once: without
/// it governor keeps one entry per caller for the life of the process.
#[tokio::test(flavor = "multi_thread")]
async fn rate_limiter_state_is_swept_repeatedly() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let sweeps = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&sweeps);
    super::spawn_cleanup(std::time::Duration::from_millis(5), move || {
        counter.fetch_add(1, Ordering::Relaxed);
    });
    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    // Loosely bounded so a loaded machine cannot make this flaky; the point
    // is that it ticks more than once, not how fast.
    assert!(
        sweeps.load(Ordering::Relaxed) >= 2,
        "the cleanup loop ran {} times; it is not repeating",
        sweeps.load(Ordering::Relaxed)
    );
}

/// Issue #73: the browser login form is throttled the way the API login
/// already was, and `/healthz` stays outside every limiter so a health probe
/// can never be made to look like a dead server.
#[tokio::test(flavor = "multi_thread")]
async fn the_login_form_is_rate_limited_and_healthz_is_not() {
    let h = harness(ServeMode::Login).await;
    h.user("root", UserKind::Admin).await;
    // An unknown username, so each rejection costs only a password hash.
    let form = "username=nobody&password=wrong";
    // The attempts go out together. A sequential loop races the bucket
    // draining against it refilling a cell every `LOGIN_RATE_PER_SECOND`
    // seconds, and an argon2 verify in an unoptimized test build can outlast
    // the refill on a loaded machine, so the budget is never spent and no
    // attempt is refused (this test failed that way in CI). Concurrent
    // attempts put every limiter check within microseconds of the others,
    // whatever the hash behind it costs: the check runs when the request
    // future is first polled, before the handler awaits anything.
    let attempts = (0..(super::LOGIN_RATE_BURST + 4)).map(|_| h.form("/login", None, form));
    let refused = futures::future::join_all(attempts)
        .await
        .into_iter()
        .filter(|(status, _, _)| *status == StatusCode::TOO_MANY_REQUESTS)
        .count();
    // Everything past the burst is refused; one cell of slack in case the
    // bucket happens to replenish one mid-pass.
    assert!(
        refused >= 3,
        "{refused} of {} attempts were refused; the form is not throttled",
        super::LOGIN_RATE_BURST + 4
    );

    for _ in 0..(super::LOGIN_RATE_BURST + 4) {
        let (status, body) = h.call(Method::GET, "/healthz", None, None).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
}

// --- jobs ----------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn jobs_report_uploads_hide_other_questions_and_cancel_by_their_owner() {
    use quack_core::jobs::{JobKind, JobSpec, JobState, Lane};

    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let member = h.user("member", UserKind::Standard).await;
    let other = h.user("other", UserKind::Standard).await;
    let viewer = h.user("viewer", UserKind::Standard).await;
    let ws = h.workspace("work", &owner).await;
    let elsewhere = h.workspace("elsewhere", &owner).await;
    for (user, role) in [
        (&member, Role::Member),
        (&other, Role::Member),
        (&viewer, Role::Viewer),
    ] {
        h.app
            .control
            .set_member(&ws, user, role)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    let owner_token = h.login("owner").await;
    let member_token = h.login("member").await;
    let other_token = h.login("other").await;
    let viewer_token = h.login("viewer").await;
    let jobs = format!("/api/v1/workspaces/{ws}/jobs");

    // An upload answers with its job, which the list reports as it ends.
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/documents"),
            &owner_token,
            serde_json::json!({ "text": "Jobs run in the background.", "title": "notes" }),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let upload_job = body["documents"][0]["job"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let doc = body["documents"][0]["id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert_eq!(
        h.wait_ready(&ws, &doc, &owner_token).await["status"],
        "ready"
    );
    let (status, body) = h.get(&format!("{jobs}/{upload_job}"), &viewer_token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["kind"], "ingest");
    assert_eq!(body["label"], "notes.md");
    assert_eq!(body["state"], "succeeded", "{body}");
    assert_eq!(body["lane"], format!("ingest:{ws}"));

    // A member's question, waiting on its cancel token.
    let question = h
        .app
        .jobs
        .submit(
            JobSpec::new(JobKind::Chat, "what is our churn?")
                .workspace(ws.clone())
                .owner(Some(member.clone()))
                .lane(Lane::serial(&LaneKey::Session(SessionId::from("s")))),
            |ctx| async move {
                ctx.cancel_token().cancelled().await;
                Err(String::from("cancelled"))
            },
        )
        .id;
    let (status, body) = h.get(&jobs, &viewer_token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["jobs"].as_array().map(Vec::len), Some(2));
    // Newest first; another member's question text stays private.
    assert_eq!(body["jobs"][0]["kind"], "chat");
    assert_eq!(body["jobs"][0]["label"], "a question in a private session");
    let (_, body) = h.get(&jobs, &owner_token).await;
    assert_eq!(body["jobs"][0]["label"], "what is our churn?");
    let (_, body) = h.get(&jobs, &member_token).await;
    assert_eq!(body["jobs"][0]["label"], "what is our churn?");
    assert_eq!(body["running"], 1);

    // Only the job's owner (or a workspace owner) cancels it; a viewer
    // cannot write at all.
    let cancel = format!("{jobs}/{question}/cancel");
    let (status, _) = h
        .call(Method::POST, &cancel, Some(&viewer_token), None)
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = h
        .call(Method::POST, &cancel, Some(&other_token), None)
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, body) = h
        .call(Method::POST, &cancel, Some(&member_token), None)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["cancel_requested"], true);
    let ended = tokio::time::timeout(std::time::Duration::from_secs(5), h.app.jobs.wait(question))
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| fail("the question never ended"));
    assert_eq!(ended.state, JobState::Cancelled);
    let denied = h
        .audit(AuditFilter {
            action: Some(String::from("cancel")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(denied.len(), 2, "one denied, one allowed: {denied:?}");

    // A job is found only under its own workspace.
    let (status, _) = h
        .get(
            &format!("/api/v1/workspaces/{elsewhere}/jobs/{upload_job}"),
            &owner_token,
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = h.get(&format!("{jobs}/not-a-job"), &owner_token).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // The web console lists the same jobs.
    let (status, html, _) = h.page(&format!("/w/{ws}/jobs"), Some(&owner_token)).await;
    assert_eq!(status, StatusCode::OK, "{html}");
    assert!(
        html.contains("notes.md") && html.contains("what is our churn?"),
        "{html}"
    );
    let (status, html, _) = h
        .page(&format!("/w/{ws}/jobs/rows"), Some(&viewer_token))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("a question in a private session"), "{html}");
    assert!(!html.contains("Cancel</button>"), "{html}");
}

#[tokio::test(flavor = "multi_thread")]
async fn uploads_are_turned_away_with_retry_after_while_the_lane_is_full() {
    use quack_core::jobs::{JobKind, JobSpec, Lane};

    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("busy", &owner).await;
    let token = h.login("owner").await;
    let release = CancellationToken::new();
    let lane = LaneKey::Ingest(ws.clone());
    for n in 0..MAX_WAITING_UPLOADS {
        let release = release.clone();
        h.app.jobs.submit(
            JobSpec::new(JobKind::Ingest, format!("held {n}"))
                .workspace(ws.clone())
                .lane(Lane::new(&lane, 1)),
            move |_| async move {
                release.cancelled().await;
                Ok(String::new())
            },
        );
    }
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("/api/v1/workspaces/{ws}/documents"))
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({ "text": "one more", "title": "late" }).to_string(),
        ))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, body, headers) = h.send(request).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(
        headers
            .get(header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok()),
        Some("30")
    );
    // Nothing was registered for the refused upload.
    let (_, listed) = h
        .get(&format!("/api/v1/workspaces/{ws}/documents"), &token)
        .await;
    assert_eq!(listed["documents"].as_array().map(Vec::len), Some(0));

    // Once the line drains, uploads are taken again.
    release.cancel();
    for _ in 0..100 {
        if h.app.jobs.lane_active(&lane) == 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/documents"),
            &token,
            serde_json::json!({ "text": "one more", "title": "late" }),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_jobs_page_follows_the_job_stream_with_the_session_cookie() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("live", &owner).await;
    let cookie = h.login("owner").await;
    let (status, html, _) = h.page(&format!("/w/{ws}/jobs"), Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains(&format!(
            "data-jobs-stream=\"/api/v1/workspaces/{ws}/jobs/stream\""
        )) && html.contains("jobs-changed from:body")
            && !html.contains("every 2s"),
        "{html}"
    );
    // The stream answers the browser's cookie; only the head is read, since
    // the body never ends.
    let request = Request::builder()
        .uri(format!("/api/v1/workspaces/{ws}/jobs/stream"))
        .header(header::COOKIE, format!("quack_session={cookie}"))
        .body(Body::empty())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let response = h
        .router
        .clone()
        .oneshot(request)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|t| t.starts_with("text/event-stream"))
    );
}

/// Poll a job until it reaches a final state.
async fn wait_for_job(h: &Harness, ws: &WorkspaceId, job: &str, token: &str) -> serde_json::Value {
    for _ in 0..200 {
        let (_, body) = h
            .get(&format!("/api/v1/workspaces/{ws}/jobs/{job}"), token)
            .await;
        if matches!(
            body["state"].as_str(),
            Some("succeeded" | "failed" | "cancelled")
        ) {
            return body;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    fail("the job never finished")
}

#[tokio::test(flavor = "multi_thread")]
async fn stale_vectors_are_reported_and_refreshed_over_the_api_and_the_page() {
    // An embedding model the job cannot reach: the refresh starts, then
    // fails in the background with the provider's error.
    let mut config = Config::default();
    config.general.embedding_model = Some(
        "ollama/embeddinggemma"
            .parse()
            .unwrap_or_else(|e: quack_core::error::Error| fail(&e.to_string())),
    );
    config.providers.insert(
        "ollama"
            .parse::<ProviderName>()
            .unwrap_or_else(|e| fail(&e.to_string())),
        ProviderConfig {
            base_url: Some(
                BaseUrl::try_from(String::from("http://127.0.0.1:9"))
                    .unwrap_or_else(|e| fail(&e.to_string())),
            ),
            embedding_dimension: Some(Dimension::new(4)),
            ..ProviderConfig::new(ProviderType::Ollama)
        },
    );
    let h = harness_with(ServeMode::Login, config).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let viewer = h.user("viewer", UserKind::Standard).await;
    let ws = h.workspace("vectors", &owner).await;
    h.app
        .control
        .set_member(&ws, &viewer, Role::Viewer)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let owner_token = h.login("owner").await;
    let viewer_token = h.login("viewer").await;
    let base = format!("/api/v1/workspaces/{ws}/embeddings");

    // Nothing stored yet: current, nothing to do.
    let (status, body) = h.get(&base, &viewer_token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["note"], serde_json::Value::Null, "{body}");
    assert_eq!(body["status"]["profile"]["model"], "embeddinggemma");
    let (status, body) = h
        .post(
            &format!("{base}/refresh"),
            &owner_token,
            serde_json::json!({}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "current", "{body}");

    // One chunk whose vector was made under a profile the workspace never
    // recorded.
    let db = h
        .app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    db.run(|db| {
        db.insert_document(
            &NewDocument::new(&DocumentId::from("d"), "a.md", "text/markdown", 1)
                .with_status(DocumentStatus::Ready),
        )?;
        db.insert_chunk(&NewChunk {
            id: &ChunkId::from("c"),
            document_id: &DocumentId::from("d"),
            chunk_index: 0,
            content: "levee report",
            heading: None,
            page: None,
            embedding: Some(&Vector::from(vec![1.0, 0.0, 0.0, 0.0])),
        })?;
        db.execute_statement("UPDATE _quack_chunks SET embedding_profile = 'older'")
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));

    let (status, body) = h.get(&base, &viewer_token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["stale_chunks"], 1, "{body}");
    assert_eq!(body["plan"]["chunks"], 1, "{body}");
    let note = body["note"].as_str().unwrap_or_default().to_owned();
    assert!(
        note.contains("1 chunks were embedded with an unrecorded profile"),
        "{note}"
    );

    // The page says so, with the button for someone who may write.
    let (status, html, _) = h
        .page(&format!("/w/{ws}/documents"), Some(&owner_token))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains("an unrecorded profile") && html.contains("/embeddings/refresh"),
        "{html}"
    );
    let (_, html, _) = h
        .page(&format!("/w/{ws}/documents"), Some(&viewer_token))
        .await;
    assert!(
        html.contains("an unrecorded profile") && !html.contains("/embeddings/refresh"),
        "{html}"
    );

    // A viewer may not start it.
    let (status, _) = h
        .post(
            &format!("{base}/refresh"),
            &viewer_token,
            serde_json::json!({}),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    let (status, body) = h
        .post(
            &format!("{base}/refresh"),
            &owner_token,
            serde_json::json!({}),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(body["plan"]["chunks"], 1, "{body}");
    let job = body["job"].as_str().unwrap_or_default().to_owned();
    let run = body["run"].as_str().unwrap_or_default().to_owned();
    let last = wait_for_job(&h, &ws, &job, &owner_token).await;
    assert_eq!(last["state"], "failed", "{last}");
    assert_eq!(last["kind"], "embeddings", "{last}");

    // Both ends of the run are in the access audit, under its run id.
    let rows = h
        .audit(AuditFilter {
            action: Some(String::from("embeddings_refresh")),
            ..AuditFilter::default()
        })
        .await;
    let for_run: Vec<&AuditRow> = rows
        .iter()
        .filter(|r| r.resource_id.as_deref() == Some(run.as_str()))
        .collect();
    assert_eq!(for_run.len(), 2, "{rows:?}");
    assert!(
        for_run.iter().any(|r| r.outcome == Outcome::Error),
        "{rows:?}"
    );
    // The vector is still stale.
    let (_, body) = h.get(&base, &viewer_token).await;
    assert_eq!(body["stale_chunks"], 1, "{body}");

    // The page's button starts another run and comes back with a notice.
    let (status, _, headers) = h
        .form(
            &format!("/w/{ws}/embeddings/refresh"),
            Some(&owner_token),
            "",
        )
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert!(
        location(&headers).contains("background"),
        "{}",
        location(&headers)
    );
}

/// What the stand-in work reports when it succeeds.
struct Done;

impl RunReport for Done {
    fn detail(&self) -> serde_json::Value {
        serde_json::json!({ "summary": "all of it" })
    }

    fn message(&self) -> String {
        String::from("done")
    }
}

/// Every background run writes its start row and exactly one closing row
/// under the same run id: allowed with the report's detail, an error with
/// the work's message, or an error saying it was cancelled before it
/// started. Three runs share one serial lane; the first holds it until
/// released, so the third is still queued when it is cancelled.
#[tokio::test(flavor = "multi_thread")]
async fn background_runs_audit_their_start_and_end_under_one_id() {
    let h = harness(ServeMode::Local).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("runs", &owner).await;
    let workspace = h
        .app
        .control
        .get_workspace(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()))
        .unwrap_or_else(|| fail("the workspace was just created"));
    let access = Access {
        identity: Identity {
            user_id: owner.clone(),
            username: String::from("owner"),
            is_admin: false,
            credential: Credential::Local,
            client_addr: None,
            request_id: None,
            channel: None,
        },
        workspace,
        role: Some(Role::Owner),
    };
    let start = |detail: &'static str| {
        let (app, access) = (Arc::clone(&h.app), access.clone());
        async move {
            BackgroundRun::start(
                &app,
                &access,
                RunKind::Graph,
                serde_json::json!({ "step": detail }),
            )
            .await
            .unwrap_or_else(|e| fail(&e.message))
        }
    };

    let (release, released) = tokio::sync::oneshot::channel::<()>();
    let holder = start("holds the lane").await;
    let holder_id = holder.id().to_owned();
    let first = holder.submit(move |_| async move {
        drop(released.await);
        Ok(Done)
    });
    let failing = start("fails").await;
    let failing_id = failing.id().to_owned();
    let second = failing.submit(|_| async { Err::<Done, _>(String::from("the model went away")) });
    let queued = start("is cancelled").await;
    let queued_id = queued.id().to_owned();
    let third = queued.submit(|_| async { Ok(Done) });
    assert!(h.app.jobs.cancel(third), "the third run is still queued");
    assert!(
        release.send(()).is_ok(),
        "the first run is waiting for the release"
    );
    for job in [first, second, third] {
        let ended =
            tokio::time::timeout(std::time::Duration::from_secs(10), h.app.jobs.wait(job)).await;
        assert!(ended.is_ok_and(|info| info.is_some()), "{job:?} ended");
    }

    // The closing row of a cancelled run is written after the job ends.
    let mut rows = Vec::new();
    for _ in 0..100 {
        rows = h
            .audit(AuditFilter {
                workspace_id: Some(ws.clone()),
                ..AuditFilter::default()
            })
            .await
            .into_iter()
            .filter(|r| r.resource_type.as_deref() == Some("graph_run"))
            .collect();
        if rows.len() == 6 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let outcomes = |run: &RunId| -> Vec<Outcome> {
        let mut found: Vec<Outcome> = rows
            .iter()
            .filter(|r| r.resource_id.as_deref() == Some(run.as_str()))
            .map(|r| r.outcome)
            .collect();
        found.sort_by_key(|o| o.as_str());
        found
    };
    assert_eq!(outcomes(&holder_id), [Outcome::Allowed, Outcome::Allowed]);
    assert_eq!(outcomes(&failing_id), [Outcome::Allowed, Outcome::Error]);
    assert_eq!(outcomes(&queued_id), [Outcome::Allowed, Outcome::Error]);
    assert!(rows.iter().all(|r| r.action == "graph_extract"));

    let details = h
        .app
        .read(&ws, |db| audit::list(db, 100))
        .await
        .unwrap_or_else(|e| fail(&e.message));
    let closing = |needle: &str| {
        details.iter().any(|d| {
            d.detail.as_ref().is_some_and(|v| {
                v["finished"] == serde_json::json!(true) && v.to_string().contains(needle)
            })
        })
    };
    assert!(closing("all of it"), "{details:?}");
    assert!(closing("the model went away"), "{details:?}");
    assert!(closing("cancelled before it started"), "{details:?}");
}

/// Log in through the web form and return the session cookie's value.
async fn web_session(h: &Harness, username: &str) -> String {
    let (status, _, headers) = h
        .form("/login", None, &format!("username={username}&password=pw"))
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    headers
        .get(header::SET_COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|c| c.split(';').next())
        .and_then(|c| c.strip_prefix("quack_session="))
        .unwrap_or_default()
        .to_owned()
}

/// The web console and the API run one operation each, so the rules the
/// API enforces hold on the web too, with the reason in the page's flash
/// slot rather than as an error page.
#[tokio::test]
async fn web_forms_follow_the_api_rules_and_say_why() {
    let h = harness(ServeMode::Login).await;
    h.user("root", UserKind::Admin).await;
    h.user("bob", UserKind::Standard).await;
    let cookie = web_session(&h, "root").await;

    // Workspace names: the API's validation and message.
    let (status, _, headers) = h.form("/workspaces", Some(&cookie), "name=a.b").await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert!(
        location(&headers).starts_with("/workspaces?error=workspace+name+must+be+non-empty"),
        "{}",
        location(&headers)
    );
    let (_, _, headers) = h.form("/workspaces", Some(&cookie), "name=team").await;
    let ws = location(&headers)
        .trim_start_matches("/w/")
        .trim_end_matches("/chat")
        .to_owned();
    let (_, _, headers) = h.form("/workspaces", Some(&cookie), "name=team").await;
    assert_eq!(location(&headers), "/workspaces?error=workspace+exists");

    // Removing someone who is not a member says so; it used to pass silently.
    let (status, _, headers) = h
        .form(&format!("/w/{ws}/members/nobody/remove"), Some(&cookie), "")
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(
        location(&headers),
        format!("/w/{ws}/settings?error=not+a+member")
    );

    // Bulk decisions need at least one candidate.
    let (_, _, headers) = h
        .form(
            &format!("/w/{ws}/ontology/candidates"),
            Some(&cookie),
            "bulk=accept",
        )
        .await;
    assert_eq!(
        location(&headers),
        format!("/w/{ws}/ontology?error=choose+at+least+one+candidate+to+accept+or+reject")
    );

    // Proposing over no tables queues nothing and says so.
    let (_, _, headers) = h
        .form(&format!("/w/{ws}/ontology/propose"), Some(&cookie), "")
        .await;
    assert_eq!(
        location(&headers),
        format!("/w/{ws}/ontology?notice=nothing+to+propose%3A+the+tables+are+already+covered")
    );

    // A second init is refused with the API's reason.
    let (_, _, headers) = h
        .form(&format!("/w/{ws}/ontology/init"), Some(&cookie), "")
        .await;
    assert_eq!(location(&headers), format!("/w/{ws}/ontology"));
    let (_, _, headers) = h
        .form(&format!("/w/{ws}/ontology/init"), Some(&cookie), "")
        .await;
    assert_eq!(
        location(&headers),
        format!("/w/{ws}/ontology?error=an+ontology+already+exists")
    );
}

/// Local mode has no logins, so no users can be added from the API
/// either; the web form already refused.
#[tokio::test]
async fn local_mode_refuses_new_users_from_the_api() {
    let h = harness(ServeMode::Local).await;
    let (status, body) = h
        .call(
            Method::POST,
            "/api/v1/admin/users",
            None,
            Some(serde_json::json!({ "username": "carol", "password": "pw" })),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}
