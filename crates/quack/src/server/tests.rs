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
use quack_core::config::Config;
use quack_core::storage::control::{AuditFilter, ControlPlane, Role, Scope};
use std::sync::Arc;
use tower::ServiceExt;

use super::state::{App, AppState};

struct Harness {
    _dir: tempfile::TempDir,
    app: App,
    router: Router,
}

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

async fn harness(local: bool) -> Harness {
    harness_with(local, Config::default()).await
}

async fn harness_with(local: bool, mut config: Config) -> Harness {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    config.general.data_dir = dir.path().to_path_buf();
    let control = ControlPlane::open(&config)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let app = Arc::new(AppState::new(config, control, local));
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

    async fn user(&self, name: &str, admin: bool) -> String {
        self.app
            .control
            .create_user(name, "pw", admin)
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

    async fn workspace(&self, name: &str, owner: &str) -> String {
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

    async fn audit(&self, filter: AuditFilter) -> Vec<quack_core::storage::control::AuditRow> {
        self.app
            .control
            .query_audit(&AuditFilter {
                limit: 100,
                ..filter
            })
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
    }

    async fn wait_ready(&self, ws: &str, doc: &str, bearer: &str) -> serde_json::Value {
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
    let h = harness(false).await;
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
    let h = harness(false).await;
    let alice = h.user("alice", false).await;
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
    assert!(
        logins
            .iter()
            .all(|r| r.user_id.as_deref() == Some(alice.as_str()))
    );
    let (status, _) = h
        .call(Method::POST, "/api/v1/auth/logout", Some(&token), None)
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = h.get("/api/v1/auth/me", &token).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test(flavor = "multi_thread")]
async fn workspaces_follow_membership_roles_and_admin_limits() {
    let h = harness(false).await;
    h.user("root", true).await;
    let bob = h.user("bob", false).await;
    let carol = h.user("carol", false).await;
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
    let ws = body["id"].as_str().unwrap_or_default().to_owned();
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
            outcome: Some(String::from("denied")),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        denied
            .iter()
            .any(|r| r.workspace_id.as_deref() == Some(ws.as_str()) && r.channel == "web")
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
    let dave_root = h.user("dave", true).await;
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
            outcome: Some(String::from("denied")),
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
    let h = harness(false).await;
    let owner = h.user("owner", false).await;
    let viewer = h.user("viewer", false).await;
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
            .any(|r| r.outcome == "denied" && r.user_id.as_deref() == Some(viewer.as_str()))
    );
    assert!(access_rows.iter().any(|r| r.outcome == "error"));
    assert!(access_rows.iter().all(|r| r.request_id.is_some()));
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
    let h = harness(false).await;
    let owner = h.user("owner", false).await;
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
        .get(&format!("/api/v1/workspaces/{ws}/search?q=flood"), &token)
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
    assert_eq!(ingests.len(), 2);
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
    let h = harness(false).await;
    let owner = h.user("owner", false).await;
    let ws = h.workspace("a", &owner).await;
    let other = h.workspace("b", &owner).await;
    let (read_token, _) = h
        .app
        .control
        .create_token(&ws, &owner, "ro", &[Scope::Read], None)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
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

    let (write_token, row) = h
        .app
        .control
        .create_token(&ws, &owner, "rw", &[Scope::Write], None)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, _) = h
        .call(
            Method::PUT,
            &format!("/api/v1/workspaces/{ws}/context"),
            Some(&write_token),
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
            .all(|r| r.channel == "api" && r.token_hash.is_some())
    );

    let (expired, _) = h
        .app
        .control
        .create_token(
            &ws,
            &owner,
            "old",
            &[Scope::Read],
            Some("2000-01-01 00:00:00"),
        )
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, _) = h
        .get(&format!("/api/v1/workspaces/{ws}/documents"), &expired)
        .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let denied = h
        .audit(AuditFilter {
            action: Some(String::from("token")),
            outcome: Some(String::from("denied")),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        denied
            .iter()
            .any(|r| r.user_id.as_deref() == Some(owner.as_str()))
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn query_endpoints_fail_cleanly_without_a_chat_model() {
    let h = harness(false).await;
    let owner = h.user("owner", false).await;
    let viewer = h.user("viewer", false).await;
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
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
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
        .get(&format!("/api/v1/workspaces/{ws}/search?q="), &owner_token)
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
    let h = harness_with(true, config).await;
    let (status, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "w" })),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let ws = body["id"].as_str().unwrap_or_default().to_owned();
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
    let h = harness(false).await;
    let owner = h.user("owner", false).await;
    let viewer = h.user("viewer", false).await;
    let other = h.user("other", false).await;
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
        let guard = db.lock().unwrap_or_else(|e| fail(&e.to_string()));
        let mine = create_session(&guard, "m", ChatMode::Chat, Some(&viewer))
            .unwrap_or_else(|e| fail(&e.to_string()));
        let theirs = create_session(&guard, "m", ChatMode::Chat, Some(&other))
            .unwrap_or_else(|e| fail(&e.to_string()));
        (mine.id, theirs.id)
    };
    let viewer_token = h.login("viewer").await;
    let owner_token = h.login("owner").await;
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
    assert_eq!(deletes.len(), 2);
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
        let guard = db.lock().unwrap_or_else(|e| fail(&e.to_string()));
        create_session(&guard, "m", ChatMode::Chat, Some(&owner))
            .unwrap_or_else(|e| fail(&e.to_string()))
            .id
    };
    let (status, _, headers) = h
        .form(&format!("/w/{ws}/chat/{fresh}/delete"), Some(&cookie), "")
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(location(&headers), format!("/w/{ws}/chat"));
}

#[tokio::test(flavor = "multi_thread")]
async fn ontology_is_versioned_over_the_api_and_the_web_page() {
    let h = harness(false).await;
    let owner = h.user("owner", false).await;
    let viewer = h.user("viewer", false).await;
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
    let h = harness(true).await;
    let (_, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "p" })),
        )
        .await;
    let ws = body["id"].as_str().unwrap_or_default().to_owned();
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
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{base}/propose"),
            None,
            Some(serde_json::json!({ "mode": "extend" })),
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
        html.contains("Review queue (1 pending)") && html.contains("orders.mode"),
        "{html}"
    );
    // The page's form with auto-accept takes the remaining proposal straight in.
    let (status, _, headers) = h
        .form(
            &format!("/w/{ws}/ontology/propose"),
            None,
            "extend=true&auto_accept=true",
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
    let h = harness(false).await;
    h.user("root", true).await;
    h.user("bob", false).await;
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
    let h = harness(true).await;
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
    let ws = body["id"].as_str().unwrap_or_default().to_owned();
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
    assert!(rows.iter().all(|r| r.user_id.as_deref() == Some("local")));
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

fn location(headers: &axum::http::HeaderMap) -> String {
    headers
        .get(header::LOCATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned()
}

#[tokio::test(flavor = "multi_thread")]
async fn web_pages_redirect_to_login_and_render_after_the_form_login() {
    let h = harness(false).await;
    h.user("root", true).await;
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
    let ws = chat_url
        .trim_start_matches("/w/")
        .trim_end_matches("/chat")
        .to_owned();

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
    let (status, _, headers) = h
        .form(
            &format!("/w/{ws}/tokens"),
            Some(&cookie),
            "name=ci&scopes=read&scopes=write",
        )
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert!(
        location(&headers).contains("?token=qk_"),
        "{}",
        location(&headers)
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
    h.user("bob", false).await;
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
    let h = harness(true).await;
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
    config.general.chat_model = Some(String::from("ollama/llama3"));
    config.providers.insert(
        String::from("ollama"),
        quack_core::config::ProviderConfig {
            provider_type: quack_core::config::ProviderType::Ollama,
            auth: quack_core::config::AuthMode::None,
            base_url: None,
            api_key_env: None,
            embedding_dimension: None,
            oauth: None,
        },
    );
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 8080));
    let text = super::banner(&config, addr, true, 0, 2);
    assert!(text.contains("http://127.0.0.1:8080/"));
    assert!(text.contains("local: no login"));
    assert!(text.contains("chat model     ollama/llama3"));
    assert!(text.contains("none (documents stored without vectors)"));
    assert!(text.contains("ollama (ollama, no auth)"));
    assert!(text.contains("workspaces     2"));
    let auth = super::banner(&config, addr, false, 3, 0);
    assert!(auth.contains("3 user(s)"));
}
