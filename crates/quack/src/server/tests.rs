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
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = Config::default();
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
