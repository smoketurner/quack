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
use axum::body::{Body, Bytes};
use axum::http::{Method, Request, StatusCode, header};
use quack_core::config::{
    BaseUrl, Config, FollowIngest, ProviderConfig, ProviderName, ProviderType, RetryPolicy,
    SecureCookies,
};
use quack_core::embedding::{Dimension, Vector};
use quack_core::error::{Error as CoreError, Result as CoreResult};
use quack_core::ids::{AuditId, ChunkId, DocumentId, RunId, SessionId, UserId, WorkspaceId};
use quack_core::ingestion::parser::PageCounts;
use quack_core::ingestion::parser::SectionKind;
use quack_core::storage::backup::Manifest;
use std::sync::Arc;
use tower::ServiceExt;

use super::state::{App, AppState, ServeMode, with_db};
use crate::server::auth::{Access, Credential, Identity};
use crate::server::queue::UploadJob;
use crate::server::run::{BackgroundRun, RunKind, RunReport};
use quack_core::jobs::{JobKind, LaneKey};
use quack_core::okf::{Bundle, BundleSink, TarSink};
use quack_core::ontology::Ontology;
use quack_core::storage::audit;
use quack_core::storage::control::{
    AuditAction, AuditEntry, AuditFilter, AuditRow, Channel, ControlPlane, IssuedToken, Membership,
    Origin, Outcome, ResourceKind, Role, Scope, Standing, UserKind,
};
use quack_core::storage::workspace::{DocumentFields, DocumentStatus, NewChunk, NewDocument};
use quack_core::web_sessions::WebSessions;

/// The audit row a test's own setup writes.
fn setup_audit() -> AuditEntry {
    AuditEntry::new(AuditAction::Admin, Outcome::Allowed, Channel::Cli)
}

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
    let sessions = Arc::new(WebSessions::new(&config.server));
    let app = Arc::new(AppState::new(config, control, mode, sessions, None));
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

    /// `send`, with the body as it came: for a file, not JSON or text.
    async fn send_bytes(
        &self,
        request: Request<Body>,
    ) -> (StatusCode, Bytes, axum::http::HeaderMap) {
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
        (status, bytes, headers)
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
            .create_user(name, "pw", kind, setup_audit())
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
            .id
    }

    async fn login(&self, name: &str) -> String {
        self.login_with(name, "pw").await
    }

    async fn login_with(&self, name: &str, password: &str) -> String {
        let (status, body) = self
            .call(
                Method::POST,
                "/api/v1/auth/login",
                None,
                Some(serde_json::json!({ "username": name, "password": password })),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body["token"].as_str().unwrap_or_default().to_owned()
    }

    async fn workspace(&self, name: &str, owner: &UserId) -> WorkspaceId {
        let ws = self
            .app
            .control
            .create_workspace(
                &name
                    .parse()
                    .unwrap_or_else(|e: CoreError| fail(&e.to_string())),
                None,
                setup_audit(),
            )
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        self.app
            .control
            .set_member(&ws.id, owner, Role::Owner, setup_audit())
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
            .rows
    }

    /// The rows `filter` matches once `done` holds for them: a streamed
    /// response is audited when its stream ends, which can be after the
    /// client has read the last byte.
    async fn audit_eventually(
        &self,
        filter: AuditFilter,
        done: impl Fn(&[AuditRow]) -> bool,
    ) -> Vec<AuditRow> {
        for _ in 0..250 {
            let rows = self.audit(filter.clone()).await;
            if done(&rows) {
                return rows;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        fail("the audit rows never arrived")
    }

    /// What a request from the workspace's owner resolves to, for work the
    /// test starts without a request.
    async fn owner_access(&self, ws: &WorkspaceId, owner: &UserId) -> Access {
        let workspace = self
            .app
            .control
            .get_workspace(ws)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
            .unwrap_or_else(|| fail("the workspace was just created"));
        Access {
            identity: Identity {
                user_id: owner.clone(),
                username: String::from("owner"),
                kind: UserKind::Standard,
                credential: Credential::Local,
                origin: Origin::from(Channel::Web),
            },
            membership: Membership {
                workspace,
                standing: Standing::Member(Role::Owner),
            },
        }
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
    assert_eq!(
        denied.first().map(|r| r.entry.outcome.as_str()),
        Some("denied")
    );
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
    let outcomes: Vec<&str> = logins.iter().map(|r| r.entry.outcome.as_str()).collect();
    assert_eq!(outcomes, ["allowed", "denied"]);
    assert!(
        logins
            .iter()
            .all(|r| r.entry.user_id.as_ref() == Some(&alice))
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
            .any(|r| r.entry.workspace_id.as_ref() == Some(&ws)
                && r.entry.origin.channel == Channel::Web)
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
    // The audit must reflect whether anything was actually removed. A
    // no-op removal that returns 404 used to audit `Allowed`, so a SIEM
    // reading the OCSF export saw a successful `Member` event for a
    // request that failed; per `Outcome::of` it is `Error`.
    let member_rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("member")),
            ..AuditFilter::default()
        })
        .await;
    // carol: added (Allowed), removed (Allowed), removed again, a no-op (Error).
    let mut carol_outcomes: Vec<Outcome> = member_rows
        .iter()
        .filter(|r| r.entry.resource_id.as_deref() == Some(carol.as_str()))
        .map(|r| r.entry.outcome)
        .collect();
    carol_outcomes.sort_by_key(|o| o.as_str());
    assert_eq!(
        carol_outcomes,
        [Outcome::Allowed, Outcome::Allowed, Outcome::Error],
        "{member_rows:?}"
    );
    // The OCSF export turns that `Error` row into a `Failure` event, never a
    // `Success` one: the SIEM or archive sees the failed shape, and the
    // `outcome=allowed` filter no longer returns it.
    let (status, ocsf) = h
        .get(
            &format!("/api/v1/admin/audit?workspace_id={ws}&action=member&format=ocsf"),
            &root,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{ocsf}");
    let member_events = ocsf["audit"].as_array().cloned().unwrap_or_default();
    assert!(
        member_events.iter().all(|e| e["class_uid"] == 6003),
        "member rows are API activity: {ocsf}"
    );
    assert_eq!(
        member_events
            .iter()
            .filter(|e| e["status_id"] == 2 && e["status"] == "Failure")
            .count(),
        1,
        "the no-op removal is the only Failure: {ocsf}"
    );
    assert!(
        member_events
            .iter()
            .any(|e| e["status_id"] == 1 && e["status"] == "Success"),
        "the successful removal is still a Success: {ocsf}"
    );
    let (status, allowed) = h
        .get(
            &format!("/api/v1/admin/audit?workspace_id={ws}&action=member&outcome=allowed"),
            &root,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{allowed}");
    assert!(
        allowed["audit"]
            .as_array()
            .is_some_and(|rows| rows.iter().all(|r| r["outcome"] == "allowed")),
        "outcome=allowed returns only allowed rows: {allowed}"
    );
    let (status, error) = h
        .get(
            &format!("/api/v1/admin/audit?workspace_id={ws}&action=member&outcome=error"),
            &root,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{error}");
    assert_eq!(
        error["audit"].as_array().map(Vec::len),
        Some(1),
        "the no-op removal is found by outcome=error: {error}"
    );
    let (status, _) = h
        .get(&format!("/api/v1/workspaces/{ws}"), &carol_token)
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}

/// Every error a client sees carries a stable `code` beside the message:
/// a handler's own, a core error's, and the framework's (a body that is not
/// JSON, a method the route lacks, a path that does not exist).
#[tokio::test(flavor = "multi_thread")]
async fn error_responses_carry_a_stable_code() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("codes", &owner).await;
    let token = h.login("owner").await;
    let code = |body: &serde_json::Value| body["code"].as_str().unwrap_or_default().to_owned();

    let (status, body) = h.get("/api/v1/workspaces", "not-a-token").await;
    assert_eq!(
        (status, code(&body).as_str()),
        (StatusCode::UNAUTHORIZED, "unauthorized"),
        "{body}"
    );
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &token,
            serde_json::json!({ "sql": "SELECT * FROM no_such_table" }),
        )
        .await;
    assert_eq!(
        (status, code(&body).as_str()),
        (StatusCode::UNPROCESSABLE_ENTITY, "sql_failed"),
        "{body}"
    );
    assert!(
        body["error"].as_str().is_some_and(|m| !m.is_empty()),
        "{body}"
    );
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/query"),
            &token,
            serde_json::json!({ "prompt": "hi" }),
        )
        .await;
    assert_eq!(
        (status, code(&body).as_str()),
        (StatusCode::BAD_REQUEST, "no_chat_model"),
        "{body}"
    );
    let (status, body) = h
        .post(
            "/api/v1/workspaces",
            &token,
            serde_json::json!({ "name": "codes" }),
        )
        .await;
    assert_eq!(
        (status, code(&body).as_str()),
        (StatusCode::FORBIDDEN, "forbidden"),
        "{body}"
    );
    let (status, body) = h
        .get(
            &format!("/api/v1/workspaces/{ws}/sessions/{}", SessionId::generate()),
            &token,
        )
        .await;
    assert_eq!(
        (status, code(&body).as_str()),
        (StatusCode::NOT_FOUND, "not_found"),
        "{body}"
    );

    let request = Request::post(format!("/api/v1/workspaces/{ws}/sql"))
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from("SELECT 1"))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, body, _) = h.send(request).await;
    assert_eq!(
        (status, code(&body).as_str()),
        (StatusCode::UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type"),
        "{body}"
    );
    let (status, body) = h
        .call(
            Method::DELETE,
            &format!("/api/v1/workspaces/{ws}/sql"),
            Some(&token),
            None,
        )
        .await;
    assert_eq!(
        (status, code(&body).as_str()),
        (StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed"),
        "{body}"
    );
    let (status, body) = h.get("/api/v1/no-such-route", &token).await;
    assert_eq!(
        (status, code(&body).as_str()),
        (StatusCode::NOT_FOUND, "not_found"),
        "{body}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_respects_roles_hides_internal_tables_and_records_detail() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let viewer = h.user("viewer", UserKind::Standard).await;
    let ws = h.workspace("data", &owner).await;
    h.app
        .control
        .set_member(&ws, &viewer, Role::Viewer, setup_audit())
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
        .post(
            &format!("/api/v1/workspaces/{ws}/tables/describe"),
            &viewer_token,
            serde_json::json!({ "name": "t" }),
        )
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
    assert!(access_rows.iter().all(|r| r.entry.resource_id.is_none()));
    assert!(
        access_rows.iter().any(
            |r| r.entry.outcome == Outcome::Denied && r.entry.user_id.as_ref() == Some(&viewer)
        )
    );
    assert!(
        access_rows
            .iter()
            .any(|r| r.entry.outcome == Outcome::Error)
    );
    assert!(
        access_rows
            .iter()
            .all(|r| r.entry.origin.request_id.is_some())
    );
}

/// A document's chunks page in order through the API and open on the web
/// passage page, where a citation link lands, with its neighbours linked;
/// a non-member is refused and the refusal is audited.
#[tokio::test(flavor = "multi_thread")]
async fn a_documents_chunks_page_through_the_api_and_open_on_the_passage_page() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let outsider = h.user("outsider", UserKind::Standard).await;
    let ws = h.workspace("docs", &owner).await;
    let db = h
        .app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    db.run(|db| {
        let id = DocumentId::from("d");
        db.insert_document(
            &NewDocument::new(&id, "policy.md", "text/markdown", 1)
                .with_status(DocumentStatus::Ready),
        )?;
        for (i, text) in [
            "Flood is excluded.",
            "Hail is covered.",
            "Claims close in 30 days.",
        ]
        .iter()
        .enumerate()
        {
            db.chunk_writer(&id, text).and_then(|writer| {
                writer.insert(&NewChunk {
                    id: &ChunkId::from(format!("c{i}")),
                    chunk_index: u32::try_from(i).unwrap_or_default(),
                    content: text,
                    heading: (i == 1).then_some("Perils"),
                    page: Some(2),
                    kind: SectionKind::Body,
                    locator: None,
                    embedding: None,
                })
            })?;
        }
        db.set_document_chunk_count(&id, 3)
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
    let token = h.login("owner").await;
    let path = format!("/api/v1/workspaces/{ws}/documents/d/chunks");

    let (status, body) = h.get(&format!("{path}?from=1&limit=1"), &token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["document_id"], "d");
    assert_eq!(body["filename"], "policy.md");
    assert_eq!(body["total"], 3);
    assert_eq!(body["from"], 1);
    let chunks = body["chunks"].as_array().cloned().unwrap_or_default();
    assert_eq!(chunks.len(), 1, "{body}");
    assert_eq!(chunks[0]["chunk_index"], 1);
    assert_eq!(chunks[0]["content"], "Hail is covered.");
    assert_eq!(chunks[0]["heading"], "Perils");
    assert_eq!(chunks[0]["page"], 2);
    let (status, body) = h.get(&path, &token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["chunks"].as_array().map(Vec::len), Some(3));
    let (status, body) = h.get(&format!("{path}?from=3"), &token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["chunks"].as_array().map(Vec::len), Some(0));
    let (status, _) = h
        .get(
            &format!("/api/v1/workspaces/{ws}/documents/nope/chunks"),
            &token,
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let outsider_token = h.login("outsider").await;
    let (status, _) = h.get(&path, &outsider_token).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let opened = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("open")),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        opened.iter().any(|r| r.entry.outcome == Outcome::Allowed
            && r.entry.user_id.as_ref() == Some(&owner)
            && r.entry.resource_id.as_deref() == Some("d")),
        "{opened:?}"
    );
    let denied = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            outcome: Some(Outcome::Denied),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        denied
            .iter()
            .any(|r| r.entry.user_id.as_ref() == Some(&outsider)),
        "{denied:?}"
    );

    // The passage page shows the chunk with links to its neighbours; a
    // position past the end is a 404 page, and a non-member is refused.
    let (_, _, headers) = h.form("/login", None, "username=owner&password=pw").await;
    let cookie = session_cookie(&headers);
    let (status, html, _) = h
        .page(&format!("/w/{ws}/documents/d/chunks/1"), Some(&cookie))
        .await;
    assert_eq!(status, StatusCode::OK, "{html}");
    assert!(html.contains("Hail is covered."), "{html}");
    assert!(html.contains("Chunk 1 of 3"), "{html}");
    assert!(html.contains("Page 2 · Under \"Perils\""), "{html}");
    assert!(
        html.contains(&format!("href=\"/w/{ws}/documents/d/chunks/0\""))
            && html.contains(&format!("href=\"/w/{ws}/documents/d/chunks/2\"")),
        "{html}"
    );
    let (status, html, _) = h
        .page(&format!("/w/{ws}/documents/d/chunks/2"), Some(&cookie))
        .await;
    assert_eq!(status, StatusCode::OK, "{html}");
    assert!(html.contains("End of document"), "{html}");
    assert!(!html.contains("/chunks/3\""), "{html}");
    let (status, html, _) = h
        .page(&format!("/w/{ws}/documents/d/chunks/3"), Some(&cookie))
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{html}");
    let (_, _, headers) = h
        .form("/login", None, "username=outsider&password=pw")
        .await;
    let (status, _, _) = h
        .page(
            &format!("/w/{ws}/documents/d/chunks/0"),
            Some(&session_cookie(&headers)),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}

/// `POST .../documents?replace={doc}` queues one file that takes the
/// document's place once ready: the old one is `superseded`, out of the
/// listing, and named by the new one's audit row; the request refuses
/// more than one file, and the web row's Replace form does the same.
#[tokio::test(flavor = "multi_thread")]
async fn a_replacement_supersedes_its_predecessor_once_ready() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("docs", &owner).await;
    let token = h.login("owner").await;
    let base = format!("/api/v1/workspaces/{ws}/documents");
    let paste = |text: &str, title: &str| serde_json::json!({ "text": text, "title": title });
    let (status, body) = h
        .post(&base, &token, paste("Flood is excluded.", "policy"))
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let old = body["documents"][0]["id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert_eq!(h.wait_ready(&ws, &old, &token).await["status"], "ready");

    let (status, body) = h
        .post(
            &format!("{base}?replace={old}"),
            &token,
            paste("Flood is covered.", "policy"),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let new = body["documents"][0]["id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert_ne!(new, old);
    assert_eq!(h.wait_ready(&ws, &new, &token).await["status"], "ready");
    let (_, replaced) = h.get(&format!("{base}/{old}"), &token).await;
    assert_eq!(replaced["status"], "superseded", "{replaced}");
    assert_eq!(replaced["superseded_by"], new, "{replaced}");
    let (_, listed) = h.get(&base, &token).await;
    let ids: Vec<&str> = listed["documents"]
        .as_array()
        .map(|docs| docs.iter().filter_map(|d| d["id"].as_str()).collect())
        .unwrap_or_default();
    assert_eq!(ids, [new.as_str()], "{listed}");
    let ingests = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("ingest")),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        ingests
            .iter()
            .any(|r| r.entry.resource_id.as_deref() == Some(new.as_str())),
        "{ingests:?}"
    );

    // A replaced document cannot be replaced again, and a replacement
    // is one file.
    let (status, body) = h
        .post(
            &format!("{base}?replace={old}"),
            &token,
            paste("Flood is excluded again.", "policy"),
        )
        .await;
    assert!(
        status.is_client_error() || status.is_server_error(),
        "{body}"
    );
    assert!(body.to_string().contains("superseded, not ready"), "{body}");
    let boundary = "two";
    let two = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.md\"\r\n\r\nA\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"b.md\"\r\n\r\nB\r\n--{boundary}--\r\n"
    );
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("{base}?replace={new}"))
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(two))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, body, _) = h.send(request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.to_string().contains("exactly one file"), "{body}");

    // The web: the live listing hides the replaced row, `?all=true` shows
    // it with its replacement, and the row's Replace form queues a file.
    let (_, _, headers) = h.form("/login", None, "username=owner&password=pw").await;
    let cookie = session_cookie(&headers);
    let (status, html, _) = h.page(&format!("/w/{ws}/documents"), Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!html.contains(&format!("id=\"doc-{old}\"")), "{html}");
    assert!(html.contains("Show replaced documents"), "{html}");
    assert!(
        html.contains(&format!("/documents/{new}/replace")),
        "{html}"
    );
    let (status, html, _) = h
        .page(&format!("/w/{ws}/documents?all=true"), Some(&cookie))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains(&format!("id=\"doc-{old}\"")), "{html}");
    assert!(html.contains(&format!("replaced by {new}")), "{html}");
    let (content_type, bytes) = multipart("policy.md", "text/markdown", "Flood is covered now.");
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("/w/{ws}/documents/{new}/replace"))
        .header(header::COOKIE, format!("quack_session={cookie}"))
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from(bytes))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, _, headers) = h.send(request).await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(location(&headers), format!("/w/{ws}/documents"));
    let (_, pending) = h.get(&format!("{base}/{new}"), &token).await;
    assert!(
        pending["superseded_by"].is_string(),
        "the replacement is on its way: {pending}"
    );
}

/// The `quack_session` cookie a login response set.
fn session_cookie(headers: &axum::http::HeaderMap) -> String {
    headers
        .get(header::SET_COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|c| c.split(';').next())
        .and_then(|c| c.strip_prefix("quack_session="))
        .unwrap_or_default()
        .to_owned()
}

/// The SQL editor's schema: members read every user table's columns as a
/// statement writes them, never an internal table; anyone else is refused,
/// and both are audited.
#[tokio::test(flavor = "multi_thread")]
async fn the_sql_schema_is_for_members_and_names_no_internal_table() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let outsider = h.user("outsider", UserKind::Standard).await;
    let ws = h.workspace("data", &owner).await;
    let owner_token = h.login("owner").await;
    let outsider_token = h.login("outsider").await;
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &owner_token,
            serde_json::json!({ "sql": "CREATE TABLE sales (region VARCHAR, \"Revenue\" INTEGER)" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let path = format!("/api/v1/workspaces/{ws}/tables/schema");
    let (status, body) = h.get(&path, &owner_token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body,
        serde_json::json!({
            "tables": [{
                "name": { "name": "sales", "sql": "sales" },
                "columns": [
                    { "name": "region", "sql": "region" },
                    { "name": "Revenue", "sql": "\"Revenue\"" },
                ],
            }],
            "truncated": false,
        })
    );
    assert!(!body.to_string().contains("_quack_"), "{body}");
    let (status, _) = h.get(&path, &outsider_token).await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    let rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("list")),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        rows.iter().any(
            |r| r.entry.outcome == Outcome::Allowed && r.entry.user_id.as_ref() == Some(&owner)
        ),
        "{rows:?}"
    );
    let denied = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            outcome: Some(Outcome::Denied),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        denied
            .iter()
            .any(|r| r.entry.user_id.as_ref() == Some(&outsider)),
        "{denied:?}"
    );
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
                hold.recv().unwrap_or_default();
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
async fn a_partly_read_document_says_so_over_rest_mcp_and_the_web() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("pages", &owner).await;
    let token = h.login("owner").await;
    let db = h
        .app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    db.run(|db| {
        let id = DocumentId::from("d");
        db.insert_document(
            &NewDocument::new(&id, "scan.pdf", "application/pdf", 1)
                .with_status(DocumentStatus::Ready),
        )?;
        db.set_document_pages(
            &id,
            Some(PageCounts {
                total: 40,
                unreadable: 3,
                empty: 2,
            }),
        )
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
    let counts = serde_json::json!({ "total": 40, "unreadable": 3, "empty": 2 });

    let base = format!("/api/v1/workspaces/{ws}/documents");
    let (status, body) = h.get(&base, &token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["documents"][0]["pages"], counts, "{body}");
    let (status, body) = h.get(&format!("{base}/d"), &token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["pages"], counts, "{body}");

    let session = mcp_session(&h, &ws, &token).await;
    let (status, body, _) = mcp_call(
        &h,
        &ws,
        Some(&token),
        Some(&session),
        rpc(
            2,
            "tools/call",
            &serde_json::json!({ "name": "list_documents", "arguments": {} }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["result"]["structuredContent"]["documents"][0]["pages"], counts,
        "{body}"
    );

    let (status, _, headers) = h.form("/login", None, "username=owner&password=pw").await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    let cookie = headers
        .get(header::SET_COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|c| c.split(';').next())
        .and_then(|c| c.strip_prefix("quack_session="))
        .unwrap_or_default()
        .to_owned();
    let note = "3 of 40 pages unreadable, 2 without text";
    for path in ["documents", "documents/rows", "documents/status"] {
        let (status, html, _) = h.page(&format!("/w/{ws}/{path}"), Some(&cookie)).await;
        assert_eq!(status, StatusCode::OK, "{path}");
        assert!(html.contains(note), "{path}: {html}");
    }
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
        .post(
            &format!("/api/v1/workspaces/{ws}/search"),
            &token,
            serde_json::json!({ "query": "flood" }),
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
        deletes.first().and_then(|r| r.entry.resource_id.clone()),
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
        .create_token(&ws, &owner, "ro", &[Scope::Read], None, setup_audit())
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
        .create_token(&ws, &owner, "rw", &[Scope::Write], None, setup_audit())
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
            .all(|r| r.entry.origin.channel == Channel::Api && r.entry.token_hash.is_some())
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
            setup_audit(),
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
    assert!(
        denied
            .iter()
            .any(|r| r.entry.user_id.as_ref() == Some(&owner))
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn query_endpoints_fail_cleanly_without_a_chat_model() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let viewer = h.user("viewer", UserKind::Standard).await;
    let ws = h.workspace("q", &owner).await;
    h.app
        .control
        .set_member(&ws, &viewer, Role::Viewer, setup_audit())
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
        .post(
            &format!("/api/v1/workspaces/{ws}/search"),
            &owner_token,
            serde_json::json!({ "query": "" }),
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
    assert_eq!(
        errors.first().map(|r| r.entry.outcome.as_str()),
        Some("error")
    );
}

/// An authorized search whose embedding provider is unreachable still writes
/// an `AuditAction::Search` row with `Outcome::Error` and an OCSF `Search`
/// Failure event — the same way `execute_sql` records a failed statement:
/// the post-authorization failure is audited before the error is propagated.
/// Regression test for the `?`-before-audit ordering in the `search` handler.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_authorized_search_is_audited_as_error() {
    let config = Config::parse(
        "[embedding]\nmodel = \"o/e\"\ndimension = 768\n[providers.o]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\n",
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

    // Authorized search whose embeddings call fails post-authorization:
    // READ is granted, but `embed_interactive` cannot reach the provider.
    let (status, _body) = h
        .call(
            Method::POST,
            &format!("/api/v1/workspaces/{ws}/search"),
            None,
            Some(serde_json::json!({ "query": "hello" })),
        )
        .await;
    assert!(
        status.is_server_error(),
        "expected 5xx from the unreachable provider, got {status}"
    );

    // The failure is audited as a Search row with Outcome::Error — the row
    // that the bug dropped entirely by propagating the error before auditing.
    let rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("search")),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        rows.iter().any(|r| r.entry.outcome == Outcome::Error),
        "expected an Outcome::Error Search row for the failed authorized search, got {rows:?}"
    );
    assert!(
        rows.iter().all(|r| r.entry.origin.request_id.is_some()),
        "the row carries the request id, like the sql rows"
    );

    // That row renders as an OCSF API Read Failure (type_uid 600302), the
    // event SIEM consumers key on for a failed read of this class.
    let events = rows
        .iter()
        .map(AuditRow::to_ocsf)
        .collect::<CoreResult<Vec<_>>>()
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        events
            .iter()
            .any(|e| e["type_uid"] == 600_302 && e["status"] == "Failure"),
        "expected a Search Failure OCSF event, got {events:?}"
    );
}

/// The other post-authorization failure family for `search`: a `DuckDB` error
/// during retrieval. With no embedding model the handler runs the keyword
/// path; dropping the terms table the search SQL joins on makes
/// `search_keyword_chunks` error inside the reader transaction. The row is
/// audited as `Outcome::Error` before the 5xx is returned — the DB leg of the
/// same fix, driven end to end through the real HTTP route.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_authorized_search_on_a_db_error_is_audited_as_error() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("searchdb", &owner).await;
    let token = h.login("owner").await;
    let db = h
        .app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    db.run(|db| db.execute_statement("DROP TABLE _quack_terms"))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));

    let (status, _body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/search"),
            &token,
            serde_json::json!({ "query": "hello" }),
        )
        .await;
    assert!(
        status.is_server_error(),
        "expected 5xx from the broken terms table, got {status}"
    );

    let rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws),
            action: Some(String::from("search")),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        rows.iter().any(|r| r.entry.outcome == Outcome::Error),
        "expected an Outcome::Error Search row for the failed DB search, got {rows:?}"
    );
}

/// A graph search or path that fails after authorization is audited as a
/// `Graph` row with `Outcome::Error` before the 5xx is returned, the same
/// ordering as `search`. Dropping the nodes table makes resolving the entry
/// point fail inside the reader transaction.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_authorized_graph_search_is_audited_as_error() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("graphdb", &owner).await;
    let token = h.login("owner").await;
    let db = h
        .app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    db.run(|db| db.execute_statement("DROP TABLE _quack_graph_nodes"))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));

    for (uri, ask) in [
        (
            format!("/api/v1/workspaces/{ws}/graph/search"),
            serde_json::json!({ "entity": "acme" }),
        ),
        (
            format!("/api/v1/workspaces/{ws}/graph/path"),
            serde_json::json!({ "from": "acme", "to": "globex" }),
        ),
    ] {
        let (status, body) = h.post(&uri, &token, ask).await;
        assert!(
            status.is_server_error(),
            "expected 5xx from the missing nodes table for {uri}, got {status}: {body}"
        );
    }

    let rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws),
            action: Some(String::from("graph")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(
        rows.iter()
            .filter(|r| r.entry.outcome == Outcome::Error)
            .count(),
        2,
        "expected an Outcome::Error Graph row for each failed request, got {rows:?}"
    );
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
            .set_member(&ws, u, Role::Viewer, setup_audit())
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
            .filter(|r| r.entry.outcome == Outcome::Denied)
            .count(),
        1
    );
    assert!(
        deletes
            .iter()
            .all(|r| r.entry.resource_type == Some(ResourceKind::Session))
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
        .set_member(&ws, &viewer, Role::Viewer, setup_audit())
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
    let (to, html) = h.land(&headers, Some(&cookie)).await;
    assert_eq!(to, format!("/w/{ws}/ontology"));
    assert!(html.contains("role=\"alert\""), "{html}");
    let (status, _, headers) = h
        .form(&format!("/w/{ws}/ontology/2/restore"), Some(&cookie), "")
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(location(&headers), format!("/w/{ws}/ontology"));
    let (status, html, _) = h.page(&format!("/w/{ws}/ontology"), Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("vendor") && html.contains("v4"), "{html}");
}

/// The ontology page edits a property's meaning and the measures through
/// forms, each a new version through the same save, and shows what the
/// save refused.
#[tokio::test(flavor = "multi_thread")]
async fn ontology_page_forms_edit_property_meanings_and_measures() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let viewer = h.user("viewer", UserKind::Standard).await;
    let ws = h.workspace("o", &owner).await;
    h.app
        .control
        .set_member(&ws, &viewer, Role::Viewer, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    h.app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message))
        .run(|db| db.execute_statement("CREATE TABLE orders AS SELECT 250 AS amount"))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let owner_token = h.login("owner").await;
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/ontology/init"),
            &owner_token,
            serde_json::json!({}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let owner_cookie = h.web_session("owner").await;
    let page = format!("/w/{ws}/ontology");
    let submit = |path: String, form: &'static str| h.submit(path, owner_cookie.clone(), form);

    let (to, html) = submit(
        format!("{page}/properties/country"),
        "description=Where+it+is+based&unit=&synonyms=nation%2C+land%2C+",
    )
    .await;
    assert_eq!(to, page);
    assert!(
        html.contains("described property country; saved as version 2"),
        "{html}"
    );
    assert!(html.contains("Where it is based"), "{html}");
    assert!(html.contains("(also: land, nation)"), "{html}");
    let (status, body) = h
        .get(&format!("/api/v1/workspaces/{ws}/ontology"), &owner_token)
        .await;
    assert_eq!(status, StatusCode::OK);
    let country = body["properties"]
        .as_array()
        .and_then(|p| p.iter().find(|p| p["id"] == "country"))
        .cloned()
        .unwrap_or_default();
    assert_eq!(country["description"], "Where it is based");
    assert_eq!(country.get("unit"), None);
    assert_eq!(country["synonyms"], serde_json::json!(["land", "nation"]));

    let (_, html) = submit(
        format!("{page}/measures"),
        "id=revenue&table=orders&expression=sum(amount)+%2F+100.0&description=",
    )
    .await;
    assert!(
        html.contains("added measure revenue; saved as version 3"),
        "{html}"
    );
    assert!(html.contains("revenue = sum(amount) / 100.0"), "{html}");

    // The save's own check refuses an expression that is not a read of the
    // table, and the page says why; nothing is stored.
    let (_, html) = submit(
        format!("{page}/measures/revenue"),
        "table=orders&expression=sum(missing_column)&description=",
    )
    .await;
    assert!(html.contains("role=\"alert\""), "{html}");
    assert!(
        html.contains("measure &#39;revenue&#39; is not a read of &#39;orders&#39;"),
        "{html}"
    );
    let (_, html) = submit(
        format!("{page}/measures"),
        "id=revenue&table=orders&expression=count(*)",
    )
    .await;
    assert!(html.contains("already exists"), "{html}");
    let (_, html) = submit(format!("{page}/properties/nope"), "description=x").await;
    assert!(html.contains("no property"), "{html}");

    let (_, html) = submit(
        format!("{page}/measures/revenue"),
        "table=orders&expression=sum(amount)&description=Gross+revenue+in+cents",
    )
    .await;
    assert!(
        html.contains("changed measure revenue; saved as version 4"),
        "{html}"
    );
    assert!(
        html.contains("revenue = sum(amount): Gross revenue in cents"),
        "{html}"
    );
    let (_, html) = submit(format!("{page}/measures/revenue/remove"), "").await;
    assert!(
        html.contains("removed measure revenue; saved as version 5"),
        "{html}"
    );
    assert!(!html.contains("revenue = "), "{html}");

    // A viewer sees no forms and may not post them.
    let viewer_cookie = h.web_session("viewer").await;
    let (status, html, _) = h.page(&page, Some(&viewer_cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        !html.contains("/ontology/properties/") && !html.contains("/ontology/measures"),
        "{html}"
    );
    let (status, _, _) = h
        .form(
            &format!("{page}/measures"),
            Some(&viewer_cookie),
            "id=x&table=orders&expression=count(*)",
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let writes = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("ontology")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(writes.len(), 5);
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
    // Three proposals, and the document pass that had no chat model.
    assert_eq!(proposes.len(), 4);
    assert_eq!(
        proposes
            .iter()
            .filter(|r| r.entry.outcome == Outcome::Error)
            .count(),
        1
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn propose_with_auto_accept_builds_a_version_scoped_to_this_run() {
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

    // An earlier plain propose leaves its run's candidates pending.
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{base}/propose"),
            None,
            Some(serde_json::json!({})),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let earlier = body["candidates"].as_u64().unwrap_or(0);
    assert!(earlier > 0, "the earlier run queued candidates: {body}");
    let earlier_run = body["run"].as_str().unwrap_or_default().to_owned();
    assert!(
        body["version"].is_null(),
        "no version without auto-accept: {body}"
    );

    // A later run supersedes the pending candidates it proposes again, so
    // drop `orders`: the next run no longer proposes it, and the earlier
    // run's `orders` candidates stay pending beside it.
    let (status, body) = h
        .call(
            Method::POST,
            &format!("/api/v1/workspaces/{ws}/sql"),
            None,
            Some(serde_json::json!({ "sql": "DROP TABLE orders" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // `auto_accept` queues this run's candidates and accepts them in one
    // call. Scoping acceptance to this run means the version contains
    // exactly what was queued, so the reported count matches the version.
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{base}/propose"),
            None,
            Some(serde_json::json!({ "auto_accept": true })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let queued = body["candidates"].as_u64().unwrap_or(0);
    assert!(queued > 0, "candidates queued and accepted: {body}");
    assert_eq!(body["version"], 1, "auto-accept built a version: {body}");
    assert!(
        body["run"].as_str().is_some(),
        "the run id is returned: {body}"
    );

    let (status, body) = h.call(Method::GET, &base, None, None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["version"], 1, "{body}");

    // Only this run's candidates were accepted: the earlier run's stay
    // pending, untouched (issue #233).
    let (_, body) = h
        .call(Method::GET, &format!("{base}/candidates"), None, None)
        .await;
    let pending = body["candidates"].as_array().cloned().unwrap_or_default();
    assert!(
        !pending.is_empty() && u64::try_from(pending.len()).unwrap_or(u64::MAX) < earlier,
        "the earlier run's orders candidates are still pending: {body}"
    );
    assert!(
        pending
            .iter()
            .all(|c| c["proposed_by"].as_str() == Some(earlier_run.as_str())),
        "every pending candidate is the earlier run's: {body}"
    );

    // A second auto-accept over the now-covered tables finds nothing new.
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{base}/propose"),
            None,
            Some(serde_json::json!({ "auto_accept": true })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["candidates"], 0, "nothing new to propose: {body}");
    assert!(body["version"].is_null(), "no version built: {body}");
    assert!(body["run"].is_null(), "no run queued: {body}");
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

    // One row a page: following next_cursor reaches every login row, and
    // the last page's cursor is null.
    let (_, all) = h.get("/api/v1/admin/audit?action=login", &root).await;
    let total = all["audit"].as_array().map_or(0, Vec::len);
    assert!(total >= 2, "{all}");
    let mut path = String::from("/api/v1/admin/audit?action=login&limit=1");
    let mut walked = Vec::new();
    loop {
        let (status, page) = h.get(&path, &root).await;
        assert_eq!(status, StatusCode::OK, "{page}");
        walked.extend(page["audit"].as_array().cloned().unwrap_or_default());
        match page["next_cursor"].as_str() {
            Some(cursor) => {
                path = format!("/api/v1/admin/audit?action=login&limit=1&cursor={cursor}");
            }
            None => break,
        }
    }
    assert_eq!(walked, all["audit"].as_array().cloned().unwrap_or_default());
    let (_, first) = h
        .get("/api/v1/admin/audit?action=login&limit=1", &root)
        .await;
    let cursor = first["next_cursor"].as_str().unwrap_or_default();
    let (status, _) = h
        .get(
            &format!("/api/v1/admin/audit?action=admin&cursor={cursor}"),
            &root,
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "a cursor keeps its filter");
    let (status, _) = h.get("/api/v1/admin/audit?cursor=nonsense", &root).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // The same page as OCSF events.
    let (status, ocsf) = h
        .get("/api/v1/admin/audit?action=login&format=ocsf", &root)
        .await;
    assert_eq!(status, StatusCode::OK, "{ocsf}");
    let events = ocsf["audit"].as_array().cloned().unwrap_or_default();
    assert_eq!(events.len(), total);
    assert!(
        events
            .iter()
            .all(|e| e["class_uid"] == 3002 && e["metadata"]["version"] == "1.9.0"),
        "{ocsf}"
    );
    let (status, _) = h.get("/api/v1/admin/audit?format=xml", &root).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
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
            .all(|r| r.entry.user_id == Some(UserId::from("local")))
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
    /// Sign `name` in through the login form; the session cookie's value.
    async fn web_session(&self, name: &str) -> String {
        let (_, _, headers) = self
            .form("/login", None, &format!("username={name}&password=pw"))
            .await;
        headers
            .get(header::SET_COOKIE)
            .and_then(|v| v.to_str().ok())
            .and_then(|c| c.split(';').next())
            .and_then(|c| c.strip_prefix("quack_session="))
            .unwrap_or_default()
            .to_owned()
    }

    /// Post `form` to `path` as a signed-in person and land where it
    /// redirects.
    async fn submit(&self, path: String, cookie: String, form: &str) -> (String, String) {
        let (status, _, headers) = self.form(&path, Some(&cookie), form).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        self.land(&headers, Some(&cookie)).await
    }

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

    /// A form's redirect followed as a browser would: where it went, and
    /// that page fetched with the session and the flash cookie the redirect
    /// set, so a message is checked where the person sees it.
    async fn land(
        &self,
        redirect: &axum::http::HeaderMap,
        session: Option<&str>,
    ) -> (String, String) {
        let to = location(redirect);
        let flash = redirect
            .get_all(header::SET_COOKIE)
            .iter()
            .filter_map(|v| v.to_str().ok())
            .find_map(|c| c.strip_prefix("quack_flash="))
            .and_then(|c| c.split(';').next())
            .map(|id| format!("quack_flash={id}"));
        let cookies: Vec<String> = session
            .map(|token| format!("quack_session={token}"))
            .into_iter()
            .chain(flash)
            .collect();
        let mut builder = Request::builder().uri(&to);
        if !cookies.is_empty() {
            builder = builder.header(header::COOKIE, cookies.join("; "));
        }
        let request = builder
            .body(Body::empty())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let (_, body, _) = self.send(request).await;
        (to, body.as_str().unwrap_or_default().to_owned())
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
    let (to, html) = h.land(&headers, None).await;
    assert_eq!(to, "/login");
    assert!(html.contains("wrong username or password"), "{html}");
    // A flash shows once: the page again has no message.
    let (_, html, _) = h.page("/login", None).await;
    assert!(!html.contains("wrong username or password"), "{html}");
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
    assert!(
        html.contains("&#60;(o )___"),
        "the header carries the duck: {html}"
    );
    // Creating it was audited, so it counts as used from then on.
    assert!(
        html.contains("Last used") && html.matches("just now").count() == 2,
        "{html}"
    );
    let (status, html, _) = h.page(&chat_url, Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains("id=\"chat\"") && html.contains(&format!("data-workspace=\"{ws}\"")),
        "{html}"
    );
    assert!(html.contains("Run changes without asking"));
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
    // The poll swaps in statuses and the pending note only: no file names,
    // no table. The upload may still be running, so any status will do.
    let (status, html, _) = h
        .page(&format!("/w/{ws}/documents/status"), Some(&cookie))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains("id=\"status-")
            && html.contains("hx-swap-oob=\"true\"")
            && html.contains("id=\"doc-pending\""),
        "{html}"
    );
    assert!(
        !html.contains("notes.md") && !html.contains("<table"),
        "{html}"
    );

    // The SQL page: the editor loads its schema from the API and enhances
    // the textarea, which stays for a browser without JavaScript.
    let (status, html, _) = h.page(&format!("/w/{ws}/sql"), Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains(&format!(
            "data-schema=\"/api/v1/workspaces/{ws}/tables/schema\""
        )) && html.contains("<script src=\"/static/js/sql-editor.min.js\" defer></script>")
            && html.contains("<textarea id=\"sql-input\""),
        "{html}"
    );
    let (status, _, _) = h.page("/static/js/sql-editor.min.js", None).await;
    assert_eq!(status, StatusCode::OK);

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
        html.contains("aria-sort=\"none\"")
            && html.contains("hx-vals='{\"sort\": 1, \"dir\": \"asc\"}'")
            && html.contains("<div class=\"max-w-xs truncate\" title=\"a\">a</div>")
            && html.contains("Download CSV"),
        "{html}"
    );

    // Rows come back in the statement's own order until a header asks for
    // another; a click sorts by that column, and the next click reverses it.
    let values =
        "SELECT+*+FROM+(VALUES+(2%2C+%27b%27)%2C+(1%2C+%27a%27)%2C+(3%2C+%27c%27))+v(n%2C+s)";
    let order = |html: &str| -> Vec<char> {
        ["a", "b", "c"]
            .iter()
            .filter_map(|v| html.find(&format!("title=\"{v}\"")).map(|at| (at, v)))
            .collect::<std::collections::BTreeMap<_, _>>()
            .into_values()
            .filter_map(|v| v.chars().next())
            .collect()
    };
    let (_, html, _) = h
        .form(
            &format!("/w/{ws}/sql"),
            Some(&cookie),
            &format!("sql={values}"),
        )
        .await;
    assert_eq!(order(&html), ['b', 'a', 'c'], "{html}");
    let (_, html, _) = h
        .form(
            &format!("/w/{ws}/sql"),
            Some(&cookie),
            &format!("sql={values}&sort=1&dir=asc"),
        )
        .await;
    assert_eq!(order(&html), ['a', 'b', 'c'], "{html}");
    // The sort is the statement's own ORDER BY: the editor takes the
    // rewritten SQL, and the CSV link and the next click carry it.
    assert!(
        html.contains("aria-sort=\"ascending\"")
            && html.contains("hx-vals='{\"sort\": 1, \"dir\": \"desc\"}'")
            && html.contains("id=\"sql-input\"")
            && html.contains("hx-swap-oob=\"true\"")
            && html.contains("v(n, s) ORDER BY n ASC NULLS LAST</textarea>")
            && html.contains("ORDER BY n ASC NULLS LAST\">"),
        "{html}"
    );
    let (_, html, _) = h
        .form(
            &format!("/w/{ws}/sql"),
            Some(&cookie),
            &format!("sql={values}&sort=2&dir=desc"),
        )
        .await;
    assert_eq!(order(&html), ['c', 'b', 'a'], "{html}");
    let (_, csv, _) = h
        .form(
            &format!("/w/{ws}/sql.csv"),
            Some(&cookie),
            &format!("sql={values}&sort=1&dir=desc"),
        )
        .await;
    assert_eq!(csv, "n,s\n3,c\n2,b\n1,a\n");
    // A write has no rows to reorder: no sort buttons, and a stray sort is ignored.
    let (_, html, _) = h
        .form(
            &format!("/w/{ws}/sql"),
            Some(&cookie),
            "sql=DROP+TABLE+IF+EXISTS+no_such_table&sort=1&dir=asc",
        )
        .await;
    assert!(
        !html.contains("aria-sort") && !html.contains("role=\"alert\""),
        "{html}"
    );
    // The download is a POST: a statement never travels in a URL.
    let (status, _, _) = h
        .page(&format!("/w/{ws}/sql.csv?sql=SELECT+1+AS+n"), Some(&cookie))
        .await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    let (status, csv, headers) = h
        .form(
            &format!("/w/{ws}/sql.csv"),
            Some(&cookie),
            "sql=SELECT+1+AS+n",
        )
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
    assert!(html.contains("<p role=\"alert\""), "{html}");

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
    assert_eq!(html.matches("aria-current=\"page\"").count(), 1, "{html}");
    assert!(html.contains("aria-current=\"page\">Context</a>"), "{html}");
    assert!(
        html.contains("href=\"#main\"")
            && html.contains("<main id=\"main\"")
            && html.contains("<nav aria-label=\"Workspace\"")
            && html.contains("aria-label=\"Workspace context\""),
        "{html}"
    );
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
    let (refused, html) = h.land(&headers, Some(&cookie)).await;
    assert_eq!(refused, format!("/w/{ws}/settings"));
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
            .unwrap_or_else(|e: CoreError| fail(&e.to_string())),
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
    assert!(text.contains("\n   <(o )___     quack "));
    assert!(text.contains("\n    ( ._> /     knowledge engine"));
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
            .unwrap_or_else(|| serde_json::Value::String(text.to_owned())),
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
        .create_token(&ws, &owner, "ro", &[Scope::Read], None, setup_audit())
        .await
        .map_or_else(
            |e| fail(&e.to_string()),
            |issued| issued.secret.expose().to_owned(),
        );
    let write_token = h
        .app
        .control
        .create_token(
            &ws,
            &owner,
            "rw",
            &[Scope::Read, Scope::Write],
            None,
            setup_audit(),
        )
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
    // control.db's access log names the resource by its template: a table's
    // name, and whatever an unknown URI says, stay in the workspace.
    let opened: Vec<Option<String>> = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("open")),
            ..AuditFilter::default()
        })
        .await
        .into_iter()
        .map(|r| r.entry.resource_id)
        .collect();
    assert!(
        opened.contains(&Some(String::from(
            "quack://workspace/tables/{name}/schema"
        ))) && opened.contains(&None)
            && opened
                .iter()
                .flatten()
                .all(|id| !id.contains("tables/t/") && !id.contains("nothing")),
        "{opened:?}"
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
        sql_rows
            .iter()
            .all(|r| r.entry.origin.channel == Channel::Mcp),
        "{sql_rows:?}"
    );
    assert_eq!(
        sql_rows
            .iter()
            .filter(|r| r.entry.outcome == Outcome::Denied)
            .count(),
        1
    );
}

/// An authorized MCP `search` whose embedding provider is unreachable
/// returns a structured tool `failure` to the client (MCP's normal
/// tool-failure semantics) AND writes an `AuditAction::Search` row with
/// `Outcome::Error` over HTTP — the row the bug dropped by returning the
/// embedding/DB failures before the auditor ever ran.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_authorized_mcp_search_is_audited_as_error() {
    let config = Config::parse(
        "[embedding]\nmodel = \"o/e\"\ndimension = 768\n[providers.o]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\n",
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    let h = harness_with(ServeMode::Login, config).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("search", &owner).await;
    let read_token = h
        .app
        .control
        .create_token(&ws, &owner, "ro", &[Scope::Read], None, setup_audit())
        .await
        .map_or_else(
            |e| fail(&e.to_string()),
            |issued| issued.secret.expose().to_owned(),
        );
    let session = mcp_session(&h, &ws, &read_token).await;
    let (status, body, _) = mcp_call(
        &h,
        &ws,
        Some(&read_token),
        Some(&session),
        rpc(
            2,
            "tools/call",
            &serde_json::json!({ "name": "search", "arguments": { "query": "hello" } }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // MCP keeps tool failures as normal results the client model can read.
    assert_eq!(body["result"]["isError"], true, "{body}");

    let rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("search")),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        rows.iter().any(|r| r.entry.outcome == Outcome::Error),
        "expected an Outcome::Error Search row for the failed authorized MCP search, got {rows:?}"
    );
    assert!(
        rows.iter().all(|r| r.entry.origin.channel == Channel::Mcp),
        "the MCP search rows are audited on the mcp channel, {rows:?}"
    );
}

/// A successful MCP `search` over HTTP returns the chunks as a structured
/// tool result and writes an `Outcome::Allowed` `Search` audit row on the mcp
/// channel — the success-path counterpart to the failure test above, so the
/// refactor does not regress the path that already worked.
#[tokio::test(flavor = "multi_thread")]
async fn a_successful_mcp_search_is_audited_as_allowed() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("mcpok", &owner).await;
    let read_token = h
        .app
        .control
        .create_token(&ws, &owner, "ro", &[Scope::Read], None, setup_audit())
        .await
        .map_or_else(
            |e| fail(&e.to_string()),
            |issued| issued.secret.expose().to_owned(),
        );
    let owner_token = h.login("owner").await;
    // Ingest a document so a keyword search has something to find (no
    // embedding model is configured, so this is the keyword-only path).
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/documents"),
            &owner_token,
            serde_json::json!({ "text": "Flood damage is excluded.", "title": "policy" }),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let doc = body["documents"][0]["id"].as_str().unwrap_or_default();
    h.wait_ready(&ws, doc, &owner_token).await;

    let session = mcp_session(&h, &ws, &read_token).await;
    let (status, body, _) = mcp_call(
        &h,
        &ws,
        Some(&read_token),
        Some(&session),
        rpc(
            2,
            "tools/call",
            &serde_json::json!({ "name": "search", "arguments": { "query": "flood" } }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["isError"], false, "{body}");
    assert_eq!(
        body["result"]["structuredContent"]["chunks"][0]["filename"], "policy.md",
        "{body}"
    );

    let rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws),
            action: Some(String::from("search")),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        rows.iter().any(|r| r.entry.outcome == Outcome::Allowed),
        "expected an Outcome::Allowed Search row for the successful MCP search, got {rows:?}"
    );
    assert!(
        rows.iter().all(|r| r.entry.origin.channel == Channel::Mcp),
        "the MCP search rows are audited on the mcp channel, {rows:?}"
    );
}

/// Concurrent MCP calls by one user with two tokens share one transport
/// (same workspace, user, and write permission), yet each call's audit row
/// carries the token and request id of the request that made it: the
/// `Access` travels with the HTTP request, not on the shared server
/// (issue #245).
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_mcp_calls_by_one_user_audit_their_own_token_and_request_id() {
    const CALLS: usize = 8;
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("mcpconc", &owner).await;
    let issue = |name: &'static str| {
        let h = &h;
        let ws = &ws;
        let owner = &owner;
        async move {
            h.app
                .control
                .create_token(ws, owner, name, &[Scope::Read], None, setup_audit())
                .await
                .unwrap_or_else(|e| fail(&e.to_string()))
        }
    };
    let first = issue("first").await;
    let second = issue("second").await;
    let tokens = [
        (
            first.secret.expose().to_owned(),
            first.row.token_hash.clone(),
        ),
        (
            second.secret.expose().to_owned(),
            second.row.token_hash.clone(),
        ),
    ];
    let session = mcp_session(&h, &ws, &tokens[0].0).await;

    let calls = (0..CALLS).flat_map(|n| {
        tokens
            .iter()
            .enumerate()
            .map(move |(which, (secret, _))| (n, which, secret))
    });
    let responses = futures::future::join_all(calls.map(|(n, which, secret)| {
        let request = Request::builder()
            .method(Method::POST)
            .uri(format!("/mcp/v1/{ws}"))
            .header(header::HOST, "localhost")
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::ACCEPT, "application/json, text/event-stream")
            .header(header::AUTHORIZATION, format!("Bearer {secret}"))
            .header("mcp-session-id", &session)
            .header(super::auth::REQUEST_ID_HEADER, format!("req-{which}-{n}"))
            .body(Body::from(
                rpc(
                    u32::try_from(n * 2 + which + 10).unwrap_or(u32::MAX),
                    "tools/call",
                    &serde_json::json!({ "name": "list_tables", "arguments": {} }),
                )
                .to_string(),
            ))
            .unwrap_or_else(|e| fail(&e.to_string()));
        h.send(request)
    }))
    .await;
    for (status, body, _) in &responses {
        assert_eq!(*status, StatusCode::OK, "{body}");
    }

    let rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("list")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(rows.len(), CALLS * 2, "one row per call, {rows:?}");
    for row in &rows {
        assert_eq!(row.entry.origin.channel, Channel::Mcp, "{row:?}");
        let request_id = row.entry.origin.request_id.as_deref().unwrap_or_default();
        let which = match request_id.split('-').nth(1) {
            Some("0") => 0,
            Some("1") => 1,
            _ => fail(&format!("unexpected request id in {row:?}")),
        };
        let expected = tokens.get(which).map(|(_, hash)| hash.as_str());
        assert_eq!(
            row.entry.token_hash.as_deref(),
            expected,
            "request {request_id} was audited with another request's token"
        );
    }
    let mut ids: Vec<&str> = rows
        .iter()
        .filter_map(|r| r.entry.origin.request_id.as_deref())
        .collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(
        ids.len(),
        CALLS * 2,
        "every request id appears once, {ids:?}"
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
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/tables/describe"),
            &token,
            serde_json::json!({ "name": "customer_secrets" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, _, headers) = h.form("/login", None, "username=owner&password=pw").await;
    let cookie = headers
        .get(header::SET_COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|c| c.split(';').next())
        .and_then(|c| c.strip_prefix("quack_session="))
        .unwrap_or_default()
        .to_owned();
    let (status, _, _) = h
        .form(
            &format!("/w/{ws}/tables"),
            Some(&cookie),
            "name=customer_secrets",
        )
        .await;
    assert_eq!(status, StatusCode::OK);

    let rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            outcome: Some(Outcome::Allowed),
            ..AuditFilter::default()
        })
        .await;
    let lists = rows
        .iter()
        .filter(|r| r.entry.action == AuditAction::List)
        .count();
    assert!(lists >= 7, "{rows:?}");
    // The API's describe and the web table page: two opens, neither
    // naming the table in control.db.
    assert_eq!(
        rows.iter()
            .filter(|r| r.entry.action == AuditAction::Open)
            .count(),
        2,
        "{rows:?}"
    );
    assert!(
        rows.iter()
            .all(|r| r.entry.resource_id.as_deref() != Some("customer_secrets")),
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
    // The harness's setup rows are the CLI's, which writes no detail.
    assert!(
        rows.iter()
            .filter(|r| r.entry.origin.channel != Channel::Cli)
            .all(|r| ids.contains(r.entry.id.as_str())),
        "{rows:?}"
    );
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
        .post(
            &format!("{base}/search"),
            "",
            serde_json::json!({ "entity": "Kenya", "hops": 1 }),
        )
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
    let (status, body) = h
        .post(
            &format!("{base}/search"),
            "",
            serde_json::json!({ "class": "vendor" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["nodes"].as_array().map(Vec::len), Some(2));
    let (status, _) = h
        .post(&format!("{base}/search"), "", serde_json::json!({}))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, body) = h
        .post(
            &format!("{base}/path"),
            "",
            serde_json::json!({ "from": "Uganda", "to": "Aurobindo", "max_hops": 6 }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["edges"].as_array().map(Vec::len), Some(6), "{body}");
    let (_, body) = h
        .post(
            &format!("{base}/path"),
            "",
            serde_json::json!({ "from": "Uganda", "to": "Aurobindo", "max_hops": 2 }),
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
    // A POST without the preview's totals, or with totals the graph no
    // longer matches, drops nothing and names the current ones.
    let (_, preview) = h.get(&format!("{base}/revalidate"), "").await;
    let stale_totals = serde_json::json!({ "dropped_nodes": 2, "dropped_edges": 1 });
    for unconfirmed in [None, Some(serde_json::json!({})), Some(stale_totals)] {
        let (status, body) = h
            .call(
                Method::POST,
                &format!("{base}/revalidate"),
                None,
                unconfirmed.clone(),
            )
            .await;
        assert_eq!(status, StatusCode::CONFLICT, "{unconfirmed:?}: {body}");
        assert!(
            body.to_string()
                .contains("nothing dropped: revalidating now drops 2 nodes and 3 edges"),
            "{unconfirmed:?}: {body}"
        );
        let (_, body) = h.get(&format!("{base}/status"), "").await;
        assert_eq!(body["stale"], true, "{unconfirmed:?}: {body}");
        assert_eq!(body["nodes"], 7, "{unconfirmed:?}: {body}");
    }
    let confirmed = serde_json::json!({
        "dropped_nodes": preview["dropped_nodes"],
        "dropped_edges": preview["dropped_edges"],
    });
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{base}/revalidate"),
            None,
            Some(confirmed),
        )
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
    let (to, html) = h.land(&headers, None).await;
    assert_eq!(to, format!("/w/{ws}/graph"));
    assert!(html.contains("unknown merge decision"), "{html}");

    // The web page renders the status and a search result.
    let (status, html, _) = h
        .form(&format!("/w/{ws}/graph"), None, "entity=Kenya")
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("Knowledge graph"), "{html}");
    assert!(html.contains("5 nodes, 3 edges"), "{html}");
    assert!(
        html.contains("Nothing matched"),
        "Kenya was dropped: {html}"
    );
    let (status, html, _) = h
        .form(&format!("/w/{ws}/graph"), None, "class=vendor")
        .await;
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

/// A rename over the API and the ontology page moves the graph with the
/// id, and a revalidation that would drop something says what and waits
/// for the counts it showed.
#[tokio::test(flavor = "multi_thread")]
async fn renames_move_the_graph_and_revalidation_shows_what_it_drops_first() {
    let h = harness(ServeMode::Local).await;
    let (_, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "r" })),
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
    let ontology_api = format!("/api/v1/workspaces/{ws}/ontology");
    let base = format!("/api/v1/workspaces/{ws}/graph");
    let (status, body) = h
        .call(Method::PUT, &ontology_api, None, Some(ontology))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{base}/extract"),
            None,
            Some(serde_json::json!({ "source": "tables" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // A class renamed over the API: the ontology's references and the
    // graph's nodes follow, and nothing is stale or droppable.
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{ontology_api}/rename"),
            None,
            Some(serde_json::json!({ "kind": "class", "from": "vendor", "to": "supplier" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["version"], 2);
    assert!(
        body["classes"]
            .as_array()
            .is_some_and(|c| c.iter().any(|c| c["id"] == "supplier")
                && !c.iter().any(|c| c["id"] == "vendor")),
        "{body}"
    );
    assert_eq!(
        body["mappings"][0]["relations"][0]["target_class"],
        "supplier"
    );
    let (_, body) = h.get(&format!("{base}/status"), "").await;
    assert_eq!(body["stale"], false, "{body}");
    assert_eq!(
        (&body["nodes"], &body["edges"]),
        (&serde_json::json!(7), &serde_json::json!(6))
    );
    let (_, body) = h
        .post(
            &format!("{base}/search"),
            "",
            serde_json::json!({ "class": "supplier" }),
        )
        .await;
    assert_eq!(body["nodes"].as_array().map(Vec::len), Some(2), "{body}");
    let (status, body) = h.get(&format!("{base}/revalidate"), "").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["dropped_nodes"], 0, "{body}");
    assert_eq!(body["dropped_edges"], 0);

    // An id that exists is refused, as is a kind that has no graph ids.
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{ontology_api}/rename"),
            None,
            Some(serde_json::json!({ "kind": "class", "from": "supplier", "to": "country" })),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body.to_string()
            .contains("class 'country' is declared twice"),
        "{body}"
    );
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{ontology_api}/rename"),
            None,
            Some(serde_json::json!({ "kind": "property", "from": "name", "to": "title" })),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body.to_string()
            .contains("only a class or a relation id can"),
        "{body}"
    );

    // A relation renamed from the ontology page's form.
    let (_, html, _) = h.page(&format!("/w/{ws}/ontology"), None).await;
    assert!(
        html.contains(&format!("action=\"/w/{ws}/ontology/rename\"")),
        "{html}"
    );
    let (status, _, headers) = h
        .form(
            &format!("/w/{ws}/ontology/rename"),
            None,
            "kind=relation&from=supplied_by&to=sourced_from",
        )
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    let (to, html) = h.land(&headers, None).await;
    assert_eq!(to, format!("/w/{ws}/ontology"));
    assert!(
        html.contains("renamed relation supplied_by to sourced_from")
            && html.contains("sourced_from: shipment → supplier"),
        "{html}"
    );
    let (_, _, headers) = h
        .form(
            &format!("/w/{ws}/ontology/rename"),
            None,
            "kind=class&from=nope&to=other",
        )
        .await;
    let (_, html) = h.land(&headers, None).await;
    assert!(
        html.contains("role=\"alert\"") && html.contains("no class &#39;nope&#39; to rename"),
        "{html}"
    );
    let (_, body) = h.get(&format!("{base}/status"), "").await;
    assert_eq!(body["stale"], false, "{body}");
    assert_eq!(body["ontology_version"], 3);
    let writes = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("ontology")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(
        writes.len(),
        3,
        "the import and the two renames: {writes:?}"
    );

    // The ontology loses a class: the preview says what a revalidation
    // drops, and reading it drops nothing.
    let (_, current) = h.get(&ontology_api, "").await;
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
    let (status, body) = h.call(Method::PUT, &ontology_api, None, Some(edited)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = h.get(&format!("{base}/revalidate"), "").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["dropped_nodes"], 2, "{body}");
    assert_eq!(body["dropped_edges"], 3);
    assert_eq!(body["classes"], serde_json::json!({ "country": 2 }));
    assert_eq!(body["relations"], serde_json::json!({ "delivered_to": 3 }));
    assert_eq!(body["version"], 4);
    let (_, body) = h.get(&format!("{base}/status"), "").await;
    assert_eq!(body["nodes"], 7, "{body}");
    assert_eq!(body["stale"], true);

    // The graph page lists the same and its button carries the counts.
    let (_, html, _) = h.page(&format!("/w/{ws}/graph"), None).await;
    assert!(
        html.contains("Revalidating drops <strong>2 nodes and 3 edges</strong>")
            && html.contains(
                "class <span class=\"font-mono\">country</span>, which the ontology no longer defines: 2 nodes"
            )
            && html.contains(
                "relation <span class=\"font-mono\">delivered_to</span>, which the ontology no longer defines: 3 edges"
            )
            && html.contains("name=\"dropped_nodes\" value=\"2\"")
            && html.contains("name=\"dropped_edges\" value=\"3\"")
            && html.contains("Drop 2 nodes and 3 edges and revalidate"),
        "{html}"
    );

    // A post that confirms nothing, or other counts, drops nothing.
    for unconfirmed in ["", "dropped_nodes=2&dropped_edges=1"] {
        let (status, _, headers) = h
            .form(&format!("/w/{ws}/graph/revalidate"), None, unconfirmed)
            .await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        let (to, html) = h.land(&headers, None).await;
        assert_eq!(to, format!("/w/{ws}/graph"));
        assert!(
            html.contains("role=\"alert\"")
                && html.contains("nothing dropped: revalidating now drops 2 nodes and 3 edges"),
            "{unconfirmed:?}: {html}"
        );
        assert!(html.contains("7 nodes, 6 edges"), "{html}");
    }
    let (_, _, headers) = h
        .form(
            &format!("/w/{ws}/graph/revalidate"),
            None,
            "dropped_nodes=2&dropped_edges=3",
        )
        .await;
    let (_, html) = h.land(&headers, None).await;
    assert!(
        html.contains("dropped 2 nodes and 3 edges") && html.contains("5 nodes, 3 edges"),
        "{html}"
    );
    let (_, body) = h.get(&format!("{base}/status"), "").await;
    assert_eq!(body["stale"], false, "{body}");
    let revalidations = h
        .audit(AuditFilter {
            workspace_id: Some(ws),
            action: Some(String::from("graph_revalidate")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(
        revalidations.len(),
        1,
        "refused posts drop and audit nothing"
    );
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
    h.audit_eventually(
        AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("export")),
            ..AuditFilter::default()
        },
        |rows| rows.iter().any(|r| r.entry.outcome == Outcome::Allowed),
    )
    .await;
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
    // The ontology `restore_into` seeded is audited as `ontology` (issue #71):
    // the bundle seeded an ontology into an empty workspace, and the audit
    // log records who did, with the version as the resource.
    let ont = h
        .audit(AuditFilter {
            workspace_id: Some(ws3.clone()),
            action: Some(String::from("ontology")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(
        ont.len(),
        1,
        "one ontology audit row for the restore: {ont:?}"
    );
    assert_eq!(
        ont[0].entry.resource_type,
        Some(ResourceKind::OntologyVersion)
    );
    assert_eq!(ont[0].entry.resource_id.as_deref(), Some("1"));
    assert_eq!(ont[0].entry.outcome, Outcome::Allowed);
    let proposed = h
        .audit(AuditFilter {
            workspace_id: Some(ws3.clone()),
            action: Some(String::from("propose")),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        proposed.is_empty(),
        "no propose row when nothing was proposed: {proposed:?}"
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
    let mut incoming = TarSink::new(Vec::new());
    for (path, content) in [
        (
            "index.md",
            "---\ntype: index\n---\n# Shipping\n\nAll weights in kg.\n",
        ),
        (
            "vendors/orgenics.md",
            "---\ntype: Vendor\ntitle: Orgenics\n---\nShips to [Kenya](../countries/kenya.md).\n",
        ),
        (
            "countries/kenya.md",
            "---\ntype: Country\ntitle: Kenya\n---\nEast Africa.\n",
        ),
    ] {
        incoming
            .file(path, content)
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("/api/v1/workspaces/{ws2}/documents"))
        .header(header::CONTENT_TYPE, "application/x-tar")
        .body(Body::from(
            incoming.finish().unwrap_or_else(|e| fail(&e.to_string())),
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
    // The candidate run `restore_into` stored is audited as `propose` (issue
    // #71), keyed by the run id the pending candidates carry, so the audit
    // trail records who queued them and correlates with the review queue.
    let proposed = h
        .audit(AuditFilter {
            workspace_id: Some(ws2.clone()),
            action: Some(String::from("propose")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(
        proposed.len(),
        1,
        "one propose audit row for the run: {proposed:?}"
    );
    assert_eq!(
        proposed[0].entry.resource_type,
        Some(ResourceKind::InductionRun)
    );
    assert_eq!(proposed[0].entry.outcome, Outcome::Allowed);
    let run_id = proposed[0].entry.resource_id.clone().unwrap_or_default();
    assert!(
        !run_id.is_empty(),
        "the run id is the audit resource: {proposed:?}"
    );
    assert!(
        body["candidates"].as_array().is_some_and(|cs| cs
            .iter()
            .all(|c| c["proposed_by"].as_str() == Some(run_id.as_str()))),
        "the candidates' proposed_by is the audited run id: {body}"
    );
    let ont = h
        .audit(AuditFilter {
            workspace_id: Some(ws2.clone()),
            action: Some(String::from("ontology")),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        ont.is_empty(),
        "no ontology audit row when the bundle carried no snapshot: {ont:?}"
    );
    let ingests = h
        .audit(AuditFilter {
            workspace_id: Some(ws2.clone()),
            action: Some(String::from("ingest")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(
        ingests.len(),
        2,
        "the two ingested documents are audited: {ingests:?}"
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
async fn import_bundle_audits_both_the_ontology_restore_and_the_candidate_run() {
    let h = harness(ServeMode::Local).await;
    let (_, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "mix" })),
        )
        .await;
    let ws = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    // A bundle that carries an ontology snapshot (so it restores) and one
    // foreign concept file (so it also proposes a class): one import does
    // both writes, so it must record both audit rows.
    let mut incoming = TarSink::new(Vec::new());
    let snapshot = "---\ntype: ontology\ngenerator: quack\n---\n# Ontology\n\n```json\n{}\n```\n";
    for (path, content) in [
        ("ontology/ontology.md", snapshot),
        (
            "widgets/acme.md",
            "---\ntype: Widget\ntitle: Acme\n---\nMade of steel.\n",
        ),
    ] {
        incoming
            .file(path, content)
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("/api/v1/workspaces/{ws}/documents"))
        .header(header::CONTENT_TYPE, "application/x-tar")
        .body(Body::from(
            incoming.finish().unwrap_or_else(|e| fail(&e.to_string())),
        ))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, body, _) = h.send(request).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(body["ontology_version"], 1, "{body}");
    assert_eq!(body["candidates"], 1, "{body}");
    assert_eq!(
        body["documents"].as_array().map(Vec::len),
        Some(1),
        "{body}"
    );

    // The ontology restore is audited as `ontology`, keyed by the version.
    let ont = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("ontology")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(ont.len(), 1, "one ontology audit row: {ont:?}");
    assert_eq!(
        ont[0].entry.resource_type,
        Some(ResourceKind::OntologyVersion)
    );
    assert_eq!(ont[0].entry.resource_id.as_deref(), Some("1"));
    assert_eq!(ont[0].entry.outcome, Outcome::Allowed);

    // The candidate run is audited as `propose`, keyed by the run id the
    // pending candidate carries as `proposed_by`.
    let proposed = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("propose")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(proposed.len(), 1, "one propose audit row: {proposed:?}");
    assert_eq!(
        proposed[0].entry.resource_type,
        Some(ResourceKind::InductionRun)
    );
    assert_eq!(proposed[0].entry.outcome, Outcome::Allowed);
    let run_id = proposed[0].entry.resource_id.clone().unwrap_or_default();
    assert!(
        !run_id.is_empty(),
        "the run id is the audit resource: {proposed:?}"
    );
    let (_, body) = h
        .get(&format!("/api/v1/workspaces/{ws}/ontology/candidates"), "")
        .await;
    assert_eq!(
        body["candidates"].as_array().map(Vec::len),
        Some(1),
        "{body}"
    );
    assert_eq!(
        body["candidates"][0]["proposed_by"].as_str(),
        Some(run_id.as_str()),
        "the candidate's proposed_by is the audited run id: {body}"
    );

    // The concept file was ingested as a document and that is audited too.
    let ingests = h
        .audit(AuditFilter {
            workspace_id: Some(ws),
            action: Some(String::from("ingest")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(
        ingests.len(),
        1,
        "the one ingested document is audited: {ingests:?}"
    );
}

/// A signed-in user may not import with the server's own credentials (S3,
/// a bearer token from the server's environment) unless the config allows
/// it; the refusal is a 403 with its code and a denied audit row. Options a
/// source cannot take are a 400.
#[tokio::test]
async fn server_credentials_are_refused_to_signed_in_users() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("olive", UserKind::Standard).await;
    let ws = h.workspace("sales", &owner).await;
    let token = h.login("olive").await;
    let path = format!("/api/v1/workspaces/{ws}/import");
    for body in [
        serde_json::json!({ "url": "s3://bucket/sales.csv", "table": "t" }),
        serde_json::json!({
            "url": "https://example.com/sales.csv",
            "table": "t",
            "bearer_env": "AWS_SECRET_ACCESS_KEY"
        }),
    ] {
        let (status, answer) = h.call(Method::POST, &path, Some(&token), Some(body)).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{answer}");
        assert_eq!(answer["code"], "server_credentials", "{answer}");
        assert!(
            answer["error"]
                .as_str()
                .unwrap_or_default()
                .contains("allow_server_credentials"),
            "{answer}"
        );
    }
    let denied = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("import")),
            outcome: Some(Outcome::Denied),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(denied.len(), 2, "{denied:?}");
    for body in [
        serde_json::json!({
            "url": "sqlite:/tmp/a.db",
            "table": "t",
            "source_table": "t",
            "headers": "X-Api-Key: k"
        }),
        serde_json::json!({
            "url": "https://example.com/a.json",
            "table": "t",
            "json_pointer": "data"
        }),
        serde_json::json!({
            "url": "https://example.com/a.json",
            "table": "t",
            "headers": "no colon here"
        }),
    ] {
        let (status, answer) = h.call(Method::POST, &path, Some(&token), Some(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{answer}");
    }
}

/// A SQLite source file with `rows` vendors, outside the data directory.
async fn vendor_source(dir: &std::path::Path, rows: &str) -> String {
    use sqlx::{Connection as _, Executor as _};
    let path = dir.join("vendors.db");
    let mut conn =
        sqlx::SqliteConnection::connect(&format!("sqlite://{}?mode=rwc", path.display()))
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
    let insert = format!("INSERT INTO vendors VALUES {rows}");
    conn.execute("CREATE TABLE IF NOT EXISTS vendors (id INTEGER, name TEXT)")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    conn.execute("DELETE FROM vendors")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    conn.execute(sqlx::query(sqlx::AssertSqlSafe(insert)))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    format!("sqlite://{}", path.display())
}

/// Start a saved import's refresh and wait for its job.
async fn refresh_job(h: &Harness, path: &str, ws: &WorkspaceId) -> serde_json::Value {
    let (status, answer) = h.call(Method::POST, path, None, None).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{answer}");
    let job = answer["job"].as_str().unwrap_or_default().to_owned();
    wait_for_job(h, ws, &job, "").await
}

/// An import saved over the API is listed, refreshes as a background job
/// (unchanged, then replaced), and can be removed; its table stays.
#[tokio::test(flavor = "multi_thread")]
async fn a_saved_import_lists_refreshes_and_is_removed_over_the_api() {
    let h = harness(ServeMode::Local).await;
    let (_, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "saved-imports" })),
        )
        .await;
    let ws = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    let source_dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let url = vendor_source(source_dir.path(), "(1, 'Orgenics'), (2, 'Aurobindo')").await;
    let imports = format!("/api/v1/workspaces/{ws}/imports");

    let body = serde_json::json!({ "url": url, "table": "vendors", "source_table": "vendors" });
    let (status, answer) = h
        .call(Method::POST, &imports, None, Some(body.clone()))
        .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "a name is needed: {answer}"
    );
    let mut named = body;
    named["save"] = serde_json::json!("vendor list");
    let (status, answer) = h
        .call(Method::POST, &imports, None, Some(named.clone()))
        .await;
    assert_eq!(status, StatusCode::CREATED, "{answer}");
    assert_eq!(answer["rows"], 2);
    assert_eq!(answer["saved"]["name"], "vendor list");
    let id = answer["saved"]["id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let (status, answer) = h.call(Method::POST, &imports, None, Some(named)).await;
    assert_eq!(status, StatusCode::CONFLICT, "{answer}");
    assert_eq!(answer["code"], "saved_import_exists");

    let refresh_path = format!("{imports}/{id}/refresh");
    let job = refresh_job(&h, &refresh_path, &ws).await;
    assert_eq!(job["state"], "succeeded", "{job}");
    assert!(
        job["outcome"]
            .as_str()
            .unwrap_or_default()
            .contains("source unchanged"),
        "{job}"
    );
    vendor_source(
        source_dir.path(),
        "(1, 'Orgenics'), (2, 'Aurobindo'), (3, 'Cipla')",
    )
    .await;
    let job = refresh_job(&h, &refresh_path, &ws).await;
    assert_eq!(job["state"], "succeeded", "{job}");
    let (_, listed) = h.get(&imports, "").await;
    assert_eq!(listed["imports"][0]["last_rows"], 3, "{listed}");
    let (_, rows) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            "",
            serde_json::json!({ "sql": "SELECT count(*) FROM vendors" }),
        )
        .await;
    assert_eq!(rows["rows"][0][0], 3, "{rows}");

    // The Tables page lists it, and its Refresh button queues a job.
    let (_, page, _) = h.page(&format!("/w/{ws}/tables"), None).await;
    assert!(page.contains("vendor list"), "{page}");
    assert!(page.contains(&format!("/imports/{id}/refresh")), "{page}");
    let (status, _, headers) = h
        .form(&format!("/w/{ws}/imports/{id}/refresh"), None, "")
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    let (to, html) = h.land(&headers, None).await;
    assert_eq!(to, format!("/w/{ws}/tables"));
    assert!(
        html.contains("refresh of vendor list queued as job"),
        "{html}"
    );

    let (status, _) = h
        .call(Method::DELETE, &format!("{imports}/{id}"), None, None)
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (_, listed) = h.get(&imports, "").await;
    assert_eq!(listed["imports"], serde_json::json!([]), "{listed}");
    let (status, _) = h
        .call(Method::POST, &format!("{imports}/{id}/refresh"), None, None)
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (_, tables) = h.get(&format!("/api/v1/workspaces/{ws}/tables"), "").await;
    assert_eq!(
        tables["tables"],
        serde_json::json!(["vendors"]),
        "the table stays"
    );
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
    let (to, html) = h.land(&headers, None).await;
    assert_eq!(to, format!("/w/{ws}/tables"));
    assert!(
        html.contains(">vendors2</h1>"),
        "the imported table opens: {html}"
    );
    let (status, _, headers) = h
        .form(
            &format!("/w/{ws}/import"),
            None,
            "url=nope&table=t&source_table=t",
        )
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    let (to, html) = h.land(&headers, None).await;
    assert_eq!(to, format!("/w/{ws}/tables"));
    assert!(html.contains("role=\"alert\""), "{html}");

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
            .filter(|r| r.entry.outcome == Outcome::Error)
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
    assert_eq!(
        denied.first().map(|r| r.entry.outcome.as_str()),
        Some("denied")
    );
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

/// Issue #246: with `[server].secure_cookies = "always"` the session cookie
/// carries `Secure` on loopback too, for a same-host TLS proxy the server
/// cannot otherwise tell from a local browser; the default leaves loopback
/// plain (the test above).
#[tokio::test(flavor = "multi_thread")]
async fn secure_cookies_always_marks_loopback_cookies_secure() {
    let mut config = Config::default();
    config.server.secure_cookies = SecureCookies::Always;
    let h = harness_with(ServeMode::Login, config).await;
    h.user("root", UserKind::Admin).await;
    let (status, _, headers) = h
        .form_from("/login", "127.0.0.1:51000", "username=root&password=pw")
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    let cookie = set_cookie(&headers);
    assert!(cookie.contains("quack_session="), "{cookie}");
    assert!(cookie.contains("Secure"), "{cookie}");
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

/// Issue #237: the limiters key on the peer address, never on a header the
/// caller writes. A fresh random `Authorization` on every attempt once
/// bought a fresh bucket, so the login limiter never refused anyone who
/// bothered to rotate it, while a second address keeps its own budget.
#[tokio::test(flavor = "multi_thread")]
async fn rotating_the_authorization_header_does_not_escape_the_login_limit() {
    let h = harness(ServeMode::Login).await;
    h.user("root", UserKind::Admin).await;
    let attempt = |n: u32, peer: &'static str| {
        let addr: std::net::SocketAddr = peer.parse().unwrap_or_else(|e| fail(&format!("{e}")));
        let mut request = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/auth/login")
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::AUTHORIZATION, format!("Bearer rotated-{n}"))
            .body(Body::from(
                serde_json::json!({ "username": "nobody", "password": "wrong" }).to_string(),
            ))
            .unwrap_or_else(|e| fail(&e.to_string()));
        request
            .extensions_mut()
            .insert(axum::extract::ConnectInfo(addr));
        h.send(request)
    };
    // Concurrent, for the reason the login form test gives.
    let attempts = (0..(super::LOGIN_RATE_BURST + 4)).map(|n| attempt(n, "203.0.113.7:51000"));
    let refused = futures::future::join_all(attempts)
        .await
        .into_iter()
        .filter(|(status, _, _)| *status == StatusCode::TOO_MANY_REQUESTS)
        .count();
    assert!(
        refused >= 3,
        "{refused} of {} attempts were refused; a rotated Authorization header bought a fresh bucket",
        super::LOGIN_RATE_BURST + 4
    );

    // Another address is another caller, with its budget untouched.
    let (status, body, _) = attempt(0, "198.51.100.9:52000").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");

    // The general limiter keys the same way: unvalidated bearers do not buy
    // an address more than its one budget anywhere else either.
    let peer: std::net::SocketAddr = "192.0.2.44:53000"
        .parse()
        .unwrap_or_else(|e| fail(&format!("{e}")));
    let requests = (0..(super::RATE_BURST + 4)).map(|n| {
        let mut request = Request::builder()
            .uri("/api/v1/workspaces")
            .header(header::AUTHORIZATION, format!("Bearer qk_rotated-{n}"))
            .body(Body::empty())
            .unwrap_or_else(|e| fail(&e.to_string()));
        request
            .extensions_mut()
            .insert(axum::extract::ConnectInfo(peer));
        h.send(request)
    });
    let refused = futures::future::join_all(requests)
        .await
        .into_iter()
        .filter(|(status, _, _)| *status == StatusCode::TOO_MANY_REQUESTS)
        .count();
    assert!(
        refused >= 3,
        "{refused} of {} requests were refused; rotated bearers escaped the general limiter",
        super::RATE_BURST + 4
    );
}

/// Static assets sit outside the general limiter: a caller who has spent
/// its budget is refused the API but still gets the stylesheet, so a page
/// it is allowed never renders unstyled.
#[tokio::test(flavor = "multi_thread")]
async fn static_assets_are_not_rate_limited() {
    let h = harness(ServeMode::Login).await;
    let peer: std::net::SocketAddr = "192.0.2.45:53001"
        .parse()
        .unwrap_or_else(|e| fail(&format!("{e}")));
    let request = |uri: &str| {
        let mut request = Request::builder()
            .uri(uri)
            .body(Body::empty())
            .unwrap_or_else(|e| fail(&e.to_string()));
        request
            .extensions_mut()
            .insert(axum::extract::ConnectInfo(peer));
        h.send(request)
    };
    let spent = (0..(super::RATE_BURST + 4)).map(|_| request("/api/v1/workspaces"));
    let refused = futures::future::join_all(spent)
        .await
        .into_iter()
        .filter(|(status, _, _)| *status == StatusCode::TOO_MANY_REQUESTS)
        .count();
    assert!(
        refused >= 3,
        "{refused} requests were refused; the budget was not spent"
    );
    for asset in [
        "/static/css/output.css",
        "/static/js/app.js",
        "/static/favicon.svg",
    ] {
        let (status, _, _) = request(asset).await;
        assert_eq!(status, StatusCode::OK, "{asset}");
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
            .set_member(&ws, user, role, setup_audit())
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

/// A queued upload waits on disk, not in memory, so one request may carry
/// more files than any in-memory line would hold: each is queued and
/// processed, and its spooled bytes are gone once its job ends.
#[tokio::test(flavor = "multi_thread")]
async fn a_large_batch_of_uploads_is_spooled_and_processed() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("batch", &owner).await;
    let token = h.login("owner").await;
    let files = 80;
    let boundary = "quackbatch";
    let body: String = (0..files)
        .map(|n| {
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"note-{n}.md\"\r\nContent-Type: text/markdown\r\n\r\n# Note {n}\n\nNumber {n}.\r\n"
            )
        })
        .chain(std::iter::once(format!("--{boundary}--\r\n")))
        .collect();
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("/api/v1/workspaces/{ws}/documents"))
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, body, _) = h.send(request).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let ids: Vec<String> = body["documents"]
        .as_array()
        .map(|docs| {
            docs.iter()
                .filter_map(|d| d["id"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(ids.len(), files);
    for id in &ids {
        let ready = h.wait_ready(&ws, id, &token).await;
        assert_eq!(ready["status"], "ready", "{ready}");
    }
    let lane = LaneKey::Workspace(JobKind::Ingest, ws.clone());
    for _ in 0..100 {
        if h.app.jobs.lane_active(&lane) == 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    // The spooled bytes go in each job's `when_ended` hook, which runs on
    // its own task after the lane slot is released: wait for it.
    let spool = h.app.config.workspace_uploads_dir(ws.as_str());
    let mut left = usize::MAX;
    for _ in 0..200 {
        left = std::fs::read_dir(&spool).map_or(0, Iterator::count);
        if left == 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert_eq!(left, 0, "spooled uploads left in {}", spool.display());
}

/// What an earlier process spooled and never processed is deleted when the
/// workspace opens; a workspace with nothing spooled opens the same way.
#[test]
fn stale_spooled_uploads_are_cleared() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = Config::default();
    config.general.data_dir = dir.path().to_path_buf();
    let ws = WorkspaceId::from("ws-spool");
    assert!(UploadJob::clear_stale(&config, &ws).is_ok());
    let spool = config.workspace_uploads_dir(ws.as_str());
    std::fs::create_dir_all(&spool).unwrap_or_else(|e| fail(&e.to_string()));
    std::fs::write(spool.join("doc-1"), b"left behind").unwrap_or_else(|e| fail(&e.to_string()));
    assert!(UploadJob::clear_stale(&config, &ws).is_ok());
    assert!(!spool.exists());
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
    config.embedding.dimension = Some(Dimension::new(4));
    config.embedding.model = Some(
        "ollama/embeddinggemma"
            .parse()
            .unwrap_or_else(|e: CoreError| fail(&e.to_string())),
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
            // Nothing listens there: retrying would only wait.
            retry: RetryPolicy::none(),
            ..ProviderConfig::new(ProviderType::Ollama)
        },
    );
    let h = harness_with(ServeMode::Login, config).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let viewer = h.user("viewer", UserKind::Standard).await;
    let ws = h.workspace("vectors", &owner).await;
    h.app
        .control
        .set_member(&ws, &viewer, Role::Viewer, setup_audit())
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
        db.chunk_writer(&DocumentId::from("d"), "levee report")
            .and_then(|writer| {
                writer.insert(&NewChunk {
                    id: &ChunkId::from("c"),
                    chunk_index: 0,
                    content: "levee report",
                    heading: None,
                    page: None,
                    kind: SectionKind::Body,
                    locator: None,
                    embedding: Some(&Vector::from(vec![1.0, 0.0, 0.0, 0.0])),
                })
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
        .filter(|r| r.entry.resource_id.as_deref() == Some(run.as_str()))
        .collect();
    assert_eq!(for_run.len(), 2, "{rows:?}");
    assert!(
        for_run.iter().any(|r| r.entry.outcome == Outcome::Error),
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
    let (_, html) = h.land(&headers, Some(&owner_token)).await;
    assert!(html.contains("background"), "{html}");
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
    let access = h.owner_access(&ws, &owner).await;
    let start = |detail: &'static str| {
        let (app, access) = (Arc::clone(&h.app), access.clone());
        async move {
            BackgroundRun::start(
                &app,
                &access,
                RunKind::GRAPH,
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
            .filter(|r| r.entry.resource_type == Some(ResourceKind::GraphRun))
            .collect();
        if rows.len() == 6 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let outcomes = |run: &RunId| -> Vec<Outcome> {
        let mut found: Vec<Outcome> = rows
            .iter()
            .filter(|r| r.entry.resource_id.as_deref() == Some(run.as_str()))
            .map(|r| r.entry.outcome)
            .collect();
        found.sort_by_key(|o| o.as_str());
        found
    };
    assert_eq!(outcomes(&holder_id), [Outcome::Allowed, Outcome::Allowed]);
    assert_eq!(outcomes(&failing_id), [Outcome::Allowed, Outcome::Error]);
    assert_eq!(outcomes(&queued_id), [Outcome::Allowed, Outcome::Error]);
    assert!(
        rows.iter()
            .all(|r| r.entry.action == AuditAction::GraphExtract)
    );

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

/// The jobs stream never ends on its own, so a stopping server ends it:
/// one open Jobs page cannot hold the process up.
#[tokio::test(flavor = "multi_thread")]
async fn the_jobs_stream_ends_when_the_server_begins_to_stop() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("live", &owner).await;
    let token = h.login("owner").await;
    let request = Request::builder()
        .uri(format!("/api/v1/workspaces/{ws}/jobs/stream"))
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let response = h
        .router
        .clone()
        .oneshot(request)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(response.status(), StatusCode::OK);

    h.app.stopping.cancel();
    let body = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        axum::body::to_bytes(response.into_body(), usize::MAX),
    )
    .await
    .unwrap_or_else(|_| fail("the stream stayed open after the server began to stop"))
    .unwrap_or_else(|e| fail(&e.to_string()));
    let body = String::from_utf8_lossy(&body);
    assert!(body.starts_with("event: jobs\n"), "{body}");
}

/// A stopping server closes what its background work opened: a running
/// run and a queued one both have their closing audit row by the time the
/// queue's shutdown returns, and work submitted afterwards is refused on
/// the record instead of being left open.
#[tokio::test(flavor = "multi_thread")]
async fn a_stopping_server_closes_its_runs_and_refuses_new_work_on_the_record() {
    let config = Config::parse(
        "[general]\nchat_model = \"o/m\"\n[providers.o]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\n",
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    let h = harness_with(ServeMode::Login, config).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("runs", &owner).await;
    let token = h.login("owner").await;
    let access = h.owner_access(&ws, &owner).await;
    let start = || async {
        BackgroundRun::start(&h.app, &access, RunKind::GRAPH, serde_json::json!({}))
            .await
            .unwrap_or_else(|e| fail(&e.message))
    };
    let (harness, workspace) = (&h, &ws);
    let closing_rows = |run: RunId| async move {
        harness
            .audit(AuditFilter {
                workspace_id: Some(workspace.clone()),
                ..AuditFilter::default()
            })
            .await
            .into_iter()
            .filter(|r| r.entry.resource_id.as_deref() == Some(run.as_str()))
            .filter(|r| r.entry.outcome == Outcome::Error)
            .count()
    };

    let running = start().await;
    let running_id = running.id().to_owned();
    let (started, has_started) = tokio::sync::oneshot::channel::<()>();
    running.submit(move |ctx| async move {
        if started.send(()).is_err() {
            return Err(String::from("nobody waited for the start"));
        }
        ctx.cancel_token().cancelled().await;
        Err::<Done, _>(String::from("cancelled"))
    });
    let queued = start().await;
    let queued_id = queued.id().to_owned();
    queued.submit(|_| async { Ok(Done) });
    assert!(has_started.await.is_ok(), "the first run holds the lane");

    h.app.stopping.cancel();
    let grace = std::time::Duration::from_secs(10);
    let left = h.app.jobs.shutdown(grace).await;
    assert!(left.is_empty(), "{left:?}");
    assert_eq!(closing_rows(running_id).await, 1, "the running run closed");
    assert_eq!(closing_rows(queued_id).await, 1, "the queued run closed");

    // A run and an upload that arrive now are taken, refused, and closed.
    let late = start().await;
    let late_id = late.id().to_owned();
    late.submit(|_| async { Ok(Done) });
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/documents"),
            &token,
            serde_json::json!({ "text": "Flood damage is excluded.", "title": "policy" }),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let document = body["documents"][0]["id"].as_str().unwrap_or_default();
    assert!(h.app.jobs.shutdown(grace).await.is_empty());
    assert_eq!(closing_rows(late_id).await, 1, "the refused run closed");
    let (_, body) = h
        .get(
            &format!("/api/v1/workspaces/{ws}/documents/{document}"),
            &token,
        )
        .await;
    assert_eq!(body["status"], "error", "{body}");
    assert!(
        body.to_string()
            .contains("cancelled before processing started"),
        "{body}"
    );

    // A turn asked now never runs, and both forms of the request say why.
    let stopping = "the server is shutting down; the turn ended without an answer";
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/query/stream"),
            &token,
            serde_json::json!({ "prompt": "hi" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // The `error` event's data is the same coded body a failed request has.
    assert_eq!(
        body,
        format!("event: error\ndata: {{\"error\":\"{stopping}\",\"code\":\"busy\"}}\n\n")
    );
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/query"),
            &token,
            serde_json::json!({ "prompt": "hi" }),
        )
        .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(
        body,
        serde_json::json!({ "error": stopping, "code": "busy" }),
        "{body}"
    );
}

/// An MCP `query` turn is cut short by the server stopping, like any other
/// turn: the session keeps the question and the cancelled answer, and the
/// turn is audited.
#[tokio::test(flavor = "multi_thread")]
async fn a_stopping_server_cancels_an_mcp_query_turn_on_the_record() {
    use quack_core::config::Config;
    use quack_core::llm::CANCELLED_NOTE;

    // A model that accepts the connection and never answers.
    let silent = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let address = silent.local_addr().unwrap_or_else(|e| fail(&e.to_string()));
    let config = Config::parse(&format!(
        "[general]\nchat_model = \"silent/model\"\n\
         [providers.silent]\ntype = \"ollama\"\nbase_url = \"http://{address}\"\n"
    ))
    .unwrap_or_else(|e| fail(&e.to_string()));
    let h = harness_with(ServeMode::Login, config).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("notes", &owner).await;
    let token = h.login("owner").await;
    let session = mcp_session(&h, &ws, &token).await;
    let sessions = format!("/api/v1/workspaces/{ws}/sessions");

    let asked = mcp_call(
        &h,
        &ws,
        Some(&token),
        Some(&session),
        rpc(
            2,
            "tools/call",
            &serde_json::json!({ "name": "query", "arguments": { "question": "how many?" } }),
        ),
    );
    // Stop the server once the turn has its session and waits on the model.
    let stopped = async {
        loop {
            let (_, body) = h.get(&sessions, &token).await;
            if body["sessions"].as_array().is_some_and(|s| !s.is_empty()) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        h.app.stopping.cancel();
    };
    let within = std::time::Duration::from_secs(30);
    let ended = tokio::time::timeout(within, async { tokio::join!(asked, stopped) }).await;
    assert!(ended.is_ok(), "the stopping server did not end the turn");

    // The call runs on a task of its own, which records the turn's end
    // after the transport has let go of the request.
    let (_, body) = h.get(&sessions, &token).await;
    let sid = body["sessions"][0]["id"].as_str().unwrap_or_default();
    let mut contents = Vec::new();
    let mut queries = Vec::new();
    for _ in 0..250 {
        let (_, recorded) = h.get(&format!("{sessions}/{sid}"), &token).await;
        contents = recorded["messages"]
            .as_array()
            .map(|messages| {
                messages
                    .iter()
                    .filter_map(|m| m["content"].as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        queries = h
            .audit(AuditFilter {
                workspace_id: Some(ws.clone()),
                action: Some(String::from("query")),
                ..AuditFilter::default()
            })
            .await;
        if contents.len() == 2 && !queries.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert_eq!(contents, ["how many?", CANCELLED_NOTE]);
    assert_eq!(queries.len(), 1, "{queries:?}");
}

/// Closing the state closes each workspace file: the writer finishes and
/// checkpoints, so no write-ahead log is left beside the file.
#[tokio::test(flavor = "multi_thread")]
async fn closing_the_state_checkpoints_each_workspace() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("closing", &owner).await;
    let db = h
        .app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    db.run(|db| db.execute_query("CREATE TABLE t AS SELECT 1 AS n"))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    drop(db);
    let mut log = h.app.config.workspace_db_path(ws.as_str()).into_os_string();
    log.push(".wal");
    let log = std::path::PathBuf::from(log);
    assert!(log.exists(), "the write is still in the log");

    h.app.close().await;
    assert!(!log.exists(), "the log was checkpointed into the file");
}

/// The grid is capped, but the download streams every row: the button
/// says so when the grid was cut, and the file is the whole result either
/// way, under the plain name.
#[tokio::test]
async fn the_csv_download_holds_every_row_past_the_grids_cap() {
    let mut config = Config::default();
    config.analysis.max_query_rows = 3;
    let h = harness_with(ServeMode::Login, config).await;
    let bob = h.user("bob", UserKind::Standard).await;
    let ws = h.workspace("team", &bob).await;
    let cookie = web_session(&h, "bob").await;
    let disposition = |headers: &axum::http::HeaderMap| {
        headers
            .get(header::CONTENT_DISPOSITION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned()
    };

    let capped = "sql=SELECT+*+FROM+range(10)+t(n)";
    let (status, html, _) = h.form(&format!("/w/{ws}/sql"), Some(&cookie), capped).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains("10 rows (showing 3)")
            && html.contains(">Download CSV (all 10 rows)</button>"),
        "{html}"
    );
    let (status, csv, headers) = h
        .form(&format!("/w/{ws}/sql.csv"), Some(&cookie), capped)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(disposition(&headers), "attachment; filename=\"query.csv\"");
    assert_eq!(csv, "n\n0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n");

    // A result within the cap: no note on the button.
    let complete = "sql=SELECT+*+FROM+range(3)+t(n)";
    let (_, html, _) = h
        .form(&format!("/w/{ws}/sql"), Some(&cookie), complete)
        .await;
    assert!(
        html.contains(">Download CSV</button>") && !html.contains("showing"),
        "{html}"
    );
    let (status, csv, _) = h
        .form(&format!("/w/{ws}/sql.csv"), Some(&cookie), complete)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(csv, "n\n0\n1\n2\n");
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

#[tokio::test]
async fn about_page_credits_the_projects_to_every_signed_in_user() {
    let h = harness(ServeMode::Login).await;
    h.user("bob", UserKind::Standard).await;
    let (status, _, headers) = h.page("/about", None).await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    assert_eq!(location(&headers), "/login");

    let cookie = web_session(&h, "bob").await;
    let (status, html, _) = h.page("/about", Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains(env!("CARGO_PKG_VERSION")), "{html}");
    for site in [
        "https://ratatui.rs",
        "https://duckdb.org",
        "https://rig.rs",
        "https://ollama.com",
    ] {
        assert!(
            html.contains(&format!("href=\"{site}\" rel=\"noopener noreferrer\"")),
            "{site}: {html}"
        );
    }
    assert!(
        html.contains("href=\"/about\" aria-current=\"page\""),
        "{html}"
    );
    // Every signed-in page links to it, not just the admin ones.
    let (_, html, _) = h.page("/workspaces", Some(&cookie)).await;
    assert!(html.contains("href=\"/about\""), "{html}");
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
    let (to, html) = h.land(&headers, Some(&cookie)).await;
    assert_eq!(to, "/workspaces");
    assert!(html.contains("workspace name must be non-empty"), "{html}");
    let (_, _, headers) = h.form("/workspaces", Some(&cookie), "name=team").await;
    let ws = location(&headers)
        .trim_start_matches("/w/")
        .trim_end_matches("/chat")
        .to_owned();
    let (_, _, headers) = h.form("/workspaces", Some(&cookie), "name=team").await;
    let (to, html) = h.land(&headers, Some(&cookie)).await;
    assert_eq!(to, "/workspaces");
    assert!(html.contains("already exists"), "{html}");

    // Removing someone who is not a member says so; it used to pass silently.
    let (status, _, headers) = h
        .form(&format!("/w/{ws}/members/nobody/remove"), Some(&cookie), "")
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    let (to, html) = h.land(&headers, Some(&cookie)).await;
    assert_eq!(to, format!("/w/{ws}/settings"));
    assert!(html.contains("not a member"), "{html}");

    // Bulk decisions need at least one candidate.
    let (_, _, headers) = h
        .form(
            &format!("/w/{ws}/ontology/candidates"),
            Some(&cookie),
            "bulk=accept",
        )
        .await;
    let (to, html) = h.land(&headers, Some(&cookie)).await;
    assert_eq!(to, format!("/w/{ws}/ontology"));
    assert!(
        html.contains("choose at least one candidate to accept or reject"),
        "{html}"
    );

    // Proposing over no tables queues nothing and says so.
    let (_, _, headers) = h
        .form(&format!("/w/{ws}/ontology/propose"), Some(&cookie), "")
        .await;
    let (_, html) = h.land(&headers, Some(&cookie)).await;
    assert!(
        html.contains("nothing to propose: the tables are already covered"),
        "{html}"
    );

    // A second init is refused with the API's reason.
    let (_, _, headers) = h
        .form(&format!("/w/{ws}/ontology/init"), Some(&cookie), "")
        .await;
    assert_eq!(location(&headers), format!("/w/{ws}/ontology"));
    let (_, _, headers) = h
        .form(&format!("/w/{ws}/ontology/init"), Some(&cookie), "")
        .await;
    let (to, html) = h.land(&headers, Some(&cookie)).await;
    assert_eq!(to, format!("/w/{ws}/ontology"));
    assert!(html.contains("an ontology already exists"), "{html}");
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

/// Logout ends a session and nothing else: an API token presented to either
/// logout route is a no-op success that records no `logout` row, while a
/// session's logout closes it and is audited (#223).
#[tokio::test(flavor = "multi_thread")]
async fn only_a_session_logout_is_audited_as_a_logout() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("a", &owner).await;
    let IssuedToken { secret, .. } = h
        .app
        .control
        .create_token(&ws, &owner, "ro", &[Scope::Read], None, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let api_token = secret.expose().to_owned();
    let logouts = || {
        h.audit(AuditFilter {
            action: Some(String::from("logout")),
            ..AuditFilter::default()
        })
    };

    let (status, _) = h
        .call(Method::POST, "/api/v1/auth/logout", Some(&api_token), None)
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let web = Request::builder()
        .method(Method::POST)
        .uri("/logout")
        .header(header::AUTHORIZATION, format!("Bearer {api_token}"))
        .body(Body::empty())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, _, headers) = h.send(web).await;
    assert!(status.is_redirection(), "{status}");
    assert_eq!(location(&headers), "/login");
    assert!(logouts().await.is_empty(), "a token ends no session");
    let (status, _) = h.get("/api/v1/auth/me", &api_token).await;
    assert_eq!(status, StatusCode::OK);

    let session = h.login("owner").await;
    let (status, _) = h
        .call(Method::POST, "/api/v1/auth/logout", Some(&session), None)
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = h.get("/api/v1/auth/me", &session).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let rows = logouts().await;
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(
        rows.iter()
            .all(|r| r.entry.token_hash.is_none() && r.entry.outcome == Outcome::Allowed),
        "{rows:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_logout_whose_audit_row_cannot_be_written_leaves_the_session_open() {
    let h = harness(ServeMode::Login).await;
    h.user("owner", UserKind::Standard).await;
    let session = h.login("owner").await;
    let url = format!(
        "sqlite:{}",
        h.app.config.general.data_dir.join("control.db").display()
    );
    let pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect(&url)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    sqlx::query("DROP TABLE audit_log")
        .execute(&pool)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    pool.close().await;

    let (status, _) = h
        .call(Method::POST, "/api/v1/auth/logout", Some(&session), None)
        .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let (status, _) = h.get("/api/v1/auth/me", &session).await;
    assert_eq!(status, StatusCode::OK);
}

/// `list`'s token branch filters out a removed non-admin member's workspace,
/// so `list` agrees with `show` for the same still-valid token, while
/// preserving the admin-without-membership path (role `null`) and the
/// current-member path — including a write-only token's workspace discovery,
/// whose scope dimension is deliberately left untouched. See the report on
/// the `remove_member`/`show` asymmetry that `list`'s token branch missed.
#[tokio::test(flavor = "multi_thread")]
async fn list_token_branch_filters_removed_non_admin_member() {
    let h = harness(ServeMode::Login).await;
    let root_id = h.user("root", UserKind::Admin).await;
    h.user("dir", UserKind::Admin).await;
    let former = h.user("former", UserKind::Standard).await;
    let root = h.login("root").await;
    let dir = h.login("dir").await;

    // `root` owns "leak"; `former` is added as a viewer.
    let (status, body) = h
        .post(
            "/api/v1/workspaces",
            &root,
            serde_json::json!({ "name": "leak" }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let ws = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/members"),
            &root,
            serde_json::json!({ "username": "former", "role": "viewer" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // Operator-mints a Read token for `former` (mirrors admin CLI create_token).
    let IssuedToken { secret, .. } = h
        .app
        .control
        .create_token(&ws, &former, "ro", &[Scope::Read], None, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let former_ro = secret.expose().to_owned();

    // PRE-REMOVAL: the token lists its bound workspace with role "viewer".
    let (status, body) = h.get("/api/v1/workspaces", &former_ro).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["workspaces"][0]["id"], ws.as_str(), "{body}");
    assert_eq!(body["workspaces"][0]["role"], "viewer", "{body}");

    // `remove_member` is the access-revocation operation; it does not revoke tokens.
    let removed = h
        .app
        .control
        .remove_member(&ws, &former, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(removed);

    // FIX: the removed non-admin's still-valid token no longer lists the workspace.
    let (status, body) = h.get("/api/v1/workspaces", &former_ro).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let leaked = body["workspaces"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .find(|w| w["id"] == ws.as_str());
    assert!(
        leaked.is_none(),
        "removed non-admin's token must not list the workspace; got {body}"
    );
    assert_eq!(
        body["workspaces"].as_array().map(Vec::len),
        Some(0),
        "the removed member's bound workspace is filtered out; got {body}"
    );

    // `list` stays unaudited (no denied `open` row), like every other branch.
    let denied_open_after_list = h
        .audit(AuditFilter {
            user_id: Some(former.clone()),
            outcome: Some(Outcome::Denied),
            ..AuditFilter::default()
        })
        .await
        .into_iter()
        .filter(|r| {
            r.entry.action == AuditAction::Open && r.entry.workspace_id.as_ref() == Some(&ws)
        })
        .count();
    assert_eq!(
        denied_open_after_list, 0,
        "list writes no denied `open` row; got {denied_open_after_list}"
    );

    // `show` still refuses the same caller — `list` now agrees with `show`.
    let (status, body) = h.get(&format!("/api/v1/workspaces/{ws}"), &former_ro).await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "show denies the removed member's token; got {body}"
    );
    let denied_open_after_show = h
        .audit(AuditFilter {
            user_id: Some(former.clone()),
            outcome: Some(Outcome::Denied),
            ..AuditFilter::default()
        })
        .await
        .into_iter()
        .filter(|r| {
            r.entry.action == AuditAction::Open && r.entry.workspace_id.as_ref() == Some(&ws)
        })
        .count();
    assert_eq!(
        denied_open_after_show, 1,
        "show writes exactly one denied `open` row; got {denied_open_after_show}"
    );

    // NO REGRESSION — an admin without membership still lists (role `null`),
    // mirroring `show`'s admin path: the `|| identity.is_admin` arm of the filter.
    let (status, body) = h
        .post(
            "/api/v1/workspaces",
            &dir,
            serde_json::json!({ "name": "other" }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let ws2 = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    // `root` is an admin but never a member of `ws2`; operator-mints a token for root.
    let IssuedToken { secret, .. } = h
        .app
        .control
        .create_token(&ws2, &root_id, "ro", &[Scope::Read], None, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let admin_ro = secret.expose().to_owned();
    let (status, body) = h.get("/api/v1/workspaces", &admin_ro).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["workspaces"][0]["id"], ws2.as_str(), "{body}");
    assert_eq!(
        body["workspaces"][0]["role"],
        serde_json::Value::Null,
        "admin-without-membership lists with role null; got {body}"
    );
    let (status, body) = h.get(&format!("/api/v1/workspaces/{ws2}"), &admin_ro).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "show admits a non-member admin's token; got {body}"
    );
    assert_eq!(body["role"], serde_json::Value::Null, "{body}");

    // NO REGRESSION — a current member's write-only token still discovers its
    // workspace (the scope dimension is deliberately untouched by this fix).
    let keeper = h.user("keeper", UserKind::Standard).await;
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/members"),
            &root,
            serde_json::json!({ "username": "keeper", "role": "member" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let IssuedToken { secret, .. } = h
        .app
        .control
        .create_token(&ws, &keeper, "wo", &[Scope::Write], None, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let keeper_wo = secret.expose().to_owned();
    let (status, body) = h.get("/api/v1/workspaces", &keeper_wo).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["workspaces"][0]["id"], ws.as_str(), "{body}");
    assert_eq!(
        body["workspaces"][0]["role"], "member",
        "a current member's write-only token still lists its role; got {body}"
    );
}

/// A workspace whose owner's streamed turns wait on writes, with a member
/// and a viewer beside the owner.
struct WaitingWrites {
    h: Harness,
    ws: WorkspaceId,
    access: Access,
    owner: String,
    other: String,
    viewer: String,
}

impl WaitingWrites {
    async fn new(config: Config) -> Self {
        let h = harness_with(ServeMode::Login, config).await;
        let owner = h.user("owner", UserKind::Standard).await;
        let other = h.user("other", UserKind::Standard).await;
        let viewer = h.user("viewer", UserKind::Standard).await;
        let ws = h.workspace("approvals", &owner).await;
        for (user, role) in [(&other, Role::Member), (&viewer, Role::Viewer)] {
            h.app
                .control
                .set_member(&ws, user, role, setup_audit())
                .await
                .unwrap_or_else(|e| fail(&e.to_string()));
        }
        let workspace = h
            .app
            .control
            .get_workspace(&ws)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
            .unwrap_or_else(|| fail("the workspace was just created"));
        let access = Access {
            identity: Identity {
                user_id: owner,
                username: String::from("owner"),
                kind: UserKind::Standard,
                credential: Credential::Local,
                origin: Origin::from(Channel::Web),
            },
            membership: Membership {
                workspace,
                standing: Standing::Member(Role::Owner),
            },
        };
        Self {
            owner: h.login("owner").await,
            other: h.login("other").await,
            viewer: h.login("viewer").await,
            h,
            ws,
            access,
        }
    }

    /// A turn asks, as `run_sql` does under `WritePolicy::Ask`, and the
    /// server holds the request: its id, and the turn waiting on it.
    async fn ask(&self, sql: &'static str) -> (String, tokio::task::JoinHandle<bool>) {
        use quack_core::analysis::events::{self, AgentEvent, TurnRecorder};
        use quack_core::analysis::policy::Hold;

        let (sink, mut events) = events::channel();
        let asked = tokio::spawn(async move {
            TurnRecorder::new(sink)
                .ask_permission(sql, Hold::NotPermitted)
                .await
        });
        let Some(AgentEvent::PermissionRequired(request)) = events.recv().await else {
            fail("no permission request")
        };
        let held =
            self.h
                .app
                .permissions
                .hold(&self.h.app, &self.access, &SessionId::from("s1"), request);
        (held.request.to_string(), asked)
    }

    async fn decide(&self, request: &str, token: &str, decision: &str) -> StatusCode {
        let path = format!(
            "/api/v1/workspaces/{}/sessions/s1/permissions/{request}",
            self.ws
        );
        let body = serde_json::json!({ "decision": decision });
        self.h.post(&path, token, body).await.0
    }

    /// The outcome of every audited answer, and the audit detail as text.
    async fn audited(&self) -> (Vec<String>, String) {
        let rows = self
            .h
            .audit(AuditFilter {
                workspace_id: Some(self.ws.clone()),
                action: Some(String::from("permission")),
                ..AuditFilter::default()
            })
            .await;
        let (status, body) = self
            .h
            .get(
                &format!("/api/v1/workspaces/{}/audit", self.ws),
                &self.owner,
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        (
            rows.iter()
                .map(|r| r.entry.outcome.as_str().to_owned())
                .collect(),
            body.to_string(),
        )
    }
}

/// A write a streamed turn waits on: only the person who asked may answer,
/// once, and only with write access; and an unknown request is not found.
/// Every answer is audited. The default timeout keeps the expiry clear of
/// the answers.
#[tokio::test(flavor = "multi_thread")]
async fn a_waiting_write_is_answered_once_by_its_asker() {
    let writes = WaitingWrites::new(Config::default()).await;

    let (id, asked) = writes.ask("INSERT INTO t VALUES (1)").await;
    assert_eq!(
        writes.decide(&id, &writes.viewer, "allow").await,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        writes.decide(&id, &writes.other, "allow").await,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        writes.decide("nope", &writes.owner, "allow").await,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        writes.decide(&id, &writes.owner, "allow").await,
        StatusCode::NO_CONTENT
    );
    assert!(asked.await.unwrap_or_default(), "the write runs");
    assert_eq!(
        writes.decide(&id, &writes.owner, "deny").await,
        StatusCode::CONFLICT
    );

    let (id, asked) = writes.ask("DELETE FROM t").await;
    assert_eq!(
        writes.decide(&id, &writes.owner, "deny").await,
        StatusCode::NO_CONTENT
    );
    assert!(!asked.await.unwrap_or(true), "the write is refused");

    // The turn stopped waiting (its stream went away) before the answer:
    // the allow reaches no turn, so nothing ran and the caller is told so,
    // and the row records the answer under decision "gone" like an expiry.
    let (id, asked) = writes.ask("UPDATE t SET a = 1").await;
    asked.abort();
    assert!(asked.await.is_err_and(|e| e.is_cancelled()));
    assert_eq!(
        writes.decide(&id, &writes.owner, "allow").await,
        StatusCode::GONE
    );
    assert_eq!(
        writes.decide(&id, &writes.owner, "allow").await,
        StatusCode::CONFLICT
    );

    let (outcomes, details) = writes.audited().await;
    // allow; deny; the allow of an ended turn; and the refused answers
    // (other, unknown, the second answers; the viewer is refused before it).
    assert_eq!(
        outcomes.iter().filter(|o| *o == "allowed").count(),
        1,
        "{outcomes:?}"
    );
    assert!(
        outcomes.iter().filter(|o| *o == "denied").count() >= 6,
        "{outcomes:?}"
    );
    for decision in ["\"allow\"", "\"deny\"", "\"gone\""] {
        assert!(details.contains(decision), "{decision}: {details}");
    }
    assert!(details.contains("UPDATE t SET a = 1"), "{details}");
}

/// `allow_write` lets a turn write until it has read document text. A
/// non-streamed turn cannot ask, so its write after a search is refused; a
/// streamed one asks the person with the reason, and their answer runs it.
#[tokio::test(flavor = "multi_thread")]
async fn a_turn_that_read_a_document_asks_or_refuses_its_write_under_allow_write() {
    use futures::StreamExt;

    use crate::scripted_ollama::{self, ScriptedOllama};
    use quack_core::analysis::policy::Hold;
    use quack_core::storage::workspace::WorkspaceDb;

    let mut script = ScriptedOllama::following_the_note();
    script.extend(ScriptedOllama::following_the_note());
    let ollama = ScriptedOllama::serve(script)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let config = ollama.config().unwrap_or_else(|e| fail(&e.to_string()));
    let h = harness_with(ServeMode::Login, config).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("notes", &owner).await;
    let token = h.login("owner").await;
    let db = h
        .app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    with_db(Arc::clone(&db), |db| {
        scripted_ollama::seed_dictating_note(db)
    })
    .await
    .unwrap_or_else(|e| fail(&e.message));
    let tables = || async {
        with_db(Arc::clone(&db), WorkspaceDb::list_tables)
            .await
            .unwrap_or_else(|e| fail(&e.message))
    };
    let body = serde_json::json!({ "prompt": "follow the maintenance note", "allow_write": true });
    let path = format!("/api/v1/workspaces/{ws}/query");

    let (status, answer) = h.post(&path, &token, body.clone()).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(answer["write_refused"], true, "{answer}");
    assert_eq!(
        answer["steps"][1]["summary"], "refused: this turn read document text",
        "{answer}"
    );
    assert_eq!(
        tables().await,
        ["customers"],
        "the dictated drop did not run"
    );

    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("{path}/stream"))
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let response = h
        .router
        .clone()
        .oneshot(request)
        .await
        .unwrap_or_else(|e| fail(&format!("request failed: {e}")));
    assert_eq!(response.status(), StatusCode::OK);
    let mut frames = response.into_body().into_data_stream();
    let mut seen = String::new();
    // The data line of the first `event` in what has streamed so far.
    let data_of = |seen: &str, event: &str| -> Option<serde_json::Value> {
        let (_, rest) = seen.split_once(&format!("event: {event}\ndata: "))?;
        let (data, _) = rest.split_once("\n\n")?;
        serde_json::from_str(data).ok()
    };
    let mut asked = None;
    let complete = loop {
        let Some(frame) = frames.next().await else {
            fail(&format!("the stream ended early: {seen}"));
        };
        let frame = frame.unwrap_or_else(|e| fail(&e.to_string()));
        seen.push_str(&String::from_utf8_lossy(&frame));
        if asked.is_none()
            && let Some(request) = data_of(&seen, "permission_required")
        {
            assert_eq!(request["sql"], scripted_ollama::DICTATED, "{request}");
            assert_eq!(request["reason"], "read_documents", "{request}");
            assert_eq!(
                request["notice"],
                Hold::ReadDocuments.notice().unwrap_or_default(),
                "{request}"
            );
            assert_eq!(
                tables().await,
                ["customers"],
                "nothing ran before the answer"
            );
            let decide = format!(
                "/api/v1/workspaces/{ws}/sessions/{}/permissions/{}",
                request["session_id"].as_str().unwrap_or_default(),
                request["request"].as_str().unwrap_or_default()
            );
            let (status, body) = h
                .post(&decide, &token, serde_json::json!({ "decision": "allow" }))
                .await;
            assert_eq!(status, StatusCode::NO_CONTENT, "{body}");
            asked = Some(request);
        }
        if let Some(complete) = data_of(&seen, "complete") {
            break complete;
        }
    };
    assert!(asked.is_some(), "the write was asked for: {seen}");
    assert_eq!(complete["write_refused"], false, "{complete}");
    assert!(tables().await.is_empty(), "the approved drop ran");
}

/// Nobody answers a waiting write: it is refused when the time is up, the
/// refusal is audited, and the request is then unknown.
#[tokio::test(flavor = "multi_thread")]
async fn an_unanswered_write_expires() {
    let mut config = Config::default();
    config.server.permission_timeout_seconds = 1;
    let writes = WaitingWrites::new(config).await;

    let (id, asked) = writes.ask("DROP TABLE t").await;
    assert!(
        !asked.await.unwrap_or(true),
        "an unanswered write is refused"
    );
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert_eq!(
        writes.decide(&id, &writes.owner, "allow").await,
        StatusCode::NOT_FOUND
    );

    let (outcomes, details) = writes.audited().await;
    assert!(
        outcomes.iter().filter(|o| *o == "denied").count() >= 2,
        "{outcomes:?}"
    );
    assert!(details.contains("\"expired\""), "{details}");
    assert!(details.contains("DROP TABLE t"), "{details}");
}

/// A workspace restricted to one provider, on a server whose embedding
/// model is on another: every path that would embed is refused before
/// anything is sent, and each refusal is a denied audit row for its action.
#[tokio::test(flavor = "multi_thread")]
async fn a_restricted_workspace_refuses_every_path_to_a_disallowed_provider() {
    // Both providers are unreachable: a request that got past the check
    // would fail as a 5xx, not as the 403 asserted below.
    let config = Config::parse(
        "[general]\nchat_model = \"local/chat\"\n\
         [embedding]\nmodel = \"hosted/embed\"\ndimension = 768\n\
         [providers.local]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\n\
         [providers.hosted]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\n",
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    let h = harness_with(ServeMode::Login, config).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("kept", &owner).await;
    let token = h.login("owner").await;
    let base = format!("/api/v1/workspaces/{ws}");
    let (status, body) = h
        .post(
            &format!("{base}/ontology/init"),
            &token,
            serde_json::json!({}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = h
        .call(
            Method::PATCH,
            &base,
            Some(&token),
            Some(serde_json::json!({ "allowed_providers": ["local"] })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // A login session's rows are on the web channel.
    let refusal = "provider 'hosted' is not allowed in this workspace, which allows only: local";
    let denied = async |action: &str, channel: Channel| {
        let rows = h
            .audit(AuditFilter {
                workspace_id: Some(ws.clone()),
                action: Some(action.to_owned()),
                ..AuditFilter::default()
            })
            .await;
        rows.iter()
            .filter(|r| r.entry.outcome == Outcome::Denied && r.entry.origin.channel == channel)
            .count()
    };
    for (path, request, action) in [
        (
            "documents",
            serde_json::json!({ "text": "Flood damage is excluded.", "title": "policy" }),
            "ingest",
        ),
        ("search", serde_json::json!({ "query": "flood" }), "search"),
        (
            "graph/search",
            serde_json::json!({ "entity": "Kenya" }),
            "graph",
        ),
        (
            "graph/path",
            serde_json::json!({ "from": "Kenya", "to": "Uganda" }),
            "graph",
        ),
        ("graph/extract", serde_json::json!({}), "graph_extract"),
        (
            "ontology/propose",
            serde_json::json!({ "documents": true }),
            "propose",
        ),
        (
            "embeddings/refresh",
            serde_json::json!({}),
            "embeddings_refresh",
        ),
        (
            "import",
            serde_json::json!({ "url": "https://example.com/rows.csv", "table": "rows" }),
            "import",
        ),
        (
            "query",
            serde_json::json!({ "prompt": "what is excluded?" }),
            "query",
        ),
    ] {
        let before = denied(action, Channel::Web).await;
        let (status, body) = h.post(&format!("{base}/{path}"), &token, request).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{path}: {body}");
        assert_eq!(body["error"], refusal, "{path}");
        assert_eq!(
            denied(action, Channel::Web).await,
            before.saturating_add(1),
            "{path} writes one denied {action} row"
        );
    }
    // Nothing was registered, and the refused turn left no session behind.
    let (_, documents) = h.get(&format!("{base}/documents"), &token).await;
    assert_eq!(documents["documents"], serde_json::json!([]), "{documents}");
    let (_, sessions) = h.get(&format!("{base}/sessions"), &token).await;
    assert_eq!(sessions["sessions"], serde_json::json!([]), "{sessions}");

    // The same turn as a stream ends in an `error` event and a denied row.
    let before = denied("query", Channel::Web).await;
    let (status, events) = h
        .post(
            &format!("{base}/query/stream"),
            &token,
            serde_json::json!({ "prompt": "what is excluded?" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{events}");
    let events = events.as_str().unwrap_or_default();
    assert!(
        events.contains("event: error") && events.contains(refusal),
        "{events}"
    );
    assert_eq!(
        denied("query", Channel::Web).await,
        before.saturating_add(1)
    );

    // MCP over HTTP: the tools answer with the refusal as a tool error.
    let session = mcp_session(&h, &ws, &token).await;
    for (id, tool, arguments, action) in [
        (
            2,
            "query",
            serde_json::json!({ "question": "what is excluded?" }),
            "query",
        ),
        (
            3,
            "search",
            serde_json::json!({ "query": "flood" }),
            "search",
        ),
        (
            4,
            "search_graph",
            serde_json::json!({ "entity": "Kenya" }),
            "graph",
        ),
        (
            5,
            "find_path",
            serde_json::json!({ "from": "Kenya", "to": "Uganda" }),
            "graph",
        ),
    ] {
        let before = denied(action, Channel::Mcp).await;
        let (status, body, _) = mcp_call(
            &h,
            &ws,
            Some(&token),
            Some(&session),
            rpc(
                id,
                "tools/call",
                &serde_json::json!({ "name": tool, "arguments": arguments }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{tool}: {body}");
        assert_eq!(body["result"]["isError"], true, "{tool}: {body}");
        let text = body["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or_default();
        assert!(text.contains(refusal), "{tool}: {text}");
        assert_eq!(
            denied(action, Channel::Mcp).await,
            before.saturating_add(1),
            "MCP {tool} writes one denied {action} row"
        );
    }

    // Lifting the restriction lets the same search through to the provider,
    // which is unreachable here: a failure, no longer a refusal.
    let (status, body) = h
        .call(
            Method::PATCH,
            &base,
            Some(&token),
            Some(serde_json::json!({ "allowed_providers": [] })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = h
        .post(
            &format!("{base}/search"),
            &token,
            serde_json::json!({ "query": "flood" }),
        )
        .await;
    assert!(status.is_server_error(), "{status}: {body}");
}

/// Saved questions over the API: saving needs the source session to be
/// visible, anyone who may read runs and lists, a run says whether the
/// data changed, a failed run is answered and audited as such, and
/// removal follows a session's rule.
#[tokio::test(flavor = "multi_thread")]
async fn saved_questions_are_saved_run_and_removed_over_the_api() {
    use quack_core::analysis::agent::AgentResponse;
    use quack_core::analysis::events::{ToolName, ToolStep};
    use quack_core::storage::sessions::{self, ChatMode};
    const OVERDUE: &str = "SELECT id FROM invoices WHERE NOT paid ORDER BY id";
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let viewer = h.user("viewer", UserKind::Standard).await;
    let other = h.user("other", UserKind::Standard).await;
    let ws = h.workspace("s", &owner).await;
    for u in [&viewer, &other] {
        h.app
            .control
            .set_member(&ws, u, Role::Viewer, setup_audit())
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    let db = h
        .app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    let (answered, no_sql) = {
        let viewer = viewer.clone();
        db.run(move |db| {
            db.execute_statement(
                "CREATE TABLE invoices (id INTEGER, paid BOOLEAN); \
                 INSERT INTO invoices VALUES (1, false), (2, true)",
            )?;
            let answered = sessions::create_session(db, "m", ChatMode::Query, Some(&viewer))?;
            let response = AgentResponse {
                content: String::from("Invoice 1 is overdue."),
                steps: vec![ToolStep {
                    tool: ToolName::RunSql,
                    detail: String::from(OVERDUE),
                    summary: String::from("1 rows"),
                    rows: Some(1),
                    result: None,
                    duration_ms: 1,
                }],
                ..AgentResponse::default()
            };
            sessions::record_turn(
                db,
                &answered.id,
                "overdue?",
                jiff::Timestamp::now(),
                &response,
            )?;
            let no_sql = sessions::create_session(db, "m", ChatMode::Chat, Some(&viewer))?;
            sessions::record_turn(
                db,
                &no_sql.id,
                "hello",
                jiff::Timestamp::now(),
                &AgentResponse {
                    content: String::from("Hello."),
                    ..AgentResponse::default()
                },
            )?;
            Ok((answered.id, no_sql.id))
        })
        .await
        .unwrap_or_else(|e| fail(&e.to_string()))
    };
    let viewer_token = h.login("viewer").await;
    let other_token = h.login("other").await;
    let owner_token = h.login("owner").await;
    let base = format!("/api/v1/workspaces/{ws}/saved");

    // Another member cannot see the session, so cannot save from it.
    let (status, _) = h
        .post(
            &base,
            &other_token,
            serde_json::json!({ "name": "overdue", "session_id": answered }),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, body) = h
        .post(
            &base,
            &viewer_token,
            serde_json::json!({ "name": "overdue", "session_id": answered }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["name"], "overdue");
    assert_eq!(body["statements"], serde_json::json!([OVERDUE]));
    assert_eq!(body["created_by"], viewer.as_str());
    let saved = body["id"].as_str().unwrap_or_default().to_owned();
    let (status, _) = h
        .post(
            &base,
            &viewer_token,
            serde_json::json!({ "name": "overdue", "session_id": answered }),
        )
        .await;
    assert_eq!(status, StatusCode::CONFLICT, "the name is taken");
    let (status, body) = h
        .post(
            &base,
            &viewer_token,
            serde_json::json!({ "name": "hello", "session_id": no_sql }),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert!(
        body["error"]
            .as_str()
            .is_some_and(|e| e.contains("ran no SQL"))
    );
    let (status, body) = h
        .post(
            &base,
            &viewer_token,
            serde_json::json!({ "name": "q", "session_id": answered, "message": 1 }),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    let saves = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("save")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(saves.len(), 4, "one allowed, three refused: {saves:?}");
    assert_eq!(
        saves
            .iter()
            .filter(|r| r.entry.outcome == Outcome::Error)
            .count(),
        3
    );

    // Anyone who may read lists, shows, and runs.
    let (status, body) = h.get(&base, &other_token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["saved"][0]["id"], saved);
    let (status, body) = h.get(&format!("{base}/{saved}"), &other_token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["question"]["name"], "overdue");
    assert_eq!(body["last_run"], serde_json::Value::Null);
    let (status, body) = h
        .post(
            &format!("{base}/{saved}/run"),
            &other_token,
            serde_json::json!({}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "ok");
    assert_eq!(body["changed"], false);
    assert_eq!(body["statements"][0]["rows"], 1);
    assert_eq!(body["statements"][0]["result"], serde_json::json!([[1]]));
    let first_run = body["id"].as_str().unwrap_or_default().to_owned();
    db.run(|db| db.execute_statement("UPDATE invoices SET paid = false"))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, body) = h
        .post(
            &format!("{base}/{saved}/run"),
            &viewer_token,
            serde_json::json!({}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["changed"], true);
    assert_eq!(body["statements"][0]["rows"], 2);
    assert_eq!(
        body["statements"][0]["result"],
        serde_json::json!([[1], [2]])
    );
    let (status, body) = h.get(&format!("{base}/{saved}/runs"), &viewer_token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["runs"].as_array().map(Vec::len), Some(2));
    assert_eq!(body["runs"][1]["id"], first_run, "newest first");
    assert_eq!(
        body["runs"][0]["statements"][0]["result"],
        serde_json::Value::Null,
        "rows are answered once, never stored"
    );
    let (status, body) = h
        .get(&format!("{base}/{saved}/runs?limit=1"), &viewer_token)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["runs"].as_array().map(Vec::len), Some(1));
    let (status, body) = h.get(&format!("{base}/{saved}"), &viewer_token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["last_run"]["changed"], true);

    // A run whose table is gone is answered, with the error, and audited
    // as an error; it is not an unchanged result.
    db.run(|db| db.execute_statement("DROP TABLE invoices"))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (status, body) = h
        .post(
            &format!("{base}/{saved}/run"),
            &viewer_token,
            serde_json::json!({}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "failed");
    assert_eq!(body["changed"], false);
    assert!(
        body["statements"][0]["error"]
            .as_str()
            .is_some_and(|e| e.contains("invoices")),
        "{body}"
    );
    let runs = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("saved_run")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(runs.len(), 3, "{runs:?}");
    assert_eq!(
        runs.iter()
            .filter(|r| r.entry.outcome == Outcome::Error)
            .count(),
        1
    );
    assert!(
        runs.iter()
            .all(|r| r.entry.resource_type == Some(ResourceKind::SavedQuestion))
    );

    // Removal: the creator or an owner; anyone else is refused and audited.
    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("{base}/{saved}"),
            Some(&other_token),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("{base}/{saved}"),
            Some(&owner_token),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("{base}/{saved}"),
            Some(&viewer_token),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = h
        .post(
            &format!("{base}/nope/run"),
            &viewer_token,
            serde_json::json!({}),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (_, body) = h.get(&base, &viewer_token).await;
    assert_eq!(body["saved"], serde_json::json!([]));
    let deletes = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("delete")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(deletes.len(), 2, "one denied, one allowed: {deletes:?}");
    assert_eq!(
        deletes
            .iter()
            .filter(|r| r.entry.outcome == Outcome::Denied)
            .count(),
        1
    );
    let detail = db
        .run(|db| audit::list(db, 100))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        detail.iter().any(|row| row.action == "saved_run"),
        "the workspace keeps the run's detail row"
    );
}

/// The shipments ontology the graph tests build from: three classes
/// keyed by name or purchase order, two relations, one mapping.
fn shipments_ontology() -> serde_json::Value {
    serde_json::json!({
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
    })
}

/// A person's node and edge writes over REST and the graph page: each
/// checked against the ontology, recorded as asserted by them, audited
/// as `graph_edit`, and shown with its assertion.
#[tokio::test(flavor = "multi_thread")]
async fn a_person_edits_graph_nodes_and_edges_over_the_api_and_the_page() {
    let h = harness(ServeMode::Local).await;
    let (_, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "edits" })),
        )
        .await;
    let ws = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    for sql in [
        "CREATE TABLE shipments (po TEXT, vendor TEXT, country TEXT)",
        "INSERT INTO shipments VALUES ('PO-1', 'Orgenics', 'Kenya')",
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
    let (status, body) = h
        .call(
            Method::PUT,
            &format!("/api/v1/workspaces/{ws}/ontology"),
            None,
            Some(shipments_ontology()),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let base = format!("/api/v1/workspaces/{ws}/graph");
    let (_, status_body) = h.get(&format!("{base}/status"), "").await;
    assert_eq!(
        status_body["pending_tables"],
        serde_json::json!(["shipments"]),
        "{status_body}"
    );
    assert_eq!(status_body["pending_chunks"], 0);
    let (status, body) = h
        .call(
            Method::POST,
            &format!("{base}/extract"),
            None,
            Some(serde_json::json!({ "source": "tables" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, status_body) = h.get(&format!("{base}/status"), "").await;
    assert_eq!(
        status_body["pending_tables"],
        serde_json::json!([]),
        "{status_body}"
    );

    // A node: 201 new, 200 when it was there, 400 outside the ontology.
    let (status, body) = h
        .post(
            &format!("{base}/nodes"),
            "",
            serde_json::json!({ "label": "Acme", "class": "vendor", "properties": { "name": "Acme" }, "note": "vendor list" }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["created"], true);
    let acme = body["id"].as_str().unwrap_or_default().to_owned();
    let (status, body) = h
        .post(
            &format!("{base}/nodes"),
            "",
            serde_json::json!({ "label": "ACME", "class": "vendor" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["created"], false);
    assert_eq!(body["id"], acme);
    let (status, body) = h
        .post(
            &format!("{base}/nodes"),
            "",
            serde_json::json!({ "label": "MV Hope", "class": "vessel" }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    // A correction, and an empty one.
    let (status, body) = h
        .call(
            Method::PATCH,
            &format!("{base}/nodes/{acme}"),
            None,
            Some(
                serde_json::json!({ "label": "Acme Pharma", "properties": { "country": "Kenya" } }),
            ),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["label"], "Acme Pharma");
    assert_eq!(body["properties"]["country"], "Kenya");
    let (status, _) = h
        .call(
            Method::PATCH,
            &format!("{base}/nodes/{acme}"),
            None,
            Some(serde_json::json!({})),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // An edge the ontology allows, and one it does not.
    let (_, body) = h
        .post(
            &format!("{base}/search"),
            "",
            serde_json::json!({ "class": "shipment" }),
        )
        .await;
    let po1 = body["nodes"][0]["id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let (status, body) = h
        .post(
            &format!("{base}/edges"),
            "",
            serde_json::json!({ "source": po1, "target": acme, "relation": "supplied_by", "note": "corrected supplier" }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let edge = body["id"].as_str().unwrap_or_default().to_owned();
    let (status, body) = h
        .post(
            &format!("{base}/edges"),
            "",
            serde_json::json!({ "source": acme, "target": po1, "relation": "supplied_by" }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.to_string().contains("does not allow"), "{body}");

    // The search shows the assertion as provenance, with its author.
    let (_, body) = h
        .post(
            &format!("{base}/search"),
            "",
            serde_json::json!({ "entity": "Acme Pharma", "hops": 1 }),
        )
        .await;
    let asserted: Vec<&serde_json::Value> = body["provenance"]
        .as_array()
        .map(|p| p.iter().filter(|p| p["asserted_at"].is_string()).collect())
        .unwrap_or_default();
    assert_eq!(asserted.len(), 2, "{body}");
    assert!(asserted.iter().all(|p| p["author"].is_string()), "{body}");
    assert!(
        asserted.iter().any(|p| p["note"] == "corrected supplier"),
        "{body}"
    );
    let (status, html, _) = h
        .form(&format!("/w/{ws}/graph"), None, "entity=Acme+Pharma")
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("asserted by"), "{html}");
    assert!(html.contains("Delete node and its edges"), "{html}");

    // The page's forms: an edit, then a delete.
    let (status, _, headers) = h
        .form(
            &format!("/w/{ws}/graph/nodes/{acme}"),
            None,
            "label=&class=&properties=country%3D%0Aregion%3DEast+Africa&note=page",
        )
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    let (_, html) = h.land(&headers, None).await;
    assert!(html.contains("updated Acme Pharma (vendor)"), "{html}");
    let (_, body) = h
        .post(
            &format!("{base}/search"),
            "",
            serde_json::json!({ "entity": "Acme Pharma", "hops": 0 }),
        )
        .await;
    assert_eq!(
        body["nodes"][0]["properties"]["region"], "East Africa",
        "{body}"
    );
    assert!(
        body["nodes"][0]["properties"]["country"].is_null(),
        "{body}"
    );
    let (status, _, headers) = h
        .form(&format!("/w/{ws}/graph/edges/{edge}/delete"), None, "")
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    let (_, html) = h.land(&headers, None).await;
    assert!(html.contains("deleted the supplied_by edge"), "{html}");

    // Deletes over REST: the node goes with what is left at it; a second
    // delete is 404.
    let (status, body) = h
        .call(Method::DELETE, &format!("{base}/nodes/{acme}"), None, None)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, _) = h
        .call(Method::DELETE, &format!("{base}/nodes/{acme}"), None, None)
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = h
        .call(Method::DELETE, &format!("{base}/edges/{edge}"), None, None)
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let edits = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("graph_edit")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(edits.len(), 7, "{edits:?}");
    assert!(
        edits
            .iter()
            .any(|r| r.entry.resource_id.as_deref() == Some(edge.as_str())
                && r.entry.resource_type == Some(ResourceKind::GraphEdge)),
        "{edits:?}"
    );
    assert!(
        edits
            .iter()
            .any(|r| r.entry.resource_id.as_deref() == Some(acme.as_str())
                && r.entry.resource_type == Some(ResourceKind::GraphNode)),
        "{edits:?}"
    );
    let db = h
        .app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    let details = with_db(db, |db| audit::list(db, 100))
        .await
        .unwrap_or_else(|e| fail(&e.message));
    let ops: Vec<&str> = details
        .iter()
        .filter(|d| d.action == "graph_edit")
        .filter_map(|d| d.detail.as_ref()?["op"].as_str())
        .collect();
    for op in ["create", "assert", "update", "delete"] {
        assert!(ops.contains(&op), "{op} missing from {ops:?}");
    }
    assert!(
        details.iter().any(|d| d.action == "graph_edit"
            && d.detail
                .as_ref()
                .is_some_and(|v| v["note"] == "corrected supplier")),
        "{details:?}"
    );
}

/// With `[graph].follow_ingest = "tables"`, a table file that replaces a
/// mapped table queues a graph run once it is ready; the ingest job
/// names it, it is audited as a graph extraction, and the status no
/// longer lists the table as pending.
#[tokio::test(flavor = "multi_thread")]
async fn an_upload_queues_the_graph_follow_up_the_setting_asks_for() {
    let mut config = Config::default();
    config.graph.follow_ingest = FollowIngest::Tables;
    let h = harness_with(ServeMode::Local, config).await;
    let (_, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "follow" })),
        )
        .await;
    let ws = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    let base = format!("/api/v1/workspaces/{ws}/documents");
    let upload = |h: &Harness, path: String, csv: &str| {
        let (content_type, body) = multipart("shipments.csv", "text/csv", csv);
        let request = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(body))
            .unwrap_or_else(|e| fail(&e.to_string()));
        let h = h.router.clone();
        async move {
            let response = h
                .oneshot(request)
                .await
                .unwrap_or_else(|e| fail(&e.to_string()));
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap_or_else(|e| fail(&e.to_string()));
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
            (status, body)
        }
    };
    // No ontology yet: the first upload makes the table and nothing follows.
    let (status, body) = upload(&h, base.clone(), "po,vendor,country\nPO-1,Orgenics,Kenya\n").await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let first = body["documents"][0]["id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let first_job = body["documents"][0]["job"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert_eq!(h.wait_ready(&ws, &first, "").await["status"], "ready");
    let done = wait_for_job(&h, &ws, &first_job, "").await;
    assert_eq!(done["state"], "succeeded", "{done}");
    assert!(
        !done["outcome"]
            .as_str()
            .unwrap_or_default()
            .contains("follow-up"),
        "{done}"
    );
    let (status, body) = h
        .call(
            Method::PUT,
            &format!("/api/v1/workspaces/{ws}/ontology"),
            None,
            Some(shipments_ontology()),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // A replacement with a new row: the graph follows.
    let (status, body) = upload(
        &h,
        format!("{base}?replace={first}"),
        "po,vendor,country\nPO-1,Orgenics,Kenya\nPO-2,Aurobindo,Uganda\n",
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let second = body["documents"][0]["id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let job = body["documents"][0]["job"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert_eq!(h.wait_ready(&ws, &second, "").await["status"], "ready");
    let done = wait_for_job(&h, &ws, &job, "").await;
    let result = done["outcome"].as_str().unwrap_or_default().to_owned();
    let follow = result
        .rsplit("graph follow-up queued as job ")
        .next()
        .filter(|rest| rest.len() < result.len())
        .unwrap_or_else(|| fail(&format!("no follow-up in {result}")))
        .to_owned();
    let followed = wait_for_job(&h, &ws, &follow, "").await;
    assert_eq!(followed["state"], "succeeded", "{followed}");
    assert_eq!(followed["kind"], "graph", "{followed}");
    assert_eq!(
        followed["outcome"], "graph: table shipments: 2 nodes, 4 edges",
        "{followed}"
    );
    let (_, status_body) = h
        .get(&format!("/api/v1/workspaces/{ws}/graph/status"), "")
        .await;
    assert_eq!(status_body["nodes"], 6, "{status_body}");
    assert_eq!(
        status_body["pending_tables"],
        serde_json::json!([]),
        "{status_body}"
    );
    let runs = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("graph_extract")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(runs.len(), 2, "{runs:?}");
    assert_eq!(
        runs[0].entry.resource_id, runs[1].entry.resource_id,
        "{runs:?}"
    );
    let db = h
        .app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    let details = with_db(db, |db| audit::list(db, 100))
        .await
        .unwrap_or_else(|e| fail(&e.message));
    let extracts: Vec<&serde_json::Value> = details
        .iter()
        .filter(|d| d.action == "graph_extract")
        .filter_map(|d| d.detail.as_ref())
        .collect();
    assert!(
        extracts.iter().any(|d| d["follow_ingest"] == "tables"),
        "{extracts:?}"
    );
    assert!(
        extracts.iter().any(|d| d["finished"] == true),
        "{extracts:?}"
    );
}

/// `POST .../sql/export` streams every row past the grid's cap in the
/// asked format, refuses a write, and audits the export with its row
/// count; the SQL page's download takes the same path.
#[tokio::test(flavor = "multi_thread")]
async fn sql_export_streams_every_row_and_refuses_writes() {
    let mut config = Config::default();
    config.analysis.max_query_rows = 10;
    let h = harness_with(ServeMode::Login, config).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("sales", &owner).await;
    let token = h.login("owner").await;
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &token,
            serde_json::json!({ "sql": "CREATE TABLE big AS SELECT range AS n FROM range(1000)" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let export = |sql: &str, format: &str| {
        Request::builder()
            .method(Method::POST)
            .uri(format!("/api/v1/workspaces/{ws}/sql/export"))
            .header(header::AUTHORIZATION, format!("Bearer {token}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::json!({ "sql": sql, "format": format }).to_string(),
            ))
            .unwrap_or_else(|e| fail(&e.to_string()))
    };
    let (status, bytes, headers) = h
        .send_bytes(export("SELECT n FROM big ORDER BY n", "csv"))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        headers
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("text/csv; charset=utf-8")
    );
    let text = String::from_utf8_lossy(&bytes);
    assert_eq!(text.lines().count(), 1001, "a header and every row");
    assert!(
        text.ends_with("999\n"),
        "{}",
        text.lines().last().unwrap_or_default()
    );
    let (status, bytes, _) = h
        .send_bytes(export("SELECT n FROM big WHERE n < 3", "ndjson"))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        String::from_utf8_lossy(&bytes),
        "{\"n\":0}\n{\"n\":1}\n{\"n\":2}\n"
    );
    let (status, bytes, _) = h.send_bytes(export("DELETE FROM big", "csv")).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let (status, _, _) = h
        .send_bytes(export("SELECT * FROM _quack_documents", "csv"))
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    // The audit row lands once the stream has ended.
    let mut exports = Vec::new();
    for _ in 0..50 {
        exports = h
            .audit(AuditFilter {
                action: Some(String::from("export")),
                ..AuditFilter::default()
            })
            .await;
        if exports.len() >= 3 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert_eq!(exports.len(), 3, "{exports:?}");
    assert!(exports.iter().any(|r| r.entry.outcome == Outcome::Denied));

    // The web download streams the whole result too.
    let (_, _, headers) = h.form("/login", None, "username=owner&password=pw").await;
    let cookie = session_cookie(&headers);
    let (status, bytes, _) = h
        .send_bytes(
            Request::builder()
                .method(Method::POST)
                .uri(format!("/w/{ws}/sql.csv"))
                .header(header::COOKIE, format!("quack_session={cookie}"))
                .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
                .body(Body::from("sql=SELECT+n+FROM+big"))
                .unwrap_or_else(|e| fail(&e.to_string())),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(String::from_utf8_lossy(&bytes).lines().count(), 1001);
}

/// An admin who is not a member may add themself only with a reason; the
/// grant is a `break_glass` row whose detail names the role, the reason,
/// and that admin rights were used. Any other grant is a `member` row with
/// the role in its detail.
#[tokio::test(flavor = "multi_thread")]
async fn an_admin_self_grant_needs_a_reason_and_is_marked_break_glass() {
    let h = harness(ServeMode::Login).await;
    let root_id = h.user("root", UserKind::Admin).await;
    let owner_id = h.user("owner", UserKind::Standard).await;
    h.user("vera", UserKind::Standard).await;
    let ws = h.workspace("finance", &owner_id).await;
    let root = h.login("root").await;
    let owner = h.login("owner").await;
    let members = format!("/api/v1/workspaces/{ws}/members");
    let (status, body) = h
        .post(
            &members,
            &root,
            serde_json::json!({ "username": "root", "role": "owner" }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("reason"),
        "{body}"
    );
    let (status, body) = h
        .post(
            &members,
            &root,
            serde_json::json!({ "username": "root", "role": "owner", "reason": "  " }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, body) = h
        .post(
            &members,
            &root,
            serde_json::json!({ "username": "root", "role": "owner", "reason": "incident 42" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // An owner adding someone else needs no reason and is a plain member row.
    let (status, body) = h
        .post(
            &members,
            &owner,
            serde_json::json!({ "username": "vera", "role": "viewer" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let glass = h
        .audit(AuditFilter {
            action: Some(String::from("break_glass")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(glass.len(), 1, "{glass:?}");
    assert_eq!(glass[0].entry.user_id.as_ref(), Some(&root_id));
    let details = h
        .app
        .read(&ws, |db| audit::list(db, 50))
        .await
        .unwrap_or_else(|e| fail(&e.message));
    let by_id = |id: &AuditId| {
        details
            .iter()
            .find(|d| &d.id == id)
            .and_then(|d| d.detail.clone())
    };
    let detail = by_id(&glass[0].entry.id).unwrap_or_default();
    assert_eq!(detail["role"], "owner", "{detail}");
    assert_eq!(detail["reason"], "incident 42");
    assert_eq!(detail["acting_as"], "admin");
    let member_rows = h
        .audit(AuditFilter {
            action: Some(String::from("member")),
            workspace_id: Some(ws.clone()),
            ..AuditFilter::default()
        })
        .await;
    let plain = member_rows
        .iter()
        .find(|r| r.entry.user_id.as_ref() == Some(&owner_id))
        .unwrap_or_else(|| fail("the owner's grant is audited"));
    assert_eq!(by_id(&plain.entry.id).unwrap_or_default()["role"], "viewer");

    // The workspace's OCSF export joins both halves: the self-grant is a
    // Medium Create event, and a query event would carry the ai profile.
    let (status, body) = h
        .get(
            &format!("/api/v1/workspaces/{ws}/audit?format=ocsf"),
            &owner,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let events = body["audit"].as_array().cloned().unwrap_or_default();
    let event = events
        .iter()
        .find(|e| e["api"]["operation"] == "break_glass")
        .unwrap_or_else(|| fail("the break_glass event is exported"));
    assert_eq!(event["severity_id"], 3, "{event}");
    assert_eq!(
        event["unmapped"]["detail"]["reason"], "incident 42",
        "{event}"
    );
    assert!(
        events
            .iter()
            .all(|e| e["class_uid"].is_number() && e["metadata"]["version"] == "1.9.0"),
        "{body}"
    );
    let (status, _) = h
        .get(&format!("/api/v1/workspaces/{ws}/audit?format=xml"), &owner)
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// An admin disables, re-enables, demotes, resets, and removes a user over
/// the API: a disabled user's session and token are refused with a denied
/// row, a removed user's name becomes "removed" in the workspace file, and
/// wrong passwords lock the account for the configured minutes.
#[tokio::test(flavor = "multi_thread")]
async fn users_are_disabled_locked_out_and_removed_over_the_api() {
    let mut config = Config::default();
    config.server.login_lockout_attempts = 2;
    let h = harness_with(ServeMode::Login, config).await;
    let root_id = h.user("root", UserKind::Admin).await;
    let bob_id = h.user("bob", UserKind::Standard).await;
    let ws = h.workspace("sales", &bob_id).await;
    let root = h.login("root").await;
    let bob = h.login("bob").await;
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &bob,
            serde_json::json!({ "sql": "CREATE TABLE t AS SELECT 1 AS a" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let token = h
        .app
        .control
        .create_token(&ws, &bob_id, "script", &[Scope::Read], None, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()))
        .secret
        .expose()
        .to_owned();

    // A standard user may not; an admin may not disable themselves.
    let (status, _) = h
        .call(
            Method::PATCH,
            &format!("/api/v1/admin/users/{root_id}"),
            Some(&bob),
            Some(serde_json::json!({ "disabled": true })),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = h
        .call(
            Method::PATCH,
            &format!("/api/v1/admin/users/{root_id}"),
            Some(&root),
            Some(serde_json::json!({ "disabled": true })),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Disabled: the session, the token, and a login are all refused, each
    // with a denied row.
    let (status, body) = h
        .call(
            Method::PATCH,
            &format!("/api/v1/admin/users/{bob_id}"),
            Some(&root),
            Some(serde_json::json!({ "disabled": true })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["disabled_at"].is_string(), "{body}");
    let (status, _) = h.get("/api/v1/auth/me", &bob).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, _) = h.get(&format!("/api/v1/workspaces/{ws}"), &token).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, _) = h
        .call(
            Method::POST,
            "/api/v1/auth/login",
            None,
            Some(serde_json::json!({ "username": "bob", "password": "pw" })),
        )
        .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let denied = h
        .audit(AuditFilter {
            user_id: Some(bob_id.clone()),
            outcome: Some(Outcome::Denied),
            ..AuditFilter::default()
        })
        .await;
    assert!(denied.len() >= 2, "{denied:?}");

    // Enabled again with a new password and the admin flag.
    let (status, body) = h
        .call(
            Method::PATCH,
            &format!("/api/v1/admin/users/{bob_id}"),
            Some(&root),
            Some(serde_json::json!({ "disabled": false, "password": "pw2", "is_admin": true })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["is_admin"], true, "{body}");
    let (status, _) = h.get(&format!("/api/v1/workspaces/{ws}"), &token).await;
    assert_eq!(status, StatusCode::OK);

    // Two wrong passwords lock the account; the right one then waits.
    for _ in 0..2 {
        let (status, _) = h
            .call(
                Method::POST,
                "/api/v1/auth/login",
                None,
                Some(serde_json::json!({ "username": "bob", "password": "nope" })),
            )
            .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }
    let (status, body) = h
        .call(
            Method::POST,
            "/api/v1/auth/login",
            None,
            Some(serde_json::json!({ "username": "bob", "password": "pw2" })),
        )
        .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("try again"),
        "{body}"
    );
    let (status, _) = h
        .call(
            Method::PATCH,
            &format!("/api/v1/admin/users/{bob_id}"),
            Some(&root),
            Some(serde_json::json!({ "disabled": false })),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let bob = h.login_with("bob", "pw2").await;

    // A person changes their own password; the other session ends, the
    // one that changed it stays.
    let other = h.login_with("bob", "pw2").await;
    let (status, _) = h
        .post(
            "/api/v1/auth/password",
            &bob,
            serde_json::json!({ "current": "wrong", "new": "pw3" }),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = h
        .post(
            "/api/v1/auth/password",
            &bob,
            serde_json::json!({ "current": "pw2", "new": "pw3" }),
        )
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = h.get("/api/v1/auth/me", &other).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, _) = h.get("/api/v1/auth/me", &bob).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = h
        .call(Method::POST, "/api/v1/auth/logout-all", Some(&bob), None)
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = h.get("/api/v1/auth/me", &bob).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // Removed: the row, the token, and the name in the workspace file.
    let (status, body) = h
        .call(
            Method::DELETE,
            &format!("/api/v1/admin/users/{bob_id}"),
            Some(&root),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        body["rows_forgotten"].as_u64().unwrap_or_default() >= 1,
        "{body}"
    );
    let (status, _) = h.get(&format!("/api/v1/workspaces/{ws}"), &token).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, body) = h.get("/api/v1/admin/users", &root).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["users"].as_array().map(Vec::len), Some(1), "{body}");
    let named = h
        .app
        .read(&ws, |db| {
            Ok(audit::list(db, 50)?
                .into_iter()
                .filter_map(|row| row.user_id)
                .collect::<Vec<_>>())
        })
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(
        !named.is_empty() && named.iter().all(|n| *n != bob_id),
        "{named:?}"
    );
    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("/api/v1/admin/users/{bob_id}"),
            Some(&root),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// Group roles are set, listed, and removed over the API by owners, and
/// shown on the Settings page only when a groups claim is configured.
#[tokio::test(flavor = "multi_thread")]
async fn group_roles_are_managed_over_the_api() {
    let h = harness(ServeMode::Login).await;
    let owner_id = h.user("owner", UserKind::Standard).await;
    let viewer_id = h.user("viewer", UserKind::Standard).await;
    let ws = h.workspace("sales", &owner_id).await;
    h.app
        .control
        .set_member(&ws, &viewer_id, Role::Viewer, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let owner = h.login("owner").await;
    let viewer = h.login("viewer").await;
    let base = format!("/api/v1/workspaces/{ws}/groups");
    let (status, _) = h
        .post(&base, &viewer, serde_json::json!({ "group": "finance" }))
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, body) = h
        .post(
            &base,
            &owner,
            serde_json::json!({ "group": "finance", "role": "owner" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["group_name"], "finance");
    assert_eq!(body["role"], "owner");
    let (status, body) = h
        .post(
            &base,
            &owner,
            serde_json::json!({ "group": "finance", "role": "viewer" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["role"], "viewer");
    let (status, _) = h
        .post(&base, &owner, serde_json::json!({ "group": " " }))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, body) = h.get(&base, &viewer).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["groups"].as_array().map(Vec::len), Some(1));
    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("{base}/finance"),
            Some(&owner),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("{base}/finance"),
            Some(&owner),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let members = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("member")),
            ..AuditFilter::default()
        })
        .await;
    assert!(members.len() >= 3, "{members:?}");
}

/// Forwarded headers name the client only when the peer is a trusted
/// proxy; from anyone else the peer is the client, whatever it says.
#[tokio::test(flavor = "multi_thread")]
async fn forwarded_headers_count_only_from_trusted_proxies() {
    let mut config = Config::default();
    config.server.trusted_proxies = vec![
        "10.0.0.0/8"
            .parse()
            .unwrap_or_else(|e: ipnet::AddrParseError| fail(&e.to_string())),
    ]
    .into();
    let h = harness_with(ServeMode::Login, config).await;
    h.user("ann", UserKind::Standard).await;
    let login = |peer: &str, forwarded: Option<&str>| {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/auth/login")
            .header(header::CONTENT_TYPE, "application/json")
            .extension(axum::extract::ConnectInfo(
                peer.parse::<std::net::SocketAddr>()
                    .unwrap_or_else(|e| fail(&e.to_string())),
            ));
        if let Some(chain) = forwarded {
            builder = builder.header("x-forwarded-for", chain);
        }
        builder
            .body(Body::from(
                serde_json::json!({ "username": "ann", "password": "pw" }).to_string(),
            ))
            .unwrap_or_else(|e| fail(&e.to_string()))
    };
    let (status, _, _) = h
        .send(login("10.0.0.1:5000", Some("203.0.113.9, 10.0.0.2")))
        .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _, _) = h
        .send(login("198.51.100.4:5000", Some("203.0.113.9")))
        .await;
    assert_eq!(status, StatusCode::OK);
    let rows = h
        .audit(AuditFilter {
            action: Some(String::from("login")),
            ..AuditFilter::default()
        })
        .await;
    let addrs: Vec<Option<&str>> = rows
        .iter()
        .map(|r| r.entry.origin.client_addr.as_deref())
        .collect();
    assert!(addrs.contains(&Some("203.0.113.9")), "{addrs:?}");
    assert!(addrs.contains(&Some("198.51.100.4")), "{addrs:?}");
    assert!(!addrs.contains(&Some("10.0.0.1")), "{addrs:?}");
}

/// `PATCH .../documents/{doc}` sets a document's own fields (title,
/// author, authored date, tags) beside its pin, audited by field name;
/// an empty body is refused; the listing and the page show them.
#[tokio::test(flavor = "multi_thread")]
async fn a_documents_fields_are_set_over_the_api_and_shown() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("meta", &owner).await;
    let token = h.login("owner").await;
    let base = format!("/api/v1/workspaces/{ws}/documents");
    let (status, body) = h
        .post(
            &base,
            &token,
            serde_json::json!({ "text": "Flood is excluded.", "title": "policy" }),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let doc = body["documents"][0]["id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert_eq!(h.wait_ready(&ws, &doc, &token).await["status"], "ready");

    let (status, body) = h
        .call(
            Method::PATCH,
            &format!("{base}/{doc}"),
            Some(&token),
            Some(serde_json::json!({
                "author": "Ada",
                "authored_at": "2026-01-05",
                "tags": ["policy", "flood"],
                "pinned": true
            })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["author"], "Ada");
    assert_eq!(body["authored_at"], "2026-01-05 00:00:00");
    assert_eq!(body["tags"], serde_json::json!(["policy", "flood"]));
    assert_eq!(body["pinned"], true);
    let (status, body) = h
        .call(
            Method::PATCH,
            &format!("{base}/{doc}"),
            Some(&token),
            Some(serde_json::json!({})),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, body) = h
        .call(
            Method::PATCH,
            &format!("{base}/{doc}"),
            Some(&token),
            Some(serde_json::json!({ "authored_at": "soon" })),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    let (_, listed) = h.get(&base, &token).await;
    assert_eq!(listed["documents"][0]["author"], "Ada", "{listed}");
    let cookie = web_session(&h, "owner").await;
    let (status, html, _) = h.page(&format!("/w/{ws}/documents"), Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("Ada") && html.contains("flood"), "{html}");

    let db = h
        .app
        .workspace_db(&ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    let details = with_db(db, |db| audit::list(db, 100))
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(
        details.iter().any(|d| d.action == "context"
            && d.detail.as_ref().is_some_and(
                |v| v["fields"] == serde_json::json!(["author", "authored_at", "tags"])
            )),
        "{details:?}"
    );
}

/// `/readyz` reports each component; `/metrics` answers loopback and
/// admins with Prometheus text that counts the requests served, and
/// refuses everyone else.
#[tokio::test(flavor = "multi_thread")]
async fn readiness_and_metrics_are_served_outside_the_limiter() {
    let h = harness(ServeMode::Login).await;
    let (status, body) = h.call(Method::GET, "/readyz", None, None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["control_db"]["status"], "ok", "{body}");
    assert_eq!(body["data_dir"]["status"], "ok", "{body}");
    assert_eq!(body["vault_key"]["status"], "ok", "{body}");

    // No peer address in a oneshot test, so not loopback: a bearer is needed.
    let (status, _) = h.call(Method::GET, "/metrics", None, None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    drop(h.user("plain", UserKind::Standard).await);
    let plain = h.login("plain").await;
    let (status, _) = h.call(Method::GET, "/metrics", Some(&plain), None).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    drop(h.user("root", UserKind::Admin).await);
    let admin = h.login("root").await;
    let request = Request::builder()
        .method(Method::GET)
        .uri("/metrics")
        .header(header::AUTHORIZATION, format!("Bearer {admin}"))
        .body(Body::empty())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let response = h
        .router
        .clone()
        .oneshot(request)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(response.status(), StatusCode::OK);
    let text = String::from_utf8(
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
            .to_vec(),
    )
    .unwrap_or_default();
    assert!(
        text.contains(r#"quack_http_requests_total{method="GET",route="/readyz",status="200"}"#),
        "{text}"
    );
    assert!(
        text.contains("quack_jobs{kind=\"ingest\",state=\"queued\"} 0"),
        "{text}"
    );
    assert!(text.contains("quack_open_workspaces 0"), "{text}");
    assert!(
        text.contains("quack_writer_waiting{priority=\"interactive\"} 0"),
        "{text}"
    );
}

/// Where [`the_workspace_file_opens_in_another_process`] finds the file.
const PROBE_FILE: &str = "QUACK_TEST_PROBE_FILE";

/// Run in a child process by [`opens_in_another_process`]: `DuckDB` there
/// takes the file only when no connection in the parent holds it, on
/// every OS (a lock on Unix, the share mode on Windows).
#[test]
#[ignore = "run in a child process by a snapshot test"]
fn the_workspace_file_opens_in_another_process() {
    let Some(path) = std::env::var_os(PROBE_FILE) else {
        return;
    };
    let read_only = duckdb::Config::default()
        .access_mode(duckdb::AccessMode::ReadOnly)
        .unwrap_or_else(|e| fail(&e.to_string()));
    duckdb::Connection::open_with_flags(path, read_only).unwrap_or_else(|e| fail(&e.to_string()));
}

/// Whether another process can open `path` now.
fn opens_in_another_process(path: &std::path::Path) -> bool {
    let probe = std::env::current_exe()
        .and_then(|test| {
            std::process::Command::new(test)
                .args([
                    "--exact",
                    "server::tests::the_workspace_file_opens_in_another_process",
                    "--ignored",
                    "--test-threads=1",
                ])
                .env(PROBE_FILE, path)
                .output()
        })
        .unwrap_or_else(|e| fail(&e.to_string()));
    let report = String::from_utf8_lossy(&probe.stdout);
    // A filter that matched nothing would pass vacuously.
    assert!(
        report.contains("1 passed") || report.contains("1 failed"),
        "{report}"
    );
    probe.status.success()
}

/// A snapshot copies the file with every connection to it closed, which
/// Windows requires (#448), and then the workspace serves reads, writes,
/// and audit rows on its reopened connections.
#[tokio::test(flavor = "multi_thread")]
async fn a_snapshot_copies_the_file_closed_and_the_workspace_reopens() {
    let h = harness(ServeMode::Login).await;
    let admin = h.user("root", UserKind::Admin).await;
    let ws = h.workspace("sales", &admin).await;
    let token = h.login("root").await;
    let sql = |s: &str| serde_json::json!({ "sql": s });
    let path = format!("/api/v1/workspaces/{ws}/sql");
    let (status, body) = h
        .post(&path, &token, sql("CREATE TABLE t AS SELECT 7 AS a"))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let file = h.app.config.workspace_db_path(ws.as_str());
    assert!(
        !opens_in_another_process(&file),
        "the open workspace let another process in"
    );

    let probed = file.clone();
    let closed = h
        .app
        .with_workspace_closed(&ws, move || Ok(opens_in_another_process(&probed)))
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(closed, "a connection held the file during the copy");
    assert!(
        !opens_in_another_process(&file),
        "the workspace did not reopen"
    );

    let (status, body) = h.post(&path, &token, sql("INSERT INTO t VALUES (8)")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = h
        .post(&path, &token, sql("SELECT sum(a) AS s FROM t"))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["rows"][0][0], 15, "{body}");
    let (status, rows) = h
        .get(&format!("/api/v1/workspaces/{ws}/audit"), &token)
        .await;
    assert_eq!(status, StatusCode::OK, "{rows}");
    assert!(
        rows["audit"].as_array().is_some_and(|rows| rows.len() >= 3),
        "{rows}"
    );
}

/// A snapshot carries the file, its members, and its settings; a restore
/// brings them back under a new id; a rename and a delete follow, the
/// delete refused while a job of the workspace is active and keeping the
/// audit rows after.
#[tokio::test(flavor = "multi_thread")]
async fn a_workspace_round_trips_through_a_snapshot_and_is_renamed_and_deleted() {
    let h = harness(ServeMode::Login).await;
    let admin = h.user("root", UserKind::Admin).await;
    let viewer = h.user("vera", UserKind::Standard).await;
    let ws = h.workspace("sales", &admin).await;
    h.app
        .control
        .set_member(&ws, &viewer, Role::Viewer, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let token = h.login("root").await;
    let sql = |s: &str| serde_json::json!({ "sql": s });
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &token,
            sql("CREATE TABLE t AS SELECT 7 AS a"),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = h
        .call(
            Method::PATCH,
            &format!("/api/v1/workspaces/{ws}"),
            Some(&token),
            Some(serde_json::json!({ "classification": "secret" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // The snapshot is a tar whose first entry is the manifest.
    let (status, tar, headers) = h
        .send_bytes(
            Request::builder()
                .method(Method::GET)
                .uri(format!("/api/v1/workspaces/{ws}/snapshot"))
                .header(header::AUTHORIZATION, format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap_or_else(|e| fail(&e.to_string())),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        headers
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("application/x-tar")
    );
    let manifest = Manifest::read(tar.as_ref()).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(manifest.name, "sales");
    assert_eq!(manifest.classification, "secret");
    assert_eq!(manifest.members.len(), 2, "{manifest:?}");
    let snapshots = h
        .audit_eventually(
            AuditFilter {
                action: Some(String::from("snapshot")),
                ..AuditFilter::default()
            },
            |rows| !rows.is_empty(),
        )
        .await;
    assert_eq!(snapshots.len(), 1);

    // Restored under a new name: the table, the setting, and the members.
    let (status, body, _) = h
        .send(
            Request::builder()
                .method(Method::POST)
                .uri("/api/v1/workspaces/restore?name=sales-copy")
                .header(header::AUTHORIZATION, format!("Bearer {token}"))
                .header(header::CONTENT_TYPE, "application/x-tar")
                .body(Body::from(tar.clone()))
                .unwrap_or_else(|e| fail(&e.to_string())),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["workspace"]["name"], "sales-copy");
    assert_eq!(body["workspace"]["classification"], "secret");
    assert_eq!(body["members_kept"], 2, "{body}");
    let copy = body["workspace"]["id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert_ne!(copy, ws.to_string());
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{copy}/sql"),
            &token,
            sql("SELECT a FROM t"),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["rows"][0][0], 7);
    let vera = h.login("vera").await;
    let (status, body) = h.get(&format!("/api/v1/workspaces/{copy}"), &vera).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["role"], "viewer", "{body}");

    // The same name again is a conflict; a non-snapshot is a bad request.
    let (status, _, _) = h
        .send(
            Request::builder()
                .method(Method::POST)
                .uri("/api/v1/workspaces/restore?name=sales-copy")
                .header(header::AUTHORIZATION, format!("Bearer {token}"))
                .body(Body::from(tar))
                .unwrap_or_else(|e| fail(&e.to_string())),
        )
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    let (status, _, _) = h
        .send(
            Request::builder()
                .method(Method::POST)
                .uri("/api/v1/workspaces/restore")
                .header(header::AUTHORIZATION, format!("Bearer {token}"))
                .body(Body::from("not a tar"))
                .unwrap_or_else(|e| fail(&e.to_string())),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _, _) = h
        .send(
            Request::builder()
                .method(Method::POST)
                .uri("/api/v1/workspaces/restore")
                .header(header::AUTHORIZATION, format!("Bearer {vera}"))
                .body(Body::empty())
                .unwrap_or_else(|e| fail(&e.to_string())),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    // Rename over PATCH; a taken name is a conflict.
    let (status, body) = h
        .call(
            Method::PATCH,
            &format!("/api/v1/workspaces/{copy}"),
            Some(&token),
            Some(serde_json::json!({ "name": "sales-2025" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["name"], "sales-2025");
    let (status, _) = h
        .call(
            Method::PATCH,
            &format!("/api/v1/workspaces/{copy}"),
            Some(&token),
            Some(serde_json::json!({ "name": "sales" })),
        )
        .await;
    assert_eq!(status, StatusCode::CONFLICT);

    // Delete: viewers may not; the owner may; the rows and the directory go,
    // the audit stays.
    let (status, _) = h
        .call(
            Method::DELETE,
            &format!("/api/v1/workspaces/{copy}"),
            Some(&vera),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let dir = h.app.config.workspace_dir(&copy);
    assert!(dir.join("data.duckdb").exists());
    let (status, body) = h
        .call(
            Method::DELETE,
            &format!("/api/v1/workspaces/{copy}"),
            Some(&token),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT, "{body}");
    assert!(!dir.exists());
    let (status, _) = h.get(&format!("/api/v1/workspaces/{copy}"), &token).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let copy_id = WorkspaceId::from(copy.clone());
    let rows = h
        .audit(AuditFilter {
            workspace_id: Some(copy_id),
            ..AuditFilter::default()
        })
        .await;
    assert!(
        rows.iter().any(|r| r.entry.action == AuditAction::Delete)
            && rows.iter().any(|r| r.entry.action == AuditAction::Restore),
        "{rows:?}"
    );
}

/// Two documents about renewals, `policy.md` tagged `2026` and `notes.md`
/// untagged, one chunk each.
async fn seed_renewal_documents(h: &Harness, ws: &WorkspaceId) {
    let db = h
        .app
        .workspace_db(ws)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    db.run(|db| {
        for (id, name) in [("pol", "policy.md"), ("not", "notes.md")] {
            let id = DocumentId::from(id);
            db.insert_document(
                &NewDocument::new(&id, name, "text/markdown", 1).with_status(DocumentStatus::Ready),
            )?;
            db.chunk_writer(&id, "Renewal terms for the policy year.")
                .and_then(|writer| {
                    writer.insert(&NewChunk {
                        id: &ChunkId::from(format!("{id}-c0")),
                        chunk_index: 0,
                        content: "Renewal terms for the policy year.",
                        heading: None,
                        page: None,
                        kind: SectionKind::Body,
                        locator: None,
                        embedding: None,
                    })
                })?;
            db.set_document_chunk_count(&id, 1)?;
        }
        db.set_document_fields(
            &DocumentId::from("pol"),
            &DocumentFields {
                tags: Some(vec![String::from("2026")]),
                ..DocumentFields::default()
            },
        )
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
}

/// `POST .../search` scopes to documents, a filter, and a mode, carries
/// each leg's rank, explains on request, refuses an unknown document with
/// 422, and audits every search.
#[tokio::test(flavor = "multi_thread")]
async fn rest_search_scopes_explains_and_is_audited() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("search", &owner).await;
    seed_renewal_documents(&h, &ws).await;
    let token = h.login("owner").await;
    let path = format!("/api/v1/workspaces/{ws}/search");
    let documents = |body: &serde_json::Value| -> Vec<String> {
        body["chunks"]
            .as_array()
            .map(|c| {
                c.iter()
                    .filter_map(|c| c["filename"].as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    };

    let (status, body) = h
        .post(&path, &token, serde_json::json!({ "query": "renewal" }))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(documents(&body).len(), 2);
    assert_eq!(body["chunks"][0]["keyword_rank"], 1, "{body}");
    assert!(body["chunks"][0]["bm25"].as_f64().is_some(), "{body}");
    assert!(body.get("explain").is_none());

    let (status, body) = h
        .post(
            &path,
            &token,
            serde_json::json!({ "query": "renewal", "document_ids": ["notes.md"], "explain": true }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(documents(&body), ["notes.md"]);
    assert_eq!(body["explain"]["keyword"].as_array().map(Vec::len), Some(1));
    assert_eq!(body["explain"]["vector"], serde_json::json!([]));
    assert_eq!(body["explain"]["rerank"], "not reranked");

    let (status, body) = h
        .post(
            &path,
            &token,
            serde_json::json!({ "query": "renewal", "filters": { "tags": ["2026"] }, "mode": "keyword" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(documents(&body), ["policy.md"]);

    let (status, body) = h
        .post(
            &path,
            &token,
            serde_json::json!({ "query": "renewal", "mode": "vector" }),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    let (status, body) = h
        .post(
            &path,
            &token,
            serde_json::json!({ "query": "renewal", "document_ids": ["missing.md"] }),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert!(body.to_string().contains("missing.md"), "{body}");

    let rows = h
        .audit(AuditFilter {
            workspace_id: Some(ws.clone()),
            action: Some(String::from("search")),
            ..AuditFilter::default()
        })
        .await;
    assert_eq!(
        rows.iter()
            .filter(|r| r.entry.outcome == Outcome::Allowed)
            .count(),
        3,
        "{rows:?}"
    );
    assert_eq!(
        rows.iter()
            .filter(|r| r.entry.outcome == Outcome::Error)
            .count(),
        2,
        "{rows:?}"
    );

    // The listing takes the same filter as query parameters.
    let listing = format!("/api/v1/workspaces/{ws}/documents");
    let (status, body) = h.get(&format!("{listing}?tags=2026"), &token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["documents"].as_array().map(Vec::len), Some(1));
    assert_eq!(body["documents"][0]["filename"], "policy.md");
    let (status, body) = h
        .get(&format!("{listing}?types=md&sources=upload"), &token)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["documents"].as_array().map(Vec::len), Some(2));
    let (status, _) = h.get(&format!("{listing}?sources=carrier"), &token).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = h.get(&format!("{listing}?types=klingon"), &token).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

/// The web Search page searches in a POST body, shows each hit's ranks
/// with a link to its passage, keeps the picked documents, and shows a
/// refused search on the page; the chat form offers the ready documents.
#[tokio::test(flavor = "multi_thread")]
async fn the_search_page_shows_ranks_and_the_chat_offers_documents() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let ws = h.workspace("searchpage", &owner).await;
    seed_renewal_documents(&h, &ws).await;
    let cookie = web_session(&h, "owner").await;
    let (status, html, _) = h.page(&format!("/w/{ws}/search"), Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK, "{html}");
    assert!(html.contains("name=\"query\""), "{html}");
    assert!(
        html.contains("<option value=\"pol\">policy.md</option>"),
        "{html}"
    );
    assert!(html.contains("aria-current=\"page\">Search</a>"), "{html}");

    let (status, html, _) = h
        .form(
            &format!("/w/{ws}/search"),
            Some(&cookie),
            "query=renewal&documents=pol&mode=keyword",
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{html}");
    assert!(html.contains("1 passages"), "{html}");
    assert!(
        html.contains(&format!("/w/{ws}/documents/pol/chunks/0")),
        "{html}"
    );
    assert!(html.contains("<option value=\"pol\" selected>"), "{html}");
    assert!(html.contains("Keyword leg: 1 candidates"), "{html}");
    assert!(html.contains("not reranked"), "{html}");

    let (status, html, _) = h
        .form(
            &format!("/w/{ws}/search"),
            Some(&cookie),
            "query=renewal&documents=missing",
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{html}");
    assert!(html.contains("role=\"alert\""), "{html}");
    assert!(html.contains("no document matches"), "{html}");

    let (status, html, _) = h.page(&format!("/w/{ws}/chat"), Some(&cookie)).await;
    assert_eq!(status, StatusCode::OK, "{html}");
    assert!(html.contains("name=\"documents\" multiple"), "{html}");
}

/// A table's note, profile warnings, and Fix type over REST and the web
/// page: writers set notes and retype, viewers read, both audited.
#[tokio::test(flavor = "multi_thread")]
async fn table_notes_profiles_and_retypes_over_rest_and_the_web() {
    let h = harness(ServeMode::Login).await;
    let owner = h.user("owner", UserKind::Standard).await;
    let viewer = h.user("viewer", UserKind::Standard).await;
    let ws = h.workspace("data", &owner).await;
    h.app
        .control
        .set_member(&ws, &viewer, Role::Viewer, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let owner_token = h.login("owner").await;
    let viewer_token = h.login("viewer").await;
    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/sql"),
            &owner_token,
            serde_json::json!({ "sql": "CREATE TABLE t AS SELECT * FROM (VALUES ('1', 'a'), ('2', 'b')) v(amount, code)" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let note = |name: &str, text: &str| serde_json::json!({ "name": name, "note": text });
    let (status, _) = h
        .call(
            Method::PUT,
            &format!("/api/v1/workspaces/{ws}/tables/note"),
            Some(&viewer_token),
            Some(note("t", "x")),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, body) = h
        .call(
            Method::PUT,
            &format!("/api/v1/workspaces/{ws}/tables/note"),
            Some(&owner_token),
            Some(note("t", "amounts in cents")),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["note"], "amounts in cents");
    let (status, _) = h
        .call(
            Method::PUT,
            &format!("/api/v1/workspaces/{ws}/tables/note"),
            Some(&owner_token),
            Some(note("ghost", "x")),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

    let (status, body) = h
        .post(
            &format!("/api/v1/workspaces/{ws}/tables/describe"),
            &viewer_token,
            serde_json::json!({ "name": "t" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["note"], "amounts in cents");
    assert_eq!(body["profile"]["row_count"], 2);
    assert_eq!(body["warnings"][0]["column"], "amount");
    assert_eq!(body["warnings"][0]["kind"], "numeric_text");
    assert_eq!(body["warnings"][0]["fix"], "DOUBLE");

    let retype =
        |column: &str, to: &str| serde_json::json!({ "name": "t", "column": column, "type": to });
    let path = format!("/api/v1/workspaces/{ws}/tables/retype");
    let (status, _) = h
        .post(&path, &viewer_token, retype("amount", "DOUBLE"))
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = h.post(&path, &owner_token, retype("amount", "MONEY")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (status, _) = h.post(&path, &owner_token, retype("code", "DOUBLE")).await;
    assert_eq!(
        status,
        StatusCode::UNPROCESSABLE_ENTITY,
        "a value that does not convert"
    );
    let (status, _) = h
        .post(
            &path,
            &owner_token,
            serde_json::json!({ "name": "_quack_meta", "column": "key", "type": "DOUBLE" }),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, body) = h
        .post(&path, &owner_token, retype("amount", "DOUBLE"))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["columns"][0]["type"], "DOUBLE");
    assert_eq!(body["warnings"], serde_json::json!([]));

    for action in ["table_note", "retype"] {
        let rows = h
            .audit(AuditFilter {
                workspace_id: Some(ws.clone()),
                action: Some(String::from(action)),
                ..AuditFilter::default()
            })
            .await;
        assert!(
            rows.iter().any(|r| r.entry.outcome == Outcome::Allowed),
            "{action}"
        );
    }

    let (_, _, headers) = h.form("/login", None, "username=owner&password=pw").await;
    let cookie = headers
        .get(header::SET_COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|c| c.split(';').next())
        .and_then(|c| c.strip_prefix("quack_session="))
        .unwrap_or_default()
        .to_owned();
    let (status, _, headers) = h
        .form(
            &format!("/w/{ws}/tables/note"),
            Some(&cookie),
            "name=t&note=one+row+per+order",
        )
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    let (_, html) = h.land(&headers, Some(&cookie)).await;
    assert!(
        html.contains("note saved") && html.contains("one row per order"),
        "{html}"
    );
    assert!(html.contains("Present") && html.contains("100%"), "{html}");
    let (status, _, headers) = h
        .form(
            &format!("/w/{ws}/tables/retype"),
            Some(&cookie),
            "name=t&column=code&type=DATE",
        )
        .await;
    assert_eq!(status, StatusCode::SEE_OTHER);
    let (_, html) = h.land(&headers, Some(&cookie)).await;
    assert!(
        html.contains("role=\"alert\"") && html.contains("does not convert"),
        "{html}"
    );
}

/// `GET .../graph/export` streams the whole graph in the format asked for,
/// with the provenance of a person's assertions, and audits `export` with
/// the counts once the stream ends; `.../ontology/schema` is the published
/// schema.
#[tokio::test(flavor = "multi_thread")]
async fn the_graph_exports_as_a_download_audited_with_its_counts() {
    let h = harness(ServeMode::Local).await;
    let (_, body) = h
        .call(
            Method::POST,
            "/api/v1/workspaces",
            None,
            Some(serde_json::json!({ "name": "graph-export" })),
        )
        .await;
    let ws = WorkspaceId::from(body["id"].as_str().unwrap_or_default());
    let base = format!("/api/v1/workspaces/{ws}");
    let (status, _) = h
        .post(&format!("{base}/ontology/init"), "", serde_json::json!({}))
        .await;
    assert_eq!(status, StatusCode::OK);
    let mut ids = Vec::new();
    for (label, class) in [("Ada <Lovelace>", "person"), ("Acme, Inc.", "organization")] {
        let (status, body) = h
            .post(
                &format!("{base}/graph/nodes"),
                "",
                serde_json::json!({ "label": label, "class": class, "note": "from the filing" }),
            )
            .await;
        assert_eq!(status, StatusCode::CREATED, "{body}");
        ids.push(body["id"].as_str().unwrap_or_default().to_owned());
    }
    let (status, body) = h
        .post(
            &format!("{base}/graph/edges"),
            "",
            serde_json::json!({ "source": ids.first(), "target": ids.get(1), "relation": "works_at" }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");

    let request = Request::builder()
        .uri(format!("{base}/graph/export?format=graphml"))
        .body(Body::empty())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let response = h
        .router
        .clone()
        .oneshot(request)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(response.status(), StatusCode::OK);
    let header_of = |name| {
        response
            .headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    };
    assert_eq!(
        header_of(header::CONTENT_TYPE).as_deref(),
        Some("application/graphml+xml")
    );
    assert_eq!(
        header_of(header::CONTENT_DISPOSITION).as_deref(),
        Some("attachment; filename=\"graph-export.graph.graphml\"")
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("Ada &lt;Lovelace&gt;"), "{text}");
    assert!(text.contains("from the filing"), "{text}");

    // Audited when the stream ends, with what went; the detail row lands
    // in the workspace after the access row.
    let mut detail = serde_json::Value::Null;
    for _ in 0..250 {
        let row = h
            .audit(AuditFilter {
                workspace_id: Some(ws.clone()),
                action: Some(String::from("export")),
                ..AuditFilter::default()
            })
            .await
            .into_iter()
            .find(|r| r.entry.outcome == Outcome::Allowed);
        if let Some(row) = row {
            let details = h
                .app
                .read(&ws, |db| audit::list(db, 50))
                .await
                .unwrap_or_else(|e| fail(&e.message));
            if let Some(found) = details
                .iter()
                .find(|d| d.id == row.entry.id)
                .and_then(|d| d.detail.clone())
            {
                detail = found;
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert_eq!(
        detail,
        serde_json::json!({ "format": "graphml", "nodes": 2, "edges": 1, "provenance": 3 })
    );

    // A format quack does not write is refused before anything streams.
    let (status, _) = h
        .call(
            Method::GET,
            &format!("{base}/graph/export?format=gexf"),
            None,
            None,
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, body) = h
        .call(Method::GET, &format!("{base}/ontology/schema"), None, None)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        Some(body),
        serde_json::to_value(Ontology::json_schema()).ok()
    );
}
