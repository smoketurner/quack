//! Sign-in through a mock issuer, driven through the router: the redirect
//! out, the callback, the user it creates, renewal, and the refusals.

#![expect(
    clippy::indexing_slicing,
    reason = "serde_json::Value indexing yields Null for a missing key, never a panic"
)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::{Form, State};
use axum::http::{Request, StatusCode, header};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use quack_core::config::{Config, OidcConfig};
use quack_core::ids::UserId;
use quack_core::llm::oauth::KeySource;
use quack_core::oidc::OidcSubject;
use quack_core::storage::control::{AuditFilter, ControlPlane, Outcome};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::Oidc;
use crate::server::state::{App, AppState, ServeMode};

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

/// What the mock issuer answers with, and what it saw.
#[derive(Default)]
struct IssuerState {
    base: String,
    id_claims: Value,
    /// `expires_in` of the tokens it issues.
    lifetime: i64,
    refresh_error: Option<&'static str>,
    refreshes: usize,
}

type Issuer = Arc<Mutex<IssuerState>>;

async fn discovery(State(issuer): State<Issuer>) -> Json<Value> {
    let base = issuer.lock().map(|s| s.base.clone()).unwrap_or_default();
    Json(json!({
        "issuer": base,
        "authorization_endpoint": format!("{base}/authorize"),
        "token_endpoint": format!("{base}/token"),
    }))
}

async fn token(
    State(issuer): State<Issuer>,
    Form(form): Form<HashMap<String, String>>,
) -> (StatusCode, Json<Value>) {
    let Ok(mut state) = issuer.lock() else {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({})));
    };
    let lifetime = state.lifetime;
    match form.get("grant_type").map(String::as_str) {
        Some("authorization_code") => {
            let id_token = format!(
                "{}.{}.sig",
                URL_SAFE_NO_PAD.encode(br#"{"alg":"RS256"}"#),
                URL_SAFE_NO_PAD.encode(state.id_claims.to_string())
            );
            (
                StatusCode::OK,
                Json(json!({
                    "access_token": "user-access", "token_type": "Bearer", "expires_in": lifetime,
                    "refresh_token": "user-refresh", "id_token": id_token,
                })),
            )
        }
        Some("refresh_token") => {
            state.refreshes = state.refreshes.saturating_add(1);
            match state.refresh_error {
                Some(error) => (StatusCode::BAD_REQUEST, Json(json!({ "error": error }))),
                None => (
                    StatusCode::OK,
                    Json(
                        json!({ "access_token": "renewed", "token_type": "Bearer", "expires_in": 3600 }),
                    ),
                ),
            }
        }
        _ => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "unsupported_grant_type" })),
        ),
    }
}

async fn start_issuer() -> Issuer {
    let Ok(listener) = tokio::net::TcpListener::bind("127.0.0.1:0").await else {
        fail("loopback bind failed");
    };
    let Ok(addr) = listener.local_addr() else {
        fail("no local address");
    };
    let issuer = Arc::new(Mutex::new(IssuerState {
        base: format!("http://{addr}"),
        lifetime: 3600,
        ..IssuerState::default()
    }));
    let routes = Router::new()
        .route("/.well-known/openid-configuration", get(discovery))
        .route("/token", post(token))
        .with_state(Arc::clone(&issuer));
    tokio::spawn(async move { axum::serve(listener, routes).await });
    issuer
}

struct Harness {
    dir: tempfile::TempDir,
    app: App,
    router: Router,
    issuer: Issuer,
}

/// A response's status, `Location`, `Set-Cookie` values, and body text.
struct Reply {
    status: StatusCode,
    location: String,
    cookies: Vec<String>,
    body: String,
}

impl Reply {
    /// The value a `Set-Cookie` gives `name`.
    fn cookie(&self, name: &str) -> Option<String> {
        self.cookies.iter().find_map(|c| {
            c.strip_prefix(&format!("{name}="))
                .and_then(|rest| rest.split(';').next())
                .map(str::to_owned)
        })
    }
}

impl Harness {
    async fn new() -> Self {
        let issuer = start_issuer().await;
        let base = issuer.lock().map(|s| s.base.clone()).unwrap_or_default();
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = Config::default();
        config.general.data_dir = dir.path().to_path_buf();
        let oidc_config = OidcConfig {
            issuer_url: base,
            client_id: String::from("quack"),
            client_secret_env: None,
            scopes: OidcConfig::default_scopes(),
            redirect_uri: format!("https://quack.example.com{}", OidcConfig::CALLBACK_PATH),
        };
        let oidc = Oidc::new(&oidc_config, config.tokens_dir(), KeySource::File)
            .unwrap_or_else(|e| fail(&e.to_string()));
        config.server.oidc = Some(oidc_config);
        let control = ControlPlane::open(&config)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let app = Arc::new(AppState::new(config, control, ServeMode::Login, Some(oidc)));
        let router = crate::server::router(Arc::clone(&app));
        Self {
            dir,
            app,
            router,
            issuer,
        }
    }

    fn issuer(&self, change: impl FnOnce(&mut IssuerState)) {
        if let Ok(mut state) = self.issuer.lock() {
            change(&mut state);
        }
    }

    async fn get(&self, uri: &str, cookie: Option<&str>) -> Reply {
        let mut request = Request::get(uri);
        if let Some(cookie) = cookie {
            request = request.header(header::COOKIE, cookie);
        }
        let request = request
            .body(Body::empty())
            .unwrap_or_else(|e| fail(&e.to_string()));
        self.send(request).await
    }

    async fn send(&self, request: Request<Body>) -> Reply {
        let response = self
            .router
            .clone()
            .oneshot(request)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let status = response.status();
        let header_text = |name| {
            response
                .headers()
                .get_all(name)
                .iter()
                .filter_map(|v| v.to_str().ok().map(str::to_owned))
                .collect::<Vec<_>>()
        };
        let location = header_text(header::LOCATION).join("");
        let cookies = header_text(header::SET_COOKIE);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        Reply {
            status,
            location,
            cookies,
            body: String::from_utf8_lossy(&bytes).into_owned(),
        }
    }

    /// Start a sign-in: the `state` and `nonce` the issuer was sent, and the
    /// state cookie the browser got.
    async fn start(&self) -> (String, String, String) {
        let reply = self.get("/login/oidc", None).await;
        assert_eq!(reply.status, StatusCode::SEE_OTHER, "{}", reply.body);
        let query: HashMap<String, String> = reply
            .location
            .parse::<axum::http::Uri>()
            .ok()
            .and_then(|uri| axum::extract::Query::try_from_uri(&uri).ok())
            .map(|axum::extract::Query(q)| q)
            .unwrap_or_default();
        let cookie = reply
            .cookie(super::STATE_COOKIE)
            .unwrap_or_else(|| fail("no state cookie"));
        assert!(
            reply.cookies.iter().any(|c| c.contains("HttpOnly")
                && c.contains(&format!("Path={}", OidcConfig::CALLBACK_PATH))),
            "{:?}",
            reply.cookies
        );
        let state = query.get("state").cloned().unwrap_or_default();
        let nonce = query.get("nonce").cloned().unwrap_or_default();
        assert_eq!(state, cookie);
        (state, nonce, cookie)
    }

    /// Sign `subject` in as `username`; the session cookie on success.
    async fn sign_in(&self, subject: &str, username: &str) -> String {
        let (state, nonce, cookie) = self.start().await;
        let base = self
            .issuer
            .lock()
            .map(|s| s.base.clone())
            .unwrap_or_default();
        self.issuer(|s| {
            s.id_claims = json!({
                "iss": base, "sub": subject, "aud": "quack",
                "exp": jiff::Timestamp::now().as_second().saturating_add(3600),
                "nonce": nonce, "preferred_username": username,
            });
        });
        let reply = self
            .get(
                &format!("{}?code=c&state={state}", OidcConfig::CALLBACK_PATH),
                Some(&format!("{}={cookie}", super::STATE_COOKIE)),
            )
            .await;
        assert_eq!(reply.status, StatusCode::SEE_OTHER, "{}", reply.body);
        assert_eq!(reply.location, "/workspaces");
        let session = reply
            .cookie(crate::server::auth::SESSION_COOKIE)
            .unwrap_or_else(|| fail("no session cookie"));
        format!("{}={session}", crate::server::auth::SESSION_COOKIE)
    }

    async fn me(&self, session: &str) -> (StatusCode, Value) {
        let reply = self.get("/api/v1/auth/me", Some(session)).await;
        (
            reply.status,
            serde_json::from_str(&reply.body).unwrap_or(Value::Null),
        )
    }

    async fn audit(&self, action: &str) -> Vec<(Outcome, Option<UserId>)> {
        let filter = AuditFilter {
            action: Some(action.to_owned()),
            limit: 100,
            ..AuditFilter::default()
        };
        self.app
            .control
            .query_audit(&filter)
            .await
            .map(|page| {
                page.rows
                    .into_iter()
                    .map(|r| (r.outcome, r.user_id))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn token_file(&self, user: &UserId) -> std::path::PathBuf {
        self.dir
            .path()
            .join("tokens")
            .join("users")
            .join(format!("{user}.json"))
    }
}

#[tokio::test]
async fn a_first_sign_in_creates_a_user_with_no_access_and_a_session() {
    let h = Harness::new().await;
    let login = h.get("/login", None).await;
    assert!(
        login.body.contains("Sign in with 127.0.0.1"),
        "{}",
        login.body
    );

    let session = h.sign_in("sub-ada", "ada").await;
    let (status, me) = h.me(&session).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(me["username"], "ada");
    assert_eq!(me["is_admin"], false);
    assert_eq!(me["via"], "session");

    let user = h
        .app
        .control
        .find_user_by_oidc_subject(&OidcSubject::from("sub-ada"))
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| fail("no user for the subject"));
    assert!(
        h.app
            .control
            .workspaces_for_user(&user.id)
            .await
            .is_ok_and(|w| w.is_empty())
    );
    assert!(h.token_file(&user.id).exists());
    let logins = h.audit("login").await;
    assert!(
        logins.contains(&(Outcome::Allowed, Some(user.id.clone()))),
        "{logins:?}"
    );

    // A second sign-in is the same user, not a new one.
    let again = h.sign_in("sub-ada", "ada-renamed").await;
    let (_, me) = h.me(&again).await;
    assert_eq!(me["id"], user.id.to_string());
    assert_eq!(me["username"], "ada");
}

#[tokio::test]
async fn a_callback_this_browser_did_not_start_is_refused_and_audited() {
    let h = Harness::new().await;
    let (state, _, cookie) = h.start().await;
    let callback = format!("{}?code=c&state={state}", OidcConfig::CALLBACK_PATH);

    let refused = |reply: &Reply, text: &str| {
        assert_eq!(reply.status, StatusCode::SEE_OTHER, "{}", reply.body);
        assert!(
            reply.location.starts_with("/login?error="),
            "{}",
            reply.location
        );
        assert!(
            reply
                .location
                .replace("%20", " ")
                .replace('+', " ")
                .contains(text),
            "{} lacks {text}",
            reply.location
        );
        assert!(reply.cookie(crate::server::auth::SESSION_COOKIE).is_none());
    };
    refused(&h.get(&callback, None).await, "could not be matched");
    refused(
        &h.get(&callback, Some(&format!("{}=other", super::STATE_COOKIE)))
            .await,
        "could not be matched",
    );
    let error = format!(
        "{}?error=access_denied&error_description=no&state={state}",
        OidcConfig::CALLBACK_PATH
    );
    let with_cookie = format!("{}={cookie}", super::STATE_COOKIE);
    refused(&h.get(&error, Some(&with_cookie)).await, "access_denied");
    // The state was spent by that callback; replaying it finds nothing.
    refused(&h.get(&callback, Some(&with_cookie)).await, "already used");

    let denied = h.audit("login").await;
    assert_eq!(denied.len(), 4, "{denied:?}");
    assert!(
        denied
            .iter()
            .all(|(o, u)| *o == Outcome::Denied && u.is_none())
    );
}

#[tokio::test]
async fn an_expiring_sign_in_is_renewed_and_a_revoked_one_ends_every_session() {
    let h = Harness::new().await;
    // A token that is already inside the renewal margin when issued.
    h.issuer(|s| s.lifetime = 30);
    let first = h.sign_in("sub-grace", "grace").await;
    let second = h.sign_in("sub-grace", "grace").await;
    let refreshes = || h.issuer.lock().map(|s| s.refreshes).unwrap_or_default();

    assert_eq!(h.me(&first).await.0, StatusCode::OK);
    assert_eq!(refreshes(), 1);
    assert_eq!(h.me(&first).await.0, StatusCode::OK);
    // The other session finds the token the first one renewed.
    assert_eq!(h.me(&second).await.0, StatusCode::OK);
    assert_eq!(refreshes(), 1);

    h.issuer(|s| s.lifetime = 30);
    let third = h.sign_in("sub-grace", "grace").await;
    h.issuer(|s| s.refresh_error = Some("invalid_grant"));
    let (status, body) = h.me(&third).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(
        body["error"]
            .as_str()
            .is_some_and(|e| e.contains("identity provider ended"))
    );
    for session in [&first, &second, &third] {
        assert_eq!(h.me(session).await.0, StatusCode::UNAUTHORIZED);
    }
    let user = h
        .app
        .control
        .find_user_by_oidc_subject(&OidcSubject::from("sub-grace"))
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| fail("no user"));
    assert!(!h.token_file(&user.id).exists());
    let ended = h.audit("session").await;
    assert!(
        ended.contains(&(Outcome::Denied, Some(user.id))),
        "{ended:?}"
    );
}

#[tokio::test]
async fn an_unreachable_issuer_keeps_the_session_and_asks_again_later() {
    let h = Harness::new().await;
    h.issuer(|s| s.lifetime = 30);
    let session = h.sign_in("sub-lin", "lin").await;
    h.issuer(|s| s.refresh_error = Some("temporarily_unavailable"));
    assert_eq!(h.me(&session).await.0, StatusCode::OK);
    assert_eq!(h.me(&session).await.0, StatusCode::OK);
    // One attempt, then the retry delay holds off the next.
    assert_eq!(h.issuer.lock().map(|s| s.refreshes).unwrap_or_default(), 1);
}

#[tokio::test]
async fn logging_out_of_the_last_session_forgets_the_token() {
    let h = Harness::new().await;
    let first = h.sign_in("sub-kay", "kay").await;
    let second = h.sign_in("sub-kay", "kay").await;
    let user = h
        .app
        .control
        .find_user_by_oidc_subject(&OidcSubject::from("sub-kay"))
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| fail("no user"));
    let log_out = |session: String| {
        Request::post("/api/v1/auth/logout")
            .header(header::COOKIE, session)
            .body(Body::empty())
            .unwrap_or_else(|e| fail(&e.to_string()))
    };
    assert_eq!(h.send(log_out(first)).await.status, StatusCode::NO_CONTENT);
    assert!(h.token_file(&user.id).exists());
    assert_eq!(h.send(log_out(second)).await.status, StatusCode::NO_CONTENT);
    assert!(!h.token_file(&user.id).exists());
}

#[tokio::test]
async fn without_oidc_there_is_no_button_and_no_route() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = Config::default();
    config.general.data_dir = dir.path().to_path_buf();
    let control = ControlPlane::open(&config)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let app = Arc::new(AppState::new(config, control, ServeMode::Login, None));
    let router = crate::server::router(app);
    for (uri, status) in [
        ("/login", StatusCode::OK),
        ("/login/oidc", StatusCode::NOT_FOUND),
    ] {
        let response = router
            .clone()
            .oneshot(
                Request::get(uri)
                    .body(Body::empty())
                    .unwrap_or_else(|e| fail(&e.to_string())),
            )
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(response.status(), status, "{uri}");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap_or_default();
        assert!(!String::from_utf8_lossy(&bytes).contains("Sign in with"));
    }
}
