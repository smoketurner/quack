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
use jiff::{SignedDuration, Timestamp};
use quack_core::config::{ClientAuth, Config, OidcConfig};
use quack_core::ids::UserId;
use quack_core::llm::oauth::client_key::{ClientKeyName, ClientKeys, PublicJwk};
use quack_core::llm::oauth::{CachedToken, KeySource};
use quack_core::oidc::OidcSubject;
use quack_core::storage::control::{AuditFilter, ControlPlane, Outcome, SealedOwner};
use quack_core::vault::Vault;
use secrecy::SecretString;
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
    /// The JWK set `/jwks` serves.
    jwks: Vec<Value>,
    refresh_error: Option<&'static str>,
    refreshes: usize,
    /// Whether discovery lists a pushed authorization request endpoint.
    par: bool,
    /// Each pushed authorization request, as sent.
    pushed: Vec<HashMap<String, String>>,
    /// Each token request, and its `Authorization` header.
    token_requests: Vec<(HashMap<String, String>, Option<String>)>,
}

type Issuer = Arc<Mutex<IssuerState>>;

async fn discovery(State(issuer): State<Issuer>) -> Json<Value> {
    let (base, par) = issuer
        .lock()
        .map(|s| (s.base.clone(), s.par))
        .unwrap_or_default();
    let mut metadata = json!({
        "issuer": base,
        "authorization_endpoint": format!("{base}/authorize"),
        "token_endpoint": format!("{base}/token"),
        "jwks_uri": format!("{base}/jwks"),
    });
    if par {
        metadata["pushed_authorization_request_endpoint"] = json!(format!("{base}/par"));
    }
    Json(metadata)
}

async fn par(
    State(issuer): State<Issuer>,
    Form(form): Form<HashMap<String, String>>,
) -> (StatusCode, Json<Value>) {
    if let Ok(mut state) = issuer.lock() {
        state.pushed.push(form);
    }
    (
        StatusCode::CREATED,
        Json(json!({ "request_uri": "urn:ietf:params:oauth:request_uri:r1", "expires_in": 60 })),
    )
}

async fn jwks(State(issuer): State<Issuer>) -> Json<Value> {
    let keys = issuer.lock().map(|s| s.jwks.clone()).unwrap_or_default();
    Json(json!({ "keys": keys }))
}

async fn token(
    State(issuer): State<Issuer>,
    headers: axum::http::HeaderMap,
    Form(form): Form<HashMap<String, String>>,
) -> (StatusCode, Json<Value>) {
    let Ok(mut state) = issuer.lock() else {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({})));
    };
    let authorization = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    state.token_requests.push((form.clone(), authorization));
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
        Some("urn:ietf:params:oauth:grant-type:token-exchange") => {
            let subject = form.get("subject_token").cloned().unwrap_or_default();
            (
                StatusCode::OK,
                Json(json!({
                    "access_token": format!("obo-{subject}"), "token_type": "Bearer",
                    "expires_in": 3600,
                    "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                })),
            )
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
        .route("/par", post(par))
        .route("/jwks", get(jwks))
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
    /// `WWW-Authenticate`.
    challenge: String,
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
        Self::with_audience(None).await
    }

    /// A harness whose `[server.oidc].audience` is `audience`: set, the API
    /// and MCP accept the issuer's access tokens.
    async fn with_audience(audience: Option<&str>) -> Self {
        Self::build(audience, "https://quack.example.com").await
    }

    /// A harness whose public URL (the origin of `redirect_uri`) is
    /// `origin`.
    async fn build(audience: Option<&str>, origin: &str) -> Self {
        Self::build_with(audience, origin, |_| {}).await
    }

    /// A harness whose `[server.oidc]` `change` adjusts.
    async fn build_with(
        audience: Option<&str>,
        origin: &str,
        change: impl FnOnce(&mut OidcConfig),
    ) -> Self {
        let issuer = start_issuer().await;
        let base = issuer.lock().map(|s| s.base.clone()).unwrap_or_default();
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = Config::default();
        config.general.data_dir = dir.path().to_path_buf();
        let oidc_config = OidcConfig {
            issuer_url: base,
            client_id: Some(String::from("quack")),
            client_secret_env: None,
            client_auth: ClientAuth::default(),
            scopes: OidcConfig::default_scopes(),
            redirect_uri: format!("{origin}{}", OidcConfig::CALLBACK_PATH),
            audience: audience.map(str::to_owned),
            subject_claim: String::from(OidcConfig::DEFAULT_SUBJECT_CLAIM),
        };
        let mut oidc_config = oidc_config;
        change(&mut oidc_config);
        let control = ControlPlane::open(&config)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let oidc = Oidc::new(
            &oidc_config,
            Vault::new(dir.path(), KeySource::File),
            control.clone(),
            ClientKeys::with_control(&config, KeySource::File, control.clone()),
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
        config.server.oidc = Some(oidc_config);
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

    /// A GET that arrives from `peer`, as the real server records it.
    async fn get_from(&self, uri: &str, cookie: Option<&str>, peer: &str) -> Reply {
        let mut request = Request::get(uri);
        if let Some(cookie) = cookie {
            request = request.header(header::COOKIE, cookie);
        }
        let mut request = request
            .body(Body::empty())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let addr: std::net::SocketAddr = peer.parse().unwrap_or_else(|e| fail(&format!("{e}")));
        request
            .extensions_mut()
            .insert(axum::extract::ConnectInfo(addr));
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
        let challenge = header_text(header::WWW_AUTHENTICATE).join("");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        Reply {
            status,
            location,
            cookies,
            challenge,
            body: String::from_utf8_lossy(&bytes).into_owned(),
        }
    }

    /// Start a sign-in: the `state` and `nonce` the issuer was sent, and the
    /// state cookie the browser got.
    async fn start(&self) -> (String, String, String) {
        let reply = self.get(OidcConfig::START_PATH, None).await;
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

    /// Whether the user's sealed token is in `control.db`.
    async fn has_token(&self, user: &UserId) -> bool {
        self.app
            .control
            .sealed(SealedOwner::User(user))
            .await
            .is_ok_and(|t| t.is_some())
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
    assert!(
        login
            .body
            .contains(&format!("href=\"{}\"", OidcConfig::START_PATH)),
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
    assert!(h.has_token(&user.id).await);
    // One vault key on disk (no keychain in tests), and no file per user.
    assert!(h.dir.path().join("vault.key").exists());
    assert!(!h.dir.path().join("tokens").exists());
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
    assert!(!h.has_token(&user.id).await);
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
    assert!(h.has_token(&user.id).await);
    assert_eq!(h.send(log_out(second)).await.status, StatusCode::NO_CONTENT);
    assert!(!h.has_token(&user.id).await);
    let logouts = h.audit("logout").await;
    assert_eq!(logouts.len(), 2, "{logouts:?}");
    assert!(
        logouts
            .iter()
            .all(|(outcome, who)| *outcome == Outcome::Allowed && who.as_ref() == Some(&user.id))
    );
}

/// A sign-in that arrives while the last logout is forgetting the user's
/// token keeps the token it stores: the logout's check and delete, and the
/// sign-in's store and session, run under one per-user lock, so the sign-in
/// waits for the delete instead of landing between the check and it (#241).
#[tokio::test(flavor = "multi_thread")]
async fn a_sign_in_racing_the_last_logout_keeps_its_token() {
    let h = Harness::new().await;
    let cookie = h.sign_in("sub-ray", "ray").await;
    let user = h
        .app
        .control
        .find_user_by_oidc_subject(&OidcSubject::from("sub-ray"))
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| fail("no user"))
        .id;
    // The logout has closed the user's last session.
    let session = cookie
        .strip_prefix(&format!("{}=", crate::server::auth::SESSION_COOKIE))
        .unwrap_or_else(|| fail("not a session cookie"));
    h.app.sessions.close(session);
    assert!(h.has_token(&user).await);

    let oidc = h.app.oidc.as_ref().unwrap_or_else(|| fail("no oidc"));
    let mut signing_in = None;
    let forgotten = oidc
        .subjects
        .forget_unless(&user, || {
            // A new sign-in starts right after the logout found no session:
            // under the old order it stored its token and opened its session
            // here, and the delete below removed that token.
            let app = Arc::clone(&h.app);
            let who = user.clone();
            signing_in = Some(tokio::spawn(async move {
                let token = CachedToken {
                    access_token: SecretString::from(String::from("fresh")),
                    expires_at: Timestamp::now()
                        .checked_add(SignedDuration::from_hours(1))
                        .unwrap_or_else(|e| fail(&e.to_string())),
                    refresh_token: None,
                };
                let oidc = app.oidc.as_ref().unwrap_or_else(|| fail("no oidc"));
                oidc.keep_and_open(&app.sessions, &who, &token).await
            }));
            // Give the sign-in every chance to run ahead of the delete.
            std::thread::sleep(std::time::Duration::from_millis(100));
            h.app.sessions.has_sessions(&user)
        })
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(forgotten, "no session was open when the logout checked");
    let opened = signing_in
        .unwrap_or_else(|| fail("the sign-in never started"))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(opened.is_ok(), "the sign-in failed");
    assert!(h.app.sessions.has_sessions(&user));
    assert!(
        h.has_token(&user).await,
        "the logout deleted the token of the sign-in that followed it"
    );

    // With that session open, logging out another one leaves the token.
    let dropped = oidc
        .forget_unless_signed_in(&h.app.sessions, &user)
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(!dropped);
    assert!(h.has_token(&user).await);
}

/// An issuer's access token is not a session: logging out with it closes
/// nothing, records no `logout`, and keeps the user's stored sign-in token
/// even while they have no session open (#223).
#[tokio::test]
async fn a_bearer_logout_keeps_the_stored_token_and_audits_nothing() {
    let (h, key, issuer) = Harness::resource().await;
    let cookie = h.sign_in("sub-kay", "kay").await;
    let user = h
        .app
        .control
        .find_user_by_oidc_subject(&OidcSubject::from("sub-kay"))
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| fail("no user"));
    // The browser session ends without a logout (a restart, an expiry), so
    // the stored token outlives every session.
    let session = cookie
        .strip_prefix(&format!("{}=", crate::server::auth::SESSION_COOKIE))
        .unwrap_or_else(|| fail("not a session cookie"));
    h.app.sessions.close(session);
    assert!(h.has_token(&user.id).await);

    let token = key.token(&issuer, "sub-kay", AUDIENCE);
    let request = Request::post("/api/v1/auth/logout")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let reply = h.send(request).await;
    assert_eq!(reply.status, StatusCode::NO_CONTENT, "{}", reply.body);
    assert!(h.has_token(&user.id).await, "a bearer ends no sign-in");
    assert!(h.audit("logout").await.is_empty());
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
        (OidcConfig::START_PATH, StatusCode::NOT_FOUND),
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

// --- access tokens as bearers (RFC 9728) -----------------------------------

/// The audience the resource tests configure.
const AUDIENCE: &str = "api://quack";

/// A P-256 key an issuer signs with, and its published JWK.
struct IssuerKey {
    encoding: jsonwebtoken::EncodingKey,
    jwk: Value,
}

impl IssuerKey {
    fn new() -> Self {
        use aws_lc_rs::signature::{ECDSA_P256_SHA256_FIXED_SIGNING, EcdsaKeyPair, KeyPair};
        let pkcs8 = EcdsaKeyPair::generate_pkcs8(
            &ECDSA_P256_SHA256_FIXED_SIGNING,
            &aws_lc_rs::rand::SystemRandom::new(),
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
        let pair = EcdsaKeyPair::from_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, pkcs8.as_ref())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let point = pair.public_key().as_ref();
        let (Some(x), Some(y)) = (point.get(1..33), point.get(33..65)) else {
            fail("not an uncompressed P-256 point");
        };
        Self {
            encoding: jsonwebtoken::EncodingKey::from_ec_der(pkcs8.as_ref()),
            jwk: json!({
                "kty": "EC", "crv": "P-256", "alg": "ES256", "kid": "k1",
                "x": URL_SAFE_NO_PAD.encode(x), "y": URL_SAFE_NO_PAD.encode(y),
            }),
        }
    }

    /// An access token for `subject` from `issuer`, with `aud` as given.
    fn token(&self, issuer: &str, subject: &str, aud: &str) -> String {
        let mut header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::ES256);
        header.kid = Some(String::from("k1"));
        let claims = json!({
            "iss": issuer, "aud": aud, "sub": subject, "scp": "quack.use",
            "exp": jiff::Timestamp::now().as_second().saturating_add(3600),
            "preferred_username": subject,
        });
        jsonwebtoken::encode(&header, &claims, &self.encoding)
            .unwrap_or_else(|e| fail(&e.to_string()))
    }
}

impl Harness {
    /// A harness that accepts access tokens, and the key its issuer signs
    /// them with.
    async fn resource() -> (Self, IssuerKey, String) {
        let h = Self::with_audience(Some(AUDIENCE)).await;
        let key = IssuerKey::new();
        let base = h.issuer.lock().map(|s| s.base.clone()).unwrap_or_default();
        h.issuer(|s| s.jwks = vec![key.jwk.clone()]);
        (h, key, base)
    }

    async fn with_bearer(&self, uri: &str, token: &str) -> Reply {
        let request = Request::get(uri)
            .header(header::AUTHORIZATION, format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap_or_else(|e| fail(&e.to_string()));
        self.send(request).await
    }
}

#[tokio::test]
async fn an_access_token_from_the_issuer_is_the_user_it_names() {
    let (h, key, issuer) = Harness::resource().await;
    let token = key.token(&issuer, "sub-mcp", AUDIENCE);

    let reply = h.with_bearer("/api/v1/auth/me", &token).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    let me: Value = serde_json::from_str(&reply.body).unwrap_or(Value::Null);
    assert_eq!(me["username"], "sub-mcp");
    assert_eq!(me["via"], "identity-provider");
    assert_eq!(me["is_admin"], false);
    let user = h
        .app
        .control
        .find_user_by_oidc_subject(&OidcSubject::from("sub-mcp"))
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
    // The same token again is the same user.
    let again = h.with_bearer("/api/v1/auth/me", &token).await;
    let again: Value = serde_json::from_str(&again.body).unwrap_or(Value::Null);
    assert_eq!(again["id"], user.id.to_string());
}

#[tokio::test]
async fn a_401_says_where_to_get_a_token_and_a_refused_one_is_invalid_token() {
    let (h, key, issuer) = Harness::resource().await;
    let metadata = "https://quack.example.com/.well-known/oauth-protected-resource";

    let anonymous = h.get("/api/v1/auth/me", None).await;
    assert_eq!(anonymous.status, StatusCode::UNAUTHORIZED);
    assert_eq!(
        anonymous.challenge,
        format!("Bearer resource_metadata=\"{metadata}\"")
    );

    let mcp = h.get("/mcp/v1/ws1", None).await;
    assert_eq!(mcp.status, StatusCode::UNAUTHORIZED);
    assert_eq!(
        mcp.challenge,
        format!("Bearer resource_metadata=\"{metadata}/mcp/v1/ws1\"")
    );

    let other_audience = key.token(&issuer, "sub-x", "api://someone-else");
    let refused = h.with_bearer("/api/v1/auth/me", &other_audience).await;
    assert_eq!(refused.status, StatusCode::UNAUTHORIZED);
    assert!(
        refused.body.contains("access token refused"),
        "{}",
        refused.body
    );
    assert_eq!(
        refused.challenge,
        format!("Bearer resource_metadata=\"{metadata}\", error=\"invalid_token\"")
    );
    assert!(
        h.app
            .control
            .find_user_by_oidc_subject(&OidcSubject::from("sub-x"))
            .await
            .is_ok_and(|u| u.is_none())
    );
    let denied = h.audit("token").await;
    assert_eq!(denied, vec![(Outcome::Denied, None)]);
}

#[tokio::test]
async fn the_metadata_names_each_resource_and_its_issuer() {
    let (h, _, issuer) = Harness::resource().await;
    let document =
        |reply: &Reply| serde_json::from_str::<Value>(&reply.body).unwrap_or(Value::Null);

    let root = h.get("/.well-known/oauth-protected-resource", None).await;
    assert_eq!(root.status, StatusCode::OK, "{}", root.body);
    let root = document(&root);
    assert_eq!(root["resource"], "https://quack.example.com");
    assert_eq!(root["authorization_servers"], json!([issuer]));
    assert_eq!(root["bearer_methods_supported"], json!(["header"]));
    // Only sign-in scopes are configured, so none are advertised.
    assert!(root.get("scopes_supported").is_none());

    let mcp = h
        .get("/.well-known/oauth-protected-resource/mcp/v1/ws1", None)
        .await;
    assert_eq!(
        document(&mcp)["resource"],
        "https://quack.example.com/mcp/v1/ws1"
    );
    let api = h
        .get("/.well-known/oauth-protected-resource/api/v1", None)
        .await;
    assert_eq!(
        document(&api)["resource"],
        "https://quack.example.com/api/v1"
    );
    for unknown in [
        "/.well-known/oauth-protected-resource/etc/passwd",
        "/.well-known/oauth-protected-resource/mcp/v1/a/b",
    ] {
        assert_eq!(
            h.get(unknown, None).await.status,
            StatusCode::NOT_FOUND,
            "{unknown}"
        );
    }
}

#[tokio::test]
async fn without_an_audience_nothing_is_published_or_accepted() {
    let h = Harness::new().await;
    let key = IssuerKey::new();
    let base = h.issuer.lock().map(|s| s.base.clone()).unwrap_or_default();
    h.issuer(|s| s.jwks = vec![key.jwk.clone()]);

    let metadata = h.get("/.well-known/oauth-protected-resource", None).await;
    assert_eq!(metadata.status, StatusCode::NOT_FOUND);
    let anonymous = h.get("/api/v1/auth/me", None).await;
    assert_eq!(anonymous.status, StatusCode::UNAUTHORIZED);
    assert!(anonymous.challenge.is_empty());
    // A well-formed token is an unknown API token here, not a user.
    let token = key.token(&base, "sub-y", AUDIENCE);
    assert_eq!(
        h.with_bearer("/api/v1/auth/me", &token).await.status,
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_mcp_client_with_the_issuers_token_opens_a_session_once_a_member() {
    use quack_core::storage::control::Role;
    let (h, key, issuer) = Harness::resource().await;
    let token = key.token(&issuer, "sub-claude", AUDIENCE);
    let ws = h
        .app
        .control
        .create_workspace("w")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()))
        .id;
    let initialize = |token: &str| {
        Request::post(format!("/mcp/v1/{ws}"))
            .header(header::HOST, "localhost")
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::ACCEPT, "application/json, text/event-stream")
            .header(header::AUTHORIZATION, format!("Bearer {token}"))
            .body(Body::from(
                json!({
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18", "capabilities": {},
                        "clientInfo": { "name": "test", "version": "0" }
                    }
                })
                .to_string(),
            ))
            .unwrap_or_else(|e| fail(&e.to_string()))
    };

    // Known now, but not a member: authenticated, then refused.
    let outsider = h.send(initialize(&token)).await;
    assert_eq!(outsider.status, StatusCode::FORBIDDEN, "{}", outsider.body);
    let user = h
        .app
        .control
        .find_user_by_oidc_subject(&OidcSubject::from("sub-claude"))
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| fail("no user"));
    assert!(
        h.app
            .control
            .set_member(&ws, &user.id, Role::Viewer)
            .await
            .is_ok()
    );
    let member = h.send(initialize(&token)).await;
    assert_eq!(member.status, StatusCode::OK, "{}", member.body);
    assert!(member.body.contains("\"serverInfo\""), "{}", member.body);
}

// --- on behalf of the caller -------------------------------------------------

/// A token for an on-behalf-of provider at the mock issuer, as the request's
/// caller: what a turn's model request would send.
async fn obo_token(app: &App) -> Result<String, crate::server::error::ApiError> {
    obo_token_as(app, quack_core::config::ClientAuth::ClientSecretPost).await
}

/// As [`obo_token`], with the provider's client authenticating by `auth`.
async fn obo_token_as(
    app: &App,
    auth: quack_core::config::ClientAuth,
) -> Result<String, crate::server::error::ApiError> {
    use quack_core::config::{Exchange, Grant, OAuthConfig};
    use quack_core::llm::oauth::TokenManager;
    use secrecy::ExposeSecret;
    let issuer = app
        .config
        .server
        .oidc
        .as_ref()
        .map(|o| o.issuer_url.clone())
        .unwrap_or_default();
    let oauth = OAuthConfig {
        issuer_url: issuer,
        client_id: Some(String::from("quack")),
        scopes: Vec::new(),
        redirect_uri: String::from("http://127.0.0.1:1/callback"),
        grant: Grant::OnBehalfOf,
        client_secret_env: (auth != ClientAuth::PrivateKeyJwt)
            .then(|| String::from("CARGO_PKG_NAME")),
        client_auth: auth,
        exchange: Exchange::TokenExchange,
        audience: Some(String::from("api://model")),
        resource: None,
        actor: false,
    };
    let name = "model".parse().map_err(|e: quack_core::error::Error| {
        crate::server::error::ApiError::internal(e.to_string())
    })?;
    let manager = TokenManager::new(&app.config, &name, oauth, KeySource::File)?;
    Ok(manager.access_token().await?.expose_secret().to_owned())
}

async fn probe(
    axum::extract::State(app): axum::extract::State<App>,
    _caller: crate::server::auth::Identity,
) -> Result<String, crate::server::error::ApiError> {
    obo_token(&app).await
}

/// The same, for a provider that signs client assertions.
async fn probe_key(
    axum::extract::State(app): axum::extract::State<App>,
    _caller: crate::server::auth::Identity,
) -> Result<String, crate::server::error::ApiError> {
    obo_token_as(&app, ClientAuth::PrivateKeyJwt).await
}

/// The same, from inside a background job the caller submits.
async fn probe_job(
    axum::extract::State(app): axum::extract::State<App>,
    _caller: crate::server::auth::Identity,
) -> Result<String, crate::server::error::ApiError> {
    use quack_core::jobs::{JobKind, JobSpec};
    let worker = Arc::clone(&app);
    let job = app
        .jobs
        .submit(JobSpec::new(JobKind::Ingest, "probe"), |_| async move {
            obo_token(&worker).await.map_err(|e| e.message)
        });
    let done = app
        .jobs
        .wait(job.id)
        .await
        .ok_or_else(|| crate::server::error::ApiError::internal("job forgotten"))?;
    done.outcome
        .ok_or_else(|| crate::server::error::ApiError::internal("no outcome"))
}

impl Harness {
    /// The probe routes, behind the same acting slot the server uses.
    async fn probe(&self, uri: &str, credential: (&str, &str)) -> Reply {
        let router = Router::new()
            .route("/probe", get(probe))
            .route("/probe-job", get(probe_job))
            .route("/probe-key", get(probe_key))
            .layer(axum::middleware::from_fn(crate::server::acting_slot))
            .with_state(Arc::clone(&self.app));
        let request = Request::get(uri)
            .header(credential.0, credential.1)
            .body(Body::empty())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let response = router
            .oneshot(request)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap_or_default();
        Reply {
            status,
            location: String::new(),
            cookies: Vec::new(),
            challenge: String::new(),
            body: String::from_utf8_lossy(&bytes).into_owned(),
        }
    }
}

#[tokio::test]
async fn a_signed_in_persons_model_requests_are_exchanged_for_them_in_jobs_too() {
    let h = Harness::new().await;
    let session = h.sign_in("sub-obo", "obo").await;
    for uri in ["/probe", "/probe-job"] {
        let reply = h.probe(uri, ("cookie", &session)).await;
        assert_eq!(reply.status, StatusCode::OK, "{uri}: {}", reply.body);
        // The stored sign-in's access token is the subject.
        assert_eq!(reply.body, "obo-user-access", "{uri}");
    }
}

#[tokio::test]
async fn a_bearer_callers_own_token_is_exchanged() {
    let (h, key, issuer) = Harness::resource().await;
    let token = key.token(&issuer, "sub-bearer", AUDIENCE);
    let reply = h
        .probe("/probe", ("authorization", &format!("Bearer {token}")))
        .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    assert_eq!(reply.body, format!("obo-{token}"));
}

#[tokio::test]
async fn a_password_user_is_refused_rather_than_sent_as_quack() {
    let h = Harness::new().await;
    let created = h
        .app
        .control
        .create_user(
            "pw",
            "secret",
            quack_core::storage::control::UserKind::Standard,
        )
        .await;
    assert!(created.is_ok());
    let login = Request::post("/api/v1/auth/login")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({ "username": "pw", "password": "secret" }).to_string(),
        ))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let login = h.send(login).await;
    let reply: Value = serde_json::from_str(&login.body).unwrap_or(Value::Null);
    let token = reply["token"].as_str().unwrap_or_default().to_owned();
    assert!(!token.is_empty(), "{}", login.body);

    let refused = h
        .probe("/probe", ("authorization", &format!("Bearer {token}")))
        .await;
    assert_eq!(refused.status, StatusCode::FORBIDDEN, "{}", refused.body);
    assert!(
        refused.body.contains("no current sign-in through"),
        "{}",
        refused.body
    );
}

#[tokio::test]
async fn a_callback_naming_another_issuer_is_refused() {
    let h = Harness::new().await;
    let (state, _, cookie) = h.start().await;
    let reply = h
        .get(
            &format!(
                "{}?code=c&state={state}&iss=https%3A%2F%2Fevil.example.com",
                OidcConfig::CALLBACK_PATH
            ),
            Some(&format!("{}={cookie}", super::STATE_COOKIE)),
        )
        .await;
    assert_eq!(reply.status, StatusCode::SEE_OTHER, "{}", reply.body);
    assert!(
        reply.location.starts_with("/login?error=") && reply.location.contains("9207"),
        "{}",
        reply.location
    );
    assert!(reply.cookie(crate::server::auth::SESSION_COOKIE).is_none());
}

/// Issue #246: a TLS-terminating proxy on the same host reaches quack over
/// loopback, so the peer alone cannot say the browser is on https. An https
/// `redirect_uri` can, and then the state cookie and the session cookie
/// both carry `Secure` on loopback too.
#[tokio::test]
async fn an_https_public_url_makes_loopback_cookies_secure() {
    let h = Harness::new().await;
    let loopback = "127.0.0.1:51000";
    let started = h.get_from(OidcConfig::START_PATH, None, loopback).await;
    assert_eq!(started.status, StatusCode::SEE_OTHER, "{}", started.body);
    let state_cookie = started
        .cookies
        .iter()
        .find(|c| c.starts_with(&format!("{}=", super::STATE_COOKIE)))
        .cloned()
        .unwrap_or_else(|| fail("no state cookie"));
    assert!(state_cookie.contains("Secure"), "{state_cookie}");

    let state = started.cookie(super::STATE_COOKIE).unwrap_or_default();
    let nonce = started
        .location
        .parse::<axum::http::Uri>()
        .ok()
        .and_then(|uri| axum::extract::Query::<HashMap<String, String>>::try_from_uri(&uri).ok())
        .and_then(|axum::extract::Query(q)| q.get("nonce").cloned())
        .unwrap_or_default();
    let base = h.issuer.lock().map(|s| s.base.clone()).unwrap_or_default();
    h.issuer(|s| {
        s.id_claims = json!({
            "iss": base, "sub": "sub-ada", "aud": "quack",
            "exp": jiff::Timestamp::now().as_second().saturating_add(3600),
            "nonce": nonce, "preferred_username": "ada",
        });
    });
    let reply = h
        .get_from(
            &format!("{}?code=c&state={state}", OidcConfig::CALLBACK_PATH),
            Some(&format!("{}={state}", super::STATE_COOKIE)),
            loopback,
        )
        .await;
    assert_eq!(reply.status, StatusCode::SEE_OTHER, "{}", reply.body);
    let session = reply
        .cookies
        .iter()
        .find(|c| c.starts_with(&format!("{}=", crate::server::auth::SESSION_COOKIE)))
        .cloned()
        .unwrap_or_else(|| fail("no session cookie"));
    assert!(session.contains("Secure"), "{session}");
}

/// Without an https public URL, loopback keeps the plain-HTTP local case:
/// no `Secure`, or the browser would never send the cookie back.
#[tokio::test]
async fn an_http_public_url_leaves_loopback_cookies_plain() {
    let h = Harness::build(None, "http://127.0.0.1:8080").await;
    let started = h
        .get_from(OidcConfig::START_PATH, None, "127.0.0.1:51000")
        .await;
    assert_eq!(started.status, StatusCode::SEE_OTHER, "{}", started.body);
    assert!(
        started.cookie(super::STATE_COOKIE).is_some(),
        "{:?}",
        started.cookies
    );
    assert!(
        started.cookies.iter().all(|c| !c.contains("Secure")),
        "{:?}",
        started.cookies
    );
    // Off loopback it is still set, whatever the public URL says.
    let remote = h
        .get_from(OidcConfig::START_PATH, None, "203.0.113.7:51000")
        .await;
    assert!(
        remote.cookies.iter().any(|c| c.contains("Secure")),
        "{:?}",
        remote.cookies
    );
}

// --- the most secure on-behalf-of setup (Vouch) ------------------------------

/// Check a client assertion with the registered key: ES256 by that key,
/// `iss` and `sub` quack's client, `aud` the issuer, a lifetime of at most a
/// minute, and a `jti` not seen before.
fn check_assertion(
    form: &HashMap<String, String>,
    key: &PublicJwk,
    base: &str,
    seen: &mut Vec<String>,
) {
    use jsonwebtoken::{Algorithm, DecodingKey, Validation};
    assert_eq!(
        form.get("client_assertion_type").map(String::as_str),
        Some("urn:ietf:params:oauth:client-assertion-type:jwt-bearer")
    );
    assert_eq!(form.get("client_id").map(String::as_str), Some("quack"));
    assert!(!form.contains_key("client_secret"), "{form:?}");
    let assertion = form.get("client_assertion").cloned().unwrap_or_default();
    let jwk: jsonwebtoken::jwk::Jwk =
        serde_json::from_value(serde_json::to_value(key).unwrap_or(Value::Null))
            .unwrap_or_else(|e| fail(&e.to_string()));
    let decoding = DecodingKey::from_jwk(&jwk).unwrap_or_else(|e| fail(&e.to_string()));
    let mut validation = Validation::new(Algorithm::ES256);
    validation.set_issuer(&["quack"]);
    validation.set_audience(&[base]);
    validation.sub = Some(String::from("quack"));
    let claims = jsonwebtoken::decode::<Value>(&assertion, &decoding, &validation)
        .unwrap_or_else(|e| fail(&format!("assertion refused: {e}")))
        .claims;
    let lifetime = claims["exp"]
        .as_i64()
        .unwrap_or_default()
        .saturating_sub(claims["iat"].as_i64().unwrap_or_default());
    assert!((1..=60).contains(&lifetime), "{claims}");
    let jti = claims["jti"].as_str().unwrap_or_default().to_owned();
    assert!(!jti.is_empty() && !seen.contains(&jti), "jti {jti} reused");
    seen.push(jti);
}

#[tokio::test]
async fn sign_in_pushes_its_request_and_signs_every_token_request_then_exchanges_for_the_person() {
    let h = Harness::build_with(None, "https://quack.example.com", |c| {
        c.client_auth = ClientAuth::PrivateKeyJwt;
    })
    .await;
    h.issuer(|s| s.par = true);
    let base = h.issuer.lock().map(|s| s.base.clone()).unwrap_or_default();

    // Out: the browser carries only a reference to the pushed request.
    let reply = h.get(OidcConfig::START_PATH, None).await;
    assert_eq!(reply.status, StatusCode::SEE_OTHER, "{}", reply.body);
    let location: axum::http::Uri = reply.location.parse().unwrap_or_else(|_| fail("location"));
    let query: HashMap<String, String> = axum::extract::Query::try_from_uri(&location)
        .map(|axum::extract::Query(q)| q)
        .unwrap_or_default();
    assert_eq!(query.len(), 2, "{}", reply.location);
    assert_eq!(query.get("client_id").map(String::as_str), Some("quack"));
    assert!(query.contains_key("request_uri"));
    let pushed = h
        .issuer
        .lock()
        .map(|s| s.pushed.clone())
        .unwrap_or_default();
    let [pushed] = pushed.as_slice() else {
        fail(&format!("{pushed:?}"));
    };
    assert_eq!(
        pushed.get("code_challenge_method").map(String::as_str),
        Some("S256")
    );
    assert!(pushed.contains_key("code_challenge"));

    // The key the assertions are signed with, as `quack auth jwks` prints it.
    let key = ClientKeys::new(&h.app.config, KeySource::File)
        .existing(&ClientKeyName::new(&base, "quack"))
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| fail("no client key was made"));
    let mut seen = Vec::new();
    check_assertion(pushed, key.jwk(), &base, &mut seen);

    // Back: the code exchange is signed too.
    let state = pushed.get("state").cloned().unwrap_or_default();
    let nonce = pushed.get("nonce").cloned().unwrap_or_default();
    let cookie = reply
        .cookie(super::STATE_COOKIE)
        .unwrap_or_else(|| fail("no state cookie"));
    assert_eq!(state, cookie);
    h.issuer(|s| {
        s.id_claims = json!({
            "iss": base, "sub": "sub-vouch", "aud": "quack",
            "exp": Timestamp::now().as_second().saturating_add(3600),
            "nonce": nonce, "email": "ada@example.com",
        });
    });
    let callback = h
        .get(
            &format!("{}?code=c&state={state}", OidcConfig::CALLBACK_PATH),
            Some(&format!("{}={cookie}", super::STATE_COOKIE)),
        )
        .await;
    assert_eq!(callback.status, StatusCode::SEE_OTHER, "{}", callback.body);
    let session = callback
        .cookie(crate::server::auth::SESSION_COOKIE)
        .unwrap_or_else(|| fail("no session cookie"));

    // On behalf of the person: the provider shares the sign-in's key and
    // sends no actor token, no secret, and no Basic header.
    let obo = h
        .probe(
            "/probe-key",
            (
                "cookie",
                &format!("{}={session}", crate::server::auth::SESSION_COOKIE),
            ),
        )
        .await;
    assert_eq!(obo.status, StatusCode::OK, "{}", obo.body);
    assert_eq!(obo.body, "obo-user-access");

    let requests = h
        .issuer
        .lock()
        .map(|s| s.token_requests.clone())
        .unwrap_or_default();
    let grants: Vec<&str> = requests
        .iter()
        .filter_map(|(form, _)| form.get("grant_type").map(String::as_str))
        .collect();
    assert_eq!(
        grants,
        [
            "authorization_code",
            "urn:ietf:params:oauth:grant-type:token-exchange"
        ]
    );
    for (form, authorization) in &requests {
        assert!(authorization.is_none(), "{authorization:?}");
        assert!(!form.contains_key("actor_token"), "{form:?}");
        check_assertion(form, key.jwk(), &base, &mut seen);
    }
}
