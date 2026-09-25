//! Token manager and login flows against a mock identity provider.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use std::path::Path;

use super::*;

/// What the mock issuer does and what it saw.
#[derive(Default)]
struct MockState {
    refresh_requests: AtomicUsize,
    refresh_fails: std::sync::atomic::AtomicBool,
    /// Device polls answered `authorization_pending` before success.
    pending_polls: AtomicUsize,
    code_exchanges: AtomicUsize,
    client_credentials_requests: AtomicUsize,
    exchanges: AtomicUsize,
    /// Refuse exchanges with `invalid_grant`.
    exchange_fails: std::sync::atomic::AtomicBool,
    last_token_body: StdMutex<String>,
    /// The `Authorization` header of the last token request.
    last_token_auth: StdMutex<String>,
    /// Serve only RFC 8414 metadata, not an `OpenID` discovery document.
    oauth_only: std::sync::atomic::AtomicBool,
    /// The `issuer` the metadata names; the base URL when unset.
    named_issuer: StdMutex<Option<String>>,
    /// `grant_types_supported`, when listed.
    grants: StdMutex<Option<Vec<String>>>,
}

struct MockIdp {
    issuer: String,
    state: Arc<MockState>,
}

impl MockIdp {
    async fn start() -> Self {
        let Ok(listener) = TcpListener::bind("127.0.0.1:0").await else {
            fail("loopback bind failed");
        };
        let Ok(addr) = listener.local_addr() else {
            fail("loopback bind failed");
        };
        let issuer = format!("http://{addr}");
        let state = Arc::new(MockState::default());
        let served = Arc::clone(&state);
        let base = issuer.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let served = Arc::clone(&served);
                let base = base.clone();
                tokio::spawn(async move { serve(stream, &base, &served).await });
            }
        });
        Self { issuer, state }
    }
}

async fn serve(mut stream: tokio::net::TcpStream, base: &str, state: &MockState) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    let (head_len, content_len) = loop {
        let Ok(n) = stream.read(&mut chunk).await else {
            return;
        };
        if n == 0 {
            return;
        }
        buf.extend_from_slice(chunk.get(..n).unwrap_or_default());
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(buf.get(..pos).unwrap_or_default()).to_string();
            let len = head
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                })
                .unwrap_or(0);
            break (pos.saturating_add(4), len);
        }
    };
    while buf.len() < head_len.saturating_add(content_len) {
        let Ok(n) = stream.read(&mut chunk).await else {
            return;
        };
        if n == 0 {
            break;
        }
        buf.extend_from_slice(chunk.get(..n).unwrap_or_default());
    }
    let head = String::from_utf8_lossy(buf.get(..head_len).unwrap_or_default()).to_string();
    let body = String::from_utf8_lossy(buf.get(head_len..).unwrap_or_default()).to_string();
    let target = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .unwrap_or("/")
        .to_owned();
    if target == "/token"
        && let Ok(mut auth) = state.last_token_auth.lock()
    {
        *auth = head
            .lines()
            .find_map(|l| {
                l.split_once(':')
                    .filter(|(k, _)| k.eq_ignore_ascii_case("authorization"))
                    .map(|(_, v)| v.trim().to_owned())
            })
            .unwrap_or_default();
    }
    let (status, json) = route(&target, &body, base, state);
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
        json.len()
    );
    drop(stream.write_all(response.as_bytes()).await);
    drop(stream.shutdown().await);
}

fn form(body: &str, key: &str) -> Option<String> {
    oauth2::url::form_urlencoded::parse(body.as_bytes())
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.into_owned())
}

fn token_json(access: &str, refresh: Option<&str>) -> String {
    let refresh = refresh.map_or(String::new(), |r| format!(",\"refresh_token\":\"{r}\""));
    format!(
        "{{\"access_token\":\"{access}\",\"token_type\":\"bearer\",\"expires_in\":3600{refresh}}}"
    )
}

/// The discovery document, or RFC 8414 metadata when `oauth_only`.
fn metadata(target: &str, base: &str, state: &MockState) -> (&'static str, String) {
    let oauth_only = state.oauth_only.load(Ordering::SeqCst);
    if oauth_only == (target == "/.well-known/openid-configuration") {
        return ("404 Not Found", String::from("{}"));
    }
    let issuer = state
        .named_issuer
        .lock()
        .ok()
        .and_then(|n| n.clone())
        .unwrap_or_else(|| base.to_owned());
    let mut metadata = serde_json::json!({
        "issuer": issuer,
        "authorization_endpoint": format!("{base}/authorize"),
        "token_endpoint": format!("{base}/token"),
        "device_authorization_endpoint": format!("{base}/device"),
    });
    if let (Some(grants), Some(fields)) = (
        state.grants.lock().ok().and_then(|g| g.clone()),
        metadata.as_object_mut(),
    ) {
        fields.insert(
            String::from("grant_types_supported"),
            serde_json::json!(grants),
        );
    }
    ("200 OK", metadata.to_string())
}

fn route(target: &str, body: &str, base: &str, state: &MockState) -> (&'static str, String) {
    match target {
        "/.well-known/openid-configuration" | "/.well-known/oauth-authorization-server" => {
            metadata(target, base, state)
        }
        "/device" => (
            "200 OK",
            format!(
                "{{\"device_code\":\"dev-1\",\"user_code\":\"ABCD-EFGH\",\"verification_uri\":\"{base}/activate\",\"expires_in\":600,\"interval\":0}}"
            ),
        ),
        "/token" => {
            if let Ok(mut last) = state.last_token_body.lock() {
                last.clone_from(&body.to_owned());
            }
            match form(body, "grant_type").as_deref() {
                Some("refresh_token") => {
                    state.refresh_requests.fetch_add(1, Ordering::SeqCst);
                    if state.refresh_fails.load(Ordering::SeqCst) {
                        (
                            "400 Bad Request",
                            String::from("{\"error\":\"invalid_grant\"}"),
                        )
                    } else {
                        (
                            "200 OK",
                            token_json("refreshed-access", Some("refreshed-refresh")),
                        )
                    }
                }
                Some("authorization_code") => {
                    state.code_exchanges.fetch_add(1, Ordering::SeqCst);
                    if form(body, "code").as_deref() == Some("the-code")
                        && form(body, "code_verifier").is_some()
                    {
                        ("200 OK", token_json("code-access", Some("code-refresh")))
                    } else {
                        (
                            "400 Bad Request",
                            String::from("{\"error\":\"invalid_grant\"}"),
                        )
                    }
                }
                Some("client_credentials") => {
                    state
                        .client_credentials_requests
                        .fetch_add(1, Ordering::SeqCst);
                    if form(body, "client_secret").as_deref() == Some(env!("CARGO_PKG_NAME")) {
                        ("200 OK", token_json("service-access", None))
                    } else {
                        (
                            "401 Unauthorized",
                            String::from("{\"error\":\"invalid_client\"}"),
                        )
                    }
                }
                Some(
                    "urn:ietf:params:oauth:grant-type:token-exchange"
                    | "urn:ietf:params:oauth:grant-type:jwt-bearer",
                ) => {
                    let n = state.exchanges.fetch_add(1, Ordering::SeqCst);
                    if state.exchange_fails.load(Ordering::SeqCst) {
                        (
                            "400 Bad Request",
                            String::from(
                                "{\"error\":\"invalid_grant\",\"error_description\":\"subject token expired\"}",
                            ),
                        )
                    } else {
                        let subject = form(body, "subject_token")
                            .or_else(|| form(body, "assertion"))
                            .unwrap_or_default();
                        ("200 OK", token_json(&format!("obo-{subject}-{n}"), None))
                    }
                }
                Some("urn:ietf:params:oauth:grant-type:device_code") => {
                    let pending = state.pending_polls.load(Ordering::SeqCst);
                    if pending > 0 {
                        state
                            .pending_polls
                            .store(pending.saturating_sub(1), Ordering::SeqCst);
                        (
                            "400 Bad Request",
                            String::from("{\"error\":\"authorization_pending\"}"),
                        )
                    } else {
                        ("200 OK", token_json("device-access", None))
                    }
                }
                _ => (
                    "400 Bad Request",
                    String::from("{\"error\":\"unsupported_grant_type\"}"),
                ),
            }
        }
        _ => ("404 Not Found", String::from("{}")),
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map_or(19876, |a| a.port())
}

/// An environment variable `cargo test` always sets, standing in for a
/// client secret so no test has to set one.
const SECRET_ENV: &str = "CARGO_PKG_NAME";

fn oauth_config(issuer: &str, grant: Grant) -> OAuthConfig {
    OAuthConfig {
        issuer_url: issuer.to_owned(),
        client_id: String::from("client-1"),
        scopes: vec![
            String::from("api://x/.default"),
            String::from("offline_access"),
        ],
        redirect_uri: format!("http://127.0.0.1:{}/callback", free_port()),
        grant,
        client_secret_env: match grant {
            Grant::ClientCredentials | Grant::OnBehalfOf => Some(String::from(SECRET_ENV)),
            Grant::AuthorizationCode | Grant::DeviceCode => None,
        },
        client_auth: ClientAuth::default(),
        exchange: Exchange::default(),
        audience: None,
        resource: None,
        actor: true,
    }
}

fn temp() -> tempfile::TempDir {
    tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()))
}

/// Fail the test with a message; `!` lets it sit in a `let ... else`.
fn name(text: &str) -> ProviderName {
    text.parse().unwrap_or_else(|e: Error| fail(&e.to_string()))
}

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

/// A config whose data directory is `dir`: the token lands in its
/// `control.db`, sealed under its `vault.key`.
fn config_at(dir: &Path) -> Config {
    let mut config = Config::default();
    config.general.data_dir = dir.to_path_buf();
    config
}

fn manager(dir: &Path, idp: &MockIdp, grant: Grant) -> TokenManager {
    match TokenManager::new(
        &config_at(dir),
        &name("p"),
        oauth_config(&idp.issuer, grant),
        KeySource::File,
    ) {
        Ok(m) => m,
        Err(e) => fail(&format!("manager build failed: {e}")),
    }
}

fn seed(offset: SignedDuration, refresh: Option<&str>) -> CachedToken {
    CachedToken {
        access_token: SecretString::from(String::from("seeded-access")),
        expires_at: Timestamp::now()
            .checked_add(offset)
            .unwrap_or(Timestamp::MAX),
        refresh_token: refresh.map(|r| SecretString::from(r.to_owned())),
    }
}

#[tokio::test]
async fn fresh_cached_token_is_reused_without_the_network() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let m = manager(dir.path(), &idp, Grant::AuthorizationCode);
    assert!(
        m.store
            .store(&seed(SignedDuration::from_hours(1), Some("r")))
            .await
            .is_ok()
    );
    let first = m.access_token().await;
    let second = m.access_token().await;
    assert!(first.is_ok_and(|t| t.expose_secret() == "seeded-access"));
    assert!(second.is_ok_and(|t| t.expose_secret() == "seeded-access"));
    assert_eq!(idp.state.refresh_requests.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn expiring_token_is_refreshed_once_under_concurrency() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let m = Arc::new(manager(dir.path(), &idp, Grant::AuthorizationCode));
    assert!(
        m.store
            .store(&seed(SignedDuration::from_secs(30), Some("r")))
            .await
            .is_ok()
    );
    let calls: Vec<_> = (0..8)
        .map(|_| {
            let m = Arc::clone(&m);
            tokio::spawn(async move { m.access_token().await })
        })
        .collect();
    for call in calls {
        let result = call.await;
        assert!(result.is_ok_and(|r| r.is_ok_and(|t| t.expose_secret() == "refreshed-access")));
    }
    assert_eq!(idp.state.refresh_requests.load(Ordering::SeqCst), 1);
    let stored = m.store.load().await;
    assert!(stored.is_ok_and(|t| t.is_some_and(|t| {
        t.access_token.expose_secret() == "refreshed-access"
            && t.refresh_token.as_ref().map(ExposeSecret::expose_secret)
                == Some("refreshed-refresh")
    })));
    let body = idp
        .state
        .last_token_body
        .lock()
        .map(|b| b.clone())
        .unwrap_or_default();
    assert!(body.contains("client_id=client-1"), "{body}");
}

#[tokio::test]
async fn refresh_keeps_the_old_refresh_token_when_the_issuer_omits_one() {
    let response: oauth2::basic::BasicTokenResponse =
        serde_json::from_str(&token_json("a", None)).unwrap_or_else(|e| fail(&e.to_string()));
    let token = CachedToken::from_response(&response);
    assert!(token.refresh_token.is_none());
    assert!(token.is_fresh(Timestamp::now(), REUSE_MARGIN));
}

#[tokio::test]
async fn missing_cache_and_failed_refresh_both_require_a_login() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let m = manager(dir.path(), &idp, Grant::AuthorizationCode);
    let err = m.access_token().await.err();
    assert!(
        err.as_ref()
            .is_some_and(|e| matches!(e, Error::AuthRequired { .. }))
    );
    assert!(err.is_some_and(|e| e.to_string().contains("quack auth login p")));

    assert!(
        m.store
            .store(&seed(SignedDuration::from_secs(-5), Some("r")))
            .await
            .is_ok()
    );
    idp.state.refresh_fails.store(true, Ordering::SeqCst);
    let err = m.access_token().await.err();
    assert!(err.is_some_and(|e| matches!(
        &e,
        Error::AuthRequired {
            reason: AuthReason::RefreshFailed(_),
            ..
        }
    )));

    assert!(
        m.store
            .store(&seed(SignedDuration::from_secs(-5), None))
            .await
            .is_ok()
    );
    let err = m.access_token().await.err();
    assert!(err.is_some_and(|e| e.to_string().contains("no refresh token")));
}

#[tokio::test]
async fn device_code_login_polls_until_approved_and_caches() {
    let idp = MockIdp::start().await;
    idp.state.pending_polls.store(2, Ordering::SeqCst);
    let dir = temp();
    let m = manager(dir.path(), &idp, Grant::DeviceCode);
    let prompts = StdMutex::new(Vec::new());
    let notify = |p: LoginPrompt| {
        if let Ok(mut v) = prompts.lock() {
            v.push(p);
        }
    };
    let token = m.login(LoginFlow::Configured, &notify).await;
    assert!(token.is_ok_and(|t| t.access_token.expose_secret() == "device-access"));
    let seen = prompts.into_inner().unwrap_or_default();
    assert!(matches!(
        seen.as_slice(),
        [LoginPrompt::DeviceCode { user_code, verification_uri, .. }]
            if user_code == "ABCD-EFGH" && verification_uri.ends_with("/activate")
    ));
    assert_eq!(idp.state.pending_polls.load(Ordering::SeqCst), 0);
    let status = m.status().await;
    assert!(
        status.is_ok_and(|s| s.token.is_some_and(|t| t.renewal == Renewal::Relogin)
            && s.key_location == KeyLocation::File)
    );
    assert!(
        m.access_token()
            .await
            .is_ok_and(|t| t.expose_secret() == "device-access")
    );
    assert!(m.logout().await.is_ok());
    assert!(m.status().await.is_ok_and(|s| s.token.is_none()));
    assert!(m.access_token().await.is_err());
}

#[tokio::test]
async fn client_credentials_needs_no_login_and_runs_again_when_the_token_runs_out() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let m = manager(dir.path(), &idp, Grant::ClientCredentials);
    let requests = || idp.state.client_credentials_requests.load(Ordering::SeqCst);

    assert!(m.status().await.is_ok_and(|s| s.token.is_none()));
    let first = m.access_token().await;
    assert!(first.is_ok_and(|t| t.expose_secret() == "service-access"));
    assert!(
        m.access_token()
            .await
            .is_ok_and(|t| t.expose_secret() == "service-access")
    );
    assert_eq!(requests(), 1);
    let body = idp
        .state
        .last_token_body
        .lock()
        .map(|b| b.clone())
        .unwrap_or_default();
    assert!(body.contains("client_id=client-1"), "{body}");
    assert!(body.contains("scope=api"), "{body}");
    assert!(
        m.status()
            .await
            .is_ok_and(|s| s.token.is_some_and(|t| t.renewal == Renewal::Regrant))
    );

    // An expiring token is replaced by the grant, never refreshed, even when
    // the issuer handed out a refresh token.
    let fresh = manager(dir.path(), &idp, Grant::ClientCredentials);
    assert!(
        fresh
            .store
            .store(&seed(SignedDuration::from_secs(30), Some("r")))
            .await
            .is_ok()
    );
    assert!(
        fresh
            .access_token()
            .await
            .is_ok_and(|t| t.expose_secret() == "service-access")
    );
    assert_eq!(requests(), 2);
    assert_eq!(idp.state.refresh_requests.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn client_credentials_login_checks_the_credentials_and_ignores_the_device_flow() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let m = manager(dir.path(), &idp, Grant::ClientCredentials);
    let prompted = AtomicUsize::new(0);
    let notify = |_: LoginPrompt| {
        prompted.fetch_add(1, Ordering::SeqCst);
    };
    let token = m.login(LoginFlow::DeviceCode, &notify).await;
    assert!(token.is_ok_and(|t| t.access_token.expose_secret() == "service-access"));
    assert_eq!(prompted.load(Ordering::SeqCst), 0);
    assert!(m.status().await.is_ok_and(|s| s.token.is_some()));
}

#[tokio::test]
async fn a_refused_client_secret_is_an_error_not_a_login_prompt() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let mut config = oauth_config(&idp.issuer, Grant::ClientCredentials);
    // Set by cargo for every test run, and not the secret the issuer expects.
    config.client_secret_env = Some(String::from("CARGO_PKG_VERSION"));
    let Ok(m) = TokenManager::new(&config_at(dir.path()), &name("p"), config, KeySource::File)
    else {
        fail("manager build failed");
    };
    let err = m.access_token().await.err();
    assert!(err.as_ref().is_some_and(|e| matches!(e, Error::Llm(_))));
    assert!(err.is_some_and(|e| e.to_string().contains("client-credentials grant failed")),);
    assert!(m.status().await.is_ok_and(|s| s.token.is_none()));
}

#[tokio::test]
async fn browser_login_rejects_bad_state_then_accepts_the_code() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let m = Arc::new(manager(dir.path(), &idp, Grant::AuthorizationCode));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let login = {
        let m = Arc::clone(&m);
        tokio::spawn(async move {
            let notify = move |p: LoginPrompt| drop(tx.send(p));
            m.login(LoginFlow::Configured, &notify).await
        })
    };
    let Some(LoginPrompt::Browser { url }) = rx.recv().await else {
        fail("expected a browser prompt");
    };
    let Ok(auth_url) = Url::parse(&url) else {
        fail(&format!("auth url unparsable: {url}"));
    };
    let q: HashMap<_, _> = auth_url.query_pairs().into_owned().collect();
    assert_eq!(
        q.get("code_challenge_method").map(String::as_str),
        Some("S256")
    );
    assert!(q.get("scope").is_some_and(|s| s.contains("offline_access")));
    let state = q.get("state").cloned().unwrap_or_default();
    let redirect = q.get("redirect_uri").cloned().unwrap_or_default();
    assert!(redirect.ends_with("/callback"));

    let http = reqwest::Client::new();
    let probe = http.get(format!("{redirect}/../favicon.ico")).send().await;
    assert!(probe.is_ok_and(|r| r.status() == 404));
    let bad = http
        .get(format!("{redirect}?code=the-code&state=nope"))
        .send()
        .await;
    assert!(bad.is_ok_and(|r| r.status() == 400));
    let good = http
        .get(format!("{redirect}?code=the-code&state={state}"))
        .send()
        .await;
    assert!(good.is_ok_and(|r| r.status() == 200));

    let token = login.await;
    assert!(token.is_ok_and(|t| t.is_ok_and(|t| t.access_token.expose_secret() == "code-access")));
    assert_eq!(idp.state.code_exchanges.load(Ordering::SeqCst), 1);
    assert!(
        m.access_token()
            .await
            .is_ok_and(|t| t.expose_secret() == "code-access")
    );
}

#[tokio::test]
async fn browser_login_reports_the_issuer_error() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let m = Arc::new(manager(dir.path(), &idp, Grant::AuthorizationCode));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let login = {
        let m = Arc::clone(&m);
        tokio::spawn(async move {
            let notify = move |p: LoginPrompt| drop(tx.send(p));
            m.login(LoginFlow::Configured, &notify).await
        })
    };
    let Some(LoginPrompt::Browser { url }) = rx.recv().await else {
        fail("expected a browser prompt");
    };
    let Ok(auth_url) = Url::parse(&url) else {
        fail("auth url unparsable");
    };
    let q: HashMap<_, _> = auth_url.query_pairs().into_owned().collect();
    let state = q.get("state").cloned().unwrap_or_default();
    let redirect = q.get("redirect_uri").cloned().unwrap_or_default();
    let sent = reqwest::Client::new()
        .get(format!(
            "{redirect}?error=access_denied&error_description=nope&state={state}"
        ))
        .send()
        .await;
    assert!(sent.is_ok());
    let err = login.await.ok().and_then(std::result::Result::err);
    assert!(err.is_some_and(|e| e.to_string().contains("access_denied: nope")));
    assert!(m.status().await.is_ok_and(|s| s.token.is_none()));
}

#[tokio::test]
async fn device_login_without_a_device_endpoint_is_an_error() {
    let dir = temp();
    let config = oauth_config("http://127.0.0.1:9", Grant::DeviceCode);
    let Ok(m) = TokenManager::new(&config_at(dir.path()), &name("p"), config, KeySource::File)
    else {
        fail("manager build failed");
    };
    let endpoints = Endpoints {
        issuer: None,
        authorization: String::from("http://127.0.0.1:9/a"),
        token: String::from("http://127.0.0.1:9/t"),
        device_authorization: None,
        jwks_uri: None,
        grant_types_supported: None,
        authorization_response_iss_parameter_supported: false,
    };
    assert!(m.endpoints.set(endpoints).is_ok());
    let err = m.login(LoginFlow::Configured, &|_| {}).await.err();
    assert!(err.is_some_and(|e| e.to_string().contains("no device_authorization_endpoint")));
}

#[tokio::test]
async fn discovery_failure_is_reported_with_the_url() {
    let dir = temp();
    let config = oauth_config("http://127.0.0.1:9", Grant::AuthorizationCode);
    let Ok(m) = TokenManager::new(&config_at(dir.path()), &name("p"), config, KeySource::File)
    else {
        fail("manager build failed");
    };
    assert!(
        m.store
            .store(&seed(SignedDuration::from_secs(-5), Some("r")))
            .await
            .is_ok()
    );
    let err = m.access_token().await.err();
    assert!(err.is_some_and(|e| e.to_string().contains("openid-configuration")));
}

#[test]
fn shared_manager_is_one_per_provider() {
    let dir = temp();
    let oauth = oauth_config("http://127.0.0.1:9", Grant::AuthorizationCode);
    let a = TokenManager::shared(&config_at(dir.path()), &name("shared"), &oauth);
    let b = TokenManager::shared(&config_at(dir.path()), &name("shared"), &oauth);
    assert!(matches!((&a, &b), (Ok(a), Ok(b)) if Arc::ptr_eq(a, b)));
    let other = TokenManager::shared(&config_at(dir.path()), &name("other"), &oauth);
    assert!(matches!((&a, &other), (Ok(a), Ok(o)) if !Arc::ptr_eq(a, o)));
}

#[test]
fn random_tokens_are_unique_and_url_safe() {
    let (a, b) = (random_token(), random_token());
    assert!(matches!((&a, &b), (Ok(a), Ok(b)) if a != b && a.len() == 43));
    assert!(a.is_ok_and(|t| {
        t.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    }));
}

#[tokio::test]
async fn a_login_is_sealed_in_control_db_and_outlives_the_process() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let m = manager(dir.path(), &idp, Grant::ClientCredentials);
    assert!(m.login(LoginFlow::Configured, &|_| {}).await.is_ok());

    // A new manager is what a new `quack` process builds.
    let later = manager(dir.path(), &idp, Grant::ClientCredentials);
    assert!(
        later
            .access_token()
            .await
            .is_ok_and(|t| t.expose_secret() == "service-access")
    );
    assert_eq!(
        idp.state.client_credentials_requests.load(Ordering::SeqCst),
        1
    );
    assert!(dir.path().join("control.db").exists());
    assert!(dir.path().join("vault.key").exists());
    assert!(!dir.path().join("tokens").exists());

    assert!(later.logout().await.is_ok());
    assert!(m.status().await.is_ok_and(|s| s.token.is_none()));
    // The vault key stays: other tokens are sealed with it.
    assert!(dir.path().join("vault.key").exists());
}

// --- on-behalf-of ------------------------------------------------------------

fn acting(user: &str, token: Option<&'static str>) -> Acting {
    Acting::fixed(
        UserId::from(user),
        token.ok_or("no identity-provider sign-in"),
    )
}

fn obo(dir: &Path, idp: &MockIdp, change: impl FnOnce(&mut OAuthConfig)) -> TokenManager {
    let mut config = oauth_config(&idp.issuer, Grant::OnBehalfOf);
    change(&mut config);
    TokenManager::new(&config_at(dir), &name("p"), config, KeySource::File)
        .unwrap_or_else(|e| fail(&e.to_string()))
}

fn last_body(idp: &MockIdp) -> String {
    idp.state
        .last_token_body
        .lock()
        .map(|b| b.clone())
        .unwrap_or_default()
}

#[tokio::test]
async fn nobody_behind_the_request_is_a_delegation_error_not_a_login_prompt() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let m = obo(dir.path(), &idp, |_| {});
    let err = m.access_token().await.err();
    assert!(
        matches!(&err, Some(Error::Delegation { reason, .. }) if reason.contains("no signed-in person")),
        "{err:?}"
    );
    let err = Acting::scope(Some(acting("ada", None)), m.access_token())
        .await
        .err();
    assert!(
        matches!(&err, Some(Error::Delegation { reason, .. }) if reason.contains("no identity-provider sign-in")),
        "{err:?}"
    );
    assert_eq!(idp.state.exchanges.load(Ordering::SeqCst), 0);
    let login = m.login(LoginFlow::Configured, &|_| {}).await.err();
    assert!(login.is_some_and(|e| e.to_string().contains("there is no login")));
}

#[tokio::test]
async fn a_token_exchange_names_the_person_and_quack_and_is_kept_per_person() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let m = obo(dir.path(), &idp, |c| {
        c.audience = Some(String::from("api://gw"));
    });

    let ada = Acting::scope(Some(acting("ada", Some("ada-at"))), m.access_token()).await;
    assert!(
        ada.as_ref()
            .is_ok_and(|t| t.expose_secret() == "obo-ada-at-0"),
        "{:?}",
        ada.err()
    );
    let body = last_body(&idp);
    for part in [
        "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Atoken-exchange",
        "subject_token=ada-at",
        "subject_token_type=urn%3Aietf%3Aparams%3Aoauth%3Atoken-type%3Aaccess_token",
        "requested_token_type=urn%3Aietf%3Aparams%3Aoauth%3Atoken-type%3Aaccess_token",
        "audience=api%3A%2F%2Fgw",
        "actor_token=service-access",
        "actor_token_type=urn%3Aietf%3Aparams%3Aoauth%3Atoken-type%3Aaccess_token",
        "client_id=client-1",
        "client_secret=",
    ] {
        assert!(body.contains(part), "{part} missing from {body}");
    }
    assert!(idp.state.last_token_auth.lock().is_ok_and(|a| a.is_empty()));

    // The same person again reuses their token; another person gets theirs.
    let again = Acting::scope(Some(acting("ada", Some("ada-at"))), m.access_token()).await;
    assert!(again.is_ok_and(|t| t.expose_secret() == "obo-ada-at-0"));
    let bob = Acting::scope(Some(acting("bob", Some("bob-at"))), m.access_token()).await;
    assert!(bob.is_ok_and(|t| t.expose_secret() == "obo-bob-at-1"));
    assert_eq!(idp.state.exchanges.load(Ordering::SeqCst), 2);
    // quack's own token (the actor) was obtained once and reused.
    assert_eq!(
        idp.state.client_credentials_requests.load(Ordering::SeqCst),
        1
    );
}

#[tokio::test]
async fn entras_form_sends_the_assertion_and_no_actor_and_basic_auth_uses_the_header() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let m = obo(dir.path(), &idp, |c| {
        c.exchange = Exchange::Entra;
        c.client_auth = ClientAuth::ClientSecretBasic;
    });
    let token = Acting::scope(Some(acting("ada", Some("ada-at"))), m.access_token()).await;
    assert!(token.is_ok_and(|t| t.expose_secret() == "obo-ada-at-0"));
    let body = last_body(&idp);
    for part in [
        "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer",
        "assertion=ada-at",
        "requested_token_use=on_behalf_of",
        "scope=api%3A%2F%2Fx%2F.default+offline_access",
    ] {
        assert!(body.contains(part), "{part} missing from {body}");
    }
    for absent in ["actor_token", "client_secret", "subject_token"] {
        assert!(!body.contains(absent), "{absent} in {body}");
    }
    assert_eq!(
        idp.state.client_credentials_requests.load(Ordering::SeqCst),
        0
    );
    let auth = idp
        .state
        .last_token_auth
        .lock()
        .map(|a| a.clone())
        .unwrap_or_default();
    let expected = base64::engine::general_purpose::STANDARD
        .encode(format!("client-1:{}", env!("CARGO_PKG_NAME")));
    assert_eq!(auth, format!("Basic {expected}"));
}

#[tokio::test]
async fn a_refused_exchange_names_the_issuers_error() {
    let idp = MockIdp::start().await;
    idp.state.exchange_fails.store(true, Ordering::SeqCst);
    let dir = temp();
    let m = obo(dir.path(), &idp, |c| c.actor = false);
    let err = Acting::scope(Some(acting("ada", Some("ada-at"))), m.access_token())
        .await
        .err();
    assert!(
        matches!(&err, Some(Error::Delegation { reason, .. })
            if reason.contains("invalid_grant") && reason.contains("subject token expired")),
        "{err:?}"
    );
    assert!(!last_body(&idp).contains("actor_token"));
}

// --- discovery ---------------------------------------------------------------

#[test]
fn rfc_8414_metadata_goes_between_the_host_and_the_issuers_path() {
    let url = |issuer: &str| Endpoints::oauth_metadata_url(issuer).unwrap_or_default();
    assert_eq!(
        url("https://login.example.com"),
        "https://login.example.com/.well-known/oauth-authorization-server"
    );
    assert_eq!(
        url("https://login.example.com/tenant/"),
        "https://login.example.com/.well-known/oauth-authorization-server/tenant"
    );
    assert!(Endpoints::oauth_metadata_url("not a url").is_err());
}

#[test]
fn a_redirects_issuer_must_match_and_is_required_when_promised() {
    let mut endpoints = Endpoints {
        issuer: Some(String::from("https://i")),
        authorization: String::new(),
        token: String::new(),
        device_authorization: None,
        jwks_uri: None,
        grant_types_supported: None,
        authorization_response_iss_parameter_supported: false,
    };
    assert!(endpoints.check_response_issuer(None).is_ok());
    assert!(endpoints.check_response_issuer(Some("https://i")).is_ok());
    assert!(
        endpoints
            .check_response_issuer(Some("https://evil"))
            .is_err()
    );
    endpoints.authorization_response_iss_parameter_supported = true;
    assert!(endpoints.check_response_issuer(None).is_err());
    assert!(endpoints.check_response_issuer(Some("https://i")).is_ok());
}

#[tokio::test]
async fn a_plain_oauth_server_is_found_through_rfc_8414() {
    let idp = MockIdp::start().await;
    idp.state.oauth_only.store(true, Ordering::SeqCst);
    let dir = temp();
    let m = manager(dir.path(), &idp, Grant::ClientCredentials);
    assert!(
        m.access_token()
            .await
            .is_ok_and(|t| t.expose_secret() == "service-access")
    );
}

#[tokio::test]
async fn metadata_for_another_issuer_is_refused_and_the_grant_is_checked() {
    let idp = MockIdp::start().await;
    if let Ok(mut named) = idp.state.named_issuer.lock() {
        *named = Some(String::from("https://elsewhere.example.com"));
    }
    let dir = temp();
    let m = manager(dir.path(), &idp, Grant::ClientCredentials);
    let err = m.access_token().await.err();
    assert!(err.is_some_and(|e| e.to_string().contains("elsewhere.example.com")));
    assert_eq!(
        idp.state.client_credentials_requests.load(Ordering::SeqCst),
        0
    );

    let idp = MockIdp::start().await;
    let m = manager(dir.path(), &idp, Grant::ClientCredentials);
    assert!(matches!(m.issuer_supports_grant().await, Ok(None)));
    let idp = MockIdp::start().await;
    if let Ok(mut grants) = idp.state.grants.lock() {
        *grants = Some(vec![String::from("authorization_code")]);
    }
    let m = manager(dir.path(), &idp, Grant::ClientCredentials);
    assert!(matches!(m.issuer_supports_grant().await, Ok(Some(false))));
    let m = manager(dir.path(), &idp, Grant::AuthorizationCode);
    assert!(matches!(m.issuer_supports_grant().await, Ok(Some(true))));
}
