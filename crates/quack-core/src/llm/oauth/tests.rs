//! Token manager and login flows against a mock identity provider.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::*;

/// What the mock issuer does and what it saw.
#[derive(Default)]
struct MockState {
    refresh_requests: AtomicUsize,
    refresh_fails: std::sync::atomic::AtomicBool,
    /// Device polls answered `authorization_pending` before success.
    pending_polls: AtomicUsize,
    code_exchanges: AtomicUsize,
    last_token_body: StdMutex<String>,
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

fn route(target: &str, body: &str, base: &str, state: &MockState) -> (&'static str, String) {
    match target {
        "/.well-known/openid-configuration" => (
            "200 OK",
            format!(
                "{{\"issuer\":\"{base}\",\"authorization_endpoint\":\"{base}/authorize\",\"token_endpoint\":\"{base}/token\",\"device_authorization_endpoint\":\"{base}/device\"}}"
            ),
        ),
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

fn oauth_config(issuer: &str, device_code: bool) -> OAuthConfig {
    OAuthConfig {
        issuer_url: issuer.to_owned(),
        client_id: String::from("client-1"),
        scopes: vec![
            String::from("api://x/.default"),
            String::from("offline_access"),
        ],
        redirect_uri: format!("http://127.0.0.1:{}/callback", free_port()),
        device_code,
        client_secret_env: None,
    }
}

fn temp() -> tempfile::TempDir {
    tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()))
}

/// Fail the test with a message; `!` lets it sit in a `let ... else`.
#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

fn manager(dir: &Path, idp: &MockIdp, device_code: bool) -> TokenManager {
    match TokenManager::new(
        dir,
        "p",
        oauth_config(&idp.issuer, device_code),
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
    let m = manager(dir.path(), &idp, false);
    assert!(
        m.cache
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
    let m = Arc::new(manager(dir.path(), &idp, false));
    assert!(
        m.cache
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
    let stored = m.cache.load().await;
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
    let token = cached_from_response(&response);
    assert!(token.refresh_token.is_none());
    assert!(token.is_fresh(Timestamp::now(), REUSE_MARGIN));
}

#[tokio::test]
async fn missing_cache_and_failed_refresh_both_require_a_login() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let m = manager(dir.path(), &idp, false);
    let err = m.access_token().await.err();
    assert!(
        err.as_ref()
            .is_some_and(|e| matches!(e, Error::AuthRequired { .. }))
    );
    assert!(err.is_some_and(|e| e.to_string().contains("quack auth login p")));

    assert!(
        m.cache
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
        m.cache
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
    let m = manager(dir.path(), &idp, true);
    let prompts = StdMutex::new(Vec::new());
    let notify = |p: LoginPrompt| {
        if let Ok(mut v) = prompts.lock() {
            v.push(p);
        }
    };
    let token = m.login(LoginOptions::default(), &notify).await;
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
        status
            .is_ok_and(|s| s.logged_in && !s.has_refresh_token && s.key_source == KeySource::File)
    );
    assert!(
        m.access_token()
            .await
            .is_ok_and(|t| t.expose_secret() == "device-access")
    );
    assert!(m.logout().await.is_ok());
    assert!(m.status().await.is_ok_and(|s| !s.logged_in));
    assert!(m.access_token().await.is_err());
}

#[tokio::test]
async fn browser_login_rejects_bad_state_then_accepts_the_code() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let m = Arc::new(manager(dir.path(), &idp, false));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let login = {
        let m = Arc::clone(&m);
        tokio::spawn(async move {
            let notify = move |p: LoginPrompt| drop(tx.send(p));
            m.login(LoginOptions::default(), &notify).await
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
    let m = Arc::new(manager(dir.path(), &idp, false));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let login = {
        let m = Arc::clone(&m);
        tokio::spawn(async move {
            let notify = move |p: LoginPrompt| drop(tx.send(p));
            m.login(LoginOptions::default(), &notify).await
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
    assert!(!m.cache.exists());
}

#[tokio::test]
async fn device_login_without_a_device_endpoint_is_an_error() {
    let dir = temp();
    let config = oauth_config("http://127.0.0.1:9", true);
    let Ok(m) = TokenManager::new(dir.path(), "p", config, KeySource::File) else {
        fail("manager build failed");
    };
    let endpoints = Endpoints {
        authorization: String::from("http://127.0.0.1:9/a"),
        token: String::from("http://127.0.0.1:9/t"),
        device_authorization: None,
    };
    assert!(m.endpoints.set(endpoints).is_ok());
    let err = m.login(LoginOptions::default(), &|_| {}).await.err();
    assert!(err.is_some_and(|e| e.to_string().contains("no device_authorization_endpoint")));
}

#[tokio::test]
async fn discovery_failure_is_reported_with_the_url() {
    let dir = temp();
    let config = oauth_config("http://127.0.0.1:9", false);
    let Ok(m) = TokenManager::new(dir.path(), "p", config, KeySource::File) else {
        fail("manager build failed");
    };
    assert!(
        m.cache
            .store(&seed(SignedDuration::from_secs(-5), Some("r")))
            .await
            .is_ok()
    );
    let err = m.access_token().await.err();
    assert!(err.is_some_and(|e| e.to_string().contains("openid-configuration")));
}

#[test]
fn shared_manager_is_one_per_provider_and_needs_the_oauth_section() {
    let dir = temp();
    let provider = ProviderConfig {
        provider_type: crate::config::ProviderType::Openai,
        auth: crate::config::AuthMode::Oauth,
        base_url: None,
        api_key_env: None,
        embedding_dimension: None,
        max_concurrent_requests: None,
        oauth: Some(oauth_config("http://127.0.0.1:9", false)),
    };
    let a = shared_manager(dir.path(), "shared", &provider);
    let b = shared_manager(dir.path(), "shared", &provider);
    assert!(matches!((&a, &b), (Ok(a), Ok(b)) if Arc::ptr_eq(a, b)));
    let none = ProviderConfig {
        oauth: None,
        ..provider
    };
    assert!(shared_manager(dir.path(), "other", &none).is_err());
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
