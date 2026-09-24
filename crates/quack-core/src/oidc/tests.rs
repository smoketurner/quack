//! Sign-in against a mock issuer, and the ID token checks on their own.
#![expect(
    clippy::indexing_slicing,
    reason = "fixtures index JSON objects they built, where a missing key is inserted, not a panic"
)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::*;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

/// What the mock issuer answers with, and what it was sent.
#[derive(Default)]
struct IssuerState {
    /// The `issuer` its discovery document names; the base URL when unset.
    named_issuer: Option<String>,
    /// Claims of the ID token the code exchange returns; none when unset.
    id_claims: Option<Value>,
    /// The `error` a refresh is refused with; a new token when unset.
    refresh_error: Option<&'static str>,
    token_bodies: Vec<String>,
}

struct MockIssuer {
    url: String,
    state: Arc<Mutex<IssuerState>>,
}

impl MockIssuer {
    async fn start() -> Self {
        let Ok(listener) = TcpListener::bind("127.0.0.1:0").await else {
            fail("loopback bind failed");
        };
        let Ok(addr) = listener.local_addr() else {
            fail("no local address");
        };
        let url = format!("http://{addr}");
        let state = Arc::new(Mutex::new(IssuerState::default()));
        let (base, served) = (url.clone(), Arc::clone(&state));
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let (base, served) = (base.clone(), Arc::clone(&served));
                tokio::spawn(async move { serve(stream, &base, &served).await });
            }
        });
        Self { url, state }
    }

    fn with(&self, change: impl FnOnce(&mut IssuerState)) {
        if let Ok(mut state) = self.state.lock() {
            change(&mut state);
        }
    }

    fn last_token_body(&self) -> String {
        self.state
            .lock()
            .ok()
            .and_then(|s| s.token_bodies.last().cloned())
            .unwrap_or_default()
    }

    fn sign_in(&self) -> SignIn {
        let config = OidcConfig {
            issuer_url: self.url.clone(),
            client_id: String::from("quack"),
            client_secret_env: None,
            scopes: OidcConfig::default_scopes(),
            redirect_uri: format!("https://quack.example.com{}", OidcConfig::CALLBACK_PATH),
        };
        SignIn::new(config).unwrap_or_else(|e| fail(&e.to_string()))
    }
}

async fn serve(mut stream: tokio::net::TcpStream, base: &str, state: &Mutex<IssuerState>) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let (head_len, body_len) = loop {
        let Ok(n) = stream.read(&mut chunk).await else {
            return;
        };
        if n == 0 {
            return;
        }
        buf.extend_from_slice(chunk.get(..n).unwrap_or_default());
        if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(buf.get(..end).unwrap_or_default()).to_lowercase();
            let len = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            break (end.saturating_add(4), len);
        }
    };
    while buf.len() < head_len.saturating_add(body_len) {
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(chunk.get(..n).unwrap_or_default()),
        }
    }
    let head = String::from_utf8_lossy(buf.get(..head_len).unwrap_or_default()).to_string();
    let body = String::from_utf8_lossy(buf.get(head_len..).unwrap_or_default()).to_string();
    let target = head.split_whitespace().nth(1).unwrap_or("/").to_owned();
    let (status, json) = answer(&target, &body, base, state);
    let json = json.to_string();
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
        json.len()
    );
    drop(stream.write_all(response.as_bytes()).await);
    drop(stream.shutdown().await);
}

fn answer(
    target: &str,
    body: &str,
    base: &str,
    state: &Mutex<IssuerState>,
) -> (&'static str, Value) {
    let Ok(mut state) = state.lock() else {
        return ("500 Internal Server Error", json!({}));
    };
    match target {
        "/.well-known/openid-configuration" => (
            "200 OK",
            json!({
                "issuer": state.named_issuer.clone().unwrap_or_else(|| base.to_owned()),
                "authorization_endpoint": format!("{base}/authorize"),
                "token_endpoint": format!("{base}/token"),
            }),
        ),
        "/token" => {
            state.token_bodies.push(body.to_owned());
            let form: HashMap<String, String> =
                oauth2::url::form_urlencoded::parse(body.as_bytes())
                    .into_owned()
                    .collect();
            match form.get("grant_type").map(String::as_str) {
                Some("authorization_code") => {
                    let mut token = json!({
                        "access_token": "user-access",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "refresh_token": "user-refresh",
                    });
                    if let Some(claims) = &state.id_claims {
                        token["id_token"] = json!(jwt(claims));
                    }
                    ("200 OK", token)
                }
                Some("refresh_token") => match state.refresh_error {
                    Some(error) => ("400 Bad Request", json!({ "error": error })),
                    None => (
                        "200 OK",
                        json!({ "access_token": "renewed-access", "token_type": "Bearer", "expires_in": 3600 }),
                    ),
                },
                _ => (
                    "400 Bad Request",
                    json!({ "error": "unsupported_grant_type" }),
                ),
            }
        }
        _ => ("404 Not Found", json!({})),
    }
}

/// An unsigned compact JWT carrying `claims`.
fn jwt(claims: &Value) -> String {
    format!(
        "{}.{}.sig",
        URL_SAFE_NO_PAD.encode(br#"{"alg":"RS256"}"#),
        URL_SAFE_NO_PAD.encode(claims.to_string())
    )
}

fn in_an_hour() -> i64 {
    Timestamp::now().as_second().saturating_add(3600)
}

fn query(url: &str) -> HashMap<String, String> {
    Url::parse(url)
        .map(|u| u.query_pairs().into_owned().collect())
        .unwrap_or_default()
}

#[tokio::test]
async fn a_sign_in_sends_pkce_state_and_nonce_and_checks_them_on_return() {
    let issuer = MockIssuer::start().await;
    let sign_in = issuer.sign_in();
    let begun = sign_in.begin().await;
    let Ok((url, pending)) = begun else {
        fail(&format!("{:?}", begun.err()));
    };
    assert!(
        url.starts_with(&format!("{}/authorize?", issuer.url)),
        "{url}"
    );
    let q = query(&url);
    assert_eq!(q.get("response_type").map(String::as_str), Some("code"));
    assert_eq!(q.get("client_id").map(String::as_str), Some("quack"));
    assert_eq!(
        q.get("code_challenge_method").map(String::as_str),
        Some("S256")
    );
    assert_eq!(q.get("state"), Some(&pending.state));
    assert!(
        q.get("scope")
            .is_some_and(|s| s.split(' ').any(|s| s == "openid"))
    );
    assert!(
        q.get("redirect_uri")
            .is_some_and(|r| r.ends_with(OidcConfig::CALLBACK_PATH))
    );
    let nonce = q.get("nonce").cloned().unwrap_or_default();
    assert!(!nonce.is_empty());

    issuer.with(|s| {
        s.id_claims = Some(json!({
            "iss": issuer.url, "sub": "subject-1", "aud": "quack", "exp": in_an_hour(),
            "nonce": nonce, "preferred_username": "ada", "email": "ada@example.com",
        }));
    });
    let signed_in = sign_in.finish("the-code", pending).await;
    let Ok(signed_in) = signed_in else {
        fail(&format!("{:?}", signed_in.err()));
    };
    assert_eq!(signed_in.subject, OidcSubject::from("subject-1"));
    assert_eq!(signed_in.username, "ada");
    assert_eq!(signed_in.token.access_token.expose_secret(), "user-access");
    assert!(signed_in.token.refresh_token.is_some());
    let body = issuer.last_token_body();
    assert!(
        body.contains("code=the-code") && body.contains("code_verifier="),
        "{body}"
    );
}

#[tokio::test]
async fn a_sign_in_with_the_wrong_nonce_or_no_id_token_is_refused() {
    let issuer = MockIssuer::start().await;
    let sign_in = issuer.sign_in();
    let Ok((_, pending)) = sign_in.begin().await else {
        fail("begin failed");
    };
    let err = sign_in.finish("the-code", pending).await.err();
    assert!(
        err.is_some_and(|e| e.to_string().contains("no ID token")),
        "a response without an ID token"
    );

    let Ok((_, pending)) = sign_in.begin().await else {
        fail("begin failed");
    };
    issuer.with(|s| {
        s.id_claims = Some(json!({
            "iss": issuer.url, "sub": "s", "aud": "quack", "exp": in_an_hour(), "nonce": "replayed",
        }));
    });
    let err = sign_in.finish("the-code", pending).await.err();
    assert!(err.is_some_and(|e| matches!(&e, Error::SignIn(m) if m.contains("nonce"))));
}

#[tokio::test]
async fn a_discovery_document_for_another_issuer_is_refused() {
    let issuer = MockIssuer::start().await;
    issuer.with(|s| s.named_issuer = Some(String::from("https://elsewhere.example.com")));
    let err = issuer.sign_in().begin().await.err();
    assert!(err.is_some_and(|e| e.to_string().contains("elsewhere.example.com")));
}

#[tokio::test]
async fn renewal_tells_a_revoked_grant_from_a_broken_client() {
    let issuer = MockIssuer::start().await;
    let sign_in = issuer.sign_in();
    let refresh = SecretString::from(String::from("user-refresh"));

    let renewed = sign_in.renew(&refresh).await;
    assert!(
        matches!(
            &renewed,
            Ok(Renewal::Renewed(token))
                if token.access_token.expose_secret() == "renewed-access"
                    && token.refresh_token.as_ref().map(ExposeSecret::expose_secret) == Some("user-refresh")
        ),
        "{renewed:?}"
    );

    issuer.with(|s| s.refresh_error = Some("invalid_grant"));
    assert!(matches!(
        sign_in.renew(&refresh).await,
        Ok(Renewal::Revoked(_))
    ));

    issuer.with(|s| s.refresh_error = Some("invalid_client"));
    assert!(sign_in.renew(&refresh).await.is_err());
}

fn claims(value: &Value) -> Claims {
    serde_json::from_value(value.clone()).unwrap_or_else(|e| fail(&e.to_string()))
}

#[test]
fn the_id_token_checks_follow_core_3_1_3_7() {
    let now = Timestamp::now();
    let base = json!({ "iss": "https://i", "sub": "s", "aud": "quack", "exp": in_an_hour(), "nonce": "n" });
    let check = |change: &dyn Fn(&mut Value)| {
        let mut value = base.clone();
        change(&mut value);
        claims(&value).check("https://i", "quack", "n", now)
    };
    assert!(check(&|_| {}).is_ok());
    assert!(check(&|v| v["iss"] = json!("https://i/")).is_err());
    assert!(check(&|v| v["aud"] = json!("someone-else")).is_err());
    assert!(check(&|v| v["aud"] = json!(["quack"])).is_ok());
    assert!(check(&|v| v["aud"] = json!(["quack", "api"])).is_err());
    assert!(
        check(&|v| {
            v["aud"] = json!(["quack", "api"]);
            v["azp"] = json!("quack");
        })
        .is_ok()
    );
    assert!(check(&|v| v["exp"] = json!(now.as_second().saturating_sub(120))).is_err());
    assert!(check(&|v| v["exp"] = json!(now.as_second().saturating_sub(30))).is_ok());
    assert!(check(&|v| v["nonce"] = json!("other")).is_err());
    assert!(
        check(&|v| {
            if let Some(map) = v.as_object_mut() {
                map.remove("nonce");
            }
        })
        .is_err()
    );
    assert!(check(&|v| v["sub"] = json!("")).is_err());
}

#[test]
fn the_username_falls_back_from_preferred_username_to_email_to_subject() {
    let mut value = json!({ "iss": "i", "sub": "s-1", "aud": "a", "exp": 0 });
    assert_eq!(claims(&value).username(), "s-1");
    value["email"] = json!("ada@example.com");
    assert_eq!(claims(&value).username(), "ada@example.com");
    value["preferred_username"] = json!("  ");
    assert_eq!(claims(&value).username(), "ada@example.com");
    value["preferred_username"] = json!("ada");
    assert_eq!(claims(&value).username(), "ada");
}

#[test]
fn a_malformed_id_token_is_a_sign_in_error() {
    for token in [
        "not-a-jwt",
        "a.!!!.c",
        &format!("a.{}.c", URL_SAFE_NO_PAD.encode("[1]")),
    ] {
        assert!(
            matches!(Claims::of(token), Err(Error::SignIn(_))),
            "{token}"
        );
    }
}
