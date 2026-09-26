//! Sign-in against a mock issuer, and the ID token checks on their own.
#![expect(
    clippy::indexing_slicing,
    reason = "fixtures index JSON objects they built, where a missing key is inserted, not a panic"
)]

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use aws_lc_rs::rand::SystemRandom;
use aws_lc_rs::signature::{ECDSA_P256_SHA256_FIXED_SIGNING, EcdsaKeyPair, KeyPair};
use jsonwebtoken::{Algorithm, EncodingKey, Header};

use super::*;
use crate::llm::oauth::client_key::{ClientKeyName, PublicJwk, tests::verify};

/// The `aud` the test sign-in accepts on access tokens.
const AUDIENCE: &str = "api://quack";

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
    /// The JWK set `/jwks` serves.
    jwks: Vec<Value>,
    jwks_fetches: usize,
    /// Whether discovery lists `pushed_authorization_request_endpoint`.
    par: bool,
    /// Each pushed authorization request's body.
    par_bodies: Vec<String>,
    /// The public key registered for quack's `private_key_jwt`.
    client_jwk: Option<PublicJwk>,
    /// Every assertion `jti` accepted: the issuer spends them.
    spent: HashSet<String>,
    /// Why each refused client assertion was refused.
    refused: Vec<String>,
}

struct MockIssuer {
    url: String,
    state: Arc<Mutex<IssuerState>>,
    /// The data directory the sign-in's client key is kept in.
    dir: tempfile::TempDir,
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
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        Self { url, state, dir }
    }

    /// The client keys of this issuer's data directory.
    fn keys(&self) -> ClientKeys {
        let mut config = crate::config::Config::default();
        config.general.data_dir = self.dir.path().to_path_buf();
        ClientKeys::new(&config, crate::llm::oauth::KeySource::File)
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
            client_auth: ClientAuth::default(),
            scopes: OidcConfig::default_scopes(),
            redirect_uri: format!("https://quack.example.com{}", OidcConfig::CALLBACK_PATH),
            audience: Some(String::from(AUDIENCE)),
            subject_claim: String::from("sub"),
        };
        SignIn::new(config, self.keys()).unwrap_or_else(|e| fail(&e.to_string()))
    }

    fn jwks_fetches(&self) -> usize {
        self.state
            .lock()
            .map(|s| s.jwks_fetches)
            .unwrap_or_default()
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
    if matches!(target, "/token" | "/par")
        && let Some(refusal) = refuse_assertion(body, base, &mut state)
    {
        return refusal;
    }
    match target {
        "/.well-known/openid-configuration" => {
            let mut metadata = json!({
                "issuer": state.named_issuer.clone().unwrap_or_else(|| base.to_owned()),
                "authorization_endpoint": format!("{base}/authorize"),
                "token_endpoint": format!("{base}/token"),
                "jwks_uri": format!("{base}/jwks"),
            });
            if state.par {
                metadata["pushed_authorization_request_endpoint"] = json!(format!("{base}/par"));
            }
            ("200 OK", metadata)
        }
        "/par" => {
            state.par_bodies.push(body.to_owned());
            (
                "201 Created",
                json!({ "request_uri": "urn:ietf:params:oauth:request_uri:pushed-1", "expires_in": 60 }),
            )
        }
        "/jwks" => {
            state.jwks_fetches = state.jwks_fetches.saturating_add(1);
            ("200 OK", json!({ "keys": state.jwks }))
        }
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

/// A client assertion checked as Vouch checks one; the refusal when it
/// fails, `None` when there is none or it passes.
fn refuse_assertion(
    body: &str,
    base: &str,
    state: &mut IssuerState,
) -> Option<(&'static str, Value)> {
    let form: HashMap<String, String> = oauth2::url::form_urlencoded::parse(body.as_bytes())
        .into_owned()
        .collect();
    let assertion = form.get("client_assertion")?;
    let state = &mut *state;
    let checked = match (&state.client_jwk, form.get("client_id").map(String::as_str)) {
        (Some(jwk), Some("quack")) => verify(jwk, assertion, "quack", base, &mut state.spent),
        (None, _) => Err(String::from("no key registered")),
        (_, other) => Err(format!("client_id {other:?}")),
    };
    let why = checked.err()?;
    state.refused.push(why);
    Some(("401 Unauthorized", json!({ "error": "invalid_client" })))
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
}

#[test]
fn the_subject_is_the_configured_claim_and_the_username_falls_back() {
    let mut value = json!({ "iss": "i", "sub": "s-1", "oid": "o-1", "aud": "a", "exp": 0 });
    let person = |value: &Value| claims(value).person;
    let subject = person(&value).subject("sub");
    assert_eq!(subject, Some(OidcSubject::from("s-1")));
    assert_eq!(
        person(&value).subject("oid"),
        Some(OidcSubject::from("o-1"))
    );
    assert_eq!(person(&value).subject("tid"), None);
    let s1 = OidcSubject::from("s-1");
    assert_eq!(person(&value).username(&s1), "s-1");
    value["email"] = json!("ada@example.com");
    assert_eq!(person(&value).username(&s1), "ada@example.com");
    value["preferred_username"] = json!("  ");
    assert_eq!(person(&value).username(&s1), "ada@example.com");
    value["preferred_username"] = json!("ada");
    assert_eq!(person(&value).username(&s1), "ada");
    value["sub"] = json!(" ");
    assert_eq!(person(&value).subject("sub"), None);
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

/// A P-256 signing key and its public JWK, as an issuer publishes it.
struct TestKey {
    kid: String,
    encoding: EncodingKey,
    jwk: Value,
}

impl TestKey {
    fn new(kid: &str) -> Self {
        let pkcs8 =
            EcdsaKeyPair::generate_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, &SystemRandom::new())
                .unwrap_or_else(|e| fail(&e.to_string()));
        let pair = EcdsaKeyPair::from_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, pkcs8.as_ref())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let point = pair.public_key().as_ref();
        let (Some(x), Some(y)) = (point.get(1..33), point.get(33..65)) else {
            fail("not an uncompressed P-256 point");
        };
        Self {
            kid: kid.to_owned(),
            encoding: EncodingKey::from_ec_der(pkcs8.as_ref()),
            jwk: json!({
                "kty": "EC", "crv": "P-256", "use": "sig", "alg": "ES256", "kid": kid,
                "x": URL_SAFE_NO_PAD.encode(x), "y": URL_SAFE_NO_PAD.encode(y),
            }),
        }
    }

    fn sign(&self, claims: &Value) -> String {
        let mut header = Header::new(Algorithm::ES256);
        header.kid = Some(self.kid.clone());
        jsonwebtoken::encode(&header, claims, &self.encoding)
            .unwrap_or_else(|e| fail(&e.to_string()))
    }
}

/// Claims of a valid access token for `issuer`, changed by `change`.
fn access_claims(issuer: &str, change: impl FnOnce(&mut Value)) -> Value {
    let mut claims = json!({
        "iss": issuer, "aud": AUDIENCE, "sub": "person-1", "oid": "object-1",
        "exp": in_an_hour(), "scp": "quack.use", "preferred_username": "ada",
    });
    change(&mut claims);
    claims
}

#[tokio::test]
async fn an_access_token_signed_by_the_issuer_names_the_person() {
    let issuer = MockIssuer::start().await;
    let key = TestKey::new("k1");
    issuer.with(|s| s.jwks = vec![key.jwk.clone()]);
    let sign_in = issuer.sign_in();

    let bearer = sign_in
        .verify_bearer(&key.sign(&access_claims(&issuer.url, |_| {})))
        .await;
    assert!(
        bearer.is_ok_and(|b| b.subject == OidcSubject::from("person-1")
            && b.username == "ada"
            && b.expires_at > Timestamp::now())
    );
    // A second token uses the cached keys; Okta's array `scp` counts too.
    let array_scope = access_claims(&issuer.url, |c| c["scp"] = json!(["quack.use"]));
    assert!(sign_in.verify_bearer(&key.sign(&array_scope)).await.is_ok());
    assert_eq!(issuer.jwks_fetches(), 1);
}

#[tokio::test]
async fn tokens_that_fail_a_check_are_refused_as_bearer_errors() {
    let issuer = MockIssuer::start().await;
    let key = TestKey::new("k1");
    issuer.with(|s| s.jwks = vec![key.jwk.clone()]);
    let sign_in = issuer.sign_in();
    let url = issuer.url.clone();
    let refused = |token: String, why: &'static str| {
        let sign_in = &sign_in;
        async move {
            let outcome = sign_in.verify_bearer(&token).await;
            assert!(
                matches!(outcome, Err(Error::Bearer(_))),
                "{why}: {outcome:?}"
            );
        }
    };

    refused(
        key.sign(&access_claims(&url, |c| c["aud"] = json!("api://other"))),
        "audience",
    )
    .await;
    refused(
        key.sign(&access_claims(&url, |c| c["iss"] = json!("https://evil"))),
        "issuer",
    )
    .await;
    refused(
        key.sign(&access_claims(&url, |c| {
            c["exp"] = json!(Timestamp::now().as_second().saturating_sub(7200));
        })),
        "expired",
    )
    .await;
    refused(
        key.sign(&access_claims(&url, |c| {
            if let Some(map) = c.as_object_mut() {
                map.remove("scp");
            }
        })),
        "an ID token has no scope",
    )
    .await;
    refused(
        key.sign(&access_claims(&url, |c| c["sub"] = json!(""))),
        "no subject",
    )
    .await;
    let hmac = jsonwebtoken::encode(
        &Header::new(Algorithm::HS256),
        &access_claims(&url, |_| {}),
        &EncodingKey::from_secret(b"shared"),
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    refused(hmac, "shared-secret algorithm").await;
    let stranger = TestKey::new("k1");
    refused(
        stranger.sign(&access_claims(&url, |_| {})),
        "same kid, other key",
    )
    .await;
    refused(String::from("not.a.jwt"), "garbage").await;
}

#[tokio::test]
async fn an_unknown_key_is_fetched_again_but_at_most_once_a_minute() {
    let issuer = MockIssuer::start().await;
    let (old, new) = (TestKey::new("old"), TestKey::new("new"));
    issuer.with(|s| s.jwks = vec![old.jwk.clone()]);
    let sign_in = issuer.sign_in();
    let claims = access_claims(&issuer.url, |_| {});
    assert!(sign_in.verify_bearer(&old.sign(&claims)).await.is_ok());

    // The issuer rotates; right after a fetch, an unknown kid is refused
    // without asking again.
    issuer.with(|s| s.jwks = vec![old.jwk.clone(), new.jwk.clone()]);
    assert!(sign_in.verify_bearer(&new.sign(&claims)).await.is_err());
    assert_eq!(issuer.jwks_fetches(), 1);

    // A minute later, the unknown kid sends quack back for the keys.
    if let Some(keys) = sign_in.keys.write().await.as_mut() {
        keys.fetched = Instant::now()
            .checked_sub(KEYS_REFETCH)
            .unwrap_or_else(Instant::now);
    }
    assert!(sign_in.verify_bearer(&new.sign(&claims)).await.is_ok());
    assert_eq!(issuer.jwks_fetches(), 2);
}

#[tokio::test]
async fn without_an_audience_no_bearer_is_accepted_and_oid_can_name_the_person() {
    let issuer = MockIssuer::start().await;
    let key = TestKey::new("k1");
    issuer.with(|s| s.jwks = vec![key.jwk.clone()]);
    let token = key.sign(&access_claims(&issuer.url, |_| {}));

    let mut config = issuer.sign_in().config;
    config.audience = None;
    let closed =
        SignIn::new(config.clone(), issuer.keys()).unwrap_or_else(|e| fail(&e.to_string()));
    assert!(matches!(
        closed.verify_bearer(&token).await,
        Err(Error::Bearer(_))
    ));
    assert_eq!(issuer.jwks_fetches(), 0);

    config.audience = Some(String::from(AUDIENCE));
    config.subject_claim = String::from("oid");
    let entra = SignIn::new(config, issuer.keys()).unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        entra
            .verify_bearer(&token)
            .await
            .is_ok_and(|b| b.subject == OidcSubject::from("object-1"))
    );
}

// --- private_key_jwt and pushed authorization requests -------------------------

impl MockIssuer {
    /// A sign-in that authenticates with `private_key_jwt`, its key
    /// registered at this issuer.
    async fn sign_in_with_key(&self) -> SignIn {
        let Ok(key) = self
            .keys()
            .key(&ClientKeyName::new(&self.url, "quack"))
            .await
        else {
            fail("no client key");
        };
        self.with(|s| s.client_jwk = Some(key.jwk().clone()));
        let mut config = self.sign_in().config;
        config.client_auth = ClientAuth::PrivateKeyJwt;
        SignIn::new(config, self.keys()).unwrap_or_else(|e| fail(&e.to_string()))
    }

    fn par_bodies(&self) -> Vec<HashMap<String, String>> {
        self.state
            .lock()
            .map(|s| {
                s.par_bodies
                    .iter()
                    .map(|b| {
                        oauth2::url::form_urlencoded::parse(b.as_bytes())
                            .into_owned()
                            .collect()
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn spent_and_refused(&self) -> (usize, Vec<String>) {
        self.state
            .lock()
            .map(|s| (s.spent.len(), s.refused.clone()))
            .unwrap_or_default()
    }
}

#[tokio::test]
async fn an_advertised_par_endpoint_takes_the_request_and_the_browser_only_a_reference() {
    let issuer = MockIssuer::start().await;
    issuer.with(|s| s.par = true);
    let sign_in = issuer.sign_in_with_key().await;
    let begun = sign_in.begin().await;
    let Ok((url, pending)) = begun else {
        fail(&format!("{:?}", begun.err()));
    };
    assert!(
        url.starts_with(&format!("{}/authorize?", issuer.url)),
        "{url}"
    );
    let q = query(&url);
    assert_eq!(q.len(), 2, "{url}");
    assert_eq!(q.get("client_id").map(String::as_str), Some("quack"));
    assert_eq!(
        q.get("request_uri").map(String::as_str),
        Some("urn:ietf:params:oauth:request_uri:pushed-1")
    );

    let pushed = issuer.par_bodies();
    let [pushed] = pushed.as_slice() else {
        fail(&format!("{pushed:?}"));
    };
    let field = |k: &str| pushed.get(k).cloned().unwrap_or_default();
    assert_eq!(field("response_type"), "code");
    assert_eq!(field("client_id"), "quack");
    assert_eq!(field("state"), pending.state);
    assert_eq!(field("nonce"), pending.nonce);
    assert_eq!(field("code_challenge_method"), "S256");
    assert_eq!(field("code_challenge").len(), 43);
    assert!(field("redirect_uri").ends_with(OidcConfig::CALLBACK_PATH));
    assert!(field("scope").split(' ').any(|s| s == "openid"));
    assert_eq!(
        field("client_assertion_type"),
        "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
    );
    assert!(!pushed.contains_key("client_secret"));

    let nonce = pending.nonce.clone();
    issuer.with(|s| {
        s.id_claims = Some(json!({
            "iss": issuer.url, "sub": "subject-1", "aud": "quack", "exp": in_an_hour(),
            "nonce": nonce,
        }));
    });
    let signed_in = sign_in.finish("the-code", pending).await;
    assert!(signed_in.is_ok(), "{:?}", signed_in.err());
    let body = issuer.last_token_body();
    assert!(body.contains("client_assertion="), "{body}");
    assert!(body.contains("client_id=quack"), "{body}");
    assert!(!body.contains("client_secret"), "{body}");
    let renewed = sign_in
        .renew(&SecretString::from(String::from("user-refresh")))
        .await;
    assert!(matches!(renewed, Ok(Renewal::Renewed(_))), "{renewed:?}");
    // Three requests, three assertions, none refused and none reused.
    let (spent, refused) = issuer.spent_and_refused();
    assert_eq!(spent, 3);
    assert!(refused.is_empty(), "{refused:?}");
}

#[tokio::test]
async fn a_confidential_client_pushes_with_its_secret_and_without_par_nothing_is_pushed() {
    let issuer = MockIssuer::start().await;
    issuer.with(|s| s.par = true);
    let mut config = issuer.sign_in().config;
    config.client_secret_env = Some(String::from("CARGO_PKG_NAME"));
    let sign_in = SignIn::new(config, issuer.keys()).unwrap_or_else(|e| fail(&e.to_string()));
    assert!(sign_in.begin().await.is_ok());
    let pushed = issuer.par_bodies();
    assert!(
        matches!(pushed.as_slice(), [p] if p.get("client_secret").map(String::as_str) == Some(env!("CARGO_PKG_NAME"))
            && !p.contains_key("client_assertion")),
        "{pushed:?}"
    );

    // Not advertised: the browser carries the request, as before.
    let plain = MockIssuer::start().await;
    let Ok((url, _)) = plain.sign_in_with_key().await.begin().await else {
        fail("begin failed");
    };
    let q = query(&url);
    assert!(
        q.contains_key("code_challenge") && q.contains_key("nonce"),
        "{url}"
    );
    assert!(!q.contains_key("request_uri"));
    assert!(plain.par_bodies().is_empty());
}

#[tokio::test]
async fn the_sign_in_and_a_provider_registered_as_the_same_client_share_one_key() {
    use crate::config::{Exchange, Grant, OAuthConfig};
    use crate::llm::oauth::{KeySource, TokenManager};
    let issuer = MockIssuer::start().await;
    let sign_in = issuer.sign_in_with_key().await;
    let mut config = crate::config::Config::default();
    config.general.data_dir = issuer.dir.path().to_path_buf();
    let oauth = OAuthConfig {
        // The same client, written with a trailing slash.
        issuer_url: format!("{}/", issuer.url),
        client_id: String::from("quack"),
        scopes: Vec::new(),
        redirect_uri: String::from(OAuthConfig::DEFAULT_REDIRECT_URI),
        grant: Grant::OnBehalfOf,
        client_secret_env: None,
        client_auth: ClientAuth::PrivateKeyJwt,
        exchange: Exchange::TokenExchange,
        audience: None,
        resource: None,
        actor: false,
    };
    let name = "gw".parse().unwrap_or_else(|e: Error| fail(&e.to_string()));
    let manager = TokenManager::new(&config, &name, oauth, KeySource::File)
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (ours, theirs) = (sign_in.credential().await, manager.credential().await);
    let (Ok(ours), Ok(theirs)) = (ours, theirs) else {
        fail("no credential");
    };
    let (Some(ours), Some(theirs)) = (ours.assertion(), theirs.assertion()) else {
        fail("not assertions");
    };
    assert_eq!(ours.thumbprint(), theirs.thumbprint());
    // Another process finds the one stored row.
    let stored = issuer
        .keys()
        .load_or_create(&ClientKeyName::new(&issuer.url, "quack"))
        .await;
    assert!(stored.is_ok_and(|k| k.thumbprint() == ours.thumbprint()));
}
