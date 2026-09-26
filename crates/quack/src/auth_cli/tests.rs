//! `quack auth jwks`, `register`, `unregister`, and `--rotate`, offline:
//! the parts that need no issuer. Registration against an issuer is tested
//! in `quack_core::llm::oauth::registration`, against the mock issuer.

use quack_core::llm::oauth::client_key::PublicJwk;

use super::*;

/// `quack auth jwks` for a provider and for `[server.oidc]` registered as
/// the same Vouch client: one key, printed as a set an issuer reads.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
#[expect(clippy::indexing_slicing, reason = "test: JSON indexing yields Null")]
async fn auth_jwks_prints_the_shared_client_key_as_a_jwk_set() {
    let dir = tempfile::tempdir().unwrap();
    let toml = "[server.oidc]\nissuer_url = \"https://us.vouch.sh\"\nclient_id = \"quack\"\n\
                client_auth = \"private_key_jwt\"\nredirect_uri = \"https://q.example.com/auth/oidc/callback\"\n\
                [providers.gw]\ntype = \"openai\"\nbase_url = \"https://gw.example.com/v1\"\nauth = \"oauth\"\n\
                [providers.gw.oauth]\nissuer_url = \"https://us.vouch.sh/\"\nclient_id = \"quack\"\n\
                client_auth = \"private_key_jwt\"\ngrant = \"on-behalf-of\"\nactor = false\n\
                [providers.plain]\ntype = \"openai\"\nbase_url = \"https://p.example.com/v1\"\nauth = \"oauth\"\n\
                [providers.plain.oauth]\nissuer_url = \"https://us.vouch.sh\"\nclient_id = \"other\"\n";
    let mut config = Config::parse(toml).unwrap();
    config.general.data_dir = dir.path().to_path_buf();

    assert!(
        client_key_state(&config, Some("gw"), KeySource::File)
            .await
            .unwrap()
            .is_some_and(|s| s.contains("quack auth jwks gw"))
    );
    let printed =
        serde_json::to_string_pretty(&client_jwks(&config, None, KeySource::File).await.unwrap())
            .unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&printed).unwrap();
    let keys = parsed["keys"].as_array().unwrap();
    assert_eq!(keys.len(), 1);
    let jwk: PublicJwk = serde_json::from_value(keys[0].clone()).unwrap();
    assert_eq!(
        (
            jwk.kty.as_str(),
            jwk.crv.as_str(),
            jwk.use_.as_str(),
            jwk.alg.as_str()
        ),
        ("EC", "P-256", "sig", "ES256")
    );
    assert_eq!(keys[0]["use"], "sig");
    // The stored key is the printed one, and the provider shares it.
    let stored = ClientKeys::new(&config, KeySource::File)
        .existing(&ClientKeyName::new("https://us.vouch.sh", "quack"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.thumbprint(), jwk.kid);
    assert_eq!(stored.jwk(), &jwk);
    let provider = client_jwks(&config, Some("gw"), KeySource::File)
        .await
        .unwrap();
    assert_eq!(provider.keys, vec![jwk.clone()]);
    assert_eq!(
        client_key_state(&config, Some("gw"), KeySource::File)
            .await
            .unwrap(),
        Some(format!("client key {}", jwk.kid))
    );
    // A client without private_key_jwt has no key to print.
    let err = client_jwks(&config, Some("plain"), KeySource::File)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("client_secret_post"), "{err}");
    assert!(
        client_key_state(&config, Some("plain"), KeySource::File)
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        client_jwks(&config, Some("nope"), KeySource::File)
            .await
            .is_err()
    );
}

/// Sign-in and an on-behalf-of provider at one issuer, neither naming a
/// `client_id`: what `quack auth register` registers one client for.
#[expect(clippy::unwrap_used, reason = "test")]
fn registered_config(dir: &std::path::Path) -> Config {
    let toml = "[server.oidc]\nissuer_url = \"https://us.vouch.sh\"\nclient_auth = \"private_key_jwt\"\n\
                redirect_uri = \"https://q.example.com/auth/oidc/callback\"\nscopes = [\"openid\", \"email\"]\n\
                [providers.gw]\ntype = \"openai\"\nbase_url = \"https://gw.example.com/v1\"\nauth = \"oauth\"\n\
                [providers.gw.oauth]\nissuer_url = \"https://us.vouch.sh\"\nclient_auth = \"private_key_jwt\"\n\
                grant = \"on-behalf-of\"\nactor = false\n";
    let mut config = Config::parse(toml).unwrap();
    config.general.data_dir = dir.to_path_buf();
    config
}

fn request(print: bool) -> RegisterArgs {
    RegisterArgs {
        issuer: None,
        token_env: None,
        client_name: String::from("quack"),
        replace: false,
        print,
        yes: false,
    }
}

/// `--print` shows the request the manual path sends, with the key `quack
/// auth jwks` prints for the unregistered client, and sends nothing.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
#[expect(clippy::indexing_slicing, reason = "test: JSON indexing yields Null")]
async fn register_print_shows_the_request_with_the_pending_key() {
    let dir = tempfile::tempdir().unwrap();
    let config = registered_config(dir.path());
    let mut out = Vec::new();
    register(
        &mut out,
        &config,
        request(true),
        KeySource::File,
        Confirm::Ask,
    )
    .await
    .unwrap();
    let printed: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(
        printed["grant_types"],
        serde_json::json!([
            "authorization_code",
            "urn:ietf:params:oauth:grant-type:token-exchange"
        ])
    );
    assert_eq!(printed["token_endpoint_auth_method"], "private_key_jwt");
    assert!(printed.get("dpop_bound_access_tokens").is_none());
    let jwks = client_jwks(&config, Some("gw"), KeySource::File)
        .await
        .unwrap();
    assert_eq!(printed["jwks"], serde_json::to_value(&jwks).unwrap());
    assert!(
        client_key_state(&config, None, KeySource::File)
            .await
            .unwrap()
            .is_some_and(|s| s.contains("quack auth register"))
    );
}

/// An open registration is warned about and needs a yes: with no terminal
/// to answer on and no `--yes`, nothing is sent.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn an_open_registration_is_refused_without_a_yes() {
    let dir = tempfile::tempdir().unwrap();
    let config = registered_config(dir.path());
    let mut out = Vec::new();
    let refused = register(
        &mut out,
        &config,
        request(false),
        KeySource::File,
        Confirm::Ask,
    )
    .await;
    let said = String::from_utf8(out).unwrap();
    assert!(said.contains("open registration"), "{said}");
    assert!(
        refused.is_err_and(|e| e.to_string().contains("nothing registered")),
        "{said}"
    );
    // A token variable that is not set is an error before anything is sent.
    let mut unset = request(false);
    unset.token_env = Some(String::from("QUACK_TEST_UNSET_TOKEN_VARIABLE"));
    let err = register(
        &mut Vec::new(),
        &config,
        unset,
        KeySource::File,
        Confirm::Assume,
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(err.contains("QUACK_TEST_UNSET_TOKEN_VARIABLE"), "{err}");
}

/// A client registered by hand rotates in two steps: `--rotate` prints the
/// key in use beside its replacement, again the same pair if repeated, and
/// only `--activate` puts the replacement in use.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn a_hand_registered_client_rotates_in_two_steps_without_losing_its_key() {
    let dir = tempfile::tempdir().unwrap();
    let toml = "[providers.gw]\ntype = \"openai\"\nbase_url = \"https://gw.example.com/v1\"\nauth = \"oauth\"\n\
                [providers.gw.oauth]\nissuer_url = \"https://idp.example.com\"\nclient_id = \"quack\"\n\
                client_auth = \"private_key_jwt\"\ngrant = \"client-credentials\"\n";
    let mut config = Config::parse(toml).unwrap();
    config.general.data_dir = dir.path().to_path_buf();
    let current = client_jwks(&config, Some("gw"), KeySource::File)
        .await
        .unwrap();

    let staged = |activate: bool| {
        let config = config.clone();
        async move {
            let (mut out, mut note) = (Vec::new(), Vec::new());
            let done = rotate(
                &mut out,
                &mut note,
                &config,
                Some("gw"),
                activate,
                KeySource::File,
            )
            .await;
            (
                done,
                serde_json::from_slice::<PublicJwks>(&out).ok(),
                String::from_utf8(note).unwrap_or_default(),
            )
        }
    };
    let (done, both, note) = staged(false).await;
    assert!(done.is_ok(), "{done:?}");
    let both = both.unwrap();
    assert_eq!(both.keys.len(), 2);
    assert_eq!(both.keys.first(), current.keys.first());
    assert!(note.contains("--rotate --activate gw"), "{note}");
    // Still signing with the old key; a repeat shows the same pair.
    assert_eq!(
        client_jwks(&config, Some("gw"), KeySource::File)
            .await
            .unwrap(),
        current
    );
    let (_, again, _) = staged(false).await;
    assert_eq!(again, Some(both.clone()));

    let (done, active, note) = staged(true).await;
    assert!(done.is_ok(), "{done:?}");
    let active = active.unwrap();
    assert_eq!(Some(active.keys.as_slice()), both.keys.get(1..));
    assert!(note.contains("Remove the old key"), "{note}");
    assert_eq!(
        client_jwks(&config, Some("gw"), KeySource::File)
            .await
            .unwrap(),
        active
    );
    // Nothing waits any more.
    let (done, _, _) = staged(true).await;
    assert!(done.is_err_and(|e| e.to_string().contains("no replacement key")));
}

/// A client without a `client_id` and without a registration has nothing to
/// rotate or unregister.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn an_unregistered_client_has_nothing_to_rotate_or_delete() {
    let dir = tempfile::tempdir().unwrap();
    let config = registered_config(dir.path());
    let err = rotate(
        &mut Vec::new(),
        &mut Vec::new(),
        &config,
        Some("gw"),
        false,
        KeySource::File,
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(err.contains("quack auth register"), "{err}");
    let err = unregister(
        &mut Vec::new(),
        &config,
        None,
        KeySource::File,
        Confirm::Assume,
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(err.contains("no client is registered"), "{err}");
}

#[test]
fn vouch_is_recognized_by_its_host() {
    assert!(is_vouch(&RegistrationName::new("https://us.vouch.sh")));
    assert!(is_vouch(&RegistrationName::new("https://vouch.sh/")));
    assert!(!is_vouch(&RegistrationName::new("https://notvouch.sh")));
    assert!(!is_vouch(&RegistrationName::new(
        "https://login.example.com/vouch.sh"
    )));
}
