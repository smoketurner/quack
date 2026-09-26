//! `quack auth jwks`, `--rotate`, `--activate`, and the key state `quack
//! auth status` shows.

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

/// A client registered by hand rotates in two steps: `--rotate` prints the
/// key in use beside its replacement, again the same pair if repeated, and
/// only `--activate` puts the replacement in use.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn a_client_rotates_in_two_steps_without_losing_its_key() {
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
    // Still signing with the old key; a repeat shows the same pair, and
    // `quack auth status` names the replacement waiting.
    assert_eq!(
        client_jwks(&config, Some("gw"), KeySource::File)
            .await
            .unwrap(),
        current
    );
    let (_, again, _) = staged(false).await;
    assert_eq!(again, Some(both.clone()));
    let state = client_key_state(&config, Some("gw"), KeySource::File)
        .await
        .unwrap()
        .unwrap();
    let next = &both.keys.last().unwrap().kid;
    assert!(
        state.contains(&format!("replacement key {next} waits")),
        "{state}"
    );
    assert!(state.contains("--rotate --activate gw"), "{state}");

    let (done, active, note) = staged(true).await;
    assert!(done.is_ok(), "{done:?}");
    let active = active.unwrap();
    assert_eq!(Some(active.keys.as_slice()), both.keys.get(1..));
    assert!(note.contains("Restart a running `quack serve`"), "{note}");
    assert_eq!(
        client_jwks(&config, Some("gw"), KeySource::File)
            .await
            .unwrap(),
        active
    );
    assert_eq!(
        client_key_state(&config, Some("gw"), KeySource::File)
            .await
            .unwrap(),
        Some(format!("client key {next}"))
    );
    // Nothing waits any more.
    let (done, _, _) = staged(true).await;
    assert!(done.is_err_and(|e| e.to_string().contains("no replacement key")));
}

/// Without a provider, `--rotate` targets the `[server.oidc]` client, and a
/// client that signs with no key has nothing to rotate.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn rotate_without_a_provider_is_the_sign_in_client() {
    let dir = tempfile::tempdir().unwrap();
    let toml = "[server.oidc]\nissuer_url = \"https://idp.example.com\"\nclient_id = \"quack\"\n\
                client_auth = \"private_key_jwt\"\nredirect_uri = \"https://q.example.com/auth/oidc/callback\"\n\
                [providers.plain]\ntype = \"openai\"\nbase_url = \"https://p.example.com/v1\"\nauth = \"oauth\"\n\
                [providers.plain.oauth]\nissuer_url = \"https://idp.example.com\"\nclient_id = \"other\"\n";
    let mut config = Config::parse(toml).unwrap();
    config.general.data_dir = dir.path().to_path_buf();
    let (mut out, mut note) = (Vec::new(), Vec::new());
    rotate(&mut out, &mut note, &config, None, false, KeySource::File)
        .await
        .unwrap();
    let note = String::from_utf8(note).unwrap();
    assert!(note.contains("no key in use yet"), "{note}");
    assert!(
        note.contains("`quack auth jwks --rotate --activate`"),
        "{note}"
    );
    let keys = ClientKeys::new(&config, KeySource::File);
    let name = ClientKeyName::new("https://idp.example.com", "quack");
    assert!(keys.waiting_replacement(&name).await.unwrap().is_some());
    let err = rotate(
        &mut Vec::new(),
        &mut Vec::new(),
        &config,
        Some("plain"),
        false,
        KeySource::File,
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(err.contains("client_secret_post"), "{err}");
}
