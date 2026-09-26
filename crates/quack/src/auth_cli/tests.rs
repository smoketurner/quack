//! `quack auth jwks` and the key state `quack auth status` shows.

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
