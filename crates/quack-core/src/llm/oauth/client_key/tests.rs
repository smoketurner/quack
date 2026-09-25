//! The client key: its JWK and thumbprint, its assertions, and how it is
//! kept. [`verify`] is what the mock issuers in the other tests check an
//! assertion with.

use std::collections::HashSet;
use std::path::Path;

use jsonwebtoken::{Algorithm, DecodingKey, Validation};

use super::*;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

/// The claims a verified assertion carries.
#[derive(Debug, Deserialize)]
struct Claims {
    iss: String,
    sub: String,
    jti: String,
    iat: i64,
    exp: i64,
}

/// Check a client assertion as an issuer does: signed with ES256 by the
/// registered `jwk`, `iss` and `sub` both `client_id`, `aud` exactly
/// `audience`, unexpired and valid for at most a minute, with a `jti` not in
/// `spent`, which it is then added to.
///
/// # Errors
///
/// Returns why the issuer would refuse it.
pub(crate) fn verify(
    jwk: &PublicJwk,
    assertion: &str,
    client_id: &str,
    audience: &str,
    spent: &mut HashSet<String>,
) -> std::result::Result<(), String> {
    let header = jsonwebtoken::decode_header(assertion).map_err(|e| e.to_string())?;
    if header.alg != Algorithm::ES256 || header.kid.as_deref() != Some(jwk.kid.as_str()) {
        return Err(format!("unexpected header {header:?}"));
    }
    if header.typ.as_deref() != Some("JWT") {
        return Err(format!("typ is {:?}", header.typ));
    }
    let registered: jsonwebtoken::jwk::Jwk =
        serde_json::from_value(serde_json::to_value(jwk).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let key = DecodingKey::from_jwk(&registered).map_err(|e| e.to_string())?;
    let mut validation = Validation::new(Algorithm::ES256);
    validation.set_issuer(&[client_id]);
    validation.set_audience(&[audience]);
    validation.sub = Some(client_id.to_owned());
    validation.set_required_spec_claims(&["exp", "iss", "aud", "sub"]);
    validation.leeway = 0;
    let claims = jsonwebtoken::decode::<Claims>(assertion, &key, &validation)
        .map_err(|e| e.to_string())?
        .claims;
    if claims.iss != client_id || claims.sub != client_id {
        return Err(String::from("iss and sub must be the client id"));
    }
    let lifetime = claims.exp.saturating_sub(claims.iat);
    if !(1..=60).contains(&lifetime) {
        return Err(format!("valid for {lifetime} s"));
    }
    if !spent.insert(claims.jti.clone()) {
        return Err(format!("jti {} was already used", claims.jti));
    }
    Ok(())
}

fn config_at(dir: &Path) -> Config {
    let mut config = Config::default();
    config.general.data_dir = dir.to_path_buf();
    config
}

fn temp() -> tempfile::TempDir {
    tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()))
}

fn name() -> ClientKeyName {
    ClientKeyName::new("https://us.vouch.sh/", "quack")
}

#[test]
fn a_key_name_is_the_issuer_without_its_slash_and_the_client() {
    assert_eq!(name().as_str(), "https://us.vouch.sh quack");
    assert_eq!(
        ClientKeyName::new("https://us.vouch.sh", "quack"),
        name(),
        "a trailing slash names the same client"
    );
}

#[test]
fn the_thumbprint_is_rfc_7638s() {
    // The P-256 key of RFC 7517 appendix A.1; the value was computed
    // independently over RFC 7638's canonical form.
    assert_eq!(
        thumbprint(
            "f83OJ3D2xF1Bg8vub9tLe1gHMzV76e8Tus9uPHvRVEU",
            "x_FEzRu9m36HLN_tue659LNpXW6pCyStikYjKIWI5a0",
        ),
        "oKIywvGUpTVTyxMQ3bwIIeQUudfr_CkLMjCE19ECD-U"
    );
}

#[test]
fn an_assertion_verifies_against_the_public_jwk_once() {
    let Ok((key, der)) = ClientKey::generate() else {
        fail("key generation failed");
    };
    let jwk = key.jwk();
    assert_eq!(
        (
            jwk.kty.as_str(),
            jwk.crv.as_str(),
            jwk.use_.as_str(),
            jwk.alg.as_str()
        ),
        ("EC", "P-256", "sig", "ES256")
    );
    assert_eq!(jwk.kid, key.thumbprint());
    let mut spent = HashSet::new();
    let Ok(first) = key.assertion("quack", "https://us.vouch.sh") else {
        fail("signing failed");
    };
    assert!(verify(jwk, &first, "quack", "https://us.vouch.sh", &mut spent).is_ok());
    // A replayed assertion is refused: its jti is spent.
    assert!(
        verify(jwk, &first, "quack", "https://us.vouch.sh", &mut spent)
            .is_err_and(|e| e.contains("already used"))
    );
    let Ok(second) = key.assertion("quack", "https://us.vouch.sh") else {
        fail("signing failed");
    };
    assert!(verify(jwk, &second, "quack", "https://us.vouch.sh", &mut spent).is_ok());
    // Another audience, another client, or another key is refused.
    let Ok(third) = key.assertion("quack", "https://us.vouch.sh") else {
        fail("signing failed");
    };
    assert!(
        verify(
            jwk,
            &third,
            "quack",
            "https://us.vouch.sh/oauth/token",
            &mut spent
        )
        .is_err()
    );
    assert!(verify(jwk, &third, "other", "https://us.vouch.sh", &mut spent).is_err());
    let Ok((other, _)) = ClientKey::generate() else {
        fail("key generation failed");
    };
    let mut forged = other.jwk().clone();
    forged.kid.clone_from(&jwk.kid);
    assert!(verify(&forged, &third, "quack", "https://us.vouch.sh", &mut spent).is_err());
    // The PKCS#8 form reopens as the same key.
    let reopened = ClientKey::from_pkcs8(&der);
    assert!(reopened.is_ok_and(|k| k.jwk() == jwk));
    assert!(ClientKey::from_pkcs8(b"not a key").is_err());
}

#[tokio::test]
async fn a_key_is_made_once_and_outlives_the_process() {
    let dir = temp();
    let keys = ClientKeys::new(&config_at(dir.path()), KeySource::File);
    assert!(keys.existing(&name()).await.is_ok_and(|k| k.is_none()));
    let Ok(made) = keys.key(&name()).await else {
        fail("no key");
    };
    // The same process gets the same key.
    let again = keys.key(&name()).await;
    assert!(again.is_ok_and(|k| Arc::ptr_eq(&k, &made)));
    // Another process on the same data directory opens the stored row.
    let other = ClientKeys::new(&config_at(dir.path()), KeySource::File);
    let stored = other.load_or_create(&name()).await;
    assert!(stored.is_ok_and(|k| k.thumbprint() == made.thumbprint()));
    // Another client is another key.
    let elsewhere = ClientKeyName::new("https://us.vouch.sh", "other");
    let other_key = other.load_or_create(&elsewhere).await;
    assert!(other_key.is_ok_and(|k| k.thumbprint() != made.thumbprint()));
    // Nothing of the key sits beside the database but the vault key.
    assert!(dir.path().join("vault.key").exists());
}

#[tokio::test]
async fn a_key_whose_vault_key_is_gone_is_replaced() {
    let dir = temp();
    let first = ClientKeys::new(&config_at(dir.path()), KeySource::File);
    let Ok(old) = first.load_or_create(&name()).await else {
        fail("no key");
    };
    assert!(std::fs::remove_file(dir.path().join("vault.key")).is_ok());
    let later = ClientKeys::new(&config_at(dir.path()), KeySource::File);
    assert!(later.load(&name()).await.is_ok_and(|k| k.is_none()));
    let Ok(new) = later.load_or_create(&name()).await else {
        fail("no replacement key");
    };
    assert_ne!(new.thumbprint(), old.thumbprint());
    // The row now holds the new key.
    let reread = ClientKeys::new(&config_at(dir.path()), KeySource::File);
    assert!(
        reread
            .load_or_create(&name())
            .await
            .is_ok_and(|k| k.thumbprint() == new.thumbprint())
    );
}
