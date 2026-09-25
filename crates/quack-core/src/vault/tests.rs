//! Sealing, opening, and the ways a value must fail to open.

use super::*;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

fn opened(result: Result<Opened>) -> Option<Vec<u8>> {
    match result {
        Ok(Opened::Plaintext(bytes)) => Some(bytes),
        Ok(Opened::KeyGone) | Err(_) => None,
    }
}

#[tokio::test]
async fn a_value_round_trips_under_a_named_key_that_persists() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let vault = Vault::new(dir.path(), KeySource::File);
    let sealed = vault
        .seal(Purpose::UserToken, "subject-1", b"the secret")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(sealed.key_id.len(), 16);
    assert!(sealed.key_id.chars().all(|c| c.is_ascii_hexdigit()));
    // A P-256 encapsulated key is an uncompressed point.
    assert_eq!(sealed.enc.len(), 65);
    assert!(!sealed.ciphertext.windows(10).any(|w| w == b"the secret"));
    assert_eq!(
        opened(vault.open(Purpose::UserToken, "subject-1", &sealed).await).as_deref(),
        Some(&b"the secret"[..])
    );
    // Two seals of the same value differ: a fresh encapsulation each time.
    let again = vault
        .seal(Purpose::UserToken, "subject-1", b"the secret")
        .await;
    assert!(again.is_ok_and(|a| a.enc != sealed.enc && a.key_id == sealed.key_id));

    assert!(dir.path().join("vault.key").exists());
    let reopened = Vault::new(dir.path(), KeySource::File);
    assert_eq!(
        opened(
            reopened
                .open(Purpose::UserToken, "subject-1", &sealed)
                .await
        )
        .as_deref(),
        Some(&b"the secret"[..])
    );
}

#[tokio::test]
async fn another_subject_or_an_altered_value_does_not_open() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let vault = Vault::new(dir.path(), KeySource::File);
    let sealed = vault
        .seal(Purpose::UserToken, "ada", b"ada's")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let moved = vault.open(Purpose::UserToken, "bob", &sealed).await;
    assert!(matches!(moved, Err(Error::Vault(_))), "{moved:?}");

    let mut altered = sealed.clone();
    if let Some(byte) = altered.ciphertext.first_mut() {
        *byte ^= 1;
    }
    assert!(matches!(
        vault.open(Purpose::UserToken, "ada", &altered).await,
        Err(Error::Vault(_))
    ));
    let mut swapped = sealed;
    if let Some(byte) = swapped.enc.last_mut() {
        *byte ^= 1;
    }
    assert!(
        vault
            .open(Purpose::UserToken, "ada", &swapped)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn a_value_whose_key_is_gone_says_so_and_opening_makes_no_key() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let vault = Vault::new(dir.path(), KeySource::File);
    let sealed = vault
        .seal(Purpose::UserToken, "ada", b"x")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));

    let elsewhere = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let other = Vault::new(elsewhere.path(), KeySource::File);
    assert!(other.seal(Purpose::UserToken, "cy", b"y").await.is_ok());
    assert!(matches!(
        other.open(Purpose::UserToken, "ada", &sealed).await,
        Ok(Opened::KeyGone)
    ));

    assert!(std::fs::remove_file(dir.path().join("vault.key")).is_ok());
    let fresh = Vault::new(dir.path(), KeySource::File);
    assert!(matches!(
        fresh.open(Purpose::UserToken, "ada", &sealed).await,
        Ok(Opened::KeyGone)
    ));
    assert!(!dir.path().join("vault.key").exists());
}

#[test]
fn each_purpose_has_its_own_info() {
    let infos: std::collections::BTreeSet<Vec<u8>> =
        Purpose::ALL.iter().map(|p| p.info()).collect();
    assert_eq!(infos.len(), Purpose::ALL.len());
    assert_eq!(Purpose::UserToken.info(), b"quack vault v1 user-token");
}
