//! User tokens through `control.db`.

use jiff::Timestamp;
use secrecy::{ExposeSecret, SecretString};

use super::*;
use crate::config::Config;
use crate::error::Error;
use crate::llm::oauth::KeySource;
use crate::oidc::OidcSubject;
use crate::storage::control::SealedOwner;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

async fn control(dir: &std::path::Path) -> ControlPlane {
    let mut config = Config::default();
    config.general.data_dir = dir.to_path_buf();
    ControlPlane::open(&config)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()))
}

async fn user(control: &ControlPlane, name: &str) -> UserId {
    control
        .oidc_user(&OidcSubject::from(name), name)
        .await
        .map_or_else(|e| fail(&e.to_string()), |u| u.id)
}

fn token(access: &str) -> CachedToken {
    CachedToken {
        access_token: SecretString::from(access.to_owned()),
        expires_at: Timestamp::UNIX_EPOCH,
        refresh_token: Some(SecretString::from(format!("{access}-refresh"))),
    }
}

#[tokio::test]
async fn a_token_round_trips_and_a_moved_row_does_not_open() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let control = control(dir.path()).await;
    let (ada, bob) = (user(&control, "ada").await, user(&control, "bob").await);
    let tokens = UserTokens::new(Vault::new(dir.path(), KeySource::File));

    assert!(tokens.load(&control, &ada).await.is_ok_and(|t| t.is_none()));
    assert!(
        tokens
            .store(&control, &ada, &token("ada-access"))
            .await
            .is_ok()
    );
    let loaded = tokens.load(&control, &ada).await;
    assert!(loaded.is_ok_and(|t| t.is_some_and(|t| {
        t.access_token.expose_secret() == "ada-access"
            && t.refresh_token
                .is_some_and(|r| r.expose_secret() == "ada-access-refresh")
    })));

    let Ok(Some(row)) = control.sealed(SealedOwner::User(&ada)).await else {
        fail("no row");
    };
    assert!(
        control
            .put_sealed(SealedOwner::User(&bob), &row)
            .await
            .is_ok()
    );
    assert!(matches!(
        tokens.load(&control, &bob).await,
        Err(Error::Vault(_))
    ));

    assert!(tokens.clear(&control, &ada).await.is_ok());
    assert!(tokens.load(&control, &ada).await.is_ok_and(|t| t.is_none()));
}

#[tokio::test]
async fn a_token_whose_key_is_gone_reads_as_none() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let control = control(dir.path()).await;
    let ada = user(&control, "ada").await;
    let tokens = UserTokens::new(Vault::new(dir.path(), KeySource::File));
    assert!(tokens.store(&control, &ada, &token("a")).await.is_ok());
    assert!(std::fs::remove_file(dir.path().join("vault.key")).is_ok());
    let fresh = UserTokens::new(Vault::new(dir.path(), KeySource::File));
    assert!(fresh.load(&control, &ada).await.is_ok_and(|t| t.is_none()));
}
