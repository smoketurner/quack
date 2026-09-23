//! The encrypted on-disk token cache: `<data_dir>/tokens/<provider>.json`.
//!
//! The file holds AES-256-GCM ciphertext under a 32-byte key that lives in
//! the OS keychain (`keychain.rs`) or, when no keychain is usable, in a
//! `<provider>.key` file beside the cache with mode 0600. The provider name is
//! the associated data, so a cache copied under another provider's name does
//! not decrypt.

use std::path::{Path, PathBuf};

use aws_lc_rs::aead::{AES_256_GCM, Aad, NONCE_LEN, Nonce, RandomizedNonceKey};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use jiff::Timestamp;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use super::keychain;
use crate::config::ProviderName;
use crate::error::{Error, Result};

const KEY_LEN: usize = 32;
const FORMAT_VERSION: u32 = 1;

/// A token as the cache holds it.
#[derive(Clone)]
pub struct CachedToken {
    pub access_token: SecretString,
    pub expires_at: Timestamp,
    pub refresh_token: Option<SecretString>,
}

impl CachedToken {
    /// Whether more than `margin` remains before expiry.
    #[must_use]
    pub fn is_fresh(&self, now: Timestamp, margin: jiff::SignedDuration) -> bool {
        self.expires_at.duration_since(now) > margin
    }
}

impl std::fmt::Debug for CachedToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedToken")
            .field("expires_at", &self.expires_at)
            .field("has_refresh_token", &self.refresh_token.is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Serialize, Deserialize)]
struct Plaintext {
    access_token: String,
    expires_at: Timestamp,
    refresh_token: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct Envelope {
    version: u32,
    nonce: String,
    ciphertext: String,
}

/// Where the cache's encryption key is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySource {
    /// The OS keychain, falling back to the key file when it is unusable.
    Keychain,
    /// Only the `<provider>.key` file (tests, and hosts with no keychain).
    File,
}

impl std::fmt::Display for KeySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Keychain => "keychain",
            Self::File => "key file",
        })
    }
}

/// One provider's cache file and its key.
#[derive(Debug, Clone)]
pub struct TokenCache {
    provider: ProviderName,
    path: PathBuf,
    key_path: PathBuf,
    key_source: KeySource,
}

impl TokenCache {
    #[must_use]
    pub fn new(tokens_dir: &Path, provider: &ProviderName, key_source: KeySource) -> Self {
        Self {
            provider: provider.clone(),
            path: tokens_dir.join(format!("{provider}.json")),
            key_path: tokens_dir.join(format!("{provider}.key")),
            key_source,
        }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Read and decrypt the cached token, or `None` when there is no cache or
    /// its key is gone (a rebooted kernel keyring, a removed keychain entry).
    ///
    /// # Errors
    ///
    /// Returns an error when the file is unreadable, malformed, or was
    /// encrypted under a different key.
    pub async fn load(&self) -> Result<Option<CachedToken>> {
        let bytes = match std::fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let Some(key) = self.key(false).await? else {
            tracing::warn!(
                provider = %self.provider,
                "token cache exists but its encryption key is gone; a new login is needed"
            );
            return Ok(None);
        };
        let envelope: Envelope = serde_json::from_slice(&bytes)?;
        if envelope.version != FORMAT_VERSION {
            return Err(Error::Llm(format!(
                "token cache {} has unsupported version {}",
                self.path.display(),
                envelope.version
            )));
        }
        let nonce = BASE64
            .decode(envelope.nonce)
            .map_err(|e| Error::Llm(format!("token cache nonce is not base64: {e}")))?;
        let nonce: [u8; NONCE_LEN] = nonce
            .try_into()
            .map_err(|_| Error::Llm(String::from("token cache nonce has the wrong length")))?;
        let mut ciphertext = BASE64
            .decode(envelope.ciphertext)
            .map_err(|e| Error::Llm(format!("token cache ciphertext is not base64: {e}")))?;
        let aead = RandomizedNonceKey::new(&AES_256_GCM, &key)
            .map_err(|_| Error::Llm(String::from("token cache key is unusable")))?;
        let plaintext = aead
            .open_in_place(
                Nonce::assume_unique_for_key(nonce),
                Aad::from(self.provider.as_str().as_bytes()),
                &mut ciphertext,
            )
            .map_err(|_| {
                Error::Llm(format!(
                    "token cache {} does not decrypt with the stored key; run `quack auth logout {}` and log in again",
                    self.path.display(),
                    self.provider
                ))
            })?;
        let plain: Plaintext = serde_json::from_slice(plaintext)?;
        Ok(Some(CachedToken {
            access_token: SecretString::from(plain.access_token),
            expires_at: plain.expires_at,
            refresh_token: plain.refresh_token.map(SecretString::from),
        }))
    }

    /// Encrypt and write the token, creating the key on first use.
    ///
    /// # Errors
    ///
    /// Returns an error when the key cannot be created or stored, or the
    /// file cannot be written.
    pub async fn store(&self, token: &CachedToken) -> Result<()> {
        let key = self
            .key(true)
            .await?
            .ok_or_else(|| Error::Llm(String::from("token cache key could not be created")))?;
        let plain = Plaintext {
            access_token: token.access_token.expose_secret().to_owned(),
            expires_at: token.expires_at,
            refresh_token: token
                .refresh_token
                .as_ref()
                .map(|t| t.expose_secret().to_owned()),
        };
        let mut in_out = serde_json::to_vec(&plain)?;
        let aead = RandomizedNonceKey::new(&AES_256_GCM, &key)
            .map_err(|_| Error::Llm(String::from("token cache key is unusable")))?;
        let nonce = aead
            .seal_in_place_append_tag(Aad::from(self.provider.as_str().as_bytes()), &mut in_out)
            .map_err(|_| Error::Llm(String::from("token cache encryption failed")))?;
        let envelope = Envelope {
            version: FORMAT_VERSION,
            nonce: BASE64.encode(nonce.as_ref()),
            ciphertext: BASE64.encode(&in_out),
        };
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        write_private(&self.path, &serde_json::to_vec(&envelope)?)
    }

    /// Delete the cache file and its key wherever it lives.
    ///
    /// # Errors
    ///
    /// Returns an error when a file or keychain entry cannot be removed.
    pub async fn clear(&self) -> Result<()> {
        remove_if_present(&self.path)?;
        remove_if_present(&self.key_path)?;
        if self.key_source == KeySource::Keychain
            && let Err(e) = keychain::delete(self.provider.as_str()).await
        {
            tracing::warn!(provider = %self.provider, error = %e, "keychain entry not removed");
        }
        Ok(())
    }

    /// The key, from the keychain when configured and usable, else the key
    /// file. With `create`, a missing key is generated and stored.
    async fn key(&self, create: bool) -> Result<Option<[u8; KEY_LEN]>> {
        if self.key_source == KeySource::Keychain {
            match self.keychain_key(create).await {
                Ok(key) => return Ok(key),
                Err(e) => {
                    tracing::warn!(provider = %self.provider, error = %e, "keychain unavailable; using the key file");
                }
            }
        }
        self.file_key(create)
    }

    async fn keychain_key(&self, create: bool) -> Result<Option<[u8; KEY_LEN]>> {
        if let Some(encoded) = keychain::get(self.provider.as_str()).await? {
            return decode_key(&encoded).map(Some);
        }
        if !create {
            return Ok(None);
        }
        let key = generate_key()?;
        keychain::set(self.provider.as_str(), &BASE64.encode(key)).await?;
        Ok(Some(key))
    }

    fn file_key(&self, create: bool) -> Result<Option<[u8; KEY_LEN]>> {
        match std::fs::read_to_string(&self.key_path) {
            Ok(encoded) => decode_key(encoded.trim()).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if !create {
                    return Ok(None);
                }
                let key = generate_key()?;
                if let Some(dir) = self.key_path.parent() {
                    std::fs::create_dir_all(dir)?;
                }
                write_private(&self.key_path, BASE64.encode(key).as_bytes())?;
                Ok(Some(key))
            }
            Err(e) => Err(e.into()),
        }
    }
}

fn generate_key() -> Result<[u8; KEY_LEN]> {
    let mut key = [0u8; KEY_LEN];
    aws_lc_rs::rand::fill(&mut key)
        .map_err(|_| Error::Llm(String::from("random key generation failed")))?;
    Ok(key)
}

fn decode_key(encoded: &str) -> Result<[u8; KEY_LEN]> {
    BASE64
        .decode(encoded)
        .ok()
        .and_then(|k| k.try_into().ok())
        .ok_or_else(|| Error::Llm(String::from("stored token cache key is malformed")))
}

fn remove_if_present(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Write a file readable only by its owner.
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(bytes)?;
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hour_one() -> Timestamp {
        Timestamp::UNIX_EPOCH
            .checked_add(jiff::SignedDuration::from_hours(1))
            .unwrap_or(Timestamp::MAX)
    }

    fn token(access: &str, refresh: Option<&str>) -> CachedToken {
        CachedToken {
            access_token: SecretString::from(access.to_owned()),
            expires_at: hour_one(),
            refresh_token: refresh.map(|r| SecretString::from(r.to_owned())),
        }
    }

    fn temp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()))
    }

    fn name(text: &str) -> ProviderName {
        text.parse().unwrap_or_else(|e: Error| fail(&e.to_string()))
    }

    /// Fail the test with a message; `!` lets it sit in a `let ... else`.
    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[tokio::test]
    async fn round_trip_with_a_file_key() {
        let dir = temp();
        let cache = TokenCache::new(dir.path(), &name("azure"), KeySource::File);
        assert!(cache.load().await.is_ok_and(|t| t.is_none()));
        assert!(cache.store(&token("at", Some("rt"))).await.is_ok());
        let loaded = cache.load().await;
        assert!(loaded.as_ref().is_ok_and(|t| {
            t.as_ref().is_some_and(|t| {
                t.access_token.expose_secret() == "at"
                    && t.refresh_token.as_ref().map(ExposeSecret::expose_secret) == Some("rt")
                    && t.expires_at == hour_one()
            })
        }));
        let raw = std::fs::read_to_string(cache.path());
        assert!(raw.is_ok_and(|r| !r.contains("at\"") && r.contains("ciphertext")));
    }

    #[cfg(unix)]
    #[test]
    fn private_files_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = temp();
        let path = dir.path().join("k");
        assert!(write_private(&path, b"x").is_ok());
        let mode = std::fs::metadata(&path).map(|m| m.permissions().mode() & 0o777);
        assert!(mode.is_ok_and(|m| m == 0o600));
    }

    #[tokio::test]
    async fn tampered_ciphertext_and_wrong_provider_are_rejected() {
        let dir = temp();
        let cache = TokenCache::new(dir.path(), &name("a"), KeySource::File);
        assert!(cache.store(&token("at", None)).await.is_ok());
        // Same key file, different associated data.
        let renamed = dir.path().join("b.json");
        assert!(std::fs::copy(cache.path(), &renamed).is_ok());
        assert!(std::fs::copy(dir.path().join("a.key"), dir.path().join("b.key")).is_ok());
        let other = TokenCache::new(dir.path(), &name("b"), KeySource::File);
        let err = other.load().await.err();
        assert!(err.is_some_and(|e| e.to_string().contains("does not decrypt")));
        // Flipped ciphertext byte.
        let Ok(text) = std::fs::read_to_string(cache.path()) else {
            fail("cache unreadable");
        };
        let Ok(mut envelope) = serde_json::from_str::<Envelope>(&text) else {
            fail("envelope unreadable");
        };
        envelope
            .ciphertext
            .truncate(envelope.ciphertext.len().saturating_sub(4));
        envelope.ciphertext.push_str("AAAA");
        let Ok(bytes) = serde_json::to_vec(&envelope) else {
            fail("envelope unwritable");
        };
        assert!(std::fs::write(cache.path(), bytes).is_ok());
        assert!(cache.load().await.is_err());
    }

    #[tokio::test]
    async fn missing_key_means_no_token_and_clear_removes_everything() {
        let dir = temp();
        let cache = TokenCache::new(dir.path(), &name("p"), KeySource::File);
        assert!(cache.store(&token("at", None)).await.is_ok());
        assert!(std::fs::remove_file(dir.path().join("p.key")).is_ok());
        assert!(cache.load().await.is_ok_and(|t| t.is_none()));
        assert!(cache.clear().await.is_ok());
        assert!(!cache.exists());
        assert!(cache.clear().await.is_ok());
    }

    #[test]
    fn freshness_uses_the_margin() {
        let t = token("a", None);
        let margin = jiff::SignedDuration::from_secs(60);
        assert!(t.is_fresh(Timestamp::UNIX_EPOCH, margin));
        let late = t
            .expires_at
            .checked_sub(jiff::SignedDuration::from_secs(30))
            .unwrap_or(Timestamp::MIN);
        assert!(!t.is_fresh(late, margin));
    }
}
