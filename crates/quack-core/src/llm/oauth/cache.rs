//! The encrypted on-disk token cache: `<data_dir>/tokens/<provider>.json`.
//!
//! The file holds AES-256-GCM ciphertext under a 32-byte key in the
//! provider's [`KeySlot`]: the OS keychain, or a `<provider>.key` file beside
//! the cache with mode 0600. The provider name is the associated data, so a
//! cache copied under another provider's name does not decrypt.

use std::path::{Path, PathBuf};

use aws_lc_rs::aead::{AES_256_GCM, Aad, NONCE_LEN, Nonce, RandomizedNonceKey};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use jiff::Timestamp;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use super::key_slot::{KeyLocation, KeySlot, KeySource, remove_if_present, write_private};
use crate::config::ProviderName;
use crate::error::{Error, Result};

const FORMAT_VERSION: u32 = 1;

/// A cache's AES-256-GCM key.
struct CacheKey([u8; 32]);

impl CacheKey {
    /// A new key from the process's CSPRNG (aws-lc-rs).
    fn generate() -> Result<Self> {
        let mut key = [0u8; 32];
        aws_lc_rs::rand::fill(&mut key)
            .map_err(|_| Error::Llm(String::from("random key generation failed")))?;
        Ok(Self(key))
    }

    /// A key as the keychain or the key file stores it: base64.
    fn decode(encoded: &str) -> Result<Self> {
        BASE64
            .decode(encoded.trim())
            .ok()
            .and_then(|k| k.try_into().ok())
            .map(Self)
            .ok_or_else(|| Error::Llm(String::from("stored token cache key is malformed")))
    }

    fn encode(&self) -> String {
        BASE64.encode(self.0)
    }

    /// The cipher this key opens and seals with.
    fn aead(&self) -> Result<RandomizedNonceKey> {
        RandomizedNonceKey::new(&AES_256_GCM, &self.0)
            .map_err(|_| Error::Llm(String::from("token cache key is unusable")))
    }
}

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

/// A token as it is serialized before sealing.
#[derive(Serialize, Deserialize)]
pub(crate) struct Plaintext {
    access_token: String,
    expires_at: Timestamp,
    refresh_token: Option<String>,
}

impl From<Plaintext> for CachedToken {
    fn from(plain: Plaintext) -> Self {
        Self {
            access_token: SecretString::from(plain.access_token),
            expires_at: plain.expires_at,
            refresh_token: plain.refresh_token.map(SecretString::from),
        }
    }
}

impl From<&CachedToken> for Plaintext {
    fn from(token: &CachedToken) -> Self {
        Self {
            access_token: token.access_token.expose_secret().to_owned(),
            expires_at: token.expires_at,
            refresh_token: token
                .refresh_token
                .as_ref()
                .map(|t| t.expose_secret().to_owned()),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Envelope {
    version: u32,
    nonce: String,
    ciphertext: String,
}

/// One provider's cache file and its key.
#[derive(Debug, Clone)]
pub struct TokenCache {
    provider: ProviderName,
    path: PathBuf,
    key: KeySlot,
}

impl TokenCache {
    #[must_use]
    pub fn new(tokens_dir: &Path, provider: &ProviderName, key_source: KeySource) -> Self {
        Self {
            provider: provider.clone(),
            path: tokens_dir.join(format!("{provider}.json")),
            key: KeySlot::new(
                format!("oauth:{provider}"),
                tokens_dir.join(format!("{provider}.key")),
                key_source,
            ),
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

    /// Where the key is: the key file when there is one, else the
    /// keychain.
    #[must_use]
    pub fn key_location(&self) -> KeyLocation {
        self.key.location()
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
        let Some(key) = self.key.read().await? else {
            tracing::warn!(
                provider = %self.provider,
                "token cache exists but its encryption key is gone; a new login is needed"
            );
            return Ok(None);
        };
        let key = CacheKey::decode(&key)?;
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
        let plaintext = key
            .aead()?
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
        Ok(Some(CachedToken::from(plain)))
    }

    /// Encrypt and write the token, creating the key on first use.
    ///
    /// # Errors
    ///
    /// Returns an error when the key cannot be created or stored, or the
    /// file cannot be written.
    pub async fn store(&self, token: &CachedToken) -> Result<()> {
        let key = self
            .key
            .read_or_create(|| CacheKey::generate().map(|k| k.encode()))
            .await?;
        let key = CacheKey::decode(&key)?;
        let mut in_out = serde_json::to_vec(&Plaintext::from(token))?;
        let nonce = key
            .aead()?
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
    /// Returns an error when a file cannot be removed.
    pub async fn clear(&self) -> Result<()> {
        remove_if_present(&self.path)?;
        self.key.delete().await
    }
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
