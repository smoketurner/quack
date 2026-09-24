//! The encrypted on-disk token caches: a provider's at
//! `<data_dir>/tokens/<provider>.json`, and each signed-in server user's at
//! `<data_dir>/tokens/users/<user-id>.json`.
//!
//! A file holds AES-256-GCM ciphertext under a 32-byte key that lives in the
//! OS keychain (`keychain.rs`) or, when no keychain is usable, in a key file
//! with mode 0600. A provider has a key of its own (`<provider>.key`); every
//! user's file shares one (`users.key`), so the number of keychain entries
//! does not grow with the number of users. The owner's name is the
//! associated data, so a cache copied under another name does not decrypt.

use std::path::{Path, PathBuf};

use aws_lc_rs::aead::{AES_256_GCM, Aad, NONCE_LEN, Nonce, RandomizedNonceKey};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use jiff::Timestamp;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use super::keychain::KeychainEntry;
use crate::config::ProviderName;
use crate::error::{Error, Result};
use crate::ids::UserId;

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

/// Whether a missing key is made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyLookup {
    /// Reading: no key means no token.
    Existing,
    /// Writing: a missing key is generated and stored.
    CreateIfMissing,
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

#[derive(Serialize, Deserialize)]
struct Plaintext {
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

/// Where the cache's encryption key may be kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySource {
    /// The OS keychain, falling back to the key file when it is unusable.
    Keychain,
    /// Only the `<provider>.key` file (tests, and hosts with no keychain).
    File,
}

/// Where a cache's key turned out to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyLocation {
    Keychain,
    File,
}

impl std::fmt::Display for KeyLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Keychain => "keychain",
            Self::File => "key file",
        })
    }
}

/// Whose token a cache holds, which decides its file, its key, and the
/// associated data that binds the ciphertext to it.
#[derive(Debug, Clone)]
enum CacheOwner {
    /// A provider's token, under a key of its own.
    Provider(ProviderName),
    /// A signed-in server user's token, under the key all users share.
    User(UserId),
}

impl CacheOwner {
    /// The keychain account and key-file stem every user's cache shares.
    const USERS: &str = "users";

    fn aad(&self) -> String {
        match self {
            Self::Provider(provider) => provider.to_string(),
            Self::User(user) => format!("user:{user}"),
        }
    }

    fn keychain(&self) -> KeychainEntry {
        KeychainEntry::new(match self {
            Self::Provider(provider) => format!("oauth:{provider}"),
            Self::User(_) => format!("oidc:{}", Self::USERS),
        })
    }

    /// Whether the key belongs to this cache alone, so clearing the cache
    /// removes it too.
    const fn owns_key(&self) -> bool {
        match self {
            Self::Provider(_) => true,
            Self::User(_) => false,
        }
    }

    /// What someone does when the cache no longer decrypts.
    fn recovery(&self) -> String {
        match self {
            Self::Provider(provider) => {
                format!("run `quack auth logout {provider}` and log in again")
            }
            Self::User(_) => String::from("the user signs in again"),
        }
    }
}

impl std::fmt::Display for CacheOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provider(provider) => write!(f, "provider '{provider}'"),
            Self::User(user) => write!(f, "user {user}"),
        }
    }
}

/// One cache file and its key.
#[derive(Debug, Clone)]
pub struct TokenCache {
    owner: CacheOwner,
    path: PathBuf,
    key_path: PathBuf,
    key_source: KeySource,
}

impl TokenCache {
    /// A provider's cache, `<tokens_dir>/<provider>.json`.
    #[must_use]
    pub fn new(tokens_dir: &Path, provider: &ProviderName, key_source: KeySource) -> Self {
        Self {
            owner: CacheOwner::Provider(provider.clone()),
            path: tokens_dir.join(format!("{provider}.json")),
            key_path: tokens_dir.join(format!("{provider}.key")),
            key_source,
        }
    }

    /// A server user's cache, `<tokens_dir>/users/<user-id>.json`, sealed
    /// under the key every user shares.
    ///
    /// # Errors
    ///
    /// Returns an error when the id could not be a file name: anything but
    /// ASCII letters, digits, and `-`.
    pub fn for_user(tokens_dir: &Path, user: &UserId, key_source: KeySource) -> Result<Self> {
        let id = user.as_str();
        if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            return Err(Error::Llm(format!(
                "user id '{id}' cannot name a token cache file"
            )));
        }
        Ok(Self {
            owner: CacheOwner::User(user.clone()),
            path: tokens_dir
                .join(CacheOwner::USERS)
                .join(format!("{id}.json")),
            key_path: tokens_dir.join(format!("{}.key", CacheOwner::USERS)),
            key_source,
        })
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
        if self.key_path.exists() {
            KeyLocation::File
        } else {
            KeyLocation::Keychain
        }
    }

    fn keychain(&self) -> KeychainEntry {
        self.owner.keychain()
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
        let Some(key) = self.key(KeyLookup::Existing).await? else {
            tracing::warn!(
                owner = %self.owner,
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
        let plaintext = key
            .aead()?
            .open_in_place(
                Nonce::assume_unique_for_key(nonce),
                Aad::from(self.owner.aad().as_bytes()),
                &mut ciphertext,
            )
            .map_err(|_| {
                Error::Llm(format!(
                    "token cache {} does not decrypt with the stored key; {}",
                    self.path.display(),
                    self.owner.recovery()
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
            .key(KeyLookup::CreateIfMissing)
            .await?
            .ok_or_else(|| Error::Llm(String::from("token cache key could not be created")))?;
        let mut in_out = serde_json::to_vec(&Plaintext::from(token))?;
        let nonce = key
            .aead()?
            .seal_in_place_append_tag(Aad::from(self.owner.aad().as_bytes()), &mut in_out)
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

    /// Delete the cache file, and its key wherever it lives when the key is
    /// this cache's alone.
    ///
    /// # Errors
    ///
    /// Returns an error when a file or keychain entry cannot be removed.
    pub async fn clear(&self) -> Result<()> {
        remove_if_present(&self.path)?;
        if !self.owner.owns_key() {
            return Ok(());
        }
        remove_if_present(&self.key_path)?;
        if self.key_source == KeySource::Keychain
            && let Err(e) = self.keychain().delete().await
        {
            tracing::warn!(owner = %self.owner, error = %e, "keychain entry not removed");
        }
        Ok(())
    }

    /// The key, from the keychain when configured and usable, else the key
    /// file.
    async fn key(&self, lookup: KeyLookup) -> Result<Option<CacheKey>> {
        if self.key_source == KeySource::Keychain {
            match self.keychain_key(lookup).await {
                Ok(key) => return Ok(key),
                Err(e) => {
                    tracing::warn!(owner = %self.owner, error = %e, "keychain unavailable; using the key file");
                }
            }
        }
        self.file_key(lookup)
    }

    async fn keychain_key(&self, lookup: KeyLookup) -> Result<Option<CacheKey>> {
        let keychain = self.keychain();
        if let Some(encoded) = keychain.get().await? {
            return CacheKey::decode(&encoded).map(Some);
        }
        if lookup == KeyLookup::Existing {
            return Ok(None);
        }
        let key = CacheKey::generate()?;
        keychain.set(&key.encode()).await?;
        Ok(Some(key))
    }

    fn file_key(&self, lookup: KeyLookup) -> Result<Option<CacheKey>> {
        match std::fs::read_to_string(&self.key_path) {
            Ok(encoded) => CacheKey::decode(&encoded).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if lookup == KeyLookup::Existing {
                    return Ok(None);
                }
                let key = CacheKey::generate()?;
                if let Some(dir) = self.key_path.parent() {
                    std::fs::create_dir_all(dir)?;
                }
                write_private(&self.key_path, key.encode().as_bytes())?;
                Ok(Some(key))
            }
            Err(e) => Err(e.into()),
        }
    }
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

    #[tokio::test]
    async fn users_share_one_key_that_clearing_a_user_keeps() {
        let dir = temp();
        let alice = UserId::from("0190a1b2-0000-7000-8000-000000000001");
        let bob = UserId::from("0190a1b2-0000-7000-8000-000000000002");
        let Ok(a) = TokenCache::for_user(dir.path(), &alice, KeySource::File) else {
            fail("cache for alice");
        };
        let Ok(b) = TokenCache::for_user(dir.path(), &bob, KeySource::File) else {
            fail("cache for bob");
        };
        assert!(a.store(&token("alice-at", Some("alice-rt"))).await.is_ok());
        assert!(b.store(&token("bob-at", None)).await.is_ok());
        assert!(a.path().ends_with(format!("users/{alice}.json")));
        assert!(dir.path().join("users.key").exists());
        assert!(!dir.path().join("users").join("users.key").exists());

        // Bob's ciphertext under Alice's name does not open: the user id is
        // the associated data.
        assert!(std::fs::copy(b.path(), a.path()).is_ok());
        let err = a.load().await.err();
        assert!(err.is_some_and(|e| e.to_string().contains("the user signs in again")));

        assert!(a.clear().await.is_ok());
        assert!(!a.exists());
        assert!(dir.path().join("users.key").exists());
        assert!(
            b.load()
                .await
                .is_ok_and(|t| t.is_some_and(|t| t.access_token.expose_secret() == "bob-at"))
        );
    }

    #[test]
    fn a_user_id_that_cannot_be_a_file_name_is_refused() {
        let dir = temp();
        for id in ["", "../escape", "a/b", "local.json"] {
            assert!(
                TokenCache::for_user(dir.path(), &UserId::from(id), KeySource::File).is_err(),
                "{id}"
            );
        }
        assert!(TokenCache::for_user(dir.path(), &UserId::from("local"), KeySource::File).is_ok());
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
