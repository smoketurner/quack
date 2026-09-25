//! Data at rest sealed with HPKE (RFC 9180, base mode): DHKEM(P-256,
//! HKDF-SHA256), HKDF-SHA256, AES-256-GCM, through rustls's aws-lc-rs
//! implementation, which keeps that suite in FIPS builds.
//!
//! The process has one HPKE key pair, in the OS keychain (entry `vault`) or,
//! where there is none, `<data_dir>/vault.key` with mode 0600: never beside
//! what it protects. A [`Sealed`] value is storage-agnostic, so each use keeps
//! it on the right side of the classification boundary (`control.db` for
//! credentials, the workspace file for workspace content). Every value is
//! sealed for a [`Purpose`] (the HPKE `info`) and a subject (the associated
//! data), so it opens only as what it was sealed as, for whom it was sealed.

use std::path::Path;

use aws_lc_rs::digest::{SHA256, digest};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use rustls::crypto::aws_lc_rs::hpke::DH_KEM_P256_HKDF_SHA256_AES_256;
use rustls::crypto::hpke::{EncapsulatedSecret, Hpke, HpkePrivateKey, HpkePublicKey};
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use crate::error::{Error, Result};
use crate::llm::oauth::{KeyLocation, KeySlot, KeySource};

/// What a sealed value is for. Each purpose seals under its own HPKE `info`,
/// so a value sealed for one cannot be opened as another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// A signed-in user's identity-provider token; the subject is the user id.
    UserToken,
    /// A model provider's OAuth token; the subject is the provider name.
    ProviderToken,
}

text_enum!(Purpose, "vault purpose", {
    UserToken => "user-token",
    ProviderToken => "provider-token",
});

impl Purpose {
    fn info(self) -> Vec<u8> {
        format!("quack vault v1 {self}").into_bytes()
    }
}

/// A sealed value as any store keeps it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sealed {
    /// Which of the vault's keys sealed it: the first 8 bytes of the public
    /// key's SHA-256, in hex.
    pub key_id: String,
    /// The HPKE encapsulated key.
    pub enc: Vec<u8>,
    pub ciphertext: Vec<u8>,
}

/// What opening a sealed value found.
#[derive(Debug)]
pub enum Opened {
    Plaintext(Vec<u8>),
    /// The key that sealed it is gone (a rebooted kernel keyring, a removed
    /// key file, a new key): the value cannot be recovered.
    KeyGone,
}

/// The suite every value is sealed with.
fn suite() -> &'static dyn Hpke {
    DH_KEM_P256_HKDF_SHA256_AES_256
}

fn vault_error(message: impl Into<String>) -> Error {
    Error::Vault(message.into())
}

/// The key pair as the keychain or the key file holds it.
#[derive(Serialize, Deserialize)]
struct StoredKey {
    public: String,
    private: String,
}

/// The vault's HPKE key pair and its id.
struct VaultKey {
    id: String,
    public: HpkePublicKey,
    private: HpkePrivateKey,
}

impl VaultKey {
    /// A new key pair, encoded for storage.
    fn generate() -> Result<String> {
        let (public, private) = suite()
            .generate_key_pair()
            .map_err(|e| vault_error(format!("generating the vault key failed: {e}")))?;
        let stored = StoredKey {
            public: BASE64.encode(&public.0),
            private: BASE64.encode(private.secret_bytes()),
        };
        Ok(serde_json::to_string(&stored)?)
    }

    fn decode(text: &str) -> Result<Self> {
        let stored: StoredKey = serde_json::from_str(text.trim())
            .map_err(|e| vault_error(format!("the vault key is malformed: {e}")))?;
        let bytes = |b64: &str| {
            BASE64
                .decode(b64)
                .map_err(|e| vault_error(format!("the vault key is not base64: {e}")))
        };
        let public = bytes(&stored.public)?;
        let id = digest(&SHA256, &public)
            .as_ref()
            .iter()
            .take(8)
            .flat_map(|byte| [byte >> 4, byte & 0x0f])
            .filter_map(|nibble| char::from_digit(u32::from(nibble), 16))
            .collect();
        Ok(Self {
            id,
            public: HpkePublicKey(public),
            private: HpkePrivateKey::from(bytes(&stored.private)?),
        })
    }
}

/// Seals and opens values under the process's key.
pub struct Vault {
    slot: KeySlot,
    key: OnceCell<VaultKey>,
}

impl std::fmt::Debug for Vault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vault").finish_non_exhaustive()
    }
}

impl Vault {
    /// The vault whose key is the keychain entry `vault`, or
    /// `<data_dir>/vault.key`.
    #[must_use]
    pub fn new(data_dir: &Path, key_source: KeySource) -> Self {
        Self {
            slot: KeySlot::new(
                String::from("vault"),
                data_dir.join("vault.key"),
                key_source,
            ),
            key: OnceCell::new(),
        }
    }

    /// Where the key is: the key file when there is one, else the keychain.
    #[must_use]
    pub fn key_location(&self) -> KeyLocation {
        self.slot.location()
    }

    /// The key, made and stored the first time something is sealed.
    async fn sealing_key(&self) -> Result<&VaultKey> {
        self.key
            .get_or_try_init(|| async {
                VaultKey::decode(&self.slot.read_or_create(VaultKey::generate).await?)
            })
            .await
    }

    /// The key, when one exists; opening never makes one.
    async fn opening_key(&self) -> Result<Option<&VaultKey>> {
        if let Some(key) = self.key.get() {
            return Ok(Some(key));
        }
        let Some(text) = self.slot.read().await? else {
            return Ok(None);
        };
        let key = VaultKey::decode(&text)?;
        Ok(Some(self.key.get_or_init(|| async { key }).await))
    }

    /// Seal `plaintext` for `purpose` and `subject`.
    ///
    /// # Errors
    ///
    /// Returns an error when the key cannot be made or read, or sealing
    /// fails.
    pub async fn seal(&self, purpose: Purpose, subject: &str, plaintext: &[u8]) -> Result<Sealed> {
        let key = self.sealing_key().await?;
        let (enc, ciphertext) = suite()
            .seal(&purpose.info(), subject.as_bytes(), plaintext, &key.public)
            .map_err(|e| vault_error(format!("sealing a {purpose} failed: {e}")))?;
        Ok(Sealed {
            key_id: key.id.clone(),
            enc: enc.0,
            ciphertext,
        })
    }

    /// Open a value sealed for `purpose` and `subject`.
    ///
    /// # Errors
    ///
    /// Returns an error when the value does not open under the key that
    /// sealed it: altered, or sealed for another purpose or subject.
    pub async fn open(&self, purpose: Purpose, subject: &str, sealed: &Sealed) -> Result<Opened> {
        let key = match self.opening_key().await? {
            Some(key) if key.id == sealed.key_id => key,
            Some(_) | None => return Ok(Opened::KeyGone),
        };
        suite()
            .open(
                &EncapsulatedSecret(sealed.enc.clone()),
                &purpose.info(),
                subject.as_bytes(),
                &sealed.ciphertext,
                &key.private,
            )
            .map(Opened::Plaintext)
            .map_err(|_| vault_error(format!("a sealed {purpose} for {subject} does not open")))
    }
}

#[cfg(test)]
mod tests;
