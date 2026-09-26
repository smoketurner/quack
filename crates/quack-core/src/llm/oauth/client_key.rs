//! The key quack signs its client assertions with (`client_auth =
//! "private_key_jwt"`, RFC 7523 section 2.2 and `OpenID` Connect Core 9):
//! instead of presenting a shared secret at the token endpoint, quack signs
//! a short-lived JWT with a private key that never leaves it, and the issuer
//! checks the signature against the public key registered for the client.
//!
//! Each client has one P-256 key ([`ClientKey`]), made with aws-lc-rs on
//! first use and kept in `control.db` (`client_keys`) as PKCS#8, sealed by
//! the vault for [`Purpose::ClientKey`]. The key is named by the client it
//! authenticates, `<issuer> <client_id>` ([`ClientKeyName`]), so
//! `[server.oidc]` and a model provider registered as the same client share
//! one key and one registered key set. [`ClientKeys`] loads it once per
//! process. A key whose vault key is gone cannot be opened again, so it is
//! replaced, and the new public key must be registered with the issuer
//! (`quack auth jwks`).
//!
//! A key can exist before the client does. A client registered with the
//! issuer through RFC 7591 learns its id only from the registration, which
//! must already carry the public key, so that key is made under the issuer's
//! name alone ([`ClientKeyName::pending`]) and moves to `<issuer>
//! <client_id>` once the issuer has assigned the id: `quack auth register`
//! moves it itself, and a client whose id the operator wrote into the file
//! after registering by hand takes it over on first use. A replacement key
//! waiting for the operator to register it is kept under
//! [`ClientKeyName::replacement`] until `quack auth jwks --rotate
//! --activate`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use aws_lc_rs::digest::{SHA256, digest};
use aws_lc_rs::rand::SystemRandom;
use aws_lc_rs::signature::{ECDSA_P256_SHA256_FIXED_SIGNING, EcdsaKeyPair, KeyPair};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use jiff::{SignedDuration, Timestamp};
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use super::key_slot::KeySource;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::storage::control::{ControlPlane, KeyChange, SealedOwner};
use crate::vault::{Opened, Purpose, Sealed, Vault};

/// The one algorithm quack signs assertions with, as issuers list it in
/// `token_endpoint_auth_signing_alg_values_supported`.
pub const ALGORITHM: &str = "ES256";

/// The `client_assertion_type` of a JWT client assertion (RFC 7523 2.2).
pub const ASSERTION_TYPE: &str = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer";

/// How long an assertion is valid: long enough for one request, short
/// enough that a copy is useless soon after.
const ASSERTION_LIFETIME: SignedDuration = SignedDuration::from_secs(60);

fn key_error(message: impl std::fmt::Display) -> Error {
    Error::Llm(format!("client key: {message}"))
}

/// The client a key authenticates: the issuer (without a trailing slash)
/// and the client id, space-separated. A URL holds no space, so the two
/// halves cannot run into each other. Before the client has an id its key
/// is named by the issuer alone ([`ClientKeyName::pending`]), and a
/// replacement waiting to be registered by `next ` and the client's name
/// ([`ClientKeyName::replacement`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClientKeyName {
    name: String,
    /// The issuer, without a trailing slash.
    issuer: String,
    role: KeyRole,
}

/// What a named key is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum KeyRole {
    /// A client's key in use.
    Client,
    /// A key made for a client the issuer has not yet assigned an id.
    Pending,
    /// A client's replacement key, not yet registered.
    Next,
}

impl ClientKeyName {
    #[must_use]
    pub fn new(issuer_url: &str, client_id: &str) -> Self {
        let issuer = issuer_url.trim_end_matches('/');
        Self {
            name: format!("{issuer} {client_id}"),
            issuer: issuer.to_owned(),
            role: KeyRole::Client,
        }
    }

    /// The key for a client not yet registered at the issuer: named by the
    /// issuer alone, since there is no client id yet. Registering moves it
    /// to [`ClientKeyName::new`].
    #[must_use]
    pub fn pending(issuer_url: &str) -> Self {
        let issuer = issuer_url.trim_end_matches('/');
        Self {
            name: issuer.to_owned(),
            issuer: issuer.to_owned(),
            role: KeyRole::Pending,
        }
    }

    /// The replacement for this client's key while the operator registers
    /// it: `next ` before the client's own name, which starts with the
    /// issuer's scheme and so can never start that way.
    #[must_use]
    pub fn replacement(&self) -> Self {
        Self {
            name: format!("next {}", self.name),
            issuer: self.issuer.clone(),
            role: KeyRole::Next,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Display for ClientKeyName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

/// A P-256 public key as a JWK (RFC 7517), with the members an issuer's
/// client registration wants: its thumbprint as `kid`, `use` `sig`, and the
/// algorithm.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicJwk {
    pub kty: String,
    pub crv: String,
    pub x: String,
    pub y: String,
    pub kid: String,
    #[serde(rename = "use")]
    pub use_: String,
    pub alg: String,
}

/// A JWK set (RFC 7517 5): what `quack auth jwks` prints, ready for the
/// issuer's `jwks` client metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicJwks {
    pub keys: Vec<PublicJwk>,
}

/// The RFC 7638 thumbprint of a P-256 key: SHA-256 over the required
/// members in lexicographic order with no whitespace, base64url without
/// padding.
fn thumbprint(x: &str, y: &str) -> String {
    let canonical = format!(r#"{{"crv":"P-256","kty":"EC","x":"{x}","y":"{y}"}}"#);
    URL_SAFE_NO_PAD.encode(digest(&SHA256, canonical.as_bytes()))
}

/// One client's signing key pair.
pub struct ClientKey {
    pair: EcdsaKeyPair,
    jwk: PublicJwk,
}

impl std::fmt::Debug for ClientKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientKey")
            .field("thumbprint", &self.jwk.kid)
            .finish_non_exhaustive()
    }
}

/// An assertion's JOSE header.
#[derive(Serialize)]
struct AssertionHeader<'a> {
    alg: &'static str,
    typ: &'static str,
    kid: &'a str,
}

/// An assertion's claims (RFC 7523 3).
#[derive(Serialize)]
struct AssertionClaims<'a> {
    iss: &'a str,
    sub: &'a str,
    aud: &'a str,
    jti: String,
    iat: i64,
    exp: i64,
}

impl ClientKey {
    /// A new key pair, and its PKCS#8 encoding for the vault to seal.
    ///
    /// # Errors
    ///
    /// Returns an error when aws-lc-rs cannot make the key.
    pub fn generate() -> Result<(Self, Vec<u8>)> {
        let pkcs8 =
            EcdsaKeyPair::generate_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, &SystemRandom::new())
                .map_err(|_| key_error("generating a P-256 key failed"))?;
        let der = pkcs8.as_ref().to_vec();
        Ok((Self::from_pkcs8(&der)?, der))
    }

    /// The key a PKCS#8 document holds.
    ///
    /// # Errors
    ///
    /// Returns an error when the document is not a P-256 private key.
    pub fn from_pkcs8(der: &[u8]) -> Result<Self> {
        let pair = EcdsaKeyPair::from_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, der)
            .map_err(|e| key_error(format!("the stored key is not a P-256 key: {e}")))?;
        // An uncompressed point: 0x04, then X and Y, 32 bytes each.
        let point = pair.public_key().as_ref();
        let (Some(x), Some(y)) = (point.get(1..33), point.get(33..65)) else {
            return Err(key_error(
                "the public key is not an uncompressed P-256 point",
            ));
        };
        let (x, y) = (URL_SAFE_NO_PAD.encode(x), URL_SAFE_NO_PAD.encode(y));
        let jwk = PublicJwk {
            kid: thumbprint(&x, &y),
            kty: String::from("EC"),
            crv: String::from("P-256"),
            x,
            y,
            use_: String::from("sig"),
            alg: String::from(ALGORITHM),
        };
        Ok(Self { pair, jwk })
    }

    /// The public key as a JWK.
    #[must_use]
    pub const fn jwk(&self) -> &PublicJwk {
        &self.jwk
    }

    /// The public key's RFC 7638 thumbprint, which is also its `kid`.
    #[must_use]
    pub fn thumbprint(&self) -> &str {
        &self.jwk.kid
    }

    /// The public key as a one-key set, for the issuer's registration.
    #[must_use]
    pub fn jwks(&self) -> PublicJwks {
        PublicJwks {
            keys: vec![self.jwk.clone()],
        }
    }

    /// A client assertion for one request: `client_id` as both `iss` and
    /// `sub`, `audience` (the issuer identifier) as `aud`, a new UUID v7 as
    /// `jti`, and a lifetime of a minute. The issuer spends each `jti`, so
    /// every request, a retry included, needs a new assertion.
    ///
    /// # Errors
    ///
    /// Returns an error when signing fails.
    pub fn assertion(&self, client_id: &str, audience: &str) -> Result<String> {
        let now = Timestamp::now();
        let header = AssertionHeader {
            alg: ALGORITHM,
            typ: "JWT",
            kid: self.thumbprint(),
        };
        let claims = AssertionClaims {
            iss: client_id,
            sub: client_id,
            aud: audience,
            jti: uuid::Uuid::now_v7().to_string(),
            iat: now.as_second(),
            exp: now
                .checked_add(ASSERTION_LIFETIME)
                .unwrap_or(Timestamp::MAX)
                .as_second(),
        };
        let input = format!(
            "{}.{}",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header)?),
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims)?)
        );
        // FIXED signing: R and S, 32 bytes each, as JWS ES256 wants.
        let signature = self
            .pair
            .sign(&SystemRandom::new(), input.as_bytes())
            .map_err(|_| key_error("signing an assertion failed"))?;
        Ok(format!(
            "{input}.{}",
            URL_SAFE_NO_PAD.encode(signature.as_ref())
        ))
    }
}

/// Which process-wide slot a key is cached in: one per data directory and
/// key name.
type CacheKey = (PathBuf, ClientKeyName);

/// One key's place in the cache, filled once.
type Slot = Arc<OnceCell<Arc<ClientKey>>>;

/// The keys this process has loaded, each loaded or made once.
fn cache() -> &'static Mutex<HashMap<CacheKey, Slot>> {
    static CACHE: OnceLock<Mutex<HashMap<CacheKey, Slot>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The client keys of one data directory: sealed in its `control.db`, which
/// is opened on first use, under its vault key.
pub struct ClientKeys {
    config: Config,
    vault: Vault,
    control: OnceCell<ControlPlane>,
}

impl std::fmt::Debug for ClientKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientKeys")
            .field("data_dir", &self.config.data_dir())
            .finish_non_exhaustive()
    }
}

impl ClientKeys {
    /// The keys of `config`'s data directory; `control.db` is opened when a
    /// key is first needed.
    #[must_use]
    pub fn new(config: &Config, key_source: KeySource) -> Self {
        Self {
            config: config.clone(),
            vault: Vault::new(config.data_dir(), key_source),
            control: OnceCell::new(),
        }
    }

    /// The keys of `config`'s data directory, through a control plane the
    /// caller already holds open.
    #[must_use]
    pub fn with_control(config: &Config, key_source: KeySource, control: ControlPlane) -> Self {
        Self {
            config: config.clone(),
            vault: Vault::new(config.data_dir(), key_source),
            control: OnceCell::new_with(Some(control)),
        }
    }

    pub(crate) async fn control(&self) -> Result<&ControlPlane> {
        self.control
            .get_or_try_init(|| ControlPlane::open(&self.config))
            .await
    }

    pub(crate) const fn vault(&self) -> &Vault {
        &self.vault
    }

    /// Drop the process's copy of the key `name`, so its next use reads the
    /// stored one: after a rotation or a registration moved or replaced it.
    pub(crate) fn forget(&self, name: &ClientKeyName) {
        let mut cache = cache().lock().unwrap_or_else(PoisonError::into_inner);
        cache.remove(&(self.config.data_dir().to_path_buf(), name.clone()));
    }

    /// Seal a key's PKCS#8 document for `name`.
    pub(crate) async fn seal(&self, name: &ClientKeyName, der: &[u8]) -> Result<Sealed> {
        self.vault
            .seal(Purpose::ClientKey, name.as_str(), der)
            .await
    }

    /// The stored key's PKCS#8 document, or `None` when there is none or
    /// the vault key that sealed it is gone.
    pub(crate) async fn stored_der(&self, name: &ClientKeyName) -> Result<Option<Vec<u8>>> {
        let owner = SealedOwner::ClientKey(name.as_str());
        let Some(sealed) = self.control().await?.sealed(owner).await? else {
            return Ok(None);
        };
        match self
            .vault
            .open(Purpose::ClientKey, name.as_str(), &sealed)
            .await?
        {
            Opened::Plaintext(der) => Ok(Some(der)),
            Opened::KeyGone => Ok(None),
        }
    }

    /// A client without a key of its own takes over the pending key made
    /// for its issuer: the one registered by hand from `quack auth register
    /// --print` before the operator wrote the client id into the file. The
    /// move is one transaction, so the key is never in both places.
    async fn adopt_pending(&self, name: &ClientKeyName) -> Result<Option<ClientKey>> {
        if name.role != KeyRole::Client {
            return Ok(None);
        }
        let pending = ClientKeyName::pending(&name.issuer);
        let Some(der) = self.stored_der(&pending).await? else {
            return Ok(None);
        };
        let key = ClientKey::from_pkcs8(&der)?;
        let sealed = self.seal(name, &der).await?;
        self.control()
            .await?
            .change_client_keys(KeyChange {
                put: Some((name.as_str(), &sealed)),
                delete: &[pending.as_str()],
            })
            .await?;
        self.forget(&pending);
        tracing::info!(
            client = %name,
            thumbprint = key.thumbprint(),
            "took over the key made for this issuer by `quack auth register --print`"
        );
        Ok(Some(key))
    }

    fn slot(&self, name: &ClientKeyName) -> Slot {
        let mut cache = cache().lock().unwrap_or_else(PoisonError::into_inner);
        let slot = cache
            .entry((self.config.data_dir().to_path_buf(), name.clone()))
            .or_default();
        Arc::clone(slot)
    }

    /// The client's key: loaded once per process, and made and stored the
    /// first time any process needs it.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored key does not open (altered, or
    /// sealed for another name), is not a P-256 key, or the row cannot be
    /// read or written.
    pub async fn key(&self, name: &ClientKeyName) -> Result<Arc<ClientKey>> {
        let slot = self.slot(name);
        // Boxed: opening `control.db` is a deep future, and this one sits
        // inside every model request's.
        slot.get_or_try_init(|| async { Box::pin(self.load_or_create(name)).await.map(Arc::new) })
            .await
            .cloned()
    }

    /// The client's key when one is stored and opens, without making one:
    /// what `quack auth status` reports.
    ///
    /// # Errors
    ///
    /// As [`ClientKeys::key`].
    pub async fn existing(&self, name: &ClientKeyName) -> Result<Option<Arc<ClientKey>>> {
        let slot = self.slot(name);
        if let Some(key) = slot.get() {
            return Ok(Some(Arc::clone(key)));
        }
        let Some(key) = Box::pin(self.load(name)).await? else {
            return Ok(None);
        };
        Ok(Some(Arc::clone(
            slot.get_or_init(|| async { Arc::new(key) }).await,
        )))
    }

    /// Start replacing the key of a client quack did not register itself:
    /// its replacement, made (or, when a replacement already waits, loaded)
    /// under [`ClientKeyName::replacement`], and its key in use, which
    /// stays in use until [`ClientKeys::activate_replacement`]. The issuer
    /// should hold both while the operator switches.
    ///
    /// # Errors
    ///
    /// Returns an error when a key cannot be loaded or made.
    pub async fn stage_replacement(
        &self,
        client: &ClientKeyName,
    ) -> Result<(Option<Arc<ClientKey>>, Arc<ClientKey>)> {
        let current = self.existing(client).await?;
        let next = self.key(&client.replacement()).await?;
        Ok((current, next))
    }

    /// Put the waiting replacement in place of the client's key, in one
    /// transaction; the old key is gone afterwards.
    ///
    /// # Errors
    ///
    /// Returns an error when no replacement waits, or the keys cannot be
    /// written.
    pub async fn activate_replacement(&self, client: &ClientKeyName) -> Result<Arc<ClientKey>> {
        let next = client.replacement();
        let der = self.stored_der(&next).await?.ok_or_else(|| {
            key_error(format!(
                "no replacement key waits for {client}; `quack auth jwks --rotate` makes one"
            ))
        })?;
        let key = ClientKey::from_pkcs8(&der)?;
        let sealed = self.seal(client, &der).await?;
        self.control()
            .await?
            .change_client_keys(KeyChange {
                put: Some((client.as_str(), &sealed)),
                delete: &[next.as_str()],
            })
            .await?;
        self.forget(client);
        self.forget(&next);
        tracing::info!(client = %client, thumbprint = key.thumbprint(), "the replacement client key is now in use");
        Ok(Arc::new(key))
    }

    /// The stored key, or `None` when there is none or the vault key that
    /// sealed it is gone.
    async fn load(&self, name: &ClientKeyName) -> Result<Option<ClientKey>> {
        self.stored_der(name)
            .await?
            .map(|der| ClientKey::from_pkcs8(&der))
            .transpose()
    }

    /// The stored key, bypassing the process cache; made and stored when
    /// there is none, and replaced when the vault key that sealed it is
    /// gone.
    pub(crate) async fn load_or_create(&self, name: &ClientKeyName) -> Result<ClientKey> {
        let control = self.control().await?;
        let owner = SealedOwner::ClientKey(name.as_str());
        let stored = control.sealed(owner).await?;
        if let Some(sealed) = &stored {
            match self
                .vault
                .open(Purpose::ClientKey, name.as_str(), sealed)
                .await?
            {
                Opened::Plaintext(der) => return ClientKey::from_pkcs8(&der),
                Opened::KeyGone => {}
            }
        } else if let Some(key) = self.adopt_pending(name).await? {
            return Ok(key);
        }
        let (key, der) = ClientKey::generate()?;
        let sealed = self
            .vault
            .seal(Purpose::ClientKey, name.as_str(), &der)
            .await?;
        if stored.is_some() {
            control.put_sealed(owner, &sealed).await?;
            tracing::warn!(
                client = %name,
                thumbprint = key.thumbprint(),
                "the vault key that sealed this client's private_key_jwt key is gone, so a new key was made; register its public key with the issuer (`quack auth jwks`), or the issuer refuses quack's client assertions"
            );
            return Ok(key);
        }
        if control.add_sealed(owner, &sealed).await? {
            tracing::info!(
                client = %name,
                thumbprint = key.thumbprint(),
                "made a new private_key_jwt client key; register its public key with the issuer (`quack auth jwks`)"
            );
            return Ok(key);
        }
        // Another process stored its key first: use that one.
        let sealed = control
            .sealed(owner)
            .await?
            .ok_or_else(|| key_error(format!("the key for {name} vanished while it was made")))?;
        match self
            .vault
            .open(Purpose::ClientKey, name.as_str(), &sealed)
            .await?
        {
            Opened::Plaintext(der) => ClientKey::from_pkcs8(&der),
            Opened::KeyGone => Err(key_error(format!(
                "the key another process stored for {name} is sealed under another vault key"
            ))),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
