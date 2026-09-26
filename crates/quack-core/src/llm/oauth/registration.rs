//! Clients quack registers with an issuer itself: OAuth 2.0 Dynamic Client
//! Registration (RFC 7591) to create one, and its management protocol (RFC
//! 7592) to read, update, and delete it afterwards.
//!
//! One registration serves every section of the configuration that names a
//! client at the issuer without a `client_id`: `[server.oidc]` and any
//! `[providers.NAME.oauth]`. That is the setup to use with an issuer such as
//! Vouch, where a single client both signs people in and exchanges their
//! tokens. The registration is kept in `control.db` (`client_registrations`)
//! under the issuer's name ([`RegistrationName`]), which is how those
//! sections find their `client_id` before they know it, together with the
//! `registration_access_token` (sealed by the vault for
//! [`Purpose::RegistrationToken`]) and the `registration_client_uri` RFC 7592
//! manages it at.
//!
//! Every registered client authenticates with `private_key_jwt`. Its key is
//! made before the registration, since the request carries the public half,
//! under the issuer's name alone ([`ClientKeyName::pending`]), and moves to
//! `<issuer> <client_id>` in the same transaction that stores the
//! registration. A rotation sends the whole current registration back with
//! the new key (RFC 7592 replaces all metadata on update) and replaces the
//! stored key only once the issuer has accepted it.

use std::collections::BTreeSet;
use std::sync::Arc;

use oauth2::url::Url;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use super::OAuthHttp;
use super::client_key::{ALGORITHM, ClientKey, ClientKeyName, ClientKeys, PublicJwks};
use crate::config::{ClientAuth, Config, Exchange, Grant, OAuthConfig, ProviderName};
use crate::error::{Error, Result};
use crate::storage::control::{KeyChange, RegistrationRow};
use crate::vault::{Opened, Purpose};

/// The metadata RFC 7592 says a client must leave out of an update: the
/// server's own bookkeeping, which it sets.
const SERVER_MANAGED: [&str; 4] = [
    "registration_access_token",
    "registration_client_uri",
    "client_secret_expires_at",
    "client_id_issued_at",
];

fn registration_error(message: impl std::fmt::Display) -> Error {
    Error::Llm(format!("client registration: {message}"))
}

/// Which registration a client uses: the issuer's, without a trailing
/// slash. One per issuer, since one registered client serves every section
/// at the issuer that names no `client_id`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RegistrationName(String);

impl RegistrationName {
    #[must_use]
    pub fn new(issuer_url: &str) -> Self {
        Self(issuer_url.trim().trim_end_matches('/').to_owned())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RegistrationName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A section of the configuration that names an OAuth client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientSection {
    /// `[server.oidc]`, the sign-in client.
    SignIn,
    /// `[providers.NAME.oauth]`.
    Provider(ProviderName),
}

impl std::fmt::Display for ClientSection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SignIn => f.write_str("[server.oidc]"),
            Self::Provider(name) => write!(f, "[providers.{name}.oauth]"),
        }
    }
}

/// A section whose client quack registers: it names no `client_id`.
#[derive(Debug, Clone)]
pub struct RegisteredSection {
    pub section: ClientSection,
    pub issuer: RegistrationName,
}

/// Every section of `config` that names no `client_id`, sign-in first.
#[must_use]
pub fn registered_sections(config: &Config) -> Vec<RegisteredSection> {
    let sign_in = config
        .server
        .oidc
        .as_ref()
        .filter(|oidc| oidc.client_id.is_none())
        .map(|oidc| RegisteredSection {
            section: ClientSection::SignIn,
            issuer: RegistrationName::new(&oidc.issuer_url),
        });
    let providers = config.providers.iter().filter_map(|(name, provider)| {
        provider
            .auth
            .oauth()
            .filter(|oauth| oauth.client_id.is_none())
            .map(|oauth| RegisteredSection {
                section: ClientSection::Provider(name.clone()),
                issuer: RegistrationName::new(&oauth.issuer_url),
            })
    });
    sign_in.into_iter().chain(providers).collect()
}

/// The issuer `quack auth register` registers at: `explicit`, or the one
/// issuer every section without a `client_id` shares.
///
/// # Errors
///
/// Returns [`Error::Config`] when no section lacks a `client_id` (there is
/// nothing to register), `explicit` names an issuer none of them uses, or
/// they use several issuers and none was named.
pub fn issuer_to_register(config: &Config, explicit: Option<&str>) -> Result<RegistrationName> {
    let sections = registered_sections(config);
    let issuers: BTreeSet<&str> = sections.iter().map(|s| s.issuer.as_str()).collect();
    if let Some(explicit) = explicit {
        let wanted = RegistrationName::new(explicit);
        if issuers.contains(wanted.as_str()) {
            return Ok(wanted);
        }
        return Err(Error::Config(format!(
            "no [server.oidc] or [providers.NAME.oauth] section at {wanted} leaves client_id out; remove client_id from the sections the registered client should serve"
        )));
    }
    let mut iter = issuers.iter();
    match (iter.next(), iter.next()) {
        (Some(only), None) => Ok(RegistrationName::new(only)),
        (None, _) => Err(Error::Config(String::from(
            "every configured OAuth client names its client_id, so there is nothing to register; leave client_id out of the sections the registered client should serve, with client_auth = \"private_key_jwt\"",
        ))),
        (Some(_), Some(_)) => Err(Error::Config(format!(
            "clients without a client_id use several issuers ({}); name one with --issuer",
            issuers.iter().copied().collect::<Vec<_>>().join(", ")
        ))),
    }
}

/// The client metadata of a registration (RFC 7591 section 2), as quack
/// sends it. It never asks for `DPoP`- or certificate-bound tokens: a bound
/// token must be presented with a proof on every request, which the model
/// APIs an on-behalf-of token is for cannot take, and an issuer such as
/// Vouch treats a client that asks for them as a FAPI client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientMetadata {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub redirect_uris: Vec<String>,
    pub grant_types: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub response_types: Vec<String>,
    pub token_endpoint_auth_method: String,
    pub token_endpoint_auth_signing_alg: String,
    pub jwks: PublicJwks,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    pub client_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub application_type: Option<String>,
}

/// What one section's client needs from the registration.
#[derive(Default)]
struct Needs {
    grants: Vec<String>,
    /// The sign-in client's https callback.
    web_redirects: Vec<String>,
    /// A provider's loopback listener, for its browser login.
    loopback_redirects: Vec<(ClientSection, String)>,
    scopes: Vec<String>,
}

impl Needs {
    fn grant(&mut self, grant: &str) {
        if !self.grants.iter().any(|g| g == grant) {
            self.grants.push(grant.to_owned());
        }
    }

    fn scopes(&mut self, scopes: &[String]) {
        for scope in scopes {
            if !self.scopes.contains(scope) {
                self.scopes.push(scope.clone());
            }
        }
    }

    fn provider(&mut self, name: &ProviderName, oauth: &OAuthConfig) {
        self.grant(oauth.grant_type());
        match oauth.grant {
            Grant::AuthorizationCode => self.loopback_redirects.push((
                ClientSection::Provider(name.clone()),
                oauth.redirect_uri.clone(),
            )),
            // quack's own token rides along as the actor.
            Grant::OnBehalfOf if oauth.actor && oauth.exchange == Exchange::TokenExchange => {
                self.grant("client_credentials");
            }
            Grant::DeviceCode | Grant::ClientCredentials | Grant::OnBehalfOf => {}
        }
        self.scopes(&oauth.scopes);
    }
}

/// The metadata for registering the client every section at `issuer`
/// without a `client_id` uses: the union of their grants and scopes, the
/// sign-in callback, `private_key_jwt` with `jwks`, and `client_name`.
///
/// `refresh_token` is asked for only when a scope is `offline_access`, the
/// scope that makes an issuer return refresh tokens; an issuer such as
/// Vouch, which issues none, is not asked for a grant it would refuse.
///
/// # Errors
///
/// Returns [`Error::Config`] when a section at the issuer does not use
/// `private_key_jwt`, or when the sign-in callback and a provider's
/// loopback login would share the registration: `OpenID` Connect Dynamic
/// Client Registration 1.0 section 2 has a web client's redirects use https
/// and not localhost, and a native client's the reverse, so one client
/// cannot take both.
pub fn metadata_for(
    config: &Config,
    issuer: &RegistrationName,
    client_name: &str,
    jwks: PublicJwks,
) -> Result<ClientMetadata> {
    let mut needs = Needs::default();
    for section in registered_sections(config) {
        if section.issuer != *issuer {
            continue;
        }
        let auth = match &section.section {
            ClientSection::SignIn => {
                let Some(oidc) = &config.server.oidc else {
                    continue;
                };
                needs.grant("authorization_code");
                needs.web_redirects.push(oidc.redirect_uri.clone());
                needs.scopes(&oidc.scopes);
                oidc.client_auth
            }
            ClientSection::Provider(name) => {
                let Some(oauth) = config.providers.get(name).and_then(|p| p.auth.oauth()) else {
                    continue;
                };
                needs.provider(name, oauth);
                oauth.client_auth
            }
        };
        if auth != ClientAuth::PrivateKeyJwt {
            return Err(Error::Config(format!(
                "{}: a registered client authenticates with client_auth = \"private_key_jwt\"",
                section.section
            )));
        }
    }
    if needs.grants.is_empty() {
        return Err(Error::Config(format!(
            "no section at {issuer} leaves client_id out, so nothing there is registered"
        )));
    }
    if let (Some(_), Some((section, _))) = (
        needs.web_redirects.first(),
        needs.loopback_redirects.first(),
    ) {
        return Err(Error::Config(format!(
            "{section} logs in through a loopback redirect, which cannot share a registration with the [server.oidc] callback (a web client takes https redirects only); give {section} a client_id of its own"
        )));
    }
    if needs.scopes.iter().any(|s| s == "offline_access") {
        needs.grant("refresh_token");
    }
    let (redirect_uris, application_type) = if needs.web_redirects.is_empty() {
        let loopback: Vec<String> = needs
            .loopback_redirects
            .into_iter()
            .map(|(_, r)| r)
            .collect();
        let native = (!loopback.is_empty()).then(|| String::from("native"));
        (loopback, native)
    } else {
        (needs.web_redirects, Some(String::from("web")))
    };
    let response_types = if needs.grants.iter().any(|g| g == "authorization_code") {
        vec![String::from("code")]
    } else {
        Vec::new()
    };
    Ok(ClientMetadata {
        redirect_uris,
        grant_types: needs.grants,
        response_types,
        token_endpoint_auth_method: ClientAuth::PrivateKeyJwt.as_str().to_owned(),
        token_endpoint_auth_signing_alg: String::from(ALGORITHM),
        jwks,
        scope: (!needs.scopes.is_empty()).then(|| needs.scopes.join(" ")),
        client_name: client_name.to_owned(),
        application_type,
    })
}

/// An issuer's answer to a registration or an update (RFC 7591 3.2.1, RFC
/// 7592 3), as far as quack keeps it.
#[derive(Debug, Deserialize)]
struct Answer {
    client_id: String,
    registration_access_token: Option<String>,
    registration_client_uri: Option<String>,
    token_endpoint_auth_method: Option<String>,
}

/// An issuer's refusal (RFC 7591 3.2.2).
#[derive(Debug, Default, Deserialize)]
struct Refusal {
    error: Option<String>,
    error_description: Option<String>,
}

fn refusal(what: &str, status: reqwest::StatusCode, body: &[u8]) -> Error {
    let refusal: Refusal = serde_json::from_slice(body).unwrap_or_default();
    registration_error(format!(
        "the issuer refused {what} ({status}): {}{}",
        refusal.error.as_deref().unwrap_or("no error code"),
        refusal
            .error_description
            .map_or(String::new(), |d| format!(" ({d})"))
    ))
}

/// What a registration came to.
#[derive(Debug)]
pub struct Registered {
    pub client_id: String,
    /// The key's RFC 7638 thumbprint.
    pub thumbprint: String,
    /// Whether RFC 7592 can manage the client: the issuer returned a
    /// `registration_access_token` and a `registration_client_uri`.
    pub manageable: bool,
    /// The authentication method the issuer registered, when it is not
    /// `private_key_jwt`.
    pub other_auth_method: Option<String>,
    /// What became of the client it replaced, with `--replace`.
    pub replaced: Option<Removal>,
}

/// What became of a registered client at the issuer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Removal {
    /// RFC 7592 deleted it.
    Deleted { client_id: String },
    /// The issuer no longer knows it or the registration token (`401`,
    /// `403`, or `404`): nothing is left to delete.
    AlreadyGone { client_id: String, status: u16 },
    /// The issuer returned nothing to manage it with: delete it in the
    /// issuer's console.
    Unmanaged { client_id: String },
}

/// What a rotation came to.
#[derive(Debug)]
pub struct Rotated {
    pub client_id: String,
    pub old_thumbprint: Option<String>,
    pub new_thumbprint: String,
    /// Whether the issuer issued a new `registration_access_token`.
    pub new_registration_token: bool,
}

/// What reading a registration back (RFC 7592 2.1) found, for `quack
/// doctor`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadBack {
    /// The issuer still describes the client.
    Readable { client_id: String },
    /// The issuer returned nothing to read it with.
    Unmanaged { client_id: String },
    /// The issuer refused (a deleted client, a revoked registration token).
    Refused { client_id: String, status: u16 },
    /// It answered with another client.
    Mismatch { stored: String, read: String },
}

impl ClientKeys {
    /// The registration kept for `issuer`, if any.
    ///
    /// # Errors
    ///
    /// Returns an error when `control.db` cannot be read.
    pub async fn registration(&self, issuer: &RegistrationName) -> Result<Option<RegistrationRow>> {
        self.control().await?.registration(issuer.as_str()).await
    }

    /// The `client_id` quack registered at `issuer`, for `section`, which
    /// names none.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] naming `quack auth register` when no
    /// client is registered there.
    pub async fn registered_client_id(&self, issuer: &str, section: &str) -> Result<String> {
        let name = RegistrationName::new(issuer);
        match self.registration(&name).await? {
            Some(row) => Ok(row.client_id),
            None => Err(Error::Config(format!(
                "{section} names no client_id, and no client is registered at {name}; run `quack auth register`, or set client_id"
            ))),
        }
    }

    /// The registration's `registration_access_token` and
    /// `registration_client_uri`, when the issuer returned both and the
    /// token still opens.
    async fn management(&self, row: &RegistrationRow) -> Result<Option<(SecretString, String)>> {
        let (Some(sealed), Some(uri)) = (&row.token, &row.registration_client_uri) else {
            return Ok(None);
        };
        match self
            .vault()
            .open(Purpose::RegistrationToken, &row.name, sealed)
            .await?
        {
            Opened::Plaintext(token) => {
                let token = String::from_utf8(token)
                    .map_err(|_| registration_error("the stored registration token is not text"))?;
                Ok(Some((SecretString::from(token), uri.clone())))
            }
            Opened::KeyGone => Err(registration_error(format!(
                "the registration token for {} was sealed under a vault key that is gone, so quack can no longer manage client {}; delete it in the issuer's console",
                row.name, row.client_id
            ))),
        }
    }
}

/// Registers, rotates, reads, and deletes the clients of one data
/// directory.
pub struct Registrar {
    keys: ClientKeys,
    http: OAuthHttp,
}

impl std::fmt::Debug for Registrar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registrar")
            .field("keys", &self.keys)
            .finish_non_exhaustive()
    }
}

impl Registrar {
    /// # Errors
    ///
    /// Returns an error when the HTTP client cannot be built.
    pub fn new(keys: ClientKeys) -> Result<Self> {
        Ok(Self {
            keys,
            http: OAuthHttp::new()?,
        })
    }

    /// The keys and registrations it manages.
    #[must_use]
    pub const fn keys(&self) -> &ClientKeys {
        &self.keys
    }

    /// The key the next registration at `issuer` carries: made, or loaded
    /// when an earlier attempt (or `--print`) made it.
    ///
    /// # Errors
    ///
    /// Returns an error when the key cannot be loaded or made.
    pub async fn pending_key(&self, issuer: &RegistrationName) -> Result<Arc<ClientKey>> {
        self.keys
            .key(&ClientKeyName::pending(issuer.as_str()))
            .await
    }

    /// Register `metadata` at the issuer's `registration_endpoint`, with
    /// `initial_token` as the bearer when given (the issuer's initial access
    /// token, or at Vouch a person's access token, which makes the client
    /// theirs), and keep the registration with its key. An existing
    /// registration is refused unless `replace`, which deletes the old
    /// client first (RFC 7592).
    ///
    /// # Errors
    ///
    /// Returns an error when a client is registered and `replace` is off,
    /// `metadata` does not carry the pending key, the issuer lists no
    /// registration endpoint or refuses, or the result cannot be stored.
    pub async fn register(
        &self,
        issuer: &RegistrationName,
        metadata: &ClientMetadata,
        initial_token: Option<&SecretString>,
        replace: bool,
    ) -> Result<Registered> {
        let (existing, der, key, endpoint) = self.prepare(issuer, metadata, replace).await?;
        let pending = ClientKeyName::pending(issuer.as_str());
        let replaced = match &existing {
            Some(row) => Some(self.remove_at_issuer(row).await?),
            None => None,
        };
        let body = serde_json::to_value(metadata)?;
        let (status, answer) = self
            .http
            .send_json(reqwest::Method::POST, &endpoint, initial_token, Some(&body))
            .await?;
        if !status.is_success() {
            return Err(refusal("the registration", status, &answer));
        }
        let answer: Answer = serde_json::from_slice(&answer).map_err(|e| {
            registration_error(format!("the issuer's answer has no client_id: {e}"))
        })?;
        let client = ClientKeyName::new(issuer.as_str(), &answer.client_id);
        let sealed_key = self.keys.seal(&client, &der).await?;
        let token = self
            .seal_token(issuer, answer.registration_access_token.as_deref())
            .await?;
        let row = RegistrationRow {
            name: issuer.as_str().to_owned(),
            client_id: answer.client_id.clone(),
            registration_client_uri: answer.registration_client_uri.clone(),
            token,
        };
        let old = existing
            .as_ref()
            .map(|old| ClientKeyName::new(issuer.as_str(), &old.client_id))
            .filter(|old| *old != client);
        let mut delete = vec![pending.as_str()];
        if let Some(old) = &old {
            delete.push(old.as_str());
        }
        self.keys
            .control()
            .await?
            .save_registration(
                &row,
                KeyChange {
                    put: Some((client.as_str(), &sealed_key)),
                    delete: &delete,
                },
            )
            .await
            .map_err(|e| {
                registration_error(format!(
                    "{issuer} registered client {}, but quack could not keep it ({e}); delete that client in the issuer's console and register again",
                    answer.client_id
                ))
            })?;
        for name in [Some(&pending), Some(&client), old.as_ref()]
            .into_iter()
            .flatten()
        {
            self.keys.forget(name);
        }
        tracing::info!(issuer = %issuer, client_id = %answer.client_id, thumbprint = key.thumbprint(), "registered a client");
        Ok(Registered {
            manageable: row.token.is_some() && row.registration_client_uri.is_some(),
            client_id: answer.client_id,
            thumbprint: key.thumbprint().to_owned(),
            other_auth_method: answer
                .token_endpoint_auth_method
                .filter(|m| m != ClientAuth::PrivateKeyJwt.as_str()),
            replaced,
        })
    }

    /// What a registration starts from: the registration it replaces, the
    /// pending key (its PKCS#8 document too), which `metadata` must carry,
    /// and the issuer's registration endpoint.
    async fn prepare(
        &self,
        issuer: &RegistrationName,
        metadata: &ClientMetadata,
        replace: bool,
    ) -> Result<(Option<RegistrationRow>, Vec<u8>, ClientKey, String)> {
        let existing = self.keys.registration(issuer).await?;
        if let Some(row) = &existing
            && !replace
        {
            return Err(Error::Config(format!(
                "client {} is already registered at {issuer}; `quack auth register --replace` deletes it and registers a new one",
                row.client_id
            )));
        }
        let pending = ClientKeyName::pending(issuer.as_str());
        let der = self.keys.stored_der(&pending).await?.ok_or_else(|| {
            registration_error(format!("no pending key for {issuer}; it was just made"))
        })?;
        let key = ClientKey::from_pkcs8(&der)?;
        if metadata.jwks != key.jwks() {
            return Err(registration_error(
                "the metadata does not carry the key made for this registration",
            ));
        }
        let endpoint = self
            .http
            .discover(issuer.as_str())
            .await?
            .registration
            .ok_or_else(|| {
                registration_error(format!(
                    "{issuer} lists no registration_endpoint, so it takes no RFC 7591 registrations; register the client in its console and set client_id"
                ))
            })?;
        Ok((existing, der, key, endpoint))
    }

    /// A registration access token sealed for keeping, when there is one.
    async fn seal_token(
        &self,
        issuer: &RegistrationName,
        token: Option<&str>,
    ) -> Result<Option<crate::vault::Sealed>> {
        match token {
            Some(token) => Ok(Some(
                self.keys
                    .vault()
                    .seal(
                        Purpose::RegistrationToken,
                        issuer.as_str(),
                        token.as_bytes(),
                    )
                    .await?,
            )),
            None => Ok(None),
        }
    }

    /// Delete the client `row` records at the issuer (RFC 7592 2.3). A
    /// client the issuer no longer knows is already gone; any other
    /// refusal stops here, so a replacement never leaves two clients.
    async fn remove_at_issuer(&self, row: &RegistrationRow) -> Result<Removal> {
        let client_id = row.client_id.clone();
        let Some((token, uri)) = self.keys.management(row).await? else {
            return Ok(Removal::Unmanaged { client_id });
        };
        let (status, body) = self
            .http
            .send_json(reqwest::Method::DELETE, &uri, Some(&token), None)
            .await?;
        match status.as_u16() {
            200..=299 => Ok(Removal::Deleted { client_id }),
            code @ (401 | 403 | 404) => Ok(Removal::AlreadyGone {
                client_id,
                status: code,
            }),
            _ => Err(refusal("deleting the client", status, &body)),
        }
    }

    /// Replace the key of the client registered at `issuer`: a new key, and
    /// the whole registration as the issuer describes it (RFC 7592 2.1)
    /// sent back with the new `jwks`, since an update replaces every field
    /// (2.2). The stored key is replaced only after the issuer accepts, so
    /// a refusal leaves the old key in use; a new registration token, when
    /// the issuer rotates it, is kept with it.
    ///
    /// # Errors
    ///
    /// Returns an error when nothing is registered, the issuer returned
    /// nothing to manage the client with, it refuses the read or the
    /// update, or the result cannot be stored.
    pub async fn rotate(&self, issuer: &RegistrationName) -> Result<Rotated> {
        let row = self.keys.registration(issuer).await?.ok_or_else(|| {
            Error::Config(format!(
                "no client is registered at {issuer}; `quack auth register` registers one"
            ))
        })?;
        let (token, uri) = self.keys.management(&row).await?.ok_or_else(|| {
            registration_error(format!(
                "{issuer} returned no registration_access_token for client {}, so quack cannot update it (RFC 7592); register the new key in the issuer's console",
                row.client_id
            ))
        })?;
        let current = self.read_metadata(&uri, &token).await?;
        let client = ClientKeyName::new(issuer.as_str(), &row.client_id);
        let old_thumbprint = self
            .keys
            .existing(&client)
            .await?
            .map(|key| key.thumbprint().to_owned());
        // Made and sealed before the update, so after the issuer accepts it
        // only the local write remains.
        let (key, der) = ClientKey::generate()?;
        let sealed_key = self.keys.seal(&client, &der).await?;
        let update = updated_metadata(current, &row.client_id, &key.jwks())?;
        let (status, answer) = self
            .http
            .send_json(reqwest::Method::PUT, &uri, Some(&token), Some(&update))
            .await?;
        if !status.is_success() {
            return Err(refusal("the new key", status, &answer));
        }
        let answer: Answer = serde_json::from_slice(&answer).map_err(|e| {
            registration_error(format!(
                "the issuer's answer to the update has no client_id: {e}"
            ))
        })?;
        if answer.client_id != row.client_id {
            return Err(registration_error(format!(
                "the issuer answered the update for client {}, not {}",
                answer.client_id, row.client_id
            )));
        }
        let new_token = answer
            .registration_access_token
            .as_deref()
            .filter(|new| *new != token.expose_secret());
        let mut updated = row.clone();
        if let Some(sealed) = self.seal_token(issuer, new_token).await? {
            updated.token = Some(sealed);
        }
        if let Some(uri) = answer.registration_client_uri {
            updated.registration_client_uri = Some(uri);
        }
        self.keys
            .control()
            .await?
            .save_registration(
                &updated,
                KeyChange {
                    put: Some((client.as_str(), &sealed_key)),
                    delete: &[],
                },
            )
            .await
            .map_err(|e| {
                registration_error(format!(
                    "the issuer accepted the new key for client {}, but quack could not keep it ({e}); run `quack auth jwks --rotate` again",
                    row.client_id
                ))
            })?;
        self.keys.forget(&client);
        tracing::info!(issuer = %issuer, client_id = %row.client_id, thumbprint = key.thumbprint(), "rotated a registered client's key");
        Ok(Rotated {
            client_id: row.client_id,
            old_thumbprint,
            new_thumbprint: key.thumbprint().to_owned(),
            new_registration_token: new_token.is_some(),
        })
    }

    /// The registration as the issuer describes it now (RFC 7592 2.1).
    async fn read_metadata(
        &self,
        uri: &str,
        token: &SecretString,
    ) -> Result<serde_json::Map<String, serde_json::Value>> {
        let (status, current) = self
            .http
            .send_json(reqwest::Method::GET, uri, Some(token), None)
            .await?;
        if !status.is_success() {
            return Err(refusal("reading the registration", status, &current));
        }
        serde_json::from_slice(&current).map_err(|e| {
            registration_error(format!("the registration read back is not an object: {e}"))
        })
    }

    /// Delete the client registered at `issuer` (RFC 7592 2.3), then its
    /// registration and its keys.
    ///
    /// # Errors
    ///
    /// Returns an error when nothing is registered, or the issuer refuses
    /// for a reason other than no longer knowing the client (nothing is
    /// then forgotten).
    pub async fn unregister(&self, issuer: &RegistrationName) -> Result<Removal> {
        let row = self
            .keys
            .registration(issuer)
            .await?
            .ok_or_else(|| Error::Config(format!("no client is registered at {issuer}")))?;
        let removal = self.remove_at_issuer(&row).await?;
        let client = ClientKeyName::new(issuer.as_str(), &row.client_id);
        let pending = ClientKeyName::pending(issuer.as_str());
        self.keys
            .control()
            .await?
            .delete_registration(
                issuer.as_str(),
                KeyChange {
                    put: None,
                    delete: &[client.as_str(), pending.as_str()],
                },
            )
            .await?;
        self.keys.forget(&client);
        self.keys.forget(&pending);
        Ok(removal)
    }

    /// Read the registration at `issuer` back from the issuer (RFC 7592
    /// 2.1); `None` when nothing is registered.
    ///
    /// # Errors
    ///
    /// Returns an error when the issuer cannot be reached or the stored
    /// token does not open.
    pub async fn read(&self, issuer: &RegistrationName) -> Result<Option<ReadBack>> {
        let Some(row) = self.keys.registration(issuer).await? else {
            return Ok(None);
        };
        let Some((token, uri)) = self.keys.management(&row).await? else {
            return Ok(Some(ReadBack::Unmanaged {
                client_id: row.client_id,
            }));
        };
        let (status, body) = self
            .http
            .send_json(reqwest::Method::GET, &uri, Some(&token), None)
            .await?;
        if !status.is_success() {
            return Ok(Some(ReadBack::Refused {
                client_id: row.client_id,
                status: status.as_u16(),
            }));
        }
        let answer: Answer = serde_json::from_slice(&body).map_err(|e| {
            registration_error(format!("the registration read back has no client_id: {e}"))
        })?;
        Ok(Some(if answer.client_id == row.client_id {
            ReadBack::Readable {
                client_id: row.client_id,
            }
        } else {
            ReadBack::Mismatch {
                stored: row.client_id,
                read: answer.client_id,
            }
        }))
    }
}

/// The update that replaces a client's key (RFC 7592 2.2): everything the
/// issuer describes, without the fields it manages itself, with the
/// `client_id` and the new `jwks` in place of any `jwks_uri`.
fn updated_metadata(
    mut current: serde_json::Map<String, serde_json::Value>,
    client_id: &str,
    jwks: &PublicJwks,
) -> Result<serde_json::Value> {
    for field in SERVER_MANAGED {
        current.remove(field);
    }
    current.remove("jwks_uri");
    current.insert(
        String::from("client_id"),
        serde_json::Value::String(client_id.to_owned()),
    );
    current.insert(String::from("jwks"), serde_json::to_value(jwks)?);
    Ok(serde_json::Value::Object(current))
}

/// Every section without a `client_id`, with the id its registration
/// gives it, or the error naming `quack auth register` for the first one
/// without: what `quack serve` checks before it starts.
///
/// # Errors
///
/// Returns [`Error::Config`] for a section whose issuer has no
/// registration, or an error reading `control.db`.
pub async fn resolve_registered(
    config: &Config,
    keys: &ClientKeys,
) -> Result<Vec<(RegisteredSection, String)>> {
    let mut resolved = Vec::new();
    for section in registered_sections(config) {
        let client_id = keys
            .registered_client_id(section.issuer.as_str(), &section.section.to_string())
            .await?;
        resolved.push((section, client_id));
    }
    Ok(resolved)
}

impl OAuthHttp {
    /// Send a JSON request for client registration, with `bearer` as the
    /// `Authorization` when given; the status and the body come back
    /// whatever they are.
    ///
    /// # Errors
    ///
    /// Returns an error when the URL is not one, or the request cannot be
    /// sent or read.
    async fn send_json(
        &self,
        method: reqwest::Method,
        url: &str,
        bearer: Option<&SecretString>,
        body: Option<&serde_json::Value>,
    ) -> Result<(reqwest::StatusCode, Vec<u8>)> {
        let parsed = Url::parse(url)
            .map_err(|e| registration_error(format!("'{url}' is not a URL: {e}")))?;
        let mut request = self
            .0
            .request(method, parsed)
            .header(reqwest::header::ACCEPT, "application/json");
        if let Some(bearer) = bearer {
            request = request.bearer_auth(bearer.expose_secret());
        }
        if let Some(body) = body {
            request = request
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(serde_json::to_vec(body)?);
        }
        let response = request
            .send()
            .await
            .map_err(|e| registration_error(format!("the request to {url} failed: {e}")))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| registration_error(format!("the answer from {url} was cut short: {e}")))?;
        Ok((status, bytes.to_vec()))
    }
}
