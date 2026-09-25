//! How quack authenticates as a client at an issuer's token endpoint (and
//! at its pushed authorization request endpoint): nothing but its
//! `client_id` for a public client, its secret, or a client assertion
//! signed with its own key (`private_key_jwt`).
//!
//! The `oauth2` crate knows only secrets, so an assertion client is given
//! to it as a public one, which puts `client_id` in the body and never an
//! HTTP Basic header, and [`OAuthHttp::sender`](super::OAuthHttp::sender)
//! adds a newly signed assertion to each request it sends. The issuer
//! spends each assertion's `jti`, so a device-code poll or any other retry
//! must never resend one.

use std::sync::Arc;

use oauth2::{AuthType, ClientSecret};
use secrecy::{ExposeSecret, SecretString};

use super::client_key::{ASSERTION_TYPE, ClientKey, ClientKeyName, ClientKeys};
use crate::config::ClientAuth;
use crate::error::Result;

/// A client's proof of who it is, ready for a request.
#[derive(Debug, Clone)]
pub(crate) enum Credential {
    /// A public client: its `client_id` only.
    Public,
    /// A confidential client's secret, in the body or, with `basic`, in an
    /// HTTP Basic header (RFC 6749 2.3.1).
    Secret { secret: SecretString, basic: bool },
    /// A signed client assertion (RFC 7523 2.2), new for every request.
    Assertion(Assertion),
}

/// What a client assertion is made from: the key, the client it names as
/// `iss` and `sub`, and the issuer it is addressed to as `aud`.
#[derive(Debug, Clone)]
pub(crate) struct Assertion {
    key: Arc<ClientKey>,
    client_id: String,
    audience: String,
}

/// What building a client's credential takes: its configuration.
pub(crate) struct Registration<'a> {
    pub(crate) auth: ClientAuth,
    pub(crate) client_id: &'a str,
    /// The configured issuer, which names the key and backs up `audience`.
    pub(crate) issuer_url: &'a str,
    /// The issuer identifier from discovery, the assertions' `aud`.
    pub(crate) audience: Option<&'a str>,
    /// The secret from `client_secret_env`, when one is named.
    pub(crate) secret: Option<String>,
}

impl Credential {
    /// The credential `client` names: an assertion for `private_key_jwt`
    /// (its key loaded, or made on first use, from `keys`), else the secret
    /// when there is one, else none.
    ///
    /// # Errors
    ///
    /// Returns an error when the client key cannot be loaded or made.
    pub(crate) async fn of(client: Registration<'_>, keys: &ClientKeys) -> Result<Self> {
        match client.auth {
            ClientAuth::PrivateKeyJwt => {
                let name = ClientKeyName::new(client.issuer_url, client.client_id);
                let key = keys.key(&name).await?;
                Ok(Self::Assertion(Assertion {
                    key,
                    client_id: client.client_id.to_owned(),
                    audience: client.audience.unwrap_or(client.issuer_url).to_owned(),
                }))
            }
            ClientAuth::ClientSecretPost | ClientAuth::ClientSecretBasic => {
                Ok(client.secret.map_or(Self::Public, |secret| Self::Secret {
                    secret: SecretString::from(secret),
                    basic: client.auth == ClientAuth::ClientSecretBasic,
                }))
            }
        }
    }

    /// How the `oauth2` crate should present the client: HTTP Basic only
    /// for a secret that goes there; the body otherwise, which for a client
    /// without a secret carries just its `client_id`.
    pub(crate) const fn auth_type(&self) -> AuthType {
        match self {
            Self::Secret { basic: true, .. } => AuthType::BasicAuth,
            Self::Secret { basic: false, .. } | Self::Public | Self::Assertion(_) => {
                AuthType::RequestBody
            }
        }
    }

    /// The secret for the `oauth2` crate's client, when this is one.
    pub(crate) fn client_secret(&self) -> Option<ClientSecret> {
        match self {
            Self::Secret { secret, .. } => {
                Some(ClientSecret::new(secret.expose_secret().to_owned()))
            }
            Self::Public | Self::Assertion(_) => None,
        }
    }

    /// The assertion maker, when this is one.
    pub(crate) const fn assertion(&self) -> Option<&Assertion> {
        match self {
            Self::Assertion(assertion) => Some(assertion),
            Self::Public | Self::Secret { .. } => None,
        }
    }

    /// Whether the client proves nothing but its `client_id`.
    pub(crate) const fn is_public(&self) -> bool {
        matches!(self, Self::Public)
    }
}

impl Assertion {
    /// `client_assertion_type` and a newly signed `client_assertion`.
    ///
    /// # Errors
    ///
    /// Returns an error when signing fails.
    pub(crate) fn params(&self) -> Result<[(&'static str, String); 2]> {
        Ok([
            ("client_assertion_type", String::from(ASSERTION_TYPE)),
            (
                "client_assertion",
                self.key.assertion(&self.client_id, &self.audience)?,
            ),
        ])
    }

    /// Append the two parameters, newly signed, to a form-encoded body.
    ///
    /// # Errors
    ///
    /// Returns an error when signing fails.
    pub(crate) fn append_to(&self, body: &mut Vec<u8>) -> Result<()> {
        let mut form = oauth2::url::form_urlencoded::Serializer::new(String::new());
        for (key, value) in self.params()? {
            form.append_pair(key, &value);
        }
        if !body.is_empty() {
            body.push(b'&');
        }
        body.extend_from_slice(form.finish().as_bytes());
        Ok(())
    }
}
