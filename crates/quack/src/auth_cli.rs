//! `quack auth jwks` and the key state `quack auth status` shows: the key a
//! `private_key_jwt` client signs with, including a client quack registers
//! with an issuer itself (RFC 7591), whose `client_id` the file leaves out.

use anyhow::Result;
use quack_core::config::{ClientAuth, Config};
use quack_core::llm::oauth::KeySource;
use quack_core::llm::oauth::client_key::{ClientKeyName, ClientKeys, PublicJwks};
use quack_core::llm::oauth::registration::RegistrationName;
use quack_core::storage::control::RegistrationRow;

/// One configured OAuth client, as the file names it.
struct Client {
    /// `[server.oidc]` or `[providers.NAME.oauth]`.
    section: String,
    issuer: String,
    /// The `client_id` the file names; `None` for a registered client.
    configured: Option<String>,
    auth: ClientAuth,
    /// How the `quack auth` commands name it: ` NAME` for a provider,
    /// nothing for the sign-in client.
    argument: String,
}

impl Client {
    /// The provider's OAuth client, or without one the `[server.oidc]`
    /// sign-in client.
    fn of(config: &Config, provider: Option<&str>) -> Result<Self> {
        if let Some(name) = provider {
            let oauth = config
                .providers
                .get(name)
                .and_then(|p| p.auth.oauth())
                .ok_or_else(|| {
                    anyhow::anyhow!("'{name}' is not a provider with auth = \"oauth\"")
                })?;
            return Ok(Self {
                section: format!("[providers.{name}.oauth]"),
                issuer: oauth.issuer_url.clone(),
                configured: oauth.client_id.clone(),
                auth: oauth.client_auth,
                argument: format!(" {name}"),
            });
        }
        let oidc = config.server.oidc.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "[server.oidc] is not configured; name a provider: `quack auth jwks PROVIDER`"
            )
        })?;
        Ok(Self {
            section: String::from("[server.oidc]"),
            issuer: oidc.issuer_url.clone(),
            configured: oidc.client_id.clone(),
            auth: oidc.client_auth,
            argument: String::new(),
        })
    }

    fn require_key(&self) -> Result<()> {
        if self.auth != ClientAuth::PrivateKeyJwt {
            anyhow::bail!(
                "{} authenticates with client_auth = \"{}\"; set client_auth = \"private_key_jwt\" to sign with a key",
                self.section,
                self.auth
            );
        }
        Ok(())
    }

    fn registration_name(&self) -> RegistrationName {
        RegistrationName::new(&self.issuer)
    }

    /// The client id in use: the file's, else the registration's.
    fn client_id<'a>(&'a self, registration: Option<&'a RegistrationRow>) -> Option<&'a str> {
        self.configured
            .as_deref()
            .or_else(|| registration.map(|row| row.client_id.as_str()))
    }

    /// Whether quack registered this client: the file leaves its id out, or
    /// names the one registered at its issuer.
    fn is_registered(&self, registration: Option<&RegistrationRow>) -> bool {
        registration.is_some_and(|row| {
            self.configured
                .as_deref()
                .is_none_or(|id| id == row.client_id)
        })
    }

    /// The key it signs with: its client's, or, for a client not registered
    /// yet, the key its registration will carry.
    fn key_name(&self, registration: Option<&RegistrationRow>) -> ClientKeyName {
        match self.client_id(registration) {
            Some(id) => ClientKeyName::new(&self.issuer, id),
            None => ClientKeyName::pending(&self.issuer),
        }
    }
}

/// The public key set of the client's `private_key_jwt` key, made and
/// stored when there is none yet. For a client the file names no id for
/// and nothing is registered yet, that is the key `quack auth register`
/// (or a registration by hand) sends.
pub(crate) async fn client_jwks(
    config: &Config,
    provider: Option<&str>,
    key_source: KeySource,
) -> Result<PublicJwks> {
    let client = Client::of(config, provider)?;
    client.require_key()?;
    let keys = ClientKeys::new(config, key_source);
    let registration = keys.registration(&client.registration_name()).await?;
    let key = keys.key(&client.key_name(registration.as_ref())).await?;
    Ok(key.jwks())
}

/// What `quack auth status` says about a client's `private_key_jwt` key:
/// its thumbprint, or that none is made yet. Nothing for other clients.
pub(crate) async fn client_key_state(
    config: &Config,
    provider: Option<&str>,
    key_source: KeySource,
) -> Result<Option<String>> {
    let client = Client::of(config, provider)?;
    if client.auth != ClientAuth::PrivateKeyJwt {
        return Ok(None);
    }
    let keys = ClientKeys::new(config, key_source);
    let registration = keys.registration(&client.registration_name()).await?;
    let key = keys
        .existing(&client.key_name(registration.as_ref()))
        .await?;
    let command = format!("quack auth jwks{}", client.argument);
    Ok(Some(match (client.client_id(registration.as_ref()), key) {
        (None, _) => format!(
            "no client is registered at {}; `quack auth register` registers one",
            client.registration_name()
        ),
        (Some(id), Some(key)) if client.is_registered(registration.as_ref()) => {
            format!("registered client {id}, client key {}", key.thumbprint())
        }
        (Some(_), Some(key)) => format!("client key {}", key.thumbprint()),
        (Some(_), None) => format!("no client key yet; `{command}` makes one"),
    }))
}

#[cfg(test)]
mod tests;
