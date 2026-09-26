//! `quack auth jwks [--rotate [--activate]]` and the key state `quack auth
//! status` shows: the key a `private_key_jwt` client signs with, and its
//! replacement while the operator registers it with the issuer.

use std::io::Write;

use anyhow::Result;
use quack_core::config::{ClientAuth, Config};
use quack_core::llm::oauth::KeySource;
use quack_core::llm::oauth::client_key::{ClientKeyName, ClientKeys, PublicJwks};

/// One configured OAuth client, as the file names it.
struct Client {
    /// `[server.oidc]` or `[providers.NAME.oauth]`.
    section: String,
    issuer: String,
    id: String,
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
                id: oauth.client_id.clone(),
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
            id: oidc.client_id.clone(),
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

    /// The key it signs with.
    fn key_name(&self) -> ClientKeyName {
        ClientKeyName::new(&self.issuer, &self.id)
    }
}

/// The public key set of the client's `private_key_jwt` key, made and
/// stored when there is none yet.
pub(crate) async fn client_jwks(
    config: &Config,
    provider: Option<&str>,
    key_source: KeySource,
) -> Result<PublicJwks> {
    let client = Client::of(config, provider)?;
    client.require_key()?;
    let key = ClientKeys::new(config, key_source)
        .key(&client.key_name())
        .await?;
    Ok(key.jwks())
}

/// What `quack auth status` says about a client's `private_key_jwt` key:
/// its thumbprint, or that none is made yet, and a replacement waiting to
/// be activated. Nothing for other clients.
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
    let name = client.key_name();
    let mut state = match keys.existing(&name).await? {
        Some(key) => format!("client key {}", key.thumbprint()),
        None => format!(
            "no client key yet; `quack auth jwks{}` makes one",
            client.argument
        ),
    };
    if let Some(next) = keys.waiting_replacement(&name).await? {
        state = format!(
            "{state}; replacement key {} waits for `quack auth jwks --rotate --activate{}`",
            next.thumbprint(),
            client.argument
        );
    }
    Ok(Some(state))
}

/// `quack auth jwks [--rotate [--activate]]` on the terminal: the key set
/// to stdout, what to do to stderr. Each is written whole once the keys are
/// ready, so no standard stream is locked across an await.
pub(crate) async fn run_jwks(
    config: &Config,
    provider: Option<&str>,
    rotate_key: bool,
    activate: bool,
) -> Result<()> {
    let (mut out, mut note) = (Vec::new(), Vec::new());
    if rotate_key {
        rotate(
            &mut out,
            &mut note,
            config,
            provider,
            activate,
            KeySource::Keychain,
        )
        .await?;
    } else {
        let jwks = client_jwks(config, provider, KeySource::Keychain).await?;
        writeln!(out, "{}", serde_json::to_string_pretty(&jwks)?)?;
    }
    if !note.is_empty() {
        let mut err = std::io::stderr().lock();
        err.write_all(&note)?;
        err.flush()?;
    }
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&out)?;
    stdout.flush()?;
    Ok(())
}

/// `quack auth jwks --rotate [--activate]`: a new key for the client, in
/// two steps that never leave quack without a key the issuer accepts.
/// `--rotate` makes the replacement (or finds the one already waiting) and
/// prints the key in use beside it, for the operator to register; the key
/// in use keeps signing. `--activate`, run once the issuer holds the new
/// key, puts it in place of the old one and prints it alone. The JWKS goes
/// to `out`; what to do next goes to `note`.
pub(crate) async fn rotate(
    out: &mut impl Write,
    note: &mut impl Write,
    config: &Config,
    provider: Option<&str>,
    activate: bool,
    key_source: KeySource,
) -> Result<()> {
    let client = Client::of(config, provider)?;
    client.require_key()?;
    let keys = ClientKeys::new(config, key_source);
    let name = client.key_name();
    let (id, issuer) = (&client.id, &client.issuer);
    if activate {
        let key = keys.activate_replacement(&name).await?;
        writeln!(out, "{}", serde_json::to_string_pretty(&key.jwks())?)?;
        writeln!(
            note,
            "quack now signs with key {} for client {id}, and the old key is deleted. Replace the key set registered for the client at {issuer} with the one above, which holds the new key alone. Restart a running `quack serve`: it holds the old key in memory until then.",
            key.thumbprint()
        )?;
        return Ok(());
    }
    let (current, next) = keys.stage_replacement(&name).await?;
    let mut both = next.jwks();
    if let Some(current) = &current {
        both.keys.insert(0, current.jwk().clone());
    }
    writeln!(out, "{}", serde_json::to_string_pretty(&both)?)?;
    let in_use = current.as_ref().map_or_else(
        || String::from("no key in use yet"),
        |key| format!("the key in use ({})", key.thumbprint()),
    );
    writeln!(
        note,
        "Register this key set for client {id} at {issuer} in place of the one there: it holds {in_use} and the new key ({}), so the issuer accepts either while you switch. quack keeps signing with the key in use, and running `--rotate` again prints this same pair. Once the issuer holds the set, run `quack auth jwks --rotate --activate{}` to sign with the new key, then restart a running `quack serve`.",
        next.thumbprint(),
        client.argument
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
