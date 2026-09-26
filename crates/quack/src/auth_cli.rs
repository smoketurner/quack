//! `quack auth jwks`, `register`, and `unregister`: the key a
//! `private_key_jwt` client signs with, and the clients quack registers
//! with an issuer itself (RFC 7591) and manages afterwards (RFC 7592).

use std::io::Write;

use anyhow::{Context, Result};
use quack_core::config::{ClientAuth, Config};
use quack_core::llm::oauth::KeySource;
use quack_core::llm::oauth::client_key::{ClientKeyName, ClientKeys, PublicJwks};
use quack_core::llm::oauth::registration::{
    ClientMetadata, Registrar, RegistrationName, Removal, issuer_to_register, metadata_for,
    registered_sections,
};
use quack_core::storage::control::RegistrationRow;
use secrecy::SecretString;

use crate::confirm::Confirm;

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

/// What `quack auth register` was asked.
#[derive(Debug, clap::Args)]
pub(crate) struct RegisterArgs {
    /// The issuer; by default the one those sections share
    #[arg(long)]
    pub(crate) issuer: Option<String>,

    /// Environment variable holding an access token to register with (the
    /// issuer's initial access token, or at Vouch your own access token,
    /// which makes the client yours); without it the registration is open
    #[arg(long, value_name = "VAR")]
    pub(crate) token_env: Option<String>,

    /// The client name the issuer shows
    #[arg(long = "name", default_value = "quack")]
    pub(crate) client_name: String,

    /// Delete the client registered at the issuer (RFC 7592) and register a
    /// new one
    #[arg(long)]
    pub(crate) replace: bool,

    /// Print the registration request as JSON and send nothing
    #[arg(long)]
    pub(crate) print: bool,

    /// Register an open client without asking
    #[arg(long)]
    pub(crate) yes: bool,
}

/// `quack auth register`: register one client for every section at the
/// issuer without a `client_id`, or with `print`, show the request only.
pub(crate) async fn register(
    out: &mut impl Write,
    config: &Config,
    request: RegisterArgs,
    key_source: KeySource,
    confirm: Confirm,
) -> Result<()> {
    let issuer = issuer_to_register(config, request.issuer.as_deref())?;
    let registrar = Registrar::new(ClientKeys::new(config, key_source))?;
    let key = registrar.pending_key(&issuer).await?;
    let metadata = metadata_for(config, &issuer, &request.client_name, key.jwks())?;
    if request.print {
        writeln!(out, "{}", serde_json::to_string_pretty(&metadata)?)?;
        return Ok(());
    }
    let initial_token = match &request.token_env {
        Some(var) => Some(SecretString::from(
            std::env::var(var)
                .with_context(|| format!("--token-env names {var}, which is not set"))?,
        )),
        None => None,
    };
    if initial_token.is_none() {
        writeln!(
            out,
            "Without --token-env this is an open registration: {issuer} makes a client that anyone with an account there can sign in to. Register with an access token (--token-env VAR) to make the client yours, and restrict who may use it at the issuer."
        )?;
        if !confirm.ask(
            out,
            &format!("Register an open client at {issuer}?"),
            Some("--yes"),
        )? {
            anyhow::bail!("nothing registered");
        }
    }
    let existing = registrar.keys().registration(&issuer).await?;
    if let (Some(row), true) = (&existing, request.replace) {
        writeln!(
            out,
            "Replacing client {}: quack deletes it at {issuer} first, and anything else configured with its client_id stops working.",
            row.client_id
        )?;
    }
    let registered = registrar
        .register(&issuer, &metadata, initial_token.as_ref(), request.replace)
        .await?;
    match &registered.replaced {
        Some(Removal::Deleted { client_id }) => {
            writeln!(out, "Deleted client {client_id} at {issuer}.")?;
        }
        Some(Removal::AlreadyGone { client_id, status }) => writeln!(
            out,
            "{issuer} no longer knew client {client_id} (HTTP {status}); nothing was left to delete."
        )?,
        Some(Removal::Unmanaged { client_id }) => writeln!(
            out,
            "quack could not delete client {client_id}: {issuer} returned no registration token for it. Delete it in the issuer's console."
        )?,
        None => {}
    }
    write_registered(out, config, &issuer, &metadata, &registered)?;
    Ok(())
}

fn write_registered(
    out: &mut impl Write,
    config: &Config,
    issuer: &RegistrationName,
    metadata: &ClientMetadata,
    registered: &quack_core::llm::oauth::registration::Registered,
) -> Result<()> {
    let id = &registered.client_id;
    writeln!(
        out,
        "Registered client {id} at {issuer} (grants: {}; key {}).",
        metadata.grant_types.join(", "),
        registered.thumbprint
    )?;
    if let Some(method) = &registered.other_auth_method {
        writeln!(
            out,
            "Warning: {issuer} registered the client with token_endpoint_auth_method \"{method}\", not \"private_key_jwt\"; quack's assertions may be refused."
        )?;
    }
    if !registered.manageable {
        writeln!(
            out,
            "{issuer} returned no registration access token, so quack cannot rotate the key or delete the client later (RFC 7592); do that in its console."
        )?;
    }
    let served: Vec<String> = registered_sections(config)
        .into_iter()
        .filter(|s| s.issuer == *issuer)
        .map(|s| s.section.to_string())
        .collect();
    writeln!(
        out,
        "\n{} leave client_id out and read it from this registration. To name it in the file instead, add to each:\n\n  client_id = \"{id}\"",
        served.join(" and ")
    )?;
    writeln!(out, "\nNext:")?;
    if is_vouch(issuer) {
        writeln!(
            out,
            "  - In the Vouch console, set the app's access scope to Organization; until then the app is open to every Vouch user (an open registration) or to its owner alone."
        )?;
    } else {
        writeln!(
            out,
            "  - At the issuer, check who may use the new client and grant it what it needs."
        )?;
    }
    writeln!(
        out,
        "  - `quack doctor` reads the registration back; then start quack (restart a running `quack serve`)."
    )?;
    Ok(())
}

/// Whether the issuer is Vouch, whose console must widen a registered app
/// to the organization.
fn is_vouch(issuer: &RegistrationName) -> bool {
    let host = issuer
        .as_str()
        .split_once("://")
        .map_or(issuer.as_str(), |(_, rest)| rest)
        .split(['/', ':'])
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    host == "vouch.sh" || host.ends_with(".vouch.sh")
}

/// `quack auth register` on the terminal. Its questions must reach the
/// screen before the answer is read, so it writes to stdout directly, which
/// locks for each write and never across a request to the issuer.
pub(crate) async fn run_register(
    config: &Config,
    request: RegisterArgs,
    confirm: Confirm,
) -> Result<()> {
    let mut out = std::io::stdout();
    register(&mut out, config, request, KeySource::Keychain, confirm).await?;
    out.flush()?;
    Ok(())
}

/// `quack auth unregister` on the terminal, written as `run_register` is.
pub(crate) async fn run_unregister(
    config: &Config,
    issuer: Option<&str>,
    confirm: Confirm,
) -> Result<()> {
    let mut out = std::io::stdout();
    unregister(&mut out, config, issuer, KeySource::Keychain, confirm).await?;
    out.flush()?;
    Ok(())
}

/// `quack auth jwks [--rotate [--activate]]` on the terminal: the key set
/// to stdout, what to do to stderr.
pub(crate) async fn run_jwks(
    config: &Config,
    provider: Option<&str>,
    rotate_key: bool,
    activate: bool,
) -> Result<()> {
    if !rotate_key {
        let jwks = client_jwks(config, provider, KeySource::Keychain).await?;
        return write_stdout(format!("{}\n", serde_json::to_string_pretty(&jwks)?).as_bytes());
    }
    let (mut out, mut note) = (Vec::new(), Vec::new());
    rotate(
        &mut out,
        &mut note,
        config,
        provider,
        activate,
        KeySource::Keychain,
    )
    .await?;
    std::io::stderr().lock().write_all(&note)?;
    write_stdout(&out)
}

fn write_stdout(bytes: &[u8]) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    out.write_all(bytes)?;
    out.flush()?;
    Ok(())
}

/// `quack auth unregister`: delete the registered client at the issuer,
/// then its registration and key.
pub(crate) async fn unregister(
    out: &mut impl Write,
    config: &Config,
    issuer: Option<&str>,
    key_source: KeySource,
    confirm: Confirm,
) -> Result<()> {
    let issuer = match issuer {
        Some(issuer) => RegistrationName::new(issuer),
        None => issuer_to_register(config, None)?,
    };
    let registrar = Registrar::new(ClientKeys::new(config, key_source))?;
    let Some(row) = registrar.keys().registration(&issuer).await? else {
        anyhow::bail!("no client is registered at {issuer}");
    };
    if !confirm.ask(
        out,
        &format!(
            "Delete client {} at {issuer}? Every section that uses it stops working until a client is registered again.",
            row.client_id
        ),
        Some("--yes"),
    )? {
        anyhow::bail!("nothing deleted");
    }
    match registrar.unregister(&issuer).await? {
        Removal::Deleted { client_id } => writeln!(
            out,
            "Deleted client {client_id} at {issuer}, and quack's record of it and its key."
        )?,
        Removal::AlreadyGone { client_id, status } => writeln!(
            out,
            "{issuer} no longer knew client {client_id} (HTTP {status}); quack's record of it and its key are deleted."
        )?,
        Removal::Unmanaged { client_id } => writeln!(
            out,
            "quack's record of client {client_id} and its key are deleted, but {issuer} returned no registration token to delete the client with; delete it in the issuer's console."
        )?,
    }
    Ok(())
}

/// `quack auth jwks --rotate [--activate]`: a new key for the client.
///
/// A client quack registered is updated at the issuer (RFC 7592) and
/// switches to the new key at once. Any other client's issuer cannot be
/// told by quack, so the switch takes two steps that never leave quack
/// without a key the issuer accepts: `--rotate` makes the replacement and
/// prints both keys for the operator to register, and `--activate`, run
/// once the issuer holds the new key, puts it in use. The JWKS goes to
/// stdout; what to do goes to `note`.
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
    let name = client.registration_name();
    let registration = keys.registration(&name).await?;
    if client.is_registered(registration.as_ref()) {
        if activate {
            anyhow::bail!(
                "--activate is for a client registered by hand: quack registered this one, and `quack auth jwks --rotate{}` replaces its key at {name} in one step",
                client.argument
            );
        }
        let rotated = Registrar::new(keys)?.rotate(&name).await?;
        writeln!(
            note,
            "{name} accepted the new key {} for client {}{}; quack signs with it from now on. Restart a running `quack serve`, which holds the old key in memory.{}",
            rotated.new_thumbprint,
            rotated.client_id,
            rotated
                .old_thumbprint
                .map_or(String::new(), |old| format!(" in place of {old}")),
            if rotated.new_registration_token {
                " The issuer also issued a new registration access token, which quack keeps."
            } else {
                ""
            }
        )?;
        return Ok(());
    }
    let Some(id) = client.configured.as_deref() else {
        anyhow::bail!(
            "{} names no client_id and no client is registered at {name}; `quack auth register` registers one",
            client.section
        );
    };
    let current = ClientKeyName::new(&client.issuer, id);
    if activate {
        let key = keys.activate_replacement(&current).await?;
        writeln!(out, "{}", serde_json::to_string_pretty(&key.jwks())?)?;
        writeln!(
            note,
            "quack now signs with key {} for client {id}. Remove the old key from the client's registration at {name}; the set above is the one to keep. Restart a running `quack serve`.",
            key.thumbprint()
        )?;
        return Ok(());
    }
    let (old, next) = keys.stage_replacement(&current).await?;
    let mut both = next.jwks();
    if let Some(old) = &old {
        both.keys.insert(0, old.jwk().clone());
    }
    writeln!(out, "{}", serde_json::to_string_pretty(&both)?)?;
    writeln!(
        note,
        "Register this key set for client {id} at {name}: it holds the key in use{} and the new one ({}), so either is accepted while you switch. Then run `quack auth jwks --rotate --activate{}` to sign with the new key.",
        old.as_ref()
            .map_or(String::new(), |old| format!(" ({})", old.thumbprint())),
        next.thumbprint(),
        client.argument
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
