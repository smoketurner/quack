//! `quack auth jwks`, `register`, and `unregister`: the key a
//! `private_key_jwt` client signs with, and the clients quack registers
//! with an issuer itself (RFC 7591) and manages afterwards (RFC 7592).

use std::io::Write;

use anyhow::{Context, Result};
use quack_core::config::{ClientAuth, Config};
use quack_core::llm::oauth::client_key::{ClientKeyName, ClientKeys, PublicJwks};
use quack_core::llm::oauth::registration::{
    ClientMetadata, Registered, Registrar, RegistrationName, Removal, SignIn, TemporaryClient,
    issuer_to_register, metadata_for, registered_sections,
};
use quack_core::llm::oauth::{KeySource, LoginFlow, LoginPrompt};
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
    let registration = keys.registration(&client.registration_name()).await?;
    let key = keys
        .existing(&client.key_name(registration.as_ref()))
        .await?;
    let command = format!("quack auth jwks{}", client.argument);
    let mut state = match (client.client_id(registration.as_ref()), key) {
        (None, _) => {
            return Ok(Some(format!(
                "no client is registered at {}; `quack auth register` registers one",
                client.registration_name()
            )));
        }
        (Some(id), Some(key)) if client.is_registered(registration.as_ref()) => {
            format!("registered client {id}, client key {}", key.thumbprint())
        }
        (Some(_), Some(key)) => format!("client key {}", key.thumbprint()),
        (Some(_), None) => format!("no client key yet; `{command}` makes one"),
    };
    if !client.is_registered(registration.as_ref())
        && let Some(next) = keys
            .waiting_replacement(&client.key_name(registration.as_ref()))
            .await?
    {
        state = format!(
            "{state}; replacement key {} waits for `quack auth jwks --rotate --activate{}`",
            next.thumbprint(),
            client.argument
        );
    }
    Ok(Some(state))
}

/// What `quack auth register` was asked.
#[derive(Debug, clap::Args)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each is a command-line switch; clap refuses the combinations that conflict"
)]
pub(crate) struct RegisterArgs {
    /// The issuer; by default the one those sections share
    #[arg(long)]
    pub(crate) issuer: Option<String>,

    /// Environment variable holding an access token to register with (the
    /// issuer's initial access token, or your own access token, which makes
    /// the client yours)
    #[arg(long, value_name = "VAR", conflicts_with_all = ["device_code", "open"])]
    pub(crate) token_env: Option<String>,

    /// Sign in with the device-code flow instead of the browser
    #[arg(long, conflicts_with = "open")]
    pub(crate) device_code: bool,

    /// Register an open client, one that anyone with an account at the
    /// issuer can sign in to, with no token and no sign-in
    #[arg(long)]
    pub(crate) open: bool,

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
/// Every run first deletes the temporary sign-in clients interrupted runs
/// left at the issuer.
pub(crate) async fn register(
    out: &mut impl Write,
    config: &Config,
    request: RegisterArgs,
    key_source: KeySource,
    confirm: Confirm,
    sign_in: SignInWith<'_>,
) -> Result<()> {
    let issuer = issuer_to_register(config, request.issuer.as_deref())?;
    let registrar = Registrar::new(ClientKeys::new(config, key_source))?;
    let key = registrar.pending_key(&issuer).await?;
    let metadata = metadata_for(config, &issuer, &request.client_name, key.jwks())?;
    if request.print {
        writeln!(out, "{}", serde_json::to_string_pretty(&metadata)?)?;
        return Ok(());
    }
    let earlier = registrar.clean_up_sign_in(&issuer).await?;
    write_temporary(out, &issuer, &earlier)?;
    let how = How::of(&request, &registrar, &issuer).await?;
    if how == How::Open {
        writeln!(
            out,
            "This is an open registration: {issuer} makes a client that anyone with an account there can sign in to. Register with an access token (--token-env VAR) to make the client yours, and restrict who may use it at the issuer."
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
            "Replacing client {}: quack registers the new client first, then deletes this one at {issuer}; anything else configured with its client_id stops working.",
            row.client_id
        )?;
    }
    let registered = match how {
        How::Token(var) => {
            let token = SecretString::from(
                std::env::var(var)
                    .with_context(|| format!("--token-env names {var}, which is not set"))?,
            );
            registrar
                .register(&issuer, &metadata, Some(&token), request.replace)
                .await?
        }
        How::Open => {
            registrar
                .register(&issuer, &metadata, None, request.replace)
                .await?
        }
        How::SignIn => {
            let flow = if request.device_code || !sign_in.browser {
                LoginFlow::DeviceCode
            } else {
                LoginFlow::Configured
            };
            register_signed_in(
                out,
                &registrar,
                &issuer,
                &metadata,
                request.replace,
                SignIn {
                    flow,
                    key_source,
                    notify: sign_in.notify,
                },
                sign_in.interrupt,
            )
            .await?
        }
    };
    write_replaced(out, &issuer, registered.replaced.as_ref())?;
    write_registered(out, config, &issuer, &metadata, &registered, how)?;
    Ok(())
}

/// Sign the person in through a temporary client and register as them;
/// an interrupt stops the sign-in and deletes the temporary client.
async fn register_signed_in(
    out: &mut impl Write,
    registrar: &Registrar,
    issuer: &RegistrationName,
    metadata: &ClientMetadata,
    replace: bool,
    sign_in: SignIn<'_>,
    interrupt: std::pin::Pin<Box<dyn Future<Output = ()> + Send + '_>>,
) -> Result<Registered> {
    writeln!(
        out,
        "Sign in to {issuer} to register the client as yours. quack registers a temporary sign-in client for this and deletes it afterwards."
    )?;
    out.flush()?;
    let signing_in = registrar.register_signed_in(issuer, metadata, replace, sign_in);
    let done = tokio::select! {
        done = signing_in => done?,
        () = interrupt => {
            // The sign-in was dropped mid-flight, before it could delete its
            // temporary client: delete it now.
            let outcomes = registrar.clean_up_sign_in(issuer).await?;
            write_temporary(out, issuer, &outcomes)?;
            anyhow::bail!("interrupted; nothing was registered");
        }
    };
    write_temporary(out, issuer, &done.earlier)?;
    if let TemporaryClient::Left { client_id, reason } = &done.temporary {
        writeln!(
            out,
            "quack could not delete the temporary sign-in client {client_id} ({reason}); it keeps a record of it, and the next `quack auth register` tries again."
        )?;
    }
    Ok(done.registered)
}

/// Say what became of the client a replacement registration replaced.
fn write_replaced(
    out: &mut impl Write,
    issuer: &RegistrationName,
    replaced: Option<&Removal>,
) -> Result<()> {
    match replaced {
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
        Some(Removal::Left { client_id, reason }) => writeln!(
            out,
            "The new client is registered and in use, but deleting the old client {client_id} failed ({reason}). Delete it in the issuer's console."
        )?,
        None => {}
    }
    Ok(())
}

/// Say what became of temporary sign-in clients.
fn write_temporary(
    out: &mut impl Write,
    issuer: &RegistrationName,
    outcomes: &[TemporaryClient],
) -> Result<()> {
    for outcome in outcomes {
        match outcome {
            TemporaryClient::Deleted { client_id } => writeln!(
                out,
                "Deleted the temporary sign-in client {client_id} at {issuer}."
            )?,
            TemporaryClient::Left { client_id, reason } => writeln!(
                out,
                "The temporary sign-in client {client_id} at {issuer} is still registered ({reason}); the next `quack auth register` tries again."
            )?,
        }
    }
    Ok(())
}

/// How `quack auth register` authorizes the registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum How<'a> {
    /// With the access token in this environment variable.
    Token(&'a str),
    /// After the person signs in, with their token: the client is theirs.
    SignIn,
    /// With nothing: anyone at the issuer can sign in to the client.
    Open,
}

impl<'a> How<'a> {
    /// `--token-env` and `--open` say; without either, the person signs in
    /// wherever the issuer's discovery document advertises what that needs
    /// (a `registration_endpoint`, public clients, and PKCE with `S256`),
    /// so an issuer that takes the registration's bearer as the client's
    /// owner, as Vouch does, makes the client theirs. Any other issuer is
    /// refused rather than registered openly, since an open client there
    /// may be anyone's to sign in to.
    async fn of(
        request: &'a RegisterArgs,
        registrar: &Registrar,
        issuer: &RegistrationName,
    ) -> Result<Self> {
        if let Some(var) = &request.token_env {
            return Ok(Self::Token(var));
        }
        if request.open {
            return Ok(Self::Open);
        }
        let missing = registrar.missing_for_sign_in(issuer).await?;
        if missing.is_empty() {
            return Ok(Self::SignIn);
        }
        anyhow::bail!(
            "{issuer} cannot sign you in to register: its discovery document lacks {}. Register with an access token (--token-env VAR), or with --open a client anyone with an account there can sign in to",
            missing.join(", ")
        )
    }
}

/// How `quack auth register` shows a sign-in to the person.
pub(crate) struct SignInWith<'a> {
    /// Whether a browser here can reach the loopback redirect; without one
    /// the sign-in uses the device-code flow.
    pub(crate) browser: bool,
    /// Shows what the person must do.
    pub(crate) notify: &'a (dyn Fn(LoginPrompt) + Sync),
    /// Completes when the person interrupts (Ctrl-C on the terminal): the
    /// sign-in stops, and its temporary client is deleted.
    pub(crate) interrupt: std::pin::Pin<Box<dyn Future<Output = ()> + Send + 'a>>,
}

fn write_registered(
    out: &mut impl Write,
    config: &Config,
    issuer: &RegistrationName,
    metadata: &ClientMetadata,
    registered: &Registered,
    how: How<'_>,
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
    match how {
        How::Open => writeln!(
            out,
            "  - An open registration has no owner, so anyone with an account at {issuer} may be able to sign in to this client. To make it yours, `quack auth register --replace`."
        )?,
        How::SignIn | How::Token(_) => writeln!(
            out,
            "  - The client was registered with {} token. An issuer that makes that account its owner lets only it sign in until the client is widened in the issuer's console (at Vouch: the Applications page, Access scope Organization).",
            if matches!(how, How::SignIn) {
                "your own"
            } else {
                "that account's"
            }
        )?,
    }
    writeln!(
        out,
        "  - `quack doctor` reads the registration back; then start quack (restart a running `quack serve`)."
    )?;
    Ok(())
}

/// `quack auth register` on the terminal. Its questions must reach the
/// screen before the answer is read, so it writes to stdout directly, which
/// locks for each write and never across a request to the issuer.
pub(crate) async fn run_register(
    config: &Config,
    request: RegisterArgs,
    confirm: Confirm,
    sign_in: SignInWith<'_>,
) -> Result<()> {
    let mut out = std::io::stdout();
    register(
        &mut out,
        config,
        request,
        KeySource::Keychain,
        confirm,
        sign_in,
    )
    .await?;
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
    step: Option<Step>,
) -> Result<()> {
    let Some(step) = step else {
        let jwks = client_jwks(config, provider, KeySource::Keychain).await?;
        return write_stdout(format!("{}\n", serde_json::to_string_pretty(&jwks)?).as_bytes());
    };
    let (mut out, mut note) = (Vec::new(), Vec::new());
    rotate(
        &mut out,
        &mut note,
        config,
        provider,
        step,
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
        // `unregister` stops with an error before forgetting a client the
        // issuer would not delete, so this does not come back from it.
        Removal::Left { client_id, reason } => writeln!(
            out,
            "Deleting client {client_id} at {issuer} failed ({reason}); quack still records it."
        )?,
    }
    Ok(())
}

/// Which step of a key rotation `quack auth jwks --rotate` takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Step {
    /// `--rotate`: a new key beside the one in use, both at the issuer.
    Stage,
    /// `--activate`: sign with the new key, and the issuer holds it alone.
    Activate,
}

impl Step {
    /// The step `quack auth jwks` flags name, or none without `--rotate`.
    pub(crate) fn of(rotate: bool, activate: bool) -> Option<Self> {
        rotate.then_some(if activate {
            Self::Activate
        } else {
            Self::Stage
        })
    }
}

/// `quack auth jwks --rotate [--activate]`: a new key for the client, in
/// two steps. `--rotate` makes the replacement beside the key in use and
/// puts both at the issuer, so either authenticates; `--activate` signs
/// with the new key and leaves the issuer holding it alone. For a client
/// quack registered and holds a registration token for, quack updates the
/// issuer itself (RFC 7592); for any other client it prints the key set to
/// register by hand. The JWKS goes to `out`; what to do next goes to `note`.
pub(crate) async fn rotate(
    out: &mut impl Write,
    note: &mut impl Write,
    config: &Config,
    provider: Option<&str>,
    step: Step,
    key_source: KeySource,
) -> Result<()> {
    let client = Client::of(config, provider)?;
    client.require_key()?;
    let registrar = Registrar::new(ClientKeys::new(config, key_source))?;
    let keys = registrar.keys();
    let name = client.registration_name();
    let registration = keys.registration(&name).await?;
    let Some(id) = client.client_id(registration.as_ref()).map(str::to_owned) else {
        anyhow::bail!(
            "{} names no client_id and no client is registered at {name}; `quack auth register` registers one",
            client.section
        );
    };
    let managed = client.is_registered(registration.as_ref())
        && registration
            .as_ref()
            .is_some_and(|row| row.token.is_some() && row.registration_client_uri.is_some());
    let current = ClientKeyName::new(&client.issuer, &id);
    let argument = &client.argument;
    match step {
        Step::Stage => {
            let (in_use, next) = keys.stage_replacement(&current).await?;
            let mut both = next.jwks();
            if let Some(in_use) = &in_use {
                both.keys.insert(0, in_use.jwk().clone());
            }
            let in_use = in_use.as_ref().map_or_else(
                || String::from("no key in use yet"),
                |key| format!("the key in use ({})", key.thumbprint()),
            );
            if managed {
                let published = registrar.publish_keys(&name, &both).await?;
                writeln!(
                    note,
                    "{name} now holds {in_use} and the new key ({}) for client {id}, so it accepts either. quack keeps signing with the key in use, and running `--rotate` again sends this same pair. Next, run `quack auth jwks --rotate --activate{argument}` to sign with the new key.{}",
                    next.thumbprint(),
                    token_note(published.new_registration_token)
                )?;
            } else {
                writeln!(out, "{}", serde_json::to_string_pretty(&both)?)?;
                writeln!(
                    note,
                    "Register this key set for client {id} at {name} in place of the one there: it holds {in_use} and the new key ({}), so the issuer accepts either while you switch. quack keeps signing with the key in use, and running `--rotate` again prints this same pair. Once the issuer holds the set, run `quack auth jwks --rotate --activate{argument}` to sign with the new key.",
                    next.thumbprint()
                )?;
            }
        }
        Step::Activate => {
            // Idempotent: with no replacement waiting, the key in use is
            // sent (or printed) alone again, so a failed update can be retried.
            let key = if keys.waiting_replacement(&current).await?.is_some() {
                keys.activate_replacement(&current).await?
            } else {
                keys.existing(&current).await?.ok_or_else(|| {
                    anyhow::anyhow!(
                        "client {id} at {name} has no key yet; `quack auth jwks --rotate{argument}` makes one"
                    )
                })?
            };
            if managed {
                let published = registrar.publish_keys(&name, &key.jwks()).await?;
                writeln!(
                    note,
                    "quack now signs with key {} for client {id}, and {name} holds it alone; the old key no longer authenticates. Restart every running `quack serve`, which signs with the old key until then.{}",
                    key.thumbprint(),
                    token_note(published.new_registration_token)
                )?;
            } else {
                writeln!(out, "{}", serde_json::to_string_pretty(&key.jwks())?)?;
                writeln!(
                    note,
                    "quack now signs with key {} for client {id}. Replace the key set registered at {name} with this one, which holds that key alone, after restarting every running `quack serve`, which signs with the old key until then.",
                    key.thumbprint()
                )?;
            }
        }
    }
    Ok(())
}

fn token_note(new_registration_token: bool) -> &'static str {
    if new_registration_token {
        " The issuer also issued a new registration access token, which quack keeps."
    } else {
        ""
    }
}

#[cfg(test)]
mod tests;
