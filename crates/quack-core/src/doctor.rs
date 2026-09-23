//! `quack doctor`: find what stands between this installation and a
//! working one, and say how to fix each thing.
//!
//! Each check is one line with a status. Nothing here changes the setup:
//! a data directory or control database that does not exist yet is
//! reported, not created, and a workspace is opened only when its file
//! already exists. Network probes (one short `GET` per model's provider,
//! plus a look for a local Ollama when no chat model is set) are skipped
//! with [`Options::offline`]. No credential is ever printed: a check says
//! which environment variable a key comes from and whether it is set.

use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use crate::config::inspect::{FileState, Inspection};
use crate::config::{AuthMode, Config, ModelRef, ProviderType};
use crate::embedding::{PromptSource, ResolvedPrompts};
use crate::error::Error;
use crate::llm::{OllamaRunningModels, oauth};
use crate::storage::control::ControlPlane;
use crate::storage::workspace::WorkspaceDb;
use secrecy::ExposeSecret;

/// Where Ollama listens when nothing says otherwise.
const OLLAMA_DEFAULT_URL: &str = "http://localhost:11434";
const OPENAI_DEFAULT_URL: &str = "https://api.openai.com/v1";
const ANTHROPIC_DEFAULT_URL: &str = "https://api.anthropic.com";

/// How one check came out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    Ok,
    /// Not a problem, but worth knowing (a feature that is off).
    Info,
    /// Works, but is insecure or will fail in some use.
    Warn,
    /// Broken: some command fails until it is fixed.
    Fail,
}

text_enum!(Status, "check status", {
    Ok => "ok",
    Info => "info",
    Warn => "warn",
    Fail => "fail",
});

/// The part of the setup a check looked at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Area {
    Config,
    Crypto,
    Data,
    ControlDb,
    Workspace,
    ChatModel,
    Embeddings,
    Server,
}

text_enum!(Area, "doctor area", {
    Config => "config",
    Crypto => "crypto",
    Data => "data",
    ControlDb => "control db",
    Workspace => "workspace",
    ChatModel => "chat model",
    Embeddings => "embeddings",
    Server => "server",
});

/// One finding: what was checked, how it came out, and what to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub area: Area,
    pub status: Status,
    pub summary: String,
    /// What to run or change, when there is something to do.
    pub fix: Option<String>,
}

impl Check {
    fn new(area: Area, status: Status, summary: impl Into<String>) -> Self {
        Self {
            area,
            status,
            summary: summary.into(),
            fix: None,
        }
    }

    fn fix(mut self, fix: impl Into<String>) -> Self {
        self.fix = Some(fix.into());
        self
    }
}

/// Every check, in the order they ran.
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub checks: Vec<Check>,
}

impl Report {
    /// How many checks came out with `status`.
    #[must_use]
    pub fn count(&self, status: Status) -> usize {
        self.checks.iter().filter(|c| c.status == status).count()
    }

    /// Whether anything is broken.
    #[must_use]
    pub fn has_failures(&self) -> bool {
        self.checks.iter().any(|c| c.status == Status::Fail)
    }

    fn push(&mut self, check: Check) {
        self.checks.push(check);
    }
}

/// What to check.
#[derive(Debug, Clone)]
pub struct Options {
    /// The workspace to open; `[general].default_workspace` when unset.
    pub workspace: Option<String>,
    /// Skip every network probe.
    pub offline: bool,
    /// How long one probe may take.
    pub timeout: Duration,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            workspace: None,
            offline: false,
            timeout: Duration::from_secs(5),
        }
    }
}

/// Run every check against the configuration `inspection` found.
pub async fn run(inspection: &Inspection, options: &Options) -> Report {
    let mut report = Report::default();
    check_config(&mut report, inspection);
    let config = &inspection.config;
    check_crypto(&mut report);
    let data_ready = check_data_dir(&mut report, config.data_dir());
    let control = if data_ready {
        check_control(&mut report, config).await
    } else {
        None
    };
    check_workspace(&mut report, config, control.as_ref(), options).await;
    let http = probe_client(options);
    check_chat_model(&mut report, config, http.as_ref()).await;
    check_embedding_model(&mut report, config, http.as_ref()).await;
    check_server(&mut report, config, control.as_ref()).await;
    report
}

fn check_config(report: &mut Report, inspection: &Inspection) {
    let path = inspection.config_path.display();
    match &inspection.file_state {
        FileState::Missing => report.push(Check::new(
            Area::Config,
            Status::Ok,
            format!("no file at {path}: built-in defaults are in force"),
        )),
        FileState::Loaded => {
            report.push(Check::new(
                Area::Config,
                Status::Ok,
                format!("{path} loaded"),
            ));
        }
        FileState::Rejected(error) => report.push(
            Check::new(
                Area::Config,
                Status::Fail,
                format!(
                    "{path} is rejected, so every other command fails: {}; \
                     the checks below use the built-in values",
                    error.trim_end()
                ),
            )
            .fix("`quack config` lists every key it recognizes and where each value came from"),
        ),
    }
    for unknown in &inspection.unknown {
        let check = Check::new(
            Area::Config,
            Status::Fail,
            format!("unknown key {}", unknown.path),
        );
        report.push(match &unknown.suggestion {
            Some(suggestion) => check.fix(format!("did you mean {suggestion}?")),
            None => check.fix("remove it; `quack config` lists the keys this binary reads"),
        });
    }
}

fn check_crypto(report: &mut Report) {
    let module = crate::crypto::provider_description();
    if aws_lc_rs::fips_version().is_some() || !cfg!(target_os = "linux") {
        report.push(Check::new(Area::Crypto, Status::Ok, module));
    } else {
        report.push(
            Check::new(
                Area::Crypto,
                Status::Warn,
                format!("{module}: this Linux build is not using the FIPS module"),
            )
            .fix("use a release binary or image, which link AWS-LC FIPS (docs/crypto.md)"),
        );
    }
}

/// Returns whether the directory exists, so the checks that read what is
/// inside it can run.
fn check_data_dir(report: &mut Report, dir: &Path) -> bool {
    let shown = dir.display();
    let metadata = match std::fs::metadata(dir) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            report.push(Check::new(
                Area::Data,
                Status::Ok,
                format!("{shown} does not exist yet; the first command creates it, private to you"),
            ));
            return false;
        }
        Err(e) => {
            report.push(
                Check::new(
                    Area::Data,
                    Status::Fail,
                    format!("cannot read {shown}: {e}"),
                )
                .fix("check the path's permissions, or point QUACK_DATA_DIR elsewhere"),
            );
            return false;
        }
    };
    if !metadata.is_dir() {
        report.push(
            Check::new(
                Area::Data,
                Status::Fail,
                format!("{shown} is not a directory"),
            )
            .fix("set [general].data_dir or QUACK_DATA_DIR to a directory"),
        );
        return false;
    }
    let probe = dir.join(format!(".quack-doctor-{}", uuid::Uuid::now_v7()));
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            drop(std::fs::remove_file(&probe));
        }
        Err(e) => {
            report.push(
                Check::new(
                    Area::Data,
                    Status::Fail,
                    format!("cannot write to {shown}: {e}"),
                )
                .fix("fix the directory's owner or permissions, or point QUACK_DATA_DIR elsewhere"),
            );
            return true;
        }
    }
    match exposed_mode(&metadata) {
        Some(mode) => report.push(
            Check::new(
                Area::Data,
                Status::Warn,
                format!(
                    "{shown} is open to other users (mode {mode:o}); it holds every workspace's \
                     content and the OAuth token caches"
                ),
            )
            .fix(format!("chmod 700 {shown}")),
        ),
        None => report.push(Check::new(
            Area::Data,
            Status::Ok,
            format!("{shown} is writable and private"),
        )),
    }
    true
}

/// The directory's permission bits when group or others have any access.
#[cfg(unix)]
fn exposed_mode(metadata: &std::fs::Metadata) -> Option<u32> {
    use std::os::unix::fs::PermissionsExt;
    let mode = metadata.permissions().mode() & 0o777;
    (mode & 0o077 != 0).then_some(mode)
}

#[cfg(not(unix))]
fn exposed_mode(_metadata: &std::fs::Metadata) -> Option<u32> {
    None
}

async fn check_control(report: &mut Report, config: &Config) -> Option<ControlPlane> {
    let path = config.control_db_path();
    if !path.exists() {
        report.push(Check::new(
            Area::ControlDb,
            Status::Ok,
            format!(
                "{} does not exist yet; the first command creates it",
                path.display()
            ),
        ));
        return None;
    }
    match ControlPlane::open(config).await {
        Ok(control) => {
            let workspaces = control.list_workspaces().await.map_or(0, |w| w.len());
            report.push(Check::new(
                Area::ControlDb,
                Status::Ok,
                format!(
                    "{} opens and is migrated; {}",
                    path.display(),
                    plural(workspaces, "workspace")
                ),
            ));
            Some(control)
        }
        Err(e) => {
            report.push(
                Check::new(
                    Area::ControlDb,
                    Status::Fail,
                    format!("{} does not open: {e}", path.display()),
                )
                .fix("a checksum error means the database was written by a different build; use that build or restore a backup"),
            );
            None
        }
    }
}

async fn check_workspace(
    report: &mut Report,
    config: &Config,
    control: Option<&ControlPlane>,
    options: &Options,
) {
    let name = options
        .workspace
        .as_deref()
        .unwrap_or(&config.general.default_workspace);
    let row = match control {
        Some(control) => match control.find_workspace_by_name(name).await {
            Ok(row) => row,
            Err(e) => {
                report.push(Check::new(
                    Area::Workspace,
                    Status::Fail,
                    format!("cannot look up workspace '{name}': {e}"),
                ));
                return;
            }
        },
        None => None,
    };
    let Some(row) = row else {
        report.push(Check::new(
            Area::Workspace,
            Status::Ok,
            format!("'{name}' does not exist yet; the first command that uses it creates it"),
        ));
        return;
    };
    if !config.workspace_db_path(&row.id).exists() {
        report.push(Check::new(
            Area::Workspace,
            Status::Ok,
            format!("'{name}' is registered; its database file is created on first use"),
        ));
        return;
    }
    match WorkspaceDb::open(config, &row.id) {
        Ok(db) => {
            let tables = db.list_tables().map_or(0, |t| t.len());
            let documents = db.list_documents().map_or(0, |d| d.len());
            report.push(Check::new(
                Area::Workspace,
                Status::Ok,
                format!(
                    "'{name}' opens: {}, {}",
                    plural(tables, "table"),
                    plural(documents, "document")
                ),
            ));
            if let Some(note) = db.embedding_status().ok().and_then(|s| s.note()) {
                report.push(
                    Check::new(Area::Workspace, Status::Warn, format!("'{name}': {note}"))
                        .fix(format!("quack embeddings refresh -w {name}")),
                );
            }
        }
        Err(Error::WorkspaceLocked { .. }) => {
            report.push(Check::new(
                Area::Workspace,
                Status::Info,
                format!(
                    "'{name}' is open in another quack process (a server or a session), \
                     so it was not checked"
                ),
            ));
        }
        Err(e) => {
            report.push(
                Check::new(
                    Area::Workspace,
                    Status::Fail,
                    format!("'{name}' does not open: {e}"),
                )
                .fix("check the file under the data directory, or restore a backup"),
            );
        }
    }
}

async fn check_chat_model(report: &mut Report, config: &Config, http: Option<&reqwest::Client>) {
    let Some(spec) = config.general.chat_model.as_deref() else {
        let suggestion = suggest_chat_model(http).await;
        report.push(
            Check::new(
                Area::ChatModel,
                Status::Warn,
                "none configured: SQL (`quack -q`, typed SQL in `quack`), ingest, and \
                 import work; questions, `ontology propose --documents`, and \
                 `graph extract` need one",
            )
            .fix(suggestion),
        );
        return;
    };
    match config.chat_model_ref() {
        Ok(model) => check_model(report, Area::ChatModel, config, model, http).await,
        Err(e) => report.push(Check::new(
            Area::ChatModel,
            Status::Fail,
            format!("\"{spec}\": {e}"),
        )),
    }
}

async fn check_embedding_model(
    report: &mut Report,
    config: &Config,
    http: Option<&reqwest::Client>,
) {
    match config.embedding_model_ref() {
        Ok(None) => report.push(
            Check::new(
                Area::Embeddings,
                Status::Info,
                "no embedding model: document search is keyword-only (BM25), and \
                 documents ingested now are stored without vectors",
            )
            .fix(
                "for semantic search set [general].embedding_model, e.g. \
                 \"ollama/nomic-embed-text\" with embedding_dimension = 768 on the provider",
            ),
        ),
        Ok(Some(model)) => {
            check_model(report, Area::Embeddings, config, model, http).await;
            report.push(prompts_check(config, model));
            if let (Some(http), ProviderType::Ollama, Some(configured)) = (
                http,
                model.provider.provider_type,
                model.provider.embedding_dimension,
            ) {
                let base = model
                    .provider
                    .base_url
                    .clone()
                    .unwrap_or_else(|| OLLAMA_DEFAULT_URL.to_owned());
                let show = OllamaShow::fetch(http, &base, model.model).await;
                if let Some(check) = width_check(model, configured, show) {
                    report.push(check);
                }
            }
        }
        Err(e) => report.push(Check::new(Area::Embeddings, Status::Fail, e.to_string())),
    }
}

/// Which input prefixes the embedding model gets, and where they come
/// from.
fn prompts_check(config: &Config, model: ModelRef<'_>) -> Check {
    let ResolvedPrompts { prompts, source } = ResolvedPrompts::for_model(config, model.model);
    match source {
        PromptSource::Family(family) if prompts.is_empty() => Check::new(
            Area::Embeddings,
            Status::Ok,
            format!(
                "{model}: {} takes no input prefixes ({})",
                family.name, family.source
            ),
        ),
        PromptSource::Family(family) => Check::new(
            Area::Embeddings,
            Status::Ok,
            format!(
                "{model}: the query, document, and similarity prefixes {} was trained with ({})",
                family.name, family.source
            ),
        ),
        PromptSource::Config => Check::new(
            Area::Embeddings,
            Status::Ok,
            format!("{model}: input prefixes from [embedding]"),
        ),
        PromptSource::Unknown => Check::new(
            Area::Embeddings,
            Status::Info,
            format!("{model}: quack knows no input prefixes for this model, so it gets none"),
        )
        .fix(
            "if its model card names query or document prefixes, set query_prefix, \
             document_prefix, and similarity_prefix under [embedding]",
        ),
    }
}

/// What Ollama's `/api/show` says about a model, read from its metadata
/// without loading it.
#[derive(serde::Deserialize)]
struct OllamaShow {
    #[serde(default)]
    model_info: serde_json::Map<String, serde_json::Value>,
}

impl OllamaShow {
    async fn fetch(
        http: &reqwest::Client,
        base: &str,
        model: &str,
    ) -> std::result::Result<Self, Probe> {
        let url = format!(
            "{}/api/show",
            base.trim_end_matches('/').trim_end_matches("/v1")
        );
        let response = http
            .post(url)
            .json(&serde_json::json!({ "model": model }))
            .send()
            .await
            .map_err(|e| Probe::Unreachable(error_chain(&e)))?;
        let status = response.status();
        if !status.is_success() {
            return Err(Probe::Unexpected(format!("HTTP {}", status.as_u16())));
        }
        response
            .json()
            .await
            .map_err(|e| Probe::Unexpected(error_chain(&e)))
    }

    /// The model's vector width: `<architecture>.embedding_length`.
    fn embedding_length(&self) -> Option<u32> {
        self.model_info
            .iter()
            .find(|(key, _)| key.ends_with(".embedding_length"))
            .and_then(|(_, value)| value.as_u64())
            .and_then(|n| u32::try_from(n).ok())
    }
}

/// Whether the configured width is the one the model makes. `None` when
/// the probe could not tell; `check_model` already reported an
/// unreachable provider or a missing model.
fn width_check(
    model: ModelRef<'_>,
    configured: u32,
    show: std::result::Result<OllamaShow, Probe>,
) -> Option<Check> {
    let reported = show.ok()?.embedding_length()?;
    Some(if reported == configured {
        Check::new(
            Area::Embeddings,
            Status::Ok,
            format!("{model}: makes {reported}-dimensional vectors, as embedding_dimension says"),
        )
    } else {
        Check::new(
            Area::Embeddings,
            Status::Fail,
            format!(
                "{model}: makes {reported}-dimensional vectors but embedding_dimension is \
                 {configured}; every embedding call fails until they agree"
            ),
        )
        .fix(format!(
            "set embedding_dimension = {reported} under [providers.{}]",
            model.provider_name
        ))
    })
}

/// Credentials, transport, and whether the provider serves the model.
async fn check_model(
    report: &mut Report,
    area: Area,
    config: &Config,
    model: ModelRef<'_>,
    http: Option<&reqwest::Client>,
) {
    let provider = model.provider;
    let name = model.provider_name;
    let base = provider
        .base_url
        .clone()
        .unwrap_or_else(|| default_base_url(provider.provider_type).to_owned());

    let credential = match model_credential(area, config, model).await {
        Ok(credential) => credential,
        Err(check) => {
            report.push(*check);
            return;
        }
    };

    if credential.is_some() && sends_in_cleartext(&base) {
        report.push(
            Check::new(
                area,
                Status::Warn,
                format!("{model}: {base} is plain HTTP to another host, so the credential crosses the network unencrypted"),
            )
            .fix(format!("use an https:// base_url for [providers.{name}]")),
        );
    }

    let Some(http) = http else {
        report.push(Check::new(
            area,
            Status::Ok,
            format!("{model}: configured (not probed: --offline)"),
        ));
        return;
    };
    let listing = list_models(http, provider.provider_type, &base, credential.as_deref()).await;
    report.push(listing_check(area, model, &base, listing));
}

/// The credential a model's provider is called with, or the failed check
/// saying why there is none.
async fn model_credential(
    area: Area,
    config: &Config,
    model: ModelRef<'_>,
) -> std::result::Result<Option<String>, Box<Check>> {
    let provider = model.provider;
    let name = model.provider_name;
    Ok(match provider.auth {
        AuthMode::None => {
            if provider.provider_type != ProviderType::Ollama {
                return Err(Box::new(
                    Check::new(
                        area,
                        Status::Fail,
                        format!(
                            "{model}: provider '{name}' ({}) needs credentials",
                            provider.provider_type
                        ),
                    )
                    .fix(format!(
                        "set auth = \"api-key\" and api_key_env = \"VAR\" under [providers.{name}]"
                    )),
                ));
            }
            None
        }
        AuthMode::ApiKey => {
            let var = provider.api_key_env.as_deref().unwrap_or_default();
            match std::env::var(var) {
                Ok(key) if !key.trim().is_empty() => Some(key),
                _ => {
                    return Err(Box::new(
                        Check::new(
                            area,
                            Status::Fail,
                            format!("{model}: provider '{name}' reads its key from {var}, which is not set"),
                        )
                        .fix(format!("export {var}=... in the environment quack runs in")),
                    ));
                }
            }
        }
        AuthMode::Oauth => match oauth_token(config, name, provider).await {
            Ok(token) => Some(token),
            Err(message) => {
                return Err(Box::new(
                    Check::new(area, Status::Fail, format!("{model}: {message}"))
                        .fix(format!("quack auth login {name}")),
                ));
            }
        },
    })
}

/// What a model-list probe says about one model.
fn listing_check(
    area: Area,
    model: ModelRef<'_>,
    base: &str,
    listing: std::result::Result<Listing, Probe>,
) -> Check {
    let provider = model.provider;
    let name = model.provider_name;
    match listing {
        Err(Probe::Unreachable(e)) => Check::new(
            area,
            Status::Fail,
            format!("{model}: cannot reach {base}: {e}"),
        )
        .fix(if provider.provider_type == ProviderType::Ollama {
            String::from("start Ollama (`ollama serve`), or set base_url to where it runs")
        } else {
            format!("check base_url under [providers.{name}] and the network")
        }),
        Err(Probe::Rejected(status)) => Check::new(
            area,
            Status::Fail,
            format!("{model}: {base} refused the credential (HTTP {status})"),
        )
        .fix(match provider.auth {
            AuthMode::Oauth => format!("quack auth login {name}"),
            _ => String::from("check the key in the environment variable"),
        }),
        Err(Probe::Unexpected(detail)) => Check::new(
            area,
            Status::Warn,
            format!("{model}: {base} answered, but not with a model list ({detail})"),
        ),
        Ok(Listing::Ollama(models)) if models.holds(model.model) => Check::new(
            area,
            Status::Ok,
            format!("{model}: reachable, model pulled"),
        ),
        Ok(Listing::Ollama(_)) => Check::new(
            area,
            Status::Fail,
            format!("{model}: Ollama at {base} does not have {}", model.model),
        )
        .fix(format!("ollama pull {}", model.model)),
        Ok(Listing::Ids(ids)) if ids.iter().any(|id| id == model.model) => Check::new(
            area,
            Status::Ok,
            format!("{model}: reachable, credential accepted, model listed"),
        ),
        Ok(Listing::Ids(_)) => Check::new(
            area,
            Status::Warn,
            format!(
                "{model}: reachable and the credential is accepted, but {} is not in the provider's model list",
                model.model
            ),
        )
        .fix("check the model name; some gateways and aliases are not listed"),
    }
}

async fn oauth_token(
    config: &Config,
    name: &str,
    provider: &crate::config::ProviderConfig,
) -> std::result::Result<String, String> {
    let manager =
        oauth::shared_manager(&config.tokens_dir(), name, provider).map_err(|e| e.to_string())?;
    let status = manager.status().await.map_err(|e| e.to_string())?;
    if !status.logged_in {
        return Err(format!("provider '{name}' uses OAuth and is not logged in"));
    }
    manager
        .access_token()
        .await
        .map(|token| token.expose_secret().to_owned())
        .map_err(|e| format!("provider '{name}': {e}"))
}

/// A config snippet for a chat model: a local Ollama's own models when one
/// answers, otherwise the general shape.
async fn suggest_chat_model(http: Option<&reqwest::Client>) -> String {
    let pulled = match http {
        Some(http) => match list_models(http, ProviderType::Ollama, OLLAMA_DEFAULT_URL, None).await
        {
            Ok(Listing::Ollama(models)) => models
                .models
                .into_iter()
                .map(|m| m.name)
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        },
        None => Vec::new(),
    };
    let snippet = |model: &str| {
        format!(
            "add to {}:\n[general]\nchat_model = \"ollama/{model}\"\n\n[providers.ollama]\ntype = \"ollama\"",
            crate::config::config_file_path().display()
        )
    };
    match pulled.first() {
        Some(first) => format!(
            "Ollama is running at {OLLAMA_DEFAULT_URL} with {}; {}",
            pulled.join(", "),
            snippet(first)
        ),
        None => format!(
            "install Ollama and `ollama pull llama3.1:8b`, then {}\n\
             (or use an OpenAI-compatible or Anthropic provider with auth = \"api-key\"; \
             QUACK_MODEL overrides chat_model for one run)",
            snippet("llama3.1:8b")
        ),
    }
}

async fn check_server(report: &mut Report, config: &Config, control: Option<&ControlPlane>) {
    let bind = &config.server.bind;
    match bind.parse::<SocketAddr>() {
        Err(e) => report.push(
            Check::new(
                Area::Server,
                Status::Fail,
                format!("[server].bind = \"{bind}\" is not an address: {e}"),
            )
            .fix("use IP:PORT, e.g. \"127.0.0.1:8080\""),
        ),
        Ok(addr) if config.server.local && !addr.ip().is_loopback() => report.push(
            Check::new(
                Area::Server,
                Status::Fail,
                format!("[server].local serves without authentication, but bind = {addr} is not loopback"),
            )
            .fix("bind 127.0.0.1, or turn local off"),
        ),
        Ok(addr) if !addr.ip().is_loopback() => report.push(
            Check::new(
                Area::Server,
                Status::Warn,
                format!(
                    "`quack serve` listens on {addr}, reachable from other machines over plain HTTP"
                ),
            )
            .fix("put a TLS-terminating proxy in front, or bind 127.0.0.1"),
        ),
        Ok(addr) => {
            let users = match control {
                Some(control) => control.list_users().await.map_or(0, |u| u.len()),
                None => 0,
            };
            let check = Check::new(
                Area::Server,
                Status::Ok,
                format!("`quack serve` binds {addr}, this machine only; {}", plural(users, "user")),
            );
            report.push(if users == 0 {
                check.fix("`quack user add NAME` before `quack serve`, or `quack serve --local` for yourself alone")
            } else {
                check
            });
        }
    }
}

enum Listing {
    Ollama(OllamaRunningModels),
    Ids(Vec<String>),
}

enum Probe {
    Unreachable(String),
    Rejected(u16),
    Unexpected(String),
}

#[derive(serde::Deserialize)]
struct IdList {
    #[serde(default)]
    data: Vec<IdEntry>,
}

#[derive(serde::Deserialize)]
struct IdEntry {
    id: String,
}

fn probe_client(options: &Options) -> Option<reqwest::Client> {
    if options.offline {
        return None;
    }
    reqwest::Client::builder()
        .timeout(options.timeout)
        .connect_timeout(options.timeout)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .inspect_err(|e| tracing::warn!(error = %e, "cannot build the probe client"))
        .ok()
}

/// One `GET` for the provider's model list.
async fn list_models(
    http: &reqwest::Client,
    provider: ProviderType,
    base: &str,
    credential: Option<&str>,
) -> std::result::Result<Listing, Probe> {
    let base = base.trim_end_matches('/');
    let request = match provider {
        ProviderType::Ollama => {
            let request = http.get(format!("{}/api/tags", base.trim_end_matches("/v1")));
            match credential {
                Some(key) => request.bearer_auth(key),
                None => request,
            }
        }
        ProviderType::Openai => {
            let request = http.get(format!("{base}/models"));
            match credential {
                Some(key) => request.bearer_auth(key),
                None => request,
            }
        }
        ProviderType::Anthropic => {
            let request = http
                .get(format!("{base}/v1/models?limit=1000"))
                .header("anthropic-version", "2023-06-01");
            match credential {
                Some(key) => request.header("x-api-key", key),
                None => request,
            }
        }
    };
    let response = request
        .send()
        .await
        .map_err(|e| Probe::Unreachable(error_chain(&e)))?;
    let status = response.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(Probe::Rejected(status.as_u16()));
    }
    if !status.is_success() {
        return Err(Probe::Unexpected(format!("HTTP {}", status.as_u16())));
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|e| Probe::Unreachable(error_chain(&e)))?;
    match provider {
        ProviderType::Ollama => serde_json::from_slice::<OllamaRunningModels>(&bytes)
            .map(Listing::Ollama)
            .map_err(|e| Probe::Unexpected(e.to_string())),
        ProviderType::Openai | ProviderType::Anthropic => serde_json::from_slice::<IdList>(&bytes)
            .map(|list| Listing::Ids(list.data.into_iter().map(|e| e.id).collect()))
            .map_err(|e| Probe::Unexpected(e.to_string())),
    }
}

fn default_base_url(provider: ProviderType) -> &'static str {
    match provider {
        ProviderType::Ollama => OLLAMA_DEFAULT_URL,
        ProviderType::Openai => OPENAI_DEFAULT_URL,
        ProviderType::Anthropic => ANTHROPIC_DEFAULT_URL,
    }
}

/// Whether a credential sent to `base` would cross the network unencrypted:
/// plain HTTP to anything but this machine.
fn sends_in_cleartext(base: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base) else {
        return false;
    };
    if url.scheme() != "http" {
        return false;
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => !ip.is_loopback(),
        Err(_) => host != "localhost",
    }
}

/// An error and its sources on one line: reqwest's own message is only
/// "error sending request", and the cause (refused, timed out, DNS) is
/// what the user needs.
fn error_chain(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

fn plural(n: usize, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(clippy::unwrap_used, reason = "test")]
    fn embedding_config(model: &str, extra: &str) -> Config {
        let config: Config = toml::from_str(&format!(
            "[general]\nembedding_model = \"o/{model}\"\n[providers.o]\ntype = \"ollama\"\n\
             embedding_dimension = 1024\n{extra}"
        ))
        .unwrap();
        config
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test")]
    fn the_width_probe_names_the_fix_when_the_model_disagrees() {
        let config = embedding_config("embeddinggemma", "");
        let model = config.embedding_model_ref().unwrap().unwrap();
        let show =
            |json: serde_json::Value| -> OllamaShow { serde_json::from_value(json).unwrap() };
        let gemma = || {
            show(serde_json::json!({
                "model_info": { "general.architecture": "gemma3", "gemma3.embedding_length": 768 }
            }))
        };
        assert_eq!(gemma().embedding_length(), Some(768));

        let wrong = width_check(model, 1024, Ok(gemma())).unwrap();
        assert_eq!(wrong.status, Status::Fail);
        assert!(
            wrong.summary.contains("768-dimensional"),
            "{}",
            wrong.summary
        );
        assert_eq!(
            wrong.fix.as_deref(),
            Some("set embedding_dimension = 768 under [providers.o]")
        );
        assert_eq!(
            width_check(model, 768, Ok(gemma())).unwrap().status,
            Status::Ok
        );
        assert!(width_check(model, 768, Ok(show(serde_json::json!({})))).is_none());
        assert!(width_check(model, 768, Err(Probe::Rejected(401))).is_none());
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test")]
    fn the_prompts_check_says_where_the_prefixes_come_from() {
        for (model, extra, status, words) in [
            (
                "embeddinggemma",
                "",
                Status::Ok,
                "EmbeddingGemma was trained with",
            ),
            ("all-minilm", "", Status::Ok, "takes no input prefixes"),
            (
                "embeddinggemma",
                "[embedding]\nquery_prefix = \"q: \"\n",
                Status::Ok,
                "from [embedding]",
            ),
            ("my-embedder", "", Status::Info, "knows no input prefixes"),
        ] {
            let config = embedding_config(model, extra);
            let check = prompts_check(&config, config.embedding_model_ref().unwrap().unwrap());
            assert_eq!(check.status, status, "{model}");
            assert!(check.summary.contains(words), "{}", check.summary);
        }
    }

    fn inspection(dir: &Path, toml: Option<&str>) -> Inspection {
        let mut inspection = Inspection::of(dir.join("config.toml"), toml);
        inspection.config.general.data_dir = dir.join("data");
        inspection
    }

    fn offline() -> Options {
        Options {
            offline: true,
            ..Options::default()
        }
    }

    fn find(report: &Report, area: Area) -> Vec<&Check> {
        report.checks.iter().filter(|c| c.area == area).collect()
    }

    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn a_fresh_install_has_no_failures_and_says_what_needs_a_model() {
        let dir = tempfile::tempdir().unwrap();
        let report = run(&inspection(dir.path(), None), &offline()).await;
        assert!(!report.has_failures(), "{report:#?}");
        let chat = find(&report, Area::ChatModel);
        assert_eq!(chat.len(), 1);
        assert_eq!(chat.first().unwrap().status, Status::Warn);
        assert!(chat.first().unwrap().summary.contains("SQL"));
        assert!(
            chat.first()
                .unwrap()
                .fix
                .as_deref()
                .unwrap()
                .contains("chat_model")
        );
        assert_eq!(
            find(&report, Area::Embeddings).first().unwrap().status,
            Status::Info
        );
        // Nothing was created by looking.
        assert!(!dir.path().join("data").exists());
    }

    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn a_rejected_file_and_its_unknown_key_are_failures() {
        let dir = tempfile::tempdir().unwrap();
        let report = run(
            &inspection(dir.path(), Some("[general]\nchat_modle = \"x/y\"\n")),
            &offline(),
        )
        .await;
        let config = find(&report, Area::Config);
        assert!(
            config.iter().all(|c| c.status == Status::Fail),
            "{config:#?}"
        );
        assert!(
            config
                .iter()
                .any(|c| c.fix.as_deref().is_some_and(|f| f.contains("chat_model")))
        );
    }

    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn a_missing_api_key_is_a_failure_naming_the_variable() {
        let dir = tempfile::tempdir().unwrap();
        let toml = "[general]\nchat_model = \"a/claude\"\n[providers.a]\ntype = \"anthropic\"\n\
                    auth = \"api-key\"\napi_key_env = \"QUACK_DOCTOR_TEST_KEY_UNSET\"\n";
        let report = run(&inspection(dir.path(), Some(toml)), &offline()).await;
        let chat = find(&report, Area::ChatModel);
        let check = chat.first().unwrap();
        assert_eq!(check.status, Status::Fail);
        assert!(check.summary.contains("QUACK_DOCTOR_TEST_KEY_UNSET"));
    }

    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn an_unreachable_ollama_is_a_failure_with_the_cause() {
        let dir = tempfile::tempdir().unwrap();
        let toml = "[general]\nchat_model = \"o/m\"\n[providers.o]\ntype = \"ollama\"\n\
                    base_url = \"http://127.0.0.1:9\"\n";
        let options = Options {
            timeout: Duration::from_secs(2),
            ..Options::default()
        };
        let report = run(&inspection(dir.path(), Some(toml)), &options).await;
        let check = *find(&report, Area::ChatModel).first().unwrap();
        assert_eq!(check.status, Status::Fail, "{check:#?}");
        assert!(check.summary.contains("cannot reach"));
        assert!(check.fix.as_deref().unwrap().contains("ollama serve"));
    }

    #[cfg(unix)]
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn a_data_dir_others_can_read_is_a_warning_and_a_fresh_one_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let inspection = inspection(dir.path(), None);
        inspection.config.ensure_dirs().unwrap();
        let data = inspection.config.data_dir();
        let mode = std::fs::metadata(data).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
        let report = run(&inspection, &offline()).await;
        assert_eq!(
            find(&report, Area::Data).first().unwrap().status,
            Status::Ok
        );

        std::fs::set_permissions(data, std::fs::Permissions::from_mode(0o755)).unwrap();
        let report = run(&inspection, &offline()).await;
        let check = *find(&report, Area::Data).first().unwrap();
        assert_eq!(check.status, Status::Warn);
        assert!(check.fix.as_deref().unwrap().starts_with("chmod 700"));
    }

    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn an_open_bind_and_local_off_loopback_are_flagged() {
        let dir = tempfile::tempdir().unwrap();
        let open = run(
            &inspection(dir.path(), Some("[server]\nbind = \"0.0.0.0:8080\"\n")),
            &offline(),
        )
        .await;
        assert_eq!(
            find(&open, Area::Server).first().unwrap().status,
            Status::Warn
        );
        let local = run(
            &inspection(
                dir.path(),
                Some("[server]\nbind = \"0.0.0.0:8080\"\nlocal = true\n"),
            ),
            &offline(),
        )
        .await;
        assert_eq!(
            find(&local, Area::Server).first().unwrap().status,
            Status::Fail
        );
    }

    #[test]
    fn cleartext_is_plain_http_off_this_machine() {
        assert!(sends_in_cleartext("http://gpu-box:11434"));
        assert!(sends_in_cleartext("http://10.0.0.5/v1"));
        assert!(!sends_in_cleartext("http://localhost:11434"));
        assert!(!sends_in_cleartext("http://127.0.0.1:11434"));
        assert!(!sends_in_cleartext("http://[::1]:11434"));
        assert!(!sends_in_cleartext("https://api.openai.com/v1"));
    }
}
