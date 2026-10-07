//! `quack doctor`: find what stands between this installation and a
//! working one, and say how to fix each thing.
//!
//! Each check is one line with a status. Nothing here changes the setup:
//! a data directory or control database that does not exist yet is
//! reported, not created, and a workspace is opened only when its file
//! already exists. Network probes (one short `GET` per model's provider,
//! plus a look for a local Ollama when no chat model is set) are skipped
//! with [`Probing::Offline`]. No credential is ever printed: a check says
//! which environment variable a key comes from and whether it is set.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use crate::config;
use crate::config::inspect::{FileState, Inspection};
use crate::config::{
    BaseUrl, BedrockEndpoint, Config, Grant, ModelRef, OAuthConfig, ProviderAuth, ProviderConfig,
    ProviderName, ProviderType,
};
use crate::crypto::CryptoModule;
use crate::embedding::{Dimension, PromptSource, ResolvedPrompts};
use crate::error::{Error, Result as CoreResult};
use crate::llm::bedrock;
use crate::llm::egress::Egress;
use crate::llm::oauth::client_key::ClientKeys;
use crate::llm::oauth::registration::{ReadBack, Registrar, RegistrationName, registered_sections};
use crate::llm::oauth::{KeySource, TokenManager};
use crate::llm::sampling::{Sampling, Wire, check_tool_calls};
use crate::llm::{ChatClient, Embeddings, ProviderModels, RerankModel};
use crate::oidc::SignIn;
use crate::proxy::Proxies;
use crate::storage::control::ControlPlane;
use crate::storage::workspace::{MetaKey, WorkspaceDb};
use crate::text::Count;
use crate::vault::Vault;
use rig::ProviderError;
use rig::error::ErrorKind;
use rig::operation::RerankRequest;
use secrecy::ExposeSecret;
use serde::Serialize;

/// How one check came out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Area {
    Config,
    Crypto,
    /// The forward proxy outbound requests use.
    Proxy,
    Data,
    #[serde(rename = "control db")]
    ControlDb,
    Workspace,
    #[serde(rename = "chat model")]
    ChatModel,
    Embeddings,
    /// The dedicated rerank model, when `[retrieval].rerank = "reranker"`.
    Reranker,
    /// The model that reads images at ingest, `[ingestion].vision_model`.
    Vision,
    Server,
    /// The OAuth clients quack registered itself (`quack auth register`).
    Auth,
    /// The key that seals every stored token (`quack vault export-key`).
    Vault,
}

text_enum!(Area, "doctor area", {
    Config => "config",
    Crypto => "crypto",
    Proxy => "proxy",
    Data => "data",
    ControlDb => "control db",
    Workspace => "workspace",
    ChatModel => "chat model",
    Embeddings => "embeddings",
    Reranker => "reranker",
    Vision => "vision",
    Server => "server",
    Auth => "auth",
    Vault => "vault",
});

/// One finding: what was checked, how it came out, and what to do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Check {
    pub area: Area,
    pub status: Status,
    pub summary: String,
    /// What to run or change, when there is something to do.
    pub fix: Option<String>,
}

/// What a workspace file records about what wrote it; a file from before
/// a value was recorded has none.
struct FileVersions {
    schema: Option<String>,
    quack: Option<String>,
    duckdb: Option<String>,
}

impl FileVersions {
    fn read(db: &WorkspaceDb) -> Result<Self, Error> {
        Ok(Self {
            schema: db.meta(MetaKey::SchemaVersion)?,
            quack: db.meta(MetaKey::WrittenByQuack)?,
            duckdb: db.meta(MetaKey::WrittenByDuckDb)?,
        })
    }
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

    /// A workspace an older quack wrote, which the next open upgrades in
    /// place; reported without opening it.
    fn pending_upgrade(name: &str, recorded: u32) -> Self {
        Self::new(
            Area::Workspace,
            Status::Info,
            format!(
                "'{name}' has schema version {recorded}; this quack upgrades it to version {} the \
                 next time it opens it, and an older quack then refuses it",
                WorkspaceDb::schema_version()
            ),
        )
        .fix(
            "to keep a way back, copy the data directory, or take a snapshot with the quack that \
             wrote it, before the next command opens the workspace",
        )
    }

    /// A workspace with no row: only the default one is created by using
    /// it, so any other name is a failure.
    fn missing_workspace(name: &str, default: &str) -> Self {
        if name == default {
            return Self::new(
                Area::Workspace,
                Status::Ok,
                format!("'{name}' does not exist yet; the first command that uses it creates it"),
            );
        }
        Self::new(
            Area::Workspace,
            Status::Fail,
            Error::NoWorkspaceNamed(name.to_owned()).to_string(),
        )
        .fix(format!("quack workspace create {name}"))
    }

    /// A workspace that opens (`opens` says so), with the versions its file
    /// records. Versions that cannot be read fail the check: an open file
    /// that does not answer for itself is not a healthy one.
    fn recorded_versions(opens: &str, versions: Result<FileVersions, Error>) -> Self {
        match versions {
            Ok(FileVersions {
                schema,
                quack,
                duckdb,
            }) => {
                let or_unrecorded =
                    |v: Option<String>| v.unwrap_or_else(|| String::from("unrecorded"));
                Self::new(
                    Area::Workspace,
                    Status::Ok,
                    format!(
                        "{opens}; schema version {}, written by quack {} with DuckDB {}",
                        or_unrecorded(schema),
                        or_unrecorded(quack),
                        or_unrecorded(duckdb)
                    ),
                )
            }
            Err(e) => Self::new(
                Area::Workspace,
                Status::Fail,
                format!("{opens}, but the versions its file records cannot be read: {e}"),
            )
            .fix("check the file under the data directory, or restore a backup"),
        }
    }

    /// Whether a turn's budgets fit the context window the listing reports
    /// for the chat model: the replayed history, the pinned documents, and
    /// the workspace context are each capped, and together they can fill it.
    fn context_window(
        config: &Config,
        model: ModelRef<'_>,
        models: &ProviderModels,
    ) -> Option<Self> {
        let window = models.get(model.model)?.context_length?;
        let history = config.analysis.history_token_budget.get();
        let pinned = config.retrieval.pinned_token_budget.get();
        let context = config.context.max_tokens.get();
        let budgets = history.saturating_add(pinned).saturating_add(context);
        (budgets > window).then(|| {
            Self::new(
                Area::ChatModel,
                Status::Warn,
                format!(
                    "{model}: a turn may replay {history} tokens of history beside {pinned} of \
                     pinned documents and {context} of workspace context, {budgets} in all, but \
                     the model's context window is {window}"
                ),
            )
            .fix(format!(
                "lower [analysis].history_token_budget, [retrieval].pinned_token_budget, or \
                 [context].max_tokens so they total under {window}"
            ))
        })
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

/// Written as `quack doctor --format json` prints it: whether anything failed,
/// the failure and warning counts, then every check.
impl Serialize for Report {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct as _;
        let mut report = serializer.serialize_struct("Report", 4)?;
        report.serialize_field("ok", &!self.has_failures())?;
        report.serialize_field("failures", &self.count(Status::Fail))?;
        report.serialize_field("warnings", &self.count(Status::Warn))?;
        report.serialize_field("checks", &self.checks)?;
        report.end()
    }
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
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// The workspace to open; `[general].default_workspace` when unset.
    pub workspace: Option<String>,
    pub probing: Probing,
}

/// Whether the checks reach the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probing {
    /// Skip every network probe.
    Offline,
    /// Probe, allowing each probe `timeout`.
    Online { timeout: Duration },
}

impl Default for Probing {
    fn default() -> Self {
        Self::Online {
            timeout: Duration::from_secs(5),
        }
    }
}

impl Probing {
    /// How long each probe may take, or `None` offline.
    const fn timeout(self) -> Option<Duration> {
        match self {
            Self::Offline => None,
            Self::Online { timeout } => Some(timeout),
        }
    }
}

/// Run every check against the configuration `inspection` found.
pub async fn run(inspection: &Inspection, options: &Options) -> Report {
    // The probes reach every configured provider and send no workspace's
    // content, so no workspace's allow-list applies to them.
    Egress::scope(Some(Egress::NoWorkspace), async {
        let mut report = Report::default();
        check_config(&mut report, inspection);
        let config = &inspection.config;
        check_crypto(&mut report);
        check_proxy(&mut report, Proxies::from_env());
        let data_ready = check_data_dir(&mut report, config.data_dir());
        let control = if data_ready {
            check_control(&mut report, config).await
        } else {
            None
        };
        check_workspace(&mut report, config, control.as_ref(), options).await;
        check_chat_model(&mut report, config, options.probing).await;
        check_embedding_model(&mut report, config, options.probing).await;
        check_reranker(&mut report, config, options.probing).await;
        check_vision_model(&mut report, config, options.probing).await;
        check_server(&mut report, config, control.as_ref()).await;
        check_sign_in(&mut report, config, options.probing).await;
        check_registrations(
            &mut report,
            config,
            control.as_ref(),
            options.probing,
            KeySource::Keychain,
        )
        .await;
        check_vault(&mut report, config, KeySource::Keychain).await;
        report
    })
    .await
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
    let module = CryptoModule::linked();
    if module.lacks_expected_fips() {
        report.push(
            Check::new(
                Area::Crypto,
                Status::Warn,
                format!("{module}: this Linux build is not using the FIPS module"),
            )
            .fix("use a release binary or image, which link AWS-LC FIPS (docs/crypto.md)"),
        );
    } else {
        report.push(Check::new(Area::Crypto, Status::Ok, module.to_string()));
    }
}

/// What the proxy variables amount to, and each one that does not do what
/// it says. No request is made.
fn check_proxy(report: &mut Report, proxies: &Proxies) {
    for problem in proxies.problems() {
        let status = if problem.is_failure() {
            Status::Fail
        } else {
            Status::Warn
        };
        report.push(Check::new(Area::Proxy, status, problem.to_string()).fix(problem.fix()));
    }
    let mut through = Vec::new();
    let mut status = Status::Ok;
    for (scheme, proxy) in [("HTTPS", proxies.https()), ("HTTP", proxies.http())] {
        let Some(proxy) = proxy else {
            continue;
        };
        match proxy.unsupported_scheme() {
            Some(unsupported) => {
                status = Status::Fail;
                through.push(format!(
                    "{scheme} through {proxy} (unsupported {unsupported}: these requests fail)"
                ));
            }
            None => through.push(format!("{scheme} through {proxy}")),
        }
    }
    if through.is_empty() {
        if proxies.problems().is_empty() {
            report.push(Check::new(
                Area::Proxy,
                Status::Ok,
                "none (no proxy variables set)",
            ));
        }
        return;
    }
    let listed = match proxies.no_proxy_entries() {
        0 => String::new(),
        1 => String::from(", NO_PROXY (1 entry)"),
        n => format!(", NO_PROXY ({n} entries)"),
    };
    report.push(Check::new(
        Area::Proxy,
        status,
        format!(
            "{}; direct: loopback, 169.254.0.0/16{listed}",
            through.join(", ")
        ),
    ));
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
                     content and the vault key that seals stored OAuth tokens"
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
                    Count(workspaces, "workspace")
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
    let default = config.general.default_workspace.as_str();
    let name = options.workspace.as_deref().unwrap_or(default);
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
        report.push(Check::missing_workspace(name, default));
        return;
    };
    if !config.workspace_db_path(row.id.as_str()).exists() {
        report.push(Check::new(
            Area::Workspace,
            Status::Ok,
            format!("'{name}' is registered; its database file is created on first use"),
        ));
        return;
    }
    // A file an older quack wrote is upgraded in place by any open: the
    // doctor only reads its version, so running it before a backup leaves
    // the way back intact.
    if let Ok(recorded) = WorkspaceDb::recorded_schema(config, row.id.as_str())
        && recorded < WorkspaceDb::schema_version()
    {
        report.push(Check::pending_upgrade(name, recorded));
        return;
    }
    match WorkspaceDb::open(config, row.id.as_str()) {
        Ok(db) => {
            let tables = db.list_tables().map_or(0, |t| t.len());
            let documents = db.list_documents().map_or(0, |d| d.len());
            let opens = format!(
                "'{name}' opens: {}, {}",
                Count(tables, "table"),
                Count(documents, "document")
            );
            report.push(Check::recorded_versions(&opens, FileVersions::read(&db)));
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
        // The error's own text ends with the same advice; the fix says it once.
        Err(Error::WorkspaceTooNew {
            recorded,
            supported,
            written_by,
            ..
        }) => {
            report.push(
                Check::new(
                    Area::Workspace,
                    Status::Fail,
                    format!(
                        "'{name}' does not open: its file has schema version {recorded}, written \
                         by {written_by}, and this quack reads up to version {supported}; the \
                         file was left as it was"
                    ),
                )
                .fix(written_by.advice()),
            );
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

async fn check_chat_model(report: &mut Report, config: &Config, probing: Probing) {
    let Some(spec) = config.general.chat_model.as_ref() else {
        let suggestion = suggest_chat_model(probing).await;
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
        Ok(model) => {
            check_model(report, Area::ChatModel, config, model, probing).await;
            report.push(sampling_check(config, model));
            if let Some(check) = background_check(config, model) {
                report.push(check);
            }
        }
        Err(e) => report.push(Check::new(
            Area::ChatModel,
            Status::Fail,
            format!("\"{spec}\": {e}"),
        )),
    }
}

/// What a chat turn sends the model: temperature, and the reasoning effort
/// if it reaches the model at all. A config every turn refuses fails here too.
fn sampling_check(config: &Config, model: ModelRef<'_>) -> Check {
    let settings = config.model_settings(model);
    let wire = Wire::of(model.provider);
    if let Err(e) = check_tool_calls(model.model, wire, settings.effort) {
        return Check::new(Area::ChatModel, Status::Fail, format!("{model}: {e}"));
    }
    match Sampling::new(model.model, wire, settings.effort, settings.temperature) {
        Ok(sampling) => match sampling.unsent_effort() {
            Some(why) => Check::new(Area::ChatModel, Status::Warn, format!("{model}: {why}")),
            None => Check::new(Area::ChatModel, Status::Ok, format!("{model}: {sampling}")),
        },
        Err(e) => Check::new(Area::ChatModel, Status::Fail, format!("{model}: {e}")).fix(format!(
            "set effort to a level it takes under [providers.{}.models.\"{}\"], \
             [providers.{}], or [analysis]",
            model.provider_name, model.model, model.provider_name
        )),
    }
}

/// What a background call (reranking, history summaries, graph extraction,
/// the ontology's document pass) sends, when its effort is not the turn's.
fn background_check(config: &Config, model: ModelRef<'_>) -> Option<Check> {
    let settings = config.model_settings(model);
    if settings.background_effort == settings.effort {
        return None;
    }
    let sampling = Sampling::new(
        model.model,
        Wire::of(model.provider),
        settings.background_effort,
        settings.temperature,
    );
    Some(match sampling {
        Ok(sampling) => match sampling.unsent_effort() {
            Some(why) => Check::new(
                Area::ChatModel,
                Status::Warn,
                format!("{model}, background calls: {why}"),
            ),
            None => Check::new(
                Area::ChatModel,
                Status::Ok,
                format!("{model}, background calls: {sampling}"),
            ),
        },
        Err(e) => Check::new(
            Area::ChatModel,
            Status::Fail,
            format!(
                "{model}, background calls: {e}; graph extraction and the ontology's document \
                 pass fail, and chat turns run without model reranking and history summaries"
            ),
        )
        .fix(format!(
            "set background_effort to a level it takes under [providers.{}.models.\"{}\"], \
             [providers.{}], or [analysis]",
            model.provider_name, model.model, model.provider_name
        )),
    })
}

async fn check_embedding_model(report: &mut Report, config: &Config, probing: Probing) {
    match config.embedding_model_ref() {
        Ok(None) => report.push(
            Check::new(
                Area::Embeddings,
                Status::Info,
                "no embedding model: document search is keyword-only (BM25), and \
                 documents ingested now are stored without vectors",
            )
            .fix(
                "for semantic search set model and dimension under [embedding], e.g. \
                 model = \"ollama/nomic-embed-text\" and dimension = 768",
            ),
        ),
        Ok(Some(model)) => {
            check_model(report, Area::Embeddings, config, model, probing).await;
            report.push(prompts_check(config, model));
            if let (Some(timeout), Some(configured)) =
                (probing.timeout(), config.embedding.dimension)
            {
                let measured = match Embeddings::from_config(config).await {
                    Ok(Some(embedder)) => {
                        match tokio::time::timeout(timeout, embedder.measure_width()).await {
                            Ok(measured) => measured.map_err(|e| e.to_string()),
                            Err(_) => {
                                Err(format!("no answer within {} seconds", timeout.as_secs()))
                            }
                        }
                    }
                    Ok(None) => return,
                    Err(e) => Err(e.to_string()),
                };
                report.push(width_check(model, configured, measured));
            }
        }
        Err(e) => report.push(Check::new(Area::Embeddings, Status::Fail, e.to_string())),
    }
}

/// The rerank model, when `[retrieval].rerank = "reranker"`: the setting
/// resolves, the provider lists the model, and one small rerank call is
/// answered.
async fn check_reranker(report: &mut Report, config: &Config, probing: Probing) {
    let model = match config.rerank_model_ref() {
        Ok(Some(model)) => model,
        Ok(None) => return,
        Err(e) => {
            report.push(Check::new(Area::Reranker, Status::Fail, e.to_string()));
            return;
        }
    };
    check_model(report, Area::Reranker, config, model, probing).await;
    let Some(timeout) = probing.timeout() else {
        return;
    };
    let answered = match model
        .provider
        .auth
        .credential(config, model.provider_name)
        .await
    {
        Ok(key) => match RerankModel::with_key(model, key.as_deref()) {
            Ok(reranker) => {
                let request = RerankRequest {
                    query: String::from("quack doctor"),
                    documents: vec![String::from("a probe"), String::from("another probe")],
                };
                match tokio::time::timeout(timeout, reranker.rank(request)).await {
                    Ok(Ok(_)) => Ok(()),
                    Ok(Err(e)) => Err(ErrorChain(&e).to_string()),
                    Err(_) => Err(format!("no answer within {} seconds", timeout.as_secs())),
                }
            }
            Err(e) => Err(e.to_string()),
        },
        Err(e) => Err(e.to_string()),
    };
    report.push(match answered {
        Ok(()) => Check::new(
            Area::Reranker,
            Status::Ok,
            format!("{model}: a rerank call was answered"),
        ),
        Err(e) => Check::new(
            Area::Reranker,
            Status::Fail,
            format!("{model}: a rerank call failed: {e}"),
        )
        .fix(format!(
            "check that the server at [providers.{}].base_url serves /rerank for {}",
            model.provider_name, model.model
        )),
    });
}

/// The vision model, when `[ingestion].vision_model` is set: the setting
/// resolves and the provider lists the model. Whether it reads images is
/// the model's own; no image is sent.
async fn check_vision_model(report: &mut Report, config: &Config, probing: Probing) {
    match config.vision_model_ref() {
        Ok(Some(model)) => check_model(report, Area::Vision, config, model, probing).await,
        Ok(None) => {}
        Err(e) => report.push(Check::new(Area::Vision, Status::Fail, e.to_string())),
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

/// Whether the configured width is the one the model makes, from one
/// embedding call's `measured` width or why the call failed.
fn width_check(
    model: ModelRef<'_>,
    configured: Dimension,
    measured: Result<usize, String>,
) -> Check {
    match measured {
        Ok(width) if configured.fits(width) => Check::new(
            Area::Embeddings,
            Status::Ok,
            format!("{model}: makes {width}-dimensional vectors, as [embedding].dimension says"),
        ),
        Ok(width) => Check::new(
            Area::Embeddings,
            Status::Fail,
            format!(
                "{model}: makes {width}-dimensional vectors but [embedding].dimension is \
                 {configured}; every embedding call fails until they agree"
            ),
        )
        .fix(format!("set dimension = {width} under [embedding]")),
        Err(e) => Check::new(
            Area::Embeddings,
            Status::Fail,
            format!("{model}: an embedding call failed: {e}"),
        ),
    }
}

/// Credentials, transport, and whether the provider serves the model.
async fn check_model(
    report: &mut Report,
    area: Area,
    config: &Config,
    model: ModelRef<'_>,
    probing: Probing,
) {
    let provider = model.provider;
    let name = model.provider_name;
    let Some(base) = provider
        .base_url
        .clone()
        .or_else(|| provider.provider_type.default_base_url())
    else {
        report.push(check_bedrock(area, model, probing.timeout()).await);
        return;
    };

    let credential = match model_credential(area, config, model).await {
        Ok(credential) => credential,
        Err(check) => {
            report.push(*check);
            return;
        }
    };

    if credential.is_some() && base.sends_in_cleartext() {
        report.push(
            Check::new(
                area,
                Status::Warn,
                format!("{model}: {base} is plain HTTP to another host, so the credential crosses the network unencrypted"),
            )
            .fix(format!("use an https:// base_url for [providers.{name}]")),
        );
    }

    let Some(timeout) = probing.timeout() else {
        report.push(Check::new(
            area,
            Status::Ok,
            format!("{model}: configured (not probed: --offline)"),
        ));
        return;
    };
    let client = if area == Area::Reranker {
        ChatClient::rerank_server(name, provider, credential.as_deref()).map(ChatClient::OpenAi)
    } else {
        ChatClient::connect(name, provider, credential.as_deref())
    };
    let listing = Probe::listing(client, timeout).await;
    report.push(listing_check(area, model, &base, &listing));
    if area == Area::ChatModel
        && let Ok(models) = &listing
        && let Some(check) = Check::context_window(config, model, models)
    {
        report.push(check);
    }
}

/// Whether the AWS SDK finds a region and credentials for a Bedrock
/// provider, where its endpoint is, and, on bedrock-mantle (the endpoint
/// that lists its models), whether the model is there. On bedrock-runtime
/// the model's access is only known from a model call.
async fn check_bedrock(area: Area, model: ModelRef<'_>, probe: Option<Duration>) -> Check {
    let (name, provider) = (model.provider_name, model.provider);
    let Some(bedrock) = provider.bedrock.as_ref() else {
        return Check::new(
            area,
            Status::Fail,
            format!("{model}: not a Bedrock provider"),
        );
    };
    let Some(endpoint) = provider.provider_type.bedrock_endpoint() else {
        return Check::new(
            area,
            Status::Fail,
            format!("{model}: not a Bedrock provider"),
        );
    };
    let surface = format!("{endpoint}, api {}", bedrock.api);
    let Some(timeout) = probe else {
        return Check::new(
            area,
            Status::Ok,
            format!("{model}: {surface} (not probed: --offline)"),
        );
    };
    let session = match bedrock::session(name, provider).await {
        Ok(session) => session,
        Err(e) => {
            return Check::new(area, Status::Fail, format!("{model}: {e}")).fix(
                match provider.auth.aws_profile() {
                    Some(profile) => format!(
                        "aws sso login --profile {profile}, or check [profile {profile}] in \
                         ~/.aws/config"
                    ),
                    None => format!(
                        "set aws_profile (and region) under [providers.{name}], or export \
                         AWS_PROFILE, or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
                    ),
                },
            );
        }
    };
    let found = format!(
        "{model}: AWS credentials found; {surface} at {} ({})",
        session.root(),
        session.region()
    );
    if endpoint == BedrockEndpoint::Runtime {
        return Check::new(
            area,
            Status::Ok,
            format!(
                "{found}; bedrock-runtime lists no models, so access is checked on the first call"
            ),
        );
    }
    let Ok(base) = BaseUrl::try_from(session.root().to_owned()) else {
        return Check::new(area, Status::Ok, found);
    };
    let listing = Probe::listing(ChatClient::bedrock(&session, name, provider), timeout).await;
    listing_check(area, model, &base, &listing)
}

/// The credential a model's provider is called with, or the failed check
/// saying why there is none.
async fn model_credential(
    area: Area,
    config: &Config,
    model: ModelRef<'_>,
) -> Result<Option<String>, Box<Check>> {
    let provider = model.provider;
    let name = model.provider_name;
    Ok(match &provider.auth {
        // The AWS SDK signs Bedrock's requests; `check_bedrock` covers it.
        ProviderAuth::Aws { .. } => None,
        ProviderAuth::None => {
            // A local rerank server (vLLM, llama.cpp) often runs keyless.
            if provider.provider_type != ProviderType::Ollama && area != Area::Reranker {
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
        ProviderAuth::ApiKey { env } => match provider.auth.credential(config, name).await {
            Ok(key) => key,
            Err(e) => {
                return Err(Box::new(
                    Check::new(area, Status::Fail, format!("{model}: {e}"))
                        .fix(format!("export {env}=... in the environment quack runs in")),
                ));
            }
        },
        // Status first: asking for a token without a login would start one.
        ProviderAuth::Oauth(oauth) => match oauth_token(config, name, oauth).await {
            Ok(OAuthProbe::Token(token)) => Some(token),
            Ok(OAuthProbe::Delegated { listed }) => {
                return Err(Box::new(Check::new(
                    area,
                    Status::Ok,
                    format!(
                        "{model}: acts on behalf of each person signed in to quack serve, without an actor token; {}; the model is reached with the first signed-in person's request",
                        if listed {
                            format!("the issuer lists {}", oauth.grant_type())
                        } else {
                            String::from("the issuer does not list its grants")
                        }
                    ),
                )));
            }
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
    base: &BaseUrl,
    listing: &Result<ProviderModels, Probe>,
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
            ProviderAuth::Oauth(_) => format!("quack auth login {name}"),
            // 401: the credentials themselves; 403: what they may do.
            ProviderAuth::Aws { .. } if *status == 403 => String::from(
                "grant the credentials' IAM principal bedrock-mantle:CreateInference (and allow \
                 it in the VPC endpoint's policy, when base_url is one)",
            ),
            ProviderAuth::Aws { .. } => match provider.auth.aws_profile() {
                Some(profile) => format!(
                    "check the credentials with `aws sts get-caller-identity --profile {profile}`, \
                     or sign in again (`aws sso login --profile {profile}`)"
                ),
                None => String::from(
                    "check the credentials with `aws sts get-caller-identity`, or sign in again",
                ),
            },
            ProviderAuth::None | ProviderAuth::ApiKey { .. } => {
                String::from("check the key in the environment variable")
            }
        }),
        Err(Probe::Unexpected(detail)) => Check::new(
            area,
            Status::Warn,
            format!("{model}: {base} answered, but not with a model list ({detail})"),
        ),
        Ok(models) => {
            let ollama = provider.provider_type == ProviderType::Ollama;
            if models.get(model.model).is_some() {
                return Check::new(
                    area,
                    Status::Ok,
                    if ollama {
                        format!("{model}: reachable, model pulled")
                    } else {
                        format!("{model}: reachable, credential accepted, model listed")
                    },
                );
            }
            let closest = models.closest(model.model);
            let nearest = if closest.is_empty() {
                String::new()
            } else {
                format!("; the closest it lists: {}", closest.join(", "))
            };
            if ollama {
                Check::new(
                    area,
                    Status::Fail,
                    format!(
                        "{model}: Ollama at {base} does not have {}{nearest}",
                        model.model
                    ),
                )
                .fix(format!("ollama pull {}", model.model))
            } else {
                Check::new(
                    area,
                    Status::Warn,
                    format!(
                        "{model}: reachable and the credential is accepted, but {} is not in the provider's model list{nearest}",
                        model.model
                    ),
                )
                .fix("check the model name; some gateways and aliases are not listed")
            }
        }
    }
}

/// What `quack doctor` can learn of an OAuth provider's credential.
enum OAuthProbe {
    /// A token to probe the model list with.
    Token(String),
    /// An on-behalf-of provider without an actor token: only a signed-in
    /// person's request obtains a token, so nothing is requested here.
    /// `listed` is whether the issuer lists its grants (and so, having
    /// passed, lists the exchange).
    Delegated { listed: bool },
}

async fn oauth_token(
    config: &Config,
    name: &ProviderName,
    oauth: &OAuthConfig,
) -> Result<OAuthProbe, String> {
    let manager = TokenManager::shared(config, name, oauth).map_err(|e| e.to_string())?;
    let supported = manager.issuer_supports_grant().await;
    if matches!(supported, Ok(Some(false))) {
        return Err(format!(
            "provider '{name}': the issuer does not list {} in grant_types_supported; check [providers.{name}.oauth].grant",
            oauth.grant_type()
        ));
    }
    let signs_in = match oauth.grant {
        Grant::AuthorizationCode | Grant::DeviceCode => true,
        Grant::ClientCredentials => false,
        // Without an actor, quack never has a token of its own for this
        // provider (Vouch refuses one that names no user), so asking for
        // one would test a grant it never uses.
        Grant::OnBehalfOf if !oauth.actor => {
            return supported
                .map(|listed| OAuthProbe::Delegated {
                    listed: listed.is_some(),
                })
                .map_err(|e| format!("provider '{name}': {e}"));
        }
        // Only a signed-in person can be acted for; quack's own token (the
        // actor) is what can be checked here.
        Grant::OnBehalfOf => {
            return manager
                .service_token()
                .await
                .map(|token| OAuthProbe::Token(token.expose_secret().to_owned()))
                .map_err(|e| format!("provider '{name}': {e}"));
        }
    };
    let status = manager.status().await.map_err(|e| e.to_string())?;
    if signs_in && status.token.is_none() {
        return Err(format!("provider '{name}' uses OAuth and is not logged in"));
    }
    manager
        .access_token()
        .await
        .map(|token| OAuthProbe::Token(token.expose_secret().to_owned()))
        .map_err(|e| format!("provider '{name}': {e}"))
}

/// A config snippet for a chat model: a local Ollama's own models when one
/// answers, otherwise the general shape.
async fn suggest_chat_model(probing: Probing) -> String {
    let local = (probing.timeout(), "ollama".parse::<ProviderName>());
    let pulled = match local {
        (Some(timeout), Ok(name)) => {
            let client =
                ChatClient::connect(&name, &ProviderConfig::new(ProviderType::Ollama), None);
            Probe::listing(client, timeout).await.map_or_else(
                |_| Vec::new(),
                |models| models.as_slice().iter().map(|m| m.id.clone()).collect(),
            )
        }
        (None, _) | (_, Err(_)) => Vec::new(),
    };
    let snippet = |model: &str| {
        format!(
            "add to {}:\n[general]\nchat_model = \"ollama/{model}\"\n\n[providers.ollama]\ntype = \"ollama\"",
            config::config_file_path().display()
        )
    };
    match pulled.first() {
        Some(first) => format!(
            "Ollama is running at {} with {}; {}",
            ProviderType::OLLAMA_BASE_URL,
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
                format!("`quack serve` binds {addr}, this machine only; {}", Count(users, "user")),
            );
            report.push(if users == 0 {
                check.fix("`quack user add NAME` before `quack serve`, or `quack serve --local` for yourself alone")
            } else {
                check
            });
        }
    }
}

/// `[server.oidc]`: the secret it names is set, and online, the issuer
/// answers discovery under the configured name.
async fn check_sign_in(report: &mut Report, config: &Config, probing: Probing) {
    let Some(oidc) = &config.server.oidc else {
        return;
    };
    let issuer = &oidc.issuer_url;
    if config.server.local {
        report.push(
            Check::new(
                Area::Server,
                Status::Warn,
                format!("[server.oidc] ({issuer}) is ignored: [server].local has no login"),
            )
            .fix("turn local off to offer sign-in, or remove [server.oidc]"),
        );
        return;
    }
    if let Some(var) = &oidc.client_secret_env
        && std::env::var_os(var).is_none()
    {
        report.push(
            Check::new(
                Area::Server,
                Status::Fail,
                format!("sign-in with {issuer}: the client secret variable {var} is not set"),
            )
            .fix(format!(
                "export {var} before `quack serve`, or remove client_secret_env for a public client"
            )),
        );
        return;
    }
    if probing == Probing::Offline {
        report.push(Check::new(
            Area::Server,
            Status::Ok,
            format!("sign-in with {issuer}: configured (not probed: --offline)"),
        ));
        return;
    }
    let sign_in = match SignIn::new(oidc.clone(), ClientKeys::new(config, KeySource::Keychain)) {
        Ok(sign_in) => sign_in,
        Err(e) => {
            report.push(Check::new(
                Area::Server,
                Status::Fail,
                format!("sign-in with {issuer}: {e}"),
            ));
            return;
        }
    };
    let keys = match &oidc.audience {
        Some(_) => sign_in.published_keys().await.map(Some),
        None => Ok(None),
    };
    report.push(match (sign_in.discover().await, keys) {
        (Ok(()), Ok(keys)) => Check::new(
            Area::Server,
            Status::Ok,
            match (keys, oidc.audience.as_deref()) {
                (Some(keys), Some(audience)) => format!(
                    "sign-in with {issuer}: the issuer answers discovery and publishes {}; access tokens for {audience} are accepted; callback {}",
                    Count(keys, "signing key"),
                    oidc.redirect_uri
                ),
                (None, _) | (_, None) => format!(
                    "sign-in with {issuer}: the issuer answers discovery; callback {}",
                    oidc.redirect_uri
                ),
            },
        ),
        (Err(e), _) | (_, Err(e)) => Check::new(
            Area::Server,
            Status::Fail,
            format!("sign-in with {issuer}: {e}"),
        )
        .fix("check [server.oidc].issuer_url, and that this host can reach the issuer"),
    });
    if let Some(claim) = &oidc.groups_claim
        && matches!(sign_in.claim_supported(claim).await, Ok(Some(false)))
    {
        report.push(
            Check::new(
                Area::Server,
                Status::Warn,
                format!(
                    "[server.oidc].groups_claim = \"{claim}\", but the issuer's claims_supported does not list it; sign-ins would revoke every provider-granted membership"
                ),
            )
            .fix("check the claim's name, and that the issuer is configured to put groups in its tokens"),
        );
    }
}

/// Where the vault key is, since every sealed token in `control.db` is
/// unreadable without it: a copy kept off this host restores them.
pub(crate) async fn check_vault(report: &mut Report, config: &Config, key_source: KeySource) {
    let vault = Vault::new(config.data_dir(), key_source);
    let check = match vault.key_text().await {
        Err(e) => Check::new(
            Area::Vault,
            Status::Fail,
            format!("the vault key cannot be read: {e}"),
        )
        .fix("unlock the keychain, or check vault.key's permissions"),
        Ok(None) => Check::new(
            Area::Vault,
            Status::Ok,
            "no vault key yet; one is made when the first token is stored",
        ),
        Ok(Some(_)) => {
            let location = vault
                .key_location()
                .await
                .map_or_else(|e| e.to_string(), |l| l.to_string());
            Check::new(
                Area::Vault,
                Status::Ok,
                format!("the vault key is in the {location}; the sealed tokens in control.db open only with it"),
            )
            .fix("`quack vault export-key --to FILE` keeps a copy off this host; a restore of control.db elsewhere needs it as vault.key")
        }
    };
    report.push(check);
}

/// Warn about the temporary sign-in clients an interrupted `quack auth
/// register` left recorded at `issuer`: anyone with an account there can
/// sign in to one until it is deleted.
async fn report_sign_in_leftovers(
    report: &mut Report,
    config: &Config,
    control: &ControlPlane,
    key_source: KeySource,
    issuer: &RegistrationName,
) {
    let leftovers = match Registrar::new(ClientKeys::with_control(
        config,
        key_source,
        control.clone(),
    )) {
        Ok(registrar) => registrar.sign_in_leftovers(issuer).await,
        Err(e) => Err(e),
    };
    if let Ok(leftovers) = leftovers
        && !leftovers.is_empty()
    {
        report.push(
            Check::new(
                Area::Auth,
                Status::Warn,
                format!(
                    "an interrupted `quack auth register` left temporary sign-in client {} registered at {issuer}; until it is deleted, anyone with an account there can sign in to it",
                    leftovers.join(", ")
                ),
            )
            .fix(format!("quack auth register --issuer {issuer} (it deletes them first)")),
        );
    }
}

/// The clients `quack auth register` registered, one per issuer that a
/// section without a `client_id` names: the registration is kept, and
/// online, the issuer still describes that client at its
/// `registration_client_uri` (RFC 7592 read).
pub(crate) async fn check_registrations(
    report: &mut Report,
    config: &Config,
    control: Option<&ControlPlane>,
    probing: Probing,
    key_source: KeySource,
) {
    let mut by_issuer: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for section in registered_sections(config) {
        by_issuer
            .entry(section.issuer.as_str().to_owned())
            .or_default()
            .push(section.section.to_string());
    }
    for (issuer, served) in by_issuer {
        let served = served.join(" and ");
        let name = RegistrationName::new(&issuer);
        let missing = || {
            Check::new(
                Area::Auth,
                Status::Fail,
                format!("no client is registered at {issuer} for {served}, which the file names no client_id for"),
            )
            .fix(format!(
                "quack auth register --issuer {issuer}, or set client_id"
            ))
        };
        let Some(control) = control else {
            report.push(missing());
            continue;
        };
        report_sign_in_leftovers(report, config, control, key_source, &name).await;
        let keys = ClientKeys::with_control(config, key_source, control.clone());
        let row = match keys.registration(&name).await {
            Ok(Some(row)) => row,
            Ok(None) => {
                report.push(missing());
                continue;
            }
            Err(e) => {
                report.push(Check::new(
                    Area::Auth,
                    Status::Fail,
                    format!("the registration at {issuer} cannot be read: {e}"),
                ));
                continue;
            }
        };
        let id = row.client_id;
        if probing == Probing::Offline {
            report.push(Check::new(
                Area::Auth,
                Status::Ok,
                format!(
                    "client {id}, registered at {issuer}, serves {served} (not read back: --offline)"
                ),
            ));
            continue;
        }
        let read = match Registrar::new(keys) {
            Ok(registrar) => registrar.read(&name).await,
            Err(e) => Err(e),
        };
        report.push(match read {
            Ok(Some(ReadBack::Readable { .. })) => Check::new(
                Area::Auth,
                Status::Ok,
                format!(
                    "client {id}, registered at {issuer}, serves {served}; the issuer still describes it"
                ),
            ),
            Ok(Some(ReadBack::Unmanaged { .. })) => Check::new(
                Area::Auth,
                Status::Warn,
                format!(
                    "client {id}, registered at {issuer}, serves {served}; quack holds no registration token for it (it was registered by hand, or the issuer returned none), so it cannot read it back or delete it, and its key rotates by hand"
                ),
            )
            .fix("manage the client in the issuer's console; rotate its key with `quack auth jwks --rotate`"),
            Ok(Some(ReadBack::Refused { status, .. })) => Check::new(
                Area::Auth,
                Status::Fail,
                format!(
                    "{issuer} refused to read back client {id} (HTTP {status}): it was deleted at the issuer, or its registration token was revoked"
                ),
            )
            .fix("quack auth register --replace"),
            Ok(Some(ReadBack::Mismatch { stored, read })) => Check::new(
                Area::Auth,
                Status::Fail,
                format!(
                    "{issuer} describes client {read} at the registration kept for client {stored}"
                ),
            )
            .fix("quack auth register --replace"),
            Ok(None) => missing(),
            Err(e) => Check::new(
                Area::Auth,
                Status::Fail,
                format!("client {id} at {issuer} cannot be read back: {e}"),
            )
            .fix(format!("check that this host can reach {issuer}")),
        });
    }
}

enum Probe {
    Unreachable(String),
    Rejected(u16),
    Unexpected(String),
}

impl Probe {
    /// What `client` lists, within `timeout`.
    async fn listing(
        client: CoreResult<ChatClient>,
        timeout: Duration,
    ) -> Result<ProviderModels, Self> {
        let client = client.map_err(|e| Self::Unexpected(e.to_string()))?;
        match tokio::time::timeout(timeout, client.models()).await {
            Ok(Ok(models)) => Ok(models),
            Ok(Err(e)) => Err(Self::of(&e)),
            Err(_) => Err(Self::Unreachable(format!(
                "no answer within {} seconds",
                timeout.as_secs()
            ))),
        }
    }

    /// A failed listing: a refused credential, another status, or no
    /// reply at all.
    fn of(error: &ProviderError) -> Self {
        let status = error
            .provider_response()
            .and_then(|response| response.status)
            .map(|status| status.as_u16());
        match status {
            Some(code @ (401 | 403)) => Self::Rejected(code),
            Some(code) => Self::Unexpected(format!("HTTP {code}")),
            None if error.kind() == ErrorKind::Http => {
                Self::Unreachable(ErrorChain(error).to_string())
            }
            None => Self::Unexpected(ErrorChain(error).to_string()),
        }
    }
}

impl From<reqwest::Error> for Probe {
    fn from(error: reqwest::Error) -> Self {
        Self::Unreachable(ErrorChain(&error).to_string())
    }
}

/// An error and its sources on one line: reqwest's own message is only
/// "error sending request", and the cause (refused, timed out, DNS) is
/// what the user needs.
struct ErrorChain<'a>(&'a dyn std::error::Error);

impl std::fmt::Display for ErrorChain<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)?;
        let mut source = self.0.source();
        while let Some(cause) = source {
            write!(f, ": {cause}")?;
            source = cause.source();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
