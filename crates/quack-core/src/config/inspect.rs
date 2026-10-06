//! What this binary makes of `config.toml`.
//!
//! [`Config::load`] either hands back a configuration or fails: a typo in a
//! key is an error (`deny_unknown_fields`), and nothing afterwards says
//! which of the values in force came from the file, which from the
//! environment, and which are built in. This module answers both questions
//! without going through that gate, so `quack config` can describe a file
//! every other command refuses to start on.
//!
//! Nothing here reads a secret. The file names the environment variables
//! that hold API keys and client secrets; an [`Inspection`] carries those
//! names and whether they are set, never their contents.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Serialize, Serializer};
use toml::{Table, Value as TomlValue};

use crate::embedding::ResolvedPrompts;
use crate::embedding::presets::Family;
use crate::error;

use super::{
    AuthMode, AwsRegion, BaseUrl, ClientAuth, Config, ENV_BIND, ENV_CONFIG_DIR, ENV_DATA_DIR,
    ENV_MODEL, Effort, Exchange, Grant, ModelSettings, ModelSpec, OAuthConfig, OidcConfig,
    Overrides, ProviderType, config_file_path,
};

/// How an unset optional setting is rendered.
pub const UNSET: &str = "(unset)";

/// Where the value in force came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// Built in: the file does not set it, or the file is not in force.
    Default,
    /// The config file sets it.
    File,
    /// The named environment variable sets it, whatever the file says.
    Env(&'static str),
    /// A client `quack auth register` registered: the `client_id` the file
    /// leaves out, read from the registration kept in `control.db`.
    Registration,
}

/// Written as it prints: `default`, `file`, or `env NAME`.
impl Serialize for Origin {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => f.write_str("default"),
            Self::File => f.write_str("file"),
            Self::Env(var) => write!(f, "env {var}"),
            Self::Registration => f.write_str("registration"),
        }
    }
}

/// One setting this binary recognizes, and what it makes of it. Values
/// are rendered as TOML, the form the config file writes them in, so a
/// string keeps its quotes.
#[derive(Debug, Clone, Serialize)]
pub struct Setting {
    /// The table it lives in: `general`, `providers.ollama`, and so on.
    pub section: String,
    pub key: &'static str,
    /// The value in force, rendered as TOML; `None` when an optional
    /// setting is unset.
    pub value: Option<String>,
    /// What it would be with no config file and no environment; `None`
    /// for a setting with no built-in value.
    pub default: Option<String>,
    pub origin: Origin,
    /// What the file says, rendered as TOML, whether or not that is the
    /// value in force.
    pub file_value: Option<String>,
    /// The environment variable that overrides this setting.
    pub env: Option<&'static str>,
}

impl Setting {
    /// The value in force, or [`UNSET`].
    #[must_use]
    pub fn display_value(&self) -> &str {
        self.value.as_deref().unwrap_or(UNSET)
    }

    /// The built-in value, or [`UNSET`].
    #[must_use]
    pub fn display_default(&self) -> &str {
        self.default.as_deref().unwrap_or(UNSET)
    }

    /// Whether nothing moved this off its built-in value.
    #[must_use]
    pub fn is_default(&self) -> bool {
        self.origin == Origin::Default
    }

    /// `section.key`, the path a config file writes it under.
    #[must_use]
    pub fn path(&self) -> String {
        format!("{}.{}", self.section, self.key)
    }
}

/// A key in the file that no setting corresponds to. Every section sets
/// `deny_unknown_fields`, so one of these is why the binary refuses the
/// file rather than a setting that quietly does nothing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnknownKey {
    /// Dotted path as the file writes it: `retrieval.topk`.
    pub path: String,
    /// The recognized key it most resembles, when one is close enough to
    /// name.
    pub suggestion: Option<String>,
}

/// An environment variable this binary reads, and whether it is set. The
/// value is never recorded: some of these hold credentials.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EnvVar {
    pub name: String,
    pub set: bool,
    /// What it does, for the listing.
    pub purpose: String,
}

/// What became of the config file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileState {
    /// No file at the path: every value is built in or from the
    /// environment.
    Missing,
    /// Read, parsed, and accepted: its values are in force.
    Loaded,
    /// Read but refused, with the error every command would fail on. The
    /// settings then carry the built-in values, since the file's are not
    /// in force.
    Rejected(String),
}

/// The whole picture: what the file holds, what this binary recognizes,
/// and where each value in force came from.
#[derive(Debug, Clone)]
pub struct Inspection {
    /// Where the config file is looked for.
    pub config_path: PathBuf,
    pub file_state: FileState,
    /// The configuration in force — the file's when it loaded, the
    /// built-in one when it did not.
    pub config: Config,
    /// Every recognized setting, in file order: `general` first, then the
    /// providers, then the remaining sections.
    pub settings: Vec<Setting>,
    /// Keys in the file that no setting corresponds to.
    pub unknown: Vec<UnknownKey>,
    /// The environment variables this configuration reads.
    pub environment: Vec<EnvVar>,
}

impl Inspection {
    /// Inspect the config file [`config_file_path`] names.
    ///
    /// Never fails: an unreadable, unparseable, or invalid file is a
    /// [`FileState::Rejected`] with the error, which is the case this
    /// command exists for.
    #[must_use]
    pub fn load() -> Self {
        let path = config_file_path();
        match std::fs::read_to_string(&path) {
            Ok(text) => Self::of(path, Some(&text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Self::of(path, None),
            Err(e) => {
                let mut inspection = Self::of(path.clone(), None);
                inspection.file_state =
                    FileState::Rejected(format!("cannot read {}: {e}", path.display()));
                inspection
            }
        }
    }

    /// The same inspection over given file contents, `None` for no file.
    #[must_use]
    pub fn of(config_path: PathBuf, contents: Option<&str>) -> Self {
        let raw = contents.and_then(|text| toml::from_str::<Table>(text).ok());
        let overrides = Overrides::from_env();
        let loaded = Config::from_contents(contents, &overrides);
        let (config, file_state) = match loaded {
            Ok(config) => (
                config,
                if contents.is_some() {
                    FileState::Loaded
                } else {
                    FileState::Missing
                },
            ),
            Err(e) => {
                let mut fallback = Config::default();
                // An override that does not parse is what the rejection names.
                if let Err(e) = overrides.apply(&mut fallback) {
                    tracing::debug!(error = %e, "an environment override does not parse");
                }
                (fallback, FileState::Rejected(e.to_string()))
            }
        };
        let in_force = file_state == FileState::Loaded;
        let settings = collect(&config, raw.as_ref(), in_force);
        let unknown = raw.as_ref().map_or_else(Vec::new, UnknownKey::find_in);
        let environment = EnvVar::read_by(&config);
        Self {
            config_path,
            file_state,
            config,
            settings,
            unknown,
            environment,
        }
    }

    /// Fill in the `client_id` each OAuth section leaves out from the
    /// client `quack auth register` registered at its issuer, marked as
    /// coming from the registration. A section whose issuer has none stays
    /// unset. `control.db` is only read, and not created when it does not
    /// exist yet.
    ///
    /// # Errors
    ///
    /// Returns an error when `control.db` exists but cannot be opened or
    /// read.
    pub async fn resolve_registered(&mut self) -> error::Result<()> {
        use crate::llm::oauth::registration::{ClientSection, registered_sections};
        use crate::storage::control::ControlPlane;
        let sections = registered_sections(&self.config);
        if sections.is_empty() || !self.config.control_db_path().exists() {
            return Ok(());
        }
        let control = ControlPlane::open(&self.config).await?;
        for registered in sections {
            let Some(row) = control.registration(registered.issuer.as_str()).await? else {
                continue;
            };
            let section = match &registered.section {
                ClientSection::SignIn => String::from("server.oidc"),
                ClientSection::Provider(name) => format!("providers.{name}.oauth"),
            };
            if let Some(setting) = self
                .settings
                .iter_mut()
                .find(|s| s.section == section && s.key == "client_id" && s.value.is_none())
            {
                setting.value = Some(quoted(&row.client_id));
                setting.origin = Origin::Registration;
            }
        }
        Ok(())
    }

    /// Whether the binary would start on this configuration.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        !matches!(self.file_state, FileState::Rejected(_))
    }

    /// The settings the file or the environment has a say in: what an
    /// operator changed, plus anything the file sets that is not in force.
    #[must_use = "returns the filtered settings"]
    pub fn changed(&self) -> impl Iterator<Item = &Setting> {
        self.settings
            .iter()
            .filter(|s| !s.is_default() || s.file_value.is_some())
    }

    /// The settings `filter` keeps, in file order.
    #[must_use]
    pub fn shown(&self, filter: SettingFilter) -> Vec<&Setting> {
        match filter {
            SettingFilter::All => self.settings.iter().collect(),
            SettingFilter::Changed => self.changed().collect(),
        }
    }

    /// The whole report as one document, as `quack config --format json` writes
    /// it.
    #[must_use]
    pub fn report(&self, filter: SettingFilter) -> Report<'_> {
        let (state, error) = match &self.file_state {
            FileState::Missing => ("missing", None),
            FileState::Loaded => ("loaded", None),
            FileState::Rejected(error) => ("rejected", Some(error.as_str())),
        };
        Report {
            config_file: ConfigFile {
                path: self.config_path.display().to_string(),
                state,
                error,
            },
            data_dir: self.config.data_dir().display().to_string(),
            settings: self.shown(filter),
            unrecognized: &self.unknown,
            environment: &self.environment,
        }
    }
}

/// Which settings a report lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingFilter {
    /// Every recognized setting.
    All,
    /// Only those the file or the environment has a say in.
    Changed,
}

/// An inspection as one document.
#[derive(Debug, Serialize)]
pub struct Report<'a> {
    pub config_file: ConfigFile<'a>,
    pub data_dir: String,
    pub settings: Vec<&'a Setting>,
    pub unrecognized: &'a [UnknownKey],
    pub environment: &'a [EnvVar],
}

/// Where the config file is and what became of it: `missing`, `loaded`,
/// or `rejected` with the error.
#[derive(Debug, Serialize)]
pub struct ConfigFile<'a> {
    pub path: String,
    pub state: &'static str,
    pub error: Option<&'a str>,
}

/// Every section and the keys it accepts. A test probes each section
/// through serde and compares the field names it reports with this list,
/// so the two cannot drift.
const SECTIONS: &[(&str, &[&str])] = &[
    ("general", &["data_dir", "default_workspace", "chat_model"]),
    (
        "ingestion",
        &[
            "chunk_size_tokens",
            "chunk_overlap_tokens",
            "embedding_batch_size",
            "embedding_concurrency",
            "tokenizer_encoding",
            "upload_max_mb",
            "max_decompressed_mb",
            "table_rows_as_table",
        ],
    ),
    (
        "embedding",
        &[
            "model",
            "dimension",
            "query_prefix",
            "document_prefix",
            "similarity_prefix",
        ],
    ),
    (
        "retrieval",
        &[
            "top_k",
            "rrf_k",
            "pinned_token_budget",
            "always_retrieve",
            "rerank",
            "rerank_candidates",
            "rerank_model",
        ],
    ),
    ("context", &["max_tokens"]),
    (
        "analysis",
        &[
            "max_query_rows",
            "query_timeout_seconds",
            "memory_limit_mb",
            "threads",
            "max_turns",
            "history_token_budget",
            "max_context_tokens",
            "extraction_timeout_seconds",
            "extraction_concurrency",
            "reader_pool_size",
            "effort",
            "background_effort",
            "compact_history",
        ],
    ),
    (
        "server",
        &[
            "bind",
            "local",
            "workers_per_workspace",
            "session_max_age_hours",
            "session_idle_minutes",
            "secure_cookies",
            "permission_timeout_seconds",
            "shutdown_grace_seconds",
        ],
    ),
    (
        "ontology",
        &[
            "key_overlap_threshold",
            "enum_max_values",
            "propose_sample_chunks",
            "min_support_documents",
        ],
    ),
    (
        "graph",
        &[
            "max_traversal_depth",
            "max_nodes",
            "merge_threshold",
            "auto_merge_threshold",
            "follow_ingest",
        ],
    ),
    (
        "import",
        &[
            "max_rows",
            "max_download_mb",
            "timeout_seconds",
            "allow_local_files",
            "allow_private_hosts",
        ],
    ),
    ("jobs", &["history"]),
];

/// The table of providers, whose sub-tables are named by the operator.
const PROVIDERS: &str = "providers";

/// The keys a `[providers.NAME]` table accepts.
const PROVIDER_KEYS: &[&str] = &[
    "type",
    "auth",
    "base_url",
    "api_key_env",
    "aws_profile",
    "api",
    "region",
    "max_concurrent_requests",
    "headers",
    "oauth",
    "temperature",
    "effort",
    "background_effort",
    "models",
];

/// The keys a `[providers.NAME.models."ID"]` table accepts.
const MODEL_KEYS: &[&str] = &["temperature", "effort", "background_effort"];

/// The keys a `[server.oidc]` table accepts.
const OIDC_KEYS: &[&str] = &[
    "issuer_url",
    "client_id",
    "client_secret_env",
    "client_auth",
    "scopes",
    "redirect_uri",
    "audience",
    "subject_claim",
];

/// The keys a `[providers.NAME.oauth]` table accepts.
const OAUTH_KEYS: &[&str] = &[
    "issuer_url",
    "client_id",
    "scopes",
    "redirect_uri",
    "grant",
    "client_secret_env",
    "client_auth",
    "exchange",
    "audience",
    "resource",
    "actor",
];

/// Every recognized setting with the value in force and where it came
/// from. `in_force` is false for a file the binary refuses: its keys are
/// still reported, but the values are the built-in ones.
fn collect(config: &Config, file: Option<&Table>, in_force: bool) -> Vec<Setting> {
    let defaults = Config::default();
    let mut inventory = Inventory {
        file,
        in_force,
        settings: Vec::new(),
    };
    general(&mut inventory, config, &defaults);
    providers(&mut inventory, config);
    ingestion(&mut inventory, config, &defaults);
    embedding(&mut inventory, config);
    retrieval(&mut inventory, config, &defaults);
    context(&mut inventory, config, &defaults);
    analysis(&mut inventory, config, &defaults);
    server(&mut inventory, config, &defaults);
    ontology(&mut inventory, config, &defaults);
    graph(&mut inventory, config, &defaults);
    import(&mut inventory, config, &defaults);
    jobs(&mut inventory, config, &defaults);
    inventory.settings
}

fn general(inventory: &mut Inventory<'_>, config: &Config, defaults: &Config) {
    let (general, default) = (&config.general, &defaults.general);
    let mut s = inventory.section("general");
    s.path(
        "data_dir",
        &general.data_dir,
        &default.data_dir,
        ENV_DATA_DIR,
    );
    s.text(
        "default_workspace",
        general.default_workspace.as_str(),
        default.default_workspace.as_str(),
        None,
    );
    let chat_model = general.chat_model.as_ref().map(ModelSpec::to_string);
    s.optional_text("chat_model", chat_model.as_deref(), Some(ENV_MODEL));
}

/// One section per configured provider, and one more for its `oauth`
/// table when it has one. A provider exists only because the file names
/// it, so these have no built-in values beyond serde's own defaults.
fn providers(inventory: &mut Inventory<'_>, config: &Config) {
    for (name, provider) in &config.providers {
        let section = format!("{PROVIDERS}.{name}");
        {
            let mut s = inventory.section(section.clone());
            s.required_text("type", &provider.provider_type.to_string());
            s.text(
                "auth",
                &provider.auth.mode().to_string(),
                &provider.provider_type.default_auth().to_string(),
                None,
            );
            s.optional_text(
                "base_url",
                provider.base_url.as_ref().map(BaseUrl::as_str),
                None,
            );
            s.optional_text("api_key_env", provider.auth.api_key_env(), None);
            s.optional_text("aws_profile", provider.auth.aws_profile(), None);
            if let (Some(bedrock), Some(endpoint)) =
                (&provider.bedrock, provider.provider_type.bedrock_endpoint())
            {
                s.text(
                    "api",
                    bedrock.api.as_str(),
                    endpoint.default_api().as_str(),
                    None,
                );
                s.optional_text(
                    "region",
                    bedrock.region.as_ref().map(AwsRegion::as_str),
                    None,
                );
            }
            if provider.provider_type == ProviderType::Openai {
                s.text(
                    "api",
                    provider.openai_chat_api().as_str(),
                    provider.openai_default_api().as_str(),
                    None,
                );
            }

            s.optional(
                "max_concurrent_requests",
                provider.max_concurrent_requests.map(|n| n.to_string()),
                Some(provider.default_request_limit().to_string()),
            );
            s.names_only("headers", provider.headers.as_ref());
            s.model_settings(provider.model_defaults);
        }
        for (model, settings) in &provider.models {
            inventory
                .section_at(&format!("{section}.models"), model)
                .model_settings(*settings);
        }
        let Some(oauth) = provider.auth.oauth() else {
            continue;
        };
        let mut s = inventory.section(format!("{section}.oauth"));
        s.required_text("issuer_url", &oauth.issuer_url);
        s.optional_text("client_id", oauth.client_id.as_deref(), None);
        s.optional(
            "scopes",
            Some(render_list(&oauth.scopes)),
            Some(String::from("[]")),
        );
        s.text(
            "redirect_uri",
            &oauth.redirect_uri,
            OAuthConfig::DEFAULT_REDIRECT_URI,
            None,
        );
        s.text(
            "grant",
            oauth.grant.as_str(),
            Grant::default().as_str(),
            None,
        );
        s.optional_text(
            "client_secret_env",
            oauth.client_secret_env.as_deref(),
            None,
        );
        s.text(
            "client_auth",
            oauth.client_auth.as_str(),
            ClientAuth::default().as_str(),
            None,
        );
        s.text(
            "exchange",
            oauth.exchange.as_str(),
            Exchange::default().as_str(),
            None,
        );
        s.optional_text("audience", oauth.audience.as_deref(), None);
        s.optional_text("resource", oauth.resource.as_deref(), None);
        s.literal("actor", oauth.actor, true);
    }
}

fn ingestion(inventory: &mut Inventory<'_>, config: &Config, defaults: &Config) {
    let (ingestion, default) = (&config.ingestion, &defaults.ingestion);
    let mut s = inventory.section("ingestion");
    s.literal(
        "chunk_size_tokens",
        ingestion.chunk_size_tokens,
        default.chunk_size_tokens,
    );
    s.literal(
        "chunk_overlap_tokens",
        ingestion.chunk_overlap_tokens,
        default.chunk_overlap_tokens,
    );
    s.literal(
        "embedding_batch_size",
        ingestion.embedding_batch_size,
        default.embedding_batch_size,
    );
    s.literal(
        "embedding_concurrency",
        ingestion.embedding_concurrency,
        default.embedding_concurrency,
    );
    s.text(
        "tokenizer_encoding",
        &ingestion.tokenizer_encoding,
        &default.tokenizer_encoding,
        None,
    );
    s.literal(
        "upload_max_mb",
        ingestion.upload_max_mb,
        default.upload_max_mb,
    );
    s.literal(
        "max_decompressed_mb",
        ingestion.max_decompressed_mb,
        default.max_decompressed_mb,
    );
    s.literal(
        "table_rows_as_table",
        ingestion.table_rows_as_table,
        default.table_rows_as_table,
    );
}

/// The model, its width, and the prefix in force for each role: the
/// file's, else the built-in one for the model's family, which is also the
/// default shown.
fn embedding(inventory: &mut Inventory<'_>, config: &Config) {
    let spec = config.embedding.model.as_ref();
    let model = spec.map_or("", ModelSpec::model);
    let prompts = ResolvedPrompts::for_model(config, model).prompts;
    let builtin = Family::of(model).map(Family::prompts);
    let mut s = inventory.section("embedding");
    let spec = spec.map(ModelSpec::to_string);
    s.optional_text("model", spec.as_deref(), None);
    s.optional(
        "dimension",
        config.embedding.dimension.map(|d| d.to_string()),
        None,
    );
    for (key, value, default) in [
        (
            "query_prefix",
            prompts.query,
            builtin.as_ref().map(|p| p.query.clone()),
        ),
        (
            "document_prefix",
            prompts.document,
            builtin.as_ref().map(|p| p.document.clone()),
        ),
        (
            "similarity_prefix",
            prompts.similarity,
            builtin.as_ref().map(|p| p.similarity.clone()),
        ),
    ] {
        s.optional(key, Some(quoted(&value)), default.as_deref().map(quoted));
    }
}

fn retrieval(inventory: &mut Inventory<'_>, config: &Config, defaults: &Config) {
    let (retrieval, default) = (&config.retrieval, &defaults.retrieval);
    let mut s = inventory.section("retrieval");
    s.literal("top_k", retrieval.top_k, default.top_k);
    s.literal("rrf_k", retrieval.rrf_k, default.rrf_k);
    s.literal(
        "pinned_token_budget",
        retrieval.pinned_token_budget,
        default.pinned_token_budget,
    );
    s.literal(
        "always_retrieve",
        retrieval.always_retrieve,
        default.always_retrieve,
    );
    s.text(
        "rerank",
        &retrieval.rerank.to_string(),
        &default.rerank.to_string(),
        None,
    );
    s.literal(
        "rerank_candidates",
        retrieval.rerank_candidates,
        default.rerank_candidates,
    );
    let rerank_model = retrieval.rerank_model.as_ref().map(ToString::to_string);
    s.optional_text("rerank_model", rerank_model.as_deref(), None);
}

fn context(inventory: &mut Inventory<'_>, config: &Config, defaults: &Config) {
    let mut s = inventory.section("context");
    s.literal(
        "max_tokens",
        config.context.max_tokens,
        defaults.context.max_tokens,
    );
}

fn analysis(inventory: &mut Inventory<'_>, config: &Config, defaults: &Config) {
    let (analysis, default) = (&config.analysis, &defaults.analysis);
    let mut s = inventory.section("analysis");
    s.literal(
        "max_query_rows",
        analysis.max_query_rows,
        default.max_query_rows,
    );
    s.literal(
        "query_timeout_seconds",
        analysis.query_timeout_seconds,
        default.query_timeout_seconds,
    );
    s.literal(
        "memory_limit_mb",
        analysis.memory_limit_mb,
        default.memory_limit_mb,
    );
    s.literal("threads", analysis.threads, default.threads);
    s.literal("max_turns", analysis.max_turns, default.max_turns);
    s.literal(
        "history_token_budget",
        analysis.history_token_budget,
        default.history_token_budget,
    );
    s.literal(
        "max_context_tokens",
        analysis.max_context_tokens,
        default.max_context_tokens,
    );
    s.literal(
        "extraction_timeout_seconds",
        analysis.extraction_timeout_seconds,
        default.extraction_timeout_seconds,
    );
    s.literal(
        "extraction_concurrency",
        analysis.extraction_concurrency,
        default.extraction_concurrency,
    );
    s.literal(
        "reader_pool_size",
        analysis.reader_pool_size,
        default.reader_pool_size,
    );
    s.optional_text("effort", analysis.effort.map(Effort::as_str), None);
    s.optional_text(
        "background_effort",
        analysis.background_effort.map(Effort::as_str),
        None,
    );
    s.literal(
        "compact_history",
        analysis.compact_history,
        default.compact_history,
    );
}

fn server(inventory: &mut Inventory<'_>, config: &Config, defaults: &Config) {
    let (server, default) = (&config.server, &defaults.server);
    let mut s = inventory.section("server");
    s.text("bind", &server.bind, &default.bind, Some(ENV_BIND));
    s.literal("local", server.local, default.local);
    s.literal(
        "workers_per_workspace",
        server.workers_per_workspace,
        default.workers_per_workspace,
    );
    s.literal(
        "session_max_age_hours",
        server.session_max_age_hours,
        default.session_max_age_hours,
    );
    s.literal(
        "session_idle_minutes",
        server.session_idle_minutes,
        default.session_idle_minutes,
    );
    s.text(
        "secure_cookies",
        server.secure_cookies.as_str(),
        default.secure_cookies.as_str(),
        None,
    );
    s.literal(
        "permission_timeout_seconds",
        server.permission_timeout_seconds,
        default.permission_timeout_seconds,
    );
    s.literal(
        "shutdown_grace_seconds",
        server.shutdown_grace_seconds,
        default.shutdown_grace_seconds,
    );
    let Some(oidc) = &server.oidc else {
        return;
    };
    let mut s = inventory.section(String::from("server.oidc"));
    s.required_text("issuer_url", &oidc.issuer_url);
    s.optional_text("client_id", oidc.client_id.as_deref(), None);
    s.optional_text("client_secret_env", oidc.client_secret_env.as_deref(), None);
    s.text(
        "client_auth",
        oidc.client_auth.as_str(),
        ClientAuth::default().as_str(),
        None,
    );
    s.optional(
        "scopes",
        Some(render_list(&oidc.scopes)),
        Some(render_list(&OidcConfig::default_scopes())),
    );
    s.required_text("redirect_uri", &oidc.redirect_uri);
    s.optional_text("audience", oidc.audience.as_deref(), None);
    s.text(
        "subject_claim",
        &oidc.subject_claim,
        OidcConfig::DEFAULT_SUBJECT_CLAIM,
        None,
    );
}

fn jobs(inventory: &mut Inventory<'_>, config: &Config, defaults: &Config) {
    let (jobs, default) = (&config.jobs, &defaults.jobs);
    let mut s = inventory.section("jobs");
    s.literal("history", jobs.history, default.history);
}

fn ontology(inventory: &mut Inventory<'_>, config: &Config, defaults: &Config) {
    let (ontology, default) = (&config.ontology, &defaults.ontology);
    let mut s = inventory.section("ontology");
    s.literal(
        "key_overlap_threshold",
        ontology.key_overlap_threshold,
        default.key_overlap_threshold,
    );
    s.literal(
        "enum_max_values",
        ontology.enum_max_values,
        default.enum_max_values,
    );
    s.literal(
        "propose_sample_chunks",
        ontology.propose_sample_chunks,
        default.propose_sample_chunks,
    );
    s.literal(
        "min_support_documents",
        ontology.min_support_documents,
        default.min_support_documents,
    );
}

fn graph(inventory: &mut Inventory<'_>, config: &Config, defaults: &Config) {
    let (graph, default) = (&config.graph, &defaults.graph);
    let mut s = inventory.section("graph");
    s.literal(
        "max_traversal_depth",
        graph.max_traversal_depth,
        default.max_traversal_depth,
    );
    s.literal("max_nodes", graph.max_nodes, default.max_nodes);
    s.literal(
        "merge_threshold",
        graph.merge_threshold,
        default.merge_threshold,
    );
    s.literal(
        "auto_merge_threshold",
        graph.auto_merge_threshold,
        default.auto_merge_threshold,
    );
    s.text(
        "follow_ingest",
        graph.follow_ingest.as_str(),
        default.follow_ingest.as_str(),
        None,
    );
}

fn import(inventory: &mut Inventory<'_>, config: &Config, defaults: &Config) {
    let (import, default) = (&config.import, &defaults.import);
    let mut s = inventory.section("import");
    s.literal("max_rows", import.max_rows, default.max_rows);
    s.literal(
        "max_download_mb",
        import.max_download_mb,
        default.max_download_mb,
    );
    s.literal(
        "timeout_seconds",
        import.timeout_seconds,
        default.timeout_seconds,
    );
    s.literal(
        "allow_local_files",
        import.allow_local_files,
        default.allow_local_files,
    );
    s.literal(
        "allow_private_hosts",
        import.allow_private_hosts,
        default.allow_private_hosts,
    );
}

impl EnvVar {
    /// The environment variables this configuration reads: the four that
    /// override settings, the proxy variables (set under either case), and
    /// the ones the providers name for their credentials. Only whether each
    /// is set, never what it holds.
    fn read_by(config: &Config) -> Vec<Self> {
        let mut vars = vec![
            Self::new(ENV_CONFIG_DIR, "the directory holding config.toml"),
            Self::new(ENV_DATA_DIR, "overrides [general].data_dir"),
            Self::new(ENV_MODEL, "overrides [general].chat_model"),
            Self::new(ENV_BIND, "overrides [server].bind"),
        ];
        for (upper, lower, purpose) in [
            (
                "HTTPS_PROXY",
                "https_proxy",
                "the proxy for https:// requests",
            ),
            ("HTTP_PROXY", "http_proxy", "the proxy for http:// requests"),
            (
                "ALL_PROXY",
                "all_proxy",
                "the proxy for a scheme without its own variable",
            ),
            (
                "NO_PROXY",
                "no_proxy",
                "hosts reached directly, beside loopback and 169.254.0.0/16",
            ),
        ] {
            vars.push(Self::either_case(upper, lower, purpose));
        }
        if let Some(secret) = config
            .server
            .oidc
            .as_ref()
            .and_then(|o| o.client_secret_env.as_ref())
        {
            vars.push(Self::new(secret, "[server.oidc].client_secret_env"));
        }
        for (name, provider) in &config.providers {
            if let Some(key) = provider.auth.api_key_env() {
                vars.push(Self::new(key, &format!("[providers.{name}].api_key_env")));
            }
            if let Some(secret) = provider
                .auth
                .oauth()
                .and_then(|o| o.client_secret_env.as_ref())
            {
                vars.push(Self::new(
                    secret,
                    &format!("[providers.{name}.oauth].client_secret_env"),
                ));
            }
        }
        if config
            .providers
            .values()
            .any(|p| p.auth.mode() == AuthMode::Aws)
        {
            // What the AWS SDK's chain reads first for a Bedrock provider.
            for (var, purpose) in [
                ("AWS_PROFILE", "the AWS profile when aws_profile is unset"),
                ("AWS_REGION", "the AWS region when region is unset"),
                (
                    "AWS_ACCESS_KEY_ID",
                    "static AWS credentials, ahead of any profile",
                ),
            ] {
                vars.push(Self::new(var, purpose));
            }
        }
        vars
    }

    /// A variable read under either case, listed by its upper-case name.
    fn either_case(upper: &str, lower: &str, purpose: &str) -> Self {
        Self {
            name: upper.to_owned(),
            set: [upper, lower]
                .into_iter()
                .any(|name| std::env::var_os(name).is_some()),
            purpose: purpose.to_owned(),
        }
    }

    fn new(name: &str, purpose: &str) -> Self {
        Self {
            name: name.to_owned(),
            set: std::env::var_os(name).is_some(),
            purpose: purpose.to_owned(),
        }
    }
}

/// How many leading characters a suggestion must share with the unknown
/// key: enough for `topk` to find `top_k` without naming something
/// unrelated.
const PREFIX_MATCH: usize = 3;

impl UnknownKey {
    /// Every key in the file that no setting corresponds to, in file order.
    fn find_in(file: &Table) -> Vec<Self> {
        let mut unknown = Vec::new();
        for (name, value) in file {
            if name == PROVIDERS {
                let Some(providers) = value.as_table() else {
                    continue;
                };
                for (provider, entry) in providers {
                    let Some(entry) = entry.as_table() else {
                        continue;
                    };
                    let section = format!("{PROVIDERS}.{provider}");
                    for key in entry.keys() {
                        if !PROVIDER_KEYS.contains(&key.as_str()) {
                            unknown.push(Self::new(&section, key, PROVIDER_KEYS));
                        }
                    }
                    let models = entry.get("models").and_then(TomlValue::as_table);
                    for (model, settings) in models.into_iter().flatten() {
                        let Some(settings) = settings.as_table() else {
                            continue;
                        };
                        let path = format!("{section}.models.{}", quoted(model));
                        for key in settings.keys() {
                            if !MODEL_KEYS.contains(&key.as_str()) {
                                unknown.push(Self::new(&path, key, MODEL_KEYS));
                            }
                        }
                    }
                    let Some(oauth) = entry.get("oauth").and_then(TomlValue::as_table) else {
                        continue;
                    };
                    for key in oauth.keys() {
                        if !OAUTH_KEYS.contains(&key.as_str()) {
                            unknown.push(Self::new(&format!("{section}.oauth"), key, OAUTH_KEYS));
                        }
                    }
                }
                continue;
            }
            let Some((_, keys)) = SECTIONS.iter().find(|(section, _)| *section == name) else {
                unknown.push(Self {
                    path: name.clone(),
                    suggestion: Self::closest(name, &Self::section_names()),
                });
                continue;
            };
            let Some(table) = value.as_table() else {
                continue;
            };
            for key in table.keys() {
                if name == "server" && key == "oidc" {
                    continue;
                }
                if !keys.contains(&key.as_str()) {
                    unknown.push(Self::new(name, key, keys));
                }
            }
            if name == "server"
                && let Some(oidc) = table.get("oidc").and_then(TomlValue::as_table)
            {
                for key in oidc.keys() {
                    if !OIDC_KEYS.contains(&key.as_str()) {
                        unknown.push(Self::new("server.oidc", key, OIDC_KEYS));
                    }
                }
            }
        }
        unknown
    }

    /// `key` in `section`, which accepts `keys`, with the section that
    /// accepts it exactly, else the closest key here.
    fn new(section: &str, key: &str, keys: &[&str]) -> Self {
        let suggestion = Self::elsewhere(section, key).or_else(|| Self::closest(key, keys));
        Self {
            path: format!("{section}.{key}"),
            suggestion,
        }
    }

    /// Another section that accepts this exact key, for a setting written
    /// under the wrong heading.
    fn elsewhere(section: &str, key: &str) -> Option<String> {
        SECTIONS
            .iter()
            .find(|(name, keys)| *name != section && keys.contains(&key))
            .map(|(name, _)| format!("[{name}].{key}"))
    }

    fn section_names() -> Vec<&'static str> {
        let mut names: Vec<&'static str> = SECTIONS.iter().map(|(name, _)| *name).collect();
        names.push(PROVIDERS);
        names
    }

    /// The candidate sharing the longest prefix with `name`, when that is
    /// at least [`PREFIX_MATCH`] characters.
    fn closest(name: &str, candidates: &[&str]) -> Option<String> {
        candidates
            .iter()
            .map(|candidate| {
                let shared = name
                    .chars()
                    .zip(candidate.chars())
                    .take_while(|(x, y)| x == y)
                    .count();
                (shared, *candidate)
            })
            .filter(|(shared, _)| *shared >= PREFIX_MATCH)
            .max_by_key(|(shared, _)| *shared)
            .map(|(_, candidate)| candidate.to_owned())
    }
}

/// Builds the settings list, resolving each setting's origin against the
/// file and the environment as it goes.
struct Inventory<'a> {
    file: Option<&'a Table>,
    in_force: bool,
    settings: Vec<Setting>,
}

impl<'a> Inventory<'a> {
    /// The section at a dotted `name` whose parts are all bare keys.
    fn section(&mut self, name: impl Into<String>) -> Section<'_, 'a> {
        let display: String = name.into();
        let path = display.split('.').map(str::to_owned).collect();
        Section {
            name: SectionName { display, path },
            inventory: self,
        }
    }

    /// The section at `path`, whose last part may be any string, such as a
    /// model id: it is shown quoted.
    fn section_at(&mut self, parent: &str, last: &str) -> Section<'_, 'a> {
        let mut path: Vec<String> = parent.split('.').map(str::to_owned).collect();
        path.push(last.to_owned());
        Section {
            name: SectionName {
                display: format!("{parent}.{}", quoted(last)),
                path,
            },
            inventory: self,
        }
    }

    fn push(
        &mut self,
        section: &SectionName,
        key: &'static str,
        value: Option<String>,
        default: Option<String>,
        env: Option<&'static str>,
    ) {
        let file_value = self.file_value(section, key).map(ToString::to_string);
        let origin = match env {
            Some(name) if std::env::var_os(name).is_some() => Origin::Env(name),
            _ if self.in_force && file_value.is_some() => Origin::File,
            _ => Origin::Default,
        };
        self.settings.push(Setting {
            section: section.display.clone(),
            key,
            value,
            default,
            origin,
            file_value,
            env,
        });
    }

    /// The raw value the file gives for a section and key.
    fn file_value(&self, section: &SectionName, key: &str) -> Option<&'a TomlValue> {
        let mut table = self.file?;
        for part in &section.path {
            table = table.get(part)?.as_table()?;
        }
        table.get(key)
    }
}

/// A section as [`Setting::section`] shows it and as the file nests it.
struct SectionName {
    display: String,
    path: Vec<String>,
}

/// One section's settings, so each call names only the key.
struct Section<'s, 'a> {
    name: SectionName,
    inventory: &'s mut Inventory<'a>,
}

impl Section<'_, '_> {
    fn optional(&mut self, key: &'static str, value: Option<String>, default: Option<String>) {
        self.inventory.push(&self.name, key, value, default, None);
    }

    /// A setting with no built-in value, such as a provider's `type`.
    fn required_text(&mut self, key: &'static str, value: &str) {
        self.inventory
            .push(&self.name, key, Some(quoted(value)), None, None);
    }

    fn optional_text(&mut self, key: &'static str, value: Option<&str>, env: Option<&'static str>) {
        let value = value.map(quoted);
        self.inventory.push(&self.name, key, value, None, env);
    }

    /// A text setting, and the environment variable that overrides it.
    fn text(&mut self, key: &'static str, value: &str, default: &str, env: Option<&'static str>) {
        self.inventory.push(
            &self.name,
            key,
            Some(quoted(value)),
            Some(quoted(default)),
            env,
        );
    }

    fn path(&mut self, key: &'static str, value: &Path, default: &Path, env: &'static str) {
        self.inventory.push(
            &self.name,
            key,
            Some(quoted(&value.display().to_string())),
            Some(quoted(&default.display().to_string())),
            Some(env),
        );
    }

    /// A table whose values can be secrets, such as a provider's headers:
    /// its keys alone, both in force and as the file wrote them.
    fn names_only(&mut self, key: &'static str, value: Option<&BTreeMap<String, String>>) {
        let names =
            |keys: Vec<&String>| render_list(&keys.into_iter().cloned().collect::<Vec<_>>());
        let value = value.map(|table| names(table.keys().collect()));
        let written = self
            .inventory
            .file_value(&self.name, key)
            .map(|raw| match raw.as_table() {
                Some(table) => names(table.keys().collect()),
                None => String::from("(not a table)"),
            });
        self.inventory.push(&self.name, key, value, None, None);
        if let Some(setting) = self.inventory.settings.last_mut() {
            setting.file_value = written;
        }
    }

    /// `temperature`, `effort`, and `background_effort`, for a provider or
    /// one of its models.
    fn model_settings(&mut self, settings: ModelSettings) {
        self.optional(
            "temperature",
            settings.temperature.map(|t| t.to_string()),
            None,
        );
        self.optional_text("effort", settings.effort.map(Effort::as_str), None);
        self.optional_text(
            "background_effort",
            settings.background_effort.map(Effort::as_str),
            None,
        );
    }

    /// A number or a flag: shown as TOML writes it, unquoted.
    fn literal<T: fmt::Display>(&mut self, key: &'static str, value: T, default: T) {
        self.inventory.push(
            &self.name,
            key,
            Some(value.to_string()),
            Some(default.to_string()),
            None,
        );
    }
}

/// A string as TOML writes it, escaping included.
fn quoted(value: &str) -> String {
    TomlValue::String(value.to_owned()).to_string()
}

fn render_list(values: &[String]) -> String {
    let items: Vec<String> = values.iter().map(|v| quoted(v)).collect();
    format!("[{}]", items.join(", "))
}

#[cfg(test)]
mod tests;
