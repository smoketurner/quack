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

use std::fmt;
use std::path::{Path, PathBuf};

use toml::{Table, Value as TomlValue};

use crate::embedding::ResolvedPrompts;
use crate::embedding::presets::Family;

use super::{
    AuthMode, BaseUrl, Config, ENV_BIND, ENV_CONFIG_DIR, ENV_DATA_DIR, ENV_MODEL, ModelSpec,
    OAuthConfig, Overrides, config_file_path,
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
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => f.write_str("default"),
            Self::File => f.write_str("file"),
            Self::Env(var) => write!(f, "env {var}"),
        }
    }
}

/// One setting this binary recognizes, and what it makes of it.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownKey {
    /// Dotted path as the file writes it: `retrieval.topk`.
    pub path: String,
    /// The recognized key it most resembles, when one is close enough to
    /// name.
    pub suggestion: Option<String>,
}

/// An environment variable this binary reads, and whether it is set. The
/// value is never recorded: some of these hold credentials.
#[derive(Debug, Clone, PartialEq, Eq)]
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
}

/// Every section and the keys it accepts. A test probes each section
/// through serde and compares the field names it reports with this list,
/// so the two cannot drift.
const SECTIONS: &[(&str, &[&str])] = &[
    (
        "general",
        &[
            "data_dir",
            "default_workspace",
            "chat_model",
            "embedding_model",
        ],
    ),
    (
        "ingestion",
        &[
            "chunk_size_tokens",
            "chunk_overlap_tokens",
            "embedding_batch_size",
            "embedding_concurrency",
            "tokenizer_encoding",
            "upload_max_mb",
        ],
    ),
    (
        "embedding",
        &["query_prefix", "document_prefix", "similarity_prefix"],
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
    "embedding_dimension",
    "max_concurrent_requests",
    "oauth",
];

/// The keys a `[providers.NAME.oauth]` table accepts.
const OAUTH_KEYS: &[&str] = &[
    "issuer_url",
    "client_id",
    "scopes",
    "redirect_uri",
    "device_code",
    "client_secret_env",
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
        &general.default_workspace,
        &default.default_workspace,
        None,
    );
    let chat_model = general.chat_model.as_ref().map(ModelSpec::to_string);
    s.optional_text("chat_model", chat_model.as_deref(), Some(ENV_MODEL));
    let embedding_model = general.embedding_model.as_ref().map(ModelSpec::to_string);
    s.optional_text("embedding_model", embedding_model.as_deref(), None);
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
                &AuthMode::default().to_string(),
                None,
            );
            s.optional_text(
                "base_url",
                provider.base_url.as_ref().map(BaseUrl::as_str),
                None,
            );
            s.optional_text("api_key_env", provider.auth.api_key_env(), None);
            s.optional(
                "embedding_dimension",
                provider.embedding_dimension.map(|d| d.to_string()),
                None,
            );
            s.optional(
                "max_concurrent_requests",
                provider.max_concurrent_requests.map(|n| n.to_string()),
                Some(provider.default_request_limit().to_string()),
            );
        }
        let Some(oauth) = provider.auth.oauth() else {
            continue;
        };
        let mut s = inventory.section(format!("{section}.oauth"));
        s.required_text("issuer_url", &oauth.issuer_url);
        s.required_text("client_id", &oauth.client_id);
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
        s.literal("device_code", oauth.device_code, false);
        s.optional_text(
            "client_secret_env",
            oauth.client_secret_env.as_deref(),
            None,
        );
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
}

/// The prefix in force for each role: the file's, else the built-in one
/// for the configured model's family, which is also the default shown.
fn embedding(inventory: &mut Inventory<'_>, config: &Config) {
    let model = config
        .general
        .embedding_model
        .as_ref()
        .map_or("", ModelSpec::model);
    let prompts = ResolvedPrompts::for_model(config, model).prompts;
    let builtin = Family::of(model).map(Family::prompts);
    let mut s = inventory.section("embedding");
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
    /// override settings, and the ones the providers name for their
    /// credentials. Only whether each is set, never what it holds.
    fn read_by(config: &Config) -> Vec<Self> {
        let mut vars = vec![
            Self::new(ENV_CONFIG_DIR, "the directory holding config.toml"),
            Self::new(ENV_DATA_DIR, "overrides [general].data_dir"),
            Self::new(ENV_MODEL, "overrides [general].chat_model"),
            Self::new(ENV_BIND, "overrides [server].bind"),
        ];
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
        vars
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
                if !keys.contains(&key.as_str()) {
                    unknown.push(Self::new(name, key, keys));
                }
            }
        }
        unknown
    }

    /// `key` in `section`, which accepts `keys`, with the closest one or
    /// the section that does accept it.
    fn new(section: &str, key: &str, keys: &[&str]) -> Self {
        let suggestion = Self::closest(key, keys).or_else(|| Self::elsewhere(section, key));
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
    fn section(&mut self, name: impl Into<String>) -> Section<'_, 'a> {
        Section {
            name: name.into(),
            inventory: self,
        }
    }

    fn push(
        &mut self,
        section: &str,
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
            section: section.to_owned(),
            key,
            value,
            default,
            origin,
            file_value,
            env,
        });
    }

    /// The raw value the file gives for a dotted section and key.
    fn file_value(&self, section: &str, key: &str) -> Option<&'a TomlValue> {
        let mut table = self.file?;
        for part in section.split('.') {
            table = table.get(part)?.as_table()?;
        }
        table.get(key)
    }
}

/// One section's settings, so each call names only the key.
struct Section<'s, 'a> {
    name: String,
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
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "tests assert on values they have just built"
)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    const SAMPLE: &str = r#"
[general]
chat_model = "ollama/llama3.1:8b"

[providers.ollama]
type = "ollama"
base_url = "http://localhost:11434"
embedding_dimension = 768

[retrieval]
top_k = 3
"#;

    fn inspect(contents: &str) -> Inspection {
        Inspection::of(PathBuf::from("/tmp/config.toml"), Some(contents))
    }

    fn setting<'a>(inspection: &'a Inspection, path: &str) -> &'a Setting {
        inspection
            .settings
            .iter()
            .find(|s| s.path() == path)
            .unwrap_or_else(|| panic!("no setting {path}"))
    }

    #[test]
    fn a_loaded_file_marks_its_own_keys_and_leaves_the_rest_default() {
        let inspection = inspect(SAMPLE);
        assert_eq!(inspection.file_state, FileState::Loaded);
        assert!(inspection.is_usable());

        let top_k = setting(&inspection, "retrieval.top_k");
        assert_eq!(top_k.origin, Origin::File);
        assert_eq!(top_k.value.as_deref(), Some("3"));
        assert_eq!(top_k.default.as_deref(), Some("8"));
        assert_eq!(top_k.file_value.as_deref(), Some("3"));

        let rrf_k = setting(&inspection, "retrieval.rrf_k");
        assert_eq!(rrf_k.origin, Origin::Default);
        assert_eq!(rrf_k.value.as_deref(), Some("60"));
        assert!(rrf_k.file_value.is_none());
    }

    #[test]
    fn providers_and_their_defaults_are_listed() {
        let inspection = inspect(SAMPLE);
        let kind = setting(&inspection, "providers.ollama.type");
        assert_eq!(kind.origin, Origin::File);
        assert_eq!(kind.value.as_deref(), Some("\"ollama\""));
        assert!(kind.default.is_none());

        let auth = setting(&inspection, "providers.ollama.auth");
        assert_eq!(auth.origin, Origin::Default);
        assert_eq!(auth.value.as_deref(), Some("\"none\""));

        let key_env = setting(&inspection, "providers.ollama.api_key_env");
        assert_eq!(key_env.display_value(), UNSET);
    }

    #[test]
    fn oauth_sections_are_listed_with_their_defaults() {
        let inspection = inspect(
            "[providers.azure]\ntype = \"openai\"\nauth = \"oauth\"\n\
             [providers.azure.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n\
             scopes = [\"a\", \"b\"]\n",
        );
        assert_eq!(inspection.file_state, FileState::Loaded);
        let scopes = setting(&inspection, "providers.azure.oauth.scopes");
        assert_eq!(scopes.value.as_deref(), Some("[\"a\", \"b\"]"));
        let redirect = setting(&inspection, "providers.azure.oauth.redirect_uri");
        assert_eq!(redirect.origin, Origin::Default);
        assert_eq!(
            redirect.value,
            Some(quoted(OAuthConfig::DEFAULT_REDIRECT_URI))
        );
    }

    #[test]
    fn a_missing_file_is_all_defaults() {
        let inspection = Inspection::of(PathBuf::from("/tmp/config.toml"), None);
        assert_eq!(inspection.file_state, FileState::Missing);
        assert!(inspection.unknown.is_empty());
        assert!(inspection.settings.iter().all(|s| s.file_value.is_none()));
        assert_eq!(
            setting(&inspection, "retrieval.top_k").value.as_deref(),
            Some("8")
        );
    }

    #[test]
    fn a_rejected_file_reports_the_error_and_falls_back_to_defaults() {
        let inspection = inspect("[retrieval]\ntopk = 3\n");
        let FileState::Rejected(error) = &inspection.file_state else {
            panic!("expected a rejected file, got {:?}", inspection.file_state);
        };
        assert!(error.contains("topk"), "{error}");
        assert!(!inspection.is_usable());

        // The file's values are not in force, so the listing shows the
        // built-in ones and says where the file disagrees.
        let top_k = setting(&inspection, "retrieval.top_k");
        assert_eq!(top_k.origin, Origin::Default);
        assert_eq!(top_k.value.as_deref(), Some("8"));

        assert_eq!(
            inspection.unknown,
            vec![UnknownKey {
                path: String::from("retrieval.topk"),
                suggestion: Some(String::from("top_k")),
            }]
        );
    }

    #[test]
    fn a_file_that_is_not_toml_is_rejected_without_a_panic() {
        let inspection = inspect("[retrieval\ntop_k = ");
        assert!(matches!(inspection.file_state, FileState::Rejected(_)));
        assert!(inspection.unknown.is_empty());
        assert_eq!(
            setting(&inspection, "retrieval.top_k").value.as_deref(),
            Some("8")
        );
    }

    #[test]
    fn unknown_sections_keys_and_misplaced_settings_are_named() {
        let inspection = inspect(
            "[genral]\nchat_model = \"a/b\"\n[analysis]\ntop_k = 1\n\
             [providers.o]\ntype = \"ollama\"\nmodel = \"x\"\n\
             [providers.o.oauth]\ntenant = \"t\"\n",
        );
        let found: Vec<(&str, Option<&str>)> = inspection
            .unknown
            .iter()
            .map(|u| (u.path.as_str(), u.suggestion.as_deref()))
            .collect();
        assert!(found.contains(&("genral", Some("general"))), "{found:?}");
        assert!(
            found.contains(&("analysis.top_k", Some("[retrieval].top_k"))),
            "{found:?}"
        );
        assert!(found.contains(&("providers.o.model", None)), "{found:?}");
        assert!(
            found.contains(&("providers.o.oauth.tenant", None)),
            "{found:?}"
        );
    }

    #[test]
    fn every_recognized_key_has_a_setting() {
        let inspection = Inspection::of(PathBuf::from("/tmp/config.toml"), None);
        let listed: BTreeSet<String> = inspection.settings.iter().map(Setting::path).collect();
        for (section, keys) in SECTIONS {
            for key in *keys {
                let path = format!("{section}.{key}");
                assert!(listed.contains(&path), "no setting for {path}");
            }
        }
        // And nothing beyond them: the providers are the only other
        // sections, and a default config has none.
        assert_eq!(
            listed.len(),
            SECTIONS.iter().map(|(_, k)| k.len()).sum::<usize>()
        );
    }

    #[test]
    fn every_provider_key_has_a_setting() {
        let inspection = inspect(
            "[providers.p]\ntype = \"openai\"\nauth = \"oauth\"\n\
             base_url = \"https://e\"\nembedding_dimension = 1536\n\
             [providers.p.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n",
        );
        let listed: BTreeSet<String> = inspection.settings.iter().map(Setting::path).collect();
        for key in PROVIDER_KEYS.iter().filter(|k| **k != "oauth") {
            assert!(listed.contains(&format!("providers.p.{key}")), "{key}");
        }
        for key in OAUTH_KEYS {
            assert!(
                listed.contains(&format!("providers.p.oauth.{key}")),
                "{key}"
            );
        }
    }

    /// The field names serde reports for a section, taken from the
    /// `deny_unknown_fields` error a probe key provokes. This is what
    /// keeps [`SECTIONS`] honest: a field added to a config struct and
    /// not to the list, or the other way round, fails the next test.
    fn fields_of(header: &str) -> BTreeSet<String> {
        let probe = format!("{header}\nquack_probe_key = 1\n");
        let error = Config::parse(&probe)
            .expect_err("a probe key is an unknown field")
            .to_string();
        let (_, expected) = error
            .split_once("expected")
            .expect("serde names the fields it expected");
        expected
            .split('`')
            .skip(1)
            .step_by(2)
            .map(String::from)
            .collect()
    }

    #[test]
    fn the_key_list_matches_the_config_structs() {
        for (section, keys) in SECTIONS {
            let declared: BTreeSet<String> = keys.iter().map(|k| (*k).to_owned()).collect();
            assert_eq!(fields_of(&format!("[{section}]")), declared, "[{section}]");
        }
        let providers: BTreeSet<String> = PROVIDER_KEYS.iter().map(|k| (*k).to_owned()).collect();
        assert_eq!(
            fields_of("[providers.p]\ntype = \"ollama\""),
            providers,
            "[providers.NAME]"
        );
        let oauth: BTreeSet<String> = OAUTH_KEYS.iter().map(|k| (*k).to_owned()).collect();
        assert_eq!(
            fields_of(
                "[providers.p]\ntype = \"ollama\"\n[providers.p.oauth]\nissuer_url = \"i\"\nclient_id = \"c\""
            ),
            oauth,
            "[providers.NAME.oauth]"
        );
    }

    #[test]
    fn the_section_list_matches_the_config_struct() {
        let declared: BTreeSet<String> = UnknownKey::section_names()
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        assert_eq!(fields_of(""), declared);
    }

    #[test]
    fn changed_lists_what_the_file_has_a_say_in() {
        let inspection = inspect(SAMPLE);
        let changed: BTreeSet<String> = inspection.changed().map(Setting::path).collect();
        assert!(changed.contains("retrieval.top_k"));
        assert!(changed.contains("general.chat_model"));
        assert!(!changed.contains("retrieval.rrf_k"));
    }

    #[test]
    fn the_environment_listing_names_provider_variables_without_reading_them() {
        let inspection = inspect(
            "[providers.a]\ntype = \"anthropic\"\nauth = \"api-key\"\napi_key_env = \"QUACK_TEST_KEY\"\n",
        );
        let names: Vec<&str> = inspection
            .environment
            .iter()
            .map(|v| v.name.as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                ENV_CONFIG_DIR,
                ENV_DATA_DIR,
                ENV_MODEL,
                ENV_BIND,
                "QUACK_TEST_KEY"
            ]
        );
    }
}
