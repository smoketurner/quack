use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::error::{Error, Result};
use crate::graph::GraphOptions;
use crate::ontology::documents::DocumentEvidenceOptions;
use crate::ontology::induction::TableEvidenceOptions;

/// The whole `config.toml`. Unknown keys anywhere are an error so a typo can
/// never silently disable a setting.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub general: GeneralConfig,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderConfig>,
    pub ingestion: IngestionConfig,
    pub retrieval: RetrievalConfig,
    pub context: ContextConfig,
    pub analysis: AnalysisConfig,
    pub server: ServerConfig,
    pub ontology: OntologyConfig,
    pub graph: GraphConfig,
    pub import: ImportConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeneralConfig {
    pub data_dir: PathBuf,
    pub default_workspace: String,
    /// `PROVIDER/MODEL` used for chat and tool calling. Override: `QUACK_MODEL`.
    pub chat_model: Option<String>,
    /// `PROVIDER/MODEL` used for embeddings. Unset means documents are stored
    /// without vectors and `search_documents` is unavailable.
    pub embedding_model: Option<String>,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            default_workspace: String::from("default"),
            chat_model: None,
            embedding_model: None,
        }
    }
}

/// Which rig client a provider entry builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderType {
    Ollama,
    #[serde(alias = "openai-compat")]
    Openai,
    Anthropic,
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Ollama => "ollama",
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
        })
    }
}

/// How a provider endpoint is authenticated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AuthMode {
    /// No credentials (local Ollama, unauthenticated gateways).
    #[default]
    None,
    /// Static key from the environment variable named by `api_key_env`.
    ApiKey,
    /// OAuth 2.0 against an identity provider (design doc 10.2): the access
    /// token from `[providers.NAME.oauth]` is the bearer for the endpoint.
    Oauth,
}

/// `[providers.NAME.oauth]`: Authorization Code with PKCE, or the device-code
/// flow, against an `OpenID` Connect issuer.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OAuthConfig {
    /// Issuer whose `/.well-known/openid-configuration` names the endpoints,
    /// e.g. `https://login.microsoftonline.com/{tenant}/v2.0`.
    pub issuer_url: String,
    pub client_id: String,
    /// Scopes requested at login, e.g.
    /// `["https://cognitiveservices.azure.com/.default", "offline_access"]`.
    /// Include `offline_access` where the issuer needs it to return a refresh
    /// token.
    #[serde(default)]
    pub scopes: Vec<String>,
    /// Loopback redirect for the browser flow.
    #[serde(default = "default_redirect_uri")]
    pub redirect_uri: String,
    /// Always use the device-code flow (headless hosts, SSH, servers).
    #[serde(default)]
    pub device_code: bool,
    /// Environment variable holding a client secret: the server as a
    /// confidential client.
    pub client_secret_env: Option<String>,
}

fn default_redirect_uri() -> String {
    String::from("http://127.0.0.1:19876/callback")
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    #[serde(rename = "type")]
    pub provider_type: ProviderType,
    #[serde(default)]
    pub auth: AuthMode,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    /// Width of the vectors this provider's embedding models produce.
    pub embedding_dimension: Option<u32>,
    /// Required when `auth = "oauth"`, forbidden otherwise.
    pub oauth: Option<OAuthConfig>,
}

/// A resolved `PROVIDER/MODEL` reference.
#[derive(Debug, Clone, Copy)]
pub struct ModelRef<'a> {
    pub provider_name: &'a str,
    pub provider: &'a ProviderConfig,
    pub model: &'a str,
}

impl std::fmt::Display for ModelRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.provider_name, self.model)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IngestionConfig {
    pub chunk_size_tokens: u32,
    pub chunk_overlap_tokens: u32,
    pub embedding_batch_size: u32,
    pub tokenizer_encoding: String,
    /// Largest upload the server accepts, in megabytes.
    pub upload_max_mb: u32,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            chunk_size_tokens: 512,
            chunk_overlap_tokens: 64,
            embedding_batch_size: 64,
            tokenizer_encoding: String::from("cl100k_base"),
            upload_max_mb: 512,
        }
    }
}

/// Ontology induction settings (design doc 6.5 and 13).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OntologyConfig {
    /// Share of a column's distinct values that must appear in another
    /// table's key column for a relation to be proposed.
    pub key_overlap_threshold: f64,
    /// A text column with at most this many distinct values is proposed as
    /// an enum property.
    pub enum_max_values: u32,
    /// Chunks sampled across documents for open extraction.
    pub propose_sample_chunks: u32,
    /// Distinct documents a document-evidence candidate needs to be shown
    /// in the main proposal.
    pub min_support_documents: u32,
}

impl Default for OntologyConfig {
    fn default() -> Self {
        Self {
            key_overlap_threshold: 0.8,
            enum_max_values: 12,
            propose_sample_chunks: 200,
            min_support_documents: 3,
        }
    }
}

impl OntologyConfig {
    /// The document-evidence tuning as the core module takes it.
    #[must_use]
    pub fn document_evidence(&self) -> DocumentEvidenceOptions {
        DocumentEvidenceOptions {
            sample_chunks: self.propose_sample_chunks,
            min_support_documents: self.min_support_documents,
            ..DocumentEvidenceOptions::default()
        }
    }

    /// The induction tuning as the core module takes it.
    #[must_use]
    pub fn table_evidence(&self) -> TableEvidenceOptions {
        TableEvidenceOptions {
            key_overlap_threshold: self.key_overlap_threshold,
            enum_max_values: self.enum_max_values,
            ..TableEvidenceOptions::default()
        }
    }
}

/// External data import (`quack import`, design doc 6.2).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImportConfig {
    /// Rows an import pulls at most; a larger file is cut to this.
    pub max_rows: u64,
    /// Bytes an HTTP(S) download may reach at most.
    pub max_download_mb: u64,
    /// How long a source may take to connect and answer.
    pub timeout_seconds: u64,
    /// Whether `quack serve` (with logins) may import `sqlite:` files
    /// from the server's disk. The CLI, the terminal, and `--local` always
    /// may: they run as the owner.
    pub allow_local_files: bool,
    /// Whether `quack serve` (with logins) may download from loopback,
    /// private, and link-local addresses, including cloud metadata
    /// endpoints. The CLI, the terminal, and `--local` always may.
    pub allow_private_hosts: bool,
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            max_rows: 1_000_000,
            max_download_mb: 512,
            timeout_seconds: 300,
            allow_local_files: false,
            allow_private_hosts: false,
        }
    }
}

/// Knowledge graph traversal and resolution (design doc 6.4 and 13).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GraphConfig {
    /// Hops a neighborhood or path query may take.
    pub max_traversal_depth: u32,
    /// Nodes a traversal or class listing returns at most.
    pub max_nodes: u32,
    /// Cosine distance under which two labels of one class are proposed
    /// as a merge for review.
    pub merge_threshold: f64,
    /// Cosine distance under which the merge happens without review.
    pub auto_merge_threshold: f64,
}

impl Default for GraphConfig {
    fn default() -> Self {
        let defaults = GraphOptions::default();
        Self {
            max_traversal_depth: defaults.max_traversal_depth,
            max_nodes: defaults.max_nodes,
            merge_threshold: defaults.merge_threshold,
            auto_merge_threshold: defaults.auto_merge_threshold,
        }
    }
}

impl GraphConfig {
    /// The tuning as the core module takes it.
    #[must_use]
    pub fn options(&self) -> GraphOptions {
        GraphOptions {
            max_traversal_depth: self.max_traversal_depth,
            max_nodes: self.max_nodes,
            merge_threshold: self.merge_threshold,
            auto_merge_threshold: self.auto_merge_threshold,
        }
    }
}

/// `quack serve` settings (design doc 12 and 13).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Listen address. Override: `QUACK_BIND`.
    pub bind: String,
    /// No authentication, one implicit user, loopback only.
    pub local: bool,
    /// Concurrent ingest workers per workspace.
    pub workers_per_workspace: u32,
    /// How long a browser session lives after login, however much it is
    /// used. Also the session cookie's `Max-Age`.
    pub session_max_age_hours: u32,
    /// How long a browser session survives with no request on it.
    pub session_idle_minutes: u32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: String::from("127.0.0.1:8080"),
            local: false,
            workers_per_workspace: 1,
            session_max_age_hours: 12,
            session_idle_minutes: 120,
        }
    }
}

impl ServerConfig {
    /// A session's absolute lifetime.
    #[must_use]
    pub fn session_max_age(&self) -> Duration {
        Duration::from_secs(u64::from(self.session_max_age_hours).saturating_mul(3600))
    }

    /// How long a session may sit unused before it is dropped.
    #[must_use]
    pub fn session_idle(&self) -> Duration {
        Duration::from_secs(u64::from(self.session_idle_minutes).saturating_mul(60))
    }
}

/// Workspace context (the owner-written instructions) settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ContextConfig {
    /// Approximate token budget for the context in the system prompt.
    pub max_tokens: u32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self { max_tokens: 4000 }
    }
}

/// Document retrieval settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RetrievalConfig {
    /// Default number of chunks returned by `search_documents`.
    pub top_k: u32,
    /// Reciprocal rank fusion constant for merging vector and keyword ranks.
    pub rrf_k: u32,
    /// Approximate token budget for pinned documents injected into the prompt.
    pub pinned_token_budget: u32,
    /// Also inject the top chunks for every user message via rig's
    /// `dynamic_context`, in addition to the `search_documents` tool.
    /// Off by default: retrieval should be a visible tool call the model
    /// chooses, not an invisible prefix on every turn.
    pub always_retrieve: bool,
    /// Reranking after hybrid fusion: `none` (the default) or `model`, the
    /// chat model ordering the candidates listwise.
    pub rerank: RerankMode,
    /// Candidates fetched for reranking before the top `k` are kept.
    pub rerank_candidates: u32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 8,
            rrf_k: 60,
            pinned_token_budget: 8000,
            always_retrieve: false,
            rerank: RerankMode::None,
            rerank_candidates: 24,
        }
    }
}

/// Which reranker, if any, orders retrieval candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RerankMode {
    None,
    Model,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnalysisConfig {
    pub max_query_rows: u32,
    pub query_timeout_seconds: u32,
    pub memory_limit_mb: u32,
    pub threads: u32,
    /// Maximum model round-trips (tool calls) per turn.
    pub max_turns: u32,
    /// Approximate token budget for prior messages replayed to the model.
    pub history_token_budget: u32,
    /// The largest context window quack asks Ollama for (`num_ctx`). Each
    /// turn requests what its prompt needs, rounded up, no more than this;
    /// Ollama's own default is 4,096 and it truncates silently past it.
    /// Other providers size their own window.
    pub max_context_tokens: u32,
    /// How long one extraction call (ontology document evidence, graph
    /// extraction) may run before the chunk is skipped.
    pub extraction_timeout_seconds: u64,
    /// Chunks extracted at once. Ollama serves one request at a time
    /// unless `OLLAMA_NUM_PARALLEL` is raised, so more only queues there.
    pub extraction_concurrency: u32,
    /// Reader connections held open per workspace handle, round-robined so
    /// concurrent reads run in parallel instead of queuing behind each
    /// other on one shared connection.
    pub reader_pool_size: u32,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            max_query_rows: 100,
            query_timeout_seconds: 30,
            memory_limit_mb: 256,
            threads: 4,
            max_turns: 10,
            history_token_budget: 32_000,
            max_context_tokens: 32_768,
            extraction_timeout_seconds: 120,
            extraction_concurrency: 1,
            reader_pool_size: 4,
        }
    }
}

const APP_NAME: &str = "quack";

/// Return the path where the config file is expected.
#[must_use]
pub fn config_file_path() -> PathBuf {
    let config_dir =
        std::env::var("QUACK_CONFIG_DIR").map_or_else(|_| default_config_dir(), PathBuf::from);
    config_dir.join("config.toml")
}

fn default_config_dir() -> PathBuf {
    dirs::home_dir().map_or_else(
        || PathBuf::from(".config").join(APP_NAME),
        |d| d.join(".config").join(APP_NAME),
    )
}

fn default_data_dir() -> PathBuf {
    dirs::home_dir().map_or_else(
        || PathBuf::from(".local/share").join(APP_NAME),
        |d| d.join(".local/share").join(APP_NAME),
    )
}

impl Config {
    /// Load configuration from the XDG config directory, falling back to defaults.
    ///
    /// Config file location: `~/.config/quack/config.toml`
    ///
    /// Data directory (databases, workspaces): `~/.local/share/quack/`
    ///
    /// `QUACK_DATA_DIR` overrides the data directory, `QUACK_CONFIG_DIR` the
    /// config directory, and `QUACK_MODEL` the chat model.
    ///
    /// # Errors
    ///
    /// Returns an error if the config file exists but cannot be read or
    /// parsed, contains unknown keys, or fails validation.
    pub fn load() -> Result<Self> {
        let config_path = config_file_path();

        let mut config = if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            Self::parse(&content)?
        } else {
            Self::default()
        };

        if let Ok(data_dir) = std::env::var("QUACK_DATA_DIR") {
            config.general.data_dir = PathBuf::from(data_dir);
        }
        if let Ok(model) = std::env::var("QUACK_MODEL") {
            config.general.chat_model = Some(model);
        }
        if let Ok(bind) = std::env::var("QUACK_BIND") {
            config.server.bind = bind;
        }

        config.validate()?;
        Ok(config)
    }

    /// Parse and validate TOML text.
    ///
    /// # Errors
    ///
    /// Returns an error on syntax errors, unknown keys, or invalid references.
    pub fn parse(toml_text: &str) -> Result<Self> {
        let config: Self = toml::from_str(toml_text)?;
        config.validate()?;
        Ok(config)
    }

    /// Check cross-field rules the parser cannot: model references name a
    /// configured provider, auth modes match the keys present, and every
    /// embedding provider declares its dimension.
    ///
    /// # Errors
    ///
    /// Returns a `Config` error describing the first violation found.
    pub fn validate(&self) -> Result<()> {
        for (name, provider) in &self.providers {
            if provider.auth != AuthMode::Oauth && provider.oauth.is_some() {
                return Err(Error::Config(format!(
                    "provider '{name}' has an [providers.{name}.oauth] section but auth is not \"oauth\""
                )));
            }
            match provider.auth {
                AuthMode::None => {
                    if provider.api_key_env.is_some() {
                        return Err(Error::Config(format!(
                            "provider '{name}' has auth = \"none\" but sets api_key_env; \
                             use auth = \"api-key\" or remove the key"
                        )));
                    }
                }
                AuthMode::ApiKey => {
                    if provider.api_key_env.is_none() {
                        return Err(Error::Config(format!(
                            "provider '{name}' has auth = \"api-key\" but no api_key_env"
                        )));
                    }
                }
                AuthMode::Oauth => {
                    if provider.api_key_env.is_some() {
                        return Err(Error::Config(format!(
                            "provider '{name}' has auth = \"oauth\" but sets api_key_env"
                        )));
                    }
                    let Some(oauth) = &provider.oauth else {
                        return Err(Error::Config(format!(
                            "provider '{name}' has auth = \"oauth\" but no [providers.{name}.oauth] section"
                        )));
                    };
                    if oauth.issuer_url.is_empty() || oauth.client_id.is_empty() {
                        return Err(Error::Config(format!(
                            "provider '{name}': [providers.{name}.oauth] needs issuer_url and client_id"
                        )));
                    }
                }
            }
        }
        if self.general.chat_model.is_some() {
            self.chat_model_ref()?;
        }
        if let Some(embed) = self.embedding_model_ref()? {
            if embed.provider.provider_type == ProviderType::Anthropic {
                return Err(Error::Config(format!(
                    "embedding_model '{embed}': anthropic does not serve embeddings"
                )));
            }
            if embed.provider.embedding_dimension.is_none() {
                return Err(Error::Config(format!(
                    "provider '{}' is used for embeddings but has no embedding_dimension",
                    embed.provider_name
                )));
            }
        }
        // Unlike the other [analysis] numbers, this one allocates OS-level
        // DuckDB connections at workspace open, one spawn_blocking round
        // trip and one writer-mutex acquisition each — an unreasonable
        // value blocks every request to that workspace until it finishes.
        if !(1..=32).contains(&self.analysis.reader_pool_size) {
            return Err(Error::Config(format!(
                "[analysis].reader_pool_size must be between 1 and 32, got {}",
                self.analysis.reader_pool_size
            )));
        }
        Ok(())
    }

    fn resolve_model<'a>(&'a self, setting: &str, spec: &'a str) -> Result<ModelRef<'a>> {
        let Some((provider_name, model)) = spec.split_once('/') else {
            return Err(Error::Config(format!(
                "{setting} = \"{spec}\" must be PROVIDER/MODEL, e.g. \"ollama/llama3.1:8b\""
            )));
        };
        if model.is_empty() {
            return Err(Error::Config(format!(
                "{setting} = \"{spec}\" is missing the model after the slash"
            )));
        }
        let provider = self.providers.get(provider_name).ok_or_else(|| {
            Error::Config(format!(
                "{setting} = \"{spec}\" names provider '{provider_name}', which is not configured; \
                 add a [providers.{provider_name}] section in {}",
                config_file_path().display()
            ))
        })?;
        Ok(ModelRef {
            provider_name,
            provider,
            model,
        })
    }

    /// The chat model, required.
    ///
    /// # Errors
    ///
    /// Returns an error if `chat_model` is unset or names an unknown provider.
    pub fn chat_model_ref(&self) -> Result<ModelRef<'_>> {
        let spec = self.general.chat_model.as_deref().ok_or_else(|| {
            Error::Config(format!(
                "no chat model configured — set [general].chat_model = \"PROVIDER/MODEL\" \
                 in {} or QUACK_MODEL",
                config_file_path().display()
            ))
        })?;
        self.resolve_model("chat_model", spec)
    }

    /// The embedding model, if configured.
    ///
    /// # Errors
    ///
    /// Returns an error if `embedding_model` names an unknown provider.
    pub fn embedding_model_ref(&self) -> Result<Option<ModelRef<'_>>> {
        self.general
            .embedding_model
            .as_deref()
            .map(|spec| self.resolve_model("embedding_model", spec))
            .transpose()
    }

    /// Directory holding the encrypted OAuth token caches, one per provider.
    #[must_use]
    pub fn tokens_dir(&self) -> PathBuf {
        self.general.data_dir.join("tokens")
    }

    #[must_use]
    pub fn control_db_path(&self) -> PathBuf {
        self.general.data_dir.join("control.db")
    }

    #[must_use]
    pub fn workspace_dir(&self, workspace_id: &str) -> PathBuf {
        self.general.data_dir.join("workspaces").join(workspace_id)
    }

    #[must_use]
    pub fn workspace_db_path(&self, workspace_id: &str) -> PathBuf {
        self.workspace_dir(workspace_id).join("data.duckdb")
    }

    #[must_use]
    pub fn workspace_files_dir(&self, workspace_id: &str) -> PathBuf {
        self.workspace_dir(workspace_id).join("files")
    }

    /// Ensure the data directory and its subdirectories exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the directories cannot be created.
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.general.data_dir)?;
        std::fs::create_dir_all(self.general.data_dir.join("workspaces"))?;
        Ok(())
    }

    #[must_use]
    pub fn data_dir(&self) -> &Path {
        &self.general.data_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: &str = r#"
[general]
chat_model = "ollama/llama3.1:8b"
embedding_model = "ollama/nomic-embed-text"

[providers.ollama]
type = "ollama"
base_url = "http://localhost:11434"
embedding_dimension = 768

[providers.anthropic]
type = "anthropic"
auth = "api-key"
api_key_env = "ANTHROPIC_API_KEY"

[retrieval]
top_k = 3
always_retrieve = true
rerank = "model"
"#;

    #[test]
    fn default_config_values() {
        let config = Config::default();
        assert_eq!(config.general.default_workspace, "default");
        assert!(config.general.chat_model.is_none());
        assert_eq!(config.ingestion.chunk_size_tokens, 512);
        assert_eq!(config.retrieval.top_k, 8);
        assert_eq!(config.retrieval.rrf_k, 60);
        assert_eq!(config.retrieval.pinned_token_budget, 8000);
        assert!(!config.retrieval.always_retrieve);
        assert_eq!(config.retrieval.rerank, RerankMode::None);
        assert_eq!(config.retrieval.rerank_candidates, 24);
        assert_eq!(config.context.max_tokens, 4000);
        assert_eq!(config.analysis.threads, 4);
        assert_eq!(config.analysis.max_turns, 10);
        assert_eq!(config.analysis.history_token_budget, 32_000);
        assert_eq!(config.ingestion.upload_max_mb, 512);
        assert_eq!(config.server.bind, "127.0.0.1:8080");
        assert!(!config.server.local);
        assert_eq!(config.server.workers_per_workspace, 1);
        assert!((config.ontology.key_overlap_threshold - 0.8).abs() < f64::EPSILON);
        assert_eq!(config.ontology.enum_max_values, 12);
        assert_eq!(config.ontology.propose_sample_chunks, 200);
        assert_eq!(config.ontology.min_support_documents, 3);
        assert!(err_of("[ontology]\nsample = 1\n").contains("sample"));
        assert_eq!(Config::default().graph.max_traversal_depth, 3);
        assert_eq!(Config::default().graph.max_nodes, 200);
        assert!(err_of("[graph]\nenabled = true\n").contains("enabled"));
        assert_eq!(Config::default().import.max_rows, 1_000_000);
        assert!(err_of("[import]\nmax = 1\n").contains("max"));
    }

    #[test]
    fn server_section_parses_and_rejects_unknown_keys() {
        let ok = Config::parse("[server]\nbind = \"0.0.0.0:9000\"\nlocal = true\n");
        assert!(ok.is_ok_and(|c| c.server.bind == "0.0.0.0:9000" && c.server.local));
        assert!(err_of("[server]\nport = 1\n").contains("port"));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn full_config_parses_and_resolves_models() {
        let config = Config::parse(FULL).unwrap();
        let chat = config.chat_model_ref().unwrap();
        assert_eq!(chat.provider_name, "ollama");
        assert_eq!(chat.model, "llama3.1:8b");
        assert_eq!(chat.provider.provider_type, ProviderType::Ollama);
        assert_eq!(chat.provider.auth, AuthMode::None);
        assert_eq!(chat.to_string(), "ollama/llama3.1:8b");
        let embed = config.embedding_model_ref().unwrap().unwrap();
        assert_eq!(embed.model, "nomic-embed-text");
        assert_eq!(embed.provider.embedding_dimension, Some(768));
        assert_eq!(config.retrieval.top_k, 3);
        assert_eq!(config.retrieval.rerank, RerankMode::Model);
        let anthropic = config.providers.get("anthropic");
        assert!(anthropic.is_some_and(|p| {
            p.provider_type == ProviderType::Anthropic && p.auth == AuthMode::ApiKey
        }));
    }

    fn err_of(toml_text: &str) -> String {
        match Config::parse(toml_text) {
            Ok(_) => String::from("<ok>"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn unknown_keys_are_rejected_at_every_level() {
        assert!(err_of("[genral]\nchat_model = \"a/b\"\n").contains("genral"));
        assert!(err_of("[general]\nchat_modle = \"a/b\"\n").contains("chat_modle"));
        assert!(err_of("[providers.o]\ntype = \"ollama\"\nmodel = \"x\"\n").contains("model"));
        assert!(err_of("[retrieval]\ntopk = 1\n").contains("topk"));
        assert!(err_of("[retrieval]\nrerank = \"bge\"\n").contains("bge"));
        assert!(err_of("[analysis]\nthread = 1\n").contains("thread"));
    }

    #[test]
    fn unknown_provider_type_is_rejected() {
        assert!(err_of("[providers.o]\ntype = \"bedrock\"\n").contains("bedrock"));
    }

    #[test]
    fn openai_compat_alias_maps_to_openai() {
        let config = Config::parse(
            "[providers.g]\ntype = \"openai-compat\"\nauth = \"api-key\"\napi_key_env = \"K\"\n",
        );
        assert!(config.is_ok_and(|c| {
            c.providers
                .get("g")
                .is_some_and(|p| p.provider_type == ProviderType::Openai)
        }));
    }

    #[test]
    fn model_reference_must_name_a_configured_provider() {
        let msg = err_of("[general]\nchat_model = \"missing/m\"\n");
        assert!(
            msg.contains("'missing'") && msg.contains("not configured"),
            "{msg}"
        );
    }

    #[test]
    fn model_reference_must_have_a_slash_and_a_model() {
        assert!(
            err_of("[general]\nchat_model = \"ollama\"\n[providers.ollama]\ntype = \"ollama\"\n")
                .contains("PROVIDER/MODEL")
        );
        assert!(
            err_of("[general]\nchat_model = \"ollama/\"\n[providers.ollama]\ntype = \"ollama\"\n")
                .contains("missing the model")
        );
    }

    #[test]
    fn auth_mode_must_agree_with_api_key_env() {
        assert!(
            err_of("[providers.o]\ntype = \"openai\"\napi_key_env = \"K\"\n")
                .contains("auth = \"none\"")
        );
        assert!(
            err_of("[providers.o]\ntype = \"openai\"\nauth = \"api-key\"\n")
                .contains("no api_key_env")
        );
        assert!(
            err_of("[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\n")
                .contains("no [providers.o.oauth] section")
        );
    }

    #[test]
    fn oauth_section_is_required_by_and_exclusive_to_oauth_mode() {
        let stray = "[providers.o]\ntype = \"openai\"\n[providers.o.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n";
        assert!(err_of(stray).contains("auth is not \"oauth\""));
        let with_key = "[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\napi_key_env = \"K\"\n[providers.o.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n";
        assert!(err_of(with_key).contains("sets api_key_env"));
        let empty = "[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.o.oauth]\nissuer_url = \"\"\nclient_id = \"c\"\n";
        assert!(err_of(empty).contains("needs issuer_url and client_id"));
        assert!(err_of("[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.o.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\ntenant = \"x\"\n").contains("tenant"));
    }

    #[test]
    fn oauth_section_defaults_and_fields_parse() {
        let config = Config::parse(
            "[providers.azure]\ntype = \"openai\"\nauth = \"oauth\"\nbase_url = \"https://r.openai.azure.com/openai/deployments/d\"\n[providers.azure.oauth]\nissuer_url = \"https://login.microsoftonline.com/t/v2.0\"\nclient_id = \"abc\"\nscopes = [\"https://cognitiveservices.azure.com/.default\", \"offline_access\"]\nclient_secret_env = \"AZURE_CLIENT_SECRET\"\n",
        );
        let Ok(config) = config else {
            return assert!(config.is_ok(), "{config:?}");
        };
        let oauth = config.providers.get("azure").and_then(|p| p.oauth.as_ref());
        assert!(oauth.is_some_and(|o| {
            o.redirect_uri == "http://127.0.0.1:19876/callback"
                && !o.device_code
                && o.scopes.len() == 2
                && o.client_secret_env.as_deref() == Some("AZURE_CLIENT_SECRET")
        }));
        assert_eq!(config.tokens_dir(), config.general.data_dir.join("tokens"));
    }

    #[test]
    fn embedding_provider_needs_dimension_and_cannot_be_anthropic() {
        let no_dim = "[general]\nembedding_model = \"o/e\"\n[providers.o]\ntype = \"ollama\"\n";
        assert!(err_of(no_dim).contains("embedding_dimension"));
        let anthropic = "[general]\nembedding_model = \"a/e\"\n[providers.a]\ntype = \"anthropic\"\nauth = \"api-key\"\napi_key_env = \"K\"\nembedding_dimension = 1\n";
        assert!(err_of(anthropic).contains("does not serve embeddings"));
    }

    #[test]
    fn chat_model_unset_is_a_clear_error_when_asked_for() {
        let config = Config::default();
        assert!(config.validate().is_ok());
        let err = config.chat_model_ref().err();
        assert!(err.is_some_and(|e| e.to_string().contains("chat_model")));
        assert!(config.embedding_model_ref().is_ok_and(|m| m.is_none()));
    }

    #[test]
    fn path_helpers_use_data_dir() {
        let mut config = Config::default();
        config.general.data_dir = PathBuf::from("/data");

        assert_eq!(config.control_db_path(), PathBuf::from("/data/control.db"));
        assert_eq!(
            config.workspace_dir("ws1"),
            PathBuf::from("/data/workspaces/ws1")
        );
        assert_eq!(
            config.workspace_db_path("ws1"),
            PathBuf::from("/data/workspaces/ws1/data.duckdb")
        );
        assert_eq!(
            config.workspace_files_dir("ws1"),
            PathBuf::from("/data/workspaces/ws1/files")
        );
    }

    #[test]
    fn default_dirs_use_xdg_layout() {
        assert!(default_data_dir().to_string_lossy().contains(APP_NAME));
        assert!(default_config_dir().to_string_lossy().contains(APP_NAME));
    }
}
