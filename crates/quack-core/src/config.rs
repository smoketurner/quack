use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

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
    pub analysis: AnalysisConfig,
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
    /// OAuth 2.0 PKCE against an identity provider (design doc 10.2). Parsed
    /// so configs can carry it, but not implemented yet.
    Oauth,
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
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            chunk_size_tokens: 512,
            chunk_overlap_tokens: 64,
            embedding_batch_size: 64,
            tokenizer_encoding: String::from("cl100k_base"),
        }
    }
}

/// Document retrieval settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RetrievalConfig {
    /// Default number of chunks returned by `search_documents`.
    pub top_k: u32,
    /// Also inject the top chunks for every user message via rig's
    /// `dynamic_context`, in addition to the `search_documents` tool.
    /// Off by default: retrieval should be a visible tool call the model
    /// chooses, not an invisible prefix on every turn.
    pub always_retrieve: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 8,
            always_retrieve: false,
        }
    }
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
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            max_query_rows: 100,
            query_timeout_seconds: 30,
            memory_limit_mb: 256,
            threads: 4,
            max_turns: 10,
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
                    return Err(Error::Config(format!(
                        "provider '{name}': auth = \"oauth\" is not implemented yet"
                    )));
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
"#;

    #[test]
    fn default_config_values() {
        let config = Config::default();
        assert_eq!(config.general.default_workspace, "default");
        assert!(config.general.chat_model.is_none());
        assert_eq!(config.ingestion.chunk_size_tokens, 512);
        assert_eq!(config.retrieval.top_k, 8);
        assert!(!config.retrieval.always_retrieve);
        assert_eq!(config.analysis.threads, 4);
        assert_eq!(config.analysis.max_turns, 10);
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
                .contains("not implemented")
        );
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
