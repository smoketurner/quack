use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub general: GeneralConfig,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderConfig>,
    pub ingestion: IngestionConfig,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct GeneralConfig {
    pub data_dir: PathBuf,
    pub default_workspace: String,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            default_workspace: String::from("default"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    #[serde(rename = "type")]
    pub provider_type: String,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub model: Option<String>,
    pub embedding_model: Option<String>,
    pub embedding_dimension: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
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

fn default_data_dir() -> PathBuf {
    dirs::home_dir().map_or_else(|| PathBuf::from(".quack"), |h| h.join(".quack"))
}

impl Config {
    /// Load configuration from `~/.quack/config.toml`, falling back to defaults.
    /// Environment variables override file values.
    ///
    /// # Errors
    ///
    /// Returns an error if the config file exists but cannot be read or parsed.
    pub fn load() -> crate::error::Result<Self> {
        let default_dir = default_data_dir();
        let config_path = default_dir.join("config.toml");

        let mut config = if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            toml::from_str(&content)?
        } else {
            Self::default()
        };

        if let Ok(data_dir) = std::env::var("QUACK_DATA_DIR") {
            config.general.data_dir = PathBuf::from(data_dir);
        }

        Ok(config)
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

    /// Find the first configured OpenAI-compatible provider.
    #[must_use]
    pub fn find_embedding_provider(&self) -> Option<(&str, &ProviderConfig)> {
        self.providers
            .iter()
            .find(|(_, p)| p.provider_type == "openai-compat" && p.embedding_model.is_some())
            .map(|(name, config)| (name.as_str(), config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = Config::default();
        assert_eq!(config.general.default_workspace, "default");
        assert_eq!(config.ingestion.chunk_size_tokens, 512);
        assert_eq!(config.ingestion.chunk_overlap_tokens, 64);
        assert_eq!(config.ingestion.embedding_batch_size, 64);
        assert_eq!(config.ingestion.tokenizer_encoding, "cl100k_base");
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
    fn data_dir_accessor() {
        let mut config = Config::default();
        config.general.data_dir = PathBuf::from("/custom");
        assert_eq!(config.data_dir(), Path::new("/custom"));
    }

    #[test]
    fn find_embedding_provider_returns_none_when_empty() {
        let config = Config::default();
        assert!(config.find_embedding_provider().is_none());
    }

    #[test]
    fn find_embedding_provider_skips_wrong_type() {
        let mut config = Config::default();
        config.providers.insert(
            "anthropic".into(),
            ProviderConfig {
                provider_type: "anthropic".into(),
                base_url: Some("https://api.anthropic.com".into()),
                api_key_env: Some("ANTHROPIC_API_KEY".into()),
                model: Some("claude-3".into()),
                embedding_model: Some("embed-model".into()),
                embedding_dimension: Some(1024),
            },
        );
        assert!(config.find_embedding_provider().is_none());
    }

    #[test]
    fn find_embedding_provider_skips_without_embedding_model() {
        let mut config = Config::default();
        config.providers.insert(
            "openai".into(),
            ProviderConfig {
                provider_type: "openai-compat".into(),
                base_url: Some("http://localhost".into()),
                api_key_env: None,
                model: Some("gpt-4".into()),
                embedding_model: None,
                embedding_dimension: None,
            },
        );
        assert!(config.find_embedding_provider().is_none());
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Some")]
    fn find_embedding_provider_returns_valid() {
        let mut config = Config::default();
        config.providers.insert(
            "ollama".into(),
            ProviderConfig {
                provider_type: "openai-compat".into(),
                base_url: Some("http://localhost:11434/v1".into()),
                api_key_env: None,
                model: None,
                embedding_model: Some("nomic-embed-text".into()),
                embedding_dimension: Some(768),
            },
        );
        let (name, provider) = config.find_embedding_provider().unwrap();
        assert_eq!(name, "ollama");
        assert_eq!(provider.embedding_dimension, Some(768));
        assert_eq!(
            provider.embedding_model.as_deref(),
            Some("nomic-embed-text")
        );
    }
}
