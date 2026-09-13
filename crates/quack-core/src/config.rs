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
    pub analysis: AnalysisConfig,
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

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct AnalysisConfig {
    pub max_query_rows: u32,
    pub query_timeout_seconds: u32,
    pub memory_limit_mb: u32,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            max_query_rows: 100,
            query_timeout_seconds: 30,
            memory_limit_mb: 256,
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
    /// `QUACK_DATA_DIR` overrides the data directory.
    /// `QUACK_CONFIG_DIR` overrides the config directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the config file exists but cannot be read or parsed.
    pub fn load() -> crate::error::Result<Self> {
        let config_dir =
            std::env::var("QUACK_CONFIG_DIR").map_or_else(|_| default_config_dir(), PathBuf::from);

        let config_path = config_dir.join("config.toml");

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

    /// Find the first configured provider with an embedding model.
    #[must_use]
    pub fn find_embedding_provider(&self) -> Option<(&str, &ProviderConfig)> {
        self.providers
            .iter()
            .find(|(_, p)| p.embedding_model.is_some())
            .map(|(name, config)| (name.as_str(), config))
    }

    /// Find the first configured provider with a chat model.
    #[must_use]
    pub fn find_chat_provider(&self) -> Option<(&str, &ProviderConfig)> {
        self.providers
            .iter()
            .find(|(_, p)| p.model.is_some())
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
    fn default_data_dir_uses_xdg() {
        let data_dir = default_data_dir();
        let data_str = data_dir.to_string_lossy();
        assert!(
            data_str.contains(APP_NAME),
            "data dir should contain app name: {data_str}"
        );
        assert!(
            !data_str.contains(".quack"),
            "data dir should not use legacy .quack path: {data_str}"
        );
    }

    #[test]
    fn default_config_dir_uses_xdg() {
        let config_dir = default_config_dir();
        let config_str = config_dir.to_string_lossy();
        assert!(
            config_str.contains(APP_NAME),
            "config dir should contain app name: {config_str}"
        );
        assert!(
            !config_str.contains(".quack"),
            "config dir should not use legacy .quack path: {config_str}"
        );
    }

    #[test]
    fn find_embedding_provider_returns_none_when_empty() {
        let config = Config::default();
        assert!(config.find_embedding_provider().is_none());
    }

    #[test]
    fn find_embedding_provider_skips_without_embedding_model() {
        let mut config = Config::default();
        config.providers.insert(
            "ollama".into(),
            ProviderConfig {
                provider_type: "ollama".into(),
                base_url: Some("http://localhost:11434".into()),
                api_key_env: None,
                model: Some("llama3.2".into()),
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
                provider_type: "ollama".into(),
                base_url: Some("http://localhost:11434".into()),
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
