use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub general: GeneralConfig,
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
}
