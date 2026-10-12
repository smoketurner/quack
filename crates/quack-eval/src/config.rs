//! The models the run uses and the `config.toml` the binary reads them
//! from. Offline there are no models: ingestion stores chunks without
//! vectors and search is keyword-only, which is what a person without a
//! model gets too.

use std::path::Path;

use anyhow::Result;
use serde::Serialize;

/// The Ollama models, from `QUACK_EVAL_*`, each defaulting to what a
/// developer machine runs (the nightly workflow names smaller ones).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Models {
    pub(crate) url: String,
    pub(crate) chat: String,
    pub(crate) embedding: String,
    pub(crate) dimension: u32,
    /// `None` skips the vision section (`QUACK_EVAL_VISION_MODEL=none`).
    pub(crate) vision: Option<String>,
    pub(crate) turn_seconds: u64,
    pub(crate) repeat: u32,
}

impl Models {
    pub(crate) fn from_env() -> Self {
        let var = |name: &str, default: &str| {
            std::env::var(name)
                .ok()
                .map(|v| v.trim().to_owned())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| default.to_owned())
        };
        let vision = var("QUACK_EVAL_VISION_MODEL", "gemma4:e4b");
        Self {
            url: var("QUACK_EVAL_OLLAMA_URL", "http://127.0.0.1:11434"),
            chat: var("QUACK_EVAL_CHAT_MODEL", "gpt-oss:20b"),
            embedding: var("QUACK_EVAL_EMBEDDING_MODEL", "embeddinggemma-2"),
            dimension: var("QUACK_EVAL_EMBEDDING_DIMENSION", "768")
                .parse()
                .unwrap_or(768),
            vision: (!vision.eq_ignore_ascii_case("none")).then_some(vision),
            turn_seconds: var("QUACK_EVAL_TURN_SECONDS", "300").parse().unwrap_or(300),
            repeat: var("QUACK_EVAL_REPEAT", "1").parse().unwrap_or(1).max(1),
        }
    }

    /// The `config.toml` naming these models on one Ollama provider.
    pub(crate) fn config_toml(&self) -> String {
        let vision = self.vision.as_ref().map_or_else(String::new, |model| {
            format!("\n[ingestion]\nvision_model = \"ollama/{model}\"\n")
        });
        format!(
            "[general]\ndefault_workspace = \"eval\"\nchat_model = \"ollama/{chat}\"\n\n\
             [providers.ollama]\ntype = \"ollama\"\nbase_url = \"{url}\"\n\n\
             [embedding]\nmodel = \"ollama/{embedding}\"\ndimension = {dimension}\n{vision}",
            chat = self.chat,
            url = self.url,
            embedding = self.embedding,
            dimension = self.dimension,
        )
    }
}

/// Write the run's `config.toml` into `data_dir`: the models under
/// `--ollama`, nothing but the default workspace otherwise.
pub(crate) fn write(data_dir: &Path, models: Option<&Models>) -> Result<()> {
    let toml = models.map_or_else(
        || String::from("[general]\ndefault_workspace = \"eval\"\n"),
        Models::config_toml,
    );
    std::fs::write(data_dir.join("config.toml"), toml)?;
    Ok(())
}
