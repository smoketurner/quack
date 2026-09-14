//! LLM provider construction on top of rig.
//!
//! Everything that turns a `[providers.<name>]` config entry into a rig
//! client lives here, so the interfaces never build providers themselves.

use rig::client::EmbeddingsClient;
use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};
use rig::prelude::*;

use crate::analysis::agent::{self, AgentResponse};
use crate::analysis::policy::WritePolicy;
use crate::config::{Config, ProviderConfig, config_file_path};
use crate::error::{Error, Result};
use crate::storage::workspace::WorkspaceDb;

/// Embedding model over every provider that supports embeddings.
#[derive(Clone)]
pub enum EmbedModel {
    Ollama(rig::providers::ollama::EmbeddingModel),
    OpenAi(
        rig::providers::openai::GenericEmbeddingModel<rig::providers::openai::OpenAICompletionsExt>,
    ),
}

impl EmbeddingModel for EmbedModel {
    const MAX_DOCUMENTS: usize = 1024;
    type Client = rig::providers::ollama::Client;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self::Ollama(rig::providers::ollama::EmbeddingModel::make(
            client, model, dims,
        ))
    }

    fn ndims(&self) -> usize {
        match self {
            Self::Ollama(m) => m.ndims(),
            Self::OpenAi(m) => m.ndims(),
        }
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> std::result::Result<Vec<Embedding>, EmbeddingError> {
        match self {
            Self::Ollama(m) => m.embed_texts(texts).await,
            Self::OpenAi(m) => m.embed_texts(texts).await,
        }
    }
}

fn resolve_api_key(provider_config: &ProviderConfig) -> Option<String> {
    provider_config
        .api_key_env
        .as_ref()
        .and_then(|env| std::env::var(env).ok())
}

fn build_ollama_client(provider_config: &ProviderConfig) -> Result<rig::providers::ollama::Client> {
    let api_key = resolve_api_key(provider_config)
        .map(rig::providers::ollama::OllamaApiKey::from)
        .unwrap_or_default();

    let mut builder = rig::providers::ollama::Client::builder().api_key(api_key);

    if let Some(base_url) = &provider_config.base_url {
        let url = base_url.trim_end_matches("/v1");
        builder = builder.base_url(url);
    }

    builder
        .build()
        .map_err(|e| Error::Llm(format!("failed to build Ollama client: {e}")))
}

fn build_openai_client(
    provider_config: &ProviderConfig,
) -> Result<rig::providers::openai::CompletionsClient> {
    let api_key = resolve_api_key(provider_config).ok_or_else(|| {
        Error::Config(String::from(
            "OpenAI provider requires api_key_env to be set",
        ))
    })?;

    let mut builder = rig::providers::openai::CompletionsClient::builder().api_key(&api_key);

    if let Some(base_url) = &provider_config.base_url {
        builder = builder.base_url(base_url);
    }

    builder
        .build()
        .map_err(|e| Error::Llm(format!("failed to build OpenAI client: {e}")))
}

fn build_anthropic_client(
    provider_config: &ProviderConfig,
) -> Result<rig::providers::anthropic::Client> {
    let api_key = resolve_api_key(provider_config).ok_or_else(|| {
        Error::Config(String::from(
            "Anthropic provider requires api_key_env to be set",
        ))
    })?;

    let mut builder = rig::providers::anthropic::Client::builder().api_key(&api_key);

    if let Some(base_url) = &provider_config.base_url {
        builder = builder.base_url(base_url);
    }

    builder
        .build()
        .map_err(|e| Error::Llm(format!("failed to build Anthropic client: {e}")))
}

fn build_embed_model_from(embed_config: &ProviderConfig) -> Result<EmbedModel> {
    let model_name = embed_config.embedding_model.as_deref().ok_or_else(|| {
        Error::Config(String::from(
            "embedding provider has no embedding_model configured",
        ))
    })?;

    let ndims = embed_config.embedding_dimension.ok_or_else(|| {
        Error::Config(String::from(
            "embedding provider has no embedding_dimension configured",
        ))
    })?;
    let ndims = usize::try_from(ndims)
        .map_err(|e| Error::Config(format!("embedding_dimension overflow: {e}")))?;

    match embed_config.provider_type.as_str() {
        "ollama" => {
            let client = build_ollama_client(embed_config)?;
            Ok(EmbedModel::Ollama(
                client.embedding_model_with_ndims(model_name, ndims),
            ))
        }
        "openai" => {
            let client = build_openai_client(embed_config)?;
            Ok(EmbedModel::OpenAi(
                client.embedding_model_with_ndims(model_name, ndims),
            ))
        }
        other => Err(Error::Config(format!(
            "provider type '{other}' does not support embeddings — use 'ollama' or 'openai'"
        ))),
    }
}

/// The configured embedding model, or `None` when no provider declares one.
///
/// # Errors
///
/// Returns an error if a provider is configured but incomplete.
pub fn optional_embedding_model(config: &Config) -> Result<Option<EmbedModel>> {
    let Some((name, embed_config)) = config.find_embedding_provider() else {
        tracing::info!("no embedding provider configured");
        return Ok(None);
    };
    tracing::info!(provider = %name, "using embedding provider");
    build_embed_model_from(embed_config).map(Some)
}

/// The configured embedding model, required.
///
/// # Errors
///
/// Returns an error if no provider declares an embedding model or the
/// provider is incomplete.
pub fn required_embedding_model(config: &Config) -> Result<EmbedModel> {
    let (name, embed_config) = config.find_embedding_provider().ok_or_else(|| {
        Error::Config(format!(
            "no embedding provider configured — add a [providers.<name>] section \
             with 'embedding_model' set in {}",
            config_file_path().display()
        ))
    })?;
    tracing::info!(provider = %name, "using embedding provider");
    build_embed_model_from(embed_config)
}

/// `provider_type/model` for status lines, or a placeholder.
#[must_use]
pub fn chat_model_display(config: &Config) -> String {
    config.find_chat_provider().map_or_else(
        || String::from("no provider"),
        |(_, c)| {
            let model = c.model.as_deref().unwrap_or("unknown");
            format!("{}/{model}", c.provider_type)
        },
    )
}

/// Run one agent turn with the configured chat and embedding providers.
///
/// This is the single dispatch point over provider types; interfaces call it
/// rather than matching on `provider_type` themselves.
///
/// # Errors
///
/// Returns an error if no chat provider is configured, a provider cannot be
/// built, or the agent turn fails.
pub async fn run_turn(
    config: &Config,
    db: WorkspaceDb,
    policy: WritePolicy,
    message: &str,
) -> Result<AgentResponse> {
    let (_, chat_config) = config.find_chat_provider().ok_or_else(|| {
        Error::Config(format!(
            "no LLM provider configured with a chat model — \
             add a [providers.<name>] section with 'model' set in {}",
            config_file_path().display()
        ))
    })?;
    let chat_model_name = chat_config
        .model
        .as_deref()
        .ok_or_else(|| Error::Config(String::from("chat provider has no model configured")))?;

    let embedding_model = required_embedding_model(config)?;

    tracing::info!(
        chat_model = %chat_model_name,
        provider_type = %chat_config.provider_type,
        "starting agent turn"
    );

    match chat_config.provider_type.as_str() {
        "ollama" => {
            let client = build_ollama_client(chat_config)?;
            agent::run_analysis(
                db,
                client.completion_model(chat_model_name),
                embedding_model,
                &config.analysis,
                &config.retrieval,
                policy,
                message,
            )
            .await
        }
        "openai" => {
            let client = build_openai_client(chat_config)?;
            agent::run_analysis(
                db,
                client.completion_model(chat_model_name),
                embedding_model,
                &config.analysis,
                &config.retrieval,
                policy,
                message,
            )
            .await
        }
        "anthropic" => {
            let client = build_anthropic_client(chat_config)?;
            agent::run_analysis(
                db,
                client.completion_model(chat_model_name),
                embedding_model,
                &config.analysis,
                &config.retrieval,
                policy,
                message,
            )
            .await
        }
        other => Err(Error::Config(format!(
            "unsupported provider type '{other}' — expected 'ollama', 'openai', or 'anthropic'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn provider(kind: &str, model: Option<&str>, embed: Option<&str>) -> ProviderConfig {
        ProviderConfig {
            provider_type: kind.to_owned(),
            base_url: None,
            api_key_env: None,
            model: model.map(str::to_owned),
            embedding_model: embed.map(str::to_owned),
            embedding_dimension: embed.map(|_| 4),
        }
    }

    fn config_with(providers: Vec<(&str, ProviderConfig)>) -> Config {
        Config {
            providers: providers
                .into_iter()
                .map(|(n, p)| (n.to_owned(), p))
                .collect::<BTreeMap<_, _>>(),
            ..Config::default()
        }
    }

    #[test]
    fn display_name_reports_provider_and_model() {
        let config = config_with(vec![("o", provider("ollama", Some("llama3"), None))]);
        assert_eq!(chat_model_display(&config), "ollama/llama3");
        assert_eq!(chat_model_display(&Config::default()), "no provider");
    }

    #[test]
    fn optional_embedding_model_is_none_without_provider() {
        assert!(matches!(
            optional_embedding_model(&Config::default()),
            Ok(None)
        ));
    }

    #[test]
    fn required_embedding_model_errors_without_provider() {
        let err = required_embedding_model(&Config::default()).err();
        assert!(err.is_some_and(|e| e.to_string().contains("no embedding provider")));
    }

    #[test]
    fn anthropic_cannot_embed() {
        let config = config_with(vec![("a", provider("anthropic", None, Some("x")))]);
        let err = required_embedding_model(&config).err();
        assert!(err.is_some_and(|e| e.to_string().contains("does not support embeddings")));
    }

    #[test]
    fn openai_requires_an_api_key_env() {
        let config = config_with(vec![("o", provider("openai", None, Some("x")))]);
        let err = required_embedding_model(&config).err();
        assert!(err.is_some_and(|e| e.to_string().contains("api_key_env")));
    }

    #[test]
    fn ollama_embedding_model_builds_without_a_key() {
        let config = config_with(vec![("o", provider("ollama", None, Some("nomic")))]);
        let model = required_embedding_model(&config);
        assert!(model.is_ok_and(|m| m.ndims() == 4));
    }
}
