use anyhow::{Context, Result};
use quack_core::config::{Config, ProviderConfig};
use rig::client::EmbeddingsClient;
use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};

#[derive(Clone)]
pub(crate) enum EmbedModel {
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
    ) -> Result<Vec<Embedding>, EmbeddingError> {
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

pub(crate) fn build_ollama_client(
    provider_config: &ProviderConfig,
) -> Result<rig::providers::ollama::Client> {
    let api_key = resolve_api_key(provider_config)
        .map(rig::providers::ollama::OllamaApiKey::from)
        .unwrap_or_default();

    let mut builder = rig::providers::ollama::Client::builder().api_key(api_key);

    if let Some(base_url) = &provider_config.base_url {
        let url = base_url.trim_end_matches("/v1");
        builder = builder.base_url(url);
    }

    builder.build().context("failed to build Ollama client")
}

pub(crate) fn build_openai_client(
    provider_config: &ProviderConfig,
) -> Result<rig::providers::openai::CompletionsClient> {
    let api_key = resolve_api_key(provider_config)
        .ok_or_else(|| anyhow::anyhow!("OpenAI provider requires api_key_env to be set"))?;

    let mut builder = rig::providers::openai::CompletionsClient::builder().api_key(&api_key);

    if let Some(base_url) = &provider_config.base_url {
        builder = builder.base_url(base_url);
    }

    builder.build().context("failed to build OpenAI client")
}

pub(crate) fn build_anthropic_client(
    provider_config: &ProviderConfig,
) -> Result<rig::providers::anthropic::Client> {
    let api_key = resolve_api_key(provider_config)
        .ok_or_else(|| anyhow::anyhow!("Anthropic provider requires api_key_env to be set"))?;

    let mut builder = rig::providers::anthropic::Client::builder().api_key(&api_key);

    if let Some(base_url) = &provider_config.base_url {
        builder = builder.base_url(base_url);
    }

    builder.build().context("failed to build Anthropic client")
}

pub(crate) fn build_rig_embedding_model(config: &Config) -> Result<(EmbedModel, String)> {
    let (name, embed_config) = config
        .find_embedding_provider()
        .ok_or_else(|| anyhow::anyhow!("no embedding provider configured"))?;

    tracing::info!(provider = %name, "using embedding provider");

    let model_name = embed_config
        .embedding_model
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("embedding provider has no embedding_model configured"))?;

    let ndims = embed_config.embedding_dimension.ok_or_else(|| {
        anyhow::anyhow!("embedding provider has no embedding_dimension configured")
    })?;

    let ndims_usize = usize::try_from(ndims).context("embedding_dimension overflow")?;

    let model = match embed_config.provider_type.as_str() {
        "ollama" => {
            let client = build_ollama_client(embed_config)?;
            EmbedModel::Ollama(client.embedding_model_with_ndims(model_name, ndims_usize))
        }
        "openai" => {
            let client = build_openai_client(embed_config)?;
            EmbedModel::OpenAi(client.embedding_model_with_ndims(model_name, ndims_usize))
        }
        other => anyhow::bail!(
            "provider type '{other}' does not support embeddings — use 'ollama' or 'openai'"
        ),
    };

    Ok((model, model_name.to_owned()))
}

pub(crate) fn build_embedding_model(config: &Config) -> Result<Option<EmbedModel>> {
    let Some((name, embed_config)) = config.find_embedding_provider() else {
        tracing::info!("no embedding provider configured — storing chunks without embeddings");
        return Ok(None);
    };

    tracing::info!(provider = %name, "using embedding provider for ingestion");

    let model_name = embed_config
        .embedding_model
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("embedding provider has no embedding_model configured"))?;

    let ndims = embed_config.embedding_dimension.ok_or_else(|| {
        anyhow::anyhow!("embedding provider has no embedding_dimension configured")
    })?;

    let ndims_usize = usize::try_from(ndims).context("embedding_dimension overflow")?;

    let model = match embed_config.provider_type.as_str() {
        "ollama" => {
            let client = build_ollama_client(embed_config)?;
            EmbedModel::Ollama(client.embedding_model_with_ndims(model_name, ndims_usize))
        }
        "openai" => {
            let client = build_openai_client(embed_config)?;
            EmbedModel::OpenAi(client.embedding_model_with_ndims(model_name, ndims_usize))
        }
        other => anyhow::bail!(
            "provider type '{other}' does not support embeddings — use 'ollama' or 'openai'"
        ),
    };

    Ok(Some(model))
}

#[must_use]
pub(crate) fn provider_display_name(config: &Config) -> String {
    config.find_chat_provider().map_or_else(
        || String::from("no provider"),
        |(_, c)| {
            let model = c.model.as_deref().unwrap_or("unknown");
            format!("{}/{model}", c.provider_type)
        },
    )
}
