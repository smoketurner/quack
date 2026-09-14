//! LLM provider construction on top of rig.
//!
//! Everything that turns a `[providers.<name>]` config entry plus a
//! `PROVIDER/MODEL` reference into a rig client lives here, so the interfaces
//! never build providers themselves.

use rig::client::EmbeddingsClient;
use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};
use std::sync::Arc;

use rig::prelude::*;

use crate::analysis::agent::{self, AgentResponse};
use crate::analysis::events::EventSink;
use crate::analysis::policy::WritePolicy;
use crate::analysis::text_to_sql::PromptOptions;
use crate::analysis::tools::SharedDb;
use crate::config::{AuthMode, Config, ModelRef, ProviderConfig, ProviderType};
use crate::error::{Error, Result};
use crate::storage::{context, sessions};

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

/// The API key for a provider, according to its `auth` mode.
fn api_key(name: &str, provider: &ProviderConfig) -> Result<Option<String>> {
    match provider.auth {
        AuthMode::None => Ok(None),
        AuthMode::ApiKey => {
            let var = provider.api_key_env.as_deref().ok_or_else(|| {
                Error::Config(format!(
                    "provider '{name}' has auth = \"api-key\" but no api_key_env"
                ))
            })?;
            let key = std::env::var(var).map_err(|_| {
                Error::Config(format!(
                    "provider '{name}' needs the API key in environment variable {var}, which is not set"
                ))
            })?;
            Ok(Some(key))
        }
        AuthMode::Oauth => Err(Error::Config(format!(
            "provider '{name}': auth = \"oauth\" is not implemented yet"
        ))),
    }
}

fn build_ollama_client(
    name: &str,
    provider: &ProviderConfig,
) -> Result<rig::providers::ollama::Client> {
    let key = api_key(name, provider)?
        .map(rig::providers::ollama::OllamaApiKey::from)
        .unwrap_or_default();

    let mut builder = rig::providers::ollama::Client::builder().api_key(key);

    if let Some(base_url) = &provider.base_url {
        let url = base_url.trim_end_matches("/v1");
        builder = builder.base_url(url);
    }

    builder
        .build()
        .map_err(|e| Error::Llm(format!("failed to build Ollama client for '{name}': {e}")))
}

fn build_openai_client(
    name: &str,
    provider: &ProviderConfig,
) -> Result<rig::providers::openai::CompletionsClient> {
    let key = api_key(name, provider)?.ok_or_else(|| {
        Error::Config(format!(
            "provider '{name}' (openai) requires auth = \"api-key\" and api_key_env"
        ))
    })?;

    let mut builder = rig::providers::openai::CompletionsClient::builder().api_key(&key);

    if let Some(base_url) = &provider.base_url {
        builder = builder.base_url(base_url);
    }

    builder
        .build()
        .map_err(|e| Error::Llm(format!("failed to build OpenAI client for '{name}': {e}")))
}

fn build_anthropic_client(
    name: &str,
    provider: &ProviderConfig,
) -> Result<rig::providers::anthropic::Client> {
    let key = api_key(name, provider)?.ok_or_else(|| {
        Error::Config(format!(
            "provider '{name}' (anthropic) requires auth = \"api-key\" and api_key_env"
        ))
    })?;

    let mut builder = rig::providers::anthropic::Client::builder().api_key(&key);

    if let Some(base_url) = &provider.base_url {
        builder = builder.base_url(base_url);
    }

    builder.build().map_err(|e| {
        Error::Llm(format!(
            "failed to build Anthropic client for '{name}': {e}"
        ))
    })
}

fn build_embed_model(model: ModelRef<'_>) -> Result<EmbedModel> {
    let ndims = model.provider.embedding_dimension.ok_or_else(|| {
        Error::Config(format!(
            "provider '{}' is used for embeddings but has no embedding_dimension",
            model.provider_name
        ))
    })?;
    let ndims = usize::try_from(ndims)
        .map_err(|e| Error::Config(format!("embedding_dimension overflow: {e}")))?;

    match model.provider.provider_type {
        ProviderType::Ollama => {
            let client = build_ollama_client(model.provider_name, model.provider)?;
            Ok(EmbedModel::Ollama(
                client.embedding_model_with_ndims(model.model, ndims),
            ))
        }
        ProviderType::Openai => {
            let client = build_openai_client(model.provider_name, model.provider)?;
            Ok(EmbedModel::OpenAi(
                client.embedding_model_with_ndims(model.model, ndims),
            ))
        }
        ProviderType::Anthropic => Err(Error::Config(format!(
            "embedding_model '{model}': anthropic does not serve embeddings"
        ))),
    }
}

/// The configured embedding model, or `None` when `[general].embedding_model`
/// is unset.
///
/// # Errors
///
/// Returns an error if the reference or provider is invalid.
pub fn optional_embedding_model(config: &Config) -> Result<Option<EmbedModel>> {
    let Some(model) = config.embedding_model_ref()? else {
        tracing::info!("no embedding model configured");
        return Ok(None);
    };
    tracing::info!(model = %model, "using embedding model");
    build_embed_model(model).map(Some)
}

/// The configured embedding model, required.
///
/// # Errors
///
/// Returns an error if `[general].embedding_model` is unset or invalid.
pub fn required_embedding_model(config: &Config) -> Result<EmbedModel> {
    let model = config.embedding_model_ref()?.ok_or_else(|| {
        Error::Config(format!(
            "no embedding model configured — set [general].embedding_model = \"PROVIDER/MODEL\" in {}",
            crate::config::config_file_path().display()
        ))
    })?;
    tracing::info!(model = %model, "using embedding model");
    build_embed_model(model)
}

/// `provider/model` for status lines, or a placeholder.
#[must_use]
pub fn chat_model_display(config: &Config) -> String {
    config
        .chat_model_ref()
        .map_or_else(|_| String::from("no chat model"), |m| m.to_string())
}

/// Run one agent turn in `session_id` with the configured chat and
/// embedding models, replaying the session's history to the model and
/// recording the turn when it completes.
///
/// This is the single dispatch point over provider types; interfaces call it
/// rather than matching on `provider_type` themselves.
///
/// # Errors
///
/// Returns an error if no chat model is configured, a provider cannot be
/// built, the session does not exist, or the agent turn fails.
pub async fn run_turn(
    config: &Config,
    db: SharedDb,
    session_id: &str,
    policy: WritePolicy,
    message: &str,
    sink: EventSink,
) -> Result<AgentResponse> {
    let chat = config.chat_model_ref()?;
    let embedding_model = required_embedding_model(config)?;

    let (prompt, history) = {
        let guard = db
            .lock()
            .map_err(|e| Error::Analysis(format!("mutex poisoned: {e}")))?;
        let session = sessions::get_session(&guard, session_id)?
            .ok_or_else(|| Error::Analysis(format!("session '{session_id}' does not exist")))?;
        let prompt = PromptOptions {
            mode: session.mode,
            pinned_token_budget: config.retrieval.pinned_token_budget,
            context: context::combined(&guard)?,
            context_max_tokens: config.context.max_tokens,
        };
        (
            prompt,
            sessions::history_for_model(&guard, session_id, config.analysis.history_token_budget)?,
        )
    };

    tracing::info!(chat_model = %chat, session = session_id, prior_messages = history.len(), "starting agent turn");

    let response = dispatch(
        config,
        Arc::clone(&db),
        chat,
        embedding_model,
        policy,
        prompt,
        history,
        message,
        sink,
    )
    .await?;

    let guard = db
        .lock()
        .map_err(|e| Error::Analysis(format!("mutex poisoned: {e}")))?;
    sessions::record_turn(&guard, session_id, message, &response)?;
    Ok(response)
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal dispatch over provider types"
)]
async fn dispatch(
    config: &Config,
    db: SharedDb,
    chat: ModelRef<'_>,
    embedding_model: EmbedModel,
    policy: WritePolicy,
    prompt: PromptOptions,
    history: Vec<rig::message::Message>,
    message: &str,
    sink: EventSink,
) -> Result<AgentResponse> {
    match chat.provider.provider_type {
        ProviderType::Ollama => {
            let client = build_ollama_client(chat.provider_name, chat.provider)?;
            agent::run_analysis(
                db,
                client.completion_model(chat.model),
                embedding_model,
                &config.analysis,
                &config.retrieval,
                policy,
                prompt,
                history,
                message,
                sink,
            )
            .await
        }
        ProviderType::Openai => {
            let client = build_openai_client(chat.provider_name, chat.provider)?;
            agent::run_analysis(
                db,
                client.completion_model(chat.model),
                embedding_model,
                &config.analysis,
                &config.retrieval,
                policy,
                prompt,
                history,
                message,
                sink,
            )
            .await
        }
        ProviderType::Anthropic => {
            let client = build_anthropic_client(chat.provider_name, chat.provider)?;
            agent::run_analysis(
                db,
                client.completion_model(chat.model),
                embedding_model,
                &config.analysis,
                &config.retrieval,
                policy,
                prompt,
                history,
                message,
                sink,
            )
            .await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(toml_text: &str) -> Config {
        match Config::parse(toml_text) {
            Ok(c) => c,
            Err(e) => panic_config(&e.to_string()),
        }
    }

    #[expect(clippy::panic, reason = "test helper: config fixtures must parse")]
    fn panic_config(msg: &str) -> Config {
        panic!("fixture config failed to parse: {msg}");
    }

    #[test]
    fn display_name_reports_model_ref_or_placeholder() {
        let config =
            parse("[general]\nchat_model = \"o/llama3\"\n[providers.o]\ntype = \"ollama\"\n");
        assert_eq!(chat_model_display(&config), "o/llama3");
        assert_eq!(chat_model_display(&Config::default()), "no chat model");
    }

    #[test]
    fn optional_embedding_model_is_none_when_unset() {
        assert!(matches!(
            optional_embedding_model(&Config::default()),
            Ok(None)
        ));
    }

    #[test]
    fn required_embedding_model_errors_when_unset() {
        let err = required_embedding_model(&Config::default()).err();
        assert!(err.is_some_and(|e| e.to_string().contains("embedding_model")));
    }

    #[test]
    fn api_key_mode_requires_the_env_var_to_be_set() {
        let config = parse(
            "[general]\nembedding_model = \"o/e\"\n[providers.o]\ntype = \"openai\"\nauth = \"api-key\"\napi_key_env = \"QUACK_TEST_KEY_THAT_IS_UNSET\"\nembedding_dimension = 4\n",
        );
        let err = required_embedding_model(&config).err();
        assert!(err.is_some_and(|e| e.to_string().contains("QUACK_TEST_KEY_THAT_IS_UNSET")));
    }

    #[test]
    fn ollama_embedding_model_builds_without_a_key() {
        let config = parse(
            "[general]\nembedding_model = \"o/nomic\"\n[providers.o]\ntype = \"ollama\"\nembedding_dimension = 4\n",
        );
        let model = required_embedding_model(&config);
        assert!(model.is_ok_and(|m| m.ndims() == 4));
    }
}
