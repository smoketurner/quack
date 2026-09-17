//! LLM provider construction on top of rig.
//!
//! Everything that turns a `[providers.<name>]` config entry plus a
//! `PROVIDER/MODEL` reference into a rig client lives here, so the interfaces
//! never build providers themselves.

pub mod oauth;

use rig::client::EmbeddingsClient;
use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};
use secrecy::ExposeSecret;
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

type OpenAiEmbeddingModel =
    rig::providers::openai::GenericEmbeddingModel<rig::providers::openai::OpenAICompletionsExt>;

/// Embedding model over every provider that supports embeddings.
#[derive(Clone)]
pub enum EmbedModel {
    Ollama(rig::providers::ollama::EmbeddingModel),
    OpenAi(OpenAiEmbeddingModel),
    /// An OpenAI-compatible endpoint behind OAuth: the bearer can change
    /// between batches of a long ingest, so the client is rebuilt per call
    /// with whatever token the manager holds then.
    OpenAiOAuth {
        manager: Arc<oauth::TokenManager>,
        base_url: Option<String>,
        model: String,
        ndims: usize,
    },
}

impl EmbedModel {
    async fn oauth_model(
        manager: &oauth::TokenManager,
        base_url: Option<&str>,
        model: &str,
        ndims: usize,
    ) -> std::result::Result<OpenAiEmbeddingModel, EmbeddingError> {
        let token = manager
            .access_token()
            .await
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;
        let client = openai_client_with_key(manager.provider(), base_url, token.expose_secret())
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;
        Ok(client.embedding_model_with_ndims(model, ndims))
    }
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
            Self::OpenAiOAuth { ndims, .. } => *ndims,
        }
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> std::result::Result<Vec<Embedding>, EmbeddingError> {
        match self {
            Self::Ollama(m) => m.embed_texts(texts).await,
            Self::OpenAi(m) => m.embed_texts(texts).await,
            Self::OpenAiOAuth {
                manager,
                base_url,
                model,
                ndims,
            } => {
                Self::oauth_model(manager, base_url.as_deref(), model, *ndims)
                    .await?
                    .embed_texts(texts)
                    .await
            }
        }
    }
}

/// Open extraction through a rig agent: one streamed prompt per chunk,
/// the text collected and parsed as JSON. Streaming is the path the chat
/// agent uses and the one Ollama answers reliably; a chunk that produces
/// nothing within [`EXTRACTION_TIMEOUT`] is an error the run skips.
struct RigExtractor {
    agent: rig::agent::Agent,
}

/// How long one chunk's extraction may take.
const EXTRACTION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

impl crate::ontology::documents::Extractor for RigExtractor {
    fn extract<'a>(&'a self, text: &'a str) -> crate::ontology::documents::ExtractFuture<'a> {
        Box::pin(async move {
            let answer = stream_answer(&self.agent, text, EXTRACTION_TIMEOUT, "extraction").await?;
            crate::ontology::documents::parse_extraction(&answer)
        })
    }
}

/// One streamed, tool-less call to `agent` with `text`, collected into the
/// answer text. Streaming keeps long generations from tripping the HTTP
/// client's read timeout; `what` names the call in errors.
///
/// # Errors
///
/// Returns an error when the call fails or produces nothing within
/// `timeout`.
pub async fn stream_answer(
    agent: &rig::agent::Agent,
    text: &str,
    timeout: std::time::Duration,
    what: &str,
) -> Result<String> {
    use futures::StreamExt;
    use rig::streaming::StreamedAssistantContent;
    let collect = async {
        let mut stream = agent
            .stream_chat(text, Vec::<rig::message::Message>::new())
            .await;
        let mut answer = String::new();
        let mut final_text: Option<String> = None;
        while let Some(item) = stream.next().await {
            match item.map_err(|e| Error::Llm(format!("{what} call failed: {e}")))? {
                rig::agent::MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::Text(t),
                ) => answer.push_str(&t.text),
                rig::agent::MultiTurnStreamItem::FinalResponse(r) => {
                    final_text = Some(r.output);
                }
                _ => {}
            }
        }
        Ok::<String, Error>(match final_text {
            Some(t) if answer.trim().is_empty() => t,
            _ => answer,
        })
    };
    tokio::time::timeout(timeout, collect).await.map_err(|_| {
        Error::Llm(format!(
            "{what} call produced nothing within {} s",
            timeout.as_secs()
        ))
    })?
}

fn extraction_agent<M>(model: M) -> RigExtractor
where
    M: rig::completion::CompletionModel + Clone + Send + Sync + 'static,
{
    RigExtractor {
        agent: rig::agent::AgentBuilder::new(model)
            .preamble(crate::ontology::documents::EXTRACTION_PROMPT)
            .temperature(0.0)
            .build(),
    }
}

/// The configured chat model as an extractor for ontology induction.
///
/// # Errors
///
/// Returns an error when no chat model is configured or the provider
/// cannot be built (a missing key, a needed login).
pub async fn chat_extractor(
    config: &Config,
) -> Result<Box<dyn crate::ontology::documents::Extractor>> {
    let chat = config.chat_model_ref()?;
    Ok(match chat.provider.provider_type {
        ProviderType::Ollama => Box::new(extraction_agent(
            build_ollama_client(config, chat.provider_name, chat.provider)
                .await?
                .completion_model(chat.model),
        )),
        ProviderType::Openai => Box::new(extraction_agent(
            build_openai_client(config, chat.provider_name, chat.provider)
                .await?
                .completion_model(chat.model),
        )),
        ProviderType::Anthropic => Box::new(extraction_agent(
            build_anthropic_client(config, chat.provider_name, chat.provider)
                .await?
                .completion_model(chat.model),
        )),
    })
}

/// A similarity check over the embedding model for clustering type and
/// relation names: cosine at or above `threshold`.
///
/// # Errors
///
/// Returns an error when embedding fails.
pub async fn name_similarity(
    model: &EmbedModel,
    names: &[String],
    threshold: f64,
) -> Result<std::collections::HashMap<(String, String), bool>> {
    let mut out = std::collections::HashMap::new();
    if names.len() < 2 {
        return Ok(out);
    }
    let embeddings = model
        .embed_texts(names.iter().map(|n| n.replace('_', " ")))
        .await
        .map_err(|e| Error::Embedding(e.to_string()))?;
    let vectors: Vec<Vec<f64>> = embeddings.into_iter().map(|e| e.vec).collect();
    for (i, a) in names.iter().enumerate() {
        for (j, b) in names.iter().enumerate() {
            if i == j {
                continue;
            }
            let (Some(va), Some(vb)) = (vectors.get(i), vectors.get(j)) else {
                continue;
            };
            out.insert((a.clone(), b.clone()), cosine(va, vb) >= threshold);
        }
    }
    Ok(out)
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Embed one query string as `f32`s, the width stored in the workspace.
///
/// # Errors
///
/// Returns an error when the provider call fails.
pub async fn embed_query(model: &EmbedModel, text: &str) -> Result<Vec<f32>> {
    let embedding = model
        .embed_text(text)
        .await
        .map_err(|e| Error::Embedding(e.to_string()))?;
    #[expect(clippy::cast_possible_truncation, reason = "stored vectors are f32")]
    Ok(embedding.vec.into_iter().map(|v| v as f32).collect())
}

/// The bearer credential for a provider, according to its `auth` mode: none,
/// the static key from the environment, or the current OAuth access token.
async fn credential(
    config: &Config,
    name: &str,
    provider: &ProviderConfig,
) -> Result<Option<String>> {
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
        AuthMode::Oauth => {
            let manager = oauth::shared_manager(&config.tokens_dir(), name, provider)?;
            let token = manager.access_token().await?;
            Ok(Some(token.expose_secret().to_owned()))
        }
    }
}

async fn build_ollama_client(
    config: &Config,
    name: &str,
    provider: &ProviderConfig,
) -> Result<rig::providers::ollama::Client> {
    let key = credential(config, name, provider)
        .await?
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

async fn build_openai_client(
    config: &Config,
    name: &str,
    provider: &ProviderConfig,
) -> Result<rig::providers::openai::CompletionsClient> {
    let key = credential(config, name, provider).await?.ok_or_else(|| {
        Error::Config(format!(
            "provider '{name}' (openai) requires auth = \"api-key\" or \"oauth\""
        ))
    })?;
    openai_client_with_key(name, provider.base_url.as_deref(), &key)
}

fn openai_client_with_key(
    name: &str,
    base_url: Option<&str>,
    key: &str,
) -> Result<rig::providers::openai::CompletionsClient> {
    let mut builder = rig::providers::openai::CompletionsClient::builder().api_key(key);

    if let Some(base_url) = base_url {
        builder = builder.base_url(base_url);
    }

    builder
        .build()
        .map_err(|e| Error::Llm(format!("failed to build OpenAI client for '{name}': {e}")))
}

async fn build_anthropic_client(
    config: &Config,
    name: &str,
    provider: &ProviderConfig,
) -> Result<rig::providers::anthropic::Client> {
    let key = credential(config, name, provider).await?.ok_or_else(|| {
        Error::Config(format!(
            "provider '{name}' (anthropic) requires auth = \"api-key\" or \"oauth\""
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

async fn build_embed_model(config: &Config, model: ModelRef<'_>) -> Result<EmbedModel> {
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
            let client = build_ollama_client(config, model.provider_name, model.provider).await?;
            Ok(EmbedModel::Ollama(
                client.embedding_model_with_ndims(model.model, ndims),
            ))
        }
        ProviderType::Openai if model.provider.auth == AuthMode::Oauth => {
            let manager =
                oauth::shared_manager(&config.tokens_dir(), model.provider_name, model.provider)?;
            // Fail here, typed, when no login exists; later batches refresh
            // on their own.
            drop(manager.access_token().await?);
            Ok(EmbedModel::OpenAiOAuth {
                manager,
                base_url: model.provider.base_url.clone(),
                model: model.model.to_owned(),
                ndims,
            })
        }
        ProviderType::Openai => {
            let client = build_openai_client(config, model.provider_name, model.provider).await?;
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
pub async fn optional_embedding_model(config: &Config) -> Result<Option<EmbedModel>> {
    let Some(model) = config.embedding_model_ref()? else {
        tracing::info!("no embedding model configured");
        return Ok(None);
    };
    tracing::info!(model = %model, "using embedding model");
    build_embed_model(config, model).await.map(Some)
}

/// The configured embedding model, required.
///
/// # Errors
///
/// Returns an error if `[general].embedding_model` is unset or invalid.
pub async fn required_embedding_model(config: &Config) -> Result<EmbedModel> {
    let model = config.embedding_model_ref()?.ok_or_else(|| {
        Error::Config(format!(
            "no embedding model configured — set [general].embedding_model = \"PROVIDER/MODEL\" in {}",
            crate::config::config_file_path().display()
        ))
    })?;
    tracing::info!(model = %model, "using embedding model");
    build_embed_model(config, model).await
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
    let embedding_model = required_embedding_model(config).await?;

    let (prompt, history) = {
        let guard = db
            .lock()
            .map_err(|e| Error::Analysis(format!("mutex poisoned: {e}")))?;
        let session = sessions::get_session(&guard, session_id)?
            .ok_or_else(|| Error::Analysis(format!("session '{session_id}' does not exist")))?;
        let prompt = PromptOptions {
            mode: session.mode,
            write_policy: policy,
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
            let client = build_ollama_client(config, chat.provider_name, chat.provider).await?;
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
            let client = build_openai_client(config, chat.provider_name, chat.provider).await?;
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
            let client = build_anthropic_client(config, chat.provider_name, chat.provider).await?;
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

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[test]
    fn display_name_reports_model_ref_or_placeholder() {
        let config =
            parse("[general]\nchat_model = \"o/llama3\"\n[providers.o]\ntype = \"ollama\"\n");
        assert_eq!(chat_model_display(&config), "o/llama3");
        assert_eq!(chat_model_display(&Config::default()), "no chat model");
    }

    #[tokio::test]
    async fn optional_embedding_model_is_none_when_unset() {
        assert!(matches!(
            optional_embedding_model(&Config::default()).await,
            Ok(None)
        ));
    }

    #[tokio::test]
    async fn required_embedding_model_errors_when_unset() {
        let err = required_embedding_model(&Config::default()).await.err();
        assert!(err.is_some_and(|e| e.to_string().contains("embedding_model")));
    }

    #[tokio::test]
    async fn api_key_mode_requires_the_env_var_to_be_set() {
        let config = parse(
            "[general]\nembedding_model = \"o/e\"\n[providers.o]\ntype = \"openai\"\nauth = \"api-key\"\napi_key_env = \"QUACK_TEST_KEY_THAT_IS_UNSET\"\nembedding_dimension = 4\n",
        );
        let err = required_embedding_model(&config).await.err();
        assert!(err.is_some_and(|e| e.to_string().contains("QUACK_TEST_KEY_THAT_IS_UNSET")));
    }

    #[tokio::test]
    async fn ollama_embedding_model_builds_without_a_key() {
        let config = parse(
            "[general]\nembedding_model = \"o/nomic\"\n[providers.o]\ntype = \"ollama\"\nembedding_dimension = 4\n",
        );
        let model = required_embedding_model(&config).await;
        assert!(model.is_ok_and(|m| m.ndims() == 4));
    }

    #[tokio::test]
    async fn oauth_provider_without_a_login_needs_auth_for_embeddings_and_chat() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = parse(
            "[general]\nchat_model = \"az/gpt\"\nembedding_model = \"az/emb\"\n[providers.az]\ntype = \"openai\"\nauth = \"oauth\"\nembedding_dimension = 4\n[providers.az.oauth]\nissuer_url = \"http://127.0.0.1:9\"\nclient_id = \"c\"\n",
        );
        config.general.data_dir = dir.path().to_path_buf();
        let err = required_embedding_model(&config).await.err();
        assert!(err.is_some_and(|e| matches!(e, Error::AuthRequired { .. })));
        let chat = config.chat_model_ref();
        let Ok(chat) = chat else {
            return assert!(chat.is_ok());
        };
        let err = build_openai_client(&config, chat.provider_name, chat.provider)
            .await
            .err();
        assert!(err.is_some_and(|e| matches!(e, Error::AuthRequired { .. })));
    }
}
