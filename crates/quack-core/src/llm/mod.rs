//! LLM provider construction on top of rig.
//!
//! Everything that turns a `[providers.<name>]` config entry plus a
//! `PROVIDER/MODEL` reference into a rig client lives here, so the interfaces
//! never build providers themselves.

pub mod oauth;

use rig::client::EmbeddingsClient;
use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};
use secrecy::ExposeSecret;
use std::sync::{Arc, Mutex};

use rig::prelude::*;

use crate::analysis::agent::{self, AgentResponse};
use crate::analysis::events::{self, AgentEvent, EventSink};
use crate::analysis::policy::WritePolicy;
use crate::analysis::text_to_sql::PromptOptions;
use crate::analysis::tools::{ReaderDb, SharedDb};
use crate::config::config_file_path;
use crate::config::{AuthMode, Config, ModelRef, ProviderConfig, ProviderType};
use crate::embedding::{Embedder, Input, Profile};
use crate::error::{Error, Result};
use crate::graph::extract as graph_extract;
use crate::ontology::{Ontology, documents};
use crate::priority::{Priority, with_priority};
use crate::storage::{context, sessions};
pub use tokio_util::sync::CancellationToken;

pub mod limit;

pub use limit::LimitedHttp;

/// Every rig client quack builds sends through [`LimitedHttp`], so each
/// provider's `max_concurrent_requests` bounds the model requests in flight.
type OllamaClient = rig::providers::ollama::Client<LimitedHttp>;
type OpenAiClient = rig::providers::openai::CompletionsClient<LimitedHttp>;
type AnthropicClient = rig::providers::anthropic::Client<LimitedHttp>;
type OpenAiEmbeddingModel = rig::providers::openai::GenericEmbeddingModel<
    rig::providers::openai::OpenAICompletionsExt,
    LimitedHttp,
>;

/// How long every Ollama request asks the server to keep the model loaded.
/// Ollama's own default is 5 minutes (`OLLAMA_KEEP_ALIVE`), which a gap
/// between tool calls or turns can exceed, and a reload of a 20B model
/// costs several seconds (measured live in the perf handoff).
pub const OLLAMA_KEEP_ALIVE: &str = "30m";

/// The smallest context window the embedding model is loaded with. Ollama
/// otherwise loads it at the model's full length (32k for qwen3-embedding,
/// 5.8 GB of KV cache against 2.1 GB at 2,048, measured live) even though
/// no input is longer than a chunk.
const OLLAMA_EMBED_MIN_CTX: u32 = 2048;

/// Ollama's `/api/embed`, with the two load options rig's own embedding
/// client never sends: `num_ctx`, sized to the longest input quack embeds
/// (a chunk) instead of the model's maximum, and `keep_alive`, so the
/// embedding model stays resident between the query embedding and the
/// chat call of one turn instead of lapsing on Ollama's 5-minute default.
#[derive(Clone)]
pub struct OllamaEmbedder {
    client: OllamaClient,
    model: String,
    ndims: usize,
    num_ctx: u32,
}

impl OllamaEmbedder {
    /// The `num_ctx` for `chunk_size_tokens`-token inputs: twice the chunk
    /// size, since the chunker counts cl100k tokens and the embedding
    /// model's tokenizer may count more, rounded up to a power of two and
    /// never below [`OLLAMA_EMBED_MIN_CTX`].
    #[must_use]
    pub fn context_window(chunk_size_tokens: u32) -> u32 {
        chunk_size_tokens
            .saturating_mul(2)
            .checked_next_power_of_two()
            .unwrap_or(u32::MAX)
            .max(OLLAMA_EMBED_MIN_CTX)
    }

    /// The request body Ollama receives for `texts`.
    #[must_use]
    pub fn request_body(&self, texts: &[String]) -> serde_json::Value {
        serde_json::json!({
            "model": self.model,
            "input": texts,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": { "num_ctx": self.num_ctx },
        })
    }
}

#[derive(serde::Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f64>>,
}

impl EmbeddingModel for OllamaEmbedder {
    const MAX_DOCUMENTS: usize = 1024;
    type Client = OllamaClient;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self {
            client: client.clone(),
            model: model.into(),
            ndims: dims.unwrap_or_default(),
            num_ctx: OLLAMA_EMBED_MIN_CTX,
        }
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> std::result::Result<Vec<Embedding>, EmbeddingError> {
        use rig::http_client::{self, HttpClientExt};

        let texts: Vec<String> = texts.into_iter().collect();
        let body = serde_json::to_vec(&self.request_body(&texts))?;
        let request = self
            .client
            .post("api/embed")?
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;
        let response = self.client.send::<_, Vec<u8>>(request).await?;
        let status = response.status();
        if !status.is_success() {
            let text = http_client::text(response).await?;
            return Err(EmbeddingError::from_http_response(status, text));
        }
        let bytes: Vec<u8> = response.into_body().await?;
        let parsed: OllamaEmbedResponse = serde_json::from_slice(&bytes)?;
        if parsed.embeddings.len() != texts.len() {
            return Err(EmbeddingError::ResponseError(format!(
                "ollama returned {} embeddings for {} inputs",
                parsed.embeddings.len(),
                texts.len()
            )));
        }
        Ok(parsed
            .embeddings
            .into_iter()
            .zip(texts)
            .map(|(vec, document)| Embedding { document, vec })
            .collect())
    }
}

/// The configured embedding model under the configured profile: what every
/// interface embeds with.
pub type Embeddings = Embedder<EmbedModel>;

/// Embedding model over every provider that supports embeddings.
#[derive(Clone)]
pub enum EmbedModel {
    Ollama(OllamaEmbedder),
    OpenAi(OpenAiEmbeddingModel),
    /// An OpenAI-compatible endpoint behind OAuth: the bearer can change
    /// between batches of a long ingest, so the client is rebuilt per call
    /// with whatever token the manager holds then.
    OpenAiOAuth {
        manager: Arc<oauth::TokenManager>,
        base_url: Option<String>,
        model: String,
        ndims: usize,
        /// The provider's limited client, reused by every rebuild.
        http: LimitedHttp,
    },
}

impl EmbedModel {
    async fn oauth_model(
        manager: &oauth::TokenManager,
        base_url: Option<&str>,
        model: &str,
        ndims: usize,
        http: &LimitedHttp,
    ) -> std::result::Result<OpenAiEmbeddingModel, EmbeddingError> {
        let token = manager
            .access_token()
            .await
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;
        let client = openai_client_with_key(
            manager.provider(),
            base_url,
            token.expose_secret(),
            http.clone(),
        )
        .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;
        Ok(client.embedding_model_with_ndims(model, ndims))
    }
}

impl EmbeddingModel for EmbedModel {
    const MAX_DOCUMENTS: usize = 1024;
    type Client = OllamaClient;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self::Ollama(OllamaEmbedder::make(client, model, dims))
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
        #[expect(
            clippy::disallowed_methods,
            reason = "dispatch to the provider's model; callers reach this through an Embedder role"
        )]
        match self {
            Self::Ollama(m) => m.embed_texts(texts).await,
            Self::OpenAi(m) => m.embed_texts(texts).await,
            Self::OpenAiOAuth {
                manager,
                base_url,
                model,
                ndims,
                http,
            } => {
                Self::oauth_model(manager, base_url.as_deref(), model, *ndims, http)
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
/// nothing within `timeout` (`[analysis].extraction_timeout_seconds`) is
/// an error the run skips.
struct RigExtractor {
    agent: rig::agent::Agent,
    timeout: std::time::Duration,
}

/// `[analysis].extraction_timeout_seconds` as a duration, at least one
/// second.
fn extraction_timeout(config: &Config) -> std::time::Duration {
    std::time::Duration::from_secs(config.analysis.extraction_timeout_seconds.max(1))
}

impl documents::Extractor for RigExtractor {
    fn extract<'a>(&'a self, text: &'a str) -> documents::ExtractFuture<'a> {
        Box::pin(async move {
            let answer = stream_answer(&self.agent, text, self.timeout, "extraction").await?;
            documents::parse_extraction(&answer)
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
                    if r.usage.has_values() {
                        tracing::debug!(
                            call = what,
                            input_tokens = r.usage.input_tokens,
                            output_tokens = r.usage.output_tokens,
                            "provider token usage"
                        );
                    }
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

fn extraction_agent<M>(model: M, timeout: std::time::Duration) -> RigExtractor
where
    M: rig::completion::CompletionModel + Clone + Send + Sync + 'static,
{
    RigExtractor {
        agent: rig::agent::AgentBuilder::new(model)
            .preamble(documents::EXTRACTION_PROMPT)
            .temperature(0.0)
            .build(),
        timeout,
    }
}

/// The chat model as a constrained graph extractor: one tool-less agent
/// whose preamble carries the ontology.
struct RigGraphExtractor {
    agent: rig::agent::Agent,
    timeout: std::time::Duration,
}

impl graph_extract::GraphExtractor for RigGraphExtractor {
    fn extract<'a>(&'a self, text: &'a str) -> graph_extract::ExtractFuture<'a> {
        Box::pin(async move {
            let answer = stream_answer(&self.agent, text, self.timeout, "graph extraction").await?;
            tracing::debug!(answer = %answer, "graph extraction answer");
            graph_extract::parse_extraction(&answer)
        })
    }
}

fn graph_agent<M>(model: M, ontology: &Ontology, timeout: std::time::Duration) -> RigGraphExtractor
where
    M: rig::completion::CompletionModel + Clone + Send + Sync + 'static,
{
    RigGraphExtractor {
        agent: rig::agent::AgentBuilder::new(model)
            .preamble(&graph_extract::prompt_for(ontology))
            .temperature(0.0)
            .build(),
        timeout,
    }
}

/// The configured chat model as a constrained extractor for the graph.
///
/// # Errors
///
/// Returns an error when no chat model is configured or the provider
/// cannot be built.
pub async fn graph_extractor(
    config: &Config,
    ontology: &Ontology,
) -> Result<Box<dyn graph_extract::GraphExtractor>> {
    let chat = config.chat_model_ref()?;
    Ok(match chat.provider.provider_type {
        ProviderType::Ollama => Box::new(graph_agent(
            build_ollama_client(config, chat.provider_name, chat.provider)
                .await?
                .completion_model(chat.model),
            ontology,
            extraction_timeout(config),
        )),
        ProviderType::Openai => Box::new(graph_agent(
            build_openai_client(config, chat.provider_name, chat.provider)
                .await?
                .completion_model(chat.model),
            ontology,
            extraction_timeout(config),
        )),
        ProviderType::Anthropic => Box::new(graph_agent(
            build_anthropic_client(config, chat.provider_name, chat.provider)
                .await?
                .completion_model(chat.model),
            ontology,
            extraction_timeout(config),
        )),
    })
}

/// The configured chat model as an extractor for ontology induction.
///
/// # Errors
///
/// Returns an error when no chat model is configured or the provider
/// cannot be built (a missing key, a needed login).
pub async fn chat_extractor(config: &Config) -> Result<Box<dyn documents::Extractor>> {
    let chat = config.chat_model_ref()?;
    Ok(match chat.provider.provider_type {
        ProviderType::Ollama => Box::new(extraction_agent(
            build_ollama_client(config, chat.provider_name, chat.provider)
                .await?
                .completion_model(chat.model),
            extraction_timeout(config),
        )),
        ProviderType::Openai => Box::new(extraction_agent(
            build_openai_client(config, chat.provider_name, chat.provider)
                .await?
                .completion_model(chat.model),
            extraction_timeout(config),
        )),
        ProviderType::Anthropic => Box::new(extraction_agent(
            build_anthropic_client(config, chat.provider_name, chat.provider)
                .await?
                .completion_model(chat.model),
            extraction_timeout(config),
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
    embedder: &Embeddings,
    names: &[String],
    threshold: f64,
) -> Result<std::collections::HashMap<(String, String), bool>> {
    let mut out = std::collections::HashMap::new();
    if names.len() < 2 {
        return Ok(out);
    }
    let inputs: Vec<Input> = names
        .iter()
        .map(|n| Input::Similarity(n.replace('_', " ")))
        .collect();
    let vectors = embedder.embed(&inputs).await?;
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

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
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

/// Ollama's model listings, `GET /api/ps` (loaded) and `GET /api/tags`
/// (pulled), which share this shape.
#[derive(serde::Deserialize)]
pub(crate) struct OllamaRunningModels {
    #[serde(default)]
    pub(crate) models: Vec<OllamaRunningModel>,
}

#[derive(serde::Deserialize)]
pub(crate) struct OllamaRunningModel {
    #[serde(default)]
    pub(crate) name: String,
    #[serde(default)]
    model: String,
}

impl OllamaRunningModels {
    /// Whether `model` is among the loaded ones; a bare name matches its
    /// `:latest` tag, which is how Ollama reports it.
    pub(crate) fn holds(&self, model: &str) -> bool {
        let wanted = if model.contains(':') {
            model.to_owned()
        } else {
            format!("{model}:latest")
        };
        self.models
            .iter()
            .any(|m| m.name == wanted || m.model == wanted || m.name == model || m.model == model)
    }
}

/// Whether Ollama already has `model` in memory (`GET /api/ps`). A first
/// request after idle loads the model, which took 5 seconds for a 12 GB
/// model measured live and shows the user nothing meanwhile, so the turn
/// says so first. Any failure to ask counts as loaded: the chat call that
/// follows reports the real error.
async fn ollama_model_resident(client: &OllamaClient, model: &str) -> bool {
    use rig::http_client::HttpClientExt;

    let listed = async {
        let request = client.get("api/ps")?.body(Vec::new())?;
        let response = client.send::<_, Vec<u8>>(request).await?;
        let bytes: Vec<u8> = response.into_body().await?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(serde_json::from_slice::<
            OllamaRunningModels,
        >(&bytes)?)
    };
    match listed.await {
        Ok(running) => running.holds(model),
        Err(e) => {
            tracing::debug!(error = %e, "could not list Ollama's loaded models");
            true
        }
    }
}

async fn build_ollama_client(
    config: &Config,
    name: &str,
    provider: &ProviderConfig,
) -> Result<OllamaClient> {
    let key = credential(config, name, provider)
        .await?
        .map(rig::providers::ollama::OllamaApiKey::from)
        .unwrap_or_default();

    let mut builder = rig::providers::ollama::Client::builder()
        .api_key(key)
        .http_client(LimitedHttp::for_provider(name, provider));

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
) -> Result<OpenAiClient> {
    let key = credential(config, name, provider).await?.ok_or_else(|| {
        Error::Config(format!(
            "provider '{name}' (openai) requires auth = \"api-key\" or \"oauth\""
        ))
    })?;
    openai_client_with_key(
        name,
        provider.base_url.as_deref(),
        &key,
        LimitedHttp::for_provider(name, provider),
    )
}

fn openai_client_with_key(
    name: &str,
    base_url: Option<&str>,
    key: &str,
    http: LimitedHttp,
) -> Result<OpenAiClient> {
    let mut builder = rig::providers::openai::CompletionsClient::builder()
        .api_key(key)
        .http_client(http);

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
) -> Result<AnthropicClient> {
    let key = credential(config, name, provider).await?.ok_or_else(|| {
        Error::Config(format!(
            "provider '{name}' (anthropic) requires auth = \"api-key\" or \"oauth\""
        ))
    })?;

    let mut builder = rig::providers::anthropic::Client::builder()
        .api_key(&key)
        .http_client(LimitedHttp::for_provider(name, provider));

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
            Ok(EmbedModel::Ollama(OllamaEmbedder {
                client,
                model: model.model.to_owned(),
                ndims,
                num_ctx: OllamaEmbedder::context_window(config.ingestion.chunk_size_tokens),
            }))
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
                http: LimitedHttp::for_provider(model.provider_name, model.provider),
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
pub async fn optional_embedding_model(config: &Config) -> Result<Option<Embeddings>> {
    let (Some(model), Some(profile)) =
        (config.embedding_model_ref()?, Profile::from_config(config)?)
    else {
        tracing::info!("no embedding model configured");
        return Ok(None);
    };
    tracing::info!(model = %model, "using embedding model");
    Ok(Some(Embedder::new(
        build_embed_model(config, model).await?,
        profile,
    )))
}

/// The configured embedding model, required.
///
/// # Errors
///
/// Returns an error if `[general].embedding_model` is unset or invalid.
pub async fn required_embedding_model(config: &Config) -> Result<Embeddings> {
    optional_embedding_model(config).await?.ok_or_else(|| {
        Error::Config(format!(
            "no embedding model configured — set [general].embedding_model = \"PROVIDER/MODEL\" in {}",
            config_file_path().display()
        ))
    })
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
/// rather than matching on `provider_type` themselves. `reader_db` is the
/// workspace handle's reader (`analysis::tools::open_reader`), built once
/// for the handle's whole lifetime by whoever opened it — not here, so
/// starting a turn never waits on `db`'s mutex to acquire one.
///
/// # Errors
///
/// Returns an error if no chat model is configured, a provider cannot be
/// built, the session does not exist, or the agent turn fails.
#[expect(
    clippy::too_many_arguments,
    reason = "one entry point per turn; interfaces call this directly"
)]
pub async fn run_turn(
    config: &Config,
    db: SharedDb,
    reader_db: ReaderDb,
    session_id: &str,
    policy: WritePolicy,
    message: &str,
    sink: EventSink,
    cancel: CancellationToken,
) -> Result<AgentResponse> {
    let chat = config.chat_model_ref()?;
    // Without an embedding provider the agent still runs: document search
    // is keyword-only and graph entry is exact (issue #58).
    let embedding_model = optional_embedding_model(config).await?;

    // On the blocking pool, in the writer's interactive line: an async
    // worker never waits on the connection.
    let (prompt, history) = {
        let session_id = session_id.to_owned();
        let pinned_token_budget = config.retrieval.pinned_token_budget;
        let context_max_tokens = config.context.max_tokens;
        let ollama_context_cap = (chat.provider.provider_type == ProviderType::Ollama)
            .then_some(config.analysis.max_context_tokens);
        let history_budget = config.analysis.history_token_budget;
        crate::analysis::tools::with_db(&db, move |guard| {
            let session = sessions::get_session(guard, &session_id)?
                .ok_or_else(|| Error::Analysis(format!("session '{session_id}' does not exist")))?;
            let prompt = PromptOptions {
                mode: session.mode,
                write_policy: policy,
                pinned_token_budget,
                context: context::combined(guard)?,
                context_max_tokens,
                ollama_context_cap,
            };
            Ok((
                prompt,
                sessions::history_for_model(guard, &session_id, history_budget)?,
            ))
        })
        .await?
    };

    tracing::info!(chat_model = %chat, session = session_id, prior_messages = history.len(), "starting agent turn");

    // Events pass through here on their way out so the text streamed so
    // far is known if the turn is cancelled (issue #45).
    let (inner_sink, mut inner_events) = events::channel();
    let streamed: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let forward = tokio::spawn({
        let streamed = Arc::clone(&streamed);
        let outer = sink.clone();
        async move {
            while let Some(event) = inner_events.recv().await {
                if let AgentEvent::TextDelta(text) = &event
                    && let Ok(mut so_far) = streamed.lock()
                {
                    so_far.push_str(text);
                }
                if outer.send(event).is_err() {
                    break;
                }
            }
        }
    });
    // Someone is watching this turn: its model calls, and the tools' calls
    // inside it, go ahead of background work at the provider (design 4.1).
    let turn = with_priority(
        Priority::Interactive,
        dispatch(
            config,
            Arc::clone(&db),
            reader_db,
            chat,
            embedding_model,
            policy,
            prompt,
            history,
            message,
            inner_sink,
        ),
    );
    let outcome = tokio::select! {
        biased;
        () = cancel.cancelled() => None,
        outcome = turn => Some(outcome),
    };
    // Dropping the turn dropped its sink; the forwarder ends with it.
    drop(forward.await);

    let response = if let Some(outcome) = outcome {
        outcome?
    } else {
        let mut content = streamed.lock().map(|s| s.clone()).unwrap_or_default();
        if !content.trim().is_empty() {
            content.push_str("\n\n");
        }
        content.push_str(CANCELLED_NOTE);
        let response = AgentResponse {
            content,
            cancelled: true,
            ..AgentResponse::default()
        };
        tracing::info!(session = session_id, "agent turn cancelled");
        drop(sink.send(AgentEvent::TurnComplete(response.clone())));
        response
    };

    let (session, text, recorded) = (session_id.to_owned(), message.to_owned(), response.clone());
    crate::analysis::tools::with_db(&db, move |guard| {
        sessions::record_turn(guard, &session, &text, &recorded)
    })
    .await?;
    Ok(response)
}

/// What a cancelled turn's recorded answer ends with.
pub const CANCELLED_NOTE: &str = "(Cancelled by the user before the answer was complete.)";

#[expect(
    clippy::too_many_arguments,
    reason = "internal dispatch over provider types"
)]
async fn dispatch(
    config: &Config,
    db: SharedDb,
    reader_db: ReaderDb,
    chat: ModelRef<'_>,
    embedding_model: Option<Embeddings>,
    policy: WritePolicy,
    prompt: PromptOptions,
    history: Vec<rig::message::Message>,
    message: &str,
    sink: EventSink,
) -> Result<AgentResponse> {
    match chat.provider.provider_type {
        ProviderType::Ollama => {
            let client = build_ollama_client(config, chat.provider_name, chat.provider).await?;
            if !ollama_model_resident(&client, chat.model).await {
                drop(sink.send(AgentEvent::Status(format!(
                    "loading {}, then thinking; Ollama loads a model on its first request and keeps it \
                     for {OLLAMA_KEEP_ALIVE}",
                    chat.model
                ))));
            }
            agent::run_analysis(
                db,
                reader_db,
                client.completion_model(chat.model),
                embedding_model,
                &config.analysis,
                &config.retrieval,
                config.graph.options(),
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
                reader_db,
                client.completion_model(chat.model),
                embedding_model,
                &config.analysis,
                &config.retrieval,
                config.graph.options(),
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
                reader_db,
                client.completion_model(chat.model),
                embedding_model,
                &config.analysis,
                &config.retrieval,
                config.graph.options(),
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
    use crate::embedding::Dimension;

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
        assert!(model.is_ok_and(|m| m.profile().dimension == Dimension::new(4)));
    }

    #[tokio::test]
    async fn ollama_embed_requests_carry_a_bounded_window_and_keep_alive() {
        let config = parse(
            "[general]\nembedding_model = \"o/nomic\"\n[providers.o]\ntype = \"ollama\"\nembedding_dimension = 4\n[ingestion]\nchunk_size_tokens = 3000\n",
        );
        let model = required_embedding_model(&config).await;
        let Some(EmbedModel::Ollama(embedder)) = model.as_ref().ok().map(Embedder::model) else {
            fail("expected the Ollama embedder")
        };
        let body = embedder.request_body(&[String::from("a chunk")]);
        assert_eq!(body.get("model"), Some(&serde_json::json!("nomic")));
        assert_eq!(body.get("input"), Some(&serde_json::json!(["a chunk"])));
        assert_eq!(
            body.get("keep_alive"),
            Some(&serde_json::json!(OLLAMA_KEEP_ALIVE))
        );
        // 3,000 tokens doubled and rounded up to a power of two.
        assert_eq!(
            body.pointer("/options/num_ctx"),
            Some(&serde_json::json!(8192))
        );
    }

    #[test]
    fn ollama_running_models_match_bare_and_tagged_names() {
        let running: OllamaRunningModels = serde_json::from_str(
            r#"{"models":[{"name":"gpt-oss:20b","model":"gpt-oss:20b"},{"name":"llama3:latest","model":"llama3:latest"}]}"#,
        )
        .unwrap_or(OllamaRunningModels { models: Vec::new() });
        assert!(running.holds("gpt-oss:20b"));
        assert!(running.holds("llama3"));
        assert!(running.holds("llama3:latest"));
        assert!(!running.holds("gpt-oss"));
        assert!(!running.holds("qwen3:8b"));
        let empty: OllamaRunningModels =
            serde_json::from_str("{}").unwrap_or(OllamaRunningModels { models: Vec::new() });
        assert!(!empty.holds("gpt-oss:20b"));
    }

    #[test]
    fn ollama_embed_window_never_drops_below_the_floor() {
        assert_eq!(OllamaEmbedder::context_window(512), OLLAMA_EMBED_MIN_CTX);
        assert_eq!(OllamaEmbedder::context_window(1024), OLLAMA_EMBED_MIN_CTX);
        assert_eq!(OllamaEmbedder::context_window(1025), 4096);
        assert_eq!(OllamaEmbedder::context_window(u32::MAX), u32::MAX);
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

    /// A cancelled turn is still a turn (issue #45): the session records
    /// the question and a cancelled answer, `TurnComplete` is emitted, and
    /// the model is never called (the token is cancelled before the turn
    /// starts, and the provider address is unreachable anyway).
    #[tokio::test]
    async fn cancelled_turns_are_recorded_and_completed() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = parse(
            "[general]\nchat_model = \"o/m\"\nembedding_model = \"o/e\"\n[providers.o]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\nembedding_dimension = 4\n",
        );
        config.general.data_dir = dir.path().to_path_buf();
        let db = crate::storage::workspace::WorkspaceDb::open(&config, "ws")
            .unwrap_or_else(|e| fail(&e.to_string()));
        let session = sessions::create_session(&db, "o/m", sessions::ChatMode::Chat, None)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let db: SharedDb = Arc::new(
            crate::storage::writer::Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())),
        );
        let reader_db =
            crate::analysis::tools::open_reader(&db, config.analysis.reader_pool_size).await;
        let (sink, mut events) = events::channel();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let response = run_turn(
            &config,
            Arc::clone(&db),
            reader_db,
            &session.id,
            WritePolicy::Deny,
            "how many storms?",
            sink,
            cancel,
        )
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(response.cancelled);
        assert_eq!(response.content, CANCELLED_NOTE);
        let last = events.recv().await;
        assert!(
            matches!(&last, Some(AgentEvent::TurnComplete(r)) if r.cancelled),
            "{last:?}"
        );
        let id = session.id.clone();
        let messages = db
            .run(move |guard| sessions::messages(guard, &id))
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let roles: Vec<sessions::MessageRole> = messages.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec![
                sessions::MessageRole::User,
                sessions::MessageRole::Assistant
            ]
        );
        assert!(
            messages
                .last()
                .is_some_and(|m| m.content.contains("Cancelled by the user"))
        );
    }
}
