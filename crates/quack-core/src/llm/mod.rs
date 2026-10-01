//! LLM provider construction on top of rig.
//!
//! Everything that turns a `[providers.<name>]` config entry plus a
//! `PROVIDER/MODEL` reference into a rig model lives here, so the interfaces
//! never build providers themselves.

pub mod acting;
pub mod bedrock;
pub mod oauth;
pub mod sampling;

use jiff::Timestamp;
use rig::embeddings::Embedding;
use rig::providers::{anthropic, ollama, openai};
use rig::streaming::{Item, StreamEvent};
use rig::{DynModel, ProviderError, operation};
use secrecy::ExposeSecret;
use serde::de::DeserializeOwned;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rig::prelude::*;

use crate::analysis::agent::{AgentResponse, Analysis};
use crate::analysis::events::{self, AgentEvent, EventSink, TurnFailure};
use crate::analysis::policy::WritePolicy;
use crate::analysis::text_to_sql::PromptOptions;
use crate::analysis::tools::{ReaderDb, SharedDb};
use crate::config::{
    BaseUrl, BedrockApi, Config, Effort, ModelRef, ModelSettings, ProviderAuth, ProviderConfig,
    ProviderName, ProviderType, config_file_path,
};
use crate::embedding::{Embedder, EmbeddingModel, Profile};
use crate::error::{Error, Record, Result};
use crate::extraction::{Extract, ExtractFuture, parse_answer};
use crate::graph::extract::Extraction;
use crate::ids::SessionId;
use crate::ontology::Ontology;
use crate::ontology::documents::{self, OpenExtraction};
use crate::priority::Priority;
use crate::storage::{context, sessions};
use sampling::{Sampled, Wire};
pub use tokio_util::sync::CancellationToken;

pub mod limit;

pub use limit::LimitedHttp;

/// A chat model with its wire and transport erased: what the agent and the
/// one-shot calls run on. Every one quack builds is [`Sampled`], and every
/// HTTP one sends through [`LimitedHttp`], so each provider's
/// `max_concurrent_requests` bounds the model requests in flight.
pub type ChatModel = DynModel<operation::Completion>;

/// One of rig's embedding models with its wire and transport erased.
type RigEmbeddingModel = DynModel<operation::Embedding>;

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

/// An Ollama server as a provider reaches it: rig's settings for it (its
/// root and bearer) on the provider's limited client.
#[derive(Clone)]
pub struct OllamaEndpoint {
    settings: ollama::OllamaConfig,
    http: LimitedHttp,
}

impl OllamaEndpoint {
    async fn build(
        config: &Config,
        name: &ProviderName,
        provider: &ProviderConfig,
    ) -> Result<Self> {
        let mut settings = ollama::OllamaConfig::new();
        if let Some(key) = provider.auth.credential(config, name).await? {
            settings = settings.with_api_key(key);
        }
        if let Some(base_url) = &provider.base_url {
            settings = settings.with_base_url(base_url.root());
        }
        Ok(Self {
            settings,
            http: LimitedHttp::for_provider(name, provider).with_headers(provider.header_map()?),
        })
    }

    /// rig's client for the server.
    fn client(&self) -> ollama::Ollama {
        self.settings.clone().connect(self.http.clone())
    }

    /// A request to `path` on the server, with the bearer when there is one.
    fn request(&self, method: http::Method, path: &str) -> http::request::Builder {
        let builder = http::Request::builder()
            .method(method)
            .uri(format!("{}/{path}", self.settings.base_url))
            .header(http::header::CONTENT_TYPE, "application/json");
        if self.settings.api_key.is_empty() {
            builder
        } else {
            builder.header(
                http::header::AUTHORIZATION,
                format!("Bearer {}", self.settings.api_key.expose()),
            )
        }
    }
}

/// Ollama's `/api/embed`, with the two load options rig's own embedding
/// wire never sends: `num_ctx`, sized to the longest input quack embeds
/// (a chunk) instead of the model's maximum, and `keep_alive`, so the
/// embedding model stays resident between the query embedding and the
/// chat call of one turn instead of lapsing on Ollama's 5-minute default.
#[derive(Clone)]
pub struct OllamaEmbedder {
    endpoint: OllamaEndpoint,
    model: String,
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
    async fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> std::result::Result<Vec<Embedding>, ProviderError> {
        use rig::http_client::HttpClientExt;

        let body = serde_json::to_vec(&self.request_body(&texts))?;
        let request = self
            .endpoint
            .request(http::Method::POST, "api/embed")
            .body(body)?;
        // A non-success status is the transport's error, body and all.
        let response = self.endpoint.http.send::<_, Vec<u8>>(request).await?;
        let bytes: Vec<u8> = response.into_body().await?;
        let parsed: OllamaEmbedResponse = serde_json::from_slice(&bytes)?;
        if parsed.embeddings.len() != texts.len() {
            return Err(ProviderError::Response(format!(
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
    OpenAi(RigEmbeddingModel),
    OpenAiOAuth(OAuthEmbedding),
    Bedrock(RigEmbeddingModel),
}

/// An OpenAI-compatible embedding endpoint behind OAuth: the bearer can
/// change between batches of a long ingest, so the client is rebuilt per
/// call with whatever token the manager holds then.
#[derive(Clone)]
pub struct OAuthEmbedding {
    manager: Arc<oauth::TokenManager>,
    base_url: Option<BaseUrl>,
    model: String,
    ndims: usize,
    /// The provider's limited client, reused by every rebuild.
    http: LimitedHttp,
}

impl OAuthEmbedding {
    /// The embedding model with the current token.
    #[expect(
        clippy::result_large_err,
        reason = "rig's ProviderError, which the embedding call it feeds returns"
    )]
    async fn model(&self) -> std::result::Result<RigEmbeddingModel, ProviderError> {
        let token = self
            .manager
            .access_token()
            .await
            .map_err(|e| ProviderError::Provider(e.to_string()))?;
        Ok(openai_client(
            self.base_url.as_ref().map(BaseUrl::as_str),
            token.expose_secret(),
            self.http.clone(),
        )
        .embedding(&self.model, Some(self.ndims))
        .erase())
    }
}

impl EmbedModel {
    /// The client for `model`, whose provider must serve embeddings.
    async fn build(config: &Config, model: ModelRef<'_>) -> Result<Self> {
        let ndims = usize::try_from(config.embedding_dimension()?.get())
            .map_err(|e| Error::Config(format!("[embedding].dimension overflow: {e}")))?;
        let (name, provider) = (model.provider_name, model.provider);
        match provider.provider_type {
            ProviderType::Ollama => Ok(Self::Ollama(OllamaEmbedder {
                endpoint: OllamaEndpoint::build(config, name, provider).await?,
                model: model.model.to_owned(),
                num_ctx: OllamaEmbedder::context_window(config.ingestion.chunk_size_tokens),
            })),
            ProviderType::Openai if let Some(oauth) = provider.auth.oauth() => {
                let manager = oauth::TokenManager::shared(config, name, oauth)?;
                // Fail here, typed, when no login exists; later batches
                // refresh on their own.
                drop(manager.access_token().await?);
                Ok(Self::OpenAiOAuth(OAuthEmbedding {
                    manager,
                    base_url: provider.base_url.clone(),
                    model: model.model.to_owned(),
                    ndims,
                    http: LimitedHttp::for_provider(name, provider)
                        .with_headers(provider.header_map()?),
                }))
            }
            ProviderType::Openai => Ok(Self::OpenAi(
                build_openai_client(config, name, provider)
                    .await?
                    .embedding(model.model, Some(ndims))
                    .erase(),
            )),
            ProviderType::Anthropic => Err(Error::Config(format!(
                "[embedding].model '{model}': anthropic does not serve embeddings"
            ))),
            ProviderType::Bedrock | ProviderType::BedrockMantle => Ok(Self::Bedrock(
                bedrock::session(name, provider)
                    .await?
                    .converse(name)?
                    .embedding(model.model, Some(ndims))
                    .erase(),
            )),
        }
    }
}

impl Embeddings {
    /// The configured embedding model, or `None` when
    /// `[embedding].model` is unset.
    ///
    /// # Errors
    ///
    /// Returns an error if the reference or provider is invalid.
    pub async fn from_config(config: &Config) -> Result<Option<Self>> {
        let (Some(model), Some(profile)) =
            (config.embedding_model_ref()?, Profile::from_config(config)?)
        else {
            tracing::info!("no embedding model configured");
            return Ok(None);
        };
        tracing::info!(model = %model, "using embedding model");
        Ok(Some(Self::new(
            EmbedModel::build(config, model).await?,
            profile,
        )))
    }

    /// The configured embedding model, required.
    ///
    /// # Errors
    ///
    /// Returns an error if `[embedding].model` is unset or invalid.
    pub async fn require(config: &Config) -> Result<Self> {
        Self::from_config(config).await?.ok_or_else(|| {
            Error::Config(format!(
                "no embedding model configured — set [embedding].model = \"PROVIDER/MODEL\" in {}",
                config_file_path().display()
            ))
        })
    }
}

impl EmbeddingModel for EmbedModel {
    async fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> std::result::Result<Vec<Embedding>, ProviderError> {
        #[expect(
            clippy::disallowed_methods,
            reason = "dispatch to the provider's model; callers reach this through an Embedder role"
        )]
        // Boxed: each provider's request future is several KB, which every
        // caller's future would otherwise carry.
        let response = match self {
            Self::Ollama(m) => return Box::pin(m.embed_texts(texts)).await,
            Self::OpenAi(m) | Self::Bedrock(m) => Box::pin(m.call(texts)).await?,
            Self::OpenAiOAuth(oauth) => Box::pin(oauth.model().await?.call(texts)).await?,
        };
        Ok(response.embeddings)
    }
}

/// The chat provider's rig client, one variant per provider type, so a
/// caller builds it once and matches only where the provider matters.
enum ChatClient {
    Ollama(OllamaEndpoint),
    /// OpenAI-compatible Chat Completions.
    OpenAi(openai::OpenAI),
    Anthropic(anthropic::Anthropic),
    /// Bedrock's Converse API through the AWS SDK. Bedrock's Chat
    /// Completions is [`Self::OpenAi`] over a signing client.
    Bedrock(bedrock::BedrockClient),
    /// An OpenAI-compatible Responses API: Bedrock's, signed, or a
    /// `type = "openai"` provider's with `api = "responses"`.
    Responses(openai::OpenAI),
}

impl ChatClient {
    async fn build(config: &Config, chat: &ModelRef<'_>) -> Result<Self> {
        let (name, provider) = (chat.provider_name, chat.provider);
        Ok(match provider.provider_type {
            ProviderType::Ollama => {
                Self::Ollama(OllamaEndpoint::build(config, name, provider).await?)
            }
            ProviderType::Openai if provider.openai_chat_api() == BedrockApi::Responses => {
                Self::Responses(build_openai_client(config, name, provider).await?)
            }
            ProviderType::Openai => {
                Self::OpenAi(build_openai_client(config, name, provider).await?)
            }
            ProviderType::Anthropic => {
                Self::Anthropic(build_anthropic_client(config, name, provider).await?)
            }
            ProviderType::Bedrock | ProviderType::BedrockMantle => {
                Self::bedrock(&*bedrock::session(name, provider).await?, name, provider)?
            }
        })
    }

    /// The client for a Bedrock provider's API on its endpoint.
    fn bedrock(
        session: &bedrock::Session,
        name: &ProviderName,
        provider: &ProviderConfig,
    ) -> Result<Self> {
        let signed = || -> Result<openai::OpenAI> {
            Ok(openai_client(
                Some(&session.openai_base()),
                SIGNED_PLACEHOLDER_KEY,
                session
                    .http(name, provider)
                    .with_headers(provider.header_map()?),
            ))
        };
        Ok(match session.api() {
            BedrockApi::Converse => Self::Bedrock(session.converse(name)?),
            BedrockApi::ChatCompletions => Self::OpenAi(signed()?),
            BedrockApi::Responses => Self::Responses(signed()?),
        })
    }

    /// The API this client calls the chat model through.
    const fn wire(&self) -> Wire {
        match self {
            Self::Ollama(_) => Wire::Ollama,
            Self::OpenAi(_) => Wire::ChatCompletions,
            Self::Anthropic(_) => Wire::Anthropic,
            Self::Bedrock(_) => Wire::Converse,
            Self::Responses(_) => Wire::Responses,
        }
    }

    /// The chat model `model`, [`Sampled`] for its API at `effort`.
    fn chat_model(
        &self,
        model: &str,
        effort: Option<Effort>,
        temperature: Option<bool>,
    ) -> Result<ChatModel> {
        let wire = self.wire();
        Ok(match self {
            Self::Ollama(endpoint) => Sampled::model(
                endpoint.client().completion(model),
                model,
                wire,
                effort,
                temperature,
            )?,
            Self::OpenAi(client) => {
                Sampled::model(client.chat(model), model, wire, effort, temperature)?
            }
            Self::Anthropic(client) => {
                Sampled::model(client.completion(model), model, wire, effort, temperature)?
            }
            Self::Bedrock(client) => {
                Sampled::model(client.completion(model), model, wire, effort, temperature)?
            }
            Self::Responses(client) => Sampled::model(
                Unstored::model(client.responses(model)),
                model,
                wire,
                effort,
                temperature,
            )?,
        })
    }

    fn one_shot(
        &self,
        model: &str,
        settings: ModelSettings,
        preamble: &str,
        timeout: Duration,
        label: &'static str,
    ) -> Result<OneShotAgent> {
        Ok(OneShotAgent::new(
            self.chat_model(model, settings.background_effort, settings.temperature)?,
            preamble,
            timeout,
            label,
        ))
    }
}

/// The key rig's `OpenAI` clients are built with when [`bedrock::Signer`]
/// replaces their `Authorization` header with a `SigV4` one.
const SIGNED_PLACEHOLDER_KEY: &str = "sigv4";

/// A Responses API wire asked to keep nothing: every request carries
/// `store: false`, so neither Bedrock nor `OpenAI` retains a copy of the
/// conversation (Bedrock keeps one for 30 days by default) and no workspace content leaves the
/// workspace file's boundary to be stored (design doc section 5). quack
/// replays history itself and never uses `previous_response_id`.
#[derive(Clone)]
struct Unstored<W>(W);

impl<W> Unstored<W> {
    /// `model`, asking to store nothing.
    fn model<T>(model: Model<W, T>) -> Model<Self, T> {
        Model::new(Self(model.wire), model.transport)
    }

    fn request(
        mut request: rig::completion::CompletionRequest,
    ) -> rig::completion::CompletionRequest {
        let mut params = match request.additional_params.take() {
            Some(serde_json::Value::Object(map)) => map,
            _ => serde_json::Map::new(),
        };
        params.insert(String::from("store"), serde_json::Value::Bool(false));
        request.additional_params = Some(serde_json::Value::Object(params));
        request
    }
}

impl<W: rig::wire::Wire<Op = operation::Completion>> rig::wire::Wire for Unstored<W> {
    type Op = operation::Completion;
    type Payload = W::Payload;
    type Frame = W::Frame;
    type Decoder<'id> = W::Decoder<'id>;

    fn describe(&self) -> rig::wire::Descriptor<'_> {
        self.0.describe()
    }

    fn encode(
        &self,
        request: rig::completion::CompletionRequest,
        mode: rig::wire::Mode,
    ) -> std::result::Result<W::Payload, rig::error::EncodeError> {
        self.0.encode(Self::request(request), mode)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        self.0.decoder()
    }
}

/// A tool-less agent at temperature 0 (where the model takes one; see
/// [`sampling`]) that answers one prompt at a time,
/// streamed and collected. Streaming is the path the chat agent uses and
/// the one Ollama answers reliably, and it keeps long generations from
/// tripping the HTTP client's read timeout. Extraction and reranking are
/// both one of these with their own preamble.
pub struct OneShotAgent {
    agent: Agent,
    timeout: Duration,
    label: &'static str,
}

impl OneShotAgent {
    /// `label` names the call in errors and logs.
    #[must_use]
    pub fn new(model: ChatModel, preamble: &str, timeout: Duration, label: &'static str) -> Self {
        Self {
            agent: AgentBuilder::new(model)
                .preamble(preamble)
                .temperature(0.0)
                .build(),
            timeout,
            label,
        }
    }

    /// The model's answer to `text`.
    ///
    /// # Errors
    ///
    /// Returns an error when the call fails or produces nothing within the
    /// timeout.
    pub async fn answer(&self, text: &str) -> Result<String> {
        use futures::StreamExt;
        let what = self.label;
        let collect = async {
            let mut stream = self.agent.prompt(text).stream();
            let mut answer = String::new();
            let mut final_text: Option<String> = None;
            while let Some(item) = stream.next().await {
                match item.map_err(|e| Error::Llm(format!("{what} call failed: {e}")))? {
                    MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                        text,
                        ..
                    })) => {
                        answer.push_str(&text);
                    }
                    MultiTurnStreamItem::FinalResponse(r) => {
                        if r.usage.is_reported() {
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
        tokio::time::timeout(self.timeout, collect)
            .await
            .map_err(|_| {
                Error::Llm(format!(
                    "{what} call produced nothing within {} s",
                    self.timeout.as_secs()
                ))
            })?
    }
}

impl<T: DeserializeOwned + Send> Extract<T> for OneShotAgent {
    fn extract<'a>(&'a self, text: &'a str) -> ExtractFuture<'a, T> {
        Box::pin(async move {
            let answer = self.answer(text).await?;
            tracing::debug!(call = self.label, answer = %answer, "extraction answer");
            parse_answer(&answer)
        })
    }
}

/// The configured chat model as a constrained extractor for the graph: its
/// preamble carries the ontology.
///
/// # Errors
///
/// Returns an error when no chat model is configured or the provider
/// cannot be built.
pub async fn graph_extractor(
    config: &Config,
    ontology: &Ontology,
) -> Result<Box<dyn Extract<Extraction>>> {
    let chat = config.chat_model_ref()?;
    Ok(Box::new(ChatClient::build(config, &chat).await?.one_shot(
        chat.model,
        config.model_settings(chat),
        &ontology.extraction_prompt(),
        config.analysis.extraction_timeout(),
        "graph extraction",
    )?))
}

/// The configured chat model as an open extractor for ontology induction.
///
/// # Errors
///
/// Returns an error when no chat model is configured or the provider
/// cannot be built (a missing key, a needed login).
pub async fn chat_extractor(config: &Config) -> Result<Box<dyn Extract<OpenExtraction>>> {
    let chat = config.chat_model_ref()?;
    Ok(Box::new(ChatClient::build(config, &chat).await?.one_shot(
        chat.model,
        config.model_settings(chat),
        documents::EXTRACTION_PROMPT,
        config.analysis.extraction_timeout(),
        "extraction",
    )?))
}

impl ProviderAuth {
    /// The bearer credential provider `name` is called with: none, the key
    /// from the environment, or the current OAuth access token. A key
    /// variable that is unset or blank is an error, not an empty key. `None`
    /// for `auth = "aws"`, whose requests the AWS SDK signs itself.
    ///
    /// # Errors
    ///
    /// Returns a `Config` error for a missing key, and
    /// [`Error::AuthRequired`] when an OAuth provider has no login.
    pub async fn credential(&self, config: &Config, name: &ProviderName) -> Result<Option<String>> {
        match self {
            Self::None | Self::Aws { .. } => Ok(None),
            Self::ApiKey { env } => match std::env::var(env) {
                Ok(key) if !key.trim().is_empty() => Ok(Some(key)),
                Ok(_) | Err(_) => Err(Error::Config(format!(
                    "provider '{name}' reads its API key from {env}, which is not set"
                ))),
            },
            Self::Oauth(oauth) => {
                let manager = oauth::TokenManager::shared(config, name, oauth)?;
                let token = manager.access_token().await?;
                Ok(Some(token.expose_secret().to_owned()))
            }
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
    /// The models Ollama has in memory (`GET /api/ps`).
    async fn loaded(
        endpoint: &OllamaEndpoint,
    ) -> std::result::Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        use rig::http_client::HttpClientExt;

        let request = endpoint
            .request(http::Method::GET, "api/ps")
            .body(Vec::new())?;
        let response = endpoint.http.send::<_, Vec<u8>>(request).await?;
        let bytes: Vec<u8> = response.into_body().await?;
        Ok(serde_json::from_slice(&bytes)?)
    }

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

async fn build_openai_client(
    config: &Config,
    name: &ProviderName,
    provider: &ProviderConfig,
) -> Result<openai::OpenAI> {
    let key = provider
        .auth
        .credential(config, name)
        .await?
        .ok_or_else(|| {
            Error::Config(format!(
                "provider '{name}' (openai) requires auth = \"api-key\" or \"oauth\""
            ))
        })?;
    Ok(openai_client(
        provider.base_url.as_ref().map(BaseUrl::as_str),
        &key,
        LimitedHttp::for_provider(name, provider).with_headers(provider.header_map()?),
    ))
}

/// rig's `OpenAI` client for `key` at `base_url` (`OpenAI`'s own when
/// `None`), sending through `http`. It serves both Chat Completions and
/// Responses.
fn openai_client(base_url: Option<&str>, key: &str, http: LimitedHttp) -> openai::OpenAI {
    let mut settings = openai::OpenAIConfig::new(key);
    if let Some(base_url) = base_url {
        settings = settings.with_base_url(base_url);
    }
    settings.connect(http)
}

async fn build_anthropic_client(
    config: &Config,
    name: &ProviderName,
    provider: &ProviderConfig,
) -> Result<anthropic::Anthropic> {
    let key = provider
        .auth
        .credential(config, name)
        .await?
        .ok_or_else(|| {
            Error::Config(format!(
                "provider '{name}' (anthropic) requires auth = \"api-key\" or \"oauth\""
            ))
        })?;
    anthropic_client(name, provider, &key)
}

/// The header an Anthropic provider's credential goes in, decided once for
/// every request quack sends one on (completions and the doctor's probe).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AnthropicCredential {
    /// `x-api-key`, the scheme Anthropic's API keys use.
    ApiKey,
    /// `Authorization: Bearer`, where gateways and Anthropic's own OAuth
    /// look for an OAuth token.
    Bearer,
}

impl AnthropicCredential {
    /// The header `auth`'s credential goes in.
    pub(crate) const fn of(auth: &ProviderAuth) -> Self {
        match auth {
            ProviderAuth::Oauth(_) => Self::Bearer,
            ProviderAuth::None | ProviderAuth::ApiKey { .. } | ProviderAuth::Aws { .. } => {
                Self::ApiKey
            }
        }
    }
}

/// The Anthropic client for `provider` and its resolved credential, sent
/// in the header [`AnthropicCredential::of`] names.
fn anthropic_client(
    name: &ProviderName,
    provider: &ProviderConfig,
    key: &str,
) -> Result<anthropic::Anthropic> {
    let http = LimitedHttp::for_provider(name, provider).with_headers(provider.header_map()?);
    let http = match AnthropicCredential::of(&provider.auth) {
        AnthropicCredential::Bearer => http.with_oauth_bearer(key)?,
        AnthropicCredential::ApiKey => http,
    };
    let mut settings = anthropic::AnthropicConfig::new(key);
    if let Some(base_url) = &provider.base_url {
        settings = settings.with_base_url(base_url.as_str());
    }
    Ok(settings.connect(http))
}

/// One agent turn an interface asks for: the workspace's writer and
/// reader, the session, the write policy, the message, where the events
/// go, and the token that cancels it. `reader_db` is the workspace
/// handle's reader (`ReaderDb::open`), built once for the handle's whole
/// lifetime by whoever opened it, so starting a turn never waits on the
/// writer to acquire one.
pub struct TurnRequest<'a> {
    pub db: SharedDb,
    pub reader_db: ReaderDb,
    pub session_id: &'a SessionId,
    pub policy: WritePolicy,
    pub message: &'a str,
    pub sink: EventSink,
    pub cancel: CancellationToken,
}

impl TurnRequest<'_> {
    /// Run the turn with the configured chat and embedding models, replaying
    /// the session's history to the model and recording the turn when it
    /// completes.
    ///
    /// This is the single dispatch point over provider types; interfaces call
    /// it rather than matching on `provider_type` themselves.
    ///
    /// # Errors
    ///
    /// Returns an error if no chat model is configured, a provider cannot be
    /// built, the session does not exist, or the agent turn fails.
    pub async fn run(self, config: &Config) -> Result<AgentResponse> {
        let Self {
            db,
            reader_db,
            session_id,
            policy,
            message,
            sink,
            cancel,
        } = self;
        let asked = Instant::now();
        let asked_at = Timestamp::now();
        // A failure before the turn begins is the turn's failure too, so an
        // interface that only reads the events still sees why.
        let StartedTurn {
            chat,
            embedding_model,
            prompt,
            history,
        } = match start_turn(config, &db, session_id, policy).await {
            Ok(started) => started,
            Err(e) => {
                drop(sink.send(AgentEvent::Failed(TurnFailure::from(&e))));
                return Err(e);
            }
        };

        tracing::info!(chat_model = %chat, session = %session_id, prior_messages = history.len(), "starting agent turn");

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
        let analysis = Analysis {
            db: Arc::clone(&db),
            reader_db,
            embedder: embedding_model,
            config: &config.analysis,
            retrieval_config: &config.retrieval,
            graph_options: config.graph.options(),
            write_policy: policy,
            prompt,
            history,
            message,
            asked,
        };
        let turn = Priority::Interactive.scope(dispatch(config, chat, analysis, inner_sink));
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
                duration_ms: Some(u64::try_from(asked.elapsed().as_millis()).unwrap_or(u64::MAX)),
                ..AgentResponse::default()
            };
            tracing::info!(session = %session_id, "agent turn cancelled");
            drop(sink.send(AgentEvent::TurnComplete(response.clone())));
            response
        };

        let (session, text, recorded) =
            (session_id.to_owned(), message.to_owned(), response.clone());
        db.run(move |guard| sessions::record_turn(guard, &session, &text, asked_at, &recorded))
            .await?;
        Ok(response)
    }
}

/// What a turn needs before the model is called.
struct StartedTurn<'c> {
    chat: ModelRef<'c>,
    /// `None` when no embedding model is configured.
    embedding_model: Option<Embeddings>,
    prompt: PromptOptions,
    /// The session's earlier messages, replayed to the model.
    history: Vec<Message>,
}

async fn start_turn<'c>(
    config: &'c Config,
    db: &SharedDb,
    session_id: &SessionId,
    policy: WritePolicy,
) -> Result<StartedTurn<'c>> {
    let chat = config.chat_model_ref()?;
    // Without an embedding provider the agent still runs: document search
    // is keyword-only and graph entry is exact (issue #58).
    let embedding_model = Embeddings::from_config(config).await?;
    // On the blocking pool, in the writer's interactive line: an async
    // worker never waits on the connection.
    let session_id = session_id.to_owned();
    let pinned_token_budget = config.retrieval.pinned_token_budget;
    let context_max_tokens = config.context.max_tokens;
    let ollama_context_cap = (chat.provider.provider_type == ProviderType::Ollama)
        .then_some(config.analysis.max_context_tokens);
    let history_budget = config.analysis.history_token_budget;
    let (prompt, history) = db
        .run(move |guard| {
            let session = sessions::get_session(guard, &session_id)?
                .ok_or_else(|| Record::Session.missing(session_id.as_str()))?;
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
        .await?;
    Ok(StartedTurn {
        chat,
        embedding_model,
        prompt,
        history,
    })
}

/// What a cancelled turn's recorded answer ends with.
pub const CANCELLED_NOTE: &str = "(Cancelled by the user before the answer was complete.)";

/// Run `analysis` on the chat model's provider.
async fn dispatch(
    config: &Config,
    chat: ModelRef<'_>,
    analysis: Analysis<'_, EmbedModel>,
    sink: EventSink,
) -> Result<AgentResponse> {
    let client = ChatClient::build(config, &chat).await?;
    let (wire, settings) = (client.wire(), config.model_settings(chat));
    sampling::check_tool_calls(chat.model, wire, settings.effort)?;
    if let ChatClient::Ollama(endpoint) = &client {
        // A first request after idle loads the model, which took 5
        // seconds for a 12 GB model measured live and shows the user
        // nothing meanwhile, so the turn says so first. Any failure to
        // ask counts as loaded: the chat call that follows reports the
        // real error.
        let resident = match OllamaRunningModels::loaded(endpoint).await {
            Ok(running) => running.holds(chat.model),
            Err(e) => {
                tracing::debug!(error = %e, "could not list Ollama's loaded models");
                true
            }
        };
        if !resident {
            drop(sink.send(AgentEvent::Status(format!(
                "loading {}, then thinking; Ollama loads a model on its first request and keeps it \
                 for {OLLAMA_KEEP_ALIVE}",
                chat.model
            ))));
        }
    }
    let model = client.chat_model(chat.model, settings.effort, settings.temperature)?;
    Box::pin(analysis.run(model, sink)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BedrockConfig;
    use crate::embedding::Dimension;
    use crate::storage::workspace::WorkspaceDb;
    use crate::storage::writer::Writer;

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

    /// One request's head and body, read off a loopback socket that
    /// answers 400 (the call itself is not the point).
    async fn capture_one() -> (String, tokio::task::JoinHandle<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let addr = listener
            .local_addr()
            .unwrap_or_else(|e| fail(&e.to_string()));
        let seen = tokio::spawn(async move {
            let Ok((mut socket, _)) = listener.accept().await else {
                return String::new();
            };
            let mut read = Vec::new();
            let mut buf = [0_u8; 8192];
            // Until the body the Content-Length names has arrived.
            while let Ok(n) = socket.read(&mut buf).await {
                if n == 0 {
                    break;
                }
                read.extend_from_slice(buf.get(..n).unwrap_or_default());
                let text = String::from_utf8_lossy(&read).to_string();
                if let Some((head, body)) = text.split_once("\r\n\r\n") {
                    let length = head
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap_or_default())
                        })
                        .unwrap_or_default();
                    if body.len() >= length {
                        break;
                    }
                }
            }
            drop(
                socket
                    .write_all(
                        b"HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
                    )
                    .await,
            );
            String::from_utf8_lossy(&read).to_string()
        });
        (format!("http://{addr}"), seen)
    }

    /// The whole chat path for Bedrock's OpenAI-compatible APIs, up to the
    /// wire: the endpoint's path, a `SigV4` header for its service in place
    /// of rig's bearer, and, for Responses, `store: false`.
    #[tokio::test]
    async fn bedrock_openai_apis_send_signed_requests_to_the_endpoint_path() {
        for (provider_type, api, path, service) in [
            (
                ProviderType::BedrockMantle,
                BedrockApi::Responses,
                "POST /v1/responses ",
                "/bedrock-mantle/aws4_request",
            ),
            (
                ProviderType::BedrockMantle,
                BedrockApi::ChatCompletions,
                "POST /v1/chat/completions ",
                "/bedrock-mantle/aws4_request",
            ),
            (
                ProviderType::Bedrock,
                BedrockApi::Responses,
                "POST /openai/v1/responses ",
                "/us-west-2/bedrock/aws4_request",
            ),
        ] {
            let (root, seen) = capture_one().await;
            let bedrock = BedrockConfig { api, region: None };
            let provider = ProviderConfig {
                bedrock: Some(bedrock.clone()),
                headers: Some(std::collections::BTreeMap::from([(
                    String::from("X-Gateway-Team"),
                    String::from("quack"),
                )])),
                ..ProviderConfig::new(provider_type)
            };
            let Some(endpoint) = provider_type.bedrock_endpoint() else {
                fail("a Bedrock type")
            };
            let name: ProviderName = "wire-test"
                .parse()
                .unwrap_or_else(|e: Error| fail(&e.to_string()));
            let session = bedrock::Session::for_test(endpoint, bedrock, &root, "us-west-2");
            let client = ChatClient::bedrock(&session, &name, &provider)
                .unwrap_or_else(|e| fail(&e.to_string()));
            let answer = client
                .one_shot(
                    "openai.gpt-oss-120b",
                    ModelSettings::default(),
                    "Answer.",
                    Duration::from_secs(10),
                    "wire test",
                )
                .unwrap_or_else(|e| fail(&e.to_string()))
                .answer("hello")
                .await;
            assert!(answer.is_err(), "the server answers 400");
            let request = seen.await.unwrap_or_else(|e| fail(&e.to_string()));
            assert!(request.starts_with(path), "{api}: {request}");
            let lower = request.to_ascii_lowercase();
            assert!(
                lower.contains("authorization: aws4-hmac-sha256 credential=akidexample/"),
                "{request}"
            );
            assert!(request.contains(service), "{request}");
            assert!(!lower.contains("bearer"), "{request}");
            assert!(lower.contains("x-gateway-team: quack"), "{request}");
            assert!(request.contains("openai.gpt-oss-120b"), "{request}");
            if api == BedrockApi::Responses {
                assert!(request.contains(r#""store":false"#), "{request}");
            }
        }
    }

    /// A provider's `headers` reach the wire on every chat client, beside
    /// the credential.
    #[tokio::test]
    async fn provider_headers_are_sent_beside_the_credential() {
        // Cargo sets this variable for every test run.
        let keyed = "auth = \"api-key\"\napi_key_env = \"CARGO_PKG_NAME\"\n";
        // Anthropic calls need a Claude model, which gets `max_tokens`.
        for (provider, auth, model, path) in [
            ("type = \"ollama\"\n", "", "m", "POST /api/chat "),
            (
                "type = \"openai\"\napi = \"chat-completions\"\n",
                keyed,
                "m",
                "POST /chat/completions ",
            ),
            (
                "type = \"openai\"\napi = \"responses\"\n",
                keyed,
                "m",
                "POST /responses ",
            ),
            (
                "type = \"anthropic\"\n",
                keyed,
                "claude-sonnet-5",
                "POST /v1/messages ",
            ),
        ] {
            let (root, seen) = capture_one().await;
            let config = parse(&format!(
                "[general]\nchat_model = \"p/{model}\"\n[providers.p]\n{provider}{auth}\
                 base_url = \"{root}\"\nheaders = {{ \"X-Gateway-Team\" = \"quack\" }}\n"
            ));
            let chat = config
                .chat_model_ref()
                .unwrap_or_else(|e| fail(&e.to_string()));
            let client = ChatClient::build(&config, &chat)
                .await
                .unwrap_or_else(|e| fail(&e.to_string()));
            let answer = client
                .one_shot(
                    model,
                    ModelSettings::default(),
                    "Answer.",
                    Duration::from_secs(10),
                    "wire test",
                )
                .unwrap_or_else(|e| fail(&e.to_string()))
                .answer("hello")
                .await;
            assert!(answer.is_err(), "the server answers 400");
            let request = seen.await.unwrap_or_else(|e| fail(&e.to_string()));
            let lower = request.to_ascii_lowercase();
            assert!(request.starts_with(path), "{provider}: {request}");
            assert!(lower.contains("x-gateway-team: quack"), "{request}");
            assert_eq!(lower.contains("quack-core"), !auth.is_empty(), "{request}");
        }
    }

    /// With `auth = "oauth"`, the request carries `Authorization: Bearer
    /// <token>` and no `x-api-key` header; with `auth = "api-key"`, it
    /// carries `x-api-key` and no `Authorization` header.
    #[tokio::test]
    async fn anthropic_sends_an_oauth_token_as_a_bearer() {
        let oauth = "auth = \"oauth\"\n[providers.p.oauth]\n\
                     issuer_url = \"http://127.0.0.1:9\"\nclient_id = \"c\"\n";
        let keyed = "auth = \"api-key\"\napi_key_env = \"CARGO_PKG_NAME\"\n";
        for (auth, bearer) in [(oauth, true), (keyed, false)] {
            let (root, seen) = capture_one().await;
            let config = parse(&format!(
                "[general]\nchat_model = \"p/claude-sonnet-5\"\n[providers.p]\n\
                 type = \"anthropic\"\nbase_url = \"{root}\"\n{auth}"
            ));
            let chat = config
                .chat_model_ref()
                .unwrap_or_else(|e| fail(&e.to_string()));
            let client = anthropic_client(chat.provider_name, chat.provider, "tok-1")
                .unwrap_or_else(|e| fail(&e.to_string()));
            let answer = ChatClient::Anthropic(client)
                .one_shot(
                    "claude-sonnet-5",
                    ModelSettings::default(),
                    "Answer.",
                    Duration::from_secs(10),
                    "wire test",
                )
                .unwrap_or_else(|e| fail(&e.to_string()))
                .answer("hello")
                .await;
            assert!(answer.is_err(), "the server answers 400");
            let request = seen.await.unwrap_or_else(|e| fail(&e.to_string()));
            let lower = request.to_ascii_lowercase();
            assert!(request.starts_with("POST /v1/messages "), "{request}");
            assert_eq!(lower.contains("authorization: "), bearer, "{request}");
            assert_eq!(
                lower.contains("authorization: bearer tok-1\r\n"),
                bearer,
                "{request}"
            );
            assert_eq!(lower.contains("x-api-key"), !bearer, "{request}");
            assert!(lower.contains("anthropic-version: "), "{request}");
        }
    }

    #[test]
    fn responses_requests_ask_bedrock_to_store_nothing() {
        let request = |params: Option<serde_json::Value>| rig::completion::CompletionRequest {
            model: None,
            chat_history: Vec::new(),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: params,
            output_schema: None,
            record_telemetry_content: false,
        };
        let sent = Unstored::<()>::request(request(None)).additional_params;
        assert_eq!(sent, Some(serde_json::json!({ "store": false })));
        // Whatever else was asked is kept, and store is forced off.
        let sent = Unstored::<()>::request(request(Some(serde_json::json!({
            "store": true,
            "reasoning": { "effort": "low" }
        }))))
        .additional_params;
        assert_eq!(
            sent,
            Some(serde_json::json!({ "store": false, "reasoning": { "effort": "low" } }))
        );
    }

    #[test]
    fn display_name_reports_model_ref_or_placeholder() {
        let config =
            parse("[general]\nchat_model = \"o/llama3\"\n[providers.o]\ntype = \"ollama\"\n");
        assert_eq!(config.chat_model_label(), "o/llama3");
        assert_eq!(Config::default().chat_model_label(), "no chat model");
    }

    #[tokio::test]
    async fn optional_embedding_model_is_none_when_unset() {
        assert!(matches!(
            Embeddings::from_config(&Config::default()).await,
            Ok(None)
        ));
    }

    #[tokio::test]
    async fn required_embedding_model_errors_when_unset() {
        let err = Embeddings::require(&Config::default()).await.err();
        assert!(err.is_some_and(|e| e.to_string().contains("[embedding].model")));
    }

    #[tokio::test]
    async fn api_key_mode_requires_the_env_var_to_be_set() {
        let config = parse(
            "[embedding]\nmodel = \"o/e\"\ndimension = 4\n[providers.o]\ntype = \"openai\"\nauth = \"api-key\"\napi_key_env = \"QUACK_TEST_KEY_THAT_IS_UNSET\"\n",
        );
        let err = Embeddings::require(&config).await.err();
        assert!(err.is_some_and(|e| e.to_string().contains("QUACK_TEST_KEY_THAT_IS_UNSET")));
    }

    #[tokio::test]
    async fn ollama_embedding_model_builds_without_a_key() {
        let config = parse(
            "[embedding]\nmodel = \"o/nomic\"\ndimension = 4\n[providers.o]\ntype = \"ollama\"\n",
        );
        let model = Embeddings::require(&config).await;
        assert!(model.is_ok_and(|m| m.profile().dimension == Dimension::new(4)));
    }

    #[tokio::test]
    async fn ollama_embed_requests_carry_a_bounded_window_and_keep_alive() {
        let config = parse(
            "[embedding]\nmodel = \"o/nomic\"\ndimension = 4\n[providers.o]\ntype = \"ollama\"\n[ingestion]\nchunk_size_tokens = 3000\n",
        );
        let model = Embeddings::require(&config).await;
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
            "[embedding]\nmodel = \"az/emb\"\ndimension = 4\n[general]\nchat_model = \"az/gpt\"\n[providers.az]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.az.oauth]\nissuer_url = \"http://127.0.0.1:9\"\nclient_id = \"c\"\n",
        );
        config.general.data_dir = dir.path().to_path_buf();
        let err = Embeddings::require(&config).await.err();
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
            "[embedding]\nmodel = \"o/e\"\ndimension = 4\n[general]\nchat_model = \"o/m\"\n[providers.o]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\n",
        );
        config.general.data_dir = dir.path().to_path_buf();
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        let session = sessions::create_session(&db, "o/m", sessions::ChatMode::Chat, None)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
        let reader_db = ReaderDb::open(&db, config.analysis.reader_pool_size).await;
        let (sink, mut events) = events::channel();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let response = TurnRequest {
            db: Arc::clone(&db),
            reader_db,
            session_id: &session.id,
            policy: WritePolicy::Deny,
            message: "how many storms?",
            sink,
            cancel,
        }
        .run(&config)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(response.cancelled);
        assert_eq!(response.content, CANCELLED_NOTE);
        assert!(
            response.duration_ms.is_some(),
            "a cancelled turn is timed too"
        );
        let last = events.recv().await;
        assert!(
            matches!(&last, Some(AgentEvent::TurnComplete(r)) if r.cancelled && r.duration_ms.is_some()),
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
                .is_some_and(|m| m.content.contains("Cancelled by the user")
                    && m.assistant().and_then(|a| a.duration_ms).is_some())
        );
    }
}
