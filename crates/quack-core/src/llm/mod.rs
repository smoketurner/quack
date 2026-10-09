//! LLM provider construction on top of rig.
//!
//! Everything that turns a `[providers.<name>]` config entry plus a
//! `PROVIDER/MODEL` reference into a rig model lives here, so the interfaces
//! never build providers themselves.

pub mod acting;
pub mod bedrock;
pub mod chat_model;
pub mod egress;
pub mod memory;
pub mod oauth;
mod slot;
pub mod titles;
pub mod vision;

use jiff::{Timestamp, Zoned};
use rig::agent::OutputMode;
use rig::embeddings::Embedding;
use rig::providers::{anthropic, ollama, openai};
use rig::streaming::{Item, StreamEvent};
use rig::{DynModel, ProviderError, operation};
use schemars::{Schema, schema_for};
use secrecy::ExposeSecret;
use serde::de::DeserializeOwned;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rig::prelude::*;

use crate::analysis::agent::{AgentResponse, Analysis, CANCELLED_NOTE, Cutoff};
use crate::analysis::events::{AgentEvent, EventSink, TurnFailure};
use crate::analysis::policy::WritePolicy;
use crate::analysis::rerank::{RERANK_PROMPT, RERANK_TIMEOUT, RerankAnswer, Reranker};
use crate::analysis::search::DocumentScope;
use crate::analysis::text_to_sql::{PromptOptions, Window};
use crate::analysis::tools::Rerank;
use crate::analysis::tools::{ReaderDb, SharedDb};
use crate::config::{
    BaseUrl, BedrockApi, BedrockEndpoint, Config, Effort, ModelRef, ModelSettings, ProviderAuth,
    ProviderConfig, ProviderName, ProviderType, RerankMode, config_file_path,
};
use crate::embedding::{Embedder, EmbeddingModel, Profile};
use crate::error::{Error, Result};
use crate::extraction::{Extract, ExtractFuture};
use crate::graph::extract::{Extraction, ExtractionAnswer};
use crate::ids::SessionId;
use crate::ontology::Ontology;
use crate::ontology::documents::{self, OpenExtraction};
use crate::priority::Priority;
use crate::storage::control::ResourceKind;
use crate::storage::{context, sessions};
pub use chat_model::ChatModel;
use chat_model::{ChatSettings, Wire};
use egress::Egress;
pub use tokio_util::sync::CancellationToken;
use vision::ImageReader;

pub mod limit;

pub use limit::LimitedHttp;

impl ModelRef<'_> {
    /// Whether the current work may send to this model, asked before its
    /// client is built so a refusal is typed and comes before any request;
    /// the provider's gates ask again for every request.
    fn permitted(&self) -> Result<()> {
        Egress::permit(
            self.provider_name,
            self.provider.provider_type,
            Some(self.model),
        )
    }
}

/// A dedicated rerank model with its wire and transport erased, sending
/// through [`LimitedHttp`] like every model quack builds.
#[derive(Clone)]
pub struct RerankModel(DynModel<operation::Rerank>);

impl RerankModel {
    /// The rerank model `[retrieval].rerank = "reranker"` names, or `None`
    /// in any other mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the setting is invalid or the provider's
    /// credential cannot be resolved.
    pub async fn from_config(config: &Config) -> Result<Option<Self>> {
        let Some(model) = config.rerank_model_ref()? else {
            return Ok(None);
        };
        let key = model
            .provider
            .auth
            .credential(config, model.provider_name)
            .await?;
        Self::with_key(model, key.as_deref()).map(Some)
    }

    /// `model` with `key`, already resolved, on [`ChatClient::rerank_server`].
    pub(crate) fn with_key(model: ModelRef<'_>, key: Option<&str>) -> Result<Self> {
        model.permitted()?;
        Ok(Self(
            ChatClient::rerank_server(model.provider_name, model.provider, key)?
                .rerank(model.model)
                .erase(),
        ))
    }

    /// Score `request`'s documents against its query.
    ///
    /// # Errors
    ///
    /// Returns the provider's error when the call fails.
    #[expect(
        clippy::result_large_err,
        reason = "rig's ProviderError, which the call returns; it keeps the failed response"
    )]
    pub async fn rank(
        &self,
        request: operation::RerankRequest,
    ) -> std::result::Result<rig::rerank::RerankResponse, ProviderError> {
        #[expect(
            clippy::disallowed_methods,
            reason = "a rerank call, not an embedding; the lint guards embedding prefixes"
        )]
        self.0.call(request).await
    }
}

/// One of rig's embedding models with its wire and transport erased.
type RigEmbeddingModel = DynModel<operation::Embedding>;

/// How long an embedding request asks Ollama to keep the embedding model
/// loaded. Ollama's own default is 5 minutes (`OLLAMA_KEEP_ALIVE`), which a
/// gap between turns can exceed, and a reload costs several seconds.
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
        Self::new(
            name,
            provider,
            provider.auth.credential(config, name).await?.as_deref(),
        )
    }

    /// The server with `key`, already resolved, as its bearer.
    fn new(name: &ProviderName, provider: &ProviderConfig, key: Option<&str>) -> Result<Self> {
        let mut settings = ollama::OllamaConfig::new();
        if let Some(key) = key {
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
    /// never below `OLLAMA_EMBED_MIN_CTX`.
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
        model.permitted()?;
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
pub(crate) enum ChatClient {
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
        chat.permitted()?;
        Self::for_provider(config, chat.provider_name, chat.provider).await
    }

    /// The client for `provider` with a placeholder in place of its
    /// credential, to check what its requests would carry without sending
    /// any. Bedrock still loads its AWS session, which names the endpoint.
    pub(crate) async fn without_credential(
        name: &ProviderName,
        provider: &ProviderConfig,
    ) -> Result<Self> {
        match provider.provider_type {
            ProviderType::Bedrock | ProviderType::BedrockMantle => {
                Self::bedrock(&*bedrock::session(name, provider).await?, name, provider)
            }
            ProviderType::Ollama | ProviderType::Openai | ProviderType::Anthropic => {
                Self::connect(name, provider, Some(UNSENT_KEY))
            }
        }
    }

    /// The client for `provider`, its credential resolved the way a turn
    /// resolves it.
    pub(crate) async fn for_provider(
        config: &Config,
        name: &ProviderName,
        provider: &ProviderConfig,
    ) -> Result<Self> {
        match provider.provider_type {
            ProviderType::Bedrock | ProviderType::BedrockMantle => {
                Self::bedrock(&*bedrock::session(name, provider).await?, name, provider)
            }
            ProviderType::Ollama | ProviderType::Openai | ProviderType::Anthropic => {
                let client = Self::connect(
                    name,
                    provider,
                    provider.auth.credential(config, name).await?.as_deref(),
                )?;
                Ok(client)
            }
        }
    }

    /// A rerank server's client, with `key` already resolved: rig's
    /// OpenAI-compatible client on llama.cpp's dialect, whose rerank path is
    /// `/rerank` under the base URL, where vLLM, llama.cpp, and Text
    /// Embeddings Inference all serve it. The bearer goes only when there
    /// is a key, since llama.cpp refuses one it was not started with. Sends
    /// through the provider's [`LimitedHttp`] and headers.
    pub(crate) fn rerank_server(
        name: &ProviderName,
        provider: &ProviderConfig,
        key: Option<&str>,
    ) -> Result<openai::OpenAI> {
        let mut settings =
            openai::OpenAIConfig::with_key(&openai::wire::LLAMACPP, key.unwrap_or_default());
        if let Some(base_url) = &provider.base_url {
            settings = settings.with_base_url(base_url.as_str());
        }
        let http = LimitedHttp::for_provider(name, provider).with_headers(provider.header_map()?);
        Ok(settings.connect(http))
    }

    /// The client for a provider reached over plain HTTPS, with `key`
    /// already resolved: the provider's limited client, headers, and the
    /// credential where a turn sends it. Bedrock signs with AWS
    /// credentials instead; build it with [`Self::for_provider`].
    pub(crate) fn connect(
        name: &ProviderName,
        provider: &ProviderConfig,
        key: Option<&str>,
    ) -> Result<Self> {
        let required = |kind: &str| {
            key.ok_or_else(|| {
                Error::Config(format!(
                    "provider '{name}' ({kind}) requires auth = \"api-key\" or \"oauth\""
                ))
            })
        };
        let http = || -> Result<LimitedHttp> {
            Ok(LimitedHttp::for_provider(name, provider).with_headers(provider.header_map()?))
        };
        let base_url = provider.base_url.as_ref().map(BaseUrl::as_str);
        Ok(match provider.provider_type {
            ProviderType::Ollama => Self::Ollama(OllamaEndpoint::new(name, provider, key)?),
            ProviderType::Openai if provider.openai_chat_api() == BedrockApi::Responses => {
                Self::Responses(openai_client(base_url, required("openai")?, http()?))
            }
            ProviderType::Openai => {
                Self::OpenAi(openai_client(base_url, required("openai")?, http()?))
            }
            ProviderType::Anthropic => {
                Self::Anthropic(anthropic_client(name, provider, required("anthropic")?)?)
            }
            ProviderType::Bedrock | ProviderType::BedrockMantle => {
                return Err(Error::Config(format!(
                    "provider '{name}' signs with AWS credentials, not a key"
                )));
            }
        })
    }

    /// The models the provider lists, through the same client, limit,
    /// headers, and credential a turn uses. Bedrock's Converse API lists
    /// none.
    #[expect(
        clippy::result_large_err,
        reason = "rig's ProviderError, which every listing returns; it keeps the failed response"
    )]
    pub(crate) async fn models(&self) -> std::result::Result<ProviderModels, ProviderError> {
        let listed = match self {
            Self::Ollama(endpoint) => endpoint.client().list_models().await?,
            Self::OpenAi(client) | Self::Responses(client) => client.list_models().await?,
            Self::Anthropic(client) => client.list_models().await?,
            Self::Bedrock(_) => {
                return Err(ProviderError::Provider(String::from(
                    "Bedrock's Converse API lists no models",
                )));
            }
        };
        Ok(ProviderModels::from(listed))
    }

    /// The client for a Bedrock provider's API on its endpoint.
    pub(crate) fn bedrock(
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

    /// The chat model `model` at `effort`, with the settings its API takes.
    /// Every HTTP one sends through [`LimitedHttp`], so each provider's
    /// `max_concurrent_requests` bounds the model requests in flight.
    pub(crate) fn chat_model(
        &self,
        model: &str,
        effort: Option<Effort>,
        temperature: Option<bool>,
    ) -> Result<ChatModel> {
        let settings = ChatSettings::new(model, self.wire(), effort, temperature);
        match self {
            Self::Ollama(endpoint) => {
                ChatModel::new(endpoint.client().native_completion(model), model, settings)
            }
            Self::OpenAi(client) => ChatModel::new(client.chat(model), model, settings),
            Self::Anthropic(client) => ChatModel::new(client.completion(model), model, settings),
            Self::Bedrock(client) => ChatModel::new(client.completion(model), model, settings),
            Self::Responses(client) => ChatModel::new(client.responses(model), model, settings),
        }
    }

    /// A background call to `model` whose answer `schema` shapes.
    fn schema_call<A>(
        &self,
        model: &str,
        settings: ModelSettings,
        task: Task<'_>,
        schema: Schema,
    ) -> Result<SchemaCall<A>> {
        Ok(SchemaCall::new(
            self.chat_model(model, settings.background_effort, settings.temperature)?,
            task,
            schema,
        ))
    }

    /// A background call a chat turn goes without when it cannot be built:
    /// a `background_effort` the model refuses must not cost the answer.
    fn optional_schema_call<A>(
        &self,
        model: &str,
        settings: ModelSettings,
        task: Task<'_>,
        schema: Schema,
    ) -> Option<SchemaCall<A>> {
        let label = task.label;
        self.schema_call(model, settings, task, schema)
            .inspect_err(|e| tracing::warn!(error = %e, "the turn runs without its {label} call"))
            .ok()
    }

    /// The chat model as a listwise reranker, at `background_effort` like
    /// every other background call; `None` when it cannot be built, and
    /// the search keeps its fused order.
    fn rerank_call(
        &self,
        model: &str,
        settings: ModelSettings,
    ) -> Option<SchemaCall<RerankAnswer>> {
        self.optional_schema_call::<RerankAnswer>(
            model,
            settings,
            Task {
                preamble: RERANK_PROMPT,
                timeout: RERANK_TIMEOUT,
                label: "rerank",
            },
            schema_for!(RerankAnswer),
        )
    }
}

impl Rerank {
    /// The reranker `[retrieval].rerank` names, for a search a person runs
    /// outside a turn; `None` when it names none or one cannot be built.
    ///
    /// # Errors
    ///
    /// A model the configuration names that does not resolve, or a
    /// provider that cannot be built.
    pub async fn from_config(config: &Config) -> Result<Option<Self>> {
        let reranker = match config.retrieval.rerank {
            RerankMode::None => return Ok(None),
            RerankMode::Reranker => match RerankModel::from_config(config).await? {
                Some(model) => Arc::new(Reranker::scored(model)),
                None => return Ok(None),
            },
            RerankMode::Model => {
                let chat = config.chat_model_ref()?;
                let client = ChatClient::build(config, &chat).await?;
                match client.rerank_call(chat.model, config.model_settings(chat)) {
                    Some(call) => Arc::new(Reranker::model(call)),
                    None => return Ok(None),
                }
            }
        };
        Ok(Some(Self {
            reranker,
            candidates: config.retrieval.rerank_candidates,
        }))
    }
}

/// The key rig's `OpenAI` clients are built with when [`bedrock::Signer`]
/// replaces their `Authorization` header with a `SigV4` one.
const SIGNED_PLACEHOLDER_KEY: &str = "sigv4";

/// The key of a client that never sends a request.
const UNSENT_KEY: &str = "unsent";

/// What one background call is for: its preamble, how long it may take,
/// and the name errors and logs give it.
#[derive(Clone, Copy)]
pub struct Task<'a> {
    pub preamble: &'a str,
    pub timeout: Duration,
    pub label: &'static str,
}

/// One tool-less model call whose answer a JSON schema shapes: rig sends the
/// schema as the provider's structured output (Ollama's `format`, `OpenAI`'s
/// `response_format` or `text.format`, Anthropic's and Bedrock's output
/// configuration), and the whole answer parses as `A`. Streamed and
/// collected under the task's timeout: streaming is the path the chat agent
/// uses and the one Ollama answers reliably, and an answer the output limit
/// cut is refused by name.
pub struct SchemaCall<A> {
    call: PlainCall,
    answer: PhantomData<fn() -> A>,
}

/// One tool-less model call: the agent, how long it may take, and what it
/// is called in errors and logs.
pub(crate) struct PlainCall {
    agent: Agent,
    timeout: Duration,
    label: &'static str,
    window: Window,
}

impl<A> SchemaCall<A> {
    #[must_use]
    pub fn new(model: ChatModel, task: Task<'_>, schema: Schema) -> Self {
        let window = model.window();
        Self {
            call: PlainCall {
                window,
                agent: model
                    .agent(0.0)
                    .preamble(task.preamble)
                    .output_schema_raw(schema)
                    .output_mode(OutputMode::Native)
                    .build(),
                timeout: task.timeout,
                label: task.label,
            },
            answer: PhantomData,
        }
    }

    /// The model's answer to `text`.
    ///
    /// # Errors
    ///
    /// Returns an error when the call fails, its answer was cut off, it
    /// produces nothing within the timeout, or the answer does not fit the
    /// schema.
    pub async fn answer(&self, text: &str) -> Result<A>
    where
        A: DeserializeOwned,
    {
        let answer = self.call.text(text).await?;
        tracing::debug!(call = self.call.label, answer = %answer, "structured answer");
        serde_json::from_str(answer.trim()).map_err(|e| {
            Error::Llm(format!(
                "the {} answer does not fit its schema: {e}",
                self.call.label
            ))
        })
    }
}

impl PlainCall {
    /// A call to `model` with `task`'s preamble that answers in text.
    pub(crate) fn new(model: ChatModel, task: Task<'_>) -> Self {
        Self {
            window: model.window(),
            agent: model.agent(0.0).preamble(task.preamble).build(),
            timeout: task.timeout,
            label: task.label,
        }
    }

    /// The model's text answer to `message`, within the call's timeout.
    async fn text(&self, message: impl Into<Message>) -> Result<String> {
        use futures::StreamExt;
        let what = self.label;
        let collect = async {
            let mut stream = self.agent.prompt(message).stream();
            let mut answer = String::new();
            let mut final_text: Option<String> = None;
            let mut cutoff: Option<Cutoff> = None;
            while let Some(item) = stream.next().await {
                let item = match item {
                    Ok(item) => item,
                    Err(_) if let Some(cut) = cutoff => return Err(cut.refusal(what, self.window)),
                    Err(e) => return Err(Error::Llm(format!("{what} call failed: {e}"))),
                };
                match item {
                    MultiTurnStreamItem::CompletionCall(call) => {
                        cutoff = Cutoff::of(call.finish_reason.as_ref());
                    }
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
                        final_text = Some(r.output());
                    }
                    _ => {}
                }
            }
            // A cut-off answer is not one to parse: say so, rather than let
            // the caller fail on half a JSON document.
            if let Some(cut) = cutoff {
                return Err(cut.refusal(what, self.window));
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

/// An extractor whose answer is `A`, handed on as the `T` it converts to.
impl<A, T> Extract<T> for SchemaCall<A>
where
    A: DeserializeOwned + Into<T>,
{
    fn extract<'a>(&'a self, text: &'a str) -> ExtractFuture<'a, T> {
        Box::pin(async move { self.answer(text).await.map(Into::into) })
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
    let call: SchemaCall<ExtractionAnswer> = ChatClient::build(config, &chat).await?.schema_call(
        chat.model,
        config.model_settings(chat),
        Task {
            preamble: &ontology.extraction_prompt(),
            timeout: config.analysis.extraction_timeout(),
            label: "graph extraction",
        },
        ontology.extraction_schema(),
    )?;
    Ok(Box::new(call))
}

/// The configured chat model as an open extractor for ontology induction.
///
/// # Errors
///
/// Returns an error when no chat model is configured or the provider
/// cannot be built (a missing key, a needed login).
pub async fn chat_extractor(config: &Config) -> Result<Box<dyn Extract<OpenExtraction>>> {
    let chat = config.chat_model_ref()?;
    let call: SchemaCall<OpenExtraction> = ChatClient::build(config, &chat).await?.schema_call(
        chat.model,
        config.model_settings(chat),
        Task {
            preamble: documents::EXTRACTION_PROMPT,
            timeout: config.analysis.extraction_timeout(),
            label: "extraction",
        },
        schema_for!(OpenExtraction),
    )?;
    Ok(Box::new(call))
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
    /// Bytes on disk (`/api/tags`) or in memory (`/api/ps`).
    #[serde(default)]
    size: u64,
}

type OllamaCallError = Box<dyn std::error::Error + Send + Sync>;

impl OllamaEndpoint {
    /// `method` on `path` with a JSON `body`, the reply parsed as `T`.
    async fn call<T: DeserializeOwned>(
        &self,
        method: http::Method,
        path: &str,
        body: Vec<u8>,
    ) -> std::result::Result<T, OllamaCallError> {
        use rig::http_client::HttpClientExt;

        let request = self.request(method, path).body(body)?;
        let response = self.http.send::<_, Vec<u8>>(request).await?;
        let bytes: Vec<u8> = response.into_body().await?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}

impl OllamaRunningModels {
    /// The models Ollama has in memory (`GET /api/ps`).
    async fn loaded(endpoint: &OllamaEndpoint) -> std::result::Result<Self, OllamaCallError> {
        endpoint.call(http::Method::GET, "api/ps", Vec::new()).await
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

/// One thing `POST /api/show` says a model can do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OllamaCapability {
    Completion,
    Tools,
    Vision,
    Embedding,
    Thinking,
    #[serde(other)]
    Other,
}

#[derive(serde::Deserialize)]
struct OllamaShow {
    #[serde(default)]
    capabilities: Vec<OllamaCapability>,
}

/// A model an Ollama server has pulled, with what it can do. An Ollama too
/// old to report capabilities reports none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OllamaModel {
    pub name: String,
    /// Bytes on disk.
    pub size: u64,
    pub capabilities: Vec<OllamaCapability>,
}

impl OllamaModel {
    /// The models the Ollama provider `name` has pulled (`GET /api/tags`),
    /// each with its capabilities (`POST /api/show`), in the server's order.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is not a configured provider or the
    /// server does not answer.
    pub async fn list(config: &Config, name: &ProviderName) -> Result<Vec<Self>> {
        let provider = config
            .providers
            .get(name)
            .ok_or_else(|| Error::Config(format!("no provider named '{name}' is configured")))?;
        let endpoint = OllamaEndpoint::build(config, name, provider).await?;
        let failed = |e: OllamaCallError| Error::Llm(format!("ollama at '{name}': {e}"));
        let pulled: OllamaRunningModels = endpoint
            .call(http::Method::GET, "api/tags", Vec::new())
            .await
            .map_err(failed)?;
        let mut models = Vec::with_capacity(pulled.models.len());
        for pulled in pulled.models {
            let body = serde_json::to_vec(&serde_json::json!({ "model": pulled.name }))?;
            let shown: OllamaShow = endpoint
                .call(http::Method::POST, "api/show", body)
                .await
                .map_err(failed)?;
            models.push(Self {
                name: pulled.name,
                size: pulled.size,
                capabilities: shown.capabilities,
            });
        }
        Ok(models)
    }

    #[must_use]
    pub fn can(&self, capability: OllamaCapability) -> bool {
        self.capabilities.contains(&capability)
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

/// The models a provider lists, in its order.
#[derive(Debug, Clone, Default)]
pub struct ProviderModels(Vec<ProviderModel>);

/// One listed model: its id and what the provider reports of its limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderModel {
    pub id: String,
    /// The context window in tokens, when the listing reports one (an
    /// OpenAI-compatible server may; Ollama's and Anthropic's do not).
    pub context_length: Option<u32>,
    pub max_output_tokens: Option<u32>,
}

impl From<rig::model::ModelList> for ProviderModels {
    fn from(list: rig::model::ModelList) -> Self {
        Self(
            list.data
                .into_iter()
                .map(|info| ProviderModel {
                    id: info.id,
                    context_length: info.context_length,
                    max_output_tokens: info.max_output_tokens,
                })
                .collect(),
        )
    }
}

impl ProviderModels {
    /// How many suggestions [`Self::closest`] gives.
    pub const SUGGESTIONS: usize = 3;

    /// The models `name` lists, through the client a turn uses.
    ///
    /// # Errors
    ///
    /// Returns an error if the provider is not configured, its credential
    /// cannot be resolved, or the listing fails.
    pub async fn fetch(config: &Config, name: &ProviderName) -> Result<Self> {
        let provider = config
            .providers
            .get(name)
            .ok_or_else(|| Error::Config(format!("no provider named '{name}' is configured")))?;
        if provider.provider_type.bedrock_endpoint() == Some(BedrockEndpoint::Runtime) {
            return Err(Error::Config(String::from(
                "bedrock-runtime lists no models; a model's access shows on its first call",
            )));
        }
        ChatClient::for_provider(config, name, provider)
            .await?
            .models()
            .await
            .map_err(|e| Error::Llm(format!("provider '{name}' could not list its models: {e}")))
    }

    #[must_use]
    pub fn as_slice(&self) -> &[ProviderModel] {
        &self.0
    }

    /// The listed `model`. A bare name matches its `:latest` tag, which is
    /// how Ollama lists it.
    #[must_use]
    pub fn get(&self, model: &str) -> Option<&ProviderModel> {
        let tagged = format!("{model}:latest");
        self.0
            .iter()
            .find(|m| m.id == model || (!model.contains(':') && m.id == tagged))
    }

    /// The listed ids nearest `model` by Jaro-Winkler similarity, best
    /// first, at most [`Self::SUGGESTIONS`].
    #[must_use]
    pub fn closest(&self, model: &str) -> Vec<&str> {
        let mut scored: Vec<(f64, &str)> = self
            .0
            .iter()
            .map(|m| (strsim::jaro_winkler(model, &m.id), m.id.as_str()))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        scored
            .into_iter()
            .take(Self::SUGGESTIONS)
            .map(|(_, id)| id)
            .collect()
    }
}

/// Every configured provider's listed models, as `/model` shows them.
pub struct ModelCatalog(Vec<(ProviderName, std::result::Result<ProviderModels, String>)>);

impl ModelCatalog {
    /// How many ids one provider's line shows before it counts the rest.
    const SHOWN: usize = 40;

    /// Each provider's listing, in config order; a provider that cannot
    /// list says why.
    pub async fn fetch(config: &Config) -> Self {
        let mut listings = Vec::new();
        for name in config.providers.keys() {
            let listed = ProviderModels::fetch(config, name)
                .await
                .map_err(|e| e.to_string());
            listings.push((name.clone(), listed));
        }
        Self(listings)
    }
}

impl std::fmt::Display for ModelCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            return f.write_str("No providers are configured.");
        }
        f.write_str("Models each provider lists:")?;
        for (name, listed) in &self.0 {
            match listed {
                Ok(models) if models.as_slice().is_empty() => {
                    write!(f, "\n  {name}: none")?;
                }
                Ok(models) => {
                    let ids: Vec<&str> = models
                        .as_slice()
                        .iter()
                        .take(Self::SHOWN)
                        .map(|m| m.id.as_str())
                        .collect();
                    write!(f, "\n  {name}: {}", ids.join(", "))?;
                    let rest = models.as_slice().len().saturating_sub(Self::SHOWN);
                    if rest > 0 {
                        write!(f, ", and {rest} more")?;
                    }
                }
                Err(e) => write!(f, "\n  {name}: {e}")?,
            }
        }
        Ok(())
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
    /// The documents the person limits the question to, each by id, id
    /// prefix, file name, or title; empty for the whole workspace.
    pub documents: &'a [String],
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
            documents,
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
            rerank_model,
            prompt,
            history,
        } = match start_turn(config, &db, session_id, policy, documents).await {
            Ok(started) => started,
            Err(e) => {
                drop(sink.send(AgentEvent::Failed(TurnFailure::from(&e))));
                return Err(e);
            }
        };

        tracing::info!(chat_model = %chat, session = %session_id, prior_messages = history.len(), "starting agent turn");

        let prompt_scope = prompt.scope.clone();
        // Someone is watching this turn: its model calls, and the tools' calls
        // inside it, go ahead of background work at the provider (design 4.1).
        let analysis = Analysis {
            db: Arc::clone(&db),
            reader_db,
            embedder: embedding_model,
            rerank_model,
            config: &config.analysis,
            retrieval_config: &config.retrieval,
            graph_options: config.graph,
            write_policy: policy,
            prompt,
            history,
            message,
            asked,
            cancel: cancel.clone(),
        };
        let turn = Priority::Interactive.scope(dispatch(config, chat, analysis, sink.clone()));
        // The turn goes first: once the model is streaming, it sees the
        // cancellation itself and ends with what it has. Before that, while
        // the prompt is assembled, nothing has streamed and the turn is
        // dropped here.
        let outcome = tokio::select! {
            biased;
            outcome = turn => Some(outcome),
            () = cancel.cancelled() => None,
        };

        let response = if let Some(outcome) = outcome {
            outcome?
        } else {
            let response = AgentResponse {
                content: String::from(CANCELLED_NOTE),
                cancelled: true,
                duration_ms: Some(u64::try_from(asked.elapsed().as_millis()).unwrap_or(u64::MAX)),
                documents: prompt_scope,
                ..AgentResponse::default()
            };
            drop(sink.send(AgentEvent::TurnComplete(response.clone())));
            response
        };
        if response.cancelled {
            tracing::info!(session = %session_id, "agent turn cancelled");
        }

        let (session, text, recorded) =
            (session_id.to_owned(), message.to_owned(), response.clone());
        db.run(move |guard| sessions::record_turn(guard, &session, &text, asked_at, &recorded))
            .await?;
        if !response.cancelled {
            titles::SessionTitler::follow_turn(config, &db, session_id).await;
        }
        Ok(response)
    }
}

/// What a turn needs before the model is called.
struct StartedTurn<'c> {
    chat: ModelRef<'c>,
    /// `None` when no embedding model is configured.
    embedding_model: Option<Embeddings>,
    /// `None` unless `[retrieval].rerank = "reranker"`.
    rerank_model: Option<RerankModel>,
    prompt: PromptOptions,
    /// The session's earlier messages, replayed to the model.
    history: Vec<Message>,
}

async fn start_turn<'c>(
    config: &'c Config,
    db: &SharedDb,
    session_id: &SessionId,
    policy: WritePolicy,
    documents: &[String],
) -> Result<StartedTurn<'c>> {
    let chat = config.chat_model_ref()?;
    // Without an embedding provider the agent still runs: document search
    // is keyword-only and graph entry is exact (issue #58).
    let embedding_model = Embeddings::from_config(config).await?;
    let rerank_model = RerankModel::from_config(config).await?;
    // On the blocking pool, in the writer's interactive line: an async
    // worker never waits on the connection.
    let session_id = session_id.to_owned();
    let pinned_token_budget = config.retrieval.pinned_token_budget;
    let context_max_tokens = config.context.max_tokens;
    let window = Wire::of(chat.provider).window();
    let read = session_id.clone();
    let documents = documents.to_vec();
    let prompt = db
        .run(move |guard| {
            let session_id = read;
            let session = sessions::get_session(guard, &session_id)?
                .ok_or_else(|| ResourceKind::Session.missing(session_id.as_str()))?;
            let prompt = PromptOptions {
                scope: DocumentScope::resolve(guard, &documents)?,
                mode: session.mode,
                today: Zoned::now().date(),
                write_policy: policy,
                pinned_token_budget,
                context: context::combined(guard)?,
                context_max_tokens,
                window,
                question: None,
            };
            Ok(prompt)
        })
        .await?;
    let history = memory::History::from_config(config, Arc::clone(db))
        .await?
        .load(&session_id)
        .await?;
    Ok(StartedTurn {
        chat,
        embedding_model,
        rerank_model,
        prompt,
        history,
    })
}

/// Run `analysis` on the chat model's provider.
async fn dispatch(
    config: &Config,
    chat: ModelRef<'_>,
    analysis: Analysis<'_, EmbedModel>,
    sink: EventSink,
) -> Result<AgentResponse> {
    let client = ChatClient::build(config, &chat).await?;
    let (wire, settings) = (client.wire(), config.model_settings(chat));
    chat_model::check_tool_calls(chat.model, wire, settings.effort)?;
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
                "loading {}, then thinking; Ollama loads a model on its first request",
                chat.model
            ))));
        }
    }
    let model = client.chat_model(chat.model, settings.effort, settings.temperature)?;
    // Built here, where the turn's client is, at `background_effort` like
    // every other background call; without it the search keeps its fused order.
    let reranker = if config.retrieval.rerank == RerankMode::Model {
        client.rerank_call(chat.model, settings)
    } else {
        None
    };
    let images = ImageReader::for_turn(&client, chat.model, settings)?;
    Box::pin(analysis.run(model, reranker, images, sink)).await
}

#[cfg(test)]
mod tests;
