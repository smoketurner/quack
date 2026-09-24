use serde::Deserialize;
use std::borrow::{Borrow, Cow};
use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use crate::embedding::Dimension;
use crate::error::{Error, Result};
use crate::graph::GraphOptions;
use crate::ontology::documents::DocumentEvidenceOptions;
use crate::ontology::induction::TableEvidenceOptions;
use crate::text::Tokens;

pub mod bedrock;
pub mod inspect;

pub use bedrock::{AwsRegion, BedrockApi, BedrockConfig, BedrockEndpoint};

/// Directory holding `config.toml`.
pub const ENV_CONFIG_DIR: &str = "QUACK_CONFIG_DIR";
/// Overrides `[general].data_dir`.
pub const ENV_DATA_DIR: &str = "QUACK_DATA_DIR";
/// Overrides `[general].chat_model`.
pub const ENV_MODEL: &str = "QUACK_MODEL";
/// Overrides `[server].bind`.
pub const ENV_BIND: &str = "QUACK_BIND";

/// The whole `config.toml`. Unknown keys anywhere are an error so a typo can
/// never silently disable a setting.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub general: GeneralConfig,
    #[serde(default)]
    pub providers: BTreeMap<ProviderName, ProviderConfig>,
    pub ingestion: IngestionConfig,
    pub embedding: EmbeddingConfig,
    pub retrieval: RetrievalConfig,
    pub context: ContextConfig,
    pub analysis: AnalysisConfig,
    pub server: ServerConfig,
    pub ontology: OntologyConfig,
    pub graph: GraphConfig,
    pub import: ImportConfig,
    pub jobs: JobsConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeneralConfig {
    pub data_dir: PathBuf,
    pub default_workspace: String,
    /// `PROVIDER/MODEL` used for chat and tool calling. Override: `QUACK_MODEL`.
    pub chat_model: Option<ModelSpec>,
    /// `PROVIDER/MODEL` used for embeddings. Unset means documents are stored
    /// without vectors and `search_documents` is unavailable.
    pub embedding_model: Option<ModelSpec>,
}

/// A `PROVIDER/MODEL` reference, split once when the config is read.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
pub struct ModelSpec {
    provider: ProviderName,
    model: String,
}

impl ModelSpec {
    #[must_use]
    pub fn provider(&self) -> &ProviderName {
        &self.provider
    }

    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl TryFrom<String> for ModelSpec {
    type Error = Error;

    fn try_from(spec: String) -> Result<Self> {
        let Some((provider, model)) = spec.split_once('/') else {
            return Err(Error::Config(format!(
                "\"{spec}\" must be PROVIDER/MODEL, e.g. \"ollama/llama3.1:8b\""
            )));
        };
        if model.is_empty() {
            return Err(Error::Config(format!(
                "\"{spec}\" is missing the model after the slash"
            )));
        }
        Ok(Self {
            provider: provider.parse()?,
            model: model.to_owned(),
        })
    }
}

impl FromStr for ModelSpec {
    type Err = Error;

    fn from_str(spec: &str) -> Result<Self> {
        Self::try_from(spec.to_owned())
    }
}

impl std::fmt::Display for ModelSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.provider, self.model)
    }
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            default_workspace: String::from("default"),
            chat_model: None,
            embedding_model: None,
        }
    }
}

/// A `[providers.NAME]` key. It names the provider's OAuth cache and key
/// files, so it is checked when the config is read: ASCII letters, digits,
/// `_`, `-`, and `.`, not starting with `.`, at most 64 characters. Nothing
/// it names can leave the tokens directory.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(try_from = "String")]
pub struct ProviderName(String);

impl ProviderName {
    const MAX_LEN: usize = 64;

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ProviderName {
    type Error = Error;

    fn try_from(name: String) -> Result<Self> {
        let allowed = |c: char| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.');
        if name.is_empty()
            || name.len() > Self::MAX_LEN
            || name.starts_with('.')
            || !name.chars().all(allowed)
        {
            return Err(Error::Config(format!(
                "provider name '{name}' must be 1 to {} ASCII letters, digits, '_', '-', or '.', \
                 not starting with '.'",
                Self::MAX_LEN
            )));
        }
        Ok(Self(name))
    }
}

impl FromStr for ProviderName {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        Self::try_from(name.to_owned())
    }
}

impl PartialEq<str> for ProviderName {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for ProviderName {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl Borrow<str> for ProviderName {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ProviderName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&self.0)
    }
}

/// Which rig client a provider entry builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderType {
    Ollama,
    #[serde(alias = "openai-compat")]
    Openai,
    Anthropic,
    /// Amazon Bedrock, on its runtime or mantle endpoint, signed with
    /// credentials from the AWS SDK's default chain.
    Bedrock,
}

text_enum!(ProviderType, "provider type", {
    Ollama => "ollama",
    Openai => "openai",
    Anthropic => "anthropic",
    Bedrock => "bedrock",
});

impl ProviderType {
    /// Ollama's API when `base_url` is unset.
    pub const OLLAMA_BASE_URL: BaseUrl = BaseUrl(Cow::Borrowed("http://localhost:11434"));

    /// Where the provider's API is when `base_url` is unset: the same
    /// defaults rig's clients use. `None` for Bedrock, whose endpoint
    /// follows its region (`llm::bedrock`).
    #[must_use]
    pub const fn default_base_url(self) -> Option<BaseUrl> {
        match self {
            Self::Ollama => Some(Self::OLLAMA_BASE_URL),
            Self::Openai => Some(BaseUrl(Cow::Borrowed("https://api.openai.com/v1"))),
            Self::Anthropic => Some(BaseUrl(Cow::Borrowed("https://api.anthropic.com"))),
            Self::Bedrock => None,
        }
    }

    /// The `auth` a provider of this type has when the file names none.
    #[must_use]
    pub const fn default_auth(self) -> AuthMode {
        match self {
            Self::Bedrock => AuthMode::Aws,
            Self::Ollama | Self::Openai | Self::Anthropic => AuthMode::None,
        }
    }
}

/// How a provider endpoint is authenticated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AuthMode {
    /// No credentials (local Ollama, unauthenticated gateways).
    #[default]
    None,
    /// Static key from the environment variable named by `api_key_env`.
    ApiKey,
    /// OAuth 2.0 against an identity provider (design doc 10.2): the access
    /// token from `[providers.NAME.oauth]` is the bearer for the endpoint.
    Oauth,
    /// The AWS SDK's default credential chain (environment, `aws_profile`
    /// or `AWS_PROFILE`, IAM Identity Center (SSO), web identity, ECS and
    /// EC2 instance roles) signs each request. Bedrock only, and its
    /// default.
    Aws,
}

text_enum!(AuthMode, "auth mode", {
    None => "none",
    ApiKey => "api-key",
    Oauth => "oauth",
    Aws => "aws",
});

/// How a provider's token is obtained from its issuer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Grant {
    /// A person signs in through the browser (Authorization Code with PKCE).
    #[default]
    AuthorizationCode,
    /// A person signs in by entering a code on another device (headless
    /// hosts, SSH).
    DeviceCode,
    /// quack authenticates as itself with its client id and secret; nobody
    /// signs in, and a token is requested again whenever one runs out.
    ClientCredentials,
}

text_enum!(Grant, "grant", {
    AuthorizationCode => "authorization-code",
    DeviceCode => "device-code",
    ClientCredentials => "client-credentials",
});

/// `[providers.NAME.oauth]`: the issuer, the client, and the grant that
/// obtains the token quack sends to the provider's endpoint.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OAuthConfig {
    /// Issuer whose `/.well-known/openid-configuration` names the endpoints,
    /// e.g. `https://login.microsoftonline.com/{tenant}/v2.0`.
    pub issuer_url: String,
    pub client_id: String,
    /// Scopes requested at login, e.g.
    /// `["https://cognitiveservices.azure.com/.default", "offline_access"]`.
    /// Include `offline_access` where the issuer needs it to return a refresh
    /// token.
    #[serde(default)]
    pub scopes: Vec<String>,
    /// Loopback redirect for the browser flow.
    #[serde(default = "OAuthConfig::default_redirect_uri")]
    pub redirect_uri: String,
    #[serde(default)]
    pub grant: Grant,
    /// Environment variable holding the client secret: quack as a
    /// confidential client. `grant = "client-credentials"` requires it.
    pub client_secret_env: Option<String>,
}

impl OAuthConfig {
    /// The loopback redirect the browser flow listens on by default.
    pub const DEFAULT_REDIRECT_URI: &str = "http://127.0.0.1:19876/callback";

    fn default_redirect_uri() -> String {
        String::from(Self::DEFAULT_REDIRECT_URI)
    }
}

/// How a provider endpoint is authenticated, with what each way needs: a
/// combination the file cannot express correctly is refused when it is
/// read, so nothing downstream checks it again.
#[derive(Debug, Clone, Default)]
pub enum ProviderAuth {
    /// No credentials (local Ollama, unauthenticated gateways).
    #[default]
    None,
    /// A static key from the environment variable `env`.
    ApiKey { env: String },
    /// OAuth 2.0 against an identity provider (design doc 10.2).
    Oauth(OAuthConfig),
    /// The AWS SDK's default credential chain, from the named profile of
    /// the shared config files when `profile` is set (otherwise
    /// `AWS_PROFILE`, then `default`).
    Aws { profile: Option<String> },
}

impl ProviderAuth {
    /// The `auth` mode the file names.
    #[must_use]
    pub const fn mode(&self) -> AuthMode {
        match self {
            Self::None => AuthMode::None,
            Self::ApiKey { .. } => AuthMode::ApiKey,
            Self::Oauth(_) => AuthMode::Oauth,
            Self::Aws { .. } => AuthMode::Aws,
        }
    }

    /// The environment variable holding the key, for `auth = "api-key"`.
    #[must_use]
    pub fn api_key_env(&self) -> Option<&str> {
        match self {
            Self::ApiKey { env } => Some(env),
            Self::None | Self::Oauth(_) | Self::Aws { .. } => None,
        }
    }

    /// The `[providers.NAME.oauth]` table, for `auth = "oauth"`.
    #[must_use]
    pub const fn oauth(&self) -> Option<&OAuthConfig> {
        match self {
            Self::Oauth(oauth) => Some(oauth),
            Self::None | Self::ApiKey { .. } | Self::Aws { .. } => None,
        }
    }

    /// The AWS profile named by `aws_profile`, for `auth = "aws"`.
    #[must_use]
    pub fn aws_profile(&self) -> Option<&str> {
        match self {
            Self::Aws { profile } => profile.as_deref(),
            Self::None | Self::ApiKey { .. } | Self::Oauth(_) => None,
        }
    }
}

/// A provider's `base_url`: an absolute `http` or `https` URL, checked
/// when the config is read.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize)]
#[serde(try_from = "String")]
pub struct BaseUrl(Cow<'static, str>);

impl BaseUrl {
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Without a trailing `/`, to append a path to.
    #[must_use]
    pub fn trimmed(&self) -> &str {
        self.0.trim_end_matches('/')
    }

    /// The server root, without a trailing `/` or `/v1`: Ollama's native
    /// API sits there, beside its OpenAI-compatible `/v1`.
    #[must_use]
    pub fn root(&self) -> &str {
        self.trimmed().trim_end_matches("/v1")
    }

    /// Whether a credential sent here would cross the network unencrypted:
    /// plain HTTP to anything but this machine.
    #[must_use]
    pub fn sends_in_cleartext(&self) -> bool {
        let Ok(url) = reqwest::Url::parse(&self.0) else {
            return false;
        };
        if url.scheme() != "http" {
            return false;
        }
        let Some(host) = url.host_str() else {
            return false;
        };
        let host = host.trim_start_matches('[').trim_end_matches(']');
        match host.parse::<std::net::IpAddr>() {
            Ok(ip) => !ip.is_loopback(),
            Err(_) => host != "localhost",
        }
    }
}

impl TryFrom<String> for BaseUrl {
    type Error = Error;

    fn try_from(url: String) -> Result<Self> {
        match reqwest::Url::parse(&url) {
            Ok(parsed) if matches!(parsed.scheme(), "http" | "https") => Ok(Self(Cow::Owned(url))),
            Ok(_) | Err(_) => Err(Error::Config(format!(
                "base_url \"{url}\" must be an absolute http:// or https:// URL"
            ))),
        }
    }
}

impl std::fmt::Display for BaseUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Model requests a provider may have in flight at once: at least one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(transparent)]
pub struct RequestLimit(NonZeroU32);

impl RequestLimit {
    /// `None` for 0, which would admit no request.
    #[must_use]
    pub const fn new(limit: u32) -> Option<Self> {
        match NonZeroU32::new(limit) {
            Some(limit) => Some(Self(limit)),
            None => None,
        }
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

impl std::fmt::Display for RequestLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// One `[providers.NAME]` entry, as read and checked.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "RawProviderConfig")]
pub struct ProviderConfig {
    pub provider_type: ProviderType,
    pub auth: ProviderAuth,
    pub base_url: Option<BaseUrl>,
    /// The endpoint, API, and region of a `type = "bedrock"` provider; set
    /// for that type and no other.
    pub bedrock: Option<BedrockConfig>,
    /// Width of the vectors this provider's embedding models produce.
    pub embedding_dimension: Option<Dimension>,
    /// Model requests in flight to this provider at once, across the whole
    /// process; the rest wait their turn (design doc 4.1). Unset: 1 for
    /// Ollama, which serves one request per model unless
    /// `OLLAMA_NUM_PARALLEL` says otherwise, 8 for hosted APIs.
    pub max_concurrent_requests: Option<RequestLimit>,
}

/// `[providers.NAME]` as the file writes it, before `auth` and the keys it
/// needs are checked against each other.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProviderConfig {
    #[serde(rename = "type")]
    provider_type: ProviderType,
    auth: Option<AuthMode>,
    base_url: Option<BaseUrl>,
    api_key_env: Option<String>,
    aws_profile: Option<String>,
    endpoint: Option<BedrockEndpoint>,
    api: Option<BedrockApi>,
    region: Option<AwsRegion>,
    embedding_dimension: Option<Dimension>,
    max_concurrent_requests: Option<RequestLimit>,
    oauth: Option<OAuthConfig>,
}

impl TryFrom<RawProviderConfig> for ProviderConfig {
    type Error = Error;

    fn try_from(raw: RawProviderConfig) -> Result<Self> {
        let mode = raw.auth.unwrap_or_else(|| raw.provider_type.default_auth());
        let bedrock = raw.provider_type == ProviderType::Bedrock;
        if bedrock != (mode == AuthMode::Aws) {
            return Err(Error::Config(String::from(if bedrock {
                "type = \"bedrock\" signs with AWS credentials; use auth = \"aws\" or leave auth unset"
            } else {
                "auth = \"aws\" is only for type = \"bedrock\""
            })));
        }
        let bedrock_config = if bedrock {
            Some(BedrockConfig::new(
                raw.endpoint,
                raw.api,
                raw.region,
                raw.base_url.as_ref(),
            )?)
        } else {
            if raw.endpoint.is_some() || raw.api.is_some() || raw.region.is_some() {
                return Err(Error::Config(String::from(
                    "endpoint, api, and region are only for type = \"bedrock\"",
                )));
            }
            None
        };
        if raw.aws_profile.is_some() && mode != AuthMode::Aws {
            return Err(Error::Config(String::from(
                "aws_profile is only for auth = \"aws\"",
            )));
        }
        let auth = match (mode, raw.api_key_env, raw.oauth) {
            (AuthMode::Aws, None, None) => ProviderAuth::Aws {
                profile: raw.aws_profile,
            },
            (AuthMode::Aws, _, _) => {
                return Err(Error::Config(String::from(
                    "auth = \"aws\" takes neither api_key_env nor an oauth section",
                )));
            }
            (AuthMode::None, None, None) => ProviderAuth::None,
            (AuthMode::ApiKey, Some(env), None) => ProviderAuth::ApiKey { env },
            (AuthMode::Oauth, None, Some(oauth)) => {
                if oauth.issuer_url.is_empty() || oauth.client_id.is_empty() {
                    return Err(Error::Config(String::from(
                        "the oauth section needs issuer_url and client_id",
                    )));
                }
                if oauth.grant == Grant::ClientCredentials && oauth.client_secret_env.is_none() {
                    return Err(Error::Config(String::from(
                        "grant = \"client-credentials\" needs client_secret_env",
                    )));
                }
                ProviderAuth::Oauth(oauth)
            }
            (AuthMode::None, Some(_), _) => {
                return Err(Error::Config(String::from(
                    "auth = \"none\" but api_key_env is set; use auth = \"api-key\" or remove the key",
                )));
            }
            (AuthMode::ApiKey, None, _) => {
                return Err(Error::Config(String::from(
                    "auth = \"api-key\" but no api_key_env",
                )));
            }
            (AuthMode::Oauth, Some(_), _) => {
                return Err(Error::Config(String::from(
                    "auth = \"oauth\" but sets api_key_env",
                )));
            }
            (AuthMode::Oauth, None, None) => {
                return Err(Error::Config(String::from(
                    "auth = \"oauth\" but has no oauth section",
                )));
            }
            (AuthMode::None | AuthMode::ApiKey, _, Some(_)) => {
                return Err(Error::Config(String::from(
                    "an oauth section is set but auth is not \"oauth\"",
                )));
            }
        };
        Ok(Self {
            provider_type: raw.provider_type,
            auth,
            base_url: raw.base_url,
            bedrock: bedrock_config,
            embedding_dimension: raw.embedding_dimension,
            max_concurrent_requests: raw.max_concurrent_requests,
        })
    }
}

impl ProviderConfig {
    /// A provider of `provider_type` with nothing else set.
    #[must_use]
    pub fn new(provider_type: ProviderType) -> Self {
        Self {
            provider_type,
            auth: match provider_type.default_auth() {
                AuthMode::Aws => ProviderAuth::Aws { profile: None },
                AuthMode::None | AuthMode::ApiKey | AuthMode::Oauth => ProviderAuth::None,
            },
            base_url: None,
            bedrock: (provider_type == ProviderType::Bedrock).then(|| BedrockConfig {
                endpoint: BedrockEndpoint::Runtime,
                api: BedrockEndpoint::Runtime.default_api(),
                region: None,
            }),
            embedding_dimension: None,
            max_concurrent_requests: None,
        }
    }

    /// The request limit when `max_concurrent_requests` is unset.
    #[must_use]
    pub const fn default_request_limit(&self) -> RequestLimit {
        RequestLimit(match self.provider_type {
            ProviderType::Ollama => NonZeroU32::MIN,
            ProviderType::Openai | ProviderType::Anthropic | ProviderType::Bedrock => {
                NonZeroU32::MIN.saturating_add(7)
            }
        })
    }

    /// Model requests this provider may have in flight at once.
    #[must_use]
    pub fn request_limit(&self) -> RequestLimit {
        self.max_concurrent_requests
            .unwrap_or_else(|| self.default_request_limit())
    }
}

/// A resolved `PROVIDER/MODEL` reference.
#[derive(Debug, Clone, Copy)]
pub struct ModelRef<'a> {
    pub provider_name: &'a ProviderName,
    pub provider: &'a ProviderConfig,
    pub model: &'a str,
}

impl std::fmt::Display for ModelRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.provider_name, self.model)
    }
}

impl ModelRef<'_> {
    /// The width its provider's embeddings have: `validate` requires it of
    /// an embedding model's provider.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider declares no `embedding_dimension`.
    pub fn dimension(&self) -> Result<Dimension> {
        self.provider.embedding_dimension.ok_or_else(|| {
            Error::Config(format!(
                "provider '{}' is used for embeddings but has no embedding_dimension",
                self.provider_name
            ))
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IngestionConfig {
    pub chunk_size_tokens: u32,
    pub chunk_overlap_tokens: u32,
    pub embedding_batch_size: u32,
    /// Embedding requests in flight at once. An OpenAI-compatible endpoint
    /// answers them in parallel; Ollama's runner embeds one input at a
    /// time unless `OLLAMA_NUM_PARALLEL` is raised, so more only queues
    /// there.
    pub embedding_concurrency: u32,
    pub tokenizer_encoding: String,
    /// Largest upload the server accepts, in megabytes.
    pub upload_max_mb: u32,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            chunk_size_tokens: 512,
            chunk_overlap_tokens: 64,
            embedding_batch_size: 64,
            embedding_concurrency: 2,
            tokenizer_encoding: String::from("cl100k_base"),
            upload_max_mb: 512,
        }
    }
}

/// Ontology induction settings (design doc 6.5 and 13).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OntologyConfig {
    /// Share of a column's distinct values that must appear in another
    /// table's key column for a relation to be proposed.
    pub key_overlap_threshold: f64,
    /// A text column with at most this many distinct values is proposed as
    /// an enum property.
    pub enum_max_values: u32,
    /// Chunks sampled across documents for open extraction.
    pub propose_sample_chunks: u32,
    /// Distinct documents a document-evidence candidate needs to be shown
    /// in the main proposal.
    pub min_support_documents: u32,
}

impl Default for OntologyConfig {
    fn default() -> Self {
        Self {
            key_overlap_threshold: 0.8,
            enum_max_values: 12,
            propose_sample_chunks: 200,
            min_support_documents: 3,
        }
    }
}

impl OntologyConfig {
    /// The document-evidence tuning as the core module takes it.
    #[must_use]
    pub fn document_evidence(&self) -> DocumentEvidenceOptions {
        DocumentEvidenceOptions {
            sample_chunks: self.propose_sample_chunks,
            min_support_documents: self.min_support_documents,
            ..DocumentEvidenceOptions::default()
        }
    }

    /// The induction tuning as the core module takes it.
    #[must_use]
    pub fn table_evidence(&self) -> TableEvidenceOptions {
        TableEvidenceOptions {
            key_overlap_threshold: self.key_overlap_threshold,
            enum_max_values: self.enum_max_values,
            ..TableEvidenceOptions::default()
        }
    }
}

/// External data import (`quack import`, design doc 6.2).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImportConfig {
    /// Rows an import pulls at most; a larger file is cut to this.
    pub max_rows: u64,
    /// Bytes an HTTP(S) download may reach at most.
    pub max_download_mb: u64,
    /// How long a source may take to connect and answer.
    pub timeout_seconds: u32,
    /// Whether `quack serve` (with logins) may import `sqlite:` files
    /// from the server's disk. The CLI, the terminal, and `--local` always
    /// may: they run as the owner.
    pub allow_local_files: bool,
    /// Whether `quack serve` (with logins) may download from loopback,
    /// private, and link-local addresses, including cloud metadata
    /// endpoints. The CLI, the terminal, and `--local` always may.
    pub allow_private_hosts: bool,
}

impl ImportConfig {
    /// How long a source may take, at least one second.
    #[must_use]
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(u64::from(self.timeout_seconds.max(1)))
    }
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            max_rows: 1_000_000,
            max_download_mb: 512,
            timeout_seconds: 300,
            allow_local_files: false,
            allow_private_hosts: false,
        }
    }
}

/// Knowledge graph traversal and resolution (design doc 6.4 and 13).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GraphConfig {
    /// Hops a neighborhood or path query may take.
    pub max_traversal_depth: u32,
    /// Nodes a traversal or class listing returns at most.
    pub max_nodes: u32,
    /// Cosine distance under which two labels of one class are proposed
    /// as a merge for review.
    pub merge_threshold: f64,
    /// Cosine distance under which the merge happens without review.
    pub auto_merge_threshold: f64,
}

impl Default for GraphConfig {
    fn default() -> Self {
        let defaults = GraphOptions::default();
        Self {
            max_traversal_depth: defaults.max_traversal_depth,
            max_nodes: defaults.max_nodes,
            merge_threshold: defaults.merge_threshold,
            auto_merge_threshold: defaults.auto_merge_threshold,
        }
    }
}

impl GraphConfig {
    /// The tuning as the core module takes it.
    #[must_use]
    pub fn options(&self) -> GraphOptions {
        GraphOptions {
            max_traversal_depth: self.max_traversal_depth,
            max_nodes: self.max_nodes,
            merge_threshold: self.merge_threshold,
            auto_merge_threshold: self.auto_merge_threshold,
        }
    }
}

/// The work queue every interface submits background work to (design doc
/// 4.1): agent turns, SQL, ingests, imports, ontology and graph runs. Jobs
/// are not counted against a pool; what they wait on is the resource they
/// use (a provider's `max_concurrent_requests`, the workspace writer) and
/// their lane.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JobsConfig {
    /// Finished jobs kept for the job list, newest first.
    pub history: u32,
}

impl Default for JobsConfig {
    fn default() -> Self {
        Self { history: 100 }
    }
}

/// `quack serve` settings (design doc 12 and 13).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Listen address. Override: `QUACK_BIND`.
    pub bind: String,
    /// No authentication, one implicit user, loopback only.
    pub local: bool,
    /// Concurrent ingest workers per workspace.
    pub workers_per_workspace: u32,
    /// How long a browser session lives after login, however much it is
    /// used. Also the session cookie's `Max-Age`.
    pub session_max_age_hours: u32,
    /// How long a browser session survives with no request on it.
    pub session_idle_minutes: u32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: String::from("127.0.0.1:8080"),
            local: false,
            workers_per_workspace: 1,
            session_max_age_hours: 12,
            session_idle_minutes: 120,
        }
    }
}

impl ServerConfig {
    /// A session's absolute lifetime.
    #[must_use]
    pub fn session_max_age(&self) -> Duration {
        Duration::from_secs(u64::from(self.session_max_age_hours).saturating_mul(3600))
    }

    /// How long a session may sit unused before it is dropped.
    #[must_use]
    pub fn session_idle(&self) -> Duration {
        Duration::from_secs(u64::from(self.session_idle_minutes).saturating_mul(60))
    }
}

/// Workspace context (the owner-written instructions) settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ContextConfig {
    /// Approximate token budget for the context in the system prompt.
    pub max_tokens: Tokens,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_tokens: Tokens::new(4000),
        }
    }
}

/// Overrides for the input prefixes the embedding model gets for each role
/// (`quack_core::embedding`). Unset keeps the built-in prefix for the
/// model's family; an empty string sends that role unprefixed. Changing
/// one changes the embedding profile: stored vectors stop being searched
/// until `quack embeddings refresh` brings them up to date.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[expect(
    clippy::struct_field_names,
    reason = "the field names are the config keys, which read as `query_prefix = ...`"
)]
pub struct EmbeddingConfig {
    /// Before a search query.
    pub query_prefix: Option<String>,
    /// Before a document chunk; `{title}` is replaced by the chunk's
    /// heading, or `none` without one.
    pub document_prefix: Option<String>,
    /// Before texts compared with each other: entity labels and names.
    pub similarity_prefix: Option<String>,
}

/// Document retrieval settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RetrievalConfig {
    /// Default number of chunks returned by `search_documents`.
    pub top_k: u32,
    /// Reciprocal rank fusion constant for merging vector and keyword ranks.
    pub rrf_k: u32,
    /// Approximate token budget for pinned documents injected into the prompt.
    pub pinned_token_budget: Tokens,
    /// Also inject the top chunks for every user message via rig's
    /// `dynamic_context`, in addition to the `search_documents` tool.
    /// Off by default: retrieval should be a visible tool call the model
    /// chooses, not an invisible prefix on every turn.
    pub always_retrieve: bool,
    /// Reranking after hybrid fusion: `none` (the default) or `model`, the
    /// chat model ordering the candidates listwise.
    pub rerank: RerankMode,
    /// Candidates fetched for reranking before the top `k` are kept.
    pub rerank_candidates: u32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 8,
            rrf_k: 60,
            pinned_token_budget: Tokens::new(8000),
            always_retrieve: false,
            rerank: RerankMode::None,
            rerank_candidates: 24,
        }
    }
}

/// Which reranker, if any, orders retrieval candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RerankMode {
    None,
    Model,
}

text_enum!(RerankMode, "rerank mode", {
    None => "none",
    Model => "model",
});

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnalysisConfig {
    pub max_query_rows: u32,
    pub query_timeout_seconds: u32,
    pub memory_limit_mb: u32,
    pub threads: u32,
    /// Maximum model round-trips (tool calls) per turn.
    pub max_turns: u32,
    /// Approximate token budget for prior messages replayed to the model.
    pub history_token_budget: Tokens,
    /// The largest context window quack asks Ollama for (`num_ctx`). Each
    /// turn requests what its prompt needs, rounded up, no more than this;
    /// Ollama's own default is 4,096 and it truncates silently past it.
    /// Other providers size their own window.
    pub max_context_tokens: Tokens,
    /// How long one extraction call (ontology document evidence, graph
    /// extraction) may run before the chunk is skipped.
    pub extraction_timeout_seconds: u32,
    /// Chunks extracted at once. Ollama serves one request at a time
    /// unless `OLLAMA_NUM_PARALLEL` is raised, so more only queues there.
    pub extraction_concurrency: u32,
    /// Reader connections held open per workspace handle, round-robined so
    /// concurrent reads run in parallel instead of queuing behind each
    /// other on one shared connection.
    pub reader_pool_size: u32,
}

impl AnalysisConfig {
    /// How long one statement may run, at least one second.
    #[must_use]
    pub fn query_timeout(&self) -> Duration {
        Duration::from_secs(u64::from(self.query_timeout_seconds.max(1)))
    }

    /// How long one extraction call may run, at least one second.
    #[must_use]
    pub fn extraction_timeout(&self) -> Duration {
        Duration::from_secs(u64::from(self.extraction_timeout_seconds.max(1)))
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            max_query_rows: 250,
            query_timeout_seconds: 30,
            memory_limit_mb: 256,
            threads: 4,
            max_turns: 15,
            history_token_budget: Tokens::new(32_000),
            max_context_tokens: Tokens::new(32_768),
            extraction_timeout_seconds: 120,
            extraction_concurrency: 1,
            reader_pool_size: 4,
        }
    }
}

const APP_NAME: &str = "quack";

/// Return the path where the config file is expected.
#[must_use]
pub fn config_file_path() -> PathBuf {
    let config_dir =
        std::env::var(ENV_CONFIG_DIR).map_or_else(|_| default_config_dir(), PathBuf::from);
    config_dir.join("config.toml")
}

fn default_config_dir() -> PathBuf {
    home_subdir(".config")
}

fn default_data_dir() -> PathBuf {
    home_subdir(".local/share")
}

/// `~/<under>/quack`, or `<under>/quack` relative to here without a home.
fn home_subdir(under: &str) -> PathBuf {
    dirs::home_dir()
        .map_or_else(|| PathBuf::from(under), |home| home.join(under))
        .join(APP_NAME)
}

/// The values the environment puts in force over the config file:
/// `QUACK_DATA_DIR`, `QUACK_MODEL`, and `QUACK_BIND`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Overrides {
    pub data_dir: Option<PathBuf>,
    pub chat_model: Option<String>,
    pub bind: Option<String>,
}

impl Overrides {
    /// The overrides this process's environment sets.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            data_dir: std::env::var_os(ENV_DATA_DIR).map(PathBuf::from),
            chat_model: std::env::var(ENV_MODEL).ok(),
            bind: std::env::var(ENV_BIND).ok(),
        }
    }

    /// Put each set value in force over `config`, after the file and before
    /// validation. `quack config` replays this to report which values the
    /// environment, rather than the file, put in force.
    ///
    /// # Errors
    ///
    /// Returns an error when `QUACK_MODEL` is not `PROVIDER/MODEL`.
    pub fn apply(&self, config: &mut Config) -> Result<()> {
        if let Some(data_dir) = &self.data_dir {
            config.general.data_dir.clone_from(data_dir);
        }
        if let Some(model) = &self.chat_model {
            let spec = model
                .parse()
                .map_err(|e| Error::Config(format!("{ENV_MODEL}: {e}")))?;
            config.general.chat_model = Some(spec);
        }
        if let Some(bind) = &self.bind {
            config.server.bind.clone_from(bind);
        }
        Ok(())
    }
}

impl Config {
    /// Load configuration from the XDG config directory, falling back to defaults.
    ///
    /// Config file location: `~/.config/quack/config.toml`
    ///
    /// Data directory (databases, workspaces): `~/.local/share/quack/`
    ///
    /// `QUACK_DATA_DIR` overrides the data directory, `QUACK_CONFIG_DIR` the
    /// config directory, and `QUACK_MODEL` the chat model.
    ///
    /// # Errors
    ///
    /// Returns an error if the config file exists but cannot be read or
    /// parsed, contains unknown keys, or fails validation.
    pub fn load() -> Result<Self> {
        let config_path = config_file_path();
        let contents = if config_path.exists() {
            Some(std::fs::read_to_string(&config_path)?)
        } else {
            None
        };
        Self::from_contents(contents.as_deref(), &Overrides::from_env())
    }

    /// The configuration in force for a config file's text (`None` when there
    /// is no file): parsed, with `overrides` applied, then validated once.
    /// Validating before the overrides would reject a file whose bad value
    /// the environment replaces.
    ///
    /// # Errors
    ///
    /// Returns an error on syntax errors, unknown keys, or a configuration
    /// that fails validation.
    pub fn from_contents(contents: Option<&str>, overrides: &Overrides) -> Result<Self> {
        let mut config = match contents {
            Some(text) => toml::from_str(text)?,
            None => Self::default(),
        };
        overrides.apply(&mut config)?;
        config.validate()?;
        Ok(config)
    }

    /// Parse and validate TOML text alone, without the environment
    /// overrides [`Self::from_contents`] applies.
    ///
    /// # Errors
    ///
    /// Returns an error on syntax errors, unknown keys, or invalid references.
    pub fn parse(toml_text: &str) -> Result<Self> {
        let config: Self = toml::from_str(toml_text)?;
        config.validate()?;
        Ok(config)
    }

    /// Check cross-field rules the parser cannot: model references name a
    /// configured provider, auth modes match the keys present, and every
    /// embedding provider declares its dimension.
    ///
    /// # Errors
    ///
    /// Returns a `Config` error describing the first violation found.
    pub fn validate(&self) -> Result<()> {
        if self.general.chat_model.is_some() {
            self.chat_model_ref()?;
        }
        if let Some(embed) = self.embedding_model_ref()? {
            if embed.provider.provider_type == ProviderType::Anthropic {
                return Err(Error::Config(format!(
                    "embedding_model '{embed}': anthropic does not serve embeddings"
                )));
            }
            if embed
                .provider
                .bedrock
                .as_ref()
                .is_some_and(|b| b.endpoint == BedrockEndpoint::Mantle)
            {
                return Err(Error::Config(format!(
                    "embedding_model '{embed}': bedrock-mantle serves no embeddings; use a \
                     provider with endpoint = \"runtime\""
                )));
            }
            embed.dimension()?;
        }
        // Unlike the other [analysis] numbers, this one allocates OS-level
        // DuckDB connections at workspace open, one spawn_blocking round
        // trip and one writer-mutex acquisition each — an unreasonable
        // value blocks every request to that workspace until it finishes.
        if !(1..=32).contains(&self.analysis.reader_pool_size) {
            return Err(Error::Config(format!(
                "[analysis].reader_pool_size must be between 1 and 32, got {}",
                self.analysis.reader_pool_size
            )));
        }
        Ok(())
    }

    fn resolve_model<'a>(&'a self, setting: &str, spec: &'a ModelSpec) -> Result<ModelRef<'a>> {
        let (provider_name, provider) =
            self.providers
                .get_key_value(spec.provider())
                .ok_or_else(|| {
                    Error::Config(format!(
                        "{setting} = \"{spec}\" names provider '{}', which is not configured; \
                     add a [providers.{}] section in {}",
                        spec.provider(),
                        spec.provider(),
                        config_file_path().display()
                    ))
                })?;
        Ok(ModelRef {
            provider_name,
            provider,
            model: spec.model(),
        })
    }

    /// The chat model, required.
    ///
    /// # Errors
    ///
    /// Returns an error if `chat_model` is unset or names an unknown provider.
    pub fn chat_model_ref(&self) -> Result<ModelRef<'_>> {
        let spec = self
            .general
            .chat_model
            .as_ref()
            .ok_or_else(|| Error::NoChatModel {
                config_file: config_file_path(),
            })?;
        self.resolve_model("chat_model", spec)
    }

    /// `provider/model` for status lines, or a placeholder.
    #[must_use]
    pub fn chat_model_label(&self) -> String {
        self.chat_model_ref()
            .map_or_else(|_| String::from("no chat model"), |m| m.to_string())
    }

    /// The embedding model, if configured.
    ///
    /// # Errors
    ///
    /// Returns an error if `embedding_model` names an unknown provider.
    pub fn embedding_model_ref(&self) -> Result<Option<ModelRef<'_>>> {
        self.general
            .embedding_model
            .as_ref()
            .map(|spec| self.resolve_model("embedding_model", spec))
            .transpose()
    }

    /// Directory holding the encrypted OAuth token caches, one per provider.
    #[must_use]
    pub fn tokens_dir(&self) -> PathBuf {
        self.general.data_dir.join("tokens")
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

    /// Ensure the data directory and its subdirectories exist. A data
    /// directory this call creates is private to the user (0700 on Unix):
    /// it holds every workspace's content, the control database, and the
    /// OAuth token caches. An existing one keeps its mode, which
    /// `quack doctor` reports when others can read it.
    ///
    /// # Errors
    ///
    /// Returns an error if the directories cannot be created.
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        let data_dir = &self.general.data_dir;
        if !data_dir.exists() {
            std::fs::create_dir_all(data_dir)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(data_dir, std::fs::Permissions::from_mode(0o700))?;
            }
        }
        std::fs::create_dir_all(data_dir.join("workspaces"))?;
        Ok(())
    }

    #[must_use]
    pub fn data_dir(&self) -> &Path {
        &self.general.data_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: &str = r#"
[general]
chat_model = "ollama/llama3.1:8b"
embedding_model = "ollama/nomic-embed-text"

[providers.ollama]
type = "ollama"
base_url = "http://localhost:11434"
embedding_dimension = 768

[providers.anthropic]
type = "anthropic"
auth = "api-key"
api_key_env = "ANTHROPIC_API_KEY"

[retrieval]
top_k = 3
always_retrieve = true
rerank = "model"
"#;

    #[test]
    fn default_config_values() {
        let config = Config::default();
        assert_eq!(config.general.default_workspace, "default");
        assert!(config.general.chat_model.is_none());
        assert_eq!(config.ingestion.chunk_size_tokens, 512);
        assert_eq!(config.retrieval.top_k, 8);
        assert_eq!(config.retrieval.rrf_k, 60);
        assert_eq!(config.retrieval.pinned_token_budget, Tokens::new(8000));
        assert!(!config.retrieval.always_retrieve);
        assert_eq!(config.retrieval.rerank, RerankMode::None);
        assert_eq!(config.retrieval.rerank_candidates, 24);
        assert_eq!(config.context.max_tokens, Tokens::new(4000));
        assert_eq!(config.analysis.threads, 4);
        assert_eq!(config.analysis.max_turns, 15);
        assert_eq!(config.analysis.history_token_budget, Tokens::new(32_000));
        assert_eq!(config.ingestion.upload_max_mb, 512);
        assert_eq!(config.ingestion.embedding_concurrency, 2);
        assert_eq!(config.server.bind, "127.0.0.1:8080");
        assert!(!config.server.local);
        assert_eq!(config.server.workers_per_workspace, 1);
        assert!((config.ontology.key_overlap_threshold - 0.8).abs() < f64::EPSILON);
        assert_eq!(config.ontology.enum_max_values, 12);
        assert_eq!(config.ontology.propose_sample_chunks, 200);
        assert_eq!(config.ontology.min_support_documents, 3);
        assert!(err_of("[ontology]\nsample = 1\n").contains("sample"));
        assert_eq!(Config::default().graph.max_traversal_depth, 3);
        assert_eq!(Config::default().graph.max_nodes, 200);
        assert!(err_of("[graph]\nenabled = true\n").contains("enabled"));
        assert_eq!(Config::default().import.max_rows, 1_000_000);
        assert!(err_of("[import]\nmax = 1\n").contains("max"));
    }

    #[test]
    fn server_section_parses_and_rejects_unknown_keys() {
        let ok = Config::parse("[server]\nbind = \"0.0.0.0:9000\"\nlocal = true\n");
        assert!(ok.is_ok_and(|c| c.server.bind == "0.0.0.0:9000" && c.server.local));
        assert!(err_of("[server]\nport = 1\n").contains("port"));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn full_config_parses_and_resolves_models() {
        let config = Config::parse(FULL).unwrap();
        let chat = config.chat_model_ref().unwrap();
        assert_eq!(chat.provider_name, "ollama");
        assert_eq!(chat.model, "llama3.1:8b");
        assert_eq!(chat.provider.provider_type, ProviderType::Ollama);
        assert_eq!(chat.provider.auth.mode(), AuthMode::None);
        assert_eq!(chat.to_string(), "ollama/llama3.1:8b");
        let embed = config.embedding_model_ref().unwrap().unwrap();
        assert_eq!(embed.model, "nomic-embed-text");
        assert_eq!(
            embed.provider.embedding_dimension,
            Some(Dimension::new(768))
        );
        assert_eq!(config.retrieval.top_k, 3);
        assert_eq!(config.retrieval.rerank, RerankMode::Model);
        let anthropic = config.providers.get("anthropic");
        assert!(anthropic.is_some_and(|p| {
            p.provider_type == ProviderType::Anthropic
                && p.auth.api_key_env() == Some("ANTHROPIC_API_KEY")
        }));
    }

    #[test]
    fn model_specs_split_once_when_read() {
        let spec: ModelSpec = "ollama/llama3.1:8b"
            .parse()
            .unwrap_or_else(|e| panic_on(&e));
        assert_eq!(spec.provider(), "ollama");
        assert_eq!(spec.model(), "llama3.1:8b");
        assert_eq!(spec.to_string(), "ollama/llama3.1:8b");
        // The model may itself contain a slash; only the first one splits.
        let nested: ModelSpec = "hf/org/model".parse().unwrap_or_else(|e| panic_on(&e));
        assert_eq!(
            (nested.provider().as_str(), nested.model()),
            ("hf", "org/model")
        );
        for bad in ["llama3", "ollama/", "/m", "bad name/m"] {
            assert!(bad.parse::<ModelSpec>().is_err(), "{bad}");
        }
        assert!(err_of("[general]\nchat_model = \"llama3\"\n").contains("PROVIDER/MODEL"));
    }

    #[test]
    fn base_urls_are_checked_and_trimmed() {
        let url =
            |text: &str| BaseUrl::try_from(String::from(text)).unwrap_or_else(|e| panic_on(&e));
        let ollama = url("http://gpu-box:11434/v1/");
        assert_eq!(ollama.trimmed(), "http://gpu-box:11434/v1");
        assert_eq!(ollama.root(), "http://gpu-box:11434");
        assert_eq!(
            url("https://api.openai.com/v1").root(),
            "https://api.openai.com"
        );
        for bad in ["localhost:11434", "ftp://h/", "not a url", ""] {
            assert!(BaseUrl::try_from(String::from(bad)).is_err(), "{bad}");
        }
        assert!(
            err_of("[providers.o]\ntype = \"ollama\"\nbase_url = \"localhost:1\"\n")
                .contains("http:// or https://")
        );
        assert_eq!(
            ProviderType::OLLAMA_BASE_URL.as_str(),
            "http://localhost:11434"
        );
    }

    #[test]
    fn cleartext_is_plain_http_off_this_machine() {
        let url =
            |text: &str| BaseUrl::try_from(String::from(text)).unwrap_or_else(|e| panic_on(&e));
        assert!(url("http://gpu-box:11434").sends_in_cleartext());
        assert!(url("http://10.0.0.5/v1").sends_in_cleartext());
        assert!(!url("http://localhost:11434").sends_in_cleartext());
        assert!(!url("http://127.0.0.1:11434").sends_in_cleartext());
        assert!(!url("http://[::1]:11434").sends_in_cleartext());
        assert!(!url("https://api.openai.com/v1").sends_in_cleartext());
    }

    #[test]
    fn request_limits_are_at_least_one() {
        assert!(
            err_of("[providers.o]\ntype = \"ollama\"\nmax_concurrent_requests = 0\n")
                .contains("nonzero")
        );
        let ollama = ProviderConfig::new(ProviderType::Ollama);
        assert_eq!(ollama.request_limit().get(), 1);
        assert_eq!(
            ProviderConfig::new(ProviderType::Openai)
                .request_limit()
                .get(),
            8
        );
    }

    /// Every combination of `auth`, `api_key_env`, and an oauth table: the
    /// three that make sense read as their `ProviderAuth`, the rest are
    /// refused when read.
    #[test]
    fn provider_auth_is_one_of_three_shapes() {
        let oauth = "[providers.p.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n";
        let case = |auth: &str, key: bool, table: bool| {
            let mut text = format!("[providers.p]\ntype = \"openai\"\nauth = \"{auth}\"\n");
            if key {
                text.push_str("api_key_env = \"K\"\n");
            }
            if table {
                text.push_str(oauth);
            }
            Config::parse(&text).map(|c| {
                c.providers
                    .get("p")
                    .map(|p| p.auth.mode())
                    .unwrap_or_default()
            })
        };
        assert!(case("none", false, false).is_ok_and(|m| m == AuthMode::None));
        assert!(case("api-key", true, false).is_ok_and(|m| m == AuthMode::ApiKey));
        assert!(case("oauth", false, true).is_ok_and(|m| m == AuthMode::Oauth));
        for (auth, key, table) in [
            ("none", true, false),
            ("none", false, true),
            ("api-key", false, false),
            ("api-key", true, true),
            ("oauth", false, false),
            ("oauth", true, true),
        ] {
            assert!(case(auth, key, table).is_err(), "{auth} {key} {table}");
        }
        assert!(
            err_of("[providers.p]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.p.oauth]\nissuer_url = \"\"\nclient_id = \"c\"\n")
                .contains("issuer_url and client_id")
        );
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn panic_on(e: &Error) -> ! {
        panic!("{e}")
    }

    #[test]
    fn an_override_rescues_the_file_value_it_replaces() {
        let file = "[general]\nchat_model = \"missing/model\"\n\n\
                    [providers.ollama]\ntype = \"ollama\"\n";
        let none = Overrides::default();
        let rejected = Config::from_contents(Some(file), &none).err();
        assert!(rejected.is_some_and(|e| e.to_string().contains("missing")));
        let rescued = Overrides {
            chat_model: Some(String::from("ollama/gpt-oss:20b")),
            ..Overrides::default()
        };
        let config = Config::from_contents(Some(file), &rescued);
        assert!(config.is_ok_and(|c| {
            c.general
                .chat_model
                .is_some_and(|m| m.to_string() == "ollama/gpt-oss:20b")
        }));
        // An override is still validated: it cannot name a missing provider.
        let bad = Overrides {
            chat_model: Some(String::from("nowhere/x")),
            ..Overrides::default()
        };
        assert!(Config::from_contents(None, &bad).is_err());
    }

    #[test]
    fn provider_names_that_could_leave_the_tokens_directory_are_rejected() {
        for bad in ["../evil", ".hidden", "a/b", "a\\\\b", "", "sp ace"] {
            let toml_text = format!("[providers.\"{bad}\"]\ntype = \"ollama\"\n");
            assert!(err_of(&toml_text).contains("provider name"), "{bad:?}");
        }
        let long = "p".repeat(65);
        assert!(long.parse::<ProviderName>().is_err());
        for good in ["ollama", "azure.openai", "corp-gw_2", &"p".repeat(64)] {
            assert!(good.parse::<ProviderName>().is_ok(), "{good}");
        }
        let config = Config::parse("[providers.\"azure.openai\"]\ntype = \"openai\"\n");
        assert!(config.is_ok_and(|c| c.providers.contains_key("azure.openai")));
    }

    #[test]
    fn overrides_replace_only_what_they_set() {
        let overrides = Overrides {
            data_dir: Some(PathBuf::from("/srv/quack")),
            bind: Some(String::from("0.0.0.0:9000")),
            ..Overrides::default()
        };
        let config = Config::from_contents(Some("[general]\n"), &overrides);
        assert!(
            config.is_ok_and(|c| c.general.data_dir == Path::new("/srv/quack")
                && c.server.bind == "0.0.0.0:9000"
                && c.general.chat_model.is_none())
        );
    }

    fn err_of(toml_text: &str) -> String {
        match Config::parse(toml_text) {
            Ok(_) => String::from("<ok>"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn unknown_keys_are_rejected_at_every_level() {
        assert!(err_of("[genral]\nchat_model = \"a/b\"\n").contains("genral"));
        assert!(err_of("[general]\nchat_modle = \"a/b\"\n").contains("chat_modle"));
        assert!(err_of("[providers.o]\ntype = \"ollama\"\nmodel = \"x\"\n").contains("model"));
        assert!(err_of("[retrieval]\ntopk = 1\n").contains("topk"));
        assert!(err_of("[retrieval]\nrerank = \"bge\"\n").contains("bge"));
        assert!(err_of("[analysis]\nthread = 1\n").contains("thread"));
    }

    #[test]
    fn unknown_provider_type_is_rejected() {
        assert!(err_of("[providers.o]\ntype = \"vertex\"\n").contains("vertex"));
    }

    #[test]
    fn bedrock_signs_with_the_aws_chain_by_default() {
        let config = Config::parse(
            "[general]\nchat_model = \"aws/us.anthropic.claude-sonnet-5\"\n\
             [providers.aws]\ntype = \"bedrock\"\naws_profile = \"dev-sso\"\nregion = \"us-west-2\"\n",
        );
        assert!(
            config.is_ok_and(|c| c.providers.get("aws").is_some_and(|p| {
                p.provider_type == ProviderType::Bedrock
                    && p.auth.mode() == AuthMode::Aws
                    && p.auth.aws_profile() == Some("dev-sso")
                    && p.bedrock.as_ref().is_some_and(|b| {
                        b.region.as_ref().map(AwsRegion::as_str) == Some("us-west-2")
                            && b.endpoint == BedrockEndpoint::Runtime
                            && b.api == BedrockApi::Converse
                    })
                    && p.request_limit().get() == 8
            }))
        );
        // Neither profile nor region is required: the SDK's chain decides.
        let config = Config::parse("[providers.b]\ntype = \"bedrock\"\nauth = \"aws\"\n");
        assert!(config.is_ok_and(|c| {
            c.providers.get("b").is_some_and(|p| {
                p.auth.aws_profile().is_none()
                    && p.bedrock.as_ref().is_some_and(|b| b.region.is_none())
            })
        }));
        assert_eq!(
            ProviderConfig::new(ProviderType::Bedrock).auth.mode(),
            AuthMode::Aws
        );
        assert!(ProviderType::Bedrock.default_base_url().is_none());
    }

    #[test]
    fn aws_settings_are_refused_where_they_mean_nothing() {
        assert!(
            err_of("[providers.b]\ntype = \"bedrock\"\nauth = \"api-key\"\napi_key_env = \"K\"\n")
                .contains("bedrock")
        );
        assert!(err_of("[providers.b]\ntype = \"bedrock\"\nauth = \"none\"\n").contains("bedrock"));
        assert!(err_of("[providers.o]\ntype = \"openai\"\nauth = \"aws\"\n").contains("aws"));
        assert!(
            err_of("[providers.o]\ntype = \"openai\"\nauth = \"api-key\"\napi_key_env = \"K\"\nregion = \"us-east-1\"\n")
                .contains("region")
        );
        assert!(
            err_of("[providers.o]\ntype = \"ollama\"\naws_profile = \"p\"\n")
                .contains("aws_profile")
        );
        assert!(
            err_of("[providers.b]\ntype = \"bedrock\"\napi_key_env = \"K\"\n")
                .contains("api_key_env")
        );
        assert!(
            err_of("[providers.b]\ntype = \"bedrock\"\nregion = \"us east\"\n").contains("region")
        );
        assert!(err_of("[providers.b]\ntype = \"bedrock\"\nregion = \"\"\n").contains("region"));
        assert!(
            err_of("[providers.o]\ntype = \"ollama\"\nendpoint = \"mantle\"\n")
                .contains("endpoint")
        );
        assert!(
            err_of(
                "[providers.b]\ntype = \"bedrock\"\nendpoint = \"mantle\"\napi = \"converse\"\n"
            )
            .contains("not served")
        );
        assert!(
            err_of(
                "[general]\nembedding_model = \"m/amazon.titan-embed-text-v2:0\"\n\
                 [providers.m]\ntype = \"bedrock\"\nendpoint = \"mantle\"\nembedding_dimension = 1024\n"
            )
            .contains("serves no embeddings")
        );
    }

    #[test]
    fn runtime_and_mantle_are_two_providers_sharing_a_profile() {
        let config = Config::parse(
            "[general]\nchat_model = \"mantle/openai.gpt-oss-120b\"\n\
             embedding_model = \"bedrock/amazon.titan-embed-text-v2:0\"\n\
             [providers.bedrock]\ntype = \"bedrock\"\naws_profile = \"sso\"\nregion = \"us-east-1\"\n\
             embedding_dimension = 1024\n\
             [providers.mantle]\ntype = \"bedrock\"\naws_profile = \"sso\"\nendpoint = \"mantle\"\n\
             base_url = \"https://vpce-0abc.bedrock-mantle.us-east-1.vpce.amazonaws.com\"\n",
        );
        let Ok(config) = config else {
            return assert!(config.is_ok(), "{:?}", config.err());
        };
        let mantle = config
            .providers
            .get("mantle")
            .and_then(|p| p.bedrock.clone());
        assert_eq!(
            mantle,
            Some(BedrockConfig {
                endpoint: BedrockEndpoint::Mantle,
                api: BedrockApi::Responses,
                region: AwsRegion::try_from(String::from("us-east-1")).ok(),
            })
        );
    }

    #[test]
    fn openai_compat_alias_maps_to_openai() {
        let config = Config::parse(
            "[providers.g]\ntype = \"openai-compat\"\nauth = \"api-key\"\napi_key_env = \"K\"\n",
        );
        assert!(config.is_ok_and(|c| {
            c.providers
                .get("g")
                .is_some_and(|p| p.provider_type == ProviderType::Openai)
        }));
    }

    #[test]
    fn model_reference_must_name_a_configured_provider() {
        let msg = err_of("[general]\nchat_model = \"missing/m\"\n");
        assert!(
            msg.contains("'missing'") && msg.contains("not configured"),
            "{msg}"
        );
    }

    #[test]
    fn model_reference_must_have_a_slash_and_a_model() {
        assert!(
            err_of("[general]\nchat_model = \"ollama\"\n[providers.ollama]\ntype = \"ollama\"\n")
                .contains("PROVIDER/MODEL")
        );
        assert!(
            err_of("[general]\nchat_model = \"ollama/\"\n[providers.ollama]\ntype = \"ollama\"\n")
                .contains("missing the model")
        );
    }

    #[test]
    fn auth_mode_must_agree_with_api_key_env() {
        assert!(
            err_of("[providers.o]\ntype = \"openai\"\napi_key_env = \"K\"\n")
                .contains("auth = \"none\"")
        );
        assert!(
            err_of("[providers.o]\ntype = \"openai\"\nauth = \"api-key\"\n")
                .contains("no api_key_env")
        );
        assert!(
            err_of("[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\n")
                .contains("has no oauth section")
        );
    }

    #[test]
    fn oauth_section_is_required_by_and_exclusive_to_oauth_mode() {
        let stray = "[providers.o]\ntype = \"openai\"\n[providers.o.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n";
        assert!(err_of(stray).contains("auth is not \"oauth\""));
        let with_key = "[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\napi_key_env = \"K\"\n[providers.o.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n";
        assert!(err_of(with_key).contains("sets api_key_env"));
        let empty = "[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.o.oauth]\nissuer_url = \"\"\nclient_id = \"c\"\n";
        assert!(err_of(empty).contains("needs issuer_url and client_id"));
        assert!(err_of("[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.o.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\ntenant = \"x\"\n").contains("tenant"));
    }

    #[test]
    fn the_grant_is_named_and_client_credentials_needs_a_secret() {
        let section = "[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.o.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n";
        let grant_of = |extra: &str| {
            Config::parse(&format!("{section}{extra}"))
                .ok()
                .and_then(|c| {
                    c.providers
                        .get("o")
                        .and_then(|p| p.auth.oauth().map(|o| o.grant))
                })
        };
        assert_eq!(
            grant_of("grant = \"device-code\"\n"),
            Some(Grant::DeviceCode)
        );
        assert_eq!(
            grant_of("grant = \"client-credentials\"\nclient_secret_env = \"S\"\n"),
            Some(Grant::ClientCredentials)
        );
        assert!(
            err_of(&format!("{section}grant = \"client-credentials\"\n"))
                .contains("needs client_secret_env")
        );
        assert!(err_of(&format!("{section}grant = \"password\"\n")).contains("password"));
        assert!(err_of(&format!("{section}device_code = true\n")).contains("device_code"));
    }

    #[test]
    fn oauth_section_defaults_and_fields_parse() {
        let config = Config::parse(
            "[providers.azure]\ntype = \"openai\"\nauth = \"oauth\"\nbase_url = \"https://r.openai.azure.com/openai/deployments/d\"\n[providers.azure.oauth]\nissuer_url = \"https://login.microsoftonline.com/t/v2.0\"\nclient_id = \"abc\"\nscopes = [\"https://cognitiveservices.azure.com/.default\", \"offline_access\"]\nclient_secret_env = \"AZURE_CLIENT_SECRET\"\n",
        );
        let Ok(config) = config else {
            return assert!(config.is_ok(), "{config:?}");
        };
        let oauth = config.providers.get("azure").and_then(|p| p.auth.oauth());
        assert!(oauth.is_some_and(|o| {
            o.redirect_uri == "http://127.0.0.1:19876/callback"
                && o.grant == Grant::AuthorizationCode
                && o.scopes.len() == 2
                && o.client_secret_env.as_deref() == Some("AZURE_CLIENT_SECRET")
        }));
        assert_eq!(config.tokens_dir(), config.general.data_dir.join("tokens"));
    }

    #[test]
    fn embedding_provider_needs_dimension_and_cannot_be_anthropic() {
        let no_dim = "[general]\nembedding_model = \"o/e\"\n[providers.o]\ntype = \"ollama\"\n";
        assert!(err_of(no_dim).contains("embedding_dimension"));
        let anthropic = "[general]\nembedding_model = \"a/e\"\n[providers.a]\ntype = \"anthropic\"\nauth = \"api-key\"\napi_key_env = \"K\"\nembedding_dimension = 1\n";
        assert!(err_of(anthropic).contains("does not serve embeddings"));
    }

    #[test]
    fn chat_model_unset_is_a_clear_error_when_asked_for() {
        let config = Config::default();
        assert!(config.validate().is_ok());
        let err = config.chat_model_ref().err();
        assert!(err.is_some_and(|e| e.to_string().contains("chat_model")));
        assert!(config.embedding_model_ref().is_ok_and(|m| m.is_none()));
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
    fn default_dirs_use_xdg_layout() {
        assert!(default_data_dir().to_string_lossy().contains(APP_NAME));
        assert!(default_config_dir().to_string_lossy().contains(APP_NAME));
    }
}
