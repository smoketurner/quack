use ipnet::IpNet;
use serde::Deserialize;
use std::borrow::{Borrow, Cow};
use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use crate::embedding::Dimension;
use crate::error::{Error, Result};
use crate::ingestion::budget::DecompressionBudget;
use crate::ontology::documents::DocumentEvidenceOptions;
use crate::ontology::induction::TableEvidenceOptions;
use crate::storage::control::WorkspaceName;
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
    pub default_workspace: WorkspaceName,
    /// `PROVIDER/MODEL` used for chat and tool calling. Override: `QUACK_MODEL`.
    pub chat_model: Option<ModelSpec>,
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
            default_workspace: WorkspaceName::default(),
            chat_model: None,
        }
    }
}

/// A `[providers.NAME]` key. It is typed on the command line (`quack auth
/// login NAME`), keys the provider's sealed token in `control.db`, and is
/// the subject that token is sealed for, so it is checked when the config is
/// read: ASCII letters, digits, `_`, and `-`, at most 64 characters. No `.`:
/// in TOML, `[providers.a.b]` is a table `b` inside provider `a`, so a dotted
/// name would only work quoted.
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
        let allowed = |c: char| c.is_ascii_alphanumeric() || matches!(c, '_' | '-');
        if name.is_empty() || name.len() > Self::MAX_LEN || !name.chars().all(allowed) {
            return Err(Error::Config(format!(
                "provider name '{name}' must be 1 to {} ASCII letters, digits, '_', or '-' \
                 (no '.': TOML reads [providers.a.b] as a table inside provider 'a')",
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
    /// Amazon Bedrock's `bedrock-runtime` endpoint, signed with credentials
    /// from the AWS SDK's default chain.
    Bedrock,
    /// Amazon Bedrock's `bedrock-mantle` endpoint, which hosts other models
    /// and APIs (`config::bedrock`), signed the same way.
    BedrockMantle,
}

text_enum!(ProviderType, "provider type", {
    Ollama => "ollama",
    Openai => "openai",
    Anthropic => "anthropic",
    Bedrock => "bedrock",
    BedrockMantle => "bedrock-mantle",
});

impl ProviderType {
    /// Ollama's API when `base_url` is unset.
    pub const OLLAMA_BASE_URL: BaseUrl = BaseUrl(Cow::Borrowed("http://localhost:11434"));

    /// The `api` a type other than the Bedrock ones sets, checked: only
    /// `openai` takes one, and never Bedrock's `converse`.
    fn openai_api(self, api: Option<BedrockApi>) -> Result<Option<BedrockApi>> {
        match (self, api) {
            (_, None) => Ok(None),
            (Self::Openai, Some(BedrockApi::Converse)) => Err(Error::Config(String::from(
                "type = \"openai\" takes api = \"chat-completions\" or \"responses\"; \
                 converse is Bedrock's own API",
            ))),
            (Self::Openai, Some(api)) => Ok(Some(api)),
            (_, Some(_)) => Err(Error::Config(String::from(
                "api is only for type = \"openai\", \"bedrock\", and \"bedrock-mantle\"",
            ))),
        }
    }

    /// Where the provider's API is when `base_url` is unset: the same
    /// defaults rig's clients use. `None` for Bedrock, whose endpoint
    /// follows its region (`llm::bedrock`).
    #[must_use]
    pub const fn default_base_url(self) -> Option<BaseUrl> {
        match self {
            Self::Ollama => Some(Self::OLLAMA_BASE_URL),
            Self::Openai => Some(BaseUrl(Cow::Borrowed("https://api.openai.com/v1"))),
            Self::Anthropic => Some(BaseUrl(Cow::Borrowed("https://api.anthropic.com"))),
            Self::Bedrock | Self::BedrockMantle => None,
        }
    }

    /// The `auth` a provider of this type has when the file names none.
    #[must_use]
    pub const fn default_auth(self) -> AuthMode {
        match self.bedrock_endpoint() {
            Some(_) => AuthMode::Aws,
            None => AuthMode::None,
        }
    }

    /// The Bedrock endpoint a provider of this type calls; `None` for the
    /// types that are not Bedrock.
    #[must_use]
    pub const fn bedrock_endpoint(self) -> Option<BedrockEndpoint> {
        match self {
            Self::Bedrock => Some(BedrockEndpoint::Runtime),
            Self::BedrockMantle => Some(BedrockEndpoint::Mantle),
            Self::Ollama | Self::Openai | Self::Anthropic => None,
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
    /// Each request reaches the provider as the person who made it: quack
    /// exchanges that person's own token for one to this provider (`quack
    /// serve` only; see [`Exchange`]).
    OnBehalfOf,
}

/// The wire form of an on-behalf-of exchange.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Exchange {
    /// RFC 8693 token exchange (Okta, Auth0, Vouch): the user's token as
    /// `subject_token`, and quack's own token as `actor_token` unless `actor`
    /// is off.
    #[default]
    TokenExchange,
    /// Microsoft Entra ID's On-Behalf-Of flow: the `jwt-bearer` grant with
    /// `requested_token_use=on_behalf_of`. It has no actor token.
    Entra,
}

text_enum!(Exchange, "exchange", {
    TokenExchange => "token-exchange",
    Entra => "entra",
});

/// How quack authenticates itself at a token endpoint: with its client
/// secret (RFC 6749 2.3.1), or with a JWT signed by its own key (RFC 7523
/// 2.2). The names are RFC 8414's. With a secret method and no
/// `client_secret_env`, quack is a public client and sends only its
/// `client_id`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
pub enum ClientAuth {
    /// The secret in the request body (Entra, Auth0's default).
    #[default]
    #[serde(rename = "client_secret_post")]
    ClientSecretPost,
    /// The secret in an HTTP Basic `Authorization` header (Okta's default).
    #[serde(rename = "client_secret_basic")]
    ClientSecretBasic,
    /// A client assertion: a short-lived ES256 JWT signed with a P-256 key
    /// quack keeps sealed in `control.db`, whose public half is registered
    /// with the issuer (`quack auth jwks`). No secret is shared.
    #[serde(rename = "private_key_jwt")]
    PrivateKeyJwt,
}

text_enum!(ClientAuth, "client authentication", {
    ClientSecretPost => "client_secret_post",
    ClientSecretBasic => "client_secret_basic",
    PrivateKeyJwt => "private_key_jwt",
});

impl ClientAuth {
    /// The rule between `client_auth` and `client_secret_env` for the
    /// section `section`: a signed assertion replaces the secret, so naming
    /// both is refused.
    fn check(self, section: &str, client_secret_env: Option<&String>) -> Result<()> {
        if self == Self::PrivateKeyJwt && client_secret_env.is_some() {
            return Err(Error::Config(format!(
                "{section}: client_auth = \"private_key_jwt\" signs with quack's own key and sends no secret; remove client_secret_env"
            )));
        }
        Ok(())
    }

    /// The rule for leaving `client_id` out: only a client quack registers
    /// itself (`quack auth register`) has none in the file, and quack
    /// registers every client it makes with `private_key_jwt`.
    fn check_client_id(self, section: &str, client_id: Option<&str>) -> Result<()> {
        match client_id {
            Some(id) if id.trim().is_empty() => Err(Error::Config(format!(
                "{section}: client_id is empty; name the client, or remove client_id for one `quack auth register` registers"
            ))),
            None if self != Self::PrivateKeyJwt => Err(Error::Config(format!(
                "{section} needs client_id, unless quack registers the client itself (`quack auth register`), which takes client_auth = \"private_key_jwt\""
            ))),
            Some(_) | None => Ok(()),
        }
    }

    /// Whether the client proves who it is: a secret is named, or it signs
    /// assertions.
    const fn is_confidential(self, has_secret: bool) -> bool {
        has_secret || matches!(self, Self::PrivateKeyJwt)
    }
}

text_enum!(Grant, "grant", {
    AuthorizationCode => "authorization-code",
    DeviceCode => "device-code",
    ClientCredentials => "client-credentials",
    OnBehalfOf => "on-behalf-of",
});

/// `[providers.NAME.oauth]`: the issuer, the client, and the grant that
/// obtains the token quack sends to the provider's endpoint.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OAuthConfig {
    /// Issuer whose `/.well-known/openid-configuration` names the endpoints,
    /// e.g. `https://login.microsoftonline.com/{tenant}/v2.0`.
    pub issuer_url: String,
    /// The client quack is at the issuer. Unset for a client quack
    /// registered itself (`quack auth register`, RFC 7591), whose id is
    /// read from the registration kept for the issuer.
    pub client_id: Option<String>,
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
    /// confidential client. `client-credentials` and `on-behalf-of` require
    /// it, unless `client_auth` is `private_key_jwt`, which excludes it.
    pub client_secret_env: Option<String>,
    /// How quack authenticates at the token endpoint.
    #[serde(default)]
    pub client_auth: ClientAuth,
    /// `on-behalf-of`: the exchange's wire form.
    #[serde(default)]
    pub exchange: Exchange,
    /// `on-behalf-of`: the RFC 8693 `audience` of the token to obtain (Okta,
    /// Auth0).
    pub audience: Option<String>,
    /// `on-behalf-of`: the RFC 8707 `resource` of the token to obtain.
    pub resource: Option<String>,
    /// `on-behalf-of` with `token-exchange`: send quack's own
    /// client-credentials token as the `actor_token`, so the issued token
    /// names quack as the actor (`act`) beside the user (`sub`).
    #[serde(default = "OAuthConfig::default_actor")]
    pub actor: bool,
}

impl OAuthConfig {
    /// The loopback redirect the browser flow listens on by default.
    pub const DEFAULT_REDIRECT_URI: &str = "http://127.0.0.1:19876/callback";

    fn default_redirect_uri() -> String {
        String::from(Self::DEFAULT_REDIRECT_URI)
    }

    const fn default_actor() -> bool {
        true
    }

    /// The grant's name in an issuer's `grant_types_supported`.
    #[must_use]
    pub const fn grant_type(&self) -> &'static str {
        match (self.grant, self.exchange) {
            (Grant::AuthorizationCode, _) => "authorization_code",
            (Grant::DeviceCode, _) => "urn:ietf:params:oauth:grant-type:device_code",
            (Grant::ClientCredentials, _) => "client_credentials",
            (Grant::OnBehalfOf, Exchange::TokenExchange) => {
                "urn:ietf:params:oauth:grant-type:token-exchange"
            }
            (Grant::OnBehalfOf, Exchange::Entra) => "urn:ietf:params:oauth:grant-type:jwt-bearer",
        }
    }

    /// The rules between the section's keys that the parser cannot check.
    fn check(&self) -> Result<()> {
        if self.issuer_url.is_empty() {
            return Err(Error::Config(String::from(
                "the oauth section needs issuer_url",
            )));
        }
        self.client_auth
            .check("the oauth section", self.client_secret_env.as_ref())?;
        self.client_auth
            .check_client_id("the oauth section", self.client_id.as_deref())?;
        if matches!(self.grant, Grant::ClientCredentials | Grant::OnBehalfOf)
            && !self
                .client_auth
                .is_confidential(self.client_secret_env.is_some())
        {
            return Err(Error::Config(format!(
                "grant = \"{}\" needs client_secret_env, or client_auth = \"private_key_jwt\"",
                self.grant
            )));
        }
        if self.grant != Grant::OnBehalfOf
            && (self.exchange != Exchange::default()
                || self.audience.is_some()
                || self.resource.is_some())
        {
            return Err(Error::Config(String::from(
                "exchange, audience, and resource apply only to grant = \"on-behalf-of\"",
            )));
        }
        Ok(())
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
    /// The `api` a `type = "openai"` provider sets: `chat-completions` or
    /// `responses`. `None` when unset or for other types; see
    /// [`Self::openai_chat_api`] for the one in force.
    pub openai_api: Option<BedrockApi>,
    /// Model requests in flight to this provider at once, across the whole
    /// process; the rest wait their turn (design doc 4.1). Unset: 1 for
    /// Ollama, which serves one request per model unless
    /// `OLLAMA_NUM_PARALLEL` says otherwise, 8 for hosted APIs.
    pub max_concurrent_requests: Option<RequestLimit>,
    /// Extra HTTP headers sent with every model request to this provider,
    /// checked as header names and values when the file is read. The
    /// credential headers are refused: rig would send one in place of the
    /// credential `auth` provides.
    pub headers: Option<BTreeMap<String, String>>,
    /// `temperature`, `effort`, and `background_effort` on
    /// `[providers.NAME]`: for every model of this provider unless its own
    /// entry in `models` says.
    pub model_defaults: ModelSettings,
    /// `[providers.NAME.models."ID"]`: one model's settings, keyed by the
    /// model id as `chat_model` names it.
    pub models: BTreeMap<String, ModelSettings>,
}

/// What requests to a model carry. A key left unset falls back to the
/// provider's, then to `[analysis]`'s efforts and `llm::sampling`'s
/// temperature rule.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelSettings {
    /// Whether quack's `temperature` is sent.
    pub temperature: Option<bool>,
    /// Reasoning effort for chat turns.
    pub effort: Option<Effort>,
    /// Reasoning effort for background calls.
    pub background_effort: Option<Effort>,
}

impl ModelSettings {
    /// These settings, with `fallback`'s for each key left unset.
    #[must_use]
    pub fn or(self, fallback: Self) -> Self {
        Self {
            temperature: self.temperature.or(fallback.temperature),
            effort: self.effort.or(fallback.effort),
            background_effort: self.background_effort.or(fallback.background_effort),
        }
    }
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
    api: Option<BedrockApi>,
    region: Option<AwsRegion>,

    max_concurrent_requests: Option<RequestLimit>,
    headers: Option<BTreeMap<String, String>>,
    oauth: Option<OAuthConfig>,
    temperature: Option<bool>,
    effort: Option<Effort>,
    background_effort: Option<Effort>,
    #[serde(default)]
    models: BTreeMap<String, ModelSettings>,
}

impl TryFrom<RawProviderConfig> for ProviderConfig {
    type Error = Error;

    fn try_from(raw: RawProviderConfig) -> Result<Self> {
        let mode = raw.auth.unwrap_or_else(|| raw.provider_type.default_auth());
        let endpoint = raw.provider_type.bedrock_endpoint();
        if endpoint.is_some() != (mode == AuthMode::Aws) {
            return Err(Error::Config(match endpoint {
                Some(endpoint) => format!(
                    "type = \"{endpoint}\" signs with AWS credentials; use auth = \"aws\" or \
                     leave auth unset"
                ),
                None => String::from(
                    "auth = \"aws\" is only for type = \"bedrock\" and \"bedrock-mantle\"",
                ),
            }));
        }
        let bedrock_config = match endpoint {
            Some(endpoint) => Some(BedrockConfig::new(
                endpoint,
                raw.api,
                raw.region,
                raw.base_url.as_ref(),
            )?),
            None if raw.region.is_some() => {
                return Err(Error::Config(String::from(
                    "region is only for type = \"bedrock\" and \"bedrock-mantle\"",
                )));
            }
            None => None,
        };
        let openai_api = match endpoint {
            Some(_) => None,
            None => raw.provider_type.openai_api(raw.api)?,
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
                oauth.check()?;
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
        let provider = Self {
            provider_type: raw.provider_type,
            auth,
            base_url: raw.base_url,
            bedrock: bedrock_config,
            openai_api,
            max_concurrent_requests: raw.max_concurrent_requests,
            headers: raw.headers.filter(|headers| !headers.is_empty()),
            model_defaults: ModelSettings {
                temperature: raw.temperature,
                effort: raw.effort,
                background_effort: raw.background_effort,
            },
            models: raw.models,
        };
        provider.check_headers()?;
        Ok(provider)
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
            bedrock: provider_type
                .bedrock_endpoint()
                .map(|endpoint| BedrockConfig {
                    api: endpoint.default_api(),
                    region: None,
                }),
            openai_api: None,
            max_concurrent_requests: None,
            headers: None,
            model_defaults: ModelSettings::default(),
            models: BTreeMap::new(),
        }
    }

    /// `model`'s settings: each key from its `[providers.NAME.models."ID"]`
    /// entry, else from `[providers.NAME]`.
    #[must_use]
    pub fn model_settings(&self, model: &str) -> ModelSettings {
        self.models
            .get(model)
            .copied()
            .unwrap_or_default()
            .or(self.model_defaults)
    }

    /// Whether `headers` can be sent: every one a valid header that does
    /// not carry the credential, and an API that goes through rig.
    fn check_headers(&self) -> Result<()> {
        let converse = self
            .bedrock
            .as_ref()
            .is_some_and(|b| b.api == BedrockApi::Converse);
        if converse && self.headers.is_some() {
            return Err(Error::Config(String::from(
                "headers are not sent through api = \"converse\", which the AWS SDK calls; \
                 use api = \"chat-completions\" or \"responses\", or remove them",
            )));
        }
        self.header_map().map(drop)
    }

    /// `headers` as HTTP headers; empty when unset.
    ///
    /// # Errors
    ///
    /// Returns a `Config` error naming a header whose name or value HTTP
    /// does not allow.
    pub fn header_map(&self) -> Result<http::HeaderMap> {
        let mut map = http::HeaderMap::new();
        for (name, value) in self.headers.iter().flatten() {
            let header = http::HeaderName::from_bytes(name.as_bytes())
                .map_err(|_| Error::Config(format!("headers: \"{name}\" is not a header name")))?;
            if header == http::header::AUTHORIZATION || header == "x-api-key" {
                return Err(Error::Config(format!(
                    "headers: \"{name}\" carries the credential, which comes from auth"
                )));
            }
            let value = http::HeaderValue::from_str(value).map_err(|_| {
                Error::Config(format!(
                    "headers: the value of \"{name}\" is not a header value"
                ))
            })?;
            map.insert(header, value);
        }
        Ok(map)
    }

    /// The API a `type = "openai"` provider's chat model is called through:
    /// the `api` it sets, else [`Self::openai_default_api`].
    #[must_use]
    pub const fn openai_chat_api(&self) -> BedrockApi {
        match self.openai_api {
            Some(api) => api,
            None => self.openai_default_api(),
        }
    }

    /// A `type = "openai"` provider's API when `api` is unset: Responses for
    /// `OpenAI` itself (no `base_url`), Chat Completions for a compatible
    /// server at a `base_url`, since many of those implement nothing else.
    #[must_use]
    pub const fn openai_default_api(&self) -> BedrockApi {
        if self.base_url.is_some() {
            BedrockApi::ChatCompletions
        } else {
            BedrockApi::Responses
        }
    }

    /// The request limit when `max_concurrent_requests` is unset.
    #[must_use]
    pub const fn default_request_limit(&self) -> RequestLimit {
        RequestLimit(match self.provider_type {
            ProviderType::Ollama => NonZeroU32::MIN,
            ProviderType::Openai
            | ProviderType::Anthropic
            | ProviderType::Bedrock
            | ProviderType::BedrockMantle => NonZeroU32::MIN.saturating_add(7),
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
    /// Megabytes a compressed file (DOCX, PPTX, a zipped workbook) may
    /// inflate to while it is parsed. The upload limit counts compressed
    /// bytes only.
    pub max_decompressed_mb: u64,
    /// A table found inside a document (a PDF, DOCX, ODT, HTML, or
    /// Markdown table) with at least this many data rows is also loaded
    /// as a table of the workspace, owned by the document, so `run_sql`
    /// can query it. Zero loads none.
    pub table_rows_as_table: u32,
}

impl IngestionConfig {
    /// What one file's archive parts may inflate to, together.
    #[must_use]
    pub const fn decompression_budget(&self) -> DecompressionBudget {
        DecompressionBudget::megabytes(self.max_decompressed_mb)
    }
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
            // Twice the upload limit: a package of stored media at that
            // limit still has as much again for its XML.
            max_decompressed_mb: 1024,
            table_rows_as_table: 20,
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

/// Knowledge graph traversal and resolution (design doc 6.4 and 13), as
/// `[graph]` sets it and the graph module takes it.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
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
    /// What an ingest or import extracts into the graph once its document
    /// is ready.
    pub follow_ingest: FollowIngest,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_traversal_depth: 3,
            max_nodes: 200,
            merge_threshold: 0.08,
            auto_merge_threshold: 0.02,
            follow_ingest: FollowIngest::Off,
        }
    }
}

/// `[graph].follow_ingest`: whether a document that becomes ready is
/// extracted into the graph at once, and from what. `all` spends one
/// model call per chunk, which is why the default is `off`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FollowIngest {
    /// Nothing follows; `quack graph extract` builds the graph.
    #[default]
    Off,
    /// The document's mapped tables, deterministically: no model calls.
    Tables,
    /// Its mapped tables, then its chunks through the chat model.
    All,
}

text_enum!(FollowIngest, "follow_ingest", {
    Off => "off",
    Tables => "tables",
    All => "all",
});

impl FollowIngest {
    #[must_use]
    pub fn is_off(self) -> bool {
        self == Self::Off
    }

    #[must_use]
    pub fn includes_documents(self) -> bool {
        match self {
            Self::All => true,
            Self::Off | Self::Tables => false,
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
    /// When the session and sign-in cookies carry `Secure`.
    pub secure_cookies: SecureCookies,
    /// How long a streamed turn waits for a person to decide a write the
    /// agent wants to run before it goes on without it.
    pub permission_timeout_seconds: u32,
    /// How long a stopping server waits for its jobs to end and its
    /// requests to finish before it exits anyway.
    pub shutdown_grace_seconds: u32,
    /// Wrong passwords in a row before an account is locked for
    /// `login_lockout_minutes`; 0 never locks (the login rate limit alone).
    pub login_lockout_attempts: u32,
    /// How long a locked account refuses logins; the lock lifts by itself.
    pub login_lockout_minutes: u32,
    /// Address ranges of the proxies in front of this server. A request
    /// from one of them is attributed to the client its forwarded headers
    /// name (`quack_core::net`); from anyone else, to the peer itself.
    pub trusted_proxies: Vec<IpNet>,
    /// Sign-in through the organization's `OpenID` Connect issuer, beside
    /// password login.
    pub oidc: Option<OidcConfig>,
}

impl ServerConfig {
    /// The login lockout these settings describe.
    #[must_use]
    pub fn lockout(&self) -> Lockout {
        Lockout {
            attempts: self.login_lockout_attempts,
            minutes: self.login_lockout_minutes,
        }
    }
}

/// When wrong passwords lock an account, and for how long: Auth0's
/// brute-force protection model, a timed lock that lifts by itself.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Lockout {
    /// Wrong passwords in a row before the lock; 0 never locks.
    pub attempts: u32,
    pub minutes: u32,
}

impl Lockout {
    /// Whether `failed` wrong passwords in a row lock the account.
    #[must_use]
    pub fn locks_after(self, failed: u32) -> bool {
        self.attempts > 0 && failed >= self.attempts
    }
}

/// `[server.oidc]`: people sign in to `quack serve` with the organization's
/// `OpenID` Connect issuer (Authorization Code with PKCE). A first sign-in
/// creates a user with no workspace access.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "RawOidcConfig")]
pub struct OidcConfig {
    /// Issuer whose `/.well-known/openid-configuration` names the endpoints;
    /// ID tokens must name exactly this issuer.
    pub issuer_url: String,
    /// The client quack is at the issuer. Unset for a client quack
    /// registered itself (`quack auth register`, RFC 7591), whose id is
    /// read from the registration kept for the issuer.
    pub client_id: Option<String>,
    /// Environment variable holding the client secret, for an issuer that
    /// registers quack as a confidential client.
    pub client_secret_env: Option<String>,
    /// How quack authenticates at the token endpoint (and at the pushed
    /// authorization request endpoint): the secret in the body by default,
    /// or a signed assertion with `private_key_jwt`.
    pub client_auth: ClientAuth,
    /// Requested at sign-in; `openid` is required, and `offline_access` is
    /// what makes most issuers return the refresh token quack keeps.
    pub scopes: Vec<String>,
    /// Where the issuer sends the browser back: this server's public URL
    /// ending in [`OidcConfig::CALLBACK_PATH`].
    pub redirect_uri: String,
    /// The `aud` of access tokens the issuer makes for quack (Entra: the
    /// API's client ID, which a v2.0 token's `aud` always is; Okta: the
    /// custom authorization server's audience; Auth0: the API identifier). When set, the API and
    /// MCP accept those tokens as bearers and quack publishes its protected
    /// resource metadata (RFC 9728).
    pub audience: Option<String>,
    /// The claim that names a person, in ID tokens and access tokens alike.
    /// `sub` by default; Entra's `sub` differs per application, so Entra
    /// deployments set `oid`.
    pub subject_claim: String,
    /// The claim that lists the person's groups (`groups` at most issuers).
    /// Set, each sign-in grants and revokes the workspace roles
    /// `group_roles` gives those groups; unset, memberships are by hand.
    pub groups_claim: Option<String>,
}

impl OidcConfig {
    /// The route that starts a sign-in: the login page's button.
    pub const START_PATH: &str = "/auth/oidc";

    /// The route that receives the issuer's redirect.
    pub const CALLBACK_PATH: &str = "/auth/oidc/callback";

    /// The claim that names a person when `subject_claim` is unset.
    pub const DEFAULT_SUBJECT_CLAIM: &str = "sub";

    fn default_subject_claim() -> String {
        String::from(Self::DEFAULT_SUBJECT_CLAIM)
    }

    /// This server's public origin (`scheme://host[:port]`), from
    /// `redirect_uri`, which was checked to be an http(s) URL when read.
    #[must_use]
    pub fn public_url(&self) -> String {
        self.redirect_uri
            .strip_suffix(Self::CALLBACK_PATH)
            .unwrap_or(&self.redirect_uri)
            .to_owned()
    }

    /// Whether `redirect_uri`, and so the server's public URL, is https.
    #[must_use]
    pub fn is_https(&self) -> bool {
        oauth2::url::Url::parse(&self.redirect_uri).is_ok_and(|url| url.scheme() == "https")
    }

    /// The scopes requested when `scopes` is unset.
    #[must_use]
    pub fn default_scopes() -> Vec<String> {
        ["openid", "profile", "email", "offline_access"]
            .map(String::from)
            .to_vec()
    }
}

/// `[server.oidc]` as the file writes it, before its values are checked.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOidcConfig {
    issuer_url: String,
    client_id: Option<String>,
    client_secret_env: Option<String>,
    #[serde(default)]
    client_auth: ClientAuth,
    #[serde(default = "OidcConfig::default_scopes")]
    scopes: Vec<String>,
    redirect_uri: String,
    audience: Option<String>,
    #[serde(default = "OidcConfig::default_subject_claim")]
    subject_claim: String,
    groups_claim: Option<String>,
}

impl TryFrom<RawOidcConfig> for OidcConfig {
    type Error = Error;

    fn try_from(raw: RawOidcConfig) -> Result<Self> {
        if raw.issuer_url.trim().is_empty() {
            return Err(Error::Config(String::from(
                "[server.oidc] needs issuer_url",
            )));
        }
        raw.client_auth
            .check("[server.oidc]", raw.client_secret_env.as_ref())?;
        raw.client_auth
            .check_client_id("[server.oidc]", raw.client_id.as_deref())?;
        if !raw.scopes.iter().any(|scope| scope == "openid") {
            return Err(Error::Config(String::from(
                "[server.oidc].scopes must include \"openid\"",
            )));
        }
        let redirect = oauth2::url::Url::parse(&raw.redirect_uri).map_err(|e| {
            Error::Config(format!(
                "[server.oidc].redirect_uri '{}' is not a URL: {e}",
                raw.redirect_uri
            ))
        })?;
        if !matches!(redirect.scheme(), "http" | "https")
            || redirect.path() != Self::CALLBACK_PATH
            || redirect.query().is_some()
        {
            return Err(Error::Config(format!(
                "[server.oidc].redirect_uri must be this server's http(s) URL ending in {}, e.g. https://quack.example.com{}",
                Self::CALLBACK_PATH,
                Self::CALLBACK_PATH
            )));
        }
        let audience = match raw.audience.as_deref().map(str::trim) {
            Some("") => {
                return Err(Error::Config(String::from(
                    "[server.oidc].audience is empty; remove it or name the tokens' aud",
                )));
            }
            other => other.map(str::to_owned),
        };
        let subject_claim = raw.subject_claim.trim().to_owned();
        if subject_claim.is_empty() {
            return Err(Error::Config(String::from(
                "[server.oidc].subject_claim is empty; the default is \"sub\"",
            )));
        }
        Ok(Self {
            issuer_url: raw.issuer_url.trim().trim_end_matches('/').to_owned(),
            client_id: raw.client_id.map(|id| id.trim().to_owned()),
            client_secret_env: raw.client_secret_env,
            client_auth: raw.client_auth,
            scopes: raw.scopes,
            redirect_uri: raw.redirect_uri,
            audience,
            subject_claim,
            groups_claim: raw
                .groups_claim
                .map(|c| c.trim().to_owned())
                .filter(|c| !c.is_empty()),
        })
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: String::from("127.0.0.1:8080"),
            local: false,
            workers_per_workspace: 1,
            session_max_age_hours: 12,
            session_idle_minutes: 120,
            secure_cookies: SecureCookies::Auto,
            permission_timeout_seconds: 300,
            shutdown_grace_seconds: 20,
            login_lockout_attempts: 5,
            login_lockout_minutes: 15,
            trusted_proxies: Vec::new(),
            oidc: None,
        }
    }
}

impl ServerConfig {
    /// A session's absolute lifetime.
    #[must_use]
    pub fn session_max_age(&self) -> Duration {
        Duration::from_secs(u64::from(self.session_max_age_hours).saturating_mul(3600))
    }

    /// How long a write waits for a person's decision, at least one second.
    #[must_use]
    pub fn permission_timeout(&self) -> Duration {
        Duration::from_secs(u64::from(self.permission_timeout_seconds.max(1)))
    }

    /// How long a stopping server waits for jobs and requests to end.
    #[must_use]
    pub fn shutdown_grace(&self) -> Duration {
        Duration::from_secs(u64::from(self.shutdown_grace_seconds))
    }

    /// How long a session may sit unused before it is dropped.
    #[must_use]
    pub fn session_idle(&self) -> Duration {
        Duration::from_secs(u64::from(self.session_idle_minutes).saturating_mul(60))
    }

    /// Whether a cookie set on a request that arrived on loopback must
    /// still carry `Secure`. Off loopback it always does. On loopback it
    /// does when the operator said so, or when the server's public URL is
    /// known to be https: a TLS-terminating proxy on the same host connects
    /// over loopback, and the browser behind it is on https. Nothing the
    /// request says about itself (`X-Forwarded-Proto`) is trusted for this.
    #[must_use]
    pub fn secure_cookies_on_loopback(&self) -> bool {
        match self.secure_cookies {
            SecureCookies::Always => true,
            SecureCookies::Auto => self.oidc.as_ref().is_some_and(OidcConfig::is_https),
        }
    }
}

/// `[server].secure_cookies`: when the session cookie and the sign-in
/// state cookie carry `Secure`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SecureCookies {
    /// Unless the request arrived on loopback and no https public URL is
    /// configured (`[server.oidc].redirect_uri`): plain HTTP on a laptop
    /// keeps working.
    Auto,
    /// On every cookie, for a same-host TLS proxy the server cannot
    /// otherwise tell apart from a local browser.
    Always,
}

text_enum!(SecureCookies, "secure_cookies value", {
    Auto => "auto",
    Always => "always",
});

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

/// The embedding model, the width of its vectors, and overrides for the
/// input prefixes it gets for each role (`quack_core::embedding`). An unset
/// prefix keeps the built-in one for the model's family; an empty string
/// sends that role unprefixed. Changing the model, the width, or a prefix
/// changes the embedding profile: stored vectors stop being searched until
/// `quack embeddings refresh` brings them up to date.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// `PROVIDER/MODEL` used for embeddings. Unset means documents are stored
    /// without vectors and `search_documents` is unavailable.
    pub model: Option<ModelSpec>,
    /// Width of the vectors `model` produces; required with it.
    pub dimension: Option<Dimension>,

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
    /// Reranking after hybrid fusion: `none` (the default); `model`, the
    /// chat model ordering the candidates listwise; or `reranker`, the
    /// dedicated rerank model `rerank_model` names scoring each one.
    pub rerank: RerankMode,
    /// Candidates fetched for reranking before the top `k` are kept.
    pub rerank_candidates: u32,
    /// `PROVIDER/MODEL` of a rerank model served at an OpenAI-compatible
    /// `/rerank` endpoint (vLLM, llama.cpp, Text Embeddings Inference),
    /// for `rerank = "reranker"`.
    pub rerank_model: Option<ModelSpec>,
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
            rerank_model: None,
        }
    }
}

/// Which reranker, if any, orders retrieval candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RerankMode {
    None,
    Model,
    Reranker,
}

text_enum!(RerankMode, "rerank mode", {
    None => "none",
    Model => "model",
    Reranker => "reranker",
});

/// How long a reasoning model thinks before it answers (`[analysis].effort`
/// and `background_effort`). `llm::sampling` sends it in the form each
/// provider and model family takes, and refuses a level the model lacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

text_enum!(Effort, "effort", {
    None => "none",
    Minimal => "minimal",
    Low => "low",
    Medium => "medium",
    High => "high",
    Xhigh => "xhigh",
    Max => "max",
});

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnalysisConfig {
    pub max_query_rows: u32,
    /// Rows of a `run_sql` or `create_chart` result kept on the step for
    /// the transcript; the model sees up to `max_query_rows`.
    pub step_result_rows: u32,
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
    /// Reasoning effort for chat turns. Unset: the model's own default.
    pub effort: Option<Effort>,
    /// Reasoning effort for background calls: graph extraction, the
    /// ontology's document pass, reranking, and history summaries. Unset:
    /// the model's default.
    pub background_effort: Option<Effort>,
    /// Replace the turns that fall outside `history_token_budget` with a
    /// summary the chat model writes, kept in the workspace, instead of
    /// dropping them. Off by default: each new summary is one more model
    /// call before a turn.
    pub compact_history: bool,
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
            step_result_rows: 50,
            query_timeout_seconds: 30,
            memory_limit_mb: 256,
            threads: 4,
            max_turns: 15,
            history_token_budget: Tokens::new(32_000),
            max_context_tokens: Tokens::new(32_768),
            extraction_timeout_seconds: 120,
            extraction_concurrency: 1,
            reader_pool_size: 4,
            effort: None,
            background_effort: None,
            compact_history: false,
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
                    "[embedding].model '{embed}': anthropic does not serve embeddings"
                )));
            }
            if embed.provider.provider_type == ProviderType::BedrockMantle {
                return Err(Error::Config(format!(
                    "[embedding].model '{embed}': bedrock-mantle serves no embeddings; use a \
                     type = \"bedrock\" provider"
                )));
            }
            self.embedding_dimension()?;
        }
        self.rerank_model_ref()?;
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

    /// What requests to `model` carry: its own settings, else its
    /// provider's, else `[analysis]`'s efforts.
    #[must_use]
    pub fn model_settings(&self, model: ModelRef<'_>) -> ModelSettings {
        model
            .provider
            .model_settings(model.model)
            .or(ModelSettings {
                temperature: None,
                effort: self.analysis.effort,
                background_effort: self.analysis.background_effort,
            })
    }

    /// `provider/model` for status lines, or a placeholder.
    #[must_use]
    pub fn chat_model_label(&self) -> String {
        self.chat_model_ref()
            .map_or_else(|_| String::from("no chat model"), |m| m.to_string())
    }

    /// The dedicated rerank model, when `[retrieval].rerank = "reranker"`.
    ///
    /// # Errors
    ///
    /// Returns an error if that mode is set without `rerank_model`, or the
    /// model's provider is unknown or has no `/rerank` endpoint.
    pub fn rerank_model_ref(&self) -> Result<Option<ModelRef<'_>>> {
        if self.retrieval.rerank != RerankMode::Reranker {
            return Ok(None);
        }
        let spec = self.retrieval.rerank_model.as_ref().ok_or_else(|| {
            Error::Config(String::from(
                "[retrieval].rerank = \"reranker\" needs rerank_model = \"PROVIDER/MODEL\"",
            ))
        })?;
        let model = self.resolve_model("[retrieval].rerank_model", spec)?;
        match model.provider.provider_type {
            ProviderType::Openai if model.provider.base_url.is_none() => {
                Err(Error::Config(format!(
                    "[retrieval].rerank_model '{model}': set base_url under [providers.{}] to the \
                 server that serves /rerank",
                    model.provider_name
                )))
            }
            ProviderType::Openai => Ok(Some(model)),
            other @ (ProviderType::Ollama
            | ProviderType::Anthropic
            | ProviderType::Bedrock
            | ProviderType::BedrockMantle) => Err(Error::Config(format!(
                "[retrieval].rerank_model '{model}': {other} has no rerank endpoint; serve the \
                 model with vLLM, llama.cpp, or Text Embeddings Inference and add that server \
                 as a type = \"openai\" provider with its base_url, or set rerank = \"model\" \
                 to rank with the chat model"
            ))),
        }
    }

    /// The embedding model, if configured.
    ///
    /// # Errors
    ///
    /// Returns an error if `[embedding].model` names an unknown provider.
    pub fn embedding_model_ref(&self) -> Result<Option<ModelRef<'_>>> {
        self.embedding
            .model
            .as_ref()
            .map(|spec| self.resolve_model("[embedding].model", spec))
            .transpose()
    }

    /// The width of the embedding model's vectors: `validate` requires it
    /// whenever a model is set.
    ///
    /// # Errors
    ///
    /// Returns an error when `[embedding].dimension` is unset.
    pub fn embedding_dimension(&self) -> Result<Dimension> {
        self.embedding.dimension.ok_or_else(|| {
            Error::Config(String::from(
                "[embedding].model is set but [embedding].dimension is not; set it to the \
                 width of the model's vectors",
            ))
        })
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

    /// Where `quack serve` keeps a queued upload's bytes until its job
    /// runs: inside the workspace, since they are its content.
    #[must_use]
    pub fn workspace_uploads_dir(&self, workspace_id: &str) -> PathBuf {
        self.workspace_dir(workspace_id).join("uploads")
    }

    /// Ensure the data directory and its subdirectories exist, and that the
    /// data directory is private to the user (0700 on Unix): it holds every
    /// workspace's content, the control database, and the vault key file.
    /// One that group or others can reach, such as one made by an older
    /// build, is tightened with a warning.
    ///
    /// # Errors
    ///
    /// Returns an error if the directories cannot be created or the data
    /// directory's mode cannot be read or set.
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        let data_dir = &self.general.data_dir;
        #[cfg(unix)]
        let existed = data_dir.exists();
        std::fs::create_dir_all(data_dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            use tracing::warn;
            let mode = std::fs::metadata(data_dir)?.permissions().mode() & 0o777;
            if mode & 0o077 != 0 {
                std::fs::set_permissions(data_dir, std::fs::Permissions::from_mode(0o700))?;
                if existed {
                    warn!(
                        "{} was open to other users (mode {mode:o}); set it to 700",
                        data_dir.display()
                    );
                }
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
mod tests;
