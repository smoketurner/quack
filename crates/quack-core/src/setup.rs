//! `quack init`: the model providers this machine can reach, and the config
//! file that names the ones a person chose. Nothing here writes a file; the
//! command line does, after `doctor` passes the text this module builds.

use std::path::PathBuf;
use std::time::Duration;

use crate::config::{BaseUrl, Config, ModelSpec, ProviderName, ProviderType};
use crate::embedding::Dimension;
use crate::error::{Error, Result};
use crate::llm::egress::Egress;
use crate::llm::{Embeddings, OllamaCapability, OllamaModel, ProviderModels};

/// How long discovery waits for a local Ollama to list its models.
const OLLAMA_TIMEOUT: Duration = Duration::from_secs(10);

/// The provider types `quack init` sets up, in the order it prefers them:
/// local first, so nothing leaves the machine unless the person chooses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Ollama,
    Anthropic,
    OpenAi,
    Bedrock,
}

impl ProviderKind {
    pub const ALL: [Self; 4] = [Self::Ollama, Self::Anthropic, Self::OpenAi, Self::Bedrock];

    /// The `[providers.NAME]` the file gets, which is also the `PROVIDER`
    /// in `PROVIDER/MODEL`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::Bedrock => "bedrock",
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Ollama => "Ollama",
            Self::Anthropic => "Anthropic",
            Self::OpenAi => "OpenAI",
            Self::Bedrock => "Amazon Bedrock",
        }
    }

    /// The kind whose `name` is `name`.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }

    /// The environment variable that holds the API key, for the kinds that
    /// take one. The file names the variable, never the key.
    #[must_use]
    pub const fn key_env(self) -> Option<&'static str> {
        match self {
            Self::Anthropic => Some("ANTHROPIC_API_KEY"),
            Self::OpenAi => Some("OPENAI_API_KEY"),
            Self::Ollama | Self::Bedrock => None,
        }
    }

    /// The embedding model a hosted provider is set up with, and its width
    /// as the provider documents it. Ollama's come from its listing;
    /// Anthropic serves none.
    #[must_use]
    pub const fn hosted_embedding(self) -> Option<(&'static str, Dimension)> {
        match self {
            Self::OpenAi => Some(("text-embedding-3-small", Dimension::new(1536))),
            Self::Bedrock => Some(("amazon.titan-embed-text-v2:0", Dimension::new(1024))),
            Self::Ollama | Self::Anthropic => None,
        }
    }

    /// The chat model offered for a provider that lists none
    /// (bedrock-runtime).
    #[must_use]
    pub const fn unlisted_chat_model(self) -> Option<&'static str> {
        match self {
            Self::Bedrock => Some("us.anthropic.claude-opus-5-5"),
            Self::Ollama | Self::Anthropic | Self::OpenAi => None,
        }
    }

    /// Whether listing this provider's models sends a credential off the
    /// machine.
    #[must_use]
    pub const fn is_hosted(self) -> bool {
        match self {
            Self::Ollama => false,
            Self::Anthropic | Self::OpenAi | Self::Bedrock => true,
        }
    }
}

/// What discovery found for one provider kind.
#[derive(Debug, Clone)]
pub enum Found {
    /// A running Ollama and the models it has pulled.
    Ollama {
        base_url: BaseUrl,
        models: Vec<OllamaModel>,
    },
    /// A hosted provider whose credential this environment holds; the text
    /// says which.
    Hosted {
        kind: ProviderKind,
        credential: String,
    },
    /// Not usable here, and why.
    Absent { kind: ProviderKind, reason: String },
}

impl Found {
    #[must_use]
    pub const fn kind(&self) -> ProviderKind {
        match self {
            Self::Ollama { .. } => ProviderKind::Ollama,
            Self::Hosted { kind, .. } | Self::Absent { kind, .. } => *kind,
        }
    }

    #[must_use]
    pub const fn is_present(&self) -> bool {
        match self {
            Self::Ollama { .. } | Self::Hosted { .. } => true,
            Self::Absent { .. } => false,
        }
    }
}

impl std::fmt::Display for Found {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ollama { base_url, models } => {
                write!(f, "✓ Ollama at {base_url}: {} models", models.len())
            }
            Self::Hosted { kind, credential } => write!(f, "✓ {}: {credential}", kind.label()),
            Self::Absent { kind, reason } => write!(f, "✗ {}: {reason}", kind.label()),
        }
    }
}

/// The environment discovery reads, taken once so tests can supply it.
#[derive(Debug, Clone, Default)]
pub struct Environment {
    pub ollama_host: Option<String>,
    pub anthropic_key: Option<String>,
    pub openai_key: Option<String>,
    pub aws_profile: Option<String>,
    pub aws_access_key: Option<String>,
    /// `~/.aws/config` or `~/.aws/credentials`, when either exists.
    pub aws_files: Option<PathBuf>,
}

impl Environment {
    #[must_use]
    pub fn from_env() -> Self {
        let set = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
        let aws_files = dirs::home_dir().and_then(|home| {
            ["config", "credentials"]
                .into_iter()
                .map(|file| home.join(".aws").join(file))
                .find(|path| path.is_file())
        });
        Self {
            ollama_host: set("OLLAMA_HOST"),
            anthropic_key: set("ANTHROPIC_API_KEY"),
            openai_key: set("OPENAI_API_KEY"),
            aws_profile: set("AWS_PROFILE"),
            aws_access_key: set("AWS_ACCESS_KEY_ID"),
            aws_files,
        }
    }

    /// Where Ollama listens: `OLLAMA_HOST` as Ollama reads it (a bare
    /// `host:port`, or a URL; port 11434 when it names none), else the
    /// default.
    ///
    /// # Errors
    ///
    /// Returns an error when `OLLAMA_HOST` is not a host or URL.
    pub fn ollama_base_url(&self) -> Result<BaseUrl> {
        let Some(host) = &self.ollama_host else {
            return Ok(ProviderType::OLLAMA_BASE_URL);
        };
        let with_scheme = if host.contains("://") {
            host.clone()
        } else {
            format!("http://{host}")
        };
        let mut url = reqwest::Url::parse(&with_scheme)
            .map_err(|e| Error::Config(format!("OLLAMA_HOST \"{host}\": {e}")))?;
        if url.port().is_none() && url.scheme() == "http" {
            url.set_port(Some(11434))
                .map_err(|()| Error::Config(format!("OLLAMA_HOST \"{host}\" takes no port")))?;
        }
        BaseUrl::try_from(url.as_str().trim_end_matches('/').to_owned())
    }

    fn hosted(&self, kind: ProviderKind) -> Found {
        let present = match kind {
            ProviderKind::Anthropic => self
                .anthropic_key
                .as_ref()
                .map(|_| "ANTHROPIC_API_KEY is set"),
            ProviderKind::OpenAi => self.openai_key.as_ref().map(|_| "OPENAI_API_KEY is set"),
            ProviderKind::Bedrock => {
                return match (&self.aws_profile, &self.aws_access_key, &self.aws_files) {
                    (Some(profile), _, _) => Found::Hosted {
                        kind,
                        credential: format!("AWS profile \"{profile}\""),
                    },
                    (None, Some(_), _) => Found::Hosted {
                        kind,
                        credential: String::from("AWS_ACCESS_KEY_ID is set"),
                    },
                    (None, None, Some(path)) => Found::Hosted {
                        kind,
                        credential: format!("profiles in {}", path.display()),
                    },
                    (None, None, None) => Found::Absent {
                        kind,
                        reason: String::from("no AWS profile or credentials"),
                    },
                };
            }
            ProviderKind::Ollama => None,
        };
        match (present, kind.key_env()) {
            (Some(credential), _) => Found::Hosted {
                kind,
                credential: credential.to_owned(),
            },
            (None, Some(env)) => Found::Absent {
                kind,
                reason: format!("{env} is not set"),
            },
            (None, None) => Found::Absent {
                kind,
                reason: String::from("not found"),
            },
        }
    }
}

/// Every provider kind, found or not, in [`ProviderKind::ALL`]'s order.
#[derive(Debug, Clone)]
pub struct Discovery(pub Vec<Found>);

impl Discovery {
    /// Look for each provider. Only the local Ollama is contacted; hosted
    /// providers are judged by the credentials the environment holds, so
    /// no key leaves the machine here.
    pub async fn run(env: &Environment) -> Self {
        let ollama = match env.ollama_base_url() {
            Ok(base_url) => Self::ollama(base_url).await,
            Err(e) => Found::Absent {
                kind: ProviderKind::Ollama,
                reason: e.to_string(),
            },
        };
        let mut found = vec![ollama];
        for kind in [
            ProviderKind::Anthropic,
            ProviderKind::OpenAi,
            ProviderKind::Bedrock,
        ] {
            found.push(env.hosted(kind));
        }
        Self(found)
    }

    async fn ollama(base_url: BaseUrl) -> Found {
        let kind = ProviderKind::Ollama;
        let plan = SetupPlan {
            ollama_base_url: Some(base_url.clone()),
            ..SetupPlan::default()
        };
        let config = match plan.provider_config(kind) {
            Ok(config) => config,
            Err(e) => {
                return Found::Absent {
                    kind,
                    reason: e.to_string(),
                };
            }
        };
        let Ok(name) = ProviderName::try_from(kind.name().to_owned()) else {
            return Found::Absent {
                kind,
                reason: String::from("invalid provider name"),
            };
        };
        let listed = Egress::scope(Some(Egress::NoWorkspace), async {
            tokio::time::timeout(OLLAMA_TIMEOUT, OllamaModel::list(&config, &name)).await
        })
        .await;
        match listed {
            Ok(Ok(models)) => Found::Ollama { base_url, models },
            Ok(Err(e)) => {
                tracing::debug!(error = %e, "ollama did not list its models");
                Found::Absent {
                    kind,
                    reason: format!("not running at {base_url}"),
                }
            }
            Err(_) => Found::Absent {
                kind,
                reason: format!(
                    "no answer from {base_url} within {} seconds",
                    OLLAMA_TIMEOUT.as_secs()
                ),
            },
        }
    }

    #[must_use]
    pub fn get(&self, kind: ProviderKind) -> Option<&Found> {
        self.0.iter().find(|found| found.kind() == kind)
    }

    /// The Ollama models that can do `capability`, in the server's order.
    #[must_use]
    pub fn ollama_models(&self, capability: OllamaCapability) -> Vec<&OllamaModel> {
        match self.get(ProviderKind::Ollama) {
            Some(Found::Ollama { models, .. }) => models
                .iter()
                .filter(|model| model.can(capability))
                .collect(),
            Some(Found::Hosted { .. } | Found::Absent { .. }) | None => Vec::new(),
        }
    }

    /// Where the found Ollama listens, when it differs from the default.
    #[must_use]
    pub fn ollama_base_url(&self) -> Option<BaseUrl> {
        match self.get(ProviderKind::Ollama) {
            Some(Found::Ollama { base_url, .. }) if *base_url != ProviderType::OLLAMA_BASE_URL => {
                Some(base_url.clone())
            }
            Some(Found::Ollama { .. } | Found::Hosted { .. } | Found::Absent { .. }) | None => None,
        }
    }
}

/// A byte count as people read it: `13 GB`, `639 MB`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteSize(pub u64);

impl std::fmt::Display for ByteSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const UNITS: [(u64, &str); 3] = [(1_000_000_000, "GB"), (1_000_000, "MB"), (1_000, "KB")];
        for (scale, unit) in UNITS {
            if self.0 < scale {
                continue;
            }
            let whole = self.0.checked_div(scale).unwrap_or(0);
            if whole >= 10 {
                return write!(f, "{whole} {unit}");
            }
            let tenth = self
                .0
                .checked_rem(scale)
                .and_then(|rest| rest.checked_mul(10))
                .and_then(|rest| rest.checked_div(scale))
                .unwrap_or(0);
            return write!(f, "{whole}.{tenth} {unit}");
        }
        write!(f, "{} B", self.0)
    }
}

/// The embedding model chosen, with its width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingChoice {
    pub model: ModelSpec,
    pub dimension: Dimension,
}

/// What `quack init` writes: a chat model, an embedding model, and the
/// providers they name.
#[derive(Debug, Clone, Default)]
pub struct SetupPlan {
    pub chat_model: Option<ModelSpec>,
    pub embedding: Option<EmbeddingChoice>,
    /// Ollama's address, when it is not the default.
    pub ollama_base_url: Option<BaseUrl>,
}

impl SetupPlan {
    /// The provider kinds the plan's models name, each once, in
    /// [`ProviderKind::ALL`]'s order.
    ///
    /// # Errors
    ///
    /// Returns an error when a model names a provider `quack init` does not
    /// set up.
    pub fn providers(&self) -> Result<Vec<ProviderKind>> {
        let mut named = Vec::new();
        let models = [
            self.chat_model.as_ref(),
            self.embedding.as_ref().map(|e| &e.model),
        ];
        for model in models.into_iter().flatten() {
            let provider = model.provider();
            let Some(kind) = ProviderKind::ALL
                .into_iter()
                .find(|kind| *provider == kind.name())
            else {
                return Err(Error::Config(format!(
                    "{model}: quack init sets up {}; edit config.toml for other providers",
                    ProviderKind::ALL.map(ProviderKind::name).join(", ")
                )));
            };
            named.push(kind);
        }
        Ok(ProviderKind::ALL
            .into_iter()
            .filter(|kind| named.contains(kind))
            .collect())
    }

    fn provider_table(&self, kind: ProviderKind) -> toml::Table {
        let mut table = toml::Table::new();
        table.insert("type".into(), kind.name().into());
        if let Some(env) = kind.key_env() {
            table.insert("auth".into(), "api-key".into());
            table.insert("api_key_env".into(), env.into());
        }
        if let (ProviderKind::Ollama, Some(base_url)) = (kind, &self.ollama_base_url) {
            table.insert("base_url".into(), base_url.to_string().into());
        }
        table
    }

    /// The config file's text.
    ///
    /// # Errors
    ///
    /// Returns an error when a model names a provider `quack init` does not
    /// set up.
    pub fn toml(&self) -> Result<String> {
        let mut file = toml::Table::new();
        if let Some(chat) = &self.chat_model {
            let mut general = toml::Table::new();
            general.insert("chat_model".into(), chat.to_string().into());
            file.insert("general".into(), general.into());
        }
        if let Some(embedding) = &self.embedding {
            let mut section = toml::Table::new();
            section.insert("model".into(), embedding.model.to_string().into());
            section.insert(
                "dimension".into(),
                i64::from(embedding.dimension.get()).into(),
            );
            file.insert("embedding".into(), section.into());
        }
        let mut providers = toml::Table::new();
        for kind in self.providers()? {
            providers.insert(kind.name().into(), self.provider_table(kind).into());
        }
        if !providers.is_empty() {
            file.insert("providers".into(), providers.into());
        }
        let body = toml::to_string(&file)
            .map_err(|e| Error::Config(format!("cannot write the config: {e}")))?;
        Ok(format!(
            "# Written by `quack init`. `quack config` lists every other setting.\n\n{body}"
        ))
    }

    /// The configuration [`Self::toml`] describes, checked as `quack` reads
    /// a file.
    ///
    /// # Errors
    ///
    /// Returns the error any other command would give for the file.
    pub fn config(&self) -> Result<Config> {
        Config::parse(&self.toml()?)
    }

    /// A configuration holding only `kind`'s provider, for the requests
    /// `quack init` sends before anything is chosen. It never retries: a
    /// provider that does not answer is reported at once.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider section does not validate.
    pub fn provider_config(&self, kind: ProviderKind) -> Result<Config> {
        let mut table = self.provider_table(kind);
        table.insert("max_retries".into(), 0.into());
        let mut providers = toml::Table::new();
        providers.insert(kind.name().into(), table.into());
        let mut file = toml::Table::new();
        file.insert("providers".into(), providers.into());
        let text = toml::to_string(&file)
            .map_err(|e| Error::Config(format!("cannot write the config: {e}")))?;
        Config::parse(&text)
    }

    /// The chat models a hosted provider lists. This sends the provider's
    /// credential to it, so the caller asks first.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider refuses or cannot be reached.
    pub async fn list_hosted(&self, kind: ProviderKind) -> Result<ProviderModels> {
        let config = self.provider_config(kind)?;
        let name = ProviderName::try_from(kind.name().to_owned())?;
        Egress::scope(
            Some(Egress::NoWorkspace),
            ProviderModels::fetch(&config, &name),
        )
        .await
    }

    /// The width of an Ollama embedding model's vectors, from one request.
    /// Ollama is local and is sent no width, so the reply is the model's own.
    ///
    /// # Errors
    ///
    /// Returns an error when the model does not answer.
    pub async fn ollama_width(&self, model: &str) -> Result<Dimension> {
        let mut config = self.provider_config(ProviderKind::Ollama)?;
        config.embedding.model = Some(format!("{}/{model}", ProviderKind::Ollama.name()).parse()?);
        config.embedding.dimension = Some(Dimension::new(1));
        let width = Egress::scope(Some(Egress::NoWorkspace), async {
            let Some(embedder) = Embeddings::from_config(&config).await? else {
                return Err(Error::Config(String::from("no embedding model")));
            };
            tokio::time::timeout(OLLAMA_TIMEOUT, embedder.measure_width())
                .await
                .map_err(|_| Error::Embedding(format!("{model} did not answer in time")))?
        })
        .await?;
        u32::try_from(width)
            .map(Dimension::new)
            .map_err(|e| Error::Embedding(format!("{model}: width {width}: {e}")))
    }
}

#[cfg(test)]
mod tests;
