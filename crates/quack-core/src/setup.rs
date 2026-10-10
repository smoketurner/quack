//! `quack init`: the model providers this machine can reach, and the config
//! file that names the ones a person chose. Nothing here writes a file; the
//! command line does, after `doctor` passes the text this module builds.

use std::path::PathBuf;
use std::time::Duration;

use toml_edit::{DocumentMut, Item, Table, TableLike, Value, value};

use crate::config::{BaseUrl, Config, ProviderName, ProviderType};
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
        let aws_files = dirs::home_dir().and_then(|home| {
            ["config", "credentials"]
                .into_iter()
                .map(|file| home.join(".aws").join(file))
                .find(|path| path.is_file())
        });
        Self::from_lookup(|name| std::env::var(name).ok(), aws_files)
    }

    /// The environment `lookup` reports. A variable that is empty or only
    /// whitespace counts as unset, as `export ANTHROPIC_API_KEY=` leaves it.
    #[must_use]
    pub fn from_lookup(
        lookup: impl Fn(&str) -> Option<String>,
        aws_files: Option<PathBuf>,
    ) -> Self {
        let set = |name: &str| {
            lookup(name)
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        };
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

/// A model on one of the providers `quack init` sets up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice {
    pub kind: ProviderKind,
    pub model: String,
}

/// The embedding model chosen, with its width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingChoice {
    pub choice: Choice,
    pub dimension: Dimension,
}

/// The chat, embedding, and decision models a config file names now, as
/// written.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[expect(
    clippy::struct_field_names,
    reason = "each field is a model, named for the config key it reads"
)]
pub struct Current {
    pub chat_model: Option<String>,
    pub embedding_model: Option<String>,
    pub decision_model: Option<String>,
}

impl Current {
    /// What `text` names.
    ///
    /// # Errors
    ///
    /// Returns an error when `text` is not TOML, which `quack init` will
    /// not edit.
    pub fn of(text: &str) -> Result<Self> {
        let doc = Self::parse(text)?;
        let read = |section: &str, key: &str| {
            doc.get(section)
                .and_then(|section| section.get(key))
                .and_then(Item::as_str)
                .map(str::to_owned)
        };
        Ok(Self {
            chat_model: read("general", "chat_model"),
            embedding_model: read("embedding", "model"),
            decision_model: read("decision", "model"),
        })
    }

    fn parse(text: &str) -> Result<DocumentMut> {
        text.parse::<DocumentMut>()
            .map_err(|e| Error::Config(format!("not valid TOML: {e}")))
    }
}

/// What `quack init` writes: a chat model, an embedding model, a decision
/// model, and the providers they name. A model left `None` keeps what the
/// file has.
#[derive(Debug, Clone, Default)]
pub struct SetupPlan {
    pub chat: Option<Choice>,
    pub embedding: Option<EmbeddingChoice>,
    /// The model that labels the text of a table's rows.
    pub decision: Option<Choice>,
    /// Ollama's address, when it is not the default.
    pub ollama_base_url: Option<BaseUrl>,
}

/// The text a new config file starts from.
const NEW_FILE: &str = "# Written by `quack init`. `quack config` lists every other setting.\n";

impl SetupPlan {
    /// Whether the plan changes anything.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.chat.is_none() && self.embedding.is_none() && self.decision.is_none()
    }

    /// The provider kinds the plan's models name, each once, in
    /// [`ProviderKind::ALL`]'s order.
    #[must_use]
    pub fn providers(&self) -> Vec<ProviderKind> {
        let named = [
            self.chat.as_ref().map(|c| c.kind),
            self.embedding.as_ref().map(|e| e.choice.kind),
            self.decision.as_ref().map(|d| d.kind),
        ];
        ProviderKind::ALL
            .into_iter()
            .filter(|kind| named.contains(&Some(*kind)))
            .collect()
    }

    fn provider_table(&self, kind: ProviderKind) -> Table {
        let mut table = Table::new();
        table.insert("type", value(kind.name()));
        if let Some(env) = kind.key_env() {
            table.insert("auth", value("api-key"));
            table.insert("api_key_env", value(env));
        }
        if let (ProviderKind::Ollama, Some(base_url)) = (kind, &self.ollama_base_url) {
            table.insert("base_url", value(base_url.to_string()));
        }
        table
    }

    /// Whether the provider section `table` is the one `kind` would be: the
    /// same type at the same address. Its other settings are the file's.
    fn reuses(&self, kind: ProviderKind, table: &dyn TableLike) -> bool {
        let text = |key: &str| table.get(key).and_then(Item::as_str);
        if text("type") != Some(kind.name()) {
            return false;
        }
        let base_url = text("base_url").map(|url| url.trim_end_matches('/'));
        match kind {
            ProviderKind::Ollama => {
                let wanted = self
                    .ollama_base_url
                    .clone()
                    .unwrap_or(ProviderType::OLLAMA_BASE_URL);
                base_url.unwrap_or(ProviderType::OLLAMA_BASE_URL.as_str())
                    == wanted.as_str().trim_end_matches('/')
            }
            ProviderKind::Anthropic | ProviderKind::OpenAi => base_url.is_none(),
            ProviderKind::Bedrock => true,
        }
    }

    /// `existing` (a config file's text, or `None` for no file) with the
    /// plan's models set, and a section added for each provider they name
    /// that the file lacks. Every other key, value, and comment stays.
    /// Also returns one line per change, for the person to confirm.
    ///
    /// # Errors
    ///
    /// Returns an error when `existing` is not TOML, or when a section
    /// `quack init` would add already exists for another provider.
    pub fn apply(&self, existing: Option<&str>) -> Result<(String, Vec<String>)> {
        let mut doc = Current::parse(existing.unwrap_or_default())?;
        let mut names = Vec::new();
        let mut added = Vec::new();
        let providers = doc.get("providers").and_then(Item::as_table_like);
        for kind in self.providers() {
            let reused = providers.and_then(|providers| {
                providers
                    .iter()
                    .find(|(_, item)| {
                        item.as_table_like()
                            .is_some_and(|table| self.reuses(kind, table))
                    })
                    .map(|(name, _)| name.to_owned())
            });
            let name = if let Some(name) = reused {
                name
            } else if providers.is_some_and(|providers| providers.contains_key(kind.name())) {
                return Err(Error::Config(format!(
                    "[providers.{}] is already a different provider; rename it, or set the \
                     model in config.toml yourself",
                    kind.name()
                )));
            } else {
                added.push(kind);
                kind.name().to_owned()
            };
            names.push((kind, name));
        }

        let mut changes = Vec::new();
        let model = |choice: &Choice| {
            let name = names
                .iter()
                .find(|(kind, _)| *kind == choice.kind)
                .map_or_else(|| choice.kind.name(), |(_, name)| name.as_str());
            format!("{name}/{}", choice.model)
        };
        if let Some(chat) = &self.chat {
            set(
                &mut doc,
                &mut changes,
                "general",
                "chat_model",
                model(chat).into(),
            )?;
        }
        if let Some(decision) = &self.decision {
            set(
                &mut doc,
                &mut changes,
                "decision",
                "model",
                model(decision).into(),
            )?;
        }
        if let Some(embedding) = &self.embedding {
            let width = i64::from(embedding.dimension.get());
            set(
                &mut doc,
                &mut changes,
                "embedding",
                "model",
                model(&embedding.choice).into(),
            )?;
            set(
                &mut doc,
                &mut changes,
                "embedding",
                "dimension",
                width.into(),
            )?;
        }
        if !added.is_empty() {
            let providers = section(&mut doc, "providers")?;
            for kind in added {
                providers.insert(kind.name(), Item::Table(self.provider_table(kind)));
                changes.push(format!("adds [providers.{}]", kind.name()));
            }
        }
        if let Some(table) = doc.get_mut("providers").and_then(Item::as_table_mut) {
            table.set_implicit(true);
        }
        let text = match existing {
            Some(_) => doc.to_string(),
            None => format!("{NEW_FILE}\n{doc}"),
        };
        Ok((text, changes))
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
        table.insert("max_retries", value(0));
        let mut doc = DocumentMut::new();
        section(&mut doc, "providers")?.insert(kind.name(), Item::Table(table));
        Config::parse(&doc.to_string())
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

/// The section `key` of `doc`, made when the file has none.
fn section<'a>(doc: &'a mut DocumentMut, key: &str) -> Result<&'a mut dyn TableLike> {
    doc.entry(key)
        .or_insert_with(toml_edit::table)
        .as_table_like_mut()
        .ok_or_else(|| Error::Config(format!("[{key}] in config.toml is not a table")))
}

/// Set `key` in section `name` to `new`, keeping the comment on the
/// line it replaces, and note the change against what was there.
fn set(
    doc: &mut DocumentMut,
    changes: &mut Vec<String>,
    name: &str,
    key: &str,
    mut new: Value,
) -> Result<()> {
    let table = section(doc, name)?;
    let bare = |value: &Value| {
        let mut value = value.clone();
        value.decor_mut().clear();
        value.to_string()
    };
    let shown = bare(&new);
    let Some(old) = table.get_mut(key).and_then(Item::as_value_mut) else {
        table.insert(key, Item::Value(new));
        changes.push(format!("[{name}].{key} = {shown}"));
        return Ok(());
    };
    let was = bare(old);
    if was != shown {
        *new.decor_mut() = old.decor().clone();
        *old = new;
        changes.push(format!("[{name}].{key} = {shown} (was {was})"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
