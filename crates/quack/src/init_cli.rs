//! `quack init`: find the model providers this machine can reach, ask which
//! to use, and write `config.toml` only after `quack doctor` passes it, so a
//! run never leaves a broken file. An existing file is never touched.

use std::fmt;
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::process::ExitCode;
use std::str::FromStr;

use anyhow::{Context, Result};
use inquire::InquireError;
use quack_core::config::inspect::Inspection;
use quack_core::config::{ModelSpec, config_file_path};
use quack_core::doctor::{self, Options};
use quack_core::embedding::Dimension;
use quack_core::error::Error as CoreError;
use quack_core::llm::{OllamaCapability, OllamaModel};
use quack_core::setup::{
    ByteSize, Discovery, EmbeddingChoice, Environment, Found, ProviderKind, SetupPlan,
};

use crate::Exit;
use crate::doctor_cli;
use crate::text_or_json::TextOrJson;

/// What to pull when Ollama runs but lacks a model for a role.
const OLLAMA_CHAT_SUGGESTION: &str = "ollama pull gpt-oss:20b";
const OLLAMA_EMBEDDING_SUGGESTION: &str = "ollama pull qwen3-embedding:0.6b";

#[derive(clap::Args)]
pub(crate) struct InitArgs {
    /// Take the default at every step instead of asking. Hosted providers
    /// are used only when --chat-model names one
    #[arg(long, short = 'y')]
    yes: bool,

    /// Print the config to stdout instead of writing it
    #[arg(long)]
    print: bool,

    /// The chat model, as PROVIDER/MODEL; PROVIDER is ollama, anthropic,
    /// openai, or bedrock
    #[arg(long, value_name = "PROVIDER/MODEL")]
    chat_model: Option<ModelSpec>,

    /// The embedding model, as PROVIDER/MODEL, or `none` for keyword search
    /// only
    #[arg(long, value_name = "PROVIDER/MODEL|none")]
    embedding_model: Option<EmbeddingFlag>,
}

/// `--embedding-model`'s value.
#[derive(Debug, Clone)]
enum EmbeddingFlag {
    None,
    Model(ModelSpec),
}

impl FromStr for EmbeddingFlag {
    type Err = CoreError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value == "none" {
            Ok(Self::None)
        } else {
            value.parse().map(Self::Model)
        }
    }
}

/// Whether a person answers the questions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Ask,
    Defaults,
}

impl InitArgs {
    pub(crate) async fn run(&self) -> Result<ExitCode> {
        let mut talk = std::io::stderr();
        match self.setup(&mut talk).await {
            Ok(code) => Ok(code),
            Err(e) => match e.downcast_ref::<InquireError>() {
                Some(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
                    writeln!(talk, "Cancelled; nothing was written.")?;
                    Ok(ExitCode::FAILURE)
                }
                Some(_) | None => Err(e),
            },
        }
    }

    async fn setup(&self, talk: &mut impl Write) -> Result<ExitCode> {
        let path = config_file_path();
        if path.exists() && !self.print {
            writeln!(
                talk,
                "{} already exists, and quack init only writes a new file.\n\
                 `quack config` shows what it holds; `quack init --print` prints a fresh \
                 config to compare.",
                path.display()
            )?;
            return Ok(Exit::Usage.into());
        }
        let mode = if self.yes {
            Mode::Defaults
        } else if std::io::stdin().is_terminal() {
            Mode::Ask
        } else {
            writeln!(
                talk,
                "quack init asks questions, and there is no terminal to answer them; \
                 --yes takes the defaults."
            )?;
            return Ok(Exit::Usage.into());
        };

        writeln!(talk, "Looking for model providers on this machine...")?;
        let discovery = Discovery::run(&Environment::from_env()).await;
        for found in &discovery.0 {
            writeln!(talk, "  {found}")?;
        }
        writeln!(talk)?;

        let mut plan = SetupPlan {
            ollama_base_url: discovery.ollama_base_url(),
            ..SetupPlan::default()
        };
        plan.chat_model = self.chat_model(mode, &discovery, &plan, talk).await?;
        plan.embedding = self.embedding(mode, &discovery, &plan, talk).await?;
        if plan.chat_model.is_none() && plan.embedding.is_none() {
            writeln!(
                talk,
                "Nothing to write: no chat or embedding model was chosen."
            )?;
            Self::explain_missing(&discovery, talk)?;
            return Ok(Exit::Usage.into());
        }
        if plan.chat_model.is_none() {
            writeln!(
                talk,
                "No chat model: questions need one. {}",
                Self::chat_hint(&discovery)
            )?;
        }

        let text = plan.toml()?;
        plan.config()
            .context("quack init built a config quack refuses; nothing was written")?;
        if self.print {
            let stdout = std::io::stdout();
            let mut out = stdout.lock();
            out.write_all(text.as_bytes())?;
            out.flush()?;
            return Ok(ExitCode::SUCCESS);
        }
        if mode == Mode::Ask {
            writeln!(talk, "{text}")?;
            let write = inquire::Confirm::new(&format!("Write {}?", path.display()))
                .with_default(true)
                .prompt()?;
            if !write {
                writeln!(talk, "Nothing was written.")?;
                return Ok(ExitCode::SUCCESS);
            }
        }

        writeln!(talk, "Checking the new config with quack doctor...")?;
        let inspection = Inspection::of(path.clone(), Some(&text));
        let report = doctor::run(&inspection, &Options::default()).await;
        doctor_cli::write(talk, &report, TextOrJson::Text)?;
        if report.has_failures() {
            writeln!(
                talk,
                "\nNothing was written, so no broken config is left behind. Fix what failed \
                 above and run quack init again."
            )?;
            return Ok(ExitCode::FAILURE);
        }
        write_new(&path, &text)?;
        writeln!(
            talk,
            "\nWrote {}. Try: quack ingest FILE, then quack -p \"a question\".",
            path.display()
        )?;
        Ok(ExitCode::SUCCESS)
    }

    async fn chat_model(
        &self,
        mode: Mode,
        discovery: &Discovery,
        plan: &SetupPlan,
        talk: &mut impl Write,
    ) -> Result<Option<ModelSpec>> {
        if let Some(model) = &self.chat_model {
            return Ok(Some(model.clone()));
        }
        let tools = discovery.ollama_models(OllamaCapability::Tools);
        if mode == Mode::Defaults {
            return tools
                .first()
                .map(|model| ollama_spec(&model.name))
                .transpose();
        }
        let present: Vec<ProviderChoice<'_>> = discovery
            .0
            .iter()
            .filter(|found| found.is_present())
            .map(ProviderChoice)
            .collect();
        if present.is_empty() {
            return Ok(None);
        }
        let choice =
            inquire::Select::new("Which provider should answer questions?", present).prompt()?;
        let kind = choice.0.kind();
        match kind {
            ProviderKind::Ollama => {
                if tools.is_empty() {
                    writeln!(
                        talk,
                        "Ollama has no model that can call tools, which quack's agent needs. \
                         Try: {OLLAMA_CHAT_SUGGESTION}"
                    )?;
                    return Ok(None);
                }
                let options = tools.into_iter().map(ModelOption).collect();
                let picked =
                    inquire::Select::new("Chat model (models that can call tools):", options)
                        .prompt()?;
                ollama_spec(&picked.0.name).map(Some)
            }
            ProviderKind::Anthropic | ProviderKind::OpenAi => {
                let listed = plan.list_hosted(kind).await;
                let model = match listed {
                    Ok(models) if !models.as_slice().is_empty() => {
                        let ids = models.as_slice().iter().map(|m| m.id.clone()).collect();
                        inquire::Select::new("Chat model:", ids).prompt()?
                    }
                    Ok(_) | Err(_) => {
                        if let Err(e) = listed {
                            writeln!(talk, "{} did not list its models: {e}", kind.label())?;
                        }
                        inquire::Text::new("Chat model id:").prompt()?
                    }
                };
                spec(kind, &model).map(Some)
            }
            ProviderKind::Bedrock => {
                let mut prompt = inquire::Text::new("Bedrock model id (Bedrock lists none):");
                if let Some(default) = kind.unlisted_chat_model() {
                    prompt = prompt.with_default(default);
                }
                spec(kind, &prompt.prompt()?).map(Some)
            }
        }
    }

    async fn embedding(
        &self,
        mode: Mode,
        discovery: &Discovery,
        plan: &SetupPlan,
        talk: &mut impl Write,
    ) -> Result<Option<EmbeddingChoice>> {
        let option = match &self.embedding_model {
            Some(EmbeddingFlag::None) => EmbeddingOption::None,
            Some(EmbeddingFlag::Model(model)) => EmbeddingOption::from_flag(model)?,
            None => {
                let options = EmbeddingOption::offered(discovery, plan);
                match mode {
                    Mode::Defaults => options.into_iter().next().unwrap_or(EmbeddingOption::None),
                    Mode::Ask => {
                        inquire::Select::new("Embedding model, for searching documents:", options)
                            .prompt()?
                    }
                }
            }
        };
        let choice = match option {
            EmbeddingOption::None => None,
            EmbeddingOption::Ollama(name) => {
                writeln!(talk, "Measuring {name}'s vector width...")?;
                let dimension = plan.ollama_width(&name).await?;
                Some(EmbeddingChoice {
                    model: ollama_spec(&name)?,
                    dimension,
                })
            }
            EmbeddingOption::Hosted(kind, model, dimension) => Some(EmbeddingChoice {
                model: spec(kind, model)?,
                dimension,
            }),
        };
        if choice.is_none()
            && discovery
                .get(ProviderKind::Ollama)
                .is_some_and(Found::is_present)
        {
            writeln!(
                talk,
                "No embedding model: document search is keyword-only. For semantic search: \
                 {OLLAMA_EMBEDDING_SUGGESTION}"
            )?;
        }
        Ok(choice)
    }

    fn chat_hint(discovery: &Discovery) -> &'static str {
        if discovery
            .get(ProviderKind::Ollama)
            .is_some_and(Found::is_present)
        {
            "Pull one and run quack init again: ollama pull gpt-oss:20b"
        } else {
            "Pass --chat-model PROVIDER/MODEL, or start Ollama and run quack init again."
        }
    }

    fn explain_missing(discovery: &Discovery, talk: &mut impl Write) -> Result<()> {
        if discovery
            .get(ProviderKind::Ollama)
            .is_some_and(Found::is_present)
        {
            writeln!(
                talk,
                "Ollama has no usable model. Try:\n  {OLLAMA_CHAT_SUGGESTION}\n  \
                 {OLLAMA_EMBEDDING_SUGGESTION}"
            )?;
        } else {
            writeln!(
                talk,
                "Start Ollama (https://ollama.com), or run quack init at a terminal to choose \
                 Anthropic, OpenAI, or Amazon Bedrock, or pass --chat-model PROVIDER/MODEL with \
                 --yes."
            )?;
        }
        Ok(())
    }
}

/// `PROVIDER/MODEL` for `kind`'s `model`.
fn spec(kind: ProviderKind, model: &str) -> Result<ModelSpec> {
    format!("{}/{model}", kind.name())
        .parse()
        .context("not a model id")
}

fn ollama_spec(model: &str) -> Result<ModelSpec> {
    spec(ProviderKind::Ollama, model)
}

/// Write `text` to `path`, which must not exist, all at once: a temporary
/// file in the same directory, then a rename that refuses to replace a file
/// that appeared meanwhile.
fn write_new(path: &Path, text: &str) -> Result<()> {
    let dir = path.parent().context("the config path has no directory")?;
    std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let mut staged = tempfile::NamedTempFile::new_in(dir)
        .with_context(|| format!("cannot write in {}", dir.display()))?;
    staged.write_all(text.as_bytes())?;
    staged.as_file().sync_all()?;
    staged
        .persist_noclobber(path)
        .map_err(|e| e.error)
        .with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

/// A found provider as the provider menu shows it.
struct ProviderChoice<'a>(&'a Found);

impl fmt::Display for ProviderChoice<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = self.0.kind();
        match kind {
            ProviderKind::Ollama => {
                write!(f, "{} (local, nothing leaves this machine)", kind.label())
            }
            ProviderKind::Anthropic | ProviderKind::OpenAi | ProviderKind::Bedrock => {
                match self.0 {
                    Found::Hosted { credential, .. } => {
                        write!(f, "{} ({credential})", kind.label())
                    }
                    Found::Ollama { .. } | Found::Absent { .. } => f.write_str(kind.label()),
                }
            }
        }
    }
}

/// An Ollama model as a menu shows it: name, size, and capabilities.
struct ModelOption<'a>(&'a OllamaModel);

impl fmt::Display for ModelOption<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let model = self.0;
        let capabilities: Vec<String> = model
            .capabilities
            .iter()
            .filter(|c| **c != OllamaCapability::Other)
            .map(|c| format!("{c:?}").to_lowercase())
            .collect();
        write!(
            f,
            "{:<32} {:>8}  {}",
            model.name,
            ByteSize(model.size).to_string(),
            capabilities.join(", ")
        )
    }
}

/// One answer to the embedding question.
#[derive(Debug, Clone, PartialEq, Eq)]
enum EmbeddingOption {
    Ollama(String),
    Hosted(ProviderKind, &'static str, Dimension),
    None,
}

impl EmbeddingOption {
    /// What the menu offers, best first: Ollama's embedding models, then
    /// the documented model of each hosted provider the chat model uses,
    /// then none.
    fn offered(discovery: &Discovery, plan: &SetupPlan) -> Vec<Self> {
        let mut options: Vec<Self> = discovery
            .ollama_models(OllamaCapability::Embedding)
            .into_iter()
            .map(|model| Self::Ollama(model.name.clone()))
            .collect();
        let chat_kind = plan
            .chat_model
            .as_ref()
            .and_then(|model| ProviderKind::named(&model.provider().to_string()));
        if let Some(kind) = chat_kind
            && let Some((model, dimension)) = kind.hosted_embedding()
        {
            options.push(Self::Hosted(kind, model, dimension));
        }
        options.push(Self::None);
        options
    }

    /// `--embedding-model`'s model: an Ollama model, or a hosted
    /// provider's documented one, whose width quack knows.
    fn from_flag(model: &ModelSpec) -> Result<Self> {
        let provider = model.provider().to_string();
        let kind = ProviderKind::named(&provider).with_context(|| {
            format!("{model}: quack init sets up ollama, openai, and bedrock embeddings")
        })?;
        match (kind, kind.hosted_embedding()) {
            (ProviderKind::Ollama, _) => Ok(Self::Ollama(model.model().to_owned())),
            (_, Some((known, dimension))) if known == model.model() => {
                Ok(Self::Hosted(kind, known, dimension))
            }
            (_, Some((known, _))) => anyhow::bail!(
                "{model}: quack init knows the width of {}/{known} only; set [embedding] in \
                 config.toml yourself for another model",
                kind.name()
            ),
            (_, None) => anyhow::bail!("{model}: {} serves no embedding models", kind.label()),
        }
    }
}

impl fmt::Display for EmbeddingOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ollama(name) => write!(f, "ollama/{name}"),
            Self::Hosted(kind, model, dimension) => {
                write!(
                    f,
                    "{}/{model} ({} dimensions)",
                    kind.name(),
                    dimension.get()
                )
            }
            Self::None => f.write_str("None: keyword search only"),
        }
    }
}

#[cfg(test)]
mod tests;
