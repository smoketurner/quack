//! `quack init`: find the model providers this machine can reach, ask which
//! models to use, and write them to `config.toml`, a new one or the one
//! already there, only after `quack doctor` passes the result, so a run
//! never leaves a broken file.

use std::fmt;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use inquire::InquireError;
use quack_core::config::inspect::Inspection;
use quack_core::config::{Config, Overrides, config_file_path};
use quack_core::doctor::{self, Options};
use quack_core::embedding::Dimension;
use quack_core::llm::{OllamaCapability, OllamaModel};
use quack_core::setup::{
    ByteSize, Choice, Current, Discovery, EmbeddingChoice, Environment, Found, ProviderKind,
    SetupPlan,
};

use crate::Exit;
use crate::doctor_cli;
use quack_cli::TextOrJson;

/// What to pull when Ollama runs but lacks a model for a role.
const OLLAMA_CHAT_SUGGESTION: &str = "ollama pull gpt-oss:20b";
const OLLAMA_EMBEDDING_SUGGESTION: &str = "ollama pull qwen3-embedding:0.6b";

/// The first `quack` on a machine with no config file and no
/// `QUACK_MODEL`: offer to run `quack init` before the session starts.
/// `Some` is the exit when the person ran it; `None` starts the session,
/// which works without a model for SQL and loading files.
pub(crate) async fn offer_on_first_run() -> Result<Option<ExitCode>> {
    let path = config_file_path();
    if path.exists() || Overrides::from_env().chat_model.is_some() {
        return Ok(None);
    }
    let mut talk = std::io::stderr();
    writeln!(
        talk,
        "No config file at {}, so no model can answer questions yet.",
        path.display()
    )?;
    let run = inquire::Confirm::new("Set up a model provider now with quack init?")
        .with_default(true)
        .with_help_message(
            "No starts the session without a model: SQL and loading files still work",
        )
        .prompt()
        .unwrap_or(false);
    if !run {
        return Ok(None);
    }
    run_init().await.map(Some)
}

/// `quack init`.
pub(crate) async fn run_init() -> Result<ExitCode> {
    let mut talk = std::io::stderr();
    match setup(&mut talk).await {
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

/// The config file as it is now.
struct ConfigFile {
    path: PathBuf,
    /// `None` when there is no file yet.
    text: Option<String>,
}

impl ConfigFile {
    fn read() -> Result<Self> {
        let path = config_file_path();
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => Some(text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
        };
        Ok(Self { path, text })
    }
}

async fn setup(talk: &mut impl Write) -> Result<ExitCode> {
    if !std::io::stdin().is_terminal() {
        writeln!(
            talk,
            "quack init asks questions, so it needs a terminal; run it in one."
        )?;
        return Ok(Exit::Usage.into());
    }
    let file = ConfigFile::read()?;
    let current = match &file.text {
        Some(text) => match Current::of(text) {
            Ok(current) => {
                writeln!(
                    talk,
                    "Editing {}. Settings you do not change stay as they are.",
                    file.path.display()
                )?;
                current
            }
            Err(e) => {
                writeln!(
                    talk,
                    "{} is {e}; fix it before quack init can edit it.",
                    file.path.display()
                )?;
                return Ok(Exit::Usage.into());
            }
        },
        None => Current::default(),
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
    plan.chat = chat_model(&discovery, &plan, &current, talk).await?;
    plan.embedding = embedding(&discovery, &plan, &current, talk).await?;
    plan.decision = decision_model(&discovery, &current)?;
    if plan.is_empty() {
        if file.text.is_some() {
            writeln!(talk, "Nothing changed.")?;
            return Ok(ExitCode::SUCCESS);
        }
        writeln!(
            talk,
            "Nothing to write: no chat or embedding model was chosen."
        )?;
        explain_missing(&discovery, talk)?;
        return Ok(Exit::Usage.into());
    }

    let (text, changes) = plan.apply(file.text.as_deref())?;
    Config::parse(&text).context("quack init built a config quack refuses; nothing was written")?;
    if changes.is_empty() {
        writeln!(talk, "Nothing changed.")?;
        return Ok(ExitCode::SUCCESS);
    }
    writeln!(talk, "Changes to {}:", file.path.display())?;
    for change in &changes {
        writeln!(talk, "  {change}")?;
    }
    let write = inquire::Confirm::new("Write them?")
        .with_default(true)
        .prompt()?;
    if !write {
        writeln!(talk, "Nothing was written.")?;
        return Ok(ExitCode::SUCCESS);
    }

    let code = check_and_write(&file.path, &text, &Options::default(), talk).await?;
    if code == ExitCode::SUCCESS && file.text.is_some() && plan.embedding.is_some() {
        writeln!(
            talk,
            "Workspaces keep their old document vectors until `quack embeddings refresh -w \
             NAME`; keyword search finds their documents meanwhile."
        )?;
    }
    Ok(code)
}

/// Run doctor on `text` as the config at `path`, and write it only when
/// no check fails.
async fn check_and_write(
    path: &Path,
    text: &str,
    options: &Options,
    talk: &mut impl Write,
) -> Result<ExitCode> {
    writeln!(talk, "Checking the new config with quack doctor...")?;
    let inspection = Inspection::of(path.to_path_buf(), Some(text));
    let report = doctor::run(&inspection, options).await;
    doctor_cli::write(talk, &report, TextOrJson::Text)?;
    if report.has_failures() {
        writeln!(
            talk,
            "\nNothing was written, so no broken config is left behind. Fix what failed \
             above and run quack init again."
        )?;
        return Ok(ExitCode::FAILURE);
    }
    replace(path, text)?;
    writeln!(
        talk,
        "\nWrote {}. Run quack to start a session, or quack ingest FILE to load a document.",
        path.display()
    )?;
    Ok(ExitCode::SUCCESS)
}

/// Write `text` to `path` all at once: a temporary file beside the real
/// one (through a symlink, beside its target), with the old file's
/// permissions, then a rename over it. An interrupted run leaves the old
/// file whole.
fn replace(path: &Path, text: &str) -> Result<()> {
    let target = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let dir = target
        .parent()
        .context("the config path has no directory")?;
    std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let mut staged = tempfile::NamedTempFile::new_in(dir)
        .with_context(|| format!("cannot write in {}", dir.display()))?;
    staged.write_all(text.as_bytes())?;
    if let Ok(metadata) = std::fs::metadata(&target) {
        staged.as_file().set_permissions(metadata.permissions())?;
    }
    staged.as_file().sync_all()?;
    staged
        .persist(&target)
        .map_err(|e| e.error)
        .with_context(|| format!("cannot write {}", target.display()))?;
    Ok(())
}

async fn chat_model(
    discovery: &Discovery,
    plan: &SetupPlan,
    current: &Current,
    talk: &mut impl Write,
) -> Result<Option<Choice>> {
    let mut options: Vec<ChatOption<'_>> = current
        .chat_model
        .iter()
        .map(|model| ChatOption::Keep(model))
        .collect();
    options.extend(
        discovery
            .0
            .iter()
            .filter(|found| found.is_present())
            .map(ChatOption::Provider),
    );
    let found = match options.as_slice() {
        [] | [ChatOption::Keep(_)] => return Ok(None),
        _ => match inquire::Select::new("Which provider should answer questions?", options)
            .prompt()?
        {
            ChatOption::Keep(_) => return Ok(None),
            ChatOption::Provider(found) => found,
        },
    };
    let kind = found.kind();
    let model = match kind {
        ProviderKind::Ollama => {
            let tools = discovery.ollama_models(OllamaCapability::Tools);
            if tools.is_empty() {
                writeln!(
                    talk,
                    "Ollama has no model that can call tools, which quack's agent needs. \
                     Try: {OLLAMA_CHAT_SUGGESTION}"
                )?;
                return Ok(None);
            }
            let options = tools.into_iter().map(ModelOption).collect();
            inquire::Select::new("Chat model (models that can call tools):", options)
                .prompt()?
                .0
                .name
                .clone()
        }
        ProviderKind::Anthropic | ProviderKind::OpenAi => {
            let listed = plan.list_hosted(kind).await;
            match listed {
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
            }
        }
        ProviderKind::Bedrock => {
            let mut prompt = inquire::Text::new("Bedrock model id (Bedrock lists none):");
            if let Some(default) = kind.unlisted_chat_model() {
                prompt = prompt.with_default(default);
            }
            prompt.prompt()?
        }
    };
    Ok(Some(Choice { kind, model }))
}

async fn embedding(
    discovery: &Discovery,
    plan: &SetupPlan,
    current: &Current,
    talk: &mut impl Write,
) -> Result<Option<EmbeddingChoice>> {
    let options = EmbeddingOption::offered(discovery, plan, current);
    let picked = match options.as_slice() {
        [] | [EmbeddingOption::Keep(_) | EmbeddingOption::None] => EmbeddingOption::None,
        _ => inquire::Select::new("Embedding model, for searching documents:", options).prompt()?,
    };
    let choice = match picked {
        EmbeddingOption::Keep(_) | EmbeddingOption::None => None,
        EmbeddingOption::Ollama(model) => {
            writeln!(talk, "Measuring {model}'s vector width...")?;
            let dimension = plan.ollama_width(&model).await?;
            Some(EmbeddingChoice {
                choice: Choice {
                    kind: ProviderKind::Ollama,
                    model,
                },
                dimension,
            })
        }
        EmbeddingOption::Hosted(kind, model, dimension) => Some(EmbeddingChoice {
            choice: Choice {
                kind,
                model: model.to_owned(),
            },
            dimension,
        }),
    };
    if choice.is_none()
        && current.embedding_model.is_none()
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

/// The decision model that labels a table's text, when Ollama has one: its
/// models whose capabilities say `decision`. Asked only where there is a
/// choice to make.
fn decision_model(discovery: &Discovery, current: &Current) -> Result<Option<Choice>> {
    let mut options: Vec<DecisionOption<'_>> = current
        .decision_model
        .iter()
        .map(|model| DecisionOption::Keep(model))
        .collect();
    options.extend(
        discovery
            .ollama_models(OllamaCapability::Decision)
            .into_iter()
            .map(DecisionOption::Ollama),
    );
    if options.is_empty() {
        return Ok(None);
    }
    if current.decision_model.is_none() {
        options.push(DecisionOption::None);
    }
    if matches!(options.as_slice(), [DecisionOption::Keep(_)]) {
        return Ok(None);
    }
    Ok(
        match inquire::Select::new(
            "Decision model, for labelling the text of a table's rows:",
            options,
        )
        .prompt()?
        {
            DecisionOption::Ollama(model) => Some(Choice {
                kind: ProviderKind::Ollama,
                model: model.name.clone(),
            }),
            DecisionOption::Keep(_) | DecisionOption::None => None,
        },
    )
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
            "Start Ollama (https://ollama.com), or set ANTHROPIC_API_KEY or OPENAI_API_KEY, or \
             sign in to AWS, then run quack init again."
        )?;
    }
    Ok(())
}

/// One answer to the chat question.
enum ChatOption<'a> {
    /// The model the file names now.
    Keep(&'a str),
    Provider(&'a Found),
}

impl fmt::Display for ChatOption<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let found = match self {
            Self::Keep(model) => return write!(f, "Keep {model}"),
            Self::Provider(found) => found,
        };
        let kind = found.kind();
        match (kind, found) {
            (ProviderKind::Ollama, _) => {
                write!(f, "{} (local, nothing leaves this machine)", kind.label())
            }
            (_, Found::Hosted { credential, .. }) => write!(f, "{} ({credential})", kind.label()),
            (_, Found::Ollama { .. } | Found::Absent { .. }) => f.write_str(kind.label()),
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

/// One answer to the decision model question.
enum DecisionOption<'a> {
    /// The model the file names now.
    Keep(&'a str),
    Ollama(&'a OllamaModel),
    None,
}

impl fmt::Display for DecisionOption<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Keep(model) => write!(f, "Keep {model}"),
            Self::Ollama(model) => write!(f, "ollama/{}", ModelOption(model)),
            Self::None => f.write_str("None: quack classify is not available"),
        }
    }
}

/// One answer to the embedding question.
#[derive(Debug, Clone, PartialEq, Eq)]
enum EmbeddingOption {
    /// The model the file names now.
    Keep(String),
    Ollama(String),
    Hosted(ProviderKind, &'static str, Dimension),
    None,
}

impl EmbeddingOption {
    /// What the menu offers, in order: the file's own model, Ollama's
    /// embedding models, the documented model of the hosted provider the
    /// chat model uses, and no model when the file has none.
    fn offered(discovery: &Discovery, plan: &SetupPlan, current: &Current) -> Vec<Self> {
        let mut options: Vec<Self> = current
            .embedding_model
            .iter()
            .map(|model| Self::Keep(model.clone()))
            .collect();
        options.extend(
            discovery
                .ollama_models(OllamaCapability::Embedding)
                .into_iter()
                .map(|model| Self::Ollama(model.name.clone())),
        );
        if let Some(chat) = &plan.chat
            && let Some((model, dimension)) = chat.kind.hosted_embedding()
        {
            options.push(Self::Hosted(chat.kind, model, dimension));
        }
        if current.embedding_model.is_none() {
            options.push(Self::None);
        }
        options
    }
}

impl fmt::Display for EmbeddingOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Keep(model) => write!(f, "Keep {model}"),
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
