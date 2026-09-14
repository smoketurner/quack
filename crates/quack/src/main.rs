#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod print;
mod terminal;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::SharedDb;
use quack_core::config::Config;
use quack_core::ingestion;
use quack_core::llm;
use quack_core::storage::control::{ControlPlane, WorkspaceRow};
use quack_core::storage::sessions;
use quack_core::storage::workspace::WorkspaceDb;
use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

/// Exit status for a usage error (bad flags, no terminal for the session).
const EXIT_USAGE: u8 = 2;
/// Exit status when the agent needed a write that was not permitted.
const EXIT_WRITE_REFUSED: u8 = 3;

#[derive(Parser)]
#[command(
    name = "quack",
    version,
    about = "Knowledge engine: documents, tables, and a knowledge graph in one workspace",
    long_about = "With no arguments, starts the interactive terminal session in a workspace.\n\
                  `-p PROMPT` asks the agent one question and prints the answer; \
                  `-q SQL` runs SQL directly. Both are pipe-friendly."
)]
struct Cli {
    /// Ask the agent one question, print the answer, and exit
    #[arg(
        short = 'p',
        long = "print",
        value_name = "PROMPT",
        conflicts_with = "query"
    )]
    prompt: Option<String>,

    /// Run SQL directly against the workspace and print the result set
    #[arg(short = 'q', long = "query", value_name = "SQL")]
    query: Option<String>,

    /// Output format. Default: table on a terminal, ndjson when piped
    /// (`-p` accepts text or json only)
    #[arg(short = 'f', long, value_enum)]
    format: Option<OutputFormat>,

    /// Workspace name (defaults to config value)
    #[arg(long, short = 'w', global = true)]
    workspace: Option<String>,

    /// Let the agent run statements that modify the workspace
    #[arg(long, global = true)]
    allow_write: bool,

    /// Continue the most recent session in the workspace
    #[arg(short = 'c', long = "continue", conflicts_with = "resume")]
    continue_latest: bool,

    /// Resume a specific session by id (prefixes accepted)
    #[arg(short = 'r', long, value_name = "SESSION_ID")]
    resume: Option<String>,

    /// Print full tool inputs and outputs to stderr in print mode
    #[arg(long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// List sessions in the workspace, most recent first
    Sessions {
        /// Emit one JSON object per session
        #[arg(long)]
        json: bool,

        /// Maximum number of sessions to show
        #[arg(long, default_value_t = 20)]
        limit: u32,
    },

    /// Export a session as a runnable .sql file or a Markdown transcript
    Export {
        /// Session id (prefixes accepted)
        session_id: String,

        /// Every executed statement, each preceded by its question
        #[arg(long, conflicts_with = "markdown")]
        sql: bool,

        /// Questions, steps, and answers as Markdown (default)
        #[arg(long)]
        markdown: bool,
    },

    /// Ingest a file into a workspace
    Ingest {
        /// File path to ingest (use - for stdin)
        file: String,

        /// Override the filename (required when reading from stdin)
        #[arg(long)]
        filename: Option<String>,

        /// Skip embedding generation
        #[arg(long)]
        no_embed: bool,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum OutputFormat {
    /// Aligned text table (SQL) or plain answer text (`-p`)
    Table,
    /// One JSON document
    Json,
    /// One JSON object per line
    Ndjson,
    /// Comma-separated values with a header row
    Csv,
    /// GitHub-flavored Markdown table
    Markdown,
    /// Plain answer text (`-p` only)
    Text,
}

impl OutputFormat {
    fn default_for(stdout_is_tty: bool) -> Self {
        if stdout_is_tty {
            Self::Table
        } else {
            Self::Ndjson
        }
    }
}

#[tokio::main]
async fn main() -> Result<ExitCode> {
    quack_core::crypto::install_default_provider()
        .context("failed to install the aws-lc-rs crypto provider")?;

    let cli = Cli::parse();

    let policy = if cli.allow_write {
        WritePolicy::Allow
    } else {
        WritePolicy::Deny
    };
    let stdout_is_tty = std::io::stdout().is_terminal();

    if let Some(prompt) = cli.prompt.as_deref() {
        return run_print_mode(&cli, prompt, policy).await;
    }

    if let Some(sql) = cli.query.as_deref() {
        init_logging();
        let format = cli
            .format
            .unwrap_or_else(|| OutputFormat::default_for(stdout_is_tty));
        run_query(sql, cli.workspace.as_deref(), format).await?;
        return Ok(ExitCode::SUCCESS);
    }

    match cli.command {
        None => run_terminal_session(&cli, stdout_is_tty).await,
        Some(Commands::Sessions { json, limit }) => {
            init_logging();
            let (config, workspace, _) = resolve_workspace(cli.workspace.as_deref()).await?;
            let ws_db = WorkspaceDb::open(&config, &workspace.id)
                .context("failed to open workspace database")?;
            list_sessions(&ws_db, json, limit)?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Export {
            session_id,
            sql,
            markdown: _,
        }) => {
            init_logging();
            let (config, workspace, _) = resolve_workspace(cli.workspace.as_deref()).await?;
            let ws_db = WorkspaceDb::open(&config, &workspace.id)
                .context("failed to open workspace database")?;
            export_session(&ws_db, &session_id, sql)?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Ingest {
            file,
            filename,
            no_embed,
        }) => {
            init_logging();
            run_ingest(
                &file,
                cli.workspace.as_deref(),
                filename.as_deref(),
                no_embed,
            )
            .await?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// `quack -p PROMPT`: one turn, answer to stdout, steps to stderr.
async fn run_print_mode(cli: &Cli, prompt: &str, policy: WritePolicy) -> Result<ExitCode> {
    init_logging();
    let format = match cli.format.unwrap_or(OutputFormat::Text) {
        OutputFormat::Json => print::PromptFormat::Json,
        OutputFormat::Text | OutputFormat::Table => print::PromptFormat::Text,
        OutputFormat::Ndjson | OutputFormat::Csv | OutputFormat::Markdown => {
            tracing::error!("-p accepts only --format text or json");
            return Ok(ExitCode::from(EXIT_USAGE));
        }
    };
    let (config, workspace, _) = resolve_workspace(cli.workspace.as_deref()).await?;
    let ws_db =
        WorkspaceDb::open(&config, &workspace.id).context("failed to open workspace database")?;
    let session_id = resolve_session(&config, &ws_db, cli.continue_latest, cli.resume.as_deref())?;
    let db: SharedDb = Arc::new(Mutex::new(ws_db));
    let outcome = print::run_prompt(
        &config,
        Arc::clone(&db),
        &session_id,
        policy,
        prompt,
        format,
        cli.verbose,
    )
    .await;
    if outcome.is_err()
        && let Ok(guard) = db.lock()
    {
        drop(sessions::delete_if_empty(&guard, &session_id));
    }
    let refused = outcome?;
    Ok(if refused {
        ExitCode::from(EXIT_WRITE_REFUSED)
    } else {
        ExitCode::SUCCESS
    })
}

/// No arguments: the interactive session, which needs a terminal.
async fn run_terminal_session(cli: &Cli, stdout_is_tty: bool) -> Result<ExitCode> {
    if !std::io::stdin().is_terminal() || !stdout_is_tty {
        init_logging();
        tracing::error!(
            "the interactive session needs a terminal; use `quack -p PROMPT` or `quack -q SQL` in pipelines"
        );
        return Ok(ExitCode::from(EXIT_USAGE));
    }
    let (config, workspace, ws_name) = resolve_workspace(cli.workspace.as_deref()).await?;
    let ws_db =
        WorkspaceDb::open(&config, &workspace.id).context("failed to open workspace database")?;
    let session_id = resolve_session(&config, &ws_db, cli.continue_latest, cli.resume.as_deref())?;
    terminal::run(
        config,
        ws_name,
        workspace.id,
        Arc::new(Mutex::new(ws_db)),
        session_id,
        cli.allow_write,
    )?;
    Ok(ExitCode::SUCCESS)
}

/// Pick the session for this run: the latest with `--continue`, a specific
/// one with `--resume`, otherwise a new one for the configured chat model.
fn resolve_session(
    config: &Config,
    db: &WorkspaceDb,
    continue_latest: bool,
    resume: Option<&str>,
) -> Result<String> {
    if let Some(prefix) = resume {
        return find_session(db, prefix).map(|s| s.id);
    }
    if continue_latest && let Some(latest) = sessions::latest_session(db)? {
        return Ok(latest.id);
    }
    let model = config
        .chat_model_ref()
        .map_or_else(|_| String::from("unconfigured"), |m| m.to_string());
    Ok(sessions::create_session(db, &model)?.id)
}

/// Resolve a full id or a unique prefix to a session.
fn find_session(db: &WorkspaceDb, prefix: &str) -> Result<sessions::SessionRow> {
    if let Some(exact) = sessions::get_session(db, prefix)? {
        return Ok(exact);
    }
    let matches: Vec<sessions::SessionRow> = sessions::list_sessions(db, 1000)?
        .into_iter()
        .filter(|s| s.id.starts_with(prefix))
        .collect();
    match matches.len() {
        0 => anyhow::bail!("no session matches '{prefix}'; run `quack sessions`"),
        1 => matches.into_iter().next().context("session vanished"),
        n => anyhow::bail!("'{prefix}' matches {n} sessions; use more of the id"),
    }
}

fn list_sessions(db: &WorkspaceDb, json: bool, limit: u32) -> Result<()> {
    let rows = sessions::list_sessions(db, limit)?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    if json {
        for row in &rows {
            serde_json::to_writer(&mut out, row)?;
            writeln!(out)?;
        }
    } else if rows.is_empty() {
        writeln!(out, "No sessions yet.")?;
    } else {
        for row in &rows {
            writeln!(
                out,
                "{}  {}  {:>3} msgs  {}  {}",
                row.id,
                row.updated_at,
                row.message_count,
                row.model,
                row.title.as_deref().unwrap_or("(untitled)")
            )?;
        }
    }
    out.flush()?;
    Ok(())
}

fn export_session(db: &WorkspaceDb, prefix: &str, as_sql: bool) -> Result<()> {
    let session = find_session(db, prefix)?;
    let rows = sessions::messages(db, &session.id)?;
    let text = if as_sql {
        sessions::export_sql(&rows)?
    } else {
        sessions::export_markdown(&session, &rows)?
    };
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    write!(out, "{text}")?;
    out.flush()?;
    Ok(())
}

/// Log to stderr for the non-interactive paths. The terminal session owns
/// the screen, so it does not install a subscriber.
fn init_logging() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
}

async fn resolve_workspace(workspace_name: Option<&str>) -> Result<(Config, WorkspaceRow, String)> {
    let config = Config::load().context("failed to load configuration")?;

    let control = ControlPlane::open(&config)
        .await
        .context("failed to open control plane")?;

    let ws_name =
        workspace_name.map_or_else(|| config.general.default_workspace.clone(), str::to_owned);
    let workspace = control
        .find_or_create_workspace(&ws_name)
        .await
        .context("failed to resolve workspace")?;

    Ok((config, workspace, ws_name))
}

async fn run_query(sql: &str, workspace_name: Option<&str>, format: OutputFormat) -> Result<()> {
    let (config, workspace, _) = resolve_workspace(workspace_name).await?;

    let ws_db =
        WorkspaceDb::open(&config, &workspace.id).context("failed to open workspace database")?;

    let results = ws_db.execute_query(sql).context("query execution failed")?;

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    match format {
        OutputFormat::Table | OutputFormat::Text => results.write_table(&mut out)?,
        OutputFormat::Json => results.write_json(&mut out)?,
        OutputFormat::Ndjson => results.write_ndjson(&mut out)?,
        OutputFormat::Csv => results.write_csv(&mut out)?,
        OutputFormat::Markdown => results.write_markdown(&mut out)?,
    }

    out.flush()?;
    Ok(())
}

async fn run_ingest(
    file: &str,
    workspace_name: Option<&str>,
    filename_override: Option<&str>,
    no_embed: bool,
) -> Result<()> {
    let (config, workspace, _) = resolve_workspace(workspace_name).await?;

    let (data, effective_filename) = read_input(file, filename_override)?;

    let file_type = ingestion::parser::detect_file_type(&effective_filename);

    if file_type.is_structured() {
        let files_dir = config.workspace_files_dir(&workspace.id);
        std::fs::create_dir_all(&files_dir).context("failed to create workspace files dir")?;
        let dest = files_dir.join(&effective_filename);
        std::fs::write(&dest, &data).context("failed to write file to workspace directory")?;
    }

    let ws_db =
        WorkspaceDb::open(&config, &workspace.id).context("failed to open workspace database")?;

    let embedding_model = if no_embed {
        None
    } else {
        llm::optional_embedding_model(&config).context("failed to build embedding model")?
    };

    let result = ingestion::ingest_file(
        &config,
        &ws_db,
        &workspace.id,
        &effective_filename,
        &data,
        embedding_model.as_ref(),
    )
    .await
    .context("ingestion failed")?;

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    writeln!(out, "Ingested: {}", result.filename)?;
    writeln!(out, "  Type: {}", result.file_type)?;
    writeln!(out, "  Document ID: {}", result.document_id)?;

    if let Some(table) = &result.table_name {
        writeln!(out, "  Table: {table}")?;
    }
    if result.chunks_stored > 0 {
        writeln!(out, "  Chunks: {}", result.chunks_stored)?;
        if embedding_model.is_some() {
            writeln!(out, "  Embeddings: generated")?;
        }
    }

    out.flush()?;
    Ok(())
}

fn read_input(file: &str, filename_override: Option<&str>) -> Result<(Vec<u8>, String)> {
    if file == "-" {
        let filename = filename_override
            .ok_or_else(|| anyhow::anyhow!("--filename is required when reading from stdin"))?
            .to_owned();

        let mut data = Vec::new();
        std::io::stdin()
            .read_to_end(&mut data)
            .context("failed to read from stdin")?;

        Ok((data, filename))
    } else {
        let path = PathBuf::from(file);
        let data = std::fs::read(&path).context("failed to read input file")?;

        let filename = filename_override
            .map(String::from)
            .or_else(|| path.file_name().and_then(|n| n.to_str()).map(String::from))
            .unwrap_or_else(|| String::from("unknown"));

        Ok((data, filename))
    }
}
