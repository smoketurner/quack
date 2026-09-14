#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod terminal;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use quack_core::analysis::policy::WritePolicy;
use quack_core::config::Config;
use quack_core::ingestion;
use quack_core::llm;
use quack_core::storage::control::{ControlPlane, WorkspaceRow};
use quack_core::storage::workspace::WorkspaceDb;
use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

/// Exit status for a usage error (bad flags, no terminal for the session).
const EXIT_USAGE: u8 = 2;
/// Exit status when the agent needed a write that was not permitted.
const EXIT_WRITE_REFUSED: u8 = 3;

#[derive(Parser)]
#[command(
    name = "quack",
    version,
    about = "Knowledge engine: documents, tables, and a knowledge graph in one workspace",
    long_about = "With no subcommand, starts the interactive terminal session in a workspace."
)]
struct Cli {
    /// Workspace name (defaults to config value)
    #[arg(long, short = 'w', global = true)]
    workspace: Option<String>,

    /// Let the agent run statements that modify the workspace
    #[arg(long, global = true)]
    allow_write: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute a SQL query against a workspace
    Query {
        /// SQL query to execute
        sql: String,

        /// Output format
        #[arg(long, short = 'f', value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
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

    /// Ask the agent one question and print the answer
    Chat {
        /// Question or message for the agent
        message: String,
    },
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Table,
    Json,
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

    match cli.command {
        None => {
            if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
                init_logging();
                tracing::error!(
                    "the interactive session needs a terminal; use `quack chat` or `quack query` in pipelines"
                );
                return Ok(ExitCode::from(EXIT_USAGE));
            }
            let (config, workspace, ws_name) = resolve_workspace(cli.workspace.as_deref()).await?;
            terminal::run(config, ws_name, workspace.id, cli.allow_write)?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Query { sql, format }) => {
            init_logging();
            run_query(&sql, cli.workspace.as_deref(), &format).await?;
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
        Some(Commands::Chat { message }) => {
            init_logging();
            let refused = run_chat(&message, cli.workspace.as_deref(), policy).await?;
            Ok(if refused {
                ExitCode::from(EXIT_WRITE_REFUSED)
            } else {
                ExitCode::SUCCESS
            })
        }
    }
}

/// Log to stderr for the non-interactive subcommands. The terminal session
/// owns the screen, so it does not install a subscriber.
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

async fn run_query(sql: &str, workspace_name: Option<&str>, format: &OutputFormat) -> Result<()> {
    let (config, workspace, _) = resolve_workspace(workspace_name).await?;

    let ws_db =
        WorkspaceDb::open(&config, &workspace.id).context("failed to open workspace database")?;

    let results = ws_db.execute_query(sql).context("query execution failed")?;

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    match format {
        OutputFormat::Table => results.write_table(&mut out)?,
        OutputFormat::Json => results.write_json(&mut out)?,
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

async fn run_chat(
    message: &str,
    workspace_name: Option<&str>,
    policy: WritePolicy,
) -> Result<bool> {
    let (config, workspace, _) = resolve_workspace(workspace_name).await?;

    let ws_db =
        WorkspaceDb::open(&config, &workspace.id).context("failed to open workspace database")?;

    let response = llm::run_turn(&config, ws_db, policy, message)
        .await
        .context("agent turn failed")?;

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    writeln!(out, "{}", response.content)?;

    if let Some(chart_spec) = &response.chart_spec {
        writeln!(out)?;
        writeln!(out, "--- ECharts Spec ---")?;
        let json =
            serde_json::to_string_pretty(chart_spec).context("failed to serialize chart spec")?;
        writeln!(out, "{json}")?;
    }

    out.flush()?;
    Ok(response.write_refused)
}
