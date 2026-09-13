#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use quack_core::config::Config;
use quack_core::ingestion;
use quack_core::llm::openai_compat::OpenAiCompatClient;
use quack_core::storage::control::ControlPlane;
use quack_core::storage::workspace::WorkspaceDb;

#[derive(Parser)]
#[command(name = "quack", version, about = "Data analysis platform")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute a SQL query against a workspace
    Query {
        /// SQL query to execute
        sql: String,

        /// Workspace name (defaults to config value)
        #[arg(long, short = 'w')]
        workspace: Option<String>,

        /// Output format
        #[arg(long, short = 'f', value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Ingest a file into a workspace
    Ingest {
        /// File path to ingest (use - for stdin)
        file: String,

        /// Workspace name (defaults to config value)
        #[arg(long, short = 'w')]
        workspace: Option<String>,

        /// Override the filename (required when reading from stdin)
        #[arg(long)]
        filename: Option<String>,

        /// Skip embedding generation
        #[arg(long)]
        no_embed: bool,
    },
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Table,
    Json,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Query {
            sql,
            workspace,
            format,
        } => run_query(&sql, workspace.as_deref(), &format).await,
        Commands::Ingest {
            file,
            workspace,
            filename,
            no_embed,
        } => run_ingest(&file, workspace.as_deref(), filename.as_deref(), no_embed).await,
    }
}

async fn run_query(sql: &str, workspace_name: Option<&str>, format: &OutputFormat) -> Result<()> {
    let config = Config::load().context("failed to load configuration")?;

    let control = ControlPlane::open(&config)
        .await
        .context("failed to open control plane")?;

    let ws_name = workspace_name.unwrap_or(config.general.default_workspace.as_str());
    let workspace = control
        .find_or_create_workspace(ws_name)
        .await
        .context("failed to resolve workspace")?;

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
    let config = Config::load().context("failed to load configuration")?;

    let control = ControlPlane::open(&config)
        .await
        .context("failed to open control plane")?;

    let ws_name = workspace_name.unwrap_or(config.general.default_workspace.as_str());
    let workspace = control
        .find_or_create_workspace(ws_name)
        .await
        .context("failed to resolve workspace")?;

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

    let provider = if no_embed {
        None
    } else {
        build_embedding_provider(&config)?
    };

    let result = ingestion::ingest_file(
        &config,
        &ws_db,
        &workspace.id,
        &effective_filename,
        &data,
        provider.as_ref(),
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
        if provider.is_some() {
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

fn build_embedding_provider(config: &Config) -> Result<Option<OpenAiCompatClient>> {
    let Some((name, provider_config)) = config.find_embedding_provider() else {
        tracing::info!("no embedding provider configured — storing chunks without embeddings");
        return Ok(None);
    };

    tracing::info!(provider = %name, "using embedding provider");

    let client = OpenAiCompatClient::from_config(provider_config)
        .context("failed to build embedding client")?;

    Ok(Some(client))
}
