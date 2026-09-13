#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::io::Write;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

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
    }
}

async fn run_query(sql: &str, workspace_name: Option<&str>, format: &OutputFormat) -> Result<()> {
    let config = quack_core::config::Config::load().context("failed to load configuration")?;

    let control = quack_core::storage::control::ControlPlane::open(&config)
        .await
        .context("failed to open control plane")?;

    let ws_name = workspace_name.unwrap_or(config.general.default_workspace.as_str());
    let workspace = control
        .find_or_create_workspace(ws_name)
        .await
        .context("failed to resolve workspace")?;

    let ws_db = quack_core::storage::workspace::WorkspaceDb::open(&config, &workspace.id)
        .context("failed to open workspace database")?;

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
