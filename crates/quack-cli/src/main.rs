#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use quack_core::analysis::agent;
use quack_core::config::{self, Config, ProviderConfig};
use quack_core::ingestion;
use quack_core::storage::control::ControlPlane;
use quack_core::storage::workspace::WorkspaceDb;
use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};
use rig::prelude::*;
use std::io::{Read, Write};
use std::path::PathBuf;

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

    /// Chat with the analysis agent
    Chat {
        /// Question or message for the agent
        message: String,

        /// Workspace name (defaults to config value)
        #[arg(long, short = 'w')]
        workspace: Option<String>,
    },
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Table,
    Json,
}

// ---------------------------------------------------------------------------
// Provider-agnostic embedding model enum
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum EmbedModel {
    Ollama(rig::providers::ollama::EmbeddingModel),
    OpenAi(
        rig::providers::openai::GenericEmbeddingModel<rig::providers::openai::OpenAICompletionsExt>,
    ),
}

impl EmbeddingModel for EmbedModel {
    const MAX_DOCUMENTS: usize = 1024;
    type Client = rig::providers::ollama::Client;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self::Ollama(rig::providers::ollama::EmbeddingModel::make(
            client, model, dims,
        ))
    }

    fn ndims(&self) -> usize {
        match self {
            Self::Ollama(m) => m.ndims(),
            Self::OpenAi(m) => m.ndims(),
        }
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        match self {
            Self::Ollama(m) => m.embed_texts(texts).await,
            Self::OpenAi(m) => m.embed_texts(texts).await,
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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
        Commands::Chat { message, workspace } => run_chat(&message, workspace.as_deref()).await,
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

    let embedding_model = if no_embed {
        None
    } else {
        build_embedding_model(&config)?
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

async fn run_chat(message: &str, workspace_name: Option<&str>) -> Result<()> {
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

    let (_, chat_config) = config.find_chat_provider().ok_or_else(|| {
        anyhow::anyhow!(
            "no LLM provider configured with a chat model — \
             add a [providers.<name>] section with 'model' set in {}",
            config::config_file_path().display()
        )
    })?;

    let chat_model_name = chat_config
        .model
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("chat provider has no model configured"))?
        .to_owned();

    let (embedding_model, embed_model_name) = build_rig_embedding_model(&config)?;

    tracing::info!(
        chat_model = %chat_model_name,
        embed_model = %embed_model_name,
        provider_type = %chat_config.provider_type,
        "starting chat agent"
    );

    let response = match chat_config.provider_type.as_str() {
        "ollama" => {
            let client = build_ollama_client(chat_config)?;
            let completion_model = client.completion_model(&chat_model_name);
            agent::run_analysis(
                ws_db,
                completion_model,
                embedding_model,
                &config.analysis,
                &config.retrieval,
                message,
            )
            .await
        }
        "openai" => {
            let client = build_openai_client(chat_config)?;
            let completion_model = client.completion_model(&chat_model_name);
            agent::run_analysis(
                ws_db,
                completion_model,
                embedding_model,
                &config.analysis,
                &config.retrieval,
                message,
            )
            .await
        }
        "anthropic" => {
            let client = build_anthropic_client(chat_config)?;
            let completion_model = client.completion_model(&chat_model_name);
            agent::run_analysis(
                ws_db,
                completion_model,
                embedding_model,
                &config.analysis,
                &config.retrieval,
                message,
            )
            .await
        }
        other => anyhow::bail!(
            "unsupported provider type '{other}' — expected 'ollama', 'openai', or 'anthropic'"
        ),
    }
    .context("agent loop failed")?;

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
    Ok(())
}

// ---------------------------------------------------------------------------
// Provider client builders
// ---------------------------------------------------------------------------

fn resolve_api_key(provider_config: &ProviderConfig) -> Option<String> {
    provider_config
        .api_key_env
        .as_ref()
        .and_then(|env| std::env::var(env).ok())
}

fn build_ollama_client(provider_config: &ProviderConfig) -> Result<rig::providers::ollama::Client> {
    let api_key = resolve_api_key(provider_config)
        .map(rig::providers::ollama::OllamaApiKey::from)
        .unwrap_or_default();

    let mut builder = rig::providers::ollama::Client::builder().api_key(api_key);

    if let Some(base_url) = &provider_config.base_url {
        let url = base_url.trim_end_matches("/v1");
        builder = builder.base_url(url);
    }

    builder.build().context("failed to build Ollama client")
}

fn build_openai_client(
    provider_config: &ProviderConfig,
) -> Result<rig::providers::openai::CompletionsClient> {
    let api_key = resolve_api_key(provider_config)
        .ok_or_else(|| anyhow::anyhow!("OpenAI provider requires api_key_env to be set"))?;

    let mut builder = rig::providers::openai::CompletionsClient::builder().api_key(&api_key);

    if let Some(base_url) = &provider_config.base_url {
        builder = builder.base_url(base_url);
    }

    builder.build().context("failed to build OpenAI client")
}

fn build_anthropic_client(
    provider_config: &ProviderConfig,
) -> Result<rig::providers::anthropic::Client> {
    let api_key = resolve_api_key(provider_config)
        .ok_or_else(|| anyhow::anyhow!("Anthropic provider requires api_key_env to be set"))?;

    let mut builder = rig::providers::anthropic::Client::builder().api_key(&api_key);

    if let Some(base_url) = &provider_config.base_url {
        builder = builder.base_url(base_url);
    }

    builder.build().context("failed to build Anthropic client")
}

// ---------------------------------------------------------------------------
// Embedding model builders (provider-agnostic via EmbedModel enum)
// ---------------------------------------------------------------------------

fn build_rig_embedding_model(config: &Config) -> Result<(EmbedModel, String)> {
    let (name, embed_config) = config.find_embedding_provider().ok_or_else(|| {
        anyhow::anyhow!(
            "no embedding provider configured — \
             add a [providers.<name>] section with 'embedding_model' set in {}",
            config::config_file_path().display()
        )
    })?;

    tracing::info!(provider = %name, "using embedding provider");

    let model_name = embed_config
        .embedding_model
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("embedding provider has no embedding_model configured"))?;

    let ndims = embed_config.embedding_dimension.ok_or_else(|| {
        anyhow::anyhow!("embedding provider has no embedding_dimension configured")
    })?;

    let ndims_usize = usize::try_from(ndims).context("embedding_dimension overflow")?;

    let model = match embed_config.provider_type.as_str() {
        "ollama" => {
            let client = build_ollama_client(embed_config)?;
            EmbedModel::Ollama(client.embedding_model_with_ndims(model_name, ndims_usize))
        }
        "openai" => {
            let client = build_openai_client(embed_config)?;
            EmbedModel::OpenAi(client.embedding_model_with_ndims(model_name, ndims_usize))
        }
        other => anyhow::bail!(
            "provider type '{other}' does not support embeddings — use 'ollama' or 'openai'"
        ),
    };

    Ok((model, model_name.to_owned()))
}

fn build_embedding_model(config: &Config) -> Result<Option<EmbedModel>> {
    let Some((name, embed_config)) = config.find_embedding_provider() else {
        tracing::info!("no embedding provider configured — storing chunks without embeddings");
        return Ok(None);
    };

    tracing::info!(provider = %name, "using embedding provider for ingestion");

    let model_name = embed_config
        .embedding_model
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("embedding provider has no embedding_model configured"))?;

    let ndims = embed_config.embedding_dimension.ok_or_else(|| {
        anyhow::anyhow!("embedding provider has no embedding_dimension configured")
    })?;

    let ndims_usize = usize::try_from(ndims).context("embedding_dimension overflow")?;

    let model = match embed_config.provider_type.as_str() {
        "ollama" => {
            let client = build_ollama_client(embed_config)?;
            EmbedModel::Ollama(client.embedding_model_with_ndims(model_name, ndims_usize))
        }
        "openai" => {
            let client = build_openai_client(embed_config)?;
            EmbedModel::OpenAi(client.embedding_model_with_ndims(model_name, ndims_usize))
        }
        other => anyhow::bail!(
            "provider type '{other}' does not support embeddings — use 'ollama' or 'openai'"
        ),
    };

    Ok(Some(model))
}
