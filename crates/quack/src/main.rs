#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod admin;
mod config_cli;
mod doctor_cli;
mod embeddings_cli;
mod graph_cli;
mod mcp;
mod ontology_cli;
mod print;
mod server;
mod terminal;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{SharedDb, open_reader};
use quack_core::config;
use quack_core::config::AuthMode;
use quack_core::config::Config;
use quack_core::crypto;
use quack_core::error::Error as CoreError;
use quack_core::import::{self, ImportPolicy, ImportRequest};
use quack_core::ingestion::{self, IngestOutcome, NewFile};
use quack_core::llm;
use quack_core::llm::oauth::{self, TokenManager};
use quack_core::llm::oauth::{LoginOptions, LoginPrompt};
use quack_core::okf::{self, Bundle};
use quack_core::ontology::store as ontology_store;
use quack_core::ontology::{Ontology, candidates};
use quack_core::progress::RunControl;
use quack_core::storage::context;
use quack_core::storage::control::{ControlPlane, WorkspaceRow};
use quack_core::storage::sessions::{self, ChatMode};
use quack_core::storage::workspace::{DocumentSource, WorkspaceDb};
use quack_core::storage::writer::Writer;
use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

/// Exit status for a usage error (bad flags, no terminal for the session).
const EXIT_USAGE: u8 = 2;
/// Exit status when the agent needed a write that was not permitted.
const EXIT_WRITE_REFUSED: u8 = 3;
/// Exit status when an OAuth provider needs `quack auth login` first.
const EXIT_AUTH_REQUIRED: u8 = 4;
/// Exit status when `quack config` finds a configuration every other
/// command would refuse (the report is still printed), or `-p` has no chat
/// model to answer with.
const EXIT_BAD_CONFIG: u8 = 2;

/// `--version` names the crypto module as well, so an operator can tell a FIPS
/// binary from a non-FIPS one without turning on `RUST_LOG=info`. `-V` stays
/// the bare version. A static because clap takes a `&'static str`.
static LONG_VERSION: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    format!(
        "{}\n{}",
        env!("CARGO_PKG_VERSION"),
        quack_core::crypto::provider_description()
    )
});

#[derive(Parser)]
#[command(
    name = "quack",
    version,
    long_version = LONG_VERSION.as_str(),
    about = "Knowledge engine: documents, tables, and a knowledge graph in one workspace",
    long_about = "With no arguments, starts the interactive terminal session in a workspace.\n\
                  `-p PROMPT` asks the agent one question and prints the answer; \
                  `-q SQL` runs SQL directly. Both are pipe-friendly."
)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each is an independent command-line switch"
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

    /// Answer mode: chat may use general knowledge; query answers only from
    /// retrieved chunks and query results
    #[arg(long, value_enum, global = true)]
    mode: Option<ModeArg>,

    /// Print full tool inputs and outputs to stderr in print mode
    #[arg(long, global = true)]
    verbose: bool,

    /// Wait for piped stdin to close before running (`-p` and `-q` load
    /// it as the `stdin` table). Without it, a pipe that has nothing to
    /// read within a second is skipped
    #[arg(long)]
    stdin: bool,

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

        /// Title to record; otherwise the first heading, when there is one
        #[arg(long)]
        title: Option<String>,

        /// Skip embedding generation
        #[arg(long)]
        no_embed: bool,

        /// Pin the document: its full text goes into every prompt
        #[arg(long)]
        pin: bool,
    },

    /// Show, edit, or move the workspace context (the owner's instructions
    /// and definitions for the agent)
    Context {
        #[command(subcommand)]
        action: Option<ContextAction>,
    },

    /// Log in to an OAuth provider, show token state, or forget a token
    Auth {
        #[command(subcommand)]
        action: AuthAction,
    },

    /// Server users: create one or list them
    User {
        #[command(subcommand)]
        action: admin::UserAction,
    },

    /// API tokens scoped to a workspace: create, list, or revoke
    Token {
        #[command(subcommand)]
        action: admin::TokenAction,
    },

    /// Workspace membership: add, remove, or list members
    Member {
        #[command(subcommand)]
        action: admin::MemberAction,
    },

    /// Read the access audit log with filters
    Audit(admin::AuditArgs),

    /// Serve the REST API and web UI
    Serve {
        /// Listen address (default from `[server].bind`, or `QUACK_BIND`)
        #[arg(long)]
        bind: Option<String>,
        /// No authentication, one implicit user; loopback only
        #[arg(long)]
        local: bool,
    },

    /// Serve the workspace as an MCP server over stdio (for Claude Code
    /// and editors); logs go to stderr
    Mcp {
        /// Let the `sql` and `query` tools run statements that modify data
        #[arg(long)]
        allow_write: bool,
    },

    /// Show, install, import, export, diff, or restore the ontology
    Ontology {
        #[command(subcommand)]
        action: ontology_cli::OntologyAction,
    },

    /// Explore, build, revalidate, and review the knowledge graph
    Graph {
        #[command(subcommand)]
        action: graph_cli::GraphAction,
    },

    /// Pull rows from Postgres, SQLite, or a data file over HTTP(S) into
    /// a workspace table (the Rust-side replacement for ATTACH)
    Import {
        /// A Postgres URL (user, password, host, database), a SQLite path
        /// as `sqlite:PATH`, or an http(s) URL of a data file
        url: String,
        /// The workspace table to create (replaced when it exists)
        #[arg(long)]
        table: String,
        /// A query to run on the source
        #[arg(long, conflicts_with = "from")]
        query: Option<String>,
        /// Pull a whole source table instead of a query
        #[arg(long, value_name = "SOURCE_TABLE")]
        from: Option<String>,
        /// Rows to pull at most (capped by `[import].max_rows`)
        #[arg(long)]
        limit: Option<u64>,
    },

    /// Move the workspace as an Open Knowledge Format bundle
    Okf {
        #[command(subcommand)]
        action: OkfAction,
    },

    /// Show what this binary makes of config.toml: every setting it
    /// recognizes, the value in force and where it came from, and the
    /// keys in the file it does not recognize
    Config {
        /// Only the settings the file or the environment has a say in
        #[arg(long)]
        changed: bool,

        /// Emit the whole report as one JSON document
        #[arg(long)]
        json: bool,
    },

    /// Check the setup and say how to fix what is wrong: the config file,
    /// the data directory, the workspace, each model's provider (reached
    /// over the network), and the server's bind address
    Doctor {
        /// Skip the network probes
        #[arg(long)]
        offline: bool,

        /// Emit the checks as one JSON document
        #[arg(long)]
        json: bool,
    },

    /// List ingested documents, or pin and unpin one
    Docs {
        /// Pin a document by id (prefixes accepted)
        #[arg(long, value_name = "DOCUMENT_ID", conflicts_with = "unpin")]
        pin: Option<String>,

        /// Unpin a document by id (prefixes accepted)
        #[arg(long, value_name = "DOCUMENT_ID")]
        unpin: Option<String>,

        /// Delete a document by id (prefixes accepted), with its chunks and
        /// the table it was loaded as
        #[arg(long, value_name = "DOCUMENT_ID")]
        delete: Option<String>,

        /// Emit one JSON object per document
        #[arg(long)]
        json: bool,
    },

    /// The workspace's vectors: refresh the ones made with another
    /// embedding model, width, or input prefixes
    Embeddings {
        #[command(subcommand)]
        action: embeddings_cli::EmbeddingsAction,
    },
}

#[derive(Subcommand)]
enum OkfAction {
    /// Write the workspace as a bundle: index.md from the context, one
    /// Markdown file per table, class, relation, property, document, and
    /// graph node, and log.md
    Export {
        /// Directory to write (created), or - for a tar archive on stdout
        dir: String,
    },
}

#[derive(Subcommand)]
enum AuthAction {
    /// Obtain a token: browser sign-in with PKCE, or a device code when no
    /// browser can open here
    Login {
        /// Provider name from [providers.NAME] with auth = "oauth"
        provider: String,

        /// Use the device-code flow even if a browser is available
        #[arg(long)]
        device_code: bool,
    },
    /// Show whether each OAuth provider has a token and when it expires
    Status {
        /// Only this provider
        provider: Option<String>,
    },
    /// Forget the cached token for a provider
    Logout {
        /// Provider name from [providers.NAME] with auth = "oauth"
        provider: String,
    },
}

#[derive(Subcommand)]
enum ContextAction {
    /// Print the current context (default)
    Show,
    /// Open the context in $EDITOR and store the result as a new version
    Edit,
    /// List versions, newest first
    History {
        #[arg(long, default_value_t = 20)]
        limit: u32,
    },
    /// Write the current context to a Markdown file
    Export {
        /// Destination path (- for stdout)
        file: String,
    },
    /// Replace the context with the contents of a Markdown file
    Import {
        /// Source path (- for stdin)
        file: String,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ModeArg {
    Chat,
    Query,
}

impl From<ModeArg> for ChatMode {
    fn from(mode: ModeArg) -> Self {
        match mode {
            ModeArg::Chat => Self::Chat,
            ModeArg::Query => Self::Query,
        }
    }
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
    crypto::install_default_provider()
        .context("failed to install the aws-lc-rs crypto provider")?;

    match run().await {
        // The reader closed the pipe (`quack ... | head -1`): the command
        // did its job, so stop quietly like `git` and `ls` do (issue #68).
        Err(e) if is_broken_pipe(&e) => Ok(ExitCode::SUCCESS),
        // Any command that reached a provider without a usable token exits
        // 4, so scripts can tell "run `quack auth login`" from a failure.
        Err(e) => match auth_exit_code(&e) {
            Some(code) => {
                tracing::error!("{e:#}");
                Ok(code)
            }
            None => Err(e),
        },
        outcome => outcome,
    }
}

/// Whether an error is a write to a closed stdout or stderr: a bare
/// `io::Error`, one behind `serde_json`, or either inside core's
/// transparent `Io` and `Json` variants (which hide them from the chain).
fn is_broken_pipe(error: &anyhow::Error) -> bool {
    use quack_core::error::Error as Core;
    error.chain().any(|cause| {
        let kind = if let Some(e) = cause.downcast_ref::<std::io::Error>() {
            Some(e.kind())
        } else if let Some(e) = cause.downcast_ref::<serde_json::Error>() {
            e.io_error_kind()
        } else {
            match cause.downcast_ref::<Core>() {
                Some(Core::Io(e)) => Some(e.kind()),
                Some(Core::Json(e)) => e.io_error_kind(),
                _ => None,
            }
        };
        kind == Some(std::io::ErrorKind::BrokenPipe)
    })
}

async fn run() -> Result<ExitCode> {
    let mut cli = Cli::parse();

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
        run_query(sql, cli.workspace.as_deref(), format, cli.stdin).await?;
        return Ok(ExitCode::SUCCESS);
    }

    match cli.command.take() {
        None => run_terminal_session(&cli, stdout_is_tty).await,
        Some(command) => run_command(&cli, command).await,
    }
}

/// Every subcommand; print mode, `-q`, and the terminal session are
/// dispatched by `main` itself.
async fn run_command(cli: &Cli, command: Commands) -> Result<ExitCode> {
    match command {
        Commands::Sessions { json, limit } => run_sessions(cli, json, limit).await,
        Commands::Export {
            session_id,
            sql,
            markdown: _,
        } => run_export(cli, &session_id, sql).await,
        Commands::Ingest {
            file,
            filename,
            title,
            no_embed,
            pin,
        } => {
            init_logging();
            run_ingest(
                &file,
                cli.workspace.as_deref(),
                filename.as_deref(),
                title.as_deref(),
                no_embed,
                pin,
            )
            .await?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Ontology { action } => run_ontology(cli, action).await,
        Commands::Graph { action } => run_graph(cli, action).await,
        Commands::Embeddings { action } => run_embeddings(cli, action).await,
        Commands::Okf {
            action: OkfAction::Export { dir },
        } => run_okf_export(cli, &dir).await,
        Commands::Import {
            url,
            table,
            query,
            from,
            limit,
        } => run_import(cli, url, table, query, from, limit).await,
        Commands::Context { action } => {
            let ws_db = open_workspace(cli).await?;
            run_context(&ws_db, action.unwrap_or(ContextAction::Show))?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Auth { action } => {
            init_logging();
            let config = Config::load().context("failed to load configuration")?;
            run_auth(&config, action).await?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Mcp { allow_write } => run_mcp(cli, allow_write).await,
        Commands::Serve { bind, local } => {
            // The server logs each request at info; other commands stay quiet.
            init_logging_at("info,sqlx=warn,hyper=warn,h2=warn");
            let config = Config::load().context("failed to load configuration")?;
            server::serve(config, bind, local).await?;
            Ok(ExitCode::SUCCESS)
        }
        command @ (Commands::User { .. }
        | Commands::Token { .. }
        | Commands::Member { .. }
        | Commands::Audit(_)) => {
            init_logging();
            let config = Config::load().context("failed to load configuration")?;
            run_admin(&config, cli.workspace.as_deref(), command).await?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Config { changed, json } => run_config(changed, json),
        Commands::Doctor { offline, json } => run_doctor(cli, offline, json).await,
        Commands::Docs {
            pin,
            unpin,
            delete,
            json,
        } => {
            let ws_db = open_workspace(cli).await?;
            run_docs(
                &ws_db,
                pin.as_deref(),
                unpin.as_deref(),
                delete.as_deref(),
                json,
            )?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// `quack config`: what this binary makes of `config.toml`. It reads the
/// file outside `Config::load`, so it reports a file every other command
/// refuses rather than failing the same way, and says so in its status.
fn run_config(changed: bool, json: bool) -> Result<ExitCode> {
    init_logging();
    let inspection = config::inspect::Inspection::load();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let usable = config_cli::run(&mut out, &inspection, json, changed)?;
    out.flush()?;
    Ok(if usable {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(EXIT_BAD_CONFIG)
    })
}

/// `quack doctor`: like `quack config`, it inspects the file outside
/// `Config::load`, so a file every other command refuses is a finding here
/// rather than the error. Exits 1 when any check fails.
async fn run_doctor(cli: &Cli, offline: bool, json: bool) -> Result<ExitCode> {
    init_logging();
    let inspection = config::inspect::Inspection::load();
    let options = quack_core::doctor::Options {
        workspace: cli.workspace.clone(),
        offline,
        ..quack_core::doctor::Options::default()
    };
    let report = quack_core::doctor::run(&inspection, &options).await;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    doctor_cli::write(&mut out, &report, json)?;
    out.flush()?;
    Ok(if report.has_failures() {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}

/// `quack -p PROMPT`: one turn, answer to stdout, steps to stderr.
async fn run_print_mode(cli: &Cli, prompt: &str, policy: WritePolicy) -> Result<ExitCode> {
    init_logging();
    let format = match cli.format.unwrap_or(OutputFormat::Text) {
        OutputFormat::Json => print::PromptFormat::Json,
        OutputFormat::Text => print::PromptFormat::Text,
        OutputFormat::Table | OutputFormat::Ndjson | OutputFormat::Csv | OutputFormat::Markdown => {
            tracing::error!("-p accepts only --format text or json");
            return Ok(ExitCode::from(EXIT_USAGE));
        }
    };
    let (config, workspace, workspace_name) = resolve_workspace(cli.workspace.as_deref()).await?;
    // Checked before the workspace opens, so a missing model is one line
    // on stderr rather than a failed turn, and leaves no session behind.
    if let Err(e) = config.chat_model_ref() {
        tracing::error!("{e}");
        return Ok(ExitCode::from(EXIT_BAD_CONFIG));
    }
    let ws_db =
        WorkspaceDb::open(&config, &workspace.id).context("failed to open workspace database")?;
    load_piped_stdin(&config, &ws_db, &workspace.id, cli.stdin).await?;
    if let Some(note) = ws_db.embedding_status()?.note() {
        tracing::warn!("{note} Run `quack embeddings refresh -w {workspace_name}` to update them.");
    }
    let session_id = resolve_session(
        &config,
        &ws_db,
        cli.continue_latest,
        cli.resume.as_deref(),
        cli.mode.map(ChatMode::from),
    )?;
    let db: SharedDb =
        Arc::new(Writer::spawn(ws_db).context("failed to start the workspace writer")?);
    let reader_db = open_reader(&db, config.analysis.reader_pool_size).await;
    let outcome = print::run_prompt(
        &config,
        Arc::clone(&db),
        reader_db,
        &session_id,
        policy,
        prompt,
        format,
        cli.verbose,
    )
    .await;
    if outcome.is_err() {
        let id = session_id.clone();
        drop(db.run(move |db| sessions::delete_if_empty(db, &id)).await);
    }
    let refused = outcome?;
    Ok(if refused {
        ExitCode::from(EXIT_WRITE_REFUSED)
    } else {
        ExitCode::SUCCESS
    })
}

/// When stdin is a pipe or file rather than a terminal, its bytes become
/// the temporary table `stdin` for this invocation (CSV, JSON, or Parquet).
///
/// A pipe that stays open with nothing to read (a supervisor's inherited
/// stdin, `sleep 1000 | quack -q ...`) would block the command forever
/// (issue #66): unless `wait` (`--stdin`) says so, a pipe gets
/// [`STDIN_GRACE`] to deliver a byte or close, and is otherwise skipped
/// with a warning.
async fn load_piped_stdin(
    config: &Config,
    db: &WorkspaceDb,
    workspace_id: &str,
    wait: bool,
) -> Result<()> {
    if std::io::stdin().is_terminal() {
        return Ok(());
    }
    if !wait && !stdin_has_data().await? {
        tracing::warn!(
            "stdin is not a terminal but had nothing to read within {} s; \
             skipping the `stdin` table (pass --stdin to wait for it)",
            STDIN_GRACE.as_secs()
        );
        return Ok(());
    }
    let mut data = Vec::new();
    std::io::stdin()
        .read_to_end(&mut data)
        .context("failed to read stdin")?;
    if let Some(table) = ingestion::load_stdin_table(config, db, workspace_id, &data)
        .context("failed to load stdin as a table")?
    {
        tracing::info!(
            table,
            bytes = data.len(),
            "stdin loaded as a temporary table"
        );
    }
    Ok(())
}

/// How long a non-terminal stdin has to deliver a byte or close.
const STDIN_GRACE: Duration = Duration::from_secs(1);

/// Whether stdin is worth reading: a pipe or socket is when it becomes
/// readable (data or end of file) within [`STDIN_GRACE`]; anything else
/// (a regular file, `/dev/null`) answers a read at once.
#[cfg(unix)]
async fn stdin_has_data() -> Result<bool> {
    use std::os::fd::{AsFd, BorrowedFd};
    use std::os::unix::fs::FileTypeExt;
    let stdin = std::io::stdin();
    let kind = std::fs::File::from(stdin.as_fd().try_clone_to_owned()?)
        .metadata()
        .context("failed to inspect stdin")?
        .file_type();
    if !kind.is_fifo() && !kind.is_socket() {
        return Ok(true);
    }
    let fd: BorrowedFd<'_> = stdin.as_fd();
    let watch = tokio::io::unix::AsyncFd::with_interest(fd, tokio::io::Interest::READABLE)
        .context("failed to watch stdin")?;
    Ok(tokio::time::timeout(STDIN_GRACE, watch.readable())
        .await
        .is_ok())
}

/// Windows has no readiness poll for stdin: read it as before.
#[cfg(not(unix))]
async fn stdin_has_data() -> Result<bool> {
    Ok(true)
}

/// Logging, then the workspace database for the workspace-local subcommands.
async fn open_workspace(cli: &Cli) -> Result<WorkspaceDb> {
    init_logging();
    let (config, workspace, _) = resolve_workspace(cli.workspace.as_deref()).await?;
    WorkspaceDb::open(&config, &workspace.id).context("failed to open workspace database")
}

/// The workspace's connection on a writer thread of its own: the commands
/// that run core's multi-step work (ingestion, import, graph and ontology
/// commands) send it their steps exactly as the server and the terminal do.
async fn open_writer(cli: &Cli) -> Result<Writer> {
    init_logging();
    let (config, workspace, _) = resolve_workspace(cli.workspace.as_deref()).await?;
    spawn_writer(&config, &workspace.id)
}

/// Open workspace `id` and move its connection onto a writer thread.
fn spawn_writer(config: &Config, id: &str) -> Result<Writer> {
    let ws_db = WorkspaceDb::open(config, id).context("failed to open workspace database")?;
    Writer::spawn(ws_db).context("failed to start the workspace writer")
}

/// `quack user|token|member|audit`: server administration from the shell.
async fn run_admin(config: &Config, workspace: Option<&str>, command: Commands) -> Result<()> {
    match command {
        Commands::User { action } => admin::run_user(config, action).await,
        Commands::Token { action } => admin::run_token(config, workspace, action).await,
        Commands::Member { action } => admin::run_member(config, workspace, action).await,
        Commands::Audit(args) => admin::run_audit(config, args).await,
        Commands::Sessions { .. }
        | Commands::Export { .. }
        | Commands::Ingest { .. }
        | Commands::Context { .. }
        | Commands::Ontology { .. }
        | Commands::Graph { .. }
        | Commands::Okf { .. }
        | Commands::Import { .. }
        | Commands::Auth { .. }
        | Commands::Serve { .. }
        | Commands::Mcp { .. }
        | Commands::Config { .. }
        | Commands::Doctor { .. }
        | Commands::Embeddings { .. }
        | Commands::Docs { .. } => Ok(()),
    }
}

/// `quack sessions`: the session list.
async fn run_sessions(cli: &Cli, json: bool, limit: u32) -> Result<ExitCode> {
    let ws_db = open_workspace(cli).await?;
    list_sessions(&ws_db, json, limit)?;
    Ok(ExitCode::SUCCESS)
}

/// `quack export SESSION`: a session as SQL or Markdown.
async fn run_export(cli: &Cli, session_id: &str, sql: bool) -> Result<ExitCode> {
    let ws_db = open_workspace(cli).await?;
    export_session(&ws_db, session_id, sql)?;
    Ok(ExitCode::SUCCESS)
}

/// `quack graph ...`: the knowledge graph from the shell.
async fn run_graph(cli: &Cli, action: graph_cli::GraphAction) -> Result<ExitCode> {
    let ws_db = open_writer(cli).await?;
    let config = Config::load().context("failed to load configuration")?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    graph_cli::run(
        &config,
        &ws_db,
        action,
        &mut out,
        &ontology_cli::chunk_progress,
    )
    .await?;
    Ok(ExitCode::SUCCESS)
}

async fn run_embeddings(cli: &Cli, action: embeddings_cli::EmbeddingsAction) -> Result<ExitCode> {
    let ws_db = open_writer(cli).await?;
    let config = Config::load().context("failed to load configuration")?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    embeddings_cli::run(
        &config,
        &ws_db,
        action,
        &mut out,
        RunControl {
            progress: &embeddings_cli::print_progress,
            cancel: None,
        },
    )
    .await?;
    Ok(ExitCode::SUCCESS)
}

async fn run_ontology(cli: &Cli, action: ontology_cli::OntologyAction) -> Result<ExitCode> {
    let ws_db = open_writer(cli).await?;
    let config = Config::load().context("failed to load configuration")?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    ontology_cli::run(
        &config,
        &ws_db,
        action,
        &mut out,
        &ontology_cli::chunk_progress,
    )
    .await?;
    Ok(ExitCode::SUCCESS)
}

/// `quack import URL --table NAME`: rows from an external source as a
/// workspace table.
async fn run_import(
    cli: &Cli,
    url: String,
    table: String,
    query: Option<String>,
    from: Option<String>,
    limit: Option<u64>,
) -> Result<ExitCode> {
    init_logging();
    let request = &ImportRequest {
        url,
        table,
        query,
        source_table: from,
        limit,
    };
    let (config, workspace, _) = resolve_workspace(cli.workspace.as_deref()).await?;
    let ws_db = spawn_writer(&config, &workspace.id)?;
    let embedding_model = llm::optional_embedding_model(&config).await?;
    let summary = import::import(
        &config,
        &ws_db,
        &workspace.id,
        request,
        ImportPolicy::owner(),
        embedding_model.as_ref(),
        None,
    )
    .await
    .context("import failed")?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(
        out,
        "Imported {} rows from {} as table \"{}\" ({} columns: {}).",
        summary.rows,
        summary.source,
        summary.table,
        summary.columns.len(),
        summary.columns.join(", ")
    )?;
    Ok(ExitCode::SUCCESS)
}

/// `quack okf export DIR`: the workspace as an Open Knowledge Format
/// bundle, a directory of Markdown files or a tar on stdout.
async fn run_okf_export(cli: &Cli, dir: &str) -> Result<ExitCode> {
    init_logging();
    let (config, workspace, name) = resolve_workspace(cli.workspace.as_deref()).await?;
    let ws_db =
        WorkspaceDb::open(&config, &workspace.id).context("failed to open workspace database")?;
    let bundle = okf::export(&ws_db, &name)?;
    if dir == "-" {
        let mut out = std::io::stdout().lock();
        out.write_all(&bundle.to_tar()?)?;
        out.flush()?;
    } else {
        bundle.write_to(&PathBuf::from(dir))?;
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        writeln!(out, "wrote {} files to {dir}", bundle.files.len())?;
    }
    Ok(ExitCode::SUCCESS)
}

/// `quack mcp`: the workspace as an MCP server on stdin and stdout.
async fn run_mcp(cli: &Cli, allow_write: bool) -> Result<ExitCode> {
    init_logging();
    let (config, workspace, _) = resolve_workspace(cli.workspace.as_deref()).await?;
    let ws_db =
        WorkspaceDb::open(&config, &workspace.id).context("failed to open workspace database")?;
    let db: SharedDb =
        Arc::new(Writer::spawn(ws_db).context("failed to start the workspace writer")?);
    let reader_db = open_reader(&db, config.analysis.reader_pool_size).await;
    let policy = if allow_write {
        WritePolicy::Allow
    } else {
        WritePolicy::Deny
    };
    mcp::serve_stdio(config, db, reader_db, workspace, policy).await?;
    Ok(ExitCode::SUCCESS)
}

/// Exit 4 when the failure is an OAuth provider without a usable token,
/// found anywhere in the error's chain: no command but `quack auth login`
/// can run a login flow, so the message names it. `main` applies this to
/// every command's error, so no command maps it itself.
fn auth_exit_code(err: &anyhow::Error) -> Option<ExitCode> {
    err.chain()
        .any(|cause| {
            matches!(
                cause.downcast_ref::<CoreError>(),
                Some(CoreError::AuthRequired { .. })
            )
        })
        .then_some(ExitCode::from(EXIT_AUTH_REQUIRED))
}

/// `quack auth login|status|logout`.
async fn run_auth(config: &Config, action: AuthAction) -> Result<()> {
    let stdout = std::io::stdout();
    match action {
        AuthAction::Login {
            provider,
            device_code,
        } => {
            let manager = oauth_manager(config, &provider)?;
            let options = LoginOptions {
                device_code: device_code || !browser_can_open(),
            };
            let token = manager.login(options, &show_login_prompt).await?;
            let mut out = stdout.lock();
            writeln!(
                out,
                "Logged in to '{provider}'; the token expires at {}{}.",
                token.expires_at,
                if token.refresh_token.is_some() {
                    " and will refresh itself"
                } else {
                    ""
                }
            )?;
            out.flush()?;
        }
        AuthAction::Status { provider } => {
            let mut names: Vec<&str> = config
                .providers
                .iter()
                .filter(|(_, p)| p.auth == AuthMode::Oauth)
                .map(|(name, _)| name.as_str())
                .collect();
            if let Some(only) = provider.as_deref() {
                names.retain(|n| *n == only);
                if names.is_empty() {
                    anyhow::bail!("'{only}' is not a provider with auth = \"oauth\"");
                }
            }
            let mut out = std::io::BufWriter::new(stdout.lock());
            if names.is_empty() {
                writeln!(out, "No providers use auth = \"oauth\".")?;
            }
            for name in names {
                let status = oauth_manager(config, name)?.status().await?;
                let state = match status.expires_at {
                    Some(at) if status.logged_in => format!(
                        "logged in, token expires {at}{}",
                        if status.has_refresh_token {
                            ", refreshable"
                        } else {
                            ", no refresh token"
                        }
                    ),
                    _ => format!("not logged in; run `quack auth login {name}`"),
                };
                writeln!(out, "{name}: {state} (key in {})", status.key_source)?;
            }
            out.flush()?;
        }
        AuthAction::Logout { provider } => {
            oauth_manager(config, &provider)?.logout().await?;
            let mut out = stdout.lock();
            writeln!(out, "Logged out of '{provider}'.")?;
            out.flush()?;
        }
    }
    Ok(())
}

fn oauth_manager(config: &Config, name: &str) -> Result<Arc<TokenManager>> {
    let provider = config.providers.get(name).ok_or_else(|| {
        anyhow::anyhow!(
            "provider '{name}' is not configured; add [providers.{name}] with auth = \"oauth\""
        )
    })?;
    if provider.auth != AuthMode::Oauth {
        anyhow::bail!("provider '{name}' does not use auth = \"oauth\"");
    }
    oauth::shared_manager(&config.tokens_dir(), name, provider)
        .context("failed to prepare the OAuth token manager")
}

/// Print what the user must do for a login step. The browser prompt also
/// tries to open the URL; if that fails the URL is on screen to copy.
fn show_login_prompt(prompt: LoginPrompt) {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let written = match prompt {
        LoginPrompt::Browser { url } => {
            let opened = open_browser(&url);
            writeln!(
                out,
                "{}\n\n  {url}\n\nWaiting for the sign-in to finish...",
                if opened {
                    "Opening your browser to sign in. If it does not appear, open this URL:"
                } else {
                    "Open this URL in a browser to sign in:"
                }
            )
        }
        LoginPrompt::DeviceCode {
            verification_uri,
            user_code,
            verification_uri_complete,
            expires_in,
        } => {
            let complete =
                verification_uri_complete.map_or(String::new(), |u| format!("\n  or open {u}"));
            writeln!(
                out,
                "Open {verification_uri} and enter the code {user_code}{complete}\n\nThe code is valid for {} minutes. Waiting for approval...",
                expires_in.as_secs().checked_div(60).unwrap_or_default()
            )
        }
    };
    if written.is_err() {
        tracing::error!("could not write the login prompt to stdout");
    }
    drop(out.flush());
}

/// Whether a browser on this machine can reach the loopback redirect: not
/// over SSH, and on Linux only with a display.
fn browser_can_open() -> bool {
    if std::env::var_os("SSH_CONNECTION").is_some() || std::env::var_os("SSH_TTY").is_some() {
        return false;
    }
    if cfg!(target_os = "linux") {
        return std::env::var_os("DISPLAY").is_some()
            || std::env::var_os("WAYLAND_DISPLAY").is_some();
    }
    true
}

/// Launch the platform's URL opener. Returns whether it started.
fn open_browser(url: &str) -> bool {
    let mut command = if cfg!(target_os = "macos") {
        std::process::Command::new("open")
    } else if cfg!(target_os = "windows") {
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "start", ""]);
        c
    } else {
        std::process::Command::new("xdg-open")
    };
    command
        .arg(url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .is_ok()
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
    let session_id = resolve_session(
        &config,
        &ws_db,
        cli.continue_latest,
        cli.resume.as_deref(),
        cli.mode.map(ChatMode::from),
    )?;
    let db: SharedDb =
        Arc::new(Writer::spawn(ws_db).context("failed to start the workspace writer")?);
    let reader_db = open_reader(&db, config.analysis.reader_pool_size).await;
    terminal::run(
        config,
        ws_name,
        workspace.id,
        db,
        reader_db,
        session_id,
        cli.allow_write,
    )
    .await?;
    Ok(ExitCode::SUCCESS)
}

/// Pick the session for this run: the latest with `--continue`, a specific
/// one with `--resume`, otherwise a new one for the configured chat model.
/// `--mode` sets the mode on a new session and overrides it on a resumed one.
fn resolve_session(
    config: &Config,
    db: &WorkspaceDb,
    continue_latest: bool,
    resume: Option<&str>,
    mode: Option<ChatMode>,
) -> Result<String> {
    let existing = if let Some(prefix) = resume {
        Some(find_session(db, prefix)?.id)
    } else if continue_latest {
        sessions::latest_session(db)?.map(|s| s.id)
    } else {
        None
    };
    if let Some(id) = existing {
        if let Some(mode) = mode {
            sessions::set_session_mode(db, &id, mode)?;
        }
        return Ok(id);
    }
    let model = config
        .chat_model_ref()
        .map_or_else(|_| String::from("unconfigured"), |m| m.to_string());
    Ok(sessions::create_session(db, &model, mode.unwrap_or_default(), None)?.id)
}

fn run_context(db: &WorkspaceDb, action: ContextAction) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    match action {
        ContextAction::Show => match context::current(db)? {
            Some(current) => {
                writeln!(
                    out,
                    "# version {} ({}){}",
                    current.version,
                    current.edited_at,
                    current
                        .edited_by
                        .as_deref()
                        .map_or(String::new(), |b| format!(" by {b}"))
                )?;
                writeln!(out, "{}", current.content)?;
            }
            None => writeln!(
                out,
                "No workspace context set. Use `quack context edit` or `quack context import FILE`."
            )?,
        },
        ContextAction::History { limit } => {
            let versions = context::history(db, limit)?;
            if versions.is_empty() {
                writeln!(out, "No versions yet.")?;
            }
            for v in versions {
                let first_line = v.content.lines().next().unwrap_or("").to_owned();
                writeln!(
                    out,
                    "v{:<4} {}  {:<12} {}",
                    v.version,
                    v.edited_at,
                    v.edited_by.as_deref().unwrap_or("-"),
                    first_line
                )?;
            }
        }
        ContextAction::Export { file } => {
            let content = context::current(db)?.map_or(String::new(), |c| c.content);
            if file == "-" {
                writeln!(out, "{content}")?;
            } else {
                std::fs::write(&file, format!("{content}\n"))
                    .with_context(|| format!("failed to write {file}"))?;
                writeln!(out, "wrote {file}")?;
            }
        }
        ContextAction::Import { file } => {
            let content = if file == "-" {
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf)?;
                buf
            } else {
                std::fs::read_to_string(&file).with_context(|| format!("failed to read {file}"))?
            };
            let stored = context::set(db, &content, None)?;
            writeln!(out, "context is now version {}", stored.version)?;
        }
        ContextAction::Edit => {
            let editor = std::env::var("VISUAL")
                .or_else(|_| std::env::var("EDITOR"))
                .context("set $EDITOR (or $VISUAL) to edit the context, or use `quack context import FILE`")?;
            let current = context::current(db)?.map_or(String::new(), |c| c.content);
            let tmp = tempfile::Builder::new()
                .prefix("quack-context-")
                .suffix(".md")
                .tempfile()
                .context("failed to create a temporary file")?;
            std::fs::write(tmp.path(), format!("{current}\n"))?;
            let status = std::process::Command::new(&editor)
                .arg(tmp.path())
                .status()
                .with_context(|| format!("failed to run {editor}"))?;
            if !status.success() {
                anyhow::bail!("{editor} exited with {status}; context unchanged");
            }
            let edited = std::fs::read_to_string(tmp.path())?;
            let stored = context::set(db, &edited, None)?;
            writeln!(out, "context is now version {}", stored.version)?;
        }
    }
    out.flush()?;
    Ok(())
}

/// `quack docs [--pin ID] [--unpin ID] [--delete ID]`: apply changes, then list.
fn run_docs(
    db: &WorkspaceDb,
    pin: Option<&str>,
    unpin: Option<&str>,
    delete: Option<&str>,
    json: bool,
) -> Result<()> {
    if let Some(prefix) = pin {
        let id = find_document(db, prefix)?;
        db.set_document_pinned(&id, true)?;
    }
    if let Some(prefix) = unpin {
        let id = find_document(db, prefix)?;
        db.set_document_pinned(&id, false)?;
    }
    if let Some(prefix) = delete {
        let id = find_document(db, prefix)?;
        let filename = db
            .document(&id)?
            .map(|d| d.filename)
            .context("document vanished")?;
        let table = ingestion::parser::detect_file_type(&filename)
            .is_structured()
            .then(|| ingestion::table_name_for(&filename));
        db.delete_document(&id, table.as_deref())?;
    }
    list_documents(db, json)
}

/// Resolve a full document id or a unique prefix.
fn find_document(db: &WorkspaceDb, prefix: &str) -> Result<String> {
    let matches: Vec<String> = db
        .list_documents()?
        .into_iter()
        .filter(|d| d.id.starts_with(prefix))
        .map(|d| d.id)
        .collect();
    match matches.len() {
        0 => anyhow::bail!("no document matches '{prefix}'; run `quack docs`"),
        1 => matches.into_iter().next().context("document vanished"),
        n => anyhow::bail!("'{prefix}' matches {n} documents; use more of the id"),
    }
}

fn list_documents(db: &WorkspaceDb, json: bool) -> Result<()> {
    let docs = db.list_documents()?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    if json {
        for doc in &docs {
            serde_json::to_writer(
                &mut out,
                &serde_json::json!({
                    "id": doc.id,
                    "filename": doc.filename,
                    "title": doc.title,
                    "mime_type": doc.mime_type,
                    "size_bytes": doc.size_bytes,
                    "sha256": doc.sha256,
                    "source": doc.source,
                    "status": doc.status,
                    "pinned": doc.pinned,
                    "chunk_count": doc.chunk_count,
                    "ingested_at": doc.ingested_at,
                }),
            )?;
            writeln!(out)?;
        }
    } else if docs.is_empty() {
        writeln!(out, "No documents yet.")?;
    } else {
        for doc in &docs {
            let title = doc
                .title
                .as_deref()
                .map_or(String::new(), |t| format!("  ({t})"));
            writeln!(
                out,
                "{}  {:<10}  {:<6}  {}  {}{title}",
                doc.id,
                doc.status,
                doc.source,
                if doc.pinned { "pinned  " } else { "        " },
                doc.filename
            )?;
        }
    }
    out.flush()?;
    Ok(())
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
                "{}  {}  {:>3} msgs  {}  {}{}",
                row.id,
                row.updated_at,
                row.message_count,
                row.model,
                row.title.as_deref().unwrap_or("(untitled)"),
                if row.shared { "  (shared)" } else { "" }
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
    init_logging_at("warn");
}

/// Log to stderr at `default` unless `RUST_LOG` says otherwise.
fn init_logging_at(default: &str) {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default)),
        )
        .init();
    // The provider is installed at the top of `main`, before any subscriber
    // exists; this is the first point where saying so reaches a log.
    crypto::log_provider();
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

async fn run_query(
    sql: &str,
    workspace_name: Option<&str>,
    format: OutputFormat,
    wait_for_stdin: bool,
) -> Result<()> {
    let (config, workspace, _) = resolve_workspace(workspace_name).await?;

    let ws_db =
        WorkspaceDb::open(&config, &workspace.id).context("failed to open workspace database")?;
    load_piped_stdin(&config, &ws_db, &workspace.id, wait_for_stdin).await?;

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
    title: Option<&str>,
    no_embed: bool,
    pin: bool,
) -> Result<()> {
    let (config, workspace, _) = resolve_workspace(workspace_name).await?;

    if file != "-" && PathBuf::from(file).is_dir() {
        return ingest_bundle(&config, &workspace.id, file, no_embed).await;
    }
    let (data, effective_filename) = read_input(file, filename_override)?;

    let ws_db = spawn_writer(&config, &workspace.id)?;

    let embedding_model = if no_embed {
        None
    } else {
        llm::optional_embedding_model(&config)
            .await
            .context("failed to build embedding model")?
    };

    let source = if file == "-" {
        DocumentSource::Stdin
    } else {
        DocumentSource::Path
    };
    let outcome = ingestion::ingest_file(
        &config,
        &ws_db,
        &workspace.id,
        &NewFile::new(&effective_filename, &data)
            .source(source)
            .title(title),
        embedding_model.as_ref(),
    )
    .await
    .context("ingestion failed")?;

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    let result = match outcome {
        IngestOutcome::Ingested(result) => result,
        IngestOutcome::Duplicate(existing) => {
            writeln!(
                out,
                "Skipped: {effective_filename} is identical to {} (id: {})",
                existing.filename, existing.id
            )?;
            out.flush()?;
            return Ok(());
        }
    };

    if pin {
        let id = result.document_id.clone();
        ws_db
            .run(move |db| db.set_document_pinned(&id, true))
            .await?;
    }

    writeln!(out, "Ingested: {}", result.filename)?;
    writeln!(out, "  Type: {}", result.file_type)?;
    writeln!(out, "  Document ID: {}", result.document_id)?;
    if pin {
        writeln!(out, "  Pinned: yes")?;
    }

    for table in &result.tables {
        writeln!(out, "  Table: {table}")?;
    }
    if result.pages_skipped > 0 {
        writeln!(
            out,
            "  Pages skipped: {} (unreadable; the rest of the document was kept)",
            result.pages_skipped
        )?;
    }
    if result.chunks_stored > 0 {
        writeln!(out, "  Chunks: {}", result.chunks_stored)?;
        if let Some(took) = result.embedding_time {
            let seconds = took.as_secs_f64();
            let per_second = if seconds > 0.0 {
                f64::from(result.chunks_stored) / seconds
            } else {
                0.0
            };
            writeln!(
                out,
                "  Embeddings: {} chunks in {seconds:.1} s ({per_second:.1}/s)",
                result.chunks_stored
            )?;
        }
    }

    out.flush()?;
    Ok(())
}

/// `quack ingest DIR`: an OKF bundle. Every concept file becomes a
/// Markdown document, its front matter and links feed the ontology review
/// queue, and `index.md` is offered as the workspace context.
async fn ingest_bundle(
    config: &Config,
    workspace_id: &str,
    dir: &str,
    no_embed: bool,
) -> Result<()> {
    let bundle = Bundle::from_dir(&PathBuf::from(dir))?;
    let ws_db = spawn_writer(config, workspace_id)?;
    let embedding_model = if no_embed {
        None
    } else {
        llm::optional_embedding_model(config)
            .await
            .context("failed to build embedding model")?
    };
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let mut stored = 0usize;
    let mut skipped = 0usize;
    for file in bundle.documents() {
        let (front, _) = okf::parse_front_matter(&file.content);
        let name = okf::document_name(&file.path);
        let outcome = ingestion::ingest_file(
            config,
            &ws_db,
            workspace_id,
            &NewFile::new(&name, file.content.as_bytes()).title(front.get("title")),
            embedding_model.as_ref(),
        )
        .await
        .with_context(|| format!("ingesting {}", file.path))?;
        match outcome {
            IngestOutcome::Ingested(_) => stored = stored.saturating_add(1),
            IngestOutcome::Duplicate(_) => skipped = skipped.saturating_add(1),
        }
    }
    writeln!(
        out,
        "Ingested {stored} concept files from {dir}{}.",
        if skipped > 0 {
            format!(" ({skipped} already present)")
        } else {
            String::new()
        }
    )?;
    let index_body = bundle
        .index()
        .map(|index| okf::parse_front_matter(&index.content).1.trim().to_owned())
        .filter(|body| !body.is_empty());
    // The ontology, the review queue, and the current context: one step on
    // the writer, its report rendered there.
    let owned_dir = dir.to_owned();
    let (report, existing) = ws_db
        .run(move |db| Ok(restore_bundle(db, &bundle, &owned_dir)))
        .await??;
    out.write_all(&report)?;
    if let Some(body) = index_body {
        if existing.as_deref() == Some(body.as_str()) {
            writeln!(out, "index.md already is the workspace context.")?;
        } else {
            write!(
                out,
                "index.md can become the workspace context{}. Apply it? [y/N] ",
                if existing.is_some() {
                    " (replacing the current one)"
                } else {
                    ""
                }
            )?;
            out.flush()?;
            let mut answer = String::new();
            std::io::stdin().read_line(&mut answer)?;
            if matches!(answer.trim(), "y" | "Y" | "yes") {
                let stored = ws_db.run(move |db| context::set(db, &body, None)).await?;
                writeln!(out, "context is now version {}", stored.version)?;
            } else {
                writeln!(
                    out,
                    "Left the context alone; `quack context import {dir}/index.md` applies it later."
                )?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

/// A bundle's ontology and review candidates, applied on the writer: the
/// report of what changed, and the workspace context now in force.
fn restore_bundle(
    ws_db: &WorkspaceDb,
    bundle: &Bundle,
    dir: &str,
) -> Result<(Vec<u8>, Option<String>)> {
    let mut out = Vec::new();
    let current = restore_bundle_ontology(ws_db, bundle, dir, &mut out)?;
    let candidates = okf::propose(bundle, current.as_ref());
    if !candidates.is_empty() {
        candidates::store_run(ws_db, &candidates)?;
        writeln!(
            out,
            "{} ontology candidates from the bundle's types and links: `quack ontology review`.",
            candidates.len()
        )?;
    }
    let existing = context::current(ws_db)?.map(|c| c.content);
    Ok((out, existing))
}

/// quack's own export carries the ontology exactly: a workspace without
/// one takes it back as it was; one that has an ontology reviews the
/// bundle's types and links as candidates instead. Returns the ontology
/// in force afterwards.
fn restore_bundle_ontology(
    ws_db: &WorkspaceDb,
    bundle: &Bundle,
    dir: &str,
    out: &mut impl Write,
) -> Result<Option<Ontology>> {
    let current = ontology_store::current(ws_db)?;
    if current.is_some() {
        return Ok(current);
    }
    let Some(snapshot) = bundle.ontology()? else {
        return Ok(None);
    };
    let restored = ontology_store::save(
        ws_db,
        &snapshot,
        None,
        Some(&format!("restored from {dir}")),
    )?;
    writeln!(
        out,
        "Restored the bundle's ontology (version {} in this workspace).",
        restored.version
    )?;
    Ok(Some(restored))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed reader surfaces as an `io::Error`, a `serde_json` error, or
    /// core's transparent `Io` and `Json` variants; each one ends the
    /// command quietly (issue #68). Anything else still reports.
    #[test]
    fn broken_pipe_is_recognised_through_every_wrapper() {
        let pipe = || std::io::Error::from(std::io::ErrorKind::BrokenPipe);
        assert!(is_broken_pipe(&anyhow::Error::from(pipe())));
        assert!(is_broken_pipe(
            &anyhow::Error::from(pipe()).context("failed to print")
        ));
        assert!(is_broken_pipe(&anyhow::Error::from(
            quack_core::error::Error::Io(pipe())
        )));
        assert!(is_broken_pipe(&anyhow::Error::from(
            quack_core::error::Error::Json(serde_json::Error::io(pipe()))
        )));
        assert!(!is_broken_pipe(&anyhow::Error::from(std::io::Error::from(
            std::io::ErrorKind::NotFound
        ))));
        assert!(!is_broken_pipe(&anyhow::anyhow!("something else")));
    }

    /// A missing login is recognised through the context a command adds
    /// (`import failed`), but not once the error has been turned into text,
    /// which is how `quack import` lost its exit 4.
    #[test]
    fn auth_required_is_recognised_through_context_only_while_typed() {
        let auth = || quack_core::error::Error::AuthRequired {
            provider: String::from("corp"),
            reason: String::from("no cached token"),
        };
        assert!(auth_exit_code(&anyhow::Error::from(auth())).is_some());
        assert!(auth_exit_code(&anyhow::Error::from(auth()).context("import failed")).is_some());
        assert!(auth_exit_code(&anyhow::anyhow!(auth().to_string())).is_none());
        assert!(auth_exit_code(&anyhow::anyhow!("something else")).is_none());
    }
}
