#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod admin;
mod auth_cli;
mod config_cli;
mod confirm;
mod doctor_cli;
mod embeddings_cli;
mod graph_cli;
mod mcp;
mod ontology_cli;
mod print;
mod progress_line;
mod server;
mod stdio;
mod terminal;
mod text_or_json;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::config::inspect::SettingFilter;
use quack_core::config::{Config, Grant};
use quack_core::crypto::{self, CryptoModule};
use quack_core::doctor::{Options, Probing};
use quack_core::error::{Error as CoreError, Record};
use quack_core::ids::{DocumentId, SessionId};
use quack_core::import::{self, ImportPolicy, ImportRequest};
use quack_core::ingestion::{self, IngestOutcome, NewFile};
use quack_core::llm::Embeddings;
use quack_core::llm::oauth::{KeySource, LoginFlow, LoginPrompt, TokenManager};
use quack_core::okf::{self, Bundle, DirSink, TarSink};
use quack_core::ontology::store::Revision;
use quack_core::prefix::PrefixMatch;
use quack_core::progress::RunControl;
use quack_core::storage::context;
use quack_core::storage::control::{ControlPlane, WorkspaceRow};
use quack_core::storage::sessions::{self, ChatMode, ExportFormat, Transcript};
use quack_core::storage::workspace::{DocumentSource, Pinning, WorkspaceDb};
use quack_core::storage::writer::Writer;
use quack_core::{config, doctor};
use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use crate::confirm::Confirm;
use crate::print::{PrintTurn, TurnOutcome};
use crate::server::state::ServeMode;
use crate::stdio::{NamedInput, StdioPath};
use crate::terminal::SessionSetup;
use crate::text_or_json::TextOrJson;

/// How a command ended when not plainly: the exit status scripts check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Exit {
    /// A usage or configuration error: bad flags, no terminal for the
    /// session, a configuration every other command refuses (`quack
    /// config` still prints its report), or `-p` with no chat model.
    Usage,
    /// The agent needed a write that was not permitted.
    WriteRefused,
    /// An OAuth provider needs `quack auth login` first.
    AuthRequired,
}

impl From<Exit> for ExitCode {
    fn from(exit: Exit) -> Self {
        Self::from(match exit {
            Exit::Usage => 2,
            Exit::WriteRefused => 3,
            Exit::AuthRequired => 4,
        })
    }
}

/// `--version` names the crypto module as well, so an operator can tell a FIPS
/// binary from a non-FIPS one without turning on `RUST_LOG=info`. `-V` stays
/// the bare version. A static because clap takes a `&'static str`.
static LONG_VERSION: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    format!("{}\n{}", env!("CARGO_PKG_VERSION"), CryptoModule::linked())
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
    Sessions(SessionsArgs),

    /// Export a session as a runnable .sql file or a Markdown transcript
    Export(ExportArgs),

    /// Ingest a file into a workspace
    Ingest(IngestArgs),

    /// Show, edit, or move the workspace context (the owner's instructions
    /// and definitions for the agent)
    Context(ContextArgs),

    /// Log in to an OAuth provider, show token state, forget a token, print
    /// or rotate a client's public key set, or register a client with the
    /// issuer
    #[command(subcommand)]
    Auth(AuthAction),

    #[command(flatten)]
    Admin(admin::AdminCommand),

    /// Serve the REST API and web UI
    Serve(ServeArgs),

    /// Serve the workspace as an MCP server over stdio (for Claude Code
    /// and editors); logs go to stderr
    Mcp(McpArgs),

    /// Show, install, import, export, diff, or restore the ontology
    #[command(subcommand)]
    Ontology(ontology_cli::OntologyAction),

    /// Explore, build, revalidate, and review the knowledge graph
    #[command(subcommand)]
    Graph(graph_cli::GraphAction),

    /// Pull rows from Postgres, SQLite, or a data file over HTTP(S) into
    /// a workspace table (the Rust-side replacement for ATTACH)
    Import(ImportArgs),

    /// Move the workspace as an Open Knowledge Format bundle
    #[command(subcommand)]
    Okf(OkfAction),

    /// Show what this binary makes of config.toml: every setting it
    /// recognizes, the value in force and where it came from, and the
    /// keys in the file it does not recognize
    Config(ConfigArgs),

    /// Check the setup and say how to fix what is wrong: the config file,
    /// the data directory, the workspace, each model's provider (reached
    /// over the network), and the server's bind address
    Doctor(DoctorArgs),

    /// List ingested documents, or pin and unpin one
    Docs(DocsArgs),

    /// The workspace's vectors: refresh the ones made with another
    /// embedding model, width, or input prefixes
    #[command(subcommand)]
    Embeddings(embeddings_cli::EmbeddingsAction),
}

#[derive(clap::Args)]
struct SessionsArgs {
    /// `json` prints one JSON object per session
    #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
    format: TextOrJson,

    /// Maximum number of sessions to show
    #[arg(long, default_value_t = 20)]
    limit: u32,
}

#[derive(clap::Args)]
struct ExportArgs {
    /// Session id (prefixes accepted)
    session_id: String,

    #[command(flatten)]
    flags: ExportFlags,
}

#[derive(clap::Args)]
struct IngestArgs {
    /// File path to ingest (use - for stdin)
    file: StdioPath,

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
}

#[derive(clap::Args)]
struct ContextArgs {
    #[command(subcommand)]
    action: Option<ContextAction>,
}

#[derive(clap::Args)]
struct ServeArgs {
    /// Listen address (default from `[server].bind`, or `QUACK_BIND`)
    #[arg(long)]
    bind: Option<String>,
    /// No authentication, one implicit user; loopback only
    #[arg(long)]
    local: bool,
}

#[derive(clap::Args)]
struct McpArgs {
    /// Let the `sql` and `query` tools run statements that modify data
    #[arg(long)]
    allow_write: bool,
}

#[derive(clap::Args)]
struct ImportArgs {
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
}

impl From<ImportArgs> for ImportRequest {
    fn from(args: ImportArgs) -> Self {
        Self {
            url: args.url.into(),
            table: args.table,
            query: args.query,
            source_table: args.from,
            limit: args.limit,
        }
    }
}

#[derive(clap::Args)]
struct ConfigArgs {
    /// Only the settings the file or the environment has a say in
    #[arg(long)]
    changed: bool,

    /// `json` prints the whole report as one JSON document
    #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
    format: TextOrJson,
}

#[derive(clap::Args)]
struct DoctorArgs {
    /// Skip the network probes
    #[arg(long)]
    offline: bool,

    /// `json` prints the checks as one JSON document
    #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
    format: TextOrJson,
}

#[derive(clap::Args)]
struct DocsArgs {
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

    /// `json` prints one JSON object per document
    #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
    format: TextOrJson,
}

#[derive(Subcommand)]
enum OkfAction {
    /// Write the workspace as a bundle: index.md from the context, one
    /// Markdown file per table, class, relation, property, document, and
    /// graph node, and log.md
    Export {
        /// Directory to write (created), or - for a tar archive on stdout
        dir: StdioPath,
    },
}

#[derive(Subcommand)]
enum AuthAction {
    /// Obtain a token: browser sign-in with PKCE, or a device code when no
    /// browser can open here; a client-credentials provider requests one
    /// with its secret
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
    /// Print the public key set (JWKS) a private-key-JWT client signs its
    /// assertions with, made on first use, to register with the issuer
    Jwks {
        /// Provider name from [providers.NAME]; without one, the
        /// [server.oidc] sign-in client
        provider: Option<String>,

        /// Replace the key: a client quack registered is updated at the
        /// issuer (RFC 7592) and switches at once; for any other client,
        /// print the current and new keys to register, then run again with
        /// --activate
        #[arg(long)]
        rotate: bool,

        /// With --rotate, for a client registered by hand: sign with the
        /// new key from now on, once the issuer holds it
        #[arg(long, requires = "rotate")]
        activate: bool,
    },
    /// Register one private-key-JWT client with the issuer (RFC 7591) for
    /// every [server.oidc] and [providers.NAME.oauth] section there that
    /// names no client id
    Register(auth_cli::RegisterArgs),
    /// Delete the client quack registered at the issuer (RFC 7592), then
    /// its registration and key
    Unregister {
        /// The issuer; by default the one the sections without a client id
        /// share
        #[arg(long)]
        issuer: Option<String>,

        /// Delete without asking
        #[arg(long)]
        yes: bool,
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
        file: StdioPath,
    },
    /// Replace the context with the contents of a Markdown file
    Import {
        /// Source path (- for stdin)
        file: StdioPath,
    },
}

/// The answer mode as a command-line or slash-command argument.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ModeArg {
    /// General knowledge allowed; cite when a source was used
    Chat,
    /// Every claim must come from a retrieved source
    Query,
}

/// `--sql` or `--markdown`: how `export` writes a session.
#[derive(clap::Args)]
pub(crate) struct ExportFlags {
    /// Every executed statement, each preceded by its question
    #[arg(long, conflicts_with = "markdown")]
    sql: bool,

    /// Questions, steps, and answers as Markdown (the default)
    #[arg(long)]
    markdown: bool,
}

impl ExportFlags {
    pub(crate) const fn format(&self) -> ExportFormat {
        if self.sql {
            ExportFormat::Sql
        } else {
            ExportFormat::Markdown
        }
    }
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

    /// The format for `-p`, which prints an answer: text or JSON.
    const fn for_prompt(self) -> Option<TextOrJson> {
        match self {
            Self::Text => Some(TextOrJson::Text),
            Self::Json => Some(TextOrJson::Json),
            Self::Table | Self::Ndjson | Self::Csv | Self::Markdown => None,
        }
    }

    /// The format for `-q`, which prints a result set.
    const fn for_query(self) -> Option<QueryFormat> {
        match self {
            Self::Table => Some(QueryFormat::Table),
            Self::Json => Some(QueryFormat::Json),
            Self::Ndjson => Some(QueryFormat::Ndjson),
            Self::Csv => Some(QueryFormat::Csv),
            Self::Markdown => Some(QueryFormat::Markdown),
            Self::Text => None,
        }
    }
}

/// How `-q` prints a result set.
#[derive(Clone, Copy, PartialEq, Eq)]
enum QueryFormat {
    Table,
    Json,
    Ndjson,
    Csv,
    Markdown,
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
    // A CSV write that failed on its output.
    fn csv_io_kind(error: &csv::Error) -> Option<std::io::ErrorKind> {
        match error.kind() {
            csv::ErrorKind::Io(e) => Some(e.kind()),
            _ => None,
        }
    }
    error.chain().any(|cause| {
        let kind = if let Some(e) = cause.downcast_ref::<std::io::Error>() {
            Some(e.kind())
        } else if let Some(e) = cause.downcast_ref::<serde_json::Error>() {
            e.io_error_kind()
        } else if let Some(e) = cause.downcast_ref::<csv::Error>() {
            csv_io_kind(e)
        } else {
            match cause.downcast_ref::<Core>() {
                Some(Core::Io(e)) => Some(e.kind()),
                Some(Core::Json(e)) => e.io_error_kind(),
                Some(Core::Csv(e)) => csv_io_kind(e),
                _ => None,
            }
        };
        kind == Some(std::io::ErrorKind::BrokenPipe)
    })
}

async fn run() -> Result<ExitCode> {
    let mut cli = Cli::parse();

    let policy = WritePolicy::Deny.allowed_if(cli.allow_write);
    let stdout_is_tty = std::io::stdout().is_terminal();

    if let Some(prompt) = cli.prompt.as_deref() {
        return run_print_mode(&cli, prompt, policy).await;
    }

    if let Some(sql) = cli.query.as_deref() {
        init_logging();
        let Some(format) = cli
            .format
            .unwrap_or_else(|| OutputFormat::default_for(stdout_is_tty))
            .for_query()
        else {
            tracing::error!("-q accepts --format table, json, ndjson, csv, or markdown");
            return Ok(ExitCode::from(Exit::Usage));
        };
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
        Commands::Sessions(args) => run_sessions(cli, &args).await,
        Commands::Export(args) => run_export(cli, &args.session_id, args.flags.format()).await,
        Commands::Ingest(args) => {
            init_logging();
            run_ingest(cli, args).await?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Ontology(action) => run_on_writer(cli, action).await,
        Commands::Graph(action) => run_on_writer(cli, action).await,
        Commands::Embeddings(action) => run_on_writer(cli, action).await,
        Commands::Okf(OkfAction::Export { dir }) => run_okf_export(cli, &dir).await,
        Commands::Import(args) => run_import(cli, args).await,
        Commands::Context(args) => {
            let ws_db = open_workspace(cli).await?;
            run_context(&ws_db, args.action.unwrap_or(ContextAction::Show))?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Auth(action) => {
            init_logging();
            let config = Config::load().context("failed to load configuration")?;
            run_auth(&config, action).await?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Mcp(args) => run_mcp(cli, args.allow_write).await,
        Commands::Serve(args) => {
            // The server logs each request at info; other commands stay quiet.
            init_logging_at("info,sqlx=warn,hyper=warn,h2=warn");
            let config = Config::load().context("failed to load configuration")?;
            let mode = if args.local {
                ServeMode::Local
            } else {
                ServeMode::Login
            };
            server::serve(config, args.bind, mode).await?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Admin(command) => {
            init_logging();
            let config = Config::load().context("failed to load configuration")?;
            command.run(&config, cli.workspace.as_deref()).await?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Config(args) => run_config(&args).await,
        Commands::Doctor(args) => run_doctor(cli, &args).await,
        Commands::Docs(args) => {
            let ws_db = open_workspace(cli).await?;
            run_docs(&ws_db, &args)?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// `quack config`: what this binary makes of `config.toml`. It reads the
/// file outside `Config::load`, so it reports a file every other command
/// refuses rather than failing the same way, and says so in its status.
async fn run_config(args: &ConfigArgs) -> Result<ExitCode> {
    init_logging();
    let mut inspection = config::inspect::Inspection::load();
    // A client_id the file leaves out comes from its registration.
    if let Err(e) = inspection.resolve_registered().await {
        tracing::warn!(error = %e, "cannot read the registered OAuth clients from control.db");
    }
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let filter = if args.changed {
        SettingFilter::Changed
    } else {
        SettingFilter::All
    };
    let usable = config_cli::run(&mut out, &inspection, args.format, filter)?;
    out.flush()?;
    Ok(if usable {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(Exit::Usage)
    })
}

/// `quack doctor`: like `quack config`, it inspects the file outside
/// `Config::load`, so a file every other command refuses is a finding here
/// rather than the error. Exits 1 when any check fails.
async fn run_doctor(cli: &Cli, args: &DoctorArgs) -> Result<ExitCode> {
    init_logging();
    let inspection = config::inspect::Inspection::load();
    let options = Options {
        workspace: cli.workspace.clone(),
        probing: if args.offline {
            Probing::Offline
        } else {
            Probing::default()
        },
    };
    let report = doctor::run(&inspection, &options).await;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    doctor_cli::write(&mut out, &report, args.format)?;
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
    let Some(format) = cli.format.unwrap_or(OutputFormat::Text).for_prompt() else {
        tracing::error!("-p accepts only --format text or json");
        return Ok(ExitCode::from(Exit::Usage));
    };
    let opened = OpenedWorkspace::resolve(cli.workspace.as_deref()).await?;
    let config = &opened.config;
    // Checked before the workspace opens, so a missing model is one line
    // on stderr rather than a failed turn, and leaves no session behind.
    if let Err(e) = config.chat_model_ref() {
        tracing::error!("{e}");
        return Ok(ExitCode::from(Exit::Usage));
    }
    let ws_db = opened.open_db()?;
    load_piped_stdin(config, &ws_db, opened.workspace.id.as_str(), cli.stdin).await?;
    if let Some(note) = ws_db.embedding_status()?.note() {
        tracing::warn!(
            "{note} Run `quack embeddings refresh -w {}` to update them.",
            opened.name
        );
    }
    let session_id = resolve_session(
        config,
        &ws_db,
        cli.session_choice(),
        cli.mode.map(ChatMode::from),
    )?;
    let (db, reader_db) = opened.shared(ws_db).await?;
    let outcome = PrintTurn {
        config,
        db: Arc::clone(&db),
        reader_db,
        session_id: &session_id,
        policy,
        prompt,
        format,
        verbose: cli.verbose,
    }
    .run()
    .await;
    if outcome.is_err() {
        let id = session_id.clone();
        drop(db.run(move |db| sessions::delete_if_empty(db, &id)).await);
    }
    Ok(match outcome? {
        TurnOutcome::WriteRefused => ExitCode::from(Exit::WriteRefused),
        TurnOutcome::Answered => ExitCode::SUCCESS,
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
    OpenedWorkspace::resolve(cli.workspace.as_deref())
        .await?
        .open_db()
}

/// `quack sessions`: the session list.
async fn run_sessions(cli: &Cli, args: &SessionsArgs) -> Result<ExitCode> {
    let ws_db = open_workspace(cli).await?;
    list_sessions(&ws_db, args.format, args.limit)?;
    Ok(ExitCode::SUCCESS)
}

/// `quack export SESSION`: a session as SQL or Markdown.
async fn run_export(cli: &Cli, session_id: &str, format: ExportFormat) -> Result<ExitCode> {
    let ws_db = open_workspace(cli).await?;
    export_session(&ws_db, session_id, format)?;
    Ok(ExitCode::SUCCESS)
}

/// A subcommand whose steps run on the workspace writer and whose report
/// goes to stdout: the graph, ontology, and embeddings commands.
trait WriterCommand {
    async fn run(
        self,
        config: &Config,
        db: &Writer,
        out: &mut impl Write,
        control: RunControl<'_>,
    ) -> Result<()>;
}

impl WriterCommand for graph_cli::GraphAction {
    async fn run(
        self,
        config: &Config,
        db: &Writer,
        out: &mut impl Write,
        control: RunControl<'_>,
    ) -> Result<()> {
        graph_cli::run(config, db, self, Confirm::Ask, out, control).await
    }
}

impl WriterCommand for ontology_cli::OntologyAction {
    async fn run(
        self,
        config: &Config,
        db: &Writer,
        out: &mut impl Write,
        control: RunControl<'_>,
    ) -> Result<()> {
        ontology_cli::run(config, db, self, Confirm::Ask, out, control).await
    }
}

impl WriterCommand for embeddings_cli::EmbeddingsAction {
    async fn run(
        self,
        config: &Config,
        db: &Writer,
        out: &mut impl Write,
        control: RunControl<'_>,
    ) -> Result<()> {
        embeddings_cli::run(config, db, self, Confirm::Ask, out, control).await
    }
}

/// Run a writer command in the command line's workspace, its progress on
/// stderr.
async fn run_on_writer(cli: &Cli, command: impl WriterCommand) -> Result<ExitCode> {
    init_logging();
    let opened = OpenedWorkspace::resolve(cli.workspace.as_deref()).await?;
    let db = opened.writer()?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    command
        .run(
            &opened.config,
            &db,
            &mut out,
            RunControl {
                progress: &progress_line::to_stderr,
                cancel: None,
            },
        )
        .await?;
    Ok(ExitCode::SUCCESS)
}

/// `quack import URL --table NAME`: rows from an external source as a
/// workspace table.
async fn run_import(cli: &Cli, args: ImportArgs) -> Result<ExitCode> {
    init_logging();
    let request = &ImportRequest::from(args);
    let opened = OpenedWorkspace::resolve(cli.workspace.as_deref()).await?;
    let (config, ws_db) = (&opened.config, opened.writer()?);
    let embedding_model = Embeddings::from_config(config).await?;
    let summary = import::Importing {
        config,
        db: &ws_db,
        workspace_id: opened.workspace.id.as_str(),
        request,
        policy: ImportPolicy::owner(),
        embedder: embedding_model.as_ref(),
        control: RunControl::unobserved(),
    }
    .run()
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
async fn run_okf_export(cli: &Cli, dir: &StdioPath) -> Result<ExitCode> {
    init_logging();
    let opened = OpenedWorkspace::resolve(cli.workspace.as_deref()).await?;
    let db = opened.open_db()?;
    match dir {
        StdioPath::Stdio => {
            let mut sink = TarSink::new(std::io::BufWriter::new(std::io::stdout().lock()));
            okf::export(&db, &opened.name, &mut sink)?;
            sink.finish()?.flush()?;
        }
        StdioPath::Path(path) => {
            let summary = okf::export(&db, &opened.name, &mut DirSink::new(path))?;
            let stdout = std::io::stdout();
            let mut out = stdout.lock();
            writeln!(out, "wrote {} files to {dir}", summary.files)?;
        }
    }
    Ok(ExitCode::SUCCESS)
}

/// `quack mcp`: the workspace as an MCP server on stdin and stdout.
async fn run_mcp(cli: &Cli, allow_write: bool) -> Result<ExitCode> {
    init_logging();
    let opened = OpenedWorkspace::resolve(cli.workspace.as_deref()).await?;
    let (db, reader_db) = opened.shared(opened.open_db()?).await?;
    let OpenedWorkspace {
        config, workspace, ..
    } = opened;
    let policy = WritePolicy::Deny.allowed_if(allow_write);
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
                Some(CoreError::AuthRequired { .. } | CoreError::Delegation { .. })
            )
        })
        .then_some(ExitCode::from(Exit::AuthRequired))
}

/// `quack auth login|status|logout`.
async fn run_auth(config: &Config, action: AuthAction) -> Result<()> {
    let stdout = std::io::stdout();
    match action {
        AuthAction::Login {
            provider,
            device_code,
        } => {
            let manager = TokenManager::for_provider(config, &provider)?;
            let flow = if device_code || !browser_can_open() {
                LoginFlow::DeviceCode
            } else {
                LoginFlow::Configured
            };
            let token = manager.login(flow, &show_login_prompt).await?;
            let mut out = stdout.lock();
            match manager.grant() {
                // `login` refuses this grant, so this only reads well.
                Grant::OnBehalfOf => writeln!(
                    out,
                    "'{provider}' acts on behalf of each person signed in to quack serve."
                )?,
                Grant::ClientCredentials => writeln!(
                    out,
                    "The client credentials for '{provider}' were accepted; the token expires at {} and a new one is requested when it runs out.",
                    token.expires_at
                )?,
                Grant::AuthorizationCode | Grant::DeviceCode => writeln!(
                    out,
                    "Logged in to '{provider}'; the token expires at {}{}.",
                    token.expires_at,
                    if token.refresh_token.is_some() {
                        " and will refresh itself"
                    } else {
                        ""
                    }
                )?,
            }
            out.flush()?;
        }
        AuthAction::Status { provider } => {
            let mut names: Vec<&str> = config
                .providers
                .iter()
                .filter(|(_, p)| p.auth.oauth().is_some())
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
                let manager = TokenManager::for_provider(config, name)?;
                let status = manager.status().await?;
                let state = token_state(name, status.token, manager.grant(), manager.sends_actor());
                let key = auth_cli::client_key_state(config, Some(name), KeySource::Keychain)
                    .await?
                    .map_or(String::new(), |key| format!("; {key}"));
                writeln!(out, "{name}: {state} (key in {}){key}", status.key_location)?;
            }
            if provider.is_none()
                && let Some(key) =
                    auth_cli::client_key_state(config, None, KeySource::Keychain).await?
            {
                writeln!(out, "[server.oidc] sign-in: {key}")?;
            }
            out.flush()?;
        }
        AuthAction::Logout { provider } => {
            TokenManager::for_provider(config, &provider)?
                .logout()
                .await?;
            let mut out = stdout.lock();
            writeln!(out, "Logged out of '{provider}'.")?;
            out.flush()?;
        }
        AuthAction::Jwks {
            provider,
            rotate,
            activate,
        } => auth_cli::run_jwks(config, provider.as_deref(), rotate, activate).await?,
        AuthAction::Register(args) => {
            let confirm = Confirm::Ask.or_yes(args.yes);
            auth_cli::run_register(config, args, confirm).await?;
        }
        AuthAction::Unregister { issuer, yes } => {
            auth_cli::run_unregister(config, issuer.as_deref(), Confirm::Ask.or_yes(yes)).await?;
        }
    }
    Ok(())
}

/// What `quack auth status` says of one provider's token.
fn token_state(
    name: &str,
    token: Option<quack_core::llm::oauth::TokenStatus>,
    grant: Grant,
    sends_actor: bool,
) -> String {
    match (token, grant) {
        // Without an actor token quack has no token of its own here; one
        // stored before `actor = false` is unused.
        (_, Grant::OnBehalfOf) if !sends_actor => String::from(
            "acts on behalf of each person signed in to quack serve, without an actor token; nothing to log in to",
        ),
        (Some(token), Grant::ClientCredentials) => {
            format!("token expires {}, {}", token.expires_at, token.renewal)
        }
        (Some(token), Grant::AuthorizationCode | Grant::DeviceCode) => format!(
            "logged in, token expires {}, {}",
            token.expires_at, token.renewal
        ),
        (None, Grant::ClientCredentials) => {
            String::from("no token yet; one is requested on first use")
        }
        (None, Grant::AuthorizationCode | Grant::DeviceCode) => {
            format!("not logged in; run `quack auth login {name}`")
        }
        (Some(token), Grant::OnBehalfOf) => format!(
            "acts on behalf of each signed-in person; quack's own token (the actor) expires {}",
            token.expires_at
        ),
        (None, Grant::OnBehalfOf) => String::from(
            "acts on behalf of each person signed in to quack serve; nothing to log in to",
        ),
    }
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
        return Ok(ExitCode::from(Exit::Usage));
    }
    let opened = OpenedWorkspace::resolve(cli.workspace.as_deref()).await?;
    let ws_db = opened.open_db()?;
    let session_id = resolve_session(
        &opened.config,
        &ws_db,
        cli.session_choice(),
        cli.mode.map(ChatMode::from),
    )?;
    let (db, reader_db) = opened.shared(ws_db).await?;
    let OpenedWorkspace {
        config,
        workspace,
        name,
    } = opened;
    terminal::run(SessionSetup {
        config,
        workspace_name: name,
        workspace_id: workspace.id,
        db,
        reader_db,
        session_id,
        writes: WritePolicy::Ask.allowed_if(cli.allow_write),
    })
    .await?;
    Ok(ExitCode::SUCCESS)
}

/// Which session a run continues.
#[derive(Debug, Clone, Copy)]
enum SessionChoice<'a> {
    New,
    /// `--continue`: the most recent one, or a new one when there is none.
    Latest,
    /// `--resume ID`: this one (prefixes accepted).
    Resume(&'a str),
}

impl Cli {
    fn session_choice(&self) -> SessionChoice<'_> {
        match (self.resume.as_deref(), self.continue_latest) {
            (Some(prefix), _) => SessionChoice::Resume(prefix),
            (None, true) => SessionChoice::Latest,
            (None, false) => SessionChoice::New,
        }
    }
}

/// Pick the session for this run: the latest with `--continue`, a specific
/// one with `--resume`, otherwise a new one for the configured chat model.
/// `--mode` sets the mode on a new session and overrides it on a resumed one.
fn resolve_session(
    config: &Config,
    db: &WorkspaceDb,
    choice: SessionChoice<'_>,
    mode: Option<ChatMode>,
) -> Result<SessionId> {
    let existing = match choice {
        SessionChoice::Resume(prefix) => Some(find_session(db, prefix)?.id),
        SessionChoice::Latest => sessions::latest_session(db)?.map(|s| s.id),
        SessionChoice::New => None,
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
            match &file {
                StdioPath::Stdio => writeln!(out, "{content}")?,
                StdioPath::Path(path) => {
                    std::fs::write(path, format!("{content}\n"))
                        .with_context(|| format!("failed to write {file}"))?;
                    writeln!(out, "wrote {file}")?;
                }
            }
        }
        ContextAction::Import { file } => {
            let content = file.read_to_string()?;
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
fn run_docs(db: &WorkspaceDb, args: &DocsArgs) -> Result<()> {
    if let Some(prefix) = args.pin.as_deref() {
        let id = find_document(db, prefix)?;
        db.set_document_pinning(&id, Pinning::Pinned)?;
    }
    if let Some(prefix) = args.unpin.as_deref() {
        let id = find_document(db, prefix)?;
        db.set_document_pinning(&id, Pinning::Unpinned)?;
    }
    if let Some(prefix) = args.delete.as_deref() {
        let id = find_document(db, prefix)?;
        db.delete_document(&id)?;
    }
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    list_documents(db, args.format, &mut out)
}

/// Resolve a full document id or a unique prefix.
fn find_document(db: &WorkspaceDb, prefix: &str) -> Result<DocumentId> {
    let document = PrefixMatch::of(db.list_documents()?, prefix, |d| d.id.as_str())
        .one(Record::Document, prefix)?;
    Ok(document.id)
}

fn list_documents(db: &WorkspaceDb, format: TextOrJson, out: &mut impl Write) -> Result<()> {
    let docs = db.list_documents()?;
    format.write_rows(out, &docs, "No documents yet.", |out, doc| {
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
        )
    })?;
    out.flush()?;
    Ok(())
}

/// Resolve a full id or a unique prefix to a session.
fn find_session(db: &WorkspaceDb, prefix: &str) -> Result<sessions::SessionRow> {
    if let Some(exact) = sessions::get_session(db, &SessionId::from(prefix))? {
        return Ok(exact);
    }
    Ok(
        PrefixMatch::of(sessions::list_sessions(db, 1000)?, prefix, |s| {
            s.id.as_str()
        })
        .one(Record::Session, prefix)?,
    )
}

fn list_sessions(db: &WorkspaceDb, format: TextOrJson, limit: u32) -> Result<()> {
    let rows = sessions::list_sessions(db, limit)?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    format.write_rows(&mut out, &rows, "No sessions yet.", |out, row| {
        writeln!(
            out,
            "{}  {}  {:>3} msgs  {}  {}{}",
            row.id,
            row.updated_at,
            row.message_count,
            row.model,
            row.title.as_deref().unwrap_or("(untitled)"),
            if row.shared { "  (shared)" } else { "" }
        )
    })?;
    out.flush()?;
    Ok(())
}

fn export_session(db: &WorkspaceDb, prefix: &str, format: ExportFormat) -> Result<()> {
    let text = Transcript::load(db, find_session(db, prefix)?)?.render(format)?;
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
    CryptoModule::linked().log();
}

/// The configuration and the workspace a command runs in: the named one,
/// or the default, created on first use.
struct OpenedWorkspace {
    config: Config,
    workspace: WorkspaceRow,
    name: String,
}

impl OpenedWorkspace {
    async fn resolve(workspace_name: Option<&str>) -> Result<Self> {
        let config = Config::load().context("failed to load configuration")?;
        let control = ControlPlane::open(&config)
            .await
            .context("failed to open control plane")?;
        let name =
            workspace_name.map_or_else(|| config.general.default_workspace.clone(), str::to_owned);
        let workspace = control
            .find_or_create_workspace(&name)
            .await
            .context("failed to resolve workspace")?;
        Ok(Self {
            config,
            workspace,
            name,
        })
    }

    /// The workspace's connection.
    fn open_db(&self) -> Result<WorkspaceDb> {
        WorkspaceDb::open(&self.config, self.workspace.id.as_str())
            .context("failed to open workspace database")
    }

    /// The workspace's connection on a writer thread of its own: the
    /// commands that run core's multi-step work (ingestion, import, graph
    /// and ontology commands) send it their steps exactly as the server
    /// and the terminal do.
    fn writer(&self) -> Result<Writer> {
        Writer::spawn(self.open_db()?).context("failed to start the workspace writer")
    }

    /// `db` moved onto its writer thread, and a reader pool over it, for
    /// the commands that run agent turns.
    async fn shared(&self, db: WorkspaceDb) -> Result<(SharedDb, ReaderDb)> {
        let db: SharedDb =
            Arc::new(Writer::spawn(db).context("failed to start the workspace writer")?);
        let reader_db = ReaderDb::open(&db, self.config.analysis.reader_pool_size).await;
        Ok((db, reader_db))
    }
}

async fn run_query(
    sql: &str,
    workspace_name: Option<&str>,
    format: QueryFormat,
    wait_for_stdin: bool,
) -> Result<()> {
    let opened = OpenedWorkspace::resolve(workspace_name).await?;
    let ws_db = opened.open_db()?;
    load_piped_stdin(
        &opened.config,
        &ws_db,
        opened.workspace.id.as_str(),
        wait_for_stdin,
    )
    .await?;

    let results = ws_db.execute_query(sql).context("query execution failed")?;

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    match format {
        QueryFormat::Table => results.write_table(&mut out)?,
        QueryFormat::Json => results.write_json(&mut out)?,
        QueryFormat::Ndjson => results.write_ndjson(&mut out)?,
        QueryFormat::Csv => results.write_csv(&mut out)?,
        QueryFormat::Markdown => results.write_markdown(&mut out)?,
    }

    out.flush()?;
    Ok(())
}

async fn run_ingest(cli: &Cli, args: IngestArgs) -> Result<()> {
    let IngestArgs {
        file,
        filename,
        title,
        no_embed,
        pin,
    } = args;
    let opened = OpenedWorkspace::resolve(cli.workspace.as_deref()).await?;
    let config = &opened.config;

    if let StdioPath::Path(dir) = &file
        && dir.is_dir()
    {
        return ingest_bundle(&opened, &dir.display().to_string(), no_embed).await;
    }
    let NamedInput {
        name: effective_filename,
        data,
    } = file.read_named(filename.as_deref())?;

    let ws_db = opened.writer()?;

    let embedding_model = if no_embed {
        None
    } else {
        Embeddings::from_config(config)
            .await
            .context("failed to build embedding model")?
    };

    let source = match file {
        StdioPath::Stdio => DocumentSource::Stdin,
        StdioPath::Path(_) => DocumentSource::Path,
    };
    let outcome = ingestion::ingest_file(
        config,
        &ws_db,
        opened.workspace.id.as_str(),
        &NewFile::new(&effective_filename, &data)
            .source(source)
            .title(title.as_deref()),
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
            .run(move |db| db.set_document_pinning(&id, Pinning::Pinned))
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
async fn ingest_bundle(opened: &OpenedWorkspace, dir: &str, no_embed: bool) -> Result<()> {
    let (config, workspace_id) = (&opened.config, opened.workspace.id.as_str());
    let bundle = Bundle::from_dir(&PathBuf::from(dir))?;
    let ws_db = opened.writer()?;
    let embedding_model = if no_embed {
        None
    } else {
        Embeddings::from_config(config)
            .await
            .context("failed to build embedding model")?
    };
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let mut stored = 0usize;
    let mut skipped = 0usize;
    for file in bundle.documents() {
        let front = okf::parse_front_matter(&file.content).front;
        let name = file.document_name();
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
        .map(|index| {
            okf::parse_front_matter(&index.content)
                .body
                .trim()
                .to_owned()
        })
        .filter(|body| !body.is_empty());
    // The ontology, the review queue, and the current context: one step on
    // the writer.
    let note = format!("restored from {dir}");
    let (report, existing) = ws_db
        .run(move |db| {
            let report = bundle.restore_into(db, Revision::reviewed(None, Some(&note)))?;
            Ok((report, context::current(db)?.map(|c| c.content)))
        })
        .await?;
    if let Some(version) = report.restored {
        writeln!(
            out,
            "Restored the bundle's ontology (version {version} in this workspace)."
        )?;
    }
    if report.candidates > 0 {
        writeln!(
            out,
            "{} ontology candidates from the bundle's types and links: `quack ontology review`.",
            report.candidates
        )?;
    }
    if let Some(body) = index_body {
        if existing.as_deref() == Some(body.as_str()) {
            writeln!(out, "index.md already is the workspace context.")?;
        } else {
            let question = format!(
                "index.md can become the workspace context{}. Apply it?",
                if existing.is_some() {
                    " (replacing the current one)"
                } else {
                    ""
                }
            );
            if Confirm::Ask.ask(&mut out, &question, None)? {
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

#[cfg(test)]
mod tests {
    use super::*;
    use quack_core::embedding::Dimension;
    use quack_core::error::AuthReason;
    use quack_core::storage::workspace::NewDocument;

    /// A closed reader surfaces as an `io::Error`, a `serde_json` or `csv`
    /// error, or core's transparent `Io`, `Json`, and `Csv` variants; each one ends the
    /// command quietly (issue #68). Anything else still reports.
    #[test]
    fn broken_pipe_is_recognised_through_every_wrapper() {
        let pipe = || std::io::Error::from(std::io::ErrorKind::BrokenPipe);
        assert!(is_broken_pipe(&anyhow::Error::from(pipe())));
        assert!(is_broken_pipe(
            &anyhow::Error::from(pipe()).context("failed to print")
        ));
        assert!(is_broken_pipe(&anyhow::Error::from(CoreError::Io(pipe()))));
        assert!(is_broken_pipe(&anyhow::Error::from(CoreError::Json(
            serde_json::Error::io(pipe())
        ))));
        assert!(is_broken_pipe(&anyhow::Error::from(csv::Error::from(
            pipe()
        ))));
        assert!(is_broken_pipe(&anyhow::Error::from(CoreError::Csv(
            csv::Error::from(pipe())
        ))));
        assert!(!is_broken_pipe(&anyhow::Error::from(std::io::Error::from(
            std::io::ErrorKind::NotFound
        ))));
        assert!(!is_broken_pipe(&anyhow::anyhow!("something else")));
    }

    /// `docs --format json` prints every recorded field, so a script can tell why a
    /// document failed.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test")]
    fn docs_json_carries_every_document_field() {
        let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
        db.insert_document(&NewDocument::new(
            &DocumentId::from("d1"),
            "broken.pdf",
            "application/pdf",
            3,
        ))
        .unwrap();
        db.mark_document_error(&DocumentId::from("d1"), "no text layer")
            .unwrap();
        let mut out = Vec::new();
        list_documents(&db, TextOrJson::Json, &mut out).unwrap();
        let row: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(row.get("status").unwrap(), "error", "{row}");
        assert_eq!(row.get("error_message").unwrap(), "no text layer", "{row}");
        for key in ["ingested_by", "tables", "filename", "sha256", "source"] {
            assert!(row.get(key).is_some(), "{key} missing: {row}");
        }
    }

    /// A missing login is recognised through the context a command adds
    /// (`import failed`), but not once the error has been turned into text,
    /// which is how `quack import` lost its exit 4.
    #[test]
    fn auth_required_is_recognised_through_context_only_while_typed() {
        let auth = || CoreError::AuthRequired {
            provider: String::from("corp"),
            reason: AuthReason::NoToken,
        };
        assert!(auth_exit_code(&anyhow::Error::from(auth())).is_some());
        assert!(auth_exit_code(&anyhow::Error::from(auth()).context("import failed")).is_some());
        assert!(auth_exit_code(&anyhow::anyhow!(auth().to_string())).is_none());
        assert!(auth_exit_code(&anyhow::anyhow!("something else")).is_none());
    }

    #[test]
    fn auth_status_names_the_actor_only_when_one_is_sent() {
        let token = quack_core::llm::oauth::TokenStatus {
            expires_at: jiff::Timestamp::UNIX_EPOCH,
            renewal: quack_core::llm::oauth::Renewal::Regrant,
        };
        let with_actor = token_state("gw", Some(token), Grant::OnBehalfOf, true);
        assert!(with_actor.contains("(the actor)"), "{with_actor}");
        for stored in [Some(token), None] {
            let vouch = token_state("gw", stored, Grant::OnBehalfOf, false);
            assert!(!vouch.contains("the actor"), "{vouch}");
            assert!(vouch.contains("without an actor token"), "{vouch}");
        }
        assert!(
            token_state("p", None, Grant::AuthorizationCode, false).contains("quack auth login p")
        );
    }
}
