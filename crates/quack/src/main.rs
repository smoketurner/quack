#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod admin;
mod ontology_cli;
mod print;
mod server;
mod terminal;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::SharedDb;
use quack_core::config::Config;
use quack_core::ingestion;
use quack_core::llm;
use quack_core::llm::oauth::{LoginOptions, LoginPrompt};
use quack_core::storage::context;
use quack_core::storage::control::{ControlPlane, WorkspaceRow};
use quack_core::storage::sessions::{self, ChatMode};
use quack_core::storage::workspace::WorkspaceDb;
use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

/// Exit status for a usage error (bad flags, no terminal for the session).
const EXIT_USAGE: u8 = 2;
/// Exit status when the agent needed a write that was not permitted.
const EXIT_WRITE_REFUSED: u8 = 3;
/// Exit status when an OAuth provider needs `quack auth login` first.
const EXIT_AUTH_REQUIRED: u8 = 4;

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

    /// Answer mode: chat may use general knowledge; query answers only from
    /// retrieved chunks and query results
    #[arg(long, value_enum, global = true)]
    mode: Option<ModeArg>,

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

    /// Show, install, import, export, diff, or restore the ontology
    Ontology {
        #[command(subcommand)]
        action: ontology_cli::OntologyAction,
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
    quack_core::crypto::install_default_provider()
        .context("failed to install the aws-lc-rs crypto provider")?;

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
        run_query(sql, cli.workspace.as_deref(), format).await?;
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
        Commands::Sessions { json, limit } => {
            let ws_db = open_workspace(cli).await?;
            list_sessions(&ws_db, json, limit)?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Export {
            session_id,
            sql,
            markdown: _,
        } => {
            let ws_db = open_workspace(cli).await?;
            export_session(&ws_db, &session_id, sql)?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Ingest {
            file,
            filename,
            no_embed,
            pin,
        } => {
            init_logging();
            let outcome = run_ingest(
                &file,
                cli.workspace.as_deref(),
                filename.as_deref(),
                no_embed,
                pin,
            )
            .await;
            if let Err(e) = &outcome
                && let Some(code) = auth_exit_code(e)
            {
                tracing::error!("{e:#}");
                return Ok(code);
            }
            outcome?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Ontology { action } => {
            let ws_db = open_workspace(cli).await?;
            ontology_cli::run(&ws_db, action)?;
            Ok(ExitCode::SUCCESS)
        }
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
    let session_id = resolve_session(
        &config,
        &ws_db,
        cli.continue_latest,
        cli.resume.as_deref(),
        cli.mode.map(ChatMode::from),
    )?;
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
    if let Err(e) = &outcome
        && let Some(code) = auth_exit_code(e)
    {
        tracing::error!("{e:#}");
        return Ok(code);
    }
    let refused = outcome?;
    Ok(if refused {
        ExitCode::from(EXIT_WRITE_REFUSED)
    } else {
        ExitCode::SUCCESS
    })
}

/// Logging, then the workspace database for the workspace-local subcommands.
async fn open_workspace(cli: &Cli) -> Result<WorkspaceDb> {
    init_logging();
    let (config, workspace, _) = resolve_workspace(cli.workspace.as_deref()).await?;
    WorkspaceDb::open(&config, &workspace.id).context("failed to open workspace database")
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
        | Commands::Auth { .. }
        | Commands::Serve { .. }
        | Commands::Docs { .. } => Ok(()),
    }
}

/// Exit 4 when the failure is an OAuth provider without a usable token:
/// print and ingest cannot run a login flow, so the message names the
/// command that can.
fn auth_exit_code(err: &anyhow::Error) -> Option<ExitCode> {
    err.chain()
        .any(|cause| {
            matches!(
                cause.downcast_ref::<quack_core::error::Error>(),
                Some(quack_core::error::Error::AuthRequired { .. })
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
                .filter(|(_, p)| p.auth == quack_core::config::AuthMode::Oauth)
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

fn oauth_manager(config: &Config, name: &str) -> Result<Arc<quack_core::llm::oauth::TokenManager>> {
    let provider = config.providers.get(name).ok_or_else(|| {
        anyhow::anyhow!(
            "provider '{name}' is not configured; add [providers.{name}] with auth = \"oauth\""
        )
    })?;
    if provider.auth != quack_core::config::AuthMode::Oauth {
        anyhow::bail!("provider '{name}' does not use auth = \"oauth\"");
    }
    quack_core::llm::oauth::shared_manager(&config.tokens_dir(), name, provider)
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
                    "mime_type": doc.mime_type,
                    "size_bytes": doc.size_bytes,
                    "status": doc.status,
                    "pinned": doc.pinned,
                }),
            )?;
            writeln!(out)?;
        }
    } else if docs.is_empty() {
        writeln!(out, "No documents yet.")?;
    } else {
        for doc in &docs {
            writeln!(
                out,
                "{}  {:<10}  {}  {}",
                doc.id,
                doc.status,
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
    pin: bool,
) -> Result<()> {
    let (config, workspace, _) = resolve_workspace(workspace_name).await?;

    let (data, effective_filename) = read_input(file, filename_override)?;

    let ws_db =
        WorkspaceDb::open(&config, &workspace.id).context("failed to open workspace database")?;

    let embedding_model = if no_embed {
        None
    } else {
        llm::optional_embedding_model(&config)
            .await
            .context("failed to build embedding model")?
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

    if pin {
        ws_db.set_document_pinned(&result.document_id, true)?;
    }

    writeln!(out, "Ingested: {}", result.filename)?;
    writeln!(out, "  Type: {}", result.file_type)?;
    writeln!(out, "  Document ID: {}", result.document_id)?;
    if pin {
        writeln!(out, "  Pinned: yes")?;
    }

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
