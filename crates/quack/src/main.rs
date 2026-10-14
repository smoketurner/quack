#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod admin;
mod auth_cli;
mod config_cli;
mod doctor_cli;
mod init_cli;
mod progress_line;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::search::{DocumentSearch, SearchDetail};
use quack_core::analysis::tools::Rerank;
use quack_core::config::inspect::SettingFilter;
use quack_core::config::{Config, Grant, LogFormat};
use quack_core::crypto::{self, CryptoModule};
use quack_core::doctor::{Options, Probing};
use quack_core::error::{Error as CoreError, Result as CoreResult};
use quack_core::graph::export::Destination;
use quack_core::graph::follow_up::FollowUp;
use quack_core::ids::DocumentId;
use quack_core::ingestion::parser::PageCounts;
use quack_core::ingestion::tree::{FileResult, Folder, FolderReport, Outcome, Prune};
use quack_core::ingestion::{self, IngestOutcome, IngestResult, NewFile, Piped};
use quack_core::jobs::JobKind;
use quack_core::llm::Embeddings;
use quack_core::llm::egress::Egress;
use quack_core::llm::oauth::{
    KeySource, Lifetime, LoginFlow, LoginPrompt, Renewal, TokenManager, TokenStatus,
};
use quack_core::okf::{self, Bundle, DirSink, TarSink};
use quack_core::ontology::store::Revision;
use quack_core::progress::RunControl;
use quack_core::proxy::Proxies;
use quack_core::storage::context;
use quack_core::storage::profile::{ColumnTypes, TableProfile};
use quack_core::storage::sessions::{
    self, ChatMode, ExportFormat, SessionViewer, Sharing, Transcript,
};
use quack_core::storage::workspace::{
    DocumentFields, DocumentInfo, DocumentListing, DocumentSource, Pinning, STRUCTURED_CELL_CHARS,
    SearchMode, Shown, StatementKind, WorkspaceDb,
};
use quack_core::storage::writer::Writer;
use quack_core::vault::Vault;
use quack_core::{config, doctor};
use std::io::{IsTerminal, Read, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use quack_cli::Confirm;
use quack_cli::TextOrJson;
use quack_cli::classify_cli::ClassifyCommand;
use quack_cli::{AnswerTo, FOLLOW_UP_GRACE, PrintTurn, TurnOutcome};
use quack_cli::{ExportFlags, ModeArg, QueryFormat};
use quack_cli::{ImportAction, ImportArgs, ImportContext};
use quack_cli::{NamedInput, StdioPath};
use quack_cli::{OpenedWorkspace, RecordedRun, SessionChoice, find_session, resolve_session};
use quack_cli::{embeddings_cli, graph_cli, ontology_cli, saved_cli, tables_cli};
use quack_core::engine::{Drain, Holder, Stop};
use quack_mcp::client::{CommandClient, HolderClient};

use quack_core::config::ServerConfig;
use quack_server::{ServeMode, Server};
use quack_terminal::{QUIT_GRACE, Quitting, SessionSetup, WebStatus};

use crate::progress_line::StderrProgress;

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
    /// `quack saved run --exit-code`: the result changed since the run
    /// before.
    Changed,
}

impl Exit {
    /// The exit a failure asks for, found anywhere in the error's chain;
    /// `None` is a plain runtime error. `main` applies this to every
    /// command's error, so no command maps it itself. An OAuth provider
    /// without a usable token is exit 4: no command but `quack auth login`
    /// can run a login flow. A workspace that does not exist is a usage
    /// error.
    fn of(err: &anyhow::Error) -> Option<Self> {
        err.chain()
            .find_map(|cause| match cause.downcast_ref::<CoreError>()? {
                CoreError::AuthRequired { .. } | CoreError::Delegation { .. } => {
                    Some(Self::AuthRequired)
                }
                CoreError::NoWorkspaceNamed(_) => Some(Self::Usage),
                _ => None,
            })
    }
}

impl From<Exit> for ExitCode {
    fn from(exit: Exit) -> Self {
        Self::from(match exit {
            Exit::Usage => 2,
            Exit::WriteRefused => 3,
            Exit::AuthRequired => 4,
            Exit::Changed => 5,
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

    /// Answer mode: chat explains terms from general knowledge but takes every
    /// fact from the workspace; query takes everything from retrieved chunks
    /// and query results
    #[arg(long, value_enum, global = true)]
    mode: Option<ModeArg>,

    /// Print full tool inputs and outputs to stderr in print mode
    #[arg(long, global = true)]
    verbose: bool,

    /// Limit the question to these documents (ids, id prefixes, or file
    /// names; repeat the flag or separate with commas)
    #[arg(
        long,
        value_name = "DOCUMENT",
        value_delimiter = ',',
        requires = "prompt"
    )]
    documents: Vec<String>,

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

    /// Ask a running `quack serve` whether it can serve (its /readyz), for
    /// a container health check: exit 0 when ready, 1 otherwise
    Ready(ReadyArgs),

    /// Serve the workspace as an MCP server over stdio (for Claude Code
    /// and editors), or relay to the quack process that holds it; logs go
    /// to stderr
    Mcp(McpArgs),

    /// List the workspace's jobs: what runs, what ran and how it ended;
    /// on a workspace another quack holds, as that process lists them
    Jobs(JobsArgs),

    /// Show, install, import, export, diff, or restore the ontology
    #[command(subcommand)]
    Ontology(ontology_cli::OntologyAction),

    /// Explore, build, revalidate, and review the knowledge graph
    #[command(subcommand)]
    Graph(graph_cli::GraphAction),

    /// Pull rows from a SQLite file, a data file over HTTP(S), or S3 into a
    /// workspace table (the Rust-side replacement for ATTACH); `list`,
    /// `refresh`, and `remove` manage imports saved with `--save`
    Import(ImportCommand),

    /// Label a table's text with a decision model, one question set for
    /// every row, into a new table that joins back by the key; `list` shows
    /// the runs
    Classify(ClassifyCommand),

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

    /// Set up or edit config.toml: find the model providers this machine
    /// can reach, ask which models to use, and write the file once `quack
    /// doctor` passes it
    Init,

    /// List ingested documents, or pin and unpin one
    Docs(DocsArgs),

    /// Search the documents without the model, showing each hit's rank in
    /// the vector and keyword legs and after reranking
    Search(SearchArgs),
    /// List the tables with their row counts, notes, and warnings; show
    /// one in full, set its note, or give a column a type
    Tables(tables_cli::TablesArgs),

    /// The workspace's vectors: refresh the ones made with another
    /// embedding model, width, or input prefixes
    #[command(subcommand)]
    Embeddings(embeddings_cli::EmbeddingsAction),

    /// Saved questions: an answer's SQL kept under a name and re-run
    /// without the model, each run saying whether the data changed
    #[command(subcommand)]
    Saved(saved_cli::SavedAction),

    /// The vault key that seals every stored token in control.db
    #[command(subcommand)]
    Vault(VaultAction),
}

#[derive(Subcommand)]
enum VaultAction {
    /// Print the vault key, or write it to a file only its owner can read;
    /// a copy of control.db restored on another host needs it as vault.key
    ExportKey {
        /// Write the key here (mode 0600) instead of printing it (`-`
        /// prints it)
        #[arg(long, value_name = "FILE")]
        to: Option<PathBuf>,
        /// Print without asking
        #[arg(short = 'y', long)]
        yes: bool,
    },
}

impl VaultAction {
    /// `quack vault export-key`: the key as stored, to a 0600 file or, after
    /// a yes, to stdout.
    async fn run(self) -> Result<ExitCode> {
        let Self::ExportKey { to, yes } = self;
        init_logging();
        let config = Config::load().context("failed to load configuration")?;
        let vault = Vault::new(config.data_dir(), KeySource::Keychain);
        let Some(key) = vault.key_text().await? else {
            anyhow::bail!("no vault key exists yet; one is made when the first token is stored");
        };
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        if let Some(path) = to.filter(|path| path.as_os_str() != "-") {
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            options
                .open(&path)
                .and_then(|mut file| {
                    file.write_all(key.as_bytes())?;
                    file.flush()
                })
                .with_context(|| format!("failed to write {}", path.display()))?;
            writeln!(out, "Wrote the vault key to {}", path.display())?;
        } else {
            // The question and a refusal go to stderr: stdout carries the
            // key alone, so `quack vault export-key > vault.key` is the key.
            let question = "The vault key unseals every token in control.db. Print it?";
            let mut err = std::io::stderr().lock();
            if !Confirm::Ask.ask_to_drop(yes, &mut err, question)? {
                writeln!(err, "Not printed.")?;
                return Ok(ExitCode::FAILURE);
            }
            writeln!(out, "{key}")?;
        }
        out.flush()?;
        Ok(ExitCode::SUCCESS)
    }
}

#[derive(clap::Args)]
struct SessionsArgs {
    /// `json` prints one JSON object per session
    #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
    format: TextOrJson,

    /// Maximum number of sessions to show
    #[arg(long, default_value_t = 20)]
    limit: u32,

    /// Show the questions and answers containing this text instead,
    /// newest first, each with its session and message number
    #[arg(long, value_name = "TEXT")]
    search: Option<String>,
}

#[derive(clap::Args)]
struct ExportArgs {
    /// Session id (prefixes accepted)
    session_id: String,

    #[command(flatten)]
    flags: ExportFlags,
}

#[derive(clap::Args)]
struct ReadyArgs {
    /// The server's base URL; default: `http://` and the `[server].bind`
    /// address, or loopback when it binds every address
    #[arg(long)]
    url: Option<String>,
}

impl ReadyArgs {
    /// The URL a server bound to `bind` answers on: an unspecified address
    /// is reached on loopback of the same family, and an IPv6 host is
    /// written in brackets.
    fn url_for(bind: SocketAddr) -> String {
        let host = match bind.ip() {
            std::net::IpAddr::V4(ip) if ip.is_unspecified() => {
                std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
            }
            std::net::IpAddr::V6(ip) if ip.is_unspecified() => {
                std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)
            }
            ip => ip,
        };
        format!("http://{}", SocketAddr::new(host, bind.port()))
    }

    /// `quack ready`: `GET /readyz` on the server and exit by its answer. The
    /// container image's health check runs this, since the image has no shell.
    async fn run(self) -> Result<ExitCode> {
        let url = if let Some(url) = self.url {
            url
        } else {
            let config = Config::load().context("failed to load configuration")?;
            let bind: SocketAddr = config.server.bind_or_default().parse().with_context(|| {
                format!(
                    "[server].bind '{}' is not a socket address",
                    config.server.bind_or_default()
                )
            })?;
            Self::url_for(bind)
        };
        let client = Proxies::from_env()
            .client()
            .timeout(Duration::from_secs(5))
            .build()
            .context("could not build the HTTP client")?;
        let readyz = format!("{}/readyz", url.trim_end_matches('/'));
        let response = client.get(&readyz).send().await;
        // The body is read before stdout is locked: no lock across an await.
        let answer = match response {
            Ok(response) if response.status().is_success() => Ok(None),
            Ok(response) => {
                let status = response.status();
                Ok(Some((status, response.text().await.unwrap_or_default())))
            }
            Err(e) => Err(e),
        };
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        match answer {
            Ok(None) => {
                writeln!(out, "ready")?;
                Ok(ExitCode::SUCCESS)
            }
            Ok(Some((status, body))) => {
                writeln!(out, "not ready ({status}): {body}")?;
                Ok(ExitCode::FAILURE)
            }
            Err(e) => {
                writeln!(out, "not ready: {readyz} did not answer: {e}")?;
                Ok(ExitCode::FAILURE)
            }
        }
    }
}

#[derive(clap::Args)]
struct IngestArgs {
    /// File path to ingest (use - for stdin)
    file: StdioPath,

    /// Override the filename (from stdin without one, the type comes from the bytes)
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

    /// The author to record, over what the file says
    #[arg(long)]
    author: Option<String>,

    /// The authored date to record (YYYY-MM-DD or ISO 8601), over what
    /// the file says
    #[arg(long, value_name = "DATE")]
    authored: Option<String>,

    /// A tag to record (repeatable), over the file's own
    #[arg(long = "tag", value_name = "TAG")]
    tags: Vec<String>,

    /// Replace a ready document: the one with the same file name, or the
    /// given id (prefixes accepted). It is superseded once this file is
    /// ready and untouched if ingestion fails; a table file takes over its
    /// table. Identical bytes are still skipped.
    #[arg(long, value_name = "DOCUMENT_ID", num_args = 0..=1, default_missing_value = "")]
    replace: Option<Replace>,

    /// With a folder: delete the documents whose file is no longer in it
    /// (without this they are only reported)
    #[arg(long)]
    prune: bool,

    /// Give columns of the loaded table a type, as COLUMN=TYPE (VARCHAR,
    /// BIGINT, DOUBLE, DATE, TIMESTAMP, BOOLEAN), comma-separated or
    /// repeated; every value must convert
    #[arg(long, value_name = "COLUMN=TYPE")]
    types: Vec<ColumnTypes>,
}

/// What `quack ingest --replace [DOCUMENT_ID]` replaces.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Replace {
    /// The newest ready document with the ingested file's name.
    SameName,
    /// The document with this id or unique id prefix.
    Document(String),
}

impl std::str::FromStr for Replace {
    type Err = std::convert::Infallible;

    /// `--replace` alone arrives as the empty default value.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match value.trim() {
            "" => Self::SameName,
            id => Self::Document(id.to_owned()),
        })
    }
}

impl Replace {
    /// The id of the ready document to replace, for a file named `filename`.
    async fn resolve(self, db: &Writer, filename: &str) -> Result<DocumentId> {
        match self {
            Self::Document(prefix) => Ok(db.run(move |db| find_document(db, &prefix)).await?),
            Self::SameName => {
                let name = filename.to_owned();
                let found = db.run(move |db| db.newest_document_named(&name)).await?;
                let Some(found) = found else {
                    anyhow::bail!(
                        "no ready document named {filename} to replace; pass its id to --replace"
                    );
                };
                Ok(found.id)
            }
        }
    }
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
struct JobsArgs {
    /// `json` prints one JSON object per job
    #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
    format: TextOrJson,
}

#[derive(clap::Args)]
#[command(args_conflicts_with_subcommands = true, subcommand_negates_reqs = true)]
struct ImportCommand {
    #[command(subcommand)]
    action: Option<ImportAction>,
    #[command(flatten)]
    run: ImportArgs,
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

    /// Add a tag to a document: the id (prefixes accepted) and the tag
    #[arg(long, num_args = 2, value_names = ["DOCUMENT_ID", "TAG"])]
    tag: Vec<String>,

    /// Remove a tag from a document: the id (prefixes accepted) and the tag
    #[arg(long, num_args = 2, value_names = ["DOCUMENT_ID", "TAG"])]
    untag: Vec<String>,

    /// Set a document's author: the id (prefixes accepted) and the name
    /// (empty to clear)
    #[arg(long, num_args = 2, value_names = ["DOCUMENT_ID", "AUTHOR"])]
    author: Vec<String>,

    /// Set a document's authored date: the id (prefixes accepted) and the
    /// date, as YYYY-MM-DD or an ISO 8601 timestamp (empty to clear)
    #[arg(long, num_args = 2, value_names = ["DOCUMENT_ID", "DATE"])]
    authored: Vec<String>,

    /// `json` prints one JSON object per document
    #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
    format: TextOrJson,

    /// List replaced documents too, each with the id that took its place
    #[arg(long)]
    all: bool,
}

#[derive(clap::Args)]
struct SearchArgs {
    /// What to search for; a "quoted phrase" must appear exactly
    query: String,

    /// Search only these documents (ids, id prefixes, file names, or titles)
    #[arg(long = "in", value_name = "DOCUMENT", num_args = 1..)]
    documents: Vec<String>,

    /// The keyword (BM25) leg alone
    #[arg(long, conflicts_with = "vector")]
    keyword: bool,

    /// The vector leg alone
    #[arg(long)]
    vector: bool,

    /// Also show each leg's candidates, the quoted-phrase filter, and the
    /// rerank outcome
    #[arg(long)]
    explain: bool,

    /// Hits to show (default `[retrieval].top_k`, at most 100)
    #[arg(long, short = 'k')]
    top_k: Option<u32>,

    /// `json` prints the hits (and with --explain, the workings) as one
    /// JSON document
    #[arg(short = 'f', long, value_enum, default_value_t = TextOrJson::Text)]
    format: TextOrJson,
}

impl SearchArgs {
    const fn mode(&self) -> SearchMode {
        match (self.keyword, self.vector) {
            (true, _) => SearchMode::Keyword,
            (false, true) => SearchMode::Vector,
            (false, false) => SearchMode::Hybrid,
        }
    }

    const fn detail(&self) -> SearchDetail {
        SearchDetail::explained(self.explain)
    }

    /// `quack search`: one search, no model call unless reranking asks the
    /// chat model; unaudited, like every command-line read.
    async fn run(&self, cli: &Cli) -> Result<ExitCode> {
        init_logging();
        let opened =
            OpenedWorkspace::resolve(cli.workspace.as_deref(), Holder::command("search")).await?;
        let config = opened.config();
        let search = DocumentSearch {
            documents: self.documents.clone(),
            mode: self.mode(),
            ..DocumentSearch::new(&self.query, self.top_k.unwrap_or(config.retrieval.top_k))?
        };
        let embedder = Embeddings::from_config(config).await?;
        let rerank = Rerank::from_config(config).await?;
        let handle = opened.handle().await?;
        let outcome = search
            .run(
                &handle.reader,
                embedder.as_ref(),
                rerank.as_ref(),
                config.retrieval.rrf_k,
            )
            .await;
        drop(handle);
        opened.finish(Duration::ZERO).await;
        let outcome = outcome?;
        let stdout = std::io::stdout();
        let mut out = std::io::BufWriter::new(stdout.lock());
        match self.format {
            TextOrJson::Text => write!(out, "{}", outcome.render(self.detail()))?,
            TextOrJson::Json => writeln!(
                out,
                "{}",
                serde_json::to_string_pretty(&outcome.body(self.detail()))?
            )?,
        }
        out.flush()?;
        Ok(ExitCode::SUCCESS)
    }
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
    /// assertions with, made on first use, to register with the issuer; or
    /// replace that key in two steps
    Jwks {
        /// Provider name from [providers.NAME]; without one, the
        /// [server.oidc] sign-in client
        provider: Option<String>,

        /// Replace the key, in two runs: --rotate makes a new key and puts
        /// it beside the key in use at the issuer (itself, for a client
        /// quack registered; printed to register by hand otherwise)
        #[arg(long)]
        rotate: bool,

        /// With --rotate, once the issuer holds both keys: sign with the new
        /// key and leave the issuer holding it alone (then restart `quack
        /// serve`)
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

#[tokio::main]
async fn main() -> Result<ExitCode> {
    crypto::install_default_provider()
        .context("failed to install the aws-lc-rs crypto provider")?;

    // One egress slot for the command, which opening its workspace fills
    // with that workspace's provider allow-list.
    match Box::pin(Egress::request(run())).await {
        // The reader closed the pipe (`quack ... | head -1`): the command
        // did its job, so stop quietly like `git` and `ls` do (issue #68).
        Err(e) if is_broken_pipe(&e) => Ok(ExitCode::SUCCESS),
        // A missing login exits 4 and a workspace that does not exist
        // exits 2, so scripts can tell either from a failure.
        Err(e) => match Exit::of(&e) {
            Some(exit) => {
                tracing::error!("{e:#}");
                Ok(ExitCode::from(exit))
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
        let Some(format) = cli.format.map_or_else(
            || Some(QueryFormat::default_for(stdout_is_tty)),
            OutputFormat::for_query,
        ) else {
            tracing::error!("-q accepts --format table, json, ndjson, csv, or markdown");
            return Ok(ExitCode::from(Exit::Usage));
        };
        run_query(sql, cli.workspace.as_deref(), format, cli.stdin).await?;
        return Ok(ExitCode::SUCCESS);
    }

    match cli.command.take() {
        None => run_terminal_session(&cli, stdout_is_tty).await,
        Some(command) => Box::pin(run_command(&cli, command)).await,
    }
}

/// Every subcommand; print mode, `-q`, and the terminal session are
/// dispatched by `main` itself.
async fn run_command(cli: &Cli, command: Commands) -> Result<ExitCode> {
    match command {
        Commands::Sessions(args) => run_sessions(cli, args).await,
        Commands::Export(args) => {
            run_export(cli, args.session_id.clone(), args.flags.format()).await
        }
        Commands::Ingest(args) => {
            init_logging();
            run_ingest(cli, args).await?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Ontology(ontology_cli::OntologyAction::Schema) => {
            ontology_cli::write_schema(&mut std::io::stdout().lock())?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Ontology(action) => run_on_writer(cli, "ontology", action).await,
        Commands::Graph(graph_cli::GraphAction::Export(args)) if args.to_stdout() => {
            run_graph_export(cli, &args).await
        }
        Commands::Graph(action) => run_on_writer(cli, "graph", action).await,
        Commands::Embeddings(action) => run_on_writer(cli, "embeddings", action).await,
        Commands::Classify(command) => run_on_writer(cli, "classify", command).await,
        Commands::Saved(action) => run_saved(cli, action).await,
        Commands::Vault(action) => action.run().await,
        Commands::Okf(OkfAction::Export { dir }) => run_okf_export(cli, &dir).await,
        Commands::Import(command) => command.run(cli).await,
        Commands::Context(args) => {
            let action = args.action.unwrap_or(ContextAction::Show);
            run_one_shot(cli, "context", move |db| run_context(db, action)).await
        }
        Commands::Auth(action) => {
            init_logging();
            let config = Config::load().context("failed to load configuration")?;
            run_auth(&config, action).await?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Mcp(args) => Box::pin(run_mcp(cli, args.allow_write)).await,
        Commands::Jobs(args) => run_jobs(cli, args).await,
        Commands::Serve(args) => {
            // The server logs at info in the file's format; other commands
            // stay quiet. Each request's access line is `quack::access` at
            // debug, so `RUST_LOG=quack::access=debug` turns it on alone.
            let config = Config::load().context("failed to load configuration")?;
            init_logging_as(
                "info,sqlx=warn,hyper=warn,h2=warn",
                config.server.log_format,
            );
            let mode = if args.local {
                ServeMode::Local
            } else {
                ServeMode::Login
            };
            quack_server::serve(config, args.bind, mode).await?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Ready(args) => args.run().await,
        Commands::Admin(command) => {
            init_logging();
            let config = Config::load().context("failed to load configuration")?;
            command.run(&config, cli.workspace.as_deref()).await?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Config(args) => run_config(&args).await,
        Commands::Doctor(args) => run_doctor(cli, &args).await,
        Commands::Init => {
            init_logging();
            init_cli::run_init().await
        }
        Commands::Docs(args) => run_one_shot(cli, "docs", move |db| run_docs(db, &args)).await,
        Commands::Search(args) => args.run(cli).await,
        Commands::Tables(args) => {
            run_one_shot(cli, "tables", move |db| {
                let stdout = std::io::stdout();
                let mut out = std::io::BufWriter::new(stdout.lock());
                args.run(db, &mut out)?;
                out.flush()?;
                Ok(())
            })
            .await
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
    // An empty question is a usage error, decided before the workspace
    // opens: no session is created, no model is called, and no title job is
    // queued to fail on a session with nothing to title.
    if prompt.trim().is_empty() {
        tracing::error!("-p needs a question: the prompt is empty");
        return Ok(ExitCode::from(Exit::Usage));
    }
    let opened = OpenedWorkspace::resolve(cli.workspace.as_deref(), Holder::command("-p")).await?;
    let config = opened.config();
    // Checked before the workspace opens, so a missing model is one line
    // on stderr rather than a failed turn, and leaves no session behind.
    if let Err(e) = config.chat_model_ref() {
        tracing::error!("{e}");
        return Ok(ExitCode::from(Exit::Usage));
    }
    // Read before the file opens, so a slow pipe never holds the workspace.
    let piped = read_piped_stdin(cli.stdin).await?;
    let handle = opened.handle().await?;
    let db = Arc::clone(&handle.writer);
    let piped = match piped {
        Some(Stdin::Document(document)) => Some(document),
        Some(Stdin::Table(data)) => {
            let ws = opened.workspace.id.as_str().to_owned();
            let load_config = config.clone();
            db.run(move |db| load_stdin_table(&load_config, db, &ws, &data))
                .await
                .context("failed to load stdin as a table")?;
            // A temp table lives on the writer's connection alone; the
            // reader pool, built at the open, routes reads there from now on.
            handle.reader.observe_write().await;
            None
        }
        None => None,
    };
    let note = db
        .run(|db| db.embedding_status().map(|status| status.note()))
        .await?;
    if let Some(note) = note {
        tracing::warn!(
            "{note} Run `quack embeddings refresh -w {}` to update them.",
            opened.workspace.name
        );
    }
    let session_id = {
        let config = config.clone();
        let choice = cli.session_choice();
        let mode = cli.mode.map(ChatMode::from);
        db.run(move |db| Ok(resolve_session(&config, db, choice, mode)))
            .await??
    };
    let mut documents = cli.documents.clone();
    if let Some(piped) = piped {
        documents.push(
            piped
                .ingest(config, opened.workspace.id.as_str(), &db)
                .await?
                .to_string(),
        );
    }
    let outcome = PrintTurn {
        config,
        db: Arc::clone(&db),
        reader_db: handle.reader.clone(),
        jobs: opened.engine.jobs.clone(),
        workspace_id: opened.workspace.id.clone(),
        session_id: &session_id,
        policy,
        prompt,
        documents: &documents,
        format,
        verbose: cli.verbose,
        answer_to: AnswerTo::Stdout,
    }
    .run()
    .await;
    if outcome.is_err() {
        let id = session_id.clone();
        drop(db.run(move |db| sessions::delete_if_empty(db, &id)).await);
    }
    // The answer is out; a labelling the turn started records its end, and
    // the title it started finishes, before exit.
    drop(handle);
    drop(db);
    opened.finish(FOLLOW_UP_GRACE).await;
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
/// What was piped in: a document to ingest, or table bytes to load as the
/// temporary `stdin` table; nothing from a terminal or an empty pipe.
async fn read_piped_stdin(wait: bool) -> Result<Option<Stdin>> {
    if std::io::stdin().is_terminal() {
        return Ok(None);
    }
    if !wait && !stdin_has_data().await? {
        tracing::warn!(
            "stdin is not a terminal but had nothing to read within {} s; \
             skipping the `stdin` table (pass --stdin to wait for it)",
            STDIN_GRACE.as_secs()
        );
        return Ok(None);
    }
    let mut data = Vec::new();
    std::io::stdin()
        .read_to_end(&mut data)
        .context("failed to read stdin")?;
    Ok(Some(match Piped::of(&data) {
        Piped::Document { name } => Stdin::Document(PipedDocument { name, data }),
        Piped::Table => Stdin::Table(data),
    }))
}

/// What came in on a pipe, with its bytes.
enum Stdin {
    Document(PipedDocument),
    Table(Vec<u8>),
}

/// Load piped table bytes as the temporary `stdin` table on `db`.
fn load_stdin_table(
    config: &Config,
    db: &WorkspaceDb,
    workspace_id: &str,
    data: &[u8],
) -> CoreResult<()> {
    if let Some(table) = ingestion::load_stdin_table(config, db, workspace_id, data)? {
        tracing::info!(
            table,
            bytes = data.len(),
            "stdin loaded as a temporary table"
        );
    }
    Ok(())
}

/// A document piped into `quack -p` or `-q`, as its bytes show it.
struct PipedDocument {
    /// `stdin.pdf`, and so on.
    name: String,
    data: Vec<u8>,
}

impl PipedDocument {
    /// Ingest the document into the workspace (or find the one already
    /// holding its bytes), for the question to be limited to.
    async fn ingest(self, config: &Config, workspace_id: &str, db: &Writer) -> Result<DocumentId> {
        let embedder = Embeddings::from_config(config)
            .await
            .context("failed to build embedding model")?;
        let outcome = ingestion::ingest_file(
            config,
            db,
            workspace_id,
            &NewFile::new(&self.name, &self.data).source(DocumentSource::Stdin),
            embedder.as_ref(),
        )
        .await
        .with_context(|| format!("failed to ingest the piped {}", self.name))?;
        let id = match outcome {
            IngestOutcome::Ingested(result) => result.document_id,
            IngestOutcome::Duplicate(existing) => existing.id,
        };
        tracing::info!(
            document = %id,
            name = %self.name,
            "piped document ingested; the question is limited to it"
        );
        Ok(id)
    }
}

/// How long a non-terminal stdin has to deliver a byte or close.
const STDIN_GRACE: Duration = Duration::from_secs(1);

/// Whether stdin is worth reading: a pipe or socket is when it becomes
/// readable (data or end of file) within [`STDIN_GRACE`]; anything else
/// (a regular file, `/dev/null`) answers a read at once.
#[cfg(unix)]
async fn stdin_has_data() -> Result<bool> {
    use rustix::event::{PollFd, PollFlags, Timespec};
    use std::os::fd::AsFd;
    use std::os::unix::fs::FileTypeExt;
    let stdin = std::fs::File::from(std::io::stdin().as_fd().try_clone_to_owned()?);
    let kind = stdin
        .metadata()
        .context("failed to inspect stdin")?
        .file_type();
    if !kind.is_fifo() && !kind.is_socket() {
        return Ok(true);
    }
    let grace = Timespec::try_from(STDIN_GRACE).context("stdin grace out of range")?;
    // tokio's `AsyncFd` has only unsafe constructors; a blocking poll on a
    // duplicate of stdin needs none.
    let ready = tokio::task::spawn_blocking(move || {
        rustix::event::poll(&mut [PollFd::new(&stdin, PollFlags::IN)], Some(&grace))
    })
    .await
    .context("stdin poll stopped")?
    .context("failed to watch stdin")?;
    Ok(ready > 0)
}

/// Windows has no readiness poll for stdin: read it as before.
#[cfg(not(unix))]
async fn stdin_has_data() -> Result<bool> {
    Ok(true)
}

/// Logging, the workspace, one step on its writer for `verb`'s work, and
/// the close: the workspace-local subcommands that read or write once and
/// exit.
async fn run_one_shot<F>(cli: &Cli, verb: &'static str, work: F) -> Result<ExitCode>
where
    F: FnOnce(&WorkspaceDb) -> Result<()> + Send + 'static,
{
    init_logging();
    let opened = OpenedWorkspace::resolve(cli.workspace.as_deref(), Holder::command(verb)).await?;
    let worked = opened.with_db(work).await;
    opened.finish(Duration::ZERO).await;
    worked?;
    Ok(ExitCode::SUCCESS)
}

/// `quack sessions`: the session list.
async fn run_sessions(cli: &Cli, args: SessionsArgs) -> Result<ExitCode> {
    run_one_shot(cli, "sessions", move |db| match &args.search {
        Some(text) => search_sessions(db, text, args.format, args.limit),
        None => list_sessions(db, args.format, args.limit),
    })
    .await
}

/// `quack export SESSION`: a session as SQL or Markdown.
async fn run_export(cli: &Cli, session_id: String, format: ExportFormat) -> Result<ExitCode> {
    run_one_shot(cli, "export", move |db| {
        export_session(db, &session_id, format)
    })
    .await
}

/// A subcommand whose steps run on the workspace writer and whose report
/// goes to stdout: the graph, ontology, and embeddings commands.
trait WriterCommand {
    /// What the run is recorded as in the workspace's jobs.
    fn kind(&self) -> JobKind;

    async fn run(
        self,
        config: &Config,
        db: &Writer,
        out: &mut impl Write,
        control: RunControl<'_>,
    ) -> Result<()>;
}

impl WriterCommand for graph_cli::GraphAction {
    fn kind(&self) -> JobKind {
        JobKind::Graph
    }

    async fn run(
        self,
        config: &Config,
        db: &Writer,
        out: &mut impl Write,
        control: RunControl<'_>,
    ) -> Result<()> {
        self.run(config, db, Confirm::Ask, out, control).await
    }
}

impl WriterCommand for ontology_cli::OntologyAction {
    fn kind(&self) -> JobKind {
        JobKind::Ontology
    }

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
    fn kind(&self) -> JobKind {
        JobKind::Embeddings
    }

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

impl WriterCommand for ClassifyCommand {
    fn kind(&self) -> JobKind {
        JobKind::Classify
    }

    async fn run(
        self,
        config: &Config,
        db: &Writer,
        out: &mut impl Write,
        control: RunControl<'_>,
    ) -> Result<()> {
        self.run(config, db, Confirm::Ask, out, control).await
    }
}

/// Run a writer command in the command line's workspace, its progress on
/// stderr, recorded as a job of the workspace (`RecordedRun`), so `quack
/// jobs` from another shell shows it running and the file keeps how it
/// ended.
async fn run_on_writer(
    cli: &Cli,
    verb: &'static str,
    command: impl WriterCommand,
) -> Result<ExitCode> {
    init_logging();
    let opened = OpenedWorkspace::resolve(cli.workspace.as_deref(), Holder::command(verb)).await?;
    let handle = opened.handle().await?;
    let run = RecordedRun::start(
        &opened.engine.jobs,
        command.kind(),
        RecordedRun::command_line(),
        opened.workspace.id.clone(),
    );
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let progress = StderrProgress::new();
    let ran = command
        .run(
            opened.config(),
            &handle.writer,
            &mut out,
            RunControl {
                progress: &|done| {
                    progress.report(done);
                    run.report(done);
                },
                cancel: None,
            },
        )
        .await;
    drop(out);
    run.finish(outcome_of(&ran, || String::from("done"))).await;
    drop(handle);
    opened.finish(Duration::ZERO).await;
    ran?;
    Ok(ExitCode::SUCCESS)
}

/// A command's outcome as its job records it: `done` on success, the
/// error's chain on failure.
fn outcome_of<T, E: std::fmt::Display>(
    ran: &std::result::Result<T, E>,
    done: impl FnOnce() -> String,
) -> std::result::Result<String, String> {
    match ran {
        Ok(_) => Ok(done()),
        Err(e) => Err(format!("{e:#}")),
    }
}

/// `quack saved ...`: the saved questions. A run that failed is the
/// command's error, after the run was printed; one that changed exits 5
/// with `--exit-code`.
async fn run_saved(cli: &Cli, action: saved_cli::SavedAction) -> Result<ExitCode> {
    init_logging();
    let opened =
        OpenedWorkspace::resolve(cli.workspace.as_deref(), Holder::command("saved")).await?;
    let handle = opened.handle().await?;
    let db = Arc::clone(&handle.writer);
    let wants_exit_code = matches!(
        action,
        saved_cli::SavedAction::Run {
            exit_code: true,
            ..
        }
    );
    let model = saved_cli::Model {
        db: Arc::clone(&db),
        reader_db: handle.reader.clone(),
        jobs: opened.engine.jobs.clone(),
        workspace_id: opened.workspace.id.clone(),
        verbose: cli.verbose,
    };
    // Not the lock: a refresh is a print-mode turn, which writes its own
    // answer to stdout before the run is printed.
    let mut out = std::io::BufWriter::new(std::io::stdout());
    let ran = saved_cli::run(opened.config(), &db, action, None, Some(model), &mut out).await;
    out.flush()?;
    drop(handle);
    drop(db);
    opened.finish(FOLLOW_UP_GRACE).await;
    let ran = ran?;
    let Some(run) = ran else {
        return Ok(ExitCode::SUCCESS);
    };
    if let Some(failure) = saved_cli::failure(&run) {
        return Err(failure);
    }
    Ok(if wants_exit_code && run.changed {
        ExitCode::from(Exit::Changed)
    } else {
        ExitCode::SUCCESS
    })
}

/// `quack import URL --table NAME`: rows from an external source as a
/// workspace table.
impl ImportCommand {
    /// Import (and with `--save`, keep the import for refreshing), or
    /// list, refresh, or remove a saved one.
    async fn run(self, cli: &Cli) -> Result<ExitCode> {
        init_logging();
        let opened =
            OpenedWorkspace::resolve(cli.workspace.as_deref(), Holder::command("import")).await?;
        let vault = Vault::new(opened.config().data_dir(), KeySource::Keychain);
        let handle = opened.handle().await?;
        let context = ImportContext {
            config: opened.config(),
            workspace: &opened.workspace.id,
            control: &opened.control,
            vault: &vault,
            db: &handle.writer,
        };
        // Recorded as an `import` job of the workspace, like the server's.
        let run = RecordedRun::start(
            &opened.engine.jobs,
            JobKind::Import,
            RecordedRun::command_line(),
            opened.workspace.id.clone(),
        );
        let ran = match self.action {
            // Written to a buffer, then to stdout: the stdout lock is never
            // held across the action's awaits, and what it reported before
            // a failure still prints.
            Some(action) => {
                let mut report = Vec::new();
                let ran = action.run(&context, &mut report).await;
                std::io::stdout().lock().write_all(&report)?;
                ran
            }
            None => self.run.run(&context).await,
        };
        run.finish(outcome_of(&ran, || String::from("done"))).await;
        drop(handle);
        opened.finish(Duration::ZERO).await;
        ran?;
        Ok(ExitCode::SUCCESS)
    }
}

/// `quack okf export DIR`: the workspace as an Open Knowledge Format
/// bundle, a directory of Markdown files or a tar on stdout.
async fn run_okf_export(cli: &Cli, dir: &StdioPath) -> Result<ExitCode> {
    init_logging();
    let opened =
        OpenedWorkspace::resolve(cli.workspace.as_deref(), Holder::command("okf export")).await?;
    let (name, dir) = (opened.workspace.name.clone(), dir.clone());
    let exported = opened
        .with_db(move |db| {
            match &dir {
                StdioPath::Stdio => {
                    let mut sink = TarSink::new(std::io::BufWriter::new(std::io::stdout().lock()));
                    okf::export(db, &name, &mut sink)?;
                    sink.finish()?.flush()?;
                }
                StdioPath::Path(path) => {
                    let summary = okf::export(db, &name, &mut DirSink::new(path))?;
                    let stdout = std::io::stdout();
                    let mut out = stdout.lock();
                    writeln!(out, "wrote {} files to {dir}", summary.files)?;
                }
            }
            Ok(())
        })
        .await;
    opened.finish(Duration::ZERO).await;
    exported?;
    Ok(ExitCode::SUCCESS)
}

/// `quack graph export - --format F`: the graph on stdout, a tar of the
/// CSV bundle or the document itself, in one read. A directory goes
/// through the writer like the other graph verbs.
async fn run_graph_export(cli: &Cli, args: &graph_cli::ExportArgs) -> Result<ExitCode> {
    init_logging();
    let opened =
        OpenedWorkspace::resolve(cli.workspace.as_deref(), Holder::command("graph export")).await?;
    let export = args.export();
    let written = opened
        .with_db(move |db| {
            let out = std::io::BufWriter::new(std::io::stdout().lock());
            Ok(db.read_only(|db| export.write(db, Destination::Stream(out)))?)
        })
        .await;
    opened.finish(Duration::ZERO).await;
    written?;
    Ok(ExitCode::SUCCESS)
}

/// `quack mcp`: the workspace as an MCP server on stdin and stdout. It
/// holds nothing between calls: each call runs in the process that holds
/// the workspace (the terminal session, a command) over its socket, or
/// opens the workspace for that call and closes it after, so Claude
/// Code's entry works whether the terminal session was started before or
/// after it.
async fn run_mcp(cli: &Cli, allow_write: bool) -> Result<ExitCode> {
    init_logging();
    let opened = OpenedWorkspace::resolve(cli.workspace.as_deref(), |_| Holder::Mcp).await?;
    let policy = WritePolicy::Deny.allowed_if(allow_write);
    quack_mcp::serve_stdio(Arc::clone(&opened.engine), opened.workspace.clone(), policy).await?;
    // A run the client left behind finishes and records its end first.
    opened.finish(quack_mcp::STDIO_GRACE).await;
    Ok(ExitCode::SUCCESS)
}

/// `quack jobs`: the workspace's jobs as this process restores them from
/// the file, or as the process holding the workspace lists them.
async fn run_jobs(cli: &Cli, args: JobsArgs) -> Result<ExitCode> {
    let opened =
        OpenedWorkspace::resolve(cli.workspace.as_deref(), Holder::command("jobs")).await?;
    let mut jobs = match opened.handle().await {
        Ok(handle) => {
            let jobs = opened.engine.jobs.list_workspace(&opened.workspace.id);
            drop(handle);
            opened.finish(Duration::ZERO).await;
            jobs
        }
        Err(e) if is_workspace_locked(&e) => {
            let workspace_dir = opened.config().workspace_dir(opened.workspace.id.as_str());
            let Some(client) = HolderClient::connect_with(
                &workspace_dir,
                &opened.workspace.id,
                CommandClient::new("quack jobs", report_to_stderr),
            )
            .await?
            else {
                return Err(e);
            };
            let jobs = client.jobs().await?;
            {
                let stderr = std::io::stderr();
                let mut err = stderr.lock();
                writeln!(err, "read through {}", client.holder())?;
            }
            client.close().await;
            jobs
        }
        Err(e) => return Err(e),
    };
    jobs.reverse();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    args.format
        .write_rows(&mut out, &jobs, "No jobs on record.", |out, job| {
            writeln!(
                out,
                "#{:<4} {:<9} {:<10} {}  {}{}",
                job.number,
                job.state,
                job.kind,
                job.queued_at,
                job.label,
                job.outcome
                    .as_deref()
                    .map_or(String::new(), |outcome| format!("  ({outcome})"))
            )
        })?;
    out.flush()?;
    Ok(ExitCode::SUCCESS)
}

/// `quack auth register`, signing the person in on this terminal when it
/// comes to that.
async fn run_register(config: &Config, args: auth_cli::RegisterArgs) -> Result<()> {
    let confirm = Confirm::Ask.or_yes(args.yes);
    auth_cli::run_register(
        config,
        args,
        confirm,
        auth_cli::SignInWith {
            browser: browser_can_open(),
            notify: &show_login_prompt,
            interrupt: Box::pin(async {
                // A signal handler that cannot be installed must not read as
                // an interruption.
                if tokio::signal::ctrl_c().await.is_err() {
                    std::future::pending::<()>().await;
                }
            }),
        },
    )
    .await
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
        } => {
            let step = auth_cli::Step::of(rotate, activate);
            auth_cli::run_jwks(config, provider.as_deref(), step).await?;
        }
        AuthAction::Register(args) => run_register(config, args).await?,
        AuthAction::Unregister { issuer, yes } => {
            auth_cli::run_unregister(config, issuer.as_deref(), Confirm::Ask.or_yes(yes)).await?;
        }
    }
    Ok(())
}

/// What `quack auth status` says of one provider's token.
fn token_state(name: &str, token: Option<TokenStatus>, grant: Grant, sends_actor: bool) -> String {
    match (token, grant) {
        // Without an actor token quack has no token of its own here; one
        // stored before `actor = false` is unused.
        (_, Grant::OnBehalfOf) if !sends_actor => String::from(
            "acts on behalf of each person signed in to quack serve, without an actor token; nothing to log in to",
        ),
        (Some(token), Grant::ClientCredentials) => match token.lifetime {
            Lifetime::Valid { expires_at } => {
                format!("token expires {expires_at}, {}", token.renewal)
            }
            Lifetime::Expired { expired_at } => format!(
                "token expired {expired_at}; the client-credentials grant runs again on next use"
            ),
        },
        (Some(token), Grant::AuthorizationCode | Grant::DeviceCode) => {
            match (token.lifetime, token.renewal) {
                (Lifetime::Valid { expires_at }, renewal) => {
                    format!("logged in, token expires {expires_at}, {renewal}")
                }
                (Lifetime::Expired { expired_at }, Renewal::Refreshable | Renewal::Regrant) => {
                    format!("logged in, token expired {expired_at}, refreshes on next use")
                }
                (Lifetime::Expired { expired_at }, Renewal::Relogin) => format!(
                    "token expired {expired_at}, no refresh token; run `quack auth login {name}`"
                ),
            }
        }
        (None, Grant::ClientCredentials) => {
            String::from("no token yet; one is requested on first use")
        }
        (None, Grant::AuthorizationCode | Grant::DeviceCode) => {
            format!("not logged in; run `quack auth login {name}`")
        }
        (Some(token), Grant::OnBehalfOf) => match token.lifetime {
            Lifetime::Valid { expires_at } => format!(
                "acts on behalf of each signed-in person; quack's own token (the actor) expires {expires_at}"
            ),
            Lifetime::Expired { expired_at } => format!(
                "acts on behalf of each signed-in person; quack's own token (the actor) expired {expired_at} and is requested again on next use"
            ),
        },
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

/// Where the session serves the web UI, REST API, and MCP beside itself:
/// `[server].bind`'s port on loopback, or a random loopback port when
/// nothing set one. The host part is not used, since local mode serves
/// without authentication and must stay on loopback; a loopback host
/// configured there is kept as written.
fn local_addr(server: &ServerConfig) -> SocketAddr {
    let configured: Option<SocketAddr> = server.bind.as_deref().and_then(|b| b.parse().ok());
    match configured {
        Some(addr) if addr.ip().is_loopback() => addr,
        Some(addr) => SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, addr.port())),
        None => SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, 0)),
    }
}

/// No arguments: the interactive session, which needs a terminal, and the
/// web UI, REST API, and MCP on loopback beside it from the same engine. A
/// server that cannot start is one transcript line.
async fn run_terminal_session(cli: &Cli, stdout_is_tty: bool) -> Result<ExitCode> {
    if !std::io::stdin().is_terminal() || !stdout_is_tty {
        init_logging();
        tracing::error!(
            "the interactive session needs a terminal; use `quack -p PROMPT` or `quack -q SQL` in pipelines"
        );
        return Ok(ExitCode::from(Exit::Usage));
    }
    if let Some(code) = init_cli::offer_on_first_run().await? {
        return Ok(code);
    }
    let opened = OpenedWorkspace::resolve(cli.workspace.as_deref(), Holder::Terminal).await?;
    let writes = WritePolicy::Ask.allowed_if(cli.allow_write);
    let handle = opened.handle().await?;
    let session_id = {
        let config = opened.config().clone();
        let choice = cli.session_choice();
        let mode = cli.mode.map(ChatMode::from);
        handle
            .writer
            .run(move |db| Ok(resolve_session(&config, db, choice, mode)))
            .await??
    };
    let addr = local_addr(&opened.config().server);
    let started = Server::start(
        Arc::clone(&opened.engine),
        opened.control.clone(),
        addr,
        ServeMode::Local,
    )
    .await;
    let (web, server) = match started {
        Ok(server) => (
            WebStatus::Listening(server.local_addr().unwrap_or(addr)),
            Some(tokio::spawn(server.run(QUIT_GRACE))),
        ),
        Err(e) => (WebStatus::Unavailable(format!("{e:#}")), None),
    };
    let setup = SessionSetup {
        engine: Arc::clone(&opened.engine),
        web,
        workspace_name: opened.workspace.name.clone(),
        workspace_id: opened.workspace.id.clone(),
        db: Arc::clone(&handle.writer),
        reader_db: handle.reader.clone(),
        session_id,
        writes,
        approvals: opened.approvals.clone(),
    };
    // The session's quit stopped the jobs (a session that failed did not:
    // the stop sequence cancels them either way); the server and the
    // socket drain under the same grace, and the writer checkpoints last
    // (`Engine::shutdown`), all before the tty is handed back.
    let quitting: Quitting = Box::new(move || {
        let socket = if opened.socket_busy() {
            ", and a statement another process sent over the workspace socket"
        } else {
            ""
        };
        let line = format!(
            "Quitting: finishing the jobs{socket}{} (up to {} s), then closing the workspace.",
            if server.is_some() {
                ", and the web server's requests"
            } else {
                ""
            },
            QUIT_GRACE.as_secs()
        );
        let server = server.map(|server| -> Drain {
            Box::pin(async move {
                if let Ok(Err(e)) = server.await {
                    tracing::warn!(error = %e, "the web server beside the session ended with an error");
                }
            })
        });
        let stop: Drain = Box::pin(async move {
            drop(handle);
            opened.stop(Stop::Cancel, QUIT_GRACE, server).await;
        });
        (line, stop)
    });
    quack_terminal::run(setup, quitting).await?;
    Ok(ExitCode::SUCCESS)
}

impl Cli {
    fn session_choice(&self) -> SessionChoice {
        match (self.resume.clone(), self.continue_latest) {
            (Some(prefix), _) => SessionChoice::Resume(prefix),
            (None, true) => SessionChoice::Latest,
            (None, false) => SessionChoice::New,
        }
    }
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
    if let [prefix, tag] = args.tag.as_slice() {
        let id = find_document(db, prefix)?;
        let mut tags = db.document(&id)?.map(|d| d.tags).unwrap_or_default();
        if !tags.iter().any(|t| t == tag) {
            tags.push(tag.clone());
        }
        db.set_document_fields(
            &id,
            &DocumentFields {
                tags: Some(tags),
                ..DocumentFields::default()
            },
        )?;
    }
    if let [prefix, tag] = args.untag.as_slice() {
        let id = find_document(db, prefix)?;
        let mut tags = db.document(&id)?.map(|d| d.tags).unwrap_or_default();
        tags.retain(|t| t != tag);
        db.set_document_fields(
            &id,
            &DocumentFields {
                tags: Some(tags),
                ..DocumentFields::default()
            },
        )?;
    }
    if let [prefix, author] = args.author.as_slice() {
        let id = find_document(db, prefix)?;
        db.set_document_fields(
            &id,
            &DocumentFields {
                author: Some(author.clone()),
                ..DocumentFields::default()
            },
        )?;
    }
    if let [prefix, date] = args.authored.as_slice() {
        let id = find_document(db, prefix)?;
        db.set_document_fields(
            &id,
            &DocumentFields {
                authored_at: Some(date.clone()),
                ..DocumentFields::default()
            },
        )?;
    }
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let shown = if args.all { Shown::All } else { Shown::Live };
    list_documents(db, args.format, shown, &mut out)
}

/// Resolve a full document id or a unique prefix.
fn find_document(db: &WorkspaceDb, prefix: &str) -> CoreResult<DocumentId> {
    Ok(db.document_by_id_prefix(prefix)?.id)
}

/// Every document `shown` covers, a page at a time.
fn list_documents(
    db: &WorkspaceDb,
    format: TextOrJson,
    shown: Shown,
    out: &mut impl Write,
) -> Result<()> {
    let mut listing = DocumentListing {
        shown,
        ..DocumentListing::first(DocumentListing::MAX_PAGE)
    };
    loop {
        let page = db.documents(&listing)?;
        format.write_rows(out, &page.documents, "No documents yet.", document_line)?;
        let Some(next) = page.next else {
            break;
        };
        listing.after = Some(next);
    }
    out.flush()?;
    Ok(())
}

/// One document as `quack docs` prints it.
fn document_line<W: Write>(out: &mut W, doc: &DocumentInfo) -> std::io::Result<()> {
    let title = doc
        .title
        .as_deref()
        .map_or(String::new(), |t| format!("  ({t})"));
    let pages = doc
        .pages
        .and_then(PageCounts::note)
        .map_or(String::new(), |note| format!("  [{note}]"));
    let replaced = doc
        .superseded_by
        .as_ref()
        .map_or(String::new(), |by| format!("  -> {by}"));
    let about = match (&doc.author, &doc.authored_at, doc.tags.is_empty()) {
        (None, None, true) => String::new(),
        (author, authored, _) => format!(
            "  [{}]",
            author
                .iter()
                .map(String::as_str)
                .chain(authored.iter().filter_map(|d| d.get(..10)))
                .chain(doc.tags.iter().map(String::as_str))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    writeln!(
        out,
        "{}  {:<10}  {:<6}  {}  {}{title}{about}{pages}{replaced}",
        doc.id,
        doc.status,
        doc.source,
        if doc.pinning == Pinning::Pinned {
            "pinned  "
        } else {
            "        "
        },
        doc.filename
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
            if row.sharing == Sharing::Shared {
                "  (shared)"
            } else {
                ""
            }
        )
    })?;
    out.flush()?;
    Ok(())
}

fn search_sessions(db: &WorkspaceDb, text: &str, format: TextOrJson, limit: u32) -> Result<()> {
    let hits = sessions::search_messages(db, text, &SessionViewer::All, limit)?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    format.write_rows(&mut out, &hits, "Nothing matched.", |out, hit| {
        writeln!(
            out,
            "{}#{}  {}  {}  ({})",
            hit.session_id,
            hit.seq,
            hit.role.as_str(),
            hit.snippet,
            hit.session_title.as_deref().unwrap_or("untitled")
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
    init_logging_as(default, LogFormat::Text);
}

/// Log to stderr at `default` unless `RUST_LOG` says otherwise, as text
/// lines or as one JSON object per line.
fn init_logging_as(default: &str, format: LogFormat) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default));
    let builder = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter);
    match format {
        LogFormat::Text => builder.init(),
        LogFormat::Json => builder.json().init(),
    }
    // The provider is installed at the top of `main`, before any subscriber
    // exists; this is the first point where saying so reaches a log.
    CryptoModule::linked().log();
}

/// `quack ingest DIR`: an OKF bundle, or every supported file under the
/// folder. The per-file flags have no meaning for a folder.
async fn ingest_dir(
    opened: &OpenedWorkspace,
    dir: &Path,
    per_file: bool,
    no_embed: bool,
    prune: bool,
) -> Result<()> {
    anyhow::ensure!(
        !per_file,
        "--replace, --pin, --title, --filename, --author, --authored, --tag, and --types take one file, not a directory"
    );
    if Bundle::is_dir(dir) {
        return ingest_bundle(opened, &dir.display().to_string(), no_embed).await;
    }
    let prune = if prune { Prune::Delete } else { Prune::Keep };
    ingest_folder(opened, dir, no_embed, prune).await
}

/// The lines `quack ingest DIR` prints for a folder: one per file, then
/// the unsupported files and the documents whose file is gone.
fn report_folder(out: &mut impl Write, dir: &Path, report: &FolderReport) -> Result<()> {
    for FileResult { relative, outcome } in &report.results {
        match outcome {
            Outcome::Ingested(id) => writeln!(out, "ingested  {relative} ({id})")?,
            Outcome::Replaced { old, new } => {
                writeln!(out, "replaced  {relative} ({old} -> {new})")?;
            }
            Outcome::Skipped(id) => writeln!(out, "skipped   {relative} (identical to {id})")?,
            Outcome::Moved { document, from } => {
                writeln!(out, "moved     {from} -> {relative} ({document})")?;
            }
            Outcome::Failed(error) => writeln!(out, "failed    {relative}: {error}")?,
        }
    }
    if !report.unsupported.is_empty() {
        writeln!(
            out,
            "Not ingested ({} unsupported): {}",
            report.unsupported.len(),
            report.unsupported.join(", ")
        )?;
    }
    if !report.gone.is_empty() {
        let names: Vec<String> = report
            .gone
            .iter()
            .map(|d| {
                format!(
                    "{} ({})",
                    d.source_path.as_deref().unwrap_or(&d.filename),
                    d.id
                )
            })
            .collect();
        match report.pruned {
            Prune::Delete => {
                writeln!(out, "Deleted ({} gone): {}", names.len(), names.join(", "))?;
            }
            Prune::Keep => writeln!(
                out,
                "Gone from {} ({}, still in the workspace; --prune deletes them): {}",
                dir.display(),
                names.len(),
                names.join(", ")
            )?,
        }
    }
    Ok(())
}

/// `quack ingest DIR` on a folder of files: every file quack can load
/// becomes a document, one line each; a changed file replaces the document
/// at its path, an unchanged one is skipped, unsupported files are listed,
/// and documents whose file is gone are reported, or deleted with
/// `--prune`. A file that fails is reported and fails the command once
/// the rest have run.
async fn ingest_folder(
    opened: &OpenedWorkspace,
    dir: &Path,
    no_embed: bool,
    prune: Prune,
) -> Result<()> {
    let (config, workspace_id) = (opened.config(), opened.workspace.id.as_str());
    let handle = opened.handle().await?;
    let ws_db: &Writer = &handle.writer;
    let embedding_model = if no_embed {
        None
    } else {
        Embeddings::from_config(config)
            .await
            .context("failed to build embedding model")?
    };
    let progress = StderrProgress::new();
    let run = RecordedRun::start(
        &opened.engine.jobs,
        JobKind::Ingest,
        format!("folder {}", dir.display()),
        opened.workspace.id.clone(),
    );
    let report = Folder {
        config,
        db: ws_db,
        workspace_id,
        root: dir,
        embedder: embedding_model.as_ref(),
        control: RunControl {
            progress: &|done| {
                progress.report(done);
                run.report(done);
            },
            cancel: None,
        },
        prune,
    }
    .run()
    .await;
    run.finish(outcome_of(&report, || {
        report.as_ref().map_or(String::new(), |report| {
            format!("{} files", report.results.len())
        })
    }))
    .await;
    let report = report.with_context(|| format!("ingesting {}", dir.display()))?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    report_folder(&mut out, dir, &report)?;
    out.flush()?;
    let stored: Vec<DocumentId> = report
        .results
        .iter()
        .filter_map(|r| match &r.outcome {
            Outcome::Ingested(id) | Outcome::Replaced { new: id, .. } => Some(id.clone()),
            Outcome::Skipped(_) | Outcome::Moved { .. } | Outcome::Failed(_) => None,
        })
        .collect();
    follow_ingest(opened, ws_db, embedding_model.as_ref(), &stored, &mut out).await?;
    out.flush()?;
    let failed = report.failed();
    if failed > 0 {
        anyhow::bail!("{failed} of {} files failed", report.results.len());
    }
    Ok(())
}

/// `quack ingest DIR` on an OKF bundle. Every concept file becomes a
/// Markdown document, its front matter and links feed the ontology review
/// queue, and `index.md` is offered as the workspace context.
async fn ingest_bundle(opened: &OpenedWorkspace, dir: &str, no_embed: bool) -> Result<()> {
    let (config, workspace_id) = (opened.config(), opened.workspace.id.as_str());
    let bundle = Bundle::from_dir(&PathBuf::from(dir))?;
    let handle = opened.handle().await?;
    let ws_db: &Writer = &handle.writer;
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
            ws_db,
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

/// The graph extraction `[graph].follow_ingest` asks for after an ingest
/// or import, run here and reported in one line; nothing when it is off.
async fn follow_ingest(
    opened: &OpenedWorkspace,
    db: &Writer,
    embedder: Option<&Embeddings>,
    documents: &[DocumentId],
    out: &mut impl Write,
) -> Result<()> {
    let progress = StderrProgress::new();
    let followed = FollowUp {
        db,
        config: opened.config(),
        embeddings: embedder,
    }
    .run(
        documents,
        RunControl {
            progress: &|done| progress.report(done),
            cancel: None,
        },
    )
    .await
    .context("graph follow-up failed")?;
    if let Some(summary) = followed {
        writeln!(out, "  Graph: {summary}")?;
    }
    Ok(())
}

async fn run_query(
    sql: &str,
    workspace_name: Option<&str>,
    format: QueryFormat,
    wait_for_stdin: bool,
) -> Result<()> {
    let opened = OpenedWorkspace::resolve(workspace_name, Holder::command("-q")).await?;
    let handle = match opened.handle().await {
        Ok(handle) => handle,
        Err(e) if is_workspace_locked(&e) => {
            return run_query_through_holder(&opened, sql, format, wait_for_stdin, e).await;
        }
        Err(e) => return Err(e),
    };
    let piped = match read_piped_stdin(wait_for_stdin).await? {
        Some(Stdin::Document(piped)) => anyhow::bail!(
            "the piped data is a document ({}); -q runs SQL over tables: ask about it with \
             `quack -p`, or keep it with `quack ingest -`",
            piped.name
        ),
        Some(Stdin::Table(data)) => Some(data),
        None => None,
    };
    // One step on the writer: the temp `stdin` table lives on that
    // connection, so the statement runs where it was loaded. The statement
    // passes the gate every typed statement passes (the terminal's direct
    // SQL, the web SQL page, the holder's `sql` tool): internal tables and
    // the reserved `graph_` and `_quack_` names are refused. Every row is
    // written, since `-q` is the pipeline's export; the row cap is the
    // interactive interfaces' (design doc 11).
    let config = opened.config().clone();
    let workspace_id = opened.workspace.id.as_str().to_owned();
    let sql = sql.to_owned();
    let results = handle
        .writer
        .run(move |db| {
            if let Some(data) = piped {
                load_stdin_table(&config, db, &workspace_id, &data)?;
            }
            let kind = db.classify_user_statement(&sql)?;
            let results = db.execute_query(&sql)?;
            if kind != StatementKind::Read
                && let Err(e) = TableProfile::refresh_stale(db)
            {
                tracing::warn!(error = %e, "could not refresh table profiles after a write");
            }
            Ok(results)
        })
        .await
        .context("query execution failed");
    drop(handle);
    opened.finish(Duration::ZERO).await;
    let results = results?;

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    format.write(&results, &mut out)?;
    out.flush()?;
    Ok(())
}

/// Whether `err` is another process's lock on the workspace file, at any
/// depth of its chain.
fn is_workspace_locked(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<CoreError>(),
            Some(CoreError::WorkspaceLocked { .. })
        )
    })
}

/// What the holder reports while a command waits on it (a write waiting
/// for approval at the terminal's prompt, and for how long), on stderr,
/// so a person or a script knows where the statement is.
fn report_to_stderr(message: &str) {
    let stderr = std::io::stderr();
    let mut err = stderr.lock();
    drop(writeln!(err, "{message}"));
}

/// `-q` on a workspace another `quack` process holds: the statement runs
/// through that process's `sql` tool over the workspace socket
/// (`quack_mcp::client`) and comes back as the same rows, with one line on
/// stderr saying so, one more when the holder capped them at its
/// `[analysis].max_query_rows` or its byte limit, and one more when it cut
/// a wide value (`STRUCTURED_CELL_CHARS`). A write waits for the holder's
/// own decision (the terminal session asks at its prompt and says so here
/// through the holder's progress report; a command refuses), and comes
/// back as the holder's refusal when it is not approved.
/// Without a holder listening, `locked` is the error. Piped data cannot
/// be loaded into a workspace this process does not hold, so a statement
/// that names `stdin` is refused.
async fn run_query_through_holder(
    opened: &OpenedWorkspace,
    sql: &str,
    format: QueryFormat,
    wait_for_stdin: bool,
    locked: anyhow::Error,
) -> Result<()> {
    let workspace_dir = opened.config().workspace_dir(opened.workspace.id.as_str());
    let Some(client) = HolderClient::connect_with(
        &workspace_dir,
        &opened.workspace.id,
        CommandClient::new("quack -q", report_to_stderr),
    )
    .await?
    else {
        return Err(locked);
    };
    if wait_for_stdin || sql.to_ascii_lowercase().contains("stdin") {
        anyhow::bail!(
            "piped data cannot be loaded while the workspace is open in {}; run the query there, \
             or quit it",
            client.holder()
        );
    }
    let rows = client.sql(sql).await?;
    {
        let stderr = std::io::stderr();
        let mut err = stderr.lock();
        writeln!(err, "read through {}", client.holder())?;
        if rows.truncated {
            writeln!(
                err,
                "{} capped the rows at its [analysis].max_query_rows or its byte limit",
                client.holder()
            )?;
        }
        if rows.cells_cut > 0 {
            writeln!(
                err,
                "{} cut {} long value{} at {STRUCTURED_CELL_CHARS} characters",
                client.holder(),
                rows.cells_cut,
                if rows.cells_cut == 1 { "" } else { "s" }
            )?;
        }
    }
    client.close().await;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    format.write(&rows.results, &mut out)?;
    out.flush()?;
    Ok(())
}

impl IngestArgs {
    /// Whether a flag that applies to one file was given.
    fn per_file(&self) -> bool {
        self.replace.is_some()
            || self.pin
            || self.title.is_some()
            || self.filename.is_some()
            || self.author.is_some()
            || self.authored.is_some()
            || !self.tags.is_empty()
            || !self.types.is_empty()
    }
}

async fn run_ingest(cli: &Cli, args: IngestArgs) -> Result<()> {
    let opened =
        OpenedWorkspace::resolve(cli.workspace.as_deref(), Holder::command("ingest")).await?;
    let ran = ingest(&opened, args).await;
    // The file or folder is done, whatever the outcome: the recorded run's
    // last snapshot reaches the file, a graph follow-up the ingest queued
    // finishes, the socket closes, and the writer checkpoints.
    opened.finish(FOLLOW_UP_GRACE).await;
    ran
}

/// `quack ingest` on an open workspace: one file, a folder, or a bundle.
/// Every handle is dropped by the time it returns.
async fn ingest(opened: &OpenedWorkspace, args: IngestArgs) -> Result<()> {
    let per_file = args.per_file();
    let IngestArgs {
        file,
        filename,
        title,
        no_embed,
        pin,
        replace,
        prune,
        author,
        authored,
        tags,
        types,
    } = args;
    let config = opened.config();

    if let StdioPath::Path(dir) = &file
        && dir.is_dir()
    {
        return ingest_dir(opened, dir, per_file, no_embed, prune).await;
    }
    if prune {
        anyhow::bail!("--prune goes with a folder");
    }
    let NamedInput {
        name: effective_filename,
        data,
    } = file.named(filename.as_deref())?;
    let handle = opened.handle().await?;
    let ws_db: &Writer = &handle.writer;
    let replaces = match replace {
        None => None,
        Some(replace) => Some(replace.resolve(ws_db, &effective_filename).await?),
    };
    let embedding_model = if no_embed {
        None
    } else {
        Embeddings::from_config(config)
            .await
            .context("failed to build embedding model")?
    };

    // Recorded as an `ingest` job of the workspace, like an upload or the
    // terminal's load, so `quack jobs` shows it and the file keeps its end.
    let run = RecordedRun::start(
        &opened.engine.jobs,
        JobKind::Ingest,
        effective_filename.clone(),
        opened.workspace.id.clone(),
    );
    let outcome = ingestion::ingest_file(
        config,
        ws_db,
        opened.workspace.id.as_str(),
        &NewFile::of(&effective_filename, data.file_data())
            .source(file.document_source())
            .title(title.as_deref())
            .replaces(replaces.as_ref())
            .types(ColumnTypes::joined(types))
            .fields(DocumentFields {
                title: None,
                author,
                authored_at: authored,
                tags: (!tags.is_empty()).then_some(tags),
            })
            .control(RunControl {
                progress: &|done| run.report(done),
                cancel: None,
            }),
        embedding_model.as_ref(),
    )
    .await;
    run.finish(outcome_of(&outcome, || {
        String::from(match &outcome {
            Ok(IngestOutcome::Ingested(_)) => "ingested",
            Ok(IngestOutcome::Duplicate(_)) => "skipped: identical bytes are already stored",
            Err(_) => "",
        })
    }))
    .await;
    let outcome = outcome.context("ingestion failed")?;
    finish_file_ingest(
        opened,
        ws_db,
        embedding_model.as_ref(),
        outcome,
        &effective_filename,
        pin,
    )
    .await
}

/// After one file was ingested: say so (or that identical bytes were
/// already stored), pin it when asked, and run the graph follow-up.
async fn finish_file_ingest(
    opened: &OpenedWorkspace,
    ws_db: &Writer,
    embedding_model: Option<&Embeddings>,
    outcome: IngestOutcome,
    effective_filename: &str,
    pin: bool,
) -> Result<()> {
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
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
    report_ingested(&mut out, &result, pin)?;
    out.flush()?;
    follow_ingest(
        opened,
        ws_db,
        embedding_model,
        &[result.document_id],
        &mut out,
    )
    .await?;
    out.flush()?;
    Ok(())
}

/// The lines `quack ingest` prints for a stored file.
fn report_ingested(out: &mut impl Write, result: &IngestResult, pin: bool) -> Result<()> {
    writeln!(out, "Ingested: {}", result.filename)?;
    writeln!(out, "  Type: {}", result.file_type)?;
    writeln!(out, "  Document ID: {}", result.document_id)?;
    if let Some(old) = &result.replaced {
        writeln!(out, "  Replaced: {old} (now superseded)")?;
    }
    if pin {
        writeln!(out, "  Pinned: yes")?;
    }

    for table in &result.tables {
        writeln!(out, "  Table: {table}")?;
    }
    if let Some(note) = result.pages.and_then(PageCounts::note) {
        writeln!(out, "  Pages: {note}")?;
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
    Ok(())
}

#[cfg(test)]
mod tests;
