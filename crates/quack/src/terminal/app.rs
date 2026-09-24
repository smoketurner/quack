use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use clap::error::ErrorKind;
use crossterm::event::{
    Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind,
};
use futures::future::BoxFuture;
use ratatui::style::{Color, Style};
use ratatui_textarea::TextArea;
use tokio::sync::{broadcast, mpsc};

use quack_core::analysis::agent::AgentResponse;
use quack_core::analysis::chart::ChartSpec;
use quack_core::analysis::citations::{Citation, Sources};
use quack_core::analysis::events::{self, AgentEvent, PermissionRequest, ToolStep};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::config::Config;
use quack_core::error::{Error as CoreError, Record, Result as CoreResult};
use quack_core::graph::query::{GraphQuery, PathEnds, PathQuery, UnknownEntity};
use quack_core::import::{self, ImportPolicy, ImportRequest};
use quack_core::ingestion::{self, IngestOutcome, NewFile};
use quack_core::jobs::{
    JobContext, JobCounts, JobId, JobInfo, JobKind, JobNumber, JobQueue, JobResult, JobSpec,
    JobState, Lane, LaneKey,
};
use quack_core::llm;
use quack_core::okf;
use quack_core::prefix::PrefixMatch;
use quack_core::priority::Priority;
use quack_core::progress::{ChunkDone, RunControl};
use quack_core::storage::context;
use quack_core::storage::sessions::{self, ChatMode, ExportFormat, MessageRole, Transcript};
use quack_core::storage::workspace::{QueryCanceller, StatementKind, WorkspaceDb};

use crate::ModeArg;
use crate::embeddings_cli::{self, EmbeddingsAction};
use crate::graph_cli::{self, GraphAction};
use crate::ontology_cli::{self, OntologyAction};
use crate::terminal::SessionSetup;
use crate::terminal::chart::ChartData;
use crate::terminal::commands::{Completion, ContextAction, GraphWalk, Input, Route, SlashCommand};
use crate::terminal::ui::{self, JobRow, Scroll, Spinner, Wrapped, one_line};

/// The spinner's frame interval; it ticks only while a job is active.
const SPINNER_MS: u64 = 80;

/// How long quitting waits for cancelled jobs to stop.
const QUIT_GRACE: Duration = Duration::from_secs(3);

/// Lines typed before, newest last; kept per data directory.
const HISTORY_LINES: usize = 500;

const WELCOME_TEXT: &str = "\
Welcome to quack!

Ask questions about your data, or type SQL (SELECT, WITH, FROM, DESCRIBE, SHOW,
SUMMARIZE, PIVOT) to run it directly. Drop a file path here to load it
(CSV, TSV, Parquet, JSON, Excel as tables; PDF, Word, PowerPoint, HTML,
Markdown, text as documents). Type /help for commands.";

/// Shown at start and in place of an answer when `[general].chat_model`
/// is unset: everything but the agent still works.
const NO_CHAT_MODEL_TEXT: &str = "\
No chat model is configured, so questions cannot be answered yet. SQL, file
loading, and every /command work without one. Set [general].chat_model (or
QUACK_MODEL) to PROVIDER/MODEL; `quack doctor` checks the setup and suggests one.";

/// The choices every write prompt offers.
const RUN_IT: &str = "Run it?  y = yes   n = no   a = yes, and allow writes for this session";

/// The answer to `a` at a write prompt.
const ALLOWED_FOR_SESSION: &str = "Allowed. Writes are permitted for the rest of this session.";

/// What a transcript message is, which decides how it is drawn.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum MessageKind {
    User,
    Assistant,
    /// A tool call: `> run_sql` plus its detail and outcome.
    Step,
    /// A direct SQL result table.
    Sql,
    System,
    Error,
}

#[derive(Debug, Clone)]
pub(crate) struct Message {
    pub(crate) kind: MessageKind,
    pub(crate) content: String,
    /// The chart an assistant answer produced (design doc 9: charts belong
    /// to messages); `/chart N` brings it into the chart pane.
    pub(crate) chart: Option<ChartData>,
    /// A step's full tool detail, shown whole when steps are expanded.
    pub(crate) detail: Option<String>,
}

impl Message {
    fn new(kind: MessageKind, content: impl Into<String>) -> Self {
        Self {
            kind,
            content: content.into(),
            chart: None,
            detail: None,
        }
    }

    /// A tool call as it starts: its header, with the detail kept for
    /// `/steps`; the outcome is appended when it finishes.
    fn step_started(tool: &str, detail: String) -> Self {
        Self {
            detail: Some(detail),
            ..Self::new(MessageKind::Step, format!("> {tool}"))
        }
    }

    /// The sources an answer cited, numbered as it cites them.
    fn sources(citations: &[Citation]) -> Self {
        Self::new(MessageKind::System, Sources(citations).to_string())
    }
}

/// A finished step as one message: the header line, the summary, and the
/// full detail kept aside for `/steps`.
impl From<&ToolStep> for Message {
    fn from(step: &ToolStep) -> Self {
        Self {
            detail: Some(step.detail.clone()),
            ..Self::new(
                MessageKind::Step,
                format!(
                    "> {}\n  {}, {} ms",
                    step.tool, step.summary, step.duration_ms
                ),
            )
        }
    }
}

/// What a job that is not an agent turn leaves for the transcript.
enum BackgroundResult {
    Done { kind: MessageKind, text: String },
    Failed(String),
}

impl BackgroundResult {
    /// The job's one-line outcome for `/jobs`: a result table's last line
    /// (its row count and time), any other text's first.
    fn outcome(&self) -> JobResult {
        match self {
            Self::Done {
                kind: MessageKind::Sql,
                text,
            } => Ok(one_line(text.lines().last().unwrap_or_default())),
            Self::Done { text, .. } => Ok(one_line(text)),
            Self::Failed(error) => Err(error.clone()),
        }
    }
}

/// A command's answer, or why it failed with its causes.
impl From<Result<String>> for BackgroundResult {
    fn from(outcome: Result<String>) -> Self {
        match outcome {
            Ok(text) => Self::Done {
                kind: MessageKind::System,
                text,
            },
            Err(e) => Self::Failed(format!("{e:#}")),
        }
    }
}

/// What the background work tells the session, on one channel, in the
/// order it happened. The event loop selects over this, the terminal's
/// input, and the job queue's broadcast; nothing polls.
enum AppMsg {
    /// An event from the agent turn run by the job (boxed: a finished
    /// turn's response is large, and most messages are text deltas).
    Turn(JobId, Box<AgentEvent>),
    /// That turn's event stream ended.
    TurnClosed(JobId),
    /// A job that is not a turn finished, with what to show.
    Finished(JobId, BackgroundResult),
    /// A command's database step answered: apply its result on the loop.
    Apply(Box<dyn FnOnce(&mut App) + Send>),
}

/// Where a command's database step, or a typed statement, runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    /// The reader pool, inside a read-only transaction: never waits on
    /// the writer.
    Read,
    /// The writer, in its interactive line.
    Write,
}

/// A command's database step, run in order by the session's worker.
type DbStep = Box<dyn FnOnce() -> BoxFuture<'static, ()> + Send>;

/// A session loaded for the transcript.
struct Replay {
    session_id: String,
    rows: Vec<sessions::MessageRow>,
}

impl Replay {
    /// `session_id`'s messages, read for the transcript.
    fn load(db: &WorkspaceDb, session_id: &str) -> CoreResult<Self> {
        Ok(Self {
            session_id: session_id.to_owned(),
            rows: sessions::messages(db, session_id)?,
        })
    }
}

/// What `/resume PREFIX` found.
enum Found {
    Current,
    One(Replay),
    None(String),
    Many(String, Vec<String>),
}

/// A submitted job as the terminal refers to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Ticket {
    id: JobId,
    number: JobNumber,
}

impl From<&JobInfo> for Ticket {
    fn from(job: &JobInfo) -> Self {
        Self {
            id: job.id,
            number: job.number,
        }
    }
}

/// A decision the user owes. Prompts are modal, answered in order, while
/// every job keeps running.
enum Prompt {
    /// A write the agent wants to make in the turn run by `job`.
    Agent {
        job: Ticket,
        request: PermissionRequest,
    },
    /// A typed statement that modifies the workspace.
    Sql(String),
}

/// An answer to a write prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Answer {
    Yes,
    No,
    /// Yes, and allow writes for the rest of the session.
    Always,
}

impl Answer {
    /// The answer a key gives, if it is one: `y`, `n` (or Esc), `a`.
    const fn of(code: KeyCode) -> Option<Self> {
        match code {
            KeyCode::Char('y' | 'Y') => Some(Self::Yes),
            KeyCode::Char('a' | 'A') => Some(Self::Always),
            KeyCode::Char('n' | 'N') | KeyCode::Esc => Some(Self::No),
            _ => None,
        }
    }
}

/// An agent turn submitted as a job, and where its text goes. Its events
/// arrive as [`AppMsg::Turn`], forwarded from its stream by a task.
struct Turn {
    job: Ticket,
    /// The session it answers in; its events render only while that
    /// session is on screen.
    session_id: String,
    /// Index of the assistant message text is streaming into, if any.
    streaming: Option<usize>,
    /// Index into `messages` of the step line being filled in.
    open_step: Option<usize>,
    /// The turn reported its end (`TurnComplete` or `Failed`).
    ended: bool,
    /// Its event stream closed; it stays until its job's end is known.
    closed: bool,
}

impl Turn {
    /// How a message names a turn whose session is not on screen.
    fn whose(&self) -> String {
        format!(
            "Job #{} in session {}",
            self.job.number,
            short_id(&self.session_id)
        )
    }
}

/// The command popup's state while a slash command is typed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Popup {
    /// Showing whenever the input has suggestions, with this entry
    /// highlighted.
    Open { selected: usize },
    /// Esc hid it; typing shows it again.
    Hidden,
}

/// Lines typed before, newest last, kept across sessions in the data
/// directory, and where Up and Down have got to in them.
struct InputHistory {
    lines: Vec<String>,
    /// The line recalled, while browsing.
    cursor: Option<usize>,
    path: PathBuf,
}

/// What Down recalls.
enum Recall {
    Line(String),
    /// Past the newest line: an empty input.
    Blank,
}

impl InputHistory {
    fn load(path: PathBuf) -> Self {
        let lines = std::fs::read_to_string(&path)
            .map(|text| text.lines().map(str::to_owned).collect())
            .unwrap_or_default();
        Self {
            lines,
            cursor: None,
            path,
        }
    }

    /// Keep a submitted line and save the newest [`HISTORY_LINES`]; a line
    /// holding a newline is kept for this session only, since the file
    /// has one line per entry.
    fn push(&mut self, line: String) {
        self.lines.push(line);
        self.cursor = None;
        let start = self.lines.len().saturating_sub(HISTORY_LINES);
        let text = self
            .lines
            .iter()
            .skip(start)
            .filter(|line| !line.contains('\n'))
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n");
        if let Err(e) = std::fs::write(&self.path, text) {
            tracing::debug!(path = %self.path.display(), error = %e, "could not save the input history");
        }
    }

    const fn browsing(&self) -> bool {
        self.cursor.is_some()
    }

    const fn leave(&mut self) {
        self.cursor = None;
    }

    /// Up: the line before the one recalled, or the newest.
    fn older(&mut self) -> Option<String> {
        let at = match self.cursor {
            None => self.lines.len().checked_sub(1)?,
            Some(at) => at.saturating_sub(1),
        };
        let line = self.lines.get(at)?.clone();
        self.cursor = Some(at);
        Some(line)
    }

    /// Down: the line after the one recalled, or a blank input past the
    /// newest; nothing when not browsing.
    fn newer(&mut self) -> Option<Recall> {
        let at = self.cursor?.saturating_add(1);
        if let Some(line) = self.lines.get(at) {
            self.cursor = Some(at);
            Some(Recall::Line(line.clone()))
        } else {
            self.cursor = None;
            Some(Recall::Blank)
        }
    }
}

/// What a command job needs from the session, owned so the job can outlive
/// the call that submitted it.
struct JobEnv {
    config: Arc<Config>,
    db: SharedDb,
    workspace_id: String,
    workspace_name: String,
}

/// A command the terminal runs as a job, reporting what it printed.
enum CliJob {
    Ontology(OntologyAction),
    Graph(GraphAction),
    Embeddings(EmbeddingsAction),
    Okf(String),
    ContextImport(String),
    ContextExport(String),
    Import(ImportRequest),
    Ingest(PathBuf),
}

impl CliJob {
    /// `/ontology VERB`. The terminal owns stdin, so nothing may prompt.
    fn ontology(mut action: OntologyAction) -> Self {
        if let OntologyAction::Propose { yes, .. } = &mut action {
            *yes = true;
        }
        Self::Ontology(action)
    }

    /// `/graph VERB`, never prompting.
    fn graph(mut action: GraphAction) -> Self {
        if let GraphAction::Extract { yes, .. } = &mut action {
            *yes = true;
        }
        Self::Graph(action)
    }

    /// `/embeddings VERB`, never prompting.
    const fn embeddings(action: &EmbeddingsAction) -> Self {
        match action {
            EmbeddingsAction::Refresh { .. } => {
                Self::Embeddings(EmbeddingsAction::Refresh { yes: true })
            }
        }
    }

    const fn kind(&self) -> JobKind {
        match self {
            Self::Ontology(_) => JobKind::Ontology,
            Self::Graph(_) => JobKind::Graph,
            Self::Embeddings(_) => JobKind::Embeddings,
            Self::Okf(_) | Self::ContextExport(_) => JobKind::Export,
            Self::ContextImport(_) | Self::Import(_) => JobKind::Import,
            Self::Ingest(_) => JobKind::Ingest,
        }
    }

    /// The job list's label.
    fn label(&self) -> String {
        match self {
            Self::Ontology(_) => String::from("Running ontology command"),
            Self::Graph(_) => String::from("Running graph command"),
            Self::Embeddings(_) => String::from("Refreshing embeddings"),
            Self::Okf(_) => String::from("Exporting the bundle"),
            Self::ContextImport(_) => String::from("Importing the context"),
            Self::ContextExport(_) => String::from("Exporting the context"),
            Self::Import(request) => format!("import {}", import::redact(&request.url)),
            Self::Ingest(path) => path.file_name().map_or_else(
                || path.display().to_string(),
                |name| name.to_string_lossy().into_owned(),
            ),
        }
    }

    /// What the transcript says when it starts.
    fn announcement(&self) -> String {
        match self {
            Self::Import(request) => format!("Importing from {}", import::redact(&request.url)),
            Self::Ingest(path) => format!("Ingesting {}", path.display()),
            Self::Ontology(_)
            | Self::Graph(_)
            | Self::Embeddings(_)
            | Self::Okf(_)
            | Self::ContextImport(_)
            | Self::ContextExport(_) => format!("{}\u{2026}", self.label()),
        }
    }

    /// Run it. Database steps go to the session's workspace writer one at
    /// a time, chunk progress goes to the job strip, and `/cancel` stops
    /// the work that checks for it between batches.
    async fn run(self, env: &JobEnv, ctx: &JobContext) -> Result<String> {
        let progress = |done: ChunkDone| ctx.progress(done.done, done.total);
        let mut out: Vec<u8> = Vec::new();
        match self {
            Self::Ontology(action) => {
                ontology_cli::run(&env.config, &env.db, action, &mut out, &progress).await?;
            }
            Self::Graph(action) => {
                graph_cli::run(&env.config, &env.db, action, &mut out, &progress).await?;
            }
            Self::Embeddings(action) => {
                let cancel = ctx.cancel_token();
                let control = RunControl {
                    progress: &progress,
                    cancel: Some(&cancel),
                };
                embeddings_cli::run(&env.config, &env.db, action, &mut out, control).await?;
            }
            Self::Okf(dir) => {
                let name = env.workspace_name.clone();
                let bundle = env.db.run(move |db| okf::export(db, &name)).await?;
                bundle.write_to(Path::new(&dir))?;
                return Ok(format!("Wrote {} files to {dir}.", bundle.files.len()));
            }
            Self::ContextImport(file) => {
                let text = std::fs::read_to_string(&file)
                    .map_err(|e| anyhow!("cannot read {file}: {e}"))?;
                let stored = env
                    .db
                    .run(move |db| context::set(db, text.trim(), None))
                    .await?;
                return Ok(format!("Context is now version {}.", stored.version));
            }
            Self::ContextExport(file) => {
                let current = env
                    .db
                    .run(context::current)
                    .await?
                    .ok_or_else(|| anyhow!("no workspace context to export"))?;
                std::fs::write(&file, &current.content)
                    .map_err(|e| anyhow!("cannot write {file}: {e}"))?;
                return Ok(format!(
                    "Wrote context version {} to {file}.",
                    current.version
                ));
            }
            Self::Import(request) => return Self::import(env, &request, ctx).await,
            Self::Ingest(path) => return Self::ingest(env, &path, ctx).await,
        }
        Ok(String::from_utf8_lossy(&out).trim_end().to_owned())
    }

    /// Rows from an external source as a workspace table.
    async fn import(env: &JobEnv, request: &ImportRequest, ctx: &JobContext) -> Result<String> {
        let embedding_model = llm::optional_embedding_model(&env.config).await?;
        let cancel = ctx.cancel_token();
        let summary = import::import(
            &env.config,
            &env.db,
            &env.workspace_id,
            request,
            ImportPolicy::owner(),
            embedding_model.as_ref(),
            Some(&cancel),
        )
        .await?;
        Ok(format!(
            "Imported {} rows from {} as table \"{}\" ({} columns).\nYou can now ask questions about this data.",
            summary.rows,
            summary.source,
            summary.table,
            summary.columns.len()
        ))
    }

    /// A file as a table or tables, or as chunks.
    async fn ingest(env: &JobEnv, path: &Path, ctx: &JobContext) -> Result<String> {
        let read = path.to_owned();
        let data = tokio::task::spawn_blocking(move || std::fs::read(read))
            .await?
            .map_err(|e| anyhow!("failed to read {}: {e}", path.display()))?;
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_owned();
        let embedding_model = llm::optional_embedding_model(&env.config).await?;
        let cancel = ctx.cancel_token();
        let outcome = ingestion::ingest_file(
            &env.config,
            &env.db,
            &env.workspace_id,
            &NewFile::new(&filename, &data).cancel(Some(&cancel)),
            embedding_model.as_ref(),
        )
        .await
        .map_err(|e| anyhow!("ingestion failed: {e}"))?;
        let result = match outcome {
            IngestOutcome::Ingested(result) => result,
            IngestOutcome::Duplicate(existing) => {
                return Ok(format!(
                    "Skipped {filename}: identical to {} (id: {}), already in the workspace.",
                    existing.filename, existing.id
                ));
            }
        };
        let tables = match result.tables.as_slice() {
            [] => String::new(),
            [table] => format!(" as table \"{table}\""),
            tables => format!(" as tables {}", tables.join(", ")),
        };
        let chunks = if result.chunks_stored > 0 {
            let embedded = if embedding_model.is_some() {
                " with embeddings"
            } else {
                ""
            };
            format!(", {} chunks{embedded}", result.chunks_stored)
        } else {
            String::new()
        };
        Ok(format!(
            "Loaded {} ({}){tables}{chunks}\nYou can now ask questions about this data.",
            result.filename, result.file_type
        ))
    }
}

/// A typed statement that passed the gate, and where it runs.
struct DirectSql {
    sql: String,
    side: Side,
}

impl DirectSql {
    /// Run it: a read inside a read-only transaction on the reader pool, a
    /// write on the writer (then let the pool notice a temp object it could
    /// not see). A cancel interrupts the statement itself.
    async fn run(
        self,
        db: SharedDb,
        reader: ReaderDb,
        max_rows: u32,
        ctx: &JobContext,
    ) -> BackgroundResult {
        let canceller = QueryCanceller::new();
        let watch = {
            let canceller = canceller.clone();
            let token = ctx.cancel_token();
            tokio::spawn(async move {
                token.cancelled().await;
                canceller.cancel();
            })
        };
        let started = Instant::now();
        let Self { sql, side } = self;
        let outcome = match side {
            Side::Write => {
                let result = db
                    .run(move |db| {
                        db.cancellable(&canceller, |db| db.execute_query_capped(&sql, max_rows))
                    })
                    .await;
                reader.observe_write().await;
                result
            }
            Side::Read => {
                reader
                    .with_db(move |db| {
                        db.cancellable(&canceller, |db| db.execute_query_capped(&sql, max_rows))
                    })
                    .await
            }
        };
        watch.abort();
        let capped = match outcome {
            Ok(capped) => capped,
            Err(e) => return BackgroundResult::Failed(e.to_string()),
        };
        let mut buf = Vec::new();
        if let Err(e) = capped.results.write_table(&mut buf) {
            return BackgroundResult::Failed(format!("failed to render results: {e}"));
        }
        let mut text = String::from_utf8_lossy(&buf).into_owned();
        if capped.truncated() {
            let omitted = format!("... {} more rows not shown\n", capped.omitted());
            text.push_str(&omitted);
        }
        let elapsed = format!("{} ms", started.elapsed().as_millis());
        text.push_str(&elapsed);
        BackgroundResult::Done {
            kind: MessageKind::Sql,
            text,
        }
    }
}

pub(crate) struct App {
    pub(crate) messages: Vec<Message>,
    pub(crate) textarea: TextArea<'static>,
    pub(crate) scroll: Scroll,
    /// How far back the transcript could scroll when last drawn.
    pub(crate) scroll_limit: Cell<usize>,
    pub(crate) should_quit: bool,
    pub(crate) spinner: Spinner,
    pub(crate) workspace_name: String,
    pub(crate) provider_display: String,
    pub(crate) session_id: String,
    pub(crate) current_chart: Option<ChartData>,
    /// Decisions owed, oldest first; the front one is on screen.
    prompts: VecDeque<Prompt>,
    /// Agent turns queued or running, in submission order.
    turns: Vec<Turn>,
    /// The work queue every submission goes through.
    jobs: JobQueue,
    job_events: broadcast::Receiver<JobInfo>,
    /// Jobs still queued or running, for the strip above the input.
    pub(crate) active_jobs: Vec<JobInfo>,
    /// Ctrl+C was pressed once while jobs were running; a second quits.
    quit_armed: bool,
    /// The session's database worker: every command's database step runs
    /// there, in the order typed, never on the event loop's thread.
    db_steps: Option<mpsc::UnboundedSender<DbStep>>,
    /// Database steps sent and not yet applied.
    pending_db: usize,
    /// While a session switch (or mode change) is on its way: the lines
    /// typed meanwhile, submitted once it lands.
    switching: Option<VecDeque<String>>,
    last_sql: Option<String>,
    history: InputHistory,
    popup: Popup,
    config: Arc<Config>,
    workspace_id: String,
    db: SharedDb,
    reader_db: ReaderDb,
    /// Writes allowed for the session (`--allow-write`, or `a` at a
    /// prompt). Shared with queued turns, which read it when they start.
    allow_write: Arc<AtomicBool>,
    /// `/steps`: show tool details whole instead of a preview.
    pub(crate) expand_steps: bool,
    /// Each message's wrapped lines, by index, with the fingerprint they
    /// were rendered from (`ui::format_messages`); interior mutability
    /// because drawing only borrows the app.
    pub(crate) wrap_cache: RefCell<Vec<Option<Wrapped>>>,
    msg_rx: mpsc::UnboundedReceiver<AppMsg>,
    msg_tx: mpsc::UnboundedSender<AppMsg>,
}

impl App {
    pub(crate) fn new(setup: SessionSetup) -> Self {
        let SessionSetup {
            config,
            workspace_name,
            workspace_id,
            db,
            reader_db,
            session_id,
            allow_write,
        } = setup;
        let (msg_tx, msg_rx) = mpsc::unbounded_channel();
        let jobs = JobQueue::from_config(&config.jobs);
        let job_events = jobs.subscribe();
        let mut app = Self {
            messages: Vec::new(),
            textarea: TextArea::default(),
            scroll: Scroll::Latest,
            scroll_limit: Cell::new(0),
            should_quit: false,
            spinner: Spinner::default(),
            workspace_name,
            provider_display: llm::chat_model_display(&config),
            session_id,
            current_chart: None,
            prompts: VecDeque::new(),
            turns: Vec::new(),
            jobs,
            job_events,
            active_jobs: Vec::new(),
            quit_armed: false,
            db_steps: None,
            pending_db: 0,
            switching: None,
            last_sql: None,
            history: InputHistory::load(config.data_dir().join("terminal_history")),
            popup: Popup::Open { selected: 0 },
            config: Arc::new(config),
            workspace_id,
            db,
            reader_db,
            allow_write: Arc::new(AtomicBool::new(allow_write)),
            expand_steps: false,
            wrap_cache: RefCell::new(Vec::new()),
            msg_rx,
            msg_tx,
        };
        app.clear_input();
        app.note(MessageKind::System, WELCOME_TEXT);
        if app.config.general.chat_model.is_none() {
            app.note(MessageKind::System, NO_CHAT_MODEL_TEXT);
        }
        app
    }

    /// Load the session's stored messages into the transcript at startup,
    /// before the loop runs (`/resume` loads through the database worker).
    pub(crate) async fn load_current_session(&mut self) -> Result<()> {
        let id = self.session_id.clone();
        let replay = self.db.run(move |db| Replay::load(db, &id)).await?;
        self.apply_replay(replay);
        Ok(())
    }

    /// Say so at startup when some vectors were made under another
    /// embedding profile or are missing: those chunks are found by keyword
    /// only until `/embeddings refresh` runs.
    pub(crate) async fn note_embedding_status(&mut self) -> Result<()> {
        let status = self.db.run(WorkspaceDb::embedding_status).await?;
        if let Some(note) = status.note() {
            self.note(
                MessageKind::System,
                format!("{note} /embeddings refresh updates them in the background."),
            );
        }
        Ok(())
    }

    /// Queued and running jobs, for the status line.
    pub(crate) fn job_counts(&self) -> JobCounts {
        self.jobs.counts(None)
    }

    /// Put a loaded session's messages in the transcript.
    fn apply_replay(&mut self, replay: Replay) {
        let Replay { session_id, rows } = replay;
        if rows.is_empty() {
            self.note(
                MessageKind::System,
                format!("Session {session_id} has no messages yet."),
            );
            return;
        }
        self.note(
            MessageKind::System,
            format!("Resumed session {session_id} ({} messages)", rows.len()),
        );
        for row in rows {
            let meta = row.metadata.as_ref();
            match row.role {
                MessageRole::User => self.note(MessageKind::User, row.content),
                MessageRole::Assistant => {
                    let mut message = Message::new(MessageKind::Assistant, row.content);
                    if let Some(spec) = meta
                        .and_then(|m| m.get("chart"))
                        .and_then(|c| serde_json::from_value::<ChartSpec>(c.clone()).ok())
                    {
                        let chart = ChartData::from_spec(&spec);
                        self.current_chart = Some(chart.clone());
                        message.chart = Some(chart);
                    }
                    self.post(message);
                    if let Some(citations) = meta
                        .and_then(|m| m.get("citations"))
                        .and_then(|c| serde_json::from_value::<Vec<Citation>>(c.clone()).ok())
                        && !citations.is_empty()
                    {
                        self.post(Message::sources(&citations));
                    }
                }
                // The stored row is the summary; the tool, its detail (the
                // SQL, the search text), and its time sit in its metadata.
                MessageRole::Tool => {
                    let text = |key: &str| {
                        meta.and_then(|m| m.get(key))
                            .and_then(serde_json::Value::as_str)
                    };
                    let step = ToolStep {
                        tool: text("tool").unwrap_or("tool").to_owned(),
                        detail: text("detail").unwrap_or_default().to_owned(),
                        summary: row.content,
                        duration_ms: meta
                            .and_then(|m| m.get("duration_ms"))
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0),
                    };
                    self.post(Message::from(&step));
                }
            }
        }
    }

    /// The event loop: one `select!` over the terminal's input stream, the
    /// session's message channel, the job queue's broadcast, and (only
    /// while a job is active) the spinner's tick. Every message already
    /// waiting is applied before the next draw, so a burst of streamed text
    /// costs one redraw, not one per delta.
    pub(crate) async fn run(self, terminal: &mut ratatui::DefaultTerminal) -> Result<()> {
        self.run_with(terminal, EventStream::new()).await
    }

    /// [`Self::run`] over any backend and input stream, so tests drive the
    /// real loop with scripted keys and a `TestBackend`.
    async fn run_with<B, S>(mut self, terminal: &mut ratatui::Terminal<B>, input: S) -> Result<()>
    where
        B: ratatui::backend::Backend,
        B::Error: std::error::Error + Send + Sync + 'static,
        S: futures::Stream<Item = std::io::Result<Event>> + Unpin,
    {
        use futures::StreamExt as _;

        let mut input = input;
        let mut spinner = tokio::time::interval(Duration::from_millis(SPINNER_MS));
        spinner.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut dirty = true;

        while !self.should_quit {
            if dirty {
                terminal.draw(|frame| ui::draw(frame, &self))?;
            }
            let spinning = self.spinner_active();
            dirty = tokio::select! {
                event = input.next() => match event {
                    Some(Ok(event)) => self.handle_terminal_event(&event),
                    Some(Err(e)) => return Err(e.into()),
                    None => break,
                },
                Some(msg) = self.msg_rx.recv() => {
                    self.handle_msg(msg);
                    true
                }
                job = self.job_events.recv() => {
                    self.handle_job_event(&job);
                    true
                }
                _ = spinner.tick(), if spinning => {
                    self.spinner.advance();
                    true
                }
            };
            if dirty {
                self.pump();
            }
        }

        self.cancel_all_jobs();
        self.wait_for_jobs(QUIT_GRACE).await;
        if let Some((db, session)) = self.session_to_forget() {
            drop(
                db.run(move |db| sessions::delete_if_empty(db, &session))
                    .await,
            );
        }
        Ok(())
    }

    /// A key press, a scroll, or a resize; returns whether to redraw.
    fn handle_terminal_event(&mut self, event: &Event) -> bool {
        match event {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                self.handle_key_event(key.code, key.modifiers);
                true
            }
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::ScrollUp => {
                    self.scroll_up(3);
                    true
                }
                MouseEventKind::ScrollDown => {
                    self.scroll_down(3);
                    true
                }
                _ => false,
            },
            Event::Resize(..) => true,
            _ => false,
        }
    }

    fn scroll_up(&mut self, lines: usize) {
        self.scroll = self.scroll.up(lines);
    }

    fn scroll_down(&mut self, lines: usize) {
        self.scroll = self.scroll.down(lines, self.scroll_limit.get());
    }

    /// After cancelling everything, give the jobs up to `grace` to stop:
    /// a turn records its cancellation, a statement is interrupted, an
    /// ingest drops its embedding requests. Whatever is still running
    /// after that is dropped with the runtime at its next await.
    async fn wait_for_jobs(&mut self, grace: Duration) {
        let expiry = tokio::time::sleep(grace);
        tokio::pin!(expiry);
        while self.jobs.counts(None).active() > 0 {
            tokio::select! {
                () = &mut expiry => return,
                Some(msg) = self.msg_rx.recv() => self.handle_msg(msg),
                job = self.job_events.recv() => self.handle_job_event(&job),
            }
        }
    }

    /// Apply every message already waiting, without waiting for more.
    /// Returns whether there was any.
    pub(crate) fn pump(&mut self) -> bool {
        let mut changed = false;
        while let Ok(msg) = self.msg_rx.try_recv() {
            self.handle_msg(msg);
            changed = true;
        }
        loop {
            match self.job_events.try_recv() {
                Ok(job) => self.handle_job_event(&Ok(job)),
                Err(broadcast::error::TryRecvError::Lagged(n)) => {
                    self.handle_job_event(&Err(broadcast::error::RecvError::Lagged(n)));
                }
                Err(_) => break,
            }
            changed = true;
        }
        changed
    }

    fn handle_msg(&mut self, msg: AppMsg) {
        match msg {
            AppMsg::Turn(job, event) => {
                let Some(at) = self.turns.iter().position(|t| t.job.id == job) else {
                    return;
                };
                let mut turn = self.turns.remove(at);
                self.handle_turn_event(&mut turn, *event);
                self.turns.insert(at, turn);
            }
            AppMsg::TurnClosed(job) => {
                // Nothing will answer its prompts now.
                self.prompts
                    .retain(|p| !matches!(p, Prompt::Agent { job: owner, .. } if owner.id == job));
                if let Some(turn) = self.turns.iter_mut().find(|t| t.job.id == job) {
                    turn.closed = true;
                }
                self.settle_turns();
            }
            AppMsg::Finished(job, result) => self.handle_background_result(job, result),
            AppMsg::Apply(apply) => {
                self.pending_db = self.pending_db.saturating_sub(1);
                apply(self);
            }
        }
    }

    /// Whether the spinner is animating, which needs a redraw every tick
    /// even with no new input or event to react to.
    fn spinner_active(&self) -> bool {
        !self.active_jobs.is_empty()
    }

    /// Whether a permission prompt is on screen (and takes the keys).
    pub(crate) fn awaiting_permission(&self) -> bool {
        !self.prompts.is_empty()
    }

    /// A job changed state, or snapshots were missed (`Lagged`): rebuild
    /// the strip from the queue itself, and settle turns waiting on their
    /// job's end.
    fn handle_job_event(
        &mut self,
        _event: &std::result::Result<JobInfo, broadcast::error::RecvError>,
    ) {
        self.active_jobs = self
            .jobs
            .list()
            .into_iter()
            .filter(|j| !j.state.is_finished())
            .collect();
        self.settle_turns();
    }

    /// Drop the turns whose stream has closed and whose end is known. One
    /// that closed without an answer or a failure failed before the turn
    /// began (no model, a missing session) or was cancelled while queued;
    /// its job says which, once it has finished, and that arrives as a job
    /// event whichever of the two messages comes first.
    fn settle_turns(&mut self) {
        let mut turns = std::mem::take(&mut self.turns);
        turns.retain(|turn| {
            if !turn.closed {
                return true;
            }
            if turn.ended {
                return false;
            }
            match self.jobs.get(turn.job.id) {
                Some(job) if !job.state.is_finished() => true,
                Some(job) => {
                    let outcome = job.outcome.unwrap_or_default();
                    if job.state == JobState::Failed {
                        self.note(MessageKind::Error, outcome);
                    } else {
                        self.note(
                            MessageKind::System,
                            format!("Job #{} {}: {outcome}.", job.number, job.state),
                        );
                    }
                    false
                }
                None => false,
            }
        });
        turns.append(&mut self.turns);
        self.turns = turns;
    }

    fn handle_turn_event(&mut self, turn: &mut Turn, event: AgentEvent) {
        let visible = turn.session_id == self.session_id;
        match event {
            AgentEvent::Status(status) if visible => self.note(MessageKind::System, status),
            AgentEvent::TextDelta(text) if visible => {
                if let Some(idx) = turn.streaming
                    && let Some(target) = self.messages.get_mut(idx)
                {
                    target.content.push_str(&text);
                    self.scroll = Scroll::Latest;
                } else {
                    self.note(MessageKind::Assistant, text);
                    turn.streaming = Some(self.messages.len().saturating_sub(1));
                }
            }
            AgentEvent::ToolStarted { tool, detail } if visible => {
                turn.streaming = None;
                self.post(Message::step_started(&tool, detail));
                turn.open_step = Some(self.messages.len().saturating_sub(1));
            }
            AgentEvent::ToolFinished(step) if visible => {
                if let Some(idx) = turn.open_step.take()
                    && let Some(msg) = self.messages.get_mut(idx)
                {
                    let line = format!("\n  {}, {} ms", step.summary, step.duration_ms);
                    msg.content.push_str(&line);
                } else {
                    self.post(Message::from(&step));
                }
            }
            AgentEvent::Status(_)
            | AgentEvent::TextDelta(_)
            | AgentEvent::ToolStarted { .. }
            | AgentEvent::ToolFinished(_) => {}
            AgentEvent::PermissionRequired(request) => {
                self.ask_for_turn(turn, visible, request);
            }
            AgentEvent::TurnComplete(response) if visible => {
                turn.ended = true;
                self.handle_turn_complete(turn, response);
            }
            AgentEvent::TurnComplete(response) => {
                turn.ended = true;
                let ended = if response.cancelled {
                    "was cancelled"
                } else {
                    "finished"
                };
                self.note(
                    MessageKind::System,
                    format!(
                        "{} {ended}; /resume {} to read it.",
                        turn.whose(),
                        turn.session_id
                    ),
                );
            }
            AgentEvent::Failed(err) => {
                turn.ended = true;
                let text = if visible {
                    err.message
                } else {
                    format!("{} failed: {err}", turn.whose())
                };
                self.note(MessageKind::Error, text);
                turn.streaming = None;
                turn.open_step = None;
            }
        }
    }

    /// Queue a turn's write request as a prompt; one from a session not on
    /// screen says whose it is.
    fn ask_for_turn(&mut self, turn: &mut Turn, visible: bool, request: PermissionRequest) {
        turn.streaming = None;
        let whose = if visible {
            String::from("The agent")
        } else {
            turn.whose()
        };
        self.note(
            MessageKind::System,
            format!(
                "{whose} wants to run a statement that modifies the workspace:\n{}\n{RUN_IT}",
                request.sql
            ),
        );
        self.prompts.push_back(Prompt::Agent {
            job: turn.job,
            request,
        });
    }

    /// Put the validated answer, its sources, chart, and graph results in
    /// the transcript.
    fn handle_turn_complete(&mut self, turn: &mut Turn, response: AgentResponse) {
        if let Some(idx) = turn.streaming
            && let Some(target) = self.messages.get_mut(idx)
        {
            // Citation validation may have renumbered or stripped markers.
            target.content.clone_from(&response.content);
        } else if !response.content.trim().is_empty() {
            self.note(MessageKind::Assistant, response.content);
        }
        if !response.citations.is_empty() {
            self.post(Message::sources(&response.citations));
        }
        if let Some(spec) = &response.chart {
            let chart = ChartData::from_spec(spec);
            self.current_chart = Some(chart.clone());
            if let Some(last) = self
                .messages
                .iter_mut()
                .rev()
                .find(|m| m.kind == MessageKind::Assistant)
            {
                last.chart = Some(chart);
            }
        }
        for result in response.graph.iter().filter(|r| !r.is_empty()) {
            self.note(MessageKind::System, result.to_string());
        }
        if response.write_refused && !self.writes_allowed() {
            self.note(
                MessageKind::System,
                "A write was refused this turn. Answer y next time, or restart with --allow-write.",
            );
        }
        turn.streaming = None;
        turn.open_step = None;
        self.scroll = Scroll::Latest;
    }

    fn writes_allowed(&self) -> bool {
        self.allow_write.load(Ordering::Relaxed)
    }

    /// The newest turn of the session on screen, queued or running.
    fn current_turn(&self) -> Option<Ticket> {
        self.turns
            .iter()
            .rev()
            .find(|t| t.session_id == self.session_id)
            .map(|t| t.job)
    }

    /// Cancel a turn: its pending permission requests are refused first so
    /// the tool returns, then the job's token stops the turn, which core
    /// records as cancelled and completes. A queued turn ends without
    /// running.
    fn cancel_turn(&mut self, job: Ticket) {
        let mut kept = VecDeque::new();
        for prompt in self.prompts.drain(..) {
            match prompt {
                Prompt::Agent {
                    job: owner,
                    request,
                } if owner.id == job.id => request.deny(),
                other => kept.push_back(other),
            }
        }
        self.prompts = kept;
        if self.jobs.cancel(job.id) {
            self.note(
                MessageKind::System,
                format!("Cancelling job #{}\u{2026}", job.number),
            );
        }
    }

    /// `/cancel N`: any job by its number.
    fn cancel_job(&mut self, number: JobNumber) {
        let Some(info) = self.jobs.by_number(number) else {
            self.note(
                MessageKind::Error,
                format!("no job #{number}; /jobs lists them"),
            );
            return;
        };
        if info.state.is_finished() {
            self.note(
                MessageKind::System,
                format!("Job #{} already {}.", info.number, info.state),
            );
        } else if info.kind == JobKind::Chat {
            self.cancel_turn(Ticket::from(&info));
        } else if self.jobs.cancel(info.id) {
            let note = if info.state == JobState::Queued {
                "it will not start"
            } else {
                "it stops at its next checkpoint, or finishes if it has none"
            };
            self.note(
                MessageKind::System,
                format!("Cancelling job #{}: {note}.", info.number),
            );
        }
    }

    /// `/jobs`: active jobs, then the most recent finished ones.
    fn show_jobs(&mut self) {
        let jobs = self.jobs.list();
        if jobs.is_empty() {
            self.note(MessageKind::System, "No jobs yet.");
            return;
        }
        let mut text = String::from("Jobs (newest last):");
        let start = jobs.len().saturating_sub(20);
        for job in jobs.iter().skip(start) {
            text.push_str("\n  ");
            text.push_str(&JobRow(job).listing());
        }
        text.push_str("\n/cancel N stops a queued or running job.");
        self.note(MessageKind::System, text);
    }

    /// Stop every job when the session ends: a running turn is recorded as
    /// cancelled rather than cut off mid-write.
    fn cancel_all_jobs(&mut self) {
        for prompt in self.prompts.drain(..) {
            if let Prompt::Agent { request, .. } = prompt {
                request.deny();
            }
        }
        for job in self.jobs.list() {
            if !job.state.is_finished() {
                self.jobs.cancel(job.id);
            }
        }
    }

    /// Clear the transcript; streaming turns start a new message.
    fn clear_transcript(&mut self) {
        self.messages.clear();
        self.current_chart = None;
        self.scroll = Scroll::Latest;
        for turn in &mut self.turns {
            turn.streaming = None;
            turn.open_step = None;
        }
    }

    fn handle_key_event(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        let ctrl_c = (code, modifiers) == (KeyCode::Char('c'), KeyModifiers::CONTROL);
        if !ctrl_c {
            self.quit_armed = false;
        }
        if ctrl_c {
            // The prompt on screen first, then this session's newest turn.
            match self.prompts.front() {
                Some(Prompt::Agent { job, .. }) => {
                    let job = *job;
                    self.cancel_turn(job);
                    return;
                }
                Some(Prompt::Sql(_)) => {
                    self.handle_permission_key(KeyCode::Esc);
                    return;
                }
                None => {
                    if let Some(job) = self.current_turn() {
                        self.cancel_turn(job);
                        return;
                    }
                }
            }
        }
        if self.awaiting_permission() {
            self.handle_permission_key(code);
            return;
        }
        if self.handle_completion_key(code, modifiers) {
            return;
        }
        match (code, modifiers) {
            (KeyCode::Char('c' | 'q'), KeyModifiers::CONTROL) => {
                let running = self.jobs.counts(None).active();
                if running > 0 && !self.quit_armed {
                    self.quit_armed = true;
                    self.note(
                        MessageKind::System,
                        format!(
                            "{running} job{} still running (/jobs). Press Ctrl+C again to quit and stop {}.",
                            if running == 1 { " is" } else { "s are" },
                            if running == 1 { "it" } else { "them" }
                        ),
                    );
                } else {
                    self.should_quit = true;
                }
            }
            (KeyCode::Esc, _) => {
                if let Some(job) = self.current_turn() {
                    self.cancel_turn(job);
                }
            }
            (KeyCode::Char('l'), KeyModifiers::CONTROL) => {
                self.clear_transcript();
                self.note(MessageKind::System, WELCOME_TEXT);
            }
            (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                self.clear_input();
                self.history.leave();
            }
            (KeyCode::Enter, KeyModifiers::NONE) => self.submit_message(),
            (KeyCode::Up, KeyModifiers::NONE) => {
                if let Some(line) = self.history.older() {
                    self.set_input(&line);
                }
            }
            (KeyCode::Down, KeyModifiers::NONE) => match self.history.newer() {
                Some(Recall::Line(line)) => self.set_input(&line),
                Some(Recall::Blank) => self.clear_input(),
                None => {}
            },
            (KeyCode::PageUp, _) => self.scroll_up(15),
            (KeyCode::PageDown, _) => self.scroll_down(15),
            (KeyCode::Home, _) if self.textarea.is_empty() => self.scroll = Scroll::Top,
            (KeyCode::End, _) if self.textarea.is_empty() => self.scroll = Scroll::Latest,
            _ => {
                if self.textarea.input(KeyEvent::new(code, modifiers)) {
                    self.reset_completion();
                }
                self.history.leave();
            }
        }
    }

    fn handle_permission_key(&mut self, code: KeyCode) {
        let Some(answer) = Answer::of(code) else {
            return;
        };
        let Some(prompt) = self.prompts.pop_front() else {
            return;
        };
        match prompt {
            Prompt::Sql(sql) => self.decide_pending_sql(sql, answer),
            Prompt::Agent { request, .. } => self.decide_agent_write(request, answer),
        }
    }

    /// The user's answer to a write the agent asked for.
    fn decide_agent_write(&mut self, request: PermissionRequest, answer: Answer) {
        match answer {
            Answer::Yes => {
                request.allow();
                self.note(MessageKind::System, "Allowed.");
            }
            Answer::Always => {
                // The rest of this turn through the request, the turns
                // after (queued ones included) through the shared flag each
                // reads when it starts.
                request.allow_for_turn();
                self.allow_write.store(true, Ordering::Relaxed);
                self.note(MessageKind::System, ALLOWED_FOR_SESSION);
            }
            Answer::No => {
                request.deny();
                self.note(MessageKind::System, "Refused.");
            }
        }
    }

    /// The user's answer to a typed statement's write prompt.
    fn decide_pending_sql(&mut self, sql: String, answer: Answer) {
        match answer {
            Answer::No => {
                self.note(MessageKind::System, "Refused.");
                return;
            }
            Answer::Always => {
                self.allow_write.store(true, Ordering::Relaxed);
                self.note(MessageKind::System, ALLOWED_FOR_SESSION);
            }
            Answer::Yes => {}
        }
        self.execute_direct_sql(sql, Side::Write);
    }

    /// What the command popup offers for the input, if it is showing:
    /// one line with the cursor at its end, not recalled from history, and
    /// not hidden with Esc.
    pub(crate) fn completion(&self) -> Option<Completion> {
        if self.popup == Popup::Hidden || self.history.browsing() || self.awaiting_permission() {
            return None;
        }
        let [line] = self.textarea.lines() else {
            return None;
        };
        if self.textarea.cursor() != (0, line.chars().count()) {
            return None;
        }
        Completion::for_line(line)
    }

    /// The highlighted entry of the command popup.
    pub(crate) fn completion_selected(&self) -> usize {
        match self.popup {
            Popup::Open { selected } => selected,
            Popup::Hidden => 0,
        }
    }

    fn reset_completion(&mut self) {
        self.popup = Popup::Open { selected: 0 };
    }

    /// Up/Down move through the popup, Tab fills the highlighted entry in,
    /// Enter fills it in and sends the line when nothing more may follow
    /// (or sends it as typed when there is nothing to fill in), and Esc
    /// hides the popup. Returns whether the key was the popup's.
    fn handle_completion_key(&mut self, code: KeyCode, modifiers: KeyModifiers) -> bool {
        if modifiers != KeyModifiers::NONE {
            return false;
        }
        let Some(completion) = self.completion() else {
            return false;
        };
        let last = completion.items.len().saturating_sub(1);
        let selected = self.completion_selected().min(last);
        match code {
            KeyCode::Up => {
                self.popup = Popup::Open {
                    selected: selected.checked_sub(1).unwrap_or(last),
                };
            }
            KeyCode::Down => {
                self.popup = Popup::Open {
                    selected: if selected >= last {
                        0
                    } else {
                        selected.saturating_add(1)
                    },
                };
            }
            KeyCode::Esc => self.popup = Popup::Hidden,
            KeyCode::Tab | KeyCode::Enter => {
                let Some(item) = completion.get(selected) else {
                    return false;
                };
                let line = self.textarea.lines().concat();
                let filled = completion.apply(&line, item);
                if code == KeyCode::Enter && filled.trim_end() == line.trim_end() {
                    return false;
                }
                let send = code == KeyCode::Enter && item.finishes();
                self.set_input(&filled);
                if send {
                    self.submit_message();
                }
            }
            _ => return false,
        }
        true
    }

    /// An empty input line, styled.
    fn clear_input(&mut self) {
        let mut textarea = TextArea::default();
        textarea.set_cursor_line_style(Style::default());
        textarea.set_cursor_style(Style::default().fg(Color::Reset).bg(Color::White));
        textarea.set_placeholder_text("Ask a question, or type SQL...");
        self.textarea = textarea;
    }

    /// The input set to `text`, as if typed.
    fn set_input(&mut self, text: &str) {
        self.reset_completion();
        self.clear_input();
        for ch in text.chars() {
            self.textarea
                .input(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
        }
    }

    /// Run a typed `/` line; a line the parser refuses is answered in the
    /// transcript (its help as a note, anything else as an error).
    fn handle_slash_command(&mut self, input: &str) {
        let command = match SlashCommand::parse(input) {
            Ok(command) => command,
            Err(e) => {
                let (kind, text) = match e.kind() {
                    ErrorKind::InvalidSubcommand => {
                        let name = input.split_whitespace().next().unwrap_or(input);
                        (MessageKind::Error, format!("unknown command: {name}"))
                    }
                    ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                        (MessageKind::System, e.to_string())
                    }
                    _ => (MessageKind::Error, e.to_string()),
                };
                self.note(kind, text.trim_end());
                return;
            }
        };
        match command {
            SlashCommand::Quit => self.should_quit = true,
            SlashCommand::Clear => {
                self.clear_transcript();
                self.note(MessageKind::System, WELCOME_TEXT);
            }
            SlashCommand::Jobs => self.show_jobs(),
            SlashCommand::Cancel { job } => self.cancel_job(job),
            SlashCommand::Help => self.note(MessageKind::System, SlashCommand::help()),
            SlashCommand::Workspace => self.show_workspace(),
            SlashCommand::Sessions => self.show_sessions(),
            SlashCommand::Resume { id } => self.switch_session(id),
            SlashCommand::New => self.new_session(),
            SlashCommand::Mode { mode: None } => self.show_mode(),
            SlashCommand::Mode { mode: Some(mode) } => self.set_mode(mode),
            SlashCommand::Docs => self.show_documents(),
            SlashCommand::Context { action: None } => self.show_context(),
            SlashCommand::Context {
                action: Some(ContextAction::Import { file }),
            } => self.run_job(CliJob::ContextImport(file)),
            SlashCommand::Context {
                action: Some(ContextAction::Export { file }),
            } => self.run_job(CliJob::ContextExport(file)),
            SlashCommand::Pin { id } => self.set_pinned(id, true),
            SlashCommand::Unpin { id } => self.set_pinned(id, false),
            SlashCommand::Tables => self.show_tables(),
            SlashCommand::Schema { table } => self.show_schema(table),
            SlashCommand::Ingest { path } => match Input::file(&path) {
                Some(path) => self.run_job(CliJob::Ingest(path)),
                None => self.note(
                    MessageKind::Error,
                    format!("'{path}' is not a file quack can ingest"),
                ),
            },
            SlashCommand::Graph {
                action: Some(action),
                ..
            } => self.run_job(CliJob::graph(action)),
            SlashCommand::Graph {
                action: None,
                walk: Some(walk),
            } => self.show_graph(walk),
            SlashCommand::Graph {
                action: None,
                walk: None,
            } => self.note(
                MessageKind::System,
                "Usage: /graph ENTITY [HOPS], or /graph --class CLASS",
            ),
            SlashCommand::Ontology { action } => self.run_job(CliJob::ontology(action)),
            SlashCommand::Delete { id } => self.delete_document(id),
            SlashCommand::Import {
                url,
                table,
                source_table,
                query,
            } => self.run_job(CliJob::Import(ImportRequest {
                url,
                table,
                query,
                source_table,
                limit: None,
            })),
            SlashCommand::Path { route } => self.show_path(&route),
            SlashCommand::Sql {
                statement: Some(sql),
            } => self.run_direct_sql(sql),
            SlashCommand::Sql { statement: None } => self.edit_last_sql(),
            SlashCommand::Share => self.set_shared(true),
            SlashCommand::Unshare => self.set_shared(false),
            SlashCommand::Export { flags, file } => self.export_session(flags.format(), file),
            SlashCommand::Okf { dir } => self.run_job(CliJob::Okf(dir)),
            SlashCommand::Embeddings { action } => self.run_job(CliJob::embeddings(&action)),
            SlashCommand::Chart { n } => self.show_chart(n),
            SlashCommand::Steps => self.toggle_steps(),
            SlashCommand::Model => self.show_models(),
        }
    }

    fn show_workspace(&mut self) {
        let text = format!(
            "Workspace: {} ({})\nSession: {}",
            self.workspace_name, self.workspace_id, self.session_id
        );
        self.note(MessageKind::System, text);
    }

    /// A bare `/sql` puts the last query back in the input to edit.
    fn edit_last_sql(&mut self) {
        match self.last_sql.clone() {
            Some(sql) => self.set_input(&sql),
            None => self.note(
                MessageKind::System,
                "No query has run yet. Use /sql STATEMENT.",
            ),
        }
    }

    fn toggle_steps(&mut self) {
        self.expand_steps = !self.expand_steps;
        self.note(
            MessageKind::System,
            if self.expand_steps {
                "Tool call details expanded."
            } else {
                "Tool call details collapsed."
            },
        );
    }

    fn show_models(&mut self) {
        let embedding = self
            .config
            .embedding_model_ref()
            .ok()
            .flatten()
            .map_or_else(
                || String::from("none (keyword search only)"),
                |m| m.to_string(),
            );
        let text = format!(
            "Chat model: {}\nEmbedding model: {embedding}",
            self.provider_display
        );
        self.note(MessageKind::System, text);
    }

    fn show_sessions(&mut self) {
        self.on_db_ok(
            Side::Read,
            |db| sessions::list_sessions(db, 20),
            |app, rows| {
                if rows.is_empty() {
                    app.note(MessageKind::System, "No sessions yet.");
                    return;
                }
                let mut text = String::from("Sessions (most recent first):");
                for row in rows {
                    let marker = if row.id == app.session_id { "*" } else { " " };
                    let line = format!(
                        "\n{marker} {}  {}  {:>3} msgs  {}",
                        row.id,
                        row.updated_at,
                        row.message_count,
                        row.title.as_deref().unwrap_or("(untitled)")
                    );
                    text.push_str(&line);
                }
                text.push_str(
                    "\nUse /resume ID to switch (any unique prefix works; ids created close \
                     together differ only near the end).",
                );
                app.note(MessageKind::System, text);
            },
        );
    }

    /// `/resume PREFIX`: find the session, then load its messages, both on
    /// the reader; input typed meanwhile waits for the switch.
    fn switch_session(&mut self, prefix: String) {
        let current = self.session_id.clone();
        self.switching.get_or_insert_with(VecDeque::new);
        self.on_db(
            Side::Read,
            move |db| {
                let sessions = sessions::list_sessions(db, 1000)?;
                let found = match PrefixMatch::of(sessions, &prefix, |s| s.id.as_str()) {
                    PrefixMatch::One(session) if session.id == current => Found::Current,
                    PrefixMatch::One(session) => Found::One(Replay::load(db, &session.id)?),
                    PrefixMatch::None => Found::None(prefix),
                    PrefixMatch::Many(sessions) => {
                        Found::Many(prefix, sessions.into_iter().map(|s| s.id).collect())
                    }
                };
                Ok(found)
            },
            |app, found| {
                match found {
                    Ok(Found::Current) => {
                        app.note(MessageKind::System, "That is the current session.");
                    }
                    Ok(Found::One(replay)) => {
                        app.forget_session_if_empty();
                        app.clear_transcript();
                        app.session_id.clone_from(&replay.session_id);
                        app.apply_replay(replay);
                    }
                    Ok(Found::None(prefix)) => {
                        app.note(MessageKind::Error, format!("no session matches '{prefix}'"));
                    }
                    Ok(Found::Many(prefix, ids)) => {
                        let mut text = format!(
                            "'{prefix}' matches {} sessions; use more of the id:",
                            ids.len()
                        );
                        for id in ids.iter().take(10) {
                            text.push_str("\n  ");
                            text.push_str(id);
                        }
                        app.note(MessageKind::Error, text);
                    }
                    Err(e) => app.note(MessageKind::Error, e.to_string()),
                }
                app.finish_switch();
            },
        );
    }

    fn new_session(&mut self) {
        let current = self.session_id.clone();
        let model = self.provider_display.clone();
        self.switching.get_or_insert_with(VecDeque::new);
        self.on_db(
            Side::Write,
            move |db| {
                let mode = sessions::get_session(db, &current)
                    .ok()
                    .flatten()
                    .map_or(ChatMode::Chat, |s| s.mode);
                sessions::create_session(db, &model, mode, None)
            },
            |app, created| {
                match created {
                    Ok(session) => {
                        app.forget_session_if_empty();
                        app.session_id = session.id;
                        app.clear_transcript();
                        let text = format!("New session {}", app.session_id);
                        app.note(MessageKind::System, text);
                    }
                    Err(e) => app.note(MessageKind::Error, e.to_string()),
                }
                app.finish_switch();
            },
        );
    }

    fn show_mode(&mut self) {
        let session = self.session_id.clone();
        self.on_db_ok(
            Side::Read,
            move |db| Ok(sessions::get_session(db, &session)?.map_or(ChatMode::Chat, |s| s.mode)),
            |app, mode| {
                app.note(
                    MessageKind::System,
                    format!("Mode: {mode}. Use /mode chat or /mode query to change it."),
                );
            },
        );
    }

    fn set_mode(&mut self, mode: ModeArg) {
        let mode = ChatMode::from(mode);
        let session = self.session_id.clone();
        // A question typed right after must start in the new mode.
        self.switching.get_or_insert_with(VecDeque::new);
        self.on_db(
            Side::Write,
            move |db| sessions::set_session_mode(db, &session, mode),
            move |app, set| {
                match set {
                    Ok(()) => app.note(
                        MessageKind::System,
                        format!("Mode set to {mode} for this session."),
                    ),
                    Err(e) => app.note(MessageKind::Error, e.to_string()),
                }
                app.finish_switch();
            },
        );
    }

    /// Run a command as a job and show what it printed.
    fn run_job(&mut self, job: CliJob) {
        let env = JobEnv {
            config: Arc::clone(&self.config),
            db: Arc::clone(&self.db),
            workspace_id: self.workspace_id.clone(),
            workspace_name: self.workspace_name.clone(),
        };
        let announcement = job.announcement();
        self.submit_work(
            job.kind(),
            job.label(),
            Some(announcement),
            move |ctx| async move { BackgroundResult::from(job.run(&env, &ctx).await) },
        );
    }

    fn show_schema(&mut self, table: String) {
        self.on_db_ok(
            Side::Read,
            move |db| {
                if db.list_tables()?.contains(&table) {
                    db.describe_table(&table)
                } else {
                    Err(CoreError::Analysis(format!("no table named '{table}'")))
                }
            },
            |app, described| {
                let mut text = format!("{} ({} rows)\n", described.table_name, described.row_count);
                for column in &described.columns {
                    let line = format!("  {} {}\n", column.name, column.column_type);
                    text.push_str(&line);
                }
                let mut buf = Vec::new();
                if described.sample_rows.write_table(&mut buf).is_ok() {
                    text.push_str(&String::from_utf8_lossy(&buf));
                }
                app.note(MessageKind::Sql, text);
            },
        );
    }

    fn delete_document(&mut self, prefix: String) {
        self.on_db_ok(
            Side::Write,
            move |db| {
                let doc = PrefixMatch::of(db.list_documents()?, &prefix, |d| d.id.as_str())
                    .one(Record::Document, &prefix)?;
                db.delete_document(&doc.id).map(|_| doc.filename)
            },
            |app, filename| {
                app.note(
                    MessageKind::System,
                    format!("Deleted {filename} with its chunks, tables, and graph rows."),
                );
            },
        );
    }

    fn set_shared(&mut self, shared: bool) {
        let session = self.session_id.clone();
        self.on_db_ok(
            Side::Write,
            move |db| sessions::set_session_shared(db, &session, shared),
            move |app, ()| {
                app.note(
                    MessageKind::System,
                    if shared {
                        "This session is shared with every member of the workspace."
                    } else {
                        "This session is yours alone again."
                    },
                );
            },
        );
    }

    /// `/export [--sql|--markdown] [FILE]`: the session to a file or into
    /// the transcript.
    fn export_session(&mut self, format: ExportFormat, file: Option<String>) {
        let session = self.session_id.clone();
        self.on_db_ok(
            Side::Read,
            move |db| {
                let found = sessions::get_session(db, &session)?
                    .ok_or_else(|| CoreError::Analysis(String::from("session vanished")))?;
                Transcript::load(db, found)?.render(format)
            },
            move |app, text| match file {
                Some(path) => match std::fs::write(&path, &text) {
                    Ok(()) => {
                        app.note(MessageKind::System, format!("Wrote the session to {path}."));
                    }
                    Err(e) => app.note(MessageKind::Error, format!("cannot write {path}: {e}")),
                },
                None => app.note(MessageKind::Sql, text),
            },
        );
    }

    fn show_context(&mut self) {
        self.on_db_ok(Side::Read, context::current, |app, current| match current {
            Some(current) => app.note(
                MessageKind::System,
                format!(
                    "Workspace context (version {}, {}):\n{}",
                    current.version, current.edited_at, current.content
                ),
            ),
            None => app.note(
                MessageKind::System,
                "No workspace context set. Use `quack context edit` or `quack context import FILE`.",
            ),
        });
    }

    /// `/graph ENTITY [HOPS]` or `/graph --class CLASS`: a tree of the
    /// neighbourhood or of the class's entities.
    fn show_graph(&mut self, walk: GraphWalk) {
        let options = self.config.graph.options();
        let query = match walk {
            GraphWalk::Class(class) => GraphQuery::new(None, Some(&class), None, None),
            GraphWalk::Entity { name, hops } => {
                GraphQuery::new(Some(&name), None, None, Some(hops.get()))
            }
        };
        let query = match query {
            Ok(query) => query,
            Err(e) => return self.note(MessageKind::Error, e.to_string()),
        };
        self.on_db_ok(
            Side::Read,
            move |db| {
                let result = query.run(db, None, &options)?;
                // A walk from an entity always holds that entity, so an
                // empty one means the name resolved to nothing.
                match query.entity.as_deref() {
                    Some(name) if result.nodes.is_empty() => {
                        Err(UnknownEntity::find(db, name, None).into())
                    }
                    Some(_) | None => Ok(result),
                }
            },
            |app, result| app.note(MessageKind::System, result.to_string()),
        );
    }

    /// `/path FROM -> TO`: the shortest relation chain.
    fn show_path(&mut self, route: &Route) {
        let options = self.config.graph.options();
        let query = match PathQuery::new(&route.from, &route.to, None) {
            Ok(query) => query,
            Err(e) => return self.note(MessageKind::Error, e.to_string()),
        };
        let shown = query.clone();
        self.on_db_ok(
            Side::Read,
            move |db| query.run(db, &PathEnds::default(), &options),
            move |app, result| {
                if result.is_empty() {
                    app.note(
                        MessageKind::System,
                        format!(
                            "No path connects {} and {} within {} hops.",
                            shown.from, shown.to, shown.max_hops
                        ),
                    );
                } else {
                    app.note(MessageKind::System, result.to_string());
                }
            },
        );
    }

    fn show_documents(&mut self) {
        self.on_db_ok(Side::Read, WorkspaceDb::list_documents, |app, docs| {
            if docs.is_empty() {
                app.note(MessageKind::System, "No documents yet.");
                return;
            }
            let mut text = String::from("Documents:");
            for doc in docs {
                let line = format!(
                    "\n  {}  {:<10}  {}  {}",
                    short_id(&doc.id),
                    doc.status,
                    if doc.pinned { "pinned" } else { "      " },
                    doc.filename
                );
                text.push_str(&line);
            }
            text.push_str("\nUse /pin ID or /unpin ID.");
            app.note(MessageKind::System, text);
        });
    }

    fn set_pinned(&mut self, prefix: String, pinned: bool) {
        self.on_db_ok(
            Side::Write,
            move |db| {
                let doc = PrefixMatch::of(db.list_documents()?, &prefix, |d| d.id.as_str())
                    .one(Record::Document, &prefix)?;
                db.set_document_pinned(&doc.id, pinned).map(|()| doc.id)
            },
            move |app, id| {
                let done = if pinned { "Pinned" } else { "Unpinned" };
                app.note(MessageKind::System, format!("{done} {}", short_id(&id)));
            },
        );
    }

    /// `/chart [N]`: the Nth chart-bearing answer's chart into the pane
    /// (the last one without N).
    fn show_chart(&mut self, n: Option<usize>) {
        let charts: Vec<ChartData> = self
            .messages
            .iter()
            .filter_map(|m| m.chart.clone())
            .collect();
        if charts.is_empty() {
            self.note(
                MessageKind::System,
                "No chart in this session yet; ask for one.",
            );
            return;
        }
        let wanted = n.unwrap_or(charts.len());
        let Some(chart) = wanted.checked_sub(1).and_then(|at| charts.get(at)) else {
            self.note(
                MessageKind::Error,
                format!("/chart takes a number from 1 to {}", charts.len()),
            );
            return;
        };
        let text = format!(
            "Showing chart {wanted} of {}: {}",
            charts.len(),
            chart.title
        );
        self.current_chart = Some(chart.clone());
        self.note(MessageKind::System, text);
    }

    /// Drop the session on screen if nothing was ever recorded in it,
    /// unless a turn of it is still queued or running (a database step, in
    /// order with the rest).
    fn forget_session_if_empty(&mut self) {
        if self.turns.iter().any(|t| t.session_id == self.session_id) {
            return;
        }
        let session = self.session_id.clone();
        self.on_db(
            Side::Write,
            move |db| sessions::delete_if_empty(db, &session),
            |_, _| {},
        );
    }

    /// The same when the session ends, after the loop: the writer and the
    /// session to drop, taken out first so no borrow of the app is held
    /// across the await.
    fn session_to_forget(&self) -> Option<(SharedDb, String)> {
        if self.turns.iter().any(|t| t.session_id == self.session_id) {
            return None;
        }
        Some((Arc::clone(&self.db), self.session_id.clone()))
    }

    /// Add a message to the transcript and follow it.
    fn post(&mut self, message: Message) {
        self.messages.push(message);
        self.scroll = Scroll::Latest;
    }

    /// Add a line of `kind` to the transcript and follow it.
    fn note(&mut self, kind: MessageKind, text: impl Into<String>) {
        self.post(Message::new(kind, text));
    }

    /// Run `work` against the workspace in the session's database worker,
    /// on `side`, and `apply` its result on the loop. Steps run one at a
    /// time in the order they were sent, so `/pin X` then `/docs` shows the
    /// pin; the loop never waits on the database.
    fn on_db<T, W, A>(&mut self, side: Side, work: W, apply: A)
    where
        T: Send + 'static,
        W: FnOnce(&WorkspaceDb) -> CoreResult<T> + Send + 'static,
        A: FnOnce(&mut App, CoreResult<T>) + Send + 'static,
    {
        let db = Arc::clone(&self.db);
        let reader = self.reader_db.clone();
        let tx = self.msg_tx.clone();
        let step: DbStep = Box::new(move || {
            Box::pin(async move {
                let result = match side {
                    Side::Read => reader.with_db(work).await,
                    Side::Write => db.run_at(Priority::Interactive, work).await,
                };
                drop(tx.send(AppMsg::Apply(Box::new(move |app: &mut App| {
                    apply(app, result);
                }))));
            })
        });
        self.pending_db = self.pending_db.saturating_add(1);
        let steps = self.db_steps.get_or_insert_with(|| {
            let (steps, mut queue) = mpsc::unbounded_channel::<DbStep>();
            tokio::spawn(async move {
                while let Some(step) = queue.recv().await {
                    step().await;
                }
            });
            steps
        });
        if steps.send(step).is_err() {
            self.pending_db = self.pending_db.saturating_sub(1);
            self.note(MessageKind::Error, "the database worker stopped");
        }
    }

    /// [`Self::on_db`] for a step whose failure is only reported: `apply`
    /// gets the value, and an error goes to the transcript.
    fn on_db_ok<T, W, A>(&mut self, side: Side, work: W, apply: A)
    where
        T: Send + 'static,
        W: FnOnce(&WorkspaceDb) -> CoreResult<T> + Send + 'static,
        A: FnOnce(&mut App, T) + Send + 'static,
    {
        self.on_db(side, work, move |app, result| match result {
            Ok(value) => apply(app, value),
            Err(e) => app.note(MessageKind::Error, e.to_string()),
        });
    }

    /// A switch landed: submit what was typed meanwhile, in order.
    fn finish_switch(&mut self) {
        let mut waiting = self.switching.take().unwrap_or_default();
        // A line that starts another switch sends the rest back to wait.
        while self.switching.is_none()
            && let Some(text) = waiting.pop_front()
        {
            self.submit_text(text);
        }
        if let Some(after) = self.switching.as_mut() {
            after.extend(waiting);
        }
    }

    fn submit_message(&mut self) {
        let text: String = self.textarea.lines().join("\n");
        let trimmed = text.trim().to_owned();
        if trimmed.is_empty() {
            return;
        }
        self.history.push(trimmed.clone());
        self.clear_input();
        self.scroll = Scroll::Latest;
        self.submit_text(trimmed);
    }

    /// Act on one submitted line. While a session switch is on its way the
    /// line waits, so it lands in the session the user now expects.
    fn submit_text(&mut self, line: String) {
        if let Some(waiting) = self.switching.as_mut() {
            let first = waiting.is_empty();
            waiting.push_back(line);
            if first {
                self.note(
                    MessageKind::System,
                    "Waiting for the session switch; this runs right after it.",
                );
            }
            return;
        }
        match Input::classify(line) {
            Input::Command(command) => self.handle_slash_command(&command),
            Input::File(path) => self.run_job(CliJob::Ingest(path)),
            Input::Sql(sql) => self.run_direct_sql(sql),
            Input::Question(question) if self.config.general.chat_model.is_none() => {
                self.note(MessageKind::User, question);
                self.note(MessageKind::System, NO_CHAT_MODEL_TEXT);
            }
            Input::Question(question) => self.start_agent_turn(question),
        }
    }

    /// `/sql`: the same gate the agent's statements pass. Internal tables
    /// are refused, an invalid statement is reported, and a write asks
    /// y/n/a unless writes are already allowed for the session.
    fn run_direct_sql(&mut self, sql: String) {
        self.note(MessageKind::User, sql.clone());
        self.last_sql = Some(sql.clone());
        // Classifying is a parse: the reader pool does it, off the loop.
        let statement = sql.clone();
        self.on_db(
            Side::Read,
            move |db| db.classify_user_statement(&statement),
            move |app, kind| app.gate_direct_sql(sql, kind),
        );
    }

    /// Run a classified statement, or ask before a write.
    fn gate_direct_sql(&mut self, sql: String, kind: CoreResult<StatementKind>) {
        match kind {
            Ok(StatementKind::Read) => self.execute_direct_sql(sql, Side::Read),
            Ok(StatementKind::Write) if self.writes_allowed() => {
                self.execute_direct_sql(sql, Side::Write);
            }
            Ok(StatementKind::Write) => {
                self.note(
                    MessageKind::System,
                    format!("This statement modifies the workspace.\n{RUN_IT}"),
                );
                self.prompts.push_back(Prompt::Sql(sql));
            }
            Ok(StatementKind::Invalid(message)) => self.note(MessageKind::Error, message),
            Err(e) => self.note(MessageKind::Error, e.to_string()),
        }
    }

    /// Run a gated statement as a job: a read on the reader pool, so it
    /// never waits on a write, and a write on the writer.
    fn execute_direct_sql(&mut self, sql: String, side: Side) {
        let db = Arc::clone(&self.db);
        let reader = self.reader_db.clone();
        let max_rows = self.config.analysis.max_query_rows;
        let label = one_line(&sql);
        let statement = DirectSql { sql, side };
        self.submit_work(JobKind::Sql, label, None, move |ctx| async move {
            statement.run(db, reader, max_rows, &ctx).await
        });
    }

    fn show_tables(&mut self) {
        self.on_db_ok(Side::Read, WorkspaceDb::list_tables, |app, tables| {
            if tables.is_empty() {
                app.note(MessageKind::System, "No tables yet.");
                return;
            }
            let mut text = String::from("Tables:");
            for table in tables {
                text.push_str("\n  ");
                text.push_str(&table);
            }
            app.note(MessageKind::System, text);
        });
    }

    /// Submit a question as a job in its session's lane: it starts once
    /// the session's previous turn has finished (its history includes that
    /// answer) and a worker is free, and streams into the transcript while
    /// everything else stays usable.
    fn start_agent_turn(&mut self, message: String) {
        self.note(MessageKind::User, message.clone());
        let behind = self.current_turn();

        let (sink, rx) = events::channel();
        let config = Arc::clone(&self.config);
        let db = Arc::clone(&self.db);
        let reader_db = self.reader_db.clone();
        let session_id = self.session_id.clone();
        let allow_write = Arc::clone(&self.allow_write);
        let spec = JobSpec::new(JobKind::Chat, one_line(&message))
            .workspace(self.workspace_id.clone())
            .lane(Lane::serial(&LaneKey::Session(session_id.clone())));
        let job = Ticket::from(&self.jobs.submit(spec, move |ctx| async move {
            // Read when the turn starts, so an `a` answered while it
            // waited applies to it.
            let policy = if allow_write.load(Ordering::Relaxed) {
                WritePolicy::Allow
            } else {
                WritePolicy::Ask
            };
            // run_turn emits TurnComplete or Failed itself; the returned
            // value is the same response, and the job keeps its outline.
            match llm::run_turn(
                &config,
                db,
                reader_db,
                &session_id,
                policy,
                &message,
                sink,
                ctx.cancel_token(),
            )
            .await
            {
                Ok(response) if response.cancelled => Err(String::from("cancelled")),
                Ok(response) => Ok(format!(
                    "answered: {} steps, {} sources",
                    response.steps.len(),
                    response.citations.len()
                )),
                Err(e) => Err(e.to_string()),
            }
        }));
        // Forward the turn's events into the session's one channel; the
        // close says the stream is done, whatever the job reports.
        let tx = self.msg_tx.clone();
        let mut events = rx;
        tokio::spawn(async move {
            while let Some(event) = events.recv().await {
                if tx.send(AppMsg::Turn(job.id, Box::new(event))).is_err() {
                    return;
                }
            }
            drop(tx.send(AppMsg::TurnClosed(job.id)));
        });
        self.turns.push(Turn {
            job,
            session_id: self.session_id.clone(),
            streaming: None,
            open_step: None,
            ended: false,
            closed: false,
        });
        if let Some(previous) = behind {
            self.note(
                MessageKind::System,
                format!(
                    "Queued as job #{}: it runs when job #{} has answered. Esc cancels it.",
                    job.number, previous.number
                ),
            );
        }
    }

    /// Submit background work that is not an agent turn. The work's
    /// result is posted to the transcript when it finishes; `announce`
    /// says what started, with the job's number.
    fn submit_work<F, Fut>(
        &mut self,
        kind: JobKind,
        label: String,
        announce: Option<String>,
        work: F,
    ) where
        F: FnOnce(JobContext) -> Fut + Send + 'static,
        Fut: Future<Output = BackgroundResult> + Send + 'static,
    {
        let tx = self.msg_tx.clone();
        let spec = JobSpec::new(kind, label).workspace(self.workspace_id.clone());
        let job = self.jobs.submit(spec, move |ctx| async move {
            let id = ctx.id();
            let result = work(ctx).await;
            let outcome = result.outcome();
            drop(tx.send(AppMsg::Finished(id, result)));
            outcome
        });
        if let Some(text) = announce {
            self.note(MessageKind::System, format!("{text} (job #{})", job.number));
        }
    }

    fn handle_background_result(&mut self, job: JobId, result: BackgroundResult) {
        match result {
            BackgroundResult::Done { kind, text } => self.note(kind, text),
            BackgroundResult::Failed(err) => match self.jobs.get(job) {
                Some(info) if info.cancel_requested => self.note(
                    MessageKind::System,
                    format!("Job #{} cancelled: {err}", info.number),
                ),
                _ => self.note(MessageKind::Error, err),
            },
        }
    }
}

fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}
#[cfg(test)]
mod tests {
    use quack_core::storage::writer::Writer;

    use quack_core::analysis::agent::AgentResponse;
    use quack_core::analysis::events::ToolStep;

    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    /// An app over a real workspace file (the background jobs open it
    /// again by id), driven without a terminal.
    fn app(dir: &std::path::Path) -> App {
        app_with(dir, Config::default())
    }

    /// `app` under `config`, its data directory moved to `dir`.
    fn app_with(dir: &std::path::Path, mut config: Config) -> App {
        config.general.data_dir = dir.to_path_buf();
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        let session = sessions::create_session(&db, "m", ChatMode::Chat, None)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
        let reader_db = ReaderDb::new(Arc::clone(&db));
        App::new(SessionSetup {
            config,
            workspace_name: String::from("ws"),
            workspace_id: String::from("ws"),
            db,
            reader_db,
            session_id: session.id,
            allow_write: false,
        })
    }

    /// Wait for the background result a command posted and apply it.
    async fn settle(app: &mut App) {
        for _ in 0..400 {
            while let Ok(msg) = app.msg_rx.try_recv() {
                let finished = matches!(msg, AppMsg::Finished(..));
                app.handle_msg(msg);
                if finished {
                    app.pump();
                    return;
                }
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        fail("no background result arrived");
    }

    /// Wait until every database step sent so far has been applied.
    async fn db_settle(app: &mut App) {
        pump_until(app, |app| app.pending_db == 0).await;
    }

    /// Pump until `done` holds.
    async fn pump_until(app: &mut App, done: impl Fn(&App) -> bool) {
        for _ in 0..400 {
            app.pump();
            if done(app) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        fail("the condition never held");
    }

    /// A turn for a job that waits until it is cancelled.
    fn waiting_turn(app: &mut App) -> Turn {
        let job = app.jobs.submit(
            JobSpec::new(JobKind::Chat, "question")
                .lane(Lane::serial(&LaneKey::Session(String::from("test")))),
            |ctx| async move {
                ctx.cancel_token().cancelled().await;
                Err(String::from("cancelled"))
            },
        );
        Turn {
            job: Ticket::from(&job),
            session_id: app.session_id.clone(),
            streaming: None,
            open_step: None,
            ended: false,
            closed: false,
        }
    }

    fn last(app: &App) -> &Message {
        app.messages.last().unwrap_or_else(|| fail("no messages"))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn slash_commands_run_sql_schema_and_cli_verbs_without_a_terminal() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());

        // A write asks first; `y` runs it as a job.
        app.handle_slash_command("/sql CREATE TABLE t AS SELECT 1 AS a, 'x' AS b");
        db_settle(&mut app).await;
        assert!(app.awaiting_permission());
        assert!(matches!(app.prompts.front(), Some(Prompt::Sql(_))));
        app.handle_key_event(KeyCode::Char('y'), KeyModifiers::NONE);
        assert!(!app.awaiting_permission());
        settle(&mut app).await;
        assert_eq!(last(&app).kind, MessageKind::Sql);

        app.handle_slash_command("/tables");
        db_settle(&mut app).await;
        assert!(last(&app).content.contains('t'), "{}", last(&app).content);
        app.handle_slash_command("/schema t");
        db_settle(&mut app).await;
        assert_eq!(last(&app).kind, MessageKind::Sql);
        assert!(
            last(&app).content.contains("a INTEGER"),
            "{}",
            last(&app).content
        );
        app.handle_slash_command("/schema nope");
        db_settle(&mut app).await;
        assert_eq!(last(&app).kind, MessageKind::Error);

        // Internal tables stay refused, a read runs without asking.
        app.handle_slash_command("/sql SELECT * FROM _quack_documents");
        db_settle(&mut app).await;
        assert_eq!(last(&app).kind, MessageKind::Error);
        app.handle_slash_command("/sql SELECT a FROM t");
        settle(&mut app).await;
        assert!(last(&app).content.contains('1'), "{}", last(&app).content);

        // The CLI verbs: clap parses them, background jobs answer.
        app.handle_slash_command("/ontology --help");
        assert!(
            last(&app).content.contains("Usage"),
            "{}",
            last(&app).content
        );
        app.handle_slash_command("/graph status");
        assert!(
            last(&app).content.contains("(job #"),
            "{}",
            last(&app).content
        );
        settle(&mut app).await;
        assert!(
            last(&app).content.contains("Graph: 0 nodes"),
            "{}",
            last(&app).content
        );
        app.handle_slash_command("/ontology init");
        settle(&mut app).await;
        app.handle_slash_command("/ontology show");
        settle(&mut app).await;
        assert!(
            last(&app).content.contains("entity"),
            "{}",
            last(&app).content
        );

        app.handle_slash_command("/export --markdown");
        db_settle(&mut app).await;
        assert_eq!(last(&app).kind, MessageKind::Sql);
        app.handle_slash_command("/share");
        db_settle(&mut app).await;
        assert!(last(&app).content.contains("shared"));
        app.handle_slash_command("/model");
        assert!(last(&app).content.contains("keyword search only"));
        app.handle_slash_command("/nope");
        assert!(last(&app).content.contains("unknown command"));

        // Every job so far is on record.
        app.handle_slash_command("/jobs");
        let listing = &last(&app).content;
        assert!(listing.contains("succeeded sql"), "{listing}");
        assert!(listing.contains("graph"), "{listing}");
        app.handle_slash_command("/cancel 1");
        assert!(last(&app).content.contains("already succeeded"));
        app.handle_slash_command("/cancel x");
        assert_eq!(last(&app).kind, MessageKind::Error);
        assert!(
            last(&app).content.contains("invalid value"),
            "{}",
            last(&app).content
        );
        app.handle_slash_command("/cancel 99");
        assert!(last(&app).content.contains("no job #99"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_events_attach_charts_and_steps_and_keys_cancel_the_turn() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());
        let mut turn = waiting_turn(&mut app);
        app.handle_turn_event(
            &mut turn,
            AgentEvent::ToolStarted {
                tool: String::from("run_sql"),
                detail: (1..=6)
                    .map(|i| format!("line {i}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            },
        );
        app.handle_turn_event(
            &mut turn,
            AgentEvent::ToolFinished(ToolStep {
                tool: String::from("run_sql"),
                detail: String::new(),
                summary: String::from("3 rows"),
                duration_ms: 4,
            }),
        );
        app.handle_turn_event(
            &mut turn,
            AgentEvent::TextDelta(String::from("**Three** rows")),
        );
        let spec: ChartSpec = serde_json::from_value(serde_json::json!({
            "title": "Rows by kind",
            "kind": "bar",
            "x": { "label": "kind", "values": ["a", "b"] },
            "series": [{ "name": "n", "values": [1.0, 2.0] }]
        }))
        .unwrap_or_else(|e| fail(&e.to_string()));
        app.handle_turn_event(
            &mut turn,
            AgentEvent::TurnComplete(AgentResponse {
                content: String::from("**Three** rows"),
                chart: Some(spec),
                ..AgentResponse::default()
            }),
        );
        assert!(turn.streaming.is_none());
        let assistant = app
            .messages
            .iter()
            .rev()
            .find(|m| m.kind == MessageKind::Assistant)
            .unwrap_or_else(|| fail("no assistant message"));
        assert!(assistant.chart.is_some(), "the chart belongs to the answer");
        assert!(app.current_chart.is_some());
        let step = app
            .messages
            .iter()
            .find(|m| m.kind == MessageKind::Step)
            .unwrap_or_else(|| fail("no step"));
        assert!(
            step.detail
                .as_deref()
                .is_some_and(|d| d.lines().count() == 6)
        );

        // Rendering folds the detail to a preview until /steps; the
        // Markdown bold survives as a span; lines wrap to the width.
        let lines = ui::format_messages(&app, 20);
        let text: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.to_string()).collect())
            .collect();
        assert!(text.iter().any(|l| l.contains("3 more lines")), "{text:?}");
        assert!(text.iter().all(|l| l.chars().count() <= 20), "{text:?}");
        app.handle_slash_command("/steps");
        let expanded = ui::format_messages(&app, 80);
        assert!(
            expanded
                .iter()
                .any(|l| l.spans.iter().any(|s| s.content.contains("line 6")))
        );
        app.current_chart = None;
        app.handle_slash_command("/chart 1");
        assert!(app.current_chart.is_some());
        app.handle_slash_command("/chart 9");
        assert_eq!(last(&app).kind, MessageKind::Error);

        // Esc while a turn runs cancels it; typing goes on meanwhile.
        let job = turn.job.id;
        app.turns.push(turn);
        app.handle_key_event(KeyCode::Char('h'), KeyModifiers::NONE);
        assert_eq!(app.textarea.lines().join(""), "h");
        app.handle_key_event(KeyCode::Esc, KeyModifiers::NONE);
        assert!(last(&app).content.contains("Cancelling job #1"));
        let finished = tokio::time::timeout(Duration::from_secs(5), app.jobs.wait(job))
            .await
            .ok()
            .flatten()
            .unwrap_or_else(|| fail("the turn did not end"));
        assert_eq!(finished.state, JobState::Cancelled);
        app.turns.clear();
        app.handle_key_event(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(app.should_quit, "Ctrl+C with nothing running quits");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_event_loop_answers_typed_sql_and_quits_on_ctrl_c() {
        use crossterm::event::KeyEvent;
        use ratatui::backend::TestBackend;

        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let app = app(dir.path());
        let (keys, input) = futures::channel::mpsc::unbounded::<std::io::Result<Event>>();
        let press = |code| Ok(Event::Key(KeyEvent::new(code, KeyModifiers::NONE)));
        for ch in "SELECT 6 * 7 AS answer".chars() {
            assert!(keys.unbounded_send(press(KeyCode::Char(ch))).is_ok());
        }
        assert!(keys.unbounded_send(press(KeyCode::Enter)).is_ok());
        let mut terminal = ratatui::Terminal::new(TestBackend::new(80, 30))
            .unwrap_or_else(|e| fail(&e.to_string()));
        let running = tokio::spawn(async move {
            let result = app.run_with(&mut terminal, input).await;
            (result, terminal)
        });
        // The answer arrives through the job queue and the loop draws it;
        // then Ctrl+C with nothing running quits.
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert!(
            keys.unbounded_send(Ok(Event::Key(KeyEvent::new(
                KeyCode::Char('c'),
                KeyModifiers::CONTROL
            ))))
            .is_ok()
        );
        let (result, terminal) = tokio::time::timeout(Duration::from_secs(10), running)
            .await
            .unwrap_or_else(|_| fail("the loop did not quit"))
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(result.is_ok(), "{result:?}");
        let screen: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        // The result table, not a session id that happens to hold "42".
        assert!(
            screen.contains("answer") && screen.contains("(1 rows)"),
            "{screen}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ctrl_c_with_jobs_running_asks_for_a_second_press() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());
        let (sql_sink, sql_rx) = tokio::sync::oneshot::channel::<()>();
        app.jobs
            .submit(JobSpec::new(JobKind::Sql, "slow"), |_| async move {
                drop(sql_rx.await);
                Ok(String::new())
            });
        pump_until(&mut app, |app| !app.active_jobs.is_empty()).await;
        assert!(!ui::job_strip(&app).is_empty(), "the strip shows it");
        app.handle_key_event(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(!app.should_quit);
        assert!(last(&app).content.contains("still running"));
        app.handle_key_event(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(app.should_quit);
        assert!(sql_sink.send(()).is_ok());
        pump_until(&mut app, |app| app.active_jobs.is_empty()).await;
        assert!(ui::job_strip(&app).is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_command_popup_picks_fills_in_and_runs() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());
        let typed = |app: &mut App, text: &str| {
            for ch in text.chars() {
                app.handle_key_event(KeyCode::Char(ch), KeyModifiers::NONE);
            }
        };
        let input = |app: &App| app.textarea.lines().join("\n");
        let highlighted = |app: &App| {
            app.completion()
                .and_then(|c| c.get(app.completion_selected()).map(|s| s.word.clone()))
        };

        // Tab fills the highlighted command in; its verbs follow.
        typed(&mut app, "/gr");
        assert_eq!(highlighted(&app).as_deref(), Some("/graph"));
        app.handle_key_event(KeyCode::Tab, KeyModifiers::NONE);
        assert_eq!(input(&app), "/graph ");
        typed(&mut app, "me");
        assert_eq!(highlighted(&app).as_deref(), Some("merges"));

        // Down and Up move the highlight and wrap; they leave history alone.
        app.handle_key_event(KeyCode::Char('u'), KeyModifiers::CONTROL);
        typed(&mut app, "/s");
        assert_eq!(highlighted(&app).as_deref(), Some("/sql"));
        app.handle_key_event(KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(highlighted(&app).as_deref(), Some("/schema"));
        app.handle_key_event(KeyCode::Up, KeyModifiers::NONE);
        app.handle_key_event(KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(highlighted(&app).as_deref(), Some("/steps"));
        assert_eq!(input(&app), "/s");

        // Esc hides it without cancelling anything; typing brings it back.
        app.handle_key_event(KeyCode::Esc, KeyModifiers::NONE);
        assert!(app.completion().is_none());
        typed(&mut app, "c");
        assert_eq!(highlighted(&app).as_deref(), Some("/schema"));

        // Enter on a command that takes nothing fills it in and runs it.
        app.handle_key_event(KeyCode::Char('u'), KeyModifiers::CONTROL);
        typed(&mut app, "/he");
        app.handle_key_event(KeyCode::Enter, KeyModifiers::NONE);
        assert!(app.textarea.is_empty());
        assert!(last(&app).content.contains("Commands:"));

        // Enter on one that takes more only fills it in; with nothing left
        // to fill, Enter sends the line as typed.
        typed(&mut app, "/mo");
        app.handle_key_event(KeyCode::Enter, KeyModifiers::NONE);
        assert_eq!(input(&app), "/mode ");
        typed(&mut app, "q");
        app.handle_key_event(KeyCode::Tab, KeyModifiers::NONE);
        assert_eq!(input(&app), "/mode query");
        app.handle_key_event(KeyCode::Enter, KeyModifiers::NONE);
        assert!(app.textarea.is_empty(), "sent as typed");
        db_settle(&mut app).await;
        assert!(
            last(&app).content.contains("query"),
            "{}",
            last(&app).content
        );

        // Plain text, and the cursor moved off the end, show no popup.
        app.handle_key_event(KeyCode::Char('u'), KeyModifiers::CONTROL);
        typed(&mut app, "what is /s");
        assert!(app.completion().is_none());
        app.handle_key_event(KeyCode::Char('u'), KeyModifiers::CONTROL);
        typed(&mut app, "/s");
        app.handle_key_event(KeyCode::Left, KeyModifiers::NONE);
        assert!(app.completion().is_none());

        // The popup's rows: the highlighted one marked, labels aligned.
        let items = Completion::for_line("/mode ")
            .map(|c| c.items)
            .unwrap_or_default();
        let rows: Vec<String> = ui::completion_lines(&items, 1)
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert!(
            rows.first().is_some_and(|r| r.starts_with("  chat ")),
            "{rows:?}"
        );
        assert!(
            rows.get(1)
                .is_some_and(|r| r.starts_with("\u{25B8} query ")),
            "{rows:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn database_commands_run_in_order_off_the_loop_and_input_waits_for_a_switch() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());
        let file = dir.path().join("notes.md");
        std::fs::write(&file, "# Notes\n\nSomething to pin.")
            .unwrap_or_else(|e| fail(&e.to_string()));
        app.run_job(CliJob::Ingest(file));
        settle(&mut app).await;

        // A write then a read, sent back to back, answer in that order: the
        // listing sees the pin.
        let id = app
            .db
            .run(WorkspaceDb::list_documents)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
            .first()
            .map_or_else(|| fail("no document"), |d| d.id.clone());
        app.handle_slash_command(&format!("/pin {id}"));
        app.handle_slash_command("/docs");
        // Neither has answered on the loop yet: nothing blocked here.
        assert_eq!(app.pending_db, 2);
        db_settle(&mut app).await;
        assert!(
            last(&app).content.contains("pinned"),
            "{}",
            last(&app).content
        );

        // Input typed during a switch waits for it, then lands in the new
        // session.
        let old = app.session_id.clone();
        app.handle_slash_command("/new");
        app.set_input("/workspace");
        app.submit_message();
        assert!(
            last(&app)
                .content
                .contains("Waiting for the session switch"),
            "{}",
            last(&app).content
        );
        db_settle(&mut app).await;
        assert_ne!(app.session_id, old);
        assert!(
            last(&app).content.contains(&app.session_id),
            "the deferred /workspace ran in the new session: {}",
            last(&app).content
        );
        assert!(app.switching.is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn embeddings_refresh_is_a_job_and_stale_vectors_are_noted_at_startup() {
        use quack_core::config::{AuthMode, ProviderConfig, ProviderName, ProviderType};
        use quack_core::storage::workspace::{DocumentStatus, NewChunk, NewDocument};

        // Without an embedding model the job says what is missing.
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());
        app.handle_slash_command("/embeddings refresh");
        settle(&mut app).await;
        assert_eq!(last(&app).kind, MessageKind::Error);
        assert!(
            last(&app).content.contains("no embedding model configured"),
            "{}",
            last(&app).content
        );

        // With one, a vector from another profile is noted when the
        // session opens.
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = Config::default();
        config.general.embedding_model = Some(String::from("ollama/embeddinggemma"));
        config.providers.insert(
            "ollama"
                .parse::<ProviderName>()
                .unwrap_or_else(|e| fail(&e.to_string())),
            ProviderConfig {
                provider_type: ProviderType::Ollama,
                auth: AuthMode::None,
                base_url: Some(String::from("http://127.0.0.1:9")),
                api_key_env: None,
                embedding_dimension: Some(4),
                max_concurrent_requests: None,
                oauth: None,
            },
        );
        let mut app = app_with(dir.path(), config);
        app.note_embedding_status()
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let before = app.messages.len();
        app.db
            .run(|db| {
                db.insert_document(
                    &NewDocument::new("d", "a.md", "text/markdown", 1)
                        .with_status(DocumentStatus::Ready),
                )?;
                db.insert_chunk(&NewChunk {
                    id: "c",
                    document_id: "d",
                    chunk_index: 0,
                    content: "levee report",
                    heading: None,
                    page: None,
                    embedding: Some(&[1.0, 0.0, 0.0, 0.0]),
                })?;
                db.execute_statement("UPDATE _quack_chunks SET embedding_profile = 'older'")
            })
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(app.messages.len(), before, "nothing to note while current");
        app.note_embedding_status()
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let note = &last(&app).content;
        assert!(
            note.contains("keyword search only") && note.contains("/embeddings refresh"),
            "{note}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_session_replays_at_startup_through_the_writer() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());
        let session = app.session_id.clone();
        app.db
            .run(move |db| {
                sessions::record_turn(
                    db,
                    &session,
                    "how many storms?",
                    &AgentResponse {
                        content: String::from("Twelve storms."),
                        ..AgentResponse::default()
                    },
                )
            })
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        app.load_current_session()
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            app.messages
                .iter()
                .any(|m| m.kind == MessageKind::Assistant && m.content == "Twelve storms."),
            "the recorded answer is back in the transcript"
        );
        assert!(
            app.messages
                .iter()
                .any(|m| m.content.contains("Resumed session"))
        );
    }

    #[test]
    fn the_transcript_rerenders_only_what_changed() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());
        let text = |lines: &[ratatui::text::Line<'_>]| -> String {
            lines
                .iter()
                .map(|l| {
                    l.spans
                        .iter()
                        .map(|s| s.content.as_ref())
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        app.messages.push(Message::new(
            MessageKind::Assistant,
            String::from("Streaming"),
        ));
        let first = ui::format_messages(&app, 60);
        let entries = app.wrap_cache.borrow().len();
        assert_eq!(entries, app.messages.len());
        let keys: Vec<Option<u64>> = app
            .wrap_cache
            .borrow()
            .iter()
            .map(|e| e.as_ref().map(|w| w.key))
            .collect();

        // A streamed delta changes the last message only.
        if let Some(last) = app.messages.last_mut() {
            last.content.push_str(" more text");
        }
        let second = ui::format_messages(&app, 60);
        assert!(text(&second).contains("Streaming more text"));
        assert_ne!(text(&first), text(&second));
        let after: Vec<Option<u64>> = app
            .wrap_cache
            .borrow()
            .iter()
            .map(|e| e.as_ref().map(|w| w.key))
            .collect();
        let unchanged = keys.iter().zip(&after).filter(|(a, b)| a == b).count();
        assert_eq!(unchanged, keys.len().saturating_sub(1));

        // A new width, /steps, and /clear all show at once.
        assert!(
            ui::format_messages(&app, 20)
                .iter()
                .all(|l| l.width() <= 20)
        );
        app.clear_transcript();
        assert!(ui::format_messages(&app, 60).is_empty());
        assert!(app.wrap_cache.borrow().is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn questions_queue_per_session_while_other_work_runs() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());
        // No chat model: each turn fails fast, but it still goes through
        // the session's lane in order and leaves the transcript usable.
        app.start_agent_turn(String::from("first?"));
        app.start_agent_turn(String::from("second?"));
        assert_eq!(app.turns.len(), 2);
        assert!(
            last(&app).content.contains("Queued as job #2"),
            "{}",
            last(&app).content
        );
        // SQL runs alongside.
        app.set_input("SELECT 6 * 7 AS answer");
        app.submit_message();
        // A job can finish between the result drain and the job-event drain
        // of one pump, so wait for the result itself, not just an idle strip.
        pump_until(&mut app, |app| {
            app.turns.is_empty()
                && app.active_jobs.is_empty()
                && app
                    .messages
                    .iter()
                    .any(|m| m.kind == MessageKind::Sql && m.content.contains("42"))
        })
        .await;
        let jobs = app.jobs.list();
        assert_eq!(jobs.len(), 3);
        let chats: Vec<_> = jobs.iter().filter(|j| j.kind == JobKind::Chat).collect();
        assert!(chats.iter().all(|j| j.state == JobState::Failed));
        // The lane ran them in order.
        assert!(
            chats
                .first()
                .and_then(|a| a.finished_at)
                .zip(chats.get(1).and_then(|b| b.started_at))
                .is_some_and(|(a_end, b_start)| a_end <= b_start),
            "{chats:#?}"
        );
        assert!(
            app.messages.iter().any(|m| m.kind == MessageKind::Error),
            "the failure is in the transcript"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn without_a_chat_model_questions_say_how_to_set_one_and_sql_still_runs() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());
        assert!(app.messages.iter().any(|m| m.content == NO_CHAT_MODEL_TEXT));

        app.set_input("how many orders shipped late?");
        app.submit_message();
        assert!(app.turns.is_empty());
        assert!(last(&app).content.contains("quack doctor"));

        app.set_input("SELECT 41 + 1 AS answer");
        app.submit_message();
        settle(&mut app).await;
        assert!(last(&app).content.contains("42"), "{}", last(&app).content);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn typed_input_is_kept_across_sessions_and_browsed_with_up_and_down() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        {
            let mut app = app(dir.path());
            app.set_input("/tables");
            app.submit_message();
            db_settle(&mut app).await;
        }
        let mut again = app(dir.path());
        assert_eq!(again.history.lines, vec![String::from("/tables")]);

        // Up recalls the newest, then older ones; Down comes back and past
        // the newest to an empty input.
        again.history.push(String::from("SELECT 1"));
        let input = |app: &App| app.textarea.lines().join("\n");
        again.handle_key_event(KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(input(&again), "SELECT 1");
        again.handle_key_event(KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(input(&again), "/tables");
        again.handle_key_event(KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(input(&again), "/tables", "the oldest stays");
        assert!(
            again.completion().is_none(),
            "no popup over a recalled line"
        );
        again.handle_key_event(KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(input(&again), "SELECT 1");
        again.handle_key_event(KeyCode::Down, KeyModifiers::NONE);
        assert!(again.textarea.is_empty());
        assert!(!again.history.browsing());

        // A line with a newline is kept for the session, not the file.
        again.history.push(String::from("a\nb"));
        assert_eq!(again.history.lines.len(), 3);
        let saved = app(dir.path());
        assert_eq!(
            saved.history.lines,
            vec![String::from("/tables"), String::from("SELECT 1")]
        );
    }

    #[test]
    fn home_page_down_and_end_move_through_the_transcript() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());
        app.scroll_limit.set(40);
        app.handle_key_event(KeyCode::PageUp, KeyModifiers::NONE);
        assert_eq!(app.scroll, Scroll::Back(15));
        app.handle_key_event(KeyCode::Home, KeyModifiers::NONE);
        assert_eq!(app.scroll, Scroll::Top);
        assert_eq!(app.scroll.to_string(), " \u{00B7} scroll: top");
        // From the top, PageDown moves down from the first line.
        app.handle_key_event(KeyCode::PageDown, KeyModifiers::NONE);
        assert_eq!(app.scroll, Scroll::Back(25));
        app.handle_key_event(KeyCode::End, KeyModifiers::NONE);
        assert_eq!(app.scroll, Scroll::Latest);
        assert!(app.scroll.to_string().is_empty());
        app.handle_key_event(KeyCode::PageUp, KeyModifiers::NONE);
        app.handle_key_event(KeyCode::PageDown, KeyModifiers::NONE);
        assert_eq!(app.scroll, Scroll::Latest);
        // A new message follows the transcript down.
        app.handle_key_event(KeyCode::PageUp, KeyModifiers::NONE);
        app.note(MessageKind::System, "news");
        assert_eq!(app.scroll, Scroll::Latest);
    }

    #[test]
    fn a_finished_job_reports_one_line() {
        let table = BackgroundResult::Done {
            kind: MessageKind::Sql,
            text: String::from("a\n1\n(1 rows)\n3 ms"),
        };
        assert_eq!(table.outcome(), Ok(String::from("3 ms")));
        let note = BackgroundResult::from(Ok(String::from("Loaded x\nYou can now ask")));
        assert_eq!(note.outcome(), Ok(String::from("Loaded x")));
        let failed = BackgroundResult::from(Err(anyhow!("outer").context("while loading")));
        assert_eq!(failed.outcome(), Err(String::from("while loading: outer")));
    }
}
