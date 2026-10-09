use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use clap::error::ErrorKind;
use crossterm::event::{
    Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
use futures::future::BoxFuture;
use ratatui::style::{Color, Style};
use ratatui_textarea::{CursorMove, TextArea};
use tokio::sync::{broadcast, mpsc};

use jiff::Timestamp;
use quack_core::analysis::agent::AgentResponse;
use quack_core::analysis::citations::{Citation, Sources};
use quack_core::analysis::events::{
    self, AgentEvent, Decision, Delivery, PermissionRequest, ToolName, ToolStep,
};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::search::{DocumentScope, DocumentSearch, SearchDetail};
use quack_core::analysis::tools::{ReaderDb, Rerank, SharedDb};
use quack_core::config::Config;
use quack_core::error::{Error as CoreError, Result as CoreResult};
use quack_core::graph::follow_up::FollowUp;
use quack_core::graph::query::{GraphQuery, PathEnds, PathQuery, UnknownEntity};
use quack_core::ids::{DocumentId, SessionId, WorkspaceId};
use quack_core::import::{self, ImportPolicy, ImportRequest, SourceHeader};
use quack_core::ingestion::parser::PageCounts;
use quack_core::ingestion::{self, IngestOutcome, NewFile};
use quack_core::jobs::{
    JobContext, JobCounts, JobId, JobInfo, JobKind, JobNumber, JobQueue, JobResult, JobSpec,
    JobState, Lane, LaneKey,
};
use quack_core::llm::oauth::KeySource;
use quack_core::llm::{self, Embeddings};
use quack_core::okf::{self, DirSink};
use quack_core::prefix::PrefixMatch;
use quack_core::priority::Priority;
use quack_core::progress::{ChunkDone, RunControl};
use quack_core::storage::context;
use quack_core::storage::control::ControlPlane;
use quack_core::storage::input_history;
use quack_core::storage::profile::TableProfile;
use quack_core::storage::sessions::{
    self, ChatMode, ExportFormat, MessageRole, SessionRow, SessionViewer, Sharing, TitleSource,
    Transcript,
};
use quack_core::storage::workspace::{
    DocumentListing, Pinning, QueryCanceller, QueryResults, SqlSchema, StatementKind, WorkspaceDb,
};
use quack_core::text::OneLine;
use quack_core::vault::Vault;

use crate::SessionSetup;
use crate::chart::ChartData;
use crate::clipboard::{Clipboard, CopyStatus};
use crate::commands::{Completion, ContextAction, FileLine, GraphWalk, Input, Route, SlashCommand};
use crate::picker::{Picked, Picker};
use crate::selection::{Edge, Located, Selection, TranscriptView};
use crate::ui::{self, JobRow, Scroll, Spinner, Wrapped, one_line};
use quack_cli::Confirm;
use quack_cli::ModeArg;
use quack_cli::TextOrJson;
use quack_cli::embeddings_cli::{self, EmbeddingsAction};
use quack_cli::graph_cli::GraphAction;
use quack_cli::ontology_cli::{self, OntologyAction};
use quack_cli::saved_cli::{self, SavedAction};
use quack_cli::tables_cli::TablesArgs;
use quack_cli::{ImportAction, ImportContext};

/// The spinner's frame interval; it ticks only while a job is active.
const SPINNER_MS: u64 = 80;

/// How long quitting waits for cancelled jobs to stop.
const QUIT_GRACE: Duration = Duration::from_secs(3);

/// Sessions `/sessions` lists at most, newest first.
const PICKER_SESSIONS: u32 = 200;

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
loading, and every /command work without one. Run `quack init` to set up a
provider, or set [general].chat_model (or QUACK_MODEL) to PROVIDER/MODEL.";

/// The key that gives each answer at a write prompt.
pub(crate) const fn answer_key(decision: Decision) -> char {
    match decision {
        Decision::Allow => 'y',
        Decision::Deny => 'n',
        Decision::AllowTurn => 'a',
    }
}

/// Every answer with its key, as `[y] Run it   [n] Don't run it   ...`.
pub(crate) fn answer_keys() -> String {
    Decision::CHOICES
        .into_iter()
        .map(|decision| format!("[{}] {}", answer_key(decision), decision.label()))
        .collect::<Vec<_>>()
        .join("   ")
}

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
    /// A file on its way in.
    Upload,
    Error,
}

#[derive(Debug, Clone)]
pub(crate) struct Message {
    pub(crate) kind: MessageKind,
    pub(crate) content: String,
    /// The chart an assistant answer produced (design doc 9: charts belong
    /// to messages), drawn in the transcript under its text.
    pub(crate) chart: Option<ChartData>,
    /// A step's full tool detail, shown whole when steps are expanded.
    pub(crate) detail: Option<String>,
    /// The rows a `run_sql` or `create_chart` step kept, shown as a
    /// table when steps are expanded.
    pub(crate) result: Option<QueryResults>,
}

impl Message {
    fn new(kind: MessageKind, content: impl Into<String>) -> Self {
        Self {
            kind,
            content: content.into(),
            chart: None,
            detail: None,
            result: None,
        }
    }

    /// A tool call as it starts: its header, with the detail kept for
    /// `/steps`; the outcome is appended when it finishes.
    fn step_started(tool: ToolName, detail: String) -> Self {
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
            result: step.result.clone(),
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
    session_id: SessionId,
    rows: Vec<sessions::MessageRow>,
}

impl Replay {
    /// `session_id`'s messages, read for the transcript.
    fn load(db: &WorkspaceDb, session_id: &SessionId) -> CoreResult<Self> {
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
    Many(String, Vec<SessionId>),
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

/// A write the agent wants to make in the turn run by `job`: a decision
/// the user owes. Prompts are modal, answered in order, while every job
/// keeps running.
enum Prompt {
    /// A write the agent wants to make in the turn run by `job`.
    Write {
        job: Ticket,
        request: PermissionRequest,
    },
    /// A deletion the person asked for, held until they confirm it, as the
    /// web asks before its delete buttons.
    Delete(Deletion),
}

impl Prompt {
    /// Whether it is a write the turn run by `job` asked for.
    fn is_write_of(&self, job: JobId) -> bool {
        match self {
            Self::Write { job: owner, .. } => owner.id == job,
            Self::Delete(_) => false,
        }
    }

    /// Refuse a write, or drop a deletion: nobody will answer it now.
    fn refuse(self) {
        match self {
            Self::Write { request, .. } => request.deny(),
            Self::Delete(_) => {}
        }
    }
}

/// What a confirmed `/delete` or the session list's `d` removes.
pub(crate) enum Deletion {
    Document { id: DocumentId, filename: String },
    Session { id: SessionId, title: String },
}

impl Deletion {
    /// The question the prompt asks.
    fn question(&self) -> String {
        match self {
            Self::Document { filename, .. } => {
                format!("Delete {filename} with its chunks, tables, and graph rows?")
            }
            Self::Session { title, .. } => {
                format!("Delete the session '{title}' and its messages?")
            }
        }
    }
}

/// The keys a deletion prompt takes.
const CONFIRM_KEYS: &str = "[y] Delete   [n] Keep";

/// What the prompt overlay asks about: the front prompt, described when
/// drawn so it never depends on what the transcript still shows.
pub(crate) struct PendingPrompt<'a> {
    /// What is asked, and for a write who asks, named from the session on
    /// screen now.
    pub(crate) heading: String,
    /// The statement a write would run; empty for a deletion.
    pub(crate) body: &'a str,
    /// Why the write is held, when there is more to say than "this writes".
    pub(crate) notice: Option<&'static str>,
    /// The answers and their keys.
    pub(crate) keys: String,
    /// Prompts queued behind this one.
    pub(crate) waiting: usize,
}

/// An agent turn submitted as a job, and where its text goes. Its events
/// arrive as [`AppMsg::Turn`], forwarded from its stream by a task.
struct Turn {
    job: Ticket,
    /// The session it answers in; its events render only while that
    /// session is on screen.
    session_id: SessionId,
    /// Index of the assistant message text is streaming into, if any.
    streaming: Option<usize>,
    /// Index into `messages` of the step line being filled in.
    open_step: Option<usize>,
    progress: TurnProgress,
    phase: Phase,
}

/// What a running turn is doing, for its row in the job strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// No output from the model yet: since the job started, or since
    /// the tool before this model call finished.
    Waiting {
        since: Option<Timestamp>,
    },
    /// The model is reasoning, which the transcript does not show.
    Thinking {
        since: Timestamp,
    },
    Answering,
    Tool(ToolName),
}

impl Phase {
    /// The phase `event` leaves the turn in, at `now`.
    fn after(self, event: &AgentEvent, now: Timestamp) -> Self {
        match event {
            AgentEvent::Reasoning => match self {
                // More reasoning in one model call keeps its start.
                Self::Thinking { .. } => self,
                Self::Waiting { .. } | Self::Answering | Self::Tool(_) => {
                    Self::Thinking { since: now }
                }
            },
            AgentEvent::TextDelta(_) => Self::Answering,
            AgentEvent::ToolStarted { tool, .. } => Self::Tool(*tool),
            AgentEvent::ToolFinished(_) => Self::Waiting { since: Some(now) },
            AgentEvent::Status(_)
            | AgentEvent::PermissionRequired(_)
            | AgentEvent::TurnComplete(_)
            | AgentEvent::Failed(_) => self,
        }
    }

    /// `thinking 41s`, as of `now`, for a job that started at `started`.
    pub(crate) fn note(self, started: Option<Timestamp>, now: Timestamp) -> String {
        let seconds = |since: Option<Timestamp>| {
            since.map_or(0, |since| now.duration_since(since).as_secs().max(0))
        };
        match self {
            Self::Waiting { since } => format!("waiting {}s", seconds(since.or(started))),
            Self::Thinking { since } => format!("thinking {}s", seconds(Some(since))),
            Self::Answering => String::from("answering"),
            Self::Tool(tool) => format!("running {tool}"),
        }
    }
}

/// Where a turn is. Its end (`TurnComplete` or `Failed`) and the close of
/// its event stream arrive in either order; a turn whose stream closed
/// without an end stays until its job's end is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnProgress {
    Streaming,
    Ended,
    Closed,
    /// Ended and closed: nothing more will arrive.
    Done,
}

impl TurnProgress {
    const fn ended(self) -> Self {
        match self {
            Self::Streaming | Self::Ended => Self::Ended,
            Self::Closed | Self::Done => Self::Done,
        }
    }

    const fn closed(self) -> Self {
        match self {
            Self::Streaming | Self::Closed => Self::Closed,
            Self::Ended | Self::Done => Self::Done,
        }
    }
}

impl Turn {
    /// How a message names a turn whose session is not on screen.
    fn whose(&self) -> String {
        format!(
            "Job #{} in session {}",
            self.job.number,
            self.session_id.short()
        )
    }

    /// Who asks for a write, seen from the session `on_screen`.
    fn speaker(&self, on_screen: &SessionId) -> String {
        if &self.session_id == on_screen {
            String::from("The agent")
        } else {
            self.whose()
        }
    }
}

/// Why a line is run as SQL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SqlIntent {
    /// `/sql` said so.
    Stated,
    /// It starts with a SQL keyword, as a question can.
    Guessed,
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

/// Lines typed before, newest last, kept across sessions in the workspace
/// (`storage::input_history`), and where Up and Down have got to in them.
#[derive(Default)]
struct InputHistory {
    lines: Vec<String>,
    /// The line recalled, while browsing.
    cursor: Option<usize>,
}

/// What Down recalls.
enum Recall {
    Line(String),
    /// Past the newest line: an empty input.
    Blank,
}

impl InputHistory {
    /// Keep a submitted line for this session; the caller stores it.
    fn push(&mut self, line: String) {
        self.lines.push(line);
        self.cursor = None;
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
    workspace_id: WorkspaceId,
    workspace_name: String,
    /// The session `/saved add` takes its last answer from.
    session_id: SessionId,
}

/// A command the terminal runs as a job, reporting what it printed.
enum CliJob {
    Ontology(OntologyAction),
    Graph(GraphAction),
    Embeddings(EmbeddingsAction),
    Saved(SavedAction),
    Okf(String),
    ContextImport(String),
    ContextExport(String),
    Import(ImportRequest),
    SavedImport(ImportAction),
    Ingest(PathBuf),
    Search(String),
}

impl CliJob {
    const fn kind(&self) -> JobKind {
        match self {
            Self::Ontology(_) => JobKind::Ontology,
            Self::Graph(_) => JobKind::Graph,
            Self::Embeddings(_) => JobKind::Embeddings,
            Self::Saved(_) => JobKind::Sql,
            Self::Okf(_) | Self::ContextExport(_) => JobKind::Export,
            Self::ContextImport(_) | Self::Import(_) | Self::SavedImport(_) => JobKind::Import,
            Self::Ingest(_) => JobKind::Ingest,
            Self::Search(_) => JobKind::Search,
        }
    }

    /// The job list's label.
    fn label(&self) -> String {
        match self {
            Self::Ontology(_) => String::from("Running ontology command"),
            Self::Graph(_) => String::from("Running graph command"),
            Self::Embeddings(_) => String::from("Refreshing embeddings"),
            Self::Saved(_) => String::from("Running saved question command"),
            Self::Okf(_) => String::from("Exporting the bundle"),
            Self::ContextImport(_) => String::from("Importing the context"),
            Self::ContextExport(_) => String::from("Exporting the context"),
            Self::Import(request) => format!("import {}", request.url),
            Self::SavedImport(action) => action.label(),
            Self::Search(query) => format!("search {}", one_line(query)),
            Self::Ingest(path) => path.file_name().map_or_else(
                || path.display().to_string(),
                |name| name.to_string_lossy().into_owned(),
            ),
        }
    }

    /// What the transcript says when it starts.
    fn announcement(&self) -> Message {
        match self {
            Self::Import(request) => Message::new(
                MessageKind::System,
                format!("Importing from {}", request.url),
            ),
            Self::Ingest(path) => {
                Message::new(MessageKind::Upload, format!("Loading {}", path.display()))
            }
            Self::Ontology(_)
            | Self::Graph(_)
            | Self::Embeddings(_)
            | Self::Saved(_)
            | Self::Okf(_)
            | Self::ContextImport(_)
            | Self::ContextExport(_)
            | Self::SavedImport(_)
            | Self::Search(_) => {
                Message::new(MessageKind::System, format!("{}\u{2026}", self.label()))
            }
        }
    }

    /// Run it. Database steps go to the session's workspace writer one at
    /// a time, chunk progress goes to the job strip, and `/cancel` stops
    /// the work that checks for it between batches.
    async fn run(self, env: &JobEnv, ctx: &JobContext) -> Result<String> {
        let progress = |done: ChunkDone| ctx.progress(done.done, done.total);
        let cancel = ctx.cancel_token();
        let control = RunControl {
            progress: &progress,
            cancel: Some(&cancel),
        };
        let mut out: Vec<u8> = Vec::new();
        match self {
            Self::Ontology(action) => {
                ontology_cli::run(
                    &env.config,
                    &env.db,
                    action,
                    Confirm::Assume,
                    &mut out,
                    control,
                )
                .await?;
            }
            Self::Graph(action) => {
                action
                    .run(&env.config, &env.db, Confirm::Assume, &mut out, control)
                    .await?;
            }
            Self::Embeddings(action) => {
                embeddings_cli::run(
                    &env.config,
                    &env.db,
                    action,
                    Confirm::Assume,
                    &mut out,
                    control,
                )
                .await?;
            }
            Self::Saved(action) => {
                saved_cli::run(
                    &env.config,
                    &env.db,
                    action,
                    Some(&env.session_id),
                    None,
                    &mut out,
                )
                .await?;
            }
            Self::Okf(dir) => {
                let name = env.workspace_name.clone();
                let target = dir.clone();
                let summary = env
                    .db
                    .run(move |db| okf::export(db, &name, &mut DirSink::new(Path::new(&target))))
                    .await?;
                return Ok(format!("Wrote {} files to {dir}.", summary.files));
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
            Self::Import(request) => return Self::import(env, &request, control).await,
            Self::SavedImport(action) => {
                let control_plane = ControlPlane::open(&env.config).await?;
                let vault = Vault::new(env.config.data_dir(), KeySource::Keychain);
                action
                    .run(
                        &ImportContext {
                            config: &env.config,
                            workspace: &env.workspace_id,
                            control: &control_plane,
                            vault: &vault,
                            db: &env.db,
                        },
                        &mut out,
                    )
                    .await?;
            }
            Self::Ingest(path) => return Self::ingest(env, &path, control).await,
            Self::Search(query) => return Self::search(env, &query).await,
        }
        Ok(String::from_utf8_lossy(&out).trim_end().to_owned())
    }

    /// `/search QUERY`: the hits with their rank in each leg, then each
    /// leg's candidates and the rerank outcome.
    async fn search(env: &JobEnv, query: &str) -> Result<String> {
        let config = &env.config;
        let search = DocumentSearch::new(query, config.retrieval.top_k)?;
        let embedder = Embeddings::from_config(config).await?;
        let rerank = Rerank::from_config(config).await?;
        let reader = ReaderDb::new(Arc::clone(&env.db));
        let outcome = search
            .run(
                &reader,
                embedder.as_ref(),
                rerank.as_ref(),
                config.retrieval.rrf_k,
            )
            .await?;
        Ok(outcome.render(SearchDetail::Workings).trim_end().to_owned())
    }

    /// Rows from an external source as a workspace table.
    async fn import(
        env: &JobEnv,
        request: &ImportRequest,
        control: RunControl<'_>,
    ) -> Result<String> {
        let embedding_model = Embeddings::from_config(&env.config).await?;
        let summary = import::Importing {
            config: &env.config,
            db: &env.db,
            workspace_id: env.workspace_id.as_str(),
            request,
            policy: ImportPolicy::owner(),
            embedder: embedding_model.as_ref(),
            control,
        }
        .run()
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
    async fn ingest(env: &JobEnv, path: &Path, control: RunControl<'_>) -> Result<String> {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_owned();
        let embedding_model = Embeddings::from_config(&env.config).await?;
        let outcome = ingestion::ingest_file(
            &env.config,
            &env.db,
            env.workspace_id.as_str(),
            &NewFile::at_path(&filename, path).control(control),
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
        let pages = result
            .pages
            .and_then(PageCounts::note)
            .map_or(String::new(), |note| {
                format!("\n{note}; the rest was kept.")
            });
        let graph = FollowUp {
            db: &env.db,
            config: &env.config,
            embeddings: embedding_model.as_ref(),
        }
        .run(std::slice::from_ref(&result.document_id), control)
        .await
        .map_err(|e| anyhow!("graph follow-up failed: {e}"))?
        .map_or(String::new(), |summary| format!("\n{summary}"));
        Ok(format!(
            "Loaded {} ({}){tables}{chunks}{pages}{graph}\nYou can now ask questions about this data.",
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
        let canceller = QueryCanceller::default();
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
                TableProfile::after_write(&db).await;
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
    /// Where the transcript was when last drawn, to place the mouse on it.
    pub(crate) view: Cell<TranscriptView>,
    pub(crate) selection: Option<Selection>,
    /// A finished selection's text, until the loop copies it.
    to_copy: Option<String>,
    pub(crate) copy_status: Option<CopyStatus>,
    pub(crate) quit: Quit,
    pub(crate) spinner: Spinner,
    pub(crate) workspace_name: String,
    pub(crate) provider_display: String,
    pub(crate) session_id: SessionId,
    /// Decisions owed, oldest first; the front one is on screen.
    prompts: VecDeque<Prompt>,
    /// Agent turns queued or running, in submission order.
    turns: Vec<Turn>,
    /// The work queue every submission goes through.
    jobs: JobQueue,
    job_events: broadcast::Receiver<JobInfo>,
    /// Jobs still queued or running, for the strip above the input.
    pub(crate) active_jobs: Vec<JobInfo>,
    /// The `/jobs` or `/sessions` box, while it is open (and takes the keys).
    pub(crate) picker: Option<Picker>,
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
    /// The tables and columns SQL completion offers, read at startup and
    /// again after anything that can change them.
    sql_schema: Arc<SqlSchema>,
    config: Arc<Config>,
    workspace_id: WorkspaceId,
    db: SharedDb,
    reader_db: ReaderDb,
    /// `--allow-write`: the agent's writes run without asking.
    allow_write: bool,
    /// `/scope`: the documents questions are limited to; every document
    /// when empty.
    pub(crate) scope: DocumentScope,
    /// `/steps`: show tool details whole instead of a preview.
    pub(crate) expand_steps: bool,
    /// Each message's wrapped lines, by index, with the fingerprint they
    /// were rendered from (`ui::format_messages`); interior mutability
    /// because drawing only borrows the app.
    pub(crate) wrap_cache: RefCell<Vec<Option<Wrapped>>>,
    msg_rx: mpsc::UnboundedReceiver<AppMsg>,
    msg_tx: mpsc::UnboundedSender<AppMsg>,
}

/// Whether the session is ending.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Quit {
    Stay,
    /// Ctrl+C was pressed once while jobs were running; a second quits,
    /// any other key disarms it.
    Armed,
    Now,
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
            writes,
        } = setup;
        let (msg_tx, msg_rx) = mpsc::unbounded_channel();
        let jobs = JobQueue::from_config(&config.jobs);
        let job_events = jobs.subscribe();
        let mut app = Self {
            messages: Vec::new(),
            textarea: TextArea::default(),
            scroll: Scroll::Latest,
            scroll_limit: Cell::new(0),
            view: Cell::new(TranscriptView::default()),
            selection: None,
            to_copy: None,
            copy_status: None,
            quit: Quit::Stay,
            spinner: Spinner::default(),
            workspace_name,
            provider_display: config.chat_model_label(),
            session_id,
            prompts: VecDeque::new(),
            turns: Vec::new(),
            jobs,
            job_events,
            active_jobs: Vec::new(),
            picker: None,
            db_steps: None,
            pending_db: 0,
            switching: None,
            last_sql: None,
            history: InputHistory::default(),
            popup: Popup::Open { selected: 0 },
            sql_schema: Arc::new(SqlSchema::default()),
            config: Arc::new(config),
            workspace_id,
            db,
            reader_db,
            allow_write: writes.allows_unasked(),
            scope: DocumentScope::default(),
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

    /// Load the workspace's input history at startup, and delete the data
    /// directory's `terminal_history` that earlier releases kept for every
    /// workspace outside any workspace file.
    pub(crate) async fn load_input_history(&mut self) -> Result<()> {
        self.history.lines = self.reader_db.with_db(input_history::recent).await?;
        let legacy = self.config.data_dir().join("terminal_history");
        match std::fs::remove_file(&legacy) {
            Ok(()) => tracing::info!(path = %legacy.display(), "deleted the old input history"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => self.note(
                MessageKind::Error,
                format!("could not delete {}: {e}", legacy.display()),
            ),
        }
        Ok(())
    }

    /// Read the tables and columns SQL completion offers, at startup.
    pub(crate) async fn load_sql_schema(&mut self) -> Result<()> {
        self.sql_schema = Arc::new(self.reader_db.with_db(WorkspaceDb::sql_schema).await?);
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
    /// What the turn running as `job` is doing, if it is a turn.
    pub(crate) fn phase_of(&self, job: &JobInfo) -> Option<Phase> {
        self.turns
            .iter()
            .find(|turn| turn.job.id == job.id)
            .map(|turn| turn.phase)
    }

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
            match row.role {
                MessageRole::User => self.note(MessageKind::User, row.content),
                MessageRole::Assistant => {
                    let meta = row.assistant().cloned().unwrap_or_default();
                    let mut message = Message::new(MessageKind::Assistant, row.content);
                    if let Some(spec) = &meta.chart {
                        message.chart = Some(ChartData::from_spec(spec));
                    }
                    self.post(message);
                    if !meta.citations.is_empty() {
                        self.post(Message::sources(&meta.citations));
                    }
                }
                // The stored row is the summary; the tool, its detail (the
                // SQL, the search text), and its time sit in its metadata.
                MessageRole::Tool => match row.tool() {
                    Some(meta) => {
                        let step = meta.step(row.content.clone());
                        self.post(Message::from(&step));
                    }
                    None => self.post(Message::new(
                        MessageKind::Step,
                        format!("> tool\n  {}", row.content),
                    )),
                },
            }
        }
    }

    /// The event loop: one `select!` over the terminal's input stream, the
    /// session's message channel, the job queue's broadcast, and (only
    /// while a job is active) the spinner's tick. Every message already
    /// waiting is applied before the next draw, so a burst of streamed text
    /// costs one redraw, not one per delta.
    pub(crate) async fn run(self, terminal: &mut ratatui::DefaultTerminal) -> Result<()> {
        let clipboard = Clipboard::new(std::io::stdout());
        self.run_with(terminal, EventStream::new(), clipboard).await
    }

    /// [`Self::run`] over any backend, input stream, and clipboard, so tests
    /// drive the real loop with scripted keys and a `TestBackend`.
    async fn run_with<B, S, W>(
        mut self,
        terminal: &mut ratatui::Terminal<B>,
        input: S,
        mut clipboard: Clipboard<W>,
    ) -> Result<()>
    where
        B: ratatui::backend::Backend,
        B::Error: std::error::Error + Send + Sync + 'static,
        S: futures::Stream<Item = std::io::Result<Event>> + Unpin,
        W: std::io::Write,
    {
        use futures::StreamExt as _;

        let mut input = input;
        let mut spinner = tokio::time::interval(Duration::from_millis(SPINNER_MS));
        spinner.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut dirty = true;

        while self.quit != Quit::Now {
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
            if let Some(text) = self.to_copy.take() {
                self.copy_status = Some(clipboard.copy(&text));
            }
            if dirty {
                self.pump();
            }
        }

        self.stop_jobs().await;
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
                self.clear_selection();
                self.handle_key_event(key.code, key.modifiers);
                true
            }
            Event::Paste(text) => {
                self.clear_selection();
                self.handle_paste(text);
                true
            }
            Event::Mouse(mouse) => self.handle_mouse(*mouse),
            Event::Resize(..) => {
                // The transcript wraps again, so its lines are other lines.
                self.clear_selection();
                true
            }
            _ => false,
        }
    }

    /// The wheel scrolls; a drag with the left button selects transcript
    /// text, and letting go copies it. Returns whether to redraw.
    fn handle_mouse(&mut self, mouse: MouseEvent) -> bool {
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                match &mut self.picker {
                    Some(picker) => picker.up(1),
                    None => self.scroll_up(3),
                }
                true
            }
            MouseEventKind::ScrollDown => {
                match &mut self.picker {
                    Some(picker) => picker.down(1),
                    None => self.scroll_down(3),
                }
                true
            }
            MouseEventKind::Down(MouseButton::Left) => {
                self.clear_selection();
                if self.picker.is_none()
                    && !self.awaiting_permission()
                    && let Some(Located {
                        position,
                        edge: Edge::Inside,
                    }) = self.view.get().locate(mouse.column, mouse.row)
                {
                    self.selection = Some(Selection::at(position));
                }
                true
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                let Some(selection) = self.selection.as_mut().filter(|s| s.is_dragging()) else {
                    return false;
                };
                let Some(located) = self.view.get().locate(mouse.column, mouse.row) else {
                    return false;
                };
                selection.extend(located.position);
                match located.edge {
                    Edge::Above => self.scroll_up(1),
                    Edge::Below => self.scroll_down(1),
                    Edge::Inside => {}
                }
                true
            }
            MouseEventKind::Up(MouseButton::Left) => self.finish_selection(),
            MouseEventKind::Down(MouseButton::Right | MouseButton::Middle)
            | MouseEventKind::Drag(MouseButton::Right | MouseButton::Middle)
            | MouseEventKind::Up(MouseButton::Right | MouseButton::Middle)
            | MouseEventKind::Moved
            | MouseEventKind::ScrollLeft
            | MouseEventKind::ScrollRight => false,
        }
    }

    /// The button came up: keep what was dragged over highlighted and
    /// queue its text for the clipboard. Returns whether to redraw.
    fn finish_selection(&mut self) -> bool {
        let Some(mut selection) = self.selection.filter(Selection::is_dragging) else {
            return false;
        };
        self.selection = None;
        let width = usize::from(self.view.get().area.width);
        let text = selection.text(&ui::format_messages(self, width));
        if !text.is_empty() {
            selection.finish();
            self.selection = Some(selection);
            self.to_copy = Some(text);
        }
        true
    }

    fn clear_selection(&mut self) {
        self.selection = None;
        self.to_copy = None;
        self.copy_status = None;
    }

    /// Pasted text. A terminal pastes the paths of files dropped on it, so
    /// a paste into an empty input that names only loadable files loads
    /// them at once; any other is typed in.
    fn handle_paste(&mut self, text: &str) {
        if self.awaiting_permission() || self.picker.is_some() {
            return;
        }
        // Terminals paste a line break as a bare carriage return; normalize
        // it once, up front, so the file-loading and typed-in branches see
        // the same input (shlex split on '\n', not '\r').
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        if self.textarea.is_empty()
            && let Some(FileLine::Files(paths)) = FileLine::of(&text)
        {
            self.scroll = Scroll::Latest;
            self.load_files(FileLine::Files(paths));
            return;
        }
        self.textarea.insert_str(&text);
        self.reset_completion();
        self.history.leave();
    }

    fn load_files(&mut self, files: FileLine) {
        match files {
            FileLine::Files(paths) => {
                for path in paths {
                    self.run_job(CliJob::Ingest(path));
                }
            }
            FileLine::Comment => self.note(MessageKind::Error, FileLine::COMMENT),
        }
    }

    fn scroll_up(&mut self, lines: usize) {
        self.scroll = self.scroll.up(lines);
    }

    fn scroll_down(&mut self, lines: usize) {
        self.scroll = self.scroll.down(lines, self.scroll_limit.get());
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
                self.prompts.retain(|prompt| !prompt.is_write_of(job));
                if let Some(turn) = self.turns.iter_mut().find(|t| t.job.id == job) {
                    turn.progress = turn.progress.closed();
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

    pub(crate) fn pending_prompt(&self) -> Option<PendingPrompt<'_>> {
        let waiting = self.prompts.len().saturating_sub(1);
        Some(match self.prompts.front()? {
            Prompt::Write { job, request } => {
                let speaker = self
                    .turns
                    .iter()
                    .find(|turn| turn.job.id == job.id)
                    .map_or_else(
                        || String::from("The agent"),
                        |turn| turn.speaker(&self.session_id),
                    );
                PendingPrompt {
                    heading: format!("{speaker}: {}", Decision::HEADING),
                    body: &request.sql,
                    notice: request.hold.notice(),
                    keys: answer_keys(),
                    waiting,
                }
            }
            Prompt::Delete(deletion) => PendingPrompt {
                heading: deletion.question(),
                body: "",
                notice: None,
                keys: String::from(CONFIRM_KEYS),
                waiting,
            },
        })
    }

    /// A job changed state, or snapshots were missed (`Lagged`): rebuild
    /// the strip from the queue itself, and settle turns waiting on their
    /// job's end.
    fn handle_job_event(
        &mut self,
        _event: &std::result::Result<JobInfo, broadcast::error::RecvError>,
    ) {
        let jobs = self.jobs.list();
        self.active_jobs = jobs
            .iter()
            .filter(|j| !j.state.is_finished())
            .cloned()
            .collect();
        if let Some(picker) = &mut self.picker {
            picker.follow_jobs(jobs);
        }
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
            match turn.progress {
                TurnProgress::Streaming | TurnProgress::Ended => return true,
                TurnProgress::Done => return false,
                TurnProgress::Closed => {}
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
        turn.phase = turn.phase.after(&event, Timestamp::now());
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
                self.post(Message::step_started(tool, detail));
                turn.open_step = Some(self.messages.len().saturating_sub(1));
            }
            AgentEvent::ToolFinished(step) if visible => {
                if let Some(idx) = turn.open_step.take()
                    && let Some(msg) = self.messages.get_mut(idx)
                {
                    let line = format!("\n  {}, {} ms", step.summary, step.duration_ms);
                    msg.content.push_str(&line);
                    msg.result = step.result;
                } else {
                    self.post(Message::from(&step));
                }
            }
            AgentEvent::Status(_)
            | AgentEvent::Reasoning
            | AgentEvent::TextDelta(_)
            | AgentEvent::ToolStarted { .. }
            | AgentEvent::ToolFinished(_) => {}
            AgentEvent::PermissionRequired(request) => {
                self.ask_for_turn(turn, request);
            }
            AgentEvent::TurnComplete(response) if visible => {
                turn.progress = turn.progress.ended();
                self.refresh_sql_schema();
                self.handle_turn_complete(turn, response);
            }
            AgentEvent::TurnComplete(response) => {
                turn.progress = turn.progress.ended();
                self.refresh_sql_schema();
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
                turn.progress = turn.progress.ended();
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
    /// screen says whose it is, and one held for more than being a write
    /// says why.
    fn ask_for_turn(&mut self, turn: &Turn, request: PermissionRequest) {
        let notice = request
            .hold
            .notice()
            .map_or(String::new(), |notice| format!("{notice}\n"));
        self.note(
            MessageKind::System,
            format!(
                "{}: {}\n{}\n{notice}{}",
                turn.speaker(&self.session_id),
                Decision::HEADING,
                request.sql,
                answer_keys()
            ),
        );
        self.prompts.push_back(Prompt::Write {
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
        if let Some(spec) = &response.chart
            && let Some(last) = self
                .messages
                .iter_mut()
                .rev()
                .find(|m| m.kind == MessageKind::Assistant)
        {
            last.chart = Some(ChartData::from_spec(spec));
        }
        for result in response.graph.iter().filter(|r| !r.is_empty()) {
            self.note(MessageKind::System, result.to_string());
        }
        if response.write_refused && !self.allow_write {
            self.note(
                MessageKind::System,
                "A write was refused this turn. Answer y next time, or restart with --allow-write.",
            );
        }
        turn.streaming = None;
        turn.open_step = None;
        self.scroll = Scroll::Latest;
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
            if prompt.is_write_of(job.id) {
                prompt.refuse();
            } else {
                kept.push_back(prompt);
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

    /// `/jobs`: every job on record in a box to move through, which
    /// follows the queue while it is open.
    fn show_jobs(&mut self) {
        let jobs = self.jobs.list();
        if jobs.is_empty() {
            self.note(MessageKind::System, "No jobs yet.");
            return;
        }
        self.picker = Some(Picker::jobs(jobs));
    }

    /// A key while the `/jobs` or `/sessions` box is open: move through
    /// it, act on the highlighted row, or close it.
    fn handle_picker_key(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        let Some(picker) = &mut self.picker else {
            return;
        };
        match (code, modifiers) {
            (KeyCode::Esc | KeyCode::Char('q'), _)
            | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                self.picker = None;
            }
            (KeyCode::Up, _) => picker.up(1),
            (KeyCode::Down, _) => picker.down(1),
            (KeyCode::PageUp, _) => picker.up(10),
            (KeyCode::PageDown, _) => picker.down(10),
            (KeyCode::Home, _) => picker.first(),
            (KeyCode::End, _) => picker.last(),
            (KeyCode::Char('c'), KeyModifiers::NONE) => {
                if let Some(Picked::Job(job)) = picker.picked() {
                    let number = job.number;
                    self.cancel_job(number);
                }
            }
            (KeyCode::Char('d'), KeyModifiers::NONE) => {
                if let Some(Picked::Session(session)) = picker.picked() {
                    let deletion = Deletion::Session {
                        id: session.id.clone(),
                        title: session
                            .title
                            .clone()
                            .unwrap_or_else(|| String::from("(untitled)")),
                    };
                    self.picker = None;
                    self.prompts.push_back(Prompt::Delete(deletion));
                }
            }
            (KeyCode::Enter, _) => match picker.picked() {
                Some(Picked::Job(job)) => {
                    let details = JobRow(job).details();
                    self.picker = None;
                    self.note(MessageKind::System, details);
                }
                Some(Picked::Session(session)) => {
                    let id = session.id.to_string();
                    self.picker = None;
                    self.switch_session(id);
                }
                None => self.picker = None,
            },
            _ => {}
        }
    }

    /// Stop every job when the session ends and give them `QUIT_GRACE` to
    /// do it: a turn records its cancellation, a statement is interrupted,
    /// an ingest drops its embedding requests. Whatever is still running
    /// after that is dropped with the runtime at its next await.
    async fn stop_jobs(&mut self) {
        for prompt in self.prompts.drain(..) {
            prompt.refuse();
        }
        let left = self.jobs.shutdown(QUIT_GRACE).await;
        if !left.is_empty() {
            tracing::warn!(jobs = left.len(), "quit with jobs still running");
        }
        self.pump();
    }

    /// Clear the transcript; streaming turns start a new message.
    fn clear_transcript(&mut self) {
        self.messages.clear();
        self.wrap_cache.borrow_mut().clear();
        self.scroll = Scroll::Latest;
        for turn in &mut self.turns {
            turn.streaming = None;
            turn.open_step = None;
        }
    }

    fn handle_key_event(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        let ctrl_c = (code, modifiers) == (KeyCode::Char('c'), KeyModifiers::CONTROL);
        if !ctrl_c && self.quit == Quit::Armed {
            self.quit = Quit::Stay;
        }
        if self.picker.is_some() && !self.awaiting_permission() {
            self.handle_picker_key(code, modifiers);
            return;
        }
        if ctrl_c {
            // The prompt on screen first, then this session's newest turn.
            match self.prompts.front() {
                Some(Prompt::Write { job, .. }) => {
                    let job = *job;
                    self.cancel_turn(job);
                    return;
                }
                Some(Prompt::Delete(_)) => {
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
                if running > 0 && self.quit == Quit::Stay {
                    self.quit = Quit::Armed;
                    self.note(
                        MessageKind::System,
                        format!(
                            "{running} job{} still running (/jobs). Press Ctrl+C again to quit and stop {}.",
                            if running == 1 { " is" } else { "s are" },
                            if running == 1 { "it" } else { "them" }
                        ),
                    );
                } else {
                    self.quit = Quit::Now;
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

    /// The answer a key gives to the front prompt, if it is one: `y`,
    /// `n` (or Esc), and for a write `a`.
    fn handle_permission_key(&mut self, code: KeyCode) {
        let answer = match code {
            KeyCode::Esc => Decision::Deny,
            KeyCode::Char(key) => {
                let Some(answer) = Decision::CHOICES
                    .into_iter()
                    .find(|decision| answer_key(*decision) == key.to_ascii_lowercase())
                else {
                    return;
                };
                answer
            }
            _ => return,
        };
        if answer == Decision::AllowTurn && matches!(self.prompts.front(), Some(Prompt::Delete(_)))
        {
            return;
        }
        let Some(prompt) = self.prompts.pop_front() else {
            return;
        };
        match prompt {
            Prompt::Write { request, .. } => {
                let reply = match request.answer(answer) {
                    Delivery::Delivered => answer.reply(),
                    Delivery::TurnGone => Delivery::TURN_GONE,
                };
                self.note(MessageKind::System, reply);
            }
            Prompt::Delete(deletion) => match answer {
                Decision::Allow => self.delete(deletion),
                Decision::Deny | Decision::AllowTurn => self.note(MessageKind::System, "Kept."),
            },
        }
    }

    /// What the popup offers for the input, if it is showing: one line
    /// not recalled from history and not hidden with Esc; a `/` command
    /// with the cursor at its end, or a SQL statement with the cursor
    /// anywhere in it.
    pub(crate) fn completion(&self) -> Option<Completion> {
        if self.popup == Popup::Hidden || self.history.browsing() || self.awaiting_permission() {
            return None;
        }
        let [line] = self.textarea.lines() else {
            return None;
        };
        let cursor = self.textarea.cursor().1;
        if line.starts_with('/') {
            return (cursor == line.chars().count())
                .then(|| Completion::for_line(line))
                .flatten();
        }
        Completion::for_sql(line, cursor, &self.sql_schema)
    }

    /// Read the tables and columns again, on the reader, after a statement, an ingest, an import, or an agent turn, any of
    /// which may have changed them. A failed read keeps the last schema.
    fn refresh_sql_schema(&mut self) {
        self.on_db(
            Side::Read,
            WorkspaceDb::sql_schema,
            |app, read| match read {
                Ok(schema) => app.sql_schema = Arc::new(schema),
                Err(e) => tracing::debug!(error = %e, "could not read the schema for completion"),
            },
        );
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
                let (filled, cursor) = completion.apply(&line, item);
                if code == KeyCode::Enter && filled.trim_end() == line.trim_end() {
                    return false;
                }
                let send = code == KeyCode::Enter && item.finishes();
                self.set_input(&filled);
                self.textarea.move_cursor(CursorMove::Jump(
                    0,
                    u16::try_from(cursor).unwrap_or(u16::MAX),
                ));
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
    /// `input` as a slash command, or `None` once the transcript says why
    /// it is not one (or shows the help it asked for).
    fn parse_slash_command(&mut self, input: &str) -> Option<SlashCommand> {
        let e = match SlashCommand::parse(input) {
            Ok(command) => return Some(command),
            Err(e) => e,
        };
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
        None
    }

    fn handle_slash_command(&mut self, input: &str) {
        let Some(command) = self.parse_slash_command(input) else {
            return;
        };
        match command {
            SlashCommand::Quit => self.quit = Quit::Now,
            SlashCommand::Clear => {
                self.clear_transcript();
                self.note(MessageKind::System, WELCOME_TEXT);
            }
            SlashCommand::Jobs => self.show_jobs(),
            SlashCommand::Cancel { job } => self.cancel_job(job),
            SlashCommand::Help => self.note(MessageKind::System, SlashCommand::help()),
            SlashCommand::Workspace => self.show_workspace(),
            SlashCommand::Sessions { query: None } => self.show_sessions(),
            SlashCommand::Sessions { query: Some(text) } => self.show_matching_sessions(text),
            SlashCommand::Rename { title } => self.rename_session(title.unwrap_or_default()),
            SlashCommand::Resume { id } => self.switch_session(id),
            SlashCommand::New => self.new_session(),
            SlashCommand::Mode { mode: None } => self.show_mode(),
            SlashCommand::Mode { mode: Some(mode) } => self.set_mode(mode),
            SlashCommand::Docs => self.show_documents(),
            SlashCommand::Search { query } => self.run_job(CliJob::Search(query)),
            SlashCommand::Context { action: None } => self.show_context(),
            SlashCommand::Context {
                action: Some(ContextAction::Import { file }),
            } => self.run_job(CliJob::ContextImport(file)),
            SlashCommand::Context {
                action: Some(ContextAction::Export { file }),
            } => self.run_job(CliJob::ContextExport(file)),
            SlashCommand::Pin { id } => self.set_pinned(id, Pinning::Pinned),
            SlashCommand::Unpin { id } => self.set_pinned(id, Pinning::Unpinned),
            SlashCommand::Tables(args) => self.tables(args),
            SlashCommand::Scope { documents } => self.set_scope(documents),
            SlashCommand::Ingest { path } => match FileLine::of(&path) {
                Some(files) => self.load_files(files),
                None => self.note(
                    MessageKind::Error,
                    format!("'{path}' is not a file quack can ingest"),
                ),
            },
            SlashCommand::Graph {
                action: Some(action),
                ..
            } => self.run_job(CliJob::Graph(action)),
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
            SlashCommand::Ontology { action } => self.run_job(CliJob::Ontology(action)),
            SlashCommand::Delete { id } => self.delete_document(id),
            SlashCommand::Import {
                action: Some(action),
                ..
            } => self.run_job(CliJob::SavedImport(action)),
            SlashCommand::Import {
                action: None,
                url,
                table,
                source_table,
                query,
                mut headers,
                bearer_env,
                json_pointer,
            } => {
                headers.extend(bearer_env.map(SourceHeader::BearerEnv));
                self.run_job(CliJob::Import(ImportRequest {
                    query,
                    source_table,
                    headers,
                    json_pointer,
                    ..ImportRequest::new(url.unwrap_or_default(), table.unwrap_or_default())
                }));
            }
            SlashCommand::Path { route } => self.show_path(&route),
            SlashCommand::Sql {
                statement: Some(sql),
            } => self.run_direct_sql(sql, SqlIntent::Stated),
            SlashCommand::Sql { statement: None } => self.edit_last_sql(),
            SlashCommand::Share => self.set_sharing(Sharing::Shared),
            SlashCommand::Unshare => self.set_sharing(Sharing::Private),
            SlashCommand::Export { flags, file } => self.export_session(flags.format(), file),
            SlashCommand::Okf { dir } => self.run_job(CliJob::Okf(dir)),
            SlashCommand::Embeddings { action } => self.run_job(CliJob::Embeddings(action)),
            SlashCommand::Saved { action } => {
                let list = SavedAction::List {
                    format: TextOrJson::Text,
                };
                self.run_job(CliJob::Saved(action.unwrap_or(list)));
            }
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
        let config = Arc::clone(&self.config);
        self.submit_work(
            JobKind::Models,
            String::from("list models"),
            Some(Message::new(
                MessageKind::System,
                "Listing each provider's models",
            )),
            move |_ctx| async move {
                BackgroundResult::Done {
                    kind: MessageKind::System,
                    text: llm::ModelCatalog::fetch(&config).await.to_string(),
                }
            },
        );
    }

    /// `/sessions`: the most recent sessions in a box to move through;
    /// Enter resumes the highlighted one.
    fn show_sessions(&mut self) {
        self.on_db_ok(
            Side::Read,
            |db| sessions::list_sessions(db, PICKER_SESSIONS),
            |app, rows| {
                if rows.is_empty() {
                    app.note(MessageKind::System, "No sessions yet.");
                    return;
                }
                app.picker = Some(Picker::sessions(rows, app.session_id.clone()));
            },
        );
    }

    /// `/sessions TEXT`: the picker, holding the sessions whose questions or
    /// answers contain `text`, newest match first.
    fn show_matching_sessions(&mut self, text: String) {
        self.on_db_ok(
            Side::Read,
            move |db| {
                let hits =
                    sessions::search_messages(db, &text, &SessionViewer::All, PICKER_SESSIONS)?;
                let mut rows: Vec<SessionRow> = Vec::new();
                for hit in hits {
                    if rows.iter().all(|row| row.id != hit.session_id)
                        && let Some(row) = sessions::get_session(db, &hit.session_id)?
                    {
                        rows.push(row);
                    }
                }
                Ok(rows)
            },
            |app, rows| {
                if rows.is_empty() {
                    app.note(MessageKind::System, "No session mentions that.");
                    return;
                }
                app.picker = Some(Picker::sessions(rows, app.session_id.clone()));
            },
        );
    }

    /// `/rename [TITLE]`: a new title, or with none the derived one again.
    fn rename_session(&mut self, title: String) {
        let session = self.session_id.clone();
        self.on_db_ok(
            Side::Write,
            move |db| sessions::set_session_title(db, &session, &title),
            |app, renamed| {
                let title = renamed.title.unwrap_or_default();
                app.note(
                    MessageKind::System,
                    match renamed.title_by {
                        TitleSource::Person => format!("Renamed to \"{title}\"."),
                        TitleSource::Derived | TitleSource::Model => {
                            format!("Named after its first question again: \"{title}\".")
                        }
                    },
                );
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
                let by_id = PrefixMatch::of(sessions.clone(), &prefix, |s| s.id.as_str());
                let by_title = || {
                    let wanted = prefix.to_lowercase();
                    let titled: Vec<SessionRow> = sessions
                        .into_iter()
                        .filter(|s| {
                            s.title
                                .as_deref()
                                .is_some_and(|t| t.to_lowercase().starts_with(&wanted))
                        })
                        .collect();
                    match titled.len() {
                        0 => PrefixMatch::None,
                        1 => titled
                            .into_iter()
                            .next()
                            .map_or(PrefixMatch::None, PrefixMatch::One),
                        _ => PrefixMatch::Many(titled),
                    }
                };
                let matched = match by_id {
                    PrefixMatch::None => by_title(),
                    found => found,
                };
                let found = match matched {
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
                            text.push_str(id.as_str());
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
            session_id: self.session_id.clone(),
        };
        let announcement = job.announcement();
        self.submit_work(
            job.kind(),
            job.label(),
            Some(announcement),
            move |ctx| async move { BackgroundResult::from(job.run(&env, &ctx).await) },
        );
    }

    /// `/tables`: what `quack tables` prints, on the writer when it sets a
    /// note or retypes a column, else on the reader.
    fn tables(&mut self, args: TablesArgs) {
        let writes = args.writes();
        let side = if writes { Side::Write } else { Side::Read };
        self.on_db_ok(
            side,
            move |db| {
                let mut out = Vec::new();
                Ok(args
                    .run(db, &mut out)
                    .map(|()| String::from_utf8_lossy(&out).trim_end().to_owned()))
            },
            move |app, printed| match printed {
                Ok(text) => {
                    app.note(MessageKind::Sql, text);
                    if writes {
                        app.refresh_sql_schema();
                    }
                }
                Err(e) => app.note(MessageKind::Error, format!("{e:#}")),
            },
        );
    }

    /// `/scope`: resolve the names now, so a typo is reported here rather
    /// than by the next question; no names is every document.
    fn set_scope(&mut self, names: Vec<String>) {
        self.on_db_ok(
            Side::Read,
            move |db| DocumentScope::resolve(db, &names),
            |app, scope| {
                let text = if scope.is_everything() {
                    String::from("Questions ask about every document.")
                } else {
                    let names: Vec<String> = scope
                        .documents()
                        .iter()
                        .map(|d| OneLine(&d.filename).to_string())
                        .collect();
                    format!(
                        "Questions ask about {} only, until /scope with no names.",
                        names.join(", ")
                    )
                };
                app.scope = scope;
                app.note(MessageKind::System, text);
            },
        );
    }

    /// `/delete`: find the document, then ask before deleting it.
    fn delete_document(&mut self, prefix: String) {
        self.on_db_ok(
            Side::Read,
            move |db| db.document_by_id_prefix(&prefix),
            |app, doc| {
                app.prompts.push_back(Prompt::Delete(Deletion::Document {
                    id: doc.id,
                    filename: doc.filename,
                }));
            },
        );
    }

    /// Delete what the person confirmed. Deleting the session on screen
    /// starts a new one, as the web lands on a fresh chat.
    fn delete(&mut self, deletion: Deletion) {
        match deletion {
            Deletion::Document { id, filename } => self.on_db_ok(
                Side::Write,
                move |db| db.delete_document(&id),
                move |app, _| {
                    app.note(
                        MessageKind::System,
                        format!("Deleted {filename} with its chunks, tables, and graph rows."),
                    );
                },
            ),
            Deletion::Session { id, title } => {
                let current = id == self.session_id;
                self.on_db_ok(
                    Side::Write,
                    move |db| sessions::delete_session(db, &id),
                    move |app, _| {
                        app.note(
                            MessageKind::System,
                            format!("Deleted the session '{title}'."),
                        );
                        if current {
                            app.new_session();
                        }
                    },
                );
            }
        }
    }

    fn set_sharing(&mut self, sharing: Sharing) {
        let session = self.session_id.clone();
        self.on_db_ok(
            Side::Write,
            move |db| sessions::set_session_sharing(db, &session, sharing),
            move |app, ()| {
                app.note(
                    MessageKind::System,
                    match sharing {
                        Sharing::Shared => {
                            "This session is shared with every member of the workspace."
                        }
                        Sharing::Private => "This session is yours alone again.",
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
        let options = self.config.graph;
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
        let options = self.config.graph;
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
        let read = |db: &WorkspaceDb| db.documents(&DocumentListing::default());
        self.on_db_ok(Side::Read, read, |app, page| {
            if page.documents.is_empty() {
                app.note(MessageKind::System, "No documents yet.");
                return;
            }
            let mut text = String::from("Documents:");
            for doc in &page.documents {
                let pages = doc
                    .pages
                    .and_then(PageCounts::note)
                    .map_or(String::new(), |note| format!("  [{note}]"));
                let line = format!(
                    "\n  {}  {:<10}  {}  {}{pages}",
                    doc.id.short(),
                    doc.status,
                    if doc.pinning == Pinning::Pinned {
                        "pinned"
                    } else {
                        "      "
                    },
                    doc.filename
                );
                text.push_str(&line);
            }
            if page.next.is_some() {
                let more = format!(
                    "\nThe newest {} of {}; `quack docs` lists them all.",
                    page.documents.len(),
                    page.total
                );
                text.push_str(&more);
            }
            text.push_str("\nUse /pin ID or /unpin ID.");
            app.note(MessageKind::System, text);
        });
    }

    fn set_pinned(&mut self, prefix: String, pinning: Pinning) {
        self.on_db_ok(
            Side::Write,
            move |db| {
                let doc = db.document_by_id_prefix(&prefix)?;
                db.set_document_pinning(&doc.id, pinning).map(|()| doc.id)
            },
            move |app, id| {
                let done = match pinning {
                    Pinning::Pinned => "Pinned",
                    Pinning::Unpinned => "Unpinned",
                };
                app.note(MessageKind::System, format!("{done} {}", id.short()));
            },
        );
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
    fn session_to_forget(&self) -> Option<(SharedDb, SessionId)> {
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
        A: FnOnce(&mut Self, CoreResult<T>) + Send + 'static,
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
                drop(tx.send(AppMsg::Apply(Box::new(move |app: &mut Self| {
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
        A: FnOnce(&mut Self, T) + Send + 'static,
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
        let line = trimmed.clone();
        self.on_db(
            Side::Write,
            move |db| input_history::push(db, &line),
            |_, result| {
                if let Err(e) = result {
                    tracing::warn!(error = %e, "could not save the input history");
                }
            },
        );
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
            Input::Files(paths) => self.load_files(paths),
            Input::Sql(sql) => self.run_direct_sql(sql, SqlIntent::Guessed),
            Input::Question(question) if self.config.general.chat_model.is_none() => {
                self.note(MessageKind::User, question);
                self.note(MessageKind::System, NO_CHAT_MODEL_TEXT);
            }
            Input::Question(question) => self.start_agent_turn(question),
        }
    }

    /// `/sql`, or a line that starts like a statement: the same gate the
    /// agent's statements pass. Internal tables are refused, and an invalid
    /// statement is reported (or asked as a question, when it was only
    /// guessed to be SQL).
    fn run_direct_sql(&mut self, sql: String, intent: SqlIntent) {
        self.note(MessageKind::User, sql.clone());
        // Classifying is a parse: the reader pool does it, off the loop.
        let statement = sql.clone();
        self.on_db(
            Side::Read,
            move |db| db.classify_user_statement(&statement),
            move |app, kind| app.gate_direct_sql(sql, kind, intent),
        );
    }

    /// Run a classified statement.
    fn gate_direct_sql(&mut self, sql: String, kind: CoreResult<StatementKind>, intent: SqlIntent) {
        // DuckDB could not parse it, so a question that happens to start
        // with a keyword goes to the model after all.
        if let Ok(StatementKind::Invalid(message)) = &kind
            && intent == SqlIntent::Guessed
            && self.config.general.chat_model.is_some()
        {
            self.note(
                MessageKind::System,
                format!("Not SQL ({message}), so asked as a question. /sql runs a line as typed."),
            );
            self.submit_turn(sql);
            return;
        }
        self.last_sql = Some(sql.clone());
        match kind {
            Ok(StatementKind::Read) => self.execute_direct_sql(sql, Side::Read),
            // A person's own statement runs as typed, as the web SQL page
            // runs it; only the agent's writes ask.
            Ok(StatementKind::Write) => self.execute_direct_sql(sql, Side::Write),
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

    /// Submit a question as a job in its session's lane: it starts once
    /// the session's previous turn has finished (its history includes that
    /// answer) and a worker is free, and streams into the transcript while
    /// everything else stays usable.
    fn start_agent_turn(&mut self, message: String) {
        self.note(MessageKind::User, message.clone());
        self.submit_turn(message);
    }

    /// [`Self::start_agent_turn`] for a message already in the transcript.
    fn submit_turn(&mut self, message: String) {
        let behind = self.current_turn();

        let (sink, rx) = events::channel();
        let config = Arc::clone(&self.config);
        let db = Arc::clone(&self.db);
        let reader_db = self.reader_db.clone();
        let session_id = self.session_id.clone();
        let policy = WritePolicy::Ask.allowed_if(self.allow_write);
        // By id: the turn resolves them again when it starts.
        let documents: Vec<String> = self
            .scope
            .documents()
            .iter()
            .map(|d| d.id.to_string())
            .collect();
        let spec = JobSpec::new(JobKind::Chat, one_line(&message))
            .workspace(self.workspace_id.clone())
            .lane(Lane::serial(&LaneKey::Session(session_id.clone())));
        let job = Ticket::from(&self.jobs.submit(spec, move |ctx| async move {
            // The turn emits TurnComplete or Failed itself; the returned
            // value is the same response, and the job keeps its outline.
            match (llm::TurnRequest {
                db,
                reader_db,
                session_id: &session_id,
                policy,
                message: &message,
                documents: &documents,
                sink,
                cancel: ctx.cancel_token(),
            })
            .run(&config)
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
            progress: TurnProgress::Streaming,
            phase: Phase::Waiting { since: None },
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
        announce: Option<Message>,
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
        if let Some(message) = announce {
            self.note(
                message.kind,
                format!("{} (job #{})", message.content, job.number),
            );
        }
    }

    fn handle_background_result(&mut self, job: JobId, result: BackgroundResult) {
        if self.jobs.get(job).is_some_and(|info| {
            matches!(info.kind, JobKind::Sql | JobKind::Ingest | JobKind::Import)
        }) {
            self.refresh_sql_schema();
        }
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

#[cfg(test)]
mod tests;
