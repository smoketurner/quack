use std::collections::VecDeque;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use clap::error::ErrorKind;
use crossterm::event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind};
use ratatui_textarea::TextArea;
use tokio::sync::{broadcast, mpsc};

use quack_core::analysis::agent::AgentResponse;
use quack_core::analysis::events::{self, AgentEvent, PermissionRequest};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::config::Config;
use quack_core::ingestion::{self, IngestOutcome, NewFile};
use quack_core::storage::sessions::{self, ChatMode, MessageRole as StoredRole};
use quack_core::storage::workspace::{
    QueryCanceller, StatementKind, WorkspaceDb, looks_like_direct_sql,
};

use crate::terminal::chart::ChartData;
use crate::terminal::commands::{Completion, ContextAction, SlashCommand, SlashLine};
use crate::terminal::ui;
use quack_core::analysis::chart::ChartSpec;
use quack_core::analysis::citations::Citation;
use quack_core::error::{Error as CoreError, Result as CoreResult};
use quack_core::graph::traverse;
use quack_core::import::{self, ImportPolicy, ImportRequest};
use quack_core::jobs::{
    JobContext, JobId, JobInfo, JobKind, JobQueue, JobSpec, JobState, Lane, LaneKey,
};
use quack_core::llm::{self, CancellationToken};
use quack_core::okf;
use quack_core::ontology::store as ontology_store;
use quack_core::priority::Priority;
use quack_core::progress::{ChunkDone, RunControl};
use quack_core::storage::context;

use crate::embeddings_cli::{self, EmbeddingsAction};
use crate::graph_cli::GraphAction;
use crate::ontology_cli::OntologyAction;

/// The spinner's frame interval; it ticks only while a job is active.
const SPINNER_MS: u64 = 80;

/// How long quitting waits for cancelled jobs to stop.
const QUIT_GRACE: Duration = Duration::from_secs(3);

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum MessageRole {
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
    pub(crate) role: MessageRole,
    pub(crate) content: String,
    /// The chart an assistant answer produced (design doc 9: charts belong
    /// to messages); `/chart N` brings it into the chart pane.
    pub(crate) chart: Option<ChartData>,
    /// A step's full tool detail, shown whole when steps are expanded.
    pub(crate) detail: Option<String>,
}

impl Message {
    fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            chart: None,
            detail: None,
        }
    }
}

/// Results from work that is not an agent turn.
enum BackgroundResult {
    Ingested { summary: String },
    SqlResult { text: String },
    Error(String),
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

/// Where a command's database step runs.
#[derive(Clone, Copy)]
enum Side {
    /// The reader pool, inside a read-only transaction: never waits on
    /// the writer.
    Read,
    /// The writer, in its interactive line.
    Write,
}

/// A command's database step, run in order by the session's worker.
type DbStep = Box<dyn FnOnce() -> futures::future::BoxFuture<'static, ()> + Send>;

/// A session loaded for the transcript.
struct Replay {
    session_id: String,
    rows: Vec<sessions::MessageRow>,
}

/// What `/resume PREFIX` found.
enum Found {
    Current,
    One(String, Replay),
    None(String),
    Many(String, Vec<String>),
}

/// A decision the user owes. Prompts are modal, answered in order, while
/// every job keeps running.
enum Prompt {
    /// A write the agent wants to make in the turn run by `job`.
    Agent {
        job: JobId,
        request: PermissionRequest,
    },
    /// A typed statement that modifies the workspace.
    Sql(String),
}

/// An agent turn submitted as a job, and where its text goes. Its events
/// arrive as [`AppMsg::Turn`], forwarded from its stream by a task.
struct Turn {
    job: JobId,
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

/// The command popup's state while a slash command is typed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Popup {
    /// Showing whenever the input has suggestions, with this entry
    /// highlighted.
    Open { selected: usize },
    /// Esc hid it; typing shows it again.
    Hidden,
}

pub(crate) struct App {
    pub(crate) messages: Vec<Message>,
    pub(crate) textarea: TextArea<'static>,
    pub(crate) scroll_offset: usize,
    pub(crate) should_quit: bool,
    pub(crate) tick: usize,
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
    input_history: Vec<String>,
    history_cursor: Option<usize>,
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
    pub(crate) wrap_cache: std::cell::RefCell<Vec<Option<ui::WrappedMessage>>>,
    /// Where typed input is kept across sessions.
    history_path: PathBuf,
    msg_rx: mpsc::UnboundedReceiver<AppMsg>,
    msg_tx: mpsc::UnboundedSender<AppMsg>,
}

impl App {
    #[expect(
        clippy::too_many_arguments,
        reason = "one constructor for the terminal session's whole state"
    )]
    pub(crate) fn new(
        workspace_name: String,
        workspace_id: String,
        provider_display: String,
        config: Arc<Config>,
        db: SharedDb,
        reader_db: ReaderDb,
        session_id: String,
        allow_write: bool,
    ) -> Self {
        let (msg_tx, msg_rx) = mpsc::unbounded_channel();
        let mut textarea = TextArea::default();
        configure_textarea(&mut textarea);

        let history_path = config.data_dir().join("terminal_history");
        let jobs = JobQueue::from_config(&config.jobs);
        let job_events = jobs.subscribe();
        let mut app = Self {
            messages: Vec::new(),
            textarea,
            scroll_offset: 0,
            should_quit: false,
            tick: 0,
            workspace_name,
            provider_display,
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
            input_history: Vec::new(),
            history_cursor: None,
            popup: Popup::Open { selected: 0 },
            config,
            workspace_id,
            db,
            reader_db,
            allow_write: Arc::new(AtomicBool::new(allow_write)),
            expand_steps: false,
            wrap_cache: std::cell::RefCell::new(Vec::new()),
            history_path,
            msg_rx,
            msg_tx,
        };
        app.input_history = load_history(&app.history_path);
        app.messages
            .push(Message::new(MessageRole::System, WELCOME_TEXT));
        if app.config.general.chat_model.is_none() {
            app.messages
                .push(Message::new(MessageRole::System, NO_CHAT_MODEL_TEXT));
        }
        app
    }

    /// Load the session's stored messages into the transcript at startup,
    /// before the loop runs (`/resume` loads through the database worker).
    pub(crate) async fn load_current_session(&mut self) -> Result<()> {
        let id = self.session_id.clone();
        let rows = self.db.run(move |db| load_session(db, &id)).await?;
        self.apply_replay(rows);
        Ok(())
    }

    /// Say so at startup when some vectors were made under another
    /// embedding profile or are missing: those chunks are found by keyword
    /// only until `/embeddings refresh` runs.
    pub(crate) async fn note_embedding_status(&mut self) -> Result<()> {
        let status = self.db.run(WorkspaceDb::embedding_status).await?;
        if let Some(note) = status.note() {
            self.messages.push(Message::new(
                MessageRole::System,
                format!("{note} /embeddings refresh updates them in the background."),
            ));
        }
        Ok(())
    }

    /// Put a loaded session's messages in the transcript.
    fn apply_replay(&mut self, replay: Replay) {
        let Replay { session_id, rows } = replay;
        if rows.is_empty() {
            self.messages.push(Message::new(
                MessageRole::System,
                format!("Session {session_id} has no messages yet."),
            ));
            return;
        }
        self.messages.push(Message::new(
            MessageRole::System,
            format!("Resumed session {session_id} ({} messages)", rows.len()),
        ));
        for row in rows {
            match row.role {
                StoredRole::User => self
                    .messages
                    .push(Message::new(MessageRole::User, row.content)),
                StoredRole::Assistant => {
                    let mut message = Message::new(MessageRole::Assistant, row.content);
                    if let Some(spec) = row
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("chart"))
                        .and_then(|c| serde_json::from_value::<ChartSpec>(c.clone()).ok())
                    {
                        let chart = ChartData::from_spec(&spec);
                        self.current_chart = Some(chart.clone());
                        message.chart = Some(chart);
                    }
                    self.messages.push(message);
                    if let Some(citations) = row
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("citations"))
                        .and_then(|c| serde_json::from_value::<Vec<Citation>>(c.clone()).ok())
                        && !citations.is_empty()
                    {
                        self.messages.push(Message::new(
                            MessageRole::System,
                            sources_footer(&citations),
                        ));
                    }
                }
                StoredRole::Tool => {
                    let tool = row
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("tool"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("tool");
                    let ms = row
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("duration_ms"))
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0);
                    // The stored row is the summary; the detail (the SQL,
                    // the search text) sits in its metadata.
                    let detail = row
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("detail"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_owned();
                    self.messages
                        .push(step_message(tool, &detail, &row.content, ms));
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
                    self.tick = self.tick.wrapping_add(1);
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
                    self.scroll_offset = self.scroll_offset.saturating_add(3);
                    true
                }
                MouseEventKind::ScrollDown => {
                    self.scroll_offset = self.scroll_offset.saturating_sub(3);
                    true
                }
                _ => false,
            },
            Event::Resize(..) => true,
            _ => false,
        }
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
                let Some(at) = self.turns.iter().position(|t| t.job == job) else {
                    return;
                };
                let mut turn = self.turns.remove(at);
                self.handle_turn_event(&mut turn, *event);
                self.turns.insert(at, turn);
            }
            AppMsg::TurnClosed(job) => {
                // Nothing will answer its prompts now.
                self.prompts
                    .retain(|p| !matches!(p, Prompt::Agent { job: owner, .. } if *owner == job));
                if let Some(turn) = self.turns.iter_mut().find(|t| t.job == job) {
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
            match self.jobs.get(turn.job) {
                Some(job) if !job.state.is_finished() => true,
                Some(job) => {
                    let outcome = job.outcome.unwrap_or_default();
                    let (role, text) = if job.state == JobState::Failed {
                        (MessageRole::Error, outcome)
                    } else {
                        (
                            MessageRole::System,
                            format!("Job #{} {}: {outcome}.", job.number, job.state),
                        )
                    };
                    self.messages.push(Message::new(role, text));
                    false
                }
                None => false,
            }
        });
        turns.append(&mut self.turns);
        self.turns = turns;
    }

    /// The job number people see for `job`.
    fn job_number(&self, job: JobId) -> String {
        self.jobs
            .get(job)
            .map_or_else(|| String::from("?"), |j| j.number.to_string())
    }

    fn handle_turn_event(&mut self, turn: &mut Turn, event: AgentEvent) {
        let visible = turn.session_id == self.session_id;
        match event {
            AgentEvent::Status(status) if visible => {
                self.messages
                    .push(Message::new(MessageRole::System, status));
                self.scroll_offset = 0;
            }
            AgentEvent::TextDelta(text) if visible => {
                if let Some(idx) = turn.streaming
                    && let Some(target) = self.messages.get_mut(idx)
                {
                    target.content.push_str(&text);
                } else {
                    self.messages
                        .push(Message::new(MessageRole::Assistant, text));
                    turn.streaming = Some(self.messages.len().saturating_sub(1));
                }
                self.scroll_offset = 0;
            }
            AgentEvent::ToolStarted { tool, detail } if visible => {
                turn.streaming = None;
                let mut message = Message::new(MessageRole::Step, format!("> {tool}"));
                message.detail = Some(detail);
                self.messages.push(message);
                turn.open_step = Some(self.messages.len().saturating_sub(1));
                self.scroll_offset = 0;
            }
            AgentEvent::ToolFinished(step) if visible => {
                let line = format!("\n  {}, {} ms", step.summary, step.duration_ms);
                if let Some(idx) = turn.open_step.take()
                    && let Some(msg) = self.messages.get_mut(idx)
                {
                    msg.content.push_str(&line);
                } else {
                    self.messages.push(step_message(
                        &step.tool,
                        &step.detail,
                        &step.summary,
                        step.duration_ms,
                    ));
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
                self.messages.push(Message::new(
                    MessageRole::System,
                    format!(
                        "Job #{} in session {} {}; /resume {} to read it.",
                        self.job_number(turn.job),
                        short_id(&turn.session_id),
                        if response.cancelled {
                            "was cancelled"
                        } else {
                            "finished"
                        },
                        turn.session_id
                    ),
                ));
            }
            AgentEvent::Failed(err) => {
                turn.ended = true;
                let text = if visible {
                    err.message
                } else {
                    format!(
                        "Job #{} in session {} failed: {err}",
                        self.job_number(turn.job),
                        short_id(&turn.session_id)
                    )
                };
                self.messages.push(Message::new(MessageRole::Error, text));
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
            format!(
                "Job #{} in session {}",
                self.job_number(turn.job),
                short_id(&turn.session_id)
            )
        };
        self.messages.push(Message::new(
            MessageRole::System,
            format!(
                "{whose} wants to run a statement that modifies the workspace:\n{}\n\
                 Run it?  y = yes   n = no   a = yes, and allow writes for this session",
                request.sql
            ),
        ));
        self.prompts.push_back(Prompt::Agent {
            job: turn.job,
            request,
        });
        self.scroll_offset = 0;
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
            self.messages
                .push(Message::new(MessageRole::Assistant, response.content));
        }
        if !response.citations.is_empty() {
            self.messages.push(Message::new(
                MessageRole::System,
                sources_footer(&response.citations),
            ));
        }
        if let Some(spec) = &response.chart {
            let chart = ChartData::from_spec(spec);
            self.current_chart = Some(chart.clone());
            if let Some(last) = self
                .messages
                .iter_mut()
                .rev()
                .find(|m| m.role == MessageRole::Assistant)
            {
                last.chart = Some(chart);
            }
        }
        for result in response.graph.iter().filter(|r| !r.is_empty()) {
            self.messages.push(Message::new(
                MessageRole::System,
                traverse::render_tree(result),
            ));
        }
        if response.write_refused && !self.writes_allowed() {
            self.messages.push(Message::new(
                MessageRole::System,
                "A write was refused this turn. Answer y next time, or restart with --allow-write.",
            ));
        }
        turn.streaming = None;
        turn.open_step = None;
        self.scroll_offset = 0;
    }

    fn writes_allowed(&self) -> bool {
        self.allow_write.load(Ordering::Relaxed)
    }

    /// The newest turn of the session on screen, queued or running.
    fn current_turn(&self) -> Option<JobId> {
        self.turns
            .iter()
            .rev()
            .find(|t| t.session_id == self.session_id)
            .map(|t| t.job)
    }

    /// Cancel a turn (issue #45): its pending permission requests are
    /// refused first so the tool returns, then the job's token stops the
    /// turn, which core records as cancelled and completes. A queued turn
    /// ends without running.
    fn cancel_turn(&mut self, job: JobId) {
        let mut kept = VecDeque::new();
        for prompt in self.prompts.drain(..) {
            match prompt {
                Prompt::Agent {
                    job: owner,
                    request,
                } if owner == job => request.deny(),
                other => kept.push_back(other),
            }
        }
        self.prompts = kept;
        if self.jobs.cancel(job) {
            self.messages.push(Message::new(
                MessageRole::System,
                format!("Cancelling job #{}…", self.job_number(job)),
            ));
        }
    }

    /// `/cancel N`: any job by its number.
    fn cancel_job(&mut self, args: &str) {
        let Some(info) = args
            .trim()
            .trim_start_matches('#')
            .parse::<u64>()
            .ok()
            .and_then(|n| self.jobs.by_number(n))
        else {
            self.messages.push(Message::new(
                MessageRole::System,
                "Usage: /cancel N, with N from /jobs",
            ));
            return;
        };
        if info.state.is_finished() {
            self.messages.push(Message::new(
                MessageRole::System,
                format!("Job #{} already {}.", info.number, info.state),
            ));
        } else if info.kind == JobKind::Chat {
            self.cancel_turn(info.id);
        } else if self.jobs.cancel(info.id) {
            let note = if info.state == JobState::Queued {
                "it will not start"
            } else {
                "it stops at its next checkpoint, or finishes if it has none"
            };
            self.messages.push(Message::new(
                MessageRole::System,
                format!("Cancelling job #{}: {note}.", info.number),
            ));
        }
    }

    /// `/jobs`: active jobs, then the most recent finished ones.
    fn show_jobs(&mut self) {
        let jobs = self.jobs.list();
        if jobs.is_empty() {
            self.messages
                .push(Message::new(MessageRole::System, "No jobs yet."));
            return;
        }
        let mut text = String::from("Jobs (newest last):");
        let start = jobs.len().saturating_sub(20);
        for job in jobs.iter().skip(start) {
            text.push_str("\n  ");
            text.push_str(&job_line(job));
        }
        text.push_str("\n/cancel N stops a queued or running job.");
        self.messages.push(Message::new(MessageRole::System, text));
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
        self.scroll_offset = 0;
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
                    self.messages.push(Message::new(
                        MessageRole::System,
                        format!(
                            "{running} job{} still running (/jobs). Press Ctrl+C again to quit and stop {}.",
                            if running == 1 { " is" } else { "s are" },
                            if running == 1 { "it" } else { "them" }
                        ),
                    ));
                    self.scroll_offset = 0;
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
                self.messages
                    .push(Message::new(MessageRole::System, WELCOME_TEXT));
            }
            (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                self.textarea = TextArea::default();
                configure_textarea(&mut self.textarea);
                self.history_cursor = None;
            }
            (KeyCode::Enter, KeyModifiers::NONE) => {
                self.submit_message();
            }
            (KeyCode::Up, KeyModifiers::NONE) => {
                self.history_up();
            }
            (KeyCode::Down, KeyModifiers::NONE) => {
                self.history_down();
            }
            (KeyCode::PageUp, _) => {
                self.scroll_offset = self.scroll_offset.saturating_add(15);
            }
            (KeyCode::PageDown, _) => {
                self.scroll_offset = self.scroll_offset.saturating_sub(15);
            }
            (KeyCode::Home, _) if self.textarea.is_empty() => {
                self.scroll_offset = usize::MAX;
            }
            (KeyCode::End, _) if self.textarea.is_empty() => {
                self.scroll_offset = 0;
            }
            _ => {
                if self
                    .textarea
                    .input(crossterm::event::KeyEvent::new(code, modifiers))
                {
                    self.reset_completion();
                }
                self.history_cursor = None;
            }
        }
    }

    fn handle_permission_key(&mut self, code: KeyCode) {
        let decision = match code {
            KeyCode::Char('y' | 'Y') => Some((true, false)),
            KeyCode::Char('a' | 'A') => Some((true, true)),
            KeyCode::Char('n' | 'N') | KeyCode::Esc => Some((false, false)),
            _ => None,
        };
        let Some((allow, for_session)) = decision else {
            return;
        };
        let Some(prompt) = self.prompts.pop_front() else {
            return;
        };
        let request = match prompt {
            Prompt::Sql(sql) => {
                self.decide_pending_sql(sql, allow, for_session);
                return;
            }
            Prompt::Agent { request, .. } => request,
        };
        if allow {
            if for_session {
                // The rest of this turn through the request, the turns
                // after (queued ones included) through the shared flag
                // each reads when it starts.
                request.allow_for_turn();
                self.allow_write.store(true, Ordering::Relaxed);
            } else {
                request.allow();
            }
            self.messages.push(Message::new(
                MessageRole::System,
                if for_session {
                    "Allowed. Writes are permitted for the rest of this session."
                } else {
                    "Allowed."
                },
            ));
        } else {
            request.deny();
            self.messages
                .push(Message::new(MessageRole::System, "Refused."));
        }
    }

    /// The user's answer to a `/sql` write prompt.
    fn decide_pending_sql(&mut self, sql: String, allow: bool, for_session: bool) {
        if !allow {
            self.messages
                .push(Message::new(MessageRole::System, "Refused."));
            return;
        }
        if for_session {
            self.allow_write.store(true, Ordering::Relaxed);
            self.messages.push(Message::new(
                MessageRole::System,
                "Allowed. Writes are permitted for the rest of this session.",
            ));
        }
        self.execute_direct_sql(sql, true);
    }

    /// What the command popup offers for the input, if it is showing:
    /// one line with the cursor at its end, not recalled from history, and
    /// not hidden with Esc.
    pub(crate) fn completion(&self) -> Option<Completion> {
        if self.popup == Popup::Hidden
            || self.history_cursor.is_some()
            || self.awaiting_permission()
        {
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
                self.set_textarea_content(&filled);
                if send {
                    self.submit_message();
                }
            }
            _ => return false,
        }
        true
    }

    fn history_up(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        let next = match self.history_cursor {
            None => self.input_history.len().saturating_sub(1),
            Some(i) => i.saturating_sub(1),
        };
        if let Some(entry) = self.input_history.get(next) {
            let text = entry.clone();
            self.history_cursor = Some(next);
            self.set_textarea_content(&text);
        }
    }

    fn history_down(&mut self) {
        let Some(current) = self.history_cursor else {
            return;
        };
        let next = current.saturating_add(1);
        if next >= self.input_history.len() {
            self.history_cursor = None;
            self.textarea = TextArea::default();
            configure_textarea(&mut self.textarea);
        } else if let Some(entry) = self.input_history.get(next) {
            let text = entry.clone();
            self.history_cursor = Some(next);
            self.set_textarea_content(&text);
        }
    }

    fn set_textarea_content(&mut self, text: &str) {
        self.reset_completion();
        self.textarea = TextArea::default();
        configure_textarea(&mut self.textarea);
        for ch in text.chars() {
            self.textarea.input(crossterm::event::KeyEvent::new(
                KeyCode::Char(ch),
                KeyModifiers::NONE,
            ));
        }
    }

    fn handle_slash_command(&mut self, input: &str) {
        let (cmd, args) = input
            .split_once(' ')
            .map_or((input, ""), |(c, a)| (c, a.trim()));
        let Some(command) = self.parse_slash_command(cmd, input) else {
            return;
        };
        match command {
            SlashCommand::Quit => self.should_quit = true,
            SlashCommand::Clear => {
                self.clear_transcript();
                self.messages
                    .push(Message::new(MessageRole::System, WELCOME_TEXT));
            }
            SlashCommand::Jobs => self.show_jobs(),
            SlashCommand::Cancel { .. } => self.cancel_job(args),
            SlashCommand::Help => {
                self.messages
                    .push(Message::new(MessageRole::System, SlashCommand::help()));
            }
            SlashCommand::Workspace => self.show_workspace(),
            SlashCommand::Sessions => self.show_sessions(),
            SlashCommand::Resume { .. } => self.switch_session(args),
            SlashCommand::New => self.new_session(),
            SlashCommand::Mode { .. } => self.set_mode(args),
            SlashCommand::Docs => self.show_documents(),
            SlashCommand::Context { action } => self.run_context_command(action.as_ref(), args),
            SlashCommand::Pin { .. } => self.set_pinned(args, true),
            SlashCommand::Unpin { .. } => self.set_pinned(args, false),
            SlashCommand::Tables => self.show_tables(),
            SlashCommand::Schema { .. } => self.show_schema(args),
            SlashCommand::Ingest { .. } => match detect_file_path(args) {
                Some(path) => self.start_ingest(path),
                None => self.messages.push(Message::new(
                    MessageRole::Error,
                    format!("'{args}' is not a file quack can ingest"),
                )),
            },
            SlashCommand::Graph {
                action: Some(action),
                ..
            } => self.run_graph_command(action),
            SlashCommand::Graph { action: None, .. } => self.show_graph(args),
            SlashCommand::Ontology { action } => self.run_ontology_command(action),
            SlashCommand::Delete { .. } => self.delete_document(args),
            SlashCommand::Import { .. } => self.start_import(args),
            SlashCommand::Path { .. } => self.show_path(args),
            SlashCommand::Sql { .. } => self.edit_or_run_sql(args),
            SlashCommand::Share => self.set_shared(true),
            SlashCommand::Unshare => self.set_shared(false),
            SlashCommand::Export { .. } => self.export_session(args),
            SlashCommand::Okf { .. } => {
                self.run_job(CliJob::Okf(args.to_owned()), "Exporting the bundle");
            }
            SlashCommand::Embeddings { action } => self.run_embeddings_command(&action),
            SlashCommand::Chart { .. } => self.show_chart(args),
            SlashCommand::Steps => self.toggle_steps(),
            SlashCommand::Model => self.show_models(),
        }
    }

    /// The command `input` names, parsed; a line clap refuses is answered
    /// in the transcript instead (its help as a note, anything else as an
    /// error).
    fn parse_slash_command(&mut self, cmd: &str, input: &str) -> Option<SlashCommand> {
        match SlashLine::try_parse_from(split_args(input)) {
            Ok(line) => Some(line.command),
            Err(e) => {
                let (role, text) = match e.kind() {
                    ErrorKind::InvalidSubcommand => {
                        (MessageRole::Error, format!("unknown command: {cmd}"))
                    }
                    ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                        (MessageRole::System, e.to_string())
                    }
                    _ => (MessageRole::Error, e.to_string()),
                };
                self.messages
                    .push(Message::new(role, text.trim_end().to_owned()));
                None
            }
        }
    }

    fn show_workspace(&mut self) {
        self.messages.push(Message::new(
            MessageRole::System,
            format!(
                "Workspace: {} ({})\nSession: {}",
                self.workspace_name, self.workspace_id, self.session_id
            ),
        ));
    }

    /// `/context`, `/context import FILE`, `/context export FILE`; the
    /// file comes from the line as typed, spaces and all.
    fn run_context_command(&mut self, action: Option<&ContextAction>, args: &str) {
        let file = args
            .split_once(' ')
            .map_or("", |(_, file)| file.trim())
            .to_owned();
        match action {
            Some(ContextAction::Import { .. }) => {
                self.run_job(CliJob::ContextImport(file), "Importing the context");
            }
            Some(ContextAction::Export { .. }) => {
                self.run_job(CliJob::ContextExport(file), "Exporting the context");
            }
            None => self.show_context(),
        }
    }

    /// `/sql STATEMENT` runs it; a bare `/sql` puts the last query back
    /// in the input to edit.
    fn edit_or_run_sql(&mut self, args: &str) {
        if !args.is_empty() {
            self.run_direct_sql(args);
            return;
        }
        match self.last_sql.clone() {
            Some(sql) => self.set_textarea_content(&sql),
            None => self.messages.push(Message::new(
                MessageRole::System,
                "No query has run yet. Use /sql STATEMENT.",
            )),
        }
    }

    fn toggle_steps(&mut self) {
        self.expand_steps = !self.expand_steps;
        self.messages.push(Message::new(
            MessageRole::System,
            if self.expand_steps {
                "Tool call details expanded."
            } else {
                "Tool call details collapsed."
            },
        ));
    }

    fn show_models(&mut self) {
        self.messages.push(Message::new(
            MessageRole::System,
            format!(
                "Chat model: {}\nEmbedding model: {}",
                self.provider_display,
                self.config
                    .embedding_model_ref()
                    .ok()
                    .flatten()
                    .map_or_else(
                        || String::from("none (keyword search only)"),
                        |m| m.to_string()
                    )
            ),
        ));
    }

    fn show_sessions(&mut self) {
        self.on_db(
            Side::Read,
            |db| sessions::list_sessions(db, 20),
            |app, listing| match listing {
                Ok(rows) if rows.is_empty() => app.note(MessageRole::System, "No sessions yet."),
                Ok(rows) => {
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
                    app.note(MessageRole::System, text);
                }
                Err(e) => app.note(MessageRole::Error, e.to_string()),
            },
        );
    }

    /// `/resume PREFIX`: find the session, then load its messages, both on
    /// the reader; input typed meanwhile waits for the switch.
    fn switch_session(&mut self, prefix: &str) {
        if prefix.is_empty() {
            self.note(MessageRole::System, "Usage: /resume SESSION_ID");
            return;
        }
        let prefix = prefix.to_owned();
        let current = self.session_id.clone();
        self.switching.get_or_insert_with(VecDeque::new);
        self.on_db(
            Side::Read,
            move |db| {
                let matches: Vec<String> = sessions::list_sessions(db, 1000)?
                    .into_iter()
                    .filter(|s| s.id.starts_with(&prefix))
                    .map(|s| s.id)
                    .collect();
                let found = match matches.as_slice() {
                    [id] if *id == current => Found::Current,
                    [id] => Found::One(id.clone(), load_session(db, id)?),
                    [] => Found::None(prefix),
                    _ => Found::Many(prefix, matches),
                };
                Ok(found)
            },
            |app, found| {
                match found {
                    Ok(Found::Current) => {
                        app.note(MessageRole::System, "That is the current session.");
                    }
                    Ok(Found::One(id, replay)) => {
                        app.forget_session_if_empty();
                        app.clear_transcript();
                        app.session_id = id;
                        app.apply_replay(replay);
                    }
                    Ok(Found::None(prefix)) => {
                        app.note(MessageRole::Error, format!("no session matches '{prefix}'"));
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
                        app.note(MessageRole::Error, text);
                    }
                    Err(e) => app.note(MessageRole::Error, e.to_string()),
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
                        app.note(MessageRole::System, text);
                    }
                    Err(e) => app.note(MessageRole::Error, e.to_string()),
                }
                app.finish_switch();
            },
        );
    }

    fn set_mode(&mut self, args: &str) {
        let session = self.session_id.clone();
        if args.is_empty() {
            self.on_db(
                Side::Read,
                move |db| {
                    Ok(sessions::get_session(db, &session)?.map_or(ChatMode::Chat, |s| s.mode))
                },
                |app, mode| match mode {
                    Ok(mode) => app.note(
                        MessageRole::System,
                        format!("Mode: {mode}. Use /mode chat or /mode query to change it."),
                    ),
                    Err(e) => app.note(MessageRole::Error, e.to_string()),
                },
            );
            return;
        }
        let mode = match args.parse::<ChatMode>() {
            Ok(mode) => mode,
            Err(e) => {
                self.note(MessageRole::Error, e.to_string());
                return;
            }
        };
        // A question typed right after must start in the new mode.
        self.switching.get_or_insert_with(VecDeque::new);
        self.on_db(
            Side::Write,
            move |db| sessions::set_session_mode(db, &session, mode),
            move |app, set| {
                match set {
                    Ok(()) => app.note(
                        MessageRole::System,
                        format!("Mode set to {mode} for this session."),
                    ),
                    Err(e) => app.note(MessageRole::Error, e.to_string()),
                }
                app.finish_switch();
            },
        );
    }

    /// `/ontology ARGS`: the CLI's `quack ontology` verbs, parsed the same
    /// way, run in the background with the answer in the transcript.
    fn run_ontology_command(&mut self, mut action: OntologyAction) {
        // The terminal owns stdin: nothing may prompt there.
        if let OntologyAction::Propose { yes, .. } = &mut action {
            *yes = true;
        }
        self.run_job(CliJob::Ontology(action), "Running ontology command");
    }

    /// `/embeddings refresh`: the CLI's `quack embeddings` verbs. The
    /// terminal owns stdin, so a refresh never asks.
    fn run_embeddings_command(&mut self, action: &EmbeddingsAction) {
        let action = match action {
            EmbeddingsAction::Refresh { .. } => EmbeddingsAction::Refresh { yes: true },
        };
        self.run_job(CliJob::Embeddings(action), "Refreshing embeddings");
    }

    /// `/graph status|extract|...`: the CLI's `quack graph` verbs.
    fn run_graph_command(&mut self, mut action: GraphAction) {
        if let GraphAction::Extract { yes, .. } = &mut action {
            *yes = true;
        }
        self.run_job(CliJob::Graph(action), "Running graph command");
    }

    /// Run an ontology, graph, bundle, or context command as a job and
    /// show what it printed. Its database steps go to the session's
    /// workspace writer one at a time, and it reports chunk progress to the
    /// job strip.
    fn run_job(&mut self, job: CliJob, label: &str) {
        let config = Arc::clone(&self.config);
        let workspace_name = self.workspace_name.clone();
        let db = Arc::clone(&self.db);
        let kind = job.kind();
        self.submit_work(
            kind,
            label.to_owned(),
            Some(&format!("{label}…")),
            move |ctx| async move {
                answered(run_job_inner(&config, &db, &workspace_name, job, &ctx).await)
            },
        );
    }
    fn show_schema(&mut self, table: &str) {
        let table = table.trim().to_owned();
        if table.is_empty() {
            self.note(MessageRole::System, "Usage: /schema TABLE");
            return;
        }
        self.on_db(
            Side::Read,
            move |db| {
                if db.list_tables()?.contains(&table) {
                    db.describe_table(&table)
                } else {
                    Err(CoreError::Analysis(format!("no table named '{table}'")))
                }
            },
            |app, described| match described {
                Ok(d) => {
                    let mut text = format!("{} ({} rows)\n", d.table_name, d.row_count);
                    for column in &d.columns {
                        let line = format!("  {} {}\n", column.name, column.column_type);
                        text.push_str(&line);
                    }
                    let mut buf = Vec::new();
                    if d.sample_rows.write_table(&mut buf).is_ok() {
                        text.push_str(&String::from_utf8_lossy(&buf));
                    }
                    app.note(MessageRole::Sql, text);
                }
                Err(e) => app.note(MessageRole::Error, e.to_string()),
            },
        );
    }

    fn delete_document(&mut self, prefix: &str) {
        let prefix = prefix.trim().to_owned();
        if prefix.is_empty() {
            self.note(MessageRole::System, "Usage: /delete DOCUMENT_ID");
            return;
        }
        self.on_db(
            Side::Write,
            move |db| {
                let doc = resolve_document(db, &prefix)?;
                let table = ingestion::parser::detect_file_type(&doc.filename)
                    .is_structured()
                    .then(|| ingestion::table_name_for(&doc.filename));
                db.delete_document(&doc.id, table.as_deref())
                    .map(|_| doc.filename)
            },
            |app, outcome| match outcome {
                Ok(filename) => app.note(
                    MessageRole::System,
                    format!("Deleted {filename} with its chunks, tables, and graph rows."),
                ),
                Err(e) => app.note(MessageRole::Error, e.to_string()),
            },
        );
    }

    fn set_shared(&mut self, shared: bool) {
        let session = self.session_id.clone();
        self.on_db(
            Side::Write,
            move |db| sessions::set_session_shared(db, &session, shared),
            move |app, outcome| match outcome {
                Ok(()) => app.note(
                    MessageRole::System,
                    if shared {
                        "This session is shared with every member of the workspace."
                    } else {
                        "This session is yours alone again."
                    },
                ),
                Err(e) => app.note(MessageRole::Error, e.to_string()),
            },
        );
    }

    /// `/export [--sql|--markdown] [FILE]`: the session as SQL or
    /// Markdown, to a file or into the transcript.
    fn export_session(&mut self, args: &str) {
        let mut sql = false;
        let mut file: Option<String> = None;
        for token in args.split_whitespace() {
            match token {
                "--sql" => sql = true,
                "--markdown" => sql = false,
                other => file = Some(other.to_owned()),
            }
        }
        let session = self.session_id.clone();
        self.on_db(
            Side::Read,
            move |db| {
                let found = sessions::get_session(db, &session)?;
                let rows = sessions::messages(db, &session)?;
                if sql {
                    sessions::export_sql(&rows)
                } else {
                    let found = found
                        .ok_or_else(|| CoreError::Analysis(String::from("session vanished")))?;
                    sessions::export_markdown(&found, &rows)
                }
            },
            move |app, text| match (text, file) {
                (Err(e), _) => app.note(MessageRole::Error, e.to_string()),
                (Ok(text), Some(path)) => match std::fs::write(&path, &text) {
                    Ok(()) => {
                        app.note(MessageRole::System, format!("Wrote the session to {path}."));
                    }
                    Err(e) => app.note(MessageRole::Error, format!("cannot write {path}: {e}")),
                },
                (Ok(text), None) => app.note(MessageRole::Sql, text),
            },
        );
    }

    fn show_context(&mut self) {
        self.on_db(
            Side::Read,
            context::current,
            |app, result| match result {
                Ok(Some(current)) => app.note(
                    MessageRole::System,
                    format!(
                        "Workspace context (version {}, {}):\n{}",
                        current.version, current.edited_at, current.content
                    ),
                ),
                Ok(None) => app.note(
                    MessageRole::System,
                    "No workspace context set. Use `quack context edit` or `quack context import FILE`.",
                ),
                Err(e) => app.note(MessageRole::Error, e.to_string()),
            },
        );
    }

    /// `/graph ENTITY [HOPS]` or `/graph --class CLASS`: a tree of the
    /// neighbourhood or of the class's entities.
    fn show_graph(&mut self, args: &str) {
        if args.is_empty() {
            self.note(
                MessageRole::System,
                "Usage: /graph ENTITY [HOPS], or /graph --class CLASS",
            );
            return;
        }
        let options = self.config.graph.options();
        let args = args.to_owned();
        self.on_db(
            Side::Read,
            move |db| {
                if let Some(class) = args.strip_prefix("--class ") {
                    let ontology = ontology_store::current(db)?;
                    traverse::by_class(
                        db,
                        ontology.as_ref(),
                        class.trim(),
                        options.max_nodes,
                        &options,
                    )
                } else {
                    let (entity, hops) = match args.rsplit_once(' ') {
                        Some((entity, hops)) if hops.parse::<u32>().is_ok() => {
                            (entity.trim(), hops.parse::<u32>().unwrap_or(2))
                        }
                        _ => (args.as_str(), 2),
                    };
                    let roots = traverse::resolve_entry(db, entity, None, None)?;
                    if roots.is_empty() {
                        return Err(CoreError::Analysis(format!("no entity matches '{entity}'")));
                    }
                    traverse::neighborhood(db, &roots, hops, None, &options)
                }
            },
            |app, outcome| match outcome {
                Ok(result) => app.note(MessageRole::System, traverse::render_tree(&result)),
                Err(e) => app.note(MessageRole::Error, e.to_string()),
            },
        );
    }

    /// `/path FROM -> TO`: the shortest relation chain.
    fn show_path(&mut self, args: &str) {
        let Some((from, to)) = args.split_once("->") else {
            self.note(MessageRole::System, "Usage: /path FROM -> TO");
            return;
        };
        let (from, to) = (from.trim().to_owned(), to.trim().to_owned());
        let options = self.config.graph.options();
        let (shown_from, shown_to) = (from.clone(), to.clone());
        self.on_db(
            Side::Read,
            move |db| {
                let a = traverse::resolve_entry(db, &from, None, None)?;
                let b = traverse::resolve_entry(db, &to, None, None)?;
                match (a.first(), b.first()) {
                    (Some(a), Some(b)) => traverse::path(db, a, b, 4, &options),
                    (None, _) => Err(CoreError::Analysis(format!("no entity matches '{from}'"))),
                    (_, None) => Err(CoreError::Analysis(format!("no entity matches '{to}'"))),
                }
            },
            move |app, outcome| match outcome {
                Ok(result) if result.is_empty() => app.note(
                    MessageRole::System,
                    format!("No path connects {shown_from} and {shown_to} within 4 hops."),
                ),
                Ok(result) => app.note(MessageRole::System, traverse::render_tree(&result)),
                Err(e) => app.note(MessageRole::Error, e.to_string()),
            },
        );
    }

    fn show_documents(&mut self) {
        self.on_db(
            Side::Read,
            WorkspaceDb::list_documents,
            |app, listing| match listing {
                Ok(docs) if docs.is_empty() => app.note(MessageRole::System, "No documents yet."),
                Ok(docs) => {
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
                    app.note(MessageRole::System, text);
                }
                Err(e) => app.note(MessageRole::Error, e.to_string()),
            },
        );
    }

    fn set_pinned(&mut self, prefix: &str, pinned: bool) {
        if prefix.is_empty() {
            self.note(
                MessageRole::System,
                if pinned {
                    "Usage: /pin DOCUMENT_ID"
                } else {
                    "Usage: /unpin DOCUMENT_ID"
                },
            );
            return;
        }
        let prefix = prefix.to_owned();
        self.on_db(
            Side::Write,
            move |db| {
                let matches: Vec<String> = db
                    .list_documents()?
                    .into_iter()
                    .filter(|d| d.id.starts_with(&prefix))
                    .map(|d| d.id)
                    .collect();
                match matches.as_slice() {
                    [id] => db.set_document_pinned(id, pinned).map(|()| id.clone()),
                    [] => Err(CoreError::Ingestion(format!(
                        "no document matches '{prefix}'"
                    ))),
                    many => Err(CoreError::Ingestion(format!(
                        "'{prefix}' matches {} documents; use more of the id",
                        many.len()
                    ))),
                }
            },
            move |app, outcome| match outcome {
                Ok(id) => app.note(
                    MessageRole::System,
                    format!(
                        "{} {}",
                        if pinned { "Pinned" } else { "Unpinned" },
                        short_id(&id)
                    ),
                ),
                Err(e) => app.note(MessageRole::Error, e.to_string()),
            },
        );
    }

    /// Drop the current session if nothing was ever recorded in it.
    /// `/chart [N]`: the Nth chart-bearing answer's chart into the pane
    /// (the last one without N).
    fn show_chart(&mut self, args: &str) {
        let charts: Vec<ChartData> = self
            .messages
            .iter()
            .filter_map(|m| m.chart.clone())
            .collect();
        if charts.is_empty() {
            self.messages.push(Message::new(
                MessageRole::System,
                "No chart in this session yet; ask for one.",
            ));
            return;
        }
        let wanted = match args.trim() {
            "" => charts.len(),
            n => match n.parse::<usize>() {
                Ok(n) if (1..=charts.len()).contains(&n) => n,
                _ => {
                    self.messages.push(Message::new(
                        MessageRole::Error,
                        format!("/chart takes a number from 1 to {}", charts.len()),
                    ));
                    return;
                }
            },
        };
        if let Some(chart) = charts.get(wanted.saturating_sub(1)) {
            self.messages.push(Message::new(
                MessageRole::System,
                format!(
                    "Showing chart {wanted} of {}: {}",
                    charts.len(),
                    chart.title
                ),
            ));
            self.current_chart = Some(chart.clone());
        }
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

    /// Push a transcript line.
    fn note(&mut self, role: MessageRole, text: impl Into<String>) {
        self.messages.push(Message::new(role, text));
        self.scroll_offset = 0;
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
            self.note(MessageRole::Error, "the database worker stopped");
        }
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

        self.input_history.push(trimmed.clone());
        save_history(&self.history_path, &self.input_history);
        self.history_cursor = None;
        self.textarea = TextArea::default();
        configure_textarea(&mut self.textarea);
        self.scroll_offset = 0;
        self.submit_text(trimmed);
    }

    /// Act on one submitted line. While a session switch is on its way the
    /// line waits, so it lands in the session the user now expects.
    fn submit_text(&mut self, trimmed: String) {
        if let Some(waiting) = self.switching.as_mut() {
            let first = waiting.is_empty();
            waiting.push_back(trimmed);
            if first {
                self.note(
                    MessageRole::System,
                    "Waiting for the session switch; this runs right after it.",
                );
            }
            return;
        }

        if trimmed.starts_with('/') {
            self.handle_slash_command(&trimmed);
            return;
        }

        if let Some(path) = detect_file_path(&trimmed) {
            self.start_ingest(path);
            return;
        }

        if looks_like_direct_sql(&trimmed) {
            self.run_direct_sql(&trimmed);
            return;
        }

        if self.config.general.chat_model.is_none() {
            self.messages.push(Message::new(MessageRole::User, trimmed));
            self.messages
                .push(Message::new(MessageRole::System, NO_CHAT_MODEL_TEXT));
            return;
        }

        self.start_agent_turn(trimmed);
    }

    /// `/import URL TABLE [SOURCE_TABLE]`: rows from an external source
    /// as a workspace table, as a job.
    fn start_import(&mut self, args: &str) {
        let tokens = split_args(args);
        let mut positional: Vec<String> = Vec::new();
        let mut query: Option<String> = None;
        let mut tokens = tokens.into_iter();
        while let Some(token) = tokens.next() {
            if token == "--query" {
                query = tokens.next();
            } else {
                positional.push(token);
            }
        }
        let mut positional = positional.into_iter();
        let (Some(url), Some(table)) = (positional.next(), positional.next()) else {
            self.messages.push(Message::new(
                MessageRole::System,
                "Usage: /import URL TABLE [SOURCE_TABLE] [--query \"SQL\"]",
            ));
            return;
        };
        let request = ImportRequest {
            url: url.clone(),
            table,
            query,
            source_table: positional.next(),
            limit: None,
        };
        let config = Arc::clone(&self.config);
        let workspace_id = self.workspace_id.clone();
        let db = Arc::clone(&self.db);
        let source = import::redact(&url);
        self.submit_work(
            JobKind::Import,
            format!("import {source}"),
            Some(&format!("Importing from {source}")),
            move |ctx| async move {
                let cancel = ctx.cancel_token();
                answered(run_import_inner(&config, &workspace_id, &db, &request, &cancel).await)
            },
        );
    }

    fn start_ingest(&mut self, path: PathBuf) {
        let config = Arc::clone(&self.config);
        let workspace_id = self.workspace_id.clone();
        let db = Arc::clone(&self.db);
        let name = path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        self.submit_work(
            JobKind::Ingest,
            name,
            Some(&format!("Ingesting {}", path.display())),
            move |ctx| async move {
                let cancel = ctx.cancel_token();
                answered(run_ingest_inner(&config, &workspace_id, &db, &path, &cancel).await)
            },
        );
    }
    /// `/sql`: the same gate the agent's statements pass. Internal tables
    /// are refused, an invalid statement is reported, and a write asks
    /// y/n/a unless writes are already allowed for the session.
    fn run_direct_sql(&mut self, sql: &str) {
        let sql = sql.trim().to_owned();
        self.messages
            .push(Message::new(MessageRole::User, sql.clone()));
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
            Ok(StatementKind::Read) => self.execute_direct_sql(sql, false),
            Ok(StatementKind::Write) if self.writes_allowed() => self.execute_direct_sql(sql, true),
            Ok(StatementKind::Write) => {
                self.messages.push(Message::new(
                    MessageRole::System,
                    "This statement modifies the workspace.\n\
                     Run it?  y = yes   n = no   a = yes, and allow writes for this session",
                ));
                self.prompts.push_back(Prompt::Sql(sql));
                self.scroll_offset = 0;
            }
            Ok(StatementKind::Invalid(message)) => {
                self.messages
                    .push(Message::new(MessageRole::Error, message));
            }
            Err(e) => {
                self.messages
                    .push(Message::new(MessageRole::Error, e.to_string()));
            }
        }
    }

    /// Run a gated statement as a job: a read on the reader pool, so it
    /// never waits on a write, and a write on the writer.
    fn execute_direct_sql(&mut self, sql: String, write: bool) {
        let db = Arc::clone(&self.db);
        let reader = self.reader_db.clone();
        let max_rows = self.config.analysis.max_query_rows;
        self.submit_work(JobKind::Sql, one_line(&sql), None, move |ctx| async move {
            // A cancel interrupts the statement itself.
            let canceller = QueryCanceller::new();
            let token = ctx.cancel_token();
            let watch = {
                let canceller = canceller.clone();
                tokio::spawn(async move {
                    token.cancelled().await;
                    canceller.cancel();
                })
            };
            let result = run_sql_task(db, reader, sql, max_rows, write, canceller).await;
            watch.abort();
            result
        });
    }
    fn show_tables(&mut self) {
        self.on_db(
            Side::Read,
            WorkspaceDb::list_tables,
            |app, listing| match listing {
                Ok(tables) if tables.is_empty() => app.note(MessageRole::System, "No tables yet."),
                Ok(tables) => {
                    let mut text = String::from("Tables:");
                    for table in tables {
                        text.push_str("\n  ");
                        text.push_str(&table);
                    }
                    app.note(MessageRole::System, text);
                }
                Err(e) => app.note(MessageRole::Error, e.to_string()),
            },
        );
    }

    /// Submit a question as a job in its session's lane: it starts once
    /// the session's previous turn has finished (its history includes that
    /// answer) and a worker is free, and streams into the transcript while
    /// everything else stays usable.
    fn start_agent_turn(&mut self, message: String) {
        self.messages
            .push(Message::new(MessageRole::User, message.clone()));
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
        let job = self.jobs.submit(spec, move |ctx| async move {
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
        });
        // Forward the turn's events into the session's one channel; the
        // close says the stream is done, whatever the job reports.
        let tx = self.msg_tx.clone();
        let mut events = rx;
        tokio::spawn(async move {
            while let Some(event) = events.recv().await {
                if tx.send(AppMsg::Turn(job, Box::new(event))).is_err() {
                    return;
                }
            }
            drop(tx.send(AppMsg::TurnClosed(job)));
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
            self.messages.push(Message::new(
                MessageRole::System,
                format!(
                    "Queued as job #{}: it runs when job #{} has answered. Esc cancels it.",
                    self.job_number(job),
                    self.job_number(previous)
                ),
            ));
        }
    }

    /// Submit background work that is not an agent turn. The work's
    /// result is posted to the transcript when it finishes; `announce`
    /// says what started, with the job's number.
    fn submit_work<F, Fut>(&mut self, kind: JobKind, label: String, announce: Option<&str>, work: F)
    where
        F: FnOnce(JobContext) -> Fut + Send + 'static,
        Fut: Future<Output = BackgroundResult> + Send + 'static,
    {
        let tx = self.msg_tx.clone();
        let spec = JobSpec::new(kind, label).workspace(self.workspace_id.clone());
        let job = self.jobs.submit(spec, move |ctx| async move {
            let id = ctx.id();
            let result = work(ctx).await;
            let outcome = match &result {
                BackgroundResult::Ingested { summary } => Ok(one_line(summary)),
                BackgroundResult::SqlResult { text } => {
                    Ok(one_line(text.lines().last().unwrap_or_default()))
                }
                BackgroundResult::Error(e) => Err(e.clone()),
            };
            drop(tx.send(AppMsg::Finished(id, result)));
            outcome
        });
        if let Some(text) = announce {
            self.messages.push(Message::new(
                MessageRole::System,
                format!("{text} (job #{})", self.job_number(job)),
            ));
        }
    }

    fn handle_background_result(&mut self, job: JobId, result: BackgroundResult) {
        let cancelled = self.jobs.get(job).is_some_and(|j| j.cancel_requested);
        match result {
            BackgroundResult::Ingested { summary } => {
                self.messages
                    .push(Message::new(MessageRole::System, summary));
            }
            BackgroundResult::SqlResult { text } => {
                self.messages.push(Message::new(MessageRole::Sql, text));
            }
            BackgroundResult::Error(err) if cancelled => {
                self.messages.push(Message::new(
                    MessageRole::System,
                    format!("Job #{} cancelled: {err}", self.job_number(job)),
                ));
            }
            BackgroundResult::Error(err) => {
                self.messages.push(Message::new(MessageRole::Error, err));
            }
        }
        self.scroll_offset = 0;
    }
}

/// Whitespace-split with single or double quotes kept together.
fn split_args(args: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    for ch in args.chars() {
        match (quote, ch) {
            (Some(q), c) if c == q => quote = None,
            (None, '"' | '\'') => quote = Some(ch),
            (None, c) if c.is_whitespace() => {
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
            }
            (_, c) => current.push(c),
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// The document whose id starts with `prefix`, when exactly one does.
fn resolve_document(
    db: &WorkspaceDb,
    prefix: &str,
) -> quack_core::error::Result<quack_core::storage::workspace::DocumentInfo> {
    let matches: Vec<_> = db
        .list_documents()?
        .into_iter()
        .filter(|d| d.id.starts_with(prefix))
        .collect();
    match matches.len() {
        1 => matches
            .into_iter()
            .next()
            .ok_or_else(|| CoreError::Ingestion(String::from("document vanished"))),
        0 => Err(CoreError::Ingestion(format!(
            "no document matches '{prefix}'"
        ))),
        n => Err(CoreError::Ingestion(format!(
            "'{prefix}' matches {n} documents; use more of the id"
        ))),
    }
}

/// A command the terminal runs as a job.
enum CliJob {
    Ontology(OntologyAction),
    Graph(GraphAction),
    Embeddings(EmbeddingsAction),
    Okf(String),
    ContextImport(String),
    ContextExport(String),
}

impl CliJob {
    const fn kind(&self) -> JobKind {
        match self {
            Self::Ontology(_) => JobKind::Ontology,
            Self::Graph(_) => JobKind::Graph,
            Self::Embeddings(_) => JobKind::Embeddings,
            Self::Okf(_) | Self::ContextExport(_) => JobKind::Export,
            Self::ContextImport(_) => JobKind::Import,
        }
    }
}

/// A job's answer as a transcript result.
fn answered(outcome: Result<String>) -> BackgroundResult {
    match outcome {
        Ok(summary) => BackgroundResult::Ingested { summary },
        Err(e) => BackgroundResult::Error(format!("{e:#}")),
    }
}

async fn run_job_inner(
    config: &Config,
    db: &SharedDb,
    workspace_name: &str,
    job: CliJob,
    ctx: &JobContext,
) -> Result<String> {
    let progress = |done: ChunkDone| ctx.progress(done.done, done.total);
    let mut out: Vec<u8> = Vec::new();
    match job {
        CliJob::Ontology(action) => {
            crate::ontology_cli::run(config, db, action, &mut out, &progress).await?;
        }
        CliJob::Graph(action) => {
            crate::graph_cli::run(config, db, action, &mut out, &progress).await?;
        }
        CliJob::Embeddings(action) => {
            // The terminal owns stdin, so the job never asks; `/cancel`
            // stops it between batches.
            let cancel = ctx.cancel_token();
            let control = RunControl {
                progress: &progress,
                cancel: Some(&cancel),
            };
            embeddings_cli::run(config, db, action, &mut out, control).await?;
        }
        CliJob::Okf(dir) => {
            let dir = dir.trim();
            if dir.is_empty() {
                anyhow::bail!("Usage: /okf DIR");
            }
            let name = workspace_name.to_owned();
            let bundle = db.run(move |db| okf::export(db, &name)).await?;
            bundle.write_to(std::path::Path::new(dir))?;
            std::io::Write::write_all(
                &mut out,
                format!("Wrote {} files to {dir}.", bundle.files.len()).as_bytes(),
            )?;
        }
        CliJob::ContextImport(file) => {
            let text = std::fs::read_to_string(&file)
                .map_err(|e| anyhow::anyhow!("cannot read {file}: {e}"))?;
            let stored = db
                .run(move |db| context::set(db, text.trim(), None))
                .await?;
            std::io::Write::write_all(
                &mut out,
                format!("Context is now version {}.", stored.version).as_bytes(),
            )?;
        }
        CliJob::ContextExport(file) => {
            let current = db
                .run(context::current)
                .await?
                .ok_or_else(|| anyhow::anyhow!("no workspace context to export"))?;
            std::fs::write(&file, &current.content)
                .map_err(|e| anyhow::anyhow!("cannot write {file}: {e}"))?;
            std::io::Write::write_all(
                &mut out,
                format!("Wrote context version {} to {file}.", current.version).as_bytes(),
            )?;
        }
    }
    Ok(String::from_utf8_lossy(&out).trim_end().to_owned())
}

/// A session's messages, read for the transcript.
fn load_session(db: &WorkspaceDb, session_id: &str) -> CoreResult<Replay> {
    Ok(Replay {
        session_id: session_id.to_owned(),
        rows: sessions::messages(db, session_id)?,
    })
}

/// A finished step as one transcript message: the header line, the
/// summary, and the full detail kept aside for `/steps`.
fn step_message(tool: &str, detail: &str, summary: &str, duration_ms: u64) -> Message {
    let mut message = Message::new(
        MessageRole::Step,
        format!("> {tool}\n  {summary}, {duration_ms} ms"),
    );
    message.detail = Some(detail.to_owned());
    message
}

/// Lines typed before, newest last; kept per data directory.
const HISTORY_LINES: usize = 500;

fn load_history(path: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .map(|text| text.lines().map(str::to_owned).collect())
        .unwrap_or_default()
}

fn save_history(path: &std::path::Path, history: &[String]) {
    let start = history.len().saturating_sub(HISTORY_LINES);
    let text = history
        .get(start..)
        .unwrap_or(history)
        .iter()
        .filter(|line| !line.contains('\n'))
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join("\n");
    if let Err(e) = std::fs::write(path, text) {
        tracing::debug!(path = %path.display(), error = %e, "could not save the input history");
    }
}

fn configure_textarea(textarea: &mut TextArea<'_>) {
    use ratatui::style::{Color, Style};

    textarea.set_cursor_line_style(Style::default());
    textarea.set_cursor_style(Style::default().fg(Color::Reset).bg(Color::White));
    textarea.set_placeholder_text("Ask a question, or type SQL...");
}

fn sources_footer(citations: &[Citation]) -> String {
    let lines: Vec<String> = citations
        .iter()
        .map(|c| format!("\n  [{}] {}", c.n, c.label()))
        .collect();
    format!("Sources:{}", lines.concat())
}

fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}

/// Run one gated statement: a read inside a read-only transaction on the
/// reader pool, a write on the writer (then let the pool notice a temp
/// object it could not see).
async fn run_sql_task(
    db: SharedDb,
    reader: ReaderDb,
    sql: String,
    max_rows: u32,
    write: bool,
    canceller: QueryCanceller,
) -> BackgroundResult {
    let started = std::time::Instant::now();
    let outcome = if write {
        let result = db
            .run(move |db| db.cancellable(&canceller, |db| db.execute_query_capped(&sql, max_rows)))
            .await;
        reader.observe_write().await;
        result
    } else {
        reader
            .with_db(move |db| {
                db.cancellable(&canceller, |db| db.execute_query_capped(&sql, max_rows))
            })
            .await
    };
    match outcome {
        Ok(capped) => {
            let mut buf = Vec::new();
            if let Err(e) = capped.results.write_table(&mut buf) {
                return BackgroundResult::Error(format!("failed to render results: {e}"));
            }
            let mut text = String::from_utf8_lossy(&buf).into_owned();
            if capped.truncated() {
                let omitted = format!("... {} more rows not shown\n", capped.omitted());
                text.push_str(&omitted);
            }
            let elapsed = format!("{} ms", started.elapsed().as_millis());
            text.push_str(&elapsed);
            BackgroundResult::SqlResult { text }
        }
        Err(e) => BackgroundResult::Error(format!("{e}")),
    }
}

/// The first line of `text`, cut to fit a job list.
fn one_line(text: &str) -> String {
    const MAX: usize = 60;
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    if line.chars().count() > MAX {
        let cut: String = line.chars().take(MAX.saturating_sub(1)).collect();
        format!("{cut}…")
    } else {
        line.to_owned()
    }
}

/// One job as `/jobs` lists it.
pub(crate) fn job_line(job: &JobInfo) -> String {
    let progress = job.progress.map(|p| format!(" {p}")).unwrap_or_default();
    let outcome = match (&job.outcome, job.state) {
        (Some(text), JobState::Succeeded | JobState::Failed | JobState::Cancelled)
            if !text.is_empty() =>
        {
            format!(" — {}", one_line(text))
        }
        _ => String::new(),
    };
    format!(
        "#{:<3} {:<9} {:<8} {}{progress}{outcome}",
        job.number,
        job.state.as_str(),
        job.kind.as_str(),
        job.label
    )
}

fn detect_file_path(input: &str) -> Option<PathBuf> {
    let cleaned = input.trim().trim_matches('\'').trim_matches('"');
    if cleaned.is_empty() || cleaned.contains('\n') {
        return None;
    }

    let path = if cleaned.starts_with('~') {
        dirs::home_dir()?.join(cleaned.strip_prefix("~/")?)
    } else {
        PathBuf::from(cleaned)
    };

    // A relative path that exists is a file too (issue #58); anything else
    // is a question for the model.
    let file_type = ingestion::parser::detect_file_type(cleaned);
    if file_type == ingestion::parser::FileType::Unknown {
        return None;
    }

    if path.is_file() { Some(path) } else { None }
}

async fn run_import_inner(
    config: &Config,
    workspace_id: &str,
    db: &SharedDb,
    request: &ImportRequest,
    cancel: &CancellationToken,
) -> Result<String> {
    let embedding_model = llm::optional_embedding_model(config).await?;
    let summary = import::import(
        config,
        db,
        workspace_id,
        request,
        ImportPolicy::owner(),
        embedding_model.as_ref(),
        Some(cancel),
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

async fn run_ingest_inner(
    config: &Config,
    workspace_id: &str,
    db: &SharedDb,
    path: &std::path::Path,
    cancel: &CancellationToken,
) -> Result<String> {
    let read = path.to_owned();
    let data = tokio::task::spawn_blocking(move || std::fs::read(read))
        .await?
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;

    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_owned();

    let embedding_model = llm::optional_embedding_model(config).await?;

    let outcome = ingestion::ingest_file(
        config,
        db,
        workspace_id,
        &NewFile::new(&filename, &data).cancel(Some(cancel)),
        embedding_model.as_ref(),
    )
    .await
    .map_err(|e| anyhow::anyhow!("ingestion failed: {e}"))?;
    let result = match outcome {
        IngestOutcome::Ingested(result) => result,
        IngestOutcome::Duplicate(existing) => {
            return Ok(format!(
                "Skipped {filename}: identical to {} (id: {}), already in the workspace.",
                existing.filename, existing.id
            ));
        }
    };

    let table_part = match result.tables.as_slice() {
        [] => String::new(),
        [table] => format!(" as table \"{table}\""),
        tables => format!(" as tables {}", tables.join(", ")),
    };
    let chunks_part = if result.chunks_stored > 0 {
        let embed = if embedding_model.is_some() {
            " with embeddings"
        } else {
            ""
        };
        format!(", {} chunks{embed}", result.chunks_stored)
    } else {
        String::new()
    };
    let summary = format!(
        "Loaded {} ({}){table_part}{chunks_part}\nYou can now ask questions about this data.",
        result.filename, result.file_type
    );

    Ok(summary)
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
        App::new(
            String::from("ws"),
            String::from("ws"),
            String::from("o/m"),
            Arc::new(config),
            db,
            reader_db,
            session.id,
            false,
        )
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
            job,
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
        assert_eq!(last(&app).role, MessageRole::Sql);

        app.handle_slash_command("/tables");
        db_settle(&mut app).await;
        assert!(last(&app).content.contains('t'), "{}", last(&app).content);
        app.handle_slash_command("/schema t");
        db_settle(&mut app).await;
        assert_eq!(last(&app).role, MessageRole::Sql);
        assert!(
            last(&app).content.contains("a INTEGER"),
            "{}",
            last(&app).content
        );
        app.handle_slash_command("/schema nope");
        db_settle(&mut app).await;
        assert_eq!(last(&app).role, MessageRole::Error);

        // Internal tables stay refused, a read runs without asking.
        app.handle_slash_command("/sql SELECT * FROM _quack_documents");
        db_settle(&mut app).await;
        assert_eq!(last(&app).role, MessageRole::Error);
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
        assert_eq!(last(&app).role, MessageRole::Sql);
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
        assert!(last(&app).content.contains("Usage"));
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
            .find(|m| m.role == MessageRole::Assistant)
            .unwrap_or_else(|| fail("no assistant message"));
        assert!(assistant.chart.is_some(), "the chart belongs to the answer");
        assert!(app.current_chart.is_some());
        let step = app
            .messages
            .iter()
            .find(|m| m.role == MessageRole::Step)
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
        assert_eq!(last(&app).role, MessageRole::Error);

        // Esc while a turn runs cancels it; typing goes on meanwhile.
        let job = turn.job;
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
        app.start_ingest(file);
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
        app.set_textarea_content("/workspace");
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
        use quack_core::config::{AuthMode, ProviderConfig, ProviderType};
        use quack_core::storage::workspace::{DocumentStatus, NewChunk, NewDocument};

        // Without an embedding model the job says what is missing.
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());
        app.handle_slash_command("/embeddings refresh");
        settle(&mut app).await;
        assert_eq!(last(&app).role, MessageRole::Error);
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
            String::from("ollama"),
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
                .any(|m| m.role == MessageRole::Assistant && m.content == "Twelve storms."),
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
            MessageRole::Assistant,
            String::from("Streaming"),
        ));
        let first = ui::format_messages(&app, 60);
        let entries = app.wrap_cache.borrow().len();
        assert_eq!(entries, app.messages.len());
        let keys: Vec<Option<u64>> = app
            .wrap_cache
            .borrow()
            .iter()
            .map(|e| e.as_ref().map(|(k, _)| *k))
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
            .map(|e| e.as_ref().map(|(k, _)| *k))
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
        app.set_textarea_content("SELECT 6 * 7 AS answer");
        app.submit_message();
        // A job can finish between the result drain and the job-event drain
        // of one pump, so wait for the result itself, not just an idle strip.
        pump_until(&mut app, |app| {
            app.turns.is_empty()
                && app.active_jobs.is_empty()
                && app
                    .messages
                    .iter()
                    .any(|m| m.role == MessageRole::Sql && m.content.contains("42"))
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
            app.messages.iter().any(|m| m.role == MessageRole::Error),
            "the failure is in the transcript"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn without_a_chat_model_questions_say_how_to_set_one_and_sql_still_runs() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());
        assert!(app.messages.iter().any(|m| m.content == NO_CHAT_MODEL_TEXT));

        app.set_textarea_content("how many orders shipped late?");
        app.submit_message();
        assert!(app.turns.is_empty());
        assert!(last(&app).content.contains("quack doctor"));

        app.set_textarea_content("SELECT 41 + 1 AS answer");
        app.submit_message();
        settle(&mut app).await;
        assert!(last(&app).content.contains("42"), "{}", last(&app).content);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn typed_input_is_kept_across_sessions_and_relative_paths_are_files() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        {
            let mut app = app(dir.path());
            app.set_textarea_content("/tables");
            app.submit_message();
            db_settle(&mut app).await;
        }
        let again = app(dir.path());
        assert_eq!(again.input_history, vec![String::from("/tables")]);

        let file = dir.path().join("notes.md");
        std::fs::write(&file, "# hi").unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            detect_file_path(&file.display().to_string()),
            Some(file.clone())
        );
        assert!(detect_file_path("what is in notes.md").is_none());
        assert!(detect_file_path("/nowhere/notes.md").is_none());
        assert_eq!(split_args("a \"b c\" 'd e' f"), ["a", "b c", "d e", "f"]);
    }
}
