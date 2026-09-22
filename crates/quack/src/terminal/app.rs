use std::collections::VecDeque;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind};
use ratatui_textarea::TextArea;
use tokio::sync::{broadcast, mpsc};

use quack_core::analysis::agent::AgentResponse;
use quack_core::analysis::events::{self, AgentEvent, EventStream, PermissionRequest};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::config::Config;
use quack_core::ingestion::{self, IngestOutcome, NewFile};
use quack_core::storage::sessions::{self, ChatMode, MessageRole as StoredRole};
use quack_core::storage::workspace::{
    QueryCanceller, StatementKind, WorkspaceDb, looks_like_direct_sql,
};

use crate::terminal::chart::ChartData;
use crate::terminal::ui;
use quack_core::analysis::chart::ChartSpec;
use quack_core::analysis::citations::Citation;
use quack_core::error::Error as CoreError;
use quack_core::graph::traverse;
use quack_core::import::{self, ImportPolicy, ImportRequest};
use quack_core::jobs::{JobContext, JobId, JobInfo, JobKind, JobQueue, JobSpec, JobState, Lane};
use quack_core::llm;
use quack_core::okf;
use quack_core::ontology::store as ontology_store;
use quack_core::storage::context;

use crate::graph_cli::GraphAction;
use crate::ontology_cli::OntologyAction;

const TICK_RATE_MS: u64 = 50;

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

const HELP_TEXT: &str = "\
Commands:
  /help             Show this help message
  /sql [STATEMENT]  Run SQL directly; with no argument, edit the last query
  /tables           List tables in the workspace
  /schema TABLE     Columns, types, and sample rows of a table
  /ingest PATH      Load a file (a bare path typed at the prompt does the same)
  /import URL TABLE [SOURCE_TABLE] [--query SQL]  Pull rows from Postgres, SQLite, or a URL
  /docs             List ingested documents
  /pin ID, /unpin ID  Pin a document's full text into every prompt
  /delete ID        Delete a document with its chunks, table, and graph rows
  /ontology ...     quack ontology: show, init, propose, review, accept, reject, export, import, versions, restore
  /graph ...        quack graph: status, extract, revalidate, review, merges, merge, reject
  /graph ENTITY [HOPS], /graph --class CLASS   Walk the knowledge graph
  /path FROM -> TO  Shortest relation chain between two entities
  /context [import FILE | export FILE]  Show, replace, or save the workspace context
  /okf DIR          Export the workspace as an Open Knowledge Format bundle
  /sessions         List recent sessions
  /resume ID        Switch to a session (id prefix accepted) and replay it
  /new              Start a fresh session
  /mode [chat|query] Show or set the answer mode (query = sources only)
  /share, /unshare  Share this session with every member, or take it back
  /export [--sql|--markdown] [FILE]  Save this session
  /jobs             List running, queued, and recent jobs
  /cancel N         Cancel job N (queued or running)
  /chart [N]        Show the chart of the Nth chart-bearing answer (default: the last)
  /steps            Expand or collapse the tool call details
  /model            Show the chat and embedding models in use
  /clear            Clear messages and chart
  /workspace        Show current workspace and session
  /quit, /exit      Exit quack

Everything you send runs as a background job, so you can keep typing: ask
the next question, run SQL, or load a file while an answer streams. Questions
in one session are answered in order; other work runs alongside, up to
[jobs].workers at once. The strip above the input shows what is running.

Shortcuts:
  Enter             Send message
  Up/Down           Browse input history (kept across sessions)
  PageUp/PageDown, mouse wheel   Scroll messages; Home/End jump
  Ctrl+U            Clear input line
  Ctrl+L            Clear screen
  Esc or Ctrl+C     Cancel this session's newest question (running or queued)
  Ctrl+C            Quit (twice while background jobs are still running)

Writes:
  SELECT queries always run. When the agent wants to modify the workspace
  you are asked: y runs it, n refuses it, a allows writes for this session.
  Start with --allow-write to skip the prompt.";

#[derive(Debug, Clone, PartialEq, Eq)]
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

/// An agent turn submitted as a job: its events and where its text goes.
struct Turn {
    job: JobId,
    /// The session it answers in; its events render only while that
    /// session is on screen.
    session_id: String,
    events: EventStream,
    /// Index of the assistant message text is streaming into, if any.
    streaming: Option<usize>,
    /// Index into `messages` of the step line being filled in.
    open_step: Option<usize>,
    /// The turn reported its end (`TurnComplete` or `Failed`).
    ended: bool,
    /// Its event stream closed; it stays until its job's end is known.
    closed: bool,
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
    last_sql: Option<String>,
    input_history: Vec<String>,
    history_cursor: Option<usize>,
    config: Arc<Config>,
    workspace_id: String,
    db: SharedDb,
    reader_db: ReaderDb,
    /// Writes allowed for the session (`--allow-write`, or `a` at a
    /// prompt). Shared with queued turns, which read it when they start.
    allow_write: Arc<AtomicBool>,
    /// `/steps`: show tool details whole instead of a preview.
    pub(crate) expand_steps: bool,
    /// Where typed input is kept across sessions.
    history_path: PathBuf,
    response_rx: mpsc::UnboundedReceiver<(JobId, BackgroundResult)>,
    response_tx: mpsc::UnboundedSender<(JobId, BackgroundResult)>,
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
    ) -> Result<Self> {
        let (response_tx, response_rx) = mpsc::unbounded_channel();
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
            last_sql: None,
            input_history: Vec::new(),
            history_cursor: None,
            config,
            workspace_id,
            db,
            reader_db,
            allow_write: Arc::new(AtomicBool::new(allow_write)),
            expand_steps: false,
            history_path,
            response_rx,
            response_tx,
        };
        app.input_history = load_history(&app.history_path);
        app.messages
            .push(Message::new(MessageRole::System, WELCOME_TEXT));
        if app.config.general.chat_model.is_none() {
            app.messages
                .push(Message::new(MessageRole::System, NO_CHAT_MODEL_TEXT));
        }
        let current = app.session_id.clone();
        app.replay_session(&current)?;
        Ok(app)
    }

    /// Load a session's stored messages into the transcript.
    fn replay_session(&mut self, session_id: &str) -> Result<()> {
        let rows = {
            let db = self
                .db
                .lock()
                .map_err(|e| anyhow::anyhow!("workspace lock poisoned: {e}"))?;
            sessions::messages(&db, session_id)?
        };
        if rows.is_empty() {
            self.messages.push(Message::new(
                MessageRole::System,
                format!("Session {session_id} has no messages yet."),
            ));
            return Ok(());
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
        Ok(())
    }

    pub(crate) fn run(mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<()> {
        let tick_rate = Duration::from_millis(TICK_RATE_MS);
        // Redrawing unconditionally at the tick rate re-renders the whole
        // transcript (markdown parsing and word-wrap over every message)
        // every 50 ms forever, including while the session sits idle with
        // nothing on screen changing. `dirty` gates the draw on there
        // being something new to show; the only thing that still needs a
        // steady redraw with nothing else happening is the spinner, which
        // animates only while a job is queued or running.
        let mut dirty = true;

        loop {
            if dirty || self.spinner_active() {
                terminal.draw(|frame| ui::draw(frame, &self))?;
                dirty = false;
            }

            if self.pump() {
                dirty = true;
            }

            if event::poll(tick_rate)? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        self.handle_key_event(key.code, key.modifiers);
                        dirty = true;
                    }
                    Event::Mouse(mouse) => match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            self.scroll_offset = self.scroll_offset.saturating_add(3);
                            dirty = true;
                        }
                        MouseEventKind::ScrollDown => {
                            self.scroll_offset = self.scroll_offset.saturating_sub(3);
                            dirty = true;
                        }
                        _ => {}
                    },
                    Event::Resize(..) => dirty = true,
                    _ => {}
                }
            }

            self.tick = self.tick.wrapping_add(1);

            if self.should_quit {
                break;
            }
        }

        self.cancel_all_jobs();
        self.wait_for_jobs(QUIT_GRACE);
        self.forget_session_if_empty();
        Ok(())
    }

    /// After cancelling everything, give the jobs up to `grace` to stop:
    /// a turn records its cancellation, a statement is interrupted. Work
    /// with no checkpoint (an ingest mid-embedding) runs on a detached
    /// thread (see [`on_blocking_thread`]), so it never holds the process
    /// open past this.
    fn wait_for_jobs(&mut self, grace: Duration) {
        let started = std::time::Instant::now();
        while self.jobs.counts(None).active() > 0 && started.elapsed() < grace {
            self.pump();
            std::thread::sleep(Duration::from_millis(TICK_RATE_MS));
        }
    }

    /// Apply everything the background work produced since the last call:
    /// finished jobs' results, agent events, and job status changes.
    /// Returns whether anything changed on screen.
    fn pump(&mut self) -> bool {
        let mut changed = false;
        while let Ok((job, result)) = self.response_rx.try_recv() {
            self.handle_background_result(job, result);
            changed = true;
        }
        if self.drain_agent_events() {
            changed = true;
        }
        if self.drain_job_events() {
            changed = true;
        }
        changed
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

    /// Refresh the job strip when any job changed state.
    fn drain_job_events(&mut self) -> bool {
        let mut changed = false;
        // A lagged receiver lost snapshots, but the list below is rebuilt
        // from the queue itself.
        while let Ok(_) | Err(broadcast::error::TryRecvError::Lagged(_)) =
            self.job_events.try_recv()
        {
            changed = true;
        }
        if changed {
            self.active_jobs = self
                .jobs
                .list()
                .into_iter()
                .filter(|j| !j.state.is_finished())
                .collect();
        }
        changed
    }

    /// Drains every turn's pending events and reports whether it handled
    /// one (or a stream closed), so the caller knows whether the screen
    /// has something new to show.
    fn drain_agent_events(&mut self) -> bool {
        let mut turns = std::mem::take(&mut self.turns);
        let mut changed = false;
        turns.retain_mut(|turn| {
            let mut pending = Vec::new();
            while !turn.closed {
                match turn.events.try_recv() {
                    Ok(event) => pending.push(event),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        turn.closed = true;
                        changed = true;
                        // Nothing will answer its prompts now.
                        self.prompts.retain(
                            |p| !matches!(p, Prompt::Agent { job, .. } if *job == turn.job),
                        );
                    }
                }
            }
            changed = changed || !pending.is_empty();
            for event in pending {
                self.handle_turn_event(turn, event);
            }
            if !turn.closed {
                return true;
            }
            if turn.ended {
                return false;
            }
            // Closed without an answer or a failure: it failed before the
            // turn began (no model, a missing session) or was cancelled
            // while queued. Its job says which, once it has finished.
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
                    changed = true;
                    false
                }
                None => false,
            }
        });
        // Nothing handled above submits a turn, but keep any that were.
        turns.append(&mut self.turns);
        self.turns = turns;
        changed
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
                    err
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
        let mut text = format!("Jobs ({} workers; newest last):", self.jobs.workers());
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
                self.textarea
                    .input(crossterm::event::KeyEvent::new(code, modifiers));
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

        match cmd {
            "/quit" | "/exit" | "/q" => {
                self.should_quit = true;
            }
            "/clear" => {
                self.clear_transcript();
                self.messages
                    .push(Message::new(MessageRole::System, WELCOME_TEXT));
            }
            "/jobs" => self.show_jobs(),
            "/cancel" => self.cancel_job(args),
            "/help" | "/?" => {
                self.messages
                    .push(Message::new(MessageRole::System, HELP_TEXT));
            }
            "/workspace" => {
                self.messages.push(Message::new(
                    MessageRole::System,
                    format!(
                        "Workspace: {} ({})\nSession: {}",
                        self.workspace_name, self.workspace_id, self.session_id
                    ),
                ));
            }
            "/sessions" => self.show_sessions(),
            "/resume" => self.switch_session(args),
            "/new" => self.new_session(),
            "/mode" => self.set_mode(args),
            "/docs" => self.show_documents(),
            "/context" => match args.split_once(' ') {
                Some(("import", file)) => {
                    self.run_job(
                        CliJob::ContextImport(file.trim().to_owned()),
                        "Importing the context",
                    );
                }
                Some(("export", file)) => {
                    self.run_job(
                        CliJob::ContextExport(file.trim().to_owned()),
                        "Exporting the context",
                    );
                }
                _ => self.show_context(),
            },
            "/pin" => self.set_pinned(args, true),
            "/unpin" => self.set_pinned(args, false),
            "/tables" => self.show_tables(),
            "/schema" => self.show_schema(args),
            "/ingest" | "/attach" => match detect_file_path(args) {
                Some(path) => self.start_ingest(path),
                None => self.messages.push(Message::new(
                    MessageRole::Error,
                    format!("'{args}' is not a file quack can ingest"),
                )),
            },
            "/graph" if is_graph_subcommand(args) => self.run_graph_command(args),
            "/graph" => self.show_graph(args),
            "/ontology" => self.run_ontology_command(args),
            "/delete" => self.delete_document(args),
            "/import" => self.start_import(args),
            "/path" => self.show_path(args),
            "/sql" => {
                if args.is_empty() {
                    match self.last_sql.clone() {
                        Some(sql) => self.set_textarea_content(&sql),
                        None => self.messages.push(Message::new(
                            MessageRole::System,
                            "No query has run yet. Use /sql STATEMENT.",
                        )),
                    }
                } else {
                    self.run_direct_sql(args);
                }
            }
            other => self.handle_session_command(other, args),
        }
    }

    /// The session, chart, and view commands.
    fn handle_session_command(&mut self, cmd: &str, args: &str) {
        match cmd {
            "/share" => self.set_shared(true),
            "/unshare" => self.set_shared(false),
            "/export" => self.export_session(args),
            "/okf" => self.run_job(CliJob::Okf(args.to_owned()), "Exporting the bundle"),
            "/chart" => self.show_chart(args),
            "/steps" => {
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
            "/model" => self.messages.push(Message::new(
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
            )),
            other => {
                self.messages.push(Message::new(
                    MessageRole::Error,
                    format!("unknown command: {other}"),
                ));
            }
        }
    }

    fn show_sessions(&mut self) {
        let listing = {
            let db = match self.db.lock() {
                Ok(db) => db,
                Err(e) => {
                    self.messages.push(Message::new(
                        MessageRole::Error,
                        format!("workspace lock poisoned: {e}"),
                    ));
                    return;
                }
            };
            sessions::list_sessions(&db, 20)
        };
        match listing {
            Ok(rows) if rows.is_empty() => self
                .messages
                .push(Message::new(MessageRole::System, "No sessions yet.")),
            Ok(rows) => {
                let mut text = String::from("Sessions (most recent first):");
                for row in rows {
                    let marker = if row.id == self.session_id { "*" } else { " " };
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
                self.messages.push(Message::new(MessageRole::System, text));
            }
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, format!("{e}"))),
        }
    }

    fn switch_session(&mut self, prefix: &str) {
        if prefix.is_empty() {
            self.messages.push(Message::new(
                MessageRole::System,
                "Usage: /resume SESSION_ID",
            ));
            return;
        }
        let found = {
            let db = match self.db.lock() {
                Ok(db) => db,
                Err(e) => {
                    self.messages.push(Message::new(
                        MessageRole::Error,
                        format!("workspace lock poisoned: {e}"),
                    ));
                    return;
                }
            };
            sessions::list_sessions(&db, 1000).map(|rows| {
                rows.into_iter()
                    .filter(|s| s.id.starts_with(prefix))
                    .collect::<Vec<_>>()
            })
        };
        match found {
            Ok(matches) if matches.len() == 1 => {
                let id = matches.into_iter().next().map(|s| s.id).unwrap_or_default();
                if id == self.session_id {
                    self.messages.push(Message::new(
                        MessageRole::System,
                        "That is the current session.",
                    ));
                } else {
                    self.forget_session_if_empty();
                    self.clear_transcript();
                    self.session_id.clone_from(&id);
                    if let Err(e) = self.replay_session(&id) {
                        self.messages
                            .push(Message::new(MessageRole::Error, format!("{e}")));
                    }
                }
            }
            Ok(matches) if matches.is_empty() => self.messages.push(Message::new(
                MessageRole::Error,
                format!("no session matches '{prefix}'"),
            )),
            Ok(matches) => {
                let mut text = format!(
                    "'{prefix}' matches {} sessions; use more of the id:",
                    matches.len()
                );
                for m in matches.iter().take(10) {
                    text.push_str("\n  ");
                    text.push_str(&m.id);
                }
                self.messages.push(Message::new(MessageRole::Error, text));
            }
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, format!("{e}"))),
        }
        self.scroll_offset = 0;
    }

    fn new_session(&mut self) {
        let created = {
            let db = match self.db.lock() {
                Ok(db) => db,
                Err(e) => {
                    self.messages.push(Message::new(
                        MessageRole::Error,
                        format!("workspace lock poisoned: {e}"),
                    ));
                    return;
                }
            };
            let mode = sessions::get_session(&db, &self.session_id)
                .ok()
                .flatten()
                .map_or(ChatMode::Chat, |s| s.mode);
            sessions::create_session(&db, &self.provider_display, mode, None)
        };
        match created {
            Ok(session) => {
                self.forget_session_if_empty();
                self.session_id = session.id;
                self.clear_transcript();
                self.messages.push(Message::new(
                    MessageRole::System,
                    format!("New session {}", self.session_id),
                ));
            }
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, format!("{e}"))),
        }
        self.scroll_offset = 0;
    }

    fn set_mode(&mut self, args: &str) {
        let Ok(db) = self.db.lock() else {
            self.messages
                .push(Message::new(MessageRole::Error, "workspace lock poisoned"));
            return;
        };
        if args.is_empty() {
            let current = sessions::get_session(&db, &self.session_id)
                .ok()
                .flatten()
                .map_or(ChatMode::Chat, |s| s.mode);
            self.messages.push(Message::new(
                MessageRole::System,
                format!("Mode: {current}. Use /mode chat or /mode query to change it."),
            ));
            return;
        }
        let Some(mode) = ChatMode::parse(args) else {
            self.messages.push(Message::new(
                MessageRole::Error,
                format!("unknown mode '{args}'; use chat or query"),
            ));
            return;
        };
        match sessions::set_session_mode(&db, &self.session_id, mode) {
            Ok(()) => self.messages.push(Message::new(
                MessageRole::System,
                format!("Mode set to {mode} for this session."),
            )),
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, format!("{e}"))),
        }
    }

    /// `/ontology ARGS`: the CLI's `quack ontology` verbs, parsed the same
    /// way, run in the background with the answer in the transcript.
    fn run_ontology_command(&mut self, args: &str) {
        match OntologyArgs::try_parse_from(split_args(args)) {
            Ok(parsed) => {
                let mut action = parsed.action;
                // The terminal owns stdin: nothing may prompt there.
                if let OntologyAction::Propose { yes, .. } = &mut action {
                    *yes = true;
                }
                self.run_job(CliJob::Ontology(action), "Running ontology command");
            }
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, e.to_string())),
        }
    }

    /// `/graph status|extract|...`: the CLI's `quack graph` verbs.
    fn run_graph_command(&mut self, args: &str) {
        match GraphArgs::try_parse_from(split_args(args)) {
            Ok(parsed) => {
                let mut action = parsed.action;
                if let GraphAction::Extract { yes, .. } = &mut action {
                    *yes = true;
                }
                self.run_job(CliJob::Graph(action), "Running graph command");
            }
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, e.to_string())),
        }
    }

    /// Run an ontology, graph, bundle, or context command as a job and
    /// show what it printed. It shares the session's workspace handle,
    /// locked only around each database step, and reports chunk progress
    /// to the job strip.
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
                on_blocking_thread(move |rt| {
                    rt.block_on(run_job_inner(&config, &db, &workspace_name, job, &ctx))
                })
                .await
            },
        );
    }
    fn show_schema(&mut self, table: &str) {
        let table = table.trim();
        if table.is_empty() {
            self.messages
                .push(Message::new(MessageRole::System, "Usage: /schema TABLE"));
            return;
        }
        let described = match self.db.lock() {
            Ok(db) => db.list_tables().and_then(|tables| {
                if tables.iter().any(|t| t == table) {
                    db.describe_table(table)
                } else {
                    Err(CoreError::Analysis(format!("no table named '{table}'")))
                }
            }),
            Err(e) => Err(CoreError::Analysis(format!("workspace lock poisoned: {e}"))),
        };
        match described {
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
                self.messages.push(Message::new(MessageRole::Sql, text));
            }
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, e.to_string())),
        }
    }

    fn delete_document(&mut self, prefix: &str) {
        let prefix = prefix.trim();
        if prefix.is_empty() {
            self.messages.push(Message::new(
                MessageRole::System,
                "Usage: /delete DOCUMENT_ID",
            ));
            return;
        }
        let outcome = match self.db.lock() {
            Ok(db) => resolve_document(&db, prefix).and_then(|doc| {
                let table = ingestion::parser::detect_file_type(&doc.filename)
                    .is_structured()
                    .then(|| ingestion::table_name_for(&doc.filename));
                db.delete_document(&doc.id, table.as_deref())
                    .map(|_| doc.filename)
            }),
            Err(e) => Err(CoreError::Analysis(format!("workspace lock poisoned: {e}"))),
        };
        match outcome {
            Ok(filename) => self.messages.push(Message::new(
                MessageRole::System,
                format!("Deleted {filename} with its chunks, tables, and graph rows."),
            )),
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, e.to_string())),
        }
    }

    fn set_shared(&mut self, shared: bool) {
        let outcome = match self.db.lock() {
            Ok(db) => sessions::set_session_shared(&db, &self.session_id, shared),
            Err(e) => Err(CoreError::Analysis(format!("workspace lock poisoned: {e}"))),
        };
        match outcome {
            Ok(()) => self.messages.push(Message::new(
                MessageRole::System,
                if shared {
                    "This session is shared with every member of the workspace."
                } else {
                    "This session is yours alone again."
                },
            )),
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, e.to_string())),
        }
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
        let text = match self.db.lock() {
            Ok(db) => sessions::get_session(&db, &self.session_id).and_then(|session| {
                let rows = sessions::messages(&db, &self.session_id)?;
                if sql {
                    sessions::export_sql(&rows)
                } else {
                    let session = session
                        .ok_or_else(|| CoreError::Analysis(String::from("session vanished")))?;
                    sessions::export_markdown(&session, &rows)
                }
            }),
            Err(e) => Err(CoreError::Analysis(format!("workspace lock poisoned: {e}"))),
        };
        let text = match text {
            Ok(text) => text,
            Err(e) => {
                self.messages
                    .push(Message::new(MessageRole::Error, e.to_string()));
                return;
            }
        };
        match file {
            Some(path) => match std::fs::write(&path, &text) {
                Ok(()) => self.messages.push(Message::new(
                    MessageRole::System,
                    format!("Wrote the session to {path}."),
                )),
                Err(e) => self.messages.push(Message::new(
                    MessageRole::Error,
                    format!("cannot write {path}: {e}"),
                )),
            },
            None => self.messages.push(Message::new(MessageRole::Sql, text)),
        }
    }

    fn show_context(&mut self) {
        let result = match self.db.lock() {
            Ok(db) => context::current(&db),
            Err(e) => {
                self.messages.push(Message::new(
                    MessageRole::Error,
                    format!("workspace lock poisoned: {e}"),
                ));
                return;
            }
        };
        match result {
            Ok(Some(current)) => self.messages.push(Message::new(
                MessageRole::System,
                format!(
                    "Workspace context (version {}, {}):\n{}",
                    current.version, current.edited_at, current.content
                ),
            )),
            Ok(None) => self.messages.push(Message::new(
                MessageRole::System,
                "No workspace context set. Use `quack context edit` or `quack context import FILE`.",
            )),
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, format!("{e}"))),
        }
    }

    /// `/graph ENTITY [HOPS]` or `/graph --class CLASS`: a tree of the
    /// neighbourhood or of the class's entities.
    fn show_graph(&mut self, args: &str) {
        if args.is_empty() {
            self.messages.push(Message::new(
                MessageRole::System,
                "Usage: /graph ENTITY [HOPS], or /graph --class CLASS",
            ));
            return;
        }
        let options = self.config.graph.options();
        let outcome = match self.db.lock() {
            Ok(db) => {
                if let Some(class) = args.strip_prefix("--class ") {
                    ontology_store::current(&db).and_then(|ontology| {
                        traverse::by_class(
                            &db,
                            ontology.as_ref(),
                            class.trim(),
                            options.max_nodes,
                            &options,
                        )
                    })
                } else {
                    let (entity, hops) = match args.rsplit_once(' ') {
                        Some((entity, hops)) if hops.parse::<u32>().is_ok() => {
                            (entity.trim(), hops.parse::<u32>().unwrap_or(2))
                        }
                        _ => (args, 2),
                    };
                    traverse::resolve_entry(&db, entity, None, None).and_then(|roots| {
                        if roots.is_empty() {
                            return Err(CoreError::Analysis(format!(
                                "no entity matches '{entity}'"
                            )));
                        }
                        traverse::neighborhood(&db, &roots, hops, None, &options)
                    })
                }
            }
            Err(e) => Err(CoreError::Analysis(format!("workspace lock poisoned: {e}"))),
        };
        match outcome {
            Ok(result) => self.messages.push(Message::new(
                MessageRole::System,
                traverse::render_tree(&result),
            )),
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, format!("{e}"))),
        }
    }

    /// `/path FROM -> TO`: the shortest relation chain.
    fn show_path(&mut self, args: &str) {
        let Some((from, to)) = args.split_once("->") else {
            self.messages
                .push(Message::new(MessageRole::System, "Usage: /path FROM -> TO"));
            return;
        };
        let (from, to) = (from.trim(), to.trim());
        let options = self.config.graph.options();
        let outcome = match self.db.lock() {
            Ok(db) => traverse::resolve_entry(&db, from, None, None).and_then(|a| {
                let b = traverse::resolve_entry(&db, to, None, None)?;
                match (a.first(), b.first()) {
                    (Some(a), Some(b)) => traverse::path(&db, a, b, 4, &options),
                    (None, _) => Err(CoreError::Analysis(format!("no entity matches '{from}'"))),
                    (_, None) => Err(CoreError::Analysis(format!("no entity matches '{to}'"))),
                }
            }),
            Err(e) => Err(CoreError::Analysis(format!("workspace lock poisoned: {e}"))),
        };
        match outcome {
            Ok(result) if result.is_empty() => self.messages.push(Message::new(
                MessageRole::System,
                format!("No path connects {from} and {to} within 4 hops."),
            )),
            Ok(result) => self.messages.push(Message::new(
                MessageRole::System,
                traverse::render_tree(&result),
            )),
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, format!("{e}"))),
        }
    }

    fn show_documents(&mut self) {
        let listing = match self.db.lock() {
            Ok(db) => db.list_documents(),
            Err(e) => {
                self.messages.push(Message::new(
                    MessageRole::Error,
                    format!("workspace lock poisoned: {e}"),
                ));
                return;
            }
        };
        match listing {
            Ok(docs) if docs.is_empty() => self
                .messages
                .push(Message::new(MessageRole::System, "No documents yet.")),
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
                self.messages.push(Message::new(MessageRole::System, text));
            }
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, format!("{e}"))),
        }
    }

    fn set_pinned(&mut self, prefix: &str, pinned: bool) {
        if prefix.is_empty() {
            self.messages.push(Message::new(
                MessageRole::System,
                if pinned {
                    "Usage: /pin DOCUMENT_ID"
                } else {
                    "Usage: /unpin DOCUMENT_ID"
                },
            ));
            return;
        }
        let outcome = match self.db.lock() {
            Ok(db) => db.list_documents().and_then(|docs| {
                let matches: Vec<String> = docs
                    .into_iter()
                    .filter(|d| d.id.starts_with(prefix))
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
            }),
            Err(e) => Err(CoreError::Analysis(format!("workspace lock poisoned: {e}"))),
        };
        match outcome {
            Ok(id) => self.messages.push(Message::new(
                MessageRole::System,
                format!(
                    "{} {}",
                    if pinned { "Pinned" } else { "Unpinned" },
                    short_id(&id)
                ),
            )),
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, format!("{e}"))),
        }
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
    /// unless a turn of it is still queued or running.
    fn forget_session_if_empty(&self) {
        if self.turns.iter().any(|t| t.session_id == self.session_id) {
            return;
        }
        if let Ok(db) = self.db.lock() {
            drop(sessions::delete_if_empty(&db, &self.session_id));
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
            move |_| async move {
                on_blocking_thread(move |rt| {
                    rt.block_on(run_import_inner(&config, &workspace_id, &db, &request))
                })
                .await
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
            move |_| async move {
                on_blocking_thread(move |rt| {
                    rt.block_on(run_ingest_inner(&config, &workspace_id, &db, &path))
                })
                .await
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
        let kind = match self.db.lock() {
            Ok(db) => db.classify_user_statement(&sql),
            Err(e) => {
                self.messages.push(Message::new(
                    MessageRole::Error,
                    format!("workspace lock poisoned: {e}"),
                ));
                return;
            }
        };
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
        let listing = match self.db.lock() {
            Ok(db) => db.list_tables(),
            Err(e) => {
                self.messages.push(Message::new(
                    MessageRole::Error,
                    format!("workspace lock poisoned: {e}"),
                ));
                return;
            }
        };
        match listing {
            Ok(tables) if tables.is_empty() => self
                .messages
                .push(Message::new(MessageRole::System, "No tables yet.")),
            Ok(tables) => {
                let mut text = String::from("Tables:");
                for table in tables {
                    text.push_str("\n  ");
                    text.push_str(&table);
                }
                self.messages.push(Message::new(MessageRole::System, text));
            }
            Err(e) => self
                .messages
                .push(Message::new(MessageRole::Error, e.to_string())),
        }
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
            .lane(Lane::serial(format!("session:{session_id}")));
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
        self.turns.push(Turn {
            job,
            session_id: self.session_id.clone(),
            events: rx,
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
        let tx = self.response_tx.clone();
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
            drop(tx.send((id, result)));
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

/// `/ontology` and `/graph` arguments, parsed as the CLI parses them.
#[derive(Parser)]
#[command(name = "/ontology", no_binary_name = true, disable_help_flag = false)]
struct OntologyArgs {
    #[command(subcommand)]
    action: OntologyAction,
}

#[derive(Parser)]
#[command(name = "/graph", no_binary_name = true)]
struct GraphArgs {
    #[command(subcommand)]
    action: GraphAction,
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

/// Whether `/graph ARGS` names a `quack graph` verb rather than an entity.
fn is_graph_subcommand(args: &str) -> bool {
    matches!(
        args.split_whitespace().next(),
        Some(
            "status"
                | "extract"
                | "revalidate"
                | "review"
                | "merges"
                | "merge"
                | "reject"
                | "search"
                | "path"
                | "help"
                | "--help"
                | "-h"
        )
    )
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

/// Work the terminal hands to a background thread with its own runtime.
enum CliJob {
    Ontology(OntologyAction),
    Graph(GraphAction),
    Okf(String),
    ContextImport(String),
    ContextExport(String),
}

impl CliJob {
    const fn kind(&self) -> JobKind {
        match self {
            Self::Ontology(_) => JobKind::Ontology,
            Self::Graph(_) => JobKind::Graph,
            Self::Okf(_) | Self::ContextExport(_) => JobKind::Export,
            Self::ContextImport(_) => JobKind::Import,
        }
    }
}

/// Run long blocking work (`DuckDB` steps between model calls, and
/// ingestion futures that are not `Send`) on a thread of its own, driving
/// any async part on the runtime, and turn its answer into a transcript
/// result. A detached thread rather than the blocking pool: the runtime
/// waits for its blocking pool when it shuts down, and quitting must not
/// wait for an ingest that has no checkpoint to stop at.
async fn on_blocking_thread(
    work: impl FnOnce(&tokio::runtime::Handle) -> Result<String> + Send + 'static,
) -> BackgroundResult {
    let handle = tokio::runtime::Handle::current();
    let (done, answer) = tokio::sync::oneshot::channel();
    let spawned = std::thread::Builder::new()
        .name(String::from("quack-job"))
        .spawn(move || drop(done.send(work(&handle))));
    if let Err(e) = spawned {
        return BackgroundResult::Error(format!("could not start the job's thread: {e}"));
    }
    match answer.await {
        Ok(Ok(summary)) => BackgroundResult::Ingested { summary },
        Ok(Err(e)) => BackgroundResult::Error(format!("{e:#}")),
        Err(_) => {
            BackgroundResult::Error(String::from("the job's thread ended before it answered"))
        }
    }
}

async fn run_job_inner(
    config: &Config,
    db: &SharedDb,
    workspace_name: &str,
    job: CliJob,
    ctx: &JobContext,
) -> Result<String> {
    let progress = |done: quack_core::progress::ChunkDone| ctx.progress(done.done, done.total);
    let mut out: Vec<u8> = Vec::new();
    let lock = || {
        db.lock()
            .map_err(|e| anyhow::anyhow!("workspace lock poisoned: {e}"))
    };
    match job {
        CliJob::Ontology(action) => {
            crate::ontology_cli::run(config, db, action, &mut out, &progress).await?;
        }
        CliJob::Graph(action) => {
            crate::graph_cli::run(config, db, action, &mut out, &progress).await?;
        }
        CliJob::Okf(dir) => {
            let dir = dir.trim();
            if dir.is_empty() {
                anyhow::bail!("Usage: /okf DIR");
            }
            let bundle = okf::export(&*lock()?, workspace_name)?;
            bundle.write_to(std::path::Path::new(dir))?;
            std::io::Write::write_all(
                &mut out,
                format!("Wrote {} files to {dir}.", bundle.files.len()).as_bytes(),
            )?;
        }
        CliJob::ContextImport(file) => {
            let text = std::fs::read_to_string(&file)
                .map_err(|e| anyhow::anyhow!("cannot read {file}: {e}"))?;
            let stored = context::set(&*lock()?, text.trim(), None)?;
            std::io::Write::write_all(
                &mut out,
                format!("Context is now version {}.", stored.version).as_bytes(),
            )?;
        }
        CliJob::ContextExport(file) => {
            let current = context::current(&*lock()?)?
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
        let result = tokio::task::spawn_blocking(move || {
            let db = db
                .lock()
                .map_err(|e| CoreError::Analysis(format!("workspace lock poisoned: {e}")))?;
            db.cancellable(&canceller, |db| db.execute_query_capped(&sql, max_rows))
        })
        .await
        .map_err(|e| CoreError::Analysis(format!("the query task failed: {e}")))
        .and_then(|r| r);
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
    let progress = job
        .progress
        .map(|p| format!(" {}/{}", p.done, p.total))
        .unwrap_or_default();
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
) -> Result<String> {
    let embedding_model = llm::optional_embedding_model(config).await?;
    let summary = import::import(
        config,
        db,
        workspace_id,
        request,
        ImportPolicy::owner(),
        embedding_model.as_ref(),
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
) -> Result<String> {
    let data = std::fs::read(path)
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
        &NewFile::new(&filename, &data),
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
    use std::sync::Mutex;

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
        let mut config = Config::default();
        config.general.data_dir = dir.to_path_buf();
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        let session = sessions::create_session(&db, "m", ChatMode::Chat, None)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let db: SharedDb = Arc::new(Mutex::new(db));
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
        .unwrap_or_else(|e| fail(&e.to_string()))
    }

    /// Wait for the background result a command posted and apply it.
    async fn settle(app: &mut App) {
        for _ in 0..400 {
            if let Ok((job, result)) = app.response_rx.try_recv() {
                app.handle_background_result(job, result);
                app.pump();
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        fail("no background result arrived");
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
    fn waiting_turn(app: &mut App) -> (Turn, events::EventSink) {
        let job = app.jobs.submit(
            JobSpec::new(JobKind::Chat, "question").lane(Lane::serial("session:test")),
            |ctx| async move {
                ctx.cancel_token().cancelled().await;
                Err(String::from("cancelled"))
            },
        );
        let (sink, events) = events::channel();
        let turn = Turn {
            job,
            session_id: app.session_id.clone(),
            events,
            streaming: None,
            open_step: None,
            ended: false,
            closed: false,
        };
        (turn, sink)
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
        assert!(app.awaiting_permission());
        assert!(matches!(app.prompts.front(), Some(Prompt::Sql(_))));
        app.handle_key_event(KeyCode::Char('y'), KeyModifiers::NONE);
        assert!(!app.awaiting_permission());
        settle(&mut app).await;
        assert_eq!(last(&app).role, MessageRole::Sql);

        app.handle_slash_command("/tables");
        assert!(last(&app).content.contains('t'), "{}", last(&app).content);
        app.handle_slash_command("/schema t");
        assert_eq!(last(&app).role, MessageRole::Sql);
        assert!(
            last(&app).content.contains("a INTEGER"),
            "{}",
            last(&app).content
        );
        app.handle_slash_command("/schema nope");
        assert_eq!(last(&app).role, MessageRole::Error);

        // Internal tables stay refused, a read runs without asking.
        app.handle_slash_command("/sql SELECT * FROM _quack_documents");
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
        assert_eq!(last(&app).role, MessageRole::Sql);
        app.handle_slash_command("/share");
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
        let (mut turn, _sink) = waiting_turn(&mut app);
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
                && app.messages.iter().any(|m| m.content.contains("42"))
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

    #[test]
    fn typed_input_is_kept_across_sessions_and_relative_paths_are_files() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        {
            let mut app = app(dir.path());
            app.set_textarea_content("/tables");
            app.submit_message();
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
