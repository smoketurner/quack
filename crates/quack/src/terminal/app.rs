use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind};
use ratatui_textarea::TextArea;
use tokio::sync::mpsc;

use quack_core::analysis::agent::AgentResponse;
use quack_core::analysis::events::{self, AgentEvent, EventStream, PermissionRequest};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::config::Config;
use quack_core::ingestion::{self, IngestOutcome, NewFile};
use quack_core::storage::sessions::{self, ChatMode, MessageRole as StoredRole};
use quack_core::storage::workspace::{StatementKind, WorkspaceDb, looks_like_direct_sql};

use crate::terminal::chart::ChartData;
use crate::terminal::ui;
use quack_core::analysis::chart::ChartSpec;
use quack_core::analysis::citations::Citation;
use quack_core::error::Error as CoreError;
use quack_core::graph::traverse;
use quack_core::import::{self, ImportPolicy, ImportRequest};
use quack_core::llm::{self, CancellationToken};
use quack_core::okf;
use quack_core::ontology::store as ontology_store;
use quack_core::storage::context;

use crate::graph_cli::GraphAction;
use crate::ontology_cli::OntologyAction;

const TICK_RATE_MS: u64 = 50;

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
  /chart [N]        Show the chart of the Nth chart-bearing answer (default: the last)
  /steps            Expand or collapse the tool call details
  /model            Show the chat and embedding models in use
  /clear            Clear messages and chart
  /workspace        Show current workspace and session
  /quit, /exit      Exit quack

Shortcuts:
  Enter             Send message
  Up/Down           Browse input history (kept across sessions)
  PageUp/PageDown, mouse wheel   Scroll messages; Home/End jump
  Ctrl+U            Clear input line
  Ctrl+L            Clear screen
  Esc or Ctrl+C     Cancel the running turn
  Ctrl+C            Quit (when nothing is running)

Writes:
  SELECT queries always run. When the agent wants to modify the workspace
  you are asked: y runs it, n refuses it, a allows writes for this session.
  Start with --allow-write to skip the prompt.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AppState {
    Idle,
    Thinking,
    Ingesting,
    RunningSql,
    AwaitingPermission,
}

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

pub(crate) struct App {
    pub(crate) messages: Vec<Message>,
    pub(crate) textarea: TextArea<'static>,
    pub(crate) state: AppState,
    pub(crate) scroll_offset: usize,
    pub(crate) should_quit: bool,
    pub(crate) tick: usize,
    pub(crate) workspace_name: String,
    pub(crate) provider_display: String,
    pub(crate) session_id: String,
    pub(crate) current_chart: Option<ChartData>,
    /// The write awaiting a decision, while `state` is `AwaitingPermission`.
    pending_permission: Option<PermissionRequest>,
    /// A `/sql` write awaiting a decision, while `state` is
    /// `AwaitingPermission` and no agent request is pending.
    pending_sql: Option<String>,
    /// Index into `messages` of the step line being filled in.
    open_step: Option<usize>,
    /// Index of the assistant message text is streaming into, if any.
    streaming: Option<usize>,
    last_sql: Option<String>,
    input_history: Vec<String>,
    history_cursor: Option<usize>,
    config: Arc<Config>,
    workspace_id: String,
    db: SharedDb,
    reader_db: ReaderDb,
    allow_write: bool,
    agent_events: Option<EventStream>,
    /// Cancels the running turn; set while `state` is `Thinking` or
    /// `AwaitingPermission` for an agent request.
    turn_cancel: Option<CancellationToken>,
    /// `/steps`: show tool details whole instead of a preview.
    pub(crate) expand_steps: bool,
    /// Where typed input is kept across sessions.
    history_path: PathBuf,
    response_rx: mpsc::UnboundedReceiver<BackgroundResult>,
    response_tx: mpsc::UnboundedSender<BackgroundResult>,
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
        let mut app = Self {
            messages: Vec::new(),
            textarea,
            state: AppState::Idle,
            scroll_offset: 0,
            should_quit: false,
            tick: 0,
            workspace_name,
            provider_display,
            session_id,
            current_chart: None,
            pending_permission: None,
            pending_sql: None,
            open_step: None,
            streaming: None,
            last_sql: None,
            input_history: Vec::new(),
            history_cursor: None,
            config,
            workspace_id,
            db,
            reader_db,
            allow_write,
            agent_events: None,
            turn_cancel: None,
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
        // animates only while a background task is in flight.
        let mut dirty = true;

        loop {
            if dirty || Self::spinner_active(&self.state) {
                terminal.draw(|frame| ui::draw(frame, &self))?;
                dirty = false;
            }

            while let Ok(result) = self.response_rx.try_recv() {
                self.handle_background_result(result);
                dirty = true;
            }
            if self.drain_agent_events() {
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

        self.forget_session_if_empty();
        Ok(())
    }

    /// Whether the state shows an animated spinner, which needs a redraw
    /// every tick even with no new input or event to react to.
    fn spinner_active(state: &AppState) -> bool {
        matches!(
            state,
            AppState::Thinking | AppState::Ingesting | AppState::RunningSql
        )
    }

    /// Drains every pending agent event and reports whether it handled
    /// one (or the stream closed), so the caller knows whether the
    /// screen has something new to show.
    fn drain_agent_events(&mut self) -> bool {
        let mut pending = Vec::new();
        let mut closed = false;
        if let Some(rx) = self.agent_events.as_mut() {
            loop {
                match rx.try_recv() {
                    Ok(event) => pending.push(event),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        closed = true;
                        break;
                    }
                }
            }
        }
        let changed = !pending.is_empty() || closed;
        for event in pending {
            self.handle_agent_event(event);
        }
        if closed {
            self.agent_events = None;
            if self.state == AppState::Thinking {
                // The task ended without TurnComplete or Failed.
                self.finish_turn();
            }
        }
        changed
    }

    fn handle_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::Status(status) => {
                self.messages
                    .push(Message::new(MessageRole::System, status));
                self.scroll_offset = 0;
            }
            AgentEvent::TextDelta(text) => {
                if let Some(idx) = self.streaming
                    && let Some(target) = self.messages.get_mut(idx)
                {
                    target.content.push_str(&text);
                } else {
                    self.messages
                        .push(Message::new(MessageRole::Assistant, text));
                    self.streaming = Some(self.messages.len().saturating_sub(1));
                }
                self.scroll_offset = 0;
            }
            AgentEvent::ToolStarted { tool, detail } => {
                self.streaming = None;
                let mut message = Message::new(MessageRole::Step, format!("> {tool}"));
                message.detail = Some(detail);
                self.messages.push(message);
                self.open_step = Some(self.messages.len().saturating_sub(1));
                self.scroll_offset = 0;
            }
            AgentEvent::ToolFinished(step) => {
                let line = format!("\n  {}, {} ms", step.summary, step.duration_ms);
                if let Some(idx) = self.open_step.take()
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
            AgentEvent::PermissionRequired(request) => {
                self.streaming = None;
                self.messages.push(Message::new(
                    MessageRole::System,
                    format!(
                        "The agent wants to run a statement that modifies the workspace:\n{}\n\
                         Run it?  y = yes   n = no   a = yes, and allow writes for this session",
                        request.sql
                    ),
                ));
                self.pending_permission = Some(request);
                self.state = AppState::AwaitingPermission;
                self.scroll_offset = 0;
            }
            AgentEvent::TurnComplete(response) => self.handle_turn_complete(response),
            AgentEvent::Failed(err) => {
                self.messages.push(Message::new(MessageRole::Error, err));
                self.finish_turn();
            }
        }
    }

    /// Put the validated answer, its sources, chart, and graph results in
    /// the transcript and end the turn.
    fn handle_turn_complete(&mut self, response: AgentResponse) {
        if let Some(idx) = self.streaming
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
        if response.write_refused && !self.allow_write {
            self.messages.push(Message::new(
                MessageRole::System,
                "A write was refused this turn. Answer y next time, or restart with --allow-write.",
            ));
        }
        self.finish_turn();
    }

    fn finish_turn(&mut self) {
        self.state = AppState::Idle;
        self.streaming = None;
        self.open_step = None;
        self.pending_permission = None;
        self.turn_cancel = None;
        self.scroll_offset = 0;
    }

    /// Cancel the running turn (issue #45): a pending permission request
    /// is refused first so the tool returns, then the token stops the
    /// turn, which core records as cancelled and completes.
    fn cancel_turn(&mut self) {
        if let Some(request) = self.pending_permission.take() {
            request.deny();
        }
        if let Some(cancel) = self.turn_cancel.take() {
            cancel.cancel();
            self.messages
                .push(Message::new(MessageRole::System, "Cancelling…"));
            self.state = AppState::Thinking;
        }
    }

    fn handle_key_event(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        if (code, modifiers) == (KeyCode::Char('c'), KeyModifiers::CONTROL)
            && self.turn_cancel.is_some()
        {
            self.cancel_turn();
            return;
        }
        if self.state == AppState::AwaitingPermission {
            self.handle_permission_key(code);
            return;
        }
        match (code, modifiers) {
            (KeyCode::Char('c' | 'q'), KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            (KeyCode::Esc, _) if self.state == AppState::Thinking => {
                self.cancel_turn();
            }
            (KeyCode::Char('l'), KeyModifiers::CONTROL) => {
                self.messages.clear();
                self.messages
                    .push(Message::new(MessageRole::System, WELCOME_TEXT));
                self.current_chart = None;
                self.scroll_offset = 0;
            }
            (KeyCode::Char('u'), KeyModifiers::CONTROL) if self.state == AppState::Idle => {
                self.textarea = TextArea::default();
                configure_textarea(&mut self.textarea);
                self.history_cursor = None;
            }
            (KeyCode::Enter, KeyModifiers::NONE) if self.state == AppState::Idle => {
                self.submit_message();
            }
            (KeyCode::Up, KeyModifiers::NONE) if self.state == AppState::Idle => {
                self.history_up();
            }
            (KeyCode::Down, KeyModifiers::NONE) if self.state == AppState::Idle => {
                self.history_down();
            }
            (KeyCode::PageUp, _) => {
                self.scroll_offset = self.scroll_offset.saturating_add(15);
            }
            (KeyCode::PageDown, _) => {
                self.scroll_offset = self.scroll_offset.saturating_sub(15);
            }
            (KeyCode::Home, _) if self.state != AppState::Idle || self.textarea.is_empty() => {
                self.scroll_offset = usize::MAX;
            }
            (KeyCode::End, _) if self.state != AppState::Idle || self.textarea.is_empty() => {
                self.scroll_offset = 0;
            }
            _ if self.state == AppState::Idle => {
                self.textarea
                    .input(crossterm::event::KeyEvent::new(code, modifiers));
                self.history_cursor = None;
            }
            _ => {}
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
        let Some(request) = self.pending_permission.take() else {
            self.decide_pending_sql(allow, for_session);
            return;
        };
        if allow {
            if for_session {
                // The rest of this turn through the request, the turns
                // after through the policy the next turn starts with.
                request.allow_for_turn();
                self.allow_write = true;
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
        self.state = AppState::Thinking;
    }

    /// The user's answer to a `/sql` write prompt.
    fn decide_pending_sql(&mut self, allow: bool, for_session: bool) {
        let Some(sql) = self.pending_sql.take() else {
            self.state = AppState::Idle;
            return;
        };
        if !allow {
            self.messages
                .push(Message::new(MessageRole::System, "Refused."));
            self.state = AppState::Idle;
            return;
        }
        if for_session {
            self.allow_write = true;
            self.messages.push(Message::new(
                MessageRole::System,
                "Allowed. Writes are permitted for the rest of this session.",
            ));
        }
        self.execute_direct_sql(sql);
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
                self.messages.clear();
                self.messages
                    .push(Message::new(MessageRole::System, WELCOME_TEXT));
                self.current_chart = None;
                self.scroll_offset = 0;
            }
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
                    self.messages.clear();
                    self.current_chart = None;
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
                self.messages.clear();
                self.current_chart = None;
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

    /// Run a job on its own thread and runtime (the workspace is opened
    /// again there, as ingestion does) and show what it printed.
    fn run_job(&mut self, job: CliJob, label: &str) {
        let config = Arc::clone(&self.config);
        let workspace_id = self.workspace_id.clone();
        let workspace_name = self.workspace_name.clone();
        let tx = self.response_tx.clone();
        self.messages
            .push(Message::new(MessageRole::System, format!("{label}…")));
        self.state = AppState::Ingesting;
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            let result = match rt {
                Ok(rt) => rt.block_on(async {
                    match run_job_inner(&config, &workspace_id, &workspace_name, job).await {
                        Ok(text) => BackgroundResult::Ingested { summary: text },
                        Err(e) => BackgroundResult::Error(format!("{e:#}")),
                    }
                }),
                Err(e) => BackgroundResult::Error(format!("runtime: {e}")),
            };
            drop(tx.send(result));
        });
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

    fn forget_session_if_empty(&self) {
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
    /// as a workspace table, on the ingest thread.
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
        let tx = self.response_tx.clone();
        self.messages.push(Message::new(
            MessageRole::System,
            format!("Importing from {}", import::redact(&url)),
        ));
        self.state = AppState::Ingesting;
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            let result = match rt {
                Ok(rt) => rt.block_on(async {
                    match run_import_inner(&config, &workspace_id, &request).await {
                        Ok(summary) => BackgroundResult::Ingested { summary },
                        Err(e) => BackgroundResult::Error(format!("{e:#}")),
                    }
                }),
                Err(e) => BackgroundResult::Error(format!("runtime: {e}")),
            };
            drop(tx.send(result));
        });
    }

    fn start_ingest(&mut self, path: PathBuf) {
        let config = Arc::clone(&self.config);
        let workspace_id = self.workspace_id.clone();
        let tx = self.response_tx.clone();
        self.messages.push(Message::new(
            MessageRole::System,
            format!("Ingesting {}", path.display()),
        ));
        self.state = AppState::Ingesting;

        // WorkspaceDb is !Sync so the ingest future is !Send.
        // Run on a dedicated thread with its own single-threaded runtime.
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            match rt {
                Ok(rt) => {
                    let result = rt.block_on(run_ingest_task(config, workspace_id, path));
                    drop(tx.send(result));
                }
                Err(e) => {
                    drop(tx.send(BackgroundResult::Error(format!(
                        "failed to create runtime: {e:#}"
                    ))));
                }
            }
        });
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
            Ok(StatementKind::Read) => self.execute_direct_sql(sql),
            Ok(StatementKind::Write) if self.allow_write => self.execute_direct_sql(sql),
            Ok(StatementKind::Write) => {
                self.messages.push(Message::new(
                    MessageRole::System,
                    "This statement modifies the workspace.\n\
                     Run it?  y = yes   n = no   a = yes, and allow writes for this session",
                ));
                self.pending_sql = Some(sql);
                self.state = AppState::AwaitingPermission;
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

    fn execute_direct_sql(&mut self, sql: String) {
        self.state = AppState::RunningSql;
        let db = Arc::clone(&self.db);
        let tx = self.response_tx.clone();
        let max_rows = self.config.analysis.max_query_rows;
        std::thread::spawn(move || {
            let result = run_sql_task(&db, &sql, max_rows);
            drop(tx.send(result));
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

    fn start_agent_turn(&mut self, message: String) {
        self.messages
            .push(Message::new(MessageRole::User, message.clone()));
        self.state = AppState::Thinking;
        self.streaming = None;
        self.open_step = None;

        let policy = if self.allow_write {
            WritePolicy::Allow
        } else {
            WritePolicy::Ask
        };
        let (sink, rx) = events::channel();
        self.agent_events = Some(rx);
        let cancel = CancellationToken::new();
        self.turn_cancel = Some(cancel.clone());

        let config = Arc::clone(&self.config);
        let db = Arc::clone(&self.db);
        let reader_db = self.reader_db.clone();
        let session_id = self.session_id.clone();
        tokio::spawn(async move {
            // run_turn emits TurnComplete or Failed itself; the returned
            // value is the same response, so it is not needed here.
            drop(
                llm::run_turn(
                    &config,
                    db,
                    reader_db,
                    &session_id,
                    policy,
                    &message,
                    sink,
                    cancel,
                )
                .await,
            );
        });
    }

    fn handle_background_result(&mut self, result: BackgroundResult) {
        match result {
            BackgroundResult::Ingested { summary } => {
                self.messages
                    .push(Message::new(MessageRole::System, summary));
            }
            BackgroundResult::SqlResult { text } => {
                self.messages.push(Message::new(MessageRole::Sql, text));
            }
            BackgroundResult::Error(err) => {
                self.messages.push(Message::new(MessageRole::Error, err));
            }
        }
        self.state = AppState::Idle;
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

async fn run_job_inner(
    config: &Config,
    workspace_id: &str,
    workspace_name: &str,
    job: CliJob,
) -> Result<String> {
    let ws_db = WorkspaceDb::open(config, workspace_id)
        .map_err(|e| anyhow::anyhow!("failed to open workspace: {e}"))?;
    let mut out: Vec<u8> = Vec::new();
    match job {
        CliJob::Ontology(action) => {
            crate::ontology_cli::run(config, &ws_db, action, &mut out).await?;
        }
        CliJob::Graph(action) => {
            crate::graph_cli::run(config, &ws_db, action, &mut out).await?;
        }
        CliJob::Okf(dir) => {
            let dir = dir.trim();
            if dir.is_empty() {
                anyhow::bail!("Usage: /okf DIR");
            }
            let bundle = okf::export(&ws_db, workspace_name)?;
            bundle.write_to(std::path::Path::new(dir))?;
            std::io::Write::write_all(
                &mut out,
                format!("Wrote {} files to {dir}.", bundle.files.len()).as_bytes(),
            )?;
        }
        CliJob::ContextImport(file) => {
            let text = std::fs::read_to_string(&file)
                .map_err(|e| anyhow::anyhow!("cannot read {file}: {e}"))?;
            let stored = context::set(&ws_db, text.trim(), None)?;
            std::io::Write::write_all(
                &mut out,
                format!("Context is now version {}.", stored.version).as_bytes(),
            )?;
        }
        CliJob::ContextExport(file) => {
            let current = context::current(&ws_db)?
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

fn run_sql_task(db: &SharedDb, sql: &str, max_rows: u32) -> BackgroundResult {
    let db = match db.lock() {
        Ok(db) => db,
        Err(e) => return BackgroundResult::Error(format!("workspace lock poisoned: {e}")),
    };
    let started = std::time::Instant::now();
    match db.execute_query_capped(sql, max_rows) {
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

async fn run_ingest_task(
    config: Arc<Config>,
    workspace_id: String,
    path: PathBuf,
) -> BackgroundResult {
    match run_ingest_inner(&config, &workspace_id, &path).await {
        Ok(summary) => BackgroundResult::Ingested { summary },
        Err(e) => BackgroundResult::Error(format!("{e:#}")),
    }
}

async fn run_import_inner(
    config: &Config,
    workspace_id: &str,
    request: &ImportRequest,
) -> Result<String> {
    let ws_db = WorkspaceDb::open(config, workspace_id)
        .map_err(|e| anyhow::anyhow!("failed to open workspace: {e}"))?;
    let embedding_model = llm::optional_embedding_model(config).await?;
    let summary = import::import(
        config,
        &ws_db,
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
    path: &std::path::Path,
) -> Result<String> {
    let data = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;

    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_owned();

    let ws_db = WorkspaceDb::open(config, workspace_id)
        .map_err(|e| anyhow::anyhow!("failed to open workspace: {e}"))?;

    let embedding_model = llm::optional_embedding_model(config).await?;

    let outcome = ingestion::ingest_file(
        config,
        &ws_db,
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
    fn settle(app: &mut App) {
        for _ in 0..200 {
            if let Ok(result) = app.response_rx.try_recv() {
                app.handle_background_result(result);
                return;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        fail("no background result arrived");
    }

    fn last(app: &App) -> &Message {
        app.messages.last().unwrap_or_else(|| fail("no messages"))
    }

    #[test]
    fn slash_commands_run_sql_schema_and_cli_verbs_without_a_terminal() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());

        // A write asks first; `y` runs it on the background thread.
        app.handle_slash_command("/sql CREATE TABLE t AS SELECT 1 AS a, 'x' AS b");
        assert_eq!(app.state, AppState::AwaitingPermission);
        assert!(app.pending_sql.is_some());
        app.handle_key_event(KeyCode::Char('y'), KeyModifiers::NONE);
        assert_eq!(app.state, AppState::RunningSql);
        settle(&mut app);
        assert_eq!(app.state, AppState::Idle);
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
        assert_eq!(app.state, AppState::RunningSql);
        settle(&mut app);

        // The CLI verbs: clap parses them, background jobs answer.
        app.handle_slash_command("/ontology --help");
        assert!(
            last(&app).content.contains("Usage"),
            "{}",
            last(&app).content
        );
        app.handle_slash_command("/graph status");
        assert_eq!(app.state, AppState::Ingesting);
        settle(&mut app);
        assert!(
            last(&app).content.contains("Graph: 0 nodes"),
            "{}",
            last(&app).content
        );
        app.handle_slash_command("/ontology init");
        settle(&mut app);
        app.handle_slash_command("/ontology show");
        settle(&mut app);
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
    }

    #[test]
    fn agent_events_attach_charts_and_steps_and_keys_cancel_the_turn() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());
        app.state = AppState::Thinking;
        app.handle_agent_event(AgentEvent::ToolStarted {
            tool: String::from("run_sql"),
            detail: (1..=6)
                .map(|i| format!("line {i}"))
                .collect::<Vec<_>>()
                .join("\n"),
        });
        app.handle_agent_event(AgentEvent::ToolFinished(ToolStep {
            tool: String::from("run_sql"),
            detail: String::new(),
            summary: String::from("3 rows"),
            duration_ms: 4,
        }));
        app.handle_agent_event(AgentEvent::TextDelta(String::from("**Three** rows")));
        let spec: ChartSpec = serde_json::from_value(serde_json::json!({
            "title": "Rows by kind",
            "kind": "bar",
            "x": { "label": "kind", "values": ["a", "b"] },
            "series": [{ "name": "n", "values": [1.0, 2.0] }]
        }))
        .unwrap_or_else(|e| fail(&e.to_string()));
        app.handle_agent_event(AgentEvent::TurnComplete(AgentResponse {
            content: String::from("**Three** rows"),
            chart: Some(spec),
            ..AgentResponse::default()
        }));
        assert_eq!(app.state, AppState::Idle);
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

        // Esc while a turn runs cancels it; Ctrl+C when idle quits.
        let token = CancellationToken::new();
        app.turn_cancel = Some(token.clone());
        app.state = AppState::Thinking;
        app.handle_key_event(KeyCode::Esc, KeyModifiers::NONE);
        assert!(token.is_cancelled());
        assert!(last(&app).content.contains("Cancelling"));
        app.finish_turn();
        app.handle_key_event(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(app.should_quit);
    }

    #[test]
    fn without_a_chat_model_questions_say_how_to_set_one_and_sql_still_runs() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut app = app(dir.path());
        assert!(app.messages.iter().any(|m| m.content == NO_CHAT_MODEL_TEXT));

        app.set_textarea_content("how many orders shipped late?");
        app.submit_message();
        assert_eq!(app.state, AppState::Idle);
        assert!(app.agent_events.is_none());
        assert!(last(&app).content.contains("quack doctor"));

        app.set_textarea_content("SELECT 41 + 1 AS answer");
        app.submit_message();
        settle(&mut app);
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
