use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui_textarea::TextArea;
use tokio::sync::mpsc;

use quack_core::analysis::events::{self, AgentEvent, EventStream, PermissionRequest};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::SharedDb;
use quack_core::config::Config;
use quack_core::ingestion;
use quack_core::storage::sessions::{self, ChatMode, MessageRole as StoredRole};
use quack_core::storage::workspace::{WorkspaceDb, looks_like_direct_sql};

use crate::terminal::chart::ChartData;
use crate::terminal::ui;

const TICK_RATE_MS: u64 = 50;

const WELCOME_TEXT: &str = "\
Welcome to quack!

Ask questions about your data, or type SQL (SELECT, WITH, FROM, DESCRIBE, SHOW,
SUMMARIZE, PIVOT) to run it directly. Drop a file path here to load it
(CSV, JSON, Parquet, PDF, TXT, MD). Type /help for commands.";

const HELP_TEXT: &str = "\
Commands:
  /help             Show this help message
  /sql [STATEMENT]  Run SQL directly; with no argument, edit the last query
  /tables           List tables in the workspace
  /sessions         List recent sessions
  /resume ID        Switch to a session (id prefix accepted) and replay it
  /new              Start a fresh session
  /mode [chat|query] Show or set the answer mode (query = sources only)
  /docs             List ingested documents
  /pin ID, /unpin ID  Pin a document's full text into every prompt
  /context          Show the workspace context the agent is given
  /clear            Clear messages and chart
  /workspace        Show current workspace and session
  /quit, /exit      Exit quack

Shortcuts:
  Enter             Send message
  Up/Down           Browse input history
  PageUp/PageDown   Scroll messages
  Ctrl+U            Clear input line
  Ctrl+L            Clear screen
  Ctrl+C            Quit

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
}

impl Message {
    fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
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
    /// Index into `messages` of the step line being filled in.
    open_step: Option<usize>,
    /// Whether the assistant message being streamed is the last message.
    streaming_assistant: bool,
    last_sql: Option<String>,
    input_history: Vec<String>,
    history_cursor: Option<usize>,
    config: Arc<Config>,
    workspace_id: String,
    db: SharedDb,
    allow_write: bool,
    agent_events: Option<EventStream>,
    response_rx: mpsc::UnboundedReceiver<BackgroundResult>,
    response_tx: mpsc::UnboundedSender<BackgroundResult>,
}

impl App {
    pub(crate) fn new(
        workspace_name: String,
        workspace_id: String,
        provider_display: String,
        config: Arc<Config>,
        db: SharedDb,
        session_id: String,
        allow_write: bool,
    ) -> Result<Self> {
        let (response_tx, response_rx) = mpsc::unbounded_channel();
        let mut textarea = TextArea::default();
        configure_textarea(&mut textarea);

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
            open_step: None,
            streaming_assistant: false,
            last_sql: None,
            input_history: Vec::new(),
            history_cursor: None,
            config,
            workspace_id,
            db,
            allow_write,
            agent_events: None,
            response_rx,
            response_tx,
        };
        app.messages
            .push(Message::new(MessageRole::System, WELCOME_TEXT));
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
            return Ok(());
        }
        self.messages.push(Message::new(
            MessageRole::System,
            format!(
                "Resumed session {} ({} messages)",
                short_id(session_id),
                rows.len()
            ),
        ));
        for row in rows {
            match row.role {
                StoredRole::User => self
                    .messages
                    .push(Message::new(MessageRole::User, row.content)),
                StoredRole::Assistant => {
                    if let Some(chart) = row.metadata.as_ref().and_then(|m| m.get("chart")) {
                        self.current_chart = ChartData::from_echart_spec(chart);
                    }
                    self.messages
                        .push(Message::new(MessageRole::Assistant, row.content));
                    if let Some(citations) = row
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("citations"))
                        .and_then(|c| {
                            serde_json::from_value::<
                                    Vec<quack_core::analysis::citations::Citation>,
                                >(c.clone())
                                .ok()
                        })
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
                    self.messages.push(Message::new(
                        MessageRole::Step,
                        format!("> {tool}\n  {}, {ms} ms", row.content),
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn run(mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<()> {
        let tick_rate = Duration::from_millis(TICK_RATE_MS);

        loop {
            terminal.draw(|frame| ui::draw(frame, &self))?;

            while let Ok(result) = self.response_rx.try_recv() {
                self.handle_background_result(result);
            }
            self.drain_agent_events();

            if event::poll(tick_rate)?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                self.handle_key_event(key.code, key.modifiers);
            }

            self.tick = self.tick.wrapping_add(1);

            if self.should_quit {
                break;
            }
        }

        self.forget_session_if_empty();
        Ok(())
    }

    fn drain_agent_events(&mut self) {
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
    }

    fn handle_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::TextDelta(text) => {
                if self.streaming_assistant
                    && let Some(last) = self.messages.last_mut()
                    && last.role == MessageRole::Assistant
                {
                    last.content.push_str(&text);
                } else {
                    self.messages
                        .push(Message::new(MessageRole::Assistant, text));
                    self.streaming_assistant = true;
                }
                self.scroll_offset = 0;
            }
            AgentEvent::ToolStarted { tool, detail } => {
                self.streaming_assistant = false;
                let mut content = format!("> {tool}");
                for line in detail.lines().take(12) {
                    content.push_str("\n  ");
                    content.push_str(line);
                }
                if detail.lines().count() > 12 {
                    content.push_str("\n  ...");
                }
                self.messages.push(Message::new(MessageRole::Step, content));
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
                    self.messages.push(Message::new(
                        MessageRole::Step,
                        format!("> {}{line}", step.tool),
                    ));
                }
            }
            AgentEvent::PermissionRequired(request) => {
                self.streaming_assistant = false;
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
            AgentEvent::TurnComplete(response) => {
                if self.streaming_assistant
                    && let Some(last) = self.messages.last_mut()
                    && last.role == MessageRole::Assistant
                {
                    // Citation validation may have renumbered or stripped markers.
                    last.content.clone_from(&response.content);
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
                if let Some(spec) = response.chart_spec {
                    self.current_chart = ChartData::from_echart_spec(&spec);
                }
                if response.write_refused && !self.allow_write {
                    self.messages.push(Message::new(
                        MessageRole::System,
                        "A write was refused this turn. Answer y next time, or restart with --allow-write.",
                    ));
                }
                self.finish_turn();
            }
            AgentEvent::Failed(err) => {
                self.messages.push(Message::new(MessageRole::Error, err));
                self.finish_turn();
            }
        }
    }

    fn finish_turn(&mut self) {
        self.state = AppState::Idle;
        self.streaming_assistant = false;
        self.open_step = None;
        self.pending_permission = None;
        self.scroll_offset = 0;
    }

    fn handle_key_event(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        if self.state == AppState::AwaitingPermission {
            self.handle_permission_key(code);
            return;
        }
        match (code, modifiers) {
            (KeyCode::Char('c' | 'q'), KeyModifiers::CONTROL) => {
                self.should_quit = true;
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
            self.state = AppState::Thinking;
            return;
        };
        if allow {
            request.allow();
            self.messages.push(Message::new(
                MessageRole::System,
                if for_session {
                    "Allowed. Writes are permitted for the rest of this session."
                } else {
                    "Allowed."
                },
            ));
            if for_session {
                self.allow_write = true;
            }
        } else {
            request.deny();
            self.messages
                .push(Message::new(MessageRole::System, "Refused."));
        }
        self.state = AppState::Thinking;
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
            "/context" => self.show_context(),
            "/pin" => self.set_pinned(args, true),
            "/unpin" => self.set_pinned(args, false),
            "/tables" => self.run_direct_sql("SHOW TABLES"),
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
                        short_id(&row.id),
                        row.updated_at,
                        row.message_count,
                        row.title.as_deref().unwrap_or("(untitled)")
                    );
                    text.push_str(&line);
                }
                text.push_str("\nUse /resume ID to switch.");
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
                self.forget_session_if_empty();
                self.messages.clear();
                self.current_chart = None;
                self.session_id.clone_from(&id);
                if let Err(e) = self.replay_session(&id) {
                    self.messages
                        .push(Message::new(MessageRole::Error, format!("{e}")));
                }
            }
            Ok(matches) if matches.is_empty() => self.messages.push(Message::new(
                MessageRole::Error,
                format!("no session matches '{prefix}'"),
            )),
            Ok(matches) => self.messages.push(Message::new(
                MessageRole::Error,
                format!(
                    "'{prefix}' matches {} sessions; use more of the id",
                    matches.len()
                ),
            )),
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
            sessions::create_session(&db, &self.provider_display, mode)
        };
        match created {
            Ok(session) => {
                self.forget_session_if_empty();
                self.session_id = session.id;
                self.messages.clear();
                self.current_chart = None;
                self.messages.push(Message::new(
                    MessageRole::System,
                    format!("New session {}", short_id(&self.session_id)),
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

    fn show_context(&mut self) {
        let result = match self.db.lock() {
            Ok(db) => quack_core::storage::context::current(&db),
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
                    [] => Err(quack_core::error::Error::Ingestion(format!(
                        "no document matches '{prefix}'"
                    ))),
                    many => Err(quack_core::error::Error::Ingestion(format!(
                        "'{prefix}' matches {} documents; use more of the id",
                        many.len()
                    ))),
                }
            }),
            Err(e) => Err(quack_core::error::Error::Analysis(format!(
                "workspace lock poisoned: {e}"
            ))),
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

        self.start_agent_turn(trimmed);
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

    fn run_direct_sql(&mut self, sql: &str) {
        let sql = sql.trim().to_owned();
        self.messages
            .push(Message::new(MessageRole::User, sql.clone()));
        self.last_sql = Some(sql.clone());
        self.state = AppState::RunningSql;

        let db = Arc::clone(&self.db);
        let tx = self.response_tx.clone();
        std::thread::spawn(move || {
            let result = run_sql_task(&db, &sql);
            drop(tx.send(result));
        });
    }

    fn start_agent_turn(&mut self, message: String) {
        self.messages
            .push(Message::new(MessageRole::User, message.clone()));
        self.state = AppState::Thinking;
        self.streaming_assistant = false;
        self.open_step = None;

        let policy = if self.allow_write {
            WritePolicy::Allow
        } else {
            WritePolicy::Ask
        };
        let (sink, rx) = events::channel();
        self.agent_events = Some(rx);

        let config = Arc::clone(&self.config);
        let db = Arc::clone(&self.db);
        let session_id = self.session_id.clone();
        tokio::spawn(async move {
            // run_turn emits TurnComplete or Failed itself; the returned
            // value is the same response, so it is not needed here.
            drop(quack_core::llm::run_turn(&config, db, &session_id, policy, &message, sink).await);
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

fn configure_textarea(textarea: &mut TextArea<'_>) {
    use ratatui::style::{Color, Style};

    textarea.set_cursor_line_style(Style::default());
    textarea.set_cursor_style(Style::default().fg(Color::Reset).bg(Color::White));
    textarea.set_placeholder_text("Ask a question, or type SQL...");
}

fn sources_footer(citations: &[quack_core::analysis::citations::Citation]) -> String {
    let lines: Vec<String> = citations
        .iter()
        .map(|c| format!("\n  [{}] {}", c.n, c.label()))
        .collect();
    format!("Sources:{}", lines.concat())
}

fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}

fn run_sql_task(db: &SharedDb, sql: &str) -> BackgroundResult {
    let db = match db.lock() {
        Ok(db) => db,
        Err(e) => return BackgroundResult::Error(format!("workspace lock poisoned: {e}")),
    };
    let started = std::time::Instant::now();
    match db.execute_query(sql) {
        Ok(results) => {
            let mut buf = Vec::new();
            let capped = results.clone_capped(200);
            if let Err(e) = capped.write_table(&mut buf) {
                return BackgroundResult::Error(format!("failed to render results: {e}"));
            }
            let mut text = String::from_utf8_lossy(&buf).into_owned();
            if results.rows.len() > 200 {
                let omitted = format!(
                    "... {} more rows not shown\n",
                    results.rows.len().saturating_sub(200)
                );
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

    if !path.is_absolute() && !cleaned.starts_with("./") {
        return None;
    }

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

    let file_type = ingestion::parser::detect_file_type(&filename);

    if file_type.is_structured() {
        let files_dir = config.workspace_files_dir(workspace_id);
        std::fs::create_dir_all(&files_dir)
            .map_err(|e| anyhow::anyhow!("failed to create workspace files dir: {e}"))?;
        let dest = files_dir.join(&filename);
        std::fs::write(&dest, &data)
            .map_err(|e| anyhow::anyhow!("failed to copy file to workspace: {e}"))?;
    }

    let ws_db = WorkspaceDb::open(config, workspace_id)
        .map_err(|e| anyhow::anyhow!("failed to open workspace: {e}"))?;

    let embedding_model = quack_core::llm::optional_embedding_model(config)?;

    let result = ingestion::ingest_file(
        config,
        &ws_db,
        workspace_id,
        &filename,
        &data,
        embedding_model.as_ref(),
    )
    .await
    .map_err(|e| anyhow::anyhow!("ingestion failed: {e}"))?;

    let table_part = result
        .table_name
        .as_ref()
        .map_or(String::new(), |t| format!(" as table \"{t}\""));
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
