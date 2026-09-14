use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui_textarea::TextArea;
use tokio::sync::mpsc;

use rig::client::CompletionClient;

use quack_core::analysis::agent;
use quack_core::analysis::policy::WritePolicy;
use quack_core::config::Config;
use quack_core::ingestion;
use quack_core::storage::workspace::WorkspaceDb;

use crate::chart::ChartData;
use crate::providers;
use crate::ui;

const TICK_RATE_MS: u64 = 50;

const WELCOME_TEXT: &str = "\
Welcome to quack!

Ask questions about your data or type SQL queries directly.
Drop a file here to load it (CSV, JSON, Parquet, PDF, TXT, MD).
Type /help for available commands.";

const HELP_TEXT: &str = "\
Commands:
  /help             Show this help message
  /clear            Clear messages and chart
  /quit, /exit      Exit quack
  /workspace        Show current workspace

Writes:
  The agent may run SELECT queries freely. Statements that modify the
  workspace are refused unless quack-tui was started with --allow-write.

Shortcuts:
  Enter             Send message
  Up/Down           Browse input history
  PageUp/PageDown   Scroll messages
  Ctrl+U            Clear input line
  Ctrl+L            Clear screen
  Ctrl+C            Quit";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AppState {
    Idle,
    Thinking,
    Ingesting,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MessageRole {
    User,
    Assistant,
    System,
    Error,
}

#[derive(Debug, Clone)]
pub(crate) struct Message {
    pub(crate) role: MessageRole,
    pub(crate) content: String,
}

impl Message {
    fn system(content: &str) -> Self {
        Self {
            role: MessageRole::System,
            content: content.to_owned(),
        }
    }

    fn user(content: String) -> Self {
        Self {
            role: MessageRole::User,
            content,
        }
    }

    fn assistant(content: String) -> Self {
        Self {
            role: MessageRole::Assistant,
            content,
        }
    }

    fn error(content: String) -> Self {
        Self {
            role: MessageRole::Error,
            content,
        }
    }
}

enum BackgroundResult {
    Chat {
        content: String,
        chart_spec: Option<serde_json::Value>,
    },
    Ingested {
        summary: String,
    },
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
    pub(crate) current_chart: Option<ChartData>,
    input_history: Vec<String>,
    history_cursor: Option<usize>,
    config: Arc<Config>,
    workspace_id: String,
    allow_write: bool,
    response_rx: mpsc::UnboundedReceiver<BackgroundResult>,
    response_tx: mpsc::UnboundedSender<BackgroundResult>,
}

impl App {
    pub(crate) fn new(
        workspace_name: String,
        workspace_id: String,
        provider_display: String,
        config: Arc<Config>,
        allow_write: bool,
    ) -> Self {
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
            current_chart: None,
            input_history: Vec::new(),
            history_cursor: None,
            config,
            workspace_id,
            allow_write,
            response_rx,
            response_tx,
        };
        app.messages.push(Message::system(WELCOME_TEXT));
        app
    }

    pub(crate) fn run(mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<()> {
        let tick_rate = Duration::from_millis(TICK_RATE_MS);

        loop {
            terminal.draw(|frame| ui::draw(frame, &self))?;

            while let Ok(result) = self.response_rx.try_recv() {
                self.handle_background_result(result);
            }

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

        Ok(())
    }

    fn handle_key_event(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        match (code, modifiers) {
            (KeyCode::Char('c' | 'q'), KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            (KeyCode::Char('l'), KeyModifiers::CONTROL) => {
                self.messages.clear();
                self.messages.push(Message::system(WELCOME_TEXT));
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
        let (cmd, _args) = input
            .split_once(' ')
            .map_or((input, ""), |(c, a)| (c, a.trim()));

        match cmd {
            "/quit" | "/exit" | "/q" => {
                self.should_quit = true;
            }
            "/clear" => {
                self.messages.clear();
                self.messages.push(Message::system(WELCOME_TEXT));
                self.current_chart = None;
                self.scroll_offset = 0;
            }
            "/help" | "/?" => {
                self.messages.push(Message::system(HELP_TEXT));
            }
            "/workspace" => {
                self.messages.push(Message::system(&format!(
                    "Workspace: {} ({})",
                    self.workspace_name, self.workspace_id
                )));
            }
            other => {
                self.messages
                    .push(Message::error(format!("unknown command: {other}")));
            }
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

        let config = Arc::clone(&self.config);
        let workspace_id = self.workspace_id.clone();
        let tx = self.response_tx.clone();

        if let Some(path) = detect_file_path(&trimmed) {
            self.messages
                .push(Message::system(&format!("Ingesting {}", path.display())));
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
        } else {
            self.messages.push(Message::user(trimmed.clone()));
            self.state = AppState::Thinking;
            let policy = if self.allow_write {
                WritePolicy::Allow
            } else {
                WritePolicy::Deny
            };

            tokio::spawn(async move {
                let result = run_agent_task(config, workspace_id, policy, trimmed).await;
                drop(tx.send(result));
            });
        }
    }

    fn handle_background_result(&mut self, result: BackgroundResult) {
        match result {
            BackgroundResult::Chat {
                content,
                chart_spec,
            } => {
                if let Some(spec) = chart_spec {
                    self.current_chart = ChartData::from_echart_spec(&spec);
                    if self.current_chart.is_none() {
                        self.messages.push(Message::assistant(content));
                    }
                } else {
                    self.messages.push(Message::assistant(content));
                }
            }
            BackgroundResult::Ingested { summary } => {
                self.messages.push(Message::system(&summary));
            }
            BackgroundResult::Error(err) => {
                self.messages.push(Message::error(err));
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
    textarea.set_placeholder_text("Type a question or SQL query...");
}

async fn run_agent_task(
    config: Arc<Config>,
    workspace_id: String,
    policy: WritePolicy,
    message: String,
) -> BackgroundResult {
    match run_agent_inner(&config, &workspace_id, policy, &message).await {
        Ok((content, chart_spec)) => BackgroundResult::Chat {
            content,
            chart_spec,
        },
        Err(e) => BackgroundResult::Error(format!("{e:#}")),
    }
}

async fn run_agent_inner(
    config: &Config,
    workspace_id: &str,
    policy: WritePolicy,
    message: &str,
) -> Result<(String, Option<serde_json::Value>)> {
    let ws_db = WorkspaceDb::open(config, workspace_id)
        .map_err(|e| anyhow::anyhow!("failed to open workspace: {e}"))?;

    let (_, chat_config) = config
        .find_chat_provider()
        .ok_or_else(|| anyhow::anyhow!("no chat provider configured"))?;

    let chat_model_name = chat_config
        .model
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("no chat model configured"))?
        .to_owned();

    let (embedding_model, _) = providers::build_rig_embedding_model(config)?;

    let response = match chat_config.provider_type.as_str() {
        "ollama" => {
            let client = providers::build_ollama_client(chat_config)?;
            let model = client.completion_model(&chat_model_name);
            agent::run_analysis(
                ws_db,
                model,
                embedding_model,
                &config.analysis,
                &config.retrieval,
                policy.clone(),
                message,
            )
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?
        }
        "openai" => {
            let client = providers::build_openai_client(chat_config)?;
            let model = client.completion_model(&chat_model_name);
            agent::run_analysis(
                ws_db,
                model,
                embedding_model,
                &config.analysis,
                &config.retrieval,
                policy.clone(),
                message,
            )
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?
        }
        "anthropic" => {
            let client = providers::build_anthropic_client(chat_config)?;
            let model = client.completion_model(&chat_model_name);
            agent::run_analysis(
                ws_db,
                model,
                embedding_model,
                &config.analysis,
                &config.retrieval,
                policy.clone(),
                message,
            )
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?
        }
        other => anyhow::bail!("unsupported provider type: {other}"),
    };

    Ok((response.content, response.chart_spec))
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

    let embedding_model = providers::build_embedding_model(config)?;

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
