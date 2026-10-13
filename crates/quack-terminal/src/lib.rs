//! The interactive terminal session (ratatui).

mod app;
mod chart;
mod clipboard;
mod commands;
mod markdown;
mod picker;
mod selection;
mod sql;
mod ui;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use quack_core::analysis::permissions::Approvals;
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::engine::Engine;
use quack_core::ids::{SessionId, WorkspaceId};

/// Everything a terminal session starts from: the process's engine (its
/// configuration and queue), the resolved workspace, its writer and
/// reader pool (the engine's handle, in the binary; a test's own in
/// tests), and the session to open.
pub struct SessionSetup {
    pub engine: Arc<Engine>,
    /// The web UI, REST API, and MCP the binary serves beside the session
    /// on loopback, from the same engine.
    pub web: WebStatus,
    pub workspace_name: String,
    pub workspace_id: WorkspaceId,
    pub db: SharedDb,
    pub reader_db: ReaderDb,
    pub session_id: SessionId,
    /// `Allow` with `--allow-write`, else `Ask`.
    pub writes: WritePolicy,
    /// Writes that `query` turns asked of this process over the workspace
    /// socket (`quack mcp` in another shell) wait on: the session answers
    /// them at its own `y`/`n`/`a` prompt.
    pub approvals: Approvals,
}

/// Whether the binary serves the web UI, REST API, and MCP beside the
/// session, and where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebStatus {
    /// Listening on loopback; the transcript's first lines say where, and
    /// that address's console lists the one workspace served.
    Listening(SocketAddr),
    /// It could not start (the port is taken, a client registration is
    /// missing); the transcript says so once, and the session runs without.
    Unavailable(String),
}

/// How long a quitting session gives its jobs, and the server beside it
/// its drain, to end.
pub const QUIT_GRACE: Duration = Duration::from_secs(3);

/// Run the terminal session against a resolved workspace until the user quits.
///
/// # Errors
///
/// Returns an error if the terminal cannot be initialized. Neither model
/// is required: without a chat model the session still runs SQL, ingests,
/// and every slash command, and a question says how to configure one;
/// without an embedding model document search is keyword-only, as in print
/// mode and the web.
pub async fn run(setup: SessionSetup) -> Result<()> {
    let mut tui_app = app::App::new(setup);
    tui_app.load_current_session().await?;
    tui_app.load_input_history().await?;
    tui_app.note_embedding_status().await?;
    tui_app.load_sql_schema().await?;

    let mut terminal = ratatui::try_init().context("failed to initialize terminal")?;
    // Mouse capture for wheel scrolling, bracketed paste so a dropped
    // file's path arrives as one event; ratatui's restore undoes neither,
    // so both are released by hand either way.
    let mouse =
        crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture).is_ok();
    let paste =
        crossterm::execute!(std::io::stdout(), crossterm::event::EnableBracketedPaste).is_ok();
    let result = tui_app.run(&mut terminal).await;
    if paste {
        drop(crossterm::execute!(
            std::io::stdout(),
            crossterm::event::DisableBracketedPaste
        ));
    }
    if mouse {
        drop(crossterm::execute!(
            std::io::stdout(),
            crossterm::event::DisableMouseCapture
        ));
    }
    drop(ratatui::try_restore());
    result
}
