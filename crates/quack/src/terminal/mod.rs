//! The interactive terminal session (ratatui).

mod app;
mod chart;
mod commands;
mod ui;

use anyhow::{Context, Result};
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::config::Config;

/// Everything a terminal session starts from: the resolved workspace, its
/// writer and reader pool, and the session to open.
pub(crate) struct SessionSetup {
    pub(crate) config: Config,
    pub(crate) workspace_name: String,
    pub(crate) workspace_id: String,
    pub(crate) db: SharedDb,
    pub(crate) reader_db: ReaderDb,
    pub(crate) session_id: String,
    /// `--allow-write`: the agent may modify the workspace without asking.
    pub(crate) allow_write: bool,
}

/// Run the terminal session against a resolved workspace until the user quits.
///
/// # Errors
///
/// Returns an error if the terminal cannot be initialized. Neither model
/// is required: without a chat model the session still runs SQL, ingests,
/// and every slash command, and a question says how to configure one;
/// without an embedding model document search is keyword-only, as in print
/// mode and the web.
pub(crate) async fn run(setup: SessionSetup) -> Result<()> {
    let mut tui_app = app::App::new(setup);
    tui_app.load_current_session().await?;
    tui_app.note_embedding_status().await?;

    let mut terminal = ratatui::try_init().context("failed to initialize terminal")?;
    // Mouse capture for wheel scrolling; ratatui's restore does not undo
    // it, so it is released by hand either way.
    let mouse =
        crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture).is_ok();
    let result = tui_app.run(&mut terminal).await;
    if mouse {
        drop(crossterm::execute!(
            std::io::stdout(),
            crossterm::event::DisableMouseCapture
        ));
    }
    drop(ratatui::try_restore());
    result
}
