//! The interactive terminal session (ratatui).

mod app;
mod chart;
mod ui;

use std::sync::Arc;

use anyhow::{Context, Result};
use quack_core::analysis::tools::SharedDb;
use quack_core::config::Config;

/// Run the terminal session against a resolved workspace until the user quits.
///
/// # Errors
///
/// Returns an error if no chat provider is configured or the terminal
/// cannot be initialized. An embedding provider is optional: without one
/// document search is keyword-only, as in print mode and the web.
pub(crate) fn run(
    config: Config,
    workspace_name: String,
    workspace_id: String,
    db: SharedDb,
    session_id: String,
    allow_write: bool,
) -> Result<()> {
    config
        .chat_model_ref()
        .context("the terminal session needs a chat model")?;

    let provider_display = quack_core::llm::chat_model_display(&config);
    let config = Arc::new(config);

    let tui_app = app::App::new(
        workspace_name,
        workspace_id,
        provider_display,
        config,
        db,
        session_id,
        allow_write,
    )?;

    let mut terminal = ratatui::try_init().context("failed to initialize terminal")?;
    // Mouse capture for wheel scrolling; ratatui's restore does not undo
    // it, so it is released by hand either way.
    let mouse =
        crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture).is_ok();
    let result = tui_app.run(&mut terminal);
    if mouse {
        drop(crossterm::execute!(
            std::io::stdout(),
            crossterm::event::DisableMouseCapture
        ));
    }
    drop(ratatui::try_restore());
    result
}
