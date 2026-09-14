//! The interactive terminal session (ratatui).

mod app;
mod chart;
mod ui;

use std::sync::Arc;

use anyhow::{Context, Result};
use quack_core::config::Config;

/// Run the terminal session against a resolved workspace until the user quits.
///
/// # Errors
///
/// Returns an error if no chat or embedding provider is configured, or if
/// the terminal cannot be initialized.
pub(crate) fn run(
    config: Config,
    workspace_name: String,
    workspace_id: String,
    allow_write: bool,
) -> Result<()> {
    if config.find_chat_provider().is_none() {
        anyhow::bail!(
            "no LLM provider configured — add a [providers.<name>] section \
             with 'model' set in {}",
            quack_core::config::config_file_path().display()
        );
    }
    if config.find_embedding_provider().is_none() {
        anyhow::bail!(
            "no embedding provider configured — add a [providers.<name>] section \
             with 'embedding_model' set in {}",
            quack_core::config::config_file_path().display()
        );
    }

    let provider_display = quack_core::llm::chat_model_display(&config);
    let config = Arc::new(config);

    let tui_app = app::App::new(
        workspace_name,
        workspace_id,
        provider_display,
        config,
        allow_write,
    );

    let mut terminal = ratatui::try_init().context("failed to initialize terminal")?;
    let result = tui_app.run(&mut terminal);
    drop(ratatui::try_restore());
    result
}
