#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;

mod app;
mod chart;
mod providers;
mod ui;

#[derive(Parser)]
#[command(name = "quack-tui", version, about = "Interactive data analysis TUI")]
struct Cli {
    /// Workspace name (defaults to config value)
    #[arg(long, short = 'w')]
    workspace: Option<String>,

    /// Let the agent run statements that modify the workspace
    #[arg(long)]
    allow_write: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    quack_core::crypto::install_default_provider()
        .context("failed to install the aws-lc-rs crypto provider")?;

    let cli = Cli::parse();
    let config = quack_core::config::Config::load().context("failed to load configuration")?;

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

    let control = quack_core::storage::control::ControlPlane::open(&config)
        .await
        .context("failed to open control plane")?;

    let ws_name = cli
        .workspace
        .unwrap_or_else(|| config.general.default_workspace.clone());
    let workspace = control
        .find_or_create_workspace(&ws_name)
        .await
        .context("failed to resolve workspace")?;

    let provider_display = providers::provider_display_name(&config);
    let config = Arc::new(config);

    let tui_app = app::App::new(
        ws_name,
        workspace.id.clone(),
        provider_display,
        config,
        cli.allow_write,
    );

    let mut terminal = ratatui::try_init().context("failed to initialize terminal")?;
    let result = tui_app.run(&mut terminal);
    drop(ratatui::try_restore());
    result
}
