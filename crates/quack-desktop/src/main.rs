//! The desktop app: a Tauri window over `quack-core`, the fourth client
//! of the engine beside the terminal, print mode, and the server
//! (`docs/desktop.md`). The frontend in `ui/` calls the commands below;
//! long work runs on the process's one job queue and reaches it as the
//! `job` event.

// `#[tauri::command]` expands to argument-parsing glue that reaches for
// `unreachable!` on a path the macro has already ruled out; every async
// command carries one. The crate's own code never uses the macro.
#![expect(
    clippy::unreachable,
    reason = "inside the code `#[tauri::command]` generates for each async command"
)]

use std::io::Write;
use std::process::ExitCode;

use anyhow::{Context, Result};
use quack_core::config::Config;
use quack_core::crypto;
use tauri::{Manager, RunEvent};

mod app;
mod audit;
mod chat;
mod documents;
mod error;
mod jobs;
mod log;
mod menu;
mod permission;
mod schema;
mod settings;
mod workspace;

use app::AppState;

fn main() -> ExitCode {
    // `quack-desktop --schema` prints the frontend's type schema and exits
    // (`make ui-types`); nothing else is a command-line option.
    if std::env::args().nth(1).as_deref() == Some("--schema") {
        return match std::io::stdout().write_all(schema::json().as_bytes()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ExitCode::FAILURE,
        };
    }
    match start() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // Before the window exists there is nothing but stderr and the log.
            tracing::error!(error = format!("{e:#}"), "the app could not start");
            let mut stderr = std::io::stderr();
            if writeln!(stderr, "quack-desktop: {e:#}").is_err() {
                // Nothing left to say it on.
            }
            ExitCode::FAILURE
        }
    }
}

fn start() -> Result<()> {
    crypto::install_default_provider()
        .context("failed to install the aws-lc-rs crypto provider")?;
    let config = Config::load().context("failed to load configuration")?;
    let logging = log::init(&config)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to start the async runtime")?;
    tauri::async_runtime::set(runtime.handle().clone());
    let state = runtime.block_on(AppState::open(config))?;
    let app = tauri::Builder::default()
        .manage(state)
        .manage(logging)
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .menu(menu::build)
        .on_menu_event(|app, event| menu::on_event(app, &event))
        .invoke_handler(tauri::generate_handler![
            workspace::list_workspaces,
            workspace::create_workspace,
            workspace::rename_workspace,
            workspace::delete_workspace,
            workspace::open_workspace,
            workspace::close_workspace,
            chat::list_sessions,
            chat::new_session,
            chat::session_messages,
            chat::set_session_mode,
            chat::set_allow_writes,
            chat::start_turn,
            chat::cancel_turn,
            documents::list_documents,
            documents::add_paths,
            documents::delete_document,
            jobs::list_jobs,
            jobs::cancel_job,
            settings::settings,
        ])
        .setup(|app| {
            let jobs = app.state::<AppState>().jobs.clone();
            jobs::forward(app.handle().clone(), jobs);
            Ok(())
        })
        .build(tauri::generate_context!())
        .context("failed to build the app")?;
    app.run(|handle, event| {
        if let RunEvent::ExitRequested { .. } = event {
            // Jobs get the grace, then every writer checkpoints.
            tauri::async_runtime::block_on(handle.state::<AppState>().shutdown());
        }
    });
    Ok(())
}
