//! Quack.app: a native window over the web UI, with the server inside the
//! process (`docs/desktop.md`). Every page, form, and asset is served by
//! `quack-server`'s router through the `quack://` scheme; the two routes
//! that stream (`query/stream`, `jobs/stream`) reach the page through
//! `bridge::stream`; a write the agent asks to run is answered in a native
//! dialog; files dropped on the window or chosen from the File menu are
//! uploaded by path.

use std::borrow::Cow;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use quack_core::config::Config;
use quack_core::crypto;
use quack_server::Embedded;
use tauri::http::{Request, Response, StatusCode};
use tauri::{AppHandle, DragDropEvent, Manager, RunEvent, WindowEvent};
use tauri_plugin_dialog::DialogExt;

mod bridge;
mod drop;
mod log;
mod menu;
mod permission;

use bridge::Streams;

/// The scheme the window loads the web UI from.
const SCHEME: &str = "quack";

fn main() -> ExitCode {
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
    let grace = config.server.shutdown_grace();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to start the async runtime")?;
    tauri::async_runtime::set(runtime.handle().clone());
    let embedded = runtime.block_on(Embedded::open(config))?;
    let app = tauri::Builder::default()
        .manage(embedded)
        .manage(Streams::default())
        .manage(logging)
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .menu(menu::build)
        .on_menu_event(|app, event| on_menu(app, &event))
        .on_window_event(|window, event| {
            if let WindowEvent::DragDrop(DragDropEvent::Drop { paths, .. }) = event {
                drop::add_to_current(window.app_handle(), paths.clone());
            }
        })
        .register_asynchronous_uri_scheme_protocol(SCHEME, |context, request, responder| {
            let embedded = context.app_handle().state::<Embedded>().inner().clone();
            tauri::async_runtime::spawn(async move {
                responder.respond(serve(&embedded, request).await);
            });
        })
        .invoke_handler(tauri::generate_handler![
            bridge::stream,
            bridge::stop_stream,
            drop::add_paths,
        ])
        .build(tauri::generate_context!())
        .context("failed to build the app")?;
    app.run(move |handle, event| {
        if let RunEvent::ExitRequested { .. } = event {
            // Jobs get the grace, then every writer checkpoints.
            tauri::async_runtime::block_on(handle.state::<Embedded>().shutdown(grace));
        }
    });
    Ok(())
}

/// One page, form, asset, or API request from the webview, answered by
/// the router. The body comes back whole; the streaming routes are not
/// loaded this way (`bridge`).
async fn serve(embedded: &Embedded, request: Request<Vec<u8>>) -> Response<Cow<'static, [u8]>> {
    let (parts, body) = request.into_parts();
    let path = parts
        .uri
        .path_and_query()
        .map_or("/", |p| p.as_str())
        .to_owned();
    let mut forwarded = Request::builder().method(parts.method).uri(&path);
    if let Some(headers) = forwarded.headers_mut() {
        *headers = parts.headers;
    }
    let forwarded = match forwarded.body(body) {
        Ok(forwarded) => forwarded,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Cow::Owned(e.to_string().into_bytes()))
                .unwrap_or_default();
        }
    };
    embedded
        .handle(forwarded)
        .await
        .map(|bytes| Cow::Owned(bytes.to_vec()))
}

/// The items the host answers itself: picking files and folders to add.
fn on_menu(app: &AppHandle, event: &tauri::menu::MenuEvent) {
    let picked = |app: &AppHandle, paths: Option<Vec<tauri_plugin_dialog::FilePath>>| {
        let paths: Vec<PathBuf> = paths
            .unwrap_or_default()
            .into_iter()
            .filter_map(|p| p.into_path().ok())
            .collect();
        if !paths.is_empty() {
            drop::add_to_current(app, paths);
        }
    };
    match event.id().as_ref() {
        menu::ADD_FILES => {
            let handle = app.clone();
            app.dialog()
                .file()
                .set_title("Add Files")
                .pick_files(move |paths| picked(&handle, paths));
        }
        menu::ADD_FOLDER => {
            let handle = app.clone();
            app.dialog()
                .file()
                .set_title("Add Folder")
                .pick_folders(move |paths| picked(&handle, paths));
        }
        _ => {}
    }
}
