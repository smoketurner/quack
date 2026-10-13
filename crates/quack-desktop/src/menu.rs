//! The native menu bar. Items the frontend acts on (a view, a new
//! session, adding files) reach it as the Tauri event `menu` with the
//! item's id; the standard items are the OS's own.

use tauri::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu};
use tauri::{AppHandle, Emitter, Wry};

/// The event a chosen item goes out on, with its id as the payload.
pub(crate) const EVENT: &str = "menu";

/// Build the menu bar.
///
/// # Errors
///
/// When the runtime cannot create an item.
pub(crate) fn build(app: &AppHandle) -> tauri::Result<Menu<Wry>> {
    let quack = Submenu::with_items(
        app,
        "Quack",
        true,
        &[
            &PredefinedMenuItem::about(app, None, None)?,
            &PredefinedMenuItem::separator(app)?,
            &MenuItem::with_id(app, "settings", "Settings…", true, Some("CmdOrCtrl+,"))?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::quit(app, None)?,
        ],
    )?;
    let file = Submenu::with_items(
        app,
        "File",
        true,
        &[
            &MenuItem::with_id(
                app,
                "file:new-workspace",
                "New Workspace…",
                true,
                None::<&str>,
            )?,
            &MenuItem::with_id(
                app,
                "file:new-session",
                "New Session",
                true,
                Some("CmdOrCtrl+N"),
            )?,
            &PredefinedMenuItem::separator(app)?,
            &MenuItem::with_id(
                app,
                "file:add-files",
                "Add Files…",
                true,
                Some("CmdOrCtrl+O"),
            )?,
            &MenuItem::with_id(
                app,
                "file:add-folder",
                "Add Folder…",
                true,
                Some("CmdOrCtrl+Shift+O"),
            )?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::close_window(app, None)?,
        ],
    )?;
    let edit = Submenu::with_items(
        app,
        "Edit",
        true,
        &[
            &PredefinedMenuItem::undo(app, None)?,
            &PredefinedMenuItem::redo(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::cut(app, None)?,
            &PredefinedMenuItem::copy(app, None)?,
            &PredefinedMenuItem::paste(app, None)?,
            &PredefinedMenuItem::select_all(app, None)?,
        ],
    )?;
    let view = Submenu::with_items(
        app,
        "View",
        true,
        &[
            &MenuItem::with_id(
                app,
                "view:workspaces",
                "Workspaces",
                true,
                Some("CmdOrCtrl+1"),
            )?,
            &MenuItem::with_id(app, "view:chat", "Chat", true, Some("CmdOrCtrl+2"))?,
            &MenuItem::with_id(
                app,
                "view:documents",
                "Documents",
                true,
                Some("CmdOrCtrl+3"),
            )?,
            &MenuItem::with_id(app, "view:settings", "Settings", true, Some("CmdOrCtrl+4"))?,
        ],
    )?;
    let window = Submenu::with_items(
        app,
        "Window",
        true,
        &[
            &PredefinedMenuItem::minimize(app, None)?,
            &PredefinedMenuItem::fullscreen(app, None)?,
        ],
    )?;
    Menu::with_items(app, &[&quack, &file, &edit, &view, &window])
}

/// A chosen item, handed to the frontend by id.
pub(crate) fn on_event(app: &AppHandle, event: &MenuEvent) {
    let id: &str = event.id().as_ref();
    if let Err(e) = app.emit(EVENT, id) {
        tracing::debug!(error = %e, id, "a menu choice could not reach the frontend");
    }
}
