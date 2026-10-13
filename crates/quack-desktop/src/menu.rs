//! The native menu bar: the app's own items, and the OS's standard ones.

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem, Submenu};
use tauri::{AppHandle, Wry};

/// File › Add Files…: files picked in the OS dialog go to the open workspace.
pub(crate) const ADD_FILES: &str = "file:add-files";
/// File › Add Folder…: every supported file under the chosen folders.
pub(crate) const ADD_FOLDER: &str = "file:add-folder";

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
            &PredefinedMenuItem::hide(app, None)?,
            &PredefinedMenuItem::quit(app, None)?,
        ],
    )?;
    let file = Submenu::with_items(
        app,
        "File",
        true,
        &[
            &MenuItem::with_id(app, ADD_FILES, "Add Files…", true, Some("CmdOrCtrl+O"))?,
            &MenuItem::with_id(
                app,
                ADD_FOLDER,
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
    let window = Submenu::with_items(
        app,
        "Window",
        true,
        &[
            &PredefinedMenuItem::minimize(app, None)?,
            &PredefinedMenuItem::fullscreen(app, None)?,
        ],
    )?;
    Menu::with_items(app, &[&quack, &file, &edit, &window])
}
