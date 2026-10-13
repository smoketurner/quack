//! Files and folders by path: an OS drop on the window, or File › Add
//! Files… and Add Folder…. Each file is uploaded to the embedded server
//! the way the Documents page's form uploads it, so it is spooled, queued
//! in the workspace's ingest lane, and shown in the job strip; a folder
//! contributes every supported file under it.

use std::path::{Path, PathBuf};

use quack_core::ingestion::tree::Tree;
use quack_server::Embedded;
use tauri::http::{Request, header};
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};

/// What a batch of paths came to.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Added {
    pub(crate) queued: usize,
    pub(crate) failed: Vec<String>,
}

impl Added {
    /// One line for the person.
    pub(crate) fn summary(&self) -> String {
        match (self.queued, self.failed.len()) {
            (n, 0) => format!("{n} files queued"),
            (n, f) => format!("{n} files queued, {f} failed: {}", self.failed.join("; ")),
        }
    }
}

/// Upload `paths` into `workspace`: every file, and every supported file
/// under each folder.
#[tauri::command]
pub(crate) async fn add_paths(
    app: AppHandle,
    workspace: String,
    paths: Vec<String>,
) -> Result<String, String> {
    let embedded = app.state::<Embedded>().inner().clone();
    let added = add(
        &embedded,
        &workspace,
        paths.iter().map(PathBuf::from).collect(),
    )
    .await;
    Ok(added.summary())
}

pub(crate) async fn add(embedded: &Embedded, workspace: &str, paths: Vec<PathBuf>) -> Added {
    let mut added = Added::default();
    for path in paths {
        let files = if path.is_dir() {
            match Tree::walk(&path) {
                Ok(tree) => tree.files.into_iter().map(|f| f.path).collect(),
                Err(e) => {
                    added.failed.push(format!("{}: {e}", path.display()));
                    continue;
                }
            }
        } else {
            vec![path]
        };
        for file in files {
            match upload(embedded, workspace, &file).await {
                Ok(()) => added.queued = added.queued.saturating_add(1),
                Err(e) => added.failed.push(format!("{}: {e}", file.display())),
            }
        }
    }
    added
}

async fn upload(embedded: &Embedded, workspace: &str, file: &Path) -> Result<(), String> {
    let name = file
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| String::from("the file has no name"))?;
    let bytes = tokio::fs::read(file).await.map_err(|e| e.to_string())?;
    let (content_type, body) = multipart(name, &bytes);
    let request = Request::builder()
        .method("POST")
        .uri(format!("/api/v1/workspaces/{workspace}/documents"))
        .header(header::CONTENT_TYPE, content_type)
        .body(body)
        .map_err(|e| e.to_string())?;
    let response = embedded.handle(request).await;
    if response.status().is_success() {
        return Ok(());
    }
    let text = String::from_utf8_lossy(response.body());
    Err(serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|v| v.get("error")?.as_str().map(str::to_owned))
        .unwrap_or_else(|| response.status().to_string()))
}

/// One `file` part, as the upload form sends it.
pub(crate) fn multipart(name: &str, bytes: &[u8]) -> (String, Vec<u8>) {
    let boundary = format!("quack-{}", uuid_like());
    let safe_name = name.replace(['"', '\r', '\n'], "_");
    let mut body = Vec::with_capacity(bytes.len().saturating_add(256));
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"file\"; filename=\"{safe_name}\"\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

/// A boundary no file is likely to contain: the time and the thread.
fn uuid_like() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("{now:x}-{:?}", std::thread::current().id()).replace(['(', ')', ' '], "")
}

/// The workspace the main window shows (`/w/{id}/...`), if any.
pub(crate) fn current_workspace(app: &AppHandle) -> Option<String> {
    let window = app.get_webview_window("main")?;
    let url = window.url().ok()?;
    let mut segments = url.path_segments()?;
    (segments.next()? == "w")
        .then(|| segments.next().map(str::to_owned))
        .flatten()
        .filter(|id| !id.is_empty())
}

/// Dropped or picked paths go to the window's workspace; without one, the
/// person is told to open one first.
pub(crate) fn add_to_current(app: &AppHandle, paths: Vec<PathBuf>) {
    let Some(workspace) = current_workspace(app) else {
        app.dialog()
            .message("Open a workspace, then drop files on it.")
            .title("quack")
            .kind(MessageDialogKind::Info)
            .show(|_| {});
        return;
    };
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let embedded = app.state::<Embedded>();
        let added = add(&embedded, &workspace, paths).await;
        tracing::info!(
            workspace,
            queued = added.queued,
            failed = added.failed.len(),
            "paths added"
        );
        if !added.failed.is_empty() {
            app.dialog()
                .message(added.summary())
                .title("Some files were not added")
                .kind(MessageDialogKind::Warning)
                .show(|_| {});
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{Added, multipart};

    #[test]
    fn the_part_is_one_file_field_under_its_boundary() {
        let (content_type, body) = multipart("a\"b.csv", b"x,y\n1,2\n");
        let boundary = content_type
            .strip_prefix("multipart/form-data; boundary=")
            .unwrap_or_default();
        let text = String::from_utf8_lossy(&body);
        assert!(text.starts_with(&format!("--{boundary}\r\n")));
        assert!(text.contains("name=\"file\"; filename=\"a_b.csv\""));
        assert!(text.contains("\r\n\r\nx,y\n1,2\n\r\n"));
        assert!(text.ends_with(&format!("--{boundary}--\r\n")));
    }

    #[test]
    fn the_summary_counts_and_names_failures() {
        assert_eq!(Added::default().summary(), "0 files queued");
        let added = Added {
            queued: 2,
            failed: vec![String::from("a: unreadable")],
        };
        assert_eq!(added.summary(), "2 files queued, 1 failed: a: unreadable");
    }
}
