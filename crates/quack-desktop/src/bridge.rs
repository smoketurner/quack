//! The two things the web UI streams, carried to the webview over Tauri's
//! IPC: a custom protocol answers a request whole, so a server-sent event
//! route (`query/stream`, `jobs/stream`) is read here, event by event,
//! and each event is sent down a [`Channel`] the page listens on. A write
//! the agent wants to run (`permission_required`) is answered in the
//! native dialog before the page hears of it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use futures::StreamExt;
use quack_server::Embedded;
use serde::{Deserialize, Serialize};
use tauri::http::{Request, StatusCode, header};
use tauri::ipc::Channel;
use tauri::{AppHandle, Manager};
use tokio_util::sync::CancellationToken;

use crate::permission;

/// What the page receives, in order: the status, then each event, then
/// the close; an error in place of the events when the route refused.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Frame {
    Open { status: u16 },
    Event { event: String, data: String },
    Error { message: String },
    Close,
}

/// One server-sent event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SseEvent {
    pub(crate) event: String,
    pub(crate) data: String,
}

/// Parses `text/event-stream` as the bytes arrive.
#[derive(Debug, Default)]
pub(crate) struct SseParser {
    buffer: String,
}

impl SseParser {
    /// Feed `chunk`; the events it completed, in order. A block with no
    /// `data:` line is a comment or a retry hint and yields nothing.
    pub(crate) fn feed(&mut self, chunk: &str) -> Vec<SseEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();
        while let Some(end) = self.buffer.find("\n\n") {
            let block: String = self.buffer.drain(..end.saturating_add(2)).collect();
            let mut event = String::from("message");
            let mut data: Vec<&str> = Vec::new();
            for line in block.lines() {
                if let Some(name) = line.strip_prefix("event:") {
                    name.trim().clone_into(&mut event);
                } else if let Some(value) = line.strip_prefix("data:") {
                    data.push(value.strip_prefix(' ').unwrap_or(value));
                }
            }
            if !data.is_empty() {
                events.push(SseEvent {
                    event,
                    data: data.join("\n"),
                });
            }
        }
        events
    }
}

/// The streams the page has open, by the id it gave each, so Stop can end
/// one; a page that goes away ends its streams through the dead channel.
#[derive(Default)]
pub(crate) struct Streams(Mutex<HashMap<String, CancellationToken>>);

impl Streams {
    fn start(&self, id: String) -> CancellationToken {
        let token = CancellationToken::new();
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id, token.clone());
        token
    }

    fn end(&self, id: &str) -> Option<CancellationToken> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(id)
    }
}

/// The write prompt as `query/stream` sends it; only what answering needs.
#[derive(Debug, Deserialize)]
struct PermissionEvent {
    request: String,
    session_id: String,
    sql: String,
    notice: Option<String>,
}

/// What the page opens: the stream's id (for Stop), the route, and the
/// JSON body of a `POST`.
#[derive(Debug, Deserialize)]
pub(crate) struct Open {
    id: String,
    method: String,
    url: String,
    body: Option<String>,
}

/// Open the route on the embedded server and send its events to
/// `on_frame` until it ends or [`stop_stream`] is called with its id.
#[tauri::command]
pub(crate) async fn stream(
    app: AppHandle,
    open: Open,
    on_frame: Channel<Frame>,
) -> Result<(), String> {
    let id = open.id.clone();
    let stop = app.state::<Streams>().start(id.clone());
    let embedded = app.state::<Embedded>().inner().clone();
    let outcome = Box::pin(run(&app, &embedded, &stop, open, &on_frame)).await;
    app.state::<Streams>().end(&id);
    if on_frame.send(Frame::Close).is_err() {
        tracing::debug!("the page left before the stream closed");
    }
    outcome
}

/// End the stream the page opened as `id`. `false` when it had ended.
#[tauri::command]
#[expect(
    clippy::needless_pass_by_value,
    reason = "a Tauri command takes its arguments by value"
)]
pub(crate) fn stop_stream(app: AppHandle, id: String) -> bool {
    app.state::<Streams>().end(&id).is_some_and(|token| {
        token.cancel();
        true
    })
}

async fn run(
    app: &AppHandle,
    embedded: &Embedded,
    stop: &CancellationToken,
    open: Open,
    on_frame: &Channel<Frame>,
) -> Result<(), String> {
    let Open {
        method, url, body, ..
    } = open;
    let mut request = Request::builder()
        .method(method.as_str())
        .uri(&url)
        .header(header::ACCEPT, "text/event-stream");
    if body.is_some() {
        request = request.header(header::CONTENT_TYPE, "application/json");
    }
    let request = request
        .body(body.unwrap_or_default().into_bytes())
        .map_err(|e| format!("bad request: {e}"))?;
    let (parts, mut chunks) = embedded.stream(request).await.into_parts();
    on_frame
        .send(Frame::Open {
            status: parts.status.as_u16(),
        })
        .map_err(|e| e.to_string())?;
    if !parts.status.is_success() {
        let mut text = String::new();
        while let Some(Ok(chunk)) = chunks.next().await {
            text.push_str(&String::from_utf8_lossy(&chunk));
        }
        on_frame
            .send(Frame::Error {
                message: error_message(parts.status, &text),
            })
            .map_err(|e| e.to_string())?;
        return Ok(());
    }
    let mut parser = SseParser::default();
    let answers = Arc::new(tokio::sync::Mutex::new(()));
    loop {
        let chunk = tokio::select! {
            () = stop.cancelled() => break,
            chunk = chunks.next() => chunk,
        };
        let Some(chunk) = chunk else { break };
        let chunk = chunk.map_err(|e| e.to_string())?;
        for event in parser.feed(&String::from_utf8_lossy(&chunk)) {
            if event.event == "permission_required" {
                answer_in_dialog(app, embedded, &url, &event.data, on_frame, &answers);
            }
            let sent = on_frame.send(Frame::Event {
                event: event.event,
                data: event.data,
            });
            if sent.is_err() {
                // The page is gone; so is anyone to show the rest to.
                return Ok(());
            }
        }
    }
    Ok(())
}

/// The message the page shows for a refused stream: the API's own
/// `error`, else the status.
fn error_message(status: StatusCode, body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("error")?.as_str().map(str::to_owned))
        .unwrap_or_else(|| {
            status
                .canonical_reason()
                .map_or_else(|| status.to_string(), str::to_owned)
        })
}

/// Show the dialog for a write prompt, give the server the answer, and
/// tell the page what was said. Runs beside the stream, which goes on.
fn answer_in_dialog(
    app: &AppHandle,
    embedded: &Embedded,
    stream_url: &str,
    data: &str,
    on_frame: &Channel<Frame>,
    one_at_a_time: &Arc<tokio::sync::Mutex<()>>,
) {
    let prompt: PermissionEvent = match serde_json::from_str(data) {
        Ok(prompt) => prompt,
        Err(e) => {
            tracing::warn!(error = %e, "a write prompt could not be read; the page's card stands");
            return;
        }
    };
    let Some(workspace) = workspace_of(stream_url).map(str::to_owned) else {
        return;
    };
    let (app, embedded, on_frame, serial) = (
        app.clone(),
        embedded.clone(),
        on_frame.clone(),
        Arc::clone(one_at_a_time),
    );
    tauri::async_runtime::spawn(async move {
        let decision = {
            // Dialogs show one at a time, in the order the prompts arrived.
            let _turn = serial.lock().await;
            permission::show(&app, &prompt.sql, prompt.notice.as_deref()).await
        };
        let url = format!(
            "/api/v1/workspaces/{workspace}/sessions/{}/permissions/{}",
            prompt.session_id, prompt.request
        );
        let answer = serde_json::json!({ "decision": decision.as_str() }).to_string();
        let request = Request::builder()
            .method("POST")
            .uri(&url)
            .header(header::CONTENT_TYPE, "application/json")
            .body(answer.into_bytes());
        let reply = match request {
            Ok(request) => {
                let response = embedded.handle(request).await;
                if response.status().is_success() {
                    decision.reply().to_owned()
                } else {
                    let text = String::from_utf8_lossy(response.body());
                    format!("Not run: {}", error_message(response.status(), &text))
                }
            }
            Err(e) => format!("Not run: {e}"),
        };
        let told = on_frame.send(Frame::Event {
            event: String::from("permission_answered"),
            data: serde_json::json!({ "request": prompt.request, "reply": reply }).to_string(),
        });
        if told.is_err() {
            tracing::debug!("the page left before the write prompt's answer");
        }
    });
}

/// The workspace id in an API path (`/api/v1/workspaces/{id}/...`).
pub(crate) fn workspace_of(url: &str) -> Option<&str> {
    let rest = url.strip_prefix("/api/v1/workspaces/")?;
    let id = rest.split(['/', '?']).next()?;
    (!id.is_empty()).then_some(id)
}

#[cfg(test)]
mod tests {
    use tauri::http::StatusCode;

    use super::{SseEvent, SseParser, error_message, workspace_of};

    /// Events arrive split anywhere; each is yielded once its blank line
    /// does, with its name and multi-line data.
    #[test]
    fn events_are_parsed_across_chunk_boundaries() {
        let mut parser = SseParser::default();
        assert!(parser.feed("event: text\ndata: Hel").is_empty());
        assert_eq!(
            parser.feed("lo\n\n: comment\n\ndata: a\ndata: b\n\nevent: x\n"),
            vec![
                SseEvent {
                    event: String::from("text"),
                    data: String::from("Hello"),
                },
                SseEvent {
                    event: String::from("message"),
                    data: String::from("a\nb"),
                },
            ]
        );
        assert_eq!(
            parser.feed("data: {\"n\":1}\n\n"),
            vec![SseEvent {
                event: String::from("x"),
                data: String::from("{\"n\":1}"),
            }]
        );
    }

    #[test]
    fn a_refusal_shows_the_apis_error_or_the_status() {
        assert_eq!(
            error_message(
                StatusCode::FORBIDDEN,
                r#"{"error":"no write access","code":"forbidden"}"#
            ),
            "no write access"
        );
        assert_eq!(error_message(StatusCode::NOT_FOUND, "nope"), "Not Found");
    }

    #[test]
    fn the_workspace_is_read_from_the_stream_url() {
        assert_eq!(
            workspace_of("/api/v1/workspaces/abc/query/stream"),
            Some("abc")
        );
        assert_eq!(workspace_of("/api/v1/workspaces/abc"), Some("abc"));
        assert_eq!(workspace_of("/w/abc/chat"), None);
        assert_eq!(workspace_of("/api/v1/workspaces//x"), None);
    }
}
