//! The workspace as an Open Knowledge Format bundle: `GET .../okf` answers
//! a tar archive of Markdown files, audited as `export` because it moves
//! content across the boundary. Import is `POST .../documents` with a tar
//! body (see `documents::upload`).
//!
//! The archive streams: the export writes it on the reader's thread into
//! a bounded channel, and the response body drains the channel, so
//! neither side holds the bundle. A client that goes away stops the
//! export at its next write.

use std::io::{self, Write};
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use quack_core::ids::WorkspaceId;
use quack_core::okf::{self, TarSink};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use tokio::sync::mpsc;

use crate::server::auth::{Access, Identity, Need};
use crate::server::error::ApiResult;
use crate::server::state::App;

/// Bytes gathered before one is sent to the body.
const CHUNK_BYTES: usize = 64 * 1024;
/// Chunks the export may run ahead of the client.
const CHUNKS_IN_FLIGHT: usize = 8;

pub(crate) async fn export(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let reader = app.reader_db(&id).await?;
    let name = access.workspace.name.clone();
    let filename = format!("{}.okf.tar", okf::slug(&name));
    let (tx, rx) = mpsc::channel::<io::Result<Bytes>>(CHUNKS_IN_FLIGHT);
    let failed = tx.clone();
    let audit_app = Arc::clone(&app);
    tokio::spawn(async move {
        let written = reader
            .with_db(move |db| {
                let mut sink = TarSink::new(BodyWriter::new(tx));
                let summary = okf::export(db, &name, &mut sink)?;
                sink.finish()?.flush()?;
                Ok(summary)
            })
            .await;
        // The export's own sender is gone with its closure; this one
        // carries a failure to the body, then closes it.
        let (outcome, detail) = match written {
            Ok(summary) => (
                Outcome::Allowed,
                serde_json::json!({ "format": "okf", "files": summary.files }),
            ),
            Err(e) => {
                tracing::warn!(workspace = %id, error = %e, "okf export failed partway");
                drop(failed.send(Err(io::Error::other(e.to_string()))).await);
                (
                    Outcome::Error,
                    serde_json::json!({ "format": "okf", "error": e.to_string() }),
                )
            }
        };
        drop(failed);
        let recorded = access
            .audit(
                &audit_app,
                AuditAction::Export,
                Some(ResourceKind::Workspace.id(id.as_str())),
                outcome,
                Some(detail),
            )
            .await;
        if let Err(e) = recorded {
            tracing::error!(workspace = %id, error = %e.message, "could not audit an okf export");
        }
    });
    let body = Body::from_stream(futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (chunk, rx))
    }));
    Ok((
        [
            (header::CONTENT_TYPE, String::from("application/x-tar")),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{filename}\""),
            ),
        ],
        body,
    )
        .into_response())
}

/// A `Write` on the export's blocking thread that sends what it gathers
/// to the response body in `CHUNK_BYTES` pieces. A closed channel (the
/// client went away) is a broken pipe, which ends the export.
struct BodyWriter {
    tx: mpsc::Sender<io::Result<Bytes>>,
    pending: Vec<u8>,
}

impl BodyWriter {
    fn new(tx: mpsc::Sender<io::Result<Bytes>>) -> Self {
        Self {
            tx,
            pending: Vec::with_capacity(CHUNK_BYTES),
        }
    }

    fn send_pending(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let chunk = Bytes::from(std::mem::replace(
            &mut self.pending,
            Vec::with_capacity(CHUNK_BYTES),
        ));
        self.tx
            .blocking_send(Ok(chunk))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "the client went away"))
    }
}

impl Write for BodyWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.pending.extend_from_slice(buf);
        if self.pending.len() >= CHUNK_BYTES {
            self.send_pending()?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.send_pending()
    }
}
