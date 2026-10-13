//! The server inside another process (`docs/desktop.md`): the same
//! router `quack serve` binds, answering requests handed to it directly,
//! with no listener. The desktop app's webview is the only caller, so
//! it runs in [`ServeMode::Local`], as the implicit owner of everything,
//! and nothing a socket would need (a token, a cookie) exists.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::ConnectInfo;
use axum::http::{Request, Response};
use futures::Stream;
use quack_core::config::Config;
use quack_core::jobs::JobQueue;
use quack_core::storage::control::ControlPlane;
use quack_core::web_sessions::WebSessions;
use tower::ServiceExt;

use crate::router;
use crate::state::{App, AppState, ServeMode};

/// The server, held by the process that embeds it.
#[derive(Clone)]
pub struct Embedded {
    app: App,
    /// Built once: the router carries the rate limiter and its sweeper.
    router: Router,
}

impl Embedded {
    /// Open the control plane and build the server in local mode.
    ///
    /// # Errors
    ///
    /// When the control plane does not open.
    pub async fn open(config: Config) -> anyhow::Result<Self> {
        let control = ControlPlane::open(&config)
            .await
            .context("failed to open control plane")?;
        let sessions = Arc::new(WebSessions::new(&config.server));
        let app = Arc::new(AppState::new(
            config,
            control,
            ServeMode::Local,
            sessions,
            None,
        ));
        let router = router(Arc::clone(&app));
        Ok(Self { app, router })
    }

    /// The process's job queue: what the desktop's job strip follows.
    #[must_use]
    pub fn jobs(&self) -> &JobQueue {
        &self.app.jobs
    }

    /// Answer one request, as the listener would, with the body read whole.
    /// A streaming route answered this way ends when the stream does;
    /// [`Self::stream`] serves those.
    pub async fn handle(&self, request: Request<Vec<u8>>) -> Response<Bytes> {
        let (parts, body) = self.respond(request).await.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX)
            .await
            .unwrap_or_else(|e| Bytes::from(format!("the response body failed: {e}")));
        Response::from_parts(parts, bytes)
    }

    /// Answer one request with its body as a stream of chunks: the way a
    /// server-sent event route (`query/stream`, `jobs/stream`) is read.
    pub async fn stream(
        &self,
        request: Request<Vec<u8>>,
    ) -> Response<impl Stream<Item = Result<Bytes, axum::Error>> + Send + use<>> {
        let (parts, body) = self.respond(request).await.into_parts();
        Response::from_parts(parts, body.into_data_stream())
    }

    async fn respond(&self, request: Request<Vec<u8>>) -> Response<Body> {
        let mut request = request.map(Body::from);
        // The caller is this process: the limiter keys and the audit rows
        // see loopback, as they do for a listener on it.
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))));
        match self.router.clone().oneshot(request).await {
            Ok(response) => response,
            // `Router`'s error type is infallible; this arm never runs.
            Err(never) => match never {},
        }
    }

    /// Stop: streams end, the jobs get `grace`, and every workspace closes
    /// so its writer checkpoints.
    pub async fn shutdown(&self, grace: Duration) {
        self.app.stopping.cancel();
        for job in self.app.jobs.shutdown(grace).await {
            tracing::warn!(job = %job.id, kind = %job.kind, state = %job.state, "job still active at exit");
        }
        self.app.close().await;
    }
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "serde_json::Value indexing yields Null for a missing key, never a panic"
)]
mod tests {
    use axum::http::{Request, StatusCode, header};
    use futures::StreamExt;
    use quack_core::config::Config;

    use super::Embedded;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    async fn open() -> (tempfile::TempDir, Embedded) {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = Config::default();
        config.general.data_dir = dir.path().to_path_buf();
        let embedded = Embedded::open(config)
            .await
            .unwrap_or_else(|e| fail(&format!("{e:#}")));
        (dir, embedded)
    }

    /// The router answers as it would a loopback caller in local mode:
    /// the root sends the person to the workspace list, with no sign-in.
    #[tokio::test]
    async fn pages_are_served_in_local_mode_without_a_sign_in() {
        let (_dir, embedded) = open().await;
        let request = Request::builder()
            .uri("/")
            .body(Vec::new())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let response = embedded.handle(request).await;
        assert_eq!(response.status(), StatusCode::SEE_OTHER);
        assert_eq!(
            response
                .headers()
                .get(header::LOCATION)
                .and_then(|v| v.to_str().ok()),
            Some("/workspaces")
        );
        let request = Request::builder()
            .uri("/workspaces")
            .body(Vec::new())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let response = embedded.handle(request).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(String::from_utf8_lossy(response.body()).contains("Workspaces"));
        embedded.shutdown(std::time::Duration::from_secs(1)).await;
    }

    /// A streaming route's chunks arrive as they are sent, and the stream
    /// ends when the server stops.
    #[tokio::test]
    async fn the_job_stream_is_read_chunk_by_chunk() {
        let (_dir, embedded) = open().await;
        let request = Request::builder()
            .method("POST")
            .uri("/api/v1/workspaces")
            .header(header::CONTENT_TYPE, "application/json")
            .body(br#"{"name":"ws"}"#.to_vec())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let created = embedded.handle(request).await;
        assert_eq!(created.status(), StatusCode::CREATED);
        let body: serde_json::Value =
            serde_json::from_slice(created.body()).unwrap_or_else(|e| fail(&e.to_string()));
        let id = body["id"].as_str().unwrap_or_default().to_owned();
        let request = Request::builder()
            .uri(format!("/api/v1/workspaces/{id}/jobs/stream"))
            .header(header::ACCEPT, "text/event-stream")
            .body(Vec::new())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let (parts, mut chunks) = embedded.stream(request).await.into_parts();
        assert_eq!(parts.status, StatusCode::OK);
        let first = chunks
            .next()
            .await
            .unwrap_or_else(|| fail("the stream ended before its first event"))
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(String::from_utf8_lossy(&first).starts_with("event: jobs\n"));
        embedded.shutdown(std::time::Duration::from_secs(1)).await;
        assert!(
            chunks.next().await.is_none(),
            "the stream outlived the server"
        );
    }
}
