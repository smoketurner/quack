//! The server on the workspace socket (`quack_core::forward`): every
//! holder but `quack serve` serves its workspace there, so `quack mcp`
//! and `quack -q` in another shell reach it whatever started first. The
//! socket speaks MCP and nothing else, in rmcp's newline-delimited
//! JSON-RPC, the same framing as stdio.

use std::time::Duration;

use quack_core::forward::{Listener, Stream};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::McpServer;

/// The accept loop over a bound [`Listener`], one MCP session per
/// connection. Dropping it ends the loop and every session and removes
/// the socket file.
pub struct SocketServer {
    stop: CancellationToken,
    /// Taken by [`Self::close`].
    accept: Option<tokio::task::JoinHandle<()>>,
    sessions: TaskTracker,
}

/// How long the accept loop waits after a failed accept before trying
/// again.
const ACCEPT_RETRY: Duration = Duration::from_millis(100);

impl SocketServer {
    /// Serve `server` to every connection `listener` accepts, on tasks
    /// of their own.
    #[must_use]
    pub fn serve(listener: Listener, server: McpServer) -> Self {
        let stop = CancellationToken::new();
        let sessions = TaskTracker::new();
        let accept = tokio::spawn(Self::accept_loop(
            listener,
            server,
            stop.clone(),
            sessions.clone(),
        ));
        Self {
            stop,
            accept: Some(accept),
            sessions,
        }
    }

    async fn accept_loop(
        listener: Listener,
        server: McpServer,
        stop: CancellationToken,
        sessions: TaskTracker,
    ) {
        loop {
            let accepted = tokio::select! {
                biased;
                () = stop.cancelled() => break,
                accepted = listener.accept() => accepted,
            };
            match accepted {
                Ok(stream) => {
                    sessions.spawn(Self::session(server.clone(), stream, stop.child_token()));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "accepting a connection to the workspace socket failed");
                    tokio::time::sleep(ACCEPT_RETRY).await;
                }
            }
        }
        // The listener's drop removes the socket file.
        drop(listener);
    }

    /// One connection's MCP session, until the client hangs up or the
    /// server stops.
    async fn session(server: McpServer, stream: Stream, stop: CancellationToken) {
        let running = match rmcp::serve_server(server, stream).await {
            Ok(running) => running,
            Err(e) => {
                tracing::debug!(error = %e, "a connection to the workspace socket did not initialize");
                return;
            }
        };
        let token = running.cancellation_token();
        let stopper = tokio::spawn(async move {
            stop.cancelled().await;
            token.cancel();
        });
        if let Err(e) = running.waiting().await {
            tracing::debug!(error = %e, "a workspace socket session ended with an error");
        }
        stopper.abort();
    }

    /// Whether a connection's session is in progress: a call relayed
    /// through this process is running.
    #[must_use]
    pub fn has_sessions(&self) -> bool {
        !self.sessions.is_empty()
    }

    /// Stop accepting, end every session, and wait up to `grace` for them.
    /// The accept loop is awaited whatever the grace: it ends at its next
    /// poll (at most the accept retry away), and its end drops the listener,
    /// which removes the socket file, so a one-shot command that held the
    /// workspace for milliseconds leaves no file behind and no warning.
    pub async fn close(mut self, grace: Duration) {
        self.stop.cancel();
        self.sessions.close();
        if let Some(accept) = self.accept.take() {
            drop(accept.await);
        }
        if tokio::time::timeout(grace, self.sessions.wait())
            .await
            .is_err()
        {
            tracing::warn!("the workspace socket's sessions did not end within the grace");
        }
    }
}

impl Drop for SocketServer {
    fn drop(&mut self) {
        // Every session ends at its next poll; the accept loop's next poll
        // drops the listener, which removes the socket file.
        self.stop.cancel();
    }
}
