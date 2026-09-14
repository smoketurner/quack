//! The agent turn as a stream of events.
//!
//! Every interface consumes the same events: the terminal renders them
//! inline, print mode writes steps to stderr and text to stdout, and later
//! the server forwards them as SSE and the session store persists them.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::sync::{mpsc, oneshot};

use super::agent::AgentResponse;

/// One tool invocation, recorded for the transcript and the final response.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolStep {
    pub tool: String,
    /// What the tool was asked to do: the SQL text, the search query, the
    /// table name.
    pub detail: String,
    /// What came back, in one line: `6 rows`, `8 chunks`, `refused`.
    pub summary: String,
    pub duration_ms: u64,
}

/// A write the agent wants to run; the interface answers with `allow` or
/// `deny`. Dropping it unanswered counts as deny.
#[derive(Debug)]
pub struct PermissionRequest {
    pub sql: String,
    reply: oneshot::Sender<bool>,
}

impl PermissionRequest {
    pub fn allow(self) {
        self.answer(true);
    }

    pub fn deny(self) {
        self.answer(false);
    }

    fn answer(self, allowed: bool) {
        if self.reply.send(allowed).is_err() {
            tracing::debug!("permission answer arrived after the tool stopped waiting");
        }
    }
}

/// What happens during a turn, in order.
#[derive(Debug)]
pub enum AgentEvent {
    /// A piece of the assistant's answer, as it streams.
    TextDelta(String),
    ToolStarted {
        tool: String,
        detail: String,
    },
    ToolFinished(ToolStep),
    /// A mutating statement needs the user's decision.
    PermissionRequired(PermissionRequest),
    /// The turn finished; the response carries the full text, the steps, the
    /// chart, and whether any write was refused.
    TurnComplete(AgentResponse),
    /// The turn failed after possibly emitting some of the above.
    Failed(String),
}

/// Sending half of the event channel.
pub type EventSink = mpsc::UnboundedSender<AgentEvent>;
/// Receiving half of the event channel.
pub type EventStream = mpsc::UnboundedReceiver<AgentEvent>;

/// Create a channel for one turn.
#[must_use]
pub fn channel() -> (EventSink, EventStream) {
    mpsc::unbounded_channel()
}

/// Shared by every tool in a turn: emits events and keeps the step list.
#[derive(Clone)]
pub struct TurnRecorder {
    sink: EventSink,
    steps: Arc<Mutex<Vec<ToolStep>>>,
}

impl TurnRecorder {
    #[must_use]
    pub fn new(sink: EventSink) -> Self {
        Self {
            sink,
            steps: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn emit(&self, event: AgentEvent) {
        // A closed receiver means the interface went away; nothing to do.
        drop(self.sink.send(event));
    }

    /// Announce a tool call. Finish the returned guard to record the result.
    #[must_use]
    pub fn start(&self, tool: &str, detail: &str) -> StepInProgress {
        self.emit(AgentEvent::ToolStarted {
            tool: tool.to_owned(),
            detail: detail.to_owned(),
        });
        StepInProgress {
            recorder: self.clone(),
            tool: tool.to_owned(),
            detail: detail.to_owned(),
            started: Instant::now(),
        }
    }

    /// Ask the interface whether a write may run. Resolves to `false` when
    /// the interface drops the request.
    pub async fn ask_permission(&self, sql: &str) -> bool {
        let (reply, answer) = oneshot::channel();
        self.emit(AgentEvent::PermissionRequired(PermissionRequest {
            sql: sql.to_owned(),
            reply,
        }));
        answer.await.unwrap_or(false)
    }

    /// The steps recorded so far, in order.
    #[must_use]
    pub fn steps(&self) -> Vec<ToolStep> {
        self.steps.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

/// A tool call that has started; call `finish` with its one-line result.
pub struct StepInProgress {
    recorder: TurnRecorder,
    tool: String,
    detail: String,
    started: Instant,
}

impl StepInProgress {
    pub fn finish(self, summary: impl Into<String>) {
        let step = ToolStep {
            tool: self.tool,
            detail: self.detail,
            summary: summary.into(),
            duration_ms: u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX),
        };
        if let Ok(mut steps) = self.recorder.steps.lock() {
            steps.push(step.clone());
        }
        self.recorder.emit(AgentEvent::ToolFinished(step));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn steps_are_emitted_and_recorded_in_order() {
        let (sink, mut rx) = channel();
        let recorder = TurnRecorder::new(sink);
        recorder.start("run_sql", "SELECT 1").finish("1 rows");
        recorder
            .start("search_documents", "flood")
            .finish("0 chunks");

        let steps = recorder.steps();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps.first().map(|s| s.tool.as_str()), Some("run_sql"));
        assert_eq!(steps.last().map(|s| s.summary.as_str()), Some("0 chunks"));

        assert!(
            matches!(rx.recv().await, Some(AgentEvent::ToolStarted { tool, .. }) if tool == "run_sql")
        );
        assert!(
            matches!(rx.recv().await, Some(AgentEvent::ToolFinished(s)) if s.detail == "SELECT 1")
        );
        assert!(matches!(
            rx.recv().await,
            Some(AgentEvent::ToolStarted { .. })
        ));
        assert!(matches!(rx.recv().await, Some(AgentEvent::ToolFinished(_))));
    }

    fn permission_request(event: Option<AgentEvent>) -> Option<PermissionRequest> {
        match event {
            Some(AgentEvent::PermissionRequired(req)) => Some(req),
            _ => None,
        }
    }

    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test asserts the event kind")]
    async fn permission_round_trip_and_dropped_request_denies() {
        let (sink, mut rx) = channel();
        let recorder = TurnRecorder::new(sink);

        let asker = recorder.clone();
        let allowed = tokio::spawn(async move { asker.ask_permission("DROP TABLE t").await });
        let req = permission_request(rx.recv().await).unwrap();
        assert_eq!(req.sql, "DROP TABLE t");
        req.allow();
        assert!(allowed.await.is_ok_and(|a| a));

        let asker = recorder.clone();
        let denied = tokio::spawn(async move { asker.ask_permission("DELETE FROM t").await });
        let req = permission_request(rx.recv().await).unwrap();
        drop(req);
        assert!(denied.await.is_ok_and(|a| !a));
    }

    #[test]
    fn emit_into_a_closed_channel_is_harmless() {
        let (sink, rx) = channel();
        drop(rx);
        let recorder = TurnRecorder::new(sink);
        recorder.start("run_sql", "SELECT 1").finish("ok");
        assert_eq!(recorder.steps().len(), 1);
    }
}
