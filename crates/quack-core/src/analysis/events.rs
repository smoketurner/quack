//! The agent turn as a stream of events.
//!
//! Every interface consumes the same events: the terminal renders them
//! inline, print mode writes steps to stderr and text to stdout, and later
//! the server forwards them as SSE and the session store persists them.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::sync::{mpsc, oneshot};

use super::agent::AgentResponse;
use super::citations::CitationRegistry;

/// One tool invocation, recorded for the transcript and the final response.
/// Lines of a step's detail every interface shows before folding the
/// rest (the terminal's `/steps` and print mode's `--verbose` show all).
pub const STEP_PREVIEW_LINES: usize = 3;

/// The first `STEP_PREVIEW_LINES` lines of a detail and how many follow.
#[must_use]
pub fn preview_detail(detail: &str) -> (Vec<&str>, usize) {
    let total = detail.lines().count();
    (
        detail.lines().take(STEP_PREVIEW_LINES).collect(),
        total.saturating_sub(STEP_PREVIEW_LINES),
    )
}

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
    reply: oneshot::Sender<Decision>,
}

/// The interface's answer to a permission request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Deny,
    /// This statement only.
    Allow,
    /// This statement and every later write in the same turn (the
    /// terminal's `a`; issue #56). The interface keeps its own flag for
    /// the turns after.
    AllowForTurn,
}

impl PermissionRequest {
    pub fn allow(self) {
        self.answer(Decision::Allow);
    }

    pub fn allow_for_turn(self) {
        self.answer(Decision::AllowForTurn);
    }

    pub fn deny(self) {
        self.answer(Decision::Deny);
    }

    fn answer(self, decision: Decision) {
        if self.reply.send(decision).is_err() {
            tracing::debug!("permission answer arrived after the tool stopped waiting");
        }
    }
}

/// What happens during a turn, in order.
#[derive(Debug)]
pub enum AgentEvent {
    /// What the turn is waiting on before any output can appear (a model
    /// Ollama has to load first), in one line. Informational: an
    /// interface that shows nothing but the answer may drop it.
    Status(String),
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
    citations: CitationRegistry,
    /// Set once the interface answered `AllowForTurn`: later writes in
    /// this turn run without asking.
    writes_granted: Arc<AtomicBool>,
    /// Embeddings computed so far this turn, by exact input text: more
    /// than one tool can resolve the same entity label (`search_documents`
    /// and `search_graph` on the same name, `find_path` reusing an entity
    /// a prior call already resolved), and this keeps a turn from paying
    /// for the same embedding call twice.
    embedding_cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    /// `[analysis].max_turns`, so a tool result can tell the model how
    /// much of the turn is left; `None` when the limit is not known.
    turn_limit: Option<usize>,
}

/// From this many calls left, a tool result tells the model to answer.
const TURNS_LOW_WATER: usize = 3;

impl TurnRecorder {
    #[must_use]
    pub fn new(sink: EventSink) -> Self {
        Self {
            sink,
            steps: Arc::new(Mutex::new(Vec::new())),
            citations: CitationRegistry::default(),
            writes_granted: Arc::new(AtomicBool::new(false)),
            embedding_cache: Arc::new(Mutex::new(HashMap::new())),
            turn_limit: None,
        }
    }

    /// Record the turn's tool-call limit so results can report it.
    #[must_use]
    pub fn with_turn_limit(mut self, max_turns: usize) -> Self {
        self.turn_limit = Some(max_turns);
        self
    }

    /// One line for the end of a tool result: which call this was out of
    /// the turn's limit, and, once `TURNS_LOW_WATER` or fewer remain, that
    /// it is time to answer. Counts the calls recorded so far, which is
    /// never fewer than the model round-trips rig counts, so the number
    /// is conservative. Empty without a limit.
    #[must_use]
    pub fn budget_note(&self) -> String {
        let Some(limit) = self.turn_limit else {
            return String::new();
        };
        let used = self.steps.lock().map_or(0, |s| s.len());
        let remaining = limit.saturating_sub(used);
        if remaining <= TURNS_LOW_WATER {
            format!(
                "(tool call {used} of at most {limit} this turn; {remaining} left, so answer \
                 from what you have)"
            )
        } else {
            format!("(tool call {used} of at most {limit} this turn)")
        }
    }

    /// This turn's cached embedding for `text`, if some earlier call this
    /// turn already computed it.
    #[must_use]
    pub fn cached_embedding(&self, text: &str) -> Option<Vec<f32>> {
        self.embedding_cache.lock().ok()?.get(text).cloned()
    }

    /// Remember `text`'s embedding for the rest of this turn.
    pub fn cache_embedding(&self, text: &str, embedding: Vec<f32>) {
        if let Ok(mut cache) = self.embedding_cache.lock() {
            cache.insert(text.to_owned(), embedding);
        }
    }

    /// The chunks retrieved so far this turn, numbered for citing.
    #[must_use]
    pub fn citations(&self) -> &CitationRegistry {
        &self.citations
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

    /// Ask the interface whether a write may run, unless an earlier
    /// answer this turn already granted every write. Resolves to `false`
    /// when the interface drops the request.
    pub async fn ask_permission(&self, sql: &str) -> bool {
        if self.writes_granted.load(Ordering::Acquire) {
            return true;
        }
        let (reply, answer) = oneshot::channel();
        self.emit(AgentEvent::PermissionRequired(PermissionRequest {
            sql: sql.to_owned(),
            reply,
        }));
        match answer.await.unwrap_or(Decision::Deny) {
            Decision::Deny => false,
            Decision::Allow => true,
            Decision::AllowForTurn => {
                self.writes_granted.store(true, Ordering::Release);
                true
            }
        }
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

    #[test]
    fn budget_note_counts_calls_and_warns_near_the_limit() {
        let (sink, _rx) = channel();
        let recorder = TurnRecorder::new(sink);
        assert_eq!(recorder.budget_note(), "");
        let recorder = recorder.with_turn_limit(5);
        recorder.start("run_sql", "SELECT 1").finish("1 rows");
        assert_eq!(
            recorder.budget_note(),
            "(tool call 1 of at most 5 this turn)"
        );
        recorder.start("run_sql", "SELECT 2").finish("1 rows");
        assert_eq!(
            recorder.budget_note(),
            "(tool call 2 of at most 5 this turn; 3 left, so answer from what you have)"
        );
        for _ in 0..4 {
            recorder.start("run_sql", "SELECT 3").finish("1 rows");
        }
        assert_eq!(
            recorder.budget_note(),
            "(tool call 6 of at most 5 this turn; 0 left, so answer from what you have)"
        );
    }

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

    #[test]
    fn embedding_cache_returns_what_it_was_given_and_nothing_else() {
        let (sink, _rx) = channel();
        let recorder = TurnRecorder::new(sink);
        assert_eq!(recorder.cached_embedding("Acme"), None);

        recorder.cache_embedding("Acme", vec![1.0, 2.0, 3.0]);
        assert_eq!(recorder.cached_embedding("Acme"), Some(vec![1.0, 2.0, 3.0]));
        // A different tool resolving the same label this turn gets the
        // same vector back rather than embedding it again.
        assert_eq!(recorder.cached_embedding("Acme"), Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(recorder.cached_embedding("Beta"), None);
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

        // `a` grants the rest of the turn: the next write is not asked.
        let asker = recorder.clone();
        let granted = tokio::spawn(async move { asker.ask_permission("UPDATE t SET a = 1").await });
        let req = permission_request(rx.recv().await).unwrap();
        req.allow_for_turn();
        assert!(granted.await.is_ok_and(|a| a));
        assert!(recorder.ask_permission("DELETE FROM t").await);
        assert!(rx.try_recv().is_err(), "no request was emitted");
        // A fresh recorder (the next turn) asks again.
        let (sink, mut rx) = channel();
        let next = TurnRecorder::new(sink);
        let asker = next.clone();
        let pending = tokio::spawn(async move { asker.ask_permission("DELETE FROM t").await });
        assert!(permission_request(rx.recv().await).is_some());
        pending.abort();
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
