//! Work queues: the asynchronous model every interface shares (design doc
//! 4.1).
//!
//! Anything slower than a keystroke — an agent turn, a SQL statement, an
//! ingest, an import, an ontology or graph run — is submitted as a job and
//! runs in the background while the interface stays responsive. A
//! [`JobQueue`] bounds how many run at once with a pool of worker slots
//! (`[jobs].workers`), and a job may also name a [`Lane`]: jobs sharing a
//! lane key run at most `limit` at a time, in the order they were submitted.
//! A chat session is a serial lane (a turn's history includes the turn
//! before it), a workspace's uploads are a lane of
//! `[server].workers_per_workspace`, and anything without a lane runs
//! whenever a worker is free.
//!
//! Every change of a job's state goes out on a broadcast channel as a
//! [`JobInfo`] snapshot, so the terminal's job strip and the web console's
//! Jobs page report the same status. The registry is in memory only: a job
//! label can name a file or quote a question, which is workspace content,
//! so it never reaches `control.db`, and a restart forgets it (the durable
//! record of what a job did is the document, table, or session it wrote).

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex, PoisonError};

use jiff::Timestamp;
use serde::Serialize;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, broadcast, oneshot};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Snapshots the broadcast channel holds for a slow subscriber before it
/// lags; a lagged subscriber resynchronizes from [`JobQueue::list`].
const EVENT_CAPACITY: usize = 256;

/// A job's id: UUID v7, generated when it is submitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(Uuid);

impl JobId {
    fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// Parse an id as [`fmt::Display`] writes it.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Uuid::parse_str(text).ok().map(Self)
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Serialize for JobId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

/// What a job does, for display and filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    /// An agent turn.
    Chat,
    /// A SQL statement typed by the user.
    Sql,
    /// A file parsed, stored, and embedded.
    Ingest,
    /// Rows pulled from an external source.
    Import,
    /// An ontology command or proposal run.
    Ontology,
    /// A graph command or extraction run.
    Graph,
    /// A bundle, context, or session written out.
    Export,
}

impl JobKind {
    /// The kind as it serializes.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Sql => "sql",
            Self::Ingest => "ingest",
            Self::Import => "import",
            Self::Ontology => "ontology",
            Self::Graph => "graph",
            Self::Export => "export",
        }
    }
}

impl fmt::Display for JobKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where a job is in its life. `Queued` and `Running` are active; the other
/// three are final.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    /// Waiting for its lane or a worker slot.
    Queued,
    Running,
    Succeeded,
    Failed,
    /// Cancelled while queued, or stopped early by its work after a
    /// cancel.
    Cancelled,
}

impl JobState {
    /// Whether the job has ended.
    #[must_use]
    pub const fn is_finished(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }

    /// The state as it serializes.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

impl fmt::Display for JobState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Jobs sharing `key` run at most `limit` at a time, in submission order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lane {
    key: String,
    limit: u32,
}

impl Lane {
    /// One job at a time: a chat session, a workspace's graph extraction.
    #[must_use]
    pub fn serial(key: impl Into<String>) -> Self {
        Self::new(key, 1)
    }

    /// Up to `limit` at a time (at least one). The first job submitted on
    /// a key fixes its limit while any job holds the lane.
    #[must_use]
    pub fn new(key: impl Into<String>, limit: u32) -> Self {
        Self {
            key: key.into(),
            limit: limit.max(1),
        }
    }

    /// The lane key, as [`JobInfo::lane`] carries it.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }
}

/// What to submit alongside the work itself.
#[derive(Debug, Clone)]
pub struct JobSpec {
    kind: JobKind,
    label: String,
    workspace_id: Option<String>,
    owner: Option<String>,
    lane: Option<Lane>,
}

impl JobSpec {
    /// A job of `kind`, shown as `label`.
    #[must_use]
    pub fn new(kind: JobKind, label: impl Into<String>) -> Self {
        Self {
            kind,
            label: label.into(),
            workspace_id: None,
            owner: None,
            lane: None,
        }
    }

    /// The workspace the job touches; the web console lists a workspace's
    /// jobs by it.
    #[must_use]
    pub fn workspace(mut self, workspace_id: impl Into<String>) -> Self {
        self.workspace_id = Some(workspace_id.into());
        self
    }

    /// The user who submitted it (server mode).
    #[must_use]
    pub fn owner(mut self, user_id: Option<impl Into<String>>) -> Self {
        self.owner = user_id.map(Into::into);
        self
    }

    /// The lane it runs in.
    #[must_use]
    pub fn lane(mut self, lane: Lane) -> Self {
        self.lane = Some(lane);
        self
    }
}

/// How far along a job with countable work is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct JobProgress {
    pub done: u32,
    pub total: u32,
}

/// A job as it stands: what every subscriber receives on each change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JobInfo {
    pub id: JobId,
    /// A short number, counting from 1 per queue, for people to type
    /// (`/cancel 3`): v7 ids submitted together share their first digits.
    pub number: u64,
    pub kind: JobKind,
    pub label: String,
    pub workspace_id: Option<String>,
    pub owner: Option<String>,
    pub lane: Option<String>,
    pub state: JobState,
    pub progress: Option<JobProgress>,
    /// The latest status line the work reported while running.
    pub status: Option<String>,
    /// The work's summary when it succeeded, its error otherwise.
    pub outcome: Option<String>,
    pub queued_at: Timestamp,
    pub started_at: Option<Timestamp>,
    pub finished_at: Option<Timestamp>,
    /// Whether a cancel was requested (the work may still be finishing).
    pub cancel_requested: bool,
}

/// Counts of active jobs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct JobCounts {
    pub queued: usize,
    pub running: usize,
}

impl JobCounts {
    /// Queued plus running.
    #[must_use]
    pub const fn active(self) -> usize {
        self.queued.saturating_add(self.running)
    }
}

/// What the work returns: a one-line summary, or the error to show.
pub type JobResult = std::result::Result<String, String>;

/// Handed to the work: its id, its cancel token, and a way to report.
#[derive(Clone)]
pub struct JobContext {
    id: JobId,
    cancel: CancellationToken,
    inner: Arc<Inner>,
}

impl JobContext {
    #[must_use]
    pub const fn id(&self) -> JobId {
        self.id
    }

    /// Cancelled when someone cancels the job; long work should watch it
    /// (an agent turn passes it to `run_turn`).
    #[must_use]
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Report `done` of `total` units finished.
    pub fn progress(&self, done: u32, total: u32) {
        self.inner.update(self.id, |info| {
            info.progress = Some(JobProgress { done, total });
        });
    }

    /// Report a status line (the latest one is kept).
    pub fn status(&self, text: impl Into<String>) {
        let text = text.into();
        self.inner.update(self.id, |info| info.status = Some(text));
    }
}

struct Entry {
    info: JobInfo,
    cancel: CancellationToken,
}

#[derive(Default)]
struct Registry {
    /// Submission order.
    order: VecDeque<JobId>,
    jobs: HashMap<JobId, Entry>,
    next_number: u64,
}

struct Inner {
    workers: Arc<Semaphore>,
    worker_count: u32,
    lanes: Mutex<HashMap<String, LaneState>>,
    registry: Mutex<Registry>,
    history: usize,
    events: broadcast::Sender<JobInfo>,
}

impl Inner {
    fn registry(&self) -> std::sync::MutexGuard<'_, Registry> {
        // A panic while holding the lock leaves plain data behind; the
        // registry stays usable.
        self.registry.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Apply `f` to the job and broadcast the result.
    fn update(&self, id: JobId, f: impl FnOnce(&mut JobInfo)) {
        let snapshot = {
            let mut registry = self.registry();
            let Some(entry) = registry.jobs.get_mut(&id) else {
                return;
            };
            f(&mut entry.info);
            entry.info.clone()
        };
        // No subscribers is fine.
        drop(self.events.send(snapshot));
    }

    /// Record the job's end and drop the oldest finished jobs past the
    /// history bound.
    fn finish(&self, id: JobId, state: JobState, outcome: Option<String>) {
        self.update(id, |info| {
            info.state = state;
            info.outcome = outcome;
            info.finished_at = Some(Timestamp::now());
        });
        let mut registry = self.registry();
        let finished = registry
            .jobs
            .values()
            .filter(|e| e.info.state.is_finished())
            .count();
        let mut excess = finished.saturating_sub(self.history);
        if excess == 0 {
            return;
        }
        let Registry { order, jobs, .. } = &mut *registry;
        order.retain(|id| {
            let drop_it = excess > 0 && jobs.get(id).is_none_or(|e| e.info.state.is_finished());
            if drop_it {
                excess = excess.saturating_sub(1);
                jobs.remove(id);
            }
            !drop_it
        });
    }

    fn lanes(&self) -> std::sync::MutexGuard<'_, HashMap<String, LaneState>> {
        self.lanes.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Take the job's place in its lane, synchronously at submit: a free
    /// slot now, or the next place in the lane's line. Deciding here rather
    /// than in the spawned task is what keeps a lane in submission order;
    /// Tokio does not run spawned tasks in the order they were spawned.
    fn enter_lane(self: &Arc<Self>, lane: &Lane) -> LaneTicket {
        let mut lanes = self.lanes();
        let state = lanes.entry(lane.key.clone()).or_insert_with(|| LaneState {
            limit: usize::try_from(lane.limit).unwrap_or(usize::MAX),
            running: 0,
            waiting: VecDeque::new(),
        });
        if state.running < state.limit {
            state.running = state.running.saturating_add(1);
            drop(lanes);
            LaneTicket::Ready(LanePermit::new(self, &lane.key))
        } else {
            let (sender, receiver) = oneshot::channel();
            state.waiting.push_back(sender);
            LaneTicket::Wait(receiver)
        }
    }

    /// A lane slot came free: hand it to the first job still waiting (one
    /// cancelled while queued has dropped its receiver and is skipped), or
    /// give it back, forgetting a lane nobody holds or waits on so
    /// per-session keys do not pile up.
    fn leave_lane(self: &Arc<Self>, key: &str) {
        let mut lanes = self.lanes();
        let Some(state) = lanes.get_mut(key) else {
            return;
        };
        while let Some(next) = state.waiting.pop_front() {
            match next.send(LanePermit::new(self, key)) {
                Ok(()) => return,
                // Disarmed, so dropping it here does not re-enter this lock.
                Err(mut unclaimed) => unclaimed.armed = false,
            }
        }
        state.running = state.running.saturating_sub(1);
        if state.running == 0 {
            lanes.remove(key);
        }
    }
}

/// One lane's slots: how many are held, and who waits for one, in order.
struct LaneState {
    limit: usize,
    running: usize,
    waiting: VecDeque<oneshot::Sender<LanePermit>>,
}

/// A held lane slot; dropping it passes the slot on.
struct LanePermit {
    inner: Arc<Inner>,
    key: String,
    armed: bool,
}

impl LanePermit {
    fn new(inner: &Arc<Inner>, key: &str) -> Self {
        Self {
            inner: Arc::clone(inner),
            key: key.to_owned(),
            armed: true,
        }
    }
}

impl Drop for LanePermit {
    fn drop(&mut self) {
        if self.armed {
            self.inner.leave_lane(&self.key);
        }
    }
}

/// A job's place in its lane, taken at submit.
enum LaneTicket {
    Ready(LanePermit),
    Wait(oneshot::Receiver<LanePermit>),
}

/// The work queue. Cheap to clone; every clone is the same queue.
#[derive(Clone)]
pub struct JobQueue {
    inner: Arc<Inner>,
}

impl JobQueue {
    /// A queue running up to `workers` jobs at once (at least one) and
    /// remembering the last `history` finished ones (at least one).
    #[must_use]
    pub fn new(workers: u32, history: u32) -> Self {
        let workers = workers.max(1);
        let (events, _) = broadcast::channel(EVENT_CAPACITY);
        Self {
            inner: Arc::new(Inner {
                workers: Arc::new(Semaphore::new(
                    usize::try_from(workers).unwrap_or(usize::MAX),
                )),
                worker_count: workers,
                lanes: Mutex::new(HashMap::new()),
                registry: Mutex::new(Registry::default()),
                history: usize::try_from(history.max(1)).unwrap_or(usize::MAX),
                events,
            }),
        }
    }

    /// The queue `[jobs]` describes.
    #[must_use]
    pub fn from_config(config: &crate::config::JobsConfig) -> Self {
        Self::new(config.workers, config.history)
    }

    /// Worker slots: how many jobs can run at once.
    #[must_use]
    pub fn workers(&self) -> u32 {
        self.inner.worker_count
    }

    /// Queue `work` and return its id at once. It starts when its lane
    /// (if any) and a worker slot are both free; a panic inside it is a
    /// failed job, not a lost one.
    ///
    /// Must be called inside a Tokio runtime.
    pub fn submit<F, Fut>(&self, spec: JobSpec, work: F) -> JobId
    where
        F: FnOnce(JobContext) -> Fut + Send + 'static,
        Fut: Future<Output = JobResult> + Send + 'static,
    {
        let id = JobId::new();
        let cancel = CancellationToken::new();
        let snapshot = {
            let mut registry = self.inner.registry();
            registry.next_number = registry.next_number.saturating_add(1);
            let info = JobInfo {
                id,
                number: registry.next_number,
                kind: spec.kind,
                label: spec.label,
                workspace_id: spec.workspace_id,
                owner: spec.owner,
                lane: spec.lane.as_ref().map(|l| l.key.clone()),
                state: JobState::Queued,
                progress: None,
                status: None,
                outcome: None,
                queued_at: Timestamp::now(),
                started_at: None,
                finished_at: None,
                cancel_requested: false,
            };
            registry.order.push_back(id);
            registry.jobs.insert(
                id,
                Entry {
                    info: info.clone(),
                    cancel: cancel.clone(),
                },
            );
            info
        };
        drop(self.inner.events.send(snapshot));

        // The lane place is taken now, so the lane runs in submission order.
        let ticket = spec.lane.as_ref().map(|lane| self.inner.enter_lane(lane));
        let inner = Arc::clone(&self.inner);
        tokio::spawn(async move {
            let ctx = JobContext {
                id,
                cancel: cancel.clone(),
                inner: Arc::clone(&inner),
            };
            run(&inner, ticket, ctx, work).await;
        });
        id
    }

    /// Every job remembered, in submission order.
    #[must_use]
    pub fn list(&self) -> Vec<JobInfo> {
        let registry = self.inner.registry();
        registry
            .order
            .iter()
            .filter_map(|id| registry.jobs.get(id).map(|e| e.info.clone()))
            .collect()
    }

    /// The jobs of one workspace, in submission order.
    #[must_use]
    pub fn list_workspace(&self, workspace_id: &str) -> Vec<JobInfo> {
        self.list()
            .into_iter()
            .filter(|j| j.workspace_id.as_deref() == Some(workspace_id))
            .collect()
    }

    #[must_use]
    pub fn get(&self, id: JobId) -> Option<JobInfo> {
        self.inner.registry().jobs.get(&id).map(|e| e.info.clone())
    }

    /// The job shown as `number`.
    #[must_use]
    pub fn by_number(&self, number: u64) -> Option<JobInfo> {
        self.inner
            .registry()
            .jobs
            .values()
            .find(|e| e.info.number == number)
            .map(|e| e.info.clone())
    }

    /// Ask the job to stop. A queued job ends as cancelled without running;
    /// a running one sees its token cancelled and stops when its work
    /// notices. Returns whether the job was active.
    #[expect(
        clippy::must_use_candidate,
        reason = "cancelling is done for its effect; the answer is optional"
    )]
    pub fn cancel(&self, id: JobId) -> bool {
        let token = {
            let registry = self.inner.registry();
            match registry.jobs.get(&id) {
                Some(entry) if !entry.info.state.is_finished() => entry.cancel.clone(),
                _ => return false,
            }
        };
        self.inner.update(id, |info| info.cancel_requested = true);
        token.cancel();
        true
    }

    /// Queued and running counts, optionally for one workspace.
    #[must_use]
    pub fn counts(&self, workspace_id: Option<&str>) -> JobCounts {
        let registry = self.inner.registry();
        let mut counts = JobCounts::default();
        for entry in registry.jobs.values() {
            if workspace_id.is_some_and(|ws| entry.info.workspace_id.as_deref() != Some(ws)) {
                continue;
            }
            match entry.info.state {
                JobState::Queued => counts.queued = counts.queued.saturating_add(1),
                JobState::Running => counts.running = counts.running.saturating_add(1),
                JobState::Succeeded | JobState::Failed | JobState::Cancelled => {}
            }
        }
        counts
    }

    /// Every change from now on, as snapshots.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<JobInfo> {
        self.inner.events.subscribe()
    }

    /// Wait for the job to finish and return its final snapshot (`None`
    /// when it is unknown or was dropped from the history).
    pub async fn wait(&self, id: JobId) -> Option<JobInfo> {
        let mut events = self.subscribe();
        loop {
            let current = self.get(id)?;
            if current.state.is_finished() {
                return Some(current);
            }
            match events.recv().await {
                Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => return self.get(id),
            }
        }
    }
}

/// Wait for the lane and a worker, run the work, and record its end. The
/// end is recorded while the permits are still held, so the next job in
/// the lane never starts before its predecessor reads as finished.
async fn run<F, Fut>(inner: &Arc<Inner>, lane: Option<LaneTicket>, ctx: JobContext, work: F)
where
    F: FnOnce(JobContext) -> Fut + Send + 'static,
    Fut: Future<Output = JobResult> + Send + 'static,
{
    let id = ctx.id;
    // The permits live in `_held` until the end is recorded.
    let (state, outcome, _held) = run_held(inner, lane, ctx, work).await;
    inner.finish(id, state, outcome);
}

/// The permits a running job holds; dropped after its end is recorded.
type Held = (Option<LanePermit>, Option<OwnedSemaphorePermit>);

async fn run_held<F, Fut>(
    inner: &Arc<Inner>,
    lane: Option<LaneTicket>,
    ctx: JobContext,
    work: F,
) -> (JobState, Option<String>, Held)
where
    F: FnOnce(JobContext) -> Fut + Send + 'static,
    Fut: Future<Output = JobResult> + Send + 'static,
{
    let cancel = ctx.cancel_token();
    let cancelled_while_queued = || {
        (
            JobState::Cancelled,
            Some(String::from("cancelled before it started")),
            (None, None),
        )
    };
    let lane_permit: Option<LanePermit> = match lane {
        Some(LaneTicket::Ready(permit)) => Some(permit),
        Some(LaneTicket::Wait(receiver)) => tokio::select! {
            biased;
            () = cancel.cancelled() => return cancelled_while_queued(),
            permit = receiver => match permit {
                Ok(permit) => Some(permit),
                Err(_) => {
                    return (JobState::Failed, Some(String::from("the lane closed")), (None, None));
                }
            },
        },
        None => None,
    };
    let worker_permit = tokio::select! {
        biased;
        () = cancel.cancelled() => return cancelled_while_queued(),
        permit = Arc::clone(&inner.workers).acquire_owned() => match permit {
            Ok(permit) => permit,
            Err(_) => {
                return (
                    JobState::Failed,
                    Some(String::from("the queue closed")),
                    (lane_permit, None),
                );
            }
        },
    };
    let id = ctx.id;
    inner.update(id, |info| {
        info.state = JobState::Running;
        info.started_at = Some(Timestamp::now());
    });
    // Its own task, so a panic in the work surfaces as a join error here.
    let outcome = tokio::spawn(work(ctx)).await;
    let (state, text) = match outcome {
        Ok(Ok(summary)) => (JobState::Succeeded, Some(summary)),
        Ok(Err(_)) if cancel.is_cancelled() => {
            (JobState::Cancelled, Some(String::from("cancelled")))
        }
        Ok(Err(error)) => (JobState::Failed, Some(error)),
        Err(join) => {
            tracing::error!(job = %id, error = %join, "job task failed");
            (
                JobState::Failed,
                Some(format!("the job stopped unexpectedly: {join}")),
            )
        }
    };
    (state, text, (lane_permit, Some(worker_permit)))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    async fn finished(queue: &JobQueue, id: JobId) -> JobInfo {
        tokio::time::timeout(Duration::from_secs(5), queue.wait(id))
            .await
            .unwrap_or_else(|_| fail("job did not finish"))
            .unwrap_or_else(|| fail("job forgotten"))
    }

    #[tokio::test]
    async fn jobs_run_report_and_finish() {
        let queue = JobQueue::new(2, 10);
        let mut events = queue.subscribe();
        let ok = queue.submit(
            JobSpec::new(JobKind::Sql, "select").workspace("ws"),
            |ctx| async move {
                ctx.progress(1, 2);
                ctx.status("halfway");
                Ok(String::from("2 rows"))
            },
        );
        let err = queue.submit(JobSpec::new(JobKind::Ingest, "bad.pdf"), |_| async {
            Err(String::from("not a pdf"))
        });
        let ok = finished(&queue, ok).await;
        assert_eq!(ok.state, JobState::Succeeded);
        assert_eq!(ok.outcome.as_deref(), Some("2 rows"));
        assert_eq!(ok.progress, Some(JobProgress { done: 1, total: 2 }));
        assert_eq!(ok.status.as_deref(), Some("halfway"));
        assert!(ok.started_at.is_some() && ok.finished_at.is_some());
        let err = finished(&queue, err).await;
        assert_eq!(err.state, JobState::Failed);
        assert_eq!(err.outcome.as_deref(), Some("not a pdf"));
        assert_eq!(err.number, 2);

        // The first event is the queued snapshot.
        let first = events.recv().await.unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(first.state, JobState::Queued);
        assert_eq!(queue.list_workspace("ws").len(), 1);
        assert_eq!(queue.counts(None).active(), 0);
        assert_eq!(queue.by_number(2).map(|j| j.id), Some(err.id));
    }

    #[tokio::test]
    async fn a_serial_lane_runs_in_order_and_workers_bound_concurrency() {
        let queue = JobQueue::new(3, 50);
        let log = Arc::new(Mutex::new(Vec::new()));
        let gate = Arc::new(tokio::sync::Notify::new());
        let mut ids = Vec::new();
        for n in 0..3_u32 {
            let log = Arc::clone(&log);
            let gate = Arc::clone(&gate);
            ids.push(queue.submit(
                JobSpec::new(JobKind::Chat, format!("turn {n}")).lane(Lane::serial("session:a")),
                move |_| async move {
                    if n == 0 {
                        gate.notified().await;
                    }
                    log.lock().unwrap_or_else(PoisonError::into_inner).push(n);
                    Ok(String::new())
                },
            ));
        }
        // Another lane is not held up by the first.
        let other = queue.submit(
            JobSpec::new(JobKind::Sql, "other").lane(Lane::serial("session:b")),
            |_| async { Ok(String::from("done")) },
        );
        assert_eq!(finished(&queue, other).await.state, JobState::Succeeded);
        let counts = queue.counts(None);
        assert_eq!((counts.running, counts.queued), (1, 2));
        gate.notify_one();
        let mut previous_end = None;
        for id in ids {
            let job = finished(&queue, id).await;
            // Each starts only after the one before it reads as finished.
            if let Some(end) = previous_end {
                assert!(job.started_at.is_some_and(|start| start >= end));
            }
            previous_end = job.finished_at;
        }
        assert_eq!(
            *log.lock().unwrap_or_else(PoisonError::into_inner),
            vec![0, 1, 2]
        );
        // The lane is forgotten once nobody holds it.
        assert!(
            queue
                .inner
                .lanes
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty()
        );

        // One worker: the second job waits for the first.
        let single = JobQueue::new(1, 10);
        let gate = Arc::new(tokio::sync::Notify::new());
        let first = single.submit(JobSpec::new(JobKind::Sql, "a"), {
            let gate = Arc::clone(&gate);
            move |_| async move {
                gate.notified().await;
                Ok(String::new())
            }
        });
        let second = single.submit(JobSpec::new(JobKind::Sql, "b"), |_| async {
            Ok(String::new())
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(single.get(second).map(|j| j.state), Some(JobState::Queued));
        gate.notify_one();
        finished(&single, first).await;
        assert_eq!(finished(&single, second).await.state, JobState::Succeeded);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_lane_keeps_submission_order_on_a_multi_threaded_runtime() {
        let queue = JobQueue::new(8, 200);
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut ids = Vec::new();
        for n in 0..50_u32 {
            let log = Arc::clone(&log);
            ids.push(queue.submit(
                JobSpec::new(JobKind::Chat, format!("{n}")).lane(Lane::serial("session:x")),
                move |_| async move {
                    log.lock().unwrap_or_else(PoisonError::into_inner).push(n);
                    Ok(String::new())
                },
            ));
        }
        // One cancelled while queued is skipped, not waited on.
        let skipped = ids.get(10).copied().unwrap_or_else(|| fail("no job 10"));
        let _ = queue.cancel(skipped);
        for id in ids {
            finished(&queue, id).await;
        }
        let ran = log.lock().unwrap_or_else(PoisonError::into_inner).clone();
        let mut sorted = ran.clone();
        sorted.sort_unstable();
        assert_eq!(ran, sorted, "the lane ran out of order");
        assert!(ran.len() >= 49);
        assert!(
            queue
                .inner
                .lanes
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty()
        );
    }

    #[tokio::test]
    async fn cancel_stops_queued_and_running_jobs_and_panics_fail() {
        let queue = JobQueue::new(1, 10);
        let running = queue.submit(JobSpec::new(JobKind::Chat, "long"), |ctx| async move {
            ctx.cancel_token().cancelled().await;
            Err(String::from("stopped"))
        });
        let queued = queue.submit(JobSpec::new(JobKind::Chat, "never"), |_| async {
            Ok(String::from("ran"))
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(queue.cancel(queued));
        let queued = finished(&queue, queued).await;
        assert_eq!(queued.state, JobState::Cancelled);
        assert!(queued.started_at.is_none());
        assert!(queue.cancel(running));
        let running = finished(&queue, running).await;
        assert_eq!(running.state, JobState::Cancelled);
        assert!(running.cancel_requested);
        assert!(
            !queue.cancel(running.id),
            "a finished job cannot be cancelled"
        );

        #[expect(clippy::panic, reason = "the panic under test")]
        let panicked = queue.submit(JobSpec::new(JobKind::Graph, "boom"), |_| async {
            panic!("boom")
        });
        let panicked = finished(&queue, panicked).await;
        assert_eq!(panicked.state, JobState::Failed);
        // The worker slot came back.
        let after = queue.submit(JobSpec::new(JobKind::Sql, "after"), |_| async {
            Ok(String::new())
        });
        assert_eq!(finished(&queue, after).await.state, JobState::Succeeded);
    }

    #[tokio::test]
    async fn history_keeps_the_newest_finished_jobs() {
        let queue = JobQueue::new(4, 2);
        let mut last = None;
        for n in 0..5 {
            let id = queue.submit(JobSpec::new(JobKind::Sql, format!("{n}")), |_| async {
                Ok(String::new())
            });
            finished(&queue, id).await;
            last = Some(id);
        }
        let listed = queue.list();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed.last().map(|j| j.id), last);
        assert_eq!(
            JobId::parse(&listed.first().map(|j| j.id.to_string()).unwrap_or_default()),
            listed.first().map(|j| j.id)
        );
    }
}
