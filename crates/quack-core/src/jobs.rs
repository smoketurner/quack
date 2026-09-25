//! Work queues: the asynchronous model every interface shares (design doc
//! 4.1).
//!
//! Anything slower than a keystroke — an agent turn, a SQL statement, an
//! ingest, an import, an ontology or graph run — is submitted as a job and
//! runs in the background while the interface stays responsive. The queue
//! does not count jobs against a pool: what a job waits on is the resource
//! it uses — a model provider's `max_concurrent_requests`
//! (`llm::LimitedHttp`), the workspace's one writer connection, the reader
//! pool — so a turn paused on a permission prompt or a slow model holds
//! nothing a quick `SELECT` needs. What the queue does decide is order: a
//! job may name a [`Lane`], and jobs sharing a lane key run at most `limit`
//! at a time, in the order they were submitted. A chat session is a serial
//! lane (a turn's history includes the turn before it), a workspace's
//! uploads are a lane of `[server].workers_per_workspace`, and anything
//! without a lane starts at once.
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
use tokio::sync::{broadcast, oneshot};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::config::JobsConfig;
use crate::ids::{SessionId, UserId, WorkspaceId};
use crate::llm::acting::Acting;
use crate::priority::Priority;

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
}

/// Reads an id as [`fmt::Display`] writes it.
impl std::str::FromStr for JobId {
    type Err = uuid::Error;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Uuid::parse_str(text.trim()).map(Self)
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

/// A job's short number, counting from 1 per queue, for people to type
/// (`/cancel 3`): v7 ids submitted together share their first digits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct JobNumber(u64);

impl JobNumber {
    const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl fmt::Display for JobNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Reads `3` or `#3`, the way a job list shows it.
impl std::str::FromStr for JobNumber {
    type Err = std::num::ParseIntError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let text = text.trim();
        text.strip_prefix('#').unwrap_or(text).parse().map(Self)
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
    /// Vectors brought up to the current embedding profile.
    Embeddings,
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
            Self::Embeddings => "embeddings",
            Self::Export => "export",
        }
    }
}

impl JobKind {
    /// The priority its model requests and workspace writes run at: work
    /// someone is watching (a turn, a statement) is interactive, the rest
    /// background (design doc 4.1).
    #[must_use]
    pub const fn priority(self) -> Priority {
        match self {
            Self::Chat | Self::Sql => Priority::Interactive,
            Self::Ingest
            | Self::Import
            | Self::Ontology
            | Self::Graph
            | Self::Embeddings
            | Self::Export => Priority::Background,
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

/// What a lane keeps in order: one key per resource whose work must not
/// overlap or run out of order.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LaneKey {
    /// One chat session's turns, answered in the order asked.
    Session(SessionId),
    /// A workspace's uploads.
    Ingest(WorkspaceId),
    /// A workspace's graph extraction.
    Graph(WorkspaceId),
    /// A workspace's ontology document pass.
    Ontology(WorkspaceId),
    /// A workspace's embedding refresh.
    Embeddings(WorkspaceId),
}

impl fmt::Display for LaneKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Session(id) => write!(f, "session:{id}"),
            Self::Ingest(workspace) => write!(f, "ingest:{workspace}"),
            Self::Graph(workspace) => write!(f, "graph:{workspace}"),
            Self::Ontology(workspace) => write!(f, "ontology:{workspace}"),
            Self::Embeddings(workspace) => write!(f, "embeddings:{workspace}"),
        }
    }
}

impl Lane {
    /// One job at a time: a chat session, a workspace's graph extraction.
    #[must_use]
    pub fn serial(key: &LaneKey) -> Self {
        Self::new(key, 1)
    }

    /// Up to `limit` at a time (at least one). The first job submitted on
    /// a key fixes its limit while any job holds the lane.
    #[must_use]
    pub fn new(key: &LaneKey, limit: u32) -> Self {
        Self {
            key: key.to_string(),
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
    workspace_id: Option<WorkspaceId>,
    owner: Option<UserId>,
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
    pub fn workspace(mut self, workspace_id: WorkspaceId) -> Self {
        self.workspace_id = Some(workspace_id);
        self
    }

    /// The user who submitted it (server mode).
    #[must_use]
    pub fn owner(mut self, user_id: Option<UserId>) -> Self {
        self.owner = user_id;
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

impl JobProgress {
    /// The share done, in whole percent rounded down, so 100 means finished;
    /// `None` when there is nothing to count.
    #[must_use]
    pub fn percent(self) -> Option<u64> {
        u64::from(self.done)
            .saturating_mul(100)
            .checked_div(u64::from(self.total))
    }
}

/// `1576/3835 (41%)`, or `0/0` when there is nothing to count.
impl fmt::Display for JobProgress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.done, self.total)?;
        match self.percent() {
            Some(percent) => write!(f, " ({percent}%)"),
            None => Ok(()),
        }
    }
}

/// A job as it stands: what every subscriber receives on each change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JobInfo {
    pub id: JobId,
    pub number: JobNumber,
    pub kind: JobKind,
    pub label: String,
    pub workspace_id: Option<WorkspaceId>,
    pub owner: Option<UserId>,
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
    /// (an agent turn passes it in its `TurnRequest`).
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
    /// The number the latest job was given.
    last_number: JobNumber,
}

struct Inner {
    lanes: Mutex<HashMap<String, LaneState>>,
    registry: Mutex<Registry>,
    history: usize,
    events: broadcast::Sender<JobInfo>,
    /// Cleanup waiters for [`JobQueue::when_ended`]: `finish` hands the job's
    /// final snapshot directly to the waiter through this map, so the callback
    /// runs even if the job is dropped from the history ring in the same
    /// `finish` call that announced its end (or the broadcast channel lags).
    ended_waiters: Mutex<HashMap<JobId, oneshot::Sender<JobInfo>>>,
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
    /// history bound. The finished snapshot is captured before any eviction
    /// and handed to any [`JobQueue::when_ended`] waiter, so cleanup runs
    /// even for a job this call drops from the history ring.
    fn finish(&self, id: JobId, state: JobState, outcome: Option<String>) {
        self.update(id, |info| {
            info.state = state;
            info.outcome = outcome;
            info.finished_at = Some(Timestamp::now());
        });
        let snapshot = {
            let mut registry = self.registry();
            let snapshot = registry.jobs.get(&id).map(|e| e.info.clone());
            let finished = registry
                .jobs
                .values()
                .filter(|e| e.info.state.is_finished())
                .count();
            let mut excess = finished.saturating_sub(self.history);
            if excess > 0 {
                let Registry { order, jobs, .. } = &mut *registry;
                order.retain(|jid| {
                    let drop_it =
                        excess > 0 && jobs.get(jid).is_none_or(|e| e.info.state.is_finished());
                    if drop_it {
                        excess = excess.saturating_sub(1);
                        jobs.remove(jid);
                    }
                    !drop_it
                });
            }
            snapshot
        };
        // Deliver the finished snapshot to any registered waiter. The
        // snapshot was captured before eviction, so a job dropped from the
        // history ring above still reaches its cleanup callback.
        if let Some(info) = snapshot
            && let Some(tx) = self.ended_waiters().remove(&id)
        {
            drop(tx.send(info));
        }
    }

    fn lanes(&self) -> std::sync::MutexGuard<'_, HashMap<String, LaneState>> {
        self.lanes.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn ended_waiters(&self) -> std::sync::MutexGuard<'_, HashMap<JobId, oneshot::Sender<JobInfo>>> {
        self.ended_waiters
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
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
                Err(unclaimed) => unclaimed.disarm(),
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
    /// The queue and lane key to hand the slot back to; `None` once
    /// disarmed.
    lane: Option<(Arc<Inner>, String)>,
}

impl LanePermit {
    fn new(inner: &Arc<Inner>, key: &str) -> Self {
        Self {
            lane: Some((Arc::clone(inner), key.to_owned())),
        }
    }

    /// Drop without passing the slot on: the lane already counted it.
    fn disarm(mut self) {
        self.lane = None;
    }
}

impl Drop for LanePermit {
    fn drop(&mut self) {
        if let Some((inner, key)) = self.lane.take() {
            inner.leave_lane(&key);
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
    /// A queue remembering the last `history` finished jobs (at least one).
    #[must_use]
    pub fn new(history: u32) -> Self {
        let (events, _) = broadcast::channel(EVENT_CAPACITY);
        Self {
            inner: Arc::new(Inner {
                lanes: Mutex::new(HashMap::new()),
                registry: Mutex::new(Registry::default()),
                history: usize::try_from(history.max(1)).unwrap_or(usize::MAX),
                events,
                ended_waiters: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// The queue `[jobs]` describes.
    #[must_use]
    pub fn from_config(config: &JobsConfig) -> Self {
        Self::new(config.history)
    }

    /// Jobs queued or running in lane `key`: a new one there waits behind
    /// them (up to the lane's limit).
    #[must_use]
    pub fn lane_active(&self, key: &LaneKey) -> usize {
        let key = key.to_string();
        self.inner
            .registry()
            .jobs
            .values()
            .filter(|e| e.info.lane.as_deref() == Some(key.as_str()) && !e.info.state.is_finished())
            .count()
    }

    /// Queue `work` and return its id at once. It starts at once, or when
    /// its lane has room; a panic inside it is a failed job, not a lost
    /// one.
    ///
    /// Must be called inside a Tokio runtime.
    pub fn submit<F, Fut>(&self, spec: JobSpec, work: F) -> JobInfo
    where
        F: FnOnce(JobContext) -> Fut + Send + 'static,
        Fut: Future<Output = JobResult> + Send + 'static,
    {
        let id = JobId::new();
        let kind = spec.kind;
        let cancel = CancellationToken::new();
        let snapshot = {
            let mut registry = self.inner.registry();
            registry.last_number = registry.last_number.next();
            let info = JobInfo {
                id,
                number: registry.last_number,
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
        drop(self.inner.events.send(snapshot.clone()));

        // The lane place is taken now, so the lane runs in submission order.
        let ticket = spec.lane.as_ref().map(|lane| self.inner.enter_lane(lane));
        // The job acts for whoever submitted it: its model requests reach an
        // on-behalf-of provider as that person, not as whoever runs next.
        let acting = Acting::current();
        let work = move |ctx: JobContext| Acting::scope(acting, work(ctx));
        let inner = Arc::clone(&self.inner);
        tokio::spawn(async move {
            let ctx = JobContext {
                id,
                cancel: cancel.clone(),
                inner: Arc::clone(&inner),
            };
            inner.run(ticket, ctx, kind, work).await;
        });
        snapshot
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
    pub fn list_workspace(&self, workspace_id: &WorkspaceId) -> Vec<JobInfo> {
        self.list()
            .into_iter()
            .filter(|j| j.workspace_id.as_ref() == Some(workspace_id))
            .collect()
    }

    #[must_use]
    pub fn get(&self, id: JobId) -> Option<JobInfo> {
        self.inner.registry().jobs.get(&id).map(|e| e.info.clone())
    }

    /// The job shown as `number`.
    #[must_use]
    pub fn by_number(&self, number: JobNumber) -> Option<JobInfo> {
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
    pub fn counts(&self, workspace_id: Option<&WorkspaceId>) -> JobCounts {
        let registry = self.inner.registry();
        let mut counts = JobCounts::default();
        for entry in registry.jobs.values() {
            if workspace_id.is_some_and(|ws| entry.info.workspace_id.as_ref() != Some(ws)) {
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

    /// Run `record` with the job's final snapshot once it ends, on a task
    /// of its own: for what the work would have recorded itself had it run
    /// to its end (a job cancelled while queued never runs it). The snapshot
    /// is delivered by [`Inner::finish`] directly through a oneshot, so
    /// `record` runs even if the job is dropped from the history ring or the
    /// broadcast channel has lagged past the finished event. A job that
    /// ended and was evicted before this is called has no snapshot left to
    /// deliver, so `record` does not run; the two real callers register
    /// immediately after [`JobQueue::submit`], so the job is never ended by
    /// then.
    pub fn when_ended<F, Fut>(&self, id: JobId, record: F)
    where
        F: FnOnce(JobInfo) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        self.inner.ended_waiters().insert(id, tx);
        // The job may have already finished (and possibly been evicted)
        // before this call. `finish` only delivers through the oneshot if it
        // runs after this insert, so resolve the race here so `record` runs
        // exactly once.
        match self.get(id) {
            Some(info) if info.state.is_finished() => {
                // Still in the history and finished. If `finish` ran before we
                // inserted, it did not see our waiter: take it and deliver
                // `record(info)` ourselves. If `finish` ran after we inserted,
                // it took the waiter and sent the snapshot to `rx` already --
                // fall through to await it.
                if self.inner.ended_waiters().remove(&id).is_some() {
                    tokio::spawn(async move {
                        record(info).await;
                    });
                    return;
                }
            }
            Some(_) => {
                // Still queued or running; `finish` will deliver via `rx`.
            }
            None => {
                // Evicted or unknown. If we still hold the waiter, the job
                // ended and was evicted before this call (no snapshot left):
                // drop the waiter and do nothing. If `finish` already took it
                // and sent, fall through to await `rx`.
                if self.inner.ended_waiters().remove(&id).is_some() {
                    return;
                }
            }
        }
        tokio::spawn(async move {
            if let Ok(ended) = rx.await {
                record(ended).await;
            }
        });
    }
}

impl JobInfo {
    /// Cancelled while still queued: its work never ran.
    #[must_use]
    pub fn never_started(&self) -> bool {
        self.state == JobState::Cancelled && self.started_at.is_none()
    }
}

/// The lane slot a running job holds; dropped after its end is recorded.
type Held = Option<LanePermit>;

/// How a job ended, and the lane slot it holds until that is recorded.
struct Ended {
    state: JobState,
    outcome: Option<String>,
    held: Held,
}

impl Inner {
    /// Wait for the lane, run the work, and record its end. The end is recorded
    /// while the lane slot is still held, so the next job in the lane never
    /// starts before its predecessor reads as finished.
    async fn run<F, Fut>(
        self: &Arc<Self>,
        lane: Option<LaneTicket>,
        ctx: JobContext,
        kind: JobKind,
        work: F,
    ) where
        F: FnOnce(JobContext) -> Fut + Send + 'static,
        Fut: Future<Output = JobResult> + Send + 'static,
    {
        let id = ctx.id;
        // The lane slot lives in `_held` until the end is recorded.
        let Ended {
            state,
            outcome,
            held: _held,
        } = self.run_held(lane, ctx, kind, work).await;
        self.finish(id, state, outcome);
    }

    async fn run_held<F, Fut>(
        &self,
        lane: Option<LaneTicket>,
        ctx: JobContext,
        kind: JobKind,
        work: F,
    ) -> Ended
    where
        F: FnOnce(JobContext) -> Fut + Send + 'static,
        Fut: Future<Output = JobResult> + Send + 'static,
    {
        let cancel = ctx.cancel_token();
        let cancelled_while_queued = || Ended {
            state: JobState::Cancelled,
            outcome: Some(String::from("cancelled before it started")),
            held: None,
        };
        let lane_permit: Option<LanePermit> = match lane {
            Some(LaneTicket::Ready(permit)) => Some(permit),
            Some(LaneTicket::Wait(receiver)) => tokio::select! {
                biased;
                () = cancel.cancelled() => return cancelled_while_queued(),
                permit = receiver => match permit {
                    Ok(permit) => Some(permit),
                    Err(_) => {
                        return Ended {
                            state: JobState::Failed,
                            outcome: Some(String::from("the lane closed")),
                            held: None,
                        };
                    }
                },
            },
            None => None,
        };
        let id = ctx.id;
        self.update(id, |info| {
            info.state = JobState::Running;
            info.started_at = Some(Timestamp::now());
        });
        // Its own task, so a panic in the work surfaces as a join error here,
        // at its kind's priority.
        let outcome = tokio::spawn(kind.priority().scope(work(ctx))).await;
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
        Ended {
            state,
            outcome: text,
            held: lane_permit,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    /// A job's `lane` shows its key as `kind:id`.
    #[test]
    fn lane_keys_read_as_kind_and_id() {
        let id = || WorkspaceId::from("w1");
        for (key, text) in [
            (LaneKey::Session(SessionId::from("w1")), "session:w1"),
            (LaneKey::Ingest(id()), "ingest:w1"),
            (LaneKey::Graph(id()), "graph:w1"),
            (LaneKey::Ontology(id()), "ontology:w1"),
            (LaneKey::Embeddings(id()), "embeddings:w1"),
        ] {
            assert_eq!(key.to_string(), text);
            assert_eq!(Lane::serial(&key).key(), text);
        }
    }

    #[test]
    fn progress_shows_percent_rounded_down() {
        let shown = |done, total| JobProgress { done, total }.to_string();
        assert_eq!(shown(1576, 3835), "1576/3835 (41%)");
        assert_eq!(shown(0, 5), "0/5 (0%)");
        assert_eq!(shown(3834, 3835), "3834/3835 (99%)");
        assert_eq!(shown(3835, 3835), "3835/3835 (100%)");
        assert_eq!(
            shown(u32::MAX, u32::MAX),
            format!("{0}/{0} (100%)", u32::MAX)
        );
        assert_eq!(shown(0, 0), "0/0");
    }

    async fn finished(queue: &JobQueue, id: JobId) -> JobInfo {
        tokio::time::timeout(Duration::from_secs(5), queue.wait(id))
            .await
            .unwrap_or_else(|_| fail("job did not finish"))
            .unwrap_or_else(|| fail("job forgotten"))
    }

    /// Whether every lane is forgotten within a few seconds. A job reads as
    /// finished just before it releases its lane slot (so the next job in the
    /// lane cannot start first), and on a multi-threaded runtime the release
    /// can land a moment after `finished` returns.
    async fn lanes_drained(queue: &JobQueue) -> bool {
        let empty = || {
            queue
                .inner
                .lanes
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty()
        };
        tokio::time::timeout(Duration::from_secs(5), async {
            while !empty() {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .is_ok()
    }

    #[tokio::test]
    async fn a_job_acts_for_whoever_submitted_it() {
        use crate::llm::acting::Acting;
        let queue = JobQueue::new(10);
        let submit = |label: &str| {
            queue.submit(JobSpec::new(JobKind::Ingest, label), |_| async {
                Ok(Acting::current()
                    .map_or_else(|| String::from("nobody"), |a| a.user().to_string()))
            })
        };
        let ada = Acting::request(async {
            Acting::fixed(UserId::from("ada"), Err("unused")).enter();
            submit("ada's upload")
        })
        .await;
        let anonymous = submit("the CLI's upload");
        assert_eq!(
            finished(&queue, ada.id).await.outcome.as_deref(),
            Some("ada")
        );
        assert_eq!(
            finished(&queue, anonymous.id).await.outcome.as_deref(),
            Some("nobody")
        );
    }

    #[tokio::test]
    async fn jobs_run_report_and_finish() {
        let queue = JobQueue::new(10);
        let mut events = queue.subscribe();
        let ok = queue
            .submit(
                JobSpec::new(JobKind::Sql, "select").workspace(WorkspaceId::from("ws")),
                |ctx| async move {
                    ctx.progress(1, 2);
                    ctx.status("halfway");
                    Ok(String::from("2 rows"))
                },
            )
            .id;
        let err = queue
            .submit(JobSpec::new(JobKind::Ingest, "bad.pdf"), |_| async {
                Err(String::from("not a pdf"))
            })
            .id;
        let ok = finished(&queue, ok).await;
        assert_eq!(ok.state, JobState::Succeeded);
        assert_eq!(ok.outcome.as_deref(), Some("2 rows"));
        assert_eq!(ok.progress, Some(JobProgress { done: 1, total: 2 }));
        assert_eq!(ok.status.as_deref(), Some("halfway"));
        assert!(ok.started_at.is_some() && ok.finished_at.is_some());
        let err = finished(&queue, err).await;
        assert_eq!(err.state, JobState::Failed);
        assert_eq!(err.outcome.as_deref(), Some("not a pdf"));
        assert_eq!(err.number, JobNumber(2));

        // The first event is the queued snapshot.
        let first = events.recv().await.unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(first.state, JobState::Queued);
        assert_eq!(queue.list_workspace(&WorkspaceId::from("ws")).len(), 1);
        assert_eq!(queue.counts(None).active(), 0);
        assert_eq!(queue.by_number(JobNumber(2)).map(|j| j.id), Some(err.id));
    }

    #[tokio::test]
    async fn a_serial_lane_runs_in_order_and_other_work_is_not_held_up() {
        let queue = JobQueue::new(50);
        let log = Arc::new(Mutex::new(Vec::new()));
        let gate = Arc::new(tokio::sync::Notify::new());
        let mut ids = Vec::new();
        for n in 0..3_u32 {
            let log = Arc::clone(&log);
            let gate = Arc::clone(&gate);
            ids.push(
                queue
                    .submit(
                        JobSpec::new(JobKind::Chat, format!("turn {n}"))
                            .lane(Lane::serial(&LaneKey::Session(SessionId::from("a")))),
                        move |_| async move {
                            if n == 0 {
                                gate.notified().await;
                            }
                            log.lock().unwrap_or_else(PoisonError::into_inner).push(n);
                            Ok(String::new())
                        },
                    )
                    .id,
            );
        }
        // Another lane is not held up by the first.
        let other = queue
            .submit(
                JobSpec::new(JobKind::Sql, "other")
                    .lane(Lane::serial(&LaneKey::Session(SessionId::from("b")))),
                |_| async { Ok(String::from("done")) },
            )
            .id;
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
        assert!(lanes_drained(&queue).await);

        // A lane two wide runs two at once and queues the third; work
        // outside any lane is never held up by it.
        let wide = JobQueue::new(10);
        let gate = Arc::new(tokio::sync::Notify::new());
        let mut held = Vec::new();
        for n in 0..3 {
            let gate = Arc::clone(&gate);
            held.push(
                wide.submit(
                    JobSpec::new(JobKind::Ingest, format!("{n}"))
                        .lane(Lane::new(&LaneKey::Ingest(WorkspaceId::from("w")), 2)),
                    move |_| async move {
                        gate.notified().await;
                        Ok(String::new())
                    },
                )
                .id,
            );
        }
        let free = wide
            .submit(JobSpec::new(JobKind::Sql, "select"), |_| async {
                Ok(String::new())
            })
            .id;
        assert_eq!(finished(&wide, free).await.state, JobState::Succeeded);
        tokio::time::sleep(Duration::from_millis(50)).await;
        let counts = wide.counts(None);
        assert_eq!((counts.running, counts.queued), (2, 1));
        assert_eq!(
            wide.lane_active(&LaneKey::Ingest(WorkspaceId::from("w"))),
            3
        );
        for _ in 0..3 {
            gate.notify_one();
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        for id in held {
            assert_eq!(finished(&wide, id).await.state, JobState::Succeeded);
        }
        assert_eq!(
            wide.lane_active(&LaneKey::Ingest(WorkspaceId::from("w"))),
            0
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_lane_keeps_submission_order_on_a_multi_threaded_runtime() {
        let queue = JobQueue::new(200);
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut ids = Vec::new();
        for n in 0..50_u32 {
            let log = Arc::clone(&log);
            ids.push(
                queue
                    .submit(
                        JobSpec::new(JobKind::Chat, format!("{n}"))
                            .lane(Lane::serial(&LaneKey::Session(SessionId::from("x")))),
                        move |_| async move {
                            log.lock().unwrap_or_else(PoisonError::into_inner).push(n);
                            Ok(String::new())
                        },
                    )
                    .id,
            );
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
        assert!(lanes_drained(&queue).await);
    }

    #[tokio::test]
    async fn cancel_stops_queued_and_running_jobs_and_panics_fail() {
        let queue = JobQueue::new(10);
        let running = queue
            .submit(
                JobSpec::new(JobKind::Chat, "long")
                    .lane(Lane::serial(&LaneKey::Session(SessionId::from("c")))),
                |ctx| async move {
                    ctx.cancel_token().cancelled().await;
                    Err(String::from("stopped"))
                },
            )
            .id;
        let queued = queue
            .submit(
                JobSpec::new(JobKind::Chat, "never")
                    .lane(Lane::serial(&LaneKey::Session(SessionId::from("c")))),
                |_| async { Ok(String::from("ran")) },
            )
            .id;
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
        let panicked = queue
            .submit(JobSpec::new(JobKind::Graph, "boom"), |_| async {
                panic!("boom")
            })
            .id;
        let panicked = finished(&queue, panicked).await;
        assert_eq!(panicked.state, JobState::Failed);
        // The worker slot came back.
        let after = queue
            .submit(JobSpec::new(JobKind::Sql, "after"), |_| async {
                Ok(String::new())
            })
            .id;
        assert_eq!(finished(&queue, after).await.state, JobState::Succeeded);
    }

    #[tokio::test]
    async fn history_keeps_the_newest_finished_jobs() {
        let queue = JobQueue::new(2);
        let mut last = None;
        for n in 0..5 {
            let id = queue
                .submit(JobSpec::new(JobKind::Sql, format!("{n}")), |_| async {
                    Ok(String::new())
                })
                .id;
            finished(&queue, id).await;
            last = Some(id);
        }
        let listed = queue.list();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed.last().map(|j| j.id), last);
        assert_eq!(
            listed
                .first()
                .map(|j| j.id.to_string())
                .unwrap_or_default()
                .parse::<JobId>()
                .ok(),
            listed.first().map(|j| j.id)
        );
    }

    /// `when_ended`'s cleanup must run even when the job is evicted from the
    /// history ring in its own `finish` call. `a` queues behind a held job in
    /// a serial lane, so it is submitted before 100 lane-less jobs that all
    /// finish first; when `a` is then cancelled it is the oldest-by-submission
    /// finished job and is evicted in the very `finish` that announced its end.
    #[tokio::test(flavor = "current_thread")]
    async fn when_ended_runs_for_a_job_evicted_in_its_own_finish() {
        let queue = JobQueue::new(100);
        let lane_key = LaneKey::Session(SessionId::from("z"));
        let gate = Arc::new(tokio::sync::Notify::new());
        let gate_for_work = Arc::clone(&gate);
        let held = queue
            .submit(
                JobSpec::new(JobKind::Chat, "held").lane(Lane::serial(&lane_key)),
                move |_| async move {
                    gate_for_work.notified().await;
                    Ok(String::new())
                },
            )
            .id;
        let a = queue
            .submit(
                JobSpec::new(JobKind::Chat, "a").lane(Lane::serial(&lane_key)),
                |_| async { Ok(String::new()) },
            )
            .id;
        for n in 0..100_u32 {
            let id = queue
                .submit(JobSpec::new(JobKind::Sql, format!("b{n}")), |_| async {
                    Ok(String::new())
                })
                .id;
            finished(&queue, id).await;
        }

        let done = Arc::new(tokio::sync::Notify::new());
        let seen = Arc::new(Mutex::new(None::<JobInfo>));
        let done_for_record = Arc::clone(&done);
        let seen_for_record = Arc::clone(&seen);
        queue.when_ended(a, move |ended| {
            let done = Arc::clone(&done_for_record);
            let seen = Arc::clone(&seen_for_record);
            async move {
                *seen.lock().unwrap_or_else(PoisonError::into_inner) = Some(ended);
                done.notify_one();
            }
        });

        let wait_for_record = tokio::time::timeout(Duration::from_secs(5), done.notified());
        queue.cancel(a);
        match wait_for_record.await {
            Ok(()) => {}
            Err(_) => fail("when_ended record never ran at default history"),
        }

        gate.notify_one();
        assert!(
            tokio::time::timeout(Duration::from_secs(5), queue.wait(held))
                .await
                .is_ok(),
            "held did not settle"
        );

        let Some(ended) = seen.lock().unwrap_or_else(PoisonError::into_inner).clone() else {
            fail("no snapshot captured")
        };
        assert_eq!(ended.state, JobState::Cancelled);
        assert!(
            ended.never_started(),
            "evicted job's cleanup got its snapshot"
        );
        assert!(queue.get(a).is_none(), "a was evicted yet record still ran");
    }

    /// The same eviction shape as above, on the production multi-threaded
    /// runtime. The evicting thread is strongly favored to re-lock before a
    /// broadcast-woken watcher resumes, so the bug fires consistently here.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn when_ended_runs_for_a_job_evicted_on_a_multi_threaded_runtime() {
        let history = 100;
        let queue = JobQueue::new(history);
        let lane_key = LaneKey::Session(SessionId::from("m"));
        let gate = Arc::new(tokio::sync::Notify::new());
        let gate_for_work = Arc::clone(&gate);
        let holder = queue
            .submit(
                JobSpec::new(JobKind::Chat, "holder").lane(Lane::serial(&lane_key)),
                move |_| async move {
                    gate_for_work.notified().await;
                    Ok(String::new())
                },
            )
            .id;
        let a = queue
            .submit(
                JobSpec::new(JobKind::Chat, "a").lane(Lane::serial(&lane_key)),
                |_| async { Ok(String::new()) },
            )
            .id;
        for n in 0..history {
            let id = queue
                .submit(JobSpec::new(JobKind::Sql, format!("b{n}")), |_| async {
                    Ok(String::new())
                })
                .id;
            finished(&queue, id).await;
        }

        let done = Arc::new(tokio::sync::Notify::new());
        let seen = Arc::new(Mutex::new(None::<JobInfo>));
        let done_for_record = Arc::clone(&done);
        let seen_for_record = Arc::clone(&seen);
        queue.when_ended(a, move |ended| {
            let done = Arc::clone(&done_for_record);
            let seen = Arc::clone(&seen_for_record);
            async move {
                *seen.lock().unwrap_or_else(PoisonError::into_inner) = Some(ended);
                done.notify_one();
            }
        });

        let wait_for_record = tokio::time::timeout(Duration::from_secs(5), done.notified());
        queue.cancel(a);
        match wait_for_record.await {
            Ok(()) => {}
            Err(_) => fail("when_ended record never ran on multi_thread"),
        }

        gate.notify_one();
        assert!(
            tokio::time::timeout(Duration::from_secs(5), queue.wait(holder))
                .await
                .is_ok(),
            "holder did not settle"
        );

        let Some(ended) = seen.lock().unwrap_or_else(PoisonError::into_inner).clone() else {
            fail("no snapshot captured")
        };
        assert_eq!(ended.state, JobState::Cancelled);
        assert!(ended.never_started());
        assert!(queue.get(a).is_none(), "a was evicted yet record still ran");
    }

    /// Many jobs sharing one serial lane, all queued behind a held job, each
    /// with a `when_ended` watcher registered while queued. As they finish in
    /// order the oldest are evicted past a small history bound; every watcher
    /// must run its `record` exactly once.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn when_ended_runs_record_exactly_once_per_job_under_eviction() {
        let history = 8;
        let queue = JobQueue::new(history);
        let lane_key = LaneKey::Session(SessionId::from("race"));
        let gate = Arc::new(tokio::sync::Notify::new());
        let gate_for_work = Arc::clone(&gate);
        let _holder = queue
            .submit(
                JobSpec::new(JobKind::Chat, "holder").lane(Lane::serial(&lane_key)),
                move |_| async move {
                    gate_for_work.notified().await;
                    Ok(String::new())
                },
            )
            .id;
        let n = 64_u32;
        let called = Arc::new(AtomicUsize::new(0));
        let mut ids = Vec::new();
        for i in 0..n {
            let info = queue.submit(
                JobSpec::new(JobKind::Chat, format!("j{i}")).lane(Lane::serial(&lane_key)),
                |_| async { Ok(String::new()) },
            );
            let id = info.id;
            let called_for_record = Arc::clone(&called);
            queue.when_ended(id, move |_ended| {
                let called = Arc::clone(&called_for_record);
                async move {
                    called.fetch_add(1, Ordering::SeqCst);
                }
            });
            ids.push(id);
        }
        gate.notify_one();
        for id in &ids {
            assert!(
                tokio::time::timeout(Duration::from_secs(5), queue.wait(*id))
                    .await
                    .is_ok(),
                "a watched job did not settle"
            );
        }
        let want = usize::try_from(n).unwrap_or(usize::MAX);
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if called.load(Ordering::SeqCst) == want {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .unwrap_or_else(|_| fail("not all when_ended records ran"));
        assert_eq!(
            called.load(Ordering::SeqCst),
            want,
            "each when_ended ran its record exactly once"
        );
        // Most watched jobs were evicted while their watcher was registered.
        let evicted = ids.iter().filter(|id| queue.get(**id).is_none()).count();
        let kept_bound = usize::try_from(history).unwrap_or(want);
        assert!(
            evicted >= want.saturating_sub(kept_bound),
            "test exercised eviction: only {evicted} of {want} were evicted"
        );
    }

    /// `when_ended` registered after the job has already finished but is still
    /// in the history: the fast path delivers `record` from the current
    /// snapshot.
    #[tokio::test]
    async fn when_ended_runs_when_registered_after_finish_while_still_in_history() {
        let queue = JobQueue::new(100);
        let job = queue
            .submit(JobSpec::new(JobKind::Sql, "ok"), |_| async {
                Ok(String::from("2 rows"))
            })
            .id;
        let finished_info = finished(&queue, job).await;
        assert_eq!(finished_info.state, JobState::Succeeded);

        let done = Arc::new(tokio::sync::Notify::new());
        let seen = Arc::new(Mutex::new(None::<JobInfo>));
        let done_for_record = Arc::clone(&done);
        let seen_for_record = Arc::clone(&seen);
        queue.when_ended(job, move |ended| {
            let done = Arc::clone(&done_for_record);
            let seen = Arc::clone(&seen_for_record);
            async move {
                *seen.lock().unwrap_or_else(PoisonError::into_inner) = Some(ended);
                done.notify_one();
            }
        });
        match tokio::time::timeout(Duration::from_secs(5), done.notified()).await {
            Ok(()) => {}
            Err(_) => fail("when_ended did not run for an already-finished job"),
        }
        let Some(ended) = seen.lock().unwrap_or_else(PoisonError::into_inner).clone() else {
            fail("no snapshot captured")
        };
        assert_eq!(ended.state, JobState::Succeeded);
        assert_eq!(ended.outcome.as_deref(), Some("2 rows"));
    }

    /// `when_ended` for a job that ended and was evicted before the call has
    /// no snapshot left to deliver: `record` must not run, and the call must
    /// not leave a task parked on a oneshot that is never sent to.
    #[tokio::test]
    async fn when_ended_does_not_run_or_hang_after_the_job_was_evicted() {
        let queue = JobQueue::new(1);
        let first = queue
            .submit(JobSpec::new(JobKind::Sql, "first"), |_| async {
                Ok(String::new())
            })
            .id;
        let second = queue
            .submit(JobSpec::new(JobKind::Sql, "second"), |_| async {
                Ok(String::new())
            })
            .id;
        finished(&queue, second).await;
        assert!(
            queue.get(first).is_none(),
            "first was evicted (history = 1)"
        );

        let called = Arc::new(AtomicBool::new(false));
        let called_for_record = Arc::clone(&called);
        // Wrap the call in a timeout to prove it returns promptly rather than
        // parking a task on a oneshot that nothing will ever send to.
        let spawn = tokio::spawn(async move {
            queue.when_ended(first, move |_ended| {
                let called = Arc::clone(&called_for_record);
                async move {
                    called.store(true, Ordering::SeqCst);
                }
            });
        });
        match tokio::time::timeout(Duration::from_secs(5), spawn).await {
            Ok(Ok(())) => {}
            Ok(Err(_)) => fail("when_ended task panicked"),
            Err(_) => fail("when_ended hung for an evicted job"),
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !called.load(Ordering::SeqCst),
            "record must not run for an already-evicted job with no snapshot"
        );
    }

    /// The snapshot `when_ended` delivers is the actual finished snapshot
    /// (state, outcome, `finished_at`), not a stale one, even when the job is
    /// evicted by a later `finish` before the watcher resumes.
    #[tokio::test]
    async fn when_ended_delivers_the_actual_finished_snapshot_for_an_evicted_job() {
        let queue = JobQueue::new(1);
        let done = Arc::new(tokio::sync::Notify::new());
        let seen = Arc::new(Mutex::new(None::<JobInfo>));
        let watched = queue
            .submit(JobSpec::new(JobKind::Sql, "watched"), |_| async {
                Ok(String::from("summary-xyz"))
            })
            .id;
        let done_for_record = Arc::clone(&done);
        let seen_for_record = Arc::clone(&seen);
        queue.when_ended(watched, move |ended| {
            let done = Arc::clone(&done_for_record);
            let seen = Arc::clone(&seen_for_record);
            async move {
                *seen.lock().unwrap_or_else(PoisonError::into_inner) = Some(ended);
                done.notify_one();
            }
        });
        let later = queue
            .submit(JobSpec::new(JobKind::Sql, "later"), |_| async {
                Ok(String::new())
            })
            .id;
        let _ = finished(&queue, later).await;
        match tokio::time::timeout(Duration::from_secs(5), done.notified()).await {
            Ok(()) => {}
            Err(_) => fail("when_ended did not run for an evicted job"),
        }
        let Some(ended) = seen.lock().unwrap_or_else(PoisonError::into_inner).clone() else {
            fail("no snapshot captured")
        };
        assert_eq!(ended.state, JobState::Succeeded);
        assert_eq!(ended.outcome.as_deref(), Some("summary-xyz"));
        assert!(ended.finished_at.is_some());
        assert!(
            queue.get(watched).is_none(),
            "watched was evicted but its snapshot was still delivered"
        );
    }
}
