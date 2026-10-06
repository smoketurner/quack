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
use std::time::Duration;

use jiff::Timestamp;
use serde::Serialize;
use tokio::sync::{broadcast, oneshot};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use uuid::Uuid;

use crate::config::JobsConfig;
use crate::ids::{SessionId, UserId, WorkspaceId};
use crate::llm::acting::Acting;
use crate::llm::egress::Egress;
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
    /// The models each provider lists, fetched for someone waiting on them.
    Models,
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
            Self::Models => "models",
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
            Self::Chat | Self::Sql | Self::Models => Priority::Interactive,
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

    /// The share done, from 0 to 1; 0 when there is nothing to count.
    #[must_use]
    pub fn ratio(self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (f64::from(self.done) / f64::from(self.total)).min(1.0)
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
pub type JobResult = Result<String, String>;

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
    /// Cleanup waiters for [`JobQueue::when_ended`], present only while the
    /// job is active: `finish` takes them as it ends the job.
    ended_waiters: HashMap<JobId, Vec<oneshot::Sender<JobInfo>>>,
    /// Set by [`JobQueue::shutdown`]: nothing submitted after it runs.
    closed: bool,
}

impl Registry {
    /// Drop the oldest finished jobs past `history`.
    fn evict_past(&mut self, history: usize) {
        let finished = self
            .jobs
            .values()
            .filter(|e| e.info.state.is_finished())
            .count();
        let mut excess = finished.saturating_sub(history);
        if excess == 0 {
            return;
        }
        let Self { order, jobs, .. } = self;
        order.retain(|jid| {
            let drop_it = excess > 0 && jobs.get(jid).is_none_or(|e| e.info.state.is_finished());
            if drop_it {
                excess = excess.saturating_sub(1);
                jobs.remove(jid);
            }
            !drop_it
        });
    }
}

struct Inner {
    lanes: Mutex<HashMap<String, LaneState>>,
    registry: Mutex<Registry>,
    history: usize,
    events: broadcast::Sender<JobInfo>,
    /// The [`JobQueue::when_ended`] tasks, so a shutdown can wait for what
    /// they record.
    recorders: TaskTracker,
}

impl Inner {
    fn registry(&self) -> std::sync::MutexGuard<'_, Registry> {
        // A panic while holding the lock leaves plain data behind; the
        // registry stays usable.
        self.registry.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Apply `f` to the job and broadcast the result. The send happens
    /// under the lock, so subscribers see changes in the order they applied.
    fn update(&self, id: JobId, f: impl FnOnce(&mut JobInfo)) {
        let mut registry = self.registry();
        let Some(entry) = registry.jobs.get_mut(&id) else {
            return;
        };
        f(&mut entry.info);
        // No subscribers is fine.
        drop(self.events.send(entry.info.clone()));
    }

    /// Record the job's end, hand the final snapshot to its
    /// [`JobQueue::when_ended`] waiters, and drop the oldest finished jobs
    /// past the history bound, all under one lock so a concurrent finish
    /// cannot evict the job in between.
    fn finish(&self, id: JobId, state: JobState, outcome: Option<String>) {
        let mut registry = self.registry();
        let Some(entry) = registry.jobs.get_mut(&id) else {
            return;
        };
        entry.info.state = state;
        entry.info.outcome = outcome;
        entry.info.finished_at = Some(Timestamp::now());
        let snapshot = entry.info.clone();
        drop(self.events.send(snapshot.clone()));
        for waiter in registry.ended_waiters.remove(&id).unwrap_or_default() {
            drop(waiter.send(snapshot.clone()));
        }
        registry.evict_past(self.history);
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
                recorders: TaskTracker::new(),
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
    /// one. After [`Self::shutdown`] the job is recorded as cancelled and
    /// `work` never runs.
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
            let mut info = JobInfo {
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
            let refused = registry.closed;
            if refused {
                info.state = JobState::Cancelled;
                info.outcome = Some(String::from(REFUSED_BY_SHUTDOWN));
                info.finished_at = Some(info.queued_at);
                info.cancel_requested = true;
            }
            registry.order.push_back(id);
            registry.jobs.insert(
                id,
                Entry {
                    info: info.clone(),
                    cancel: cancel.clone(),
                },
            );
            drop(self.inner.events.send(info.clone()));
            if refused {
                registry.evict_past(self.inner.history);
                return info;
            }
            info
        };

        // The lane place is taken now, so the lane runs in submission order.
        let ticket = spec.lane.as_ref().map(|lane| self.inner.enter_lane(lane));
        // The job acts for whoever submitted it: its model requests reach an
        // on-behalf-of provider as that person, not as whoever runs next.
        // It sends where its submitter may: the workspace's provider
        // allow-list goes with it, and a job submitted under none sends nothing.
        let (acting, egress) = (Acting::current(), Egress::current());
        let work = move |ctx: JobContext| Acting::scope(acting, Egress::scope(egress, work(ctx)));
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
            let mut registry = self.inner.registry();
            let Some(entry) = registry.jobs.get_mut(&id) else {
                return false;
            };
            if entry.info.state.is_finished() {
                return false;
            }
            entry.info.cancel_requested = true;
            drop(self.inner.events.send(entry.info.clone()));
            entry.cancel.clone()
        };
        token.cancel();
        true
    }

    /// Stop the queue for good. From here on a submitted job is recorded as
    /// cancelled without running. Every queued job is cancelled, every
    /// running job sees its token cancelled, and this waits up to `grace`
    /// for them to end and for what [`Self::when_ended`] records about
    /// them. Returns the jobs still active when the grace ran out, which
    /// the runtime drops at their next await.
    pub async fn shutdown(&self, grace: Duration) -> Vec<JobInfo> {
        let mut events = self.subscribe();
        let tokens: Vec<CancellationToken> = {
            let mut registry = self.inner.registry();
            registry.closed = true;
            registry
                .jobs
                .values_mut()
                .filter(|entry| !entry.info.state.is_finished())
                .map(|entry| {
                    entry.info.cancel_requested = true;
                    drop(self.inner.events.send(entry.info.clone()));
                    entry.cancel.clone()
                })
                .collect()
        };
        for token in tokens {
            token.cancel();
        }
        self.inner.recorders.close();
        let settled = async {
            while self.counts(None).active() > 0 {
                if events.recv().await == Err(broadcast::error::RecvError::Closed) {
                    break;
                }
            }
            self.inner.recorders.wait().await;
        };
        if tokio::time::timeout(grace, settled).await.is_err() {
            tracing::warn!(
                recording = self.inner.recorders.len(),
                "the job queue's shutdown grace ran out"
            );
        }
        self.list()
            .into_iter()
            .filter(|job| !job.state.is_finished())
            .collect()
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
    /// to its end (a job cancelled while queued never runs it). `record`
    /// runs even if the job is later dropped from the history or the
    /// broadcast lags; a job already ended and dropped has no snapshot left,
    /// so `record` does not run.
    pub fn when_ended<F, Fut>(&self, id: JobId, record: F)
    where
        F: FnOnce(JobInfo) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        {
            let mut registry = self.inner.registry();
            let Some(entry) = registry.jobs.get(&id) else {
                return;
            };
            if entry.info.state.is_finished() {
                drop(tx.send(entry.info.clone()));
            } else {
                registry.ended_waiters.entry(id).or_default().push(tx);
            }
        }
        self.inner.recorders.spawn(async move {
            if let Ok(ended) = rx.await {
                record(ended).await;
            }
        });
    }
}

/// The outcome of a job submitted after [`JobQueue::shutdown`].
const REFUSED_BY_SHUTDOWN: &str = "refused: the queue is shutting down";

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
        // Cancelled before its task first ran: still queued, so it never starts.
        if cancel.is_cancelled() {
            return Ended {
                held: lane_permit,
                ..cancelled_while_queued()
            };
        }
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
mod tests;
