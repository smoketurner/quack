//! Audited background runs (design doc 4.1): embeddings refresh, graph
//! extraction, the ontology document pass, an import refreshed, a table
//! classified. Each starts on a request and finishes in the background, as
//! a job in its workspace's serial lane for its kind, and is audited twice
//! under one run id: when it starts and when it ends (or is cancelled
//! before it starts). The audit rows are written by whoever holds the
//! workspace: the server's `Access`, the desktop app's local identity; a
//! [`RunAudit`] is that writer.

use std::fmt;
use std::future::Future;

use serde_json::Value;

use crate::classify;
use crate::embedding::refresh;
use crate::graph::extract;
use crate::graph::follow_up::FollowUpSummary;
use crate::graph::resolve::ResolutionSummary;
use crate::ids::{RunId, UserId, WorkspaceId};
use crate::import::{ImportSummary, LoadStatus};
use crate::jobs::{JobContext, JobId, JobKind, JobQueue, JobSpec, Lane, LaneKey};
use crate::ontology::documents;
use crate::storage::control::{AuditAction, AuditResource, Outcome, ResourceKind};

/// What a background run is: the job kind whose workspace lane it takes,
/// how it is audited, and what the job list calls it. One row per run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunKind {
    job: JobKind,
    action: AuditAction,
    resource: ResourceKind,
    label: &'static str,
    /// What the closing row says of a run cancelled before it started.
    unstarted: &'static str,
}

/// What the closing row says of a run cancelled before it started.
const UNSTARTED: &str = "cancelled before it started";

impl RunKind {
    pub const EMBEDDINGS: Self = Self {
        job: JobKind::Embeddings,
        action: AuditAction::EmbeddingsRefresh,
        resource: ResourceKind::EmbeddingsRun,
        label: "embeddings refresh",
        unstarted: UNSTARTED,
    };
    pub const GRAPH: Self = Self {
        job: JobKind::Graph,
        action: AuditAction::GraphExtract,
        resource: ResourceKind::GraphRun,
        label: "graph extraction",
        unstarted: UNSTARTED,
    };
    /// The extraction that follows an ingest under `[graph].follow_ingest`.
    pub const FOLLOW_UP: Self = Self {
        job: JobKind::Graph,
        action: AuditAction::GraphExtract,
        resource: ResourceKind::GraphRun,
        label: "graph follow-up",
        unstarted: UNSTARTED,
    };
    /// A saved import run again (`POST .../imports/{id}/refresh`).
    pub const IMPORT_REFRESH: Self = Self {
        job: JobKind::Import,
        action: AuditAction::Import,
        resource: ResourceKind::SavedImport,
        label: "import refresh",
        unstarted: UNSTARTED,
    };
    /// A table's text labelled by the decision model.
    pub const CLASSIFY: Self = Self {
        job: JobKind::Classify,
        action: AuditAction::Classify,
        resource: ResourceKind::ClassificationRun,
        label: "classify",
        unstarted: "cancelled before it started; the questions were not kept",
    };
    pub const ONTOLOGY: Self = Self {
        job: JobKind::Ontology,
        action: AuditAction::Propose,
        resource: ResourceKind::InductionRun,
        label: "ontology document pass",
        unstarted: UNSTARTED,
    };

    /// The job kind whose workspace lane the run takes.
    #[must_use]
    pub fn job(&self) -> JobKind {
        self.job
    }

    /// What the job list calls the run.
    #[must_use]
    pub fn label(&self) -> &'static str {
        self.label
    }
}

/// What a finished run reports: the closing audit row's detail (beside
/// `finished: true`) and the job's one-line result.
pub trait RunReport: Send + 'static {
    fn detail(&self) -> Value;
    fn message(&self) -> String;
}

impl RunReport for refresh::Summary {
    fn detail(&self) -> Value {
        serde_json::json!({ "summary": self })
    }

    fn message(&self) -> String {
        format!(
            "{} chunks and {} graph node labels refreshed",
            self.chunks, self.nodes
        )
    }
}

impl RunReport for ImportSummary {
    fn detail(&self) -> Value {
        serde_json::json!({ "summary": self })
    }

    fn message(&self) -> String {
        match self.status {
            LoadStatus::Unchanged => format!("{}: source unchanged", self.table),
            LoadStatus::Loaded => format!("{}: {} rows, replaced", self.table, self.rows),
        }
    }
}

impl RunReport for classify::Run {
    fn detail(&self) -> Value {
        serde_json::json!({ "summary": self })
    }

    fn message(&self) -> String {
        self.to_string()
    }
}

/// A graph document pass: the extraction, then the resolution after it.
pub struct GraphReport {
    pub summary: extract::RunSummary,
    pub resolution: ResolutionSummary,
}

impl RunReport for GraphReport {
    fn detail(&self) -> Value {
        serde_json::json!({ "summary": self.summary, "resolution": self.resolution })
    }

    fn message(&self) -> String {
        format!(
            "{} nodes, {} edges from {} chunks",
            self.summary.nodes, self.summary.edges, self.summary.chunks
        )
    }
}

impl RunReport for FollowUpSummary {
    fn detail(&self) -> Value {
        serde_json::json!({ "summary": self })
    }

    fn message(&self) -> String {
        self.to_string()
    }
}

impl RunReport for documents::RunSummary {
    fn detail(&self) -> Value {
        serde_json::json!({ "summary": self })
    }

    fn message(&self) -> String {
        format!(
            "{} candidates from {} chunks",
            self.candidates, self.sampled_chunks
        )
    }
}

/// One audit row of a run: the opening row with the request's detail, or
/// the closing row with `finished: true` and the report.
#[derive(Debug, Clone)]
pub struct RunRow {
    pub action: AuditAction,
    pub resource: ResourceKind,
    pub run: RunId,
    pub outcome: Outcome,
    pub detail: Value,
}

impl RunRow {
    /// The run, as the row names it.
    #[must_use]
    pub fn resource(&self) -> AuditResource<'_> {
        self.resource.clone().id(&self.run)
    }
}

/// Who writes a run's audit rows, the `audit_log` row and its
/// `_quack_audit` detail under one id: the server's `Access` for the person
/// whose request started the run, the app's local identity on channel
/// `desktop`.
pub trait RunAudit: Clone + Send + Sync + 'static {
    /// Why a row could not be written; the request fails with it when it is
    /// the opening row, and it is logged when it is the closing one.
    type Error: fmt::Display + Send;

    fn record(&self, row: RunRow) -> impl Future<Output = Result<(), Self::Error>> + Send;
}

/// Whose run it is: the workspace whose lanes it takes and the person the
/// job list shows as its owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOwner {
    pub workspace: WorkspaceId,
    pub user: Option<UserId>,
}

/// One run, from its start audit row to its closing one.
#[derive(Clone)]
pub struct BackgroundRun<A: RunAudit> {
    jobs: JobQueue,
    audit: A,
    owner: RunOwner,
    id: RunId,
    kind: RunKind,
}

impl<A: RunAudit> BackgroundRun<A> {
    /// Mint the run id and write the start audit row with `detail`.
    ///
    /// # Errors
    ///
    /// The auditor's error when the start row could not be written: the
    /// request fails, and no job is queued.
    pub async fn start(
        jobs: &JobQueue,
        audit: A,
        owner: RunOwner,
        kind: RunKind,
        detail: Value,
    ) -> Result<Self, A::Error> {
        let id = RunId::generate();
        audit
            .record(RunRow {
                action: kind.action.clone(),
                resource: kind.resource.clone(),
                run: id.clone(),
                outcome: Outcome::Allowed,
                detail,
            })
            .await?;
        Ok(Self {
            jobs: jobs.clone(),
            audit,
            owner,
            id,
            kind,
        })
    }

    #[must_use]
    pub fn id(&self) -> &RunId {
        &self.id
    }

    #[must_use]
    pub fn kind(&self) -> &RunKind {
        &self.kind
    }

    /// Run `work` as a job in the workspace's lane for this kind, then write
    /// the closing audit row from its result. A cancel before the job starts
    /// writes that row instead.
    pub fn submit<F, Fut, R>(self, work: F) -> JobId
    where
        F: FnOnce(JobContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<R, String>> + Send + 'static,
        R: RunReport,
    {
        let spec = JobSpec::new(self.kind.job, self.kind.label)
            .workspace(self.owner.workspace.clone())
            .owner(self.owner.user.clone())
            .lane(Lane::serial(&LaneKey::Workspace(
                self.kind.job,
                self.owner.workspace.clone(),
            )));
        let jobs = self.jobs.clone();
        let unstarted = self.clone();
        let unstarted_note = self.kind.unstarted;
        let id = jobs
            .submit(spec, move |ctx| async move {
                let result = work(ctx).await;
                match result {
                    Ok(report) => {
                        let mut detail = report.detail();
                        if let Some(fields) = detail.as_object_mut() {
                            fields.insert(String::from("finished"), Value::Bool(true));
                        }
                        self.finish(Outcome::Allowed, detail).await;
                        Ok(report.message())
                    }
                    Err(e) => {
                        tracing::warn!(run = %self.id, kind = self.kind.label, error = %e, "background run failed");
                        self.finish(
                            Outcome::Error,
                            serde_json::json!({ "finished": true, "error": e }),
                        )
                        .await;
                        Err(e)
                    }
                }
            })
            .id;
        jobs.when_ended(id, move |ended| async move {
            if ended.never_started() {
                unstarted
                    .finish(
                        Outcome::Error,
                        serde_json::json!({ "finished": true, "error": unstarted_note }),
                    )
                    .await;
            }
        });
        id
    }

    /// The closing audit row. The job has already done its work, so a
    /// failed write is logged rather than returned.
    async fn finish(&self, outcome: Outcome, detail: Value) {
        let row = RunRow {
            action: self.kind.action.clone(),
            resource: self.kind.resource.clone(),
            run: self.id.clone(),
            outcome,
            detail,
        };
        if let Err(e) = self.audit.record(row).await {
            tracing::error!(run = %self.id, kind = self.kind.label, error = %e, "audit write failed at the end of a background run");
        }
    }
}
