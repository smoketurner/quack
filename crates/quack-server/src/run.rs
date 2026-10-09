//! A long run the server starts on a request and finishes in the
//! background: embeddings refresh, graph document extraction, and the
//! ontology document pass. Each is audited twice under one run id, when it
//! starts and when it ends (or is cancelled before it starts), and runs as
//! a job in its workspace's serial lane for its kind.

use std::future::Future;
use std::sync::Arc;

use std::time::Duration;

use quack_core::analysis::tools::SharedDb;
use quack_core::classify::ClassificationRun;
use quack_core::embedding::refresh;
use quack_core::graph::extract;
use quack_core::graph::follow_up::{FollowUp, FollowUpSummary};
use quack_core::graph::resolve::ResolutionSummary;
use quack_core::ids::{DocumentId, RunId};
use quack_core::import::{ImportSummary, LoadStatus};
use quack_core::jobs::{JobContext, JobId, JobKind, JobSpec, Lane, LaneKey};
use quack_core::llm::Embeddings;
use quack_core::ontology::{documents, store as ontology_store};
use quack_core::progress::{ChunkDone, RunControl};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use serde_json::Value;

use crate::auth::Access;
use crate::error::ApiResult;
use crate::state::App;

/// What a background run is: the job kind whose workspace lane it takes,
/// how it is audited, and what the job list calls it. One row per run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RunKind {
    job: JobKind,
    action: AuditAction,
    resource: ResourceKind,
    label: &'static str,
}

impl RunKind {
    pub(crate) const EMBEDDINGS: Self = Self {
        job: JobKind::Embeddings,
        action: AuditAction::EmbeddingsRefresh,
        resource: ResourceKind::EmbeddingsRun,
        label: "embeddings refresh",
    };
    pub(crate) const GRAPH: Self = Self {
        job: JobKind::Graph,
        action: AuditAction::GraphExtract,
        resource: ResourceKind::GraphRun,
        label: "graph extraction",
    };
    /// The extraction that follows an ingest under `[graph].follow_ingest`.
    pub(crate) const FOLLOW_UP: Self = Self {
        job: JobKind::Graph,
        action: AuditAction::GraphExtract,
        resource: ResourceKind::GraphRun,
        label: "graph follow-up",
    };
    /// A saved import run again (`POST .../imports/{id}/refresh`).
    pub(crate) const IMPORT_REFRESH: Self = Self {
        job: JobKind::Import,
        action: AuditAction::Import,
        resource: ResourceKind::SavedImport,
        label: "import refresh",
    };
    /// A table's text labelled by the decision model.
    pub(crate) const CLASSIFY: Self = Self {
        job: JobKind::Classify,
        action: AuditAction::Classify,
        resource: ResourceKind::ClassificationRun,
        label: "classify",
    };
    pub(crate) const ONTOLOGY: Self = Self {
        job: JobKind::Ontology,
        action: AuditAction::Propose,
        resource: ResourceKind::InductionRun,
        label: "ontology document pass",
    };
}

/// What a finished run reports: the closing audit row's detail (beside
/// `finished: true`) and the job's one-line result.
pub(crate) trait RunReport: Send + 'static {
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

impl RunReport for ClassificationRun {
    fn detail(&self) -> Value {
        serde_json::json!({ "summary": self })
    }

    fn message(&self) -> String {
        self.to_string()
    }
}

/// A graph document pass: the extraction, then the resolution after it.
pub(crate) struct GraphReport {
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

/// One run, from its start audit row to its closing one.
#[derive(Clone)]
pub(crate) struct BackgroundRun {
    app: App,
    access: Access,
    id: RunId,
    kind: RunKind,
}

impl BackgroundRun {
    /// Mint the run id and write the start audit row with `detail`; the
    /// request fails if that write does.
    pub(crate) async fn start(
        app: &App,
        access: &Access,
        kind: RunKind,
        detail: Value,
    ) -> ApiResult<Self> {
        let id = RunId::generate();
        access
            .audit(
                app,
                kind.action.clone(),
                Some(kind.resource.clone().id(&id)),
                Outcome::Allowed,
                Some(detail),
            )
            .await?;
        Ok(Self {
            app: Arc::clone(app),
            access: access.clone(),
            id,
            kind,
        })
    }

    pub(crate) fn id(&self) -> &RunId {
        &self.id
    }

    /// Run `work` as a job in the workspace's lane for this kind, then write
    /// the closing audit row from its result. A cancel before the job starts
    /// writes that row instead.
    pub(crate) fn submit<F, Fut, R>(self, work: F) -> JobId
    where
        F: FnOnce(JobContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<R, String>> + Send + 'static,
        R: RunReport,
    {
        let workspace_id = self.access.membership.workspace.id.clone();
        let spec = JobSpec::new(self.kind.job, self.kind.label)
            .workspace(workspace_id.clone())
            .owner(Some(self.access.identity.user_id.clone()))
            .lane(Lane::serial(&LaneKey::Workspace(
                self.kind.job,
                workspace_id,
            )));
        let jobs = self.app.jobs.clone();
        let unstarted = self.clone();
        let id = jobs.submit(spec, move |ctx| async move {
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
        }).id;
        jobs.when_ended(id, move |ended| async move {
            if ended.never_started() {
                unstarted
                    .finish(
                        Outcome::Error,
                        serde_json::json!({ "finished": true, "error": "cancelled before it started" }),
                    )
                    .await;
            }
        });
        id
    }

    /// The closing audit row. The job has already done its work, so a
    /// failed write is logged rather than returned.
    async fn finish(&self, outcome: Outcome, detail: Value) {
        if let Err(e) = self
            .access
            .audit(
                &self.app,
                self.kind.action.clone(),
                Some(self.kind.resource.clone().id(&self.id)),
                outcome,
                Some(detail),
            )
            .await
        {
            tracing::error!(run = %self.id, kind = self.kind.label, error = %e.message, "audit write failed at the end of a background run");
        }
    }
}

/// How many seconds a follow-up waits for the workspace's extraction slot
/// before giving up: a request-time table extraction holds it briefly.
const SLOT_WAIT_SECONDS: u32 = 60;

/// Queue the graph extraction that follows `documents` becoming ready,
/// when `[graph].follow_ingest` asks for one: an audited background run
/// in the workspace's graph lane, so it waits behind an extraction in
/// progress. `None` when nothing follows.
pub(crate) async fn follow_ingest(
    app: &App,
    access: &Access,
    db: SharedDb,
    embedder: Option<Embeddings>,
    documents: Vec<DocumentId>,
) -> ApiResult<Option<JobId>> {
    if app.config.graph.follow_ingest.is_off() || documents.is_empty() {
        return Ok(None);
    }
    // Nothing to extract into without an ontology: no run, no audit rows.
    let workspace_id = access.membership.workspace.id.clone();
    if app
        .read(&workspace_id, ontology_store::latest_version)
        .await?
        .is_none()
    {
        return Ok(None);
    }
    let run = BackgroundRun::start(
        app,
        access,
        RunKind::FOLLOW_UP,
        serde_json::json!({
            "documents": documents,
            "follow_ingest": app.config.graph.follow_ingest.as_str(),
        }),
    )
    .await?;
    let app = Arc::clone(app);
    let job = run.submit(move |ctx| async move {
        let mut slot = None;
        for _ in 0..SLOT_WAIT_SECONDS {
            slot = app.begin_extraction(&workspace_id);
            if slot.is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        let Some(slot) = slot else {
            return Err(String::from(
                "a graph extraction is still running for this workspace; run `graph extract` later",
            ));
        };
        let progress = |done: ChunkDone| ctx.progress(done.done, done.total);
        let cancel = ctx.cancel_token();
        let control = RunControl {
            progress: &progress,
            cancel: Some(&cancel),
        };
        let outcome = FollowUp {
            db: &db,
            config: &app.config,
            embeddings: embedder.as_ref(),
        }
        .run(&documents, control)
        .await;
        drop(slot);
        match outcome {
            Ok(Some(summary)) => Ok(summary),
            Ok(None) => Ok(FollowUpSummary::default()),
            Err(e) => Err(e.to_string()),
        }
    });
    Ok(Some(job))
}
