//! A long run the server starts on a request and finishes in the
//! background: embeddings refresh, graph document extraction, and the
//! ontology document pass. Each is audited twice under one run id, when it
//! starts and when it ends (or is cancelled before it starts), and runs as
//! a job in its workspace's serial lane for its kind.

use std::future::Future;
use std::sync::Arc;

use quack_core::embedding::refresh;
use quack_core::graph::extract;
use quack_core::graph::resolve::ResolutionSummary;
use quack_core::jobs::{JobContext, JobId, JobKind, JobSpec, Lane, LaneKey};
use quack_core::ontology::documents;
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use serde_json::Value;

use crate::server::auth::Access;
use crate::server::error::ApiResult;
use crate::server::state::App;

/// What a background run is: how it is audited, which queue it takes, and
/// what the job list calls it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunKind {
    Embeddings,
    Graph,
    Ontology,
}

impl RunKind {
    fn action(self) -> AuditAction {
        match self {
            Self::Embeddings => AuditAction::EmbeddingsRefresh,
            Self::Graph => AuditAction::GraphExtract,
            Self::Ontology => AuditAction::Propose,
        }
    }

    fn resource(self) -> ResourceKind {
        match self {
            Self::Embeddings => ResourceKind::EmbeddingsRun,
            Self::Graph => ResourceKind::GraphRun,
            Self::Ontology => ResourceKind::InductionRun,
        }
    }

    fn job(self) -> JobKind {
        match self {
            Self::Embeddings => JobKind::Embeddings,
            Self::Graph => JobKind::Graph,
            Self::Ontology => JobKind::Ontology,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Embeddings => "embeddings refresh",
            Self::Graph => "graph extraction",
            Self::Ontology => "ontology document pass",
        }
    }

    fn lane(self, workspace_id: &str) -> LaneKey {
        let workspace = workspace_id.to_owned();
        match self {
            Self::Embeddings => LaneKey::Embeddings(workspace),
            Self::Graph => LaneKey::Graph(workspace),
            Self::Ontology => LaneKey::Ontology(workspace),
        }
    }
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
    id: String,
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
        let id = uuid::Uuid::now_v7().to_string();
        access
            .audit(
                app,
                kind.action(),
                Some(kind.resource().id(&id)),
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

    pub(crate) fn id(&self) -> &str {
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
        let workspace_id = self.access.workspace.id.clone();
        let spec = JobSpec::new(self.kind.job(), self.kind.label())
            .workspace(workspace_id.clone())
            .owner(Some(self.access.identity.user_id.clone()))
            .lane(Lane::serial(&self.kind.lane(workspace_id.as_str())));
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
                    tracing::warn!(run = %self.id, kind = ?self.kind, error = %e, "background run failed");
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
                self.kind.action(),
                Some(self.kind.resource().id(&self.id)),
                outcome,
                Some(detail),
            )
            .await
        {
            tracing::error!(run = %self.id, kind = ?self.kind, error = %e.message, "audit write failed at the end of a background run");
        }
    }
}
