//! The caller of one MCP tool call or resource read, and the one way a call
//! is audited: run the work, then record what happened to it. `record`, which
//! takes an outcome, is private to this module, so a tool cannot record
//! `Allowed` before its work runs or skip the row when the work fails.

use std::future::Future;

use quack_core::classify::LabelJobs;
use quack_core::error::Error as CoreError;
use quack_core::ids::{SessionId, UserId, WorkspaceId};
use quack_core::jobs::JobQueue;
use quack_core::llm::acting::Acting;
use quack_core::storage::control::{AuditAction, AuditResource, Outcome};
use quack_core::storage::sessions::SessionViewer;
use rmcp::ErrorData as McpError;
use rmcp::model::CallToolResult;

use super::{McpCaller, failure, internal};

/// How a failed piece of a call is recorded: denied when the workspace's
/// provider allow-list or the connection's permissions refused it, an error
/// otherwise.
pub(super) trait Failure {
    fn outcome(&self) -> Outcome;
}

impl Failure for CoreError {
    fn outcome(&self) -> Outcome {
        Outcome::of_failure(self)
    }
}

impl Failure for &CoreError {
    fn outcome(&self) -> Outcome {
        Outcome::of_failure(self)
    }
}

impl Failure for McpError {
    fn outcome(&self) -> Outcome {
        Outcome::Error
    }
}

/// A call refused after authorization, with the tool failure it answers.
pub(super) struct Rejected {
    outcome: Outcome,
    pub(super) message: String,
}

impl Rejected {
    /// The connection may not do this.
    pub(super) fn denied(message: impl Into<String>) -> Self {
        Self {
            outcome: Outcome::Denied,
            message: message.into(),
        }
    }

    /// The request is malformed or names something that is not there.
    pub(super) fn invalid(message: impl Into<String>) -> Self {
        Self {
            outcome: Outcome::Error,
            message: message.into(),
        }
    }
}

impl Failure for Rejected {
    fn outcome(&self) -> Outcome {
        self.outcome
    }
}

/// The caller of one tool call or resource read.
#[derive(Clone)]
pub(super) enum Caller {
    /// Stdio or the socket: nobody to audit or act for.
    Unaudited,
    /// HTTP: the request's own caller.
    Audited(Box<McpCaller>),
}

impl Caller {
    /// Write the audit rows; a failed write fails the call, so nothing
    /// audited proceeds unlogged.
    async fn record(
        &self,
        action: AuditAction,
        resource: Option<AuditResource<'_>>,
        outcome: Outcome,
        detail: Option<serde_json::Value>,
    ) -> Result<(), McpError> {
        let Self::Audited(caller) = self else {
            return Ok(());
        };
        #[expect(
            clippy::disallowed_methods,
            reason = "the one place a call's outcome is passed in: `audited`, `guarded`, and `refused` derive it from the work's result"
        )]
        let recorded = caller
            .recorder
            .record(action.clone(), resource, outcome, detail)
            .await;
        recorded.map_err(|e| {
            tracing::error!(error = %e, %action, "audit write failed");
            internal(format!("audit write failed: {e}"))
        })?;
        Ok(())
    }

    /// Run `work`, then record it as `action`: allowed when it succeeded,
    /// otherwise as its failure says. The work's own result comes back, so
    /// a tool failure stays an ordinary tool result.
    pub(super) async fn audited<T, E: Failure>(
        &self,
        action: AuditAction,
        resource: Option<AuditResource<'_>>,
        detail: serde_json::Value,
        work: impl Future<Output = Result<T, E>>,
    ) -> Result<Result<T, E>, McpError> {
        let result = work.await;
        let outcome = match &result {
            Ok(_) => Outcome::Allowed,
            Err(e) => e.outcome(),
        };
        self.record(action, resource, outcome, Some(detail)).await?;
        Ok(result)
    }

    /// [`Caller::audited`] for work whose success is recorded by what comes
    /// next (a turn's own row): only a failure is recorded here.
    pub(super) async fn guarded<T, E: Failure>(
        &self,
        action: AuditAction,
        detail: serde_json::Value,
        work: impl Future<Output = Result<T, E>>,
    ) -> Result<Result<T, E>, McpError> {
        let result = work.await;
        if let Err(e) = &result {
            self.record(action, None, e.outcome(), Some(detail)).await?;
        }
        Ok(result)
    }

    /// Record a call that `failure` stopped before its work began.
    pub(super) async fn refused(
        &self,
        action: AuditAction,
        detail: serde_json::Value,
        failure: &impl Failure,
    ) -> Result<(), McpError> {
        self.record(action, None, failure.outcome(), Some(detail))
            .await
    }

    /// Audit the outcome of a `classify` call that started no run (a
    /// preview, or a call refused before the run began) and give its tool
    /// result.
    pub(super) async fn finish_label(
        &self,
        action: AuditAction,
        detail: serde_json::Value,
        done: Result<serde_json::Value, CoreError>,
    ) -> Result<CallToolResult, McpError> {
        let done = self.audited(action, None, detail, async { done }).await?;
        Ok(match done {
            Ok(body) => CallToolResult::structured(body),
            Err(e) => failure(e.to_string()),
        })
    }

    /// The tool result when `error` kept a model from being built, recorded
    /// as denied when the workspace's provider allow-list refused it and as
    /// an error otherwise.
    pub(super) async fn unbuilt(
        &self,
        action: AuditAction,
        detail: serde_json::Value,
        error: &CoreError,
    ) -> Result<CallToolResult, McpError> {
        self.refused(action, detail, &error).await?;
        Ok(failure(error.to_string()))
    }

    /// Which sessions the caller may read: every one over stdio, and the
    /// server user's over HTTP.
    pub(super) fn session_viewer(&self) -> SessionViewer {
        match self {
            Self::Unaudited => SessionViewer::All,
            Self::Audited(caller) => caller.session_viewer.clone(),
        }
    }

    /// The job queue a table this caller has labelled goes through; over
    /// HTTP, the end of each run is audited too.
    pub(super) fn label_jobs(
        &self,
        queue: &JobQueue,
        workspace: &WorkspaceId,
        session: Option<SessionId>,
    ) -> LabelJobs {
        let jobs = LabelJobs::new(queue.clone()).workspace(workspace.clone());
        match self {
            Self::Unaudited => jobs,
            Self::Audited(caller) => jobs.on_end(caller.recorder.on_classify_end(session)),
        }
    }

    /// The server user making the call; nobody over stdio.
    pub(super) fn user_id(&self) -> Option<UserId> {
        match self {
            Self::Unaudited => None,
            Self::Audited(caller) => Some(caller.recorder.user_id().clone()),
        }
    }

    /// Whom model requests are made for (over HTTP, the request's user at
    /// an on-behalf-of provider; over stdio, nobody).
    pub(super) fn acting(&self) -> Option<Acting> {
        match self {
            Self::Unaudited => None,
            Self::Audited(caller) => caller.acting.clone(),
        }
    }
}
