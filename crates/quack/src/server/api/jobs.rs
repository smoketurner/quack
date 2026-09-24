//! A workspace's jobs (design doc 4.1): what is queued, running, and
//! recently finished, as a list, one job, a live SSE stream of changes, and
//! a cancel. Viewers see every job of the workspace; a question's text
//! shows only to whoever may read its session (its owner, or an owner or
//! admin), like the session itself.

use std::convert::Infallible;

use axum::Json;
use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use quack_core::ids::WorkspaceId;
use quack_core::jobs::{JobId, JobInfo, JobKind};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use tokio::sync::broadcast;

use super::StreamEvent;
use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::App;

/// Shown in place of a question the caller may not read.
const PRIVATE_QUESTION: &str = "a question in a private session";

/// What the caller may see of the workspace's jobs, and do to them.
impl Access {
    /// `job` as this caller may see it: another member's question is not
    /// shown.
    fn redact(&self, mut job: JobInfo) -> JobInfo {
        if job.kind == JobKind::Chat && !self.owns(job.owner.as_ref()) {
            job.label = String::from(PRIVATE_QUESTION);
            job.outcome = None;
            job.status = None;
        }
        job
    }

    /// The workspace's jobs, newest first, as the caller may see them.
    pub(crate) fn visible_jobs(&self, app: &App) -> Vec<JobInfo> {
        let mut jobs: Vec<JobInfo> = app
            .jobs
            .list_workspace(&self.workspace.id)
            .into_iter()
            .map(|job| self.redact(job))
            .collect();
        jobs.reverse();
        jobs
    }

    /// Whether the caller may cancel `job`: their own, or any as an owner
    /// or admin; and never without the write permission.
    pub(crate) fn may_cancel(&self, job: &JobInfo) -> bool {
        self.permits(Need::WRITE) && self.owns(job.owner.as_ref())
    }

    /// The job, if it belongs to this workspace.
    fn find_job(&self, app: &App, job: &str) -> ApiResult<JobInfo> {
        job.parse::<JobId>()
            .ok()
            .and_then(|id| app.jobs.get(id))
            .filter(|j| j.workspace_id.as_ref() == Some(&self.workspace.id))
            .ok_or_else(|| ApiError::not_found(format!("job '{job}' not found")))
    }

    /// The cancel the API and the web page share, audited either way.
    pub(crate) async fn cancel_job(&self, app: &App, job: &str) -> ApiResult<JobInfo> {
        let found = self.find_job(app, job)?;
        let job_id = found.id.to_string();
        if !self.may_cancel(&found) {
            self.audit(
                app,
                AuditAction::Cancel,
                Some(ResourceKind::Job.id(&job_id)),
                Outcome::Denied,
                None,
            )
            .await?;
            return Err(ApiError::forbidden(
                "only the job's owner, a workspace owner, or an admin may cancel it",
            ));
        }
        let active = app.jobs.cancel(found.id);
        self.audit(
            app,
            AuditAction::Cancel,
            Some(ResourceKind::Job.id(&job_id)),
            Outcome::Allowed,
            Some(serde_json::json!({ "kind": found.kind, "was_active": active })),
        )
        .await?;
        let current = app.jobs.get(found.id).unwrap_or(found);
        Ok(self.redact(current))
    }
}

pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, AuditAction::List, "jobs").await?;
    let jobs = access.visible_jobs(&app);
    let counts = app.jobs.counts(Some(&id));
    Ok(Json(serde_json::json!({
        "jobs": jobs,
        "queued": counts.queued,
        "running": counts.running,
    })))
}

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path((id, job)): Path<(WorkspaceId, String)>,
) -> ApiResult<Json<JobInfo>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, AuditAction::Show, "job").await?;
    let job = access.find_job(&app, &job)?;
    Ok(Json(access.redact(job)))
}

/// Ask a job to stop: a queued one never starts, a running one stops at
/// its next checkpoint (an agent turn is recorded as cancelled).
pub(crate) async fn cancel(
    State(app): State<App>,
    identity: Identity,
    Path((id, job)): Path<(WorkspaceId, String)>,
) -> ApiResult<Json<JobInfo>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    Ok(Json(access.cancel_job(&app, &job).await?))
}

/// Every change to the workspace's jobs as SSE `job` events (a `JobInfo`
/// each), starting with the current list as one `jobs` event. A client
/// that falls behind gets a fresh `jobs` event rather than a gap.
pub(crate) async fn stream(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, AuditAction::Stream, "jobs").await?;
    let receiver = app.jobs.subscribe();
    let first = StreamEvent::Jobs
        .event()
        .json_data(access.visible_jobs(&app))
        .unwrap_or_default();
    let state = (receiver, app, access, Some(first));
    let stream = futures::stream::unfold(state, |(mut receiver, app, access, first)| async move {
        if let Some(event) = first {
            return Some((Ok(event), (receiver, app, access, None)));
        }
        loop {
            match receiver.recv().await {
                Ok(job) if job.workspace_id.as_ref() == Some(&access.workspace.id) => {
                    let event = StreamEvent::Job
                        .event()
                        .json_data(access.redact(job))
                        .unwrap_or_default();
                    return Some((Ok(event), (receiver, app, access, None)));
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    let event = StreamEvent::Jobs
                        .event()
                        .json_data(access.visible_jobs(&app))
                        .unwrap_or_default();
                    return Some((Ok(event), (receiver, app, access, None)));
                }
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}
