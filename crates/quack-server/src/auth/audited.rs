//! The one way a handler audits a request's work: run it, then record what
//! happened to it. The forms that take a caller-chosen outcome stay private
//! to `auth`, so a handler cannot record `Allowed` before the work runs, or
//! skip the row when the work fails after the caller was authorized.

use axum::http::StatusCode;
use quack_core::analysis::events::Decision;
use quack_core::ids::AuditId;
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use std::future::Future;

use super::Access;
use crate::error::{ApiError, ApiResult};
use crate::state::App;

/// What a request's audit rows will say: the action, the resource it named,
/// and the detail kept inside the workspace.
#[derive(Debug, Clone)]
pub(crate) struct Audit {
    action: AuditAction,
    resource: Option<(ResourceKind, String)>,
    detail: Option<serde_json::Value>,
    /// The work wrote its own row (a change committed with its access row,
    /// a background run's start and end).
    already_recorded: bool,
    /// The work returned a result that records its own failure.
    ended_in_failure: bool,
    /// A failure of the work is a refusal, whatever its status.
    refused: bool,
}

impl Audit {
    pub(crate) const fn of(action: AuditAction) -> Self {
        Self {
            action,
            resource: None,
            detail: None,
            already_recorded: false,
            ended_in_failure: false,
            refused: false,
        }
    }

    /// A read that returns a listing or a page: what was read in the detail.
    pub(crate) fn read(action: AuditAction, what: &str) -> Self {
        Self::of(action).detail(serde_json::json!({ "what": what }))
    }

    #[must_use]
    pub(crate) fn detail(self, detail: serde_json::Value) -> Self {
        Self {
            detail: Some(detail),
            ..self
        }
    }

    #[must_use]
    pub(crate) fn resource(self, kind: ResourceKind, id: impl Into<String>) -> Self {
        Self {
            resource: Some((kind, id.into())),
            ..self
        }
    }

    /// The action the row records, once the work knows which it was.
    pub(crate) fn set_action(&mut self, action: AuditAction) {
        self.action = action;
    }

    /// Name what the work produced, once it is known.
    pub(crate) fn set_resource(&mut self, kind: ResourceKind, id: impl Into<String>) {
        self.resource = Some((kind, id.into()));
    }

    /// Replace the detail with one only the work can tell.
    pub(crate) fn set_detail(&mut self, detail: serde_json::Value) {
        self.detail = Some(detail);
    }

    /// The work has written the request's row itself, so nothing else is
    /// recorded whatever happens next: a change committed in one transaction
    /// with its access row, or a background run that audits its own start
    /// and end.
    pub(crate) fn already_recorded(&mut self) {
        self.already_recorded = true;
    }

    /// The work is refusing the request (a statement the caller may not
    /// run): its failure is recorded as denied, whatever status answers it.
    pub(crate) fn refused(&mut self) {
        self.refused = true;
    }

    /// The work returned normally with a result that says it failed (a
    /// saved question whose statement errored): the row is an error.
    pub(crate) fn ended_in_failure(&mut self) {
        self.ended_in_failure = true;
    }
}

impl ApiError {
    /// `Denied` for a refusal (a provider the workspace's allow-list
    /// refused, a check inside the handler that said no: every 403), `Error` for any
    /// other failure.
    fn outcome(&self) -> Outcome {
        if self.status == StatusCode::FORBIDDEN {
            Outcome::Denied
        } else {
            Outcome::Error
        }
    }
}

impl Access {
    /// Run `work`, then record its outcome under `audit`: `Allowed` when it
    /// succeeded, `Denied` when a provider was refused, `Error` for any
    /// other failure, so a request that fails after authorization still
    /// leaves its rows. A row that cannot be written after `work` succeeded
    /// is `unaudited`; after `work` failed, `work`'s error is returned.
    pub(crate) async fn audited<T>(
        &self,
        app: &App,
        audit: Audit,
        work: impl Future<Output = ApiResult<T>>,
    ) -> ApiResult<T> {
        self.audited_with(app, audit, async |_| work.await).await
    }

    /// [`Access::audited`] where `work` completes the audit as it learns
    /// what the row should say (a created id, counts); a failure is
    /// recorded as `work` left the audit.
    pub(crate) async fn audited_with<T>(
        &self,
        app: &App,
        mut audit: Audit,
        work: impl AsyncFnOnce(&mut Audit) -> ApiResult<T>,
    ) -> ApiResult<T> {
        // Boxed: the work and the audit write are each large, and both are
        // held across a suspension here.
        let result = Box::pin(work(&mut audit)).await;
        if audit.already_recorded {
            return result;
        }
        let outcome = match &result {
            Ok(_) if audit.ended_in_failure => Outcome::Error,
            Ok(_) => Outcome::Allowed,
            Err(_) if audit.refused => Outcome::Denied,
            Err(error) => error.outcome(),
        };
        let recorded = Box::pin(self.record(app, audit, outcome)).await;
        match result {
            Ok(value) => recorded.map(|_| value),
            Err(error) => {
                if let Err(unwritten) = recorded {
                    tracing::error!(
                        error = %unwritten.message,
                        "the audit row of a failed request could not be written"
                    );
                }
                Err(error)
            }
        }
    }

    /// Record a refusal that is the request's whole outcome: a check inside
    /// a handler (a write on a read endpoint) said no.
    pub(crate) async fn denied(&self, app: &App, audit: Audit) -> ApiResult<AuditId> {
        self.record(app, audit, Outcome::Denied).await
    }

    /// Record the answer to a write a turn waited on: the answer is the
    /// work, and the row says allowed or denied as the person decided.
    pub(crate) async fn answered(
        &self,
        app: &App,
        audit: Audit,
        answer: Decision,
    ) -> ApiResult<AuditId> {
        self.record(app, audit, answer.outcome()).await
    }

    async fn record(&self, app: &App, audit: Audit, outcome: Outcome) -> ApiResult<AuditId> {
        self.audit(
            app,
            audit.action,
            audit
                .resource
                .as_ref()
                .map(|(kind, id)| kind.clone().id(id)),
            outcome,
            audit.detail,
        )
        .await
    }
}
