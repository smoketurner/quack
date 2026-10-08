//! `[server.webhooks]`: a POST to the operator's endpoint when a background
//! job finishes, so a pipeline can react without polling (issue #420).
//!
//! The body carries no workspace content: ids, kind, state, times, and
//! progress, never the job's label or outcome text. A receiver that wants
//! those asks `GET .../jobs/{job}` with its own token, which keeps
//! authorization and the audit with the existing route. Each delivery is
//! signed with HMAC-SHA256 under the secret in `secret_env` over
//! `<timestamp>.<body>`, as `X-Quack-Signature: sha256=<hex>`, with the
//! Unix timestamp in `X-Quack-Timestamp`: a receiver that refuses an old
//! timestamp refuses a replayed delivery.

use std::sync::Arc;
use std::time::Duration;

use aws_lc_rs::hmac;
use serde::Serialize;
use tokio::sync::{Semaphore, broadcast};
use tokio_util::task::TaskTracker;

use super::{JobInfo, JobKind, JobProgress, JobState};
use crate::config::WebhookConfig;
use crate::crypto::hex_lower;
use crate::error::{Error, Result};
use crate::ids::{UserId, WorkspaceId};
use crate::jobs::{JobId, JobNumber};
use crate::proxy::Proxies;

/// The header that carries the signature.
pub const SIGNATURE_HEADER: &str = "x-quack-signature";

/// The header that carries the signed Unix timestamp.
pub const TIMESTAMP_HEADER: &str = "x-quack-timestamp";

/// Deliveries in flight at once; the rest wait their turn on their tasks.
const IN_FLIGHT: usize = 4;

/// How long a failed delivery waits before its one retry.
const RETRY_AFTER: Duration = Duration::from_secs(2);

/// Where finished jobs are reported, and how.
pub struct Webhook {
    url: reqwest::Url,
    key: hmac::Key,
    kinds: Vec<JobKind>,
    timeout: Duration,
}

/// What a webhook says about a finished job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JobFinished {
    pub job_id: JobId,
    pub number: JobNumber,
    pub kind: JobKind,
    pub state: JobState,
    pub workspace_id: Option<WorkspaceId>,
    pub owner: Option<UserId>,
    pub queued_at: jiff::Timestamp,
    pub started_at: Option<jiff::Timestamp>,
    pub finished_at: Option<jiff::Timestamp>,
    pub progress: Option<JobProgress>,
}

impl From<&JobInfo> for JobFinished {
    fn from(job: &JobInfo) -> Self {
        Self {
            job_id: job.id,
            number: job.number,
            kind: job.kind,
            state: job.state,
            workspace_id: job.workspace_id.clone(),
            owner: job.owner.clone(),
            queued_at: job.queued_at,
            started_at: job.started_at,
            finished_at: job.finished_at,
            progress: job.progress,
        }
    }
}

impl Webhook {
    /// The webhook `[server.webhooks]` names; `None` when it names none.
    ///
    /// # Errors
    ///
    /// Returns a configuration error for a URL that does not parse or a
    /// `secret_env` that is not set: the server does not send unsigned.
    pub fn from_config(config: Option<&WebhookConfig>) -> Result<Option<Self>> {
        let Some(config) = config else {
            return Ok(None);
        };
        Self::with_secret(config, std::env::var(&config.secret_env).ok()).map(Some)
    }

    fn with_secret(config: &WebhookConfig, secret: Option<String>) -> Result<Self> {
        let url = reqwest::Url::parse(&config.url)
            .map_err(|e| Error::Config(format!("[server.webhooks].url: {e}")))?;
        let secret = secret.filter(|s| !s.trim().is_empty()).ok_or_else(|| {
            Error::Config(format!(
                "[server.webhooks].secret_env names {}, which is not set; webhooks are always signed",
                config.secret_env
            ))
        })?;
        Ok(Self {
            url,
            key: hmac::Key::new(hmac::HMAC_SHA256, secret.as_bytes()),
            kinds: config.kinds.clone(),
            timeout: Duration::from_secs(u64::from(config.timeout_seconds.max(1))),
        })
    }

    /// Whether `job` is one this webhook reports: finished, and of a kind it
    /// names (every kind but chat turns when it names none).
    fn reports(&self, job: &JobInfo) -> bool {
        job.state.is_finished()
            && if self.kinds.is_empty() {
                job.kind != JobKind::Chat
            } else {
                self.kinds.contains(&job.kind)
            }
    }

    /// `sha256=<hex>` of `<timestamp>.<body>` under the secret.
    fn signature(&self, timestamp: i64, body: &[u8]) -> String {
        let mut signed = format!("{timestamp}.").into_bytes();
        signed.extend_from_slice(body);
        format!(
            "sha256={}",
            hex_lower(hmac::sign(&self.key, &signed).as_ref())
        )
    }

    /// Report every finished job `jobs` broadcasts, each on a task of its
    /// own so a slow endpoint never holds up reading the broadcast, until
    /// the queue's channel closes: jobs that end while the server stops
    /// are reported too. The task ends once its deliveries have.
    #[must_use]
    pub fn spawn(self, mut jobs: broadcast::Receiver<JobInfo>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let client = match Proxies::from_env().client().timeout(self.timeout).build() {
                Ok(client) => client,
                Err(e) => {
                    tracing::error!(error = %e, "the webhook client could not be built; webhooks are off");
                    return;
                }
            };
            let hook = Arc::new(self);
            let slots = Arc::new(Semaphore::new(IN_FLIGHT));
            let deliveries = TaskTracker::new();
            loop {
                match jobs.recv().await {
                    Ok(job) if hook.reports(&job) => {
                        let (hook, client, slots) =
                            (Arc::clone(&hook), client.clone(), Arc::clone(&slots));
                        deliveries.spawn(async move {
                            let Ok(_slot) = slots.acquire_owned().await else {
                                return;
                            };
                            hook.deliver(&client, &job).await;
                        });
                    }
                    Ok(_) => {}
                    Err(broadcast::error::RecvError::Lagged(missed)) => {
                        tracing::warn!(
                            missed,
                            "the webhook fell behind; some finished jobs were not reported"
                        );
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        deliveries.close();
                        deliveries.wait().await;
                        return;
                    }
                }
            }
        })
    }

    /// POST the job, and once more after a short wait if that fails; each
    /// attempt is signed with its own timestamp.
    async fn deliver(&self, client: &reqwest::Client, job: &JobInfo) {
        let body = match serde_json::to_vec(&JobFinished::from(job)) {
            Ok(body) => body,
            Err(e) => {
                tracing::warn!(error = %e, "a finished job could not be written for the webhook");
                return;
            }
        };
        for attempt in 1..=2_u8 {
            let timestamp = jiff::Timestamp::now().as_second();
            let sent = client
                .post(self.url.clone())
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .header(TIMESTAMP_HEADER, timestamp.to_string())
                .header(SIGNATURE_HEADER, self.signature(timestamp, &body))
                .body(body.clone())
                .send()
                .await;
            match sent {
                Ok(response) if response.status().is_success() => return,
                Ok(response) => {
                    tracing::warn!(job = %job.id, attempt, status = %response.status(), "the webhook endpoint refused a finished job");
                }
                Err(e) => {
                    tracing::warn!(job = %job.id, attempt, error = %e, "the webhook endpoint could not be reached");
                }
            }
            if attempt == 1 {
                tokio::time::sleep(RETRY_AFTER).await;
            }
        }
    }
}

#[cfg(test)]
mod tests;
