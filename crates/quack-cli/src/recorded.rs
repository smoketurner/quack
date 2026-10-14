//! A command's run recorded as a job of the engine's queue, so the
//! workspace file (`_quack_jobs`) and `quack jobs` in another shell show
//! it the way they show the terminal's and the server's runs (issue
//! #570). The command runs where it always did, on its own task with its
//! output and its prompts on stdout and stderr; the job mirrors it:
//! queued when the command starts, its progress as the run reports it,
//! finished with the command's outcome, and waited for, so the engine's
//! recorder writes the row before the workspace closes.

use quack_core::ids::WorkspaceId;
use quack_core::jobs::{JobId, JobKind, JobQueue, JobSpec};
use quack_core::progress::ChunkDone;
use quack_core::text::OneLine;
use tokio::sync::{mpsc, oneshot};

/// Characters of a label at most.
const LABEL_CHARS: usize = 80;

/// One command's run, mirrored by a job on the engine's queue.
pub struct RecordedRun {
    jobs: JobQueue,
    id: JobId,
    progress: mpsc::UnboundedSender<ChunkDone>,
    done: Option<oneshot::Sender<Result<String, String>>>,
}

impl RecordedRun {
    /// Start the job that mirrors a run of `kind` in `workspace`, shown as
    /// `label`, running from now.
    #[must_use]
    pub fn start(
        jobs: &JobQueue,
        kind: JobKind,
        label: impl Into<String>,
        workspace: WorkspaceId,
    ) -> Self {
        let (progress, mut reported) = mpsc::unbounded_channel::<ChunkDone>();
        let (done, mut ended) = oneshot::channel();
        let spec = JobSpec::new(kind, label).workspace(workspace);
        let job = jobs.submit(spec, move |ctx| async move {
            loop {
                tokio::select! {
                    Some(done) = reported.recv() => ctx.progress(done.done, done.total),
                    outcome = &mut ended => {
                        return outcome.unwrap_or_else(|_| {
                            Err(String::from("the command ended without reporting its outcome"))
                        });
                    }
                }
            }
        });
        Self {
            jobs: jobs.clone(),
            id: job.id,
            progress,
            done: Some(done),
        }
    }

    /// The command line, as a job label: every argument after the binary,
    /// on one line, at most `LABEL_CHARS` (80) characters.
    #[must_use]
    pub fn command_line() -> String {
        let line = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
        OneLine(&line)
            .to_string()
            .chars()
            .take(LABEL_CHARS)
            .collect()
    }

    /// One unit of the run finished: the job's progress follows.
    pub fn report(&self, done: ChunkDone) {
        if self.progress.send(done).is_err() {
            // The job has already ended; nothing is listening.
        }
    }

    /// End the job with the command's outcome and wait until the queue has
    /// recorded it.
    pub async fn finish(mut self, outcome: Result<String, String>) {
        if let Some(done) = self.done.take() {
            drop(done.send(outcome));
        }
        self.jobs.wait(self.id).await;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use quack_core::ids::WorkspaceId;
    use quack_core::jobs::{JobKind, JobProgress, JobQueue, JobState};
    use quack_core::progress::ChunkDone;

    use super::RecordedRun;

    /// The job runs while the command does, carries its progress, and ends
    /// with its outcome.
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn the_job_mirrors_the_run() {
        let jobs = JobQueue::new(10);
        let run = RecordedRun::start(&jobs, JobKind::Ingest, "notes.md", WorkspaceId::from("ws"));
        run.report(ChunkDone {
            done: 2,
            total: 5,
            failed: 0,
            took: Duration::from_millis(1),
            elapsed: Duration::from_millis(2),
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        let listed = jobs.list();
        assert_eq!(listed.len(), 1, "{listed:?}");
        assert_eq!(listed.first().map(|j| j.state), Some(JobState::Running));
        assert_eq!(
            listed.first().and_then(|j| j.progress),
            Some(JobProgress { done: 2, total: 5 })
        );
        run.finish(Ok(String::from("3 chunks"))).await;
        let ended = jobs.list();
        assert_eq!(ended.first().map(|j| j.state), Some(JobState::Succeeded));
        assert_eq!(
            ended.first().and_then(|j| j.outcome.as_deref()),
            Some("3 chunks")
        );

        let failed = RecordedRun::start(&jobs, JobKind::Import, "orders", WorkspaceId::from("ws"));
        failed.finish(Err(String::from("404"))).await;
        let listed = jobs.list();
        let last = listed.last().unwrap();
        assert_eq!(last.state, JobState::Failed);
        assert_eq!(last.outcome.as_deref(), Some("404"));
    }
}
