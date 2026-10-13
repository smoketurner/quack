//! The job queue's broadcast forwarded to the frontend as the Tauri event
//! `job` (design 3.4), for every workspace the app has open, plus the
//! list and the cancel.

use quack_core::ids::WorkspaceId;
use quack_core::jobs::{JobId, JobInfo, JobQueue};
use tauri::{AppHandle, Emitter, State};
use tokio::sync::broadcast::error::RecvError;

use crate::app::AppState;
use crate::error::{CommandError, CommandResult};

/// The event every job snapshot goes out on.
pub(crate) const EVENT: &str = "job";

/// Forward every `JobInfo` the queue broadcasts, for the life of the app.
/// A lagged subscriber resynchronizes from the list, as the design says.
pub(crate) fn forward(app: AppHandle, jobs: JobQueue) {
    tauri::async_runtime::spawn(async move {
        let mut events = jobs.subscribe();
        loop {
            match events.recv().await {
                Ok(info) => emit(&app, &info),
                Err(RecvError::Lagged(skipped)) => {
                    tracing::warn!(skipped, "job events lagged; resending the list");
                    for info in jobs.list() {
                        emit(&app, &info);
                    }
                }
                Err(RecvError::Closed) => break,
            }
        }
    });
}

fn emit(app: &AppHandle, info: &JobInfo) {
    if let Err(e) = app.emit(EVENT, info) {
        tracing::debug!(error = %e, "a job event could not reach the frontend");
    }
}

/// Every job the queue knows, newest first; `workspace` narrows it.
#[tauri::command]
pub(crate) fn list_jobs(
    state: State<'_, AppState>,
    workspace: Option<WorkspaceId>,
) -> Vec<JobInfo> {
    match workspace {
        Some(id) => state.jobs.list_workspace(&id),
        None => state.jobs.list(),
    }
}

/// Cancel `job`. `false` when it had already ended.
#[tauri::command]
pub(crate) fn cancel_job(state: State<'_, AppState>, job: String) -> CommandResult<bool> {
    let id: JobId = job
        .parse()
        .map_err(|e| CommandError::invalid(format!("not a job id: {e}")))?;
    Ok(state.jobs.cancel(id))
}
