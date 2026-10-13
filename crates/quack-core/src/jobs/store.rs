//! The jobs of a workspace, kept in its file (`_quack_jobs`) so a restart
//! shows what ran and how it ended. A job's label can quote a question or
//! name a file, which is workspace content, so the rows live in the
//! workspace's `DuckDB` file and never in `control.db` (design doc 4.1).
//!
//! The engine writes a row for every snapshot the queue broadcasts about a
//! job of an open workspace ([`upsert`]), and reads the newest rows back
//! when it opens the file ([`recent`]), after marking the jobs an earlier
//! process left `queued` or `running` as failed ([`fail_interrupted`]).

use jiff::Timestamp;
use jiff::civil::DateTime;
use jiff::tz::TimeZone;

use super::{JobId, JobInfo, JobKind, JobNumber, JobProgress, JobState};
use crate::error::{Error, Result};
use crate::ids::{UserId, WorkspaceId};
use crate::storage::workspace::WorkspaceDb;

/// The table, created at open.
pub const DDL: &str = "CREATE TABLE IF NOT EXISTS _quack_jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    owner TEXT,
    lane TEXT,
    state TEXT NOT NULL,
    progress_done INTEGER,
    progress_total INTEGER,
    status TEXT,
    outcome TEXT,
    queued_at TIMESTAMP NOT NULL,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    cancel_requested BOOLEAN NOT NULL,
    holder TEXT NOT NULL
);";

/// What the closing row says of a job a process left behind.
pub const INTERRUPTED: &str = "the process running it ended before it finished";

/// `DuckDB`'s naive timestamp text for `ts`, in UTC.
fn timestamp_param(ts: Timestamp) -> String {
    ts.to_zoned(TimeZone::UTC)
        .datetime()
        .strftime("%Y-%m-%d %H:%M:%S%.6f")
        .to_string()
}

/// The timestamp `DuckDB` printed as `YYYY-MM-DD HH:MM:SS[.ffffff]`.
fn timestamp_of(text: &str) -> Result<Timestamp> {
    let civil: DateTime = text
        .parse()
        .map_err(|e: jiff::Error| Error::Analysis(format!("malformed job time {text:?}: {e}")))?;
    civil
        .to_zoned(TimeZone::UTC)
        .map(|zoned| zoned.timestamp())
        .map_err(|e| Error::Analysis(format!("malformed job time {text:?}: {e}")))
}

/// Write `job` as it stands (insert, or update every field of its row);
/// `holder` names the process running it.
///
/// # Errors
///
/// Returns the statement's error.
pub fn upsert(db: &WorkspaceDb, job: &JobInfo, holder: &str) -> Result<()> {
    let (done, total) = match job.progress {
        Some(JobProgress { done, total }) => (Some(done), Some(total)),
        None => (None, None),
    };
    db.connection().execute(
        "INSERT OR REPLACE INTO _quack_jobs (id, kind, label, owner, lane, state, \
         progress_done, progress_total, status, outcome, queued_at, started_at, finished_at, \
         cancel_requested, holder) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        duckdb::params![
            job.id.to_string(),
            job.kind.as_str(),
            job.label,
            job.owner,
            job.lane,
            job.state.as_str(),
            done,
            total,
            job.status,
            job.outcome,
            timestamp_param(job.queued_at),
            job.started_at.map(timestamp_param),
            job.finished_at.map(timestamp_param),
            job.cancel_requested,
            holder,
        ],
    )?;
    Ok(())
}

/// Mark every job still `queued` or `running` as failed: the process that
/// ran it is gone (this one is opening the file). Returns how many.
///
/// # Errors
///
/// Returns the statement's error.
pub fn fail_interrupted(db: &WorkspaceDb) -> Result<usize> {
    let changed = db.connection().execute(
        "UPDATE _quack_jobs SET state = ?, outcome = ?, finished_at = ?, cancel_requested = true \
         WHERE state IN (?, ?)",
        duckdb::params![
            JobState::Failed.as_str(),
            INTERRUPTED,
            timestamp_param(Timestamp::now()),
            JobState::Queued.as_str(),
            JobState::Running.as_str(),
        ],
    )?;
    Ok(changed)
}

/// The newest `limit` jobs, oldest first, each for `workspace`; a job's
/// number is the queue's to give when it restores them.
///
/// # Errors
///
/// Returns the query's error, or a row that does not read back.
pub fn recent(db: &WorkspaceDb, workspace: &WorkspaceId, limit: u32) -> Result<Vec<JobInfo>> {
    let mut statement = db.connection().prepare(
        "SELECT id, kind, label, owner, lane, state, progress_done, progress_total, status, \
         outcome, CAST(queued_at AS TEXT), CAST(started_at AS TEXT), CAST(finished_at AS TEXT), \
         cancel_requested FROM _quack_jobs ORDER BY queued_at DESC, id DESC LIMIT ?",
    )?;
    let rows = statement.query_map(duckdb::params![limit], |row| {
        Ok(Row {
            id: row.get(0)?,
            kind: row.get(1)?,
            label: row.get(2)?,
            owner: row.get(3)?,
            lane: row.get(4)?,
            state: row.get(5)?,
            progress_done: row.get(6)?,
            progress_total: row.get(7)?,
            status: row.get(8)?,
            outcome: row.get(9)?,
            queued_at: row.get(10)?,
            started_at: row.get(11)?,
            finished_at: row.get(12)?,
            cancel_requested: row.get(13)?,
        })
    })?;
    let mut jobs = Vec::new();
    for row in rows {
        jobs.push(row?.into_info(workspace)?);
    }
    jobs.reverse();
    Ok(jobs)
}

/// One row as `DuckDB` returns it.
struct Row {
    id: String,
    kind: String,
    label: String,
    owner: Option<UserId>,
    lane: Option<String>,
    state: String,
    progress_done: Option<u32>,
    progress_total: Option<u32>,
    status: Option<String>,
    outcome: Option<String>,
    queued_at: String,
    started_at: Option<String>,
    finished_at: Option<String>,
    cancel_requested: bool,
}

impl Row {
    fn into_info(self, workspace: &WorkspaceId) -> Result<JobInfo> {
        let malformed = |what: &str, value: &str| {
            Error::Analysis(format!("malformed job {what} {value:?} in _quack_jobs"))
        };
        let id: JobId = self.id.parse().map_err(|_| malformed("id", &self.id))?;
        let kind = JobKind::ALL
            .iter()
            .copied()
            .find(|kind| kind.as_str() == self.kind)
            .ok_or_else(|| malformed("kind", &self.kind))?;
        let state = JobState::ALL
            .iter()
            .copied()
            .find(|state| state.as_str() == self.state)
            .ok_or_else(|| malformed("state", &self.state))?;
        let progress = match (self.progress_done, self.progress_total) {
            (Some(done), Some(total)) => Some(JobProgress { done, total }),
            (None, _) | (_, None) => None,
        };
        Ok(JobInfo {
            id,
            number: JobNumber::default(),
            kind,
            label: self.label,
            workspace_id: Some(workspace.clone()),
            owner: self.owner,
            lane: self.lane,
            state,
            progress,
            status: self.status,
            outcome: self.outcome,
            queued_at: timestamp_of(&self.queued_at)?,
            started_at: self.started_at.as_deref().map(timestamp_of).transpose()?,
            finished_at: self.finished_at.as_deref().map(timestamp_of).transpose()?,
            cancel_requested: self.cancel_requested,
        })
    }
}
