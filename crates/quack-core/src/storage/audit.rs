//! `_quack_audit`: the content half of the audit, inside the boundary.
//!
//! Every row shares its UUID v7 id with a `control.db.audit_log` row
//! (`storage::control::AuditEntry`). The access row says who touched which
//! resource; this row says what was done: the SQL, the file names, the
//! context version. Members of the workspace read it; admins do not.

use std::sync::Arc;

use crate::classify::{OnEnd, RunEnded};
use crate::error::Result;
use crate::ids::{AuditId, SessionId, UserId, WorkspaceId};
use crate::jobs::runs::RunRow;
use crate::storage::control::{
    AuditAction, AuditEntry, AuditResource, ControlPlane, Origin, Outcome, ResourceKind,
};
use crate::storage::workspace::WorkspaceDb;
use crate::storage::writer::{Lease, Writer};

/// The workspace's insert-only audit connection (design doc 4.1): a clone
/// of the writer's connection on a thread of its own that only ever
/// appends detail rows. `DuckDB` lets separate connections commit at once
/// unless they change the same rows, and a detail row is only ever a new
/// row, so a request records its audit detail without waiting for a write
/// in progress on the writer (a large load, a user's `CREATE TABLE AS`).
#[derive(Debug)]
pub struct AuditLog(Writer);

impl AuditLog {
    /// Clone `db` (the workspace's confined writer connection, before it
    /// moves to its writer) as the audit connection.
    ///
    /// # Errors
    ///
    /// When the clone or its thread cannot be made.
    pub fn open(db: &WorkspaceDb) -> Result<Self> {
        Ok(Self(Writer::spawn(db.try_clone_reader()?)?))
    }

    /// Append `detail` and await its commit.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub async fn record(&self, detail: AuditDetail) -> Result<()> {
        self.0.run(move |db| detail.write(db)).await
    }

    /// The audit connection, lent as [`Writer::lend`] lends the writer's.
    ///
    /// # Errors
    ///
    /// When the audit connection's thread has stopped.
    pub async fn lend(&self) -> Result<Lease> {
        self.0.lend().await
    }
}

/// What one access-audit row did, recorded under the same id.
#[derive(Debug, Clone)]
pub struct AuditDetail {
    /// The `control.db.audit_log` row's id.
    pub id: AuditId,
    pub user_id: Option<UserId>,
    pub action: String,
    /// The SQL, the file names, the context version: whatever the action
    /// touched.
    pub detail: serde_json::Value,
}

impl AuditDetail {
    /// Append this row.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub fn write(&self, db: &WorkspaceDb) -> Result<()> {
        db.connection().execute(
            "INSERT INTO _quack_audit (id, user_id, action, detail) VALUES (?, ?, ?, ?)",
            duckdb::params![
                self.id,
                self.user_id,
                self.action,
                serde_json::to_string(&self.detail)?
            ],
        )?;
        Ok(())
    }
}

/// A stored detail row.
#[derive(Debug, Clone, serde::Serialize, utoipa::ToSchema)]
pub struct AuditDetailRow {
    pub id: AuditId,
    pub timestamp: String,
    pub user_id: Option<UserId>,
    pub action: String,
    pub detail: Option<serde_json::Value>,
}

/// Detail rows, newest first.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn list(db: &WorkspaceDb, limit: u32) -> Result<Vec<AuditDetailRow>> {
    let mut stmt = db.connection().prepare(
        "SELECT id, CAST(timestamp AS VARCHAR), user_id, action, CAST(detail AS VARCHAR) \
         FROM _quack_audit ORDER BY id DESC LIMIT ?",
    )?;
    let rows = stmt.query_map(duckdb::params![i64::from(limit)], |row| {
        let detail: Option<String> = row.get(4)?;
        Ok(AuditDetailRow {
            id: row.get(0)?,
            timestamp: row.get(1)?,
            user_id: row.get(2)?,
            action: row.get(3)?,
            detail: detail.and_then(|d| serde_json::from_str(&d).ok()),
        })
    })?;
    Ok(rows.collect::<duckdb::Result<_>>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::embedding::Dimension;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[test]
    fn detail_rows_round_trip_newest_first() {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        let first = AuditId::generate();
        let second = AuditId::generate();
        assert!(
            AuditDetail {
                id: first,
                user_id: Some(UserId::from("u1")),
                action: String::from("sql"),
                detail: serde_json::json!({"sql": "SELECT 1"})
            }
            .write(&db)
            .is_ok()
        );
        assert!(
            AuditDetail {
                id: second.clone(),
                user_id: None,
                action: String::from("ingest"),
                detail: serde_json::json!({"filename": "a.csv"})
            }
            .write(&db)
            .is_ok()
        );
        let rows = list(&db, 10);
        assert!(rows.is_ok_and(|r| {
            r.len() == 2
                && r.first()
                    .is_some_and(|r| r.id == second && r.user_id.is_none())
                && r.last().is_some_and(|r| {
                    r.detail
                        .as_ref()
                        .and_then(|d| d.get("sql"))
                        .and_then(|s| s.as_str())
                        == Some("SELECT 1")
                })
        }));
        assert!(list(&db, 1).is_ok_and(|r| r.len() == 1));
    }

    /// The audit connection's premise (design doc 4.1): an insert-only
    /// connection cloned from the writer commits its detail rows while
    /// the writer is mid-write, whether that write is a transaction left
    /// open or a long statement running on another thread, and both
    /// survive a reopen.
    #[test]
    fn detail_rows_commit_while_the_writer_is_mid_write() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = Config::default();
        config.general.data_dir = dir.path().to_path_buf();
        let ok = |r: Result<()>| r.unwrap_or_else(|e| fail(&e.to_string()));
        let open = || WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        let clone = |db: &WorkspaceDb| {
            db.try_clone_reader()
                .unwrap_or_else(|e| fail(&e.to_string()))
        };
        let detail = serde_json::json!({ "what": "tables" });
        let mut ids = Vec::new();
        {
            let writer = open();
            let audit_conn = clone(&writer);
            let reader = clone(&writer);

            // A write transaction left open: DDL and a load, uncommitted.
            ok(writer.execute_statement("BEGIN TRANSACTION"));
            ok(writer
                .execute_statement("CREATE TABLE held AS SELECT range AS n FROM range(100000)"));
            let id = AuditId::generate();
            ok(AuditDetail {
                id: id.clone(),
                user_id: Some(UserId::from("u")),
                action: String::from("list"),
                detail: detail.clone(),
            }
            .write(&audit_conn));
            ids.push(id);
            // Committed and visible before the writer commits.
            assert!(list(&reader, 10).is_ok_and(|r| r.len() == 1));
            ok(writer.execute_statement("COMMIT"));

            // A long statement running on the writer's thread.
            let running = std::thread::spawn(move || {
                writer.execute_statement(
                    "CREATE TABLE loaded AS SELECT range AS n, md5(range::VARCHAR) AS h \
                     FROM range(1000000)",
                )
            });
            let before = ids.len();
            while !running.is_finished() && ids.len() < before + 20 {
                let id = AuditId::generate();
                ok(AuditDetail {
                    id: id.clone(),
                    user_id: Some(UserId::from("u")),
                    action: String::from("list"),
                    detail: detail.clone(),
                }
                .write(&audit_conn));
                ids.push(id);
            }
            let overlapped = ids.len().saturating_sub(before);
            assert!(
                overlapped >= 5,
                "only {overlapped} inserts overlapped the load"
            );
            ok(running.join().unwrap_or_else(|_| fail("the load panicked")));
        }
        let reopened = open();
        let rows = list(&reopened, 1000).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(rows.len(), ids.len());
        let count = |table: &str| {
            reopened
                .execute_query(&format!("SELECT count(*) FROM {table}"))
                .ok()
                .and_then(|r| r.rows.first().and_then(|row| row.first()).cloned())
        };
        assert_eq!(count("held"), Some(serde_json::json!(100_000)));
        assert_eq!(count("loaded"), Some(serde_json::json!(1_000_000)));
    }
}

/// Both halves of the audit for one caller in one workspace: the access
/// row in `control.db` and the detail row in `_quack_audit`, under one
/// id. The server builds one per request from the caller it resolved;
/// an MCP call over HTTP and a background run record through it, so
/// nothing audited proceeds unlogged.
#[derive(Clone)]
pub struct Recorder {
    control: ControlPlane,
    log: Arc<AuditLog>,
    origin: Origin,
    user_id: UserId,
    token_hash: Option<String>,
    workspace_id: WorkspaceId,
}

impl Recorder {
    #[must_use]
    pub fn new(
        control: ControlPlane,
        log: Arc<AuditLog>,
        origin: Origin,
        user_id: UserId,
        token_hash: Option<String>,
        workspace_id: WorkspaceId,
    ) -> Self {
        Self {
            control,
            log,
            origin,
            user_id,
            token_hash,
            workspace_id,
        }
    }

    /// The person every row is written for.
    #[must_use]
    pub fn user_id(&self) -> &UserId {
        &self.user_id
    }

    /// An `audit_log` entry for this caller in this workspace, for a change
    /// that commits it itself.
    #[must_use]
    pub fn entry(&self, action: AuditAction, outcome: Outcome) -> AuditEntry {
        let mut entry = AuditEntry::new(action, outcome, self.origin.clone());
        entry.user_id = Some(self.user_id.clone());
        entry.token_hash.clone_from(&self.token_hash);
        entry.in_workspace(&self.workspace_id)
    }

    /// Write both rows and give the id they share: the detail first, in the
    /// workspace file the work it records has just changed, then the access
    /// row in `control.db`. The two cannot share a transaction, so a failed
    /// access row leaves the work standing (#600): the detail still names
    /// who did it, and the access row is logged whole, content-free, for the
    /// operator to restore.
    ///
    /// # Errors
    ///
    /// Returns the audit connection's or the control plane's error.
    pub async fn record(
        &self,
        action: AuditAction,
        resource: Option<AuditResource<'_>>,
        outcome: Outcome,
        detail: Option<serde_json::Value>,
    ) -> Result<AuditId> {
        let mut entry = self.entry(action, outcome);
        if let Some(resource) = resource {
            entry = entry.on(resource);
        }
        self.record_detail(&entry, detail).await?;
        if let Err(e) = self.control.record_audit(&entry).await {
            let row = serde_json::to_string(&entry).unwrap_or_else(|_| entry.id.to_string());
            tracing::error!(
                error = %e,
                row,
                "the access audit row could not be written; the work it records stands"
            );
            return Err(e);
        }
        Ok(entry.id)
    }

    /// The `_quack_audit` half of an `audit_log` row already written.
    ///
    /// # Errors
    ///
    /// Returns the audit connection's error.
    pub async fn record_detail(
        &self,
        entry: &AuditEntry,
        detail: Option<serde_json::Value>,
    ) -> Result<()> {
        // Its own connection: a request never waits for a write in
        // progress on the writer just to record that it happened.
        self.log
            .record(AuditDetail {
                id: entry.id.clone(),
                user_id: Some(self.user_id.clone()),
                action: entry.action.to_string(),
                detail: detail.unwrap_or_else(|| serde_json::json!({})),
            })
            .await
    }

    /// The hook that audits a labelling run an agent turn or an MCP call
    /// started, when it ends and under the run's id, so a cancelled turn or
    /// a client that left still leaves its row. `session` is the turn's,
    /// when there is one.
    #[must_use]
    pub fn on_classify_end(&self, session: Option<SessionId>) -> OnEnd {
        let recorder = self.clone();
        OnEnd::new(move |ended: RunEnded| {
            let (recorder, session) = (recorder.clone(), session.clone());
            async move {
                let RunEnded { run, outcome } = ended;
                let detail = serde_json::json!({
                    "table": run.source_table,
                    "output_table": run.output_table,
                    "sentence": run.sentence,
                    "summary": run.to_string(),
                    "session_id": session,
                });
                let resource = ResourceKind::ClassificationRun.id(&run.id);
                if let Err(e) = recorder
                    .record(AuditAction::Classify, Some(resource), outcome, Some(detail))
                    .await
                {
                    tracing::error!(error = %e, "audit write failed");
                }
            }
        })
    }
}

impl Recorder {
    /// Write a background run's row (its start, or its end): both audit
    /// halves, as every row this recorder writes.
    ///
    /// # Errors
    ///
    /// Returns the write's error.
    pub async fn record_run(&self, row: RunRow) -> Result<()> {
        let resource = row.resource();
        let RunRow {
            action,
            outcome,
            detail,
            ..
        } = &row;
        Self::record(
            self,
            action.clone(),
            Some(resource),
            *outcome,
            Some(detail.clone()),
        )
        .await
        .map(drop)
    }
}
