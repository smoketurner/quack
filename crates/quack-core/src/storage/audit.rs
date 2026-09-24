//! `_quack_audit`: the content half of the audit, inside the boundary.
//!
//! Every row shares its UUID v7 id with a `control.db.audit_log` row
//! (`storage::control::AuditEntry`). The access row says who touched which
//! resource; this row says what was done: the SQL, the file names, the
//! context version. Members of the workspace read it; admins do not.

use crate::error::Result;
use crate::ids::{AuditId, UserId};
use crate::storage::workspace::WorkspaceDb;
use crate::storage::writer::Writer;

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
#[derive(Debug, Clone, serde::Serialize)]
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

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[test]
    fn detail_rows_round_trip_newest_first() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        let first = AuditId::generate();
        let second = AuditId::generate();
        assert!(
            AuditDetail {
                id: first.clone(),
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
