//! `_quack_audit`: the content half of the audit, inside the boundary.
//!
//! Every row shares its UUID v7 id with a `control.db.audit_log` row
//! (`storage::control::AuditEntry`). The access row says who touched which
//! resource; this row says what was done: the SQL, the file names, the
//! context version. Members of the workspace read it; admins do not.

use crate::error::Result;
use crate::storage::workspace::WorkspaceDb;

/// A stored detail row.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AuditDetailRow {
    pub id: String,
    pub timestamp: String,
    pub user_id: Option<String>,
    pub action: String,
    pub detail: Option<serde_json::Value>,
}

/// Append the detail for an access-audit row of the same id.
///
/// # Errors
///
/// Returns an error if the insert fails.
pub fn record(
    db: &WorkspaceDb,
    id: &str,
    user_id: Option<&str>,
    action: &str,
    detail: &serde_json::Value,
) -> Result<()> {
    db.connection().execute(
        "INSERT INTO _quack_audit (id, user_id, action, detail) VALUES (?, ?, ?, ?)",
        duckdb::params![id, user_id, action, serde_json::to_string(detail)?],
    )?;
    Ok(())
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
    let mut rows = stmt.query(duckdb::params![i64::from(limit)])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let detail: Option<String> = row.get(4)?;
        out.push(AuditDetailRow {
            id: row.get(0)?,
            timestamp: row.get(1)?,
            user_id: row.get(2)?,
            action: row.get(3)?,
            detail: detail.and_then(|d| serde_json::from_str(&d).ok()),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[test]
    fn detail_rows_round_trip_newest_first() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        let first = uuid::Uuid::now_v7().to_string();
        let second = uuid::Uuid::now_v7().to_string();
        assert!(
            record(
                &db,
                &first,
                Some("u1"),
                "sql",
                &serde_json::json!({"sql": "SELECT 1"})
            )
            .is_ok()
        );
        assert!(
            record(
                &db,
                &second,
                None,
                "ingest",
                &serde_json::json!({"filename": "a.csv"})
            )
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
}
