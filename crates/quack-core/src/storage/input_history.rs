//! Lines typed at the terminal prompt, so Up recalls them in a later
//! session. They repeat the workspace's questions and SQL, so they live in
//! the workspace file and go with it when it is deleted.

use super::workspace::WorkspaceDb;
use crate::error::Result;
use crate::ids::InputLineId;

pub const DDL: &str = "CREATE TABLE IF NOT EXISTS _quack_input_history (
    id TEXT PRIMARY KEY,
    line TEXT NOT NULL,
    typed_at TIMESTAMP DEFAULT now()
);";

/// How many lines a workspace keeps.
pub const KEPT: u32 = 500;

/// The newest [`KEPT`] lines, oldest first.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn recent(db: &WorkspaceDb) -> Result<Vec<String>> {
    let mut stmt = db.connection().prepare(
        "SELECT line FROM (SELECT id, line FROM _quack_input_history ORDER BY id DESC LIMIT ?) \
         ORDER BY id",
    )?;
    let rows = stmt.query_map(duckdb::params![KEPT], |row| row.get(0))?;
    Ok(rows.collect::<duckdb::Result<_>>()?)
}

/// Keep `line` and drop what falls past the newest [`KEPT`].
///
/// # Errors
///
/// Returns an error if a statement fails.
pub fn push(db: &WorkspaceDb, line: &str) -> Result<()> {
    let conn = db.connection();
    conn.execute(
        "INSERT INTO _quack_input_history (id, line) VALUES (?, ?)",
        duckdb::params![InputLineId::generate(), line],
    )?;
    conn.execute(
        "DELETE FROM _quack_input_history WHERE id NOT IN \
         (SELECT id FROM _quack_input_history ORDER BY id DESC LIMIT ?)",
        duckdb::params![KEPT],
    )?;
    Ok(())
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
    fn keeps_the_newest_lines_oldest_first_newlines_included() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = Config::default();
        config.general.data_dir = dir.path().to_path_buf();
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            recent(&db)
                .unwrap_or_else(|e| fail(&e.to_string()))
                .is_empty()
        );

        let lines: Vec<String> = (0..=KEPT).map(|n| format!("SELECT {n}\n")).collect();
        for line in &lines {
            push(&db, line).unwrap_or_else(|e| fail(&e.to_string()));
        }
        let kept = recent(&db).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(kept, lines.get(1..).unwrap_or_default());
        let rows: u32 = db
            .connection()
            .query_row("SELECT count(*) FROM _quack_input_history", [], |row| {
                row.get(0)
            })
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(rows, KEPT, "the oldest is deleted, not only hidden");
    }
}
