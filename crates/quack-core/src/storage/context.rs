//! The workspace context: the instructions and definitions a workspace owner
//! writes for the agent (persona, column meanings, metric definitions,
//! caveats). Stored and versioned inside the workspace file; a global,
//! unclassified prefix may live at `~/.config/quack/context.md`.

use crate::error::Result;

use super::workspace::WorkspaceDb;
use crate::config;
use crate::error::Error;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ContextVersion {
    pub version: i64,
    pub content: String,
    pub edited_by: Option<String>,
    pub edited_at: String,
}

/// The current context, if any has been set.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn current(db: &WorkspaceDb) -> Result<Option<ContextVersion>> {
    Ok(history(db, 1)?.into_iter().next())
}

/// Versions, newest first.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn history(db: &WorkspaceDb, limit: u32) -> Result<Vec<ContextVersion>> {
    let mut stmt = db.connection().prepare(
        "SELECT version, content, edited_by, CAST(edited_at AS VARCHAR) \
         FROM _quack_context ORDER BY version DESC LIMIT ?",
    )?;
    let rows = stmt.query_map(duckdb::params![i64::from(limit)], |row| {
        Ok(ContextVersion {
            version: row.get(0)?,
            content: row.get(1)?,
            edited_by: row.get(2)?,
            edited_at: row.get(3)?,
        })
    })?;
    Ok(rows.collect::<duckdb::Result<_>>()?)
}

/// Record a new version. Content identical to the current version is not
/// recorded again; the current version is returned instead.
///
/// # Errors
///
/// Returns an error if the insert fails.
pub fn set(db: &WorkspaceDb, content: &str, edited_by: Option<&str>) -> Result<ContextVersion> {
    let normalized = content.trim_end().to_owned();
    if let Some(existing) = current(db)?
        && existing.content == normalized
    {
        return Ok(existing);
    }
    let next: i64 = db.connection().query_row(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM _quack_context",
        [],
        |row| row.get(0),
    )?;
    db.connection().execute(
        "INSERT INTO _quack_context (version, content, edited_by) VALUES (?, ?, ?)",
        duckdb::params![next, normalized, edited_by],
    )?;
    current(db)?.ok_or_else(|| Error::Analysis(String::from("context vanished after insert")))
}

/// Path of the global context prefix, next to `config.toml`.
#[must_use]
pub fn global_context_path() -> std::path::PathBuf {
    config::config_file_path().with_file_name("context.md")
}

/// The global prefix, if the file exists and is not blank.
///
/// # Errors
///
/// Returns an error if the file exists but cannot be read.
pub fn load_global() -> Result<Option<String>> {
    let path = global_context_path();
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)?;
    Ok((!text.trim().is_empty()).then(|| text.trim_end().to_owned()))
}

/// Global prefix and workspace context joined for the prompt, `None` when
/// both are absent.
///
/// # Errors
///
/// Returns an error if either source cannot be read.
pub fn combined(db: &WorkspaceDb) -> Result<Option<String>> {
    let global = load_global()?;
    let workspace = current(db)?.map(|c| c.content);
    Ok(match (global, workspace) {
        (None, None) => None,
        (Some(g), None) => Some(g),
        (None, Some(w)) => Some(w),
        (Some(g), Some(w)) => Some(format!("{g}\n\n{w}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> WorkspaceDb {
        WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| open_failed(&e.to_string()))
    }

    #[expect(clippy::panic, reason = "test helper: in-memory DuckDB must open")]
    fn open_failed(msg: &str) -> WorkspaceDb {
        panic!("in-memory DuckDB failed to open: {msg}");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn versions_increase_and_identical_content_is_not_repeated() {
        let db = db();
        assert!(current(&db).unwrap().is_none());
        let v1 = set(&db, "# Claims\n\nAmounts are in cents.\n\n", Some("justin")).unwrap();
        assert_eq!(v1.version, 1);
        assert_eq!(v1.content, "# Claims\n\nAmounts are in cents.");
        assert_eq!(v1.edited_by.as_deref(), Some("justin"));
        let again = set(&db, "# Claims\n\nAmounts are in cents.", None).unwrap();
        assert_eq!(again.version, 1);
        let v2 = set(&db, "# Claims\n\nAmounts are in dollars.", None).unwrap();
        assert_eq!(v2.version, 2);
        assert_eq!(current(&db).unwrap().unwrap().version, 2);
        let all = history(&db, 10).unwrap();
        assert_eq!(
            all.iter().map(|c| c.version).collect::<Vec<_>>(),
            vec![2, 1]
        );
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn combined_joins_global_and_workspace() {
        let db = db();
        // No global file in the test config dir; workspace only.
        set(&db, "workspace rules", None).unwrap();
        let text = combined(&db).unwrap();
        assert!(text.is_some_and(|t| t.ends_with("workspace rules")));
    }
}
