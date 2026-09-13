use std::io::Write;

use crate::config::Config;

/// Query result set from a `DuckDB` workspace database.
pub struct QueryResults {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
}

/// Wraps a `DuckDB` connection for a single workspace.
pub struct WorkspaceDb {
    conn: duckdb::Connection,
}

impl WorkspaceDb {
    /// Open (or create) the `DuckDB` database for a workspace.
    ///
    /// # Errors
    ///
    /// Returns an error if the database file cannot be created or opened.
    pub fn open(config: &Config, workspace_id: &str) -> crate::error::Result<Self> {
        let db_path = config.workspace_db_path(workspace_id);
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let files_dir = config.workspace_files_dir(workspace_id);
        std::fs::create_dir_all(&files_dir)?;

        let conn = duckdb::Connection::open(&db_path)?;
        Ok(Self { conn })
    }

    /// Execute an arbitrary SQL statement and return the results.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid or execution fails.
    pub fn execute_query(&self, sql: &str) -> crate::error::Result<QueryResults> {
        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query([])?;

        let (columns, column_count) = {
            let Some(stmt_ref) = rows.as_ref() else {
                return Ok(QueryResults {
                    columns: Vec::new(),
                    rows: Vec::new(),
                });
            };
            let count = stmt_ref.column_count();
            if count == 0 {
                return Ok(QueryResults {
                    columns: Vec::new(),
                    rows: Vec::new(),
                });
            }
            (stmt_ref.column_names(), count)
        };

        let mut result_rows: Vec<Vec<serde_json::Value>> = Vec::new();
        while let Some(row) = rows.next()? {
            let mut values = Vec::with_capacity(column_count);
            for i in 0..column_count {
                values.push(extract_value(row, i));
            }
            result_rows.push(values);
        }

        Ok(QueryResults {
            columns,
            rows: result_rows,
        })
    }

    /// Access the underlying `DuckDB` connection.
    #[must_use]
    pub fn connection(&self) -> &duckdb::Connection {
        &self.conn
    }
}

fn extract_value(row: &duckdb::Row<'_>, idx: usize) -> serde_json::Value {
    if let Ok(v) = row.get::<_, Option<i64>>(idx) {
        return match v {
            Some(n) => serde_json::Value::Number(n.into()),
            None => serde_json::Value::Null,
        };
    }
    if let Ok(v) = row.get::<_, Option<f64>>(idx) {
        return match v {
            Some(n) => serde_json::Number::from_f64(n)
                .map_or(serde_json::Value::Null, serde_json::Value::Number),
            None => serde_json::Value::Null,
        };
    }
    if let Ok(v) = row.get::<_, Option<bool>>(idx) {
        return match v {
            Some(b) => serde_json::Value::Bool(b),
            None => serde_json::Value::Null,
        };
    }
    if let Ok(v) = row.get::<_, Option<String>>(idx) {
        return match v {
            Some(s) => serde_json::Value::String(s),
            None => serde_json::Value::Null,
        };
    }
    serde_json::Value::Null
}

fn display_json_value(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::Null => String::from("NULL"),
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

impl QueryResults {
    /// Write results as a human-readable aligned table.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to `out` fails.
    pub fn write_table(&self, out: &mut impl Write) -> crate::error::Result<()> {
        if self.columns.is_empty() {
            writeln!(out, "OK")?;
            return Ok(());
        }

        let display_rows: Vec<Vec<String>> = self
            .rows
            .iter()
            .map(|row| row.iter().map(display_json_value).collect())
            .collect();

        let mut widths: Vec<usize> = self.columns.iter().map(String::len).collect();
        for row in &display_rows {
            for (w, val) in widths.iter_mut().zip(row.iter()) {
                *w = (*w).max(val.len());
            }
        }

        for (i, (col, width)) in self.columns.iter().zip(widths.iter()).enumerate() {
            if i > 0 {
                write!(out, " | ")?;
            }
            write!(out, "{col:<width$}")?;
        }
        writeln!(out)?;

        for (i, width) in widths.iter().enumerate() {
            if i > 0 {
                write!(out, "-+-")?;
            }
            for _ in 0..*width {
                write!(out, "-")?;
            }
        }
        writeln!(out)?;

        for row in &display_rows {
            for (i, (val, width)) in row.iter().zip(widths.iter()).enumerate() {
                if i > 0 {
                    write!(out, " | ")?;
                }
                write!(out, "{val:<width$}")?;
            }
            writeln!(out)?;
        }

        let row_count = self.rows.len();
        writeln!(out, "({row_count} rows)")?;
        Ok(())
    }

    /// Write results as a JSON array of objects.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to `out` or JSON serialization fails.
    pub fn write_json(&self, out: &mut impl Write) -> crate::error::Result<()> {
        let json_rows: Vec<serde_json::Map<String, serde_json::Value>> = self
            .rows
            .iter()
            .map(|row| {
                self.columns
                    .iter()
                    .zip(row.iter())
                    .map(|(col, val)| (col.clone(), val.clone()))
                    .collect()
            })
            .collect();

        serde_json::to_writer_pretty(&mut *out, &json_rows)?;
        writeln!(out)?;
        Ok(())
    }
}
