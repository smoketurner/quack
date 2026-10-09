//! `quack tables`: the workspace's tables with their row counts, notes,
//! and profile warnings; one table in full; a note set or cleared; a
//! column retyped.

use std::io::Write;

use anyhow::{Context, Result};
use quack_core::analysis::table_search;
use quack_core::storage::profile::{ColumnTypes, TableNote};
use quack_core::storage::workspace::WorkspaceDb;

use crate::text_or_json::TextOrJson;

#[derive(clap::Args)]
pub struct TablesArgs {
    /// One table to show in full; without it, every table in a line
    table: Option<String>,

    /// Set the table's note, which every description of the table and the
    /// agent's prompt carry (an empty note removes it)
    #[arg(long, requires = "table", value_name = "TEXT")]
    note: Option<String>,

    /// Give a column of the table a type, as COLUMN=TYPE (VARCHAR, BIGINT,
    /// DOUBLE, DATE, TIMESTAMP, BOOLEAN); every value must convert
    #[arg(long, requires = "table", value_name = "COLUMN=TYPE")]
    retype: Vec<ColumnTypes>,

    /// `json` prints one JSON object per table
    #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
    format: TextOrJson,
}

impl TablesArgs {
    /// Whether it sets a note or retypes a column.
    #[must_use]
    pub fn writes(&self) -> bool {
        self.note.is_some() || !self.retype.is_empty()
    }

    /// Show a table, or set its note or retype columns first, and print it on `out`.
    ///
    /// # Errors
    ///
    /// Returns the database, model, or I/O error the command meets.
    pub fn run(self, db: &WorkspaceDb, out: &mut impl Write) -> Result<()> {
        let Some(table) = self.table else {
            return Self::list(db, self.format, out);
        };
        if !db.list_tables()?.contains(&table) {
            anyhow::bail!("no table named {table}");
        }
        if let Some(note) = &self.note {
            TableNote::set(db, &table, note, None)?;
        }
        let retype = ColumnTypes::joined(self.retype);
        if !retype.is_empty() {
            retype.apply(db, &table)?;
        }
        let described = db
            .describe_table(&table)
            .with_context(|| format!("could not describe {table}"))?;
        match self.format {
            TextOrJson::Json => {
                writeln!(out, "{}", serde_json::to_string_pretty(&described.body())?)?;
            }
            TextOrJson::Text => {
                write!(out, "{described}")?;
                let fixes: Vec<String> = described
                    .warnings
                    .iter()
                    .filter_map(|flagged| {
                        let fix = flagged.warning.fix()?;
                        Some(format!("{}={fix}", flagged.column))
                    })
                    .collect();
                if !fixes.is_empty() {
                    writeln!(
                        out,
                        "\nFix with: quack tables {table} --retype {}",
                        fixes.join(",")
                    )?;
                }
            }
        }
        Ok(())
    }

    fn list(db: &WorkspaceDb, format: TextOrJson, out: &mut impl Write) -> Result<()> {
        let mut rows = Vec::new();
        for table in table_search::user_tables(db)? {
            let described = db.describe_table(&table)?;
            rows.push(serde_json::json!({
                "table": described.table_name,
                "row_count": described.row_count,
                "note": described.note,
                "warnings": described.warnings.len(),
            }));
        }
        format.write_rows(out, &rows, "No tables.", |out, row| {
            let warnings = row
                .get("warnings")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            write!(
                out,
                "{} ({} rows",
                row.get("table")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or(""),
                row.get("row_count")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(0)
            )?;
            if warnings > 0 {
                write!(out, ", {warnings} warnings")?;
            }
            write!(out, ")")?;
            if let Some(note) = row.get("note").and_then(serde_json::Value::as_str) {
                write!(out, ": {}", note.lines().next().unwrap_or(""))?;
            }
            writeln!(out)
        })
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "test setup")]

    use super::*;
    use quack_core::embedding::Dimension;
    use quack_core::storage::profile::TableProfile;

    fn run(db: &WorkspaceDb, args: &[&str]) -> Result<String> {
        use clap::Parser;
        #[derive(clap::Parser)]
        struct Wrapper {
            #[command(flatten)]
            args: TablesArgs,
        }
        let parsed =
            Wrapper::try_parse_from(std::iter::once("tables").chain(args.iter().copied()))?;
        let mut out = Vec::new();
        parsed.args.run(db, &mut out)?;
        Ok(String::from_utf8(out)?)
    }

    #[test]
    fn tables_lists_describes_notes_and_retypes() {
        let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
        db.execute_statement(
            "CREATE TABLE orders AS SELECT * FROM (VALUES ('1', '10')) t(id, amount)",
        )
        .unwrap();
        TableProfile::refresh_stale(&db).unwrap();
        assert_eq!(run(&db, &[]).unwrap(), "orders (1 rows, 2 warnings)\n");

        let shown = run(&db, &["orders", "--note", "amounts in cents"]).unwrap();
        assert!(
            shown.contains("Note (from the owner): amounts in cents"),
            "{shown}"
        );
        assert!(
            shown.contains("amount: numbers stored as text (100% parse as numbers); cast before summing or comparing\n")
                && shown.contains("Fix with: quack tables orders --retype id=DOUBLE,amount=DOUBLE"),
            "{shown}"
        );
        let fixed = run(&db, &["orders", "--retype", "amount=DOUBLE,id=BIGINT"]).unwrap();
        assert!(
            fixed.contains("  - amount (DOUBLE)") && !fixed.contains("Warnings"),
            "{fixed}"
        );
        assert_eq!(
            run(&db, &[]).unwrap(),
            "orders (1 rows): amounts in cents\n"
        );

        let json = run(&db, &["orders", "--format", "json"]).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value.get("note").and_then(|n| n.as_str()),
            Some("amounts in cents")
        );

        assert!(run(&db, &["ghost"]).is_err());
        assert!(run(&db, &["orders", "--retype", "id=DATE"]).is_err());
        assert!(run(&db, &["--note", "x"]).is_err(), "a note needs a table");
    }
}
