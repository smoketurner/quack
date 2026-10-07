//! What a table's columns hold (issue #403): each column's counts, a few
//! common values, and how much of a text column reads as numbers or dates,
//! kept in `_quack_table_profiles` with the row count they were taken at.
//! A profile is taken when a table is loaded or imported and again after a
//! write changes the row count; one whose count no longer matches is not
//! shown, so a stale profile never misleads. Warnings are worked out when
//! read, since whether a column is a key depends on the ontology.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::storage::workspace::{INTERNAL_PREFIX, WorkspaceDb, quote_ident};

/// The profiles and the owners' notes on tables.
pub const DDL: &str = "CREATE TABLE IF NOT EXISTS _quack_table_profiles (
    table_name TEXT PRIMARY KEY,
    row_count BIGINT NOT NULL,
    profiled_at TIMESTAMP DEFAULT now(),
    columns JSON NOT NULL
);
CREATE TABLE IF NOT EXISTS _quack_table_notes (
    table_name TEXT PRIMARY KEY,
    note TEXT NOT NULL,
    edited_by TEXT,
    edited_at TIMESTAMP DEFAULT now()
);";

/// Columns past this many are left out of a profile: one statement
/// aggregates every profiled column, and a very wide table would make it
/// enormous.
const PROFILED_COLUMNS: usize = 200;
/// Common values kept per column.
const SAMPLES: usize = 3;
/// A kept value longer than this is cut, with an ellipsis.
const SAMPLE_CHARS: usize = 60;
/// A column at least this empty is worth a warning.
const HIGH_NULL_SHARE: f64 = 0.5;
/// A text column whose values read as numbers or dates at least this often
/// was probably loaded with the wrong type.
const MISTYPED_SHARE: f64 = 0.9;

/// What a column's `DuckDB` type says about its values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnKind {
    Boolean,
    /// A date or timestamp.
    Temporal,
    /// An integer, float, or decimal type.
    Numeric,
    /// Text: the values decide what it holds.
    Text,
    /// Anything else (lists, structs, blobs).
    Other,
}

impl ColumnKind {
    #[must_use]
    pub fn of(duckdb_type: &str) -> Self {
        const NUMERIC: [&str; 11] = [
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
            "FLOAT",
            "DOUBLE",
        ];
        let t = duckdb_type.trim().to_ascii_uppercase();
        if t == "BOOLEAN" {
            Self::Boolean
        } else if t.starts_with("DATE") || t.starts_with("TIMESTAMP") {
            Self::Temporal
        } else if NUMERIC.contains(&t.as_str()) || t.starts_with("DECIMAL") {
            Self::Numeric
        } else if t == "VARCHAR" {
            Self::Text
        } else {
            Self::Other
        }
    }
}

/// A type a column can be given: by `--types col=TYPE` at ingest or import,
/// or by the Fix-type action. A closed list, since the name goes into the
/// statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "UPPERCASE")]
pub enum ColumnType {
    Varchar,
    Bigint,
    Double,
    Date,
    Timestamp,
    Boolean,
}

text_enum!(ColumnType, "column type", {
    Varchar => "VARCHAR",
    Bigint => "BIGINT",
    Double => "DOUBLE",
    Date => "DATE",
    Timestamp => "TIMESTAMP",
    Boolean => "BOOLEAN",
});

/// Columns to retype after a load, from `--types col=TYPE,...`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ColumnTypes(Vec<(String, ColumnType)>);

impl ColumnTypes {
    #[must_use]
    pub fn new(types: Vec<(String, ColumnType)>) -> Self {
        Self(types)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Several `--types` flags as one list.
    #[must_use]
    pub fn joined(all: Vec<Self>) -> Self {
        Self(all.into_iter().flat_map(|t| t.0).collect())
    }

    /// Give each named column of `table` its type. Every value must
    /// convert: a value that does not fails the whole change, naming it,
    /// rather than turning into an empty cell.
    ///
    /// # Errors
    ///
    /// Returns an error naming the column when the table lacks it or one
    /// of its values does not convert.
    pub fn apply(&self, db: &WorkspaceDb, table: &str) -> Result<()> {
        for (column, kind) in &self.0 {
            Retype {
                table,
                column,
                to: *kind,
            }
            .run(db)?;
        }
        Ok(())
    }
}

impl ColumnTypes {
    /// After a file loads as `tables`: give each named column its type in
    /// every table that has it, then profile the tables.
    ///
    /// # Errors
    ///
    /// Returns an `Ingestion` error when no loaded table has a named column
    /// or a value does not convert.
    pub fn finish_load(&self, db: &WorkspaceDb, tables: &[String]) -> Result<()> {
        for (column, kind) in &self.0 {
            let mut found = false;
            for table in tables {
                if db
                    .describe_columns(table)?
                    .iter()
                    .any(|c| &c.name == column)
                {
                    Retype {
                        table,
                        column,
                        to: *kind,
                    }
                    .run(db)?;
                    found = true;
                }
            }
            if !found {
                return Err(Error::Ingestion(format!(
                    "--types names column '{column}', which the loaded {} not have",
                    if tables.len() == 1 {
                        "table does"
                    } else {
                        "tables do"
                    }
                )));
            }
        }
        for table in tables {
            TableProfile::refresh_or_warn(db, table);
        }
        Ok(())
    }
}

/// `col=TYPE`, several separated by commas or given one flag at a time.
/// The `COLUMN=TYPE,...` text `FromStr` reads back.
impl fmt::Display for ColumnTypes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (column, kind)) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(",")?;
            }
            write!(f, "{column}={}", kind.as_str())?;
        }
        Ok(())
    }
}

impl std::str::FromStr for ColumnTypes {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        let mut out = Vec::new();
        for pair in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let Some((column, kind)) = pair.split_once('=') else {
                return Err(Error::Ingestion(format!(
                    "'{pair}' is not COLUMN=TYPE (types: {})",
                    ColumnType::ALL
                        .iter()
                        .map(|t| t.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            };
            let column = column.trim();
            if column.is_empty() {
                return Err(Error::Ingestion(format!("'{pair}' names no column")));
            }
            out.push((column.to_owned(), kind.parse()?));
        }
        Ok(Self(out))
    }
}

/// One column's change of type.
#[derive(Debug, Clone, Copy)]
pub struct Retype<'a> {
    pub table: &'a str,
    pub column: &'a str,
    pub to: ColumnType,
}

impl Retype<'_> {
    /// `ALTER TABLE ... SET DATA TYPE` with a strict cast, then a fresh
    /// profile.
    ///
    /// # Errors
    ///
    /// Returns an `Analysis` error naming the column when the table lacks
    /// it or a value does not convert.
    pub fn run(&self, db: &WorkspaceDb) -> Result<()> {
        let columns = db.describe_columns(self.table)?;
        if !columns.iter().any(|c| c.name == self.column) {
            return Err(Error::Analysis(format!(
                "table '{}' has no column '{}'",
                self.table, self.column
            )));
        }
        let column = quote_ident(self.column);
        let sql = format!(
            "ALTER TABLE {} ALTER COLUMN {column} SET DATA TYPE {} USING CAST({column} AS {})",
            quote_ident(self.table),
            self.to,
            self.to
        );
        db.execute_statement(&sql).map_err(|e| {
            Error::Analysis(format!(
                "column '{}' of '{}' does not convert to {}: {e}",
                self.column, self.table, self.to
            ))
        })?;
        TableProfile::refresh(db, self.table)?;
        tracing::info!(table = self.table, column = self.column, to = %self.to, "changed a column's type");
        Ok(())
    }
}

/// One column's counts and common values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ColumnProfile {
    pub name: String,
    pub duckdb_type: String,
    pub non_null: u64,
    pub distinct: u64,
    /// The most common values, as text, cut to 60 characters.
    pub samples: Vec<String>,
    /// For a text column, the share of non-null values that cast to a
    /// number; 0 otherwise.
    #[serde(default)]
    pub number_share: f64,
    /// For a text column, the share of non-null values that cast to a
    /// date; 0 otherwise.
    #[serde(default)]
    pub date_share: f64,
}

impl ColumnProfile {
    #[must_use]
    pub fn kind(&self) -> ColumnKind {
        ColumnKind::of(&self.duckdb_type)
    }

    /// What is worth telling someone about the column, given the table's
    /// row count and whether the column is the table's key.
    #[must_use]
    pub fn warnings(&self, rows: u64, role: ColumnRole) -> Vec<ColumnWarning> {
        let mut out = Vec::new();
        if rows == 0 {
            return out;
        }
        if self.non_null == 0 {
            out.push(ColumnWarning::AllNull);
            return out;
        }
        let empty = Share::of(rows.saturating_sub(self.non_null), rows);
        if empty.0 >= HIGH_NULL_SHARE {
            out.push(ColumnWarning::HighNullShare { share: empty });
        }
        if self.kind() == ColumnKind::Text {
            if self.number_share >= MISTYPED_SHARE {
                out.push(ColumnWarning::NumericText {
                    share: Share(self.number_share),
                });
            } else if self.date_share >= MISTYPED_SHARE {
                out.push(ColumnWarning::DateText {
                    share: Share(self.date_share),
                });
            }
        }
        if role == ColumnRole::Key && self.distinct < self.non_null {
            out.push(ColumnWarning::DuplicateKey {
                duplicates: self.non_null.saturating_sub(self.distinct),
            });
        }
        out
    }
}

/// Whether a column identifies its rows: the key column of the table's
/// ontology mapping, or a column named `id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnRole {
    Key,
    Value,
}

/// A fraction between 0 and 1, shown as a percentage.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, utoipa::ToSchema)]
#[serde(transparent)]
pub struct Share(pub f64);

impl Share {
    #[must_use]
    pub fn of(part: u64, whole: u64) -> Self {
        if whole == 0 {
            return Self(0.0);
        }
        #[expect(clippy::cast_precision_loss, reason = "a ratio of counts")]
        Self(part as f64 / whole as f64)
    }

    /// Every value counted, not just nearly every.
    #[must_use]
    pub fn is_whole(self) -> bool {
        self.0 >= 1.0
    }
}

impl fmt::Display for Share {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.0}%", (self.0 * 100.0).clamp(0.0, 100.0))
    }
}

/// Something about a column that makes a query over it likely to go wrong.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, utoipa::ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ColumnWarning {
    /// Every value is empty.
    AllNull,
    /// At least half the values are empty.
    HighNullShare { share: Share },
    /// Numbers stored as text: sums and comparisons treat them as strings.
    NumericText { share: Share },
    /// Dates stored as text: ranges and `date_trunc` do not apply.
    DateText { share: Share },
    /// The key column repeats values, so joins on it multiply rows.
    DuplicateKey { duplicates: u64 },
}

impl ColumnWarning {
    /// The type that fixes this warning, when every value converts to it.
    #[must_use]
    pub fn fix(self) -> Option<ColumnType> {
        match self {
            Self::NumericText { share } if share.is_whole() => Some(ColumnType::Double),
            Self::DateText { share } if share.is_whole() => Some(ColumnType::Date),
            Self::AllNull
            | Self::HighNullShare { .. }
            | Self::NumericText { .. }
            | Self::DateText { .. }
            | Self::DuplicateKey { .. } => None,
        }
    }
}

impl fmt::Display for ColumnWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllNull => f.write_str("every value is empty"),
            Self::HighNullShare { share } => write!(f, "{share} of values are empty"),
            Self::NumericText { share } => write!(
                f,
                "numbers stored as text ({share} parse as numbers); cast before summing or comparing"
            ),
            Self::DateText { share } => write!(
                f,
                "dates stored as text ({share} parse as dates); cast before filtering by date"
            ),
            Self::DuplicateKey { duplicates } => {
                write!(
                    f,
                    "key column is not unique ({duplicates} repeated); joins on it multiply rows"
                )
            }
        }
    }
}

/// A table's profile as stored.
#[derive(Debug, Clone, PartialEq, Serialize, utoipa::ToSchema)]
pub struct TableProfile {
    pub table: String,
    pub row_count: u64,
    pub profiled_at: String,
    pub columns: Vec<ColumnProfile>,
}

/// One column's warning, by name.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Flagged {
    pub column: String,
    pub warning: ColumnWarning,
}

impl TableProfile {
    /// Profile `table` now, without storing it: one statement over every
    /// profiled column.
    ///
    /// # Errors
    ///
    /// Returns an error if the table cannot be read.
    pub fn compute(db: &WorkspaceDb, table: &str) -> Result<Self> {
        let described = db.describe_columns(table)?;
        let columns: Vec<_> = described.into_iter().take(PROFILED_COLUMNS).collect();
        let mut select = vec![String::from("count(*)")];
        for column in &columns {
            let q = quote_ident(&column.name);
            select.push(format!("count({q})"));
            select.push(format!("count(DISTINCT {q})"));
            select.push(format!(
                "CAST(to_json(approx_top_k(CAST({q} AS VARCHAR), {SAMPLES})) AS VARCHAR)"
            ));
            if ColumnKind::of(&column.column_type) == ColumnKind::Text {
                select.push(format!(
                    "avg(CASE WHEN {q} IS NULL THEN NULL WHEN TRY_CAST({q} AS DOUBLE) IS NOT NULL THEN 1.0 ELSE 0.0 END)"
                ));
                select.push(format!(
                    "avg(CASE WHEN {q} IS NULL THEN NULL WHEN TRY_CAST({q} AS DATE) IS NOT NULL THEN 1.0 ELSE 0.0 END)"
                ));
            } else {
                select.push(String::from("NULL::DOUBLE"));
                select.push(String::from("NULL::DOUBLE"));
            }
        }
        let sql = format!("SELECT {} FROM {}", select.join(", "), quote_ident(table));
        db.under_timeout(|db| {
            db.connection()
                .query_row(&sql, [], |row| {
                    let rows: i64 = row.get(0)?;
                    let mut profiled = Vec::with_capacity(columns.len());
                    for (i, column) in columns.iter().enumerate() {
                        let at = i.saturating_mul(5).saturating_add(1);
                        let non_null: i64 = row.get(at)?;
                        let distinct: i64 = row.get(at.saturating_add(1))?;
                        let samples: Option<String> = row.get(at.saturating_add(2))?;
                        let samples: Vec<Option<String>> = samples
                            .and_then(|json| serde_json::from_str(&json).ok())
                            .unwrap_or_default();
                        let number_share: Option<f64> = row.get(at.saturating_add(3))?;
                        let date_share: Option<f64> = row.get(at.saturating_add(4))?;
                        profiled.push(ColumnProfile {
                            name: column.name.clone(),
                            duckdb_type: column.column_type.to_ascii_uppercase(),
                            non_null: u64::try_from(non_null).unwrap_or(0),
                            distinct: u64::try_from(distinct).unwrap_or(0),
                            samples: samples.into_iter().flatten().map(|s| cut(&s)).collect(),
                            number_share: number_share.unwrap_or(0.0),
                            date_share: date_share.unwrap_or(0.0),
                        });
                    }
                    Ok(Self {
                        table: table.to_owned(),
                        row_count: u64::try_from(rows).unwrap_or(0),
                        profiled_at: String::new(),
                        columns: profiled,
                    })
                })
                .map_err(Error::from)
        })
    }

    /// Profile `table` and store the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the table cannot be read or the row not written.
    pub fn refresh(db: &WorkspaceDb, table: &str) -> Result<Self> {
        let profile = Self::compute(db, table)?;
        db.connection().execute(
            "INSERT OR REPLACE INTO _quack_table_profiles (table_name, row_count, profiled_at, columns) \
             VALUES (?, ?, now(), ?)",
            duckdb::params![
                profile.table,
                i64::try_from(profile.row_count).unwrap_or(i64::MAX),
                serde_json::to_string(&profile.columns)?
            ],
        )?;
        Ok(profile)
    }

    /// Profile `table`, or say why not and go on: a profile is advice, so a
    /// table it cannot be taken of (an unusual column type) still loads.
    pub fn refresh_or_warn(db: &WorkspaceDb, table: &str) {
        if let Err(e) = Self::refresh(db, table) {
            tracing::warn!(table, error = %e, "could not profile the table");
        }
    }

    /// Profile every stored table whose row count changed since its
    /// profile, or that has none, and forget the profiles of tables that
    /// are gone. Run after a statement that may have written; returns how
    /// many it profiled.
    ///
    /// # Errors
    ///
    /// Returns an error if the catalog or the stored profiles cannot be read.
    pub fn refresh_stale(db: &WorkspaceDb) -> Result<usize> {
        let stored = Self::stored_counts(db)?;
        let tables = Self::base_tables(db)?;
        let mut profiled = 0_usize;
        for table in &tables {
            let rows = match db.count_rows(table) {
                Ok(rows) => u64::try_from(rows).unwrap_or(0),
                Err(e) => {
                    tracing::warn!(table, error = %e, "could not count the table's rows");
                    continue;
                }
            };
            if stored.get(table) == Some(&rows) {
                continue;
            }
            Self::refresh_or_warn(db, table);
            profiled = profiled.saturating_add(1);
        }
        for gone in stored.keys().filter(|t| !tables.contains(t)) {
            db.connection().execute(
                "DELETE FROM _quack_table_profiles WHERE table_name = ?",
                duckdb::params![gone],
            )?;
        }
        Ok(profiled)
    }

    /// [`Self::refresh_stale`] after a statement that wrote; a failure is
    /// logged, never the statement's.
    pub fn after_write(db: &WorkspaceDb) {
        if let Err(e) = Self::refresh_stale(db) {
            tracing::warn!(error = %e, "could not refresh table profiles after a write");
        }
    }

    /// The user's stored tables: views are left out, since profiling one
    /// runs its query.
    fn base_tables(db: &WorkspaceDb) -> Result<Vec<String>> {
        let mut stmt = db.connection().prepare(
            "SELECT table_name FROM duckdb_tables() \
             WHERE schema_name = 'main' AND NOT temporary AND NOT starts_with(lower(table_name), ?) \
             ORDER BY table_name",
        )?;
        let names = stmt
            .query_map(duckdb::params![INTERNAL_PREFIX], |row| row.get(0))?
            .collect::<duckdb::Result<Vec<String>>>()?;
        Ok(names)
    }

    fn stored_counts(db: &WorkspaceDb) -> Result<BTreeMap<String, u64>> {
        let mut stmt = db
            .connection()
            .prepare("SELECT table_name, row_count FROM _quack_table_profiles")?;
        let mut rows = stmt.query([])?;
        let mut out = BTreeMap::new();
        while let Some(row) = rows.next()? {
            let count: i64 = row.get(1)?;
            out.insert(row.get(0)?, u64::try_from(count).unwrap_or(0));
        }
        Ok(out)
    }

    /// The stored profile of `table` when it was taken at `rows` rows;
    /// `None` when there is none or the table has changed since.
    ///
    /// # Errors
    ///
    /// Returns an error if the row cannot be read or does not parse.
    pub fn current(db: &WorkspaceDb, table: &str, rows: u64) -> Result<Option<Self>> {
        let mut stmt = db.connection().prepare(
            "SELECT row_count, CAST(profiled_at AS VARCHAR), CAST(columns AS VARCHAR) \
             FROM _quack_table_profiles WHERE table_name = ?",
        )?;
        let mut found = stmt.query(duckdb::params![table])?;
        let Some(row) = found.next()? else {
            return Ok(None);
        };
        let row_count: i64 = row.get(0)?;
        let row_count = u64::try_from(row_count).unwrap_or(0);
        if row_count != rows {
            return Ok(None);
        }
        let columns: String = row.get(2)?;
        Ok(Some(Self {
            table: table.to_owned(),
            row_count,
            profiled_at: row.get(1)?,
            columns: serde_json::from_str(&columns)?,
        }))
    }

    /// Every stored profile, by table, whatever its row count: the common
    /// values the table search reads.
    ///
    /// # Errors
    ///
    /// Returns an error if the rows cannot be read or do not parse.
    pub fn all(db: &WorkspaceDb) -> Result<BTreeMap<String, Self>> {
        let mut stmt = db.connection().prepare(
            "SELECT table_name, row_count, CAST(profiled_at AS VARCHAR), CAST(columns AS VARCHAR) \
             FROM _quack_table_profiles",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = BTreeMap::new();
        while let Some(row) = rows.next()? {
            let table: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            let columns: String = row.get(3)?;
            out.insert(
                table.clone(),
                Self {
                    table,
                    row_count: u64::try_from(count).unwrap_or(0),
                    profiled_at: row.get(2)?,
                    columns: serde_json::from_str(&columns)?,
                },
            );
        }
        Ok(out)
    }

    #[must_use]
    pub fn column(&self, name: &str) -> Option<&ColumnProfile> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// The column whose values are all present and all different, an `id`
    /// first, then one ending in `id`: what looks like the table's key.
    #[must_use]
    pub fn key_column(&self) -> Option<&str> {
        let rows = self.row_count;
        self.columns
            .iter()
            .filter(|c| rows > 0 && c.non_null == rows && c.distinct == rows)
            .min_by_key(|c| {
                let lower = c.name.to_ascii_lowercase();
                if lower == "id" {
                    0
                } else if lower.ends_with("_id") || lower.ends_with("id") {
                    1
                } else {
                    2
                }
            })
            .map(|c| c.name.as_str())
    }

    /// Every column's warnings, in column order. `key` is the column the
    /// table's ontology mapping names as its key; a column named `id` is
    /// held to the same rule.
    #[must_use]
    pub fn warnings(&self, key: Option<&str>) -> Vec<Flagged> {
        let mut out = Vec::new();
        for column in &self.columns {
            let role =
                if key == Some(column.name.as_str()) || column.name.eq_ignore_ascii_case("id") {
                    ColumnRole::Key
                } else {
                    ColumnRole::Value
                };
            for warning in column.warnings(self.row_count, role) {
                out.push(Flagged {
                    column: column.name.clone(),
                    warning,
                });
            }
        }
        out
    }
}

/// `text` cut to [`SAMPLE_CHARS`], with an ellipsis when cut.
fn cut(text: &str) -> String {
    let mut out: String = text.chars().take(SAMPLE_CHARS).collect();
    if text.chars().count() > SAMPLE_CHARS {
        out.push('\u{2026}');
    }
    out
}

/// An owner's note on a table: what it holds, where it comes from, how to
/// read it. Shown under the table wherever it is described.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TableNote {
    pub table: String,
    pub note: String,
    pub edited_by: Option<String>,
    pub edited_at: String,
}

impl TableNote {
    /// The longest note kept, in characters: a note is a few lines, and
    /// it goes into every prompt that describes its table.
    pub const MAX_CHARS: usize = 2_000;

    /// The note on `table`, if it has one.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get(db: &WorkspaceDb, table: &str) -> Result<Option<Self>> {
        Ok(Self::query(db, Some(table))?.into_values().next())
    }

    /// Every note, by table.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn all(db: &WorkspaceDb) -> Result<BTreeMap<String, Self>> {
        Self::query(db, None)
    }

    fn query(db: &WorkspaceDb, table: Option<&str>) -> Result<BTreeMap<String, Self>> {
        let mut stmt = db.connection().prepare(
            "SELECT table_name, note, edited_by, CAST(edited_at AS VARCHAR) FROM _quack_table_notes \
             WHERE ? IS NULL OR table_name = ?",
        )?;
        let mut rows = stmt.query(duckdb::params![table, table])?;
        let mut out = BTreeMap::new();
        while let Some(row) = rows.next()? {
            let note = Self {
                table: row.get(0)?,
                note: row.get(1)?,
                edited_by: row.get(2)?,
                edited_at: row.get(3)?,
            };
            out.insert(note.table.clone(), note);
        }
        Ok(out)
    }

    /// Set the note on `table`, or remove it when `note` is blank.
    ///
    /// # Errors
    ///
    /// Returns an error when the table does not exist, the note is longer
    /// than [`Self::MAX_CHARS`], or the write fails.
    pub fn set(db: &WorkspaceDb, table: &str, note: &str, by: Option<&str>) -> Result<()> {
        if !db.list_tables()?.iter().any(|t| t == table) {
            return Err(Error::Analysis(format!("no table named '{table}'")));
        }
        let note = note.trim();
        if note.chars().count() > Self::MAX_CHARS {
            return Err(Error::Analysis(format!(
                "a table note holds at most {} characters",
                Self::MAX_CHARS
            )));
        }
        if note.is_empty() {
            db.connection().execute(
                "DELETE FROM _quack_table_notes WHERE table_name = ?",
                duckdb::params![table],
            )?;
        } else {
            db.connection().execute(
                "INSERT OR REPLACE INTO _quack_table_notes (table_name, note, edited_by, edited_at) \
                 VALUES (?, ?, ?, now())",
                duckdb::params![table, note, by],
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
