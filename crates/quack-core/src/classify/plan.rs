//! What a run will do, worked out by reading alone: the source table, its
//! key and text columns, the output table and its columns, why the run
//! labels every row again, and how many rows are left to label.

use super::columns::OutputColumns;
use super::{
    DraftQuestion, Effect, Error, KeyCandidate, KeyColumn, KeyReason, LabelSet, Outline,
    RelabelReason, Rows, Run, TEXT_COLUMNS,
};
use crate::error::{Error as CoreError, Result};
use crate::ids::DocumentId;
use crate::ingestion::TableName;
use crate::llm::decision::STATE_CHARS;
use crate::storage::profile::{ColumnKind, TableProfile};
use crate::storage::workspace::{ColumnInfo, DocumentSource, WorkspaceDb, quote_ident};
use crate::storage::writer::Claimed;

/// Closest-to-unique columns a refusal names.
const CLOSEST: usize = 3;
/// A text column serves as a key only when its values average at most this
/// many characters.
const SHORT_KEY_CHARS: f64 = 64.0;

/// A set checked against the workspace.
#[derive(Debug, Clone)]
pub(super) struct Plan {
    pub source: TableName,
    pub key: KeyColumn,
    /// The set as the catalog spells it, with the key's reason worked out
    /// from the table.
    pub set: LabelSet,
    /// Which rows the run labels: those the output lacks, or every row.
    pub rows: Rows,
    /// Why the run labels every row again, when it does.
    pub because: Option<RelabelReason>,
    /// The labels table, spelled as the catalog spells it when it exists
    /// and in lower case when it does not, so one output has one spelling.
    pub output: TableName,
    pub output_exists: bool,
    /// The document that owns the output, when one does.
    pub document: Option<DocumentId>,
    pub columns: OutputColumns,
    /// Rows a run would label now.
    pub remaining: u64,
}

impl KeyColumn {
    /// Whether values of `duckdb_type` can serve as a key: the ones that
    /// read back from their text form.
    fn allows(duckdb_type: &str) -> bool {
        let upper = duckdb_type.trim().to_ascii_uppercase();
        ColumnKind::of(&upper) != ColumnKind::Other || upper == "UUID"
    }

    /// `column` as the key of `source`, after checking that it is present
    /// and different in every row and reads back from text unchanged.
    pub(super) fn checked(
        db: &WorkspaceDb,
        source: &TableName,
        column: &ColumnInfo,
    ) -> Result<Self> {
        let key = Self {
            name: column.name.clone(),
            duckdb_type: column.column_type.clone(),
        };
        if !Self::allows(&key.duckdb_type) {
            return Err(Error::KeyType {
                column: key.name,
                duckdb_type: key.duckdb_type,
            }
            .into());
        }
        let (name, ty) = (quote_ident(&key.name), &key.duckdb_type);
        let sql = format!(
            "SELECT count(*), count({name}), count(DISTINCT {name}), \
             count(*) FILTER (WHERE CAST(CAST({name} AS VARCHAR) AS {ty}) IS DISTINCT FROM {name}) \
             FROM {}",
            quote_ident(source.as_str())
        );
        let counts: [i64; 4] = db.connection().query_row(&sql, [], |row| {
            Ok([row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?])
        })?;
        let [rows, present, distinct, unreadable] = counts.map(|n| u64::try_from(n).unwrap_or(0));
        if present != rows || distinct != present {
            return Err(Error::KeyNotUnique {
                table: source.to_string(),
                column: key.name,
                rows,
                distinct,
                missing: rows.saturating_sub(present),
            }
            .into());
        }
        if unreadable > 0 {
            return Err(Error::KeyType {
                column: key.name,
                duckdb_type: key.duckdb_type,
            }
            .into());
        }
        Ok(key)
    }

    /// The table's key: an id-like column first, then a column of whole
    /// numbers, then one of short text, each with every value present and
    /// different. Never a float, a date, or long text.
    pub(super) fn resolve(
        db: &WorkspaceDb,
        source: &TableName,
        columns: &[ColumnInfo],
    ) -> Result<Self> {
        let mut candidates: Vec<&ColumnInfo> = columns
            .iter()
            .filter(|c| TableProfile::is_id_name(&c.name))
            .collect();
        candidates.sort_by_key(|c| !c.name.eq_ignore_ascii_case("id"));
        for column in candidates {
            if let Some(key) = Self::tried(db, source, column)? {
                return Ok(key);
            }
        }
        let profile = TableProfile::compute(db, source.as_str())?;
        let rows = profile.row_count;
        let unique = |column: &&ColumnInfo| {
            rows > 0
                && profile
                    .column(&column.name)
                    .is_some_and(|p| p.non_null == rows && p.distinct == rows)
        };
        let whole: Vec<&ColumnInfo> = columns
            .iter()
            .filter(|c| ColumnKind::is_integer(&c.column_type))
            .filter(unique)
            .collect();
        for column in whole {
            if let Some(key) = Self::tried(db, source, column)? {
                return Ok(key);
            }
        }
        let short: Vec<&ColumnInfo> = columns
            .iter()
            .filter(|c| ColumnKind::of(&c.column_type) == ColumnKind::Text)
            .filter(unique)
            .collect();
        for column in short {
            if Self::averages_short(db, source, column)?
                && let Some(key) = Self::tried(db, source, column)?
            {
                return Ok(key);
            }
        }
        Err(Self::none_found(&profile, source))
    }

    /// `column` as the key, or `None` when it does not qualify.
    fn tried(db: &WorkspaceDb, source: &TableName, column: &ColumnInfo) -> Result<Option<Self>> {
        match Self::checked(db, source, column) {
            Ok(key) => Ok(Some(key)),
            Err(CoreError::Classify(Error::KeyNotUnique { .. } | Error::KeyType { .. })) => {
                Ok(None)
            }
            Err(other) => Err(other),
        }
    }

    /// Whether the values of a text column average at most
    /// [`SHORT_KEY_CHARS`] characters.
    fn averages_short(db: &WorkspaceDb, source: &TableName, column: &ColumnInfo) -> Result<bool> {
        let sql = format!(
            "SELECT avg(length(CAST({} AS VARCHAR))) FROM {}",
            quote_ident(&column.name),
            quote_ident(source.as_str())
        );
        let average: Option<f64> = db.connection().query_row(&sql, [], |row| row.get(0))?;
        Ok(average.is_some_and(|average| average <= SHORT_KEY_CHARS))
    }

    /// The refusal for a table with no usable key, naming the unique
    /// columns of a type that cannot serve and those that come closest.
    fn none_found(profile: &TableProfile, source: &TableName) -> CoreError {
        let rows = profile.row_count;
        let mut unique = Vec::new();
        let mut near = Vec::new();
        for column in &profile.columns {
            let candidate = KeyCandidate {
                column: column.name.clone(),
                distinct: column.distinct,
                missing: rows.saturating_sub(column.non_null),
            };
            if rows > 0 && column.non_null == rows && column.distinct == rows {
                unique.push(column.name.clone());
            } else {
                near.push(candidate);
            }
        }
        near.sort_by_key(|c| {
            (
                c.missing
                    .saturating_add(rows.saturating_sub(c.missing).saturating_sub(c.distinct)),
                c.column.clone(),
            )
        });
        near.truncate(CLOSEST);
        Error::NoKey {
            table: source.to_string(),
            unique,
            closest: near,
        }
        .into()
    }
}

impl Plan {
    /// The column of `columns` named `wanted`, without regard to case.
    fn column<'a>(columns: &'a [ColumnInfo], wanted: &str) -> Option<&'a ColumnInfo> {
        columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(wanted.trim()))
    }

    /// The text columns the set names, as the catalog spells them.
    fn text_columns(
        source: &TableName,
        columns: &[ColumnInfo],
        wanted: &[String],
    ) -> Result<Vec<String>> {
        let mut resolved: Vec<String> = Vec::with_capacity(wanted.len());
        let mut gone = Vec::new();
        for name in wanted {
            match Self::column(columns, name) {
                Some(column) if !resolved.contains(&column.name) => {
                    resolved.push(column.name.clone());
                }
                Some(_) => {}
                None => gone.push(name.clone()),
            }
        }
        if !gone.is_empty() {
            return Err(Error::SetColumnsGone {
                table: source.to_string(),
                columns: gone,
            }
            .into());
        }
        match resolved.len() {
            0 => Err(Error::NoText {
                table: source.to_string(),
            }
            .into()),
            n if n > TEXT_COLUMNS => Err(Error::TextColumns(n).into()),
            _ => Ok(resolved),
        }
    }

    /// The labels table for `wanted`, in the catalog's spelling when it
    /// exists, and the document that owns it. A table of another name's
    /// making is refused.
    fn output(
        db: &WorkspaceDb,
        catalog: &[String],
        wanted: &TableName,
    ) -> Result<(TableName, bool, Option<DocumentId>)> {
        let existing = TableName::in_catalog(catalog, wanted.as_str());
        let output = existing.clone().unwrap_or_else(|| wanted.lowercased());
        let owner = db.table_owner(output.as_str())?;
        let document = match owner {
            Some(doc) if doc.source == DocumentSource::Classify => Some(doc.id),
            Some(_) => {
                return Err(Error::OutputTaken {
                    table: output.to_string(),
                }
                .into());
            }
            None if existing.is_some() => {
                return Err(Error::OutputTaken {
                    table: output.to_string(),
                }
                .into());
            }
            None => None,
        };
        Ok((output, existing.is_some(), document))
    }

    /// The name the labels of `source` go to.
    pub(super) fn output_name(source: &TableName) -> TableName {
        TableName::sanitized(&format!("{source}_labels"))
    }

    /// Check the set for `table` against `db` and work out the rest. The
    /// plan labels the rows the output lacks until [`Self::labelling`] says
    /// otherwise.
    ///
    /// # Errors
    ///
    /// Returns a [`Error`] naming what cannot be carried out, or a
    /// query error.
    pub(super) fn resolve(db: &WorkspaceDb, table: &str, set: &LabelSet) -> Result<Self> {
        let catalog = db.list_tables()?;
        let source = TableName::exact(&catalog, table)?;
        let described = db.describe_columns(source.as_str())?;
        let text_columns = Self::text_columns(&source, &described, &set.text_columns)?;
        let Some(key_column) = Self::column(&described, &set.key_column) else {
            return Err(Error::SetColumnsGone {
                table: source.to_string(),
                columns: vec![set.key_column.clone()],
            }
            .into());
        };
        let key = KeyColumn::checked(db, &source, key_column)?;
        let wanted = Self::output_name(&source);
        wanted.check_unreserved()?;
        let (output, output_exists, document) = Self::output(db, &catalog, &wanted)?;
        let columns = OutputColumns::new(source.as_str(), &key.name, &set.questions)?;
        let set = LabelSet {
            key_reason: KeyReason::of(&key.name),
            key_column: key.name.clone(),
            text_columns,
            questions: set.questions.clone(),
            sentence: set.sentence.clone(),
        };
        let mut plan = Self {
            source,
            key,
            set,
            rows: Rows::Missing,
            because: None,
            output,
            output_exists,
            document,
            columns,
            remaining: 0,
        };
        plan.remaining = plan.count_remaining(db)?;
        Ok(plan)
    }

    /// Why a run of this plan labels every row again: the set or the model
    /// is not what the labels were made with, or `asked` says so. `None`
    /// for a run that adds the rows the output lacks, or makes the output.
    pub(super) fn why_relabel(
        &self,
        in_force: Option<&Run>,
        digest: &str,
        asked: Rows,
    ) -> Option<RelabelReason> {
        if !self.output_exists {
            return None;
        }
        let changed = in_force.and_then(|run| {
            if !run.key_column.eq_ignore_ascii_case(&self.key.name)
                || !run.key_type.eq_ignore_ascii_case(&self.key.duckdb_type)
                || !run.source_table.eq_ignore_ascii_case(self.source.as_str())
            {
                Some(RelabelReason::KeyChanged)
            } else if run.questions != self.set.questions || !self.reads_columns(&run.text_columns)
            {
                Some(RelabelReason::QuestionsChanged)
            } else if run.model_digest != digest {
                Some(RelabelReason::ModelChanged)
            } else {
                None
            }
        });
        changed.or_else(|| (asked == Rows::All).then_some(RelabelReason::Asked))
    }

    /// This plan, labelling the rows `asked` for, or every row for
    /// `because`; the rows left to label are counted again.
    pub(super) fn labelling(
        mut self,
        db: &WorkspaceDb,
        asked: Rows,
        because: Option<RelabelReason>,
    ) -> Result<Self> {
        self.rows = if self.output_exists && because.is_some() {
            Rows::All
        } else if self.output_exists {
            asked
        } else {
            Rows::Missing
        };
        self.because = because.filter(|_| self.output_exists);
        self.remaining = self.count_remaining(db)?;
        Ok(self)
    }

    /// Rows a run labels now: the source rows whose key the output does
    /// not hold, or every row when the run starts a new output or labels
    /// everything again.
    fn count_remaining(&self, db: &WorkspaceDb) -> Result<u64> {
        let skip_labelled = self.rows == Rows::Missing && self.output_exists;
        let sql = format!(
            "SELECT count(*) FROM {source} s{anti}",
            source = quote_ident(self.source.as_str()),
            anti = if skip_labelled {
                format!(" WHERE {}", self.not_in(self.output.as_str()))
            } else {
                String::new()
            }
        );
        let count: i64 = db.connection().query_row(&sql, [], |row| row.get(0))?;
        Ok(u64::try_from(count).unwrap_or(0))
    }

    /// What the plan would do, as an interface says it.
    pub(super) fn outline(&self, estimate_seconds: Option<u64>) -> Outline {
        let effect = match (self.output_exists, self.rows) {
            (false, _) => Effect::NewTable,
            (true, Rows::Missing) => Effect::AddsRows,
            (true, Rows::All) => Effect::ReplacesLabels {
                because: self.because.unwrap_or(RelabelReason::Asked),
            },
        };
        Outline {
            source_table: self.source.to_string(),
            output_table: self.output.to_string(),
            key_column: self.key.name.clone(),
            key_reason: self.set.key_reason,
            text_columns: self.set.text_columns.clone(),
            remaining: self.remaining,
            questions: self
                .set
                .questions
                .iter()
                .map(|(name, question)| DraftQuestion::of(name, question))
                .collect(),
            effect,
            estimate_seconds,
        }
    }

    /// Whether the text columns a run read are those this plan reads, without
    /// regard to case.
    fn reads_columns(&self, recorded: &[String]) -> bool {
        recorded.len() == self.set.text_columns.len()
            && recorded
                .iter()
                .zip(&self.set.text_columns)
                .all(|(recorded, planned)| recorded.eq_ignore_ascii_case(planned))
    }

    /// The claim a run into this plan's output takes.
    pub(super) fn claimed(&self) -> Claimed {
        Claimed(self.output.as_str().to_ascii_lowercase())
    }

    /// Whether the output is still the table of labels of the document the
    /// plan found, and not a table that took its name since.
    pub(super) fn still_owns_output(&self, db: &WorkspaceDb) -> Result<bool> {
        Ok(db.table_owner(self.output.as_str())?.is_some_and(|owner| {
            owner.source == DocumentSource::Classify && self.document.as_ref() == Some(&owner.id)
        }))
    }

    /// A condition on source rows `s` that holds when `target` has no row
    /// with the same key.
    pub(super) fn not_in(&self, target: &str) -> String {
        let key = quote_ident(&self.key.name);
        format!(
            "NOT EXISTS (SELECT 1 FROM {} o WHERE o.{key} = s.{key})",
            quote_ident(target)
        )
    }

    /// The hidden table a run that labels every row writes into until it
    /// is complete.
    pub(super) fn stage(&self) -> String {
        TableName::stage_of(self.output.as_str())
    }

    /// The table this run's labels are written into.
    pub(super) fn target(&self) -> String {
        if self.rows == Rows::All && self.output_exists {
            self.stage()
        } else {
            self.output.to_string()
        }
    }

    /// The `SELECT` list of a page: the key as text, then each text column
    /// cut to one character past the longest state.
    pub(super) fn page_select(&self) -> String {
        let key = quote_ident(&self.key.name);
        let mut select = vec![format!("CAST(s.{key} AS VARCHAR)")];
        for column in &self.set.text_columns {
            select.push(format!(
                "left(CAST(s.{} AS VARCHAR), {})",
                quote_ident(column),
                STATE_CHARS.saturating_add(1)
            ));
        }
        select.join(", ")
    }

    /// `CAST(? AS <the key's type>)`: a key bound as text.
    pub(super) fn key_param(&self) -> String {
        format!("CAST(? AS {})", self.key.duckdb_type)
    }
}
