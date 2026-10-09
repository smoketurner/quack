//! What a classification will do, worked out by reading alone: the source
//! table and its text columns, the key, the output table and its columns,
//! and how many rows are left to label.

use super::columns::OutputColumns;
use super::{
    Classification, ClassificationOutline, ClassificationRun, ClassifyError, Effect, KeyCandidate,
    KeyColumn, OutlineQuestion, QuestionSet, Rows, TEXT_COLUMNS,
};
use crate::error::{Error, Result};
use crate::ids::DocumentId;
use crate::ingestion::TableName;
use crate::llm::decision::STATE_CHARS;
use crate::storage::profile::{ColumnKind, TableProfile};
use crate::storage::workspace::{ColumnInfo, DocumentSource, WorkspaceDb, quote_ident};
use crate::storage::writer::Claimed;

/// Closest-to-unique columns a refusal names.
const CLOSEST: usize = 3;

/// A classification checked against the workspace.
#[derive(Debug, Clone)]
pub(super) struct Plan {
    pub source: TableName,
    pub key: KeyColumn,
    pub text_columns: Vec<String>,
    pub set: QuestionSet,
    pub rows: Rows,
    /// The labels table, spelled as the catalog spells it when it exists
    /// and in lower case when it does not, so one output has one spelling
    /// whatever the question set's name was typed as.
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
    fn checked(db: &WorkspaceDb, source: &TableName, column: &ColumnInfo) -> Result<Self> {
        let key = Self {
            name: column.name.clone(),
            duckdb_type: column.column_type.clone(),
        };
        if !Self::allows(&key.duckdb_type) {
            return Err(ClassifyError::KeyType {
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
            return Err(ClassifyError::KeyNotUnique {
                table: source.to_string(),
                column: key.name,
                rows,
                distinct,
                missing: rows.saturating_sub(present),
            }
            .into());
        }
        if unreadable > 0 {
            return Err(ClassifyError::KeyType {
                column: key.name,
                duckdb_type: key.duckdb_type,
            }
            .into());
        }
        Ok(key)
    }

    /// The key the request names, or the table's id column.
    fn resolve(
        db: &WorkspaceDb,
        source: &TableName,
        columns: &[ColumnInfo],
        wanted: Option<&str>,
    ) -> Result<Self> {
        if let Some(wanted) = wanted {
            let column = Plan::column(source, columns, wanted)?;
            return Self::checked(db, source, column);
        }
        let mut candidates: Vec<&ColumnInfo> = columns
            .iter()
            .filter(|c| TableProfile::is_id_name(&c.name))
            .collect();
        candidates.sort_by_key(|c| !c.name.eq_ignore_ascii_case("id"));
        for column in candidates {
            match Self::checked(db, source, column) {
                Ok(key) => return Ok(key),
                Err(Error::Classify(
                    ClassifyError::KeyNotUnique { .. } | ClassifyError::KeyType { .. },
                )) => {}
                Err(other) => return Err(other),
            }
        }
        Err(Self::none_found(db, source)?)
    }

    /// The refusal for a table with no id column, naming the columns that
    /// could be the key and those that come closest.
    fn none_found(db: &WorkspaceDb, source: &TableName) -> Result<Error> {
        let profile = TableProfile::compute(db, source.as_str())?;
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
        Ok(ClassifyError::NoKey {
            table: source.to_string(),
            unique,
            closest: near,
        }
        .into())
    }
}

impl Plan {
    /// The column of `source` named `wanted`, without regard to case.
    fn column<'a>(
        source: &TableName,
        columns: &'a [ColumnInfo],
        wanted: &str,
    ) -> Result<&'a ColumnInfo> {
        columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(wanted.trim()))
            .ok_or_else(|| {
                ClassifyError::NoColumn {
                    table: source.to_string(),
                    column: wanted.to_owned(),
                }
                .into()
            })
    }

    /// The text columns the request names, as the catalog spells them.
    fn text_columns(
        source: &TableName,
        columns: &[ColumnInfo],
        wanted: &[String],
    ) -> Result<Vec<String>> {
        if wanted.is_empty() || wanted.len() > TEXT_COLUMNS {
            return Err(ClassifyError::TextColumns(wanted.len()).into());
        }
        let mut resolved: Vec<String> = Vec::with_capacity(wanted.len());
        for name in wanted {
            let column = Self::column(source, columns, name)?;
            if resolved.contains(&column.name) {
                return Err(ClassifyError::DuplicateColumn(column.name.clone()).into());
            }
            resolved.push(column.name.clone());
        }
        Ok(resolved)
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
                return Err(ClassifyError::OutputTaken {
                    table: output.to_string(),
                }
                .into());
            }
            None if existing.is_some() => {
                return Err(ClassifyError::OutputTaken {
                    table: output.to_string(),
                }
                .into());
            }
            None => None,
        };
        Ok((output, existing.is_some(), document))
    }

    /// Check `request` against `db` and work out the rest.
    ///
    /// # Errors
    ///
    /// Returns a [`ClassifyError`] naming what cannot be carried out, or a
    /// query error.
    pub(super) fn resolve(db: &WorkspaceDb, request: &Classification) -> Result<Self> {
        let catalog = db.list_tables()?;
        let source = TableName::exact(&catalog, &request.table)?;
        let described = db.describe_columns(source.as_str())?;
        let text_columns = Self::text_columns(&source, &described, &request.text_columns)?;
        let key = KeyColumn::resolve(db, &source, &described, request.key.as_deref())?;
        let set = request.question_set.clone();
        let wanted = TableName::sanitized(&format!("{source}_{}", set.name));
        wanted.check_unreserved()?;
        let (output, output_exists, document) = Self::output(db, &catalog, &wanted)?;
        let columns = OutputColumns::new(source.as_str(), &key.name, &set.questions)?;
        let mut plan = Self {
            source,
            key,
            text_columns,
            set,
            rows: request.rows,
            output,
            output_exists,
            document,
            columns,
            remaining: 0,
        };
        plan.remaining = plan.count_remaining(db)?;
        Ok(plan)
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

    /// Refuse a run that adds rows to an output labelled under another
    /// definition than this plan's and the model weights `digest`: a run
    /// that labels everything again replaces the labels, so it never is.
    pub(super) fn check_definition(
        &self,
        in_force: Option<&ClassificationRun>,
        digest: &str,
    ) -> std::result::Result<(), ClassifyError> {
        let Some(run) = in_force.filter(|_| self.rows == Rows::Missing && self.output_exists)
        else {
            return Ok(());
        };
        let differs = run.differs(self, digest);
        if differs.is_empty() {
            return Ok(());
        }
        Err(ClassifyError::DefinitionChanged {
            table: self.output.to_string(),
            differs,
        })
    }

    /// The claim a run into this plan's output takes.
    pub(super) fn claimed(&self) -> Claimed {
        Claimed::Classify(self.output.as_str().to_ascii_lowercase())
    }

    /// Whether the output is still the table of labels of the document the
    /// plan found, and not a table that took its name since.
    pub(super) fn still_owns_output(&self, db: &WorkspaceDb) -> Result<bool> {
        Ok(db.table_owner(self.output.as_str())?.is_some_and(|owner| {
            owner.source == DocumentSource::Classify && self.document.as_ref() == Some(&owner.id)
        }))
    }

    /// What the plan would do, as an interface says it.
    pub(super) fn outline(&self) -> ClassificationOutline {
        let effect = match (self.output_exists, self.rows) {
            (false, _) => Effect::NewTable,
            (true, Rows::Missing) => Effect::AddsRows,
            (true, Rows::All) => Effect::ReplacesLabels,
        };
        ClassificationOutline {
            source_table: self.source.to_string(),
            output_table: self.output.to_string(),
            key_column: self.key.name.clone(),
            remaining: self.remaining,
            questions: self
                .set
                .questions
                .iter()
                .map(|(name, question)| OutlineQuestion {
                    name: name.to_string(),
                    instructions: question.instructions().as_str().to_owned(),
                })
                .collect(),
            effect,
        }
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
        for column in &self.text_columns {
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
