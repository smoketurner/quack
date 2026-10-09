//! `_quack_classifications` and the writes of a run: starting one, adding a
//! page of labels, and ending it.

use duckdb::types::Value;
use serde::Serialize;

use super::plan::Plan;
use super::{ClassificationRun, ClassifyError, Rows, RunStatus};
use crate::crypto::sha256_hex;
use crate::error::{Error, Result};
use crate::ids::{DocumentId, RunId};
use crate::ingestion::TableName;
use crate::storage::control::ResourceKind;
use crate::storage::workspace::{
    DocumentSource, DocumentStatus, NewDocument, WorkspaceDb, quote_ident,
};
use crate::storage::writer::Claims;

/// One row per run. The definition columns say what the output's rows were
/// labelled under; the newest that counts is the definition in force
/// ([`ClassificationRun::in_force`]).
pub(crate) const DDL: &str = "CREATE TABLE IF NOT EXISTS _quack_classifications (
    id TEXT PRIMARY KEY,
    output_table TEXT NOT NULL,
    document_id TEXT NOT NULL,
    source_table TEXT NOT NULL,
    key_column TEXT NOT NULL,
    key_type TEXT NOT NULL,
    text_columns JSON NOT NULL,
    set_name TEXT NOT NULL,
    question_set JSON NOT NULL,
    model TEXT NOT NULL,
    model_digest TEXT NOT NULL,
    rows_scope TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    started_by TEXT,
    started_at TIMESTAMP NOT NULL DEFAULT now(),
    finished_at TIMESTAMP,
    labelled UBIGINT NOT NULL DEFAULT 0,
    cut UBIGINT NOT NULL DEFAULT 0,
    empty UBIGINT NOT NULL DEFAULT 0,
    skipped UBIGINT NOT NULL DEFAULT 0
);";

const COLUMNS: &str = "id, output_table, document_id, source_table, key_column, \
     CAST(text_columns AS VARCHAR), CAST(question_set AS VARCHAR), model, model_digest, \
     rows_scope, status, CAST(started_at AS VARCHAR), CAST(finished_at AS VARCHAR), \
     labelled, cut, empty, skipped, error, key_type";

/// The runs a workspace has recorded, newest first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub struct ClassificationRuns {
    pub runs: Vec<ClassificationRun>,
}

/// A row selected with `COLUMNS`.
impl TryFrom<&duckdb::Row<'_>> for ClassificationRun {
    type Error = Error;

    fn try_from(row: &duckdb::Row<'_>) -> Result<Self> {
        let text_columns: String = row.get(5)?;
        let question_set: String = row.get(6)?;
        Ok(Self {
            id: row.get(0)?,
            output_table: row.get(1)?,
            document_id: row.get(2)?,
            source_table: row.get(3)?,
            key_column: row.get(4)?,
            text_columns: serde_json::from_str(&text_columns)?,
            question_set: serde_json::from_str(&question_set)?,
            model: row.get(7)?,
            model_digest: row.get(8)?,
            rows: row.get(9)?,
            status: row.get(10)?,
            started_at: row.get(11)?,
            finished_at: row.get(12)?,
            labelled: row.get(13)?,
            cut: row.get(14)?,
            empty: row.get(15)?,
            skipped: row.get(16)?,
            error: row.get(17)?,
            key_type: row.get(18)?,
        })
    }
}

impl ClassificationRun {
    /// The run with this id.
    ///
    /// # Errors
    ///
    /// Returns `NotFound` when there is none, or a query error.
    pub fn get(db: &WorkspaceDb, id: &RunId) -> Result<Self> {
        let sql = format!("SELECT {COLUMNS} FROM _quack_classifications WHERE id = ?");
        let mut stmt = db.connection().prepare(&sql)?;
        let mut rows = stmt.query(duckdb::params![id])?;
        rows.next()?
            .map(Self::try_from)
            .transpose()?
            .ok_or_else(|| ResourceKind::ClassificationRun.missing(id.as_str()))
    }

    /// The newest `limit` runs, newest first.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list(db: &WorkspaceDb, limit: u32) -> Result<ClassificationRuns> {
        let sql = format!("SELECT {COLUMNS} FROM _quack_classifications ORDER BY id DESC LIMIT ?");
        let mut stmt = db.connection().prepare(&sql)?;
        let mut rows = stmt.query(duckdb::params![limit])?;
        let mut runs = Vec::new();
        while let Some(row) = rows.next()? {
            runs.push(Self::try_from(row)?);
        }
        Ok(ClassificationRuns { runs })
    }

    /// The definition the rows of `output` were labelled under: the newest
    /// run for it, only while the table's owner is the labels document that
    /// run made (a table that took the name after that document went is
    /// not labelled by it), leaving out a run that labels every row again and has
    /// not completed, since its rows are in a staging table that may never
    /// replace the output.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn in_force(db: &WorkspaceDb, output: &str) -> Result<Option<Self>> {
        let Some(owner) = db
            .table_owner(output)?
            .filter(|document| document.source == DocumentSource::Classify)
        else {
            return Ok(None);
        };
        let sql = format!(
            "SELECT {COLUMNS} FROM _quack_classifications \
             WHERE lower(output_table) = lower(?) AND document_id = ? \
             AND NOT (rows_scope = ? AND status <> ?) \
             ORDER BY id DESC LIMIT 1"
        );
        let mut stmt = db.connection().prepare(&sql)?;
        let mut rows = stmt.query(duckdb::params![
            output,
            owner.id,
            Rows::All,
            RunStatus::Completed
        ])?;
        rows.next()?.map(Self::try_from).transpose()
    }

    /// What this run's definition and `plan`'s differ in, by name. Table,
    /// column, and type names are compared without regard to case.
    pub(super) fn differs(&self, plan: &Plan, digest: &str) -> Vec<&'static str> {
        let mut differs = Vec::new();
        if !self.source_table.eq_ignore_ascii_case(plan.source.as_str()) {
            differs.push("source tables");
        }
        if !self.key_column.eq_ignore_ascii_case(&plan.key.name) {
            differs.push("keys");
        }
        if !self.key_type.eq_ignore_ascii_case(&plan.key.duckdb_type) {
            differs.push("key types");
        }
        let same_text = self.text_columns.len() == plan.text_columns.len()
            && self
                .text_columns
                .iter()
                .zip(&plan.text_columns)
                .all(|(recorded, planned)| recorded.eq_ignore_ascii_case(planned));
        if !same_text {
            differs.push("text columns");
        }
        if self.question_set.questions != plan.set.questions {
            differs.push("questions");
        }
        if self.model_digest != digest {
            differs.push("model weights");
        }
        differs
    }
}

/// What a run starts from.
pub(super) struct Start<'a> {
    pub plan: &'a Plan,
    pub run_id: &'a RunId,
    pub model: &'a str,
    pub digest: &'a str,
    pub started_by: Option<&'a str>,
    /// The runs this process holds. Read inside the writer step that marks
    /// the dead ones, so a run that claims and starts in between is seen.
    pub claims: &'a Claims,
}

impl Start<'_> {
    /// Begin the run in one transaction: mark the runs left running by a
    /// process that died `interrupted`, check the definition and the key,
    /// create the output (or the staging table that will replace it), and
    /// record the run as running.
    pub(super) fn apply(&self, db: &WorkspaceDb) -> Result<()> {
        let plan = self.plan;
        db.write_transaction(|db| {
            self.mark_dead_runs(db)?;
            db.connection().execute_batch(&format!(
                "DROP TABLE IF EXISTS {}",
                quote_ident(&plan.stage())
            ))?;
            let document = if plan.output_exists {
                self.check_existing(db)?;
                if plan.rows == Rows::All {
                    self.create_table(db, &plan.stage())?;
                }
                plan.document
                    .clone()
                    .ok_or_else(|| ClassifyError::OutputTaken {
                        table: plan.output.to_string(),
                    })?
            } else {
                if TableName::in_catalog(&db.list_tables()?, plan.output.as_str()).is_some() {
                    return Err(ClassifyError::OutputTaken {
                        table: plan.output.to_string(),
                    }
                    .into());
                }
                self.create_table(db, plan.output.as_str())?;
                match &plan.document {
                    Some(document) => document.clone(),
                    None => self.insert_document(db)?,
                }
            };
            self.insert_run(db, &document)
        })
    }

    /// Mark `interrupted` every run recorded as running that no run in this
    /// process holds: the claims are the registry of live runs, and this
    /// run's own claim is held, so an earlier run into its output is dead.
    fn mark_dead_runs(&self, db: &WorkspaceDb) -> Result<()> {
        let mut stmt = db
            .connection()
            .prepare("SELECT id, output_table FROM _quack_classifications WHERE status = ?")?;
        let (own, live) = (
            self.plan.output.as_str().to_ascii_lowercase(),
            self.claims.labelling(),
        );
        let running = stmt
            .query_map(duckdb::params![RunStatus::Running], |row| {
                Ok((row.get::<_, RunId>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<duckdb::Result<Vec<_>>>()?;
        for (id, output) in running {
            let output = output.to_ascii_lowercase();
            if output != own && live.contains(&output) {
                continue;
            }
            db.connection().execute(
                "UPDATE _quack_classifications SET status = ?, finished_at = now() WHERE id = ?",
                duckdb::params![RunStatus::Interrupted, id],
            )?;
        }
        Ok(())
    }

    /// An existing output must still be the plan's table of labels, with
    /// its key constraint, and a run that adds rows to it must run under the
    /// definition it was labelled under.
    fn check_existing(&self, db: &WorkspaceDb) -> Result<()> {
        let plan = self.plan;
        if !plan.still_owns_output(db)? {
            return Err(ClassifyError::OutputTaken {
                table: plan.output.to_string(),
            }
            .into());
        }
        let keyed: i64 = db.connection().query_row(
            "SELECT count(*) FROM duckdb_constraints() \
             WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'",
            duckdb::params![plan.output.as_str()],
            |row| row.get(0),
        )?;
        if keyed == 0 && plan.rows == Rows::Missing {
            return Err(ClassifyError::KeyLost {
                table: plan.output.to_string(),
                column: plan.key.name.clone(),
            }
            .into());
        }
        let in_force = ClassificationRun::in_force(db, plan.output.as_str())?;
        plan.check_definition(in_force.as_ref(), self.digest)?;
        Ok(())
    }

    /// An empty table of the labels' shape, keyed like the source: the key
    /// is selected from the source so its type and collation are the
    /// source's own.
    fn create_table(&self, db: &WorkspaceDb, name: &str) -> Result<()> {
        let plan = self.plan;
        let key = quote_ident(&plan.key.name);
        db.connection().execute_batch(&format!(
            "CREATE TABLE {table} AS SELECT s.{key}, {labels} FROM {source} s WHERE false; \
             ALTER TABLE {table} ADD PRIMARY KEY ({key})",
            table = quote_ident(name),
            labels = plan.columns.null_selects(),
            source = quote_ident(plan.source.as_str()),
        ))?;
        Ok(())
    }

    /// The document that owns the output table, so it is listed, can be
    /// deleted, and is dropped with its table.
    fn insert_document(&self, db: &WorkspaceDb) -> Result<DocumentId> {
        let plan = self.plan;
        let id = DocumentId::generate();
        let definition = serde_json::to_string(&plan.set)?;
        let title = format!("{} labelled by {}", plan.source, plan.set.name);
        db.insert_document(&NewDocument {
            id: &id,
            filename: plan.output.as_str(),
            title: Some(&title),
            mime_type: "application/x-quack-classification",
            size_bytes: definition.len(),
            sha256: &sha256_hex(definition.as_bytes()),
            source: DocumentSource::Classify,
            status: DocumentStatus::Ready,
            ingested_by: self.started_by,
            source_root: None,
            source_path: None,
        })?;
        db.set_document_tables(&id, &[plan.output.to_string()])?;
        Ok(id)
    }

    fn insert_run(&self, db: &WorkspaceDb, document: &DocumentId) -> Result<()> {
        let plan = self.plan;
        let scope = if plan.output_exists {
            plan.rows
        } else {
            Rows::Missing
        };
        db.connection().execute(
            "INSERT INTO _quack_classifications (id, output_table, document_id, source_table, \
             key_column, key_type, text_columns, set_name, question_set, model, model_digest, \
             rows_scope, status, started_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            duckdb::params![
                self.run_id,
                plan.output.as_str(),
                document,
                plan.source.as_str(),
                plan.key.name,
                plan.key.duckdb_type,
                serde_json::to_string(&plan.text_columns)?,
                plan.set.name.as_str(),
                serde_json::to_string(&plan.set)?,
                self.model,
                self.digest,
                scope,
                RunStatus::Running,
                self.started_by,
            ],
        )?;
        Ok(())
    }
}

/// How one row of a page came out.
#[derive(Debug)]
pub(super) enum Written {
    /// Labelled by the model, from text that was cut or not.
    Labelled { cut: bool },
    /// No text: NULL labels.
    Empty,
}

/// The rows of one page to add, and the rows of it not written.
pub(super) struct PageWrite {
    /// Each row's key as text and its label cells.
    pub rows: Vec<(String, Vec<Value>, Written)>,
    pub skipped: u64,
}

impl PageWrite {
    /// Add the page to `target` and its counts to the run's, in one
    /// transaction. A key the table holds already is left as it is and not
    /// counted. Returns the rows added.
    pub(super) fn write(
        self,
        db: &WorkspaceDb,
        plan: &Plan,
        run: &RunId,
        target: &str,
    ) -> Result<u64> {
        let key = quote_ident(&plan.key.name);
        let mut columns = vec![key];
        columns.extend(
            plan.columns
                .column_names()
                .iter()
                .skip(1)
                .map(|name| quote_ident(name)),
        );
        let placeholders = std::iter::once(plan.key_param())
            .chain(std::iter::repeat_n(
                String::from("?"),
                columns.len().saturating_sub(1),
            ))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "INSERT INTO {} ({}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
            quote_ident(target),
            columns.join(", ")
        );
        db.write_transaction(|db| {
            let mut stmt = db.connection().prepare(&sql)?;
            let (mut labelled, mut cut, mut empty) = (0_u64, 0_u64, 0_u64);
            for (key, cells, kind) in self.rows {
                let params = std::iter::once(Value::Text(key)).chain(cells);
                if stmt.execute(duckdb::params_from_iter(params))? == 0 {
                    continue;
                }
                match kind {
                    Written::Labelled { cut: true } => {
                        labelled = labelled.saturating_add(1);
                        cut = cut.saturating_add(1);
                    }
                    Written::Labelled { cut: false } => labelled = labelled.saturating_add(1),
                    Written::Empty => empty = empty.saturating_add(1),
                }
            }
            db.connection().execute(
                "UPDATE _quack_classifications SET labelled = labelled + ?, cut = cut + ?, \
                 empty = empty + ?, skipped = skipped + ? WHERE id = ?",
                duckdb::params![labelled, cut, empty, self.skipped, run],
            )?;
            Ok(labelled.saturating_add(empty))
        })
    }
}

/// How a run ended.
#[derive(Debug)]
pub(super) enum Ended {
    Completed,
    Stopped { status: RunStatus, error: String },
}

impl Ended {
    /// Record the end in one transaction: a completed run that labelled
    /// every row again swaps its staging table in for the output, and a
    /// stopped one drops it, leaving the old labels and their definition.
    pub(super) fn record(
        self,
        db: &WorkspaceDb,
        plan: &Plan,
        run: &RunId,
    ) -> Result<ClassificationRun> {
        let staged = plan.rows == Rows::All && plan.output_exists;
        let (mut status, mut error) = match self {
            Self::Completed => (RunStatus::Completed, None),
            Self::Stopped { status, error } => (status, Some(error)),
        };
        let (record, swap) = db.write_transaction(|db| {
            let mut swap = Swap::Discard;
            if staged {
                let (output, stage) = (
                    quote_ident(plan.output.as_str()),
                    quote_ident(&plan.stage()),
                );
                swap = if status == RunStatus::Completed {
                    Self::swap(db, plan)?
                } else {
                    Swap::Discard
                };
                match swap {
                    Swap::Replace => db.connection().execute_batch(&format!(
                        "DROP TABLE {output}; ALTER TABLE {stage} RENAME TO {output}"
                    ))?,
                    Swap::Install => db
                        .connection()
                        .execute_batch(&format!("ALTER TABLE {stage} RENAME TO {output}"))?,
                    Swap::Blocked | Swap::Discard => {
                        db.connection()
                            .execute_batch(&format!("DROP TABLE IF EXISTS {stage}"))?;
                        db.connection().execute(
                            "UPDATE _quack_classifications SET labelled = 0, cut = 0, empty = 0 \
                             WHERE id = ?",
                            duckdb::params![run],
                        )?;
                        if swap == Swap::Blocked {
                            status = RunStatus::Failed;
                            error = Some(
                                ClassifyError::OutputTaken {
                                    table: plan.output.to_string(),
                                }
                                .to_string(),
                            );
                        }
                    }
                }
            }
            db.connection().execute(
                "UPDATE _quack_classifications SET status = ?, error = ?, finished_at = now() \
                 WHERE id = ?",
                duckdb::params![status, error, run],
            )?;
            Ok((ClassificationRun::get(db, run)?, swap))
        })?;
        if swap == Swap::Blocked {
            return Err(ClassifyError::OutputTaken {
                table: plan.output.to_string(),
            }
            .into());
        }
        Ok(record)
    }

    /// What may become of a staging table whose run completed: it replaces
    /// the output while the output is still this run's labels, takes its
    /// place when the output is gone, and is dropped when something else
    /// holds the name now.
    fn swap(db: &WorkspaceDb, plan: &Plan) -> Result<Swap> {
        if TableName::in_catalog(&db.list_tables()?, plan.output.as_str()).is_none() {
            return Ok(Swap::Install);
        }
        Ok(if plan.still_owns_output(db)? {
            Swap::Replace
        } else {
            Swap::Blocked
        })
    }
}

/// What a completed run that labelled every row again does with its
/// staging table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Swap {
    Replace,
    Install,
    Blocked,
    /// The run did not complete.
    Discard,
}
