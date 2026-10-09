//! Labelling a table's text with a decision model (issue #472).
//!
//! A [`Classification`] names a table, the text columns of its rows, and a
//! [`QuestionSet`]. A run asks the decision model the questions about every
//! row and writes the answers to a new table with one row per source row,
//! joined back by the source's key, so a person can `GROUP BY` and filter
//! on them.
//!
//! Rows are read a page at a time by keyset on the key, never by `OFFSET`,
//! and the answers are written one page per transaction, so a cancel or a
//! failure keeps every row already labelled. A run labels the rows whose
//! key the output does not hold yet; [`Rows::All`] labels every row again
//! into a hidden staging table that replaces the output once it is
//! complete, so the old labels serve until then. Rows whose text changed
//! after they were labelled are not relabelled by a plain run.
//!
//! Each run is recorded in `_quack_classifications` with the definition it
//! ran under; the definition in force for an output decides whether a later
//! run may add rows to it.

mod columns;
mod pipeline;
mod plan;
mod store;
#[cfg(test)]
mod tests;

use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

pub use pipeline::Labelling;
pub use store::ClassificationRuns;
pub(crate) use store::DDL;

use crate::error::{Error, Result};
use crate::ids::{DocumentId, RunId, UserId, WorkspaceId};
use crate::jobs::{JobKind, JobQueue, JobSpec};
use crate::llm::decision::{DecisionModel, QuestionName, QuestionSetError, Questions};
use crate::priority::Priority;
use crate::progress::{ChunkDone, RunControl};
use crate::storage::control::Outcome;
use crate::storage::workspace::QueryResults;
use crate::storage::writer::Writer;
use crate::text::OneLine;

/// The most bytes of a question set a request carries: far above what 64
/// questions of 26 options need, and small enough to read in no time.
pub const MAX_QUESTION_SET_BYTES: usize = 64 * 1024;

/// Text columns one classification reads.
const TEXT_COLUMNS: usize = 8;

/// The name of a question set. It becomes part of the output table's name,
/// so it follows the rule of a question's name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SetName(QuestionName);

impl SetName {
    /// The name as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl TryFrom<String> for SetName {
    type Error = QuestionSetError;

    fn try_from(name: String) -> std::result::Result<Self, Self::Error> {
        QuestionName::try_from(name).map(Self)
    }
}

impl From<SetName> for String {
    fn from(name: SetName) -> Self {
        name.0.into()
    }
}

impl fmt::Display for SetName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A named list of decision questions: the JSON file `quack classify
/// --questions` reads, and the form the REST route and the MCP tool take.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QuestionSet {
    #[schema(value_type = String)]
    #[schemars(with = "String")]
    pub name: SetName,
    pub questions: Questions,
}

impl QuestionSet {
    /// Read a question set from JSON text: the file `quack classify
    /// --questions` takes and the Tables page's field.
    ///
    /// # Errors
    ///
    /// Returns [`QuestionSetError::SetTooLarge`] for text past
    /// [`MAX_QUESTION_SET_BYTES`], which is refused before it is parsed,
    /// and [`QuestionSetError::Unreadable`] or the error of the set for
    /// text that is not a valid set.
    pub fn parse(text: &str) -> std::result::Result<Self, QuestionSetError> {
        if text.len() > MAX_QUESTION_SET_BYTES {
            return Err(QuestionSetError::SetTooLarge {
                bytes: text.len(),
                max: MAX_QUESTION_SET_BYTES,
            });
        }
        serde_json::from_str(text).map_err(|e| QuestionSetError::Unreadable(e.to_string()))
    }
}

/// Which rows a run labels.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum Rows {
    /// Rows whose key the output does not hold yet.
    #[default]
    Missing,
    /// Every row, replacing the output's labels when the run completes.
    All,
}

text_enum!(Rows, "rows", {
    Missing => "missing",
    All => "all",
});
text_enum_sql!(Rows);

/// What to label and how: the one request every interface sends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, JsonSchema)]
pub struct Classification {
    /// The table whose rows are labelled, by its exact name.
    pub table: String,
    /// The columns whose text the model reads, one to eight.
    pub text_columns: Vec<String>,
    /// The column that identifies a row, joining the output back to the
    /// table. Unset, an `id` column or one ending in `_id` whose values are
    /// all present and different is used.
    #[serde(default)]
    pub key: Option<String>,
    /// The questions asked about every row.
    pub question_set: QuestionSet,
    /// Which rows to label; unset, those the output lacks.
    #[serde(default)]
    pub rows: Rows,
}

/// Whether the caller waits for the run, and how much it will wait for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waiting {
    /// A background job: any size.
    Job,
    /// A caller that waits for the answer (an agent turn, an MCP call): the
    /// run is refused when its rows times its questions exceed `budget`
    /// (`[decision].interactive_budget`).
    Caller { budget: u64 },
}

/// What a run would do to the output table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    /// The output does not exist yet.
    NewTable,
    /// The output exists and the run adds the rows it lacks.
    AddsRows,
    /// The output exists and the run replaces its labels when it completes.
    ReplacesLabels,
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NewTable => "new table",
            Self::AddsRows => "adds rows",
            Self::ReplacesLabels => "replaces its labels",
        })
    }
}

/// What a run would do, worked out before it starts: how much it labels,
/// where, and with which questions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub struct ClassificationOutline {
    pub source_table: String,
    pub output_table: String,
    pub key_column: String,
    /// Rows a run would label now.
    pub remaining: u64,
    /// The questions, in order.
    pub questions: Vec<OutlineQuestion>,
    pub effect: Effect,
}

/// One question of an outline: its name and what it asks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub struct OutlineQuestion {
    pub name: String,
    pub instructions: String,
}

impl ClassificationOutline {
    /// Whether a caller that `waiting` allows may wait for the run.
    ///
    /// # Errors
    ///
    /// Returns [`ClassifyError::TooLargeToWait`] when the rows times the
    /// questions are more answers than the caller waits for.
    pub fn within(&self, waiting: Waiting) -> std::result::Result<(), ClassifyError> {
        let Waiting::Caller { budget } = waiting else {
            return Ok(());
        };
        let answers = self
            .remaining
            .saturating_mul(u64::try_from(self.questions.len()).unwrap_or(u64::MAX));
        if answers > budget {
            return Err(ClassifyError::TooLargeToWait {
                source_table: self.source_table.clone(),
                rows: self.remaining,
                questions: self.questions.len(),
                budget,
            });
        }
        Ok(())
    }

    /// The question names, comma-separated.
    #[must_use]
    pub fn question_names(&self) -> String {
        self.questions
            .iter()
            .map(|q| q.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The statement a person is asked to allow: comment lines saying what
    /// the run does and what each question asks, as the permission prompt
    /// shows them. Each name and instruction is one line, so text a table
    /// or a question file supplied cannot add lines of its own.
    #[must_use]
    pub fn statement(&self, model: &str) -> String {
        let mut lines = vec![format!(
            "-- label {} rows of {} into {} ({}) with {model}",
            self.remaining,
            OneLine(&self.source_table),
            OneLine(&self.output_table),
            self.effect,
        )];
        for question in &self.questions {
            lines.push(format!(
                "-- {}: {}",
                OneLine(&question.name),
                OneLine(&question.instructions)
            ));
        }
        lines.join("\n")
    }
}

impl fmt::Display for ClassificationOutline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} rows of {} (key {}) into {} ({}); {} questions: {}",
            self.remaining,
            self.source_table,
            self.key_column,
            self.output_table,
            self.effect,
            self.questions.len(),
            self.question_names()
        )
    }
}

/// The arguments of the agent's `classify_rows` tool and of the MCP
/// `classify` tool: a labelling, or with `preview` its first rows.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
pub struct ClassifyArgs {
    /// What to label and how.
    #[serde(flatten)]
    pub classification: Classification,
    /// Label only this many rows (1 to 100) and show them, writing nothing
    #[serde(default)]
    pub preview: Option<u32>,
}

/// A column that identifies a row of the source table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KeyColumn {
    pub name: String,
    /// The column's `DuckDB` type, as the catalog spells it.
    pub duckdb_type: String,
}

/// A column that comes close to identifying a row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub struct KeyCandidate {
    pub column: String,
    pub distinct: u64,
    pub missing: u64,
}

impl fmt::Display for KeyCandidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({} distinct, {} missing)",
            self.column, self.distinct, self.missing
        )
    }
}

/// Why a table's rows cannot be labelled as asked.
#[derive(Debug, thiserror::Error)]
pub enum ClassifyError {
    #[error("no table named '{0}'")]
    NoTable(String),
    #[error("table '{table}' has no column '{column}'")]
    NoColumn { table: String, column: String },
    #[error("a classification reads one to {TEXT_COLUMNS} text columns, not {0}")]
    TextColumns(usize),
    #[error("the text column '{0}' is named twice")]
    DuplicateColumn(String),
    #[error("{}", NoKey(.table, .unique, .closest))]
    NoKey {
        table: String,
        unique: Vec<String>,
        closest: Vec<KeyCandidate>,
    },
    #[error(
        "'{column}' cannot be the key of {table}: {rows} rows, {distinct} different values, \
         {missing} missing; a key has one different value in every row"
    )]
    KeyNotUnique {
        table: String,
        column: String,
        rows: u64,
        distinct: u64,
        missing: u64,
    },
    #[error(
        "'{column}' ({duckdb_type}) cannot be a key: its values do not read back from text \
         unchanged; choose another column"
    )]
    KeyType { column: String, duckdb_type: String },
    #[error(
        "the key of {table} ('{column}') no longer has its primary key constraint, as a rebuild \
         of the table with CREATE OR REPLACE drops it; label every row again with --all (\"rows\": \
         \"all\")"
    )]
    KeyLost { table: String, column: String },
    #[error(
        "two columns of the labels table would be named '{0}' (names are compared without regard \
         to case); rename a question"
    )]
    ColumnClash(String),
    #[error(
        "'{table}' exists and is not a table of labels; choose another name for the question set"
    )]
    OutputTaken { table: String },
    #[error(
        "{table} was labelled with other {}; label every row again with --all (\"rows\": \"all\")",
        .differs.join(" and ")
    )]
    DefinitionChanged {
        table: String,
        differs: Vec<&'static str>,
    },
    #[error(
        "{table} is being labelled by another run; wait for it or cancel it (quack jobs, /jobs, or \
         the Jobs page)"
    )]
    Running { table: String },
    #[error(
        "labelling {rows} rows of {source_table} with {questions} questions is too long to wait \
         for here (at most {budget} answers, rows times questions, [decision].interactive_budget); \
         run it as a job with quack classify or POST /api/v1/workspaces/{{id}}/tables/classify, \
         or preview fewer rows here"
    )]
    TooLargeToWait {
        source_table: String,
        rows: u64,
        questions: usize,
        budget: u64,
    },
    #[error("a labels table is relabelled with quack classify --all, not replaced by a file")]
    ReplaceRefused,
}

/// The text of [`ClassifyError::NoKey`].
struct NoKey<'a>(&'a str, &'a [String], &'a [KeyCandidate]);

impl fmt::Display for NoKey<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(table, unique, closest) = self;
        write!(
            f,
            "{table} has no id column (named id, or ending in _id or Id) whose values are all \
             present and all different, so a label cannot be joined back to its row."
        )?;
        if !unique.is_empty() {
            write!(
                f,
                " Unique columns that could serve: {}.",
                unique.join(", ")
            )?;
        }
        if !closest.is_empty() {
            let closest: Vec<String> = closest.iter().map(ToString::to_string).collect();
            write!(f, " Closest to unique otherwise: {}.", closest.join(", "))?;
        }
        write!(
            f,
            " Choose the key (--key on the command line), or add one: CREATE TABLE {table}_keyed \
             AS SELECT row_number() OVER () AS row_id, * FROM {table}"
        )
    }
}

/// How far a run came.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Running,
    Completed,
    /// Stopped by a cancel; the rows labelled before stay.
    Cancelled,
    /// Stopped by an error; the rows labelled before stay.
    Failed,
    /// Found still running when the next run started: the process died.
    Interrupted,
}

text_enum!(RunStatus, "classification status", {
    Running => "running",
    Completed => "completed",
    Cancelled => "cancelled",
    Failed => "failed",
    Interrupted => "interrupted",
});
text_enum_sql!(RunStatus);

/// One run of a [`Classification`], as `_quack_classifications` records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub struct ClassificationRun {
    pub id: RunId,
    /// The table of labels.
    pub output_table: String,
    /// The document that owns the output table.
    pub document_id: DocumentId,
    pub source_table: String,
    pub key_column: String,
    /// The key's `DuckDB` type when the run began: an output labelled under
    /// another type is labelled again, not added to.
    pub key_type: String,
    pub text_columns: Vec<String>,
    pub question_set: QuestionSet,
    /// The model, as `provider/model`.
    pub model: String,
    /// The digest of the model's weights when the run began.
    pub model_digest: String,
    /// Which rows the run labelled. A first run on a missing output is
    /// `missing` whatever was asked.
    pub rows: Rows,
    pub status: RunStatus,
    /// Rows labelled by the model.
    pub labelled: u64,
    /// Rows labelled from text cut to fit the model.
    pub cut: u64,
    /// Rows with no text, written with NULL labels.
    pub empty: u64,
    /// Rows the model refused at every length, not written; the next run
    /// tries them again.
    pub skipped: u64,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub error: Option<String>,
}

impl fmt::Display for ClassificationRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            output_table,
            status,
            labelled,
            cut,
            empty,
            skipped,
            ..
        } = self;
        match status {
            RunStatus::Completed => write!(f, "Wrote {output_table}: ")?,
            RunStatus::Running
            | RunStatus::Cancelled
            | RunStatus::Failed
            | RunStatus::Interrupted => write!(f, "{output_table} ({status}): ")?,
        }
        write!(
            f,
            "{labelled} rows labelled, {cut} cut to fit the model, {empty} empty, {skipped} \
             skipped"
        )?;
        if *skipped > 0 {
            f.write_str(
                " (the model refused their text at every length; the next run tries them again)",
            )?;
        }
        f.write_str(".")
    }
}

/// The first rows of a run, labelled and not written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub struct ClassificationPreview {
    pub key_column: String,
    pub output_table: String,
    pub result: QueryResults,
    pub labelled: u32,
    pub cut: u32,
    pub empty: u32,
    pub skipped: u32,
    pub took_ms: u64,
    /// Rows a run would label now.
    pub remaining: u64,
}

impl fmt::Display for ClassificationPreview {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let took = Duration::from_millis(self.took_ms);
        write!(
            f,
            "{} rows labelled in {:.1} s; {} cut to fit the model, {} empty, {} skipped. Run \
             without --preview to label {} rows.",
            self.labelled.saturating_add(self.empty),
            took.as_secs_f64(),
            self.cut,
            self.empty,
            self.skipped,
            self.remaining
        )
    }
}

impl ClassificationRun {
    /// What each column of the output means, the key first: the sentences
    /// `describe_table` gives the agent and the Tables page.
    #[must_use]
    pub fn column_meanings(&self) -> Vec<(String, String)> {
        columns::OutputColumns::new(
            &self.source_table,
            &self.key_column,
            &self.question_set.questions,
        )
        .map(|columns| columns.meanings())
        .unwrap_or_default()
    }
}

/// How a run that started ended, as an interface audits it.
#[derive(Debug, Clone)]
pub struct RunEnded {
    /// The run, as `_quack_classifications` holds it after the end.
    pub run: ClassificationRun,
    /// `Allowed` when the run completed, and why not otherwise.
    pub outcome: Outcome,
}

/// What an interface does when a run it follows has ended. It runs inside
/// the job, so it happens even when nobody waits for the run any more.
#[derive(Clone)]
pub struct OnEnd(Arc<dyn Fn(RunEnded) -> BoxFuture<'static, ()> + Send + Sync>);

impl OnEnd {
    /// A hook that runs `hook` with how the run ended.
    pub fn new<F, Fut>(hook: F) -> Self
    where
        F: Fn(RunEnded) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        Self(Arc::new(move |ended| Box::pin(hook(ended))))
    }

    /// Report how the run `id` ended with `result`. A run that never began
    /// (its request was refused) has no record, and nothing to report.
    async fn report(&self, db: &Writer, id: &RunId, result: &Result<ClassificationRun>) {
        let run = if let Ok(run) = result {
            Some(run.clone())
        } else {
            let id = id.clone();
            db.run(move |db| ClassificationRun::get(db, &id)).await.ok()
        };
        let Some(run) = run else {
            return;
        };
        let outcome = result
            .as_ref()
            .map_or_else(Outcome::of_failure, |_| Outcome::Allowed);
        (self.0)(RunEnded { run, outcome }).await;
    }
}

/// The job queue the runs of callers that wait go through, so a run is
/// listed with the other jobs, stopped by the queue's shutdown, and
/// finished (and reported) even when the caller is gone.
#[derive(Clone)]
pub struct LabelJobs {
    queue: JobQueue,
    workspace: Option<WorkspaceId>,
    on_end: Option<OnEnd>,
}

impl LabelJobs {
    /// Runs go through `queue`.
    #[must_use]
    pub const fn new(queue: JobQueue) -> Self {
        Self {
            queue,
            workspace: None,
            on_end: None,
        }
    }

    /// The jobs belong to this workspace in the queue's listings.
    #[must_use]
    pub fn workspace(mut self, workspace: WorkspaceId) -> Self {
        self.workspace = Some(workspace);
        self
    }

    /// Call `hook` when a run that began has ended.
    #[must_use]
    pub fn on_end(mut self, hook: OnEnd) -> Self {
        self.on_end = Some(hook);
        self
    }
}

/// A run for a caller that waits (an agent turn, an MCP call).
pub struct Tracked {
    /// The workspace's writer.
    pub db: Arc<Writer>,
    /// The model that labels.
    pub decision: DecisionModel,
    /// Who started the run, recorded on it and as the job's owner.
    pub started_by: Option<UserId>,
    /// The id the run is recorded under.
    pub run_id: RunId,
    /// How much the caller waits for.
    pub waiting: Waiting,
    /// Stops the run when the caller is cancelled; the run then records
    /// that it was cancelled.
    pub cancel: CancellationToken,
    /// The queue the run is a job of.
    pub jobs: LabelJobs,
}

impl Tracked {
    /// Submit the run of `request` and wait for its result.
    async fn submit(
        self,
        request: Classification,
        progress: impl Fn(ChunkDone) + Send + Sync + 'static,
    ) -> Result<ClassificationRun> {
        let Self {
            db,
            decision,
            started_by,
            run_id,
            waiting,
            cancel,
            jobs,
        } = self;
        let LabelJobs {
            queue,
            workspace,
            on_end,
        } = jobs;
        let label = format!("label {} by {}", request.table, request.question_set.name);
        let mut spec = JobSpec::new(JobKind::Classify, label).owner(started_by.clone());
        if let Some(workspace) = workspace {
            spec = spec.workspace(workspace);
        }
        let (sender, receiver) = oneshot::channel();
        queue.submit(spec, move |ctx| async move {
            let (run_cancel, job_cancel) = (cancel.child_token(), ctx.cancel_token());
            let report = |done: ChunkDone| {
                ctx.progress(done.done, done.total);
                progress(done);
            };
            let started_by_text = started_by.as_ref().map(ToString::to_string);
            let priority = match waiting {
                Waiting::Caller { .. } => Priority::Interactive,
                Waiting::Job => Priority::current(),
            };
            let mut run = Box::pin(priority.scope(request.run(Labelling {
                db: &db,
                decision: &decision,
                started_by: started_by_text.as_deref(),
                run_id: run_id.clone(),
                waiting,
                control: RunControl {
                    progress: &report,
                    cancel: Some(&run_cancel),
                },
            })));
            let result = loop {
                tokio::select! {
                    result = &mut run => break result,
                    () = job_cancel.cancelled(), if !run_cancel.is_cancelled() => {
                        run_cancel.cancel();
                    }
                }
            };
            if let Some(on_end) = &on_end {
                on_end.report(&db, &run_id, &result).await;
            }
            let summary = result
                .as_ref()
                .map(ToString::to_string)
                .map_err(ToString::to_string);
            drop(sender.send(result));
            summary
        });
        receiver.await.map_err(|_| {
            Error::Analysis(String::from(
                "the labelling job did not run to its end (the server is stopping)",
            ))
        })?
    }
}

impl Classification {
    /// [`Self::run`] as a job of the queue, awaited by the caller. If the
    /// caller is dropped (a cancelled turn, an MCP client that left) the job
    /// goes on to its end and records it, and the queue's shutdown waits
    /// for it.
    ///
    /// # Errors
    ///
    /// As [`Self::run`], or [`Error::Analysis`] when the job did not run to
    /// its end (the queue was shutting down).
    pub async fn run_as_job(
        self,
        tracked: Tracked,
        progress: impl Fn(ChunkDone) + Send + Sync + 'static,
    ) -> Result<ClassificationRun> {
        tracked.submit(self, progress).await
    }

    /// What a run would do, without doing it: the checks of a run, even
    /// that of the definition an existing output was labelled under, and
    /// its row count, read from the workspace.
    ///
    /// # Errors
    ///
    /// Returns a [`ClassifyError`] when the request cannot be carried out,
    /// or the error of the model's server asked for its weights' digest.
    pub async fn outline(
        &self,
        writer: &Writer,
        decision: &DecisionModel,
    ) -> Result<ClassificationOutline> {
        pipeline::outline(self, writer, decision).await
    }

    /// Label the first `n` rows (1 to 100) by key and return them, writing
    /// nothing: no table, no record, no claim.
    ///
    /// # Errors
    ///
    /// Returns a [`ClassifyError`] when the request cannot be carried out,
    /// [`crate::error::Error::DecisionRefused`] when the model refuses the
    /// question set, or the model's or the database's error.
    pub async fn preview(
        &self,
        writer: &Writer,
        decision: &DecisionModel,
        n: u32,
        waiting: Waiting,
        control: RunControl<'_>,
    ) -> Result<ClassificationPreview> {
        pipeline::preview(self, writer, decision, n, waiting, control).await
    }

    /// Label the table's rows.
    ///
    /// # Errors
    ///
    /// Returns a [`ClassifyError`] when the request cannot be carried out,
    /// [`crate::error::Error::Cancelled`] after a cancel, or the model's or
    /// the database's error. The rows labelled before a stop stay.
    pub async fn run(&self, labelling: Labelling<'_>) -> Result<ClassificationRun> {
        pipeline::run(self, labelling).await
    }
}
