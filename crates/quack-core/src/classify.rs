//! Labelling a table's text with a decision model (issue #472).
//!
//! A person names a table and says in a sentence what they want to know
//! about each row. The chat model drafts the questions ([`Draft`]); a person
//! approves them by starting a run, and the run records them as the labels'
//! [`LabelSet`]. A run asks the decision model the questions about every row
//! and writes the answers to a new table with one row per source row,
//! joined back by the source's key, so a person can `GROUP BY` and filter
//! on them.
//!
//! Rows are read a page at a time by keyset on the key, never by `OFFSET`,
//! and the answers are written one page per transaction, so a cancel or a
//! failure keeps every row already labelled. A run labels the rows whose
//! key the output does not hold yet; a run that labels every row again
//! writes into a hidden staging table that replaces the output once it is
//! complete, so the old labels serve until then. A run does that by itself
//! when the questions, the key, or the model's weights are not those the
//! labels were made with. Rows whose text changed after they were labelled
//! are not relabelled by a plain run.
//!
//! Each run is recorded in `_quack_classifications` with the set it ran
//! under; the set in force for an output decides whether a later run adds
//! rows to it or labels every row again.

mod columns;
mod draft;
mod drafting;
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

pub use draft::{
    Draft, DraftAnswer, DraftOption, DraftOrigin, DraftQuestion, KeyReason, LabelSet, QuestionKind,
};
pub use drafting::{DraftContext, Drafter, PROMPT as DRAFT_PROMPT, SAMPLE_ROWS};
pub use pipeline::Labelling;
pub(crate) use store::DDL;
pub use store::Runs;

use crate::error::{Error as CoreError, Result};
use crate::ids::{DocumentId, RunId, UserId, WorkspaceId};
use crate::jobs::{JobKind, JobQueue, JobSpec};
use crate::llm::decision::{DecisionModel, Questions};
use crate::priority::Priority;
use crate::progress::{ChunkDone, RunControl};
use crate::storage::control::Outcome;
use crate::storage::workspace::QueryResults;
use crate::storage::writer::Writer;
use crate::text::{OneLine, Thousands};

/// Text columns one set reads.
pub(crate) const TEXT_COLUMNS: usize = 8;

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

/// What to label and how: the request REST and MCP take.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, JsonSchema)]
pub struct Request {
    /// The table whose rows are labelled, by its exact name.
    pub table: String,
    /// What the person wants to know about each row. The chat model drafts
    /// questions from it, or revises the last approved ones; the same
    /// sentence as last time uses the last approved questions.
    #[serde(default)]
    pub sentence: Option<String>,
    /// The questions to ask, as a preview returned them. They win over a
    /// sentence; with neither, the last approved questions are used.
    #[serde(default)]
    pub set: Option<LabelSet>,
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

/// Why a run labels every row again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum RelabelReason {
    /// The person asked for it.
    Asked,
    /// The questions, or the text columns they read, are not those the
    /// labels were made with.
    QuestionsChanged,
    /// The decision model's weights changed (`ollama pull`).
    ModelChanged,
    /// The key, or its type, changed.
    KeyChanged,
}

impl fmt::Display for RelabelReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Asked => "every row was asked for",
            Self::QuestionsChanged => "the questions changed",
            Self::ModelChanged => "the decision model's weights changed (ollama pull)",
            Self::KeyChanged => "the key changed",
        })
    }
}

/// What a run would do to the output table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Effect {
    /// The output does not exist yet.
    NewTable,
    /// The output exists and the run adds the rows it lacks.
    AddsRows,
    /// The output exists and the run replaces its labels when it completes.
    ReplacesLabels { because: RelabelReason },
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NewTable => f.write_str("new table"),
            Self::AddsRows => f.write_str("adds rows"),
            Self::ReplacesLabels { because } => {
                write!(f, "replaces its labels: {because}")
            }
        }
    }
}

/// How long a run takes, in words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Estimate(pub u64);

impl Estimate {
    /// The words with a capital first letter, to start a sentence.
    #[must_use]
    pub fn sentence(self) -> String {
        let words = self.to_string();
        let mut chars = words.chars();
        chars
            .next()
            .map(|first| first.to_uppercase().chain(chars).collect())
            .unwrap_or_default()
    }
}

impl fmt::Display for Estimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let minutes = self.0.saturating_add(30).saturating_div(60);
        if self.0 < 45 {
            f.write_str("under a minute")
        } else if minutes < 120 {
            write!(
                f,
                "about {} minute{}",
                minutes.max(1),
                if minutes.max(1) == 1 { "" } else { "s" }
            )
        } else {
            write!(
                f,
                "about {} hours",
                minutes.saturating_add(30).saturating_div(60)
            )
        }
    }
}

/// What a run would do, worked out before it starts: how much it labels,
/// where, and with which questions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub struct Outline {
    pub source_table: String,
    pub output_table: String,
    pub key_column: String,
    pub key_reason: KeyReason,
    pub text_columns: Vec<String>,
    /// Rows a run would label now.
    pub remaining: u64,
    /// The questions, in order.
    pub questions: Vec<DraftQuestion>,
    pub effect: Effect,
    /// About how long the run takes, from earlier runs of these questions
    /// or from the preview; none where nothing measured it.
    pub estimate_seconds: Option<u64>,
}

impl Outline {
    /// Whether a caller that `waiting` allows may wait for the run.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooLargeToWait`] when the rows times the
    /// questions are more answers than the caller waits for.
    pub fn within(&self, waiting: Waiting) -> std::result::Result<(), Error> {
        let Waiting::Caller { budget } = waiting else {
            return Ok(());
        };
        let answers = self
            .remaining
            .saturating_mul(u64::try_from(self.questions.len()).unwrap_or(u64::MAX));
        if answers > budget {
            return Err(Error::TooLargeToWait {
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

    /// The estimate in words, when there is one.
    #[must_use]
    pub fn estimate(&self) -> Option<Estimate> {
        self.estimate_seconds.map(Estimate)
    }

    /// The outline with an estimate worked out from a preview of `rows`
    /// rows that took `ask_ms` to ask, when nothing measured one before.
    #[must_use]
    pub fn estimated_from(mut self, preview: &Preview) -> Self {
        if self.estimate_seconds.is_none() {
            self.estimate_seconds = preview.seconds_for(self.remaining);
        }
        self
    }

    /// The statement a person is asked to allow: comment lines saying what
    /// the run does, which columns it reads, and what each question asks,
    /// as the permission prompt shows them. Each name and instruction is
    /// one line, so text a table or a person supplied cannot add lines of
    /// its own.
    #[must_use]
    pub fn statement(&self, model: &str) -> String {
        let estimate = self
            .estimate()
            .map(|estimate| format!(", {estimate}"))
            .unwrap_or_default();
        let mut lines = vec![
            format!(
                "-- label {} rows of {} into {} ({}) with {model}, {} questions{estimate}",
                Thousands(self.remaining),
                OneLine(&self.source_table),
                OneLine(&self.output_table),
                self.effect,
                self.questions.len(),
            ),
            format!(
                "-- key {}{}; reads {}",
                OneLine(&self.key_column),
                self.key_reason
                    .note()
                    .map(|note| format!(" ({note})"))
                    .unwrap_or_default(),
                self.text_columns
                    .iter()
                    .map(|c| OneLine(c).to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ];
        lines.extend(self.questions.iter().map(DraftQuestion::card_line));
        lines.join("\n")
    }
}

impl fmt::Display for Outline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} rows of {} (key {}) into {} ({}); {} questions: {}",
            Thousands(self.remaining),
            self.source_table,
            self.key_column,
            self.output_table,
            self.effect,
            self.questions.len(),
            self.question_names()
        )
    }
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
pub enum Error {
    #[error("no table named '{0}'")]
    NoTable(String),
    #[error(
        "{table} has no questions yet; say what you want to know about each row: quack classify \
         {table} \"which department should handle each ticket\""
    )]
    NoQuestions { table: String },
    #[error(
        "drafting questions from a sentence needs a chat model; set [general].chat_model = \
         \"PROVIDER/MODEL\" (quack init sets one up). A table with approved questions is labelled \
         again without one: quack classify {table}"
    )]
    NoDrafter { table: String },
    #[error(
        "{table} has no text column to read (columns of numbers, dates, ids, or mostly empty \
         values are left out)"
    )]
    NoText { table: String },
    #[error(
        "the chat model's questions were refused twice ({reason}); say it another way, or name \
         the answers you want: quack classify {table} \"department: billing, technical, sales, \
         or other\""
    )]
    DraftRefused { table: String, reason: String },
    #[error(
        "the questions read {}, which {table} no longer has; say a sentence to draft new \
         questions",
        .columns.join(", ")
    )]
    SetColumnsGone { table: String, columns: Vec<String> },
    #[error("a set reads one to {TEXT_COLUMNS} text columns, not {0}")]
    TextColumns(usize),
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
    #[error("'{table}' exists and is not a table of labels")]
    OutputTaken { table: String },
    #[error(
        "{table} is being labelled by another run; wait for it or cancel it (quack jobs, /jobs, or \
         the Jobs page)"
    )]
    Running { table: String },
    #[error(
        "labelling {rows} rows of {source_table} with {questions} questions is too long to wait \
         for here (at most {budget} answers, rows times questions, [decision].interactive_budget); \
         run it as a job with quack classify or POST /api/v1/workspaces/{{id}}/tables/classify, \
         or preview fewer rows here. Nothing was labelled, and questions drafted in this call \
         were not kept"
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

/// The text of [`Error::NoKey`].
struct NoKey<'a>(&'a str, &'a [String], &'a [KeyCandidate]);

impl fmt::Display for NoKey<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(table, unique, closest) = self;
        write!(
            f,
            "{table} has no column whose values are all present and all different (an id, a \
             number, or short text), so a label cannot be joined back to its row."
        )?;
        if !unique.is_empty() {
            write!(
                f,
                " Unique columns of a type that cannot be a key: {}.",
                unique.join(", ")
            )?;
        }
        if !closest.is_empty() {
            let closest: Vec<String> = closest.iter().map(ToString::to_string).collect();
            write!(f, " Closest to unique otherwise: {}.", closest.join(", "))?;
        }
        write!(
            f,
            " Add one: CREATE TABLE {table}_keyed AS SELECT row_number() OVER () AS row_id, * \
             FROM {table}"
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

/// One run, as `_quack_classifications` records it. The set it ran under
/// is the approval: the newest run for a table is the last approved set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub struct Run {
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
    pub key_reason: KeyReason,
    pub text_columns: Vec<String>,
    pub questions: Questions,
    /// What the person asked for, when the questions came from a sentence.
    pub sentence: Option<String>,
    /// The chat model that drafted the questions, when one did.
    pub drafted_by_model: Option<String>,
    /// The decision model, as `provider/model`.
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
    /// Milliseconds spent on the requests to the decision model.
    pub ask_ms: u64,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub error: Option<String>,
}

impl fmt::Display for Run {
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
            "{} rows labelled, {} cut to fit the model, {} empty, {} skipped",
            Thousands(*labelled),
            Thousands(*cut),
            Thousands(*empty),
            Thousands(*skipped)
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
pub struct Preview {
    pub key_column: String,
    pub output_table: String,
    /// The rows as the table of labels would hold them.
    pub result: QueryResults,
    /// The same rows as a screen shows them: a choice as `label (p)`, a
    /// score as its level, a yes/no as its probability, and a `*` after the
    /// key of a row whose text was cut.
    pub compact: QueryResults,
    pub labelled: u32,
    pub cut: u32,
    pub empty: u32,
    pub skipped: u32,
    pub took_ms: u64,
    /// Milliseconds spent on the requests to the decision model alone.
    pub ask_ms: u64,
    /// Rows a run would label now.
    pub remaining: u64,
}

impl Preview {
    /// About how many seconds `rows` rows take, at the rate this preview
    /// asked its own; none when it asked nothing.
    #[must_use]
    pub fn seconds_for(&self, rows: u64) -> Option<u64> {
        let asked = u64::from(self.labelled).saturating_add(u64::from(self.skipped));
        let per_row_ms = self.ask_ms.checked_div(asked)?;
        Some(per_row_ms.saturating_mul(rows).saturating_div(1000))
    }
}

impl fmt::Display for Preview {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let took = Duration::from_millis(self.took_ms);
        write!(
            f,
            "{} rows in {:.1} s; {} cut to fit the model, {} empty, {} skipped.",
            self.labelled.saturating_add(self.empty),
            took.as_secs_f64(),
            self.cut,
            self.empty,
            self.skipped,
        )
    }
}

impl Run {
    /// What each column of the output means, the key first: the sentences
    /// `describe_table` gives the agent and the Tables page.
    #[must_use]
    pub fn column_meanings(&self) -> Vec<(String, String)> {
        columns::OutputColumns::new(&self.source_table, &self.key_column, &self.questions)
            .map(|columns| columns.meanings())
            .unwrap_or_default()
    }

    /// The set this run ran under.
    #[must_use]
    pub fn label_set(&self) -> LabelSet {
        LabelSet {
            key_column: self.key_column.clone(),
            key_reason: self.key_reason,
            text_columns: self.text_columns.clone(),
            questions: self.questions.clone(),
            sentence: self.sentence.clone(),
        }
    }
}

/// What a draft, its outline, and what came of it look like together: the
/// one answer REST, MCP, and the agent give.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub struct Report {
    pub draft: Draft,
    pub outline: Outline,
    pub preview: Option<Preview>,
    pub run: Option<Run>,
}

/// How a run that started ended, as an interface audits it.
#[derive(Debug, Clone)]
pub struct RunEnded {
    /// The run, as `_quack_classifications` holds it after the end.
    pub run: Run,
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
    async fn report(&self, db: &Writer, id: &RunId, result: &Result<Run>) {
        let run = if let Ok(run) = result {
            Some(run.clone())
        } else {
            let id = id.clone();
            db.run(move |db| Run::get(db, &id)).await.ok()
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
pub struct LabellingJob {
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

impl LabellingJob {
    /// Submit the run of `draft` and wait for its result.
    async fn submit(
        self,
        draft: Draft,
        rows: Rows,
        progress: impl Fn(ChunkDone) + Send + Sync + 'static,
    ) -> Result<Run> {
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
        let label = format!("label {}", draft.table);
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
            let mut run = Box::pin(priority.scope(draft.run(
                Labelling {
                    db: &db,
                    decision: &decision,
                    started_by: started_by_text.as_deref(),
                    run_id: run_id.clone(),
                    waiting,
                    control: RunControl {
                        progress: &report,
                        cancel: Some(&run_cancel),
                    },
                },
                rows,
            )));
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
            CoreError::Analysis(String::from(
                "the labelling job did not run to its end (the server is stopping)",
            ))
        })?
    }
}

impl Draft {
    /// [`Self::run`] as a job of the queue, awaited by the caller. If the
    /// caller is dropped (a cancelled turn, an MCP client that left) the job
    /// goes on to its end and records it, and the queue's shutdown waits
    /// for it.
    ///
    /// # Errors
    ///
    /// As [`Self::run`], or [`CoreError::Analysis`] when the job did not run to
    /// its end (the queue was shutting down).
    pub async fn run_as_job(
        self,
        tracked: LabellingJob,
        rows: Rows,
        progress: impl Fn(ChunkDone) + Send + Sync + 'static,
    ) -> Result<Run> {
        tracked.submit(self, rows, progress).await
    }

    /// What a run would do, without doing it: the checks of a run and its
    /// row count, read from the workspace.
    ///
    /// # Errors
    ///
    /// Returns a [`Error`] when the set cannot be run, or the error
    /// of the model's server asked for its weights' digest.
    pub async fn outline(
        &self,
        writer: &Writer,
        decision: &DecisionModel,
        rows: Rows,
    ) -> Result<Outline> {
        pipeline::outline(self, writer, decision, rows).await
    }

    /// Label the first `n` rows (1 to 100) by key and return them, writing
    /// nothing: no table, no record, no claim. With `Rows::Missing` they
    /// are the first rows the output lacks.
    ///
    /// # Errors
    ///
    /// Returns a [`Error`] when the set cannot be run,
    /// [`crate::error::Error::DecisionRefused`] when the model refuses the
    /// questions, or the model's or the database's error.
    pub async fn preview(
        &self,
        writer: &Writer,
        decision: &DecisionModel,
        n: u32,
        rows: Rows,
        waiting: Waiting,
        control: RunControl<'_>,
    ) -> Result<Preview> {
        Box::pin(pipeline::preview(
            self, writer, decision, n, rows, waiting, control,
        ))
        .await
    }

    /// Label the table's rows, recording this draft's set as approved.
    ///
    /// # Errors
    ///
    /// Returns a [`Error`] when the set cannot be run,
    /// [`crate::error::Error::Cancelled`] after a cancel, or the model's or
    /// the database's error. The rows labelled before a stop stay.
    pub async fn run(&self, labelling: Labelling<'_>, rows: Rows) -> Result<Run> {
        Box::pin(pipeline::run(self, labelling, rows)).await
    }
}
