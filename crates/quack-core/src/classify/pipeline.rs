//! Reading a table page by page, asking the decision model about each row,
//! and writing the answers (design doc 6.6).

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;

use super::columns::OutputColumns;
use super::plan::Plan;
use super::store::{Ended, PageWrite, Start, Written};
use super::{
    Classification, ClassificationOutline, ClassificationPreview, ClassificationRun, ClassifyError,
    Rows, RunStatus, Waiting,
};
use crate::error::{Error, Result};
use crate::ids::RunId;
use crate::llm::decision::{Answers, Asked, Asker, DecisionModel, State};
use crate::progress::{ChunkDone, RunControl};
use crate::storage::profile::TableProfile;
use crate::storage::workspace::{QueryResults, WorkspaceDb, quote_ident};
use crate::storage::writer::Writer;

/// Rows read, asked about, and written at a time.
const PAGE_ROWS: u32 = 256;
/// Rows a preview labels at most.
const PREVIEW_ROWS: u32 = 100;

/// What a run labels with, and how it is watched.
pub struct Labelling<'a> {
    pub db: &'a Writer,
    pub decision: &'a DecisionModel,
    /// Who started the run, as the server's user id.
    pub started_by: Option<&'a str>,
    pub run_id: RunId,
    pub waiting: Waiting,
    pub control: RunControl<'a>,
}

/// A source row: its key as text and each text column's value.
struct SourceRow {
    key: String,
    texts: Vec<(String, Option<String>)>,
}

/// A reader connection: a clone of the writer's, lent to one read at a
/// time on the blocking pool, each in its own read-only transaction.
struct Reader(Option<WorkspaceDb>);

impl Reader {
    async fn open(writer: &Writer) -> Result<Self> {
        Ok(Self(Some(writer.run(WorkspaceDb::try_clone_reader).await?)))
    }

    async fn read<T: Send + 'static>(
        &mut self,
        f: impl FnOnce(&WorkspaceDb) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let db = self.0.take().ok_or(Error::WriterStopped)?;
        let (db, outcome) = tokio::task::spawn_blocking(move || {
            let outcome = db.read_only(f);
            (db, outcome)
        })
        .await
        .map_err(|e| Error::Analysis(format!("the labelling read failed: {e}")))?;
        self.0 = Some(db);
        outcome
    }
}

impl SourceRow {
    fn from_row(row: &duckdb::Row<'_>, texts: &[String]) -> Result<Self> {
        let mut values = Vec::with_capacity(texts.len());
        for (at, column) in texts.iter().enumerate() {
            values.push((column.clone(), row.get(at.saturating_add(1))?));
        }
        Ok(Self {
            key: row.get(0)?,
            texts: values,
        })
    }
}

impl Plan {
    /// The next page of source rows after `cursor` that `target` does not
    /// hold: the keys of the next [`PAGE_ROWS`] such rows are found first,
    /// so a table that is mostly labelled is not read through.
    fn read_page(
        &self,
        db: &WorkspaceDb,
        target: &str,
        cursor: Option<&str>,
    ) -> Result<Vec<SourceRow>> {
        let key = quote_ident(&self.key.name);
        let source = quote_ident(self.source.as_str());
        let mut conditions = Vec::new();
        if cursor.is_some() {
            conditions.push(format!("s.{key} > {}", self.key_param()));
        }
        conditions.push(self.not_in(target));
        let bounds = format!(
            "SELECT CAST(min(k) AS VARCHAR), CAST(max(k) AS VARCHAR) FROM \
             (SELECT s.{key} AS k FROM {source} s WHERE {} ORDER BY s.{key} LIMIT {PAGE_ROWS})",
            conditions.join(" AND ")
        );
        let (first, last): (Option<String>, Option<String>) =
            db.connection()
                .query_row(&bounds, duckdb::params_from_iter(cursor), |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })?;
        let (Some(first), Some(last)) = (first, last) else {
            return Ok(Vec::new());
        };
        let kp = self.key_param();
        let sql = format!(
            "SELECT {} FROM {source} s WHERE s.{key} BETWEEN {kp} AND {kp} AND {} ORDER BY s.{key}",
            self.page_select(),
            self.not_in(target)
        );
        self.collect(db, &sql, duckdb::params_from_iter([first, last]))
    }

    /// The first `n` source rows by key.
    fn read_first(&self, db: &WorkspaceDb, n: u32) -> Result<Vec<SourceRow>> {
        let sql = format!(
            "SELECT {} FROM {} s ORDER BY s.{} LIMIT {n}",
            self.page_select(),
            quote_ident(self.source.as_str()),
            quote_ident(&self.key.name)
        );
        self.collect(db, &sql, duckdb::params![])
    }

    fn collect(
        &self,
        db: &WorkspaceDb,
        sql: &str,
        params: impl duckdb::Params,
    ) -> Result<Vec<SourceRow>> {
        let mut stmt = db.connection().prepare(sql)?;
        let mut rows = stmt.query(params)?;
        let mut page = Vec::new();
        while let Some(row) = rows.next()? {
            page.push(SourceRow::from_row(row, &self.text_columns)?);
        }
        Ok(page)
    }
}

/// How a row was answered, by its key.
struct Labelled {
    key: String,
    outcome: Outcome,
}

enum Outcome {
    Answered(Answers),
    Empty,
    Unfit,
}

/// Asks the questions about rows, a few at a time, in order.
struct PageAsker<'a> {
    asker: Asker<'a>,
    control: RunControl<'a>,
    concurrency: usize,
}

impl PageAsker<'_> {
    /// The rows answered before the first failure, and the failure, if
    /// any. A row with no text is not sent.
    async fn label(&self, rows: Vec<SourceRow>) -> (Vec<Labelled>, Option<Error>) {
        let ask = |row: SourceRow| async move {
            let state = State::new(row.texts);
            let outcome = if state.is_empty() {
                Ok(Outcome::Empty)
            } else {
                self.control
                    .or_cancelled(self.asker.ask(&state))
                    .await
                    .map(|asked| match asked {
                        Asked::Answered(answers) => Outcome::Answered(answers),
                        Asked::Unfit => Outcome::Unfit,
                    })
            };
            (row.key, outcome)
        };
        let mut answered = Vec::with_capacity(rows.len());
        let mut results = futures::stream::iter(rows)
            .map(ask)
            .buffered(self.concurrency.max(1));
        while let Some((key, outcome)) = results.next().await {
            match outcome {
                Ok(outcome) => answered.push(Labelled { key, outcome }),
                Err(error) => return (answered, Some(error)),
            }
        }
        (answered, None)
    }
}

impl Plan {
    /// The page to write for `labelled`: a row with an answer or no text
    /// is written, a row the model refused is counted.
    fn page_write(&self, labelled: Vec<Labelled>) -> PageWrite {
        let mut write = PageWrite {
            rows: Vec::with_capacity(labelled.len()),
            skipped: 0,
        };
        for Labelled { key, outcome } in labelled {
            match outcome {
                Outcome::Answered(answers) => {
                    let cut = answers.truncated();
                    write.rows.push((
                        key,
                        self.columns.cells(Some(&answers)),
                        Written::Labelled { cut },
                    ));
                }
                Outcome::Empty => write
                    .rows
                    .push((key, self.columns.cells(None), Written::Empty)),
                Outcome::Unfit => write.skipped = write.skipped.saturating_add(1),
            }
        }
        write
    }
}

/// The rows of a run read, asked about, and written.
struct Pipeline<'a> {
    plan: Arc<Plan>,
    writer: &'a Writer,
    reader: Reader,
    labeller: PageAsker<'a>,
    run: &'a RunId,
    target: String,
    started: Instant,
    done: u64,
    skipped: u64,
}

impl Pipeline<'_> {
    /// Label every page, writing each as it is answered. A stop keeps the
    /// rows answered before it.
    async fn drive(&mut self) -> Result<()> {
        let mut cursor: Option<String> = None;
        loop {
            self.labeller.control.check()?;
            let (plan, target, after) =
                (Arc::clone(&self.plan), self.target.clone(), cursor.clone());
            let page = self
                .reader
                .read(move |db| plan.read_page(db, &target, after.as_deref()))
                .await?;
            let Some(last) = page.last().map(|row| row.key.clone()) else {
                return Ok(());
            };
            let page_started = Instant::now();
            let (labelled, stopped) = self.labeller.label(page).await;
            self.write(labelled).await?;
            self.report(page_started.elapsed());
            if let Some(error) = stopped {
                return Err(error);
            }
            cursor = Some(last);
        }
    }

    async fn write(&mut self, labelled: Vec<Labelled>) -> Result<()> {
        let write = self.plan.page_write(labelled);
        self.skipped = self.skipped.saturating_add(write.skipped);
        let (plan, run, target) = (
            Arc::clone(&self.plan),
            self.run.clone(),
            self.target.clone(),
        );
        let added = self
            .writer
            .run(move |db| write.write(db, &plan, &run, &target))
            .await?;
        self.done = self.done.saturating_add(added);
        Ok(())
    }

    fn report(&self, took: Duration) {
        let total = u32::try_from(self.plan.remaining).unwrap_or(u32::MAX);
        let seen = self.done.saturating_add(self.skipped);
        (self.labeller.control.progress)(ChunkDone {
            done: u32::try_from(seen).unwrap_or(u32::MAX).min(total),
            total,
            failed: u32::try_from(self.skipped).unwrap_or(u32::MAX),
            took,
            elapsed: self.started.elapsed(),
        });
    }
}

/// [`Classification::run`].
pub(super) async fn run(
    request: &Classification,
    labelling: Labelling<'_>,
) -> Result<ClassificationRun> {
    let Labelling {
        db,
        decision,
        started_by,
        run_id,
        waiting,
        control,
    } = labelling;
    let mut reader = Reader::open(db).await?;
    let plan = {
        let request = request.clone();
        Arc::new(reader.read(move |db| Plan::resolve(db, &request)).await?)
    };
    let _claim = db
        .claim(plan.claimed())
        .ok_or_else(|| ClassifyError::Running {
            table: plan.output.to_string(),
        })?;
    plan.outline().within(waiting)?;
    let asker = decision.asker(&plan.set.questions).await?;
    let digest = decision.digest().await?;
    let (start_plan, start_run) = (Arc::clone(&plan), run_id.clone());
    let (model, started_by) = (decision.label().to_owned(), started_by.map(str::to_owned));
    let claims = db.claims();
    db.run(move |db| {
        Start {
            plan: &start_plan,
            run_id: &start_run,
            model: &model,
            digest: &digest,
            started_by: started_by.as_deref(),
            claims: &claims,
        }
        .apply(db)
    })
    .await?;
    let mut pipeline = Pipeline {
        target: plan.target(),
        plan: Arc::clone(&plan),
        writer: db,
        reader,
        labeller: PageAsker {
            asker,
            control,
            concurrency: decision.concurrency(),
        },
        run: &run_id,
        started: Instant::now(),
        done: 0,
        skipped: 0,
    };
    let outcome = pipeline.drive().await;
    let ended = match &outcome {
        Ok(()) => Ended::Completed,
        Err(Error::Cancelled) => Ended::Stopped {
            status: RunStatus::Cancelled,
            error: Error::Cancelled.to_string(),
        },
        Err(other) => Ended::Stopped {
            status: RunStatus::Failed,
            error: other.to_string(),
        },
    };
    let (end_plan, end_run) = (Arc::clone(&plan), run_id.clone());
    let recorded = db
        .run(move |db| ended.record(db, &end_plan, &end_run))
        .await;
    TableProfile::after_write(db).await;
    outcome?;
    recorded
}

/// [`Classification::outline`].
pub(super) async fn outline(
    request: &Classification,
    writer: &Writer,
    decision: &DecisionModel,
) -> Result<ClassificationOutline> {
    let mut reader = Reader::open(writer).await?;
    let request = request.clone();
    let plan = reader.read(move |db| Plan::resolve(db, &request)).await?;
    // Not held here: the run takes it. Another run holding it is refused now.
    let free = writer.claim(plan.claimed());
    if free.is_none() {
        return Err(ClassifyError::Running {
            table: plan.output.to_string(),
        }
        .into());
    }
    drop(free);
    if plan.output_exists && plan.rows == Rows::Missing {
        let digest = decision.digest().await?;
        let output = plan.output.to_string();
        let in_force = reader
            .read(move |db| ClassificationRun::in_force(db, &output))
            .await?;
        plan.check_definition(in_force.as_ref(), &digest)?;
    }
    Ok(plan.outline())
}

/// [`Classification::preview`].
pub(super) async fn preview(
    request: &Classification,
    writer: &Writer,
    decision: &DecisionModel,
    n: u32,
    waiting: Waiting,
    control: RunControl<'_>,
) -> Result<ClassificationPreview> {
    let started = Instant::now();
    let mut reader = Reader::open(writer).await?;
    let plan = {
        let request = request.clone();
        Arc::new(reader.read(move |db| Plan::resolve(db, &request)).await?)
    };
    let mut asked = plan.outline();
    asked.remaining = u64::from(n.clamp(1, PREVIEW_ROWS));
    asked.within(waiting)?;
    let asker = decision.asker(&plan.set.questions).await?;
    let rows = {
        let plan = Arc::clone(&plan);
        let n = n.clamp(1, PREVIEW_ROWS);
        reader.read(move |db| plan.read_first(db, n)).await?
    };
    let labeller = PageAsker {
        asker,
        control,
        concurrency: decision.concurrency(),
    };
    let (labelled, stopped) = labeller.label(rows).await;
    if let Some(error) = stopped {
        return Err(error);
    }
    let (mut shown, mut cut, mut empty, mut skipped) = (0_u32, 0_u32, 0_u32, 0_u32);
    let mut table = Vec::with_capacity(labelled.len());
    for Labelled { key, outcome } in labelled {
        let cells = match &outcome {
            Outcome::Answered(answers) => {
                shown = shown.saturating_add(1);
                if answers.truncated() {
                    cut = cut.saturating_add(1);
                }
                plan.columns.cells(Some(answers))
            }
            Outcome::Empty => {
                empty = empty.saturating_add(1);
                plan.columns.cells(None)
            }
            Outcome::Unfit => {
                skipped = skipped.saturating_add(1);
                continue;
            }
        };
        table.push(OutputColumns::json_row(&key, cells));
    }
    Ok(ClassificationPreview {
        key_column: plan.key.name.clone(),
        output_table: plan.output.to_string(),
        result: QueryResults {
            columns: plan.columns.column_names(),
            rows: table,
        },
        labelled: shown,
        cut,
        empty,
        skipped,
        took_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        remaining: plan.remaining,
    })
}
