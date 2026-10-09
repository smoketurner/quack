//! `classify_rows`: the agent labels a table's text with the workspace's
//! decision model (issue #472).
//!
//! A preview labels the first rows and writes nothing. A run creates or
//! adds to a table of labels, so it goes through the turn's write policy
//! like any other write, and it is refused beyond what a turn can wait for
//! (`[decision].interactive_budget`); a larger table is labelled with
//! `quack classify` or the REST route, as a job.

use std::sync::Arc;

use rig::tool::{Tool, ToolContext, ToolExecutionError};

use super::{DEFAULT_STEP_RESULT_ROWS, ReaderDb, SharedDb, ToolArgs, Turn};
use crate::analysis::events::{StepInProgress, ToolName};
use crate::analysis::policy::Hold;
use crate::classify::{
    ClassificationPreview, ClassificationRun, ClassifyArgs, LabelJobs, Tracked, Waiting,
};
use crate::config::Config;
use crate::error::Result;
use crate::ids::{RunId, UserId};
use crate::llm::decision::DecisionModel;
use crate::progress::{ChunkDone, RunControl};

/// The decision model a turn labels with, how much it may label while the
/// turn waits, and the job queue its runs go through.
#[derive(Clone)]
pub struct Labeller {
    model: DecisionModel,
    budget: u64,
    /// The server user asking; recorded on the runs and as their owner.
    user: Option<UserId>,
    jobs: LabelJobs,
}

impl Labeller {
    /// The workspace's decision model, or `None` when none is configured
    /// or the workspace's provider list keeps it out: the tool is then not
    /// offered.
    ///
    /// # Errors
    ///
    /// Returns an error if the setting is invalid or the provider's
    /// credential cannot be resolved.
    pub async fn from_config(
        config: &Config,
        jobs: LabelJobs,
        user: Option<&UserId>,
    ) -> Result<Option<Self>> {
        Ok(DecisionModel::offered(config).await?.map(|model| Self {
            model,
            budget: config.decision.interactive_budget,
            user: user.cloned(),
            jobs,
        }))
    }
}

/// The `classify_rows` tool.
pub struct ClassifyRowsTool {
    db: SharedDb,
    /// Told when a run wrote tables, as `run_sql` tells it.
    reader: ReaderDb,
    labeller: Labeller,
}

impl ClassifyRowsTool {
    /// The tool over a workspace's writer and reader, labelling with `labeller`.
    #[must_use]
    pub const fn new(db: SharedDb, reader: ReaderDb, labeller: Labeller) -> Self {
        Self {
            db,
            reader,
            labeller,
        }
    }

    /// Preview the labels, or weigh the run against the write policy and
    /// run it.
    async fn attempt(
        &self,
        turn: &Turn,
        args: &ClassifyArgs,
        control: RunControl<'_>,
        run_id: &RunId,
    ) -> Result<Attempted> {
        let (db, model) = (&*self.db, &self.labeller.model);
        let request = &args.classification;
        let waiting = Waiting::Caller {
            budget: self.labeller.budget,
        };
        if let Some(rows) = args.preview {
            return request
                .preview(db, model, rows, waiting, control)
                .await
                .map(Attempted::Previewed);
        }
        let outline = request.outline(db, model).await?;
        outline.within(waiting)?;
        if let Err(hold) = turn.permit_write(&outline.statement(model.label())).await {
            return Ok(Attempted::Held(hold));
        }
        let recorder = turn.recorder.clone();
        request
            .clone()
            .run_as_job(
                Tracked {
                    db: Arc::clone(&self.db),
                    decision: model.clone(),
                    started_by: self.labeller.user.clone(),
                    run_id: run_id.clone(),
                    waiting,
                    cancel: turn.cancel().clone(),
                    jobs: self.labeller.jobs.clone(),
                },
                move |done: ChunkDone| {
                    recorder.status(format!("labelled {} of {} rows", done.done, done.total));
                },
            )
            .await
            .map(|run| Attempted::Ran(Box::new(run)))
    }
}

/// What a call came to.
enum Attempted {
    Previewed(ClassificationPreview),
    Ran(Box<ClassificationRun>),
    Held(Hold),
}

impl Attempted {
    /// Close `step` with the outcome and give the model its result.
    fn finish(self, step: StepInProgress, turn: &Turn) -> Result<String> {
        Ok(match self {
            Self::Held(hold) => {
                step.finish(hold.summary());
                String::from(hold.refusal())
            }
            Self::Previewed(preview) => {
                let mut table = Vec::new();
                if let Err(error) = preview.result.write_table(&mut table) {
                    return Err(step.fail(error));
                }
                let text = format!(
                    "Key: {}\n{}{preview}",
                    preview.key_column,
                    String::from_utf8_lossy(&table),
                );
                let rows = u64::try_from(preview.result.rows.len()).unwrap_or(u64::MAX);
                step.finish_with_result(rows, preview.result, DEFAULT_STEP_RESULT_ROWS);
                text
            }
            Self::Ran(run) => {
                step.finish_run(
                    format!("{} rows labelled into {}", run.labelled, run.output_table),
                    run.id.clone(),
                );
                let columns = run
                    .column_meanings()
                    .into_iter()
                    .map(|(name, meaning)| format!("{name}: {meaning}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "{run}\nQuery it with SQL, joined to {} on {}. Columns:\n{columns}\n{}",
                    run.source_table,
                    run.key_column,
                    turn.recorder.budget_note()
                )
            }
        })
    }
}

impl Tool for ClassifyRowsTool {
    const NAME: &'static str = ToolName::ClassifyRows.as_str();
    type Error = ToolExecutionError;
    type Args = ClassifyArgs;
    type Output = String;

    fn description(&self) -> String {
        format!(
            "Label the text of a table's rows with the workspace's decision model. It answers a \
             fixed set of questions about each row (pick one of 2 to 26 options, true or false, \
             or a level on a rubric) with probabilities, and the answers go to a new table named \
             <table>_<set name> that joins back to the table by its key, so SQL can GROUP BY and \
             filter on them. Pass `preview` (1 to 100) first to see the labels on the first rows; \
             a preview writes nothing. A run labels the rows the output lacks, or every row again \
             with rows = \"all\"; it creates or changes a table, so it needs write permission. A \
             run of more than {} answers (rows times questions) is refused here: tell the user to \
             run `quack classify` for it. The model reads English, and about 2,500 characters of a \
             row.",
            self.labeller.budget
        )
    }

    fn parameters(&self) -> serde_json::Value {
        ClassifyArgs::schema()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> std::result::Result<Self::Output, Self::Error> {
        let turn = Turn::of(context)?;
        let request = &args.classification;
        let step = turn.recorder.start(
            ToolName::ClassifyRows,
            &format!("{} by {}", request.table, request.question_set.name),
        );
        let progress = |done: ChunkDone| {
            turn.recorder
                .status(format!("labelled {} of {} rows", done.done, done.total));
        };
        let control = RunControl {
            progress: &progress,
            cancel: Some(turn.cancel()),
        };
        let run_id = RunId::generate();
        match self.attempt(&turn, &args, control, &run_id).await {
            Ok(attempted) => {
                if matches!(attempted, Attempted::Ran(_)) {
                    self.reader.observe_write().await;
                }
                Ok(attempted.finish(step, &turn)?)
            }
            Err(error) => {
                let id = run_id.clone();
                let began = self
                    .db
                    .run(move |db| ClassificationRun::get(db, &id))
                    .await
                    .is_ok();
                Err(step.fail_run(error, began.then_some(run_id)).into())
            }
        }
    }
}
