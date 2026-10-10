//! `classify_rows`: the agent labels a table's text with the workspace's
//! decision model (issue #472).
//!
//! The agent names a table and says in a sentence what the person wants to
//! know about each row; the chat model drafts the questions. A preview
//! labels the first rows and writes nothing. A run creates or adds to a
//! table of labels, so it goes through the turn's write policy like any
//! other write, and it is refused beyond what a turn can wait for
//! (`[decision].interactive_budget`); a larger table is labelled with
//! `quack classify` or the REST route, as a job. The questions are stored
//! only by a run that was allowed.

use std::sync::Arc;

use rig::tool::{Tool, ToolContext, ToolExecutionError};
use schemars::JsonSchema;
use serde::Deserialize;

use super::{DEFAULT_STEP_RESULT_ROWS, ReaderDb, SharedDb, ToolArgs, Turn};
use crate::analysis::events::{StepInProgress, ToolName};
use crate::analysis::policy::Hold;
use crate::classify::{self, Draft, DraftContext, Drafter, LabelJobs, LabellingJob, Rows, Waiting};
use crate::config::Config;
use crate::error::Result;
use crate::ids::{RunId, UserId};
use crate::llm::ChatDrafter;
use crate::llm::decision::DecisionModel;
use crate::progress::{ChunkDone, RunControl};
use crate::text::{OneLine, Thousands};

/// Rows the approval card shows the labels of.
const CARD_ROWS: u32 = 3;

/// What the model passes to `classify_rows`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
pub struct ClassifyRowsArgs {
    /// The table whose rows are labelled, by name.
    pub table: String,
    /// What the person wants to know about each row, in their words. The
    /// questions are drafted from it. Leave it out to use the questions
    /// approved before.
    #[serde(default)]
    pub sentence: Option<String>,
    /// Label only this many rows (1 to 100) and show them, writing nothing.
    /// Do this first.
    #[serde(default)]
    pub preview: Option<u32>,
}

/// The decision model a turn labels with, how much it may label while the
/// turn waits, and the job queue its runs go through.
#[derive(Clone)]
pub struct Labeller {
    model: DecisionModel,
    budget: u64,
    /// What drafts questions from a sentence; none without a chat model.
    drafter: Option<ChatDrafter>,
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
            drafter: ChatDrafter::from_config(config),
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

    /// Draft the questions, then preview the labels, or weigh the run
    /// against the write policy and run it.
    async fn attempt(
        &self,
        turn: &Turn,
        args: &ClassifyRowsArgs,
        control: RunControl<'_>,
        run_id: &RunId,
    ) -> Result<Attempted> {
        let (db, model) = (&*self.db, &self.labeller.model);
        let waiting = Waiting::Caller {
            budget: self.labeller.budget,
        };
        let drafter: Option<&dyn Drafter> = self
            .labeller
            .drafter
            .as_ref()
            .map(|d| -> &dyn Drafter { d });
        let context = DraftContext {
            db,
            decision: model,
            drafter,
        };
        let draft = Draft::prepare(&context, &args.table, args.sentence.as_deref()).await?;
        if let Some(rows) = args.preview {
            let preview = draft
                .preview(db, model, rows, Rows::Missing, waiting, control)
                .await?;
            return Ok(Attempted::Previewed(Box::new((draft, preview))));
        }
        let outline = draft.outline(db, model, Rows::Missing).await?;
        if outline.remaining == 0 {
            return Ok(Attempted::Nothing(Box::new(outline)));
        }
        outline.within(waiting)?;
        let card = draft
            .preview(db, model, CARD_ROWS, Rows::Missing, waiting, control)
            .await?;
        let outline = outline.estimated_from(&card);
        if let Err(hold) = turn
            .permit_write(&Self::statement(&outline, model.label(), &card))
            .await
        {
            return Ok(Attempted::Held(hold));
        }
        let recorder = turn.recorder.clone();
        draft
            .run_as_job(
                LabellingJob {
                    db: Arc::clone(&self.db),
                    decision: model.clone(),
                    started_by: self.labeller.user.clone(),
                    run_id: run_id.clone(),
                    waiting,
                    cancel: turn.cancel().clone(),
                    jobs: self.labeller.jobs.clone(),
                },
                Rows::Missing,
                move |done: ChunkDone| {
                    recorder.status(format!(
                        "labelled {} of {} rows",
                        Thousands(u64::from(done.done)),
                        Thousands(u64::from(done.total))
                    ));
                },
            )
            .await
            .map(|run| Attempted::Ran(Box::new(run)))
    }

    /// What a person is asked to allow: the outline, then the labels of the
    /// first rows.
    fn statement(outline: &classify::Outline, model: &str, card: &classify::Preview) -> String {
        let mut lines = vec![outline.statement(model)];
        let shown: Vec<String> = card
            .compact
            .rows
            .iter()
            .map(|row| {
                let cells: Vec<String> = row
                    .iter()
                    .zip(&card.compact.columns)
                    .enumerate()
                    .map(|(at, (value, column))| {
                        let value = value.as_str().unwrap_or_default();
                        if at == 0 {
                            OneLine(value).to_string()
                        } else {
                            format!(
                                "{}={}",
                                OneLine(column.trim_end_matches(" (p)")),
                                OneLine(value)
                            )
                        }
                    })
                    .collect();
                match cells.split_first() {
                    Some((key, rest)) => format!("{key}: {}", rest.join(", ")),
                    None => String::new(),
                }
            })
            .collect();
        if !shown.is_empty() {
            lines.push(format!("-- preview: {}", shown.join(" | ")));
        }
        lines.join("\n")
    }
}

/// What a call came to.
enum Attempted {
    Previewed(Box<(Draft, classify::Preview)>),
    Ran(Box<classify::Run>),
    Held(Hold),
    /// The table of labels has every row already.
    Nothing(Box<classify::Outline>),
}

impl Attempted {
    /// Close `step` with the outcome and give the model its result.
    fn finish(self, step: StepInProgress, turn: &Turn) -> Result<String> {
        Ok(match self {
            Self::Held(hold) => {
                step.finish(hold.summary());
                String::from(hold.refusal())
            }
            Self::Nothing(outline) => {
                step.finish("nothing to label");
                format!(
                    "Nothing to label: {} has every row of {}.",
                    outline.output_table, outline.source_table
                )
            }
            Self::Previewed(shown) => {
                let (draft, preview) = *shown;
                let mut table = Vec::new();
                if let Err(error) = preview.compact.write_table(&mut table) {
                    return Err(step.fail(error));
                }
                let text = format!(
                    "{draft}\nNothing is kept until the labelling runs.\nPreview, {}:\n{}{preview}",
                    if preview.cut > 0 {
                        "a * marks a row cut to fit the model"
                    } else {
                        "the rows the next run labels first"
                    },
                    String::from_utf8_lossy(&table),
                );
                let rows = u64::try_from(preview.compact.rows.len()).unwrap_or(u64::MAX);
                step.finish_with_result(rows, preview.compact, DEFAULT_STEP_RESULT_ROWS);
                text
            }
            Self::Ran(run) => {
                step.finish_run(
                    format!(
                        "{} rows labelled into {}",
                        Thousands(run.labelled),
                        run.output_table
                    ),
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
    type Args = ClassifyRowsArgs;
    type Output = String;

    fn description(&self) -> String {
        format!(
            "Label the text of a table's rows with the workspace's decision model. Say in \
             `sentence`, in the person's words, what they want to know about each row; the \
             questions (pick one of several options, yes or no, or a level on a scale) are \
             drafted from it, and the answers, with probabilities, go to a table named \
             <table>_labels that joins back to the table by its key, so SQL can GROUP BY and \
             filter on them. Pass `preview` (1 to 100) first to see the questions and the labels \
             on the first rows; a preview writes nothing and keeps nothing. A run labels the rows \
             the table of labels lacks; it creates or changes a table, so it needs write \
             permission and the person approves it. Leave `sentence` out to use the questions \
             approved before. A run of more than {} answers (rows times questions) is refused \
             here: tell the user to run `quack classify` for it. The model reads English, and \
             about 2,500 characters of a row.",
            self.labeller.budget
        )
    }

    fn parameters(&self) -> serde_json::Value {
        ClassifyRowsArgs::schema()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> std::result::Result<Self::Output, Self::Error> {
        let turn = Turn::of(context)?;
        let step = turn.recorder.start(
            ToolName::ClassifyRows,
            &match &args.sentence {
                Some(sentence) => format!("{}: {}", args.table, OneLine(sentence)),
                None => args.table.clone(),
            },
        );
        let progress = |done: ChunkDone| {
            turn.recorder.status(format!(
                "labelled {} of {} rows",
                Thousands(u64::from(done.done)),
                Thousands(u64::from(done.total))
            ));
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
                    .run(move |db| classify::Run::get(db, &id))
                    .await
                    .is_ok();
                Err(step.fail_run(error, began.then_some(run_id)).into())
            }
        }
    }
}
