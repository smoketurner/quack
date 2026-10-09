//! `quack classify`: label a table's text with a decision model (issue
//! #472). A person names a table and says in a sentence what they want to
//! know about each row; the chat model drafts the questions, the decision
//! model answers them for a preview, and a yes runs the labelling. The
//! terminal's `/classify` runs the same steps, asking in its own prompt.

use std::io::Write;

use anyhow::Result;
use clap::Subcommand;
use quack_core::classify::{
    self, Draft, DraftContext, DraftOrigin, Drafter, Effect, Labelling, RelabelReason, Rows,
    SAMPLE_ROWS, Waiting,
};
use quack_core::config::Config;
use quack_core::error::{Error, Result as CoreResult};
use quack_core::ids::RunId;
use quack_core::llm::ChatDrafter;
use quack_core::llm::decision::DecisionModel;
use quack_core::progress::RunControl;
use quack_core::storage::workspace::WorkspaceDb;
use quack_core::storage::writer::Writer;
use quack_core::text::Thousands;
use serde::Serialize;

use crate::args::QueryFormat;
use crate::confirm::Confirm;
use crate::text_or_json::TextOrJson;

/// Runs `list` shows at most.
const LISTED_RUNS: u32 = 100;
/// Rows an interactive run previews unless `--preview` says otherwise.
const PREVIEW_ROWS: u32 = 10;

/// What `quack classify` does besides labelling a table.
#[derive(Subcommand)]
pub enum ClassifyAction {
    /// List the runs that labelled tables, newest first
    List {
        /// `json` prints one JSON object per run
        #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
        format: TextOrJson,
    },
    /// Show the questions last approved for a table
    Show {
        /// The table whose questions are shown
        table: String,
        /// `json` prints the questions and the labels as one JSON document
        #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
        format: TextOrJson,
    },
}

/// The arguments of a labelling.
#[derive(clap::Args)]
pub struct ClassifyArgs {
    /// The table whose rows are labelled (a table named `list` or `show`
    /// is labelled through the API or the agent: those are the subcommands
    /// here)
    #[arg(required = true)]
    table: Option<String>,
    /// What you want to know about each row, in your words; the questions
    /// are drafted from it. Leave it out to use the questions approved
    /// before.
    #[arg(value_name = "SENTENCE")]
    sentence: Option<String>,
    /// Label without previewing or asking
    #[arg(short = 'y', long)]
    yes: bool,
    /// Label every row again; the old labels serve until the run completes
    #[arg(long)]
    all: bool,
    /// Label only the first N rows (1 to 100), show them, and stop
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..=100))]
    preview: Option<u32>,
}

/// `quack classify TABLE [SENTENCE]`, `quack classify list`, or `quack
/// classify show TABLE`.
#[derive(clap::Args)]
#[command(args_conflicts_with_subcommands = true, subcommand_negates_reqs = true)]
pub struct ClassifyCommand {
    #[command(subcommand)]
    action: Option<ClassifyAction>,
    #[command(flatten)]
    run: ClassifyArgs,
}

/// A labelling worked out and shown, waiting for a yes.
pub struct Prepared {
    draft: Draft,
    outline: classify::Outline,
    preview: Option<classify::Preview>,
    rows: Rows,
    /// The sentence the person typed, when they typed one.
    asked: Option<String>,
    /// Whether questions had been approved before this call.
    earlier: bool,
    /// The decision model, as `provider/model`.
    decision: String,
}

impl ClassifyArgs {
    /// The sentence, when one was given.
    #[must_use]
    pub fn sentence(&self) -> Option<String> {
        let sentence = self.sentence.as_deref()?.trim();
        (!sentence.is_empty()).then(|| sentence.to_owned())
    }

    /// The table.
    #[must_use]
    pub fn table(&self) -> &str {
        self.table.as_deref().unwrap_or_default()
    }

    /// Whether `-y` was given.
    #[must_use]
    pub const fn yes(&self) -> bool {
        self.yes
    }

    /// Rows to preview and stop, when `--preview` was given.
    #[must_use]
    pub const fn preview_only(&self) -> Option<u32> {
        self.preview
    }

    /// Whether the table has questions to draft or revise, so the chat
    /// model will be asked.
    ///
    /// # Errors
    ///
    /// Returns the database's error.
    pub async fn drafting(&self, db: &Writer) -> Result<Option<DraftOrigin>> {
        let (table, sentence) = (self.table().to_owned(), self.sentence());
        Ok(db
            .run(move |db| Draft::drafting(db, &table, sentence.as_deref()))
            .await?)
    }

    /// Draft or find the questions, preview the labels, and work out what
    /// the run would do; nothing is stored. Status lines go to `status`.
    ///
    /// # Errors
    ///
    /// Returns the refusal ([`quack_core::classify::Error`]), the
    /// models' or the database's error.
    pub async fn prepare(
        &self,
        config: &Config,
        db: &Writer,
        status: &mut impl Write,
        control: RunControl<'_>,
    ) -> Result<Prepared> {
        let decision = DecisionModel::from_config(config)
            .await?
            .ok_or(Error::NoDecisionModel)?;
        let drafter = ChatDrafter::from_config(config);
        let (table, sentence) = (self.table().to_owned(), self.sentence());
        let drafting = self.drafting(db).await?;
        if let (Some(origin), Some(drafter)) = (drafting, &drafter) {
            writeln!(
                status,
                "{} questions with {} from {SAMPLE_ROWS} sample rows; this can take a minute or \
                 two...",
                if origin == DraftOrigin::Revised {
                    "Revising"
                } else {
                    "Drafting"
                },
                drafter.label()
            )?;
            status.flush()?;
        }
        let earlier = db
            .run({
                let table = table.clone();
                move |db| Ok(Draft::approved(db, &table)?.is_some())
            })
            .await?;
        let context = DraftContext {
            db,
            decision: &decision,
            drafter: drafter.as_ref().map(|d| -> &dyn Drafter { d }),
        };
        let draft = Draft::prepare(&context, &table, sentence.as_deref()).await?;
        let rows = if self.all { Rows::All } else { Rows::Missing };
        let outline = draft.outline(db, &decision, rows).await?;
        let wanted = self
            .preview
            .or_else(|| (!self.yes).then_some(PREVIEW_ROWS))
            .filter(|_| outline.remaining > 0 || rows == Rows::All);
        let preview = match wanted {
            Some(n) => Some(
                draft
                    .preview(db, &decision, n, rows, Waiting::Job, control)
                    .await?,
            ),
            None => None,
        };
        let outline = match &preview {
            Some(preview) => outline.estimated_from(preview),
            None => outline,
        };
        Ok(Prepared {
            draft,
            outline,
            preview,
            rows,
            asked: sentence,
            earlier,
            decision: decision.label().to_owned(),
        })
    }
}

impl Prepared {
    /// Whether the output already holds every row, and nothing was asked
    /// to label again.
    #[must_use]
    pub fn nothing_to_label(&self) -> bool {
        self.outline.remaining == 0 && self.outline.effect == Effect::AddsRows
    }

    /// Whether the run replaces the labels already there.
    #[must_use]
    pub fn relabels(&self) -> bool {
        matches!(self.outline.effect, Effect::ReplacesLabels { .. })
    }

    /// What is said when there is nothing to label.
    #[must_use]
    pub fn nothing_to_label_note(&self) -> String {
        format!(
            "Nothing to label: {} has every row of {}. --all labels every row again.",
            self.outline.output_table, self.outline.source_table
        )
    }

    /// The draft that a yes runs.
    #[must_use]
    pub const fn draft(&self) -> &Draft {
        &self.draft
    }

    /// The question a yes answers.
    #[must_use]
    pub fn question(&self) -> String {
        let outline = &self.outline;
        let count = Thousands(outline.remaining);
        let questions = outline.questions.len();
        let what = match outline.effect {
            Effect::NewTable => format!(
                "Label all {count} rows into {} with {questions} questions?",
                outline.output_table
            ),
            Effect::AddsRows => format!(
                "Label {count} new rows into {} with {questions} questions?",
                outline.output_table
            ),
            Effect::ReplacesLabels { .. } => format!(
                "Label all {count} rows into {} again with {questions} questions?",
                outline.output_table
            ),
        };
        match outline.estimate() {
            Some(estimate) => format!("{what} {}.", estimate.sentence()),
            None => what,
        }
    }

    /// What is said when the answer is no.
    #[must_use]
    pub fn declined(&self) -> String {
        let table = &self.outline.source_table;
        if self.earlier {
            format!(
                "Not labelled; these questions were not kept. The same sentence drafts them \
                 again; quack classify {table} uses the questions approved before."
            )
        } else {
            String::from("Not labelled; these questions were not kept.")
        }
    }

    /// The table, the questions, and the preview, as the screen shows them.
    ///
    /// # Errors
    ///
    /// Returns the error from writing to `out`.
    pub fn show(&self, out: &mut impl Write) -> Result<()> {
        let draft = &self.draft;
        writeln!(out, "{}", draft.header())?;
        match draft.origin {
            DraftOrigin::Drafted | DraftOrigin::Revised => {
                let verb = if draft.origin == DraftOrigin::Drafted {
                    "Drafted"
                } else {
                    "Revised"
                };
                writeln!(out, "{verb} from {} sample rows:\n", draft.sample_rows)?;
            }
            DraftOrigin::Reused if self.asked.is_some() => {
                writeln!(
                    out,
                    "Same question as last time; using the stored questions.\n"
                )?;
            }
            DraftOrigin::Reused => writeln!(out, "{}", Self::approved_line(draft))?,
            DraftOrigin::Given => {}
        }
        writeln!(out, "{}", draft.set.block(false))?;
        if let Some(preview) = &self.preview {
            writeln!(out, "\nPreview of {} rows:", preview.compact.rows.len())?;
            QueryFormat::Table.write(&preview.compact, out)?;
            writeln!(out, "{preview}")?;
            if preview.cut > 0 {
                writeln!(
                    out,
                    "{} of {} cut to fit the model (*)",
                    preview.cut,
                    preview.compact.rows.len()
                )?;
            }
        }
        if let Effect::ReplacesLabels { because } = self.outline.effect {
            writeln!(out, "\n{}", Self::relabel_note(because))?;
        }
        Ok(())
    }

    /// `Questions approved 2026-10-09 ("the sentence"):`
    fn approved_line(draft: &Draft) -> String {
        let when = draft
            .approved_at
            .as_deref()
            .map(|at| at.chars().take(10).collect::<String>())
            .unwrap_or_default();
        match &draft.set.sentence {
            Some(sentence) => format!("Questions approved {when} (\"{sentence}\"):"),
            None => format!("Questions approved {when}:"),
        }
    }

    /// Why every row is labelled again, and that the labels serve meanwhile.
    fn relabel_note(because: RelabelReason) -> String {
        const SERVE: &str = "the current labels serve until the new ones are complete.";
        match because {
            RelabelReason::Asked => format!("Every row is labelled again, as asked; {SERVE}"),
            RelabelReason::QuestionsChanged => {
                format!("The questions changed, so every row is labelled again; {SERVE}")
            }
            RelabelReason::ModelChanged => format!(
                "The decision model's weights changed (ollama pull), so every row is labelled \
                 again; {SERVE}"
            ),
            RelabelReason::KeyChanged => {
                format!("The key changed, so every row is labelled again; {SERVE}")
            }
        }
    }

    /// Label the rows, recording the questions as approved.
    ///
    /// # Errors
    ///
    /// Returns the refusal, the model's or the database's error, or an I/O
    /// error from `out`.
    pub async fn run(
        self,
        config: &Config,
        db: &Writer,
        out: &mut impl Write,
        control: RunControl<'_>,
    ) -> Result<()> {
        let decision = DecisionModel::from_config(config)
            .await?
            .ok_or(Error::NoDecisionModel)?;
        writeln!(
            out,
            "Labelling {} rows of {} with {} ({} questions: {})",
            Thousands(self.outline.remaining),
            self.outline.source_table,
            self.decision,
            self.outline.questions.len(),
            self.outline.question_names()
        )?;
        out.flush()?;
        let run = self
            .draft
            .run(
                Labelling {
                    db,
                    decision: &decision,
                    started_by: None,
                    run_id: RunId::generate(),
                    waiting: Waiting::Job,
                    control,
                },
                self.rows,
            )
            .await?;
        writeln!(out, "{run}")?;
        Ok(())
    }
}

/// The questions last approved for a table, with the labels they made.
#[derive(Serialize)]
struct Shown {
    draft: Draft,
    labels: Option<Labels>,
}

/// The table of labels and the run that made it.
#[derive(Serialize)]
struct Labels {
    table: String,
    rows: u64,
    last_run: classify::Run,
    /// Whether the labels were made with other questions than the last
    /// approved: a relabel was stopped.
    earlier_questions: bool,
}

impl Shown {
    /// Read what `table` has approved.
    fn read(db: &WorkspaceDb, table: &str) -> CoreResult<Option<Self>> {
        let Some(draft) = Draft::approved(db, table)? else {
            return Ok(None);
        };
        let Some(last_run) = classify::Run::last_approved(db, &draft.table)? else {
            return Ok(None);
        };
        let labels = if db
            .list_tables()?
            .iter()
            .any(|t| t.eq_ignore_ascii_case(&draft.output_table))
        {
            let in_force = classify::Run::in_force(db, &draft.output_table)?;
            Some(Labels {
                table: draft.output_table.clone(),
                rows: u64::try_from(db.count_rows(&draft.output_table)?).unwrap_or(0),
                earlier_questions: in_force
                    .as_ref()
                    .is_some_and(|made| made.questions != last_run.questions),
                last_run,
            })
        } else {
            None
        };
        Ok(Some(Self { draft, labels }))
    }

    fn write(&self, out: &mut impl Write) -> Result<()> {
        let draft = &self.draft;
        writeln!(out, "{}", draft.header())?;
        let when = draft
            .approved_at
            .as_deref()
            .map(|at| at.chars().take(16).collect::<String>())
            .unwrap_or_default();
        let by = draft
            .drafted_by_model
            .as_deref()
            .map(|model| format!("; drafted by {model}"))
            .unwrap_or_default();
        match &draft.set.sentence {
            Some(sentence) => writeln!(out, "Asked: \"{sentence}\" (approved {when}{by})")?,
            None => writeln!(out, "Approved {when}{by}")?,
        }
        writeln!(out, "\n{}", draft.set.block(true))?;
        if !draft.columns_gone.is_empty() {
            writeln!(
                out,
                "\nThe table no longer has {}; say a sentence to draft new questions.",
                draft.columns_gone.join(", ")
            )?;
        }
        if let Some(labels) = &self.labels {
            writeln!(
                out,
                "\nLabels: {}, {} rows; last run {} {} with {}{}.",
                labels.table,
                Thousands(labels.rows),
                labels
                    .last_run
                    .started_at
                    .chars()
                    .take(10)
                    .collect::<String>(),
                labels.last_run.status,
                labels.last_run.model,
                if labels.earlier_questions {
                    "; made with earlier questions, so the next run labels every row again"
                } else {
                    ""
                }
            )?;
        }
        Ok(())
    }
}

impl ClassifyCommand {
    /// What a job running it is called.
    #[must_use]
    pub fn label(&self) -> String {
        match &self.action {
            Some(ClassifyAction::List { .. }) => String::from("classify list"),
            Some(ClassifyAction::Show { table, .. }) => format!("classify show {table}"),
            None => format!("classify {}", self.run.table()),
        }
    }

    /// The arguments of a labelling, when this is one.
    #[must_use]
    pub fn labelling(&self) -> Option<&ClassifyArgs> {
        self.action.is_none().then_some(&self.run)
    }

    /// List the runs, show a table's questions, or label the table:
    /// draft, preview, ask, run. Status lines go to stderr; the report to
    /// `out`.
    ///
    /// # Errors
    ///
    /// Returns the refusal, the model's or the database's error, or an I/O
    /// error from `out`.
    pub async fn run(
        self,
        config: &Config,
        db: &Writer,
        confirm: Confirm,
        out: &mut impl Write,
        control: RunControl<'_>,
    ) -> Result<()> {
        let args = match self.action {
            Some(ClassifyAction::List { format }) => return Self::list(db, format, out).await,
            Some(ClassifyAction::Show { table, format }) => {
                return Self::show(db, &table, format, out).await;
            }
            None => self.run,
        };
        let prepared = args
            .prepare(config, db, &mut std::io::stderr(), control)
            .await?;
        if prepared.nothing_to_label() {
            writeln!(out, "{}", prepared.nothing_to_label_note())?;
            return Ok(());
        }
        prepared.show(out)?;
        if args.preview.is_some() {
            return Ok(());
        }
        if !args.yes {
            let question = prepared.question();
            let go = if prepared.relabels() {
                confirm.ask_to_drop(false, out, &question)?
            } else {
                confirm.ask(out, &question, Some("--yes"))?
            };
            if !go {
                writeln!(out, "{}", prepared.declined())?;
                return Ok(());
            }
        }
        prepared.run(config, db, out, control).await
    }

    async fn show(
        db: &Writer,
        table: &str,
        format: TextOrJson,
        out: &mut impl Write,
    ) -> Result<()> {
        let wanted = table.to_owned();
        let shown = db.run(move |db| Shown::read(db, &wanted)).await?;
        match (shown, format) {
            (Some(shown), TextOrJson::Json) => {
                writeln!(out, "{}", serde_json::to_string_pretty(&shown)?)?;
            }
            (Some(shown), TextOrJson::Text) => shown.write(out)?,
            (None, TextOrJson::Json) => writeln!(out, "null")?,
            (None, TextOrJson::Text) => {
                writeln!(out, "{table} has no approved questions yet.")?;
            }
        }
        Ok(())
    }

    /// The runs that labelled tables, newest first, as text or JSON.
    async fn list(db: &Writer, format: TextOrJson, out: &mut impl Write) -> Result<()> {
        let runs = db
            .run(|db| classify::Run::list(db, LISTED_RUNS))
            .await?
            .runs;
        format.write_rows(
            out,
            &runs,
            "No table has been labelled yet; label one with `quack classify TABLE \"what you want to know about each row\"`.",
            |out, run| {
                writeln!(
                    out,
                    "{}  {} -> {}  {}  {} labelled, {} cut, {} empty, {} skipped{}",
                    run.started_at,
                    run.source_table,
                    run.output_table,
                    run.status,
                    run.labelled,
                    run.cut,
                    run.empty,
                    run.skipped,
                    run.error
                        .as_deref()
                        .map(|e| format!(": {e}"))
                        .unwrap_or_default()
                )
            },
        )
    }
}

#[cfg(test)]
mod tests;
