//! `quack classify`: label a table's text with a decision model (issue
//! #472), or list the runs that did. The command line and the terminal's
//! `/classify` run the same arguments.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Subcommand;
use quack_core::classify::{
    Classification, ClassificationPreview, ClassificationRun, Labelling, MAX_QUESTION_SET_BYTES,
    QuestionSet, Rows, Waiting,
};
use quack_core::config::Config;
use quack_core::error::Error;
use quack_core::ids::RunId;
use quack_core::llm::decision::DecisionModel;
use quack_core::progress::RunControl;
use quack_core::storage::writer::Writer;

use crate::args::QueryFormat;
use crate::text_or_json::TextOrJson;

/// Runs `list` shows at most.
const LISTED_RUNS: u32 = 100;

/// What `quack classify` does besides labelling a table.
#[derive(Subcommand)]
pub enum ClassifyAction {
    /// List the runs that labelled tables, newest first
    List {
        /// `json` prints one JSON object per run
        #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
        format: TextOrJson,
    },
}

/// The arguments of a labelling.
#[derive(clap::Args)]
pub struct ClassifyArgs {
    /// The table whose rows are labelled (a table named `list` is labelled
    /// through the API or the agent: `list` is the subcommand here)
    #[arg(required = true)]
    table: Option<String>,
    /// The columns whose text the model reads (one to eight), comma-separated
    #[arg(
        long = "text",
        required = true,
        value_delimiter = ',',
        value_name = "COLUMN"
    )]
    text: Vec<String>,
    /// The JSON file of questions: `{"name": ..., "questions": {...}}`
    #[arg(long, required = true, value_name = "FILE")]
    questions: Option<PathBuf>,
    /// The column that identifies a row; without it an `id` column, or one
    /// ending in `_id`, whose values are all different is used
    #[arg(long, value_name = "COLUMN")]
    key: Option<String>,
    /// Label only the first N rows (1 to 100), show them, and write nothing
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..=100))]
    preview: Option<u32>,
    /// Label every row again; the old labels serve until the run completes
    #[arg(long)]
    all: bool,
    /// `json` prints the preview or the run as one JSON document
    #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
    format: TextOrJson,
}

/// `quack classify TABLE ...` or `quack classify list`.
#[derive(clap::Args)]
#[command(args_conflicts_with_subcommands = true, subcommand_negates_reqs = true)]
pub struct ClassifyCommand {
    #[command(subcommand)]
    action: Option<ClassifyAction>,
    #[command(flatten)]
    run: ClassifyArgs,
}

impl ClassifyArgs {
    /// The request these arguments make, reading the question file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or is not a question
    /// set.
    pub fn classification(&self) -> Result<Classification> {
        let file = self
            .questions
            .as_deref()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let mut text = String::new();
        File::open(&file)
            .and_then(|file| {
                file.take(
                    u64::try_from(MAX_QUESTION_SET_BYTES)
                        .unwrap_or(u64::MAX)
                        .saturating_add(1),
                )
                .read_to_string(&mut text)
            })
            .with_context(|| format!("cannot read {}", file.display()))?;
        let question_set = QuestionSet::parse(&text)
            .with_context(|| format!("{} is not a question set", file.display()))?;
        Ok(Classification {
            table: self.table.clone().unwrap_or_default(),
            text_columns: self.text.clone(),
            key: self.key.clone(),
            question_set,
            rows: if self.all { Rows::All } else { Rows::Missing },
        })
    }

    /// A job's label.
    #[must_use]
    pub fn label(&self) -> String {
        format!("classify {}", self.table.as_deref().unwrap_or_default())
    }
}

impl ClassifyArgs {
    /// The rows of a preview as a table under the key they join on, then how
    /// they came out.
    fn show(&self, preview: &ClassificationPreview, out: &mut impl Write) -> Result<()> {
        match self.format {
            TextOrJson::Json => writeln!(out, "{}", serde_json::to_string_pretty(preview)?)?,
            TextOrJson::Text => {
                writeln!(out, "Key: {}", preview.key_column)?;
                QueryFormat::Table.write(&preview.result, out)?;
                writeln!(out, "{preview}")?;
            }
        }
        Ok(())
    }
}

impl ClassifyCommand {
    /// The runs that labelled tables, newest first, as text or JSON.
    async fn list(db: &Writer, format: TextOrJson, out: &mut impl Write) -> Result<()> {
        let runs = db
            .run(|db| ClassificationRun::list(db, LISTED_RUNS))
            .await?
            .runs;
        format.write_rows(
            out,
            &runs,
            "No table has been labelled yet; label one with `quack classify TABLE --text COLUMN --questions FILE`.",
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

    /// What a job running it is called.
    #[must_use]
    pub fn label(&self) -> String {
        match &self.action {
            Some(ClassifyAction::List { .. }) => String::from("classify list"),
            None => self.run.label(),
        }
    }

    /// List the runs, preview the labels, or label the table, reporting to
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
        out: &mut impl Write,
        control: RunControl<'_>,
    ) -> Result<()> {
        let args = match self.action {
            Some(ClassifyAction::List { format }) => return Self::list(db, format, out).await,
            None => self.run,
        };
        let request = args.classification()?;
        let decision = DecisionModel::from_config(config)
            .await?
            .ok_or(Error::NoDecisionModel)?;
        if let Some(n) = args.preview {
            let preview = request
                .preview(db, &decision, n, Waiting::Job, control)
                .await?;
            return args.show(&preview, out);
        }
        let outline = request.outline(db, &decision).await?;
        if args.format == TextOrJson::Text {
            writeln!(
                out,
                "Labelling {} rows of {} (key {}) with {} ({} questions: {})",
                outline.remaining,
                outline.source_table,
                outline.key_column,
                decision.label(),
                outline.questions.len(),
                outline.question_names()
            )?;
            out.flush()?;
        }
        let run = request
            .run(Labelling {
                db,
                decision: &decision,
                started_by: None,
                run_id: RunId::generate(),
                waiting: Waiting::Job,
                control,
            })
            .await?;
        args.format.write(out, &run)?;
        if args.format == TextOrJson::Text {
            writeln!(out)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests assert on values they have just built"
)]
mod tests {
    use clap::Parser;
    use quack_core::embedding::Dimension;
    use quack_core::llm::egress::Egress;
    use quack_core::storage::workspace::WorkspaceDb;
    use quack_testkit::DecisionStub;

    use super::*;

    #[derive(Parser)]
    struct Line {
        #[command(flatten)]
        command: ClassifyCommand,
    }

    const QUESTIONS: &str = r#"{"name": "triage", "questions": {
        "department": {"type": "choice", "instructions": "Which department?",
                       "criteria": {"billing": "Invoices", "technical": null, "none": null}},
        "churn": {"type": "noul", "instructions": "Will they cancel?"}}}"#;

    fn parse(words: &[&str]) -> Result<ClassifyCommand, clap::Error> {
        let mut line = vec!["classify"];
        line.extend_from_slice(words);
        Line::try_parse_from(line).map(|l| l.command)
    }

    #[test]
    fn the_arguments_of_a_run_and_of_list_parse() {
        let run = parse(&[
            "tickets",
            "--text",
            "subject,body",
            "--questions",
            "q.json",
            "--preview",
            "20",
            "--all",
        ]);
        assert!(run.is_ok_and(|c| c.action.is_none()
            && c.run.text == ["subject", "body"]
            && c.run.preview == Some(20)
            && c.run.all));
        assert!(parse(&["list"]).is_ok_and(|c| c.action.is_some()));
        assert!(parse(&["list", "--format", "json"]).is_ok());
    }

    #[test]
    fn a_run_needs_a_table_text_columns_and_questions_and_a_sane_preview() {
        assert!(parse(&[]).is_err());
        assert!(parse(&["tickets", "--questions", "q.json"]).is_err());
        assert!(parse(&["tickets", "--text", "subject"]).is_err());
        for n in ["0", "101"] {
            let line = [
                "tickets",
                "--text",
                "subject",
                "--questions",
                "q.json",
                "--preview",
                n,
            ];
            assert!(parse(&line).is_err(), "--preview {n}");
        }
    }

    #[tokio::test]
    async fn a_preview_a_run_and_the_listing_print_what_they_found() {
        let stub = DecisionStub::start().await;
        let config = Config::parse(&format!(
            "[providers.local]\ntype = \"ollama\"\nbase_url = \"{}\"\nmax_retries = 0\n\
             [decision]\nmodel = \"local/laya\"\n",
            stub.base_url()
        ))
        .unwrap();
        let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
        db.execute_statement(
            "CREATE TABLE tickets AS SELECT range AS id, \
             CASE WHEN range % 2 = 0 THEN 'billing' ELSE 'technical' END AS subject, \
             'we will cancel' AS body FROM range(5)",
        )
        .unwrap();
        let db = Writer::spawn(db).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("triage.json");
        std::fs::write(&file, QUESTIONS).unwrap();
        let file = file.display().to_string();
        let base = [
            "tickets",
            "--text",
            "subject,body",
            "--questions",
            file.as_str(),
        ];
        let control = RunControl::unobserved;

        Egress::scope(Some(Egress::NoWorkspace), async {
            let mut out = Vec::new();
            let preview = parse(&[base.as_slice(), &["--preview", "2"]].concat()).unwrap();
            preview
                .run(&config, &db, &mut out, control())
                .await
                .unwrap();
            let printed = String::from_utf8_lossy(&out).into_owned();
            assert!(printed.starts_with("Key: id\n"), "{printed}");
            assert!(printed.contains("department_confidence"), "{printed}");
            assert!(printed.contains("2 rows labelled"), "{printed}");
            assert!(
                printed.contains("Run without --preview to label 5 rows."),
                "{printed}"
            );

            let mut out = Vec::new();
            let run = parse(&base).unwrap();
            run.run(&config, &db, &mut out, control()).await.unwrap();
            assert_eq!(
                String::from_utf8_lossy(&out),
                "Labelling 5 rows of tickets (key id) with local/laya (2 questions: department, churn)\n\
                 Wrote tickets_triage: 5 rows labelled, 0 cut to fit the model, 0 empty, 0 skipped.\n"
            );

            let mut out = Vec::new();
            let again = parse(&[base.as_slice(), &["--format", "json"]].concat()).unwrap();
            again.run(&config, &db, &mut out, control()).await.unwrap();
            let json: serde_json::Value = serde_json::from_slice(&out).unwrap();
            assert_eq!(json.get("labelled"), Some(&serde_json::json!(0)));
            assert_eq!(json.get("status"), Some(&serde_json::json!("completed")));

            let mut out = Vec::new();
            let list = parse(&["list"]).unwrap();
            list.run(&config, &db, &mut out, control()).await.unwrap();
            let listed = String::from_utf8_lossy(&out).into_owned();
            assert_eq!(listed.lines().count(), 2, "{listed}");
            assert!(
                listed.contains("tickets -> tickets_triage  completed"),
                "{listed}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn without_a_decision_model_it_says_what_to_set() {
        let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
        let db = Writer::spawn(db).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("triage.json");
        std::fs::write(&file, QUESTIONS).unwrap();
        let file = file.display().to_string();
        let command =
            parse(&["tickets", "--text", "subject", "--questions", file.as_str()]).unwrap();
        let refused = Egress::scope(Some(Egress::NoWorkspace), async {
            command
                .run(
                    &Config::default(),
                    &db,
                    &mut Vec::new(),
                    RunControl::unobserved(),
                )
                .await
        })
        .await
        .unwrap_err();
        assert!(
            refused.to_string().contains("[decision].model"),
            "{refused}"
        );
    }
}
