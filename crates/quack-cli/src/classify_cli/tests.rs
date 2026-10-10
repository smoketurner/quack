#![expect(
    clippy::unwrap_used,
    reason = "tests assert on values they have just built"
)]
#![expect(
    clippy::indexing_slicing,
    reason = "serde_json::Value indexing yields Null for a missing key, never a panic"
)]

use clap::Parser;
use quack_core::embedding::Dimension;
use quack_core::llm::egress::Egress;
use quack_core::storage::workspace::WorkspaceDb;
use quack_testkit::{DecisionStub, Reply, ScriptedOllama};

use super::*;

#[derive(Parser)]
struct Line {
    #[command(flatten)]
    command: ClassifyCommand,
}

fn parse(words: &[&str]) -> Result<ClassifyCommand, clap::Error> {
    let mut line = vec!["classify"];
    line.extend_from_slice(words);
    Line::try_parse_from(line).map(|l| l.command)
}

const SENTENCE: &str = "which department, and will they cancel";

/// What the chat model drafts for the tickets.
const DRAFT: &str = r#"{"text_columns": ["subject"], "questions": [
    {"name": "department", "type": "choice", "instructions": "Which department?",
     "options": [{"label": "billing", "description": ""}, {"label": "technical", "description": ""}],
     "levels": []},
    {"name": "churn", "type": "noul", "instructions": "Will they cancel?",
     "options": [], "levels": []}]}"#;

/// The same questions with one more, so the next draft revises them.
const REVISED: &str = r#"{"text_columns": ["subject"], "questions": [
    {"name": "department", "type": "choice", "instructions": "Which department?",
     "options": [{"label": "billing", "description": ""}, {"label": "technical", "description": ""}],
     "levels": []},
    {"name": "urgent", "type": "noul", "instructions": "Is it urgent?",
     "options": [], "levels": []}]}"#;

/// A workspace with a `tickets` table of three rows, the decision model and
/// the chat model that draft [`DRAFT`] and then [`REVISED`] twice.
struct Fixture {
    config: Config,
    db: Writer,
    _stub: DecisionStub,
    _chat: ScriptedOllama,
}

impl Fixture {
    async fn new() -> Self {
        let stub = DecisionStub::start().await;
        let chat = ScriptedOllama::serve(vec![
            Reply::Text(DRAFT),
            Reply::Text(REVISED),
            Reply::Text(REVISED),
        ])
        .await
        .unwrap();
        let config = chat
            .config_with(&format!(
                "[providers.local]\ntype = \"ollama\"\nbase_url = \"{}\"\nmax_retries = 0\n\
                 [decision]\nmodel = \"local/laya\"\n",
                stub.base_url()
            ))
            .unwrap();
        let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
        db.execute_statement(
            "CREATE TABLE tickets AS SELECT range AS id, \
             CASE WHEN range % 2 = 0 THEN 'billing' ELSE 'technical' END AS subject \
             FROM range(3)",
        )
        .unwrap();
        Self {
            config,
            db: Writer::spawn(db).unwrap(),
            _stub: stub,
            _chat: chat,
        }
    }

    /// `words` as a command, run with `confirm`: what it printed, or why not.
    async fn run(&self, words: &[&str], confirm: Confirm) -> Result<String> {
        let command = parse(words).unwrap();
        let mut out = Vec::new();
        Egress::scope(Some(Egress::NoWorkspace), async {
            command
                .run(
                    &self.config,
                    &self.db,
                    confirm,
                    &mut out,
                    RunControl::unobserved(),
                )
                .await
        })
        .await?;
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    async fn runs(&self) -> Vec<classify::Run> {
        self.db
            .run(|db| classify::Run::list(db, 100))
            .await
            .unwrap()
            .runs
    }
}

#[test]
fn the_arguments_of_a_labelling_list_and_show_parse() {
    let run = parse(&["tickets", "which department", "-y", "--all"]).unwrap();
    let args = run.labelling().unwrap();
    assert_eq!(args.table(), "tickets");
    assert_eq!(args.sentence().as_deref(), Some("which department"));
    assert!(args.yes());
    let preview = parse(&["tickets", "--preview", "20"]).unwrap();
    assert_eq!(preview.labelling().unwrap().preview_only(), Some(20));
    assert_eq!(preview.labelling().unwrap().sentence(), None);
    assert!(parse(&["list"]).is_ok_and(|c| c.labelling().is_none()));
    assert!(parse(&["list", "--format", "json"]).is_ok());
    assert!(parse(&["show", "tickets"]).is_ok_and(|c| c.labelling().is_none()));
}

#[test]
fn a_labelling_needs_a_table_and_the_old_flags_are_gone() {
    assert!(parse(&[]).is_err());
    assert!(parse(&["tickets", "--preview", "0"]).is_err());
    assert!(parse(&["tickets", "--preview", "101"]).is_err());
    for removed in ["--text", "--questions", "--key", "--format"] {
        assert!(parse(&["tickets", removed, "x"]).is_err(), "{removed}");
    }
    assert!(parse(&["show"]).is_err());
}

#[tokio::test]
async fn a_sentence_is_drafted_previewed_and_labelled_on_a_yes() {
    let fixture = Fixture::new().await;
    let words = ["tickets", SENTENCE];
    let printed = fixture.run(&words, Confirm::Assume).await.unwrap();
    assert!(
        printed.starts_with(
            "tickets: 3 rows, key id, text in subject.\nDrafted from 3 sample rows:\n"
        ),
        "{printed}"
    );
    assert!(
        printed.contains("  department   choice   billing \u{b7} technical\n"),
        "{printed}"
    );
    assert!(printed.contains("\nPreview of 3 rows:\n"), "{printed}");
    assert!(printed.contains("department (p)"), "{printed}");
    assert!(printed.contains("3 rows in "), "{printed}");
    assert!(
        printed.contains(
            "Labelling 3 rows of tickets with local/laya (2 questions: department, churn)\n"
        ),
        "{printed}"
    );
    assert!(
        printed.contains("Wrote tickets_labels: 3 rows labelled, 0 cut to fit the model"),
        "{printed}"
    );
    let runs = fixture.runs().await;
    assert_eq!(runs.len(), 1);
    assert_eq!(
        runs.first().and_then(|run| run.sentence.as_deref()),
        Some(SENTENCE)
    );
}

#[tokio::test]
async fn without_a_terminal_nothing_is_labelled_or_kept_and_the_question_is_said() {
    let fixture = Fixture::new().await;
    let printed = fixture
        .run(&["tickets", SENTENCE], Confirm::Ask)
        .await
        .unwrap();
    assert!(
        printed.contains(
            "Label all 3 rows into tickets_labels with 2 questions? Under a minute. No terminal \
             to answer on; --yes goes ahead.\nNot labelled; these questions were not kept.\n"
        ),
        "{printed}"
    );
    assert!(fixture.runs().await.is_empty());
    let tables = fixture.db.run(WorkspaceDb::list_tables).await.unwrap();
    assert_eq!(tables, ["tickets"]);
}

#[tokio::test]
async fn yes_labels_without_a_preview_and_preview_stops_without_asking() {
    let fixture = Fixture::new().await;
    let stopped = fixture
        .run(&["tickets", "--preview", "2", SENTENCE], Confirm::Ask)
        .await
        .unwrap();
    assert!(stopped.contains("\nPreview of 2 rows:\n"), "{stopped}");
    assert!(!stopped.contains("Label all"), "{stopped}");
    assert!(!stopped.contains("Labelling"), "{stopped}");
    assert!(fixture.runs().await.is_empty());

    let ran = fixture
        .run(&["tickets", "-y", SENTENCE], Confirm::Ask)
        .await
        .unwrap();
    assert!(!ran.contains("Preview of"), "{ran}");
    assert!(ran.contains("Drafted from 3 sample rows:"), "{ran}");
    assert!(
        ran.contains("Wrote tickets_labels: 3 rows labelled"),
        "{ran}"
    );
}

#[tokio::test]
async fn a_rerun_uses_the_approved_questions_and_a_new_sentence_revises_them() {
    let fixture = Fixture::new().await;
    fixture
        .run(&["tickets", "-y", SENTENCE], Confirm::Ask)
        .await
        .unwrap();

    let nothing = fixture.run(&["tickets"], Confirm::Ask).await.unwrap();
    assert!(
        nothing.contains("Nothing to label: tickets_labels has every row of tickets. --all labels every row again."),
        "{nothing}"
    );

    let same = fixture
        .run(
            &["tickets", "--all", "--preview", "2", SENTENCE],
            Confirm::Ask,
        )
        .await
        .unwrap();
    assert!(
        same.contains("Same question as last time; using the stored questions.\n"),
        "{same}"
    );
    assert!(
        same.contains("Every row is labelled again, as asked; the current labels serve until"),
        "{same}"
    );

    // A different sentence revises the questions, which labels every row
    // again, and nobody is there to approve dropping the labels.
    let revised = fixture
        .run(&["tickets", "and is it urgent"], Confirm::Ask)
        .await;
    let error = revised.unwrap_err().to_string();
    assert!(
        error.contains("Nobody to ask here, so nothing was dropped; --yes goes ahead."),
        "{error}"
    );
    assert!(
        error.contains("Label all 3 rows into tickets_labels again with 2 questions?"),
        "{error}"
    );
    assert_eq!(
        fixture.runs().await.len(),
        1,
        "no run began for the refused revision"
    );

    let yes = fixture
        .run(&["tickets", "-y", "and is it urgent"], Confirm::Ask)
        .await
        .unwrap();
    assert!(yes.contains("Revised from 3 sample rows:"), "{yes}");
    assert!(
        yes.contains("The questions changed, so every row is labelled again"),
        "{yes}"
    );
    let newest = fixture.runs().await.remove(0);
    assert_eq!((newest.rows, newest.labelled), (Rows::All, 3));
}

#[tokio::test]
async fn a_table_without_questions_says_what_to_say() {
    let fixture = Fixture::new().await;
    let error = fixture.run(&["tickets"], Confirm::Ask).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("tickets has no questions yet; say what you want to know"),
        "{error}"
    );
    let shown = fixture
        .run(&["show", "tickets"], Confirm::Ask)
        .await
        .unwrap();
    assert_eq!(shown, "tickets has no approved questions yet.\n");
}

#[tokio::test]
async fn show_prints_the_approved_questions_and_the_labels_they_made() {
    let fixture = Fixture::new().await;
    fixture
        .run(&["tickets", "-y", SENTENCE], Confirm::Ask)
        .await
        .unwrap();
    let text = fixture
        .run(&["show", "tickets"], Confirm::Ask)
        .await
        .unwrap();
    assert!(
        text.starts_with("tickets: 3 rows, key id, text in subject.\nAsked: \"which department"),
        "{text}"
    );
    assert!(text.contains("drafted by scripted/model)"), "{text}");
    assert!(
        text.contains("  department   choice   billing \u{b7} technical"),
        "{text}"
    );
    assert!(
        text.contains("\nLabels: tickets_labels, 3 rows; last run "),
        "{text}"
    );
    let json = fixture
        .run(&["show", "tickets", "--format", "json"], Confirm::Ask)
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["draft"]["set"]["sentence"], SENTENCE);
    assert_eq!(value["labels"]["rows"], 3);

    let list = fixture.run(&["list"], Confirm::Ask).await.unwrap();
    assert!(
        list.contains("tickets -> tickets_labels  completed"),
        "{list}"
    );
}

#[tokio::test]
async fn without_a_decision_model_it_says_what_to_set() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    let db = Writer::spawn(db).unwrap();
    let command = parse(&["tickets", SENTENCE]).unwrap();
    let refused = Egress::scope(Some(Egress::NoWorkspace), async {
        command
            .run(
                &Config::default(),
                &db,
                Confirm::Assume,
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
