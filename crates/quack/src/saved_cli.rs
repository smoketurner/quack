//! `quack saved`: a question a team asks repeatedly, saved once with the
//! read statements its answer ran, re-run without the model, each run
//! saying whether the data changed. cron is the scheduler: `quack saved
//! run NAME --exit-code` exits 5 when the result changed.

use std::io::{IsTerminal, Write};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use clap::Subcommand;
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::config::Config;
use quack_core::ids::SessionId;
use quack_core::saved::{self, Answer, RunStatus, SavedQuestion, SavedRun, StatementRun};
use quack_core::storage::control::ResourceKind;
use quack_core::storage::sessions;
use quack_core::storage::workspace::QueryResults;
use quack_core::storage::writer::Writer;

use crate::QueryFormat;
use crate::TITLE_GRACE;
use crate::print::{AnswerTo, PrintTurn};
use crate::text_or_json::TextOrJson;
use quack_core::llm::titles::SessionTitler;

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum SavedAction {
    /// List the saved questions
    List {
        /// `json` prints one JSON object per saved question
        #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
        format: TextOrJson,
    },
    /// Save an answer under a name: the question it answered and the read
    /// statements it ran, to re-run without the model
    Add {
        name: String,
        /// The session the answer is in (prefixes accepted); `quack -p -f
        /// json` prints it as `session_id`
        #[arg(long, value_name = "SESSION_ID")]
        from_session: Option<String>,
        /// The answer's message number in the session; the last answer
        /// otherwise
        #[arg(long, value_name = "N")]
        message: Option<i64>,
    },
    /// Run the saved SQL without the model and say whether the result
    /// changed since the run before
    Run {
        name: String,
        /// Ask the model again in a new session, writes denied, and pin the
        /// SQL its answer ran; the next run compares with this one
        #[arg(long)]
        refresh: bool,
        /// Exit 5 when the result changed
        #[arg(long)]
        exit_code: bool,
        /// How to print the run. Default: table on a terminal, ndjson when
        /// piped
        #[arg(short = 'f', long, value_enum)]
        format: Option<QueryFormat>,
    },
    /// The pinned SQL and the last run
    Show {
        name: String,
        /// `json` prints the saved question and its last run as one document
        #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
        format: TextOrJson,
    },
    /// Remove a saved question and its runs
    Remove { name: String },
}

/// What `--refresh` needs beyond the writer: the handles a print-mode
/// turn takes. The command line has them; the terminal refreshes through
/// its own chat instead.
pub(crate) struct Model {
    pub db: SharedDb,
    pub reader_db: ReaderDb,
    /// Full tool inputs on stderr.
    pub verbose: bool,
}

/// Run `action` against the workspace behind `db`. `session` is where
/// `add` looks without `--from-session` (the terminal's current session).
/// Returns the run for `run`, so the caller can set the exit status.
///
/// # Errors
///
/// Returns an error when the saved question, session, or message does
/// not exist, the answer cannot be saved, or a refresh fails.
pub(crate) async fn run(
    config: &Config,
    db: &Writer,
    action: SavedAction,
    session: Option<&SessionId>,
    model: Option<Model>,
    out: &mut impl Write,
) -> Result<Option<SavedRun>> {
    match action {
        SavedAction::List { format } => {
            let questions = db.run(saved::list).await?;
            format.write_rows(out, &questions, "No saved questions yet.", |out, q| {
                writeln!(
                    out,
                    "{}  {:<5}  {:>2} statements  {}  {}",
                    q.name,
                    q.mode,
                    q.statements.len(),
                    q.pinned_at,
                    q.question
                )
            })?;
        }
        SavedAction::Add {
            name,
            from_session,
            message,
        } => {
            let session = match (from_session, session) {
                (Some(prefix), _) => db.run(move |db| crate::find_session(db, &prefix)).await?.id,
                (None, Some(current)) => current.clone(),
                (None, None) => return Err(anyhow!("give --from-session SESSION_ID")),
            };
            let answer = message.map_or(Answer::Last, Answer::Seq);
            let saved = db
                .run(move |db| saved::save(db, &name, &session, answer, None))
                .await?;
            writeln!(
                out,
                "Saved '{}' with {} statement{}: quack saved run {}",
                saved.name,
                saved.statements.len(),
                if saved.statements.len() == 1 { "" } else { "s" },
                saved.name
            )?;
        }
        SavedAction::Run {
            name,
            refresh,
            format,
            ..
        } => {
            let mut question = find(db, &name).await?;
            if refresh {
                let Some(model) = model else {
                    return Err(anyhow!(
                        "refresh from the command line: quack saved run {name} --refresh"
                    ));
                };
                question = refresh_pin(config, &model, question).await?;
            }
            let max_rows = config.analysis.max_query_rows;
            let run = db
                .run(move |db| saved::run(db, &question, max_rows))
                .await?;
            // The terminal runs on a terminal, so its transcript gets a table.
            let format =
                format.unwrap_or_else(|| QueryFormat::default_for(std::io::stdout().is_terminal()));
            write_run(out, &run, format)?;
            return Ok(Some(run));
        }
        SavedAction::Show { name, format } => {
            let question = find(db, &name).await?;
            let id = question.id.clone();
            let last_run = db
                .run(move |db| saved::runs(db, &id, 1))
                .await?
                .into_iter()
                .next();
            write_show(out, &question, last_run.as_ref(), format)?;
        }
        SavedAction::Remove { name } => {
            let question = find(db, &name).await?;
            let id = question.id;
            db.run(move |db| saved::remove(db, &id)).await?;
            writeln!(out, "Removed saved question '{}'.", question.name)?;
        }
    }
    Ok(None)
}

/// The saved question named `name`.
async fn find(db: &Writer, name: &str) -> Result<SavedQuestion> {
    let wanted = name.to_owned();
    Ok(db
        .run(move |db| saved::by_name(db, &wanted))
        .await?
        .ok_or_else(|| ResourceKind::SavedQuestion.missing(name))?)
}

/// Ask `question` again as a print-mode turn in a new session of its
/// mode, writes denied (the steps on stderr, the answer on stdout), and
/// pin the statements the answer ran. A turn that fails leaves no empty
/// session behind.
async fn refresh_pin(
    config: &Config,
    model: &Model,
    question: SavedQuestion,
) -> Result<SavedQuestion> {
    let chat_model = config.chat_model_ref()?.to_string();
    let mode = question.mode;
    let session_id = model
        .db
        .run(move |db| sessions::create_session(db, &chat_model, mode, None))
        .await?
        .id;
    let outcome = PrintTurn {
        config,
        db: Arc::clone(&model.db),
        reader_db: model.reader_db.clone(),
        session_id: &session_id,
        policy: WritePolicy::Deny,
        prompt: &question.question,
        documents: &[],
        format: TextOrJson::Text,
        verbose: model.verbose,
        // The run's result owns stdout; the refreshed answer is commentary.
        answer_to: AnswerTo::Stderr,
    }
    .run()
    .await;
    SessionTitler::finish_pending(TITLE_GRACE).await;
    if outcome.is_err() {
        let id = session_id.clone();
        drop(
            model
                .db
                .run(move |db| sessions::delete_if_empty(db, &id))
                .await,
        );
    }
    outcome?;
    let id = question.id;
    let pinned = model
        .db
        .run(move |db| saved::repin(db, &id, &session_id, Answer::Last))
        .await?;
    writeln!(
        std::io::stderr(),
        "refreshed '{}': {} statement{} pinned from session {}",
        pinned.name,
        pinned.statements.len(),
        if pinned.statements.len() == 1 {
            ""
        } else {
            "s"
        },
        pinned.session_id
    )?;
    Ok(pinned)
}

/// Print a saved question with its last run.
fn write_show(
    out: &mut impl Write,
    question: &SavedQuestion,
    last_run: Option<&SavedRun>,
    format: TextOrJson,
) -> Result<()> {
    match format {
        TextOrJson::Json => writeln!(
            out,
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "question": question,
                "last_run": last_run,
            }))?
        )?,
        TextOrJson::Text => {
            writeln!(out, "{}: {}", question.name, question.question)?;
            writeln!(
                out,
                "mode {}, pinned from session {} ({})",
                question.mode, question.session_id, question.pinned_at
            )?;
            for (n, sql) in question.statements.iter().enumerate() {
                writeln!(
                    out,
                    "\n-- statement {}\n{}",
                    n.saturating_add(1),
                    sql.trim()
                )?;
            }
            writeln!(out)?;
            // A stored run has counts and digests, never rows.
            match last_run {
                Some(run) => {
                    writeln!(out, "last run {}: {}", run.ran_at, run.verdict())?;
                    for (n, statement) in run.statements.iter().enumerate() {
                        writeln!(
                            out,
                            "  statement {}: {}",
                            n.saturating_add(1),
                            statement.outcome()
                        )?;
                    }
                }
                None => writeln!(out, "never run")?,
            }
        }
    }
    Ok(())
}

/// The rows a statement kept, as a result set.
fn kept_rows(statement: &StatementRun) -> Option<QueryResults> {
    statement.result.as_ref().map(|rows| QueryResults {
        columns: statement.columns.clone(),
        rows: rows.clone(),
    })
}

/// Print a run in `format`: `json` is the run as one document (the
/// verdict, each statement's digest and counts, the rows within the cap,
/// and the run id); `table` is each statement with its rows and ends with
/// one word a person reads, `changed`, `unchanged`, or `failed`; the rest
/// are each statement's rows in that format, a blank line between
/// statements, with a failed or uncapped statement noted on stderr.
fn write_run(out: &mut impl Write, run: &SavedRun, format: QueryFormat) -> Result<()> {
    match format {
        QueryFormat::Json => writeln!(out, "{}", serde_json::to_string_pretty(run)?)?,
        QueryFormat::Table => {
            for (n, statement) in run.statements.iter().enumerate() {
                writeln!(out, "-- statement {}", n.saturating_add(1))?;
                writeln!(out, "{}", statement.sql.trim())?;
                if let Some(rows) = kept_rows(statement) {
                    rows.write_table(out)?;
                }
                let kept = if statement.result.is_some() || statement.error.is_some() {
                    ""
                } else {
                    " (past the row cap, so not kept)"
                };
                writeln!(out, "{}{kept}", statement.outcome())?;
                writeln!(out)?;
            }
            writeln!(out, "run {}: {}", run.id, run.verdict())?;
        }
        QueryFormat::Ndjson | QueryFormat::Csv | QueryFormat::Markdown => {
            for (n, statement) in run.statements.iter().enumerate() {
                if n > 0 {
                    writeln!(out)?;
                }
                match (kept_rows(statement), &statement.error) {
                    (Some(rows), _) => format.write(&rows, out)?,
                    (None, Some(error)) => {
                        tracing::error!("statement {}: {error}", n.saturating_add(1));
                    }
                    (None, None) => tracing::warn!(
                        "statement {}: {} rows is past the row cap, so they were not kept",
                        n.saturating_add(1),
                        statement.rows.unwrap_or_default()
                    ),
                }
            }
        }
    }
    Ok(())
}

/// Whether a run should end the command with an error: a statement failed.
pub(crate) fn failure(run: &SavedRun) -> Option<anyhow::Error> {
    if run.status != RunStatus::Failed {
        return None;
    }
    let first = run
        .statements
        .iter()
        .enumerate()
        .find_map(|(n, s)| {
            s.error
                .as_ref()
                .map(|e| format!("statement {}: {e}", n.saturating_add(1)))
        })
        .unwrap_or_else(|| String::from("a statement failed"));
    Some(anyhow!("run {} failed: {first}", run.id))
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::indexing_slicing,
        reason = "serde_json::Value indexing yields Null for a missing key, never a panic; the statements were just asserted to exist"
    )]

    use super::*;
    use crate::scripted_ollama::{Reply, ScriptedOllama};
    use jiff::Timestamp;
    use quack_core::analysis::agent::AgentResponse;
    use quack_core::analysis::events::{ToolName, ToolStep};
    use quack_core::embedding::Dimension;
    use quack_core::llm::egress::Egress;
    use quack_core::storage::control::AllowedProviders;
    use quack_core::storage::sessions::ChatMode;
    use quack_core::storage::workspace::WorkspaceDb;
    use std::sync::Arc;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    const OVERDUE: &str = "SELECT id FROM invoices WHERE NOT paid ORDER BY id";

    /// A workspace with an `invoices` table and one answered session that
    /// ran [`OVERDUE`].
    fn workspace() -> (SharedDb, SessionId) {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        db.execute_statement(
            "CREATE TABLE invoices (id INTEGER, paid BOOLEAN); \
             INSERT INTO invoices VALUES (1, false), (2, true)",
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
        let session = sessions::create_session(&db, "m", ChatMode::Query, None)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let response = AgentResponse {
            content: String::from("Invoice 1 is overdue."),
            steps: vec![ToolStep {
                tool: ToolName::RunSql,
                detail: String::from(OVERDUE),
                summary: String::from("1 rows"),
                rows: Some(1),
                result: None,
                duration_ms: 1,
            }],
            ..AgentResponse::default()
        };
        sessions::record_turn(&db, &session.id, "overdue?", Timestamp::now(), &response)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
        (db, session.id)
    }

    /// `action` with no model and no current session, as the command line
    /// runs it: its output, and the run when it was one.
    async fn quack(db: &Writer, action: SavedAction) -> (String, Result<Option<SavedRun>>) {
        let mut out = Vec::new();
        let result = run(&Config::default(), db, action, None, None, &mut out).await;
        (String::from_utf8_lossy(&out).into_owned(), result)
    }

    fn list(format: TextOrJson) -> SavedAction {
        SavedAction::List { format }
    }

    fn add(name: &str, session: Option<&SessionId>, message: Option<i64>) -> SavedAction {
        SavedAction::Add {
            name: name.to_owned(),
            from_session: session.map(ToString::to_string),
            message,
        }
    }

    fn show(name: &str, format: TextOrJson) -> SavedAction {
        SavedAction::Show {
            name: name.to_owned(),
            format,
        }
    }

    fn run_named(name: &str, format: QueryFormat) -> SavedAction {
        SavedAction::Run {
            name: name.to_owned(),
            refresh: false,
            exit_code: true,
            format: Some(format),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    async fn every_verb_prints_and_a_run_says_changed_or_not() {
        let (db, session) = workspace();
        let (text, _) = quack(&db, list(TextOrJson::Text)).await;
        assert_eq!(text, "No saved questions yet.\n");

        let (text, nowhere) = quack(&db, add("overdue", None, None)).await;
        assert!(text.is_empty());
        assert!(
            nowhere.unwrap_err().to_string().contains("--from-session"),
            "the command line has no current session"
        );
        let prefix = SessionId::from(session.short());
        let (text, added) = quack(&db, add("overdue", Some(&prefix), None)).await;
        added.unwrap();
        assert_eq!(
            text,
            "Saved 'overdue' with 1 statement: quack saved run overdue\n"
        );
        let (_, again) = quack(&db, add("overdue", Some(&session), None)).await;
        assert!(again.unwrap_err().to_string().contains("already exists"));
        let (_, not_answer) = quack(&db, add("q", Some(&session), Some(1))).await;
        assert!(
            not_answer
                .unwrap_err()
                .to_string()
                .contains("not an answer")
        );

        let (text, _) = quack(&db, list(TextOrJson::Text)).await;
        assert!(
            text.starts_with("overdue  query   1 statements  "),
            "{text}"
        );
        assert!(text.ends_with("  overdue?\n"), "{text}");
        let (json, _) = quack(&db, list(TextOrJson::Json)).await;
        let row: serde_json::Value = serde_json::from_str(json.trim()).unwrap();
        assert_eq!(row["name"], "overdue");
        assert_eq!(row["statements"], serde_json::json!([OVERDUE]));

        let (text, first) = quack(&db, run_named("overdue", QueryFormat::Table)).await;
        let first = first.unwrap().unwrap();
        assert!(!first.changed);
        assert!(failure(&first).is_none());
        assert!(text.contains("-- statement 1\n"), "{text}");
        assert!(text.contains("1 rows, unchanged\n"), "{text}");
        assert!(
            text.ends_with(&format!("run {}: unchanged\n", first.id)),
            "{text}"
        );

        db.run(|db| db.execute_statement("UPDATE invoices SET paid = false WHERE id = 2"))
            .await
            .unwrap();
        let (json, second) = quack(&db, run_named("overdue", QueryFormat::Json)).await;
        let second = second.unwrap().unwrap();
        assert!(second.changed);
        let object: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(object["changed"], true);
        assert_eq!(object["id"], second.id.as_str());
        assert_eq!(object["statements"][0]["rows"], 2);
        assert_eq!(
            object["statements"][0]["digest"].as_str().map(str::len),
            Some(64)
        );

        let (csv, _) = quack(&db, run_named("overdue", QueryFormat::Csv)).await;
        assert_eq!(csv, "id\n1\n2\n");
        let (ndjson, _) = quack(&db, run_named("overdue", QueryFormat::Ndjson)).await;
        assert_eq!(ndjson, "{\"id\":1}\n{\"id\":2}\n");
        let (markdown, _) = quack(&db, run_named("overdue", QueryFormat::Markdown)).await;
        assert!(markdown.starts_with("| id |\n"), "{markdown}");

        let (text, _) = quack(&db, show("overdue", TextOrJson::Text)).await;
        assert!(text.starts_with("overdue: overdue?\n"), "{text}");
        assert!(text.contains(OVERDUE), "{text}");
        assert!(text.contains("last run "), "{text}");
        assert!(
            text.ends_with(": unchanged\n  statement 1: 2 rows, unchanged\n"),
            "{text}"
        );
        let (json, _) = quack(&db, show("overdue", TextOrJson::Json)).await;
        let object: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(object["question"]["name"], "overdue");
        assert_eq!(object["last_run"]["changed"], false);

        let (_, missing) = quack(&db, run_named("nope", QueryFormat::Table)).await;
        assert_eq!(
            missing.unwrap_err().to_string(),
            "saved question 'nope' does not exist"
        );
        let (text, removed) = quack(
            &db,
            SavedAction::Remove {
                name: String::from("overdue"),
            },
        )
        .await;
        removed.unwrap();
        assert_eq!(text, "Removed saved question 'overdue'.\n");
        let (text, _) = quack(&db, list(TextOrJson::Text)).await;
        assert_eq!(text, "No saved questions yet.\n");
    }

    /// A run whose table is gone prints the error and is the command's
    /// failure, so cron sees exit 1 rather than an unchanged result.
    #[tokio::test(flavor = "multi_thread")]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    async fn a_failed_run_is_printed_and_is_the_commands_error() {
        let (db, session) = workspace();
        quack(&db, add("overdue", Some(&session), None))
            .await
            .1
            .unwrap();
        db.run(|db| db.execute_statement("DROP TABLE invoices"))
            .await
            .unwrap();
        let (text, failed) = quack(&db, run_named("overdue", QueryFormat::Table)).await;
        let failed = failed.unwrap().unwrap();
        assert_eq!(failed.status, RunStatus::Failed);
        assert!(text.contains("error: "), "{text}");
        assert!(
            text.ends_with(&format!("run {}: failed\n", failed.id)),
            "{text}"
        );
        let error = failure(&failed).unwrap().to_string();
        assert!(
            error.starts_with(&format!("run {} failed: statement 1: ", failed.id)),
            "{error}"
        );
        let (_, refresh_without_model) = quack(
            &db,
            SavedAction::Run {
                name: String::from("overdue"),
                refresh: true,
                exit_code: false,
                format: Some(QueryFormat::Table),
            },
        )
        .await;
        assert!(
            refresh_without_model
                .unwrap_err()
                .to_string()
                .contains("quack saved run overdue --refresh")
        );
    }

    /// `--refresh` asks the model again, writes denied, and pins the SQL
    /// its new answer ran; the run after compares with nothing older.
    #[tokio::test(flavor = "multi_thread")]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    async fn refresh_asks_the_model_again_and_repins_the_sql() {
        let ollama = ScriptedOllama::serve(vec![
            Reply::Call {
                tool: "run_sql",
                args: serde_json::json!({ "query": "UPDATE invoices SET paid = true" }),
            },
            Reply::Call {
                tool: "run_sql",
                args: serde_json::json!({ "query": "SELECT count(*) AS n FROM invoices" }),
            },
            Reply::Text("Two invoices."),
        ])
        .await
        .unwrap();
        let config = ollama.config().unwrap();
        let (db, session) = workspace();
        quack(&db, add("overdue", Some(&session), None))
            .await
            .1
            .unwrap();
        let model = Model {
            db: Arc::clone(&db),
            reader_db: ReaderDb::open(&db, config.analysis.reader_pool_size).await,
            verbose: false,
        };
        let mut out = Vec::new();
        let egress = Egress::Workspace(AllowedProviders::All);
        let refreshed = Egress::scope(
            Some(egress),
            run(
                &config,
                &db,
                SavedAction::Run {
                    name: String::from("overdue"),
                    refresh: true,
                    exit_code: false,
                    format: Some(QueryFormat::Table),
                },
                None,
                Some(model),
                &mut out,
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(!refreshed.changed, "the refresh changed the SQL");
        assert_eq!(
            refreshed.statements.len(),
            1,
            "the refused write was not pinned"
        );
        assert_eq!(
            refreshed.statements[0].sql,
            "SELECT count(*) AS n FROM invoices"
        );
        let question = db
            .run(|db| saved::by_name(db, "overdue"))
            .await
            .unwrap()
            .unwrap();
        assert_ne!(question.session_id, session, "pinned from the new session");
        let paid: i64 = db
            .run(|db| {
                Ok(db.connection().query_row(
                    "SELECT count(*) FROM invoices WHERE paid",
                    [],
                    |r| r.get(0),
                )?)
            })
            .await
            .unwrap();
        assert_eq!(paid, 1, "the refresh turn could not write");
    }
}
