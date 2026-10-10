//! One-shot print mode: `quack -p PROMPT`.
//!
//! The answer goes to stdout; every step (tool call, result size, timing)
//! goes to stderr so pipelines stay clean. `--format json` collects the
//! whole turn into one object instead of streaming.

use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use quack_core::analysis::agent::AgentResponse;
use quack_core::analysis::citations::Sources;
use quack_core::analysis::events::{self, AgentEvent, DetailPreview, ToolName, ToolStep};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::classify::LabelJobs;
use quack_core::config::Config;
use quack_core::ids::SessionId;
use quack_core::jobs::JobQueue;
use quack_core::llm;
use quack_core::llm::egress::Egress;

use crate::text_or_json::TextOrJson;

/// How long a command that answers one question waits, after printing, for
/// the session title and history summary its turn started.
pub const FOLLOW_UP_GRACE: Duration = Duration::from_secs(30);

/// A cancellation token that Ctrl+C trips, for as long as the guard
/// lives: the turn is then recorded as cancelled with whatever streamed,
/// and a second Ctrl+C is left to the runtime. Dropping the guard stops
/// the watcher, whichever way the turn ended.
struct CtrlCGuard {
    cancel: llm::CancellationToken,
    watcher: tokio::task::JoinHandle<()>,
}

impl CtrlCGuard {
    fn new() -> Self {
        let cancel = llm::CancellationToken::new();
        let watcher = tokio::spawn({
            let cancel = cancel.clone();
            async move {
                if tokio::signal::ctrl_c().await.is_ok() {
                    cancel.cancel();
                }
            }
        });
        Self { cancel, watcher }
    }
}

impl Drop for CtrlCGuard {
    fn drop(&mut self) {
        self.watcher.abort();
    }
}

/// How a printed turn ended, for the exit status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnOutcome {
    Answered,
    /// The agent needed a write the policy did not permit.
    WriteRefused,
}

/// One print-mode turn: the workspace, the session, the write policy, the
/// prompt, and how to print it.
pub struct PrintTurn<'a> {
    pub config: &'a Config,
    pub db: SharedDb,
    pub reader_db: ReaderDb,
    pub session_id: &'a SessionId,
    pub policy: WritePolicy,
    pub prompt: &'a str,
    /// The documents the question is limited to; empty for all.
    pub documents: &'a [String],
    pub format: TextOrJson,
    /// Full tool inputs and outputs on stderr.
    pub verbose: bool,
    /// The stream the answer goes to.
    pub answer_to: AnswerTo,
}

/// Where a turn's answer goes: stdout for `quack -p`, whose answer is its
/// output; stderr when the command's own result owns stdout, as `saved run
/// --refresh` prints the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerTo {
    Stdout,
    Stderr,
}

impl AnswerTo {
    fn stream(self) -> AnswerStream {
        match self {
            Self::Stdout => AnswerStream::Stdout(std::io::stdout()),
            Self::Stderr => AnswerStream::Stderr(std::io::stderr()),
        }
    }
}

/// The unlocked stream an answer is written to.
enum AnswerStream {
    Stdout(std::io::Stdout),
    Stderr(std::io::Stderr),
}

impl AnswerStream {
    fn is_terminal(&self) -> bool {
        match self {
            Self::Stdout(out) => out.is_terminal(),
            Self::Stderr(err) => err.is_terminal(),
        }
    }
}

impl Write for AnswerStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Stdout(out) => out.write(buf),
            Self::Stderr(err) => err.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Stdout(out) => out.flush(),
            Self::Stderr(err) => err.flush(),
        }
    }
}

impl PrintTurn<'_> {
    /// Run the turn and print it.
    ///
    /// # Errors
    ///
    /// Returns the error from the turn, the database, or writing the answer.
    pub async fn run(self) -> Result<TurnOutcome> {
        let (config, db, reader_db, session_id) =
            (self.config, self.db, self.reader_db, self.session_id);
        let (policy, prompt, format, verbose) =
            (self.policy, self.prompt, self.format, self.verbose);
        let documents = self.documents.to_vec();
        let (sink, mut events) = events::channel();

        let interrupt = CtrlCGuard::new();
        let cancel = interrupt.cancel.clone();
        // A table the agent labels is a job of this queue, which the command
        // waits for before it exits, so an interrupted run records its end.
        let jobs = JobQueue::from_config(&config.jobs);
        let labelling = LabelJobs::new(jobs.clone());
        let turn = tokio::spawn({
            let config = config.clone();
            let prompt = prompt.to_owned();
            let session_id = session_id.to_owned();
            // The turn's own task sends where the command may.
            Egress::scope(Egress::current(), async move {
                llm::TurnRequest {
                    db,
                    reader_db,
                    session_id: &session_id,
                    policy,
                    message: &prompt,
                    documents: &documents,
                    user: None,
                    labelling,
                    sink,
                    cancel,
                }
                .run(&config)
                .await
            })
        });

        // Never hold the stdout or stderr locks across an await: the tracing
        // subscriber writes to stderr from the agent's threads, and holding the
        // lock here deadlocks the turn the moment a tool logs anything.
        let mut err = std::io::stderr();
        let mut out = self.answer_to.stream();
        // What went to stdout as it streamed, to compare with the validated
        // answer at the end. Text streams only on a terminal: a pipeline gets
        // the validated answer alone (issue #64).
        let mut streamed = String::new();
        let stream_live = format == TextOrJson::Text && out.is_terminal();
        // Once a search has run, the answer may carry [n] markers that citation
        // validation renumbers after the stream ends, so buffer instead of
        // printing text that would then need to be reprinted.
        let mut searched = false;

        // On a terminal, show what the turn is waiting on (the model, or a
        // tool) with the elapsed time, and erase it before any real output.
        let mut spinner = Spinner::new(std::io::stderr().is_terminal());
        let mut ticker = tokio::time::interval(Duration::from_millis(250));

        loop {
            let event = tokio::select! {
                event = events.recv() => match event {
                    Some(event) => event,
                    None => break,
                },
                _ = ticker.tick(), if spinner.active() => {
                    spinner.draw(&mut err)?;
                    continue;
                }
            };
            spinner.clear(&mut err)?;
            match event {
                AgentEvent::Status(status) => spinner.set(&status),
                AgentEvent::Reasoning => spinner.set("thinking"),
                AgentEvent::TextDelta(text) => {
                    if stream_live && !searched {
                        write!(out, "{text}")?;
                        out.flush()?;
                        streamed.push_str(&text);
                        spinner.stop();
                    } else {
                        spinner.set("answering");
                    }
                }
                AgentEvent::ToolStarted { tool, detail } => {
                    if tool == ToolName::SearchDocuments {
                        searched = true;
                    }
                    write_started(&mut err, tool, &detail, verbose)?;
                    spinner.set(&format!("running {tool}"));
                }
                AgentEvent::ToolFinished(step) => {
                    write_finished(&mut err, &step)?;
                    spinner.set("thinking");
                }
                AgentEvent::PermissionRequired(request) => {
                    // Print mode never prompts; the policy is Allow or Deny.
                    request.deny();
                }
                AgentEvent::TurnComplete(_) | AgentEvent::Failed(_) => spinner.stop(),
            }
        }
        spinner.clear(&mut err)?;

        let turn = turn.await;
        jobs.shutdown(FOLLOW_UP_GRACE).await;
        let response = turn
            .context("agent task panicked")?
            .context("agent turn failed")?;
        drop(interrupt);

        write_answer(&mut out, format, &streamed, &response, session_id)?;
        out.flush()?;
        writeln!(err, "session {session_id}")?;

        Ok(if response.write_refused {
            TurnOutcome::WriteRefused
        } else {
            TurnOutcome::Answered
        })
    }
}

/// The answer after the turn, as text or as the response object.
fn write_answer(
    out: &mut impl Write,
    format: TextOrJson,
    streamed: &str,
    response: &AgentResponse,
    session_id: &SessionId,
) -> Result<()> {
    match format {
        TextOrJson::Text => write_text_answer(out, streamed, response)?,
        TextOrJson::Json => {
            serde_json::to_writer_pretty(&mut *out, &response.body(session_id))?;
            writeln!(out)?;
        }
    }
    Ok(())
}

/// The text-mode answer after the turn: the validated content when
/// nothing streamed, what the turn appended when the stream stands, the
/// validated content again when validation changed what streamed, then
/// the chart note and the sources.
fn write_text_answer(out: &mut impl Write, streamed: &str, response: &AgentResponse) -> Result<()> {
    if streamed.is_empty() {
        write!(out, "{}", response.content)?;
    } else if let Some(rest) = response.content.strip_prefix(streamed) {
        // What streamed stands; a stopped or cancelled turn appends a note.
        write!(out, "{rest}")?;
    } else {
        // Validation changed the text after it streamed (the model
        // invented citation markers nothing was retrieved for):
        // what stands is the validated answer, so show it.
        if !streamed.ends_with('\n') {
            writeln!(out)?;
        }
        writeln!(out)?;
        writeln!(out, "{VALIDATED_NOTE}")?;
        write!(out, "{}", response.content)?;
    }
    if !response.content.ends_with('\n') {
        writeln!(out)?;
    }
    if let Some(chart) = &response.chart {
        // The spec itself is in `--format json`; text mode says a
        // chart exists rather than dropping it.
        writeln!(out)?;
        writeln!(
            out,
            "Chart: {} ({}{} chart of {}, {} points; --format json carries the spec)",
            chart.title,
            if chart.stacked { "stacked " } else { "" },
            chart.kind.as_str(),
            chart.series_names().join(", "),
            chart.points()
        )?;
    }
    if !response.citations.is_empty() {
        writeln!(out)?;
        writeln!(out, "{}", Sources(&response.citations))?;
    }
    Ok(())
}

/// The line printed between streamed text and the validated answer that
/// replaced it.
const VALIDATED_NOTE: &str = "(The text above streamed before validation, which removed \
                              citation markers nothing was retrieved for. The validated \
                              answer follows.)";

/// A one-line progress indicator on stderr: `⠋ thinking 4s`. Inactive when
/// stderr is not a terminal, so pipelines see only the step lines.
struct Spinner {
    enabled: bool,
    running: bool,
    stage: String,
    started: Instant,
    frame: usize,
    drawn: bool,
}

impl Spinner {
    const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            running: true,
            stage: String::from("thinking"),
            started: Instant::now(),
            frame: 0,
            drawn: false,
        }
    }

    fn active(&self) -> bool {
        self.enabled && self.running
    }

    fn set(&mut self, stage: &str) {
        stage.clone_into(&mut self.stage);
        self.running = true;
    }

    fn stop(&mut self) {
        self.running = false;
    }

    fn draw(&mut self, err: &mut impl Write) -> Result<()> {
        let glyph = Self::FRAMES.get(self.frame).copied().unwrap_or('.');
        self.frame = self.frame.wrapping_add(1);
        if self.frame >= Self::FRAMES.len() {
            self.frame = 0;
        }
        write!(
            err,
            "\r\x1b[2K{glyph} {} {}s",
            self.stage,
            self.started.elapsed().as_secs()
        )?;
        err.flush()?;
        self.drawn = true;
        Ok(())
    }

    /// Erase the indicator line so the next write starts clean.
    fn clear(&mut self, err: &mut impl Write) -> Result<()> {
        if self.drawn {
            write!(err, "\r\x1b[2K")?;
            err.flush()?;
            self.drawn = false;
        }
        Ok(())
    }
}

fn write_started(err: &mut impl Write, tool: ToolName, detail: &str, verbose: bool) -> Result<()> {
    writeln!(err, "> {tool}")?;
    if detail.is_empty() {
        return Ok(());
    }
    if verbose {
        for line in detail.lines() {
            writeln!(err, "  {line}")?;
        }
        return Ok(());
    }
    // The same preview the terminal shows collapsed.
    let DetailPreview {
        lines: shown,
        hidden: more,
    } = DetailPreview::of(detail);
    for line in shown {
        writeln!(err, "  {line}")?;
    }
    if more > 0 {
        writeln!(err, "  ({more} more lines; --verbose shows them)")?;
    }
    Ok(())
}

fn write_finished(err: &mut impl Write, step: &ToolStep) -> Result<()> {
    writeln!(err, "  {}, {} ms", step.summary, step.duration_ms)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use quack_core::analysis::agent::AgentResponse;
    use quack_core::analysis::policy::Approver;
    use quack_core::storage::control::AllowedProviders;
    use quack_core::storage::sessions::{self, ChatMode};
    use quack_core::storage::workspace::WorkspaceDb;
    use quack_core::storage::writer::Writer;
    use quack_testkit::{self, ScriptedOllama};
    use std::sync::Arc;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    /// `--allow-write` lets a turn write until it has read document text.
    /// Print mode cannot ask, so the write after is refused, which is exit
    /// status 3, and the statement did not run.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_turn_that_read_a_document_is_refused_its_write_under_allow_write() {
        let ollama = ScriptedOllama::serve(ScriptedOllama::following_the_note())
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = ollama.config().unwrap_or_else(|e| fail(&e.to_string()));
        config.general.data_dir = dir.path().to_path_buf();
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        quack_testkit::seed_dictating_note(&db).unwrap_or_else(|e| fail(&e.to_string()));
        let session = sessions::create_session(&db, "scripted/model", ChatMode::Chat, None)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
        let turn = PrintTurn {
            config: &config,
            db: Arc::clone(&db),
            reader_db: ReaderDb::open(&db, config.analysis.reader_pool_size).await,
            session_id: &session.id,
            policy: WritePolicy::Allow(Approver::Nobody),
            prompt: "follow the maintenance note",
            documents: &[],
            format: TextOrJson::Json,
            verbose: false,
            answer_to: AnswerTo::Stdout,
        };
        // The command's scope: the workspace allows every provider.
        let egress = Egress::Workspace(AllowedProviders::All);
        let outcome = Egress::scope(Some(egress), turn.run())
            .await
            .unwrap_or_else(|e| fail(&format!("{e:#}")));
        assert_eq!(outcome, TurnOutcome::WriteRefused);
        let tables = db
            .run(WorkspaceDb::list_tables)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(tables, ["customers"], "the dictated drop did not run");
    }

    /// With `[analysis].title_sessions`, the chat model titles a session
    /// after its first turn, in the background; later turns do not ask again.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_model_titles_a_session_after_its_first_turn() {
        let ollama = ScriptedOllama::serve(vec![
            quack_testkit::Reply::Text("Two vendors were late in March."),
            quack_testkit::Reply::Text(r#"{"title": "Late vendors in March"}"#),
            quack_testkit::Reply::Text("Cipla was one of them."),
        ])
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = ollama.config().unwrap_or_else(|e| fail(&e.to_string()));
        config.general.data_dir = dir.path().to_path_buf();
        config.analysis.title_sessions = true;
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        let session = sessions::create_session(&db, "scripted/model", ChatMode::Chat, None)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
        let reader_db = ReaderDb::open(&db, config.analysis.reader_pool_size).await;
        let ask = |prompt: &'static str| PrintTurn {
            config: &config,
            db: Arc::clone(&db),
            reader_db: reader_db.clone(),
            session_id: &session.id,
            policy: WritePolicy::Deny,
            prompt,
            documents: &[],
            format: TextOrJson::Json,
            verbose: false,
            answer_to: AnswerTo::Stdout,
        };
        let egress = || Some(Egress::Workspace(AllowedProviders::All));
        Egress::scope(egress(), ask("which vendors were late in March?").run())
            .await
            .unwrap_or_else(|e| fail(&format!("{e:#}")));
        let title = || async {
            let id = session.id.clone();
            db.run(move |db| sessions::get_session(db, &id))
                .await
                .ok()
                .flatten()
                .and_then(|s| s.title)
        };
        let mut seen = None;
        for _ in 0..200 {
            seen = title().await;
            if seen.as_deref() == Some("Late vendors in March") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(seen.as_deref(), Some("Late vendors in March"));
        Egress::scope(egress(), ask("which ones?").run())
            .await
            .unwrap_or_else(|e| fail(&format!("{e:#}")));
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            title().await.as_deref(),
            Some("Late vendors in March"),
            "a later turn does not title again"
        );
    }

    /// What stands on stdout is the validated answer (issue #64): printed
    /// whole when nothing streamed, not repeated when the stream matched
    /// it, and printed again after a note when validation changed it.
    #[test]
    fn text_answer_is_the_validated_content() {
        let response = AgentResponse {
            content: String::from("Top 5: Texas"),
            ..AgentResponse::default()
        };
        let mut nothing = Vec::new();
        write_text_answer(&mut nothing, "", &response).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(String::from_utf8_lossy(&nothing), "Top 5: Texas\n");

        let mut same = Vec::new();
        write_text_answer(&mut same, "Top 5: Texas", &response)
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(String::from_utf8_lossy(&same), "\n");

        let stopped = AgentResponse {
            content: String::from("Top 5: Texas\n\n(cancelled)"),
            ..AgentResponse::default()
        };
        let mut appended = Vec::new();
        write_text_answer(&mut appended, "Top 5: Texas", &stopped)
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(String::from_utf8_lossy(&appended), "\n\n(cancelled)\n");

        let mut changed = Vec::new();
        write_text_answer(&mut changed, "Top 5: Texas [1]\n\nSources: [1]", &response)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let changed = String::from_utf8_lossy(&changed);
        assert_eq!(
            changed,
            format!("\n\n{VALIDATED_NOTE}\nTop 5: Texas\n"),
            "{changed}"
        );
    }

    /// Steps on stderr fold to the shared preview unless `--verbose`, and
    /// a finished step is one line (issue #63: print mode had no tests).
    #[test]
    fn steps_fold_to_the_shared_preview_unless_verbose() {
        let detail = (1..=5)
            .map(|i| format!("SELECT {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut folded = Vec::new();
        write_started(&mut folded, ToolName::RunSql, &detail, false)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let folded = String::from_utf8_lossy(&folded);
        assert!(folded.starts_with("> run_sql\n  SELECT 1\n"), "{folded}");
        assert!(
            folded.contains("  SELECT 3\n") && !folded.contains("SELECT 4"),
            "{folded}"
        );
        assert!(
            folded.contains("(2 more lines; --verbose shows them)"),
            "{folded}"
        );

        let mut whole = Vec::new();
        write_started(&mut whole, ToolName::RunSql, &detail, true)
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(String::from_utf8_lossy(&whole).contains("  SELECT 5\n"));

        let mut empty = Vec::new();
        write_started(&mut empty, ToolName::ListTables, "", false)
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(String::from_utf8_lossy(&empty), "> list_tables\n");

        let mut finished = Vec::new();
        write_finished(
            &mut finished,
            &ToolStep {
                tool: ToolName::RunSql,
                detail: String::new(),
                summary: String::from("3 rows"),
                rows: Some(3),
                result: None,
                duration_ms: 12,
                run: None,
            },
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(String::from_utf8_lossy(&finished), "  3 rows, 12 ms\n");
    }
}
