//! One-shot print mode: `quack -p PROMPT`.
//!
//! The answer goes to stdout; every step (tool call, result size, timing)
//! goes to stderr so pipelines stay clean. `--format json` collects the
//! whole turn into one object instead of streaming.

use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use quack_core::analysis::events::{self, AgentEvent, ToolStep};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::config::Config;
use quack_core::llm;

/// A token that Ctrl+C cancels, and the task watching for it. The turn
/// is then recorded as cancelled with whatever streamed; a second Ctrl+C
/// is left to the runtime.
fn cancel_on_ctrl_c() -> (llm::CancellationToken, tokio::task::JoinHandle<()>) {
    let cancel = llm::CancellationToken::new();
    let interrupt = tokio::spawn({
        let cancel = cancel.clone();
        async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                cancel.cancel();
            }
        }
    });
    (cancel, interrupt)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptFormat {
    Text,
    Json,
}

/// Run one turn in `session_id` and print it. Returns whether a write was
/// refused.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors llm::run_turn plus print mode's own formatting options"
)]
pub(crate) async fn run_prompt(
    config: &Config,
    db: SharedDb,
    reader_db: ReaderDb,
    session_id: &str,
    policy: WritePolicy,
    prompt: &str,
    format: PromptFormat,
    verbose: bool,
) -> Result<bool> {
    let (sink, mut events) = events::channel();

    let (cancel, interrupt) = cancel_on_ctrl_c();
    let turn = tokio::spawn({
        let config = config.clone();
        let prompt = prompt.to_owned();
        let session_id = session_id.to_owned();
        async move {
            llm::run_turn(
                &config,
                db,
                reader_db,
                &session_id,
                policy,
                &prompt,
                sink,
                cancel,
            )
            .await
        }
    });

    // Never hold the stdout or stderr locks across an await: the tracing
    // subscriber writes to stderr from the agent's threads, and holding the
    // lock here deadlocks the turn the moment a tool logs anything.
    let mut err = std::io::stderr();
    let mut out = std::io::stdout();
    // What went to stdout as it streamed, to compare with the validated
    // answer at the end. Text streams only on a terminal: a pipeline gets
    // the validated answer alone (issue #64).
    let mut streamed = String::new();
    let stream_live = format == PromptFormat::Text && out.is_terminal();
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
                if tool == "search_documents" {
                    searched = true;
                }
                write_started(&mut err, &tool, &detail, verbose)?;
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

    let response = turn
        .await
        .context("agent task panicked")?
        .context("agent turn failed")?;
    interrupt.abort();

    match format {
        PromptFormat::Text => write_text_answer(&mut out, &streamed, &response)?,
        PromptFormat::Json => {
            let object = response.to_json(session_id);
            serde_json::to_writer_pretty(&mut out, &object)?;
            writeln!(out)?;
        }
    }
    out.flush()?;
    writeln!(err, "session {session_id}")?;

    Ok(response.write_refused)
}

/// The text-mode answer after the turn: the validated content when
/// nothing streamed, what the turn appended when the stream stands, the
/// validated content again when validation changed what streamed, then
/// the chart note and the sources.
fn write_text_answer(
    out: &mut impl Write,
    streamed: &str,
    response: &quack_core::analysis::agent::AgentResponse,
) -> Result<()> {
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
            "Chart: {} ({} chart, {} points; --format json carries the spec)",
            chart.title,
            chart.kind.as_str(),
            chart.points()
        )?;
    }
    if !response.citations.is_empty() {
        writeln!(out)?;
        writeln!(out, "Sources:")?;
        for citation in &response.citations {
            writeln!(out, "  [{}] {}", citation.n, citation.label())?;
        }
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

fn write_started(err: &mut impl Write, tool: &str, detail: &str, verbose: bool) -> Result<()> {
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
    let (shown, more) = events::preview_detail(detail);
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

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
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
        write_started(&mut folded, "run_sql", &detail, false)
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
        write_started(&mut whole, "run_sql", &detail, true)
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(String::from_utf8_lossy(&whole).contains("  SELECT 5\n"));

        let mut empty = Vec::new();
        write_started(&mut empty, "list_tables", "", false)
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(String::from_utf8_lossy(&empty), "> list_tables\n");

        let mut finished = Vec::new();
        write_finished(
            &mut finished,
            &ToolStep {
                tool: String::from("run_sql"),
                detail: String::new(),
                summary: String::from("3 rows"),
                duration_ms: 12,
            },
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(String::from_utf8_lossy(&finished), "  3 rows, 12 ms\n");
    }
}
