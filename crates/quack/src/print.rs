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
use quack_core::analysis::tools::SharedDb;
use quack_core::config::Config;
use quack_core::llm;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptFormat {
    Text,
    Json,
}

/// Run one turn in `session_id` and print it. Returns whether a write was
/// refused.
pub(crate) async fn run_prompt(
    config: &Config,
    db: SharedDb,
    session_id: &str,
    policy: WritePolicy,
    prompt: &str,
    format: PromptFormat,
    verbose: bool,
) -> Result<bool> {
    let (sink, mut events) = events::channel();

    let turn = tokio::spawn({
        let config = config.clone();
        let prompt = prompt.to_owned();
        let session_id = session_id.to_owned();
        async move { llm::run_turn(&config, db, &session_id, policy, &prompt, sink).await }
    });

    // Never hold the stdout or stderr locks across an await: the tracing
    // subscriber writes to stderr from the agent's threads, and holding the
    // lock here deadlocks the turn the moment a tool logs anything.
    let mut err = std::io::stderr();
    let mut out = std::io::stdout();
    let mut streamed_any = false;
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
            AgentEvent::TextDelta(text) => {
                if format == PromptFormat::Text && !searched {
                    write!(out, "{text}")?;
                    out.flush()?;
                    streamed_any = true;
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

    match format {
        PromptFormat::Text => {
            if !streamed_any {
                write!(out, "{}", response.content)?;
            }
            if !response.content.ends_with('\n') {
                writeln!(out)?;
            }
            if !response.citations.is_empty() {
                writeln!(out)?;
                writeln!(out, "Sources:")?;
                for citation in &response.citations {
                    writeln!(out, "  [{}] {}", citation.n, citation.label())?;
                }
            }
        }
        PromptFormat::Json => {
            let object = serde_json::json!({
                "answer": response.content,
                "steps": response.steps,
                "citations": response.citations,
                "chart": response.chart,
                "graph": response.graph,
                "write_refused": response.write_refused,
                "session_id": session_id,
            });
            serde_json::to_writer_pretty(&mut out, &object)?;
            writeln!(out)?;
        }
    }
    out.flush()?;
    writeln!(err, "session {session_id}")?;

    Ok(response.write_refused)
}

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
    if !detail.is_empty() {
        let shown: String = if verbose {
            detail.to_owned()
        } else {
            detail.chars().take(400).collect()
        };
        for line in shown.lines() {
            writeln!(err, "  {line}")?;
        }
        if shown.chars().count() < detail.chars().count() {
            writeln!(err, "  ...")?;
        }
    }
    Ok(())
}

fn write_finished(err: &mut impl Write, step: &ToolStep) -> Result<()> {
    writeln!(err, "  {}, {} ms", step.summary, step.duration_ms)?;
    Ok(())
}
