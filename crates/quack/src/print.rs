//! One-shot print mode: `quack -p PROMPT`.
//!
//! The answer goes to stdout; every step (tool call, result size, timing)
//! goes to stderr so pipelines stay clean. `--format json` collects the
//! whole turn into one object instead of streaming.

use std::io::Write;

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

    let stderr = std::io::stderr();
    let stdout = std::io::stdout();
    let mut err = stderr.lock();
    let mut out = stdout.lock();
    let mut streamed_any = false;

    while let Some(event) = events.recv().await {
        match event {
            AgentEvent::TextDelta(text) => {
                if format == PromptFormat::Text {
                    write!(out, "{text}")?;
                    out.flush()?;
                    streamed_any = true;
                }
            }
            AgentEvent::ToolStarted { tool, detail } => {
                write_started(&mut err, &tool, &detail, verbose)?;
            }
            AgentEvent::ToolFinished(step) => {
                write_finished(&mut err, &step)?;
            }
            AgentEvent::PermissionRequired(request) => {
                // Print mode never prompts; the policy is Allow or Deny.
                request.deny();
            }
            AgentEvent::TurnComplete(_) | AgentEvent::Failed(_) => {}
        }
    }

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
        }
        PromptFormat::Json => {
            let object = serde_json::json!({
                "answer": response.content,
                "steps": response.steps,
                "chart": response.chart_spec,
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
