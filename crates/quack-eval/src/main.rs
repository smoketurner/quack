//! The evaluation harness (`make eval`, `make eval-offline`, `make
//! eval-scale`): it drives the built `quack` binary as a person would, one
//! process per command over a temporary data directory, and scores what
//! the binary's own JSON output says. Nothing here fails the run on a
//! wrong answer: every case is scored and printed as its section ends, so a
//! change to a prompt or a model shows up as a number to compare between
//! runs, and `QUACK_EVAL_OUT=path.json` keeps the numbers.
//!
//! Offline (no `--ollama`): ingestion by format, retrieval by keyword and
//! hybrid search, and ontology induction, with no model anywhere. With
//! `--ollama`: the same with real embeddings, then the agent's answers, the
//! vision model, and with `--scale FILE` the agent over one large real table.

mod agent;
mod config;
mod formats;
mod induction;
mod quack;
mod retrieval;
mod vision;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;

use crate::quack::{Quack, progress};

#[derive(Debug, Parser)]
#[command(
    name = "quack-eval",
    about = "Run the evaluation suite against the built quack binary"
)]
struct Cli {
    /// The quack binary to drive.
    #[arg(long, default_value_os_t = default_binary())]
    quack: PathBuf,
    /// The fixture directory.
    #[arg(long, default_value_os_t = default_fixture())]
    fixture: PathBuf,
    /// Use the Ollama models named by `QUACK_EVAL_*` for embeddings, the
    /// agent, and the vision model.
    #[arg(long)]
    ollama: bool,
    /// A large CSV to load as the table `tornadoes` and ask the scale
    /// questions over (needs --ollama).
    #[arg(long, value_name = "FILE")]
    scale: Option<PathBuf>,
    /// Write the report as JSON here too (or set `QUACK_EVAL_OUT`).
    #[arg(long, value_name = "FILE", env = "QUACK_EVAL_OUT")]
    out: Option<PathBuf>,
}

/// `part / whole` over counts of fixture-sized things, where the precision
/// of a `u32` is more than enough.
#[must_use]
pub(crate) fn share(part: usize, whole: usize) -> f64 {
    let f = |n: usize| f64::from(u32::try_from(n).unwrap_or(u32::MAX));
    f(part) / f(whole)
}

fn default_binary() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/quack")
}

fn default_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixture")
}

/// The whole report, as `QUACK_EVAL_OUT` writes it.
#[derive(Debug, Serialize)]
struct Report {
    formats: formats::FormatsReport,
    retrieval: retrieval::RetrievalReport,
    induction: induction::InductionReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    models: Option<config::Models>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent: Option<agent::AgentReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vision: Option<vision::VisionReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scale: Option<ScaleReport>,
}

/// The agent over one large real table.
#[derive(Debug, Serialize)]
struct ScaleReport {
    file: String,
    rows: u64,
    load_seconds: f64,
    agent: agent::AgentReport,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            progress(&format!("eval failed: {e:#}"));
            ExitCode::FAILURE
        }
    }
}

fn run(cli: &Cli) -> Result<()> {
    let mut out = std::io::stdout().lock();
    let data_dir = tempfile::tempdir().context("creating the data directory")?;
    let models = cli.ollama.then(config::Models::from_env);
    config::write(data_dir.path(), models.as_ref())?;
    let turn = models.as_ref().map_or(60, |m| m.turn_seconds);
    let quack = Quack::new(
        cli.quack.clone(),
        data_dir.path(),
        Duration::from_secs(turn),
    )?;
    if cli.scale.is_some() && models.is_none() {
        bail!("--scale needs --ollama: the scale questions go to the chat model");
    }
    let fixture = &cli.fixture;

    let formats = formats::evaluate(&quack, fixture)?;
    formats.print(&mut out)?;

    quack.create_workspace(retrieval::WORKSPACE)?;
    ingest_dir(&quack, retrieval::WORKSPACE, &fixture.join("documents"))?;
    ingest_dir(&quack, retrieval::WORKSPACE, &fixture.join("tables"))?;
    let embedding = models
        .as_ref()
        .map_or("none (keyword only)", |m| m.embedding.as_str());
    let retrieval = retrieval::evaluate(&quack, fixture, embedding, models.is_some())?;
    retrieval.print(&mut out)?;
    let induction = induction::evaluate(&quack, fixture)?;
    induction.print(&mut out)?;

    let mut report = Report {
        formats,
        retrieval,
        induction,
        models: models.clone(),
        agent: None,
        vision: None,
        scale: None,
    };
    if let Some(models) = &models {
        const WORKSPACE: &str = "agent";
        progress("\nIngesting the fixture again for the agent...");
        quack.create_workspace(WORKSPACE)?;
        ingest_dir(&quack, WORKSPACE, &fixture.join("documents"))?;
        ingest_dir(&quack, WORKSPACE, &fixture.join("tables"))?;
        ingest_dir(&quack, WORKSPACE, &fixture.join("agent_tables"))?;
        let agent = agent::evaluate(
            &quack,
            WORKSPACE,
            &fixture.join("agent_questions.json"),
            models.repeat,
        )?;
        agent.print(
            &mut out,
            &format!(
                "Agent answers ({} questions, chat model {})",
                agent.n, models.chat
            ),
        )?;
        report.agent = Some(agent);
        if let Some(model) = &models.vision {
            let vision = vision::evaluate(&quack, WORKSPACE, fixture, model)?;
            vision.print(&mut out)?;
            report.vision = Some(vision);
        }
        if let Some(csv) = &cli.scale {
            let scale = scale(&quack, fixture, csv, models)?;
            scale.agent.print(
                &mut out,
                &format!(
                    "Scale ({}: {} rows loaded in {:.1}s, chat model {})",
                    scale.file, scale.rows, scale.load_seconds, models.chat
                ),
            )?;
            report.scale = Some(scale);
        }
    }
    if let Some(path) = &cli.out {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

/// `quack ingest` every file in `dir`, in name order, failing on the first
/// the binary refuses: the fixture is meant to load.
fn ingest_dir(quack: &Quack, workspace: &str, dir: &Path) -> Result<()> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    for file in files {
        let run = quack.ingest(&file, workspace)?;
        if !run.ok() {
            bail!("ingesting {}: {}", file.display(), run.failure());
        }
    }
    Ok(())
}

/// Load `csv` as the table `tornadoes` in a workspace of its own and ask
/// the scale questions there: a table of tens of thousands of rows and a
/// text column with more values than the prompt lists, which the fixture's
/// tables never show.
fn scale(
    quack: &Quack,
    fixture: &Path,
    csv: &Path,
    models: &config::Models,
) -> Result<ScaleReport> {
    const WORKSPACE: &str = "scale";
    let file = csv
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("tornadoes.csv")
        .to_owned();
    progress(&format!("\nLoading {file} as the table tornadoes..."));
    quack.create_workspace(WORKSPACE)?;
    let run = quack.ingest(csv, WORKSPACE)?;
    if !run.ok() {
        bail!("loading {}: {}", csv.display(), run.failure());
    }
    let rows = quack
        .query("SELECT count(*) AS n FROM tornadoes", WORKSPACE)?
        .first()
        .and_then(|r| r.first())
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    progress(&format!("  {rows} rows in {:.1}s", run.seconds));
    let agent = agent::evaluate(
        quack,
        WORKSPACE,
        &fixture.join("scale_questions.json"),
        models.repeat,
    )?;
    Ok(ScaleReport {
        file,
        rows,
        load_seconds: run.seconds,
        agent,
    })
}
