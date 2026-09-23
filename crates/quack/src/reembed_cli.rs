//! `quack reembed`: bring a workspace's vectors up to the configured
//! embedding profile, after the embedding model, its width, or its input
//! prefixes changed. Until then the chunks those vectors belong to are
//! found by keyword search only.

use std::io::{self, Write};

use anyhow::Result;
use quack_core::config::Config;
use quack_core::embedding::reembed::{self, Plan};
use quack_core::llm;
use quack_core::progress::{ChunkDone, RunControl};
use quack_core::storage::workspace::WorkspaceDb;
use quack_core::storage::writer::Writer;

use crate::graph_cli::confirm;

/// Whether to ask before spending the model calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Confirm {
    /// Show the plan and read the answer from stdin.
    Ask,
    /// Go ahead: `-y`, or a terminal job, which may not read stdin.
    Assume,
}

/// Say what a run would do, ask when `confirm_first` says to, then run it.
///
/// # Errors
///
/// Returns an error when no embedding model is configured, or the run
/// fails.
pub(crate) async fn run(
    config: &Config,
    db: &Writer,
    confirm_first: Confirm,
    out: &mut impl Write,
    control: RunControl<'_>,
) -> Result<()> {
    let embedder = llm::required_embedding_model(config).await?;
    let plan = Plan::from_status(&db.run(WorkspaceDb::embedding_status).await?);
    if plan.is_empty() {
        writeln!(
            out,
            "Every vector was made with {}; nothing to re-embed.",
            embedder.profile().describe()
        )?;
        return Ok(());
    }
    writeln!(out, "{plan}")?;
    if confirm_first == Confirm::Ask && !confirm(out)? {
        writeln!(out, "Nothing changed.")?;
        return Ok(());
    }
    let summary = reembed::run(
        db,
        &embedder,
        config.ingestion.embedding_batch_size,
        control,
    )
    .await?;
    writeln!(out, "{summary}")?;
    Ok(())
}

/// Progress on stderr, for the shell: chunks and node labels together.
pub(crate) fn print_progress(done: ChunkDone) {
    drop(writeln!(
        io::stderr(),
        "{}/{} re-embedded; {} s elapsed",
        done.done,
        done.total,
        done.elapsed.as_secs()
    ));
}
