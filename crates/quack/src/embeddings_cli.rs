//! `quack embeddings`: the workspace's vectors. `refresh` brings them up
//! to the configured embedding profile, after the embedding model, its
//! width, or its input prefixes changed. Until then the chunks those
//! vectors belong to are found by keyword search only.

use std::io::Write;

use anyhow::Result;
use clap::Subcommand;
use quack_core::config::Config;
use quack_core::embedding::refresh::{self, Plan};
use quack_core::llm::Embeddings;
use quack_core::progress::RunControl;
use quack_core::storage::workspace::WorkspaceDb;
use quack_core::storage::writer::Writer;

use crate::confirm::Confirm;

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum EmbeddingsAction {
    /// Embed again the chunks and graph labels whose vectors were made
    /// with another embedding model, width, or input prefixes, or have
    /// none; until then they are found by keyword search only
    Refresh {
        /// Do not ask before spending the model calls
        #[arg(long, short = 'y')]
        yes: bool,
    },
}

/// Run `action` against the workspace behind `db`.
///
/// # Errors
///
/// Returns an error when no embedding model is configured, or the run
/// fails.
pub(crate) async fn run(
    config: &Config,
    db: &Writer,
    action: EmbeddingsAction,
    confirm: Confirm,
    out: &mut impl Write,
    control: RunControl<'_>,
) -> Result<()> {
    match action {
        EmbeddingsAction::Refresh { yes } => {
            refresh(config, db, confirm.or_yes(yes), out, control).await
        }
    }
}

/// Say what a refresh would do, ask when `confirm_first` says to, then
/// run it.
async fn refresh(
    config: &Config,
    db: &Writer,
    confirm_first: Confirm,
    out: &mut impl Write,
    control: RunControl<'_>,
) -> Result<()> {
    let embedder = Embeddings::require(config).await?;
    let plan = Plan::from_status(&db.run(WorkspaceDb::embedding_status).await?);
    if plan.is_empty() {
        writeln!(
            out,
            "Every vector was made with {}; nothing to refresh.",
            embedder.profile()
        )?;
        return Ok(());
    }
    writeln!(out, "{plan}")?;
    if !confirm_first.ask(out, "Proceed?", Some("--yes"))? {
        writeln!(out, "Nothing changed.")?;
        return Ok(());
    }
    let summary = refresh::run(
        db,
        &embedder,
        config.ingestion.embedding_batch_size,
        control,
    )
    .await?;
    writeln!(out, "{summary}")?;
    Ok(())
}
