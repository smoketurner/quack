//! `quack reembed`: bring a workspace's vectors up to the configured
//! embedding profile, after the embedding model, its width, or its input
//! prefixes changed. Until then the chunks those vectors belong to are
//! found by keyword search only.

use std::io::Write;

use anyhow::Result;
use quack_core::config::Config;
use quack_core::embedding::Profile;
use quack_core::embedding::reembed::{self, Plan, Progress, Stage, Summary};
use quack_core::llm;
use quack_core::storage::workspace::{EmbeddingStatus, WorkspaceDb};
use quack_core::storage::writer::Writer;
use tokio_util::sync::CancellationToken;

use crate::graph_cli::confirm;

/// Say what a run would do, ask unless `yes`, then run it. `progress`
/// hears after every batch; `cancel` stops between batches.
///
/// # Errors
///
/// Returns an error when no embedding model is configured, or the run
/// fails.
pub(crate) async fn run(
    config: &Config,
    db: &Writer,
    yes: bool,
    out: &mut impl Write,
    progress: &(dyn Fn(Progress) + Sync),
    cancel: Option<&CancellationToken>,
) -> Result<()> {
    let embedder = llm::required_embedding_model(config).await?;
    let status = db.run(WorkspaceDb::embedding_status).await?;
    let plan = Plan::from_status(&status);
    let profile = embedder.profile().describe();
    if plan.is_empty() {
        writeln!(
            out,
            "Every vector was made with {profile}; nothing to re-embed."
        )?;
        return Ok(());
    }
    write_plan(out, &plan, &status, &profile)?;
    if !yes && !confirm(out)? {
        writeln!(out, "Nothing changed.")?;
        return Ok(());
    }
    let summary = reembed::run(
        db,
        &embedder,
        config.ingestion.embedding_batch_size,
        progress,
        cancel,
    )
    .await?;
    write_summary(out, &summary, &profile)?;
    Ok(())
}

/// What a run will do, before it starts.
pub(crate) fn write_plan(
    out: &mut impl Write,
    plan: &Plan,
    status: &EmbeddingStatus,
    profile: &str,
) -> Result<()> {
    if let Some((stored, configured)) = plan.retype {
        writeln!(
            out,
            "The workspace stores {stored}-dimensional vectors and the configured model makes \
             {configured}-dimensional ones: every stored vector is dropped first, and chunks \
             are found by keyword search until they are re-embedded."
        )?;
    }
    for group in &status.stale {
        let made_with = group
            .profile
            .as_ref()
            .map_or_else(|| String::from("an unrecorded profile"), Profile::describe);
        writeln!(
            out,
            "{} chunks were embedded with {made_with}.",
            group.chunks
        )?;
    }
    if status.missing_chunks > 0 {
        writeln!(out, "{} chunks have no vector.", status.missing_chunks)?;
    }
    writeln!(
        out,
        "Re-embed {} chunks and {} graph node labels with {profile}.",
        plan.chunks, plan.nodes
    )?;
    Ok(())
}

fn write_summary(out: &mut impl Write, summary: &Summary, profile: &str) -> Result<()> {
    if let Some(from) = summary.retyped_from {
        writeln!(out, "Dropped the {from}-dimensional vectors.")?;
    }
    writeln!(
        out,
        "Re-embedded {} chunks and {} graph node labels with {profile}.",
        summary.chunks, summary.nodes
    )?;
    Ok(())
}

/// Progress on stderr, for the shell.
pub(crate) fn print_progress(progress: Progress) {
    let what = match progress.stage {
        Stage::Chunks => "chunks",
        Stage::Nodes => "graph node labels",
    };
    if progress.total == 0 {
        return;
    }
    drop(writeln!(
        std::io::stderr(),
        "{what}: {}/{} re-embedded",
        progress.done,
        progress.total
    ));
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;
    use quack_core::embedding::Prompts;
    use quack_core::storage::workspace::StaleVectors;

    #[test]
    fn the_plan_names_the_old_profile_the_missing_and_a_width_change() {
        let old = Profile::new("nomic-embed-text", 768, Prompts::default());
        let status = EmbeddingStatus {
            profile: Some(Profile::new("embeddinggemma", 1024, Prompts::default())),
            column_dimension: 768,
            current_chunks: 0,
            missing_chunks: 2,
            stale: vec![StaleVectors {
                profile: Some(old),
                chunks: 40,
            }],
            stale_nodes: 3,
        };
        let plan = Plan::from_status(&status);
        let mut out = Vec::new();
        write_plan(
            &mut out,
            &plan,
            &status,
            "embeddinggemma (1024 dimensions, no prefixes)",
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("stores 768-dimensional vectors"), "{text}");
        assert!(
            text.contains(
                "40 chunks were embedded with nomic-embed-text (768 dimensions, no prefixes)."
            ),
            "{text}"
        );
        assert!(text.contains("2 chunks have no vector."), "{text}");
        assert!(
            text.contains("Re-embed 42 chunks and 3 graph node labels"),
            "{text}"
        );
    }
}
