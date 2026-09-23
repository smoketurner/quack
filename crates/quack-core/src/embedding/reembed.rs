//! Bring a workspace's vectors up to the current embedding profile.
//!
//! A chunk or graph node whose vector is missing, or was made under another
//! profile, is found by keyword search only (and never merged or matched
//! by label). `run` re-embeds them in place, oldest first, one batch per
//! model call and one transaction per batch, so a cancelled or failed run
//! keeps what it finished and a later run picks up the rest. When the
//! configured width differs from the stored one, the vector columns are
//! retyped first, which drops every stored vector: the operator asked for
//! exactly that by running it.

use rig::embeddings::EmbeddingModel;
use tokio_util::sync::CancellationToken;

use super::{DocumentInput, Embedder};
use crate::error::{Error, Result};
use crate::graph::resolve;
use crate::storage::workspace::EmbeddingStatus;
use crate::storage::writer::Writer;

/// Which vectors a run is working on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Chunks,
    Nodes,
}

/// Where a run is, after each batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    pub stage: Stage,
    pub done: u32,
    /// The stage's count when the run started; chunks ingested meanwhile
    /// are embedded by their own ingest.
    pub total: u32,
}

/// What a run will do, for the caller to show before starting it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Plan {
    /// Chunks with a missing or stale vector.
    pub chunks: u64,
    /// Graph nodes with a stale label vector. Nodes never embedded get
    /// one too, as graph resolution would give them.
    pub nodes: u64,
    /// `(stored, configured)` widths when the columns must be retyped,
    /// dropping every stored vector.
    pub retype: Option<(u32, u32)>,
}

impl Plan {
    /// The plan for a workspace in `status`.
    #[must_use]
    pub fn from_status(status: &EmbeddingStatus) -> Self {
        let retype = status
            .profile
            .as_ref()
            .filter(|p| p.dimension != status.column_dimension)
            .map(|p| (status.column_dimension, p.dimension));
        Self {
            chunks: status
                .stale_chunks()
                .saturating_add(status.missing_chunks)
                .saturating_add(if retype.is_some() {
                    status.current_chunks
                } else {
                    0
                }),
            nodes: status.stale_nodes,
            retype,
        }
    }

    /// Whether there is anything to do.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks == 0 && self.nodes == 0 && self.retype.is_none()
    }
}

/// What a run did.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct Summary {
    /// The width the columns had, when they were retyped.
    pub retyped_from: Option<u32>,
    pub chunks: u32,
    pub nodes: u32,
}

/// Re-embed every chunk and node whose vector is missing or stale, in
/// batches of `batch_size`. `progress` hears after every batch; `cancel`
/// stops between batches with [`Error::Cancelled`], keeping what was done.
///
/// # Errors
///
/// Returns an error when the workspace was opened under another profile
/// than `embedder`'s, when embedding or a write fails, or
/// [`Error::Cancelled`].
pub async fn run<M: EmbeddingModel>(
    db: &Writer,
    embedder: &Embedder<M>,
    batch_size: u32,
    progress: &(dyn Fn(Progress) + Sync),
    cancel: Option<&CancellationToken>,
) -> Result<Summary> {
    let fingerprint = embedder.profile().fingerprint();
    let dimension = embedder.profile().dimension;
    let (status, retyped_from) = db
        .run(move |db| {
            if db.embedding_fingerprint() != Some(fingerprint.as_str()) {
                return Err(Error::Config(
                    "the workspace was opened under a different embedding profile than the \
                     model re-embedding it; reopen it with the current configuration"
                        .into(),
                ));
            }
            let status = db.embedding_status()?;
            let column = status.column_dimension;
            if column == dimension {
                return Ok((status, None));
            }
            tracing::info!(from = column, to = dimension, "retyping the vector columns");
            db.retype_vectors(dimension)?;
            Ok((db.embedding_status()?, Some(column)))
        })
        .await?;
    let mut summary = Summary {
        retyped_from,
        ..Summary::default()
    };

    let total = u32::try_from(status.stale_chunks().saturating_add(status.missing_chunks))
        .unwrap_or(u32::MAX);
    let batch = batch_size.max(1);
    progress(Progress {
        stage: Stage::Chunks,
        done: 0,
        total,
    });
    loop {
        if cancel.is_some_and(CancellationToken::is_cancelled) {
            return Err(Error::Cancelled);
        }
        let pending = db.run(move |db| db.chunks_needing_embedding(batch)).await?;
        if pending.is_empty() {
            break;
        }
        let inputs: Vec<DocumentInput> = pending
            .iter()
            .map(|chunk| DocumentInput {
                title: chunk.heading.clone(),
                text: chunk.content.clone(),
            })
            .collect();
        let vectors = embedder.documents(&inputs).await?;
        let count = u32::try_from(pending.len()).unwrap_or(u32::MAX);
        db.run(move |db| {
            db.write_transaction(|db| {
                for (chunk, vector) in pending.iter().zip(&vectors) {
                    db.set_chunk_embedding(&chunk.id, vector)?;
                }
                Ok(())
            })
        })
        .await?;
        summary.chunks = summary.chunks.saturating_add(count);
        progress(Progress {
            stage: Stage::Chunks,
            done: summary.chunks,
            total: total.max(summary.chunks),
        });
    }

    let nodes_total = u32::try_from(status.stale_nodes).unwrap_or(u32::MAX);
    progress(Progress {
        stage: Stage::Nodes,
        done: 0,
        total: nodes_total,
    });
    let on_nodes = |done: u32| {
        progress(Progress {
            stage: Stage::Nodes,
            done,
            total: nodes_total.max(done),
        });
    };
    summary.nodes = resolve::embed_nodes(db, embedder, &on_nodes, cancel).await?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{Profile, Prompts};

    fn status(
        profile_dim: Option<u32>,
        column: u32,
        current: u64,
        missing: u64,
        stale: u64,
    ) -> EmbeddingStatus {
        EmbeddingStatus {
            profile: profile_dim.map(|d| Profile::new("m", d, Prompts::default())),
            column_dimension: column,
            current_chunks: current,
            missing_chunks: missing,
            stale: if stale == 0 {
                Vec::new()
            } else {
                vec![crate::storage::workspace::StaleVectors {
                    profile: None,
                    chunks: stale,
                }]
            },
            stale_nodes: 0,
        }
    }

    #[test]
    fn a_plan_counts_stale_and_missing_and_everything_on_a_width_change() {
        let same = Plan::from_status(&status(Some(4), 4, 10, 2, 3));
        assert_eq!(same.chunks, 5);
        assert_eq!(same.retype, None);
        let wider = Plan::from_status(&status(Some(8), 4, 0, 2, 3));
        assert_eq!(wider.retype, Some((4, 8)));
        assert_eq!(wider.chunks, 5);
        assert!(Plan::from_status(&status(Some(4), 4, 10, 0, 0)).is_empty());
    }
}
