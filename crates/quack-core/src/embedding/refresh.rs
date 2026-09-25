//! Bring a workspace's vectors up to the current embedding profile.
//!
//! A chunk or graph node whose vector is missing, or was made under another
//! profile, is found by keyword search only (and never merged or matched
//! by label). `run` embeds them again in place, oldest first, one batch per
//! model call and one transaction per batch, so a cancelled or failed run
//! keeps what it finished and a later run picks up the rest. When the
//! configured width differs from the stored one, the vector columns are
//! retyped first, which drops every stored vector: the operator asked for
//! exactly that by running it.

use std::fmt;
use std::time::Instant;

use rig::embeddings::EmbeddingModel;

use super::{Dimension, Embedder, EmbeddingStatus, Input, Profile, StaleVectors};
use crate::error::{Error, Result};
use crate::graph::{resolve, store as graph_store};
use crate::progress::{ChunkDone, RunControl};
use crate::storage::workspace::PendingChunk;
use crate::storage::writer::Writer;

/// A change of vector width: the columns hold `stored`-wide vectors and the
/// configured model makes `configured`-wide ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Retype {
    pub stored: Dimension,
    pub configured: Dimension,
}

/// What a run will do, for the caller to show before starting it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Plan {
    /// The profile the vectors will be made under; `None` without a model.
    pub profile: Option<Profile>,
    /// The stale chunk vectors, by the profile they were made under.
    pub stale: Vec<StaleVectors>,
    /// Chunks with no vector.
    pub missing_chunks: u64,
    /// Chunks with a missing or stale vector (every chunk when the width
    /// changes).
    pub chunks: u64,
    /// Graph nodes with a stale label vector. Nodes never embedded get
    /// one too, as graph resolution would give them.
    pub nodes: u64,
    /// When the columns must be retyped, dropping every stored vector.
    pub retype: Option<Retype>,
}

impl Plan {
    /// The plan for a workspace in `status`.
    #[must_use]
    pub fn from_status(status: &EmbeddingStatus) -> Self {
        let retype = status
            .profile
            .as_ref()
            .filter(|p| p.dimension != status.column_dimension)
            .map(|p| Retype {
                stored: status.column_dimension,
                configured: p.dimension,
            });
        let rewritten = if retype.is_some() {
            status.current_chunks
        } else {
            0
        };
        Self {
            profile: status.profile.clone(),
            stale: status.stale.clone(),
            missing_chunks: status.missing_chunks,
            chunks: status
                .stale_chunks()
                .saturating_add(status.missing_chunks)
                .saturating_add(rewritten),
            nodes: status.nodes_needing_embedding,
            retype,
        }
    }

    /// Whether there is anything to do.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks == 0 && self.nodes == 0 && self.retype.is_none()
    }
}

/// The plan as the operator reads it before agreeing to it.
impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(Retype { stored, configured }) = self.retype {
            writeln!(
                f,
                "The workspace stores {stored}-dimensional vectors and the configured model \
                 makes {configured}-dimensional ones: every stored vector is dropped first, and \
                 chunks are found by keyword search until they are refreshed."
            )?;
        }
        for group in &self.stale {
            writeln!(
                f,
                "{} chunks were embedded with {}.",
                group.chunks,
                group.made_with()
            )?;
        }
        if self.missing_chunks > 0 {
            writeln!(f, "{} chunks have no vector.", self.missing_chunks)?;
        }
        write!(
            f,
            "Refresh {} chunks and {} graph node labels",
            self.chunks, self.nodes
        )?;
        match &self.profile {
            Some(profile) => write!(f, " with {profile}."),
            None => f.write_str("."),
        }
    }
}

/// What a run did.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Summary {
    /// The profile the vectors were made under.
    pub profile: Profile,
    /// The width the columns had, when they were retyped.
    pub retyped_from: Option<Dimension>,
    pub chunks: u64,
    pub nodes: u64,
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(from) = self.retyped_from {
            writeln!(f, "Dropped the {from}-dimensional vectors.")?;
        }
        write!(
            f,
            "Refreshed {} chunks and {} graph node labels with {}.",
            self.chunks, self.nodes, self.profile
        )
    }
}

/// The workspace once it is ready to take vectors of the embedder's width.
struct Prepared {
    status: EmbeddingStatus,
    retyped_from: Option<Dimension>,
    nodes: u32,
}

impl Prepared {
    /// Check the workspace runs under `embedder`'s profile, retype its
    /// vector columns when the width changed, and count what there is to
    /// do.
    async fn load<M: EmbeddingModel>(db: &Writer, embedder: &Embedder<M>) -> Result<Self> {
        let fingerprint = embedder.profile().fingerprint();
        let dimension = embedder.profile().dimension;
        db.run(move |db| {
            if db.embedding_fingerprint() != Some(&fingerprint) {
                return Err(Error::Config(
                    "the workspace was opened under a different embedding profile than the \
                     model refreshing it; reopen it with the current configuration"
                        .into(),
                ));
            }
            let column = db.embedding_dimension();
            let retyped_from = if column == dimension {
                None
            } else {
                tracing::info!(from = %column, to = %dimension, "retyping the vector columns");
                db.retype_vectors(dimension)?;
                Some(column)
            };
            Ok(Self {
                status: db.embedding_status()?,
                retyped_from,
                nodes: graph_store::count_nodes_needing_embedding(db)?,
            })
        })
        .await
    }
}

/// Embed again every chunk and node whose vector is missing or stale, in
/// batches of `batch_size`. Progress counts chunks and node labels
/// together; a cancel stops between batches, keeping what was done.
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
    control: RunControl<'_>,
) -> Result<Summary> {
    let started = Instant::now();
    let Prepared {
        status,
        retyped_from,
        nodes,
    } = Prepared::load(db, embedder).await?;
    // Progress counts in `u32`, like every run's; the summary keeps the
    // exact counts.
    let chunk_total = u32::try_from(status.stale_chunks().saturating_add(status.missing_chunks))
        .unwrap_or(u32::MAX);
    let total = chunk_total.saturating_add(nodes);
    let batch = batch_size.max(1);
    let mut chunks: u64 = 0;
    let mut chunks_done: u32 = 0;
    loop {
        control.check()?;
        let batch_started = Instant::now();
        let pending = db.run(move |db| db.chunks_needing_embedding(batch)).await?;
        if pending.is_empty() {
            break;
        }
        let inputs: Vec<Input> = pending.iter().map(PendingChunk::embedding_input).collect();
        let vectors = embedder.embed(&inputs).await?;
        let count = pending.len();
        db.run(move |db| {
            db.write_transaction(|db| {
                for (chunk, vector) in pending.iter().zip(&vectors) {
                    db.set_chunk_embedding(&chunk.id, vector)?;
                }
                Ok(())
            })
        })
        .await?;
        chunks = chunks.saturating_add(u64::try_from(count).unwrap_or(u64::MAX));
        chunks_done = u32::try_from(chunks).unwrap_or(u32::MAX);
        (control.progress)(ChunkDone {
            done: chunks_done,
            total: total.max(chunks_done),
            failed: 0,
            took: batch_started.elapsed(),
            elapsed: started.elapsed(),
        });
    }

    // Node batches report on the same scale, after the chunks.
    let on_nodes = |done: ChunkDone| {
        let done_total = chunks_done.saturating_add(done.done);
        (control.progress)(ChunkDone {
            done: done_total,
            total: total.max(done_total),
            elapsed: started.elapsed(),
            ..done
        });
    };
    let nodes = resolve::embed_nodes(
        db,
        embedder,
        RunControl {
            progress: &on_nodes,
            cancel: control.cancel,
        },
    )
    .await?;
    Ok(Summary {
        profile: embedder.profile().clone(),
        retyped_from,
        chunks,
        nodes: u64::from(nodes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::Prompts;

    fn status(
        profile_dim: Option<u32>,
        column: u32,
        current: u64,
        missing: u64,
        stale: u64,
    ) -> EmbeddingStatus {
        EmbeddingStatus {
            profile: profile_dim.map(|d| Profile::new("m", Dimension::new(d), Prompts::default())),
            column_dimension: Dimension::new(column),
            current_chunks: current,
            missing_chunks: missing,
            stale: if stale == 0 {
                Vec::new()
            } else {
                vec![StaleVectors {
                    profile: None,
                    chunks: stale,
                }]
            },
            stale_nodes: 0,
            nodes_needing_embedding: 0,
        }
    }

    #[test]
    fn a_plan_counts_stale_and_missing_and_everything_on_a_width_change() {
        let same = Plan::from_status(&status(Some(4), 4, 10, 2, 3));
        assert_eq!(same.chunks, 5);
        assert_eq!(same.retype, None);
        let wider = Plan::from_status(&status(Some(8), 4, 0, 2, 3));
        assert_eq!(
            wider.retype,
            Some(Retype {
                stored: Dimension::new(4),
                configured: Dimension::new(8)
            })
        );
        assert_eq!(wider.chunks, 5);
        assert!(Plan::from_status(&status(Some(4), 4, 10, 0, 0)).is_empty());
    }

    #[test]
    fn the_plan_reads_as_the_old_profile_the_missing_and_a_width_change() {
        let text = Plan::from_status(&EmbeddingStatus {
            profile: Some(Profile::new(
                "embeddinggemma",
                Dimension::new(1024),
                Prompts::default(),
            )),
            column_dimension: Dimension::new(768),
            current_chunks: 0,
            missing_chunks: 2,
            stale: vec![StaleVectors {
                profile: Some(Profile::new(
                    "nomic-embed-text",
                    Dimension::new(768),
                    Prompts::default(),
                )),
                chunks: 40,
            }],
            stale_nodes: 3,
            nodes_needing_embedding: 3,
        })
        .to_string();
        assert_eq!(
            text,
            "The workspace stores 768-dimensional vectors and the configured model makes \
             1024-dimensional ones: every stored vector is dropped first, and chunks are found \
             by keyword search until they are refreshed.\n\
             40 chunks were embedded with nomic-embed-text (768 dimensions, no prefixes).\n\
             2 chunks have no vector.\n\
             Refresh 42 chunks and 3 graph node labels with embeddinggemma (1024 dimensions, \
             no prefixes)."
        );
    }
}
