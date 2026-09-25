//! How a workspace's stored vectors stand against the configured embedding
//! profile. `WorkspaceDb::embedding_status` counts them; this is what the
//! counts mean.

use serde::Serialize;

use super::{Dimension, Profile};

/// Stored chunk vectors made under a profile other than the current one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StaleVectors {
    /// `None` when the fingerprint was never recorded.
    pub profile: Option<Profile>,
    pub chunks: u64,
}

/// How a workspace's vectors stand against the current embedding profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EmbeddingStatus {
    /// `None` without an embedding model.
    pub profile: Option<Profile>,
    /// Width of the stored vectors, which differs from the profile's
    /// until `quack embeddings refresh` runs after an `[embedding].dimension` change.
    pub column_dimension: Dimension,
    /// Chunks searchable by vector now.
    pub current_chunks: u64,
    /// Chunks with no vector: ingested without a model, or while the width
    /// was changing.
    pub missing_chunks: u64,
    /// Chunks with a vector from another profile, by profile: searched by
    /// keyword only.
    pub stale: Vec<StaleVectors>,
    /// Graph nodes whose label vector is from another profile.
    pub stale_nodes: u64,
    /// Graph nodes whose label vector needs re-embedding: missing
    /// (`embedding IS NULL`, as `upsert_node` leaves a freshly-added node)
    /// or made under another profile. The count `run` embeds, matching
    /// what `count_nodes_needing_embedding` returns.
    pub nodes_needing_embedding: u64,
}

impl EmbeddingStatus {
    /// Chunks with a vector from another profile.
    #[must_use]
    pub fn stale_chunks(&self) -> u64 {
        self.stale
            .iter()
            .map(|s| s.chunks)
            .fold(0, u64::saturating_add)
    }

    /// What the out-of-date chunk vectors mean for search, in one sentence,
    /// `None` when every one is current or there is no model to refresh
    /// with. Each interface adds how to refresh from where the operator is.
    #[must_use]
    pub fn note(&self) -> Option<String> {
        let profile = self.profile.as_ref()?;
        if self.stale_chunks() == 0 && self.missing_chunks == 0 {
            return None;
        }
        let mut parts = Vec::new();
        for group in &self.stale {
            parts.push(format!(
                "{} chunks were embedded with {}",
                group.chunks,
                group.made_with()
            ));
        }
        if self.missing_chunks > 0 {
            parts.push(format!("{} chunks have no vector", self.missing_chunks));
        }
        Some(format!(
            "{}; the configured model is {}, so they are found by keyword search only.",
            parts.join(", "),
            profile
        ))
    }
}

impl StaleVectors {
    /// The profile these vectors were made under, described.
    #[must_use]
    pub fn made_with(&self) -> String {
        self.profile
            .as_ref()
            .map_or_else(|| String::from("an unrecorded profile"), Profile::to_string)
    }
}
