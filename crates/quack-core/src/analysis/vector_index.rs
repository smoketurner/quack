use rig::vector_store::request::Filter;
use rig::vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndex};
use serde::Deserialize;
use serde_json::json;

use super::tools::ReaderDb;
use crate::embedding::{Embedder, Input};
use crate::storage::workspace::{ChunkScope, ChunkSearchResult};

/// Chunks returned when a request's sample count does not fit.
const DEFAULT_SAMPLES: u32 = 5;

pub struct DuckDbVectorIndex<M> {
    db: ReaderDb,
    embedder: Embedder<M>,
}

impl<M> DuckDbVectorIndex<M> {
    pub fn new(db: ReaderDb, embedder: Embedder<M>) -> Self {
        Self { db, embedder }
    }
}

/// A store error rig can carry, from any of ours.
fn store_error(e: impl std::fmt::Display) -> VectorStoreError {
    VectorStoreError::datastore(std::io::Error::other(e.to_string()))
}

impl<M> DuckDbVectorIndex<M>
where
    M: rig::embeddings::EmbeddingModel + Send + Sync,
{
    /// The chunks nearest the request's query, across the workspace.
    async fn search(
        &self,
        req: &VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> Result<Vec<ChunkSearchResult>, VectorStoreError> {
        let query_vec = self
            .embedder
            .embed_one(&Input::Query(req.query().to_owned()))
            .await
            .map_err(store_error)?;
        let samples = u32::try_from(req.samples()).unwrap_or(DEFAULT_SAMPLES);
        self.db
            .with_db(move |db| db.search_similar_chunks(&query_vec, samples, &ChunkScope::all()))
            .await
            .map_err(store_error)
    }
}

impl<M> VectorStoreIndex for DuckDbVectorIndex<M>
where
    M: rig::embeddings::EmbeddingModel + Send + Sync,
{
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        self.search(&req)
            .await?
            .into_iter()
            .map(|chunk| {
                let value = json!({
                    "content": chunk.content,
                    "source_document": chunk.document_id,
                    "filename": chunk.filename,
                });
                let doc: T = serde_json::from_value(value)?;
                Ok((chunk.score, chunk.id.into_string(), doc))
            })
            .collect()
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(self
            .search(&req)
            .await?
            .into_iter()
            .map(|chunk| (chunk.score, chunk.id.into_string()))
            .collect())
    }
}
