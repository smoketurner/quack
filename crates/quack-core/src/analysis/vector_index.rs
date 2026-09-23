use rig::vector_store::request::Filter;
use rig::vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndex};
use serde::Deserialize;
use serde_json::json;

use super::tools::ReaderDb;
use crate::embedding::{Embedder, Input, Vector};
use crate::storage::workspace::ChunkScope;

pub struct DuckDbVectorIndex<M> {
    db: ReaderDb,
    embedder: Embedder<M>,
}

impl<M> DuckDbVectorIndex<M> {
    pub fn new(db: ReaderDb, embedder: Embedder<M>) -> Self {
        Self { db, embedder }
    }
}

impl<M> DuckDbVectorIndex<M>
where
    M: rig::embeddings::EmbeddingModel + Send + Sync,
{
    async fn query_vector(&self, query: &str) -> Result<Vector, VectorStoreError> {
        self.embedder
            .embed_one(&Input::Query(query.to_owned()))
            .await
            .map_err(|e| VectorStoreError::datastore(std::io::Error::other(e.to_string())))
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
        let query_vec = self.query_vector(req.query()).await?;

        let samples = u32::try_from(req.samples()).unwrap_or(5);

        let results = self
            .db
            .with_db(move |db| db.search_similar_chunks(&query_vec, samples, &ChunkScope::all()))
            .await
            .map_err(|e| VectorStoreError::datastore(std::io::Error::other(e.to_string())))?;

        results
            .into_iter()
            .map(|chunk| {
                let score = chunk.score;
                let value = json!({
                    "content": chunk.content,
                    "source_document": chunk.document_id,
                    "filename": chunk.filename,
                });
                let doc: T = serde_json::from_value(value)?;
                Ok((score, chunk.id, doc))
            })
            .collect()
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let query_vec = self.query_vector(req.query()).await?;

        let samples = u32::try_from(req.samples()).unwrap_or(5);

        let results = self
            .db
            .with_db(move |db| db.search_similar_chunks(&query_vec, samples, &ChunkScope::all()))
            .await
            .map_err(|e| VectorStoreError::datastore(std::io::Error::other(e.to_string())))?;

        Ok(results
            .into_iter()
            .map(|chunk| (chunk.score, chunk.id))
            .collect())
    }
}
