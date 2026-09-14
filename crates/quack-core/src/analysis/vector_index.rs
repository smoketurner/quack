use rig::vector_store::request::Filter;
use rig::vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndex};
use serde::Deserialize;
use serde_json::json;

use super::tools::SharedDb;

pub struct DuckDbVectorIndex<M> {
    db: SharedDb,
    embedding_model: M,
}

impl<M> DuckDbVectorIndex<M> {
    pub fn new(db: SharedDb, embedding_model: M) -> Self {
        Self {
            db,
            embedding_model,
        }
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
        let embedding = self.embedding_model.embed_text(req.query()).await?;

        #[expect(
            clippy::cast_possible_truncation,
            reason = "f64 -> f32 is acceptable for embedding vectors stored in DuckDB"
        )]
        let query_vec: Vec<f32> = embedding.vec.into_iter().map(|v| v as f32).collect();

        let samples = u32::try_from(req.samples()).unwrap_or(5);

        let results = {
            let db = self.db.lock().map_err(|_| {
                VectorStoreError::datastore(std::io::Error::other("mutex poisoned"))
            })?;
            db.search_similar_chunks(&query_vec, samples, &[])
                .map_err(|e| VectorStoreError::datastore(std::io::Error::other(e.to_string())))?
        };

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
        let embedding = self.embedding_model.embed_text(req.query()).await?;

        #[expect(
            clippy::cast_possible_truncation,
            reason = "f64 -> f32 is acceptable for embedding vectors stored in DuckDB"
        )]
        let query_vec: Vec<f32> = embedding.vec.into_iter().map(|v| v as f32).collect();

        let samples = u32::try_from(req.samples()).unwrap_or(5);

        let results = {
            let db = self.db.lock().map_err(|_| {
                VectorStoreError::datastore(std::io::Error::other("mutex poisoned"))
            })?;
            db.search_similar_chunks(&query_vec, samples, &[])
                .map_err(|e| VectorStoreError::datastore(std::io::Error::other(e.to_string())))?
        };

        Ok(results
            .into_iter()
            .map(|chunk| (chunk.score, chunk.id))
            .collect())
    }
}
