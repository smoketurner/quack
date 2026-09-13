pub mod openai_compat;

use crate::error::Result;

/// A provider that can generate vector embeddings for text.
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embeddings for a batch of texts.
    ///
    /// Returns one vector per input text, each of dimension
    /// [`embedding_dimension`](Self::embedding_dimension).
    fn embed(&self, texts: &[&str]) -> impl Future<Output = Result<Vec<Vec<f32>>>> + Send;

    /// The fixed dimension of vectors returned by [`embed`](Self::embed).
    fn embedding_dimension(&self) -> u32;

    /// The model name used for embeddings.
    fn model_name(&self) -> &str;
}

use std::future::Future;
