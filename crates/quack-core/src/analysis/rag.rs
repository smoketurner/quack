use std::fmt::Write;

use crate::error::{Error, Result};
use crate::llm::LlmProvider;
use crate::storage::workspace::WorkspaceDb;

/// Search documents by embedding similarity and return formatted results.
///
/// # Errors
///
/// Returns an error if embedding generation or search fails.
pub async fn search_documents<P: LlmProvider>(
    db: &WorkspaceDb,
    provider: &P,
    query: &str,
    top_k: u32,
) -> Result<String> {
    let embeddings = provider.embed(&[query]).await?;
    let query_embedding = embeddings
        .into_iter()
        .next()
        .ok_or_else(|| Error::Analysis("no embedding returned for query".into()))?;

    let results = db.search_similar_chunks(&query_embedding, top_k)?;

    if results.is_empty() {
        return Ok(String::from("No matching documents found."));
    }

    let mut output = String::new();
    for (i, chunk) in results.iter().enumerate() {
        let idx = i.wrapping_add(1);
        writeln!(
            output,
            "--- Result {idx} (document: {}, distance: {:.4}) ---",
            chunk.document_id, chunk.distance,
        )?;
        writeln!(output, "{}", chunk.content)?;
        writeln!(output)?;
    }

    Ok(output)
}
