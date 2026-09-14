pub mod chunker;
pub mod parser;

use std::sync::{Arc, Mutex};

use rig::embeddings::EmbeddingModel;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::storage::workspace::{NewChunk, WorkspaceDb, quote_ident};

/// Access to a workspace database for ingestion: a bare handle, or a
/// shared one that is locked only around each database step so embedding
/// calls run with the workspace free for other requests.
pub trait DbHandle {
    /// Run `f` against the database.
    ///
    /// # Errors
    ///
    /// Returns `f`'s error, or an error when a shared handle is poisoned.
    fn with<R>(&self, f: impl FnOnce(&WorkspaceDb) -> Result<R>) -> Result<R>;
}

impl DbHandle for WorkspaceDb {
    fn with<R>(&self, f: impl FnOnce(&WorkspaceDb) -> Result<R>) -> Result<R> {
        f(self)
    }
}

impl DbHandle for Arc<Mutex<WorkspaceDb>> {
    fn with<R>(&self, f: impl FnOnce(&WorkspaceDb) -> Result<R>) -> Result<R> {
        let guard = self
            .lock()
            .map_err(|e| Error::Ingestion(format!("workspace mutex poisoned: {e}")))?;
        f(&guard)
    }
}

/// Result of ingesting a single file into a workspace.
#[derive(Debug)]
pub struct IngestResult {
    pub document_id: String,
    pub filename: String,
    pub file_type: parser::FileType,
    pub chunks_stored: u32,
    pub table_name: Option<String>,
}

/// Ingest a file into a workspace, producing either a `DuckDB` table (structured)
/// or embedded chunks (unstructured). Registers the document and processes
/// it in one go; the server registers first and processes from its queue.
///
/// # Errors
///
/// Returns an error if the file type is unsupported or the file cannot be
/// parsed or stored; the document row then carries the error.
pub async fn ingest_file<M: EmbeddingModel, D: DbHandle>(
    config: &Config,
    db: &D,
    workspace_id: &str,
    filename: &str,
    data: &[u8],
    embedding_model: Option<&M>,
) -> Result<IngestResult> {
    let doc_id = db.with(|db| register_document(db, filename, data.len()))?;
    process_document(
        config,
        db,
        workspace_id,
        &doc_id,
        filename,
        data,
        embedding_model,
    )
    .await
}

/// Insert the document row with status `queued` and return its id. Fails
/// before writing anything for a file type nothing can parse.
///
/// # Errors
///
/// Returns `UnsupportedFileType` or a storage error.
pub fn register_document(db: &WorkspaceDb, filename: &str, size_bytes: usize) -> Result<String> {
    let file_type = parser::detect_file_type(filename);
    if matches!(file_type, parser::FileType::Unknown) {
        return Err(Error::UnsupportedFileType(filename.to_owned()));
    }
    let doc_id = uuid::Uuid::now_v7().to_string();
    db.insert_document(
        &doc_id,
        filename,
        file_type.mime_type(),
        size_bytes,
        "queued",
    )?;
    Ok(doc_id)
}

/// Parse, store, and embed a registered document, moving its status from
/// `processing` to `ready`, or to `error` with the message when it fails.
///
/// # Errors
///
/// Returns the failure after recording it on the document row.
pub async fn process_document<M: EmbeddingModel, D: DbHandle>(
    config: &Config,
    db: &D,
    workspace_id: &str,
    doc_id: &str,
    filename: &str,
    data: &[u8],
    embedding_model: Option<&M>,
) -> Result<IngestResult> {
    db.with(|db| db.update_document_status(doc_id, "processing"))?;
    let outcome = process_inner(
        config,
        db,
        workspace_id,
        doc_id,
        filename,
        data,
        embedding_model,
    )
    .await;
    match &outcome {
        Ok(_) => db.with(|db| db.update_document_status(doc_id, "ready"))?,
        Err(e) => db.with(|db| db.mark_document_error(doc_id, &e.to_string()))?,
    }
    outcome
}

async fn process_inner<M: EmbeddingModel, D: DbHandle>(
    config: &Config,
    db: &D,
    workspace_id: &str,
    doc_id: &str,
    filename: &str,
    data: &[u8],
    embedding_model: Option<&M>,
) -> Result<IngestResult> {
    let file_type = parser::detect_file_type(filename);
    match file_type {
        parser::FileType::Csv | parser::FileType::Parquet | parser::FileType::Json => {
            let table_name = db.with(|db| {
                ingest_structured(config, db, workspace_id, filename, data, &file_type)
            })?;
            Ok(IngestResult {
                document_id: doc_id.to_owned(),
                filename: filename.to_owned(),
                file_type,
                chunks_stored: 0,
                table_name: Some(table_name),
            })
        }
        parser::FileType::Pdf | parser::FileType::Text | parser::FileType::Markdown => {
            let sections = parser::extract_sections(&file_type, data)?;
            let chunks = chunker::chunk_sections(
                &sections,
                config.ingestion.chunk_size_tokens,
                config.ingestion.chunk_overlap_tokens,
                &config.ingestion.tokenizer_encoding,
            )?;
            let chunk_count = embed_and_store(db, doc_id, &chunks, embedding_model).await?;
            Ok(IngestResult {
                document_id: doc_id.to_owned(),
                filename: filename.to_owned(),
                file_type,
                chunks_stored: chunk_count,
                table_name: None,
            })
        }
        parser::FileType::Unknown => Err(Error::UnsupportedFileType(filename.to_owned())),
    }
}

/// The table a structured file loads into: its stem with anything outside
/// `[A-Za-z0-9_]` replaced by `_`.
#[must_use]
pub fn table_name_for(filename: &str) -> String {
    sanitize_table_name(filename)
}

/// Write the bytes under `files/` and load them as a table with `DuckDB`'s
/// reader for the type. The path is bound, never interpolated.
fn ingest_structured(
    config: &Config,
    db: &WorkspaceDb,
    workspace_id: &str,
    filename: &str,
    data: &[u8],
    file_type: &parser::FileType,
) -> Result<String> {
    let files_dir = config.workspace_files_dir(workspace_id);
    std::fs::create_dir_all(&files_dir)?;
    let dest = files_dir.join(
        std::path::Path::new(filename)
            .file_name()
            .ok_or_else(|| Error::Ingestion(format!("'{filename}' is not a file name")))?,
    );
    std::fs::write(&dest, data)?;

    let table_name = sanitize_table_name(filename);
    let path = dest.to_string_lossy();

    let reader = match file_type {
        parser::FileType::Csv => "read_csv_auto",
        parser::FileType::Parquet => "read_parquet",
        parser::FileType::Json => "read_json_auto",
        parser::FileType::Pdf
        | parser::FileType::Text
        | parser::FileType::Markdown
        | parser::FileType::Unknown => {
            return Err(Error::Ingestion("not a structured file type".into()));
        }
    };

    let create_sql = format!(
        "CREATE OR REPLACE TABLE {} AS SELECT * FROM {reader}(?)",
        quote_ident(&table_name)
    );
    db.execute_with_params(&create_sql, duckdb::params![path.as_ref()])?;
    tracing::info!(table = %table_name, file = %filename, "created table from structured file");

    Ok(table_name)
}

async fn embed_and_store<M: EmbeddingModel, D: DbHandle>(
    db: &D,
    document_id: &str,
    chunks: &[chunker::Chunk],
    embedding_model: Option<&M>,
) -> Result<u32> {
    let stored = db.with(|db| {
        let mut stored: u32 = 0;
        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_id = uuid::Uuid::now_v7().to_string();
            let idx =
                u32::try_from(i).map_err(|_| Error::Ingestion("chunk index overflow".into()))?;
            db.insert_chunk(&NewChunk {
                id: &chunk_id,
                document_id,
                chunk_index: idx,
                content: &chunk.content,
                heading: chunk.heading.as_deref(),
                page: chunk.page,
                embedding: None,
            })?;
            stored = stored
                .checked_add(1)
                .ok_or_else(|| Error::Ingestion("chunk count overflow".into()))?;
        }
        Ok(stored)
    })?;

    if let Some(model) = embedding_model {
        let batch_size = 64usize;
        let mut offset = 0usize;

        while offset < chunks.len() {
            let end = chunks.len().min(offset.saturating_add(batch_size));
            let batch_texts: Vec<String> = chunks
                .get(offset..end)
                .ok_or_else(|| Error::Ingestion("batch slice out of bounds".into()))?
                .iter()
                .map(chunker::Chunk::embedding_input)
                .collect();

            let embeddings = model
                .embed_texts(batch_texts)
                .await
                .map_err(|e| Error::Embedding(e.to_string()))?;

            db.with(|db| {
                for (j, embedding) in embeddings.into_iter().enumerate() {
                    let chunk_idx = u32::try_from(offset.saturating_add(j))
                        .map_err(|_| Error::Ingestion("chunk index overflow".into()))?;

                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "f64 -> f32 is acceptable for embedding vectors stored in DuckDB"
                    )]
                    let vec_f32: Vec<f32> = embedding.vec.into_iter().map(|v| v as f32).collect();

                    db.update_chunk_embedding(document_id, chunk_idx, &vec_f32)?;
                }
                Ok(())
            })?;

            offset = end;
        }

        tracing::info!(
            document_id = %document_id,
            chunk_count = %stored,
            "embedded chunks"
        );
    }

    Ok(stored)
}

fn sanitize_table_name(filename: &str) -> String {
    let stem = std::path::Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("imported");

    stem.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_strips_extension() {
        assert_eq!(sanitize_table_name("data.csv"), "data");
    }

    #[test]
    fn sanitize_replaces_non_alphanumeric() {
        assert_eq!(sanitize_table_name("my-data file.csv"), "my_data_file");
    }

    #[test]
    fn sanitize_preserves_underscores() {
        assert_eq!(sanitize_table_name("my_data.json"), "my_data");
    }

    #[test]
    fn sanitize_no_extension() {
        assert_eq!(sanitize_table_name("readme"), "readme");
    }

    #[test]
    fn sanitize_empty_uses_fallback() {
        assert_eq!(sanitize_table_name(""), "imported");
    }

    #[test]
    fn sanitize_dotfile_replaces_leading_dot() {
        assert_eq!(sanitize_table_name(".hidden"), "_hidden");
    }

    #[test]
    fn sanitize_multiple_extensions() {
        assert_eq!(sanitize_table_name("data.2024.csv"), "data_2024");
    }

    #[test]
    fn sanitize_preserves_alphanumeric() {
        assert_eq!(sanitize_table_name("Sales2024.parquet"), "Sales2024");
    }
}
