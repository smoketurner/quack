pub mod chunker;
pub mod parser;

use rig::embeddings::EmbeddingModel;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::storage::workspace::{WorkspaceDb, quote_ident};

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
/// or embedded chunks (unstructured).
///
/// # Errors
///
/// Returns an error if the file cannot be read, parsed, or stored.
pub async fn ingest_file<M: EmbeddingModel>(
    config: &Config,
    db: &WorkspaceDb,
    workspace_id: &str,
    filename: &str,
    data: &[u8],
    embedding_model: Option<&M>,
) -> Result<IngestResult> {
    let file_type = parser::detect_file_type(filename);
    let doc_id = uuid::Uuid::now_v7().to_string();

    match file_type {
        parser::FileType::Csv | parser::FileType::Parquet | parser::FileType::Json => {
            let table_name = ingest_structured(config, db, workspace_id, filename, &file_type)?;

            db.insert_document(
                &doc_id,
                filename,
                file_type.mime_type(),
                data.len(),
                "ready",
            )?;

            Ok(IngestResult {
                document_id: doc_id,
                filename: filename.to_owned(),
                file_type,
                chunks_stored: 0,
                table_name: Some(table_name),
            })
        }

        parser::FileType::Pdf | parser::FileType::Text | parser::FileType::Markdown => {
            let text = parser::extract_text(&file_type, data)?;

            db.insert_document(
                &doc_id,
                filename,
                file_type.mime_type(),
                data.len(),
                "processing",
            )?;

            let chunks = chunker::chunk_text(
                &text,
                config.ingestion.chunk_size_tokens,
                config.ingestion.chunk_overlap_tokens,
                &config.ingestion.tokenizer_encoding,
            )?;

            let chunk_count = embed_and_store(db, &doc_id, &chunks, embedding_model).await?;

            db.update_document_status(&doc_id, "ready")?;

            Ok(IngestResult {
                document_id: doc_id,
                filename: filename.to_owned(),
                file_type,
                chunks_stored: chunk_count,
                table_name: None,
            })
        }

        parser::FileType::Unknown => Err(Error::UnsupportedFileType(filename.to_owned())),
    }
}

fn ingest_structured(
    config: &Config,
    db: &WorkspaceDb,
    workspace_id: &str,
    filename: &str,
    file_type: &parser::FileType,
) -> Result<String> {
    let files_dir = config.workspace_files_dir(workspace_id);
    std::fs::create_dir_all(&files_dir)?;

    let dest = files_dir.join(filename);
    if !dest.exists() {
        return Err(Error::Ingestion(format!(
            "file not found at {}; copy it to the workspace files directory first",
            dest.display()
        )));
    }

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

async fn embed_and_store<M: EmbeddingModel>(
    db: &WorkspaceDb,
    document_id: &str,
    chunks: &[String],
    embedding_model: Option<&M>,
) -> Result<u32> {
    let mut stored: u32 = 0;

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_id = uuid::Uuid::now_v7().to_string();
        let idx = u32::try_from(i).map_err(|_| Error::Ingestion("chunk index overflow".into()))?;

        db.insert_chunk(&chunk_id, document_id, idx, chunk, None)?;
        stored = stored
            .checked_add(1)
            .ok_or_else(|| Error::Ingestion("chunk count overflow".into()))?;
    }

    if let Some(model) = embedding_model {
        let batch_size = 64usize;
        let mut offset = 0usize;

        while offset < chunks.len() {
            let end = chunks.len().min(offset.saturating_add(batch_size));
            let batch_texts: Vec<String> = chunks
                .get(offset..end)
                .ok_or_else(|| Error::Ingestion("batch slice out of bounds".into()))?
                .to_vec();

            let embeddings = model
                .embed_texts(batch_texts)
                .await
                .map_err(|e| Error::Embedding(e.to_string()))?;

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
