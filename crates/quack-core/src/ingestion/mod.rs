pub mod chunker;
pub mod parser;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::llm::EmbeddingProvider;
use crate::storage::workspace::WorkspaceDb;

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
pub async fn ingest_file<P: EmbeddingProvider>(
    config: &Config,
    db: &WorkspaceDb,
    workspace_id: &str,
    filename: &str,
    data: &[u8],
    provider: Option<&P>,
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

            let chunk_count = embed_and_store(db, &doc_id, &chunks, provider).await?;

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
    let path_str = dest.to_string_lossy();

    let create_sql = match file_type {
        parser::FileType::Csv => {
            format!(
                "CREATE OR REPLACE TABLE \"{table_name}\" AS SELECT * FROM read_csv_auto('{path_str}')"
            )
        }
        parser::FileType::Parquet => {
            format!(
                "CREATE OR REPLACE TABLE \"{table_name}\" AS SELECT * FROM read_parquet('{path_str}')"
            )
        }
        parser::FileType::Json => {
            format!(
                "CREATE OR REPLACE TABLE \"{table_name}\" AS SELECT * FROM read_json_auto('{path_str}')"
            )
        }
        _ => return Err(Error::Ingestion("not a structured file type".into())),
    };

    db.execute_statement(&create_sql)?;
    tracing::info!(table = %table_name, file = %filename, "created table from structured file");

    Ok(table_name)
}

async fn embed_and_store<P: EmbeddingProvider>(
    db: &WorkspaceDb,
    document_id: &str,
    chunks: &[String],
    provider: Option<&P>,
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

    if let Some(prov) = provider {
        let batch_size = 64usize;
        let mut offset = 0usize;

        while offset < chunks.len() {
            let end = chunks.len().min(offset.saturating_add(batch_size));
            let batch_texts: Vec<&str> = chunks
                .get(offset..end)
                .ok_or_else(|| Error::Ingestion("batch slice out of bounds".into()))?
                .iter()
                .map(String::as_str)
                .collect();

            let embeddings = prov.embed(&batch_texts).await?;

            for (j, embedding) in embeddings.into_iter().enumerate() {
                let chunk_idx = u32::try_from(offset.saturating_add(j))
                    .map_err(|_| Error::Ingestion("chunk index overflow".into()))?;
                db.update_chunk_embedding(document_id, chunk_idx, &embedding)?;
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
