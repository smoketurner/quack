pub mod chunker;
pub mod html;
pub mod office;
pub mod parser;
pub mod xlsx;

use std::sync::{Arc, Mutex};

use rig::embeddings::EmbeddingModel;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::storage::control::sha256_hex;
use crate::storage::workspace::{
    DocumentInfo, DocumentSource, NewChunk, NewDocument, WorkspaceDb, quote_ident,
};

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
    /// Tables a structured file loaded into: one, or one per workbook sheet.
    pub tables: Vec<String>,
}

/// What `ingest_file` did: stored the file, or skipped it because a
/// document with identical bytes is already in the workspace.
#[derive(Debug)]
pub enum IngestOutcome {
    Ingested(IngestResult),
    Duplicate(Box<DocumentInfo>),
}

impl IngestOutcome {
    /// The result when the file was stored.
    #[must_use]
    pub fn ingested(self) -> Option<IngestResult> {
        match self {
            Self::Ingested(result) => Some(result),
            Self::Duplicate(_) => None,
        }
    }
}

/// A file to ingest: its name and bytes, where it came from, an optional
/// title (else parsed from the content), and the server user uploading it.
#[derive(Debug, Clone, Copy)]
pub struct NewFile<'a> {
    pub filename: &'a str,
    pub data: &'a [u8],
    pub source: DocumentSource,
    pub title: Option<&'a str>,
    pub ingested_by: Option<&'a str>,
}

impl<'a> NewFile<'a> {
    /// A file read from a path with no title or uploader.
    #[must_use]
    pub fn new(filename: &'a str, data: &'a [u8]) -> Self {
        Self {
            filename,
            data,
            source: DocumentSource::Path,
            title: None,
            ingested_by: None,
        }
    }

    #[must_use]
    pub fn source(mut self, source: DocumentSource) -> Self {
        self.source = source;
        self
    }

    #[must_use]
    pub fn title(mut self, title: Option<&'a str>) -> Self {
        self.title = title;
        self
    }

    #[must_use]
    pub fn ingested_by(mut self, user: Option<&'a str>) -> Self {
        self.ingested_by = user;
        self
    }
}

/// Outcome of registering a file: a new `queued` document id, or the
/// existing document whose bytes are identical.
#[derive(Debug)]
pub enum Registration {
    New(String),
    Duplicate(Box<DocumentInfo>),
}

/// Ingest a file into a workspace, producing either a `DuckDB` table (structured)
/// or embedded chunks (unstructured). Registers the document and processes
/// it in one go; the server registers first and processes from its queue.
/// Identical bytes already in the workspace are skipped.
///
/// # Errors
///
/// Returns an error if the file type is unsupported or the file cannot be
/// parsed or stored; the document row then carries the error.
pub async fn ingest_file<M: EmbeddingModel, D: DbHandle>(
    config: &Config,
    db: &D,
    workspace_id: &str,
    file: &NewFile<'_>,
    embedding_model: Option<&M>,
) -> Result<IngestOutcome> {
    let doc_id = match db.with(|db| register_document(db, file))? {
        Registration::New(id) => id,
        Registration::Duplicate(existing) => return Ok(IngestOutcome::Duplicate(existing)),
    };
    let result = process_document(
        config,
        db,
        workspace_id,
        &doc_id,
        file.filename,
        file.data,
        embedding_model,
    )
    .await?;
    Ok(IngestOutcome::Ingested(result))
}

/// Insert the document row with status `queued` and return its id, or the
/// document already holding the same bytes (by SHA-256) so the caller can
/// skip it. Fails before writing anything for a file type nothing can parse.
///
/// # Errors
///
/// Returns `UnsupportedFileType` or a storage error.
pub fn register_document(db: &WorkspaceDb, file: &NewFile<'_>) -> Result<Registration> {
    let file_type = parser::detect_file_type(file.filename);
    if matches!(file_type, parser::FileType::Unknown) {
        return Err(Error::UnsupportedFileType(file.filename.to_owned()));
    }
    let sha256 = sha256_hex(file.data);
    if let Some(existing) = db.document_by_sha256(&sha256)? {
        if db.document_intact(&existing)? {
            return Ok(Registration::Duplicate(Box::new(existing)));
        }
        // Its table was dropped (or its chunks are gone): the row no
        // longer describes anything, so it fails and the bytes load
        // again as a new document.
        tracing::warn!(document = %existing.id, file = %existing.filename, "document lost its table or chunks; re-ingesting");
        db.mark_document_error(
            &existing.id,
            "its table or chunks were dropped; the file was ingested again",
        )?;
    }
    if matches!(
        file_type,
        parser::FileType::Csv | parser::FileType::Parquet | parser::FileType::Json
    ) {
        check_table_free(db, &sanitize_table_name(file.filename), None)?;
    }
    let doc_id = uuid::Uuid::now_v7().to_string();
    let title = file.title.map(str::trim).filter(|t| !t.is_empty());
    db.insert_document(&NewDocument {
        id: &doc_id,
        filename: file.filename,
        title,
        mime_type: file_type.mime_type(),
        size_bytes: file.data.len(),
        sha256: &sha256,
        source: file.source,
        status: "queued",
        ingested_by: file.ingested_by,
    })?;
    Ok(Registration::New(doc_id))
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
        Ok(result) => db.with(|db| {
            db.set_document_chunk_count(doc_id, result.chunks_stored)?;
            db.set_document_tables(doc_id, &result.tables)?;
            db.update_document_status(doc_id, "ready")
        })?,
        Err(e) => db.with(|db| {
            db.discard_chunks(doc_id)?;
            db.mark_document_error(doc_id, &e.to_string())
        })?,
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
                ingest_structured(config, db, workspace_id, doc_id, filename, data, &file_type)
            })?;
            Ok(IngestResult {
                document_id: doc_id.to_owned(),
                filename: filename.to_owned(),
                file_type,
                chunks_stored: 0,
                tables: vec![table_name],
            })
        }
        parser::FileType::Xlsx => {
            let tables =
                db.with(|db| ingest_workbook(config, db, workspace_id, doc_id, filename, data))?;
            Ok(IngestResult {
                document_id: doc_id.to_owned(),
                filename: filename.to_owned(),
                file_type,
                chunks_stored: 0,
                tables,
            })
        }
        parser::FileType::Pdf
        | parser::FileType::Text
        | parser::FileType::Markdown
        | parser::FileType::Html
        | parser::FileType::Docx
        | parser::FileType::Pptx => {
            let extracted = parser::extract(&file_type, data)?;
            if let Some(title) = extracted.title() {
                db.with(|db| db.set_document_title_if_empty(doc_id, title))?;
            }
            let sections = extracted.sections;
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
                tables: Vec::new(),
            })
        }
        parser::FileType::Unknown => Err(Error::UnsupportedFileType(filename.to_owned())),
    }
}

/// One document per table: refuse when a live document other than
/// `owner` already loaded `table`.
fn check_table_free(db: &WorkspaceDb, table: &str, owner: Option<&str>) -> Result<()> {
    match db.table_owner(table)? {
        Some(doc) if owner != Some(doc.id.as_str()) => Err(Error::TableTaken {
            table: table.to_owned(),
            document: doc.id,
            filename: doc.filename,
        }),
        _ => Ok(()),
    }
}

/// The table piped data loads into for one invocation.
pub const STDIN_TABLE: &str = "stdin";

/// Load bytes piped into the CLI as the temporary table `stdin` on this
/// connection: JSON when they start with `{` or `[`, Parquet by its magic,
/// else CSV (delimiter sniffed). The bytes pass through `files/` because
/// the connection reads nothing outside the workspace directory; the file
/// is removed once the table is materialized. Empty input loads nothing.
///
/// # Errors
///
/// Returns an error if the bytes cannot be parsed or written.
pub fn load_stdin_table(
    config: &Config,
    db: &WorkspaceDb,
    workspace_id: &str,
    data: &[u8],
) -> Result<Option<String>> {
    if data.iter().all(u8::is_ascii_whitespace) {
        return Ok(None);
    }
    let first = data.iter().find(|b| !b.is_ascii_whitespace()).copied();
    let reader = if data.starts_with(b"PAR1") {
        "read_parquet"
    } else if matches!(first, Some(b'{' | b'[')) {
        "read_json_auto"
    } else {
        "read_csv_auto"
    };
    let files_dir = config.workspace_files_dir(workspace_id);
    std::fs::create_dir_all(&files_dir)?;
    let dest = files_dir.join(format!(".stdin-{}", uuid::Uuid::now_v7()));
    std::fs::write(&dest, data)?;
    let path = dest.to_string_lossy();
    let create_sql = format!(
        "CREATE OR REPLACE TEMP TABLE {} AS SELECT * FROM {reader}(?)",
        quote_ident(STDIN_TABLE)
    );
    let loaded = db.execute_with_params(&create_sql, duckdb::params![path.as_ref()]);
    if let Err(e) = std::fs::remove_file(&dest) {
        tracing::warn!(path = %dest.display(), error = %e, "could not remove the stdin scratch file");
    }
    loaded?;
    tracing::info!(
        table = STDIN_TABLE,
        reader,
        bytes = data.len(),
        "loaded piped data"
    );
    Ok(Some(STDIN_TABLE.to_owned()))
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
    doc_id: &str,
    filename: &str,
    data: &[u8],
    file_type: &parser::FileType,
) -> Result<String> {
    let table_name = sanitize_table_name(filename);
    check_table_free(db, &table_name, Some(doc_id))?;
    let files_dir = config.workspace_files_dir(workspace_id);
    std::fs::create_dir_all(&files_dir)?;
    let dest = files_dir.join(
        std::path::Path::new(filename)
            .file_name()
            .ok_or_else(|| Error::Ingestion(format!("'{filename}' is not a file name")))?,
    );
    std::fs::write(&dest, data)?;

    let path = dest.to_string_lossy();

    let reader = match file_type {
        parser::FileType::Csv => "read_csv_auto",
        parser::FileType::Parquet => "read_parquet",
        parser::FileType::Json => "read_json_auto",
        parser::FileType::Xlsx
        | parser::FileType::Pdf
        | parser::FileType::Text
        | parser::FileType::Markdown
        | parser::FileType::Html
        | parser::FileType::Docx
        | parser::FileType::Pptx
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

/// Every data sheet of a workbook as its own table: `<stem>` for a single
/// sheet, `<stem>_<sheet>` otherwise. The sheets pass through `files/` as
/// CSV for `DuckDB`'s reader (the `excel` extension is not in the static
/// binary).
fn ingest_workbook(
    config: &Config,
    db: &WorkspaceDb,
    workspace_id: &str,
    doc_id: &str,
    filename: &str,
    data: &[u8],
) -> Result<Vec<String>> {
    let sheets = xlsx::sheets(data)?;
    let files_dir = config.workspace_files_dir(workspace_id);
    std::fs::create_dir_all(&files_dir)?;
    let stem = sanitize_table_name(filename);
    let single = sheets.len() == 1;
    let names: Vec<String> = sheets
        .iter()
        .map(|sheet| {
            if single {
                stem.clone()
            } else {
                format!("{stem}_{}", sanitize_identifier(&sheet.sheet))
            }
        })
        .collect();
    for name in &names {
        check_table_free(db, name, Some(doc_id))?;
    }
    let mut tables = Vec::with_capacity(sheets.len());
    for (sheet, table_name) in sheets.into_iter().zip(names) {
        let dest = files_dir.join(format!("{table_name}.csv"));
        std::fs::write(&dest, &sheet.csv)?;
        let path = dest.to_string_lossy();
        let create_sql = format!(
            "CREATE OR REPLACE TABLE {} AS SELECT * FROM read_csv_auto(?, header = true)",
            quote_ident(&table_name)
        );
        db.execute_with_params(&create_sql, duckdb::params![path.as_ref()])?;
        tracing::info!(table = %table_name, sheet = %sheet.sheet, rows = sheet.rows, file = %filename, "created table from workbook sheet");
        tables.push(table_name);
    }
    Ok(tables)
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
    sanitize_identifier(stem)
}

fn sanitize_identifier(name: &str) -> String {
    name.chars()
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
