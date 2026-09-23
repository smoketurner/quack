pub mod chunker;
pub mod html;
pub mod office;
pub mod parser;
pub mod xlsx;

use std::time::{Duration, Instant};

use rig::embeddings::EmbeddingModel;
use tokio_util::sync::CancellationToken;

use crate::config::Config;
use crate::embedding::{Embedder, Input};
use crate::error::{Error, Result};
use crate::storage::control::sha256_hex;
use crate::storage::workspace::{
    DocumentInfo, DocumentSource, NewChunk, NewDocument, WorkspaceDb, quote_ident,
};
use crate::storage::writer::Writer;

/// Result of ingesting a single file into a workspace.
#[derive(Debug)]
pub struct IngestResult {
    pub document_id: String,
    pub filename: String,
    pub file_type: parser::FileType,
    pub chunks_stored: u32,
    /// Tables a structured file loaded into: one, or one per workbook sheet.
    pub tables: Vec<String>,
    /// Pages the parser could not read and skipped (PDF only).
    pub pages_skipped: u32,
    /// How long embedding the chunks took, when a model ran.
    pub embedding_time: Option<Duration>,
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
    /// Stops the ingest between steps and mid-embedding when cancelled.
    pub cancel: Option<&'a CancellationToken>,
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
            cancel: None,
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

    /// Stop when `cancel` is cancelled (a job's token).
    #[must_use]
    pub fn cancel(mut self, cancel: Option<&'a CancellationToken>) -> Self {
        self.cancel = cancel;
        self
    }
}

/// Run `work` unless `cancel` fires first, in which case it is dropped
/// (an embedding request in flight is abandoned) and the answer is
/// [`Error::Cancelled`].
///
/// # Errors
///
/// `work`'s error, or [`Error::Cancelled`].
pub async fn or_cancelled<T>(
    cancel: Option<&CancellationToken>,
    work: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    match cancel {
        None => work.await,
        Some(token) => tokio::select! {
            biased;
            () = token.cancelled() => Err(Error::Cancelled),
            result = work => result,
        },
    }
}

/// [`Error::Cancelled`] when `cancel` has fired, for the checks between
/// steps.
fn check_cancel(cancel: Option<&CancellationToken>) -> Result<()> {
    if cancel.is_some_and(CancellationToken::is_cancelled) {
        Err(Error::Cancelled)
    } else {
        Ok(())
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
pub async fn ingest_file<M: EmbeddingModel>(
    config: &Config,
    db: &Writer,
    workspace_id: &str,
    file: &NewFile<'_>,
    embedder: Option<&Embedder<M>>,
) -> Result<IngestOutcome> {
    let pending = Pending::of(file)?;
    let doc_id = match db.run(move |db| pending.register(db)).await? {
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
        embedder,
        file.cancel,
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
    Pending::of(file)?.register(db)
}

/// What registering a file writes, owned and without its bytes, so the
/// write can be handed to the workspace writer while the bytes stay here.
struct Pending {
    filename: String,
    title: Option<String>,
    ingested_by: Option<String>,
    source: DocumentSource,
    size_bytes: usize,
    sha256: String,
    file_type: parser::FileType,
}

impl Pending {
    /// Hash the bytes and refuse a type nothing can parse, before any write.
    fn of(file: &NewFile<'_>) -> Result<Self> {
        let file_type = parser::detect_file_type(file.filename);
        if matches!(file_type, parser::FileType::Unknown) {
            return Err(Error::UnsupportedFileType(file.filename.to_owned()));
        }
        Ok(Self {
            filename: file.filename.to_owned(),
            title: file
                .title
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(str::to_owned),
            ingested_by: file.ingested_by.map(str::to_owned),
            source: file.source,
            size_bytes: file.data.len(),
            sha256: sha256_hex(file.data),
            file_type,
        })
    }

    fn register(&self, db: &WorkspaceDb) -> Result<Registration> {
        let Self {
            filename,
            title,
            ingested_by,
            source,
            size_bytes,
            sha256,
            file_type,
        } = self;
        register_pending(
            db,
            filename,
            title.as_deref(),
            ingested_by.as_deref(),
            *source,
            *size_bytes,
            sha256,
            file_type,
        )
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the document row's fields, destructured from Pending"
)]
fn register_pending(
    db: &WorkspaceDb,
    filename: &str,
    title: Option<&str>,
    ingested_by: Option<&str>,
    source: DocumentSource,
    size_bytes: usize,
    sha256: &str,
    file_type: &parser::FileType,
) -> Result<Registration> {
    if let Some(existing) = db.document_by_sha256(sha256)? {
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
        check_table_free(db, &sanitize_table_name(filename), None)?;
    }
    let doc_id = uuid::Uuid::now_v7().to_string();
    db.insert_document(&NewDocument {
        id: &doc_id,
        filename,
        title,
        mime_type: file_type.mime_type(),
        size_bytes,
        sha256,
        source,
        status: "queued",
        ingested_by,
    })?;
    Ok(Registration::New(doc_id))
}

/// Parse, store, and embed a registered document, moving its status from
/// `processing` to `ready`, or to `error` with the message when it fails
/// (`cancelled` when `cancel` fired: the chunks stored so far are
/// discarded, as for any failure).
///
/// # Errors
///
/// Returns the failure after recording it on the document row.
#[expect(
    clippy::too_many_arguments,
    reason = "the document's identity, its bytes, the model, and the cancel token"
)]
pub async fn process_document<M: EmbeddingModel>(
    config: &Config,
    db: &Writer,
    workspace_id: &str,
    doc_id: &str,
    filename: &str,
    data: &[u8],
    embedder: Option<&Embedder<M>>,
    cancel: Option<&CancellationToken>,
) -> Result<IngestResult> {
    let id = doc_id.to_owned();
    db.run(move |db| db.update_document_status(&id, "processing"))
        .await?;
    let outcome = match check_cancel(cancel) {
        Ok(()) => {
            process_inner(
                config,
                db,
                workspace_id,
                doc_id,
                filename,
                data,
                embedder,
                cancel,
            )
            .await
        }
        Err(e) => Err(e),
    };
    let id = doc_id.to_owned();
    match &outcome {
        Ok(result) => {
            let (chunks, tables) = (result.chunks_stored, result.tables.clone());
            db.run(move |db| {
                db.set_document_chunk_count(&id, chunks)?;
                db.set_document_tables(&id, &tables)?;
                db.update_document_status(&id, "ready")
            })
            .await?;
        }
        Err(e) => {
            let message = e.to_string();
            db.run(move |db| {
                db.discard_chunks(&id)?;
                db.mark_document_error(&id, &message)
            })
            .await?;
        }
    }
    outcome
}

#[expect(
    clippy::too_many_arguments,
    reason = "process_document's arguments, passed through"
)]
async fn process_inner<M: EmbeddingModel>(
    config: &Config,
    db: &Writer,
    workspace_id: &str,
    doc_id: &str,
    filename: &str,
    data: &[u8],
    embedder: Option<&Embedder<M>>,
    cancel: Option<&CancellationToken>,
) -> Result<IngestResult> {
    let file_type = parser::detect_file_type(filename);
    match file_type {
        parser::FileType::Csv | parser::FileType::Parquet | parser::FileType::Json => {
            // One step on the writer: check the table is free, write the
            // bytes under `files/`, load them.
            let step = StructuredLoad {
                config: config.clone(),
                workspace_id: workspace_id.to_owned(),
                doc_id: doc_id.to_owned(),
                filename: filename.to_owned(),
                data: data.to_vec(),
                file_type: file_type.clone(),
            };
            let table_name = db.run(move |db| step.load(db)).await?;
            Ok(IngestResult {
                document_id: doc_id.to_owned(),
                filename: filename.to_owned(),
                file_type,
                chunks_stored: 0,
                tables: vec![table_name],
                pages_skipped: 0,
                embedding_time: None,
            })
        }
        parser::FileType::Xlsx => {
            // Parsing the workbook is the slow part: off the runtime's
            // workers, and not on the writer.
            let bytes = data.to_vec();
            let sheets = parse_off_runtime(move || xlsx::sheets(&bytes)).await?;
            let files_dir = config.workspace_files_dir(workspace_id);
            let (id, name) = (doc_id.to_owned(), filename.to_owned());
            let tables = db
                .run(move |db| ingest_workbook(db, &files_dir, &id, &name, sheets))
                .await?;
            Ok(IngestResult {
                document_id: doc_id.to_owned(),
                filename: filename.to_owned(),
                file_type,
                chunks_stored: 0,
                tables,
                pages_skipped: 0,
                embedding_time: None,
            })
        }
        parser::FileType::Pdf
        | parser::FileType::Text
        | parser::FileType::Markdown
        | parser::FileType::Html
        | parser::FileType::Docx
        | parser::FileType::Pptx => {
            // Parsing and chunking are the slow, CPU-bound part: off the
            // runtime's workers, and not on the writer.
            let parsing = Parsing::new(config, &file_type, filename, data);
            let Parsed {
                title,
                pages_skipped,
                chunks,
            } = parse_off_runtime(move || parsing.run()).await?;
            if let Some(title) = title {
                let id = doc_id.to_owned();
                db.run(move |db| db.set_document_title_if_empty(&id, &title))
                    .await?;
            }
            if pages_skipped > 0 {
                tracing::warn!(
                    document = %doc_id,
                    file = %filename,
                    pages_skipped,
                    "ingested with unreadable pages skipped"
                );
            }
            check_cancel(cancel)?;
            let (chunk_count, embedding_time) = embed_and_store(
                db,
                doc_id,
                &chunks,
                embedder,
                EmbedPlan {
                    batch_size: config.ingestion.embedding_batch_size,
                    concurrency: config.ingestion.embedding_concurrency,
                    cancel,
                },
            )
            .await?;
            Ok(IngestResult {
                document_id: doc_id.to_owned(),
                filename: filename.to_owned(),
                file_type,
                chunks_stored: chunk_count,
                tables: Vec::new(),
                pages_skipped,
                embedding_time,
            })
        }
        parser::FileType::Unknown => Err(Error::UnsupportedFileType(filename.to_owned())),
    }
}

/// A document's bytes on their way to be parsed and chunked.
struct Parsing {
    file_type: parser::FileType,
    stem: Option<String>,
    data: Vec<u8>,
    chunk_size: u32,
    chunk_overlap: u32,
    encoding: String,
}

/// What parsing a document found.
struct Parsed {
    title: Option<String>,
    pages_skipped: u32,
    chunks: Vec<chunker::Chunk>,
}

impl Parsing {
    fn new(config: &Config, file_type: &parser::FileType, filename: &str, data: &[u8]) -> Self {
        Self {
            file_type: file_type.clone(),
            stem: std::path::Path::new(filename)
                .file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_owned),
            data: data.to_vec(),
            chunk_size: config.ingestion.chunk_size_tokens,
            chunk_overlap: config.ingestion.chunk_overlap_tokens,
            encoding: config.ingestion.tokenizer_encoding.clone(),
        }
    }

    fn run(self) -> Result<Parsed> {
        let extracted = parser::extract(&self.file_type, &self.data)?;
        let chunks = chunker::chunk_document(
            &extracted,
            self.stem.as_deref(),
            self.chunk_size,
            self.chunk_overlap,
            &self.encoding,
        )?;
        Ok(Parsed {
            title: extracted.title().map(str::to_owned),
            pages_skipped: extracted.pages_skipped,
            chunks,
        })
    }
}

/// Run CPU-bound parsing on the blocking pool, so an async worker (a
/// terminal's input loop, a server's handlers) never stalls on a large
/// file.
async fn parse_off_runtime<T: Send + 'static>(
    parse: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    tokio::task::spawn_blocking(parse)
        .await
        .map_err(|e| Error::Ingestion(format!("parsing stopped before it finished: {e}")))?
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
) -> Result<Option<&'static str>> {
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
    Ok(Some(STDIN_TABLE))
}

/// The table a structured file loads into: its stem with anything outside
/// `[A-Za-z0-9_]` replaced by `_`.
#[must_use]
pub fn table_name_for(filename: &str) -> String {
    sanitize_table_name(filename)
}

/// A structured file's load, owned, for the workspace writer's thread.
struct StructuredLoad {
    config: Config,
    workspace_id: String,
    doc_id: String,
    filename: String,
    data: Vec<u8>,
    file_type: parser::FileType,
}

impl StructuredLoad {
    fn load(&self, db: &WorkspaceDb) -> Result<String> {
        ingest_structured(
            &self.config,
            db,
            &self.workspace_id,
            &self.doc_id,
            &self.filename,
            &self.data,
            &self.file_type,
        )
    }
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
    db: &WorkspaceDb,
    files_dir: &std::path::Path,
    doc_id: &str,
    filename: &str,
    sheets: Vec<xlsx::SheetCsv>,
) -> Result<Vec<String>> {
    std::fs::create_dir_all(files_dir)?;
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

/// Store `chunks`, then embed them `batch_size` at a time
/// (`[ingestion].embedding_batch_size`, at least one per request) with up
/// to `concurrency` requests in flight (`[ingestion].embedding_concurrency`),
/// each batch's vectors written as one transaction as soon as it returns.
/// Returns the chunk count and, when a model ran, how long embedding took.
/// How `embed_and_store` sends its batches.
struct EmbedPlan<'a> {
    batch_size: u32,
    concurrency: u32,
    /// Checked while each batch is in flight: a cancel drops the requests.
    cancel: Option<&'a CancellationToken>,
}

async fn embed_and_store<M: EmbeddingModel>(
    db: &Writer,
    document_id: &str,
    chunks: &[chunker::Chunk],
    embedder: Option<&Embedder<M>>,
    plan: EmbedPlan<'_>,
) -> Result<(u32, Option<Duration>)> {
    use futures::StreamExt as _;

    let EmbedPlan {
        batch_size,
        concurrency,
        cancel,
    } = plan;

    let (owned, id) = (chunks.to_vec(), document_id.to_owned());
    let chunk_ids = db
        .run(move |db| {
            db.write_transaction(|db| {
                let mut ids = Vec::with_capacity(owned.len());
                for (i, chunk) in owned.iter().enumerate() {
                    let chunk_id = uuid::Uuid::now_v7().to_string();
                    let idx = u32::try_from(i)
                        .map_err(|_| Error::Ingestion("chunk index overflow".into()))?;
                    db.insert_chunk(&NewChunk {
                        id: &chunk_id,
                        document_id: &id,
                        chunk_index: idx,
                        content: &chunk.content,
                        heading: chunk.heading.as_deref(),
                        page: chunk.page,
                        embedding: None,
                    })?;
                    ids.push(chunk_id);
                }
                Ok(ids)
            })
        })
        .await?;
    let stored = u32::try_from(chunk_ids.len())
        .map_err(|_| Error::Ingestion("chunk count overflow".into()))?;

    let Some(embedder) = embedder else {
        return Ok((stored, None));
    };
    if chunks.is_empty() {
        return Ok((stored, Some(Duration::ZERO)));
    }
    let dimension = embedder.profile().dimension;
    if db.run(move |db| Ok(db.embedding_dimension())).await? != dimension {
        // The configured width changed and the workspace still holds
        // vectors of the old one: the chunks are found by keyword until
        // `quack embeddings refresh` retypes the columns and embeds them.
        tracing::warn!(
            document_id = %document_id,
            "chunks stored without vectors: run `quack embeddings refresh` after the embedding width change"
        );
        return Ok((stored, None));
    }

    let started = Instant::now();
    let batch_size = usize::try_from(batch_size.max(1))
        .map_err(|_| Error::Ingestion("embedding_batch_size overflow".into()))?;
    let concurrency = usize::try_from(concurrency.max(1)).unwrap_or(1);
    // Batches are collected before the futures are built: a closure that
    // takes the slice by reference would tie each future's type to that
    // borrow and fail the `Send` check the server's handlers need.
    let batches_input: Vec<(Vec<String>, Vec<Input>)> = chunk_ids
        .chunks(batch_size)
        .zip(chunks.chunks(batch_size))
        .map(|(ids, slice)| {
            (
                ids.to_vec(),
                slice.iter().map(chunker::Chunk::embedding_input).collect(),
            )
        })
        .collect();
    let calls = batches_input.into_iter().map(|(ids, inputs)| async move {
        let vectors = embedder.embed(&inputs).await?;
        Ok::<_, Error>((ids, vectors))
    });
    let mut batches: u32 = 0;
    let mut stream = futures::stream::iter(calls).buffered(concurrency);
    while let Some(next) = or_cancelled(cancel, async { Ok(stream.next().await) }).await? {
        let (ids, vectors) = next?;
        batches = batches.saturating_add(1);
        db.run(move |db| {
            db.write_transaction(|db| {
                for (chunk_id, vector) in ids.iter().zip(&vectors) {
                    db.set_chunk_embedding(chunk_id, vector)?;
                }
                Ok(())
            })
        })
        .await?;
    }

    let elapsed = started.elapsed();
    let seconds = elapsed.as_secs_f64();
    let per_second = if seconds > 0.0 {
        f64::from(stored) / seconds
    } else {
        0.0
    };
    tracing::info!(
        document_id = %document_id,
        chunk_count = %stored,
        batches,
        concurrency,
        seconds = %format!("{seconds:.1}"),
        chunks_per_second = %format!("{per_second:.1}"),
        "embedded chunks"
    );
    Ok((stored, Some(elapsed)))
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
