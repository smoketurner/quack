#![expect(clippy::unwrap_used, reason = "test assertions use unwrap throughout")]

use std::collections::BTreeMap;
use std::path::{MAIN_SEPARATOR, Path};

use quack_core::analysis::citations::Citation;
use quack_core::config::{
    AnalysisConfig, BaseUrl, Config, ContextConfig, DecisionConfig, EmbeddingConfig, GeneralConfig,
    GraphConfig, ImportConfig, IngestionConfig, JobsConfig, OntologyConfig, ProviderConfig,
    ProviderType, RetrievalConfig, ServerConfig,
};
use quack_core::embedding::refresh::{Plan, Retype};
use quack_core::embedding::{Dimension, Embedder, EmbeddingModel, Profile, Prompts, Vector};
use quack_core::error::Error;
use quack_core::graph::{Properties, Standing, store as graph_store};
use quack_core::ids::{ChunkId, ClassId, DocumentId};
use quack_core::import::{HostReach, ImportPolicy, ImportRequest};
use quack_core::ingestion::parser::SectionKind;
use quack_core::ingestion::parser::{FileType, PageCounts};
use quack_core::ingestion::tree::{Folder, FolderReport, Outcome, Prune};
use quack_core::llm::CancellationToken;
use quack_core::llm::egress::Egress;
use quack_core::progress::{ChunkDone, RunControl};
use quack_core::storage::control::{ControlPlane, WorkspaceName};
use quack_core::storage::profile::TableProfile;
use quack_core::storage::workspace::{
    ChunkScope, DocumentFields, DocumentInfo, DocumentListing, DocumentSource, DocumentStatus,
    HybridLimits, MetaKey, NewChunk, NewDocument, PinnedText, Pinning, Shown, StatementKind,
    WorkspaceDb,
};
use quack_core::storage::writer::Writer;
use quack_core::text::Tokens;
use quack_core::{import, ingestion};
use rig::ProviderError;
use rig::embeddings::Embedding;

const TEST_DIM: usize = 4;
const TEST_DIM_U32: u32 = 4;

struct MockEmbeddingModel {
    dim: usize,
}

/// `model` under the profile `test_config` configures: `mock-model`,
/// `TEST_DIM` wide, no prefixes.
fn embedder<M: EmbeddingModel>(model: M) -> Embedder<M> {
    Embedder::new(
        model,
        Profile::new(
            "mock-model",
            Dimension::new(TEST_DIM_U32),
            Prompts::default(),
        ),
    )
}

/// `values` as a vector of their own width.
fn vector(values: &[f32]) -> Vector {
    let width = u32::try_from(values.len()).unwrap();
    Vector::new(values.to_vec(), Dimension::new(width)).unwrap()
}

/// Records the size of every batch it is asked to embed.
struct BatchRecordingModel {
    batches: std::sync::Mutex<Vec<usize>>,
}

impl EmbeddingModel for BatchRecordingModel {
    fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> impl Future<Output = Result<Vec<Embedding>, ProviderError>> + Send {
        if let Ok(mut batches) = self.batches.lock() {
            batches.push(texts.len());
        }
        std::future::ready(Ok(texts
            .into_iter()
            .map(|document| Embedding {
                document,
                vec: vec![0.1_f64; TEST_DIM],
            })
            .collect()))
    }
}

/// An embedding provider that is down: every call fails.
struct FailingEmbeddingModel;

impl EmbeddingModel for FailingEmbeddingModel {
    fn embed_texts(
        &self,
        _texts: Vec<String>,
    ) -> impl Future<Output = Result<Vec<Embedding>, ProviderError>> + Send {
        std::future::ready(Err(ProviderError::Provider(String::from(
            "connection refused",
        ))))
    }
}

impl EmbeddingModel for MockEmbeddingModel {
    fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> impl Future<Output = Result<Vec<Embedding>, ProviderError>> + Send {
        let mut result = Vec::new();
        for text in texts {
            result.push(Embedding {
                document: text,
                vec: vec![0.1_f64; self.dim],
            });
        }
        std::future::ready(Ok(result))
    }
}

/// A writer over a second connection to `db`'s database, for ingestion
/// and import, while the test's own statements and assertions keep `db`.
fn writer_of(db: &WorkspaceDb) -> Writer {
    Writer::spawn(db.try_clone_reader().unwrap()).unwrap()
}

fn test_config(data_dir: &Path) -> Config {
    let mut providers = BTreeMap::new();
    providers.insert(
        "mock".parse().unwrap(),
        ProviderConfig {
            base_url: Some(BaseUrl::try_from(String::from("http://localhost:9999")).unwrap()),
            ..ProviderConfig::new(ProviderType::Ollama)
        },
    );
    Config {
        general: GeneralConfig {
            data_dir: data_dir.to_path_buf(),
            default_workspace: WorkspaceName::default(),
            chat_model: None,
        },
        providers,
        ingestion: IngestionConfig {
            table_rows_as_table: 20,
            chunk_size_tokens: 50,
            chunk_overlap_tokens: 10,
            embedding_batch_size: 64,
            embedding_concurrency: 2,
            tokenizer_encoding: String::from("cl100k_base"),
            upload_max_mb: 512,
            max_decompressed_mb: 1024,
            vision_model: None,
        },
        embedding: EmbeddingConfig {
            model: Some("mock/mock-model".parse().unwrap()),
            dimension: Some(Dimension::new(TEST_DIM_U32)),
            ..EmbeddingConfig::default()
        },
        retrieval: RetrievalConfig::default(),
        context: ContextConfig::default(),
        analysis: AnalysisConfig::default(),
        server: ServerConfig::default(),
        ontology: OntologyConfig::default(),
        graph: GraphConfig::default(),
        decision: DecisionConfig::default(),
        import: ImportConfig::default(),
        jobs: JobsConfig::default(),
    }
}

fn test_config_no_provider(data_dir: &Path) -> Config {
    Config {
        general: GeneralConfig {
            data_dir: data_dir.to_path_buf(),
            default_workspace: WorkspaceName::default(),
            chat_model: None,
        },
        providers: BTreeMap::new(),
        ingestion: IngestionConfig::default(),
        embedding: EmbeddingConfig::default(),
        retrieval: RetrievalConfig::default(),
        context: ContextConfig::default(),
        analysis: AnalysisConfig::default(),
        server: ServerConfig::default(),
        ontology: OntologyConfig::default(),
        graph: GraphConfig::default(),
        decision: DecisionConfig::default(),
        import: ImportConfig::default(),
        jobs: JobsConfig::default(),
    }
}

// ---------------------------------------------------------------------------
// Ingestion pipeline tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ingest_text_without_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let workspace_id = "ws-text-no-embed";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();

    let writer = writer_of(&db);

    let data = b"Hello world. This is a test document for ingestion testing.";
    let result = ingestion::ingest_file::<MockEmbeddingModel>(
        &config,
        &writer,
        workspace_id,
        &ingestion::NewFile::new("test.txt", data),
        None,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();

    assert_eq!(result.filename, "test.txt");
    assert_eq!(result.file_type, FileType::Text);
    assert!(result.chunks_stored > 0);
    assert!(result.tables.is_empty());

    let qr = db
        .execute_query("SELECT COUNT(*) AS cnt FROM _quack_chunks")
        .unwrap();
    let count = qr.rows.first().unwrap().first().unwrap();
    assert_ne!(count, &serde_json::Value::Number(0.into()));
}

#[tokio::test]
async fn ingest_text_with_mock_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-text-embed";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();

    let writer = writer_of(&db);
    let model = embedder(MockEmbeddingModel { dim: TEST_DIM });

    let data = b"This is a longer document with enough words to produce at least one chunk. \
                 We need to make sure the embedding pipeline works end to end with our mock.";
    let result = ingestion::ingest_file(
        &config,
        &writer,
        workspace_id,
        &ingestion::NewFile::new("embed_test.txt", data),
        Some(&model),
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();

    assert_eq!(result.file_type, FileType::Text);
    assert!(result.chunks_stored > 0);

    let qr = db
        .execute_query("SELECT COUNT(*) AS cnt FROM _quack_chunks WHERE embedding IS NOT NULL")
        .unwrap();
    let count = qr.rows.first().unwrap().first().unwrap();
    assert_ne!(count, &serde_json::Value::Number(0.into()));
}

/// Counts how many embed requests are in flight at once and answers each
/// input with its own length, the first request slowest, so batches
/// finish out of order.
struct InFlightModel {
    in_flight: std::sync::atomic::AtomicUsize,
    peak: std::sync::atomic::AtomicUsize,
    calls: std::sync::atomic::AtomicUsize,
}

impl EmbeddingModel for InFlightModel {
    fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> impl Future<Output = Result<Vec<Embedding>, ProviderError>> + Send {
        use std::sync::atomic::Ordering;
        async move {
            let now = self
                .in_flight
                .fetch_add(1, Ordering::SeqCst)
                .saturating_add(1);
            self.peak.fetch_max(now, Ordering::SeqCst);
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let millis = if call == 0 { 40 } else { 2 };
            tokio::time::sleep(std::time::Duration::from_millis(millis)).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(texts
                .into_iter()
                .map(|document| {
                    let len = f64::from(u32::try_from(document.len()).unwrap_or(u32::MAX));
                    Embedding {
                        document,
                        vec: vec![len, 0.0, 0.0, 0.0],
                    }
                })
                .collect())
        }
    }
}

async fn ingest_six_sections(
    config: &Config,
    workspace_id: &str,
    model: &Embedder<InFlightModel>,
) -> (WorkspaceDb, ingestion::IngestResult) {
    let db = WorkspaceDb::open(config, workspace_id).unwrap();
    let writer = writer_of(&db);
    let sections: Vec<String> = (1..=6)
        .map(|i| format!("# Section {i}\n\nA short paragraph about topic number {i}.\n"))
        .collect();
    let data = sections.join("\n");
    let result = ingestion::ingest_file(
        config,
        &writer,
        workspace_id,
        &ingestion::NewFile::new("sections.md", data.as_bytes()),
        Some(model),
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    (db, result)
}

/// Chunks whose stored vector is not the one made from their own text.
fn mismatched_vectors(db: &WorkspaceDb) -> serde_json::Value {
    let qr = db
        .execute_query(
            "SELECT count(*) FROM _quack_chunks \
             WHERE embedding IS NULL \
                OR embedding[1] != length(heading || chr(10) || chr(10) || content)",
        )
        .unwrap();
    qr.rows.first().unwrap().first().unwrap().clone()
}

#[tokio::test]
async fn embedding_concurrency_overlaps_requests_and_keeps_vectors_with_their_chunks() {
    use std::sync::atomic::Ordering;
    let dir = tempfile::tempdir().unwrap();
    let mut config = test_config(dir.path());
    config.ingestion.embedding_batch_size = 1;
    config.ingestion.embedding_concurrency = 3;
    let model = embedder(InFlightModel {
        in_flight: std::sync::atomic::AtomicUsize::new(0),
        peak: std::sync::atomic::AtomicUsize::new(0),
        calls: std::sync::atomic::AtomicUsize::new(0),
    });

    let (db, result) = ingest_six_sections(&config, "ws-concurrent", &model).await;

    assert_eq!(result.chunks_stored, 6);
    assert!(result.embedding_time.is_some());
    assert_eq!(model.model().calls.load(Ordering::SeqCst), 6);
    let peak = model.model().peak.load(Ordering::SeqCst);
    assert!((2..=3).contains(&peak), "peak in-flight requests: {peak}");
    assert_eq!(mismatched_vectors(&db), serde_json::Value::Number(0.into()));
}

#[tokio::test]
async fn embedding_concurrency_of_one_stays_serial() {
    use std::sync::atomic::Ordering;
    let dir = tempfile::tempdir().unwrap();
    let mut config = test_config(dir.path());
    config.ingestion.embedding_batch_size = 1;
    config.ingestion.embedding_concurrency = 1;
    let model = embedder(InFlightModel {
        in_flight: std::sync::atomic::AtomicUsize::new(0),
        peak: std::sync::atomic::AtomicUsize::new(0),
        calls: std::sync::atomic::AtomicUsize::new(0),
    });

    let (db, _) = ingest_six_sections(&config, "ws-serial", &model).await;

    assert_eq!(model.model().peak.load(Ordering::SeqCst), 1);
    assert_eq!(mismatched_vectors(&db), serde_json::Value::Number(0.into()));
}

#[tokio::test]
async fn embedding_batch_size_bounds_every_embed_request() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = test_config(dir.path());
    config.ingestion.embedding_batch_size = 2;
    let workspace_id = "ws-batch";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();

    let writer = writer_of(&db);
    let model = embedder(BatchRecordingModel {
        batches: std::sync::Mutex::new(Vec::new()),
    });

    // Five headed sections, each its own chunk at 50 tokens.
    let sections: Vec<String> = (1..=5)
        .map(|i| format!("# Section {i}\n\nA short paragraph about topic number {i}.\n"))
        .collect();
    let data = sections.join("\n");
    let reported = std::sync::Mutex::new(Vec::new());
    let progress = |done: ChunkDone| {
        reported.lock().unwrap().push((done.done, done.total));
    };
    let control = RunControl {
        progress: &progress,
        cancel: None,
    };
    let result = ingestion::ingest_file(
        &config,
        &writer,
        workspace_id,
        &ingestion::NewFile::new("batches.md", data.as_bytes()).control(control),
        Some(&model),
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();

    // The total with nothing done, then one report per stored batch,
    // counting chunks, ending at all of them.
    let reported = reported.into_inner().unwrap();
    let batch_count = model.model().batches.lock().unwrap().len();
    assert_eq!(reported.len(), batch_count + 1, "{reported:?}");
    assert_eq!(reported.first().copied(), Some((0, result.chunks_stored)));
    assert!(reported.is_sorted_by(|a, b| a.0 < b.0), "{reported:?}");
    assert_eq!(
        reported.last().copied(),
        Some((result.chunks_stored, result.chunks_stored))
    );

    let batches = model.model().batches.lock().unwrap().clone();
    let total: usize = batches.iter().sum();
    assert_eq!(total, result.chunks_stored as usize);
    assert!(
        batches.len() >= 2,
        "expected several batches, got {batches:?}"
    );
    assert!(
        batches.iter().all(|&n| (1..=2).contains(&n)),
        "every batch within the configured size: {batches:?}"
    );
}

#[tokio::test]
async fn ingest_csv_structured() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-csv";

    let files_dir = config.workspace_files_dir(workspace_id);
    std::fs::create_dir_all(&files_dir).unwrap();
    let csv_content = b"name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago\n";
    std::fs::write(files_dir.join("people.csv"), csv_content).unwrap();

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();

    let writer = writer_of(&db);

    let result = ingestion::ingest_file::<MockEmbeddingModel>(
        &config,
        &writer,
        workspace_id,
        &ingestion::NewFile::new("people.csv", csv_content),
        None,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();

    assert_eq!(result.file_type, FileType::Csv);
    assert_eq!(result.chunks_stored, 0);
    assert_eq!(result.tables, ["people"]);

    let qr = db
        .execute_query("SELECT COUNT(*) AS cnt FROM people")
        .unwrap();
    let count = qr.rows.first().unwrap().first().unwrap();
    assert_eq!(count, &serde_json::Value::Number(3.into()));
}

/// A file named by its path loads without being read into memory: a table
/// copies into `files/`, a text document chunks, the same bytes in memory
/// are a duplicate of it, a file already in `files/` keeps its contents,
/// and an empty file is refused.
#[tokio::test]
async fn a_file_on_disk_ingests_from_its_path() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-replace").unwrap();
    let writer = writer_of(&db);
    let count = |table: &str| {
        db.execute_query(&format!("SELECT count(*) FROM {table}"))
            .unwrap()
            .rows
            .first()
            .and_then(|r| r.first())
            .cloned()
    };
    let csv = b"name,age\nAlice,30\nBob,25\n";
    let source = dir.path().join("people.csv");
    std::fs::write(&source, csv).unwrap();
    let loaded = ingest(
        &config,
        &writer,
        ingestion::NewFile::at_path("people.csv", &source),
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    assert_eq!(loaded.tables, ["people"]);
    assert_eq!(count("people"), Some(serde_json::json!(2)));
    let files = config.workspace_files_dir("ws-replace");
    assert_eq!(std::fs::read(files.join("people.csv")).unwrap(), csv);
    let again = ingest(&config, &writer, ingestion::NewFile::new("people.csv", csv))
        .await
        .unwrap();
    assert!(
        matches!(again, ingestion::IngestOutcome::Duplicate(_)),
        "{again:?}"
    );

    // A file ingested from `files/` itself is not truncated by the copy.
    let towns = b"town\nOslo\nLima\nPune\n";
    let inside = files.join("towns.csv");
    std::fs::write(&inside, towns).unwrap();
    ingest(
        &config,
        &writer,
        ingestion::NewFile::at_path("towns.csv", &inside),
    )
    .await
    .unwrap();
    assert_eq!(count("towns"), Some(serde_json::json!(3)));
    assert_eq!(std::fs::read(&inside).unwrap(), towns);

    let note = dir.path().join("note.md");
    std::fs::write(&note, b"# Flood\n\nFlood is excluded.\n").unwrap();
    let chunked = ingest(
        &config,
        &writer,
        ingestion::NewFile::at_path("note.md", &note),
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    assert!(chunked.chunks_stored > 0);

    let empty = dir.path().join("empty.csv");
    std::fs::write(&empty, b"").unwrap();
    let refused = ingest(
        &config,
        &writer,
        ingestion::NewFile::at_path("empty.csv", &empty),
    )
    .await;
    assert!(matches!(refused, Err(Error::EmptyFile(_))), "{refused:?}");
}

/// Delimited files load with their sniffed dialect, and a one-column sniff
/// is checked against the type's own separator: a genuine one-column file
/// loads, in either type.
#[tokio::test]
async fn delimited_files_load_by_their_dialect() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-dialects";
    std::fs::create_dir_all(config.workspace_files_dir(workspace_id)).unwrap();
    let db = WorkspaceDb::open(&config, workspace_id).unwrap();
    let writer = writer_of(&db);
    let ingest = |name: &'static str, bytes: &'static [u8]| {
        let (config, writer) = (&config, &writer);
        async move {
            ingestion::ingest_file::<MockEmbeddingModel>(
                config,
                writer,
                workspace_id,
                &ingestion::NewFile::new(name, bytes),
                None,
            )
            .await
            .map(|outcome| outcome.ingested().unwrap())
        }
    };
    let shape = |table: &str| {
        let columns = db
            .execute_query(&format!("SELECT * FROM \"{table}\" LIMIT 0"))
            .unwrap()
            .columns;
        let rows = db
            .execute_query(&format!("SELECT count(*) FROM \"{table}\""))
            .unwrap()
            .rows;
        (columns, rows.first().unwrap().first().unwrap().clone())
    };

    for (name, bytes, table, columns, rows) in [
        (
            "single.csv",
            &b"name\nalpha\nbeta\n"[..],
            "single",
            vec!["name"],
            2,
        ),
        (
            "semi.csv",
            &b"a;b\n1;2\n3;4\n"[..],
            "semi",
            vec!["a", "b"],
            2,
        ),
        (
            "tabbed.tsv",
            &b"a\tb\n1\t2\n"[..],
            "tabbed",
            vec!["a", "b"],
            1,
        ),
        ("lone.tsv", &b"only\nx\n"[..], "lone", vec!["only"], 1),
        (
            "header_only.csv",
            &b"a,b,c\n"[..],
            "header_only",
            vec!["a", "b", "c"],
            0,
        ),
    ] {
        let result = ingest(name, bytes).await.unwrap();
        assert_eq!(result.tables, [table], "{name}");
        assert_eq!(
            shape(table),
            (
                columns.iter().map(|c| (*c).to_owned()).collect(),
                serde_json::json!(rows)
            ),
            "{name}"
        );
    }
    assert_eq!(
        ingest("tabbed2.tsv", b"x\ty\n1\t2\n")
            .await
            .unwrap()
            .file_type,
        FileType::Tsv
    );
}

/// A file no separator splits consistently is refused instead of loading
/// its raw lines, and an empty file is refused before anything is written.
#[tokio::test]
async fn malformed_and_empty_delimited_files_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-dialects";
    std::fs::create_dir_all(config.workspace_files_dir(workspace_id)).unwrap();
    let db = WorkspaceDb::open(&config, workspace_id).unwrap();
    let writer = writer_of(&db);
    let ingest = |name: &'static str, bytes: &'static [u8]| {
        let (config, writer) = (&config, &writer);
        async move {
            ingestion::ingest_file::<MockEmbeddingModel>(
                config,
                writer,
                workspace_id,
                &ingestion::NewFile::new(name, bytes),
                None,
            )
            .await
            .map(|outcome| outcome.ingested().unwrap())
        }
    };

    let malformed = ingest("malformed.csv", b"a,b\n1,2,3,4\n\"unterminated,5\n6\n").await;
    assert!(
        matches!(&malformed, Err(Error::Ingestion(m)) if m.contains("'malformed.csv' does not parse as comma-separated values")),
        "{malformed:?}"
    );
    assert!(
        !db.list_tables()
            .unwrap()
            .contains(&String::from("malformed"))
    );

    let empty = ingest("empty.csv", b"").await;
    assert!(
        matches!(&empty, Err(Error::EmptyFile(name)) if name == "empty.csv"),
        "{empty:?}"
    );
    assert!(
        db.documents(&DocumentListing::default())
            .unwrap()
            .documents
            .iter()
            .all(|d| d.filename != "empty.csv")
    );
}

/// Ingest `filename`/`data` and return the persisted `DocumentInfo` plus
/// the returned error, so each case asserts both the returned variant and
/// the user-facing `error_message` column in one place.
async fn ingest_and_persisted(
    config: &Config,
    workspace_id: &str,
    filename: &str,
    data: &[u8],
) -> (Error, WorkspaceDb, Option<String>) {
    let db = WorkspaceDb::open(config, workspace_id).unwrap();
    let writer = writer_of(&db);
    let model = embedder(MockEmbeddingModel { dim: TEST_DIM });
    let result = ingestion::ingest_file(
        config,
        &writer,
        workspace_id,
        &ingestion::NewFile::new(filename, data),
        Some(&model),
    )
    .await;
    assert!(
        result.is_err(),
        "a malformed structured file should fail to ingest"
    );
    let err = result.unwrap_err();
    let doc = db
        .documents(&DocumentListing::default())
        .unwrap()
        .documents
        .into_iter()
        .find(|d| d.filename == filename)
        .unwrap();
    assert_eq!(doc.status, DocumentStatus::Error, "{filename}");
    (err, db, doc.error_message)
}

/// The user-facing message (the returned variant's `Display`, which is
/// what `Processing::run` persists) must name the file and its type but
/// must never carry the absolute on-disk path of the workspace `files/`
/// directory.
fn assert_scrubbed(
    config: &Config,
    workspace_id: &str,
    filename: &str,
    err: &Error,
    message: Option<&str>,
    type_label: &str,
) {
    // (1) the returned variant is the scrubbed `Ingestion` message, not a
    //     raw `Error::DuckDb` carrying the absolute path.
    assert!(
        matches!(err, Error::Ingestion(_)),
        "expected Error::Ingestion for {filename}, got {err:?}"
    );
    // `Processing::run` does `let message = e.to_string();` so the
    // persisted `DocumentInfo.error_message` is exactly the error's
    // `Display`, which the template renders verbatim.
    let msg = err.to_string();
    assert!(
        msg.contains(filename),
        "the message should name the file {filename}: {msg}"
    );
    assert!(
        msg.contains(type_label),
        "the message should name the type {type_label}: {msg}"
    );

    // (2) `DocumentInfo.error_message`, which `Processing::run` persists via
    //     `e.to_string()` and the template renders verbatim.
    let message = message.unwrap();
    assert_eq!(
        message,
        msg.as_str(),
        "the persisted message should match the returned one"
    );
    let leaked = config.workspace_files_dir(workspace_id).join(filename);
    assert!(
        !message.contains(leaked.to_str().unwrap()),
        "{filename}: the absolute path leaked into error_message: {message}"
    );
    // The workspace directory layout (`workspaces/<id>/files/`) and the
    // data_dir root must not appear either: only the `data_dir` is the
    // net-new disclosure a workspace is confined from reaching.
    let data_dir = config.general.data_dir.to_string_lossy();
    assert!(
        !message.contains(&*data_dir),
        "{filename}: the data_dir prefix leaked into error_message: {message}"
    );
    assert!(
        !message.contains(&format!("workspaces{MAIN_SEPARATOR}")),
        "{filename}: the workspaces layout leaked into error_message: {message}"
    );
}

/// `sniff_csv` rejects this before column-counting: a stray non-UTF-8 byte
/// in the body. The raw `Error::DuckDb` carries `file = <abs path>`; the
/// fix scrubs it into a path-free `Error::Ingestion`. This is the exact
/// repro from the bug report (arm (b), comma-separated).
#[tokio::test]
async fn sniff_csv_failure_on_non_utf8_csv_leaks_no_path() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-leak";
    std::fs::create_dir_all(config.workspace_files_dir(workspace_id)).unwrap();
    let (err, _db, message) =
        ingest_and_persisted(&config, workspace_id, "bad.csv", b"name\nalpha\xff\nbeta\n").await;
    assert_scrubbed(
        &config,
        workspace_id,
        "bad.csv",
        &err,
        message.as_deref(),
        "comma-separated",
    );
}

/// The same scrubbing covers the TSV path (arm (b), tab-separated): the
/// separator the file's type names appears in the message, the path does
/// not.
#[tokio::test]
async fn sniff_csv_failure_on_non_utf8_tsv_leaks_no_path() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-leak-tsv";
    std::fs::create_dir_all(config.workspace_files_dir(workspace_id)).unwrap();
    let (err, _db, message) =
        ingest_and_persisted(&config, workspace_id, "bad.tsv", b"a\tb\nalpha\xff\tbeta\n").await;
    assert_scrubbed(
        &config,
        workspace_id,
        "bad.tsv",
        &err,
        message.as_deref(),
        "tab-separated",
    );
}

/// A non-`Csv` reader that fails to parse — bytes that are not a Parquet
/// file — flows through arm (a) (`execute_with_params` on `read_parquet`).
/// Its raw `Error::DuckDb` likewise names the absolute path; the fix scrubs
/// it too.
#[tokio::test]
async fn read_parquet_failure_leaks_no_path() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-leak-parquet";
    std::fs::create_dir_all(config.workspace_files_dir(workspace_id)).unwrap();
    let (err, _db, message) = ingest_and_persisted(
        &config,
        workspace_id,
        "bad.parquet",
        b"this is not a parquet file at all",
    )
    .await;
    assert_scrubbed(
        &config,
        workspace_id,
        "bad.parquet",
        &err,
        message.as_deref(),
        "Parquet",
    );
}

/// A JSON file whose bytes do not parse (arm (a) on `read_json_auto`) is
/// scrubbed the same way.
#[tokio::test]
async fn read_json_failure_leaks_no_path() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-leak-json";
    std::fs::create_dir_all(config.workspace_files_dir(workspace_id)).unwrap();
    let (err, _db, message) =
        ingest_and_persisted(&config, workspace_id, "bad.json", b"{ not valid json").await;
    assert_scrubbed(
        &config,
        workspace_id,
        "bad.json",
        &err,
        message.as_deref(),
        "JSON",
    );
}

#[tokio::test]
async fn ingest_json_structured() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-json";

    let files_dir = config.workspace_files_dir(workspace_id);
    std::fs::create_dir_all(&files_dir).unwrap();
    let json_content = br#"[{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}]"#;
    std::fs::write(files_dir.join("scores.json"), json_content).unwrap();

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();

    let writer = writer_of(&db);

    let result = ingestion::ingest_file::<MockEmbeddingModel>(
        &config,
        &writer,
        workspace_id,
        &ingestion::NewFile::new("scores.json", json_content),
        None,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();

    assert_eq!(result.file_type, FileType::Json);
    assert_eq!(result.tables, ["scores"]);

    let qr = db
        .execute_query("SELECT COUNT(*) AS cnt FROM scores")
        .unwrap();
    let count = qr.rows.first().unwrap().first().unwrap();
    assert_eq!(count, &serde_json::Value::Number(2.into()));
}

#[tokio::test]
async fn ingest_unknown_file_type_returns_error() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-unknown";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();

    let writer = writer_of(&db);

    let result = ingestion::ingest_file::<MockEmbeddingModel>(
        &config,
        &writer,
        workspace_id,
        &ingestion::NewFile::new("scan.tiff", b"fake image data"),
        None,
    )
    .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, Error::UnsupportedFileType(_)),
        "expected UnsupportedFileType, got: {err}"
    );
}

/// An empty file of a chunked type is refused like an empty table file:
/// a document with nothing in it would only ever answer "no results".
#[tokio::test]
async fn ingest_empty_text_file_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-empty";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();

    let writer = writer_of(&db);

    let result = ingestion::ingest_file::<MockEmbeddingModel>(
        &config,
        &writer,
        workspace_id,
        &ingestion::NewFile::new("empty.txt", b""),
        None,
    )
    .await;

    assert!(
        matches!(&result, Err(Error::EmptyFile(name)) if name == "empty.txt"),
        "{result:?}"
    );
    assert!(
        db.documents(&DocumentListing::default())
            .unwrap()
            .documents
            .is_empty()
    );
}

#[tokio::test]
async fn ingest_markdown_as_unstructured() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-md";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();

    let writer = writer_of(&db);

    let data = b"# Heading\n\nSome paragraph text.\n\n- item 1\n- item 2\n";
    let result = ingestion::ingest_file::<MockEmbeddingModel>(
        &config,
        &writer,
        workspace_id,
        &ingestion::NewFile::new("notes.md", data),
        None,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();

    assert_eq!(result.file_type, FileType::Markdown);
    assert!(result.chunks_stored > 0);
}

// ---------------------------------------------------------------------------
// WorkspaceDb CRUD tests
// ---------------------------------------------------------------------------

#[test]
fn workspace_db_document_crud() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-doc-crud";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();

    db.insert_document(
        &NewDocument::new(&DocumentId::from("doc-1"), "test.txt", "text/plain", 100)
            .with_status(DocumentStatus::Queued),
    )
    .unwrap();

    let qr = db
        .execute_query("SELECT id, filename, status FROM _quack_documents WHERE id = 'doc-1'")
        .unwrap();
    assert_eq!(qr.rows.len(), 1);

    db.update_document_status(&DocumentId::from("doc-1"), DocumentStatus::Ready)
        .unwrap();

    let qr = db
        .execute_query("SELECT status FROM _quack_documents WHERE id = 'doc-1'")
        .unwrap();
    let status = qr.rows.first().unwrap().first().unwrap();
    assert_eq!(status, &serde_json::Value::String("ready".into()));
}

#[test]
fn workspace_db_chunk_without_embedding() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-chunk-no-emb";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();
    db.insert_document(
        &NewDocument::new(&DocumentId::from("doc-1"), "test.txt", "text/plain", 100)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();

    db.chunk_writer(&DocumentId::from("doc-1"), "hello world")
        .and_then(|writer| {
            writer.insert(&NewChunk {
                id: &ChunkId::from("c1"),
                chunk_index: 0,
                content: "hello world",
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: None,
            })
        })
        .unwrap();

    let qr = db
        .execute_query("SELECT id, content FROM _quack_chunks WHERE id = 'c1'")
        .unwrap();
    assert_eq!(qr.rows.len(), 1);
    let content = qr.rows.first().unwrap().get(1).unwrap();
    assert_eq!(content, &serde_json::Value::String("hello world".into()));
}

#[test]
fn workspace_db_chunk_with_embedding() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-chunk-emb";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();
    db.insert_document(
        &NewDocument::new(&DocumentId::from("doc-1"), "test.txt", "text/plain", 100)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();

    let embedding = Vector::from(vec![0.5_f32, 0.3, -0.2, 0.8]);
    db.chunk_writer(&DocumentId::from("doc-1"), "embedded chunk")
        .and_then(|writer| {
            writer.insert(&NewChunk {
                id: &ChunkId::from("c1"),
                chunk_index: 0,
                content: "embedded chunk",
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: Some(&embedding),
            })
        })
        .unwrap();

    let qr = db
        .execute_query("SELECT content FROM _quack_chunks WHERE embedding IS NOT NULL")
        .unwrap();
    assert_eq!(qr.rows.len(), 1);
}

#[test]
fn workspace_db_set_chunk_embedding() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-update-emb";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();
    db.insert_document(
        &NewDocument::new(&DocumentId::from("doc-1"), "test.txt", "text/plain", 100)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();

    db.chunk_writer(&DocumentId::from("doc-1"), "hello world")
        .and_then(|writer| {
            writer.insert(&NewChunk {
                id: &ChunkId::from("c1"),
                chunk_index: 0,
                content: "hello world",
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: None,
            })
        })
        .unwrap();

    // Embedding should be NULL initially
    let qr = db
        .execute_query(
            "SELECT COUNT(*) AS cnt FROM _quack_chunks WHERE id = 'c1' AND embedding IS NULL",
        )
        .unwrap();
    let count = qr.rows.first().unwrap().first().unwrap();
    assert_eq!(count, &serde_json::Value::Number(1.into()));

    // Update with an embedding
    db.set_chunk_embedding(&ChunkId::from("c1"), &vector(&[1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    // Embedding should now be non-NULL
    let qr = db
        .execute_query(
            "SELECT COUNT(*) AS cnt FROM _quack_chunks WHERE id = 'c1' AND embedding IS NOT NULL",
        )
        .unwrap();
    let count = qr.rows.first().unwrap().first().unwrap();
    assert_eq!(count, &serde_json::Value::Number(1.into()));
}

#[test]
fn workspace_db_execute_statement() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-stmt";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();

    db.execute_statement("CREATE TABLE test_table (id INTEGER, value TEXT)")
        .unwrap();
    db.execute_statement("INSERT INTO test_table VALUES (1, 'hello')")
        .unwrap();

    let qr = db.execute_query("SELECT * FROM test_table").unwrap();
    assert_eq!(qr.columns.len(), 2);
    assert_eq!(qr.rows.len(), 1);
}

#[test]
fn workspace_db_search_returns_filename_and_honors_document_filter() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let db = WorkspaceDb::open(&config, "ws-search-filter").unwrap();

    db.insert_document(
        &NewDocument::new(
            &DocumentId::from("doc-a"),
            "policy.pdf",
            "application/pdf",
            10,
        )
        .with_status(DocumentStatus::Ready),
    )
    .unwrap();
    db.insert_document(
        &NewDocument::new(&DocumentId::from("doc-b"), "faq.md", "text/markdown", 10)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();
    db.chunk_writer(&DocumentId::from("doc-a"), "flood exclusion")
        .and_then(|writer| {
            writer.insert(&NewChunk {
                id: &ChunkId::from("a0"),
                chunk_index: 0,
                content: "flood exclusion",
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: Some(&Vector::from(vec![1.0, 0.0, 0.0, 0.0])),
            })
        })
        .unwrap();
    db.chunk_writer(&DocumentId::from("doc-b"), "claims timeline")
        .and_then(|writer| {
            writer.insert(&NewChunk {
                id: &ChunkId::from("b0"),
                chunk_index: 0,
                content: "claims timeline",
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: Some(&Vector::from(vec![0.9, 0.1, 0.0, 0.0])),
            })
        })
        .unwrap();

    let query = Vector::from(vec![1.0_f32, 0.0, 0.0, 0.0]);

    let all = db
        .search_similar_chunks(&query, 5, &ChunkScope::all())
        .unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all.first().unwrap().filename, "policy.pdf");
    assert_eq!(all.last().unwrap().filename, "faq.md");

    let only_b = db
        .search_similar_chunks(
            &query,
            5,
            &ChunkScope::documents([DocumentId::from("doc-b")]),
        )
        .unwrap();
    assert_eq!(only_b.len(), 1);
    assert_eq!(
        only_b.first().unwrap().document_id,
        DocumentId::from("doc-b")
    );
    assert_eq!(only_b.first().unwrap().filename, "faq.md");

    let none = db
        .search_similar_chunks(
            &query,
            5,
            &ChunkScope::documents([DocumentId::from("missing")]),
        )
        .unwrap();
    assert!(none.is_empty());
}

#[test]
fn workspace_db_search_similar_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-search";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();
    db.insert_document(
        &NewDocument::new(&DocumentId::from("doc-1"), "test.txt", "text/plain", 100)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();

    db.chunk_writer(&DocumentId::from("doc-1"), "first chunk")
        .and_then(|writer| {
            writer.insert(&NewChunk {
                id: &ChunkId::from("c1"),
                chunk_index: 0,
                content: "first chunk",
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: Some(&Vector::from(vec![1.0, 0.0, 0.0, 0.0])),
            })
        })
        .unwrap();
    db.chunk_writer(&DocumentId::from("doc-1"), "second chunk")
        .and_then(|writer| {
            writer.insert(&NewChunk {
                id: &ChunkId::from("c2"),
                chunk_index: 1,
                content: "second chunk",
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: Some(&Vector::from(vec![0.0, 1.0, 0.0, 0.0])),
            })
        })
        .unwrap();
    db.chunk_writer(&DocumentId::from("doc-1"), "third chunk")
        .and_then(|writer| {
            writer.insert(&NewChunk {
                id: &ChunkId::from("c3"),
                chunk_index: 2,
                content: "third chunk",
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: Some(&Vector::from(vec![0.7, 0.7, 0.0, 0.0])),
            })
        })
        .unwrap();

    let query = Vector::from(vec![1.0_f32, 0.0, 0.0, 0.0]);
    let results = db
        .search_similar_chunks(&query, 3, &ChunkScope::all())
        .unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results.first().unwrap().content, "first chunk");
    assert_eq!(results.last().unwrap().content, "second chunk");
}

// ---------------------------------------------------------------------------
// Statement classification, internal-table refusal, limits, timeout (#7)
// ---------------------------------------------------------------------------

fn kind(db: &WorkspaceDb, sql: &str) -> StatementKind {
    db.classify_statement(sql).unwrap()
}

#[test]
fn classify_select_shapes_as_read() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(TEST_DIM_U32)).unwrap();
    db.execute_statement("CREATE TABLE t(a INT, b INT)")
        .unwrap();
    for sql in [
        "SELECT 1",
        "select a, sum(b) from t group by 1",
        "WITH x AS (SELECT a FROM t) SELECT * FROM x",
        "FROM t SELECT a",
        "SELECT * FROM t QUALIFY row_number() OVER () = 1",
        "DESCRIBE t",
        "SHOW TABLES",
        "SUMMARIZE t",
        "PIVOT t ON a USING sum(b)",
        "EXPLAIN SELECT 1",
        "  describe t ;  ",
    ] {
        assert_eq!(kind(&db, sql), StatementKind::Read, "{sql}");
    }
}

#[test]
fn classify_mutations_and_escapes_as_write() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(TEST_DIM_U32)).unwrap();
    for sql in [
        "CREATE TABLE t(a INT)",
        "DROP TABLE t",
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a = 2",
        "DELETE FROM t",
        "ALTER TABLE t ADD COLUMN b INT",
        "COPY t TO '/tmp/x.csv'",
        "ATTACH 'other.duckdb' AS other",
        "SET memory_limit = '10GB'",
        "INSTALL httpfs",
        "LOAD httpfs",
        "PRAGMA enable_progress_bar",
        "CALL pragma_version()",
        "SELECT 1; DROP TABLE t",
        "DESCRIBE t; DROP TABLE t",
        "CREATE TABLE u AS SELECT 1",
        "EXPLAIN ANALYZE INSERT INTO t VALUES (1)",
    ] {
        assert_eq!(kind(&db, sql), StatementKind::Write, "{sql}");
    }
}

/// `EXPLAIN ANALYZE` executes the statement it explains, unlike a plain
/// `EXPLAIN`, which only prints a plan; before this fix it classified as
/// `Read` and ran unguarded, letting `EXPLAIN ANALYZE INSERT ...` mutate
/// under `WritePolicy::Deny`, and, on `POST /sql`, under a viewer role that
/// never checks `Need::WRITE` for a `Read`-classified statement.
#[test]
fn explain_analyze_is_write_and_does_not_mutate_through_read_only() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(TEST_DIM_U32)).unwrap();
    db.execute_statement("CREATE TABLE t(a INT)").unwrap();
    db.execute_statement("INSERT INTO t VALUES (1)").unwrap();

    assert_eq!(
        kind(&db, "EXPLAIN ANALYZE INSERT INTO t VALUES (2)"),
        StatementKind::Write
    );

    // The same statement, run the way a `Read`-classified one would run
    // inside `read_only`: it must fail, not mutate.
    let err = db
        .read_only(|db| db.execute_statement("EXPLAIN ANALYZE INSERT INTO t VALUES (2)"))
        .err();
    assert!(err.is_some(), "EXPLAIN ANALYZE mutated inside read_only");
    let rows = db.execute_query("SELECT count(*) FROM t").unwrap();
    assert_eq!(
        rows.rows.first().and_then(|r| r.first()),
        Some(&serde_json::Value::Number(1.into()))
    );
}

#[test]
fn classify_syntax_errors_as_invalid() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(TEST_DIM_U32)).unwrap();
    assert!(matches!(kind(&db, "SELEC 1"), StatementKind::Invalid(_)));
    assert!(matches!(kind(&db, ""), StatementKind::Invalid(_)));
}

#[test]
fn internal_tables_are_detected_in_parsed_and_unparsed_statements() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(TEST_DIM_U32)).unwrap();
    db.execute_statement("CREATE TABLE sales(a INT)").unwrap();
    assert!(
        db.references_internal_table("SELECT * FROM _quack_chunks")
            .unwrap()
    );
    assert!(
        db.references_internal_table("SELECT content FROM main.\"_Quack_Chunks\" c")
            .unwrap()
    );
    assert!(
        db.references_internal_table("SELECT * FROM sales JOIN _quack_documents d ON true")
            .unwrap()
    );
    assert!(
        db.references_internal_table("DESCRIBE _quack_documents")
            .unwrap()
    );
    assert!(
        db.references_internal_table("DROP TABLE _quack_chunks")
            .unwrap()
    );
    assert!(!db.references_internal_table("SELECT * FROM sales").unwrap());
    assert!(
        !db.references_internal_table("SELECT '_quack_documents' AS label")
            .unwrap()
    );
    assert!(
        !db.references_internal_table("SELECT * FROM documents")
            .unwrap()
    );
    assert!(
        !db.list_tables()
            .unwrap()
            .iter()
            .any(|t| t.starts_with("_quack_"))
    );
}

#[test]
fn long_running_statement_is_interrupted_at_timeout() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(TEST_DIM_U32))
        .unwrap()
        .with_query_timeout(std::time::Duration::from_millis(200));
    let started = std::time::Instant::now();
    let result = db.execute_query(
        "SELECT count(*) FROM range(100000000) a, range(100000000) b WHERE a.range = b.range + 1",
    );
    let elapsed = started.elapsed();
    let err = result.err().unwrap();
    assert!(
        matches!(err, Error::QueryTimeout { timeout } if timeout == std::time::Duration::from_millis(200)),
        "unexpected error: {err}"
    );
    assert!(
        err.to_string().contains("200ms query timeout"),
        "unexpected message: {err}"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "took {elapsed:?}"
    );

    // The connection is still usable afterwards.
    let ok = db.execute_query("SELECT 1 AS one").unwrap();
    assert_eq!(ok.rows.len(), 1);
}

#[test]
fn open_applies_memory_and_thread_limits() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = test_config(dir.path());
    config.analysis.memory_limit_mb = 123;
    config.analysis.threads = 2;
    let db = WorkspaceDb::open(&config, "ws-limits").unwrap();
    let rows = db
        .execute_query("SELECT current_setting('memory_limit'), current_setting('threads')")
        .unwrap();
    let row = rows.rows.first().unwrap();
    let mem = row.first().unwrap().to_string();
    assert!(
        mem.contains("123") || mem.contains("117"),
        "memory_limit was {mem}"
    );
    assert_eq!(row.last().unwrap().to_string().trim_matches('"'), "2");
}

// ---------------------------------------------------------------------------
// Workspace meta, dimension reconciliation, legacy rename (#9)
// ---------------------------------------------------------------------------

#[test]
fn open_records_schema_version_and_embedding_meta() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let db = WorkspaceDb::open(&config, "ws-meta").unwrap();
    assert_eq!(
        db.meta(MetaKey::SchemaVersion).unwrap().as_deref(),
        Some("15")
    );
    assert_eq!(
        db.meta(MetaKey::EmbeddingDimension).unwrap().as_deref(),
        Some("4")
    );
    // The profile table replaces the model key.
    assert_eq!(db.meta(MetaKey::EmbeddingModel).unwrap(), None);
    let profile = Profile::new(
        "mock-model",
        Dimension::new(TEST_DIM_U32),
        Prompts::default(),
    );
    assert_eq!(db.embedding_profile(), Some(&profile));
    let recorded = db
        .execute_query("SELECT fingerprint FROM _quack_embedding_profiles")
        .unwrap();
    assert_eq!(
        recorded.rows.first().and_then(|r| r.first()),
        Some(&serde_json::Value::String(
            profile.fingerprint().as_str().to_owned()
        ))
    );
    assert_eq!(db.embedding_dimension(), Dimension::new(TEST_DIM_U32));
    assert!(db.list_tables().unwrap().is_empty());
}

#[test]
fn reopen_without_provider_keeps_recorded_dimension() {
    let dir = tempfile::tempdir().unwrap();
    let with = test_config(dir.path());
    drop(WorkspaceDb::open(&with, "ws-dim").unwrap());

    let without = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&without, "ws-dim").unwrap();
    assert_eq!(db.embedding_dimension(), Dimension::new(TEST_DIM_U32));
}

#[test]
fn dimension_change_with_stored_embeddings_keeps_them_until_refresh() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    {
        let db = WorkspaceDb::open(&config, "ws-mismatch").unwrap();
        db.insert_document(
            &NewDocument::new(&DocumentId::from("d"), "a.txt", "text/plain", 1)
                .with_status(DocumentStatus::Ready),
        )
        .unwrap();
        db.chunk_writer(&DocumentId::from("d"), "x")
            .and_then(|writer| {
                writer.insert(&NewChunk {
                    id: &ChunkId::from("c"),
                    chunk_index: 0,
                    content: "x",
                    heading: None,
                    page: None,
                    kind: SectionKind::Body,
                    locator: None,
                    embedding: Some(&Vector::from(vec![1.0, 0.0, 0.0, 0.0])),
                })
            })
            .unwrap();
    }
    let mut changed = test_config(dir.path());
    changed.embedding.dimension = Some(Dimension::new(8));
    changed.embedding.model = Some("mock/other-model".parse().unwrap());
    // Opening still works: the old vectors stay, at their width, unsearched.
    let db = WorkspaceDb::open(&changed, "ws-mismatch").unwrap();
    assert_eq!(db.embedding_dimension(), Dimension::new(4));
    assert!(!db.embedding_dimension().fits(8));
    assert!(
        db.search_similar_chunks(&Vector::from(vec![0.5; 8]), 5, &ChunkScope::all())
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        db.search_keyword_chunks("x", 5, &ChunkScope::all())
            .unwrap()
            .len(),
        1
    );
    let err = db
        .set_chunk_embedding(&ChunkId::from("c"), &vector(&[0.5; 8]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("quack embeddings refresh"), "{err}");
    let status = db.embedding_status().unwrap();
    assert_eq!(status.column_dimension, Dimension::new(4));
    assert_eq!(status.stale_chunks(), 1);
    let made_with = status.stale.first().and_then(|s| s.profile.clone());
    assert_eq!(
        made_with,
        Some(Profile::new(
            "mock-model",
            Dimension::new(4),
            Prompts::default()
        ))
    );
    let note = status.note().unwrap();
    assert!(
        note.contains("1 chunks were embedded with mock-model (4 dimensions, no prefixes)")
            && note.contains("other-model (8 dimensions, no prefixes)"),
        "{note}"
    );
    let plan = Plan::from_status(&status);
    assert_eq!(
        plan.retype,
        Some(Retype {
            stored: Dimension::new(4),
            configured: Dimension::new(8)
        })
    );
    assert_eq!(plan.chunks, 1);
}

#[test]
fn dimension_change_without_embeddings_adopts_new_width() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    {
        let db = WorkspaceDb::open(&config, "ws-adopt").unwrap();
        db.insert_document(
            &NewDocument::new(&DocumentId::from("d"), "a.txt", "text/plain", 1)
                .with_status(DocumentStatus::Ready),
        )
        .unwrap();
        db.chunk_writer(&DocumentId::from("d"), "x")
            .and_then(|writer| {
                writer.insert(&NewChunk {
                    id: &ChunkId::from("c"),
                    chunk_index: 0,
                    content: "x",
                    heading: None,
                    page: None,
                    kind: SectionKind::Body,
                    locator: None,
                    embedding: None,
                })
            })
            .unwrap();
    }
    {
        // A node label embedding of the old width: cleared on reopen and
        // recomputed by the next resolution pass.
        let db = WorkspaceDb::open(&config, "ws-adopt").unwrap();
        let node = graph_store::upsert_node(
            &db,
            &graph_store::NewNode {
                label: String::from("Kenya"),
                class_id: ClassId::from("country"),
                properties: Properties::default(),
                standing: Standing::Reviewed,
            },
        )
        .unwrap();
        db.set_node_embedding(&node, &vector(&[1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        assert!(
            graph_store::nodes_needing_embedding(&db, 10)
                .unwrap()
                .is_empty()
        );
    }
    let mut changed = test_config(dir.path());
    changed.embedding.dimension = Some(Dimension::new(8));
    let db = WorkspaceDb::open(&changed, "ws-adopt").unwrap();
    assert_eq!(db.embedding_dimension(), Dimension::new(8));
    let unembedded = graph_store::nodes_needing_embedding(&db, 10).unwrap();
    assert_eq!(unembedded.len(), 1);
    let node_id = unembedded.first().map(|n| n.id.clone()).unwrap();
    db.set_node_embedding(&node_id, &vector(&[0.5; 8])).unwrap();
    assert_eq!(
        db.meta(MetaKey::EmbeddingDimension).unwrap().as_deref(),
        Some("8")
    );
    // The chunk ingested without an embedding survives, term index and
    // all, and the new width takes embeddings.
    let kept = db.chunks_by_ids(&[ChunkId::from("c")]).unwrap();
    assert_eq!(kept.len(), 1);
    assert!(
        !db.search_keyword_chunks("x", 5, &ChunkScope::all())
            .unwrap()
            .is_empty()
    );
    db.chunk_writer(&DocumentId::from("d"), "y")
        .and_then(|writer| {
            writer.insert(&NewChunk {
                id: &ChunkId::from("c8"),
                chunk_index: 1,
                content: "y",
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: Some(&Vector::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            })
        })
        .unwrap();
    db.chunk_writer(&DocumentId::from("d"), "y")
        .and_then(|writer| {
            writer.insert(&NewChunk {
                id: &ChunkId::from("c2"),
                chunk_index: 1,
                content: "y",
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: Some(&Vector::from(vec![0.5; 8])),
            })
        })
        .unwrap();
}

#[test]
fn legacy_unprefixed_tables_are_renamed_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let path = config.workspace_db_path("ws-legacy");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    {
        let conn = duckdb::Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE documents (id TEXT PRIMARY KEY, filename TEXT NOT NULL, mime_type TEXT, \
             size_bytes BIGINT, ingested_at TIMESTAMP DEFAULT now(), status TEXT DEFAULT 'pending', \
             error_message TEXT);
             CREATE TABLE chunks (id TEXT PRIMARY KEY, document_id TEXT NOT NULL, chunk_index INTEGER \
             NOT NULL, content TEXT NOT NULL, embedding FLOAT[4], token_count INTEGER);
             INSERT INTO documents (id, filename) VALUES ('old', 'old.txt');",
        )
        .unwrap();
    }
    let db = WorkspaceDb::open(&config, "ws-legacy").unwrap();
    let docs = db.documents(&DocumentListing::default()).unwrap().documents;
    assert_eq!(docs.len(), 1);
    let old = docs.first().unwrap();
    assert_eq!(old.filename, "old.txt");
    // The old default status, `pending`, was never processed.
    assert_eq!(old.status, DocumentStatus::Error);
    assert!(
        old.error_message
            .as_deref()
            .is_some_and(|m| m.contains("upload it again"))
    );
    // A row from before page counts were recorded carries none.
    assert_eq!(old.pages, None);
    assert!(db.list_tables().unwrap().is_empty());
}

#[tokio::test]
async fn control_db_migrates_to_the_latest_version_and_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let control = ControlPlane::open(&config).await.unwrap();
    let latest = ControlPlane::latest_schema_version();
    // The newest file in migrations/ is the one the binary carries.
    assert!(latest >= 5, "{latest}");
    assert_eq!(control.schema_version().await.unwrap(), latest);
    // Reopening is a no-op.
    let again = ControlPlane::open(&config).await.unwrap();
    assert_eq!(again.schema_version().await.unwrap(), latest);
}

// ---------------------------------------------------------------------------
// Bound parameters and quoted identifiers (#10)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ingest_csv_with_quote_in_filename() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let workspace_id = "ws-quote";
    let db = WorkspaceDb::open(&config, workspace_id).unwrap();
    let writer = writer_of(&db);

    let filename = "it's a file.csv";
    let files_dir = config.workspace_files_dir(workspace_id);
    std::fs::create_dir_all(&files_dir).unwrap();
    std::fs::write(files_dir.join(filename), "a,b\n1,2\n3,4\n").unwrap();

    let result = ingestion::ingest_file(
        &config,
        &writer,
        workspace_id,
        &ingestion::NewFile::new(filename, b"a,b\n1,2\n3,4\n"),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    assert_eq!(result.tables, ["it_s_a_file"]);
    let rows = db
        .execute_query("SELECT sum(a) AS s FROM it_s_a_file")
        .unwrap();
    assert_eq!(rows.rows.first().unwrap().first().unwrap().to_string(), "4");
}

#[test]
fn describe_table_handles_quoted_identifier() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(TEST_DIM_U32)).unwrap();
    db.execute_statement("CREATE TABLE \"odd \"\"name\"\"\" (x INT)")
        .unwrap();
    db.execute_statement("INSERT INTO \"odd \"\"name\"\"\" VALUES (7)")
        .unwrap();
    let desc = db.describe_table("odd \"name\"").unwrap();
    assert_eq!(desc.columns.first().unwrap().name, "x");
    assert_eq!(desc.sample_rows.rows.len(), 1);
    let err = db.describe_table("nope\"; DROP TABLE x; --").err().unwrap();
    assert!(
        err.to_string().contains("does not exist") || err.to_string().contains("Catalog"),
        "{err}"
    );
}

// ---------------------------------------------------------------------------
// Hybrid retrieval, chunk metadata, pinning (#15)
// ---------------------------------------------------------------------------

fn seeded_for_search(config: &Config, ws: &str) -> WorkspaceDb {
    let db = WorkspaceDb::open(config, ws).unwrap();
    db.insert_document(
        &NewDocument::new(
            &DocumentId::from("doc-a"),
            "policy.pdf",
            "application/pdf",
            10,
        )
        .with_status(DocumentStatus::Ready),
    )
    .unwrap();
    db.insert_document(
        &NewDocument::new(&DocumentId::from("doc-b"), "faq.md", "text/markdown", 10)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();
    db.chunk_writer(
        &DocumentId::from("doc-a"),
        "Flood damage is excluded from coverage.",
    )
    .and_then(|writer| {
        writer.insert(&NewChunk {
            id: &ChunkId::from("a0"),
            chunk_index: 0,
            content: "Flood damage is excluded from coverage.",
            heading: Some("Exclusions"),
            page: Some(12),
            kind: SectionKind::Body,
            locator: None,
            embedding: Some(&Vector::from(vec![1.0, 0.0, 0.0, 0.0])),
        })
    })
    .unwrap();
    db.chunk_writer(
        &DocumentId::from("doc-b"),
        "Policy POL-8841 renews every March.",
    )
    .and_then(|writer| {
        writer.insert(&NewChunk {
            id: &ChunkId::from("b0"),
            chunk_index: 0,
            content: "Policy POL-8841 renews every March.",
            heading: None,
            page: None,
            kind: SectionKind::Body,
            locator: None,
            embedding: Some(&Vector::from(vec![0.0, 1.0, 0.0, 0.0])),
        })
    })
    .unwrap();
    db.chunk_writer(
        &DocumentId::from("doc-b"),
        "Claims close within thirty days of filing.",
    )
    .and_then(|writer| {
        writer.insert(&NewChunk {
            id: &ChunkId::from("b1"),
            chunk_index: 1,
            content: "Claims close within thirty days of filing.",
            heading: Some("Claims"),
            page: None,
            kind: SectionKind::Body,
            locator: None,
            embedding: Some(&Vector::from(vec![0.0, 0.0, 1.0, 0.0])),
        })
    })
    .unwrap();
    db
}

#[test]
fn chunk_metadata_round_trips_through_search() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let db = seeded_for_search(&config, "ws-meta-search");
    let hits = db
        .search_similar_chunks(
            &Vector::from(vec![1.0, 0.0, 0.0, 0.0]),
            1,
            &ChunkScope::all(),
        )
        .unwrap();
    let top = hits.first().unwrap();
    assert_eq!(top.id, ChunkId::from("a0"));
    assert_eq!(top.heading.as_deref(), Some("Exclusions"));
    assert_eq!(top.page, Some(12));
    assert!(top.score > 0.99, "{}", top.score);
}

/// A chunk scope narrows both legs of retrieval, and a scope that names
/// no chunks returns nothing rather than widening to the workspace.
#[test]
fn a_chunk_scope_narrows_both_legs_and_an_empty_one_finds_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let db = seeded_for_search(&config, "ws-chunk-scope");

    let only_a0 = ChunkScope::all().and_chunks([ChunkId::from("a0")]);
    let keyword = db.search_keyword_chunks("flood", 5, &only_a0).unwrap();
    assert_eq!(
        keyword.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["a0"]
    );
    let vector = db
        .search_similar_chunks(&Vector::from(vec![0.0, 0.0, 1.0, 0.0]), 5, &only_a0)
        .unwrap();
    assert_eq!(
        vector.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["a0"],
        "the closer chunk b1 is outside the scope"
    );
    let hybrid = db
        .search_hybrid_chunks(
            "claims",
            &Vector::from(vec![0.0, 0.0, 1.0, 0.0]),
            HybridLimits {
                top_k: 5,
                rrf_k: 60,
            },
            &only_a0,
        )
        .unwrap();
    assert!(hybrid.iter().all(|h| h.id == "a0"), "{hybrid:?}");

    // Both filters at once, contradicting each other.
    let crossed =
        ChunkScope::documents([DocumentId::from("doc-b")]).and_chunks([ChunkId::from("a0")]);
    assert!(
        db.search_keyword_chunks("flood", 5, &crossed)
            .unwrap()
            .is_empty()
    );

    // An entity whose chunk set came back empty must find nothing.
    let nothing = ChunkScope::all().and_chunks(Vec::new());
    assert!(nothing.is_empty());
    assert!(
        db.search_keyword_chunks("flood", 5, &nothing)
            .unwrap()
            .is_empty()
    );
    assert!(
        db.search_similar_chunks(&Vector::from(vec![1.0, 0.0, 0.0, 0.0]), 5, &nothing)
            .unwrap()
            .is_empty()
    );
    assert!(!ChunkScope::all().is_empty());
}

#[test]
fn keyword_search_finds_exact_tokens_the_vector_misses() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let db = seeded_for_search(&config, "ws-keyword");
    let hits = db
        .search_keyword_chunks("POL-8841", 5, &ChunkScope::all())
        .unwrap();
    assert_eq!(
        hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["b0"]
    );
    let none = db
        .search_keyword_chunks("zebra", 5, &ChunkScope::all())
        .unwrap();
    assert!(none.is_empty());
    let filtered = db
        .search_keyword_chunks(
            "flood",
            5,
            &ChunkScope::documents([DocumentId::from("doc-b")]),
        )
        .unwrap();
    assert!(filtered.is_empty());
    // Stemming: an inflected query finds the base form in the chunk.
    let stemmed = db
        .search_keyword_chunks("exclusions", 5, &ChunkScope::all())
        .unwrap();
    assert!(!stemmed.is_empty());
    assert!(
        stemmed
            .iter()
            .all(|h| h.content.contains("excluded") || h.heading.as_deref() == Some("Exclusions")),
        "{stemmed:?}"
    );
}

#[test]
fn keyword_search_on_empty_workspace_is_empty() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let db = WorkspaceDb::open(&config, "ws-noindex").unwrap();
    assert!(
        db.search_keyword_chunks("anything", 3, &ChunkScope::all())
            .unwrap()
            .is_empty()
    );
    assert!(
        db.search_keyword_chunks("   ", 3, &ChunkScope::all())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn keyword_search_ranks_by_bm25_and_uses_headings() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let db = seeded_for_search(&config, "ws-bm25");
    // "flood" appears in a0's content; "exclusions" only in its heading.
    let hits = db
        .search_keyword_chunks("flood exclusions", 5, &ChunkScope::all())
        .unwrap();
    assert_eq!(hits.first().map(|h| h.id.as_str()), Some("a0"));
    assert_eq!(hits.len(), 1);
    // "thirty" only in b1; "claims" also only in b1's content here.
    let hits = db
        .search_keyword_chunks("claims thirty", 5, &ChunkScope::all())
        .unwrap();
    assert_eq!(hits.first().map(|h| h.id.as_str()), Some("b1"));
    assert!(hits.iter().all(|h| h.score > 0.0));
}

#[test]
fn legacy_workspace_gets_its_terms_indexed_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    {
        let db = WorkspaceDb::open(&config, "ws-reindex").unwrap();
        db.insert_document(
            &NewDocument::new(&DocumentId::from("d"), "a.md", "text/markdown", 1)
                .with_status(DocumentStatus::Ready),
        )
        .unwrap();
        db.chunk_writer(&DocumentId::from("d"), "renewal POL-8841 notice")
            .and_then(|writer| {
                writer.insert(&NewChunk {
                    id: &ChunkId::from("c0"),
                    chunk_index: 0,
                    content: "renewal POL-8841 notice",
                    heading: None,
                    page: None,
                    kind: SectionKind::Body,
                    locator: None,
                    embedding: None,
                })
            })
            .unwrap();
        // Simulate a v5 workspace: unstemmed term rows, old version recorded.
        db.execute_statement("DELETE FROM _quack_terms").unwrap();
        db.execute_statement("INSERT INTO _quack_terms VALUES ('c0', 'renewal', 1)")
            .unwrap();
        db.execute_statement("UPDATE _quack_meta SET value = '5' WHERE key = 'schema_version'")
            .unwrap();
        assert!(
            db.search_keyword_chunks("8841", 3, &ChunkScope::all())
                .unwrap()
                .is_empty()
        );
    }
    let db = WorkspaceDb::open(&config, "ws-reindex").unwrap();
    assert_eq!(
        db.meta(MetaKey::SchemaVersion).unwrap().as_deref(),
        Some("15")
    );
    let hits = db
        .search_keyword_chunks("8841", 3, &ChunkScope::all())
        .unwrap();
    assert_eq!(hits.first().map(|h| h.id.as_str()), Some("c0"));
    // Rebuilt with stems: the inflected query matches now.
    let renewals = db
        .search_keyword_chunks("renewals", 3, &ChunkScope::all())
        .unwrap();
    assert_eq!(renewals.first().map(|h| h.id.as_str()), Some("c0"));
    assert!(
        !db.list_tables()
            .unwrap()
            .iter()
            .any(|t| t.starts_with("fts_"))
    );
}

#[test]
fn hybrid_search_fuses_vector_and_keyword_rankings() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let db = seeded_for_search(&config, "ws-hybrid");
    // Vector nearest is a0 (flood); the keyword "POL-8841" only matches b0.
    let hits = db
        .search_hybrid_chunks(
            "POL-8841",
            &Vector::from(vec![1.0, 0.0, 0.0, 0.0]),
            HybridLimits {
                top_k: 3,
                rrf_k: 60,
            },
            &ChunkScope::all(),
        )
        .unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids.len(), 3);
    // b0 ranks first in keyword and second in vector, so it wins the fusion.
    assert_eq!(ids.first().copied(), Some("b0"), "{ids:?}");
    assert!(ids.contains(&"a0"));
    let top_score = hits.first().unwrap().score;
    assert!(hits.iter().all(|h| h.score <= top_score));

    let limited = db
        .search_hybrid_chunks(
            "flood",
            &Vector::from(vec![1.0, 0.0, 0.0, 0.0]),
            HybridLimits {
                top_k: 1,
                rrf_k: 60,
            },
            &ChunkScope::all(),
        )
        .unwrap();
    assert_eq!(limited.len(), 1);
    assert_eq!(limited.first().unwrap().id, ChunkId::from("a0"));
}

#[tokio::test]
async fn ingest_markdown_stores_headings_and_pinned_flag() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-md-meta").unwrap();
    let writer = writer_of(&db);
    let md = b"# Exclusions\n\nFlood is excluded.\n\n# Claims\n\nClose in thirty days.\n";
    let result = ingestion::ingest_file(
        &config,
        &writer,
        "ws-md-meta",
        &ingestion::NewFile::new("rules.md", md),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    assert_eq!(result.chunks_stored, 2);
    let rows = db
        .execute_query("SELECT heading, page FROM _quack_chunks ORDER BY chunk_index")
        .unwrap();
    assert_eq!(rows.rows.len(), 2);
    assert_eq!(
        rows.rows.first().unwrap().first().unwrap().as_str(),
        Some("Exclusions")
    );
    assert!(rows.rows.first().unwrap().last().unwrap().is_null());
    let hits = db
        .search_keyword_chunks("thirty", 5, &ChunkScope::all())
        .unwrap();
    assert_eq!(
        hits.first().map(|h| h.heading.as_deref()),
        Some(Some("Claims"))
    );

    let doc = db
        .documents(&DocumentListing::default())
        .unwrap()
        .documents
        .into_iter()
        .next()
        .unwrap();
    assert_eq!(doc.pinning, Pinning::Unpinned);
    db.set_document_pinning(&doc.id, Pinning::Pinned).unwrap();
    let pinned = db.pinned_documents(Tokens::new(u32::MAX)).unwrap();
    assert_eq!(pinned.len(), 1);
    assert!(matches!(
        &pinned.first().unwrap().text,
        PinnedText::Included(text) if text.contains("Flood is excluded.")
    ));
}

#[tokio::test]
async fn identical_bytes_are_skipped_and_a_failed_document_is_retried() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-dedup").unwrap();
    let writer = writer_of(&db);
    let md = b"# Renewal terms\n\nRenewals close in thirty days.\n";
    let first = ingestion::ingest_file(
        &config,
        &writer,
        "ws-dedup",
        &ingestion::NewFile::new("terms.md", md).source(DocumentSource::Stdin),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();

    let doc = db.document(&first.document_id).unwrap().unwrap();
    assert_eq!(doc.title.as_deref(), Some("Renewal terms"));
    assert_eq!(doc.source, DocumentSource::Stdin);
    assert_eq!(doc.sha256.as_deref().map(str::len), Some(64));
    assert_eq!(doc.chunk_count, Some(1));
    assert_eq!(doc.display_name(), "Renewal terms");

    // Same bytes under another name: skipped, naming the existing document.
    let again = ingestion::ingest_file(
        &config,
        &writer,
        "ws-dedup",
        &ingestion::NewFile::new("copy.md", md).title(Some("Copy")),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
    .unwrap();
    let existing = match again {
        ingestion::IngestOutcome::Duplicate(existing) => Some(existing),
        ingestion::IngestOutcome::Ingested(_) => None,
    };
    assert_eq!(existing.map(|d| d.id), Some(first.document_id.clone()));
    assert_eq!(
        db.documents(&DocumentListing::default())
            .unwrap()
            .documents
            .len(),
        1
    );

    // An explicit title wins over the parsed heading.
    let titled = ingestion::ingest_file(
        &config,
        &writer,
        "ws-dedup",
        &ingestion::NewFile::new("other.md", b"# Heading\n\nBody.\n").title(Some(" Given ")),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    let doc = db.document(&titled.document_id).unwrap().unwrap();
    assert_eq!(doc.title.as_deref(), Some("Given"));
    assert_eq!(doc.source, DocumentSource::Path);

    // A document that failed does not block a retry of the same bytes.
    let bad = b"%PDF-1.4 not really a pdf";
    let failed = ingestion::ingest_file(
        &config,
        &writer,
        "ws-dedup",
        &ingestion::NewFile::new("scan.pdf", bad),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await;
    assert!(failed.is_err());
    let errored = db
        .documents(&DocumentListing::default())
        .unwrap()
        .documents
        .into_iter()
        .find(|d| d.filename == "scan.pdf")
        .unwrap();
    assert_eq!(errored.status, DocumentStatus::Error);
    let retry =
        ingestion::register_document(&db, &config, &ingestion::NewFile::new("scan.pdf", bad))
            .unwrap();
    assert!(matches!(retry, ingestion::Registration::New(_)));
}

#[tokio::test]
async fn a_long_pdf_ingests_every_page_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-long-pdf").unwrap();
    let writer = writer_of(&db);

    let mut pdf = pdf_oxide::writer::DocumentBuilder::new().title("Long Report");
    for page in 1..=60 {
        pdf.letter_page()
            .at(72.0, 720.0)
            .text(&format!("Page {page} of the long report"))
            .done();
    }
    let bytes = pdf.build().unwrap();

    let result = ingestion::ingest_file(
        &config,
        &writer,
        "ws-long-pdf",
        &ingestion::NewFile::new("report.pdf", &bytes),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    assert_eq!(result.pages.and_then(PageCounts::note), None);
    assert!(result.chunks_stored > 0);

    let doc = db.document(&result.document_id).unwrap().unwrap();
    assert_eq!(doc.status, DocumentStatus::Ready);
    assert_eq!(doc.title.as_deref(), Some("Long Report"));

    // Pages are windowed as one text: the first chunk starts on page 1,
    // every page's line survives, and the document title heads each chunk.
    let sql = format!(
        "SELECT min(page), \
                count(*) FILTER (WHERE content LIKE '%Page 40 of%'), \
                count(*) FILTER (WHERE content LIKE '%Page 60 of%'), \
                count(*) FILTER (WHERE heading = 'Long Report') = count(*) \
         FROM _quack_chunks WHERE document_id = '{}'",
        result.document_id
    );
    let qr = db.execute_query(&sql).unwrap();
    let row = qr.rows.first().unwrap();
    assert_eq!(row.first(), Some(&serde_json::Value::Number(1.into())));
    assert_eq!(row.get(1), Some(&serde_json::Value::Number(1.into())));
    assert_eq!(row.get(2), Some(&serde_json::Value::Number(1.into())));
    assert_eq!(row.get(3), Some(&serde_json::Value::Bool(true)));
}

#[test]
fn piped_bytes_load_as_a_temporary_stdin_table() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-stdin").unwrap();

    assert_eq!(
        ingestion::load_stdin_table(&config, &db, "ws-stdin", b"  \n").unwrap(),
        None
    );

    let csv = b"region\ttotal\nnorth\t10\nsouth\t20\n";
    assert_eq!(
        ingestion::load_stdin_table(&config, &db, "ws-stdin", csv).unwrap(),
        Some("stdin")
    );
    let rows = db
        .execute_query("SELECT sum(total) AS s FROM stdin")
        .unwrap();
    assert_eq!(
        rows.rows.first().and_then(|r| r.first()),
        Some(&serde_json::json!(30))
    );
    assert!(db.list_tables().unwrap().contains(&String::from("stdin")));
    // The scratch file is gone; the table lives on the connection only.
    let leftovers: Vec<_> = std::fs::read_dir(config.workspace_files_dir("ws-stdin"))
        .unwrap()
        .flatten()
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");

    let json = b"[{\"k\": \"a\", \"v\": 1}, {\"k\": \"b\", \"v\": 2}]";
    ingestion::load_stdin_table(&config, &db, "ws-stdin", json).unwrap();
    let rows = db.execute_query("SELECT count(*) AS n FROM stdin").unwrap();
    assert_eq!(
        rows.rows.first().and_then(|r| r.first()),
        Some(&serde_json::json!(2))
    );

    // Closed first: Windows lets no second handle open the file (#448).
    drop(db);
    let reopened = WorkspaceDb::open(&config, "ws-stdin").unwrap();
    assert!(
        !reopened
            .list_tables()
            .unwrap()
            .contains(&String::from("stdin"))
    );
}

/// A minimal two-sheet workbook with inline strings, enough for calamine.
fn tiny_xlsx() -> Vec<u8> {
    use std::io::Write as _;
    let parts: [(&str, &str); 6] = [
        (
            "[Content_Types].xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>"#,
        ),
        (
            "_rels/.rels",
            r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#,
        ),
        (
            "xl/workbook.xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sales Q1" sheetId="1" r:id="rId1"/><sheet name="Notes" sheetId="2" r:id="rId2"/></sheets></workbook>"#,
        ),
        (
            "xl/_rels/workbook.xml.rels",
            r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/></Relationships>"#,
        ),
        (
            "xl/worksheets/sheet1.xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>region</t></is></c><c r="B1" t="inlineStr"><is><t>total</t></is></c><c r="C1" t="inlineStr"><is><t>when</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>north</t></is></c><c r="B2"><v>10</v></c><c r="C2" s="1"><v>45000</v></c></row><row r="3"><c r="A3" t="inlineStr"><is><t>south, east</t></is></c><c r="B3"><v>20.5</v></c><c r="C3" s="1"><v>45001</v></c></row></sheetData></worksheet>"#,
        ),
        (
            "xl/worksheets/sheet2.xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>note</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>only one column</t></is></c></row></sheetData></worksheet>"#,
        ),
    ];
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = zip::ZipWriter::new(&mut cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (name, content) in parts {
            writer.start_file(name, options).unwrap();
            writer.write_all(content.as_bytes()).unwrap();
        }
        writer.finish().unwrap();
    }
    cursor.into_inner()
}

#[tokio::test]
async fn workbook_loads_one_table_per_sheet_and_delete_drops_them() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-xlsx").unwrap();
    let writer = writer_of(&db);
    let bytes = tiny_xlsx();
    let result = ingestion::ingest_file(
        &config,
        &writer,
        "ws-xlsx",
        &ingestion::NewFile::new("Region Sales.xlsx", &bytes),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    assert_eq!(result.file_type, FileType::Xlsx);
    assert_eq!(
        result.tables,
        ["Region_Sales_Sales_Q1", "Region_Sales_Notes"]
    );
    let mut tables = db.list_tables().unwrap();
    tables.sort();
    assert_eq!(tables, ["Region_Sales_Notes", "Region_Sales_Sales_Q1"]);

    let rows = db
        .execute_query("SELECT region, total FROM Region_Sales_Sales_Q1 ORDER BY total")
        .unwrap();
    assert_eq!(
        rows.rows,
        vec![
            vec![serde_json::json!("north"), serde_json::json!(10.0)],
            vec![serde_json::json!("south, east"), serde_json::json!(20.5)],
        ]
    );
    let doc = db.document(&result.document_id).unwrap().unwrap();
    assert_eq!(
        doc.tables.as_deref(),
        Some(
            &[
                "Region_Sales_Sales_Q1".to_owned(),
                "Region_Sales_Notes".to_owned()
            ][..]
        )
    );

    let files_dir = config.workspace_files_dir("ws-xlsx");
    assert_eq!(
        std::fs::read_dir(&files_dir).unwrap().count(),
        2,
        "one CSV per sheet"
    );
    assert!(db.delete_document(&result.document_id).unwrap());
    assert!(db.list_tables().unwrap().is_empty());
    assert!(db.document(&result.document_id).unwrap().is_none());
    // The workbook and its per-sheet CSVs are gone from files/ too.
    assert_eq!(std::fs::read_dir(&files_dir).unwrap().count(), 0);
}

/// Like `tiny_xlsx`, but its two sheet names (`Sales Q1` and `Sales-Q1`)
/// both sanitize to `Sales_Q1`, so a workbook's table names collide.
fn colliding_xlsx() -> Vec<u8> {
    use std::io::Write as _;
    let parts: [(&str, &str); 6] = [
        (
            "[Content_Types].xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>"#,
        ),
        (
            "_rels/.rels",
            r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#,
        ),
        (
            "xl/workbook.xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sales Q1" sheetId="1" r:id="rId1"/><sheet name="Sales-Q1" sheetId="2" r:id="rId2"/></sheets></workbook>"#,
        ),
        (
            "xl/_rels/workbook.xml.rels",
            r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/></Relationships>"#,
        ),
        (
            "xl/worksheets/sheet1.xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>region</t></is></c><c r="B1" t="inlineStr"><is><t>total</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>north</t></is></c><c r="B2"><v>10</v></c></row></sheetData></worksheet>"#,
        ),
        (
            "xl/worksheets/sheet2.xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>note</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>second sheet only</t></is></c></row></sheetData></worksheet>"#,
        ),
    ];
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = zip::ZipWriter::new(&mut cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (name, content) in parts {
            writer.start_file(name, options).unwrap();
            writer.write_all(content.as_bytes()).unwrap();
        }
        writer.finish().unwrap();
    }
    cursor.into_inner()
}

#[tokio::test]
async fn workbook_colliding_sheet_names_fail_instead_of_overwriting_a_sheet() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-collide").unwrap();
    let writer = writer_of(&db);

    let outcome = ingestion::ingest_file(
        &config,
        &writer,
        "ws-collide",
        &ingestion::NewFile::new("Region Sales.xlsx", &colliding_xlsx()),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await;

    // The ingest is refused: the two sheet names collide on one table after
    // sanitization, instead of one sheet silently overwriting the other.
    let err = outcome.err().unwrap();
    let message = err.to_string();
    assert!(
        matches!(&err, Error::Ingestion(_)),
        "expected Error::Ingestion, got: {err}"
    );
    assert!(
        message.contains("Sales Q1") && message.contains("Sales-Q1"),
        "the error should name both colliding sheets: {message}"
    );
    assert!(
        message.contains("Region_Sales_Sales_Q1"),
        "the error should name the colliding table: {message}"
    );

    // No table was created, so no query can read the wrong sheet's rows.
    assert!(
        db.list_tables().unwrap().is_empty(),
        "no tables should survive a refused collision"
    );
    let files_dir = config.workspace_files_dir("ws-collide");
    assert_eq!(
        std::fs::read_dir(&files_dir).unwrap().count(),
        0,
        "no per-sheet CSVs should be written on a refused collision"
    );

    // The document row records the failure — not `ready` with a duplicated
    // `tables` list advertising a table that no longer holds its sheet's data.
    let docs = db.documents(&DocumentListing::default()).unwrap().documents;
    assert_eq!(docs.len(), 1, "one document row for the failed ingest");
    let doc = docs.first().unwrap();
    assert_eq!(doc.status, DocumentStatus::Error);
    assert!(
        doc.tables.is_none(),
        "no tables recorded for a failed ingest"
    );
    assert!(
        doc.error_message
            .as_deref()
            .is_some_and(|m| m.contains("Sales Q1") && m.contains("Sales-Q1")),
        "the document's error_message should record the collision: {message}"
    );

    // Re-ingesting the same bytes after the failure (no duplicate lives) can
    // retry once the sheets are renamed: the failed row is gone, so a fresh
    // ingest starts clean rather than being treated as a duplicate.
    assert!(
        db.delete_document(&doc.id).unwrap(),
        "the error document can be deleted"
    );
    assert!(
        db.documents(&DocumentListing::default())
            .unwrap()
            .documents
            .is_empty()
    );
    assert!(db.list_tables().unwrap().is_empty());
    assert_eq!(std::fs::read_dir(&files_dir).unwrap().count(), 0);
}

/// A two-sheet workbook whose sheet names are `s1` and `s2`, each with one
/// header row and one data row (`a`/`b` for sheet 1, `c`/`d` for sheet 2).
/// Used to exercise collisions on characters other than space-vs-hyphen
/// (e.g. `Sheet 1` vs `Sheet_1`, where the underscore is already in one name).
fn two_sheet_xlsx(s1: &str, s2: &str) -> Vec<u8> {
    use std::io::Write as _;
    let workbook_xml = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="{s1}" sheetId="1" r:id="rId1"/><sheet name="{s2}" sheetId="2" r:id="rId2"/></sheets></workbook>"#,
    );
    let parts: [(&str, &str); 6] = [
        (
            "[Content_Types].xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>"#,
        ),
        (
            "_rels/.rels",
            r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#,
        ),
        ("xl/workbook.xml", workbook_xml.as_str()),
        (
            "xl/_rels/workbook.xml.rels",
            r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/></Relationships>"#,
        ),
        (
            "xl/worksheets/sheet1.xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>b</t></is></c></row></sheetData></worksheet>"#,
        ),
        (
            "xl/worksheets/sheet2.xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>c</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>d</t></is></c></row></sheetData></worksheet>"#,
        ),
    ];
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = zip::ZipWriter::new(&mut cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (name, content) in parts {
            writer.start_file(name, options).unwrap();
            writer.write_all(content.as_bytes()).unwrap();
        }
        writer.finish().unwrap();
    }
    cursor.into_inner()
}

#[tokio::test]
async fn workbook_colliding_sheet_names_with_an_underscore_in_one_name_are_refused() {
    // `Sheet 1` (space) and `Sheet_1` (underscore) both sanitize to
    // `Sheet_1`, so the table names collide. This confirms the fix is
    // data-driven (not specific to the space-vs-hyphen pair) and that an
    // underscore already in one sheet's name is caught too.
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-underscore").unwrap();
    let writer = writer_of(&db);

    let outcome = ingestion::ingest_file(
        &config,
        &writer,
        "ws-underscore",
        &ingestion::NewFile::new("Book.xlsx", &two_sheet_xlsx("Sheet 1", "Sheet_1")),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await;

    let err = outcome.err().unwrap();
    assert!(
        matches!(&err, Error::Ingestion(_)),
        "expected Error::Ingestion, got: {err}"
    );
    let message = err.to_string();
    assert!(
        message.contains("Sheet 1") && message.contains("Sheet_1"),
        "the error should name both colliding sheets: {message}"
    );
    assert!(
        message.contains("Book_Sheet_1"),
        "the error should name the colliding table: {message}"
    );
    assert!(
        db.list_tables().unwrap().is_empty(),
        "no tables should survive a refused collision"
    );
    let files_dir = config.workspace_files_dir("ws-underscore");
    assert_eq!(
        std::fs::read_dir(&files_dir).unwrap().count(),
        0,
        "no per-sheet CSVs should be written on a refused collision"
    );
    let docs = db.documents(&DocumentListing::default()).unwrap().documents;
    assert_eq!(docs.len(), 1);
    let doc = docs.first().unwrap();
    assert_eq!(doc.status, DocumentStatus::Error);
    assert!(doc.tables.is_none());
}

#[tokio::test]
async fn office_and_html_documents_are_chunked_with_titles() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-office").unwrap();
    let writer = writer_of(&db);
    let page = b"<html><head><title>Renewal Guide</title></head><body><h1>Terms</h1><p>Thirty days.</p></body></html>";
    let result = ingestion::ingest_file(
        &config,
        &writer,
        "ws-office",
        &ingestion::NewFile::new("guide.html", page),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    assert_eq!(result.file_type, FileType::Html);
    assert_eq!(result.chunks_stored, 1);
    let doc = db.document(&result.document_id).unwrap().unwrap();
    assert_eq!(doc.title.as_deref(), Some("Renewal Guide"));
    let chunk = db
        .execute_query("SELECT heading, content FROM _quack_chunks")
        .unwrap();
    assert_eq!(
        chunk.rows,
        vec![vec![
            serde_json::json!("Terms"),
            serde_json::json!("Thirty days.")
        ]]
    );

    let failed = ingestion::ingest_file(
        &config,
        &writer,
        "ws-office",
        &ingestion::NewFile::new("deck.pptx", b"not a package"),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await;
    assert!(failed.is_err());
    let errored = db
        .documents(&DocumentListing::default())
        .unwrap()
        .documents
        .into_iter()
        .find(|d| d.filename == "deck.pptx")
        .unwrap();
    assert!(
        errored
            .error_message
            .is_some_and(|m| m.contains("not a PowerPoint file"))
    );
}

#[tokio::test]
async fn a_pdf_page_without_text_is_counted_on_the_document() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-pages").unwrap();
    let writer = writer_of(&db);
    // Three pages; the second carries no text, as a scanned image would.
    let mut pdf = pdf_oxide::writer::DocumentBuilder::new().title("Mixed");
    pdf.letter_page().at(72.0, 720.0).text("First page").done();
    pdf.letter_page().done();
    pdf.letter_page().at(72.0, 720.0).text("Third page").done();
    let bytes = pdf.build().unwrap();

    let result = ingestion::ingest_file(
        &config,
        &writer,
        "ws-pages",
        &ingestion::NewFile::new("mixed.pdf", &bytes),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    let counts = PageCounts {
        total: 3,
        unreadable: 0,
        empty: 1,
    };
    assert_eq!(result.pages, Some(counts));
    assert_eq!(
        result.pages.and_then(PageCounts::note).as_deref(),
        Some("1 of 3 pages without text")
    );

    let doc = db.document(&result.document_id).unwrap().unwrap();
    assert_eq!(doc.status, DocumentStatus::Ready);
    assert_eq!(doc.pages, Some(counts));
    assert_eq!(
        doc.pages.and_then(PageCounts::note).as_deref(),
        Some("1 of 3 pages without text")
    );

    // A source without pages records none.
    let text = ingestion::ingest_file(
        &config,
        &writer,
        "ws-pages",
        &ingestion::NewFile::new("notes.md", b"# Notes\n\nNo pages here.\n"),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    assert_eq!(text.pages, None);
    let doc = db.document(&text.document_id).unwrap().unwrap();
    assert_eq!(doc.pages, None);
}

/// An image is refused before it is registered when no vision model is
/// set.
#[tokio::test]
async fn an_image_without_a_vision_model_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-images").unwrap();
    let refused = ingestion::register_document(
        &db,
        &config,
        &ingestion::NewFile::new("chart.png", b"\x89PNG\r\n\x1a\n"),
    );
    assert!(
        matches!(refused, Err(Error::NoVisionModel(ref name)) if name == "chart.png"),
        "{refused:?}"
    );
    assert!(
        !every_document(&db)
            .iter()
            .any(|d| d.filename == "chart.png")
    );
}

/// An image the vision model could not read fails as a document and
/// leaves no stored copy behind.
#[tokio::test]
async fn an_image_the_model_cannot_read_leaves_no_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut config: Config = toml::from_str(
        "[ingestion]\nvision_model = \"down/vision\"\n\
         [providers.down]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\nmax_retries = 0\n",
    )
    .unwrap();
    config.general.data_dir = dir.path().to_path_buf();
    let db = WorkspaceDb::open(&config, "ws-unread").unwrap();
    let writer = writer_of(&db);
    let failed = Egress::scope(
        Some(Egress::NoWorkspace),
        ingestion::ingest_file(
            &config,
            &writer,
            "ws-unread",
            &ingestion::NewFile::new("chart.png", b"\x89PNG\r\n\x1a\n"),
            None::<&Embedder<MockEmbeddingModel>>,
        ),
    )
    .await;
    assert!(failed.is_err());
    let document = every_document(&db)
        .into_iter()
        .find(|d| d.filename == "chart.png")
        .unwrap();
    assert_eq!(document.status, DocumentStatus::Error);
    let files: Vec<String> = std::fs::read_dir(config.workspace_files_dir("ws-unread"))
        .map(|entries| {
            entries
                .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        !files.iter().any(|f| f.starts_with(document.id.as_str())),
        "{files:?}"
    );
}

/// Deleting an image document removes the image it keeps.
#[tokio::test]
async fn deleting_an_image_document_removes_its_image() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-image").unwrap();
    let id = DocumentId::from("img");
    db.insert_document(
        &NewDocument::new(&id, "chart.png", "image/png", 8).with_status(DocumentStatus::Ready),
    )
    .unwrap();
    let document = db.document(&id).unwrap().unwrap();
    let image = db.stored_image(&document).unwrap();
    assert_eq!(
        image.path(),
        config.workspace_files_dir("ws-image").join("img.png")
    );
    std::fs::create_dir_all(image.path().parent().unwrap()).unwrap();
    std::fs::write(image.path(), b"png").unwrap();
    assert!(db.has_images().unwrap());

    assert!(db.delete_document(&id).unwrap());
    assert!(!image.path().exists());
    assert!(!db.has_images().unwrap());
}

/// `tiny_xlsx` with one more sheet part of `megabytes` of spaces: a few
/// kilobytes more on disk.
fn padded_xlsx(megabytes: usize) -> Vec<u8> {
    use std::io::Write as _;
    let mut writer = zip::ZipWriter::new_append(std::io::Cursor::new(tiny_xlsx())).unwrap();
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    writer
        .start_file("xl/worksheets/sheet3.xml", options)
        .unwrap();
    writer
        .write_all(" ".repeat(megabytes.saturating_mul(1024 * 1024)).as_bytes())
        .unwrap();
    writer.finish().unwrap().into_inner()
}

/// A Word or `PowerPoint` package of one part: a word, then `megabytes` of
/// spaces.
/// A DOCX or PPTX package (by `part`, its main content part) whose text is
/// padded with `megabytes` of spaces, so it inflates far past its size.
fn padded_package(part: &str, megabytes: usize) -> Vec<u8> {
    use std::io::Write as _;
    let padding = " ".repeat(megabytes.saturating_mul(1024 * 1024));
    let docx = part.starts_with("word/");
    let main = if docx {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t xml:space="preserve">Padded{padding}</w:t></w:r></w:p></w:body></w:document>"#
        )
    } else {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?><p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:nvSpPr><p:cNvPr id="2" name="Body"/><p:cNvSpPr/><p:nvPr><p:ph type="body"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Padded{padding}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"#
        )
    };
    let (types, rels): (String, &str) = if docx {
        (
            String::from(
                r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>"#,
            ),
            r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>"#,
        )
    } else {
        (
            String::from(
                r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/></Types>"#,
            ),
            r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>"#,
        )
    };
    let mut parts: Vec<(&str, &str)> = vec![("[Content_Types].xml", &types), ("_rels/.rels", rels)];
    if !docx {
        parts.push((
            "ppt/presentation.xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst></p:presentation>"#,
        ));
        parts.push((
            "ppt/_rels/presentation.xml.rels",
            r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>"#,
        ));
    }
    parts.push((part, &main));
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (name, content) in parts {
        writer.start_file(name, options).unwrap();
        writer.write_all(content.as_bytes()).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

#[tokio::test]
async fn a_file_that_inflates_past_the_limit_ends_in_error_and_a_normal_one_loads() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = test_config_no_provider(dir.path());
    config.ingestion.max_decompressed_mb = 1;
    let db = WorkspaceDb::open(&config, "ws-bomb").unwrap();
    let writer = writer_of(&db);

    let bombs = [
        ("bomb.docx", padded_package("word/document.xml", 2)),
        ("bomb.pptx", padded_package("ppt/slides/slide1.xml", 2)),
        ("bomb.xlsx", padded_xlsx(2)),
    ];
    for (filename, bytes) in &bombs {
        assert!(bytes.len() < 64 * 1024, "{filename}: {} bytes", bytes.len());
        let err = ingestion::ingest_file(
            &config,
            &writer,
            "ws-bomb",
            &ingestion::NewFile::new(filename, bytes),
            None::<&Embedder<MockEmbeddingModel>>,
        )
        .await
        .err()
        .unwrap();
        assert!(matches!(&err, Error::Ingestion(_)), "{filename}: {err}");
        let doc = db
            .documents(&DocumentListing::default())
            .unwrap()
            .documents
            .into_iter()
            .find(|d| d.filename == *filename)
            .unwrap();
        assert_eq!(doc.status, DocumentStatus::Error, "{filename}");
        let message = doc.error_message.unwrap();
        assert!(
            message.contains("more than [ingestion].max_decompressed_mb (1 MB)"),
            "{filename}: {message}"
        );
    }
    assert!(db.list_tables().unwrap().is_empty());

    // Under the same limit, files that fit still load.
    let normal = [
        ("fits.docx", padded_package("word/document.xml", 0)),
        ("fits.pptx", padded_package("ppt/slides/slide1.xml", 0)),
        ("fits.xlsx", tiny_xlsx()),
    ];
    for (filename, bytes) in &normal {
        let loaded = ingestion::ingest_file(
            &config,
            &writer,
            "ws-bomb",
            &ingestion::NewFile::new(filename, bytes),
            None::<&Embedder<MockEmbeddingModel>>,
        )
        .await
        .unwrap()
        .ingested()
        .unwrap();
        assert!(
            loaded.chunks_stored > 0 || !loaded.tables.is_empty(),
            "{filename}"
        );
    }

    // The padded workbook is a workbook: a limit above its size loads it.
    config.ingestion.max_decompressed_mb = 4;
    let loaded = ingestion::ingest_file(
        &config,
        &writer,
        "ws-bomb",
        &ingestion::NewFile::new("padded.xlsx", &padded_xlsx(2)),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    assert_eq!(loaded.tables.len(), 2);
}

#[tokio::test]
async fn sqlite_sources_import_as_tables_with_every_column_as_text_then_sniffed() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-import").unwrap();
    let writer = writer_of(&db);
    // Beside the data directory, not inside it: quack's own files are
    // refused as a source (issue #69).
    let source_dir = tempfile::tempdir().unwrap();
    let source_path = source_dir.path().join("source.db");
    {
        use sqlx::{Connection as _, Executor as _};
        let url = format!("sqlite://{}?mode=rwc", source_path.display());
        let mut conn = sqlx::SqliteConnection::connect(&url).await.unwrap();
        conn.execute("CREATE TABLE orders (id INTEGER, region TEXT, total REAL, placed TEXT)")
            .await
            .unwrap();
        conn.execute(
            "INSERT INTO orders VALUES (1, 'north', 10.5, '2024-01-02'), (2, 'south, east', 20, NULL), (3, 'west', 7.25, '2024-03-04')",
        )
        .await
        .unwrap();
    }
    let url = format!("sqlite://{}", source_path.display());
    let request = ImportRequest {
        source_table: Some(String::from("orders")),
        ..ImportRequest::new(url.clone(), String::from("Orders Import"))
    };
    let summary = import::Importing {
        config: &config,
        db: &writer,
        workspace_id: "ws-import",
        request: &request,
        policy: ImportPolicy::owner(),
        embedder: None::<&Embedder<MockEmbeddingModel>>,
        control: RunControl::unobserved(),
    }
    .run()
    .await
    .unwrap();
    assert_eq!(summary.table, "Orders_Import");
    assert_eq!(summary.rows, 3);
    assert_eq!(summary.columns, ["id", "region", "total", "placed"]);
    assert_eq!(summary.source, url);
    let rows = db
        .execute_query("SELECT region, total, placed FROM Orders_Import ORDER BY id")
        .unwrap();
    assert_eq!(
        rows.rows,
        vec![
            vec![
                serde_json::json!("north"),
                serde_json::json!(10.5),
                serde_json::json!("2024-01-02")
            ],
            vec![
                serde_json::json!("south, east"),
                serde_json::json!(20.0),
                serde_json::Value::Null
            ],
            vec![
                serde_json::json!("west"),
                serde_json::json!(7.25),
                serde_json::json!("2024-03-04")
            ],
        ]
    );
    let doc = db.document(&summary.document_id).unwrap().unwrap();
    assert_eq!(doc.source, DocumentSource::Import);
    assert_eq!(doc.title.as_deref(), Some(url.as_str()));
    assert_eq!(
        doc.tables.as_deref(),
        Some(&[String::from("Orders_Import")][..])
    );

    // A query with a limit, into another table.
    let request = ImportRequest {
        query: Some(String::from(
            "SELECT region, total * 2 AS doubled FROM orders ORDER BY id",
        )),
        limit: Some(2),
        ..ImportRequest::new(url.clone(), String::from("big"))
    };
    let summary = import::Importing {
        config: &config,
        db: &writer,
        workspace_id: "ws-import",
        request: &request,
        policy: ImportPolicy::owner(),
        embedder: None::<&Embedder<MockEmbeddingModel>>,
        control: RunControl::unobserved(),
    }
    .run()
    .await
    .unwrap();
    assert_eq!((summary.rows, summary.columns.len()), (2, 2));
}

/// A query token that is also a SQL word stays a word (issue #62).
#[test]
fn keyword_search_treats_null_as_a_word() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let db = WorkspaceDb::open(&config, "ws-null").unwrap();
    db.insert_document(
        &NewDocument::new(&DocumentId::from("d"), "a.md", "text/markdown", 1)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();
    db.chunk_writer(&DocumentId::from("d"), "The null hypothesis was rejected.")
        .and_then(|writer| {
            writer.insert(&NewChunk {
                id: &ChunkId::from("c"),
                chunk_index: 0,
                content: "The null hypothesis was rejected.",
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: None,
            })
        })
        .unwrap();
    assert_eq!(
        db.search_keyword_chunks("null", 5, &ChunkScope::all())
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        db.search_keyword_chunks("null hypothesis", 5, &ChunkScope::all())
            .unwrap()
            .len(),
        1
    );
}

/// Only ready documents are searchable, and a pass that fails after
/// writing chunks takes them back out (issue #52).
#[tokio::test]
async fn failed_documents_are_not_searchable_and_leave_no_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let db = WorkspaceDb::open(&config, "ws-failed").unwrap();
    let writer = writer_of(&db);
    db.insert_document(
        &NewDocument::new(&DocumentId::from("d"), "a.md", "text/markdown", 1)
            .with_status(DocumentStatus::Error),
    )
    .unwrap();
    db.chunk_writer(&DocumentId::from("d"), "zebra crossing")
        .and_then(|writer| {
            writer.insert(&NewChunk {
                id: &ChunkId::from("c"),
                chunk_index: 0,
                content: "zebra crossing",
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: Some(&Vector::from(vec![1.0, 0.0, 0.0, 0.0])),
            })
        })
        .unwrap();
    assert!(
        db.search_keyword_chunks("zebra", 5, &ChunkScope::all())
            .unwrap()
            .is_empty()
    );
    assert!(
        db.search_similar_chunks(
            &Vector::from(vec![1.0, 0.0, 0.0, 0.0]),
            5,
            &ChunkScope::all()
        )
        .unwrap()
        .is_empty()
    );
    db.update_document_status(&DocumentId::from("d"), DocumentStatus::Ready)
        .unwrap();
    assert_eq!(
        db.search_keyword_chunks("zebra", 5, &ChunkScope::all())
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        db.search_similar_chunks(
            &Vector::from(vec![1.0, 0.0, 0.0, 0.0]),
            5,
            &ChunkScope::all()
        )
        .unwrap()
        .len(),
        1
    );

    // Embedding fails after the chunks were written: the row is an error
    // and nothing of it stays searchable or counted.
    let failed = ingestion::ingest_file(
        &config,
        &writer,
        "ws-failed",
        &ingestion::NewFile::new("notes.md", b"# Notes\n\nA giraffe walked by.\n"),
        Some(&embedder(FailingEmbeddingModel)),
    )
    .await;
    assert!(failed.is_err());
    let doc = db
        .documents(&DocumentListing::default())
        .unwrap()
        .documents
        .into_iter()
        .find(|d| d.filename == "notes.md")
        .unwrap();
    assert_eq!(doc.status, DocumentStatus::Error);
    assert_eq!(doc.chunk_count, None);
    let orphans: i64 = db
        .connection()
        .query_row(
            "SELECT count(*) FROM _quack_chunks WHERE document_id = ?",
            [&doc.id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(orphans, 0);
    let terms: i64 = db
        .connection()
        .query_row(
            "SELECT count(*) FROM _quack_terms WHERE term = 'giraff'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(terms, 0);
}

/// One document per table (issue #51): a changed file with the same
/// name is refused while its predecessor lives; identical bytes after
/// the table was dropped load again as a new document and the stale row
/// fails; rows a restart left queued are failed on open.
#[tokio::test]
async fn tables_have_one_owner_and_dedup_needs_the_table_to_exist() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-owner").unwrap();
    let writer = writer_of(&db);
    let first_bytes = b"region,total\nnorth,1\n";
    let ingested = |outcome: ingestion::IngestOutcome| match outcome {
        ingestion::IngestOutcome::Ingested(r) => Some(r),
        ingestion::IngestOutcome::Duplicate(_) => None,
    };
    let first = ingested(
        ingestion::ingest_file(
            &config,
            &writer,
            "ws-owner",
            &ingestion::NewFile::new("sales.csv", first_bytes),
            None::<&Embedder<MockEmbeddingModel>>,
        )
        .await
        .unwrap(),
    )
    .unwrap();
    assert_eq!(first.tables, vec![String::from("sales")]);

    let changed = ingestion::ingest_file(
        &config,
        &writer,
        "ws-owner",
        &ingestion::NewFile::new("sales.csv", b"region,total\nnorth,2\n"),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await;
    let err = changed.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(err.contains("belongs to document"), "{err}");
    assert!(err.contains(first.document_id.as_str()), "{err}");
    assert_eq!(
        db.documents(&DocumentListing::default())
            .unwrap()
            .documents
            .len(),
        1,
        "nothing was registered"
    );
    let rows: i64 = db
        .connection()
        .query_row("SELECT total FROM sales", [], |r| r.get(0))
        .unwrap();
    assert_eq!(rows, 1, "the owner's table is untouched");

    // The table is dropped behind the document's back: the same bytes
    // are no longer a duplicate.
    db.execute_statement("DROP TABLE sales").unwrap();
    let again = ingestion::ingest_file(
        &config,
        &writer,
        "ws-owner",
        &ingestion::NewFile::new("sales.csv", first_bytes),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
    .unwrap();
    let reloaded = ingested(again).unwrap();
    assert_ne!(reloaded.document_id, first.document_id);
    assert!(db.list_tables().unwrap().contains(&String::from("sales")));
    let stale = db.document(&first.document_id).unwrap().unwrap();
    assert_eq!(stale.status, DocumentStatus::Error);
    assert!(
        db.table_owner("sales")
            .unwrap()
            .is_some_and(|d| d.id == reloaded.document_id)
    );

    // A queued row from a process that died is failed when the server
    // opens the workspace.
    let registration = ingestion::register_document(
        &db,
        &config,
        &ingestion::NewFile::new("later.csv", b"a\n1\n"),
    )
    .unwrap();
    let ingestion::Registration::New(queued) = registration else {
        return assert!(matches!(registration, ingestion::Registration::New(_)));
    };
    assert_eq!(db.fail_stale_uploads().unwrap(), 1);
    let failed = db.document(&queued).unwrap().unwrap();
    assert_eq!(failed.status, DocumentStatus::Error);
    assert!(failed.error_message.is_some_and(|m| m.contains("restart")));
    assert_eq!(db.fail_stale_uploads().unwrap(), 0);
}

/// The server's default policy keeps a logged-in user off the server's
/// disk: a `sqlite:` path is refused before anything is opened.
#[tokio::test]
async fn server_policy_refuses_local_sqlite_files() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-import-policy").unwrap();
    let writer = writer_of(&db);
    let server_policy = ImportPolicy::server(&config);
    assert!(!server_policy.local_files);
    assert_eq!(server_policy.hosts, HostReach::PublicOnly);
    let local_file = import::Importing {
        config: &config,
        db: &writer,
        workspace_id: "ws-import-policy",
        request: &ImportRequest {
            source_table: Some(String::from("users")),
            ..ImportRequest::new(
                format!("sqlite://{}", dir.path().join("control.db").display()),
                String::from("x"),
            )
        },
        policy: server_policy,
        embedder: None::<&Embedder<MockEmbeddingModel>>,
        control: RunControl::unobserved(),
    }
    .run()
    .await;
    assert!(local_file.is_err_and(|e| e.to_string().contains("allow_local_files")));
    assert!(db.list_tables().unwrap().is_empty());
}

#[tokio::test]
async fn sqlite_import_errors_are_specific_and_duplicates_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-import-errors").unwrap();
    let writer = writer_of(&db);
    let source_dir = tempfile::tempdir().unwrap();
    let source_path = source_dir.path().join("source.db");
    {
        use sqlx::{Connection as _, Executor as _};
        let url = format!("sqlite://{}?mode=rwc", source_path.display());
        let mut conn = sqlx::SqliteConnection::connect(&url).await.unwrap();
        conn.execute("CREATE TABLE orders (id INTEGER, region TEXT)")
            .await
            .unwrap();
        conn.execute("INSERT INTO orders VALUES (1, 'north')")
            .await
            .unwrap();
    }
    let url = format!("sqlite://{}", source_path.display());
    let first = ImportRequest {
        source_table: Some(String::from("orders")),
        ..ImportRequest::new(url.clone(), String::from("Orders Import"))
    };
    let summary = import::Importing {
        config: &config,
        db: &writer,
        workspace_id: "ws-import-errors",
        request: &first,
        policy: ImportPolicy::owner(),
        embedder: None::<&Embedder<MockEmbeddingModel>>,
        control: RunControl::unobserved(),
    }
    .run()
    .await
    .unwrap();
    assert_eq!(summary.rows, 1);
    // The same rows again are a duplicate; a bad query and a bad URL are errors.
    let again = import::Importing {
        config: &config,
        db: &writer,
        workspace_id: "ws-import-errors",
        request: &first,
        policy: ImportPolicy::owner(),
        embedder: None::<&Embedder<MockEmbeddingModel>>,
        control: RunControl::unobserved(),
    }
    .run()
    .await;
    assert!(again.is_err_and(|e| e.to_string().contains("identical")));
    let bad = import::Importing {
        config: &config,
        db: &writer,
        workspace_id: "ws-import-errors",
        request: &ImportRequest {
            query: Some(String::from("SELECT * FROM nope")),
            ..ImportRequest::new(url, String::from("x"))
        },
        policy: ImportPolicy::owner(),
        embedder: None::<&Embedder<MockEmbeddingModel>>,
        control: RunControl::unobserved(),
    }
    .run()
    .await;
    assert!(bad.is_err_and(|e| e.to_string().contains("rejected the query")));
    let unsupported = import::Importing {
        config: &config,
        db: &writer,
        workspace_id: "ws-import-errors",
        request: &ImportRequest {
            source_table: Some(String::from("t")),
            ..ImportRequest::new(String::from("mysql://h/db"), String::from("x"))
        },
        policy: ImportPolicy::owner(),
        embedder: None::<&Embedder<MockEmbeddingModel>>,
        control: RunControl::unobserved(),
    }
    .run()
    .await;
    assert!(unsupported.is_err());
    // Delete through the document row drops the imported table.
    assert!(db.delete_document(&summary.document_id).unwrap());
    assert!(
        !db.list_tables()
            .unwrap()
            .contains(&String::from("Orders_Import"))
    );
}

/// Answers every batch only after `delay`, so a cancel can land while a
/// request is in flight.
struct SlowModel {
    delay: std::time::Duration,
}

impl EmbeddingModel for SlowModel {
    fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> impl Future<Output = Result<Vec<Embedding>, ProviderError>> + Send {
        let delay = self.delay;
        async move {
            tokio::time::sleep(delay).await;
            Ok(texts
                .into_iter()
                .map(|document| Embedding {
                    document,
                    vec: vec![0.1_f64; TEST_DIM],
                })
                .collect())
        }
    }
}

#[tokio::test]
async fn a_cancelled_ingest_stops_mid_embedding_and_leaves_no_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-cancel";
    let db = WorkspaceDb::open(&config, workspace_id).unwrap();
    let writer = writer_of(&db);
    let model = embedder(SlowModel {
        delay: std::time::Duration::from_secs(60),
    });
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        trigger.cancel();
    });
    let started = std::time::Instant::now();
    let outcome = ingestion::ingest_file(
        &config,
        &writer,
        workspace_id,
        &ingestion::NewFile::new("long.md", b"# Long\n\nSome text to embed.").control(RunControl {
            progress: &|_| {},
            cancel: Some(&cancel),
        }),
        Some(&model),
    )
    .await;
    assert!(matches!(outcome, Err(Error::Cancelled)), "{outcome:?}");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "the request in flight was abandoned, not waited out"
    );
    let documents = db.documents(&DocumentListing::default()).unwrap().documents;
    let document = documents.first().unwrap();
    assert_eq!(document.status, DocumentStatus::Error);
    assert_eq!(document.error_message.as_deref(), Some("cancelled"));
    let qr = db
        .execute_query("SELECT COUNT(*) AS cnt FROM _quack_chunks")
        .unwrap();
    assert_eq!(
        qr.rows.first().unwrap().first().unwrap(),
        &serde_json::Value::Number(0.into())
    );

    // Cancelled before it starts: nothing is parsed or stored.
    let early = CancellationToken::new();
    early.cancel();
    let outcome = ingestion::ingest_file(
        &config,
        &writer,
        workspace_id,
        &ingestion::NewFile::new("other.md", b"# Other").control(RunControl {
            progress: &|_| {},
            cancel: Some(&early),
        }),
        Some(&model),
    )
    .await;
    assert!(matches!(outcome, Err(Error::Cancelled)));
}

/// Write `text` at `path` under `root`, making the directories.
fn write_under(root: &Path, path: &str, text: &str) {
    let path = root.join(path);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// A folder run over `root` into `ws-folder` without an embedding model.
async fn folder_run(config: &Config, writer: &Writer, root: &Path, prune: Prune) -> FolderReport {
    Folder {
        config,
        db: writer,
        workspace_id: "ws-folder",
        root,
        embedder: None::<&Embedder<MockEmbeddingModel>>,
        control: RunControl::unobserved(),
        prune,
    }
    .run()
    .await
    .unwrap()
}

/// Each result as its path and the kind of its outcome.
fn outcome_kinds(report: &FolderReport) -> Vec<(String, &'static str)> {
    report
        .results
        .iter()
        .map(|r| {
            let kind = match r.outcome {
                Outcome::Ingested(_) => "ingested",
                Outcome::Replaced { .. } => "replaced",
                Outcome::Skipped(_) => "skipped",
                Outcome::Moved { .. } => "moved",
                Outcome::Failed(_) => "failed",
            };
            (r.relative.clone(), kind)
        })
        .collect()
}

/// A file moved within the folder is followed to its new path, so
/// `--prune` never deletes the only document of content that is still
/// there; content that moved into a new path while its old path changed is
/// ingested, not skipped; and symbolic links are not followed.
#[tokio::test]
async fn a_moved_file_is_followed_and_never_pruned() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-folder").unwrap();
    let writer = writer_of(&db);
    let root = dir.path().join("notes");
    write_under(&root, "a.md", "Alpha content.");
    write_under(&root, "b.md", "Bravo content.");
    folder_run(&config, &writer, &root, Prune::Keep).await;
    let canonical = std::fs::canonicalize(&root).unwrap();
    let root_text = canonical.to_string_lossy();
    let alpha = db
        .newest_document_at_path(&root_text, "a.md")
        .unwrap()
        .unwrap();

    // a.md moves to sub/c.md.
    std::fs::create_dir_all(root.join("sub")).unwrap();
    std::fs::rename(root.join("a.md"), root.join("sub/c.md")).unwrap();
    let moved = folder_run(&config, &writer, &root, Prune::Delete).await;
    assert_eq!(
        outcome_kinds(&moved),
        [
            (String::from("b.md"), "skipped"),
            (String::from("sub/c.md"), "moved"),
        ]
    );
    assert!(moved.gone.is_empty(), "{:?}", moved.gone);
    let at_new = db
        .newest_document_at_path(&root_text, "sub/c.md")
        .unwrap()
        .unwrap();
    assert_eq!(at_new.id, alpha.id);
    assert!(
        db.newest_document_at_path(&root_text, "a.md")
            .unwrap()
            .is_none()
    );

    // b.md's old content moves to a.md, which sorts first, while b.md
    // changes: b.md replaces its document first, so a.md is new content.
    write_under(&root, "a.md", "Bravo content.");
    write_under(&root, "b.md", "Bravo, revised.");
    let swapped = folder_run(&config, &writer, &root, Prune::Delete).await;
    assert_eq!(
        outcome_kinds(&swapped),
        [
            (String::from("a.md"), "ingested"),
            (String::from("b.md"), "replaced"),
            (String::from("sub/c.md"), "skipped"),
        ]
    );
    let live: Vec<String> = db
        .documents(&DocumentListing::default())
        .unwrap()
        .documents
        .into_iter()
        .filter_map(|d| d.source_path)
        .collect();
    assert_eq!(live.len(), 3, "{live:?}");

    // A link to a directory above the folder is not walked.
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(dir.path(), root.join("up")).unwrap();
        let linked = folder_run(&config, &writer, &root, Prune::Keep).await;
        assert!(
            linked
                .results
                .iter()
                .all(|r| !r.relative.starts_with("up/")),
            "{:?}",
            outcome_kinds(&linked)
        );
    }
}

/// A folder run ingests every supported file with its path, lists the
/// rest, counts a file that fails as one outcome among the others, and
/// skips every unchanged file on the next run.
#[tokio::test]
async fn a_folder_run_ingests_supported_files_and_skips_them_next_time() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-folder").unwrap();
    let writer = writer_of(&db);
    let root = dir.path().join("contracts");
    write_under(&root, "policy.md", "Flood is excluded.");
    write_under(&root, "rates/sales.csv", "region,total\nnorth,1\n");
    write_under(&root, "notes.xyz", "?");
    write_under(&root, "broken.csv", "a,b\n1,2,3,4\n\"unterminated,5\n6\n");
    let canonical = std::fs::canonicalize(&root).unwrap();
    let root_text = canonical.to_string_lossy();

    let first = folder_run(&config, &writer, &root, Prune::Keep).await;
    assert_eq!(
        outcome_kinds(&first),
        [
            (String::from("broken.csv"), "failed"),
            (String::from("policy.md"), "ingested"),
            (String::from("rates/sales.csv"), "ingested"),
        ]
    );
    assert_eq!(first.unsupported, ["notes.xyz"]);
    assert!(first.gone.is_empty());
    assert_eq!(first.failed(), 1);
    let policy = db
        .newest_document_at_path(&root_text, "policy.md")
        .unwrap()
        .unwrap();
    assert_eq!(policy.source_path.as_deref(), Some("policy.md"));
    assert_eq!(policy.source_root.as_deref(), Some(root_text.as_ref()));
    assert_eq!(policy.filename, "policy.md");
    assert_eq!(
        db.newest_document_at_path(&root_text, "rates/sales.csv")
            .unwrap()
            .unwrap()
            .tables,
        Some(vec![String::from("sales")])
    );
    assert!(
        db.newest_document_at_path(&root_text, "broken.csv")
            .unwrap()
            .is_none()
    );
    assert!(
        db.newest_document_at_path("/elsewhere", "policy.md")
            .unwrap()
            .is_none()
    );

    let again = folder_run(&config, &writer, &root, Prune::Keep).await;
    assert_eq!(
        outcome_kinds(&again),
        [
            (String::from("broken.csv"), "failed"),
            (String::from("policy.md"), "skipped"),
            (String::from("rates/sales.csv"), "skipped"),
        ]
    );
}

/// On a later run a changed file replaces the document at its path, and
/// a document whose file is gone is reported and kept, or deleted with
/// its table when pruning.
#[tokio::test]
async fn a_folder_rerun_replaces_changed_files_and_reports_or_prunes_gone_ones() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-folder").unwrap();
    let writer = writer_of(&db);
    let root = dir.path().join("contracts");
    write_under(&root, "policy.md", "Flood is excluded.");
    write_under(&root, "rates/sales.csv", "region,total\nnorth,1\n");
    folder_run(&config, &writer, &root, Prune::Keep).await;
    let canonical = std::fs::canonicalize(&root).unwrap();
    let root_text = canonical.to_string_lossy();
    let policy = db
        .newest_document_at_path(&root_text, "policy.md")
        .unwrap()
        .unwrap();

    write_under(&root, "policy.md", "Flood is covered.");
    std::fs::remove_file(root.join("rates/sales.csv")).unwrap();
    let changed = folder_run(&config, &writer, &root, Prune::Keep).await;
    let replaced = changed
        .results
        .iter()
        .find(|r| r.relative == "policy.md")
        .unwrap();
    assert!(
        matches!(&replaced.outcome, Outcome::Replaced { old, .. } if *old == policy.id),
        "{replaced:?}"
    );
    assert_eq!(
        db.document(&policy.id).unwrap().unwrap().status,
        DocumentStatus::Superseded
    );
    let successor = db
        .newest_document_at_path(&root_text, "policy.md")
        .unwrap()
        .unwrap();
    assert_ne!(successor.id, policy.id);
    assert_eq!(
        changed
            .gone
            .iter()
            .map(|d| d.source_path.clone())
            .collect::<Vec<_>>(),
        [Some(String::from("rates/sales.csv"))]
    );
    assert_eq!(changed.pruned, Prune::Keep);
    assert!(db.list_tables().unwrap().contains(&String::from("sales")));

    let pruned = folder_run(&config, &writer, &root, Prune::Delete).await;
    assert_eq!(pruned.gone.len(), 1);
    assert_eq!(pruned.pruned, Prune::Delete);
    assert!(
        db.newest_document_at_path(&root_text, "rates/sales.csv")
            .unwrap()
            .is_none()
    );
    assert!(!db.list_tables().unwrap().contains(&String::from("sales")));
    assert!(
        folder_run(&config, &writer, &root, Prune::Keep)
            .await
            .gone
            .is_empty()
    );
}

/// Two folders fed into one workspace keep to themselves: the same
/// relative path under each is its own document, a run of one folder
/// never reports or prunes the other's documents, and a change replaces
/// only within its root.
#[tokio::test]
async fn folders_sharing_a_relative_path_are_separate_documents() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-folder").unwrap();
    let writer = writer_of(&db);
    let (east, west) = (dir.path().join("east"), dir.path().join("west"));
    write_under(&east, "policy.md", "East: flood is excluded.");
    write_under(&east, "only-east.md", "East only.");
    write_under(&west, "policy.md", "West: flood is covered.");
    let east_run = folder_run(&config, &writer, &east, Prune::Keep).await;
    let west_run = folder_run(&config, &writer, &west, Prune::Keep).await;
    assert_eq!(
        outcome_kinds(&east_run),
        [
            (String::from("only-east.md"), "ingested"),
            (String::from("policy.md"), "ingested"),
        ]
    );
    assert_eq!(
        outcome_kinds(&west_run),
        [(String::from("policy.md"), "ingested")]
    );
    assert!(east_run.gone.is_empty() && west_run.gone.is_empty());
    let root_of = |path: &Path| -> String {
        std::fs::canonicalize(path)
            .unwrap()
            .to_string_lossy()
            .into_owned()
    };
    let (east_root, west_root) = (root_of(&east), root_of(&west));
    let east_policy = db
        .newest_document_at_path(&east_root, "policy.md")
        .unwrap()
        .unwrap();
    let west_policy = db
        .newest_document_at_path(&west_root, "policy.md")
        .unwrap()
        .unwrap();
    assert_ne!(east_policy.id, west_policy.id);
    assert_eq!(db.documents_under(&east_root).unwrap().len(), 2);
    assert_eq!(db.documents_under(&west_root).unwrap().len(), 1);

    // A change in west replaces west's copy only, and a pruning run of
    // west touches nothing of east's.
    write_under(&west, "policy.md", "West: flood is now excluded.");
    let west_again = folder_run(&config, &writer, &west, Prune::Delete).await;
    assert!(
        matches!(
            west_again.results.first().map(|r| &r.outcome),
            Some(Outcome::Replaced { old, .. }) if *old == west_policy.id
        ),
        "{west_again:?}"
    );
    assert!(west_again.gone.is_empty());
    assert_eq!(
        db.document(&east_policy.id).unwrap().unwrap().status,
        DocumentStatus::Ready
    );
    assert_eq!(db.documents_under(&east_root).unwrap().len(), 2);

    // Removing east's extra file and pruning east deletes that one alone.
    std::fs::remove_file(east.join("only-east.md")).unwrap();
    let east_pruned = folder_run(&config, &writer, &east, Prune::Delete).await;
    assert_eq!(
        east_pruned
            .gone
            .iter()
            .map(|d| d.source_path.clone())
            .collect::<Vec<_>>(),
        [Some(String::from("only-east.md"))]
    );
    assert_eq!(db.documents_under(&east_root).unwrap().len(), 1);
    assert_eq!(db.documents_under(&west_root).unwrap().len(), 1);
    assert_eq!(
        db.documents(&DocumentListing::default())
            .unwrap()
            .documents
            .len(),
        2
    );
}

/// `ingest_file` into `ws-replace` without an embedding model, for the
/// replacement tests.
async fn ingest(
    config: &Config,
    writer: &Writer,
    file: ingestion::NewFile<'_>,
) -> Result<ingestion::IngestOutcome, Error> {
    ingestion::ingest_file(
        config,
        writer,
        "ws-replace",
        &file,
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
}

/// A changed file replaces its predecessor: the old document is
/// `superseded` once the new one is ready, leaves search, the listing,
/// and the prompt, keeps its chunks for earlier citations, and hands its
/// pin on. Identical bytes are still a duplicate, and a document that is
/// not ready, or missing, is refused.
#[tokio::test]
async fn a_changed_file_replaces_its_predecessor() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-replace").unwrap();
    let writer = writer_of(&db);
    let ingested = |outcome: ingestion::IngestOutcome| outcome.ingested().unwrap();

    let first = ingested(
        ingest(
            &config,
            &writer,
            ingestion::NewFile::new("policy.md", b"Flood is excluded."),
        )
        .await
        .unwrap(),
    );
    db.set_document_pinning(&first.document_id, Pinning::Pinned)
        .unwrap();
    let second = ingested(
        ingest(
            &config,
            &writer,
            ingestion::NewFile::new("policy.md", b"Flood is covered.")
                .replaces(Some(&first.document_id)),
        )
        .await
        .unwrap(),
    );
    assert_eq!(second.replaced.as_ref(), Some(&first.document_id));
    let old = db.document(&first.document_id).unwrap().unwrap();
    assert_eq!(old.status, DocumentStatus::Superseded);
    assert_eq!(old.superseded_by.as_ref(), Some(&second.document_id));
    let new = db.document(&second.document_id).unwrap().unwrap();
    assert_eq!(
        (new.status, new.pinning),
        (DocumentStatus::Ready, Pinning::Pinned)
    );
    assert_eq!(
        db.documents(&DocumentListing::default())
            .unwrap()
            .documents
            .iter()
            .map(|d| &d.id)
            .collect::<Vec<_>>(),
        [&second.document_id]
    );
    assert_eq!(every_document(&db).len(), 2);
    assert_eq!(db.recent_documents(10).unwrap().1, 1);
    assert_eq!(
        db.document_chunks(&first.document_id, 0, 10).unwrap().len(),
        1
    );
    let hits = db
        .search_keyword_chunks("flood", 5, &ChunkScope::all())
        .unwrap();
    assert_eq!(
        hits.iter().map(|h| &h.document_id).collect::<Vec<_>>(),
        [&second.document_id]
    );
    assert_eq!(db.pinned_documents(Tokens::new(u32::MAX)).unwrap().len(), 1);

    // Identical bytes are a duplicate even as a replacement; a replaced
    // document, or a missing one, cannot be replaced.
    let same = ingest(
        &config,
        &writer,
        ingestion::NewFile::new("policy.md", b"Flood is covered.")
            .replaces(Some(&first.document_id)),
    )
    .await
    .unwrap();
    assert!(matches!(same, ingestion::IngestOutcome::Duplicate(d) if d.id == second.document_id));
    let stale = ingest(
        &config,
        &writer,
        ingestion::NewFile::new("policy.md", b"Flood is excluded again.")
            .replaces(Some(&first.document_id)),
    )
    .await;
    assert!(
        matches!(&stale, Err(Error::Ingestion(m)) if m.contains("it is superseded, not ready")),
        "{stale:?}"
    );
    let missing = ingest(
        &config,
        &writer,
        ingestion::NewFile::new("policy.md", b"Flood is excluded again.")
            .replaces(Some(&DocumentId::from("nope"))),
    )
    .await;
    assert!(
        matches!(missing, Err(Error::NotFound { .. })),
        "{missing:?}"
    );
    assert_eq!(every_document(&db).len(), 2, "nothing registered");
}

/// A table file takes over its predecessor's table; a failed replacement
/// leaves the document and its table as they were; a second replacement
/// of a document whose first is on its way is refused; deleting a
/// replaced table document leaves the table to its successor.
#[tokio::test]
async fn a_table_replacement_swaps_the_table_and_a_failed_one_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-replace").unwrap();
    let writer = writer_of(&db);
    let ingested = |outcome: ingestion::IngestOutcome| outcome.ingested().unwrap();

    let sales = ingested(
        ingest(
            &config,
            &writer,
            ingestion::NewFile::new("sales.csv", b"region,total\nnorth,1\n"),
        )
        .await
        .unwrap(),
    );
    let sales2 = ingested(
        ingest(
            &config,
            &writer,
            ingestion::NewFile::new("sales.csv", b"region,total\nnorth,1\nsouth,2\n")
                .replaces(Some(&sales.document_id)),
        )
        .await
        .unwrap(),
    );
    assert_eq!(sales2.tables, vec![String::from("sales")]);
    let rows = || -> i64 {
        db.connection()
            .query_row("SELECT count(*) FROM sales", [], |r| r.get(0))
            .unwrap()
    };
    assert_eq!(rows(), 2);
    assert_eq!(
        db.table_owner("sales").unwrap().map(|d| d.id),
        Some(sales2.document_id.clone())
    );
    assert_eq!(
        db.document(&sales.document_id).unwrap().unwrap().status,
        DocumentStatus::Superseded
    );

    // A failed replacement leaves the document and its table as they were.
    let broken = ingest(
        &config,
        &writer,
        ingestion::NewFile::new("sales.csv", b"a,b\n1,2,3,4\n\"unterminated,5\n6\n")
            .replaces(Some(&sales2.document_id)),
    )
    .await;
    assert!(matches!(broken, Err(Error::Ingestion(_))), "{broken:?}");
    let kept = db.document(&sales2.document_id).unwrap().unwrap();
    assert_eq!(
        (kept.status, kept.superseded_by),
        (DocumentStatus::Ready, None)
    );
    assert_eq!(rows(), 2);
    let failed = every_document(&db)
        .into_iter()
        .find(|d| d.status == DocumentStatus::Error)
        .unwrap();
    assert_eq!(failed.superseded_by, None);

    // While one replacement is on its way, a second is refused.
    let pending = ingestion::register_document(
        &db,
        &config,
        &ingestion::NewFile::new("sales.csv", b"region,total\nnorth,3\n")
            .replaces(Some(&sales2.document_id)),
    )
    .unwrap();
    assert!(matches!(pending, ingestion::Registration::New(_)));
    let twice = ingestion::register_document(
        &db,
        &config,
        &ingestion::NewFile::new("sales.csv", b"region,total\nnorth,4\n")
            .replaces(Some(&sales2.document_id)),
    );
    assert!(
        matches!(&twice, Err(Error::Ingestion(m)) if m.contains("already being processed")),
        "{twice:?}"
    );

    // Deleting a replaced table document leaves the table to its successor.
    assert!(db.delete_document(&sales.document_id).unwrap());
    assert_eq!(rows(), 2);
    assert!(db.list_tables().unwrap().contains(&String::from("sales")));
}

/// Captions and source code are chunked with locators that reach the
/// stored chunk, the search hit, and the citation label.
#[tokio::test]
async fn captions_and_code_cite_their_locators() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-locators").unwrap();
    let writer = writer_of(&db);
    let vtt = "WEBVTT\n\n00:12:04.000 --> 00:12:06.000\nThe renewal grace period is thirty days.\n\n00:30:00.000 --> 00:30:02.000\nUnrelated closing remarks.\n";
    let code: String = (1..=80)
        .map(|i| format!("fn step_{i}() {{ let renewal_period = {i}; }}"))
        .collect::<Vec<_>>()
        .join("\n");
    for (name, data) in [
        ("meeting.vtt", vtt.as_bytes()),
        ("main.rs", code.as_bytes()),
    ] {
        ingestion::ingest_file(
            &config,
            &writer,
            "ws-locators",
            &ingestion::NewFile::new(name, data),
            None::<&Embedder<MockEmbeddingModel>>,
        )
        .await
        .unwrap()
        .ingested()
        .unwrap();
    }
    let rows = db
        .execute_query(
            "SELECT d.filename, c.kind, c.locator FROM _quack_chunks c \
             JOIN _quack_documents d ON d.id = c.document_id ORDER BY d.filename, c.chunk_index",
        )
        .unwrap();
    let placed: Vec<(String, String, String)> = rows
        .rows
        .iter()
        .map(|r| {
            (
                r.first()
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_owned(),
                r.get(1)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_owned(),
                r.get(2)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_owned(),
            )
        })
        .collect();
    assert!(
        placed
            .iter()
            .any(|(f, k, l)| f == "meeting.vtt" && k == "body" && l == "12:04"),
        "{placed:?}"
    );
    assert!(
        placed
            .iter()
            .any(|(f, _, l)| f == "meeting.vtt" && l == "30:00"),
        "{placed:?}"
    );
    let code_chunks: Vec<&(String, String, String)> =
        placed.iter().filter(|(f, _, _)| f == "main.rs").collect();
    assert!(code_chunks.len() > 1, "{placed:?}");
    assert!(code_chunks.iter().all(|(_, k, _)| k == "code"));
    assert_eq!(
        code_chunks.first().map(|(_, _, l)| l.as_str()),
        Some("line 1")
    );
    assert!(
        code_chunks
            .iter()
            .skip(1)
            .all(|(_, _, l)| l.starts_with("line ") && l != "line 1")
    );

    let hits = db
        .search_keyword_chunks("grace period", 5, &ChunkScope::all())
        .unwrap();
    let hit = hits.first().unwrap();
    assert_eq!(hit.locator.as_deref(), Some("12:04"));
    let label = Citation::new(1, hit).label();
    assert!(
        label.starts_with("meeting.vtt, 12:04, ingested "),
        "{label}"
    );
}

/// A Markdown file's front matter lands on the document row, a table big
/// enough becomes a table of the workspace owned by the document, the
/// uploader's own fields win, and a person's edits replace them.
#[tokio::test]
async fn front_matter_tables_and_edits_land_on_the_document() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = test_config_no_provider(dir.path());
    config.ingestion.table_rows_as_table = 2;
    let db = WorkspaceDb::open(&config, "ws-meta").unwrap();
    let writer = writer_of(&db);
    let md = "---\ntitle: Limits\nauthor: Ada\ndate: 2026-01-05\ntags: [policy, limits]\nowner: claims\n---\n\n# Limits\n\n| Peril | Limit |\n|---|---|\n| Fire | 1000 |\n| Flood | 0 |\n| Wind | 250 |\n\nAfter the table.\n";
    let result = ingestion::ingest_file(
        &config,
        &writer,
        "ws-meta",
        &ingestion::NewFile::new("notes.md", md.as_bytes()),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    assert_eq!(result.tables, ["notes_table1"]);
    let count = db
        .execute_query("SELECT count(*) FROM notes_table1 WHERE \"Limit\" > 0")
        .unwrap();
    assert_eq!(
        count
            .rows
            .first()
            .and_then(|r| r.first())
            .and_then(serde_json::Value::as_i64),
        Some(2)
    );
    let doc = db.document(&result.document_id).unwrap().unwrap();
    assert_eq!(doc.title.as_deref(), Some("Limits"));
    assert_eq!(doc.author.as_deref(), Some("Ada"));
    assert_eq!(doc.authored_at.as_deref(), Some("2026-01-05 00:00:00"));
    assert_eq!(doc.tags, ["policy", "limits"]);
    assert_eq!(
        doc.metadata.get("owner").map(String::as_str),
        Some("claims")
    );
    assert_eq!(
        doc.tables.as_deref(),
        Some(&[String::from("notes_table1")][..])
    );
    let kinds = db
        .execute_query("SELECT kind FROM _quack_chunks ORDER BY chunk_index")
        .unwrap();
    let kinds: Vec<&str> = kinds
        .rows
        .iter()
        .filter_map(|r| r.first().and_then(|v| v.as_str()))
        .collect();
    assert_eq!(kinds, ["table", "body"]);
}

/// The uploader's fields win over the file's own, and a person's edits
/// replace them: an empty text clears, a bad date is refused, a missing
/// document is not found.
#[tokio::test]
async fn the_uploaders_fields_win_and_a_person_edits_them() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-meta-edits").unwrap();
    let writer = writer_of(&db);
    let second = ingestion::ingest_file(
        &config,
        &writer,
        "ws-meta-edits",
        &ingestion::NewFile::new("again.md", b"---\nauthor: Ada\n---\n\ntext\n").fields(
            DocumentFields {
                author: Some(String::from("Grace")),
                tags: Some(vec![String::from("given")]),
                ..DocumentFields::default()
            },
        ),
        None::<&Embedder<MockEmbeddingModel>>,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    let doc = db.document(&second.document_id).unwrap().unwrap();
    assert_eq!(doc.author.as_deref(), Some("Grace"));
    assert_eq!(doc.tags, ["given"]);
    db.set_document_fields(
        &second.document_id,
        &DocumentFields {
            title: Some(String::from("Edited")),
            author: Some(String::new()),
            authored_at: Some(String::from("2026-02-01")),
            tags: Some(vec![String::from(" one "), String::new()]),
        },
    )
    .unwrap();
    let doc = db.document(&second.document_id).unwrap().unwrap();
    assert_eq!(doc.title.as_deref(), Some("Edited"));
    assert_eq!(doc.author, None);
    assert_eq!(doc.authored_at.as_deref(), Some("2026-02-01 00:00:00"));
    assert_eq!(doc.tags, ["one"]);
    let bad = db
        .set_document_fields(
            &second.document_id,
            &DocumentFields {
                authored_at: Some(String::from("yesterday")),
                ..DocumentFields::default()
            },
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(bad.contains("is not a date"), "{bad}");
    assert!(
        db.set_document_fields(&DocumentId::from("absent"), &DocumentFields::default())
            .is_err()
    );
}

/// A loaded table is profiled at once; `--types` retypes its columns
/// strictly, and a reserved name is refused before anything loads.
#[tokio::test]
async fn a_loaded_table_is_profiled_retyped_and_reserved_names_refused() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-replace").unwrap();
    let writer = writer_of(&db);
    let csv = b"id,amount,placed\n1,100,2026-01-02\n2,250,2026-01-03\n";

    ingest(&config, &writer, ingestion::NewFile::new("orders.csv", csv))
        .await
        .unwrap();
    let profile = TableProfile::current(&db, "orders", 2).unwrap().unwrap();
    assert_eq!(profile.column("amount").unwrap().distinct, 2);

    let typed = b"code,amount,placed\nA,100,2026-01-02\nB,250,2026-01-03\n";
    ingest(
        &config,
        &writer,
        ingestion::NewFile::new("typed.csv", typed)
            .types("amount=DOUBLE,placed=VARCHAR".parse().unwrap()),
    )
    .await
    .unwrap();
    let columns = db.describe_columns("typed").unwrap();
    let kind = |name: &str| {
        columns
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.column_type.clone())
    };
    assert_eq!(kind("amount").as_deref(), Some("DOUBLE"));
    assert_eq!(kind("placed").as_deref(), Some("VARCHAR"));
    assert_eq!(
        TableProfile::current(&db, "typed", 2)
            .unwrap()
            .unwrap()
            .column("amount")
            .unwrap()
            .duckdb_type,
        "DOUBLE"
    );

    let wrong = ingest(
        &config,
        &writer,
        ingestion::NewFile::new("wrong.csv", b"code\nA\n").types("code=BIGINT".parse().unwrap()),
    )
    .await;
    assert!(wrong.is_err_and(|e| e.to_string().contains("does not convert")));
    let missing = ingest(
        &config,
        &writer,
        ingestion::NewFile::new("missing.csv", b"code\nA\n").types("ghost=BIGINT".parse().unwrap()),
    )
    .await;
    assert!(missing.is_err_and(|e| e.to_string().contains("'ghost'")));

    for name in ["graph_vendors.csv", "_quack_meta.csv"] {
        let refused = ingest(&config, &writer, ingestion::NewFile::new(name, b"a\n1\n")).await;
        assert!(
            refused.is_err_and(|e| e.to_string().contains("reserves")),
            "{name}"
        );
    }
    assert!(
        !db.list_tables()
            .unwrap()
            .iter()
            .any(|t| t.starts_with("graph_v"))
    );
}

/// A workspace from before profiles gets every table profiled on open.
#[test]
fn an_older_workspace_is_profiled_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    {
        let db = WorkspaceDb::open(&config, "ws-old").unwrap();
        db.execute_statement("CREATE TABLE t AS SELECT range AS n FROM range(4)")
            .unwrap();
        db.execute_statement("DELETE FROM _quack_table_profiles")
            .unwrap();
        db.execute_statement("UPDATE _quack_meta SET value = '11' WHERE key = 'schema_version'")
            .unwrap();
    }
    let db = WorkspaceDb::open(&config, "ws-old").unwrap();
    assert!(TableProfile::current(&db, "t", 4).unwrap().is_some());
}

/// Every document row, replaced ones included.
fn every_document(db: &WorkspaceDb) -> Vec<DocumentInfo> {
    db.documents(&DocumentListing {
        shown: Shown::All,
        limit: DocumentListing::MAX_PAGE,
        ..DocumentListing::default()
    })
    .unwrap()
    .documents
}
