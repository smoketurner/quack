#![expect(clippy::unwrap_used, reason = "test assertions use unwrap throughout")]

use std::collections::BTreeMap;
use std::path::Path;

use quack_core::config::{
    AnalysisConfig, AuthMode, Config, ContextConfig, GeneralConfig, GraphConfig, ImportConfig,
    IngestionConfig, OntologyConfig, ProviderConfig, ProviderType, RetrievalConfig, ServerConfig,
};
use quack_core::ingestion;
use quack_core::ingestion::parser::FileType;
use quack_core::storage::control::ControlPlane;
use quack_core::storage::workspace::{
    ChunkScope, DocumentSource, NewChunk, NewDocument, StatementKind, WorkspaceDb,
};
use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};

const TEST_DIM: usize = 4;
const TEST_DIM_U32: u32 = 4;

struct MockEmbeddingModel {
    dim: usize,
}

/// Records the size of every batch it is asked to embed.
struct BatchRecordingModel {
    batches: std::sync::Mutex<Vec<usize>>,
}

impl EmbeddingModel for BatchRecordingModel {
    const MAX_DOCUMENTS: usize = 1024;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
        Self {
            batches: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn ndims(&self) -> usize {
        TEST_DIM
    }

    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + Send {
        let texts: Vec<String> = texts.into_iter().collect();
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
    const MAX_DOCUMENTS: usize = 1024;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
        Self
    }

    fn ndims(&self) -> usize {
        TEST_DIM
    }

    fn embed_texts(
        &self,
        _texts: impl IntoIterator<Item = String> + Send,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + Send {
        std::future::ready(Err(EmbeddingError::ProviderError(String::from(
            "connection refused",
        ))))
    }
}

impl EmbeddingModel for MockEmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
        Self { dim: TEST_DIM }
    }

    fn ndims(&self) -> usize {
        self.dim
    }

    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + Send {
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

fn test_config(data_dir: &Path) -> Config {
    let mut providers = BTreeMap::new();
    providers.insert(
        "mock".into(),
        ProviderConfig {
            provider_type: ProviderType::Ollama,
            auth: AuthMode::None,
            base_url: Some("http://localhost:9999".into()),
            api_key_env: None,
            embedding_dimension: Some(TEST_DIM_U32),
            max_concurrent_requests: None,
            oauth: None,
        },
    );
    Config {
        general: GeneralConfig {
            data_dir: data_dir.to_path_buf(),
            default_workspace: "test".into(),
            chat_model: None,
            embedding_model: Some("mock/mock-model".into()),
        },
        providers,
        ingestion: IngestionConfig {
            chunk_size_tokens: 50,
            chunk_overlap_tokens: 10,
            embedding_batch_size: 64,
            embedding_concurrency: 2,
            tokenizer_encoding: String::from("cl100k_base"),
            upload_max_mb: 512,
        },
        retrieval: RetrievalConfig::default(),
        context: ContextConfig::default(),
        analysis: AnalysisConfig::default(),
        server: ServerConfig::default(),
        ontology: OntologyConfig::default(),
        graph: GraphConfig::default(),
        import: ImportConfig::default(),
        jobs: quack_core::config::JobsConfig::default(),
    }
}

fn test_config_no_provider(data_dir: &Path) -> Config {
    Config {
        general: GeneralConfig {
            data_dir: data_dir.to_path_buf(),
            default_workspace: "test".into(),
            chat_model: None,
            embedding_model: None,
        },
        providers: BTreeMap::new(),
        ingestion: IngestionConfig::default(),
        retrieval: RetrievalConfig::default(),
        context: ContextConfig::default(),
        analysis: AnalysisConfig::default(),
        server: ServerConfig::default(),
        ontology: OntologyConfig::default(),
        graph: GraphConfig::default(),
        import: ImportConfig::default(),
        jobs: quack_core::config::JobsConfig::default(),
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

    let data = b"Hello world. This is a test document for ingestion testing.";
    let result = ingestion::ingest_file::<MockEmbeddingModel, _>(
        &config,
        &db,
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
    let model = MockEmbeddingModel { dim: TEST_DIM };

    let data = b"This is a longer document with enough words to produce at least one chunk. \
                 We need to make sure the embedding pipeline works end to end with our mock.";
    let result = ingestion::ingest_file(
        &config,
        &db,
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
    const MAX_DOCUMENTS: usize = 1024;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
        Self {
            in_flight: std::sync::atomic::AtomicUsize::new(0),
            peak: std::sync::atomic::AtomicUsize::new(0),
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn ndims(&self) -> usize {
        TEST_DIM
    }

    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + Send {
        use std::sync::atomic::Ordering;
        let texts: Vec<String> = texts.into_iter().collect();
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
    model: &InFlightModel,
) -> (WorkspaceDb, ingestion::IngestResult) {
    let db = WorkspaceDb::open(config, workspace_id).unwrap();
    let sections: Vec<String> = (1..=6)
        .map(|i| format!("# Section {i}\n\nA short paragraph about topic number {i}.\n"))
        .collect();
    let data = sections.join("\n");
    let result = ingestion::ingest_file(
        config,
        &db,
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
    let model = InFlightModel::make(&(), "mock", None);

    let (db, result) = ingest_six_sections(&config, "ws-concurrent", &model).await;

    assert_eq!(result.chunks_stored, 6);
    assert!(result.embedding_time.is_some());
    assert_eq!(model.calls.load(Ordering::SeqCst), 6);
    let peak = model.peak.load(Ordering::SeqCst);
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
    let model = InFlightModel::make(&(), "mock", None);

    let (db, _) = ingest_six_sections(&config, "ws-serial", &model).await;

    assert_eq!(model.peak.load(Ordering::SeqCst), 1);
    assert_eq!(mismatched_vectors(&db), serde_json::Value::Number(0.into()));
}

#[tokio::test]
async fn embedding_batch_size_bounds_every_embed_request() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = test_config(dir.path());
    config.ingestion.embedding_batch_size = 2;
    let workspace_id = "ws-batch";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();
    let model = BatchRecordingModel::make(&(), "mock", None);

    // Five headed sections, each its own chunk at 50 tokens.
    let sections: Vec<String> = (1..=5)
        .map(|i| format!("# Section {i}\n\nA short paragraph about topic number {i}.\n"))
        .collect();
    let data = sections.join("\n");
    let result = ingestion::ingest_file(
        &config,
        &db,
        workspace_id,
        &ingestion::NewFile::new("batches.md", data.as_bytes()),
        Some(&model),
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();

    let batches = model.batches.lock().unwrap().clone();
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

    let result = ingestion::ingest_file::<MockEmbeddingModel, _>(
        &config,
        &db,
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

    let result = ingestion::ingest_file::<MockEmbeddingModel, _>(
        &config,
        &db,
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

    let result = ingestion::ingest_file::<MockEmbeddingModel, _>(
        &config,
        &db,
        workspace_id,
        &ingestion::NewFile::new("image.png", b"fake image data"),
        None,
    )
    .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, quack_core::error::Error::UnsupportedFileType(_)),
        "expected UnsupportedFileType, got: {err}"
    );
}

#[tokio::test]
async fn ingest_empty_text_file() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-empty";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();

    let result = ingestion::ingest_file::<MockEmbeddingModel, _>(
        &config,
        &db,
        workspace_id,
        &ingestion::NewFile::new("empty.txt", b""),
        None,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();

    assert_eq!(result.file_type, FileType::Text);
    assert_eq!(result.chunks_stored, 0);
}

#[tokio::test]
async fn ingest_markdown_as_unstructured() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-md";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();

    let data = b"# Heading\n\nSome paragraph text.\n\n- item 1\n- item 2\n";
    let result = ingestion::ingest_file::<MockEmbeddingModel, _>(
        &config,
        &db,
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
        &NewDocument::new("doc-1", "test.txt", "text/plain", 100).with_status("pending"),
    )
    .unwrap();

    let qr = db
        .execute_query("SELECT id, filename, status FROM _quack_documents WHERE id = 'doc-1'")
        .unwrap();
    assert_eq!(qr.rows.len(), 1);

    db.update_document_status("doc-1", "ready").unwrap();

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
        &NewDocument::new("doc-1", "test.txt", "text/plain", 100).with_status("ready"),
    )
    .unwrap();

    db.insert_chunk(&NewChunk {
        id: "c1",
        document_id: "doc-1",
        chunk_index: 0,
        content: "hello world",
        heading: None,
        page: None,
        embedding: None,
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
        &NewDocument::new("doc-1", "test.txt", "text/plain", 100).with_status("ready"),
    )
    .unwrap();

    let embedding = [0.5_f32, 0.3, -0.2, 0.8];
    db.insert_chunk(&NewChunk {
        id: "c1",
        document_id: "doc-1",
        chunk_index: 0,
        content: "embedded chunk",
        heading: None,
        page: None,
        embedding: Some(&embedding),
    })
    .unwrap();

    let qr = db
        .execute_query("SELECT content FROM _quack_chunks WHERE embedding IS NOT NULL")
        .unwrap();
    assert_eq!(qr.rows.len(), 1);
}

#[test]
fn workspace_db_update_chunk_embedding() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-update-emb";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();
    db.insert_document(
        &NewDocument::new("doc-1", "test.txt", "text/plain", 100).with_status("ready"),
    )
    .unwrap();

    db.insert_chunk(&NewChunk {
        id: "c1",
        document_id: "doc-1",
        chunk_index: 0,
        content: "hello world",
        heading: None,
        page: None,
        embedding: None,
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
    let embedding = [1.0_f32, 0.0, 0.0, 0.0];
    db.update_chunk_embedding("doc-1", 0, &embedding).unwrap();

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
        &NewDocument::new("doc-a", "policy.pdf", "application/pdf", 10).with_status("ready"),
    )
    .unwrap();
    db.insert_document(
        &NewDocument::new("doc-b", "faq.md", "text/markdown", 10).with_status("ready"),
    )
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: "a0",
        document_id: "doc-a",
        chunk_index: 0,
        content: "flood exclusion",
        heading: None,
        page: None,
        embedding: Some(&[1.0, 0.0, 0.0, 0.0]),
    })
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: "b0",
        document_id: "doc-b",
        chunk_index: 0,
        content: "claims timeline",
        heading: None,
        page: None,
        embedding: Some(&[0.9, 0.1, 0.0, 0.0]),
    })
    .unwrap();

    let query = [1.0_f32, 0.0, 0.0, 0.0];

    let all = db
        .search_similar_chunks(&query, 5, &ChunkScope::all())
        .unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all.first().unwrap().filename, "policy.pdf");
    assert_eq!(all.last().unwrap().filename, "faq.md");

    let only_b = db
        .search_similar_chunks(&query, 5, &ChunkScope::documents([String::from("doc-b")]))
        .unwrap();
    assert_eq!(only_b.len(), 1);
    assert_eq!(only_b.first().unwrap().document_id, "doc-b");
    assert_eq!(only_b.first().unwrap().filename, "faq.md");

    let none = db
        .search_similar_chunks(&query, 5, &ChunkScope::documents([String::from("missing")]))
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
        &NewDocument::new("doc-1", "test.txt", "text/plain", 100).with_status("ready"),
    )
    .unwrap();

    db.insert_chunk(&NewChunk {
        id: "c1",
        document_id: "doc-1",
        chunk_index: 0,
        content: "first chunk",
        heading: None,
        page: None,
        embedding: Some(&[1.0, 0.0, 0.0, 0.0]),
    })
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: "c2",
        document_id: "doc-1",
        chunk_index: 1,
        content: "second chunk",
        heading: None,
        page: None,
        embedding: Some(&[0.0, 1.0, 0.0, 0.0]),
    })
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: "c3",
        document_id: "doc-1",
        chunk_index: 2,
        content: "third chunk",
        heading: None,
        page: None,
        embedding: Some(&[0.7, 0.7, 0.0, 0.0]),
    })
    .unwrap();

    let query = [1.0_f32, 0.0, 0.0, 0.0];
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
    let db = WorkspaceDb::open_in_memory(TEST_DIM_U32).unwrap();
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
    let db = WorkspaceDb::open_in_memory(TEST_DIM_U32).unwrap();
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
    let db = WorkspaceDb::open_in_memory(TEST_DIM_U32).unwrap();
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
    let db = WorkspaceDb::open_in_memory(TEST_DIM_U32).unwrap();
    assert!(matches!(kind(&db, "SELEC 1"), StatementKind::Invalid(_)));
    assert!(matches!(kind(&db, ""), StatementKind::Invalid(_)));
}

#[test]
fn internal_tables_are_detected_in_parsed_and_unparsed_statements() {
    let db = WorkspaceDb::open_in_memory(TEST_DIM_U32).unwrap();
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
    let db = WorkspaceDb::open_in_memory(TEST_DIM_U32)
        .unwrap()
        .with_query_timeout(std::time::Duration::from_millis(200));
    let started = std::time::Instant::now();
    let result = db.execute_query(
        "SELECT count(*) FROM range(100000000) a, range(100000000) b WHERE a.range = b.range + 1",
    );
    let elapsed = started.elapsed();
    let err = result.err().unwrap();
    assert!(
        err.to_string().to_lowercase().contains("interrupt"),
        "unexpected error: {err}"
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
    assert_eq!(db.meta("schema_version").unwrap().as_deref(), Some("7"));
    assert_eq!(
        db.meta("embedding_dimension").unwrap().as_deref(),
        Some("4")
    );
    assert_eq!(
        db.meta("embedding_model").unwrap().as_deref(),
        Some("mock-model")
    );
    assert_eq!(db.embedding_dimension(), TEST_DIM_U32);
    assert!(db.list_tables().unwrap().is_empty());
}

#[test]
fn reopen_without_provider_keeps_recorded_dimension() {
    let dir = tempfile::tempdir().unwrap();
    let with = test_config(dir.path());
    drop(WorkspaceDb::open(&with, "ws-dim").unwrap());

    let without = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&without, "ws-dim").unwrap();
    assert_eq!(db.embedding_dimension(), TEST_DIM_U32);
}

#[test]
fn dimension_change_with_stored_embeddings_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    {
        let db = WorkspaceDb::open(&config, "ws-mismatch").unwrap();
        db.insert_document(&NewDocument::new("d", "a.txt", "text/plain", 1).with_status("ready"))
            .unwrap();
        db.insert_chunk(&NewChunk {
            id: "c",
            document_id: "d",
            chunk_index: 0,
            content: "x",
            heading: None,
            page: None,
            embedding: Some(&[1.0, 0.0, 0.0, 0.0]),
        })
        .unwrap();
    }
    let mut changed = test_config(dir.path());
    if let Some(p) = changed.providers.get_mut("mock") {
        p.embedding_dimension = Some(8);
    }
    changed.general.embedding_model = Some("mock/other-model".into());
    let err = WorkspaceDb::open(&changed, "ws-mismatch").err().unwrap();
    let msg = err.to_string();
    assert!(
        msg.contains("4-dimensional") && msg.contains("8-dimensional"),
        "{msg}"
    );
    assert!(
        msg.contains("mock-model") && msg.contains("other-model"),
        "{msg}"
    );
}

#[test]
fn dimension_change_without_embeddings_adopts_new_width() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    {
        let db = WorkspaceDb::open(&config, "ws-adopt").unwrap();
        db.insert_document(&NewDocument::new("d", "a.txt", "text/plain", 1).with_status("ready"))
            .unwrap();
        db.insert_chunk(&NewChunk {
            id: "c",
            document_id: "d",
            chunk_index: 0,
            content: "x",
            heading: None,
            page: None,
            embedding: None,
        })
        .unwrap();
    }
    {
        // A node label embedding of the old width: cleared on reopen and
        // recomputed by the next resolution pass.
        let db = WorkspaceDb::open(&config, "ws-adopt").unwrap();
        let node = quack_core::graph::store::upsert_node(
            &db,
            &quack_core::graph::store::NewNode {
                label: String::from("Kenya"),
                class_id: String::from("country"),
                properties: serde_json::json!({}),
                provisional: false,
            },
        )
        .unwrap();
        quack_core::graph::store::set_node_embedding(&db, &node, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(
            quack_core::graph::store::nodes_without_embedding(&db, 10)
                .unwrap()
                .is_empty()
        );
    }
    let mut changed = test_config(dir.path());
    if let Some(p) = changed.providers.get_mut("mock") {
        p.embedding_dimension = Some(8);
    }
    let db = WorkspaceDb::open(&changed, "ws-adopt").unwrap();
    assert_eq!(db.embedding_dimension(), 8);
    let unembedded = quack_core::graph::store::nodes_without_embedding(&db, 10).unwrap();
    assert_eq!(unembedded.len(), 1);
    let node_id = unembedded.first().map(|n| n.id.clone()).unwrap();
    quack_core::graph::store::set_node_embedding(&db, &node_id, &[0.5; 8]).unwrap();
    assert_eq!(
        db.meta("embedding_dimension").unwrap().as_deref(),
        Some("8")
    );
    // The chunk ingested without an embedding survives, term index and
    // all, and the new width takes embeddings.
    let kept = db.chunks_by_ids(&[String::from("c")]).unwrap();
    assert_eq!(kept.len(), 1);
    assert!(
        !db.search_keyword_chunks("x", 5, &ChunkScope::all())
            .unwrap()
            .is_empty()
    );
    db.insert_chunk(&NewChunk {
        id: "c8",
        document_id: "d",
        chunk_index: 1,
        content: "y",
        heading: None,
        page: None,
        embedding: Some(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    })
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: "c2",
        document_id: "d",
        chunk_index: 1,
        content: "y",
        heading: None,
        page: None,
        embedding: Some(&[0.5; 8]),
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
    let docs = db.list_documents().unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs.first().unwrap().filename, "old.txt");
    assert!(db.list_tables().unwrap().is_empty());
}

#[tokio::test]
async fn control_db_migrates_to_the_latest_version_and_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let control = ControlPlane::open(&config).await.unwrap();
    assert_eq!(control.schema_version().await.unwrap(), 3);
    // Reopening is a no-op.
    let again = ControlPlane::open(&config).await.unwrap();
    assert_eq!(again.schema_version().await.unwrap(), 3);
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

    let filename = "it's a file.csv";
    let files_dir = config.workspace_files_dir(workspace_id);
    std::fs::create_dir_all(&files_dir).unwrap();
    std::fs::write(files_dir.join(filename), "a,b\n1,2\n3,4\n").unwrap();

    let result = ingestion::ingest_file(
        &config,
        &db,
        workspace_id,
        &ingestion::NewFile::new(filename, b"a,b\n1,2\n3,4\n"),
        None::<&MockEmbeddingModel>,
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
    let db = WorkspaceDb::open_in_memory(TEST_DIM_U32).unwrap();
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
        &NewDocument::new("doc-a", "policy.pdf", "application/pdf", 10).with_status("ready"),
    )
    .unwrap();
    db.insert_document(
        &NewDocument::new("doc-b", "faq.md", "text/markdown", 10).with_status("ready"),
    )
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: "a0",
        document_id: "doc-a",
        chunk_index: 0,
        content: "Flood damage is excluded from coverage.",
        heading: Some("Exclusions"),
        page: Some(12),
        embedding: Some(&[1.0, 0.0, 0.0, 0.0]),
    })
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: "b0",
        document_id: "doc-b",
        chunk_index: 0,
        content: "Policy POL-8841 renews every March.",
        heading: None,
        page: None,
        embedding: Some(&[0.0, 1.0, 0.0, 0.0]),
    })
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: "b1",
        document_id: "doc-b",
        chunk_index: 1,
        content: "Claims close within thirty days of filing.",
        heading: Some("Claims"),
        page: None,
        embedding: Some(&[0.0, 0.0, 1.0, 0.0]),
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
        .search_similar_chunks(&[1.0, 0.0, 0.0, 0.0], 1, &ChunkScope::all())
        .unwrap();
    let top = hits.first().unwrap();
    assert_eq!(top.id, "a0");
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

    let only_a0 = ChunkScope::all().and_chunks([String::from("a0")]);
    let keyword = db.search_keyword_chunks("flood", 5, &only_a0).unwrap();
    assert_eq!(
        keyword.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["a0"]
    );
    let vector = db
        .search_similar_chunks(&[0.0, 0.0, 1.0, 0.0], 5, &only_a0)
        .unwrap();
    assert_eq!(
        vector.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["a0"],
        "the closer chunk b1 is outside the scope"
    );
    let hybrid = db
        .search_hybrid_chunks("claims", &[0.0, 0.0, 1.0, 0.0], 5, 60, &only_a0)
        .unwrap();
    assert!(hybrid.iter().all(|h| h.id == "a0"), "{hybrid:?}");

    // Both filters at once, contradicting each other.
    let crossed = ChunkScope::documents([String::from("doc-b")]).and_chunks([String::from("a0")]);
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
        db.search_similar_chunks(&[1.0, 0.0, 0.0, 0.0], 5, &nothing)
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
        .search_keyword_chunks("flood", 5, &ChunkScope::documents([String::from("doc-b")]))
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
        db.insert_document(&NewDocument::new("d", "a.md", "text/markdown", 1).with_status("ready"))
            .unwrap();
        db.insert_chunk(&NewChunk {
            id: "c0",
            document_id: "d",
            chunk_index: 0,
            content: "renewal POL-8841 notice",
            heading: None,
            page: None,
            embedding: None,
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
    assert_eq!(db.meta("schema_version").unwrap().as_deref(), Some("7"));
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
        .search_hybrid_chunks("POL-8841", &[1.0, 0.0, 0.0, 0.0], 3, 60, &ChunkScope::all())
        .unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids.len(), 3);
    // b0 ranks first in keyword and second in vector, so it wins the fusion.
    assert_eq!(ids.first().copied(), Some("b0"), "{ids:?}");
    assert!(ids.contains(&"a0"));
    let top_score = hits.first().unwrap().score;
    assert!(hits.iter().all(|h| h.score <= top_score));

    let limited = db
        .search_hybrid_chunks("flood", &[1.0, 0.0, 0.0, 0.0], 1, 60, &ChunkScope::all())
        .unwrap();
    assert_eq!(limited.len(), 1);
    assert_eq!(limited.first().unwrap().id, "a0");
}

#[tokio::test]
async fn ingest_markdown_stores_headings_and_pinned_flag() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-md-meta").unwrap();
    let md = b"# Exclusions\n\nFlood is excluded.\n\n# Claims\n\nClose in thirty days.\n";
    let result = ingestion::ingest_file(
        &config,
        &db,
        "ws-md-meta",
        &ingestion::NewFile::new("rules.md", md),
        None::<&MockEmbeddingModel>,
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

    let doc = db.list_documents().unwrap().into_iter().next().unwrap();
    assert!(!doc.pinned);
    db.set_document_pinned(&doc.id, true).unwrap();
    let pinned = db.pinned_documents().unwrap();
    assert_eq!(pinned.len(), 1);
    assert!(pinned.first().unwrap().1.contains("Flood is excluded."));
}

#[tokio::test]
async fn identical_bytes_are_skipped_and_a_failed_document_is_retried() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-dedup").unwrap();
    let md = b"# Renewal terms\n\nRenewals close in thirty days.\n";
    let first = ingestion::ingest_file(
        &config,
        &db,
        "ws-dedup",
        &ingestion::NewFile::new("terms.md", md).source(DocumentSource::Stdin),
        None::<&MockEmbeddingModel>,
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
        &db,
        "ws-dedup",
        &ingestion::NewFile::new("copy.md", md).title(Some("Copy")),
        None::<&MockEmbeddingModel>,
    )
    .await
    .unwrap();
    let existing = match again {
        ingestion::IngestOutcome::Duplicate(existing) => Some(existing),
        ingestion::IngestOutcome::Ingested(_) => None,
    };
    assert_eq!(existing.map(|d| d.id), Some(first.document_id.clone()));
    assert_eq!(db.list_documents().unwrap().len(), 1);

    // An explicit title wins over the parsed heading.
    let titled = ingestion::ingest_file(
        &config,
        &db,
        "ws-dedup",
        &ingestion::NewFile::new("other.md", b"# Heading\n\nBody.\n").title(Some(" Given ")),
        None::<&MockEmbeddingModel>,
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
        &db,
        "ws-dedup",
        &ingestion::NewFile::new("scan.pdf", bad),
        None::<&MockEmbeddingModel>,
    )
    .await;
    assert!(failed.is_err());
    let errored = db
        .list_documents()
        .unwrap()
        .into_iter()
        .find(|d| d.filename == "scan.pdf")
        .unwrap();
    assert_eq!(errored.status, "error");
    let retry =
        ingestion::register_document(&db, &ingestion::NewFile::new("scan.pdf", bad)).unwrap();
    assert!(matches!(retry, ingestion::Registration::New(_)));
}

#[tokio::test]
async fn a_long_pdf_ingests_every_page_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-long-pdf").unwrap();

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
        &db,
        "ws-long-pdf",
        &ingestion::NewFile::new("report.pdf", &bytes),
        None::<&MockEmbeddingModel>,
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    assert_eq!(result.pages_skipped, 0);
    assert!(result.chunks_stored > 0);

    let doc = db.document(&result.document_id).unwrap().unwrap();
    assert_eq!(doc.status, "ready");
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
        ingestion::load_stdin_table(&config, &db, "ws-stdin", csv)
            .unwrap()
            .as_deref(),
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
    let bytes = tiny_xlsx();
    let result = ingestion::ingest_file(
        &config,
        &db,
        "ws-xlsx",
        &ingestion::NewFile::new("Region Sales.xlsx", &bytes),
        None::<&MockEmbeddingModel>,
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
    assert!(db.delete_document(&result.document_id, None).unwrap());
    assert!(db.list_tables().unwrap().is_empty());
    assert!(db.document(&result.document_id).unwrap().is_none());
    // The workbook and its per-sheet CSVs are gone from files/ too.
    assert_eq!(std::fs::read_dir(&files_dir).unwrap().count(), 0);
}

#[tokio::test]
async fn office_and_html_documents_are_chunked_with_titles() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-office").unwrap();
    let page = b"<html><head><title>Renewal Guide</title></head><body><h1>Terms</h1><p>Thirty days.</p></body></html>";
    let result = ingestion::ingest_file(
        &config,
        &db,
        "ws-office",
        &ingestion::NewFile::new("guide.html", page),
        None::<&MockEmbeddingModel>,
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
        &db,
        "ws-office",
        &ingestion::NewFile::new("deck.pptx", b"not a package"),
        None::<&MockEmbeddingModel>,
    )
    .await;
    assert!(failed.is_err());
    let errored = db
        .list_documents()
        .unwrap()
        .into_iter()
        .find(|d| d.filename == "deck.pptx")
        .unwrap();
    assert!(
        errored
            .error_message
            .is_some_and(|m| m.contains("not a PPTX"))
    );
}

#[tokio::test]
async fn sqlite_sources_import_as_tables_with_every_column_as_text_then_sniffed() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-import").unwrap();
    // Beside the data directory, not inside it: quack's own files are
    // refused as a source (issue #69).
    let source_dir = tempfile::tempdir().unwrap();
    let source_path = source_dir.path().join("source.db");
    {
        use sqlx::Connection as _;
        use sqlx::Executor as _;
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
    let request = quack_core::import::ImportRequest {
        url: url.clone(),
        table: String::from("Orders Import"),
        query: None,
        source_table: Some(String::from("orders")),
        limit: None,
    };
    let summary = quack_core::import::import(
        &config,
        &db,
        "ws-import",
        &request,
        quack_core::import::ImportPolicy::owner(),
        None::<&MockEmbeddingModel>,
        None,
    )
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
    let request = quack_core::import::ImportRequest {
        url: url.clone(),
        table: String::from("big"),
        query: Some(String::from(
            "SELECT region, total * 2 AS doubled FROM orders ORDER BY id",
        )),
        source_table: None,
        limit: Some(2),
    };
    let summary = quack_core::import::import(
        &config,
        &db,
        "ws-import",
        &request,
        quack_core::import::ImportPolicy::owner(),
        None::<&MockEmbeddingModel>,
        None,
    )
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
    db.insert_document(&NewDocument::new("d", "a.md", "text/markdown", 1).with_status("ready"))
        .unwrap();
    db.insert_chunk(&NewChunk {
        id: "c",
        document_id: "d",
        chunk_index: 0,
        content: "The null hypothesis was rejected.",
        heading: None,
        page: None,
        embedding: None,
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
    db.insert_document(&NewDocument::new("d", "a.md", "text/markdown", 1).with_status("error"))
        .unwrap();
    db.insert_chunk(&NewChunk {
        id: "c",
        document_id: "d",
        chunk_index: 0,
        content: "zebra crossing",
        heading: None,
        page: None,
        embedding: Some(&[1.0, 0.0, 0.0, 0.0]),
    })
    .unwrap();
    assert!(
        db.search_keyword_chunks("zebra", 5, &ChunkScope::all())
            .unwrap()
            .is_empty()
    );
    assert!(
        db.search_similar_chunks(&[1.0, 0.0, 0.0, 0.0], 5, &ChunkScope::all())
            .unwrap()
            .is_empty()
    );
    db.update_document_status("d", "ready").unwrap();
    assert_eq!(
        db.search_keyword_chunks("zebra", 5, &ChunkScope::all())
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        db.search_similar_chunks(&[1.0, 0.0, 0.0, 0.0], 5, &ChunkScope::all())
            .unwrap()
            .len(),
        1
    );

    // Embedding fails after the chunks were written: the row is an error
    // and nothing of it stays searchable or counted.
    let failed = ingestion::ingest_file(
        &config,
        &db,
        "ws-failed",
        &ingestion::NewFile::new("notes.md", b"# Notes\n\nA giraffe walked by.\n"),
        Some(&FailingEmbeddingModel),
    )
    .await;
    assert!(failed.is_err());
    let doc = db
        .list_documents()
        .unwrap()
        .into_iter()
        .find(|d| d.filename == "notes.md")
        .unwrap();
    assert_eq!(doc.status, "error");
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
    let first_bytes = b"region,total\nnorth,1\n";
    let ingested = |outcome: ingestion::IngestOutcome| match outcome {
        ingestion::IngestOutcome::Ingested(r) => Some(r),
        ingestion::IngestOutcome::Duplicate(_) => None,
    };
    let first = ingested(
        ingestion::ingest_file(
            &config,
            &db,
            "ws-owner",
            &ingestion::NewFile::new("sales.csv", first_bytes),
            None::<&MockEmbeddingModel>,
        )
        .await
        .unwrap(),
    )
    .unwrap();
    assert_eq!(first.tables, vec![String::from("sales")]);

    let changed = ingestion::ingest_file(
        &config,
        &db,
        "ws-owner",
        &ingestion::NewFile::new("sales.csv", b"region,total\nnorth,2\n"),
        None::<&MockEmbeddingModel>,
    )
    .await;
    let err = changed.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(err.contains("belongs to document"), "{err}");
    assert!(err.contains(&first.document_id), "{err}");
    assert_eq!(
        db.list_documents().unwrap().len(),
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
        &db,
        "ws-owner",
        &ingestion::NewFile::new("sales.csv", first_bytes),
        None::<&MockEmbeddingModel>,
    )
    .await
    .unwrap();
    let reloaded = ingested(again).unwrap();
    assert_ne!(reloaded.document_id, first.document_id);
    assert!(db.list_tables().unwrap().contains(&String::from("sales")));
    let stale = db.document(&first.document_id).unwrap().unwrap();
    assert_eq!(stale.status, "error");
    assert!(
        db.table_owner("sales")
            .unwrap()
            .is_some_and(|d| d.id == reloaded.document_id)
    );

    // A queued row from a process that died is failed when the server
    // opens the workspace.
    let registration =
        ingestion::register_document(&db, &ingestion::NewFile::new("later.csv", b"a\n1\n"))
            .unwrap();
    let ingestion::Registration::New(queued) = registration else {
        return assert!(matches!(registration, ingestion::Registration::New(_)));
    };
    assert_eq!(db.fail_stale_uploads().unwrap(), 1);
    let failed = db.document(&queued).unwrap().unwrap();
    assert_eq!(failed.status, "error");
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
    let server_policy = quack_core::import::ImportPolicy::server(&config);
    assert!(!server_policy.local_files);
    assert!(!server_policy.private_hosts);
    let local_file = quack_core::import::import(
        &config,
        &db,
        "ws-import-policy",
        &quack_core::import::ImportRequest {
            url: format!("sqlite://{}", dir.path().join("control.db").display()),
            table: String::from("x"),
            query: None,
            source_table: Some(String::from("users")),
            limit: None,
        },
        server_policy,
        None::<&MockEmbeddingModel>,
        None,
    )
    .await;
    assert!(local_file.is_err_and(|e| e.to_string().contains("allow_local_files")));
    assert!(db.list_tables().unwrap().is_empty());
}

#[tokio::test]
async fn sqlite_import_errors_are_specific_and_duplicates_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config_no_provider(dir.path());
    let db = WorkspaceDb::open(&config, "ws-import-errors").unwrap();
    let source_dir = tempfile::tempdir().unwrap();
    let source_path = source_dir.path().join("source.db");
    {
        use sqlx::Connection as _;
        use sqlx::Executor as _;
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
    let first = quack_core::import::ImportRequest {
        url: url.clone(),
        table: String::from("Orders Import"),
        query: None,
        source_table: Some(String::from("orders")),
        limit: None,
    };
    let summary = quack_core::import::import(
        &config,
        &db,
        "ws-import-errors",
        &first,
        quack_core::import::ImportPolicy::owner(),
        None::<&MockEmbeddingModel>,
        None,
    )
    .await
    .unwrap();
    assert_eq!(summary.rows, 1);
    // The same rows again are a duplicate; a bad query and a bad URL are errors.
    let again = quack_core::import::import(
        &config,
        &db,
        "ws-import-errors",
        &quack_core::import::ImportRequest {
            url: url.clone(),
            table: String::from("Orders Import"),
            query: None,
            source_table: Some(String::from("orders")),
            limit: None,
        },
        quack_core::import::ImportPolicy::owner(),
        None::<&MockEmbeddingModel>,
        None,
    )
    .await;
    assert!(again.is_err_and(|e| e.to_string().contains("identical")));
    let bad = quack_core::import::import(
        &config,
        &db,
        "ws-import-errors",
        &quack_core::import::ImportRequest {
            url,
            table: String::from("x"),
            query: Some(String::from("SELECT * FROM nope")),
            source_table: None,
            limit: None,
        },
        quack_core::import::ImportPolicy::owner(),
        None::<&MockEmbeddingModel>,
        None,
    )
    .await;
    assert!(bad.is_err_and(|e| e.to_string().contains("rejected the query")));
    let unsupported = quack_core::import::import(
        &config,
        &db,
        "ws-import-errors",
        &quack_core::import::ImportRequest {
            url: String::from("mysql://h/db"),
            table: String::from("x"),
            query: None,
            source_table: Some(String::from("t")),
            limit: None,
        },
        quack_core::import::ImportPolicy::owner(),
        None::<&MockEmbeddingModel>,
        None,
    )
    .await;
    assert!(unsupported.is_err());
    // Delete through the document row drops the imported table.
    assert!(db.delete_document(&summary.document_id, None).unwrap());
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
    const MAX_DOCUMENTS: usize = 1024;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
        Self {
            delay: std::time::Duration::from_secs(60),
        }
    }

    fn ndims(&self) -> usize {
        TEST_DIM
    }

    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + Send {
        let texts: Vec<String> = texts.into_iter().collect();
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
    let model = SlowModel {
        delay: std::time::Duration::from_secs(60),
    };
    let cancel = quack_core::llm::CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        trigger.cancel();
    });
    let started = std::time::Instant::now();
    let outcome = ingestion::ingest_file(
        &config,
        &db,
        workspace_id,
        &ingestion::NewFile::new("long.md", b"# Long\n\nSome text to embed.").cancel(Some(&cancel)),
        Some(&model),
    )
    .await;
    assert!(
        matches!(outcome, Err(quack_core::error::Error::Cancelled)),
        "{outcome:?}"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "the request in flight was abandoned, not waited out"
    );
    let documents = db.list_documents().unwrap();
    let document = documents.first().unwrap();
    assert_eq!(document.status, "error");
    assert_eq!(document.error_message.as_deref(), Some("cancelled"));
    let qr = db
        .execute_query("SELECT COUNT(*) AS cnt FROM _quack_chunks")
        .unwrap();
    assert_eq!(
        qr.rows.first().unwrap().first().unwrap(),
        &serde_json::Value::Number(0.into())
    );

    // Cancelled before it starts: nothing is parsed or stored.
    let early = quack_core::llm::CancellationToken::new();
    early.cancel();
    let outcome = ingestion::ingest_file(
        &config,
        &db,
        workspace_id,
        &ingestion::NewFile::new("other.md", b"# Other").cancel(Some(&early)),
        Some(&model),
    )
    .await;
    assert!(matches!(outcome, Err(quack_core::error::Error::Cancelled)));
}
