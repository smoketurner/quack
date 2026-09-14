#![expect(clippy::unwrap_used, reason = "test assertions use unwrap throughout")]

use std::collections::BTreeMap;
use std::path::Path;

use quack_core::config::{
    AnalysisConfig, Config, GeneralConfig, IngestionConfig, ProviderConfig, RetrievalConfig,
};
use quack_core::ingestion;
use quack_core::ingestion::parser::FileType;
use quack_core::storage::workspace::{StatementKind, WorkspaceDb};
use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};

const TEST_DIM: usize = 4;
const TEST_DIM_U32: u32 = 4;

struct MockEmbeddingModel {
    dim: usize,
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
            provider_type: "openai-compat".into(),
            base_url: Some("http://localhost:9999".into()),
            api_key_env: None,
            model: None,
            embedding_model: Some("mock-model".into()),
            embedding_dimension: Some(TEST_DIM_U32),
        },
    );
    Config {
        general: GeneralConfig {
            data_dir: data_dir.to_path_buf(),
            default_workspace: "test".into(),
        },
        providers,
        ingestion: IngestionConfig {
            chunk_size_tokens: 50,
            chunk_overlap_tokens: 10,
            embedding_batch_size: 64,
            tokenizer_encoding: String::from("cl100k_base"),
        },
        retrieval: RetrievalConfig::default(),
        analysis: AnalysisConfig::default(),
    }
}

fn test_config_no_provider(data_dir: &Path) -> Config {
    Config {
        general: GeneralConfig {
            data_dir: data_dir.to_path_buf(),
            default_workspace: "test".into(),
        },
        providers: BTreeMap::new(),
        ingestion: IngestionConfig::default(),
        retrieval: RetrievalConfig::default(),
        analysis: AnalysisConfig::default(),
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
    let result = ingestion::ingest_file::<MockEmbeddingModel>(
        &config,
        &db,
        workspace_id,
        "test.txt",
        data,
        None,
    )
    .await
    .unwrap();

    assert_eq!(result.filename, "test.txt");
    assert_eq!(result.file_type, FileType::Text);
    assert!(result.chunks_stored > 0);
    assert!(result.table_name.is_none());

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
        "embed_test.txt",
        data,
        Some(&model),
    )
    .await
    .unwrap();

    assert_eq!(result.file_type, FileType::Text);
    assert!(result.chunks_stored > 0);

    let qr = db
        .execute_query("SELECT COUNT(*) AS cnt FROM _quack_chunks WHERE embedding IS NOT NULL")
        .unwrap();
    let count = qr.rows.first().unwrap().first().unwrap();
    assert_ne!(count, &serde_json::Value::Number(0.into()));
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

    let result = ingestion::ingest_file::<MockEmbeddingModel>(
        &config,
        &db,
        workspace_id,
        "people.csv",
        csv_content,
        None,
    )
    .await
    .unwrap();

    assert_eq!(result.file_type, FileType::Csv);
    assert_eq!(result.chunks_stored, 0);
    assert_eq!(result.table_name.as_deref(), Some("people"));

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

    let result = ingestion::ingest_file::<MockEmbeddingModel>(
        &config,
        &db,
        workspace_id,
        "scores.json",
        json_content,
        None,
    )
    .await
    .unwrap();

    assert_eq!(result.file_type, FileType::Json);
    assert_eq!(result.table_name.as_deref(), Some("scores"));

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

    let result = ingestion::ingest_file::<MockEmbeddingModel>(
        &config,
        &db,
        workspace_id,
        "image.png",
        b"fake image data",
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

    let result = ingestion::ingest_file::<MockEmbeddingModel>(
        &config,
        &db,
        workspace_id,
        "empty.txt",
        b"",
        None,
    )
    .await
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
    let result = ingestion::ingest_file::<MockEmbeddingModel>(
        &config,
        &db,
        workspace_id,
        "notes.md",
        data,
        None,
    )
    .await
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

    db.insert_document("doc-1", "test.txt", "text/plain", 100, "pending")
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
    db.insert_document("doc-1", "test.txt", "text/plain", 100, "ready")
        .unwrap();

    db.insert_chunk("c1", "doc-1", 0, "hello world", None)
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
    db.insert_document("doc-1", "test.txt", "text/plain", 100, "ready")
        .unwrap();

    let embedding = [0.5_f32, 0.3, -0.2, 0.8];
    db.insert_chunk("c1", "doc-1", 0, "embedded chunk", Some(&embedding))
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
    db.insert_document("doc-1", "test.txt", "text/plain", 100, "ready")
        .unwrap();

    db.insert_chunk("c1", "doc-1", 0, "hello world", None)
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

    db.insert_document("doc-a", "policy.pdf", "application/pdf", 10, "ready")
        .unwrap();
    db.insert_document("doc-b", "faq.md", "text/markdown", 10, "ready")
        .unwrap();
    db.insert_chunk(
        "a0",
        "doc-a",
        0,
        "flood exclusion",
        Some(&[1.0, 0.0, 0.0, 0.0]),
    )
    .unwrap();
    db.insert_chunk(
        "b0",
        "doc-b",
        0,
        "claims timeline",
        Some(&[0.9, 0.1, 0.0, 0.0]),
    )
    .unwrap();

    let query = [1.0_f32, 0.0, 0.0, 0.0];

    let all = match db.search_similar_chunks(&query, 5, &[]) {
        Ok(r) => r,
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("array_cosine_distance") || msg.contains("Catalog Error"),
                "unexpected error: {msg}"
            );
            return;
        }
    };
    assert_eq!(all.len(), 2);
    assert_eq!(all.first().unwrap().filename, "policy.pdf");
    assert_eq!(all.last().unwrap().filename, "faq.md");

    let only_b = db
        .search_similar_chunks(&query, 5, &[String::from("doc-b")])
        .unwrap();
    assert_eq!(only_b.len(), 1);
    assert_eq!(only_b.first().unwrap().document_id, "doc-b");
    assert_eq!(only_b.first().unwrap().filename, "faq.md");

    let none = db
        .search_similar_chunks(&query, 5, &[String::from("missing")])
        .unwrap();
    assert!(none.is_empty());
}

#[test]
fn workspace_db_search_similar_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let workspace_id = "ws-search";

    let db = WorkspaceDb::open(&config, workspace_id).unwrap();
    db.insert_document("doc-1", "test.txt", "text/plain", 100, "ready")
        .unwrap();

    db.insert_chunk("c1", "doc-1", 0, "first chunk", Some(&[1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    db.insert_chunk(
        "c2",
        "doc-1",
        1,
        "second chunk",
        Some(&[0.0, 1.0, 0.0, 0.0]),
    )
    .unwrap();
    db.insert_chunk("c3", "doc-1", 2, "third chunk", Some(&[0.7, 0.7, 0.0, 0.0]))
        .unwrap();

    let query = [1.0_f32, 0.0, 0.0, 0.0];
    match db.search_similar_chunks(&query, 3, &[]) {
        Ok(results) => {
            assert_eq!(results.len(), 3);
            let first = results.first().unwrap();
            assert_eq!(first.content, "first chunk");
            let last = results.last().unwrap();
            assert_eq!(last.content, "second chunk");
        }
        Err(e) => {
            // vss extension may not be available in all environments
            let msg = e.to_string();
            assert!(
                msg.contains("array_cosine_distance") || msg.contains("Catalog Error"),
                "unexpected error: {msg}"
            );
        }
    }
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
    ] {
        assert_eq!(kind(&db, sql), StatementKind::Write, "{sql}");
    }
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
    assert_eq!(db.meta("schema_version").unwrap().as_deref(), Some("1"));
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
        db.insert_document("d", "a.txt", "text/plain", 1, "ready")
            .unwrap();
        db.insert_chunk("c", "d", 0, "x", Some(&[1.0, 0.0, 0.0, 0.0]))
            .unwrap();
    }
    let mut changed = test_config(dir.path());
    if let Some(p) = changed.providers.get_mut("mock") {
        p.embedding_dimension = Some(8);
        p.embedding_model = Some("other-model".into());
    }
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
        db.insert_document("d", "a.txt", "text/plain", 1, "ready")
            .unwrap();
        db.insert_chunk("c", "d", 0, "x", None).unwrap();
    }
    let mut changed = test_config(dir.path());
    if let Some(p) = changed.providers.get_mut("mock") {
        p.embedding_dimension = Some(8);
    }
    let db = WorkspaceDb::open(&changed, "ws-adopt").unwrap();
    assert_eq!(db.embedding_dimension(), 8);
    assert_eq!(
        db.meta("embedding_dimension").unwrap().as_deref(),
        Some("8")
    );
    db.insert_chunk("c2", "d", 1, "y", Some(&[0.5; 8])).unwrap();
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
async fn control_db_migrates_to_v2_and_drops_content_tables() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let control = quack_core::storage::control::ControlPlane::open(&config)
        .await
        .unwrap();
    assert_eq!(control.schema_version().await.unwrap(), 2);
    // Reopening is a no-op.
    let again = quack_core::storage::control::ControlPlane::open(&config)
        .await
        .unwrap();
    assert_eq!(again.schema_version().await.unwrap(), 2);
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
        filename,
        b"a,b\n1,2\n3,4\n",
        None::<&MockEmbeddingModel>,
    )
    .await
    .unwrap();
    assert_eq!(result.table_name.as_deref(), Some("it_s_a_file"));
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
