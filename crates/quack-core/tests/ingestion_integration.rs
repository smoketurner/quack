#![expect(clippy::unwrap_used, reason = "test assertions use unwrap throughout")]

use std::collections::BTreeMap;
use std::path::Path;

use quack_core::config::{AnalysisConfig, Config, GeneralConfig, IngestionConfig, ProviderConfig};
use quack_core::ingestion;
use quack_core::ingestion::parser::FileType;
use quack_core::llm::EmbeddingProvider;
use quack_core::storage::workspace::WorkspaceDb;

const TEST_DIM: u32 = 4;

struct MockEmbeddingProvider {
    dim: u32,
}

impl EmbeddingProvider for MockEmbeddingProvider {
    fn embed(
        &self,
        texts: &[&str],
    ) -> impl std::future::Future<Output = quack_core::error::Result<Vec<Vec<f32>>>> + Send {
        let dim = self.dim as usize;
        let mut result = Vec::with_capacity(texts.len());
        for _ in texts {
            result.push(vec![0.1_f32; dim]);
        }
        std::future::ready(Ok(result))
    }

    fn embedding_dimension(&self) -> u32 {
        self.dim
    }

    fn model_name(&self) -> &'static str {
        "mock-model"
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
            embedding_dimension: Some(TEST_DIM),
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
    let result = ingestion::ingest_file::<MockEmbeddingProvider>(
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
        .execute_query("SELECT COUNT(*) AS cnt FROM chunks")
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
    let provider = MockEmbeddingProvider { dim: TEST_DIM };

    let data = b"This is a longer document with enough words to produce at least one chunk. \
                 We need to make sure the embedding pipeline works end to end with our mock.";
    let result = ingestion::ingest_file(
        &config,
        &db,
        workspace_id,
        "embed_test.txt",
        data,
        Some(&provider),
    )
    .await
    .unwrap();

    assert_eq!(result.file_type, FileType::Text);
    assert!(result.chunks_stored > 0);

    let qr = db
        .execute_query("SELECT COUNT(*) AS cnt FROM chunks WHERE embedding IS NOT NULL")
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

    let result = ingestion::ingest_file::<MockEmbeddingProvider>(
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

    let result = ingestion::ingest_file::<MockEmbeddingProvider>(
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

    let result = ingestion::ingest_file::<MockEmbeddingProvider>(
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

    let result = ingestion::ingest_file::<MockEmbeddingProvider>(
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
    let result = ingestion::ingest_file::<MockEmbeddingProvider>(
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
        .execute_query("SELECT id, filename, status FROM documents WHERE id = 'doc-1'")
        .unwrap();
    assert_eq!(qr.rows.len(), 1);

    db.update_document_status("doc-1", "ready").unwrap();

    let qr = db
        .execute_query("SELECT status FROM documents WHERE id = 'doc-1'")
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
        .execute_query("SELECT id, content FROM chunks WHERE id = 'c1'")
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
        .execute_query("SELECT content FROM chunks WHERE embedding IS NOT NULL")
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
        .execute_query("SELECT COUNT(*) AS cnt FROM chunks WHERE id = 'c1' AND embedding IS NULL")
        .unwrap();
    let count = qr.rows.first().unwrap().first().unwrap();
    assert_eq!(count, &serde_json::Value::Number(1.into()));

    // Update with an embedding
    let embedding = [1.0_f32, 0.0, 0.0, 0.0];
    db.update_chunk_embedding("doc-1", 0, &embedding).unwrap();

    // Embedding should now be non-NULL
    let qr = db
        .execute_query(
            "SELECT COUNT(*) AS cnt FROM chunks WHERE id = 'c1' AND embedding IS NOT NULL",
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
    match db.search_similar_chunks(&query, 3) {
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
