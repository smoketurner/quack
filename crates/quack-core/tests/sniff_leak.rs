#![expect(clippy::unwrap_used, reason = "test assertions use unwrap throughout")]

//! Regression test for the ingestion path-disclosure leak: a file whose
//! reader (`sniff_csv` for CSV/TSV, `read_parquet`/`read_json_auto` for the
//! rest) fails must not surface `DuckDB`'s absolute `file = <path>` line to
//! the person. `Processing::run` persists `e.to_string()` into
//! `DocumentInfo.error_message`, which the UI renders verbatim, so every
//! reader-failure arm in `TableLoad::create` scrubs the raw `Error::DuckDb`
//! into a path-free `Error::Ingestion` (and logs the detail) instead.

use std::collections::BTreeMap;
use std::path::{MAIN_SEPARATOR, Path};

use quack_core::config::{
    AnalysisConfig, BaseUrl, Config, ContextConfig, EmbeddingConfig, GeneralConfig, GraphConfig,
    ImportConfig, IngestionConfig, JobsConfig, OntologyConfig, ProviderConfig, ProviderType,
    RetrievalConfig, ServerConfig,
};
use quack_core::embedding::{Dimension, Embedder, EmbeddingModel, Profile, Prompts};
use quack_core::error::Error;
use quack_core::ingestion;
use quack_core::storage::workspace::{DocumentStatus, WorkspaceDb};
use quack_core::storage::writer::Writer;
use rig::ProviderError;
use rig::embeddings::Embedding;

const TEST_DIM: usize = 4;
const TEST_DIM_U32: u32 = 4;

struct MockEmbeddingModel {
    dim: usize,
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

/// `model` under the profile `test_config` configures: `mock-model`,
/// `TEST_DIM` wide, no prefixes.
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
            default_workspace: "test".into(),
            chat_model: None,
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
        import: ImportConfig::default(),
        jobs: JobsConfig::default(),
    }
}

/// A writer over a second connection to `db`'s database, for ingestion.
fn writer_of(db: &WorkspaceDb) -> Writer {
    Writer::spawn(db.try_clone_reader().unwrap()).unwrap()
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
        .list_documents()
        .unwrap()
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
