#![expect(clippy::unwrap_used, reason = "test assertions use unwrap throughout")]

//! Embedding profiles end to end: which prefix each role gets, how a
//! workspace records the profile of every stored vector, what happens to
//! vectors made under an older profile or width, and the refresh that
//! brings them up to date.

use std::path::Path;
use std::sync::{Arc, Mutex};

use quack_core::config::Config;
use quack_core::embedding::refresh::{self, Plan, Retype};
use quack_core::embedding::{Dimension, Embedder, Profile, Prompts, Vector};
use quack_core::error::Error;
use quack_core::graph::store::{self as graph_store, NewNode};
use quack_core::graph::{Properties, Standing};
use quack_core::ids::{ChunkId, ClassId, DocumentId};
use quack_core::ingestion::{self, NewFile};
use quack_core::progress::{ChunkDone, RunControl};
use quack_core::storage::workspace::{
    ChunkScope, DocumentStatus, HybridLimits, MetaKey, NewChunk, NewDocument, WorkspaceDb,
};
use quack_core::storage::writer::Writer;
use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};
use tokio_util::sync::CancellationToken;

/// Records every input and answers with vectors of `width`.
#[derive(Clone)]
struct Tape {
    width: usize,
    inputs: Arc<Mutex<Vec<String>>>,
}

impl Tape {
    fn new(width: usize) -> Self {
        Self {
            width,
            inputs: Arc::default(),
        }
    }

    fn inputs(&self) -> Vec<String> {
        self.inputs.lock().unwrap().clone()
    }
}

impl EmbeddingModel for Tape {
    const MAX_DOCUMENTS: usize = 1024;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
        Self::new(4)
    }

    fn ndims(&self) -> usize {
        self.width
    }

    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> impl Future<Output = Result<Vec<Embedding>, EmbeddingError>> + Send {
        let texts: Vec<String> = texts.into_iter().collect();
        self.inputs.lock().unwrap().extend(texts.iter().cloned());
        let mut vec = vec![0.0; self.width];
        if let Some(first) = vec.first_mut() {
            *first = 1.0;
        }
        std::future::ready(Ok(texts
            .into_iter()
            .map(|document| Embedding {
                document,
                vec: vec.clone(),
            })
            .collect()))
    }
}

/// A config whose embedding model is `mock/MODEL`, `dimension` wide, with
/// `extra` TOML appended to its `[embedding]` table.
fn config(dir: &Path, model: &str, dimension: u32, extra: &str) -> Config {
    let text = format!(
        "[general]\ndata_dir = {:?}\n\
         [providers.mock]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\n\
         [embedding]\nmodel = \"mock/{model}\"\ndimension = {dimension}\n{extra}",
        dir.display().to_string()
    );
    let config: Config = toml::from_str(&text).unwrap();
    config.validate().unwrap();
    config
}

/// The configured profile over `model`.
fn embedder(config: &Config, model: Tape) -> Embedder<Tape> {
    Embedder::new(model, Profile::from_config(config).unwrap().unwrap())
}

/// A writer over a second connection to `db`'s database.
fn writer_of(db: &WorkspaceDb) -> Writer {
    Writer::spawn(db.try_clone_reader().unwrap()).unwrap()
}

/// One ready document with `chunks` chunks, each with a vector of `width`.
fn seed(db: &WorkspaceDb, chunks: u32, width: usize) {
    db.insert_document(
        &NewDocument::new(&DocumentId::from("d"), "a.md", "text/markdown", 1)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();
    let vector = vec![1.0_f32; width];
    for i in 0..chunks {
        db.insert_chunk(&NewChunk {
            id: &ChunkId::from(format!("c{i}")),
            document_id: &DocumentId::from("d"),
            chunk_index: i,
            content: &format!("storm report {i}"),
            heading: Some("Reports"),
            page: None,
            embedding: Some(&Vector::from(vector.clone())),
        })
        .unwrap();
    }
}

/// Turn `db` back into a workspace written before profiles existed:
/// untagged vectors, the model under the old meta key, schema version 7.
fn make_legacy(db: &WorkspaceDb, model: &str) {
    for sql in [
        "UPDATE _quack_chunks SET embedding_profile = NULL",
        "UPDATE _quack_graph_nodes SET embedding_profile = NULL",
        "DELETE FROM _quack_embedding_profiles",
        "UPDATE _quack_meta SET value = '7' WHERE key = 'schema_version'",
    ] {
        db.execute_statement(sql).unwrap();
    }
    db.set_meta(MetaKey::EmbeddingModel, model).unwrap();
}

fn vector_hits(db: &WorkspaceDb, width: usize) -> usize {
    db.search_similar_chunks(&Vector::from(vec![1.0; width]), 10, &ChunkScope::all())
        .unwrap()
        .len()
}

#[test]
fn vectors_made_before_profiles_keep_working_when_nothing_changed() {
    let dir = tempfile::tempdir().unwrap();
    // A model with no known prefixes: its vectors were always made unprefixed.
    let plain = config(dir.path(), "all-minilm", 4, "");
    {
        let db = WorkspaceDb::open(&plain, "ws").unwrap();
        seed(&db, 3, 4);
        make_legacy(&db, "all-minilm:latest");
    }
    let db = WorkspaceDb::open(&plain, "ws").unwrap();
    let status = db.embedding_status().unwrap();
    assert_eq!(status.current_chunks, 3);
    assert_eq!(status.stale_chunks(), 0);
    assert_eq!(status.note(), None);
    assert!(Plan::from_status(&status).is_empty());
    assert_eq!(vector_hits(&db, 4), 3);
    assert_eq!(db.meta(MetaKey::EmbeddingModel).unwrap(), None);
}

#[test]
fn vectors_made_before_prefixes_are_stale_for_a_prefixed_model() {
    let dir = tempfile::tempdir().unwrap();
    let gemma = config(dir.path(), "embeddinggemma", 4, "");
    {
        let db = WorkspaceDb::open(&gemma, "ws").unwrap();
        seed(&db, 3, 4);
        make_legacy(&db, "embeddinggemma:latest");
    }
    let db = WorkspaceDb::open(&gemma, "ws").unwrap();
    let status = db.embedding_status().unwrap();
    assert_eq!(status.current_chunks, 0);
    assert_eq!(status.stale_chunks(), 3);
    let note = status.note().unwrap();
    assert!(
        note.contains("3 chunks were embedded with embeddinggemma (4 dimensions, no prefixes)")
            && note
                .contains("the configured model is embeddinggemma (4 dimensions, with prefixes)")
            && note.contains("keyword search only"),
        "{note}"
    );
    // Stale vectors are not searched; keyword search still finds the chunks.
    assert_eq!(vector_hits(&db, 4), 0);
    let hybrid = db
        .search_hybrid_chunks(
            "storm",
            &Vector::from(vec![1.0; 4]),
            HybridLimits {
                top_k: 10,
                rrf_k: 60,
            },
            &ChunkScope::all(),
        )
        .unwrap();
    assert_eq!(hybrid.len(), 3);
}

#[tokio::test]
async fn refresh_brings_stale_chunks_and_nodes_up_to_date_with_the_role_prefixes() {
    let dir = tempfile::tempdir().unwrap();
    let gemma = config(dir.path(), "embeddinggemma", 4, "");
    {
        let db = WorkspaceDb::open(&gemma, "ws").unwrap();
        seed(&db, 3, 4);
        let node = graph_store::upsert_node(
            &db,
            &NewNode {
                label: String::from("Acme"),
                class_id: ClassId::from("organization"),
                properties: Properties::default(),
                standing: Standing::Reviewed,
            },
        )
        .unwrap();
        let vector = Vector::new(vec![1.0; 4], Dimension::new(4)).unwrap();
        db.set_node_embedding(&node, &vector).unwrap();
        make_legacy(&db, "embeddinggemma");
    }
    let db = WorkspaceDb::open(&gemma, "ws").unwrap();
    assert_eq!(db.embedding_status().unwrap().stale_nodes, 1);
    assert!(
        graph_store::nearest_nodes(&db, &Vector::from(vec![1.0; 4]), None, 5)
            .unwrap()
            .is_empty(),
        "a stale label vector is not matched"
    );
    let tape = Tape::new(4);
    let writer = writer_of(&db);
    let seen = Mutex::new(Vec::new());
    let progress = |done: ChunkDone| seen.lock().unwrap().push((done.done, done.total));
    let control = RunControl {
        progress: &progress,
        cancel: None,
    };
    let summary = refresh::run(&writer, &embedder(&gemma, tape.clone()), 2, control)
        .await
        .unwrap();
    assert_eq!(
        (summary.chunks, summary.nodes, summary.retyped_from),
        (3, 1, None)
    );
    assert_eq!(
        tape.inputs(),
        [
            "title: Reports | text: storm report 0",
            "title: Reports | text: storm report 1",
            "title: Reports | text: storm report 2",
            "task: sentence similarity | query: Acme (organization)",
        ]
    );
    // Chunks and node labels count together: two chunk batches, then one
    // node batch.
    assert_eq!(seen.into_inner().unwrap(), [(2, 4), (3, 4), (4, 4)]);
    let status = db.embedding_status().unwrap();
    assert_eq!(
        (
            status.current_chunks,
            status.stale_chunks(),
            status.stale_nodes
        ),
        (3, 0, 0)
    );
    assert_eq!(vector_hits(&db, 4), 3);
    assert_eq!(
        graph_store::nearest_nodes(&db, &Vector::from(vec![1.0; 4]), None, 5)
            .unwrap()
            .len(),
        1
    );
    // A second run has nothing to do.
    let again = refresh::run(
        &writer,
        &embedder(&gemma, Tape::new(4)),
        2,
        RunControl::unobserved(),
    )
    .await
    .unwrap();
    assert_eq!((again.chunks, again.nodes), (0, 0));
}

#[tokio::test]
async fn a_configured_prefix_change_makes_vectors_stale() {
    let dir = tempfile::tempdir().unwrap();
    let before = config(dir.path(), "nomic-embed-text", 4, "");
    {
        let db = WorkspaceDb::open(&before, "ws").unwrap();
        let writer = writer_of(&db);
        let tape = Tape::new(4);
        ingestion::ingest_file(
            &before,
            &writer,
            "ws",
            &NewFile::new("notes.md", b"# Notes\n\nThe levee held."),
            Some(&embedder(&before, tape.clone())),
        )
        .await
        .unwrap();
        assert_eq!(tape.inputs(), ["search_document: Notes\n\nThe levee held."]);
        assert_eq!(db.embedding_status().unwrap().current_chunks, 1);
    }
    let after = config(
        dir.path(),
        "nomic-embed-text",
        4,
        "document_prefix = \"\"\n",
    );
    let db = WorkspaceDb::open(&after, "ws").unwrap();
    let status = db.embedding_status().unwrap();
    assert_eq!(status.stale_chunks(), 1);
    let old = status
        .stale
        .first()
        .and_then(|s| s.profile.clone())
        .unwrap();
    assert_eq!(old.prompts.document, "search_document: ");
    let tape = Tape::new(4);
    refresh::run(
        &writer_of(&db),
        &embedder(&after, tape.clone()),
        8,
        RunControl::unobserved(),
    )
    .await
    .unwrap();
    assert_eq!(tape.inputs(), ["Notes\n\nThe levee held."]);
    assert_eq!(db.embedding_status().unwrap().current_chunks, 1);
}

#[tokio::test]
async fn a_width_change_stores_new_chunks_without_vectors_until_refresh_retypes() {
    let dir = tempfile::tempdir().unwrap();
    let narrow = config(dir.path(), "all-minilm", 4, "");
    {
        let db = WorkspaceDb::open(&narrow, "ws").unwrap();
        seed(&db, 2, 4);
    }
    let wide = config(dir.path(), "all-minilm", 8, "");
    let db = WorkspaceDb::open(&wide, "ws").unwrap();
    let reader = db.try_clone_reader().unwrap();
    let writer = writer_of(&db);

    // Ingesting now keeps the document, keyword-searchable, with no vector.
    let tape = Tape::new(8);
    let result = ingestion::ingest_file(
        &wide,
        &writer,
        "ws",
        &NewFile::new("later.md", b"# Later\n\nA later report."),
        Some(&embedder(&wide, tape.clone())),
    )
    .await
    .unwrap()
    .ingested()
    .unwrap();
    assert_eq!(result.chunks_stored, 1);
    assert!(
        tape.inputs().is_empty(),
        "no model call while the width differs"
    );
    let status = db.embedding_status().unwrap();
    assert_eq!((status.missing_chunks, status.stale_chunks()), (1, 2));
    let plan = Plan::from_status(&status);
    assert_eq!(
        (plan.retype, plan.chunks),
        (
            Some(Retype {
                stored: Dimension::new(4),
                configured: Dimension::new(8)
            }),
            3
        )
    );

    let summary = refresh::run(
        &writer,
        &embedder(&wide, Tape::new(8)),
        8,
        RunControl::unobserved(),
    )
    .await
    .unwrap();
    assert_eq!(
        (summary.retyped_from, summary.chunks),
        (Some(Dimension::new(4)), 3)
    );
    // Every connection to the workspace sees the new width.
    assert_eq!(reader.embedding_dimension(), Dimension::new(8));
    assert_eq!(vector_hits(&reader, 8), 3);
    assert_eq!(
        db.meta(MetaKey::EmbeddingDimension).unwrap().as_deref(),
        Some("8")
    );
}

#[tokio::test]
async fn a_cancelled_refresh_keeps_the_batches_it_finished() {
    let dir = tempfile::tempdir().unwrap();
    let gemma = config(dir.path(), "embeddinggemma", 4, "");
    {
        let db = WorkspaceDb::open(&gemma, "ws").unwrap();
        seed(&db, 4, 4);
        make_legacy(&db, "embeddinggemma");
    }
    let db = WorkspaceDb::open(&gemma, "ws").unwrap();
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    // Cancel once the first batch is stored.
    let progress = move |done: ChunkDone| {
        if done.done >= 1 {
            trigger.cancel();
        }
    };
    let outcome = refresh::run(
        &writer_of(&db),
        &embedder(&gemma, Tape::new(4)),
        1,
        RunControl {
            progress: &progress,
            cancel: Some(&cancel),
        },
    )
    .await;
    assert!(matches!(outcome, Err(Error::Cancelled)), "{outcome:?}");
    let status = db.embedding_status().unwrap();
    assert_eq!((status.current_chunks, status.stale_chunks()), (1, 3));
}

#[tokio::test]
async fn refresh_refuses_an_embedder_under_another_profile() {
    let dir = tempfile::tempdir().unwrap();
    let gemma = config(dir.path(), "embeddinggemma", 4, "");
    let db = WorkspaceDb::open(&gemma, "ws").unwrap();
    let other = Embedder::new(
        Tape::new(4),
        Profile::new("other", Dimension::new(4), Prompts::default()),
    );
    let err = refresh::run(&writer_of(&db), &other, 8, RunControl::unobserved())
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("different embedding profile"), "{err}");
}

#[tokio::test]
async fn a_model_answering_with_the_wrong_width_fails_the_document_with_the_fix() {
    let dir = tempfile::tempdir().unwrap();
    let configured = config(dir.path(), "embeddinggemma", 4, "");
    let db = WorkspaceDb::open(&configured, "ws").unwrap();
    let writer = writer_of(&db);
    let err = ingestion::ingest_file(
        &configured,
        &writer,
        "ws",
        &NewFile::new("a.md", b"# A\n\nText."),
        Some(&embedder(&configured, Tape::new(6))),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("returned 6-dimensional vectors")
            && err.contains("[embedding].dimension is 4")
            && err.contains("dimension = 6 under [embedding]"),
        "{err}"
    );
    let documents = db.list_documents().unwrap();
    assert_eq!(documents.first().map(|d| d.status.as_str()), Some("error"));
}
