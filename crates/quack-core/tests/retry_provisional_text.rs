#![expect(clippy::unwrap_used, reason = "test assertions use unwrap throughout")]

//! Retried model turns must not leak their provisional streamed text into the
//! final answer. Two hook-retry paths exist in rig: the `InvalidToolCalls` hook
//! (rig abandons the turn and emits only `CompletionCall`) and the `EmptyAnswer`
//! hook (rig yields `ModelTurnRetried` after the turn's `CompletionCall`). A
//! whitespace-only turn trips `EmptyAnswer`; a text + misnamed-tool-call turn
//! trips `InvalidToolCalls`. In both, the rejected turn's streamed text must
//! be discarded so `AgentResponse.content` matches rig's `FinalResponse.output`.

use std::sync::Arc;
use std::time::Instant;

use quack_core::analysis::agent::Analysis;
use quack_core::analysis::events;
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::text_to_sql::PromptOptions;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::config::{AnalysisConfig, RetrievalConfig};
use quack_core::embedding::{Dimension, Embedder, EmbeddingModel};
use quack_core::graph::GraphOptions;
use quack_core::ids::{ChunkId, DocumentId};
use quack_core::storage::sessions::ChatMode;
use quack_core::storage::workspace::{DocumentStatus, NewChunk, NewDocument, WorkspaceDb};
use quack_core::storage::writer::Writer;
use quack_core::text::Tokens;
use rig::ProviderError;
use rig::completion::Usage;
use rig::embeddings::Embedding;
use rig::test_utils::{MockCompletionModel, MockStreamEvent};

#[derive(Clone)]
struct NoEmbedding;

impl EmbeddingModel for NoEmbedding {
    fn embed_texts(
        &self,
        _texts: Vec<String>,
    ) -> impl Future<Output = Result<Vec<Embedding>, ProviderError>> + Send {
        std::future::ready(Err(ProviderError::Provider(String::from(
            "no embedding model",
        ))))
    }
}

fn workspace() -> SharedDb {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    db.connection()
        .execute_batch(
            "CREATE TABLE sales (region VARCHAR, revenue INTEGER);
             INSERT INTO sales VALUES ('north', 10), ('south', 20);",
        )
        .unwrap();
    db.insert_document(
        &NewDocument::new(&DocumentId::from("doc-1"), "policy.md", "text/markdown", 10)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: &ChunkId::from("c0"),
        document_id: &DocumentId::from("doc-1"),
        chunk_index: 0,
        content: "Refunds are issued within 30 days of purchase.",
        heading: Some("Refunds"),
        page: None,
        embedding: None,
    })
    .unwrap();
    Arc::new(Writer::spawn(db).unwrap())
}

fn turn(mut events: Vec<MockStreamEvent>) -> Vec<MockStreamEvent> {
    events.push(MockStreamEvent::final_response(Usage::default()));
    events
}

fn call(id: &str, tool: &str, args: serde_json::Value) -> MockStreamEvent {
    MockStreamEvent::tool_call(id, tool, args)
}

fn text(text: &str) -> MockStreamEvent {
    MockStreamEvent::text(text)
}

async fn run(model: &MockCompletionModel, message: &str) -> String {
    let db = workspace();
    let analysis_config = AnalysisConfig::default();
    let retrieval_config = RetrievalConfig::default();
    let (sink, mut stream) = events::channel();
    let analysis = Analysis::<NoEmbedding> {
        db: Arc::clone(&db),
        reader_db: ReaderDb::new(Arc::clone(&db)),
        embedder: None::<Embedder<NoEmbedding>>,
        rerank_model: None,
        config: &analysis_config,
        retrieval_config: &retrieval_config,
        graph_options: GraphOptions::default(),
        write_policy: WritePolicy::Deny,
        prompt: PromptOptions {
            mode: ChatMode::Chat,
            write_policy: WritePolicy::Deny,
            pinned_token_budget: Tokens::new(1_000),
            context: None,
            context_max_tokens: Tokens::new(1_000),
            ollama_context_cap: None,
        },
        history: Vec::new(),
        message,
        asked: Instant::now(),
    };
    let response = analysis.run(model.clone().erase(), None, sink).await;
    while stream.try_recv().is_ok() {}
    response.unwrap().content
}

#[tokio::test(flavor = "multi_thread")]
async fn invalid_tool_call_retry_leaks_provisional_text() {
    let model = MockCompletionModel::from_stream_turns([
        turn(vec![
            text("Let me look into that. "),
            call("t1", "default_api", serde_json::json!({})),
        ]),
        turn(vec![text("There is one table, sales.")]),
    ]);
    let content = run(&model, "tables?").await;
    assert_eq!(content, "There is one table, sales.");
}

#[tokio::test(flavor = "multi_thread")]
async fn empty_answer_whitespace_retry_leaks_provisional_text() {
    let model = MockCompletionModel::from_stream_turns([
        turn(vec![text("   ")]),
        turn(vec![text("Two regions.")]),
    ]);
    let content = run(&model, "regions?").await;
    assert_eq!(content, "Two regions.");
}
