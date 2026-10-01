#![expect(clippy::unwrap_used, reason = "test assertions use unwrap throughout")]
#![expect(
    clippy::indexing_slicing,
    reason = "tests index the requests and citations they just asserted exist"
)]

//! Whole agent turns through rig with a scripted model: the event order,
//! the response object, the recorded session, and how the turn recovers
//! from an invalid tool call or an empty reply.

use std::sync::Arc;
use std::time::Instant;

use jiff::Timestamp;
use quack_core::analysis::agent::{AgentResponse, Analysis};
use quack_core::analysis::events::{self, AgentEvent, ToolName};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::text_to_sql::PromptOptions;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::config::{AnalysisConfig, RetrievalConfig};
use quack_core::embedding::{Dimension, Embedder, EmbeddingModel};
use quack_core::error::Result as TurnResult;
use quack_core::graph::GraphOptions;
use quack_core::ids::{ChunkId, DocumentId};
use quack_core::storage::sessions::{self, ChatMode, MessageRole};
use quack_core::storage::workspace::{DocumentStatus, NewChunk, NewDocument, WorkspaceDb};
use quack_core::storage::writer::Writer;
use quack_core::text::Tokens;
use rig::ProviderError;
use rig::completion::Usage;
use rig::embeddings::Embedding;
use rig::message::Message;
use rig::test_utils::{MockCompletionModel, MockStreamEvent};

/// The turns run keyword-only; no embedding model is ever called.
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

/// A workspace with a `sales` table and one document of two chunks.
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
    for (i, content) in [
        "Refunds are issued within 30 days of purchase.",
        "Refunds for damaged goods need a photo.",
    ]
    .into_iter()
    .enumerate()
    {
        db.insert_chunk(&NewChunk {
            id: &ChunkId::from(format!("c{i}")),
            document_id: &DocumentId::from("doc-1"),
            chunk_index: u32::try_from(i).unwrap(),
            content,
            heading: Some("Refunds"),
            page: None,
            embedding: None,
        })
        .unwrap();
    }
    Arc::new(Writer::spawn(db).unwrap())
}

/// A model turn that ends normally.
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

/// What one turn produced: the response (or the error), and every event
/// in the order it was sent.
struct Ran {
    response: TurnResult<AgentResponse>,
    events: Vec<AgentEvent>,
}

impl Ran {
    fn answer(&self) -> &AgentResponse {
        self.response.as_ref().unwrap()
    }

    /// The tool events as `started:tool` and `finished:tool`.
    fn tool_events(&self) -> Vec<String> {
        self.events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolStarted { tool, .. } => Some(format!("started:{tool}")),
                AgentEvent::ToolFinished(step) => Some(format!("finished:{}", step.tool)),
                _ => None,
            })
            .collect()
    }
}

async fn run_turn(
    db: &SharedDb,
    model: &MockCompletionModel,
    policy: WritePolicy,
    history: Vec<Message>,
    message: &str,
) -> Ran {
    let analysis_config = AnalysisConfig::default();
    let retrieval_config = RetrievalConfig::default();
    let (sink, mut stream) = events::channel();
    let analysis = Analysis::<NoEmbedding> {
        db: Arc::clone(db),
        reader_db: ReaderDb::new(Arc::clone(db)),
        embedder: None::<Embedder<NoEmbedding>>,
        rerank_model: None,
        config: &analysis_config,
        retrieval_config: &retrieval_config,
        graph_options: GraphOptions::default(),
        write_policy: policy,
        prompt: PromptOptions {
            mode: ChatMode::Chat,
            write_policy: policy,
            pinned_token_budget: Tokens::new(1_000),
            context: None,
            context_max_tokens: Tokens::new(1_000),
            ollama_context_cap: None,
        },
        history,
        message,
        asked: Instant::now(),
    };
    let response = analysis.run(model.clone().erase(), sink).await;
    let mut events = Vec::new();
    while let Ok(event) = stream.try_recv() {
        events.push(event);
    }
    Ran { response, events }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_turn_streams_its_tools_in_order_and_records_the_session() {
    let db = workspace();
    let model = MockCompletionModel::from_stream_turns([
        turn(vec![call("t1", "list_tables", serde_json::json!({}))]),
        turn(vec![call(
            "t2",
            "search_documents",
            serde_json::json!({ "query": "refunds" }),
        )]),
        turn(vec![call(
            "t3",
            "run_sql",
            serde_json::json!({ "query": "INSERT INTO sales VALUES ('east', 5)" }),
        )]),
        // Cites the second passage only, and a number no tool returned.
        turn(vec![text("Refunds need a photo [2], see also [7].")]),
    ]);
    let ran = run_turn(&db, &model, WritePolicy::Deny, Vec::new(), "refunds?").await;
    let response = ran.answer();

    assert_eq!(
        ran.tool_events(),
        [
            "started:list_tables",
            "finished:list_tables",
            "started:search_documents",
            "finished:search_documents",
            "started:run_sql",
            "finished:run_sql",
        ]
    );
    assert!(
        matches!(ran.events.last(), Some(AgentEvent::TurnComplete(_))),
        "{:?}",
        ran.events.last()
    );
    assert!(response.write_refused);
    let rows: i64 = db
        .run(|db| {
            Ok(db
                .connection()
                .query_row("SELECT count(*) FROM sales", [], |row| row.get(0))?)
        })
        .await
        .unwrap();
    assert_eq!(rows, 2, "the refused insert did not run");

    // The cited passage is renumbered from 1 and the unregistered marker
    // is gone.
    assert_eq!(response.content, "Refunds need a photo [1], see also .");
    assert_eq!(response.citations.len(), 1);
    assert_eq!(response.citations[0].n, 1);
    assert_eq!(response.citations[0].document_id, DocumentId::from("doc-1"));
    let tools: Vec<ToolName> = response.steps.iter().map(|s| s.tool).collect();
    assert_eq!(
        tools,
        [
            ToolName::ListTables,
            ToolName::SearchDocuments,
            ToolName::RunSql
        ]
    );

    // Recorded as the interfaces record it: the question, one tool row per
    // step, then the answer.
    let recorded = response.clone();
    let roles = db
        .run(move |db| {
            let session = sessions::create_session(db, "mock/model", ChatMode::Chat, None)?;
            sessions::record_turn(db, &session.id, "refunds?", Timestamp::now(), &recorded)?;
            sessions::messages(db, &session.id)
        })
        .await
        .unwrap()
        .into_iter()
        .map(|m| m.role)
        .collect::<Vec<_>>();
    assert_eq!(
        roles,
        [
            MessageRole::User,
            MessageRole::Tool,
            MessageRole::Tool,
            MessageRole::Tool,
            MessageRole::Assistant
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_tool_is_retried_and_the_real_one_runs() {
    let db = workspace();
    let model = MockCompletionModel::from_stream_turns([
        turn(vec![call("t1", "default_api", serde_json::json!({}))]),
        turn(vec![call("t2", "list_tables", serde_json::json!({}))]),
        turn(vec![text("There is one table, sales.")]),
    ]);
    let ran = run_turn(&db, &model, WritePolicy::Deny, Vec::new(), "tables?").await;
    assert_eq!(ran.answer().content, "There is one table, sales.");
    assert_eq!(
        ran.tool_events(),
        ["started:list_tables", "finished:list_tables"]
    );
    // The retry told the model which tools exist.
    let retry = serde_json::to_string(&model.requests()[1].chat_history).unwrap();
    assert!(
        retry.contains("There is no tool named `default_api`") && retry.contains("list_tables"),
        "{retry}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tool_name_in_the_wrong_case_is_repaired() {
    let db = workspace();
    let model = MockCompletionModel::from_stream_turns([
        turn(vec![call("t1", "List_Tables", serde_json::json!({}))]),
        turn(vec![text("One table.")]),
    ]);
    let ran = run_turn(&db, &model, WritePolicy::Deny, Vec::new(), "tables?").await;
    assert_eq!(ran.answer().content, "One table.");
    assert_eq!(
        ran.tool_events(),
        ["started:list_tables", "finished:list_tables"]
    );
    assert_eq!(model.request_count(), 2, "repaired without another try");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_tool_past_the_retry_budget_stops_with_the_note() {
    let db = workspace();
    let model = MockCompletionModel::from_stream_turns([
        turn(vec![
            text("Let me look. "),
            call("t1", "default_api", serde_json::json!({})),
        ]),
        turn(vec![call("t2", "default_api", serde_json::json!({}))]),
        turn(vec![call("t3", "default_api", serde_json::json!({}))]),
        turn(vec![text("never reached")]),
    ]);
    let ran = run_turn(&db, &model, WritePolicy::Deny, Vec::new(), "tables?").await;
    let content = &ran.answer().content;
    assert!(
        content.contains("The model called a tool that does not exist (default_api)"),
        "{content}"
    );
    assert_eq!(model.request_count(), 3, "the first call and two retries");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_empty_reply_is_asked_for_again() {
    let db = workspace();
    let model = MockCompletionModel::from_stream_turns([
        turn(Vec::new()),
        turn(vec![text("Two regions.")]),
    ]);
    let ran = run_turn(&db, &model, WritePolicy::Deny, Vec::new(), "regions?").await;
    assert_eq!(ran.answer().content, "Two regions.");
    let retry = serde_json::to_string(&model.requests()[1].chat_history).unwrap();
    assert!(retry.contains("Your last reply was empty"), "{retry}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_second_empty_reply_ends_the_turn_with_the_note() {
    let db = workspace();
    let model = MockCompletionModel::from_stream_turns([
        turn(Vec::new()),
        turn(Vec::new()),
        turn(vec![text("never reached")]),
    ]);
    let ran = run_turn(&db, &model, WritePolicy::Deny, Vec::new(), "regions?").await;
    assert_eq!(
        ran.answer().content,
        "(The model returned no text; ask again or narrow the question.)"
    );
    assert_eq!(model.request_count(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_malformed_history_is_dropped_with_a_warning() {
    let db = workspace();
    let model = MockCompletionModel::from_stream_turns([turn(vec![text("Hello.")])]);
    // Two assistant messages in a row: rig refuses such a transcript.
    let history = vec![
        Message::user("hi"),
        Message::assistant("one"),
        Message::assistant("two"),
    ];
    let ran = run_turn(&db, &model, WritePolicy::Deny, history, "hello").await;
    let content = &ran.answer().content;
    assert!(content.starts_with("Hello."), "{content}");
    assert!(
        content.contains("The session's earlier messages could not be replayed"),
        "{content}"
    );
    assert!(
        model.requests()[0]
            .chat_history
            .iter()
            .all(|m| !matches!(m, Message::Assistant { .. })),
        "the model saw none of the history"
    );
}
