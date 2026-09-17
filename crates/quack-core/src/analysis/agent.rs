use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::prelude::*;
use rig::streaming::StreamedAssistantContent;

use crate::config::{AnalysisConfig, RerankMode, RetrievalConfig};
use crate::error::{Error, Result};

use super::chart::ChartSpec;
use super::citations::{self, Citation};
use super::events::{AgentEvent, EventSink, ToolStep, TurnRecorder};
use super::policy::{RefusalFlag, WritePolicy};
use super::rerank::ModelReranker;
use super::text_to_sql::{self, PromptOptions};
use super::tools::{
    CreateChartTool, DescribeTableTool, ListDocumentsTool, ListTablesTool, RunSqlTool,
    SearchDocumentsTool, SharedDb,
};
use super::vector_index::DuckDbVectorIndex;

/// Everything a turn produced, delivered with `AgentEvent::TurnComplete` and
/// returned from `run_analysis`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentResponse {
    pub content: String,
    pub steps: Vec<ToolStep>,
    /// Sources the answer cites, numbered as they appear in `content`.
    pub citations: Vec<Citation>,
    pub chart: Option<ChartSpec>,
    /// At least one mutating statement was refused during this turn.
    pub write_refused: bool,
}

/// Run the rig agent with all analysis tools for a single user question,
/// emitting `AgentEvent`s on `sink` as the turn progresses. `history` is
/// the prior conversation to replay to the model (see
/// `storage::sessions::history_for_model`).
///
/// Text streams as `TextDelta`; every tool call is bracketed by
/// `ToolStarted`/`ToolFinished`; a write under `WritePolicy::Ask` pauses on
/// `PermissionRequired` until the interface answers. The final
/// `TurnComplete` (or `Failed`) is also the function's return value.
///
/// # Errors
///
/// Returns an error if system prompt generation, agent building, or the
/// model call fails.
#[expect(
    clippy::too_many_arguments,
    reason = "one entry point per turn; the interfaces call llm::run_turn, which packs config"
)]
pub async fn run_analysis<M>(
    db: SharedDb,
    completion_model: impl rig::completion::CompletionModel + Clone + 'static,
    embedding_model: M,
    analysis_config: &AnalysisConfig,
    retrieval_config: &RetrievalConfig,
    write_policy: WritePolicy,
    prompt: PromptOptions,
    history: Vec<rig::message::Message>,
    user_message: &str,
    sink: EventSink,
) -> Result<AgentResponse>
where
    M: rig::embeddings::EmbeddingModel + Clone + Send + Sync + 'static,
{
    let recorder = TurnRecorder::new(sink);
    match run_inner(
        db,
        completion_model,
        embedding_model,
        analysis_config,
        retrieval_config,
        write_policy,
        prompt,
        history,
        user_message,
        &recorder,
    )
    .await
    {
        Ok(response) => {
            recorder.emit(AgentEvent::TurnComplete(response.clone()));
            Ok(response)
        }
        Err(e) => {
            recorder.emit(AgentEvent::Failed(e.to_string()));
            Err(e)
        }
    }
}

/// The `search_documents` tool, with the chat model as reranker when
/// `[retrieval].rerank = "model"`.
fn search_tool<M>(
    shared_db: SharedDb,
    completion_model: &(impl rig::completion::CompletionModel + Clone + 'static),
    embedding_model: M,
    retrieval_config: &RetrievalConfig,
    recorder: &TurnRecorder,
) -> SearchDocumentsTool<M> {
    let search = SearchDocumentsTool::new(
        shared_db,
        embedding_model,
        retrieval_config.top_k,
        retrieval_config.rrf_k,
        recorder.clone(),
    );
    match retrieval_config.rerank {
        RerankMode::None => search,
        RerankMode::Model => search.with_reranker(
            Arc::new(ModelReranker::new(completion_model.clone())),
            retrieval_config.rerank_candidates,
        ),
    }
}

#[expect(clippy::too_many_arguments, reason = "mirrors run_analysis")]
async fn run_inner<M>(
    shared_db: SharedDb,
    completion_model: impl rig::completion::CompletionModel + Clone + 'static,
    embedding_model: M,
    analysis_config: &AnalysisConfig,
    retrieval_config: &RetrievalConfig,
    write_policy: WritePolicy,
    prompt: PromptOptions,
    history: Vec<rig::message::Message>,
    user_message: &str,
    recorder: &TurnRecorder,
) -> Result<AgentResponse>
where
    M: rig::embeddings::EmbeddingModel + Clone + Send + Sync + 'static,
{
    let system_prompt = {
        let db = shared_db
            .lock()
            .map_err(|e| Error::Analysis(format!("mutex poisoned: {e}")))?;
        text_to_sql::build_system_prompt(&db, &prompt)?
    };
    let chart_spec: Arc<Mutex<Option<ChartSpec>>> = Arc::new(Mutex::new(None));
    let refused = RefusalFlag::default();

    let search = search_tool(
        Arc::clone(&shared_db),
        &completion_model,
        embedding_model.clone(),
        retrieval_config,
        recorder,
    );
    let mut builder = completion_model
        .into_agent_builder()
        .preamble(&system_prompt)
        .tool(search)
        .tool(RunSqlTool::new(
            Arc::clone(&shared_db),
            analysis_config.max_query_rows,
            write_policy,
            refused.clone(),
            recorder.clone(),
        ))
        .tool(DescribeTableTool::new(
            Arc::clone(&shared_db),
            recorder.clone(),
        ))
        .tool(ListTablesTool::new(
            Arc::clone(&shared_db),
            recorder.clone(),
        ))
        .tool(ListDocumentsTool::new(
            Arc::clone(&shared_db),
            recorder.clone(),
        ))
        .tool(CreateChartTool::new(
            Arc::clone(&shared_db),
            Arc::clone(&chart_spec),
            recorder.clone(),
        ))
        .temperature(0.1);

    if retrieval_config.always_retrieve {
        let samples = usize::try_from(retrieval_config.top_k)
            .map_err(|e| Error::Analysis(format!("top_k overflow: {e}")))?;
        let vector_index = DuckDbVectorIndex::new(Arc::clone(&shared_db), embedding_model);
        builder = builder.dynamic_context(samples, vector_index);
    }

    let agent = builder.build();
    let max_turns = usize::try_from(analysis_config.max_turns)
        .map_err(|e| Error::Analysis(format!("max_turns overflow: {e}")))?;

    let mut stream = agent
        .stream_chat(user_message, history)
        .max_turns(max_turns)
        .await;

    let mut streamed = String::new();
    let mut final_text: Option<String> = None;

    while let Some(item) = stream.next().await {
        match item.map_err(|e| Error::Analysis(e.to_string()))? {
            rig::agent::MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::Text(text),
            ) => {
                streamed.push_str(&text.text);
                recorder.emit(AgentEvent::TextDelta(text.text));
            }
            rig::agent::MultiTurnStreamItem::FinalResponse(response) => {
                final_text = Some(response.output);
            }
            rig::agent::MultiTurnStreamItem::StreamAssistantItem(_)
            | rig::agent::MultiTurnStreamItem::StreamUserItem(_)
            | rig::agent::MultiTurnStreamItem::ToolExecutionCommitted { .. }
            | rig::agent::MultiTurnStreamItem::CompletionCall(_)
            | rig::agent::MultiTurnStreamItem::ModelTurnRetried { .. } => {}
        }
    }

    // Prefer what the model streamed for the last turn; fall back to the
    // aggregated final text when a provider did not stream deltas.
    let raw = match final_text {
        Some(text) if streamed.trim().is_empty() => text,
        Some(_) | None => streamed,
    };
    let (content, cited) = citations::validate(&raw, &recorder.citations().all());

    let chart = chart_spec
        .lock()
        .map_err(|e| Error::Analysis(format!("mutex poisoned: {e}")))?
        .take();

    Ok(AgentResponse {
        content,
        steps: recorder.steps(),
        citations: cited,
        chart,
        write_refused: refused.was_refused(),
    })
}
