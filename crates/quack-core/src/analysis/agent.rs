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
    CreateChartTool, DescribeTableTool, FindPathTool, GraphResults, ListDocumentsTool,
    ListTablesTool, RunSqlTool, SearchDocumentsTool, SearchGraphTool, SharedDb,
};
use super::vector_index::DuckDbVectorIndex;
use crate::graph::GraphResult;
use crate::storage::sessions::ChatMode;

/// Everything a turn produced, delivered with `AgentEvent::TurnComplete` and
/// returned from `run_analysis`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentResponse {
    pub content: String,
    pub steps: Vec<ToolStep>,
    /// Sources the answer cites, numbered as they appear in `content`.
    pub citations: Vec<Citation>,
    pub chart: Option<ChartSpec>,
    /// What the graph tools returned this turn, in call order.
    #[serde(default)]
    pub graph: Vec<GraphResult>,
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
    graph_options: crate::graph::GraphOptions,
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
        graph_options,
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
    graph_options: crate::graph::GraphOptions,
    write_policy: WritePolicy,
    prompt: PromptOptions,
    history: Vec<rig::message::Message>,
    user_message: &str,
    recorder: &TurnRecorder,
) -> Result<AgentResponse>
where
    M: rig::embeddings::EmbeddingModel + Clone + Send + Sync + 'static,
{
    let (system_prompt, graph_enabled) = {
        let db = shared_db
            .lock()
            .map_err(|e| Error::Analysis(format!("mutex poisoned: {e}")))?;
        (
            text_to_sql::build_system_prompt(&db, &prompt)?,
            crate::graph::store::status(&db)?.enabled(),
        )
    };
    let chart_spec: Arc<Mutex<Option<ChartSpec>>> = Arc::new(Mutex::new(None));
    let graph_results: GraphResults = Arc::new(Mutex::new(Vec::new()));
    let refused = RefusalFlag::default();
    // Query mode never answers from an unreviewed graph.
    let exclude_provisional = prompt.mode == ChatMode::Query;

    let agent = build_agent(
        completion_model,
        embedding_model,
        &system_prompt,
        &BuildContext {
            shared_db: Arc::clone(&shared_db),
            analysis_config,
            retrieval_config,
            graph_options,
            graph_enabled,
            exclude_provisional,
            write_policy,
            chart_spec: Arc::clone(&chart_spec),
            graph_results: Arc::clone(&graph_results),
            refused: refused.clone(),
            recorder: recorder.clone(),
        },
    )?;
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
    finish_turn(
        content,
        cited,
        &chart_spec,
        &graph_results,
        recorder,
        &refused,
    )
}

/// What `build_agent` needs besides the models and the prompt.
struct BuildContext<'a> {
    shared_db: SharedDb,
    analysis_config: &'a AnalysisConfig,
    retrieval_config: &'a RetrievalConfig,
    graph_options: crate::graph::GraphOptions,
    graph_enabled: bool,
    exclude_provisional: bool,
    write_policy: WritePolicy,
    chart_spec: Arc<Mutex<Option<ChartSpec>>>,
    graph_results: GraphResults,
    refused: RefusalFlag,
    recorder: TurnRecorder,
}

/// The rig agent with every tool this workspace and mode register.
fn build_agent<M>(
    completion_model: impl rig::completion::CompletionModel + Clone + 'static,
    embedding_model: M,
    system_prompt: &str,
    ctx: &BuildContext<'_>,
) -> Result<rig::agent::Agent>
where
    M: rig::embeddings::EmbeddingModel + Clone + Send + Sync + 'static,
{
    let search = search_tool(
        Arc::clone(&ctx.shared_db),
        &completion_model,
        embedding_model.clone(),
        ctx.retrieval_config,
        &ctx.recorder,
    );
    let mut builder = completion_model
        .into_agent_builder()
        .preamble(system_prompt)
        .tool(search)
        .tool(RunSqlTool::new(
            Arc::clone(&ctx.shared_db),
            ctx.analysis_config.max_query_rows,
            ctx.write_policy,
            ctx.refused.clone(),
            ctx.recorder.clone(),
        ))
        .tool(DescribeTableTool::new(
            Arc::clone(&ctx.shared_db),
            ctx.recorder.clone(),
        ))
        .tool(ListTablesTool::new(
            Arc::clone(&ctx.shared_db),
            ctx.recorder.clone(),
        ))
        .tool(ListDocumentsTool::new(
            Arc::clone(&ctx.shared_db),
            ctx.recorder.clone(),
        ))
        .tool(CreateChartTool::new(
            Arc::clone(&ctx.shared_db),
            Arc::clone(&ctx.chart_spec),
            ctx.recorder.clone(),
        ))
        .temperature(0.1);

    if ctx.graph_enabled {
        builder = builder
            .tool(SearchGraphTool::new(
                Arc::clone(&ctx.shared_db),
                Some(embedding_model.clone()),
                ctx.graph_options,
                ctx.exclude_provisional,
                Arc::clone(&ctx.graph_results),
                ctx.recorder.clone(),
            ))
            .tool(FindPathTool::new(
                Arc::clone(&ctx.shared_db),
                Some(embedding_model.clone()),
                ctx.graph_options,
                ctx.exclude_provisional,
                Arc::clone(&ctx.graph_results),
                ctx.recorder.clone(),
            ));
    }

    if ctx.retrieval_config.always_retrieve {
        let samples = usize::try_from(ctx.retrieval_config.top_k)
            .map_err(|e| Error::Analysis(format!("top_k overflow: {e}")))?;
        let vector_index = DuckDbVectorIndex::new(Arc::clone(&ctx.shared_db), embedding_model);
        builder = builder.dynamic_context(samples, vector_index);
    }

    Ok(builder.build())
}

/// Assemble the response once the stream has ended: the validated text,
/// the chart and graph results the tools left behind, and the steps.
fn finish_turn(
    content: String,
    citations: Vec<Citation>,
    chart_spec: &Arc<Mutex<Option<ChartSpec>>>,
    graph_results: &GraphResults,
    recorder: &TurnRecorder,
    refused: &RefusalFlag,
) -> Result<AgentResponse> {
    let chart = chart_spec
        .lock()
        .map_err(|e| Error::Analysis(format!("mutex poisoned: {e}")))?
        .take();
    let graph = std::mem::take(
        &mut *graph_results
            .lock()
            .map_err(|e| Error::Analysis(format!("mutex poisoned: {e}")))?,
    );
    Ok(AgentResponse {
        content,
        steps: recorder.steps(),
        citations,
        chart,
        graph,
        write_refused: refused.was_refused(),
    })
}
