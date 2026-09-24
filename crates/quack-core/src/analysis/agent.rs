use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::prelude::*;
use rig::streaming::StreamedAssistantContent;

use crate::config::{AnalysisConfig, RerankMode, RetrievalConfig};
use crate::embedding::Embedder;
use crate::error::{Error, Result};

use super::chart::ChartSpec;
use super::citations::{self, Citation};
use super::events::{AgentEvent, EventSink, ToolStep, TurnFailure, TurnRecorder};
use super::policy::{RefusalFlag, WritePolicy};
use super::rerank::ModelReranker;
use super::text_to_sql::{self, PromptOptions};
use super::tools::{
    CreateChartTool, DescribeClassTool, DescribeTableTool, FindPathTool, GraphResults, GraphTools,
    ListDocumentsTool, ListTablesTool, ReaderDb, RunSqlTool, SearchDocumentsTool, SearchGraphTool,
    SharedDb, ToolDeps,
};
use super::vector_index::DuckDbVectorIndex;
use crate::graph::{GraphOptions, GraphResult, store as graph_store};
use crate::llm::OLLAMA_KEEP_ALIVE;
use crate::ontology::store as ontology_store;
use crate::storage::sessions::ChatMode;

/// What the provider charged for a turn. Every budget quack computes
/// itself — the history trim, Ollama's `num_ctx` — is a four-characters-
/// per-token estimate; this is the measured count the provider reported,
/// for the response object and the transcript.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[expect(
    clippy::struct_field_names,
    reason = "these are the field names of rig's Usage and of every provider's API, and they are the response object's JSON keys"
)]
pub struct TokenUsage {
    /// Prompt tokens: the system prompt, the replayed history, the tool
    /// results, and the question, summed over every completion request
    /// the turn made.
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Kept separate because some providers report only this one.
    pub total_tokens: u64,
}

impl From<rig::completion::Usage> for TokenUsage {
    fn from(usage: rig::completion::Usage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

impl TokenUsage {
    /// Add one completion request's counts, for the turns that never reach
    /// a final response.
    fn add(&mut self, usage: rig::completion::Usage) {
        self.input_tokens = self.input_tokens.saturating_add(usage.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(usage.output_tokens);
        self.total_tokens = self.total_tokens.saturating_add(usage.total_tokens);
    }

    /// The counts, unless every one is zero — rig's sentinel for a provider
    /// that reported no usage at all.
    fn reported(self) -> Option<Self> {
        (self != Self::default()).then_some(self)
    }
}

/// Everything a turn produced, delivered with `AgentEvent::TurnComplete` and
/// returned from `run_analysis`.
#[derive(Debug, Clone, Default, serde::Serialize)]
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
    /// The user cancelled the turn; `content` holds what streamed before.
    #[serde(default)]
    pub cancelled: bool,
    /// Tokens the provider reported for the turn. `None` when it reported
    /// none — a local model often does, and zeroes there would read as a
    /// turn that cost nothing.
    #[serde(default)]
    pub usage: Option<TokenUsage>,
}

/// One SQL statement the turn ran, as the response object lists it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct QueryRun {
    pub sql: String,
    /// Rows the statement produced, when the step reported a count.
    pub rows: Option<u64>,
    pub duration_ms: u64,
}

impl AgentResponse {
    /// The `run_sql` steps as statements with their row counts.
    #[must_use]
    pub fn queries(&self) -> Vec<QueryRun> {
        self.steps
            .iter()
            .filter(|s| s.tool == "run_sql")
            .map(|s| QueryRun {
                sql: s.detail.clone(),
                rows: s
                    .summary
                    .split_whitespace()
                    .next()
                    .and_then(|n| n.parse::<u64>().ok()),
                duration_ms: s.duration_ms,
            })
            .collect()
    }

    /// The one response object every interface returns (design doc 11.2,
    /// issue #50): print mode's `--format json`, the REST body and SSE
    /// `complete` event, and the MCP structured content all carry this.
    #[must_use]
    pub fn to_json(&self, session_id: &str) -> serde_json::Value {
        let citations: Vec<serde_json::Value> = self
            .citations
            .iter()
            .map(|c| {
                serde_json::json!({
                    "n": c.n,
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "filename": c.filename,
                    "chunk_index": c.chunk_index,
                    "page": c.page,
                    "heading": c.heading,
                    "label": c.label(),
                })
            })
            .collect();
        serde_json::json!({
            "answer": self.content,
            "citations": citations,
            "queries": self.queries(),
            "steps": self.steps,
            "graph": self.graph,
            "chart": self.chart,
            "write_refused": self.write_refused,
            "cancelled": self.cancelled,
            "usage": self.usage,
            "session_id": session_id,
        })
    }
}

/// Run the rig agent with all analysis tools for a single user question,
/// emitting `AgentEvent`s on `sink` as the turn progresses. `history` is
/// the prior conversation to replay to the model (see
/// `storage::sessions::history_for_model`). `reader_db` is the workspace
/// handle's reader, built once for its whole lifetime by
/// [`super::tools::open_reader`] — acquiring one here would mean every turn
/// waits on the writer mutex before it can even start, exactly when a slow
/// write is most likely to be holding it.
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
    reader_db: ReaderDb,
    completion_model: impl rig::completion::CompletionModel + Clone + 'static,
    embedding_model: Option<Embedder<M>>,
    analysis_config: &AnalysisConfig,
    retrieval_config: &RetrievalConfig,
    graph_options: GraphOptions,
    write_policy: WritePolicy,
    prompt: PromptOptions,
    history: Vec<rig::message::Message>,
    user_message: &str,
    sink: EventSink,
) -> Result<AgentResponse>
where
    M: rig::embeddings::EmbeddingModel + Clone + Send + Sync + 'static,
{
    let max_turns = usize::try_from(analysis_config.max_turns)
        .map_err(|e| Error::Analysis(format!("max_turns overflow: {e}")))?;
    let recorder = TurnRecorder::new(sink).with_turn_limit(max_turns);
    match run_inner(
        db,
        reader_db,
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
            recorder.emit(AgentEvent::Failed(TurnFailure::from(&e)));
            Err(e)
        }
    }
}

/// The `search_documents` tool, with the chat model as reranker when
/// `[retrieval].rerank = "model"`.
fn search_tool<M>(
    reader_db: ReaderDb,
    completion_model: &(impl rig::completion::CompletionModel + Clone + 'static),
    embedding_model: Option<Embedder<M>>,
    retrieval_config: &RetrievalConfig,
    recorder: &TurnRecorder,
) -> SearchDocumentsTool<M> {
    let search = SearchDocumentsTool::new(
        reader_db,
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

/// The system prompt and what the workspace models, read through the
/// turn's reader connection: the two decide which tools register.
struct PromptAndModel {
    system_prompt: String,
    /// The graph has nodes, so `search_graph` and `find_path` can answer.
    graph_enabled: bool,
    /// An ontology exists, so `describe_class` has something to describe —
    /// true even before anything has been extracted into the graph.
    ontology_present: bool,
}

async fn system_prompt_and_model(
    reader_db: &ReaderDb,
    prompt: &PromptOptions,
) -> Result<PromptAndModel> {
    let prompt_for_db = prompt.clone();
    reader_db
        .with_db(move |db| {
            Ok(PromptAndModel {
                system_prompt: text_to_sql::build_system_prompt(db, &prompt_for_db)?,
                graph_enabled: graph_store::status(db)?.enabled(),
                ontology_present: ontology_store::current(db)?.is_some(),
            })
        })
        .await
}

#[expect(clippy::too_many_arguments, reason = "mirrors run_analysis")]
async fn run_inner<M>(
    shared_db: SharedDb,
    reader_db: ReaderDb,
    completion_model: impl rig::completion::CompletionModel + Clone + 'static,
    embedding_model: Option<Embedder<M>>,
    analysis_config: &AnalysisConfig,
    retrieval_config: &RetrievalConfig,
    graph_options: GraphOptions,
    write_policy: WritePolicy,
    prompt: PromptOptions,
    history: Vec<rig::message::Message>,
    user_message: &str,
    recorder: &TurnRecorder,
) -> Result<AgentResponse>
where
    M: rig::embeddings::EmbeddingModel + Clone + Send + Sync + 'static,
{
    let PromptAndModel {
        system_prompt,
        graph_enabled,
        ontology_present,
    } = system_prompt_and_model(&reader_db, &prompt).await?;
    let chart_spec: Arc<Mutex<Option<ChartSpec>>> = Arc::new(Mutex::new(None));
    let graph_results: GraphResults = Arc::new(Mutex::new(Vec::new()));
    let refused = RefusalFlag::default();
    // Query mode never answers from an unreviewed graph.
    let exclude_provisional = prompt.mode == ChatMode::Query;

    let context_window = prompt
        .ollama_context_cap
        .map(|cap| ollama_window(cap, &system_prompt, &history, user_message));
    let agent = build_agent(
        completion_model,
        embedding_model,
        &system_prompt,
        &BuildContext {
            shared_db: Arc::clone(&shared_db),
            reader_db,
            analysis_config,
            retrieval_config,
            graph_options,
            graph_enabled,
            ontology_present,
            exclude_provisional,
            write_policy,
            context_window,
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
    let mut stopped: Option<String> = None;
    // The final response carries rig's aggregate for the whole run; the
    // per-call counts are the fallback for a turn that derails before it,
    // which is exactly the turn whose cost is worth knowing.
    let mut aggregate: Option<TokenUsage> = None;
    let mut per_call = TokenUsage::default();

    while let Some(item) = stream.next().await {
        let item = match item {
            Ok(item) => item,
            Err(e) => {
                // A turn the model derailed (an unknown tool, the turn
                // limit) or that failed after text was streamed is still
                // a turn: keep the text, say what happened, record it. A
                // model that could not be reached at all stays an error.
                if streamed.trim().is_empty() && !is_prompt_error(&e) {
                    return Err(Error::Analysis(e.to_string()));
                }
                tracing::warn!(error = %e, "agent turn stopped early");
                stopped = Some(explain_stream_error(
                    &e,
                    analysis_config,
                    context_window.is_some(),
                ));
                break;
            }
        };
        match item {
            rig::agent::MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::Text(text),
            ) => {
                streamed.push_str(&text.text);
                recorder.emit(AgentEvent::TextDelta(text.text));
            }
            rig::agent::MultiTurnStreamItem::FinalResponse(response) => {
                if response.usage.has_values() {
                    aggregate = Some(response.usage.into());
                }
                final_text = Some(response.output);
            }
            rig::agent::MultiTurnStreamItem::CompletionCall(call) => per_call.add(call.usage),
            rig::agent::MultiTurnStreamItem::StreamAssistantItem(_)
            | rig::agent::MultiTurnStreamItem::StreamUserItem(_)
            | rig::agent::MultiTurnStreamItem::ToolExecutionCommitted { .. }
            | rig::agent::MultiTurnStreamItem::ModelTurnRetried { .. } => {}
        }
    }

    let raw = turn_text(streamed, final_text, stopped, context_window.is_some());
    let (content, cited) = citations::validate(&raw, &recorder.citations().all());
    finish_turn(
        content,
        cited,
        &chart_spec,
        &graph_results,
        recorder,
        &refused,
        aggregate.or_else(|| per_call.reported()),
    )
}

/// The `num_ctx` for this turn (issue #40): Ollama loads a model with a
/// 4,096-token window unless the request says otherwise and truncates
/// the front of a longer prompt, which is where the tool guidance is.
fn ollama_window(
    cap: u32,
    system_prompt: &str,
    history: &[rig::message::Message],
    user_message: &str,
) -> u32 {
    let history_chars = serde_json::to_string(history).map_or(0, |h| h.len());
    let chars = system_prompt
        .len()
        .saturating_add(history_chars)
        .saturating_add(user_message.len());
    let prompt_tokens = chars.div_ceil(4);
    if prompt_tokens > cap as usize {
        tracing::warn!(
            prompt_tokens,
            cap,
            "the prompt is larger than [analysis].max_context_tokens; Ollama will truncate it"
        );
    }
    text_to_sql::ollama_context_size(chars, cap)
}

/// The answer text: what the model streamed for the last turn, else the
/// aggregated final text when a provider did not stream deltas, with a
/// note when the turn stopped early or produced nothing.
fn turn_text(
    streamed: String,
    final_text: Option<String>,
    stopped: Option<String>,
    ollama: bool,
) -> String {
    let mut raw = match final_text {
        Some(text) if streamed.trim().is_empty() => text,
        Some(_) | None => streamed,
    };
    let note = stopped.or_else(|| {
        raw.trim().is_empty().then(|| {
            String::from(if ollama {
                "The model returned no text. With Ollama this usually means the answer or the \
                 prompt did not fit the context window; raise [analysis].max_context_tokens or \
                 ask a narrower question."
            } else {
                "The model returned no text; ask again or narrow the question."
            })
        })
    });
    if let Some(reason) = note {
        if !raw.trim().is_empty() {
            raw.push_str("\n\n");
        }
        raw.push('(');
        raw.push_str(&reason);
        raw.push(')');
    }
    raw
}

/// Whether a stream error came from the agent loop itself (rig's
/// `PromptError`: an unknown tool, the turn limit) rather than from the
/// provider call.
fn is_prompt_error(error: &rig::agent::StreamingError) -> bool {
    matches!(error, rig::agent::StreamingError::Prompt(_))
}

/// A user-facing sentence for a stream error worth keeping the turn for.
fn explain_stream_error(
    error: &rig::agent::StreamingError,
    analysis_config: &AnalysisConfig,
    ollama: bool,
) -> String {
    use rig::completion::PromptError;
    match error {
        rig::agent::StreamingError::Prompt(prompt_error) => match prompt_error.as_ref() {
            PromptError::UnknownToolCall { tool_name, .. } => format!(
                "The model called a tool that does not exist ({tool_name}), so the turn \
                 stopped.{}",
                if ollama {
                    " With Ollama this usually means the prompt was cut to the context window; \
                     check [analysis].max_context_tokens and the model's own limit."
                } else {
                    ""
                }
            ),
            PromptError::MaxTurnsError { .. } => format!(
                "The turn reached the limit of {} tool calls ([analysis].max_turns) before \
                 the model answered.",
                analysis_config.max_turns
            ),
            PromptError::PromptCancelled { reason, .. } => {
                format!("The turn was cancelled: {reason}")
            }
            PromptError::CompletionError(e) => format!("The model call failed: {e}"),
            PromptError::MemoryError(e) => format!("The turn failed: {e}"),
        },
        rig::agent::StreamingError::Completion(e) => {
            format!("The model call failed part way through: {e}")
        }
    }
}

/// What `build_agent` needs besides the models and the prompt.
struct BuildContext<'a> {
    /// The writer connection: only `run_sql` gets it, since it may write.
    shared_db: SharedDb,
    /// A reader connection for every tool that only reads.
    reader_db: ReaderDb,
    analysis_config: &'a AnalysisConfig,
    retrieval_config: &'a RetrievalConfig,
    graph_options: GraphOptions,
    graph_enabled: bool,
    ontology_present: bool,
    exclude_provisional: bool,
    write_policy: WritePolicy,
    /// `num_ctx` for Ollama; `None` for other providers.
    context_window: Option<u32>,
    chart_spec: Arc<Mutex<Option<ChartSpec>>>,
    graph_results: GraphResults,
    refused: RefusalFlag,
    recorder: TurnRecorder,
}

/// The rig agent with every tool this workspace and mode register.
fn build_agent<M>(
    completion_model: impl rig::completion::CompletionModel + Clone + 'static,
    embedding_model: Option<Embedder<M>>,
    system_prompt: &str,
    ctx: &BuildContext<'_>,
) -> Result<rig::agent::Agent>
where
    M: rig::embeddings::EmbeddingModel + Clone + Send + Sync + 'static,
{
    let search = search_tool(
        ctx.reader_db.clone(),
        &completion_model,
        embedding_model.clone(),
        ctx.retrieval_config,
        &ctx.recorder,
    )
    .with_graph(ctx.graph_enabled);
    let deps = ToolDeps {
        db: ctx.reader_db.clone(),
        recorder: ctx.recorder.clone(),
    };
    let mut builder = completion_model
        .into_agent_builder()
        .preamble(system_prompt)
        .tool(search)
        .tool(RunSqlTool::new(
            Arc::clone(&ctx.shared_db),
            ctx.reader_db.clone(),
            ctx.analysis_config.max_query_rows,
            ctx.write_policy,
            ctx.refused.clone(),
            ctx.recorder.clone(),
        ))
        .tool(DescribeTableTool(deps.clone()))
        .tool(ListTablesTool(deps.clone()))
        .tool(ListDocumentsTool(deps.clone()))
        .tool(CreateChartTool::new(
            ctx.reader_db.clone(),
            Arc::clone(&ctx.chart_spec),
            ctx.recorder.clone(),
        ))
        .temperature(0.1);
    if let Some(num_ctx) = ctx.context_window {
        // `keep_alive` is Ollama-only too (rig lifts it out of
        // `additional_params` into the request's top-level field, never
        // into `options`). Nothing was setting it, so every request fell
        // back to Ollama's own default (`OLLAMA_KEEP_ALIVE`, 5 minutes
        // unless the operator changed it) each time it decided whether to
        // keep the model loaded. A turn with several tool calls, or an
        // idle stretch between turns in a TUI or web session, can leave a
        // gap longer than that, which pays a multi-second reload the same
        // way a changed `num_ctx` does (measured live, both in the perf
        // handoff). Sending it explicitly on every request keeps the
        // model warm through longer gaps regardless of the server's
        // default.
        builder = builder.additional_params(
            serde_json::json!({ "num_ctx": num_ctx, "keep_alive": OLLAMA_KEEP_ALIVE }),
        );
    }

    // The ontology is describable as soon as it exists: the prompt block
    // is capped, so a class the model wants the detail of may not be in it
    // even when nothing has been extracted into the graph yet.
    if ctx.ontology_present {
        builder = builder.tool(DescribeClassTool(deps));
    }

    if ctx.graph_enabled {
        let graph = GraphTools {
            db: ctx.reader_db.clone(),
            embedding_model: embedding_model.clone(),
            options: ctx.graph_options,
            exclude_provisional: ctx.exclude_provisional,
            results: Arc::clone(&ctx.graph_results),
            recorder: ctx.recorder.clone(),
        };
        builder = builder
            .tool(SearchGraphTool(graph.clone()))
            .tool(FindPathTool(graph));
    }

    if ctx.retrieval_config.always_retrieve
        && let Some(embedding_model) = embedding_model
    {
        let samples = usize::try_from(ctx.retrieval_config.top_k)
            .map_err(|e| Error::Analysis(format!("top_k overflow: {e}")))?;
        let vector_index = DuckDbVectorIndex::new(ctx.reader_db.clone(), embedding_model);
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
    usage: Option<TokenUsage>,
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
        cancelled: false,
        usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::completion::PromptError;

    fn prompt_error(e: PromptError) -> rig::agent::StreamingError {
        rig::agent::StreamingError::Prompt(Box::new(e))
    }

    #[test]
    fn stream_errors_are_explained_for_the_user() {
        let config = AnalysisConfig::default();
        let unknown = prompt_error(PromptError::UnknownToolCall {
            tool_name: String::from("container.exec"),
            available_tools: vec![String::from("run_sql")],
            allowed_tools: vec![String::from("run_sql")],
            chat_history: Box::new(Vec::new()),
        });
        assert!(is_prompt_error(&unknown));
        let text = explain_stream_error(&unknown, &config, true);
        assert!(text.contains("container.exec"), "{text}");
        assert!(text.contains("max_context_tokens"), "{text}");
        let text = explain_stream_error(&unknown, &config, false);
        assert!(!text.contains("Ollama"), "{text}");

        let limit = prompt_error(PromptError::MaxTurnsError {
            max_turns: 10,
            chat_history: Box::new(Vec::new()),
            prompt: Box::new(rig::message::Message::user("q")),
        });
        let text = explain_stream_error(&limit, &config, false);
        assert!(
            text.contains(&format!("{} tool calls", config.max_turns)),
            "{text}"
        );

        let provider = rig::agent::StreamingError::Completion(
            rig::completion::CompletionError::ProviderError(String::from("connection refused")),
        );
        assert!(!is_prompt_error(&provider));
        let text = explain_stream_error(&provider, &config, false);
        assert!(text.contains("connection refused"), "{text}");
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "test asserts fixed keys")]
    fn to_json_carries_every_field_and_derives_queries() {
        let response = AgentResponse {
            content: String::from("12 storms [1]"),
            steps: vec![
                ToolStep {
                    tool: String::from("run_sql"),
                    detail: String::from("SELECT count(*) FROM events"),
                    summary: String::from("1 rows"),
                    duration_ms: 7,
                },
                ToolStep {
                    tool: String::from("search_documents"),
                    detail: String::from("storms"),
                    summary: String::from("3 chunks"),
                    duration_ms: 4,
                },
            ],
            citations: vec![Citation {
                n: 1,
                chunk_id: String::from("c"),
                document_id: String::from("d"),
                filename: String::from("noaa.pdf"),
                chunk_index: 2,
                page: Some(4),
                heading: None,
            }],
            ..AgentResponse::default()
        };
        let json = response.to_json("s1");
        assert_eq!(json["answer"], "12 storms [1]");
        assert_eq!(json["queries"][0]["sql"], "SELECT count(*) FROM events");
        assert_eq!(json["queries"][0]["rows"], 1);
        assert_eq!(json["queries"].as_array().map(Vec::len), Some(1));
        assert_eq!(json["citations"][0]["label"], "noaa.pdf, page 4");
        assert_eq!(json["citations"][0]["chunk_id"], "c");
        assert_eq!(json["session_id"], "s1");
        assert_eq!(json["write_refused"], false);
        assert_eq!(json["cancelled"], false);
        assert!(json["graph"].is_array() && json["chart"].is_null());
        // A provider that reported nothing leaves `usage` null rather than
        // claiming the turn was free.
        assert!(json["usage"].is_null());

        let counted = AgentResponse {
            usage: Some(TokenUsage {
                input_tokens: 980,
                output_tokens: 43,
                total_tokens: 1_023,
            }),
            ..AgentResponse::default()
        };
        let json = counted.to_json("s1");
        assert_eq!(json["usage"]["input_tokens"], 980);
        assert_eq!(json["usage"]["output_tokens"], 43);
        assert_eq!(json["usage"]["total_tokens"], 1_023);
    }

    #[test]
    fn per_call_usage_accumulates_across_a_turns_completion_requests() {
        let call = |input: u64, output: u64| rig::completion::Usage {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input.saturating_add(output),
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        };
        let mut usage = TokenUsage::default();
        usage.add(call(400, 20));
        usage.add(call(650, 35));
        assert_eq!(
            usage,
            TokenUsage {
                input_tokens: 1_050,
                output_tokens: 55,
                total_tokens: 1_105,
            }
        );
        assert_eq!(usage.reported(), Some(usage));
        // A provider that reports nothing leaves the accumulator at its
        // default, which `run_inner` reads as "no counts", not zero cost.
        let mut none = TokenUsage::default();
        none.add(call(0, 0));
        assert_eq!(none, TokenUsage::default());
        assert_eq!(none.reported(), None);
    }

    #[test]
    fn turn_text_keeps_streamed_text_and_notes_early_stops() {
        assert_eq!(
            turn_text(
                String::from("so far"),
                None,
                Some(String::from("why")),
                false
            ),
            "so far\n\n(why)"
        );
        assert_eq!(
            turn_text(String::new(), Some(String::from("final")), None, false),
            "final"
        );
        let empty = turn_text(String::new(), Some(String::new()), None, true);
        assert!(empty.contains("max_context_tokens"), "{empty}");
        let empty = turn_text(String::new(), None, None, false);
        assert!(!empty.contains("Ollama"), "{empty}");
    }
}
