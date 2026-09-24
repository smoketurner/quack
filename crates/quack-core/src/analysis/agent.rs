use std::sync::{Arc, Mutex, PoisonError};

use futures::StreamExt;
use rig::prelude::*;
use rig::streaming::StreamedAssistantContent;

use crate::config::{AnalysisConfig, RetrievalConfig};
use crate::embedding::Embedder;
use crate::error::{Error, Result};
use crate::ids::SessionId;
use crate::text::Tokens;

use super::chart::ChartSpec;
use super::citations::{Citation, CitedAnswer};
use super::events::{AgentEvent, EventSink, ToolName, ToolStep, TurnFailure, TurnRecorder};
use super::policy::{RefusalFlag, WritePolicy};
use super::text_to_sql::{Modeled, PromptOptions, SystemPrompt};
use super::tools::{
    CreateChartTool, DescribeClassTool, DescribeTableTool, FindPathTool, GraphResults, GraphTools,
    ListDocumentsTool, ListTablesTool, ReaderDb, RunSqlTool, SearchDocumentsTool, SearchGraphTool,
    SharedDb, ToolDeps, TurnSlot,
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
            .filter(|s| s.tool == ToolName::RunSql)
            .map(|s| QueryRun {
                sql: s.detail.clone(),
                rows: s.rows,
                duration_ms: s.duration_ms,
            })
            .collect()
    }

    /// The one response object every interface returns (design doc 11.2,
    /// issue #50): print mode's `--format json`, the REST body and SSE
    /// `complete` event, and the MCP structured content all carry this.
    #[must_use]
    pub fn to_json(&self, session_id: &SessionId) -> serde_json::Value {
        let citations: Vec<serde_json::Value> = self
            .citations
            .iter()
            .map(|c| {
                let mut value = serde_json::json!(c);
                if let Some(fields) = value.as_object_mut() {
                    fields.insert(String::from("label"), c.label().into());
                }
                value
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

/// One question for the agent: the workspace handles, the embedding model
/// (none for keyword-only search), the analysis settings, the write
/// policy, the prompt options, the history to replay (see
/// `storage::sessions::history_for_model`), and the message. `reader_db`
/// is the workspace handle's reader, built once for its whole lifetime by
/// [`ReaderDb::open`]: acquiring one per turn would make every turn wait
/// on the writer before it can start, exactly when a slow write is most
/// likely to be holding it.
pub struct Analysis<'a, M> {
    pub db: SharedDb,
    pub reader_db: ReaderDb,
    pub embedder: Option<Embedder<M>>,
    pub config: &'a AnalysisConfig,
    pub retrieval_config: &'a RetrievalConfig,
    pub graph_options: GraphOptions,
    pub write_policy: WritePolicy,
    pub prompt: PromptOptions,
    pub history: Vec<rig::message::Message>,
    pub message: &'a str,
}

impl<M> Analysis<'_, M>
where
    M: rig::embeddings::EmbeddingModel + Clone + Send + Sync + 'static,
{
    /// Run the rig agent with all analysis tools, emitting `AgentEvent`s
    /// on `sink` as the turn progresses.
    ///
    /// Text streams as `TextDelta`; every tool call is bracketed by
    /// `ToolStarted`/`ToolFinished`; a write under `WritePolicy::Ask`
    /// pauses on `PermissionRequired` until the interface answers. The
    /// final `TurnComplete` (or `Failed`) is also the return value.
    ///
    /// # Errors
    ///
    /// Returns an error if system prompt generation, agent building, or
    /// the model call fails.
    pub async fn run(
        self,
        completion_model: impl rig::completion::CompletionModel + Clone + 'static,
        sink: EventSink,
    ) -> Result<AgentResponse> {
        let max_turns = usize::try_from(self.config.max_turns)
            .map_err(|e| Error::Analysis(format!("max_turns overflow: {e}")))?;
        let recorder = TurnRecorder::new(sink).with_turn_limit(max_turns);
        match self.run_inner(completion_model, &recorder).await {
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
}

/// The system prompt and what the workspace models, read through the
/// turn's reader connection: the two decide which tools register.
struct PromptAndModel {
    system_prompt: String,
    modeled: Modeled,
}

impl PromptAndModel {
    async fn read(reader_db: &ReaderDb, prompt: &PromptOptions) -> Result<Self> {
        let prompt = prompt.clone();
        reader_db
            .with_db(move |db| {
                Ok(Self {
                    system_prompt: SystemPrompt::build(db, &prompt)?,
                    modeled: Modeled::of(
                        ontology_store::current(db)?.as_ref(),
                        &graph_store::status(db)?,
                    ),
                })
            })
            .await
    }
}

/// Who sizes the model's context window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Window {
    /// The provider sizes its own.
    Provider,
    /// Ollama, asked for this `num_ctx`.
    Ollama(OllamaWindow),
}

/// The `num_ctx` to ask Ollama for: the prompt's estimated tokens plus
/// room for tool results and the answer, rounded up to 8,192, between
/// 8,192 and `cap`. Ollama's default of 4,096 truncates the front of
/// most workspace prompts, which loses the tool guidance and the question.
///
/// `num_ctx` is a load option: asking Ollama for a different value than
/// the one the model is already loaded with forces a full model reload,
/// which measured 4-5 seconds for `gpt-oss:20b` on this machine (`ollama
/// serve`, repeated `/api/generate` calls that only changed `num_ctx`) —
/// against single-digit milliseconds for a request that keeps the same
/// value. A session's history only grows turn over turn until the
/// history trim caps it, so the requested size is non-decreasing within
/// a session; the step below is deliberately coarse (four tiers instead
/// of one every 2,048 tokens) so a growing conversation crosses it, and
/// pays that reload, at most three times instead of up to twelve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OllamaWindow(u32);

impl OllamaWindow {
    const HEADROOM: u32 = 8_192;
    const FLOOR: u32 = 8_192;
    const STEP: u32 = 8_192;

    /// The window for a turn: the system prompt, the replayed history, and
    /// the question, under `cap` (`[analysis].max_context_tokens`). Ollama
    /// loads a model with a 4,096-token window unless the request says
    /// otherwise and truncates the front of a longer prompt, which is where
    /// the tool guidance is.
    fn for_turn(
        cap: u32,
        system_prompt: &str,
        history: &[rig::message::Message],
        user_message: &str,
    ) -> Self {
        let history_chars = serde_json::to_string(history).map_or(0, |h| h.len());
        let prompt = Tokens::of_chars(
            system_prompt
                .len()
                .saturating_add(history_chars)
                .saturating_add(user_message.len()),
        );
        if prompt.get() > cap {
            tracing::warn!(
                prompt_tokens = %prompt,
                cap,
                "the prompt is larger than [analysis].max_context_tokens; Ollama will truncate it"
            );
        }
        Self::for_prompt(prompt, cap)
    }

    /// The window for a prompt of `prompt` tokens under `cap`.
    fn for_prompt(prompt: Tokens, cap: u32) -> Self {
        let needed = prompt.get().saturating_add(Self::HEADROOM);
        let rounded = needed
            .div_ceil(Self::STEP)
            .saturating_mul(Self::STEP)
            .max(Self::FLOOR);
        Self(rounded.min(cap.max(Self::FLOOR)))
    }
}

impl<M> Analysis<'_, M>
where
    M: rig::embeddings::EmbeddingModel + Clone + Send + Sync + 'static,
{
    async fn run_inner(
        self,
        completion_model: impl rig::completion::CompletionModel + Clone + 'static,
        recorder: &TurnRecorder,
    ) -> Result<AgentResponse> {
        let Self {
            db: shared_db,
            reader_db,
            embedder: embedding_model,
            config: analysis_config,
            retrieval_config,
            graph_options,
            write_policy,
            prompt,
            history,
            message: user_message,
        } = self;
        let PromptAndModel {
            system_prompt,
            modeled,
        } = PromptAndModel::read(&reader_db, &prompt).await?;
        let outputs = TurnOutputs::new(recorder.clone());
        let window = prompt.ollama_context_cap.map_or(Window::Provider, |cap| {
            Window::Ollama(OllamaWindow::for_turn(
                cap,
                &system_prompt,
                &history,
                user_message,
            ))
        });
        let agent = BuildContext {
            shared_db: Arc::clone(&shared_db),
            reader_db,
            analysis_config,
            retrieval_config,
            graph_options,
            modeled,
            mode: prompt.mode,
            write_policy,
            window,
            outputs: outputs.clone(),
        }
        .build_agent(completion_model, embedding_model, &system_prompt)?;
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
                    let stop = StreamStop(&e);
                    if streamed.trim().is_empty() && !stop.by_agent_loop() {
                        return Err(Error::Analysis(e.to_string()));
                    }
                    tracing::warn!(error = %e, "agent turn stopped early");
                    stopped = Some(stop.explain(analysis_config.max_turns, window));
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

        let raw = turn_text(streamed, final_text, stopped, window);
        Ok(outputs.finish(
            recorder.citations().validate(&raw),
            aggregate.or_else(|| per_call.reported()),
        ))
    }
}

/// The answer text: what the model streamed for the last turn, else the
/// aggregated final text when a provider did not stream deltas, with a
/// note when the turn stopped early or produced nothing.
fn turn_text(
    streamed: String,
    final_text: Option<String>,
    stopped: Option<String>,
    window: Window,
) -> String {
    let mut raw = match final_text {
        Some(text) if streamed.trim().is_empty() => text,
        Some(_) | None => streamed,
    };
    let note = stopped.or_else(|| {
        raw.trim().is_empty().then(|| {
            String::from(if matches!(window, Window::Ollama(_)) {
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

/// Why a turn's stream ended early.
struct StreamStop<'a>(&'a rig::agent::StreamingError);

impl StreamStop<'_> {
    /// Whether the agent loop itself stopped it (rig's `PromptError`: an
    /// unknown tool, the turn limit) rather than the provider call failing.
    const fn by_agent_loop(&self) -> bool {
        matches!(self.0, rig::agent::StreamingError::Prompt(_))
    }

    /// A user-facing sentence, for a stop worth keeping the turn for.
    fn explain(&self, max_turns: u32, window: Window) -> String {
        use rig::completion::PromptError;
        match self.0 {
            rig::agent::StreamingError::Prompt(prompt_error) => match prompt_error.as_ref() {
                PromptError::UnknownToolCall { tool_name, .. } => format!(
                    "The model called a tool that does not exist ({tool_name}), so the turn \
                     stopped.{}",
                    match window {
                        Window::Ollama(_) => {
                            " With Ollama this usually means the prompt was cut to the context \
                             window; check [analysis].max_context_tokens and the model's own \
                             limit."
                        }
                        Window::Provider => "",
                    }
                ),
                PromptError::MaxTurnsError { .. } => format!(
                    "The turn reached the limit of {max_turns} tool calls ([analysis].max_turns) \
                     before the model answered."
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
}

/// What the tools leave for the response, and the turn's record of steps.
#[derive(Clone)]
struct TurnOutputs {
    chart: TurnSlot<ChartSpec>,
    graph: GraphResults,
    refused: RefusalFlag,
    recorder: TurnRecorder,
}

impl TurnOutputs {
    fn new(recorder: TurnRecorder) -> Self {
        Self {
            chart: TurnSlot::default(),
            graph: Arc::new(Mutex::new(Vec::new())),
            refused: RefusalFlag::default(),
            recorder,
        }
    }

    /// The response once the stream has ended: the checked answer, the
    /// chart and graph results the tools left behind, and the steps.
    fn finish(&self, answer: CitedAnswer, usage: Option<TokenUsage>) -> AgentResponse {
        let graph = std::mem::take(&mut *self.graph.lock().unwrap_or_else(PoisonError::into_inner));
        AgentResponse {
            content: answer.text,
            steps: self.recorder.steps(),
            citations: answer.citations,
            chart: self.chart.take(),
            graph,
            write_refused: self.refused.was_refused(),
            cancelled: false,
            usage,
        }
    }
}

/// What building the agent needs besides the models and the prompt.
struct BuildContext<'a> {
    /// The writer connection: only `run_sql` gets it, since it may write.
    shared_db: SharedDb,
    /// A reader connection for every tool that only reads.
    reader_db: ReaderDb,
    analysis_config: &'a AnalysisConfig,
    retrieval_config: &'a RetrievalConfig,
    graph_options: GraphOptions,
    modeled: Modeled,
    mode: ChatMode,
    write_policy: WritePolicy,
    window: Window,
    outputs: TurnOutputs,
}

impl BuildContext<'_> {
    /// The rig agent with every tool this workspace and mode register.
    fn build_agent<M>(
        &self,
        completion_model: impl rig::completion::CompletionModel + Clone + 'static,
        embedding_model: Option<Embedder<M>>,
        system_prompt: &str,
    ) -> Result<rig::agent::Agent>
    where
        M: rig::embeddings::EmbeddingModel + Clone + Send + Sync + 'static,
    {
        let ctx = self;
        let search = SearchDocumentsTool::from_config(
            ctx.reader_db.clone(),
            &completion_model,
            embedding_model.clone(),
            ctx.retrieval_config,
            ctx.outputs.recorder.clone(),
        )
        .with_model(ctx.modeled);
        let deps = ToolDeps {
            db: ctx.reader_db.clone(),
            recorder: ctx.outputs.recorder.clone(),
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
                ctx.outputs.refused.clone(),
                ctx.outputs.recorder.clone(),
            ))
            .tool(DescribeTableTool(deps.clone()))
            .tool(ListTablesTool(deps.clone()))
            .tool(ListDocumentsTool(deps.clone()))
            .tool(CreateChartTool::new(
                ctx.reader_db.clone(),
                ctx.outputs.chart.clone(),
                ctx.outputs.recorder.clone(),
            ))
            .temperature(0.1);
        if let Window::Ollama(OllamaWindow(num_ctx)) = ctx.window {
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
        if ctx.modeled.has_ontology() {
            builder = builder.tool(DescribeClassTool(deps));
        }

        if ctx.modeled.has_graph() {
            let graph = GraphTools {
                db: ctx.reader_db.clone(),
                embedding_model: embedding_model.clone(),
                options: ctx.graph_options,
                mode: ctx.mode,
                results: Arc::clone(&ctx.outputs.graph),
                recorder: ctx.outputs.recorder.clone(),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ollama_window_rounds_up_within_bounds() {
        let window = |tokens, cap| OllamaWindow::for_prompt(Tokens::new(tokens), cap).0;
        assert_eq!(window(0, 32_768), 8_192);
        assert_eq!(window(1_000, 32_768), 16_384);
        // 12,875 prompt tokens plus headroom rounds to 24,576.
        assert_eq!(window(12_875, 32_768), 24_576);
        assert_eq!(window(100_000, 32_768), 32_768);
        assert_eq!(window(100_000, 2_048), 8_192);
    }
    use crate::ids::{ChunkId, DocumentId};
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
        assert!(StreamStop(&unknown).by_agent_loop());
        let text =
            StreamStop(&unknown).explain(config.max_turns, Window::Ollama(OllamaWindow(8_192)));
        assert!(text.contains("container.exec"), "{text}");
        assert!(text.contains("max_context_tokens"), "{text}");
        let text = StreamStop(&unknown).explain(config.max_turns, Window::Provider);
        assert!(!text.contains("Ollama"), "{text}");

        let limit = prompt_error(PromptError::MaxTurnsError {
            max_turns: 10,
            chat_history: Box::new(Vec::new()),
            prompt: Box::new(rig::message::Message::user("q")),
        });
        let text = StreamStop(&limit).explain(config.max_turns, Window::Provider);
        assert!(
            text.contains(&format!("{} tool calls", config.max_turns)),
            "{text}"
        );

        let provider = rig::agent::StreamingError::Completion(
            rig::completion::CompletionError::ProviderError(String::from("connection refused")),
        );
        assert!(!StreamStop(&provider).by_agent_loop());
        let text = StreamStop(&provider).explain(config.max_turns, Window::Provider);
        assert!(text.contains("connection refused"), "{text}");
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "test asserts fixed keys")]
    fn to_json_carries_every_field_and_derives_queries() {
        let response = AgentResponse {
            content: String::from("12 storms [1]"),
            steps: vec![
                ToolStep {
                    tool: ToolName::RunSql,
                    detail: String::from("SELECT count(*) FROM events"),
                    summary: String::from("the summary is not parsed"),
                    rows: Some(1),
                    duration_ms: 7,
                },
                ToolStep {
                    tool: ToolName::SearchDocuments,
                    detail: String::from("storms"),
                    summary: String::from("3 chunks"),
                    rows: None,
                    duration_ms: 4,
                },
            ],
            citations: vec![Citation {
                n: 1,
                chunk_id: ChunkId::from("c"),
                document_id: DocumentId::from("d"),
                filename: String::from("noaa.pdf"),
                chunk_index: 2,
                page: Some(4),
                heading: None,
            }],
            ..AgentResponse::default()
        };
        let json = response.to_json(&SessionId::from("s1"));
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
        let json = counted.to_json(&SessionId::from("s1"));
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
                Window::Provider
            ),
            "so far\n\n(why)"
        );
        assert_eq!(
            turn_text(
                String::new(),
                Some(String::from("final")),
                None,
                Window::Provider
            ),
            "final"
        );
        let empty = turn_text(
            String::new(),
            Some(String::new()),
            None,
            Window::Ollama(OllamaWindow(8_192)),
        );
        assert!(empty.contains("max_context_tokens"), "{empty}");
        let empty = turn_text(String::new(), None, None, Window::Provider);
        assert!(!empty.contains("Ollama"), "{empty}");
    }
}
