use std::sync::{Arc, PoisonError};
use std::time::Instant;

use futures::StreamExt;
use rig::completion::PromptError;
use rig::prelude::*;
use rig::streaming::{Item, StreamEvent};

use crate::config::{AnalysisConfig, RetrievalConfig};
use crate::embedding::{Embedder, EmbeddingModel};
use crate::error::{Error, Result};
use crate::ids::SessionId;

use super::chart::ChartSpec;
use super::citations::{Citation, CitedAnswer};
use super::events::{AgentEvent, EventSink, ToolName, ToolStep, TurnFailure, TurnRecorder};
use super::hooks::{EmptyAnswer, INVALID_TOOL_CALL_RETRIES, InvalidToolCalls};
use super::policy::WritePolicy;
use super::rerank::RerankAnswer;
use super::text_to_sql::{Modeled, PromptOptions, SystemPrompt, Window};
use super::tools::{
    CreateChartTool, DescribeClassTool, DescribeTableTool, FindPathTool, GraphTools,
    ListDocumentsTool, ListTablesTool, ReaderDb, RunSqlTool, SearchDocumentsTool, SearchGraphTool,
    SharedDb, Turn,
};
use super::vector_index::DuckDbVectorIndex;
use crate::graph::{GraphOptions, GraphResult, store as graph_store};
use crate::llm::{ChatModel, OLLAMA_KEEP_ALIVE, RerankModel, SchemaCall};
use crate::ontology::store as ontology_store;
use crate::storage::sessions::ChatMode;

/// What the provider charged for a turn. Every budget quack computes
/// itself, such as the history trim, is a four-characters-per-token
/// estimate; this is the measured count the provider reported,
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

/// A counter the provider did not report counts as zero here; a turn where
/// it reported none at all is `None` (see [`TokenUsage::reported`]).
impl From<rig::completion::Usage> for TokenUsage {
    fn from(usage: rig::completion::Usage) -> Self {
        Self {
            input_tokens: usage.input_tokens.unwrap_or_default(),
            output_tokens: usage.output_tokens.unwrap_or_default(),
            total_tokens: usage.total_tokens.unwrap_or_default(),
        }
    }
}

impl TokenUsage {
    /// Add one completion request's counts, for the turns that never reach
    /// a final response.
    fn add(&mut self, usage: rig::completion::Usage) {
        let usage = Self::from(usage);
        self.input_tokens = self.input_tokens.saturating_add(usage.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(usage.output_tokens);
        self.total_tokens = self.total_tokens.saturating_add(usage.total_tokens);
    }

    /// The counts, unless every one is zero: a provider that reported no
    /// usage at all, or only zeroes.
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
    /// Milliseconds from the question to the answer, prompt assembly
    /// included. `None` on a response no turn timed.
    #[serde(default)]
    pub duration_ms: Option<u64>,
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
            "duration_ms": self.duration_ms,
            "session_id": session_id,
        })
    }
}

/// One question for the agent: the workspace handles, the embedding model
/// (none for keyword-only search), the analysis settings, the write
/// policy, the prompt options, the history to replay (see
/// `llm::memory::History`), and the message. `reader_db`
/// is the workspace handle's reader, built once for its whole lifetime by
/// [`ReaderDb::open`]: acquiring one per turn would make every turn wait
/// on the writer before it can start, exactly when a slow write is most
/// likely to be holding it.
pub struct Analysis<'a, M> {
    pub db: SharedDb,
    pub reader_db: ReaderDb,
    pub embedder: Option<Embedder<M>>,
    /// The dedicated rerank model, when `[retrieval].rerank = "reranker"`.
    pub rerank_model: Option<RerankModel>,
    pub config: &'a AnalysisConfig,
    pub retrieval_config: &'a RetrievalConfig,
    pub graph_options: GraphOptions,
    pub write_policy: WritePolicy,
    pub prompt: PromptOptions,
    pub history: Vec<Message>,
    pub message: &'a str,
    /// When the question arrived; the answer's `duration_ms` counts from here.
    pub asked: Instant,
}

impl<M> Analysis<'_, M>
where
    M: EmbeddingModel + Clone + Send + Sync + 'static,
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
        completion_model: ChatModel,
        reranker_call: Option<SchemaCall<RerankAnswer>>,
        sink: EventSink,
    ) -> Result<AgentResponse> {
        let max_turns = usize::try_from(self.config.max_turns)
            .map_err(|e| Error::Analysis(format!("max_turns overflow: {e}")))?;
        let recorder = TurnRecorder::new(sink).with_turn_limit(max_turns);
        match self
            .run_inner(completion_model, reranker_call, &recorder)
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

/// A model call that stopped before its answer was whole: rig's
/// normalized `Length` or `ContentFilter` finish reason. rig ends a call
/// that stopped so with no answer at all as an error of its own, whose
/// advice ("raise `max_tokens`") names no setting quack has; this is what
/// quack says instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cutoff {
    /// The output limit, which with Ollama is the context window.
    Length,
    /// The provider's content filter.
    Filtered,
}

impl Cutoff {
    pub(crate) fn of(reason: Option<&rig::completion::FinishReason>) -> Option<Self> {
        use rig::completion::FinishReason;
        match reason? {
            FinishReason::Length => Some(Self::Length),
            FinishReason::ContentFilter => Some(Self::Filtered),
            FinishReason::Stop | FinishReason::ToolCalls | FinishReason::Other(_) => None,
        }
    }

    /// The note a turn carries; `answered` when part of the answer came
    /// through before the stop.
    fn note(self, answered: bool, window: Window) -> String {
        let advice = match window {
            Window::Ollama => {
                " With Ollama the answer shares the context window with the prompt and the \
                 model's reasoning; raise the server's window (OLLAMA_CONTEXT_LENGTH) or ask a \
                 narrower question."
            }
            Window::Provider => " Ask a narrower question.",
        };
        match (self, answered) {
            (Self::Length, false) => {
                format!("The model reached its output limit before it answered.{advice}")
            }
            (Self::Length, true) => {
                format!("The answer was cut off at the model's output limit.{advice}")
            }
            (Self::Filtered, false) => {
                String::from("The provider's content filter stopped the answer before it began.")
            }
            (Self::Filtered, true) => {
                String::from("The provider's content filter cut the answer off.")
            }
        }
    }

    /// Why a one-shot call's answer cannot be used: a cut-off answer is
    /// not one to parse.
    pub(crate) fn refusal(self, what: &str) -> Error {
        Error::Llm(match self {
            Self::Length => format!(
                "the {what} answer was cut off at the model's output limit (with Ollama, the \
                 server's context window: OLLAMA_CONTEXT_LENGTH)"
            ),
            Self::Filtered => {
                format!("the {what} answer was stopped by the provider's content filter")
            }
        })
    }
}

impl<M> Analysis<'_, M>
where
    M: EmbeddingModel + Clone + Send + Sync + 'static,
{
    async fn run_inner(
        self,
        completion_model: ChatModel,
        reranker_call: Option<SchemaCall<RerankAnswer>>,
        recorder: &TurnRecorder,
    ) -> Result<AgentResponse> {
        let Self {
            db: shared_db,
            reader_db,
            embedder: embedding_model,
            rerank_model,
            config: analysis_config,
            retrieval_config,
            graph_options,
            write_policy,
            prompt,
            history,
            message: user_message,
            asked,
        } = self;
        let read = PromptAndModel::read(&reader_db, &prompt).await?;
        let turn = Turn::new(recorder.clone(), write_policy);
        let Replay { history, dropped } = Replay::check(history);
        let window = prompt.window;
        let agent = BuildContext {
            shared_db: Arc::clone(&shared_db),
            reader_db,
            analysis_config,
            retrieval_config,
            graph_options,
            modeled: read.modeled,
            mode: prompt.mode,
            window,
            rerank_model,
            reranker_call,
        }
        .build_agent(completion_model, embedding_model, &read.system_prompt)?;
        let max_turns = usize::try_from(analysis_config.max_turns)
            .map_err(|e| Error::Analysis(format!("max_turns overflow: {e}")))?;

        let mut stream = agent
            .prompt(user_message)
            .history(history)
            .max_turns(max_turns)
            .max_invalid_tool_call_retries(INVALID_TOOL_CALL_RETRIES)
            .tool_context(turn.context())
            .stream();

        let mut streamed = String::new();
        let mut final_text: Option<String> = None;
        let mut stopped: Option<String> = None;
        // The final response carries rig's aggregate for the whole run; the
        // per-call counts are the fallback for a turn that derails before it,
        // which is exactly the turn whose cost is worth knowing.
        let mut aggregate: Option<TokenUsage> = None;
        let mut per_call = TokenUsage::default();
        // How the latest model call stopped, when it stopped short.
        let mut cutoff: Option<Cutoff> = None;

        while let Some(item) = stream.next().await {
            let item = match item {
                Ok(item) => item,
                Err(e) => {
                    // A turn the model derailed (an unknown tool, the turn
                    // limit), that the output limit or a filter cut, or that
                    // failed after text was streamed is still a turn: keep
                    // the text, say what happened, record it. A model that
                    // could not be reached at all stays an error.
                    let stop = StreamStop(&e);
                    if streamed.trim().is_empty() && cutoff.is_none() && !stop.by_agent_loop() {
                        return Err(Error::Analysis(e.to_string()));
                    }
                    tracing::warn!(error = %e, "agent turn stopped early");
                    stopped = Some(cutoff.map_or_else(
                        || stop.explain(analysis_config.max_turns, window),
                        |cut| cut.note(!streamed.trim().is_empty(), window),
                    ));
                    break;
                }
            };
            match item {
                MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                    text,
                    ..
                })) => {
                    streamed.push_str(&text);
                    recorder.emit(AgentEvent::TextDelta(text));
                }
                MultiTurnStreamItem::FinalResponse(response) => {
                    if response.usage.is_reported() {
                        aggregate = Some(response.usage.into());
                    }
                    final_text = Some(response.output());
                }
                MultiTurnStreamItem::CompletionCall(call) => {
                    per_call.add(call.usage);
                    cutoff = Cutoff::of(call.finish_reason.as_ref());
                }
                MultiTurnStreamItem::StreamAssistantItem(_)
                | MultiTurnStreamItem::ToolResult { .. }
                | MultiTurnStreamItem::ToolCall { .. }
                | MultiTurnStreamItem::ToolExecutionCommitted { .. }
                | MultiTurnStreamItem::ModelTurnRetried { .. } => {}
            }
        }

        // A turn that answered but was cut short says so; rig counts a
        // partial answer as a valid one.
        let stopped = stopped.or_else(|| cutoff.map(|cut| cut.note(true, window)));
        let mut answer = turn_text(streamed, final_text, stopped, window, |text| {
            recorder.citations().validate(text)
        });
        if let Some(note) = dropped {
            answer.text.push_str("\n\n(");
            answer.text.push_str(&note);
            answer.text.push(')');
        }
        Ok(turn.finish(answer, aggregate.or_else(|| per_call.reported()), asked))
    }
}

/// The history a turn replays. One rig would refuse mid-turn is dropped up
/// front, with the note the answer carries, so the question still gets an
/// answer.
struct Replay {
    history: Vec<Message>,
    dropped: Option<String>,
}

impl Replay {
    fn check(history: Vec<Message>) -> Self {
        match rig::transcript::validate_canonical(&history) {
            Ok(()) => Self {
                history,
                dropped: None,
            },
            Err(e) => {
                tracing::warn!(error = %e, "the session history is malformed; running without it");
                Self {
                    history: Vec::new(),
                    dropped: Some(format!(
                        "The session's earlier messages could not be replayed ({e}), so this \
                         answer does not take them into account."
                    )),
                }
            }
        }
    }
}

/// The answer: what the model streamed for the last turn, else the
/// aggregated final text when a provider did not stream deltas, put through
/// `check` (the citation check), then quack's own note when the turn
/// stopped early or the check left no text. The note goes on after the
/// check, which drops leaked channel tokens such as `[analysis]` and would
/// otherwise eat quack's references to `[analysis]` settings.
fn turn_text(
    streamed: String,
    final_text: Option<String>,
    stopped: Option<String>,
    window: Window,
    check: impl FnOnce(&str) -> CitedAnswer,
) -> CitedAnswer {
    let raw = match final_text {
        Some(text) if streamed.trim().is_empty() => text,
        Some(_) | None => streamed,
    };
    let mut answer = check(&raw);
    let note = stopped.or_else(|| {
        answer.text.trim().is_empty().then(|| {
            String::from(match window {
                Window::Ollama => {
                    "The model returned no text. With Ollama this usually means the answer or \
                     the prompt did not fit the context window; raise the server's window \
                     (OLLAMA_CONTEXT_LENGTH) or ask a narrower question."
                }
                Window::Provider => "The model returned no text; ask again or narrow the question.",
            })
        })
    });
    if let Some(reason) = note {
        if !answer.text.trim().is_empty() {
            answer.text.push_str("\n\n");
        }
        answer.text.push('(');
        answer.text.push_str(&reason);
        answer.text.push(')');
    }
    answer
}

/// Why a turn's stream ended early.
struct StreamStop<'a>(&'a PromptError);

impl StreamStop<'_> {
    /// Whether the agent loop itself stopped it (an unknown tool, the turn
    /// limit) rather than the provider call failing.
    const fn by_agent_loop(&self) -> bool {
        match self.0 {
            PromptError::UnknownToolCall { .. }
            | PromptError::MaxTurns { .. }
            | PromptError::Cancelled { .. }
            | PromptError::Memory(_) => true,
            PromptError::Provider(_) | PromptError::Report(_) => false,
        }
    }

    /// A user-facing sentence, for a stop worth keeping the turn for.
    fn explain(&self, max_turns: u32, window: Window) -> String {
        match self.0 {
            PromptError::UnknownToolCall { tool_name, .. } => format!(
                "The model called a tool that does not exist ({tool_name}), so the turn \
                 stopped.{}",
                match window {
                    Window::Ollama => {
                        " With Ollama this usually means the prompt was cut to the context \
                         window; check the server's window (OLLAMA_CONTEXT_LENGTH) and the \
                         model's own limit."
                    }
                    Window::Provider => "",
                }
            ),
            PromptError::MaxTurns { .. } => format!(
                "The turn reached the limit of {max_turns} tool calls ([analysis].max_turns) \
                 before the model answered."
            ),
            PromptError::Cancelled { reason, .. } => {
                format!("The turn was cancelled: {reason}")
            }
            PromptError::Provider(e) => format!("The model call failed part way through: {e}"),
            PromptError::Memory(e) => format!("The turn failed: {e}"),
            PromptError::Report(report) => format!("The turn failed: {report}"),
        }
    }
}

impl Turn {
    /// The response once the stream has ended: the checked answer, the
    /// chart and graph results the tools left behind, and the steps.
    fn finish(
        &self,
        answer: CitedAnswer,
        usage: Option<TokenUsage>,
        asked: Instant,
    ) -> AgentResponse {
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
            duration_ms: Some(u64::try_from(asked.elapsed().as_millis()).unwrap_or(u64::MAX)),
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
    window: Window,
    rerank_model: Option<RerankModel>,
    /// The model reranker's one-shot, sampled with `background_effort` by
    /// `dispatch`'s `schema_call`. `Some` only when `rerank = "model"`; the
    /// search tool wires it in, the other rerank modes ignore it.
    reranker_call: Option<SchemaCall<RerankAnswer>>,
}

impl BuildContext<'_> {
    /// The rig agent with every tool this workspace and mode register.
    fn build_agent<M>(
        self,
        completion_model: ChatModel,
        embedding_model: Option<Embedder<M>>,
        system_prompt: &str,
    ) -> Result<Agent>
    where
        M: EmbeddingModel + Clone + Send + Sync + 'static,
    {
        let ctx = self;
        let search = SearchDocumentsTool::from_config(
            ctx.reader_db.clone(),
            ctx.reranker_call,
            ctx.rerank_model.clone(),
            embedding_model.clone(),
            ctx.retrieval_config,
        )
        .with_model(ctx.modeled);
        let reader = || ctx.reader_db.clone();
        let mut builder = AgentBuilder::new(completion_model)
            .preamble(system_prompt)
            .tool(search)
            .tool(RunSqlTool::new(
                Arc::clone(&ctx.shared_db),
                reader(),
                ctx.analysis_config.max_query_rows,
            ))
            .tool(DescribeTableTool(reader()))
            .tool(ListTablesTool(reader()))
            .tool(ListDocumentsTool(reader()))
            .tool(CreateChartTool::new(reader()))
            .temperature(0.1)
            .add_hook(InvalidToolCalls)
            .add_hook(EmptyAnswer);
        if ctx.window == Window::Ollama {
            // Ollama unloads a model 5 minutes after its last request unless
            // the request says otherwise, and a reload costs seconds.
            builder =
                builder.additional_params(serde_json::json!({ "keep_alive": OLLAMA_KEEP_ALIVE }));
        }

        // The ontology is describable as soon as it exists: the prompt block
        // is capped, so a class the model wants the detail of may not be in it
        // even when nothing has been extracted into the graph yet.
        if ctx.modeled.has_ontology() {
            builder = builder.tool(DescribeClassTool(reader()));
        }

        if ctx.modeled.has_graph() {
            let graph = GraphTools {
                db: ctx.reader_db.clone(),
                embedding_model: embedding_model.clone(),
                options: ctx.graph_options,
                mode: ctx.mode,
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
    use crate::analysis::citations::CitationRegistry;

    use crate::ids::{ChunkId, DocumentId};
    #[test]
    fn stream_errors_are_explained_for_the_user() {
        let config = AnalysisConfig::default();
        let unknown = PromptError::UnknownToolCall {
            tool_name: String::from("container.exec"),
            available_tools: vec![String::from("run_sql")],
            allowed_tools: vec![String::from("run_sql")],
            chat_history: Vec::new(),
        };
        assert!(StreamStop(&unknown).by_agent_loop());
        let text = StreamStop(&unknown).explain(config.max_turns, Window::Ollama);
        assert!(text.contains("container.exec"), "{text}");
        assert!(text.contains("OLLAMA_CONTEXT_LENGTH"), "{text}");
        let text = StreamStop(&unknown).explain(config.max_turns, Window::Provider);
        assert!(!text.contains("Ollama"), "{text}");

        let limit = PromptError::MaxTurns {
            max_turns: 10,
            chat_history: Vec::new(),
            prompt: Message::user("q"),
        };
        let text = StreamStop(&limit).explain(config.max_turns, Window::Provider);
        assert!(
            text.contains(&format!("{} tool calls", config.max_turns)),
            "{text}"
        );

        let provider =
            PromptError::Provider(ProviderError::Provider(String::from("connection refused")));
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
            input_tokens: Some(input),
            output_tokens: Some(output),
            total_tokens: Some(input.saturating_add(output)),
            ..rig::completion::Usage::default()
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
        none.add(rig::completion::Usage::default());
        assert_eq!(none, TokenUsage::default());
        assert_eq!(none.reported(), None);
    }

    #[test]
    fn turn_text_keeps_streamed_text_and_notes_early_stops() {
        let as_is = |text: &str| CitedAnswer {
            text: text.to_owned(),
            citations: Vec::new(),
        };
        let text = |streamed: &str, final_text: Option<&str>, stopped: Option<&str>, window| {
            turn_text(
                streamed.to_owned(),
                final_text.map(str::to_owned),
                stopped.map(str::to_owned),
                window,
                as_is,
            )
            .text
        };
        assert_eq!(
            text("so far", None, Some("why"), Window::Provider),
            "so far\n\n(why)"
        );
        assert_eq!(text("", Some("final"), None, Window::Provider), "final");
        let empty = text("", Some(""), None, Window::Ollama);
        assert!(empty.contains("OLLAMA_CONTEXT_LENGTH"), "{empty}");
        let empty = text("", None, None, Window::Provider);
        assert!(!empty.contains("Ollama"), "{empty}");
    }

    /// The note goes on after the citation check: an answer the check
    /// empties (a lone invented marker) gets the no-text note, and the
    /// check, which drops leaked `[analysis]` channel tokens, never sees
    /// quack's own `[analysis]` setting names.
    #[test]
    fn notes_are_added_after_the_citation_check() {
        let check = |text: &str| CitationRegistry::default().validate(text);
        let answer = turn_text(String::from("[7]"), None, None, Window::Ollama, check);
        assert!(
            answer.text.starts_with("(The model returned no text.")
                && answer.text.contains("OLLAMA_CONTEXT_LENGTH"),
            "{}",
            answer.text
        );
        let answer = turn_text(
            String::from("Partly [analysis]answered"),
            None,
            Some(String::from("see [analysis].max_turns")),
            Window::Provider,
            check,
        );
        assert_eq!(answer.text, "Partly answered\n\n(see [analysis].max_turns)");
    }
}
