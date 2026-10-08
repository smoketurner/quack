use std::mem;
use std::sync::{Arc, PoisonError};
use std::time::Instant;

use futures::StreamExt;
use rig::completion::PromptError;
use rig::prelude::*;
use rig::streaming::{Item, PartKind, StreamEvent};

use crate::config::{AnalysisConfig, GraphConfig, RetrievalConfig};
use crate::embedding::{Embedder, EmbeddingModel, Input};
use crate::error::{Error, Result};
use crate::ids::SessionId;
use crate::text::Tokens;

use super::chart::ChartSpec;
use super::citations::{Citation, CitedAnswer};
use super::events::{AgentEvent, EventSink, ToolName, ToolStep, TurnFailure, TurnRecorder};
use super::hooks::{EmptyAnswer, INVALID_TOOL_CALL_RETRIES, InvalidToolCalls};
use super::policy::WritePolicy;
use super::rerank::RerankAnswer;
use super::search::DocumentScope;
use super::table_search::{TableCards, TableLayout, user_tables};
use super::text_to_sql::{Modeled, PromptOptions, Question, SystemPrompt};
use super::tools::{
    CreateChartTool, DescribeClassTool, DescribeTableTool, FindPathTool, FindTablesTool,
    GraphTools, ListDocumentsTool, ListTablesTool, ReadDocumentTool, ReaderDb, RunSqlTool,
    SearchDocumentsTool, SearchGraphTool, SharedDb, Turn, ViewImageTool,
};
use super::vector_index::DuckDbVectorIndex;
use crate::graph::{GraphResult, store as graph_store};
use crate::llm::vision::ImageReader;
use crate::llm::{ChatModel, OLLAMA_KEEP_ALIVE, RerankModel, SchemaCall};
use crate::ontology::store as ontology_store;
use crate::storage::sessions::ChatMode;

/// What the provider charged for a turn. Every budget quack computes
/// itself — the history trim, Ollama's `num_ctx` — is a four-characters-
/// per-token estimate; this is the measured count the provider reported,
/// for the response object and the transcript.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    utoipa::ToSchema,
)]
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
/// it reported none at all is `None` (see `TokenUsage::reported`).
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
    /// The documents the person limited the question to; empty for the
    /// whole workspace.
    #[serde(default)]
    pub documents: DocumentScope,
}

/// One SQL statement the turn ran, as the response object lists it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, utoipa::ToSchema)]
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
    pub fn body(&self, session_id: &SessionId) -> AgentResponseBody {
        AgentResponseBody {
            answer: self.content.clone(),
            citations: self
                .citations
                .iter()
                .map(|c| LabeledCitation {
                    label: c.label(),
                    citation: c.clone(),
                })
                .collect(),
            queries: self.queries(),
            steps: self.steps.clone(),
            graph: self.graph.clone(),
            chart: self.chart.clone(),
            write_refused: self.write_refused,
            cancelled: self.cancelled,
            usage: self.usage,
            duration_ms: self.duration_ms,
            documents: self.documents.clone(),
            session_id: session_id.clone(),
        }
    }
}

/// The response object (design doc 11.2): the answer, its sources, and
/// what the turn did to reach it.
#[derive(Debug, Clone, serde::Serialize, utoipa::ToSchema)]
pub struct AgentResponseBody {
    pub answer: String,
    pub citations: Vec<LabeledCitation>,
    /// The `run_sql` steps, as statements with their row counts.
    pub queries: Vec<QueryRun>,
    pub steps: Vec<ToolStep>,
    /// What the graph tools returned, in call order.
    pub graph: Vec<GraphResult>,
    pub chart: Option<ChartSpec>,
    /// At least one mutating statement was refused during the turn.
    pub write_refused: bool,
    /// The turn was cancelled; `answer` holds what streamed before.
    pub cancelled: bool,
    /// Tokens the provider reported; `null` when it reported none.
    pub usage: Option<TokenUsage>,
    /// Milliseconds from the question to the answer.
    pub duration_ms: Option<u64>,
    /// The documents the question was limited to; empty for all.
    pub documents: DocumentScope,
    pub session_id: SessionId,
}

/// A citation with the label every interface shows for it.
#[derive(Debug, Clone, serde::Serialize, utoipa::ToSchema)]
pub struct LabeledCitation {
    #[serde(flatten)]
    pub citation: Citation,
    /// `file.pdf p. 3`, `notes.md § Heading`: the source as a person reads it.
    pub label: String,
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
    pub graph_options: GraphConfig,
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
        images: Option<ImageReader>,
        sink: EventSink,
    ) -> Result<AgentResponse> {
        let max_turns = usize::try_from(self.config.max_turns)
            .map_err(|e| Error::Analysis(format!("max_turns overflow: {e}")))?;
        let recorder = TurnRecorder::new(sink).with_turn_limit(max_turns);
        let mut this = self;
        this.prompt.question = Some(
            Question::of_turn(
                this.message,
                this.retrieval_config.rrf_k,
                &this.reader_db,
                &this.db,
                this.embedder.as_ref(),
                &recorder,
            )
            .await,
        );
        match this
            .run_inner(completion_model, reranker_call, images, &recorder)
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
    tables: TableLayout,
    /// Whether a ready document is an image, for `view_image` to look at.
    has_images: bool,
}

impl Question {
    /// The turn's question with its embedding, after bringing the stored
    /// table vectors up to date: both only when an embedding model exists
    /// and the workspace has more tables than the prompt describes. A model
    /// that fails leaves the keyword ranking, with a warning.
    async fn of_turn<M: EmbeddingModel>(
        text: &str,
        rrf_k: u32,
        reader: &ReaderDb,
        writer: &SharedDb,
        embedder: Option<&Embedder<M>>,
        recorder: &TurnRecorder,
    ) -> Self {
        let mut question = Self {
            text: text.to_owned(),
            vector: None,
            rrf_k,
        };
        let Some(embedder) = embedder else {
            return question;
        };
        let layout = reader
            .with_db(|db| Ok(TableLayout::of(user_tables(db)?.len())))
            .await;
        match layout {
            Ok(TableLayout::Ranked) => {}
            Ok(TableLayout::AllDescribed) => return question,
            Err(e) => {
                tracing::warn!(error = %e, "could not count the tables; ranking tables by keyword");
                return question;
            }
        }
        match TableCards::refresh_vectors(reader, writer, embedder).await {
            Ok(made) => tracing::debug!(tables = made, "embedded table cards"),
            Err(e) => {
                tracing::warn!(error = %e, "could not embed the table cards; ranking tables by keyword");
                return question;
            }
        }
        match recorder
            .embed_cached(embedder, Input::Query(text.to_owned()))
            .await
        {
            Ok(vector) => question.vector = Some(vector),
            Err(e) => {
                tracing::warn!(error = %e, "could not embed the question; ranking tables by keyword");
            }
        }
        question
    }
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
                    tables: TableLayout::of(user_tables(db)?.len()),
                    has_images: db.has_images()?,
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
            _ => None,
        }
    }

    /// The note a turn carries; `answered` when part of the answer came
    /// through before the stop.
    fn note(self, answered: bool, window: Window) -> String {
        let advice = match window {
            Window::Ollama(_) => {
                " With Ollama the answer shares the context window with the prompt and the \
                 model's reasoning; raise [analysis].max_context_tokens or ask a narrower \
                 question."
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
                 context window: [analysis].max_context_tokens)"
            ),
            Self::Filtered => {
                format!("the {what} answer was stopped by the provider's content filter")
            }
        })
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

impl Window {
    /// Ollama's window when the prompt options cap one, else the provider's.
    fn for_turn(
        prompt: &PromptOptions,
        system_prompt: &str,
        history: &[Message],
        user_message: &str,
    ) -> Self {
        prompt.ollama_context_cap.map_or(Self::Provider, |cap| {
            Self::Ollama(OllamaWindow::for_turn(
                cap,
                system_prompt,
                history,
                user_message,
            ))
        })
    }
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
    fn for_turn(cap: Tokens, system_prompt: &str, history: &[Message], user_message: &str) -> Self {
        let history_chars = serde_json::to_string(history).map_or(0, |h| h.len());
        let prompt = Tokens::of_chars(
            system_prompt
                .len()
                .saturating_add(history_chars)
                .saturating_add(user_message.len()),
        );
        if prompt > cap {
            tracing::warn!(
                prompt_tokens = %prompt,
                %cap,
                "the prompt is larger than [analysis].max_context_tokens; Ollama will truncate it"
            );
        }
        Self::for_prompt(prompt, cap)
    }

    /// The window for a prompt of `prompt` tokens under `cap`.
    fn for_prompt(prompt: Tokens, cap: Tokens) -> Self {
        let needed = prompt.get().saturating_add(Self::HEADROOM);
        let rounded = needed
            .div_ceil(Self::STEP)
            .saturating_mul(Self::STEP)
            .max(Self::FLOOR);
        Self(rounded.min(cap.get().max(Self::FLOOR)))
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
        images: Option<ImageReader>,
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
        let turn = Turn::new(recorder.clone(), write_policy).within(prompt.scope.clone());
        let Replay { history, dropped } = Replay::check(history);
        let window = Window::for_turn(&prompt, &read.system_prompt, &history, user_message);
        let agent = BuildContext {
            shared_db: Arc::clone(&shared_db),
            reader_db,
            analysis_config,
            retrieval_config,
            graph_options,
            modeled: read.modeled,
            tables: read.tables,
            mode: prompt.mode,
            window,
            rerank_model,
            reranker_call,
            images: images.filter(|_| read.has_images),
            turn: turn.clone(),
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

        let mut output = ModelOutput::default();
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
                    if output.text.trim().is_empty() && cutoff.is_none() && !stop.by_agent_loop() {
                        return Err(Error::Analysis(e.to_string()));
                    }
                    tracing::warn!(error = %e, "agent turn stopped early");
                    stopped = Some(cutoff.map_or_else(
                        || stop.explain(analysis_config.max_turns, window),
                        |cut| cut.note(!output.text.trim().is_empty(), window),
                    ));
                    break;
                }
            };
            match item {
                MultiTurnStreamItem::StreamAssistantItem(Item::Event(event)) => {
                    output.take(event, recorder);
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
                    output.call_ended();
                }
                MultiTurnStreamItem::ToolCall { .. }
                | MultiTurnStreamItem::ToolExecutionCommitted { .. } => output.call_kept(),
                MultiTurnStreamItem::ModelTurnRetried { .. } => output.call_rejected(),
                _ => {}
            }
        }

        // A turn that answered but was cut short says so; rig counts a
        // partial answer as a valid one.
        let stopped = stopped.or_else(|| cutoff.map(|cut| cut.note(true, window)));
        let mut answer = turn_text(output.text, final_text, stopped, window, |text| {
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
        if !answer.text.trim().is_empty() {
            answer.text.push_str("\n\n");
        }
        answer.text.push('(');
        answer.text.push_str(&reason);
        answer.text.push(')');
    }
    answer
}

/// What the model has streamed this turn: its answer so far, and whether
/// the model call under way has been reported as reasoning.
///
/// A call's text is provisional until rig runs a tool it asked for: a call
/// rig rejects and asks again is left out of its final text, so it is left
/// out here too.
#[derive(Default)]
struct ModelOutput {
    text: String,
    /// How much of `text` came from calls whose tools ran.
    kept: usize,
    /// The last call ended and none of its tools has run yet.
    ended: bool,
    reasoning: bool,
}

impl ModelOutput {
    /// Keep answer text and pass it on, and tell the interface once per
    /// model call that the model is reasoning.
    fn take(&mut self, event: StreamEvent, recorder: &TurnRecorder) {
        // Another call starting after one that ran no tool: rig rejected it.
        if mem::take(&mut self.ended) {
            self.text.truncate(self.kept);
        }
        match event {
            StreamEvent::Text { text, .. } => {
                self.text.push_str(&text);
                recorder.emit(AgentEvent::TextDelta(text));
            }
            StreamEvent::Reasoning { .. }
            | StreamEvent::Start {
                kind: PartKind::Reasoning,
                ..
            } => {
                if !mem::replace(&mut self.reasoning, true) {
                    recorder.emit(AgentEvent::Reasoning);
                }
            }
            StreamEvent::Start { .. } | StreamEvent::Arguments { .. } | StreamEvent::End { .. } => {
            }
        }
    }

    fn call_ended(&mut self) {
        self.ended = true;
        self.reasoning = false;
    }

    /// A tool the last call asked for ran, so its text stays.
    const fn call_kept(&mut self) {
        self.kept = self.text.len();
        self.ended = false;
    }

    /// A hook rejected the last call for another try.
    fn call_rejected(&mut self) {
        self.text.truncate(self.kept);
        self.ended = false;
    }
}

/// Why a turn's stream ended early.
struct StreamStop<'a>(&'a PromptError);

impl StreamStop<'_> {
    /// Whether the agent loop itself stopped it (an unknown tool, the turn
    /// limit) rather than the provider call failing.
    const fn by_agent_loop(&self) -> bool {
        matches!(
            self.0,
            PromptError::UnknownToolCall { .. }
                | PromptError::MaxTurns { .. }
                | PromptError::Cancelled { .. }
                | PromptError::Memory(_)
        )
    }

    /// A user-facing sentence, for a stop worth keeping the turn for.
    fn explain(&self, max_turns: u32, window: Window) -> String {
        match self.0 {
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
            other => format!("The turn failed: {other}"),
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
        let graph = mem::take(&mut *self.graph.lock().unwrap_or_else(PoisonError::into_inner));
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
            documents: self.scope().clone(),
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
    graph_options: GraphConfig,
    modeled: Modeled,
    tables: TableLayout,
    mode: ChatMode,
    window: Window,
    rerank_model: Option<RerankModel>,
    /// The model reranker's one-shot, sampled with `background_effort` by
    /// `dispatch`'s `schema_call`. `Some` only when `rerank = "model"`; the
    /// search tool wires it in, the other rerank modes ignore it.
    reranker_call: Option<SchemaCall<RerankAnswer>>,
    /// The chat model reading images for `view_image`, when it reads them
    /// and the workspace holds one.
    images: Option<ImageReader>,
    /// The turn the agent runs: `always_retrieve` records on it that it
    /// put chunk text in the prompt.
    turn: Turn,
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
            .tool(ReadDocumentTool::new(reader(), ctx.retrieval_config))
            .tool(
                RunSqlTool::new(
                    Arc::clone(&ctx.shared_db),
                    reader(),
                    ctx.analysis_config.max_query_rows,
                )
                .with_step_rows(
                    usize::try_from(ctx.analysis_config.step_result_rows).unwrap_or(usize::MAX),
                ),
            )
            .tool(DescribeTableTool(reader()))
            .tool(ListTablesTool(reader()))
            .tool(ListDocumentsTool(reader()))
            .tool(CreateChartTool::new(reader()).with_step_rows(
                usize::try_from(ctx.analysis_config.step_result_rows).unwrap_or(usize::MAX),
            ))
            .temperature(0.1)
            .add_hook(InvalidToolCalls)
            .add_hook(EmptyAnswer);
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
            builder = builder.tool(DescribeClassTool(reader()));
        }

        if let Some(images) = ctx.images {
            builder = builder.tool(ViewImageTool::new(reader(), images));
        }

        if ctx.tables == TableLayout::Ranked {
            builder = builder.tool(FindTablesTool::new(
                reader(),
                embedding_model.clone(),
                ctx.retrieval_config.rrf_k,
            ));
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
            let vector_index =
                DuckDbVectorIndex::new(ctx.reader_db.clone(), embedding_model, ctx.turn);
            builder = builder.dynamic_context(samples, vector_index);
        }

        Ok(builder.build())
    }
}

#[cfg(test)]
mod tests;
