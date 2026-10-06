use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use rig::tool::{Tool, ToolContext};
use schemars::generate::SchemaSettings;
use schemars::transform::{Transform, transform_subschemas};
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::Deserialize;
use serde_json::json;

use crate::ids::{ChunkId, NodeId};
use crate::storage::workspace::{
    ChunkScope, ChunkSearchResult, DocumentInfo, DocumentStatus, HybridLimits, StatementKind,
    TEMP_OBJECT_REFUSED, WorkspaceDb, creates_temp_object, quote_ident,
};
use crate::storage::writer::Writer;

use super::chart::{ChartKind, ChartSpec};
use super::citations::{ChunkLocation, Markers};
use super::events::{DetailPreview, ToolName, TurnRecorder};
use super::policy::{Exposure, Hold, RefusalFlag, WriteDecision, WritePolicy};
use super::rerank::{self, ModelReranker, RerankAnswer, Reranker, ScoredReranker};
use super::text_to_sql::Modeled;
use crate::config::{GraphConfig, RerankMode, RetrievalConfig};
use crate::embedding::{Embedder, EmbeddingModel, Input, Vector};
use crate::error::Error;
use crate::ingestion::parser::PageCounts;
use crate::llm::{RerankModel, SchemaCall};
use crate::ontology::{ClassRelations, Ontology, store as ontology_store};
use crate::storage::sessions::ChatMode;
use crate::text::{Fenced, NonBlankText, OneLine, Tokens};

/// A workspace's writer: its one write connection, on a thread of its own
/// with a two-tier line of work ([`crate::storage::writer`]).
pub type SharedDb = Arc<Writer>;

/// One reader connection of a pool: a `try_clone_reader` clone of the
/// writer, used by one read at a time.
type ReaderConn = Arc<Mutex<WorkspaceDb>>;

/// Where one read runs.
enum Slot<'a> {
    Reader(&'a ReaderConn),
    /// The writer, when the pool is degraded or has no readers.
    Writer(&'a SharedDb),
}

/// Shared state behind every clone of one workspace handle's `ReaderDb`.
struct ReaderPool {
    /// A small fixed pool of reader clones, round-robined so concurrent
    /// reads run in parallel instead of queuing behind each other on one
    /// shared connection.
    readers: Vec<ReaderConn>,
    next: AtomicUsize,
    writer: SharedDb,
    /// Set once a write is observed to have created a temp object on the
    /// writer (see [`ReaderDb::observe_write`]): from then on every pick
    /// routes to the writer instead. A `DuckDB` temp object never appears
    /// on a reader clone and never goes away for the process's life, so
    /// this is sticky rather than rechecked per call.
    degraded: AtomicBool,
}

/// A workspace handle for a tool that only ever reads. The pool behind it
/// is private: the only way to query through a `ReaderDb` is
/// [`ReaderDb::with_db`], which always scopes the work inside
/// [`WorkspaceDb::read_only`], so a write attempted through one fails at
/// the database rather than merely by the caller remembering to wrap it.
#[derive(Clone)]
pub struct ReaderDb(Arc<ReaderPool>);

impl ReaderDb {
    /// Wrap `db` as its own single-entry pool: every query still goes
    /// through [`WorkspaceDb::read_only`], so a write is refused, but
    /// there is no separate connection. For tests that want the
    /// read-only net without a real clone, and as the degraded fallback
    /// [`ReaderDb::open`] returns when a clone would not serve.
    #[must_use]
    pub fn new(db: SharedDb) -> Self {
        Self::from_pool(Vec::new(), db)
    }

    fn from_pool(readers: Vec<ReaderConn>, writer: SharedDb) -> Self {
        Self(Arc::new(ReaderPool {
            readers,
            next: AtomicUsize::new(0),
            writer,
            degraded: AtomicBool::new(false),
        }))
    }

    /// The connection this call should use: the writer once degraded,
    /// otherwise the first pool entry, starting from the round-robin
    /// cursor, that is not currently locked by another call — so a slot
    /// mid-scan does not stall every Nth read behind it. If every slot is
    /// busy, this falls back to the cursor's slot like plain round-robin
    /// and blocks there, same as before this preference existed. The
    /// probe is best-effort: another task can take the chosen slot before
    /// the caller actually locks it, which only costs that caller the
    /// same wait a busy slot would have anyway.
    fn pick(&self) -> Slot<'_> {
        let len = self.0.readers.len();
        if self.0.degraded.load(Ordering::Relaxed) || len == 0 {
            return Slot::Writer(&self.0.writer);
        }
        let start = self.0.next.fetch_add(1, Ordering::Relaxed);
        for offset in 0..len {
            let i = start.wrapping_add(offset).checked_rem(len).unwrap_or(0);
            let Some(candidate) = self.0.readers.get(i) else {
                continue;
            };
            if let Ok(guard) = candidate.try_lock() {
                drop(guard);
                return Slot::Reader(candidate);
            }
        }
        let i = start.checked_rem(len).unwrap_or(0);
        self.0
            .readers
            .get(i)
            .map_or(Slot::Writer(&self.0.writer), Slot::Reader)
    }

    /// Run `f` inside a read-only transaction on a reader connection (on
    /// the blocking pool), or on the writer when the pool is empty or
    /// degraded.
    ///
    /// # Errors
    ///
    /// Returns `f`'s error, or an error from locking, the blocking task,
    /// or the transaction itself.
    pub async fn with_db<T>(
        &self,
        f: impl FnOnce(&WorkspaceDb) -> error::Result<T> + Send + 'static,
    ) -> error::Result<T>
    where
        T: Send + 'static,
    {
        match self.pick() {
            Slot::Writer(writer) => writer.run(move |db| db.read_only(f)).await,
            Slot::Reader(conn) => {
                let conn = Arc::clone(conn);
                tokio::task::spawn_blocking(move || {
                    let guard = conn
                        .lock()
                        .map_err(|e| Error::Analysis(format!("reader lock poisoned: {e}")))?;
                    guard.read_only(f)
                })
                .await
                .map_err(|e| Error::Analysis(format!("database task failed: {e}")))?
            }
        }
    }

    /// Check the writer for a temp object created since this handle was
    /// opened and, if one exists, degrade every clone of this `ReaderDb`
    /// to the writer for good. Call this after any statement that ran on
    /// the writer and was classified a write: `creates_temp_object`'s
    /// text match on the SQL catches the obvious `CREATE TEMP TABLE`
    /// case with a friendly refusal before it runs, but cannot be
    /// complete (a leading comment, a semicolon before the real
    /// statement, a multi-statement batch all defeat it), so this is the
    /// actual correctness backstop — observed from `DuckDB`'s own
    /// catalog after the fact, not predicted from the statement text.
    pub async fn observe_write(&self) {
        if self.0.degraded.load(Ordering::Relaxed) {
            return;
        }
        let has_temp = self
            .0
            .writer
            .run(WorkspaceDb::has_temp_tables)
            .await
            .unwrap_or(false);
        if has_temp {
            self.0.degraded.store(true, Ordering::Relaxed);
        }
    }

    /// Build a reader pool for a workspace handle's whole lifetime — once,
    /// not once per turn, so acquiring one never waits behind a slow write
    /// on the writer. `pool_size` genuine [`WorkspaceDb::try_clone_reader`]
    /// clones, unless the writer already has a temp table (the CLI's piped
    /// `stdin`) a clone could not see, in which case every reader-routed
    /// tool shares the writer from the start. A clone that fails
    /// (allocation, a `DuckDB` internal error) degrades that one pool slot
    /// to the writer, with a warning, rather than the caller aborting.
    pub async fn open(shared_db: &SharedDb, pool_size: u32) -> Self {
        let pool_size = pool_size.max(1);
        // One step on the writer for the temp check and every clone,
        // instead of `pool_size + 1` separate ones: shortens how long a
        // concurrent first-time open of this workspace can overlap another
        // one, and is simply less work.
        let opened = shared_db
            .run(move |db| {
                if db.has_temp_tables()? {
                    return Ok(PoolOpen::WriterOnly);
                }
                Ok(PoolOpen::Clones(
                    (0..pool_size).map(|_| db.try_clone_reader()).collect(),
                ))
            })
            .await
            .unwrap_or_else(|e| {
                tracing::warn!(
                    error = %e,
                    "failed to open the reader pool; every read will share the writer"
                );
                PoolOpen::WriterOnly
            });
        let PoolOpen::Clones(clones) = opened else {
            return Self::new(Arc::clone(shared_db));
        };
        let readers: Vec<ReaderConn> = clones
            .into_iter()
            .filter_map(|clone| match clone {
                Ok(db) => Some(Arc::new(Mutex::new(db))),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "failed to clone a reader connection; the pool is one smaller"
                    );
                    None
                }
            })
            .collect();
        Self::from_pool(readers, Arc::clone(shared_db))
    }
}

/// What opening a reader pool found on the writer.
enum PoolOpen {
    /// A temp table a clone could not see: every read shares the writer.
    WriterOnly,
    /// One clone attempt per pool slot.
    Clones(Vec<error::Result<WorkspaceDb>>),
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("query error: {0}")]
    Query(String),
    #[error("analysis error: {0}")]
    Analysis(String),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("format error: {0}")]
    Fmt(String),
}

impl From<std::fmt::Error> for ToolError {
    fn from(e: std::fmt::Error) -> Self {
        Self::Fmt(e.to_string())
    }
}

/// A core error as the model sees it, keeping its own category instead of
/// always wrapping it as a query error — an `Error::Analysis` already reads
/// as `"analysis error: ..."`, so wrapping it again in `ToolError::Query`
/// would show the model `"query error: analysis error: ..."`.
impl From<Error> for ToolError {
    fn from(e: Error) -> Self {
        match e {
            Error::Analysis(msg) => Self::Analysis(msg),
            other => Self::Query(other.to_string()),
        }
    }
}

/// What one turn's tools share, handed to each call as a runtime scope of
/// rig's `ToolContext` ([`Turn::context`]): the record of steps, citations,
/// and cached embeddings; the write policy, whether the turn has read
/// document text, and whether a write was refused; the statements `run_sql`
/// ran; and the chart and graph results the response carries. The tools
/// hold only the workspace and its settings.
#[derive(Clone)]
pub struct Turn {
    pub recorder: TurnRecorder,
    pub policy: WritePolicy,
    pub refused: RefusalFlag,
    pub chart: TurnSlot<ChartSpec>,
    pub graph: GraphResults,
    /// What the turn has read that could dictate a write; the policy
    /// decides each write given it.
    exposure: Arc<Mutex<Exposure>>,
    /// Each statement `run_sql` ran with its parse tree blanked of literals
    /// (`WorkspaceDb::statement_shape`), to spot the model re-running one
    /// statement once per value.
    shapes: Arc<Mutex<Vec<(String, String)>>>,
}

impl Turn {
    #[must_use]
    pub fn new(recorder: TurnRecorder, policy: WritePolicy) -> Self {
        Self {
            recorder,
            policy,
            refused: RefusalFlag::default(),
            chart: TurnSlot::default(),
            graph: Arc::new(Mutex::new(Vec::new())),
            exposure: Arc::new(Mutex::new(Exposure::None)),
            shapes: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The context the turn's run hands its tools.
    #[must_use]
    pub fn context(&self) -> ToolContext {
        ToolContext::new().with_scope(Arc::new(self.clone()))
    }

    /// Record that a tool handed the model document or graph text. From
    /// here on no write of this turn runs without a person's approval.
    pub fn read_documents(&self) {
        *self.exposure.lock().unwrap_or_else(PoisonError::into_inner) = Exposure::Documents;
    }

    /// Number `chunks` for citing. The model is about to read them, so any
    /// at all is document text the turn has read.
    fn cite(&self, chunks: &[ChunkSearchResult]) -> Markers {
        if !chunks.is_empty() {
            self.read_documents();
        }
        self.recorder.citations().register(chunks)
    }

    /// What the turn has read so far.
    #[must_use]
    pub fn exposure(&self) -> Exposure {
        *self.exposure.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The turn a tool call belongs to.
    fn of(context: &ToolContext) -> Result<Arc<Self>, ToolError> {
        context.scope::<Self>().ok_or_else(|| {
            ToolError::Analysis(String::from("the tool was called outside an agent turn"))
        })
    }
}

/// A tool's arguments, whose JSON schema is what the model is shown.
pub trait ToolArgs: JsonSchema {
    /// The schema, or a bare object should it fail to serialize. Subschemas
    /// are inlined and optional arguments carry no `null` type, since some
    /// OpenAI-compatible servers reject a whole request over a `$ref` or a
    /// type array; leaving an argument out of `required` already makes it
    /// optional.
    #[must_use]
    fn schema() -> serde_json::Value
    where
        Self: Sized,
    {
        let generator = SchemaSettings::draft2020_12()
            .with(|settings| settings.inline_subschemas = true)
            .with_transform(NonNullable)
            .into_generator();
        serde_json::to_value(generator.into_root_schema_for::<Self>())
            .unwrap_or_else(|_| json!({"type": "object"}))
    }
}

impl<T: JsonSchema> ToolArgs for T {}

/// Drops `null` from every `type` array, and the null branch from every
/// `anyOf`, leaving the one type an optional argument has when it is sent.
#[derive(Clone)]
struct NonNullable;

impl Transform for NonNullable {
    fn transform(&mut self, schema: &mut Schema) {
        if let Some(serde_json::Value::Array(types)) = schema.get_mut("type") {
            types.retain(|t| t != "null");
            if let [only] = types.as_slice() {
                let only = only.clone();
                schema.insert("type".to_owned(), only);
            }
        }
        if let Some(serde_json::Value::Array(branches)) = schema.get_mut("anyOf") {
            branches.retain(|branch| branch != &json!({"type": "null"}));
            if let [only] = branches.as_slice()
                && let Some(only) = only.as_object().cloned()
            {
                schema.remove("anyOf");
                for (key, value) in only {
                    schema.insert(key, value);
                }
            }
        }
        transform_subschemas(self, schema);
    }
}

/// The arguments of a tool that takes none; whatever the model sends is
/// ignored.
pub struct NoArgs;

impl<'de> Deserialize<'de> for NoArgs {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        serde::de::IgnoredAny::deserialize(deserializer).map(|_| Self)
    }
}

impl JsonSchema for NoArgs {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("NoArgs")
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        json_schema!({ "type": "object", "properties": {} })
    }
}

/// An optional text argument, trimmed, and absent when blank: a model that
/// sends `""` for an argument it meant to leave out has left it out.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NonBlank(Option<String>);

impl NonBlank {
    /// Text as a caller gave it, trimmed, and absent when blank.
    #[must_use]
    pub fn new(text: Option<&str>) -> Self {
        Self(text.and_then(str::non_blank).map(str::to_owned))
    }

    #[must_use]
    pub fn get(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

impl<'de> Deserialize<'de> for NonBlank {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = Option::<String>::deserialize(deserializer)?;
        Ok(Self::new(text.as_deref()))
    }
}

/// Shown to the model as the optional string it is.
impl JsonSchema for NonBlank {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> Cow<'static, str> {
        Option::<String>::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        Option::<String>::json_schema(generator)
    }
}

/// Prefix of a `run_sql` or `describe_table` result that carries a `DuckDB`
/// error instead of rows. The model reads what follows and retries.
pub const SQL_ERROR_PREFIX: &str = "SQL error: ";

/// Message returned to the model when a statement touches internal tables.
pub const INTERNAL_TABLE_REFUSED: &str =
    "This statement references quack's internal tables, which are not available to queries.";

/// What the gate decided about a statement.
#[derive(Debug, PartialEq, Eq)]
enum Gate {
    /// Run it inside a read-only transaction.
    Read,
    /// Run it bare: a write the policy allowed.
    Write,
    /// Do not run a statement that is not a write the policy weighed;
    /// hand this text back to the model.
    Reject(String),
    /// A write that may not run, and why: the one shape of a refused write.
    Refused(Hold),
}

/// What a statement from the agent passes before it runs: no internal
/// tables, a valid parse, and for a write, no temp object and the write
/// policy.
#[derive(Clone)]
struct SqlGate {
    /// Where the statement is classified: a parse, so a reader serves,
    /// never the writer's line.
    db: ReaderDb,
}

impl SqlGate {
    /// Classify `sql` for `run_sql`: a write runs only if `turn`'s write
    /// policy allows it given what the turn has read, and a refusal is
    /// recorded on the turn. A permission prompt holds no connection while
    /// it waits.
    async fn check(&self, sql: &str, turn: &Turn) -> Result<Gate, ToolError> {
        let Some(kind) = self.classify(sql).await? else {
            return Ok(Gate::Reject(String::from(INTERNAL_TABLE_REFUSED)));
        };
        match kind {
            StatementKind::Read => Ok(Gate::Read),
            StatementKind::Invalid(msg) => Ok(Gate::Reject(format!("SQL syntax error: {msg}"))),
            StatementKind::Write => {
                if creates_temp_object(sql) {
                    // A mutating statement the caller wanted to run did
                    // not run, as with a refused write:
                    // AgentResponse::write_refused should say so.
                    turn.refused.set();
                    tracing::info!(sql, "refused a statement that would create a temp object");
                    return Ok(Gate::Reject(String::from(TEMP_OBJECT_REFUSED)));
                }
                let hold = match turn.policy.decide(turn.exposure()) {
                    WriteDecision::Run => return Ok(Gate::Write),
                    WriteDecision::Ask(hold) => {
                        if turn.recorder.ask_permission(sql, hold).await {
                            return Ok(Gate::Write);
                        }
                        hold
                    }
                    WriteDecision::Refuse(hold) => hold,
                };
                turn.refused.set();
                tracing::info!(sql, %hold, "refused write statement from agent");
                Ok(Gate::Refused(hold))
            }
        }
    }

    /// Classify `sql` for a chart, which only reads: any write is
    /// refused as not permitted, and that is not the turn's refused write.
    async fn check_read_only(&self, sql: &str) -> Result<Gate, ToolError> {
        Ok(match self.classify(sql).await? {
            None => Gate::Reject(String::from(INTERNAL_TABLE_REFUSED)),
            Some(StatementKind::Read) => Gate::Read,
            Some(StatementKind::Invalid(msg)) => Gate::Reject(format!("SQL syntax error: {msg}")),
            Some(StatementKind::Write) => Gate::Refused(Hold::NotPermitted),
        })
    }

    /// The statement's kind on a reader, or `None` when it names an
    /// internal table.
    async fn classify(&self, sql: &str) -> Result<Option<StatementKind>, ToolError> {
        let sql = sql.to_owned();
        Ok(self
            .db
            .with_db(move |db| {
                if db.references_internal_table(&sql)? {
                    return Ok(None);
                }
                db.classify_statement(&sql).map(Some)
            })
            .await?)
    }
}

// ---------------------------------------------------------------------------
// run_sql
// ---------------------------------------------------------------------------

pub struct RunSqlTool {
    db: SharedDb,
    /// Its reader classifies statements, and is told through
    /// [`ReaderDb::observe_write`] when a write runs here: `run_sql` never
    /// reads through it.
    gate: SqlGate,
    max_query_rows: u32,
}

impl RunSqlTool {
    #[must_use]
    pub const fn new(db: SharedDb, reader_db: ReaderDb, max_query_rows: u32) -> Self {
        Self {
            db,
            gate: SqlGate { db: reader_db },
            max_query_rows,
        }
    }

    /// The note for a statement that repeats an earlier one this turn with
    /// only its literals changed: the one-query-per-group loop that burns
    /// the turn (a 20B model asked for deaths by state and weather ran the
    /// same GROUP BY once per state until it hit `max_turns`). Records
    /// `sql` for the calls after it.
    fn repeated_note(turn: &Turn, sql: &str, shape: Option<String>) -> Option<String> {
        let shape = shape?;
        let mut shapes = turn.shapes.lock().ok()?;
        let earlier = shapes
            .iter()
            .find(|(earlier, earlier_shape)| {
                *earlier_shape == shape && earlier.trim() != sql.trim()
            })
            .map(|(earlier, _)| earlier.clone());
        shapes.push((sql.to_owned(), shape));
        let preview = DetailPreview::of(earlier.as_deref()?).lines;
        Some(format!(
            "Note: this statement repeats an earlier one with different literal values ({}). \
             Do not run it once per value: one statement covers every group at once with \
             GROUP BY (arg_max(label, measure) picks each group's top label), WHERE col IN \
             (...), or QUALIFY row_number() OVER (PARTITION BY group_col ORDER BY measure DESC) \
             <= n.",
            preview.join(" ")
        ))
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct RunSqlArgs {
    /// The SQL query to execute
    pub query: String,
}

impl Tool for RunSqlTool {
    const NAME: &'static str = ToolName::RunSql.as_str();
    type Error = ToolError;
    type Args = RunSqlArgs;
    type Output = String;

    fn description(&self) -> String {
        format!(
            "Execute a SQL query against the workspace DuckDB database. SELECT queries always run; \
             statements that modify data need the user's write permission and may be refused. \
             Returns up to {} rows as a formatted table.",
            self.max_query_rows
        )
    }

    fn parameters(&self) -> serde_json::Value {
        RunSqlArgs::schema()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let turn = Turn::of(context)?;
        let step = turn.recorder.start(ToolName::RunSql, args.query.trim());
        let read_only = match self.gate.check(&args.query, &turn).await? {
            Gate::Reject(message) => {
                step.finish("refused");
                return Ok(message);
            }
            Gate::Refused(hold) => {
                step.finish(hold.summary());
                return Ok(String::from(hold.refusal()));
            }
            Gate::Read => true,
            Gate::Write => false,
        };
        // DuckDB blocks for up to the query timeout: keep that off
        // the async workers, and keep the lock inside the blocking
        // thread with it. A statement the gate classified `Read`
        // still runs inside a read-only transaction, the same net
        // every other tool has, in case a future parser
        // divergence ever let a mutation through as `Read`.
        let sql = args.query.clone();
        let max_rows = self.max_query_rows;
        let (results, shape) = self
            .db
            .run(move |db| {
                let shape = db.statement_shape(&sql)?;
                let results = if read_only {
                    db.read_only(|db| Ok(db.execute_query_capped(&sql, max_rows)))?
                } else {
                    db.execute_query_capped(&sql, max_rows)
                };
                Ok((results, shape))
            })
            .await?;
        if !read_only {
            // Whatever ran might have created a temp object
            // `creates_temp_object` did not catch (a leading
            // comment, a multi-statement batch); check the
            // writer's catalog regardless of whether the
            // statement itself errored, since an earlier
            // statement in a batch can have already run.
            self.gate.db.observe_write().await;
        }
        match results {
            Ok(results) => {
                step.finish_rows(u64::try_from(results.total_rows).unwrap_or(u64::MAX));
                let mut text = results.to_model_text()?;
                if let Some(note) = Self::repeated_note(&turn, &args.query, shape) {
                    text.push('\n');
                    text.push_str(&note);
                    text.push('\n');
                }
                text.push('\n');
                text.push_str(&turn.recorder.budget_note());
                Ok(text)
            }
            // A failed statement is a result, not a tool failure: rig
            // hides a tool error's message from the model, but DuckDB's
            // text (candidate bindings, the missing table) is exactly
            // what it needs to fix the statement and retry.
            Err(e) => Ok(format!("{SQL_ERROR_PREFIX}{}", step.fail(e))),
        }
    }
}

// ---------------------------------------------------------------------------
// search_documents
// ---------------------------------------------------------------------------

/// `search_documents`'s `top_k` argument is model-supplied and had no
/// ceiling: `args.top_k.unwrap_or(default).max(1)` only floors it, so a
/// call with an implausibly large `top_k` ran the vector and keyword
/// scans with that `LIMIT` and returned every one of those chunks' full
/// text as the tool result, unbounded by anything else in the turn. This
/// caps it at a size still far more than any question needs (the default
/// is 8), while leaving room for a model that deliberately wants a wider
/// sweep.
const MAX_SEARCH_TOP_K: u32 = 50;

/// A reranker, and how many candidates to over-fetch for it before the
/// top `k` are kept.
pub struct Rerank {
    pub reranker: Arc<dyn Reranker>,
    pub candidates: u32,
}

pub struct SearchDocumentsTool<M> {
    db: ReaderDb,
    /// `None` runs keyword search alone: a workspace without an embedding
    /// provider still answers from its documents.
    embedding_model: Option<Embedder<M>>,
    default_top_k: u32,
    rrf_k: u32,
    rerank: Option<Rerank>,
    /// Whether the graph has nodes, so the `entity` argument can resolve.
    /// Without one the argument is left out of the tool's schema and
    /// description: a model shown it tries it, is refused, and spends a
    /// second round trip (and twice the tokens, measured live) reaching the
    /// same answer.
    modeled: Modeled,
}

impl<M> SearchDocumentsTool<M> {
    /// The tool with `retrieval`'s `top_k` and `rrf_k`, and no reranker.
    pub const fn new(
        db: ReaderDb,
        embedding_model: Option<Embedder<M>>,
        retrieval: &RetrievalConfig,
    ) -> Self {
        Self {
            db,
            embedding_model,
            default_top_k: retrieval.top_k,
            rrf_k: retrieval.rrf_k,
            rerank: None,
            modeled: Modeled::Nothing,
        }
    }

    /// The tool as `[retrieval]` configures it: the chat model as reranker
    /// when `rerank = "model"` (`reranker_call`, already built with
    /// `background_effort`), `rerank_model` when `rerank = "reranker"`.
    pub fn from_config(
        db: ReaderDb,
        reranker_call: Option<SchemaCall<RerankAnswer>>,
        rerank_model: Option<RerankModel>,
        embedding_model: Option<Embedder<M>>,
        retrieval: &RetrievalConfig,
    ) -> Self {
        let search = Self::new(db, embedding_model, retrieval);
        let reranker: Arc<dyn Reranker> = match (retrieval.rerank, reranker_call, rerank_model) {
            (RerankMode::None, _, _) => return search,
            (RerankMode::Model, Some(call), _) => Arc::new(ModelReranker::from_call(call)),
            (RerankMode::Model, None, _) => {
                tracing::warn!(
                    "rerank = \"model\" but the background call was not built; keeping the fused \
                     order"
                );
                return search;
            }
            (RerankMode::Reranker, _, Some(model)) => Arc::new(ScoredReranker::new(model)),
            (RerankMode::Reranker, _, None) => {
                tracing::warn!(
                    "rerank = \"reranker\" without a rerank model; keeping the fused order"
                );
                return search;
            }
        };
        search.with_reranker(Rerank {
            reranker,
            candidates: retrieval.rerank_candidates,
        })
    }

    /// Offer the `entity` argument when `modeled` has a graph.
    #[must_use]
    pub fn with_model(mut self, modeled: Modeled) -> Self {
        self.modeled = modeled;
        self
    }

    /// Let `rerank` order the candidates before the top `k` are returned.
    #[must_use]
    pub fn with_reranker(mut self, rerank: Rerank) -> Self {
        self.rerank = Some(rerank);
        self
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct SearchDocumentsArgs {
    /// Natural-language query to search the ingested documents for
    pub query: String,
    /// Number of chunks to return (default from config)
    pub top_k: Option<u32>,
    /// Restrict the search to these documents: ids from `list_documents`
    /// (prefixes accepted) or exact file names
    #[serde(default)]
    pub document_ids: Vec<String>,
    /// Restrict the search to passages this entity was extracted from,
    /// named as it appears in the knowledge graph
    #[serde(default)]
    pub entity: NonBlank,
}

impl<M> Tool for SearchDocumentsTool<M>
where
    M: EmbeddingModel + Send + Sync,
{
    const NAME: &'static str = ToolName::SearchDocuments.as_str();
    type Error = ToolError;
    type Args = SearchDocumentsArgs;
    type Output = String;

    fn description(&self) -> String {
        let mut text = String::from(
            "Search the ingested documents by meaning and by keyword. Returns the most relevant \
             text chunks, each numbered [n] with its source file, page, and heading, for citing \
             in the answer",
        );
        if self.modeled.has_graph() {
            text.push_str(
                ", and names the graph entities each chunk was the source of. Pass entity to \
                 search only the passages one entity was extracted from",
            );
        }
        text.push('.');
        text
    }

    fn parameters(&self) -> serde_json::Value {
        let mut schema = SearchDocumentsArgs::schema();
        if !self.modeled.has_graph()
            && let Some(properties) = schema
                .get_mut("properties")
                .and_then(serde_json::Value::as_object_mut)
        {
            properties.remove("entity");
        }
        schema
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let turn = Turn::of(context)?;
        let entity = args.entity.get();
        let detail = match (args.document_ids.is_empty(), entity) {
            (true, None) => args.query.clone(),
            (true, Some(entity)) => format!("{} (about {entity})", args.query),
            (false, None) => format!("{} (in {})", args.query, args.document_ids.join(", ")),
            (false, Some(entity)) => format!(
                "{} (about {entity}, in {})",
                args.query,
                args.document_ids.join(", ")
            ),
        };
        let step = turn.recorder.start(ToolName::SearchDocuments, &detail);
        let query_vec: Option<Vector> = match &self.embedding_model {
            None => None,
            Some(model) => {
                match turn
                    .recorder
                    .embed_cached(model, Input::Query(args.query.clone()))
                    .await
                {
                    Ok(vector) => Some(vector),
                    Err(e) => return Err(ToolError::Embedding(step.fail(e).to_string())),
                }
            }
        };

        let top_k = args
            .top_k
            .unwrap_or(self.default_top_k)
            .clamp(1, MAX_SEARCH_TOP_K);
        let fetch = self
            .rerank
            .as_ref()
            .map_or(top_k, |rerank| top_k.max(rerank.candidates));

        // The entity's own embedding, not the query's: it resolves a label,
        // so `search_documents(query, entity)` is one embed call each,
        // cached against a later call (search_graph, find_path) that
        // resolves the same label again this turn.
        let entity_vec = match entity {
            Some(entity) => {
                // Its resolution answers with the entity's chunks or with
                // the graph's closest labels.
                turn.read_documents();
                turn.recorder
                    .embed_label(self.embedding_model.as_ref(), entity)
                    .await?
            }
            None => None,
        };

        let query = args.query.clone();
        let document_ids = args.document_ids.clone();
        let entity = entity.map(str::to_owned);
        let rrf_k = self.rrf_k;
        let results = self
            .db
            .with_db(move |db| {
                let mut scope = ChunkScope::for_documents(db, &document_ids)?;
                if let Some(entity) = entity.as_deref() {
                    scope = scope.and_chunks(entity_chunks(db, entity, entity_vec.as_ref())?);
                }
                match &query_vec {
                    Some(vector) => db.search_hybrid_chunks(
                        &query,
                        vector,
                        HybridLimits {
                            top_k: fetch,
                            rrf_k,
                        },
                        &scope,
                    ),
                    None => db.search_keyword_chunks(&query, fetch, &scope),
                }
            })
            .await;
        let results = match results {
            Ok(results) => results,
            Err(e) => return Err(step.fail(e.into())),
        };
        let (results, note) = match &self.rerank {
            Some(rerank) => {
                let keep = usize::try_from(top_k).unwrap_or(usize::MAX);
                let rerank::Reranked {
                    results: kept,
                    outcome,
                } = rerank::apply(rerank.reranker.as_ref(), &args.query, results, keep).await;
                let note = match outcome {
                    rerank::RerankOutcome::Skipped => String::new(),
                    rerank::RerankOutcome::Reranked(name) => format!(", reranked by {name}"),
                    rerank::RerankOutcome::Failed(_) => String::from(", reranking failed"),
                };
                (kept, note)
            }
            None => (results, String::new()),
        };
        step.finish(format!("{} chunks{note}", results.len()));
        let chunk_ids: Vec<ChunkId> = results.iter().map(|r| r.id.clone()).collect();
        // Best effort: the annotation is extra context, so a graph that
        // cannot be read must not fail a search that already succeeded.
        let entities = self
            .db
            .with_db(move |db| graph::store::entities_of_chunks(db, &chunk_ids, CHUNK_ENTITIES))
            .await
            .unwrap_or_default();
        let markers = turn.cite(&results);
        format_search_results(&results, markers, &entities).map_err(Into::into)
    }
}

/// Entities named per retrieved chunk before the rest are counted.
const CHUNK_ENTITIES: usize = 8;

/// The chunks an entity was extracted from, for `search_documents(entity)`.
/// A name that resolves to nothing is an error naming the closest labels,
/// and an entity that exists only in mapped tables says so: both beat an
/// empty result the model reads as "the documents do not cover this".
fn entity_chunks(
    db: &WorkspaceDb,
    entity: &str,
    embedding: Option<&Vector>,
) -> error::Result<Vec<ChunkId>> {
    let nodes = graph::traverse::resolve_entry(db, entity, None, embedding)?;
    if nodes.is_empty() {
        let unknown = UnknownEntity::find(db, entity, embedding);
        return Err(if unknown.closest.is_empty() {
            Error::Analysis(format!(
                "{unknown}; drop the entity argument to search every document"
            ))
        } else {
            unknown.into()
        });
    }
    let ids: Vec<NodeId> = nodes.iter().map(|n| n.id.clone()).collect();
    let chunks = graph::store::chunks_of_nodes(db, &ids)?;
    if chunks.is_empty() {
        return Err(Error::Analysis(format!(
            "'{entity}' is in the graph, but only from table rows, so no document passage is \
             tied to it; search_graph has its connections, or drop the entity argument to search \
             every document"
        )));
    }
    Ok(chunks)
}

/// Render search hits as numbered, citable chunks.
///
/// # Errors
///
/// Returns an error only if formatting into the output buffer fails.
pub fn format_search_results(
    results: &[ChunkSearchResult],
    markers: Markers,
    entities: &BTreeMap<ChunkId, Vec<String>>,
) -> Result<String, std::fmt::Error> {
    if results.is_empty() {
        return Ok(String::from(
            "No relevant chunks found. Tell the user the documents do not appear to cover this.",
        ));
    }
    let mut out = format!(
        "Retrieved chunks. Cite each fact you use with the chunk's [n] marker at the end of the \
         sentence. {}\n\n",
        Fenced::NOTICE
    );
    for (i, chunk) in results.iter().enumerate() {
        let n = markers.nth(i);
        // Graph entities stay on the metadata line: on its own line under
        // the header a model quotes the list back as if it were part of
        // the passage (seen live).
        let graph = entities
            .get(&chunk.id)
            .filter(|e| !e.is_empty())
            .map_or(String::new(), |e| {
                format!(", graph entities: {}", e.join(", "))
            });
        writeln!(
            out,
            "[{n}] {} (document_id: {}, chunk {}, score {:.4}{graph})",
            ChunkLocation::from(chunk),
            chunk.document_id,
            chunk.chunk_index,
            chunk.score
        )?;
        writeln!(out, "{}", Fenced(chunk.content.trim()))?;
        writeln!(out)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// read_document
// ---------------------------------------------------------------------------

/// Chunks one `read_document` call fetches at most; the token budget then
/// decides how many of them the model sees.
const MAX_READ_CHUNKS: u32 = 50;

pub struct ReadDocumentTool {
    db: ReaderDb,
    /// Chunk text one call hands the model, at most:
    /// `[retrieval].pinned_token_budget`, the budget a whole-document read
    /// already has.
    budget: Tokens,
}

impl ReadDocumentTool {
    #[must_use]
    pub const fn new(db: ReaderDb, retrieval: &RetrievalConfig) -> Self {
        Self {
            db,
            budget: retrieval.pinned_token_budget,
        }
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct ReadDocumentArgs {
    /// The document: an id from `list_documents` (a prefix is enough) or
    /// its exact file name
    pub document: String,
    /// Position of the first chunk to read, counting from 0 (default 0)
    pub from: Option<u32>,
    /// How many chunks to read (default: as many as fit the budget, at
    /// most 50)
    pub limit: Option<u32>,
}

impl Tool for ReadDocumentTool {
    const NAME: &'static str = ToolName::ReadDocument.as_str();
    type Error = ToolError;
    type Args = ReadDocumentArgs;
    type Output = String;

    fn description(&self) -> String {
        String::from(
            "Read one document's chunks in order from a position, for a whole section or a \
             document's start rather than the best-matching passages. Returns consecutive \
             chunks numbered [n] for citing, like search_documents, within a token budget, \
             and says where to continue.",
        )
    }

    fn parameters(&self) -> serde_json::Value {
        ReadDocumentArgs::schema()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let turn = Turn::of(context)?;
        let from = args.from.unwrap_or(0);
        let limit = args
            .limit
            .unwrap_or(MAX_READ_CHUNKS)
            .clamp(1, MAX_READ_CHUNKS);
        let step = turn.recorder.start(
            ToolName::ReadDocument,
            &format!("{} from {from}", args.document),
        );
        let wanted = args.document;
        let read = self
            .db
            .with_db(move |db| {
                let documents = db.list_documents()?;
                let document = DocumentInfo::find(&documents, &wanted)?.clone();
                if document.status != DocumentStatus::Ready {
                    return Err(Error::Analysis(format!(
                        "{} is {}, not ready, so its text cannot be read",
                        OneLine(&document.filename),
                        document.status
                    )));
                }
                let Some(total) = document.chunk_count.filter(|n| *n > 0) else {
                    return Err(Error::Analysis(format!(
                        "{} holds no text chunks: a tabular file is loaded as a table, which \
                         run_sql reads",
                        OneLine(&document.filename)
                    )));
                };
                let chunks = db.document_chunks(&document.id, from, limit)?;
                Ok((document, chunks, total))
            })
            .await;
        let (document, chunks, total) = match read {
            Ok(read) => read,
            Err(e) => return Err(step.fail(e.into())),
        };
        // At least one chunk goes out whatever its size; the rest only
        // while they fit the budget.
        let mut kept: Vec<ChunkSearchResult> = Vec::with_capacity(chunks.len());
        let mut used = Tokens::default();
        for chunk in chunks {
            let cost = Tokens::estimate(&chunk.content);
            if !kept.is_empty() && used.saturating_add(cost) > self.budget {
                break;
            }
            used = used.saturating_add(cost);
            kept.push(chunk);
        }
        step.finish(format!("{} chunks", kept.len()));
        let filename = OneLine(&document.filename);
        let (Some(first), Some(last)) = (kept.first(), kept.last()) else {
            return Ok(format!(
                "{filename} has {total} chunks, positions 0 to {}; from = {from} is past the \
                 end.",
                total.saturating_sub(1)
            ));
        };
        let (first, last) = (first.chunk_index, last.chunk_index);
        let markers = turn.cite(&kept);
        let mut out = format_search_results(&kept, markers, &BTreeMap::new())?;
        if i64::from(last).saturating_add(1) >= total {
            writeln!(
                out,
                "End of {filename}: chunks {first} to {last} of {total}."
            )?;
        } else {
            writeln!(
                out,
                "Chunks {first} to {last} of {total} in {filename}; call read_document again \
                 with from = {} for the rest.",
                last.saturating_add(1)
            )?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// describe_table
// ---------------------------------------------------------------------------

pub struct DescribeTableTool(pub ReaderDb);

#[derive(Deserialize, JsonSchema)]
pub struct DescribeTableArgs {
    /// Name of the table to describe
    pub table_name: String,
}

impl Tool for DescribeTableTool {
    const NAME: &'static str = ToolName::DescribeTable.as_str();
    type Error = ToolError;
    type Args = DescribeTableArgs;
    type Output = String;

    fn description(&self) -> String {
        String::from(
            "Describe one table in the workspace: its row count, every column with its DuckDB \
             type, and up to 3 sample rows. Use it for a table the system prompt lists without \
             columns or sample rows, or to confirm exact column names before writing SQL. A name \
             that matches no table returns an error followed by the names of the tables that \
             exist. It does not profile values; for min, max, null share, or distinct counts, \
             run SUMMARIZE <table> with run_sql.",
        )
    }

    fn parameters(&self) -> serde_json::Value {
        DescribeTableArgs::schema()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let turn = Turn::of(context)?;
        let step = turn
            .recorder
            .start(ToolName::DescribeTable, &args.table_name);
        let table_name = args.table_name.clone();
        let outcome = self
            .0
            .with_db(move |db| {
                Ok(match db.describe_table(&table_name) {
                    Ok(d) => Ok(d),
                    Err(e) => {
                        let tables = db.list_tables().unwrap_or_default();
                        Err((e.to_string(), tables))
                    }
                })
            })
            .await?;
        let desc = match outcome {
            Ok(d) => d,
            Err((message, tables)) => {
                let message = step.fail(message);
                return Ok(format!(
                    "{SQL_ERROR_PREFIX}{message}\nTables in this workspace: {}",
                    if tables.is_empty() {
                        String::from("none")
                    } else {
                        tables.join(", ")
                    }
                ));
            }
        };

        let mut output = String::new();
        writeln!(output, "Table: {}", desc.table_name)?;
        writeln!(output, "Rows: {}", desc.row_count)?;
        writeln!(output, "Columns:")?;
        for col in &desc.columns {
            writeln!(output, "  - {} ({})", col.name, col.column_type)?;
        }

        if !desc.sample_rows.rows.is_empty() {
            writeln!(output, "\nSample rows:")?;
            let mut buf = Vec::new();
            if desc.sample_rows.write_table(&mut buf).is_ok()
                && let Ok(text) = String::from_utf8(buf)
            {
                write!(output, "{text}")?;
            }
        }

        step.finish(format!("{} columns", desc.columns.len()));
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// list_tables
// ---------------------------------------------------------------------------

pub struct ListTablesTool(pub ReaderDb);

impl Tool for ListTablesTool {
    const NAME: &'static str = ToolName::ListTables.as_str();
    type Error = ToolError;
    type Args = NoArgs;
    type Output = String;

    fn description(&self) -> String {
        String::from(
            "List every user table in the workspace with its row count, one per line. The \
             system prompt already lists the tables, so call this to re-check after a statement \
             created, replaced, or dropped one. quack's internal tables are not listed and cannot \
             be queried. Takes no arguments.",
        )
    }

    fn parameters(&self) -> serde_json::Value {
        NoArgs::schema()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let turn = Turn::of(context)?;
        let step = turn.recorder.start(ToolName::ListTables, "");
        // One reader round trip for the listing and every table's row
        // count, rather than one per table; each count is its own
        // timeout-guarded statement, so one huge table cannot pin the
        // transaction indefinitely.
        let tables: Vec<(String, Option<i64>)> = self
            .0
            .with_db(|db| {
                let tables = db.list_tables()?;
                Ok(tables
                    .into_iter()
                    .map(|t| {
                        let count = db.under_timeout(|db| db.count_rows(&t)).ok();
                        (t, count)
                    })
                    .collect())
            })
            .await?;
        step.finish(format!("{} tables", tables.len()));
        if tables.is_empty() {
            return Ok(String::from("No tables found in this workspace."));
        }
        let mut output = String::from("Tables:\n");
        for (table, count) in &tables {
            match count {
                Some(n) => writeln!(output, "- {table} ({n} rows)")?,
                None => writeln!(output, "- {table}")?,
            }
        }
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// list_documents
// ---------------------------------------------------------------------------

pub struct ListDocumentsTool(pub ReaderDb);

impl Tool for ListDocumentsTool {
    const NAME: &'static str = ToolName::ListDocuments.as_str();
    type Error = ToolError;
    type Args = NoArgs;
    type Output = String;

    fn description(&self) -> String {
        String::from(
            "List every ingested document with its id, file name, status (queued, processing, \
             ready, or error), MIME type, source, and title when it has one. A document's text is \
             searchable once its status is ready; a tabular file is loaded as a table instead. A \
             document marked with unreadable pages or pages without text was only partly read: \
             those pages are not searchable. To \
             search within particular documents, pass their ids (a prefix is enough) or exact \
             file names as search_documents' document_ids. Takes no arguments.",
        )
    }

    fn parameters(&self) -> serde_json::Value {
        NoArgs::schema()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let turn = Turn::of(context)?;
        let step = turn.recorder.start(ToolName::ListDocuments, "");
        let docs = self.0.with_db(WorkspaceDb::list_documents).await?;
        step.finish(format!("{} documents", docs.len()));
        if docs.is_empty() {
            return Ok(String::from("No documents found in this workspace."));
        }
        let mut output = String::from("Documents:\n");
        for doc in &docs {
            let title = doc
                .title
                .as_deref()
                .map_or(String::new(), |t| format!(", title: {}", OneLine(t)));
            let pages = PageCounts::suffix(doc.pages);
            writeln!(
                output,
                "- {} (id: {}, status: {}, type: {}, source: {}{title}{pages})",
                OneLine(&doc.filename),
                doc.id,
                doc.status,
                doc.mime_type.as_deref().unwrap_or("unknown"),
                doc.source,
            )?;
        }
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// create_chart
// ---------------------------------------------------------------------------

/// A value one tool leaves for the end of the turn: the chart.
#[derive(Debug)]
pub struct TurnSlot<T>(Arc<Mutex<Option<T>>>);

impl<T> Clone for TurnSlot<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T> Default for TurnSlot<T> {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }
}

impl<T> TurnSlot<T> {
    /// Leave `value`, replacing what an earlier call left. The lock is
    /// never poisoned: a panic aborts the process.
    pub fn put(&self, value: T) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(value);
    }

    /// What was left, emptying the slot.
    #[must_use]
    pub fn take(&self) -> Option<T> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).take()
    }
}

pub struct CreateChartTool {
    /// Charts only read: the gate lets no write through.
    gate: SqlGate,
}

impl CreateChartTool {
    #[must_use]
    pub const fn new(db: ReaderDb) -> Self {
        Self {
            gate: SqlGate { db },
        }
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct CreateChartArgs {
    /// SQL query to get chart data (at most 200 rows; aggregate first)
    pub sql: String,
    /// Kind of chart: bar, line, scatter, or pie. The schema lists the kinds;
    /// the text is parsed leniently so a near miss comes back as a message
    /// the model can act on rather than a rejected call.
    #[schemars(with = "ChartKind")]
    pub kind: String,
    /// Column for the x axis (category labels; slice names for pie)
    pub x: String,
    /// Numeric column for the y axis (slice values for pie)
    pub y: String,
    /// Chart title
    pub title: String,
}

impl Tool for CreateChartTool {
    const NAME: &'static str = ToolName::CreateChart.as_str();
    type Error = ToolError;
    type Args = CreateChartArgs;
    type Output = String;

    fn description(&self) -> String {
        String::from(
            "Draw a chart from a SQL query: runs the query and renders a bar, line, scatter, or pie \
             chart of column y against column x. The query must return at most 200 rows.",
        )
    }

    fn parameters(&self) -> serde_json::Value {
        CreateChartArgs::schema()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let turn = Turn::of(context)?;
        let step = turn.recorder.start(ToolName::CreateChart, args.sql.trim());
        let rejected = match self.gate.check_read_only(&args.sql).await? {
            Gate::Reject(message) => Some(message),
            Gate::Refused(hold) => Some(String::from(hold.refusal())),
            Gate::Read | Gate::Write => None,
        };
        if let Some(message) = rejected {
            step.finish("rejected");
            return Ok(format!("Chart query rejected. {message}"));
        }

        let sql = args.sql.clone();
        let results = self.gate.db.with_db(move |db| db.execute_query(&sql)).await;
        let results = match results {
            Ok(r) => r,
            Err(e) => return Err(step.fail(e.into())),
        };

        if results.rows.is_empty() {
            step.finish("0 rows");
            return Ok(String::from(
                "Query returned no rows — cannot generate chart.",
            ));
        }

        let spec = args.kind.parse::<ChartKind>().and_then(|kind| {
            ChartSpec::from_results(&results, kind, &args.x, &args.y, &args.title)
        });
        let spec = match spec {
            Ok(spec) => spec,
            Err(e) => return Ok(format!("Chart not created: {}", step.fail(e))),
        };

        let summary = format!(
            "{} chart \"{}\" with {} points ({} by {})",
            spec.kind.as_str(),
            spec.title,
            spec.points(),
            args.y,
            args.x
        );
        turn.chart.put(spec);

        step.finish(format!("{} points", results.rows.len()));
        Ok(format!(
            "Chart created and shown to the user: {summary}. Describe what it shows; do not repeat the data."
        ))
    }
}

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// search_graph and find_path
// ---------------------------------------------------------------------------

use crate::error;
use crate::graph::query::{GraphQuery, Listed, OntologyId, PathEnds, PathQuery, UnknownEntity};
use crate::graph::store::ClassCensus;
use crate::graph::{self, GraphResult, Origin};

/// The graph results a turn produced, kept for the response.
pub type GraphResults = Arc<Mutex<Vec<GraphResult>>>;

/// What the graph tools are built from.
#[derive(Clone)]
pub struct GraphTools<M> {
    pub db: ReaderDb,
    /// `None` resolves entities by exact label and alias only.
    pub embedding_model: Option<Embedder<M>>,
    pub options: GraphConfig,
    /// Query mode does not answer from provisional nodes.
    pub mode: ChatMode,
}

/// A graph result as the turn's mode may show it.
struct Shown {
    result: GraphResult,
    /// There were matches, and every one was provisional.
    all_provisional: bool,
}

impl<M> GraphTools<M> {
    /// `result` without provisional nodes when the mode excludes them.
    fn shown(&self, result: GraphResult) -> Shown {
        let had_matches = !result.nodes.is_empty();
        let result = match self.mode {
            ChatMode::Query => result.without_provisional(),
            ChatMode::Chat => result,
        };
        Shown {
            all_provisional: had_matches && result.nodes.is_empty(),
            result,
        }
    }

    /// Keep `result` for the turn's response.
    fn keep(turn: &Turn, result: GraphResult) {
        if let Ok(mut results) = turn.graph.lock() {
            results.push(result);
        }
    }
}

pub struct SearchGraphTool<M>(pub GraphTools<M>);

/// A graph search as every interface takes it: the agent tool's
/// arguments, the MCP tool's, the REST body, and what the CLI and the
/// web form fill in.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct SearchGraphArgs {
    /// The entity to start from (its name as it appears in the data); omit
    /// to list every entity of `class`
    #[serde(default)]
    pub entity: NonBlank,
    /// Restrict the entry point, or the listing, to this ontology class id
    #[serde(default)]
    pub class: NonBlank,
    /// Follow only this relation id
    #[serde(default)]
    pub relation: NonBlank,
    /// How many hops out from the entity (default 2)
    pub hops: Option<u32>,
}

impl SearchGraphArgs {
    /// The query these arguments ask for.
    ///
    /// # Errors
    ///
    /// Returns an error when there is neither an entity nor a class.
    pub fn query(&self) -> Result<GraphQuery, Error> {
        GraphQuery::new(
            self.entity.get(),
            self.class.get(),
            self.relation.get(),
            self.hops,
        )
    }
}

impl<M> Tool for SearchGraphTool<M>
where
    M: EmbeddingModel + Send + Sync,
{
    const NAME: &'static str = ToolName::SearchGraph.as_str();
    type Error = ToolError;
    type Args = SearchGraphArgs;
    type Output = String;

    fn description(&self) -> String {
        String::from(
            "Explore the knowledge graph: the entities connected to a named entity within a few \
             hops (optionally along one relation), or every entity of an ontology class. Each \
             node and edge comes with provenance (the document chunk or table row it was \
             extracted from), which you cite like search results.",
        )
    }

    fn parameters(&self) -> serde_json::Value {
        SearchGraphArgs::schema()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let tools = &self.0;
        let turn = Turn::of(context)?;
        let detail = match (args.entity.get(), args.class.get()) {
            (Some(e), Some(c)) => format!("{e} ({c})"),
            (Some(e), None) => e.to_owned(),
            (None, Some(c)) => format!("class {c}"),
            (None, None) => String::new(),
        };
        let step = turn.recorder.start(ToolName::SearchGraph, &detail);
        let query = match args.query() {
            Ok(query) => query,
            Err(e) => return Err(step.fail(e.into())),
        };
        let embedding = match query.entity.as_deref() {
            Some(e) => {
                turn.recorder
                    .embed_label(tools.embedding_model.as_ref(), e)
                    .await?
            }
            None => None,
        };
        let options = tools.options;
        let lookup = tools
            .db
            .with_db(move |db| {
                let result = query.run(db, embedding.as_ref(), &options)?;
                let suggestions = if result.nodes.is_empty() {
                    query.suggestions(db, embedding.as_ref())?
                } else {
                    Vec::new()
                };
                Ok((result, suggestions))
            })
            .await;
        let (result, suggestions) = match lookup {
            Ok(lookup) => lookup,
            Err(e) => {
                // A refusal can name the graph's closest labels.
                turn.read_documents();
                return Err(step.fail(e.into()));
            }
        };
        let Shown {
            result,
            all_provisional,
        } = tools.shown(result);
        if result.nodes.is_empty() {
            let empty = if all_provisional {
                EmptyLookup::AllProvisional
            } else {
                EmptyLookup::NoMatch(&suggestions)
            };
            step.finish(empty.summary());
            if !suggestions.is_empty() {
                turn.read_documents();
            }
            let text = empty.text()?;
            GraphTools::<M>::keep(&turn, result);
            return Ok(text);
        }
        let of = match result.total_nodes.filter(|_| result.truncated) {
            Some(total) => format!(" of {total}"),
            None => String::new(),
        };
        step.finish(format!(
            "{}{of} nodes, {} edges",
            result.nodes.len(),
            result.edges.len()
        ));
        let text = format_graph_result(&result, &turn, &tools.db).await?;
        GraphTools::<M>::keep(&turn, result);
        Ok(text)
    }
}

pub struct FindPathTool<M>(pub GraphTools<M>);

/// A path request as every interface takes it, like [`SearchGraphArgs`].
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct FindPathArgs {
    /// The entity to start from
    pub from: String,
    /// The entity to reach
    pub to: String,
    /// Longest path to consider (default 4)
    pub max_hops: Option<u32>,
}

impl FindPathArgs {
    /// The query these arguments ask for.
    ///
    /// # Errors
    ///
    /// Returns an error when either end is blank.
    pub fn query(&self) -> Result<PathQuery, Error> {
        PathQuery::new(&self.from, &self.to, self.max_hops)
    }
}

impl<M> Tool for FindPathTool<M>
where
    M: EmbeddingModel + Send + Sync,
{
    const NAME: &'static str = ToolName::FindPath.as_str();
    type Error = ToolError;
    type Args = FindPathArgs;
    type Output = String;

    fn description(&self) -> String {
        String::from(
            "Find the shortest chain of relations connecting two entities in the knowledge \
             graph, with provenance for every hop.",
        )
    }

    fn parameters(&self) -> serde_json::Value {
        FindPathArgs::schema()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let tools = &self.0;
        let turn = Turn::of(context)?;
        let step = turn.recorder.start(
            ToolName::FindPath,
            &format!("{} -> {}", args.from.trim(), args.to.trim()),
        );
        let query = match args.query() {
            Ok(query) => query,
            Err(e) => return Err(step.fail(e.into())),
        };
        let ends = PathEnds {
            from: turn
                .recorder
                .embed_label(tools.embedding_model.as_ref(), &query.from)
                .await?,
            to: turn
                .recorder
                .embed_label(tools.embedding_model.as_ref(), &query.to)
                .await?,
        };
        let options = tools.options;
        let path = query.clone();
        let result = tools
            .db
            .with_db(move |db| path.run(db, &ends, &options))
            .await;
        let result = match result {
            Ok(result) => result,
            Err(e) => {
                // A refusal can name the graph's closest labels.
                turn.read_documents();
                return Err(step.fail(e.into()));
            }
        };
        let Shown {
            result,
            all_provisional,
        } = tools.shown(result);
        let PathQuery { from, to, max_hops } = query;
        if result.nodes.is_empty() {
            if all_provisional {
                step.finish("path is provisional");
                return Ok(format!(
                    "A path connects {from} and {to}, but it runs through provisional entities: \
                     they were built from an ontology version nobody has reviewed, and query mode \
                     does not answer from those. Tell the user the path is unreviewed and that \
                     `quack graph review` accepts it."
                ));
            }
            step.finish("no path");
            return Ok(format!(
                "No path connects {from} and {to} within {max_hops} hops."
            ));
        }
        step.finish(format!("{} hops", result.edges.len()));
        let text = format_graph_result(&result, &turn, &tools.db).await?;
        GraphTools::<M>::keep(&turn, result);
        Ok(text)
    }
}

/// Why a graph lookup has nothing to show.
enum EmptyLookup<'a> {
    /// There were matches, and query mode dropped every one as provisional.
    AllProvisional,
    /// Nothing matched; the closest labels, when there are any.
    NoMatch(&'a [String]),
}

impl EmptyLookup<'_> {
    /// The step's one-line result.
    fn summary(&self) -> &'static str {
        match self {
            Self::AllProvisional => "matches are provisional",
            Self::NoMatch(_) => "0 nodes, 0 edges",
        }
    }

    /// What the model is told: that the graph has only unreviewed matches,
    /// that it has nothing, or which labels to try instead.
    fn text(&self) -> Result<String, ToolError> {
        let suggestions = match self {
            Self::AllProvisional => {
                return Ok(String::from(
                    "The graph has matches, but all of them are provisional: they were built from \
                     an ontology version nobody has reviewed, and query mode does not answer from \
                     those. Tell the user the graph has unreviewed matches and that `quack graph \
                     review` accepts them.",
                ));
            }
            Self::NoMatch(suggestions) => suggestions,
        };
        let mut out = String::from("No matching entities in the graph.");
        if suggestions.is_empty() {
            out.push_str(" Tell the user the graph has nothing on this.");
            return Ok(out);
        }
        write!(
            out,
            " The closest labels in the graph are: {}. Search again with one of them if that is \
             what the user meant; otherwise tell the user the graph has nothing on this.",
            suggestions
                .iter()
                .map(|label| OneLine(label).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        Ok(out)
    }
}

/// The most characters one graph result may put into a turn. A listing of
/// `max_nodes` entities, each with its properties, is otherwise large
/// enough to push a fixed Ollama `num_ctx` over: the front of the prompt
/// is cut, the tool list goes with it, and the model invents a tool name
/// (seen live on gpt-oss:20b, the failure #40 describes).
const MAX_GRAPH_TEXT_CHARS: usize = 6_000;

/// Chunk sources cited from one graph result before the rest are counted.
const MAX_GRAPH_SOURCES: usize = 10;

/// Table rows named in one graph result before the rest are counted.
const MAX_GRAPH_ROWS: usize = 20;

/// An oversized rendering cut at a line boundary, keeping the summary line
/// (which carries the totals) and saying what to do instead.
fn trim_graph_text(text: &str, budget: usize) -> String {
    let summary = text.trim_end().lines().next_back().unwrap_or_default();
    let mut kept = String::new();
    let mut shown: usize = 0;
    for line in text.lines() {
        if kept.chars().count().saturating_add(line.chars().count()) > budget {
            break;
        }
        kept.push_str(line);
        kept.push('\n');
        shown = shown.saturating_add(1);
    }
    format!(
        "{kept}... {} more lines not shown: narrow the search with a class, a relation, or fewer \
         hops, and use describe_class to count a class.\n{summary}\n",
        text.lines().count().saturating_sub(shown)
    )
}

/// Render a graph result for the model: the tree, then the sources each
/// node and edge came from, registered as citable `[n]` markers (chunks)
/// or named as table rows. Bounded as a whole, not just per node. The
/// callers answer an empty result themselves. The turn has then read
/// graph text.
async fn format_graph_result(
    result: &GraphResult,
    turn: &Turn,
    db: &ReaderDb,
) -> Result<String, ToolError> {
    turn.read_documents();
    let tree = result.to_string();
    let mut out = if tree.chars().count() > MAX_GRAPH_TEXT_CHARS {
        trim_graph_text(&tree, MAX_GRAPH_TEXT_CHARS)
    } else {
        tree
    };
    // Bounded like the tree: a 200-node listing can carry a source per
    // node, and every one of them would be quoted in full.
    let all_chunk_ids: std::collections::BTreeSet<ChunkId> = result
        .provenance
        .iter()
        .filter_map(|p| p.origin.chunk_id().cloned())
        .collect();
    let hidden_chunks = all_chunk_ids.len().saturating_sub(MAX_GRAPH_SOURCES);
    let chunk_ids: Vec<ChunkId> = all_chunk_ids.into_iter().take(MAX_GRAPH_SOURCES).collect();
    let (chunks, ontology) = db
        .with_db(move |db| {
            let chunks = db.chunks_by_ids(&chunk_ids)?;
            Ok((chunks, ontology_store::current(db)?))
        })
        .await?;
    if !chunks.is_empty() {
        let markers = turn.cite(&chunks);
        writeln!(
            out,
            "\nSources (cite with the [n] marker). {}",
            Fenced::NOTICE
        )?;
        for (i, chunk) in chunks.iter().enumerate() {
            let n = markers.nth(i);
            let excerpt: String = chunk.content.trim().chars().take(200).collect();
            // The page, not the heading: the excerpt shows the passage.
            let location = ChunkLocation {
                heading: None,
                ..ChunkLocation::from(chunk)
            };
            writeln!(out, "[{n}] {location}:\n{}", Fenced(&excerpt))?;
        }
        if hidden_chunks > 0 {
            writeln!(out, "... and {hidden_chunks} more sources")?;
        }
    }
    let rows: std::collections::BTreeSet<String> = result
        .provenance
        .iter()
        .filter_map(|p| match &p.origin {
            Origin::Row {
                table_name,
                row_key,
            } => Some(
                RowReference {
                    ontology: ontology.as_ref(),
                    table: table_name,
                    row_key: Some(row_key),
                }
                .to_string(),
            ),
            Origin::Chunk { .. } => None,
        })
        .collect();
    if !rows.is_empty() {
        let hidden_rows = rows.len().saturating_sub(MAX_GRAPH_ROWS);
        let shown: Vec<String> = rows.into_iter().take(MAX_GRAPH_ROWS).collect();
        let more = if hidden_rows > 0 {
            format!("; ... and {hidden_rows} more rows")
        } else {
            String::new()
        };
        writeln!(
            out,
            "\nFrom table rows (run_sql can read them): {}{more}",
            shown.join("; ")
        )?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// describe_class
// ---------------------------------------------------------------------------

/// Sample entity labels `describe_class` shows for a class.
const CLASS_SAMPLES: u32 = 10;

pub struct DescribeClassTool(pub ReaderDb);

#[derive(Deserialize, JsonSchema)]
pub struct DescribeClassArgs {
    /// The ontology class id to describe
    pub class_id: String,
}

impl Tool for DescribeClassTool {
    const NAME: &'static str = ToolName::DescribeClass.as_str();
    type Error = ToolError;
    type Args = DescribeClassArgs;
    type Output = String;

    fn description(&self) -> String {
        String::from(
            "Describe one ontology class: what it inherits from, its subclasses, its typed \
             properties, the relations it can take part in, the table it is mapped to, and how \
             many entities of it the knowledge graph holds, with a few example names. Use it to \
             get exact class and relation ids before calling search_graph, and to count the \
             entities of a class — a class listing stops at the node limit, this does not.",
        )
    }

    fn parameters(&self) -> serde_json::Value {
        DescribeClassArgs::schema()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let turn = Turn::of(context)?;
        let class_id = args.class_id.trim().to_owned();
        let step = turn.recorder.start(ToolName::DescribeClass, &class_id);
        let text = self
            .0
            .with_db(move |db| {
                let ontology = ontology_store::current(db)?;
                OntologyId::Class(&class_id).check(ontology.as_ref())?;
                let Some(ontology) = ontology else {
                    return Err(Error::Analysis(String::from(
                        "this workspace has no ontology",
                    )));
                };
                let classes = ontology.class_and_descendants(&class_id);
                let census = graph::store::class_census(db, &classes, CLASS_SAMPLES)?;
                let text = ClassDescription {
                    ontology: &ontology,
                    class_id: &class_id,
                    census: &census,
                }
                .to_string();
                Ok((text, census.samples.is_empty()))
            })
            .await;
        match text {
            Ok((text, unnamed)) => {
                // Example names are graph labels, as a graph search's are.
                if !unnamed {
                    turn.read_documents();
                }
                step.finish(format!("{} lines", text.lines().count()));
                Ok(text)
            }
            Err(e) => Err(step.fail(e.into())),
        }
    }
}

/// One class as the model sees it: the ontology's view of it plus what
/// the graph holds of it and its subclasses.
struct ClassDescription<'a> {
    ontology: &'a Ontology,
    class_id: &'a str,
    /// Entities of the class in the graph, and a few of their labels.
    census: &'a ClassCensus,
}

impl std::fmt::Display for ClassDescription<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            ontology,
            class_id,
            census: ClassCensus { total, samples },
        } = self;
        writeln!(
            f,
            "Class {class_id} (inherits: {})",
            ontology.ancestry(class_id).join(" -> ")
        )?;
        if let Some(class) = ontology.class(class_id) {
            if let Some(description) = class.description.as_deref() {
                writeln!(f, "  {description}")?;
            }
            if let Some(key) = class.key.as_deref() {
                writeln!(f, "Key property: {key}")?;
            }
        }

        let subclasses: Vec<&str> = ontology
            .subclasses(class_id)
            .iter()
            .map(|c| c.id.as_str())
            .collect();
        if subclasses.is_empty() {
            writeln!(f, "Subclasses: none")?;
        } else {
            writeln!(
                f,
                "Subclasses: {} (search_graph on this class covers them too)",
                Listed(&subclasses)
            )?;
        }

        let properties: Vec<String> = ontology
            .class_properties(class_id)
            .iter()
            .map(|id| match ontology.property(id) {
                Some(property) if property.values.is_empty() => {
                    format!("{id} ({})", property.kind.as_str())
                }
                Some(property) => format!("{id} (enum: {})", property.values.join(", ")),
                None => id.clone(),
            })
            .collect();
        if properties.is_empty() {
            writeln!(f, "Properties: none")?;
        } else {
            writeln!(f, "Properties: {}", properties.join(", "))?;
        }

        let ClassRelations { from, to } = ontology.relations_of(class_id);
        let from: Vec<String> = from
            .iter()
            .map(|r| format!("{} -> {}", r.id, r.range))
            .collect();
        let to: Vec<String> = to
            .iter()
            .map(|r| format!("{} from {}", r.id, r.domain))
            .collect();
        for (label, relations) in [("Relations from it", from), ("Relations to it", to)] {
            if relations.is_empty() {
                writeln!(f, "{label}: none")?;
            } else {
                writeln!(f, "{label}: {}", relations.join("; "))?;
            }
        }

        if let Some(mapping) = ontology.mapping_for(class_id) {
            writeln!(
                f,
                "Mapped table: {} (key column {}); run_sql can query it directly",
                mapping.table, mapping.key
            )?;
        }

        let samples: Vec<String> = samples
            .iter()
            .map(|label| OneLine(label).to_string())
            .collect();
        if *total == 0 {
            writeln!(f, "In the graph: no entities of this class")
        } else if samples.len() < usize::try_from(*total).unwrap_or(usize::MAX) {
            writeln!(
                f,
                "In the graph: {total} entities, for example {}",
                samples.join(", ")
            )
        } else {
            writeln!(
                f,
                "In the graph: {total} entities \u{2014} {}",
                samples.join(", ")
            )
        }
    }
}

/// A row's provenance as something the model can act on: the mapping
/// knows which column holds the key, so the row reads as a predicate
/// (`orders WHERE order_id = 'A-42'`) rather than as prose the model has
/// to guess a column name from. The mapping is what built the node in the
/// first place (design doc 6.3, table mapping).
struct RowReference<'a> {
    ontology: Option<&'a Ontology>,
    table: &'a str,
    row_key: Option<&'a str>,
}

impl std::fmt::Display for RowReference<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let table = self.table;
        let Some(key) = self.row_key else {
            return write!(f, "{table} (row key unknown)");
        };
        match self.ontology.and_then(|o| o.mapping_for_table(table)) {
            Some(mapping) => write!(
                f,
                "{} WHERE {} = '{}'",
                quote_ident(table),
                quote_ident(&mapping.key),
                key.replace('\'', "''")
            ),
            None => write!(f, "{table} row {key}"),
        }
    }
}
