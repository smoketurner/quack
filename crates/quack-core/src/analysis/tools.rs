use std::fmt::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rig::embeddings::EmbeddingModel;
use rig::tool::{Tool, ToolContext};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::storage::workspace::{
    ChunkScope, ChunkSearchResult, StatementKind, WorkspaceDb, quote_ident,
};
use crate::storage::writer::Writer;

use super::chart::{self, ChartSpec};
use super::events::TurnRecorder;
use super::policy::{RefusalFlag, WritePolicy};
use super::rerank::{self, Reranker};
use super::text_to_sql;
use crate::embedding::{Embedder, Input, Vector};
use crate::error::Error;
use crate::ontology::store as ontology_store;
use crate::ontology::{self, Ontology};

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
    /// [`open_reader`] returns when a clone would not serve.
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
        f: impl FnOnce(&WorkspaceDb) -> crate::error::Result<T> + Send + 'static,
    ) -> crate::error::Result<T>
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
        let has_temp = with_db(&self.0.writer, WorkspaceDb::has_temp_tables)
            .await
            .unwrap_or(false);
        if has_temp {
            self.0.degraded.store(true, Ordering::Relaxed);
        }
    }
}

/// Build a reader pool for a workspace handle's whole lifetime — once, not
/// once per turn, so acquiring one never waits behind a slow write on the
/// writer. `pool_size` genuine
/// [`WorkspaceDb::try_clone_reader`] clones, unless the writer already has
/// a temp table (the CLI's piped `stdin`) a clone could not see, in which
/// case every reader-routed tool shares the writer from the start. A clone
/// that fails (allocation, a `DuckDB` internal error) degrades that one
/// pool slot to the writer, with a warning, rather than the caller
/// aborting.
pub async fn open_reader(shared_db: &SharedDb, pool_size: u32) -> ReaderDb {
    let pool_size = pool_size.max(1);
    // One step on the writer for the temp check and every clone, instead
    // of `pool_size + 1` separate ones:
    // shortens how long a concurrent first-time open of this workspace
    // can overlap another one, and is simply less work.
    let outcome = with_db(shared_db, move |db| {
        if db.has_temp_tables()? {
            return Ok((true, Vec::new()));
        }
        let clones = (0..pool_size).map(|_| db.try_clone_reader()).collect();
        Ok((false, clones))
    })
    .await;
    let (has_temp_tables, clones) = outcome.unwrap_or_else(|e| {
        tracing::warn!(
            error = %e,
            "failed to open the reader pool; every read will share the writer"
        );
        (true, Vec::new())
    });
    if has_temp_tables {
        return ReaderDb::new(Arc::clone(shared_db));
    }
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
    ReaderDb::from_pool(readers, Arc::clone(shared_db))
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

/// Map a core DB error onto the `ToolError` shown to the model, keeping
/// its own category instead of always wrapping it as a query error — an
/// `Error::Analysis` already reads as `"analysis error: ..."`, so wrapping
/// it again in `ToolError::Query` would show the model
/// `"query error: analysis error: ..."`.
fn tool_error(e: crate::error::Error) -> ToolError {
    match e {
        Error::Analysis(msg) => ToolError::Analysis(msg),
        other => ToolError::Query(other.to_string()),
    }
}

/// Run `f` on the workspace's writer and await it (at the calling task's
/// priority); no async worker waits on the connection. `run_sql` writes
/// through this; every tool that only reads goes through
/// [`ReaderDb::with_db`] instead.
///
/// # Errors
///
/// Returns `f`'s error, or the writer's (a panic in `f`, a stopped writer).
pub(crate) async fn with_db<T>(
    db: &SharedDb,
    f: impl FnOnce(&WorkspaceDb) -> crate::error::Result<T> + Send + 'static,
) -> crate::error::Result<T>
where
    T: Send + 'static,
{
    db.run(f).await
}

/// Prefix of a `run_sql` or `describe_table` result that carries a `DuckDB`
/// error instead of rows. The model reads what follows and retries.
pub const SQL_ERROR_PREFIX: &str = "SQL error: ";

/// Message returned to the model when a write is refused.
pub const WRITE_REFUSED: &str = "This statement would modify the workspace and was not permitted. \
Do not retry it. Tell the user it needs write permission (re-run with --allow-write).";

/// Message returned to the model when a statement touches internal tables.
pub const INTERNAL_TABLE_REFUSED: &str =
    "This statement references quack's internal tables, which are not available to queries.";

/// What the gate decided about a statement.
enum Gate {
    /// Run it; the classification the gate already computed, so the
    /// caller can run a `Read` inside [`WorkspaceDb::read_only`] and a
    /// `Write` bare instead of re-deciding.
    Run(StatementKind),
    /// Do not run; hand this text back to the model.
    Reject(String),
}

/// Message returned to the caller when a statement would create a temp
/// table or view.
pub const TEMP_OBJECT_REFUSED: &str = "Temporary tables and views are not visible to every reader \
     for the rest of this workspace's session; create a regular table instead \
     (CREATE TABLE, without TEMP or TEMPORARY).";

/// Whether `sql` is a `CREATE [OR REPLACE] {TEMP | TEMPORARY} ...`
/// statement. `DuckDB` temp objects are connection-local, so one created
/// on the writer would be invisible to every reader-routed tool for the
/// rest of the workspace handle's life (they run on other connections);
/// every path that can run a write (`gate_statement`, the REST and MCP
/// `sql` handlers) refuses these outright with this as the friendly,
/// fail-fast message. It cannot be a complete check — a leading comment,
/// a semicolon before the real statement, or a multi-statement batch all
/// defeat a text match — so [`ReaderDb::observe_write`] is the actual
/// correctness backstop; this is the fast path for the obvious case.
#[must_use]
pub fn creates_temp_object(sql: &str) -> bool {
    let mut words = sql.split_whitespace().map(str::to_uppercase);
    if words.next().as_deref() != Some("CREATE") {
        return false;
    }
    let mut word = words.next();
    if word.as_deref() == Some("OR") {
        if words.next().as_deref() != Some("REPLACE") {
            return false;
        }
        word = words.next();
    }
    matches!(word.as_deref(), Some("TEMP" | "TEMPORARY"))
}

/// Classify a statement and apply the write policy. Takes and releases the
/// database lock itself so a permission prompt never holds it.
async fn gate_statement(
    db: &ReaderDb,
    sql: &str,
    policy: WritePolicy,
    refused: &RefusalFlag,
    recorder: &TurnRecorder,
) -> Result<Gate, ToolError> {
    // Classification is a parse: a reader serves, never the writer's line.
    let sql_owned = sql.to_owned();
    let kind = db
        .with_db(move |db| {
            if db.references_internal_table(&sql_owned)? {
                return Ok(None);
            }
            db.classify_statement(&sql_owned).map(Some)
        })
        .await
        .map_err(tool_error)?;
    let Some(kind) = kind else {
        return Ok(Gate::Reject(String::from(INTERNAL_TABLE_REFUSED)));
    };
    match kind {
        StatementKind::Read => Ok(Gate::Run(kind)),
        StatementKind::Invalid(msg) => Ok(Gate::Reject(format!("SQL syntax error: {msg}"))),
        StatementKind::Write => {
            if creates_temp_object(sql) {
                // A mutating statement the caller wanted to run did not
                // run, same as WRITE_REFUSED: AgentResponse::write_refused
                // should say so.
                refused.set();
                tracing::info!(sql, "refused a statement that would create a temp object");
                return Ok(Gate::Reject(String::from(TEMP_OBJECT_REFUSED)));
            }
            let allowed = match policy {
                WritePolicy::Allow => true,
                WritePolicy::Deny => false,
                WritePolicy::Ask => recorder.ask_permission(sql).await,
            };
            if allowed {
                Ok(Gate::Run(kind))
            } else {
                refused.set();
                tracing::info!(sql, "refused write statement from agent");
                Ok(Gate::Reject(String::from(WRITE_REFUSED)))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// run_sql
// ---------------------------------------------------------------------------

pub struct RunSqlTool {
    db: SharedDb,
    /// The workspace's reader, purely to call [`ReaderDb::observe_write`]
    /// after a write runs here: `run_sql` never reads through it.
    reader_db: ReaderDb,
    max_query_rows: u32,
    policy: WritePolicy,
    refused: RefusalFlag,
    recorder: TurnRecorder,
    /// Each statement run this turn with its parse tree blanked of
    /// literals (`WorkspaceDb::statement_shape`), to spot the model
    /// re-running one statement once per value.
    shapes: Mutex<Vec<(String, String)>>,
}

impl RunSqlTool {
    #[must_use]
    pub fn new(
        db: SharedDb,
        reader_db: ReaderDb,
        max_query_rows: u32,
        policy: WritePolicy,
        refused: RefusalFlag,
        recorder: TurnRecorder,
    ) -> Self {
        Self {
            db,
            reader_db,
            max_query_rows,
            policy,
            refused,
            recorder,
            shapes: Mutex::new(Vec::new()),
        }
    }

    /// An earlier statement this turn that `sql` repeats with only its
    /// literals changed: the one-query-per-group loop that burns the turn
    /// (a 20B model asked for deaths by state and weather ran the same
    /// GROUP BY once per state until it hit `max_turns`). Records `sql`
    /// for the calls after it.
    fn repeated_shape(&self, sql: &str, shape: Option<String>) -> Option<String> {
        let shape = shape?;
        let mut shapes = self.shapes.lock().ok()?;
        let earlier = shapes
            .iter()
            .find(|(earlier, earlier_shape)| {
                *earlier_shape == shape && earlier.trim() != sql.trim()
            })
            .map(|(earlier, _)| earlier.clone());
        shapes.push((sql.to_owned(), shape));
        earlier
    }
}

/// The note under a result whose statement repeats `earlier` with other
/// literals.
fn per_group_note(earlier: &str) -> String {
    let (preview, _) = super::events::preview_detail(earlier);
    format!(
        "Note: this statement repeats an earlier one with different literal values ({}). \
         Do not run it once per value: one statement covers every group at once with \
         GROUP BY (arg_max(label, measure) picks each group's top label), WHERE col IN (...), \
         or QUALIFY row_number() OVER (PARTITION BY group_col ORDER BY measure DESC) <= n.",
        preview.join(" ")
    )
}

#[derive(Deserialize, JsonSchema)]
pub struct RunSqlArgs {
    /// The SQL query to execute
    pub query: String,
}

impl Tool for RunSqlTool {
    const NAME: &'static str = "run_sql";
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
        serde_json::to_value(schemars::schema_for!(RunSqlArgs))
            .unwrap_or_else(|_| json!({"type": "object"}))
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let step = self.recorder.start(Self::NAME, args.query.trim());
        match gate_statement(
            &self.reader_db,
            &args.query,
            self.policy,
            &self.refused,
            &self.recorder,
        )
        .await?
        {
            Gate::Reject(message) => {
                step.finish("refused");
                Ok(message)
            }
            Gate::Run(kind) => {
                // DuckDB blocks for up to the query timeout: keep that off
                // the async workers, and keep the lock inside the blocking
                // thread with it. A statement the gate classified `Read`
                // still runs inside a read-only transaction, the same net
                // every other tool has, in case a future parser
                // divergence ever let a mutation through as `Read`.
                let sql = args.query.clone();
                let max_rows = self.max_query_rows;
                let read_only = matches!(kind, StatementKind::Read);
                let (results, shape) = with_db(&self.db, move |db| {
                    let shape = db.statement_shape(&sql)?;
                    let results = if read_only {
                        db.read_only(|db| Ok(db.execute_query_capped(&sql, max_rows)))?
                    } else {
                        db.execute_query_capped(&sql, max_rows)
                    };
                    Ok((results, shape))
                })
                .await
                .map_err(tool_error)?;
                if !read_only {
                    // Whatever ran might have created a temp object
                    // `creates_temp_object` did not catch (a leading
                    // comment, a multi-statement batch); check the
                    // writer's catalog regardless of whether the
                    // statement itself errored, since an earlier
                    // statement in a batch can have already run.
                    self.reader_db.observe_write().await;
                }
                match results {
                    Ok(results) => {
                        step.finish(format!("{} rows", results.total_rows));
                        let mut text =
                            text_to_sql::format_query_result(&results).map_err(tool_error)?;
                        if let Some(earlier) = self.repeated_shape(&args.query, shape) {
                            text.push('\n');
                            text.push_str(&per_group_note(&earlier));
                            text.push('\n');
                        }
                        text.push('\n');
                        text.push_str(&self.recorder.budget_note());
                        Ok(text)
                    }
                    // A failed statement is a result, not a tool failure: rig
                    // hides a tool error's message from the model, but DuckDB's
                    // text (candidate bindings, the missing table) is exactly
                    // what it needs to fix the statement and retry.
                    Err(e) => {
                        step.finish(format!("error: {e}"));
                        Ok(format!("{SQL_ERROR_PREFIX}{e}"))
                    }
                }
            }
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

pub struct SearchDocumentsTool<M> {
    db: ReaderDb,
    /// `None` runs keyword search alone: a workspace without an embedding
    /// provider still answers from its documents.
    embedding_model: Option<Embedder<M>>,
    default_top_k: u32,
    rrf_k: u32,
    reranker: Option<Arc<dyn Reranker>>,
    rerank_candidates: u32,
    recorder: TurnRecorder,
    /// The graph has nodes, so the `entity` argument can resolve. Without
    /// one the argument is left out of the tool's schema and description:
    /// a model shown it tries it, is refused, and spends a second round
    /// trip (and twice the tokens, measured live) reaching the same answer.
    graph_enabled: bool,
}

impl<M> SearchDocumentsTool<M> {
    pub fn new(
        db: ReaderDb,
        embedding_model: Option<Embedder<M>>,
        default_top_k: u32,
        rrf_k: u32,
        recorder: TurnRecorder,
    ) -> Self {
        Self {
            db,
            embedding_model,
            default_top_k,
            rrf_k,
            reranker: None,
            rerank_candidates: 0,
            recorder,
            graph_enabled: false,
        }
    }

    /// Offer the `entity` argument: only when the graph has nodes.
    #[must_use]
    pub fn with_graph(mut self, graph_enabled: bool) -> Self {
        self.graph_enabled = graph_enabled;
        self
    }

    /// Over-fetch `candidates` and let `reranker` order them before the
    /// top `k` are returned.
    #[must_use]
    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>, candidates: u32) -> Self {
        self.reranker = Some(reranker);
        self.rerank_candidates = candidates;
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
    pub entity: Option<String>,
}

/// Map what the model passed (an id, an id prefix, or a file name) to
/// document ids. Anything that matches nothing is an error naming the
/// documents that exist, so the model retries instead of getting an empty
/// result it reads as "the workspace has nothing on this".
fn resolve_document_ids(db: &WorkspaceDb, wanted: &[String]) -> crate::error::Result<Vec<String>> {
    if wanted.is_empty() {
        return Ok(Vec::new());
    }
    let documents = db.list_documents()?;
    let mut resolved = Vec::with_capacity(wanted.len());
    for want in wanted {
        let want = want.trim();
        let found = documents
            .iter()
            .find(|d| d.id == want || d.filename == want)
            .or_else(|| {
                documents
                    .iter()
                    .find(|d| !want.is_empty() && d.id.starts_with(want))
            });
        let Some(d) = found else {
            let known: Vec<String> = documents
                .iter()
                .map(|d| format!("{} ({})", d.id, d.filename))
                .collect();
            return Err(Error::Analysis(format!(
                "no document matches '{want}'; pass an id from list_documents or omit \
                 document_ids to search everything. Documents: {}",
                known.join(", ")
            )));
        };
        resolved.push(d.id.clone());
    }
    Ok(resolved)
}

impl<M> Tool for SearchDocumentsTool<M>
where
    M: EmbeddingModel + Send + Sync,
{
    const NAME: &'static str = "search_documents";
    type Error = ToolError;
    type Args = SearchDocumentsArgs;
    type Output = String;

    fn description(&self) -> String {
        let mut text = String::from(
            "Search the ingested documents by meaning and by keyword. Returns the most relevant \
             text chunks, each numbered [n] with its source file, page, and heading, for citing \
             in the answer",
        );
        if self.graph_enabled {
            text.push_str(
                ", and names the graph entities each chunk was the source of. Pass entity to \
                 search only the passages one entity was extracted from",
            );
        }
        text.push('.');
        text
    }

    fn parameters(&self) -> serde_json::Value {
        let mut schema = serde_json::to_value(schemars::schema_for!(SearchDocumentsArgs))
            .unwrap_or_else(|_| json!({"type": "object"}));
        if !self.graph_enabled
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
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let entity = args
            .entity
            .as_deref()
            .map(str::trim)
            .filter(|e| !e.is_empty());
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
        let step = self.recorder.start(Self::NAME, &detail);
        let query_vec: Option<Vector> = match &self.embedding_model {
            None => None,
            Some(model) => {
                match cached_embed(model, &self.recorder, Input::Query(args.query.clone())).await {
                    Ok(vector) => Some(vector),
                    Err(e) => {
                        step.finish(format!("error: {e}"));
                        return Err(ToolError::Embedding(e.to_string()));
                    }
                }
            }
        };

        let top_k = args
            .top_k
            .unwrap_or(self.default_top_k)
            .clamp(1, MAX_SEARCH_TOP_K);
        let fetch = if self.reranker.is_some() {
            top_k.max(self.rerank_candidates)
        } else {
            top_k
        };

        // The entity's own embedding, not the query's: it resolves a label,
        // so `search_documents(query, entity)` is one embed call each,
        // cached against a later call (search_graph, find_path) that
        // resolves the same label again this turn.
        let entity_vec = match entity {
            Some(entity) => {
                label_embedding(self.embedding_model.as_ref(), &self.recorder, entity).await?
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
                let mut scope = ChunkScope::documents(resolve_document_ids(db, &document_ids)?);
                if let Some(entity) = entity.as_deref() {
                    scope = scope.and_chunks(entity_chunks(db, entity, entity_vec.as_deref())?);
                }
                match &query_vec {
                    Some(vector) => db.search_hybrid_chunks(&query, vector, fetch, rrf_k, &scope),
                    None => db.search_keyword_chunks(&query, fetch, &scope),
                }
            })
            .await;
        let results = match results {
            Ok(results) => results,
            Err(e) => {
                let e = tool_error(e);
                step.finish(format!("error: {e}"));
                return Err(e);
            }
        };
        let (results, note) = match &self.reranker {
            Some(reranker) => {
                let keep = usize::try_from(top_k).unwrap_or(usize::MAX);
                let (kept, outcome) =
                    rerank::apply(reranker.as_ref(), &args.query, results, keep).await;
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
        let chunk_ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
        // Best effort: the annotation is extra context, so a graph that
        // cannot be read must not fail a search that already succeeded.
        let entities = self
            .db
            .with_db(move |db| graph::store::entities_of_chunks(db, &chunk_ids, CHUNK_ENTITIES))
            .await
            .unwrap_or_default();
        let first = self.recorder.citations().register(&results);
        format_search_results(&results, first, &entities).map_err(Into::into)
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
    embedding: Option<&[f32]>,
) -> crate::error::Result<Vec<String>> {
    let nodes = graph::traverse::resolve_entry(db, entity, None, embedding)?;
    if nodes.is_empty() {
        let suggestions = graph::traverse::suggest_entities(db, entity, None, embedding)?;
        if suggestions.is_empty() {
            return Err(Error::Analysis(format!(
                "no entity '{entity}' in the knowledge graph; drop the entity argument to search \
                 every document"
            )));
        }
        return Err(Error::Analysis(format!(
            "no entity '{entity}' in the knowledge graph; the closest labels are: {}",
            suggestions.join(", ")
        )));
    }
    let ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
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
    first_marker: u32,
    entities: &std::collections::BTreeMap<String, Vec<String>>,
) -> Result<String, std::fmt::Error> {
    if results.is_empty() {
        return Ok(String::from(
            "No relevant chunks found. Tell the user the documents do not appear to cover this.",
        ));
    }
    let mut out = String::from(
        "Retrieved chunks. Cite each fact you use with the chunk's [n] marker at the end of the sentence.\n\n",
    );
    for (i, chunk) in results.iter().enumerate() {
        let n = first_marker.saturating_add(u32::try_from(i).unwrap_or(u32::MAX));
        let page = chunk.page.map_or(String::new(), |p| format!(", page {p}"));
        let heading = chunk
            .heading
            .as_deref()
            .map_or(String::new(), |h| format!(", under \"{h}\""));
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
            "[{n}] {}{page}{heading} (document_id: {}, chunk {}, score {:.4}{graph})",
            chunk.filename, chunk.document_id, chunk.chunk_index, chunk.score
        )?;
        writeln!(out, "{}", chunk.content.trim())?;
        writeln!(out)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// describe_table
// ---------------------------------------------------------------------------

pub struct DescribeTableTool {
    db: ReaderDb,
    recorder: TurnRecorder,
}

impl DescribeTableTool {
    #[must_use]
    pub fn new(db: ReaderDb, recorder: TurnRecorder) -> Self {
        Self { db, recorder }
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct DescribeTableArgs {
    /// Name of the table to describe
    pub table_name: String,
}

impl Tool for DescribeTableTool {
    const NAME: &'static str = "describe_table";
    type Error = ToolError;
    type Args = DescribeTableArgs;
    type Output = String;

    fn description(&self) -> String {
        String::from("Returns column names, types, and 3 sample rows for a table in the workspace.")
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(DescribeTableArgs))
            .unwrap_or_else(|_| json!({"type": "object"}))
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let step = self.recorder.start(Self::NAME, &args.table_name);
        let table_name = args.table_name.clone();
        let outcome = self
            .db
            .with_db(move |db| {
                Ok(match db.describe_table(&table_name) {
                    Ok(d) => Ok(d),
                    Err(e) => {
                        let tables = db.list_tables().unwrap_or_default();
                        Err((e.to_string(), tables))
                    }
                })
            })
            .await
            .map_err(tool_error)?;
        let desc = match outcome {
            Ok(d) => d,
            Err((message, tables)) => {
                step.finish(format!("error: {message}"));
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

pub struct ListTablesTool {
    db: ReaderDb,
    recorder: TurnRecorder,
}

impl ListTablesTool {
    #[must_use]
    pub fn new(db: ReaderDb, recorder: TurnRecorder) -> Self {
        Self { db, recorder }
    }
}

impl Tool for ListTablesTool {
    const NAME: &'static str = "list_tables";
    type Error = ToolError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        String::from("Returns all user-created tables in the workspace.")
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let step = self.recorder.start(Self::NAME, "");
        // One reader round trip for the listing and every table's row
        // count, rather than one per table; each count is its own
        // timeout-guarded statement, so one huge table cannot pin the
        // transaction indefinitely.
        let tables: Vec<(String, Option<i64>)> = self
            .db
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
            .await
            .map_err(tool_error)?;
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

pub struct ListDocumentsTool {
    db: ReaderDb,
    recorder: TurnRecorder,
}

impl ListDocumentsTool {
    #[must_use]
    pub fn new(db: ReaderDb, recorder: TurnRecorder) -> Self {
        Self { db, recorder }
    }
}

impl Tool for ListDocumentsTool {
    const NAME: &'static str = "list_documents";
    type Error = ToolError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        String::from("Returns all ingested documents with their status.")
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let step = self.recorder.start(Self::NAME, "");
        let docs = self
            .db
            .with_db(WorkspaceDb::list_documents)
            .await
            .map_err(tool_error)?;
        step.finish(format!("{} documents", docs.len()));
        if docs.is_empty() {
            return Ok(String::from("No documents found in this workspace."));
        }
        let mut output = String::from("Documents:\n");
        for doc in &docs {
            let title = doc
                .title
                .as_deref()
                .map_or(String::new(), |t| format!(", title: {t}"));
            writeln!(
                output,
                "- {} (id: {}, status: {}, type: {}, source: {}{title})",
                doc.filename,
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

pub struct CreateChartTool {
    db: ReaderDb,
    chart_spec: Arc<Mutex<Option<ChartSpec>>>,
    recorder: TurnRecorder,
}

impl CreateChartTool {
    pub fn new(
        db: ReaderDb,
        chart_spec: Arc<Mutex<Option<ChartSpec>>>,
        recorder: TurnRecorder,
    ) -> Self {
        Self {
            db,
            chart_spec,
            recorder,
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
    #[schemars(with = "crate::analysis::chart::ChartKind")]
    pub kind: String,
    /// Column for the x axis (category labels; slice names for pie)
    pub x: String,
    /// Numeric column for the y axis (slice values for pie)
    pub y: String,
    /// Chart title
    pub title: String,
}

impl Tool for CreateChartTool {
    const NAME: &'static str = "create_chart";
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
        serde_json::to_value(schemars::schema_for!(CreateChartArgs))
            .unwrap_or_else(|_| json!({"type": "object"}))
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let step = self.recorder.start(Self::NAME, args.sql.trim());
        // Charts are read-only: never prompt, never write.
        if let Gate::Reject(message) = gate_statement(
            &self.db,
            &args.sql,
            WritePolicy::Deny,
            &RefusalFlag::default(),
            &self.recorder,
        )
        .await?
        {
            step.finish("rejected");
            return Ok(format!("Chart query rejected. {message}"));
        }

        let sql = args.sql.clone();
        let results = self.db.with_db(move |db| db.execute_query(&sql)).await;
        let results = match results {
            Ok(r) => r,
            Err(e) => {
                let e = tool_error(e);
                step.finish(format!("error: {e}"));
                return Err(e);
            }
        };

        if results.rows.is_empty() {
            step.finish("0 rows");
            return Ok(String::from(
                "Query returned no rows — cannot generate chart.",
            ));
        }

        let spec =
            match chart::generate_chart_spec(&results, &args.kind, &args.x, &args.y, &args.title) {
                Ok(spec) => spec,
                Err(e) => {
                    step.finish(format!("error: {e}"));
                    return Ok(format!("Chart not created: {e}"));
                }
            };

        let summary = format!(
            "{} chart \"{}\" with {} points ({} by {})",
            spec.kind.as_str(),
            spec.title,
            spec.points(),
            args.y,
            args.x
        );
        {
            let mut guard = self
                .chart_spec
                .lock()
                .map_err(|e| ToolError::Analysis(format!("mutex poisoned: {e}")))?;
            *guard = Some(spec);
        }

        step.finish(format!("{} points", results.rows.len()));
        Ok(format!(
            "Chart created and shown to the user: {summary}. Describe what it shows; do not repeat the data."
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::embedding::{Dimension, Profile, Prompts};
    use crate::storage::workspace::NewDocument;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail_test(msg: &str) -> ! {
        panic!("{msg}")
    }

    /// Counts calls to `embed_texts` so a test can assert a cache actually
    /// prevented one, rather than merely returning a plausible-looking
    /// vector either way.
    struct CountingEmbeddingModel {
        calls: Arc<AtomicUsize>,
    }

    impl rig::embeddings::EmbeddingModel for CountingEmbeddingModel {
        const MAX_DOCUMENTS: usize = 1024;
        type Client = ();

        fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
            Self {
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn ndims(&self) -> usize {
            4
        }

        fn embed_texts(
            &self,
            texts: impl IntoIterator<Item = String> + Send,
        ) -> impl std::future::Future<
            Output = Result<Vec<rig::embeddings::Embedding>, rig::embeddings::EmbeddingError>,
        > + Send {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let result = texts
                .into_iter()
                .map(|text| rig::embeddings::Embedding {
                    document: text,
                    vec: vec![0.1_f64; 4],
                })
                .collect();
            std::future::ready(Ok(result))
        }
    }

    #[tokio::test]
    async fn cached_embed_asks_the_model_only_once_per_text() {
        let calls = Arc::new(AtomicUsize::new(0));
        let model = Embedder::new(
            CountingEmbeddingModel {
                calls: Arc::clone(&calls),
            },
            Profile::new("m", Dimension::new(4), Prompts::default()),
        );
        let (sink, _rx) = crate::analysis::events::channel();
        let recorder = TurnRecorder::new(sink);
        let name = |text: &str| Input::Similarity(text.to_owned());

        let first = cached_embed(&model, &recorder, name("Acme")).await;
        let second = cached_embed(&model, &recorder, name("Acme")).await;
        let other = cached_embed(&model, &recorder, name("Beta")).await;
        let as_query = cached_embed(&model, &recorder, Input::Query("Acme".into())).await;

        assert!(first.is_ok());
        assert_eq!(first.as_ref().ok(), second.as_ref().ok());
        assert!(other.is_ok());
        assert!(as_query.is_ok());
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "one call for \"Acme\", one for the different text \"Beta\", one for \"Acme\" \
             as a query rather than a name, none for the repeat"
        );
    }

    #[test]
    fn unknown_class_and_relation_ids_are_refused_with_the_real_ones() {
        let ontology = Ontology::builtin_default();
        assert!(check_class(Some(&ontology), "organization").is_ok());
        // The root class and `mentions` are implicit: never declared, always valid.
        assert!(check_class(Some(&ontology), ontology::ROOT_CLASS).is_ok());
        assert!(check_relation(Some(&ontology), ontology::MENTIONS_RELATION).is_ok());
        assert!(check_relation(Some(&ontology), "works_at").is_ok());

        let err = check_class(Some(&ontology), "organisation")
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            err.contains("no class 'organisation'") && err.contains("organization"),
            "{err}"
        );
        let err = check_relation(Some(&ontology), "employed_by")
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            err.contains("no relation 'employed_by'") && err.contains("works_at"),
            "{err}"
        );
        assert!(check_class(None, "organization").is_err());
        assert!(check_relation(None, "works_at").is_err());
    }

    /// Two chunks about hail, the denser one second, for the search tests.
    async fn seed_hail_chunks(db: &SharedDb) {
        use crate::storage::workspace::{NewChunk, NewDocument};
        db.run(|guard| {
            guard.insert_document(
                &NewDocument::new("d", "storms.md", "text/markdown", 1)
                    .with_status(crate::storage::workspace::DocumentStatus::Ready),
            )?;
            for (i, text) in [
                "Hail fell on Denver.",
                "Hail and hail again in Denver county.",
            ]
            .iter()
            .enumerate()
            {
                guard.insert_chunk(&NewChunk {
                    id: &format!("c{i}"),
                    document_id: "d",
                    chunk_index: u32::try_from(i).unwrap_or(0),
                    content: text,
                    heading: None,
                    page: None,
                    embedding: None,
                })?;
            }
            Ok(())
        })
        .await
        .unwrap_or_else(|e| fail_test(&e.to_string()));
    }

    #[test]
    fn an_entity_filter_resolves_to_its_chunks_or_says_why_it_cannot() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail_test(&e.to_string()));
        assert!(
            db.insert_document(
                &NewDocument::new("doc-1", "notes.md", "text/markdown", 1)
                    .with_status(crate::storage::workspace::DocumentStatus::Ready)
            )
            .is_ok()
        );
        assert!(
            db.insert_chunk(&crate::storage::workspace::NewChunk {
                id: "c1",
                document_id: "doc-1",
                chunk_index: 0,
                content: "Acme ships to Kenya.",
                heading: None,
                page: None,
                embedding: None,
            })
            .is_ok()
        );
        let node = |label: &str| crate::graph::store::NewNode {
            label: String::from(label),
            class_id: String::from("organization"),
            properties: json!({}),
            provisional: false,
        };
        let acme = graph::store::upsert_node(&db, &node("Acme"))
            .unwrap_or_else(|e| fail_test(&e.to_string()));
        assert!(
            graph::store::add_provenance(
                &db,
                &acme,
                &graph::store::Source::chunk("doc-1", "c1", 1.0)
            )
            .is_ok()
        );
        let from_table = graph::store::upsert_node(&db, &node("Orgenics"))
            .unwrap_or_else(|e| fail_test(&e.to_string()));
        assert!(
            graph::store::add_provenance(
                &db,
                &from_table,
                &graph::store::Source::row("vendors", "V-1")
            )
            .is_ok()
        );

        assert_eq!(
            entity_chunks(&db, "acme", None).unwrap_or_default(),
            [String::from("c1")],
            "the entry point normalizes the label"
        );

        // In the graph, but only from a table: the model is told to use
        // search_graph rather than reading an empty document search.
        let tables_only = entity_chunks(&db, "Orgenics", None)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            tables_only.contains("only from table rows") && tables_only.contains("search_graph"),
            "{tables_only}"
        );

        // Not in the graph at all, with and without a near label.
        let near = entity_chunks(&db, "Acme Corporation", None)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            near.contains("the closest labels are: Acme (organization)"),
            "{near}"
        );
        let nothing = entity_chunks(&db, "Helsinki", None)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            nothing.contains("drop the entity argument") && !nothing.contains("closest"),
            "{nothing}"
        );
    }

    #[test]
    fn row_provenance_becomes_a_predicate_when_the_class_is_mapped() {
        let mut ontology = Ontology::builtin_default();
        ontology.mappings.push(crate::ontology::Mapping {
            table: String::from("orders"),
            class: String::from("organization"),
            key: String::from("order id"),
            properties: BTreeMap::new(),
            relations: Vec::new(),
        });
        assert_eq!(
            row_reference(Some(&ontology), "orders", Some("A-42")),
            "\"orders\" WHERE \"order id\" = 'A-42'"
        );
        // A quote in the key is escaped, not left to break the statement.
        assert_eq!(
            row_reference(Some(&ontology), "orders", Some("O'Hara")),
            "\"orders\" WHERE \"order id\" = 'O''Hara'"
        );
        // Without a mapping the column is unknown: say the row, do not guess.
        assert_eq!(
            row_reference(Some(&ontology), "audit", Some("7")),
            "audit row 7"
        );
        assert_eq!(row_reference(None, "orders", Some("7")), "orders row 7");
        assert_eq!(
            row_reference(Some(&ontology), "orders", None),
            "orders (row key unknown)"
        );
    }

    #[test]
    fn describe_class_covers_the_ontology_and_the_graph() {
        let ontology = Ontology::builtin_default();
        let census = (3, vec![String::from("Ada"), String::from("Alan")]);
        let text = describe_class(&ontology, "person", &census);
        assert!(
            text.contains("Class person (inherits: person -> entity)"),
            "{text}"
        );
        assert!(
            text.contains("Properties: email (string), title (string)"),
            "{text}"
        );
        assert!(
            text.contains("Relations from it: works_at -> organization"),
            "{text}"
        );
        // Inherited from `entity`, which every class is a subclass of.
        assert!(text.contains("part_of"), "{text}");
        assert!(text.contains("Subclasses: none"), "{text}");
        assert!(
            text.contains("In the graph: 3 entities, for example Ada, Alan"),
            "{text}"
        );

        let empty = describe_class(&ontology, "product", &(0, Vec::new()));
        assert!(empty.contains("In the graph: no entities"), "{empty}");
        assert!(empty.contains("Properties: none"), "{empty}");
        assert!(empty.contains("produced_by -> organization"), "{empty}");
        // `part_of` ranges over `entity`, so every class is a target of it.
        assert!(
            empty.contains("Relations to it: part_of from entity"),
            "{empty}"
        );
    }

    #[test]
    fn long_id_lists_are_counted_rather_than_pasted() {
        let ids: Vec<String> = (0..(LISTED_IDS + 5)).map(|i| format!("c{i:03}")).collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let text = listed(&refs);
        assert!(text.starts_with("c000, c001"), "{text}");
        assert!(text.ends_with("... and 5 more"), "{text}");
        assert!(!text.contains("c040"), "{text}");
        assert_eq!(listed(&["a", "b"]), "a, b");
    }

    #[test]
    fn an_oversized_rendering_is_cut_but_keeps_its_totals() {
        let mut lines: Vec<String> = Vec::new();
        for i in 0..400 {
            lines.push(format!(
                "Node {i:03} (storm_event) {{event_type: Tornado, state: OKLAHOMA}}"
            ));
        }
        let body = format!("{}\n", lines.join("\n"));
        let text = format!("{body}200 of 1529 matching nodes, 0 edges, 200 sources — cut off\n");
        let trimmed = trim_graph_text(&text, MAX_GRAPH_TEXT_CHARS);
        assert!(
            trimmed.chars().count() < text.chars().count(),
            "it should be shorter"
        );
        assert!(trimmed.contains("Node 000"), "{trimmed}");
        assert!(!trimmed.contains("Node 399"), "the tail is cut");
        // The summary line survives, so the totals are never what gets lost.
        assert!(trimmed.contains("200 of 1529 matching nodes"), "{trimmed}");
        assert!(
            trimmed.contains("more lines not shown") && trimmed.contains("describe_class"),
            "{trimmed}"
        );
        // Comfortably inside a turn's budget once cut.
        assert!(
            trimmed.chars().count() < MAX_GRAPH_TEXT_CHARS + 400,
            "{}",
            trimmed.chars().count()
        );
    }

    #[test]
    fn an_empty_result_says_which_kind_of_empty_it_is() {
        let nothing = empty_graph_text(false, &[]).unwrap_or_default();
        assert!(
            nothing.contains("the graph has nothing on this"),
            "{nothing}"
        );

        let suggested =
            empty_graph_text(false, &[String::from("Acme (organization)")]).unwrap_or_default();
        assert!(
            suggested.contains("Acme (organization)") && suggested.contains("Search again"),
            "{suggested}"
        );

        // Provisional matches were found and then stripped: the workspace
        // has the entity, query mode just will not answer from it.
        let stripped = empty_graph_text(true, &[]).unwrap_or_default();
        assert!(
            stripped.contains("provisional") && stripped.contains("quack graph review"),
            "{stripped}"
        );
        assert!(!stripped.contains("No matching entities"), "{stripped}");
    }

    #[test]
    fn document_ids_resolve_by_id_prefix_or_filename() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail_test(&e.to_string()));
        assert!(
            db.insert_document(
                &NewDocument::new("01a0-first", "policy.pdf", "application/pdf", 1)
                    .with_status(crate::storage::workspace::DocumentStatus::Ready)
            )
            .is_ok()
        );
        assert!(
            db.insert_document(
                &NewDocument::new("01b0-second", "notes.md", "text/markdown", 1)
                    .with_status(crate::storage::workspace::DocumentStatus::Ready)
            )
            .is_ok()
        );
        let by_name = resolve_document_ids(&db, &[String::from("policy.pdf")]);
        assert!(by_name.is_ok_and(|ids| ids == ["01a0-first"]));
        let by_prefix = resolve_document_ids(&db, &[String::from("01b0")]);
        assert!(by_prefix.is_ok_and(|ids| ids == ["01b0-second"]));
        let by_id =
            resolve_document_ids(&db, &[String::from("01a0-first"), String::from("notes.md")]);
        assert!(by_id.is_ok_and(|ids| ids == ["01a0-first", "01b0-second"]));
        assert!(resolve_document_ids(&db, &[]).is_ok_and(|ids| ids.is_empty()));
        let err = resolve_document_ids(&db, &[String::from("missing.pdf")]).err();
        assert!(err.is_some_and(|e| {
            let text = e.to_string();
            text.contains("no document matches 'missing.pdf'") && text.contains("policy.pdf")
        }));
    }

    fn hit(n: u32, filename: &str, content: &str) -> ChunkSearchResult {
        ChunkSearchResult {
            id: format!("c{n}"),
            content: content.to_owned(),
            document_id: String::from("doc-1"),
            chunk_index: n,
            filename: filename.to_owned(),
            heading: (n == 0).then(|| String::from("Exclusions")),
            page: (n == 0).then_some(12),
            score: 0.125,
        }
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn format_search_results_numbers_hits_with_filename() {
        let out = format_search_results(
            &[
                hit(0, "policy.pdf", "  Flood is excluded.  "),
                hit(1, "faq.md", "Claims close in 30 days."),
            ],
            1,
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(
            out.contains(
                "\n[1] policy.pdf, page 12, under \"Exclusions\" (document_id: doc-1, chunk 0, score 0.1250)\n"
            ),
            "{out}"
        );
        assert!(out.starts_with("Retrieved chunks. Cite"));
        assert!(out.contains("\nFlood is excluded.\n"));
        assert!(out.contains("[2] faq.md (document_id: doc-1, chunk 1, score 0.1250)\n"));
        assert!(out.contains("Claims close in 30 days."));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn search_results_name_the_entities_a_chunk_was_the_source_of() {
        let entities = BTreeMap::from([(
            String::from("c0"),
            vec![
                String::from("OKLAHOMA (state)"),
                String::from("EF4 (scale)"),
            ],
        )]);
        let out = format_search_results(
            &[hit(0, "efscale.html", "Damage indicators.")],
            1,
            &entities,
        )
        .unwrap();
        // On the metadata line, not above the passage: a line of its own
        // gets quoted back as though it were the document's text.
        assert!(
            out.contains("graph entities: OKLAHOMA (state), EF4 (scale))"),
            "{out}"
        );
        assert!(out.contains("\nDamage indicators.\n"), "{out}");
        // A chunk with no entities keeps the plain metadata line.
        let none =
            format_search_results(&[hit(0, "efscale.html", "x")], 1, &BTreeMap::new()).unwrap();
        assert!(!none.contains("graph entities"), "{none}");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn format_search_results_continues_numbering() {
        let out = format_search_results(&[hit(0, "a.md", "x")], 5, &BTreeMap::new()).unwrap();
        assert!(out.contains("\n[5] a.md"), "{out}");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn format_search_results_empty_tells_model_to_say_so() {
        let out = format_search_results(&[], 4, &BTreeMap::new()).unwrap();
        assert!(out.contains("No relevant chunks found"));
    }

    fn shared_db() -> SharedDb {
        Arc::new(
            Writer::spawn(
                WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| unreachable_db(&e.to_string())),
            )
            .unwrap_or_else(|e| fail_test(&e.to_string())),
        )
    }

    #[expect(clippy::panic, reason = "test helper: in-memory DuckDB must open")]
    fn unreachable_db(msg: &str) -> WorkspaceDb {
        panic!("in-memory DuckDB failed to open: {msg}");
    }

    #[tokio::test]
    async fn gate_runs_reads_and_rejects_internal_tables_and_syntax_errors() {
        let (sink, _rx) = super::super::events::channel();
        let recorder = TurnRecorder::new(sink);
        let db = shared_db();
        let refused = RefusalFlag::default();
        assert!(matches!(
            gate_statement(
                &ReaderDb::new(Arc::clone(&db)),
                "SELECT 1",
                WritePolicy::Deny,
                &refused,
                &recorder
            )
            .await,
            Ok(Gate::Run(StatementKind::Read))
        ));
        assert!(matches!(
            gate_statement(&ReaderDb::new(Arc::clone(&db)), "SELECT * FROM _quack_chunks", WritePolicy::Allow, &refused, &recorder).await,
            Ok(Gate::Reject(m)) if m == INTERNAL_TABLE_REFUSED
        ));
        assert!(matches!(
            gate_statement(&ReaderDb::new(Arc::clone(&db)), "SELEC 1", WritePolicy::Allow, &refused, &recorder).await,
            Ok(Gate::Reject(m)) if m.starts_with("SQL syntax error")
        ));
        assert!(!refused.was_refused());
    }

    #[test]
    fn creates_temp_object_detects_temp_and_temporary_create_statements() {
        assert!(creates_temp_object("CREATE TEMP TABLE t AS SELECT 1"));
        assert!(creates_temp_object("create temporary table t(a int)"));
        assert!(creates_temp_object(
            "CREATE OR REPLACE TEMP TABLE t AS SELECT 1"
        ));
        assert!(!creates_temp_object("CREATE TABLE t(a INT)"));
        assert!(!creates_temp_object("CREATE OR REPLACE TABLE t(a INT)"));
        assert!(!creates_temp_object("SELECT 1"));
    }

    /// The text matcher is a fast path for the obvious case, not a
    /// complete check — these four all reach the writer undetected. Pinned
    /// here so the limitation is explicit; `observe_write` (tested below)
    /// is what actually closes the gap they leave.
    #[test]
    fn creates_temp_object_misses_known_bypasses() {
        assert!(!creates_temp_object(
            "-- scratch\nCREATE TEMP TABLE c1(a INT)"
        ));
        assert!(!creates_temp_object(
            "/* scratch */ CREATE TEMP TABLE c2(a INT)"
        ));
        assert!(!creates_temp_object(
            "SELECT 1; CREATE TEMP TABLE c3(a INT)"
        ));
        assert!(!creates_temp_object("; CREATE TEMP TABLE c4(a INT)"));
    }

    /// The actual correctness backstop for the bypasses above: once a temp
    /// object appears on the writer by any means, `observe_write` degrades
    /// every clone of that `ReaderDb` to the writer, so a table a bypass
    /// created is still visible to reads.
    #[tokio::test]
    async fn observe_write_degrades_every_clone_once_a_temp_table_appears() {
        let db = shared_db();
        let reader_db = open_reader(&db, 2).await;
        let reader_clone = reader_db.clone();

        // Before the write: the reader pool is real clones, so a temp
        // table on the writer is not yet visible to them.
        with_db(&db, |db| {
            db.execute_statement("CREATE TEMP TABLE scratch AS SELECT 1 AS a")
        })
        .await
        .unwrap_or_else(|e| fail_test(&e.to_string()));
        assert!(
            reader_db
                .with_db(|db| db.execute_query("SELECT * FROM scratch"))
                .await
                .is_err()
        );

        reader_db.observe_write().await;

        // Now every clone of the ReaderDb sees it, because the degrade is
        // sticky state shared behind the `Arc`, not per-clone.
        assert!(
            reader_db
                .with_db(|db| db.execute_query("SELECT * FROM scratch"))
                .await
                .is_ok()
        );
        assert!(
            reader_clone
                .with_db(|db| db.execute_query("SELECT * FROM scratch"))
                .await
                .is_ok()
        );
    }

    /// The four `creates_temp_object` bypasses: a leading line comment, a
    /// leading block comment, a leading semicolon, and a harmless first
    /// statement ahead of the real one. Each reaches `run_sql`'s writer
    /// undetected (`creates_temp_object_misses_known_bypasses` pins that),
    /// but correctness does not rest on the detector — this runs each one
    /// through the real tool and then reads the table back through the
    /// reader, proving `observe_write`'s post-write degrade catches what
    /// the pre-check misses. Asserting only that the detector misses them
    /// would just re-encode the brittleness the sticky degrade replaces.
    #[tokio::test]
    async fn run_sql_bypasses_are_still_visible_to_reads_after_they_run() {
        for bypass in [
            "-- scratch\nCREATE TEMP TABLE scratch(a INT)",
            "/* scratch */ CREATE TEMP TABLE scratch(a INT)",
            "; CREATE TEMP TABLE scratch(a INT)",
            "SELECT 1; CREATE TEMP TABLE scratch(a INT)",
        ] {
            let db = shared_db();
            let reader_db = open_reader(&db, 2).await;
            let (sink, _rx) = super::super::events::channel();
            let recorder = TurnRecorder::new(sink);
            let tool = RunSqlTool::new(
                Arc::clone(&db),
                reader_db.clone(),
                100,
                WritePolicy::Allow,
                RefusalFlag::default(),
                recorder,
            );
            let out = tool
                .call(
                    &mut ToolContext::new(),
                    RunSqlArgs {
                        query: String::from(bypass),
                    },
                )
                .await
                .unwrap_or_else(|e| fail_test(&format!("{bypass}: tool call failed: {e}")));
            assert!(
                !out.starts_with(SQL_ERROR_PREFIX),
                "{bypass}: statement did not run: {out}"
            );

            let visible = reader_db
                .with_db(|db| db.execute_query("SELECT * FROM scratch"))
                .await;
            assert!(
                visible.is_ok(),
                "{bypass}: reader still cannot see the bypass table: {visible:?}"
            );
        }
    }

    /// A temp table created mid-turn would be invisible to every
    /// reader-routed tool for the rest of the turn, so `run_sql` refuses to
    /// create one outright rather than let that happen.
    #[tokio::test]
    async fn gate_refuses_statements_that_create_temp_tables() {
        let (sink, _rx) = super::super::events::channel();
        let recorder = TurnRecorder::new(sink);
        let db = shared_db();
        let refused = RefusalFlag::default();
        assert!(matches!(
            gate_statement(
                &ReaderDb::new(Arc::clone(&db)),
                "CREATE TEMP TABLE t AS SELECT 1",
                WritePolicy::Allow,
                &refused,
                &recorder
            )
            .await,
            Ok(Gate::Reject(m)) if m == TEMP_OBJECT_REFUSED
        ));
        assert!(refused.was_refused());
    }

    /// Without an embedding model the search tool answers from the term
    /// index alone (issue #58); with a reranker the fused order is handed
    /// to it and the step says so (issue #63).
    #[tokio::test]
    async fn search_tool_runs_keyword_only_without_a_model_and_applies_the_reranker() {
        struct Reverse;
        impl Reranker for Reverse {
            fn rank<'a>(
                &'a self,
                _q: &'a str,
                c: &'a [ChunkSearchResult],
            ) -> rerank::RankFuture<'a> {
                Box::pin(async move { Ok((0..c.len()).rev().collect()) })
            }
            fn name(&self) -> &'static str {
                "reverse"
            }
        }
        let db = shared_db();
        seed_hail_chunks(&db).await;
        let (sink, _rx) = super::super::events::channel();
        let recorder = TurnRecorder::new(sink);
        let tool = SearchDocumentsTool::<crate::llm::EmbedModel>::new(
            ReaderDb::new(Arc::clone(&db)),
            None,
            5,
            60,
            recorder.clone(),
        );
        let text = tool
            .call(
                &mut ToolContext::new(),
                SearchDocumentsArgs {
                    query: String::from("hail"),
                    top_k: None,
                    document_ids: Vec::new(),
                    entity: None,
                },
            )
            .await
            .unwrap_or_else(|e| fail_test(&e.to_string()));
        assert!(text.contains("Denver"), "{text}");
        let first_plain = text.find("hail again").unwrap_or(usize::MAX);
        let second_plain = text.find("Hail fell").unwrap_or(usize::MAX);
        assert!(
            first_plain < second_plain,
            "BM25 puts the denser chunk first: {text}"
        );

        let reranked = SearchDocumentsTool::<crate::llm::EmbedModel>::new(
            ReaderDb::new(Arc::clone(&db)),
            None,
            5,
            60,
            recorder.clone(),
        )
        .with_reranker(Arc::new(Reverse), 5);
        let text = reranked
            .call(
                &mut ToolContext::new(),
                SearchDocumentsArgs {
                    query: String::from("hail"),
                    top_k: None,
                    document_ids: Vec::new(),
                    entity: None,
                },
            )
            .await
            .unwrap_or_else(|e| fail_test(&e.to_string()));
        assert!(
            text.find("Hail fell").unwrap_or(usize::MAX)
                < text.find("hail again").unwrap_or(usize::MAX),
            "the reranker reversed the order: {text}"
        );
        let last = recorder.steps().last().map(|s| s.summary.clone());
        assert!(
            last.as_deref()
                .is_some_and(|s| s.contains("reranked by reverse")),
            "{last:?}"
        );
    }

    #[test]
    fn search_documents_offers_the_entity_argument_only_with_a_graph() {
        let (sink, _rx) = super::super::events::channel();
        let tool = SearchDocumentsTool::<crate::llm::EmbedModel>::new(
            ReaderDb::new(shared_db()),
            None,
            5,
            60,
            TurnRecorder::new(sink),
        );
        let has_entity = |tool: &SearchDocumentsTool<crate::llm::EmbedModel>| {
            tool.parameters().pointer("/properties/entity").is_some()
        };

        assert!(!has_entity(&tool));
        assert!(tool.parameters().pointer("/properties/query").is_some());
        assert!(!tool.description().contains("entity"));

        let tool = tool.with_graph(true);
        assert!(has_entity(&tool));
        assert!(tool.description().contains("Pass entity"));
    }

    #[tokio::test]
    async fn search_documents_top_k_is_capped_regardless_of_what_the_model_asks_for() {
        let db = shared_db();
        db.run(|guard| {
            guard.insert_document(
                &NewDocument::new("d", "storms.md", "text/markdown", 1)
                    .with_status(crate::storage::workspace::DocumentStatus::Ready),
            )?;
            for i in 0..(MAX_SEARCH_TOP_K * 2) {
                guard.insert_chunk(&crate::storage::workspace::NewChunk {
                    id: &format!("c{i}"),
                    document_id: "d",
                    chunk_index: i,
                    content: &format!("Hail fell in county {i}."),
                    heading: None,
                    page: None,
                    embedding: None,
                })?;
            }
            Ok(())
        })
        .await
        .unwrap_or_else(|e| fail_test(&e.to_string()));
        let (sink, _rx) = super::super::events::channel();
        let recorder = TurnRecorder::new(sink);
        let tool = SearchDocumentsTool::<crate::llm::EmbedModel>::new(
            ReaderDb::new(Arc::clone(&db)),
            None,
            5,
            60,
            recorder,
        );
        let text = tool
            .call(
                &mut ToolContext::new(),
                SearchDocumentsArgs {
                    query: String::from("hail"),
                    // Twice the cap and then some: a model is free to ask
                    // for this, and used to get every chunk it named back
                    // in full.
                    top_k: Some(1_000_000),
                    document_ids: Vec::new(),
                    entity: None,
                },
            )
            .await
            .unwrap_or_else(|e| fail_test(&e.to_string()));
        let returned = text.matches("(document_id: d, chunk ").count();
        assert_eq!(
            u32::try_from(returned).unwrap_or(u32::MAX),
            MAX_SEARCH_TOP_K,
            "{text}"
        );
    }

    #[tokio::test]
    async fn run_sql_caps_rows_and_reports_the_rest() {
        let (sink, _rx) = super::super::events::channel();
        let recorder = TurnRecorder::new(sink);
        let db = shared_db();
        let tool = RunSqlTool::new(
            Arc::clone(&db),
            ReaderDb::new(db),
            2,
            WritePolicy::Deny,
            RefusalFlag::default(),
            recorder.clone(),
        );
        let out = tool
            .call(
                &mut ToolContext::new(),
                RunSqlArgs {
                    query: String::from("SELECT range AS n FROM range(5)"),
                },
            )
            .await;
        let text = match out {
            Ok(text) => text,
            Err(e) => fail_test(&format!("expected tool text, got error: {e}")),
        };
        assert!(text.contains("3 more rows not shown"), "{text}");
        let numeric_rows = text
            .lines()
            .filter(|l| !l.trim().is_empty() && l.trim().chars().all(|c| c.is_ascii_digit()))
            .count();
        assert_eq!(numeric_rows, 2, "{text}");
        let last = recorder.steps().last().map(|s| s.summary.clone());
        assert_eq!(last.as_deref(), Some("5 rows"));
    }

    /// The one-query-per-group loop: the second statement, the first with
    /// another literal, comes back with the note and the turn budget; a
    /// different statement gets the budget alone.
    #[tokio::test]
    async fn run_sql_flags_a_statement_repeated_with_other_literals() {
        let (sink, _rx) = super::super::events::channel();
        let recorder = TurnRecorder::new(sink).with_turn_limit(15);
        let db = shared_db();
        let tool = RunSqlTool::new(
            Arc::clone(&db),
            ReaderDb::new(db),
            100,
            WritePolicy::Deny,
            RefusalFlag::default(),
            recorder,
        );
        let run = |query: &str| {
            let query = query.to_owned();
            let tool = &tool;
            async move {
                match tool
                    .call(&mut ToolContext::new(), RunSqlArgs { query })
                    .await
                {
                    Ok(text) => text,
                    Err(e) => fail_test(&format!("expected tool text, got error: {e}")),
                }
            }
        };
        let first = run("SELECT range AS n FROM range(5) WHERE n = 1").await;
        assert!(!first.contains("repeats an earlier one"), "{first}");
        assert!(
            first.ends_with("(tool call 1 of at most 15 this turn)"),
            "{first}"
        );
        let second = run("SELECT range AS n FROM range(5) WHERE n = 3").await;
        assert!(second.contains("repeats an earlier one"), "{second}");
        assert!(second.contains("WHERE n = 1"), "{second}");
        assert!(second.contains("arg_max"), "{second}");
        assert!(
            second.ends_with("(tool call 2 of at most 15 this turn)"),
            "{second}"
        );
        let third = run("SELECT count() FROM range(5)").await;
        assert!(!third.contains("repeats an earlier one"), "{third}");
        // The second statement again: still the first with another literal.
        let again = run("SELECT range AS n FROM range(5) WHERE n = 3").await;
        assert!(again.contains("WHERE n = 1"), "{again}");
    }

    /// The retry loop the prompt promises: a binder error, with `DuckDB`'s
    /// candidate bindings, comes back as tool text the model can act on
    /// rather than as a tool failure whose message rig withholds.
    #[tokio::test]
    async fn run_sql_hands_duckdb_errors_to_the_model_with_candidate_bindings() {
        let (sink, _rx) = super::super::events::channel();
        let recorder = TurnRecorder::new(sink);
        let db = shared_db();
        assert!(
            db.run(|guard| guard.execute_statement("CREATE TABLE trips(trip_distance DOUBLE)"))
                .await
                .is_ok()
        );
        let tool = RunSqlTool::new(
            Arc::clone(&db),
            ReaderDb::new(Arc::clone(&db)),
            100,
            WritePolicy::Deny,
            RefusalFlag::default(),
            recorder.clone(),
        );
        let out = tool
            .call(
                &mut ToolContext::new(),
                RunSqlArgs {
                    query: String::from("SELECT count(*) FROM trips WHERE distance > 10"),
                },
            )
            .await;
        let text = match out {
            Ok(text) => text,
            Err(e) => fail_test(&format!("expected tool text, got error: {e}")),
        };
        assert!(text.starts_with(SQL_ERROR_PREFIX), "{text}");
        assert!(text.contains("Candidate bindings"), "{text}");
        assert!(text.contains("trip_distance"), "{text}");
        let last = recorder.steps().last().map(|s| s.summary.clone());
        assert!(
            last.as_deref().is_some_and(|d| d.starts_with("error: ")),
            "{last:?}"
        );

        // A missing table names the tables that do exist.
        let describe = DescribeTableTool::new(ReaderDb::new(Arc::clone(&db)), recorder.clone());
        let out = describe
            .call(
                &mut ToolContext::new(),
                DescribeTableArgs {
                    table_name: String::from("trip"),
                },
            )
            .await;
        let text = match out {
            Ok(text) => text,
            Err(e) => fail_test(&format!("expected tool text, got error: {e}")),
        };
        assert!(text.starts_with(SQL_ERROR_PREFIX), "{text}");
        assert!(text.contains("Tables in this workspace: trips"), "{text}");

        // And a good statement still returns rows, with the count in the step.
        let out = tool
            .call(
                &mut ToolContext::new(),
                RunSqlArgs {
                    query: String::from("SELECT count(*) AS n FROM trips WHERE trip_distance > 10"),
                },
            )
            .await;
        assert!(out.is_ok_and(|t| t.contains('n') && !t.starts_with(SQL_ERROR_PREFIX)));
    }

    #[tokio::test]
    async fn gate_applies_allow_and_deny_to_writes() {
        let (sink, _rx) = super::super::events::channel();
        let recorder = TurnRecorder::new(sink);
        let db = shared_db();
        let refused = RefusalFlag::default();
        assert!(matches!(
            gate_statement(
                &ReaderDb::new(Arc::clone(&db)),
                "CREATE TABLE t(a INT)",
                WritePolicy::Allow,
                &refused,
                &recorder
            )
            .await,
            Ok(Gate::Run(StatementKind::Write))
        ));
        assert!(!refused.was_refused());
        assert!(matches!(
            gate_statement(&ReaderDb::new(Arc::clone(&db)), "DROP TABLE t", WritePolicy::Deny, &refused, &recorder).await,
            Ok(Gate::Reject(m)) if m == WRITE_REFUSED
        ));
        assert!(refused.was_refused());
    }

    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test asserts the event kind")]
    async fn gate_ask_waits_for_the_interface() {
        use super::super::events::AgentEvent;
        let (sink, mut rx) = super::super::events::channel();
        let recorder = TurnRecorder::new(sink);
        let db = shared_db();
        let refused = RefusalFlag::default();

        let gate = tokio::spawn({
            let db = Arc::clone(&db);
            let recorder = recorder.clone();
            let refused = refused.clone();
            async move {
                matches!(
                    gate_statement(
                        &ReaderDb::new(Arc::clone(&db)),
                        "DELETE FROM t",
                        WritePolicy::Ask,
                        &refused,
                        &recorder
                    )
                    .await,
                    Ok(Gate::Run(StatementKind::Write))
                )
            }
        });
        let req = match rx.recv().await {
            Some(AgentEvent::PermissionRequired(req)) => Some(req),
            _ => None,
        }
        .unwrap();
        assert_eq!(req.sql, "DELETE FROM t");
        req.allow();
        assert!(gate.await.is_ok_and(|ran| ran));
        assert!(!refused.was_refused());
    }

    /// The model sees the chart kinds in the tool's schema, not only in
    /// prose.
    #[test]
    fn the_chart_schema_lists_every_kind() {
        let schema = serde_json::to_value(schemars::schema_for!(CreateChartArgs))
            .map(|s| s.to_string())
            .unwrap_or_default();
        for kind in crate::analysis::chart::ChartKind::ALL {
            assert!(schema.contains(&format!("\"{kind}\"")), "{kind}: {schema}");
        }
    }
}

// ---------------------------------------------------------------------------
// search_graph and find_path
// ---------------------------------------------------------------------------

use crate::graph::{self, GraphResult};

/// The graph results a turn produced, kept for the response.
pub type GraphResults = Arc<Mutex<Vec<GraphResult>>>;

pub struct SearchGraphTool<M> {
    db: ReaderDb,
    embedding_model: Option<Embedder<M>>,
    options: graph::GraphOptions,
    exclude_provisional: bool,
    results: GraphResults,
    recorder: TurnRecorder,
}

impl<M> SearchGraphTool<M> {
    pub fn new(
        db: ReaderDb,
        embedding_model: Option<Embedder<M>>,
        options: graph::GraphOptions,
        exclude_provisional: bool,
        results: GraphResults,
        recorder: TurnRecorder,
    ) -> Self {
        Self {
            db,
            embedding_model,
            options,
            exclude_provisional,
            results,
            recorder,
        }
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct SearchGraphArgs {
    /// The entity to start from (its name as it appears in the data); omit
    /// to list every entity of `class`
    pub entity: Option<String>,
    /// Restrict the entry point, or the listing, to this ontology class id
    pub class: Option<String>,
    /// Follow only this relation id
    pub relation: Option<String>,
    /// How many hops out from the entity (default 2)
    pub hops: Option<u32>,
}

/// What a graph lookup produced: the result, and the labels to offer when
/// it found nothing.
struct GraphLookup {
    result: GraphResult,
    suggestions: Vec<String>,
}

/// One `search_graph` call's arguments, trimmed and defaulted, with the
/// entity's embedding already taken (embedding is async, the lookup is
/// not).
struct GraphQuery {
    entity: Option<String>,
    class: Option<String>,
    relation: Option<String>,
    hops: u32,
    embedding: Option<Vector>,
}

/// A `search_graph` lookup on the database thread: refuse ids the ontology
/// does not define, traverse from the entity or list the class, and gather
/// the labels to suggest when nothing matched.
fn lookup_graph(
    db: &WorkspaceDb,
    query: &GraphQuery,
    options: &graph::GraphOptions,
) -> crate::error::Result<GraphLookup> {
    let ontology = ontology_store::current(db)?;
    let class = query.class.as_deref();
    let relation = query.relation.as_deref();
    let embedding = query.embedding.as_deref();
    if let Some(class) = class {
        check_class(ontology.as_ref(), class)?;
    }
    if let Some(relation) = relation {
        check_relation(ontology.as_ref(), relation)?;
    }
    let result = if let Some(entity) = query.entity.as_deref() {
        let roots = graph::traverse::resolve_entry(db, entity, class, embedding)?;
        graph::traverse::neighborhood(db, &roots, query.hops, relation, options)?
    } else {
        graph::traverse::by_class(
            db,
            ontology.as_ref(),
            class.unwrap_or_default(),
            options.max_nodes,
            options,
        )?
    };
    let suggestions = match query.entity.as_deref() {
        Some(entity) if result.nodes.is_empty() => {
            graph::traverse::suggest_entities(db, entity, class, embedding)?
        }
        Some(_) | None => Vec::new(),
    };
    Ok(GraphLookup {
        result,
        suggestions,
    })
}

/// Ids past this many are counted rather than listed when an unknown id is
/// refused: an induced ontology can carry a class per table.
const LISTED_IDS: usize = 40;

/// `a, b, c ... and N more`, so a refusal names what exists without
/// pasting a whole ontology into the model's context.
fn listed(ids: &[&str]) -> String {
    let mut shown: Vec<String> = ids
        .iter()
        .take(LISTED_IDS)
        .map(|id| (*id).to_owned())
        .collect();
    let hidden = ids.len().saturating_sub(shown.len());
    if hidden > 0 {
        shown.push(format!("... and {hidden} more"));
    }
    shown.join(", ")
}

/// Refuse a class the ontology does not define, naming the ones it does.
/// `resolve_document_ids` sets the contract for `search_documents`: an id
/// that matches nothing is an error the model can correct, never an empty
/// result it reads as "the workspace has nothing on this".
fn check_class(ontology: Option<&Ontology>, class_id: &str) -> crate::error::Result<()> {
    let Some(ontology) = ontology else {
        return Err(Error::Analysis(String::from(
            "this workspace has no ontology, so it has no classes to search by",
        )));
    };
    if class_id == ontology::ROOT_CLASS || ontology.class(class_id).is_some() {
        return Ok(());
    }
    let mut ids: Vec<&str> = vec![ontology::ROOT_CLASS];
    ids.extend(ontology.classes.iter().map(|c| c.id.as_str()));
    Err(Error::Analysis(format!(
        "no class '{class_id}' in the ontology; the classes are: {}",
        listed(&ids)
    )))
}

/// Refuse a relation the ontology does not define, naming the ones it does.
fn check_relation(ontology: Option<&Ontology>, relation_id: &str) -> crate::error::Result<()> {
    let Some(ontology) = ontology else {
        return Err(Error::Analysis(String::from(
            "this workspace has no ontology, so it has no relations to follow",
        )));
    };
    if relation_id == ontology::MENTIONS_RELATION || ontology.relation(relation_id).is_some() {
        return Ok(());
    }
    let mut ids: Vec<&str> = vec![ontology::MENTIONS_RELATION];
    ids.extend(ontology.relations.iter().map(|r| r.id.as_str()));
    Err(Error::Analysis(format!(
        "no relation '{relation_id}' in the ontology; the relations are: {}",
        listed(&ids)
    )))
}

/// Embed `text` once per turn: cached on the turn recorder by exact input
/// text, so a turn that asks to embed the same query or entity label more
/// than once (retrieval and a rerank check, or the same entity resolved
/// by `search_documents`, `search_graph`, and `find_path` in one turn)
/// pays for the model call only the first time.
///
/// # Errors
///
/// Returns the model's error, unwrapped so a caller keeps its usual
/// message formatting.
async fn cached_embed<M: EmbeddingModel>(
    embedder: &Embedder<M>,
    recorder: &TurnRecorder,
    input: Input,
) -> crate::error::Result<Vector> {
    if let Some(cached) = recorder.cached_embedding(&input) {
        return Ok(cached);
    }
    let vector = embedder.embed_one(&input).await?;
    recorder.cache_embedding(input, vector.clone());
    Ok(vector)
}

/// Embed a label for fuzzy entry-point resolution, when a model exists.
/// The label's embedding for fuzzy entry: `None` without a model, an
/// error when the model fails (issue #62: a silent `None` degraded the
/// search to exact matches without saying so).
async fn label_embedding<M: EmbeddingModel>(
    embedder: Option<&Embedder<M>>,
    recorder: &TurnRecorder,
    label: &str,
) -> Result<Option<Vector>, ToolError> {
    let Some(embedder) = embedder else {
        return Ok(None);
    };
    let vector = cached_embed(embedder, recorder, Input::Similarity(label.to_owned()))
        .await
        .map_err(|e| ToolError::Analysis(format!("embedding failed: {e}")))?;
    Ok(Some(vector))
}

impl<M> Tool for SearchGraphTool<M>
where
    M: EmbeddingModel + Send + Sync,
{
    const NAME: &'static str = "search_graph";
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
        serde_json::to_value(schemars::schema_for!(SearchGraphArgs))
            .unwrap_or_else(|_| json!({"type": "object"}))
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let entity = args
            .entity
            .as_deref()
            .map(str::trim)
            .filter(|e| !e.is_empty());
        let class = args
            .class
            .as_deref()
            .map(str::trim)
            .filter(|c| !c.is_empty());
        let detail = match (entity, class) {
            (Some(e), Some(c)) => format!("{e} ({c})"),
            (Some(e), None) => e.to_owned(),
            (None, Some(c)) => format!("class {c}"),
            (None, None) => String::new(),
        };
        let step = self.recorder.start(Self::NAME, &detail);
        if entity.is_none() && class.is_none() {
            step.finish("error: nothing to search");
            return Err(ToolError::Analysis(String::from(
                "give an entity to start from, a class to list, or both",
            )));
        }
        let embedding = match entity {
            Some(e) => label_embedding(self.embedding_model.as_ref(), &self.recorder, e).await?,
            None => None,
        };
        let query = GraphQuery {
            entity: entity.map(str::to_owned),
            class: class.map(str::to_owned),
            relation: args
                .relation
                .as_deref()
                .map(str::trim)
                .filter(|r| !r.is_empty())
                .map(str::to_owned),
            hops: args.hops.unwrap_or(2).max(1),
            embedding,
        };
        let options = self.options;
        let lookup = self
            .db
            .with_db(move |db| lookup_graph(db, &query, &options))
            .await;
        let lookup = match lookup {
            Ok(lookup) => lookup,
            Err(e) => {
                let e = tool_error(e);
                step.finish(format!("error: {e}"));
                return Err(e);
            }
        };
        let had_matches = !lookup.result.nodes.is_empty();
        let result = if self.exclude_provisional {
            lookup.result.without_provisional()
        } else {
            lookup.result
        };
        let stripped = had_matches && result.nodes.is_empty();
        step.finish(if stripped {
            String::from("matches are provisional")
        } else {
            let of = match result.total_nodes.filter(|_| result.truncated) {
                Some(total) => format!(" of {total}"),
                None => String::new(),
            };
            format!(
                "{}{of} nodes, {} edges",
                result.nodes.len(),
                result.edges.len()
            )
        });
        let text = format_graph_result(
            &result,
            &self.recorder,
            &self.db,
            stripped,
            &lookup.suggestions,
        )
        .await?;
        if let Ok(mut results) = self.results.lock() {
            results.push(result);
        }
        Ok(text)
    }
}

pub struct FindPathTool<M> {
    db: ReaderDb,
    embedding_model: Option<Embedder<M>>,
    options: graph::GraphOptions,
    exclude_provisional: bool,
    results: GraphResults,
    recorder: TurnRecorder,
}

impl<M> FindPathTool<M> {
    pub fn new(
        db: ReaderDb,
        embedding_model: Option<Embedder<M>>,
        options: graph::GraphOptions,
        exclude_provisional: bool,
        results: GraphResults,
        recorder: TurnRecorder,
    ) -> Self {
        Self {
            db,
            embedding_model,
            options,
            exclude_provisional,
            results,
            recorder,
        }
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct FindPathArgs {
    /// The entity to start from
    pub from: String,
    /// The entity to reach
    pub to: String,
    /// Longest path to consider (default 4)
    pub max_hops: Option<u32>,
}

impl<M> Tool for FindPathTool<M>
where
    M: EmbeddingModel + Send + Sync,
{
    const NAME: &'static str = "find_path";
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
        serde_json::to_value(schemars::schema_for!(FindPathArgs))
            .unwrap_or_else(|_| json!({"type": "object"}))
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let from = args.from.trim();
        let to = args.to.trim();
        let step = self.recorder.start(Self::NAME, &format!("{from} -> {to}"));
        if from.is_empty() || to.is_empty() {
            step.finish("error: both ends are needed");
            return Err(ToolError::Analysis(String::from("give both entities")));
        }
        let from_embedding =
            label_embedding(self.embedding_model.as_ref(), &self.recorder, from).await?;
        let to_embedding =
            label_embedding(self.embedding_model.as_ref(), &self.recorder, to).await?;
        let max_hops = args.max_hops.unwrap_or(4).max(1);
        let from_owned = from.to_owned();
        let to_owned = to.to_owned();
        let options = self.options;
        let result = self
            .db
            .with_db(move |db| {
                let a = graph::traverse::resolve_entry(
                    db,
                    &from_owned,
                    None,
                    from_embedding.as_deref(),
                )?;
                let b =
                    graph::traverse::resolve_entry(db, &to_owned, None, to_embedding.as_deref())?;
                match (a.first(), b.first()) {
                    (Some(a), Some(b)) => graph::traverse::path(db, a, b, max_hops, &options),
                    (None, _) => Err(unresolved_entity(
                        db,
                        &from_owned,
                        from_embedding.as_deref(),
                    )),
                    (_, None) => Err(unresolved_entity(db, &to_owned, to_embedding.as_deref())),
                }
            })
            .await;
        let result = match result {
            Ok(result) => result,
            Err(e) => {
                let e = tool_error(e);
                step.finish(format!("error: {e}"));
                return Err(e);
            }
        };
        let had_matches = !result.nodes.is_empty();
        let result = if self.exclude_provisional {
            result.without_provisional()
        } else {
            result
        };
        let stripped = had_matches && result.nodes.is_empty();
        if result.nodes.is_empty() {
            if stripped {
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
        let text = format_graph_result(&result, &self.recorder, &self.db, false, &[]).await?;
        if let Ok(mut results) = self.results.lock() {
            results.push(result);
        }
        Ok(text)
    }
}

/// The error for a path endpoint that resolved to nothing, carrying the
/// nearest labels so the model can call again with a real one.
fn unresolved_entity(db: &WorkspaceDb, entity: &str, embedding: Option<&[f32]>) -> Error {
    let suggestions =
        graph::traverse::suggest_entities(db, entity, None, embedding).unwrap_or_default();
    if suggestions.is_empty() {
        return Error::Analysis(format!("no entity matches '{entity}'"));
    }
    Error::Analysis(format!(
        "no entity matches '{entity}'; the closest labels in the graph are: {}",
        suggestions.join(", ")
    ))
}

/// What the model is told when a lookup came back empty: that the graph
/// has nothing, that it has only unreviewed matches, or which labels to
/// try instead.
fn empty_graph_text(
    stripped_provisional: bool,
    suggestions: &[String],
) -> Result<String, ToolError> {
    if stripped_provisional {
        return Ok(String::from(
            "The graph has matches, but all of them are provisional: they were built from an \
             ontology version nobody has reviewed, and query mode does not answer from those. \
             Tell the user the graph has unreviewed matches and that `quack graph review` \
             accepts them.",
        ));
    }
    let mut out = String::from("No matching entities in the graph.");
    if suggestions.is_empty() {
        out.push_str(" Tell the user the graph has nothing on this.");
        return Ok(out);
    }
    write!(
        out,
        " The closest labels in the graph are: {}. Search again with one of them if that is what \
         the user meant; otherwise tell the user the graph has nothing on this.",
        suggestions.join(", ")
    )?;
    Ok(out)
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
/// or named as table rows. Bounded as a whole, not just per node.
async fn format_graph_result(
    result: &GraphResult,
    recorder: &TurnRecorder,
    db: &ReaderDb,
    stripped_provisional: bool,
    suggestions: &[String],
) -> Result<String, ToolError> {
    if result.nodes.is_empty() {
        return empty_graph_text(stripped_provisional, suggestions);
    }
    let tree = graph::traverse::render_tree(result);
    let mut out = if tree.chars().count() > MAX_GRAPH_TEXT_CHARS {
        trim_graph_text(&tree, MAX_GRAPH_TEXT_CHARS)
    } else {
        tree
    };
    // Bounded like the tree: a 200-node listing can carry a source per
    // node, and every one of them would be quoted in full.
    let all_chunk_ids: std::collections::BTreeSet<String> = result
        .provenance
        .iter()
        .filter_map(|p| p.chunk_id.clone())
        .collect();
    let hidden_chunks = all_chunk_ids.len().saturating_sub(MAX_GRAPH_SOURCES);
    let chunk_ids: Vec<String> = all_chunk_ids.into_iter().take(MAX_GRAPH_SOURCES).collect();
    let (chunks, ontology) = db
        .with_db(move |db| {
            let chunks = db.chunks_by_ids(&chunk_ids)?;
            Ok((chunks, ontology_store::current(db)?))
        })
        .await
        .map_err(tool_error)?;
    if !chunks.is_empty() {
        let first = recorder.citations().register(&chunks);
        writeln!(out, "\nSources (cite with the [n] marker):")?;
        for (i, chunk) in chunks.iter().enumerate() {
            let n = first.saturating_add(u32::try_from(i).unwrap_or(u32::MAX));
            let page = chunk.page.map_or(String::new(), |p| format!(", page {p}"));
            let excerpt: String = chunk.content.trim().chars().take(200).collect();
            writeln!(out, "[{n}] {}{page}: {excerpt}", chunk.filename)?;
        }
        if hidden_chunks > 0 {
            writeln!(out, "... and {hidden_chunks} more sources")?;
        }
    }
    let rows: std::collections::BTreeSet<String> = result
        .provenance
        .iter()
        .filter_map(|p| {
            let table = p.table_name.as_deref()?;
            Some(row_reference(
                ontology.as_ref(),
                table,
                p.row_key.as_deref(),
            ))
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

pub struct DescribeClassTool {
    db: ReaderDb,
    recorder: TurnRecorder,
}

impl DescribeClassTool {
    #[must_use]
    pub fn new(db: ReaderDb, recorder: TurnRecorder) -> Self {
        Self { db, recorder }
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct DescribeClassArgs {
    /// The ontology class id to describe
    pub class_id: String,
}

impl Tool for DescribeClassTool {
    const NAME: &'static str = "describe_class";
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
        serde_json::to_value(schemars::schema_for!(DescribeClassArgs))
            .unwrap_or_else(|_| json!({"type": "object"}))
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let class_id = args.class_id.trim().to_owned();
        let step = self.recorder.start(Self::NAME, &class_id);
        let text = self
            .db
            .with_db(move |db| {
                let ontology = ontology_store::current(db)?;
                check_class(ontology.as_ref(), &class_id)?;
                let Some(ontology) = ontology else {
                    return Err(Error::Analysis(String::from(
                        "this workspace has no ontology",
                    )));
                };
                let classes = class_and_subclasses(&ontology, &class_id);
                let census = graph::store::class_census(db, &classes, CLASS_SAMPLES)?;
                Ok(describe_class(&ontology, &class_id, &census))
            })
            .await;
        match text {
            Ok(text) => {
                step.finish(format!("{} lines", text.lines().count()));
                Ok(text)
            }
            Err(e) => {
                let e = tool_error(e);
                step.finish(format!("error: {e}"));
                Err(e)
            }
        }
    }
}

/// A class and every class under it, the set `search_graph(class)` lists
/// and `class_census` counts.
fn class_and_subclasses(ontology: &Ontology, class_id: &str) -> Vec<String> {
    let mut out = vec![class_id.to_owned()];
    for class in &ontology.classes {
        if class.id != class_id && ontology.is_subclass_of(&class.id, class_id) {
            out.push(class.id.clone());
        }
    }
    out
}

/// One class as the model sees it: the ontology's view of it plus what
/// the graph actually holds.
fn describe_class(ontology: &Ontology, class_id: &str, census: &(u64, Vec<String>)) -> String {
    let mut lines = vec![format!(
        "Class {class_id} (inherits: {})",
        ontology.ancestry(class_id).join(" -> ")
    )];
    if let Some(class) = ontology.class(class_id) {
        if let Some(description) = class.description.as_deref() {
            lines.push(format!("  {description}"));
        }
        if let Some(key) = class.key.as_deref() {
            lines.push(format!("Key property: {key}"));
        }
    }

    let subclasses: Vec<&str> = ontology
        .subclasses(class_id)
        .iter()
        .map(|c| c.id.as_str())
        .collect();
    lines.push(if subclasses.is_empty() {
        String::from("Subclasses: none")
    } else {
        format!(
            "Subclasses: {} (search_graph on this class covers them too)",
            listed(&subclasses)
        )
    });

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
    lines.push(if properties.is_empty() {
        String::from("Properties: none")
    } else {
        format!("Properties: {}", properties.join(", "))
    });

    let (from, to) = ontology.relations_of(class_id);
    lines.push(relation_line("Relations from it", &from, |relation| {
        format!("{} -> {}", relation.id, relation.range)
    }));
    lines.push(relation_line("Relations to it", &to, |relation| {
        format!("{} from {}", relation.id, relation.domain)
    }));

    if let Some(mapping) = ontology.mapping_for(class_id) {
        lines.push(format!(
            "Mapped table: {} (key column {}); run_sql can query it directly",
            mapping.table, mapping.key
        ));
    }

    let (total, samples) = census;
    lines.push(if *total == 0 {
        String::from("In the graph: no entities of this class")
    } else if samples.len() < usize::try_from(*total).unwrap_or(usize::MAX) {
        format!(
            "In the graph: {total} entities, for example {}",
            samples.join(", ")
        )
    } else {
        format!(
            "In the graph: {total} entities \u{2014} {}",
            samples.join(", ")
        )
    });

    let mut out = lines.join("\n");
    out.push('\n');
    out
}

/// One `Relations ...` line, or `none` when the class takes part in no
/// relation in that direction.
fn relation_line(
    label: &str,
    relations: &[&crate::ontology::Relation],
    render: impl Fn(&crate::ontology::Relation) -> String,
) -> String {
    if relations.is_empty() {
        return format!("{label}: none");
    }
    let rendered: Vec<String> = relations.iter().map(|r| render(r)).collect();
    format!("{label}: {}", rendered.join("; "))
}

/// A row's provenance as something the model can act on: the mapping
/// knows which column holds the key, so the row reads as a predicate
/// (`orders WHERE order_id = 'A-42'`) rather than as prose the model has
/// to guess a column name from. The mapping is what built the node in the
/// first place (design doc 6.3, table mapping).
fn row_reference(ontology: Option<&Ontology>, table: &str, row_key: Option<&str>) -> String {
    let Some(key) = row_key else {
        return format!("{table} (row key unknown)");
    };
    let column = ontology
        .and_then(|o| o.mappings.iter().find(|m| m.table == table))
        .map(|m| m.key.as_str());
    match column {
        Some(column) => format!(
            "{} WHERE {} = '{}'",
            quote_ident(table),
            quote_ident(column),
            key.replace('\'', "''")
        ),
        None => format!("{table} row {key}"),
    }
}
