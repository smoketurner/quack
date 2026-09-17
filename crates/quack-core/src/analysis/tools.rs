use std::fmt::Write;
use std::sync::{Arc, Mutex};

use rig::embeddings::EmbeddingModel;
use rig::tool::{Tool, ToolContext};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::storage::workspace::{ChunkSearchResult, StatementKind, WorkspaceDb};

use super::chart::{self, ChartSpec};
use super::events::TurnRecorder;
use super::policy::{RefusalFlag, WritePolicy};
use super::rerank::{self, Reranker};
use super::text_to_sql;
use crate::error::Error;
use crate::ontology::store as ontology_store;

pub type SharedDb = Arc<Mutex<WorkspaceDb>>;

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

fn lock(db: &SharedDb) -> Result<std::sync::MutexGuard<'_, WorkspaceDb>, ToolError> {
    db.lock()
        .map_err(|e| ToolError::Query(format!("mutex poisoned: {e}")))
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
    Run,
    /// Do not run; hand this text back to the model.
    Reject(String),
}

/// Classify a statement and apply the write policy. Takes and releases the
/// database lock itself so a permission prompt never holds it.
async fn gate_statement(
    db: &SharedDb,
    sql: &str,
    policy: WritePolicy,
    refused: &RefusalFlag,
    recorder: &TurnRecorder,
) -> Result<Gate, ToolError> {
    let kind = {
        let guard = lock(db)?;
        if guard
            .references_internal_table(sql)
            .map_err(|e| ToolError::Query(e.to_string()))?
        {
            return Ok(Gate::Reject(String::from(INTERNAL_TABLE_REFUSED)));
        }
        guard
            .classify_statement(sql)
            .map_err(|e| ToolError::Query(e.to_string()))?
    };
    match kind {
        StatementKind::Read => Ok(Gate::Run),
        StatementKind::Invalid(msg) => Ok(Gate::Reject(format!("SQL syntax error: {msg}"))),
        StatementKind::Write => {
            let allowed = match policy {
                WritePolicy::Allow => true,
                WritePolicy::Deny => false,
                WritePolicy::Ask => recorder.ask_permission(sql).await,
            };
            if allowed {
                Ok(Gate::Run)
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
    max_query_rows: u32,
    policy: WritePolicy,
    refused: RefusalFlag,
    recorder: TurnRecorder,
}

impl RunSqlTool {
    pub fn new(
        db: SharedDb,
        max_query_rows: u32,
        policy: WritePolicy,
        refused: RefusalFlag,
        recorder: TurnRecorder,
    ) -> Self {
        Self {
            db,
            max_query_rows,
            policy,
            refused,
            recorder,
        }
    }
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
        String::from(
            "Execute a SQL query against the workspace DuckDB database. SELECT queries always run; \
             statements that modify data need the user's write permission and may be refused. \
             Returns up to 100 rows as a formatted table.",
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
            &self.db,
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
            Gate::Run => {
                let results = {
                    let db = lock(&self.db)?;
                    db.execute_query(&args.query)
                };
                match results {
                    Ok(results) => {
                        step.finish(format!("{} rows", results.rows.len()));
                        text_to_sql::format_query_result(&results, self.max_query_rows)
                            .map_err(|e| ToolError::Query(e.to_string()))
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

pub struct SearchDocumentsTool<M> {
    db: SharedDb,
    embedding_model: M,
    default_top_k: u32,
    rrf_k: u32,
    reranker: Option<Arc<dyn Reranker>>,
    rerank_candidates: u32,
    recorder: TurnRecorder,
}

impl<M> SearchDocumentsTool<M> {
    pub fn new(
        db: SharedDb,
        embedding_model: M,
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
        }
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
}

/// Map what the model passed (an id, an id prefix, or a file name) to
/// document ids. Anything that matches nothing is an error naming the
/// documents that exist, so the model retries instead of getting an empty
/// result it reads as "the workspace has nothing on this".
fn resolve_document_ids(db: &WorkspaceDb, wanted: &[String]) -> Result<Vec<String>, ToolError> {
    if wanted.is_empty() {
        return Ok(Vec::new());
    }
    let documents = db
        .list_documents()
        .map_err(|e| ToolError::Query(e.to_string()))?;
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
            return Err(ToolError::Query(format!(
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
        String::from(
            "Search the ingested documents by meaning and by keyword. Returns the most relevant \
             text chunks, each numbered [n] with its source file, page, and heading, for citing \
             in the answer.",
        )
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(SearchDocumentsArgs))
            .unwrap_or_else(|_| json!({"type": "object"}))
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let detail = if args.document_ids.is_empty() {
            args.query.clone()
        } else {
            format!("{} (in {})", args.query, args.document_ids.join(", "))
        };
        let step = self.recorder.start(Self::NAME, &detail);
        let embedding = match self.embedding_model.embed_text(&args.query).await {
            Ok(e) => e,
            Err(e) => {
                step.finish(format!("error: {e}"));
                return Err(ToolError::Embedding(e.to_string()));
            }
        };

        #[expect(
            clippy::cast_possible_truncation,
            reason = "f64 -> f32 is acceptable for embedding vectors stored in DuckDB"
        )]
        let query_vec: Vec<f32> = embedding.vec.into_iter().map(|v| v as f32).collect();

        let top_k = args.top_k.unwrap_or(self.default_top_k).max(1);
        let fetch = if self.reranker.is_some() {
            top_k.max(self.rerank_candidates)
        } else {
            top_k
        };

        let results = {
            let db = lock(&self.db)?;
            resolve_document_ids(&db, &args.document_ids).and_then(|ids| {
                db.search_hybrid_chunks(&args.query, &query_vec, fetch, self.rrf_k, &ids)
                    .map_err(|e| ToolError::Query(e.to_string()))
            })
        };
        let results = match results {
            Ok(results) => results,
            Err(e) => {
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
        let first = self.recorder.citations().register(&results);
        format_search_results(&results, first).map_err(Into::into)
    }
}

/// Render search hits as numbered, citable chunks.
///
/// # Errors
///
/// Returns an error only if formatting into the output buffer fails.
pub fn format_search_results(
    results: &[ChunkSearchResult],
    first_marker: u32,
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
        writeln!(
            out,
            "[{n}] {}{page}{heading} (document_id: {}, chunk {}, score {:.4})",
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
    db: SharedDb,
    recorder: TurnRecorder,
}

impl DescribeTableTool {
    pub fn new(db: SharedDb, recorder: TurnRecorder) -> Self {
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

    #[expect(
        clippy::unused_async_trait_impl,
        reason = "Tool trait requires async fn"
    )]
    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let step = self.recorder.start(Self::NAME, &args.table_name);
        let desc = {
            let db = lock(&self.db)?;
            match db.describe_table(&args.table_name) {
                Ok(d) => d,
                Err(e) => {
                    step.finish(format!("error: {e}"));
                    let tables = db.list_tables().unwrap_or_default();
                    return Ok(format!(
                        "{SQL_ERROR_PREFIX}{e}\nTables in this workspace: {}",
                        if tables.is_empty() {
                            String::from("none")
                        } else {
                            tables.join(", ")
                        }
                    ));
                }
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
    db: SharedDb,
    recorder: TurnRecorder,
}

impl ListTablesTool {
    pub fn new(db: SharedDb, recorder: TurnRecorder) -> Self {
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

    #[expect(
        clippy::unused_async_trait_impl,
        reason = "Tool trait requires async fn"
    )]
    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let step = self.recorder.start(Self::NAME, "");
        let tables = {
            let db = lock(&self.db)?;
            db.list_tables()
                .map_err(|e| ToolError::Query(e.to_string()))?
        };
        step.finish(format!("{} tables", tables.len()));
        if tables.is_empty() {
            return Ok(String::from("No tables found in this workspace."));
        }
        let mut output = String::from("Tables:\n");
        for table in &tables {
            let count = {
                let db = lock(&self.db)?;
                db.count_rows(table).ok()
            };
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
    db: SharedDb,
    recorder: TurnRecorder,
}

impl ListDocumentsTool {
    pub fn new(db: SharedDb, recorder: TurnRecorder) -> Self {
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

    #[expect(
        clippy::unused_async_trait_impl,
        reason = "Tool trait requires async fn"
    )]
    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let step = self.recorder.start(Self::NAME, "");
        let docs = {
            let db = lock(&self.db)?;
            db.list_documents()
                .map_err(|e| ToolError::Query(e.to_string()))?
        };
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
    db: SharedDb,
    chart_spec: Arc<Mutex<Option<ChartSpec>>>,
    recorder: TurnRecorder,
}

impl CreateChartTool {
    pub fn new(
        db: SharedDb,
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
    /// Kind of chart: bar, line, scatter, or pie
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

        let results = {
            let db = lock(&self.db)?;
            db.execute_query(&args.sql)
                .map_err(|e| ToolError::Query(e.to_string()))
        };
        let results = match results {
            Ok(r) => r,
            Err(e) => {
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
    use super::*;
    use crate::storage::workspace::NewDocument;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail_test(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[test]
    fn document_ids_resolve_by_id_prefix_or_filename() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail_test(&e.to_string()));
        assert!(
            db.insert_document(
                &NewDocument::new("01a0-first", "policy.pdf", "application/pdf", 1)
                    .with_status("ready")
            )
            .is_ok()
        );
        assert!(
            db.insert_document(
                &NewDocument::new("01b0-second", "notes.md", "text/markdown", 1)
                    .with_status("ready")
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
    fn format_search_results_continues_numbering() {
        let out = format_search_results(&[hit(0, "a.md", "x")], 5).unwrap();
        assert!(out.contains("\n[5] a.md"), "{out}");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn format_search_results_empty_tells_model_to_say_so() {
        let out = format_search_results(&[], 4).unwrap();
        assert!(out.contains("No relevant chunks found"));
    }

    fn shared_db() -> SharedDb {
        Arc::new(Mutex::new(
            WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| unreachable_db(&e.to_string())),
        ))
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
            gate_statement(&db, "SELECT 1", WritePolicy::Deny, &refused, &recorder).await,
            Ok(Gate::Run)
        ));
        assert!(matches!(
            gate_statement(&db, "SELECT * FROM _quack_chunks", WritePolicy::Allow, &refused, &recorder).await,
            Ok(Gate::Reject(m)) if m == INTERNAL_TABLE_REFUSED
        ));
        assert!(matches!(
            gate_statement(&db, "SELEC 1", WritePolicy::Allow, &refused, &recorder).await,
            Ok(Gate::Reject(m)) if m.starts_with("SQL syntax error")
        ));
        assert!(!refused.was_refused());
    }

    /// The retry loop the prompt promises: a binder error, with `DuckDB`'s
    /// candidate bindings, comes back as tool text the model can act on
    /// rather than as a tool failure whose message rig withholds.
    #[tokio::test]
    async fn run_sql_hands_duckdb_errors_to_the_model_with_candidate_bindings() {
        let (sink, _rx) = super::super::events::channel();
        let recorder = TurnRecorder::new(sink);
        let db = shared_db();
        {
            let guard = lock(&db).unwrap_or_else(|e| fail_test(&e.to_string()));
            assert!(
                guard
                    .execute_statement("CREATE TABLE trips(trip_distance DOUBLE)")
                    .is_ok()
            );
        }
        let tool = RunSqlTool::new(
            Arc::clone(&db),
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
        let describe = DescribeTableTool::new(Arc::clone(&db), recorder.clone());
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
                &db,
                "CREATE TABLE t(a INT)",
                WritePolicy::Allow,
                &refused,
                &recorder
            )
            .await,
            Ok(Gate::Run)
        ));
        assert!(!refused.was_refused());
        assert!(matches!(
            gate_statement(&db, "DROP TABLE t", WritePolicy::Deny, &refused, &recorder).await,
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
                    gate_statement(&db, "DELETE FROM t", WritePolicy::Ask, &refused, &recorder)
                        .await,
                    Ok(Gate::Run)
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
}

// ---------------------------------------------------------------------------
// search_graph and find_path
// ---------------------------------------------------------------------------

use crate::graph::{self, GraphResult};

/// The graph results a turn produced, kept for the response.
pub type GraphResults = Arc<Mutex<Vec<GraphResult>>>;

pub struct SearchGraphTool<M> {
    db: SharedDb,
    embedding_model: Option<M>,
    options: graph::GraphOptions,
    exclude_provisional: bool,
    results: GraphResults,
    recorder: TurnRecorder,
}

impl<M> SearchGraphTool<M> {
    pub fn new(
        db: SharedDb,
        embedding_model: Option<M>,
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

/// Embed a label for fuzzy entry-point resolution, when a model exists.
async fn label_embedding<M: EmbeddingModel>(model: Option<&M>, label: &str) -> Option<Vec<f32>> {
    let model = model?;
    let embedding = model.embed_text(label).await.ok()?;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64 -> f32 is acceptable for embedding vectors"
    )]
    Some(embedding.vec.into_iter().map(|v| v as f32).collect())
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
            Some(e) => label_embedding(self.embedding_model.as_ref(), e).await,
            None => None,
        };
        let hops = args.hops.unwrap_or(2).max(1);
        let relation = args
            .relation
            .as_deref()
            .map(str::trim)
            .filter(|r| !r.is_empty());
        let result = {
            let db = lock(&self.db)?;
            let outcome = if let Some(e) = entity {
                graph::traverse::resolve_entry(&db, e, class, embedding.as_deref()).and_then(
                    |roots| {
                        graph::traverse::neighborhood(&db, &roots, hops, relation, &self.options)
                    },
                )
            } else {
                ontology_store::current(&db).and_then(|o| {
                    graph::traverse::by_class(
                        &db,
                        o.as_ref(),
                        class.unwrap_or_default(),
                        self.options.max_nodes,
                        &self.options,
                    )
                })
            };
            outcome.map_err(|e| ToolError::Query(e.to_string()))
        };
        let result = match result {
            Ok(result) => result,
            Err(e) => {
                step.finish(format!("error: {e}"));
                return Err(e);
            }
        };
        let result = if self.exclude_provisional {
            result.without_provisional()
        } else {
            result
        };
        step.finish(format!(
            "{} nodes, {} edges",
            result.nodes.len(),
            result.edges.len()
        ));
        let text = format_graph_result(&result, &self.recorder, &self.db)?;
        if let Ok(mut results) = self.results.lock() {
            results.push(result);
        }
        Ok(text)
    }
}

pub struct FindPathTool<M> {
    db: SharedDb,
    embedding_model: Option<M>,
    options: graph::GraphOptions,
    exclude_provisional: bool,
    results: GraphResults,
    recorder: TurnRecorder,
}

impl<M> FindPathTool<M> {
    pub fn new(
        db: SharedDb,
        embedding_model: Option<M>,
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
        let from_embedding = label_embedding(self.embedding_model.as_ref(), from).await;
        let to_embedding = label_embedding(self.embedding_model.as_ref(), to).await;
        let max_hops = args.max_hops.unwrap_or(4).max(1);
        let result = {
            let db = lock(&self.db)?;
            let outcome =
                graph::traverse::resolve_entry(&db, from, None, from_embedding.as_deref())
                    .and_then(|a| {
                        let b =
                            graph::traverse::resolve_entry(&db, to, None, to_embedding.as_deref())?;
                        Ok((a, b))
                    })
                    .and_then(|(a, b)| match (a.first(), b.first()) {
                        (Some(a), Some(b)) => {
                            graph::traverse::path(&db, a, b, max_hops, &self.options)
                        }
                        (None, _) => Err(Error::Analysis(format!("no entity matches '{from}'"))),
                        (_, None) => Err(Error::Analysis(format!("no entity matches '{to}'"))),
                    });
            outcome.map_err(|e| ToolError::Query(e.to_string()))
        };
        let result = match result {
            Ok(result) => result,
            Err(e) => {
                step.finish(format!("error: {e}"));
                return Err(e);
            }
        };
        let result = if self.exclude_provisional {
            result.without_provisional()
        } else {
            result
        };
        if result.nodes.is_empty() {
            step.finish("no path");
            return Ok(format!(
                "No path connects {from} and {to} within {max_hops} hops."
            ));
        }
        step.finish(format!("{} hops", result.edges.len()));
        let text = format_graph_result(&result, &self.recorder, &self.db)?;
        if let Ok(mut results) = self.results.lock() {
            results.push(result);
        }
        Ok(text)
    }
}

/// Render a graph result for the model: the tree, then the sources each
/// node and edge came from, registered as citable `[n]` markers (chunks)
/// or named as table rows.
fn format_graph_result(
    result: &GraphResult,
    recorder: &TurnRecorder,
    db: &SharedDb,
) -> Result<String, ToolError> {
    if result.nodes.is_empty() {
        return Ok(String::from(
            "No matching entities in the graph. Tell the user the graph has nothing on this.",
        ));
    }
    let mut out = graph::traverse::render_tree(result);
    let chunk_ids: Vec<String> = result
        .provenance
        .iter()
        .filter_map(|p| p.chunk_id.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let chunks = {
        let db = lock(db)?;
        db.chunks_by_ids(&chunk_ids)
            .map_err(|e| ToolError::Query(e.to_string()))?
    };
    if !chunks.is_empty() {
        let first = recorder.citations().register(&chunks);
        writeln!(out, "\nSources (cite with the [n] marker):")?;
        for (i, chunk) in chunks.iter().enumerate() {
            let n = first.saturating_add(u32::try_from(i).unwrap_or(u32::MAX));
            let page = chunk.page.map_or(String::new(), |p| format!(", page {p}"));
            let excerpt: String = chunk.content.trim().chars().take(200).collect();
            writeln!(out, "[{n}] {}{page}: {excerpt}", chunk.filename)?;
        }
    }
    let rows: std::collections::BTreeSet<String> = result
        .provenance
        .iter()
        .filter_map(|p| {
            p.table_name
                .as_deref()
                .map(|t| format!("{t} row {}", p.row_key.as_deref().unwrap_or("?")))
        })
        .collect();
    if !rows.is_empty() {
        writeln!(
            out,
            "\nFrom table rows: {}",
            rows.into_iter().collect::<Vec<_>>().join("; ")
        )?;
    }
    Ok(out)
}
