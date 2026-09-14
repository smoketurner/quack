use std::fmt::Write;
use std::sync::{Arc, Mutex};

use rig::embeddings::EmbeddingModel;
use rig::tool::{Tool, ToolContext};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::storage::workspace::{ChunkSearchResult, StatementKind, WorkspaceDb};

use super::chart;
use super::events::TurnRecorder;
use super::policy::{RefusalFlag, WritePolicy};
use super::text_to_sql;

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
                tracing::warn!(sql, "refused write statement from agent");
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
                        .map_err(|e| ToolError::Query(e.to_string()))
                };
                match results {
                    Ok(results) => {
                        step.finish(format!("{} rows", results.rows.len()));
                        text_to_sql::format_query_result(&results, self.max_query_rows)
                            .map_err(|e| ToolError::Query(e.to_string()))
                    }
                    Err(e) => {
                        step.finish(format!("error: {e}"));
                        Err(e)
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
    recorder: TurnRecorder,
}

impl<M> SearchDocumentsTool<M> {
    pub fn new(
        db: SharedDb,
        embedding_model: M,
        default_top_k: u32,
        recorder: TurnRecorder,
    ) -> Self {
        Self {
            db,
            embedding_model,
            default_top_k,
            recorder,
        }
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct SearchDocumentsArgs {
    /// Natural-language query to search the ingested documents for
    pub query: String,
    /// Number of chunks to return (default from config)
    pub top_k: Option<u32>,
    /// Restrict the search to these document ids (from `list_documents`)
    #[serde(default)]
    pub document_ids: Vec<String>,
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
            "Semantic search over the ingested documents. Returns the most relevant text chunks, \
             each numbered [n] with its source filename, for citing in the answer.",
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
        let step = self.recorder.start(Self::NAME, &args.query);
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

        let results = {
            let db = lock(&self.db)?;
            db.search_similar_chunks(&query_vec, top_k, &args.document_ids)
                .map_err(|e| ToolError::Query(e.to_string()))
        };
        match results {
            Ok(results) => {
                step.finish(format!("{} chunks", results.len()));
                format_search_results(&results).map_err(Into::into)
            }
            Err(e) => {
                step.finish(format!("error: {e}"));
                Err(e)
            }
        }
    }
}

/// Render search hits as numbered, citable chunks.
///
/// # Errors
///
/// Returns an error only if formatting into the output buffer fails.
pub fn format_search_results(results: &[ChunkSearchResult]) -> Result<String, std::fmt::Error> {
    if results.is_empty() {
        return Ok(String::from(
            "No relevant chunks found. Tell the user the documents do not appear to cover this.",
        ));
    }
    let mut out = String::new();
    for (i, chunk) in results.iter().enumerate() {
        let n = i.saturating_add(1);
        writeln!(
            out,
            "[{n}] {} (document_id: {}, chunk {}, distance {:.3})",
            chunk.filename, chunk.document_id, chunk.chunk_index, chunk.distance
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
            db.describe_table(&args.table_name)
                .map_err(|e| ToolError::Query(e.to_string()))
        };
        let desc = match desc {
            Ok(d) => d,
            Err(e) => {
                step.finish(format!("error: {e}"));
                return Err(e);
            }
        };

        let mut output = String::new();
        writeln!(output, "Table: {}", desc.table_name)?;
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
            writeln!(output, "- {table}")?;
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
            writeln!(
                output,
                "- {} (id: {}, status: {}, type: {})",
                doc.filename,
                doc.id,
                doc.status,
                doc.mime_type.as_deref().unwrap_or("unknown"),
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
    chart_spec: Arc<Mutex<Option<serde_json::Value>>>,
    recorder: TurnRecorder,
}

impl CreateChartTool {
    pub fn new(
        db: SharedDb,
        chart_spec: Arc<Mutex<Option<serde_json::Value>>>,
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
    /// SQL query to get chart data
    pub sql: String,
    /// Type of chart: bar, line, scatter, area, or pie
    pub chart_type: String,
    /// Column name for the x-axis (or category for pie charts)
    pub x: String,
    /// Column name for the y-axis (or value for pie charts)
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
            "Generate an ECharts chart specification from a SQL query result. Runs the SQL, then produces a chart spec.",
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
            chart::generate_chart_spec(&results, &args.chart_type, &args.x, &args.y, &args.title)
                .map_err(|e| ToolError::Analysis(e.to_string()))?;

        let spec_json = serde_json::to_string_pretty(&spec)
            .map_err(|e| ToolError::Analysis(format!("failed to serialize chart spec: {e}")))?;

        {
            let mut guard = self
                .chart_spec
                .lock()
                .map_err(|e| ToolError::Analysis(format!("mutex poisoned: {e}")))?;
            *guard = Some(spec);
        }

        step.finish(format!(
            "{} chart, {} rows",
            args.chart_type,
            results.rows.len()
        ));
        Ok(format!(
            "Chart generated successfully. ECharts spec:\n{spec_json}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(n: u32, filename: &str, content: &str) -> ChunkSearchResult {
        ChunkSearchResult {
            id: format!("c{n}"),
            content: content.to_owned(),
            document_id: String::from("doc-1"),
            chunk_index: n,
            filename: filename.to_owned(),
            distance: 0.125,
        }
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn format_search_results_numbers_hits_with_filename() {
        let out = format_search_results(&[
            hit(0, "policy.pdf", "  Flood is excluded.  "),
            hit(1, "faq.md", "Claims close in 30 days."),
        ])
        .unwrap();
        assert!(out.starts_with("[1] policy.pdf (document_id: doc-1, chunk 0, distance 0.125)\n"));
        assert!(out.contains("\nFlood is excluded.\n"));
        assert!(out.contains("[2] faq.md (document_id: doc-1, chunk 1, distance 0.125)\n"));
        assert!(out.contains("Claims close in 30 days."));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn format_search_results_empty_tells_model_to_say_so() {
        let out = format_search_results(&[]).unwrap();
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
