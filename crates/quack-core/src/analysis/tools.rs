use std::fmt::Write;
use std::sync::{Arc, Mutex};

use rig::tool::{Tool, ToolContext};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::storage::workspace::WorkspaceDb;

use super::chart;
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

// ---------------------------------------------------------------------------
// run_sql
// ---------------------------------------------------------------------------

pub struct RunSqlTool {
    db: SharedDb,
    max_query_rows: u32,
}

impl RunSqlTool {
    pub fn new(db: SharedDb, max_query_rows: u32) -> Self {
        Self { db, max_query_rows }
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
            "Execute a read-only SQL query against the workspace DuckDB database. Returns up to 100 rows as a formatted table.",
        )
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(RunSqlArgs))
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
        let db = self
            .db
            .lock()
            .map_err(|e| ToolError::Query(format!("mutex poisoned: {e}")))?;
        let results = db
            .execute_query(&args.query)
            .map_err(|e| ToolError::Query(e.to_string()))?;
        text_to_sql::format_query_result(&results, self.max_query_rows)
            .map_err(|e| ToolError::Query(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// describe_table
// ---------------------------------------------------------------------------

pub struct DescribeTableTool {
    db: SharedDb,
}

impl DescribeTableTool {
    pub fn new(db: SharedDb) -> Self {
        Self { db }
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
        let desc = {
            let db = self
                .db
                .lock()
                .map_err(|e| ToolError::Query(format!("mutex poisoned: {e}")))?;
            db.describe_table(&args.table_name)
                .map_err(|e| ToolError::Query(e.to_string()))?
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

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// list_tables
// ---------------------------------------------------------------------------

pub struct ListTablesTool {
    db: SharedDb,
}

impl ListTablesTool {
    pub fn new(db: SharedDb) -> Self {
        Self { db }
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
        let tables = {
            let db = self
                .db
                .lock()
                .map_err(|e| ToolError::Query(format!("mutex poisoned: {e}")))?;
            db.list_tables()
                .map_err(|e| ToolError::Query(e.to_string()))?
        };
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
}

impl ListDocumentsTool {
    pub fn new(db: SharedDb) -> Self {
        Self { db }
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
        let docs = {
            let db = self
                .db
                .lock()
                .map_err(|e| ToolError::Query(format!("mutex poisoned: {e}")))?;
            db.list_documents()
                .map_err(|e| ToolError::Query(e.to_string()))?
        };
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
}

impl CreateChartTool {
    pub fn new(db: SharedDb, chart_spec: Arc<Mutex<Option<serde_json::Value>>>) -> Self {
        Self { db, chart_spec }
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

    #[expect(
        clippy::unused_async_trait_impl,
        reason = "Tool trait requires async fn"
    )]
    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let results = {
            let db = self
                .db
                .lock()
                .map_err(|e| ToolError::Query(format!("mutex poisoned: {e}")))?;
            db.execute_query(&args.sql)
                .map_err(|e| ToolError::Query(e.to_string()))?
        };

        if results.rows.is_empty() {
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

        Ok(format!(
            "Chart generated successfully. ECharts spec:\n{spec_json}"
        ))
    }
}
