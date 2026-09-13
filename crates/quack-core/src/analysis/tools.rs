use serde::Deserialize;
use serde_json::json;

use crate::llm::{FunctionDefinition, ToolDefinition};

fn tool(name: &str, description: &str, parameters: serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        tool_type: String::from("function"),
        function: FunctionDefinition {
            name: name.to_owned(),
            description: description.to_owned(),
            parameters,
        },
    }
}

pub(super) fn all_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        tool(
            "run_sql",
            "Execute a read-only SQL query against the workspace DuckDB database. Returns up to 100 rows as a formatted table.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute"
                    }
                },
                "required": ["query"]
            }),
        ),
        tool(
            "search_documents",
            "Vector similarity search over ingested document chunks. Returns the most relevant text chunks with their source document.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }),
        ),
        tool(
            "describe_table",
            "Returns column names, types, and 3 sample rows for a table in the workspace.",
            json!({
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to describe"
                    }
                },
                "required": ["table_name"]
            }),
        ),
        tool(
            "list_tables",
            "Returns all user-created tables in the workspace.",
            json!({
                "type": "object",
                "properties": {}
            }),
        ),
        tool(
            "list_documents",
            "Returns all ingested documents with their status.",
            json!({
                "type": "object",
                "properties": {}
            }),
        ),
        tool(
            "create_chart",
            "Generate an ECharts chart specification from a SQL query result. Runs the SQL, then produces a chart spec.",
            json!({
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL query to get chart data"
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "scatter", "area", "pie"],
                        "description": "Type of chart to generate"
                    },
                    "x": {
                        "type": "string",
                        "description": "Column name for the x-axis (or category for pie charts)"
                    },
                    "y": {
                        "type": "string",
                        "description": "Column name for the y-axis (or value for pie charts)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title"
                    }
                },
                "required": ["sql", "chart_type", "x", "y", "title"]
            }),
        ),
    ]
}

// --- Tool argument structs ---

#[derive(Deserialize)]
pub(super) struct RunSqlArgs {
    pub query: String,
}

#[derive(Deserialize)]
pub(super) struct SearchDocumentsArgs {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
}

fn default_top_k() -> u32 {
    5
}

#[derive(Deserialize)]
pub(super) struct DescribeTableArgs {
    pub table_name: String,
}

#[derive(Deserialize)]
pub(super) struct CreateChartArgs {
    pub sql: String,
    pub chart_type: String,
    pub x: String,
    pub y: String,
    pub title: String,
}
