use std::sync::{Arc, Mutex};

use rig::prelude::*;

use crate::config::AnalysisConfig;
use crate::error::{Error, Result};
use crate::storage::workspace::WorkspaceDb;

use super::text_to_sql;
use super::tools::{
    CreateChartTool, DescribeTableTool, ListDocumentsTool, ListTablesTool, RunSqlTool, SharedDb,
};
use super::vector_index::DuckDbVectorIndex;

const RAG_TOP_K: usize = 5;

#[derive(Debug)]
pub struct AgentResponse {
    pub content: String,
    pub chart_spec: Option<serde_json::Value>,
}

/// Run the rig agent with all analysis tools for a single user question.
///
/// Document context is automatically retrieved via rig's `.dynamic_context()`
/// RAG pattern, while structured data tools (SQL, tables, charts) remain as
/// explicit tool calls the LLM chooses to invoke.
///
/// # Errors
///
/// Returns an error if system prompt generation, agent building, or prompting fails.
pub async fn run_analysis<M>(
    db: WorkspaceDb,
    completion_model: impl rig::completion::CompletionModel + 'static,
    embedding_model: M,
    analysis_config: &AnalysisConfig,
    user_message: &str,
) -> Result<AgentResponse>
where
    M: rig::embeddings::EmbeddingModel + Send + Sync + 'static,
{
    let system_prompt = text_to_sql::build_system_prompt(&db)?;
    let shared_db: SharedDb = Arc::new(Mutex::new(db));
    let chart_spec: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));

    let vector_index = DuckDbVectorIndex::new(Arc::clone(&shared_db), embedding_model);

    let agent = completion_model
        .into_agent_builder()
        .preamble(&system_prompt)
        .dynamic_context(RAG_TOP_K, vector_index)
        .tool(RunSqlTool::new(
            Arc::clone(&shared_db),
            analysis_config.max_query_rows,
        ))
        .tool(DescribeTableTool::new(Arc::clone(&shared_db)))
        .tool(ListTablesTool::new(Arc::clone(&shared_db)))
        .tool(ListDocumentsTool::new(Arc::clone(&shared_db)))
        .tool(CreateChartTool::new(
            Arc::clone(&shared_db),
            Arc::clone(&chart_spec),
        ))
        .temperature(0.1)
        .default_max_turns(10)
        .build();

    let content = agent
        .prompt(user_message)
        .await
        .map_err(|e| Error::Analysis(e.to_string()))?;

    let chart = chart_spec
        .lock()
        .map_err(|e| Error::Analysis(format!("mutex poisoned: {e}")))?
        .take();

    Ok(AgentResponse {
        content,
        chart_spec: chart,
    })
}
