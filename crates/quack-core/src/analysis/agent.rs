use std::sync::{Arc, Mutex};

use rig::prelude::*;

use crate::config::{AnalysisConfig, RetrievalConfig};
use crate::error::{Error, Result};
use crate::storage::workspace::WorkspaceDb;

use super::policy::{RefusalFlag, WritePolicy};
use super::text_to_sql;
use super::tools::{
    CreateChartTool, DescribeTableTool, ListDocumentsTool, ListTablesTool, RunSqlTool,
    SearchDocumentsTool, SharedDb,
};
use super::vector_index::DuckDbVectorIndex;

#[derive(Debug)]
pub struct AgentResponse {
    pub content: String,
    pub chart_spec: Option<serde_json::Value>,
    /// At least one mutating statement was refused during this turn.
    pub write_refused: bool,
}

/// Run the rig agent with all analysis tools for a single user question.
///
/// Document retrieval is the `search_documents` tool, which the model calls
/// when a question is about document content. When
/// `retrieval.always_retrieve` is set, the top chunks are additionally
/// injected on every turn via rig's `dynamic_context`.
///
/// # Errors
///
/// Returns an error if system prompt generation, agent building, or prompting fails.
pub async fn run_analysis<M>(
    db: WorkspaceDb,
    completion_model: impl rig::completion::CompletionModel + 'static,
    embedding_model: M,
    analysis_config: &AnalysisConfig,
    retrieval_config: &RetrievalConfig,
    write_policy: WritePolicy,
    user_message: &str,
) -> Result<AgentResponse>
where
    M: rig::embeddings::EmbeddingModel + Clone + Send + Sync + 'static,
{
    let system_prompt = text_to_sql::build_system_prompt(&db)?;
    let shared_db: SharedDb = Arc::new(Mutex::new(db));
    let chart_spec: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
    let refused = RefusalFlag::default();

    let mut builder = completion_model
        .into_agent_builder()
        .preamble(&system_prompt)
        .tool(SearchDocumentsTool::new(
            Arc::clone(&shared_db),
            embedding_model.clone(),
            retrieval_config.top_k,
        ))
        .tool(RunSqlTool::new(
            Arc::clone(&shared_db),
            analysis_config.max_query_rows,
            write_policy,
            refused.clone(),
        ))
        .tool(DescribeTableTool::new(Arc::clone(&shared_db)))
        .tool(ListTablesTool::new(Arc::clone(&shared_db)))
        .tool(ListDocumentsTool::new(Arc::clone(&shared_db)))
        .tool(CreateChartTool::new(
            Arc::clone(&shared_db),
            Arc::clone(&chart_spec),
        ))
        .temperature(0.1)
        .default_max_turns(10);

    if retrieval_config.always_retrieve {
        let samples = usize::try_from(retrieval_config.top_k)
            .map_err(|e| Error::Analysis(format!("top_k overflow: {e}")))?;
        let vector_index = DuckDbVectorIndex::new(Arc::clone(&shared_db), embedding_model);
        builder = builder.dynamic_context(samples, vector_index);
    }

    let agent = builder.build();

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
        write_refused: refused.was_refused(),
    })
}
