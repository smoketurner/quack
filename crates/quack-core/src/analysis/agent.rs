use std::fmt::Write;

use crate::config::AnalysisConfig;
use crate::error::{Error, Result};
use crate::llm::{CompletionRequest, FinishReason, LlmProvider, Message};
use crate::storage::workspace::WorkspaceDb;

use super::chart;
use super::rag;
use super::text_to_sql;
use super::tools;

const MAX_TOOL_ROUNDS: u32 = 10;

/// Result of an agent conversation turn.
#[derive(Debug)]
pub struct AgentResponse {
    pub content: String,
    pub chart_spec: Option<serde_json::Value>,
    pub messages: Vec<Message>,
}

/// Run the tool-calling agent loop for a single user question.
///
/// # Errors
///
/// Returns an error if the LLM call or tool execution fails.
pub async fn run_agent_loop<P: LlmProvider>(
    db: &WorkspaceDb,
    provider: &P,
    analysis_config: &AnalysisConfig,
    user_message: &str,
    conversation_history: &[Message],
) -> Result<AgentResponse> {
    let system_prompt = text_to_sql::build_system_prompt(db)?;
    let tool_defs = tools::all_tool_definitions();

    let mut messages: Vec<Message> = Vec::new();
    messages.push(Message::system(system_prompt));
    messages.extend_from_slice(conversation_history);
    messages.push(Message::user(user_message));

    let mut chart_spec: Option<serde_json::Value> = None;

    for round in 0..MAX_TOOL_ROUNDS {
        tracing::debug!(round, "agent loop iteration");

        let request = CompletionRequest {
            messages: messages.clone(),
            tools: Some(tool_defs.clone()),
            temperature: Some(0.1),
            max_tokens: None,
        };

        let response = provider.complete(request).await?;

        tracing::debug!(
            finish_reason = ?response.finish_reason,
            tool_call_count = response.tool_calls.len(),
            "LLM response received"
        );

        if response.tool_calls.is_empty() || response.finish_reason == FinishReason::Stop {
            messages.push(Message::assistant(&response.content));
            return Ok(AgentResponse {
                content: response.content,
                chart_spec,
                messages,
            });
        }

        messages.push(Message::assistant_with_tool_calls(
            response.content.clone(),
            response.tool_calls.clone(),
        ));

        for tool_call in &response.tool_calls {
            let result =
                execute_tool(db, provider, analysis_config, tool_call, &mut chart_spec).await;

            let result_text = match result {
                Ok(text) => text,
                Err(e) => format!("Error: {e}"),
            };

            tracing::debug!(
                tool = %tool_call.function.name,
                result_len = result_text.len(),
                "tool execution complete"
            );

            messages.push(Message::tool_result(&tool_call.id, result_text));
        }

        tracing::debug!(round, "completed tool round, continuing loop");
    }

    Err(Error::Analysis(format!(
        "agent loop exceeded {MAX_TOOL_ROUNDS} tool rounds"
    )))
}

async fn execute_tool<P: LlmProvider>(
    db: &WorkspaceDb,
    provider: &P,
    analysis_config: &AnalysisConfig,
    tool_call: &crate::llm::ToolCall,
    chart_spec: &mut Option<serde_json::Value>,
) -> Result<String> {
    match tool_call.function.name.as_str() {
        "run_sql" => {
            let args: tools::RunSqlArgs = serde_json::from_str(&tool_call.function.arguments)
                .map_err(|e| Error::Analysis(format!("invalid run_sql args: {e}")))?;
            execute_run_sql(db, analysis_config, &args.query)
        }
        "search_documents" => {
            let args: tools::SearchDocumentsArgs =
                serde_json::from_str(&tool_call.function.arguments)
                    .map_err(|e| Error::Analysis(format!("invalid search_documents args: {e}")))?;
            rag::search_documents(db, provider, &args.query, args.top_k).await
        }
        "describe_table" => {
            let args: tools::DescribeTableArgs =
                serde_json::from_str(&tool_call.function.arguments)
                    .map_err(|e| Error::Analysis(format!("invalid describe_table args: {e}")))?;
            execute_describe_table(db, &args.table_name)
        }
        "list_tables" => execute_list_tables(db),
        "list_documents" => execute_list_documents(db),
        "create_chart" => {
            let args: tools::CreateChartArgs = serde_json::from_str(&tool_call.function.arguments)
                .map_err(|e| Error::Analysis(format!("invalid create_chart args: {e}")))?;
            execute_create_chart(db, &args, chart_spec)
        }
        other => Err(Error::Analysis(format!("unknown tool: {other}"))),
    }
}

fn execute_run_sql(
    db: &WorkspaceDb,
    analysis_config: &AnalysisConfig,
    query: &str,
) -> Result<String> {
    let results = db.execute_query(query)?;
    text_to_sql::format_query_result(&results, analysis_config.max_query_rows)
}

fn execute_describe_table(db: &WorkspaceDb, table_name: &str) -> Result<String> {
    let desc = db.describe_table(table_name)?;

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

fn execute_list_tables(db: &WorkspaceDb) -> Result<String> {
    let tables = db.list_tables()?;
    if tables.is_empty() {
        return Ok(String::from("No tables found in this workspace."));
    }
    let mut output = String::from("Tables:\n");
    for table in &tables {
        writeln!(output, "- {table}")?;
    }
    Ok(output)
}

fn execute_list_documents(db: &WorkspaceDb) -> Result<String> {
    let docs = db.list_documents()?;
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

fn execute_create_chart(
    db: &WorkspaceDb,
    args: &tools::CreateChartArgs,
    chart_spec: &mut Option<serde_json::Value>,
) -> Result<String> {
    let results = db.execute_query(&args.sql)?;

    if results.rows.is_empty() {
        return Ok(String::from(
            "Query returned no rows — cannot generate chart.",
        ));
    }

    let spec =
        chart::generate_chart_spec(&results, &args.chart_type, &args.x, &args.y, &args.title)?;

    let spec_json = serde_json::to_string_pretty(&spec)
        .map_err(|e| Error::Analysis(format!("failed to serialize chart spec: {e}")))?;

    *chart_spec = Some(spec);

    Ok(format!(
        "Chart generated successfully. ECharts spec:\n{spec_json}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn execute_list_tables_empty_workspace() {
        let db = WorkspaceDb::open_in_memory(768).unwrap();
        let result = execute_list_tables(&db);
        assert!(result.is_ok_and(|s| s.contains("No tables")));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn execute_list_documents_empty_workspace() {
        let db = WorkspaceDb::open_in_memory(768).unwrap();
        let result = execute_list_documents(&db);
        assert!(result.is_ok_and(|s| s.contains("No documents")));
    }
}
