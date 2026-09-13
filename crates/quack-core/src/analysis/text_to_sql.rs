use crate::error::Result;
use crate::storage::workspace::WorkspaceDb;
use std::fmt::Write;

/// Build a system prompt with all available table schemas and document context.
///
/// # Errors
///
/// Returns an error if schema introspection fails.
pub fn build_system_prompt(db: &WorkspaceDb) -> Result<String> {
    let mut prompt = String::from(
        "You are a data analysis assistant. You help users explore and analyze data stored in a DuckDB database.\n\n\
         You have access to tools that let you run SQL queries, search documents, describe tables, and create charts.\n\n\
         When answering analytical questions about structured data:\n\
         1. First use list_tables or describe_table to understand the available data\n\
         2. Write and execute SQL queries using run_sql\n\
         3. Explain the results in natural language\n\
         4. If the user asks for a visualization, use create_chart\n\n\
         When answering questions about document content:\n\
         1. Use search_documents to find relevant text chunks\n\
         2. Synthesize an answer from the retrieved chunks\n\
         3. Cite the source documents\n\n\
         DuckDB SQL dialect notes:\n\
         - Use LIMIT for row limits\n\
         - Supports LIST, STRUCT, MAP types\n\
         - EXCLUDE clause on SELECT *\n\
         - String concatenation with || operator\n\
         - ILIKE for case-insensitive matching\n\
         - Use double quotes for identifiers with special characters\n\n",
    );

    let tables = db.list_tables()?;
    if !tables.is_empty() {
        writeln!(prompt, "Available tables:")?;
        for table in &tables {
            writeln!(prompt, "- {table}")?;
            if let Ok(desc) = db.describe_table(table) {
                writeln!(prompt, "  Columns:")?;
                for col in &desc.columns {
                    writeln!(prompt, "    - {} ({})", col.name, col.column_type)?;
                }
                if !desc.sample_rows.rows.is_empty() {
                    writeln!(prompt, "  Sample data:")?;
                    let mut buf = Vec::new();
                    if desc.sample_rows.write_table(&mut buf).is_ok()
                        && let Ok(text) = String::from_utf8(buf)
                    {
                        for line in text.lines() {
                            writeln!(prompt, "    {line}")?;
                        }
                    }
                }
            }
        }
        writeln!(prompt)?;
    }

    let docs = db.list_documents()?;
    if !docs.is_empty() {
        writeln!(prompt, "Ingested documents:")?;
        for doc in &docs {
            writeln!(
                prompt,
                "- {} (status: {}, type: {})",
                doc.filename,
                doc.status,
                doc.mime_type.as_deref().unwrap_or("unknown"),
            )?;
        }
        writeln!(prompt)?;
    }

    if tables.is_empty() && docs.is_empty() {
        writeln!(
            prompt,
            "No tables or documents have been ingested yet. Let the user know they can ingest files first."
        )?;
    }

    Ok(prompt)
}

/// Format a query result as a text table string, capped at `max_rows`.
///
/// # Errors
///
/// Returns an error if formatting fails.
pub fn format_query_result(
    results: &crate::storage::workspace::QueryResults,
    max_rows: u32,
) -> Result<String> {
    let max = usize::try_from(max_rows)
        .map_err(|e| crate::error::Error::Analysis(format!("max_rows overflow: {e}")))?;

    let capped = if results.rows.len() > max {
        let mut capped = results.clone_capped(max_rows);
        let total = results.rows.len();
        capped.rows.push(vec![serde_json::Value::String(format!(
            "... ({} more rows not shown)",
            total.saturating_sub(max)
        ))]);
        capped
    } else {
        results.clone_capped(u32::MAX)
    };

    let mut buf = Vec::new();
    capped.write_table(&mut buf)?;
    String::from_utf8(buf).map_err(|e| crate::error::Error::Analysis(format!("UTF-8 error: {e}")))
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        // Text-to-SQL tests require a DuckDB workspace
    }
}
