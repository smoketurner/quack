use crate::error::Result;
use crate::storage::sessions::ChatMode;
use crate::storage::workspace::WorkspaceDb;
use std::fmt::Write;

/// Build the system prompt: role and mode, tool guidance, table schemas,
/// documents, and the full text of pinned documents within `pinned_token_budget`.
///
/// # Errors
///
/// Returns an error if schema introspection fails.
pub fn build_system_prompt(
    db: &WorkspaceDb,
    mode: ChatMode,
    pinned_token_budget: u32,
) -> Result<String> {
    let mut prompt = String::from(
        "You are a data analysis assistant. You help users explore and analyze data stored in a DuckDB database.\n\n\
         You have access to tools that let you run SQL queries, search documents, describe tables, and create charts.\n\n\
         When answering analytical questions about structured data:\n\
         1. First use list_tables or describe_table to understand the available data\n\
         2. Write and execute SQL queries using run_sql\n\
         3. Explain the results in natural language\n\
         4. If the user asks for a visualization, use create_chart\n\n\
         When answering questions about document content:\n\
         1. Call search_documents with the user's question (rephrase and search again if the first results miss)\n\
         2. Answer only from the returned chunks; if none are relevant, say the documents do not cover it\n\
         3. Cite each claim with the chunk's [n] marker and name the source file\n\n\
         DuckDB SQL dialect notes:\n\
         - Use LIMIT for row limits\n\
         - Supports LIST, STRUCT, MAP types\n\
         - EXCLUDE clause on SELECT *\n\
         - String concatenation with || operator\n\
         - ILIKE for case-insensitive matching\n\
         - Use double quotes for identifiers with special characters\n\n",
    );

    match mode {
        ChatMode::Chat => prompt.push_str(
            "Mode: chat. You may draw on general knowledge, but whenever you use a retrieved chunk \
             or a query result, cite it.\n\n",
        ),
        ChatMode::Query => prompt.push_str(
            "Mode: query. Every factual claim must come from a retrieved chunk (cite its [n] marker) \
             or from a query you ran this turn. Do not answer from memory. If search and queries \
             find nothing relevant, say that the workspace does not cover the question and stop.\n\n",
        ),
    }

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

    append_pinned_documents(&mut prompt, db, pinned_token_budget)?;

    if tables.is_empty() && docs.is_empty() {
        writeln!(
            prompt,
            "No tables or documents have been ingested yet. Let the user know they can ingest files first."
        )?;
    }

    Ok(prompt)
}

/// Inject the full text of pinned documents, skipping any that would push
/// the total past `pinned_token_budget` (four characters per token).
fn append_pinned_documents(
    prompt: &mut String,
    db: &WorkspaceDb,
    pinned_token_budget: u32,
) -> Result<()> {
    let pinned = db.pinned_documents()?;
    if pinned.is_empty() {
        return Ok(());
    }
    let budget = usize::try_from(pinned_token_budget).unwrap_or(usize::MAX);
    let mut used = 0usize;
    writeln!(
        prompt,
        "Pinned documents (full text, always in effect; cite them by filename):"
    )?;
    for (doc, text) in &pinned {
        let cost = text.len().div_ceil(4);
        if used.saturating_add(cost) > budget {
            writeln!(
                prompt,
                "--- {} (omitted: pinned text exceeds the {pinned_token_budget}-token budget) ---",
                doc.filename
            )?;
            continue;
        }
        used = used.saturating_add(cost);
        writeln!(prompt, "--- {} ---", doc.filename)?;
        writeln!(prompt, "{text}")?;
        writeln!(prompt, "--- end {} ---", doc.filename)?;
    }
    writeln!(prompt)?;
    Ok(())
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
    use super::*;
    use crate::storage::workspace::NewChunk;

    fn db() -> WorkspaceDb {
        WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| open_failed(&e.to_string()))
    }

    #[expect(clippy::panic, reason = "test helper: in-memory DuckDB must open")]
    fn open_failed(msg: &str) -> WorkspaceDb {
        panic!("in-memory DuckDB failed to open: {msg}");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn prompt_states_mode_and_lists_tables_and_documents() {
        let db = db();
        db.execute_statement("CREATE TABLE claims(id INT, amount INT)")
            .unwrap();
        db.insert_document("d1", "policy.pdf", "application/pdf", 1, "ready")
            .unwrap();
        let chat = build_system_prompt(&db, ChatMode::Chat, 1000).unwrap();
        assert!(chat.contains("Mode: chat."));
        assert!(chat.contains("- claims"));
        assert!(chat.contains("- policy.pdf (status: ready"));
        assert!(!chat.contains("Pinned documents"));
        let query = build_system_prompt(&db, ChatMode::Query, 1000).unwrap();
        assert!(query.contains("Mode: query."));
        assert!(query.contains("Do not answer from memory"));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn pinned_documents_are_injected_within_budget() {
        let db = db();
        db.insert_document("d1", "rules.md", "text/markdown", 1, "ready")
            .unwrap();
        db.insert_document("d2", "big.md", "text/markdown", 1, "ready")
            .unwrap();
        let big = "x".repeat(400);
        let chunks = [
            ("d1", "first rule"),
            ("d1", "second rule"),
            ("d2", big.as_str()),
        ];
        for (i, (doc, text)) in chunks.iter().enumerate() {
            db.insert_chunk(&NewChunk {
                id: &format!("c{i}"),
                document_id: doc,
                chunk_index: u32::try_from(i).unwrap(),
                content: text,
                heading: None,
                page: None,
                embedding: None,
            })
            .unwrap();
        }
        db.set_document_pinned("d1", true).unwrap();
        db.set_document_pinned("d2", true).unwrap();
        // Budget of 20 tokens fits rules.md (~6 tokens) but not big.md (100).
        let prompt = build_system_prompt(&db, ChatMode::Chat, 20).unwrap();
        assert!(
            prompt.contains("--- rules.md ---\nfirst rule\nsecond rule\n--- end rules.md ---"),
            "{prompt}"
        );
        assert!(
            prompt.contains("--- big.md (omitted: pinned text exceeds the 20-token budget) ---")
        );
        assert!(db.set_document_pinned("missing", true).is_err());
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn empty_workspace_prompt_says_so() {
        let prompt = build_system_prompt(&db(), ChatMode::Chat, 100).unwrap();
        assert!(prompt.contains("No tables or documents have been ingested yet"));
    }
}
