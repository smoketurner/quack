use crate::analysis::policy::WritePolicy;
use crate::error::Result;
use crate::storage::sessions::ChatMode;
use crate::storage::workspace::WorkspaceDb;
use std::fmt::Write;

/// `DuckDB`'s Friendly SQL idioms, one line each, for the system prompt. Kept to
/// what the confined workspace connection (design doc 7.4) can run: nothing
/// here needs a file, an extension, or a `SET`. Revisit when the bundled
/// `DuckDB` version changes; the prompt states that version above this block.
const DIALECT_REFERENCE: &str = "\
- The tables listed below are all the data. File reads (FROM 'x.csv', read_csv, \
read_parquet, read_text), ATTACH, INSTALL, LOAD, and SET are blocked on this connection; \
never try them\n\
- FROM-first: FROM t WHERE x > 10 (implicit SELECT *); LIMIT n caps rows; count() needs no \
argument\n\
- GROUP BY ALL groups by every non-aggregate column; ORDER BY ALL orders by every column\n\
- SELECT * EXCLUDE (a, b) drops columns; SELECT * REPLACE (round(x) AS x) rewrites one in \
place\n\
- COLUMNS(*) or COLUMNS('regex') applies one expression across columns: SELECT min(COLUMNS(*)) \
FROM t\n\
- Column aliases are reusable in WHERE, GROUP BY, HAVING, and later select items: SELECT a + 1 \
AS b, b * 2 AS c\n\
- Conditional aggregation: count() FILTER (WHERE x > 10); top n per group: \
arg_max(name, score, 3) or max(score, 3) return lists\n\
- GROUPING SETS, CUBE, and ROLLUP for multi-level totals; PIVOT t ON col USING sum(v) and \
UNPIVOT reshape between wide and long\n\
- DESCRIBE t shows columns and types; SUMMARIZE t profiles every column\n\
- Joins: ASOF JOIN matches the nearest earlier key, POSITIONAL JOIN pairs rows by position, \
LATERAL correlates a subquery with the rows before it\n\
- Lists and structs: [1, 2, 3] is a list, l[1] is its first element and l[-1] its last, \
{'a': 1} is a struct read as s.a, [x * 2 FOR x IN l] is a comprehension, UNNEST(l) explodes\n\
- Strings: || concatenates, ILIKE is case-insensitive, 'text'.upper() chains a function with \
a dot, s[1:3] slices, regexp_matches(s, 'pat') tests a pattern\n\
- Dates and times: date_trunc('month', ts), strftime(ts, '%Y-%m-%d'), ts + INTERVAL 7 DAY, \
CAST('2024-01-31' AS DATE), current_date\n\
- Identifiers with spaces or capitals need double quotes; string literals use single quotes\n\
- Writes, when permitted: CREATE OR REPLACE TABLE t AS SELECT ..., INSERT INTO t BY NAME \
SELECT ..., INSERT OR REPLACE INTO t ...\n";

/// Everything that shapes the system prompt besides the workspace itself.
#[derive(Debug, Clone)]
pub struct PromptOptions {
    pub mode: ChatMode,
    /// What happens to mutating SQL this turn; the model is told so it
    /// attempts statements through the tool instead of refusing on its own.
    pub write_policy: WritePolicy,
    /// Budget for pinned document text (four characters per token).
    pub pinned_token_budget: u32,
    /// Global prefix plus workspace context, already joined, if any.
    pub context: Option<String>,
    /// Budget for `context` (four characters per token).
    pub context_max_tokens: u32,
}

/// Build the system prompt in the order the design fixes (section 7.2):
/// role and mode, tool guidance and dialect, tables, documents and pinned
/// text, the workspace context, and the permission rules.
///
/// # Errors
///
/// Returns an error if schema introspection fails.
pub fn build_system_prompt(db: &WorkspaceDb, options: &PromptOptions) -> Result<String> {
    let mut prompt = String::from(
        "You are a data analysis assistant working inside one workspace that holds tables, \
         documents, or both. Answer by using the tools: run SQL rather than estimating, search \
         the documents rather than recalling, state assumptions, and when a question is \
         ambiguous ask one clarifying question instead of guessing.\n\n",
    );

    match options.mode {
        ChatMode::Chat => prompt.push_str(
            "Mode: chat. You may draw on general knowledge, but whenever you use a retrieved chunk \
             or a query result, cite it.\n\n",
        ),
        ChatMode::Query => prompt.push_str(
            "Mode: query. Every factual claim must come from a retrieved chunk or from a query you \
             ran this turn. Do not answer from memory. If search and queries find nothing \
             relevant, say that the workspace does not cover the question and stop. Every \
             sentence that states something from a document MUST end with that chunk's [n] \
             marker, e.g. \"Flood damage is excluded [2].\" An answer about the documents with \
             no [n] markers is wrong.\n\n",
        ),
    }

    prompt.push_str(
        "When answering analytical questions about structured data:\n\
         1. First use list_tables or describe_table to understand the available data; run \
         SUMMARIZE <table> when you need min, max, null share, or distinct counts per column \
         before choosing a filter\n\
         2. Write and execute SQL queries using run_sql\n\
         3. If run_sql returns an error, read it: DuckDB names candidate columns for a \
         misspelled one and describe_table shows the real names. Fix the statement and run \
         it again; do not give up after one error and do not ask the user to correct SQL\n\
         4. Explain the results in natural language\n\
         5. If the user asks for a visualization, use create_chart\n\n\
         When answering questions about document content:\n\
         1. Call search_documents with the user's question (rephrase and search again if the first results miss)\n\
         2. Answer only from the returned chunks; if none are relevant, say the documents do not cover it\n\
         3. Cite each claim inline with the chunk's [n] marker, e.g. \"Flood is excluded [2].\"\n\
         4. Do not write a Sources or References section; one is appended for you from the markers\n\n",
    );

    let version = db.duckdb_version()?;
    writeln!(
        prompt,
        "DuckDB {version} SQL reference. This is DuckDB's dialect, not Postgres, MySQL, or \
         SQLite; prefer these idioms:"
    )?;
    prompt.push_str(DIALECT_REFERENCE);
    prompt.push('\n');

    let tables = append_tables(&mut prompt, db)?;

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

    append_pinned_documents(&mut prompt, db, options.pinned_token_budget)?;

    if let Some(ontology) = crate::ontology::store::current(db)? {
        prompt.push_str(&ontology.render_for_prompt());
        writeln!(prompt)?;
    }

    if tables.is_empty() && docs.is_empty() {
        writeln!(
            prompt,
            "No tables or documents have been ingested yet. Let the user know they can ingest files first."
        )?;
        writeln!(prompt)?;
    }

    append_context(&mut prompt, options)?;

    prompt.push_str(permissions_text(options.write_policy));

    Ok(prompt)
}

/// The tables block: every user table with its row count, columns, and three
/// sample rows. Returns the table names so the caller knows whether the
/// workspace is empty.
fn append_tables(prompt: &mut String, db: &WorkspaceDb) -> Result<Vec<String>> {
    let tables = db.list_tables()?;
    if !tables.is_empty() {
        writeln!(prompt, "Available tables:")?;
        for table in &tables {
            let Ok(desc) = db.describe_table(table) else {
                writeln!(prompt, "- {table}")?;
                continue;
            };
            writeln!(prompt, "- {table} ({} rows)", desc.row_count)?;
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
        writeln!(prompt)?;
    }
    Ok(tables)
}

/// The permissions paragraph for the write policy in force.
fn permissions_text(policy: WritePolicy) -> &'static str {
    match policy {
        WritePolicy::Allow => {
            "Permissions: SELECT queries always run. The user has permitted statements that \
             modify the workspace for this session, so when asked to change data, run the \
             statement with run_sql rather than asking for confirmation.\n"
        }
        WritePolicy::Ask => {
            "Permissions: SELECT queries always run. When you run a statement that modifies the \
             workspace, the user is asked to approve it before it executes, so when asked to \
             change data, run the statement with run_sql rather than asking for confirmation \
             yourself. If the tool reports it was refused, do not retry it; tell the user.\n"
        }
        WritePolicy::Deny => {
            "Permissions: SELECT queries always run. Statements that modify the workspace are \
             not permitted in this session; if the user asks for one, still attempt it once \
             with run_sql so the refusal is recorded, then tell the user it needs write \
             permission (--allow-write). Do not retry.\n"
        }
    }
}

/// The owner-written context, truncated to `context_max_tokens` with a note
/// so the model knows it is incomplete.
fn append_context(prompt: &mut String, options: &PromptOptions) -> Result<()> {
    let Some(context) = options.context.as_deref() else {
        return Ok(());
    };
    let budget_chars = usize::try_from(options.context_max_tokens)
        .unwrap_or(usize::MAX)
        .saturating_mul(4);
    writeln!(
        prompt,
        "Workspace context (written by the workspace owner; follow it over general knowledge):"
    )?;
    if context.len() <= budget_chars {
        writeln!(prompt, "{context}")?;
    } else {
        let cut: String = context.chars().take(budget_chars).collect();
        tracing::warn!(
            max_tokens = options.context_max_tokens,
            "workspace context exceeds the token budget and was truncated"
        );
        writeln!(prompt, "{cut}")?;
        writeln!(
            prompt,
            "[context truncated at {} tokens; ask the owner to shorten it]",
            options.context_max_tokens
        )?;
    }
    writeln!(prompt)?;
    Ok(())
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

    fn options(mode: ChatMode, pinned: u32) -> PromptOptions {
        PromptOptions {
            mode,
            write_policy: WritePolicy::Deny,
            pinned_token_budget: pinned,
            context: None,
            context_max_tokens: 4000,
        }
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn context_is_placed_after_documents_and_truncated_to_budget() {
        let db = db();
        db.insert_document("d1", "policy.pdf", "application/pdf", 1, "ready")
            .unwrap();
        let mut opts = options(ChatMode::Chat, 100);
        opts.context = Some(String::from("Amounts are in cents."));
        let prompt = build_system_prompt(&db, &opts).unwrap();
        let docs_at = prompt.find("Ingested documents:").unwrap();
        let ctx_at = prompt.find("Workspace context").unwrap();
        let perms_at = prompt.find("Permissions:").unwrap();
        assert!(docs_at < ctx_at && ctx_at < perms_at);
        assert!(prompt.contains("Amounts are in cents.\n"));
        assert!(!prompt.contains("truncated"));

        opts.context = Some("x".repeat(100));
        opts.context_max_tokens = 5;
        let prompt = build_system_prompt(&db, &opts).unwrap();
        assert!(prompt.contains(&"x".repeat(20)));
        assert!(!prompt.contains(&"x".repeat(21)));
        assert!(prompt.contains("[context truncated at 5 tokens"));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn prompt_states_mode_and_lists_tables_and_documents() {
        let db = db();
        db.execute_statement("CREATE TABLE claims(id INT, amount INT)")
            .unwrap();
        db.insert_document("d1", "policy.pdf", "application/pdf", 1, "ready")
            .unwrap();
        let chat = build_system_prompt(&db, &options(ChatMode::Chat, 1000)).unwrap();
        assert!(chat.contains("Mode: chat."));
        assert!(chat.contains("- claims (0 rows)"));
        db.execute_statement("INSERT INTO claims VALUES (1, 10), (2, 20)")
            .unwrap();
        let counted = build_system_prompt(&db, &options(ChatMode::Chat, 1000)).unwrap();
        assert!(counted.contains("- claims (2 rows)"), "{counted}");
        assert!(chat.contains("- policy.pdf (status: ready"));
        assert!(!chat.contains("Pinned documents"));
        assert!(
            chat.contains("needs write\n             permission")
                || chat.contains("needs write permission")
        );
        let mut allowed = options(ChatMode::Chat, 1000);
        allowed.write_policy = WritePolicy::Allow;
        let allowed = build_system_prompt(&db, &allowed).unwrap();
        assert!(allowed.contains("has permitted statements that"));
        let mut ask = options(ChatMode::Chat, 1000);
        ask.write_policy = WritePolicy::Ask;
        let ask = build_system_prompt(&db, &ask).unwrap();
        assert!(ask.contains("the user is asked to approve it"));
        let query = build_system_prompt(&db, &options(ChatMode::Query, 1000)).unwrap();
        assert!(query.contains("Mode: query."));
        assert!(query.contains("Do not answer from memory"));
        assert!(query.contains("MUST end with that chunk's [n]"));
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
        let prompt = build_system_prompt(&db, &options(ChatMode::Chat, 20)).unwrap();
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
    fn dialect_reference_is_pinned_to_the_bundled_duckdb_and_stays_in_the_sandbox() {
        let db = db();
        let prompt = build_system_prompt(&db, &options(ChatMode::Chat, 100)).unwrap();
        let version = db.duckdb_version().unwrap();
        assert!(version.starts_with('v'), "{version}");
        assert!(prompt.contains(&format!("DuckDB {version} SQL reference")));
        for idiom in [
            "GROUP BY ALL",
            "SUMMARIZE t profiles every column",
            "EXCLUDE (a, b)",
            "count() FILTER",
            "ASOF JOIN",
            "arg_max(name, score, 3)",
        ] {
            assert!(prompt.contains(idiom), "missing {idiom}");
        }
        // The retry rule and the sandbox note are what the error loop relies on.
        assert!(prompt.contains("Fix the statement and run it again"));
        assert!(prompt.contains("ATTACH, INSTALL, LOAD, and SET are blocked"));
        // Nothing in the reference needs an extension the static binary lacks.
        for banned in ["httpfs", "read_xlsx", "st_read", "SET VARIABLE", "INSTALL "] {
            assert!(
                !DIALECT_REFERENCE.contains(banned),
                "reference mentions {banned}"
            );
        }
        let tools_at = prompt.find("When answering analytical questions").unwrap();
        let dialect_at = prompt.find("SQL reference").unwrap();
        let perms_at = prompt.find("Permissions:").unwrap();
        assert!(tools_at < dialect_at && dialect_at < perms_at);
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn empty_workspace_prompt_says_so() {
        let prompt = build_system_prompt(&db(), &options(ChatMode::Chat, 100)).unwrap();
        assert!(prompt.contains("No tables or documents have been ingested yet"));
    }
}
