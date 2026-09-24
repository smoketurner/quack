use crate::analysis::policy::WritePolicy;
use crate::error::Result;
use crate::graph::store as graph_store;
use crate::ontology::store as ontology_store;
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
- Conditional aggregation: count() FILTER (WHERE x > 10)\n\
- One statement per question, never one per group: GROUP BY g with arg_max(label, measure) \
gives each group's top label, arg_max(label, measure, 3) its top three as a list, and \
QUALIFY row_number() OVER (PARTITION BY g ORDER BY measure DESC) <= 3 its top three rows; \
WHERE g IN ('a', 'b') covers a chosen set\n\
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
    /// For Ollama, the cap on the context window the turn requests
    /// (`[analysis].max_context_tokens`); `None` for providers that size
    /// their own.
    pub ollama_context_cap: Option<u32>,
}

/// The system prompt, assembled in the order the design fixes (section
/// 7.2): role and mode, tool guidance and dialect, tables, documents and
/// pinned text, the workspace context, and the permission rules.
#[derive(Debug, Default)]
pub struct SystemPrompt {
    text: String,
}

/// Classes, relations and mappings past this many are counted rather than
/// listed in the ontology block; `describe_class` has the rest. An
/// ontology induced from a wide workspace carries a class per table, which
/// would crowd out the guidance and the question (the bound the tables
/// block has had since issue #40).
const PROMPT_ONTOLOGY_ITEMS: usize = 30;

/// Tables past this many are listed by name and row count only.
const DETAILED_TABLES: usize = 25;
/// Columns past this many per table are counted, not listed.
const LISTED_COLUMNS: usize = 40;
/// Sample rows are shown only for tables up to this wide.
const SAMPLED_COLUMNS: usize = 20;
/// A sample cell longer than this is cut, with an ellipsis.
const SAMPLE_CELL_CHARS: usize = 60;

impl SystemPrompt {
    /// The prompt for `db` under `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if schema introspection fails.
    pub fn build(db: &WorkspaceDb, options: &PromptOptions) -> Result<String> {
        let mut prompt = Self::default();
        prompt.text.push_str(
            "You are a data analysis assistant working inside one workspace that holds tables, \
             documents, or both. Answer by using the tools: run SQL rather than estimating, \
             search the documents rather than recalling, state assumptions, and when a question \
             is ambiguous ask one clarifying question instead of guessing.\n\n",
        );

        match options.mode {
            ChatMode::Chat => prompt.text.push_str(
                "Mode: chat. You may draw on general knowledge, but whenever you use a retrieved \
                 chunk or a query result, cite it.\n\n",
            ),
            ChatMode::Query => prompt.text.push_str(
                "Mode: query. Every factual claim must come from a retrieved chunk or from a \
                 query you ran this turn. Do not answer from memory. If search and queries find \
                 nothing relevant, say that the workspace does not cover the question and stop. \
                 Every sentence that states something from a document MUST end with that chunk's \
                 [n] marker, e.g. \"Flood damage is excluded [2].\" An answer about the documents \
                 with no [n] markers is wrong.\n\n",
            ),
        }

        let ontology = ontology_store::current(db)?;
        let graph = graph_store::status(db)?;
        prompt.tool_guidance(graph.enabled(), ontology.is_some());

        let version = db.duckdb_version()?;
        writeln!(
            prompt.text,
            "DuckDB {version} SQL reference. This is DuckDB's dialect, not Postgres, MySQL, or \
             SQLite; prefer these idioms:"
        )?;
        prompt.text.push_str(DIALECT_REFERENCE);
        prompt.text.push('\n');

        let tables = prompt.tables(db)?;
        let documents = prompt.documents(db)?;
        prompt.pinned_documents(db, options.pinned_token_budget)?;

        if let Some(ontology) = ontology {
            prompt
                .text
                .push_str(&ontology.render_capped(PROMPT_ONTOLOGY_ITEMS));
            if graph.enabled() {
                writeln!(
                    prompt.text,
                    "Knowledge graph: {} nodes, {} edges typed by this ontology{}{}. search_graph \
                     and find_path read it; both return provenance to cite.",
                    graph.nodes,
                    graph.edges,
                    if graph.provisional() {
                        " (provisional: built from an unreviewed ontology; say so when you use it)"
                    } else {
                        ""
                    },
                    if graph.stale {
                        " (stale: the ontology changed since it was built)"
                    } else {
                        ""
                    }
                )?;
            }
            writeln!(prompt.text)?;
        }

        if tables.is_empty() && documents == 0 {
            writeln!(
                prompt.text,
                "No tables or documents have been ingested yet. Let the user know they can ingest files first."
            )?;
            writeln!(prompt.text)?;
        }

        prompt.context(options)?;

        prompt
            .text
            .push_str(options.write_policy.prompt_paragraph());

        Ok(prompt.text)
    }

    /// The numbered procedures, one per substrate. The table, SQL, chart
    /// and document tools are always registered; the graph block appears
    /// only when the graph tools do and the `describe_class` line only when
    /// an ontology exists (design doc 7.2), since guidance for a tool the
    /// model cannot call is worse than none.
    fn tool_guidance(&mut self, graph_enabled: bool, ontology_present: bool) {
        self.text.push_str(
            "When answering analytical questions about structured data:\n\
             1. First use list_tables or describe_table to understand the available data; run \
             SUMMARIZE <table> when you need min, max, null share, or distinct counts per column \
             before choosing a filter\n\
             2. Write and execute SQL queries using run_sql\n\
             3. If run_sql returns an error, read it: DuckDB names candidate columns for a \
             misspelled one and describe_table shows the real names. Fix the statement and run \
             it again; do not give up after one error and do not ask the user to correct SQL\n\
             4. A result that ends with \"more rows not shown\" was cut at the row limit and is \
             not the whole answer: aggregate further, filter with WHERE, or add ORDER BY and \
             LIMIT, in one statement. Never re-run a statement once per group; GROUP BY, \
             IN (...), or a window covers every group at once\n\
             5. Every run_sql result ends with how many tool calls the turn has left; plan \
             the remaining statements and answer before they run out\n\
             6. Explain the results in natural language\n\
             7. If the user asks for a visualization, use create_chart\n\n\
             When answering questions about document content:\n\
             1. Call search_documents with the user's question (rephrase and search again if the first results miss)\n\
             2. Answer only from the returned chunks; if none are relevant, say the documents do not cover it\n\
             3. Cite each claim inline with the chunk's [n] marker, e.g. \"Flood is excluded [2].\"\n\
             4. Do not write a Sources or References section; one is appended for you from the markers\n\n",
        );

        if ontology_present {
            self.text.push_str(
                "The ontology block below is capped. describe_class gives one class in full: what \
                 it inherits, its subclasses, its typed properties with their enum values, the \
                 relations it takes part in, the table it is mapped to, and how many entities of \
                 it the graph holds. Use it to get an exact id before searching, and for the count \
                 of a class — a class listing stops at the node limit, describe_class does not.\n\n",
            );
        }

        if graph_enabled {
            self.text.push_str(
                "When answering questions about how entities relate:\n\
                 1. Call search_graph with an entity's name for what it connects to, or with an \
                 ontology class id to list the entities of that class\n\
                 2. Call find_path when the question is how two named entities connect\n\
                 3. Use the class and relation ids from the ontology below, or describe_class to \
                 check one; a wrong id comes back as an error naming the real ones, and a name that \
                 matches no entity comes back with the closest labels, so call again rather than \
                 giving up\n\
                 4. Cite the [n] markers the results register, the same way you cite search_documents. \
                 A result that names table rows can be read with run_sql: it gives the predicate\n\
                 5. A result that says it was cut off at the node limit is not the whole answer; \
                 narrow the class or count with describe_class instead of counting the lines\n\
                 6. If the graph has nothing, search the documents before telling the user the \
                 workspace does not cover the question\n\n",
            );
        }
    }

    /// The document inventory block; returns how many documents it listed
    /// so the caller can tell an empty workspace from a full one.
    fn documents(&mut self, db: &WorkspaceDb) -> Result<usize> {
        let docs = db.list_documents()?;
        if docs.is_empty() {
            return Ok(0);
        }
        writeln!(self.text, "Ingested documents:")?;
        for doc in &docs {
            let title = doc
                .title
                .as_deref()
                .map_or(String::new(), |t| format!(" \"{t}\""));
            writeln!(
                self.text,
                "- {}{title} (status: {}, type: {})",
                doc.filename,
                doc.status,
                doc.mime_type.as_deref().unwrap_or("unknown"),
            )?;
        }
        writeln!(self.text)?;
        Ok(docs.len())
    }

    /// The tables block: every user table with its row count, columns, and
    /// three sample rows, bounded so a wide or narrative table cannot crowd
    /// the tool guidance and the question out of a small context window
    /// (issue #40): the model has `describe_table` for the rest. Returns
    /// the table names so the caller knows whether the workspace is empty.
    fn tables(&mut self, db: &WorkspaceDb) -> Result<Vec<String>> {
        let tables = db.list_tables()?;
        if tables.is_empty() {
            return Ok(tables);
        }
        writeln!(self.text, "Available tables:")?;
        for (index, table) in tables.iter().enumerate() {
            // Tables past the detail cap only ever print their row count,
            // so only ask for that: `describe_table` also runs `DESCRIBE`
            // and a sample-row `SELECT`, whose output would be thrown away
            // below. A workspace with far more tables than the cap (a
            // per-table induced ontology, say) otherwise pays for a full
            // describe and sample of every excess table on every turn for
            // nothing.
            if index >= DETAILED_TABLES {
                let Ok(row_count) = db.count_rows(table) else {
                    writeln!(self.text, "- {table}")?;
                    continue;
                };
                writeln!(self.text, "- {table} ({row_count} rows)")?;
                continue;
            }
            let Ok(desc) = db.describe_table(table) else {
                writeln!(self.text, "- {table}")?;
                continue;
            };
            writeln!(self.text, "- {table} ({} rows)", desc.row_count)?;
            writeln!(self.text, "  Columns:")?;
            for col in desc.columns.iter().take(LISTED_COLUMNS) {
                writeln!(self.text, "    - {} ({})", col.name, col.column_type)?;
            }
            if desc.columns.len() > LISTED_COLUMNS {
                writeln!(
                    self.text,
                    "    ... and {} more columns; describe_table lists them all",
                    desc.columns.len().saturating_sub(LISTED_COLUMNS)
                )?;
            }
            if desc.sample_rows.rows.is_empty() {
                continue;
            }
            if desc.columns.len() > SAMPLED_COLUMNS {
                writeln!(
                    self.text,
                    "  Sample rows omitted ({} columns); describe_table shows them",
                    desc.columns.len()
                )?;
                continue;
            }
            writeln!(self.text, "  Sample data:")?;
            let mut buf = Vec::new();
            if desc
                .sample_rows
                .with_cells_cut(SAMPLE_CELL_CHARS)
                .write_table(&mut buf)
                .is_ok()
                && let Ok(text) = String::from_utf8(buf)
            {
                for line in text.lines() {
                    writeln!(self.text, "    {line}")?;
                }
            }
        }
        if tables.len() > DETAILED_TABLES {
            writeln!(
                self.text,
                "Only the first {DETAILED_TABLES} tables are described here; use describe_table for the others."
            )?;
        }
        writeln!(self.text)?;
        Ok(tables)
    }

    /// The owner-written context, truncated to `context_max_tokens` with a
    /// note so the model knows it is incomplete.
    fn context(&mut self, options: &PromptOptions) -> Result<()> {
        let Some(context) = options.context.as_deref() else {
            return Ok(());
        };
        let budget_chars = usize::try_from(options.context_max_tokens)
            .unwrap_or(usize::MAX)
            .saturating_mul(4);
        writeln!(
            self.text,
            "Workspace context (written by the workspace owner; follow it over general knowledge):"
        )?;
        if context.len() <= budget_chars {
            writeln!(self.text, "{context}")?;
        } else {
            let cut: String = context.chars().take(budget_chars).collect();
            tracing::warn!(
                max_tokens = options.context_max_tokens,
                "workspace context exceeds the token budget and was truncated"
            );
            writeln!(self.text, "{cut}")?;
            writeln!(
                self.text,
                "[context truncated at {} tokens; ask the owner to shorten it]",
                options.context_max_tokens
            )?;
        }
        writeln!(self.text)?;
        Ok(())
    }

    /// The full text of pinned documents, skipping any that would push the
    /// total past `pinned_token_budget` (four characters per token).
    fn pinned_documents(&mut self, db: &WorkspaceDb, pinned_token_budget: u32) -> Result<()> {
        let pinned = db.pinned_documents()?;
        if pinned.is_empty() {
            return Ok(());
        }
        let budget = usize::try_from(pinned_token_budget).unwrap_or(usize::MAX);
        let mut used = 0usize;
        writeln!(
            self.text,
            "Pinned documents (full text, always in effect; cite them by filename):"
        )?;
        for (doc, text) in &pinned {
            let cost = text.len().div_ceil(4);
            if used.saturating_add(cost) > budget {
                writeln!(
                    self.text,
                    "--- {} (omitted: pinned text exceeds the {pinned_token_budget}-token budget) ---",
                    doc.filename
                )?;
                continue;
            }
            used = used.saturating_add(cost);
            writeln!(self.text, "--- {} ---", doc.filename)?;
            writeln!(self.text, "{text}")?;
            writeln!(self.text, "--- end {} ---", doc.filename)?;
        }
        writeln!(self.text)?;
        Ok(())
    }
}

/// The `num_ctx` to ask Ollama for: the prompt's estimated tokens plus
/// room for tool results and the answer, rounded up to 8,192, between
/// 8,192 and `cap`. Ollama's default of 4,096 truncates the front of
/// most workspace prompts, which loses the tool guidance and the question.
///
/// `num_ctx` is a load option: asking Ollama for a different value than
/// the one the model is already loaded with forces a full model reload,
/// which measured 4-5 seconds for `gpt-oss:20b` on this machine (`ollama
/// serve`, repeated `/api/generate` calls that only changed `num_ctx`) —
/// against single-digit milliseconds for a request that keeps the same
/// value. A session's history only grows turn over turn until the
/// history trim caps it, so the requested size is non-decreasing within
/// a session; the step below is deliberately coarse (four tiers instead
/// of one every 2,048 tokens) so a growing conversation crosses it, and
/// pays that reload, at most three times instead of up to twelve.
#[must_use]
pub fn ollama_context_size(prompt_chars: usize, cap: u32) -> u32 {
    const HEADROOM: u32 = 8_192;
    const FLOOR: u32 = 8_192;
    const STEP: u32 = 8_192;
    let prompt_tokens = u32::try_from(prompt_chars.div_ceil(4)).unwrap_or(u32::MAX);
    let needed = prompt_tokens.saturating_add(HEADROOM);
    let rounded = needed.div_ceil(STEP).saturating_mul(STEP).max(FLOOR);
    rounded.min(cap.max(FLOOR))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Properties;
    use crate::graph::store::NewNode;
    use crate::ontology::Ontology;
    use crate::ontology::store::Revision;
    use crate::storage::workspace::{DocumentStatus, NewChunk, NewDocument};

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
            ollama_context_cap: None,
        }
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn the_graph_procedure_appears_only_once_the_graph_has_nodes() {
        const PROCEDURE: &str = "When answering questions about how entities relate";
        let db = db();
        ontology_store::save(
            &db,
            &Ontology::builtin_default(),
            Revision::reviewed(Some("tester"), None),
        )
        .unwrap();

        // An ontology alone registers no graph tools, so it gets no procedure.
        let without = SystemPrompt::build(&db, &options(ChatMode::Chat, 0)).unwrap();
        assert!(!without.contains(PROCEDURE), "{without}");
        assert!(without.contains("Ontology (version"), "{without}");

        graph_store::upsert_node(
            &db,
            &NewNode {
                label: String::from("Acme"),
                class_id: String::from("organization"),
                properties: Properties::default(),
                provisional: false,
            },
        )
        .unwrap();
        let with = SystemPrompt::build(&db, &options(ChatMode::Chat, 0)).unwrap();
        assert!(with.contains(PROCEDURE), "{with}");
        assert!(
            with.contains("search_graph") && with.contains("find_path"),
            "{with}"
        );
        // The guidance comes before the ontology it refers to (design doc 7.2).
        assert!(
            with.find(PROCEDURE) < with.find("Ontology (version"),
            "{with}"
        );
        assert!(with.contains("Knowledge graph: 1 nodes, 0 edges"), "{with}");
    }

    /// The stable part of the prompt (role, tool guidance, dialect, table
    /// and document schema, ontology) must come out byte-identical across
    /// two calls with nothing in the workspace changed, and the whole
    /// prompt otherwise (the workspace context, which can differ by
    /// caller) must too. Ollama keeps a KV cache for the common prefix of
    /// consecutive requests to the same loaded model; a stable part that
    /// changed for no reason (nondeterministic ordering, a timestamp, a
    /// session id) would silently defeat that cache on every turn.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn the_prompt_is_byte_identical_across_repeated_calls_with_no_workspace_change() {
        let db = db();
        db.execute_statement("CREATE TABLE claims(id INT, amount INT, status VARCHAR)")
            .unwrap();
        db.execute_statement("INSERT INTO claims VALUES (1, 100, 'paid'), (2, 200, 'denied')")
            .unwrap();
        db.insert_document(
            &NewDocument::new("d1", "policy.pdf", "application/pdf", 1)
                .with_status(DocumentStatus::Ready),
        )
        .unwrap();
        ontology_store::save(
            &db,
            &Ontology::builtin_default(),
            Revision::reviewed(Some("tester"), None),
        )
        .unwrap();
        graph_store::upsert_node(
            &db,
            &NewNode {
                label: String::from("Acme"),
                class_id: String::from("organization"),
                properties: Properties::default(),
                provisional: false,
            },
        )
        .unwrap();
        let mut opts = options(ChatMode::Chat, 1000);
        opts.context = Some(String::from("Amounts are in cents."));

        let first = SystemPrompt::build(&db, &opts).unwrap();
        let second = SystemPrompt::build(&db, &opts).unwrap();
        assert_eq!(first, second);

        // The volatile, caller-supplied part (the workspace context) comes
        // after every part the workspace itself determines.
        let role_at = first.find("You are a data analysis assistant").unwrap();
        let guidance_at = first.find("When answering analytical questions").unwrap();
        let dialect_at = first.find("SQL reference").unwrap();
        let tables_at = first.find("Available tables:").unwrap();
        let documents_at = first.find("Ingested documents:").unwrap();
        let ontology_at = first.find("Ontology (version").unwrap();
        let context_at = first.find("Workspace context").unwrap();
        assert!(role_at < guidance_at);
        assert!(guidance_at < dialect_at);
        assert!(dialect_at < tables_at);
        assert!(tables_at < documents_at);
        assert!(documents_at < ontology_at);
        assert!(ontology_at < context_at, "{first}");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn context_is_placed_after_documents_and_truncated_to_budget() {
        let db = db();
        db.insert_document(
            &NewDocument::new("d1", "policy.pdf", "application/pdf", 1)
                .with_status(DocumentStatus::Ready),
        )
        .unwrap();
        let mut opts = options(ChatMode::Chat, 100);
        opts.context = Some(String::from("Amounts are in cents."));
        let prompt = SystemPrompt::build(&db, &opts).unwrap();
        let docs_at = prompt.find("Ingested documents:").unwrap();
        let ctx_at = prompt.find("Workspace context").unwrap();
        let perms_at = prompt.find("Permissions:").unwrap();
        assert!(docs_at < ctx_at && ctx_at < perms_at);
        assert!(prompt.contains("Amounts are in cents.\n"));
        assert!(!prompt.contains("truncated"));

        opts.context = Some("x".repeat(100));
        opts.context_max_tokens = 5;
        let prompt = SystemPrompt::build(&db, &opts).unwrap();
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
        db.insert_document(
            &NewDocument::new("d1", "policy.pdf", "application/pdf", 1)
                .with_status(DocumentStatus::Ready),
        )
        .unwrap();
        let chat = SystemPrompt::build(&db, &options(ChatMode::Chat, 1000)).unwrap();
        assert!(chat.contains("Mode: chat."));
        assert!(chat.contains("- claims (0 rows)"));
        db.execute_statement("INSERT INTO claims VALUES (1, 10), (2, 20)")
            .unwrap();
        let counted = SystemPrompt::build(&db, &options(ChatMode::Chat, 1000)).unwrap();
        assert!(counted.contains("- claims (2 rows)"), "{counted}");
        assert!(chat.contains("- policy.pdf (status: ready"));
        assert!(!chat.contains("Pinned documents"));
        assert!(
            chat.contains("needs write\n             permission")
                || chat.contains("needs write permission")
        );
        let mut allowed = options(ChatMode::Chat, 1000);
        allowed.write_policy = WritePolicy::Allow;
        let allowed = SystemPrompt::build(&db, &allowed).unwrap();
        assert!(allowed.contains("has permitted statements that"));
        let mut ask = options(ChatMode::Chat, 1000);
        ask.write_policy = WritePolicy::Ask;
        let ask = SystemPrompt::build(&db, &ask).unwrap();
        assert!(ask.contains("the user is asked to approve it"));
        let query = SystemPrompt::build(&db, &options(ChatMode::Query, 1000)).unwrap();
        assert!(query.contains("Mode: query."));
        assert!(query.contains("Do not answer from memory"));
        assert!(query.contains("MUST end with that chunk's [n]"));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn pinned_documents_are_injected_within_budget() {
        let db = db();
        db.insert_document(
            &NewDocument::new("d1", "rules.md", "text/markdown", 1)
                .with_status(DocumentStatus::Ready),
        )
        .unwrap();
        db.insert_document(
            &NewDocument::new("d2", "big.md", "text/markdown", 1)
                .with_status(DocumentStatus::Ready),
        )
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
        let prompt = SystemPrompt::build(&db, &options(ChatMode::Chat, 20)).unwrap();
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
        let prompt = SystemPrompt::build(&db, &options(ChatMode::Chat, 100)).unwrap();
        let version = db.duckdb_version().unwrap();
        assert!(version.starts_with('v'), "{version}");
        assert!(prompt.contains(&format!("DuckDB {version} SQL reference")));
        for idiom in [
            "GROUP BY ALL",
            "SUMMARIZE t profiles every column",
            "EXCLUDE (a, b)",
            "count() FILTER",
            "ASOF JOIN",
            "arg_max(label, measure, 3)",
            "QUALIFY row_number() OVER (PARTITION BY g",
            "never one per group",
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

    /// A wide table lists its first columns and counts the rest, shows
    /// no sample rows, and a long narrative cell is cut; tables past the
    /// detailed count appear by name only (issue #40).
    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn prompt_bounds_wide_tables_long_cells_and_many_tables() {
        let db = db();
        let columns: Vec<String> = (0..50).map(|i| format!("c{i} INT")).collect();
        db.execute_statement(&format!("CREATE TABLE a_wide({})", columns.join(", ")))
            .unwrap();
        db.execute_statement(&format!(
            "INSERT INTO a_wide VALUES ({})",
            (0..50)
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))
        .unwrap();
        let narrative = "x".repeat(500);
        db.execute_statement(&format!(
            "CREATE TABLE notes AS SELECT 1 AS id, '{narrative}' AS body"
        ))
        .unwrap();
        for i in 0..30 {
            db.execute_statement(&format!("CREATE TABLE t{i:02}(id INT)"))
                .unwrap();
        }
        let prompt = SystemPrompt::build(&db, &options(ChatMode::Chat, 1000)).unwrap();
        assert!(prompt.contains("- c39 (INTEGER)"), "{prompt}");
        assert!(!prompt.contains("- c40 (INTEGER)"), "{prompt}");
        assert!(prompt.contains("... and 10 more columns"), "{prompt}");
        assert!(
            prompt.contains("Sample rows omitted (50 columns)"),
            "{prompt}"
        );
        assert!(!prompt.contains(&narrative), "{prompt}");
        assert!(
            prompt.contains(&format!("{}\u{2026}", "x".repeat(60))),
            "{prompt}"
        );
        assert!(
            prompt.contains("Only the first 25 tables are described"),
            "{prompt}"
        );
        // The 32 tables all appear by name; the last ones without columns.
        assert!(prompt.contains("- t29 (0 rows)"), "{prompt}");
        assert_eq!(prompt.matches("  Columns:").count(), 25, "{prompt}");
    }

    #[test]
    fn ollama_context_size_rounds_up_within_bounds() {
        assert_eq!(ollama_context_size(0, 32_768), 8_192);
        assert_eq!(ollama_context_size(4 * 1_000, 32_768), 16_384);
        // 12,875 prompt tokens plus headroom rounds to 24,576.
        assert_eq!(ollama_context_size(4 * 12_875, 32_768), 24_576);
        assert_eq!(ollama_context_size(4 * 100_000, 32_768), 32_768);
        assert_eq!(ollama_context_size(4 * 100_000, 2_048), 8_192);
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn empty_workspace_prompt_says_so() {
        let prompt = SystemPrompt::build(&db(), &options(ChatMode::Chat, 100)).unwrap();
        assert!(prompt.contains("No tables or documents have been ingested yet"));
    }
}
