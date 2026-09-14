use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::config::Config;

/// Tables quack manages inside a workspace database. Hidden from the agent's
/// table listing and refused in agent SQL.
pub const INTERNAL_TABLES: &[&str] = &[
    "_quack_meta",
    "_quack_documents",
    "_quack_chunks",
    "_quack_sessions",
    "_quack_messages",
];

/// Every internal table carries this prefix; anything starting with it is hidden.
pub const INTERNAL_PREFIX: &str = "_quack_";

/// Schema version of the internal tables, recorded in `_quack_meta`.
const WORKSPACE_SCHEMA_VERSION: &str = "2";

/// Width used when no embedding provider is configured and the workspace has
/// not recorded one yet.
const DEFAULT_EMBEDDING_DIMENSION: u32 = 1024;

/// What a SQL statement would do if executed, decided by the `DuckDB` parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatementKind {
    /// A `SELECT` (or an equivalent read-only statement such as `DESCRIBE`).
    Read,
    /// Anything that could mutate state, load code, or reach outside the
    /// workspace: DDL, DML, `COPY`, `ATTACH`, `SET`, `INSTALL`, `LOAD`, ...
    Write,
    /// The `DuckDB` parser rejected it; execution will report the syntax error.
    Invalid(String),
}

/// Leading keywords that mark terminal input as SQL to run directly rather
/// than a question for the agent.
const DIRECT_SQL_KEYWORDS: &[&str] = &[
    "SELECT",
    "WITH",
    "FROM",
    "DESCRIBE",
    "SHOW",
    "PIVOT",
    "UNPIVOT",
    "SUMMARIZE",
    "EXPLAIN",
];

/// Whether interactive input should run as SQL instead of going to the agent.
#[must_use]
pub fn looks_like_direct_sql(input: &str) -> bool {
    let word: String = input
        .trim_start()
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .collect();
    !word.is_empty()
        && DIRECT_SQL_KEYWORDS
            .iter()
            .any(|k| k.eq_ignore_ascii_case(&word))
}

/// Read-only statements `DuckDB` cannot serialize to JSON but which never mutate.
const READ_ONLY_KEYWORDS: &[&str] = &[
    "DESCRIBE",
    "SHOW",
    "SUMMARIZE",
    "PIVOT",
    "UNPIVOT",
    "EXPLAIN",
];

/// Query result set from a `DuckDB` workspace database.
#[derive(Debug)]
pub struct QueryResults {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
}

/// Wraps a `DuckDB` connection for a single workspace.
pub struct WorkspaceDb {
    conn: duckdb::Connection,
    embedding_dimension: u32,
    query_timeout: Duration,
}

impl WorkspaceDb {
    /// Open an in-memory `DuckDB` database (for tests).
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created.
    pub fn open_in_memory(embedding_dimension: u32) -> crate::error::Result<Self> {
        let conn = duckdb::Connection::open_in_memory()?;
        let db = Self {
            conn,
            embedding_dimension,
            query_timeout: Duration::from_secs(30),
        };
        db.create_internal_tables()?;
        Ok(db)
    }

    /// Override the per-statement timeout (tests and callers with special needs).
    #[must_use]
    pub fn with_query_timeout(mut self, timeout: Duration) -> Self {
        self.query_timeout = timeout;
        self
    }

    /// Open (or create) the `DuckDB` database for a workspace.
    ///
    /// Loads the vss extension, creates the `_quack_` internal tables if they
    /// do not exist, and reconciles the embedding dimension recorded in
    /// `_quack_meta` with the configured provider.
    ///
    /// # Errors
    ///
    /// Returns an error if the database file cannot be created or opened.
    pub fn open(config: &Config, workspace_id: &str) -> crate::error::Result<Self> {
        let db_path = config.workspace_db_path(workspace_id);
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let files_dir = config.workspace_files_dir(workspace_id);
        std::fs::create_dir_all(&files_dir)?;

        let embedding = config.embedding_model_ref()?;
        let configured_dimension = embedding.and_then(|m| m.provider.embedding_dimension);
        let configured_model = embedding.map(|m| m.model);

        let conn = duckdb::Connection::open(&db_path)?;

        let mut db = Self {
            conn,
            embedding_dimension: configured_dimension.unwrap_or(DEFAULT_EMBEDDING_DIMENSION),
            query_timeout: Duration::from_secs(u64::from(config.analysis.query_timeout_seconds)),
        };
        db.apply_resource_limits(config)?;
        db.load_vss();
        db.rename_legacy_tables()?;
        db.reconcile_embedding_dimension(configured_dimension, configured_model)
            .or_else(|e| match e {
                // A brand-new workspace has no meta table yet; create it first.
                crate::error::Error::DuckDb(_) => {
                    db.create_internal_tables()?;
                    db.reconcile_embedding_dimension(configured_dimension, configured_model)
                }
                other => Err(other),
            })?;
        db.create_internal_tables()?;
        Ok(db)
    }

    /// Cap memory and parallelism for every statement on this connection.
    fn apply_resource_limits(&self, config: &Config) -> crate::error::Result<()> {
        let memory_limit = format!("{}MB", config.analysis.memory_limit_mb);
        self.conn
            .execute("SET memory_limit = ?", duckdb::params![memory_limit])?;
        self.conn.execute(
            "SET threads = ?",
            duckdb::params![i64::from(config.analysis.threads.max(1))],
        )?;
        Ok(())
    }

    /// Classify a statement as read, write, or invalid using `DuckDB`'s parser.
    ///
    /// `json_serialize_sql` succeeds only for `SELECT`-shaped statements; a
    /// "not implemented" error means some other statement type (or several
    /// statements) and is treated as a write. A short allowlist of read-only
    /// keywords (`DESCRIBE`, `SHOW`, `SUMMARIZE`, `PIVOT`, `UNPIVOT`,
    /// `EXPLAIN`) covers statements `DuckDB` cannot serialize but which never
    /// mutate, and only when the input is a single statement.
    ///
    /// # Errors
    ///
    /// Returns an error if the classification query itself fails.
    pub fn classify_statement(&self, sql: &str) -> crate::error::Result<StatementKind> {
        let serialized: String = self.conn.query_row(
            "SELECT json_serialize_sql(?::VARCHAR)",
            duckdb::params![sql],
            |row| row.get(0),
        )?;
        let parsed: serde_json::Value = serde_json::from_str(&serialized)?;
        let is_error = parsed
            .get("error")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        if !is_error {
            let statement_count = parsed
                .get("statements")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            return Ok(if statement_count == 0 {
                StatementKind::Invalid(String::from("empty statement"))
            } else {
                StatementKind::Read
            });
        }
        let error_type = parsed
            .get("error_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if error_type == "parser" {
            let message = parsed
                .get("error_message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("syntax error");
            return Ok(StatementKind::Invalid(message.to_owned()));
        }
        Ok(if is_single_read_only_statement(sql) {
            StatementKind::Read
        } else {
            StatementKind::Write
        })
    }

    /// Names of tables a statement references, as `DuckDB` parsed them, or
    /// `None` when `DuckDB` cannot serialize the statement.
    fn referenced_base_tables(&self, sql: &str) -> crate::error::Result<Option<Vec<String>>> {
        let serialized: String = self.conn.query_row(
            "SELECT json_serialize_sql(?::VARCHAR)",
            duckdb::params![sql],
            |row| row.get(0),
        )?;
        let parsed: serde_json::Value = serde_json::from_str(&serialized)?;
        let is_error = parsed
            .get("error")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        if is_error {
            return Ok(None);
        }
        let mut names = Vec::new();
        collect_table_names(&parsed, &mut names);
        Ok(Some(names))
    }

    /// Whether a statement touches any of quack's internal tables.
    ///
    /// Uses the parsed table references when `DuckDB` can serialize the
    /// statement, so string literals do not count; falls back to a
    /// conservative token scan for everything else.
    ///
    /// # Errors
    ///
    /// Returns an error if the classification query fails.
    pub fn references_internal_table(&self, sql: &str) -> crate::error::Result<bool> {
        match self.referenced_base_tables(sql)? {
            Some(names) => Ok(names.iter().any(|n| is_internal_name(n))),
            None => Ok(mentions_internal_table_token(sql)),
        }
    }

    fn load_vss(&self) {
        if let Err(e) = self.conn.execute("INSTALL vss", []) {
            tracing::debug!(err = %e, "vss INSTALL skipped (may already be installed)");
        }
        if let Err(e) = self.conn.execute("LOAD vss", []) {
            tracing::warn!(err = %e, "failed to load vss extension — vector indexing unavailable");
        }
    }

    fn create_internal_tables(&self) -> crate::error::Result<()> {
        self.rename_legacy_tables()?;
        let dim = self.embedding_dimension;
        let sql = format!(
            "CREATE TABLE IF NOT EXISTS _quack_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS _quack_documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                mime_type TEXT,
                size_bytes BIGINT,
                ingested_at TIMESTAMP DEFAULT now(),
                status TEXT DEFAULT 'pending',
                error_message TEXT
            );
            CREATE TABLE IF NOT EXISTS _quack_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding FLOAT[{dim}],
                token_count INTEGER
            );
            CREATE TABLE IF NOT EXISTS _quack_sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                mode TEXT NOT NULL DEFAULT 'chat',
                model TEXT NOT NULL,
                created_by TEXT,
                shared BOOLEAN NOT NULL DEFAULT false,
                created_at TIMESTAMP DEFAULT now(),
                updated_at TIMESTAMP DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS _quack_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT now(),
                UNIQUE (session_id, seq)
            );"
        );
        self.conn.execute_batch(&sql)?;
        self.set_meta("schema_version", WORKSPACE_SCHEMA_VERSION)?;
        self.set_meta("embedding_dimension", &dim.to_string())?;
        Ok(())
    }

    /// Workspaces created before the `_quack_` prefix keep their data.
    fn rename_legacy_tables(&self) -> crate::error::Result<()> {
        for (old, new) in [
            ("documents", "_quack_documents"),
            ("chunks", "_quack_chunks"),
        ] {
            let old_exists = self.table_exists(old)?;
            let new_exists = self.table_exists(new)?;
            if old_exists && !new_exists {
                self.conn
                    .execute(&format!("ALTER TABLE {old} RENAME TO {new}"), [])?;
                tracing::info!(old, new, "renamed legacy internal table");
            }
        }
        Ok(())
    }

    fn table_exists(&self, name: &str) -> crate::error::Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
            duckdb::params![name],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Read a `_quack_meta` value, if the table and key exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn meta(&self, key: &str) -> crate::error::Result<Option<String>> {
        if !self.table_exists("_quack_meta")? {
            return Ok(None);
        }
        let mut stmt = self
            .conn
            .prepare("SELECT value FROM _quack_meta WHERE key = ?")?;
        let mut rows = stmt.query(duckdb::params![key])?;
        match rows.next()? {
            Some(row) => Ok(Some(row.get(0)?)),
            None => Ok(None),
        }
    }

    fn set_meta(&self, key: &str, value: &str) -> crate::error::Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO _quack_meta (key, value) VALUES (?, ?)",
            duckdb::params![key, value],
        )?;
        Ok(())
    }

    /// The embedding width this workspace stores.
    #[must_use]
    pub fn embedding_dimension(&self) -> u32 {
        self.embedding_dimension
    }

    /// Reconcile the configured embedding dimension with what the workspace
    /// recorded. A fresh or empty workspace adopts the configured value; a
    /// workspace that already holds embeddings of a different width is an
    /// error, never a silent mismatch.
    fn reconcile_embedding_dimension(
        &mut self,
        configured: Option<u32>,
        configured_model: Option<&str>,
    ) -> crate::error::Result<()> {
        let recorded = self
            .meta("embedding_dimension")?
            .and_then(|v| v.parse::<u32>().ok());
        let recorded_model = self.meta("embedding_model")?;

        match (recorded, configured) {
            (Some(rec), Some(conf)) if rec != conf => {
                let stored_chunks: i64 = self.conn.query_row(
                    "SELECT count(*) FROM _quack_chunks WHERE embedding IS NOT NULL",
                    [],
                    |row| row.get(0),
                )?;
                if stored_chunks > 0 {
                    return Err(crate::error::Error::Config(format!(
                        "workspace embeddings are {rec}-dimensional ({}) but the configured \
                         provider produces {conf}-dimensional ({}) vectors; re-ingest the \
                         documents or switch back to the original embedding model",
                        recorded_model.as_deref().unwrap_or("unknown model"),
                        configured_model.unwrap_or("unknown model"),
                    )));
                }
                tracing::info!(
                    from = rec,
                    to = conf,
                    "no embeddings stored; adopting the configured embedding dimension"
                );
                self.conn.execute_batch(&format!(
                    "DROP TABLE _quack_chunks;
                     CREATE TABLE _quack_chunks (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding FLOAT[{conf}],
                        token_count INTEGER
                    );"
                ))?;
                self.embedding_dimension = conf;
            }
            (Some(rec), None) => self.embedding_dimension = rec,
            (None, Some(conf)) => self.embedding_dimension = conf,
            (Some(_), Some(_)) | (None, None) => {}
        }
        self.set_meta("embedding_dimension", &self.embedding_dimension.to_string())?;
        if let Some(model) = configured_model {
            self.set_meta("embedding_model", model)?;
        }
        Ok(())
    }

    /// Insert a document metadata row.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub fn insert_document(
        &self,
        id: &str,
        filename: &str,
        mime_type: &str,
        size_bytes: usize,
        status: &str,
    ) -> crate::error::Result<()> {
        let size = i64::try_from(size_bytes)
            .map_err(|_| crate::error::Error::Ingestion("file size overflow".into()))?;

        self.conn.execute(
            "INSERT INTO _quack_documents (id, filename, mime_type, size_bytes, status) VALUES (?, ?, ?, ?, ?)",
            duckdb::params![id, filename, mime_type, size, status],
        )?;
        Ok(())
    }

    /// Update a document's status.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn update_document_status(&self, id: &str, status: &str) -> crate::error::Result<()> {
        self.conn.execute(
            "UPDATE _quack_documents SET status = ? WHERE id = ?",
            duckdb::params![status, id],
        )?;
        Ok(())
    }

    /// Insert a text chunk, optionally with an embedding vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub fn insert_chunk(
        &self,
        id: &str,
        document_id: &str,
        chunk_index: u32,
        content: &str,
        embedding: Option<&[f32]>,
    ) -> crate::error::Result<()> {
        match embedding {
            Some(emb) => {
                let sql = format!(
                    "INSERT INTO _quack_chunks (id, document_id, chunk_index, content, embedding) \
                     VALUES (?, ?, ?, ?, ?::{})",
                    self.vector_type()
                );
                self.conn.execute(
                    &sql,
                    duckdb::params![id, document_id, chunk_index, content, format_embedding(emb)],
                )?;
            }
            None => {
                self.conn.execute(
                    "INSERT INTO _quack_chunks (id, document_id, chunk_index, content) VALUES (?, ?, ?, ?)",
                    duckdb::params![id, document_id, chunk_index, content],
                )?;
            }
        }
        Ok(())
    }

    /// Update a chunk's embedding vector by document ID and chunk index.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn update_chunk_embedding(
        &self,
        document_id: &str,
        chunk_index: u32,
        embedding: &[f32],
    ) -> crate::error::Result<()> {
        let sql = format!(
            "UPDATE _quack_chunks SET embedding = ?::{} \
             WHERE document_id = ? AND chunk_index = ?",
            self.vector_type()
        );
        self.conn.execute(
            &sql,
            duckdb::params![format_embedding(embedding), document_id, chunk_index],
        )?;
        Ok(())
    }

    /// The `FLOAT[N]` type of this workspace's embedding column. `N` is a
    /// validated integer, the only value ever interpolated into vector SQL.
    fn vector_type(&self) -> String {
        format!("FLOAT[{}]", self.embedding_dimension)
    }

    /// Create an HNSW index on the chunks embedding column for cosine similarity.
    ///
    /// Requires the vss extension to be loaded.
    ///
    /// # Errors
    ///
    /// Returns an error if index creation fails (e.g., vss not loaded or
    /// no embeddings stored yet).
    pub fn create_embedding_index(&self) -> crate::error::Result<()> {
        self.conn
            .execute("SET hnsw_enable_experimental_persistence = true", [])?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_idx \
             ON _quack_chunks USING HNSW (embedding) \
             WITH (metric = 'cosine')",
            [],
        )?;
        tracing::info!("created HNSW cosine index on _quack_chunks.embedding");
        Ok(())
    }

    /// Search for the most similar chunks to a query embedding.
    ///
    /// When `document_ids` is non-empty the search is restricted to those
    /// documents. Results carry the source filename for citations.
    ///
    /// # Errors
    ///
    /// Returns an error if the search query fails.
    pub fn search_similar_chunks(
        &self,
        query_embedding: &[f32],
        top_k: u32,
        document_ids: &[String],
    ) -> crate::error::Result<Vec<ChunkSearchResult>> {
        let filter = if document_ids.is_empty() {
            String::new()
        } else {
            let placeholders = vec!["?"; document_ids.len()].join(", ");
            format!(" AND c.document_id IN ({placeholders})")
        };
        let sql = format!(
            "SELECT c.id, c.content, c.document_id, c.chunk_index, d.filename, \
                    array_cosine_distance(c.embedding, ?::{}) AS distance \
             FROM _quack_chunks c \
             JOIN _quack_documents d ON d.id = c.document_id \
             WHERE c.embedding IS NOT NULL{filter} \
             ORDER BY distance ASC \
             LIMIT ?",
            self.vector_type()
        );

        let query_literal = format_embedding(query_embedding);
        let limit = i64::from(top_k);
        let mut stmt = self.conn.prepare(&sql)?;
        let mut params: Vec<&dyn duckdb::ToSql> =
            Vec::with_capacity(document_ids.len().saturating_add(2));
        params.push(&query_literal);
        for id in document_ids {
            params.push(id);
        }
        params.push(&limit);
        let mut rows = stmt.query(params.as_slice())?;
        let mut results = Vec::new();

        while let Some(row) = rows.next()? {
            results.push(ChunkSearchResult {
                id: row.get(0)?,
                content: row.get(1)?,
                document_id: row.get(2)?,
                chunk_index: row.get(3)?,
                filename: row.get(4)?,
                distance: row.get(5)?,
            });
        }

        Ok(results)
    }

    /// Execute an arbitrary SQL statement and return the results.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid or execution fails.
    pub fn execute_query(&self, sql: &str) -> crate::error::Result<QueryResults> {
        let _guard = self.arm_timeout();
        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query([])?;

        let (columns, column_count) = {
            let Some(stmt_ref) = rows.as_ref() else {
                return Ok(QueryResults {
                    columns: Vec::new(),
                    rows: Vec::new(),
                });
            };
            let count = stmt_ref.column_count();
            if count == 0 {
                return Ok(QueryResults {
                    columns: Vec::new(),
                    rows: Vec::new(),
                });
            }
            (stmt_ref.column_names(), count)
        };

        let mut result_rows: Vec<Vec<serde_json::Value>> = Vec::new();
        while let Some(row) = rows.next()? {
            let mut values = Vec::with_capacity(column_count);
            for i in 0..column_count {
                values.push(extract_value(row, i));
            }
            result_rows.push(values);
        }

        Ok(QueryResults {
            columns,
            rows: result_rows,
        })
    }

    /// Execute a SQL statement that does not return rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid or execution fails.
    pub fn execute_statement(&self, sql: &str) -> crate::error::Result<()> {
        self.execute_with_params(sql, [])
    }

    /// Execute a parameterized statement that does not return rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid or execution fails.
    pub fn execute_with_params<P: duckdb::Params>(
        &self,
        sql: &str,
        params: P,
    ) -> crate::error::Result<()> {
        let _guard = self.arm_timeout();
        self.conn.execute(sql, params)?;
        Ok(())
    }

    /// Start a watchdog that interrupts the connection if the statement runs
    /// past the configured timeout. Dropping the guard disarms it.
    fn arm_timeout(&self) -> TimeoutGuard {
        let done = Arc::new(AtomicBool::new(false));
        let handle = self.conn.interrupt_handle();
        let timeout = self.query_timeout;
        let done_for_thread = Arc::clone(&done);
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            while started.elapsed() < timeout {
                if done_for_thread.load(Ordering::Acquire) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            if !done_for_thread.load(Ordering::Acquire) {
                tracing::warn!(?timeout, "statement exceeded timeout; interrupting");
                handle.interrupt();
            }
        });
        TimeoutGuard { done }
    }

    /// List all user-created tables in the workspace (excludes internal tables).
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_tables(&self) -> crate::error::Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name",
        )?;
        let mut rows = stmt.query([])?;
        let mut tables = Vec::new();
        while let Some(row) = rows.next()? {
            let name: String = row.get(0)?;
            if !is_internal_name(&name) {
                tables.push(name);
            }
        }
        Ok(tables)
    }

    /// Describe a table's columns (name, type) and return up to 3 sample rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist or the query fails.
    pub fn describe_table(&self, table_name: &str) -> crate::error::Result<TableDescription> {
        let describe_sql = format!("DESCRIBE {}", quote_ident(table_name));
        let mut stmt = self.conn.prepare(&describe_sql)?;
        let mut rows = stmt.query([])?;
        let mut columns = Vec::new();
        while let Some(row) = rows.next()? {
            columns.push(ColumnInfo {
                name: row.get(0)?,
                column_type: row.get(1)?,
            });
        }

        let sample_sql = format!("SELECT * FROM {} LIMIT 3", quote_ident(table_name));
        let sample = self.execute_query(&sample_sql)?;

        Ok(TableDescription {
            table_name: table_name.to_owned(),
            columns,
            sample_rows: sample,
        })
    }

    /// List all ingested documents with their status.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_documents(&self) -> crate::error::Result<Vec<DocumentInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, filename, mime_type, size_bytes, status FROM _quack_documents ORDER BY ingested_at DESC",
        )?;
        let mut rows = stmt.query([])?;
        let mut docs = Vec::new();
        while let Some(row) = rows.next()? {
            docs.push(DocumentInfo {
                id: row.get(0)?,
                filename: row.get(1)?,
                mime_type: row.get(2)?,
                size_bytes: row.get(3)?,
                status: row.get(4)?,
            });
        }
        Ok(docs)
    }

    /// Access the underlying `DuckDB` connection.
    #[must_use]
    pub fn connection(&self) -> &duckdb::Connection {
        &self.conn
    }
}

/// Column metadata from DESCRIBE.
#[derive(Debug)]
pub struct ColumnInfo {
    pub name: String,
    pub column_type: String,
}

/// Full table description with schema and sample data.
#[derive(Debug)]
pub struct TableDescription {
    pub table_name: String,
    pub columns: Vec<ColumnInfo>,
    pub sample_rows: QueryResults,
}

/// Document metadata row.
#[derive(Debug)]
pub struct DocumentInfo {
    pub id: String,
    pub filename: String,
    pub mime_type: Option<String>,
    pub size_bytes: Option<i64>,
    pub status: String,
}

/// A chunk returned from vector similarity search.
#[derive(Debug)]
pub struct ChunkSearchResult {
    pub id: String,
    pub content: String,
    pub document_id: String,
    pub chunk_index: u32,
    pub filename: String,
    pub distance: f64,
}

struct TimeoutGuard {
    done: Arc<AtomicBool>,
}

impl Drop for TimeoutGuard {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
    }
}

/// True when `sql` is one statement that starts with a read-only keyword
/// `DuckDB` cannot serialize. A semicolon anywhere but the very end disqualifies
/// it, so `DESCRIBE t; DROP TABLE t` is not read-only.
fn is_single_read_only_statement(sql: &str) -> bool {
    let trimmed = sql.trim().trim_end_matches(';').trim_end();
    if trimmed.contains(';') {
        return false;
    }
    let Some(first) = trimmed.split_whitespace().next() else {
        return false;
    };
    READ_ONLY_KEYWORDS
        .iter()
        .any(|k| k.eq_ignore_ascii_case(first))
}

/// Walk a serialized statement tree collecting `table_name` values from
/// base-table references.
fn collect_table_names(node: &serde_json::Value, out: &mut Vec<String>) {
    match node {
        serde_json::Value::Object(map) => {
            let is_base_table = map
                .get("type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|t| t == "BASE_TABLE");
            if is_base_table
                && let Some(name) = map.get("table_name").and_then(serde_json::Value::as_str)
            {
                out.push(name.to_owned());
            }
            for child in map.values() {
                collect_table_names(child, out);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_table_names(item, out);
            }
        }
        serde_json::Value::Null
        | serde_json::Value::Bool(_)
        | serde_json::Value::Number(_)
        | serde_json::Value::String(_) => {}
    }
}

/// Conservative token scan used for statements the parser will not serialize.
fn is_internal_name(name: &str) -> bool {
    name.to_ascii_lowercase().starts_with(INTERNAL_PREFIX)
}

fn mentions_internal_table_token(sql: &str) -> bool {
    sql.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .any(is_internal_name)
}

/// Quote a SQL identifier for `DuckDB`: wrap in double quotes and double
/// any embedded double quote. This is the only way identifiers enter SQL.
#[must_use]
pub fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Render a vector as the string literal `DuckDB` casts to `FLOAT[N]`. The
/// result is always bound as a parameter, never interpolated.
fn format_embedding(embedding: &[f32]) -> String {
    let inner: Vec<String> = embedding.iter().map(|v| format!("{v}")).collect();
    format!("[{}]", inner.join(","))
}

fn extract_value(row: &duckdb::Row<'_>, idx: usize) -> serde_json::Value {
    if let Ok(v) = row.get::<_, Option<i64>>(idx) {
        return match v {
            Some(n) => serde_json::Value::Number(n.into()),
            None => serde_json::Value::Null,
        };
    }
    if let Ok(v) = row.get::<_, Option<f64>>(idx) {
        return match v {
            Some(n) => serde_json::Number::from_f64(n)
                .map_or(serde_json::Value::Null, serde_json::Value::Number),
            None => serde_json::Value::Null,
        };
    }
    if let Ok(v) = row.get::<_, Option<bool>>(idx) {
        return match v {
            Some(b) => serde_json::Value::Bool(b),
            None => serde_json::Value::Null,
        };
    }
    if let Ok(v) = row.get::<_, Option<String>>(idx) {
        return match v {
            Some(s) => serde_json::Value::String(s),
            None => serde_json::Value::Null,
        };
    }
    serde_json::Value::Null
}

fn display_json_value(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::Null => String::from("NULL"),
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

impl QueryResults {
    /// Return a copy of the results with at most `max_rows` rows.
    #[must_use]
    pub fn clone_capped(&self, max_rows: u32) -> Self {
        let limit = max_rows as usize;
        Self {
            columns: self.columns.clone(),
            rows: self.rows.iter().take(limit).cloned().collect(),
        }
    }

    /// Write results as a human-readable aligned table.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to `out` fails.
    pub fn write_table(&self, out: &mut impl Write) -> crate::error::Result<()> {
        if self.columns.is_empty() {
            writeln!(out, "OK")?;
            return Ok(());
        }

        let display_rows: Vec<Vec<String>> = self
            .rows
            .iter()
            .map(|row| row.iter().map(display_json_value).collect())
            .collect();

        let mut widths: Vec<usize> = self.columns.iter().map(String::len).collect();
        for row in &display_rows {
            for (w, val) in widths.iter_mut().zip(row.iter()) {
                *w = (*w).max(val.len());
            }
        }

        for (i, (col, width)) in self.columns.iter().zip(widths.iter()).enumerate() {
            if i > 0 {
                write!(out, " | ")?;
            }
            write!(out, "{col:<width$}")?;
        }
        writeln!(out)?;

        for (i, width) in widths.iter().enumerate() {
            if i > 0 {
                write!(out, "-+-")?;
            }
            for _ in 0..*width {
                write!(out, "-")?;
            }
        }
        writeln!(out)?;

        for row in &display_rows {
            for (i, (val, width)) in row.iter().zip(widths.iter()).enumerate() {
                if i > 0 {
                    write!(out, " | ")?;
                }
                write!(out, "{val:<width$}")?;
            }
            writeln!(out)?;
        }

        let row_count = self.rows.len();
        writeln!(out, "({row_count} rows)")?;
        Ok(())
    }

    /// One JSON object per line.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or writing fails.
    pub fn write_ndjson(&self, out: &mut impl Write) -> crate::error::Result<()> {
        // Written by hand so keys keep column order; serde_json's map sorts.
        for row in &self.rows {
            let mut fields = Vec::with_capacity(self.columns.len());
            for (column, value) in self.columns.iter().zip(row) {
                fields.push(format!(
                    "{}:{}",
                    serde_json::to_string(column)?,
                    serde_json::to_string(value)?
                ));
            }
            writeln!(out, "{{{}}}", fields.join(","))?;
        }
        Ok(())
    }

    /// RFC 4180 CSV with a header row; fields containing a comma, quote, or
    /// newline are quoted and embedded quotes doubled.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn write_csv(&self, out: &mut impl Write) -> crate::error::Result<()> {
        fn field(value: &str) -> String {
            if value.contains([',', '"', '\n', '\r']) {
                format!("\"{}\"", value.replace('"', "\"\""))
            } else {
                value.to_owned()
            }
        }
        let header: Vec<String> = self.columns.iter().map(|c| field(c)).collect();
        writeln!(out, "{}", header.join(","))?;
        for row in &self.rows {
            let cells: Vec<String> = row
                .iter()
                .map(|v| match v {
                    serde_json::Value::Null => String::new(),
                    other => field(&display_json_value(other)),
                })
                .collect();
            writeln!(out, "{}", cells.join(","))?;
        }
        Ok(())
    }

    /// A GitHub-flavored Markdown table.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn write_markdown(&self, out: &mut impl Write) -> crate::error::Result<()> {
        fn cell(value: &str) -> String {
            value.replace('|', "\\|").replace('\n', " ")
        }
        if self.columns.is_empty() {
            writeln!(out, "OK")?;
            return Ok(());
        }
        let header: Vec<String> = self.columns.iter().map(|c| cell(c)).collect();
        writeln!(out, "| {} |", header.join(" | "))?;
        let rule: Vec<&str> = self.columns.iter().map(|_| "---").collect();
        writeln!(out, "| {} |", rule.join(" | "))?;
        for row in &self.rows {
            let cells: Vec<String> = row.iter().map(|v| cell(&display_json_value(v))).collect();
            writeln!(out, "| {} |", cells.join(" | "))?;
        }
        Ok(())
    }

    /// Write results as a JSON array of objects.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to `out` or JSON serialization fails.
    pub fn write_json(&self, out: &mut impl Write) -> crate::error::Result<()> {
        let json_rows: Vec<serde_json::Map<String, serde_json::Value>> = self
            .rows
            .iter()
            .map(|row| {
                self.columns
                    .iter()
                    .zip(row.iter())
                    .map(|(col, val)| (col.clone(), val.clone()))
                    .collect()
            })
            .collect();

        serde_json::to_writer_pretty(&mut *out, &json_rows)?;
        writeln!(out)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> QueryResults {
        QueryResults {
            columns: vec![String::from("name"), String::from("n")],
            rows: vec![
                vec![
                    serde_json::Value::String(String::from("a,b")),
                    serde_json::Value::Number(1.into()),
                ],
                vec![
                    serde_json::Value::String(String::from("say \"hi\"")),
                    serde_json::Value::Null,
                ],
            ],
        }
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn write_ndjson_one_object_per_line() {
        let mut buf = Vec::new();
        sample().write_ndjson(&mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines.first().copied(), Some(r#"{"name":"a,b","n":1}"#));
        assert_eq!(
            lines.last().copied(),
            Some(r#"{"name":"say \"hi\"","n":null}"#)
        );
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn write_csv_quotes_and_escapes() {
        let mut buf = Vec::new();
        sample().write_csv(&mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert_eq!(text, "name,n\n\"a,b\",1\n\"say \"\"hi\"\"\",\n");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn write_markdown_renders_table() {
        let mut buf = Vec::new();
        sample().write_markdown(&mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.starts_with("| name | n |\n| --- | --- |\n| a,b | 1 |\n"));
        assert!(text.contains("| say \"hi\" | NULL |"));
    }

    #[test]
    fn direct_sql_detection_by_leading_keyword() {
        for yes in [
            "SELECT 1",
            "  with x as (select 1) select * from x",
            "FROM t",
            "describe t",
            "SHOW TABLES",
            "summarize t",
            "PIVOT t ON a",
            "explain select 1",
            "select(1)",
        ] {
            assert!(looks_like_direct_sql(yes), "{yes}");
        }
        for no in [
            "what were sales by region",
            "",
            "   ",
            "/sql select 1",
            "selected items please",
            "DROP TABLE t",
            "insert into t values (1)",
        ] {
            assert!(!looks_like_direct_sql(no), "{no}");
        }
    }

    #[test]
    fn quote_ident_wraps_and_escapes() {
        assert_eq!(quote_ident("sales"), "\"sales\"");
        assert_eq!(quote_ident("odd name"), "\"odd name\"");
        assert_eq!(quote_ident("x\"y"), "\"x\"\"y\"");
    }

    #[test]
    fn format_embedding_multiple_values() {
        let emb = [1.0_f32, 2.5, -3.0];
        assert_eq!(format_embedding(&emb), "[1,2.5,-3]");
    }

    #[test]
    fn format_embedding_empty() {
        let emb: [f32; 0] = [];
        assert_eq!(format_embedding(&emb), "[]");
    }

    #[test]
    fn format_embedding_single_value() {
        let emb = [0.5_f32];
        assert_eq!(format_embedding(&emb), "[0.5]");
    }

    #[test]
    fn display_json_null() {
        assert_eq!(display_json_value(&serde_json::Value::Null), "NULL");
    }

    #[test]
    fn display_json_string() {
        let val = serde_json::Value::String("hello".into());
        assert_eq!(display_json_value(&val), "hello");
    }

    #[test]
    fn display_json_number() {
        let val = serde_json::Value::Number(42.into());
        assert_eq!(display_json_value(&val), "42");
    }

    #[test]
    fn display_json_bool_true() {
        assert_eq!(display_json_value(&serde_json::Value::Bool(true)), "true");
    }

    #[test]
    fn display_json_bool_false() {
        assert_eq!(display_json_value(&serde_json::Value::Bool(false)), "false");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts output format")]
    fn write_table_empty_columns_prints_ok() {
        let results = QueryResults {
            columns: Vec::new(),
            rows: Vec::new(),
        };
        let mut buf = Vec::new();
        results.write_table(&mut buf).unwrap();
        assert_eq!(String::from_utf8_lossy(&buf), "OK\n");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts output format")]
    fn write_table_renders_aligned_columns() {
        let results = QueryResults {
            columns: vec!["id".into(), "name".into()],
            rows: vec![
                vec![
                    serde_json::Value::Number(1.into()),
                    serde_json::Value::String("alice".into()),
                ],
                vec![
                    serde_json::Value::Number(2.into()),
                    serde_json::Value::String("bob".into()),
                ],
            ],
        };
        let mut buf = Vec::new();
        results.write_table(&mut buf).unwrap();
        let output = String::from_utf8_lossy(&buf);
        assert!(output.contains("id"));
        assert!(output.contains("name"));
        assert!(output.contains("alice"));
        assert!(output.contains("bob"));
        assert!(output.contains("(2 rows)"));
        assert!(output.contains("-+-"));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts output format")]
    fn write_json_produces_valid_array() {
        let results = QueryResults {
            columns: vec!["id".into(), "val".into()],
            rows: vec![vec![
                serde_json::Value::Number(1.into()),
                serde_json::Value::String("x".into()),
            ]],
        };
        let mut buf = Vec::new();
        results.write_json(&mut buf).unwrap();
        let output = String::from_utf8_lossy(&buf);
        let parsed: Vec<serde_json::Map<String, serde_json::Value>> =
            serde_json::from_str(&output).unwrap();
        assert_eq!(parsed.len(), 1);
        let first = parsed.first().unwrap();
        assert_eq!(first.get("id"), Some(&serde_json::Value::Number(1.into())));
        assert_eq!(
            first.get("val"),
            Some(&serde_json::Value::String("x".into()))
        );
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts output format")]
    fn write_json_empty_rows_produces_empty_array() {
        let results = QueryResults {
            columns: vec!["a".into()],
            rows: Vec::new(),
        };
        let mut buf = Vec::new();
        results.write_json(&mut buf).unwrap();
        let output = String::from_utf8_lossy(&buf);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
        assert!(parsed.is_empty());
    }
}
