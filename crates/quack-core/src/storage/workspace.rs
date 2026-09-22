use std::io::Write;
use std::path::Path;
use std::time::Duration;

use crate::config::Config;
use crate::error::{Error, Result};

/// Tables quack manages inside a workspace database. Hidden from the agent's
/// table listing and refused in agent SQL.
pub const INTERNAL_TABLES: &[&str] = &[
    "_quack_meta",
    "_quack_documents",
    "_quack_chunks",
    "_quack_sessions",
    "_quack_messages",
    "_quack_terms",
    "_quack_context",
];

/// BM25 parameters for the keyword index quack maintains in `_quack_terms`.
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

/// How far a quoted-phrase query over-fetches BM25 candidates before the
/// substring post-filter, since `_quack_terms` carries no positions.
const PHRASE_OVER_FETCH: u32 = 4;
/// Absolute cap on phrase-search candidates, regardless of `top_k`.
const PHRASE_CANDIDATE_CAP: u32 = 500;

/// Every internal table carries this prefix; anything starting with it is hidden.
pub const INTERNAL_PREFIX: &str = "_quack_";

/// Schema version of the internal tables, recorded in `_quack_meta`.
const WORKSPACE_SCHEMA_VERSION: &str = "7";

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
#[derive(Debug, Clone)]
pub struct QueryResults {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
}

/// Query results with at most a caller's cap of rows kept, plus how many
/// the statement produced in all.
#[derive(Debug, Clone)]
pub struct CappedResults {
    pub results: QueryResults,
    pub total_rows: usize,
}

impl CappedResults {
    /// Whether rows were dropped to honor the cap.
    #[must_use]
    pub fn truncated(&self) -> bool {
        self.total_rows > self.results.rows.len()
    }

    /// Rows the cap dropped.
    #[must_use]
    pub fn omitted(&self) -> usize {
        self.total_rows.saturating_sub(self.results.rows.len())
    }
}

/// The ontology tables (design doc 5.4), created with the other internal
/// tables.
const ONTOLOGY_DDL: &str = "            CREATE TABLE IF NOT EXISTS _quack_ontology_versions (
                version INTEGER PRIMARY KEY,
                snapshot JSON NOT NULL,
                author TEXT,
                note TEXT,
                created_at TIMESTAMP DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS _quack_ontology_classes (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                label TEXT NOT NULL,
                description TEXT,
                key_property TEXT,
                since_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS _quack_ontology_relations (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                description TEXT,
                domain_class TEXT NOT NULL,
                range_class TEXT NOT NULL,
                since_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS _quack_ontology_properties (
                id TEXT NOT NULL,
                class_id TEXT NOT NULL,
                label TEXT NOT NULL,
                type TEXT NOT NULL,
                enum_values JSON,
                since_version INTEGER NOT NULL,
                PRIMARY KEY (id, class_id)
            );
            CREATE TABLE IF NOT EXISTS _quack_ontology_mappings (
                id TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                class_id TEXT NOT NULL,
                key_column TEXT NOT NULL,
                property_map JSON NOT NULL,
                relation_map JSON NOT NULL,
                since_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS _quack_ontology_candidates (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                proposal JSON NOT NULL,
                evidence JSON NOT NULL,
                confidence FLOAT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                proposed_by TEXT NOT NULL,
                decided_by TEXT,
                decided_at TIMESTAMP
            );";

/// Wraps a `DuckDB` connection for a single workspace.
pub struct WorkspaceDb {
    conn: duckdb::Connection,
    embedding_dimension: u32,
    query_timeout: Duration,
    /// `files/` under the workspace directory, where ingested files are
    /// kept; `None` in memory.
    files_dir: Option<std::path::PathBuf>,
}

impl WorkspaceDb {
    /// Open an in-memory `DuckDB` database (for tests).
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created.
    pub fn open_in_memory(embedding_dimension: u32) -> Result<Self> {
        let conn = duckdb::Connection::open_in_memory()?;
        let db = Self {
            conn,
            embedding_dimension,
            query_timeout: Duration::from_secs(30),
            files_dir: None,
        };
        db.confine_to(None)?;
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
    /// Creates the `_quack_` internal tables if they
    /// do not exist, and reconciles the embedding dimension recorded in
    /// `_quack_meta` with the configured provider.
    ///
    /// # Errors
    ///
    /// Returns an error if the database file cannot be created or opened.
    pub fn open(config: &Config, workspace_id: &str) -> Result<Self> {
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
            files_dir: Some(files_dir),
        };
        db.apply_resource_limits(config)?;
        let workspace_dir = std::fs::canonicalize(config.workspace_dir(workspace_id))?;
        db.confine_to(Some(&workspace_dir))?;
        db.rename_legacy_tables()?;
        db.reconcile_embedding_dimension(configured_dimension, configured_model)
            .or_else(|e| match e {
                // A brand-new workspace has no meta table yet; create it first.
                Error::DuckDb(_) => {
                    db.create_internal_tables()?;
                    db.reconcile_embedding_dimension(configured_dimension, configured_model)
                }
                other => Err(other),
            })?;
        db.create_internal_tables()?;
        Ok(db)
    }

    /// Open a second, reader connection to the same database:
    /// `duckdb::Connection::try_clone` opens a new connection against the
    /// already-open `DatabaseInstance`, so it succeeds even though
    /// `lock_configuration` is set — that lock freezes the *configuration*
    /// (`allowed_directories`, resource limits, ...), which is shared by
    /// every connection to the instance, not the ability to open more of
    /// them. `Connection::open` is not an option here: it would create a
    /// second `DatabaseInstance` and take the file's exclusive lock.
    ///
    /// The clone inherits the confinement and resource limits already
    /// locked in on `self`, so it must never call [`Self::confine_to`] or
    /// [`Self::apply_resource_limits`] again — both `SET`s would fail once
    /// the configuration is locked.
    ///
    /// # Errors
    ///
    /// Returns an error if `DuckDB` cannot open the new connection.
    pub fn try_clone_reader(&self) -> Result<Self> {
        Ok(Self {
            conn: self.conn.try_clone()?,
            embedding_dimension: self.embedding_dimension,
            query_timeout: self.query_timeout,
            files_dir: self.files_dir.clone(),
        })
    }

    /// Whether this connection has any temp tables: only ever the CLI's
    /// piped-stdin table, `ingestion::STDIN_TABLE`, loaded before a
    /// workspace handle's first turn — the agent itself is refused any
    /// statement that would create one (`analysis::tools::gate_statement`),
    /// so none can appear later. `DuckDB` temp tables are connection-local,
    /// so a [`Self::try_clone_reader`] clone would not see one: a caller
    /// building a reader for a workspace handle's lifetime
    /// (`analysis::tools::open_reader`) checks this once, at open, and
    /// reuses the writer instead when it's true.
    ///
    /// # Errors
    ///
    /// Returns an error if the catalog query fails.
    pub fn has_temp_tables(&self) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT count(*) FROM duckdb_tables() WHERE temporary",
            [],
            |row| row.get(0),
        )?;
        Ok(n > 0)
    }

    /// Cap memory and parallelism for every statement on this connection.
    fn apply_resource_limits(&self, config: &Config) -> Result<()> {
        let memory_limit = format!("{}MB", config.analysis.memory_limit_mb);
        self.conn
            .execute("SET memory_limit = ?", duckdb::params![memory_limit])?;
        self.conn.execute(
            "SET threads = ?",
            duckdb::params![i64::from(config.analysis.threads.max(1))],
        )?;
        Ok(())
    }

    /// Confine every statement on this connection to the workspace (design
    /// doc 7.4). `DuckDB`'s file readers, replacement scans (`FROM 'x.csv'`),
    /// `COPY`, `ATTACH`, `INSTALL`, and `LOAD` may touch nothing outside
    /// `allowed_dir` (the workspace directory, whose `files/` holds the
    /// ingested originals), secrets never persist, and the configuration is
    /// then locked so no later statement, agent-written or user-typed, can
    /// widen it or lift the resource limits. Read classification alone does
    /// not cover this: `SELECT * FROM read_text('/etc/passwd')` is a read.
    ///
    /// The allow-list must be set while external access is still enabled,
    /// and the lock must come last; `DuckDB` refuses both in any other order.
    fn confine_to(&self, allowed_dir: Option<&Path>) -> Result<()> {
        match allowed_dir {
            Some(dir) => {
                let dir = dir.to_string_lossy();
                self.conn.execute(
                    "SET allowed_directories = [?]",
                    duckdb::params![dir.as_ref()],
                )?;
            }
            None => {
                self.conn.execute("SET allowed_directories = []", [])?;
            }
        }
        self.conn.execute_batch(
            "SET enable_external_access = false;\n\
             SET allow_persistent_secrets = false;\n\
             SET lock_configuration = true;",
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
    pub fn classify_statement(&self, sql: &str) -> Result<StatementKind> {
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

    /// The statement's parse tree with every constant blanked and the
    /// source offsets dropped, so two statements that differ only in the
    /// values they filter on serialize the same; `None` when `DuckDB`
    /// cannot serialize the statement (anything but a `SELECT` shape).
    ///
    /// # Errors
    ///
    /// Returns an error if the serialization query itself fails.
    pub fn statement_shape(&self, sql: &str) -> Result<Option<String>> {
        let serialized: String = self.conn.query_row(
            "SELECT json_serialize_sql(?::VARCHAR)",
            duckdb::params![sql],
            |row| row.get(0),
        )?;
        let mut parsed: serde_json::Value = serde_json::from_str(&serialized)?;
        let is_error = parsed
            .get("error")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        if is_error {
            return Ok(None);
        }
        blank_constants(&mut parsed);
        Ok(Some(parsed.to_string()))
    }

    /// Names of tables a statement references, as `DuckDB` parsed them, or
    /// `None` when `DuckDB` cannot serialize the statement.
    fn referenced_base_tables(&self, sql: &str) -> Result<Option<Vec<String>>> {
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

    /// Classify a statement a user or client wrote: `_quack_` tables are
    /// refused outright, then the statement is read, write, or invalid.
    ///
    /// # Errors
    ///
    /// Returns `Error::Analysis` when the statement reaches an internal
    /// table, or a storage error when classification fails.
    pub fn classify_user_statement(&self, sql: &str) -> Result<StatementKind> {
        if self.references_internal_table(sql)? {
            return Err(Error::Analysis(String::from(
                "internal tables are not accessible",
            )));
        }
        self.classify_statement(sql)
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
    pub fn references_internal_table(&self, sql: &str) -> Result<bool> {
        match self.referenced_base_tables(sql)? {
            Some(names) => Ok(names.iter().any(|n| is_internal_name(n))),
            None => Ok(mentions_internal_table_token(sql)),
        }
    }

    fn create_internal_tables(&self) -> Result<()> {
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
                title TEXT,
                mime_type TEXT,
                size_bytes BIGINT,
                sha256 TEXT,
                source TEXT,
                ingested_at TIMESTAMP DEFAULT now(),
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                pinned BOOLEAN NOT NULL DEFAULT false,
                chunk_count INTEGER,
                ingested_by TEXT,
                tables JSON
            );
            CREATE TABLE IF NOT EXISTS _quack_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                heading TEXT,
                page INTEGER,
                embedding FLOAT[{dim}],
                token_count INTEGER
            );
            ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS pinned BOOLEAN DEFAULT false;
            ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS title TEXT;
            ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS sha256 TEXT;
            ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS source TEXT;
            ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS chunk_count INTEGER;
            ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS ingested_by TEXT;
            ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS tables JSON;
            ALTER TABLE _quack_chunks ADD COLUMN IF NOT EXISTS heading TEXT;
            ALTER TABLE _quack_chunks ADD COLUMN IF NOT EXISTS page INTEGER;
            CREATE TABLE IF NOT EXISTS _quack_terms (
                chunk_id TEXT NOT NULL,
                term TEXT NOT NULL,
                tf INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS _quack_terms_term_idx ON _quack_terms (term);
            CREATE INDEX IF NOT EXISTS _quack_terms_chunk_idx ON _quack_terms (chunk_id);
            DROP SCHEMA IF EXISTS fts_main__quack_chunks CASCADE;
            CREATE TABLE IF NOT EXISTS _quack_context (
                version INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                edited_by TEXT,
                edited_at TIMESTAMP DEFAULT now()
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
            );
            CREATE TABLE IF NOT EXISTS _quack_audit (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT now(),
                user_id TEXT,
                action TEXT NOT NULL,
                detail JSON
            );"
        );
        self.conn.execute_batch(&sql)?;
        self.conn.execute_batch(ONTOLOGY_DDL)?;
        self.conn.execute_batch(&crate::graph::ddl(dim))?;
        let recorded = self
            .meta("schema_version")?
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        // Version 4 introduced the term index; version 6 changed its tokens
        // (stemming) and version 7 added joined identifier terms, so older
        // workspaces rebuild it on open.
        if recorded < 7 && self.chunk_count()? > 0 {
            tracing::info!("indexing existing chunks for keyword search");
            self.reindex_terms()?;
        }
        self.set_meta("schema_version", WORKSPACE_SCHEMA_VERSION)?;
        self.set_meta("embedding_dimension", &dim.to_string())?;
        Ok(())
    }

    /// Workspaces created before the `_quack_` prefix keep their data.
    fn rename_legacy_tables(&self) -> Result<()> {
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

    fn table_exists(&self, name: &str) -> Result<bool> {
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
    pub fn meta(&self, key: &str) -> Result<Option<String>> {
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

    /// Write a `_quack_meta` entry.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub fn set_meta_public(&self, key: &str, value: &str) -> Result<()> {
        self.set_meta(key, value)
    }

    /// Store a vector in `column` of the row of `table` whose `id` matches.
    /// `table` and `column` are quack's own identifiers, never user input.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn set_vector(&self, table: &str, column: &str, id: &str, embedding: &[f32]) -> Result<()> {
        let sql = format!(
            "UPDATE {} SET {} = ?::{} WHERE id = ?",
            quote_ident(table),
            quote_ident(column),
            self.vector_type()
        );
        self.conn
            .execute(&sql, duckdb::params![format_embedding(embedding), id])?;
        Ok(())
    }

    /// The `FLOAT[N]` type of this workspace's vectors.
    #[must_use]
    pub fn vector_type_public(&self) -> String {
        self.vector_type()
    }

    fn set_meta(&self, key: &str, value: &str) -> Result<()> {
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
    ) -> Result<()> {
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
                    return Err(Error::Config(format!(
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
                    "no chunk embeddings stored; adopting the configured embedding dimension"
                );
                // Chunks ingested without an embedding provider (and their
                // term index) stay; only the vector column changes width.
                // Node label embeddings are recomputed by the next
                // resolution pass, so they are cleared and resized.
                // No stored vector survives the change, so the column is
                // retyped through NULL: DuckDB cannot cast even a NULL
                // FLOAT[4] to FLOAT[8] on its own.
                self.conn.execute_batch(&format!(
                    "ALTER TABLE _quack_chunks ALTER embedding SET DATA TYPE FLOAT[{conf}] USING NULL::FLOAT[{conf}];"
                ))?;
                if self.table_exists("_quack_graph_nodes")? {
                    self.conn.execute_batch(&format!(
                        "ALTER TABLE _quack_graph_nodes ALTER embedding SET DATA TYPE FLOAT[{conf}] USING NULL::FLOAT[{conf}];"
                    ))?;
                }
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

    /// Drop a document's chunks and their term index, and clear its chunk
    /// count: what a failed processing pass leaves behind must not stay
    /// searchable (issue #52).
    ///
    /// # Errors
    ///
    /// Returns an error if a delete fails.
    pub fn discard_chunks(&self, document_id: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM _quack_graph_extracted WHERE chunk_id IN (SELECT id FROM _quack_chunks WHERE document_id = ?)",
            duckdb::params![document_id],
        )?;
        self.conn.execute(
            "DELETE FROM _quack_terms WHERE chunk_id IN (SELECT id FROM _quack_chunks WHERE document_id = ?)",
            duckdb::params![document_id],
        )?;
        self.conn.execute(
            "DELETE FROM _quack_chunks WHERE document_id = ?",
            duckdb::params![document_id],
        )?;
        self.conn.execute(
            "UPDATE _quack_documents SET chunk_count = NULL WHERE id = ?",
            duckdb::params![document_id],
        )?;
        Ok(())
    }

    /// Register a document row. Ingestion computes the hash and checks for
    /// a duplicate first; see `ingestion::register_document`.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub fn insert_document(&self, doc: &NewDocument<'_>) -> Result<()> {
        let size = i64::try_from(doc.size_bytes)
            .map_err(|_| Error::Ingestion("file size overflow".into()))?;

        self.conn.execute(
            "INSERT INTO _quack_documents (id, filename, title, mime_type, size_bytes, sha256, source, status, ingested_by) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            duckdb::params![
                doc.id,
                doc.filename,
                doc.title,
                doc.mime_type,
                size,
                doc.sha256,
                doc.source.as_str(),
                doc.status,
                doc.ingested_by,
            ],
        )?;
        Ok(())
    }

    /// The document whose content hashes to `sha256`, if one was ingested
    /// and did not fail. Re-uploads of identical bytes are skipped through
    /// this lookup; a failed document is not a match so it can be retried.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn document_by_sha256(&self, sha256: &str) -> Result<Option<DocumentInfo>> {
        let sql = format!(
            "{DOCUMENT_SELECT} WHERE sha256 = ? AND status <> 'error' ORDER BY ingested_at, id LIMIT 1"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(duckdb::params![sha256])?;
        match rows.next()? {
            Some(row) => Ok(Some(document_from_row(row)?)),
            None => Ok(None),
        }
    }

    /// The live (non-error) document that loaded `table`, if any: one
    /// document owns a table (issue #51).
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn table_owner(&self, table: &str) -> Result<Option<DocumentInfo>> {
        let sql = format!(
            "{DOCUMENT_SELECT} WHERE status <> 'error' AND tables IS NOT NULL \
             AND list_contains(CAST(tables AS VARCHAR[]), ?) ORDER BY ingested_at, id LIMIT 1"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(duckdb::params![table])?;
        match rows.next()? {
            Some(row) => Ok(Some(document_from_row(row)?)),
            None => Ok(None),
        }
    }

    /// Whether what a ready document loaded is still there: every table
    /// it recorded exists, and a chunked document still has chunks. A
    /// document still queued or processing counts as intact.
    ///
    /// # Errors
    ///
    /// Returns an error if a query fails.
    pub fn document_intact(&self, doc: &DocumentInfo) -> Result<bool> {
        if doc.status != "ready" {
            return Ok(true);
        }
        if let Some(tables) = &doc.tables
            && !tables.is_empty()
        {
            let existing = self.list_tables()?;
            return Ok(tables.iter().all(|t| existing.contains(t)));
        }
        if doc.chunk_count.is_some_and(|n| n > 0) {
            let chunks: i64 = self.conn.query_row(
                "SELECT count(*) FROM _quack_chunks WHERE document_id = ?",
                duckdb::params![doc.id],
                |r| r.get(0),
            )?;
            return Ok(chunks > 0);
        }
        Ok(true)
    }

    /// Fail every document still `queued` or `processing`: called once
    /// when a server opens the workspace, since upload bytes live only in
    /// the memory of the process that took them, so nothing can finish a
    /// row a restart interrupted (issue #51). Returns how many.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn fail_stale_uploads(&self) -> Result<usize> {
        let changed = self.conn.execute(
            "UPDATE _quack_documents SET status = 'error', \
             error_message = 'interrupted by a restart before it was processed; upload it again' \
             WHERE status IN ('queued', 'processing')",
            [],
        )?;
        Ok(changed)
    }

    /// Set the title parsed from the content, when the caller gave none.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn set_document_title_if_empty(&self, id: &str, title: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE _quack_documents SET title = ? WHERE id = ? AND title IS NULL",
            duckdb::params![title, id],
        )?;
        Ok(())
    }

    /// Record the tables a structured document loaded into, so deleting
    /// the document drops them.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn set_document_tables(&self, id: &str, tables: &[String]) -> Result<()> {
        let json = serde_json::to_string(tables)?;
        self.conn.execute(
            "UPDATE _quack_documents SET tables = ? WHERE id = ?",
            duckdb::params![json, id],
        )?;
        Ok(())
    }

    /// Record how many chunks a processed document produced.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn set_document_chunk_count(&self, id: &str, count: u32) -> Result<()> {
        self.conn.execute(
            "UPDATE _quack_documents SET chunk_count = ? WHERE id = ?",
            duckdb::params![count, id],
        )?;
        Ok(())
    }

    /// Update a document's status and clear any earlier error.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn update_document_status(&self, id: &str, status: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE _quack_documents SET status = ?, error_message = NULL WHERE id = ?",
            duckdb::params![status, id],
        )?;
        Ok(())
    }

    /// Mark a document as failed with the reason.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn mark_document_error(&self, id: &str, message: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE _quack_documents SET status = 'error', error_message = ? WHERE id = ?",
            duckdb::params![message, id],
        )?;
        Ok(())
    }

    /// One document by id.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn document(&self, id: &str) -> Result<Option<DocumentInfo>> {
        let sql = format!("{DOCUMENT_SELECT} WHERE id = ?");
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(duckdb::params![id])?;
        match rows.next()? {
            Some(row) => Ok(Some(document_from_row(row)?)),
            None => Ok(None),
        }
    }

    /// Remove a document with its chunks and term index. The tables it
    /// loaded into are dropped too: those recorded on the row, or for rows
    /// from before that was recorded, `fallback_table`. Graph nodes and
    /// edges whose only provenance was the document or its tables go with
    /// it (issue #43), as do its files under `files/`. Returns whether the
    /// document existed.
    ///
    /// # Errors
    ///
    /// Returns an error if any delete fails.
    pub fn delete_document(&self, id: &str, fallback_table: Option<&str>) -> Result<bool> {
        let Some(doc) = self.document(id)? else {
            return Ok(false);
        };
        let tables: Vec<String> = match doc.tables {
            Some(tables) => tables,
            None => fallback_table.map(str::to_owned).into_iter().collect(),
        };
        self.forget_graph_provenance(id, &tables)?;
        self.conn.execute(
            "DELETE FROM _quack_graph_extracted WHERE chunk_id IN (SELECT id FROM _quack_chunks WHERE document_id = ?)",
            duckdb::params![id],
        )?;
        self.conn.execute(
            "DELETE FROM _quack_terms WHERE chunk_id IN (SELECT id FROM _quack_chunks WHERE document_id = ?)",
            duckdb::params![id],
        )?;
        self.conn.execute(
            "DELETE FROM _quack_chunks WHERE document_id = ?",
            duckdb::params![id],
        )?;
        self.conn.execute(
            "DELETE FROM _quack_documents WHERE id = ?",
            duckdb::params![id],
        )?;
        for table in &tables {
            self.conn
                .execute_batch(&format!("DROP TABLE IF EXISTS {}", quote_ident(table)))?;
        }
        self.remove_document_files(&doc.filename, &tables);
        Ok(true)
    }

    /// Drop the provenance a document and its tables gave the graph, then
    /// the nodes and edges left without any provenance at all (an edge
    /// whose endpoint goes falls with it, as `graph::store::delete_nodes`
    /// does).
    fn forget_graph_provenance(&self, document_id: &str, tables: &[String]) -> Result<()> {
        let table_list = sql_text_list(tables);
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT subject_id FROM _quack_provenance \
             WHERE document_id = ? OR list_contains(?::VARCHAR[], table_name)",
        )?;
        let mut rows = stmt.query(duckdb::params![document_id, table_list])?;
        let mut touched: Vec<String> = Vec::new();
        while let Some(row) = rows.next()? {
            touched.push(row.get(0)?);
        }
        drop(rows);
        drop(stmt);
        if touched.is_empty() {
            return Ok(());
        }
        self.conn.execute(
            "DELETE FROM _quack_provenance \
             WHERE document_id = ? OR list_contains(?::VARCHAR[], table_name)",
            duckdb::params![document_id, table_list],
        )?;
        let touched_list = sql_text_list(&touched);
        let orphan_nodes = self.conn.execute(
            "DELETE FROM _quack_graph_nodes WHERE list_contains(?::VARCHAR[], id) \
             AND NOT EXISTS (SELECT 1 FROM _quack_provenance p WHERE p.subject_id = _quack_graph_nodes.id)",
            duckdb::params![touched_list],
        )?;
        let orphan_edges = self.conn.execute(
            "DELETE FROM _quack_graph_edges WHERE (list_contains(?::VARCHAR[], id) \
             AND NOT EXISTS (SELECT 1 FROM _quack_provenance p WHERE p.subject_id = _quack_graph_edges.id)) \
             OR NOT EXISTS (SELECT 1 FROM _quack_graph_nodes n WHERE n.id = source_node_id) \
             OR NOT EXISTS (SELECT 1 FROM _quack_graph_nodes n WHERE n.id = target_node_id)",
            duckdb::params![touched_list],
        )?;
        self.conn.execute_batch(
            "DELETE FROM _quack_provenance WHERE NOT EXISTS \
               (SELECT 1 FROM _quack_graph_nodes n WHERE n.id = subject_id) \
             AND NOT EXISTS (SELECT 1 FROM _quack_graph_edges e WHERE e.id = subject_id); \
             DELETE FROM _quack_graph_merges WHERE NOT EXISTS \
               (SELECT 1 FROM _quack_graph_nodes n WHERE n.id = keep_node_id) \
             OR NOT EXISTS (SELECT 1 FROM _quack_graph_nodes n WHERE n.id = drop_node_id);",
        )?;
        tracing::info!(
            document_id,
            orphan_nodes,
            orphan_edges,
            "removed graph rows that only the deleted document supported"
        );
        Ok(())
    }

    /// Remove what ingestion wrote under `files/` for a document: the file
    /// itself and, for workbooks and imports, one CSV per table. A missing
    /// file is fine; any other failure is logged, since the rows are gone.
    fn remove_document_files(&self, filename: &str, tables: &[String]) {
        let Some(files_dir) = &self.files_dir else {
            return;
        };
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        if let Some(name) = Path::new(filename).file_name() {
            candidates.push(files_dir.join(name));
        }
        for table in tables {
            candidates.push(files_dir.join(format!("{table}.csv")));
        }
        for path in candidates {
            match std::fs::remove_file(&path) {
                Ok(()) => tracing::debug!(path = %path.display(), "removed an ingested file"),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "could not remove an ingested file");
                }
            }
        }
    }

    /// Insert a text chunk, optionally with an embedding vector, and index
    /// its terms for keyword search.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub fn insert_chunk(&self, chunk: &NewChunk<'_>) -> Result<()> {
        let page = chunk.page.map(i64::from);
        let terms = term_frequencies(chunk.content, chunk.heading);
        let length = term_count(&terms);
        match chunk.embedding {
            Some(emb) => {
                let sql = format!(
                    "INSERT INTO _quack_chunks (id, document_id, chunk_index, content, heading, page, token_count, embedding) \
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?::{})",
                    self.vector_type()
                );
                self.conn.execute(
                    &sql,
                    duckdb::params![
                        chunk.id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.heading,
                        page,
                        length,
                        format_embedding(emb)
                    ],
                )?;
            }
            None => {
                self.conn.execute(
                    "INSERT INTO _quack_chunks (id, document_id, chunk_index, content, heading, page, token_count) \
                     VALUES (?, ?, ?, ?, ?, ?, ?)",
                    duckdb::params![
                        chunk.id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.heading,
                        page,
                        length
                    ],
                )?;
            }
        }
        self.insert_terms(chunk.id, &terms)?;
        Ok(())
    }

    fn insert_terms(&self, chunk_id: &str, terms: &[(String, u32)]) -> Result<()> {
        if terms.is_empty() {
            return Ok(());
        }
        let mut appender = self.conn.appender("_quack_terms")?;
        for (term, tf) in terms {
            appender.append_row(duckdb::params![chunk_id, term, i64::from(*tf)])?;
        }
        appender.flush()?;
        Ok(())
    }

    fn chunk_count(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT count(*) FROM _quack_chunks", [], |row| row.get(0))?)
    }

    /// Rebuild the keyword index from every stored chunk.
    ///
    /// # Errors
    ///
    /// Returns an error if reading chunks or writing terms fails.
    pub fn reindex_terms(&self) -> Result<()> {
        self.conn.execute("DELETE FROM _quack_terms", [])?;
        let mut stmt = self
            .conn
            .prepare("SELECT id, content, heading FROM _quack_chunks")?;
        let mut rows = stmt.query([])?;
        let mut chunks: Vec<(String, String, Option<String>)> = Vec::new();
        while let Some(row) = rows.next()? {
            chunks.push((row.get(0)?, row.get(1)?, row.get(2)?));
        }
        for (id, content, heading) in &chunks {
            let terms = term_frequencies(content, heading.as_deref());
            self.conn.execute(
                "UPDATE _quack_chunks SET token_count = ? WHERE id = ?",
                duckdb::params![term_count(&terms), id],
            )?;
            self.insert_terms(id, &terms)?;
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
    ) -> Result<()> {
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

    /// Keyword search: BM25 over the terms quack indexed at ingest, scored in
    /// SQL from `_quack_terms` and each chunk's term count. Needs no
    /// extension and no rebuild step.
    ///
    /// A `"..."` quoted phrase in `query` is an adjacency requirement: `_quack_terms`
    /// carries no positions (`docs/migrations.md`), so BM25 still ranks by the
    /// phrase's own tokens, but the candidates are over-fetched and then
    /// post-filtered to those whose content or heading contains the phrase as a
    /// case-insensitive, whitespace-normalized substring. A phrase that matches no
    /// candidate returns an empty result rather than falling back to the
    /// unfiltered ranking. Unbalanced quotes are treated as ordinary text.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn search_keyword_chunks(
        &self,
        query: &str,
        top_k: u32,
        scope: &ChunkScope,
    ) -> Result<Vec<ChunkSearchResult>> {
        if scope.is_empty() {
            return Ok(Vec::new());
        }
        let terms: Vec<String> = term_frequencies(query, None)
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let phrases = extract_phrases(query);
        let fetch_k = if phrases.is_empty() {
            top_k
        } else {
            top_k
                .saturating_mul(PHRASE_OVER_FETCH)
                .min(PHRASE_CANDIDATE_CAP)
                .max(top_k)
        };
        // Quoted, so a token such as `null` stays a word and not a NULL
        // element (issue #62).
        let term_list = sql_text_list(&terms);
        let filter = scope.sql();
        let sql = format!(
            "WITH q AS (SELECT DISTINCT unnest(?::VARCHAR[]) AS term), \
                  stats AS (SELECT count(*) AS n, avg(token_count) AS avgdl \
                            FROM _quack_chunks WHERE token_count > 0), \
                  df AS (SELECT t.term, count(DISTINCT t.chunk_id) AS df \
                         FROM _quack_terms t JOIN q ON q.term = t.term GROUP BY t.term), \
                  scored AS (SELECT t.chunk_id, \
                         sum(ln(1 + (s.n - d.df + 0.5) / (d.df + 0.5)) \
                             * (t.tf * ({BM25_K1} + 1)) \
                             / (t.tf + {BM25_K1} * (1 - {BM25_B} + {BM25_B} * ch.token_count / s.avgdl))) AS score \
                         FROM _quack_terms t \
                         JOIN df d ON d.term = t.term \
                         JOIN _quack_chunks ch ON ch.id = t.chunk_id, stats s \
                         GROUP BY t.chunk_id) \
             SELECT c.id, c.content, c.document_id, c.chunk_index, d.filename, c.heading, c.page, sc.score \
             FROM scored sc \
             JOIN _quack_chunks c ON c.id = sc.chunk_id \
             JOIN _quack_documents d ON d.id = c.document_id \
             WHERE sc.score > 0 AND d.status = 'ready'{filter} \
             ORDER BY sc.score DESC, c.chunk_index ASC \
             LIMIT ?"
        );
        let limit = i64::from(fetch_k);
        let mut stmt = self.conn.prepare(&sql)?;
        let mut params: Vec<&dyn duckdb::ToSql> = Vec::with_capacity(scope.len().saturating_add(2));
        params.push(&term_list);
        scope.bind(&mut params);
        params.push(&limit);
        let mut rows = stmt.query(params.as_slice())?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            results.push(chunk_from_row(row, 7)?);
        }
        if phrases.is_empty() {
            return Ok(results);
        }
        filter_by_phrases(&mut results, &phrases);
        results.truncate(usize::try_from(top_k).unwrap_or(usize::MAX));
        Ok(results)
    }

    /// Hybrid retrieval: vector and keyword rankings fused with reciprocal
    /// rank fusion (`score = sum over rankings of 1 / (rrf_k + rank)`).
    ///
    /// A quoted phrase in `query_text` filters the fused result the same way
    /// [`Self::search_keyword_chunks`] filters its own: the vector leg knows
    /// nothing about phrases, so both legs are over-fetched and the phrase
    /// substring filter runs once on the fused ranking.
    ///
    /// # Errors
    ///
    /// Returns an error if either search fails.
    pub fn search_hybrid_chunks(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: u32,
        rrf_k: u32,
        scope: &ChunkScope,
    ) -> Result<Vec<ChunkSearchResult>> {
        let phrases = extract_phrases(query_text);
        let candidates = top_k.saturating_mul(2).max(1);
        let fuse_k = if phrases.is_empty() {
            top_k
        } else {
            candidates
                .saturating_mul(PHRASE_OVER_FETCH)
                .min(PHRASE_CANDIDATE_CAP)
                .max(top_k)
        };
        let vector = self.search_similar_chunks(query_embedding, fuse_k, scope)?;
        let keyword = self.search_keyword_chunks(query_text, fuse_k, scope)?;
        let mut fused = fuse_rankings(vector, keyword, fuse_k, rrf_k);
        if phrases.is_empty() {
            return Ok(fused);
        }
        filter_by_phrases(&mut fused, &phrases);
        fused.truncate(usize::try_from(top_k).unwrap_or(usize::MAX));
        Ok(fused)
    }

    /// Search for the most similar chunks to a query embedding. `score` is
    /// `1 / (1 + cosine distance)`.
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
        scope: &ChunkScope,
    ) -> Result<Vec<ChunkSearchResult>> {
        if scope.is_empty() {
            return Ok(Vec::new());
        }
        let filter = scope.sql();
        let sql = format!(
            "SELECT c.id, c.content, c.document_id, c.chunk_index, d.filename, c.heading, c.page, \
                    1.0 / (1.0 + array_cosine_distance(c.embedding, ?::{})) AS score \
             FROM _quack_chunks c \
             JOIN _quack_documents d ON d.id = c.document_id \
             WHERE c.embedding IS NOT NULL AND d.status = 'ready'{filter} \
             ORDER BY score DESC \
             LIMIT ?",
            self.vector_type()
        );

        let query_literal = format_embedding(query_embedding);
        let limit = i64::from(top_k);
        let mut stmt = self.conn.prepare(&sql)?;
        let mut params: Vec<&dyn duckdb::ToSql> = Vec::with_capacity(scope.len().saturating_add(2));
        params.push(&query_literal);
        scope.bind(&mut params);
        params.push(&limit);
        let mut rows = stmt.query(params.as_slice())?;
        let mut results = Vec::new();

        while let Some(row) = rows.next()? {
            results.push(chunk_from_row(row, 7)?);
        }

        Ok(results)
    }

    /// Chunks by id, in the order given, with the citation metadata a
    /// search hit carries (score 1).
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn chunks_by_ids(&self, ids: &[String]) -> Result<Vec<ChunkSearchResult>> {
        let mut out = Vec::with_capacity(ids.len());
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.content, c.document_id, c.chunk_index, d.filename, c.heading, c.page, 1.0 \
             FROM _quack_chunks c JOIN _quack_documents d ON d.id = c.document_id WHERE c.id = ?",
        )?;
        for id in ids {
            let mut rows = stmt.query(duckdb::params![id])?;
            if let Some(row) = rows.next()? {
                out.push(chunk_from_row(row, 7)?);
            }
        }
        Ok(out)
    }

    /// Pin or unpin a document. Pinned documents are injected in full into
    /// the system prompt.
    ///
    /// # Errors
    ///
    /// Returns an error if the document does not exist or the update fails.
    pub fn set_document_pinned(&self, document_id: &str, pinned: bool) -> Result<()> {
        let changed = self.conn.execute(
            "UPDATE _quack_documents SET pinned = ? WHERE id = ?",
            duckdb::params![pinned, document_id],
        )?;
        if changed == 0 {
            return Err(Error::Ingestion(format!(
                "document '{document_id}' does not exist"
            )));
        }
        Ok(())
    }

    /// Full text of every pinned document, in chunk order.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn pinned_documents(&self) -> Result<Vec<(DocumentInfo, String)>> {
        let mut out = Vec::new();
        for doc in self.list_documents()?.into_iter().filter(|d| d.pinned) {
            let mut stmt = self.conn.prepare(
                "SELECT content FROM _quack_chunks WHERE document_id = ? ORDER BY chunk_index",
            )?;
            let mut rows = stmt.query(duckdb::params![doc.id])?;
            let mut parts: Vec<String> = Vec::new();
            while let Some(row) = rows.next()? {
                parts.push(row.get(0)?);
            }
            out.push((doc, parts.join("\n")));
        }
        Ok(out)
    }

    /// Execute an arbitrary SQL statement and return every row.
    ///
    /// The owner's tool (`quack -q`): nothing bounds the result. Every
    /// other caller uses [`Self::execute_query_capped`].
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid or execution fails.
    pub fn execute_query(&self, sql: &str) -> Result<QueryResults> {
        Ok(self.read_rows(sql, None)?.results)
    }

    /// Execute a statement keeping at most `max_rows` rows. Rows past the
    /// cap are counted, never converted, so a `SELECT *` over a large
    /// table costs `max_rows` values of memory on the Rust side.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid or execution fails.
    pub fn execute_query_capped(&self, sql: &str, max_rows: u32) -> Result<CappedResults> {
        self.read_rows(sql, Some(max_rows as usize))
    }

    fn read_rows(&self, sql: &str, keep: Option<usize>) -> Result<CappedResults> {
        let _guard = self.arm_timeout();
        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query([])?;

        let empty = CappedResults {
            results: QueryResults {
                columns: Vec::new(),
                rows: Vec::new(),
            },
            total_rows: 0,
        };
        let (columns, column_count) = {
            let Some(stmt_ref) = rows.as_ref() else {
                return Ok(empty);
            };
            let count = stmt_ref.column_count();
            if count == 0 {
                return Ok(empty);
            }
            (stmt_ref.column_names(), count)
        };

        let mut result_rows: Vec<Vec<serde_json::Value>> = Vec::new();
        let mut total_rows: usize = 0;
        while let Some(row) = rows.next()? {
            total_rows = total_rows.saturating_add(1);
            if keep.is_some_and(|keep| result_rows.len() >= keep) {
                continue;
            }
            let mut values = Vec::with_capacity(column_count);
            for i in 0..column_count {
                values.push(extract_value(row, i));
            }
            result_rows.push(values);
        }

        Ok(CappedResults {
            results: QueryResults {
                columns,
                rows: result_rows,
            },
            total_rows,
        })
    }

    /// Execute a SQL statement that does not return rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid or execution fails.
    pub fn execute_statement(&self, sql: &str) -> Result<()> {
        self.execute_with_params(sql, [])
    }

    /// Execute a parameterized statement that does not return rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid or execution fails.
    pub fn execute_with_params<P: duckdb::Params>(&self, sql: &str, params: P) -> Result<()> {
        let _guard = self.arm_timeout();
        self.conn.execute(sql, params)?;
        Ok(())
    }

    /// Run `f` under the statement watchdog: if it is still going after
    /// the configured query timeout, the connection is interrupted and
    /// the statement inside fails. For multi-statement work (a graph
    /// batch) that would otherwise run unbounded.
    ///
    /// # Errors
    ///
    /// Returns `f`'s error, including the interruption.
    pub fn under_timeout<R>(&self, f: impl FnOnce(&Self) -> Result<R>) -> Result<R> {
        let _guard = self.arm_timeout();
        f(self)
    }

    /// Run `f` so that `canceller` can interrupt its statements while it
    /// runs; one cancelled before `f` starts fails without running it.
    ///
    /// # Errors
    ///
    /// Returns `f`'s error (an interrupted statement's included), or
    /// [`Error::Analysis`] when the work was cancelled before it started.
    pub fn cancellable<R>(
        &self,
        canceller: &QueryCanceller,
        f: impl FnOnce(&Self) -> Result<R>,
    ) -> Result<R> {
        {
            let mut slot = canceller.slot();
            if slot.cancelled {
                return Err(Error::Analysis(String::from("cancelled")));
            }
            slot.running = Some(self.conn.interrupt_handle());
        }
        let result = f(self);
        canceller.slot().running = None;
        result
    }

    /// Run `f`'s reads inside `BEGIN TRANSACTION READ ONLY`, scoped to `f`
    /// alone: reader connections use this for every query so no transaction
    /// outlives one piece of work, pinning an old snapshot and blocking
    /// checkpointing. `DuckDB`'s `unchecked_transaction` cannot pass the
    /// `READ ONLY` modifier, so the statements are issued directly; a write
    /// `f` attempts fails with `DuckDB`'s own error, before it reaches the
    /// write-gating the agent and API already apply. Safe on the writer
    /// connection too for a statement already known to be a read (`run_sql`
    /// does this); only a statement that might write must stay outside it.
    /// Not re-entrant: `f` must not call this again on the same connection
    /// (`DuckDB` rejects a nested `BEGIN`), and it never will through
    /// today's callers, all of which run one statement or a bounded set of
    /// them without transaction control of their own.
    ///
    /// # Errors
    ///
    /// Returns `f`'s error (the transaction is rolled back), or the error
    /// from beginning or committing the transaction itself.
    pub fn read_only<R>(&self, f: impl FnOnce(&Self) -> Result<R>) -> Result<R> {
        let mut guard = ReadOnlyGuard::begin(&self.conn)?;
        let value = f(self)?;
        guard.commit()?;
        Ok(value)
    }

    /// Run `f`'s writes as one transaction: committed when `f` returns
    /// `Ok`, rolled back when it errors. A document's chunks, or a batch
    /// of embeddings, then land in one commit instead of one per row. Not
    /// re-entrant, like [`Self::read_only`].
    ///
    /// # Errors
    ///
    /// Returns `f`'s error, or the error from beginning or committing the
    /// transaction itself.
    pub fn write_transaction<R>(&self, f: impl FnOnce(&Self) -> Result<R>) -> Result<R> {
        let tx = self.conn.unchecked_transaction()?;
        let value = f(self)?;
        tx.commit()?;
        Ok(value)
    }

    /// Start a watchdog that interrupts the connection if the statement runs
    /// past the configured timeout. Dropping the guard disarms it.
    fn arm_timeout(&self) -> TimeoutGuard {
        let (disarm, armed) = std::sync::mpsc::channel::<()>();
        let handle = self.conn.interrupt_handle();
        let timeout = self.query_timeout;
        // The watchdog sleeps on the channel: dropping the guard closes it
        // and wakes the thread at once, so nothing polls (issue #62).
        std::thread::spawn(move || {
            if armed.recv_timeout(timeout) == Err(std::sync::mpsc::RecvTimeoutError::Timeout) {
                tracing::warn!(?timeout, "statement exceeded timeout; interrupting");
                handle.interrupt();
            }
        });
        TimeoutGuard { _disarm: disarm }
    }

    /// List all user-created tables in the workspace (excludes internal tables).
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_tables(&self) -> Result<Vec<String>> {
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
    pub fn describe_table(&self, table_name: &str) -> Result<TableDescription> {
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
        let row_count = self.count_rows(table_name)?;

        Ok(TableDescription {
            table_name: table_name.to_owned(),
            columns,
            row_count,
            sample_rows: sample,
        })
    }

    /// Exact row count of one table.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist or the query fails.
    pub fn count_rows(&self, table_name: &str) -> Result<i64> {
        let sql = format!("SELECT count(*) FROM {}", quote_ident(table_name));
        let count: i64 = self.conn.query_row(&sql, [], |row| row.get(0))?;
        Ok(count)
    }

    /// The version of the `DuckDB` library compiled into this binary, such as
    /// `v1.5.0`; the system prompt pins its dialect notes to it.
    ///
    /// # Errors
    ///
    /// Returns an error if the version query fails.
    pub fn duckdb_version(&self) -> Result<String> {
        let version: String =
            self.conn
                .query_row("SELECT library_version FROM pragma_version()", [], |row| {
                    row.get(0)
                })?;
        Ok(version)
    }

    /// List all ingested documents with their status.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_documents(&self) -> Result<Vec<DocumentInfo>> {
        let sql = format!("{DOCUMENT_SELECT} ORDER BY ingested_at DESC, id DESC");
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        let mut docs = Vec::new();
        while let Some(row) = rows.next()? {
            docs.push(document_from_row(row)?);
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
    /// Exact row count at describe time.
    pub row_count: i64,
    pub sample_rows: QueryResults,
}

/// Document metadata row.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DocumentInfo {
    pub id: String,
    pub filename: String,
    /// Given at upload or parsed from the content (first heading, HTML
    /// title); `None` when neither exists.
    pub title: Option<String>,
    pub mime_type: Option<String>,
    pub size_bytes: Option<i64>,
    /// Lowercase hex SHA-256 of the bytes; `None` only for rows written
    /// before the column existed.
    pub sha256: Option<String>,
    pub source: DocumentSource,
    /// `queued`, `processing`, `ready`, or `error`.
    pub status: String,
    pub error_message: Option<String>,
    pub pinned: bool,
    /// Chunks stored once processed; `None` until then and for tables.
    pub chunk_count: Option<i64>,
    /// Server user who uploaded it; `None` from the CLI.
    pub ingested_by: Option<String>,
    /// Tables a structured document loaded into; `None` until processed
    /// and for rows written before this was recorded.
    pub tables: Option<Vec<String>>,
    pub ingested_at: String,
}

impl DocumentInfo {
    /// The title when one exists, else the filename.
    #[must_use]
    pub fn display_name(&self) -> &str {
        self.title.as_deref().unwrap_or(&self.filename)
    }
}

/// How a document reached the workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentSource {
    /// A file sent through the API or web UI.
    Upload,
    /// Text pasted through the API or web UI.
    Paste,
    /// A file path given to the CLI or terminal.
    Path,
    /// Bytes piped into the CLI.
    Stdin,
    /// Rows pulled from an external database or a URL (`quack import`).
    Import,
}

impl DocumentSource {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Upload => "upload",
            Self::Paste => "paste",
            Self::Path => "path",
            Self::Stdin => "stdin",
            Self::Import => "import",
        }
    }

    fn from_column(value: Option<&str>) -> Self {
        match value {
            Some("paste") => Self::Paste,
            Some("path") => Self::Path,
            Some("stdin") => Self::Stdin,
            Some("import") => Self::Import,
            _ => Self::Upload,
        }
    }
}

impl std::fmt::Display for DocumentSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A document row to insert.
#[derive(Debug, Clone, Copy)]
pub struct NewDocument<'a> {
    pub id: &'a str,
    pub filename: &'a str,
    pub title: Option<&'a str>,
    pub mime_type: &'a str,
    pub size_bytes: usize,
    pub sha256: &'a str,
    pub source: DocumentSource,
    pub status: &'a str,
    pub ingested_by: Option<&'a str>,
}

impl<'a> NewDocument<'a> {
    /// A `queued` upload with no title, hash, or uploader; the fields are
    /// public for the rest.
    #[must_use]
    pub fn new(id: &'a str, filename: &'a str, mime_type: &'a str, size_bytes: usize) -> Self {
        Self {
            id,
            filename,
            title: None,
            mime_type,
            size_bytes,
            sha256: "",
            source: DocumentSource::Upload,
            status: "queued",
            ingested_by: None,
        }
    }

    #[must_use]
    pub fn with_status(mut self, status: &'a str) -> Self {
        self.status = status;
        self
    }
}

const DOCUMENT_SELECT: &str = "SELECT id, filename, mime_type, size_bytes, status, error_message, \
     COALESCE(pinned, false), CAST(ingested_at AS VARCHAR), title, sha256, source, chunk_count, \
     ingested_by, CAST(tables AS VARCHAR) FROM _quack_documents";

fn document_from_row(row: &duckdb::Row<'_>) -> duckdb::Result<DocumentInfo> {
    Ok(DocumentInfo {
        id: row.get(0)?,
        filename: row.get(1)?,
        mime_type: row.get(2)?,
        size_bytes: row.get(3)?,
        status: row.get(4)?,
        error_message: row.get(5)?,
        pinned: row.get(6)?,
        ingested_at: row.get(7)?,
        title: row.get(8)?,
        sha256: row.get(9)?,
        source: DocumentSource::from_column(row.get::<_, Option<String>>(10)?.as_deref()),
        chunk_count: row.get(11)?,
        ingested_by: row.get(12)?,
        tables: row
            .get::<_, Option<String>>(13)?
            .and_then(|json| serde_json::from_str(&json).ok()),
    })
}

/// A chunk to store.
#[derive(Debug, Clone, Copy)]
pub struct NewChunk<'a> {
    pub id: &'a str,
    pub document_id: &'a str,
    pub chunk_index: u32,
    pub content: &'a str,
    pub heading: Option<&'a str>,
    pub page: Option<u32>,
    pub embedding: Option<&'a [f32]>,
}

/// A chunk returned from retrieval, with what a citation needs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChunkSearchResult {
    pub id: String,
    pub content: String,
    pub document_id: String,
    pub chunk_index: u32,
    pub filename: String,
    pub heading: Option<String>,
    pub page: Option<u32>,
    /// Higher is better. Vector-only: `1 / (1 + distance)`; keyword-only:
    /// BM25; hybrid: reciprocal rank fusion.
    pub score: f64,
}

fn chunk_from_row(row: &duckdb::Row<'_>, score_idx: usize) -> duckdb::Result<ChunkSearchResult> {
    let page: Option<i64> = row.get(6)?;
    Ok(ChunkSearchResult {
        id: row.get(0)?,
        content: row.get(1)?,
        document_id: row.get(2)?,
        chunk_index: row.get(3)?,
        filename: row.get(4)?,
        heading: row.get(5)?,
        page: page.and_then(|p| u32::try_from(p).ok()),
        score: row.get(score_idx)?,
    })
}

/// Punctuation that joins alphanumeric runs into one identifier (`POL-8841`,
/// `v1.2.3`, `ns/part:7`) without introducing whitespace.
fn is_identifier_joiner(c: char) -> bool {
    matches!(c, '-' | '.' | '_' | '/' | ':')
}

/// Lowercased alphanumeric runs, stemmed; the same rule indexes chunks and
/// parses queries. A run joined by identifier punctuation with no
/// whitespace (`POL-8841`, `v1.2.3`, `ABC_123`, `ns/part:7`) additionally
/// indexes its punctuation-stripped, lowercased, unstemmed form (`pol8841`)
/// alongside the split, stemmed pieces (`pol`, `8841`), so the query
/// `POL-8841` matches a document containing that exact identifier ahead of
/// one that merely contains `pol` and `8841` apart. Because the joined form
/// is derived the same way on both sides, a bare run like `pol8841` in text
/// is also found by the query `POL-8841` — desirable, since both spell the
/// same identifier. Ordinary prose has no joiner in a run, so it tokenizes
/// exactly as before.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    static STEMMER: std::sync::LazyLock<rust_stemmers::Stemmer> = std::sync::LazyLock::new(|| {
        rust_stemmers::Stemmer::create(rust_stemmers::Algorithm::English)
    });
    let mut terms = Vec::new();
    for run in text.split(|c: char| !(c.is_alphanumeric() || is_identifier_joiner(c))) {
        let run = run.trim_matches(|c: char| !c.is_alphanumeric());
        if run.is_empty() {
            continue;
        }
        for token in run.split(|c: char| !c.is_alphanumeric()) {
            if !token.is_empty() {
                terms.push(STEMMER.stem(&token.to_lowercase()).into_owned());
            }
        }
        if run.contains(is_identifier_joiner) {
            let alnum: String = run.chars().filter(|c| c.is_alphanumeric()).collect();
            terms.push(alnum.to_lowercase());
        }
    }
    terms
}

/// Quoted phrases from a keyword query: each `"..."` pair is taken as an
/// exact adjacency requirement. An odd number of `"` characters is an
/// unbalanced quote, so the whole query is left as ordinary text instead of
/// guessing which quote was meant to close.
fn extract_phrases(query: &str) -> Vec<String> {
    if !query.matches('"').count().is_multiple_of(2) {
        return Vec::new();
    }
    query
        .split('"')
        .enumerate()
        .filter_map(|(i, s)| (i % 2 == 1).then_some(s.trim()))
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Collapse runs of whitespace to a single space and trim the ends, so
/// phrase matching does not care whether a chunk wrapped the phrase across a
/// line break.
fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether `phrase` occurs in `text` as a case-insensitive,
/// whitespace-normalized substring.
fn contains_phrase(text: &str, phrase: &str) -> bool {
    normalize_whitespace(text)
        .to_lowercase()
        .contains(&normalize_whitespace(phrase).to_lowercase())
}

/// Keep only the results whose content or heading contains every phrase,
/// preserving the existing (ranked) order. Shared by
/// [`WorkspaceDb::search_keyword_chunks`] and
/// [`WorkspaceDb::search_hybrid_chunks`], since the latter's vector leg
/// carries no phrase information of its own.
fn filter_by_phrases(results: &mut Vec<ChunkSearchResult>, phrases: &[String]) {
    results.retain(|r| {
        phrases.iter().all(|phrase| {
            contains_phrase(&r.content, phrase)
                || r.heading
                    .as_deref()
                    .is_some_and(|h| contains_phrase(h, phrase))
        })
    });
}

/// Term frequencies for a chunk's content plus its heading.
fn term_frequencies(content: &str, heading: Option<&str>) -> Vec<(String, u32)> {
    let mut counts: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
    for term in tokenize(content)
        .into_iter()
        .chain(heading.map(tokenize).unwrap_or_default())
    {
        let entry = counts.entry(term).or_insert(0);
        *entry = entry.saturating_add(1);
    }
    counts.into_iter().collect()
}

/// Total term occurrences, the chunk length BM25 normalizes by.
fn term_count(terms: &[(String, u32)]) -> i64 {
    terms
        .iter()
        .fold(0i64, |acc, (_, tf)| acc.saturating_add(i64::from(*tf)))
}

/// Which chunks a search may return. Empty means the whole workspace; a
/// scope narrows it to certain documents, to an explicit set of chunks
/// (the chunks a graph entity was extracted from), or to both at once.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkScope {
    /// Empty means every document: `resolve_document_ids` refuses an id
    /// that matches nothing, so an empty list never means "none".
    documents: Vec<String>,
    /// `None` means no chunk restriction at all; `Some(ids)` means exactly
    /// those chunks, and `Some(empty)` means none — an entity whose chunks
    /// came back empty must return no rows, not the whole workspace.
    chunks: Option<Vec<String>>,
}

impl ChunkScope {
    /// Every ready chunk in the workspace.
    #[must_use]
    pub fn all() -> Self {
        Self::default()
    }

    /// Only chunks belonging to these documents.
    #[must_use]
    pub fn documents<I: IntoIterator<Item = String>>(ids: I) -> Self {
        Self {
            documents: ids.into_iter().collect(),
            chunks: None,
        }
    }

    /// Narrow further to these chunk ids, however few.
    #[must_use]
    pub fn and_chunks<I: IntoIterator<Item = String>>(mut self, ids: I) -> Self {
        self.chunks = Some(ids.into_iter().collect());
        self
    }

    /// Whether the scope names nothing at all, so a search must return no
    /// rows instead of widening to the whole workspace.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.as_ref().is_some_and(Vec::is_empty)
    }

    /// The `AND ...` fragment for a query that has `_quack_chunks` as `c`.
    fn sql(&self) -> String {
        let mut clauses = Vec::new();
        if !self.documents.is_empty() {
            let placeholders = vec!["?"; self.documents.len()].join(", ");
            clauses.push(format!(" AND c.document_id IN ({placeholders})"));
        }
        if let Some(chunks) = self.chunks.as_ref().filter(|ids| !ids.is_empty()) {
            let placeholders = vec!["?"; chunks.len()].join(", ");
            clauses.push(format!(" AND c.id IN ({placeholders})"));
        }
        clauses.concat()
    }

    /// How many parameters [`ChunkScope::bind`] will push.
    fn len(&self) -> usize {
        self.documents
            .len()
            .saturating_add(self.chunks.as_ref().map_or(0, Vec::len))
    }

    /// Push the scope's parameters, in the order [`ChunkScope::sql`] names
    /// them.
    fn bind<'a>(&'a self, params: &mut Vec<&'a dyn duckdb::ToSql>) {
        for id in &self.documents {
            params.push(id);
        }
        for id in self.chunks.iter().flatten() {
            params.push(id);
        }
    }
}

/// Reciprocal rank fusion of two rankings of the same chunk space.
fn fuse_rankings(
    vector: Vec<ChunkSearchResult>,
    keyword: Vec<ChunkSearchResult>,
    top_k: u32,
    rrf_k: u32,
) -> Vec<ChunkSearchResult> {
    let mut fused: Vec<ChunkSearchResult> = Vec::new();
    let k = f64::from(rrf_k);
    for ranking in [vector, keyword] {
        for (rank, mut hit) in ranking.into_iter().enumerate() {
            let contribution = 1.0 / (k + f64::from(u32::try_from(rank).unwrap_or(u32::MAX)) + 1.0);
            if let Some(existing) = fused.iter_mut().find(|h| h.id == hit.id) {
                existing.score += contribution;
            } else {
                hit.score = contribution;
                fused.push(hit);
            }
        }
    }
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.chunk_index.cmp(&b.chunk_index))
    });
    fused.truncate(usize::try_from(top_k).unwrap_or(usize::MAX));
    fused
}

/// Lets another thread stop the statements one piece of work runs, and
/// only while that work runs: [`WorkspaceDb::cancellable`] registers the
/// connection's interrupt handle for the length of the work and removes it
/// before the connection is released, so a late cancel never interrupts
/// whatever the connection runs next. A cancel before the work starts makes
/// it fail at once. How the work queue stops a running SQL job.
#[derive(Clone, Default)]
pub struct QueryCanceller(std::sync::Arc<std::sync::Mutex<CancelSlot>>);

#[derive(Default)]
struct CancelSlot {
    running: Option<std::sync::Arc<duckdb::InterruptHandle>>,
    cancelled: bool,
}

impl QueryCanceller {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn slot(&self) -> std::sync::MutexGuard<'_, CancelSlot> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Interrupt the statement running under this canceller, if any, and
    /// refuse any work that has not started yet.
    pub fn cancel(&self) {
        let mut slot = self.slot();
        slot.cancelled = true;
        if let Some(handle) = &slot.running {
            handle.interrupt();
        }
    }
}

/// Disarms the watchdog when dropped: the closed channel wakes it.
struct TimeoutGuard {
    _disarm: std::sync::mpsc::Sender<()>,
}

/// Guards one `BEGIN TRANSACTION READ ONLY` on a reader connection.
/// Dropping without committing rolls back, so an early `?` return inside
/// [`WorkspaceDb::read_only`] never leaves the transaction open.
struct ReadOnlyGuard<'a> {
    conn: &'a duckdb::Connection,
    committed: bool,
}

impl<'a> ReadOnlyGuard<'a> {
    fn begin(conn: &'a duckdb::Connection) -> Result<Self> {
        conn.execute_batch("BEGIN TRANSACTION READ ONLY")?;
        Ok(Self {
            conn,
            committed: false,
        })
    }

    fn commit(&mut self) -> Result<()> {
        self.conn.execute_batch("COMMIT")?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for ReadOnlyGuard<'_> {
    fn drop(&mut self) {
        if !self.committed
            && let Err(e) = self.conn.execute_batch("ROLLBACK")
        {
            tracing::warn!(error = %e, "failed to roll back a read-only transaction");
        }
    }
}

/// True when `sql` is one statement that starts with a read-only keyword
/// `DuckDB` cannot serialize. A semicolon anywhere but the very end disqualifies
/// it, so `DESCRIBE t; DROP TABLE t` is not read-only. `EXPLAIN` alone only
/// prints a plan, but `EXPLAIN ANALYZE` executes it — `EXPLAIN ANALYZE
/// INSERT INTO t VALUES (1)` really inserts — so that combination is never
/// read-only regardless of what it explains.
fn is_single_read_only_statement(sql: &str) -> bool {
    let trimmed = sql.trim().trim_end_matches(';').trim_end();
    if trimmed.contains(';') {
        return false;
    }
    let mut words = trimmed.split_whitespace();
    let Some(first) = words.next() else {
        return false;
    };
    if first.eq_ignore_ascii_case("EXPLAIN")
        && words
            .next()
            .is_some_and(|w| w.eq_ignore_ascii_case("ANALYZE"))
    {
        return false;
    }
    READ_ONLY_KEYWORDS
        .iter()
        .any(|k| k.eq_ignore_ascii_case(first))
}

/// Walk a serialized statement tree collecting `table_name` values from
/// base-table references.
/// Null out the `value` of every `CONSTANT` node and drop every
/// `query_location`, which shifts with a literal's length.
fn blank_constants(node: &mut serde_json::Value) {
    match node {
        serde_json::Value::Object(map) => {
            map.remove("query_location");
            if map.get("class").and_then(serde_json::Value::as_str) == Some("CONSTANT") {
                map.insert(String::from("value"), serde_json::Value::Null);
            }
            for child in map.values_mut() {
                blank_constants(child);
            }
        }
        serde_json::Value::Array(items) => {
            for child in items {
                blank_constants(child);
            }
        }
        _ => {}
    }
}

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

/// A `DuckDB` list literal of text values, bound as `?::VARCHAR[]`.
fn sql_text_list(items: &[String]) -> String {
    let quoted: Vec<String> = items
        .iter()
        .map(|item| format!("'{}'", item.replace('\'', "''")))
        .collect();
    format!("[{}]", quoted.join(","))
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

/// A vector as the list literal `DuckDB` casts to `FLOAT[N]`.
#[must_use]
pub fn embedding_literal(embedding: &[f32]) -> String {
    format_embedding(embedding)
}

fn extract_value(row: &duckdb::Row<'_>, idx: usize) -> serde_json::Value {
    match row.get_ref(idx) {
        Ok(value) => json_of(duckdb::types::Value::from(value)),
        Err(_) => serde_json::Value::Null,
    }
}

/// A `DuckDB` value as JSON: numbers stay numbers (integers beyond i64 and
/// decimals keep their digits as strings when they would not round-trip),
/// dates and times are ISO 8601 text, blobs are base64, and lists, structs,
/// and maps nest.
fn json_of(value: duckdb::types::Value) -> serde_json::Value {
    use duckdb::types::Value;
    use serde_json::Value as Json;
    match value {
        Value::Null => Json::Null,
        Value::Boolean(b) => Json::Bool(b),
        Value::TinyInt(n) => Json::from(n),
        Value::SmallInt(n) => Json::from(n),
        Value::Int(n) => Json::from(n),
        Value::BigInt(n) => Json::from(n),
        Value::UTinyInt(n) => Json::from(n),
        Value::USmallInt(n) => Json::from(n),
        Value::UInt(n) => Json::from(n),
        Value::UBigInt(n) => Json::from(n),
        Value::HugeInt(n) => match i64::try_from(n) {
            Ok(n) => Json::from(n),
            Err(_) => Json::String(n.to_string()),
        },
        Value::UHugeInt(n) => match u64::try_from(n) {
            Ok(n) => Json::from(n),
            Err(_) => Json::String(n.to_string()),
        },
        Value::Float(f) => {
            serde_json::Number::from_f64(f64::from(f)).map_or(Json::Null, Json::Number)
        }
        Value::Double(f) => serde_json::Number::from_f64(f).map_or(Json::Null, Json::Number),
        Value::Decimal(d) => {
            let text = d.to_string();
            text.parse::<serde_json::Number>()
                .map_or(Json::String(text), Json::Number)
        }
        Value::Timestamp(unit, n) => Json::String(timestamp_text(unit, n)),
        Value::Date32(days) => Json::String(date_text(days)),
        Value::Time64(unit, n) => Json::String(time_text(unit, n)),
        Value::Interval {
            months,
            days,
            nanos,
        } => Json::String(format!("{months} months {days} days {nanos} ns")),
        Value::Text(s) | Value::Enum(s) => Json::String(s),
        Value::Blob(bytes) | Value::Geometry(bytes) => {
            use base64::Engine as _;
            Json::String(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        Value::List(items) | Value::Array(items) => {
            Json::Array(items.into_iter().map(json_of).collect())
        }
        Value::Struct(fields) => {
            let mut object = serde_json::Map::new();
            for (key, value) in fields.iter() {
                object.insert(key.clone(), json_of(value.clone()));
            }
            Json::Object(object)
        }
        Value::Map(entries) => {
            let mut object = serde_json::Map::new();
            for (key, value) in entries.iter() {
                let key = match json_of(key.clone()) {
                    Json::String(s) => s,
                    other => other.to_string(),
                };
                object.insert(key, json_of(value.clone()));
            }
            Json::Object(object)
        }
        Value::Union(inner) => json_of(*inner),
        // The enum is non-exhaustive; a type this build does not know
        // renders through Debug rather than silently as null.
        other => Json::String(format!("{other:?}")),
    }
}

fn unit_to_nanos(unit: duckdb::types::TimeUnit, n: i64) -> Option<i128> {
    let n = i128::from(n);
    match unit {
        duckdb::types::TimeUnit::Second => n.checked_mul(1_000_000_000),
        duckdb::types::TimeUnit::Millisecond => n.checked_mul(1_000_000),
        duckdb::types::TimeUnit::Microsecond => n.checked_mul(1_000),
        duckdb::types::TimeUnit::Nanosecond => Some(n),
    }
}

/// `YYYY-MM-DD HH:MM:SS[.ffffff]`, as `DuckDB` prints a naive timestamp.
fn timestamp_text(unit: duckdb::types::TimeUnit, n: i64) -> String {
    let Some(nanos) = unit_to_nanos(unit, n) else {
        return n.to_string();
    };
    let Ok(ts) = jiff::Timestamp::from_nanosecond(nanos) else {
        return n.to_string();
    };
    let civil = ts.to_zoned(jiff::tz::TimeZone::UTC).datetime();
    if civil.subsec_nanosecond() == 0 {
        civil.strftime("%Y-%m-%d %H:%M:%S").to_string()
    } else {
        civil.strftime("%Y-%m-%d %H:%M:%S%.6f").to_string()
    }
}

fn date_text(days: i32) -> String {
    jiff::civil::date(1970, 1, 1)
        .checked_add(jiff::Span::new().days(days))
        .map_or_else(|_| days.to_string(), |d| d.to_string())
}

fn time_text(unit: duckdb::types::TimeUnit, n: i64) -> String {
    let Some(nanos) = unit_to_nanos(unit, n) else {
        return n.to_string();
    };
    let Ok(nanos) = i64::try_from(nanos) else {
        return n.to_string();
    };
    jiff::civil::time(0, 0, 0, 0)
        .checked_add(jiff::Span::new().nanoseconds(nanos))
        .map_or_else(
            |_| n.to_string(),
            |t| {
                if t.subsec_nanosecond() == 0 {
                    t.strftime("%H:%M:%S").to_string()
                } else {
                    t.strftime("%H:%M:%S%.6f").to_string()
                }
            },
        )
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
    /// Write results as a human-readable aligned table.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to `out` fails.
    pub fn write_table(&self, out: &mut impl Write) -> Result<()> {
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
    pub fn write_ndjson(&self, out: &mut impl Write) -> Result<()> {
        // Written by hand so keys keep column order; serde_json's map sorts.
        let keys = self.json_keys();
        for row in &self.rows {
            let mut fields = Vec::with_capacity(keys.len());
            for (column, value) in keys.iter().zip(row) {
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
    pub fn write_csv(&self, out: &mut impl Write) -> Result<()> {
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
    pub fn write_markdown(&self, out: &mut impl Write) -> Result<()> {
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
    pub fn write_json(&self, out: &mut impl Write) -> Result<()> {
        let keys = self.json_keys();
        let json_rows: Vec<serde_json::Map<String, serde_json::Value>> = self
            .rows
            .iter()
            .map(|row| {
                keys.iter()
                    .zip(row.iter())
                    .map(|(col, val)| (col.clone(), val.clone()))
                    .collect()
            })
            .collect();

        serde_json::to_writer_pretty(&mut *out, &json_rows)?;
        writeln!(out)?;
        Ok(())
    }

    /// The column names as object keys: a repeated name gets a numeric
    /// suffix (`a`, `a_1`, `a_2`) the way `DuckDB` itself writes JSON,
    /// so a self-join keeps every column (issue #65).
    #[must_use]
    pub fn json_keys(&self) -> Vec<String> {
        let mut taken: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut keys = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            let mut key = column.clone();
            let mut n: u32 = 0;
            while !taken.insert(key.clone()) {
                n = n.saturating_add(1);
                key = format!("{column}_{n}");
            }
            keys.push(key);
        }
        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn config_in(dir: &Path) -> crate::config::Config {
        let mut config = crate::config::Config::default();
        config.general.data_dir = dir.join("data");
        config
    }

    /// Two statements that differ only in their literals, spacing, and
    /// case share a shape; a different column, a different statement kind,
    /// and anything `DuckDB` cannot serialize do not.
    #[test]
    fn a_canceller_interrupts_only_the_work_it_guards() {
        let db = WorkspaceDb::open_in_memory(4)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .with_query_timeout(Duration::from_secs(60));
        // Cancelled before the work starts: it never runs.
        let early = QueryCanceller::new();
        early.cancel();
        assert!(db.cancellable(&early, |_| Ok(())).is_err());

        // Cancelled while a long statement runs: the statement stops.
        let canceller = QueryCanceller::new();
        let remote = canceller.clone();
        let stopper = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            remote.cancel();
        });
        let started = std::time::Instant::now();
        let outcome = db.cancellable(&canceller, |db| {
            db.execute_query_capped(
                "SELECT count(*) FROM range(100000000000) t(i) WHERE i % 7 = 0",
                10,
            )
        });
        assert!(outcome.is_err(), "the statement was interrupted");
        assert!(started.elapsed() < Duration::from_secs(30));
        assert!(stopper.join().is_ok());

        // The connection is free again, and a canceller no longer guarding
        // anything interrupts nothing.
        let after = QueryCanceller::new();
        assert!(
            db.cancellable(&after, |db| db.execute_query_capped("SELECT 1", 10))
                .is_ok()
        );
        after.cancel();
        assert!(db.execute_query_capped("SELECT 2", 10).is_ok());
    }

    #[test]
    fn statement_shape_ignores_literals_and_source_positions() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        let shape = |sql: &str| {
            db.statement_shape(sql)
                .unwrap_or_else(|e| fail(&e.to_string()))
        };
        let nevada = shape("SELECT x FROM t WHERE s = 'NEVADA' AND n > 10 LIMIT 5");
        let carolina = shape("select x\n from t where s='NORTH CAROLINA' and n > 250 limit 5");
        assert!(nevada.is_some());
        assert_eq!(nevada, carolina);
        assert_ne!(
            nevada,
            shape("SELECT y FROM t WHERE s = 'NEVADA' AND n > 10 LIMIT 5")
        );
        assert_ne!(nevada, shape("SELECT x FROM t WHERE s = 'NEVADA' LIMIT 5"));
        assert_eq!(shape("CREATE TABLE t2 AS SELECT 1"), None);
        assert_eq!(shape("SELECT FROM WHERE"), None);
    }

    /// Design doc 7.4: a read-classified statement may still name a file, so
    /// the connection itself is confined to the workspace directory and then
    /// locked, for agent SQL and user SQL alike.
    #[test]
    fn workspace_connection_is_confined_to_its_directory_and_locked() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let config = config_in(dir.path());
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));

        let outside = dir.path().join("outside.csv");
        std::fs::write(&outside, "a\n1\n").unwrap_or_else(|e| fail(&e.to_string()));
        let inside = config.workspace_files_dir("ws").join("inside.csv");
        std::fs::write(&inside, "a\n1\n2\n").unwrap_or_else(|e| fail(&e.to_string()));

        for sql in [
            format!("SELECT * FROM read_csv_auto('{}')", outside.display()),
            format!("SELECT * FROM read_text('{}')", outside.display()),
            format!("SELECT * FROM '{}'", outside.display()),
            format!(
                "ATTACH '{}' AS other",
                dir.path().join("other.duckdb").display()
            ),
            String::from("INSTALL httpfs"),
            String::from("SET memory_limit = '8GB'"),
            String::from("SET enable_external_access = true"),
            String::from("SET allowed_directories = ['/']"),
        ] {
            let err = db.execute_query(&sql).err();
            assert!(err.is_some(), "ran outside the sandbox: {sql}");
            let text = err.map(|e| e.to_string()).unwrap_or_default();
            // Replacement scans are simply gone, so `FROM 'file'` is a
            // catalog miss; everything else is a permission or lock error.
            assert!(
                text.contains("Permission Error")
                    || text.contains("locked")
                    || text.contains("Catalog Error"),
                "{sql}: {text}"
            );
        }

        // Ingestion's own reads under files/ still work, through the same reader.
        let rows = db
            .execute_query(&format!(
                "SELECT count(*) FROM read_csv_auto('{}')",
                inside.display()
            ))
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(rows.rows.len(), 1);
        assert_eq!(
            rows.rows.first().and_then(|r| r.first()),
            Some(&serde_json::Value::Number(2.into()))
        );
        // Ordinary statements are untouched.
        assert!(db.execute_statement("CREATE TABLE t(a INT)").is_ok());
        assert!(db.execute_query("SELECT * FROM t").is_ok());
    }

    /// Proof-of-concept for the reader connection: `try_clone`
    /// succeeds once `lock_configuration = true` (set by `confine_to` on
    /// open), because `DuckDB` locks the connection's *configuration*, not
    /// its ability to open more connections to the same database; and a
    /// write inside `BEGIN TRANSACTION READ ONLY` is rejected by `DuckDB`
    /// itself, before it ever reaches the workspace's write-gating.
    #[test]
    fn reader_connection_clones_after_lock_configuration_and_cannot_write() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        let reader = db.conn.try_clone().unwrap_or_else(|e| fail(&e.to_string()));

        reader
            .execute_batch("BEGIN TRANSACTION READ ONLY")
            .unwrap_or_else(|e| fail(&e.to_string()));
        let err = reader
            .execute_batch("CREATE TABLE t(a INT)")
            .err()
            .unwrap_or_else(|| fail("write inside a read-only transaction should have failed"));
        // The connection and its in-memory database are dropped at the end
        // of the test; no need to end the transaction explicitly.
        assert!(
            err.to_string().to_lowercase().contains("read"),
            "unexpected error: {err}"
        );
    }

    /// The actual guarded path (`try_clone_reader` plus `read_only`), not
    /// just the raw statements the proof above assumes: a write attempted
    /// inside `read_only` on a reader clone is rejected, and the connection
    /// is still usable for a genuine read right after.
    #[test]
    fn read_only_on_a_reader_clone_rejects_a_write() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        let reader = db
            .try_clone_reader()
            .unwrap_or_else(|e| fail(&e.to_string()));

        let err = reader
            .read_only(|db| db.execute_statement("CREATE TABLE t(a INT)"))
            .err()
            .unwrap_or_else(|| fail("a write inside read_only on a reader should have failed"));
        assert!(
            err.to_string().to_lowercase().contains("read"),
            "unexpected error: {err}"
        );
        assert!(reader.read_only(|db| db.execute_query("SELECT 1")).is_ok());
    }

    /// A write on the writer, made before a reader is cloned from it, is
    /// still visible to the reader afterward, and so is a write made even
    /// later: `DuckDB` snapshots a transaction at `BEGIN`, not at
    /// `try_clone`, and the writer's statements autocommit.
    #[test]
    fn reader_clone_sees_the_writers_prior_and_later_writes() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        db.execute_statement("CREATE TABLE t(a INT)")
            .unwrap_or_else(|e| fail(&e.to_string()));
        db.execute_statement("INSERT INTO t VALUES (1)")
            .unwrap_or_else(|e| fail(&e.to_string()));
        let reader = db
            .try_clone_reader()
            .unwrap_or_else(|e| fail(&e.to_string()));

        let count = |reader: &WorkspaceDb| -> i64 {
            reader
                .read_only(|db| db.execute_query("SELECT count(*) FROM t"))
                .unwrap_or_else(|e| fail(&e.to_string()))
                .rows
                .first()
                .and_then(|row| row.first())
                .and_then(serde_json::Value::as_i64)
                .unwrap_or_else(|| fail("no count returned"))
        };
        assert_eq!(count(&reader), 1);

        db.execute_statement("INSERT INTO t VALUES (2)")
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(count(&reader), 2);
    }

    /// A closure that errors inside `read_only` rolls the transaction
    /// back, and the connection is immediately reusable for another
    /// `read_only` call: `ReadOnlyGuard::drop` did not leave one open.
    #[test]
    fn read_only_rolls_back_on_error_and_the_connection_stays_usable() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            db.read_only(|db| db.execute_query("SELECT * FROM no_such_table"))
                .is_err()
        );
        assert!(db.read_only(|db| db.execute_query("SELECT 1")).is_ok());
    }

    /// The reader clone inherits confinement: it cannot read outside the
    /// workspace directory either, exactly like the writer (mirrors
    /// `workspace_connection_is_confined_to_its_directory_and_locked`).
    #[test]
    fn reader_clone_is_confined_like_the_writer() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let config = config_in(dir.path());
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        let reader = db
            .try_clone_reader()
            .unwrap_or_else(|e| fail(&e.to_string()));

        let outside = dir.path().join("outside.csv");
        std::fs::write(&outside, "a\n1\n").unwrap_or_else(|e| fail(&e.to_string()));
        let err = reader
            .execute_query(&format!(
                "SELECT * FROM read_csv_auto('{}')",
                outside.display()
            ))
            .err();
        assert!(err.is_some(), "the reader escaped the sandbox");
        let text = err.map(|e| e.to_string()).unwrap_or_default();
        // `read_csv_auto` on a real, existing file outside the workspace
        // has only one legitimate way to fail: the confinement check.
        // Unlike the writer's confinement test, nothing here can produce a
        // Catalog Error, so that arm would only ever hide an unrelated
        // regression (`read_csv_auto` itself going missing, say).
        assert!(text.contains("Permission Error"), "{text}");

        let inside = config.workspace_files_dir("ws").join("inside.csv");
        std::fs::write(&inside, "a\n1\n2\n").unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            reader
                .execute_query(&format!(
                    "SELECT * FROM read_csv_auto('{}')",
                    inside.display()
                ))
                .is_ok()
        );
    }

    /// `DuckDB` temp tables (the CLI's `stdin` table, `ingestion::STDIN_TABLE`)
    /// are connection-local: a reader clone opened after the writer created
    /// one does not see it. Tools reading through the reader must never be
    /// pointed at it; only the writer connection (`run_sql`) can.
    #[test]
    fn reader_connection_cannot_see_the_writers_temp_tables() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        db.execute_statement("CREATE TEMP TABLE stdin AS SELECT 1 AS a")
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(db.execute_query("SELECT * FROM stdin").is_ok());

        let reader = db
            .try_clone_reader()
            .unwrap_or_else(|e| fail(&e.to_string()));
        let err = reader
            .execute_query("SELECT * FROM stdin")
            .err()
            .unwrap_or_else(|| fail("the reader should not see the writer's temp table"));
        assert!(err.to_string().contains("stdin"), "{err}");
    }

    #[test]
    fn has_temp_tables_reports_a_piped_stdin_table() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            !db.has_temp_tables()
                .unwrap_or_else(|e| fail(&e.to_string()))
        );
        db.execute_statement("CREATE TEMP TABLE stdin AS SELECT 1 AS a")
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            db.has_temp_tables()
                .unwrap_or_else(|e| fail(&e.to_string()))
        );
    }

    #[test]
    fn in_memory_connection_reads_no_files_at_all() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let file = dir.path().join("x.csv");
        std::fs::write(&file, "a\n1\n").unwrap_or_else(|e| fail(&e.to_string()));
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        let err = db
            .execute_query(&format!(
                "SELECT * FROM read_csv_auto('{}')",
                file.display()
            ))
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("Permission Error"), "{err}");
        assert!(db.execute_statement("SET threads = 1").is_err());
    }

    #[test]
    fn query_values_keep_fractions_dates_and_nested_types() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        let results = db
            .execute_query(
                "SELECT 20.5 AS dec, 20.5::DOUBLE AS dbl, 3.25::FLOAT AS flt, \
                 DATE '2024-01-02' AS d, TIMESTAMP '2024-01-02 03:04:05' AS ts, \
                 TIMESTAMP '2024-01-02 03:04:05.25' AS tsf, TIME '03:04:05' AS t, \
                 12345678901234567890::HUGEINT AS big, [1, 2] AS arr, {'a': 1, 'b': 'x'} AS st, \
                 MAP {'k': 1} AS m, NULL AS n, 'text' AS s, true AS b, 7::UTINYINT AS u",
            )
            .unwrap_or_else(|e| fail(&e.to_string()));
        let row = results.rows.first().unwrap_or_else(|| fail("no row"));
        let expected = serde_json::json!([
            20.5,
            20.5,
            3.25,
            "2024-01-02",
            "2024-01-02 03:04:05",
            "2024-01-02 03:04:05.250000",
            "03:04:05",
            "12345678901234567890",
            [1, 2],
            {"a": 1, "b": "x"},
            {"k": 1},
            null,
            "text",
            true,
            7
        ]);
        assert_eq!(serde_json::Value::Array(row.clone()), expected);
    }

    #[test]
    fn describe_table_reports_the_exact_row_count() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        assert!(db.execute_statement("CREATE TABLE t(a INT)").is_ok());
        assert!(
            db.execute_statement("INSERT INTO t VALUES (1), (2), (3)")
                .is_ok()
        );
        let desc = db
            .describe_table("t")
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(desc.row_count, 3);
        assert_eq!(desc.sample_rows.rows.len(), 3);
        assert!(db.count_rows("missing").is_err());
        let version = db.duckdb_version().unwrap_or_else(|e| fail(&e.to_string()));
        assert!(version.starts_with('v'), "{version}");
    }

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
    fn capped_query_keeps_the_cap_and_counts_the_rest() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        let capped = db
            .execute_query_capped("SELECT range AS n FROM range(10)", 3)
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(capped.results.columns, vec![String::from("n")]);
        assert_eq!(capped.results.rows.len(), 3);
        assert_eq!(capped.total_rows, 10);
        assert!(capped.truncated());
        assert_eq!(capped.omitted(), 7);

        let exact = db
            .execute_query_capped("SELECT range AS n FROM range(3)", 3)
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(exact.results.rows.len(), 3);
        assert!(!exact.truncated());
        assert_eq!(exact.omitted(), 0);

        let none = db
            .execute_query_capped("SELECT 1 WHERE false", 3)
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(none.total_rows, 0);
        assert!(!none.truncated());

        let all = db
            .execute_query("SELECT range AS n FROM range(10)")
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(all.rows.len(), 10);
    }

    /// Columns that share a name keep every value under suffixed keys in
    /// both JSON shapes; CSV, table, and markdown already kept them
    /// (issue #65).
    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn json_writers_keep_columns_that_share_a_name() {
        let results = QueryResults {
            columns: vec![
                String::from("a"),
                String::from("a"),
                String::from("a_1"),
                String::from("a"),
            ],
            rows: vec![vec![
                serde_json::Value::from(1),
                serde_json::Value::from(2),
                serde_json::Value::from(3),
                serde_json::Value::from(4),
            ]],
        };
        assert_eq!(results.json_keys(), ["a", "a_1", "a_1_1", "a_2"]);
        let mut buf = Vec::new();
        results.write_ndjson(&mut buf).unwrap();
        assert_eq!(
            String::from_utf8(buf).unwrap(),
            "{\"a\":1,\"a_1\":2,\"a_1_1\":3,\"a_2\":4}\n"
        );
        let mut buf = Vec::new();
        results.write_json(&mut buf).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(
            parsed,
            serde_json::json!([{ "a": 1, "a_1": 2, "a_1_1": 3, "a_2": 4 }])
        );
        assert_eq!(sample().json_keys(), ["name", "n"]);
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
    fn tokenize_lowercases_and_splits_on_punctuation() {
        assert_eq!(
            tokenize("Policy POL-8841 renews; see \"Exclusions\" (page 12)."),
            vec![
                "polici", "pol", "8841", "pol8841", "renew", "see", "exclus", "page", "12"
            ]
        );
        // Inflections meet at one stem; codes and numbers are untouched.
        assert_eq!(
            tokenize("renewal renewals renewing"),
            vec!["renew", "renew", "renew"]
        );
        assert_eq!(tokenize("AB-12X9"), vec!["ab", "12x9", "ab12x9"]);
        assert!(tokenize("  --- ").is_empty());
    }

    #[test]
    fn tokenize_indexes_the_joined_form_of_an_identifier() {
        // Hyphen, dot, underscore, slash, and colon all join.
        assert_eq!(tokenize("v1.2.3"), vec!["v1", "2", "3", "v123"]);
        assert_eq!(tokenize("ABC_123"), vec!["abc", "123", "abc123"]);
        assert_eq!(tokenize("ns/part:7"), vec!["ns", "part", "7", "nspart7"]);
        // A joiner touching whitespace does not merge across words: prose
        // punctuation still tokenizes exactly as before.
        assert_eq!(
            tokenize("end of sentence - new sentence."),
            vec!["end", "of", "sentenc", "new", "sentenc"]
        );
        // The query side uses the same function, so a bare joined form
        // already in text (`pol8841`) is found by the query `POL-8841`.
        assert!(tokenize("POL-8841").contains(&String::from("pol8841")));
        assert_eq!(tokenize("pol8841"), vec!["pol8841"]);
        // The joined form uses full Unicode case folding, not ASCII-only
        // lowercasing, so a non-ASCII identifier's casing does not change
        // which term it indexes: `Ünit-9` in text and `ünit-9` in a query
        // must both produce the joined term `ünit9`.
        assert_eq!(tokenize("Ünit-9").last(), tokenize("ünit-9").last(),);
        assert_eq!(tokenize("Ünit-9").last(), Some(&String::from("ünit9")));
    }

    #[test]
    fn term_frequencies_count_heading_too() {
        let tf = term_frequencies("flood flood damage", Some("Flood Exclusions"));
        assert_eq!(
            tf,
            vec![
                (String::from("damag"), 1),
                (String::from("exclus"), 1),
                (String::from("flood"), 3),
            ]
        );
        assert_eq!(term_count(&tf), 5);
    }

    #[test]
    fn extract_phrases_reads_balanced_quotes() {
        assert_eq!(
            extract_phrases("\"flood exclusion\""),
            vec![String::from("flood exclusion")]
        );
        assert_eq!(
            extract_phrases("find \"flood exclusion\" near \"water damage\""),
            vec![
                String::from("flood exclusion"),
                String::from("water damage")
            ]
        );
        assert!(extract_phrases("no quotes here").is_empty());
        assert!(extract_phrases("\"\"").is_empty());
        // Unbalanced quotes: an odd count is ordinary text, not a phrase.
        assert!(extract_phrases("say \"hello").is_empty());
        assert!(extract_phrases("a \"b\" c\" d").is_empty());
    }

    #[test]
    fn contains_phrase_normalizes_case_and_whitespace() {
        assert!(contains_phrase(
            "the FLOOD   Exclusion\napplies here",
            "flood exclusion"
        ));
        assert!(!contains_phrase("flood and exclusion", "flood exclusion"));
        assert!(contains_phrase("Flood Exclusion", "  flood   exclusion  "));
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

    fn insert_ready_document(db: &WorkspaceDb, id: &str) {
        db.insert_document(&NewDocument::new(id, "doc.txt", "text/plain", 10).with_status("ready"))
            .unwrap_or_else(|e| fail(&e.to_string()));
    }

    fn insert_text_chunk(
        db: &WorkspaceDb,
        id: &str,
        document_id: &str,
        chunk_index: u32,
        content: &str,
    ) {
        db.insert_chunk(&NewChunk {
            id,
            document_id,
            chunk_index,
            content,
            heading: None,
            page: None,
            embedding: None,
        })
        .unwrap_or_else(|e| fail(&e.to_string()));
    }

    /// The joined identifier term must rank a chunk containing the exact
    /// identifier above one that only contains its split pieces apart.
    #[test]
    fn search_keyword_chunks_ranks_the_exact_identifier_above_split_terms() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        insert_ready_document(&db, "doc1");
        insert_text_chunk(&db, "c1", "doc1", 0, "Policy POL-8841 covers water damage.");
        insert_text_chunk(
            &db,
            "c2",
            "doc1",
            1,
            "The pol number appears here, and the 8841 total appears elsewhere in this paragraph.",
        );

        let results = db
            .search_keyword_chunks("POL-8841", 10, &ChunkScope::all())
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            results.first().map(|r| r.id.as_str()),
            Some("c1"),
            "the exact identifier should outrank its split pieces: {results:?}"
        );
    }

    #[test]
    fn search_keyword_chunks_filters_candidates_by_quoted_phrase() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        insert_ready_document(&db, "doc1");
        insert_text_chunk(
            &db,
            "c1",
            "doc1",
            0,
            "The flood exclusion applies to basements.",
        );
        // Same two words, not adjacent: matches the bag-of-words ranking but
        // not the phrase.
        insert_text_chunk(
            &db,
            "c2",
            "doc1",
            1,
            "Exclusion of flood risk is handled in a separate clause.",
        );

        let results = db
            .search_keyword_chunks("\"flood exclusion\"", 10, &ChunkScope::all())
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(results.len(), 1);
        assert_eq!(results.first().map(|r| r.id.as_str()), Some("c1"));
    }

    #[test]
    fn search_keyword_chunks_phrase_match_in_heading() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        insert_ready_document(&db, "doc1");
        db.insert_chunk(&NewChunk {
            id: "c1",
            document_id: "doc1",
            chunk_index: 0,
            content: "See below for what is not covered.",
            heading: Some("Flood Exclusion"),
            page: None,
            embedding: None,
        })
        .unwrap_or_else(|e| fail(&e.to_string()));

        let results = db
            .search_keyword_chunks("\"flood exclusion\"", 10, &ChunkScope::all())
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(results.len(), 1);
        assert_eq!(results.first().map(|r| r.id.as_str()), Some("c1"));
    }

    #[test]
    fn search_keyword_chunks_phrase_with_no_match_returns_empty() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        insert_ready_document(&db, "doc1");
        insert_text_chunk(
            &db,
            "c1",
            "doc1",
            0,
            "Exclusion of flood risk is handled in a separate clause.",
        );

        let results = db
            .search_keyword_chunks("\"flood exclusion\"", 10, &ChunkScope::all())
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            results.is_empty(),
            "a phrase with no exact match must not fall back to unfiltered candidates: {results:?}"
        );
    }

    #[test]
    fn search_keyword_chunks_unbalanced_quote_is_ordinary_text() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        insert_ready_document(&db, "doc1");
        insert_text_chunk(
            &db,
            "c1",
            "doc1",
            0,
            "Exclusion of flood risk is handled in a separate clause.",
        );

        let results = db
            .search_keyword_chunks("\"flood exclusion", 10, &ChunkScope::all())
            .unwrap_or_else(|e| fail(&e.to_string()));
        // No phrase requirement kicks in: ordinary bag-of-words matching
        // still finds the chunk even though the words are not adjacent.
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_hybrid_chunks_ranks_identifier_and_filters_phrase() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        insert_ready_document(&db, "doc1");
        let embedding = [0.1_f32, 0.2, 0.3, 0.4];
        db.insert_chunk(&NewChunk {
            id: "c1",
            document_id: "doc1",
            chunk_index: 0,
            content: "Policy POL-8841 covers water damage.",
            heading: None,
            page: None,
            embedding: Some(&embedding),
        })
        .unwrap_or_else(|e| fail(&e.to_string()));
        db.insert_chunk(&NewChunk {
            id: "c2",
            document_id: "doc1",
            chunk_index: 1,
            content: "The pol number appears here, and the 8841 total appears elsewhere in this paragraph.",
            heading: None,
            page: None,
            embedding: Some(&embedding),
        })
        .unwrap_or_else(|e| fail(&e.to_string()));

        let results = db
            .search_hybrid_chunks("POL-8841", &embedding, 10, 60, &ChunkScope::all())
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(results.first().map(|r| r.id.as_str()), Some("c1"));
    }

    #[test]
    fn search_hybrid_chunks_phrase_filters_to_matching_chunks() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        insert_ready_document(&db, "doc1");
        let embedding = [0.1_f32, 0.2, 0.3, 0.4];
        db.insert_chunk(&NewChunk {
            id: "c1",
            document_id: "doc1",
            chunk_index: 0,
            content: "The flood exclusion applies to basements.",
            heading: None,
            page: None,
            embedding: Some(&embedding),
        })
        .unwrap_or_else(|e| fail(&e.to_string()));
        db.insert_chunk(&NewChunk {
            id: "c2",
            document_id: "doc1",
            chunk_index: 1,
            content: "Exclusion of flood risk is handled in a separate clause.",
            heading: None,
            page: None,
            embedding: Some(&embedding),
        })
        .unwrap_or_else(|e| fail(&e.to_string()));

        let results = db
            .search_hybrid_chunks(
                "\"flood exclusion\"",
                &embedding,
                10,
                60,
                &ChunkScope::all(),
            )
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(results.len(), 1);
        assert_eq!(results.first().map(|r| r.id.as_str()), Some("c1"));
    }

    /// Version 7 added joined identifier terms, so a workspace still
    /// recorded at an older version must reindex `_quack_terms` on open
    /// (`WorkspaceDb::create_internal_tables`).
    #[test]
    fn opening_an_older_workspace_reindexes_terms_for_identifier_search() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let config = config_in(dir.path());
        {
            let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
            insert_ready_document(&db, "doc1");
            insert_text_chunk(&db, "c1", "doc1", 0, "Policy POL-8841 applies.");
            // Simulate a workspace indexed before version 7: the joined
            // identifier term is missing and the recorded version rolls back.
            db.execute_statement("DELETE FROM _quack_terms WHERE term = 'pol8841'")
                .unwrap_or_else(|e| fail(&e.to_string()));
            db.set_meta_public("schema_version", "6")
                .unwrap_or_else(|e| fail(&e.to_string()));
        }

        let reopened = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            reopened
                .meta("schema_version")
                .unwrap_or_else(|e| fail(&e.to_string())),
            Some(String::from(WORKSPACE_SCHEMA_VERSION))
        );
        let rows = reopened
            .execute_query("SELECT count(*) FROM _quack_terms WHERE term = 'pol8841'")
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            rows.rows.first().and_then(|r| r.first()),
            Some(&serde_json::Value::Number(1.into())),
            "reopening should have rebuilt the term index with the joined form"
        );
    }
}
