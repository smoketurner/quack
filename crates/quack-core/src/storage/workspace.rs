use std::io::Write;

use crate::config::Config;

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
        };
        db.create_internal_tables()?;
        Ok(db)
    }

    /// Open (or create) the `DuckDB` database for a workspace.
    ///
    /// Loads the vss extension and creates the internal schema tables
    /// (documents, chunks) if they do not exist. The `embedding_dimension`
    /// sets the fixed-size `FLOAT[N]` column width for vector storage.
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

        let embedding_dimension = config
            .find_embedding_provider()
            .and_then(|(_, p)| p.embedding_dimension)
            .unwrap_or(1024);

        let conn = duckdb::Connection::open(&db_path)?;

        let db = Self {
            conn,
            embedding_dimension,
        };
        db.load_vss();
        db.create_internal_tables()?;
        Ok(db)
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
        let dim = self.embedding_dimension;
        let sql = format!(
            "CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                mime_type TEXT,
                size_bytes BIGINT,
                ingested_at TIMESTAMP DEFAULT now(),
                status TEXT DEFAULT 'pending',
                error_message TEXT
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding FLOAT[{dim}],
                token_count INTEGER
            );"
        );
        self.conn.execute_batch(&sql)?;
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
            "INSERT INTO documents (id, filename, mime_type, size_bytes, status) VALUES (?, ?, ?, ?, ?)",
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
            "UPDATE documents SET status = ? WHERE id = ?",
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
                let emb_str = format_embedding(emb);
                let dim = self.embedding_dimension;
                let sql = format!(
                    "INSERT INTO chunks (id, document_id, chunk_index, content, embedding) \
                     VALUES (?, ?, ?, ?, {emb_str}::FLOAT[{dim}])"
                );
                self.conn
                    .execute(&sql, duckdb::params![id, document_id, chunk_index, content])?;
            }
            None => {
                self.conn.execute(
                    "INSERT INTO chunks (id, document_id, chunk_index, content) VALUES (?, ?, ?, ?)",
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
        let emb_str = format_embedding(embedding);
        let dim = self.embedding_dimension;
        let sql = format!(
            "UPDATE chunks SET embedding = {emb_str}::FLOAT[{dim}] \
             WHERE document_id = ? AND chunk_index = ?"
        );
        self.conn
            .execute(&sql, duckdb::params![document_id, chunk_index])?;
        Ok(())
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
             ON chunks USING HNSW (embedding) \
             WITH (metric = 'cosine')",
            [],
        )?;
        tracing::info!("created HNSW cosine index on chunks.embedding");
        Ok(())
    }

    /// Search for the most similar chunks to a query embedding.
    ///
    /// # Errors
    ///
    /// Returns an error if the search query fails.
    pub fn search_similar_chunks(
        &self,
        query_embedding: &[f32],
        top_k: u32,
    ) -> crate::error::Result<Vec<ChunkSearchResult>> {
        let emb_str = format_embedding(query_embedding);
        let dim = self.embedding_dimension;
        let sql = format!(
            "SELECT c.id, c.content, c.document_id, c.chunk_index, \
                    array_cosine_distance(c.embedding, {emb_str}::FLOAT[{dim}]) AS distance \
             FROM chunks c \
             WHERE c.embedding IS NOT NULL \
             ORDER BY distance ASC \
             LIMIT {top_k}"
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        let mut results = Vec::new();

        while let Some(row) = rows.next()? {
            results.push(ChunkSearchResult {
                id: row.get(0)?,
                content: row.get(1)?,
                document_id: row.get(2)?,
                chunk_index: row.get(3)?,
                distance: row.get(4)?,
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
        self.conn.execute(sql, [])?;
        Ok(())
    }

    /// List all user-created tables in the workspace (excludes internal tables).
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_tables(&self) -> crate::error::Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_name NOT IN ('documents', 'chunks')")?;
        let mut rows = stmt.query([])?;
        let mut tables = Vec::new();
        while let Some(row) = rows.next()? {
            tables.push(row.get::<_, String>(0)?);
        }
        Ok(tables)
    }

    /// Describe a table's columns (name, type) and return up to 3 sample rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist or the query fails.
    pub fn describe_table(&self, table_name: &str) -> crate::error::Result<TableDescription> {
        let describe_sql = format!("DESCRIBE \"{table_name}\"");
        let mut stmt = self.conn.prepare(&describe_sql)?;
        let mut rows = stmt.query([])?;
        let mut columns = Vec::new();
        while let Some(row) = rows.next()? {
            columns.push(ColumnInfo {
                name: row.get(0)?,
                column_type: row.get(1)?,
            });
        }

        let sample_sql = format!("SELECT * FROM \"{table_name}\" LIMIT 3");
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
            "SELECT id, filename, mime_type, size_bytes, status FROM documents ORDER BY ingested_at DESC",
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
    pub distance: f64,
}

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
