//! External data import, the Rust-side replacement for `DuckDB`'s `ATTACH`
//! (design doc 6.2, issue #21): the scanner and httpfs extensions cannot
//! be compiled into the static binary, so rows are pulled here and loaded
//! as a workspace table through the same path a CSV upload takes. Sources:
//! Postgres and SQLite through sqlx (every column cast to text on the
//! source side, so any type comes through), and a CSV, Parquet, JSON, or
//! workbook file over HTTP(S) through reqwest. Credentials in the URL are
//! used once and never stored: the document row and the audit detail
//! carry the redacted URL.

use std::fmt::Write as _;
use std::time::Duration;

use rig::embeddings::EmbeddingModel;
use sqlx::{
    AssertSqlSafe, Column, Connection as _, Executor as _, Row, SqlSafeStr as _, Statement as _,
};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::ingestion::{self, DbHandle, IngestOutcome, NewFile};
use crate::storage::workspace::DocumentSource;

/// What to import and where to put it.
#[derive(Debug, Clone)]
pub struct ImportRequest {
    /// `postgres://...`, `sqlite://path` or `sqlite:path`, or an
    /// `http(s)://` URL of a data file.
    pub url: String,
    /// The workspace table to create (replaced when it exists).
    pub table: String,
    /// A query to run on the source, else `SELECT * FROM source_table`.
    pub query: Option<String>,
    pub source_table: Option<String>,
    /// Rows to pull at most; capped by `[import].max_rows`.
    pub limit: Option<u64>,
}

/// What an import did.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ImportSummary {
    pub table: String,
    pub rows: u64,
    pub columns: Vec<String>,
    /// The source with any password removed.
    pub source: String,
    pub document_id: String,
}

/// The kind of source a URL names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceKind {
    Postgres,
    Sqlite,
    Http,
}

/// Classify a source URL.
///
/// # Errors
///
/// Returns an error for a scheme quack does not import from.
pub fn source_kind(url: &str) -> Result<SourceKind> {
    let lower = url.trim().to_ascii_lowercase();
    if lower.starts_with("postgres://") || lower.starts_with("postgresql://") {
        Ok(SourceKind::Postgres)
    } else if lower.starts_with("sqlite:") {
        Ok(SourceKind::Sqlite)
    } else if lower.starts_with("http://") || lower.starts_with("https://") {
        Ok(SourceKind::Http)
    } else if lower.starts_with("mysql://") || lower.starts_with("s3://") {
        Err(Error::Ingestion(format!(
            "{} sources are not supported yet; Postgres, SQLite, and HTTP(S) files are",
            lower.split("://").next().unwrap_or("such")
        )))
    } else {
        Err(Error::Ingestion(String::from(
            "the source must be a postgres://, sqlite:, or http(s):// URL",
        )))
    }
}

/// The URL with the password of its user info removed.
#[must_use]
pub fn redact(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_owned();
    };
    let Some((userinfo, host)) = rest.split_once('@') else {
        return url.to_owned();
    };
    let user = userinfo.split_once(':').map_or(userinfo, |(u, _)| u);
    if user.is_empty() {
        format!("{scheme}://{host}")
    } else {
        format!("{scheme}://{user}:***@{host}")
    }
}

/// A table name for the workspace: `[A-Za-z0-9_]` runs, else `_`.
fn table_name(raw: &str) -> Result<String> {
    let name = ingestion::table_name_for(raw);
    if name.is_empty() || name == "imported" && raw.trim().is_empty() {
        return Err(Error::Ingestion(String::from("a table name is needed")));
    }
    Ok(name)
}

/// Pull the rows and load them as `request.table`. The source's rows go
/// through `files/<table>.csv` and `read_csv_auto`, so the table is
/// registered as a document (source `import`) and can be deleted like
/// any other.
///
/// # Errors
///
/// Returns an error when the URL is unsupported, the source cannot be
/// reached or queried, or the load fails.
pub async fn import<M: EmbeddingModel, D: DbHandle>(
    config: &Config,
    db: &D,
    workspace_id: &str,
    request: &ImportRequest,
    embedding_model: Option<&M>,
) -> Result<ImportSummary> {
    let table = table_name(&request.table)?;
    let limit = request
        .limit
        .unwrap_or(config.import.max_rows)
        .min(config.import.max_rows)
        .max(1);
    let timeout = Duration::from_secs(config.import.timeout_seconds.max(1));
    let source = redact(&request.url);
    let (filename, bytes, columns, rows) = match source_kind(&request.url)? {
        SourceKind::Http => {
            let (filename, bytes) = fetch_http(&request.url, &table, timeout).await?;
            (filename, bytes, Vec::new(), None)
        }
        SourceKind::Postgres | SourceKind::Sqlite => {
            let sql = source_query(request)?;
            let (columns, records) =
                tokio::time::timeout(timeout, fetch_rows(&request.url, &sql, limit))
                    .await
                    .map_err(|_| {
                        Error::Ingestion(format!(
                            "the source did not answer within {} s",
                            timeout.as_secs()
                        ))
                    })??;
            let rows = u64::try_from(records.len()).unwrap_or(u64::MAX);
            (
                format!("{table}.csv"),
                csv_bytes(&columns, &records),
                columns,
                Some(rows),
            )
        }
    };
    let outcome = ingestion::ingest_file(
        config,
        db,
        workspace_id,
        &NewFile::new(&filename, &bytes)
            .source(DocumentSource::Import)
            .title(Some(&source)),
        embedding_model,
    )
    .await?;
    let result = match outcome {
        IngestOutcome::Ingested(result) => result,
        IngestOutcome::Duplicate(existing) => {
            return Err(Error::Ingestion(format!(
                "the source's rows are identical to document {} ({}); delete it first to reload",
                existing.id, existing.filename
            )));
        }
    };
    let loaded = result
        .tables
        .first()
        .cloned()
        .ok_or_else(|| Error::Ingestion(String::from("the import produced no table")))?;
    let (columns, rows) = match rows {
        Some(rows) => (columns, rows),
        None => db.with(|db| {
            let described = db.describe_table(&loaded)?;
            Ok((
                described.columns.into_iter().map(|c| c.name).collect(),
                u64::try_from(described.row_count).unwrap_or(0),
            ))
        })?,
    };
    tracing::info!(table = %loaded, rows, source = %source, "imported external data");
    Ok(ImportSummary {
        table: loaded,
        rows,
        columns,
        source,
        document_id: result.document_id,
    })
}

/// The inner query the source runs: the caller's, or the whole source
/// table.
fn source_query(request: &ImportRequest) -> Result<String> {
    match (&request.query, &request.source_table) {
        (Some(query), _) if !query.trim().is_empty() => {
            Ok(query.trim().trim_end_matches(';').to_owned())
        }
        (_, Some(table)) if !table.trim().is_empty() => {
            let name = table.trim();
            if !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
            {
                return Err(Error::Ingestion(format!(
                    "source table '{name}' must be a plain identifier (letters, digits, _, .)"
                )));
            }
            Ok(format!("SELECT * FROM {name}"))
        }
        _ => Err(Error::Ingestion(String::from(
            "give a query or a source table",
        ))),
    }
}

/// Run `inner` on the source with every column cast to text and at most
/// `limit` rows; returns the column names and the rows as text cells.
async fn fetch_rows(
    url: &str,
    inner: &str,
    limit: u64,
) -> Result<(Vec<String>, Vec<Vec<Option<String>>>)> {
    sqlx::any::install_default_drivers();
    let mut conn = sqlx::AnyConnection::connect(url)
        .await
        .map_err(|e| Error::Ingestion(format!("cannot connect to the source: {e}")))?;
    let probe = format!("SELECT * FROM ({inner}) AS quack_q LIMIT 0");
    let prepared = (&mut conn)
        .prepare(AssertSqlSafe(probe).into_sql_str())
        .await
        .map_err(|e| Error::Ingestion(format!("the source rejected the query: {e}")))?;
    let columns: Vec<String> = prepared
        .columns()
        .iter()
        .map(|c| c.name().to_owned())
        .collect();
    if columns.is_empty() {
        return Err(Error::Ingestion(String::from(
            "the query returns no columns",
        )));
    }
    let mut select = String::from("SELECT ");
    for (i, column) in columns.iter().enumerate() {
        if i > 0 {
            select.push_str(", ");
        }
        let quoted = format!("\"{}\"", column.replace('"', "\"\""));
        write!(select, "CAST({quoted} AS TEXT) AS {quoted}")?;
    }
    write!(select, " FROM ({inner}) AS quack_q LIMIT {limit}")?;
    let rows = sqlx::query(AssertSqlSafe(select))
        .fetch_all(&mut conn)
        .await
        .map_err(|e| Error::Ingestion(format!("the source query failed: {e}")))?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let mut cells = Vec::with_capacity(columns.len());
        for i in 0..columns.len() {
            let cell: Option<String> = row.try_get(i).map_err(|e| {
                Error::Ingestion(format!(
                    "cannot read column {}: {e}",
                    columns.get(i).map_or("?", String::as_str)
                ))
            })?;
            cells.push(cell);
        }
        out.push(cells);
    }
    Ok((columns, out))
}

/// Rows as CSV with a header; `DuckDB` sniffs the types back.
fn csv_bytes(columns: &[String], rows: &[Vec<Option<String>>]) -> Vec<u8> {
    let field = |value: &str| -> String {
        if value.contains([',', '"', '\n', '\r']) {
            format!("\"{}\"", value.replace('"', "\"\""))
        } else {
            value.to_owned()
        }
    };
    let mut text = columns
        .iter()
        .map(|c| field(c))
        .collect::<Vec<_>>()
        .join(",");
    text.push('\n');
    for row in rows {
        let line = row
            .iter()
            .map(|cell| cell.as_deref().map_or(String::new(), field))
            .collect::<Vec<_>>()
            .join(",");
        text.push_str(&line);
        text.push('\n');
    }
    text.into_bytes()
}

/// Download a data file; the workspace file name keeps the URL's
/// extension so the usual reader loads it, under the requested table name.
async fn fetch_http(url: &str, table: &str, timeout: Duration) -> Result<(String, Vec<u8>)> {
    let extension = url
        .split('?')
        .next()
        .and_then(|path| path.rsplit('/').next())
        .and_then(|name| {
            name.rsplit_once('.')
                .map(|(_, ext)| ext.to_ascii_lowercase())
        })
        .filter(|ext| {
            matches!(
                ext.as_str(),
                "csv" | "tsv" | "parquet" | "json" | "jsonl" | "ndjson" | "xlsx"
            )
        })
        .ok_or_else(|| {
            Error::Ingestion(String::from(
                "the URL must end in .csv, .tsv, .parquet, .json, .jsonl, or .xlsx",
            ))
        })?;
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| Error::Ingestion(e.to_string()))?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| Error::Ingestion(format!("download failed: {e}")))?;
    if !response.status().is_success() {
        return Err(Error::Ingestion(format!(
            "download failed: the server answered {}",
            response.status()
        )));
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|e| Error::Ingestion(format!("download failed: {e}")))?;
    Ok((format!("{table}.{extension}"), bytes.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_classify_and_redact() {
        assert_eq!(
            source_kind("postgres://u:p@h/db").ok(),
            Some(SourceKind::Postgres)
        );
        assert_eq!(
            source_kind("sqlite:/tmp/x.db").ok(),
            Some(SourceKind::Sqlite)
        );
        assert_eq!(source_kind("https://x/y.csv").ok(), Some(SourceKind::Http));
        assert!(source_kind("mysql://h/db").is_err());
        assert!(source_kind("ftp://h/f").is_err());
        assert_eq!(
            redact("postgres://alice:secret@db.local:5432/sales"),
            "postgres://alice:***@db.local:5432/sales"
        );
        assert_eq!(
            redact("postgres://db.local/sales"),
            "postgres://db.local/sales"
        );
        assert_eq!(redact("sqlite:/tmp/x.db"), "sqlite:/tmp/x.db");
    }

    #[test]
    fn queries_come_from_the_request_and_tables_are_checked() {
        let base = ImportRequest {
            url: String::new(),
            table: String::from("t"),
            query: None,
            source_table: None,
            limit: None,
        };
        assert!(source_query(&base).is_err());
        let by_table = ImportRequest {
            source_table: Some(String::from("public.orders")),
            ..base.clone()
        };
        assert_eq!(
            source_query(&by_table).ok().as_deref(),
            Some("SELECT * FROM public.orders")
        );
        let bad = ImportRequest {
            source_table: Some(String::from("orders; DROP TABLE x")),
            ..base.clone()
        };
        assert!(source_query(&bad).is_err());
        let by_query = ImportRequest {
            query: Some(String::from(" SELECT 1 AS n; ")),
            ..base
        };
        assert_eq!(
            source_query(&by_query).ok().as_deref(),
            Some("SELECT 1 AS n")
        );
    }

    #[test]
    fn csv_quotes_and_leaves_nulls_empty() {
        let bytes = csv_bytes(
            &[String::from("a"), String::from("b,c")],
            &[vec![Some(String::from("x\"y")), None]],
        );
        assert_eq!(String::from_utf8_lossy(&bytes), "a,\"b,c\"\n\"x\"\"y\",\n");
    }
}
