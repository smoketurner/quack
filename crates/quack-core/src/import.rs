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
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use futures::TryStreamExt as _;
use rig::embeddings::EmbeddingModel;
use sqlx::{
    AssertSqlSafe, Column, Connection as _, Executor as _, Row, SqlSafeStr as _, Statement as _,
};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::ingestion::{self, IngestOutcome, NewFile};
use crate::storage::workspace::{DocumentSource, WorkspaceDb, quote_ident};
use crate::storage::writer::Writer;
use tokio_util::sync::CancellationToken;

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

/// Which sources a caller may reach (issue #42). The owner's interfaces
/// (`quack import`, the terminal, `quack serve --local`) reach anything;
/// the server with logins is held to `[import]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImportPolicy {
    /// `sqlite:` paths on the local disk.
    pub local_files: bool,
    /// HTTP(S) hosts that resolve to loopback, private, or link-local
    /// addresses.
    pub private_hosts: bool,
}

impl ImportPolicy {
    /// Everything: the caller owns the host.
    #[must_use]
    pub const fn owner() -> Self {
        Self {
            local_files: true,
            private_hosts: true,
        }
    }

    /// What `[import]` grants a server user.
    #[must_use]
    pub fn server(config: &Config) -> Self {
        Self {
            local_files: config.import.allow_local_files,
            private_hosts: config.import.allow_private_hosts,
        }
    }
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

/// The file a `sqlite:` URL names: the scheme and any `//` stripped, the
/// query string dropped.
fn sqlite_path(url: &str) -> &str {
    let rest = url.trim().get("sqlite:".len()..).unwrap_or_default();
    let rest = rest.strip_prefix("//").unwrap_or(rest);
    rest.split_once('?').map_or(rest, |(path, _)| path)
}

/// Whether a `sqlite:` URL points inside `[general].data_dir`: the control
/// database and the workspace files are quack's own, and nothing in them
/// belongs in a workspace table, whoever asks (issue #69).
fn under_data_dir(config: &Config, url: &str) -> bool {
    let path = std::path::Path::new(sqlite_path(url));
    let data_dir = &config.general.data_dir;
    let inside = |p: &std::path::Path, d: &std::path::Path| p.starts_with(d);
    if inside(path, data_dir) {
        return true;
    }
    match (std::fs::canonicalize(path), std::fs::canonicalize(data_dir)) {
        (Ok(p), Ok(d)) => inside(&p, &d),
        _ => false,
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
/// reached or queried, or the load fails; [`Error::Cancelled`] when
/// `cancel` fires first (a download or query in flight is abandoned).
pub async fn import<M: EmbeddingModel>(
    config: &Config,
    db: &Writer,
    workspace_id: &str,
    request: &ImportRequest,
    policy: ImportPolicy,
    embedding_model: Option<&M>,
    cancel: Option<&CancellationToken>,
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
            let download = Download {
                timeout,
                max_mb: config.import.max_download_mb,
                private_hosts: policy.private_hosts,
                redirects: policy.private_hosts,
            };
            let (filename, bytes) =
                ingestion::or_cancelled(cancel, fetch_http(&request.url, &table, &download))
                    .await?;
            (filename, bytes, Vec::new(), None)
        }
        SourceKind::Sqlite if !policy.local_files => {
            return Err(Error::Ingestion(String::from(
                "files on the server's disk cannot be imported through the server; \
                 run `quack import` on the host, or set [import].allow_local_files",
            )));
        }
        SourceKind::Sqlite if under_data_dir(config, &request.url) => {
            return Err(Error::Ingestion(String::from(
                "quack's own data directory (control.db and the workspace files) \
                 cannot be imported into a workspace",
            )));
        }
        SourceKind::Postgres | SourceKind::Sqlite => {
            let sql = source_query(request)?;
            let fetch = async {
                tokio::time::timeout(timeout, fetch_rows(&request.url, &sql, limit))
                    .await
                    .map_err(|_| {
                        Error::Ingestion(format!(
                            "the source did not answer within {} s",
                            timeout.as_secs()
                        ))
                    })?
            };
            let fetched = ingestion::or_cancelled(cancel, fetch).await?;
            (
                format!("{table}.csv"),
                fetched.csv.into_bytes(),
                fetched.columns,
                Some(fetched.rows),
            )
        }
    };
    let outcome = ingestion::ingest_file(
        config,
        db,
        workspace_id,
        &NewFile::new(&filename, &bytes)
            .source(DocumentSource::Import)
            .title(Some(&source))
            .cancel(cancel),
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
    let (columns, rows) = if let Some(rows) = rows {
        (columns, rows)
    } else {
        let table = loaded.clone();
        db.run(move |db| cap_loaded_table(db, &table, limit))
            .await?
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

/// A file came in whole, so the row cap applies after the load, as
/// `--limit` does on a query source: the table keeps its first `limit`
/// rows. Returns the columns and the rows kept.
fn cap_loaded_table(db: &WorkspaceDb, table: &str, limit: u64) -> Result<(Vec<String>, u64)> {
    let described = db.describe_table(table)?;
    let mut rows = u64::try_from(described.row_count).unwrap_or(0);
    if rows > limit {
        db.execute_with_params(
            &format!("DELETE FROM {} WHERE rowid >= ?", quote_ident(table)),
            duckdb::params![limit],
        )?;
        tracing::info!(table, rows, limit, "cut the imported file to the row cap");
        rows = limit;
    }
    Ok((
        described.columns.into_iter().map(|c| c.name).collect(),
        rows,
    ))
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
/// What a query source yielded, already as CSV: rows are written as they
/// arrive, so the memory cost is the file once, not the rows as strings
/// plus the file (issue #62).
struct Fetched {
    columns: Vec<String>,
    csv: String,
    rows: u64,
}

async fn fetch_rows(url: &str, inner: &str, limit: u64) -> Result<Fetched> {
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
    let mut csv = Csv::with_header(&columns);
    let mut rows = 0u64;
    let mut stream = sqlx::query(AssertSqlSafe(select)).fetch(&mut conn);
    while let Some(row) = stream
        .try_next()
        .await
        .map_err(|e| Error::Ingestion(format!("the source query failed: {e}")))?
    {
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
        csv.row(&cells);
        rows = rows.saturating_add(1);
    }
    Ok(Fetched {
        columns,
        csv: csv.text,
        rows,
    })
}

/// CSV text built a row at a time; `DuckDB` sniffs the types back.
struct Csv {
    text: String,
}

impl Csv {
    fn with_header(columns: &[String]) -> Self {
        let mut csv = Self {
            text: String::new(),
        };
        csv.line(columns.iter().map(|c| Some(c.as_str())));
        csv
    }

    fn row(&mut self, cells: &[Option<String>]) {
        self.line(cells.iter().map(Option::as_deref));
    }

    fn line<'a>(&mut self, cells: impl Iterator<Item = Option<&'a str>>) {
        for (i, cell) in cells.enumerate() {
            if i > 0 {
                self.text.push(',');
            }
            let Some(value) = cell else {
                continue;
            };
            if value.contains([',', '"', '\n', '\r']) {
                self.text.push('"');
                self.text.push_str(&value.replace('"', "\"\""));
                self.text.push('"');
            } else {
                self.text.push_str(value);
            }
        }
        self.text.push('\n');
    }
}

/// How a download is bounded.
struct Download {
    timeout: Duration,
    max_mb: u64,
    private_hosts: bool,
    /// Follow redirects; off whenever hosts are checked, since a
    /// redirect's target would escape the check.
    redirects: bool,
}

/// Download a data file; the workspace file name keeps the URL's
/// extension so the usual reader loads it, under the requested table name.
///
/// When private hosts are off, the name is resolved first, every address
/// is checked, and the connection is pinned to those addresses so a
/// second lookup cannot answer differently.
async fn fetch_http(url: &str, table: &str, download: &Download) -> Result<(String, Vec<u8>)> {
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
    let parsed = reqwest::Url::parse(url).map_err(|e| Error::Ingestion(format!("bad URL: {e}")))?;
    let mut builder = reqwest::Client::builder().timeout(download.timeout);
    if !download.private_hosts {
        let host = parsed
            .host_str()
            .ok_or_else(|| Error::Ingestion(String::from("the URL has no host")))?;
        let port = parsed
            .port_or_known_default()
            .ok_or_else(|| Error::Ingestion(String::from("the URL has no port")))?;
        let addresses = public_addresses(host, port).await?;
        builder = builder.resolve_to_addrs(host, &addresses);
    }
    if !download.redirects {
        builder = builder.redirect(reqwest::redirect::Policy::none());
    }
    let client = builder
        .build()
        .map_err(|e| Error::Ingestion(e.to_string()))?;
    let mut response = client
        .get(parsed)
        .send()
        .await
        .map_err(|e| Error::Ingestion(format!("download failed: {e}")))?;
    if response.status().is_redirection() {
        return Err(Error::Ingestion(format!(
            "download failed: the server answered {}; import the URL it points to",
            response.status()
        )));
    }
    if !response.status().is_success() {
        return Err(Error::Ingestion(format!(
            "download failed: the server answered {}",
            response.status()
        )));
    }
    let max_bytes = download.max_mb.saturating_mul(1024 * 1024);
    let too_large = || {
        Error::Ingestion(format!(
            "the file is larger than [import].max_download_mb ({} MB)",
            download.max_mb
        ))
    };
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes)
    {
        return Err(too_large());
    }
    let mut bytes: Vec<u8> = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| Error::Ingestion(format!("download failed: {e}")))?
    {
        let next = u64::try_from(bytes.len().saturating_add(chunk.len())).unwrap_or(u64::MAX);
        if next > max_bytes {
            return Err(too_large());
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok((format!("{table}.{extension}"), bytes))
}

/// The addresses `host` resolves to, all of them public.
async fn public_addresses(host: &str, port: u16) -> Result<Vec<SocketAddr>> {
    let addresses: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| Error::Ingestion(format!("cannot resolve {host}: {e}")))?
        .collect();
    if addresses.is_empty() {
        return Err(Error::Ingestion(format!("cannot resolve {host}")));
    }
    if let Some(private) = addresses.iter().find(|a| is_private_address(a.ip())) {
        return Err(Error::Ingestion(format!(
            "{host} resolves to {}, a private address; the server does not import from \
             its own network (set [import].allow_private_hosts to allow it)",
            private.ip()
        )));
    }
    Ok(addresses)
}

/// Loopback, unspecified, link-local (cloud metadata lives at
/// 169.254.169.254), RFC 1918 and carrier-grade NAT ranges, multicast, and
/// their IPv6 counterparts, including IPv4-mapped addresses.
fn is_private_address(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, _, _] = v4.octets();
            v4.is_loopback()
                || v4.is_unspecified()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_multicast()
                || (a == 100 && (64..=127).contains(&b))
        }
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => is_private_address(IpAddr::V4(v4)),
            None => {
                v6.is_loopback()
                    || v6.is_unspecified()
                    || v6.is_unique_local()
                    || v6.is_unicast_link_local()
                    || v6.is_multicast()
            }
        },
    }
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
    fn private_addresses_are_recognized() {
        let private = [
            "127.0.0.1",
            "0.0.0.0",
            "10.1.2.3",
            "172.16.0.9",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "224.0.0.1",
            "::1",
            "::",
            "fd00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
        ];
        for text in private {
            let ip: IpAddr = text.parse().unwrap_or_else(|_| IpAddr::from([1, 1, 1, 1]));
            assert!(is_private_address(ip), "{text}");
        }
        let public = [
            "1.1.1.1",
            "8.8.8.8",
            "100.128.0.1",
            "2606:4700::1111",
            "::ffff:1.1.1.1",
        ];
        for text in public {
            let ip: IpAddr = text
                .parse()
                .unwrap_or_else(|_| IpAddr::from([127, 0, 0, 1]));
            assert!(!is_private_address(ip), "{text}");
        }
    }

    #[tokio::test]
    async fn loopback_hosts_are_refused_without_the_private_host_grant() {
        let download = Download {
            timeout: Duration::from_secs(5),
            max_mb: 1,
            private_hosts: false,
            redirects: false,
        };
        let err = fetch_http("http://127.0.0.1:9/x.csv", "x", &download)
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("private address"), "{err}");
        let err = fetch_http("http://localhost:9/x.csv", "x", &download)
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("private address"), "{err}");
    }

    /// One HTTP exchange on a loopback port: answer `response` to the
    /// first connection and return the URL to fetch.
    async fn serve_once(response: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap_or_else(|e| unreachable_bind(&e.to_string()));
        let port = listener.local_addr().map(|a| a.port()).unwrap_or_default();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut buf = [0_u8; 4096];
            drop(socket.read(&mut buf).await);
            drop(socket.write_all(response.as_bytes()).await);
            drop(socket.shutdown().await);
        });
        format!("http://127.0.0.1:{port}/data.csv")
    }

    #[expect(clippy::panic, reason = "test helper: a loopback port must bind")]
    fn unreachable_bind(msg: &str) -> tokio::net::TcpListener {
        panic!("cannot bind a loopback port: {msg}")
    }

    #[tokio::test]
    async fn downloads_stop_at_the_byte_cap_and_do_not_follow_redirects() {
        let owner = Download {
            timeout: Duration::from_secs(5),
            max_mb: 0,
            private_hosts: true,
            redirects: false,
        };
        let url = serve_once(
            "HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n\r\na,b\n1,2\n3,4\n",
        )
        .await;
        let err = fetch_http(&url, "t", &owner)
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("max_download_mb"), "{err}");

        let url = serve_once(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n6\r\na,b\n1,\r\n6\r\n2\n3,4\n\r\n0\r\n\r\n",
        )
        .await;
        let err = fetch_http(&url, "t", &owner)
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("max_download_mb"), "{err}");

        let roomy = Download { max_mb: 1, ..owner };
        let url = serve_once(
            "HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n\r\na,b\n1,2\n3,4\n",
        )
        .await;
        let (name, bytes) = fetch_http(&url, "t", &roomy)
            .await
            .unwrap_or_else(|e| (e.to_string(), Vec::new()));
        assert_eq!(name, "t.csv");
        assert_eq!(bytes, b"a,b\n1,2\n3,4\n");

        let url = serve_once(
            "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:9/other.csv\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )
        .await;
        let err = fetch_http(&url, "t", &roomy)
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("302 Found"), "{err}");
    }

    /// A file source has no query to limit: the cap applies after the
    /// load and the table keeps the first `limit` rows.
    #[tokio::test]
    async fn downloaded_files_are_cut_to_the_row_cap() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| no_tempdir(&e.to_string()));
        let mut config = Config::default();
        config.general.data_dir = dir.path().to_path_buf();
        let db = open_writer(&config);
        let url = serve_once(
            "HTTP/1.1 200 OK\r\nContent-Length: 16\r\nConnection: close\r\n\r\nn,s\n1,a\n2,b\n3,c\n",
        )
        .await;
        let request = ImportRequest {
            url,
            table: String::from("rows"),
            query: None,
            source_table: None,
            limit: Some(2),
        };
        let summary = import(
            &config,
            &db,
            "ws",
            &request,
            ImportPolicy::owner(),
            None::<&crate::llm::EmbedModel>,
            None,
        )
        .await
        .unwrap_or_else(|e| no_import(&e.to_string()));
        assert_eq!(summary.table, "rows");
        assert_eq!(summary.rows, 2);
        assert_eq!(summary.columns, vec![String::from("n"), String::from("s")]);
        let kept = db
            .run(|db| db.execute_query("SELECT n FROM rows ORDER BY n"))
            .await
            .map(|r| r.rows.len())
            .unwrap_or_default();
        assert_eq!(kept, 2);
    }

    #[expect(clippy::panic, reason = "test helper: a temp dir must exist")]
    fn no_tempdir(msg: &str) -> tempfile::TempDir {
        panic!("cannot create a temp dir: {msg}")
    }

    #[expect(clippy::panic, reason = "test helper: the fixture files must exist")]
    fn no_file(msg: &str) {
        panic!("cannot write a fixture file: {msg}")
    }

    #[expect(clippy::panic, reason = "test helper: the workspace must open")]
    fn open_writer(config: &Config) -> Writer {
        WorkspaceDb::open(config, "ws")
            .and_then(Writer::spawn)
            .unwrap_or_else(|e| panic!("cannot open the workspace: {e}"))
    }

    #[expect(clippy::panic, reason = "test asserts Ok")]
    fn no_import(msg: &str) -> ImportSummary {
        panic!("import failed: {msg}")
    }

    /// The control database and workspace files stay out of workspaces
    /// even for the owner (issue #69), through every spelling of the URL;
    /// a symlink into the data directory does not slip past.
    #[tokio::test]
    async fn sqlite_imports_refuse_quacks_own_data_directory() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| no_tempdir(&e.to_string()));
        let mut config = Config::default();
        config.general.data_dir = dir.path().join("data");
        std::fs::create_dir_all(&config.general.data_dir)
            .unwrap_or_else(|e| no_file(&e.to_string()));
        let control = config.general.data_dir.join("control.db");
        std::fs::write(&control, b"").unwrap_or_else(|e| no_file(&e.to_string()));
        let link = dir.path().join("elsewhere.db");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&control, &link).unwrap_or_else(|e| no_file(&e.to_string()));
        let db = open_writer(&config);
        let control = control.display();
        let mut urls = vec![
            format!("sqlite:{control}"),
            format!("sqlite://{control}"),
            format!("SQLite:{control}?mode=ro"),
        ];
        if cfg!(unix) {
            urls.push(format!("sqlite:{}", link.display()));
        }
        for url in urls {
            let request = ImportRequest {
                url: url.clone(),
                table: String::from("x"),
                query: None,
                source_table: Some(String::from("users")),
                limit: None,
            };
            let err = import(
                &config,
                &db,
                "ws",
                &request,
                ImportPolicy::owner(),
                None::<&crate::llm::EmbedModel>,
                None,
            )
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
            assert!(err.contains("own data directory"), "{url}: {err}");
        }
        assert!(!under_data_dir(&config, "sqlite:/tmp/other.db"));
        assert_eq!(sqlite_path("sqlite://a/b.db?x=1"), "a/b.db");
    }

    #[test]
    fn csv_quotes_and_leaves_nulls_empty() {
        let mut csv = Csv::with_header(&[String::from("a"), String::from("b,c")]);
        csv.row(&[Some(String::from("x\"y")), None]);
        assert_eq!(csv.text, "a,\"b,c\"\n\"x\"\"y\",\n");
    }
}
