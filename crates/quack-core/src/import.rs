//! External data import, the Rust-side replacement for `DuckDB`'s `ATTACH`
//! (design doc 6.2, issue #21): the scanner and httpfs extensions cannot
//! be compiled into the static binary, so rows are pulled here and loaded
//! as a workspace table through the same path a CSV upload takes. Sources:
//! Postgres and SQLite through sqlx (every column cast to text on the
//! source side, so any type comes through), and a CSV, Parquet, JSON, or
//! workbook file over HTTP(S) through reqwest. Credentials in the URL are
//! used once and never stored: the document row and the audit detail
//! carry the redacted URL.

use std::fmt::{self, Write as _};
use std::net::{IpAddr, SocketAddr};
use std::path::Path;
use std::time::Duration;

use futures::TryStreamExt as _;
use rig::embeddings::EmbeddingModel;
use sqlx::{
    AssertSqlSafe, Column, Connection as _, Executor as _, Row, SqlSafeStr as _, Statement as _,
};

use crate::config::Config;
use crate::embedding::Embedder;
use crate::error::{Error, Result};
use crate::ids::DocumentId;
use crate::ingestion::parser::{FileType, Load};
use crate::ingestion::{self, IngestOutcome, NewFile, TableName};
use crate::progress::RunControl;
use crate::storage::workspace::{DocumentSource, WorkspaceDb, quote_ident};
use crate::storage::writer::Writer;

/// What to import and where to put it.
#[derive(Debug, Clone)]
pub struct ImportRequest {
    pub url: SourceUrl,
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
    pub hosts: HostReach,
}

/// Which HTTP(S) hosts a download may reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostReach {
    /// Only hosts whose every address is public; the connection is pinned
    /// to the checked addresses and redirects are not followed, since a
    /// redirect's target would escape the check.
    PublicOnly,
    /// Any host, loopback, private, and link-local included; redirects are
    /// followed.
    Any,
}

impl ImportPolicy {
    /// Everything: the caller owns the host.
    #[must_use]
    pub const fn owner() -> Self {
        Self {
            local_files: true,
            hosts: HostReach::Any,
        }
    }

    /// What `[import]` grants a server user.
    #[must_use]
    pub fn server(config: &Config) -> Self {
        Self {
            local_files: config.import.allow_local_files,
            hosts: if config.import.allow_private_hosts {
                HostReach::Any
            } else {
                HostReach::PublicOnly
            },
        }
    }
}

/// One import: the configuration, the workspace's writer and id, what to
/// import, which sources the caller may reach, the embedding model (none
/// stores no vectors), and the control that can stop it.
pub struct Importing<'a, M> {
    pub config: &'a Config,
    pub db: &'a Writer,
    pub workspace_id: &'a str,
    pub request: &'a ImportRequest,
    pub policy: ImportPolicy,
    pub embedder: Option<&'a Embedder<M>>,
    pub control: RunControl<'a>,
}

/// What an import did.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ImportSummary {
    pub table: String,
    pub rows: u64,
    pub columns: Vec<String>,
    /// The source with any password removed.
    pub source: String,
    pub document_id: DocumentId,
}

/// The kind of source a URL names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceKind {
    Postgres,
    Sqlite,
    Http,
}

/// A source URL: `postgres://...`, `sqlite://path` or `sqlite:path`, or an
/// `http(s)://` URL of a data file. It can carry a password, so `Debug`
/// and `Display` show it redacted; only [`SourceUrl::expose`] gives the
/// whole of it, to connect with.
#[derive(Clone, PartialEq, Eq)]
pub struct SourceUrl(String);

impl From<String> for SourceUrl {
    fn from(url: String) -> Self {
        Self(url)
    }
}

impl From<&str> for SourceUrl {
    fn from(url: &str) -> Self {
        Self(url.to_owned())
    }
}

impl fmt::Debug for SourceUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SourceUrl").field(&self.redacted()).finish()
    }
}

/// The URL with the password of its user info removed.
impl fmt::Display for SourceUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.redacted())
    }
}

impl SourceUrl {
    /// The whole URL, password included, for the one call that connects.
    fn expose(&self) -> &str {
        &self.0
    }

    /// The kind of source this names.
    ///
    /// # Errors
    ///
    /// Returns an error for a scheme quack does not import from.
    pub fn kind(&self) -> Result<SourceKind> {
        let lower = self.0.trim().to_ascii_lowercase();
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
    pub fn redacted(&self) -> String {
        let url = self.0.as_str();
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

    /// The file a `sqlite:` URL names: the scheme and any `//` stripped,
    /// the query string dropped.
    fn sqlite_path(&self) -> &Path {
        let rest = self.0.trim().get("sqlite:".len()..).unwrap_or_default();
        let rest = rest.strip_prefix("//").unwrap_or(rest);
        Path::new(rest.split_once('?').map_or(rest, |(path, _)| path))
    }

    /// Whether a `sqlite:` URL points inside `data_dir`: the control
    /// database and the workspace files are quack's own, and nothing in
    /// them belongs in a workspace table, whoever asks.
    fn is_under(&self, data_dir: &Path) -> bool {
        let path = self.sqlite_path();
        if path.starts_with(data_dir) {
            return true;
        }
        match (std::fs::canonicalize(path), std::fs::canonicalize(data_dir)) {
            (Ok(p), Ok(d)) => p.starts_with(d),
            _ => false,
        }
    }
}

/// What a source yielded, ready to load: the staging file's name and
/// bytes, and, for a query source, its columns and row count.
struct Pulled {
    filename: String,
    bytes: Vec<u8>,
    /// Empty for a downloaded file, which is described after the load.
    columns: Vec<String>,
    /// `None` for a downloaded file, whose rows are capped after the load.
    rows: Option<u64>,
}

impl<M: EmbeddingModel> Importing<'_, M> {
    /// Pull the rows and load them as the request's table. The source's
    /// rows go through `files/<table>.csv` and `read_csv_auto`, so the
    /// table is registered as a document (source `import`) and can be
    /// deleted like any other.
    ///
    /// # Errors
    ///
    /// Returns an error when the URL is unsupported, the source cannot be
    /// reached or queried, or the load fails; [`Error::Cancelled`] when
    /// the control is cancelled first (a download or query in flight is
    /// abandoned).
    pub async fn run(self) -> Result<ImportSummary> {
        let (config, db, request) = (self.config, self.db, self.request);
        let table = TableName::given(&request.table)?;
        let limit = request
            .limit
            .unwrap_or(config.import.max_rows)
            .min(config.import.max_rows)
            .max(1);
        let source = request.url.redacted();
        let pulled = self.pull(&table, limit).await?;
        let outcome = ingestion::ingest_file(
            config,
            db,
            self.workspace_id,
            &NewFile::new(&pulled.filename, &pulled.bytes)
                .source(DocumentSource::Import)
                .title(Some(&source))
                .control(self.control),
            self.embedder,
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
        let KeptRows { columns, rows } = if let Some(rows) = pulled.rows {
            KeptRows {
                columns: pulled.columns,
                rows,
            }
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

    /// Fetch the source as a staging file for `table`: a download, or at
    /// most `limit` rows of a query written as CSV.
    async fn pull(&self, table: &TableName, limit: u64) -> Result<Pulled> {
        let (config, request, policy, control) =
            (self.config, self.request, self.policy, self.control);
        let timeout = config.import.timeout();
        match request.url.kind()? {
            SourceKind::Http => {
                let download = Download {
                    timeout,
                    max_mb: config.import.max_download_mb,
                    hosts: policy.hosts,
                };
                control
                    .or_cancelled(download.fetch(&request.url, table.as_str()))
                    .await
            }
            SourceKind::Sqlite if !policy.local_files => Err(Error::Ingestion(String::from(
                "files on the server's disk cannot be imported through the server; \
                 run `quack import` on the host, or set [import].allow_local_files",
            ))),
            SourceKind::Sqlite if request.url.is_under(&config.general.data_dir) => {
                Err(Error::Ingestion(String::from(
                    "quack's own data directory (control.db and the workspace files) \
                     cannot be imported into a workspace",
                )))
            }
            SourceKind::Postgres | SourceKind::Sqlite => {
                let sql = request.source_query()?;
                let fetch = async {
                    tokio::time::timeout(timeout, fetch_rows(request.url.expose(), &sql, limit))
                        .await
                        .map_err(|_| {
                            Error::Ingestion(format!(
                                "the source did not answer within {} s",
                                timeout.as_secs()
                            ))
                        })?
                };
                let fetched = control.or_cancelled(fetch).await?;
                Ok(Pulled {
                    filename: format!("{table}.csv"),
                    bytes: fetched.csv,
                    columns: fetched.columns,
                    rows: Some(fetched.rows),
                })
            }
        }
    }
}

/// A loaded table's columns and how many rows it kept.
struct KeptRows {
    columns: Vec<String>,
    rows: u64,
}

/// A file came in whole, so the row cap applies after the load, as
/// `--limit` does on a query source: the table keeps its first `limit`
/// rows.
fn cap_loaded_table(db: &WorkspaceDb, table: &str, limit: u64) -> Result<KeptRows> {
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
    Ok(KeptRows {
        columns: described.columns.into_iter().map(|c| c.name).collect(),
        rows,
    })
}

impl ImportRequest {
    /// The inner query the source runs: the caller's, or the whole source
    /// table.
    fn source_query(&self) -> Result<String> {
        match (&self.query, &self.source_table) {
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
}

/// Run `inner` on the source with every column cast to text and at most
/// `limit` rows; returns the column names and the rows as text cells.
/// What a query source yielded, already as CSV: rows are written as they
/// arrive, so the memory cost is the file once, not the rows as strings
/// plus the file (issue #62).
struct Fetched {
    columns: Vec<String>,
    csv: Vec<u8>,
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
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record(&columns)?;
    let mut rows = 0u64;
    let mut stream = sqlx::query(AssertSqlSafe(select)).fetch(&mut conn);
    while let Some(row) = stream
        .try_next()
        .await
        .map_err(|e| Error::Ingestion(format!("the source query failed: {e}")))?
    {
        let mut cells = Vec::with_capacity(columns.len());
        for i in 0..columns.len() {
            // NULL is an empty field, which `read_csv_auto` reads back as
            // NULL.
            let cell: Option<String> = row.try_get(i).map_err(|e| {
                Error::Ingestion(format!(
                    "cannot read column {}: {e}",
                    columns.get(i).map_or("?", String::as_str)
                ))
            })?;
            cells.push(cell.unwrap_or_default());
        }
        writer.write_record(&cells)?;
        rows = rows.saturating_add(1);
    }
    let csv = writer.into_inner().map_err(|e| Error::Io(e.into_error()))?;
    Ok(Fetched { columns, csv, rows })
}

/// How a download is bounded.
struct Download {
    timeout: Duration,
    max_mb: u64,
    hosts: HostReach,
}

impl Download {
    /// Download a data file; the workspace file name keeps the URL's
    /// extension so the usual reader loads it, under the requested table name.
    ///
    /// When only public hosts may be reached, the name is resolved first,
    /// every address is checked, and the connection is pinned to those
    /// addresses so a second lookup cannot answer differently.
    async fn fetch(&self, url: &SourceUrl, table: &str) -> Result<Pulled> {
        let download = self;
        let url = url.expose();
        // The same extensions `quack ingest` loads as tables.
        let extension = url
            .split(['?', '#'])
            .next()
            .and_then(|path| path.rsplit('/').next())
            .filter(|name| FileType::of(name).is_some_and(|t| !matches!(t.load(), Load::Chunks(_))))
            .and_then(|name| name.rsplit_once('.'))
            .map(|(_, ext)| ext.to_ascii_lowercase())
            .ok_or_else(|| {
                let accepted: Vec<String> = FileType::table_extensions()
                    .map(|e| format!(".{e}"))
                    .collect();
                Error::Ingestion(format!(
                    "the URL must name a table file ending in {}",
                    accepted.join(", ")
                ))
            })?;
        let parsed =
            reqwest::Url::parse(url).map_err(|e| Error::Ingestion(format!("bad URL: {e}")))?;
        let mut builder = reqwest::Client::builder().timeout(download.timeout);
        if download.hosts == HostReach::PublicOnly {
            let host = parsed
                .host_str()
                .ok_or_else(|| Error::Ingestion(String::from("the URL has no host")))?;
            let port = parsed
                .port_or_known_default()
                .ok_or_else(|| Error::Ingestion(String::from("the URL has no port")))?;
            let addresses = public_addresses(host, port).await?;
            builder = builder.resolve_to_addrs(host, &addresses);
        }
        if download.hosts == HostReach::PublicOnly {
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
        Ok(Pulled {
            filename: format!("{table}.{extension}"),
            bytes,
            columns: Vec::new(),
            rows: None,
        })
    }
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
    use crate::llm::Embeddings;

    #[test]
    fn urls_classify_and_redact() {
        assert_eq!(
            SourceUrl::from("postgres://u:p@h/db").kind().ok(),
            Some(SourceKind::Postgres)
        );
        assert_eq!(
            SourceUrl::from("sqlite:/tmp/x.db").kind().ok(),
            Some(SourceKind::Sqlite)
        );
        assert_eq!(
            SourceUrl::from("https://x/y.csv").kind().ok(),
            Some(SourceKind::Http)
        );
        assert!(SourceUrl::from("mysql://h/db").kind().is_err());
        assert!(SourceUrl::from("ftp://h/f").kind().is_err());
        assert_eq!(
            SourceUrl::from("postgres://alice:secret@db.local:5432/sales").redacted(),
            "postgres://alice:***@db.local:5432/sales"
        );
        assert_eq!(
            SourceUrl::from("postgres://db.local/sales").redacted(),
            "postgres://db.local/sales"
        );
        assert_eq!(
            SourceUrl::from("sqlite:/tmp/x.db").redacted(),
            "sqlite:/tmp/x.db"
        );
    }

    #[test]
    fn queries_come_from_the_request_and_tables_are_checked() {
        let base = ImportRequest {
            url: SourceUrl::from(""),
            table: String::from("t"),
            query: None,
            source_table: None,
            limit: None,
        };
        assert!(base.source_query().is_err());
        let by_table = ImportRequest {
            source_table: Some(String::from("public.orders")),
            ..base.clone()
        };
        assert_eq!(
            by_table.source_query().ok().as_deref(),
            Some("SELECT * FROM public.orders")
        );
        let bad = ImportRequest {
            source_table: Some(String::from("orders; DROP TABLE x")),
            ..base.clone()
        };
        assert!(bad.source_query().is_err());
        let by_query = ImportRequest {
            query: Some(String::from(" SELECT 1 AS n; ")),
            ..base
        };
        assert_eq!(
            by_query.source_query().ok().as_deref(),
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
            hosts: HostReach::PublicOnly,
        };
        let err = download
            .fetch(&SourceUrl::from("http://127.0.0.1:9/x.csv"), "x")
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("private address"), "{err}");
        let err = download
            .fetch(&SourceUrl::from("http://localhost:9/x.csv"), "x")
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("private address"), "{err}");
    }

    /// A URL names a file `quack ingest` loads as a table, or it is
    /// refused before any network use; a type the URL passes goes on to
    /// the host check, which refuses loopback here.
    #[tokio::test]
    async fn urls_take_every_extension_ingest_loads_as_a_table() {
        let download = Download {
            timeout: Duration::from_secs(5),
            max_mb: 1,
            hosts: HostReach::PublicOnly,
        };
        for accepted in [
            "http://127.0.0.1:9/book.ods",
            "http://127.0.0.1:9/old.XLS?download=1",
            "http://127.0.0.1:9/data.pq#part",
            "http://127.0.0.1:9/rows.ndjson",
        ] {
            let err = download
                .fetch(&SourceUrl::from(accepted), "t")
                .await
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(err.contains("private address"), "{accepted}: {err}");
        }
        for refused in ["http://127.0.0.1:9/notes.pdf", "http://127.0.0.1:9/data"] {
            let err = download
                .fetch(&SourceUrl::from(refused), "t")
                .await
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(
                err.contains("must name a table file") && err.contains(".ods"),
                "{refused}: {err}"
            );
        }
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
    async fn downloads_stop_at_the_byte_cap_and_the_owner_follows_redirects() {
        let owner = Download {
            timeout: Duration::from_secs(5),
            max_mb: 0,
            hosts: HostReach::Any,
        };
        let url = serve_once(
            "HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n\r\na,b\n1,2\n3,4\n",
        )
        .await;
        let err = owner
            .fetch(&SourceUrl::from(url.as_str()), "t")
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("max_download_mb"), "{err}");

        let url = serve_once(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n6\r\na,b\n1,\r\n6\r\n2\n3,4\n\r\n0\r\n\r\n",
        )
        .await;
        let err = owner
            .fetch(&SourceUrl::from(url.as_str()), "t")
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
        let fetched = roomy.fetch(&SourceUrl::from(url.as_str()), "t").await;
        assert!(
            fetched.is_ok_and(|p| p.filename == "t.csv" && p.bytes == b"a,b\n1,2\n3,4\n"),
            "the capped download of a small file succeeds"
        );

        // The owner may reach any host, so a redirect is followed: here to
        // a closed port, which fails the download itself.
        let url = serve_once(
            "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:9/other.csv\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )
        .await;
        let err = roomy
            .fetch(&SourceUrl::from(url.as_str()), "t")
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            err.contains("download failed") && !err.contains("302 Found"),
            "{err}"
        );
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
            url: url.into(),
            table: String::from("rows"),
            query: None,
            source_table: None,
            limit: Some(2),
        };
        let summary = Importing {
            config: &config,
            db: &db,
            workspace_id: "ws",
            request: &request,
            policy: ImportPolicy::owner(),
            embedder: None::<&Embeddings>,
            control: RunControl::unobserved(),
        }
        .run()
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
                url: url.clone().into(),
                table: String::from("x"),
                query: None,
                source_table: Some(String::from("users")),
                limit: None,
            };
            let err = Importing {
                config: &config,
                db: &db,
                workspace_id: "ws",
                request: &request,
                policy: ImportPolicy::owner(),
                embedder: None::<&Embeddings>,
                control: RunControl::unobserved(),
            }
            .run()
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
            assert!(err.contains("own data directory"), "{url}: {err}");
        }
        assert!(!SourceUrl::from("sqlite:/tmp/other.db").is_under(&config.general.data_dir));
        assert_eq!(
            SourceUrl::from("sqlite://a/b.db?x=1").sqlite_path(),
            Path::new("a/b.db")
        );
    }
}
