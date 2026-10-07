//! External data import, the Rust-side replacement for `DuckDB`'s `ATTACH`
//! (design doc 6.2, issue #21): the scanner and httpfs extensions cannot
//! be compiled into the static binary, so rows are pulled here and loaded
//! as a workspace table through the same path a CSV upload takes. Sources:
//! a SQLite file through sqlx, opened read-only (every column cast to text
//! on the source side, so any type comes through), and a CSV, Parquet,
//! JSON, or workbook file over HTTP(S) through reqwest, with any headers
//! the caller gives, or from Amazon S3 (`s3`). Credentials in the URL or a
//! header are used once and never stored: the document row and the audit
//! detail carry the redacted URL and the headers' names.

use std::fmt::{self, Write as _};
use std::net::{IpAddr, SocketAddr};
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use http::{HeaderName, HeaderValue, StatusCode};

use crate::embedding::EmbeddingModel;
use futures::TryStreamExt as _;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::{
    AssertSqlSafe, Column, ConnectOptions as _, Executor as _, Row, SqlSafeStr as _, Statement as _,
};

use crate::config::Config;
use crate::embedding::Embedder;
use crate::error::{Error, Result};
use crate::ids::DocumentId;
use crate::ingestion::parser::FileType;
use crate::ingestion::{self, IngestOutcome, NewFile, TableName};
use crate::progress::RunControl;
use crate::proxy::{Proxies, Route};
use crate::storage::profile::{ColumnTypes, TableProfile};
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
    /// Types to give the loaded table's columns.
    pub types: ColumnTypes,
    /// Headers an HTTP(S) download sends.
    pub headers: Vec<SourceHeader>,
    /// Where a JSON download's rows sit, when an envelope wraps them.
    pub json_pointer: Option<JsonPointer>,
    /// The document a refresh replaces: it keeps serving until the new
    /// rows are ready, and identical rows leave it in place.
    pub replaces: Option<DocumentId>,
}

/// One header an HTTP(S) download sends. `Debug` shows only the name, so a
/// token never reaches a log.
#[derive(Clone)]
pub enum SourceHeader {
    /// A header sent as given: `Name: value`.
    Given {
        name: HeaderName,
        value: HeaderValue,
    },
    /// `Authorization: Bearer` with a token read from this process's
    /// environment variable when the download starts, never stored.
    BearerEnv(String),
}

impl fmt::Debug for SourceHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Given { name, .. } => write!(f, "{name}: ***"),
            Self::BearerEnv(variable) => write!(f, "authorization: Bearer ${variable}"),
        }
    }
}

/// `Name: value`, as `curl -H` takes it.
impl FromStr for SourceHeader {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        let (name, value) = text
            .split_once(':')
            .ok_or_else(|| Error::Ingestion(String::from("a header is NAME: VALUE")))?;
        let name = HeaderName::from_str(name.trim())
            .map_err(|e| Error::Ingestion(format!("bad header name: {e}")))?;
        let mut value = HeaderValue::from_str(value.trim())
            .map_err(|_| Error::Ingestion(format!("bad value for header {name}")))?;
        value.set_sensitive(true);
        Ok(Self::Given { name, value })
    }
}

impl SourceHeader {
    /// The header's name, which is all an audit row records.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Given { name, .. } => name.as_str(),
            Self::BearerEnv(_) => "authorization",
        }
    }

    /// The header to send; `variable` looks up an environment variable.
    fn resolve(
        &self,
        variable: impl Fn(&str) -> Option<String>,
    ) -> Result<(HeaderName, HeaderValue)> {
        match self {
            Self::Given { name, value } => Ok((name.clone(), value.clone())),
            Self::BearerEnv(name) => {
                let token = variable(name)
                    .filter(|token| !token.trim().is_empty())
                    .ok_or_else(|| Error::Ingestion(format!("{name} is not set")))?;
                let mut value = HeaderValue::from_str(&format!("Bearer {}", token.trim()))
                    .map_err(|_| Error::Ingestion(format!("{name} is not a usable token")))?;
                value.set_sensitive(true);
                Ok((http::header::AUTHORIZATION, value))
            }
        }
    }
}

/// Where a JSON document's rows sit (RFC 6901): `/data/items` for
/// `{"data": {"items": [...]}}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonPointer(String);

impl FromStr for JsonPointer {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        let text = text.trim();
        if !text.is_empty() && !text.starts_with('/') {
            return Err(Error::Ingestion(format!(
                "a JSON pointer starts with / (RFC 6901), as /data/items; got {text}"
            )));
        }
        Ok(Self(text.to_owned()))
    }
}

impl fmt::Display for JsonPointer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl JsonPointer {
    /// The array of rows this points at inside `document`, as JSON.
    fn rows(&self, document: &[u8]) -> Result<Vec<u8>> {
        let value: serde_json::Value = serde_json::from_slice(document)
            .map_err(|e| Error::Ingestion(format!("the download is not JSON: {e}")))?;
        let rows = value
            .pointer(&self.0)
            .ok_or_else(|| Error::Ingestion(format!("the JSON has nothing at {self}")))?;
        if !rows.is_array() {
            return Err(Error::Ingestion(format!(
                "the JSON at {self} is not an array of rows"
            )));
        }
        Ok(serde_json::to_vec(rows)?)
    }
}

/// Which sources a caller may reach (issue #42). The owner's interfaces
/// (`quack import`, the terminal, `quack serve --local`) reach anything;
/// the server with logins is held to `[import]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImportPolicy {
    /// `sqlite:` paths on the local disk.
    pub local_files: bool,
    pub hosts: HostReach,
    pub credentials: CredentialReach,
}

/// Whether an import may authenticate as this process: S3 with its AWS
/// identity, or a bearer token from its environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialReach {
    /// The process's own credentials.
    Process,
    /// Only what the caller gives in the URL or a header.
    CallerOnly,
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
            credentials: CredentialReach::Process,
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
            credentials: if config.import.allow_server_credentials {
                CredentialReach::Process
            } else {
                CredentialReach::CallerOnly
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
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, utoipa::ToSchema)]
pub struct ImportSummary {
    pub table: String,
    pub rows: u64,
    pub columns: Vec<String>,
    /// The source with any password removed.
    pub source: String,
    pub document_id: DocumentId,
    pub status: LoadStatus,
}

/// Whether an import loaded rows, or found its source as it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum LoadStatus {
    Loaded,
    /// A refresh found the same bytes as the document it replaces, and
    /// left that document and its table in place.
    Unchanged,
}

/// The kind of source a URL names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Sqlite,
    Http,
    S3,
}

/// A source URL: `sqlite://path` or `sqlite:path`, an `http(s)://` URL of a
/// data file, or `s3://bucket/key`. It can carry a password, so `Debug`
/// and `Display` show it redacted; only `SourceUrl::expose` gives the
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
        if lower.starts_with("sqlite:") {
            Ok(SourceKind::Sqlite)
        } else if lower.starts_with("http://") || lower.starts_with("https://") {
            Ok(SourceKind::Http)
        } else if lower.starts_with("s3://") {
            Ok(SourceKind::S3)
        } else {
            Err(Error::Ingestion(String::from(
                "the source must be a sqlite: file, an http(s):// data file, or s3://bucket/key",
            )))
        }
    }

    /// The URL with the password of its user info replaced by `***`.
    ///
    /// A real URL parser splits the user info from the host at the *last*
    /// `@` before the path (RFC 3986), so a raw `@` inside the password no
    /// longer leaks its suffix, and an `@` in the path or query of a URL
    /// with no user info no longer inserts a spurious `:***`. URLs with no
    /// user info (sqlite paths, plain `https://host/file`) are returned
    /// verbatim. Anything `Url::parse` rejects (a raw `/`, `?`, or `#` in
    /// the password, an invalid port) never comes back verbatim: everything
    /// between `://` and the last `@` of the whole string is masked, keeping
    /// only a username that is plainly one (see `mask_userinfo`), so the
    /// redacted value never carries the password into audit rows, titles,
    /// or logs.
    #[must_use]
    pub fn redacted(&self) -> String {
        let url = self.0.as_str();
        let Ok(mut parsed) = reqwest::Url::parse(url) else {
            return mask_userinfo(url);
        };
        if parsed.username().is_empty() && parsed.password().is_none() {
            // No credentials to redact: keep the URL verbatim so sqlite
            // paths and `@`-in-path/query URLs are unchanged (the old
            // first-`@` split corrupted the latter with a spurious `:***`).
            return url.to_owned();
        }
        // A real parser splits the user info from the host at the *last*
        // `@` before the path (RFC 3986), and percent-encodes any raw `@`
        // in the password, so the redaction below never confuses one with
        // the other the way the old `split_once('@')` on the raw string did.
        if parsed.username().is_empty() {
            // `:password@` with no username: the old redaction dropped the
            // whole user info. The rendered URL has exactly one authority
            // `@`, so split there and keep the rest verbatim.
            return drop_userinfo(parsed.as_str());
        }
        // Username present: replace the whole password with `***`. The
        // parser already percent-encoded any `@` in it, so the marker
        // supplants the entire value; `set_password` needs an authority and
        // a username to succeed, both of which hold here.
        if parsed.set_password(Some("***")).is_ok() {
            return parsed.to_string();
        }
        mask_userinfo(url)
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

/// Drop the user info from a parsed URL's rendered form. `Url::parse`
/// percent-encodes any `@` in the password, so the first `@` after the
/// scheme is the one authority separator; everything up to it is the
/// user info and is discarded, keeping the host, port, path, query, and
/// fragment intact. Anything without that shape round-trips unchanged.
fn drop_userinfo(rendered: &str) -> String {
    let Some((scheme, rest)) = rendered.split_once("://") else {
        return rendered.to_owned();
    };
    let Some((_userinfo, tail)) = rest.split_once('@') else {
        return mask_userinfo(rendered);
    };
    format!("{scheme}://{tail}")
}

/// Mask the credentials of a URL no parser accepted, so none of it can leak.
///
/// Without a parse there is no telling where a password with a raw `/`, `?`,
/// `#`, or `@` ends, so everything between `://` and the *last* `@` of the
/// whole string is treated as user info: it may over-mask a path or query
/// that holds an `@`, but it never leaves a password fragment behind. The
/// username is kept only when the text before the first `:` is non-empty and
/// holds none of `/?#@` (so it cannot be the tail of a password or a path);
/// otherwise the whole user info becomes `***`. A string with no `://` or no
/// `@` after it carries no user info and is returned unchanged.
fn mask_userinfo(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_owned();
    };
    let Some((userinfo, host)) = rest.rsplit_once('@') else {
        return url.to_owned();
    };
    let user = userinfo.split_once(':').map_or(userinfo, |(user, _)| user);
    if user.is_empty() || user.contains(['/', '?', '#', '@']) {
        format!("{scheme}://***@{host}")
    } else {
        format!("{scheme}://{user}:***@{host}")
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
        table.check_unreserved()?;
        let limit = request
            .limit
            .unwrap_or(config.import.max_rows)
            .min(config.import.max_rows)
            .max(1);
        let source = request.url.redacted();
        request.check(request.url.kind()?, self.policy)?;
        let mut pulled = self.pull(&table, limit).await?;
        if let Some(pointer) = &request.json_pointer {
            pulled.bytes = pointer.rows(&pulled.bytes)?;
        }
        let outcome = ingestion::ingest_file(
            config,
            db,
            self.workspace_id,
            &NewFile::new(&pulled.filename, &pulled.bytes)
                .source(DocumentSource::Import)
                .title(Some(&source))
                .types(request.types.clone())
                .replaces(request.replaces.as_ref())
                .control(self.control),
            self.embedder,
        )
        .await?;
        let result = match outcome {
            IngestOutcome::Ingested(result) => result,
            IngestOutcome::Duplicate(existing)
                if request.replaces.as_ref() == Some(&existing.id) =>
            {
                let table = existing
                    .tables
                    .as_ref()
                    .and_then(|tables| tables.first())
                    .cloned()
                    .unwrap_or_else(|| table.as_str().to_owned());
                let described = {
                    let table = table.clone();
                    db.run(move |db| db.describe_table(&table)).await?
                };
                tracing::info!(table = %table, source = %source, "the import's source is unchanged");
                return Ok(ImportSummary {
                    table,
                    rows: u64::try_from(described.row_count).unwrap_or(0),
                    columns: described.columns.into_iter().map(|c| c.name).collect(),
                    source,
                    document_id: existing.id,
                    status: LoadStatus::Unchanged,
                });
            }
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
            db.run(move |db| {
                let kept = cap_loaded_table(db, &table, limit)?;
                TableProfile::refresh_or_warn(db, &table);
                Ok(kept)
            })
            .await?
        };
        tracing::info!(table = %loaded, rows, source = %source, "imported external data");
        Ok(ImportSummary {
            table: loaded,
            rows,
            columns,
            source,
            document_id: result.document_id,
            status: LoadStatus::Loaded,
        })
    }

    /// Fetch the source as a staging file for `table`: a download, or at
    /// most `limit` rows of a query written as CSV.
    async fn pull(&self, table: &TableName, limit: u64) -> Result<Pulled> {
        let (config, request, policy, control) =
            (self.config, self.request, self.policy, self.control);
        let timeout = config.import.timeout();
        let download = Download {
            timeout,
            max_mb: config.import.max_download_mb,
            hosts: policy.hosts,
            proxies: Proxies::from_env(),
        };
        match request.url.kind()? {
            SourceKind::Http => {
                let headers = request
                    .headers
                    .iter()
                    .map(|header| header.resolve(|name| std::env::var(name).ok()))
                    .collect::<Result<Vec<_>>>()?;
                control
                    .or_cancelled(download.fetch(&request.url, table.as_str(), headers))
                    .await
            }
            SourceKind::S3 => {
                let object = S3Object::parse(&request.url)?;
                control
                    .or_cancelled(download.fetch_s3(&object, table.as_str()))
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
            SourceKind::Sqlite => {
                let sql = request.source_query()?;
                let fetch = async {
                    tokio::time::timeout(
                        timeout,
                        fetch_rows(request.url.sqlite_path(), &sql, limit),
                    )
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
    /// An import of `url` into `table`, with no query, source table, limit,
    /// types, headers, or JSON pointer yet.
    #[must_use]
    pub fn new(url: impl Into<SourceUrl>, table: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            table: table.into(),
            query: None,
            source_table: None,
            limit: None,
            types: ColumnTypes::default(),
            headers: Vec::new(),
            json_pointer: None,
            replaces: None,
        }
    }

    /// Refuse options the source cannot take, and credentials the caller
    /// may not use: S3 signs with this process's AWS identity, and a bearer
    /// token from the environment is this process's secret.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ServerCredentials`] for credentials the policy does
    /// not cover, and an ingestion error for a misplaced option.
    pub fn check(&self, kind: SourceKind, policy: ImportPolicy) -> Result<()> {
        if !self.headers.is_empty() && kind != SourceKind::Http {
            return Err(Error::Ingestion(String::from(
                "headers go with an http(s):// source",
            )));
        }
        if self.json_pointer.is_some() && kind == SourceKind::Sqlite {
            return Err(Error::Ingestion(String::from(
                "a JSON pointer goes with a downloaded JSON file",
            )));
        }
        let process_credentials = kind == SourceKind::S3
            || self
                .headers
                .iter()
                .any(|header| matches!(header, SourceHeader::BearerEnv(_)));
        if process_credentials && policy.credentials == CredentialReach::CallerOnly {
            return Err(Error::ServerCredentials);
        }
        Ok(())
    }

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

async fn fetch_rows(path: &Path, inner: &str, limit: u64) -> Result<Fetched> {
    // Read-only, and opened by path rather than parsed from a URL, so a
    // Windows path needs no rewriting and the source is never changed.
    let mut conn = SqliteConnectOptions::new()
        .filename(path)
        .read_only(true)
        .connect()
        .await
        .map_err(|e| Error::Ingestion(format!("cannot open the source: {e}")))?;
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
struct Download<'a> {
    timeout: Duration,
    max_mb: u64,
    hosts: HostReach,
    proxies: &'a Proxies,
}

impl Download<'_> {
    /// The extension of `name` when `quack ingest` loads it as a table, so
    /// the usual reader takes the download under the requested table name.
    fn table_extension(name: &str) -> Result<String> {
        name.split(['?', '#'])
            .next()
            .and_then(|path| path.rsplit('/').next())
            .filter(|name| FileType::of(name).is_some_and(|t| t.load().makes_tables()))
            .and_then(|name| name.rsplit_once('.'))
            .map(|(_, ext)| ext.to_ascii_lowercase())
            .ok_or_else(|| {
                let accepted: Vec<String> = FileType::table_extensions()
                    .map(|e| format!(".{e}"))
                    .collect();
                Error::Ingestion(format!(
                    "the source must name a table file ending in {}",
                    accepted.join(", ")
                ))
            })
    }

    /// Download the data file at `url`, sending `headers`.
    async fn fetch(
        &self,
        url: &SourceUrl,
        table: &str,
        headers: Vec<(HeaderName, HeaderValue)>,
    ) -> Result<Pulled> {
        let extension = Self::table_extension(url.expose())?;
        let parsed = reqwest::Url::parse(url.expose())
            .map_err(|e| Error::Ingestion(format!("bad URL: {e}")))?;
        let response = self.get(parsed, headers).await?;
        Ok(Pulled {
            filename: format!("{table}.{extension}"),
            bytes: self.body(response).await?,
            columns: Vec::new(),
            rows: None,
        })
    }

    /// Download an S3 object. A bucket in another region than the
    /// configured one answers 301 with its region; the GET is signed again
    /// for that region once.
    async fn fetch_s3(&self, object: &S3Object, table: &str) -> Result<Pulled> {
        const BUCKET_REGION: &str = "x-amz-bucket-region";
        let extension = Self::table_extension(object.key())?;
        let signed = Box::pin(object.signed_get(self.proxies, None)).await?;
        let mut response = self.get(signed.url, signed.headers).await?;
        if response.status() == StatusCode::MOVED_PERMANENTLY
            && let Some(region) = response
                .headers()
                .get(BUCKET_REGION)
                .and_then(|region| region.to_str().ok())
                .map(str::to_owned)
        {
            let signed = Box::pin(object.signed_get(self.proxies, Some(&region))).await?;
            response = self.get(signed.url, signed.headers).await?;
        }
        Ok(Pulled {
            filename: format!("{table}.{extension}"),
            bytes: self.body(response).await?,
            columns: Vec::new(),
            rows: None,
        })
    }

    /// Send a GET for `url` with `headers`; the response comes back whatever
    /// its status.
    ///
    /// When only public hosts may be reached, the name is resolved first,
    /// every address is checked, and the connection is pinned to those
    /// addresses so a second lookup cannot answer differently. Through a
    /// proxy the proxy resolves the name, and may be the only resolver that
    /// can, so only an address written in the URL is checked here.
    async fn get(
        &self,
        url: reqwest::Url,
        headers: Vec<(HeaderName, HeaderValue)>,
    ) -> Result<reqwest::Response> {
        let mut builder = self.proxies.client().timeout(self.timeout);
        if self.hosts == HostReach::PublicOnly {
            let host = url
                .host_str()
                .ok_or_else(|| Error::Ingestion(String::from("the URL has no host")))?;
            let port = url
                .port_or_known_default()
                .ok_or_else(|| Error::Ingestion(String::from("the URL has no port")))?;
            match self.proxies.route(&url) {
                Route::Direct => {
                    let addresses = public_addresses(host, port).await?;
                    builder = builder.resolve_to_addrs(host, &addresses);
                }
                Route::Proxied => {
                    // An IPv6 host is bracketed in a URL.
                    let literal = host.trim_start_matches('[').trim_end_matches(']');
                    if let Ok(ip) = literal.parse::<IpAddr>() {
                        refuse_private(host, ip)?;
                    }
                }
            }
            builder = builder.redirect(reqwest::redirect::Policy::none());
        }
        let client = builder
            .build()
            .map_err(|e| Error::Ingestion(e.to_string()))?;
        let mut request = client.get(url);
        for (name, value) in headers {
            request = request.header(name, value);
        }
        request
            .send()
            .await
            .map_err(|e| Error::Ingestion(format!("download failed: {e}")))
    }

    /// A successful response's body, within `[import].max_download_mb`.
    async fn body(&self, mut response: reqwest::Response) -> Result<Vec<u8>> {
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
        let max_bytes = self.max_mb.saturating_mul(1024 * 1024);
        let too_large = || {
            Error::Ingestion(format!(
                "the file is larger than [import].max_download_mb ({} MB)",
                self.max_mb
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
        Ok(bytes)
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
    for address in &addresses {
        refuse_private(host, address.ip())?;
    }
    Ok(addresses)
}

fn refuse_private(host: &str, ip: IpAddr) -> Result<()> {
    if is_private_address(ip) {
        return Err(Error::Ingestion(format!(
            "{host} resolves to {ip}, a private address; the server does not import from \
             its own network (set [import].allow_private_hosts to allow it)"
        )));
    }
    Ok(())
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

mod s3;
mod saved;

use s3::S3Object;
pub use saved::{ImportSecrets, KeepSecret, RefreshWith, SavedImport};

#[cfg(test)]
mod tests;
