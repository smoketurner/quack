use std::fmt;
use std::path::PathBuf;

use thiserror::Error;

use crate::llm::egress::Refusal;
use crate::saved::Unsavable;
use crate::storage::control::ResourceKind;
use crate::storage::workspace::DuckDbMessage;

#[derive(Debug, Error)]
pub enum Error {
    #[error("configuration error: {0}")]
    Config(String),

    #[error(transparent)]
    Sqlite(#[from] sqlx::Error),

    /// Shown through `DuckDbMessage`, and not a `source`, so nothing that
    /// walks the error chain prints the raw report with its suggestions.
    #[error("{}", DuckDbMessage(.0))]
    DuckDb(duckdb::Error),

    /// A command named a workspace that does not exist.
    #[error("no workspace named '{0}'; create it with: quack workspace create {0}")]
    NoWorkspaceNamed(String),

    /// A workspace by that name is already there; none was created.
    #[error("workspace '{0}' already exists")]
    WorkspaceExists(String),

    /// A saved question by that name is already there; none was saved.
    #[error("saved question '{0}' already exists")]
    SavedQuestionExists(String),

    /// A saved import already has this name.
    #[error("a saved import named '{0}' exists; refresh it, or remove it first")]
    SavedImportExists(String),

    /// An answer that cannot become a saved question, and why.
    #[error("cannot save this answer: {0}")]
    Unsavable(#[from] Unsavable),

    /// Text that cannot name a workspace.
    #[error("workspace name must be non-empty and contain no slashes or dots")]
    InvalidWorkspaceName,

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("LLM error: {0}")]
    Llm(String),

    /// A value sealed at rest could not be sealed or opened.
    #[error("sealed data error: {0}")]
    Vault(String),

    /// An identity-provider access token presented as a bearer was refused:
    /// not signed by the issuer, for another audience, expired, or not an
    /// access token.
    #[error("access token refused: {0}")]
    Bearer(String),

    /// A provider that acts on behalf of each person could not here: nobody
    /// is signed in behind the request, the person has no token from the
    /// identity provider, or the issuer refused the exchange.
    #[error("provider '{provider}' acts on behalf of the signed-in person and could not: {reason}")]
    Delegation { provider: String, reason: String },

    /// A sign-in through the server's `OpenID` Connect issuer was refused or
    /// could not be verified.
    #[error("sign-in failed: {0}")]
    SignIn(String),

    /// An OAuth provider has no usable token and no login flow can run here.
    #[error("provider '{provider}' needs a login ({reason}); run `quack auth login {provider}`")]
    AuthRequired {
        provider: String,
        reason: AuthReason,
    },

    /// The workspace's provider allow-list refused a model request;
    /// nothing was sent.
    #[error(transparent)]
    ProviderRefused(#[from] Refusal),

    /// A model request made by work that entered no `llm::egress::Egress`
    /// scope, so no allow-list could be checked; nothing was sent.
    #[error(
        "a model request to provider '{provider}' was made outside any workspace scope and was \
         not sent; this is a bug in quack"
    )]
    ModelRequestUnscoped { provider: String },

    /// No `[general].chat_model` (nor `QUACK_MODEL`) is set, so nothing can
    /// answer a question.
    #[error(
        "no chat model configured — run `quack init` to set up a provider, or set \
         [general].chat_model = \"PROVIDER/MODEL\" in {} or QUACK_MODEL (SQL with `quack -q` \
         needs no model)",
        config_file.display()
    )]
    NoChatModel { config_file: PathBuf },

    /// A record the caller named does not exist.
    #[error("{} '{id}' does not exist", kind.label())]
    NotFound { kind: ResourceKind, id: String },

    /// An id prefix the caller gave names more than one record.
    #[error("'{prefix}' matches {count} {}s; use more of the id", kind.label())]
    Ambiguous {
        kind: ResourceKind,
        prefix: String,
        count: usize,
    },

    /// Another process holds the workspace file open.
    #[error("workspace file {} is open in another quack process", path.display())]
    WorkspaceLocked { path: PathBuf },

    /// A newer quack upgraded the workspace file past the schema this one
    /// knows; it is left as it was.
    #[error(
        "workspace file {} has schema version {recorded}, written by {written_by}; this quack ({}) \
         reads up to version {supported}: {}",
        path.display(),
        env!("CARGO_PKG_VERSION"),
        written_by.advice()
    )]
    WorkspaceTooNew {
        path: PathBuf,
        recorded: u32,
        supported: u32,
        written_by: WrittenBy,
    },

    /// The workspace file's recorded schema version is not a number, so
    /// nothing says which schema it holds; it is left as it was.
    #[error(
        "workspace file {} records schema version '{recorded}', which is not a number; it was \
         left as it was",
        path.display()
    )]
    WorkspaceSchemaUnreadable { path: PathBuf, recorded: String },

    /// The workspace's writer thread is gone, so no write can run.
    #[error("the workspace writer has stopped")]
    WriterStopped,

    /// A write panicked on the writer thread; the writer carries on.
    #[error("the workspace write failed: {0}")]
    WritePanicked(String),

    #[error("analysis error: {0}")]
    Analysis(String),

    #[error("ingestion error: {0}")]
    Ingestion(String),

    /// An admin disabled the account a credential names.
    #[error("account disabled")]
    AccountDisabled,

    /// An import would authenticate as this process (S3 with its AWS
    /// identity, or a bearer token from its environment) for a caller
    /// `[import].allow_server_credentials` does not cover.
    #[error(
        "S3 and environment-variable credentials use the server's own identity; run \
         `quack import` on the host, or set [import].allow_server_credentials"
    )]
    ServerCredentials,

    /// A workspace snapshot that is not one, is from a newer quack, or
    /// names a path outside the workspace (`storage::backup`).
    #[error("snapshot error: {0}")]
    Snapshot(String),

    /// The work was cancelled (a job's cancel token) before it finished.
    #[error("cancelled")]
    Cancelled,

    /// A statement ran past the query timeout and the watchdog stopped it.
    #[error(
        "the statement ran longer than the {timeout:?} query timeout \
         ([analysis].query_timeout_seconds) and was stopped"
    )]
    QueryTimeout { timeout: std::time::Duration },

    /// A structured file would load into a table another document owns
    /// (issue #51): one document per table.
    #[error(
        "table '{table}' belongs to document {document} ({filename}); delete that document first, or rename the file"
    )]
    TableTaken {
        table: String,
        document: String,
        filename: String,
    },

    #[error("ontology error: {0}")]
    Ontology(String),

    /// Text that names none of an enum's values: a role, a scope, a mode.
    #[error("unknown {what} '{value}'; use one of: {allowed}")]
    UnknownValue {
        what: &'static str,
        value: String,
        allowed: String,
    },

    /// A file with no bytes: there is nothing to load or chunk.
    #[error("'{0}' is empty; there is nothing to ingest")]
    EmptyFile(String),

    #[error("unsupported file type: {0}")]
    UnsupportedFileType(String),

    /// An image arrived with no `[ingestion].vision_model` to read it.
    #[error("{0} is an image; set [ingestion].vision_model to describe images")]
    NoVisionModel(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    TomlParse(#[from] toml::de::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Csv(#[from] csv::Error),

    #[error(transparent)]
    SeaQuery(#[from] sea_query::error::Error),

    #[error("format error: {0}")]
    Fmt(#[from] fmt::Error),
}

impl Error {
    /// Bytes the tar reader could not take as a workspace snapshot.
    #[must_use]
    pub fn not_a_snapshot(e: &std::io::Error) -> Self {
        Self::Snapshot(format!("not a workspace snapshot: {e}"))
    }

    /// Whether a workspace's provider allow-list refused the work: the one
    /// place that decides it, for the audit outcome, a turn's failure kind,
    /// and anything else that answers a refusal differently.
    #[must_use]
    pub const fn is_provider_refusal(&self) -> bool {
        matches!(self, Self::ProviderRefused(_))
    }
}

impl From<duckdb::Error> for Error {
    fn from(error: duckdb::Error) -> Self {
        Self::DuckDb(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Why an OAuth provider has no token to hand out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthReason {
    /// Nothing is cached: nobody has logged in, or the cache was removed.
    NoToken,
    /// The cached token expired and came without a refresh token.
    ExpiredNoRefresh,
    /// The cached token expired and refreshing it failed.
    RefreshFailed(String),
}

impl fmt::Display for AuthReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoToken => f.write_str("no token is cached"),
            Self::ExpiredNoRefresh => {
                f.write_str("the token expired and the issuer gave no refresh token")
            }
            Self::RefreshFailed(e) => write!(f, "refresh failed: {e}"),
        }
    }
}

/// The quack version a workspace file says last wrote it; files from before
/// the version was recorded have none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenBy(pub Option<String>);

/// Who wrote the file: that version exactly.
impl fmt::Display for WrittenBy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(version) => write!(f, "quack {version}"),
            None => f.write_str("a newer quack"),
        }
    }
}

impl WrittenBy {
    /// What to do about a file this quack is too old for: that version
    /// or any newer one opens it.
    #[must_use]
    pub fn advice(&self) -> String {
        const RESTORE: &str = "restore the copy of the workspace made before the upgrade";
        match &self.0 {
            Some(version) => format!("run quack {version} or newer, or {RESTORE}"),
            None => format!("run a newer quack, or {RESTORE}"),
        }
    }
}
