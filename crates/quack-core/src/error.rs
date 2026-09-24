use std::fmt;
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("configuration error: {0}")]
    Config(String),

    #[error(transparent)]
    Sqlite(#[from] sqlx::Error),

    #[error(transparent)]
    DuckDb(#[from] duckdb::Error),

    #[error("workspace not found: {0}")]
    WorkspaceNotFound(String),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("LLM error: {0}")]
    Llm(String),

    /// An OAuth provider has no usable token and no login flow can run here.
    #[error("provider '{provider}' needs a login ({reason}); run `quack auth login {provider}`")]
    AuthRequired {
        provider: String,
        reason: AuthReason,
    },

    /// No `[general].chat_model` (nor `QUACK_MODEL`) is set, so nothing can
    /// answer a question.
    #[error(
        "no chat model configured — set [general].chat_model = \"PROVIDER/MODEL\" in {} or \
         QUACK_MODEL; `quack doctor` checks the setup and suggests one (SQL with `quack -q` \
         needs no model)",
        config_file.display()
    )]
    NoChatModel { config_file: PathBuf },

    /// A record the caller named does not exist.
    #[error("{record} '{id}' does not exist")]
    NotFound { record: Record, id: String },

    /// An id prefix the caller gave names more than one record.
    #[error("'{prefix}' matches {count} {record}s; use more of the id")]
    Ambiguous {
        record: Record,
        prefix: String,
        count: usize,
    },

    /// Another process holds the workspace file open.
    #[error("workspace file {} is open in another quack process", path.display())]
    WorkspaceLocked { path: PathBuf },

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

    /// The work was cancelled (a job's cancel token) before it finished.
    #[error("cancelled")]
    Cancelled,

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

    #[error("unsupported file type: {0}")]
    UnsupportedFileType(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    TomlParse(#[from] toml::de::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    SeaQuery(#[from] sea_query::error::Error),

    #[error("format error: {0}")]
    Fmt(#[from] std::fmt::Error),
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

/// The kinds of record a caller names by id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Record {
    Session,
    Document,
    OntologyVersion,
    MergeProposal,
    Candidate,
    Token,
}

text_enum!(Record, "record", {
    Session => "session",
    Document => "document",
    OntologyVersion => "ontology version",
    MergeProposal => "merge proposal",
    Candidate => "candidate",
    Token => "token",
});

impl Record {
    /// The error for this kind of record with `id` missing.
    #[must_use]
    pub fn missing(self, id: impl Into<String>) -> Error {
        Error::NotFound {
            record: self,
            id: id.into(),
        }
    }
}
