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
    AuthRequired { provider: String, reason: String },

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
