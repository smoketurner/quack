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
