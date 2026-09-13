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
}

pub type Result<T> = std::result::Result<T, Error>;
