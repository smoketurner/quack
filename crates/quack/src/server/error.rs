//! One error type for every handler: a status and a message, rendered as
//! `{"error": "..."}`. Core errors map by kind; nothing leaks a stack.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use quack_core::error::Error as CoreError;

#[derive(Debug)]
pub(crate) struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub(crate) fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }

    pub(crate) fn unauthorized(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, message)
    }

    pub(crate) fn forbidden(message: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, message)
    }

    pub(crate) fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, message)
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        if self.status.is_server_error() {
            tracing::error!(status = %self.status, "{}", self.message);
        }
        (
            self.status,
            Json(serde_json::json!({ "error": self.message })),
        )
            .into_response()
    }
}

impl From<CoreError> for ApiError {
    fn from(err: CoreError) -> Self {
        let status = match &err {
            CoreError::AuthRequired { .. } => StatusCode::SERVICE_UNAVAILABLE,
            CoreError::WorkspaceNotFound(_) => StatusCode::NOT_FOUND,
            CoreError::Config(_) | CoreError::UnsupportedFileType(_) | CoreError::Ontology(_) => {
                StatusCode::BAD_REQUEST
            }
            CoreError::Analysis(_) => StatusCode::UNPROCESSABLE_ENTITY,
            CoreError::Sqlite(_)
            | CoreError::DuckDb(_)
            | CoreError::Embedding(_)
            | CoreError::Llm(_)
            | CoreError::Ingestion(_)
            | CoreError::Io(_)
            | CoreError::TomlParse(_)
            | CoreError::Json(_)
            | CoreError::SeaQuery(_)
            | CoreError::Fmt(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self::new(status, err.to_string())
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(err: serde_json::Error) -> Self {
        Self::internal(err.to_string())
    }
}

pub(crate) type ApiResult<T> = std::result::Result<T, ApiError>;
