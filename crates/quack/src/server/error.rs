//! One error type for every handler: a status and a message, rendered as
//! `{"error": "..."}`. Core errors map by kind; nothing leaks a stack.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use quack_core::analysis::events::{FailureKind, TurnFailure};
use quack_core::error::Error as CoreError;

#[derive(Debug)]
pub(crate) struct ApiError {
    pub status: StatusCode,
    pub message: String,
    /// Seconds for a `Retry-After` header, on a 503 that is only busy.
    pub retry_after: Option<u32>,
}

impl ApiError {
    pub(crate) fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            retry_after: None,
        }
    }

    /// 503 with `Retry-After`: the request is fine, the server is full.
    pub(crate) fn busy(message: impl Into<String>, retry_after_seconds: u32) -> Self {
        Self {
            retry_after: Some(retry_after_seconds),
            ..Self::new(StatusCode::SERVICE_UNAVAILABLE, message)
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
        if self.status.is_server_error() && self.retry_after.is_none() {
            tracing::error!(status = %self.status, "{}", self.message);
        }
        let mut response = (
            self.status,
            Json(serde_json::json!({ "error": self.message })),
        )
            .into_response();
        if let Some(seconds) = self.retry_after {
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from(seconds),
            );
        }
        response
    }
}

impl From<CoreError> for ApiError {
    fn from(err: CoreError) -> Self {
        let status = match &err {
            // The request is fine; the server cannot serve it until a login
            // happens or another process lets go of the workspace file.
            CoreError::AuthRequired { .. } | CoreError::WorkspaceLocked { .. } => {
                StatusCode::SERVICE_UNAVAILABLE
            }
            CoreError::WorkspaceNotFound(_) | CoreError::NotFound { .. } => StatusCode::NOT_FOUND,
            CoreError::Config(_)
            | CoreError::NoChatModel { .. }
            | CoreError::UnsupportedFileType(_)
            | CoreError::Ontology(_) => StatusCode::BAD_REQUEST,
            CoreError::Analysis(_) | CoreError::UnknownValue { .. } => {
                StatusCode::UNPROCESSABLE_ENTITY
            }
            // A request is never cancelled through its own handler today (only
            // background jobs are); should one be, it lost to a later action.
            CoreError::TableTaken { .. } | CoreError::Cancelled => StatusCode::CONFLICT,
            CoreError::Sqlite(_)
            | CoreError::DuckDb(_)
            | CoreError::Embedding(_)
            | CoreError::Llm(_)
            | CoreError::Ingestion(_)
            | CoreError::Io(_)
            | CoreError::TomlParse(_)
            | CoreError::Json(_)
            | CoreError::SeaQuery(_)
            | CoreError::Fmt(_)
            | CoreError::WriterStopped
            | CoreError::WritePanicked(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self::new(status, err.to_string())
    }
}

/// A failed agent turn, answered by what kind of failure it was.
impl From<TurnFailure> for ApiError {
    fn from(failure: TurnFailure) -> Self {
        let status = match failure.kind {
            FailureKind::AuthRequired => StatusCode::SERVICE_UNAVAILABLE,
            FailureKind::NoChatModel => StatusCode::BAD_REQUEST,
            FailureKind::NotFound => StatusCode::NOT_FOUND,
            FailureKind::Other => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self::new(status, failure.message)
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(err: serde_json::Error) -> Self {
        Self::internal(err.to_string())
    }
}

pub(crate) type ApiResult<T> = std::result::Result<T, ApiError>;
