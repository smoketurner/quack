//! One error type for every handler: a status, a stable code, and a
//! message, rendered as `{"error": "...", "code": "..."}`. Core errors map
//! by kind; nothing leaks a stack. The code is the contract (issue #419);
//! the message is for a person and may change in any release.

use axum::Json;
use axum::extract::Request;
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use quack_core::analysis::events::{FailureKind, TurnFailure};
use quack_core::error::Error as CoreError;
use serde::Serialize;
use utoipa::ToSchema;

/// What kind of failure an error response reports. A client branches on
/// this, never on the message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ErrorCode {
    /// The request is malformed or names something it may not.
    BadRequest,
    /// No credential, or one that is not valid.
    Unauthorized,
    /// The caller is known and may not do this.
    Forbidden,
    /// What the request names does not exist, or the caller may not see it.
    NotFound,
    /// The route exists, without this method.
    MethodNotAllowed,
    /// The request conflicts with what already exists or is running.
    Conflict,
    /// What the request addressed existed, and is over.
    Gone,
    /// The body is larger than `[ingestion].upload_max_mb`.
    PayloadTooLarge,
    /// The body is not JSON (or multipart) where the route takes it.
    UnsupportedMediaType,
    /// Well-formed, but its content cannot be carried out.
    Unprocessable,
    /// Too many requests from this address; wait and retry.
    RateLimited,
    /// The server is full or stopping; retry after `Retry-After`.
    Busy,
    /// The request ran longer than the server allows.
    Timeout,
    /// The server failed; the message says how.
    Internal,
    /// A model provider needs `quack auth login` first.
    AuthRequired,
    /// Another process holds the workspace file open.
    WorkspaceLocked,
    /// A newer quack wrote the workspace file.
    WorkspaceTooNew,
    /// No workspace has that name.
    WorkspaceNotFound,
    /// A sign-in through the identity provider was refused.
    SignInFailed,
    /// An identity-provider access token was refused.
    InvalidBearer,
    /// An admin disabled the account.
    AccountDisabled,
    /// A provider that acts for the signed-in person could not.
    DelegationFailed,
    /// The workspace's provider allow-list refused the model.
    ProviderRefused,
    /// An import would use the server's own credentials, which
    /// `[import].allow_server_credentials` does not allow.
    ServerCredentials,
    /// The server's configuration does not allow the request.
    InvalidConfig,
    /// An id prefix matches more than one record.
    Ambiguous,
    /// No chat model is configured.
    NoChatModel,
    /// The file's type is not one quack ingests.
    UnsupportedFileType,
    /// The file has no bytes.
    EmptyFile,
    /// The text cannot name a workspace.
    InvalidWorkspaceName,
    /// The snapshot is not one, or is from a newer quack.
    InvalidSnapshot,
    /// The ontology does not validate.
    InvalidOntology,
    /// The agent could not carry the request out.
    AnalysisFailed,
    /// A value names none of the allowed ones (a role, a scope, a mode).
    UnknownValue,
    /// The answer cannot become a saved question.
    Unsavable,
    /// A statement ran past the query timeout.
    QueryTimeout,
    /// A SQL statement failed.
    SqlFailed,
    /// An import's source could not be read or loaded.
    ImportFailed,
    /// The file would load into a table another document owns.
    TableTaken,
    /// A workspace by that name exists.
    WorkspaceExists,
    /// A saved question by that name exists.
    SavedQuestionExists,
    /// A saved import already has the name.
    SavedImportExists,
    /// The work was cancelled before it finished.
    Cancelled,
}

impl ErrorCode {
    /// The code a bare status carries: every error built from a status
    /// alone, and every response the framework built itself.
    fn for_status(status: StatusCode) -> Self {
        match status {
            StatusCode::BAD_REQUEST => Self::BadRequest,
            StatusCode::UNAUTHORIZED => Self::Unauthorized,
            StatusCode::FORBIDDEN => Self::Forbidden,
            StatusCode::NOT_FOUND => Self::NotFound,
            StatusCode::METHOD_NOT_ALLOWED => Self::MethodNotAllowed,
            StatusCode::CONFLICT => Self::Conflict,
            StatusCode::GONE => Self::Gone,
            StatusCode::PAYLOAD_TOO_LARGE => Self::PayloadTooLarge,
            StatusCode::UNSUPPORTED_MEDIA_TYPE => Self::UnsupportedMediaType,
            StatusCode::UNPROCESSABLE_ENTITY => Self::Unprocessable,
            StatusCode::TOO_MANY_REQUESTS => Self::RateLimited,
            StatusCode::SERVICE_UNAVAILABLE => Self::Busy,
            StatusCode::GATEWAY_TIMEOUT => Self::Timeout,
            _ => Self::Internal,
        }
    }

    /// The code and status a core error answers with.
    fn of(err: &CoreError) -> (Self, StatusCode) {
        match err {
            // The request is fine; the server cannot serve it until a login
            // happens, another process lets go of the workspace file, or a
            // quack new enough for the file runs.
            CoreError::AuthRequired { .. } => (Self::AuthRequired, StatusCode::SERVICE_UNAVAILABLE),
            CoreError::WorkspaceLocked { .. } => {
                (Self::WorkspaceLocked, StatusCode::SERVICE_UNAVAILABLE)
            }
            CoreError::WorkspaceTooNew { .. } => {
                (Self::WorkspaceTooNew, StatusCode::SERVICE_UNAVAILABLE)
            }
            CoreError::NoWorkspaceNamed(_) => (Self::WorkspaceNotFound, StatusCode::NOT_FOUND),
            CoreError::NotFound { .. } => (Self::NotFound, StatusCode::NOT_FOUND),
            CoreError::SignIn(_) => (Self::SignInFailed, StatusCode::UNAUTHORIZED),
            CoreError::Bearer(_) => (Self::InvalidBearer, StatusCode::UNAUTHORIZED),
            CoreError::AccountDisabled => (Self::AccountDisabled, StatusCode::UNAUTHORIZED),
            // The caller is known; this provider cannot act for them, or
            // the workspace's allow-list keeps its content from the provider.
            CoreError::Delegation { .. } => (Self::DelegationFailed, StatusCode::FORBIDDEN),
            CoreError::ProviderRefused(_) => (Self::ProviderRefused, StatusCode::FORBIDDEN),
            CoreError::ServerCredentials => (Self::ServerCredentials, StatusCode::FORBIDDEN),
            CoreError::Config(_) => (Self::InvalidConfig, StatusCode::BAD_REQUEST),
            CoreError::Ambiguous { .. } => (Self::Ambiguous, StatusCode::BAD_REQUEST),
            CoreError::NoChatModel { .. } => (Self::NoChatModel, StatusCode::BAD_REQUEST),
            CoreError::UnsupportedFileType(_) => {
                (Self::UnsupportedFileType, StatusCode::BAD_REQUEST)
            }
            CoreError::EmptyFile(_) => (Self::EmptyFile, StatusCode::BAD_REQUEST),
            CoreError::InvalidWorkspaceName => {
                (Self::InvalidWorkspaceName, StatusCode::BAD_REQUEST)
            }
            CoreError::Snapshot(_) => (Self::InvalidSnapshot, StatusCode::BAD_REQUEST),
            CoreError::Ontology(_) => (Self::InvalidOntology, StatusCode::BAD_REQUEST),
            CoreError::Analysis(_) => (Self::AnalysisFailed, StatusCode::UNPROCESSABLE_ENTITY),
            CoreError::UnknownValue { .. } => {
                (Self::UnknownValue, StatusCode::UNPROCESSABLE_ENTITY)
            }
            CoreError::Unsavable(_) => (Self::Unsavable, StatusCode::UNPROCESSABLE_ENTITY),
            CoreError::QueryTimeout { .. } => {
                (Self::QueryTimeout, StatusCode::UNPROCESSABLE_ENTITY)
            }
            CoreError::TableTaken { .. } => (Self::TableTaken, StatusCode::CONFLICT),
            CoreError::WorkspaceExists(_) => (Self::WorkspaceExists, StatusCode::CONFLICT),
            CoreError::SavedQuestionExists(_) => (Self::SavedQuestionExists, StatusCode::CONFLICT),
            CoreError::SavedImportExists(_) => (Self::SavedImportExists, StatusCode::CONFLICT),
            // A request is never cancelled through its own handler today (only
            // background jobs are); should one be, it lost to a later action.
            CoreError::Cancelled => (Self::Cancelled, StatusCode::CONFLICT),
            CoreError::Sqlite(_)
            | CoreError::DuckDb(_)
            | CoreError::Embedding(_)
            | CoreError::Llm(_)
            | CoreError::Vault(_)
            | CoreError::Ingestion(_)
            | CoreError::Io(_)
            | CoreError::TomlParse(_)
            | CoreError::Json(_)
            | CoreError::Csv(_)
            | CoreError::SeaQuery(_)
            | CoreError::Fmt(_)
            | CoreError::WriterStopped
            | CoreError::WritePanicked(_)
            | CoreError::WorkspaceSchemaUnreadable { .. }
            | CoreError::ModelRequestUnscoped { .. } => {
                (Self::Internal, StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }

    /// The code and status a failed agent turn answers with.
    fn of_turn(kind: FailureKind) -> (Self, StatusCode) {
        match kind {
            FailureKind::AuthRequired => (Self::AuthRequired, StatusCode::SERVICE_UNAVAILABLE),
            FailureKind::NoChatModel => (Self::NoChatModel, StatusCode::BAD_REQUEST),
            FailureKind::NotFound => (Self::NotFound, StatusCode::NOT_FOUND),
            FailureKind::ProviderNotAllowed => (Self::ProviderRefused, StatusCode::FORBIDDEN),
            FailureKind::Other => (Self::Internal, StatusCode::INTERNAL_SERVER_ERROR),
        }
    }
}

/// The body of every error response, and the data of the SSE `error`
/// event.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ErrorBody {
    /// For a person; it may change in any release.
    pub error: String,
    pub code: ErrorCode,
}

#[derive(Debug)]
pub(crate) struct ApiError {
    pub status: StatusCode,
    pub code: ErrorCode,
    pub message: String,
    /// Seconds for a `Retry-After` header, on a 503 that is only busy.
    pub retry_after: Option<u32>,
    /// A `WWW-Authenticate` value, on a 401 that tells the client where to
    /// get a token (RFC 6750, RFC 9728).
    pub challenge: Option<String>,
}

impl ApiError {
    pub(crate) fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self::coded(status, ErrorCode::for_status(status), message)
    }

    fn coded(status: StatusCode, code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            retry_after: None,
            challenge: None,
        }
    }

    /// The same error, with a `WWW-Authenticate` challenge.
    #[must_use]
    pub(crate) fn with_challenge(self, challenge: String) -> Self {
        Self {
            challenge: Some(challenge),
            ..self
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

    /// 409: the request conflicts with what already exists or is running.
    pub(crate) fn conflict(message: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, message)
    }

    /// 410: what the request addressed existed, and is over.
    pub(crate) fn gone(message: impl Into<String>) -> Self {
        Self::new(StatusCode::GONE, message)
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }

    /// This error as a 422: the caller's own statement or source failed. A
    /// specific code (a timeout) stays; a generic one becomes `fallback`.
    #[must_use]
    pub(crate) fn unprocessable_as(self, fallback: ErrorCode) -> Self {
        let code = match self.code {
            ErrorCode::Internal | ErrorCode::Unprocessable => fallback,
            specific => specific,
        };
        Self::coded(StatusCode::UNPROCESSABLE_ENTITY, code, self.message)
    }

    /// The body every interface sends for this error.
    pub(crate) fn body(&self) -> ErrorBody {
        ErrorBody {
            error: self.message.clone(),
            code: self.code,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        if self.status.is_server_error() && self.retry_after.is_none() {
            tracing::error!(status = %self.status, code = ?self.code, "{}", self.message);
        }
        let mut response = (self.status, Json(self.body())).into_response();
        if let Some(challenge) = self
            .challenge
            .as_deref()
            .and_then(|c| HeaderValue::from_str(c).ok())
        {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, challenge);
        }
        if let Some(seconds) = self.retry_after {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(seconds));
        }
        // Marks the body as already coded for `coded_errors`.
        response.extensions_mut().insert(self.code);
        response
    }
}

/// The most bytes of a framework-built error body kept as its message: a
/// rejection's text is one line.
const FRAMEWORK_MESSAGE_LIMIT: usize = 4096;

/// Middleware over `/api/v1`: an error response the framework built itself
/// (a body that is not JSON, a method the route lacks, the rate limiter, the
/// timeout) gets the same `{"error", "code"}` body as a handler's, so every
/// error a client sees carries a code.
pub(crate) async fn coded_errors(request: Request, next: Next) -> Response {
    let api = request.uri().path().starts_with("/api/");
    let response = next.run(request).await;
    let status = response.status();
    if !api
        || !(status.is_client_error() || status.is_server_error())
        || response.extensions().get::<ErrorCode>().is_some()
    {
        return response;
    }
    let (parts, body) = response.into_parts();
    let text = axum::body::to_bytes(body, FRAMEWORK_MESSAGE_LIMIT)
        .await
        .map(|bytes| String::from_utf8_lossy(&bytes).trim().to_owned())
        .unwrap_or_default();
    let message = if text.is_empty() {
        status
            .canonical_reason()
            .unwrap_or("request failed")
            .to_lowercase()
    } else {
        text
    };
    let mut coded = ApiError::new(status, message).into_response();
    // The framework's own headers stay: `Allow` on a 405, `Retry-After` on
    // a 429, the request id.
    for (name, value) in &parts.headers {
        if name != header::CONTENT_TYPE && name != header::CONTENT_LENGTH {
            coded.headers_mut().insert(name.clone(), value.clone());
        }
    }
    coded
}

impl From<CoreError> for ApiError {
    fn from(err: CoreError) -> Self {
        let (code, status) = ErrorCode::of(&err);
        Self::coded(status, code, err.to_string())
    }
}

/// A failed agent turn, answered by what kind of failure it was.
impl From<TurnFailure> for ApiError {
    fn from(failure: TurnFailure) -> Self {
        let (code, status) = ErrorCode::of_turn(failure.kind);
        Self::coded(status, code, failure.message)
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(err: serde_json::Error) -> Self {
        Self::internal(err.to_string())
    }
}

pub(crate) type ApiResult<T> = Result<T, ApiError>;

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "serde_json::Value indexing yields Null for a missing key, never a panic"
)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use axum::Router;
    use axum::body::Body;
    use axum::routing::get;
    use quack_core::error::AuthReason;
    use quack_core::llm::egress::Refusal;
    use quack_core::saved::Unsavable;
    use quack_core::storage::control::{AllowedProviders, ResourceKind};
    use tower::ServiceExt;

    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn text(value: &str) -> String {
        value.to_owned()
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "one case per core error, as a table")]
    fn core_errors_map_to_their_codes_and_statuses() {
        let cases = [
            (
                CoreError::AuthRequired {
                    provider: text("p"),
                    reason: AuthReason::NoToken,
                },
                ErrorCode::AuthRequired,
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                CoreError::WorkspaceLocked {
                    path: PathBuf::from("w"),
                },
                ErrorCode::WorkspaceLocked,
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                CoreError::NoWorkspaceNamed(text("w")),
                ErrorCode::WorkspaceNotFound,
                StatusCode::NOT_FOUND,
            ),
            (
                ResourceKind::Session.missing("s"),
                ErrorCode::NotFound,
                StatusCode::NOT_FOUND,
            ),
            (
                CoreError::SignIn(text("x")),
                ErrorCode::SignInFailed,
                StatusCode::UNAUTHORIZED,
            ),
            (
                CoreError::Bearer(text("x")),
                ErrorCode::InvalidBearer,
                StatusCode::UNAUTHORIZED,
            ),
            (
                CoreError::AccountDisabled,
                ErrorCode::AccountDisabled,
                StatusCode::UNAUTHORIZED,
            ),
            (
                CoreError::Delegation {
                    provider: text("p"),
                    reason: text("r"),
                },
                ErrorCode::DelegationFailed,
                StatusCode::FORBIDDEN,
            ),
            (
                CoreError::ProviderRefused(Refusal::Provider {
                    provider: text("p"),
                    allowed: AllowedProviders::default(),
                }),
                ErrorCode::ProviderRefused,
                StatusCode::FORBIDDEN,
            ),
            (
                CoreError::Config(text("x")),
                ErrorCode::InvalidConfig,
                StatusCode::BAD_REQUEST,
            ),
            (
                CoreError::Ambiguous {
                    kind: ResourceKind::Document,
                    prefix: text("a"),
                    count: 2,
                },
                ErrorCode::Ambiguous,
                StatusCode::BAD_REQUEST,
            ),
            (
                CoreError::NoChatModel {
                    config_file: PathBuf::from("c"),
                },
                ErrorCode::NoChatModel,
                StatusCode::BAD_REQUEST,
            ),
            (
                CoreError::UnsupportedFileType(text("x")),
                ErrorCode::UnsupportedFileType,
                StatusCode::BAD_REQUEST,
            ),
            (
                CoreError::EmptyFile(text("x")),
                ErrorCode::EmptyFile,
                StatusCode::BAD_REQUEST,
            ),
            (
                CoreError::InvalidWorkspaceName,
                ErrorCode::InvalidWorkspaceName,
                StatusCode::BAD_REQUEST,
            ),
            (
                CoreError::Snapshot(text("x")),
                ErrorCode::InvalidSnapshot,
                StatusCode::BAD_REQUEST,
            ),
            (
                CoreError::Ontology(text("x")),
                ErrorCode::InvalidOntology,
                StatusCode::BAD_REQUEST,
            ),
            (
                CoreError::Analysis(text("x")),
                ErrorCode::AnalysisFailed,
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                CoreError::UnknownValue {
                    what: "role",
                    value: text("x"),
                    allowed: text("y"),
                },
                ErrorCode::UnknownValue,
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                CoreError::Unsavable(Unsavable::NoSql),
                ErrorCode::Unsavable,
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                CoreError::QueryTimeout {
                    timeout: Duration::from_secs(1),
                },
                ErrorCode::QueryTimeout,
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                CoreError::TableTaken {
                    table: text("t"),
                    document: text("d"),
                    filename: text("f"),
                },
                ErrorCode::TableTaken,
                StatusCode::CONFLICT,
            ),
            (
                CoreError::WorkspaceExists(text("w")),
                ErrorCode::WorkspaceExists,
                StatusCode::CONFLICT,
            ),
            (
                CoreError::SavedQuestionExists(text("s")),
                ErrorCode::SavedQuestionExists,
                StatusCode::CONFLICT,
            ),
            (
                CoreError::Cancelled,
                ErrorCode::Cancelled,
                StatusCode::CONFLICT,
            ),
            (
                CoreError::Llm(text("x")),
                ErrorCode::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                CoreError::WriterStopped,
                ErrorCode::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ];
        for (err, code, status) in cases {
            let message = err.to_string();
            let api = ApiError::from(err);
            assert_eq!((api.code, api.status), (code, status), "{message}");
            assert_eq!(api.message, message);
        }
    }

    #[test]
    fn turn_failures_map_to_their_codes() {
        for (kind, code, status) in [
            (
                FailureKind::AuthRequired,
                ErrorCode::AuthRequired,
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                FailureKind::NoChatModel,
                ErrorCode::NoChatModel,
                StatusCode::BAD_REQUEST,
            ),
            (
                FailureKind::NotFound,
                ErrorCode::NotFound,
                StatusCode::NOT_FOUND,
            ),
            (
                FailureKind::ProviderNotAllowed,
                ErrorCode::ProviderRefused,
                StatusCode::FORBIDDEN,
            ),
            (
                FailureKind::Other,
                ErrorCode::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ] {
            let api = ApiError::from(TurnFailure {
                message: text("m"),
                kind,
            });
            assert_eq!((api.code, api.status), (code, status), "{kind:?}");
        }
    }

    #[test]
    fn a_bare_status_carries_its_own_code() {
        for (status, code) in [
            (StatusCode::BAD_REQUEST, "bad_request"),
            (StatusCode::UNAUTHORIZED, "unauthorized"),
            (StatusCode::FORBIDDEN, "forbidden"),
            (StatusCode::NOT_FOUND, "not_found"),
            (StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed"),
            (StatusCode::CONFLICT, "conflict"),
            (StatusCode::GONE, "gone"),
            (StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large"),
            (StatusCode::UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type"),
            (StatusCode::UNPROCESSABLE_ENTITY, "unprocessable"),
            (StatusCode::TOO_MANY_REQUESTS, "rate_limited"),
            (StatusCode::SERVICE_UNAVAILABLE, "busy"),
            (StatusCode::GATEWAY_TIMEOUT, "timeout"),
            (StatusCode::INTERNAL_SERVER_ERROR, "internal"),
            (StatusCode::IM_A_TEAPOT, "internal"),
        ] {
            assert_eq!(
                serde_json::to_value(ErrorCode::for_status(status)).ok(),
                Some(serde_json::json!(code)),
                "{status}"
            );
        }
    }

    #[test]
    fn a_users_failed_statement_keeps_a_specific_code() {
        let timeout = ApiError::from(CoreError::QueryTimeout {
            timeout: Duration::from_secs(1),
        })
        .unprocessable_as(ErrorCode::SqlFailed);
        assert_eq!(
            (timeout.code, timeout.status),
            (ErrorCode::QueryTimeout, StatusCode::UNPROCESSABLE_ENTITY)
        );
        let failed =
            ApiError::from(CoreError::Llm(text("x"))).unprocessable_as(ErrorCode::SqlFailed);
        assert_eq!(
            (failed.code, failed.status),
            (ErrorCode::SqlFailed, StatusCode::UNPROCESSABLE_ENTITY)
        );
    }

    async fn call(
        router: Router,
        path: &str,
    ) -> (StatusCode, axum::http::HeaderMap, serde_json::Value) {
        let response = router
            .oneshot(
                axum::http::Request::get(path)
                    .body(Body::empty())
                    .unwrap_or_else(|e| fail(&e.to_string())),
            )
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let (parts, body) = response.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let value = serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| serde_json::json!(String::from_utf8_lossy(&bytes)));
        (parts.status, parts.headers, value)
    }

    /// The framework's own errors under `/api/` get the coded body and keep
    /// their headers; a handler's are left as they are; other paths keep
    /// whatever they had.
    #[tokio::test]
    async fn framework_errors_under_the_api_get_a_code() {
        let router = Router::new()
            .route(
                "/api/v1/plain",
                get(|| async {
                    (
                        StatusCode::TOO_MANY_REQUESTS,
                        [(header::RETRY_AFTER, "7")],
                        "Too Many Requests! Wait for 7s",
                    )
                }),
            )
            .route(
                "/api/v1/coded",
                get(|| async { ApiError::conflict("taken") }),
            )
            .route("/api/v1/post-only", axum::routing::post(|| async { "ok" }))
            .route(
                "/page",
                get(|| async { (StatusCode::BAD_REQUEST, "plain") }),
            )
            .layer(axum::middleware::from_fn(coded_errors));

        let (status, headers, body) = call(router.clone(), "/api/v1/plain").await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(body["code"], "rate_limited");
        assert_eq!(body["error"], "Too Many Requests! Wait for 7s");
        assert_eq!(
            headers
                .get(header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok()),
            Some("7")
        );

        let (status, _, body) = call(router.clone(), "/api/v1/coded").await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(
            body,
            serde_json::json!({ "error": "taken", "code": "conflict" })
        );

        let (status, headers, body) = call(router.clone(), "/api/v1/post-only").await;
        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(body["code"], "method_not_allowed");
        assert_eq!(body["error"], "method not allowed");
        assert!(headers.contains_key(header::ALLOW));

        let (status, _, body) = call(router.clone(), "/api/v1/missing").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["code"], "not_found");

        let (status, _, body) = call(router, "/page").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body, "plain");
    }
}
