//! What a command returns when it fails: the message every interface
//! shows, and a kind the frontend can tell apart ("no chat model" from
//! "not found"), as the REST API's `ErrorCode` does for its clients.

use quack_core::analysis::events::{FailureKind, TurnFailure};
use quack_core::error::Error as CoreError;
use quack_core::forward;
use serde::Serialize;

/// The failures the frontend answers differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ErrorKind {
    /// The workspace, session, document, or job does not exist.
    NotFound,
    /// No chat model is configured: Settings says what to set.
    NoChatModel,
    /// A provider needs `quack auth login` first.
    AuthRequired,
    /// The workspace's provider allow-list refused a model the work needs.
    ProviderNotAllowed,
    /// Another process holds the workspace file.
    Locked,
    /// The request itself was wrong: a bad name, an empty message.
    Invalid,
    /// A model, tool, or storage failure.
    Internal,
}

/// A failed command, as the frontend receives it.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub(crate) struct CommandError {
    pub(crate) kind: ErrorKind,
    pub(crate) message: String,
}

impl CommandError {
    pub(crate) fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub(crate) fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::NotFound, message)
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Invalid, message)
    }
}

impl From<CoreError> for CommandError {
    fn from(error: CoreError) -> Self {
        let kind = match &error {
            CoreError::NotFound { .. } | CoreError::NoWorkspaceNamed(_) => ErrorKind::NotFound,
            CoreError::NoChatModel { .. } => ErrorKind::NoChatModel,
            CoreError::AuthRequired { .. } => ErrorKind::AuthRequired,
            CoreError::InvalidWorkspaceName => ErrorKind::Invalid,
            refused if refused.is_provider_refusal() => ErrorKind::ProviderNotAllowed,
            locked if forward::is_locked(locked) => ErrorKind::Locked,
            _ => ErrorKind::Internal,
        };
        Self::new(kind, error.to_string())
    }
}

impl From<TurnFailure> for CommandError {
    fn from(failure: TurnFailure) -> Self {
        Self::new(kind_of(failure.kind), failure.message)
    }
}

impl From<anyhow::Error> for CommandError {
    fn from(error: anyhow::Error) -> Self {
        match error.downcast::<CoreError>() {
            Ok(core) => Self::from(core),
            Err(other) => Self::new(ErrorKind::Internal, format!("{other:#}")),
        }
    }
}

impl From<tauri::Error> for CommandError {
    fn from(error: tauri::Error) -> Self {
        Self::new(ErrorKind::Internal, error.to_string())
    }
}

/// The frontend's kind for how a turn failed.
pub(crate) fn kind_of(kind: FailureKind) -> ErrorKind {
    match kind {
        FailureKind::AuthRequired => ErrorKind::AuthRequired,
        FailureKind::NoChatModel => ErrorKind::NoChatModel,
        FailureKind::NotFound => ErrorKind::NotFound,
        FailureKind::ProviderNotAllowed => ErrorKind::ProviderNotAllowed,
        FailureKind::Other => ErrorKind::Internal,
    }
}

pub(crate) type CommandResult<T> = Result<T, CommandError>;
