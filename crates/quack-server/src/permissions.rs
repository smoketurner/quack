//! The server's end of a held write (`quack_core::analysis::permissions`):
//! a streamed turn's `PermissionRequest` is held under
//! `[server].permission_timeout_seconds` for the person whose turn asked,
//! who answers through `POST .../sessions/{sid}/permissions/{request}`;
//! an expiry is audited as a denied `permission`.

use std::sync::Arc;

use quack_core::analysis::events::{Decision, PermissionRequest};
use quack_core::analysis::permissions::{Asker, Held, Refusal};
use quack_core::ids::{PermissionId, SessionId};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};

use super::auth::Access;
use super::error::ApiError;
use super::state::App;

/// Hold `request` for the turn `access` started in `session`. When nobody
/// has answered after `[server].permission_timeout_seconds`, the write is
/// refused and that is audited.
pub(crate) fn hold(
    app: &App,
    access: &Access,
    session: &SessionId,
    request: PermissionRequest,
) -> Held {
    let timeout = app.config.server.permission_timeout();
    let (app, access, session) = (Arc::clone(app), access.clone(), session.clone());
    app.permissions.clone().hold(
        asker(&access, &session),
        request,
        timeout,
        move |expired| async move {
            if let Err(e) = access
                .audit(
                    &app,
                    AuditAction::Permission,
                    Some(ResourceKind::Session.id(&session)),
                    Outcome::Denied,
                    Some(expired.detail()),
                )
                .await
            {
                tracing::error!(error = %e.message, "audit write failed");
            }
        },
    )
}

/// Give `answer` to the write `request` waits on in `session`, for the
/// person `access` names. Returns the statement once the turn has the
/// answer.
///
/// # Errors
///
/// A [`Refusal`] as `Permissions::decide` gives it.
pub(crate) fn decide(
    app: &App,
    access: &Access,
    session: &SessionId,
    request: &PermissionId,
    answer: Decision,
) -> Result<String, Refusal> {
    app.permissions
        .decide(request, &asker(access, session), answer)
}

/// Whose turn asked: `access` in `session`.
fn asker(access: &Access, session: &SessionId) -> Asker {
    Asker {
        workspace: access.membership.workspace.id.clone(),
        session: session.clone(),
        user: access.identity.user_id.clone(),
    }
}

impl From<Refusal> for ApiError {
    fn from(refusal: Refusal) -> Self {
        let message = refusal.message();
        match refusal {
            Refusal::Unknown => Self::not_found(message),
            Refusal::Decided => Self::conflict(message),
            Refusal::NotYours => Self::forbidden(message),
            Refusal::Gone { .. } => Self::gone(message),
        }
    }
}
