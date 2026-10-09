//! Finding a session by id or prefix.

use quack_core::error::Result as CoreResult;
use quack_core::ids::SessionId;
use quack_core::prefix::PrefixMatch;
use quack_core::storage::control::ResourceKind;
use quack_core::storage::sessions;
use quack_core::storage::workspace::WorkspaceDb;

/// Resolve a full id or a unique prefix to a session.
///
/// # Errors
///
/// Returns a database error, or the not-found or ambiguous-prefix error when
/// `prefix` names no session or several.
pub fn find_session(db: &WorkspaceDb, prefix: &str) -> CoreResult<sessions::SessionRow> {
    if let Some(exact) = sessions::get_session(db, &SessionId::from(prefix))? {
        return Ok(exact);
    }
    PrefixMatch::of(sessions::list_sessions(db, 1000)?, prefix, |s| {
        s.id.as_str()
    })
    .one(ResourceKind::Session, prefix)
}
