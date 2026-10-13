//! Finding a session by id or prefix, and choosing the one a run continues.

use quack_core::config::Config;
use quack_core::error::Result as CoreResult;
use quack_core::ids::SessionId;
use quack_core::prefix::PrefixMatch;
use quack_core::storage::control::ResourceKind;
use quack_core::storage::sessions::{self, ChatMode};
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

/// Which session a run continues. Owned, so it travels into the writer's
/// closure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionChoice {
    /// A new session.
    New,
    /// `--continue`: the most recent one, or a new one when there is none.
    Latest,
    /// `--resume ID`: this one (prefixes accepted).
    Resume(String),
}

/// Pick the session for this run: the latest with [`SessionChoice::Latest`],
/// a specific one with [`SessionChoice::Resume`], otherwise a new one for
/// the configured chat model. `mode` sets the mode on a new session and
/// overrides it on a resumed one.
///
/// # Errors
///
/// Returns a database error, or the not-found or ambiguous-prefix error when
/// a resumed prefix names no session or several.
pub fn resolve_session(
    config: &Config,
    db: &WorkspaceDb,
    choice: SessionChoice,
    mode: Option<ChatMode>,
) -> anyhow::Result<SessionId> {
    let existing = match choice {
        SessionChoice::Resume(prefix) => Some(find_session(db, &prefix)?.id),
        SessionChoice::Latest => sessions::latest_session(db)?.map(|s| s.id),
        SessionChoice::New => None,
    };
    if let Some(id) = existing {
        if let Some(mode) = mode {
            sessions::set_session_mode(db, &id, mode)?;
        }
        return Ok(id);
    }
    let model = config
        .chat_model_ref()
        .map_or_else(|_| String::from("unconfigured"), |m| m.to_string());
    Ok(sessions::create_session(db, &model, mode.unwrap_or_default(), None)?.id)
}
