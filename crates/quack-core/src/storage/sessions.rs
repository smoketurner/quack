//! Sessions and messages, stored inside the workspace file.
//!
//! A session is a conversation; a message is one entry in it. Every agent
//! turn records the user message, one tool message per recorded step, and
//! the assistant answer, so a session is a complete record that can be
//! resumed, listed, and exported without leaving the classification boundary.

use std::fmt::Write as _;

use crate::analysis::agent::AgentResponse;
use crate::error::{Error, Record, Result};
use crate::ids::{SessionId, UserId};

use super::workspace::WorkspaceDb;

/// How the agent may answer in a session.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum ChatMode {
    /// General knowledge allowed; cite when a source was used.
    #[default]
    Chat,
    /// Every claim must come from a retrieved chunk or a query result; say
    /// so when nothing relevant was found.
    Query,
}

text_enum!(ChatMode, "mode", { Chat => "chat", Query => "query" });

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    Tool,
}

text_enum!(MessageRole, "message role", {
    User => "user",
    Assistant => "assistant",
    Tool => "tool",
});

#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionRow {
    pub id: SessionId,
    pub title: Option<String>,
    pub mode: ChatMode,
    pub model: String,
    /// The server user who started it; `None` from the CLI and TUI.
    pub created_by: Option<UserId>,
    /// Visible to every member of the workspace, not only the creator.
    pub shared: bool,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: i64,
}

/// A row selected with [`SESSION_COLUMNS`].
impl TryFrom<&duckdb::Row<'_>> for SessionRow {
    type Error = duckdb::Error;

    fn try_from(row: &duckdb::Row<'_>) -> duckdb::Result<Self> {
        let mode: String = row.get(2)?;
        Ok(Self {
            id: row.get(0)?,
            title: row.get(1)?,
            mode: mode.parse().unwrap_or_default(),
            model: row.get(3)?,
            created_at: row.get(4)?,
            updated_at: row.get(5)?,
            message_count: row.get(6)?,
            created_by: row.get(7)?,
            shared: row.get(8)?,
        })
    }
}

impl SessionRow {
    /// Whether `viewer` may read this session: every session for
    /// [`SessionViewer::All`], else its creator's, and any shared or
    /// ownerless one.
    #[must_use]
    pub fn visible_to(&self, viewer: &SessionViewer) -> bool {
        match viewer {
            SessionViewer::All => true,
            SessionViewer::User(user_id) => {
                self.shared || self.created_by.as_ref().is_none_or(|c| c == user_id)
            }
        }
    }
}

/// Who is asking for sessions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionViewer {
    /// A workspace owner, an admin, or a local caller: every session.
    All,
    /// A server user: their own sessions, shared ones, and ownerless ones
    /// (started from the CLI or the terminal).
    User(UserId),
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MessageRow {
    pub id: String,
    pub session_id: SessionId,
    pub seq: i64,
    pub role: MessageRole,
    pub content: String,
    pub metadata: Option<serde_json::Value>,
    pub created_at: String,
}

/// A row of `id, session_id, seq, role, content, metadata, created_at`.
impl TryFrom<&duckdb::Row<'_>> for MessageRow {
    type Error = Error;

    fn try_from(row: &duckdb::Row<'_>) -> Result<Self> {
        let role: String = row.get(3)?;
        let metadata: Option<String> = row.get(5)?;
        Ok(Self {
            id: row.get(0)?,
            session_id: row.get(1)?,
            seq: row.get(2)?,
            role: role.parse()?,
            content: row.get(4)?,
            metadata: metadata.as_deref().map(serde_json::from_str).transpose()?,
            created_at: row.get(6)?,
        })
    }
}

/// Longest title derived from the first user message.
const TITLE_CHARS: usize = 80;

const SESSION_COLUMNS: &str = "s.id, s.title, s.mode, s.model, CAST(s.created_at AS VARCHAR), \
     CAST(s.updated_at AS VARCHAR), \
     (SELECT count(*) FROM _quack_messages m WHERE m.session_id = s.id), s.created_by, \
     COALESCE(s.shared, false)";

/// Start a new session for `model` (`provider/model`) in `mode`, owned by
/// `created_by` in server mode.
///
/// # Errors
///
/// Returns an error if the insert fails.
pub fn create_session(
    db: &WorkspaceDb,
    model: &str,
    mode: ChatMode,
    created_by: Option<&UserId>,
) -> Result<SessionRow> {
    let id = SessionId::generate();
    db.connection().execute(
        "INSERT INTO _quack_sessions (id, model, mode, created_by) VALUES (?, ?, ?, ?)",
        duckdb::params![id, model, mode.as_str(), created_by],
    )?;
    get_session(db, &id)?
        .ok_or_else(|| Error::Analysis(String::from("session vanished after insert")))
}

/// Change a session's mode.
///
/// # Errors
///
/// Returns an error if the session does not exist or the update fails.
pub fn set_session_mode(db: &WorkspaceDb, session_id: &SessionId, mode: ChatMode) -> Result<()> {
    let changed = db.connection().execute(
        "UPDATE _quack_sessions SET mode = ? WHERE id = ?",
        duckdb::params![mode.as_str(), session_id],
    )?;
    if changed == 0 {
        return Err(Record::Session.missing(session_id.as_str()));
    }
    Ok(())
}

/// The session with the given id.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn get_session(db: &WorkspaceDb, id: &SessionId) -> Result<Option<SessionRow>> {
    let sql = format!("SELECT {SESSION_COLUMNS} FROM _quack_sessions s WHERE s.id = ?");
    let mut stmt = db.connection().prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![id])?;
    match rows.next()? {
        Some(row) => Ok(Some(SessionRow::try_from(row)?)),
        None => Ok(None),
    }
}

/// The most recently updated session, if any.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn latest_session(db: &WorkspaceDb) -> Result<Option<SessionRow>> {
    Ok(list_sessions(db, 1)?.into_iter().next())
}

/// Sessions, most recently updated first.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn list_sessions(db: &WorkspaceDb, limit: u32) -> Result<Vec<SessionRow>> {
    let sql = format!(
        "SELECT {SESSION_COLUMNS} FROM _quack_sessions s ORDER BY s.updated_at DESC, s.id DESC LIMIT ?"
    );
    let mut stmt = db.connection().prepare(&sql)?;
    let rows = stmt.query_map(duckdb::params![i64::from(limit)], |row| {
        SessionRow::try_from(row)
    })?;
    Ok(rows.collect::<duckdb::Result<_>>()?)
}

/// The sessions `viewer` may see, most recently updated first
/// ([`SessionRow::visible_to`]).
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn list_sessions_for(
    db: &WorkspaceDb,
    limit: u32,
    viewer: &SessionViewer,
) -> Result<Vec<SessionRow>> {
    let SessionViewer::User(user_id) = viewer else {
        return list_sessions(db, limit);
    };
    let sql = format!(
        "SELECT {SESSION_COLUMNS} FROM _quack_sessions s \
         WHERE s.created_by IS NULL OR s.created_by = ? OR s.shared \
         ORDER BY s.updated_at DESC, s.id DESC LIMIT ?"
    );
    let mut stmt = db.connection().prepare(&sql)?;
    let rows = stmt.query_map(duckdb::params![user_id, i64::from(limit)], |row| {
        SessionRow::try_from(row)
    })?;
    Ok(rows.collect::<duckdb::Result<_>>()?)
}

/// Share a session with every member of the workspace, or take it back.
///
/// # Errors
///
/// Returns an error if the update fails or the session does not exist.
pub fn set_session_shared(db: &WorkspaceDb, session_id: &SessionId, shared: bool) -> Result<()> {
    let changed = db.connection().execute(
        "UPDATE _quack_sessions SET shared = ? WHERE id = ?",
        duckdb::params![shared, session_id],
    )?;
    if changed == 0 {
        return Err(Record::Session.missing(session_id.as_str()));
    }
    Ok(())
}

/// Append one message and return its sequence number.
///
/// # Errors
///
/// Returns an error if the session does not exist or the insert fails.
pub fn append_message(
    db: &WorkspaceDb,
    session_id: &SessionId,
    role: MessageRole,
    content: &str,
    metadata: Option<&serde_json::Value>,
) -> Result<i64> {
    if get_session(db, session_id)?.is_none() {
        return Err(Record::Session.missing(session_id.as_str()));
    }
    let conn = db.connection();
    let seq: i64 = conn.query_row(
        "SELECT COALESCE(MAX(seq), 0) + 1 FROM _quack_messages WHERE session_id = ?",
        duckdb::params![session_id],
        |row| row.get(0),
    )?;
    let metadata_text = metadata.map(serde_json::to_string).transpose()?;
    conn.execute(
        "INSERT INTO _quack_messages (id, session_id, seq, role, content, metadata) \
         VALUES (?, ?, ?, ?, ?, ?)",
        duckdb::params![
            uuid::Uuid::now_v7().to_string(),
            session_id,
            seq,
            role.as_str(),
            content,
            metadata_text
        ],
    )?;
    conn.execute(
        "UPDATE _quack_sessions SET updated_at = now() WHERE id = ?",
        duckdb::params![session_id],
    )?;
    Ok(seq)
}

/// All messages of a session in order.
///
/// # Errors
///
/// Returns an error if the query fails or a stored role is unknown.
pub fn messages(db: &WorkspaceDb, session_id: &SessionId) -> Result<Vec<MessageRow>> {
    let mut stmt = db.connection().prepare(
        "SELECT id, session_id, seq, role, content, CAST(metadata AS VARCHAR), \
                CAST(created_at AS VARCHAR) \
         FROM _quack_messages WHERE session_id = ? ORDER BY seq",
    )?;
    let mut rows = stmt.query(duckdb::params![session_id])?;
    let mut out = Vec::new();
    // Not `query_map`: a stored role that does not parse is this crate's
    // error, which a `duckdb::Result` closure cannot carry.
    while let Some(row) = rows.next()? {
        out.push(MessageRow::try_from(row)?);
    }
    Ok(out)
}

/// Record a completed turn: the user message, one tool message per step,
/// and the assistant answer. Sets the session title from the first user
/// message.
///
/// # Errors
///
/// Returns an error if any insert fails.
pub fn record_turn(
    db: &WorkspaceDb,
    session_id: &SessionId,
    user_message: &str,
    response: &AgentResponse,
) -> Result<()> {
    let session =
        get_session(db, session_id)?.ok_or_else(|| Record::Session.missing(session_id.as_str()))?;

    append_message(db, session_id, MessageRole::User, user_message, None)?;

    for step in &response.steps {
        let metadata = serde_json::json!({
            "tool": step.tool,
            "detail": step.detail,
            "duration_ms": step.duration_ms,
        });
        append_message(
            db,
            session_id,
            MessageRole::Tool,
            &step.summary,
            Some(&metadata),
        )?;
    }

    let mut metadata = serde_json::Map::new();
    if let Some(chart) = &response.chart {
        metadata.insert(String::from("chart"), serde_json::to_value(chart)?);
    }
    if !response.citations.is_empty() {
        metadata.insert(
            String::from("citations"),
            serde_json::to_value(&response.citations)?,
        );
    }
    if response.write_refused {
        metadata.insert(String::from("write_refused"), serde_json::Value::Bool(true));
    }
    if !response.graph.is_empty() {
        metadata.insert(
            String::from("graph"),
            serde_json::to_value(&response.graph)?,
        );
    }
    if let Some(usage) = response.usage {
        metadata.insert(String::from("usage"), serde_json::to_value(usage)?);
    }
    let metadata = if metadata.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(metadata))
    };
    append_message(
        db,
        session_id,
        MessageRole::Assistant,
        &response.content,
        metadata.as_ref(),
    )?;

    if session.title.is_none() {
        let title: String = user_message
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(TITLE_CHARS)
            .collect();
        db.connection().execute(
            "UPDATE _quack_sessions SET title = ? WHERE id = ?",
            duckdb::params![title, session_id],
        )?;
    }
    Ok(())
}

/// Rough token estimate used for trimming history: four characters per
/// token, which errs on the side of sending less.
fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Prior user and assistant messages to replay to the model, newest last,
/// trimmed from the oldest end to fit `token_budget`. Tool messages are not
/// replayed; the assistant text already describes what the tools found.
///
/// # Errors
///
/// Returns an error if the messages cannot be read.
pub fn history_for_model(
    db: &WorkspaceDb,
    session_id: &SessionId,
    token_budget: u32,
) -> Result<Vec<rig::message::Message>> {
    let stored = messages(db, session_id)?;
    let budget = usize::try_from(token_budget).unwrap_or(usize::MAX);

    let mut kept: Vec<rig::message::Message> = Vec::new();
    let mut used = 0usize;
    for row in stored.iter().rev() {
        let message = match row.role {
            MessageRole::User => rig::message::Message::user(row.content.clone()),
            MessageRole::Assistant => rig::message::Message::assistant(row.content.clone()),
            MessageRole::Tool => continue,
        };
        let cost = estimate_tokens(&row.content);
        if used.saturating_add(cost) > budget {
            break;
        }
        used = used.saturating_add(cost);
        kept.push(message);
    }
    kept.reverse();
    Ok(kept)
}

/// How a session is exported.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    /// Questions, steps, and answers.
    #[default]
    Markdown,
    /// Every executed statement, each preceded by its question, as a
    /// runnable `.sql` file.
    Sql,
}

/// A session with every message in it, as the exports read it.
#[derive(Debug, Clone)]
pub struct Transcript {
    pub session: SessionRow,
    pub messages: Vec<MessageRow>,
}

impl Transcript {
    /// The session's transcript.
    ///
    /// # Errors
    ///
    /// Returns an error if a read fails.
    pub fn load(db: &WorkspaceDb, session: SessionRow) -> Result<Self> {
        let messages = messages(db, &session.id)?;
        Ok(Self { session, messages })
    }

    /// The transcript in `format`.
    ///
    /// # Errors
    ///
    /// Returns an error only if formatting into the output buffer fails.
    pub fn render(&self, format: ExportFormat) -> Result<String> {
        match format {
            ExportFormat::Markdown => self.to_markdown(),
            ExportFormat::Sql => self.to_sql(),
        }
    }

    /// Every executed statement in order, each preceded by the question that
    /// led to it, as a runnable `.sql` file.
    fn to_sql(&self) -> Result<String> {
        let mut out = String::new();
        let mut question: Option<&str> = None;
        for row in &self.messages {
            match row.role {
                MessageRole::User => question = Some(row.content.as_str()),
                MessageRole::Tool => {
                    let Some(meta) = &row.metadata else { continue };
                    let tool = meta.get("tool").and_then(serde_json::Value::as_str);
                    if !matches!(tool, Some("run_sql" | "create_chart")) {
                        continue;
                    }
                    let Some(sql) = meta.get("detail").and_then(serde_json::Value::as_str) else {
                        continue;
                    };
                    if let Some(q) = question.take() {
                        for line in q.lines() {
                            writeln!(out, "-- {line}")?;
                        }
                    }
                    writeln!(out, "-- {}", row.content)?;
                    let statement = sql.trim().trim_end_matches(';');
                    writeln!(out, "{statement};\n")?;
                }
                MessageRole::Assistant => {}
            }
        }
        Ok(out)
    }

    /// The transcript as Markdown: questions, steps, and answers.
    fn to_markdown(&self) -> Result<String> {
        let session = &self.session;
        let mut out = String::new();
        let title = session.title.as_deref().unwrap_or("Session");
        writeln!(out, "# {title}\n")?;
        writeln!(
            out,
            "Session `{}` · model `{}` · started {}\n",
            session.id, session.model, session.created_at
        )?;
        for row in &self.messages {
            match row.role {
                MessageRole::User => {
                    writeln!(out, "## {}\n", row.content.trim())?;
                }
                MessageRole::Tool => {
                    let tool = row
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("tool"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("tool");
                    let detail = row
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("detail"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("");
                    let ms = row
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("duration_ms"))
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0);
                    writeln!(out, "**{tool}** — {}, {ms} ms\n", row.content)?;
                    if !detail.is_empty() {
                        let lang = if matches!(tool, "run_sql" | "create_chart") {
                            "sql"
                        } else {
                            "text"
                        };
                        writeln!(out, "```{lang}\n{}\n```\n", detail.trim())?;
                    }
                }
                MessageRole::Assistant => {
                    writeln!(out, "{}\n", row.content.trim())?;
                    if row.metadata.as_ref().and_then(|m| m.get("chart")).is_some() {
                        writeln!(out, "_(chart attached)_\n")?;
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Remove a session and every message in it. Returns whether it existed.
///
/// # Errors
///
/// Returns an error if a delete fails.
pub fn delete_session(db: &WorkspaceDb, session_id: &SessionId) -> Result<bool> {
    if get_session(db, session_id)?.is_none() {
        return Ok(false);
    }
    db.connection().execute(
        "DELETE FROM _quack_messages WHERE session_id = ?",
        duckdb::params![session_id],
    )?;
    db.connection().execute(
        "DELETE FROM _quack_sessions WHERE id = ?",
        duckdb::params![session_id],
    )?;
    Ok(true)
}

/// Remove a session that never recorded a message (a failed first turn).
/// Returns whether it was removed.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn delete_if_empty(db: &WorkspaceDb, session_id: &SessionId) -> Result<bool> {
    let Some(session) = get_session(db, session_id)? else {
        return Ok(false);
    };
    if session.message_count > 0 {
        return Ok(false);
    }
    db.connection().execute(
        "DELETE FROM _quack_sessions WHERE id = ?",
        duckdb::params![session_id],
    )?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::agent::TokenUsage;
    use crate::analysis::events::ToolStep;

    fn db() -> WorkspaceDb {
        WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| open_failed(&e.to_string()))
    }

    #[expect(clippy::panic, reason = "test helper: in-memory DuckDB must open")]
    fn open_failed(msg: &str) -> WorkspaceDb {
        panic!("in-memory DuckDB failed to open: {msg}");
    }

    fn response(content: &str, steps: Vec<ToolStep>) -> AgentResponse {
        AgentResponse {
            content: content.to_owned(),
            steps,
            citations: Vec::new(),
            chart: None,
            graph: Vec::new(),
            write_refused: false,
            cancelled: false,
            usage: None,
        }
    }

    fn step(tool: &str, detail: &str, summary: &str) -> ToolStep {
        ToolStep {
            tool: tool.to_owned(),
            detail: detail.to_owned(),
            summary: summary.to_owned(),
            duration_ms: 7,
        }
    }

    #[test]
    fn delete_session_removes_its_messages_too() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        let session =
            create_session(&db, "m", ChatMode::Chat, None).unwrap_or_else(|e| fail(&e.to_string()));
        assert!(append_message(&db, &session.id, MessageRole::User, "hi", None).is_ok());
        assert!(delete_session(&db, &session.id).is_ok_and(|d| d));
        assert!(get_session(&db, &session.id).is_ok_and(|s| s.is_none()));
        let left: i64 = db
            .connection()
            .query_row("SELECT count(*) FROM _quack_messages", [], |r| r.get(0))
            .unwrap_or(-1);
        assert_eq!(left, 0);
        assert!(delete_session(&db, &session.id).is_ok_and(|d| !d));
    }

    #[test]
    fn created_by_filters_the_listing_unless_the_viewer_sees_all() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        let mine = create_session(&db, "m", ChatMode::Chat, Some(&UserId::from("u1")))
            .unwrap_or_else(|e| fail(&e.to_string()));
        let theirs = create_session(&db, "m", ChatMode::Chat, Some(&UserId::from("u2")))
            .unwrap_or_else(|e| fail(&e.to_string()));
        let cli =
            create_session(&db, "m", ChatMode::Chat, None).unwrap_or_else(|e| fail(&e.to_string()));
        let visible = list_sessions_for(&db, 10, &SessionViewer::User(UserId::from("u1")));
        assert!(visible.is_ok_and(|v| {
            v.iter()
                .map(|s| s.id.as_str())
                .eq([cli.id.as_str(), mine.id.as_str()])
        }));
        assert!(list_sessions_for(&db, 10, &SessionViewer::All).is_ok_and(|v| v.len() == 3));
        assert!(
            mine.visible_to(&SessionViewer::User(UserId::from("u1")))
                && !theirs.visible_to(&SessionViewer::User(UserId::from("u1")))
        );
        assert!(
            theirs.visible_to(&SessionViewer::All)
                && cli.visible_to(&SessionViewer::User(UserId::from("u1")))
        );
        assert_eq!(mine.created_by, Some(UserId::from("u1")));
        assert!(!mine.shared);

        // Sharing opens the session to other members; unsharing closes it.
        set_session_shared(&db, &theirs.id, true).unwrap_or_else(|e| fail(&e.to_string()));
        let theirs = get_session(&db, &theirs.id)
            .ok()
            .flatten()
            .unwrap_or_else(|| fail("session vanished"));
        assert!(theirs.shared && theirs.visible_to(&SessionViewer::User(UserId::from("u1"))));
        assert!(
            list_sessions_for(&db, 10, &SessionViewer::User(UserId::from("u1")))
                .is_ok_and(|v| v.len() == 3)
        );
        set_session_shared(&db, &theirs.id, false).unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            list_sessions_for(&db, 10, &SessionViewer::User(UserId::from("u1")))
                .is_ok_and(|v| v.len() == 2)
        );
        assert!(set_session_shared(&db, &SessionId::from("missing"), true).is_err());
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn record_turn_writes_user_tool_and_assistant_in_order_and_titles_session() {
        let db = db();
        let session = create_session(&db, "ollama/llama3", ChatMode::Chat, None).unwrap();
        assert!(session.title.is_none());
        assert_eq!(session.message_count, 0);
        assert_eq!(session.mode, ChatMode::Chat);
        set_session_mode(&db, &session.id, ChatMode::Query).unwrap();
        assert_eq!(
            get_session(&db, &session.id).unwrap().unwrap().mode,
            ChatMode::Query
        );
        assert!(set_session_mode(&db, &SessionId::from("missing"), ChatMode::Chat).is_err());

        record_turn(
            &db,
            &session.id,
            "  how many   claims are open?  ",
            &response(
                "There are 4 open claims.",
                vec![step("run_sql", "SELECT count(*) FROM claims", "1 rows")],
            ),
        )
        .unwrap();

        let rows = messages(&db, &session.id).unwrap();
        let roles: Vec<MessageRole> = rows.iter().map(|r| r.role).collect();
        assert_eq!(
            roles,
            vec![MessageRole::User, MessageRole::Tool, MessageRole::Assistant]
        );
        assert_eq!(
            rows.iter().map(|r| r.seq).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        let tool = rows.iter().find(|r| r.role == MessageRole::Tool).unwrap();
        assert_eq!(tool.content, "1 rows");
        assert_eq!(
            tool.metadata
                .as_ref()
                .and_then(|m| m.get("tool"))
                .and_then(|v| v.as_str()),
            Some("run_sql")
        );

        let session = get_session(&db, &session.id).unwrap().unwrap();
        assert_eq!(session.title.as_deref(), Some("how many claims are open?"));
        assert_eq!(session.message_count, 3);
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn token_usage_round_trips_through_the_assistant_metadata() {
        let db = db();
        let session = create_session(&db, "m", ChatMode::Chat, None).unwrap();
        let mut answer = response("There are 4 open claims.", vec![]);
        answer.usage = Some(TokenUsage {
            input_tokens: 1_204,
            output_tokens: 57,
            total_tokens: 1_261,
        });
        record_turn(&db, &session.id, "how many?", &answer).unwrap();

        let rows = messages(&db, &session.id).unwrap();
        let assistant = rows
            .iter()
            .find(|r| r.role == MessageRole::Assistant)
            .unwrap();
        let usage = assistant
            .metadata
            .as_ref()
            .and_then(|m| m.get("usage"))
            .unwrap();
        let count = |key: &str| usage.get(key).and_then(serde_json::Value::as_u64);
        assert_eq!(count("input_tokens"), Some(1_204));
        assert_eq!(count("output_tokens"), Some(57));
        assert_eq!(count("total_tokens"), Some(1_261));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn a_turn_without_reported_usage_records_no_usage_key() {
        let db = db();
        let session = create_session(&db, "m", ChatMode::Chat, None).unwrap();
        record_turn(&db, &session.id, "q", &response("a", vec![])).unwrap();

        let rows = messages(&db, &session.id).unwrap();
        let assistant = rows
            .iter()
            .find(|r| r.role == MessageRole::Assistant)
            .unwrap();
        // No chart, citations, graph or usage: the whole metadata column
        // stays NULL rather than holding an empty object.
        assert!(assistant.metadata.is_none());
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn latest_session_is_the_most_recently_updated() {
        let db = db();
        let first = create_session(&db, "m", ChatMode::Query, None).unwrap();
        let second = create_session(&db, "m", ChatMode::Query, None).unwrap();
        assert_eq!(latest_session(&db).unwrap().unwrap().id, second.id);
        record_turn(&db, &first.id, "q", &response("a", vec![])).unwrap();
        assert_eq!(latest_session(&db).unwrap().unwrap().id, first.id);
        assert_eq!(list_sessions(&db, 10).unwrap().len(), 2);
    }

    #[test]
    fn append_to_missing_session_is_an_error() {
        let db = db();
        let err = append_message(&db, &SessionId::from("nope"), MessageRole::User, "x", None).err();
        assert!(err.is_some_and(|e| matches!(
            e,
            Error::NotFound {
                record: Record::Session,
                ..
            }
        )));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn history_skips_tool_messages_and_trims_oldest_first() {
        let db = db();
        let session = create_session(&db, "m", ChatMode::Query, None).unwrap();
        record_turn(
            &db,
            &session.id,
            "first question",
            &response("first answer", vec![step("run_sql", "SELECT 1", "1 rows")]),
        )
        .unwrap();
        record_turn(
            &db,
            &session.id,
            "second question",
            &response("second answer", vec![]),
        )
        .unwrap();

        let all = history_for_model(&db, &session.id, 10_000).unwrap();
        assert_eq!(all.len(), 4);

        // "second question" + "second answer" ≈ 8 tokens; budget of 9 keeps only those two.
        let trimmed = history_for_model(&db, &session.id, 9).unwrap();
        assert_eq!(trimmed.len(), 2);

        let none = history_for_model(&db, &session.id, 1).unwrap();
        assert!(none.is_empty());
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn export_sql_pairs_questions_with_statements() {
        let db = db();
        let session = create_session(&db, "m", ChatMode::Query, None).unwrap();
        record_turn(
            &db,
            &session.id,
            "open claims?",
            &response(
                "4",
                vec![
                    step("list_tables", "", "2 tables"),
                    step(
                        "run_sql",
                        "SELECT count(*) FROM claims WHERE open;",
                        "1 rows",
                    ),
                ],
            ),
        )
        .unwrap();
        let session = get_session(&db, &session.id).unwrap().unwrap();
        let sql = Transcript::load(&db, session)
            .unwrap()
            .render(ExportFormat::Sql)
            .unwrap();
        assert_eq!(
            sql,
            "-- open claims?\n-- 1 rows\nSELECT count(*) FROM claims WHERE open;\n\n"
        );
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn delete_if_empty_removes_only_sessions_without_messages() {
        let db = db();
        let empty = create_session(&db, "m", ChatMode::Query, None).unwrap();
        let used = create_session(&db, "m", ChatMode::Query, None).unwrap();
        record_turn(&db, &used.id, "q", &response("a", vec![])).unwrap();
        assert!(delete_if_empty(&db, &empty.id).unwrap());
        assert!(!delete_if_empty(&db, &used.id).unwrap());
        assert!(!delete_if_empty(&db, &SessionId::from("missing")).unwrap());
        assert_eq!(list_sessions(&db, 10).unwrap().len(), 1);
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn export_markdown_has_headings_steps_and_answers() {
        let db = db();
        let session = create_session(&db, "ollama/llama3", ChatMode::Chat, None).unwrap();
        record_turn(
            &db,
            &session.id,
            "open claims?",
            &response("Four.", vec![step("run_sql", "SELECT 1", "1 rows")]),
        )
        .unwrap();
        let session = get_session(&db, &session.id).unwrap().unwrap();
        let md = Transcript::load(&db, session)
            .unwrap()
            .render(ExportFormat::Markdown)
            .unwrap();
        assert!(md.starts_with("# open claims?\n"));
        assert!(md.contains("## open claims?\n"));
        assert!(md.contains("**run_sql** — 1 rows, 7 ms\n\n```sql\nSELECT 1\n```"));
        assert!(md.trim_end().ends_with("Four."));
    }
}
