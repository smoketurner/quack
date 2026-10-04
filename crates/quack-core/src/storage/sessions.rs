//! Sessions and messages, stored inside the workspace file.
//!
//! A session is a conversation; a message is one entry in it. Every agent
//! turn records the user message, one tool message per recorded step, and
//! the assistant answer, so a session is a complete record that can be
//! resumed, listed, and exported without leaving the classification boundary.

use std::fmt::Write as _;
use std::sync::Arc;

use jiff::Timestamp;

use crate::analysis::agent::{AgentResponse, TokenUsage};
use crate::analysis::chart::ChartSpec;
use crate::analysis::citations::Citation;
use crate::analysis::events::{ToolName, ToolStep};
use crate::error::{Error, Record, Result};
use crate::graph::GraphResult;
use crate::ids::{MessageId, SessionId, SummaryId, UserId};
use crate::text::Tokens;
use rig::id::ConversationId;
use rig::memory::{ConversationMemory, MemoryError, MemoryPolicy, TokenWindowMemory};
use rig::message::{AssistantContent, Message, UserContent};
use rig::wasm_compat::WasmBoxedFuture;

use super::workspace::WorkspaceDb;
use super::writer::Writer;

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

/// Who besides its creator may read a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(from = "bool")]
pub enum Sharing {
    Private,
    /// Every member of the workspace.
    Shared,
}

flag_enum!(Sharing, false => Private, true => Shared);

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

/// What a tool message stores beside its summary, which is the message's
/// content.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolMeta {
    pub tool: ToolName,
    pub detail: String,
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<u64>,
}

impl ToolMeta {
    /// The step again, with the summary the message stored as its content.
    #[must_use]
    pub fn step(&self, summary: String) -> ToolStep {
        ToolStep {
            tool: self.tool,
            detail: self.detail.clone(),
            summary,
            rows: self.rows,
            duration_ms: self.duration_ms,
        }
    }
}

impl From<&ToolStep> for ToolMeta {
    fn from(step: &ToolStep) -> Self {
        Self {
            tool: step.tool,
            detail: step.detail.clone(),
            duration_ms: step.duration_ms,
            rows: step.rows,
        }
    }
}

/// What an assistant message stores beside its answer. Empty parts are
/// left out of the stored JSON.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AssistantMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chart: Option<ChartSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub citations: Vec<Citation>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub write_refused: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub graph: Vec<GraphResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// How long the turn took, in milliseconds; absent on turns recorded before it was kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

impl AssistantMeta {
    /// The parts of `response` the transcript keeps, or `None` when it has
    /// none of them.
    #[must_use]
    pub fn of(response: &AgentResponse) -> Option<Self> {
        let meta = Self {
            chart: response.chart.clone(),
            citations: response.citations.clone(),
            write_refused: response.write_refused,
            graph: response.graph.clone(),
            usage: response.usage,
            duration_ms: response.duration_ms,
        };
        (meta != Self::default()).then_some(meta)
    }
}

/// A message's metadata, by its role. Serialized untagged, so the stored
/// JSON and every API body are the fields themselves.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(untagged)]
pub enum MessageMeta {
    Tool(ToolMeta),
    Assistant(AssistantMeta),
}

impl MessageMeta {
    /// Decode a stored metadata column for a message of `role`. A column
    /// that does not decode is dropped with a warning rather than making
    /// the whole session unreadable.
    fn decode(id: &MessageId, role: MessageRole, text: &str) -> Option<Self> {
        let decoded = match role {
            MessageRole::User => return None,
            MessageRole::Tool => serde_json::from_str(text).map(Self::Tool),
            MessageRole::Assistant => serde_json::from_str(text).map(Self::Assistant),
        };
        decoded
            .inspect_err(
                |error| tracing::warn!(message = %id, %error, "unreadable message metadata"),
            )
            .ok()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MessageRow {
    pub id: MessageId,
    pub session_id: SessionId,
    pub seq: i64,
    pub role: MessageRole,
    pub content: String,
    pub metadata: Option<MessageMeta>,
    pub created_at: String,
}

impl MessageRow {
    /// The tool call a tool message records.
    #[must_use]
    pub fn tool(&self) -> Option<&ToolMeta> {
        match &self.metadata {
            Some(MessageMeta::Tool(meta)) => Some(meta),
            Some(MessageMeta::Assistant(_)) | None => None,
        }
    }

    /// The chart, citations, graph results, and usage an answer carried.
    #[must_use]
    pub fn assistant(&self) -> Option<&AssistantMeta> {
        match &self.metadata {
            Some(MessageMeta::Assistant(meta)) => Some(meta),
            Some(MessageMeta::Tool(_)) | None => None,
        }
    }
}

/// A row of `id, session_id, seq, role, content, metadata, created_at`.
impl TryFrom<&duckdb::Row<'_>> for MessageRow {
    type Error = Error;

    fn try_from(row: &duckdb::Row<'_>) -> Result<Self> {
        let id: MessageId = row.get(0)?;
        let role: MessageRole = row.get::<_, String>(3)?.parse()?;
        let metadata: Option<String> = row.get(5)?;
        Ok(Self {
            metadata: metadata
                .as_deref()
                .and_then(|text| MessageMeta::decode(&id, role, text)),
            id,
            session_id: row.get(1)?,
            seq: row.get(2)?,
            role,
            content: row.get(4)?,
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
pub fn set_session_sharing(
    db: &WorkspaceDb,
    session_id: &SessionId,
    sharing: Sharing,
) -> Result<()> {
    let changed = db.connection().execute(
        "UPDATE _quack_sessions SET shared = ? WHERE id = ?",
        duckdb::params![sharing, session_id],
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
    metadata: Option<&MessageMeta>,
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
            MessageId::generate(),
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

/// Record a completed turn: the user message, stamped `asked_at`, one tool
/// message per step, and the assistant answer. Sets the session title from
/// the first user message.
///
/// # Errors
///
/// Returns an error if any insert fails.
pub fn record_turn(
    db: &WorkspaceDb,
    session_id: &SessionId,
    user_message: &str,
    asked_at: Timestamp,
    response: &AgentResponse,
) -> Result<()> {
    let session =
        get_session(db, session_id)?.ok_or_else(|| Record::Session.missing(session_id.as_str()))?;

    // The turn is recorded once it ends; the question keeps the time it was asked.
    let seq = append_message(db, session_id, MessageRole::User, user_message, None)?;
    db.connection().execute(
        "UPDATE _quack_messages SET created_at = CAST(? AS TIMESTAMP) \
         WHERE session_id = ? AND seq = ?",
        duckdb::params![
            asked_at.strftime("%Y-%m-%d %H:%M:%S%.6f").to_string(),
            session_id,
            seq
        ],
    )?;

    for step in &response.steps {
        append_message(
            db,
            session_id,
            MessageRole::Tool,
            &step.summary,
            Some(&MessageMeta::Tool(ToolMeta::from(step))),
        )?;
    }

    let metadata = AssistantMeta::of(response).map(MessageMeta::Assistant);
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

/// The session's turns to replay to the model, oldest first: each question
/// and its answer, tool rows skipped (the answer already says what the
/// tools found). A turn with no text on either side is left out whole: it
/// tells the model nothing, and a provider that drops empty text blocks
/// (Anthropic) would be left with a message of no content and refuse every
/// later turn of the session.
///
/// # Errors
///
/// Returns an error if the messages cannot be read.
pub fn session_turns(db: &WorkspaceDb, session_id: &SessionId) -> Result<Vec<Message>> {
    let stored = messages(db, session_id)?;
    let mut turns = Vec::new();
    let mut question = None;
    for row in &stored {
        match row.role {
            MessageRole::User => question = Some(row),
            MessageRole::Assistant => {
                let Some(asked) = question.take() else {
                    continue;
                };
                if asked.content.trim().is_empty() || row.content.trim().is_empty() {
                    continue;
                }
                turns.push(Message::user(asked.content.clone()));
                turns.push(Message::assistant(row.content.clone()));
            }
            MessageRole::Tool => {}
        }
    }
    Ok(turns)
}

/// A session's replayable turns as rig's [`ConversationMemory`], under the
/// session's id: `load` reads [`session_turns`]. quack records each turn
/// itself ([`record_turn`]: the checked answer, its tool rows, and its
/// usage), so `append` and `clear` store nothing, and a turn is in the next
/// `load` as soon as it is recorded.
pub struct SessionMemory {
    db: Arc<Writer>,
}

impl SessionMemory {
    #[must_use]
    pub const fn new(db: Arc<Writer>) -> Self {
        Self { db }
    }
}

impl ConversationMemory for SessionMemory {
    fn load<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, std::result::Result<Vec<Message>, MemoryError>> {
        let session = SessionId::from(conversation_id.as_str());
        Box::pin(async move {
            self.db
                .run(move |db| session_turns(db, &session))
                .await
                .map_err(MemoryError::backend)
        })
    }

    fn append<'a>(
        &'a self,
        _conversation_id: &'a ConversationId,
        _messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, std::result::Result<(), MemoryError>> {
        Box::pin(async { Ok(()) })
    }

    fn clear<'a>(
        &'a self,
        _conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, std::result::Result<(), MemoryError>> {
        Box::pin(async { Ok(()) })
    }
}

/// The words a message says: its text parts joined, without tool calls,
/// tool results, images, or reasoning.
pub trait SpokenText {
    fn spoken_text(&self) -> String;
}

impl SpokenText for Message {
    fn spoken_text(&self) -> String {
        match self {
            Self::User { content } => content
                .iter()
                .filter_map(|part| match part {
                    UserContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect(),
            Self::Assistant(turn) => turn
                .content
                .iter()
                .filter_map(|part| match part {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect(),
            Self::System { content } => content.clone(),
        }
    }
}

/// The history window: rig's [`TokenWindowMemory`] over
/// `[analysis].history_token_budget`, counting four characters per token
/// ([`Tokens::estimate`]) so budgets read as before, then the window's
/// leading answer dropped when its question fell outside: a replayed thread
/// opens on a question.
pub struct TranscriptWindow(TokenWindowMemory);

impl TranscriptWindow {
    #[must_use]
    pub fn new(budget: Tokens) -> Self {
        let budget = usize::try_from(budget.get()).unwrap_or(usize::MAX);
        Self(TokenWindowMemory::new(budget, |message: &Message| {
            usize::try_from(Tokens::estimate(&message.spoken_text()).get()).unwrap_or(usize::MAX)
        }))
    }
}

impl MemoryPolicy for TranscriptWindow {
    fn apply(&self, messages: Vec<Message>) -> std::result::Result<Vec<Message>, MemoryError> {
        Ok(self.apply_with_demoted(messages)?.0)
    }

    fn apply_with_demoted(
        &self,
        messages: Vec<Message>,
    ) -> std::result::Result<(Vec<Message>, Vec<Message>), MemoryError> {
        let (mut kept, mut demoted) = self.0.apply_with_demoted(messages)?;
        let answers = kept
            .iter()
            .take_while(|m| matches!(m, Message::Assistant { .. }))
            .count();
        demoted.extend(kept.drain(..answers));
        Ok((kept, demoted))
    }
}

/// A summary of a session's earliest replayable messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredSummary {
    /// How many of [`session_turns`]' messages, oldest first, it covers.
    pub covers: usize,
    pub text: String,
}

/// The session's newest summary, if one was made: the row with the greatest
/// `id`, a UUID v7 so ids sort by creation time (`ids.rs`), not the row that
/// covers the most messages — `covers` is not monotonic once a budget grows
/// and the compactor re-summarizes from scratch over a smaller prefix.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn latest_summary(db: &WorkspaceDb, session_id: &SessionId) -> Result<Option<StoredSummary>> {
    let mut stmt = db.connection().prepare(
        "SELECT covers, summary FROM _quack_session_summaries WHERE session_id = ? \
         ORDER BY id DESC LIMIT 1",
    )?;
    let mut rows = stmt.query(duckdb::params![session_id])?;
    let Some(row) = rows.next()? else {
        return Ok(None);
    };
    let covers: i64 = row.get(0)?;
    Ok(Some(StoredSummary {
        covers: usize::try_from(covers).unwrap_or_default(),
        text: row.get(1)?,
    }))
}

/// Keep a new summary of the session's first `covers` replayable messages.
///
/// # Errors
///
/// Returns an error if the insert fails.
pub fn save_summary(
    db: &WorkspaceDb,
    session_id: &SessionId,
    covers: usize,
    text: &str,
) -> Result<()> {
    db.connection().execute(
        "INSERT INTO _quack_session_summaries (id, session_id, covers, summary) VALUES (?, ?, ?, ?)",
        duckdb::params![
            SummaryId::generate(),
            session_id,
            i64::try_from(covers).unwrap_or(i64::MAX),
            text
        ],
    )?;
    Ok(())
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
                    let Some(meta) = row.tool().filter(|m| m.tool.takes_sql()) else {
                        continue;
                    };
                    let sql = &meta.detail;
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
                    let Some(meta) = row.tool() else {
                        writeln!(out, "**tool** — {}\n", row.content)?;
                        continue;
                    };
                    writeln!(
                        out,
                        "**{}** — {}, {} ms\n",
                        meta.tool, row.content, meta.duration_ms
                    )?;
                    if !meta.detail.is_empty() {
                        let lang = if meta.tool.takes_sql() { "sql" } else { "text" };
                        writeln!(out, "```{lang}\n{}\n```\n", meta.detail.trim())?;
                    }
                }
                MessageRole::Assistant => {
                    writeln!(out, "{}\n", row.content.trim())?;
                    if row.assistant().is_some_and(|m| m.chart.is_some()) {
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
        "DELETE FROM _quack_session_summaries WHERE session_id = ?",
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
    use crate::embedding::Dimension;

    /// The history the window keeps of a session's turns under `budget`.
    fn windowed(db: &WorkspaceDb, session: &SessionId, budget: u32) -> Result<Vec<Message>> {
        let turns = session_turns(db, session)?;
        TranscriptWindow::new(Tokens::new(budget))
            .apply(turns)
            .map_err(|e| Error::Analysis(e.to_string()))
    }

    /// The trim the window replaced: whole turns, newest first, until the
    /// next one would pass the budget.
    fn turn_by_turn(turns: &[Message], budget: u32) -> Vec<Message> {
        let mut kept: Vec<Message> = Vec::new();
        let mut used = Tokens::default();
        for pair in turns.chunks(2).rev() {
            let cost = pair.iter().fold(Tokens::default(), |sum, m| {
                sum.saturating_add(Tokens::estimate(&m.spoken_text()))
            });
            if used.saturating_add(cost) > Tokens::new(budget) {
                break;
            }
            used = used.saturating_add(cost);
            kept.splice(0..0, pair.iter().cloned());
        }
        kept
    }

    /// rig's token window keeps exactly what the turn-by-turn trim kept,
    /// at every budget.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn the_window_keeps_what_the_turn_by_turn_trim_kept() {
        let db = db();
        let session = create_session(&db, "m", ChatMode::Chat, None).unwrap();
        for (question, answer) in [
            ("a", "bbbbbbbbbbbbbbbbbbbb"),
            ("cccccccccccccccc", "d"),
            ("eeeeeeee", "ffffffff"),
            ("g", "h"),
            ("iiiiiiiiiiiiiiiiiiiiiiii", "jjjjjjjjjjjj"),
            ("kkk", "llllll"),
        ] {
            record_turn(
                &db,
                &session.id,
                question,
                Timestamp::now(),
                &response(answer, vec![]),
            )
            .unwrap();
        }
        let turns = session_turns(&db, &session.id).unwrap();
        for budget in 0..=128 {
            assert_eq!(
                windowed(&db, &session.id, budget).unwrap(),
                turn_by_turn(&turns, budget),
                "budget {budget}"
            );
        }
    }

    /// A flag enum reads the JSON boolean an API body carries and gives
    /// the same boolean back.
    #[test]
    fn sharing_reads_and_gives_back_the_json_flag() {
        let parsed: Vec<Sharing> = ["true", "false"]
            .iter()
            .filter_map(|s| serde_json::from_str(s).ok())
            .collect();
        assert_eq!(parsed, [Sharing::Shared, Sharing::Private]);
        assert!(bool::from(Sharing::Shared) && !bool::from(Sharing::Private));
        assert!(serde_json::from_str::<Sharing>("\"shared\"").is_err());
    }

    fn db() -> WorkspaceDb {
        WorkspaceDb::open_in_memory(Dimension::new(4))
            .unwrap_or_else(|e| open_failed(&e.to_string()))
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
            duration_ms: None,
        }
    }

    fn step(tool: ToolName, detail: &str, summary: &str) -> ToolStep {
        ToolStep {
            tool,
            detail: detail.to_owned(),
            summary: summary.to_owned(),
            rows: None,
            duration_ms: 7,
        }
    }

    #[test]
    fn delete_session_removes_its_messages_too() {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
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
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
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
        set_session_sharing(&db, &theirs.id, Sharing::Shared)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let theirs = get_session(&db, &theirs.id)
            .ok()
            .flatten()
            .unwrap_or_else(|| fail("session vanished"));
        assert!(theirs.shared && theirs.visible_to(&SessionViewer::User(UserId::from("u1"))));
        assert!(
            list_sessions_for(&db, 10, &SessionViewer::User(UserId::from("u1")))
                .is_ok_and(|v| v.len() == 3)
        );
        set_session_sharing(&db, &theirs.id, Sharing::Private)
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            list_sessions_for(&db, 10, &SessionViewer::User(UserId::from("u1")))
                .is_ok_and(|v| v.len() == 2)
        );
        assert!(set_session_sharing(&db, &SessionId::from("missing"), Sharing::Shared).is_err());
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
            Timestamp::now(),
            &response(
                "There are 4 open claims.",
                vec![step(
                    ToolName::RunSql,
                    "SELECT count(*) FROM claims",
                    "1 rows",
                )],
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
        assert_eq!(tool.tool().map(|m| m.tool), Some(ToolName::RunSql));

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
        record_turn(&db, &session.id, "how many?", Timestamp::now(), &answer).unwrap();

        let rows = messages(&db, &session.id).unwrap();
        let assistant = rows
            .iter()
            .find(|r| r.role == MessageRole::Assistant)
            .unwrap();
        assert_eq!(
            assistant.assistant().and_then(|m| m.usage),
            Some(TokenUsage {
                input_tokens: 1_204,
                output_tokens: 57,
                total_tokens: 1_261,
            })
        );
    }

    /// The question keeps the time it was asked, not the time the turn was
    /// recorded, and the answer keeps how long it took. Both columns hold
    /// UTC, which the web UI turns into the viewer's time zone.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn a_turn_keeps_when_it_was_asked_and_how_long_it_took() {
        let db = db();
        let session = create_session(&db, "m", ChatMode::Chat, None).unwrap();
        let asked = Timestamp::now()
            .checked_sub(jiff::SignedDuration::from_secs(90))
            .unwrap();
        let mut answer = response("4", vec![]);
        answer.duration_ms = Some(2_345);
        record_turn(&db, &session.id, "how many?", asked, &answer).unwrap();

        let rows = messages(&db, &session.id).unwrap();
        let utc = |text: &str| {
            text.parse::<jiff::civil::DateTime>()
                .unwrap()
                .to_zoned(jiff::tz::TimeZone::UTC)
                .unwrap()
                .timestamp()
        };
        let question = rows.iter().find(|r| r.role == MessageRole::User).unwrap();
        assert_eq!(
            utc(&question.created_at).as_microsecond(),
            asked.as_microsecond()
        );
        let assistant = rows
            .iter()
            .find(|r| r.role == MessageRole::Assistant)
            .unwrap();
        let recorded = utc(&assistant.created_at);
        assert!(
            Timestamp::now().duration_since(recorded).abs() < jiff::SignedDuration::from_secs(60),
            "the default now() is not UTC: {recorded}"
        );
        assert_eq!(
            assistant.assistant().and_then(|m| m.duration_ms),
            Some(2_345)
        );
    }

    /// The typed metadata writes the same JSON keys the column always held,
    /// leaving out what a message did not have, and reads it back.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn metadata_is_stored_under_its_field_names_and_read_back() {
        let db = db();
        let session = create_session(&db, "m", ChatMode::Chat, None).unwrap();
        let mut sql = step(ToolName::RunSql, "SELECT 1", "1 rows");
        sql.rows = Some(1);
        let mut answer = response("One.", vec![sql]);
        answer.write_refused = true;
        record_turn(&db, &session.id, "one?", Timestamp::now(), &answer).unwrap();

        let stored: Vec<String> = {
            let conn = db.connection();
            let mut stmt = conn
                .prepare(
                    "SELECT CAST(metadata AS VARCHAR) FROM _quack_messages \
                     WHERE metadata IS NOT NULL ORDER BY seq",
                )
                .unwrap();
            stmt.query_map([], |row| row.get(0))
                .unwrap()
                .collect::<duckdb::Result<_>>()
                .unwrap()
        };
        let json: Vec<serde_json::Value> = stored
            .iter()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        assert_eq!(
            json,
            vec![
                serde_json::json!({"tool": "run_sql", "detail": "SELECT 1", "duration_ms": 7, "rows": 1}),
                serde_json::json!({"write_refused": true}),
            ]
        );

        let rows = messages(&db, &session.id).unwrap();
        let tool = rows.iter().find_map(MessageRow::tool).unwrap();
        assert_eq!(tool.step(String::from("1 rows")).rows, Some(1));
        let assistant = rows.iter().find_map(MessageRow::assistant).unwrap();
        assert!(assistant.write_refused && assistant.chart.is_none());
    }

    /// A stored column that no longer decodes (a tool this build does not
    /// know) loses its metadata, not the session.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn metadata_that_does_not_decode_is_dropped_not_fatal() {
        let db = db();
        let session = create_session(&db, "m", ChatMode::Chat, None).unwrap();
        db.connection()
            .execute(
                "INSERT INTO _quack_messages (id, session_id, seq, role, content, metadata) \
                 VALUES ('m1', ?, 1, 'tool', 'done', '{\"tool\": \"retired_tool\"}')",
                duckdb::params![session.id],
            )
            .unwrap();
        let rows = messages(&db, &session.id).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows.first().is_some_and(|r| r.metadata.is_none()));
        let markdown = Transcript::load(&db, get_session(&db, &session.id).unwrap().unwrap())
            .unwrap()
            .render(ExportFormat::Markdown)
            .unwrap();
        assert!(markdown.contains("**tool** — done"), "{markdown}");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn a_turn_without_reported_usage_records_no_usage_key() {
        let db = db();
        let session = create_session(&db, "m", ChatMode::Chat, None).unwrap();
        record_turn(
            &db,
            &session.id,
            "q",
            Timestamp::now(),
            &response("a", vec![]),
        )
        .unwrap();

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
        record_turn(
            &db,
            &first.id,
            "q",
            Timestamp::now(),
            &response("a", vec![]),
        )
        .unwrap();
        assert_eq!(latest_session(&db).unwrap().unwrap().id, first.id);
        assert_eq!(list_sessions(&db, 10).unwrap().len(), 2);
    }

    /// `latest_summary` returns the newest summary — the row with the
    /// greatest `id`, a UUID v7 sorted by creation time — not the one that
    /// covers the most messages, because `covers` is not monotonic once a
    /// budget grows and the compactor re-summarizes over a smaller prefix.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn latest_summary_returns_the_newest_summary_not_the_most_covering() {
        let db = db();
        let session = create_session(&db, "m", ChatMode::Chat, None).unwrap();
        // The older row covers more; the newer row covers less. The stored
        // `id`s are UUID v7, so they sort by creation time within one process.
        save_summary(&db, &session.id, 6, "the older broader summary").unwrap();
        save_summary(&db, &session.id, 2, "the newer narrower summary").unwrap();
        let latest = latest_summary(&db, &session.id).unwrap().unwrap();
        assert_eq!(latest.covers, 2);
        assert_eq!(latest.text, "the newer narrower summary");
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
            Timestamp::now(),
            &response(
                "first answer",
                vec![step(ToolName::RunSql, "SELECT 1", "1 rows")],
            ),
        )
        .unwrap();
        record_turn(
            &db,
            &session.id,
            "second question",
            Timestamp::now(),
            &response("second answer", vec![]),
        )
        .unwrap();

        let all = windowed(&db, &session.id, 10_000).unwrap();
        assert_eq!(all.len(), 4);

        // "second question" + "second answer" ≈ 8 tokens; budget of 9 keeps only those two.
        let trimmed = windowed(&db, &session.id, 9).unwrap();
        assert_eq!(trimmed.len(), 2);

        let none = windowed(&db, &session.id, 1).unwrap();
        assert!(none.is_empty());
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn history_does_not_start_with_an_orphaned_assistant() {
        let db = db();
        let session = create_session(&db, "m", ChatMode::Query, None).unwrap();
        record_turn(
            &db,
            &session.id,
            "first question",
            Timestamp::now(),
            &response(
                "first answer",
                vec![step(ToolName::RunSql, "SELECT 1", "1 rows")],
            ),
        )
        .unwrap();
        record_turn(
            &db,
            &session.id,
            "second question",
            Timestamp::now(),
            &response("second answer", vec![]),
        )
        .unwrap();

        // "second answer" is 13 bytes = 4 tokens, so a budget of 4 admits only
        // the newest assistant and not its preceding user. The kept suffix
        // would be an orphaned Assistant; it must be dropped, leaving an empty
        // history rather than one that opens with an answer whose question
        // was cut for budget.
        let trimmed = windowed(&db, &session.id, 4).unwrap();
        assert!(
            trimmed.is_empty(),
            "expected empty history, got {trimmed:?}"
        );
        assert!(
            !matches!(trimmed.first(), Some(Message::Assistant { .. })),
            "orphaned Assistant first: {trimmed:?}"
        );
    }

    /// A turn whose answer (or question) holds no text is left out whole:
    /// the replayed thread still alternates, and no message reaches the
    /// provider without content.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn history_skips_turns_with_no_text() {
        let db = db();
        let session = create_session(&db, "m", ChatMode::Chat, None).unwrap();
        record_turn(
            &db,
            &session.id,
            "first",
            Timestamp::now(),
            &response("one", vec![]),
        )
        .unwrap();
        append_message(&db, &session.id, MessageRole::User, "second", None).unwrap();
        append_message(&db, &session.id, MessageRole::Assistant, "  ", None).unwrap();
        append_message(&db, &session.id, MessageRole::User, "", None).unwrap();
        append_message(&db, &session.id, MessageRole::Assistant, "orphaned", None).unwrap();
        record_turn(
            &db,
            &session.id,
            "third",
            Timestamp::now(),
            &response("three", vec![]),
        )
        .unwrap();

        let history = windowed(&db, &session.id, 10_000).unwrap();
        let texts: Vec<String> = history
            .iter()
            .map(|m| match m {
                Message::User { .. } => format!("user: {m:?}"),
                Message::Assistant { .. } => format!("assistant: {m:?}"),
                Message::System { .. } => format!("system: {m:?}"),
            })
            .collect();
        assert_eq!(history.len(), 4, "{texts:#?}");
        for (text, expected) in texts.iter().zip(["first", "one", "third", "three"]) {
            assert!(text.contains(expected), "{text} lacks {expected}");
        }
        assert!(
            texts.iter().step_by(2).all(|t| t.starts_with("user:")),
            "{texts:#?}"
        );
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn history_is_always_user_anchored_and_strictly_alternating() {
        let db = db();
        let session = create_session(&db, "m", ChatMode::Query, None).unwrap();
        // Five turns with deliberately varied per-message sizes so a budget
        // sweep crosses trim boundaries of both kept-count parities.
        let turns = [
            ("aaaa", "bbbbbbbb"),
            ("ccc", "dddddddddd"),
            ("eeeeeeee", "ff"),
            ("gggggg", "hhhhhhhhhhhh"),
            ("iiiiiiiiii", "jjjj"),
        ];
        for (question, answer) in turns {
            record_turn(
                &db,
                &session.id,
                question,
                Timestamp::now(),
                &response(answer, vec![step(ToolName::RunSql, "SELECT 1", "1 rows")]),
            )
            .unwrap();
        }

        for budget in 0..=128u32 {
            let trimmed = windowed(&db, &session.id, budget).unwrap();
            assert!(
                matches!(trimmed.first(), None | Some(Message::User { .. })),
                "budget {budget}: history opens with an orphaned Assistant: {trimmed:?}"
            );
            if !trimmed.is_empty() {
                assert!(
                    matches!(trimmed.last(), Some(Message::Assistant { .. })),
                    "budget {budget}: history does not end on an Assistant: {trimmed:?}"
                );
                let mut want_user = true;
                for message in &trimmed {
                    let is_user = matches!(message, Message::User { .. });
                    assert_eq!(
                        is_user, want_user,
                        "budget {budget}: non-alternating role {message:?}"
                    );
                    want_user = !want_user;
                }
            }
        }

        assert!(windowed(&db, &session.id, 0).unwrap().is_empty());
        let full = windowed(&db, &session.id, 1_000).unwrap();
        assert_eq!(full.len(), 10);
        assert!(matches!(full.first(), Some(Message::User { .. })));
        assert!(matches!(full.last(), Some(Message::Assistant { .. })));
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
            Timestamp::now(),
            &response(
                "4",
                vec![
                    step(ToolName::ListTables, "", "2 tables"),
                    step(
                        ToolName::RunSql,
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
        record_turn(&db, &used.id, "q", Timestamp::now(), &response("a", vec![])).unwrap();
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
            Timestamp::now(),
            &response("Four.", vec![step(ToolName::RunSql, "SELECT 1", "1 rows")]),
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
