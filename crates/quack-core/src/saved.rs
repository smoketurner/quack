//! Saved questions (design doc section 8): a question a team asks
//! repeatedly, saved once with the read statements its answer ran, and
//! re-run without the model. Each run digests every statement's whole
//! result set and says whether any of them differs from the run before.
//! Nothing here schedules anything: cron runs `quack saved run`.

use crate::analysis::events::ToolName;
use crate::error::{Error, Result};
use crate::ids::{RunId, SavedId, SessionId, UserId};
use crate::storage::control::ResourceKind;
use crate::storage::sessions::{self, ChatMode, MessageRole, MessageRow, SessionRow};
use crate::storage::workspace::{DigestedResults, StatementKind, WorkspaceDb};

/// The saved questions and their runs, created with the other internal
/// tables. A run's `statements` carry each statement's SQL and digest,
/// never its rows.
pub const DDL: &str = "CREATE TABLE IF NOT EXISTS _quack_saved_questions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    question TEXT NOT NULL,
    mode TEXT NOT NULL,
    statements JSON NOT NULL,
    session_id TEXT NOT NULL,
    created_by TEXT,
    created_at TIMESTAMP DEFAULT now(),
    pinned_at TIMESTAMP DEFAULT now()
);
CREATE TABLE IF NOT EXISTS _quack_saved_runs (
    id TEXT PRIMARY KEY,
    saved_id TEXT NOT NULL,
    ran_at TIMESTAMP DEFAULT now(),
    status TEXT NOT NULL,
    changed BOOLEAN NOT NULL,
    statements JSON NOT NULL
);";

/// Runs kept per saved question, newest first, beside the newest
/// completed one.
const KEPT_RUNS: u32 = 100;

/// A question with the statements its answer ran.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, utoipa::ToSchema)]
pub struct SavedQuestion {
    pub id: SavedId,
    pub name: String,
    pub question: String,
    pub mode: ChatMode,
    /// The read statements the answer ran, in order.
    pub statements: Vec<String>,
    /// The session the statements were pinned from.
    pub session_id: SessionId,
    pub created_by: Option<UserId>,
    pub created_at: String,
    pub pinned_at: String,
}

const QUESTION_COLUMNS: &str = "id, name, question, mode, CAST(statements AS VARCHAR), \
     session_id, created_by, CAST(created_at AS VARCHAR), CAST(pinned_at AS VARCHAR)";

/// A row selected with `QUESTION_COLUMNS`.
impl TryFrom<&duckdb::Row<'_>> for SavedQuestion {
    type Error = Error;

    fn try_from(row: &duckdb::Row<'_>) -> Result<Self> {
        let mode: String = row.get(3)?;
        let statements: String = row.get(4)?;
        Ok(Self {
            id: row.get(0)?,
            name: row.get(1)?,
            question: row.get(2)?,
            mode: mode.parse()?,
            statements: serde_json::from_str(&statements)?,
            session_id: row.get(5)?,
            created_by: row.get(6)?,
            created_at: row.get(7)?,
            pinned_at: row.get(8)?,
        })
    }
}

/// Which answer of a session to pin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answer {
    /// The session's last answer.
    Last,
    /// The answer with this message sequence number.
    Seq(i64),
}

/// Why an answer cannot become a saved question.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Unsavable {
    #[error("the session has no answer yet")]
    NoAnswer,
    #[error("message {0} is not an answer")]
    NotAnAnswer(i64),
    #[error("the answer ran no SQL that returned rows; a saved question re-runs SQL")]
    NoSql,
    #[error("the answer ran a statement that writes ({0}); a saved question re-runs reads only")]
    Wrote(String),
}

/// The question and the read statements one answer ran.
struct Pinned {
    question: String,
    statements: Vec<String>,
}

impl Pinned {
    /// The turn that ends with `answer` in `session`: the question it
    /// answered, and the `run_sql` statements that returned rows, each
    /// classified again as a read.
    fn from_session(db: &WorkspaceDb, session: &SessionRow, answer: Answer) -> Result<Self> {
        let messages = sessions::messages(db, &session.id)?;
        let end = match answer {
            Answer::Last => messages
                .iter()
                .rposition(|m| m.role == MessageRole::Assistant)
                .ok_or(Unsavable::NoAnswer)?,
            Answer::Seq(seq) => {
                let at = messages
                    .iter()
                    .position(|m| m.seq == seq)
                    .ok_or_else(|| ResourceKind::Message.missing(seq.to_string()))?;
                if messages
                    .get(at)
                    .is_some_and(|m| m.role != MessageRole::Assistant)
                {
                    return Err(Unsavable::NotAnAnswer(seq).into());
                }
                at
            }
        };
        let turn: Vec<&MessageRow> = messages
            .get(..end)
            .unwrap_or_default()
            .iter()
            .rev()
            .take_while(|m| m.role != MessageRole::User)
            .collect();
        let question = messages
            .get(..end)
            .unwrap_or_default()
            .iter()
            .rfind(|m| m.role == MessageRole::User)
            .map(|m| m.content.clone())
            .ok_or(Unsavable::NoAnswer)?;
        let mut statements = Vec::new();
        for message in turn.into_iter().rev() {
            let Some(meta) = message
                .tool()
                .filter(|m| m.tool == ToolName::RunSql && m.rows.is_some())
            else {
                continue;
            };
            match db.classify_user_statement(&meta.detail)? {
                StatementKind::Read => statements.push(meta.detail.clone()),
                StatementKind::Write => return Err(Unsavable::Wrote(meta.detail.clone()).into()),
                StatementKind::Invalid(message) => {
                    return Err(Error::Analysis(format!("SQL syntax error: {message}")));
                }
            }
        }
        if statements.is_empty() {
            return Err(Unsavable::NoSql.into());
        }
        Ok(Self {
            question,
            statements,
        })
    }
}

/// Save `answer` of `session_id` under `name`: the question it answered,
/// the session's mode, and the read statements it ran.
///
/// # Errors
///
/// Returns `SavedQuestionExists` for a name in use, `Unsavable` for an
/// answer that ran no read or ran a write, `NotFound` for a missing
/// session or message, or a storage error.
pub fn save(
    db: &WorkspaceDb,
    name: &str,
    session_id: &SessionId,
    answer: Answer,
    created_by: Option<&UserId>,
) -> Result<SavedQuestion> {
    let name = name.trim();
    if name.is_empty() {
        return Err(Error::Analysis(String::from(
            "a saved question needs a name",
        )));
    }
    if by_name(db, name)?.is_some() {
        return Err(Error::SavedQuestionExists(name.to_owned()));
    }
    let session = sessions::get_session(db, session_id)?
        .ok_or_else(|| ResourceKind::Session.missing(session_id.as_str()))?;
    let pinned = Pinned::from_session(db, &session, answer)?;
    let id = SavedId::generate();
    db.connection().execute(
        "INSERT INTO _quack_saved_questions \
         (id, name, question, mode, statements, session_id, created_by) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        duckdb::params![
            id,
            name,
            pinned.question,
            session.mode.as_str(),
            serde_json::to_string(&pinned.statements)?,
            session_id,
            created_by
        ],
    )?;
    by_id(db, &id)?.ok_or_else(|| ResourceKind::SavedQuestion.missing(id.as_str()))
}

/// Pin the statements of `answer` in `session_id` to the saved question
/// `id` in place of the ones it has: a refresh asked the question again.
/// The next run compares with the run before only if the statements came
/// out the same.
///
/// # Errors
///
/// As [`save`], and `NotFound` when `id` names no saved question.
pub fn repin(
    db: &WorkspaceDb,
    id: &SavedId,
    session_id: &SessionId,
    answer: Answer,
) -> Result<SavedQuestion> {
    let session = sessions::get_session(db, session_id)?
        .ok_or_else(|| ResourceKind::Session.missing(session_id.as_str()))?;
    let pinned = Pinned::from_session(db, &session, answer)?;
    let changed = db.connection().execute(
        "UPDATE _quack_saved_questions \
         SET statements = ?, session_id = ?, pinned_at = now() WHERE id = ?",
        duckdb::params![serde_json::to_string(&pinned.statements)?, session_id, id],
    )?;
    if changed == 0 {
        return Err(ResourceKind::SavedQuestion.missing(id.as_str()));
    }
    by_id(db, id)?.ok_or_else(|| ResourceKind::SavedQuestion.missing(id.as_str()))
}

/// Every saved question, by name.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn list(db: &WorkspaceDb) -> Result<Vec<SavedQuestion>> {
    let sql = format!("SELECT {QUESTION_COLUMNS} FROM _quack_saved_questions ORDER BY name");
    let mut stmt = db.connection().prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(SavedQuestion::try_from(row)?);
    }
    Ok(out)
}

/// The saved question named `name`.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn by_name(db: &WorkspaceDb, name: &str) -> Result<Option<SavedQuestion>> {
    let sql = format!("SELECT {QUESTION_COLUMNS} FROM _quack_saved_questions WHERE name = ?");
    let mut stmt = db.connection().prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![name.trim()])?;
    rows.next()?.map(SavedQuestion::try_from).transpose()
}

/// The saved question with `id`.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn by_id(db: &WorkspaceDb, id: &SavedId) -> Result<Option<SavedQuestion>> {
    let sql = format!("SELECT {QUESTION_COLUMNS} FROM _quack_saved_questions WHERE id = ?");
    let mut stmt = db.connection().prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![id])?;
    rows.next()?.map(SavedQuestion::try_from).transpose()
}

/// Remove a saved question and its runs. Returns whether it existed.
///
/// # Errors
///
/// Returns an error if a delete fails.
pub fn remove(db: &WorkspaceDb, id: &SavedId) -> Result<bool> {
    db.write_transaction(|db| {
        db.connection().execute(
            "DELETE FROM _quack_saved_runs WHERE saved_id = ?",
            duckdb::params![id],
        )?;
        let removed = db.connection().execute(
            "DELETE FROM _quack_saved_questions WHERE id = ?",
            duckdb::params![id],
        )?;
        Ok(removed > 0)
    })
}

/// How a run ended.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, utoipa::ToSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    /// Every statement ran.
    Ok,
    /// A statement failed; the run compared nothing.
    Failed,
}

text_enum!(RunStatus, "run status", { Ok => "ok", Failed => "failed" });

/// One statement of a run.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
pub struct StatementRun {
    pub sql: String,
    /// The digest of the columns and every row in any order
    /// (`WorkspaceDb::execute_query_digested`); absent when the statement
    /// failed.
    pub digest: Option<String>,
    /// Rows the statement produced; absent when it failed.
    pub rows: Option<u64>,
    /// Whether the digest differs from the run compared with.
    pub changed: bool,
    pub columns: Vec<String>,
    /// The rows, when the result fit the row cap. Only the run that
    /// produced them carries them; a run read back from storage has none.
    pub result: Option<Vec<Vec<serde_json::Value>>>,
    pub error: Option<String>,
}

impl StatementRun {
    /// What `sql` produced, against `previous`, the same statement in the
    /// run compared with.
    fn of(sql: &str, outcome: Result<DigestedResults>, previous: Option<&Self>) -> Self {
        match outcome {
            Ok(digested) => {
                let kept = digested.results;
                let changed =
                    previous.is_some_and(|p| p.digest.as_deref() != Some(digested.digest.as_str()));
                let truncated = kept.truncated();
                Self {
                    sql: sql.to_owned(),
                    digest: Some(digested.digest),
                    rows: Some(u64::try_from(kept.total_rows).unwrap_or(u64::MAX)),
                    changed,
                    columns: kept.results.columns,
                    result: (!truncated).then_some(kept.results.rows),
                    error: None,
                }
            }
            Err(e) => Self {
                sql: sql.to_owned(),
                digest: None,
                rows: None,
                changed: false,
                columns: Vec::new(),
                result: None,
                error: Some(e.to_string()),
            },
        }
    }
}

/// One run of a saved question.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, utoipa::ToSchema)]
pub struct SavedRun {
    pub id: RunId,
    pub saved_id: SavedId,
    pub ran_at: String,
    pub status: RunStatus,
    /// Whether any statement's result differs from the run compared with:
    /// the newest completed run, when it ran the same statements. The
    /// first run is not changed, nor is a failed one, nor the first after
    /// a refresh that changed the SQL.
    pub changed: bool,
    pub statements: Vec<StatementRun>,
}

impl SavedRun {
    /// One word for how the run ended: `changed`, `unchanged`, or `failed`.
    #[must_use]
    pub fn verdict(&self) -> &'static str {
        match (self.status, self.changed) {
            (RunStatus::Failed, _) => "failed",
            (RunStatus::Ok, true) => "changed",
            (RunStatus::Ok, false) => "unchanged",
        }
    }
}

const RUN_COLUMNS: &str =
    "id, saved_id, CAST(ran_at AS VARCHAR), status, changed, CAST(statements AS VARCHAR)";

/// A row selected with `RUN_COLUMNS`.
impl TryFrom<&duckdb::Row<'_>> for SavedRun {
    type Error = Error;

    fn try_from(row: &duckdb::Row<'_>) -> Result<Self> {
        let status: String = row.get(3)?;
        let statements: String = row.get(5)?;
        Ok(Self {
            id: row.get(0)?,
            saved_id: row.get(1)?,
            ran_at: row.get(2)?,
            status: status.parse()?,
            changed: row.get(4)?,
            statements: serde_json::from_str(&statements)?,
        })
    }
}

/// The classification and execution one statement of a run gets: a read
/// again, then inside a read-only transaction with the row cap and the
/// query timeout, exactly as the agent's `run_sql` runs it.
fn execute(db: &WorkspaceDb, sql: &str, max_rows: u32) -> Result<DigestedResults> {
    match db.classify_user_statement(sql)? {
        StatementKind::Read => db.read_only(|db| db.execute_query_digested(sql, max_rows)),
        StatementKind::Write => Err(Error::Analysis(String::from(
            "the statement writes; a saved question re-runs reads only",
        ))),
        StatementKind::Invalid(message) => {
            Err(Error::Analysis(format!("SQL syntax error: {message}")))
        }
    }
}

/// Run every statement of `question` and record the run: each result's
/// digest and row count, and whether any digest differs from the newest
/// completed run, when that run ran the same statements. The returned run
/// carries each result's rows when they fit `max_rows`; the record does
/// not. A statement that fails is recorded with its error and fails the
/// run.
///
/// # Errors
///
/// Returns an error if the run cannot be recorded.
pub fn run(db: &WorkspaceDb, question: &SavedQuestion, max_rows: u32) -> Result<SavedRun> {
    let previous = last_completed(db, &question.id)?.filter(|run| {
        run.statements
            .iter()
            .map(|s| &s.sql)
            .eq(&question.statements)
    });
    let mut statements = Vec::with_capacity(question.statements.len());
    for (n, sql) in question.statements.iter().enumerate() {
        let before = previous.as_ref().and_then(|p| p.statements.get(n));
        statements.push(StatementRun::of(sql, execute(db, sql, max_rows), before));
    }
    let status = if statements.iter().any(|s| s.error.is_some()) {
        RunStatus::Failed
    } else {
        RunStatus::Ok
    };
    let changed = status == RunStatus::Ok && statements.iter().any(|s| s.changed);
    let rows: Vec<_> = statements.iter_mut().map(|s| s.result.take()).collect();
    let id = RunId::generate();
    db.connection().execute(
        "INSERT INTO _quack_saved_runs (id, saved_id, status, changed, statements) \
         VALUES (?, ?, ?, ?, ?)",
        duckdb::params![
            id,
            question.id,
            status.as_str(),
            changed,
            serde_json::to_string(&statements)?
        ],
    )?;
    // A saved question run from cron every minute would add half a million
    // runs a year; the newest are kept, and the newest completed one, which
    // the next run compares with, whatever its age.
    db.connection().execute(
        "DELETE FROM _quack_saved_runs WHERE saved_id = ? \
         AND id NOT IN (SELECT id FROM _quack_saved_runs WHERE saved_id = ? \
                        ORDER BY id DESC LIMIT ?) \
         AND id NOT IN (SELECT id FROM _quack_saved_runs WHERE saved_id = ? AND status = ? \
                        ORDER BY id DESC LIMIT 1)",
        duckdb::params![
            question.id,
            question.id,
            KEPT_RUNS,
            question.id,
            RunStatus::Ok.as_str()
        ],
    )?;
    let mut run = runs(db, &question.id, 1)?
        .into_iter()
        .next()
        .ok_or_else(|| Error::Analysis(String::from("run vanished after insert")))?;
    for (statement, rows) in run.statements.iter_mut().zip(rows) {
        statement.result = rows;
    }
    Ok(run)
}

/// The newest run that every statement of ran: what the next run
/// compares with.
fn last_completed(db: &WorkspaceDb, id: &SavedId) -> Result<Option<SavedRun>> {
    let sql = format!(
        "SELECT {RUN_COLUMNS} FROM _quack_saved_runs \
         WHERE saved_id = ? AND status = ? ORDER BY id DESC LIMIT 1"
    );
    let mut stmt = db.connection().prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![id, RunStatus::Ok.as_str()])?;
    rows.next()?.map(SavedRun::try_from).transpose()
}

/// The newest `limit` runs of a saved question, newest first.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn runs(db: &WorkspaceDb, id: &SavedId, limit: u32) -> Result<Vec<SavedRun>> {
    let sql = format!(
        "SELECT {RUN_COLUMNS} FROM _quack_saved_runs WHERE saved_id = ? ORDER BY id DESC LIMIT ?"
    );
    let mut stmt = db.connection().prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![id, i64::from(limit)])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(SavedRun::try_from(row)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::indexing_slicing,
        reason = "tests index the statements they just asserted exist"
    )]

    use super::*;
    use crate::analysis::agent::AgentResponse;
    use crate::analysis::events::ToolStep;
    use crate::embedding::Dimension;
    use jiff::Timestamp;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn db() -> WorkspaceDb {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        db.execute_statement(
            "CREATE TABLE invoices (id INTEGER, due DATE, paid BOOLEAN); \
             INSERT INTO invoices VALUES (1, '2026-09-01', false), (2, '2026-09-15', true)",
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
        db
    }

    fn step(sql: &str, rows: Option<u64>) -> ToolStep {
        ToolStep {
            tool: ToolName::RunSql,
            detail: sql.to_owned(),
            summary: rows.map_or_else(|| String::from("refused"), |n| format!("{n} rows")),
            rows,
            result: None,
            duration_ms: 1,
        }
    }

    /// A session with one turn whose answer ran `steps`.
    fn session_with(db: &WorkspaceDb, question: &str, steps: Vec<ToolStep>) -> SessionId {
        let session = sessions::create_session(db, "m", ChatMode::Query, None)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let response = AgentResponse {
            content: String::from("One invoice is overdue."),
            steps,
            ..AgentResponse::default()
        };
        sessions::record_turn(db, &session.id, question, Timestamp::now(), &response)
            .unwrap_or_else(|e| fail(&e.to_string()));
        session.id
    }

    const OVERDUE: &str = "SELECT id FROM invoices WHERE NOT paid ORDER BY id";

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn save_pins_the_question_mode_and_read_statements_of_the_last_answer() {
        let db = db();
        let session = session_with(
            &db,
            "which invoices are overdue?",
            vec![
                step("SELECT count(*) FROM invoices", Some(1)),
                step("SELECT * FROM nowhere", None),
                step(OVERDUE, Some(1)),
            ],
        );
        let saved = save(&db, " overdue ", &session, Answer::Last, None).unwrap();
        assert_eq!(saved.name, "overdue");
        assert_eq!(saved.question, "which invoices are overdue?");
        assert_eq!(saved.mode, ChatMode::Query);
        assert_eq!(saved.statements, ["SELECT count(*) FROM invoices", OVERDUE]);
        assert_eq!(by_name(&db, "overdue").unwrap(), Some(saved.clone()));
        assert_eq!(by_id(&db, &saved.id).unwrap(), Some(saved.clone()));
        assert_eq!(list(&db).unwrap(), std::slice::from_ref(&saved));

        let again = save(&db, "overdue", &session, Answer::Last, None).unwrap_err();
        assert_eq!(again.to_string(), "saved question 'overdue' already exists");
        let unnamed = save(&db, "  ", &session, Answer::Last, None).unwrap_err();
        assert!(unnamed.to_string().contains("needs a name"), "{unnamed}");
        let missing = save(&db, "x", &SessionId::from("nope"), Answer::Last, None).unwrap_err();
        assert_eq!(missing.to_string(), "session 'nope' does not exist");

        // The answer is message 5 (the question, three tool rows, the
        // answer): by sequence number, or not an answer.
        let by_seq = save(&db, "by-seq", &session, Answer::Seq(5), None).unwrap();
        assert_eq!(by_seq.statements, saved.statements);
        let not_answer = save(&db, "q", &session, Answer::Seq(1), None).unwrap_err();
        assert_eq!(
            not_answer.to_string(),
            "cannot save this answer: message 1 is not an answer"
        );
        let no_message = save(&db, "q", &session, Answer::Seq(9), None).unwrap_err();
        assert_eq!(no_message.to_string(), "message '9' does not exist");

        assert!(remove(&db, &saved.id).unwrap());
        assert!(!remove(&db, &saved.id).unwrap());
        assert_eq!(list(&db).unwrap().len(), 1);
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn an_answer_without_a_read_or_with_a_write_cannot_be_saved() {
        let db = db();
        let no_sql = session_with(&db, "hello", vec![]);
        let err = save(&db, "a", &no_sql, Answer::Last, None).unwrap_err();
        assert!(matches!(err, Error::Unsavable(Unsavable::NoSql)), "{err}");

        let wrote = session_with(
            &db,
            "mark 1 paid",
            vec![
                step(OVERDUE, Some(1)),
                step("UPDATE invoices SET paid = true WHERE id = 1", Some(1)),
            ],
        );
        let err = save(&db, "b", &wrote, Answer::Last, None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "cannot save this answer: the answer ran a statement that writes \
             (UPDATE invoices SET paid = true WHERE id = 1); a saved question re-runs reads only"
        );

        let empty = sessions::create_session(&db, "m", ChatMode::Chat, None).unwrap();
        let err = save(&db, "c", &empty.id, Answer::Last, None).unwrap_err();
        assert!(
            matches!(err, Error::Unsavable(Unsavable::NoAnswer)),
            "{err}"
        );
        assert!(list(&db).unwrap().is_empty());
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn runs_digest_each_result_and_report_a_change_between_runs() {
        let db = db();
        let session = session_with(&db, "overdue?", vec![step(OVERDUE, Some(1))]);
        let saved = save(&db, "overdue", &session, Answer::Last, None).unwrap();

        let first = run(&db, &saved, 100).unwrap();
        assert_eq!(first.status, RunStatus::Ok);
        assert!(!first.changed, "the first run compares with nothing");
        let statement = &first.statements[0];
        assert_eq!(statement.rows, Some(1));
        assert_eq!(statement.columns, ["id"]);
        assert_eq!(statement.result, Some(vec![vec![serde_json::json!(1)]]));
        assert_eq!(statement.digest.as_ref().map(String::len), Some(64));

        let second = run(&db, &saved, 100).unwrap();
        assert!(!second.changed);
        assert_eq!(second.statements[0].digest, statement.digest);

        db.execute_statement("INSERT INTO invoices VALUES (3, '2026-10-01', false)")
            .unwrap();
        let third = run(&db, &saved, 100).unwrap();
        assert!(third.changed);
        assert_eq!(third.verdict(), "changed");
        assert_eq!(third.statements[0].rows, Some(2));
        assert_ne!(third.statements[0].digest, statement.digest);

        let fourth = run(&db, &saved, 100).unwrap();
        assert!(!fourth.changed, "nothing moved since the third run");

        // A result past the cap is digested whole but its rows are not kept.
        let capped = run(&db, &saved, 1).unwrap();
        assert_eq!(capped.statements[0].rows, Some(2));
        assert_eq!(capped.statements[0].result, None);
        assert_eq!(capped.statements[0].digest, fourth.statements[0].digest);

        let history = runs(&db, &saved.id, 10).unwrap();
        assert_eq!(history.len(), 5);
        assert_eq!(history[0].id, capped.id, "newest first");
        assert!(
            history.iter().all(|r| r.statements[0].result.is_none()),
            "rows are returned once, never stored"
        );
        assert_eq!(history[1].statements[0].digest, fourth.statements[0].digest);
        assert!(remove(&db, &saved.id).unwrap());
        assert!(runs(&db, &saved.id, 10).unwrap().is_empty());
    }

    /// The run history is bounded: the newest runs are kept, and the
    /// newest completed one even when a long streak of failures follows it.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn old_runs_are_dropped_but_the_last_completed_one_is_kept() {
        let db = db();
        let session = session_with(&db, "overdue?", vec![step(OVERDUE, Some(1))]);
        let saved = save(&db, "overdue", &session, Answer::Last, None).unwrap();
        for _ in 0..105 {
            run(&db, &saved, 100).unwrap();
        }
        assert_eq!(runs(&db, &saved.id, 1_000).unwrap().len(), 100);

        let completed = run(&db, &saved, 100).unwrap();
        db.execute_statement("DROP TABLE invoices").unwrap();
        for _ in 0..100 {
            assert_eq!(run(&db, &saved, 100).unwrap().status, RunStatus::Failed);
        }
        let kept = runs(&db, &saved.id, 1_000).unwrap();
        assert_eq!(kept.len(), 101, "100 newest, and the last completed");
        assert_eq!(
            last_completed(&db, &saved.id).unwrap().map(|r| r.id),
            Some(completed.id)
        );
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn a_run_whose_table_is_gone_fails_and_is_recorded_as_failed() {
        let db = db();
        let session = session_with(&db, "overdue?", vec![step(OVERDUE, Some(1))]);
        let saved = save(&db, "overdue", &session, Answer::Last, None).unwrap();
        run(&db, &saved, 100).unwrap();
        db.execute_statement("DROP TABLE invoices").unwrap();

        let failed = run(&db, &saved, 100).unwrap();
        assert_eq!(failed.status, RunStatus::Failed);
        assert_eq!(failed.verdict(), "failed");
        assert!(!failed.changed);
        let statement = &failed.statements[0];
        assert!(
            statement
                .error
                .as_deref()
                .is_some_and(|e| e.contains("invoices")),
            "{statement:?}"
        );
        assert_eq!(statement.rows, None);
        assert_eq!(
            runs(&db, &saved.id, 10).unwrap()[0].status,
            RunStatus::Failed
        );

        // Back, the next run compares with the last completed one.
        db.execute_statement(
            "CREATE TABLE invoices (id INTEGER, due DATE, paid BOOLEAN); \
             INSERT INTO invoices VALUES (1, '2026-09-01', false)",
        )
        .unwrap();
        let back = run(&db, &saved, 100).unwrap();
        assert_eq!(back.status, RunStatus::Ok);
        assert!(!back.changed);
    }

    /// A pinned statement is classified again when it runs, so a write or
    /// an internal-table read planted in storage fails the run instead of
    /// running.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn a_pinned_statement_is_classified_again_before_it_runs() {
        let db = db();
        let session = session_with(&db, "overdue?", vec![step(OVERDUE, Some(1))]);
        let saved = save(&db, "overdue", &session, Answer::Last, None).unwrap();
        db.connection()
            .execute(
                "UPDATE _quack_saved_questions SET statements = ? WHERE id = ?",
                duckdb::params![
                    serde_json::to_string(&["DELETE FROM invoices", "SELECT * FROM _quack_meta"])
                        .unwrap(),
                    saved.id
                ],
            )
            .unwrap();
        let planted = by_id(&db, &saved.id).unwrap().unwrap();
        let failed = run(&db, &planted, 100).unwrap();
        assert_eq!(failed.status, RunStatus::Failed);
        assert!(
            failed.statements[0]
                .error
                .as_deref()
                .is_some_and(|e| e.contains("re-runs reads only")),
            "{failed:?}"
        );
        assert!(
            failed.statements[1]
                .error
                .as_deref()
                .is_some_and(|e| e.contains("internal tables")),
            "{failed:?}"
        );
        let count: i64 = db
            .connection()
            .query_row("SELECT count(*) FROM invoices", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 2, "the planted write did not run");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn repin_replaces_the_statements_and_compares_only_with_a_run_of_the_same_sql() {
        let db = db();
        let session = session_with(&db, "overdue?", vec![step(OVERDUE, Some(1))]);
        let saved = save(&db, "overdue", &session, Answer::Last, None).unwrap();
        run(&db, &saved, 100).unwrap();

        // The same SQL pinned again: the next run still compares with the
        // run before it.
        let same = session_with(&db, "overdue?", vec![step(OVERDUE, Some(1))]);
        let repinned = repin(&db, &saved.id, &same, Answer::Last).unwrap();
        assert_eq!(repinned.session_id, same);
        assert_eq!(repinned.statements, saved.statements);
        db.execute_statement("INSERT INTO invoices VALUES (3, '2026-10-01', false)")
            .unwrap();
        assert!(run(&db, &repinned, 100).unwrap().changed);

        let other = session_with(
            &db,
            "overdue?",
            vec![step("SELECT id, due FROM invoices WHERE NOT paid", Some(2))],
        );
        let repinned = repin(&db, &saved.id, &other, Answer::Last).unwrap();
        assert_eq!(repinned.session_id, other);
        assert_eq!(
            repinned.statements,
            ["SELECT id, due FROM invoices WHERE NOT paid"]
        );
        assert_eq!(repinned.question, saved.question);
        let first_of_new_sql = run(&db, &repinned, 100).unwrap();
        assert!(
            !first_of_new_sql.changed,
            "the run before ran other SQL, so there is nothing to compare with"
        );
        db.execute_statement("INSERT INTO invoices VALUES (4, '2026-10-02', false)")
            .unwrap();
        assert!(
            run(&db, &repinned, 100).unwrap().changed,
            "the second run of the new SQL compares with the first"
        );

        let missing = repin(&db, &SavedId::from("nope"), &other, Answer::Last).unwrap_err();
        assert_eq!(missing.to_string(), "saved question 'nope' does not exist");
        let no_sql = session_with(&db, "hi", vec![]);
        assert!(repin(&db, &saved.id, &no_sql, Answer::Last).is_err());
        assert_eq!(
            by_id(&db, &saved.id).unwrap().unwrap().session_id,
            other,
            "a refused repin changes nothing"
        );
        let gone = repin(&db, &saved.id, &SessionId::from("nope"), Answer::Last).unwrap_err();
        assert_eq!(gone.to_string(), "session 'nope' does not exist");
    }
}
