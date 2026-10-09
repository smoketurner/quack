use super::*;
use crate::embedding::Dimension;
use crate::ids::DocumentId;
use crate::storage::workspace::{DocumentStatus, NewDocument};

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
    WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| open_failed(&e.to_string()))
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
        documents: DocumentScope::default(),
    }
}

fn step(tool: ToolName, detail: &str, summary: &str) -> ToolStep {
    ToolStep {
        tool,
        detail: detail.to_owned(),
        summary: summary.to_owned(),
        rows: None,
        result: None,
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
    assert_eq!(mine.sharing, Sharing::Private);

    // Sharing opens the session to other members; unsharing closes it.
    set_session_sharing(&db, &theirs.id, Sharing::Shared).unwrap_or_else(|e| fail(&e.to_string()));
    let theirs = get_session(&db, &theirs.id)
        .ok()
        .flatten()
        .unwrap_or_else(|| fail("session vanished"));
    assert_eq!(theirs.sharing, Sharing::Shared);
    assert!(theirs.visible_to(&SessionViewer::User(UserId::from("u1"))));
    assert!(
        list_sessions_for(&db, 10, &SessionViewer::User(UserId::from("u1")))
            .is_ok_and(|v| v.len() == 3)
    );
    set_session_sharing(&db, &theirs.id, Sharing::Private).unwrap_or_else(|e| fail(&e.to_string()));
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
    let missing = |err: Option<Error>| {
        err.is_some_and(|e| {
            matches!(
                e,
                Error::NotFound {
                    kind: ResourceKind::Session,
                    ..
                }
            )
        })
    };
    let nope = SessionId::from("nope");
    assert!(missing(
        append_message(&db, &nope, MessageRole::User, "x", None).err()
    ));
    let answer = AgentResponse::default();
    assert!(missing(
        record_turn(&db, &nope, "x", Timestamp::now(), &answer).err()
    ));
    assert!(messages(&db, &nope).is_ok_and(|m| m.is_empty()));
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

#[test]
#[expect(clippy::unwrap_used, reason = "test asserts Ok")]
fn the_documents_a_question_was_limited_to_are_kept_on_its_message() {
    let db = db();
    db.insert_document(
        &NewDocument::new(&DocumentId::from("d1"), "policy.md", "text/markdown", 1)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();
    let scope = DocumentScope::resolve(&db, &[String::from("policy.md")]).unwrap();
    let session = create_session(&db, "m", ChatMode::Chat, None).unwrap();
    let mut answer = response("Renewals are yearly.", vec![]);
    answer.documents = scope.clone();
    record_turn(&db, &session.id, "renewals?", Timestamp::now(), &answer).unwrap();
    record_turn(
        &db,
        &session.id,
        "and claims?",
        Timestamp::now(),
        &response("Claims close in 30 days.", vec![]),
    )
    .unwrap();
    let rows = messages(&db, &session.id).unwrap();
    let asked: Vec<Option<&MessageMeta>> = rows
        .iter()
        .filter(|r| r.role == MessageRole::User)
        .map(|r| r.metadata.as_ref())
        .collect();
    assert_eq!(
        asked,
        [
            Some(&MessageMeta::User(UserMeta { documents: scope })),
            None
        ],
        "an unscoped question stores no metadata"
    );
}

/// A person's title sticks: the model's does not replace it, and a blank
/// rename gives back the title the first question derives.
#[test]
fn a_person_renames_a_session_and_the_model_never_overrides_it() {
    let db = db();
    let session =
        create_session(&db, "p/m", ChatMode::Chat, None).unwrap_or_else(|e| fail(&e.to_string()));
    record_turn(
        &db,
        &session.id,
        "  which   vendors were late in March?  ",
        Timestamp::now(),
        &response("Two vendors.", vec![]),
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    let derived = get_session(&db, &session.id)
        .ok()
        .flatten()
        .unwrap_or_else(|| fail("no session"));
    assert_eq!(
        derived.title.as_deref(),
        Some("which vendors were late in March?")
    );
    assert_eq!(derived.title_by, TitleSource::Derived);

    assert!(set_model_title(&db, &session.id, "Late vendors, March").unwrap_or(false));
    let renamed = set_session_title(&db, &session.id, "  Vendor   delays ")
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(renamed.title.as_deref(), Some("Vendor delays"));
    assert_eq!(renamed.title_by, TitleSource::Person);
    assert!(!set_model_title(&db, &session.id, "Something else").unwrap_or(true));
    assert_eq!(
        get_session(&db, &session.id)
            .ok()
            .flatten()
            .and_then(|s| s.title),
        Some(String::from("Vendor delays"))
    );

    let restored =
        set_session_title(&db, &session.id, "   ").unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        restored.title.as_deref(),
        Some("which vendors were late in March?")
    );
    assert_eq!(restored.title_by, TitleSource::Derived);
    assert!(set_session_title(&db, &SessionId::from("missing"), "x").is_err());
}

/// Search matches questions and answers case-insensitively, takes `%` and
/// `_` as themselves, skips tool rows, and never shows another member's
/// private session, so its count gives nothing away either.
#[test]
fn message_search_matches_text_and_keeps_to_visible_sessions() {
    let db = db();
    let ada = UserId::from("ada");
    let bob = UserId::from("bob");
    let turn = |owner: &UserId, question: &str, answer: &str| {
        let session = create_session(&db, "p/m", ChatMode::Chat, Some(owner))
            .unwrap_or_else(|e| fail(&e.to_string()));
        record_turn(
            &db,
            &session.id,
            question,
            Timestamp::now(),
            &response(
                answer,
                vec![step(ToolName::RunSql, "SELECT 'Freight'", "1 rows")],
            ),
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
        session
    };
    let mine = turn(
        &ada,
        "How much did FREIGHT cost?",
        "Freight was 12% of spend.",
    );
    let private = turn(&bob, "freight for bob only", "Bob's freight is private.");
    let shared = turn(
        &bob,
        "freight shared with the team",
        "Shared freight answer.",
    );
    set_session_sharing(&db, &shared.id, Sharing::Shared).unwrap_or_else(|e| fail(&e.to_string()));

    let for_ada = search_messages(&db, "freight", &SessionViewer::User(ada), 50)
        .unwrap_or_else(|e| fail(&e.to_string()));
    let sessions: Vec<&SessionId> = for_ada.iter().map(|h| &h.session_id).collect();
    assert!(sessions.contains(&&mine.id) && sessions.contains(&&shared.id));
    assert!(!sessions.contains(&&private.id), "{for_ada:?}");
    assert_eq!(
        for_ada.len(),
        4,
        "a question and an answer in each visible session"
    );
    assert!(for_ada.iter().all(|h| h.role != MessageRole::Tool));
    let first = for_ada
        .iter()
        .find(|h| h.session_id == mine.id && h.role == MessageRole::User)
        .unwrap_or_else(|| fail("no hit"));
    assert_eq!(first.seq, 1);
    assert!(first.snippet.contains("FREIGHT"), "{first:?}");
    assert_eq!(
        first.session_title.as_deref(),
        Some("How much did FREIGHT cost?")
    );

    let everyone = search_messages(&db, "freight", &SessionViewer::All, 50)
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(everyone.len(), 6);
    let percent = search_messages(&db, "12%", &SessionViewer::All, 50)
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(percent.len(), 1, "% is matched as itself");
    assert!(
        search_messages(&db, "1_%", &SessionViewer::All, 50)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .is_empty(),
        "_ is matched as itself"
    );
    assert!(
        search_messages(&db, "  ", &SessionViewer::All, 50)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .is_empty()
    );
    assert_eq!(
        search_messages(&db, "freight", &SessionViewer::All, 2)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .len(),
        2
    );
}
