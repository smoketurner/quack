use quack_core::storage::writer::Writer;

use quack_core::analysis::agent::AgentResponse;
use quack_core::analysis::chart::ChartSpec;
use quack_core::analysis::events::ToolStep;
use quack_core::analysis::policy::Hold;
use quack_core::ingestion::parser::SectionKind;

use super::*;
use crate::terminal::commands::Suggestion;
use crate::terminal::selection::{Position, Row};
use quack_core::ids::{ChunkId, DocumentId};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::Widget;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

/// A turn's end and its stream's close arrive in either order, and only
/// both together finish it.
#[test]
fn a_turn_is_done_once_it_has_ended_and_closed() {
    let start = TurnProgress::Streaming;
    assert_eq!(start.ended(), TurnProgress::Ended);
    assert_eq!(start.closed(), TurnProgress::Closed);
    assert_eq!(start.ended().closed(), TurnProgress::Done);
    assert_eq!(start.closed().ended(), TurnProgress::Done);
    assert_eq!(start.ended().ended(), TurnProgress::Ended);
}

/// An app over a real workspace file (the background jobs open it
/// again by id), driven without a terminal.
fn app(dir: &Path) -> App {
    app_with(dir, Config::default())
}

/// A file as a terminal pastes it when it is dropped: spaces escaped on
/// Unix, the path in double quotes on Windows.
fn dropped_path(file: &Path) -> String {
    let path = file.display().to_string();
    if cfg!(windows) {
        format!("\"{path}\"")
    } else {
        path.replace(' ', "\\ ")
    }
}

/// `app` under `config`, its data directory moved to `dir`.
fn app_with(dir: &Path, config: Config) -> App {
    app_in(dir, "ws", config)
}

/// `app_with` over workspace `workspace` of the data directory `dir`.
fn app_in(dir: &Path, workspace: &str, mut config: Config) -> App {
    config.general.data_dir = dir.to_path_buf();
    let db = WorkspaceDb::open(&config, workspace).unwrap_or_else(|e| fail(&e.to_string()));
    let session = sessions::create_session(&db, "m", ChatMode::Chat, None)
        .unwrap_or_else(|e| fail(&e.to_string()));
    let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
    let reader_db = ReaderDb::new(Arc::clone(&db));
    App::new(SessionSetup {
        config,
        workspace_name: String::from(workspace),
        workspace_id: WorkspaceId::from(workspace),
        db,
        reader_db,
        session_id: session.id,
        writes: WritePolicy::Ask,
    })
}

/// Wait for the background result a command posted and apply it.
async fn settle(app: &mut App) {
    for _ in 0..400 {
        while let Ok(msg) = app.msg_rx.try_recv() {
            let finished = matches!(msg, AppMsg::Finished(..));
            app.handle_msg(msg);
            if finished {
                app.pump();
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    fail("no background result arrived");
}

/// Wait until no job is queued or running, then take in what they sent:
/// each job posts its result before it ends.
async fn settle_jobs(app: &mut App) {
    pump_until(app, |app| app.jobs.counts(None).active() == 0).await;
    app.pump();
}

/// Wait until every database step sent so far has been applied.
async fn db_settle(app: &mut App) {
    pump_until(app, |app| app.pending_db == 0).await;
}

/// Pump until `done` holds.
async fn pump_until(app: &mut App, done: impl Fn(&App) -> bool) {
    for _ in 0..400 {
        app.pump();
        if done(app) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    fail("the condition never held");
}

/// A turn for a job that waits until it is cancelled.
fn waiting_turn(app: &App) -> Turn {
    let job = app.jobs.submit(
        JobSpec::new(JobKind::Chat, "question")
            .lane(Lane::serial(&LaneKey::Session(SessionId::from("test")))),
        |ctx| async move {
            ctx.cancel_token().cancelled().await;
            Err(String::from("cancelled"))
        },
    );
    Turn {
        job: Ticket::from(&job),
        session_id: app.session_id.clone(),
        streaming: None,
        open_step: None,
        progress: TurnProgress::Streaming,
        phase: Phase::Waiting { since: None },
    }
}

fn last(app: &App) -> &Message {
    app.messages.last().unwrap_or_else(|| fail("no messages"))
}

#[tokio::test(flavor = "multi_thread")]
async fn slash_commands_run_sql_schema_and_cli_verbs_without_a_terminal() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());

    // A write asks first; `y` runs it as a job.
    app.handle_slash_command("/sql CREATE TABLE t AS SELECT 1 AS a, 'x' AS b");
    db_settle(&mut app).await;
    assert!(app.awaiting_permission());
    assert!(matches!(app.prompts.front(), Some(Prompt::Sql(_))));
    app.handle_key_event(KeyCode::Char('y'), KeyModifiers::NONE);
    assert!(!app.awaiting_permission());
    settle(&mut app).await;
    assert_eq!(last(&app).kind, MessageKind::Sql);

    app.handle_slash_command("/tables");
    db_settle(&mut app).await;
    assert!(last(&app).content.contains('t'), "{}", last(&app).content);
    app.handle_slash_command("/schema t");
    db_settle(&mut app).await;
    assert_eq!(last(&app).kind, MessageKind::Sql);
    assert!(
        last(&app).content.contains("a INTEGER"),
        "{}",
        last(&app).content
    );
    app.handle_slash_command("/schema nope");
    db_settle(&mut app).await;
    assert_eq!(last(&app).kind, MessageKind::Error);

    // Internal tables stay refused, a read runs without asking.
    app.handle_slash_command("/sql SELECT * FROM _quack_documents");
    db_settle(&mut app).await;
    assert_eq!(last(&app).kind, MessageKind::Error);
    app.handle_slash_command("/sql SELECT a FROM t");
    settle(&mut app).await;
    assert!(last(&app).content.contains('1'), "{}", last(&app).content);

    // The CLI verbs: clap parses them, background jobs answer.
    app.handle_slash_command("/ontology --help");
    assert!(
        last(&app).content.contains("Usage"),
        "{}",
        last(&app).content
    );
    app.handle_slash_command("/graph status");
    assert!(
        last(&app).content.contains("(job #"),
        "{}",
        last(&app).content
    );
    settle(&mut app).await;
    assert!(
        last(&app).content.contains("Graph: 0 nodes"),
        "{}",
        last(&app).content
    );
    app.handle_slash_command("/ontology init");
    settle(&mut app).await;
    app.handle_slash_command("/ontology show");
    settle(&mut app).await;
    assert!(
        last(&app).content.contains("entity"),
        "{}",
        last(&app).content
    );

    app.handle_slash_command("/export --markdown");
    db_settle(&mut app).await;
    assert_eq!(last(&app).kind, MessageKind::Sql);
    app.handle_slash_command("/share");
    db_settle(&mut app).await;
    assert!(last(&app).content.contains("shared"));
    app.handle_slash_command("/model");
    let configured = app.messages.iter().rev().nth(1).map(|m| m.content.as_str());
    assert!(
        configured.is_some_and(|text| text.contains("keyword search only")),
        "{configured:?}"
    );
    assert!(
        last(&app)
            .content
            .contains("Listing each provider's models")
    );
    settle(&mut app).await;
    assert_eq!(last(&app).content, "No providers are configured.");
    app.handle_slash_command("/nope");
    assert!(last(&app).content.contains("unknown command"));

    // Every job so far is on record.
    app.handle_slash_command("/jobs");
    let listing = screen(&app).join("\n");
    assert!(listing.contains(" Jobs "), "{listing}");
    assert!(listing.contains("succeeded sql"), "{listing}");
    assert!(listing.contains("graph"), "{listing}");
    app.handle_key_event(KeyCode::Esc, KeyModifiers::NONE);
    assert!(app.picker.is_none());
    app.handle_slash_command("/cancel 1");
    assert!(last(&app).content.contains("already succeeded"));
    app.handle_slash_command("/cancel x");
    assert_eq!(last(&app).kind, MessageKind::Error);
    assert!(
        last(&app).content.contains("invalid value"),
        "{}",
        last(&app).content
    );
    app.handle_slash_command("/cancel 99");
    assert!(last(&app).content.contains("no job #99"));
}

#[tokio::test(flavor = "multi_thread")]
async fn agent_events_attach_charts_and_steps_and_keys_cancel_the_turn() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let mut turn = waiting_turn(&app);
    app.handle_turn_event(
        &mut turn,
        AgentEvent::ToolStarted {
            tool: ToolName::RunSql,
            detail: (1..=6)
                .map(|i| format!("line {i}"))
                .collect::<Vec<_>>()
                .join("\n"),
        },
    );
    app.handle_turn_event(
        &mut turn,
        AgentEvent::ToolFinished(ToolStep {
            tool: ToolName::RunSql,
            detail: String::new(),
            summary: String::from("3 rows"),
            rows: Some(3),
            result: None,
            duration_ms: 4,
        }),
    );
    app.handle_turn_event(
        &mut turn,
        AgentEvent::TextDelta(String::from("**Three** rows")),
    );
    let spec: ChartSpec = serde_json::from_value(serde_json::json!({
        "title": "Rows by kind",
        "kind": "bar",
        "x": { "label": "kind", "values": ["a", "b"] },
        "series": [{ "name": "n", "values": [1.0, 2.0] }]
    }))
    .unwrap_or_else(|e| fail(&e.to_string()));
    app.handle_turn_event(
        &mut turn,
        AgentEvent::TurnComplete(AgentResponse {
            content: String::from("**Three** rows"),
            chart: Some(spec),
            ..AgentResponse::default()
        }),
    );
    assert!(turn.streaming.is_none());
    let assistant = app
        .messages
        .iter()
        .rev()
        .find(|m| m.kind == MessageKind::Assistant)
        .unwrap_or_else(|| fail("no assistant message"));
    assert!(assistant.chart.is_some(), "the chart belongs to the answer");
    let drawn: Vec<String> = ui::format_messages(&app, 80)
        .iter()
        .map(|row| row.line.to_string())
        .collect();
    let at = |text: &str| drawn.iter().position(|l| l.contains(text));
    let (answer, chart) = (at("Three rows"), at(" Rows by kind "));
    assert!(answer.is_some() && answer < chart, "{drawn:?}");
    let step = app
        .messages
        .iter()
        .find(|m| m.kind == MessageKind::Step)
        .unwrap_or_else(|| fail("no step"));
    assert!(
        step.detail
            .as_deref()
            .is_some_and(|d| d.lines().count() == 6)
    );

    // Rendering folds the detail to a preview until /steps; the
    // Markdown bold survives as a span; lines wrap to the width.
    let lines = ui::format_messages(&app, 20);
    let text: Vec<String> = lines
        .iter()
        .map(|l| l.line.spans.iter().map(|s| s.content.to_string()).collect())
        .collect();
    assert!(text.iter().any(|l| l.contains("3 more lines")), "{text:?}");
    assert!(text.iter().all(|l| l.chars().count() <= 20), "{text:?}");
    app.handle_slash_command("/steps");
    let expanded = ui::format_messages(&app, 80);
    assert!(
        expanded
            .iter()
            .any(|l| l.line.spans.iter().any(|s| s.content.contains("line 6")))
    );

    // Esc while a turn runs cancels it; typing goes on meanwhile.
    let job = turn.job.id;
    app.turns.push(turn);
    app.handle_key_event(KeyCode::Char('h'), KeyModifiers::NONE);
    assert_eq!(app.textarea.lines().join(""), "h");
    app.handle_key_event(KeyCode::Esc, KeyModifiers::NONE);
    assert!(last(&app).content.contains("Cancelling job #1"));
    let finished = tokio::time::timeout(Duration::from_secs(5), app.jobs.wait(job))
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| fail("the turn did not end"));
    assert_eq!(finished.state, JobState::Cancelled);
    app.turns.clear();
    app.handle_key_event(KeyCode::Char('c'), KeyModifiers::CONTROL);
    assert_eq!(app.quit, Quit::Now, "Ctrl+C with nothing running quits");
}

/// Text streamed before a tool call, then more text after it, must end as
/// one assistant message holding the validated full answer: the
/// pre-tool preamble is not left behind as an orphaned partial.
#[tokio::test(flavor = "multi_thread")]
async fn a_turn_with_text_then_a_tool_then_text_keeps_one_assistant_message() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let mut turn = waiting_turn(&app);

    // The common preamble before the first tool call.
    app.handle_turn_event(
        &mut turn,
        AgentEvent::TextDelta(String::from("Let me look up the data.")),
    );
    app.handle_turn_event(
        &mut turn,
        AgentEvent::ToolStarted {
            tool: ToolName::RunSql,
            detail: String::from("SELECT 1"),
        },
    );
    app.handle_turn_event(
        &mut turn,
        AgentEvent::ToolFinished(ToolStep {
            tool: ToolName::RunSql,
            detail: String::new(),
            summary: String::from("1 rows"),
            rows: Some(1),
            result: None,
            duration_ms: 1,
        }),
    );
    // More text streams after the tool returns.
    app.handle_turn_event(
        &mut turn,
        AgentEvent::TextDelta(String::from(" The result is 1.")),
    );
    // The agent core returns the whole turn's accumulated, validated answer.
    app.handle_turn_event(
        &mut turn,
        AgentEvent::TurnComplete(AgentResponse {
            content: String::from("Let me look up the data. The result is 1."),
            ..AgentResponse::default()
        }),
    );

    let assistants = app
        .messages
        .iter()
        .filter(|m| m.kind == MessageKind::Assistant)
        .count();
    assert_eq!(
        assistants, 1,
        "one assistant message per turn, but the transcript was {:?}",
        app.messages
    );
    let assistant = app
        .messages
        .iter()
        .find(|m| m.kind == MessageKind::Assistant)
        .unwrap_or_else(|| fail("no assistant message"));
    assert_eq!(
        assistant.content,
        "Let me look up the data. The result is 1."
    );
    // The tool call is still its own neighboring step message.
    assert_eq!(
        app.messages
            .iter()
            .filter(|m| m.kind == MessageKind::Step)
            .count(),
        1
    );
    assert!(turn.streaming.is_none(), "the turn stopped streaming");
    app.turns.clear();
}

/// A write tool pauses on `PermissionRequired` between its `ToolStarted`
/// and `ToolFinished`; the streaming target must survive both the tool
/// start and the permission prompt so the turn still ends with one
/// assistant message.
#[tokio::test(flavor = "multi_thread")]
async fn a_turn_with_text_then_a_write_permission_then_text_keeps_one_assistant_message() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let mut turn = waiting_turn(&app);

    app.handle_turn_event(
        &mut turn,
        AgentEvent::TextDelta(String::from("Let me update the table.")),
    );
    app.handle_turn_event(
        &mut turn,
        AgentEvent::ToolStarted {
            tool: ToolName::RunSql,
            detail: String::from("DELETE FROM t"),
        },
    );
    // A real `PermissionRequired` from a `TurnRecorder` so its answer
    // oneshot is live and the terminal's `y` resolves it.
    let (sink, mut rx) = events::channel();
    let recorder = events::TurnRecorder::new(sink);
    let pending = tokio::spawn(async move {
        recorder
            .ask_permission("DELETE FROM t", Hold::NotPermitted)
            .await
    });
    let request = match rx
        .recv()
        .await
        .unwrap_or_else(|| fail("no permission event"))
    {
        AgentEvent::PermissionRequired(request) => request,
        other => fail(&format!("expected PermissionRequired, got {other:?}")),
    };
    app.handle_turn_event(&mut turn, AgentEvent::PermissionRequired(request));
    assert!(app.awaiting_permission(), "the write prompt is on screen");
    app.handle_permission_key(KeyCode::Char('y'));
    assert!(!app.awaiting_permission());
    let allowed = pending.await.unwrap_or_else(|e| fail(&e.to_string()));
    assert!(allowed, "the recorder saw the user's `yes`");

    app.handle_turn_event(
        &mut turn,
        AgentEvent::ToolFinished(ToolStep {
            tool: ToolName::RunSql,
            detail: String::new(),
            summary: String::from("1 rows"),
            rows: Some(1),
            result: None,
            duration_ms: 1,
        }),
    );
    app.handle_turn_event(&mut turn, AgentEvent::TextDelta(String::from(" Done.")));
    app.handle_turn_event(
        &mut turn,
        AgentEvent::TurnComplete(AgentResponse {
            content: String::from("Let me update the table. Done."),
            ..AgentResponse::default()
        }),
    );

    let assistants = app
        .messages
        .iter()
        .filter(|m| m.kind == MessageKind::Assistant)
        .count();
    assert_eq!(
        assistants, 1,
        "one assistant message after a write tool, but the transcript was {:?}",
        app.messages
    );
    let assistant = app
        .messages
        .iter()
        .find(|m| m.kind == MessageKind::Assistant)
        .unwrap_or_else(|| fail("no assistant message"));
    assert_eq!(assistant.content, "Let me update the table. Done.");
    assert!(turn.streaming.is_none());
    app.turns.clear();
}

/// Have the turn run by `job` ask to run `sql`, as its forwarding task
/// does; the handle resolves to the answer.
async fn ask_to_write(
    app: &mut App,
    job: JobId,
    sql: &'static str,
) -> tokio::task::JoinHandle<bool> {
    ask_to_write_held(app, job, sql, Hold::NotPermitted).await
}

/// The same, for a write held for `hold`.
async fn ask_to_write_held(
    app: &mut App,
    job: JobId,
    sql: &'static str,
    hold: Hold,
) -> tokio::task::JoinHandle<bool> {
    let (sink, mut rx) = events::channel();
    let recorder = events::TurnRecorder::new(sink);
    let pending = tokio::spawn(async move { recorder.ask_permission(sql, hold).await });
    let request = match rx.recv().await.unwrap_or_else(|| fail("no event")) {
        AgentEvent::PermissionRequired(request) => request,
        other => fail(&format!("expected PermissionRequired, got {other:?}")),
    };
    app.msg_tx
        .send(AppMsg::Turn(
            job,
            Box::new(AgentEvent::PermissionRequired(request)),
        ))
        .unwrap_or_else(|e| fail(&e.to_string()));
    app.pump();
    pending
}

/// A buffer's rows as text.
fn rows_of(buffer: &Buffer) -> Vec<String> {
    buffer
        .content()
        .chunks(usize::from(buffer.area.width).max(1))
        .map(|row| row.iter().map(ratatui::buffer::Cell::symbol).collect())
        .collect()
}

/// The rows of a frame drawn 80 columns by 30 rows.
fn screen(app: &App) -> Vec<String> {
    let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(80, 30))
        .unwrap_or_else(|e| fail(&e.to_string()));
    terminal
        .draw(|frame| ui::draw(frame, app))
        .unwrap_or_else(|e| fail(&e.to_string()));
    rows_of(terminal.backend().buffer())
}

/// The input rows of a drawn frame: everything below the transcript's
/// last separator.
fn overlay(app: &App) -> String {
    let rows = screen(app);
    let start = rows
        .iter()
        .rposition(|row| row.contains("Run this statement?"))
        .and_then(|end| {
            rows.iter()
                .take(end)
                .rposition(|row| row.trim_start().starts_with('\u{2500}'))
        })
        .unwrap_or_else(|| fail(&format!("no permission overlay:\n{}", rows.join("\n"))));
    rows.get(start..).unwrap_or_default().join("\n")
}

#[tokio::test(flavor = "multi_thread")]
async fn the_overlay_shows_a_pending_agent_write_after_the_transcript_clears() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let turn = waiting_turn(&app);
    let job = turn.job.id;
    app.turns.push(turn);
    let pending = ask_to_write(&mut app, job, "DELETE FROM t").await;

    app.clear_transcript();
    assert!(app.messages.is_empty());
    let drawn = overlay(&app);
    assert!(drawn.contains("The agent wants to run:"), "{drawn}");
    assert!(drawn.contains("DELETE FROM t"), "{drawn}");

    app.handle_key_event(KeyCode::Char('n'), KeyModifiers::NONE);
    assert!(!app.awaiting_permission());
    assert!(!pending.await.unwrap_or_else(|e| fail(&e.to_string())));
}

/// Writes allowed for the session (`a`, `--allow-write`) still ask once a
/// turn has read document text, and the prompt says why.
#[tokio::test(flavor = "multi_thread")]
async fn a_write_held_for_document_text_asks_with_the_reason_though_writes_are_allowed() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    app.allow_write.store(true, Ordering::Relaxed);
    let turn = waiting_turn(&app);
    let job = turn.job.id;
    app.turns.push(turn);
    let pending = ask_to_write_held(&mut app, job, "DELETE FROM t", Hold::ReadDocuments).await;

    assert!(app.awaiting_permission());
    let notice = Hold::ReadDocuments
        .notice()
        .unwrap_or_else(|| fail("no notice"));
    assert!(
        app.messages.iter().any(|m| m.content.contains(notice)),
        "{:?}",
        app.messages
    );
    let drawn = overlay(&app);
    assert!(drawn.contains("DELETE FROM t"), "{drawn}");
    assert!(drawn.contains("This turn read document text"), "{drawn}");

    app.handle_key_event(KeyCode::Char('y'), KeyModifiers::NONE);
    assert!(pending.await.unwrap_or_else(|e| fail(&e.to_string())));

    // A write that is only a write carries no notice.
    let plain = ask_to_write(&mut app, job, "DELETE FROM t").await;
    assert!(!overlay(&app).contains("read document text"));
    app.handle_key_event(KeyCode::Char('n'), KeyModifiers::NONE);
    assert!(!plain.await.unwrap_or_else(|e| fail(&e.to_string())));
}

#[tokio::test(flavor = "multi_thread")]
async fn an_agent_write_stays_on_screen_across_a_new_session() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let turn = waiting_turn(&app);
    let job = turn.job.id;
    app.turns.push(turn);
    let pending = ask_to_write(&mut app, job, "DELETE FROM t").await;

    app.handle_slash_command("/new");
    db_settle(&mut app).await;
    assert!(
        !app.messages
            .iter()
            .any(|m| m.content.contains("DELETE FROM t")),
        "the switch cleared the transcript's note"
    );
    let drawn = overlay(&app);
    assert!(drawn.contains("DELETE FROM t"), "{drawn}");

    app.handle_key_event(KeyCode::Char('n'), KeyModifiers::NONE);
    assert!(!app.awaiting_permission());
    assert!(!pending.await.unwrap_or_else(|e| fail(&e.to_string())));
}

#[tokio::test(flavor = "multi_thread")]
async fn after_resume_the_overlay_names_the_writes_own_session() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let other = app
        .db
        .run_at(Priority::Interactive, |db| {
            sessions::create_session(db, "m", ChatMode::Chat, None)
        })
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let turn = waiting_turn(&app);
    let job = turn.job.id;
    let whose = turn.whose();
    app.turns.push(turn);
    let pending = ask_to_write(&mut app, job, "DELETE FROM t").await;

    app.handle_slash_command(&format!("/resume {}", other.id));
    db_settle(&mut app).await;
    assert_eq!(app.session_id, other.id);
    let drawn = overlay(&app);
    assert!(drawn.contains(&format!("{whose} wants to run:")), "{drawn}");
    assert!(drawn.contains("DELETE FROM t"), "{drawn}");

    app.handle_key_event(KeyCode::Char('n'), KeyModifiers::NONE);
    assert!(!pending.await.unwrap_or_else(|e| fail(&e.to_string())));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_typed_write_stays_on_screen_when_a_new_session_lands_after_it() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    // Classifying the write and switching sessions are applied in the
    // order typed: the prompt first, then the clear.
    app.handle_slash_command("/sql DELETE FROM t");
    app.handle_slash_command("/new");
    db_settle(&mut app).await;
    assert!(matches!(app.prompts.front(), Some(Prompt::Sql(_))));
    let drawn = overlay(&app);
    assert!(
        drawn.contains("Your statement modifies the workspace:"),
        "{drawn}"
    );
    assert!(drawn.contains("DELETE FROM t"), "{drawn}");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_overlay_asks_about_the_front_prompt_and_counts_the_rest() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let turn = waiting_turn(&app);
    let job = turn.job.id;
    app.turns.push(turn);
    let first = ask_to_write(&mut app, job, "DELETE FROM first_table").await;
    let second = ask_to_write(&mut app, job, "DELETE FROM second_table").await;

    let drawn = overlay(&app);
    assert!(drawn.contains("DELETE FROM first_table"), "{drawn}");
    assert!(!drawn.contains("second_table"), "{drawn}");
    assert!(drawn.contains("(+1 more waiting)"), "{drawn}");

    app.handle_key_event(KeyCode::Char('n'), KeyModifiers::NONE);
    let drawn = overlay(&app);
    assert!(drawn.contains("DELETE FROM second_table"), "{drawn}");
    assert!(!drawn.contains("more waiting"), "{drawn}");
    app.handle_key_event(KeyCode::Char('n'), KeyModifiers::NONE);
    assert!(!first.await.unwrap_or_else(|e| fail(&e.to_string())));
    assert!(!second.await.unwrap_or_else(|e| fail(&e.to_string())));
}

#[test]
fn a_long_statement_is_capped_with_a_count_of_the_rest() {
    let sql = (0..10)
        .map(|n| format!("UPDATE t SET a = {n}"))
        .collect::<Vec<_>>()
        .join("\n");
    let pending = PendingWrite {
        heading: String::from("The agent wants to run:"),
        sql: &sql,
        notice: None,
        waiting: 0,
    };
    let lines: Vec<String> = pending
        .lines(80, 4)
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(lines.len(), 6, "{lines:?}");
    assert!(
        lines.get(3).is_some_and(|l| l.contains("SET a = 2")),
        "{lines:?}"
    );
    assert!(
        lines.get(4).is_some_and(|l| l.contains("7 more lines")),
        "{lines:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_file_loads_at_once_and_other_pastes_are_typed_in() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let file = dir.path().join("team notes.md");
    std::fs::write(&file, "# Notes\n\nThe team met.").unwrap_or_else(|e| fail(&e.to_string()));
    let dropped = dropped_path(&file);

    // The path a terminal pastes for a dropped file: loaded, not typed.
    assert!(app.handle_terminal_event(&Event::Paste(dropped.clone())));
    assert!(app.textarea.is_empty());
    let announced = last(&app);
    assert_eq!(announced.kind, MessageKind::Upload);
    assert!(
        announced.content.contains("team notes.md (job #"),
        "{}",
        announced.content
    );
    settle(&mut app).await;
    assert_ne!(
        last(&app).kind,
        MessageKind::Error,
        "{}",
        last(&app).content
    );

    // A drop of several files whose paths the terminal joins with a
    // carriage return (CRLF or a bare CR) loads them at once too: both
    // branches agree '\r' is a line break. Before the fix, the raw paste
    // reached shlex, which splits only on ' ' | '\t' | '\n', so an
    // internal '\r'/'\r\n' kept the paths stuck together and the drop was
    // misrouted to the typed-in branch — escaped paths sat in the input
    // and no Upload was announced until the user pressed Enter.
    let more = dir.path().join("more notes.md");
    std::fs::write(&more, "# More notes.").unwrap_or_else(|e| fail(&e.to_string()));
    let more_dropped = dropped_path(&more);
    for sep in ["\r\n", "\r"] {
        let before = app.messages.len();
        assert!(app.handle_terminal_event(&Event::Paste(format!("{dropped}{sep}{more_dropped}"))));
        assert!(
            app.textarea.is_empty(),
            "drop joined by {sep:?} was typed in, not loaded"
        );
        assert_eq!(
            app.messages.len(),
            before + 2,
            "a {sep:?}-separated drop announces one Upload per file"
        );
        let first = app
            .messages
            .get(before)
            .unwrap_or_else(|| fail("no first upload"));
        assert_eq!(first.kind, MessageKind::Upload);
        assert!(
            first.content.contains("team notes.md (job #"),
            "{}",
            first.content
        );
        let second = app
            .messages
            .get(before + 1)
            .unwrap_or_else(|| fail("no second upload"));
        assert_eq!(second.kind, MessageKind::Upload);
        assert!(
            second.content.contains("more notes.md (job #"),
            "{}",
            second.content
        );
        // Both files' jobs: a finish left unread would land in a later
        // step's message count (#448).
        settle_jobs(&mut app).await;
        assert_ne!(
            last(&app).kind,
            MessageKind::Error,
            "{}",
            last(&app).content
        );
    }

    // Any other paste is typed in, line breaks kept.
    app.handle_terminal_event(&Event::Paste(String::from("SELECT 1\rFROM t\r\nLIMIT 1")));
    assert_eq!(app.textarea.lines(), ["SELECT 1", "FROM t", "LIMIT 1"]);

    // A path pasted into text already typed joins it and waits for Enter.
    app.set_input("/ingest ");
    let before = app.messages.len();
    app.handle_terminal_event(&Event::Paste(dropped.clone()));
    assert_eq!(app.textarea.lines(), [format!("/ingest {dropped}")]);
    assert_eq!(app.messages.len(), before);
    app.submit_message();
    assert_eq!(last(&app).kind, MessageKind::Upload);
    settle(&mut app).await;

    // Names cut short by a word starting with `#` load nothing, typed
    // as a line or after /ingest, and the error says why.
    for line in [
        format!("{dropped} #drafts.md"),
        format!("/ingest {dropped} #drafts.md"),
    ] {
        let before = app.messages.len();
        app.submit_text(line);
        assert_eq!(app.messages.len(), before + 1);
        assert_eq!(last(&app).kind, MessageKind::Error);
        assert_eq!(last(&app).content, FileLine::COMMENT);
    }

    // A write prompt takes keys only; a paste does not answer or queue.
    app.clear_input();
    app.handle_slash_command("/sql CREATE TABLE t AS SELECT 1 AS a");
    db_settle(&mut app).await;
    assert!(app.awaiting_permission());
    let before = app.messages.len();
    app.handle_terminal_event(&Event::Paste(dropped));
    assert_eq!(app.messages.len(), before);
    assert!(app.textarea.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_event_loop_answers_typed_sql_and_quits_on_ctrl_c() {
    use crossterm::event::KeyEvent;
    use ratatui::backend::TestBackend;

    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let app = app(dir.path());
    let (keys, input) = futures::channel::mpsc::unbounded::<std::io::Result<Event>>();
    let press = |code| Ok(Event::Key(KeyEvent::new(code, KeyModifiers::NONE)));
    for ch in "SELECT 6 * 7 AS answer".chars() {
        assert!(keys.unbounded_send(press(KeyCode::Char(ch))).is_ok());
    }
    assert!(keys.unbounded_send(press(KeyCode::Enter)).is_ok());
    let mut terminal =
        ratatui::Terminal::new(TestBackend::new(80, 30)).unwrap_or_else(|e| fail(&e.to_string()));
    let running = tokio::spawn(async move {
        let result = app
            .run_with(&mut terminal, input, Clipboard::terminal_only(Vec::new()))
            .await;
        (result, terminal)
    });
    // The answer arrives through the job queue and the loop draws it;
    // then Ctrl+C with nothing running quits.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(
        keys.unbounded_send(Ok(Event::Key(KeyEvent::new(
            KeyCode::Char('c'),
            KeyModifiers::CONTROL
        ))))
        .is_ok()
    );
    let (result, terminal) = tokio::time::timeout(Duration::from_secs(10), running)
        .await
        .unwrap_or_else(|_| fail("the loop did not quit"))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(result.is_ok(), "{result:?}");
    let screen: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(ratatui::buffer::Cell::symbol)
        .collect();
    // The result table, not a session id that happens to hold "42".
    assert!(
        screen.contains("answer") && screen.contains("(1 rows)"),
        "{screen}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn ctrl_c_with_jobs_running_asks_for_a_second_press() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let (sql_sink, sql_rx) = tokio::sync::oneshot::channel::<()>();
    app.jobs
        .submit(JobSpec::new(JobKind::Sql, "slow"), |_| async move {
            drop(sql_rx.await);
            Ok(String::new())
        });
    pump_until(&mut app, |app| !app.active_jobs.is_empty()).await;
    assert_eq!(ui::JobStrip::of(&app).height(), 1, "the strip shows it");
    app.handle_key_event(KeyCode::Char('c'), KeyModifiers::CONTROL);
    assert_eq!(app.quit, Quit::Armed);
    assert!(last(&app).content.contains("still running"));
    app.handle_key_event(KeyCode::Char('c'), KeyModifiers::CONTROL);
    assert_eq!(app.quit, Quit::Now);
    assert!(sql_sink.send(()).is_ok());
    pump_until(&mut app, |app| app.active_jobs.is_empty()).await;
    assert_eq!(ui::JobStrip::of(&app).height(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_jobs_box_follows_the_queue_and_cancels_the_highlighted_job() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    app.handle_slash_command("/jobs");
    assert!(app.picker.is_none(), "nothing to list");
    assert_eq!(last(&app).content, "No jobs yet.");

    let (_first_sink, first_rx) = tokio::sync::oneshot::channel::<()>();
    app.jobs
        .submit(JobSpec::new(JobKind::Sql, "first"), |_| async move {
            drop(first_rx.await);
            Ok(String::new())
        });
    // Running, not merely queued: the rows below name the state.
    let running = |app: &App| {
        app.active_jobs
            .iter()
            .filter(|job| job.state == JobState::Running)
            .count()
    };
    pump_until(&mut app, |app| running(app) == 1).await;
    app.handle_slash_command("/jobs");
    let row_of = |app: &App, label: &str| {
        screen(app)
            .into_iter()
            .find(|row| row.contains(label))
            .unwrap_or_else(|| fail(&format!("no row for {label}")))
    };
    assert!(row_of(&app, "first").contains("\u{25B8} 1    running"));

    // A job submitted while the box is open appears above, and the
    // highlight stays on the job it was on.
    app.jobs
        .submit(JobSpec::new(JobKind::Sql, "second"), |ctx| async move {
            ctx.cancel_token().cancelled().await;
            Ok(String::new())
        });
    pump_until(&mut app, |app| running(app) == 2).await;
    assert!(row_of(&app, "first").contains('\u{25B8}'));
    assert!(!row_of(&app, "second").contains('\u{25B8}'));

    // Keys go to the box, not the input; `c` cancels the highlighted job.
    app.handle_key_event(KeyCode::Up, KeyModifiers::NONE);
    app.handle_key_event(KeyCode::Char('c'), KeyModifiers::NONE);
    assert!(app.textarea.is_empty());
    assert!(last(&app).content.contains("Cancelling job #2"));
    pump_until(&mut app, |app| app.active_jobs.len() == 1).await;
    // The box follows the job to its end: this one's work returns
    // as soon as it is cancelled.
    let ended = row_of(&app, "second");
    assert!(ended.contains("\u{25B8} 2    succeeded"), "{ended}");

    // Enter posts the highlighted job's details and closes the box.
    app.handle_key_event(KeyCode::End, KeyModifiers::NONE);
    app.handle_key_event(KeyCode::Enter, KeyModifiers::NONE);
    assert!(app.picker.is_none());
    assert_eq!(last(&app).content, "Job #1 running sql first");
    app.handle_key_event(KeyCode::Char('x'), KeyModifiers::NONE);
    assert_eq!(
        app.textarea.lines().join(""),
        "x",
        "the input has the keys again"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_sessions_box_resumes_the_highlighted_session() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let current = app.session_id.clone();
    let newer = app
        .db
        .run_at(Priority::Interactive, |db| {
            sessions::create_session(db, "m", ChatMode::Chat, None)
        })
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));

    app.handle_slash_command("/sessions");
    db_settle(&mut app).await;
    let drawn = screen(&app).join("\n");
    assert!(drawn.contains(" Sessions "), "{drawn}");
    assert!(drawn.contains("enter resume"), "{drawn}");
    // The session on screen is marked and highlighted, under the
    // newer one.
    assert!(
        drawn.contains(&format!("\u{25B8} * {}", current.short())),
        "{drawn}"
    );
    app.handle_key_event(KeyCode::Up, KeyModifiers::NONE);
    app.handle_key_event(KeyCode::Enter, KeyModifiers::NONE);
    assert!(app.picker.is_none());
    db_settle(&mut app).await;
    assert_eq!(app.session_id, newer.id);
}

#[test]
fn a_turns_phase_follows_its_events_and_counts_seconds() {
    let at = |second: i64| {
        Timestamp::from_second(1_000_000 + second).unwrap_or_else(|e| fail(&e.to_string()))
    };
    let started = Some(at(0));
    let phase = Phase::Waiting { since: None };
    assert_eq!(phase.note(started, at(3)), "waiting 3s");
    assert_eq!(phase.note(None, at(3)), "waiting 0s", "not started yet");

    let phase = phase.after(&AgentEvent::Reasoning, at(3));
    assert_eq!(phase.note(started, at(44)), "thinking 41s");
    // More reasoning in the same model call keeps counting from its start.
    let phase = phase.after(&AgentEvent::Reasoning, at(20));
    assert_eq!(phase.note(started, at(44)), "thinking 41s");
    // A status line says nothing about the phase.
    let phase = phase.after(&AgentEvent::Status(String::from("loading")), at(21));
    assert_eq!(phase, Phase::Thinking { since: at(3) });

    let phase = phase.after(
        &AgentEvent::ToolStarted {
            tool: ToolName::RunSql,
            detail: String::new(),
        },
        at(50),
    );
    assert_eq!(phase.note(started, at(51)), "running run_sql");
    // After a tool the model is called again, and the wait starts over.
    let phase = Phase::Waiting {
        since: Some(at(52)),
    };
    assert_eq!(phase.note(started, at(59)), "waiting 7s");
    let phase = phase.after(&AgentEvent::TextDelta(String::from("Hi")), at(60));
    assert_eq!(phase.note(started, at(61)), "answering");
    // A clock that steps back never shows a negative count.
    assert_eq!(
        Phase::Thinking { since: at(9) }.note(started, at(5)),
        "thinking 0s"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_job_strip_says_what_a_running_turn_is_doing() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let turn = waiting_turn(&app);
    let job = turn.job.id;
    app.turns.push(turn);
    pump_until(&mut app, |app| {
        app.active_jobs
            .iter()
            .any(|info| info.id == job && info.state == JobState::Running)
    })
    .await;
    let strip = |app: &App| {
        screen(app)
            .into_iter()
            .find(|row| row.contains("chat question"))
            .unwrap_or_else(|| fail("no strip row"))
    };
    assert!(
        strip(&app).contains("question  waiting "),
        "{}",
        strip(&app)
    );

    let event = |app: &mut App, event: AgentEvent| {
        let at = app
            .turns
            .iter()
            .position(|t| t.job.id == job)
            .unwrap_or_else(|| fail("no turn"));
        let mut turn = app.turns.remove(at);
        app.handle_turn_event(&mut turn, event);
        app.turns.insert(at, turn);
    };
    event(&mut app, AgentEvent::Reasoning);
    assert!(
        strip(&app).contains("question  thinking 0s"),
        "{}",
        strip(&app)
    );
    event(&mut app, AgentEvent::TextDelta(String::from("The answer")));
    assert!(
        strip(&app).contains("question  answering"),
        "{}",
        strip(&app)
    );
    // Reasoning itself adds nothing to the transcript.
    assert_eq!(last(&app).content, "The answer");
    app.stop_jobs().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_job_strip_draws_reported_progress_as_a_gauge() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let (sink, rx) = tokio::sync::oneshot::channel::<()>();
    app.jobs
        .submit(JobSpec::new(JobKind::Sql, "slow"), |ctx| async move {
            ctx.progress(1, 4);
            drop(rx.await);
            Ok(String::new())
        });
    pump_until(&mut app, |app| {
        app.active_jobs.iter().any(|job| job.progress.is_some())
    })
    .await;
    let rows = screen(&app);
    let row = rows
        .iter()
        .find(|row| row.contains("slow"))
        .unwrap_or_else(|| fail(&rows.join("\n")));
    // A quarter of the gauge's columns are filled.
    let gauge = format!(
        "slow  1/4 (25%) {}{} ",
        "\u{2501}".repeat(4),
        "\u{2500}".repeat(12)
    );
    assert!(row.contains(&gauge), "{row}");
    assert!(sink.send(()).is_ok());
    pump_until(&mut app, |app| app.active_jobs.is_empty()).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_command_popup_picks_fills_in_and_runs() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let typed = |app: &mut App, text: &str| {
        for ch in text.chars() {
            app.handle_key_event(KeyCode::Char(ch), KeyModifiers::NONE);
        }
    };
    let input = |app: &App| app.textarea.lines().join("\n");
    let highlighted = |app: &App| {
        app.completion()
            .and_then(|c| c.get(app.completion_selected()).map(|s| s.word.clone()))
    };

    // Tab fills the highlighted command in; its verbs follow.
    typed(&mut app, "/gr");
    assert_eq!(highlighted(&app).as_deref(), Some("/graph"));
    app.handle_key_event(KeyCode::Tab, KeyModifiers::NONE);
    assert_eq!(input(&app), "/graph ");
    typed(&mut app, "me");
    assert_eq!(highlighted(&app).as_deref(), Some("merges"));

    // Down and Up move the highlight and wrap; they leave history alone.
    app.handle_key_event(KeyCode::Char('u'), KeyModifiers::CONTROL);
    typed(&mut app, "/s");
    assert_eq!(highlighted(&app).as_deref(), Some("/sql"));
    app.handle_key_event(KeyCode::Down, KeyModifiers::NONE);
    assert_eq!(highlighted(&app).as_deref(), Some("/schema"));
    app.handle_key_event(KeyCode::Up, KeyModifiers::NONE);
    app.handle_key_event(KeyCode::Up, KeyModifiers::NONE);
    assert_eq!(highlighted(&app).as_deref(), Some("/steps"));
    assert_eq!(input(&app), "/s");

    // Esc hides it without cancelling anything; typing brings it back.
    app.handle_key_event(KeyCode::Esc, KeyModifiers::NONE);
    assert!(app.completion().is_none());
    typed(&mut app, "c");
    assert_eq!(highlighted(&app).as_deref(), Some("/schema"));

    // Enter on a command that takes nothing fills it in and runs it.
    app.handle_key_event(KeyCode::Char('u'), KeyModifiers::CONTROL);
    typed(&mut app, "/he");
    app.handle_key_event(KeyCode::Enter, KeyModifiers::NONE);
    assert!(app.textarea.is_empty());
    assert!(last(&app).content.contains("Commands:"));

    // Enter on one that takes more only fills it in; with nothing left
    // to fill, Enter sends the line as typed.
    typed(&mut app, "/mo");
    app.handle_key_event(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(input(&app), "/mode ");
    typed(&mut app, "q");
    app.handle_key_event(KeyCode::Tab, KeyModifiers::NONE);
    assert_eq!(input(&app), "/mode query");
    app.handle_key_event(KeyCode::Enter, KeyModifiers::NONE);
    assert!(app.textarea.is_empty(), "sent as typed");
    db_settle(&mut app).await;
    assert!(
        last(&app).content.contains("query"),
        "{}",
        last(&app).content
    );

    // Plain text, and the cursor moved off the end, show no popup.
    app.handle_key_event(KeyCode::Char('u'), KeyModifiers::CONTROL);
    typed(&mut app, "what is /s");
    assert!(app.completion().is_none());
    app.handle_key_event(KeyCode::Char('u'), KeyModifiers::CONTROL);
    typed(&mut app, "/s");
    app.handle_key_event(KeyCode::Left, KeyModifiers::NONE);
    assert!(app.completion().is_none());

    // The popup's rows: the highlighted one marked, labels aligned.
    let items = Completion::for_line("/mode ")
        .map(|c| c.items)
        .unwrap_or_default();
    let popup = |items: &[Suggestion], selected| {
        let mut buffer = Buffer::empty(Rect::new(0, 0, 60, 8));
        (&ui::CompletionPopup::new(items, selected)).render(buffer.area, &mut buffer);
        rows_of(&buffer)
    };
    let rows = popup(&items, 1);
    assert!(
        rows.first().is_some_and(|r| r.starts_with("  chat ")),
        "{rows:?}"
    );
    assert!(
        rows.get(1)
            .is_some_and(|r| r.starts_with("\u{25B8} query ")),
        "{rows:?}"
    );

    // More entries than rows: the list scrolls to the highlighted one.
    let commands = Completion::for_line("/")
        .map(|c| c.items)
        .unwrap_or_default();
    assert!(commands.len() > 10, "{}", commands.len());
    let rows = popup(&commands, 10);
    let label = commands.get(10).map_or("", |s| s.label.as_str());
    assert!(
        rows.last()
            .is_some_and(|r| r.starts_with(&format!("\u{25B8} {label}"))),
        "{rows:?}"
    );
    assert_eq!(rows.iter().filter(|r| r.starts_with('\u{25B8}')).count(), 1);
}

/// The words the popup offers for the input as it stands.
fn offered(app: &App) -> Vec<String> {
    app.completion()
        .map(|c| c.items.into_iter().map(|s| s.word).collect())
        .unwrap_or_default()
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_completion_follows_ingests_and_typed_statements() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    app.load_sql_schema()
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    app.set_input("SELECT * FROM sa");
    assert!(offered(&app).is_empty(), "no tables yet");

    let file = dir.path().join("sales.csv");
    std::fs::write(&file, "region,revenue\nnorth,10\n").unwrap_or_else(|e| fail(&e.to_string()));
    app.run_job(CliJob::Ingest(file));
    settle(&mut app).await;
    db_settle(&mut app).await;
    app.set_input("SELECT * FROM sa");
    assert_eq!(offered(&app), ["sales"]);
    app.set_input("SELECT * FROM sales s WHERE s.re");
    assert_eq!(offered(&app), ["region", "revenue"]);

    app.allow_write.store(true, Ordering::Relaxed);
    app.handle_slash_command("/sql CREATE TABLE stores (id INTEGER)");
    settle(&mut app).await;
    db_settle(&mut app).await;
    app.set_input("SELECT * FROM st");
    assert_eq!(offered(&app), ["stores"]);
    assert!(
        app.sql_schema
            .tables
            .iter()
            .all(|t| !t.name.name.starts_with("_quack_")),
        "{:?}",
        app.sql_schema
    );

    // Tab fills the name in where the cursor is, and leaves the rest.
    app.set_input("SELECT re FROM sales");
    app.textarea.move_cursor(CursorMove::Jump(0, 9));
    assert!(app.handle_completion_key(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(app.textarea.lines().concat(), "SELECT region FROM sales");
    assert_eq!(app.textarea.cursor().1, 13);
}

#[tokio::test(flavor = "multi_thread")]
async fn database_commands_run_in_order_off_the_loop_and_input_waits_for_a_switch() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let file = dir.path().join("notes.md");
    std::fs::write(&file, "# Notes\n\nSomething to pin.").unwrap_or_else(|e| fail(&e.to_string()));
    app.run_job(CliJob::Ingest(file));
    settle(&mut app).await;
    // The ingest refreshed the completion schema on the worker.
    db_settle(&mut app).await;

    // A write then a read, sent back to back, answer in that order: the
    // listing sees the pin.
    let id = app
        .db
        .run(WorkspaceDb::list_documents)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()))
        .first()
        .map_or_else(|| fail("no document"), |d| d.id.clone());
    app.handle_slash_command(&format!("/pin {id}"));
    app.handle_slash_command("/docs");
    // Neither has answered on the loop yet: nothing blocked here.
    assert_eq!(app.pending_db, 2);
    db_settle(&mut app).await;
    assert!(
        last(&app).content.contains("pinned"),
        "{}",
        last(&app).content
    );

    // Input typed during a switch waits for it, then lands in the new
    // session.
    let old = app.session_id.clone();
    app.handle_slash_command("/new");
    app.set_input("/workspace");
    app.submit_message();
    assert!(
        last(&app)
            .content
            .contains("Waiting for the session switch"),
        "{}",
        last(&app).content
    );
    db_settle(&mut app).await;
    assert_ne!(app.session_id, old);
    assert!(
        last(&app).content.contains(app.session_id.as_str()),
        "the deferred /workspace ran in the new session: {}",
        last(&app).content
    );
    assert!(app.switching.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn embeddings_refresh_is_a_job_and_stale_vectors_are_noted_at_startup() {
    use quack_core::config::{BaseUrl, ProviderConfig, ProviderName, ProviderType};
    use quack_core::embedding::{Dimension, Vector};
    use quack_core::storage::workspace::{DocumentStatus, NewChunk, NewDocument};

    // Without an embedding model the job says what is missing.
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    app.handle_slash_command("/embeddings refresh");
    settle(&mut app).await;
    assert_eq!(last(&app).kind, MessageKind::Error);
    assert!(
        last(&app).content.contains("no embedding model configured"),
        "{}",
        last(&app).content
    );

    // With one, a vector from another profile is noted when the
    // session opens.
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = Config::default();
    config.embedding.dimension = Some(Dimension::new(4));
    config.embedding.model = Some(
        "ollama/embeddinggemma"
            .parse()
            .unwrap_or_else(|e: CoreError| fail(&e.to_string())),
    );
    config.providers.insert(
        "ollama"
            .parse::<ProviderName>()
            .unwrap_or_else(|e| fail(&e.to_string())),
        ProviderConfig {
            base_url: Some(
                BaseUrl::try_from(String::from("http://127.0.0.1:9"))
                    .unwrap_or_else(|e| fail(&e.to_string())),
            ),
            ..ProviderConfig::new(ProviderType::Ollama)
        },
    );
    let mut app = app_with(dir.path(), config);
    app.note_embedding_status()
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let before = app.messages.len();
    app.db
        .run(|db| {
            db.insert_document(
                &NewDocument::new(&DocumentId::from("d"), "a.md", "text/markdown", 1)
                    .with_status(DocumentStatus::Ready),
            )?;
            db.insert_chunk(&NewChunk {
                id: &ChunkId::from("c"),
                document_id: &DocumentId::from("d"),
                chunk_index: 0,
                content: "levee report",
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: Some(&Vector::from(vec![1.0, 0.0, 0.0, 0.0])),
            })?;
            db.execute_statement("UPDATE _quack_chunks SET embedding_profile = 'older'")
        })
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(app.messages.len(), before, "nothing to note while current");
    app.note_embedding_status()
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let note = &last(&app).content;
    assert!(
        note.contains("keyword search only") && note.contains("/embeddings refresh"),
        "{note}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_replays_at_startup_through_the_writer() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let session = app.session_id.clone();
    app.db
        .run(move |db| {
            sessions::record_turn(
                db,
                &session,
                "how many storms?",
                Timestamp::now(),
                &AgentResponse {
                    content: String::from("Twelve storms."),
                    ..AgentResponse::default()
                },
            )
        })
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    app.load_current_session()
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        app.messages
            .iter()
            .any(|m| m.kind == MessageKind::Assistant && m.content == "Twelve storms."),
        "the recorded answer is back in the transcript"
    );
    assert!(
        app.messages
            .iter()
            .any(|m| m.content.contains("Resumed session"))
    );
}

#[test]
fn the_transcript_rerenders_only_what_changed() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    let text = |rows: &[Row]| -> String {
        rows.iter()
            .map(|l| {
                l.line
                    .spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    app.messages.push(Message::new(
        MessageKind::Assistant,
        String::from("Streaming"),
    ));
    let first = ui::format_messages(&app, 60);
    let entries = app.wrap_cache.borrow().len();
    assert_eq!(entries, app.messages.len());
    let keys: Vec<Option<u64>> = app
        .wrap_cache
        .borrow()
        .iter()
        .map(|e| e.as_ref().map(|w| w.key))
        .collect();

    // A streamed delta changes the last message only.
    if let Some(last) = app.messages.last_mut() {
        last.content.push_str(" more text");
    }
    let second = ui::format_messages(&app, 60);
    assert!(text(&second).contains("Streaming more text"));
    assert_ne!(text(&first), text(&second));
    let after: Vec<Option<u64>> = app
        .wrap_cache
        .borrow()
        .iter()
        .map(|e| e.as_ref().map(|w| w.key))
        .collect();
    let unchanged = keys.iter().zip(&after).filter(|(a, b)| a == b).count();
    assert_eq!(unchanged, keys.len().saturating_sub(1));

    // A new width, /steps, and /clear all show at once.
    assert!(
        ui::format_messages(&app, 20)
            .iter()
            .all(|l| l.line.width() <= 20)
    );
    app.clear_transcript();
    assert!(ui::format_messages(&app, 60).is_empty());
    assert!(app.wrap_cache.borrow().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn questions_queue_per_session_while_other_work_runs() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    // No chat model: each turn fails fast, but it still goes through
    // the session's lane in order and leaves the transcript usable.
    app.start_agent_turn(String::from("first?"));
    app.start_agent_turn(String::from("second?"));
    assert_eq!(app.turns.len(), 2);
    assert!(
        last(&app).content.contains("Queued as job #2"),
        "{}",
        last(&app).content
    );
    // SQL runs alongside.
    app.set_input("SELECT 6 * 7 AS answer");
    app.submit_message();
    // A job can finish between the result drain and the job-event drain
    // of one pump, so wait for the result itself, not just an idle strip.
    pump_until(&mut app, |app| {
        app.turns.is_empty()
            && app.active_jobs.is_empty()
            && app
                .messages
                .iter()
                .any(|m| m.kind == MessageKind::Sql && m.content.contains("42"))
    })
    .await;
    let jobs = app.jobs.list();
    assert_eq!(jobs.len(), 3);
    let chats: Vec<_> = jobs.iter().filter(|j| j.kind == JobKind::Chat).collect();
    assert!(chats.iter().all(|j| j.state == JobState::Failed));
    // The lane ran them in order.
    assert!(
        chats
            .first()
            .and_then(|a| a.finished_at)
            .zip(chats.get(1).and_then(|b| b.started_at))
            .is_some_and(|(a_end, b_start)| a_end <= b_start),
        "{chats:#?}"
    );
    assert!(
        app.messages.iter().any(|m| m.kind == MessageKind::Error),
        "the failure is in the transcript"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn without_a_chat_model_questions_say_how_to_set_one_and_sql_still_runs() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    assert!(app.messages.iter().any(|m| m.content == NO_CHAT_MODEL_TEXT));

    app.set_input("how many orders shipped late?");
    app.submit_message();
    assert!(app.turns.is_empty());
    assert!(last(&app).content.contains("quack doctor"));

    app.set_input("SELECT 41 + 1 AS answer");
    app.submit_message();
    settle(&mut app).await;
    assert!(last(&app).content.contains("42"), "{}", last(&app).content);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_question_that_starts_like_sql_is_asked_when_it_does_not_parse() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = Config::default();
    config.general.chat_model = "ollama/llama3".parse().ok();
    assert!(config.general.chat_model.is_some());
    let mut app = app_with(dir.path(), config);
    let chat_jobs = |app: &App| {
        app.jobs
            .list()
            .iter()
            .filter(|job| job.kind == JobKind::Chat)
            .count()
    };

    app.set_input("show me the first five rows");
    app.submit_message();
    db_settle(&mut app).await;
    assert_eq!(chat_jobs(&app), 1, "it became a turn");
    for job in app.jobs.list() {
        app.jobs.cancel(job.id);
    }
    assert!(
        app.messages
            .iter()
            .any(|m| m.kind == MessageKind::System && m.content.starts_with("Not SQL (")),
    );
    // Shown once, and not remembered as the last statement.
    let shown = app
        .messages
        .iter()
        .filter(|m| m.kind == MessageKind::User)
        .count();
    assert_eq!(shown, 1);
    assert!(app.last_sql.is_none());

    // `/sql` says it is a statement, so its syntax error is reported.
    app.handle_slash_command("/sql show me the first five rows");
    db_settle(&mut app).await;
    assert_eq!(last(&app).kind, MessageKind::Error);
    assert_eq!(chat_jobs(&app), 1);

    // A statement that parses runs as SQL, even when it then fails.
    app.set_input("SELECT * FROM no_such_table");
    app.submit_message();
    settle(&mut app).await;
    assert_eq!(last(&app).kind, MessageKind::Error);
    assert!(last(&app).content.contains("no_such_table"));
    assert_eq!(chat_jobs(&app), 1);
    assert_eq!(app.last_sql.as_deref(), Some("SELECT * FROM no_such_table"));
}

#[tokio::test(flavor = "multi_thread")]
async fn without_a_chat_model_a_line_that_starts_like_sql_reports_its_syntax_error() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    app.set_input("show me the first five rows");
    app.submit_message();
    db_settle(&mut app).await;
    assert_eq!(last(&app).kind, MessageKind::Error);
    assert!(app.jobs.list().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn typed_input_is_kept_across_sessions_and_browsed_with_up_and_down() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    {
        let mut app = app(dir.path());
        app.set_input("/tables");
        app.submit_message();
        db_settle(&mut app).await;
    }
    // History is the data directory's, not the workspace's. Another
    // workspace proves it: the first app's writer may still hold "ws",
    // which Windows locks exclusively until the last handle closes.
    let mut again = app_in(dir.path(), "ws2", Config::default());
    assert_eq!(again.history.lines, vec![String::from("/tables")]);

    // Up recalls the newest, then older ones; Down comes back and past
    // the newest to an empty input.
    again.history.push(String::from("SELECT 1"));
    let input = |app: &App| app.textarea.lines().join("\n");
    again.handle_key_event(KeyCode::Up, KeyModifiers::NONE);
    assert_eq!(input(&again), "SELECT 1");
    again.handle_key_event(KeyCode::Up, KeyModifiers::NONE);
    assert_eq!(input(&again), "/tables");
    again.handle_key_event(KeyCode::Up, KeyModifiers::NONE);
    assert_eq!(input(&again), "/tables", "the oldest stays");
    assert!(
        again.completion().is_none(),
        "no popup over a recalled line"
    );
    again.handle_key_event(KeyCode::Down, KeyModifiers::NONE);
    assert_eq!(input(&again), "SELECT 1");
    again.handle_key_event(KeyCode::Down, KeyModifiers::NONE);
    assert!(again.textarea.is_empty());
    assert!(!again.history.browsing());

    // A line with a newline is kept for the session, not the file.
    again.history.push(String::from("a\nb"));
    assert_eq!(again.history.lines.len(), 3);
    let saved = app(dir.path());
    assert_eq!(
        saved.history.lines,
        vec![String::from("/tables"), String::from("SELECT 1")]
    );
}

#[test]
fn a_scrollbar_follows_the_transcript_once_it_outgrows_the_screen() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    // The last column of the transcript's rows: its first and last.
    let bar = |app: &App| {
        let rows = screen(app);
        let column = |row: usize| rows.get(row).and_then(|r| r.chars().last());
        (column(2), column(25))
    };
    app.messages.clear();
    app.note(MessageKind::System, "short");
    assert_eq!(bar(&app), (Some(' '), Some(' ')), "nothing to scroll");
    for n in 0..60 {
        app.note(MessageKind::System, format!("line {n}"));
    }
    assert_eq!(
        bar(&app),
        (Some('\u{2502}'), Some('\u{2588}')),
        "at the newest"
    );
    app.handle_key_event(KeyCode::Home, KeyModifiers::NONE);
    assert_eq!(
        bar(&app),
        (Some('\u{2588}'), Some('\u{2502}')),
        "at the top"
    );
}

#[test]
fn home_page_down_and_end_move_through_the_transcript() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    app.scroll_limit.set(40);
    app.handle_key_event(KeyCode::PageUp, KeyModifiers::NONE);
    assert_eq!(app.scroll, Scroll::Back(15));
    app.handle_key_event(KeyCode::Home, KeyModifiers::NONE);
    assert_eq!(app.scroll, Scroll::Top);
    // From the top, PageDown moves down from the first line.
    app.handle_key_event(KeyCode::PageDown, KeyModifiers::NONE);
    assert_eq!(app.scroll, Scroll::Back(25));
    app.handle_key_event(KeyCode::End, KeyModifiers::NONE);
    assert_eq!(app.scroll, Scroll::Latest);
    app.handle_key_event(KeyCode::PageUp, KeyModifiers::NONE);
    app.handle_key_event(KeyCode::PageDown, KeyModifiers::NONE);
    assert_eq!(app.scroll, Scroll::Latest);
    // A new message follows the transcript down.
    app.handle_key_event(KeyCode::PageUp, KeyModifiers::NONE);
    app.note(MessageKind::System, "news");
    assert_eq!(app.scroll, Scroll::Latest);
}

fn mouse(kind: MouseEventKind, column: u16, row: u16) -> Event {
    Event::Mouse(MouseEvent {
        kind,
        column,
        row,
        modifiers: KeyModifiers::NONE,
    })
}

/// An app whose transcript is two notes, drawn once so the mouse can
/// be placed on it: `alpha beta` on screen row 2 and `gamma` on row 4.
fn app_with_two_notes(dir: &Path) -> App {
    let mut app = app(dir);
    app.messages.clear();
    app.note(MessageKind::System, "alpha beta");
    app.note(MessageKind::System, "gamma");
    drop(screen(&app));
    app
}

#[test]
fn a_drag_over_the_transcript_selects_it_and_letting_go_copies_it() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app_with_two_notes(dir.path());
    let left = MouseButton::Left;

    assert!(app.handle_terminal_event(&mouse(MouseEventKind::Down(left), 9, 2)));
    assert!(app.handle_terminal_event(&mouse(MouseEventKind::Drag(left), 5, 4)));
    assert!(app.to_copy.is_none(), "nothing is copied mid-drag");
    assert!(app.handle_terminal_event(&mouse(MouseEventKind::Up(left), 5, 4)));
    assert_eq!(app.to_copy.as_deref(), Some("beta\n\ngam"));
    assert!(
        app.selection.is_some_and(|s| !s.is_dragging()),
        "the highlight stays"
    );
    // A drag with no button down before it, as after the highlight.
    assert!(!app.handle_terminal_event(&mouse(MouseEventKind::Drag(left), 9, 2)));
    assert!(!app.handle_terminal_event(&mouse(MouseEventKind::Up(left), 9, 2)));

    // The outcome replaces the key hints until the next key.
    app.copy_status = Some(CopyStatus::Failed(String::from("no clipboard")));
    let status = |app: &App| screen(app).last().cloned().unwrap_or_default();
    assert_eq!(status(&app).trim(), "copy failed: no clipboard");
    app.handle_terminal_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('x'),
        KeyModifiers::NONE,
    )));
    assert!(app.selection.is_none() && app.to_copy.is_none());
    assert!(status(&app).contains("enter send"), "{}", status(&app));
}

#[test]
fn a_click_a_resize_and_a_paste_clear_the_selection() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app_with_two_notes(dir.path());
    let left = MouseButton::Left;
    let select = |app: &mut App| {
        app.handle_terminal_event(&mouse(MouseEventKind::Down(left), 3, 2));
        app.handle_terminal_event(&mouse(MouseEventKind::Drag(left), 7, 2));
        app.handle_terminal_event(&mouse(MouseEventKind::Up(left), 7, 2));
        assert_eq!(app.to_copy.take().as_deref(), Some("alpha"));
        assert!(app.selection.is_some());
    };

    select(&mut app);
    app.handle_terminal_event(&mouse(MouseEventKind::Down(left), 4, 2));
    app.handle_terminal_event(&mouse(MouseEventKind::Up(left), 4, 2));
    assert!(app.selection.is_none() && app.to_copy.is_none());

    select(&mut app);
    app.handle_terminal_event(&Event::Resize(60, 20));
    assert!(app.selection.is_none());

    select(&mut app);
    app.handle_terminal_event(&Event::Paste(String::from("typed")));
    assert!(app.selection.is_none());

    // Only blank cells: nothing to copy, nothing left highlighted.
    app.handle_terminal_event(&mouse(MouseEventKind::Down(left), 20, 3));
    app.handle_terminal_event(&mouse(MouseEventKind::Drag(left), 40, 3));
    app.handle_terminal_event(&mouse(MouseEventKind::Up(left), 40, 3));
    assert!(app.selection.is_none() && app.to_copy.is_none());

    // The other buttons select nothing.
    app.handle_terminal_event(&mouse(MouseEventKind::Down(MouseButton::Right), 3, 2));
    assert!(app.selection.is_none());
}

#[test]
fn a_press_outside_the_transcript_or_under_a_list_selects_nothing() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app_with_two_notes(dir.path());
    let left = MouseButton::Left;
    // The header, then the input.
    for row in [0, 27] {
        app.handle_terminal_event(&mouse(MouseEventKind::Down(left), 5, row));
        assert!(app.selection.is_none(), "row {row}");
    }
    app.picker = Some(Picker::jobs(Vec::new()));
    app.handle_terminal_event(&mouse(MouseEventKind::Down(left), 5, 2));
    assert!(app.selection.is_none());
    app.picker = None;
    app.handle_terminal_event(&mouse(MouseEventKind::Down(left), 5, 2));
    assert!(app.selection.is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_press_while_a_write_waits_for_an_answer_selects_nothing() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app_with_two_notes(dir.path());
    app.handle_slash_command("/sql CREATE TABLE t AS SELECT 1 AS a");
    db_settle(&mut app).await;
    assert!(app.awaiting_permission());
    drop(screen(&app));
    app.handle_terminal_event(&mouse(MouseEventKind::Down(MouseButton::Left), 5, 2));
    assert!(app.selection.is_none());
}

#[test]
fn a_drag_past_an_edge_scrolls_and_the_wheel_still_does() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    app.messages.clear();
    for n in 0..60 {
        app.note(MessageKind::System, format!("line {n}"));
    }
    drop(screen(&app));
    let view = app.view.get();
    let left = MouseButton::Left;

    app.handle_terminal_event(&mouse(MouseEventKind::Down(left), 3, 10));
    app.handle_terminal_event(&mouse(MouseEventKind::Drag(left), 3, 0));
    assert_eq!(app.scroll, Scroll::Back(1), "above the transcript");
    app.handle_terminal_event(&mouse(MouseEventKind::Drag(left), 3, 28));
    assert_eq!(app.scroll, Scroll::Latest, "below it");
    app.handle_terminal_event(&mouse(MouseEventKind::Up(left), 3, 28));
    // From the press to the transcript's end: `line N`, a blank, ...
    let first = view
        .top
        .saturating_add(8)
        .checked_div(2)
        .unwrap_or_default();
    let expected: Vec<String> = (first..60).map(|n| format!("line {n}")).collect();
    assert_eq!(app.to_copy.take(), Some(expected.join("\n\n")));

    app.handle_terminal_event(&mouse(MouseEventKind::ScrollUp, 3, 10));
    assert_eq!(app.scroll, Scroll::Back(3));
    app.handle_terminal_event(&mouse(MouseEventKind::ScrollDown, 3, 10));
    assert_eq!(app.scroll, Scroll::Latest);
    assert_eq!(
        app.view.get().locate(3, 10).map(|l| l.position),
        Some(Position {
            line: view.top.saturating_add(8),
            column: 3
        })
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_event_loop_highlights_a_selection_and_copies_it() {
    use base64::Engine as _;
    use ratatui::backend::TestBackend;
    use ratatui::style::Modifier;

    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let app = app_with_two_notes(dir.path());
    let (events, input) = futures::channel::mpsc::unbounded::<std::io::Result<Event>>();
    let left = MouseButton::Left;
    for event in [
        mouse(MouseEventKind::Down(left), 3, 2),
        mouse(MouseEventKind::Drag(left), 7, 2),
        mouse(MouseEventKind::Up(left), 7, 2),
    ] {
        assert!(events.unbounded_send(Ok(event)).is_ok());
    }
    // The stream's end stops the loop.
    drop(events);
    let mut terminal =
        ratatui::Terminal::new(TestBackend::new(80, 30)).unwrap_or_else(|e| fail(&e.to_string()));
    let mut sent = Vec::new();
    let result = app
        .run_with(&mut terminal, input, Clipboard::terminal_only(&mut sent))
        .await;
    assert!(result.is_ok(), "{result:?}");

    let sequence = String::from_utf8_lossy(&sent);
    let copied = sequence
        .strip_prefix("\u{1b}]52;c;")
        .and_then(|rest| rest.strip_suffix("\u{1b}\\"))
        .and_then(|payload| {
            base64::engine::general_purpose::STANDARD
                .decode(payload)
                .ok()
        });
    assert_eq!(copied.as_deref(), Some(b"alpha".as_slice()), "{sequence}");

    let buffer = terminal.backend().buffer();
    let reversed: String = buffer
        .content()
        .iter()
        .filter(|cell| cell.modifier.contains(Modifier::REVERSED))
        .map(ratatui::buffer::Cell::symbol)
        .collect();
    assert_eq!(reversed, "alpha");
    let status = rows_of(buffer).last().cloned().unwrap_or_default();
    assert_eq!(
        status.trim(),
        "sent 5 characters to the terminal's clipboard"
    );
}

#[test]
fn a_finished_job_reports_one_line() {
    let table = BackgroundResult::Done {
        kind: MessageKind::Sql,
        text: String::from("a\n1\n(1 rows)\n3 ms"),
    };
    assert_eq!(table.outcome(), Ok(String::from("3 ms")));
    let note = BackgroundResult::from(Ok(String::from("Loaded x\nYou can now ask")));
    assert_eq!(note.outcome(), Ok(String::from("Loaded x")));
    let failed = BackgroundResult::from(Err(anyhow!("outer").context("while loading")));
    assert_eq!(failed.outcome(), Err(String::from("while loading: outer")));
}

#[test]
fn clear_transcript_clears_the_wrap_cache_so_a_replay_renders_fresh() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut app = app(dir.path());
    app.messages
        .push(Message::new(MessageKind::Assistant, String::from("cached")));
    drop(ui::format_messages(&app, 60));
    assert!(!app.wrap_cache.borrow().is_empty());
    app.clear_transcript();
    assert!(
        app.wrap_cache.borrow().is_empty(),
        "clear_transcript must reset the render cache so a repopulation cannot reuse stale slots",
    );
}
