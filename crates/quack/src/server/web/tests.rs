use quack_core::analysis::events::ToolName;
use quack_core::ids::{MessageId, WorkspaceId};
use quack_core::storage::control::{AllowedProviders, WorkspaceRow};
use quack_core::storage::sessions::{AssistantMeta, MessageMeta, ToolMeta};

use super::*;

/// A stale graph whose preview could not be counted still gets its
/// page: the banner carries the reason and offers no drop.
#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn a_failed_revalidation_preview_is_shown_in_the_stale_banner() {
    let page = GraphPage {
        page: Page {
            title: String::from("Graph"),
            tab: Tab::Graph,
            username: String::from("ada"),
            kind: UserKind::Standard,
            local: false,
            workspace: Some(WsNav {
                membership: Membership {
                    workspace: WorkspaceRow {
                        id: WorkspaceId::from("w1"),
                        name: String::from("sales"),
                        classification: String::new(),
                        allowed_providers: AllowedProviders::All,
                    },
                    standing: Standing::Member(Role::Owner),
                },
                can_write: true,
                can_manage: true,
            }),
        },
        status: GraphStatus {
            nodes: 7,
            stale: true,
            ..GraphStatus::default()
        },
        drift: Vec::new(),
        has_ontology: true,
        revalidation: Some(Err(String::from("no ontology to validate against"))),
        merges: Vec::new(),
        query: GraphQueryView::default(),
        result: None,
        error: None,
        notice: None,
    };
    let html = page.render().unwrap();
    assert!(html.contains("7 nodes"), "{html}");
    assert!(
        html.contains(
            "What revalidating would drop could not be counted: no ontology to validate against"
        ),
        "{html}"
    );
    assert!(!html.contains("graph/revalidate"), "{html}");
}

#[test]
fn cells_render_strings_bare_and_null_empty() {
    let text = |v: &serde_json::Value| JsonText(v).to_string();
    assert_eq!(text(&serde_json::json!("s")), "s");
    assert_eq!(text(&serde_json::Value::Null), "");
    assert_eq!(text(&serde_json::json!(4.5)), "4.5");
}

/// A reloaded session shows each answer with the steps recorded before
/// it, and a tool row whose metadata did not decode is left out.
#[test]
fn transcript_folds_tool_rows_into_the_answer_after_them() {
    let row =
        |seq: i64, role: MessageRole, content: &str, metadata: Option<MessageMeta>| MessageRow {
            id: MessageId::from(format!("m{seq}")),
            session_id: SessionId::from("s"),
            seq,
            role,
            content: content.to_owned(),
            metadata,
            created_at: String::new(),
        };
    let tool = |detail: &str| {
        Some(MessageMeta::Tool(ToolMeta {
            result: None,
            tool: ToolName::RunSql,
            detail: detail.to_owned(),
            duration_ms: 3,
            rows: Some(1),
        }))
    };
    let rows = vec![
        row(1, MessageRole::User, "how many?", None),
        row(2, MessageRole::Tool, "1 rows", tool("SELECT 1")),
        row(3, MessageRole::Tool, "lost", None),
        row(4, MessageRole::Tool, "1 rows", tool("SELECT 2")),
        row(
            5,
            MessageRole::Assistant,
            "Two.",
            Some(MessageMeta::Assistant(AssistantMeta {
                write_refused: true,
                ..AssistantMeta::default()
            })),
        ),
        row(6, MessageRole::User, "and now?", None),
        row(7, MessageRole::Assistant, "Same.", None),
    ];
    let views = MessageView::transcript(&rows);
    let shape: Vec<(&str, Vec<&str>)> = views
        .iter()
        .map(|v| {
            (
                v.role.as_str(),
                v.steps.iter().map(|s| s.detail.as_str()).collect(),
            )
        })
        .collect();
    assert_eq!(
        shape,
        vec![
            ("user", vec![]),
            ("assistant", vec!["SELECT 1", "SELECT 2"]),
            ("user", vec![]),
            ("assistant", vec![]),
        ]
    );
}

/// A stored `TIMESTAMP` is UTC text; the page gets an instant the
/// browser can localize, and text it cannot parse shows no time at all.
#[test]
fn moments_read_duckdb_timestamps_as_utc() {
    let at = Moment::from_utc_text("2026-09-30 14:03:22.123456");
    assert_eq!(
        at.as_ref().map(Moment::iso).as_deref(),
        Some("2026-09-30T14:03:22.123456Z")
    );
    assert_eq!(
        at.as_ref().map(Moment::utc).as_deref(),
        Some("2026-09-30 14:03 UTC")
    );
    assert_eq!(
        Moment::from_utc_text("2026-09-30 14:03:22")
            .map(|m| m.iso())
            .as_deref(),
        Some("2026-09-30T14:03:22Z")
    );
    assert!(Moment::from_utc_text("").is_none());
    assert!(Moment::from_utc_text("yesterday").is_none());
    // control.db keeps SQLite's CURRENT_TIMESTAMP text and RFC 3339 expiries.
    assert_eq!(
        Moment::from_utc_text("2026-10-01T02:37:57Z")
            .map(|m| m.iso())
            .as_deref(),
        Some("2026-10-01T02:37:57Z")
    );
}

/// A stored time becomes a `<time>` element the browser localizes; text
/// that is not a time is shown as it is, escaped.
#[test]
fn when_renders_time_elements_and_escapes_anything_else() {
    assert_eq!(
        When::Relative.html("2020-01-01 02:37:57"),
        "<time datetime=\"2020-01-01T02:37:57Z\" data-when=\"relative\">2020-01-01</time>"
    );
    let now: Timestamp = "2026-10-01T12:00:00Z"
        .parse()
        .unwrap_or(Timestamp::UNIX_EPOCH);
    let ago = |at: &str| {
        Moment::from_utc_text(at)
            .map(|m| m.ago(now))
            .unwrap_or_default()
    };
    assert_eq!(ago("2026-10-01 11:59:30"), "just now");
    assert_eq!(ago("2026-10-01 11:55:00"), "5 min ago");
    assert_eq!(ago("2026-10-01 09:00:00"), "3 h ago");
    assert_eq!(ago("2026-09-29 12:00:00"), "2026-09-29");
    assert!(
        When::Clock
            .html("2026-10-01 02:37:57")
            .contains("data-when=\"clock\"")
    );
    assert_eq!(
        When::Clock.html("<b>soon</b>"),
        "&#60;b&#62;soon&#60;/b&#62;"
    );
}

/// An answer shows how long it took; a question and an answer recorded
/// before durations were kept show none.
#[test]
fn transcript_carries_the_answer_duration() {
    let row = |seq: i64, role: MessageRole, metadata: Option<MessageMeta>| MessageRow {
        id: MessageId::from(format!("m{seq}")),
        session_id: SessionId::from("s"),
        seq,
        role,
        content: String::from("x"),
        metadata,
        created_at: String::from("2026-09-30 14:03:22"),
    };
    let timed = Some(MessageMeta::Assistant(AssistantMeta {
        duration_ms: Some(2_345),
        ..AssistantMeta::default()
    }));
    let views = MessageView::transcript(&[
        row(1, MessageRole::User, None),
        row(2, MessageRole::Assistant, timed),
        row(3, MessageRole::User, None),
        row(4, MessageRole::Assistant, None),
    ]);
    let durations: Vec<Option<u64>> = views.iter().map(|v| v.duration_ms).collect();
    assert_eq!(durations, vec![None, Some(2_345), None, None]);
    assert!(views.iter().all(|v| v.at.is_some()));
}
