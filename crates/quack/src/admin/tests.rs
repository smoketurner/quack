use super::*;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use quack_core::error::Error as CoreError;

#[derive(clap::Parser)]
#[command(no_binary_name = true)]
enum Line {
    #[command(subcommand)]
    Workspace(WorkspaceAction),
    #[command(subcommand)]
    User(UserAction),
    #[command(subcommand)]
    Token(TokenAction),
    #[command(subcommand)]
    Member(MemberAction),
    Audit(AuditArgs),
}

/// Roles, scopes, and outcomes are checked by clap as they are typed,
/// with the values each accepts.
#[test]
fn roles_scopes_and_outcomes_parse_at_the_command_line() {
    let parse = |args: &[&str]| <Line as clap::Parser>::try_parse_from(args);
    assert!(matches!(
        parse(&["member", "add", "ann", "--role", "Owner"]),
        Ok(Line::Member(MemberAction::Add {
            role: Role::Owner,
            ..
        }))
    ));
    assert!(matches!(
        parse(&["token", "create", "--user", "u", "--name", "n", "--scopes", "read,write"]),
        Ok(Line::Token(TokenAction::Create { scopes, .. }))
            if scopes == [Scope::Read, Scope::Write]
    ));
    assert!(matches!(
        parse(&["audit", "--outcome", "denied"]),
        Ok(Line::Audit(AuditArgs {
            outcome: Some(Outcome::Denied),
            ..
        }))
    ));
    let refused = parse(&["member", "add", "ann", "--role", "boss"])
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        refused.contains("use one of: viewer, member, owner"),
        "{refused}"
    );
    assert!(
        parse(&[
            "token", "create", "--user", "u", "--name", "n", "--scopes", "root"
        ])
        .is_err()
    );
    assert!(parse(&["audit", "--outcome", "maybe"]).is_err());
}

/// `workspace create` takes only a name the API would accept, and
/// refuses any other with the same message.
#[test]
fn workspace_create_refuses_a_name_the_api_would() {
    let parse = |args: &[&str]| <Line as clap::Parser>::try_parse_from(args);
    assert!(matches!(
        parse(&["workspace", "create", "sales"]),
        Ok(Line::Workspace(WorkspaceAction::Create { name })) if name.as_str() == "sales"
    ));
    assert!(matches!(
        parse(&["workspace", "rename", "sales", "sales-2025"]),
        Ok(Line::Workspace(WorkspaceAction::Rename { name, new_name }))
            if name == "sales" && new_name.as_str() == "sales-2025"
    ));
    assert!(matches!(
        parse(&["workspace", "delete", "sales", "-y"]),
        Ok(Line::Workspace(WorkspaceAction::Delete { name, yes: true })) if name == "sales"
    ));
    assert!(matches!(
        parse(&["workspace", "snapshot", "sales", "--to", "s.tar"]),
        Ok(Line::Workspace(WorkspaceAction::Snapshot { name, to: Some(to) }))
            if name == "sales" && to.as_os_str() == "s.tar"
    ));
    assert!(matches!(
        parse(&["workspace", "restore", "-", "--name", "copy"]),
        Ok(Line::Workspace(WorkspaceAction::Restore { file, name: Some(name) }))
            if file.as_os_str() == "-" && name.as_str() == "copy"
    ));
    assert!(matches!(
        parse(&["user", "disable", "bob"]),
        Ok(Line::User(UserAction::Disable { username })) if username == "bob"
    ));
    assert!(matches!(
        parse(&["user", "admin", "bob", "--off"]),
        Ok(Line::User(UserAction::Admin { username, off: true })) if username == "bob"
    ));
    assert!(matches!(
        parse(&["user", "remove", "bob", "-y"]),
        Ok(Line::User(UserAction::Remove { username, yes: true })) if username == "bob"
    ));
    assert!(matches!(
        parse(&["member", "add", "--group", "finance", "--role", "owner"]),
        Ok(Line::Member(MemberAction::Add { username: None, group: Some(g), role: Role::Owner }))
            if g == "finance"
    ));
    assert!(matches!(
        parse(&["member", "add", "bob"]),
        Ok(Line::Member(MemberAction::Add { username: Some(u), group: None, role: Role::Member }))
            if u == "bob"
    ));
    assert!(parse(&["member", "add"]).is_err());
    assert!(parse(&["member", "add", "bob", "--group", "finance"]).is_err());
    assert!(matches!(
        parse(&["member", "remove", "--group", "finance"]),
        Ok(Line::Member(MemberAction::Remove { username: None, group: Some(g) })) if g == "finance"
    ));
    for refused in ["a.b/c", "a\\b", ""] {
        let message = parse(&["workspace", "create", refused])
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            message.contains(&CoreError::InvalidWorkspaceName.to_string()),
            "{refused:?}: {message}"
        );
    }
}

// --- read_hidden_line: control-key handling --------------------------------

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}");
}

/// A single key press with the given modifiers. `KeyEvent::new` defaults to
/// `KeyEventKind::Press`, which is what crossterm reports on Unix without
/// keyboard-enhancement flags — the regime `read_password` runs in.
fn key(code: KeyCode, modifiers: KeyModifiers) -> Event {
    Event::Key(KeyEvent::new(code, modifiers))
}

/// Plain typing: one `Char(c)` press with no modifiers per character.
fn typed(s: &str) -> Vec<Event> {
    s.chars()
        .map(|c| key(KeyCode::Char(c), KeyModifiers::NONE))
        .collect()
}

/// Drive `read_hidden_line_from` with `events` in order, until Enter returns
/// or Ctrl-C bails. Exhausting the script without Enter returns an
/// `UnexpectedEof` error, so a forgotten terminator can never hang a test.
fn drive(events: &[Event]) -> Result<String> {
    let mut idx = 0_usize;
    let len = events.len();
    let next = || {
        let i = idx;
        idx = i.saturating_add(1);
        events.get(i).cloned().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("scripted events exhausted at index {i} of {len}"),
            )
        })
    };
    read_hidden_line_from(next)
}

/// The happy path: a typed password is assembled verbatim and returned on
/// Enter.
#[test]
fn read_hidden_line_assembles_a_plain_password_verbatim() {
    let mut events = typed("hunter2");
    events.push(key(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(
        drive(&events).unwrap_or_else(|e| fail(&e.to_string())),
        "hunter2"
    );
}

/// The reported bug: a Ctrl-D inserted while typing must not append `'d'`
/// to the password (crossterm 0.29 parses `0x04` as `Char('d')` + CONTROL).
#[test]
fn ctrl_d_inserted_while_typing_is_ignored() {
    let mut events = typed("hunter2");
    events.push(key(KeyCode::Char('d'), KeyModifiers::CONTROL));
    events.push(key(KeyCode::Enter, KeyModifiers::NONE));
    // Before the fix this returned "hunter2d".
    assert_eq!(
        drive(&events).unwrap_or_else(|e| fail(&e.to_string())),
        "hunter2"
    );
}

/// Non-letter control bytes — Ctrl-Space (`0x00` -> `Char(' ')` + CONTROL)
/// and Ctrl-4 (`0x1C` -> `Char('4')` + CONTROL) — also arrive as `Char(_)`
/// with CONTROL and must be ignored.
#[test]
fn ctrl_space_and_ctrl_4_are_ignored() {
    let mut events = typed("pw");
    events.push(key(KeyCode::Char(' '), KeyModifiers::CONTROL));
    events.push(key(KeyCode::Char('4'), KeyModifiers::CONTROL));
    events.push(key(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(
        drive(&events).unwrap_or_else(|e| fail(&e.to_string())),
        "pw"
    );
}

/// Ctrl-C still cancels the prompt, surfacing the `cancelled` error that
/// `read_password` propagates after restoring the cooked terminal mode.
#[test]
fn ctrl_c_cancels_with_the_cancelled_message() {
    let events = vec![
        key(KeyCode::Char('a'), KeyModifiers::NONE),
        key(KeyCode::Char('c'), KeyModifiers::CONTROL),
    ];
    let err = drive(&events)
        .err()
        .unwrap_or_else(|| fail("expected Ctrl-C to cancel the read"));
    assert!(err.to_string().contains("cancelled"), "{err}");
}

/// A literal lowercase `'c'` typed without Control is still appended; only
/// the Ctrl-C combo cancels, so the new guard must not swallow plain 'c'.
#[test]
fn a_plain_c_without_control_is_appended() {
    let events = vec![
        key(KeyCode::Char('a'), KeyModifiers::NONE),
        key(KeyCode::Char('c'), KeyModifiers::NONE),
        key(KeyCode::Char('c'), KeyModifiers::NONE),
        key(KeyCode::Enter, KeyModifiers::NONE),
    ];
    assert_eq!(
        drive(&events).unwrap_or_else(|e| fail(&e.to_string())),
        "acc"
    );
}

/// `AltGr` on Windows is CONTROL|ALT, and it types characters such as
/// `@` and `€` on many layouts: those are kept, not taken for Ctrl combos.
#[test]
fn altgr_characters_are_kept() {
    let altgr = KeyModifiers::CONTROL | KeyModifiers::ALT;
    let events = vec![
        key(KeyCode::Char('a'), KeyModifiers::NONE),
        key(KeyCode::Char('@'), altgr),
        key(KeyCode::Char('€'), altgr),
        key(KeyCode::Char('{'), altgr),
        key(KeyCode::Enter, KeyModifiers::NONE),
    ];
    assert_eq!(
        drive(&events).unwrap_or_else(|e| fail(&e.to_string())),
        "a@€{"
    );
}

/// Windows reports key releases too: only presses type, so a character
/// or a backspace does not happen twice.
#[test]
fn key_releases_are_ignored() {
    let release = |code| {
        Event::Key(KeyEvent::new_with_kind(
            code,
            KeyModifiers::NONE,
            KeyEventKind::Release,
        ))
    };
    let events = vec![
        key(KeyCode::Char('a'), KeyModifiers::NONE),
        release(KeyCode::Char('a')),
        key(KeyCode::Char('b'), KeyModifiers::NONE),
        release(KeyCode::Char('b')),
        key(KeyCode::Backspace, KeyModifiers::NONE),
        release(KeyCode::Backspace),
        key(KeyCode::Char('c'), KeyModifiers::NONE),
        release(KeyCode::Char('c')),
        key(KeyCode::Enter, KeyModifiers::NONE),
    ];
    assert_eq!(
        drive(&events).unwrap_or_else(|e| fail(&e.to_string())),
        "ac"
    );
}

/// Regression guard: filtering must not eat `Shift`. crossterm pairs an
/// uppercase `Char` with `KeyModifiers::SHIFT`, so capital letters typed
/// with `Shift` are still appended verbatim.
#[test]
fn uppercase_typed_with_shift_is_kept() {
    let events = vec![
        key(KeyCode::Char('H'), KeyModifiers::SHIFT),
        key(KeyCode::Char('i'), KeyModifiers::NONE),
        key(KeyCode::Enter, KeyModifiers::NONE),
    ];
    assert_eq!(
        drive(&events).unwrap_or_else(|e| fail(&e.to_string())),
        "Hi"
    );
}
