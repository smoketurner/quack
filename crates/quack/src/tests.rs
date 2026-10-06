use super::*;
use quack_core::embedding::Dimension;
use quack_core::error::AuthReason;
use quack_core::ingestion::parser::PageCounts;
use quack_core::llm::oauth::Renewal;
use quack_core::storage::workspace::{DocumentStatus, NewDocument};

/// A closed reader surfaces as an `io::Error`, a `serde_json` or `csv`
/// error, or core's transparent `Io`, `Json`, and `Csv` variants; each one ends the
/// command quietly (issue #68). Anything else still reports.
#[test]
fn broken_pipe_is_recognised_through_every_wrapper() {
    let pipe = || std::io::Error::from(std::io::ErrorKind::BrokenPipe);
    assert!(is_broken_pipe(&anyhow::Error::from(pipe())));
    assert!(is_broken_pipe(
        &anyhow::Error::from(pipe()).context("failed to print")
    ));
    assert!(is_broken_pipe(&anyhow::Error::from(CoreError::Io(pipe()))));
    assert!(is_broken_pipe(&anyhow::Error::from(CoreError::Json(
        serde_json::Error::io(pipe())
    ))));
    assert!(is_broken_pipe(&anyhow::Error::from(csv::Error::from(
        pipe()
    ))));
    assert!(is_broken_pipe(&anyhow::Error::from(CoreError::Csv(
        csv::Error::from(pipe())
    ))));
    assert!(!is_broken_pipe(&anyhow::Error::from(std::io::Error::from(
        std::io::ErrorKind::NotFound
    ))));
    assert!(!is_broken_pipe(&anyhow::anyhow!("something else")));
}

/// `docs --format json` prints every recorded field, so a script can tell why a
/// document failed.
#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn docs_json_carries_every_document_field() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    db.insert_document(&NewDocument::new(
        &DocumentId::from("d1"),
        "broken.pdf",
        "application/pdf",
        3,
    ))
    .unwrap();
    db.mark_document_error(&DocumentId::from("d1"), "no text layer")
        .unwrap();
    let mut out = Vec::new();
    list_documents(&db, TextOrJson::Json, Shown::Live, &mut out).unwrap();
    let row: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(row.get("status").unwrap(), "error", "{row}");
    assert_eq!(row.get("error_message").unwrap(), "no text layer", "{row}");
    for key in [
        "ingested_by",
        "tables",
        "filename",
        "sha256",
        "source",
        "pages",
        "superseded_by",
        "source_path",
    ] {
        assert!(row.get(key).is_some(), "{key} missing: {row}");
    }
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn docs_show_the_pages_missing_from_a_document() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    let id = DocumentId::from("d1");
    db.insert_document(&NewDocument::new(&id, "scan.pdf", "application/pdf", 3))
        .unwrap();
    db.set_document_pages(
        &id,
        Some(PageCounts {
            total: 40,
            unreadable: 3,
            empty: 2,
        }),
    )
    .unwrap();
    let mut out = Vec::new();
    list_documents(&db, TextOrJson::Json, Shown::Live, &mut out).unwrap();
    let row: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(
        row.get("pages").unwrap(),
        &serde_json::json!({ "total": 40, "unreadable": 3, "empty": 2 }),
        "{row}"
    );
    let mut out = Vec::new();
    list_documents(&db, TextOrJson::Text, Shown::Live, &mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(
        text.contains("scan.pdf  [3 of 40 pages unreadable, 2 without text]"),
        "{text}"
    );
}

/// `docs` lists what the workspace holds now; `--all` adds replaced
/// documents, each naming the one that took its place.
#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn docs_hide_replaced_documents_unless_asked() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    let (old, new) = (DocumentId::from("d1"), DocumentId::from("d2"));
    db.insert_document(
        &NewDocument::new(&old, "policy.md", "text/markdown", 3).with_status(DocumentStatus::Ready),
    )
    .unwrap();
    db.insert_document(
        &NewDocument::new(&new, "policy.md", "text/markdown", 4).with_status(DocumentStatus::Ready),
    )
    .unwrap();
    db.begin_replacement(&old, &new).unwrap();
    assert_eq!(db.finish_replacement(&new).unwrap(), Some(old));
    let mut out = Vec::new();
    list_documents(&db, TextOrJson::Text, Shown::Live, &mut out).unwrap();
    let live = String::from_utf8(out).unwrap();
    assert!(
        live.contains("d2  ready") && !live.contains("d1  "),
        "{live}"
    );
    let mut out = Vec::new();
    list_documents(&db, TextOrJson::Text, Shown::All, &mut out).unwrap();
    let all = String::from_utf8(out).unwrap();
    assert!(all.contains("d1  superseded"), "{all}");
    assert!(all.contains("policy.md  -> d2"), "{all}");
    assert!(all.contains("d2  ready"), "{all}");
}

/// `--replace` alone means the document with the file's name; with a
/// value, that id.
#[test]
fn replace_flag_parses_with_and_without_an_id() {
    assert_eq!("".parse(), Ok(Replace::SameName));
    assert_eq!(" ".parse(), Ok(Replace::SameName));
    assert_eq!("0199".parse(), Ok(Replace::Document(String::from("0199"))));
}

/// A missing login is recognised through the context a command adds
/// (`import failed`), but not once the error has been turned into text,
/// which is how `quack import` lost its exit 4.
#[test]
fn auth_required_is_recognised_through_context_only_while_typed() {
    let auth = || CoreError::AuthRequired {
        provider: String::from("corp"),
        reason: AuthReason::NoToken,
    };
    let exit = Some(Exit::AuthRequired);
    assert_eq!(Exit::of(&anyhow::Error::from(auth())), exit);
    assert_eq!(
        Exit::of(&anyhow::Error::from(auth()).context("import failed")),
        exit
    );
    assert_eq!(Exit::of(&anyhow::anyhow!(auth().to_string())), None);
    assert_eq!(Exit::of(&anyhow::anyhow!("something else")), None);
}

/// The exit statuses scripts check, each its own number; a changed
/// saved question is the one that is not an error.
#[test]
fn exit_statuses_are_distinct_and_a_change_is_five() {
    assert_eq!(ExitCode::from(Exit::Usage), ExitCode::from(2));
    assert_eq!(ExitCode::from(Exit::WriteRefused), ExitCode::from(3));
    assert_eq!(ExitCode::from(Exit::AuthRequired), ExitCode::from(4));
    assert_eq!(ExitCode::from(Exit::Changed), ExitCode::from(5));
}

fn config_in(dir: &Path) -> Config {
    let mut config = Config::default();
    config.general.data_dir = dir.join("data");
    config
}

/// `-w` with a name no workspace has is a usage error that says how to
/// create it, and it creates nothing.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn an_unknown_workspace_is_a_usage_error_and_creates_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let config = config_in(dir.path());
    let refused = OpenedWorkspace::in_config(config.clone(), Some("slaes"))
        .await
        .err()
        .unwrap();
    assert_eq!(
        format!("{refused:#}"),
        "no workspace named 'slaes'; create it with: quack workspace create slaes"
    );
    assert_eq!(Exit::of(&refused), Some(Exit::Usage));
    let control = ControlPlane::open(&config).await.unwrap();
    assert!(control.list_workspaces().await.unwrap().is_empty());
}

/// A workspace made with `quack workspace create` is the one `-w` then
/// opens, and creating it twice is refused.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn a_created_workspace_opens_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let config = config_in(dir.path());
    let create = || admin::WorkspaceAction::Create {
        name: "sales".parse().unwrap(),
    };
    admin::run_workspace(&config, create()).await.unwrap();
    let opened = OpenedWorkspace::in_config(config.clone(), Some("sales"))
        .await
        .unwrap();
    assert_eq!(opened.name, "sales");
    opened.open_db().unwrap();

    let again = admin::run_workspace(&config, create()).await.unwrap_err();
    assert_eq!(again.to_string(), "workspace 'sales' already exists");
    let control = ControlPlane::open(&config).await.unwrap();
    assert_eq!(control.list_workspaces().await.unwrap().len(), 1);
}

/// With no `-w`, a new data directory gets the default workspace.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn no_workspace_flag_creates_the_default_on_a_fresh_data_dir() {
    let dir = tempfile::tempdir().unwrap();
    let config = config_in(dir.path());
    let opened = OpenedWorkspace::in_config(config.clone(), None)
        .await
        .unwrap();
    assert_eq!(opened.name, "default");
    let again = OpenedWorkspace::in_config(config, Some("default"))
        .await
        .unwrap();
    assert_eq!(again.workspace.id, opened.workspace.id);
}

#[test]
fn auth_status_names_the_actor_only_when_one_is_sent() {
    let token = TokenStatus {
        expires_at: jiff::Timestamp::UNIX_EPOCH,
        renewal: Renewal::Regrant,
    };
    let with_actor = token_state("gw", Some(token), Grant::OnBehalfOf, true);
    assert!(with_actor.contains("(the actor)"), "{with_actor}");
    for stored in [Some(token), None] {
        let vouch = token_state("gw", stored, Grant::OnBehalfOf, false);
        assert!(!vouch.contains("the actor"), "{vouch}");
        assert!(vouch.contains("without an actor token"), "{vouch}");
    }
    assert!(token_state("p", None, Grant::AuthorizationCode, false).contains("quack auth login p"));
}
