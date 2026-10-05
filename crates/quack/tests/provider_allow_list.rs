//! The CLI under a workspace's provider allow-list: the binary itself, run
//! against a data directory whose workspace is restricted.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::{Command, Output, Stdio};

use quack_core::config::Config;
use quack_core::storage::control::{
    AuditAction, AuditEntry, Channel, ControlPlane, Outcome, ProviderAllowList, WorkspaceChanges,
};

/// The chat model is on `local` and the embedding model on `hosted`; both
/// are unreachable, so a request that got past the check fails differently.
const CONFIG: &str = "[general]\nchat_model = \"local/chat\"\n\
    [embedding]\nmodel = \"hosted/embed\"\ndimension = 768\n\
    [providers.local]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\n\
    [providers.hosted]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\n";

const REFUSAL: &str =
    "provider 'hosted' is not allowed in this workspace, which allows only: local";

fn quack(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_quack"))
        .args(args)
        .env("QUACK_CONFIG_DIR", dir.join("config"))
        .env("QUACK_DATA_DIR", dir.join("data"))
        .env_remove("QUACK_MODEL")
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|e| fail(&format!("quack did not run: {e}")))
}

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

#[tokio::test]
async fn a_restricted_workspace_refuses_print_mode_and_ingest_on_a_disallowed_provider() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let root = dir.path();
    std::fs::create_dir_all(root.join("config")).unwrap_or_else(|e| fail(&e.to_string()));
    std::fs::write(root.join("config/config.toml"), CONFIG)
        .unwrap_or_else(|e| fail(&e.to_string()));
    let notes = root.join("notes.md");
    std::fs::write(&notes, "# Policy\n\nFlood damage is excluded.\n")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let notes = notes.display().to_string();

    let mut config = Config::parse(CONFIG).unwrap_or_else(|e| fail(&e.to_string()));
    config.general.data_dir = root.join("data");
    let control = ControlPlane::open(&config)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let setup = || AuditEntry::new(AuditAction::Admin, Outcome::Allowed, Channel::Cli);
    let create = async |name: &str| {
        let name = name.parse().unwrap_or_else(|_| fail("workspace name"));
        control
            .create_workspace(&name, None, setup())
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
    };
    create("open").await;
    let kept = create("kept").await;
    control
        .update_workspace(
            &kept.id,
            &WorkspaceChanges {
                classification: None,
                allowed_providers: ProviderAllowList::Only(BTreeSet::from([String::from("local")])),
            },
        )
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    drop(control);

    for args in [
        vec!["-w", "kept", "ingest", notes.as_str()],
        vec!["-w", "kept", "-p", "what is excluded?"],
    ] {
        let run = quack(root, &args);
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert_eq!(run.status.code(), Some(1), "{args:?}: {stderr}");
        assert!(stderr.contains(REFUSAL), "{args:?}: {stderr}");
    }
    // The refused ingest registered nothing.
    let docs = quack(root, &["-w", "kept", "docs"]);
    assert!(docs.status.success(), "{docs:?}");
    assert!(
        !String::from_utf8_lossy(&docs.stdout).contains("notes.md"),
        "{docs:?}"
    );

    // A workspace that allows every provider gets as far as the provider,
    // which is unreachable here: a failure, not a refusal.
    let open = quack(root, &["-w", "open", "ingest", notes.as_str()]);
    let stderr = String::from_utf8_lossy(&open.stderr);
    assert!(!open.status.success(), "{stderr}");
    assert!(!stderr.contains("is not allowed"), "{stderr}");
}
