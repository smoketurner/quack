//! `quack ingest FILE --pin` on a single file: the binary itself, run over
//! a file that is new, and again over one whose bytes are already stored.

use std::path::Path;
use std::process::{Command, Output, Stdio};

/// No model at all: the file loads without embeddings.
const CONFIG: &str = "[general]\n";

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

fn write(root: &Path, path: &str, text: &str) {
    let path = root.join(path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| fail(&e.to_string()));
    }
    std::fs::write(path, text).unwrap_or_else(|e| fail(&e.to_string()));
}

/// The command's stdout and stderr as text, after asserting its exit.
fn ran(output: &Output, succeeded: bool) -> (String, String) {
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert_eq!(
        output.status.success(),
        succeeded,
        "exit {:?}\n{stdout}\n{stderr}",
        output.status.code()
    );
    (stdout, stderr)
}

/// A duplicate ingest with `--pin` pins the existing document and says so,
/// where a duplicate without `--pin` leaves it as it was.
#[test]
fn a_duplicate_ingest_with_pin_pins_the_existing_document() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let root = dir.path();
    write(root, "config/config.toml", CONFIG);
    let file = root.join("policy.md");
    write(root, "policy.md", "# Policy\n\nFlood is excluded.\n");
    let file_arg = file.display().to_string();

    // First ingest without --pin: the document is stored, unpinned.
    let (stdout, _) = ran(&quack(root, &["ingest", &file_arg, "--no-embed"]), true);
    assert!(stdout.contains("Ingested:"), "{stdout}");
    let (docs, _) = ran(&quack(root, &["docs"]), true);
    assert!(docs.contains("policy.md"), "{docs}");
    assert!(
        !docs.contains("pinned"),
        "the document should not be pinned yet: {docs}"
    );

    // Re-ingest the same bytes with --pin: skipped, but the existing document
    // is pinned and the message says so.
    let (stdout, _) = ran(
        &quack(root, &["ingest", &file_arg, "--no-embed", "--pin"]),
        true,
    );
    assert!(
        stdout.contains("Skipped:") && stdout.contains("pinned the existing document"),
        "{stdout}"
    );
    let (docs, _) = ran(&quack(root, &["docs"]), true);
    assert!(
        docs.contains("pinned"),
        "the document should now be pinned: {docs}"
    );

    // A duplicate without --pin does not unpin: the document stays pinned.
    let (stdout, _) = ran(&quack(root, &["ingest", &file_arg, "--no-embed"]), true);
    assert!(
        stdout.contains("Skipped:") && !stdout.contains("pinned"),
        "no pinning suffix without --pin: {stdout}"
    );
    let (docs, _) = ran(&quack(root, &["docs"]), true);
    assert!(
        docs.contains("pinned"),
        "the document should still be pinned: {docs}"
    );
}

/// A duplicate ingest without `--pin` leaves an unpinned document unpinned.
#[test]
fn a_duplicate_ingest_without_pin_does_not_pin() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let root = dir.path();
    write(root, "config/config.toml", CONFIG);
    let file = root.join("notes.md");
    write(root, "notes.md", "# Notes\n\nSome notes.\n");
    let file_arg = file.display().to_string();

    // First ingest without --pin: the document is stored, unpinned.
    let (stdout, _) = ran(&quack(root, &["ingest", &file_arg, "--no-embed"]), true);
    assert!(stdout.contains("Ingested:"), "{stdout}");

    // Re-ingest the same bytes without --pin: skipped, not pinned.
    let (stdout, _) = ran(&quack(root, &["ingest", &file_arg, "--no-embed"]), true);
    assert!(
        stdout.contains("Skipped:") && !stdout.contains("pinned"),
        "no pinning without --pin: {stdout}"
    );
    let (docs, _) = ran(&quack(root, &["docs"]), true);
    assert!(
        !docs.contains("pinned"),
        "the document should not be pinned: {docs}"
    );
}

/// A fresh ingest with `--pin` pins the new document, as it always has.
#[test]
fn a_fresh_ingest_with_pin_pins_the_new_document() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let root = dir.path();
    write(root, "config/config.toml", CONFIG);
    let file = root.join("fresh.md");
    write(root, "fresh.md", "# Fresh\n\nA new reference document.\n");
    let file_arg = file.display().to_string();

    let (stdout, _) = ran(
        &quack(root, &["ingest", &file_arg, "--no-embed", "--pin"]),
        true,
    );
    assert!(stdout.contains("Ingested:"), "{stdout}");
    assert!(stdout.contains("Pinned: yes"), "{stdout}");
    let (docs, _) = ran(&quack(root, &["docs"]), true);
    assert!(
        docs.contains("pinned"),
        "the new document should be pinned: {docs}"
    );
}

/// A duplicate ingest writes the fields it was given onto the stored
/// document instead of dropping them.
#[test]
fn a_duplicate_ingest_writes_its_given_fields() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let root = dir.path();
    write(root, "config/config.toml", CONFIG);
    write(root, "notes.md", "# Notes\n\nQuarterly figures.\n");
    let file_arg = root.join("notes.md").display().to_string();
    ran(&quack(root, &["ingest", &file_arg, "--no-embed"]), true);
    let (stdout, _) = ran(
        &quack(
            root,
            &[
                "ingest",
                &file_arg,
                "--no-embed",
                "--title",
                "Q3 notes",
                "--author",
                "Ada",
                "--tag",
                "finance",
            ],
        ),
        true,
    );
    assert!(stdout.contains("Skipped:"), "{stdout}");
    let (listing, _) = ran(&quack(root, &["docs", "--format", "json"]), true);
    for given in ["Q3 notes", "Ada", "finance"] {
        assert!(listing.contains(given), "{given} missing from {listing}");
    }
}
