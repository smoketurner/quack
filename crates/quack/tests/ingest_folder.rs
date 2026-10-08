//! `quack ingest DIR` on a folder of files: the binary itself, run twice
//! over a folder that changes between runs.

use std::path::Path;
use std::process::{Command, Output, Stdio};

/// No model at all: the folder loads without embeddings.
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

#[test]
fn a_folder_is_ingested_file_by_file_and_a_rerun_reports_what_changed() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let root = dir.path();
    write(root, "config/config.toml", CONFIG);
    let folder = root.join("contracts");
    write(&folder, "a.md", "# A\n\nFlood is excluded.\n");
    write(&folder, "rates/b.csv", "region,total\nnorth,1\n");
    write(&folder, "c.xyz", "?");
    write(&folder, ".drafts/d.md", "hidden");
    let folder_arg = folder.display().to_string();

    let (stdout, _) = ran(&quack(root, &["ingest", &folder_arg, "--no-embed"]), true);
    assert!(stdout.contains("ingested  a.md ("), "{stdout}");
    assert!(stdout.contains("ingested  rates/b.csv ("), "{stdout}");
    assert!(
        stdout.contains("Not ingested (1 unsupported): c.xyz"),
        "{stdout}"
    );
    assert!(
        !stdout.contains("d.md") && !stdout.contains("Gone"),
        "{stdout}"
    );

    let (stdout, _) = ran(&quack(root, &["ingest", &folder_arg, "--no-embed"]), true);
    assert!(stdout.contains("skipped   a.md (identical to"), "{stdout}");
    assert!(
        stdout.contains("skipped   rates/b.csv (identical to"),
        "{stdout}"
    );

    write(&folder, "a.md", "# A\n\nFlood is covered.\n");
    std::fs::remove_file(folder.join("rates/b.csv")).unwrap_or_else(|e| fail(&e.to_string()));
    let (stdout, _) = ran(&quack(root, &["ingest", &folder_arg, "--no-embed"]), true);
    assert!(stdout.contains("replaced  a.md ("), "{stdout}");
    assert!(
        stdout.contains("still in the workspace; --prune deletes them): rates/b.csv ("),
        "{stdout}"
    );
    let (docs, _) = ran(&quack(root, &["docs"]), true);
    assert_eq!(docs.matches("a.md").count(), 1, "one live a.md: {docs}");
    assert!(docs.contains("b.csv"), "{docs}");
    let (all, _) = ran(&quack(root, &["docs", "--all"]), true);
    assert_eq!(all.matches("a.md").count(), 2, "{all}");
    assert!(all.contains("superseded"), "{all}");

    let (stdout, _) = ran(
        &quack(root, &["ingest", &folder_arg, "--no-embed", "--prune"]),
        true,
    );
    assert!(
        stdout.contains("Deleted (1 gone): rates/b.csv ("),
        "{stdout}"
    );
    let (docs, _) = ran(&quack(root, &["docs"]), true);
    assert!(!docs.contains("b.csv"), "{docs}");

    // A file that fails is reported with the rest and fails the command;
    // the flags for one file, or for a folder, do not cross over.
    write(&folder, "broken.csv", "a,b\n1,2,3,4\n\"unterminated,5\n6\n");
    let (stdout, stderr) = ran(&quack(root, &["ingest", &folder_arg, "--no-embed"]), false);
    assert!(stdout.contains("failed    broken.csv:"), "{stdout}");
    assert!(stderr.contains("1 of 2 files failed"), "{stderr}");
    let a = folder.join("a.md").display().to_string();
    let (_, stderr) = ran(
        &quack(root, &["ingest", &a, "--no-embed", "--prune"]),
        false,
    );
    assert!(stderr.contains("--prune goes with a folder"), "{stderr}");
    let (_, stderr) = ran(&quack(root, &["ingest", &folder_arg, "--pin"]), false);
    assert!(
        stderr.contains("take one file, not a directory"),
        "{stderr}"
    );
}
