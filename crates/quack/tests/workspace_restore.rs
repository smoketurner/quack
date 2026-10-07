//! `quack workspace snapshot NAME --to -` piped into `quack workspace
//! restore -`: the binary itself, with the tar passing through stdout and
//! stdin.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

fn quack(dir: &Path, args: &[&str], stdin: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_quack"))
        .args(args)
        .env("QUACK_CONFIG_DIR", dir.join("config"))
        .env("QUACK_DATA_DIR", dir.join("data"))
        .env_remove("QUACK_MODEL")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| fail(&format!("quack did not run: {e}")));
    if let Some(mut pipe) = child.stdin.take() {
        pipe.write_all(stdin)
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    let output = child
        .wait_with_output()
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        output.status.success(),
        "quack {args:?}: exit {:?}\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

#[test]
fn a_snapshot_on_stdin_restores_and_leaves_no_spool_behind() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let root = dir.path();
    std::fs::create_dir_all(root.join("config")).unwrap_or_else(|e| fail(&e.to_string()));
    std::fs::write(root.join("config/config.toml"), "[general]\n")
        .unwrap_or_else(|e| fail(&e.to_string()));
    quack(root, &["workspace", "create", "ws"], b"");
    quack(
        root,
        &["-w", "ws", "-q", "CREATE TABLE t AS SELECT 42 AS n"],
        b"",
    );
    let tar = quack(root, &["workspace", "snapshot", "ws", "--to", "-"], b"").stdout;

    quack(root, &["workspace", "restore", "-", "--name", "copy"], &tar);
    let out = quack(
        root,
        &["-w", "copy", "-q", "SELECT n FROM t", "-f", "csv"],
        b"",
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("42"));
    let left: Vec<_> = std::fs::read_dir(root.join("data"))
        .unwrap_or_else(|e| fail(&e.to_string()))
        .filter_map(Result::ok)
        .map(|entry| entry.file_name())
        .filter(|name| name.to_string_lossy().starts_with(".tmp"))
        .collect();
    assert!(left.is_empty(), "spool left behind: {left:?}");
}
