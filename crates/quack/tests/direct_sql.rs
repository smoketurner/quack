//! `quack -q` passes the typed-SQL gate: internal tables and reserved
//! names are refused as they are in the terminal, on the web SQL page, and
//! through a holder; every row of a read is written.

use std::path::Path;
use std::process::{Command, Output, Stdio};

fn quack(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_quack"))
        .args(args)
        .env("QUACK_CONFIG_DIR", dir.join("config"))
        .env("QUACK_DATA_DIR", dir.join("data"))
        .env_remove("QUACK_MODEL")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|e| fail(&format!("quack did not run: {e}")))
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

#[test]
fn q_refuses_internal_tables_and_reserved_names_and_writes_every_row() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let root = dir.path();
    std::fs::create_dir_all(root.join("config")).unwrap_or_else(|e| fail(&e.to_string()));
    std::fs::write(root.join("config/config.toml"), "[general]\n")
        .unwrap_or_else(|e| fail(&e.to_string()));

    let internal = quack(root, &["-q", "SELECT * FROM _quack_documents LIMIT 1"]);
    assert!(!internal.status.success(), "{}", stderr_of(&internal));
    assert!(
        stderr_of(&internal).contains("internal tables are not accessible"),
        "{}",
        stderr_of(&internal)
    );
    for reserved in [
        "CREATE TABLE _quack_x AS SELECT 1",
        "CREATE TABLE graph_x AS SELECT 1",
    ] {
        let refused = quack(root, &["-q", reserved]);
        assert!(!refused.status.success(), "{reserved} ran");
        // A `_quack_` name is an internal table; a `graph_` name is the
        // views' reserved prefix.
        let message = stderr_of(&refused);
        assert!(
            message.contains("internal tables are not accessible")
                || message.contains("graph_ prefix is reserved"),
            "{reserved}: {message}"
        );
    }
    let listed = quack(root, &["-q", "SHOW TABLES", "-f", "csv"]);
    assert!(listed.status.success(), "{}", stderr_of(&listed));
    assert!(
        !String::from_utf8_lossy(&listed.stdout).contains("graph_x"),
        "{}",
        String::from_utf8_lossy(&listed.stdout)
    );

    // Every row, past the interactive interfaces' cap.
    let rows = quack(
        root,
        &["-q", "SELECT range AS n FROM range(1000)", "-f", "csv"],
    );
    assert!(rows.status.success(), "{}", stderr_of(&rows));
    assert_eq!(String::from_utf8_lossy(&rows.stdout).lines().count(), 1001);
}
