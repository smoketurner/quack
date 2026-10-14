//! The command line's long runs are recorded in the workspace's jobs
//! (`_quack_jobs`), so `quack jobs` shows what ran and how it ended.

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
fn cli_runs_are_recorded_in_the_workspace_jobs() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let root = dir.path();
    std::fs::create_dir_all(root.join("config")).unwrap_or_else(|e| fail(&e.to_string()));
    std::fs::write(root.join("config/config.toml"), "[general]\n")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let notes = root.join("notes.md");
    std::fs::write(&notes, "# Notes\n\nFlood is excluded.\n")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let notes = notes.to_str().unwrap_or_else(|| fail("path is not UTF-8"));
    let states = root.join("states.csv");
    std::fs::write(&states, "name,abbr\nTexas,TX\nOhio,OH\n")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let states = states.to_str().unwrap_or_else(|| fail("path is not UTF-8"));

    let ingested = quack(root, &["ingest", notes, "--no-embed"]);
    assert!(ingested.status.success(), "{}", stderr_of(&ingested));
    let loaded = quack(root, &["ingest", states, "--no-embed"]);
    assert!(loaded.status.success(), "{}", stderr_of(&loaded));
    let missing = quack(root, &["ingest", &format!("{notes}.gone")]);
    assert!(!missing.status.success());
    let status = quack(root, &["graph", "status"]);
    assert!(status.status.success(), "{}", stderr_of(&status));
    // Identical bytes are skipped at once: the quickest ingest there is,
    // whose recorded end must still reach the file before the process
    // exits (the stop sequence drains the recorder).
    let again = root.join("again.csv");
    std::fs::copy(states, &again).unwrap_or_else(|e| fail(&e.to_string()));
    let again = again.to_str().unwrap_or_else(|| fail("path is not UTF-8"));
    for _ in 0..3 {
        let skipped = quack(root, &["ingest", again, "--no-embed"]);
        assert!(skipped.status.success(), "{}", stderr_of(&skipped));
        assert!(
            String::from_utf8_lossy(&skipped.stdout).contains("Skipped: again.csv is identical"),
            "{}",
            String::from_utf8_lossy(&skipped.stdout)
        );
    }

    let jobs = quack(root, &["jobs"]);
    assert!(jobs.status.success(), "{}", stderr_of(&jobs));
    assert!(
        !stderr_of(&jobs).contains("left active by an earlier process"),
        "{}",
        stderr_of(&jobs)
    );
    // Newest first; the ingest that found no file never opened the
    // workspace, so it is not recorded.
    let listed = String::from_utf8_lossy(&jobs.stdout);
    let mut lines = listed.lines();
    for _ in 0..3 {
        let skipped = lines.next().unwrap_or_default();
        assert!(
            skipped.contains("succeeded")
                && skipped.contains("again.csv")
                && skipped.contains("(skipped: identical bytes are already stored)"),
            "{listed}"
        );
    }
    let first = lines.next().unwrap_or_default();
    assert!(
        first.contains("succeeded") && first.contains("graph status") && first.contains("(done)"),
        "{listed}"
    );
    let second = lines.next().unwrap_or_default();
    assert!(
        second.contains("succeeded")
            && second.contains("ingest")
            && second.contains("states.csv")
            && second.contains("(ingested)"),
        "{listed}"
    );
    let third = lines.next().unwrap_or_default();
    assert!(
        third.contains("succeeded") && third.contains("ingest") && third.contains("notes.md"),
        "{listed}"
    );
    assert_eq!(lines.next(), None, "{listed}");
}
