//! A workspace another `quack` holds: `-q` runs through the holder's
//! socket, `jobs` reads through it, and `mcp` relays to it. The holder is
//! this test's own `OpenedWorkspace` (what the terminal session and every
//! command hold); the other side is the binary.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::Duration;

use quack_cli::OpenedWorkspace;
use quack_core::config::Config;
use quack_core::forward::Holder;

fn command(dir: &Path, args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_quack"));
    command
        .args(args)
        .env("QUACK_CONFIG_DIR", dir.join("config"))
        .env("QUACK_DATA_DIR", dir.join("data"))
        .env_remove("QUACK_MODEL")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

fn quack(dir: &Path, args: &[&str]) -> Output {
    command(dir, args)
        .spawn()
        .unwrap_or_else(|e| fail(&format!("quack did not run: {e}")))
        .wait_with_output()
        .unwrap_or_else(|e| fail(&e.to_string()))
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

#[tokio::test(flavor = "multi_thread")]
async fn commands_reach_the_workspace_through_its_holder() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let root = dir.path();
    std::fs::create_dir_all(root.join("config")).unwrap_or_else(|e| fail(&e.to_string()));
    std::fs::write(root.join("config/config.toml"), "[general]\n")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let created = quack(root, &["workspace", "create", "ws"]);
    assert!(created.status.success(), "{}", stderr_of(&created));
    let loaded = quack(
        root,
        &["-w", "ws", "-q", "CREATE TABLE t AS SELECT 42 AS n"],
    );
    assert!(loaded.status.success(), "{}", stderr_of(&loaded));

    // Hold the workspace as the terminal session does: the socket is served.
    let mut config = Config::default();
    config.general.data_dir = root.join("data");
    let mut opened = OpenedWorkspace::in_config(config, Some("ws"), Holder::Terminal)
        .await
        .unwrap_or_else(|e| fail(&format!("{e:#}")));
    let handle = opened
        .handle()
        .await
        .unwrap_or_else(|e| fail(&format!("{e:#}")));

    let read = tokio::task::spawn_blocking({
        let root = root.to_path_buf();
        move || quack(&root, &["-w", "ws", "-q", "SELECT n FROM t", "-f", "csv"])
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(read.status.success(), "{}", stderr_of(&read));
    assert_eq!(String::from_utf8_lossy(&read.stdout).trim(), "n\n42");
    assert!(
        stderr_of(&read).contains("read through the quack terminal session"),
        "{}",
        stderr_of(&read)
    );

    let write = tokio::task::spawn_blocking({
        let root = root.to_path_buf();
        move || quack(&root, &["-w", "ws", "-q", "CREATE TABLE u (x INTEGER)"])
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(!write.status.success());
    let refused = stderr_of(&write);
    assert!(
        refused.contains("the quack terminal session") && refused.contains("cannot write"),
        "{refused}"
    );

    let jobs = tokio::task::spawn_blocking({
        let root = root.to_path_buf();
        move || quack(&root, &["-w", "ws", "jobs"])
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(jobs.status.success(), "{}", stderr_of(&jobs));
    assert!(
        stderr_of(&jobs).contains("read through the quack terminal session"),
        "{}",
        stderr_of(&jobs)
    );

    // `quack mcp` relays: an initialize round trip names the holder.
    let relayed = tokio::task::spawn_blocking({
        let root = root.to_path_buf();
        move || initialize_over_mcp(&root)
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
    let (line, status) = relayed;
    assert!(status.success(), "{status:?}");
    let reply: serde_json::Value =
        serde_json::from_str(&line).unwrap_or_else(|e| fail(&format!("{e}: {line}")));
    assert_eq!(
        reply.pointer("/result/serverInfo/title"),
        Some(&serde_json::Value::from("the quack terminal session")),
        "{line}"
    );

    drop(handle);
    opened.close_socket(Duration::from_secs(5)).await;
    opened.finish(Duration::ZERO).await;
}

/// Send `initialize` to `quack mcp` and return its first reply line and
/// the exit status once stdin closes.
fn initialize_over_mcp(root: &Path) -> (String, std::process::ExitStatus) {
    let mut child = command(root, &["-w", "ws", "mcp"])
        .spawn()
        .unwrap_or_else(|e| fail(&format!("quack mcp did not run: {e}")));
    let mut stdin = child.stdin.take().unwrap_or_else(|| fail("no stdin"));
    let stdout = child.stdout.take().unwrap_or_else(|| fail("no stdout"));
    stdin
                .write_all(
                    br#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#,
                )
                .unwrap_or_else(|e| fail(&e.to_string()));
    stdin
        .write_all(b"\n")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let mut line = String::new();
    BufReader::new(stdout)
        .read_line(&mut line)
        .unwrap_or_else(|e| fail(&e.to_string()));
    drop(stdin);
    let status = child.wait().unwrap_or_else(|e| fail(&e.to_string()));
    (line, status)
}
