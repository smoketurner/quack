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

    // `quack mcp` reaches the holder: this process holds the file, so the
    // statement can only have run through its socket.
    let relayed = tokio::task::spawn_blocking({
        let root = root.to_path_buf();
        move || {
            let mut mcp = McpChild::start(&root);
            let initialized = mcp.initialize();
            let rows = mcp.sql("SELECT n FROM t");
            (initialized, rows, mcp.finish())
        }
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
    let (initialized, rows, status) = relayed;
    assert!(status.success(), "{status:?}");
    assert_eq!(
        initialized.pointer("/result/serverInfo/description"),
        Some(&serde_json::Value::from("quack mcp")),
        "{initialized}"
    );
    assert_eq!(
        rows.pointer("/result/structuredContent/rows/0/0"),
        Some(&serde_json::Value::from(42)),
        "{rows}"
    );

    drop(handle);
    opened.close_socket(Duration::from_secs(5)).await;
    opened.finish(Duration::ZERO).await;
}

/// `quack mcp` holds nothing between calls: started before anything holds
/// the workspace, it opens the file for a call and lets it go, so a
/// session can take the file while it runs, and its next call goes
/// through that session.
#[tokio::test(flavor = "multi_thread")]
async fn quack_mcp_holds_nothing_between_calls() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let root = dir.path();
    std::fs::create_dir_all(root.join("config")).unwrap_or_else(|e| fail(&e.to_string()));
    std::fs::write(root.join("config/config.toml"), "[general]\n")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let created = quack(root, &["workspace", "create", "ws"]);
    assert!(created.status.success(), "{}", stderr_of(&created));
    let loaded = quack(root, &["-w", "ws", "-q", "CREATE TABLE t AS SELECT 7 AS n"]);
    assert!(loaded.status.success(), "{}", stderr_of(&loaded));

    let mut mcp = tokio::task::spawn_blocking({
        let root = root.to_path_buf();
        move || {
            let mut mcp = McpChild::start(&root);
            let initialized = mcp.initialize();
            assert_eq!(
                initialized.pointer("/result/serverInfo/description"),
                Some(&serde_json::Value::from("quack mcp")),
                "{initialized}"
            );
            let rows = mcp.sql("SELECT n FROM t");
            assert_eq!(
                rows.pointer("/result/structuredContent/rows/0/0"),
                Some(&serde_json::Value::from(7)),
                "{rows}"
            );
            mcp
        }
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));

    // The file is free between calls: a session takes it.
    let mut config = Config::default();
    config.general.data_dir = root.join("data");
    let mut opened = OpenedWorkspace::in_config(config, Some("ws"), Holder::Terminal)
        .await
        .unwrap_or_else(|e| fail(&format!("{e:#}")));
    let handle = opened
        .handle()
        .await
        .unwrap_or_else(|e| fail(&format!("{e:#}")));

    // Its next call runs in the session.
    let (rows, status) = tokio::task::spawn_blocking(move || {
        let rows = mcp.sql("SELECT n + 1 AS m FROM t");
        (rows, mcp.finish())
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(status.success(), "{status:?}");
    assert_eq!(
        rows.pointer("/result/structuredContent/rows/0/0"),
        Some(&serde_json::Value::from(8)),
        "{rows}"
    );

    drop(handle);
    opened.close_socket(Duration::from_secs(5)).await;
    opened.finish(Duration::ZERO).await;
}

/// A `quack mcp` child driven over its stdin and stdout, one JSON-RPC
/// line each way.
struct McpChild {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    next_id: u64,
}

impl McpChild {
    fn start(root: &Path) -> Self {
        let mut child = command(root, &["-w", "ws", "mcp"])
            .spawn()
            .unwrap_or_else(|e| fail(&format!("quack mcp did not run: {e}")));
        let stdin = child.stdin.take().unwrap_or_else(|| fail("no stdin"));
        let stdout = BufReader::new(child.stdout.take().unwrap_or_else(|| fail("no stdout")));
        Self {
            child,
            stdin,
            stdout,
            next_id: 1,
        }
    }

    /// Send `method` with `params` and read the reply.
    fn request(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let id = self.next_id;
        self.next_id += 1;
        let request = serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": method, "params": params,
        });
        self.stdin
            .write_all(format!("{request}\n").as_bytes())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let mut line = String::new();
        self.stdout
            .read_line(&mut line)
            .unwrap_or_else(|e| fail(&e.to_string()));
        serde_json::from_str(&line).unwrap_or_else(|e| fail(&format!("{e}: {line}")))
    }

    fn initialize(&mut self) -> serde_json::Value {
        let reply = self.request(
            "initialize",
            serde_json::json!({
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            }),
        );
        let initialized = serde_json::json!({
            "jsonrpc": "2.0", "method": "notifications/initialized",
        });
        self.stdin
            .write_all(format!("{initialized}\n").as_bytes())
            .unwrap_or_else(|e| fail(&e.to_string()));
        reply
    }

    fn sql(&mut self, statement: &str) -> serde_json::Value {
        self.request(
            "tools/call",
            serde_json::json!({ "name": "sql", "arguments": { "sql": statement } }),
        )
    }

    /// Close stdin and wait for the exit.
    fn finish(mut self) -> std::process::ExitStatus {
        drop(self.stdin);
        self.child.wait().unwrap_or_else(|e| fail(&e.to_string()))
    }
}
