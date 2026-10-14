//! A workspace another `quack` holds: `-q` runs through the holder's
//! socket, `jobs` reads through it, and `mcp` relays to it. The holder is
//! this test's own `OpenedWorkspace` (what the terminal session and every
//! command hold); the other side is the binary. Unix only: the socket is.
#![cfg(unix)]

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

    q_and_jobs_read_through_the_holder(root).await;
    other_commands_are_refused_naming_the_holder(root).await;

    // `quack mcp` reaches the holder: this process holds the file, so the
    // statement can only have run through its socket.
    let relayed = tokio::task::spawn_blocking({
        let root = root.to_path_buf();
        move || {
            let mut mcp = McpChild::start(&root);
            let initialized = mcp.initialize();
            let rows = mcp.sql("SELECT n FROM t");
            // The holder's own errors keep their codes through the relay.
            let no_tool = mcp.request(
                "tools/call",
                &serde_json::json!({ "name": "nope", "arguments": {} }),
            );
            let no_resource = mcp.request(
                "resources/read",
                &serde_json::json!({ "uri": "quack://workspace/nothing-here" }),
            );
            (initialized, rows, no_tool, no_resource, mcp.finish())
        }
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
    let (initialized, rows, no_tool, no_resource, status) = relayed;
    assert!(status.success(), "{status:?}");
    assert_eq!(
        no_tool.pointer("/error/code"),
        Some(&serde_json::Value::from(-32602)),
        "{no_tool}"
    );
    assert_eq!(
        no_resource.pointer("/error/code"),
        Some(&serde_json::Value::from(-32002)),
        "{no_resource}"
    );
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

/// Two `quack mcp` processes on one workspace nobody else holds both
/// work: whichever opens the file for a call serves the socket while it
/// holds it, and the other relays through it (or waits out the moment
/// between the open and the socket), so interleaved calls from two
/// editors never fail on the lock or blame `quack serve`.
#[tokio::test(flavor = "multi_thread")]
async fn two_quack_mcp_processes_share_one_workspace() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let root = dir.path();
    std::fs::create_dir_all(root.join("config")).unwrap_or_else(|e| fail(&e.to_string()));
    std::fs::write(root.join("config/config.toml"), "[general]\n")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let created = quack(root, &["workspace", "create", "ws"]);
    assert!(created.status.success(), "{}", stderr_of(&created));
    let loaded = quack(root, &["-w", "ws", "-q", "CREATE TABLE t AS SELECT 3 AS n"]);
    assert!(loaded.status.success(), "{}", stderr_of(&loaded));

    let root = root.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let mut a = McpChild::start(&root);
        let mut b = McpChild::start(&root);
        a.initialize();
        b.initialize();
        // A statement that takes a moment, so the calls overlap.
        let slow = "SELECT count(*) AS c FROM range(20000000) WHERE range % 7 = 0";
        let answers = std::thread::scope(|scope| {
            let mut threads = Vec::new();
            for child in [&mut a, &mut b] {
                threads
                    .push(scope.spawn(move || (0..3).map(|_| child.sql(slow)).collect::<Vec<_>>()));
            }
            threads
                .into_iter()
                .map(|t| t.join().unwrap_or_else(|_| fail("a caller panicked")))
                .collect::<Vec<_>>()
        });
        for answer in answers.iter().flatten() {
            assert_eq!(
                answer.pointer("/result/structuredContent/rows/0/0"),
                Some(&serde_json::Value::from(2_857_143)),
                "{answer}"
            );
        }
        assert!(a.finish().success());
        assert!(b.finish().success());
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
}

/// `-q` reads through the holder's `sql` tool (a write is the holder's
/// refusal) and `jobs` reads its jobs resource, each saying so on stderr.
async fn q_and_jobs_read_through_the_holder(root: &Path) {
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
    // The session's prompt would ask; this test's holder has nobody at it.
    assert!(!write.status.success());
    let refused = stderr_of(&write);
    assert!(
        refused.contains("the quack terminal session") && refused.contains("not approved"),
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
}

/// Every command but `-q` and `jobs` is refused on a held workspace,
/// naming the holder, what still works, and what to quit it for.
async fn other_commands_are_refused_naming_the_holder(root: &Path) {
    let notes = root.join("notes.md");
    std::fs::write(&notes, "# Notes\n\nA line.\n").unwrap_or_else(|e| fail(&e.to_string()));
    let notes = notes
        .to_str()
        .unwrap_or_else(|| fail("path is not UTF-8"))
        .to_owned();
    for (args, verb) in [
        (vec![String::from("sessions")], "sessions"),
        (vec![String::from("ingest"), notes.clone()], "ingest"),
    ] {
        let refused = tokio::task::spawn_blocking({
            let root = root.to_path_buf();
            move || {
                let mut full = vec!["-w", "ws"];
                full.extend(args.iter().map(String::as_str));
                quack(&root, &full)
            }
        })
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(!refused.status.success(), "{verb} ran on a held workspace");
        let message = stderr_of(&refused);
        assert!(
            message.contains("the workspace is open in the quack terminal session")
                && message.contains("`quack -q` and `quack jobs` read through it")
                && message.contains(&format!("quit it to run `quack {verb}`")),
            "{verb}: {message}"
        );
    }
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
    fn request(&mut self, method: &str, params: &serde_json::Value) -> serde_json::Value {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
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
            &serde_json::json!({
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
            &serde_json::json!({ "name": "sql", "arguments": { "sql": statement } }),
        )
    }

    /// Close stdin and wait for the exit.
    fn finish(mut self) -> std::process::ExitStatus {
        drop(self.stdin);
        self.child.wait().unwrap_or_else(|e| fail(&e.to_string()))
    }
}
