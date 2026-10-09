//! `quack init`, the binary itself: against a stand-in Ollama, with
//! nothing to find, and over a config that already exists.

use std::path::Path;
use std::process::{Command, Output, Stdio};

use axum::routing::{get, post};

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

/// `quack init ARGS` with its config and data under `dir`, Ollama at
/// `ollama`, and no hosted API keys. `HOME` stays, since doctor's vault
/// check reads the keychain it names; `--yes` never picks Bedrock, so
/// AWS profiles there change nothing.
fn init(dir: &Path, ollama: &str, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_quack"))
        .arg("init")
        .args(args)
        .env("QUACK_CONFIG_DIR", dir.join("config"))
        .env("QUACK_DATA_DIR", dir.join("data"))
        .env("OLLAMA_HOST", ollama)
        .env_remove("QUACK_MODEL")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("OPENAI_API_KEY")
        .env_remove("AWS_PROFILE")
        .env_remove("AWS_ACCESS_KEY_ID")
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|e| fail(&format!("quack did not run: {e}")))
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// An Ollama with a tools model and an embedding model whose vectors are
/// 4 wide, on its own runtime thread.
fn ollama_stub() -> String {
    let tags = serde_json::json!({ "models": [
        { "name": "chat:8b", "model": "chat:8b", "size": 5_200_000_000_u64 },
        { "name": "embed:small", "model": "embed:small", "size": 639_000_000_u64 },
    ]})
    .to_string();
    let app = axum::Router::new()
        .route("/api/tags", get(move || async move { tags }))
        .route("/api/version", get(|| async { r#"{"version":"0.40.1"}"# }))
        .route(
            "/api/show",
            post(|body: String| async move {
                if body.contains("embed:small") {
                    r#"{"capabilities":["embedding"]}"#
                } else {
                    r#"{"capabilities":["completion","tools"]}"#
                }
            }),
        )
        .route(
            "/api/embed",
            post(|| async { r#"{"embeddings":[[0.1,0.2,0.3,0.4]]}"# }),
        );
    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").unwrap_or_else(|e| fail(&e.to_string()));
    let address = listener
        .local_addr()
        .unwrap_or_else(|e| fail(&e.to_string()));
    listener
        .set_nonblocking(true)
        .unwrap_or_else(|e| fail(&e.to_string()));
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|e| fail(&e.to_string()));
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::from_std(listener)
                .unwrap_or_else(|e| fail(&e.to_string()));
            drop(axum::serve(listener, app).await);
        });
    });
    format!("http://{address}")
}

/// A loopback address nothing listens on.
fn refused() -> String {
    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").unwrap_or_else(|e| fail(&e.to_string()));
    let address = listener
        .local_addr()
        .unwrap_or_else(|e| fail(&e.to_string()));
    drop(listener);
    format!("http://{address}")
}

#[test]
fn yes_writes_the_top_ollama_models_once_doctor_passes() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let output = init(dir.path(), &ollama_stub(), &["--yes"]);
    let said = stderr(&output);
    assert!(output.status.success(), "{said}");
    assert!(said.contains("✓ Ollama at"), "{said}");
    assert!(
        said.contains("✗ Anthropic: ANTHROPIC_API_KEY is not set"),
        "{said}"
    );
    let written = std::fs::read_to_string(dir.path().join("config/config.toml"))
        .unwrap_or_else(|e| fail(&format!("{e}\n{said}")));
    assert!(
        written.contains("chat_model = \"ollama/chat:8b\""),
        "{written}"
    );
    assert!(
        written.contains("model = \"ollama/embed:small\""),
        "{written}"
    );
    assert!(written.contains("dimension = 4"), "{written}");

    // A second run never touches the file it wrote.
    let again = init(dir.path(), &ollama_stub(), &["--yes"]);
    assert_eq!(again.status.code(), Some(2), "{}", stderr(&again));
    assert!(stderr(&again).contains("already exists"));
    let unchanged = std::fs::read_to_string(dir.path().join("config/config.toml"))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(unchanged, written);
}

#[test]
fn print_writes_nothing_to_disk() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let output = init(dir.path(), &ollama_stub(), &["--yes", "--print"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let printed = String::from_utf8_lossy(&output.stdout);
    assert!(
        printed.contains("chat_model = \"ollama/chat:8b\""),
        "{printed}"
    );
    assert!(!dir.path().join("config/config.toml").exists());
}

#[test]
fn a_config_that_fails_doctor_is_never_written() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    // The named chat model is not one the stand-in Ollama has pulled.
    let output = init(
        dir.path(),
        &ollama_stub(),
        &[
            "--yes",
            "--chat-model",
            "ollama/missing:1b",
            "--embedding-model",
            "none",
        ],
    );
    let said = stderr(&output);
    assert_eq!(output.status.code(), Some(1), "{said}");
    assert!(
        said.contains("fail  chat model  ollama/missing:1b"),
        "{said}"
    );
    assert!(said.contains("Nothing was written"), "{said}");
    assert!(!dir.path().join("config/config.toml").exists());
}

#[test]
fn yes_with_nothing_found_says_what_to_do_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let output = init(dir.path(), &refused(), &["--yes"]);
    let said = stderr(&output);
    assert_eq!(output.status.code(), Some(2), "{said}");
    assert!(said.contains("Start Ollama"), "{said}");
    assert!(said.contains("--chat-model PROVIDER/MODEL"), "{said}");
    assert!(!dir.path().join("config/config.toml").exists());
}

#[test]
fn without_a_terminal_it_asks_for_yes() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let output = init(dir.path(), &refused(), &[]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("--yes takes the defaults"));
}
