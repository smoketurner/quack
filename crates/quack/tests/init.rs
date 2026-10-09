//! `quack init`, the binary itself. Its questions need a terminal, so this
//! checks only what happens without one; the menus and the checked write
//! are tested in `init_cli`.

use std::path::Path;
use std::process::{Command, Output, Stdio};

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

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

#[test]
fn without_a_terminal_init_says_so_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let output = quack(dir.path(), &["init"]);
    let said = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{said}");
    assert!(said.contains("needs a terminal"), "{said}");
    assert!(!dir.path().join("config/config.toml").exists());
}

#[test]
fn without_a_terminal_an_existing_file_is_untouched() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let config = dir.path().join("config");
    std::fs::create_dir_all(&config).unwrap_or_else(|e| fail(&e.to_string()));
    let old = "# mine\n[general]\n";
    std::fs::write(config.join("config.toml"), old).unwrap_or_else(|e| fail(&e.to_string()));
    let output = quack(dir.path(), &["init"]);
    assert_eq!(output.status.code(), Some(2));
    let now = std::fs::read_to_string(config.join("config.toml"))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(now, old);
}

#[test]
fn init_takes_no_flags() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    for flag in ["--yes", "--print", "--chat-model", "--embedding-model"] {
        let output = quack(dir.path(), &["init", flag]);
        assert_eq!(output.status.code(), Some(2), "{flag}");
    }
}
