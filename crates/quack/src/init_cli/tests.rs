//! Tests for `quack init`'s menus and the checked write.

use super::*;

use quack_core::config::BaseUrl;
use quack_core::doctor::Probing;

fn ollama_model(name: &str, capabilities: Vec<OllamaCapability>) -> OllamaModel {
    OllamaModel {
        name: name.to_owned(),
        size: 639_000_000,
        capabilities,
    }
}

#[expect(clippy::unwrap_used, reason = "test")]
fn ollama_found(models: Vec<OllamaModel>) -> Found {
    Found::Ollama {
        base_url: BaseUrl::try_from(String::from("http://localhost:11434")).unwrap(),
        models,
    }
}

fn chat_on(kind: ProviderKind) -> SetupPlan {
    SetupPlan {
        chat: Some(Choice {
            kind,
            model: String::from("m"),
        }),
        ..SetupPlan::default()
    }
}

#[test]
fn a_new_file_offers_ollama_then_the_chat_providers_model_then_none() {
    let discovery = Discovery(vec![ollama_found(vec![
        ollama_model("chat:8b", vec![OllamaCapability::Tools]),
        ollama_model("embed:small", vec![OllamaCapability::Embedding]),
    ])]);
    assert_eq!(
        EmbeddingOption::offered(
            &discovery,
            &chat_on(ProviderKind::Bedrock),
            &Current::default()
        ),
        [
            EmbeddingOption::Ollama(String::from("embed:small")),
            EmbeddingOption::Hosted(
                ProviderKind::Bedrock,
                "amazon.titan-embed-text-v2:0",
                Dimension::new(1024)
            ),
            EmbeddingOption::None,
        ]
    );
    // Anthropic serves no embeddings, and nothing else was found.
    assert_eq!(
        EmbeddingOption::offered(
            &Discovery(Vec::new()),
            &chat_on(ProviderKind::Anthropic),
            &Current::default()
        ),
        [EmbeddingOption::None]
    );
}

#[test]
fn an_existing_model_is_offered_first_and_never_dropped_for_none() {
    let current = Current {
        chat_model: None,
        embedding_model: Some(String::from("local/nomic-embed-text")),
        decision_model: None,
    };
    let discovery = Discovery(vec![ollama_found(vec![ollama_model(
        "embed:small",
        vec![OllamaCapability::Embedding],
    )])]);
    assert_eq!(
        EmbeddingOption::offered(&discovery, &SetupPlan::default(), &current),
        [
            EmbeddingOption::Keep(String::from("local/nomic-embed-text")),
            EmbeddingOption::Ollama(String::from("embed:small")),
        ]
    );
    assert_eq!(
        EmbeddingOption::Keep(String::from("local/nomic-embed-text")).to_string(),
        "Keep local/nomic-embed-text"
    );
}

#[test]
fn a_model_menu_line_shows_size_and_capabilities() {
    let line = ModelOption(&ollama_model(
        "embed:small",
        vec![OllamaCapability::Embedding, OllamaCapability::Other],
    ))
    .to_string();
    assert!(line.starts_with("embed:small "), "{line}");
    assert!(line.contains("639 MB  embedding"), "{line}");
    assert!(!line.contains("other"), "{line}");
}

#[test]
#[cfg(unix)]
#[expect(clippy::unwrap_used, reason = "test")]
fn replace_writes_a_new_file_and_replaces_one_whole_with_its_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("quack").join("config.toml");
    replace(&path, "[general]\n").unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "[general]\n");

    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
    replace(&path, "[embedding]\n").unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "[embedding]\n");
    let mode = std::fs::metadata(&path).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o640);
    // The staged file did not stay behind.
    assert_eq!(
        std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
        1
    );
}

#[test]
#[cfg(unix)]
#[expect(clippy::unwrap_used, reason = "test")]
fn replace_writes_through_a_symlink_and_keeps_it() {
    let dir = tempfile::tempdir().unwrap();
    let real = dir.path().join("dotfiles").join("quack.toml");
    std::fs::create_dir_all(real.parent().unwrap()).unwrap();
    std::fs::write(&real, "[general]\n").unwrap();
    let link = dir.path().join("config.toml");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    replace(&link, "[embedding]\n").unwrap();
    assert!(
        std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(std::fs::read_to_string(&real).unwrap(), "[embedding]\n");
}

/// A loopback address nothing listens on.
#[expect(clippy::unwrap_used, reason = "test")]
fn refused() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{address}")
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn a_config_that_fails_doctor_leaves_the_old_file_byte_for_byte() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    let old = "# mine\n[general]\ndefault_workspace = \"research\"\n";
    std::fs::write(&path, old).unwrap();
    let candidate = format!(
        "[general]\ndata_dir = '{}'\nchat_model = \"o/chat:8b\"\n\n[providers.o]\ntype = \
         \"ollama\"\nbase_url = \"{}\"\nmax_retries = 0\n",
        dir.path().join("data").display(),
        refused()
    );
    let options = Options {
        workspace: None,
        probing: Probing::Online {
            timeout: std::time::Duration::from_secs(2),
        },
    };
    let mut talk = Vec::new();
    let code = check_and_write(&path, &candidate, &options, &mut talk)
        .await
        .unwrap();
    let said = String::from_utf8(talk).unwrap();
    assert_eq!(code, ExitCode::FAILURE, "{said}");
    assert!(said.contains("fail  chat model  o/chat:8b"), "{said}");
    assert!(said.contains("Nothing was written"), "{said}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), old);
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn a_config_that_passes_doctor_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    let candidate = format!(
        "[general]\ndata_dir = '{}'\n",
        dir.path().join("data").display()
    );
    let options = Options {
        workspace: None,
        probing: Probing::Offline,
    };
    let mut talk = Vec::new();
    let code = check_and_write(&path, &candidate, &options, &mut talk)
        .await
        .unwrap();
    let said = String::from_utf8(talk).unwrap();
    assert_eq!(code, ExitCode::SUCCESS, "{said}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), candidate);
}
