//! Tests for `quack init`'s menus, flags, and file write.

use super::*;

use quack_core::config::BaseUrl;
use quack_core::setup::Found;

#[expect(clippy::unwrap_used, reason = "test")]
fn model(spec: &str) -> ModelSpec {
    spec.parse().unwrap()
}

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

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn the_embedding_flag_takes_a_model_or_none() {
    assert!(matches!(
        "none".parse::<EmbeddingFlag>().unwrap(),
        EmbeddingFlag::None
    ));
    assert!(matches!(
        "ollama/embed:small".parse::<EmbeddingFlag>().unwrap(),
        EmbeddingFlag::Model(_)
    ));
    assert!("no-slash".parse::<EmbeddingFlag>().is_err());
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn an_embedding_flag_needs_a_width_quack_knows() {
    assert_eq!(
        EmbeddingOption::from_flag(&model("ollama/embed:small")).unwrap(),
        EmbeddingOption::Ollama(String::from("embed:small"))
    );
    assert_eq!(
        EmbeddingOption::from_flag(&model("openai/text-embedding-3-small")).unwrap(),
        EmbeddingOption::Hosted(
            ProviderKind::OpenAi,
            "text-embedding-3-small",
            Dimension::new(1536)
        )
    );
    let refused = |spec: &str| {
        EmbeddingOption::from_flag(&model(spec))
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    };
    assert!(
        refused("openai/text-embedding-3-large")
            .contains("knows the width of openai/text-embedding-3-small")
    );
    assert!(refused("anthropic/claude-x").contains("serves no embedding models"));
    assert!(refused("gateway/embed").contains("sets up ollama, openai, and bedrock"));
}

#[test]
fn ollama_embedding_models_come_first_and_none_last() {
    let discovery = Discovery(vec![ollama_found(vec![
        ollama_model("chat:8b", vec![OllamaCapability::Tools]),
        ollama_model("embed:small", vec![OllamaCapability::Embedding]),
    ])]);
    let plan = SetupPlan {
        chat_model: Some(model("bedrock/us.anthropic.claude-x")),
        ..SetupPlan::default()
    };
    assert_eq!(
        EmbeddingOption::offered(&discovery, &plan),
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
    let plan = SetupPlan {
        chat_model: Some(model("anthropic/claude-x")),
        ..SetupPlan::default()
    };
    assert_eq!(
        EmbeddingOption::offered(&Discovery(Vec::new()), &plan),
        [EmbeddingOption::None]
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
#[expect(clippy::unwrap_used, reason = "test")]
fn a_new_file_is_written_whole_and_an_existing_one_is_never_replaced() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("quack").join("config.toml");
    write_new(&path, "[general]\n").unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "[general]\n");

    assert!(write_new(&path, "[embedding]\n").is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "[general]\n");
    // The staged file did not stay behind.
    assert_eq!(
        std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
        1
    );
}
