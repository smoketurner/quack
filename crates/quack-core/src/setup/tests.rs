//! Tests for `quack init`'s discovery and the file it writes.

use super::*;

use axum::routing::{get, post};

/// An Ollama with three models: one that calls tools, one that embeds, and
/// one from a server too old to report capabilities. `/api/embed` answers
/// with a 4-wide vector.
#[expect(clippy::unwrap_used, reason = "test")]
async fn ollama_stub() -> String {
    let tags = serde_json::json!({ "models": [
        { "name": "chat:8b", "model": "chat:8b", "size": 5_200_000_000_u64 },
        { "name": "embed:small", "model": "embed:small", "size": 639_000_000_u64 },
        { "name": "old:latest", "model": "old:latest", "size": 1_000_u64 },
    ]})
    .to_string();
    let app = axum::Router::new()
        .route("/api/tags", get(move || async move { tags }))
        .route(
            "/api/show",
            post(|body: String| async move {
                let asked: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
                let reply = match asked.get("model").and_then(serde_json::Value::as_str) {
                    Some("chat:8b") => {
                        serde_json::json!({ "capabilities": ["completion", "tools", "thinking"] })
                    }
                    Some("embed:small") => serde_json::json!({ "capabilities": ["embedding"] }),
                    Some(_) | None => serde_json::json!({ "details": {} }),
                };
                reply.to_string()
            }),
        )
        .route(
            "/api/embed",
            post(|| async {
                serde_json::json!({ "embeddings": [[0.1, 0.2, 0.3, 0.4]] }).to_string()
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        drop(axum::serve(listener, app).await);
    });
    base
}

/// A loopback address nothing listens on.
#[expect(clippy::unwrap_used, reason = "test")]
async fn refused_port() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{address}")
}

fn with_ollama(host: &str) -> Environment {
    Environment {
        ollama_host: Some(host.to_owned()),
        ..Environment::default()
    }
}

#[tokio::test]
#[expect(clippy::unwrap_used, clippy::panic, reason = "test")]
async fn discovery_lists_ollama_models_with_their_capabilities() {
    let base = ollama_stub().await;
    let discovery = Discovery::run(&with_ollama(&base)).await;
    let Some(Found::Ollama { models, .. }) = discovery.get(ProviderKind::Ollama) else {
        panic!("{discovery:?}");
    };
    assert_eq!(models.len(), 3);
    let tools: Vec<&str> = discovery
        .ollama_models(OllamaCapability::Tools)
        .iter()
        .map(|m| m.name.as_str())
        .collect();
    assert_eq!(tools, ["chat:8b"]);
    let embedding: Vec<&str> = discovery
        .ollama_models(OllamaCapability::Embedding)
        .iter()
        .map(|m| m.name.as_str())
        .collect();
    assert_eq!(embedding, ["embed:small"]);
    // A server too old to report capabilities offers the model for nothing.
    let old = models.iter().find(|m| m.name == "old:latest").unwrap();
    assert!(old.capabilities.is_empty());
    assert_eq!(models.first().unwrap().size, 5_200_000_000);
    assert_eq!(discovery.ollama_base_url().unwrap().to_string(), base);
}

#[tokio::test]
async fn a_refused_ollama_is_absent_not_an_error() {
    let base = refused_port().await;
    let discovery = Discovery::run(&with_ollama(&base)).await;
    let found = discovery.get(ProviderKind::Ollama);
    assert!(
        matches!(found, Some(Found::Absent { reason, .. }) if reason.contains("not running")),
        "{found:?}"
    );
    assert!(discovery.ollama_models(OllamaCapability::Tools).is_empty());
    assert!(discovery.ollama_base_url().is_none());
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn an_ollama_embedding_model_is_measured_with_one_call() {
    let base = ollama_stub().await;
    let plan = SetupPlan {
        decision: None,
        ollama_base_url: Some(BaseUrl::try_from(base).unwrap()),
        ..SetupPlan::default()
    };
    assert_eq!(
        plan.ollama_width("embed:small").await.unwrap(),
        Dimension::new(4)
    );
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn ollama_host_is_read_as_ollama_reads_it() {
    let url = |host: Option<&str>| {
        Environment {
            ollama_host: host.map(str::to_owned),
            ..Environment::default()
        }
        .ollama_base_url()
        .unwrap()
        .to_string()
    };
    assert_eq!(url(None), "http://localhost:11434");
    assert_eq!(url(Some("127.0.0.1:11434")), "http://127.0.0.1:11434");
    assert_eq!(url(Some("0.0.0.0")), "http://0.0.0.0:11434");
    assert_eq!(url(Some("http://gpu-box:8000/")), "http://gpu-box:8000");
    assert_eq!(
        url(Some("https://ollama.example.com")),
        "https://ollama.example.com"
    );
    assert!(
        Environment {
            ollama_host: Some(String::from("http://[bad")),
            ..Environment::default()
        }
        .ollama_base_url()
        .is_err()
    );
}

#[tokio::test]
async fn hosted_providers_are_found_by_their_credentials_alone() {
    let env = Environment {
        ollama_host: Some(refused_port().await),
        anthropic_key: Some(String::from("sk-ant-secret")),
        aws_profile: Some(String::from("dev")),
        ..Environment::default()
    };
    let discovery = Discovery::run(&env).await;
    let line = |kind| {
        discovery
            .get(kind)
            .map(ToString::to_string)
            .unwrap_or_default()
    };
    assert_eq!(
        line(ProviderKind::Anthropic),
        "✓ Anthropic: ANTHROPIC_API_KEY is set"
    );
    assert_eq!(
        line(ProviderKind::OpenAi),
        "✗ OpenAI: OPENAI_API_KEY is not set"
    );
    assert_eq!(
        line(ProviderKind::Bedrock),
        "✓ Amazon Bedrock: AWS profile \"dev\""
    );
    for found in &discovery.0 {
        assert!(!found.to_string().contains("secret"), "{found}");
    }

    let bedrock = |env: Environment| env.hosted(ProviderKind::Bedrock).to_string();
    assert_eq!(
        bedrock(Environment {
            aws_access_key: Some(String::from("AKIA")),
            ..Environment::default()
        }),
        "✓ Amazon Bedrock: AWS_ACCESS_KEY_ID is set"
    );
    assert_eq!(
        bedrock(Environment {
            aws_files: Some(PathBuf::from("/home/me/.aws/config")),
            ..Environment::default()
        }),
        "✓ Amazon Bedrock: profiles in /home/me/.aws/config"
    );
    assert_eq!(
        bedrock(Environment::default()),
        "✗ Amazon Bedrock: no AWS profile or credentials"
    );
}

fn choice(kind: ProviderKind, model: &str) -> Choice {
    Choice {
        kind,
        model: model.to_owned(),
    }
}

fn embedding(kind: ProviderKind, model: &str, width: u32) -> EmbeddingChoice {
    EmbeddingChoice {
        choice: choice(kind, model),
        dimension: Dimension::new(width),
    }
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn each_provider_type_writes_a_new_file_quack_loads() {
    let plans = [
        SetupPlan {
            chat: Some(choice(ProviderKind::Ollama, "chat:8b")),
            embedding: Some(embedding(ProviderKind::Ollama, "embed:small", 1024)),
            decision: None,
            ollama_base_url: Some(BaseUrl::try_from(String::from("http://gpu-box:11434")).unwrap()),
        },
        SetupPlan {
            chat: Some(choice(ProviderKind::Anthropic, "claude-x")),
            ..SetupPlan::default()
        },
        SetupPlan {
            chat: Some(choice(ProviderKind::OpenAi, "gpt-x")),
            embedding: Some(embedding(
                ProviderKind::OpenAi,
                "text-embedding-3-small",
                1536,
            )),
            decision: None,
            ollama_base_url: None,
        },
        SetupPlan {
            chat: Some(choice(ProviderKind::Bedrock, "us.anthropic.claude-x")),
            embedding: Some(embedding(
                ProviderKind::Bedrock,
                "amazon.titan-embed-text-v2:0",
                1024,
            )),
            decision: None,
            ollama_base_url: None,
        },
    ];
    for plan in plans {
        let (text, changes) = plan.apply(None).unwrap();
        let config = Config::parse(&text).unwrap();
        let chat = plan.chat.as_ref().unwrap();
        assert_eq!(
            config.general.chat_model.map(|m| m.to_string()),
            Some(format!("{}/{}", chat.kind.name(), chat.model)),
            "{text}"
        );
        assert_eq!(
            config.embedding.dimension,
            plan.embedding.as_ref().map(|e| e.dimension),
            "{text}"
        );
        assert_eq!(config.providers.len(), plan.providers().len(), "{text}");
        assert!(text.starts_with("# Written by `quack init`"), "{text}");
        assert!(
            changes.iter().any(|c| c.starts_with("adds [providers.")),
            "{changes:?}"
        );
    }
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn a_new_file_names_the_key_variable_and_only_the_providers_used() {
    let plan = SetupPlan {
        chat: Some(choice(ProviderKind::Anthropic, "claude-x")),
        embedding: Some(embedding(ProviderKind::Ollama, "embed:small", 1024)),
        decision: None,
        ollama_base_url: None,
    };
    let (text, _) = plan.apply(None).unwrap();
    assert!(
        text.contains("api_key_env = \"ANTHROPIC_API_KEY\""),
        "{text}"
    );
    assert!(
        !text.contains("openai") && !text.contains("bedrock"),
        "{text}"
    );
    // The default Ollama address is left to the default.
    assert!(!text.contains("base_url"), "{text}");
    assert_eq!(
        plan.providers(),
        [ProviderKind::Ollama, ProviderKind::Anthropic]
    );
}

/// A file someone wrote by hand, with comments, other settings, and a
/// provider under its own name.
const EXISTING: &str = r#"# my quack setup
[general]
chat_model = "local/old:7b"   # the small one
default_workspace = "research"

[embedding]
model = "local/nomic-embed-text"
dimension = 768
query_prefix = "search_query: "

[providers.local]
type = "ollama"
max_concurrent_requests = 2

[analysis]
effort = "high"
"#;

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn editing_keeps_every_other_setting_and_comment_and_reuses_a_matching_provider() {
    let plan = SetupPlan {
        chat: Some(choice(ProviderKind::Ollama, "chat:8b")),
        embedding: Some(embedding(ProviderKind::Ollama, "embed:small", 1024)),
        decision: None,
        ollama_base_url: None,
    };
    let (text, changes) = plan.apply(Some(EXISTING)).unwrap();
    let config = Config::parse(&text).unwrap();
    assert_eq!(
        config.general.chat_model.map(|m| m.to_string()).as_deref(),
        Some("local/chat:8b"),
        "{text}"
    );
    assert!(text.starts_with("# my quack setup\n"), "{text}");
    assert!(
        text.contains("chat_model = \"local/chat:8b\"   # the small one"),
        "{text}"
    );
    assert!(text.contains("default_workspace = \"research\""), "{text}");
    assert!(text.contains("query_prefix = \"search_query: \""), "{text}");
    assert!(text.contains("max_concurrent_requests = 2"), "{text}");
    assert!(text.contains("effort = \"high\""), "{text}");
    assert!(text.contains("dimension = 1024"), "{text}");
    assert!(!text.contains("[providers.ollama]"), "{text}");
    assert_eq!(
        changes,
        [
            "[general].chat_model = \"local/chat:8b\" (was \"local/old:7b\")",
            "[embedding].model = \"local/embed:small\" (was \"local/nomic-embed-text\")",
            "[embedding].dimension = 1024 (was 768)",
        ]
    );
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn editing_adds_a_missing_provider_and_leaves_what_it_does_not_choose() {
    let plan = SetupPlan {
        chat: Some(choice(ProviderKind::Anthropic, "claude-x")),
        embedding: None,
        decision: None,
        ollama_base_url: None,
    };
    let (text, changes) = plan.apply(Some(EXISTING)).unwrap();
    let config = Config::parse(&text).unwrap();
    assert_eq!(
        config.general.chat_model.map(|m| m.to_string()).as_deref(),
        Some("anthropic/claude-x")
    );
    assert_eq!(
        config.embedding.model.map(|m| m.to_string()).as_deref(),
        Some("local/nomic-embed-text")
    );
    assert!(config.providers.contains_key("local"), "{text}");
    assert!(config.providers.contains_key("anthropic"), "{text}");
    assert_eq!(
        changes,
        [
            "[general].chat_model = \"anthropic/claude-x\" (was \"local/old:7b\")",
            "adds [providers.anthropic]",
        ]
    );
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn choosing_what_the_file_already_has_changes_nothing() {
    let plan = SetupPlan {
        chat: Some(choice(ProviderKind::Ollama, "old:7b")),
        embedding: Some(embedding(ProviderKind::Ollama, "nomic-embed-text", 768)),
        decision: None,
        ollama_base_url: None,
    };
    let (text, changes) = plan.apply(Some(EXISTING)).unwrap();
    assert!(changes.is_empty(), "{changes:?}");
    assert_eq!(text, EXISTING);
}

#[test]
fn a_section_of_the_same_name_for_another_provider_is_refused() {
    let existing = "[providers.anthropic]\ntype = \"openai\"\nbase_url = \"https://gateway.example.com/v1\"\nauth = \"api-key\"\napi_key_env = \"GATEWAY_KEY\"\n";
    let plan = SetupPlan {
        chat: Some(choice(ProviderKind::Anthropic, "claude-x")),
        ..SetupPlan::default()
    };
    let error = plan
        .apply(Some(existing))
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        error.contains("[providers.anthropic] is already a different provider"),
        "{error}"
    );
}

#[test]
fn a_file_that_is_not_toml_is_not_edited() {
    let plan = SetupPlan {
        chat: Some(choice(ProviderKind::Ollama, "chat:8b")),
        ..SetupPlan::default()
    };
    assert!(plan.apply(Some("[general\nchat_model = 1")).is_err());
    assert!(Current::of("[general").is_err());
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn current_reads_the_models_the_file_names() {
    assert_eq!(
        Current::of(EXISTING).unwrap(),
        Current {
            chat_model: Some(String::from("local/old:7b")),
            embedding_model: Some(String::from("local/nomic-embed-text")),
            decision_model: None,
        }
    );
    assert_eq!(
        Current::of("[decision]\nmodel = \"local/laya\"\n")
            .unwrap()
            .decision_model
            .as_deref(),
        Some("local/laya")
    );
    assert_eq!(Current::of("").unwrap(), Current::default());
}

#[test]
fn byte_sizes_read_as_people_write_them() {
    assert_eq!(ByteSize(13_000_000_000).to_string(), "13 GB");
    assert_eq!(ByteSize(5_200_000_000).to_string(), "5.2 GB");
    assert_eq!(ByteSize(639_000_000).to_string(), "639 MB");
    assert_eq!(ByteSize(1_500).to_string(), "1.5 KB");
    assert_eq!(ByteSize(512).to_string(), "512 B");
}

#[test]
fn provider_kinds_round_trip_their_names() {
    for kind in ProviderKind::ALL {
        assert_eq!(ProviderKind::named(kind.name()), Some(kind));
    }
    assert_eq!(ProviderKind::named("gateway"), None);
}

#[test]
fn an_empty_or_blank_variable_counts_as_unset() {
    let env = Environment::from_lookup(
        |name| match name {
            "ANTHROPIC_API_KEY" => Some(String::new()),
            "OPENAI_API_KEY" => Some(String::from("  \t")),
            "OLLAMA_HOST" => Some(String::from(" ")),
            "AWS_PROFILE" => Some(String::from(" dev ")),
            _ => None,
        },
        None,
    );
    assert_eq!(env.anthropic_key, None);
    assert_eq!(env.openai_key, None);
    assert_eq!(env.ollama_host, None);
    assert_eq!(env.aws_access_key, None);
    assert_eq!(env.aws_profile.as_deref(), Some("dev"));
    assert_eq!(
        env.hosted(ProviderKind::Anthropic).to_string(),
        "✗ Anthropic: ANTHROPIC_API_KEY is not set"
    );
    assert_eq!(
        env.ollama_base_url()
            .map(|url| url.to_string())
            .ok()
            .as_deref(),
        Some("http://localhost:11434")
    );
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn a_decision_model_is_written_beside_the_others_and_names_its_provider() {
    let plan = SetupPlan {
        decision: Some(choice(ProviderKind::Ollama, "laya")),
        ..SetupPlan::default()
    };
    assert!(!plan.is_empty());
    assert_eq!(plan.providers(), [ProviderKind::Ollama]);
    let (text, changes) = plan.apply(None).unwrap();
    assert!(text.contains("[decision]"), "{text}");
    let config = Config::parse(&text).unwrap();
    assert_eq!(
        config
            .decision
            .model
            .as_ref()
            .map(ToString::to_string)
            .as_deref(),
        Some("ollama/laya")
    );
    assert!(
        changes.iter().any(|c| c.contains("[decision].model")),
        "{changes:?}"
    );

    let (kept, changes) = plan
        .apply(Some(
            "[decision]\nmodel = \"ollama/laya\"\n[providers.ollama]\ntype = \"ollama\"\n",
        ))
        .unwrap();
    assert!(changes.is_empty(), "{changes:?}");
    assert!(kept.contains("ollama/laya"));
}
