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

#[expect(clippy::unwrap_used, reason = "test")]
fn model(spec: &str) -> ModelSpec {
    spec.parse().unwrap()
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn each_provider_type_writes_a_file_quack_loads() {
    let plans = [
        SetupPlan {
            chat_model: Some(model("ollama/chat:8b")),
            embedding: Some(EmbeddingChoice {
                model: model("ollama/embed:small"),
                dimension: Dimension::new(1024),
            }),
            ollama_base_url: Some(BaseUrl::try_from(String::from("http://gpu-box:11434")).unwrap()),
        },
        SetupPlan {
            chat_model: Some(model("anthropic/claude-x")),
            ..SetupPlan::default()
        },
        SetupPlan {
            chat_model: Some(model("openai/gpt-x")),
            embedding: Some(EmbeddingChoice {
                model: model("openai/text-embedding-3-small"),
                dimension: Dimension::new(1536),
            }),
            ollama_base_url: None,
        },
        SetupPlan {
            chat_model: Some(model("bedrock/us.anthropic.claude-x")),
            embedding: Some(EmbeddingChoice {
                model: model("bedrock/amazon.titan-embed-text-v2:0"),
                dimension: Dimension::new(1024),
            }),
            ollama_base_url: None,
        },
    ];
    for plan in plans {
        let text = plan.toml().unwrap();
        let config = Config::parse(&text).unwrap();
        assert_eq!(
            config.general.chat_model.as_ref().map(ToString::to_string),
            plan.chat_model.as_ref().map(ToString::to_string),
            "{text}"
        );
        assert_eq!(
            config.embedding.dimension,
            plan.embedding.as_ref().map(|e| e.dimension),
            "{text}"
        );
        assert_eq!(
            config.providers.len(),
            plan.providers().unwrap().len(),
            "{text}"
        );
    }
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn the_file_names_the_key_variable_and_only_the_providers_used() {
    let plan = SetupPlan {
        chat_model: Some(model("anthropic/claude-x")),
        embedding: Some(EmbeddingChoice {
            model: model("ollama/embed:small"),
            dimension: Dimension::new(1024),
        }),
        ollama_base_url: None,
    };
    let text = plan.toml().unwrap();
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
        plan.providers().unwrap(),
        [ProviderKind::Ollama, ProviderKind::Anthropic]
    );
}

#[test]
fn a_model_on_another_provider_is_refused() {
    let plan = SetupPlan {
        chat_model: Some(model("gateway/x")),
        ..SetupPlan::default()
    };
    let error = plan
        .toml()
        .map(|_| ())
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        error.contains("quack init sets up ollama, anthropic, openai, bedrock"),
        "{error}"
    );
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
