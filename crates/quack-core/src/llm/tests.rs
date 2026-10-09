use super::*;
use crate::analysis::events;
use crate::analysis::rerank::{RerankOutcome, Reranked, Reranker, apply};
use crate::config::BedrockConfig;
use crate::embedding::Dimension;
use crate::ids::{ChunkId, ClassId, DocumentId, RelationId};
use crate::ingestion::parser::SectionKind;
use crate::ontology::Relation;
use crate::storage::workspace::{ChunkSearchResult, Ranks, WorkspaceDb};
use crate::storage::writer::Writer;

fn parse(toml_text: &str) -> Config {
    match Config::parse(toml_text) {
        Ok(c) => c,
        Err(e) => panic_config(&e.to_string()),
    }
}

#[expect(clippy::panic, reason = "test helper: config fixtures must parse")]
fn panic_config(msg: &str) -> Config {
    panic!("fixture config failed to parse: {msg}");
}

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

#[test]
fn the_catalog_lists_each_provider_and_says_why_one_cannot() {
    let name = |n: &str| {
        n.parse::<ProviderName>()
            .unwrap_or_else(|e| fail(&e.to_string()))
    };
    let many: Vec<_> = (0..42)
        .map(|i| rig::model::ModelInfo::from_id(format!("m{i}")))
        .collect();
    let catalog = ModelCatalog(vec![
        (
            name("local"),
            Ok(ProviderModels::from(rig::model::ModelList::new(vec![
                rig::model::ModelInfo::from_id("gpt-oss:20b"),
                rig::model::ModelInfo::from_id("qwen3-embedding:0.6b"),
            ]))),
        ),
        (
            name("gateway"),
            Ok(ProviderModels::from(rig::model::ModelList::new(many))),
        ),
        (name("empty"), Ok(ProviderModels::default())),
        (name("down"), Err(String::from("connection refused"))),
    ]);
    let text = catalog.to_string();
    assert!(
        text.starts_with(
            "Models each provider lists:\n  local: gpt-oss:20b, qwen3-embedding:0.6b\n"
        ),
        "{text}"
    );
    assert!(text.contains(", m39, and 2 more\n"), "{text}");
    assert!(text.contains("\n  empty: none\n"), "{text}");
    assert!(text.ends_with("\n  down: connection refused"), "{text}");
    assert_eq!(
        ModelCatalog(Vec::new()).to_string(),
        "No providers are configured."
    );
}

#[test]
fn a_bare_ollama_name_finds_its_latest_tag() {
    let models = ProviderModels::from(rig::model::ModelList::new(vec![
        rig::model::ModelInfo::from_id("llama3.1:latest"),
        rig::model::ModelInfo::from_id("gpt-oss:20b"),
    ]));
    assert!(models.get("llama3.1").is_some());
    assert!(models.get("gpt-oss:20b").is_some());
    assert!(models.get("gpt-oss").is_none());
    assert_eq!(models.closest("gpt-os:20b").first(), Some(&"gpt-oss:20b"));
}

/// Graph extraction sends the ontology's schema as the provider's
/// structured output, so the model is held to its class and relation
/// ids: Ollama's `format`, Chat Completions' strict `response_format`.
#[tokio::test]
async fn graph_extraction_sends_the_ontology_schema() {
    Egress::scope(Some(Egress::NoWorkspace), async {
    let keyed = "auth = \"api-key\"\napi_key_env = \"CARGO_PKG_NAME\"\n";
    for (provider, field) in [
        ("type = \"ollama\"\n", r#""format":{"#),
        (
            "type = \"openai\"\napi = \"chat-completions\"\n",
            r#""response_format":{"type":"json_schema","json_schema":{"#,
        ),
    ] {
        let (root, seen) = capture_one().await;
        let auth = if provider.contains("openai") {
            keyed
        } else {
            ""
        };
        let config = parse(&format!(
            "[general]\nchat_model = \"p/m\"\n[providers.p]\n{provider}{auth}base_url = \"{root}\"\n"
        ));
        let mut ontology = Ontology::default();
        ontology.relations.push(Relation {
            id: RelationId::from("ships_to"),
            label: None,
            description: None,
            domain: ClassId::from("entity"),
            range: ClassId::from("entity"),
        });
        let extractor = graph_extractor(&config, &ontology)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(extractor.extract("Orgenics ships to Kenya.").await.is_err());
        let request = seen.await.unwrap_or_else(|e| fail(&e.to_string()));
        assert!(request.contains(field), "{provider}: {request}");
        assert!(
            request.contains(r#""enum":["mentions","ships_to"]"#)
                && request.contains(r#""enum":["entity"]"#),
            "{provider}: {request}"
        );
    }
    })
    .await;
}

/// A small schema the wire tests send.
fn test_schema() -> Schema {
    schema_for!(RerankAnswer)
}

/// One request's head and body, read off a loopback socket that
/// answers 400 (the call itself is not the point).
async fn capture_one() -> (String, tokio::task::JoinHandle<String>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let addr = listener
        .local_addr()
        .unwrap_or_else(|e| fail(&e.to_string()));
    let seen = tokio::spawn(async move {
        let Ok((mut socket, _)) = listener.accept().await else {
            return String::new();
        };
        let mut read = Vec::new();
        let mut buf = [0_u8; 8192];
        // Until the body the Content-Length names has arrived.
        while let Ok(n) = socket.read(&mut buf).await {
            if n == 0 {
                break;
            }
            read.extend_from_slice(buf.get(..n).unwrap_or_default());
            let text = String::from_utf8_lossy(&read).to_string();
            if let Some((head, body)) = text.split_once("\r\n\r\n") {
                let length = head
                    .lines()
                    .find_map(|l| {
                        l.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap_or_default())
                    })
                    .unwrap_or_default();
                if body.len() >= length {
                    break;
                }
            }
        }
        drop(
            socket
                .write_all(
                    b"HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
                )
                .await,
        );
        String::from_utf8_lossy(&read).to_string()
    });
    (format!("http://{addr}"), seen)
}

fn rerank_hit(n: u32) -> ChunkSearchResult {
    ChunkSearchResult {
        id: ChunkId::from(format!("c{n}")),
        content: format!("passage {n}"),
        document_id: DocumentId::from("d"),
        chunk_index: n,
        filename: String::from("f.md"),
        heading: None,
        page: None,
        score: 1.0,
        kind: SectionKind::Body,
        locator: None,
        ingested_at: jiff::civil::DateTime::constant(2026, 10, 5, 0, 0, 0, 0),
        ranks: Ranks::default(),
    }
}

/// The model reranker is a background call: like graph extraction, the
/// ontology document pass, and history summaries, it runs at
/// `[analysis].background_effort`, never the chat turn's `effort`. Built
/// through `ChatClient::schema_call` (the same route its siblings take),
/// so the divergent `background_effort` reaches the rerank request.
#[tokio::test]
async fn the_model_reranker_uses_background_effort_not_turn_effort() {
    Egress::scope(Some(Egress::NoWorkspace), async {
        let (root, seen) = capture_one().await;
        let config = parse(&format!(
            "[general]\nchat_model = \"p/gpt-5.6-sol\"\n\
         [analysis]\neffort = \"xhigh\"\nbackground_effort = \"low\"\n\
         [retrieval]\nrerank = \"model\"\n\
         [providers.p]\ntype = \"openai\"\napi = \"chat-completions\"\n\
         auth = \"api-key\"\napi_key_env = \"CARGO_PKG_NAME\"\nbase_url = \"{root}\"\n"
        ));
        // Build the rerank one-shot the way `dispatch` does: through
        // `schema_call`, which builds the model at `background_effort`.
        let call = {
            let chat = config
                .chat_model_ref()
                .unwrap_or_else(|e| fail(&e.to_string()));
            let client = ChatClient::build(&config, &chat)
                .await
                .unwrap_or_else(|e| fail(&e.to_string()));
            let settings = config.model_settings(chat);
            client
                .schema_call::<RerankAnswer>(
                    chat.model,
                    settings,
                    Task {
                        preamble: RERANK_PROMPT,
                        timeout: RERANK_TIMEOUT,
                        label: "rerank",
                    },
                    schema_for!(RerankAnswer),
                )
                .unwrap_or_else(|e| fail(&e.to_string()))
        };
        let reranker = Reranker::model(call);
        // Two candidates, so `apply` makes the ranking call rather than skip.
        let Reranked { outcome, .. } = apply(
            &reranker,
            "which passage?",
            vec![rerank_hit(1), rerank_hit(2)],
            1,
        )
        .await;
        assert!(
            matches!(&outcome, RerankOutcome::Failed(_)),
            "the reranker must have made its call: {outcome:?}"
        );
        let request = seen.await.unwrap_or_else(|e| fail(&e.to_string()));
        let body = request.split("\r\n\r\n").nth(1).unwrap_or_default();
        assert!(
            body.contains(r#""reasoning_effort":"low""#),
            "the rerank call must use background_effort (\"low\"), not the turn effort: {request}"
        );
        assert!(
            !body.contains("xhigh"),
            "the turn effort leaked into the rerank call: {request}"
        );
    })
    .await;
}

/// A loopback Ollama that answers every request with `stream`, an
/// NDJSON chat stream, so a test can script how the model stops.
async fn scripted_ollama(stream: &'static str) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let addr = listener
        .local_addr()
        .unwrap_or_else(|e| fail(&e.to_string()));
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let mut read = Vec::new();
            let mut buf = [0_u8; 8192];
            while let Ok(n) = socket.read(&mut buf).await {
                if n == 0 {
                    break;
                }
                read.extend_from_slice(buf.get(..n).unwrap_or_default());
                let text = String::from_utf8_lossy(&read).to_string();
                if let Some((head, body)) = text.split_once("\r\n\r\n") {
                    let length = head
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap_or_default())
                        })
                        .unwrap_or_default();
                    if body.len() >= length {
                        break;
                    }
                }
            }
            let reply = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{stream}",
                stream.len()
            );
            drop(socket.write_all(reply.as_bytes()).await);
        }
    });
    format!("http://{addr}")
}

/// A model that thinks until it hits the output limit, answering nothing.
const THOUGHT_ONLY: &str = concat!(
    r#"{"model":"m","created_at":"2026-01-01T00:00:00Z","message":{"role":"assistant","content":"","thinking":"Let me work through every table first."},"done":false}"#,
    "\n",
    r#"{"model":"m","created_at":"2026-01-01T00:00:00Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"length","prompt_eval_count":10,"eval_count":5}"#,
    "\n",
);

/// A model that starts answering and is cut off at the output limit.
const CUT_SHORT: &str = concat!(
    r#"{"model":"m","created_at":"2026-01-01T00:00:00Z","message":{"role":"assistant","content":"The regions are north, south"},"done":false}"#,
    "\n",
    r#"{"model":"m","created_at":"2026-01-01T00:00:00Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"length","prompt_eval_count":10,"eval_count":5}"#,
    "\n",
);

/// A one-shot call (extraction, reranking) whose answer the output limit
/// cut, or never let start, is refused by name rather than handed on as
/// half a JSON document or as rig's advice to raise a `max_tokens`
/// quack has no setting for.
#[tokio::test]
async fn a_one_shot_answer_cut_at_the_output_limit_is_refused() {
    Egress::scope(Some(Egress::NoWorkspace), async {
    for stream in [THOUGHT_ONLY, CUT_SHORT] {
        let root = scripted_ollama(stream).await;
        let config = parse(&format!(
            "[general]\nchat_model = \"o/m\"\n[providers.o]\ntype = \"ollama\"\nbase_url = \"{root}\"\n"
        ));
        let chat = config
            .chat_model_ref()
            .unwrap_or_else(|e| fail(&e.to_string()));
        let client = ChatClient::build(&config, &chat)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let answer = client
            .schema_call::<serde_json::Value>(
                "m",
                ModelSettings::default(),
                Task {
                    preamble: "Answer.",
                    timeout: Duration::from_secs(10),
                    label: "graph extraction",
                },
                test_schema(),
            )
            .unwrap_or_else(|e| fail(&e.to_string()))
            .answer("hello")
            .await;
        let message = answer.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(
            message.contains(
                "the graph extraction answer was cut off at the model's output limit"
            ) && message.contains("OLLAMA_CONTEXT_LENGTH")
                && !message.contains("max_tokens for this request"),
            "{message}"
        );
    }
    })
    .await;
}

/// An agent turn the output limit stopped says so in quack's words: no
/// answer at all points at the Ollama window setting, and a partial one
/// is kept with a note that it was cut off.
#[tokio::test]
async fn a_turn_cut_at_the_output_limit_says_so() {
    Box::pin(Egress::scope(Some(Egress::NoWorkspace), async {
    for (stream, kept, note) in [
        (
            THOUGHT_ONLY,
            "",
            "The model reached its output limit before it answered.",
        ),
        (
            CUT_SHORT,
            "The regions are north, south",
            "The answer was cut off at the model's output limit.",
        ),
    ] {
        let root = scripted_ollama(stream).await;
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = parse(&format!(
            "[general]\nchat_model = \"o/m\"\n[providers.o]\ntype = \"ollama\"\nbase_url = \"{root}\"\n"
        ));
        config.general.data_dir = dir.path().to_path_buf();
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        let session = sessions::create_session(&db, "o/m", sessions::ChatMode::Chat, None)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
        let reader_db = ReaderDb::open(&db, config.analysis.reader_pool_size).await;
        let (sink, _events) = events::channel();
        let response = TurnRequest {
            db: Arc::clone(&db),
            reader_db,
            session_id: &session.id,
            policy: WritePolicy::Deny,
            message: "list the regions",
            documents: &[],
            sink,
            cancel: CancellationToken::new(),
        }
        .run(&config)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            response.content.starts_with(kept)
                && response.content.contains(note)
                && response.content.contains("OLLAMA_CONTEXT_LENGTH")
                && !response.content.contains("max_tokens for this request"),
            "{}",
            response.content
        );
    }
    }))
    .await;
}

/// The whole chat path for Bedrock's OpenAI-compatible APIs, up to the
/// wire: the endpoint's path, a `SigV4` header for its service in place
/// of rig's bearer, and, for Responses, `store: false`.
#[tokio::test]
async fn bedrock_openai_apis_send_signed_requests_to_the_endpoint_path() {
    Egress::scope(Some(Egress::NoWorkspace), async {
        for (provider_type, api, path, service) in [
            (
                ProviderType::BedrockMantle,
                BedrockApi::Responses,
                "POST /v1/responses ",
                "/bedrock-mantle/aws4_request",
            ),
            (
                ProviderType::BedrockMantle,
                BedrockApi::ChatCompletions,
                "POST /v1/chat/completions ",
                "/bedrock-mantle/aws4_request",
            ),
            (
                ProviderType::Bedrock,
                BedrockApi::Responses,
                "POST /openai/v1/responses ",
                "/us-west-2/bedrock/aws4_request",
            ),
        ] {
            let (root, seen) = capture_one().await;
            let bedrock = BedrockConfig { api, region: None };
            let provider = ProviderConfig {
                bedrock: Some(bedrock.clone()),
                headers: Some(std::collections::BTreeMap::from([(
                    String::from("X-Gateway-Team"),
                    String::from("quack"),
                )])),
                ..ProviderConfig::new(provider_type)
            };
            let Some(endpoint) = provider_type.bedrock_endpoint() else {
                fail("a Bedrock type")
            };
            let name: ProviderName = "wire-test"
                .parse()
                .unwrap_or_else(|e: Error| fail(&e.to_string()));
            let session = bedrock::Session::for_test(endpoint, bedrock, &root, "us-west-2");
            let client = ChatClient::bedrock(&session, &name, &provider)
                .unwrap_or_else(|e| fail(&e.to_string()));
            let answer = client
                .schema_call::<serde_json::Value>(
                    "openai.gpt-oss-120b",
                    ModelSettings::default(),
                    Task {
                        preamble: "Answer.",
                        timeout: Duration::from_secs(10),
                        label: "wire test",
                    },
                    test_schema(),
                )
                .unwrap_or_else(|e| fail(&e.to_string()))
                .answer("hello")
                .await;
            assert!(answer.is_err(), "the server answers 400");
            let request = seen.await.unwrap_or_else(|e| fail(&e.to_string()));
            assert!(request.starts_with(path), "{api}: {request}");
            let lower = request.to_ascii_lowercase();
            assert!(
                lower.contains("authorization: aws4-hmac-sha256 credential=akidexample/"),
                "{request}"
            );
            assert!(request.contains(service), "{request}");
            assert!(!lower.contains("bearer"), "{request}");
            assert!(lower.contains("x-gateway-team: quack"), "{request}");
            assert!(request.contains("openai.gpt-oss-120b"), "{request}");
            if api == BedrockApi::Responses {
                assert!(request.contains(r#""store":false"#), "{request}");
            }
        }
    })
    .await;
}

/// A provider's `headers` reach the wire on every chat client, beside
/// the credential.
#[tokio::test]
async fn provider_headers_are_sent_beside_the_credential() {
    Egress::scope(Some(Egress::NoWorkspace), async {
        // Cargo sets this variable for every test run.
        let keyed = "auth = \"api-key\"\napi_key_env = \"CARGO_PKG_NAME\"\n";
        // Anthropic calls need a Claude model, which gets `max_tokens`.
        for (provider, auth, model, path) in [
            ("type = \"ollama\"\n", "", "m", "POST /api/chat "),
            (
                "type = \"openai\"\napi = \"chat-completions\"\n",
                keyed,
                "m",
                "POST /chat/completions ",
            ),
            (
                "type = \"openai\"\napi = \"responses\"\n",
                keyed,
                "m",
                "POST /responses ",
            ),
            (
                "type = \"anthropic\"\n",
                keyed,
                "claude-sonnet-5",
                "POST /v1/messages ",
            ),
        ] {
            let (root, seen) = capture_one().await;
            let config = parse(&format!(
                "[general]\nchat_model = \"p/{model}\"\n[providers.p]\n{provider}{auth}\
             base_url = \"{root}\"\nheaders = {{ \"X-Gateway-Team\" = \"quack\" }}\n"
            ));
            let chat = config
                .chat_model_ref()
                .unwrap_or_else(|e| fail(&e.to_string()));
            let client = ChatClient::build(&config, &chat)
                .await
                .unwrap_or_else(|e| fail(&e.to_string()));
            let answer = client
                .schema_call::<serde_json::Value>(
                    model,
                    ModelSettings::default(),
                    Task {
                        preamble: "Answer.",
                        timeout: Duration::from_secs(10),
                        label: "wire test",
                    },
                    test_schema(),
                )
                .unwrap_or_else(|e| fail(&e.to_string()))
                .answer("hello")
                .await;
            assert!(answer.is_err(), "the server answers 400");
            let request = seen.await.unwrap_or_else(|e| fail(&e.to_string()));
            let lower = request.to_ascii_lowercase();
            assert!(request.starts_with(path), "{provider}: {request}");
            assert!(lower.contains("x-gateway-team: quack"), "{request}");
            assert_eq!(lower.contains("quack-core"), !auth.is_empty(), "{request}");
        }
    })
    .await;
}

/// With `auth = "oauth"`, the request carries `Authorization: Bearer
/// <token>` and no `x-api-key` header; with `auth = "api-key"`, it
/// carries `x-api-key` and no `Authorization` header.
#[tokio::test]
async fn anthropic_sends_an_oauth_token_as_a_bearer() {
    Egress::scope(Some(Egress::NoWorkspace), async {
        let oauth = "auth = \"oauth\"\n[providers.p.oauth]\n\
                 issuer_url = \"http://127.0.0.1:9\"\nclient_id = \"c\"\n";
        let keyed = "auth = \"api-key\"\napi_key_env = \"CARGO_PKG_NAME\"\n";
        for (auth, bearer) in [(oauth, true), (keyed, false)] {
            let (root, seen) = capture_one().await;
            let config = parse(&format!(
                "[general]\nchat_model = \"p/claude-sonnet-5\"\n[providers.p]\n\
             type = \"anthropic\"\nbase_url = \"{root}\"\n{auth}"
            ));
            let chat = config
                .chat_model_ref()
                .unwrap_or_else(|e| fail(&e.to_string()));
            let client = anthropic_client(chat.provider_name, chat.provider, "tok-1")
                .unwrap_or_else(|e| fail(&e.to_string()));
            let answer = ChatClient::Anthropic(client)
                .schema_call::<serde_json::Value>(
                    "claude-sonnet-5",
                    ModelSettings::default(),
                    Task {
                        preamble: "Answer.",
                        timeout: Duration::from_secs(10),
                        label: "wire test",
                    },
                    test_schema(),
                )
                .unwrap_or_else(|e| fail(&e.to_string()))
                .answer("hello")
                .await;
            assert!(answer.is_err(), "the server answers 400");
            let request = seen.await.unwrap_or_else(|e| fail(&e.to_string()));
            let lower = request.to_ascii_lowercase();
            assert!(request.starts_with("POST /v1/messages "), "{request}");
            assert_eq!(lower.contains("authorization: "), bearer, "{request}");
            assert_eq!(
                lower.contains("authorization: bearer tok-1\r\n"),
                bearer,
                "{request}"
            );
            assert_eq!(lower.contains("x-api-key"), !bearer, "{request}");
            assert!(lower.contains("anthropic-version: "), "{request}");
        }
    })
    .await;
}

/// A background call carries what its API is asked on every request: the
/// Responses API stores nothing, and Ollama gets no load options, so its
/// server sizes the window and decides how long the model stays loaded.
#[tokio::test]
async fn background_calls_carry_store_and_no_ollama_load() {
    Egress::scope(Some(Egress::NoWorkspace), async {
        let keyed = "auth = \"api-key\"\napi_key_env = \"CARGO_PKG_NAME\"\n";
        for (provider, auth, sent, unsent) in [
            (
                "type = \"openai\"\napi = \"responses\"\n",
                keyed,
                vec![r#""store":false"#],
                vec![],
            ),
            ("type = \"ollama\"\n", "", vec![], vec!["num_ctx", "keep_alive"]),
        ] {
            let (root, seen) = capture_one().await;
            let config = parse(&format!(
                "[general]\nchat_model = \"p/m\"\n[providers.p]\n{provider}{auth}base_url = \"{root}\"\n"
            ));
            let extractor = graph_extractor(&config, &Ontology::default())
                .await
                .unwrap_or_else(|e| fail(&e.to_string()));
            assert!(extractor.extract("Orgenics ships to Kenya.").await.is_err());
            let request = seen.await.unwrap_or_else(|e| fail(&e.to_string()));
            for field in sent {
                assert!(request.contains(field), "{provider}: {field} in {request}");
            }
            for field in unsent {
                assert!(!request.contains(field), "{provider}: {field} in {request}");
            }
        }
    })
    .await;
}

/// The configured effort reaches each API as the field rig writes for it.
#[tokio::test]
async fn effort_goes_out_as_each_api_s_field() {
    Egress::scope(Some(Egress::NoWorkspace), async {
        let keyed = "auth = \"api-key\"\napi_key_env = \"CARGO_PKG_NAME\"\n";
        for (provider, model, effort, fields) in [
            (
                "type = \"anthropic\"\n",
                "claude-opus-5-5",
                "high",
                vec![r#""effort":"high""#, r#""thinking":{"type":"adaptive""#],
            ),
            (
                "type = \"openai\"\napi = \"responses\"\n",
                "gpt-5.6-sol",
                "max",
                vec![r#""effort":"max""#],
            ),
            (
                "type = \"openai\"\napi = \"chat-completions\"\n",
                "corp-reasoner",
                "medium",
                vec![r#""reasoning_effort":"medium""#],
            ),
            ("type = \"ollama\"\n", "gpt-oss:20b", "low", vec![r#""think":"low""#]),
            ("type = \"ollama\"\n", "qwen3:8b", "none", vec![r#""think":false"#]),
        ] {
            let (root, seen) = capture_one().await;
            let auth = if provider.contains("ollama") { "" } else { keyed };
            let config = parse(&format!(
                "[general]\nchat_model = \"p/{model}\"\n[analysis]\nbackground_effort = \"{effort}\"\n\
                 [providers.p]\n{provider}{auth}base_url = \"{root}\"\n"
            ));
            let extractor = graph_extractor(&config, &Ontology::default())
                .await
                .unwrap_or_else(|e| fail(&format!("{model}: {e}")));
            assert!(extractor.extract("Orgenics ships to Kenya.").await.is_err());
            let request = seen.await.unwrap_or_else(|e| fail(&e.to_string()));
            for field in fields {
                assert!(request.contains(field), "{model}: {field} in {request}");
            }
        }
    })
    .await;
}

/// A level rig refuses for the model fails when the chat model is built,
/// before any request, with rig's reason; one it takes builds.
#[tokio::test]
async fn an_effort_rig_refuses_fails_when_the_model_is_built() {
    let keyed = "auth = \"api-key\"\napi_key_env = \"CARGO_PKG_NAME\"\n";
    let built = |provider: &str, model: &str, effort: Effort| {
        let config = parse(&format!(
            "[general]\nchat_model = \"p/{model}\"\n[providers.p]\n{provider}"
        ));
        async move {
            let chat = config
                .chat_model_ref()
                .unwrap_or_else(|e| fail(&e.to_string()));
            ChatClient::without_credential(chat.provider_name, chat.provider)
                .await
                .unwrap_or_else(|e| fail(&e.to_string()))
                .chat_model(chat.model, Some(effort), None)
                .map(drop)
        }
    };
    let anthropic = format!("type = \"anthropic\"\n{keyed}");
    for (provider, model, effort) in [
        (anthropic.as_str(), "claude-opus-5-5", Effort::None),
        (anthropic.as_str(), "claude-opus-5-5", Effort::Minimal),
        ("type = \"ollama\"\n", "gpt-oss:20b", Effort::Minimal),
        ("type = \"ollama\"\n", "gpt-oss:20b", Effort::Xhigh),
    ] {
        let Err(e) = built(provider, model, effort).await else {
            fail(&format!("{model} took {effort}"))
        };
        let message = e.to_string();
        assert!(
            message.contains(&format!("effort \"{effort}\"")),
            "{message}"
        );
        assert!(message.contains(model), "{message}");
    }
    for (provider, model, effort) in [
        (anthropic.as_str(), "claude-opus-5-5", Effort::Max),
        ("type = \"ollama\"\n", "gpt-oss:20b", Effort::High),
    ] {
        assert!(
            built(provider, model, effort).await.is_ok(),
            "{model} {effort}"
        );
    }
}

#[test]
fn display_name_reports_model_ref_or_placeholder() {
    let config = parse("[general]\nchat_model = \"o/llama3\"\n[providers.o]\ntype = \"ollama\"\n");
    assert_eq!(config.chat_model_label(), "o/llama3");
    assert_eq!(Config::default().chat_model_label(), "no chat model");
}

#[tokio::test]
async fn optional_embedding_model_is_none_when_unset() {
    assert!(matches!(
        Embeddings::from_config(&Config::default()).await,
        Ok(None)
    ));
}

#[tokio::test]
async fn required_embedding_model_errors_when_unset() {
    let err = Embeddings::require(&Config::default()).await.err();
    assert!(err.is_some_and(|e| e.to_string().contains("[embedding].model")));
}

#[tokio::test]
async fn api_key_mode_requires_the_env_var_to_be_set() {
    Egress::scope(Some(Egress::NoWorkspace), async {
    let config = parse(
        "[embedding]\nmodel = \"o/e\"\ndimension = 4\n[providers.o]\ntype = \"openai\"\nauth = \"api-key\"\napi_key_env = \"QUACK_TEST_KEY_THAT_IS_UNSET\"\n",
    );
    let err = Embeddings::require(&config).await.err();
    assert!(err.is_some_and(|e| e.to_string().contains("QUACK_TEST_KEY_THAT_IS_UNSET")));
    })
    .await;
}

#[tokio::test]
async fn ollama_embedding_model_builds_without_a_key() {
    Egress::scope(Some(Egress::NoWorkspace), async {
        let config = parse(
            "[embedding]\nmodel = \"o/nomic\"\ndimension = 4\n[providers.o]\ntype = \"ollama\"\n",
        );
        let model = Embeddings::require(&config).await;
        assert!(model.is_ok_and(|m| m.profile().dimension == Dimension::new(4)));
    })
    .await;
}

#[tokio::test]
async fn ollama_embed_requests_carry_a_bounded_window_and_keep_alive() {
    Egress::scope(Some(Egress::NoWorkspace), async {
    let config = parse(
        "[embedding]\nmodel = \"o/nomic\"\ndimension = 4\n[providers.o]\ntype = \"ollama\"\n[ingestion]\nchunk_size_tokens = 3000\n",
    );
    let model = Embeddings::require(&config).await;
    let Some(EmbedModel::Ollama(embedder)) = model.as_ref().ok().map(Embedder::model) else {
        fail("expected the Ollama embedder")
    };
    let body = embedder.request_body(&[String::from("a chunk")]);
    assert_eq!(body.get("model"), Some(&serde_json::json!("nomic")));
    assert_eq!(body.get("input"), Some(&serde_json::json!(["a chunk"])));
    assert_eq!(
        body.get("keep_alive"),
        Some(&serde_json::json!(OLLAMA_KEEP_ALIVE))
    );
    // 3,000 tokens doubled and rounded up to a power of two.
    assert_eq!(
        body.pointer("/options/num_ctx"),
        Some(&serde_json::json!(8192))
    );
    })
    .await;
}

#[test]
fn ollama_running_models_match_bare_and_tagged_names() {
    let running: OllamaRunningModels = serde_json::from_str(
        r#"{"models":[{"name":"gpt-oss:20b","model":"gpt-oss:20b"},{"name":"llama3:latest","model":"llama3:latest"}]}"#,
    )
    .unwrap_or(OllamaRunningModels { models: Vec::new() });
    assert!(running.holds("gpt-oss:20b"));
    assert!(running.holds("llama3"));
    assert!(running.holds("llama3:latest"));
    assert!(!running.holds("gpt-oss"));
    assert!(!running.holds("qwen3:8b"));
    let empty: OllamaRunningModels =
        serde_json::from_str("{}").unwrap_or(OllamaRunningModels { models: Vec::new() });
    assert!(!empty.holds("gpt-oss:20b"));
}

#[test]
fn ollama_embed_window_never_drops_below_the_floor() {
    assert_eq!(OllamaEmbedder::context_window(512), OLLAMA_EMBED_MIN_CTX);
    assert_eq!(OllamaEmbedder::context_window(1024), OLLAMA_EMBED_MIN_CTX);
    assert_eq!(OllamaEmbedder::context_window(1025), 4096);
    assert_eq!(OllamaEmbedder::context_window(u32::MAX), u32::MAX);
}

#[tokio::test]
async fn oauth_provider_without_a_login_needs_auth_for_embeddings_and_chat() {
    Egress::scope(Some(Egress::NoWorkspace), async {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = parse(
        "[embedding]\nmodel = \"az/emb\"\ndimension = 4\n[general]\nchat_model = \"az/gpt\"\n[providers.az]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.az.oauth]\nissuer_url = \"http://127.0.0.1:9\"\nclient_id = \"c\"\n",
    );
    config.general.data_dir = dir.path().to_path_buf();
    let err = Embeddings::require(&config).await.err();
    assert!(err.is_some_and(|e| matches!(e, Error::AuthRequired { .. })));
    let chat = config.chat_model_ref();
    let Ok(chat) = chat else {
        return assert!(chat.is_ok());
    };
    let err = build_openai_client(&config, chat.provider_name, chat.provider)
        .await
        .err();
    assert!(err.is_some_and(|e| matches!(e, Error::AuthRequired { .. })));
    })
    .await;
}

/// A cancelled turn is still a turn (issue #45): the session records
/// the question and a cancelled answer, `TurnComplete` is emitted, and
/// the model is never called (the token is cancelled before the turn
/// starts, and the provider address is unreachable anyway).
#[tokio::test]
async fn cancelled_turns_are_recorded_and_completed() {
    Box::pin(Egress::scope(Some(Egress::NoWorkspace), async {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = parse(
        "[embedding]\nmodel = \"o/e\"\ndimension = 4\n[general]\nchat_model = \"o/m\"\n[providers.o]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\n",
    );
    config.general.data_dir = dir.path().to_path_buf();
    let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
    let session = sessions::create_session(&db, "o/m", sessions::ChatMode::Chat, None)
        .unwrap_or_else(|e| fail(&e.to_string()));
    let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
    let reader_db = ReaderDb::open(&db, config.analysis.reader_pool_size).await;
    let (sink, mut events) = events::channel();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let response = TurnRequest {
        db: Arc::clone(&db),
        reader_db,
        session_id: &session.id,
        policy: WritePolicy::Deny,
        message: "how many storms?",
        documents: &[],
        sink,
        cancel,
    }
    .run(&config)
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(response.cancelled);
    assert_eq!(response.content, CANCELLED_NOTE);
    assert!(
        response.duration_ms.is_some(),
        "a cancelled turn is timed too"
    );
    let last = events.recv().await;
    assert!(
        matches!(&last, Some(AgentEvent::TurnComplete(r)) if r.cancelled && r.duration_ms.is_some()),
        "{last:?}"
    );
    let id = session.id.clone();
    let messages = db
        .run(move |guard| sessions::messages(guard, &id))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let roles: Vec<sessions::MessageRole> = messages.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        vec![
            sessions::MessageRole::User,
            sessions::MessageRole::Assistant
        ]
    );
    assert!(
        messages
            .last()
            .is_some_and(|m| m.content.contains("Cancelled by the user")
                && m.assistant().and_then(|a| a.duration_ms).is_some())
    );
    }))
    .await;
}
