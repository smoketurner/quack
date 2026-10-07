use super::*;

const FULL: &str = r#"
[general]
chat_model = "ollama/llama3.1:8b"

[embedding]
model = "ollama/nomic-embed-text"
dimension = 768


[providers.ollama]
type = "ollama"
base_url = "http://localhost:11434"


[providers.anthropic]
type = "anthropic"
auth = "api-key"
api_key_env = "ANTHROPIC_API_KEY"

[retrieval]
top_k = 3
always_retrieve = true
rerank = "model"
"#;

/// `default_workspace` takes only a name a workspace may have.
#[test]
fn a_default_workspace_no_workspace_may_take_is_refused() {
    let named = |name: &str| {
        toml::from_str::<Config>(&format!("[general]\ndefault_workspace = \"{name}\"\n"))
    };
    assert!(named(" sales ").is_ok_and(|c| c.general.default_workspace.as_str() == "sales"));
    let refused = named("a.b/c")
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        refused.contains(&Error::InvalidWorkspaceName.to_string()),
        "{refused}"
    );
}

#[test]
fn default_config_values() {
    let config = Config::default();
    assert_eq!(config.general.default_workspace.as_str(), "default");
    assert!(config.general.chat_model.is_none());
    assert_eq!(config.ingestion.chunk_size_tokens, 512);
    assert_eq!(config.retrieval.top_k, 8);
    assert_eq!(config.retrieval.rrf_k, 60);
    assert_eq!(config.retrieval.pinned_token_budget, Tokens::new(8000));
    assert!(!config.retrieval.always_retrieve);
    assert_eq!(config.retrieval.rerank, RerankMode::None);
    assert_eq!(config.retrieval.rerank_candidates, 24);
    assert_eq!(config.context.max_tokens, Tokens::new(4000));
    assert_eq!(config.analysis.threads, 4);
    assert_eq!(config.analysis.max_turns, 15);
    assert_eq!(config.analysis.history_token_budget, Tokens::new(32_000));
    assert_eq!(config.ingestion.upload_max_mb, 512);
    assert_eq!(config.ingestion.max_decompressed_mb, 1024);
    assert_eq!(config.ingestion.embedding_concurrency, 2);
    assert_eq!(config.server.bind, "127.0.0.1:8080");
    assert!(!config.server.local);
    assert_eq!(config.server.workers_per_workspace, 1);
    assert!((config.ontology.key_overlap_threshold - 0.8).abs() < f64::EPSILON);
    assert_eq!(config.ontology.enum_max_values, 12);
    assert_eq!(config.ontology.propose_sample_chunks, 200);
    assert_eq!(config.ontology.min_support_documents, 3);
    assert!(err_of("[ontology]\nsample = 1\n").contains("sample"));
    assert_eq!(Config::default().graph.max_traversal_depth, 3);
    assert_eq!(Config::default().server.log_format, LogFormat::Text);
    assert_eq!(Config::default().graph.max_nodes, 200);
    assert!(err_of("[graph]\nenabled = true\n").contains("enabled"));
    assert_eq!(Config::default().import.max_rows, 1_000_000);
    assert!(err_of("[import]\nmax = 1\n").contains("max"));
}

#[test]
fn server_section_parses_and_rejects_unknown_keys() {
    let ok = Config::parse("[server]\nbind = \"0.0.0.0:9000\"\nlocal = true\n");
    assert!(ok.is_ok_and(|c| c.server.bind == "0.0.0.0:9000" && c.server.local));
    assert!(err_of("[server]\nport = 1\n").contains("port"));
}

/// Issue #246: loopback cookies are plain unless the public URL is https
/// or the operator asked for `Secure` everywhere.
#[test]
fn secure_cookies_on_loopback_follow_the_setting_and_the_public_url() {
    let default = Config::parse("");
    assert!(default.is_ok_and(|c| {
        c.server.secure_cookies == SecureCookies::Auto && !c.server.secure_cookies_on_loopback()
    }));
    let always = Config::parse("[server]\nsecure_cookies = \"always\"\n");
    assert!(always.is_ok_and(|c| c.server.secure_cookies_on_loopback()));
    assert!(err_of("[server]\nsecure_cookies = \"never\"\n").contains("never"));
    let oidc = |origin: &str| {
        format!(
            "[server.oidc]\nissuer_url = \"https://idp\"\nclient_id = \"q\"\n\
                 redirect_uri = \"{origin}{}\"\n",
            OidcConfig::CALLBACK_PATH
        )
    };
    let https = Config::parse(&oidc("https://quack.example.com"));
    assert!(https.is_ok_and(|c| c.server.secure_cookies_on_loopback()));
    let http = Config::parse(&oidc("http://127.0.0.1:8080"));
    assert!(http.is_ok_and(|c| !c.server.secure_cookies_on_loopback()));
}

#[test]
#[expect(clippy::unwrap_used, reason = "test asserts Ok")]
fn full_config_parses_and_resolves_models() {
    let config = Config::parse(FULL).unwrap();
    let chat = config.chat_model_ref().unwrap();
    assert_eq!(chat.provider_name, "ollama");
    assert_eq!(chat.model, "llama3.1:8b");
    assert_eq!(chat.provider.provider_type, ProviderType::Ollama);
    assert_eq!(chat.provider.auth.mode(), AuthMode::None);
    assert_eq!(chat.to_string(), "ollama/llama3.1:8b");
    let embed = config.embedding_model_ref().unwrap().unwrap();
    assert_eq!(embed.model, "nomic-embed-text");
    assert_eq!(config.embedding_dimension().ok(), Some(Dimension::new(768)));
    assert_eq!(config.retrieval.top_k, 3);
    assert_eq!(config.retrieval.rerank, RerankMode::Model);
    let anthropic = config.providers.get("anthropic");
    assert!(anthropic.is_some_and(|p| {
        p.provider_type == ProviderType::Anthropic
            && p.auth.api_key_env() == Some("ANTHROPIC_API_KEY")
    }));
}

#[test]
fn model_specs_split_once_when_read() {
    let spec: ModelSpec = "ollama/llama3.1:8b"
        .parse()
        .unwrap_or_else(|e| panic_on(&e));
    assert_eq!(spec.provider(), "ollama");
    assert_eq!(spec.model(), "llama3.1:8b");
    assert_eq!(spec.to_string(), "ollama/llama3.1:8b");
    // The model may itself contain a slash; only the first one splits.
    let nested: ModelSpec = "hf/org/model".parse().unwrap_or_else(|e| panic_on(&e));
    assert_eq!(
        (nested.provider().as_str(), nested.model()),
        ("hf", "org/model")
    );
    for bad in ["llama3", "ollama/", "/m", "bad name/m"] {
        assert!(bad.parse::<ModelSpec>().is_err(), "{bad}");
    }
    assert!(err_of("[general]\nchat_model = \"llama3\"\n").contains("PROVIDER/MODEL"));
}

#[test]
fn base_urls_are_checked_and_trimmed() {
    let url = |text: &str| BaseUrl::try_from(String::from(text)).unwrap_or_else(|e| panic_on(&e));
    let ollama = url("http://gpu-box:11434/v1/");
    assert_eq!(ollama.trimmed(), "http://gpu-box:11434/v1");
    assert_eq!(ollama.root(), "http://gpu-box:11434");
    assert_eq!(
        url("https://api.openai.com/v1").root(),
        "https://api.openai.com"
    );
    for bad in ["localhost:11434", "ftp://h/", "not a url", ""] {
        assert!(BaseUrl::try_from(String::from(bad)).is_err(), "{bad}");
    }
    assert!(
        err_of("[providers.o]\ntype = \"ollama\"\nbase_url = \"localhost:1\"\n")
            .contains("http:// or https://")
    );
    assert_eq!(
        ProviderType::OLLAMA_BASE_URL.as_str(),
        "http://localhost:11434"
    );
}

#[test]
fn cleartext_is_plain_http_off_this_machine() {
    let url = |text: &str| BaseUrl::try_from(String::from(text)).unwrap_or_else(|e| panic_on(&e));
    assert!(url("http://gpu-box:11434").sends_in_cleartext());
    assert!(url("http://10.0.0.5/v1").sends_in_cleartext());
    assert!(!url("http://localhost:11434").sends_in_cleartext());
    assert!(!url("http://127.0.0.1:11434").sends_in_cleartext());
    assert!(!url("http://[::1]:11434").sends_in_cleartext());
    assert!(!url("https://api.openai.com/v1").sends_in_cleartext());
}

#[test]
fn request_limits_are_at_least_one() {
    assert!(
        err_of("[providers.o]\ntype = \"ollama\"\nmax_concurrent_requests = 0\n")
            .contains("nonzero")
    );
    let ollama = ProviderConfig::new(ProviderType::Ollama);
    assert_eq!(ollama.request_limit().get(), 1);
    assert_eq!(
        ProviderConfig::new(ProviderType::Openai)
            .request_limit()
            .get(),
        8
    );
}

/// Every combination of `auth`, `api_key_env`, and an oauth table: the
/// three that make sense read as their `ProviderAuth`, the rest are
/// refused when read.
#[test]
fn provider_auth_is_one_of_three_shapes() {
    let oauth = "[providers.p.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n";
    let case = |auth: &str, key: bool, table: bool| {
        let mut text = format!("[providers.p]\ntype = \"openai\"\nauth = \"{auth}\"\n");
        if key {
            text.push_str("api_key_env = \"K\"\n");
        }
        if table {
            text.push_str(oauth);
        }
        Config::parse(&text).map(|c| {
            c.providers
                .get("p")
                .map(|p| p.auth.mode())
                .unwrap_or_default()
        })
    };
    assert!(case("none", false, false).is_ok_and(|m| m == AuthMode::None));
    assert!(case("api-key", true, false).is_ok_and(|m| m == AuthMode::ApiKey));
    assert!(case("oauth", false, true).is_ok_and(|m| m == AuthMode::Oauth));
    for (auth, key, table) in [
        ("none", true, false),
        ("none", false, true),
        ("api-key", false, false),
        ("api-key", true, true),
        ("oauth", false, false),
        ("oauth", true, true),
    ] {
        assert!(case(auth, key, table).is_err(), "{auth} {key} {table}");
    }
    assert!(
            err_of("[providers.p]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.p.oauth]\nissuer_url = \"\"\nclient_id = \"c\"\n")
                .contains("needs issuer_url")
        );
}

#[expect(clippy::panic, reason = "test failure path")]
fn panic_on(e: &Error) -> ! {
    panic!("{e}")
}

#[test]
fn an_override_rescues_the_file_value_it_replaces() {
    let file = "[general]\nchat_model = \"missing/model\"\n\n\
                    [providers.ollama]\ntype = \"ollama\"\n";
    let none = Overrides::default();
    let rejected = Config::from_contents(Some(file), &none).err();
    assert!(rejected.is_some_and(|e| e.to_string().contains("missing")));
    let rescued = Overrides {
        chat_model: Some(String::from("ollama/gpt-oss:20b")),
        ..Overrides::default()
    };
    let config = Config::from_contents(Some(file), &rescued);
    assert!(config.is_ok_and(|c| {
        c.general
            .chat_model
            .is_some_and(|m| m.to_string() == "ollama/gpt-oss:20b")
    }));
    // An override is still validated: it cannot name a missing provider.
    let bad = Overrides {
        chat_model: Some(String::from("nowhere/x")),
        ..Overrides::default()
    };
    assert!(Config::from_contents(None, &bad).is_err());
}

#[test]
fn provider_names_outside_the_allowed_characters_are_rejected() {
    for bad in [
        "../evil",
        ".hidden",
        "a/b",
        "a\\\\b",
        "",
        "sp ace",
        "azure.openai",
    ] {
        let toml_text = format!("[providers.\"{bad}\"]\ntype = \"ollama\"\n");
        assert!(err_of(&toml_text).contains("provider name"), "{bad:?}");
    }
    let long = "p".repeat(65);
    assert!(long.parse::<ProviderName>().is_err());
    for good in ["ollama", "azure-openai", "corp-gw_2", &"p".repeat(64)] {
        assert!(good.parse::<ProviderName>().is_ok(), "{good}");
    }
    // Unquoted, the dotted form is a table inside provider "azure".
    assert!(
        err_of("[providers.azure.openai]\ntype = \"openai\"\n").contains("openai"),
        "a dotted header is not one provider"
    );
}

#[test]
fn overrides_replace_only_what_they_set() {
    let overrides = Overrides {
        data_dir: Some(PathBuf::from("/srv/quack")),
        bind: Some(String::from("0.0.0.0:9000")),
        ..Overrides::default()
    };
    let config = Config::from_contents(Some("[general]\n"), &overrides);
    assert!(
        config.is_ok_and(|c| c.general.data_dir == Path::new("/srv/quack")
            && c.server.bind == "0.0.0.0:9000"
            && c.general.chat_model.is_none())
    );
}

fn err_of(toml_text: &str) -> String {
    match Config::parse(toml_text) {
        Ok(_) => String::from("<ok>"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn unknown_keys_are_rejected_at_every_level() {
    assert!(err_of("[genral]\nchat_model = \"a/b\"\n").contains("genral"));
    assert!(err_of("[general]\nchat_modle = \"a/b\"\n").contains("chat_modle"));
    assert!(err_of("[providers.o]\ntype = \"ollama\"\nmodel = \"x\"\n").contains("model"));
    assert!(err_of("[retrieval]\ntopk = 1\n").contains("topk"));
    assert!(err_of("[retrieval]\nrerank = \"bge\"\n").contains("bge"));
    let reranker = |provider: &str| {
        err_of(&format!(
            "[retrieval]\nrerank = \"reranker\"\nrerank_model = \"r/bge\"\n[providers.r]\n{provider}"
        ))
    };
    let ollama = reranker("type = \"ollama\"\n");
    assert!(
        ollama.contains("ollama has no rerank endpoint") && ollama.contains("vLLM"),
        "{ollama}"
    );
    assert!(reranker("type = \"openai\"\n").contains("set base_url"));
    assert!(
        err_of("[retrieval]\nrerank = \"reranker\"\n").contains("needs rerank_model"),
        "the mode without its model"
    );
    assert!(err_of("[analysis]\nthread = 1\n").contains("thread"));
}

#[test]
fn unknown_provider_type_is_rejected() {
    assert!(err_of("[providers.o]\ntype = \"vertex\"\n").contains("vertex"));
}

#[test]
fn provider_headers_are_checked_when_read() {
    let base = "[providers.o]\ntype = \"openai\"\nauth = \"api-key\"\napi_key_env = \"K\"\n";
    let headers_of = |text: &str| {
        Config::parse(text)
            .ok()
            .and_then(|c| c.providers.get("o").map(|p| p.headers.clone()))
    };
    assert_eq!(headers_of(base), Some(None));
    let set = format!("{base}headers = {{ \"X-Team\" = \"quack\", \"x-trace\" = \"1\" }}\n");
    assert_eq!(
        headers_of(&set),
        Some(Some(BTreeMap::from([
            (String::from("X-Team"), String::from("quack")),
            (String::from("x-trace"), String::from("1")),
        ])))
    );
    let map = Config::parse(&set)
        .ok()
        .and_then(|c| c.providers.get("o").map(ProviderConfig::header_map))
        .and_then(Result::ok)
        .unwrap_or_default();
    assert_eq!(
        map.get("x-team").and_then(|v| v.to_str().ok()),
        Some("quack")
    );
    assert_eq!(headers_of(&format!("{base}headers = {{}}\n")), Some(None));

    assert!(err_of(&format!("{base}headers = {{ \"X Team\" = \"q\" }}\n")).contains("X Team"));
    let value = err_of(&format!("{base}headers = {{ \"X-Team\" = \"a\\nb\" }}\n"));
    assert!(
        value.contains("X-Team") && !value.contains("a\nb"),
        "{value}"
    );
    assert!(err_of(&format!("{base}headers = {{ \"X-Team\" = 1 }}\n")).contains("headers"));
    for credential in ["Authorization", "authorization", "X-Api-Key"] {
        let err = err_of(&format!("{base}headers = {{ \"{credential}\" = \"k\" }}\n"));
        assert!(err.contains("credential"), "{err}");
    }
}

#[test]
fn converse_takes_no_headers() {
    let bedrock = "[providers.b]\ntype = \"bedrock\"\nheaders = { \"X-Team\" = \"q\" }\n";
    assert!(err_of(bedrock).contains("converse"));
    let chat = format!("{bedrock}api = \"chat-completions\"\n");
    assert!(Config::parse(&chat).is_ok());
}

#[test]
fn openai_takes_chat_completions_or_responses() {
    let api_of = |text: &str| {
        Config::parse(text)
            .ok()
            .and_then(|c| c.providers.get("o").map(ProviderConfig::openai_chat_api))
    };
    let base = "[providers.o]\ntype = \"openai\"\nauth = \"api-key\"\napi_key_env = \"K\"\n";
    // OpenAI itself defaults to Responses; a compatible server to Chat Completions.
    assert_eq!(api_of(base), Some(BedrockApi::Responses));
    let gateway = format!("{base}base_url = \"https://models.example.com/v1\"\n");
    assert_eq!(api_of(&gateway), Some(BedrockApi::ChatCompletions));
    assert_eq!(
        api_of(&format!("{gateway}api = \"responses\"\n")),
        Some(BedrockApi::Responses)
    );
    assert_eq!(
        api_of(&format!("{base}api = \"chat-completions\"\n")),
        Some(BedrockApi::ChatCompletions)
    );
    assert!(err_of(&format!("{base}api = \"converse\"\n")).contains("converse"));
    assert!(
        err_of("[providers.a]\ntype = \"anthropic\"\napi = \"responses\"\n")
            .contains("api is only for")
    );
    assert!(
        err_of("[providers.o]\ntype = \"ollama\"\nregion = \"us-east-1\"\n").contains("region")
    );
}

#[test]
fn effort_is_unset_or_a_known_level() {
    let unset = Config::parse("").map(|c| (c.analysis.effort, c.analysis.background_effort));
    assert!(unset.is_ok_and(|e| e == (None, None)));
    let set = Config::parse("[analysis]\neffort = \"xhigh\"\nbackground_effort = \"low\"\n")
        .map(|c| (c.analysis.effort, c.analysis.background_effort));
    assert!(set.is_ok_and(|e| e == (Some(Effort::Xhigh), Some(Effort::Low))));
    assert_ne!(err_of("[analysis]\neffort = \"extreme\"\n"), "<ok>");
}

#[test]
fn a_model_s_settings_override_its_provider_s_then_analysis_key_by_key() {
    let config = Config::parse(
        "[general]\nchat_model = \"gw/Corp.Reasoner-2\"\n\
             [analysis]\neffort = \"medium\"\nbackground_effort = \"low\"\n\
             [providers.gw]\ntype = \"openai\"\nbase_url = \"https://gw.example\"\n\
             temperature = true\neffort = \"high\"\n\
             [providers.gw.models.\"Corp.Reasoner-2\"]\neffort = \"xhigh\"\n\
             [providers.gw.models.plain]\ntemperature = false\n",
    );
    let config = config.unwrap_or_else(|e| panic_on(&e));
    let chat = config.chat_model_ref().unwrap_or_else(|e| panic_on(&e));
    assert_eq!(
        config.model_settings(chat),
        ModelSettings {
            temperature: Some(true),
            effort: Some(Effort::Xhigh),
            background_effort: Some(Effort::Low),
            images: None,
        }
    );
    assert_eq!(
        chat.provider.model_settings("plain"),
        ModelSettings {
            temperature: Some(false),
            effort: Some(Effort::High),
            background_effort: None,
            images: None,
        }
    );
    assert_eq!(
        chat.provider.model_settings("other"),
        chat.provider.model_defaults
    );
    let bare = ProviderConfig::new(ProviderType::Openai);
    assert_eq!(bare.model_settings("any"), ModelSettings::default());
    let provider = "[providers.gw]\ntype = \"openai\"\n[providers.gw.models.m]\n";
    assert_ne!(err_of(&format!("{provider}temp = false\n")), "<ok>");
    assert_ne!(err_of(&format!("{provider}effort = \"extreme\"\n")), "<ok>");
}

#[test]
fn bedrock_signs_with_the_aws_chain_by_default() {
    let config = Config::parse(
        "[general]\nchat_model = \"aws/us.anthropic.claude-opus-5-5\"\n\
             [providers.aws]\ntype = \"bedrock\"\naws_profile = \"dev-sso\"\nregion = \"us-west-2\"\n",
    );
    assert!(
        config.is_ok_and(|c| c.providers.get("aws").is_some_and(|p| {
            p.provider_type == ProviderType::Bedrock
                && p.auth.mode() == AuthMode::Aws
                && p.auth.aws_profile() == Some("dev-sso")
                && p.bedrock.as_ref().is_some_and(|b| {
                    b.region.as_ref().map(AwsRegion::as_str) == Some("us-west-2")
                        && b.api == BedrockApi::Converse
                })
                && p.request_limit().get() == 8
        }))
    );
    // Neither profile nor region is required: the SDK's chain decides.
    let config = Config::parse("[providers.b]\ntype = \"bedrock\"\nauth = \"aws\"\n");
    assert!(config.is_ok_and(|c| {
        c.providers.get("b").is_some_and(|p| {
            p.auth.aws_profile().is_none() && p.bedrock.as_ref().is_some_and(|b| b.region.is_none())
        })
    }));
    assert_eq!(
        ProviderConfig::new(ProviderType::Bedrock).auth.mode(),
        AuthMode::Aws
    );
    assert!(ProviderType::Bedrock.default_base_url().is_none());
    assert_eq!(
        ProviderType::Bedrock.bedrock_endpoint(),
        Some(BedrockEndpoint::Runtime)
    );
    let mantle = ProviderConfig::new(ProviderType::BedrockMantle);
    assert_eq!(mantle.auth.mode(), AuthMode::Aws);
    assert_eq!(mantle.bedrock.map(|b| b.api), Some(BedrockApi::Responses));
    assert_eq!(
        mantle.provider_type.bedrock_endpoint(),
        Some(BedrockEndpoint::Mantle)
    );
}

#[test]
fn aws_settings_are_refused_where_they_mean_nothing() {
    assert!(
        err_of("[providers.b]\ntype = \"bedrock\"\nauth = \"api-key\"\napi_key_env = \"K\"\n")
            .contains("bedrock")
    );
    assert!(err_of("[providers.b]\ntype = \"bedrock\"\nauth = \"none\"\n").contains("bedrock"));
    assert!(err_of("[providers.o]\ntype = \"openai\"\nauth = \"aws\"\n").contains("aws"));
    assert!(
            err_of("[providers.o]\ntype = \"openai\"\nauth = \"api-key\"\napi_key_env = \"K\"\nregion = \"us-east-1\"\n")
                .contains("region")
        );
    assert!(
        err_of("[providers.o]\ntype = \"ollama\"\naws_profile = \"p\"\n").contains("aws_profile")
    );
    assert!(
        err_of("[providers.b]\ntype = \"bedrock\"\napi_key_env = \"K\"\n").contains("api_key_env")
    );
    assert!(err_of("[providers.b]\ntype = \"bedrock\"\nregion = \"us east\"\n").contains("region"));
    assert!(err_of("[providers.b]\ntype = \"bedrock\"\nregion = \"\"\n").contains("region"));
    assert!(err_of("[providers.o]\ntype = \"ollama\"\napi = \"responses\"\n").contains("api"));
    // The endpoint is the type, never a key of its own.
    assert!(
        err_of("[providers.b]\ntype = \"bedrock\"\nendpoint = \"mantle\"\n").contains("endpoint")
    );
    assert!(
        err_of("[providers.m]\ntype = \"bedrock-mantle\"\napi = \"converse\"\n")
            .contains("not served")
    );
    assert!(
        err_of("[providers.m]\ntype = \"bedrock-mantle\"\nauth = \"none\"\n")
            .contains("bedrock-mantle")
    );
    assert!(
        err_of(
            "[embedding]\nmodel = \"m/amazon.titan-embed-text-v2:0\"\ndimension = 1024\n\
                 [providers.m]\ntype = \"bedrock-mantle\"\n"
        )
        .contains("serves no embeddings")
    );
}

#[test]
fn runtime_and_mantle_are_two_providers_sharing_a_profile() {
    let config = Config::parse(
        "[general]\nchat_model = \"mantle/openai.gpt-oss-120b\"\n\
             [embedding]\nmodel = \"bedrock/amazon.titan-embed-text-v2:0\"\ndimension = 1024\n\
             [providers.bedrock]\ntype = \"bedrock\"\naws_profile = \"sso\"\nregion = \"us-east-1\"\n\
             [providers.mantle]\ntype = \"bedrock-mantle\"\naws_profile = \"sso\"\n\
             base_url = \"https://vpce-0abc.bedrock-mantle.us-east-1.vpce.amazonaws.com\"\n",
    );
    let Ok(config) = config else {
        return assert!(config.is_ok(), "{:?}", config.err());
    };
    let mantle = config
        .providers
        .get("mantle")
        .and_then(|p| p.bedrock.clone());
    assert_eq!(
        mantle,
        Some(BedrockConfig {
            api: BedrockApi::Responses,
            region: AwsRegion::try_from(String::from("us-east-1")).ok(),
        })
    );
}

#[test]
fn openai_compat_alias_maps_to_openai() {
    let config = Config::parse(
        "[providers.g]\ntype = \"openai-compat\"\nauth = \"api-key\"\napi_key_env = \"K\"\n",
    );
    assert!(config.is_ok_and(|c| {
        c.providers
            .get("g")
            .is_some_and(|p| p.provider_type == ProviderType::Openai)
    }));
}

#[test]
fn model_reference_must_name_a_configured_provider() {
    let msg = err_of("[general]\nchat_model = \"missing/m\"\n");
    assert!(
        msg.contains("'missing'") && msg.contains("not configured"),
        "{msg}"
    );
}

#[test]
fn model_reference_must_have_a_slash_and_a_model() {
    assert!(
        err_of("[general]\nchat_model = \"ollama\"\n[providers.ollama]\ntype = \"ollama\"\n")
            .contains("PROVIDER/MODEL")
    );
    assert!(
        err_of("[general]\nchat_model = \"ollama/\"\n[providers.ollama]\ntype = \"ollama\"\n")
            .contains("missing the model")
    );
}

#[test]
fn auth_mode_must_agree_with_api_key_env() {
    assert!(
        err_of("[providers.o]\ntype = \"openai\"\napi_key_env = \"K\"\n")
            .contains("auth = \"none\"")
    );
    assert!(
        err_of("[providers.o]\ntype = \"openai\"\nauth = \"api-key\"\n").contains("no api_key_env")
    );
    assert!(
        err_of("[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\n")
            .contains("has no oauth section")
    );
}

#[test]
fn oauth_section_is_required_by_and_exclusive_to_oauth_mode() {
    let stray = "[providers.o]\ntype = \"openai\"\n[providers.o.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n";
    assert!(err_of(stray).contains("auth is not \"oauth\""));
    let with_key = "[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\napi_key_env = \"K\"\n[providers.o.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n";
    assert!(err_of(with_key).contains("sets api_key_env"));
    let empty = "[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.o.oauth]\nissuer_url = \"\"\nclient_id = \"c\"\n";
    assert!(err_of(empty).contains("needs issuer_url"));
    assert!(err_of("[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.o.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\ntenant = \"x\"\n").contains("tenant"));
}

#[test]
fn the_oidc_section_is_checked_when_read() {
    let section = |extra: &str| {
        format!(
            "[server.oidc]\nissuer_url = \"https://login.example.com/\"\nclient_id = \"quack\"\n{extra}"
        )
    };
    let good = section("redirect_uri = \"https://quack.example.com/auth/oidc/callback\"\n");
    let config = Config::parse(&good);
    let Ok(config) = config else {
        return assert!(config.is_ok(), "{config:?}");
    };
    let oidc = config.server.oidc;
    assert!(
        oidc.is_some_and(|o| o.issuer_url == "https://login.example.com"
            && o.scopes == OidcConfig::default_scopes()
            && o.client_secret_env.is_none()
            && o.audience.is_none()
            && o.subject_claim == "sub"
            && o.public_url() == "https://quack.example.com")
    );
    assert!(Config::default().server.oidc.is_none());
    let entra = Config::parse(&section(
        "redirect_uri = \"http://q:8080/auth/oidc/callback\"\naudience = \" api://quack \"\nsubject_claim = \"oid\"\n",
    ));
    assert!(entra.is_ok_and(|c| {
        c.server.oidc.is_some_and(|o| {
            o.audience.as_deref() == Some("api://quack")
                && o.subject_claim == "oid"
                && o.public_url() == "http://q:8080"
        })
    }));
    let callback = "redirect_uri = \"https://q/auth/oidc/callback\"\n";
    assert!(
        err_of(&section(&format!("{callback}audience = \" \"\n"))).contains("audience is empty")
    );
    assert!(
        err_of(&section(&format!("{callback}subject_claim = \"\"\n")))
            .contains("subject_claim is empty")
    );

    assert!(
        err_of(&section("redirect_uri = \"https://q/callback\"\n"))
            .contains("ending in /auth/oidc/callback")
    );
    assert!(
        err_of(&section("redirect_uri = \"ftp://q/auth/oidc/callback\"\n")).contains("ending in")
    );
    assert!(err_of(&section("redirect_uri = \"not a url\"\n")).contains("is not a URL"));
    assert!(
        err_of(&section(
            "redirect_uri = \"https://q/auth/oidc/callback\"\nscopes = [\"email\"]\n"
        ))
        .contains("must include \"openid\"")
    );
    assert!(err_of("[server.oidc]\nissuer_url = \" \"\nclient_id = \"c\"\nredirect_uri = \"https://q/auth/oidc/callback\"\n").contains("needs issuer_url"));
    assert!(
        err_of(&section(
            "redirect_uri = \"https://q/auth/oidc/callback\"\ntenant = \"t\"\n"
        ))
        .contains("tenant")
    );
}

#[test]
fn the_grant_is_named_and_client_credentials_needs_a_secret() {
    let section = "[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.o.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n";
    let grant_of = |extra: &str| {
        Config::parse(&format!("{section}{extra}"))
            .ok()
            .and_then(|c| {
                c.providers
                    .get("o")
                    .and_then(|p| p.auth.oauth().map(|o| o.grant))
            })
    };
    assert_eq!(
        grant_of("grant = \"device-code\"\n"),
        Some(Grant::DeviceCode)
    );
    assert_eq!(
        grant_of("grant = \"client-credentials\"\nclient_secret_env = \"S\"\n"),
        Some(Grant::ClientCredentials)
    );
    assert!(
        err_of(&format!("{section}grant = \"client-credentials\"\n"))
            .contains("needs client_secret_env")
    );
    assert!(err_of(&format!("{section}grant = \"password\"\n")).contains("password"));
    assert!(err_of(&format!("{section}device_code = true\n")).contains("device_code"));
    let obo = |extra: &str| {
        Config::parse(&format!(
            "{section}grant = \"on-behalf-of\"\nclient_secret_env = \"S\"\n{extra}"
        ))
        .ok()
        .and_then(|c| c.providers.get("o").and_then(|p| p.auth.oauth().cloned()))
    };
    let entra = obo("exchange = \"entra\"\nclient_auth = \"client_secret_basic\"\n");
    assert!(entra.is_some_and(|o| o.grant == Grant::OnBehalfOf
        && o.exchange == Exchange::Entra
        && o.client_auth == ClientAuth::ClientSecretBasic
        && o.actor));
    let okta = obo("audience = \"api://gw\"\nactor = false\n");
    assert!(okta.is_some_and(|o| o.exchange == Exchange::TokenExchange
        && o.audience.as_deref() == Some("api://gw")
        && !o.actor
        && o.client_auth == ClientAuth::ClientSecretPost));
    assert!(
        err_of(&format!("{section}grant = \"on-behalf-of\"\n"))
            .contains("grant = \"on-behalf-of\" needs client_secret_env")
    );
    assert!(
        err_of(&format!("{section}audience = \"api://gw\"\n"))
            .contains("apply only to grant = \"on-behalf-of\"")
    );
    assert!(err_of(&format!("{section}client_auth = \"mtls\"\n")).contains("mtls"));
    let named = |extra: &str| obo(extra).map(|o| o.grant_type());
    assert_eq!(
        named(""),
        Some("urn:ietf:params:oauth:grant-type:token-exchange")
    );
    assert_eq!(
        named("exchange = \"entra\"\n"),
        Some("urn:ietf:params:oauth:grant-type:jwt-bearer")
    );
}

#[test]
fn private_key_jwt_replaces_the_secret_and_refuses_one_beside_it() {
    let provider = "[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.o.oauth]\nissuer_url = \"https://us.vouch.sh\"\nclient_id = \"c\"\nclient_auth = \"private_key_jwt\"\n";
    let oauth_of = |extra: &str| {
        Config::parse(&format!("{provider}{extra}"))
            .ok()
            .and_then(|c| c.providers.get("o").and_then(|p| p.auth.oauth().cloned()))
    };
    // No secret, and still a confidential client for these grants.
    let vouch = oauth_of("grant = \"on-behalf-of\"\nactor = false\n");
    assert!(
        vouch.is_some_and(|o| o.client_auth == ClientAuth::PrivateKeyJwt
            && o.client_secret_env.is_none()
            && !o.actor)
    );
    assert!(oauth_of("grant = \"client-credentials\"\n").is_some());
    assert!(oauth_of("").is_some_and(|o| o.grant == Grant::AuthorizationCode));
    assert!(
        err_of(&format!("{provider}client_secret_env = \"S\"\n"))
            .contains("remove client_secret_env")
    );
    // A secret method without a secret is still refused for these grants.
    let public = "[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.o.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\nclient_auth = \"client_secret_basic\"\ngrant = \"on-behalf-of\"\n";
    assert!(err_of(public).contains("or client_auth = \"private_key_jwt\""));

    let oidc = "[server.oidc]\nissuer_url = \"https://us.vouch.sh\"\nclient_id = \"c\"\nredirect_uri = \"https://q/auth/oidc/callback\"\n";
    let sign_in = |extra: &str| {
        Config::parse(&format!("{oidc}{extra}"))
            .ok()
            .and_then(|c| c.server.oidc)
            .map(|o| o.client_auth)
    };
    assert_eq!(sign_in(""), Some(ClientAuth::ClientSecretPost));
    assert_eq!(
        sign_in("client_auth = \"private_key_jwt\"\n"),
        Some(ClientAuth::PrivateKeyJwt)
    );
    assert_eq!(
        sign_in("client_auth = \"client_secret_basic\"\nclient_secret_env = \"S\"\n"),
        Some(ClientAuth::ClientSecretBasic)
    );
    assert!(
        err_of(&format!(
            "{oidc}client_auth = \"private_key_jwt\"\nclient_secret_env = \"S\"\n"
        ))
        .contains("[server.oidc]: client_auth = \"private_key_jwt\"")
    );
    assert!(
        err_of(&format!("{oidc}client_auth = \"tls_client_auth\"\n")).contains("tls_client_auth")
    );
}

#[test]
fn client_id_may_be_left_out_only_for_a_registered_key_client() {
    let provider = |extra: &str| {
        format!(
            "[providers.o]\ntype = \"openai\"\nauth = \"oauth\"\n[providers.o.oauth]\nissuer_url = \"https://us.vouch.sh\"\n{extra}"
        )
    };
    let registered = Config::parse(&provider(
        "client_auth = \"private_key_jwt\"\ngrant = \"on-behalf-of\"\nactor = false\n",
    ));
    assert!(registered.is_ok_and(|c| {
        c.providers
            .get("o")
            .and_then(|p| p.auth.oauth())
            .is_some_and(|o| o.client_id.is_none())
    }));
    assert!(err_of(&provider("")).contains("quack auth register"));
    assert!(err_of(&provider("client_id = \" \"\n")).contains("client_id is empty"));

    let oidc = |extra: &str| {
        format!(
            "[server.oidc]\nissuer_url = \"https://us.vouch.sh\"\nredirect_uri = \"https://q/auth/oidc/callback\"\n{extra}"
        )
    };
    let sign_in = Config::parse(&oidc("client_auth = \"private_key_jwt\"\n"));
    assert!(sign_in.is_ok_and(|c| c.server.oidc.is_some_and(|o| o.client_id.is_none())));
    assert!(err_of(&oidc("")).contains("[server.oidc] needs client_id"));
}

#[test]
fn oauth_section_defaults_and_fields_parse() {
    let config = Config::parse(
        "[providers.azure]\ntype = \"openai\"\nauth = \"oauth\"\nbase_url = \"https://r.openai.azure.com/openai/deployments/d\"\n[providers.azure.oauth]\nissuer_url = \"https://login.microsoftonline.com/t/v2.0\"\nclient_id = \"abc\"\nscopes = [\"https://cognitiveservices.azure.com/.default\", \"offline_access\"]\nclient_secret_env = \"AZURE_CLIENT_SECRET\"\n",
    );
    let Ok(config) = config else {
        return assert!(config.is_ok(), "{config:?}");
    };
    let oauth = config.providers.get("azure").and_then(|p| p.auth.oauth());
    assert!(oauth.is_some_and(|o| {
        o.redirect_uri == "http://127.0.0.1:19876/callback"
            && o.grant == Grant::AuthorizationCode
            && o.scopes.len() == 2
            && o.client_secret_env.as_deref() == Some("AZURE_CLIENT_SECRET")
    }));
}

#[test]
fn embedding_model_needs_dimension_and_cannot_be_anthropic() {
    let no_dim = "[embedding]\nmodel = \"o/e\"\n[providers.o]\ntype = \"ollama\"\n";
    assert!(err_of(no_dim).contains("[embedding].dimension"));
    let anthropic = "[embedding]\nmodel = \"a/e\"\ndimension = 1\n[general]\n[providers.a]\ntype = \"anthropic\"\nauth = \"api-key\"\napi_key_env = \"K\"\n";
    assert!(err_of(anthropic).contains("does not serve embeddings"));
}

#[test]
fn chat_model_unset_is_a_clear_error_when_asked_for() {
    let config = Config::default();
    assert!(config.validate().is_ok());
    let err = config.chat_model_ref().err();
    assert!(err.is_some_and(|e| e.to_string().contains("chat_model")));
    assert!(config.embedding_model_ref().is_ok_and(|m| m.is_none()));
}

#[test]
fn path_helpers_use_data_dir() {
    let mut config = Config::default();
    config.general.data_dir = PathBuf::from("/data");

    assert_eq!(config.control_db_path(), PathBuf::from("/data/control.db"));
    assert_eq!(
        config.workspace_dir("ws1"),
        PathBuf::from("/data/workspaces/ws1")
    );
    assert_eq!(
        config.workspace_db_path("ws1"),
        PathBuf::from("/data/workspaces/ws1/data.duckdb")
    );
    assert_eq!(
        config.workspace_files_dir("ws1"),
        PathBuf::from("/data/workspaces/ws1/files")
    );
}

#[test]
fn default_dirs_use_xdg_layout() {
    assert!(default_data_dir().to_string_lossy().contains(APP_NAME));
    assert!(default_config_dir().to_string_lossy().contains(APP_NAME));
}

/// A data directory group or others can reach is made private, a new
/// one is created private, and a private one is left alone.
#[cfg(unix)]
#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn ensure_dirs_makes_the_data_dir_private() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config::default();
    config.general.data_dir = dir.path().join("data");
    let data = config.data_dir();
    let mode = || std::fs::metadata(data).unwrap().permissions().mode() & 0o777;

    config.ensure_dirs().unwrap();
    assert_eq!(mode(), 0o700);
    assert!(data.join("workspaces").is_dir());

    for open in [0o755, 0o750, 0o705, 0o777] {
        std::fs::set_permissions(data, std::fs::Permissions::from_mode(open)).unwrap();
        config.ensure_dirs().unwrap();
        assert_eq!(mode(), 0o700, "from {open:o}");
    }

    std::fs::set_permissions(data, std::fs::Permissions::from_mode(0o500)).unwrap();
    config.ensure_dirs().unwrap();
    assert_eq!(mode(), 0o500);
    std::fs::set_permissions(data, std::fs::Permissions::from_mode(0o700)).unwrap();
}
