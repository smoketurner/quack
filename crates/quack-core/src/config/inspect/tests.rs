#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "tests assert on values they have just built"
)]

use std::collections::BTreeSet;

use super::*;

const SAMPLE: &str = r#"
[general]
chat_model = "ollama/llama3.1:8b"

[providers.ollama]
type = "ollama"
base_url = "http://localhost:11434"

[embedding]
model = "ollama/nomic-embed-text"
dimension = 768

[retrieval]
top_k = 3
"#;

fn inspect(contents: &str) -> Inspection {
    Inspection::of(PathBuf::from("/tmp/config.toml"), Some(contents))
}

fn setting<'a>(inspection: &'a Inspection, path: &str) -> &'a Setting {
    inspection
        .settings
        .iter()
        .find(|s| s.path() == path)
        .unwrap_or_else(|| panic!("no setting {path}"))
}

#[test]
fn a_loaded_file_marks_its_own_keys_and_leaves_the_rest_default() {
    let inspection = inspect(SAMPLE);
    assert_eq!(inspection.file_state, FileState::Loaded);
    assert!(inspection.is_usable());

    let top_k = setting(&inspection, "retrieval.top_k");
    assert_eq!(top_k.origin, Origin::File);
    assert_eq!(top_k.value.as_deref(), Some("3"));
    assert_eq!(top_k.default.as_deref(), Some("8"));
    assert_eq!(top_k.file_value.as_deref(), Some("3"));

    let rrf_k = setting(&inspection, "retrieval.rrf_k");
    assert_eq!(rrf_k.origin, Origin::Default);
    assert_eq!(rrf_k.value.as_deref(), Some("60"));
    assert!(rrf_k.file_value.is_none());
}

#[test]
fn providers_and_their_defaults_are_listed() {
    let inspection = inspect(SAMPLE);
    let kind = setting(&inspection, "providers.ollama.type");
    assert_eq!(kind.origin, Origin::File);
    assert_eq!(kind.value.as_deref(), Some("\"ollama\""));
    assert!(kind.default.is_none());

    let auth = setting(&inspection, "providers.ollama.auth");
    assert_eq!(auth.origin, Origin::Default);
    assert_eq!(auth.value.as_deref(), Some("\"none\""));

    let key_env = setting(&inspection, "providers.ollama.api_key_env");
    assert_eq!(key_env.display_value(), UNSET);
}

#[test]
fn provider_headers_are_listed_by_name_only() {
    let provider = "[providers.p]\ntype = \"ollama\"\nheaders = { \"X-Team\" = \"s3cret\" }\n";
    let inspection = inspect(provider);
    let headers = setting(&inspection, "providers.p.headers");
    assert_eq!(headers.origin, Origin::File);
    assert_eq!(headers.value.as_deref(), Some("[\"X-Team\"]"));
    assert_eq!(headers.file_value.as_deref(), Some("[\"X-Team\"]"));
    let json = serde_json::to_string(&inspection.report(SettingFilter::All)).unwrap_or_default();
    assert!(
        json.contains("X-Team") && !json.contains("s3cret"),
        "{json}"
    );
}

#[test]
fn oauth_sections_are_listed_with_their_defaults() {
    let inspection = inspect(
        "[providers.azure]\ntype = \"openai\"\nauth = \"oauth\"\n\
             [providers.azure.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n\
             scopes = [\"a\", \"b\"]\n",
    );
    assert_eq!(inspection.file_state, FileState::Loaded);
    let scopes = setting(&inspection, "providers.azure.oauth.scopes");
    assert_eq!(scopes.value.as_deref(), Some("[\"a\", \"b\"]"));
    let grant = setting(&inspection, "providers.azure.oauth.grant");
    assert_eq!(grant.origin, Origin::Default);
    assert_eq!(grant.value, Some(quoted("authorization-code")));
    let redirect = setting(&inspection, "providers.azure.oauth.redirect_uri");
    assert_eq!(redirect.origin, Origin::Default);
    assert_eq!(
        redirect.value,
        Some(quoted(OAuthConfig::DEFAULT_REDIRECT_URI))
    );
}

#[test]
fn a_missing_file_is_all_defaults() {
    let inspection = Inspection::of(PathBuf::from("/tmp/config.toml"), None);
    assert_eq!(inspection.file_state, FileState::Missing);
    assert!(inspection.unknown.is_empty());
    assert!(inspection.settings.iter().all(|s| s.file_value.is_none()));
    assert_eq!(
        setting(&inspection, "retrieval.top_k").value.as_deref(),
        Some("8")
    );
}

#[test]
fn a_rejected_file_reports_the_error_and_falls_back_to_defaults() {
    let inspection = inspect("[retrieval]\ntopk = 3\n");
    let FileState::Rejected(error) = &inspection.file_state else {
        panic!("expected a rejected file, got {:?}", inspection.file_state);
    };
    assert!(error.contains("topk"), "{error}");
    assert!(!inspection.is_usable());

    // The file's values are not in force, so the listing shows the
    // built-in ones and says where the file disagrees.
    let top_k = setting(&inspection, "retrieval.top_k");
    assert_eq!(top_k.origin, Origin::Default);
    assert_eq!(top_k.value.as_deref(), Some("8"));

    assert_eq!(
        inspection.unknown,
        vec![UnknownKey {
            path: String::from("retrieval.topk"),
            suggestion: Some(String::from("top_k")),
        }]
    );
}

#[test]
fn a_file_that_is_not_toml_is_rejected_without_a_panic() {
    let inspection = inspect("[retrieval\ntop_k = ");
    assert!(matches!(inspection.file_state, FileState::Rejected(_)));
    assert!(inspection.unknown.is_empty());
    assert_eq!(
        setting(&inspection, "retrieval.top_k").value.as_deref(),
        Some("8")
    );
}

#[test]
fn unknown_sections_keys_and_misplaced_settings_are_named() {
    let inspection = inspect(
        "[genral]\nchat_model = \"a/b\"\n[analysis]\ntop_k = 1\n\
             [providers.o]\ntype = \"ollama\"\nmodel = \"x\"\n\
             [providers.o.oauth]\ntenant = \"t\"\n",
    );
    let found: Vec<(&str, Option<&str>)> = inspection
        .unknown
        .iter()
        .map(|u| (u.path.as_str(), u.suggestion.as_deref()))
        .collect();
    assert!(found.contains(&("genral", Some("general"))), "{found:?}");
    assert!(
        found.contains(&("analysis.top_k", Some("[retrieval].top_k"))),
        "{found:?}"
    );
    assert!(
        found.contains(&("providers.o.model", Some("[embedding].model"))),
        "{found:?}"
    );
    assert!(
        found.contains(&("providers.o.oauth.tenant", None)),
        "{found:?}"
    );
}

#[test]
fn every_recognized_key_has_a_setting() {
    let inspection = Inspection::of(PathBuf::from("/tmp/config.toml"), None);
    let listed: BTreeSet<String> = inspection.settings.iter().map(Setting::path).collect();
    for (section, keys) in SECTIONS {
        for key in *keys {
            let path = format!("{section}.{key}");
            assert!(listed.contains(&path), "no setting for {path}");
        }
    }
    // And nothing beyond them: the providers are the only other
    // sections, and a default config has none.
    assert_eq!(
        listed.len(),
        SECTIONS.iter().map(|(_, k)| k.len()).sum::<usize>()
    );
}

#[test]
fn every_oidc_key_has_a_setting_and_strays_are_named() {
    let inspection = inspect(
        "[server.oidc]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n\
             client_secret_env = \"S\"\nredirect_uri = \"https://q/auth/oidc/callback\"\n",
    );
    let listed: BTreeSet<String> = inspection.settings.iter().map(Setting::path).collect();
    for key in OIDC_KEYS {
        assert!(listed.contains(&format!("server.oidc.{key}")), "{key}");
    }
    let scopes = setting(&inspection, "server.oidc.scopes");
    assert_eq!(scopes.origin, Origin::Default);
    assert!(inspection.environment.iter().any(|v| v.name == "S"));

    let file = "[server.oidc]\nissuer = \"x\"\n".parse::<Table>();
    let unknown = file.map(|f| UnknownKey::find_in(&f)).unwrap_or_default();
    assert!(
        unknown.iter().any(|u| u.path == "server.oidc.issuer"),
        "{:?}",
        unknown.iter().map(|u| &u.path).collect::<Vec<_>>()
    );
    assert!(!unknown.iter().any(|u| u.path == "server.oidc"));
}

#[test]
fn every_provider_key_has_a_setting() {
    let inspection = inspect(
        "[providers.p]\ntype = \"openai\"\nauth = \"oauth\"\n\
             base_url = \"https://e\"\n\
             [providers.p.oauth]\nissuer_url = \"https://i\"\nclient_id = \"c\"\n",
    );
    let listed: BTreeSet<String> = inspection.settings.iter().map(Setting::path).collect();
    let bedrock_only = ["region"];
    for key in PROVIDER_KEYS
        .iter()
        .filter(|k| !["oauth", "models"].contains(*k) && !bedrock_only.contains(*k))
    {
        assert!(listed.contains(&format!("providers.p.{key}")), "{key}");
    }
    let bedrock = inspect("[providers.b]\ntype = \"bedrock-mantle\"\n");
    let listed_bedrock: BTreeSet<String> = bedrock.settings.iter().map(Setting::path).collect();
    for key in bedrock_only {
        assert!(
            listed_bedrock.contains(&format!("providers.b.{key}")),
            "{key}"
        );
    }
    for key in OAUTH_KEYS {
        assert!(
            listed.contains(&format!("providers.p.oauth.{key}")),
            "{key}"
        );
    }
}

/// The field names serde reports for a section, taken from the
/// `deny_unknown_fields` error a probe key provokes. This is what
/// keeps [`SECTIONS`] honest: a field added to a config struct and
/// not to the list, or the other way round, fails the next test.
fn fields_of(header: &str) -> BTreeSet<String> {
    let probe = format!("{header}\nquack_probe_key = 1\n");
    let error = Config::parse(&probe)
        .expect_err("a probe key is an unknown field")
        .to_string();
    let (_, expected) = error
        .split_once("expected")
        .expect("serde names the fields it expected");
    expected
        .split('`')
        .skip(1)
        .step_by(2)
        .map(String::from)
        .collect()
}

#[test]
fn the_key_list_matches_the_config_structs() {
    for (section, keys) in SECTIONS {
        let mut declared: BTreeSet<String> = keys.iter().map(|k| (*k).to_owned()).collect();
        // A nested table, checked on its own below.
        if *section == "server" {
            declared.insert(String::from("oidc"));
        }
        assert_eq!(fields_of(&format!("[{section}]")), declared, "[{section}]");
    }
    let oidc: BTreeSet<String> = OIDC_KEYS.iter().map(|k| (*k).to_owned()).collect();
    assert_eq!(fields_of("[server.oidc]"), oidc, "[server.oidc]");
    let providers: BTreeSet<String> = PROVIDER_KEYS.iter().map(|k| (*k).to_owned()).collect();
    assert_eq!(
        fields_of("[providers.p]\ntype = \"ollama\""),
        providers,
        "[providers.NAME]"
    );
    let oauth: BTreeSet<String> = OAUTH_KEYS.iter().map(|k| (*k).to_owned()).collect();
    assert_eq!(
        fields_of(
            "[providers.p]\ntype = \"ollama\"\n[providers.p.oauth]\nissuer_url = \"i\"\nclient_id = \"c\""
        ),
        oauth,
        "[providers.NAME.oauth]"
    );
    let model: BTreeSet<String> = MODEL_KEYS.iter().map(|k| (*k).to_owned()).collect();
    assert_eq!(
        fields_of("[providers.p]\ntype = \"ollama\"\n[providers.p.models.\"m\"]"),
        model,
        "[providers.NAME.models.\"ID\"]"
    );
}

#[test]
fn model_settings_are_listed_under_the_quoted_model_id() {
    let inspection = inspect(
        "[providers.p]\ntype = \"openai\"\n\
             [providers.p.models.\"gpt-5.6\"]\ntemperature = false\n\
             effort = \"high\"\n",
    );
    let temperature = setting(&inspection, "providers.p.models.\"gpt-5.6\".temperature");
    assert_eq!(temperature.origin, Origin::File);
    assert_eq!(temperature.value.as_deref(), Some("false"));
    let effort = setting(&inspection, "providers.p.models.\"gpt-5.6\".effort");
    assert_eq!(effort.value.as_deref(), Some("\"high\""));
    assert_eq!(effort.file_value.as_deref(), Some("\"high\""));

    let file = "[providers.p]\ntype = \"openai\"\n[providers.p.models.\"m\"]\ntemp = false\n"
        .parse::<Table>();
    let unknown = file.map(|f| UnknownKey::find_in(&f)).unwrap_or_default();
    let found = unknown
        .iter()
        .find(|u| u.path == "providers.p.models.\"m\".temp")
        .unwrap_or_else(|| panic!("{:?}", unknown.iter().map(|u| &u.path).collect::<Vec<_>>()));
    assert_eq!(found.suggestion.as_deref(), Some("temperature"));
}

#[test]
fn the_section_list_matches_the_config_struct() {
    let declared: BTreeSet<String> = UnknownKey::section_names()
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    assert_eq!(fields_of(""), declared);
}

#[test]
fn changed_lists_what_the_file_has_a_say_in() {
    let inspection = inspect(SAMPLE);
    let changed: BTreeSet<String> = inspection.changed().map(Setting::path).collect();
    assert!(changed.contains("retrieval.top_k"));
    assert!(changed.contains("general.chat_model"));
    assert!(!changed.contains("retrieval.rrf_k"));
}

#[test]
fn the_environment_listing_names_provider_variables_without_reading_them() {
    let inspection = inspect(
        "[providers.a]\ntype = \"anthropic\"\nauth = \"api-key\"\napi_key_env = \"QUACK_TEST_KEY\"\n",
    );
    let names: Vec<&str> = inspection
        .environment
        .iter()
        .map(|v| v.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec![
            ENV_CONFIG_DIR,
            ENV_DATA_DIR,
            ENV_MODEL,
            ENV_BIND,
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "QUACK_TEST_KEY"
        ]
    );
}
