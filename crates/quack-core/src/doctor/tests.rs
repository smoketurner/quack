use super::*;
use crate::config::ProviderConfig;
use crate::proxy::{Environment, Variable};
use crate::storage::control::{AuditAction, AuditEntry, Channel, Outcome};

fn proxy_checks(environment: Environment) -> Vec<Check> {
    let mut report = Report::default();
    check_proxy(&mut report, &Proxies::new(environment));
    report.checks
}

#[test]
fn the_proxy_check_says_what_goes_where() {
    let none = proxy_checks(Environment::default());
    assert_eq!(
        none,
        [Check::new(
            Area::Proxy,
            Status::Ok,
            "none (no proxy variables set)"
        )]
    );

    let both = proxy_checks(Environment {
        http: Variable::named("HTTP_PROXY", "http://user:s3cret@proxy.corp:8080"),
        https: Variable::named("HTTPS_PROXY", "http://user:s3cret@proxy.corp:8080"),
        all: None,
        no: Variable::named("NO_PROXY", "internal.corp,10.0.0.0/8,.models.corp"),
    });
    assert_eq!(
        both,
        [Check::new(
            Area::Proxy,
            Status::Ok,
            "HTTPS through proxy.corp:8080, HTTP through proxy.corp:8080; direct: loopback, \
             169.254.0.0/16, NO_PROXY (3 entries)"
        )]
    );

    let https_only = proxy_checks(Environment {
        https: Variable::named("https_proxy", "proxy.corp:3128"),
        ..Environment::default()
    });
    assert_eq!(
        https_only.first().map(|c| c.summary.as_str()),
        Some("HTTPS through proxy.corp:3128; direct: loopback, 169.254.0.0/16")
    );
}

#[test]
fn the_proxy_check_fails_what_cannot_work_and_warns_on_dead_no_proxy_entries() {
    let socks = proxy_checks(Environment {
        https: Variable::named("HTTPS_PROXY", "socks5://127.0.0.1:1080"),
        ..Environment::default()
    });
    let first = socks.first().cloned();
    assert_eq!(
        first,
        Some(
            Check::new(
                Area::Proxy,
                Status::Fail,
                "HTTPS_PROXY is a socks5 proxy, which quack does not support; requests \
                 through it fail"
            )
            .fix("set HTTPS_PROXY to an http:// or https:// proxy URL, or unset it")
        )
    );

    assert_eq!(
        socks.get(1).cloned(),
        Some(Check::new(
            Area::Proxy,
            Status::Fail,
            "HTTPS through 127.0.0.1:1080 (unsupported socks5: these requests fail); direct: \
             loopback, 169.254.0.0/16"
        ))
    );

    let unusable = proxy_checks(Environment {
        http: Variable::named("HTTP_PROXY", "ftp://proxy.corp"),
        ..Environment::default()
    });
    assert_eq!(
        unusable,
        [Check::new(
            Area::Proxy,
            Status::Fail,
            "HTTP_PROXY is not an http:// or https:// proxy URL; requests go direct until \
             it is fixed"
        )
        .fix("set HTTP_PROXY to an http:// or https:// proxy URL, or unset it")]
    );

    let glob = proxy_checks(Environment {
        https: Variable::named("HTTPS_PROXY", "http://proxy.corp:8080"),
        no: Variable::named("NO_PROXY", "*.internal"),
        ..Environment::default()
    });
    assert_eq!(
        glob.first().cloned(),
        Some(
            Check::new(
                Area::Proxy,
                Status::Warn,
                "NO_PROXY entry \"*.internal\" never matches"
            )
            .fix("write \".internal\"")
        )
    );
    assert_eq!(glob.len(), 2);
}

#[expect(clippy::unwrap_used, reason = "test")]
fn embedding_config(model: &str, extra: &str) -> Config {
    let config: Config = toml::from_str(&format!(
        "[providers.o]\ntype = \"ollama\"\n\
         [embedding]\nmodel = \"o/{model}\"\ndimension = 1024\n{extra}"
    ))
    .unwrap();
    config
}

/// An Ollama whose `/api/embed` answers every request with one vector of
/// `width` numbers, for `requests` connections.
#[expect(clippy::unwrap_used, reason = "test")]
async fn embed_server(width: usize, requests: usize) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        for _ in 0..requests {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0_u8; 8192];
            drop(stream.read(&mut buf).await);
            let body = serde_json::json!({ "embeddings": [vec![0.1_f32; width]] }).to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            drop(stream.write_all(response.as_bytes()).await);
        }
    });
    base
}

/// The width comes from an embedding call, not the model's metadata,
/// which can name an inner width a final projection changes.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn the_width_probe_measures_a_call_and_names_the_fix() {
    Egress::scope(Some(Egress::NoWorkspace), async {
        let config = |base: &str, dimension: u32| -> Config {
            toml::from_str(&format!(
                "[providers.o]\ntype = \"ollama\"\nbase_url = \"{base}\"\nmax_retries = 0\n\
                 [embedding]\nmodel = \"o/embeddinggemma-2\"\ndimension = {dimension}\n"
            ))
            .unwrap()
        };
        let measure = |config: Config| async move {
            let embedder = Embeddings::from_config(&config).await.unwrap().unwrap();
            let model = config.embedding_model_ref().unwrap().unwrap();
            let dimension = config.embedding.dimension.unwrap();
            let check = width_check(
                model,
                dimension,
                embedder.measure_width().await.map_err(|e| e.to_string()),
            );
            (check.status, check.summary, check.fix)
        };

        let (status, summary, fix) = measure(config(&embed_server(768, 1).await, 1024)).await;
        assert_eq!(status, Status::Fail);
        assert!(
            summary.contains("makes 768-dimensional vectors"),
            "{summary}"
        );
        assert_eq!(
            fix.as_deref(),
            Some("set dimension = 768 under [embedding]")
        );

        let (status, summary, _) = measure(config(&embed_server(768, 1).await, 768)).await;
        assert_eq!(status, Status::Ok, "{summary}");

        // Nothing listening: the call's failure is the finding.
        let (status, summary, _) = measure(config("http://127.0.0.1:9", 768)).await;
        assert_eq!(status, Status::Fail);
        assert!(summary.contains("an embedding call failed"), "{summary}");
    })
    .await;
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn the_prompts_check_says_where_the_prefixes_come_from() {
    for (model, extra, status, words) in [
        (
            "embeddinggemma",
            "",
            Status::Ok,
            "EmbeddingGemma was trained with",
        ),
        ("all-minilm", "", Status::Ok, "takes no input prefixes"),
        (
            "embeddinggemma",
            "query_prefix = \"q: \"\n",
            Status::Ok,
            "from [embedding]",
        ),
        ("my-embedder", "", Status::Info, "knows no input prefixes"),
    ] {
        let config = embedding_config(model, extra);
        let check = prompts_check(&config, config.embedding_model_ref().unwrap().unwrap());
        assert_eq!(check.status, status, "{model}");
        assert!(check.summary.contains(words), "{}", check.summary);
    }
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn the_chat_model_check_says_what_a_turn_sends() {
    let gateway = "[providers.gw]\ntype = \"openai\"\nbase_url = \"https://gw.example\"\n\
                   api = \"responses\"\n";
    for (chat, extra, status, words) in [
        (
            "gw/corp-reasoner",
            "[analysis]\neffort = \"medium\"\n",
            Status::Ok,
            "sends no temperature, reasoning effort medium",
        ),
        (
            "gw/gpt-5.6-sol",
            "[analysis]\neffort = \"medium\"\n[providers.gw.models.\"gpt-5.6-sol\"]\n\
             effort = \"minimal\"\n",
            Status::Fail,
            "effort \"minimal\" for gpt-5.6-sol",
        ),
        (
            "ol/llama3.1:8b",
            "[analysis]\neffort = \"high\"\n[providers.ol]\ntype = \"ollama\"\n",
            Status::Ok,
            "sends temperature, reasoning effort high",
        ),
        (
            "ol/gpt-oss:20b",
            "[analysis]\neffort = \"xhigh\"\n[providers.ol]\ntype = \"ollama\"\n",
            Status::Fail,
            "Ollama has no `xhigh` thinking level",
        ),
    ] {
        let toml = format!("[general]\nchat_model = \"{chat}\"\n{gateway}{extra}");
        let config = Config::parse(&toml).unwrap();
        let check = chat_settings_check(&config, config.chat_model_ref().unwrap()).await;
        assert_eq!(check.status, status, "{chat}: {}", check.summary);
        assert!(check.summary.contains(words), "{}", check.summary);
    }
}

/// A level doctor could not check is a warning that says so, never an ok.
#[test]
fn an_effort_doctor_cannot_check_is_a_warning() {
    let chat = ChatSettings::new("m", Wire::Converse, Some(Effort::High), None);
    let check = |outcome: EffortCheck| outcome.check("br/m", chat, "; costs", String::from("fix"));
    assert_eq!(check(EffortCheck::Accepted).status, Status::Ok);
    let unchecked = check(EffortCheck::Unchecked(Error::Config(String::from(
        "no region",
    ))));
    assert_eq!(unchecked.status, Status::Warn);
    assert!(
        unchecked
            .summary
            .contains("not checked here: configuration error: no region"),
        "{}",
        unchecked.summary
    );
    let refused = check(EffortCheck::Refused(Error::Config(String::from(
        "no such level",
    ))));
    assert_eq!(refused.status, Status::Fail);
    assert!(
        refused.summary.ends_with("no such level; costs"),
        "{}",
        refused.summary
    );
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn the_chat_model_check_fails_what_every_turn_refuses() {
    let gateway = "[providers.gw]\ntype = \"openai\"\nbase_url = \"https://gw.example\"\n";
    for (chat, extra, status) in [
        ("gw/gpt-5.6-sol", "", Status::Fail),
        ("gw/gpt-5.6-sol", "api = \"responses\"\n", Status::Ok),
        (
            "gw/gpt-5.6-sol",
            "[analysis]\neffort = \"none\"\n",
            Status::Ok,
        ),
        (
            "gw/gpt-5.6-sol",
            "effort = \"high\"\n[analysis]\neffort = \"none\"\n",
            Status::Fail,
        ),
        ("gw/gpt-6-luna", "", Status::Ok),
    ] {
        let toml = format!("[general]\nchat_model = \"{chat}\"\n{gateway}{extra}");
        let config = Config::parse(&toml).unwrap();
        let check = chat_settings_check(&config, config.chat_model_ref().unwrap()).await;
        assert_eq!(check.status, status, "{toml}: {}", check.summary);
    }
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn the_background_effort_is_checked_when_it_is_not_the_turns() {
    let gateway = "[providers.gw]\ntype = \"openai\"\nbase_url = \"https://gw.example\"\n";
    for (analysis, status) in [
        ("effort = \"none\"\n", Some(Status::Ok)),
        ("effort = \"none\"\nbackground_effort = \"none\"\n", None),
        (
            "effort = \"none\"\nbackground_effort = \"minimal\"\n",
            Some(Status::Fail),
        ),
    ] {
        let toml =
            format!("[general]\nchat_model = \"gw/gpt-5.6-sol\"\n{gateway}[analysis]\n{analysis}");
        let config = Config::parse(&toml).unwrap();
        let model = config.chat_model_ref().unwrap();
        assert_eq!(
            chat_settings_check(&config, model).await.status,
            Status::Ok,
            "{toml}"
        );
        let check = background_check(&config, model).await;
        assert_eq!(check.as_ref().map(|c| c.status), status, "{toml}");
        if let Some(check) = check {
            assert!(
                check.summary.contains("background calls"),
                "{}",
                check.summary
            );
            assert_eq!(
                check.summary.contains("\"minimal\""),
                check.status == Status::Fail,
                "{}",
                check.summary
            );
        }
    }
    let toml = format!(
        "[general]\nchat_model = \"gw/gpt-5.6-sol\"\n{gateway}api = \"responses\"\n\
         [analysis]\nbackground_effort = \"low\"\n"
    );
    let config = Config::parse(&toml).unwrap();
    let check = background_check(&config, config.chat_model_ref().unwrap())
        .await
        .unwrap();
    assert_eq!(check.status, Status::Ok, "{}", check.summary);
}

fn inspection(dir: &Path, toml: Option<&str>) -> Inspection {
    let mut inspection = Inspection::of(dir.join("config.toml"), toml);
    inspection.config.general.data_dir = dir.join("data");
    inspection
}

fn offline() -> Options {
    Options {
        probing: Probing::Offline,
        ..Options::default()
    }
}

fn find(report: &Report, area: Area) -> Vec<&Check> {
    report.checks.iter().filter(|c| c.area == area).collect()
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn sign_in_is_checked_for_its_secret_and_local_mode() {
    let dir = tempfile::tempdir().unwrap();
    let oidc = "[server.oidc]\nissuer_url = \"https://login.example.com\"\nclient_id = \"quack\"\nredirect_uri = \"https://q.example.com/auth/oidc/callback\"\n";
    let mut found = Vec::new();
    for extra in [
        "",
        "client_secret_env = \"QUACK_TEST_UNSET_OIDC_SECRET\"\n",
        "[server]\nlocal = true\n",
    ] {
        let toml = format!("{oidc}{extra}");
        let report = run(&inspection(dir.path(), Some(&toml)), &offline()).await;
        let checks: Vec<(Status, String)> = find(&report, Area::Server)
            .into_iter()
            .filter(|c| c.summary.contains("login.example.com"))
            .map(|c| (c.status, c.summary.clone()))
            .collect();
        found.push(checks);
    }
    let [plain, secret, local]: [Vec<(Status, String)>; 3] = found.try_into().unwrap();
    assert!(
        matches!(plain.as_slice(), [(Status::Ok, s)] if s.contains("not probed")),
        "{plain:?}"
    );
    assert!(
        matches!(secret.as_slice(), [(Status::Fail, s)] if s.contains("QUACK_TEST_UNSET_OIDC_SECRET")),
        "{secret:?}"
    );
    assert!(
        matches!(local.as_slice(), [(Status::Warn, s)] if s.contains("ignored")),
        "{local:?}"
    );
}

/// The workspace check names the versions the file records, and a file
/// a newer quack upgraded is a failure whose fix names that quack.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn the_workspace_check_reports_versions_and_fails_on_a_newer_file() {
    let dir = tempfile::tempdir().unwrap();
    let inspection = inspection(dir.path(), None);
    let config = &inspection.config;
    let control = ControlPlane::open(config).await.unwrap();
    let entry = AuditEntry::new(AuditAction::Admin, Outcome::Allowed, Channel::Cli);
    let workspace = control
        .create_workspace(&config.general.default_workspace, None, entry)
        .await
        .unwrap();
    drop(control);
    let db = WorkspaceDb::open(config, workspace.id.as_str()).unwrap();
    let schema = db.meta(MetaKey::SchemaVersion).unwrap().unwrap();
    drop(db);

    let report = run(&inspection, &offline()).await;
    let checks = find(&report, Area::Workspace);
    let check = checks.first().unwrap();
    assert_eq!(check.status, Status::Ok, "{check:?}");
    let versions = format!(
        "schema version {schema}, written by quack {} with DuckDB v",
        env!("CARGO_PKG_VERSION")
    );
    assert!(check.summary.contains(&versions), "{check:?}");

    // An older file is reported, not upgraded: the doctor leaves the way
    // back to the quack that wrote it.
    let db = WorkspaceDb::open(config, workspace.id.as_str()).unwrap();
    db.set_meta(MetaKey::SchemaVersion, "11").unwrap();
    drop(db);
    let report = run(&inspection, &offline()).await;
    let checks = find(&report, Area::Workspace);
    let check = checks.first().unwrap();
    assert_eq!(check.status, Status::Info, "{check:?}");
    assert!(check.summary.contains("schema version 11"), "{check:?}");
    assert!(
        check
            .fix
            .as_deref()
            .unwrap()
            .contains("copy the data directory")
    );
    assert_eq!(
        WorkspaceDb::recorded_schema(config, workspace.id.as_str()).unwrap(),
        11,
        "the doctor left the file as it was"
    );

    let db = WorkspaceDb::open(config, workspace.id.as_str()).unwrap();
    db.set_meta(MetaKey::SchemaVersion, "99").unwrap();
    db.set_meta(MetaKey::WrittenByQuack, "9.9.9").unwrap();
    drop(db);

    let report = run(&inspection, &offline()).await;
    let checks = find(&report, Area::Workspace);
    let check = checks.first().unwrap();
    assert_eq!(check.status, Status::Fail, "{check:?}");
    assert!(check.summary.contains("schema version 99"), "{check:?}");
    assert!(
        check.summary.contains("written by quack 9.9.9,"),
        "{check:?}"
    );
    let fix = check.fix.as_deref().unwrap();
    assert!(fix.contains("run quack 9.9.9 or newer"), "{fix}");
    assert!(fix.contains("restore the copy"), "{fix}");
    // The advice is the fix alone, not the summary too.
    assert!(!check.summary.contains("or newer"), "{check:?}");

    // Versions that cannot be read fail the check instead of printing
    // as unrecorded.
    let unreadable = Check::recorded_versions(
        "'ws' opens: 0 tables, 0 documents",
        Err(Error::Config(String::from("the meta table did not answer"))),
    );
    assert_eq!(unreadable.status, Status::Fail, "{unreadable:?}");
    assert!(
        unreadable.summary.contains("cannot be read:")
            && unreadable.summary.contains("the meta table did not answer"),
        "{unreadable:?}"
    );
    assert!(unreadable.fix.is_some(), "{unreadable:?}");
    let unrecorded = Check::recorded_versions(
        "'ws' opens",
        Ok(FileVersions {
            schema: None,
            quack: None,
            duckdb: None,
        }),
    );
    assert!(
        unrecorded.summary.contains("schema version unrecorded"),
        "{unrecorded:?}"
    );
}

/// A named workspace that does not exist is a failure with the command
/// that creates it; the default one is created by its first use.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn a_missing_workspace_fails_unless_it_is_the_default() {
    let dir = tempfile::tempdir().unwrap();
    let inspection = inspection(dir.path(), None);
    drop(ControlPlane::open(&inspection.config).await.unwrap());
    let named = |name: &str| Options {
        workspace: Some(name.to_owned()),
        ..offline()
    };

    let report = run(&inspection, &named("slaes")).await;
    let checks = find(&report, Area::Workspace);
    let check = checks.first().unwrap();
    assert_eq!(check.status, Status::Fail, "{check:?}");
    assert!(check.summary.contains("no workspace named 'slaes'"));
    assert_eq!(check.fix.as_deref(), Some("quack workspace create slaes"));

    for options in [offline(), named("default")] {
        let report = run(&inspection, &options).await;
        let checks = find(&report, Area::Workspace);
        let check = checks.first().unwrap();
        assert_eq!(check.status, Status::Ok, "{check:?}");
        assert!(check.summary.contains("does not exist yet"), "{check:?}");
    }
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn a_fresh_install_has_no_failures_and_says_what_needs_a_model() {
    let dir = tempfile::tempdir().unwrap();
    let report = run(&inspection(dir.path(), None), &offline()).await;
    assert!(!report.has_failures(), "{report:#?}");
    let chat = find(&report, Area::ChatModel);
    assert_eq!(chat.len(), 1);
    assert_eq!(chat.first().unwrap().status, Status::Warn);
    assert!(chat.first().unwrap().summary.contains("SQL"));
    assert!(
        chat.first()
            .unwrap()
            .fix
            .as_deref()
            .unwrap()
            .contains("quack init")
    );
    assert_eq!(
        find(&report, Area::Embeddings).first().unwrap().status,
        Status::Info
    );
    // Nothing was created by looking.
    assert!(!dir.path().join("data").exists());
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn a_rejected_file_and_its_unknown_key_are_failures() {
    let dir = tempfile::tempdir().unwrap();
    let report = run(
        &inspection(dir.path(), Some("[general]\nchat_modle = \"x/y\"\n")),
        &offline(),
    )
    .await;
    let config = find(&report, Area::Config);
    assert!(
        config.iter().all(|c| c.status == Status::Fail),
        "{config:#?}"
    );
    assert!(
        config
            .iter()
            .any(|c| c.fix.as_deref().is_some_and(|f| f.contains("chat_model")))
    );
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn a_missing_api_key_is_a_failure_naming_the_variable() {
    let dir = tempfile::tempdir().unwrap();
    let toml = "[general]\nchat_model = \"a/claude\"\n[providers.a]\ntype = \"anthropic\"\n\
                auth = \"api-key\"\napi_key_env = \"QUACK_DOCTOR_TEST_KEY_UNSET\"\n";
    let report = run(&inspection(dir.path(), Some(toml)), &offline()).await;
    let chat = find(&report, Area::ChatModel);
    let check = chat.first().unwrap();
    assert_eq!(check.status, Status::Fail);
    assert!(check.summary.contains("QUACK_DOCTOR_TEST_KEY_UNSET"));
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn an_unreachable_ollama_is_a_failure_with_the_cause() {
    let dir = tempfile::tempdir().unwrap();
    let toml = "[general]\nchat_model = \"o/m\"\n[providers.o]\ntype = \"ollama\"\n\
                base_url = \"http://127.0.0.1:9\"\n";
    let options = Options {
        probing: Probing::Online {
            timeout: Duration::from_secs(2),
        },
        ..Options::default()
    };
    let report = run(&inspection(dir.path(), Some(toml)), &options).await;
    let check = *find(&report, Area::ChatModel).first().unwrap();
    assert_eq!(check.status, Status::Fail, "{check:#?}");
    assert!(check.summary.contains("cannot reach"));
    assert!(check.fix.as_deref().unwrap().contains("ollama serve"));
}

#[cfg(unix)]
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn a_data_dir_others_can_read_is_a_warning_and_a_fresh_one_is_private() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let inspection = inspection(dir.path(), None);
    inspection.config.ensure_dirs().unwrap();
    let data = inspection.config.data_dir();
    let mode = std::fs::metadata(data).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o700);
    let report = run(&inspection, &offline()).await;
    assert_eq!(
        find(&report, Area::Data).first().unwrap().status,
        Status::Ok
    );

    std::fs::set_permissions(data, std::fs::Permissions::from_mode(0o755)).unwrap();
    let report = run(&inspection, &offline()).await;
    let check = *find(&report, Area::Data).first().unwrap();
    assert_eq!(check.status, Status::Warn);
    assert!(check.fix.as_deref().unwrap().starts_with("chmod 700"));
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn an_open_bind_and_local_off_loopback_are_flagged() {
    let dir = tempfile::tempdir().unwrap();
    let open = run(
        &inspection(dir.path(), Some("[server]\nbind = \"0.0.0.0:8080\"\n")),
        &offline(),
    )
    .await;
    assert_eq!(
        find(&open, Area::Server).first().unwrap().status,
        Status::Warn
    );
    let local = run(
        &inspection(
            dir.path(),
            Some("[server]\nbind = \"0.0.0.0:8080\"\nlocal = true\n"),
        ),
        &offline(),
    )
    .await;
    assert_eq!(
        find(&local, Area::Server).first().unwrap().status,
        Status::Fail
    );
}

/// A model list server that answers one request with the model `m` and
/// hands back that request, lowercased.
#[expect(clippy::unwrap_used, reason = "test")]
async fn one_listing() -> (BaseUrl, tokio::task::JoinHandle<String>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let seen = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 4096];
        let n = stream.read(&mut buf).await.unwrap_or(0);
        let body = r#"{"data":[{"id":"m","display_name":"M"}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        drop(stream.write_all(response.as_bytes()).await);
        String::from_utf8_lossy(buf.get(..n).unwrap_or_default()).to_ascii_lowercase()
    });
    (BaseUrl::try_from(base).unwrap(), seen)
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn the_listing_probe_sends_provider_headers_beside_its_credential() {
    Egress::scope(Some(Egress::NoWorkspace), async {
        let (base, seen) = one_listing().await;
        let provider = ProviderConfig {
            headers: Some(BTreeMap::from([(
                String::from("X-Gateway-Team"),
                String::from("quack"),
            )])),
            base_url: Some(base),
            ..ProviderConfig::new(ProviderType::Openai)
        };
        let name: ProviderName = "gateway".parse().unwrap();
        let client = ChatClient::connect(&name, &provider, Some("key"));
        let listing = Probe::listing(client, Duration::from_secs(5)).await;
        assert!(matches!(&listing, Ok(models) if models.get("m").is_some()));
        let request = seen.await.unwrap();
        assert!(request.starts_with("get /models "), "{request}");
        assert!(request.contains("x-gateway-team: quack"), "{request}");
        assert!(request.contains("authorization: bearer key"), "{request}");
    })
    .await;
}

/// An Anthropic provider's probe sends its credential where completions
/// do: an OAuth token as a bearer, an API key as `x-api-key`.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn the_anthropic_probe_sends_an_oauth_token_as_a_bearer() {
    Egress::scope(Some(Egress::NoWorkspace), async {
        let oauth = "auth = \"oauth\"\n[providers.p.oauth]\n\
                 issuer_url = \"http://127.0.0.1:9\"\nclient_id = \"c\"\n";
        let keyed = "auth = \"api-key\"\napi_key_env = \"CARGO_PKG_NAME\"\n";
        for (auth, bearer) in [(oauth, true), (keyed, false)] {
            let (base, seen) = one_listing().await;
            let config: Config = toml::from_str(&format!(
                "[general]\nchat_model = \"p/m\"\n[providers.p]\n\
             type = \"anthropic\"\nbase_url = \"{base}\"\n{auth}"
            ))
            .unwrap();
            let chat = config.chat_model_ref().unwrap();
            let client = ChatClient::connect(chat.provider_name, chat.provider, Some("tok-1"));
            let listing = Probe::listing(client, Duration::from_secs(5)).await;
            assert!(matches!(&listing, Ok(models) if models.get("m").is_some()));
            let request = seen.await.unwrap();
            assert!(request.starts_with("get /v1/models "), "{request}");
            assert!(request.contains("anthropic-version: "), "{request}");
            assert_eq!(
                request.contains("authorization: bearer tok-1\r\n"),
                bearer,
                "{request}"
            );
            assert_eq!(
                request.contains("x-api-key: tok-1\r\n"),
                !bearer,
                "{request}"
            );
        }
    })
    .await;
}

fn listed(models: &[(&str, Option<u32>)]) -> ProviderModels {
    ProviderModels::from(rig::model::ModelList::new(
        models
            .iter()
            .map(|(id, window)| rig::model::ModelInfo {
                context_length: *window,
                ..rig::model::ModelInfo::from_id(*id)
            })
            .collect(),
    ))
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn a_model_missing_from_the_listing_names_the_closest_ones() {
    let base = BaseUrl::try_from(String::from("http://127.0.0.1:11434")).unwrap();
    for (provider, status) in [("ollama", Status::Fail), ("openai", Status::Warn)] {
        let config: Config = toml::from_str(&format!(
            "[general]\nchat_model = \"p/gpt-oss:20\"\n[providers.p]\ntype = \"{provider}\"\n\
             auth = \"api-key\"\napi_key_env = \"CARGO_PKG_NAME\"\n"
        ))
        .unwrap();
        let model = config.chat_model_ref().unwrap();
        let models = listed(&[
            ("llama3.1:8b", None),
            ("gpt-oss:20b", None),
            ("gpt-oss:120b", None),
            ("qwen3:4b", None),
        ]);
        let check = listing_check(Area::ChatModel, model, &base, &Ok(models));
        assert_eq!(check.status, status, "{provider}");
        assert!(
            check
                .summary
                .ends_with("the closest it lists: gpt-oss:20b, gpt-oss:120b, qwen3:4b"),
            "{}",
            check.summary
        );
    }
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn a_context_window_smaller_than_the_budgets_is_a_warning() {
    let config: Config = toml::from_str(
        "[general]\nchat_model = \"p/small\"\n[providers.p]\ntype = \"openai\"\n\
         auth = \"api-key\"\napi_key_env = \"CARGO_PKG_NAME\"\n",
    )
    .unwrap();
    let model = config.chat_model_ref().unwrap();
    let budgets = config
        .analysis
        .history_token_budget
        .get()
        .saturating_add(config.retrieval.pinned_token_budget.get())
        .saturating_add(config.context.max_tokens.get());
    let small = listed(&[("small", Some(budgets.saturating_sub(1)))]);
    let check = Check::context_window(&config, model, &small).unwrap();
    assert_eq!(check.status, Status::Warn);
    assert!(
        check.summary.contains(&format!("{budgets} in all")),
        "{}",
        check.summary
    );
    let roomy = listed(&[("small", Some(budgets))]);
    assert!(Check::context_window(&config, model, &roomy).is_none());
    let unreported = listed(&[("small", None)]);
    assert!(Check::context_window(&config, model, &unreported).is_none());
}

/// A rerank server: `GET /v1/models` lists `bge-reranker`, and `POST
/// /v1/rerank` scores the documents. Serves `requests` connections.
#[expect(clippy::unwrap_used, reason = "test")]
async fn rerank_server(requests: usize) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    tokio::spawn(async move {
        for _ in 0..requests {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0_u8; 4096];
            let n = stream.read(&mut buf).await.unwrap_or(0);
            let head = String::from_utf8_lossy(buf.get(..n).unwrap_or_default()).to_string();
            let body = if head.starts_with("GET /v1/models") {
                r#"{"data":[{"id":"bge-reranker"}]}"#
            } else {
                r#"{"results":[{"index":1,"relevance_score":0.7},{"index":0,"relevance_score":0.1}]}"#
            };
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            drop(stream.write_all(response.as_bytes()).await);
        }
    });
    base
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn the_rerank_model_is_listed_and_answers_a_probe() {
    Egress::scope(Some(Egress::NoWorkspace), async {
        let config = |base: &str, model: &str| -> Config {
            toml::from_str(&format!(
                "[retrieval]\nrerank = \"reranker\"\nrerank_model = \"tei/{model}\"\n\
             [providers.tei]\ntype = \"openai\"\nbase_url = \"{base}\"\n"
            ))
            .unwrap()
        };
        let probing = Probing::Online {
            timeout: Duration::from_secs(5),
        };
        let base = rerank_server(2).await;
        let mut report = Report::default();
        check_reranker(&mut report, &config(&base, "bge-reranker"), probing).await;
        let checks = find(&report, Area::Reranker);
        let summaries: Vec<(Status, &str)> = checks
            .iter()
            .map(|c| (c.status, c.summary.as_str()))
            .collect();
        assert_eq!(
            summaries,
            [
                (
                    Status::Ok,
                    "tei/bge-reranker: reachable, credential accepted, model listed"
                ),
                (Status::Ok, "tei/bge-reranker: a rerank call was answered"),
            ]
        );

        // A model the server does not list is named with what it does.
        let base = rerank_server(2).await;
        let mut report = Report::default();
        check_reranker(&mut report, &config(&base, "bge-rerank"), probing).await;
        let listing = find(&report, Area::Reranker);
        assert!(
            listing.first().is_some_and(|c| c.status == Status::Warn
                && c.summary.ends_with("the closest it lists: bge-reranker")),
            "{listing:?}"
        );

        // Nothing to check in another mode.
        let mut report = Report::default();
        check_reranker(&mut report, &Config::default(), probing).await;
        assert!(find(&report, Area::Reranker).is_empty());
    })
    .await;
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn the_decision_model_is_listed_reports_its_capability_and_answers_a_probe() {
    use crate::llm::decision::stub::{DecisionStub, Fault};

    Egress::scope(Some(Egress::NoWorkspace), async {
        let config = |base: &str, model: &str| -> Config {
            Config::parse(&format!(
                "[providers.local]\ntype = \"ollama\"\nbase_url = \"{base}\"\nmax_retries = 0\n\
                 [decision]\nmodel = \"local/{model}\"\n"
            ))
            .unwrap()
        };
        let probing = Probing::Online {
            timeout: Duration::from_secs(5),
        };
        let stub = DecisionStub::start().await;
        let mut report = Report::default();
        check_decision_model(&mut report, &config(stub.base_url(), "laya"), probing).await;
        let checks = find(&report, Area::Decision);
        let last = checks.last().unwrap();
        assert_eq!(
            (last.status, last.summary.as_str()),
            (Status::Ok, "local/laya: a decision call was answered"),
            "{checks:?}"
        );
        assert!(checks.iter().all(|c| c.status == Status::Ok), "{checks:?}");

        let refusing =
            DecisionStub::with_rule(|_| Some(Fault::new(404, "404 page not found"))).await;
        let mut report = Report::default();
        check_decision_model(&mut report, &config(refusing.base_url(), "laya"), probing).await;
        let failed = find(&report, Area::Decision);
        let last = failed.last().unwrap();
        assert_eq!(last.status, Status::Fail, "{failed:?}");
        assert!(last.summary.contains("404 page not found"), "{last:?}");
        assert!(
            last.fix.as_deref().is_some_and(|f| f.contains("0.40.0")),
            "{last:?}"
        );

        let mut report = Report::default();
        check_decision_model(&mut report, &Config::default(), probing).await;
        assert!(find(&report, Area::Decision).is_empty());

        let mut report = Report::default();
        check_decision_model(
            &mut report,
            &config(stub.base_url(), "laya"),
            Probing::Offline,
        )
        .await;
        let offline = find(&report, Area::Decision);
        assert!(
            offline.iter().all(|c| c.status == Status::Ok),
            "{offline:?}"
        );
        assert_eq!(stub.requests(), 2, "offline sends nothing");
    })
    .await;
}

/// A mock issuer for the doctor: its discovery document lists `grants`,
/// and it counts token requests.
#[expect(clippy::unwrap_used, reason = "test")]
async fn grant_listing_issuer(
    grants: &'static [&'static str],
) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = listener
        .local_addr()
        .map(|a| format!("http://{a}"))
        .unwrap_or_default();
    let tokens = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (served, counted) = (base.clone(), std::sync::Arc::clone(&tokens));
    tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).await.unwrap_or(0);
            let head = String::from_utf8_lossy(buf.get(..n).unwrap_or_default()).to_string();
            let target = head.split_whitespace().nth(1).unwrap_or("/").to_owned();
            let (status, body) = if target == "/.well-known/openid-configuration" {
                (
                    "200 OK",
                    serde_json::json!({
                        "issuer": served,
                        "authorization_endpoint": format!("{served}/authorize"),
                        "token_endpoint": format!("{served}/token"),
                        "grant_types_supported": grants,
                    })
                    .to_string(),
                )
            } else {
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                (
                    "400 Bad Request",
                    String::from("{\"error\":\"invalid_client\"}"),
                )
            };
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            drop(stream.write_all(response.as_bytes()).await);
        }
    });
    (base, tokens)
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn on_behalf_of_without_an_actor_checks_the_grant_list_and_requests_no_token() {
    let options = Options {
        probing: Probing::Online {
            timeout: Duration::from_secs(5),
        },
        ..Options::default()
    };
    let provider = |issuer: &str| {
        format!(
            "[general]\nchat_model = \"gw/m\"\n[providers.gw]\ntype = \"openai\"\n\
             base_url = \"https://gw.example.com/v1\"\nauth = \"oauth\"\n[providers.gw.oauth]\n\
             issuer_url = \"{issuer}\"\nclient_id = \"quack\"\nclient_auth = \"private_key_jwt\"\n\
             grant = \"on-behalf-of\"\nactor = false\n"
        )
    };

    let (issuer, tokens) = grant_listing_issuer(&[
        "authorization_code",
        "urn:ietf:params:oauth:grant-type:token-exchange",
    ])
    .await;
    let dir = tempfile::tempdir().unwrap();
    let report = run(&inspection(dir.path(), Some(&provider(&issuer))), &options).await;
    let chat = find(&report, Area::ChatModel);
    let check = *chat.first().unwrap();
    assert_eq!(check.status, Status::Ok, "{check:#?}");
    assert!(check.summary.contains("on behalf of"), "{check:#?}");
    assert!(check.summary.contains("token-exchange"), "{check:#?}");
    assert!(!check.summary.contains("actor)"), "{check:#?}");
    // No client-credentials token, nor any other, was asked for.
    assert_eq!(tokens.load(std::sync::atomic::Ordering::SeqCst), 0);

    let (issuer, tokens) = grant_listing_issuer(&["authorization_code"]).await;
    let dir = tempfile::tempdir().unwrap();
    let report = run(&inspection(dir.path(), Some(&provider(&issuer))), &options).await;
    let check = *find(&report, Area::ChatModel).first().unwrap();
    assert_eq!(check.status, Status::Fail, "{check:#?}");
    assert!(
        check.summary.contains("grant_types_supported"),
        "{check:#?}"
    );
    assert_eq!(tokens.load(std::sync::atomic::Ordering::SeqCst), 0);
}
