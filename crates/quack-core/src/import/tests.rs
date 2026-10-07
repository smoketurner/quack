use super::*;
use crate::llm::Embeddings;
use crate::proxy::{Environment, Variable};
use crate::storage::profile::ColumnTypes;

#[test]
fn urls_classify_and_redact() {
    assert_eq!(
        SourceUrl::from("sqlite:/tmp/x.db").kind().ok(),
        Some(SourceKind::Sqlite)
    );
    assert_eq!(
        SourceUrl::from("https://x/y.csv").kind().ok(),
        Some(SourceKind::Http)
    );
    for unsupported in [
        "postgres://u:p@h/db",
        "postgresql://h/db",
        "mysql://h/db",
        "s3://b/k.csv",
    ] {
        assert!(
            SourceUrl::from(unsupported).kind().is_err(),
            "{unsupported}"
        );
    }
    assert!(SourceUrl::from("ftp://h/f").kind().is_err());
    assert_eq!(
        SourceUrl::from("postgres://alice:secret@db.local:5432/sales").redacted(),
        "postgres://alice:***@db.local:5432/sales"
    );
    assert_eq!(
        SourceUrl::from("postgres://db.local/sales").redacted(),
        "postgres://db.local/sales"
    );
    assert_eq!(
        SourceUrl::from("sqlite:/tmp/x.db").redacted(),
        "sqlite:/tmp/x.db"
    );
}

/// A raw `@` in the password must not leak its suffix, an `@` in the
/// path or query of a URL with no user info must not gain a spurious
/// `:***`, and the empty-username `:password@` shape still drops the
/// whole user info.
#[test]
fn redaction_drops_the_whole_password_and_keeps_non_userinfo_urls() {
    // Passwords carrying a raw `@`: the whole password is gone.
    assert_eq!(
        SourceUrl::from("postgres://user:p@ss@host/db").redacted(),
        "postgres://user:***@host/db"
    );
    assert_eq!(
        SourceUrl::from("postgres://user:p@ss@10.255.255.1:1/db").redacted(),
        "postgres://user:***@10.255.255.1:1/db"
    );
    // A percent-encoded `@` already redacted correctly; it still does.
    assert_eq!(
        SourceUrl::from("postgres://user:p%40ss@host/db").redacted(),
        "postgres://user:***@host/db"
    );
    // Username only: the marker password is inserted, as before.
    assert_eq!(
        SourceUrl::from("postgres://alice@db.local/sales").redacted(),
        "postgres://alice:***@db.local/sales"
    );
    // Empty username with a password: the user info is dropped.
    assert_eq!(
        SourceUrl::from("postgres://:secret@host/db").redacted(),
        "postgres://host/db"
    );
    assert_eq!(
        SourceUrl::from("postgres://:p@ss@host/db").redacted(),
        "postgres://host/db"
    );
    assert_eq!(
        SourceUrl::from("postgres://:secret@[::1]:5432/db?x=1#f").redacted(),
        "postgres://[::1]:5432/db?x=1#f"
    );
    // An `@` in the path or query with no user info is left untouched.
    assert_eq!(
        SourceUrl::from("https://example.com/data@2024/sales.csv").redacted(),
        "https://example.com/data@2024/sales.csv"
    );
    assert_eq!(
        SourceUrl::from("https://example.com/search?q=foo@bar").redacted(),
        "https://example.com/search?q=foo@bar"
    );
    // No credentials and a non-special scheme round-trip verbatim,
    // including a mixed-case scheme (no parser normalization here).
    assert_eq!(
        SourceUrl::from("SQLite:/tmp/x.db?mode=ro").redacted(),
        "SQLite:/tmp/x.db?mode=ro"
    );
    assert_eq!(
        SourceUrl::from("sqlite://a/b.db?x=1").redacted(),
        "sqlite://a/b.db?x=1"
    );
}

/// URLs `Url::parse` rejects never come back verbatim: a raw `/`, `?`, or
/// `#` in the password, or an invalid port, is masked from `://` to the
/// last `@`, and no fragment of the password survives.
#[test]
fn unparseable_urls_never_leak_the_password() {
    for (url, password, expected) in [
        (
            "postgres://user:pa/ss@host/db",
            "pa/ss",
            "postgres://user:***@host/db",
        ),
        (
            "postgres://user:pa?ss@host/db",
            "pa?ss",
            "postgres://user:***@host/db",
        ),
        (
            "postgres://user:pa#ss@host/db",
            "pa#ss",
            "postgres://user:***@host/db",
        ),
        (
            "postgres://user:s3cr3t@host:99999/db",
            "s3cr3t",
            "postgres://user:***@host:99999/db",
        ),
        (
            "postgres://user:s3cr3t@host:port/db",
            "s3cr3t",
            "postgres://user:***@host:port/db",
        ),
        (
            "postgres://:pa/ss@host/db",
            "pa/ss",
            "postgres://***@host/db",
        ),
        (
            "postgres://user:pa/ss@p@host:1/db",
            "pa/ss@p",
            "postgres://user:***@host:1/db",
        ),
    ] {
        assert!(reqwest::Url::parse(url).is_err(), "{url} should not parse");
        let red = SourceUrl::from(url).redacted();
        assert_eq!(red, expected, "{url}");
        for fragment in [password, "pa", "ss", "s3cr3t"] {
            assert!(
                !red.contains(fragment),
                "{url}: password fragment {fragment:?} leaked into {red}"
            );
        }
    }
}

/// The fallback leaves strings with no user info alone.
#[test]
fn masking_leaves_strings_without_user_info_unchanged() {
    assert_eq!(mask_userinfo("not a url"), "not a url");
    assert_eq!(
        mask_userinfo("postgres://host:99999/db"),
        "postgres://host:99999/db"
    );
    assert_eq!(mask_userinfo("sqlite:/tmp/a@b.db"), "sqlite:/tmp/a@b.db");
}

/// The redacted value holds none of the password and reparses to the
/// same host and path as the URL the import actually connects through.
#[expect(clippy::unwrap_used, reason = "test fixtures are known-good URLs")]
#[test]
fn redacted_url_reparses_to_the_same_host_with_no_password() {
    for url in [
        "postgres://user:p@ss@host/db",
        "postgres://user:p@ss@10.255.255.1:1/db",
        "postgres://alice:secret@db.local:5432/sales",
        "postgres://:secret@host/db",
        "postgres://:p@ss@host/db",
    ] {
        let red = SourceUrl::from(url).redacted();
        assert!(!red.contains("secret"), "{url}: secret leaked into {red}");
        assert!(!red.contains("p@ss"), "{url}: password leaked into {red}");
        assert!(!red.contains("ss@"), "{url}: '@' suffix leaked into {red}");
        let real = reqwest::Url::parse(url).unwrap();
        let redacted = reqwest::Url::parse(&red).unwrap();
        assert_eq!(real.host_str(), redacted.host_str(), "{url}: host changed");
        assert_eq!(real.path(), redacted.path(), "{url}: path changed");
    }
}

#[test]
fn queries_come_from_the_request_and_tables_are_checked() {
    let base = ImportRequest {
        url: SourceUrl::from(""),
        table: String::from("t"),
        query: None,
        source_table: None,
        limit: None,
        types: ColumnTypes::default(),
    };
    assert!(base.source_query().is_err());
    let by_table = ImportRequest {
        source_table: Some(String::from("public.orders")),
        ..base.clone()
    };
    assert_eq!(
        by_table.source_query().ok().as_deref(),
        Some("SELECT * FROM public.orders")
    );
    let bad = ImportRequest {
        source_table: Some(String::from("orders; DROP TABLE x")),
        ..base.clone()
    };
    assert!(bad.source_query().is_err());
    let by_query = ImportRequest {
        query: Some(String::from(" SELECT 1 AS n; ")),
        ..base
    };
    assert_eq!(
        by_query.source_query().ok().as_deref(),
        Some("SELECT 1 AS n")
    );
}

#[test]
fn private_addresses_are_recognized() {
    let private = [
        "127.0.0.1",
        "0.0.0.0",
        "10.1.2.3",
        "172.16.0.9",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "224.0.0.1",
        "::1",
        "::",
        "fd00::1",
        "fe80::1",
        "::ffff:127.0.0.1",
        "::ffff:10.0.0.1",
    ];
    for text in private {
        let ip: IpAddr = text.parse().unwrap_or_else(|_| IpAddr::from([1, 1, 1, 1]));
        assert!(is_private_address(ip), "{text}");
    }
    let public = [
        "1.1.1.1",
        "8.8.8.8",
        "100.128.0.1",
        "2606:4700::1111",
        "::ffff:1.1.1.1",
    ];
    for text in public {
        let ip: IpAddr = text
            .parse()
            .unwrap_or_else(|_| IpAddr::from([127, 0, 0, 1]));
        assert!(!is_private_address(ip), "{text}");
    }
}

#[tokio::test]
async fn loopback_hosts_are_refused_without_the_private_host_grant() {
    let download = Download {
        timeout: Duration::from_secs(5),
        max_mb: 1,
        hosts: HostReach::PublicOnly,
        proxies: &Proxies::new(Environment::default()),
    };
    let err = download
        .fetch(&SourceUrl::from("http://127.0.0.1:9/x.csv"), "x")
        .await
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("private address"), "{err}");
    let err = download
        .fetch(&SourceUrl::from("http://localhost:9/x.csv"), "x")
        .await
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("private address"), "{err}");
}

/// A URL names a file `quack ingest` loads as a table, or it is
/// refused before any network use; a type the URL passes goes on to
/// the host check, which refuses loopback here.
#[tokio::test]
async fn urls_take_every_extension_ingest_loads_as_a_table() {
    let download = Download {
        timeout: Duration::from_secs(5),
        max_mb: 1,
        hosts: HostReach::PublicOnly,
        proxies: &Proxies::new(Environment::default()),
    };
    for accepted in [
        "http://127.0.0.1:9/book.ods",
        "http://127.0.0.1:9/old.XLS?download=1",
        "http://127.0.0.1:9/data.pq#part",
        "http://127.0.0.1:9/rows.ndjson",
    ] {
        let err = download
            .fetch(&SourceUrl::from(accepted), "t")
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("private address"), "{accepted}: {err}");
    }
    for refused in ["http://127.0.0.1:9/notes.pdf", "http://127.0.0.1:9/data"] {
        let err = download
            .fetch(&SourceUrl::from(refused), "t")
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            err.contains("must name a table file") && err.contains(".ods"),
            "{refused}: {err}"
        );
    }
}

/// One HTTP exchange on a loopback port: answer `response` to the
/// first connection and return the URL to fetch.
async fn serve_once(response: &'static str) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap_or_else(|e| unreachable_bind(&e.to_string()));
    let port = listener.local_addr().map(|a| a.port()).unwrap_or_default();
    tokio::spawn(async move {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        let Ok((mut socket, _)) = listener.accept().await else {
            return;
        };
        let mut buf = [0_u8; 4096];
        drop(socket.read(&mut buf).await);
        drop(socket.write_all(response.as_bytes()).await);
        drop(socket.shutdown().await);
    });
    format!("http://127.0.0.1:{port}/data.csv")
}

/// Through a proxy the import neither resolves the name nor pins an
/// address: the request for a name no resolver answers reaches the
/// proxy, which is what a network with no outside resolver needs.
#[tokio::test]
async fn a_public_only_download_through_a_proxy_leaves_the_name_to_the_proxy() {
    let proxy =
        serve_once("HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\na,b\n1,2\n")
            .await;
    let proxy = proxy.trim_end_matches("/data.csv");
    let proxies = Proxies::new(Environment {
        http: Variable::named("HTTP_PROXY", proxy),
        ..Environment::default()
    });
    let download = Download {
        timeout: Duration::from_secs(5),
        max_mb: 1,
        hosts: HostReach::PublicOnly,
        proxies: &proxies,
    };
    let pulled = download
        .fetch(&SourceUrl::from("http://files.invalid/data.csv"), "t")
        .await;
    assert!(pulled.is_ok(), "{:?}", pulled.err().map(|e| e.to_string()));

    for private in ["http://10.0.0.1/x.csv", "http://[fd00::1]/x.csv"] {
        let err = download
            .fetch(&SourceUrl::from(private), "t")
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("a private address"), "{private}: {err}");
    }
}

/// Loopback is never proxied, so it still meets the address check.
#[tokio::test]
async fn a_proxy_does_not_open_this_machine_to_a_public_only_download() {
    let proxies = Proxies::new(Environment {
        http: Variable::named("HTTP_PROXY", "http://127.0.0.1:9"),
        ..Environment::default()
    });
    let download = Download {
        timeout: Duration::from_secs(5),
        max_mb: 1,
        hosts: HostReach::PublicOnly,
        proxies: &proxies,
    };
    for local in ["http://127.0.0.1:9/x.csv", "http://localhost:9/x.csv"] {
        let err = download
            .fetch(&SourceUrl::from(local), "t")
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("a private address"), "{local}: {err}");
    }
}

#[expect(clippy::panic, reason = "test helper: a loopback port must bind")]
fn unreachable_bind(msg: &str) -> tokio::net::TcpListener {
    panic!("cannot bind a loopback port: {msg}")
}

#[tokio::test]
async fn downloads_stop_at_the_byte_cap_and_the_owner_follows_redirects() {
    let owner = Download {
        timeout: Duration::from_secs(5),
        max_mb: 0,
        hosts: HostReach::Any,
        proxies: &Proxies::new(Environment::default()),
    };
    let url = serve_once(
        "HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n\r\na,b\n1,2\n3,4\n",
    )
    .await;
    let err = owner
        .fetch(&SourceUrl::from(url.as_str()), "t")
        .await
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("max_download_mb"), "{err}");

    let url = serve_once(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n6\r\na,b\n1,\r\n6\r\n2\n3,4\n\r\n0\r\n\r\n",
    )
    .await;
    let err = owner
        .fetch(&SourceUrl::from(url.as_str()), "t")
        .await
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("max_download_mb"), "{err}");

    let roomy = Download { max_mb: 1, ..owner };
    let url = serve_once(
        "HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n\r\na,b\n1,2\n3,4\n",
    )
    .await;
    let fetched = roomy.fetch(&SourceUrl::from(url.as_str()), "t").await;
    assert!(
        fetched.is_ok_and(|p| p.filename == "t.csv" && p.bytes == b"a,b\n1,2\n3,4\n"),
        "the capped download of a small file succeeds"
    );

    // The owner may reach any host, so a redirect is followed: here to
    // a closed port, which fails the download itself.
    let url = serve_once(
        "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:9/other.csv\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    )
    .await;
    let err = roomy
        .fetch(&SourceUrl::from(url.as_str()), "t")
        .await
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        err.contains("download failed") && !err.contains("302 Found"),
        "{err}"
    );
}

/// A file source has no query to limit: the cap applies after the
/// load and the table keeps the first `limit` rows.
#[tokio::test]
async fn downloaded_files_are_cut_to_the_row_cap() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| no_tempdir(&e.to_string()));
    let mut config = Config::default();
    config.general.data_dir = dir.path().to_path_buf();
    let db = open_writer(&config);
    let url = serve_once(
        "HTTP/1.1 200 OK\r\nContent-Length: 16\r\nConnection: close\r\n\r\nn,s\n1,a\n2,b\n3,c\n",
    )
    .await;
    let request = ImportRequest {
        url: url.into(),
        table: String::from("rows"),
        query: None,
        source_table: None,
        limit: Some(2),
        types: ColumnTypes::default(),
    };
    let summary = Importing {
        config: &config,
        db: &db,
        workspace_id: "ws",
        request: &request,
        policy: ImportPolicy::owner(),
        embedder: None::<&Embeddings>,
        control: RunControl::unobserved(),
    }
    .run()
    .await
    .unwrap_or_else(|e| no_import(&e.to_string()));
    assert_eq!(summary.table, "rows");
    assert_eq!(summary.rows, 2);
    assert_eq!(summary.columns, vec![String::from("n"), String::from("s")]);
    let kept = db
        .run(|db| db.execute_query("SELECT n FROM rows ORDER BY n"))
        .await
        .map(|r| r.rows.len())
        .unwrap_or_default();
    assert_eq!(kept, 2);
}

/// A raw `@` in an HTTP import's password is masked out of the summary
/// source — the same value that becomes the document title and the
/// tracing-log field — while the download still reaches the host the
/// parser identifies (the owner may reach loopback), so the `ss@`
/// suffix the old first-`@` split leaked no longer lands anywhere.
#[tokio::test]
async fn an_http_import_with_a_raw_at_in_the_password_masks_the_source() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| no_tempdir(&e.to_string()));
    let mut config = Config::default();
    config.general.data_dir = dir.path().to_path_buf();
    let db = open_writer(&config);
    let served = serve_once(
        "HTTP/1.1 200 OK\r\nContent-Length: 16\r\nConnection: close\r\n\r\nn,s\n1,a\n2,b\n3,c\n",
    )
    .await;
    // `serve_once` returns `http://127.0.0.1:<port>/data.csv`; inject
    // `user:p@ss@` userinfo so the password carries a raw `@`.
    let tail = match served.strip_prefix("http://") {
        Some(rest) => rest,
        None => served.as_str(),
    };
    let url = format!("http://user:p@ss@{tail}");
    let request = ImportRequest {
        url: url.into(),
        table: String::from("rows"),
        query: None,
        source_table: None,
        limit: None,
        types: ColumnTypes::default(),
    };
    let summary = Importing {
        config: &config,
        db: &db,
        workspace_id: "ws",
        request: &request,
        policy: ImportPolicy::owner(),
        embedder: None::<&Embeddings>,
        control: RunControl::unobserved(),
    }
    .run()
    .await
    .unwrap_or_else(|e| no_import(&e.to_string()));
    assert_eq!(summary.source, format!("http://user:***@{tail}"));
    assert!(!summary.source.contains("ss@"), "{}", summary.source);
    assert!(!summary.source.contains("p@ss"), "{}", summary.source);
}

#[expect(clippy::panic, reason = "test helper: a temp dir must exist")]
fn no_tempdir(msg: &str) -> tempfile::TempDir {
    panic!("cannot create a temp dir: {msg}")
}

#[expect(clippy::panic, reason = "test helper: the fixture files must exist")]
fn no_file(msg: &str) {
    panic!("cannot write a fixture file: {msg}")
}

#[expect(clippy::panic, reason = "test helper: the workspace must open")]
fn open_writer(config: &Config) -> Writer {
    WorkspaceDb::open(config, "ws")
        .and_then(Writer::spawn)
        .unwrap_or_else(|e| panic!("cannot open the workspace: {e}"))
}

#[expect(clippy::panic, reason = "test asserts Ok")]
fn no_import(msg: &str) -> ImportSummary {
    panic!("import failed: {msg}")
}

/// The control database and workspace files stay out of workspaces
/// even for the owner (issue #69), through every spelling of the URL;
/// a symlink into the data directory does not slip past.
#[tokio::test]
async fn sqlite_imports_refuse_quacks_own_data_directory() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| no_tempdir(&e.to_string()));
    let mut config = Config::default();
    config.general.data_dir = dir.path().join("data");
    std::fs::create_dir_all(&config.general.data_dir).unwrap_or_else(|e| no_file(&e.to_string()));
    let control = config.general.data_dir.join("control.db");
    std::fs::write(&control, b"").unwrap_or_else(|e| no_file(&e.to_string()));
    let link = dir.path().join("elsewhere.db");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&control, &link).unwrap_or_else(|e| no_file(&e.to_string()));
    let db = open_writer(&config);
    let control = control.display();
    let mut urls = vec![
        format!("sqlite:{control}"),
        format!("sqlite://{control}"),
        format!("SQLite:{control}?mode=ro"),
    ];
    if cfg!(unix) {
        urls.push(format!("sqlite:{}", link.display()));
    }
    for url in urls {
        let request = ImportRequest {
            url: url.clone().into(),
            table: String::from("x"),
            query: None,
            source_table: Some(String::from("users")),
            limit: None,
            types: ColumnTypes::default(),
        };
        let err = Importing {
            config: &config,
            db: &db,
            workspace_id: "ws",
            request: &request,
            policy: ImportPolicy::owner(),
            embedder: None::<&Embeddings>,
            control: RunControl::unobserved(),
        }
        .run()
        .await
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(err.contains("own data directory"), "{url}: {err}");
    }
    assert!(!SourceUrl::from("sqlite:/tmp/other.db").is_under(&config.general.data_dir));
    assert_eq!(
        SourceUrl::from("sqlite://a/b.db?x=1").sqlite_path(),
        Path::new("a/b.db")
    );
}

/// The source opens read-only by path: a file that is not there is an
/// error, never an empty database created in its place.
#[tokio::test]
async fn a_missing_sqlite_source_is_an_error_and_is_not_created() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| no_tempdir(&e.to_string()));
    let missing = dir.path().join("absent.db");
    let err = fetch_rows(&missing, "SELECT 1", 10)
        .await
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("cannot open the source"), "{err}");
    assert!(!missing.exists());
}
