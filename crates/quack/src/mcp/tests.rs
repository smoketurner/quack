use quack_core::storage::writer::Writer;

use quack_core::config::Config;

use super::*;
use crate::scripted_ollama::{self, ScriptedOllama};
use quack_core::analysis::policy::{Approver, Hold};
use quack_core::ids::WorkspaceId;
use quack_core::storage::control::AllowedProviders;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

fn server(dir: &std::path::Path, policy: WritePolicy) -> McpServer {
    server_on(&workspace(dir), policy)
}

/// The workspace a test's servers share: opened once, since Windows locks
/// the file exclusively and a second open in the process is refused.
fn workspace(dir: &std::path::Path) -> (Config, SharedDb) {
    let mut config = Config::default();
    config.general.data_dir = dir.to_path_buf();
    let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
    let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
    (config, db)
}

fn server_on((config, db): &(Config, SharedDb), policy: WritePolicy) -> McpServer {
    let reader = ReaderDb::new(Arc::clone(db));
    McpServer::new(McpSetup {
        config: config.clone(),
        db: Arc::clone(db),
        reader,
        workspace: WorkspaceRow {
            id: WorkspaceId::from("ws"),
            name: String::from("stdio"),
            classification: String::from("internal"),
            allowed_providers: AllowedProviders::All,
        },
        policy,
        user_id: None,
        auditor: Auditor::None,
    })
}

fn field(result: &CallToolResult, key: &str) -> serde_json::Value {
    result
        .structured_content
        .as_ref()
        .and_then(|v| v.get(key))
        .cloned()
        .unwrap_or_default()
}

fn error_text(result: &CallToolResult) -> String {
    assert_eq!(result.is_error, Some(true));
    result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.clone()))
        .collect()
}

/// `--allow-write` lets a turn write until it has read document text;
/// nobody can approve a write here, so the one after is refused, and the
/// result says so in its structured content and its text.
#[tokio::test(flavor = "multi_thread")]
async fn a_turn_that_read_a_document_is_refused_its_write_under_allow_write() {
    let ollama = ScriptedOllama::serve(ScriptedOllama::following_the_note())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = ollama.config().unwrap_or_else(|e| fail(&e.to_string()));
    config.general.data_dir = dir.path().to_path_buf();
    let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
    scripted_ollama::seed_dictating_note(&db).unwrap_or_else(|e| fail(&e.to_string()));
    let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
    let reader = ReaderDb::open(&db, config.analysis.reader_pool_size).await;
    let server = McpServer::new(McpSetup {
        config,
        db: Arc::clone(&db),
        reader,
        workspace: WorkspaceRow {
            id: WorkspaceId::from("ws"),
            name: String::from("stdio"),
            classification: String::from("internal"),
            allowed_providers: AllowedProviders::All,
        },
        policy: WritePolicy::Allow(Approver::Nobody),
        user_id: None,
        auditor: Auditor::None,
    });
    let result = server
        .query(
            Parameters(QueryArgs {
                question: String::from("follow the maintenance note"),
                session_id: None,
                mode: None,
                document_ids: Vec::new(),
            }),
            Extensions::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(field(&result, "write_refused"), true, "{result:?}");
    let refused = field(&result, "steps")
        .pointer("/1")
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        refused.get("detail").and_then(|d| d.as_str()),
        Some(scripted_ollama::DICTATED),
        "{refused}"
    );
    assert_eq!(
        refused.get("summary").and_then(|s| s.as_str()),
        Some(Hold::ReadDocuments.summary()),
        "{refused}"
    );
    let text: String = result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.clone()))
        .collect();
    assert!(text.contains("was refused; its step says why"), "{text}");
    let tables = db
        .run(WorkspaceDb::list_tables)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(tables, ["customers"], "the dictated drop did not run");
}

#[tokio::test(flavor = "multi_thread")]
async fn stdio_tools_gate_writes_and_serve_resources() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let shared = workspace(dir.path());
    let read_only = server_on(&shared, WritePolicy::Deny);
    let denied = read_only
        .sql(
            Parameters(SqlArgs {
                sql: String::from("CREATE TABLE t AS SELECT 1 AS n"),
            }),
            Extensions::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(error_text(&denied).contains("cannot write"));
    let internal = read_only
        .sql(
            Parameters(SqlArgs {
                sql: String::from("SELECT * FROM _quack_documents"),
            }),
            Extensions::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(error_text(&internal).contains("internal tables"));
    let bad = read_only
        .sql(
            Parameters(SqlArgs {
                sql: String::from("SELEC 1"),
            }),
            Extensions::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert_eq!(bad.is_error, Some(true));

    let writer = server_on(&shared, WritePolicy::Allow(Approver::Nobody));
    let created = writer
        .sql(
            Parameters(SqlArgs {
                sql: String::from("CREATE TABLE t AS SELECT 1 AS n UNION ALL SELECT 2"),
            }),
            Extensions::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert_eq!(created.is_error, Some(false));
    let tables = writer
        .list_tables(Extensions::default())
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert_eq!(field(&tables, "tables"), serde_json::json!(["t"]));
    let described = writer
        .describe_table(
            Parameters(DescribeTableArgs {
                table: String::from("t"),
            }),
            Extensions::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert_eq!(field(&described, "row_count"), 2);
    assert_eq!(
        field(&described, "profile").get("row_count"),
        Some(&serde_json::json!(2)),
        "a table a statement made is profiled at once"
    );
    assert!(field(&described, "warnings").is_array());
    assert_eq!(
        field(&described, "columns")
            .get(0)
            .and_then(|c| c.get("name")),
        Some(&serde_json::json!("n"))
    );
    let missing = writer
        .describe_table(
            Parameters(DescribeTableArgs {
                table: String::from("zz"),
            }),
            Extensions::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(error_text(&missing).contains("no table"));
    let documents = writer
        .list_documents(
            Parameters(ListDocumentsArgs::default()),
            Extensions::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert_eq!(field(&documents, "documents"), serde_json::json!([]));
    let empty = writer
        .search(
            Parameters(SearchArgs {
                query: String::from("  "),
                ..SearchArgs::default()
            }),
            Extensions::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert_eq!(empty.is_error, Some(true));
}

/// `search` scopes to named documents, refusing one that is not there, and
/// explains its legs on request.
#[tokio::test(flavor = "multi_thread")]
async fn stdio_search_scopes_to_documents_and_explains() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let writer = server(dir.path(), WritePolicy::Deny);
    let unknown = writer
        .search(
            Parameters(SearchArgs {
                query: String::from("renewal"),
                document_ids: vec![String::from("missing.md")],
                ..SearchArgs::default()
            }),
            Extensions::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(error_text(&unknown).contains("no document matches 'missing.md'"));
    let explained = writer
        .search(
            Parameters(SearchArgs {
                query: String::from("renewal"),
                explain: true,
                ..SearchArgs::default()
            }),
            Extensions::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert_eq!(field(&explained, "chunks"), serde_json::json!([]));
    assert_eq!(
        field(&explained, "explain").get("rerank"),
        Some(&serde_json::json!("not reranked"))
    );
}

/// `query` with a model that cannot answer: the failure is reported,
/// the session it made is gone, and the next call is not stuck on a
/// deleted session id. A session the caller named survives the
/// failure, and a session nobody made is refused.
#[tokio::test(flavor = "multi_thread")]
async fn query_failures_leave_no_session_and_named_sessions_are_checked() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = Config::parse(
        "[general]\nchat_model = \"o/m\"\n[providers.o]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\n",
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    config.general.data_dir = dir.path().to_path_buf();
    let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
    let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
    let reader = ReaderDb::open(&db, config.analysis.reader_pool_size).await;
    let server = McpServer::new(McpSetup {
        config,
        db: Arc::clone(&db),
        reader,
        workspace: WorkspaceRow {
            id: WorkspaceId::from("ws"),
            name: String::from("stdio"),
            classification: String::from("internal"),
            allowed_providers: AllowedProviders::All,
        },
        policy: WritePolicy::Deny,
        user_id: None,
        auditor: Auditor::None,
    });
    let ask = |session_id: Option<&str>, mode: Option<&str>| {
        Parameters(QueryArgs {
            question: String::from("how many?"),
            session_id: session_id.map(str::to_owned),
            mode: mode.map(str::to_owned),
            document_ids: Vec::new(),
        })
    };
    let session_count = || async {
        db.run(|db| sessions::list_sessions(db, 10))
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
            .len()
    };

    for _ in 0..2 {
        let failed = server
            .query(ask(None, None), Extensions::default())
            .await
            .unwrap_or_else(|e| fail(&e.message));
        let text = error_text(&failed);
        assert!(text.contains("the agent turn failed"), "{text}");
        assert!(!text.contains("does not exist"), "{text}");
        assert_eq!(session_count().await, 0);
    }

    let bad_mode = server
        .query(ask(None, Some("loud")), Extensions::default())
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(
        error_text(&bad_mode).contains("unknown mode 'loud'; use one of: chat, query"),
        "{}",
        error_text(&bad_mode)
    );

    let unknown = server
        .query(ask(Some("nope"), None), Extensions::default())
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(error_text(&unknown).contains("does not exist"));

    // A question limited to a document that is not there fails before the
    // model is asked, and leaves no session behind.
    let scoped = server
        .query(
            Parameters(QueryArgs {
                question: String::from("how many?"),
                session_id: None,
                mode: None,
                document_ids: vec![String::from("missing.md")],
            }),
            Extensions::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(
        error_text(&scoped).contains("no document matches 'missing.md'"),
        "{}",
        error_text(&scoped)
    );
    assert_eq!(session_count().await, 0);

    let existing = db
        .run(|db| sessions::create_session(db, "o/m", ChatMode::Chat, None))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()))
        .id;
    let failed = server
        .query(
            ask(Some(existing.as_str()), Some("query")),
            Extensions::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(error_text(&failed).contains("the agent turn failed"));
    let kept = db
        .run(move |db| sessions::get_session(db, &existing))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    // The mode given with an existing session id does not change it.
    assert_eq!(kept.map(|s| s.mode), Some(ChatMode::Chat));
}

#[tokio::test(flavor = "multi_thread")]
async fn stdio_resources_render_tables_context_and_schemas() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let writer = server(dir.path(), WritePolicy::Allow(Approver::Nobody));
    let created = writer
        .sql(
            Parameters(SqlArgs {
                sql: String::from("CREATE TABLE t AS SELECT 1 AS n UNION ALL SELECT 2"),
            }),
            Extensions::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert_eq!(created.is_error, Some(false));
    assert_eq!(
        writer
            .resource_text(WorkspaceResource::Tables)
            .await
            .unwrap_or_else(|e| fail(&e.message))
            .as_deref(),
        Some("{\"tables\":[\"t\"]}")
    );
    assert_eq!(
        writer
            .resource_text(WorkspaceResource::Context)
            .await
            .unwrap_or_else(|e| fail(&e.message))
            .as_deref(),
        Some("")
    );
    assert!(
        writer
            .resource_text(WorkspaceResource::Schema("t"))
            .await
            .unwrap_or_else(|e| fail(&e.message))
            .is_some_and(|t| t.contains("\"row_count\":2"))
    );
    assert_eq!(
        writer
            .resource_text(WorkspaceResource::Schema("zz"))
            .await
            .unwrap_or_else(|e| fail(&e.message)),
        None
    );
    let schema = writer
        .resource_text(WorkspaceResource::OntologySchema)
        .await
        .unwrap_or_else(|e| fail(&e.message))
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok());
    assert_eq!(schema, serde_json::to_value(Ontology::json_schema()).ok());
    let info = writer.get_info();
    assert!(info.instructions.is_some_and(|i| i.contains("'stdio'")));
}

#[test]
fn resource_uris_round_trip() {
    let schema = WorkspaceResource::Schema("orders");
    for resource in WorkspaceResource::FIXED.into_iter().chain([schema]) {
        assert_eq!(WorkspaceResource::parse(&resource.uri()), Some(resource));
    }
    assert_eq!(
        schema.uri(),
        "quack://workspace/tables/orders/schema",
        "the template's shape"
    );
    assert_eq!(WorkspaceResource::parse("quack://elsewhere"), None);
    assert_eq!(WorkspaceResource::parse("quack://workspace/tables/t"), None);
    assert_eq!(WorkspaceResource::parse("quack://workspace/sessions"), None);
    assert_eq!(
        WorkspaceResource::OntologySchema.uri(),
        "quack://workspace/ontology/schema"
    );
}
