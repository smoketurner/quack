#![expect(
    clippy::indexing_slicing,
    reason = "serde_json::Value indexing yields Null for a missing key, never a panic"
)]

use quack_core::storage::writer::Writer;

use quack_core::config::Config;

use super::*;
use quack_core::analysis::events::Delivery;
use quack_core::analysis::policy::{Approver, Hold};
use quack_core::classify::LabelSet;
use quack_core::ids::WorkspaceId;
use quack_core::jobs::JobState;
use quack_core::storage::control::AllowedProviders;
use quack_testkit::{self, ScriptedOllama};

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
        transport: Transport::Stdio,
        stopping: CancellationToken::new(),
        jobs: JobQueue::new(10),
        holder: String::from("quack mcp"),
        approvals: Approvals::new(),
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

fn text_of(result: &CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.clone()))
        .collect()
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
    quack_testkit::seed_dictating_note(&db).unwrap_or_else(|e| fail(&e.to_string()));
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
        transport: Transport::Stdio,
        stopping: CancellationToken::new(),
        jobs: JobQueue::new(10),
        holder: String::from("quack mcp"),
        approvals: Approvals::new(),
    });
    let result = server
        .query_for(
            QueryArgs {
                question: String::from("follow the maintenance note"),
                session_id: None,
                mode: None,
                document_ids: Vec::new(),
            },
            &Extensions::default(),
            Requester::default(),
            CancellationToken::new(),
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
        Some(quack_testkit::DICTATED),
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

/// `list_tables` gives the schema in one call and `sql` answers with the
/// typed table the agent's `run_sql` gives, the JSON beside it.
#[tokio::test(flavor = "multi_thread")]
async fn list_tables_and_sql_answer_in_typed_text() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let shared = workspace(dir.path());
    let writer = server_on(&shared, WritePolicy::Allow(Approver::Nobody));
    let created = writer
        .sql(
            Parameters(SqlArgs {
                sql: String::from("CREATE TABLE t AS SELECT 1 AS n UNION ALL SELECT 2"),
            }),
            Extensions::default(),
            Requester::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert_eq!(created.is_error, Some(false));
    let tables = writer
        .list_tables(Extensions::default())
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert_eq!(
        field(&tables, "tables"),
        serde_json::json!([{
            "name": "t",
            "estimated_rows": 2,
            "columns": [{ "name": "n", "type": "INTEGER" }],
        }])
    );
    assert_eq!(text_of(&tables), "t (table, ~2 rows): n INTEGER");
    let read = writer
        .sql(
            Parameters(SqlArgs {
                sql: String::from("SELECT n, n * 1.5 AS half FROM t ORDER BY n"),
            }),
            Extensions::default(),
            Requester::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert_eq!(
        text_of(&read),
        "| n:INTEGER | half:DECIMAL(12,1) |\n|---|---|\n| 1 | 1.5 |\n| 2 | 3 |\n(2 rows)\n"
    );
    assert_eq!(
        field(&read, "column_types"),
        serde_json::json!(["INTEGER", "DECIMAL(12,1)"])
    );
    assert_eq!(field(&read, "row_count_exact"), serde_json::json!(true));
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
            Requester::default(),
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
            Requester::default(),
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
            Requester::default(),
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
            Requester::default(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert_eq!(created.is_error, Some(false));
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
        transport: Transport::Stdio,
        stopping: CancellationToken::new(),
        jobs: JobQueue::new(10),
        holder: String::from("quack mcp"),
        approvals: Approvals::new(),
    });
    let ask = |session_id: Option<&str>, mode: Option<&str>| QueryArgs {
        question: String::from("how many?"),
        session_id: session_id.map(str::to_owned),
        mode: mode.map(str::to_owned),
        document_ids: Vec::new(),
    };
    let ext = Extensions::default();
    let run = |args: QueryArgs| {
        server.query_for(args, &ext, Requester::default(), CancellationToken::new())
    };
    let session_count = || async {
        db.run(|db| sessions::list_sessions(db, 10))
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
            .len()
    };

    for _ in 0..2 {
        let failed = run(ask(None, None))
            .await
            .unwrap_or_else(|e| fail(&e.message));
        let text = error_text(&failed);
        assert!(text.contains("the agent turn failed"), "{text}");
        assert!(!text.contains("does not exist"), "{text}");
        assert_eq!(session_count().await, 0);
    }

    let bad_mode = run(ask(None, Some("loud")))
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(
        error_text(&bad_mode).contains("unknown mode 'loud'; use one of: chat, query"),
        "{}",
        error_text(&bad_mode)
    );

    let unknown = run(ask(Some("nope"), None))
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(error_text(&unknown).contains("does not exist"));

    // A question limited to a document that is not there fails before the
    // model is asked, and leaves no session behind.
    let scoped = run(QueryArgs {
        question: String::from("how many?"),
        session_id: None,
        mode: None,
        document_ids: vec![String::from("missing.md")],
    })
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
    let failed = run(ask(Some(existing.as_str()), Some("query")))
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
            Requester::default(),
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

/// The questions a `classify` call sends back: what a preview returned.
fn labelled_set() -> LabelSet {
    serde_json::from_value(serde_json::json!({
        "key_column": "id",
        "text_columns": ["subject"],
        "questions": {
            "department": {"type": "choice", "instructions": "Which?",
                           "criteria": {"billing": null, "technical": null}},
            "churn": {"type": "noul", "instructions": "Will they cancel?"}
        }
    }))
    .unwrap_or_else(|e| fail(&e.to_string()))
}

/// A `classify` call for `table` with the questions given back.
fn classify_args(table: &str, preview: Option<u32>) -> ClassifyToolArgs {
    use quack_core::classify::Rows;

    ClassifyToolArgs {
        request: classify::Request {
            table: table.to_owned(),
            sentence: None,
            set: Some(labelled_set()),
            rows: Rows::Missing,
        },
        preview,
    }
}

/// `classify` previews for any connection, and runs only where writes are
/// allowed and only up to `[decision].interactive_budget` answers.
#[tokio::test(flavor = "multi_thread")]
async fn classify_previews_anywhere_and_runs_where_writes_are_allowed_up_to_the_budget() {
    let stub = quack_testkit::DecisionStub::start().await;
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = Config::parse(&format!(
        "[providers.local]\ntype = \"ollama\"\nbase_url = \"{}\"\nmax_retries = 0\n\
         [decision]\nmodel = \"local/laya\"\ninteractive_budget = 12\n",
        stub.base_url()
    ))
    .unwrap_or_else(|e| fail(&e.to_string()));
    config.general.data_dir = dir.path().to_path_buf();
    let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
    db.execute_statement(
        "CREATE TABLE tickets AS SELECT range AS id, 'billing issue' AS subject FROM range(5)",
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    db.execute_statement("CREATE TABLE many AS SELECT range AS id, 'a' AS subject FROM range(50)")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
    let shared = (config, db);
    let ext = Extensions::default();
    let none = Progress::none;

    let read_only = server_on(&shared, WritePolicy::Deny);
    let preview = read_only
        .classify_for(classify_args("tickets", Some(2)), &ext, none())
        .await
        .unwrap_or_else(|e| fail(&e.message));
    let report = field(&preview, "preview");
    assert_eq!(report["labelled"], 2, "{preview:?}");
    assert_eq!(report["remaining"], 5);
    assert_eq!(field(&preview, "draft")["origin"], "given");
    assert_eq!(field(&preview, "outline")["effect"]["kind"], "new_table");
    let denied = read_only
        .classify_for(classify_args("tickets", None), &ext, none())
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(error_text(&denied).contains("cannot write"), "{denied:?}");

    let writer = server_on(&shared, WritePolicy::Allow(Approver::Nobody));
    let run = writer
        .classify_for(classify_args("tickets", None), &ext, none())
        .await
        .unwrap_or_else(|e| fail(&e.message));
    let ran = field(&run, "run");
    assert_eq!(ran["labelled"], 5, "{run:?}");
    assert_eq!(ran["status"], "completed");
    assert_eq!(ran["output_table"], "tickets_labels");

    let too_big = writer
        .classify_for(classify_args("many", None), &ext, none())
        .await
        .unwrap_or_else(|e| fail(&e.message));
    let text = error_text(&too_big);
    assert!(
        text.contains("labelling 50 rows of many with 2 questions is too long to wait for here")
            && text.contains("quack classify"),
        "{text}"
    );
    let fits = writer
        .classify_for(classify_args("many", Some(5)), &ext, none())
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert_eq!(
        field(&fits, "preview")["labelled"],
        5,
        "a preview has no budget: {fits:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn classify_without_a_decision_model_says_what_to_set() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let server = server(dir.path(), WritePolicy::Allow(Approver::Nobody));
    let refused = server
        .classify_for(
            classify_args("t", None),
            &Extensions::default(),
            Progress::none(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(
        error_text(&refused).contains("[decision].model"),
        "{refused:?}"
    );
}

/// A sentence needs a chat model to draft the questions: without one the
/// call says what to set, and a preview keeps nothing.
#[tokio::test(flavor = "multi_thread")]
async fn classify_with_a_sentence_and_no_chat_model_says_what_to_set() {
    use quack_core::classify::Rows;

    let stub = quack_testkit::DecisionStub::start().await;
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = Config::parse(&format!(
        "[providers.local]\ntype = \"ollama\"\nbase_url = \"{}\"\nmax_retries = 0\n\
         [decision]\nmodel = \"local/laya\"\n",
        stub.base_url()
    ))
    .unwrap_or_else(|e| fail(&e.to_string()));
    config.general.data_dir = dir.path().to_path_buf();
    let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
    db.execute_statement(
        "CREATE TABLE tickets AS SELECT range AS id, 'a' AS subject FROM range(5)",
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
    let server = server_on(&(config, db), WritePolicy::Deny);
    let refused = server
        .classify_for(
            ClassifyToolArgs {
                request: classify::Request {
                    table: String::from("tickets"),
                    sentence: Some(String::from("what is each ticket about")),
                    set: None,
                    rows: Rows::Missing,
                },
                preview: Some(3),
            },
            &Extensions::default(),
            Progress::none(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.message));
    assert!(
        error_text(&refused).contains("needs a chat model"),
        "{refused:?}"
    );
}

/// A server over a seeded workspace whose model follows the note, reached
/// through an in-memory transport by a client built from `handler`.
async fn connect<H: rmcp::ClientHandler>(
    handler: H,
) -> (
    ScriptedOllama,
    tempfile::TempDir,
    rmcp::service::RunningService<rmcp::RoleClient, H>,
) {
    let ollama = ScriptedOllama::serve(ScriptedOllama::following_the_note())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = ollama.config().unwrap_or_else(|e| fail(&e.to_string()));
    config.general.data_dir = dir.path().to_path_buf();
    let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
    quack_testkit::seed_dictating_note(&db).unwrap_or_else(|e| fail(&e.to_string()));
    let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
    let reader = ReaderDb::open(&db, config.analysis.reader_pool_size).await;
    let server = McpServer::new(McpSetup {
        config,
        db,
        reader,
        workspace: WorkspaceRow {
            id: WorkspaceId::from("ws"),
            name: String::from("stdio"),
            classification: String::from("internal"),
            allowed_providers: AllowedProviders::All,
        },
        policy: WritePolicy::Allow(Approver::Nobody),
        user_id: None,
        transport: Transport::Stdio,
        stopping: CancellationToken::new(),
        jobs: JobQueue::new(10),
        holder: String::from("quack mcp"),
        approvals: Approvals::new(),
    });
    let (server_side, client_side) = tokio::io::duplex(1 << 16);
    tokio::spawn(async move {
        match rmcp::ServiceExt::serve(server, server_side).await {
            Ok(running) => drop(running.waiting().await),
            Err(e) => tracing::error!(error = %e, "the test server did not start"),
        }
    });
    let client = rmcp::ServiceExt::serve(handler, client_side)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    (ollama, dir, client)
}

/// A client that keeps every progress message it is sent.
#[derive(Clone, Default)]
struct Progressed(Arc<std::sync::Mutex<Vec<String>>>);

impl rmcp::ClientHandler for Progressed {
    fn get_info(&self) -> rmcp::model::ClientConfig {
        rmcp::model::ClientConfig::default()
    }

    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: rmcp::service::NotificationContext<rmcp::RoleClient>,
    ) {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(params.message.unwrap_or_default());
    }
}

fn question() -> rmcp::model::JsonObject {
    let args = serde_json::json!({ "question": "follow the maintenance note" });
    serde_json::from_value(args).unwrap_or_else(|e| fail(&e.to_string()))
}

/// A `query` call with a progress token hears the turn start before any
/// work, then each tool call as it starts and ends; the answer still
/// comes back whole.
#[tokio::test(flavor = "multi_thread")]
async fn query_reports_its_turn_to_a_client_with_a_progress_token() {
    let progressed = Progressed::default();
    let (_ollama, _dir, client) = connect(progressed.clone()).await;
    let mut params = CallToolRequestParams::new("query").with_arguments(question());
    params.meta = Some(rmcp::model::RequestMetaObject::with_progress_token(
        ProgressToken(rmcp::model::NumberOrString::Number(7)),
    ));
    let response = client
        .call_tool_once(params)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let CallToolResponse::Complete(result) = response else {
        fail("a client without the tasks extension gets the result itself")
    };
    assert_eq!(field(&result, "write_refused"), true, "{result:?}");
    let messages = progressed
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(
        messages.first().map(String::as_str),
        Some(TURN_STARTING),
        "{messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|m| m.starts_with("search_documents: maintenance note")),
        "{messages:?}"
    );
    assert!(
        messages.iter().any(|m| m.starts_with("run_sql: ")),
        "{messages:?}"
    );
    drop(client.cancel().await);
}

/// A client that declared the tasks extension gets a task back at once,
/// polls it to completion, and reads the same answer from it; without
/// the declaration the same server answers directly.
#[tokio::test(flavor = "multi_thread")]
async fn query_runs_as_a_task_for_a_client_that_declares_tasks() {
    use rmcp::model::{ClientCapabilities, ClientConfig, TaskPayload};

    let handler = ClientConfig::new(
        ClientCapabilities::builder().enable_tasks().build(),
        Implementation::new("test", "0"),
    );
    let (_ollama, _dir, client) = connect(handler).await;
    let response = client
        .call_tool_once(CallToolRequestParams::new("query").with_arguments(question()))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let CallToolResponse::Task(created) = response else {
        fail("a client with the tasks extension gets a task")
    };
    assert_eq!(
        created.task.status_message.as_deref(),
        Some(TURN_STARTING),
        "{created:?}"
    );
    let task_id = created.task.task_id.clone();
    let done = loop {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let info = client
            .get_task(GetTaskParams::new(task_id.clone()))
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        if info.task.status().is_terminal() {
            break info.task;
        }
    };
    let TaskPayload::Completed { result } = done.payload else {
        fail(&format!("the task did not complete: {:?}", done.task))
    };
    let result: CallToolResult = serde_json::from_value(serde_json::Value::Object(result))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(field(&result, "write_refused"), true, "{result:?}");
    assert!(
        field(&result, "session_id").as_str().is_some(),
        "{result:?}"
    );
    drop(client.cancel().await);
}

/// The server on the workspace socket answers `initialize` with the
/// holder's name, `sql` with the rows (a write refused by the holder's
/// policy), and `quack://workspace/jobs` with the workspace's jobs; the
/// socket file goes with the server.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn the_socket_serves_the_holder_to_a_client() {
    use crate::client::HolderClient;
    use crate::socket::SocketServer;
    use quack_core::forward::Listener;
    use quack_core::jobs::{JobKind, JobSpec};
    use quack_core::storage::workspace::STRUCTURED_CELL_CHARS;

    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let (config, db) = workspace(dir.path());
    db.run(|db| {
        db.execute_statement("CREATE TABLE t AS SELECT 1 AS n UNION ALL SELECT 2")
            .map(drop)
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
    let jobs = JobQueue::new(10);
    let workspace_id = WorkspaceId::from("ws");
    let server = McpServer::new(McpSetup {
        config: config.clone(),
        db: Arc::clone(&db),
        reader: ReaderDb::new(Arc::clone(&db)),
        workspace: WorkspaceRow {
            id: workspace_id.clone(),
            name: String::from("ws"),
            classification: String::from("internal"),
            allowed_providers: AllowedProviders::All,
        },
        policy: WritePolicy::Deny,
        user_id: None,
        transport: Transport::Socket,
        stopping: CancellationToken::new(),
        jobs: jobs.clone(),
        holder: String::from("the quack terminal session"),
        approvals: Approvals::new(),
    });
    let done = jobs.submit(
        JobSpec::new(JobKind::Ingest, "notes.md").workspace(workspace_id.clone()),
        |_| async { Ok(String::from("1 chunk")) },
    );
    jobs.wait(done.id).await;
    let workspace_dir = config.workspace_dir("ws");
    let listener = Listener::bind(&workspace_dir, &workspace_id)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let socket_path = listener.at().path().to_path_buf();
    let socket = SocketServer::serve(listener, server);

    let client = HolderClient::connect(&workspace_dir, &workspace_id)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()))
        .unwrap_or_else(|| fail("no holder answered"));
    assert_eq!(client.holder(), "the quack terminal session");
    let rows = client
        .sql("SELECT n FROM t ORDER BY n")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(rows.results.columns, vec!["n"]);
    assert_eq!(rows.results.rows.len(), 2);
    assert!(!rows.truncated);
    assert_eq!(rows.cells_cut, 0);
    // A wide value comes back bounded, not whole: the holder cut it and
    // says so.
    let wide = client
        .sql("SELECT repeat('x', 200000) AS big")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(wide.cells_cut, 1);
    let cell = wide
        .results
        .rows
        .first()
        .and_then(|row| row.first())
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| fail("no cell"));
    assert_eq!(
        cell.len(),
        STRUCTURED_CELL_CHARS + "\u{2026}(+134464 chars)".len()
    );
    assert!(cell.ends_with("\u{2026}(+134464 chars)"), "{}", cell.len());
    let refused = client
        .sql("CREATE TABLE u (x INTEGER)")
        .await
        .err()
        .unwrap_or_else(|| fail("a write ran through the holder"));
    let message = refused.to_string();
    assert!(
        message.contains("the quack terminal session")
            && message.contains("does not run writes for another process")
            && message.contains("--allow-write"),
        "{message}"
    );
    let listed = client.jobs().await.unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].label, "notes.md");
    assert_eq!(listed[0].outcome.as_deref(), Some("1 chunk"));
    client.close().await;

    socket.close(Duration::from_secs(5)).await;
    assert!(!socket_path.exists());
    assert!(
        HolderClient::connect(&workspace_dir, &workspace_id)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
            .is_none()
    );
}

/// Closing the socket server with no grace at all, before its accept loop
/// was ever polled (a one-shot command that held the workspace for
/// milliseconds), still removes the socket file: the accept loop's end is
/// awaited whatever the grace, and only the sessions are given up on.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn closing_the_socket_at_once_leaves_no_socket_file() {
    use crate::socket::SocketServer;
    use quack_core::forward::Listener;

    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let (config, db) = workspace(dir.path());
    let server = server_on(&(config.clone(), db), WritePolicy::Deny);
    let workspace_id = WorkspaceId::from("ws");
    let listener = Listener::bind(&config.workspace_dir("ws"), &workspace_id)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let socket_path = listener.at().path().to_path_buf();
    assert!(socket_path.exists());
    SocketServer::serve(listener, server)
        .close(Duration::ZERO)
        .await;
    assert!(!socket_path.exists(), "the socket file was left behind");
}

/// A `sql` write sent over the socket under `Ask` (the terminal session
/// holds the workspace) is a `sql` job that asks at the session's prompt
/// through the approvals: a yes runs it, a no refuses it, and with nobody
/// listening it is refused at once.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn a_socket_sql_write_asks_the_listener_or_is_refused() {
    use crate::client::{CommandClient, HolderClient};
    use crate::socket::SocketServer;
    use quack_core::analysis::events::Decision;
    use quack_core::forward::Listener;

    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let (config, db) = workspace(dir.path());
    let approvals = Approvals::new();
    let jobs = JobQueue::new(10);
    let workspace_id = WorkspaceId::from("ws");
    let server = McpServer::new(McpSetup {
        config: config.clone(),
        db: Arc::clone(&db),
        reader: ReaderDb::new(Arc::clone(&db)),
        workspace: WorkspaceRow {
            id: workspace_id.clone(),
            name: String::from("ws"),
            classification: String::from("internal"),
            allowed_providers: AllowedProviders::All,
        },
        policy: WritePolicy::Ask,
        user_id: None,
        transport: Transport::Socket,
        stopping: CancellationToken::new(),
        jobs: jobs.clone(),
        holder: String::from("the quack terminal session"),
        approvals: approvals.clone(),
    });
    let workspace_dir = config.workspace_dir("ws");
    let listener = Listener::bind(&workspace_dir, &workspace_id)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let socket = SocketServer::serve(listener, server);
    // The client names itself, and hears where its write waits.
    let reports: Arc<std::sync::Mutex<Vec<String>>> = Arc::default();
    let client = HolderClient::connect_with(&workspace_dir, &workspace_id, {
        let reports = Arc::clone(&reports);
        CommandClient::new("quack -q", move |message| {
            if let Ok(mut reports) = reports.lock() {
                reports.push(message.to_owned());
            }
        })
    })
    .await
    .unwrap_or_else(|e| fail(&e.to_string()))
    .unwrap_or_else(|| fail("no holder answered"));

    // Nobody listens: refused at once, and audited as a job that did not run.
    let refused = client
        .sql("CREATE TABLE u (x INTEGER)")
        .await
        .err()
        .unwrap_or_else(|| fail("a write ran with nobody to approve it"));
    assert!(
        refused
            .to_string()
            .contains("not approved at the quack terminal session"),
        "{refused}"
    );

    // The session listens: the first ask is a yes, the second a no.
    let mut asked = approvals.listen();
    let answering = tokio::spawn({
        let jobs = jobs.clone();
        async move {
            let first = asked.recv().await.unwrap_or_else(|| fail("nothing asked"));
            assert_eq!(first.request.sql, "CREATE TABLE u (x INTEGER)");
            assert!(first.session.is_none());
            assert_eq!(first.client, "quack -q");
            let job = jobs
                .get(first.job)
                .unwrap_or_else(|| fail("the statement is not a job"));
            assert_eq!(job.kind, JobKind::Sql);
            assert_eq!(job.label, "CREATE TABLE u (x INTEGER)");
            assert_eq!(first.request.answer(Decision::Allow), Delivery::Delivered);
            let second = asked.recv().await.unwrap_or_else(|| fail("nothing asked"));
            assert_eq!(second.request.answer(Decision::Deny), Delivery::Delivered);
        }
    });
    let ran = client
        .sql("CREATE TABLE u (x INTEGER)")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(ran.results.rows.is_empty(), "{ran:?}");
    let refused = client
        .sql("CREATE TABLE v (x INTEGER)")
        .await
        .err()
        .unwrap_or_else(|| fail("a refused write ran"));
    assert!(refused.to_string().contains("not approved"), "{refused}");
    answering.await.unwrap_or_else(|e| fail(&e.to_string()));
    let tables = client
        .sql("SELECT table_name FROM information_schema.tables WHERE table_name IN ('u', 'v') ORDER BY 1")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(tables.results.rows, vec![vec![serde_json::json!("u")]]);
    client.close().await;
    socket.close(Duration::from_secs(5)).await;
    let reported = reports.lock().map(|r| r.clone()).unwrap_or_default();
    assert!(
        reported.iter().any(|message| {
            message.contains("waiting for approval at the quack terminal session (up to 300 s)")
        }),
        "{reported:?}"
    );
}

/// A `query` turn that wants a write under `Ask` asks whoever listens on
/// the approvals (the terminal session, over the socket) and runs it on a
/// yes; with nobody listening the write is refused.
#[tokio::test(flavor = "multi_thread")]
async fn a_socket_turns_write_asks_the_listener_or_is_refused() {
    use quack_core::analysis::events::Decision;

    for listened in [true, false] {
        let ollama = ScriptedOllama::serve(ScriptedOllama::following_the_note())
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = ollama.config().unwrap_or_else(|e| fail(&e.to_string()));
        config.general.data_dir = dir.path().to_path_buf();
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        quack_testkit::seed_dictating_note(&db).unwrap_or_else(|e| fail(&e.to_string()));
        let db: SharedDb = Arc::new(Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string())));
        let reader = ReaderDb::open(&db, config.analysis.reader_pool_size).await;
        let approvals = Approvals::new();
        let jobs = JobQueue::new(10);
        let server = McpServer::new(McpSetup {
            config,
            db: Arc::clone(&db),
            reader,
            workspace: WorkspaceRow {
                id: WorkspaceId::from("ws"),
                name: String::from("ws"),
                classification: String::from("internal"),
                allowed_providers: AllowedProviders::All,
            },
            policy: WritePolicy::Ask,
            user_id: None,
            transport: Transport::Socket,
            stopping: CancellationToken::new(),
            jobs: jobs.clone(),
            holder: String::from("the quack terminal session"),
            approvals: approvals.clone(),
        });
        let answering = listened.then(|| {
            let mut asked = approvals.listen();
            let jobs = jobs.clone();
            tokio::spawn(async move {
                let asked = asked
                    .recv()
                    .await
                    .unwrap_or_else(|| fail("nothing was asked"));
                assert_eq!(asked.request.sql, quack_testkit::DICTATED);
                // The turn is a job the person can see.
                let job = jobs
                    .get(asked.job)
                    .unwrap_or_else(|| fail("the turn is not a job"));
                assert_eq!(job.kind, JobKind::Chat);
                assert!(job.label.starts_with("mcp: "), "{}", job.label);
                asked.request.answer(Decision::Allow)
            })
        });
        let result = server
            .query_for(
                QueryArgs {
                    question: String::from("follow the maintenance note"),
                    session_id: None,
                    mode: None,
                    document_ids: Vec::new(),
                },
                &Extensions::default(),
                Requester::default(),
                CancellationToken::new(),
            )
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(field(&result, "write_refused"), !listened, "{result:?}");
        if let Some(answering) = answering {
            let delivery = answering.await.unwrap_or_else(|e| fail(&e.to_string()));
            assert_eq!(delivery, Delivery::Delivered);
        }
        // The answer comes off the events a moment before the job records
        // its end.
        let listed = jobs.list();
        assert_eq!(listed.len(), 1, "{listed:?}");
        let ended = jobs
            .wait(listed[0].id)
            .await
            .unwrap_or_else(|| fail("the turn's job vanished"));
        assert_eq!(ended.state, JobState::Succeeded, "{ended:?}");
    }
}
