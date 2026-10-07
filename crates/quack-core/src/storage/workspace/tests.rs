use super::*;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

#[test]
fn the_sql_schema_quotes_what_needs_it_and_hides_internal_tables() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    let wide: Vec<String> = (0..=SqlSchema::MAX_COLUMNS)
        .map(|i| format!("1 AS c{i}"))
        .collect();
    for sql in [
        String::from(
            "CREATE TABLE sales (region VARCHAR, \"Revenue\" INTEGER, \"select\" INTEGER)",
        ),
        String::from("CREATE TABLE \"Order Items\" (id INTEGER)"),
        format!("CREATE TABLE wide AS SELECT {}", wide.join(", ")),
    ] {
        db.execute_statement(&sql)
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    let schema = db.sql_schema().unwrap_or_else(|e| fail(&e.to_string()));
    let names: Vec<(&str, &str)> = schema
        .tables
        .iter()
        .map(|t| (t.name.name.as_str(), t.name.sql.as_str()))
        .collect();
    assert_eq!(
        names,
        [
            ("Order Items", "\"Order Items\""),
            ("sales", "sales"),
            ("wide", "wide")
        ]
    );
    let columns = |at: usize| -> Vec<&str> {
        schema
            .tables
            .get(at)
            .map(|t| t.columns.iter().map(|c| c.sql.as_str()).collect())
            .unwrap_or_default()
    };
    let sales = columns(1);
    assert_eq!(sales, ["region", "\"Revenue\"", "\"select\""]);
    assert_eq!(
        columns(2).len(),
        usize::try_from(SqlSchema::MAX_COLUMNS).unwrap_or_default()
    );
    assert!(schema.truncated, "the wide table's last column was cut");
}

/// A row from before `tables` was recorded drops the one table named
/// after its file, and nothing else; a chunked document has none.
#[test]
fn deleting_a_row_without_recorded_tables_drops_its_file_table() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    for sql in [
        "CREATE TABLE sales AS SELECT 1 AS n",
        "CREATE TABLE sales_notes AS SELECT 2 AS n",
    ] {
        db.execute_statement(sql)
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    db.insert_document(&NewDocument::new(
        &DocumentId::from("d1"),
        "sales.csv",
        "text/csv",
        1,
    ))
    .unwrap_or_else(|e| fail(&e.to_string()));
    let doc = db
        .document(&DocumentId::from("d1"))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(doc.as_ref().is_some_and(|d| d.tables.is_none()));
    assert_eq!(
        doc.map(|d| d.fallback_tables()),
        Some(vec![String::from("sales")])
    );
    assert!(
        db.delete_document(&DocumentId::from("d1"))
            .is_ok_and(|existed| existed)
    );
    assert!(db.table_exists("sales").is_ok_and(|exists| !exists));
    assert!(db.table_exists("sales_notes").is_ok_and(|exists| exists));

    // A queued row with the same file name has no tables yet: deleting
    // it leaves the table another document loaded.
    db.execute_statement("CREATE TABLE sales AS SELECT 3 AS n")
        .unwrap_or_else(|e| fail(&e.to_string()));
    db.insert_document(&NewDocument::new(
        &DocumentId::from("owner"),
        "sales.csv",
        "text/csv",
        1,
    ))
    .unwrap_or_else(|e| fail(&e.to_string()));
    db.set_document_tables(&DocumentId::from("owner"), &[String::from("sales")])
        .unwrap_or_else(|e| fail(&e.to_string()));
    db.insert_document(&NewDocument::new(
        &DocumentId::from("queued"),
        "sales.csv",
        "text/csv",
        1,
    ))
    .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        db.delete_document(&DocumentId::from("queued"))
            .is_ok_and(|existed| existed)
    );
    assert!(db.table_exists("sales").is_ok_and(|exists| exists));

    db.insert_document(&NewDocument::new(
        &DocumentId::from("d2"),
        "notes.md",
        "text/markdown",
        1,
    ))
    .unwrap_or_else(|e| fail(&e.to_string()));
    let notes = db
        .document(&DocumentId::from("d2"))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(notes.is_some_and(|d| d.fallback_tables().is_empty()));
}

/// A stored source reads back as written; a NULL (rows from before the
/// column) is an upload; anything else is an error, not a guess.
#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn document_sources_read_back_and_unknown_ones_are_refused() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    let d1 = DocumentId::from("d1");
    let mut doc = NewDocument::new(&d1, "a.txt", "text/plain", 1);
    doc.source = DocumentSource::Import;
    db.insert_document(&doc).unwrap();
    let read = db
        .document(&DocumentId::from("d1"))
        .unwrap()
        .map(|d| d.source);
    assert_eq!(read, Some(DocumentSource::Import));
    db.connection()
        .execute("UPDATE _quack_documents SET source = NULL", [])
        .unwrap();
    let read = db
        .document(&DocumentId::from("d1"))
        .unwrap()
        .map(|d| d.source);
    assert_eq!(read, Some(DocumentSource::Upload));
    db.connection()
        .execute("UPDATE _quack_documents SET source = 'fax'", [])
        .unwrap();
    assert!(db.document(&DocumentId::from("d1")).is_err());
}

fn config_in(dir: &Path) -> Config {
    let mut config = Config::default();
    config.general.data_dir = dir.join("data");
    config
}

/// Two statements that differ only in their literals, spacing, and
/// case share a shape; a different column, a different statement kind,
/// and anything `DuckDB` cannot serialize do not.
#[test]
fn a_canceller_interrupts_only_the_work_it_guards() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4))
        .unwrap_or_else(|e| fail(&e.to_string()))
        .with_query_timeout(Duration::from_secs(60));
    // Cancelled before the work starts: it never runs.
    let early = QueryCanceller::default();
    early.cancel();
    assert!(matches!(
        db.cancellable(&early, |_| Ok(())),
        Err(Error::Cancelled)
    ));

    // Cancelled while a long statement runs: the statement stops.
    let canceller = QueryCanceller::default();
    let remote = canceller.clone();
    let stopper = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        remote.cancel();
    });
    let started = std::time::Instant::now();
    let outcome = db.cancellable(&canceller, |db| {
        db.execute_query_capped(
            "SELECT count(*) FROM range(100000000000) t(i) WHERE i % 7 = 0",
            10,
        )
    });
    assert!(outcome.is_err(), "the statement was interrupted");
    assert!(started.elapsed() < Duration::from_secs(30));
    assert!(stopper.join().is_ok());

    // The connection is free again, and a canceller no longer guarding
    // anything interrupts nothing.
    let after = QueryCanceller::default();
    assert!(
        db.cancellable(&after, |db| db.execute_query_capped("SELECT 1", 10))
            .is_ok()
    );
    after.cancel();
    assert!(db.execute_query_capped("SELECT 2", 10).is_ok());
}

#[test]
fn statement_shape_ignores_literals_and_source_positions() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    let shape = |sql: &str| {
        db.statement_shape(sql)
            .unwrap_or_else(|e| fail(&e.to_string()))
    };
    let nevada = shape("SELECT x FROM t WHERE s = 'NEVADA' AND n > 10 LIMIT 5");
    let carolina = shape("select x\n from t where s='NORTH CAROLINA' and n > 250 limit 5");
    assert!(nevada.is_some());
    assert_eq!(nevada, carolina);
    assert_ne!(
        nevada,
        shape("SELECT y FROM t WHERE s = 'NEVADA' AND n > 10 LIMIT 5")
    );
    assert_ne!(nevada, shape("SELECT x FROM t WHERE s = 'NEVADA' LIMIT 5"));
    assert_eq!(shape("CREATE TABLE t2 AS SELECT 1"), None);
    assert_eq!(shape("SELECT FROM WHERE"), None);
}

/// An error never suggests one of quack's internal tables, on the
/// writer or a reader clone: a misspelled name gets the user's own near
/// matches or none, and every other error reads as `DuckDB` wrote it.
#[test]
fn errors_never_suggest_internal_tables() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let config = config_in(dir.path());
    let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
    db.execute_query("CREATE TABLE orders (id INTEGER, customer VARCHAR, customers VARCHAR)")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let reader = db
        .try_clone_reader()
        .unwrap_or_else(|e| fail(&e.to_string()));
    for conn in [&db, &reader] {
        let error = |sql: &str| {
            conn.execute_query(sql)
                .err()
                .map_or_else(|| fail(&format!("{sql} ran")), |e| e.to_string())
        };
        // The only near match is internal: no suggestion at all.
        assert_eq!(
            error("SELECT * FROM _quack_meto"),
            "Catalog Error: Table with name _quack_meto does not exist!"
        );
        let missing = error("SELECT * FROM no_such_table");
        assert!(
            missing.starts_with("Catalog Error: Table with name no_such_table does not exist!")
                && !missing.contains("_quack_"),
            "{missing}"
        );
        // The user's own near matches still come through.
        assert_eq!(
            error("SELECT * FROM orderz"),
            "Catalog Error: Table with name orderz does not exist!\nDid you mean \"orders\"?"
        );
        assert_eq!(
            error("SELECT customr FROM orders"),
            "Binder Error: Referenced column \"customr\" not found in FROM clause!\n\
             Candidate bindings: \"customer\", \"customers\""
        );
        // Everything else reads as it always did.
        assert_eq!(
            error("SELEC 1"),
            "Parser Error: syntax error at or near \"SELEC\""
        );
        assert_eq!(
            error("INSERT INTO orders VALUES ('x', 'a', 'b')"),
            "Conversion Error: Could not convert string 'x' to INT32"
        );
    }
}

/// Design doc 7.4: a read-classified statement may still name a file, so
/// the connection itself is confined to the workspace directory and then
/// locked, for agent SQL and user SQL alike.
#[test]
fn workspace_connection_is_confined_to_its_directory_and_locked() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let config = config_in(dir.path());
    let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));

    let outside = dir.path().join("outside.csv");
    std::fs::write(&outside, "a\n1\n").unwrap_or_else(|e| fail(&e.to_string()));
    let inside = config.workspace_files_dir("ws").join("inside.csv");
    std::fs::write(&inside, "a\n1\n2\n").unwrap_or_else(|e| fail(&e.to_string()));

    for sql in [
        format!("SELECT * FROM read_csv_auto('{}')", outside.display()),
        format!("SELECT * FROM read_text('{}')", outside.display()),
        format!("SELECT * FROM '{}'", outside.display()),
        format!(
            "ATTACH '{}' AS other",
            dir.path().join("other.duckdb").display()
        ),
        String::from("INSTALL httpfs"),
        String::from("SET memory_limit = '8GB'"),
        String::from("SET enable_external_access = true"),
        String::from("SET allowed_directories = ['/']"),
    ] {
        let err = db.execute_query(&sql).err();
        assert!(err.is_some(), "ran outside the sandbox: {sql}");
        let text = err.map(|e| e.to_string()).unwrap_or_default();
        // Replacement scans are simply gone, so `FROM 'file'` is a
        // catalog miss; everything else is a permission or lock error.
        assert!(
            text.contains("Permission Error")
                || text.contains("locked")
                || text.contains("Catalog Error"),
            "{sql}: {text}"
        );
    }

    // Ingestion's own reads under files/ still work, through the same reader.
    let rows = db
        .execute_query(&format!(
            "SELECT count(*) FROM read_csv_auto('{}')",
            inside.display()
        ))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(rows.rows.len(), 1);
    assert_eq!(
        rows.rows.first().and_then(|r| r.first()),
        Some(&serde_json::Value::Number(2.into()))
    );
    // Ordinary statements are untouched.
    assert!(db.execute_statement("CREATE TABLE t(a INT)").is_ok());
    assert!(db.execute_query("SELECT * FROM t").is_ok());
}

/// Proof-of-concept for the reader connection: `try_clone`
/// succeeds once `lock_configuration = true` (set by `confine_to` on
/// open), because `DuckDB` locks the connection's *configuration*, not
/// its ability to open more connections to the same database; and a
/// write inside `BEGIN TRANSACTION READ ONLY` is rejected by `DuckDB`
/// itself, before it ever reaches the workspace's write-gating.
#[test]
fn reader_connection_clones_after_lock_configuration_and_cannot_write() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    let reader = db.conn.try_clone().unwrap_or_else(|e| fail(&e.to_string()));

    reader
        .execute_batch("BEGIN TRANSACTION READ ONLY")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let err = reader
        .execute_batch("CREATE TABLE t(a INT)")
        .err()
        .unwrap_or_else(|| fail("write inside a read-only transaction should have failed"));
    // The connection and its in-memory database are dropped at the end
    // of the test; no need to end the transaction explicitly.
    assert!(
        err.to_string().to_lowercase().contains("read"),
        "unexpected error: {err}"
    );
}

/// The actual guarded path (`try_clone_reader` plus `read_only`), not
/// just the raw statements the proof above assumes: a write attempted
/// inside `read_only` on a reader clone is rejected, and the connection
/// is still usable for a genuine read right after.
#[test]
fn read_only_on_a_reader_clone_rejects_a_write() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    let reader = db
        .try_clone_reader()
        .unwrap_or_else(|e| fail(&e.to_string()));

    let err = reader
        .read_only(|db| db.execute_statement("CREATE TABLE t(a INT)"))
        .err()
        .unwrap_or_else(|| fail("a write inside read_only on a reader should have failed"));
    assert!(
        err.to_string().to_lowercase().contains("read"),
        "unexpected error: {err}"
    );
    assert!(reader.read_only(|db| db.execute_query("SELECT 1")).is_ok());
}

/// A write on the writer, made before a reader is cloned from it, is
/// still visible to the reader afterward, and so is a write made even
/// later: `DuckDB` snapshots a transaction at `BEGIN`, not at
/// `try_clone`, and the writer's statements autocommit.
#[test]
fn reader_clone_sees_the_writers_prior_and_later_writes() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    db.execute_statement("CREATE TABLE t(a INT)")
        .unwrap_or_else(|e| fail(&e.to_string()));
    db.execute_statement("INSERT INTO t VALUES (1)")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let reader = db
        .try_clone_reader()
        .unwrap_or_else(|e| fail(&e.to_string()));

    let count = |reader: &WorkspaceDb| -> i64 {
        reader
            .read_only(|db| db.execute_query("SELECT count(*) FROM t"))
            .unwrap_or_else(|e| fail(&e.to_string()))
            .rows
            .first()
            .and_then(|row| row.first())
            .and_then(serde_json::Value::as_i64)
            .unwrap_or_else(|| fail("no count returned"))
    };
    assert_eq!(count(&reader), 1);

    db.execute_statement("INSERT INTO t VALUES (2)")
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(count(&reader), 2);
}

/// A closure that errors inside `read_only` rolls the transaction
/// back, and the connection is immediately reusable for another
/// `read_only` call: `ReadOnlyGuard::drop` did not leave one open.
#[test]
fn read_only_rolls_back_on_error_and_the_connection_stays_usable() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        db.read_only(|db| db.execute_query("SELECT * FROM no_such_table"))
            .is_err()
    );
    assert!(db.read_only(|db| db.execute_query("SELECT 1")).is_ok());
}

/// The reader clone inherits confinement: it cannot read outside the
/// workspace directory either, exactly like the writer (mirrors
/// `workspace_connection_is_confined_to_its_directory_and_locked`).
#[test]
fn reader_clone_is_confined_like_the_writer() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let config = config_in(dir.path());
    let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
    let reader = db
        .try_clone_reader()
        .unwrap_or_else(|e| fail(&e.to_string()));

    let outside = dir.path().join("outside.csv");
    std::fs::write(&outside, "a\n1\n").unwrap_or_else(|e| fail(&e.to_string()));
    let err = reader
        .execute_query(&format!(
            "SELECT * FROM read_csv_auto('{}')",
            outside.display()
        ))
        .err();
    assert!(err.is_some(), "the reader escaped the sandbox");
    let text = err.map(|e| e.to_string()).unwrap_or_default();
    // `read_csv_auto` on a real, existing file outside the workspace
    // has only one legitimate way to fail: the confinement check.
    // Unlike the writer's confinement test, nothing here can produce a
    // Catalog Error, so that arm would only ever hide an unrelated
    // regression (`read_csv_auto` itself going missing, say).
    assert!(text.contains("Permission Error"), "{text}");

    let inside = config.workspace_files_dir("ws").join("inside.csv");
    std::fs::write(&inside, "a\n1\n2\n").unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        reader
            .execute_query(&format!(
                "SELECT * FROM read_csv_auto('{}')",
                inside.display()
            ))
            .is_ok()
    );
}

/// `DuckDB` temp tables (the CLI's `stdin` table, `ingestion::STDIN_TABLE`)
/// are connection-local: a reader clone opened after the writer created
/// one does not see it. Tools reading through the reader must never be
/// pointed at it; only the writer connection (`run_sql`) can.
#[test]
fn reader_connection_cannot_see_the_writers_temp_tables() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    db.execute_statement("CREATE TEMP TABLE stdin AS SELECT 1 AS a")
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(db.execute_query("SELECT * FROM stdin").is_ok());

    let reader = db
        .try_clone_reader()
        .unwrap_or_else(|e| fail(&e.to_string()));
    let err = reader
        .execute_query("SELECT * FROM stdin")
        .err()
        .unwrap_or_else(|| fail("the reader should not see the writer's temp table"));
    assert!(err.to_string().contains("stdin"), "{err}");
}

#[test]
fn has_temp_tables_reports_a_piped_stdin_table() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        !db.has_temp_tables()
            .unwrap_or_else(|e| fail(&e.to_string()))
    );
    db.execute_statement("CREATE TEMP TABLE stdin AS SELECT 1 AS a")
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        db.has_temp_tables()
            .unwrap_or_else(|e| fail(&e.to_string()))
    );
}

#[test]
fn in_memory_connection_reads_no_files_at_all() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let file = dir.path().join("x.csv");
    std::fs::write(&file, "a\n1\n").unwrap_or_else(|e| fail(&e.to_string()));
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    let err = db
        .execute_query(&format!(
            "SELECT * FROM read_csv_auto('{}')",
            file.display()
        ))
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("Permission Error"), "{err}");
    assert!(db.execute_statement("SET threads = 1").is_err());
}

#[test]
fn query_values_keep_fractions_dates_and_nested_types() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    let results = db
        .execute_query(
            "SELECT 20.5 AS dec, 20.5::DOUBLE AS dbl, 3.25::FLOAT AS flt, \
             DATE '2024-01-02' AS d, TIMESTAMP '2024-01-02 03:04:05' AS ts, \
             TIMESTAMP '2024-01-02 03:04:05.25' AS tsf, TIME '03:04:05' AS t, \
             12345678901234567890::HUGEINT AS big, [1, 2] AS arr, {'a': 1, 'b': 'x'} AS st, \
             MAP {'k': 1} AS m, NULL AS n, 'text' AS s, true AS b, 7::UTINYINT AS u",
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
    let row = results.rows.first().unwrap_or_else(|| fail("no row"));
    let expected = serde_json::json!([
        20.5,
        20.5,
        3.25,
        "2024-01-02",
        "2024-01-02 03:04:05",
        "2024-01-02 03:04:05.250000",
        "03:04:05",
        "12345678901234567890",
        [1, 2],
        {"a": 1, "b": "x"},
        {"k": 1},
        null,
        "text",
        true,
        7
    ]);
    assert_eq!(serde_json::Value::Array(row.clone()), expected);
}

/// `DECIMAL` cells stay JSON numbers when their normalized digits survive
/// an `f64` (so `12.50` and `100.00` in a money column are numbers like
/// every other row), and keep their exact digit string only when they
/// would lose digits (more than about 16 significant digits). Guards the
/// `json_of` docstring contract.
#[test]
fn decimal_cells_stay_numbers_unless_they_lose_digits() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    db.execute_statement(
        "CREATE TABLE d (hi DECIMAL(20,2), wide DECIMAL(38,10), price DECIMAL(10,2), \
         round DECIMAL(10,2), big DECIMAL(18,2), int DECIMAL(5,0))",
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    db.execute_statement(
        "INSERT INTO d VALUES (123456789012345678.99, 99999999999999999999.1234567890, \
         12.50, 100.00, 1234567890123.45, 42)",
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    let results = db
        .execute_query("SELECT hi, wide, price, round, big, int FROM d")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let row = results.rows.first().unwrap_or_else(|| fail("no row"));

    // Only the cells with more significant digits than an f64 holds keep
    // their text; the declared scale's trailing zeros do not make a cell
    // a string.
    assert_eq!(
        serde_json::Value::Array(row.clone()),
        serde_json::json!([
            "123456789012345678.99",
            "99999999999999999999.1234567890",
            12.5,
            100,
            1_234_567_890_123.45,
            42
        ])
    );
}

/// One money column is all numbers: trailing zeros from the declared
/// scale never turn some of its cells into strings.
#[test]
#[expect(clippy::unwrap_used, reason = "test asserts Ok")]
fn a_money_column_does_not_mix_numbers_and_strings() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    db.execute_statement("CREATE TABLE m (amount DECIMAL(12,2))")
        .unwrap();
    db.execute_statement("INSERT INTO m VALUES (12.50), (100.00), (3.99), (0.10), (-7.00)")
        .unwrap();
    let results = db
        .execute_query("SELECT amount FROM m ORDER BY rowid")
        .unwrap();
    let cells: Vec<serde_json::Value> = results
        .rows
        .iter()
        .filter_map(|row| row.first().cloned())
        .collect();
    assert_eq!(
        serde_json::Value::Array(cells),
        serde_json::json!([12.5, 100, 3.99, 0.1, -7])
    );
}

/// The exact `DECIMAL` digits of a cell that would lose them survive
/// every output sink: the JSON serializers (REST `/sql`, MCP `sql`,
/// `quack -q` JSON/NdJSON) quote the strings, and the text paths (table,
/// CSV, markdown) keep the raw digits via `Cell::label`. A cell
/// that fits stays a number everywhere.
#[test]
#[expect(clippy::unwrap_used, reason = "test asserts Ok")]
fn decimal_digits_survive_every_output_sink() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    db.execute_statement("CREATE TABLE d (hi DECIMAL(20,2), scaled DECIMAL(3,2))")
        .unwrap();
    db.execute_statement("INSERT INTO d VALUES (123456789012345678.99, 1.50)")
        .unwrap();
    let results = db.execute_query("SELECT hi, scaled FROM d").unwrap();

    // NdJSON: the lossy cell is a quoted string, the other a number.
    let mut buf = Vec::new();
    results.write_ndjson(&mut buf).unwrap();
    assert_eq!(
        String::from_utf8(buf).unwrap(),
        "{\"hi\":\"123456789012345678.99\",\"scaled\":1.5}\n"
    );

    // JSON array: same cells.
    let mut buf = Vec::new();
    results.write_json(&mut buf).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&buf).unwrap();
    assert_eq!(
        parsed,
        serde_json::json!([{ "hi": "123456789012345678.99", "scaled": 1.5 }])
    );

    // CSV: the text path keeps the digits, unquoted.
    let mut buf = Vec::new();
    results.write_csv(&mut buf).unwrap();
    assert_eq!(
        String::from_utf8(buf).unwrap(),
        "hi,scaled\n123456789012345678.99,1.5\n"
    );

    // Markdown: the text path keeps the digits verbatim.
    let mut buf = Vec::new();
    results.write_markdown(&mut buf).unwrap();
    assert_eq!(
        String::from_utf8(buf).unwrap(),
        "| hi | scaled |\n| --- | --- |\n| 123456789012345678.99 | 1.5 |\n"
    );

    // Table: the aligned text path keeps the digits and never shows
    // the `f64` approximation.
    let mut buf = Vec::new();
    results.write_table(&mut buf).unwrap();
    let table = String::from_utf8(buf).unwrap();
    assert!(table.contains("123456789012345678.99"), "{table}");
    assert!(table.contains("1.5"), "{table}");
    assert!(!table.contains("1.2345678901234566e+17"), "{table}");
}

#[test]
fn describe_table_reports_the_exact_row_count() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    assert!(db.execute_statement("CREATE TABLE t(a INT)").is_ok());
    assert!(
        db.execute_statement("INSERT INTO t VALUES (1), (2), (3)")
            .is_ok()
    );
    let desc = db
        .describe_table("t")
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(desc.row_count, 3);
    assert_eq!(desc.sample_rows.rows.len(), 3);
    assert!(db.count_rows("missing").is_err());
    let version = db.duckdb_version().unwrap_or_else(|e| fail(&e.to_string()));
    assert!(version.starts_with('v'), "{version}");
}

fn sample() -> QueryResults {
    QueryResults {
        columns: vec![String::from("name"), String::from("n")],
        rows: vec![
            vec![
                serde_json::Value::String(String::from("a,b")),
                serde_json::Value::Number(1.into()),
            ],
            vec![
                serde_json::Value::String(String::from("say \"hi\"")),
                serde_json::Value::Null,
            ],
        ],
    }
}

/// The digest covers every row, kept or not, and the column names,
/// so a change past the cap, a renamed column, or a repeated row
/// changes it; the same rows in another order give the same digest.
#[test]
fn digested_query_covers_the_rows_past_the_cap_in_any_order() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    let digested = |sql: &str, cap: u32| {
        db.execute_query_digested(sql, cap)
            .unwrap_or_else(|e| fail(&e.to_string()))
    };
    let ten = digested("SELECT range AS n FROM range(10)", 3);
    assert_eq!(ten.results.results.rows.len(), 3);
    assert_eq!(ten.results.total_rows, 10);
    assert_eq!(ten.digest.len(), 64);
    assert_eq!(
        digested("SELECT range AS n FROM range(10)", 100).digest,
        ten.digest
    );
    assert_ne!(
        digested("SELECT range AS n FROM range(11)", 3).digest,
        ten.digest
    );
    assert_ne!(
        digested("SELECT range AS m FROM range(10)", 3).digest,
        ten.digest
    );
    assert_eq!(
        digested("SELECT range AS n FROM range(10) ORDER BY n DESC", 3).digest,
        ten.digest,
        "row order does not count"
    );
    assert_ne!(
        digested("SELECT 1 AS n UNION ALL SELECT 1", 3).digest,
        digested("SELECT 1 AS n", 3).digest,
        "a repeated row counts"
    );
    assert_ne!(
        digested("SELECT 1 AS n UNION ALL SELECT 1 UNION ALL SELECT 2", 3).digest,
        digested("SELECT 2 AS n", 3).digest,
        "a repeated row does not cancel out"
    );
    assert_eq!(
        digested("SELECT 1 AS n WHERE false", 3).digest,
        digested("SELECT 2 AS n WHERE false", 3).digest
    );
}

/// A `GROUP BY` without `ORDER BY` returns its groups in whatever
/// order the threads finish, so two runs over the same data must
/// digest the same.
#[test]
fn digested_group_by_is_the_same_across_runs() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    db.execute_statement(
        "CREATE TABLE events (bucket INTEGER, n INTEGER); \
         INSERT INTO events SELECT range % 64, range FROM range(100000)",
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    let sql = "SELECT bucket, count(*) AS c, sum(n) AS s FROM events GROUP BY bucket";
    let first = db
        .execute_query_digested(sql, 10)
        .unwrap_or_else(|e| fail(&e.to_string()));
    for _ in 0..5 {
        let again = db
            .execute_query_digested(sql, 10)
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(again.digest, first.digest);
        assert_eq!(again.results.total_rows, 64);
    }
    let ordered = db
        .execute_query_digested(&format!("{sql} ORDER BY bucket DESC"), 10)
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(ordered.digest, first.digest);
}

#[test]
fn capped_query_keeps_the_cap_and_counts_the_rest() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    let capped = db
        .execute_query_capped("SELECT range AS n FROM range(10)", 3)
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(capped.results.columns, vec![String::from("n")]);
    assert_eq!(capped.results.rows.len(), 3);
    assert_eq!(capped.total_rows, 10);
    assert!(capped.truncated());
    assert_eq!(capped.omitted(), 7);

    let exact = db
        .execute_query_capped("SELECT range AS n FROM range(3)", 3)
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(exact.results.rows.len(), 3);
    assert!(!exact.truncated());
    assert_eq!(exact.omitted(), 0);

    let none = db
        .execute_query_capped("SELECT 1 WHERE false", 3)
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(none.total_rows, 0);
    assert!(!none.truncated());

    let all = db
        .execute_query("SELECT range AS n FROM range(10)")
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(all.rows.len(), 10);
}

/// Columns that share a name keep every value under suffixed keys in
/// both JSON shapes; CSV, table, and markdown already kept them
/// (issue #65).
#[test]
#[expect(clippy::unwrap_used, reason = "test asserts Ok")]
fn json_writers_keep_columns_that_share_a_name() {
    let results = QueryResults {
        columns: vec![
            String::from("a"),
            String::from("a"),
            String::from("a_1"),
            String::from("a"),
        ],
        rows: vec![vec![
            serde_json::Value::from(1),
            serde_json::Value::from(2),
            serde_json::Value::from(3),
            serde_json::Value::from(4),
        ]],
    };
    assert_eq!(results.json_keys(), ["a", "a_1", "a_1_1", "a_2"]);
    let mut buf = Vec::new();
    results.write_ndjson(&mut buf).unwrap();
    assert_eq!(
        String::from_utf8(buf).unwrap(),
        "{\"a\":1,\"a_1\":2,\"a_1_1\":3,\"a_2\":4}\n"
    );
    let mut buf = Vec::new();
    results.write_json(&mut buf).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&buf).unwrap();
    assert_eq!(
        parsed,
        serde_json::json!([{ "a": 1, "a_1": 2, "a_1_1": 3, "a_2": 4 }])
    );
    assert_eq!(sample().json_keys(), ["name", "n"]);
}

#[test]
#[expect(clippy::unwrap_used, reason = "test asserts Ok")]
fn write_ndjson_one_object_per_line() {
    let mut buf = Vec::new();
    sample().write_ndjson(&mut buf).unwrap();
    let text = String::from_utf8(buf).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines.first().copied(), Some(r#"{"name":"a,b","n":1}"#));
    assert_eq!(
        lines.last().copied(),
        Some(r#"{"name":"say \"hi\"","n":null}"#)
    );
}

#[test]
#[expect(clippy::unwrap_used, reason = "test asserts Ok")]
fn write_csv_quotes_and_escapes() {
    let mut buf = Vec::new();
    sample().write_csv(&mut buf).unwrap();
    let text = String::from_utf8(buf).unwrap();
    assert_eq!(text, "name,n\n\"a,b\",1\n\"say \"\"hi\"\"\",\n");
}

#[test]
#[expect(clippy::unwrap_used, reason = "test asserts Ok")]
fn write_markdown_renders_table() {
    let mut buf = Vec::new();
    sample().write_markdown(&mut buf).unwrap();
    let text = String::from_utf8(buf).unwrap();
    assert!(text.starts_with("| name | n |\n| --- | --- |\n| a,b | 1 |\n"));
    assert!(text.contains("| say \"hi\" | NULL |"));
}

#[test]
fn direct_sql_detection_by_leading_keyword() {
    for yes in [
        "SELECT 1",
        "  with x as (select 1) select * from x",
        "FROM t",
        "describe t",
        "SHOW TABLES",
        "summarize t",
        "PIVOT t ON a",
        "explain select 1",
        "select(1)",
    ] {
        assert!(looks_like_direct_sql(yes), "{yes}");
    }
    for no in [
        "what were sales by region",
        "",
        "   ",
        "/sql select 1",
        "selected items please",
        "DROP TABLE t",
        "insert into t values (1)",
    ] {
        assert!(!looks_like_direct_sql(no), "{no}");
    }
}

#[test]
fn tokenize_lowercases_and_splits_on_punctuation() {
    assert_eq!(
        Analyzer::default().terms("Policy POL-8841 renews; see \"Exclusions\" (page 12)."),
        vec![
            "polici", "pol", "8841", "pol8841", "renew", "see", "exclus", "page", "12"
        ]
    );
    // Inflections meet at one stem; codes and numbers are untouched.
    assert_eq!(
        Analyzer::default().terms("renewal renewals renewing"),
        vec!["renew", "renew", "renew"]
    );
    assert_eq!(
        Analyzer::default().terms("AB-12X9"),
        vec!["ab", "12x9", "ab12x9"]
    );
    assert!(Analyzer::default().terms("  --- ").is_empty());
}

#[test]
fn tokenize_indexes_the_joined_form_of_an_identifier() {
    // Hyphen, dot, underscore, slash, and colon all join.
    assert_eq!(
        Analyzer::default().terms("v1.2.3"),
        vec!["v1", "2", "3", "v123"]
    );
    assert_eq!(
        Analyzer::default().terms("ABC_123"),
        vec!["abc", "123", "abc123"]
    );
    assert_eq!(
        Analyzer::default().terms("ns/part:7"),
        vec!["ns", "part", "7", "nspart7"]
    );
    // A joiner touching whitespace does not merge across words: prose
    // punctuation still tokenizes exactly as before.
    assert_eq!(
        Analyzer::default().terms("end of sentence - new sentence."),
        vec!["end", "of", "sentenc", "new", "sentenc"]
    );
    // The query side uses the same function, so a bare joined form
    // already in text (`pol8841`) is found by the query `POL-8841`.
    assert!(
        Analyzer::default()
            .terms("POL-8841")
            .contains(&String::from("pol8841"))
    );
    assert_eq!(Analyzer::default().terms("pol8841"), vec!["pol8841"]);
    // The joined form uses full Unicode case folding, not ASCII-only
    // lowercasing, so a non-ASCII identifier's casing does not change
    // which term it indexes: `Ünit-9` in text and `ünit-9` in a query
    // must both produce the joined term `ünit9`.
    assert_eq!(
        Analyzer::default().terms("Ünit-9").last(),
        Analyzer::default().terms("ünit-9").last(),
    );
    assert_eq!(
        Analyzer::default().terms("Ünit-9").last(),
        Some(&String::from("ünit9"))
    );
}

#[test]
fn term_frequencies_count_heading_too() {
    let tf = TermFrequencies::of(
        &Analyzer::default(),
        "flood flood damage",
        Some("Flood Exclusions"),
    );
    assert_eq!(
        tf.0,
        vec![
            (String::from("damag"), 1),
            (String::from("exclus"), 1),
            (String::from("flood"), 3),
        ]
    );
    assert_eq!(tf.total(), 5);
}

#[test]
fn phrases_read_balanced_quotes() {
    assert_eq!(
        Phrases::parse("\"flood exclusion\"").0,
        vec![String::from("flood exclusion")]
    );
    assert_eq!(
        Phrases::parse("find \"flood exclusion\" near \"water damage\"").0,
        vec![
            String::from("flood exclusion"),
            String::from("water damage")
        ]
    );
    assert!(Phrases::parse("no quotes here").0.is_empty());
    assert!(Phrases::parse("\"\"").0.is_empty());
    // Unbalanced quotes: an odd count is ordinary text, not a phrase.
    assert!(Phrases::parse("say \"hello").0.is_empty());
    assert!(Phrases::parse("a \"b\" c\" d").0.is_empty());
}

#[test]
fn phrases_match_ignoring_case_and_whitespace() {
    assert!(Phrases::contains(
        "the FLOOD   Exclusion\napplies here",
        "flood exclusion"
    ));
    assert!(!Phrases::contains("flood and exclusion", "flood exclusion"));
    assert!(Phrases::contains(
        "Flood Exclusion",
        "  flood   exclusion  "
    ));
}

#[test]
fn phrases_over_fetch_only_when_there_is_one() {
    assert_eq!(Phrases::parse("plain words").fetch(10, 20), 10);
    assert_eq!(Phrases::parse("\"a b\"").fetch(10, 20), 80);
    assert_eq!(
        Phrases::parse("\"a b\"").fetch(10, 1000),
        PHRASE_CANDIDATE_CAP
    );
    assert_eq!(Phrases::parse("\"a b\"").fetch(900, 20), 900);
}

#[test]
fn quote_ident_wraps_and_escapes() {
    assert_eq!(quote_ident("sales"), "\"sales\"");
    assert_eq!(quote_ident("odd name"), "\"odd name\"");
    assert_eq!(quote_ident("x\"y"), "\"x\"\"y\"");
}

#[test]
fn display_json_null() {
    assert_eq!(Cell(&serde_json::Value::Null).label(), "NULL");
}

#[test]
fn display_json_string() {
    let val = serde_json::Value::String("hello".into());
    assert_eq!(Cell(&val).label(), "hello");
}

#[test]
fn display_json_number() {
    let val = serde_json::Value::Number(42.into());
    assert_eq!(Cell(&val).label(), "42");
}

#[test]
fn display_json_bool_true() {
    assert_eq!(Cell(&serde_json::Value::Bool(true)).label(), "true");
}

#[test]
fn display_json_bool_false() {
    assert_eq!(Cell(&serde_json::Value::Bool(false)).label(), "false");
}

#[test]
#[expect(clippy::unwrap_used, reason = "test asserts output format")]
fn write_table_empty_columns_prints_ok() {
    let results = QueryResults {
        columns: Vec::new(),
        rows: Vec::new(),
    };
    let mut buf = Vec::new();
    results.write_table(&mut buf).unwrap();
    assert_eq!(String::from_utf8_lossy(&buf), "OK\n");
}

#[test]
#[expect(clippy::unwrap_used, reason = "test asserts output format")]
fn write_table_renders_aligned_columns() {
    let results = QueryResults {
        columns: vec!["id".into(), "name".into()],
        rows: vec![
            vec![
                serde_json::Value::Number(1.into()),
                serde_json::Value::String("alice".into()),
            ],
            vec![
                serde_json::Value::Number(2.into()),
                serde_json::Value::String("bob".into()),
            ],
        ],
    };
    let mut buf = Vec::new();
    results.write_table(&mut buf).unwrap();
    let output = String::from_utf8_lossy(&buf);
    assert!(output.contains("id"));
    assert!(output.contains("name"));
    assert!(output.contains("alice"));
    assert!(output.contains("bob"));
    assert!(output.contains("(2 rows)"));
    assert!(output.contains("-+-"));
}

#[test]
#[expect(clippy::unwrap_used, reason = "test asserts output format")]
fn write_json_produces_valid_array() {
    let results = QueryResults {
        columns: vec!["id".into(), "val".into()],
        rows: vec![vec![
            serde_json::Value::Number(1.into()),
            serde_json::Value::String("x".into()),
        ]],
    };
    let mut buf = Vec::new();
    results.write_json(&mut buf).unwrap();
    let output = String::from_utf8_lossy(&buf);
    let parsed: Vec<serde_json::Map<String, serde_json::Value>> =
        serde_json::from_str(&output).unwrap();
    assert_eq!(parsed.len(), 1);
    let first = parsed.first().unwrap();
    assert_eq!(first.get("id"), Some(&serde_json::Value::Number(1.into())));
    assert_eq!(
        first.get("val"),
        Some(&serde_json::Value::String("x".into()))
    );
}

#[test]
#[expect(clippy::unwrap_used, reason = "test asserts output format")]
fn write_json_empty_rows_produces_empty_array() {
    let results = QueryResults {
        columns: vec!["a".into()],
        rows: Vec::new(),
    };
    let mut buf = Vec::new();
    results.write_json(&mut buf).unwrap();
    let output = String::from_utf8_lossy(&buf);
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
    assert!(parsed.is_empty());
}

/// Up to `limit` items spread evenly over the groups: each group (one
/// document's chunks, in order) gets an equal quota, taken at evenly
/// spaced positions, so a sample covers every document and not just the
/// front matter of the first.
fn evenly_spaced<T>(groups: impl IntoIterator<Item = Vec<T>>, limit: usize) -> Vec<T> {
    let groups: Vec<Vec<T>> = groups.into_iter().filter(|g| !g.is_empty()).collect();
    if limit == 0 || groups.is_empty() {
        return Vec::new();
    }
    let quota = limit.div_ceil(groups.len()).max(1);
    let mut chosen = Vec::with_capacity(limit);
    for group in groups {
        let len = group.len();
        let take = quota.min(len);
        let mut positions: Vec<usize> = (0..take)
            .map(|k| {
                k.saturating_mul(len)
                    .checked_div(take)
                    .unwrap_or(0)
                    .min(len.saturating_sub(1))
            })
            .collect();
        positions.dedup();
        let mut positions = positions.into_iter().peekable();
        for (index, item) in group.into_iter().enumerate() {
            if positions.peek() == Some(&index) {
                positions.next();
                chosen.push(item);
            }
        }
    }
    chosen.truncate(limit);
    chosen
}

#[test]
fn a_sample_spreads_across_every_group() {
    let groups = vec![(0..10).collect::<Vec<u32>>(), vec![100, 101], vec![]];
    assert_eq!(evenly_spaced(groups.clone(), 4), vec![0, 5, 100, 101]);
    assert_eq!(evenly_spaced(groups.clone(), 3), vec![0, 5, 100]);
    assert!(evenly_spaced(groups, 0).is_empty());
    assert_eq!(evenly_spaced(vec![vec![1, 2, 3]], 10), vec![1, 2, 3]);
}

/// The SQL sampler picks exactly what `evenly_spaced` picks from the
/// same documents in the same order, for every limit, from both pools.
#[test]
fn the_sql_sample_matches_evenly_spaced() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    // Ids sort in insertion order, so ingest order and id order agree.
    let sizes = [10_u32, 2, 7, 1, 13, 3];
    let mut groups: Vec<Vec<String>> = Vec::new();
    for (d, size) in sizes.iter().enumerate() {
        let doc = format!("d{d}");
        insert_ready_document(&db, &doc);
        let mut group = Vec::new();
        for i in 0..*size {
            let id = format!("{doc}-c{i:02}");
            insert_text_chunk(
                &db,
                &id,
                &doc,
                i,
                "a passage long enough to count as more than a line of text",
            );
            group.push(id);
        }
        groups.push(group);
    }
    for pool in [SamplePool::NotGraphExtracted, SamplePool::Substantive] {
        assert_eq!(
            db.pool_size(pool).unwrap_or_else(|e| fail(&e.to_string())),
            36
        );
        for limit in 0..40_u32 {
            let expected = evenly_spaced(groups.clone(), limit as usize);
            let sampled = db
                .sample_chunk_ids(pool, limit)
                .unwrap_or_else(|e| fail(&e.to_string()));
            let sampled: Vec<String> = sampled.into_iter().map(ChunkId::into_string).collect();
            assert_eq!(sampled, expected, "{pool:?} limit {limit}");
        }
    }
}

/// A rebuild read a page at a time indexes every chunk, the last page
/// short, exactly as the inserts did.
#[test]
fn a_paged_reindex_restores_every_chunk() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    insert_ready_document(&db, "d0");
    for i in 0..7 {
        insert_text_chunk(&db, &format!("c{i}"), "d0", i, &format!("flood report {i}"));
    }
    let snapshot = |db: &WorkspaceDb| {
        db.execute_query(
            "SELECT (SELECT count(*) FROM _quack_terms), (SELECT sum(token_count) FROM _quack_chunks)",
        )
        .unwrap_or_else(|e| fail(&e.to_string()))
        .rows
    };
    let indexed = snapshot(&db);
    db.execute_statement("DELETE FROM _quack_terms")
        .unwrap_or_else(|e| fail(&e.to_string()));
    db.reindex_terms_by(3)
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(snapshot(&db), indexed);
}

/// Paging by id visits every chunk of the pool once, in id order.
#[test]
fn csv_quotes_only_what_needs_it_and_keeps_a_lone_empty_field() {
    let written = |columns: &[&str], rows: Vec<Vec<serde_json::Value>>| {
        let results = QueryResults {
            columns: columns.iter().map(|c| (*c).to_owned()).collect(),
            rows,
        };
        let mut out = Vec::new();
        results.write_csv(&mut out).map_or_else(
            |e| e.to_string(),
            |()| String::from_utf8_lossy(&out).into_owned(),
        )
    };
    assert_eq!(
        written(
            &["name", "note"],
            vec![
                vec![serde_json::json!("x,y"), serde_json::json!("say \"hi\"")],
                vec![serde_json::json!(""), serde_json::Value::Null],
            ]
        ),
        "name,note\n\"x,y\",\"say \"\"hi\"\"\"\n,\n"
    );
    // One empty field alone would be a blank line, which readers skip.
    assert_eq!(
        written(&["n"], vec![vec![serde_json::Value::Null]]),
        "n\n\"\"\n"
    );
}

#[test]
fn chunk_pages_visit_the_pool_once() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    insert_ready_document(&db, "d0");
    for i in 0..7 {
        insert_text_chunk(&db, &format!("c{i}"), "d0", i, "some text");
    }
    let mut seen = Vec::new();
    let mut after: Option<ChunkId> = None;
    loop {
        let page = db
            .chunk_page(SamplePool::NotGraphExtracted, after.as_ref(), 3)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let Some(last) = page.last() else { break };
        after = Some(last.id.clone());
        seen.extend(page.into_iter().map(|c| c.id.into_string()));
    }
    assert_eq!(seen, ["c0", "c1", "c2", "c3", "c4", "c5", "c6"]);
}

#[test]
fn page_counts_are_stored_on_the_document_and_cleared_with_none() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    insert_ready_document(&db, "doc1");
    let id = DocumentId::from("doc1");
    let pages = |db: &WorkspaceDb| {
        db.document(&id)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .map(|doc| (doc.pages, doc.pages.and_then(PageCounts::note)))
    };
    assert_eq!(pages(&db), Some((None, None)));

    let counts = PageCounts {
        total: 40,
        unreadable: 3,
        empty: 2,
    };
    db.set_document_pages(&id, Some(counts))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        pages(&db),
        Some((
            Some(counts),
            Some(String::from("3 of 40 pages unreadable, 2 without text"))
        ))
    );
    let listed = db.list_documents().unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(listed.first().and_then(|doc| doc.pages), Some(counts));

    db.set_document_pages(&id, None)
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(pages(&db), Some((None, None)));
}

fn insert_ready_document(db: &WorkspaceDb, id: &str) {
    db.insert_document(
        &NewDocument::new(&DocumentId::from(id), "doc.txt", "text/plain", 10)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
}

fn insert_text_chunk(
    db: &WorkspaceDb,
    id: &str,
    document_id: &str,
    chunk_index: u32,
    content: &str,
) {
    db.chunk_writer(&DocumentId::from(document_id), content)
        .and_then(|writer| {
            writer.insert(&NewChunk {
                id: &ChunkId::from(id),
                chunk_index,
                content,
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: None,
            })
        })
        .unwrap_or_else(|e| fail(&e.to_string()));
}

/// The joined identifier term must rank a chunk containing the exact
/// identifier above one that only contains its split pieces apart.
#[test]
fn search_keyword_chunks_ranks_the_exact_identifier_above_split_terms() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    insert_ready_document(&db, "doc1");
    insert_text_chunk(&db, "c1", "doc1", 0, "Policy POL-8841 covers water damage.");
    insert_text_chunk(
        &db,
        "c2",
        "doc1",
        1,
        "The pol number appears here, and the 8841 total appears elsewhere in this paragraph.",
    );

    let results = db
        .search_keyword_chunks("POL-8841", 10, &ChunkScope::all())
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        results.first().map(|r| r.id.as_str()),
        Some("c1"),
        "the exact identifier should outrank its split pieces: {results:?}"
    );
}

#[test]
fn search_keyword_chunks_filters_candidates_by_quoted_phrase() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    insert_ready_document(&db, "doc1");
    insert_text_chunk(
        &db,
        "c1",
        "doc1",
        0,
        "The flood exclusion applies to basements.",
    );
    // Same two words, not adjacent: matches the bag-of-words ranking but
    // not the phrase.
    insert_text_chunk(
        &db,
        "c2",
        "doc1",
        1,
        "Exclusion of flood risk is handled in a separate clause.",
    );

    let results = db
        .search_keyword_chunks("\"flood exclusion\"", 10, &ChunkScope::all())
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(results.len(), 1);
    assert_eq!(results.first().map(|r| r.id.as_str()), Some("c1"));
}

#[test]
fn search_keyword_chunks_phrase_match_in_heading() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    insert_ready_document(&db, "doc1");
    db.chunk_writer(
        &DocumentId::from("doc1"),
        "See below for what is not covered.",
    )
    .and_then(|writer| {
        writer.insert(&NewChunk {
            id: &ChunkId::from("c1"),
            chunk_index: 0,
            content: "See below for what is not covered.",
            heading: Some("Flood Exclusion"),
            page: None,
            kind: SectionKind::Body,
            locator: None,
            embedding: None,
        })
    })
    .unwrap_or_else(|e| fail(&e.to_string()));

    let results = db
        .search_keyword_chunks("\"flood exclusion\"", 10, &ChunkScope::all())
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(results.len(), 1);
    assert_eq!(results.first().map(|r| r.id.as_str()), Some("c1"));
}

#[test]
fn search_keyword_chunks_phrase_with_no_match_returns_empty() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    insert_ready_document(&db, "doc1");
    insert_text_chunk(
        &db,
        "c1",
        "doc1",
        0,
        "Exclusion of flood risk is handled in a separate clause.",
    );

    let results = db
        .search_keyword_chunks("\"flood exclusion\"", 10, &ChunkScope::all())
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        results.is_empty(),
        "a phrase with no exact match must not fall back to unfiltered candidates: {results:?}"
    );
}

#[test]
fn search_keyword_chunks_unbalanced_quote_is_ordinary_text() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    insert_ready_document(&db, "doc1");
    insert_text_chunk(
        &db,
        "c1",
        "doc1",
        0,
        "Exclusion of flood risk is handled in a separate clause.",
    );

    let results = db
        .search_keyword_chunks("\"flood exclusion", 10, &ChunkScope::all())
        .unwrap_or_else(|e| fail(&e.to_string()));
    // No phrase requirement kicks in: ordinary bag-of-words matching
    // still finds the chunk even though the words are not adjacent.
    assert_eq!(results.len(), 1);
}

#[test]
fn search_hybrid_chunks_ranks_identifier_and_filters_phrase() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    insert_ready_document(&db, "doc1");
    let embedding = Vector::from(vec![0.1_f32, 0.2, 0.3, 0.4]);
    db.chunk_writer(
        &DocumentId::from("doc1"),
        "Policy POL-8841 covers water damage.",
    )
    .and_then(|writer| {
        writer.insert(&NewChunk {
            id: &ChunkId::from("c1"),
            chunk_index: 0,
            content: "Policy POL-8841 covers water damage.",
            heading: None,
            page: None,
            kind: SectionKind::Body,
            locator: None,
            embedding: Some(&embedding),
        })
    })
    .unwrap_or_else(|e| fail(&e.to_string()));
    db.chunk_writer(&DocumentId::from("doc1"), "The pol number appears here, and the 8841 total appears elsewhere in this paragraph.").and_then(|writer| writer.insert(&NewChunk {
        id: &ChunkId::from("c2"),
        chunk_index: 1,
        content: "The pol number appears here, and the 8841 total appears elsewhere in this paragraph.",
        heading: None,
        page: None,
        kind: SectionKind::Body,
        locator: None,
        embedding: Some(&embedding),
    }))
    .unwrap_or_else(|e| fail(&e.to_string()));

    let results = db
        .search_hybrid_chunks(
            "POL-8841",
            &embedding,
            HybridLimits {
                top_k: 10,
                rrf_k: 60,
            },
            &ChunkScope::all(),
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(results.first().map(|r| r.id.as_str()), Some("c1"));
}

#[test]
fn search_hybrid_chunks_phrase_filters_to_matching_chunks() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    insert_ready_document(&db, "doc1");
    let embedding = Vector::from(vec![0.1_f32, 0.2, 0.3, 0.4]);
    db.chunk_writer(
        &DocumentId::from("doc1"),
        "The flood exclusion applies to basements.",
    )
    .and_then(|writer| {
        writer.insert(&NewChunk {
            id: &ChunkId::from("c1"),
            chunk_index: 0,
            content: "The flood exclusion applies to basements.",
            heading: None,
            page: None,
            kind: SectionKind::Body,
            locator: None,
            embedding: Some(&embedding),
        })
    })
    .unwrap_or_else(|e| fail(&e.to_string()));
    db.chunk_writer(
        &DocumentId::from("doc1"),
        "Exclusion of flood risk is handled in a separate clause.",
    )
    .and_then(|writer| {
        writer.insert(&NewChunk {
            id: &ChunkId::from("c2"),
            chunk_index: 1,
            content: "Exclusion of flood risk is handled in a separate clause.",
            heading: None,
            page: None,
            kind: SectionKind::Body,
            locator: None,
            embedding: Some(&embedding),
        })
    })
    .unwrap_or_else(|e| fail(&e.to_string()));

    let results = db
        .search_hybrid_chunks(
            "\"flood exclusion\"",
            &embedding,
            HybridLimits {
                top_k: 10,
                rrf_k: 60,
            },
            &ChunkScope::all(),
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(results.len(), 1);
    assert_eq!(results.first().map(|r| r.id.as_str()), Some("c1"));
}

/// Version 10 stores whether an ontology version was reviewed; a
/// workspace from before it said so only in the note, so opening it
/// marks those versions auto-accepted and leaves the others reviewed.
#[test]
fn opening_an_older_workspace_reads_auto_acceptance_from_the_note() {
    use crate::ontology::Ontology;
    use crate::ontology::store::{self, Revision};
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let config = config_in(dir.path());
    {
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        for note in ["seeded", "auto-accepted 3 candidate(s)"] {
            store::save(
                &db,
                &Ontology::builtin_default(),
                Revision::reviewed(None, Some(note)),
            )
            .unwrap_or_else(|e| fail(&e.to_string()));
        }
        // A file from before version 10 has no acceptance column.
        db.execute_statement("ALTER TABLE _quack_ontology_versions DROP COLUMN acceptance")
            .unwrap_or_else(|e| fail(&e.to_string()));
        db.set_meta(MetaKey::SchemaVersion, "9")
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    // The second open runs while the first is still live, so it replays
    // the first one's column change from the write-ahead log, as the next
    // start does after a process dies before a checkpoint. Windows lets no
    // second handle open the file (#448), so there it follows the close.
    let open = || WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
    let upgraded = open();
    let reopened = if cfg!(windows) {
        drop(upgraded);
        open()
    } else {
        let reopened = open();
        drop(upgraded);
        reopened
    };
    let acceptance: Vec<Acceptance> = store::versions(&reopened, 10)
        .unwrap_or_else(|e| fail(&e.to_string()))
        .into_iter()
        .map(|v| v.acceptance)
        .collect();
    assert_eq!(acceptance, [Acceptance::Auto, Acceptance::Reviewed]);
}

/// The `storage_version` tag `DuckDB` reports for the file `conn` has open.
fn storage_version(conn: &duckdb::Connection, database: &str) -> String {
    conn.query_row(
        "SELECT tags['storage_version']::VARCHAR FROM duckdb_databases() \
         WHERE database_name = ?",
        duckdb::params![database],
        |row| row.get(0),
    )
    .unwrap_or_else(|e| fail(&e.to_string()))
}

/// A file recorded under a newer schema version is refused before any
/// statement changes it: the versions it recorded stay, and a table
/// this binary would create is not created.
#[test]
fn a_file_from_a_newer_quack_is_refused_and_left_as_it_was() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let config = config_in(dir.path());
    let newer = WORKSPACE_SCHEMA_VERSION.saturating_add(1);
    {
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        db.execute_statement("DROP TABLE _quack_session_summaries")
            .unwrap_or_else(|e| fail(&e.to_string()));
        db.set_meta(MetaKey::SchemaVersion, &newer.to_string())
            .unwrap_or_else(|e| fail(&e.to_string()));
        db.set_meta(MetaKey::WrittenByQuack, "9.9.9")
            .unwrap_or_else(|e| fail(&e.to_string()));
    }

    let Err(refused) = WorkspaceDb::open(&config, "ws") else {
        fail("a newer file opened")
    };
    let Error::WorkspaceTooNew {
        recorded,
        supported,
        written_by,
        ..
    } = &refused
    else {
        fail(&refused.to_string())
    };
    assert_eq!((*recorded, *supported), (newer, WORKSPACE_SCHEMA_VERSION));
    assert_eq!(written_by, &WrittenBy(Some(String::from("9.9.9"))));
    let message = refused.to_string();
    assert!(
        message.contains("written by quack 9.9.9; this quack"),
        "{message}"
    );
    assert!(message.contains("run quack 9.9.9 or newer"), "{message}");

    let conn = duckdb::Connection::open(config.workspace_db_path("ws"))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let recorded = |key| WorkspaceDb::recorded(&conn, key).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(recorded(MetaKey::SchemaVersion), Some(newer.to_string()));
    assert_eq!(
        recorded(MetaKey::WrittenByQuack),
        Some(String::from("9.9.9"))
    );
    let recreated: bool = conn
        .query_row(
            "SELECT count(*) > 0 FROM duckdb_tables() \
             WHERE table_name = '_quack_session_summaries'",
            [],
            |row| row.get(0),
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(!recreated, "the refusal ran table definitions");
}

/// A file from before the writer was recorded names no version to run.
#[test]
fn a_newer_file_with_no_recorded_writer_asks_for_a_newer_quack() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let config = config_in(dir.path());
    {
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        db.set_meta(MetaKey::SchemaVersion, "99")
            .unwrap_or_else(|e| fail(&e.to_string()));
        db.delete_meta(MetaKey::WrittenByQuack)
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    let Err(refused) = WorkspaceDb::open(&config, "ws") else {
        fail("a newer file opened")
    };
    let message = refused.to_string();
    assert!(message.contains("written by a newer quack;"), "{message}");
    assert!(message.contains("run a newer quack"), "{message}");
}

/// A schema version that is not a number says nothing about the
/// file's schema, so the file is refused before any statement changes
/// it, not read as version 0.
#[test]
fn a_schema_version_that_is_not_a_number_is_refused_and_left_as_it_was() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let config = config_in(dir.path());
    {
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        db.execute_statement("DROP TABLE _quack_session_summaries")
            .unwrap_or_else(|e| fail(&e.to_string()));
        db.set_meta(MetaKey::SchemaVersion, "eleven")
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    let Err(refused) = WorkspaceDb::open(&config, "ws") else {
        fail("a file with an unreadable schema version opened")
    };
    assert!(
        matches!(
            &refused,
            Error::WorkspaceSchemaUnreadable { recorded, .. } if recorded == "eleven"
        ),
        "{refused}"
    );
    assert!(
        refused
            .to_string()
            .contains("records schema version 'eleven', which is not a number"),
        "{refused}"
    );
    let conn = duckdb::Connection::open(config.workspace_db_path("ws"))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let recreated: bool = conn
        .query_row(
            "SELECT count(*) > 0 FROM duckdb_tables() \
             WHERE table_name = '_quack_session_summaries'",
            [],
            |row| row.get(0),
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(!recreated, "the refusal ran table definitions");
}

/// Every open records the quack and `DuckDB` versions that wrote the file.
#[test]
fn opening_a_workspace_records_its_writer() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let config = config_in(dir.path());
    {
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        db.set_meta(MetaKey::WrittenByQuack, "0.0.1")
            .unwrap_or_else(|e| fail(&e.to_string()));
        db.set_meta(MetaKey::WrittenByDuckDb, "v0.0.1")
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
    let meta = |key| db.meta(key).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        meta(MetaKey::WrittenByQuack).as_deref(),
        Some(env!("CARGO_PKG_VERSION"))
    );
    assert_eq!(
        meta(MetaKey::WrittenByDuckDb),
        Some(db.duckdb_version().unwrap_or_else(|e| fail(&e.to_string())))
    );
}

/// The storage compatibility version quack names is the bundled
/// `DuckDB`'s own default, so naming it changes nothing: a new workspace
/// file gets the format a plain `DuckDB` open gives one, and a file
/// created in another format keeps it through an open that writes.
#[test]
fn naming_the_storage_compatibility_version_changes_no_file() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let config = config_in(dir.path());
    let plain_path = dir.path().join("plain.duckdb");
    let plain = duckdb::Connection::open(&plain_path).unwrap_or_else(|e| fail(&e.to_string()));
    let default: String = plain
        .query_row(
            "SELECT current_setting('storage_compatibility_version')",
            [],
            |row| row.get(0),
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        default, STORAGE_COMPATIBILITY_VERSION,
        "the bundled DuckDB's default changed: decide the format of new workspace files \
         and document the step in docs/migrations.md"
    );

    let new = WorkspaceDb::open(&config, "new").unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        storage_version(&new.conn, "data"),
        storage_version(&plain, "plain")
    );

    // A file in the newest format this DuckDB writes, where a workspace
    // file would be.
    let newest_path = config.workspace_db_path("newest");
    std::fs::create_dir_all(config.workspace_dir("newest"))
        .unwrap_or_else(|e| fail(&e.to_string()));
    plain
        .execute_batch(&format!(
            "ATTACH '{}' AS newest (STORAGE_VERSION 'latest'); \
             CREATE TABLE newest.t AS SELECT 1 AS a; CHECKPOINT newest;",
            newest_path.display()
        ))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let created_as = storage_version(&plain, "newest");
    assert_ne!(created_as, storage_version(&plain, "plain"));
    plain
        .execute_batch("DETACH newest")
        .unwrap_or_else(|e| fail(&e.to_string()));

    let opened = WorkspaceDb::open(&config, "newest").unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(storage_version(&opened.conn, "data"), created_as);
    drop(opened);
    let reopened = duckdb::Connection::open(&newest_path).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(storage_version(&reopened, "data"), created_as);
}

/// Version 7 added joined identifier terms, so a workspace still
/// recorded at an older version must reindex `_quack_terms` on open
/// (`WorkspaceDb::create_internal_tables`).
#[test]
fn opening_an_older_workspace_reindexes_terms_for_identifier_search() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let config = config_in(dir.path());
    {
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        insert_ready_document(&db, "doc1");
        insert_text_chunk(&db, "c1", "doc1", 0, "Policy POL-8841 applies.");
        // Simulate a workspace indexed before version 7: the joined
        // identifier term is missing and the recorded version rolls back.
        db.execute_statement("DELETE FROM _quack_terms WHERE term = 'pol8841'")
            .unwrap_or_else(|e| fail(&e.to_string()));
        db.set_meta(MetaKey::SchemaVersion, "6")
            .unwrap_or_else(|e| fail(&e.to_string()));
    }

    let reopened = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        reopened
            .meta(MetaKey::SchemaVersion)
            .unwrap_or_else(|e| fail(&e.to_string())),
        Some(WORKSPACE_SCHEMA_VERSION.to_string())
    );
    let rows = reopened
        .execute_query("SELECT count(*) FROM _quack_terms WHERE term = 'pol8841'")
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        rows.rows.first().and_then(|r| r.first()),
        Some(&serde_json::Value::Number(1.into())),
        "reopening should have rebuilt the term index with the joined form"
    );
}

/// A workspace that, before symmetric dedup (`<` `MERGE_DEDUP`), picked up
/// opposing-orientation rows for the same merge pair collapses them on
/// open to a single row, keeping the more-decided one so a reviewer's
/// rejection is not lost; a single-orientation pair is untouched.
#[test]
fn opening_an_older_workspace_collapses_opposing_merge_duplicates() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let config = config_in(dir.path());
    {
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        // The same pair, once rejected (keep=A, drop=B) and once a later
        // pending duplicate in the flipped orientation (keep=B, drop=A).
        db.execute_statement(
            "INSERT INTO _quack_graph_merges \
             (id, keep_node_id, drop_node_id, distance, status, decided_at) \
             VALUES ('m1', 'A', 'B', 0.0, 'rejected', \
                     TIMESTAMP '2026-01-01 00:00:00')",
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
        db.execute_statement(
            "INSERT INTO _quack_graph_merges \
             (id, keep_node_id, drop_node_id, distance, status) \
             VALUES ('m2', 'B', 'A', 0.0, 'pending')",
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
        // An unrelated single-orientation pair stays as it is.
        db.execute_statement(
            "INSERT INTO _quack_graph_merges \
             (id, keep_node_id, drop_node_id, distance, status, decided_at) \
             VALUES ('m3', 'C', 'D', 0.0, 'rejected', \
                     TIMESTAMP '2026-02-01 00:00:00')",
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
        // Pretend it was last recorded before the symmetric-dedup step.
        db.set_meta(MetaKey::SchemaVersion, "10")
            .unwrap_or_else(|e| fail(&e.to_string()));
    }

    let reopened = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        reopened
            .meta(MetaKey::SchemaVersion)
            .unwrap_or_else(|e| fail(&e.to_string())),
        Some(WORKSPACE_SCHEMA_VERSION.to_string())
    );
    // One row per pair: the rejected row survived, the pending duplicate
    // was removed, the unrelated pair is still there.
    let count = reopened
        .execute_query("SELECT count(*) FROM _quack_graph_merges")
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        count.rows.first().and_then(|r| r.first()),
        Some(&serde_json::json!(2)),
        "one row per pair"
    );
    let kept = reopened
        .execute_query("SELECT keep_node_id, status FROM _quack_graph_merges WHERE id = 'm1'")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let row = kept
        .rows
        .first()
        .unwrap_or_else(|| fail("the rejected row m1 should have survived"));
    assert_eq!(row.first(), Some(&serde_json::json!("A")));
    assert_eq!(row.get(1), Some(&serde_json::json!("rejected")));
    let gone = reopened
        .execute_query("SELECT count(*) FROM _quack_graph_merges WHERE id = 'm2'")
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        gone.rows.first().and_then(|r| r.first()),
        Some(&serde_json::json!(0)),
        "the pending duplicate in the flipped orientation must be gone"
    );
    let single = reopened
        .execute_query("SELECT status FROM _quack_graph_merges WHERE id = 'm3'")
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        single.rows.first().and_then(|r| r.first()),
        Some(&serde_json::json!("rejected"))
    );
}

/// Sample rows in the system prompt are cut per cell: nested values
/// too, which once put a 58,000-character JSON cell (and a header
/// padded to match) into every turn's prompt.
#[test]
fn cells_are_cut_whatever_their_type() {
    let long = "x".repeat(100);
    let results = QueryResults {
        columns: vec![String::from("s"), String::from("j"), String::from("n")],
        rows: vec![vec![
            serde_json::Value::String(long.clone()),
            serde_json::json!({ "a": [long.clone(), long] }),
            serde_json::json!(12_345),
        ]],
    };
    let cut = results.with_cells_cut(10);
    let cells: Vec<String> = cut.rows.iter().flatten().map(|v| Cell(v).label()).collect();
    assert_eq!(
        cells,
        vec![
            format!("{}\u{2026}", "x".repeat(10)),
            String::from("{\"a\":[\"xxx\u{2026}"),
            String::from("12345"),
        ]
    );
}

/// A sort rewrites the statement's own `ORDER BY`, replacing any it
/// had and sitting before its `LIMIT`; the column is named when the
/// name resolves and given by position when it does not. Anything that
/// is not one `SELECT` has no rows of its own to reorder.
#[test]
fn sorting_rewrites_the_statements_own_order_by() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    db.execute_query(
        "CREATE TABLE t AS SELECT * FROM (VALUES (2, 'b'), (NULL, 'n'), (1, 'a')) v(x, y)",
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    let rewrite = |sql: &str, column: usize, direction: SortDirection| {
        let sortable = db
            .sortable(sql)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .unwrap_or_else(|| fail(&format!("{sql} is not sortable")));
        let column = std::num::NonZeroUsize::new(column).unwrap_or_else(|| fail("zero column"));
        db.sort_statement(&sortable, ResultSort { column, direction })
            .unwrap_or_else(|e| fail(&e.to_string()))
    };
    let ys = |sql: &str| -> Vec<String> {
        db.execute_query(sql)
            .unwrap_or_else(|e| fail(&format!("{sql}: {e}")))
            .rows
            .iter()
            .filter_map(|r| {
                r.get(1)
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .collect()
    };

    let sql = rewrite("SELECT x, y FROM t", 1, SortDirection::Asc);
    assert_eq!(sql, "SELECT x, y FROM t ORDER BY x ASC NULLS LAST");
    assert_eq!(ys(&sql), ["a", "b", "n"]);

    // Sorting the rewritten statement again replaces its ORDER BY.
    let again = rewrite(&sql, 2, SortDirection::Desc);
    assert_eq!(again, "SELECT x, y FROM t ORDER BY y DESC NULLS LAST");
    assert_eq!(ys(&again), ["n", "b", "a"]);

    // Comments and the semicolon go; the sort comes before the LIMIT.
    let limited = rewrite(
        "-- note\nSELECT * FROM t /* all */ LIMIT 2;",
        1,
        SortDirection::Desc,
    );
    assert_eq!(
        limited,
        "SELECT * FROM t ORDER BY x DESC NULLS LAST LIMIT 2"
    );
    assert_eq!(ys(&limited), ["b", "a"]);

    // An unnamed expression, or a name two columns share, sorts by position.
    let counted = rewrite(
        "SELECT y, count(*) FROM t GROUP BY y",
        2,
        SortDirection::Desc,
    );
    assert!(counted.ends_with("ORDER BY 2 DESC NULLS LAST"), "{counted}");
    let shared = rewrite("SELECT x AS v, y AS v FROM t", 2, SortDirection::Asc);
    assert!(shared.ends_with("ORDER BY 2 ASC NULLS LAST"), "{shared}");
    assert!(db.execute_query(&shared).is_ok());

    for unsortable in [
        "CREATE TABLE u (x INT)",
        "INSERT INTO t VALUES (3, 'c')",
        "SELECT 1; SELECT 2",
        "SELEC broken",
        "",
    ] {
        assert!(
            db.sortable(unsortable)
                .unwrap_or_else(|e| fail(&e.to_string()))
                .is_none(),
            "{unsortable}"
        );
    }
}

/// A document's chunks page in document order from any position, whatever
/// its status, and the model's name for a document resolves by id, exact
/// file name, or id prefix, else errors with the documents there are.
#[test]
fn document_chunks_page_in_order_and_documents_resolve_by_id_name_or_prefix() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    let id = DocumentId::from("doc-aaaa");
    db.insert_document(
        &NewDocument::new(&id, "policy.md", "text/markdown", 1).with_status(DocumentStatus::Error),
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    for i in 0..4u32 {
        db.chunk_writer(&id, &format!("part {i}"))
            .and_then(|writer| {
                writer.insert(&NewChunk {
                    id: &ChunkId::from(format!("c{i}")),
                    chunk_index: i,
                    content: &format!("part {i}"),
                    heading: None,
                    page: Some(i.saturating_add(1)),
                    kind: SectionKind::Body,
                    locator: None,
                    embedding: None,
                })
            })
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    let positions = |from: u32, limit: u32| -> Vec<u32> {
        db.document_chunks(&id, from, limit)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .iter()
            .map(|c| c.chunk_index)
            .collect()
    };
    assert_eq!(positions(0, 10), [0, 1, 2, 3]);
    assert_eq!(positions(1, 2), [1, 2]);
    assert_eq!(positions(3, 2), [3]);
    assert!(positions(4, 2).is_empty());
    let page = db
        .document_chunks(&id, 2, 1)
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        page.first()
            .map(|c| (c.content.as_str(), c.page, c.filename.as_str())),
        Some(("part 2", Some(3), "policy.md"))
    );

    let documents = db.list_documents().unwrap_or_else(|e| fail(&e.to_string()));
    for want in ["doc-aaaa", "policy.md", "doc-a", " doc-aaaa "] {
        assert_eq!(
            DocumentInfo::find(&documents, want)
                .map(|d| d.id.as_str())
                .ok(),
            Some("doc-aaaa"),
            "{want}"
        );
    }
    assert!(
        DocumentInfo::find(&documents, " ")
            .err()
            .is_some_and(|e| e.to_string().contains("no document named"))
    );
    for want in ["pol", "doc-b"] {
        let error = DocumentInfo::find(&documents, want)
            .map(|d| d.id.clone())
            .map_err(|e| e.to_string());
        let Err(error) = error else {
            fail(&format!("'{want}' matched a document"))
        };
        assert!(
            error.contains(&format!("no document matches '{want}'"))
                && error.contains("doc-aaaa (policy.md)"),
            "{error}"
        );
    }
    // An exact title names a document too; one several share is refused.
    let mut titled = documents;
    for (doc, title) in titled.iter_mut().zip(["Returns policy"]) {
        doc.title = Some(String::from(title));
    }
    assert_eq!(
        DocumentInfo::find(&titled, "Returns policy")
            .map(|d| d.id.as_str())
            .ok(),
        Some("doc-aaaa")
    );
    assert!(DocumentInfo::find(&titled, "returns policy").is_err());
    let mut twice = titled.clone();
    twice.extend(titled.iter().cloned().map(|mut d| {
        d.id = DocumentId::from("doc-bbbb");
        d.filename = String::from("policy-2.md");
        d
    }));
    let shared = DocumentInfo::find(&twice, "Returns policy")
        .map(|d| d.id.clone())
        .map_err(|e| e.to_string());
    assert!(
        shared
            .as_ref()
            .is_err_and(|e| e.contains("2 documents are titled 'Returns policy'")
                && e.contains("doc-bbbb (policy-2.md, \"Returns policy\")")),
        "{shared:?}"
    );
}

/// Every row of a statement streams out in each format, the shapes the
/// collected writers make, with a NULL as an empty CSV cell and JSON null.
#[test]
fn stream_query_writes_every_row_in_each_format() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    db.execute_query(
        "CREATE TABLE t AS SELECT * FROM (VALUES (1, 'a'), (2, NULL), (3, 'c,\"q\"')) v(n, s)",
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    let stream = |format| {
        let mut out = Vec::new();
        let rows = db
            .read_only(|db| db.stream_query("SELECT n, s FROM t ORDER BY n", format, &mut out))
            .unwrap_or_else(|e| fail(&e.to_string()));
        (rows, String::from_utf8(out).unwrap_or_default())
    };
    let (rows, csv) = stream(ExportFormat::Csv);
    assert_eq!(rows, 3);
    assert_eq!(csv, "n,s\n1,a\n2,\n3,\"c,\"\"q\"\"\"\n");
    let (rows, ndjson) = stream(ExportFormat::Ndjson);
    assert_eq!(rows, 3);
    assert_eq!(
        ndjson,
        "{\"n\":1,\"s\":\"a\"}\n{\"n\":2,\"s\":null}\n{\"n\":3,\"s\":\"c,\\\"q\\\"\"}\n"
    );
    let (rows, json) = stream(ExportFormat::Json);
    assert_eq!(rows, 3);
    let parsed: serde_json::Value =
        serde_json::from_str(&json).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(parsed.as_array().map(Vec::len), Some(3));
    assert_eq!(
        parsed.get(1).and_then(|r| r.get("s")),
        Some(&serde_json::Value::Null)
    );
    // A statement with a result shape but no rows: the header alone.
    let (rows, csv) = stream_statement(&db, "CREATE TABLE empty_t (a INT)", ExportFormat::Csv);
    assert_eq!((rows, csv.as_str()), (0, "Count\n"));
    assert!(
        db.read_only(|db| db.stream_query(
            "SELECT * FROM nope",
            ExportFormat::Csv,
            &mut Vec::new()
        ))
        .is_err()
    );
}

fn stream_statement(db: &WorkspaceDb, sql: &str, format: ExportFormat) -> (u64, String) {
    let mut out = Vec::new();
    let rows = db
        .stream_query(sql, format, &mut out)
        .unwrap_or_else(|e| fail(&e.to_string()));
    (rows, String::from_utf8(out).unwrap_or_default())
}

/// A ready document whose language is detected from its one chunk of
/// `content`.
fn language_document(db: &WorkspaceDb, id: &str, content: &str) {
    insert_ready_document(db, id);
    insert_text_chunk(db, &format!("{id}-c0"), id, 0, content);
}

const GERMAN_TEXT: &str = "Die Versicherungsverträge werden jedes Jahr erneuert. Der Kunde \
    erhält rechtzeitig eine Mitteilung über die neuen Bedingungen und kann widersprechen.";

#[test]
fn a_german_document_is_stemmed_as_german_and_found_by_another_inflection() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    language_document(&db, "de", GERMAN_TEXT);
    let document = db
        .document(&DocumentId::from("de"))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(document.and_then(|d| d.language).as_deref(), Some("deu"));
    assert_eq!(
        db.meta(MetaKey::Languages)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .as_deref(),
        Some("german")
    );
    let hits = db
        .search_keyword_chunks("Versicherungsvertrag", 5, &ChunkScope::all())
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(hits.first().map(|h| h.id.as_str()), Some("de-c0"));
}

#[test]
fn a_query_reaches_documents_in_every_language_of_the_workspace() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    language_document(&db, "de", GERMAN_TEXT);
    let english = "The insurance policies are renewed every year, and the customer is told.";
    language_document(&db, "en", english);
    let japanese = "保険契約の更新手続きについて説明します。";
    language_document(&db, "ja", japanese);
    assert_eq!(
        db.meta(MetaKey::Languages)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .as_deref(),
        Some("english,german,unstemmed")
    );
    let found = |query: &str| -> Vec<String> {
        db.search_keyword_chunks(query, 5, &ChunkScope::all())
            .unwrap_or_else(|e| fail(&e.to_string()))
            .into_iter()
            .map(|h| h.document_id.into_string())
            .collect()
    };
    assert_eq!(found("renewal"), ["en"]);
    assert_eq!(found("Versicherungsvertrag"), ["de"]);
    assert_eq!(found("更新手続き"), ["ja"]);
}

/// The writer resolves the language once: every chunk it stores is indexed
/// under it, whatever the row says later and whatever each chunk's own text
/// would be detected as.
#[test]
fn a_document_s_chunks_are_indexed_under_the_language_resolved_once() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    insert_ready_document(&db, "de");
    let writer = db
        .chunk_writer(&DocumentId::from("de"), GERMAN_TEXT)
        .unwrap_or_else(|e| fail(&e.to_string()));
    let document = db
        .document(&DocumentId::from("de"))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(document.and_then(|d| d.language).as_deref(), Some("deu"));
    db.execute_statement("UPDATE _quack_documents SET language = 'eng'")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let chunk = |id: &str, content: &str| {
        writer
            .insert(&NewChunk {
                id: &ChunkId::from(id),
                chunk_index: 0,
                content,
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: None,
            })
            .unwrap_or_else(|e| fail(&e.to_string()));
    };
    chunk("c0", "Die Versicherungsverträge");
    chunk("c1", "The renewals");
    let terms = |chunk: &str| {
        db.execute_query(&format!(
            "SELECT string_agg(term, ',' ORDER BY term) FROM _quack_terms WHERE chunk_id = '{chunk}'"
        ))
        .unwrap_or_else(|e| fail(&e.to_string()))
        .rows
    };
    assert_eq!(
        terms("c0"),
        vec![vec![serde_json::json!("die,versicherungsvertrag")]]
    );
    assert_eq!(terms("c1"), vec![vec![serde_json::json!("renewal,the")]]);
}

#[test]
fn the_languages_setting_fixes_what_a_document_is_detected_as() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4))
        .unwrap_or_else(|e| fail(&e.to_string()))
        .with_languages(LanguageSetting::Only(vec![Language::ENGLISH]));
    language_document(&db, "de", GERMAN_TEXT);
    let document = db
        .document(&DocumentId::from("de"))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(document.and_then(|d| d.language).as_deref(), Some("eng"));
}

/// A workspace from before detection has no language on its documents and
/// English terms: opening it under the new schema detects each document
/// and reindexes every chunk under its language.
#[test]
fn the_upgrade_detects_languages_and_reindexes_the_terms() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    language_document(&db, "de", GERMAN_TEXT);
    db.execute_statement("UPDATE _quack_documents SET language = NULL")
        .unwrap_or_else(|e| fail(&e.to_string()));
    db.execute_statement("DELETE FROM _quack_terms")
        .unwrap_or_else(|e| fail(&e.to_string()));
    db.delete_meta(MetaKey::Languages)
        .unwrap_or_else(|e| fail(&e.to_string()));
    db.set_meta(MetaKey::SchemaVersion, &MERGE_DEDUP.to_string())
        .unwrap_or_else(|e| fail(&e.to_string()));
    db.upgrade_data(db.embedding_dimension())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let document = db
        .document(&DocumentId::from("de"))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(document.and_then(|d| d.language).as_deref(), Some("deu"));
    assert_eq!(
        db.meta(MetaKey::SchemaVersion)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .as_deref(),
        Some(WORKSPACE_SCHEMA_VERSION.to_string().as_str())
    );
    let hits = db
        .search_keyword_chunks("Versicherungsvertrag", 5, &ChunkScope::all())
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(hits.len(), 1);
}

fn vector_chunk(db: &WorkspaceDb, id: &str, index: u32, content: &str, vector: [f32; 4]) {
    let embedding = Vector::from(vector.to_vec());
    db.chunk_writer(&DocumentId::from("doc1"), content)
        .and_then(|writer| {
            writer.insert(&NewChunk {
                id: &ChunkId::from(id),
                chunk_index: index,
                content,
                heading: None,
                page: None,
                kind: SectionKind::Body,
                locator: None,
                embedding: Some(&embedding),
            })
        })
        .unwrap_or_else(|e| fail(&e.to_string()));
}

#[test]
fn an_explained_search_keeps_each_legs_rank_and_score_on_the_fused_hits() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    insert_ready_document(&db, "doc1");
    vector_chunk(
        &db,
        "near",
        0,
        "Hail damage to roofs.",
        [1.0, 0.0, 0.0, 0.0],
    );
    vector_chunk(
        &db,
        "word",
        1,
        "Flood exclusion applies.",
        [0.0, 1.0, 0.0, 0.0],
    );
    let query = Vector::from(vec![1.0_f32, 0.0, 0.0, 0.0]);
    let limits = HybridLimits {
        top_k: 5,
        rrf_k: 60,
    };
    let explained = db
        .explain_search("flood", &query, limits, &ChunkScope::all())
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(explained.vector.len(), 2);
    assert_eq!(explained.keyword.len(), 1);
    assert!(explained.phrases.is_empty() && explained.phrase_note().is_none());
    let hit = |id: &str| {
        explained
            .fused
            .iter()
            .find(|h| h.id.as_str() == id)
            .map_or_else(|| fail(&format!("{id} missing")), |h| h.ranks)
    };
    let word = hit("word");
    assert_eq!(word.vector_rank, Some(2));
    assert_eq!(word.keyword_rank, Some(1));
    assert!(word.bm25.is_some_and(|b| b > 0.0) && word.vector_score.is_some());
    let near = hit("near");
    assert_eq!((near.vector_rank, near.keyword_rank), (Some(1), None));
    assert_eq!(near.vector_score, Some(1.0));
    // Hybrid search is the explanation's fused list.
    let hybrid = db
        .search_hybrid_chunks("flood", &query, limits, &ChunkScope::all())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let ids = |hits: &[ChunkSearchResult]| -> Vec<String> {
        hits.iter().map(|h| h.id.to_string()).collect()
    };
    assert_eq!(ids(&hybrid), ids(&explained.fused));
}

#[test]
fn each_search_mode_runs_its_own_legs() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    insert_ready_document(&db, "doc1");
    vector_chunk(
        &db,
        "near",
        0,
        "Hail damage to roofs.",
        [1.0, 0.0, 0.0, 0.0],
    );
    vector_chunk(
        &db,
        "word",
        1,
        "Flood exclusion applies.",
        [0.0, 1.0, 0.0, 0.0],
    );
    let query = Vector::from(vec![1.0_f32, 0.0, 0.0, 0.0]);
    let limits = HybridLimits {
        top_k: 1,
        rrf_k: 60,
    };
    let run = |mode: SearchMode, vector: Option<&Vector>| {
        db.search_chunks("flood", vector, mode, limits, &ChunkScope::all())
    };
    let keyword = run(SearchMode::Keyword, Some(&query)).unwrap_or_else(|e| fail(&e.to_string()));
    assert!(keyword.vector.is_empty());
    assert_eq!(keyword.fused.first().map(|h| h.id.as_str()), Some("word"));
    let vector = run(SearchMode::Vector, Some(&query)).unwrap_or_else(|e| fail(&e.to_string()));
    assert!(vector.keyword.is_empty());
    assert_eq!(vector.fused.first().map(|h| h.id.as_str()), Some("near"));
    // Hybrid without a vector is keyword alone; vector without one fails.
    let fallback = run(SearchMode::Hybrid, None).unwrap_or_else(|e| fail(&e.to_string()));
    assert!(fallback.vector.is_empty() && !fallback.fused.is_empty());
    let refused = run(SearchMode::Vector, None).err().map(|e| e.to_string());
    assert!(refused.is_some_and(|e| e.contains("embedding model")));
    // A quoted phrase filters the vector leg too, and says so.
    let phrased = db
        .search_chunks(
            "\"hail damage\"",
            Some(&query),
            SearchMode::Vector,
            HybridLimits {
                top_k: 5,
                rrf_k: 60,
            },
            &ChunkScope::all(),
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(phrased.fused.len(), 1);
    assert!(
        phrased
            .phrase_note()
            .is_some_and(|n| n.contains("\"hail damage\""))
    );
}

/// Documents with metadata for the filter tests: a tagged PDF by Ana from
/// 2026, an untagged Markdown file from 2024, and a pasted note.
fn filtered_documents(db: &WorkspaceDb) {
    for (id, filename, mime, source) in [
        (
            "pdf",
            "policy.pdf",
            "application/pdf",
            DocumentSource::Upload,
        ),
        ("md", "notes.md", "text/markdown", DocumentSource::Path),
        ("txt", "paste.txt", "text/plain", DocumentSource::Paste),
    ] {
        db.insert_document(&NewDocument {
            source,
            ..NewDocument::new(&DocumentId::from(id), filename, mime, 10)
                .with_status(DocumentStatus::Ready)
        })
        .unwrap_or_else(|e| fail(&e.to_string()));
        insert_text_chunk(db, &format!("{id}-c0"), id, 0, "renewal terms apply");
    }
    let set = |id: &str, fields: DocumentFields| {
        db.set_document_fields(&DocumentId::from(id), &fields)
            .unwrap_or_else(|e| fail(&e.to_string()));
    };
    set(
        "pdf",
        DocumentFields {
            author: Some(String::from("Ana Lima")),
            authored_at: Some(String::from("2026-03-01")),
            tags: Some(vec![String::from("Policy"), String::from("2026")]),
            ..DocumentFields::default()
        },
    );
    set(
        "md",
        DocumentFields {
            authored_at: Some(String::from("2024-06-30")),
            ..DocumentFields::default()
        },
    );
}

#[test]
fn a_document_filter_narrows_the_listing_and_both_search_legs() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    filtered_documents(&db);
    let listed = |filter: DocumentFilter| -> Vec<String> {
        let mut ids: Vec<String> = db
            .list_documents_matching(&filter)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .into_iter()
            .map(|d| d.id.into_string())
            .collect();
        ids.sort();
        ids
    };
    let date = |text: &str| text.parse::<jiff::civil::Date>().ok();
    assert_eq!(listed(DocumentFilter::default()), ["md", "pdf", "txt"]);
    assert_eq!(
        listed(DocumentFilter {
            types: vec![String::from("PDF"), String::from(".md")],
            ..DocumentFilter::default()
        }),
        ["md", "pdf"]
    );
    assert_eq!(
        listed(DocumentFilter {
            types: vec![String::from("text/plain")],
            ..DocumentFilter::default()
        }),
        ["txt"]
    );
    assert_eq!(
        listed(DocumentFilter {
            sources: vec![DocumentSource::Paste, DocumentSource::Path],
            ..DocumentFilter::default()
        }),
        ["md", "txt"]
    );
    assert_eq!(
        listed(DocumentFilter {
            tags: vec![String::from("policy")],
            ..DocumentFilter::default()
        }),
        ["pdf"]
    );
    assert_eq!(
        listed(DocumentFilter {
            author: Some(String::from("LIMA")),
            ..DocumentFilter::default()
        }),
        ["pdf"]
    );
    assert_eq!(
        listed(DocumentFilter {
            since: date("2025-01-01"),
            until: date("2026-12-31"),
            types: vec![String::from("pdf"), String::from("md")],
            ..DocumentFilter::default()
        }),
        ["pdf"]
    );
    assert_eq!(
        listed(DocumentFilter {
            until: date("2024-12-31"),
            ..DocumentFilter::default()
        }),
        ["md"]
    );
    let scope = ChunkScope::all()
        .with_filter(&DocumentFilter {
            tags: vec![String::from("2026")],
            ..DocumentFilter::default()
        })
        .unwrap_or_else(|e| fail(&e.to_string()));
    let keyword = db
        .search_keyword_chunks("renewal", 5, &scope)
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        keyword.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        ["pdf-c0"]
    );
}

#[test]
fn a_bad_document_filter_is_refused_with_the_reason() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    let unknown = db
        .list_documents_matching(&DocumentFilter {
            types: vec![String::from("klingon")],
            ..DocumentFilter::default()
        })
        .err()
        .map(|e| e.to_string());
    assert!(unknown.is_some_and(|e| e.contains("unknown document type 'klingon'")));
    let backwards = ChunkScope::all()
        .with_filter(&DocumentFilter {
            since: "2026-02-01".parse().ok(),
            until: "2026-01-01".parse().ok(),
            ..DocumentFilter::default()
        })
        .err()
        .map(|e| e.to_string());
    assert!(backwards.is_some_and(|e| e.contains("is after until")));
    assert!(DocumentFilter::default().is_empty());
}

/// An id prefix or a file name that several documents share is refused,
/// never a pick of the first, and a miss in a large workspace lists 20
/// documents and counts the rest.
#[test]
fn ambiguous_document_names_are_refused_and_misses_are_capped() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    for (id, filename) in [("doc-aaaa", "policy.md"), ("doc-bbbb", "policy-2.md")] {
        db.insert_document(&NewDocument::new(
            &DocumentId::from(id),
            filename,
            "text/markdown",
            1,
        ))
        .unwrap_or_else(|e| fail(&e.to_string()));
    }
    let twice = db.list_documents().unwrap_or_else(|e| fail(&e.to_string()));
    let titled = twice.clone();
    let prefix = DocumentInfo::find(&twice, "doc-")
        .err()
        .map(|e| e.to_string());
    assert!(
        prefix
            .as_deref()
            .is_some_and(|e| e.contains("2 documents have ids starting with 'doc-'")),
        "{prefix:?}"
    );
    let mut same_name = twice;
    for document in &mut same_name {
        document.filename = String::from("policy.md");
    }
    let named = DocumentInfo::find(&same_name, "policy.md")
        .err()
        .map(|e| e.to_string());
    assert!(
        named
            .as_deref()
            .is_some_and(|e| e.contains("2 documents are named 'policy.md'")),
        "{named:?}"
    );
    // A miss in a large workspace lists 20 documents and counts the rest.
    let many: Vec<DocumentInfo> = (0..25)
        .map(|i| {
            let mut d = titled
                .first()
                .cloned()
                .unwrap_or_else(|| fail("a document"));
            d.id = DocumentId::from(format!("id-{i:02}"));
            d
        })
        .collect();
    let miss = DocumentInfo::find(&many, "nothing")
        .err()
        .map(|e| e.to_string());
    assert!(
        miss.as_deref().is_some_and(|e| e.contains("id-19")
            && !e.contains("id-20")
            && e.contains("and 5 more")),
        "{miss:?}"
    );
}

/// A deleted user leaves no id or name behind in any table that records
/// who made or changed something.
#[test]
fn forget_user_rewrites_every_table_that_names_a_person() {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    for sql in [
        "INSERT INTO _quack_saved_questions \
         (id, name, question, mode, statements, session_id, created_by) \
         VALUES ('q', 'n', 'q', 'chat', '[]', 's', 'u-bob')",
        "INSERT INTO _quack_imports (id, name, url, table_name, created_by) \
         VALUES ('i', 'n', 'https://x/f.csv', 't', 'u-bob')",
        "INSERT INTO _quack_table_notes (table_name, note, edited_by) VALUES ('t', 'n', 'bob')",
        "INSERT INTO _quack_provenance (subject_id, author) VALUES ('s', 'bob')",
        "INSERT INTO _quack_graph_merges (id, keep_node_id, drop_node_id, distance, decided_by) \
         VALUES ('m', 'a', 'b', 0.1, 'bob')",
    ] {
        db.execute_statement(sql)
            .unwrap_or_else(|e| fail(&format!("{sql}: {e}")));
    }
    let changed = db
        .forget_user(&UserId::from("u-bob"), "bob")
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(changed, 5);
    for (table, column) in [
        ("_quack_saved_questions", "created_by"),
        ("_quack_imports", "created_by"),
        ("_quack_table_notes", "edited_by"),
        ("_quack_provenance", "author"),
        ("_quack_graph_merges", "decided_by"),
    ] {
        let left: i64 = db
            .connection()
            .query_row(
                &format!("SELECT count(*) FROM {table} WHERE {column} IN ('u-bob', 'bob')"),
                [],
                |row| row.get(0),
            )
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(left, 0, "{table}.{column}");
    }
}
