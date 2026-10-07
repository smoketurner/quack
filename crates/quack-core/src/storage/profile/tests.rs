#![expect(clippy::unwrap_used, reason = "test setup")]

use super::*;
use crate::embedding::Dimension;

fn db() -> WorkspaceDb {
    WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap()
}

fn warnings_of(profile: &TableProfile, key: Option<&str>) -> Vec<(String, ColumnWarning)> {
    profile
        .warnings(key)
        .into_iter()
        .map(|f| (f.column, f.warning))
        .collect()
}

#[test]
fn a_profile_counts_samples_and_flags_mistyped_text() {
    let db = db();
    db.execute_statement(
        "CREATE TABLE orders AS SELECT * FROM (VALUES \
         (1, '100', '2026-01-02', NULL, 'a'), \
         (2, '250', '2026-01-03', NULL, 'b'), \
         (2, '7', 'soon', NULL, NULL)) t(id, amount, placed, empty, maybe)",
    )
    .unwrap();
    let profile = TableProfile::refresh(&db, "orders").unwrap();
    assert_eq!(profile.row_count, 3);
    let amount = profile.column("amount").unwrap();
    assert_eq!((amount.non_null, amount.distinct), (3, 3));
    assert_eq!(amount.samples.len(), 3);
    assert!(amount.number_share >= 1.0);
    let placed = profile.column("placed").unwrap();
    assert!((placed.date_share - 2.0 / 3.0).abs() < 1e-9);

    let warnings = warnings_of(&profile, None);
    assert!(warnings.contains(&(
        String::from("amount"),
        ColumnWarning::NumericText { share: Share(1.0) }
    )));
    assert!(warnings.contains(&(String::from("empty"), ColumnWarning::AllNull)));
    assert!(warnings.contains(&(
        String::from("id"),
        ColumnWarning::DuplicateKey { duplicates: 1 }
    )));
    // Two of three dates is below the bar, and a third empty is not "high".
    assert!(!warnings.iter().any(|(c, _)| c == "placed" || c == "maybe"));
    assert_eq!(
        ColumnWarning::NumericText { share: Share(1.0) }.fix(),
        Some(ColumnType::Double)
    );
    assert_eq!(
        ColumnWarning::NumericText { share: Share(0.95) }.fix(),
        None,
        "a fix is offered only when every value converts"
    );
}

#[test]
fn a_stored_profile_is_shown_only_at_its_row_count() {
    let db = db();
    db.execute_statement("CREATE TABLE t AS SELECT range AS n FROM range(5)")
        .unwrap();
    assert_eq!(TableProfile::refresh_stale(&db).unwrap(), 1);
    assert!(TableProfile::current(&db, "t", 5).unwrap().is_some());
    assert_eq!(
        TableProfile::refresh_stale(&db).unwrap(),
        0,
        "nothing changed"
    );

    db.execute_statement("INSERT INTO t VALUES (9)").unwrap();
    assert!(
        TableProfile::current(&db, "t", 6).unwrap().is_none(),
        "a stale profile is not shown"
    );
    assert_eq!(TableProfile::refresh_stale(&db).unwrap(), 1);
    assert!(TableProfile::current(&db, "t", 6).unwrap().is_some());

    db.execute_statement("DROP TABLE t").unwrap();
    TableProfile::refresh_stale(&db).unwrap();
    assert!(
        TableProfile::all(&db).unwrap().is_empty(),
        "a dropped table's profile goes"
    );
}

#[test]
fn views_and_empty_tables_profile_without_warnings() {
    let db = db();
    db.execute_statement("CREATE TABLE e (a VARCHAR)").unwrap();
    db.execute_statement("CREATE VIEW v AS SELECT 1 AS x")
        .unwrap();
    TableProfile::refresh_stale(&db).unwrap();
    let all = TableProfile::all(&db).unwrap();
    assert!(all.contains_key("e") && !all.contains_key("v"));
    assert!(all.get("e").is_some_and(|p| p.warnings(None).is_empty()));
}

#[test]
fn column_types_parse_and_retype_strictly() {
    let types: ColumnTypes = "amount=double, placed = DATE".parse().unwrap();
    assert_eq!(
        types,
        ColumnTypes::new(vec![
            (String::from("amount"), ColumnType::Double),
            (String::from("placed"), ColumnType::Date),
        ])
    );
    assert!("amount".parse::<ColumnTypes>().is_err());
    assert!("=DOUBLE".parse::<ColumnTypes>().is_err());
    assert!("amount=MONEY".parse::<ColumnTypes>().is_err());

    let db = db();
    db.execute_statement(
        "CREATE TABLE t AS SELECT * FROM (VALUES ('1', 'x'), ('2', 'y')) v(amount, label)",
    )
    .unwrap();
    "amount=BIGINT"
        .parse::<ColumnTypes>()
        .unwrap()
        .apply(&db, "t")
        .unwrap();
    let profile = TableProfile::current(&db, "t", 2).unwrap().unwrap();
    assert_eq!(profile.column("amount").unwrap().duckdb_type, "BIGINT");

    let refused = "label=DOUBLE"
        .parse::<ColumnTypes>()
        .unwrap()
        .apply(&db, "t");
    assert!(refused.is_err_and(|e| e.to_string().contains("does not convert")));
    let missing = "ghost=DOUBLE"
        .parse::<ColumnTypes>()
        .unwrap()
        .apply(&db, "t");
    assert!(missing.is_err_and(|e| e.to_string().contains("no column 'ghost'")));
}

#[test]
fn notes_set_replace_clear_and_refuse() {
    let db = db();
    db.execute_statement("CREATE TABLE orders (id INTEGER)")
        .unwrap();
    TableNote::set(&db, "orders", " amounts are in cents ", Some("ana")).unwrap();
    let note = TableNote::get(&db, "orders").unwrap().unwrap();
    assert_eq!(note.note, "amounts are in cents");
    assert_eq!(note.edited_by.as_deref(), Some("ana"));
    TableNote::set(&db, "orders", "in dollars", None).unwrap();
    assert_eq!(TableNote::all(&db).unwrap().len(), 1);
    TableNote::set(&db, "orders", "  ", None).unwrap();
    assert!(TableNote::get(&db, "orders").unwrap().is_none());

    assert!(TableNote::set(&db, "ghost", "x", None).is_err());
    let long = "x".repeat(TableNote::MAX_CHARS + 1);
    assert!(TableNote::set(&db, "orders", &long, None).is_err());
}

/// Column types are stored as named pairs: a column whose name holds a
/// comma or an equals sign, which the `col=TYPE` text cannot carry, comes
/// back whole.
#[test]
fn column_types_round_trip_any_column_name() {
    let types = ColumnTypes::new(vec![
        (String::from("total, net"), ColumnType::Double),
        (String::from("a=b"), ColumnType::Bigint),
    ]);
    let json = serde_json::to_string(&types).unwrap_or_default();
    assert_eq!(
        json,
        r#"[{"column":"total, net","type":"DOUBLE"},{"column":"a=b","type":"BIGINT"}]"#
    );
    let back: ColumnTypes = serde_json::from_str(&json).unwrap_or_default();
    assert_eq!(back, types);
}
