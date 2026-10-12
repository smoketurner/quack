use super::*;
use crate::embedding::Dimension;
use crate::storage::workspace::WorkspaceDb;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

fn schema() -> SqlSchema {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    for sql in [
        "CREATE TABLE sales (region VARCHAR, revenue INTEGER, sold DATE)",
        "CREATE TABLE stores (id INTEGER, \"Region\" VARCHAR)",
        "CREATE TABLE \"Order Items\" (sku VARCHAR)",
    ] {
        db.execute_statement(sql)
            .unwrap_or_else(|e| fail(&e.to_string()));
    }
    db.sql_schema().unwrap_or_else(|e| fail(&e.to_string()))
}

/// The cursor at `|` (the end when there is none), as character offsets.
fn split(line: &str) -> (String, usize) {
    let cursor = line.find('|').map_or_else(
        || line.chars().count(),
        |at| line.get(..at).map_or(0, |head| head.chars().count()),
    );
    (line.replace('|', ""), cursor)
}

/// The words offered, names only.
fn offered(line: &str) -> Vec<String> {
    let (line, cursor) = split(line);
    SqlCompletion::for_sql(&line, cursor, &schema(), &SqlVocabulary::none())
        .map(|c| c.items.into_iter().map(|s| s.word).collect())
        .unwrap_or_default()
}

#[test]
fn tables_complete_after_from_and_join() {
    assert_eq!(offered("SELECT * FROM sa"), ["sales"]);
    assert_eq!(
        offered("SELECT * FROM "),
        ["\"Order Items\"", "sales", "stores"]
    );
    assert_eq!(offered("SELECT * FROM sales JOIN st"), ["stores"]);
    assert_eq!(offered("SELECT * FROM sales, st"), ["stores"]);
    assert_eq!(offered("describe sa"), ["sales"]);
    assert_eq!(offered("SUMMARIZE o"), ["\"Order Items\""]);
}

#[test]
fn columns_come_from_the_tables_the_statement_names() {
    assert_eq!(offered("SELECT re| FROM sales"), ["region", "revenue"]);
    assert_eq!(
        offered("SELECT * FROM sales WHERE r"),
        ["region", "revenue"]
    );
    // Both tables' columns, each once, then table names to qualify with.
    assert_eq!(
        offered("SELECT * FROM sales JOIN stores ON s"),
        ["sold", "sales", "stores"]
    );
    assert_eq!(offered("SELECT * FROM stores ORDER BY r"), ["\"Region\""]);
}

#[test]
fn a_comma_from_list_keeps_resolving_tables_after_an_unresolved_name() {
    assert_eq!(
        offered("WITH cte AS (SELECT 1 AS x) SELECT re| FROM cte, sales"),
        ["region", "revenue"]
    );
    assert_eq!(
        offered("WITH cte AS (SELECT 1 AS x) SELECT re| FROM cte JOIN sales"),
        ["region", "revenue"]
    );
    assert_eq!(
        offered("SELECT re| FROM cte, typo, sales"),
        ["region", "revenue"]
    );
    assert_eq!(
        offered("SELECT re| FROM cte AS c, sales"),
        ["region", "revenue"]
    );
    assert_eq!(
        offered("SELECT re| FROM cte c, sales"),
        ["region", "revenue"]
    );
    assert_eq!(
        offered("SELECT s.| FROM cte, sales AS s"),
        ["region", "revenue", "sold"]
    );
    assert!(offered("SELECT c.| FROM cte c, sales").is_empty());
    assert_eq!(offered("SELECT * FROM cte, st"), ["stores"]);
}

#[test]
fn a_blank_word_gets_columns_where_an_expression_must_follow() {
    assert_eq!(
        offered("SELECT * FROM sales WHERE "),
        [
            "region",
            "revenue",
            "sold",
            "\"Order Items\"",
            "sales",
            "stores"
        ]
    );
    assert_eq!(
        offered("SELECT | FROM sales"),
        [
            "region",
            "revenue",
            "sold",
            "\"Order Items\"",
            "sales",
            "stores"
        ]
    );
    assert_eq!(
        offered("SELECT * FROM sales WHERE region = 'n' AND ")
            .first()
            .map(String::as_str),
        Some("region")
    );
    assert_eq!(
        offered("SELECT region, | FROM sales")
            .first()
            .map(String::as_str),
        Some("region")
    );
    // Not where the statement may already be whole.
    for line in [
        "SELECT * FROM sales WHERE revenue > 1 ",
        "SELECT * ",
        "SELECT * FROM sales ORDER BY revenue DESC ",
    ] {
        assert!(offered(line).is_empty(), "{line}: {:?}", offered(line));
    }
}

#[test]
fn a_qualifier_offers_only_its_tables_columns() {
    assert_eq!(
        offered("SELECT s.| FROM sales s"),
        ["region", "revenue", "sold"]
    );
    assert_eq!(
        offered("SELECT s.re| FROM sales AS s"),
        ["region", "revenue"]
    );
    assert_eq!(offered("SELECT stores.| FROM stores"), ["id", "\"Region\""]);
    assert!(offered("SELECT x.| FROM sales s").is_empty());
}

#[test]
fn nothing_is_offered_where_no_name_belongs() {
    for line in [
        "hello there",
        "SELECT * FROM sales s",
        "SELECT * FROM sales AS s",
        "SELECT region AS r",
        "SELECT * FROM sales ORDER BY revenue DESC",
        "SELECT * FROM sales WHERE region = 'n",
        "SELECT * FROM sales WHERE region = 'north'",
        "SELECT * FROM sales LIMIT 5",
        "SELECT 1 ",
        "SELECT * FROM sales -- re",
        "SELECT \"re",
    ] {
        assert!(offered(line).is_empty(), "{line}: {:?}", offered(line));
    }
}

/// The name typed in full comes first, and the word's span is in
/// characters, so a multi-byte name before the cursor shifts nothing.
#[test]
fn a_name_typed_in_full_comes_first_and_the_span_counts_characters() {
    assert_eq!(offered("SELECT * FROM sales ORDER BY region"), ["region"]);
    let schema = schema();
    let (line, cursor) = split("SELECT 'é', re| FROM sales");
    let completion = SqlCompletion::for_sql(&line, cursor, &schema, &SqlVocabulary::none())
        .unwrap_or_else(|| fail("no completion"));
    assert_eq!((completion.start, completion.end), (cursor - 2, cursor));
    assert_eq!(
        completion.items.first().map(|i| i.word.as_str()),
        Some("region")
    );
}

/// With `DuckDB`'s vocabulary, an expression offers keywords and functions
/// after the names, once a letter is typed, at most twelve of each; a slot
/// for a table name does not, and a blank slot does not.
#[test]
fn the_vocabulary_follows_the_names_in_an_expression() {
    let words = SqlVocabulary::duckdb().unwrap_or_else(|e| fail(&e.to_string()));
    let (line, cursor) = split("SELECT * FROM sales WHERE re|");
    let items = SqlCompletion::for_sql(&line, cursor, &schema(), &words)
        .unwrap_or_else(|| fail("no completion"))
        .items;
    let kinds: Vec<(&str, &SqlItemKind)> =
        items.iter().map(|i| (i.word.as_str(), &i.kind)).collect();
    assert_eq!(kinds.first(), Some(&("region", &SqlItemKind::Column)));
    assert_eq!(kinds.get(1), Some(&("revenue", &SqlItemKind::Column)));
    let of = |kind: SqlItemKind| items.iter().filter(|i| i.kind == kind).count();
    assert_eq!(
        (of(SqlItemKind::Keyword), of(SqlItemKind::Function)),
        (12, 12),
        "{kinds:?}"
    );
    let has =
        |word: &str, kind: SqlItemKind| items.iter().any(|i| i.word == word && i.kind == kind);
    assert!(has("REAL", SqlItemKind::Keyword), "{kinds:?}");
    assert!(has("reduce", SqlItemKind::Function), "{kinds:?}");
    let (line, cursor) = split("SELECT * FROM sales WHERE regexp_m|");
    let narrowed = SqlCompletion::for_sql(&line, cursor, &schema(), &words)
        .unwrap_or_else(|| fail("no completion"))
        .items;
    assert!(
        narrowed
            .iter()
            .any(|i| i.word == "regexp_matches" && i.kind == SqlItemKind::Function),
        "{narrowed:?}"
    );
    let (line, cursor) = split("SELECT * FROM re|");
    assert!(SqlCompletion::for_sql(&line, cursor, &schema(), &words).is_none());
    let (line, cursor) = split("SELECT * FROM sales WHERE |");
    let blank = SqlCompletion::for_sql(&line, cursor, &schema(), &words)
        .unwrap_or_else(|| fail("no completion"));
    assert!(blank.items.iter().all(|i| i.kind != SqlItemKind::Keyword));
    // A whole keyword with no name of that spelling is the keyword: nothing.
    let (line, cursor) = split("SELECT * FROM sales WHERE region IS|");
    assert!(SqlCompletion::for_sql(&line, cursor, &schema(), &words).is_none());
}
