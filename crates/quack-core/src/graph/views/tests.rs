#![expect(clippy::unwrap_used, reason = "test setup")]

use super::*;
use crate::config::Config;
use crate::embedding::Dimension;
use crate::error::Error;
use crate::ontology::store::{self, Revision};

fn db() -> WorkspaceDb {
    WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap()
}

const ONTOLOGY: &str = r#"{
  "classes": [
    {"id": "organization", "properties": ["country", "founded", "revenue", "public", "id"]},
    {"id": "supplier", "parent": "organization", "properties": ["tier"]}
  ],
  "relations": [{"id": "supplies", "domain": "supplier", "range": "organization"}],
  "properties": [
    {"id": "country", "type": "string"},
    {"id": "founded", "type": "date"},
    {"id": "revenue", "type": "number"},
    {"id": "public", "type": "boolean"},
    {"id": "id", "type": "string"},
    {"id": "tier", "type": "enum", "values": ["one", "two"]}
  ]
}"#;

fn save(db: &WorkspaceDb, json: &str) {
    store::save(
        db,
        &Ontology::from_json(json).unwrap(),
        Revision::reviewed(None, None),
    )
    .unwrap();
}

fn node(db: &WorkspaceDb, id: &str, label: &str, class: &str, properties: &str) {
    db.connection()
        .execute(
            "INSERT INTO _quack_graph_nodes (id, label, normalized_label, class_id, properties) \
             VALUES (?, ?, lower(?), ?, ?)",
            duckdb::params![id, label, label, class, properties],
        )
        .unwrap();
}

fn columns(db: &WorkspaceDb, view: &str) -> Vec<(String, String)> {
    db.describe_columns(view)
        .unwrap()
        .into_iter()
        .map(|c| (c.name, c.column_type))
        .collect()
}

#[test]
fn class_views_type_each_property_and_include_subclasses() {
    let db = db();
    save(&db, ONTOLOGY);
    node(
        &db,
        "n1",
        "Acme",
        "supplier",
        r#"{"country": "NL", "founded": "1999-03-01", "revenue": "12.5", "public": "true", "tier": "one", "id": "A-1"}"#,
    );
    node(
        &db,
        "n2",
        "Globex",
        "organization",
        r#"{"country": "US", "revenue": "oops"}"#,
    );

    let org = columns(&db, "graph_organization");
    let names: Vec<&str> = org.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "id",
            "label",
            "class_id",
            "provisional",
            "country",
            "founded",
            "id_property",
            "public",
            "revenue"
        ]
    );
    let typed = |name: &str| org.iter().find(|(n, _)| n == name).map(|(_, t)| t.clone());
    assert_eq!(typed("founded").as_deref(), Some("DATE"));
    assert_eq!(typed("revenue").as_deref(), Some("DOUBLE"));
    assert_eq!(typed("public").as_deref(), Some("BOOLEAN"));
    assert!(
        !org.iter()
            .any(|(n, _)| n == "embedding" || n == "normalized_label"),
        "internal columns never surface"
    );

    let rows = db
        .execute_query("SELECT label, revenue, id_property FROM graph_organization ORDER BY label")
        .unwrap();
    assert_eq!(rows.rows.len(), 2, "the subclass's rows are included");
    assert_eq!(
        rows.rows
            .first()
            .and_then(|r| r.get(1))
            .cloned()
            .unwrap_or_default(),
        serde_json::json!(12.5)
    );
    assert_eq!(
        rows.rows
            .first()
            .and_then(|r| r.get(2))
            .cloned()
            .unwrap_or_default(),
        serde_json::json!("A-1")
    );
    assert_eq!(
        rows.rows
            .get(1)
            .and_then(|r| r.get(1))
            .cloned()
            .unwrap_or_default(),
        serde_json::Value::Null,
        "a bad number is empty"
    );

    let supplier = columns(&db, "graph_supplier");
    assert!(
        supplier.iter().any(|(n, _)| n == "tier") && supplier.iter().any(|(n, _)| n == "country")
    );
}

#[test]
fn aggregating_a_view_is_a_read_and_changing_one_is_refused() {
    let db = db();
    save(&db, ONTOLOGY);
    node(&db, "n1", "Acme", "supplier", r#"{"country": "NL"}"#);
    assert_eq!(
        db.classify_user_statement("SELECT country, count(*) FROM graph_supplier GROUP BY 1")
            .unwrap(),
        StatementKind::Read
    );
    let refused = |sql: &str| matches!(db.classify_user_statement(sql), Err(Error::Analysis(m)) if m == RESERVED_REFUSED);
    assert!(refused("DROP VIEW graph_supplier"));
    assert!(refused("CREATE OR REPLACE VIEW graph_supplier AS SELECT 1"));
    assert!(refused("CREATE TABLE graph_x AS SELECT 1"));
    assert!(refused("ALTER VIEW graph_edges RENAME TO e"));
    assert!(
        db.classify_user_statement("CREATE VIEW v AS SELECT * FROM _quack_graph_nodes")
            .is_err_and(|e| e.to_string().contains("internal tables")),
        "a view over the base tables stays refused"
    );
    assert!(
        db.classify_user_statement("SELECT * FROM _quack_graph_nodes")
            .is_err(),
        "the base tables stay refused"
    );
    assert_eq!(
        db.classify_user_statement("DESCRIBE graph_supplier")
            .unwrap(),
        StatementKind::Read
    );
}

#[test]
fn edges_view_names_both_ends() {
    let db = db();
    save(&db, ONTOLOGY);
    node(&db, "n1", "Acme", "supplier", "{}");
    node(&db, "n2", "Globex", "organization", "{}");
    db.connection()
        .execute(
            "INSERT INTO _quack_graph_edges (id, source_node_id, target_node_id, relation_id, properties) \
             VALUES ('e1', 'n1', 'n2', 'supplies', '{\"since\": 2020}')",
            [],
        )
        .unwrap();
    let rows = db
        .execute_query(
            "SELECT source_label, relation_id, target_label, provisional FROM graph_edges",
        )
        .unwrap();
    assert_eq!(
        rows.rows,
        [vec![
            serde_json::json!("Acme"),
            serde_json::json!("supplies"),
            serde_json::json!("Globex"),
            serde_json::json!(false)
        ]]
    );
    let names: Vec<String> = columns(&db, EDGES_VIEW)
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(
        names,
        [
            "id",
            "source_id",
            "source_label",
            "relation_id",
            "target_id",
            "target_label",
            "provisional",
            "properties"
        ]
    );
}

#[test]
fn saving_adds_and_drops_views_and_spares_user_objects() {
    let db = db();
    db.execute_statement("CREATE VIEW graph_mine AS SELECT 1 AS x")
        .unwrap();
    save(&db, ONTOLOGY);
    let views = names(&db).unwrap();
    assert!(views.contains("graph_organization") && views.contains("graph_supplier"));
    assert!(views.contains(EDGES_VIEW) && !views.contains("graph_mine"));

    save(
        &db,
        r#"{"classes": [{"id": "organization", "properties": ["country"]}, {"id": "person"}],
            "properties": [{"id": "country", "type": "string"}]}"#,
    );
    let views = names(&db).unwrap();
    assert!(views.contains("graph_person"));
    assert!(
        !views.contains("graph_supplier"),
        "a removed class's view is dropped"
    );
    assert!(
        db.list_tables()
            .unwrap()
            .contains(&String::from("graph_mine")),
        "a view quack did not make is left alone"
    );
    assert_eq!(columns(&db, "graph_organization").len(), 5);
}

#[test]
fn a_user_table_holding_a_view_name_is_skipped() {
    let db = db();
    db.execute_statement("CREATE TABLE graph_person (x INTEGER)")
        .unwrap();
    save(&db, r#"{"classes": [{"id": "person"}, {"id": "place"}]}"#);
    let views = names(&db).unwrap();
    assert!(views.contains("graph_place") && !views.contains("graph_person"));
    assert_eq!(
        columns(&db, "graph_person"),
        [(String::from("x"), String::from("INTEGER"))]
    );
}

#[test]
fn views_survive_reopening_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config::default();
    config.general.data_dir = dir.path().to_path_buf();
    {
        let db = WorkspaceDb::open(&config, "ws").unwrap();
        save(&db, ONTOLOGY);
        node(&db, "n1", "Acme", "supplier", r#"{"country": "NL"}"#);
    }
    let db = WorkspaceDb::open(&config, "ws").unwrap();
    let rows = db
        .execute_query("SELECT country FROM graph_organization")
        .unwrap();
    assert_eq!(rows.rows, [vec![serde_json::json!("NL")]]);
    assert!(names(&db).unwrap().contains("graph_supplier"));
}
