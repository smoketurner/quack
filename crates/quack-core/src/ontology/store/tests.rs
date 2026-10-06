use super::*;
use crate::embedding::Dimension;
use crate::ids::ClassId;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

fn db() -> WorkspaceDb {
    WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()))
}

#[test]
fn save_current_versions_and_restore_round_trip() {
    let db = db();
    assert!(current(&db).is_ok_and(|o| o.is_none()));
    assert!(latest_version(&db).is_ok_and(|v| v.is_none()));
    let v1 = save(
        &db,
        &Ontology::builtin_default(),
        Revision::reviewed(Some("alice"), Some("default")),
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(v1.version, OntologyVersion::new(1));
    let live = current(&db)
        .unwrap_or_else(|e| fail(&e.to_string()))
        .unwrap_or_else(|| fail("no ontology"));
    assert_eq!(live.classes.len(), 7);
    assert_eq!(live.relations.len(), 5);
    assert_eq!(live.properties.len(), 5);
    assert!(
        live.class("person").is_some_and(
            |c| c.properties == ["title", "email"] || c.properties == ["email", "title"]
        )
    );

    let mut edited = live;
    edited.classes.push(Class {
        id: ClassId::from("vendor"),
        parent: ClassId::from("organization"),
        label: Some(String::from("Vendor")),
        description: None,
        key: None,
        properties: Vec::new(),
    });
    edited.classes.retain(|c| c.id != "concept");
    let v2 =
        save(&db, &edited, Revision::reviewed(None, None)).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(v2.version, OntologyVersion::new(2));
    let since: Vec<(String, i64)> = {
        let mut stmt = db.connection().prepare("SELECT id, since_version FROM _quack_ontology_classes WHERE id IN ('person', 'vendor') ORDER BY id").unwrap_or_else(|e| fail(&e.to_string()));
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
            .unwrap_or_else(|e| fail(&e.to_string()));
        rows.flatten().collect()
    };
    assert_eq!(
        since,
        [(String::from("person"), 1), (String::from("vendor"), 2)]
    );

    let headers = versions(&db, 10).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        headers.iter().map(|h| h.version.get()).collect::<Vec<_>>(),
        [2, 1]
    );
    assert_eq!(
        headers.last().and_then(|h| h.author.clone()).as_deref(),
        Some("alice")
    );
    let old = version(&db, OntologyVersion::FIRST)
        .unwrap_or_else(|e| fail(&e.to_string()))
        .unwrap_or_else(|| fail("no v1"));
    assert!(old.class("concept").is_some() && old.class("vendor").is_none());
    let diff = v2.diff(&old);
    assert_eq!(diff.classes.added, ["vendor"]);
    assert_eq!(diff.classes.removed, ["concept"]);

    let restored =
        restore(&db, OntologyVersion::FIRST, Some("bob")).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(restored.version, OntologyVersion::new(3));
    assert!(restored.class("concept").is_some() && restored.class("vendor").is_none());
    let nine = OntologyVersion::new(9).unwrap_or(OntologyVersion::FIRST);
    assert!(version(&db, nine).is_ok_and(|v| v.is_none()));
    assert!(restore(&db, nine, None).is_err());
}

#[test]
fn a_saved_ontology_reloads_equal_and_diffs_empty() {
    let db = db();
    let saved = save(
        &db,
        &Ontology::builtin_default(),
        Revision::reviewed(None, None),
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    let live = current(&db)
        .unwrap_or_else(|e| fail(&e.to_string()))
        .unwrap_or_else(|| fail("no ontology"));
    assert_eq!(live, saved);
    assert!(live.diff(&Ontology::builtin_default()).is_empty());
    let again =
        save(&db, &live, Revision::reviewed(None, None)).unwrap_or_else(|e| fail(&e.to_string()));
    assert!(again.diff(&live).is_empty());
    let snapshot = version(&db, OntologyVersion::FIRST)
        .unwrap_or_else(|e| fail(&e.to_string()))
        .unwrap_or_else(|| fail("no v1"));
    assert_eq!(snapshot, live);
}

#[test]
fn mappings_must_name_real_tables_and_columns() {
    let db = db();
    assert!(
        db.execute_statement("CREATE TABLE claims (claim_id TEXT, amount DOUBLE, policy_id TEXT)")
            .is_ok()
    );
    let json = r#"{"classes": [{"id": "claim", "key": "claim_id", "properties": ["claim_id", "amount"]}, {"id": "policy", "key": "policy_number", "properties": ["policy_number"]}], "relations": [{"id": "filed_against", "domain": "claim", "range": "policy"}], "properties": [{"id": "claim_id", "type": "string"}, {"id": "amount", "type": "number"}, {"id": "policy_number", "type": "string"}], "mappings": [{"table": "claims", "class": "claim", "key": "claim_id", "properties": {"amount": "amount"}, "relations": [{"relation": "filed_against", "column": "policy_id", "target_class": "policy", "target_key": "policy_number"}]}]}"#;
    let ontology = Ontology::from_json(json).unwrap_or_else(|e| fail(&e.to_string()));
    let saved = save(&db, &ontology, Revision::reviewed(None, None))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let live = current(&db)
        .unwrap_or_else(|e| fail(&e.to_string()))
        .unwrap_or_else(|| fail("no ontology"));
    assert_eq!(live.mappings, saved.mappings);
    assert_eq!(live.mappings.first().map(|m| m.relations.len()), Some(1));

    // A mapping to a table the workspace no longer has (a deleted
    // document) is kept and flagged by graph status, not refused.
    let mut gone_table = ontology.clone();
    gone_table
        .mappings
        .iter_mut()
        .for_each(|m| m.table = String::from("nope"));
    assert!(save(&db, &gone_table, Revision::reviewed(None, None)).is_ok());
    let mut wrong_column = ontology;
    wrong_column
        .mappings
        .iter_mut()
        .for_each(|m| m.key = String::from("ghost"));
    let wrong = save(&db, &wrong_column, Revision::reviewed(None, None)).err();
    assert!(wrong.is_some_and(|e| e.to_string().contains("column 'ghost'")));
    assert_eq!(
        latest_version(&db).ok().flatten(),
        OntologyVersion::new(2),
        "failed saves write nothing"
    );
}

// A property that lives on more than one class has one row per
// `(class, property)` membership in `_quack_ontology_properties`
// (`PRIMARY KEY (id, class_id)`), and `since_version` is the version
// it first appeared *on that class*. The carry-over must look it up
// per membership, not collapse to a global minimum across all classes
// that share the property id.

#[test]
fn property_since_version_is_per_class_not_global_min() {
    let db = db();
    let mut v1 = Ontology::builtin_default();
    if let Some(c) = v1.classes.iter_mut().find(|c| c.id == "place") {
        c.properties.retain(|p| p != "country");
    }
    save(&db, &v1, Revision::reviewed(None, None)).unwrap_or_else(|e| fail(&e.to_string()));
    let v2 = Ontology::builtin_default();
    save(&db, &v2, Revision::reviewed(None, None)).unwrap_or_else(|e| fail(&e.to_string()));
    let since_for = |class_id: &str| -> i64 {
        let mut stmt = db
            .connection()
            .prepare(
                "SELECT since_version FROM _quack_ontology_properties \
                 WHERE id = 'country' AND class_id = ?",
            )
            .unwrap_or_else(|e| fail(&e.to_string()));
        stmt.query_row(duckdb::params![class_id], |r| r.get::<_, i64>(0))
            .unwrap_or_else(|e| fail(&e.to_string()))
    };
    assert_eq!(since_for("place"), 2, "country on place is new in v2");
    // `country` on `organization` landed at v1; a global-min read would
    // make this v2 row drift to 1 on any later save whose `previous`
    // already contains the membership.
    assert_eq!(
        since_for("organization"),
        1,
        "country on organization is from v1"
    );
    save(&db, &v2, Revision::reviewed(None, None)).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        since_for("place"),
        2,
        "an unchanged item must keep the version it first appeared in"
    );
    assert_eq!(
        since_for("organization"),
        1,
        "the original membership is untouched by the re-save"
    );
}

#[test]
fn property_since_version_survives_restore() {
    let db = db();
    let mut v1 = Ontology::builtin_default();
    if let Some(c) = v1.classes.iter_mut().find(|c| c.id == "place") {
        c.properties.retain(|p| p != "country");
    }
    save(&db, &v1, Revision::reviewed(None, None)).unwrap_or_else(|e| fail(&e.to_string()));
    let v2 = Ontology::builtin_default();
    let saved =
        save(&db, &v2, Revision::reviewed(None, None)).unwrap_or_else(|e| fail(&e.to_string()));
    let v2_version = saved
        .version
        .unwrap_or_else(|| fail("saved ontology has no version"));
    // restore v2 right after saving v2: previous current is v2 (which has
    // place+country), so the membership `existed` and is carried over.
    // Its true first-appearance on `place` is v2; the global-min read
    // would instead write 1.
    restore(&db, v2_version, None).unwrap_or_else(|e| fail(&e.to_string()));
    let mut stmt = db
        .connection()
        .prepare(
            "SELECT since_version FROM _quack_ontology_properties \
             WHERE id = 'country' AND class_id = 'place'",
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
    let since: i64 = stmt
        .query_row([], |r| r.get(0))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        since, 2,
        "carried-over membership must keep its true first-appearance (v2), not the global min (v1)"
    );
}

#[test]
fn property_meanings_and_measures_round_trip_and_version() {
    let db = db();
    assert!(
        db.execute_statement("CREATE TABLE orders (order_id TEXT, amount BIGINT)")
            .is_ok()
    );
    let json = r#"{"classes": [{"id": "order", "key": "order_id", "properties": ["order_id", "amount"]}],
        "properties": [{"id": "order_id", "type": "string"},
                       {"id": "amount", "type": "number", "description": "order total", "unit": "cents", "synonyms": ["total", "value", "total"]}],
        "mappings": [{"table": "orders", "class": "order", "key": "order_id", "properties": {"amount": "amount"}}],
        "measures": [{"id": "revenue", "description": "in dollars", "table": "orders", "expression": "sum(amount) / 100.0"}]}"#;
    let ontology = Ontology::from_json(json).unwrap_or_else(|e| fail(&e.to_string()));
    let saved = save(&db, &ontology, Revision::reviewed(None, None))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let live = current(&db)
        .unwrap_or_else(|e| fail(&e.to_string()))
        .unwrap_or_else(|| fail("no ontology"));
    assert_eq!(live, saved);
    let amount = live.property("amount").unwrap_or_else(|| fail("no amount"));
    assert_eq!(amount.unit.as_deref(), Some("cents"));
    assert_eq!(
        amount.synonyms,
        ["total", "value"],
        "synonyms are sorted and deduplicated"
    );
    assert_eq!(live.measures_on("orders").len(), 1);

    let described = db
        .describe_table("orders")
        .unwrap_or_else(|e| fail(&e.to_string()));
    let column = described
        .columns
        .iter()
        .find(|c| c.name == "amount")
        .unwrap_or_else(|| fail("no amount column"));
    assert_eq!(
        column.meaning.as_ref().map(ToString::to_string).as_deref(),
        Some(": order total [cents] (also: total, value)")
    );
    assert_eq!(described.measures.len(), 1);

    let mut edited = live;
    edited.measures.clear();
    let v2 =
        save(&db, &edited, Revision::reviewed(None, None)).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(v2.diff(&saved).measures.removed, ["revenue"]);
}

#[test]
fn measures_must_be_one_read_of_their_table() {
    let db = db();
    assert!(
        db.execute_statement("CREATE TABLE orders (amount BIGINT)")
            .is_ok()
    );
    let with = |expression: &str| {
        let json = serde_json::json!({
            "measures": [{"id": "m", "table": "orders", "expression": expression}]
        });
        Ontology::from_json(&json.to_string())
            .and_then(|o| save(&db, &o, Revision::reviewed(None, None)))
    };
    assert!(with("sum(amount)").is_ok());
    assert!(with("sum(ghost)").is_err_and(|e| e.to_string().contains("not a read")));
    assert!(with("1; DROP TABLE orders; SELECT 1").is_err());
    assert!(with("(SELECT count(*) FROM _quack_meta)").is_err());
    assert!(with("sum(").is_err());
    assert!(
        db.list_tables()
            .is_ok_and(|t| t.contains(&String::from("orders")))
    );
    let unknown_table = serde_json::json!({
        "measures": [{"id": "m", "table": "gone", "expression": "count(*)"}]
    });
    assert!(
        Ontology::from_json(&unknown_table.to_string())
            .and_then(|o| save(&db, &o, Revision::reviewed(None, None)))
            .is_ok(),
        "a measure over a missing table is kept, like a mapping"
    );
    let bad_id = serde_json::json!({
        "measures": [{"id": "Bad Id", "table": "orders", "expression": "count(*)"}]
    });
    assert!(Ontology::from_json(&bad_id.to_string()).is_err());
}
