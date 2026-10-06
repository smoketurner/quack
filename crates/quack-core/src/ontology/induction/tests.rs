use super::*;
use crate::embedding::Dimension;
use crate::ontology::OntologyVersion;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

fn db() -> WorkspaceDb {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    for sql in [
        "CREATE TABLE policies (policy_number TEXT, holder TEXT, effective DATE, premium DOUBLE)",
        "CREATE TABLE claims (claim_id INTEGER, policy_number TEXT, amount DOUBLE, status TEXT, filed TEXT, active BOOLEAN)",
    ] {
        assert!(db.execute_statement(sql).is_ok());
    }
    for i in 0..30 {
        assert!(
            db.execute_statement(&format!(
                "INSERT INTO policies VALUES ('P{i}', 'Holder {i}', DATE '2024-01-01', {i}.5)"
            ))
            .is_ok()
        );
        let status = ["filed", "paid", "denied"]
            .get(i % 3)
            .copied()
            .unwrap_or("filed");
        assert!(
            db.execute_statement(&format!(
                "INSERT INTO claims VALUES ({i}, 'P{}', {i}.0, '{status}', '2024-02-{:02}', {})",
                i % 25,
                (i % 28).saturating_add(1),
                i % 2 == 0
            ))
            .is_ok()
        );
    }
    db
}

#[test]
fn column_types_sort_into_kinds() {
    for (duckdb_type, kind) in [
        ("BOOLEAN", ColumnKind::Boolean),
        ("date", ColumnKind::Temporal),
        ("TIMESTAMP WITH TIME ZONE", ColumnKind::Temporal),
        ("BIGINT", ColumnKind::Numeric),
        ("DECIMAL(18,3)", ColumnKind::Numeric),
        ("double", ColumnKind::Numeric),
        ("VARCHAR", ColumnKind::Other),
        ("INTEGER[]", ColumnKind::Other),
    ] {
        assert_eq!(ColumnKind::of(duckdb_type), kind, "{duckdb_type}");
    }
}

#[test]
fn names_are_snake_case_singular_and_prefixed() {
    assert_eq!(SnakeId::from_name("Ship Mode").as_str(), "ship_mode");
    assert_eq!(SnakeId::from_name("po / so #").as_str(), "po_so");
    assert_eq!(SnakeId::from_name("2024").as_str(), "t_2024");
    assert_eq!(SnakeId::singular_from("shipments").as_str(), "shipment");
    assert_eq!(SnakeId::singular_from("policies").as_str(), "policy");
    assert_eq!(SnakeId::singular_from("address").as_str(), "address");
    assert_eq!(SnakeId::singular_from("bus").as_str(), "bus");
    assert_eq!(relation_id_for_column("policy_id"), "has_policy");
    assert_eq!(relation_id_for_column("policy_number"), "has_policy_number");
}

#[test]
fn shared_foreign_key_column_names_get_distinct_relations() {
    let db = db();
    assert!(
        db.execute_statement(
            "CREATE TABLE notes (note_id INTEGER, policy_number VARCHAR, body VARCHAR)"
        )
        .is_ok()
    );
    for i in 0..40_u32 {
        assert!(
            db.execute_statement(&format!(
                "INSERT INTO notes VALUES ({i}, 'P{}', 'note {i}')",
                i % 25
            ))
            .is_ok()
        );
    }
    let candidates = propose_from_tables(&db, None, &TableEvidenceOptions::default())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let relations: Vec<&Relation> = candidates
        .iter()
        .filter_map(|c| match &c.proposal {
            Proposal::Relation(r) => Some(r),
            Proposal::Class(_) | Proposal::Property { .. } | Proposal::Mapping(_) => None,
        })
        .collect();
    assert_eq!(relations.len(), 2, "one relation per source table");
    assert!(
        relations
            .iter()
            .any(|r| r.id == "has_policy_number" && r.domain == "claim")
    );
    assert!(
        relations
            .iter()
            .any(|r| r.id == "note_has_policy_number" && r.domain == "note")
    );
    let accepted: Vec<(Proposal, Decision)> = candidates
        .iter()
        .map(|c| (c.proposal.clone(), Decision::Accept))
        .collect();
    let ontology = apply(None, &accepted).unwrap_or_else(|e| fail(&e.to_string()));
    assert!(ontology.validate().is_ok());
}

#[test]
fn tables_propose_classes_keys_typed_properties_relations_and_mappings() {
    let db = db();
    let candidates = propose_from_tables(&db, None, &TableEvidenceOptions::default())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let find = |kind: &str, id: &str| {
        candidates
            .iter()
            .find(|c| c.proposal.kind().as_str() == kind && c.proposal.id() == id)
    };
    let claim = find("class", "claim").unwrap_or_else(|| fail("no claim class"));
    assert!(
        matches!(&claim.proposal, Proposal::Class(c) if c.key.as_deref() == Some("claim_id") && c.properties.len() == 6)
    );
    let policy = find("class", "policy").unwrap_or_else(|| fail("no policy class"));
    assert!(
        matches!(&policy.proposal, Proposal::Class(c) if c.key.as_deref() == Some("policy_number"))
    );
    let status = find("property", "status").unwrap_or_else(|| fail("no status"));
    assert!(
        matches!(&status.proposal, Proposal::Property { property, .. } if property.kind == PropertyType::Enum && property.values == ["denied", "filed", "paid"])
    );
    let filed = find("property", "filed").unwrap_or_else(|| fail("no filed"));
    assert!(
        matches!(&filed.proposal, Proposal::Property { property, .. } if property.kind == PropertyType::Date),
        "text dates are dates"
    );
    assert!(
        matches!(&find("property", "amount").map(|c| &c.proposal), Some(Proposal::Property { property, .. }) if property.kind == PropertyType::Number)
    );
    assert!(
        matches!(&find("property", "active").map(|c| &c.proposal), Some(Proposal::Property { property, .. }) if property.kind == PropertyType::Boolean)
    );
    let relation = find("relation", "has_policy_number").unwrap_or_else(|| fail("no relation"));
    assert!(
        matches!(&relation.proposal, Proposal::Relation(r) if r.domain == "claim" && r.range == "policy")
    );
    assert!(relation.confidence >= 0.8);
    let mapping = find("mapping", "claims").unwrap_or_else(|| fail("no mapping"));
    assert!(
        matches!(&mapping.proposal, Proposal::Mapping(m) if m.key == "claim_id" && m.relations.len() == 1 && m.properties.len() == 6)
    );

    // Accepting everything yields a valid ontology with the mappings.
    let accepted: Vec<(Proposal, Decision)> = candidates
        .iter()
        .map(|c| (c.proposal.clone(), Decision::Accept))
        .collect();
    let ontology = apply(None, &accepted).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(ontology.classes.len(), 2);
    assert_eq!(ontology.mappings.len(), 2);
    assert!(ontology.class_properties("claim").contains("status"));
    // Extend mode proposes nothing new once everything is in.
    let again = propose_from_tables(&db, Some(&ontology), &TableEvidenceOptions::default())
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(again.is_empty(), "{}", again.len());
    // A table mapped to a class named differently from the table is
    // covered too: nothing is proposed for it (issue #55).
    let mut renamed = ontology;
    for class in &mut renamed.classes {
        if class.id == "claim" {
            class.id = ClassId::from("insurance_claim");
        }
    }
    for mapping in &mut renamed.mappings {
        if mapping.class == "claim" {
            mapping.class = ClassId::from("insurance_claim");
        }
    }
    for relation in &mut renamed.relations {
        if relation.domain == "claim" {
            relation.domain = ClassId::from("insurance_claim");
        }
        if relation.range == "claim" {
            relation.range = ClassId::from("insurance_claim");
        }
    }
    let again = propose_from_tables(&db, Some(&renamed), &TableEvidenceOptions::default())
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(again.is_empty(), "{again:?}");
}

#[test]
fn rename_merge_and_reparent_follow_through_references() {
    let db = db();
    let candidates = propose_from_tables(&db, None, &TableEvidenceOptions::default())
        .unwrap_or_else(|e| fail(&e.to_string()));
    let mut base = Ontology::builtin_default();
    base.version = Some(OntologyVersion::FIRST);
    let decisions: Vec<(Proposal, Decision)> = candidates
        .iter()
        .map(|c| {
            let decision = match (&c.proposal, c.proposal.id()) {
                (Proposal::Class(_), "claim") => Decision::Reparent(String::from("event")),
                (Proposal::Class(_), "policy") => {
                    Decision::Rename(String::from("insurance_policy"))
                }
                (Proposal::Relation(_), _) => Decision::Rename(String::from("filed_against")),
                (Proposal::Property { .. }, "holder") => Decision::MergeInto(String::from("title")),
                _ => Decision::Accept,
            };
            (c.proposal.clone(), decision)
        })
        .collect();
    let ontology = apply(Some(&base), &decisions).unwrap_or_else(|e| fail(&e.to_string()));
    assert!(ontology.class("insurance_policy").is_some() && ontology.class("policy").is_none());
    assert!(ontology.class("claim").is_some_and(|c| c.parent == "event"));
    let relation = ontology
        .relation("filed_against")
        .unwrap_or_else(|| fail("no relation"));
    assert_eq!(
        (relation.domain.as_str(), relation.range.as_str()),
        ("claim", "insurance_policy")
    );
    let mapping = ontology
        .mappings
        .iter()
        .find(|m| m.table == "claims")
        .unwrap_or_else(|| fail("no mapping"));
    assert_eq!(
        mapping
            .relations
            .first()
            .map(|r| (r.relation.as_str(), r.target_class.as_str())),
        Some(("filed_against", "insurance_policy"))
    );
    let policies = ontology
        .mappings
        .iter()
        .find(|m| m.table == "policies")
        .unwrap_or_else(|| fail("no mapping"));
    assert_eq!(
        policies.properties.get("holder").map(String::as_str),
        Some("title")
    );
    assert!(ontology.property("holder").is_none());
    assert!(
        ontology
            .class_properties("insurance_policy")
            .contains("title")
    );
    assert_eq!(ontology.classes.len(), 9);

    let bad = vec![(
        Proposal::Class(Class {
            id: ClassId::from("x"),
            parent: ClassId::from("ghost"),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        }),
        Decision::Accept,
    )];
    assert!(apply(None, &bad).is_err());
}
