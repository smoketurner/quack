use super::*;

#[test]
fn summary_nests_subclasses_under_parents() {
    let mut ontology = Ontology::builtin_default();
    ontology.classes.push(Class {
        id: ClassId::from("vendor"),
        parent: ClassId::from("organization"),
        label: None,
        description: None,
        key: None,
        properties: Vec::new(),
    });
    let text = ontology.render_summary();
    assert!(
        text.contains("  - organization {industry, country}\n    - vendor\n"),
        "{text}"
    );
    assert!(text.contains("  - works_at: person -> organization"));
    assert!(text.contains("  - date: date"));
}

#[test]
fn versions_count_from_one() {
    assert_eq!(OntologyVersion::new(0), None);
    assert_eq!(OntologyVersion::after(None), OntologyVersion::FIRST);
    let second = OntologyVersion::after(Some(OntologyVersion::FIRST));
    assert_eq!(second.get(), 2);
    assert_eq!(second.previous(), Some(OntologyVersion::FIRST));
    assert_eq!(OntologyVersion::FIRST.previous(), None);
    assert_eq!(
        " 7 "
            .parse::<OntologyVersion>()
            .map(OntologyVersion::get)
            .ok(),
        Some(7)
    );
    for bad in ["0", "-1", "v2", ""] {
        assert!(bad.parse::<OntologyVersion>().is_err(), "{bad}");
    }
}

#[test]
fn an_unsaved_version_is_absent_in_json_and_read_from_zero_or_null() {
    let json = Ontology::builtin_default().to_json().unwrap_or_default();
    assert!(!json.contains("\"version\""), "{json}");
    for version in ["0", "null"] {
        let text = format!(r#"{{"version": {version}}}"#);
        assert_eq!(
            Ontology::from_json(&text).map(|o| o.version).ok(),
            Some(None),
            "{text}"
        );
    }
    let saved = Ontology::from_json(r#"{"version": 3}"#)
        .map(|o| o.version)
        .ok();
    assert_eq!(saved, Some(OntologyVersion::new(3)));
    assert!(Ontology::from_json(r#"{"version": -1}"#).is_err());
}

#[test]
fn snake_ids_are_checked_not_repaired() {
    assert_eq!(
        SnakeId::try_from("ship_mode_2").map(SnakeId::into_string),
        Ok(String::from("ship_mode_2"))
    );
    for bad in ["", "Ship", "2024", "_x", "ship mode", "ship-mode"] {
        let refused = SnakeId::try_from(bad).map_err(|e| e.for_item(ItemKind::Class).to_string());
        assert_eq!(
            refused,
            Err(format!(
                "ontology error: class id '{bad}' must be snake_case: a lowercase letter, then \
                     lowercase letters, digits, or underscores"
            )),
            "{bad}"
        );
    }
}

fn err_of(json: &str) -> String {
    match Ontology::from_json(json) {
        Ok(_) => String::from("<ok>"),
        Err(e) => e.to_string(),
    }
}

const INSURANCE: &str = r#"{
  "classes": [
    { "id": "organization", "properties": ["country"] },
    { "id": "vendor", "parent": "organization" },
    { "id": "policy", "key": "policy_number", "properties": ["policy_number", "effective_date"] },
    { "id": "claim", "key": "claim_id", "properties": ["claim_id", "amount", "status"] }
  ],
  "relations": [
    { "id": "issued_by", "domain": "policy", "range": "organization" },
    { "id": "filed_against", "domain": "claim", "range": "policy" }
  ],
  "properties": [
    { "id": "country", "type": "string" },
    { "id": "policy_number", "type": "string" },
    { "id": "effective_date", "type": "date" },
    { "id": "claim_id", "type": "string" },
    { "id": "amount", "type": "number" },
    { "id": "status", "type": "enum", "values": ["filed", "paid", "denied"] }
  ],
  "mappings": [
    {
      "table": "claims", "class": "claim", "key": "claim_id",
      "properties": { "amount": "amount", "status": "status" },
      "relations": [
        { "relation": "filed_against", "column": "policy_id", "target_class": "policy", "target_key": "policy_number" }
      ]
    }
  ]
}"#;

#[test]
fn the_design_example_parses_and_round_trips_through_json() {
    let ontology = Ontology::from_json(INSURANCE).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(ontology.classes.len(), 4);
    assert!(ontology.is_subclass_of("vendor", "organization"));
    assert!(ontology.is_subclass_of("vendor", ROOT_CLASS));
    assert!(!ontology.is_subclass_of("policy", "organization"));
    assert!(ontology.class_properties("vendor").contains("country"));
    let json = ontology.to_json().unwrap_or_else(|e| fail(&e.to_string()));
    let again = Ontology::from_json(&json).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(again, ontology);
}

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

#[test]
fn validation_names_each_violation() {
    let c = |body: &str| format!("{{\"classes\": [{body}]}}");
    assert!(err_of(&c(r#"{"id": "Bad-Id"}"#)).contains("snake_case"));
    assert!(err_of(&c(r#"{"id": "entity"}"#)).contains("implicit root"));
    assert!(err_of(&c(r#"{"id": "a"}, {"id": "a"}"#)).contains("declared twice"));
    assert!(err_of(&c(r#"{"id": "a", "parent": "ghost"}"#)).contains("unknown parent"));
    assert!(
        err_of(&c(
            r#"{"id": "a", "parent": "b"}, {"id": "b", "parent": "a"}"#
        ))
        .contains("cycle")
    );
    assert!(err_of(&c(r#"{"id": "a", "properties": ["nope"]}"#)).contains("unknown property"));
    assert!(err_of(r#"{"classes": [{"id": "a", "key": "k"}], "properties": [{"id": "k", "type": "string"}]}"#).contains("not one of its properties"));
    assert!(err_of(r#"{"properties": [{"id": "s", "type": "enum"}]}"#).contains("needs values"));
    assert!(
        err_of(r#"{"properties": [{"id": "s", "type": "string", "values": ["x"]}]}"#)
            .contains("cannot list values")
    );
    assert!(
        err_of(r#"{"relations": [{"id": "mentions", "domain": "entity", "range": "entity"}]}"#)
            .contains("implicit")
    );
    assert!(
        err_of(r#"{"relations": [{"id": "r", "domain": "nope", "range": "entity"}]}"#)
            .contains("unknown domain")
    );
    assert!(
        err_of(
            r#"{"classes": [{"id": "a"}], "mappings": [{"table": "t", "class": "a", "key": ""}]}"#
        )
        .contains("needs a table and a key")
    );
    assert!(err_of(r#"{"classes": [{"id": "a"}], "mappings": [{"table": "t", "class": "ghost", "key": "id"}]}"#).contains("unknown class"));
    assert!(err_of(r#"{"classes": [{"id": "a"}], "mappings": [{"table": "t", "class": "a", "key": "id", "properties": {"c": "p"}}]}"#).contains("does not carry"));
    let wrong_direction = r#"{"classes": [{"id": "a"}, {"id": "b", "key": "k", "properties": ["k"]}], "properties": [{"id": "k", "type": "string"}], "relations": [{"id": "r", "domain": "b", "range": "a"}], "mappings": [{"table": "t", "class": "a", "key": "id", "relations": [{"relation": "r", "column": "c", "target_class": "b", "target_key": "k"}]}]}"#;
    assert!(err_of(wrong_direction).contains("goes b -> a"));
    assert!(err_of("{\"classes\": [").contains("does not parse"));
    assert!(err_of(&c(r#"{"id": "a", "colour": "red"}"#)).contains("does not parse"));
}

#[test]
fn the_builtin_default_is_valid_and_renders_for_the_prompt() {
    let default = Ontology::builtin_default();
    assert!(default.validate().is_ok());
    let text = default.render_for_prompt();
    assert!(text.contains("person: entity {email, title}"), "{text}");
    assert!(text.contains("works_at: person -> organization"));
    assert!(text.contains("mentions: entity -> entity"));
    assert!(!text.contains("mapped tables"));
}

#[test]
fn the_capped_rendering_counts_what_it_leaves_out() {
    let default = Ontology::builtin_default();
    let capped = default.render_capped(2);
    assert!(capped.contains("person: entity"), "{capped}");
    assert!(!capped.contains("concept: entity"), "{capped}");
    assert!(
        capped.contains("and 5 more classes; describe_class shows any class by id"),
        "{capped}"
    );
    assert!(capped.contains("and 3 more relations"), "{capped}");
    // `mentions` is implicit and always named, cap or no cap.
    assert!(capped.contains("mentions: entity -> entity"), "{capped}");
    // Uncapped, nothing is counted away.
    let full = default.render_capped(usize::MAX);
    assert_eq!(full, default.render_for_prompt());
    assert!(!full.contains("more classes"), "{full}");
}

#[test]
fn relations_and_subclasses_follow_inheritance() {
    let ontology = Ontology::builtin_default();
    let ClassRelations { from, to } = ontology.relations_of("person");
    let from: Vec<&str> = from.iter().map(|r| r.id.as_str()).collect();
    let to: Vec<&str> = to.iter().map(|r| r.id.as_str()).collect();
    // `works_at` is the class's own; the `entity`-domain ones are inherited.
    assert!(
        from.contains(&"works_at") && from.contains(&"located_in"),
        "{from:?}"
    );
    assert!(!from.contains(&"produced_by"), "{from:?}");
    assert!(to.contains(&"part_of"), "{to:?}");
    assert_eq!(ontology.subclasses("entity").len(), ontology.classes.len());
    assert!(ontology.subclasses("person").is_empty());
    assert!(ontology.mapping_for("person").is_none());
}

#[test]
fn diff_reports_added_removed_and_changed_ids() {
    let base = Ontology::from_json(INSURANCE).unwrap_or_else(|e| fail(&e.to_string()));
    let mut next = base.clone();
    next.version = OntologyVersion::new(2);
    next.classes.retain(|c| c.id != "vendor");
    next.classes.push(Class {
        id: ClassId::from("adjuster"),
        parent: ClassId::from("organization"),
        label: None,
        description: None,
        key: None,
        properties: Vec::new(),
    });
    if let Some(status) = next.properties.iter_mut().find(|p| p.id == "status") {
        status.values.push(String::from("under_review"));
    }
    let diff = next.diff(&base);
    assert_eq!(diff.classes.added, ["adjuster"]);
    assert_eq!(diff.classes.removed, ["vendor"]);
    assert_eq!(diff.properties.changed, ["status"]);
    assert!(diff.relations.is_empty() && diff.mappings.is_empty());
    let text = diff.to_string();
    assert!(
        text.contains("+ adjuster") && text.contains("- vendor") && text.contains("~ status"),
        "{text}"
    );
    assert!(base.diff(&base).is_empty());
}

/// `docs/ontology.schema.json` is what `Ontology::json_schema` generates;
/// regenerate it with `quack ontology schema > docs/ontology.schema.json`.
#[test]
#[expect(clippy::unwrap_used, reason = "test reads committed files")]
fn the_committed_schema_is_the_generated_one() {
    let docs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs");
    let committed: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(docs.join("ontology.schema.json")).unwrap())
            .unwrap();
    assert_eq!(
        committed,
        serde_json::to_value(Ontology::json_schema()).unwrap(),
        "docs/ontology.schema.json is out of date: run `quack ontology schema > docs/ontology.schema.json`"
    );
}

/// The schema describes what the parser takes: the strict, renamed, and
/// defaulted fields show in it, and the example pack's ontology parses.
#[test]
#[expect(clippy::unwrap_used, reason = "test reads committed files")]
fn the_schema_matches_what_import_accepts() {
    let schema = serde_json::to_value(Ontology::json_schema()).unwrap();
    assert_eq!(
        schema.get("additionalProperties"),
        Some(&serde_json::json!(false))
    );
    let definitions = schema.get("$defs").unwrap();
    let required = definitions
        .pointer("/Property/required")
        .and_then(serde_json::Value::as_array)
        .unwrap();
    assert!(
        required.contains(&serde_json::json!("type")),
        "{required:?}"
    );
    assert_eq!(
        definitions.pointer("/Class/properties/parent/default"),
        Some(&serde_json::json!("entity"))
    );

    let example = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/storms/ontology.json");
    let parsed = Ontology::from_json(&std::fs::read_to_string(example).unwrap());
    assert!(parsed.is_ok(), "{parsed:?}");
}
