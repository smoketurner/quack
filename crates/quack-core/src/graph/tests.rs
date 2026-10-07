use super::*;

#[test]
fn labels_normalize_case_and_whitespace() {
    assert_eq!(
        NormalizedLabel::new("  Acme   Corp \n").as_str(),
        "acme corp"
    );
    assert_eq!(NormalizedLabel::new("USAID").as_str(), "usaid");
    assert!(NormalizedLabel::new(" \t ").is_empty());
}

#[test]
fn properties_render_sorted_and_flattened() {
    let text = Properties::from(serde_json::json!({
        "status": "open",
        "amount": 1200.5,
        "aliases": ["Acme Ltd", "Acme"],
        "closed": null,
        "note": "  padded  ",
    }))
    .to_string();
    assert_eq!(
        text,
        "{aliases: Acme Ltd, Acme, amount: 1200.5, note: padded, status: open}"
    );
}

#[test]
fn properties_without_values_render_as_nothing() {
    for value in [
        serde_json::json!({}),
        serde_json::json!(null),
        serde_json::json!("not an object"),
        serde_json::json!({ "empty": "", "unset": null }),
    ] {
        assert_eq!(Properties::from(value).to_string(), "");
    }
}

#[test]
fn a_long_value_is_cut_and_extra_properties_are_counted() {
    let long = "x".repeat(PROPERTY_VALUE_CHARS + 10);
    let text = Properties::from(serde_json::json!({ "note": long })).to_string();
    assert!(text.ends_with("\u{2026}}"), "{text}");
    // "{note: " + the cut value + the ellipsis + "}"
    assert_eq!(text.chars().count(), PROPERTY_VALUE_CHARS + 9);

    let mut wide = serde_json::Map::new();
    for i in 0..(RENDERED_PROPERTIES + 3) {
        wide.insert(format!("p{i:02}"), serde_json::json!("v"));
    }
    let text = Properties::from(wide).to_string();
    assert!(text.contains("p00: v") && text.contains("p07: v"), "{text}");
    assert!(!text.contains("p08"), "{text}");
    assert!(text.ends_with("... 3 more}"), "{text}");
}

#[test]
fn properties_read_anything_and_keep_only_objects() {
    let parsed: Properties = serde_json::from_str("\"a string\"").unwrap_or_default();
    assert!(parsed.is_empty());
    let parsed: Properties = serde_json::from_str(r#"{"a": 1}"#).unwrap_or_default();
    assert_eq!(parsed.get("a"), Some(&serde_json::json!(1)));
    assert!(Properties::from_column(None).is_empty());
    assert!(Properties::from_column(Some("not json")).is_empty());
    assert_eq!(Properties::from_column(Some(r#"{"a": 1}"#)), parsed);
    assert_eq!(parsed.to_json(), r#"{"a":1}"#);
}

#[test]
fn filling_keeps_existing_values() {
    let mut kept = Properties::from(serde_json::json!({ "a": 1 }));
    kept.fill_from(&Properties::from(serde_json::json!({ "a": 2, "b": 3 })));
    assert_eq!(
        kept,
        Properties::from(serde_json::json!({ "a": 1, "b": 3 }))
    );
}

#[test]
fn origins_serialize_flat_and_round_trip() {
    let chunk = Provenance {
        subject_id: String::from("n"),
        origin: Origin::from_columns(ProvenanceColumns {
            document_id: Some(DocumentId::from("d")),
            chunk_id: ChunkId::from("c"),
            ..ProvenanceColumns::default()
        }),
        confidence: 0.5,
    };
    let row = Provenance {
        subject_id: String::from("n"),
        origin: Origin::from_columns(ProvenanceColumns {
            table_name: String::from("t"),
            row_key: String::from("k"),
            ..ProvenanceColumns::default()
        }),
        confidence: 1.0,
    };
    let manual = Provenance {
        subject_id: String::from("n"),
        origin: Origin::from_columns(ProvenanceColumns {
            author: Some(String::from("ada")),
            note: Some(String::from("checked the filing")),
            asserted_at: Some(String::from("2026-10-06 12:00:00")),
            ..ProvenanceColumns::default()
        }),
        confidence: 1.0,
    };
    assert_eq!(chunk.origin.chunk_id(), Some(&ChunkId::from("c")));
    assert_eq!(row.origin.chunk_id(), None);
    assert_eq!(manual.origin.chunk_id(), None);
    assert_eq!(
        manual.origin.assertion().as_deref(),
        Some("asserted by ada: checked the filing")
    );
    assert_eq!(chunk.origin.assertion(), None);
    assert_eq!(
        serde_json::to_value(&manual).ok(),
        Some(serde_json::json!({
            "subject_id": "n", "author": "ada", "note": "checked the filing",
            "asserted_at": "2026-10-06 12:00:00", "confidence": 1.0
        }))
    );
    assert_eq!(
        serde_json::to_value(&chunk).ok(),
        Some(serde_json::json!({
            "subject_id": "n", "document_id": "d", "chunk_id": "c", "confidence": 0.5
        }))
    );
    assert_eq!(
        serde_json::to_value(&row).ok(),
        Some(serde_json::json!({
            "subject_id": "n", "table_name": "t", "row_key": "k", "confidence": 1.0
        }))
    );
    for p in [chunk, row, manual] {
        let back: Option<Provenance> = serde_json::to_string(&p)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        assert_eq!(back, Some(p));
    }
}

#[test]
fn provisional_results_are_dropped_with_their_edges_and_provenance() {
    let node = |id: &str, standing: Standing| Node {
        id: NodeId::from(id.to_owned()),
        label: id.to_owned(),
        class_id: ClassId::from("entity"),
        properties: Properties::default(),
        standing,
    };
    let result = GraphResult {
        nodes: vec![
            node("a", Standing::Reviewed),
            node("b", Standing::Provisional),
            node("c", Standing::Reviewed),
        ],
        edges: vec![
            Edge {
                id: EdgeId::from("ab"),
                source_node_id: NodeId::from("a"),
                target_node_id: NodeId::from("b"),
                relation_id: RelationId::from("mentions"),
                weight: 1.0,
                properties: Properties::default(),
                standing: Standing::Reviewed,
            },
            Edge {
                id: EdgeId::from("ac"),
                source_node_id: NodeId::from("a"),
                target_node_id: NodeId::from("c"),
                relation_id: RelationId::from("mentions"),
                weight: 1.0,
                properties: Properties::default(),
                standing: Standing::Reviewed,
            },
        ],
        provenance: vec![
            Provenance {
                subject_id: String::from("b"),
                origin: Origin::from_columns(ProvenanceColumns::default()),
                confidence: 1.0,
            },
            Provenance {
                subject_id: String::from("ac"),
                origin: Origin::from_columns(ProvenanceColumns::default()),
                confidence: 1.0,
            },
        ],
        roots: vec![NodeId::from("a"), NodeId::from("b")],
        ..GraphResult::default()
    };
    let kept = result.without_provisional();
    assert_eq!(kept.nodes.len(), 2);
    assert_eq!(
        kept.edges.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
        ["ac"]
    );
    assert_eq!(kept.provenance.len(), 1);
    assert_eq!(kept.roots, [NodeId::from("a")]);
    assert_eq!(kept.status.dropped_provisional, 1);
    let mut kept = kept.without_provisional();
    assert_eq!(kept.status.dropped_provisional, 1);

    // The line under the result says what was left out and how current it is.
    kept.status.stale = true;
    assert_eq!(
        kept.summary().to_string(),
        "2 nodes, 1 edges, 1 sources; 1 provisional nodes left out, since query mode answers \
         from reviewed ones only; the graph was built with an older ontology version"
    );
    assert!(kept.to_string().ends_with(&format!("{}\n", kept.summary())));
}

#[test]
fn drift_accumulates_and_counts_distinct_names() {
    let mut drift = Drift::default();
    drift.classes.bump("vessel");
    drift.classes.bump("vessel");
    drift.relations.bump("docked_at");
    let mut total = Drift::default();
    total.absorb(&drift);
    total.absorb(&drift);
    assert_eq!(total.classes.get("vessel"), 4);
    assert_eq!(total.total(), 2);
}
