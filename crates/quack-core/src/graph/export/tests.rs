use std::io::Read as _;

use super::*;
use crate::embedding::Dimension;
use crate::graph::Properties;
use crate::graph::store::{self, Assertion, NewNode, Source as NewSource};
use crate::ids::{ClassId, NodeId};
use crate::ontology::store::{Revision, save};

/// A label and a property value with what CSV and XML must escape: a
/// comma, quotes, a line break, markup, and a control character XML 1.0
/// cannot carry at all.
const AWKWARD: &str = "Acme, \"Inc.\" <b>&</b>\nline two \u{7}";

struct Fixture {
    db: WorkspaceDb,
    ada: NodeId,
}

#[expect(clippy::unwrap_used, reason = "test fixture")]
fn fixture() -> Fixture {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    save(
        &db,
        &Ontology::builtin_default(),
        Revision::reviewed(Some("tester"), None),
    )
    .unwrap();
    let node = |label: &str, class: &str, properties: Properties, standing| {
        store::upsert_node(
            &db,
            &NewNode {
                label: label.to_owned(),
                class_id: ClassId::from(class),
                properties,
                standing,
            },
        )
        .unwrap()
    };
    let ada = node(
        "Ada",
        "person",
        Properties::from(serde_json::json!({ "note": AWKWARD })),
        Standing::Reviewed,
    );
    let acme = node(
        AWKWARD,
        "organization",
        Properties::default(),
        Standing::Reviewed,
    );
    let draft = node(
        "Draft",
        "person",
        Properties::default(),
        Standing::Provisional,
    );
    let works = store::upsert_edge(
        &db,
        &ada,
        &acme,
        "works_at",
        &Properties::from(serde_json::json!({ "since": 2020 })),
        Standing::Reviewed,
    )
    .unwrap();
    // A reviewed edge to a provisional node goes only with provisional rows.
    store::upsert_edge(
        &db,
        &draft,
        &acme,
        "works_at",
        &Properties::default(),
        Standing::Reviewed,
    )
    .unwrap();
    let chunk = NewSource::chunk(&DocumentId::from("doc"), &ChunkId::from("chunk-1"), 0.75);
    store::add_provenance(&db, &ada, &chunk).unwrap();
    store::add_provenance(&db, &works, &chunk).unwrap();
    store::add_provenance(&db, &acme, &NewSource::row("companies", "42")).unwrap();
    store::add_provenance(&db, &draft, &chunk).unwrap();
    store::create_node(
        &db,
        &NewNode {
            label: String::from("Grace"),
            class_id: ClassId::from("person"),
            properties: Properties::default(),
            standing: Standing::Reviewed,
        },
        &Assertion {
            author: Some(String::from("ada")),
            note: Some(String::from("met her")),
        },
    )
    .unwrap();
    Fixture { db, ada }
}

fn export(provisional: ProvisionalExport, format: GraphFormat) -> GraphExport {
    GraphExport {
        format,
        provisional,
    }
}

#[expect(clippy::unwrap_used, reason = "test")]
fn stream(db: &WorkspaceDb, export: GraphExport) -> (Vec<u8>, ExportSummary) {
    let mut out = Vec::new();
    let summary = db
        .read_only(|db| export.write(db, Destination::Stream(&mut out)))
        .unwrap();
    (out, summary)
}

#[expect(clippy::unwrap_used, reason = "test")]
fn records(text: &[u8]) -> Vec<csv::StringRecord> {
    csv::Reader::from_reader(text)
        .records()
        .collect::<std::result::Result<_, _>>()
        .unwrap()
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn a_csv_bundle_leaves_out_provisional_rows_and_keeps_every_source() {
    let Fixture { db, ada } = fixture();
    let dir = tempfile::tempdir().unwrap();
    let summary = db
        .read_only(|db| {
            export(ProvisionalExport::Exclude, GraphFormat::Csv)
                .write::<Vec<u8>>(db, Destination::Dir(dir.path()))
        })
        .unwrap();
    assert_eq!(
        summary,
        ExportSummary {
            format: GraphFormat::Csv,
            nodes: 3,
            edges: 1,
            provenance: 4,
        }
    );
    let read = |name: &str| std::fs::read(dir.path().join(name)).unwrap();

    let nodes_csv = read("nodes.csv");
    assert!(nodes_csv.starts_with(b"id,label,class_id,properties,provisional\n"));
    let nodes = records(&nodes_csv);
    let labels: Vec<&str> = nodes.iter().filter_map(|r| r.get(1)).collect();
    assert!(
        labels.contains(&AWKWARD) && labels.contains(&"Grace"),
        "{labels:?}"
    );
    assert!(!labels.contains(&"Draft"), "{labels:?}");
    let ada_row = nodes
        .iter()
        .find(|r| r.get(0) == Some(ada.as_str()))
        .unwrap();
    let properties: serde_json::Value = serde_json::from_str(ada_row.get(3).unwrap()).unwrap();
    assert_eq!(properties, serde_json::json!({ "note": AWKWARD }));
    assert_eq!(ada_row.get(4), Some("false"));

    let edges_csv = read("edges.csv");
    assert!(edges_csv.starts_with(b"id,source,target,relation_id,weight,properties,provisional\n"));
    let edges = records(&edges_csv);
    assert_eq!(edges.len(), 1);
    assert_eq!(edges.first().and_then(|r| r.get(1)), Some(ada.as_str()));
    assert_eq!(edges.first().and_then(|r| r.get(3)), Some("works_at"));

    let provenance = records(&read("provenance.csv"));
    assert_eq!(provenance.len(), 4);
    let manual = provenance.iter().find(|r| r.get(6) == Some("ada")).unwrap();
    assert_eq!(manual.get(7), Some("met her"));
    assert!(!manual.get(8).unwrap_or_default().is_empty(), "{manual:?}");
    assert!(
        provenance
            .iter()
            .any(|r| r.get(3) == Some("companies") && r.get(4) == Some("42"))
    );
    assert!(
        provenance
            .iter()
            .any(|r| r.get(2) == Some("chunk-1") && r.get(5) == Some("0.75"))
    );
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn a_csv_tar_holds_the_three_files_with_provisional_rows_when_asked() {
    let Fixture { db, .. } = fixture();
    let (bytes, summary) = stream(&db, export(ProvisionalExport::Include, GraphFormat::Csv));
    assert_eq!(
        (summary.nodes, summary.edges, summary.provenance),
        (4, 2, 5)
    );
    let mut archive = tar::Archive::new(bytes.as_slice());
    let mut files = Vec::new();
    for entry in archive.entries().unwrap() {
        let mut entry = entry.unwrap();
        let name = entry.path().unwrap().display().to_string();
        let mut text = Vec::new();
        entry.read_to_end(&mut text).unwrap();
        files.push((name, text));
    }
    let names: Vec<&str> = files.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["nodes.csv", "edges.csv", "provenance.csv"]);
    let rows: Vec<usize> = files.iter().map(|(_, text)| records(text).len()).collect();
    assert_eq!(rows, [4, 2, 5]);
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn an_empty_graph_exports_headers_and_an_empty_document() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    let (bytes, summary) = stream(&db, export(ProvisionalExport::Exclude, GraphFormat::Csv));
    assert_eq!(
        (summary.nodes, summary.edges, summary.provenance),
        (0, 0, 0)
    );
    let mut archive = tar::Archive::new(bytes.as_slice());
    assert_eq!(archive.entries().unwrap().count(), 3);

    let (bytes, summary) = stream(&db, export(ProvisionalExport::Exclude, GraphFormat::JsonLd));
    assert_eq!(summary.nodes, 0);
    let document: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // The implicit root class and `mentions` relation, and nothing else.
    assert_eq!(
        document
            .get("@graph")
            .and_then(serde_json::Value::as_array)
            .map(Vec::len),
        Some(2)
    );

    let (bytes, _) = stream(
        &db,
        export(ProvisionalExport::Exclude, GraphFormat::GraphMl),
    );
    let (nodes, edges) = graphml_elements(&bytes);
    assert_eq!((nodes, edges), (0, 0));
}

/// Read a `GraphML` document to its end, which fails on anything not
/// well-formed, and count its nodes and edges.
#[expect(clippy::unwrap_used, reason = "test")]
fn graphml_elements(bytes: &[u8]) -> (usize, usize) {
    let mut reader = quick_xml::Reader::from_reader(bytes);
    let mut buf = Vec::new();
    let (mut nodes, mut edges) = (0_usize, 0_usize);
    loop {
        match reader.read_event_into(&mut buf).unwrap() {
            Event::Eof => break,
            Event::Start(start) => match start.name().into_inner() {
                "node" => nodes = nodes.saturating_add(1),
                "edge" => edges = edges.saturating_add(1),
                _ => {}
            },
            _ => {}
        }
        buf.clear();
    }
    (nodes, edges)
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn graphml_is_well_formed_with_escaped_text_and_provenance() {
    let Fixture { db, ada } = fixture();
    let (bytes, summary) = stream(
        &db,
        export(ProvisionalExport::Exclude, GraphFormat::GraphMl),
    );
    assert_eq!(
        (summary.nodes, summary.edges, summary.provenance),
        (3, 1, 4)
    );
    assert_eq!(graphml_elements(&bytes), (3, 1));
    let text = String::from_utf8(bytes).unwrap();
    assert!(text.contains(r#"<graphml xmlns="http://graphml.graphdrawing.org/xmlns">"#));
    assert!(text.contains(&format!(r#"<node id="{ada}">"#)), "{text}");
    assert!(text.contains("&lt;b&gt;&amp;&lt;/b&gt;"), "{text}");
    assert!(!text.contains('\u{7}'), "{text}");
    assert!(text.contains('\u{fffd}'), "{text}");
    assert!(text.contains("met her"), "{text}");
    assert!(!text.contains("Draft"), "{text}");
}

#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn json_ld_types_nodes_by_class_and_edges_by_relation() {
    let Fixture { db, ada } = fixture();
    let (bytes, summary) = stream(&db, export(ProvisionalExport::Include, GraphFormat::JsonLd));
    assert_eq!(
        (summary.nodes, summary.edges, summary.provenance),
        (4, 2, 5)
    );
    let document: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let context = document.get("@context").unwrap();
    assert_eq!(
        context.get("class"),
        Some(&serde_json::json!("urn:quack:class:"))
    );
    assert_eq!(
        context.get("relation"),
        Some(&serde_json::json!("urn:quack:relation:"))
    );
    let items = document
        .get("@graph")
        .and_then(serde_json::Value::as_array)
        .unwrap();
    let find = |id: &str| {
        items
            .iter()
            .find(|i| i.get("@id") == Some(&serde_json::json!(id)))
            .unwrap()
    };
    let works_at = find("relation:works_at");
    assert_eq!(
        works_at.get("domain"),
        Some(&serde_json::json!("class:person"))
    );
    assert_eq!(
        find("class:person").get("@type"),
        Some(&serde_json::json!("rdfs:Class"))
    );

    let node = find(&format!("node:{ada}"));
    assert_eq!(node.get("@type"), Some(&serde_json::json!("class:person")));
    assert_eq!(
        node.pointer("/properties/note"),
        Some(&serde_json::json!(AWKWARD))
    );
    assert_eq!(
        node.pointer("/provenance/0"),
        Some(
            &serde_json::json!({ "document_id": "doc", "chunk_id": "chunk-1", "confidence": 0.75 })
        )
    );
    let grace = items
        .iter()
        .find(|i| i.get("label") == Some(&serde_json::json!("Grace")))
        .unwrap();
    assert_eq!(
        grace.pointer("/provenance/0/author"),
        Some(&serde_json::json!("ada"))
    );
    let edges: Vec<&serde_json::Value> = items
        .iter()
        .filter(|i| i.get("@type") == Some(&serde_json::json!("rdf:Statement")))
        .collect();
    assert_eq!(edges.len(), 2);
    assert!(
        edges
            .iter()
            .all(|e| e.get("predicate") == Some(&serde_json::json!("relation:works_at")))
    );
    assert!(
        items
            .iter()
            .any(|i| i.get("provisional") == Some(&serde_json::json!(true)))
    );
}

#[test]
fn xml_text_replaces_only_what_xml_cannot_carry() {
    assert!(matches!(XmlText::new("plain <&> \t\n").0, Cow::Borrowed(_)));
    assert_eq!(
        XmlText::new("a\u{0}b\u{1f}c\u{ffff}").as_ref(),
        "a\u{fffd}b\u{fffd}c\u{fffd}"
    );
    assert_eq!(
        XmlText::new("caf\u{e9} \u{85}").as_ref(),
        "caf\u{e9} \u{85}"
    );
}

#[test]
fn formats_read_their_names() {
    assert_eq!(
        "GraphML".parse::<GraphFormat>().ok(),
        Some(GraphFormat::GraphMl)
    );
    assert_eq!(
        "jsonld".parse::<GraphFormat>().ok(),
        Some(GraphFormat::JsonLd)
    );
    assert!("gexf".parse::<GraphFormat>().is_err());
    assert_eq!(
        serde_json::to_value(GraphFormat::JsonLd).ok(),
        Some(serde_json::json!("jsonld"))
    );
}
