use super::*;
use crate::embedding::Dimension;
use crate::graph::{Properties, Standing, store as graph_store};
use crate::ids::{AuditId, UserId};
use crate::ontology::induction::ItemKind;
use crate::ontology::store::Revision;
use crate::storage::audit;

#[test]
fn front_matter_parses_scalars_and_both_tag_forms() {
    let WithFrontMatter { front, body } = parse_front_matter(
        "---\ntype: DuckDB Table\ntitle: \"Sales, Q1\"\ntags: [a, 'b c']\n---\n\n# Body\n",
    );
    assert_eq!(front.get("type"), Some("DuckDB Table"));
    assert_eq!(front.get("title"), Some("Sales, Q1"));
    assert_eq!(front.tags, ["a", "b c"]);
    assert_eq!(body, "# Body\n");
    let WithFrontMatter { front, body } =
        parse_front_matter("---\ntype: x\ntags:\n  - one\n  - two\nresource: r\n---\ntext");
    assert_eq!(front.tags, ["one", "two"]);
    assert_eq!(front.get("resource"), Some("r"));
    assert_eq!(body, "text");
    let WithFrontMatter { front, body } = parse_front_matter("no front matter");
    assert!(front.fields.is_empty());
    assert_eq!(body, "no front matter");
}

#[test]
fn links_resolve_relative_paths_and_skip_urls() {
    let found = links(
        "entities/vendor/orgenics.md",
        "see [Kenya](../country/kenya.md) and [site](https://x.y/z.md) and [self](#top) and [t](../../tables/shipments.md)",
    );
    assert_eq!(found, ["entities/country/kenya.md", "tables/shipments.md"]);
}

#[test]
fn exported_structural_types_are_never_domain_types() {
    for &concept in ConceptType::ALL {
        assert!(ConceptType::is_structural(&type_id(&concept.to_string())));
    }
    assert!(ConceptType::is_structural("duckdb_table"));
    assert!(!ConceptType::is_structural("organization"));
    assert!(!ConceptType::is_structural(""));
}

#[test]
fn slugs_and_type_ids_normalize() {
    assert_eq!(slug("Orgenics Ltd."), "orgenics-ltd");
    assert_eq!(slug("  "), "untitled");
    assert_eq!(type_id("DuckDB Table"), "duckdb_table");
    assert_eq!(type_id("Organizations"), "organization");
    assert_eq!(type_id("class"), "class");
    assert_eq!(type_id("3d models"), "t_3d_model");
    assert_eq!(
        BundleFile {
            path: String::from("entities/vendor/x.md"),
            content: String::new(),
        }
        .document_name(),
        "entities__vendor__x.md"
    );
}

/// The entity query names files with a SQL copy of `slug`; a label on
/// which the two disagree would link to a file that does not exist.
#[test]
fn the_sql_slug_matches_slug() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4))
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    let labels = [
        "Orgenics Ltd.",
        "  ",
        "",
        "--a--b--",
        "C++ & C#",
        "snake_case_id",
        "3D Models",
        "Ünïcode Straße",
        "\u{130}stanbul",
        "\u{212a}elvin",
        "日本",
        "a\u{00a0}b\tc",
    ];
    for label in labels {
        let from_sql: String = db
            .connection()
            .query_row(
                concat!("SELECT ", sql_slug!("s"), " FROM (SELECT ?::VARCHAR AS s)"),
                duckdb::params![label],
                |row| row.get(0),
            )
            .unwrap_or_else(|e| unreachable_db(&e.to_string()));
        assert_eq!(from_sql, slug(label), "{label:?}");
    }
}

#[test]
fn tar_round_trips_and_ignores_non_markdown() {
    let mut bundle = Bundle::default();
    let mut tar = TarSink::new(Vec::new());
    for (path, content) in [
        ("index.md", "---\ntype: index\n---\nhi"),
        ("a/b.md", "---\ntype: thing\n---\nbody"),
    ] {
        bundle.push(path, content.to_owned());
        tar.file(path, content).unwrap_or_default();
    }
    let bytes = tar.finish().unwrap_or_default();
    let back = Bundle::from_tar(&bytes).unwrap_or_default();
    assert_eq!(back, bundle);
    assert!(Bundle::from_tar(b"not a tar").is_err());
}

/// A workspace with an ontology, an audit detail row, and two nodes of
/// one class whose labels share a slug, linked by an edge; returns it
/// and how many classes the saved ontology has.
fn harbour_workspace() -> (WorkspaceDb, usize) {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4))
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    let mut ontology = Ontology::builtin_default();
    ontology.classes.push(Class {
        id: ClassId::from("harbour"),
        parent: ClassId::from(String::from(ontology::ROOT_CLASS)),
        label: None,
        description: None,
        key: None,
        properties: Vec::new(),
    });
    ontology.relations.push(Relation {
        id: RelationId::from("near"),
        label: None,
        description: None,
        domain: ClassId::from("harbour"),
        range: ClassId::from("harbour"),
    });
    let saved = ontology_store::save(&db, &ontology, Revision::reviewed(None, None))
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    audit::AuditDetail {
        id: AuditId::from("a1"),
        user_id: Some(UserId::from("u")),
        action: String::from("sql"),
        detail: serde_json::json!({"sql": "SELECT secret"}),
    }
    .write(&db)
    .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    let node = |label: &str| graph_store::NewNode {
        label: label.to_owned(),
        class_id: ClassId::from("harbour"),
        properties: Properties::default(),
        standing: Standing::Reviewed,
    };
    // Two labels, one slug: the second file gets a suffix and the
    // link from the first must point at it.
    let a = graph_store::upsert_node(&db, &node("Kenya-Coast"))
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    let b = graph_store::upsert_node(&db, &node("Kenya Coast"))
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    for id in [&a, &b] {
        graph_store::add_provenance(&db, id, &graph_store::Source::row("t", "k"))
            .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    }
    let edge = graph_store::upsert_edge(
        &db,
        &a,
        &b,
        "near",
        &Properties::default(),
        Standing::Reviewed,
    )
    .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    graph_store::add_provenance(&db, &edge, &graph_store::Source::row("t", "k"))
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    (db, saved.classes.len())
}

/// The export is one-way knowledge (issue #53): no audit detail, the
/// ontology as an exact snapshot, entity files with ids and links
/// that resolve even when two labels share a slug, and quack's stubs
/// neither ingested nor proposed on re-import.
#[test]
fn export_carries_ids_resolved_links_the_ontology_and_no_audit() {
    let (db, saved_classes) = harbour_workspace();
    let mut bundle = Bundle::default();
    let summary = export(&db, "ws", &mut bundle).unwrap_or_else(|e| unreachable_db(&e.to_string()));
    assert_eq!(summary.files, bundle.files.len());
    let log = bundle
        .files
        .iter()
        .find(|f| f.path == "log.md")
        .map(|f| f.content.clone())
        .unwrap_or_default();
    assert!(
        !log.contains("secret") && !log.contains("Activity"),
        "{log}"
    );
    let snapshot = bundle
        .ontology()
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    assert_eq!(snapshot.map(|o| o.classes.len()), Some(saved_classes));
    let entities: Vec<&BundleFile> = bundle
        .files
        .iter()
        .filter(|f| f.path.starts_with("entities/"))
        .collect();
    assert_eq!(
        entities.len(),
        2,
        "{:?}",
        bundle.files.iter().map(|f| &f.path).collect::<Vec<_>>()
    );
    let paths: Vec<&str> = entities.iter().map(|f| f.path.as_str()).collect();
    // "Kenya Coast" sorts first, so it keeps the plain name.
    assert!(
        paths.contains(&"entities/harbour/kenya-coast.md")
            && paths
                .iter()
                .any(|p| p.starts_with("entities/harbour/kenya-coast-")),
        "{paths:?}"
    );
    for file in &entities {
        assert!(file.content.contains("\nid: "), "{}", file.content);
        for link in links(&file.path, &file.content) {
            // Provenance links name table and document stubs, which a
            // real export writes alongside; entity links must resolve.
            assert!(
                link.starts_with("ontology/")
                    || link.starts_with("tables/")
                    || link.starts_with("documents/")
                    || paths.contains(&link.as_str()),
                "dangling link {link} in {}",
                file.path
            );
        }
    }
    assert!(entities.iter().any(|f| f.content.contains("- near: [")));
    // Re-import: nothing to ingest as a document, and nothing to
    // propose beyond what the restored ontology already has.
    assert_eq!(bundle.documents().count(), 0);
    let snapshot = bundle
        .ontology()
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    assert!(propose(&bundle, snapshot.as_ref()).is_empty());
}

/// Streamed into a tar or a directory, the export reads back as the
/// same files it collects in memory.
#[test]
fn the_export_streams_to_a_tar_or_a_directory() {
    let (db, _) = harbour_workspace();
    let mut bundle = Bundle::default();
    export(&db, "ws", &mut bundle).unwrap_or_else(|e| unreachable_db(&e.to_string()));
    let mut tar = TarSink::new(Vec::new());
    export(&db, "ws", &mut tar).unwrap_or_else(|e| unreachable_db(&e.to_string()));
    let bytes = tar
        .finish()
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    let from_tar = Bundle::from_tar(&bytes).unwrap_or_else(|e| unreachable_db(&e.to_string()));
    assert_eq!(from_tar, bundle);
    let dir = tempfile::tempdir().unwrap_or_else(|e| unreachable_db(&e.to_string()));
    export(&db, "ws", &mut DirSink::new(dir.path()))
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    let mut from_dir =
        Bundle::from_dir(dir.path()).unwrap_or_else(|e| unreachable_db(&e.to_string()));
    let mut expected = bundle.clone();
    from_dir.files.sort_by(|a, b| a.path.cmp(&b.path));
    expected.files.sort_by(|a, b| a.path.cmp(&b.path));
    assert_eq!(from_dir, expected);
    // An export is a bundle by its index; a folder of files is not one.
    assert!(Bundle::is_dir(dir.path()));
    let plain = tempfile::tempdir().unwrap_or_else(|e| unreachable_db(&e.to_string()));
    std::fs::write(plain.path().join("notes.md"), "# Notes")
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    assert!(!Bundle::is_dir(plain.path()));
    std::fs::write(plain.path().join(LOG), "# Log")
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    assert!(Bundle::is_dir(plain.path()));
}

#[expect(clippy::panic, reason = "test helper: the fixture must build")]
fn unreachable_db<T>(msg: &str) -> T {
    panic!("fixture failed: {msg}")
}

#[test]
fn proposals_come_from_types_links_and_resource() {
    let mut bundle = Bundle::default();
    bundle.push("index.md", String::from("---\ntype: index\n---\n"));
    bundle.push(
        "vendors/a.md",
        String::from("---\ntype: Vendors\ntitle: A\n---\nships to [K](../countries/k.md)"),
    );
    bundle.push(
        "vendors/b.md",
        String::from("---\ntype: vendors\ntitle: B\n---\n[K](../countries/k.md) again"),
    );
    bundle.push(
        "countries/k.md",
        String::from("---\ntype: country\ntitle: K\n---\n"),
    );
    bundle.push(
        "docs/d.md",
        String::from("---\ntype: document\nresource: d.pdf\n---\n"),
    );
    bundle.push("junk.md", String::from("no front matter"));
    let candidates = propose(&bundle, None);
    let ids: Vec<String> = candidates
        .iter()
        .map(|c| format!("{}:{}", c.proposal.kind(), c.proposal.id()))
        .collect();
    assert_eq!(
        ids,
        [
            "class:country",
            "class:vendor",
            "relation:vendor_links_country"
        ]
    );
    // With a document class in the ontology, `resource` is proposed on it.
    let with_documents = Ontology::builtin_default();
    let candidates = propose(&bundle, Some(&with_documents));
    assert!(
        candidates
            .iter()
            .any(|c| c.proposal.kind() == ItemKind::Property && c.proposal.id() == "resource")
    );
    let mut current = Ontology::default();
    current.classes.push(Class {
        id: ClassId::from("vendor"),
        parent: ClassId::from(String::from(ontology::ROOT_CLASS)),
        label: None,
        description: None,
        key: None,
        properties: Vec::new(),
    });
    let again = propose(&bundle, Some(&current));
    assert!(again.iter().all(|c| c.proposal.id() != "vendor"));
}

/// A `- id: [..](..)` link proposes that id; a relation the ontology
/// already has between the two classes, under any id or on an
/// ancestor, proposes nothing (issue #70: re-importing quack's own
/// bundle queued every link again).
#[test]
fn named_links_keep_their_id_and_covered_relations_are_skipped() {
    let mut bundle = Bundle::default();
    bundle.push(
        "entities/a.md",
        String::from(
            "---\ntype: vendor\ntitle: A\ngenerator: quack\n---\n## Links\n\n- ships_to: [K](../entities/k.md)\n- Note: [K](../entities/k.md)\n",
        ),
    );
    bundle.push(
        "entities/k.md",
        String::from("---\ntype: country\ntitle: K\ngenerator: quack\n---\n"),
    );
    let ids = |candidates: &[Candidate]| -> Vec<String> {
        candidates
            .iter()
            .filter(|c| c.proposal.kind() == ItemKind::Relation)
            .map(|c| c.proposal.id().to_owned())
            .collect()
    };
    assert_eq!(
        ids(&propose(&bundle, None)),
        ["ships_to", "vendor_links_country"]
    );
    let relation = |id: &str, domain: &str, range: &str| Relation {
        id: RelationId::from(String::from(id)),
        label: None,
        description: None,
        domain: ClassId::from(String::from(domain)),
        range: ClassId::from(String::from(range)),
    };
    let class = |id: &str, parent: &str| Class {
        id: ClassId::from(String::from(id)),
        parent: ClassId::from(String::from(parent)),
        label: None,
        description: None,
        key: None,
        properties: Vec::new(),
    };
    let mut current = Ontology::default();
    current
        .classes
        .push(class("organisation", ontology::ROOT_CLASS));
    current.classes.push(class("vendor", "organisation"));
    current.classes.push(class("country", ontology::ROOT_CLASS));
    current
        .relations
        .push(relation("based_in", "organisation", "country"));
    assert!(ids(&propose(&bundle, Some(&current))).is_empty());
    current.relations.clear();
    current
        .relations
        .push(relation("ships_to", "vendor", "vendor"));
    assert_eq!(
        ids(&propose(&bundle, Some(&current))),
        ["vendor_links_country"]
    );
    assert_eq!(labelled_link("  - killed_in: [x](y.md)"), Some("killed_in"));
    assert_eq!(labelled_link("- Killed In: [x](y.md)"), None);
    assert_eq!(labelled_link("- table [t](../t.md)"), None);
}

/// Two file names with one slug get two document files, and an entity's
/// provenance links the file of the document it came from.
#[test]
fn documents_whose_names_share_a_slug_get_their_own_files() {
    use crate::ids::{ChunkId, DocumentId};
    use crate::storage::workspace::NewDocument;

    let (db, _) = harbour_workspace();
    for (id, name) in [("doc-aaaaaa", "Report.pdf"), ("doc-bbbbbb", "report.pdf")] {
        db.insert_document(&NewDocument::new(
            &DocumentId::from(id),
            name,
            "application/pdf",
            1,
        ))
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    }
    let node = graph_store::upsert_node(
        &db,
        &graph_store::NewNode {
            label: String::from("Mombasa"),
            class_id: ClassId::from("harbour"),
            properties: Properties::default(),
            standing: Standing::Reviewed,
        },
    )
    .unwrap_or_else(|e| unreachable_db(&e.to_string()));
    graph_store::add_provenance(
        &db,
        &node,
        &graph_store::Source::chunk(&DocumentId::from("doc-bbbbbb"), &ChunkId::from("c1"), 0.9),
    )
    .unwrap_or_else(|e| unreachable_db(&e.to_string()));

    let mut bundle = Bundle::default();
    export(&db, "ws", &mut bundle).unwrap_or_else(|e| unreachable_db(&e.to_string()));
    let mut documents: Vec<&str> = bundle
        .files
        .iter()
        .map(|f| f.path.as_str())
        .filter(|p| p.starts_with("documents/"))
        .collect();
    documents.sort_unstable();
    assert_eq!(
        documents,
        ["documents/report-pdf-bbbbbb.md", "documents/report-pdf.md"]
    );
    let mombasa = bundle
        .files
        .iter()
        .find(|f| f.path == "entities/harbour/mombasa.md")
        .map(|f| f.content.clone())
        .unwrap_or_default();
    assert!(
        mombasa.contains("[report.pdf](../../documents/report-pdf-bbbbbb.md)"),
        "{mombasa}"
    );
}
