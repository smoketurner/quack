#![expect(clippy::unwrap_used, reason = "test assertions use unwrap throughout")]

//! The knowledge graph end to end inside one workspace file: table mapping
//! extraction, constrained extraction with a canned model, resolution,
//! traversal, revalidation, provisional handling, and merge review.

use std::collections::BTreeMap;

use quack_core::embedding::{Embedder, Profile, Prompts, Role};
use quack_core::graph::extract::{Extraction, GraphExtractor};
use quack_core::graph::store::NewNode;
use quack_core::graph::{GraphOptions, extract, resolve, store as graph_store, tables, traverse};
use quack_core::ontology::{self, Class, Mapping, MappingRelation, Ontology, Relation, store};
use quack_core::storage::workspace::{NewChunk, NewDocument, WorkspaceDb};
use quack_core::storage::writer::Writer;
use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};

const DIM: usize = 4;

/// A writer over a second connection to `db`'s database, for the steps
/// that take one, while the test reads and writes through `db`.
fn writer_of(db: &WorkspaceDb) -> Writer {
    Writer::spawn(db.try_clone_reader().unwrap()).unwrap()
}

/// Labels sharing a first letter embed close together; others far apart.
struct LetterEmbedding;

/// `LetterEmbedding` with no prefixes, so a label reaches it unchanged.
fn letters() -> Embedder<LetterEmbedding> {
    Embedder::new(
        LetterEmbedding,
        Profile::new("letters", 4, Prompts::default()),
    )
}

impl EmbeddingModel for LetterEmbedding {
    const MAX_DOCUMENTS: usize = 1024;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
        Self
    }

    fn ndims(&self) -> usize {
        DIM
    }

    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + Send {
        let mut out = Vec::new();
        for text in texts {
            let first = text.to_lowercase().chars().next().unwrap_or('z');
            // A tiny second component so labels of one letter are close but
            // not identical.
            let wobble = f64::from(u32::try_from(text.len() % 3).unwrap_or(0)) * 0.01;
            let vec = match first {
                'a' => vec![1.0, wobble, 0.0, 0.0],
                'k' => vec![0.0, 1.0, wobble, 0.0],
                'u' => vec![0.0, 0.0, 1.0, wobble],
                _ => vec![wobble, 0.0, 0.0, 1.0],
            };
            out.push(Embedding {
                document: text,
                vec,
            });
        }
        async move { Ok(out) }
    }
}

fn ontology() -> Ontology {
    let class = |id: &str, parent: &str, key: Option<&str>| Class {
        id: id.to_owned(),
        parent: parent.to_owned(),
        label: None,
        description: None,
        key: key.map(str::to_owned),
        properties: key
            .into_iter()
            .map(str::to_owned)
            .chain((id == "shipment").then(|| String::from("mode")))
            .collect(),
    };
    let relation = |id: &str, domain: &str, range: &str| Relation {
        id: id.to_owned(),
        label: None,
        description: None,
        domain: domain.to_owned(),
        range: range.to_owned(),
    };
    Ontology {
        version: 0,
        classes: vec![
            class("organization", ontology::ROOT_CLASS, None),
            class("vendor", "organization", Some("name")),
            class("country", ontology::ROOT_CLASS, Some("name")),
            class("shipment", ontology::ROOT_CLASS, Some("po")),
        ],
        relations: vec![
            relation("ships_to", "organization", "country"),
            relation("supplied_by", "shipment", "vendor"),
            relation("delivered_to", "shipment", "country"),
        ],
        properties: vec![
            ontology::Property {
                id: String::from("name"),
                label: None,
                kind: ontology::PropertyType::String,
                values: Vec::new(),
            },
            ontology::Property {
                id: String::from("po"),
                label: None,
                kind: ontology::PropertyType::String,
                values: Vec::new(),
            },
            ontology::Property {
                id: String::from("mode"),
                label: None,
                kind: ontology::PropertyType::String,
                values: Vec::new(),
            },
        ],
        mappings: vec![Mapping {
            table: String::from("shipments"),
            class: String::from("shipment"),
            key: String::from("po"),
            properties: BTreeMap::from([(String::from("mode"), String::from("mode"))]),
            relations: vec![
                MappingRelation {
                    relation: String::from("supplied_by"),
                    column: String::from("vendor"),
                    target_class: String::from("vendor"),
                    target_key: String::from("name"),
                },
                MappingRelation {
                    relation: String::from("delivered_to"),
                    column: String::from("country"),
                    target_class: String::from("country"),
                    target_key: String::from("name"),
                },
            ],
        }],
    }
}

fn workspace() -> WorkspaceDb {
    let db = WorkspaceDb::open_in_memory(4).unwrap();
    db.execute_statement(
        "CREATE TABLE shipments AS SELECT * FROM (VALUES \
         ('PO-1', 'Orgenics', 'Kenya', 'Air'), \
         ('PO-2', 'Orgenics', 'Uganda', 'Sea'), \
         ('PO-3', 'Aurobindo', 'Kenya', NULL), \
         (NULL, 'Nobody', 'Nowhere', 'Air')) AS t(po, vendor, country, mode)",
    )
    .unwrap();
    db.insert_document(
        &NewDocument::new("doc-1", "notes.md", "text/markdown", 10).with_status("ready"),
    )
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: "c1",
        document_id: "doc-1",
        chunk_index: 0,
        content: "Orgenics ships to Kenya from its plant.",
        heading: Some("Vendors"),
        page: None,
        embedding: None,
    })
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: "c2",
        document_id: "doc-1",
        chunk_index: 1,
        content: "FAIL this one",
        heading: None,
        page: None,
        embedding: None,
    })
    .unwrap();
    store::save(&db, &ontology(), Some("test"), Some("fixture")).unwrap();
    db
}

struct Canned;

impl GraphExtractor for Canned {
    fn extract<'a>(&'a self, text: &'a str) -> extract::ExtractFuture<'a> {
        Box::pin(async move {
            if text.contains("FAIL") {
                return Err(quack_core::error::Error::Llm(String::from("boom")));
            }
            let extraction: Extraction = serde_json::from_str(
                r#"{"nodes": [
                    {"label": "Orgenics Ltd", "class": "vendor", "properties": {"name": "Orgenics Ltd"}},
                    {"label": "Kenya", "class": "country"},
                    {"label": "MV Hope", "class": "vessel"}
                ], "edges": [
                    {"source": "Orgenics Ltd", "target": "Kenya", "relation": "ships_to"},
                    {"source": "Orgenics Ltd", "target": "Kenya", "relation": "docked_at"}
                ]}"#,
            )
            .unwrap();
            Ok(extraction)
        })
    }
}

/// A mapped table larger than one batch is extracted whole, in key
/// order across batches, and a neighbourhood walk never visits more than
/// `max_nodes` (issue #48).
#[test]
fn large_tables_extract_in_batches_and_neighbourhoods_stay_bounded() {
    let db = WorkspaceDb::open_in_memory(4).unwrap();
    let rows = tables::BATCH_ROWS * 2 + 1;
    db.execute_statement(&format!(
        "CREATE TABLE shipments AS SELECT 'PO-' || lpad(range::VARCHAR, 5, '0') AS po, \
         'V' || (range % 3) AS vendor, 'C' || (range % 2) AS country, 'Air' AS mode \
         FROM range({rows})"
    ))
    .unwrap();
    store::save(&db, &ontology(), Some("test"), Some("fixture")).unwrap();
    let current = store::current(&db).unwrap().unwrap();
    let summaries = tables::extract(&db, &current, false).unwrap();
    let first = summaries.first().unwrap();
    assert_eq!(
        (first.rows, first.nodes, first.edges),
        (rows, rows, rows * 2)
    );
    let status = graph_store::status(&db).unwrap();
    assert_eq!(status.nodes, u64::from(rows) + 3 + 2);
    // Re-running stays idempotent across batches.
    tables::extract(&db, &current, false).unwrap();
    assert_eq!(
        graph_store::status(&db).unwrap().nodes,
        u64::from(rows) + 3 + 2
    );

    // Vendor V0 is a hub with about a third of the shipments: a bounded
    // walk out of it returns at most max_nodes.
    let options = GraphOptions {
        max_nodes: 7,
        ..GraphOptions::default()
    };
    let hub = traverse::resolve_entry(&db, "V0", Some("vendor"), None).unwrap();
    let found = traverse::neighborhood(&db, &hub, 3, None, &options).unwrap();
    assert_eq!(found.nodes.len(), 7, "{}", found.nodes.len());
}

/// Sampling spreads across documents (issue #60): a second, longer
/// document contributes as many chunks as the first to a sample of 2.
#[test]
fn extraction_samples_evenly_across_documents() {
    let db = workspace();
    db.insert_document(
        &NewDocument::new("doc-2", "long.md", "text/markdown", 10).with_status("ready"),
    )
    .unwrap();
    for i in 0..4 {
        db.insert_chunk(&NewChunk {
            id: &format!("l{i}"),
            document_id: "doc-2",
            chunk_index: i,
            content: "Filler text about nothing in particular.",
            heading: None,
            page: None,
            embedding: None,
        })
        .unwrap();
    }
    let sampled = extract::chunks(&db, Some(2)).unwrap();
    let docs: std::collections::BTreeSet<&str> =
        sampled.iter().map(|c| c.document_id.as_str()).collect();
    assert_eq!(sampled.len(), 2, "{sampled:?}");
    assert_eq!(docs.len(), 2, "one chunk from each document: {sampled:?}");
    assert_eq!(extract::chunks(&db, Some(0)).unwrap().len(), 0);
    assert_eq!(extract::chunks(&db, None).unwrap().len(), 6);
}

/// Resolution respects provenance (issue #41): two nodes from keyed rows
/// are never merged or proposed however close their labels; a keyed node
/// and an extracted look-alike are proposed for review, never
/// auto-merged; two extracted look-alikes still auto-merge.
#[tokio::test]
async fn resolution_never_merges_keyed_rows_and_only_auto_merges_extracted_nodes() {
    let db = workspace();
    let writer = writer_of(&db);
    let current = store::current(&db).unwrap().unwrap();
    tables::extract(&db, &current, false).unwrap();
    let node = |label: &str| NewNode {
        label: label.to_owned(),
        class_id: String::from("country"),
        properties: serde_json::json!({}),
        provisional: false,
    };
    // "Kenya" (from the table) and "Kenya Coast" embed identically under
    // LetterEmbedding (same first letter, same length mod 3) and share a
    // token: the old pass auto-merged such pairs.
    let coast = graph_store::upsert_node(&db, &node("Kenya Coast")).unwrap();
    graph_store::add_provenance(&db, &coast, &graph_store::Source::row("places", "KC")).unwrap();
    // An extracted look-alike of a keyed node, and two extracted
    // look-alikes of each other.
    let doc = |chunk: &str| graph_store::Source::chunk("doc-1", chunk, 0.9);
    let kenya_ltd = graph_store::upsert_node(&db, &node("Kenya Ltd")).unwrap();
    graph_store::add_provenance(&db, &kenya_ltd, &doc("c1")).unwrap();
    let uganda_north = graph_store::upsert_node(&db, &node("Uganda North")).unwrap();
    graph_store::add_provenance(&db, &uganda_north, &doc("c1")).unwrap();
    let uganda_south = graph_store::upsert_node(&db, &node("Uganda South")).unwrap();
    graph_store::add_provenance(&db, &uganda_south, &doc("c2")).unwrap();
    let before = graph_store::status(&db).unwrap().nodes;

    let options = GraphOptions {
        merge_threshold: 0.5,
        auto_merge_threshold: 0.05,
        ..GraphOptions::default()
    };
    let resolved = resolve::resolve(&writer, Some(&letters()), &options)
        .await
        .unwrap();
    assert_eq!(resolved.auto_merged, 1, "{resolved:?}");
    assert_eq!(graph_store::status(&db).unwrap().nodes, before - 1);
    // Both keyed Kenyas survive.
    assert!(graph_store::node(&db, &coast).unwrap().is_some());
    let pending = resolve::pending(&db).unwrap();
    let pairs: Vec<(String, String)> = pending
        .iter()
        .map(|m| (m.keep.label.clone(), m.drop.label.clone()))
        .collect();
    // Kenya and Kenya Coast (both keyed) are never paired with each other.
    assert!(
        !pairs.iter().any(|(k, d)| {
            (k == "Kenya" && d == "Kenya Coast") || (k == "Kenya Coast" && d == "Kenya")
        }),
        "{pairs:?}"
    );
    // The keyed node is the one kept; the extracted look-alike would drop.
    assert!(
        pairs.contains(&(String::from("Kenya"), String::from("Kenya Ltd"))),
        "{pairs:?}"
    );
    assert!(graph_store::node(&db, &kenya_ltd).unwrap().is_some());
    // The two Ugandas from documents merged on their own.
    let ugandas = [uganda_north, uganda_south]
        .iter()
        .filter(|id| graph_store::node(&db, id).unwrap().is_some())
        .count();
    assert_eq!(ugandas, 1);
}

/// Deleting a document takes with it the nodes and edges only it
/// supported (issue #43): a document's chunk provenance, then a table's
/// row provenance when the document that loaded the table goes. What
/// other sources still support stays, the mapping to the dropped table
/// is flagged rather than fatal, and the ontology still saves.
#[test]
fn deleting_a_document_removes_the_graph_rows_only_it_supported() {
    let db = workspace();
    let current = store::current(&db).unwrap().unwrap();
    tables::extract(&db, &current, false).unwrap();
    assert_eq!(graph_store::status(&db).unwrap().nodes, 7);

    // A second document adds one node of its own, one edge of its own,
    // and a second source for a vendor the table already produced.
    db.insert_document(
        &NewDocument::new("doc-2", "extra.md", "text/markdown", 10).with_status("ready"),
    )
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: "c3",
        document_id: "doc-2",
        chunk_index: 0,
        content: "Orgenics ships to Nowhere.",
        heading: None,
        page: None,
        embedding: None,
    })
    .unwrap();
    let node = |label: &str, class: &str| NewNode {
        label: label.to_owned(),
        class_id: class.to_owned(),
        properties: serde_json::json!({}),
        provisional: false,
    };
    let source = graph_store::Source::chunk("doc-2", "c3", 0.9);
    let orgenics = graph_store::upsert_node(&db, &node("Orgenics", "vendor")).unwrap();
    graph_store::add_provenance(&db, &orgenics, &source).unwrap();
    let nowhere = graph_store::upsert_node(&db, &node("Nowhere", "country")).unwrap();
    graph_store::add_provenance(&db, &nowhere, &source).unwrap();
    let edge = graph_store::upsert_edge(
        &db,
        &orgenics,
        &nowhere,
        "ships_to",
        &serde_json::json!({}),
        false,
    )
    .unwrap();
    graph_store::add_provenance(&db, &edge, &source).unwrap();
    assert_eq!(graph_store::status(&db).unwrap().nodes, 8);

    assert!(db.delete_document("doc-2", None).unwrap());
    let status = graph_store::status(&db).unwrap();
    assert_eq!((status.nodes, status.edges), (7, 6));
    assert!(graph_store::node(&db, &nowhere).unwrap().is_none());
    assert!(graph_store::node(&db, &orgenics).unwrap().is_some());
    assert!(
        graph_store::provenance_of(&db, std::slice::from_ref(&orgenics))
            .unwrap()
            .iter()
            .all(|p| p.document_id.is_none())
    );

    // The document that loaded the mapped table: the table drops, and
    // with it every node and edge the rows supported.
    db.insert_document(
        &NewDocument::new("doc-t", "shipments.csv", "text/csv", 10).with_status("ready"),
    )
    .unwrap();
    db.set_document_tables("doc-t", &[String::from("shipments")])
        .unwrap();
    assert!(db.delete_document("doc-t", None).unwrap());
    assert!(db.list_tables().unwrap().is_empty());
    let status = graph_store::status(&db).unwrap();
    assert_eq!((status.nodes, status.edges), (0, 0));
    assert_eq!(status.missing_tables, vec![String::from("shipments")]);
    let orphans: i64 = db
        .connection()
        .query_row("SELECT count(*) FROM _quack_provenance", [], |r| r.get(0))
        .unwrap();
    assert_eq!(orphans, 0);

    let summaries = tables::extract(&db, &current, false).unwrap();
    assert_eq!(summaries.len(), 1);
    assert!(summaries.first().unwrap().skipped.is_some());
    assert!(store::save(&db, &current, Some("test"), Some("still saves")).is_ok());
}

#[tokio::test]
async fn tables_documents_resolution_and_traversal_end_to_end() {
    let db = workspace();
    let writer = writer_of(&db);
    let current = store::current(&db).unwrap().unwrap();

    // Table mapping: three keyed rows become shipments with edges to
    // vendors and countries; the NULL-keyed row is skipped.
    let summaries = tables::extract(&db, &current, false).unwrap();
    assert_eq!(summaries.len(), 1);
    let first = summaries.first().unwrap();
    assert_eq!((first.rows, first.nodes, first.edges), (3, 3, 6));
    let status = graph_store::status(&db).unwrap();
    assert_eq!(status.nodes, 3 + 2 + 2, "shipments, vendors, countries");
    assert_eq!(status.edges, 6);
    assert!(status.enabled() && !status.provisional());
    // Re-running is idempotent.
    tables::extract(&db, &current, false).unwrap();
    assert_eq!(graph_store::status(&db).unwrap().nodes, 7);

    // Constrained extraction: the vessel and docked_at are drift, the
    // failed chunk is skipped, Kenya merges with the table's Kenya.
    let chunks = extract::chunks(&db, None).unwrap();
    assert_eq!(chunks.len(), 2);
    let summary = extract::run(&writer, chunks, &Canned, &current, false, 2, &|_| {})
        .await
        .unwrap();
    // Both chunks are on record (the failed one is not), so the next run
    // sends only the failed one again.
    assert_eq!(graph_store::extracted_chunks(&db).unwrap(), 1);
    let remaining = extract::chunks(&db, None).unwrap();
    assert_eq!(remaining.len(), 1, "{remaining:?}");
    assert_eq!(remaining.first().map(|c| c.chunk_id.as_str()), Some("c2"));
    assert_eq!(
        (
            summary.chunks,
            summary.failed_chunks,
            summary.nodes,
            summary.edges
        ),
        (2, 1, 2, 1)
    );
    assert_eq!(summary.drift.classes.get("vessel"), Some(&1));
    assert_eq!(summary.drift.relations.get("docked_at"), Some(&1));
    let status = graph_store::status(&db).unwrap();
    assert_eq!(status.nodes, 8, "Orgenics Ltd is new; Kenya merged");
    assert_eq!(status.drift.total(), 2);

    // Resolution: Orgenics and Orgenics Ltd share a token and embed close,
    // so a merge is proposed; Aurobindo stays apart.
    let options = GraphOptions {
        merge_threshold: 0.5,
        auto_merge_threshold: 0.0,
        ..GraphOptions::default()
    };
    let resolved = resolve::resolve(&writer, Some(&letters()), &options)
        .await
        .unwrap();
    assert_eq!(resolved.embedded, 8);
    assert_eq!(resolved.auto_merged, 0);
    let pending = resolve::pending(&db).unwrap();
    assert_eq!(pending.len(), 1, "{pending:?}");
    let labels = [
        pending.first().unwrap().keep.label.as_str(),
        pending.first().unwrap().drop.label.as_str(),
    ];
    assert!(labels.contains(&"Orgenics") && labels.contains(&"Orgenics Ltd"));
    assert_eq!(graph_store::status(&db).unwrap().pending_merges, 1);

    // Traversal before the merge: Kenya's neighbourhood reaches both
    // vendors through the shipments and the direct ships_to edge.
    let roots = traverse::resolve_entry(&db, "kenya", None, None).unwrap();
    assert_eq!(roots.len(), 1);
    let hood = traverse::neighborhood(&db, &roots, 2, None, &options).unwrap();
    let labels: Vec<&str> = hood.nodes.iter().map(|n| n.label.as_str()).collect();
    assert!(
        labels.contains(&"PO-1")
            && labels.contains(&"Orgenics Ltd")
            && labels.contains(&"Aurobindo"),
        "{labels:?}"
    );
    assert!(!labels.contains(&"Uganda"), "Uganda is three hops away");
    assert!(
        hood.provenance
            .iter()
            .any(|p| p.chunk_id.as_deref() == Some("c1"))
    );
    assert!(
        hood.provenance
            .iter()
            .any(|p| p.table_name.as_deref() == Some("shipments"))
    );
    let only = traverse::neighborhood(&db, &roots, 2, Some("delivered_to"), &options).unwrap();
    assert!(only.edges.iter().all(|e| e.relation_id == "delivered_to"));

    // Fuzzy entry: an unknown spelling resolves through the embedding.
    let vector = letters().one(Role::Similarity, "Kenia").await.unwrap();
    let fuzzy = traverse::resolve_entry(&db, "Kenia", Some("country"), Some(&vector)).unwrap();
    assert_eq!(fuzzy.first().map(|n| n.label.as_str()), Some("Kenya"));
    assert!(
        traverse::resolve_entry(&db, "Kenia", None, None)
            .unwrap()
            .is_empty()
    );

    paths_merges_and_listing(&db, &options, &hood, &roots, &current, &pending);
}

fn paths_merges_and_listing(
    db: &WorkspaceDb,
    options: &GraphOptions,
    hood: &quack_core::graph::GraphResult,
    roots: &[quack_core::graph::Node],
    current: &Ontology,
    pending: &[resolve::MergeProposal],
) {
    // Path: Uganda to Aurobindo goes through PO-2, Orgenics, ... no: through
    // PO-2 -> Orgenics -> PO-1 -> Kenya -> PO-3 -> Aurobindo (5 hops), too
    // long for 4; Uganda to Kenya is 4 hops.
    let uganda = traverse::resolve_entry(db, "Uganda", None, None).unwrap();
    let kenya = roots.first().unwrap();
    let path = traverse::path(db, uganda.first().unwrap(), kenya, 4, options).unwrap();
    assert_eq!(path.edges.len(), 4, "{path:?}");
    assert_eq!(path.nodes.first().map(|n| n.label.as_str()), Some("Uganda"));
    assert_eq!(path.nodes.last().map(|n| n.label.as_str()), Some("Kenya"));
    let none = traverse::path(db, uganda.first().unwrap(), kenya, 2, options).unwrap();
    assert!(none.is_empty());

    // By class with subclass expansion: organizations include vendors.
    let orgs = traverse::by_class(db, Some(current), "organization", 50, options).unwrap();
    assert_eq!(orgs.nodes.len(), 3, "{orgs:?}");
    let tree = traverse::render_tree(hood);
    assert!(
        tree.contains("Kenya (country)") && tree.contains("<- delivered_to"),
        "{tree}"
    );

    // Accepting the merge folds Orgenics Ltd into Orgenics: its edge and
    // provenance move, the alias is kept, and the path shortens.
    let proposal = resolve::accept(db, &pending.first().unwrap().id, Some("tester")).unwrap();
    let kept = graph_store::node(db, &proposal.keep.id).unwrap().unwrap();
    assert!(
        kept.properties
            .get("aliases")
            .and_then(|a| a.as_array())
            .is_some_and(|a| a.len() == 1)
    );
    assert!(graph_store::node(db, &proposal.drop.id).unwrap().is_none());
    assert_eq!(graph_store::status(db).unwrap().nodes, 7);
    let path = traverse::path(db, uganda.first().unwrap(), kenya, 4, options).unwrap();
    assert_eq!(
        path.edges.len(),
        3,
        "the merged vendor's ships_to edge shortens it"
    );
    assert!(resolve::pending(db).unwrap().is_empty());
    assert!(resolve::accept(db, &proposal.id, None).is_err());
}

#[tokio::test]
async fn stale_graphs_revalidate_and_provisional_results_are_excluded() {
    let db = workspace();
    let current = store::current(&db).unwrap().unwrap();
    tables::extract(&db, &current, true).unwrap();
    graph_store::set_built_with(&db, current.version).unwrap();
    let status = graph_store::status(&db).unwrap();
    assert!(status.provisional() && !status.stale);

    // Query mode drops provisional results entirely.
    let roots = traverse::resolve_entry(&db, "Kenya", None, None).unwrap();
    let hood = traverse::neighborhood(&db, &roots, 1, None, &GraphOptions::default()).unwrap();
    assert!(!hood.is_empty());
    assert!(hood.without_provisional().is_empty());
    graph_store::mark_reviewed(&db).unwrap();
    assert!(!graph_store::status(&db).unwrap().provisional());

    // Dropping the country class and the delivered_to relation from the
    // ontology makes the graph stale; revalidate removes what no longer
    // fits and records the new version.
    let mut edited = current.clone();
    edited.classes.retain(|c| c.id != "country");
    edited
        .relations
        .retain(|r| r.id != "delivered_to" && r.id != "ships_to");
    if let Some(mapping) = edited.mappings.first_mut() {
        mapping.relations.retain(|r| r.target_class != "country");
    }
    let saved = store::save(&db, &edited, Some("test"), Some("drop countries")).unwrap();
    let status = graph_store::status(&db).unwrap();
    assert!(status.stale);
    let outcome = graph_store::revalidate(&db).unwrap();
    assert_eq!(outcome.dropped_nodes, 2, "Kenya and Uganda");
    assert_eq!(outcome.dropped_edges, 0, "their edges went with them");
    assert_eq!(outcome.version, saved.version);
    let status = graph_store::status(&db).unwrap();
    assert_eq!(status.nodes, 5);
    assert_eq!(status.edges, 3);
    assert!(!status.stale);

    graph_store::clear(&db).unwrap();
    let status = graph_store::status(&db).unwrap();
    assert!(!status.enabled());
    assert_eq!(status.built_with_version, 0);
}

#[test]
fn auto_accepted_ontologies_are_provisional_until_reviewed() {
    let db = WorkspaceDb::open_in_memory(4).unwrap();
    assert!(!store::current_is_auto_accepted(&db).unwrap());
    store::save(
        &db,
        &Ontology::builtin_default(),
        None,
        Some("auto-accepted 3 candidate(s)"),
    )
    .unwrap();
    assert!(store::current_is_auto_accepted(&db).unwrap());
    store::save(&db, &Ontology::builtin_default(), None, Some("reviewed")).unwrap();
    assert!(!store::current_is_auto_accepted(&db).unwrap());
}

/// A class listing says how many nodes it left behind, and the census
/// counts them without a traversal — the count user SQL cannot get, since
/// `_quack_` tables are closed to it.
#[test]
fn class_listings_report_the_total_they_were_capped_from() {
    let db = workspace();
    for i in 0..12 {
        graph_store::upsert_node(
            &db,
            &NewNode {
                label: format!("Country {i:02}"),
                class_id: String::from("country"),
                properties: serde_json::json!({}),
                provisional: false,
            },
        )
        .unwrap();
    }
    let current = ontology();
    let options = GraphOptions {
        max_nodes: 5,
        ..GraphOptions::default()
    };

    let capped = traverse::by_class(&db, Some(&current), "country", 5, &options).unwrap();
    assert_eq!(capped.nodes.len(), 5);
    assert_eq!(capped.total_nodes, Some(12));
    assert!(capped.truncated);
    let tree = traverse::render_tree(&capped);
    assert!(tree.contains("5 of 12 matching nodes"), "{tree}");
    assert!(tree.contains("cut off at the node limit"), "{tree}");

    let whole =
        traverse::by_class(&db, Some(&current), "country", 50, &GraphOptions::default()).unwrap();
    assert_eq!(whole.nodes.len(), 12);
    assert!(!whole.truncated);
    assert!(!traverse::render_tree(&whole).contains("cut off"), "{tree}");

    // The census counts the class and its subclasses without listing them.
    let (total, samples) = graph_store::class_census(&db, &[String::from("country")], 3).unwrap();
    assert_eq!(total, 12);
    assert_eq!(samples, ["Country 00", "Country 01", "Country 02"]);
    let (none, _) = graph_store::class_census(&db, &[String::from("vendor")], 3).unwrap();
    assert_eq!(none, 0);
}

/// Provenance joins the two substrates both ways: an entity names the
/// chunks it was extracted from, and a chunk names the entities in it.
#[tokio::test]
async fn provenance_maps_between_entities_and_chunks() {
    let db = workspace();
    let writer = writer_of(&db);
    let current = ontology();
    tables::extract(&db, &current, false).unwrap();
    let chunks = extract::chunks(&db, None).unwrap();
    extract::run(&writer, chunks, &Canned, &current, false, 2, &|_| {})
        .await
        .unwrap();

    let orgenics = traverse::resolve_entry(&db, "Orgenics Ltd", None, None).unwrap();
    let ids: Vec<String> = orgenics.iter().map(|n| n.id.clone()).collect();
    assert!(
        !ids.is_empty(),
        "the canned extractor produces Orgenics Ltd"
    );
    let chunks = graph_store::chunks_of_nodes(&db, &ids).unwrap();
    assert_eq!(chunks, ["c1"], "extracted from the one chunk that parsed");

    // A node built from a table row has no chunk provenance at all.
    let po = traverse::resolve_entry(&db, "PO-1", None, None).unwrap();
    let po_ids: Vec<String> = po.iter().map(|n| n.id.clone()).collect();
    assert!(
        graph_store::chunks_of_nodes(&db, &po_ids)
            .unwrap()
            .is_empty(),
        "table rows leave table provenance, not chunks"
    );

    let entities = graph_store::entities_of_chunks(&db, &[String::from("c1")], 8).unwrap();
    let in_c1 = entities.get("c1").cloned().unwrap_or_default();
    assert!(
        in_c1.contains(&String::from("Orgenics Ltd (vendor)")),
        "{in_c1:?}"
    );
    assert!(
        in_c1.contains(&String::from("Kenya (country)")),
        "{in_c1:?}"
    );
    // Bounded per chunk, and a chunk nothing was extracted from is absent.
    let capped = graph_store::entities_of_chunks(&db, &[String::from("c1")], 1).unwrap();
    assert_eq!(capped.get("c1").map(Vec::len), Some(1));
    assert!(
        !graph_store::entities_of_chunks(&db, &[String::from("c2")], 8)
            .unwrap()
            .contains_key("c2")
    );
}

/// A lookup that matched nothing offers the labels it came closest to —
/// by overlap either way, then by embedding — so the model can call again
/// instead of being told the graph is empty.
#[tokio::test]
async fn a_missed_lookup_suggests_the_labels_that_exist() {
    let db = workspace();
    for (label, class_id) in [
        ("Orgenics Ltd", "vendor"),
        ("Kenya", "country"),
        ("Uganda", "country"),
    ] {
        graph_store::upsert_node(
            &db,
            &NewNode {
                label: String::from(label),
                class_id: String::from(class_id),
                properties: serde_json::json!({}),
                provisional: false,
            },
        )
        .unwrap();
    }

    // The query inside a label, and a label inside the query.
    assert_eq!(
        traverse::suggest_entities(&db, "Orgenics", None, None).unwrap(),
        ["Orgenics Ltd (vendor)"]
    );
    assert_eq!(
        traverse::suggest_entities(&db, "Kenya Ports Authority", None, None).unwrap(),
        ["Kenya (country)"]
    );

    // A typo the overlap test cannot catch: one dropped or changed letter
    // still suggests the real label (issue #92, seen live on OKLAHOMS).
    assert_eq!(
        traverse::suggest_entities(&db, "Ugandq", None, None).unwrap(),
        ["Uganda (country)"]
    );
    assert_eq!(
        traverse::suggest_entities(&db, "Kenyaa", None, None).unwrap(),
        ["Kenya (country)"]
    );

    // Held to the class when one was given, and empty when nothing is close.
    assert!(
        traverse::suggest_entities(&db, "Orgenics", Some("country"), None)
            .unwrap()
            .is_empty()
    );
    assert!(
        traverse::suggest_entities(&db, "Helsinki", None, None)
            .unwrap()
            .is_empty()
    );
    assert!(
        traverse::suggest_entities(&db, "   ", None, None)
            .unwrap()
            .is_empty()
    );

    // With an embedding, a label sharing no text still comes back: these
    // are the matches `resolve_entry` rejected as too far to be the entity.
    for node in graph_store::nodes_needing_embedding(&db, 10).unwrap() {
        let label = letters().one(Role::Similarity, &node.label).await.unwrap();
        db.set_node_embedding(&node.id, &label).unwrap();
    }
    let query = letters().one(Role::Similarity, "Kampala").await.unwrap();
    let suggestions = traverse::suggest_entities(&db, "Kampala", None, Some(&query)).unwrap();
    assert!(
        suggestions.contains(&String::from("Kenya (country)")),
        "{suggestions:?}"
    );
}
