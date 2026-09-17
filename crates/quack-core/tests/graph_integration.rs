#![expect(clippy::unwrap_used, reason = "test assertions use unwrap throughout")]

//! The knowledge graph end to end inside one workspace file: table mapping
//! extraction, constrained extraction with a canned model, resolution,
//! traversal, revalidation, provisional handling, and merge review.

use std::collections::BTreeMap;

use quack_core::graph::extract::{Extraction, GraphExtractor};
use quack_core::graph::{GraphOptions, extract, resolve, store as graph_store, tables, traverse};
use quack_core::ontology::{self, Class, Mapping, MappingRelation, Ontology, Relation, store};
use quack_core::storage::workspace::{NewChunk, NewDocument, WorkspaceDb};
use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};

const DIM: usize = 4;

/// Labels sharing a first letter embed close together; others far apart.
struct LetterEmbedding;

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

#[tokio::test]
async fn tables_documents_resolution_and_traversal_end_to_end() {
    let db = workspace();
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
    let summary = extract::run(&db, chunks, &Canned, &current, false)
        .await
        .unwrap();
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
    let resolved = resolve::resolve(&db, Some(&LetterEmbedding), &options)
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
    let embedding = LetterEmbedding.embed_text("Kenia").await.unwrap();
    // The fixture embeds 0, 1, or a hundredth: exact in f32.
    let vector: Vec<f32> = embedding
        .vec
        .iter()
        .map(|v| {
            if *v >= 1.0 {
                1.0
            } else if *v > 0.0 {
                0.01
            } else {
                0.0
            }
        })
        .collect();
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
