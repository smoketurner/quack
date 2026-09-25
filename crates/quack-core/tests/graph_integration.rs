#![expect(clippy::unwrap_used, reason = "test assertions use unwrap throughout")]

//! The knowledge graph end to end inside one workspace file: table mapping
//! extraction, constrained extraction with a canned model, resolution,
//! traversal, revalidation, provisional handling, and merge review.

use std::collections::BTreeMap;

use quack_core::embedding::{Dimension, Embedder, Input, Profile, Prompts, Vector};
use quack_core::error::Error;
use quack_core::extraction::{Extract, ExtractFuture, ExtractionRun};
use quack_core::graph::extract::{ChunkPlan, Extraction};
use quack_core::graph::resolve::MergeDecision;
use quack_core::graph::store::NewNode;
use quack_core::graph::traverse::Hops;
use quack_core::graph::{
    GraphOptions, GraphResult, Node, Origin, Properties, Standing, extract, resolve,
    store as graph_store, tables, traverse,
};
use quack_core::ids::{ChunkId, ClassId, DocumentId, NodeId, RelationId};
use quack_core::ontology::store::Revision;
use quack_core::ontology::{self, Class, Mapping, MappingRelation, Ontology, Relation, store};
use quack_core::progress::RunControl;
use quack_core::storage::workspace::{
    DocumentStatus, NewChunk, NewDocument, SamplePool, WorkspaceDb,
};
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

/// A name's `LetterEmbedding` vector, compared with labels.
async fn name_vector(name: &str) -> Vector {
    letters()
        .embed_one(&Input::Similarity(name.to_owned()))
        .await
        .unwrap()
}

/// `LetterEmbedding` with no prefixes, so a label reaches it unchanged.
fn letters() -> Embedder<LetterEmbedding> {
    Embedder::new(
        LetterEmbedding,
        Profile::new("letters", Dimension::new(4), Prompts::default()),
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
        id: ClassId::from(id.to_owned()),
        parent: ClassId::from(parent.to_owned()),
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
        id: RelationId::from(id.to_owned()),
        label: None,
        description: None,
        domain: ClassId::from(domain.to_owned()),
        range: ClassId::from(range.to_owned()),
    };
    Ontology {
        version: None,
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
            class: ClassId::from("shipment"),
            key: String::from("po"),
            properties: BTreeMap::from([(String::from("mode"), String::from("mode"))]),
            relations: vec![
                MappingRelation {
                    relation: RelationId::from("supplied_by"),
                    column: String::from("vendor"),
                    target_class: ClassId::from("vendor"),
                    target_key: String::from("name"),
                },
                MappingRelation {
                    relation: RelationId::from("delivered_to"),
                    column: String::from("country"),
                    target_class: ClassId::from("country"),
                    target_key: String::from("name"),
                },
            ],
        }],
    }
}

fn workspace() -> WorkspaceDb {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    db.execute_statement(
        "CREATE TABLE shipments AS SELECT * FROM (VALUES \
         ('PO-1', 'Orgenics', 'Kenya', 'Air'), \
         ('PO-2', 'Orgenics', 'Uganda', 'Sea'), \
         ('PO-3', 'Aurobindo', 'Kenya', NULL), \
         (NULL, 'Nobody', 'Nowhere', 'Air')) AS t(po, vendor, country, mode)",
    )
    .unwrap();
    db.insert_document(
        &NewDocument::new(&DocumentId::from("doc-1"), "notes.md", "text/markdown", 10)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: &ChunkId::from("c1"),
        document_id: &DocumentId::from("doc-1"),
        chunk_index: 0,
        content: "Orgenics ships to Kenya from its plant.",
        heading: Some("Vendors"),
        page: None,
        embedding: None,
    })
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: &ChunkId::from("c2"),
        document_id: &DocumentId::from("doc-1"),
        chunk_index: 1,
        content: "FAIL this one",
        heading: None,
        page: None,
        embedding: None,
    })
    .unwrap();
    store::save(
        &db,
        &ontology(),
        Revision::reviewed(Some("test"), Some("fixture")),
    )
    .unwrap();
    db
}

struct Canned;

impl Extract<Extraction> for Canned {
    fn extract<'a>(&'a self, text: &'a str) -> ExtractFuture<'a, Extraction> {
        Box::pin(async move {
            if text.contains("FAIL") {
                return Err(Error::Llm(String::from("boom")));
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
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    let rows = tables::BATCH_ROWS * 2 + 1;
    db.execute_statement(&format!(
        "CREATE TABLE shipments AS SELECT 'PO-' || lpad(range::VARCHAR, 5, '0') AS po, \
         'V' || (range % 3) AS vendor, 'C' || (range % 2) AS country, 'Air' AS mode \
         FROM range({rows})"
    ))
    .unwrap();
    store::save(
        &db,
        &ontology(),
        Revision::reviewed(Some("test"), Some("fixture")),
    )
    .unwrap();
    let current = store::current(&db).unwrap().unwrap();
    let summaries = tables::extract(&db, &current, Standing::Reviewed).unwrap();
    let first = summaries.first().unwrap();
    assert_eq!(
        (first.rows, first.nodes, first.edges),
        (rows, rows, rows * 2)
    );
    let status = graph_store::status(&db).unwrap();
    assert_eq!(status.nodes, u64::from(rows) + 3 + 2);
    // Re-running stays idempotent across batches.
    tables::extract(&db, &current, Standing::Reviewed).unwrap();
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
    let found = traverse::neighborhood(&db, &hub, Hops::new(3), None, &options).unwrap();
    assert_eq!(found.nodes.len(), 7, "{}", found.nodes.len());
}

/// Sampling spreads across documents (issue #60): a second, longer
/// document contributes as many chunks as the first to a sample of 2.
#[test]
fn extraction_samples_evenly_across_documents() {
    let db = workspace();
    db.insert_document(
        &NewDocument::new(&DocumentId::from("doc-2"), "long.md", "text/markdown", 10)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();
    for i in 0..4 {
        db.insert_chunk(&NewChunk {
            id: &ChunkId::from(format!("l{i}")),
            document_id: &DocumentId::from("doc-2"),
            chunk_index: i,
            content: "Filler text about nothing in particular.",
            heading: None,
            page: None,
            embedding: None,
        })
        .unwrap();
    }
    let sampled = db
        .sample_chunk_ids(SamplePool::NotGraphExtracted, 2)
        .unwrap();
    assert_eq!(
        ChunkPlan::new(&db, Some(2)).unwrap(),
        ChunkPlan::Sample(sampled.clone())
    );
    let chunks = db.chunks_by_ids(&sampled).unwrap();
    let docs: std::collections::BTreeSet<&str> =
        chunks.iter().map(|c| c.document_id.as_str()).collect();
    assert_eq!(chunks.len(), 2, "{sampled:?}");
    assert_eq!(docs.len(), 2, "one chunk from each document: {sampled:?}");
    assert!(ChunkPlan::new(&db, Some(0)).unwrap().is_empty());
    assert_eq!(
        ChunkPlan::new(&db, None).unwrap(),
        ChunkPlan::All { total: 6 }
    );
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
    tables::extract(&db, &current, Standing::Reviewed).unwrap();
    let node = |label: &str| NewNode {
        label: label.to_owned(),
        class_id: ClassId::from("country"),
        properties: Properties::default(),
        standing: Standing::Reviewed,
    };
    // "Kenya" (from the table) and "Kenya Coast" embed identically under
    // LetterEmbedding (same first letter, same length mod 3) and share a
    // token: the old pass auto-merged such pairs.
    let coast = graph_store::upsert_node(&db, &node("Kenya Coast")).unwrap();
    graph_store::add_provenance(&db, &coast, &graph_store::Source::row("places", "KC")).unwrap();
    // An extracted look-alike of a keyed node, and two extracted
    // look-alikes of each other.
    let doc = |chunk: &str| {
        graph_store::Source::chunk(&DocumentId::from("doc-1"), &ChunkId::from(chunk), 0.9)
    };
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

/// A rejected merge is not proposed again when a later resolution pass
/// flips its keep/drop orientation (provenance tilt): the dedup must
/// recognize the pair in either orientation (`MergeStatus::Rejected`'s
/// "kept apart; not proposed again" contract).
#[tokio::test]
async fn rejected_merge_is_not_reproposed_when_provenance_flips_orientation() {
    let db = workspace();
    let writer = writer_of(&db);
    let node = |label: &str| NewNode {
        label: label.to_owned(),
        class_id: ClassId::from("acmeorg"),
        properties: Properties::default(),
        standing: Standing::Reviewed,
    };
    let chunk =
        |c: &str| graph_store::Source::chunk(&DocumentId::from("doc-1"), &ChunkId::from(c), 0.9);
    let options = GraphOptions {
        merge_threshold: 0.5,
        auto_merge_threshold: -1.0,
        ..GraphOptions::default()
    };
    let acme = graph_store::upsert_node(&db, &node("Acme")).unwrap();
    graph_store::add_provenance(&db, &acme, &chunk("c1")).unwrap();
    graph_store::add_provenance(&db, &acme, &chunk("c2")).unwrap();
    let acme_corp = graph_store::upsert_node(&db, &node("Acme Corp")).unwrap();
    graph_store::add_provenance(&db, &acme_corp, &chunk("c1")).unwrap();
    db.insert_chunk(&NewChunk {
        id: &ChunkId::from("c3"),
        document_id: &DocumentId::from("doc-1"),
        chunk_index: 2,
        content: "Acme Corp is an acme.",
        heading: None,
        page: None,
        embedding: None,
    })
    .unwrap();
    // First pass: Acme (2 provenance) keeps; Acme Corp (1) drops.
    resolve::resolve(&writer, Some(&letters()), &options)
        .await
        .unwrap();
    let pending = resolve::pending(&db).unwrap();
    assert_eq!(pending.len(), 1, "one proposal for the Acme/Acme Corp pair");
    let first = pending.first().unwrap();
    assert_eq!(first.keep.id, acme, "{first:?}");
    assert_eq!(first.drop.id, acme_corp, "{first:?}");
    // A reviewer rejects the pair.
    resolve::decide(
        &db,
        first.id.as_str(),
        MergeDecision::Reject,
        Some("tester"),
    )
    .unwrap();
    assert!(resolve::pending(&db).unwrap().is_empty());
    // Later ingestion tilts provenance the other way: Acme Corp now leads.
    graph_store::add_provenance(&db, &acme_corp, &chunk("c2")).unwrap();
    graph_store::add_provenance(&db, &acme_corp, &chunk("c3")).unwrap();
    resolve::resolve(&writer, Some(&letters()), &options)
        .await
        .unwrap();
    let repending = resolve::pending(&db).unwrap();
    assert!(
        repending.is_empty(),
        "a rejected pair must not be re-proposed, but got {repending:?}"
    );
}

/// A still-pending pair is not duplicated when a later resolution pass flips
/// its keep/drop orientation: the same node pair appears at most once in
/// the review queue, regardless of orientation.
#[tokio::test]
async fn pending_pair_is_not_duplicated_when_provenance_flips_orientation() {
    let db = workspace();
    let writer = writer_of(&db);
    let node = |label: &str| NewNode {
        label: label.to_owned(),
        class_id: ClassId::from("acmeorg"),
        properties: Properties::default(),
        standing: Standing::Reviewed,
    };
    let chunk =
        |c: &str| graph_store::Source::chunk(&DocumentId::from("doc-1"), &ChunkId::from(c), 0.9);
    let options = GraphOptions {
        merge_threshold: 0.5,
        auto_merge_threshold: -1.0,
        ..GraphOptions::default()
    };
    let acme = graph_store::upsert_node(&db, &node("Acme")).unwrap();
    graph_store::add_provenance(&db, &acme, &chunk("c1")).unwrap();
    graph_store::add_provenance(&db, &acme, &chunk("c2")).unwrap();
    let acme_corp = graph_store::upsert_node(&db, &node("Acme Corp")).unwrap();
    graph_store::add_provenance(&db, &acme_corp, &chunk("c1")).unwrap();
    db.insert_chunk(&NewChunk {
        id: &ChunkId::from("c3"),
        document_id: &DocumentId::from("doc-1"),
        chunk_index: 2,
        content: "Acme Corp is an acme.",
        heading: None,
        page: None,
        embedding: None,
    })
    .unwrap();
    resolve::resolve(&writer, Some(&letters()), &options)
        .await
        .unwrap();
    assert_eq!(resolve::pending(&db).unwrap().len(), 1);
    // A second pass after provenance tilts the other way must not add a
    // second row for the same pair in the swapped orientation.
    graph_store::add_provenance(&db, &acme_corp, &chunk("c2")).unwrap();
    graph_store::add_provenance(&db, &acme_corp, &chunk("c3")).unwrap();
    resolve::resolve(&writer, Some(&letters()), &options)
        .await
        .unwrap();
    let pending = resolve::pending(&db).unwrap();
    assert_eq!(
        pending.len(),
        1,
        "the same pair must not produce duplicate proposals, but got {pending:?}"
    );
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
    tables::extract(&db, &current, Standing::Reviewed).unwrap();
    assert_eq!(graph_store::status(&db).unwrap().nodes, 7);

    // A second document adds one node of its own, one edge of its own,
    // and a second source for a vendor the table already produced.
    db.insert_document(
        &NewDocument::new(&DocumentId::from("doc-2"), "extra.md", "text/markdown", 10)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();
    db.insert_chunk(&NewChunk {
        id: &ChunkId::from("c3"),
        document_id: &DocumentId::from("doc-2"),
        chunk_index: 0,
        content: "Orgenics ships to Nowhere.",
        heading: None,
        page: None,
        embedding: None,
    })
    .unwrap();
    let node = |label: &str, class: &str| NewNode {
        label: label.to_owned(),
        class_id: ClassId::from(class.to_owned()),
        properties: Properties::default(),
        standing: Standing::Reviewed,
    };
    let source = graph_store::Source::chunk(&DocumentId::from("doc-2"), &ChunkId::from("c3"), 0.9);
    let orgenics = graph_store::upsert_node(&db, &node("Orgenics", "vendor")).unwrap();
    graph_store::add_provenance(&db, &orgenics, &source).unwrap();
    let nowhere = graph_store::upsert_node(&db, &node("Nowhere", "country")).unwrap();
    graph_store::add_provenance(&db, &nowhere, &source).unwrap();
    let edge = graph_store::upsert_edge(
        &db,
        &orgenics,
        &nowhere,
        "ships_to",
        &Properties::default(),
        Standing::Reviewed,
    )
    .unwrap();
    graph_store::add_provenance(&db, &edge, &source).unwrap();
    assert_eq!(graph_store::status(&db).unwrap().nodes, 8);

    assert!(db.delete_document(&DocumentId::from("doc-2")).unwrap());
    let status = graph_store::status(&db).unwrap();
    assert_eq!((status.nodes, status.edges), (7, 6));
    assert!(graph_store::node(&db, &nowhere).unwrap().is_none());
    assert!(graph_store::node(&db, &orgenics).unwrap().is_some());
    assert!(
        graph_store::provenance_of(&db, std::slice::from_ref(&orgenics))
            .unwrap()
            .iter()
            .all(|p| !matches!(
                p.origin,
                Origin::Chunk {
                    document_id: Some(_),
                    ..
                }
            ))
    );

    // The document that loaded the mapped table: the table drops, and
    // with it every node and edge the rows supported.
    db.insert_document(
        &NewDocument::new(&DocumentId::from("doc-t"), "shipments.csv", "text/csv", 10)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();
    db.set_document_tables(&DocumentId::from("doc-t"), &[String::from("shipments")])
        .unwrap();
    assert!(db.delete_document(&DocumentId::from("doc-t")).unwrap());
    assert!(db.list_tables().unwrap().is_empty());
    let status = graph_store::status(&db).unwrap();
    assert_eq!((status.nodes, status.edges), (0, 0));
    assert_eq!(status.missing_tables, vec![String::from("shipments")]);
    let orphans: i64 = db
        .connection()
        .query_row("SELECT count(*) FROM _quack_provenance", [], |r| r.get(0))
        .unwrap();
    assert_eq!(orphans, 0);

    let summaries = tables::extract(&db, &current, Standing::Reviewed).unwrap();
    assert_eq!(summaries.len(), 1);
    assert!(summaries.first().unwrap().skipped.is_some());
    assert!(
        store::save(
            &db,
            &current,
            Revision::reviewed(Some("test"), Some("still saves"))
        )
        .is_ok()
    );
}

#[tokio::test]
async fn tables_documents_resolution_and_traversal_end_to_end() {
    let db = workspace();
    let writer = writer_of(&db);
    let current = store::current(&db).unwrap().unwrap();

    // Table mapping: three keyed rows become shipments with edges to
    // vendors and countries; the NULL-keyed row is skipped.
    let summaries = tables::extract(&db, &current, Standing::Reviewed).unwrap();
    assert_eq!(summaries.len(), 1);
    let first = summaries.first().unwrap();
    assert_eq!((first.rows, first.nodes, first.edges), (3, 3, 6));
    let status = graph_store::status(&db).unwrap();
    assert_eq!(status.nodes, 3 + 2 + 2, "shipments, vendors, countries");
    assert_eq!(status.edges, 6);
    assert!(status.enabled() && !status.provisional());
    // Re-running is idempotent.
    tables::extract(&db, &current, Standing::Reviewed).unwrap();
    assert_eq!(graph_store::status(&db).unwrap().nodes, 7);

    // Constrained extraction: the vessel and docked_at are drift, the
    // failed chunk is skipped, Kenya merges with the table's Kenya.
    let plan = ChunkPlan::new(&db, None).unwrap();
    assert_eq!(plan.len(), 2);
    let summary = extract::run(
        &writer,
        &plan,
        &current,
        Standing::Reviewed,
        ExtractionRun {
            extractor: &Canned,
            concurrency: 2,
            control: RunControl::unobserved(),
        },
    )
    .await
    .unwrap();
    // Both chunks are on record (the failed one is not), so the next run
    // sends only the failed one again.
    assert_eq!(graph_store::extracted_chunks(&db).unwrap(), 1);
    let remaining = db
        .chunk_page(SamplePool::NotGraphExtracted, None, 10)
        .unwrap();
    assert_eq!(remaining.len(), 1, "{remaining:?}");
    assert_eq!(remaining.first().map(|c| c.id.as_str()), Some("c2"));
    assert_eq!(
        (
            summary.chunks,
            summary.failed_chunks,
            summary.nodes,
            summary.edges
        ),
        (2, 1, 2, 1)
    );
    assert_eq!(summary.drift.classes.get("vessel"), 1);
    assert_eq!(summary.drift.relations.get("docked_at"), 1);
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
    let hood = traverse::neighborhood(&db, &roots, Hops::new(2), None, &options).unwrap();
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
            .any(|p| p.origin.chunk_id() == Some(&ChunkId::from("c1")))
    );
    assert!(
        hood.provenance.iter().any(
            |p| matches!(&p.origin, Origin::Row { table_name, .. } if table_name == "shipments")
        )
    );
    let only =
        traverse::neighborhood(&db, &roots, Hops::new(2), Some("delivered_to"), &options).unwrap();
    assert!(only.edges.iter().all(|e| e.relation_id == "delivered_to"));

    // Fuzzy entry: an unknown spelling resolves through the embedding.
    let vector = name_vector("Kenia").await;
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
    hood: &GraphResult,
    roots: &[Node],
    current: &Ontology,
    pending: &[resolve::MergeProposal],
) {
    // Path: Uganda to Aurobindo goes through PO-2, Orgenics, ... no: through
    // PO-2 -> Orgenics -> PO-1 -> Kenya -> PO-3 -> Aurobindo (5 hops), too
    // long for 4; Uganda to Kenya is 4 hops.
    let uganda = traverse::resolve_entry(db, "Uganda", None, None).unwrap();
    let kenya = roots.first().unwrap();
    let path = traverse::path(db, uganda.first().unwrap(), kenya, Hops::new(4), options).unwrap();
    assert_eq!(path.edges.len(), 4, "{path:?}");
    assert_eq!(path.nodes.first().map(|n| n.label.as_str()), Some("Uganda"));
    assert_eq!(path.nodes.last().map(|n| n.label.as_str()), Some("Kenya"));
    let none = traverse::path(db, uganda.first().unwrap(), kenya, Hops::new(2), options).unwrap();
    assert!(none.is_empty());

    // By class with subclass expansion: organizations include vendors.
    let orgs = traverse::by_class(db, Some(current), "organization", 50, options).unwrap();
    assert_eq!(orgs.nodes.len(), 3, "{orgs:?}");
    let tree = hood.to_string();
    assert!(
        tree.contains("Kenya (country)") && tree.contains("<- delivered_to"),
        "{tree}"
    );

    // Accepting the merge folds Orgenics Ltd into Orgenics: its edge and
    // provenance move, the alias is kept, and the path shortens.
    let proposal = resolve::decide(
        db,
        pending.first().unwrap().id.as_str(),
        MergeDecision::Accept,
        Some("tester"),
    )
    .unwrap();
    let kept = graph_store::node(db, &proposal.keep.id).unwrap().unwrap();
    assert!(
        kept.properties
            .get("aliases")
            .and_then(|a| a.as_array())
            .is_some_and(|a| a.len() == 1)
    );
    // The absorbed label is now an alias; entry-point lookup matches it the
    // same way as the primary label — case- and whitespace-insensitively.
    // The terminal never passes an embedder, so this is the path it takes.
    assert!(
        kept.properties
            .aliases()
            .contains(&String::from("Orgenics Ltd")),
        "{:?}",
        kept.properties.aliases()
    );
    assert_eq!(
        traverse::resolve_entry(db, "orgenics ltd", None, None)
            .unwrap()
            .first()
            .map(|n| n.label.as_str()),
        Some(kept.label.as_str()),
        "a case-differing alias resolves to the kept node"
    );
    assert_eq!(
        traverse::resolve_entry(db, "ORGENICS  ltd", None, None)
            .unwrap()
            .first()
            .map(|n| n.label.as_str()),
        Some(kept.label.as_str()),
        "a whitespace-differing alias resolves to the kept node"
    );
    assert!(graph_store::node(db, &proposal.drop.id).unwrap().is_none());
    assert_eq!(graph_store::status(db).unwrap().nodes, 7);
    let path = traverse::path(db, uganda.first().unwrap(), kenya, Hops::new(4), options).unwrap();
    assert_eq!(
        path.edges.len(),
        3,
        "the merged vendor's ships_to edge shortens it"
    );
    assert!(resolve::pending(db).unwrap().is_empty());
    assert!(resolve::decide(db, proposal.id.as_str(), MergeDecision::Accept, None).is_err());
}

#[tokio::test]
async fn stale_graphs_revalidate_and_provisional_results_are_excluded() {
    let db = workspace();
    let current = store::current(&db).unwrap().unwrap();
    tables::extract(&db, &current, Standing::Provisional).unwrap();
    graph_store::set_built_with(&db, current.saved_version().unwrap()).unwrap();
    let status = graph_store::status(&db).unwrap();
    assert!(status.provisional() && !status.stale);

    // Query mode drops provisional results entirely.
    let roots = traverse::resolve_entry(&db, "Kenya", None, None).unwrap();
    let hood =
        traverse::neighborhood(&db, &roots, Hops::new(1), None, &GraphOptions::default()).unwrap();
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
    let saved = store::save(
        &db,
        &edited,
        Revision::reviewed(Some("test"), Some("drop countries")),
    )
    .unwrap();
    let status = graph_store::status(&db).unwrap();
    assert!(status.stale);
    let outcome = graph_store::revalidate(&db).unwrap();
    assert_eq!(outcome.dropped_nodes, 2, "Kenya and Uganda");
    assert_eq!(outcome.dropped_edges, 0, "their edges went with them");
    assert_eq!(Some(outcome.version), saved.version);
    let status = graph_store::status(&db).unwrap();
    assert_eq!(status.nodes, 5);
    assert_eq!(status.edges, 3);
    assert!(!status.stale);

    graph_store::clear(&db).unwrap();
    let status = graph_store::status(&db).unwrap();
    assert!(!status.enabled());
    assert_eq!(status.built_with_version, None);
}

/// Revalidation drops edges whose relation no longer fits their ends,
/// checked once per relation and class combination, and edges left
/// dangling, with their provenance, and keeps every node.
#[test]
fn revalidation_drops_edges_that_no_longer_fit_and_dangling_ones() {
    let db = workspace();
    let current = store::current(&db).unwrap().unwrap();
    tables::extract(&db, &current, Standing::Reviewed).unwrap();
    let before = graph_store::status(&db).unwrap();

    // An edge between two nodes that do not exist, with provenance.
    let dangling = graph_store::upsert_edge(
        &db,
        &NodeId::from("no-such-source"),
        &NodeId::from("no-such-target"),
        "mentions",
        &Properties::default(),
        Standing::Reviewed,
    )
    .unwrap();
    graph_store::add_provenance(&db, &dangling, &graph_store::Source::row("shipments", "x"))
        .unwrap();

    // `supplied_by` now ends at a country, so every shipment -> vendor
    // edge of it no longer fits.
    let mut edited = current.clone();
    for relation in &mut edited.relations {
        if relation.id == "supplied_by" {
            relation.range = ClassId::from("country");
        }
    }
    if let Some(mapping) = edited.mappings.first_mut() {
        mapping.relations.retain(|r| r.relation != "supplied_by");
    }
    store::save(
        &db,
        &edited,
        Revision::reviewed(None, Some("supplied_by moved")),
    )
    .unwrap();

    let outcome = graph_store::revalidate(&db).unwrap();
    assert_eq!(outcome.dropped_nodes, 0);
    assert_eq!(
        outcome.dropped_edges, 4,
        "three supplied_by edges and the dangling one"
    );
    let after = graph_store::status(&db).unwrap();
    assert_eq!(after.nodes, before.nodes);
    assert_eq!(after.edges, before.edges - 3);
    assert!(
        graph_store::provenance_of(&db, &[dangling])
            .unwrap()
            .is_empty(),
        "the dangling edge's provenance went with it"
    );
}

#[test]
fn auto_accepted_ontologies_are_provisional_until_reviewed() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    assert_eq!(store::current_standing(&db).unwrap(), Standing::Reviewed);
    store::save(
        &db,
        &Ontology::builtin_default(),
        Revision::auto(None, Some("auto-accepted 3 candidate(s)")),
    )
    .unwrap();
    assert_eq!(store::current_standing(&db).unwrap(), Standing::Provisional);
    store::save(
        &db,
        &Ontology::builtin_default(),
        Revision::reviewed(None, Some("reviewed")),
    )
    .unwrap();
    assert_eq!(store::current_standing(&db).unwrap(), Standing::Reviewed);
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
                class_id: ClassId::from("country"),
                properties: Properties::default(),
                standing: Standing::Reviewed,
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
    let tree = capped.to_string();
    assert!(tree.contains("5 of 12 matching nodes"), "{tree}");
    assert!(tree.contains("cut off at the node limit"), "{tree}");

    let whole =
        traverse::by_class(&db, Some(&current), "country", 50, &GraphOptions::default()).unwrap();
    assert_eq!(whole.nodes.len(), 12);
    assert!(!whole.truncated);
    assert!(!whole.to_string().contains("cut off"), "{tree}");

    // The census counts the class and its subclasses without listing them.
    let census = graph_store::class_census(&db, &[ClassId::from("country")], 3).unwrap();
    assert_eq!(census.total, 12);
    assert_eq!(census.samples, ["Country 00", "Country 01", "Country 02"]);
    let none = graph_store::class_census(&db, &[ClassId::from("vendor")], 3).unwrap();
    assert_eq!(none.total, 0);
}

/// A run over more chunks than one page reads every chunk once, page by
/// page: the fixture's two plus 150 more, three pages of 64.
#[tokio::test]
async fn an_extraction_run_reads_its_chunks_a_page_at_a_time() {
    let db = workspace();
    db.insert_document(
        &NewDocument::new(&DocumentId::from("doc-2"), "long.md", "text/markdown", 10)
            .with_status(DocumentStatus::Ready),
    )
    .unwrap();
    for i in 0..150 {
        db.insert_chunk(&NewChunk {
            id: &ChunkId::from(format!("l{i:03}")),
            document_id: &DocumentId::from("doc-2"),
            chunk_index: i,
            content: "Filler text about nothing in particular.",
            heading: None,
            page: None,
            embedding: None,
        })
        .unwrap();
    }
    let writer = writer_of(&db);
    let current = store::current(&db).unwrap().unwrap();
    let plan = ChunkPlan::new(&db, None).unwrap();
    assert_eq!(plan, ChunkPlan::All { total: 152 });
    let seen = std::sync::Mutex::new(Vec::new());
    let progress = |done: quack_core::progress::ChunkDone| {
        seen.lock().unwrap().push(done.done);
    };
    let control = RunControl {
        progress: &progress,
        cancel: None,
    };
    let summary = extract::run(
        &writer,
        &plan,
        &current,
        Standing::Reviewed,
        ExtractionRun {
            extractor: &Canned,
            concurrency: 4,
            control,
        },
    )
    .await
    .unwrap();
    assert_eq!((summary.chunks, summary.failed_chunks), (152, 1));
    assert_eq!(graph_store::extracted_chunks(&db).unwrap(), 151);
    assert_eq!(seen.into_inner().unwrap(), (1..=152).collect::<Vec<u32>>());

    // A sample reads only its chunks, across the same pages. Each document
    // gets a quota of 50, which the two-chunk one cannot fill.
    graph_store::clear(&db).unwrap();
    let plan = ChunkPlan::new(&db, Some(100)).unwrap();
    assert_eq!(plan.len(), 52);
    let summary = extract::run(
        &writer,
        &plan,
        &current,
        Standing::Reviewed,
        ExtractionRun {
            extractor: &Canned,
            concurrency: 4,
            control: RunControl::unobserved(),
        },
    )
    .await
    .unwrap();
    assert_eq!(summary.chunks, 52);
    assert_eq!(
        graph_store::extracted_chunks(&db).unwrap(),
        52 - u64::from(summary.failed_chunks)
    );
}

/// Provenance joins the two substrates both ways: an entity names the
/// chunks it was extracted from, and a chunk names the entities in it.
#[tokio::test]
async fn provenance_maps_between_entities_and_chunks() {
    let db = workspace();
    let writer = writer_of(&db);
    let current = store::current(&db).unwrap().unwrap();
    tables::extract(&db, &current, Standing::Reviewed).unwrap();
    let plan = ChunkPlan::new(&db, None).unwrap();
    extract::run(
        &writer,
        &plan,
        &current,
        Standing::Reviewed,
        ExtractionRun {
            extractor: &Canned,
            concurrency: 2,
            control: RunControl::unobserved(),
        },
    )
    .await
    .unwrap();

    let orgenics = traverse::resolve_entry(&db, "Orgenics Ltd", None, None).unwrap();
    let ids: Vec<NodeId> = orgenics.iter().map(|n| n.id.clone()).collect();
    assert!(
        !ids.is_empty(),
        "the canned extractor produces Orgenics Ltd"
    );
    let chunks = graph_store::chunks_of_nodes(&db, &ids).unwrap();
    assert_eq!(
        chunks,
        [ChunkId::from("c1")],
        "extracted from the one chunk that parsed"
    );

    // A node built from a table row has no chunk provenance at all.
    let po = traverse::resolve_entry(&db, "PO-1", None, None).unwrap();
    let po_ids: Vec<NodeId> = po.iter().map(|n| n.id.clone()).collect();
    assert!(
        graph_store::chunks_of_nodes(&db, &po_ids)
            .unwrap()
            .is_empty(),
        "table rows leave table provenance, not chunks"
    );

    let entities = graph_store::entities_of_chunks(&db, &[ChunkId::from("c1")], 8).unwrap();
    let in_c1 = entities
        .get(&ChunkId::from("c1"))
        .cloned()
        .unwrap_or_default();
    assert!(
        in_c1.contains(&String::from("Orgenics Ltd (vendor)")),
        "{in_c1:?}"
    );
    assert!(
        in_c1.contains(&String::from("Kenya (country)")),
        "{in_c1:?}"
    );
    // Bounded per chunk, and a chunk nothing was extracted from is absent.
    let capped = graph_store::entities_of_chunks(&db, &[ChunkId::from("c1")], 1).unwrap();
    assert_eq!(capped.get(&ChunkId::from("c1")).map(Vec::len), Some(1));
    assert!(
        !graph_store::entities_of_chunks(&db, &[ChunkId::from("c2")], 8)
            .unwrap()
            .contains_key(&ChunkId::from("c2"))
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
                class_id: ClassId::from(String::from(class_id)),
                properties: Properties::default(),
                standing: Standing::Reviewed,
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
        db.set_node_embedding(&node.id, &name_vector(&node.label).await)
            .unwrap();
    }
    let query = name_vector("Kampala").await;
    let suggestions = traverse::suggest_entities(&db, "Kampala", None, Some(&query)).unwrap();
    assert!(
        suggestions.contains(&String::from("Kenya (country)")),
        "{suggestions:?}"
    );
}

/// Entry-point lookup matches an alias the same way it matches a primary
/// label: case- and whitespace-insensitively. Aliases are stored raw (the
/// absorbed node's label, case and inner whitespace preserved), so a typed
/// alias that differs only in casing or whitespace must still resolve — the
/// terminal reproduces this unconditionally, since it never passes an
/// embedder and so never reaches the embedding fallback.
#[test]
fn aliases_resolve_case_and_whitespace_insensitively_like_primary_labels() {
    let db = workspace();
    let node = |label: &str, class: &str, aliases: &[&str]| {
        let mut properties = Properties::default();
        if !aliases.is_empty() {
            properties.set_aliases(
                &aliases
                    .iter()
                    .copied()
                    .map(String::from)
                    .collect::<Vec<_>>(),
            );
        }
        graph_store::upsert_node(
            &db,
            &NewNode {
                label: String::from(label),
                class_id: ClassId::from(String::from(class)),
                properties,
                standing: Standing::Reviewed,
            },
        )
        .unwrap()
    };
    // An abbreviation: the alias is textually divergent from the label, so
    // `suggest_entities` (which scores against the *label*) cannot rescue it.
    node("IBM", "vendor", &["International Business Machines"]);
    // Token-sharing trading names (the auto-merge path would consider these),
    // also divergent from the kept label.
    node("Northwind Trading Co", "vendor", &["Pacific Trading Group"]);
    // A node with no aliases, and a node of another class.
    node("Acme", "vendor", &[]);
    node("Kenya", "country", &["Republic of Kenya"]);

    let label_of = |entity: &str, class: Option<&str>| -> String {
        traverse::resolve_entry(&db, entity, class, None)
            .unwrap()
            .first()
            .map(|n| n.label.clone())
            .unwrap_or_default()
    };

    // Case-differing alias resolves, no embedder (the terminal's worst case).
    assert_eq!(label_of("international business machines", None), "IBM");
    // All-caps.
    assert_eq!(label_of("INTERNATIONAL BUSINESS MACHINES", None), "IBM");
    // Inner-whitespace collapse.
    assert_eq!(label_of("International  Business   Machines", None), "IBM");
    // Surrounding whitespace.
    assert_eq!(label_of(" International Business Machines ", None), "IBM");
    // Mixed-case alias of the token-sharing pair.
    assert_eq!(
        label_of("PaCiFiC TrAdInG GrOuP", None),
        "Northwind Trading Co"
    );
    // A multi-word alias with case and whitespace differences.
    assert_eq!(label_of("republic   of  KENYA", None), "Kenya");

    // Exact-case alias still resolves (no regression on the alias arm).
    assert_eq!(label_of("International Business Machines", None), "IBM");

    // The class filter applies to the alias arm too: asking the country class
    // for a vendor's alias finds nothing; asking the vendor class finds it.
    assert!(
        traverse::resolve_entry(
            &db,
            "International Business Machines",
            Some("country"),
            None
        )
        .unwrap()
        .is_empty()
    );
    assert_eq!(
        label_of("international business machines", Some("vendor")),
        "IBM"
    );

    // Primary labels still resolve case-insensitively (no regression on the
    // label arm), and an alias of one class does not resolve another class's
    // node.
    assert_eq!(label_of("ibm", None), "IBM");
    assert_eq!(
        label_of("NORTHWIND TRADING CO", None),
        "Northwind Trading Co"
    );
    assert_eq!(label_of("republic of kenya", Some("country")), "Kenya");
    // The vendor's alias is not a country and vice versa.
    assert!(
        traverse::resolve_entry(&db, "Republic of Kenya", Some("vendor"), None)
            .unwrap()
            .is_empty()
    );

    // A name that is neither a label nor an alias of any node resolves to
    // nothing (no false positive), and one alias-bearing node does not pull in
    // another: each alias resolves to exactly one node.
    assert!(
        traverse::resolve_entry(&db, "Nonexistent Corp", None, None)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        traverse::resolve_entry(&db, "pacific trading group", None, None)
            .unwrap()
            .len(),
        1
    );

    // With the case-differing alias now resolving directly, the suggestion
    // fallback is not consulted for it; a *nonexistent* name still offers
    // labels close to its spelling, and a divergent alias that resolves is
    // not duplicated as a suggestion.
    assert!(
        traverse::suggest_entities(&db, "international business machines", None, None)
            .unwrap()
            .is_empty(),
        "the alias resolves directly, so no suggestion is needed"
    );
    assert_eq!(
        traverse::suggest_entities(&db, "Ibm", None, None).unwrap(),
        ["IBM (vendor)"]
    );
}
