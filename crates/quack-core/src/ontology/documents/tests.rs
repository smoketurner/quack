use super::*;
use crate::embedding::Dimension;
use crate::extraction::{Extract, ExtractFuture};
use crate::ingestion::parser::SectionKind;
use crate::progress::ChunkDone;
use crate::progress::RunControl;
use crate::storage::workspace::{NewChunk, NewDocument};
use schemars::schema_for;

struct Canned;

impl Extract<OpenExtraction> for Canned {
    fn extract<'a>(&'a self, text: &'a str) -> ExtractFuture<'a, OpenExtraction> {
        Box::pin(async move {
            if text.contains("FAIL") {
                return Err(Error::Ontology(String::from("boom")));
            }
            serde_json::from_str::<OpenExtraction>(&format!(
                    r#"{{"entities": [{{"name": "Orgenics", "type": "vendor"}}, {{"name": "Orgenics", "type": "organization"}}, {{"name": "USAID", "type": "organization"}}, {{"name": "Kenya", "type": "country"}}, {{"name": "{}", "type": "Shipment"}}],
"relations": [{{"subject": "Orgenics", "relation": "ships to", "object": "Kenya"}}],
"attributes": [{{"entity": "Orgenics", "name": "founded", "value": "1983"}}, {{"entity": "Kenya", "name": "region", "value": "East Africa"}}]}}"#,
                    text.split_whitespace().next().unwrap_or("x")
                ))
                .map_err(|e| Error::Ontology(e.to_string()))
        })
    }
}

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

fn workspace_with_docs() -> WorkspaceDb {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    for d in 1..=4 {
        let doc = format!("doc{d}");
        assert!(
            db.insert_document(
                &NewDocument::new(
                    &DocumentId::from(doc.as_str()),
                    &format!("{doc}.md"),
                    "text/markdown",
                    10
                )
                .with_status(DocumentStatus::Ready)
            )
            .is_ok()
        );
        for i in 0..5_u32 {
            let content = format!(
                "S{d}{i} passage number {i} of document {d} with enough words to count as text"
            );
            let id = format!("{doc}-{i}");
            assert!(
                db.chunk_writer(&DocumentId::from(doc.as_str()), &content)
                    .and_then(|writer| writer.insert(&NewChunk {
                        id: &ChunkId::from(id.as_str()),
                        chunk_index: i,
                        content: &content,
                        heading: None,
                        page: None,
                        kind: SectionKind::Body,
                        locator: None,
                        embedding: None
                    }))
                    .is_ok()
            );
        }
    }
    assert!(
        db.insert_document(
            &NewDocument::new(&DocumentId::from("pending"), "p.md", "text/markdown", 1)
                .with_status(DocumentStatus::Queued)
        )
        .is_ok()
    );
    assert!(
        db.chunk_writer(
            &DocumentId::from("pending"),
            "not ready but long enough to pass the length filter here"
        )
        .and_then(|writer| writer.insert(&NewChunk {
            id: &ChunkId::from("p-0"),
            chunk_index: 0,
            content: "not ready but long enough to pass the length filter here",
            heading: None,
            page: None,
            kind: SectionKind::Body,
            locator: None,
            embedding: None
        }))
        .is_ok()
    );
    db
}

#[test]
fn sampling_is_stratified_across_ready_documents() {
    let db = workspace_with_docs();
    let cost = estimate(
        &db,
        &DocumentEvidenceOptions {
            sample_chunks: 8,
            ..DocumentEvidenceOptions::default()
        },
    )
    .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!((cost.documents, cost.chunks, cost.model_calls), (4, 8, 8));
    let sample = sample_chunks(&db, 8).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(sample.len(), 8);
    let per_doc: BTreeMap<&str, usize> = sample.iter().fold(BTreeMap::new(), |mut m, c| {
        *m.entry(c.document_id.as_str()).or_default() += 1;
        m
    });
    assert!(per_doc.values().all(|n| *n == 2), "{per_doc:?}");
    assert!(sample.iter().all(|c| c.document_id != "pending"));
    let ids: Vec<&str> = sample
        .iter()
        .filter(|c| c.document_id == "doc1")
        .map(|c| c.id.as_str())
        .collect();
    assert_eq!(ids, ["doc1-0", "doc1-2"], "evenly spaced");
    assert!(sample_chunks(&db, 0).is_ok_and(|s| s.is_empty()));
    let all = sample_chunks(&db, 100).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(all.len(), 20);
}

#[tokio::test]
async fn observations_become_classes_relations_hierarchy_and_properties() {
    let db = workspace_with_docs();
    let sample = sample_chunks(&db, 8).unwrap_or_else(|e| fail(&e.to_string()));
    let Observed {
        observations,
        failed: failures,
    } = observe(
        &sample,
        ExtractionRun {
            extractor: &Canned,
            concurrency: 1,
            control: RunControl::unobserved(),
        },
    )
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!((observations.len(), failures), (8, 0));
    let options = DocumentEvidenceOptions {
        min_support_documents: 3,
        ..DocumentEvidenceOptions::default()
    };
    let candidates = propose(&observations, None, &options, None);
    let find = |kind: &str, id: &str| {
        candidates
            .iter()
            .find(|c| c.proposal.kind().as_str() == kind && c.proposal.id() == id)
    };
    let vendor = find("class", "vendor").unwrap_or_else(|| fail("no vendor"));
    assert!(
        matches!(&vendor.proposal, Proposal::Class(c) if c.parent == "organization"),
        "vendor mentions are also organizations"
    );
    assert!(!vendor.low_support && vendor.confidence >= 0.9);
    assert!(
        find("class", "shipment").is_some(),
        "types normalize to snake_case singular"
    );
    assert!(
        matches!(&find("class", "organization").map(|c| &c.proposal), Some(Proposal::Class(c)) if c.parent == "entity")
    );
    let ships = find("relation", "ships_to").unwrap_or_else(|| fail("no relation"));
    assert!(
        matches!(&ships.proposal, Proposal::Relation(r) if r.domain == "vendor" && r.range == "country"),
        "{:?}",
        ships.proposal
    );
    assert!(
        ships
            .evidence
            .get("examples")
            .and_then(|e| e.as_array())
            .is_some_and(|e| e.len() == 3
                && e.first()
                    .and_then(|x| x.get("chunk_id"))
                    .is_some_and(serde_json::Value::is_string))
    );
    let founded = find("property", "founded").unwrap_or_else(|| fail("no founded"));
    assert!(
        matches!(&founded.proposal, Proposal::Property { class, property } if class == "vendor" && property.kind == PropertyType::Number)
    );
    assert!(
        matches!(&find("property", "region").map(|c| &c.proposal), Some(Proposal::Property { property, .. }) if property.kind == PropertyType::String)
    );

    // Extend mode skips what exists; low support marks thin evidence.
    let mut existing = Ontology::builtin_default();
    existing.classes.push(Class {
        id: ClassId::from("vendor"),
        parent: ClassId::from("organization"),
        label: None,
        description: None,
        key: None,
        properties: Vec::new(),
    });
    let extended = propose(&observations, Some(&existing), &options, None);
    assert!(
        extended
            .iter()
            .all(|c| c.proposal.id() != "vendor" && c.proposal.id() != "organization")
    );
    let thin = propose(
        observations.get(..1).unwrap_or_default(),
        None,
        &DocumentEvidenceOptions {
            min_support_documents: 3,
            ..DocumentEvidenceOptions::default()
        },
        None,
    );
    assert!(thin.iter().all(|c| c.low_support));
}

#[tokio::test]
async fn failed_chunks_are_skipped_and_all_failures_is_an_error() {
    let chunks = vec![
        SampledChunk {
            id: ChunkId::from("a"),
            document_id: DocumentId::from("d"),
            filename: String::from("f"),
            content: String::from("FAIL here"),
        },
        SampledChunk {
            id: ChunkId::from("b"),
            document_id: DocumentId::from("d"),
            filename: String::from("f"),
            content: String::from("Fine passage"),
        },
    ];
    let seen = std::sync::Mutex::new(Vec::new());
    let progress = |done: ChunkDone| {
        if let Ok(mut seen) = seen.lock() {
            seen.push((done.done, done.total, done.failed));
        }
    };
    let control = RunControl {
        progress: &progress,
        cancel: None,
    };
    let Observed {
        observations,
        failed: failures,
    } = observe(
        &chunks,
        ExtractionRun {
            extractor: &Canned,
            concurrency: 2,
            control,
        },
    )
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!((observations.len(), failures), (1, 1));
    // Every chunk reports once, in order, with the failure count so
    // far (issue #67: the run was silent until the end).
    assert_eq!(
        seen.lock().map(|s| s.clone()).unwrap_or_default(),
        [(1, 2, 1), (2, 2, 1)]
    );
    let all_bad = vec![SampledChunk {
        id: ChunkId::from("a"),
        document_id: DocumentId::from("d"),
        filename: String::from("f"),
        content: String::from("FAIL"),
    }];
    assert!(
        observe(
            &all_bad,
            ExtractionRun {
                extractor: &Canned,
                concurrency: 1,
                control: RunControl::unobserved()
            }
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn a_cancelled_run_stops_before_the_next_chunk() {
    let chunks = vec![SampledChunk {
        id: ChunkId::from("a"),
        document_id: DocumentId::from("d"),
        filename: String::from("f"),
        content: String::from("Fine passage"),
    }];
    let cancel = tokio_util::sync::CancellationToken::new();
    cancel.cancel();
    let control = RunControl {
        progress: &|_| {},
        cancel: Some(&cancel),
    };
    assert!(matches!(
        observe(
            &chunks,
            ExtractionRun {
                extractor: &Canned,
                concurrency: 1,
                control
            }
        )
        .await,
        Err(Error::Cancelled)
    ));
}

#[test]
fn vocabulary_merges_plurals_and_near_synonyms() {
    let mut counts = Tally::default();
    counts.add("Vendors", 5);
    counts.add("vendor", 2);
    counts.add("supplier", 1);
    let exact = Vocabulary::build(&counts, None);
    assert_eq!(exact.id("Vendors"), "vendor");
    assert_eq!(exact.id("vendor"), "vendor");
    assert_eq!(exact.id("supplier"), "supplier");
    let near =
        |a: &str, b: &str| (a == "vendor" && b == "supplier") || (a == "supplier" && b == "vendor");
    let clustered = Vocabulary::build(&counts, Some(&near));
    assert_eq!(clustered.id("supplier"), "vendor", "the frequent name wins");
    assert!(
        serde_json::from_str::<OpenExtraction>("{\"entities\": [{\"name\": \"x\"}]}").is_err(),
        "type is required"
    );
    let schema = serde_json::to_string(&schema_for!(OpenExtraction)).unwrap_or_default();
    assert!(
        schema.contains("\"entities\"") && schema.contains("\"subject\""),
        "{schema}"
    );
    assert_eq!(
        PropertyType::infer(&[String::from("2024-01-05"), String::from("3 May 2020")]),
        PropertyType::Date
    );
    assert_eq!(
        PropertyType::infer(&[String::from("$4,500"), String::from("12")]),
        PropertyType::Number
    );
    assert_eq!(
        PropertyType::infer(&[String::from("true"), String::from("False")]),
        PropertyType::Boolean
    );
}
