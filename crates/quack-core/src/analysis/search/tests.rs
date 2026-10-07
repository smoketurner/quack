use std::sync::Arc;

use super::*;
use crate::analysis::events::{self, TurnRecorder};
use crate::analysis::policy::WritePolicy;
use crate::analysis::rerank::{RankFuture, Ranked, Reranker};
use crate::analysis::tools::{
    NonBlank, ReadDocumentArgs, ReadDocumentTool, SearchDocumentsArgs, SearchDocumentsTool, Turn,
};
use crate::config::RetrievalConfig;
use crate::embedding::Dimension;
use crate::ingestion::parser::SectionKind;
use crate::llm::EmbedModel;
use crate::storage::workspace::{NewChunk, NewDocument};
use crate::storage::writer::Writer;
use rig::tool::Tool;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

/// A workspace of three documents about renewals: two ready, one still
/// queued.
fn workspace() -> WorkspaceDb {
    let db =
        WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
    for (id, name, status) in [
        ("01a-policy", "policy.md", DocumentStatus::Ready),
        ("01b-notes", "notes.md", DocumentStatus::Ready),
        ("01c-draft", "draft.md", DocumentStatus::Queued),
    ] {
        db.insert_document(
            &NewDocument::new(&DocumentId::from(id), name, "text/markdown", 1).with_status(status),
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
        db.insert_chunk(&NewChunk {
            id: &ChunkId::from(format!("{id}-c0")),
            document_id: &DocumentId::from(id),
            chunk_index: 0,
            content: "Renewal terms for the policy year.",
            heading: None,
            page: None,
            kind: SectionKind::Body,
            locator: None,
            embedding: None,
        })
        .unwrap_or_else(|e| fail(&e.to_string()));
    }
    db
}

fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(|n| (*n).to_owned()).collect()
}

fn ids(ids: &[&str]) -> Vec<DocumentId> {
    ids.iter().map(|i| DocumentId::from(*i)).collect()
}

#[test]
fn a_scope_resolves_ready_documents_by_name_or_prefix_once_each() {
    let db = workspace();
    let scope = DocumentScope::resolve(&db, &names(&["policy.md", "01b", "01a-policy"]))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let scoped: Vec<&str> = scope.documents().iter().map(|d| d.id.as_str()).collect();
    assert_eq!(scoped, ["01a-policy", "01b-notes"]);
    assert_eq!(scope.names(), "policy.md, notes.md");
    assert!(
        scope
            .prompt_note()
            .is_some_and(|n| n.contains("policy.md (id: 01a-policy)"))
    );
    let everything = DocumentScope::resolve(&db, &[]).unwrap_or_else(|e| fail(&e.to_string()));
    assert!(everything.is_everything() && everything.prompt_note().is_none());
    let unknown = DocumentScope::resolve(&db, &names(&["missing.md"]))
        .err()
        .map(|e| e.to_string());
    assert!(unknown.is_some_and(|e| e.contains("no document matches 'missing.md'")));
    let queued = DocumentScope::resolve(&db, &names(&["draft.md"]))
        .err()
        .map(|e| e.to_string());
    assert!(queued.is_some_and(|e| e.contains("draft.md is queued, not ready")));
}

#[test]
fn a_scope_narrows_what_the_model_names_and_refuses_what_lies_outside() {
    let db = workspace();
    let everything = DocumentScope::default();
    assert_eq!(
        everything.narrow(ids(&["01b-notes"])).ok(),
        Some(ids(&["01b-notes"]))
    );
    assert_eq!(everything.narrow(Vec::new()).ok(), Some(Vec::new()));
    let policy = DocumentScope::resolve(&db, &names(&["policy.md"]))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        policy.narrow(Vec::new()).ok(),
        Some(ids(&["01a-policy"])),
        "nothing named: the person's documents"
    );
    assert_eq!(
        policy.narrow(ids(&["01a-policy", "01b-notes"])).ok(),
        Some(ids(&["01a-policy"])),
        "the overlap"
    );
    let outside = policy
        .narrow(ids(&["01b-notes"]))
        .err()
        .map(|e| e.to_string());
    assert!(outside.is_some_and(|e| e.contains("limited this question to policy.md")));
}

#[test]
fn a_search_needs_a_query_and_bounds_its_hits() {
    assert!(DocumentSearch::new("   ", 5).is_err());
    let search = DocumentSearch::new("  renewal ", 0).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!((search.query.as_str(), search.top_k), ("renewal", 1));
    assert_eq!(
        DocumentSearch::new("x", 10_000).map(|s| s.top_k).ok(),
        Some(MAX_TOP_K)
    );
    assert_eq!(search.describe(), "renewal");
    let narrowed = DocumentSearch {
        documents: names(&["policy.md"]),
        entity: Some(String::from("Acme")),
        filter: DocumentFilter {
            tags: names(&["2026"]),
            ..DocumentFilter::default()
        },
        mode: SearchMode::Keyword,
        ..search
    };
    assert_eq!(
        narrowed.describe(),
        "renewal (about Acme, in policy.md, where {\"tags\":[\"2026\"]}, keyword only)"
    );
}

#[test]
fn a_search_stays_within_the_persons_scope_and_its_own_documents() {
    let db = workspace();
    let limits = HybridLimits {
        top_k: 5,
        rrf_k: 60,
    };
    let search = DocumentSearch::new("renewal", 5).unwrap_or_else(|e| fail(&e.to_string()));
    let found = |search: &DocumentSearch, within: &DocumentScope| -> Vec<String> {
        search
            .explain(&db, &SearchVectors::default(), limits, within)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .fused
            .into_iter()
            .map(|h| h.document_id.into_string())
            .collect()
    };
    let mut all = found(&search, &DocumentScope::default());
    all.sort();
    assert_eq!(
        all,
        ["01a-policy", "01b-notes"],
        "a queued document is not searched"
    );
    let policy = DocumentScope::resolve(&db, &names(&["policy.md"]))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(found(&search, &policy), ["01a-policy"]);
    let notes = DocumentSearch {
        documents: names(&["notes.md"]),
        ..search
    };
    assert_eq!(found(&notes, &DocumentScope::default()), ["01b-notes"]);
    let outside = notes
        .explain(&db, &SearchVectors::default(), limits, &policy)
        .err()
        .map(|e| e.to_string());
    assert!(outside.is_some_and(|e| e.contains("limited this question")));
}

struct Reverse;

impl Reranker for Reverse {
    fn rank<'a>(&'a self, _query: &'a str, candidates: &'a [ChunkSearchResult]) -> RankFuture<'a> {
        Box::pin(async move {
            Ok((0..candidates.len())
                .rev()
                .map(|index| Ranked {
                    index,
                    score: Some(0.5),
                })
                .collect())
        })
    }

    fn name(&self) -> &'static str {
        "reverse"
    }
}

#[tokio::test]
async fn an_outcome_reranks_keeps_top_k_and_renders_its_workings() {
    let db = workspace();
    let search = DocumentSearch::new("renewal", 1).unwrap_or_else(|e| fail(&e.to_string()));
    let explained = search
        .explain(
            &db,
            &SearchVectors::default(),
            HybridLimits {
                top_k: 5,
                rrf_k: 60,
            },
            &DocumentScope::default(),
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(explained.fused.len(), 2);
    let plain = SearchOutcome::rerank(explained.clone(), None, "renewal", 1).await;
    assert_eq!(plain.explanation.fused.len(), 1);
    assert_eq!(plain.rerank, RerankOutcome::Skipped);
    let rerank = Rerank {
        reranker: Arc::new(Reverse),
        candidates: 5,
    };
    let reranked = SearchOutcome::rerank(explained, Some(&rerank), "renewal", 1).await;
    let top = reranked.explanation.fused.first().map(|h| h.ranks);
    assert_eq!(top.and_then(|r| r.rerank_rank), Some(1));
    assert_eq!(top.and_then(|r| r.rerank_score), Some(0.5));
    let json = serde_json::json!(reranked.body(SearchDetail::Workings));
    assert_eq!(
        json.pointer("/chunks/0/rerank_rank"),
        Some(&serde_json::json!(1))
    );
    assert_eq!(
        json.pointer("/explain/rerank"),
        Some(&serde_json::json!("reranked by the reverse"))
    );
    assert!(json.pointer("/explain/keyword/1").is_some());
    assert!(reranked.body(SearchDetail::Hits).explain.is_none());
    let hits = reranked.render(SearchDetail::Hits);
    assert!(
        hits.starts_with("1. policy.md") || hits.starts_with("1. notes.md"),
        "{hits}"
    );
    assert!(
        hits.contains("keyword #") && hits.contains("rerank #1 (0.5000)"),
        "{hits}"
    );
    assert!(!hits.contains("Keyword leg"));
    let workings = reranked.render(SearchDetail::Workings);
    assert!(workings.contains("Keyword leg: 2 candidates"), "{workings}");
    assert!(workings.contains("Vector leg: 0 candidates"), "{workings}");
    assert!(
        workings.ends_with("Rerank: reranked by the reverse\n"),
        "{workings}"
    );
    let nothing = SearchOutcome::rerank(SearchExplanation::default(), None, "q", 5).await;
    assert!(
        nothing
            .render(SearchDetail::Hits)
            .starts_with("No passages found.")
    );
}

#[tokio::test]
async fn the_search_tool_keeps_within_the_turns_scope() {
    let db = Arc::new(Writer::spawn(workspace()).unwrap_or_else(|e| fail(&e.to_string())));
    let scope = db
        .run(|db| DocumentScope::resolve(db, &names(&["policy.md"])))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let (sink, _rx) = events::channel();
    let turn = Turn::new(TurnRecorder::new(sink), WritePolicy::Deny).within(scope);
    let tool = SearchDocumentsTool::<EmbedModel>::new(
        ReaderDb::new(Arc::clone(&db)),
        None,
        &RetrievalConfig::default(),
    );
    let args = |documents: &[&str]| SearchDocumentsArgs {
        query: String::from("renewal"),
        top_k: None,
        document_ids: names(documents),
        entity: NonBlank::default(),
        filters: DocumentFilter::default(),
    };
    let text = tool
        .call(&mut turn.context(), args(&[]))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        text.contains("policy.md") && !text.contains("notes.md"),
        "{text}"
    );
    let outside = tool
        .call(&mut turn.context(), args(&["notes.md"]))
        .await
        .err()
        .map(|e| e.to_string());
    assert!(
        outside.is_some_and(|e| e.contains("limited this question to policy.md")),
        "the model cannot widen the person's scope"
    );
    let read = ReadDocumentTool::new(ReaderDb::new(Arc::clone(&db)), &RetrievalConfig::default());
    let refused = read
        .call(
            &mut turn.context(),
            ReadDocumentArgs {
                document: String::from("notes.md"),
                from: None,
                limit: None,
            },
        )
        .await
        .err()
        .map(|e| e.to_string());
    assert!(
        refused.is_some_and(|e| e.contains("limited this question to policy.md")),
        "nor read a whole document outside it"
    );
}
