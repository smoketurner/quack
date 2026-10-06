#![expect(clippy::unwrap_used, reason = "test setup")]

use std::sync::Arc;

use super::*;
use crate::embedding::{Dimension, Profile, Prompts};
use crate::ontology::store::Revision;
use crate::storage::workspace::WorkspaceDb;
use rig::ProviderError;
use rig::embeddings::Embedding;

/// Text about fruit embeds along one axis, anything else along another.
struct FruitEmbedding;

impl EmbeddingModel for FruitEmbedding {
    fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> impl Future<Output = std::result::Result<Vec<Embedding>, ProviderError>> + Send {
        let out = texts
            .into_iter()
            .map(|text| {
                let lower = text.to_lowercase();
                let fruit = ["apple", "banana", "pear"]
                    .iter()
                    .any(|f| lower.contains(f));
                Embedding {
                    vec: if fruit {
                        vec![1.0, 0.0, 0.0, 0.0]
                    } else {
                        vec![0.0, 1.0, 0.0, 0.0]
                    },
                    document: text,
                }
            })
            .collect();
        async move { Ok(out) }
    }
}

fn profile() -> Profile {
    Profile::new("fruit", Dimension::new(4), Prompts::default())
}

fn embedder() -> Embedder<FruitEmbedding> {
    Embedder::new(FruitEmbedding, profile())
}

/// `extra` filler tables plus the ones the tests look for.
fn workspace(extra: usize) -> WorkspaceDb {
    let db = WorkspaceDb::open_in_memory_with_profile(profile()).unwrap();
    for i in 0..extra {
        db.execute_statement(&format!(
            "CREATE TABLE filler_{i:02} AS SELECT 'x{i}' AS code, {i} AS quantity"
        ))
        .unwrap();
    }
    db.execute_statement(
        "CREATE TABLE zz_orders AS SELECT * FROM (VALUES ('o1', 1250, 'Rotterdam')) t(order_id, amount, city)",
    )
    .unwrap();
    db.execute_statement("CREATE TABLE produce AS SELECT 'apple' AS item, 3 AS crates")
        .unwrap();
    TableProfile::refresh_stale(&db).unwrap();
    db
}

fn ranked(db: &WorkspaceDb, query: &str, vector: Option<&Vector>) -> Vec<String> {
    TableCards::read(db, ontology_store::current(db).unwrap().as_ref())
        .unwrap()
        .rank(db, query, vector, 5, 60)
        .unwrap()
        .into_iter()
        .map(|r| r.table)
        .collect()
}

#[test]
fn the_layout_turns_ranked_past_the_detail_cap() {
    assert_eq!(TableLayout::of(DETAILED_TABLES), TableLayout::AllDescribed);
    assert_eq!(
        TableLayout::of(DETAILED_TABLES.saturating_add(1)),
        TableLayout::Ranked
    );
}

#[test]
fn notes_values_and_ontology_words_rank_a_table_first() {
    let db = workspace(30);
    assert!(ranked(&db, "", None).is_empty());
    assert!(
        ranked(&db, "zebra", None).is_empty(),
        "no shared term, no table"
    );

    TableNote::set(
        &db,
        "zz_orders",
        "One row per order; amount is in cents",
        None,
    )
    .unwrap();
    assert_eq!(
        ranked(&db, "total order amount in dollars last month", None)
            .first()
            .map(String::as_str),
        Some("zz_orders")
    );
    assert_eq!(
        ranked(&db, "orders shipped to rotterdam", None)
            .first()
            .map(String::as_str),
        Some("zz_orders"),
        "a common value is on the card"
    );

    let json = r#"{"classes": [{"id": "sale", "key": "order_id", "properties": ["order_id", "amount"]}],
        "properties": [{"id": "order_id", "type": "string"},
                       {"id": "amount", "type": "number", "synonyms": ["turnover"]}],
        "mappings": [{"table": "zz_orders", "class": "sale", "key": "order_id", "properties": {"amount": "amount"}}]}"#;
    ontology_store::save(
        &db,
        &Ontology::from_json(json).unwrap(),
        Revision::reviewed(None, None),
    )
    .unwrap();
    assert_eq!(
        ranked(&db, "turnover", None),
        ["zz_orders"],
        "a synonym from the ontology"
    );
    assert!(
        !ranked(&db, "sale", None)
            .iter()
            .any(|t| t.starts_with("graph_")),
        "graph views are not ranked"
    );
}

#[tokio::test]
async fn card_vectors_are_made_once_and_rank_without_shared_words() {
    let db = workspace(30);
    let reader_conn = db.try_clone_reader().unwrap();
    let writer = Arc::new(Writer::spawn(db).unwrap());
    let reader = ReaderDb::new(Arc::clone(&writer));
    let made = TableCards::refresh_vectors(&reader, &writer, &embedder())
        .await
        .unwrap();
    assert_eq!(made, 32);
    assert_eq!(
        TableCards::refresh_vectors(&reader, &writer, &embedder())
            .await
            .unwrap(),
        0,
        "unchanged cards keep their vectors"
    );

    let query = embedder()
        .embed_one(&Input::Query(String::from("banana")))
        .await
        .unwrap();
    assert!(
        ranked(&reader_conn, "banana", None).is_empty(),
        "no card says banana"
    );
    assert_eq!(
        ranked(&reader_conn, "banana", Some(&query))
            .first()
            .map(String::as_str),
        Some("produce")
    );

    writer
        .run(|db| TableNote::set(db, "produce", "crates of pears", None))
        .await
        .unwrap();
    writer
        .run(|db| db.execute_statement("DROP TABLE filler_00"))
        .await
        .unwrap();
    assert_eq!(
        TableCards::refresh_vectors(&reader, &writer, &embedder())
            .await
            .unwrap(),
        1,
        "only the changed card is embedded again"
    );
    let stored: i64 = writer
        .run(|db| {
            Ok(db
                .connection()
                .query_row("SELECT count(*) FROM _quack_table_cards", [], |r| r.get(0))?)
        })
        .await
        .unwrap();
    assert_eq!(stored, 31, "a dropped table's vector goes");
}

#[tokio::test]
async fn a_narrow_workspace_makes_no_card_vectors() {
    let db = workspace(3);
    let writer = Arc::new(Writer::spawn(db).unwrap());
    let reader = ReaderDb::new(Arc::clone(&writer));
    assert_eq!(
        TableCards::refresh_vectors(&reader, &writer, &embedder())
            .await
            .unwrap(),
        0
    );
}
