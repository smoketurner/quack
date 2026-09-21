//! Standalone timing harness for the query pipeline, not part of the test
//! suite: builds a synthetic workspace (thousands of chunks, tens of
//! thousands of term rows, hundreds of tables) and times the pieces of a
//! turn that run on every request: the hybrid retrieval scan and the system
//! prompt assembly. Run with `cargo run --release --example
//! bench_query_pipeline -p quack-core`.

#![expect(clippy::print_stdout, reason = "a benchmark harness, not shipped code")]
#![expect(clippy::unwrap_used, reason = "a throwaway harness")]
#![expect(
    clippy::arithmetic_side_effects,
    reason = "a throwaway harness over small, fixed loop bounds"
)]
#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "synthetic embedding values and small counters in a throwaway harness"
)]
#![expect(
    clippy::indexing_slicing,
    reason = "a fixed, non-empty word list indexed by a modulo in a throwaway harness"
)]

use std::time::Instant;

use quack_core::analysis::text_to_sql::{self, PromptOptions};
use quack_core::storage::sessions::ChatMode;
use quack_core::storage::workspace::{ChunkScope, NewChunk, NewDocument, WorkspaceDb};

const EMBEDDING_DIM: u32 = 384;
const DOCUMENTS: usize = 40;
const CHUNKS_PER_DOC: usize = 100; // 4,000 chunks total
const TABLES: usize = 150;
const TABLE_ROWS: usize = 10_000;

fn synthetic_embedding(seed: usize) -> Vec<f32> {
    (0..EMBEDDING_DIM as usize)
        .map(|i| ((seed * 31 + i) % 997) as f32 / 997.0)
        .collect()
}

fn lorem(seed: usize) -> String {
    let words = [
        "quack",
        "duckdb",
        "workspace",
        "chunk",
        "embedding",
        "vector",
        "keyword",
        "rank",
        "policy",
        "claim",
        "exclusion",
        "flood",
        "coverage",
        "premium",
        "audit",
        "session",
        "ontology",
        "graph",
        "entity",
        "relation",
    ];
    let mut out = String::new();
    for i in 0..120 {
        out.push_str(words[(seed + i) % words.len()]);
        out.push(' ');
    }
    out
}

fn main() {
    let db = WorkspaceDb::open_in_memory(EMBEDDING_DIM).unwrap();

    let t0 = Instant::now();
    for d in 0..DOCUMENTS {
        let doc_id = format!("doc-{d}");
        let filename = format!("policy-{d}.pdf");
        db.insert_document(
            &NewDocument::new(&doc_id, &filename, "application/pdf", 10_000).with_status("ready"),
        )
        .unwrap();
        for c in 0..CHUNKS_PER_DOC {
            let chunk_id = format!("chunk-{d}-{c}");
            let content = lorem(d * CHUNKS_PER_DOC + c);
            let embedding = synthetic_embedding(d * CHUNKS_PER_DOC + c);
            db.insert_chunk(&NewChunk {
                id: &chunk_id,
                document_id: &doc_id,
                chunk_index: c as u32,
                content: &content,
                heading: None,
                page: Some(c as u32),
                embedding: Some(&embedding),
            })
            .unwrap();
        }
    }
    let ingest_elapsed = t0.elapsed();
    println!(
        "ingest {} documents x {} chunks = {} chunks: {:?}",
        DOCUMENTS,
        CHUNKS_PER_DOC,
        DOCUMENTS * CHUNKS_PER_DOC,
        ingest_elapsed
    );

    let t0 = Instant::now();
    for t in 0..TABLES {
        let name = format!("table_{t}");
        db.execute_statement(&format!(
            "CREATE TABLE \"{name}\" (id BIGINT, label VARCHAR, amount DOUBLE)"
        ))
        .unwrap();
        db.execute_statement(&format!(
            "INSERT INTO \"{name}\" SELECT i, 'row-' || i, i * 1.5 FROM range({TABLE_ROWS}) t(i)"
        ))
        .unwrap();
    }
    println!(
        "create {TABLES} tables x {TABLE_ROWS} rows: {:?}",
        t0.elapsed()
    );

    // --- hybrid retrieval ---
    let query_vec = synthetic_embedding(12345);
    let scope = ChunkScope::all();
    let mut durations = Vec::new();
    for _ in 0..20 {
        let t0 = Instant::now();
        let results = db
            .search_hybrid_chunks("flood exclusion premium", &query_vec, 8, 60, &scope)
            .unwrap();
        durations.push(t0.elapsed());
        assert!(!results.is_empty());
    }
    report("search_hybrid_chunks", &durations);

    // --- system prompt assembly ---
    quack_core::ontology::store::save(
        &db,
        &quack_core::ontology::Ontology::builtin_default(),
        Some("bench"),
        None,
    )
    .unwrap();
    let options = PromptOptions {
        mode: ChatMode::Chat,
        write_policy: quack_core::analysis::policy::WritePolicy::Deny,
        pinned_token_budget: 4_000,
        context: Some(String::from(
            "This workspace tracks insurance claims and their supporting policy documents.",
        )),
        context_max_tokens: 2_000,
        ollama_context_cap: None,
    };
    let mut durations = Vec::new();
    let mut prompt = String::new();
    for _ in 0..10 {
        let t0 = Instant::now();
        prompt = text_to_sql::build_system_prompt(&db, &options).unwrap();
        durations.push(t0.elapsed());
        assert!(!prompt.is_empty());
    }
    report("build_system_prompt", &durations);
    println!(
        "system prompt size: {} chars, ~{} tokens (4 chars/token estimate)",
        prompt.chars().count(),
        prompt.chars().count().div_ceil(4)
    );
}

fn report(label: &str, durations: &[std::time::Duration]) {
    let total: std::time::Duration = durations.iter().sum();
    let count = u32::try_from(durations.len()).unwrap();
    println!(
        "{label} x{}: avg {:?}, min {:?}, max {:?}",
        durations.len(),
        total / count,
        durations.iter().min().unwrap(),
        durations.iter().max().unwrap()
    );
}
