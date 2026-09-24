//! Retrieval latency against workspace size: the exact vector scan
//! (`search_similar_chunks`), the BM25 leg (`search_keyword_chunks`), and the
//! fused path (`search_hybrid_chunks`) over a synthetic workspace of 10k,
//! 100k, and (opt-in) 1M chunks. The numbers in design doc section 15 come
//! from here.
//!
//! ```text
//! cargo bench -p quack-core --bench retrieval
//! QUACK_BENCH_CHUNKS=1000000 cargo bench -p quack-core --bench retrieval
//! ```
//!
//! `QUACK_BENCH_CHUNKS` is the largest size measured (default 100,000),
//! `QUACK_BENCH_DIM` the embedding width (default 1,024, qwen3-embedding's).
//! The workspace is in memory, so the times exclude first-touch disk reads
//! and are what the query itself costs.

#![expect(clippy::unwrap_used, reason = "a benchmark harness")]
#![expect(
    clippy::arithmetic_side_effects,
    clippy::integer_division,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "synthetic data and small counters in a benchmark harness"
)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use quack_core::embedding::{Dimension, Vector};
use quack_core::ids::{ChunkId, DocumentId};
use quack_core::storage::workspace::{
    ChunkScope, DocumentStatus, HybridLimits, NewChunk, NewDocument, WorkspaceDb,
};

const SIZES: [usize; 3] = [10_000, 100_000, 1_000_000];
const CHUNKS_PER_DOC: usize = 1_000;
const WORDS_PER_CHUNK: usize = 60;
const VOCABULARY: usize = 2_000;
const TOP_K: u32 = 8;
const RRF_K: u32 = 60;
const QUERY_TEXT: &str = "w3 w17 w250 w1999";

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A deterministic pseudo-random vector; the scan's cost depends on the
/// width, not the values.
fn embedding(seed: usize, dim: usize) -> Vector {
    Vector::from(
        (0..dim)
            .map(|i| (((seed * 7919 + i * 104_729) % 65_521) as f32 / 65_521.0) - 0.5)
            .collect::<Vec<f32>>(),
    )
}

/// Sixty pseudo-words from a 2,000-word vocabulary with a Zipf-like skew,
/// so the term index has long and short posting lists as real text does.
fn content(seed: usize) -> String {
    let mut words = Vec::with_capacity(WORDS_PER_CHUNK);
    for i in 0..WORDS_PER_CHUNK {
        let r = (seed * 2_654_435_761 + i * 40_503) % 1_000_003;
        let id = (r % VOCABULARY) * (r % VOCABULARY) / VOCABULARY;
        words.push(format!("w{id}"));
    }
    words.join(" ")
}

/// Grow `db` from `from` to `to` chunks.
fn fill(db: &WorkspaceDb, from: usize, to: usize, dim: usize) {
    for doc in (from / CHUNKS_PER_DOC)..(to / CHUNKS_PER_DOC) {
        let doc_id = format!("doc-{doc}");
        db.insert_document(
            &NewDocument::new(
                &DocumentId::from(doc_id.as_str()),
                &format!("doc-{doc}.md"),
                "text/markdown",
                1,
            )
            .with_status(DocumentStatus::Ready),
        )
        .unwrap();
        for c in 0..CHUNKS_PER_DOC {
            let n = doc * CHUNKS_PER_DOC + c;
            db.insert_chunk(&NewChunk {
                id: &ChunkId::from(format!("chunk-{n}")),
                document_id: &DocumentId::from(doc_id.as_str()),
                chunk_index: c as u32,
                content: &content(n),
                heading: None,
                page: None,
                embedding: Some(&embedding(n, dim)),
            })
            .unwrap();
        }
    }
}

fn retrieval(c: &mut Criterion) {
    let max_chunks = env_usize("QUACK_BENCH_CHUNKS", 100_000);
    let dim = env_usize("QUACK_BENCH_DIM", 1_024);
    let db = WorkspaceDb::open_in_memory(Dimension::new(dim as u32)).unwrap();
    let query_vec = embedding(usize::MAX / 3, dim);
    let scope = ChunkScope::all();

    let mut filled = 0usize;
    for size in SIZES.into_iter().filter(|&s| s <= max_chunks) {
        fill(&db, filled, size, dim);
        filled = size;

        let mut group = c.benchmark_group("retrieval");
        group.throughput(Throughput::Elements(size as u64));
        // A query over a million chunks takes most of a second; ten samples
        // of a few seconds each keep the large point affordable.
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(if size >= 1_000_000 { 30 } else { 10 }));

        group.bench_with_input(BenchmarkId::new("vector", size), &size, |b, _| {
            b.iter(|| {
                let hits = db
                    .search_similar_chunks(black_box(&query_vec), TOP_K, &scope)
                    .unwrap();
                assert_eq!(hits.len(), TOP_K as usize);
            });
        });
        group.bench_with_input(BenchmarkId::new("keyword", size), &size, |b, _| {
            b.iter(|| {
                let hits = db
                    .search_keyword_chunks(black_box(QUERY_TEXT), TOP_K, &scope)
                    .unwrap();
                assert!(!hits.is_empty());
            });
        });
        group.bench_with_input(BenchmarkId::new("hybrid", size), &size, |b, _| {
            b.iter(|| {
                let hits = db
                    .search_hybrid_chunks(
                        black_box(QUERY_TEXT),
                        black_box(&query_vec),
                        HybridLimits {
                            top_k: TOP_K,
                            rrf_k: RRF_K,
                        },
                        &scope,
                    )
                    .unwrap();
                assert!(!hits.is_empty());
            });
        });
        group.finish();
    }
}

criterion_group!(benches, retrieval);
criterion_main!(benches);
