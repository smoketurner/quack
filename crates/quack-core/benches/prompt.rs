//! System prompt assembly on a large workspace: `SystemPrompt::build` runs
//! on every turn, so its cost against many tables (150 here, more than the
//! 25 the prompt details) is what a request pays before the model sees
//! anything.
//!
//! ```text
//! cargo bench -p quack-core --bench prompt
//! ```

#![expect(clippy::unwrap_used, reason = "a benchmark harness")]
#![expect(
    clippy::cast_possible_truncation,
    reason = "small counters in a benchmark harness"
)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::text_to_sql::{self, PromptOptions};
use quack_core::ids::{ChunkId, DocumentId};
use quack_core::ontology::store::Revision;
use quack_core::ontology::{Ontology, store as ontology_store};
use quack_core::storage::sessions::ChatMode;
use quack_core::storage::workspace::{DocumentStatus, NewChunk, NewDocument, WorkspaceDb};

const EMBEDDING_DIM: u32 = 384;
const DOCUMENTS: usize = 40;
const CHUNKS_PER_DOC: usize = 25;
const TABLES: usize = 150;
const TABLE_ROWS: usize = 10_000;

fn workspace() -> WorkspaceDb {
    let db = WorkspaceDb::open_in_memory(EMBEDDING_DIM).unwrap();
    for d in 0..DOCUMENTS {
        let doc_id = format!("doc-{d}");
        db.insert_document(
            &NewDocument::new(
                &DocumentId::from(doc_id.as_str()),
                &format!("policy-{d}.pdf"),
                "application/pdf",
                10_000,
            )
            .with_status(DocumentStatus::Ready),
        )
        .unwrap();
        for c in 0..CHUNKS_PER_DOC {
            db.insert_chunk(&NewChunk {
                id: &ChunkId::from(format!("chunk-{d}-{c}")),
                document_id: &DocumentId::from(doc_id.as_str()),
                chunk_index: c as u32,
                content: "flood exclusion premium coverage claim policy audit",
                heading: None,
                page: Some(c as u32),
                embedding: None,
            })
            .unwrap();
        }
    }
    for t in 0..TABLES {
        db.execute_statement(&format!(
            "CREATE TABLE \"table_{t}\" (id BIGINT, label VARCHAR, amount DOUBLE)"
        ))
        .unwrap();
        db.execute_statement(&format!(
            "INSERT INTO \"table_{t}\" SELECT i, 'row-' || i, i * 1.5 FROM range({TABLE_ROWS}) t(i)"
        ))
        .unwrap();
    }
    ontology_store::save(
        &db,
        &Ontology::builtin_default(),
        Revision::reviewed(Some("bench"), None),
    )
    .unwrap();
    db
}

fn prompt(c: &mut Criterion) {
    let db = workspace();
    let options = PromptOptions {
        mode: ChatMode::Chat,
        write_policy: WritePolicy::Deny,
        pinned_token_budget: 4_000,
        context: Some(String::from(
            "This workspace tracks insurance claims and their supporting policy documents.",
        )),
        context_max_tokens: 2_000,
        ollama_context_cap: None,
    };
    c.bench_function("build_system_prompt/150_tables", |b| {
        b.iter(|| {
            let prompt = text_to_sql::SystemPrompt::build(&db, black_box(&options)).unwrap();
            assert!(!prompt.is_empty());
        });
    });
}

criterion_group!(benches, prompt);
criterion_main!(benches);
