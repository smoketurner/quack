//! Extraction that follows an ingest or import under
//! `[graph].follow_ingest`: the new document's mapped tables, and with
//! `all` its chunks through the chat model, then resolution. The server
//! runs it as a background graph job, the command line and the terminal
//! after their own ingest.

use std::collections::BTreeSet;
use std::fmt;

use super::extract::{self, ChunkPlan, RunSummary};
use super::resolve::{self, ResolutionSummary};
use super::tables::{self, MappingSummary};
use super::{Standing, store};
use crate::config::Config;
use crate::error::Result;
use crate::extraction::{Extract, ExtractionRun};
use crate::ids::{ChunkId, DocumentId};
use crate::llm::{self, Embeddings};
use crate::ontology::{Ontology, store as ontology_store};
use crate::progress::RunControl;
use crate::storage::writer::Writer;

/// What a follow-up did.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct FollowUpSummary {
    pub tables: Vec<MappingSummary>,
    /// The document pass, when `follow_ingest = "all"` and the documents
    /// had unextracted chunks.
    pub chunks: Option<RunSummary>,
    pub resolution: ResolutionSummary,
}

/// One line, as the ingest's report and the job list show it.
impl fmt::Display for FollowUpSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        for table in &self.tables {
            parts.push(match &table.skipped {
                Some(reason) => format!("table {} skipped ({reason})", table.table),
                None => format!(
                    "table {}: {} nodes, {} edges",
                    table.table, table.nodes, table.edges
                ),
            });
        }
        if let Some(chunks) = &self.chunks {
            parts.push(format!(
                "{} chunks: {} nodes, {} edges",
                chunks.chunks, chunks.nodes, chunks.edges
            ));
        }
        if self.resolution.auto_merged > 0 || self.resolution.proposed > 0 {
            parts.push(format!(
                "{} merged, {} proposed",
                self.resolution.auto_merged, self.resolution.proposed
            ));
        }
        if parts.is_empty() {
            return f.write_str("graph: nothing new to extract");
        }
        write!(f, "graph: {}", parts.join("; "))
    }
}

/// Extract `documents` into the graph as `config.graph.follow_ingest`
/// says: `None` when that is `off` or there is no ontology. Table batches
/// go to the writer one at a time; the document pass makes one model call
/// per chunk, under `control`.
///
/// # Errors
///
/// Returns an error when a batch, the document pass, or resolution fails.
pub async fn after_documents(
    db: &Writer,
    config: &Config,
    embeddings: Option<&Embeddings>,
    documents: &[DocumentId],
    control: RunControl<'_>,
) -> Result<Option<FollowUpSummary>> {
    let mode = config.graph.follow_ingest;
    if mode.is_off() || documents.is_empty() {
        return Ok(None);
    }
    let Some(ontology) = db.run(ontology_store::current).await? else {
        return Ok(None);
    };
    let standing = db.run(ontology_store::current_standing).await?;
    let ids = documents.to_vec();
    let owned: BTreeSet<String> = db
        .run(move |db| {
            let mut tables = BTreeSet::new();
            for id in &ids {
                if let Some(document) = db.document(id)? {
                    tables.extend(document.tables.unwrap_or_default());
                }
            }
            Ok(tables)
        })
        .await?;
    let mut summary = FollowUpSummary::default();
    for mapping in ontology
        .mappings
        .iter()
        .filter(|m| owned.contains(&m.table))
    {
        let mut total = MappingSummary {
            table: mapping.table.clone(),
            ..MappingSummary::default()
        };
        let mut offset = 0;
        loop {
            let mapping = mapping.clone();
            let batch = db
                .run(move |db| tables::extract_batch(db, &mapping, standing, offset))
                .await?;
            total.absorb(&batch.summary);
            let Some(next) = batch.next_offset else {
                break;
            };
            offset = next;
        }
        summary.tables.push(total);
    }
    if mode.includes_documents() {
        let ids = documents.to_vec();
        let chunks = db
            .run(move |db| store::unextracted_chunks_of(db, &ids))
            .await?;
        if !chunks.is_empty() {
            let extractor = llm::graph_extractor(config, &ontology).await?;
            summary.chunks = Some(
                extract_chunks(
                    db,
                    config,
                    &ontology,
                    standing,
                    chunks,
                    extractor.as_ref(),
                    control,
                )
                .await?,
            );
        }
    }
    summary.resolution = resolve::resolve(db, embeddings, &config.graph).await?;
    let version = ontology.saved_version()?;
    db.run(move |db| store::set_built_with(db, version)).await?;
    Ok(Some(summary))
}

async fn extract_chunks(
    db: &Writer,
    config: &Config,
    ontology: &Ontology,
    standing: Standing,
    chunks: Vec<ChunkId>,
    extractor: &dyn Extract<extract::Extraction>,
    control: RunControl<'_>,
) -> Result<RunSummary> {
    extract::run(
        db,
        &ChunkPlan::Sample(chunks),
        ontology,
        standing,
        ExtractionRun {
            extractor,
            concurrency: config.analysis.extraction_concurrency,
            control,
        },
    )
    .await
}
