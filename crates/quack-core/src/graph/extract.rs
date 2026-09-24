//! Constrained extraction: each chunk goes to the chat model with the
//! ontology, and must answer with nodes and edges that fit it. Anything
//! outside the ontology is dropped and counted as drift (design doc 6.4).

use std::collections::BTreeMap;

use futures::StreamExt as _;
use serde::{Deserialize, Serialize};

use super::store::{self, NewNode, Source};
use super::{Drift, NormalizedLabel, Properties};
use crate::error::{Error, Result};
use crate::extraction::{Extracted, ExtractionRun, Passage, RunProgress, extractions};
use crate::ids::{ChunkId, DocumentId};
use crate::ontology::{self, Ontology, OntologyVersion};
use crate::storage::workspace::{ChunkSearchResult, SamplePool, WorkspaceDb};
use crate::storage::writer::Writer;

/// What the model returns for one chunk.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Extraction {
    #[serde(default)]
    pub nodes: Vec<ExtractedNode>,
    #[serde(default)]
    pub edges: Vec<ExtractedEdge>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedNode {
    pub label: String,
    pub class: String,
    #[serde(default)]
    pub properties: Properties,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    #[serde(default)]
    pub properties: Properties,
}

impl Ontology {
    /// The graph extractor's preamble, with this ontology rendered into it.
    #[must_use]
    pub fn extraction_prompt(&self) -> String {
        let mut prompt = String::from(
            "Extract the entities and relations a passage states, using only this ontology. \
         Return only JSON with this shape and nothing else:\n\
         {\"nodes\": [{\"label\": \"...\", \"class\": \"...\", \"properties\": {}}], \
         \"edges\": [{\"source\": \"...\", \"target\": \"...\", \"relation\": \"...\", \"properties\": {}}]}\n\
         Rules: `class` must be one of the class ids below and `relation` one of the relation \
         ids (use `mentions` when the passage links two entities without a listed relation). \
         `source` and `target` must be labels from `nodes`. Labels are the entity's name as \
         written, once each. Properties use the class's property ids with values from the \
         passage. Skip anything that does not fit the ontology; do not invent classes, \
         relations, or facts. Leave a list empty rather than guessing.\n\n",
        );
        prompt.push_str(&self.render_for_prompt());
        prompt
    }
}

/// An extraction filtered against the ontology: what to store, and what
/// the passage tried to say that the ontology has no place for.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Validated {
    pub nodes: Vec<ExtractedNode>,
    pub edges: Vec<ExtractedEdge>,
    pub drift: Drift,
    /// Edges dropped because their endpoints were not listed nodes or
    /// their classes did not fit the relation's domain and range.
    pub invalid_edges: u32,
}

impl Extraction {
    /// Keep what fits: nodes of known classes, edges of known relations
    /// between listed nodes whose classes fit the domain and range. Unknown
    /// classes and relations are counted as drift.
    #[must_use]
    pub fn validate(self, ontology: &Ontology) -> Validated {
        let mut out = Validated::default();
        let mut class_of: BTreeMap<NormalizedLabel, String> = BTreeMap::new();
        for node in self.nodes {
            let label = node.label.trim();
            if label.is_empty() {
                continue;
            }
            let class = node.class.trim().to_lowercase();
            if class != ontology::ROOT_CLASS && ontology.class(&class).is_none() {
                out.drift.classes.bump(&class);
                continue;
            }
            let key = NormalizedLabel::new(label);
            if class_of.contains_key(&key) {
                continue;
            }
            class_of.insert(key, class.clone());
            out.nodes.push(ExtractedNode {
                label: label.to_owned(),
                class,
                properties: node.properties,
            });
        }
        for edge in self.edges {
            let relation = edge.relation.trim().to_lowercase();
            if relation != ontology::MENTIONS_RELATION && ontology.relation(&relation).is_none() {
                out.drift.relations.bump(&relation);
                continue;
            }
            let (Some(source_class), Some(target_class)) = (
                class_of.get(&NormalizedLabel::new(&edge.source)),
                class_of.get(&NormalizedLabel::new(&edge.target)),
            ) else {
                out.invalid_edges = out.invalid_edges.saturating_add(1);
                continue;
            };
            if !ontology.allows_edge(&relation, source_class, target_class) {
                out.invalid_edges = out.invalid_edges.saturating_add(1);
                continue;
            }
            out.edges.push(ExtractedEdge {
                source: edge.source.trim().to_owned(),
                target: edge.target.trim().to_owned(),
                relation,
                properties: edge.properties,
            });
        }
        out
    }
}

/// A chunk to extract from.
#[derive(Debug, Clone)]
pub struct ChunkText {
    pub chunk_id: ChunkId,
    pub document_id: DocumentId,
    pub text: String,
}

impl Passage for ChunkText {
    fn id(&self) -> &str {
        self.chunk_id.as_str()
    }

    fn text(&self) -> &str {
        &self.text
    }
}

/// Chunks an extraction run reads from the workspace at a time: a run
/// holds one page's text, never every chunk's.
const PAGE: u32 = 64;

/// The chunks an extraction run covers, chosen before it starts: every
/// chunk of a ready document the graph has not extracted, or an even
/// sample of them (issue #60: the first N chunks by ingest order were one
/// document's front matter). A run records each chunk it processed, so
/// the next run sends only what is new; `graph extract --reset` clears
/// the record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkPlan {
    /// Every unextracted chunk, read a page at a time by id.
    All { total: usize },
    /// The sampled chunks, by id.
    Sample(Vec<ChunkId>),
}

impl ChunkPlan {
    /// Every unextracted chunk, or `sample` of them spread evenly over
    /// their documents. Only the sample's ids are read here, never text.
    ///
    /// # Errors
    ///
    /// Returns an error if a query fails.
    pub fn new(db: &WorkspaceDb, sample: Option<u32>) -> Result<Self> {
        let pool = SamplePool::NotGraphExtracted;
        Ok(match sample {
            None => Self::All {
                total: usize::try_from(db.pool_size(pool)?).unwrap_or(usize::MAX),
            },
            Some(limit) => Self::Sample(db.sample_chunk_ids(pool, limit)?),
        })
    }

    /// How many chunks the run will read.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::All { total } => *total,
            Self::Sample(ids) => ids.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The next page of the plan after `cursor`, which it advances; `None`
    /// once the plan is read.
    async fn page(&self, db: &Writer, cursor: &mut PlanCursor) -> Result<Option<Vec<ChunkText>>> {
        let chunks = match self {
            Self::All { .. } => {
                let after = cursor.after.take();
                db.run(move |db| db.chunk_page(SamplePool::NotGraphExtracted, after.as_ref(), PAGE))
                    .await?
            }
            Self::Sample(ids) => {
                let page: Vec<ChunkId> = ids
                    .iter()
                    .skip(cursor.consumed)
                    .take(usize::try_from(PAGE).unwrap_or(usize::MAX))
                    .cloned()
                    .collect();
                if page.is_empty() {
                    return Ok(None);
                }
                cursor.consumed = cursor.consumed.saturating_add(page.len());
                // A chunk whose document was deleted since is left out.
                db.run(move |db| db.chunks_by_ids(&page)).await?
            }
        };
        if let Self::All { .. } = self {
            let Some(last) = chunks.last() else {
                return Ok(None);
            };
            cursor.after = Some(last.id.clone());
        }
        Ok(Some(chunks.into_iter().map(ChunkText::from).collect()))
    }
}

/// How far a run has read its [`ChunkPlan`].
#[derive(Debug, Default)]
struct PlanCursor {
    /// Sampled ids taken so far.
    consumed: usize,
    /// The last chunk id read, for the next page of every chunk.
    after: Option<ChunkId>,
}

/// A chunk as the extractor reads it: its heading above its text.
impl From<ChunkSearchResult> for ChunkText {
    fn from(chunk: ChunkSearchResult) -> Self {
        Self {
            text: match chunk.heading {
                Some(h) => format!("{h}\n\n{}", chunk.content),
                None => chunk.content,
            },
            chunk_id: chunk.id,
            document_id: chunk.document_id,
        }
    }
}

/// What an extraction run did.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct RunSummary {
    pub chunks: u32,
    pub failed_chunks: u32,
    pub nodes: u32,
    pub edges: u32,
    pub invalid_edges: u32,
    pub drift: Drift,
}

/// Confidence recorded for model-extracted provenance.
const MODEL_CONFIDENCE: f64 = 0.8;

/// Run constrained extraction over the chunks `plan` names, a page at a
/// time with up to the run's concurrency of model calls in flight, and
/// store what fits; each finished chunk goes to the run's control, which
/// can also stop the run. A failed chunk is logged and skipped; only every
/// chunk failing is an error.
///
/// # Errors
///
/// Returns an error when every chunk fails, a write fails, or the run is
/// cancelled.
pub async fn run(
    db: &Writer,
    plan: &ChunkPlan,
    ontology: &Ontology,
    provisional: bool,
    extraction: ExtractionRun<'_, Extraction>,
) -> Result<RunSummary> {
    let ExtractionRun {
        extractor,
        concurrency,
        control,
    } = extraction;
    let mut pass = Pass {
        db,
        ontology,
        provisional,
        version: ontology.saved_version()?,
        summary: RunSummary::default(),
    };
    let mut run = RunProgress::new(plan.len(), control.progress);
    let mut cursor = PlanCursor::default();
    while let Some(page) = plan.page(db, &mut cursor).await? {
        let mut calls = extractions(extractor, &page, concurrency);
        while let Some(Extracted {
            passage: chunk,
            outcome,
            took,
        }) = control
            .or_cancelled(async { Ok(calls.next().await) })
            .await?
        {
            let kept = pass.keep(chunk, outcome).await?;
            run.finished(took, kept);
        }
    }
    let mut summary = pass.summary;
    summary.chunks = run.total();
    summary.failed_chunks = run.failed();
    if run.all_failed() {
        return Err(Error::Llm(String::from(
            "every chunk failed extraction; check the model and provider",
        )));
    }
    let drift = summary.drift.clone();
    db.run(move |db| store::record_drift(db, &drift)).await?;
    Ok(summary)
}

/// One extraction run's writes and tallies.
struct Pass<'a> {
    db: &'a Writer,
    ontology: &'a Ontology,
    provisional: bool,
    version: OntologyVersion,
    summary: RunSummary,
}

impl Pass<'_> {
    /// Validate and store one chunk's extraction, and record the chunk as
    /// extracted; whether the model answered.
    async fn keep(&mut self, chunk: &ChunkText, outcome: Result<Extraction>) -> Result<bool> {
        let extraction = match outcome {
            Ok(extraction) => extraction,
            Err(e) => {
                tracing::warn!(chunk = %chunk.chunk_id, error = %e, "extraction failed; skipping chunk");
                return Ok(false);
            }
        };
        tracing::debug!(
            chunk = %chunk.chunk_id,
            nodes = extraction.nodes.len(),
            edges = extraction.edges.len(),
            "graph extraction parsed"
        );
        let validated = extraction.validate(self.ontology);
        self.summary.invalid_edges = self
            .summary
            .invalid_edges
            .saturating_add(validated.invalid_edges);
        self.summary.drift.absorb(&validated.drift);
        let (document_id, chunk_id) = (chunk.document_id.clone(), chunk.chunk_id.clone());
        let (provisional, version) = (self.provisional, self.version);
        let (nodes, edges) = self
            .db
            .run(move |db| {
                db.under_timeout(|db| {
                    let counts = store_validated(
                        db,
                        &validated,
                        &Source::chunk(&document_id, &chunk_id, MODEL_CONFIDENCE),
                        provisional,
                    )?;
                    store::record_extracted(db, &chunk_id, version, counts)?;
                    Ok(counts)
                })
            })
            .await?;
        if nodes == 0 && edges == 0 {
            tracing::debug!(chunk = %chunk.chunk_id, "graph extraction kept nothing from this chunk");
        }
        self.summary.nodes = self.summary.nodes.saturating_add(nodes);
        self.summary.edges = self.summary.edges.saturating_add(edges);
        Ok(true)
    }
}

/// Store one validated extraction with `source` as provenance; returns the
/// node and edge counts touched.
///
/// # Errors
///
/// Returns an error if a write fails.
pub fn store_validated(
    db: &WorkspaceDb,
    validated: &Validated,
    source: &Source,
    provisional: bool,
) -> Result<(u32, u32)> {
    let mut ids: BTreeMap<NormalizedLabel, String> = BTreeMap::new();
    let mut nodes = 0u32;
    for node in &validated.nodes {
        let id = store::upsert_node(
            db,
            &NewNode {
                label: node.label.clone(),
                class_id: node.class.clone(),
                properties: node.properties.clone(),
                provisional,
            },
        )?;
        store::add_provenance(db, &id, source)?;
        ids.insert(NormalizedLabel::new(&node.label), id);
        nodes = nodes.saturating_add(1);
    }
    let mut edges = 0u32;
    for edge in &validated.edges {
        let (Some(s), Some(t)) = (
            ids.get(&NormalizedLabel::new(&edge.source)),
            ids.get(&NormalizedLabel::new(&edge.target)),
        ) else {
            continue;
        };
        let id = store::upsert_edge(db, s, t, &edge.relation, &edge.properties, provisional)?;
        store::add_provenance(db, &id, source)?;
        edges = edges.saturating_add(1);
    }
    Ok((nodes, edges))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::parse_answer;
    use crate::ontology::{Class, Relation};

    fn ontology() -> Ontology {
        let mut o = Ontology::default();
        o.classes.push(Class {
            id: String::from("organization"),
            parent: String::from(ontology::ROOT_CLASS),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        });
        o.classes.push(Class {
            id: String::from("vendor"),
            parent: String::from("organization"),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        });
        o.classes.push(Class {
            id: String::from("country"),
            parent: String::from(ontology::ROOT_CLASS),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        });
        o.relations.push(Relation {
            id: String::from("ships_to"),
            label: None,
            description: None,
            domain: String::from("organization"),
            range: String::from("country"),
        });
        o
    }

    #[test]
    fn validate_keeps_what_fits_and_counts_the_rest() {
        let extraction = parse_answer::<Extraction>(
            r#"Here you go: {"nodes": [
                {"label": "Orgenics", "class": "Vendor"},
                {"label": "Kenya", "class": "country", "properties": {"region": "East Africa"}},
                {"label": "MV Hope", "class": "vessel"},
                {"label": "orgenics", "class": "vendor"},
                {"label": "  ", "class": "country"}
            ], "edges": [
                {"source": "Orgenics", "target": "Kenya", "relation": "ships_to"},
                {"source": "Kenya", "target": "Orgenics", "relation": "ships_to"},
                {"source": "Orgenics", "target": "MV Hope", "relation": "mentions"},
                {"source": "Orgenics", "target": "Kenya", "relation": "docked_at"},
                {"source": "Orgenics", "target": "Kenya", "relation": "mentions", "properties": 3}
            ]}"#,
        )
        .unwrap_or_default();
        let v = extraction.validate(&ontology());
        assert_eq!(
            v.nodes
                .iter()
                .map(|n| (n.label.as_str(), n.class.as_str()))
                .collect::<Vec<_>>(),
            [("Orgenics", "vendor"), ("Kenya", "country")]
        );
        assert_eq!(
            v.edges
                .iter()
                .map(|e| e.relation.as_str())
                .collect::<Vec<_>>(),
            ["ships_to", "mentions"]
        );
        assert_eq!(
            v.invalid_edges, 2,
            "range mismatch and an unlisted endpoint"
        );
        assert_eq!(v.drift.classes.get("vessel"), 1);
        assert_eq!(v.drift.relations.get("docked_at"), 1);
        assert!(v.edges.iter().all(|e| e.properties.is_empty()));
    }

    #[test]
    fn parse_rejects_prose_and_the_wrong_shape() {
        assert!(parse_answer::<Extraction>("no json here").is_err());
        assert!(parse_answer::<Extraction>(r#"{"nodes": "nope"}"#).is_err());
        assert_eq!(
            parse_answer::<Extraction>("{}").map_or(9, |e| e.nodes.len()),
            0
        );
    }

    #[test]
    fn prompt_carries_the_ontology() {
        let prompt = ontology().extraction_prompt();
        assert!(prompt.contains("ships_to: organization -> country"));
        assert!(prompt.starts_with("Extract the entities"));
    }
}
