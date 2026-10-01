//! Constrained extraction: each chunk goes to the chat model with the
//! ontology, and must answer with nodes and edges that fit it. Anything
//! outside the ontology is dropped and counted as drift (design doc 6.4).

use std::collections::BTreeMap;

use futures::StreamExt as _;
use schemars::Schema;
use serde::{Deserialize, Serialize};

use super::store::{self, ChunkYield, NewNode, Source};
use super::{Drift, NormalizedLabel, Properties, Standing};
use crate::error::{Error, Result};
use crate::extraction::{Extracted, ExtractionRun, Passage, RunProgress, extractions};
use crate::ids::{ChunkId, ClassId, DocumentId, NodeId};
use crate::ontology::{self, Ontology, OntologyVersion};
use crate::storage::workspace::{ChunkSearchResult, SamplePool, WorkspaceDb};
use crate::storage::writer::Writer;

/// What the model returns for one chunk.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Extraction {
    #[serde(default)]
    pub nodes: Vec<ExtractedNode>,
    #[serde(default)]
    pub edges: Vec<ExtractedEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedNode {
    pub label: String,
    pub class: String,
    #[serde(default)]
    pub properties: Properties,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    #[serde(default)]
    pub properties: Properties,
}

/// What the model answers for one chunk, as [`Ontology::extraction_schema`]
/// describes it. Properties are name and value pairs: a strict
/// structured-output schema (`OpenAI`'s, Anthropic's) cannot describe an
/// object with free keys.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct ExtractionAnswer {
    #[serde(default)]
    nodes: Vec<NodeAnswer>,
    #[serde(default)]
    edges: Vec<EdgeAnswer>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct NodeAnswer {
    label: String,
    class: String,
    #[serde(default)]
    properties: Vec<PropertyAnswer>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct EdgeAnswer {
    source: String,
    target: String,
    relation: String,
    #[serde(default)]
    properties: Vec<PropertyAnswer>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct PropertyAnswer {
    name: String,
    value: String,
}

impl From<Vec<PropertyAnswer>> for Properties {
    fn from(pairs: Vec<PropertyAnswer>) -> Self {
        Self::from(serde_json::Value::Object(
            pairs
                .into_iter()
                .map(|p| (p.name, serde_json::Value::String(p.value)))
                .collect(),
        ))
    }
}

impl From<ExtractionAnswer> for Extraction {
    fn from(answer: ExtractionAnswer) -> Self {
        Self {
            nodes: answer
                .nodes
                .into_iter()
                .map(|n| ExtractedNode {
                    label: n.label,
                    class: n.class,
                    properties: n.properties.into(),
                })
                .collect(),
            edges: answer
                .edges
                .into_iter()
                .map(|e| ExtractedEdge {
                    source: e.source,
                    target: e.target,
                    relation: e.relation,
                    properties: e.properties.into(),
                })
                .collect(),
        }
    }
}

impl Ontology {
    /// The shape the graph extractor's answer must have: `class` one of
    /// this ontology's class ids, `relation` one of its relation ids
    /// (`mentions` among them), and property names its property ids. A
    /// provider that holds the model to it leaves nothing to count as drift.
    #[must_use]
    pub fn extraction_schema(&self) -> Schema {
        let mut names: Vec<&str> = self.properties.iter().map(|p| p.id.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        let name = if names.is_empty() {
            serde_json::json!({ "type": "string" })
        } else {
            serde_json::json!({ "type": "string", "enum": names })
        };
        let properties = serde_json::json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": { "name": name, "value": { "type": "string" } },
                "required": ["name", "value"],
            },
        });
        Schema::try_from(serde_json::json!({
            "title": "graph_extraction",
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": { "type": "string" },
                            "class": { "type": "string", "enum": self.class_ids() },
                            "properties": properties,
                        },
                        "required": ["label", "class", "properties"],
                    },
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": { "type": "string" },
                            "target": { "type": "string" },
                            "relation": { "type": "string", "enum": self.relation_ids() },
                            "properties": properties,
                        },
                        "required": ["source", "target", "relation", "properties"],
                    },
                },
            },
            "required": ["nodes", "edges"],
        }))
        .unwrap_or_default()
    }

    /// The graph extractor's preamble, with this ontology rendered into it.
    #[must_use]
    pub fn extraction_prompt(&self) -> String {
        let mut prompt = String::from(
            "Extract the entities and relations a passage states, using only this ontology. \
         Answer with `nodes` (each a `label`, a `class`, and `properties` as name and value \
         pairs) and `edges` (each a `source`, a `target`, a `relation`, and `properties`).\n\
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
    standing: Standing,
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
        standing,
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
    standing: Standing,
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
        let (standing, version) = (self.standing, self.version);
        let ChunkYield { nodes, edges } = self
            .db
            .run(move |db| {
                db.under_timeout(|db| {
                    let counts = store_validated(
                        db,
                        &validated,
                        &Source::chunk(&document_id, &chunk_id, MODEL_CONFIDENCE),
                        standing,
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
    standing: Standing,
) -> Result<ChunkYield> {
    let mut ids: BTreeMap<NormalizedLabel, NodeId> = BTreeMap::new();
    let mut nodes = 0u32;
    for node in &validated.nodes {
        let id = store::upsert_node(
            db,
            &NewNode {
                label: node.label.clone(),
                class_id: ClassId::from(node.class.clone()),
                properties: node.properties.clone(),
                standing,
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
        let id = store::upsert_edge(db, s, t, &edge.relation, &edge.properties, standing)?;
        store::add_provenance(db, &id, source)?;
        edges = edges.saturating_add(1);
    }
    Ok(ChunkYield { nodes, edges })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::RelationId;
    use crate::ontology::{Class, Relation};

    fn ontology() -> Ontology {
        let mut o = Ontology::default();
        o.classes.push(Class {
            id: ClassId::from("organization"),
            parent: ClassId::from(String::from(ontology::ROOT_CLASS)),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        });
        o.classes.push(Class {
            id: ClassId::from("vendor"),
            parent: ClassId::from("organization"),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        });
        o.classes.push(Class {
            id: ClassId::from("country"),
            parent: ClassId::from(String::from(ontology::ROOT_CLASS)),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        });
        o.relations.push(Relation {
            id: RelationId::from("ships_to"),
            label: None,
            description: None,
            domain: ClassId::from("organization"),
            range: ClassId::from("country"),
        });
        o
    }

    #[test]
    fn validate_keeps_what_fits_and_counts_the_rest() {
        let extraction = serde_json::from_str::<Extraction>(
            r#"{"nodes": [
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

    /// The ids an `enum` in the schema at `path` allows.
    fn allowed(schema: &Schema, path: &str) -> Vec<String> {
        schema
            .as_value()
            .pointer(path)
            .and_then(serde_json::Value::as_array)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| id.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn the_schema_allows_exactly_the_ontologys_ids() {
        let mut ontology = ontology();
        ontology.properties.push(ontology::Property {
            id: String::from("region"),
            label: None,
            kind: ontology::PropertyType::String,
            values: Vec::new(),
        });
        let schema = ontology.extraction_schema();
        assert_eq!(
            allowed(&schema, "/properties/nodes/items/properties/class/enum"),
            ["entity", "organization", "vendor", "country"]
        );
        assert_eq!(
            allowed(&schema, "/properties/edges/items/properties/relation/enum"),
            ["mentions", "ships_to"]
        );
        assert_eq!(
            allowed(
                &schema,
                "/properties/nodes/items/properties/properties/items/properties/name/enum"
            ),
            ["region"]
        );
    }

    #[test]
    fn an_answer_reads_its_property_pairs_as_properties() {
        let answer: ExtractionAnswer = serde_json::from_str(
            r#"{"nodes": [{"label": "Kenya", "class": "country",
                 "properties": [{"name": "region", "value": "East Africa"}]}],
               "edges": [{"source": "Orgenics", "target": "Kenya", "relation": "ships_to",
                 "properties": []}]}"#,
        )
        .unwrap_or_default();
        let extraction = Extraction::from(answer);
        assert_eq!(
            extraction
                .nodes
                .first()
                .and_then(|n| n.properties.get("region")),
            Some(&serde_json::json!("East Africa"))
        );
        assert_eq!(extraction.edges.len(), 1);
        assert!(serde_json::from_str::<ExtractionAnswer>(r#"{"nodes": "nope"}"#).is_err());
    }

    #[test]
    fn prompt_carries_the_ontology() {
        let prompt = ontology().extraction_prompt();
        assert!(prompt.contains("ships_to: organization -> country"));
        assert!(prompt.starts_with("Extract the entities"));
    }
}
