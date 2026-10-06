//! Ontology induction from document evidence (design doc 6.5): open
//! extraction on a stratified sample of chunks, vocabulary normalization,
//! structure inference, and support scoring. Model calls go through
//! [`Extract`](crate::extraction::Extract), so the pipeline is tested with a canned one.

use std::collections::{BTreeMap, BTreeSet};

use futures::StreamExt as _;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::induction::{Candidate, Proposal};
use super::{Class, Ontology, Property, PropertyType, ROOT_CLASS, Relation, SnakeId};
use crate::embedding::Input;
use crate::error::{Error, Result};
use crate::extraction::{Extracted, ExtractionRun, Passage, RunProgress, Tally, extractions};
use crate::graph::NormalizedLabel;
use crate::ids::{ChunkId, ClassId, DocumentId, RelationId};
use crate::llm::Embeddings;
use crate::storage::workspace::{DocumentStatus, SamplePool, WorkspaceDb};

/// What open extraction returns for one chunk; the schema it derives is
/// what the model is held to.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "open_extraction")]
pub struct OpenExtraction {
    #[serde(default)]
    pub entities: Vec<OpenEntity>,
    #[serde(default)]
    pub relations: Vec<OpenRelation>,
    #[serde(default)]
    pub attributes: Vec<OpenAttribute>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct OpenEntity {
    pub name: String,
    #[serde(rename = "type")]
    pub type_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct OpenRelation {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct OpenAttribute {
    pub entity: String,
    pub name: String,
    pub value: String,
}

/// The open extractor's preamble; [`OpenExtraction`]'s schema carries the
/// answer's shape.
pub const EXTRACTION_PROMPT: &str = "Read the passage and list what it mentions: its \
`entities` (each a `name` and a `type`), the `relations` between them (a `subject`, a \
`relation`, and an `object`), and `attributes` (an `entity`, a `name`, and a `value`).\n\
Types and relation names are short lowercase nouns or verbs (organization, vendor, \
country, shipped_to, issued_by). Subjects and objects of relations must be entity names \
from the list. Attributes are facts with a value (amount: 4500, effective_date: 2024-03-01). \
Leave a list empty rather than inventing.";

/// Whether two distinct ids name the same thing (an embedding cosine
/// check); `None` means exact matching only.
pub type Similarity<'a> = Option<&'a dyn Fn(&str, &str) -> bool>;

/// Which pairs of type or relation names mean the same thing: those whose
/// embeddings' cosine is at or above a threshold.
#[derive(Debug, Default)]
pub struct SimilarNames(std::collections::HashSet<(String, String)>);

impl SimilarNames {
    /// Embed `names` and keep every pair at or above `threshold`.
    ///
    /// # Errors
    ///
    /// Returns an error when embedding fails.
    pub async fn embed(embedder: &Embeddings, names: &[String], threshold: f64) -> Result<Self> {
        let mut out = Self::default();
        if names.len() < 2 {
            return Ok(out);
        }
        let inputs: Vec<Input> = names
            .iter()
            .map(|n| Input::Similarity(n.replace('_', " ")))
            .collect();
        let vectors = embedder.embed(&inputs).await?;
        for (i, (a, va)) in names.iter().zip(&vectors).enumerate() {
            for (b, vb) in names.iter().zip(&vectors).skip(i.saturating_add(1)) {
                if va.cosine(vb) >= threshold {
                    out.0.insert((a.clone(), b.clone()));
                }
            }
        }
        Ok(out)
    }

    /// Whether `a` and `b` name the same thing, in either order.
    #[must_use]
    pub fn same(&self, a: &str, b: &str) -> bool {
        self.0.contains(&(a.to_owned(), b.to_owned()))
            || self.0.contains(&(b.to_owned(), a.to_owned()))
    }
}

/// Tuning for document evidence.
#[derive(Debug, Clone, Copy)]
pub struct DocumentEvidenceOptions {
    /// Chunks sampled across documents.
    pub sample_chunks: u32,
    /// Distinct documents a candidate needs to be shown in the main
    /// proposal; below it the candidate is stored as `low_support`.
    pub min_support_documents: u32,
    /// Cosine similarity at which two type or relation names cluster
    /// when an embedding model is available.
    pub cluster_threshold: f64,
}

impl Default for DocumentEvidenceOptions {
    fn default() -> Self {
        Self {
            sample_chunks: 200,
            min_support_documents: 3,
            cluster_threshold: 0.9,
        }
    }
}

/// A sampled chunk.
#[derive(Debug, Clone)]
pub struct SampledChunk {
    pub id: ChunkId,
    pub document_id: DocumentId,
    pub filename: String,
    pub content: String,
}

impl Passage for SampledChunk {
    fn id(&self) -> &str {
        self.id.as_str()
    }

    fn text(&self) -> &str {
        &self.content
    }
}

/// What a run will cost, shown before it starts.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct CostEstimate {
    pub documents: u32,
    pub chunks: u32,
    /// One extraction call per sampled chunk.
    pub model_calls: u32,
}

/// The documents and chunks a sample would cover.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn estimate(db: &WorkspaceDb, options: &DocumentEvidenceOptions) -> Result<CostEstimate> {
    let (documents, chunks): (i64, i64) = db.connection().query_row(
        "SELECT count(DISTINCT c.document_id), count(*) FROM _quack_chunks c \
         JOIN _quack_documents d ON d.id = c.document_id \
         WHERE d.status = ? AND length(c.content) > 40",
        [DocumentStatus::Ready],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    let documents = u32::try_from(documents).unwrap_or(u32::MAX);
    let chunks = u32::try_from(chunks)
        .unwrap_or(u32::MAX)
        .min(options.sample_chunks);
    Ok(CostEstimate {
        documents,
        chunks,
        model_calls: chunks,
    })
}

/// A stratified sample: every ready document contributes chunks spaced
/// evenly through it, up to `sample_chunks` in total.
///
/// # Errors
///
/// Returns an error if a query fails.
pub fn sample_chunks(db: &WorkspaceDb, sample: u32) -> Result<Vec<SampledChunk>> {
    let ids = db.sample_chunk_ids(SamplePool::Substantive, sample)?;
    Ok(db
        .chunks_by_ids(&ids)?
        .into_iter()
        .map(|chunk| SampledChunk {
            id: chunk.id,
            document_id: chunk.document_id,
            filename: chunk.filename,
            content: chunk.content,
        })
        .collect())
}

/// The outcome of a document-evidence run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunSummary {
    pub sampled_chunks: u32,
    pub failed_chunks: u32,
    pub candidates: u32,
    pub low_support: u32,
}

/// Sample, extract, cluster, and propose: the whole document-evidence
/// pass. With an embedding model, near-synonymous type and relation names
/// cluster by cosine similarity; without one, only exact ids merge. The
/// caller samples and stores, so the model calls run without holding the
/// workspace.
///
/// # Errors
///
/// Returns an error when every extraction failed or embedding fails.
pub async fn run(
    sample: Vec<SampledChunk>,
    current: Option<&Ontology>,
    options: &DocumentEvidenceOptions,
    embeddings: Option<&Embeddings>,
    extraction: ExtractionRun<'_, OpenExtraction>,
) -> Result<DocumentProposal> {
    let sampled = u32::try_from(sample.len()).unwrap_or(u32::MAX);
    let Observed {
        observations,
        failed,
    } = observe(&sample, extraction).await?;
    let table = match embeddings {
        Some(model) => {
            let mut names: BTreeSet<String> = BTreeSet::new();
            for o in &observations {
                for e in &o.extraction.entities {
                    names.insert(SnakeId::singular_from(&e.type_name).into_string());
                }
                for r in &o.extraction.relations {
                    names.insert(SnakeId::singular_from(&r.relation).into_string());
                }
            }
            let names: Vec<String> = names.into_iter().collect();
            Some(SimilarNames::embed(model, &names, options.cluster_threshold).await?)
        }
        None => None,
    };
    let lookup: &dyn Fn(&str, &str) -> bool = &|a, b| table.as_ref().is_some_and(|t| t.same(a, b));
    let similarity: Similarity<'_> = table.is_some().then_some(lookup);
    let candidates = propose(&observations, current, options, similarity);
    let low = u32::try_from(candidates.iter().filter(|c| c.low_support).count()).unwrap_or(0);
    let summary = RunSummary {
        sampled_chunks: sampled,
        failed_chunks: failed,
        candidates: u32::try_from(candidates.len()).unwrap_or(u32::MAX),
        low_support: low,
    };
    Ok(DocumentProposal {
        candidates,
        summary,
    })
}

/// What the document pass proposes, and what the run did.
#[derive(Debug, Clone)]
pub struct DocumentProposal {
    pub candidates: Vec<Candidate>,
    pub summary: RunSummary,
}

/// The extractions that came back, and how many chunks failed.
#[derive(Debug, Clone)]
pub struct Observed {
    pub observations: Vec<Observation>,
    pub failed: u32,
}

/// One extraction with where it came from.
#[derive(Debug, Clone)]
pub struct Observation {
    pub chunk_id: ChunkId,
    pub document_id: DocumentId,
    pub extraction: OpenExtraction,
}

/// Run the extractor over the sample, up to the run's concurrency of
/// chunks at a time, reporting each finished chunk to its control. Failed
/// chunks are skipped and counted, so one bad answer does not sink the
/// run.
///
/// # Errors
///
/// Returns an error only when every chunk failed.
pub async fn observe(
    chunks: &[SampledChunk],
    extraction: ExtractionRun<'_, OpenExtraction>,
) -> Result<Observed> {
    let ExtractionRun {
        extractor,
        concurrency,
        control,
    } = extraction;
    let mut run = RunProgress::new(chunks.len(), control.progress);
    let mut calls = extractions(extractor, chunks, concurrency);
    let mut observations = Vec::with_capacity(chunks.len());
    while let Some(Extracted {
        passage: chunk,
        outcome,
        took,
    }) = control
        .or_cancelled(async { Ok(calls.next().await) })
        .await?
    {
        match outcome {
            Ok(extraction) => {
                observations.push(Observation {
                    chunk_id: chunk.id.clone(),
                    document_id: chunk.document_id.clone(),
                    extraction,
                });
                run.finished(took, true);
            }
            Err(e) => {
                tracing::warn!(chunk = %chunk.id, error = %e, "extraction failed for a chunk");
                run.finished(took, false);
            }
        }
    }
    drop(calls);
    if run.all_failed() {
        return Err(Error::Ontology(format!(
            "extraction failed for all {} sampled chunks",
            chunks.len()
        )));
    }
    Ok(Observed {
        observations,
        failed: run.failed(),
    })
}

/// A canonical id per raw name. Exact `snake_case` ids and their plurals
/// merge always; near-synonyms merge when `similarity` says so.
pub struct Vocabulary {
    canonical: BTreeMap<String, String>,
}

impl Vocabulary {
    /// Build from the raw names with their frequencies. `similarity`
    /// receives two distinct ids and answers whether they name the same
    /// thing (an embedding cosine check, or `None` for exact matching only).
    #[must_use]
    pub fn build(counts: &Tally, similarity: Similarity<'_>) -> Self {
        let mut groups: Vec<(String, Vec<String>)> = Vec::new();
        for raw in counts.names() {
            let id = SnakeId::singular_from(raw).into_string();
            let mut placed = false;
            for (leader, members) in &mut groups {
                let same = *leader == id || similarity.is_some_and(|f| f(leader, &id));
                if same {
                    members.push(raw.to_owned());
                    placed = true;
                    break;
                }
            }
            if !placed {
                groups.push((id, vec![raw.to_owned()]));
            }
        }
        let mut canonical = BTreeMap::new();
        for (_, members) in groups {
            // The most frequent raw name decides the id.
            let best = members
                .iter()
                .max_by_key(|m| counts.get(m))
                .cloned()
                .unwrap_or_default();
            let id = SnakeId::singular_from(&best).into_string();
            for member in members {
                canonical.insert(member, id.clone());
            }
        }
        Self { canonical }
    }

    #[must_use]
    pub fn id(&self, raw: &str) -> String {
        self.canonical
            .get(raw)
            .cloned()
            .unwrap_or_else(|| SnakeId::singular_from(raw).into_string())
    }
}

#[derive(Default)]
struct Support {
    occurrences: u32,
    documents: BTreeSet<DocumentId>,
    examples: Vec<serde_json::Value>,
}

impl Support {
    fn note(&mut self, observation: &Observation, example: serde_json::Value) {
        self.occurrences = self.occurrences.saturating_add(1);
        self.documents.insert(observation.document_id.clone());
        if self.examples.len() < 3 {
            let mut example = example;
            if let serde_json::Value::Object(map) = &mut example {
                map.insert(
                    String::from("chunk_id"),
                    serde_json::Value::String(observation.chunk_id.to_string()),
                );
            }
            self.examples.push(example);
        }
    }
    fn confidence(&self, min_support: u32) -> f64 {
        let by_count = f64::from(self.occurrences.min(10)) / 10.0;
        let docs = u32::try_from(self.documents.len()).unwrap_or(u32::MAX);
        let by_docs = f64::from(docs.min(min_support.max(1))) / f64::from(min_support.max(1));
        0.5f64.mul_add(by_count, 0.5 * by_docs)
    }
    /// Seen in fewer documents than a proposal needs to be shown.
    fn is_low(&self, min_support_documents: u32) -> bool {
        u32::try_from(self.documents.len()).unwrap_or(0) < min_support_documents
    }
    fn evidence(&self, extra: serde_json::Value) -> serde_json::Value {
        let mut evidence = serde_json::json!({
            "occurrences": self.occurrences,
            "documents": self.documents.len(),
            "examples": self.examples,
        });
        if let (serde_json::Value::Object(map), serde_json::Value::Object(more)) =
            (&mut evidence, extra)
        {
            map.extend(more);
        }
        evidence
    }
}

impl PropertyType {
    /// The type every non-blank value fits: boolean, number, date, else
    /// string.
    fn infer(values: &[String]) -> Self {
        let trimmed: Vec<&str> = values
            .iter()
            .map(|v| v.trim())
            .filter(|v| !v.is_empty())
            .collect();
        if trimmed.is_empty() {
            return Self::String;
        }
        if trimmed
            .iter()
            .all(|v| v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("false"))
        {
            return Self::Boolean;
        }
        if trimmed
            .iter()
            .all(|v| v.replace([',', '$', '%'], "").parse::<f64>().is_ok())
        {
            return Self::Number;
        }
        let looks_like_date = |value: &str| {
            let digits = value.chars().filter(char::is_ascii_digit).count();
            digits >= 4
                && (value.contains('-') || value.contains('/') || value.contains(' '))
                && value.len() <= 30
        };
        if trimmed.iter().all(|v| looks_like_date(v)) {
            return Self::Date;
        }
        Self::String
    }
}

/// One attribute of one class: how often documents gave it, and the
/// values they gave.
#[derive(Default)]
struct AttributeStats {
    support: Support,
    values: Vec<String>,
}

/// Parents inferred from co-labelled mentions: a class whose entities are
/// mostly also labelled with a bigger class gets it as parent (`vendor`
/// under `organization`).
struct Hierarchy(BTreeMap<String, String>);

impl Hierarchy {
    fn infer(
        entity_types: &BTreeMap<NormalizedLabel, BTreeSet<String>>,
        class_support: &BTreeMap<String, Support>,
    ) -> Self {
        let mut pairs: BTreeMap<(String, String), u32> = BTreeMap::new();
        for classes in entity_types.values() {
            for a in classes {
                for b in classes {
                    if a != b {
                        let entry = pairs.entry((a.clone(), b.clone())).or_default();
                        *entry = entry.saturating_add(1);
                    }
                }
            }
        }
        let size = |c: &str| class_support.get(c).map_or(0, |s| s.occurrences);
        let entities_of = |c: &str| entity_types.values().filter(|set| set.contains(c)).count();
        let mut parents = BTreeMap::new();
        for ((child, parent), shared) in &pairs {
            if size(child) >= size(parent) {
                continue;
            }
            let child_entities = u32::try_from(entities_of(child)).unwrap_or(u32::MAX).max(1);
            if f64::from(*shared) / f64::from(child_entities) >= 0.8 {
                let better = parents
                    .get(child)
                    .is_none_or(|existing: &String| size(existing) < size(parent));
                if better {
                    parents.insert(child.clone(), parent.clone());
                }
            }
        }
        Self(parents)
    }

    fn parent(&self, class: &str) -> Option<&String> {
        self.0.get(class)
    }

    /// The class and its inferred ancestors, nearest first.
    fn ancestry(&self, class: &str) -> Vec<String> {
        let mut out = vec![class.to_owned()];
        let mut current = class;
        while let Some(parent) = self.0.get(current) {
            if out.contains(parent) {
                break;
            }
            out.push(parent.clone());
            current = parent;
        }
        out
    }

    /// The single most common endpoint class, or the nearest ancestor every
    /// endpoint shares when they are mixed, or `entity`.
    fn generalize(&self, counts: &Tally) -> String {
        let total = counts.total();
        let Some((top, n)) = counts.iter().max_by_key(|(_, n)| *n) else {
            return String::from(ROOT_CLASS);
        };
        if total == 0 {
            return String::from(ROOT_CLASS);
        }
        if f64::from(n) / f64::from(total) >= 0.8 {
            return top.to_owned();
        }
        for ancestor in self.ancestry(top) {
            if counts.names().all(|c| self.ancestry(c).contains(&ancestor)) {
                return ancestor;
            }
        }
        String::from(ROOT_CLASS)
    }
}

/// Endpoint counts for one relation.
#[derive(Default)]
struct RelationStats {
    support: Option<Support>,
    domains: Tally,
    ranges: Tally,
}

/// What the observations say, before it becomes candidates.
struct Evidence {
    class_support: BTreeMap<String, Support>,
    entity_class: BTreeMap<NormalizedLabel, String>,
    hierarchy: Hierarchy,
    relations: Vocabulary,
}

impl Evidence {
    fn gather(observations: &[Observation], similarity: Similarity<'_>) -> Self {
        let mut type_counts = Tally::default();
        let mut relation_counts = Tally::default();
        for o in observations {
            for e in &o.extraction.entities {
                type_counts.bump(&e.type_name);
            }
            for r in &o.extraction.relations {
                relation_counts.bump(&r.relation);
            }
        }
        let types = Vocabulary::build(&type_counts, similarity);
        let relations = Vocabulary::build(&relation_counts, similarity);
        let mut class_support: BTreeMap<String, Support> = BTreeMap::new();
        let mut entity_types: BTreeMap<NormalizedLabel, BTreeSet<String>> = BTreeMap::new();
        let mut entity_class: BTreeMap<NormalizedLabel, String> = BTreeMap::new();
        for o in observations {
            for e in &o.extraction.entities {
                let class = types.id(&e.type_name);
                class_support.entry(class.clone()).or_default().note(
                    o,
                    serde_json::json!({ "mention": e.name, "type": e.type_name }),
                );
                let key = NormalizedLabel::new(&e.name);
                entity_types
                    .entry(key.clone())
                    .or_default()
                    .insert(class.clone());
                entity_class.entry(key).or_insert(class);
            }
        }
        let hierarchy = Hierarchy::infer(&entity_types, &class_support);
        Self {
            class_support,
            entity_class,
            hierarchy,
            relations,
        }
    }
}

/// Turn observations into candidates: classes with an inferred hierarchy,
/// relations with domain and range, recurring attributes as properties.
/// Candidates below `min_support_documents` are marked low support. With
/// `current`, ids the ontology already has are skipped.
#[must_use]
pub fn propose(
    observations: &[Observation],
    current: Option<&Ontology>,
    options: &DocumentEvidenceOptions,
    similarity: Similarity<'_>,
) -> Vec<Candidate> {
    let evidence = Evidence::gather(observations, similarity);
    let mut candidates = Vec::new();
    propose_classes(&evidence, current, options, &mut candidates);
    propose_relations(observations, &evidence, current, options, &mut candidates);
    propose_attributes(observations, &evidence, current, options, &mut candidates);
    candidates
}

fn propose_classes(
    evidence: &Evidence,
    current: Option<&Ontology>,
    options: &DocumentEvidenceOptions,
    candidates: &mut Vec<Candidate>,
) {
    for (class, support) in &evidence.class_support {
        if current.map_or(class == ROOT_CLASS, |o| o.defines_class(class)) {
            continue;
        }
        let parent = evidence
            .hierarchy
            .parent(class)
            .cloned()
            .unwrap_or_else(|| String::from(ROOT_CLASS));
        candidates.push(Candidate {
            proposal: Proposal::Class(Class {
                id: ClassId::from(class.clone()),
                parent: ClassId::from(parent),
                label: None,
                description: None,
                key: None,
                properties: Vec::new(),
            }),
            evidence: support.evidence(serde_json::json!({ "source": "documents" })),
            confidence: support.confidence(options.min_support_documents),
            low_support: support.is_low(options.min_support_documents),
        });
    }
}

fn propose_relations(
    observations: &[Observation],
    evidence: &Evidence,
    current: Option<&Ontology>,
    options: &DocumentEvidenceOptions,
    candidates: &mut Vec<Candidate>,
) {
    let mut stats: BTreeMap<String, RelationStats> = BTreeMap::new();
    for o in observations {
        for r in &o.extraction.relations {
            let id = evidence.relations.id(&r.relation);
            let entry = stats.entry(id).or_default();
            entry.support.get_or_insert_with(Support::default).note(
                o,
                serde_json::json!({ "subject": r.subject, "relation": r.relation, "object": r.object }),
            );
            if let Some(c) = evidence.entity_class.get(&NormalizedLabel::new(&r.subject)) {
                entry.domains.bump(c);
            }
            if let Some(c) = evidence.entity_class.get(&NormalizedLabel::new(&r.object)) {
                entry.ranges.bump(c);
            }
        }
    }
    for (id, stat) in &stats {
        let Some(support) = &stat.support else {
            continue;
        };
        if current.map_or(id == super::MENTIONS_RELATION, |o| o.defines_relation(id)) {
            continue;
        }
        candidates.push(Candidate {
            proposal: Proposal::Relation(Relation {
                id: RelationId::from(id.clone()),
                label: None,
                description: None,
                domain: ClassId::from(evidence.hierarchy.generalize(&stat.domains)),
                range: ClassId::from(evidence.hierarchy.generalize(&stat.ranges)),
            }),
            evidence: support.evidence(serde_json::json!({
                "source": "documents",
                "domains": stat.domains,
                "ranges": stat.ranges,
            })),
            confidence: support.confidence(options.min_support_documents),
            low_support: support.is_low(options.min_support_documents),
        });
    }
}

fn propose_attributes(
    observations: &[Observation],
    evidence: &Evidence,
    current: Option<&Ontology>,
    options: &DocumentEvidenceOptions,
    candidates: &mut Vec<Candidate>,
) {
    let mut stats: BTreeMap<(String, String), AttributeStats> = BTreeMap::new();
    for o in observations {
        for a in &o.extraction.attributes {
            let Some(class) = evidence.entity_class.get(&NormalizedLabel::new(&a.entity)) else {
                continue;
            };
            let property = SnakeId::from_name(&a.name).into_string();
            let entry = stats.entry((class.clone(), property)).or_default();
            entry.support.note(
                o,
                serde_json::json!({ "entity": a.entity, "value": a.value }),
            );
            entry.values.push(a.value.clone());
        }
    }
    for ((class, property), AttributeStats { support, values }) in &stats {
        if support.occurrences < 2
            || current.is_some_and(|o| o.class_properties(class).contains(property))
        {
            continue;
        }
        candidates.push(Candidate {
            proposal: Proposal::Property {
                class: class.clone(),
                property: Property {
                    id: property.clone(),
                    label: None,
                    kind: PropertyType::infer(values),
                    values: Vec::new(),
                },
            },
            evidence: support.evidence(serde_json::json!({ "source": "documents" })),
            confidence: support.confidence(options.min_support_documents),
            low_support: support.is_low(options.min_support_documents),
        });
    }
}

#[cfg(test)]
mod tests;
