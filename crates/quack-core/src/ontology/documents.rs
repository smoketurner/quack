//! Ontology induction from document evidence (design doc 6.5): open
//! extraction on a stratified sample of chunks, vocabulary normalization,
//! structure inference, and support scoring. Model calls go through
//! [`Extract`], so the pipeline is tested with a canned one.

use std::collections::{BTreeMap, BTreeSet};

use futures::StreamExt as _;
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

/// What open extraction returns for one chunk.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenExtraction {
    #[serde(default)]
    pub entities: Vec<OpenEntity>,
    #[serde(default)]
    pub relations: Vec<OpenRelation>,
    #[serde(default)]
    pub attributes: Vec<OpenAttribute>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenEntity {
    pub name: String,
    #[serde(rename = "type")]
    pub type_name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenRelation {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAttribute {
    pub entity: String,
    pub name: String,
    pub value: String,
}

/// The extraction prompt the model answers with JSON.
pub const EXTRACTION_PROMPT: &str = "Read the passage and list what it mentions. Return only JSON with this shape and \
nothing else:\n\
{\"entities\": [{\"name\": \"...\", \"type\": \"...\"}], \
\"relations\": [{\"subject\": \"...\", \"relation\": \"...\", \"object\": \"...\"}], \
\"attributes\": [{\"entity\": \"...\", \"name\": \"...\", \"value\": \"...\"}]}\n\
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
) -> Result<(Vec<Candidate>, RunSummary)> {
    let sampled = u32::try_from(sample.len()).unwrap_or(u32::MAX);
    let (observations, failed) = observe(&sample, extraction).await?;
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
    let lookup = |a: &str, b: &str| table.as_ref().is_some_and(|t| t.same(a, b));
    let similarity: Similarity<'_> = if table.is_some() { Some(&lookup) } else { None };
    let candidates = propose(&observations, current, options, similarity);
    let low = u32::try_from(candidates.iter().filter(|c| c.low_support).count()).unwrap_or(0);
    let summary = RunSummary {
        sampled_chunks: sampled,
        failed_chunks: failed,
        candidates: u32::try_from(candidates.len()).unwrap_or(u32::MAX),
        low_support: low,
    };
    Ok((candidates, summary))
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
) -> Result<(Vec<Observation>, u32)> {
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
    Ok((observations, run.failed()))
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
mod tests {
    use super::*;
    use crate::extraction::{Extract, ExtractFuture, parse_answer};
    use crate::progress::ChunkDone;
    use crate::progress::RunControl;
    use crate::storage::workspace::{NewChunk, NewDocument};

    struct Canned;

    impl Extract<OpenExtraction> for Canned {
        fn extract<'a>(&'a self, text: &'a str) -> ExtractFuture<'a, OpenExtraction> {
            Box::pin(async move {
                if text.contains("FAIL") {
                    return Err(Error::Ontology(String::from("boom")));
                }
                parse_answer::<OpenExtraction>(&format!(
                    r#"Sure! {{"entities": [{{"name": "Orgenics", "type": "vendor"}}, {{"name": "Orgenics", "type": "organization"}}, {{"name": "USAID", "type": "organization"}}, {{"name": "Kenya", "type": "country"}}, {{"name": "{}", "type": "Shipment"}}],
"relations": [{{"subject": "Orgenics", "relation": "ships to", "object": "Kenya"}}],
"attributes": [{{"entity": "Orgenics", "name": "founded", "value": "1983"}}, {{"entity": "Kenya", "name": "region", "value": "East Africa"}}]}}"#,
                    text.split_whitespace().next().unwrap_or("x")
                ))
            })
        }
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn workspace_with_docs() -> WorkspaceDb {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
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
                    db.insert_chunk(&NewChunk {
                        id: &ChunkId::from(id.as_str()),
                        document_id: &DocumentId::from(doc.as_str()),
                        chunk_index: i,
                        content: &content,
                        heading: None,
                        page: None,
                        embedding: None
                    })
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
            db.insert_chunk(&NewChunk {
                id: &ChunkId::from("p-0"),
                document_id: &DocumentId::from("pending"),
                chunk_index: 0,
                content: "not ready but long enough to pass the length filter here",
                heading: None,
                page: None,
                embedding: None
            })
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
        let (observations, failures) = observe(
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
        let (observations, failures) = observe(
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
        let near = |a: &str, b: &str| {
            (a == "vendor" && b == "supplier") || (a == "supplier" && b == "vendor")
        };
        let clustered = Vocabulary::build(&counts, Some(&near));
        assert_eq!(clustered.id("supplier"), "vendor", "the frequent name wins");
        assert_eq!(
            parse_answer::<OpenExtraction>("junk")
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default(),
            "ontology error: the model returned no JSON object"
        );
        assert!(
            parse_answer::<OpenExtraction>("{\"entities\": [{\"name\": \"x\"}]}").is_err(),
            "type is required"
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
}
