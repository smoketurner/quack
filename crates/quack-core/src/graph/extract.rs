//! Constrained extraction: each chunk goes to the chat model with the
//! ontology, and must answer with nodes and edges that fit it. Anything
//! outside the ontology is dropped and counted as drift (design doc 6.4).

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

use super::Drift;
use super::store::{self, NewNode, Source};
use crate::error::{Error, Result};
use crate::ingestion::DbHandle;
use crate::ontology::{self, Ontology};
use crate::storage::workspace::WorkspaceDb;

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
    pub properties: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    #[serde(default)]
    pub properties: serde_json::Value,
}

/// Boxed future so implementations can be trait objects.
pub type ExtractFuture<'a> = Pin<Box<dyn Future<Output = Result<Extraction>> + Send + 'a>>;

/// Constrained extraction over one passage; the prompt already carries the
/// ontology.
pub trait GraphExtractor: Send + Sync {
    fn extract<'a>(&'a self, text: &'a str) -> ExtractFuture<'a>;
}

/// The preamble for the chat model, with the ontology rendered into it.
#[must_use]
pub fn prompt_for(ontology: &Ontology) -> String {
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
    prompt.push_str(&ontology.render_for_prompt());
    prompt
}

/// Parse the model's answer leniently: the first `{` to the last `}`.
///
/// # Errors
///
/// Returns an error when no JSON object of the right shape parses.
pub fn parse_extraction(answer: &str) -> Result<Extraction> {
    let start = answer.find('{');
    let end = answer.rfind('}');
    let (Some(start), Some(end)) = (start, end) else {
        return Err(Error::Ontology(String::from(
            "the model returned no JSON object",
        )));
    };
    let slice = answer.get(start..=end).unwrap_or(answer);
    serde_json::from_str(slice)
        .map_err(|e| Error::Ontology(format!("the model's JSON does not parse: {e}")))
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

/// Keep what fits: nodes of known classes, edges of known relations
/// between listed nodes whose classes fit the domain and range. Unknown
/// classes and relations are counted as drift.
#[must_use]
pub fn validate(ontology: &Ontology, extraction: Extraction) -> Validated {
    let mut out = Validated::default();
    let mut class_of: BTreeMap<String, String> = BTreeMap::new();
    for node in extraction.nodes {
        let label = node.label.trim();
        if label.is_empty() {
            continue;
        }
        let class = node.class.trim().to_lowercase();
        if class != ontology::ROOT_CLASS && ontology.class(&class).is_none() {
            out.drift.note_class(&class);
            continue;
        }
        let key = super::normalize_label(label);
        if class_of.contains_key(&key) {
            continue;
        }
        class_of.insert(key, class.clone());
        out.nodes.push(ExtractedNode {
            label: label.to_owned(),
            class,
            properties: if node.properties.is_object() {
                node.properties
            } else {
                serde_json::json!({})
            },
        });
    }
    for edge in extraction.edges {
        let relation = edge.relation.trim().to_lowercase();
        if relation != ontology::MENTIONS_RELATION && ontology.relation(&relation).is_none() {
            out.drift.note_relation(&relation);
            continue;
        }
        let (Some(source_class), Some(target_class)) = (
            class_of.get(&super::normalize_label(&edge.source)),
            class_of.get(&super::normalize_label(&edge.target)),
        ) else {
            out.invalid_edges = out.invalid_edges.saturating_add(1);
            continue;
        };
        if !store::edge_fits(ontology, &relation, source_class, target_class) {
            out.invalid_edges = out.invalid_edges.saturating_add(1);
            continue;
        }
        out.edges.push(ExtractedEdge {
            source: edge.source.trim().to_owned(),
            target: edge.target.trim().to_owned(),
            relation,
            properties: if edge.properties.is_object() {
                edge.properties
            } else {
                serde_json::json!({})
            },
        });
    }
    out
}

/// A chunk to extract from.
#[derive(Debug, Clone)]
pub struct ChunkText {
    pub chunk_id: String,
    pub document_id: String,
    pub text: String,
}

/// Chunks of ready documents not yet extracted under any ontology
/// version, sampled evenly across documents when `limit` is given
/// (issue #60: the first N chunks by ingest order were one document's
/// front matter). A run records each chunk it processed, so the next run
/// sends only what is new; `graph extract --reset` clears the record.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn chunks(db: &WorkspaceDb, limit: Option<u32>) -> Result<Vec<ChunkText>> {
    let mut stmt = db.connection().prepare(
        "SELECT c.id, c.document_id, c.content, c.heading FROM _quack_chunks c \
         JOIN _quack_documents d ON d.id = c.document_id \
         WHERE d.status = 'ready' \
           AND NOT EXISTS (SELECT 1 FROM _quack_graph_extracted x WHERE x.chunk_id = c.id) \
         ORDER BY d.ingested_at, d.id, c.chunk_index",
    )?;
    let mut rows = stmt.query([])?;
    let mut by_document: BTreeMap<String, Vec<ChunkText>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    while let Some(row) = rows.next()? {
        let heading: Option<String> = row.get(3)?;
        let content: String = row.get(2)?;
        let document_id: String = row.get(1)?;
        if !by_document.contains_key(&document_id) {
            order.push(document_id.clone());
        }
        by_document
            .entry(document_id.clone())
            .or_default()
            .push(ChunkText {
                chunk_id: row.get(0)?,
                document_id,
                text: match heading {
                    Some(h) => format!("{h}\n\n{content}"),
                    None => content,
                },
            });
    }
    let Some(limit) = limit else {
        return Ok(order
            .into_iter()
            .filter_map(|id| by_document.remove(&id))
            .flatten()
            .collect());
    };
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    if limit == 0 || order.is_empty() {
        return Ok(Vec::new());
    }
    // Every document gets an equal quota, spaced evenly through it.
    let quota = limit.div_ceil(order.len()).max(1);
    let mut chosen = Vec::with_capacity(limit);
    for id in &order {
        let Some(mut chunks) = by_document.remove(id) else {
            continue;
        };
        let take = quota.min(chunks.len());
        let mut positions: Vec<usize> = (0..take)
            .map(|k| {
                k.saturating_mul(chunks.len())
                    .checked_div(take)
                    .unwrap_or(0)
                    .min(chunks.len().saturating_sub(1))
            })
            .collect();
        positions.dedup();
        // Highest first, so removing by index leaves the lower ones valid.
        for position in positions.into_iter().rev() {
            if position < chunks.len() {
                chosen.push(chunks.remove(position));
            }
        }
    }
    chosen.sort_by(|a, b| {
        order
            .iter()
            .position(|d| d == &a.document_id)
            .cmp(&order.iter().position(|d| d == &b.document_id))
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });
    chosen.truncate(limit);
    Ok(chosen)
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

/// Run constrained extraction over `chunks` and store what fits. A failed
/// chunk is logged and skipped; only every chunk failing is an error.
///
/// # Errors
///
/// Returns an error when every chunk fails or a write fails.
pub async fn run(
    db: &impl DbHandle,
    chunks: Vec<ChunkText>,
    extractor: &dyn GraphExtractor,
    ontology: &Ontology,
    provisional: bool,
) -> Result<RunSummary> {
    let mut summary = RunSummary {
        chunks: u32::try_from(chunks.len()).unwrap_or(u32::MAX),
        ..RunSummary::default()
    };
    for chunk in chunks {
        let extraction = match extractor.extract(&chunk.text).await {
            Ok(extraction) => extraction,
            Err(e) => {
                tracing::warn!(chunk = %chunk.chunk_id, error = %e, "extraction failed; skipping chunk");
                summary.failed_chunks = summary.failed_chunks.saturating_add(1);
                continue;
            }
        };
        tracing::debug!(
            chunk = %chunk.chunk_id,
            nodes = extraction.nodes.len(),
            edges = extraction.edges.len(),
            "graph extraction parsed"
        );
        let validated = validate(ontology, extraction);
        summary.invalid_edges = summary
            .invalid_edges
            .saturating_add(validated.invalid_edges);
        summary.drift.absorb(&validated.drift);
        let (nodes, edges) = db.with(|db| {
            db.under_timeout(|db| {
                let counts = store_validated(
                    db,
                    &validated,
                    &Source::chunk(&chunk.document_id, &chunk.chunk_id, MODEL_CONFIDENCE),
                    provisional,
                )?;
                store::record_extracted(db, &chunk.chunk_id, ontology.version, counts)?;
                Ok(counts)
            })
        })?;
        if nodes == 0 && edges == 0 {
            tracing::debug!(chunk = %chunk.chunk_id, "graph extraction kept nothing from this chunk");
        }
        summary.nodes = summary.nodes.saturating_add(nodes);
        summary.edges = summary.edges.saturating_add(edges);
    }
    if summary.chunks > 0 && summary.failed_chunks == summary.chunks {
        return Err(Error::Llm(String::from(
            "every chunk failed extraction; check the model and provider",
        )));
    }
    db.with(|db| store::record_drift(db, &summary.drift, false))?;
    Ok(summary)
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
    let mut ids: BTreeMap<String, String> = BTreeMap::new();
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
        ids.insert(super::normalize_label(&node.label), id);
        nodes = nodes.saturating_add(1);
    }
    let mut edges = 0u32;
    for edge in &validated.edges {
        let (Some(s), Some(t)) = (
            ids.get(&super::normalize_label(&edge.source)),
            ids.get(&super::normalize_label(&edge.target)),
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
        let extraction = parse_extraction(
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
        let v = validate(&ontology(), extraction);
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
        assert_eq!(v.drift.classes.get("vessel"), Some(&1));
        assert_eq!(v.drift.relations.get("docked_at"), Some(&1));
        assert!(v.edges.iter().all(|e| e.properties.is_object()));
    }

    #[test]
    fn parse_rejects_prose_and_the_wrong_shape() {
        assert!(parse_extraction("no json here").is_err());
        assert!(parse_extraction(r#"{"nodes": "nope"}"#).is_err());
        assert_eq!(parse_extraction("{}").map_or(9, |e| e.nodes.len()), 0);
    }

    #[test]
    fn prompt_carries_the_ontology() {
        let prompt = prompt_for(&ontology());
        assert!(prompt.contains("ships_to: organization -> country"));
        assert!(prompt.starts_with("Extract the entities"));
    }
}
