//! The knowledge graph (design doc 6.4): nodes typed by the ontology's
//! classes, edges typed by its relations, and a provenance row for every
//! node and edge back to the chunk or table row it came from.
//!
//! Extraction is either deterministic from mapped tables (`tables`) or
//! constrained extraction from chunks with the chat model (`extract`);
//! resolution merges on `(normalized_label, class)` and proposes fuzzy
//! merges for review (`resolve`); traversal is plain SQL (`traverse`).
//! Everything lives in `_quack_graph_*` and `_quack_provenance` inside the
//! workspace file.

pub mod extract;
pub mod resolve;
pub mod store;
pub mod tables;
pub mod traverse;

use serde::{Deserialize, Serialize};

use crate::embedding::{Dimension, Input};

/// A stored node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub label: String,
    pub class_id: String,
    #[serde(default)]
    pub properties: serde_json::Value,
    pub provisional: bool,
}

impl Node {
    /// What the node's label vector is made from: its label and class,
    /// compared with other labels and with names looked up among them.
    #[must_use]
    pub fn embedding_input(&self) -> Input {
        Input::Similarity(format!(
            "{} ({})",
            self.label,
            self.class_id.replace('_', " ")
        ))
    }
}

/// A stored edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub id: String,
    pub source_node_id: String,
    pub target_node_id: String,
    pub relation_id: String,
    pub weight: f64,
    #[serde(default)]
    pub properties: serde_json::Value,
    pub provisional: bool,
}

/// Where a node or edge came from: a chunk of a document, or a row of a
/// mapped table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub subject_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_key: Option<String>,
    pub confidence: f64,
}

/// A traversal or listing: the nodes and edges found, each with its
/// provenance, and the nodes the query started from.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphResult {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub provenance: Vec<Provenance>,
    /// Ids of the nodes the query resolved its entry point to.
    #[serde(default)]
    pub roots: Vec<String>,
    /// How many nodes matched before `max_nodes` applied, when the query
    /// could count them (a class listing). `None` for a walk, which stops
    /// at the cap without knowing what it did not visit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_nodes: Option<u64>,
    /// Whether `max_nodes` cut this result short. A reader that does not
    /// check it reads a capped listing as the whole population.
    #[serde(default)]
    pub truncated: bool,
}

impl GraphResult {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Drop provisional nodes and edges (query mode never answers from an
    /// unreviewed graph).
    #[must_use]
    pub fn without_provisional(mut self) -> Self {
        self.nodes.retain(|n| !n.provisional);
        let kept: std::collections::BTreeSet<&str> =
            self.nodes.iter().map(|n| n.id.as_str()).collect();
        self.edges.retain(|e| {
            !e.provisional
                && kept.contains(e.source_node_id.as_str())
                && kept.contains(e.target_node_id.as_str())
        });
        let subjects: std::collections::BTreeSet<String> = self
            .nodes
            .iter()
            .map(|n| n.id.clone())
            .chain(self.edges.iter().map(|e| e.id.clone()))
            .collect();
        self.provenance.retain(|p| subjects.contains(&p.subject_id));
        self.roots.retain(|r| kept.contains(r.as_str()));
        self
    }
}

/// Counts of what the corpus tried to express that the ontology has no
/// place for, accumulated across extraction runs (design doc 6.5, drift).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Drift {
    #[serde(default)]
    pub classes: std::collections::BTreeMap<String, u32>,
    #[serde(default)]
    pub relations: std::collections::BTreeMap<String, u32>,
}

impl Drift {
    /// Distinct things the ontology lacks.
    #[must_use]
    pub fn total(&self) -> usize {
        self.classes.len().saturating_add(self.relations.len())
    }

    pub fn note_class(&mut self, class: &str) {
        let entry = self.classes.entry(class.to_owned()).or_default();
        *entry = entry.saturating_add(1);
    }

    pub fn note_relation(&mut self, relation: &str) {
        let entry = self.relations.entry(relation.to_owned()).or_default();
        *entry = entry.saturating_add(1);
    }

    pub fn absorb(&mut self, other: &Self) {
        for (k, v) in &other.classes {
            let entry = self.classes.entry(k.clone()).or_default();
            *entry = entry.saturating_add(*v);
        }
        for (k, v) in &other.relations {
            let entry = self.relations.entry(k.clone()).or_default();
            *entry = entry.saturating_add(*v);
        }
    }
}

/// What the interfaces show about the graph: size, whether it is
/// provisional (built from an auto-accepted ontology) or stale (the
/// ontology moved on), and the drift the corpus expressed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphStatus {
    pub nodes: u64,
    pub edges: u64,
    pub provisional_nodes: u64,
    /// The ontology version the graph was last built or revalidated with;
    /// `0` when never built.
    pub built_with_version: u32,
    pub ontology_version: u32,
    pub stale: bool,
    pub pending_merges: u64,
    pub drift: Drift,
    /// Tables the ontology maps that are no longer in the workspace
    /// (their document was deleted): extraction skips them until the
    /// mapping is removed or the data comes back.
    #[serde(default)]
    pub missing_tables: Vec<String>,
}

impl GraphStatus {
    /// Whether the graph exists at all; tools register only then.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.nodes > 0
    }

    #[must_use]
    pub fn provisional(&self) -> bool {
        self.provisional_nodes > 0
    }
}

/// The `label` the tables store: lowercased, whitespace collapsed.
#[must_use]
pub fn normalize_label(label: &str) -> String {
    label
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// What a graph extraction reads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExtractSource {
    /// Mapped tables, then document chunks.
    #[default]
    All,
    /// Only the mapped tables: no model calls.
    Tables,
    /// Only the document chunks.
    Documents,
}

text_enum!(ExtractSource, "extract source", {
    All => "all",
    Tables => "tables",
    Documents => "documents",
});

impl ExtractSource {
    #[must_use]
    pub fn includes_tables(self) -> bool {
        match self {
            Self::All | Self::Tables => true,
            Self::Documents => false,
        }
    }

    #[must_use]
    pub fn includes_documents(self) -> bool {
        match self {
            Self::All | Self::Documents => true,
            Self::Tables => false,
        }
    }
}

/// Tuning for traversal and resolution, from `[graph]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphOptions {
    pub max_traversal_depth: u32,
    pub max_nodes: u32,
    /// Cosine distance under which two labels of one class are proposed
    /// as a merge.
    pub merge_threshold: f64,
    /// Cosine distance under which the merge happens without review.
    pub auto_merge_threshold: f64,
}

impl Default for GraphOptions {
    fn default() -> Self {
        Self {
            max_traversal_depth: 3,
            max_nodes: 200,
            merge_threshold: 0.08,
            auto_merge_threshold: 0.02,
        }
    }
}

/// The graph tables, created with the workspace's embedding dimension.
#[must_use]
pub fn ddl(dimension: Dimension) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS _quack_graph_nodes (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            normalized_label TEXT NOT NULL,
            class_id TEXT NOT NULL,
            properties JSON,
            embedding FLOAT[{dimension}],
            provisional BOOLEAN NOT NULL DEFAULT false,
            UNIQUE (normalized_label, class_id)
        );
        ALTER TABLE _quack_graph_nodes ADD COLUMN IF NOT EXISTS embedding_profile TEXT;
        CREATE TABLE IF NOT EXISTS _quack_graph_edges (
            id TEXT PRIMARY KEY,
            source_node_id TEXT NOT NULL,
            target_node_id TEXT NOT NULL,
            relation_id TEXT NOT NULL,
            weight DOUBLE DEFAULT 1.0,
            properties JSON,
            provisional BOOLEAN NOT NULL DEFAULT false
        );
        CREATE INDEX IF NOT EXISTS _quack_graph_edges_source_idx ON _quack_graph_edges (source_node_id);
        CREATE INDEX IF NOT EXISTS _quack_graph_edges_target_idx ON _quack_graph_edges (target_node_id);
        CREATE TABLE IF NOT EXISTS _quack_provenance (
            subject_id TEXT NOT NULL,
            document_id TEXT,
            chunk_id TEXT NOT NULL DEFAULT '',
            table_name TEXT NOT NULL DEFAULT '',
            row_key TEXT NOT NULL DEFAULT '',
            confidence DOUBLE,
            PRIMARY KEY (subject_id, chunk_id, table_name, row_key)
        );
        CREATE TABLE IF NOT EXISTS _quack_graph_extracted (
            chunk_id TEXT PRIMARY KEY,
            ontology_version INTEGER NOT NULL,
            nodes INTEGER NOT NULL,
            edges INTEGER NOT NULL,
            extracted_at TIMESTAMP DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS _quack_graph_merges (
            id TEXT PRIMARY KEY,
            keep_node_id TEXT NOT NULL,
            drop_node_id TEXT NOT NULL,
            distance DOUBLE NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            decided_by TEXT,
            decided_at TIMESTAMP,
            UNIQUE (keep_node_id, drop_node_id)
        );"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_normalize_case_and_whitespace() {
        assert_eq!(normalize_label("  Acme   Corp \n"), "acme corp");
        assert_eq!(normalize_label("USAID"), "usaid");
    }

    #[test]
    fn provisional_results_are_dropped_with_their_edges_and_provenance() {
        let node = |id: &str, provisional: bool| Node {
            id: id.to_owned(),
            label: id.to_owned(),
            class_id: String::from("entity"),
            properties: serde_json::Value::Null,
            provisional,
        };
        let result = GraphResult {
            nodes: vec![node("a", false), node("b", true), node("c", false)],
            edges: vec![
                Edge {
                    id: String::from("ab"),
                    source_node_id: String::from("a"),
                    target_node_id: String::from("b"),
                    relation_id: String::from("mentions"),
                    weight: 1.0,
                    properties: serde_json::Value::Null,
                    provisional: false,
                },
                Edge {
                    id: String::from("ac"),
                    source_node_id: String::from("a"),
                    target_node_id: String::from("c"),
                    relation_id: String::from("mentions"),
                    weight: 1.0,
                    properties: serde_json::Value::Null,
                    provisional: false,
                },
            ],
            provenance: vec![
                Provenance {
                    subject_id: String::from("b"),
                    document_id: None,
                    chunk_id: None,
                    table_name: None,
                    row_key: None,
                    confidence: 1.0,
                },
                Provenance {
                    subject_id: String::from("ac"),
                    document_id: None,
                    chunk_id: None,
                    table_name: None,
                    row_key: None,
                    confidence: 1.0,
                },
            ],
            roots: vec![String::from("a"), String::from("b")],
            ..GraphResult::default()
        };
        let kept = result.without_provisional();
        assert_eq!(kept.nodes.len(), 2);
        assert_eq!(
            kept.edges.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
            ["ac"]
        );
        assert_eq!(kept.provenance.len(), 1);
        assert_eq!(kept.roots, ["a"]);
    }

    #[test]
    fn drift_accumulates_and_counts_distinct_names() {
        let mut drift = Drift::default();
        drift.note_class("vessel");
        drift.note_class("vessel");
        drift.note_relation("docked_at");
        let mut total = Drift::default();
        total.absorb(&drift);
        total.absorb(&drift);
        assert_eq!(total.classes.get("vessel"), Some(&4));
        assert_eq!(total.total(), 2);
    }
}
