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
pub mod query;
pub mod resolve;
pub mod store;
pub mod tables;
pub mod traverse;

use std::fmt;

use duckdb::types::ToSqlOutput;
use serde::{Deserialize, Deserializer, Serialize};

use crate::embedding::{Dimension, Input};
use crate::extraction::Tally;
use crate::ids::{ChunkId, DocumentId};
use crate::ontology::OntologyVersion;

/// A stored node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub label: String,
    pub class_id: String,
    #[serde(default)]
    pub properties: Properties,
    pub provisional: bool,
}

/// `label (class)`, how every listing names a node.
impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.label, self.class_id)
    }
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
    pub properties: Properties,
    pub provisional: bool,
}

/// Properties past this many are counted rather than rendered: a node
/// built from a wide mapped table carries one per column, and the tree has
/// to stay readable inside a system prompt.
const RENDERED_PROPERTIES: usize = 8;

/// A rendered property value longer than this is cut, with an ellipsis.
const PROPERTY_VALUE_CHARS: usize = 60;

/// The property holding a node's other names.
const ALIASES: &str = "aliases";

/// A node's or edge's properties: always an object. Whatever else a model
/// answered or a column held reads as no properties, so no caller checks
/// the shape again.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Properties(serde_json::Map<String, serde_json::Value>);

impl Properties {
    /// From a stored JSON column: `NULL` or text that does not parse is
    /// no properties.
    #[must_use]
    pub fn from_column(text: Option<&str>) -> Self {
        text.and_then(|t| serde_json::from_str::<serde_json::Value>(t).ok())
            .map(Self::from)
            .unwrap_or_default()
    }

    #[must_use]
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.0.get(key)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &serde_json::Value)> {
        self.0.iter()
    }

    /// Add every property of `other` this one lacks; values already here
    /// win.
    pub fn fill_from(&mut self, other: &Self) {
        for (key, value) in &other.0 {
            self.0.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }

    /// The other names the node is known by, merged in from nodes that
    /// resolution folded into it; entry-point lookup matches them too.
    #[must_use]
    pub fn aliases(&self) -> Vec<String> {
        self.0
            .get(ALIASES)
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn set_aliases(&mut self, aliases: &[String]) {
        self.0
            .insert(String::from(ALIASES), serde_json::json!(aliases));
    }

    /// The stored form, for the JSON column.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::Value::Object(self.0.clone()).to_string()
    }

    /// One value, flattened and cut: arrays join their elements, strings
    /// lose their quotes, and null and empty strings render as nothing so
    /// an unset property is left out rather than shown as noise.
    fn value_text(value: &serde_json::Value) -> String {
        let text = match value {
            serde_json::Value::Null => String::new(),
            serde_json::Value::String(text) => text.trim().to_owned(),
            serde_json::Value::Array(items) => {
                let rendered: Vec<String> = items
                    .iter()
                    .map(Self::value_text)
                    .filter(|item| !item.is_empty())
                    .collect();
                rendered.join(", ")
            }
            other => other.to_string(),
        };
        if text.chars().count() > PROPERTY_VALUE_CHARS {
            let mut cut: String = text.chars().take(PROPERTY_VALUE_CHARS).collect();
            cut.push('\u{2026}');
            return cut;
        }
        text
    }
}

impl From<serde_json::Value> for Properties {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::Object(map) => Self(map),
            serde_json::Value::Null
            | serde_json::Value::Bool(_)
            | serde_json::Value::Number(_)
            | serde_json::Value::String(_)
            | serde_json::Value::Array(_) => Self::default(),
        }
    }
}

impl From<serde_json::Map<String, serde_json::Value>> for Properties {
    fn from(map: serde_json::Map<String, serde_json::Value>) -> Self {
        Self(map)
    }
}

/// Any JSON value: a model that answers `"properties": null` or a string
/// still yields an extraction, with no properties.
impl<'de> Deserialize<'de> for Properties {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        serde_json::Value::deserialize(deserializer).map(Self::from)
    }
}

/// `{key: value, key: value}`, bounded, or nothing when no property has a
/// value. The ontology types them (6.3) and the extractors fill them in;
/// without this the model only ever sees labels and classes.
impl fmt::Display for Properties {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = Vec::new();
        let mut dropped: usize = 0;
        for (key, value) in &self.0 {
            let value = Self::value_text(value);
            if value.is_empty() {
                continue;
            }
            if parts.len() >= RENDERED_PROPERTIES {
                dropped = dropped.saturating_add(1);
                continue;
            }
            parts.push(format!("{key}: {value}"));
        }
        if parts.is_empty() {
            return Ok(());
        }
        if dropped > 0 {
            parts.push(format!("... {dropped} more"));
        }
        write!(f, "{{{}}}", parts.join(", "))
    }
}

/// Where a node or edge came from. Serialized flat into [`Provenance`],
/// as `document_id` and `chunk_id` or `table_name` and `row_key`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Origin {
    /// A chunk of a document.
    Chunk {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        document_id: Option<DocumentId>,
        chunk_id: ChunkId,
    },
    /// A row of a mapped table, by its key.
    Row { table_name: String, row_key: String },
}

impl Origin {
    /// From the `_quack_provenance` columns, where the unused pair is
    /// stored as empty text: a row when the table is named, else a chunk.
    #[must_use]
    pub fn from_columns(
        document_id: Option<DocumentId>,
        chunk_id: ChunkId,
        table_name: String,
        row_key: String,
    ) -> Self {
        if table_name.is_empty() {
            Self::Chunk {
                document_id,
                chunk_id,
            }
        } else {
            Self::Row {
                table_name,
                row_key,
            }
        }
    }

    /// The chunk, when this is one.
    #[must_use]
    pub fn chunk_id(&self) -> Option<&ChunkId> {
        match self {
            Self::Chunk { chunk_id, .. } => Some(chunk_id),
            Self::Row { .. } => None,
        }
    }
}

/// A node's or edge's source, with how sure the extraction was.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub subject_id: String,
    #[serde(flatten)]
    pub origin: Origin,
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
    pub classes: Tally,
    #[serde(default)]
    pub relations: Tally,
}

impl Drift {
    /// Distinct things the ontology lacks.
    #[must_use]
    pub fn total(&self) -> usize {
        self.classes.len().saturating_add(self.relations.len())
    }

    pub fn absorb(&mut self, other: &Self) {
        self.classes.absorb(&other.classes);
        self.relations.absorb(&other.relations);
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
    /// `None` when never built.
    pub built_with_version: Option<OntologyVersion>,
    /// The newest saved ontology version; `None` when none was saved.
    pub ontology_version: Option<OntologyVersion>,
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

/// The status as `quack graph status` prints it, one line per finding.
impl fmt::Display for GraphStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ontology_version = self
            .ontology_version
            .map_or_else(|| String::from("none"), |v| v.to_string());
        let built_with = self.built_with_version.map_or_else(
            || String::from("never built"),
            |v| format!("built with {v}"),
        );
        writeln!(
            f,
            "Graph: {} nodes, {} edges (ontology version {ontology_version}, {built_with}){}{}",
            self.nodes,
            self.edges,
            if self.stale {
                "; stale: run `quack graph revalidate` or `quack graph extract`"
            } else {
                ""
            },
            if self.provisional() {
                "; provisional: built from an unreviewed ontology, `quack graph review` clears it"
            } else {
                ""
            }
        )?;
        if self.pending_merges > 0 {
            writeln!(
                f,
                "{} merge proposals pending: `quack graph merges`",
                self.pending_merges
            )?;
        }
        if !self.missing_tables.is_empty() {
            writeln!(
                f,
                "Mapped tables no longer in the workspace (extraction skips them): {}",
                self.missing_tables.join(", ")
            )?;
        }
        if self.drift.total() > 0 {
            let mut items: Vec<String> = self
                .drift
                .classes
                .iter()
                .map(|(k, v)| format!("class {k} ({v})"))
                .chain(
                    self.drift
                        .relations
                        .iter()
                        .map(|(k, v)| format!("relation {k} ({v})")),
                )
                .collect();
            items.sort();
            writeln!(
                f,
                "The corpus expressed {} things the ontology lacks: {}",
                self.drift.total(),
                items.join(", ")
            )?;
        }
        Ok(())
    }
}

/// A label as the graph compares it: lowercased, whitespace collapsed.
/// Nodes merge on it with their class, and it is the tables'
/// `normalized_label`.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NormalizedLabel(String);

impl NormalizedLabel {
    #[must_use]
    pub fn new(label: &str) -> Self {
        Self(
            label
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase(),
        )
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for NormalizedLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl duckdb::ToSql for NormalizedLabel {
    fn to_sql(&self) -> duckdb::Result<ToSqlOutput<'_>> {
        self.0.to_sql()
    }
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
        assert_eq!(
            NormalizedLabel::new("  Acme   Corp \n").as_str(),
            "acme corp"
        );
        assert_eq!(NormalizedLabel::new("USAID").as_str(), "usaid");
        assert!(NormalizedLabel::new(" \t ").is_empty());
    }

    #[test]
    fn properties_render_sorted_and_flattened() {
        let text = Properties::from(serde_json::json!({
            "status": "open",
            "amount": 1200.5,
            "aliases": ["Acme Ltd", "Acme"],
            "closed": null,
            "note": "  padded  ",
        }))
        .to_string();
        assert_eq!(
            text,
            "{aliases: Acme Ltd, Acme, amount: 1200.5, note: padded, status: open}"
        );
    }

    #[test]
    fn properties_without_values_render_as_nothing() {
        for value in [
            serde_json::json!({}),
            serde_json::json!(null),
            serde_json::json!("not an object"),
            serde_json::json!({ "empty": "", "unset": null }),
        ] {
            assert_eq!(Properties::from(value).to_string(), "");
        }
    }

    #[test]
    fn a_long_value_is_cut_and_extra_properties_are_counted() {
        let long = "x".repeat(PROPERTY_VALUE_CHARS + 10);
        let text = Properties::from(serde_json::json!({ "note": long })).to_string();
        assert!(text.ends_with("\u{2026}}"), "{text}");
        // "{note: " + the cut value + the ellipsis + "}"
        assert_eq!(text.chars().count(), PROPERTY_VALUE_CHARS + 9);

        let mut wide = serde_json::Map::new();
        for i in 0..(RENDERED_PROPERTIES + 3) {
            wide.insert(format!("p{i:02}"), serde_json::json!("v"));
        }
        let text = Properties::from(wide).to_string();
        assert!(text.contains("p00: v") && text.contains("p07: v"), "{text}");
        assert!(!text.contains("p08"), "{text}");
        assert!(text.ends_with("... 3 more}"), "{text}");
    }

    #[test]
    fn properties_read_anything_and_keep_only_objects() {
        let parsed: Properties = serde_json::from_str("\"a string\"").unwrap_or_default();
        assert!(parsed.is_empty());
        let parsed: Properties = serde_json::from_str(r#"{"a": 1}"#).unwrap_or_default();
        assert_eq!(parsed.get("a"), Some(&serde_json::json!(1)));
        assert!(Properties::from_column(None).is_empty());
        assert!(Properties::from_column(Some("not json")).is_empty());
        assert_eq!(Properties::from_column(Some(r#"{"a": 1}"#)), parsed);
        assert_eq!(parsed.to_json(), r#"{"a":1}"#);
    }

    #[test]
    fn filling_keeps_existing_values() {
        let mut kept = Properties::from(serde_json::json!({ "a": 1 }));
        kept.fill_from(&Properties::from(serde_json::json!({ "a": 2, "b": 3 })));
        assert_eq!(
            kept,
            Properties::from(serde_json::json!({ "a": 1, "b": 3 }))
        );
    }

    #[test]
    fn origins_serialize_flat_and_round_trip() {
        let chunk = Provenance {
            subject_id: String::from("n"),
            origin: Origin::from_columns(
                Some(DocumentId::from("d")),
                ChunkId::from("c"),
                String::new(),
                String::new(),
            ),
            confidence: 0.5,
        };
        let row = Provenance {
            subject_id: String::from("n"),
            origin: Origin::from_columns(
                None,
                ChunkId::from(String::new()),
                String::from("t"),
                String::from("k"),
            ),
            confidence: 1.0,
        };
        assert_eq!(chunk.origin.chunk_id(), Some(&ChunkId::from("c")));
        assert_eq!(row.origin.chunk_id(), None);
        assert_eq!(
            serde_json::to_value(&chunk).ok(),
            Some(serde_json::json!({
                "subject_id": "n", "document_id": "d", "chunk_id": "c", "confidence": 0.5
            }))
        );
        assert_eq!(
            serde_json::to_value(&row).ok(),
            Some(serde_json::json!({
                "subject_id": "n", "table_name": "t", "row_key": "k", "confidence": 1.0
            }))
        );
        for p in [chunk, row] {
            let back: Option<Provenance> = serde_json::to_string(&p)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok());
            assert_eq!(back, Some(p));
        }
    }

    #[test]
    fn provisional_results_are_dropped_with_their_edges_and_provenance() {
        let node = |id: &str, provisional: bool| Node {
            id: id.to_owned(),
            label: id.to_owned(),
            class_id: String::from("entity"),
            properties: Properties::default(),
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
                    properties: Properties::default(),
                    provisional: false,
                },
                Edge {
                    id: String::from("ac"),
                    source_node_id: String::from("a"),
                    target_node_id: String::from("c"),
                    relation_id: String::from("mentions"),
                    weight: 1.0,
                    properties: Properties::default(),
                    provisional: false,
                },
            ],
            provenance: vec![
                Provenance {
                    subject_id: String::from("b"),
                    origin: Origin::from_columns(
                        None,
                        ChunkId::from(String::new()),
                        String::new(),
                        String::new(),
                    ),
                    confidence: 1.0,
                },
                Provenance {
                    subject_id: String::from("ac"),
                    origin: Origin::from_columns(
                        None,
                        ChunkId::from(String::new()),
                        String::new(),
                        String::new(),
                    ),
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
        drift.classes.bump("vessel");
        drift.classes.bump("vessel");
        drift.relations.bump("docked_at");
        let mut total = Drift::default();
        total.absorb(&drift);
        total.absorb(&drift);
        assert_eq!(total.classes.get("vessel"), 4);
        assert_eq!(total.total(), 2);
    }
}
