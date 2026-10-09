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

pub mod export;
pub mod extract;
pub mod follow_up;
pub mod query;
pub mod resolve;
pub mod store;
pub mod tables;
pub mod traverse;
pub mod views;

use std::collections::BTreeMap;
use std::fmt;

use crate::error::{Error, Result as CoreResult};

use duckdb::types::ToSqlOutput;
use serde::{Deserialize, Deserializer, Serialize};

use crate::embedding::{Dimension, Input};
use crate::extraction::Tally;
use crate::ids::{ChunkId, ClassId, DocumentId, EdgeId, NodeId, RelationId};
use crate::ontology::OntologyVersion;
use crate::text::OneLine;

/// Whether a graph write rests on a reviewed ontology, or on one that was
/// auto-accepted and so is provisional until someone reviews it. Binds
/// as the `provisional` column's boolean, and serializes as it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "bool", into = "bool")]
pub enum Standing {
    #[default]
    Reviewed,
    Provisional,
}

flag_enum!(Standing, false => Reviewed, true => Provisional);

/// A stored node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Node {
    pub id: NodeId,
    pub label: String,
    pub class_id: ClassId,
    #[serde(default)]
    pub properties: Properties,
    #[serde(rename = "provisional")]
    pub standing: Standing,
}

/// `label (class)`, how every listing names a node.
impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", OneLine(&self.label), self.class_id)
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
            self.class_id.as_str().replace('_', " ")
        ))
    }
}

/// A stored edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Edge {
    pub id: EdgeId,
    pub source_node_id: NodeId,
    pub target_node_id: NodeId,
    pub relation_id: RelationId,
    pub weight: f64,
    #[serde(default)]
    pub properties: Properties,
    #[serde(rename = "provisional")]
    pub standing: Standing,
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
/// the shape again. Sorted by name, so rendering and the stored form do not
/// depend on the order a model or a row gave them in.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(transparent)]
pub struct Properties(BTreeMap<String, serde_json::Value>);

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

    /// `KEY=VALUE` pairs as properties, as `quack graph add` and `set`
    /// take them: a value that reads as a JSON number or boolean is stored
    /// as one, anything else as text.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] naming a pair without `=` or a key.
    pub fn parse_pairs<S: AsRef<str>>(
        pairs: impl IntoIterator<Item = S>,
    ) -> CoreResult<serde_json::Map<String, serde_json::Value>> {
        let mut out = serde_json::Map::new();
        for pair in pairs {
            let pair = pair.as_ref();
            let Some((key, value)) = pair.split_once('=') else {
                return Err(Error::Config(format!("'{pair}' is not KEY=VALUE")));
            };
            let key = key.trim();
            if key.is_empty() {
                return Err(Error::Config(format!("'{pair}' has no key")));
            }
            let value = serde_json::from_str::<serde_json::Value>(value.trim())
                .ok()
                .filter(|v| v.is_number() || v.is_boolean())
                .unwrap_or_else(|| serde_json::Value::String(value.trim().to_owned()));
            out.insert(key.to_owned(), value);
        }
        Ok(out)
    }

    /// The graph page's `KEY=VALUE` lines as an edit for [`Self::patch`]:
    /// blank lines skipped, and an empty value is `null`, which removes
    /// the key.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] naming a line without `=` or a key.
    pub fn parse_patch_lines(text: &str) -> CoreResult<serde_json::Map<String, serde_json::Value>> {
        let mut patch = Self::parse_pairs(text.lines().map(str::trim).filter(|l| !l.is_empty()))?;
        for value in patch.values_mut() {
            if value.as_str().is_some_and(str::is_empty) {
                *value = serde_json::Value::Null;
            }
        }
        Ok(patch)
    }

    /// Apply a person's edit: each value sets its key, `null` removes it.
    pub fn patch(&mut self, edit: &serde_json::Map<String, serde_json::Value>) {
        for (key, value) in edit {
            if value.is_null() {
                self.0.remove(key);
            } else {
                self.0.insert(key.clone(), value.clone());
            }
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
        serde_json::Value::Object(self.0.clone().into_iter().collect()).to_string()
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
            serde_json::Value::Object(map) => Self::from(map),
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
        Self(map.into_iter().collect())
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
            parts.push(format!("{}: {}", OneLine(key), OneLine(&value)));
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
/// as `document_id` and `chunk_id`, `table_name` and `row_key`, or the
/// person who asserted it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(untagged)]
#[schema(as = ProvenanceOrigin)]
pub enum Origin {
    /// A person stated it (`quack graph add`, the API, the graph page).
    /// First, so a row with `asserted_at` never reads as a chunk.
    Manual {
        /// The server user; `None` from the command line.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        author: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
        asserted_at: String,
    },
    /// A chunk of a document.
    Chunk {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        document_id: Option<DocumentId>,
        chunk_id: ChunkId,
    },
    /// A row of a mapped table, by its key.
    Row { table_name: String, row_key: String },
}

/// The `_quack_provenance` columns of one row, as the store reads them.
#[derive(Debug, Clone)]
pub struct ProvenanceColumns {
    pub document_id: Option<DocumentId>,
    /// Empty when the row is not a chunk's.
    pub chunk_id: ChunkId,
    pub table_name: String,
    pub row_key: String,
    pub author: Option<String>,
    pub note: Option<String>,
    pub asserted_at: Option<String>,
}

impl Default for ProvenanceColumns {
    fn default() -> Self {
        Self {
            document_id: None,
            chunk_id: ChunkId::from(String::new()),
            table_name: String::new(),
            row_key: String::new(),
            author: None,
            note: None,
            asserted_at: None,
        }
    }
}

impl Origin {
    /// From the `_quack_provenance` columns, where an unused text column
    /// is empty: an assertion when it has a time, a row when the table is
    /// named, else a chunk.
    #[must_use]
    pub fn from_columns(columns: ProvenanceColumns) -> Self {
        let ProvenanceColumns {
            document_id,
            chunk_id,
            table_name,
            row_key,
            author,
            note,
            asserted_at,
        } = columns;
        if let Some(asserted_at) = asserted_at {
            return Self::Manual {
                author,
                note,
                asserted_at,
            };
        }
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
            Self::Row { .. } | Self::Manual { .. } => None,
        }
    }

    /// "asserted by {author}: {note}" for a manual origin, as every
    /// rendering shows it; `None` for the others.
    #[must_use]
    pub fn assertion(&self) -> Option<String> {
        let Self::Manual { author, note, .. } = self else {
            return None;
        };
        let mut text = match author {
            Some(author) => format!("asserted by {}", OneLine(author)),
            None => String::from("asserted by hand"),
        };
        if let Some(note) = note.as_deref().filter(|n| !n.trim().is_empty()) {
            text.push_str(": ");
            text.push_str(&OneLine(note).to_string());
        }
        Some(text)
    }
}

/// A node's or edge's source, with how sure the extraction was.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Provenance {
    pub subject_id: String,
    #[serde(flatten)]
    pub origin: Origin,
    pub confidence: f64,
}

/// A traversal or listing: the nodes and edges found, each with its
/// provenance, and the nodes the query started from.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GraphResult {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub provenance: Vec<Provenance>,
    /// Ids of the nodes the query resolved its entry point to.
    #[serde(default)]
    pub roots: Vec<NodeId>,
    /// How many nodes matched before `max_nodes` applied, when the query
    /// could count them (a class listing). `None` for a walk, which stops
    /// at the cap without knowing what it did not visit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_nodes: Option<u64>,
    /// Whether `max_nodes` cut this result short. A reader that does not
    /// check it reads a capped listing as the whole population.
    #[serde(default)]
    pub truncated: bool,
    /// The graph this result came from, so a reader can judge the answer
    /// without asking for the status separately.
    #[serde(default)]
    pub status: GraphStatusSummary,
}

/// What a reader of one graph result needs to know about the whole graph:
/// which ontology built it, whether it lags the current one, how much of
/// it is provisional, how much the corpus expressed that the ontology
/// lacks, and how many provisional nodes this result left out.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GraphStatusSummary {
    pub built_with_version: Option<OntologyVersion>,
    pub ontology_version: Option<OntologyVersion>,
    pub stale: bool,
    pub provisional_nodes: u64,
    /// Distinct classes and relations the corpus expressed that the
    /// ontology lacks ([`Drift::total`]).
    pub drift_total: u64,
    /// Provisional nodes query mode dropped from this result.
    #[serde(default)]
    pub dropped_provisional: u64,
}

impl GraphResult {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Drop provisional nodes and edges (query mode never answers from an
    /// unreviewed graph), counting the nodes dropped in the status.
    #[must_use]
    pub fn without_provisional(mut self) -> Self {
        let before = self.nodes.len();
        self.nodes.retain(|n| n.standing == Standing::Reviewed);
        let dropped = before.saturating_sub(self.nodes.len());
        self.status.dropped_provisional = self
            .status
            .dropped_provisional
            .saturating_add(u64::try_from(dropped).unwrap_or(u64::MAX));
        let kept: std::collections::BTreeSet<&str> =
            self.nodes.iter().map(|n| n.id.as_str()).collect();
        self.edges.retain(|e| {
            e.standing == Standing::Reviewed
                && kept.contains(e.source_node_id.as_str())
                && kept.contains(e.target_node_id.as_str())
        });
        let subjects: std::collections::BTreeSet<String> = self
            .nodes
            .iter()
            .map(|n| n.id.to_string())
            .chain(self.edges.iter().map(|e| e.id.to_string()))
            .collect();
        self.provenance.retain(|p| subjects.contains(&p.subject_id));
        self.roots.retain(|r| kept.contains(r.as_str()));
        self
    }
}

/// Counts of what the corpus tried to express that the ontology has no
/// place for, accumulated across extraction runs (design doc 6.5, drift).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
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

/// The graph's size and its summary: what a turn's prompt and tools
/// need, without [`GraphStatus`]'s per-table and per-chunk checks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphSize {
    pub nodes: u64,
    pub edges: u64,
    pub summary: GraphStatusSummary,
}

impl GraphSize {
    /// Whether the graph exists at all; tools register only then.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.nodes > 0
    }

    #[must_use]
    pub fn provisional(&self) -> bool {
        self.summary.provisional_nodes > 0
    }
}

/// What the interfaces show about the graph: size, whether it is
/// provisional (built from an auto-accepted ontology) or stale (the
/// ontology moved on), and the drift the corpus expressed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
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
    /// Chunks of ready documents no extraction has read yet.
    #[serde(default)]
    pub pending_chunks: u64,
    /// Mapped tables whose rows changed since table extraction last read
    /// them (re-ingested, re-imported, or updated by SQL), or that it
    /// never read.
    #[serde(default)]
    pub pending_tables: Vec<String>,
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
            || String::from("never built; run `quack graph extract`"),
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
        if self.pending_chunks > 0 {
            writeln!(
                f,
                "{} chunks not yet extracted: `quack graph extract --source documents`",
                self.pending_chunks
            )?;
        }
        if !self.pending_tables.is_empty() {
            writeln!(
                f,
                "Mapped tables changed since the graph read them: {} (`quack graph extract --source tables`)",
                self.pending_tables.join(", ")
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
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
        ALTER TABLE _quack_provenance ADD COLUMN IF NOT EXISTS author TEXT;
        ALTER TABLE _quack_provenance ADD COLUMN IF NOT EXISTS note TEXT;
        ALTER TABLE _quack_provenance ADD COLUMN IF NOT EXISTS asserted_at TIMESTAMP;
        CREATE TABLE IF NOT EXISTS _quack_graph_tables_built (
            table_name TEXT PRIMARY KEY,
            document_id TEXT,
            ontology_version INTEGER NOT NULL,
            row_count BIGINT NOT NULL,
            row_hash TEXT NOT NULL,
            built_at TIMESTAMP DEFAULT now()
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
mod tests;
