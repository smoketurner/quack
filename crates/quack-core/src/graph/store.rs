//! Persistence for nodes, edges, provenance, and the graph's bookkeeping in
//! `_quack_meta`: the ontology version it was built with, and drift.

use std::collections::BTreeMap;
use std::fmt;

use duckdb::OptionalExt as _;
use duckdb::types::ToSqlOutput;

use super::resolve::MergeStatus;
use super::{
    Drift, Edge, GraphStatus, Node, NormalizedLabel, Origin, Properties, Provenance,
    ProvenanceColumns, Standing, tables,
};
use crate::embedding::Vector;
use crate::error::{Error, Result};
use crate::ids::{ChunkId, ClassId, DocumentId, EdgeId, NodeId, RelationId};
use crate::ontology::{IdRenames, Ontology, OntologyVersion, store as ontology_store};
use crate::storage::control::ResourceKind;
use crate::storage::workspace::{MetaKey, SamplePool, WorkspaceDb};

/// A node to store: merged into an existing one with the same normalized
/// label and class, else inserted.
#[derive(Debug, Clone)]
pub struct NewNode {
    pub label: String,
    pub class_id: ClassId,
    pub properties: Properties,
    pub standing: Standing,
}

/// A source to attach to a node or edge.
#[derive(Debug, Clone)]
pub struct Source {
    pub origin: Origin,
    pub confidence: f64,
}

impl Source {
    #[must_use]
    pub fn chunk(document_id: &DocumentId, chunk_id: &ChunkId, confidence: f64) -> Self {
        Self {
            origin: Origin::Chunk {
                document_id: Some(document_id.clone()),
                chunk_id: chunk_id.clone(),
            },
            confidence,
        }
    }

    #[must_use]
    pub fn row(table_name: &str, row_key: &str) -> Self {
        Self {
            origin: Origin::Row {
                table_name: table_name.to_owned(),
                row_key: row_key.to_owned(),
            },
            confidence: 1.0,
        }
    }
}

/// A person's statement that a node or edge holds: who, and why. Written
/// as one manual provenance row per subject, replacing an earlier one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Assertion {
    /// The server user; `None` from the command line.
    pub author: Option<String>,
    pub note: Option<String>,
}

impl Assertion {
    /// The row to write: `asserted_at` is the database's `now()`.
    fn write(&self, db: &WorkspaceDb, subject_id: &str) -> Result<()> {
        let note = self
            .note
            .as_deref()
            .map(str::trim)
            .filter(|n| !n.is_empty());
        db.connection().execute(
            "INSERT OR REPLACE INTO _quack_provenance              (subject_id, document_id, chunk_id, table_name, row_key, confidence, author, note, asserted_at)              VALUES (?, NULL, '', '', '', 1.0, ?, ?, now())",
            duckdb::params![subject_id, self.author.as_deref(), note],
        )?;
        Ok(())
    }
}

/// Insert or merge a node and return its id. Merging keeps the existing
/// label, adds properties the existing node lacks, and clears the
/// provisional flag only when the new evidence is not provisional either.
///
/// # Errors
///
/// Returns an error if a write fails.
pub fn upsert_node(db: &WorkspaceDb, node: &NewNode) -> Result<NodeId> {
    let normalized = NormalizedLabel::new(&node.label);
    if normalized.is_empty() {
        return Err(Error::Analysis(String::from("a node needs a label")));
    }
    let conn = db.connection();
    // `optional`, not `.ok()`: a broken query must fail, not read as "not
    // found" and then insert a duplicate.
    let existing: Option<(NodeId, Option<String>, bool)> = conn
        .query_row(
            "SELECT id, CAST(properties AS VARCHAR), provisional FROM _quack_graph_nodes \
             WHERE normalized_label = ? AND class_id = ?",
            duckdb::params![normalized, node.class_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    if let Some((id, properties, provisional)) = existing {
        let mut merged = Properties::from_column(properties.as_deref());
        merged.fill_from(&node.properties);
        conn.execute(
            "UPDATE _quack_graph_nodes SET properties = ?, provisional = ? WHERE id = ?",
            duckdb::params![
                merged.to_json(),
                provisional && node.standing == Standing::Provisional,
                id
            ],
        )?;
        return Ok(id);
    }
    let id = NodeId::generate();
    conn.execute(
        "INSERT INTO _quack_graph_nodes (id, label, normalized_label, class_id, properties, provisional) \
         VALUES (?, ?, ?, ?, ?, ?)",
        duckdb::params![
            id,
            node.label.trim(),
            normalized,
            node.class_id,
            node.properties.to_json(),
            node.standing
        ],
    )?;
    Ok(id)
}

/// Insert an edge unless the same triple exists; returns the edge id.
///
/// # Errors
///
/// Returns an error if a write fails.
pub fn upsert_edge(
    db: &WorkspaceDb,
    source: &NodeId,
    target: &NodeId,
    relation_id: &str,
    properties: &Properties,
    standing: Standing,
) -> Result<EdgeId> {
    let conn = db.connection();
    let existing: Option<EdgeId> = conn
        .query_row(
            "SELECT id FROM _quack_graph_edges \
             WHERE source_node_id = ? AND target_node_id = ? AND relation_id = ?",
            duckdb::params![source, target, relation_id],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(id) = existing {
        if standing == Standing::Reviewed {
            conn.execute(
                "UPDATE _quack_graph_edges SET provisional = false WHERE id = ?",
                duckdb::params![id],
            )?;
        }
        return Ok(id);
    }
    let id = EdgeId::generate();
    conn.execute(
        "INSERT INTO _quack_graph_edges (id, source_node_id, target_node_id, relation_id, properties, provisional) \
         VALUES (?, ?, ?, ?, ?, ?)",
        duckdb::params![
            id,
            source,
            target,
            relation_id,
            properties.to_json(),
            standing
        ],
    )?;
    Ok(id)
}

/// Attach a source to a node or edge; the same source twice is one row.
///
/// # Errors
///
/// Returns an error if the insert fails.
pub fn add_provenance(
    db: &WorkspaceDb,
    subject_id: &(impl AsRef<str> + ?Sized),
    source: &Source,
) -> Result<()> {
    let subject_id = subject_id.as_ref();
    // The unused pair is stored as empty text: it is part of the key.
    let (document_id, chunk_id, table_name, row_key) = match &source.origin {
        Origin::Chunk {
            document_id,
            chunk_id,
        } => (
            document_id.as_ref().map(DocumentId::as_str),
            chunk_id.as_str(),
            "",
            "",
        ),
        Origin::Row {
            table_name,
            row_key,
        } => (None, "", table_name.as_str(), row_key.as_str()),
        Origin::Manual { author, note, .. } => {
            return Assertion {
                author: author.clone(),
                note: note.clone(),
            }
            .write(db, subject_id);
        }
    };
    db.connection().execute(
        "INSERT OR IGNORE INTO _quack_provenance (subject_id, document_id, chunk_id, table_name, row_key, confidence) \
         VALUES (?, ?, ?, ?, ?, ?)",
        duckdb::params![
            subject_id,
            document_id,
            chunk_id,
            table_name,
            row_key,
            source.confidence
        ],
    )?;
    Ok(())
}

const NODE_COLUMNS: &str =
    "id, label, class_id, CAST(properties AS VARCHAR), provisional FROM _quack_graph_nodes";

/// A node from a row selected with [`NODE_COLUMNS`].
impl TryFrom<&duckdb::Row<'_>> for Node {
    type Error = duckdb::Error;

    fn try_from(row: &duckdb::Row<'_>) -> duckdb::Result<Self> {
        let properties: Option<String> = row.get(3)?;
        Ok(Self {
            id: row.get(0)?,
            label: row.get(1)?,
            class_id: row.get(2)?,
            properties: Properties::from_column(properties.as_deref()),
            standing: row.get(4)?,
        })
    }
}

const EDGE_COLUMNS: &str = "id, source_node_id, target_node_id, relation_id, weight, \
     CAST(properties AS VARCHAR), provisional FROM _quack_graph_edges";

/// An edge from a row selected with [`EDGE_COLUMNS`].
impl TryFrom<&duckdb::Row<'_>> for Edge {
    type Error = duckdb::Error;

    fn try_from(row: &duckdb::Row<'_>) -> duckdb::Result<Self> {
        let properties: Option<String> = row.get(5)?;
        Ok(Self {
            id: row.get(0)?,
            source_node_id: row.get(1)?,
            target_node_id: row.get(2)?,
            relation_id: row.get(3)?,
            weight: row.get::<_, Option<f64>>(4)?.unwrap_or(1.0),
            properties: Properties::from_column(properties.as_deref()),
            standing: row.get(6)?,
        })
    }
}

/// Every node id, ordered by class then label.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn all_node_ids(db: &WorkspaceDb) -> Result<Vec<NodeId>> {
    let mut stmt = db
        .connection()
        .prepare("SELECT id FROM _quack_graph_nodes ORDER BY class_id, label, id")?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(row.get(0)?);
    }
    Ok(out)
}

/// The chunks any of these nodes were extracted from. This is the edge
/// between the graph and retrieval: it turns an entity into the passages
/// that mention it (design doc 6.4, provenance).
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn chunks_of_nodes(db: &WorkspaceDb, node_ids: &[NodeId]) -> Result<Vec<ChunkId>> {
    let mut stmt = db.connection().prepare(
        "SELECT DISTINCT chunk_id FROM _quack_provenance \
         WHERE list_contains(?::VARCHAR[], subject_id) AND chunk_id <> '' ORDER BY chunk_id",
    )?;
    let mut rows = stmt.query(duckdb::params![IdList::new(node_ids)])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(row.get::<_, ChunkId>(0)?);
    }
    Ok(out)
}

/// The entities each of these chunks was the source of, as
/// `label (class)`, at most `per_chunk` each. The reverse of
/// [`chunks_of_nodes`]: it tells a retrieved passage what the graph
/// already knows is in it.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn entities_of_chunks(
    db: &WorkspaceDb,
    chunk_ids: &[ChunkId],
    per_chunk: usize,
) -> Result<BTreeMap<ChunkId, Vec<String>>> {
    let mut stmt = db.connection().prepare(
        "SELECT n.id, n.label, n.class_id, CAST(n.properties AS VARCHAR), n.provisional, p.chunk_id \
         FROM _quack_provenance p JOIN _quack_graph_nodes n ON n.id = p.subject_id \
         WHERE list_contains(?::VARCHAR[], p.chunk_id) ORDER BY p.chunk_id, n.label",
    )?;
    let mut rows = stmt.query(duckdb::params![IdList::new(chunk_ids)])?;
    let mut out: BTreeMap<ChunkId, Vec<String>> = BTreeMap::new();
    while let Some(row) = rows.next()? {
        let chunk_id: ChunkId = row.get(5)?;
        let entity = Node::try_from(row)?.to_string();
        let entities = out.entry(chunk_id).or_default();
        if entities.len() < per_chunk && !entities.contains(&entity) {
            entities.push(entity);
        }
    }
    Ok(out)
}

/// How many nodes a set of classes has, and the first few of their labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassCensus {
    pub total: u64,
    /// Labels in order, at most the number asked for.
    pub samples: Vec<String>,
}

/// How many nodes carry any of `class_ids`, and the first few labels.
/// Counting is the one graph question traversal cannot answer: user SQL
/// may not read `_quack_` tables, and a class listing stops at
/// `max_nodes`.
///
/// # Errors
///
/// Returns an error if a query fails.
pub fn class_census(db: &WorkspaceDb, class_ids: &[ClassId], samples: u32) -> Result<ClassCensus> {
    let total = class_count(db, class_ids)?;
    let mut stmt = db.connection().prepare(
        "SELECT label FROM _quack_graph_nodes WHERE list_contains(?::VARCHAR[], class_id) \
         ORDER BY label LIMIT ?",
    )?;
    let mut rows = stmt.query(duckdb::params![IdList::new(class_ids), i64::from(samples)])?;
    let mut labels = Vec::new();
    while let Some(row) = rows.next()? {
        labels.push(row.get::<_, String>(0)?);
    }
    Ok(ClassCensus {
        total,
        samples: labels,
    })
}

/// How many nodes carry any of `class_ids`.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn class_count(db: &WorkspaceDb, class_ids: &[ClassId]) -> Result<u64> {
    Ok(db.connection().query_row(
        "SELECT count(*) FROM _quack_graph_nodes WHERE list_contains(?::VARCHAR[], class_id)",
        duckdb::params![IdList::new(class_ids)],
        |row| row.get(0),
    )?)
}

/// One node by id.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn node(db: &WorkspaceDb, id: &NodeId) -> Result<Option<Node>> {
    let sql = format!("SELECT {NODE_COLUMNS} WHERE id = ?");
    let mut stmt = db.connection().prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![id])?;
    match rows.next()? {
        Some(row) => Ok(Some(Node::try_from(row)?)),
        None => Ok(None),
    }
}

/// Nodes by id, in the order given.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn nodes(db: &WorkspaceDb, ids: &[NodeId]) -> Result<Vec<Node>> {
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        if let Some(node) = node(db, id)? {
            out.push(node);
        }
    }
    Ok(out)
}

/// Which edges of a set of nodes to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeScope {
    /// Both ends in the set.
    Among,
    /// Either end in the set.
    Touching,
}

impl EdgeScope {
    fn joiner(self) -> &'static str {
        match self {
            Self::Among => "AND",
            Self::Touching => "OR",
        }
    }
}

/// The edges of `node_ids` in `scope`.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn edges(db: &WorkspaceDb, node_ids: &[NodeId], scope: EdgeScope) -> Result<Vec<Edge>> {
    if node_ids.is_empty() {
        return Ok(Vec::new());
    }
    let sql = format!(
        "SELECT {EDGE_COLUMNS} WHERE list_contains(?::VARCHAR[], source_node_id) \
         {} list_contains(?::VARCHAR[], target_node_id) ORDER BY relation_id, id",
        scope.joiner()
    );
    let list = IdList::new(node_ids);
    let mut stmt = db.connection().prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![list, list])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(Edge::try_from(row)?);
    }
    Ok(out)
}

/// Ids bound as one parameter: a `DuckDB` list literal the statement casts
/// with `?::VARCHAR[]`, so a set of any size is one bound value.
pub(crate) struct IdList(String);

impl IdList {
    pub(crate) fn new(ids: &[impl AsRef<str>]) -> Self {
        let quoted: Vec<String> = ids
            .iter()
            .map(|id| format!("'{}'", id.as_ref().replace('\'', "''")))
            .collect();
        Self(format!("[{}]", quoted.join(",")))
    }
}

impl duckdb::ToSql for IdList {
    fn to_sql(&self) -> duckdb::Result<ToSqlOutput<'_>> {
        self.0.to_sql()
    }
}

/// Provenance rows for the given nodes and edges.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn provenance_of(db: &WorkspaceDb, subject_ids: &[impl AsRef<str>]) -> Result<Vec<Provenance>> {
    if subject_ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut stmt = db.connection().prepare(
        "SELECT subject_id, document_id, chunk_id, table_name, row_key, confidence, \
                author, note, CAST(asserted_at AS VARCHAR) \
         FROM _quack_provenance WHERE list_contains(?::VARCHAR[], subject_id) \
         ORDER BY subject_id, chunk_id, table_name, row_key",
    )?;
    let mut rows = stmt.query(duckdb::params![IdList::new(subject_ids)])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(Provenance {
            subject_id: row.get(0)?,
            origin: Origin::from_columns(ProvenanceColumns {
                document_id: row.get(1)?,
                chunk_id: row.get(2)?,
                table_name: row.get(3)?,
                row_key: row.get(4)?,
                author: row.get(6)?,
                note: row.get(7)?,
                asserted_at: row.get(8)?,
            }),
            confidence: row.get::<_, Option<f64>>(5)?.unwrap_or(1.0),
        });
    }
    Ok(out)
}

/// The version the graph was last built or revalidated with.
///
/// # Errors
///
/// Returns an error if the read fails.
pub fn built_with(db: &WorkspaceDb) -> Result<Option<OntologyVersion>> {
    Ok(db
        .meta(MetaKey::GraphBuiltWithOntologyVersion)?
        .and_then(|v| v.parse().ok()))
}

/// Record the ontology version the graph now matches.
///
/// # Errors
///
/// Returns an error if the write fails.
pub fn set_built_with(db: &WorkspaceDb, version: OntologyVersion) -> Result<()> {
    db.set_meta(MetaKey::GraphBuiltWithOntologyVersion, &version.to_string())
}

/// The accumulated drift.
///
/// # Errors
///
/// Returns an error if the read fails.
pub fn drift(db: &WorkspaceDb) -> Result<Drift> {
    Ok(db
        .meta(MetaKey::GraphDrift)?
        .and_then(|v| serde_json::from_str(&v).ok())
        .unwrap_or_default())
}

/// Add a run's drift to the stored total.
///
/// # Errors
///
/// Returns an error if the write fails.
pub fn record_drift(db: &WorkspaceDb, run: &Drift) -> Result<()> {
    let mut total = drift(db)?;
    total.absorb(run);
    db.set_meta(MetaKey::GraphDrift, &serde_json::to_string(&total)?)
}

/// Size, provisional and stale flags, pending merges, and drift.
///
/// # Errors
///
/// Returns an error if a read fails.
pub fn status(db: &WorkspaceDb) -> Result<GraphStatus> {
    let conn = db.connection();
    let (nodes, provisional_nodes): (i64, i64) = conn.query_row(
        "SELECT count(*), count(*) FILTER (WHERE provisional) FROM _quack_graph_nodes",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    let edges: i64 = conn.query_row("SELECT count(*) FROM _quack_graph_edges", [], |r| r.get(0))?;
    let pending_merges: i64 = conn.query_row(
        "SELECT count(*) FROM _quack_graph_merges WHERE status = ?",
        [MergeStatus::Pending],
        |r| r.get(0),
    )?;
    let built_with_version = built_with(db)?;
    let ontology_version = ontology_store::latest_version(db)?;
    let (missing_tables, pending_tables) = match ontology_store::current(db)? {
        Some(ontology) => {
            let tables = db.list_tables()?;
            let missing = ontology
                .mappings
                .iter()
                .map(|m| m.table.clone())
                .filter(|t| !tables.contains(t))
                .collect();
            (missing, tables::pending(db, &ontology, &tables)?)
        }
        None => (Vec::new(), Vec::new()),
    };
    Ok(GraphStatus {
        nodes: u64::try_from(nodes).unwrap_or(0),
        edges: u64::try_from(edges).unwrap_or(0),
        provisional_nodes: u64::try_from(provisional_nodes).unwrap_or(0),
        built_with_version,
        ontology_version,
        // Stale needs a recorded build that lags and something built: a
        // never-built graph (`None`) and an empty one are not stale.
        stale: nodes > 0 && built_with_version.is_some_and(|built| Some(built) < ontology_version),
        pending_merges: u64::try_from(pending_merges).unwrap_or(0),
        drift: drift(db)?,
        missing_tables,
        pending_chunks: db.pool_size(SamplePool::NotGraphExtracted)?,
        pending_tables,
    })
}

/// What [`clear`] leaves standing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    /// Nodes and edges a person asserted, with their manual provenance
    /// (an edge only while both its ends stay).
    Asserted,
    Nothing,
}

/// Remove every node, edge, provenance row, merge proposal, the record of
/// extracted chunks and built tables, and the drift, except what `keep`
/// names.
///
/// # Errors
///
/// Returns an error if a delete fails.
pub fn clear(db: &WorkspaceDb, keep: Keep) -> Result<()> {
    const ASSERTED: &str = "SELECT subject_id FROM _quack_provenance WHERE asserted_at IS NOT NULL";
    let conn = db.connection();
    match keep {
        Keep::Nothing => conn.execute_batch(
            "DELETE FROM _quack_provenance; DELETE FROM _quack_graph_edges; \
             DELETE FROM _quack_graph_nodes;",
        )?,
        Keep::Asserted => conn.execute_batch(&format!(
            "DELETE FROM _quack_graph_nodes WHERE id NOT IN ({ASSERTED}); \
             DELETE FROM _quack_graph_edges WHERE id NOT IN ({ASSERTED}) \
                OR source_node_id NOT IN (SELECT id FROM _quack_graph_nodes) \
                OR target_node_id NOT IN (SELECT id FROM _quack_graph_nodes); \
             DELETE FROM _quack_provenance WHERE asserted_at IS NULL \
                OR (subject_id NOT IN (SELECT id FROM _quack_graph_nodes) \
                    AND subject_id NOT IN (SELECT id FROM _quack_graph_edges));"
        ))?,
    }
    conn.execute_batch(
        "DELETE FROM _quack_graph_merges; DELETE FROM _quack_graph_extracted; \
         DELETE FROM _quack_graph_tables_built;",
    )?;
    db.set_meta(MetaKey::GraphDrift, "{}")?;
    db.delete_meta(MetaKey::GraphBuiltWithOntologyVersion)
}

/// A node as a person refers to it: its id, or its exact label (an alias
/// counts), in one class or any.
///
/// # Errors
///
/// Returns an error when nothing matches, or more than one node does (the
/// error names them, so the caller can give the id).
pub fn find_node(db: &WorkspaceDb, reference: &str, class_id: Option<&str>) -> Result<Node> {
    let reference = reference.trim();
    if let Some(found) = node(db, &NodeId::from(reference))? {
        return Ok(found);
    }
    let matches = super::traverse::resolve_entry(db, reference, class_id, None)?;
    match matches.as_slice() {
        [] => Err(ResourceKind::GraphNode.missing(reference)),
        [one] => Ok(one.clone()),
        many => Err(Error::Analysis(format!(
            "'{reference}' names {} nodes; give the id: {}",
            many.len(),
            many.iter()
                .map(|n| format!("{n} {}", n.id))
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

/// What a person's node write did: the node, and whether it is new.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Asserted<T> {
    #[serde(flatten)]
    pub subject: T,
    pub created: bool,
}

/// The ontology a person's write is checked against.
fn asserting_ontology(db: &WorkspaceDb) -> Result<Ontology> {
    ontology_store::current(db)?.ok_or_else(|| {
        Error::Ontology(String::from(
            "no ontology yet: the graph takes only classes and relations an ontology defines",
        ))
    })
}

/// Assert a node: a new one of `class_id`, or the one that already has
/// this label and class, which takes the given properties (its own
/// values give way) and stops being provisional. Writes the manual
/// provenance row either way, in one transaction.
///
/// # Errors
///
/// Returns an error when there is no ontology, the class is not in it,
/// the label is blank, or a write fails.
pub fn create_node(
    db: &WorkspaceDb,
    node: &NewNode,
    assertion: &Assertion,
) -> Result<Asserted<Node>> {
    let ontology = asserting_ontology(db)?;
    if !ontology.defines_class(node.class_id.as_str()) {
        return Err(Error::Ontology(format!(
            "no class '{}' in the ontology; the classes are {}",
            node.class_id,
            ontology.class_ids().join(", ")
        )));
    }
    let normalized = NormalizedLabel::new(&node.label);
    if normalized.is_empty() {
        return Err(Error::Analysis(String::from("a node needs a label")));
    }
    let tx = db.connection().unchecked_transaction()?;
    let existing: Option<(NodeId, Option<String>)> = tx
        .query_row(
            "SELECT id, CAST(properties AS VARCHAR) FROM _quack_graph_nodes \
             WHERE normalized_label = ? AND class_id = ?",
            duckdb::params![normalized, node.class_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    let (id, created) = if let Some((id, properties)) = existing {
        let mut merged = node.properties.clone();
        merged.fill_from(&Properties::from_column(properties.as_deref()));
        tx.execute(
            "UPDATE _quack_graph_nodes SET properties = ?, provisional = false WHERE id = ?",
            duckdb::params![merged.to_json(), id],
        )?;
        (id, false)
    } else {
        let id = NodeId::generate();
        tx.execute(
            "INSERT INTO _quack_graph_nodes (id, label, normalized_label, class_id, properties, provisional) \
             VALUES (?, ?, ?, ?, ?, false)",
            duckdb::params![
                id,
                node.label.trim(),
                normalized,
                node.class_id,
                node.properties.to_json()
            ],
        )?;
        (id, true)
    };
    assertion.write(db, id.as_str())?;
    tx.commit()?;
    let subject =
        self::node(db, &id)?.ok_or_else(|| ResourceKind::GraphNode.missing(id.as_str()))?;
    Ok(Asserted { subject, created })
}

/// What a person changes on a node; a field left `None` stays.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeEdit {
    pub label: Option<String>,
    pub class: Option<ClassId>,
    /// Merged into the node's properties: a value sets its key, `null`
    /// removes it.
    pub properties: Option<serde_json::Map<String, serde_json::Value>>,
}

impl NodeEdit {
    /// Whether anything is set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.label.is_none() && self.class.is_none() && self.properties.is_none()
    }
}

/// Correct a node: its label, class, or properties, under the ontology,
/// with the manual provenance row written. A class change must leave
/// every edge at the node allowed; a label change must not collide with
/// another node of the class.
///
/// # Errors
///
/// Returns an error when the node is missing, the change breaks the
/// ontology or collides, or a write fails.
pub fn update_node(
    db: &WorkspaceDb,
    id: &NodeId,
    edit: &NodeEdit,
    assertion: &Assertion,
) -> Result<Node> {
    let ontology = asserting_ontology(db)?;
    let current = node(db, id)?.ok_or_else(|| ResourceKind::GraphNode.missing(id.as_str()))?;
    let label = edit
        .label
        .as_deref()
        .map_or_else(|| current.label.clone(), |l| l.trim().to_owned());
    let normalized = NormalizedLabel::new(&label);
    if normalized.is_empty() {
        return Err(Error::Analysis(String::from("a node needs a label")));
    }
    let class_id = edit
        .class
        .clone()
        .unwrap_or_else(|| current.class_id.clone());
    if !ontology.defines_class(class_id.as_str()) {
        return Err(Error::Ontology(format!(
            "no class '{class_id}' in the ontology; the classes are {}",
            ontology.class_ids().join(", ")
        )));
    }
    let tx = db.connection().unchecked_transaction()?;
    let taken: Option<NodeId> = tx
        .query_row(
            "SELECT id FROM _quack_graph_nodes WHERE normalized_label = ? AND class_id = ? AND id <> ?",
            duckdb::params![normalized, class_id, id],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(taken) = taken {
        return Err(Error::Analysis(format!(
            "another {class_id} node already has the label '{label}' ({taken}); merge them instead"
        )));
    }
    if class_id != current.class_id {
        for edge in edges(db, std::slice::from_ref(id), EdgeScope::Touching)? {
            let (source, target) = if edge.source_node_id == *id {
                (class_id.clone(), class_of(db, &edge.target_node_id)?)
            } else {
                (class_of(db, &edge.source_node_id)?, class_id.clone())
            };
            if !ontology.allows_edge(edge.relation_id.as_str(), source.as_str(), target.as_str()) {
                return Err(Error::Ontology(format!(
                    "as a {class_id} the node could not take its edge {} ({} -> {}); delete that edge first",
                    edge.relation_id, source, target
                )));
            }
        }
    }
    let mut properties = current.properties;
    if let Some(patch) = &edit.properties {
        properties.patch(patch);
    }
    tx.execute(
        "UPDATE _quack_graph_nodes SET label = ?, normalized_label = ?, class_id = ?, \
         properties = ?, provisional = false WHERE id = ?",
        duckdb::params![label, normalized, class_id, properties.to_json(), id],
    )?;
    assertion.write(db, id.as_str())?;
    tx.commit()?;
    node(db, id)?.ok_or_else(|| ResourceKind::GraphNode.missing(id.as_str()))
}

/// The class of a node that must exist.
fn class_of(db: &WorkspaceDb, id: &NodeId) -> Result<ClassId> {
    Ok(node(db, id)?
        .ok_or_else(|| ResourceKind::GraphNode.missing(id.as_str()))?
        .class_id)
}

/// Delete a node with its edges, their provenance, its provenance, and
/// its merge proposals, in one transaction.
///
/// # Errors
///
/// Returns an error when the node is missing or a delete fails.
pub fn delete_node(db: &WorkspaceDb, id: &NodeId) -> Result<Node> {
    let current = node(db, id)?.ok_or_else(|| ResourceKind::GraphNode.missing(id.as_str()))?;
    let tx = db.connection().unchecked_transaction()?;
    tx.execute(
        "DELETE FROM _quack_provenance WHERE subject_id IN \
         (SELECT id FROM _quack_graph_edges WHERE source_node_id = ? OR target_node_id = ?)",
        duckdb::params![id, id],
    )?;
    tx.execute(
        "DELETE FROM _quack_graph_edges WHERE source_node_id = ? OR target_node_id = ?",
        duckdb::params![id, id],
    )?;
    tx.execute(
        "DELETE FROM _quack_provenance WHERE subject_id = ?",
        duckdb::params![id],
    )?;
    tx.execute(
        "DELETE FROM _quack_graph_merges WHERE keep_node_id = ? OR drop_node_id = ?",
        duckdb::params![id, id],
    )?;
    tx.execute(
        "DELETE FROM _quack_graph_nodes WHERE id = ?",
        duckdb::params![id],
    )?;
    tx.commit()?;
    Ok(current)
}

/// An edge a person asserts between two existing nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewEdge {
    pub source: NodeId,
    pub target: NodeId,
    pub relation: RelationId,
    pub properties: Properties,
}

/// Assert an edge: a new one, or the one that already joins these nodes
/// by this relation, which stops being provisional. The relation must
/// join the two nodes' classes under the ontology. Writes the manual
/// provenance row either way, in one transaction.
///
/// # Errors
///
/// Returns an error when a node is missing, the ontology does not allow
/// the edge, or a write fails.
pub fn create_edge(
    db: &WorkspaceDb,
    edge: &NewEdge,
    assertion: &Assertion,
) -> Result<Asserted<Edge>> {
    let ontology = asserting_ontology(db)?;
    let source = class_of(db, &edge.source)?;
    let target = class_of(db, &edge.target)?;
    if !ontology.defines_relation(edge.relation.as_str()) {
        return Err(Error::Ontology(format!(
            "no relation '{}' in the ontology; the relations are {}",
            edge.relation,
            ontology.relation_ids().join(", ")
        )));
    }
    if !ontology.allows_edge(edge.relation.as_str(), source.as_str(), target.as_str()) {
        return Err(Error::Ontology(format!(
            "the ontology does not allow {} from a {source} to a {target}",
            edge.relation
        )));
    }
    let tx = db.connection().unchecked_transaction()?;
    let existing: Option<EdgeId> = tx
        .query_row(
            "SELECT id FROM _quack_graph_edges \
             WHERE source_node_id = ? AND target_node_id = ? AND relation_id = ?",
            duckdb::params![edge.source, edge.target, edge.relation],
            |r| r.get(0),
        )
        .optional()?;
    let (id, created) = if let Some(id) = existing {
        let mut properties = edge.properties.clone();
        let current: Option<String> = tx.query_row(
            "SELECT CAST(properties AS VARCHAR) FROM _quack_graph_edges WHERE id = ?",
            duckdb::params![id],
            |r| r.get(0),
        )?;
        properties.fill_from(&Properties::from_column(current.as_deref()));
        tx.execute(
            "UPDATE _quack_graph_edges SET properties = ?, provisional = false WHERE id = ?",
            duckdb::params![properties.to_json(), id],
        )?;
        (id, false)
    } else {
        let id = EdgeId::generate();
        tx.execute(
            "INSERT INTO _quack_graph_edges (id, source_node_id, target_node_id, relation_id, properties, provisional) \
             VALUES (?, ?, ?, ?, ?, false)",
            duckdb::params![
                id,
                edge.source,
                edge.target,
                edge.relation,
                edge.properties.to_json()
            ],
        )?;
        (id, true)
    };
    assertion.write(db, id.as_str())?;
    tx.commit()?;
    let subject =
        self::edge(db, &id)?.ok_or_else(|| ResourceKind::GraphEdge.missing(id.as_str()))?;
    Ok(Asserted { subject, created })
}

/// One edge by id.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn edge(db: &WorkspaceDb, id: &EdgeId) -> Result<Option<Edge>> {
    let sql = format!("SELECT {EDGE_COLUMNS} WHERE id = ?");
    let mut stmt = db.connection().prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![id])?;
    match rows.next()? {
        Some(row) => Ok(Some(Edge::try_from(row)?)),
        None => Ok(None),
    }
}

/// Delete an edge with its provenance.
///
/// # Errors
///
/// Returns an error when the edge is missing or a delete fails.
pub fn delete_edge(db: &WorkspaceDb, id: &EdgeId) -> Result<Edge> {
    let current = edge(db, id)?.ok_or_else(|| ResourceKind::GraphEdge.missing(id.as_str()))?;
    let tx = db.connection().unchecked_transaction()?;
    tx.execute(
        "DELETE FROM _quack_provenance WHERE subject_id = ?",
        duckdb::params![id],
    )?;
    tx.execute(
        "DELETE FROM _quack_graph_edges WHERE id = ?",
        duckdb::params![id],
    )?;
    tx.commit()?;
    Ok(current)
}

/// The nodes and edges one extraction stored.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChunkYield {
    pub nodes: u32,
    pub edges: u32,
}

/// Note that a chunk was extracted under an ontology version, with what
/// it yielded, so incremental runs skip it (issue #60).
///
/// # Errors
///
/// Returns an error if the write fails.
pub fn record_extracted(
    db: &WorkspaceDb,
    chunk_id: &ChunkId,
    ontology_version: OntologyVersion,
    yielded: ChunkYield,
) -> Result<()> {
    db.connection().execute(
        "INSERT OR REPLACE INTO _quack_graph_extracted (chunk_id, ontology_version, nodes, edges) \
         VALUES (?, ?, ?, ?)",
        duckdb::params![chunk_id, ontology_version, yielded.nodes, yielded.edges],
    )?;
    Ok(())
}

/// The chunks of `documents` no extraction has read, in document order.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn unextracted_chunks_of(db: &WorkspaceDb, documents: &[DocumentId]) -> Result<Vec<ChunkId>> {
    let mut stmt = db.connection().prepare(
        "SELECT c.id FROM _quack_chunks c \
         WHERE list_contains(?::VARCHAR[], c.document_id) \
           AND NOT EXISTS (SELECT 1 FROM _quack_graph_extracted x WHERE x.chunk_id = c.id) \
         ORDER BY c.document_id, c.chunk_index",
    )?;
    let mut rows = stmt.query(duckdb::params![IdList::new(documents)])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(row.get::<_, ChunkId>(0)?);
    }
    Ok(out)
}

/// How many chunks the record says were extracted.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn extracted_chunks(db: &WorkspaceDb) -> Result<u64> {
    let count: i64 =
        db.connection()
            .query_row("SELECT count(*) FROM _quack_graph_extracted", [], |r| {
                r.get(0)
            })?;
    Ok(u64::try_from(count).unwrap_or(0))
}

/// Mark every node and edge reviewed (no longer provisional).
///
/// # Errors
///
/// Returns an error if an update fails.
pub fn mark_reviewed(db: &WorkspaceDb) -> Result<()> {
    let conn = db.connection();
    conn.execute_batch(
        "UPDATE _quack_graph_nodes SET provisional = false; \
         UPDATE _quack_graph_edges SET provisional = false;",
    )?;
    Ok(())
}

/// Move nodes and edges to renamed class and relation ids, inside the
/// save that writes ontology version `next`. A graph that matched the
/// version before still matches, so its recorded version advances with it.
///
/// # Errors
///
/// Returns an error when rows left from an earlier ontology already carry
/// a new id (moving onto them would merge two classes or relations), or a
/// write fails.
pub(crate) fn rename_ids(
    db: &WorkspaceDb,
    renames: &IdRenames,
    next: OntologyVersion,
) -> Result<()> {
    let conn = db.connection();
    for (old, new) in &renames.classes {
        let taken: u64 = conn.query_row(
            "SELECT count(*) FROM _quack_graph_nodes WHERE class_id = ?",
            duckdb::params![new],
            |row| row.get(0),
        )?;
        if taken > 0 {
            return Err(Error::Ontology(format!(
                "the graph still holds {taken} nodes of a class '{new}' from an earlier ontology; \
                 run `quack graph revalidate` before renaming '{old}' to it"
            )));
        }
        conn.execute(
            "UPDATE _quack_graph_nodes SET class_id = ? WHERE class_id = ?",
            duckdb::params![new, old],
        )?;
    }
    for (old, new) in &renames.relations {
        let taken: u64 = conn.query_row(
            "SELECT count(*) FROM _quack_graph_edges WHERE relation_id = ?",
            duckdb::params![new],
            |row| row.get(0),
        )?;
        if taken > 0 {
            return Err(Error::Ontology(format!(
                "the graph still holds {taken} edges of a relation '{new}' from an earlier ontology; \
                 run `quack graph revalidate` before renaming '{old}' to it"
            )));
        }
        conn.execute(
            "UPDATE _quack_graph_edges SET relation_id = ? WHERE relation_id = ?",
            duckdb::params![new, old],
        )?;
    }
    if built_with(db)?.is_some_and(|built| Some(built) == next.previous()) {
        set_built_with(db, next)?;
    }
    Ok(())
}

/// What a revalidation drops: what [`revalidate`] removed, or what
/// [`Self::preview`] says it would, so an interface can ask first.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Revalidation {
    pub dropped_nodes: u64,
    /// Every edge that goes, those of dropped nodes included.
    pub dropped_edges: u64,
    /// The ontology version the graph matches afterwards.
    pub version: OntologyVersion,
    /// Nodes per class id the ontology no longer defines.
    pub classes: BTreeMap<String, u64>,
    /// Edges per relation id the ontology no longer defines. The other
    /// dropped edges lose an end or no longer fit their relation.
    pub relations: BTreeMap<String, u64>,
}

impl Revalidation {
    /// Count what the current ontology no longer allows, changing
    /// nothing.
    ///
    /// # Errors
    ///
    /// Returns an error when there is no ontology or a read fails.
    pub fn preview(db: &WorkspaceDb) -> Result<Self> {
        Ok(Self::find(db)?.0)
    }

    /// What the current ontology no longer allows: the counts, and the
    /// edges that go as the sets [`revalidate`] deletes. The nodes that go
    /// are those of `classes`. Both the preview and the run start here, so
    /// the run deletes what the preview counted. It reads one row per
    /// class and per (relation, source class, target class) combination,
    /// not per node or edge.
    fn find(db: &WorkspaceDb) -> Result<(Self, Vec<EdgeSet>)> {
        let ontology = validating_ontology(db)?;
        let mut found = Self {
            version: ontology.saved_version()?,
            dropped_nodes: 0,
            dropped_edges: 0,
            classes: BTreeMap::new(),
            relations: BTreeMap::new(),
        };
        let conn = db.connection();
        let mut stmt =
            conn.prepare("SELECT class_id, count(*) FROM _quack_graph_nodes GROUP BY class_id")?;
        let per_class = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<duckdb::Result<Vec<(String, u64)>>>()?;
        for (class, nodes) in per_class {
            if !ontology.defines_class(&class) {
                found.dropped_nodes = found.dropped_nodes.saturating_add(nodes);
                found.classes.insert(class, nodes);
            }
        }
        let mut stmt = conn.prepare(
            "SELECT e.relation_id, s.class_id, t.class_id, count(*) FROM _quack_graph_edges e \
             LEFT JOIN _quack_graph_nodes s ON s.id = e.source_node_id \
             LEFT JOIN _quack_graph_nodes t ON t.id = e.target_node_id \
             GROUP BY ALL",
        )?;
        let combinations = stmt
            .query_map([], |row| {
                let set = EdgeSet {
                    relation: row.get(0)?,
                    source: row.get(1)?,
                    target: row.get(2)?,
                };
                Ok((set, row.get(3)?))
            })?
            .collect::<duckdb::Result<Vec<(EdgeSet, u64)>>>()?;
        let mut sets = Vec::new();
        for (set, edges) in combinations {
            // An edge stays only when both ends exist, both stay, and the
            // relation still joins their classes.
            let stays = match (&set.source, &set.target) {
                (Some(source), Some(target)) => {
                    ontology.defines_class(source)
                        && ontology.defines_class(target)
                        && ontology.allows_edge(&set.relation, source, target)
                }
                (Some(_) | None, None) | (None, Some(_)) => false,
            };
            if stays {
                continue;
            }
            found.dropped_edges = found.dropped_edges.saturating_add(edges);
            if !ontology.defines_relation(&set.relation) {
                let count = found.relations.entry(set.relation.clone()).or_default();
                *count = count.saturating_add(edges);
            }
            sets.push(set);
        }
        Ok((found, sets))
    }

    /// Whether revalidating would drop nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dropped_nodes == 0 && self.dropped_edges == 0
    }

    /// Dropped edges of relations the ontology still defines: those that
    /// lose an end or no longer fit the relation's domain and range.
    #[must_use]
    pub fn misfit_edges(&self) -> u64 {
        self.relations
            .values()
            .fold(self.dropped_edges, |rest, edges| {
                rest.saturating_sub(*edges)
            })
    }
}

/// A preview as `quack graph revalidate` prints it before asking.
impl fmt::Display for Revalidation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return writeln!(
                f,
                "Nothing to drop: every node and edge fits ontology version {}.",
                self.version
            );
        }
        writeln!(
            f,
            "Revalidating against ontology version {} drops {} nodes and {} edges:",
            self.version, self.dropped_nodes, self.dropped_edges
        )?;
        for (class, nodes) in &self.classes {
            writeln!(
                f,
                "  class {class}, which the ontology no longer defines: {nodes} nodes"
            )?;
        }
        for (relation, edges) in &self.relations {
            writeln!(
                f,
                "  relation {relation}, which the ontology no longer defines: {edges} edges"
            )?;
        }
        match self.misfit_edges() {
            0 => Ok(()),
            edges => writeln!(
                f,
                "  {edges} edges that lose an end or no longer fit their relation's domain and range"
            ),
        }
    }
}

/// The ontology a graph is validated against.
fn validating_ontology(db: &WorkspaceDb) -> Result<Ontology> {
    ontology_store::current(db)?
        .ok_or_else(|| Error::Ontology(String::from("no ontology to validate against")))
}

/// Bring a stale graph in line with the current ontology without a model
/// call: nodes whose class no longer exists go, and edges whose relation
/// is gone, whose endpoints no longer fit its domain and range, or that
/// lose an end go. It deletes what [`Revalidation::preview`] counts, in
/// SQL, a set at a time, so the work does not grow with the graph in
/// memory.
///
/// # Errors
///
/// Returns an error when there is no ontology or a write fails.
pub fn revalidate(db: &WorkspaceDb) -> Result<Revalidation> {
    let (found, edge_sets) = Revalidation::find(db)?;
    // Edges first: a set is found through the nodes at its ends.
    for set in &edge_sets {
        set.delete(db)?;
    }
    let classes: Vec<&String> = found.classes.keys().collect();
    delete_nodes_of(db, &IdList::new(&classes))?;
    set_built_with(db, found.version)?;
    Ok(found)
}

/// Delete every node whose class is in `classes`, with its provenance and
/// its merge proposals. Its edges are gone already.
fn delete_nodes_of(db: &WorkspaceDb, classes: &IdList) -> Result<()> {
    const NODES: &str =
        "SELECT id FROM _quack_graph_nodes WHERE list_contains(?::VARCHAR[], class_id)";
    let conn = db.connection();
    conn.execute(
        &format!("DELETE FROM _quack_provenance WHERE subject_id IN ({NODES})"),
        duckdb::params![classes],
    )?;
    conn.execute(
        &format!(
            "DELETE FROM _quack_graph_merges \
             WHERE keep_node_id IN ({NODES}) OR drop_node_id IN ({NODES})"
        ),
        duckdb::params![classes, classes],
    )?;
    conn.execute(
        "DELETE FROM _quack_graph_nodes WHERE list_contains(?::VARCHAR[], class_id)",
        duckdb::params![classes],
    )?;
    Ok(())
}

/// Every edge of `relation` from a node of class `source` to a node of
/// class `target`; an end that no longer exists has no class.
struct EdgeSet {
    relation: String,
    source: Option<String>,
    target: Option<String>,
}

impl EdgeSet {
    /// Delete the set's edges and their provenance.
    fn delete(&self, db: &WorkspaceDb) -> Result<()> {
        const EDGES: &str = "SELECT e.id FROM _quack_graph_edges e \
             LEFT JOIN _quack_graph_nodes s ON s.id = e.source_node_id \
             LEFT JOIN _quack_graph_nodes t ON t.id = e.target_node_id \
             WHERE e.relation_id = ? AND s.class_id IS NOT DISTINCT FROM ? \
             AND t.class_id IS NOT DISTINCT FROM ?";
        let Self {
            relation,
            source,
            target,
        } = self;
        let conn = db.connection();
        conn.execute(
            &format!("DELETE FROM _quack_provenance WHERE subject_id IN ({EDGES})"),
            duckdb::params![relation, source, target],
        )?;
        conn.execute(
            &format!("DELETE FROM _quack_graph_edges WHERE id IN ({EDGES})"),
            duckdb::params![relation, source, target],
        )?;
        Ok(())
    }
}

/// Nodes whose label vector is missing or was made under another profile.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn nodes_needing_embedding(db: &WorkspaceDb, limit: u32) -> Result<Vec<Node>> {
    let sql = format!(
        "SELECT {NODE_COLUMNS} WHERE embedding IS NULL OR embedding_profile IS DISTINCT FROM ? \
         ORDER BY id LIMIT ?"
    );
    let mut stmt = db.connection().prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![
        db.embedding_fingerprint(),
        i64::from(limit)
    ])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(Node::try_from(row)?);
    }
    Ok(out)
}

/// How many nodes [`nodes_needing_embedding`] would work through.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn count_nodes_needing_embedding(db: &WorkspaceDb) -> Result<u32> {
    let count: i64 = db.connection().query_row(
        "SELECT count(*) FROM _quack_graph_nodes \
         WHERE embedding IS NULL OR embedding_profile IS DISTINCT FROM ?",
        duckdb::params![db.embedding_fingerprint()],
        |row| row.get(0),
    )?;
    Ok(u32::try_from(count).unwrap_or(u32::MAX))
}

/// A node near a query vector.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeMatch {
    pub node: Node,
    /// Cosine distance from the query.
    pub distance: f64,
}

/// Nodes whose label embedding is within `max_distance` (cosine) of
/// `query`, nearest first, optionally within one class.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn nearest_nodes(
    db: &WorkspaceDb,
    query: &Vector,
    class_id: Option<&str>,
    limit: u32,
) -> Result<Vec<NodeMatch>> {
    if !db.embedding_dimension().fits(query.len()) {
        return Ok(Vec::new());
    }
    let sql = format!(
        "SELECT id, label, class_id, CAST(properties AS VARCHAR), provisional, \
                array_cosine_distance(embedding, ?::{vt}) AS d \
         FROM _quack_graph_nodes \
         WHERE embedding IS NOT NULL AND embedding_profile IS NOT DISTINCT FROM ? \
           AND (? IS NULL OR class_id = ?) ORDER BY d LIMIT ?",
        vt = db.vector_type()
    );
    let literal = query.sql_literal();
    let mut stmt = db.connection().prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![
        literal,
        db.embedding_fingerprint(),
        class_id,
        class_id,
        i64::from(limit)
    ])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(NodeMatch {
            node: Node::try_from(row)?,
            distance: row.get(5)?,
        });
    }
    Ok(out)
}
