//! Persistence for nodes, edges, provenance, and the graph's bookkeeping in
//! `_quack_meta`: the ontology version it was built with, and drift.

use std::collections::BTreeMap;

use duckdb::OptionalExt as _;
use duckdb::types::ToSqlOutput;

use super::resolve::MergeStatus;
use super::{Drift, Edge, GraphStatus, Node, NormalizedLabel, Origin, Properties, Provenance};
use crate::error::{Error, Result};
use crate::ontology::{self, OntologyVersion, store as ontology_store};
use crate::storage::workspace::{MetaKey, WorkspaceDb, embedding_literal};

/// A node to store: merged into an existing one with the same normalized
/// label and class, else inserted.
#[derive(Debug, Clone)]
pub struct NewNode {
    pub label: String,
    pub class_id: String,
    pub properties: Properties,
    pub provisional: bool,
}

/// A source to attach to a node or edge.
#[derive(Debug, Clone)]
pub struct Source {
    pub origin: Origin,
    pub confidence: f64,
}

impl Source {
    #[must_use]
    pub fn chunk(document_id: &str, chunk_id: &str, confidence: f64) -> Self {
        Self {
            origin: Origin::Chunk {
                document_id: Some(document_id.to_owned()),
                chunk_id: chunk_id.to_owned(),
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

/// Insert or merge a node and return its id. Merging keeps the existing
/// label, adds properties the existing node lacks, and clears the
/// provisional flag only when the new evidence is not provisional either.
///
/// # Errors
///
/// Returns an error if a write fails.
pub fn upsert_node(db: &WorkspaceDb, node: &NewNode) -> Result<String> {
    let normalized = NormalizedLabel::new(&node.label);
    if normalized.is_empty() {
        return Err(Error::Analysis(String::from("a node needs a label")));
    }
    let conn = db.connection();
    // `optional`, not `.ok()`: a broken query must fail, not read as "not
    // found" and then insert a duplicate.
    let existing: Option<(String, Option<String>, bool)> = conn
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
            duckdb::params![merged.to_json(), provisional && node.provisional, id],
        )?;
        return Ok(id);
    }
    let id = uuid::Uuid::now_v7().to_string();
    conn.execute(
        "INSERT INTO _quack_graph_nodes (id, label, normalized_label, class_id, properties, provisional) \
         VALUES (?, ?, ?, ?, ?, ?)",
        duckdb::params![
            id,
            node.label.trim(),
            normalized,
            node.class_id,
            node.properties.to_json(),
            node.provisional
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
    source: &str,
    target: &str,
    relation_id: &str,
    properties: &Properties,
    provisional: bool,
) -> Result<String> {
    let conn = db.connection();
    let existing: Option<String> = conn
        .query_row(
            "SELECT id FROM _quack_graph_edges \
             WHERE source_node_id = ? AND target_node_id = ? AND relation_id = ?",
            duckdb::params![source, target, relation_id],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(id) = existing {
        if !provisional {
            conn.execute(
                "UPDATE _quack_graph_edges SET provisional = false WHERE id = ?",
                duckdb::params![id],
            )?;
        }
        return Ok(id);
    }
    let id = uuid::Uuid::now_v7().to_string();
    conn.execute(
        "INSERT INTO _quack_graph_edges (id, source_node_id, target_node_id, relation_id, properties, provisional) \
         VALUES (?, ?, ?, ?, ?, ?)",
        duckdb::params![
            id,
            source,
            target,
            relation_id,
            properties.to_json(),
            provisional
        ],
    )?;
    Ok(id)
}

/// Attach a source to a node or edge; the same source twice is one row.
///
/// # Errors
///
/// Returns an error if the insert fails.
pub fn add_provenance(db: &WorkspaceDb, subject_id: &str, source: &Source) -> Result<()> {
    // The unused pair is stored as empty text: it is part of the key.
    let (document_id, chunk_id, table_name, row_key) = match &source.origin {
        Origin::Chunk {
            document_id,
            chunk_id,
        } => (document_id.as_deref(), chunk_id.as_str(), "", ""),
        Origin::Row {
            table_name,
            row_key,
        } => (None, "", table_name.as_str(), row_key.as_str()),
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
            provisional: row.get(4)?,
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
            provisional: row.get(6)?,
        })
    }
}

/// Every node id, ordered by class then label.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn all_node_ids(db: &WorkspaceDb) -> Result<Vec<String>> {
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
pub fn chunks_of_nodes(db: &WorkspaceDb, node_ids: &[String]) -> Result<Vec<String>> {
    let mut stmt = db.connection().prepare(
        "SELECT DISTINCT chunk_id FROM _quack_provenance \
         WHERE list_contains(?::VARCHAR[], subject_id) AND chunk_id <> '' ORDER BY chunk_id",
    )?;
    let mut rows = stmt.query(duckdb::params![IdList::new(node_ids)])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(row.get::<_, String>(0)?);
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
    chunk_ids: &[String],
    per_chunk: usize,
) -> Result<BTreeMap<String, Vec<String>>> {
    let mut stmt = db.connection().prepare(
        "SELECT n.id, n.label, n.class_id, CAST(n.properties AS VARCHAR), n.provisional, p.chunk_id \
         FROM _quack_provenance p JOIN _quack_graph_nodes n ON n.id = p.subject_id \
         WHERE list_contains(?::VARCHAR[], p.chunk_id) ORDER BY p.chunk_id, n.label",
    )?;
    let mut rows = stmt.query(duckdb::params![IdList::new(chunk_ids)])?;
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    while let Some(row) = rows.next()? {
        let chunk_id: String = row.get(5)?;
        let entity = Node::try_from(row)?.to_string();
        let entities = out.entry(chunk_id).or_default();
        if entities.len() < per_chunk && !entities.contains(&entity) {
            entities.push(entity);
        }
    }
    Ok(out)
}

/// How many nodes carry any of `class_ids`, and the first few labels.
/// Counting is the one graph question traversal cannot answer: user SQL
/// may not read `_quack_` tables, and a class listing stops at
/// `max_nodes`.
///
/// # Errors
///
/// Returns an error if a query fails.
pub fn class_census(
    db: &WorkspaceDb,
    class_ids: &[String],
    samples: u32,
) -> Result<(u64, Vec<String>)> {
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
    Ok((total, labels))
}

/// How many nodes carry any of `class_ids`.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn class_count(db: &WorkspaceDb, class_ids: &[String]) -> Result<u64> {
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
pub fn node(db: &WorkspaceDb, id: &str) -> Result<Option<Node>> {
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
pub fn nodes(db: &WorkspaceDb, ids: &[String]) -> Result<Vec<Node>> {
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
pub fn edges(db: &WorkspaceDb, node_ids: &[String], scope: EdgeScope) -> Result<Vec<Edge>> {
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
    pub(crate) fn new(ids: &[String]) -> Self {
        let quoted: Vec<String> = ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "''")))
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
pub fn provenance_of(db: &WorkspaceDb, subject_ids: &[String]) -> Result<Vec<Provenance>> {
    if subject_ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut stmt = db.connection().prepare(
        "SELECT subject_id, document_id, chunk_id, table_name, row_key, confidence \
         FROM _quack_provenance WHERE list_contains(?::VARCHAR[], subject_id) \
         ORDER BY subject_id, chunk_id, table_name, row_key",
    )?;
    let mut rows = stmt.query(duckdb::params![IdList::new(subject_ids)])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(Provenance {
            subject_id: row.get(0)?,
            origin: Origin::from_columns(row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?),
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
    let missing_tables = match ontology_store::current(db)? {
        Some(ontology) => {
            let tables = db.list_tables()?;
            ontology
                .mappings
                .iter()
                .map(|m| m.table.clone())
                .filter(|t| !tables.contains(t))
                .collect()
        }
        None => Vec::new(),
    };
    Ok(GraphStatus {
        nodes: u64::try_from(nodes).unwrap_or(0),
        edges: u64::try_from(edges).unwrap_or(0),
        provisional_nodes: u64::try_from(provisional_nodes).unwrap_or(0),
        built_with_version,
        ontology_version,
        stale: nodes > 0 && built_with_version < ontology_version,
        pending_merges: u64::try_from(pending_merges).unwrap_or(0),
        drift: drift(db)?,
        missing_tables,
    })
}

/// Remove every node, edge, provenance row, merge proposal, and the drift.
///
/// # Errors
///
/// Returns an error if a delete fails.
pub fn clear(db: &WorkspaceDb) -> Result<()> {
    let conn = db.connection();
    conn.execute_batch(
        "DELETE FROM _quack_provenance; DELETE FROM _quack_graph_edges; \
         DELETE FROM _quack_graph_nodes; DELETE FROM _quack_graph_merges; \
         DELETE FROM _quack_graph_extracted;",
    )?;
    db.set_meta(MetaKey::GraphDrift, "{}")?;
    db.delete_meta(MetaKey::GraphBuiltWithOntologyVersion)
}

/// Note that a chunk was extracted under an ontology version, with what
/// it yielded, so incremental runs skip it (issue #60).
///
/// # Errors
///
/// Returns an error if the write fails.
pub fn record_extracted(
    db: &WorkspaceDb,
    chunk_id: &str,
    ontology_version: OntologyVersion,
    (nodes, edges): (u32, u32),
) -> Result<()> {
    db.connection().execute(
        "INSERT OR REPLACE INTO _quack_graph_extracted (chunk_id, ontology_version, nodes, edges) \
         VALUES (?, ?, ?, ?)",
        duckdb::params![chunk_id, ontology_version, nodes, edges],
    )?;
    Ok(())
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

/// What revalidation removed.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Revalidation {
    pub dropped_nodes: u64,
    pub dropped_edges: u64,
    pub version: OntologyVersion,
}

/// Bring a stale graph in line with the current ontology without a model
/// call: nodes whose class no longer exists go, and edges whose relation
/// is gone or whose endpoints no longer fit its domain and range go. What
/// goes is chosen in SQL; only the distinct (relation, source class,
/// target class) combinations the edges use are read, to check each
/// against the ontology once, so the work does not grow with the graph in
/// memory.
///
/// # Errors
///
/// Returns an error when there is no ontology or a write fails.
pub fn revalidate(db: &WorkspaceDb) -> Result<Revalidation> {
    let ontology = ontology_store::current(db)?
        .ok_or_else(|| Error::Ontology(String::from("no ontology to validate against")))?;
    let version = ontology.saved_version()?;
    let mut classes: Vec<String> = ontology.classes.iter().map(|c| c.id.clone()).collect();
    classes.push(String::from(ontology::ROOT_CLASS));
    let dropped_nodes = delete_nodes_outside(db, &IdList::new(&classes))?;

    let conn = db.connection();
    let mut stmt = conn.prepare(
        "SELECT DISTINCT e.relation_id, s.class_id, t.class_id FROM _quack_graph_edges e \
         JOIN _quack_graph_nodes s ON s.id = e.source_node_id \
         JOIN _quack_graph_nodes t ON t.id = e.target_node_id",
    )?;
    let combinations = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<duckdb::Result<Vec<(String, String, String)>>>()?;
    drop(stmt);
    let mut dropped_edges: u64 = 0;
    for (relation, source, target) in combinations {
        if !ontology.allows_edge(&relation, &source, &target) {
            let set = EdgeSet::Combination {
                relation,
                source,
                target,
            };
            dropped_edges = dropped_edges.saturating_add(set.delete(db)?);
        }
    }
    // Edges whose endpoint vanished with a dropped node are gone already;
    // any left dangling (orphaned by an older bug) go too.
    dropped_edges = dropped_edges.saturating_add(EdgeSet::Dangling.delete(db)?);
    set_built_with(db, version)?;
    Ok(Revalidation {
        dropped_nodes,
        dropped_edges,
        version,
    })
}

/// Delete every node whose class is not in `classes`, with its edges,
/// their provenance, its own, and its merge proposals; returns how many
/// nodes went.
fn delete_nodes_outside(db: &WorkspaceDb, classes: &IdList) -> Result<u64> {
    const NODES: &str =
        "SELECT id FROM _quack_graph_nodes WHERE NOT list_contains(?::VARCHAR[], class_id)";
    let conn = db.connection();
    let count: u64 = conn.query_row(
        &format!("SELECT count(*) FROM ({NODES})"),
        duckdb::params![classes],
        |row| row.get(0),
    )?;
    if count == 0 {
        return Ok(0);
    }
    conn.execute(
        &format!(
            "DELETE FROM _quack_provenance WHERE subject_id IN (SELECT id FROM _quack_graph_edges \
             WHERE source_node_id IN ({NODES}) OR target_node_id IN ({NODES}))"
        ),
        duckdb::params![classes, classes],
    )?;
    conn.execute(
        &format!(
            "DELETE FROM _quack_graph_edges \
             WHERE source_node_id IN ({NODES}) OR target_node_id IN ({NODES})"
        ),
        duckdb::params![classes, classes],
    )?;
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
        "DELETE FROM _quack_graph_nodes WHERE NOT list_contains(?::VARCHAR[], class_id)",
        duckdb::params![classes],
    )?;
    Ok(count)
}

/// Edges revalidation drops, as a query over the stored graph.
enum EdgeSet {
    /// Every edge of `relation` from a `source` node to a `target` node.
    Combination {
        relation: String,
        source: String,
        target: String,
    },
    /// Edges with an end that no longer exists.
    Dangling,
}

impl EdgeSet {
    /// The `SELECT id` of the set, and its parameters.
    fn query(&self) -> (&'static str, Vec<&dyn duckdb::ToSql>) {
        match self {
            Self::Combination {
                relation,
                source,
                target,
            } => (
                "SELECT e.id FROM _quack_graph_edges e \
                 JOIN _quack_graph_nodes s ON s.id = e.source_node_id \
                 JOIN _quack_graph_nodes t ON t.id = e.target_node_id \
                 WHERE e.relation_id = ? AND s.class_id = ? AND t.class_id = ?",
                vec![relation, source, target],
            ),
            Self::Dangling => (
                "SELECT e.id FROM _quack_graph_edges e \
                 WHERE NOT EXISTS (SELECT 1 FROM _quack_graph_nodes n WHERE n.id = e.source_node_id) \
                 OR NOT EXISTS (SELECT 1 FROM _quack_graph_nodes n WHERE n.id = e.target_node_id)",
                Vec::new(),
            ),
        }
    }

    /// Delete the set's edges and their provenance; returns how many edges
    /// went.
    fn delete(&self, db: &WorkspaceDb) -> Result<u64> {
        let (edges, params) = self.query();
        let conn = db.connection();
        let count: u64 = conn.query_row(
            &format!("SELECT count(*) FROM ({edges})"),
            params.as_slice(),
            |row| row.get(0),
        )?;
        if count == 0 {
            return Ok(0);
        }
        conn.execute(
            &format!("DELETE FROM _quack_provenance WHERE subject_id IN ({edges})"),
            params.as_slice(),
        )?;
        conn.execute(
            &format!("DELETE FROM _quack_graph_edges WHERE id IN ({edges})"),
            params.as_slice(),
        )?;
        Ok(count)
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

/// Nodes whose label embedding is within `max_distance` (cosine) of
/// `query`, nearest first, optionally within one class.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn nearest_nodes(
    db: &WorkspaceDb,
    query: &[f32],
    class_id: Option<&str>,
    limit: u32,
) -> Result<Vec<(Node, f64)>> {
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
    let literal = embedding_literal(query);
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
        let node = Node::try_from(row)?;
        let distance: f64 = row.get(5)?;
        out.push((node, distance));
    }
    Ok(out)
}
