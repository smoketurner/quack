//! Persistence for nodes, edges, provenance, and the graph's bookkeeping in
//! `_quack_meta`: the ontology version it was built with, and drift.

use std::collections::{BTreeMap, BTreeSet};

use super::resolve::MergeStatus;
use super::{Drift, Edge, GraphStatus, Node, Provenance, normalize_label};
use crate::error::{Error, Result};
use crate::ontology::{self, Ontology, store as ontology_store};
use crate::storage::workspace::{MetaKey, WorkspaceDb, embedding_literal};

/// A node to store: merged into an existing one with the same normalized
/// label and class, else inserted.
#[derive(Debug, Clone)]
pub struct NewNode {
    pub label: String,
    pub class_id: String,
    pub properties: serde_json::Value,
    pub provisional: bool,
}

/// A source to attach to a node or edge.
#[derive(Debug, Clone, Default)]
pub struct Source {
    pub document_id: Option<String>,
    pub chunk_id: Option<String>,
    pub table_name: Option<String>,
    pub row_key: Option<String>,
    pub confidence: f64,
}

impl Source {
    #[must_use]
    pub fn chunk(document_id: &str, chunk_id: &str, confidence: f64) -> Self {
        Self {
            document_id: Some(document_id.to_owned()),
            chunk_id: Some(chunk_id.to_owned()),
            table_name: None,
            row_key: None,
            confidence,
        }
    }

    #[must_use]
    pub fn row(table_name: &str, row_key: &str) -> Self {
        Self {
            document_id: None,
            chunk_id: None,
            table_name: Some(table_name.to_owned()),
            row_key: Some(row_key.to_owned()),
            confidence: 1.0,
        }
    }
}

/// A lookup's row, `None` when there is none, and any other failure as
/// the error it is (issue #62: `.ok()` turned a broken query into "not
/// found" and then a UNIQUE violation).
fn optional<T>(result: duckdb::Result<T>) -> Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
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
    let normalized = normalize_label(&node.label);
    if normalized.is_empty() {
        return Err(Error::Analysis(String::from("a node needs a label")));
    }
    let conn = db.connection();
    let existing: Option<(String, Option<String>, bool)> = optional(conn.query_row(
        "SELECT id, CAST(properties AS VARCHAR), provisional FROM _quack_graph_nodes \
         WHERE normalized_label = ? AND class_id = ?",
        duckdb::params![normalized, node.class_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ))?;
    if let Some((id, properties, provisional)) = existing {
        let mut merged: serde_json::Value = properties
            .and_then(|p| serde_json::from_str(&p).ok())
            .unwrap_or_else(|| serde_json::json!({}));
        if let (Some(target), Some(incoming)) =
            (merged.as_object_mut(), node.properties.as_object())
        {
            for (k, v) in incoming {
                target.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
        conn.execute(
            "UPDATE _quack_graph_nodes SET properties = ?, provisional = ? WHERE id = ?",
            duckdb::params![
                serde_json::to_string(&merged)?,
                provisional && node.provisional,
                id
            ],
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
            serde_json::to_string(&node.properties)?,
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
    properties: &serde_json::Value,
    provisional: bool,
) -> Result<String> {
    let conn = db.connection();
    let existing: Option<String> = optional(conn.query_row(
        "SELECT id FROM _quack_graph_edges \
         WHERE source_node_id = ? AND target_node_id = ? AND relation_id = ?",
        duckdb::params![source, target, relation_id],
        |r| r.get(0),
    ))?;
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
            serde_json::to_string(properties)?,
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
    db.connection().execute(
        "INSERT OR IGNORE INTO _quack_provenance (subject_id, document_id, chunk_id, table_name, row_key, confidence) \
         VALUES (?, ?, ?, ?, ?, ?)",
        duckdb::params![
            subject_id,
            source.document_id,
            source.chunk_id.clone().unwrap_or_default(),
            source.table_name.clone().unwrap_or_default(),
            source.row_key.clone().unwrap_or_default(),
            source.confidence
        ],
    )?;
    Ok(())
}

const NODE_COLUMNS: &str =
    "id, label, class_id, CAST(properties AS VARCHAR), provisional FROM _quack_graph_nodes";

pub(crate) fn node_from_row(row: &duckdb::Row<'_>) -> duckdb::Result<Node> {
    let properties: Option<String> = row.get(3)?;
    Ok(Node {
        id: row.get(0)?,
        label: row.get(1)?,
        class_id: row.get(2)?,
        properties: properties
            .and_then(|p| serde_json::from_str(&p).ok())
            .unwrap_or(serde_json::Value::Null),
        provisional: row.get(4)?,
    })
}

const EDGE_COLUMNS: &str = "id, source_node_id, target_node_id, relation_id, weight, \
     CAST(properties AS VARCHAR), provisional FROM _quack_graph_edges";

pub(crate) fn edge_from_row(row: &duckdb::Row<'_>) -> duckdb::Result<Edge> {
    let properties: Option<String> = row.get(5)?;
    Ok(Edge {
        id: row.get(0)?,
        source_node_id: row.get(1)?,
        target_node_id: row.get(2)?,
        relation_id: row.get(3)?,
        weight: row.get::<_, Option<f64>>(4)?.unwrap_or(1.0),
        properties: properties
            .and_then(|p| serde_json::from_str(&p).ok())
            .unwrap_or(serde_json::Value::Null),
        provisional: row.get(6)?,
    })
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
    let mut rows = stmt.query(duckdb::params![id_list(node_ids)])?;
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
        "SELECT p.chunk_id, n.label, n.class_id FROM _quack_provenance p \
         JOIN _quack_graph_nodes n ON n.id = p.subject_id \
         WHERE list_contains(?::VARCHAR[], p.chunk_id) ORDER BY p.chunk_id, n.label",
    )?;
    let mut rows = stmt.query(duckdb::params![id_list(chunk_ids)])?;
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    while let Some(row) = rows.next()? {
        let chunk_id: String = row.get(0)?;
        let entity = format!(
            "{} ({})",
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?
        );
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
    let total: u64 = db.connection().query_row(
        "SELECT count(*) FROM _quack_graph_nodes WHERE list_contains(?::VARCHAR[], class_id)",
        duckdb::params![id_list(class_ids)],
        |row| row.get(0),
    )?;
    let mut stmt = db.connection().prepare(
        "SELECT label FROM _quack_graph_nodes WHERE list_contains(?::VARCHAR[], class_id) \
         ORDER BY label LIMIT ?",
    )?;
    let mut rows = stmt.query(duckdb::params![id_list(class_ids), i64::from(samples)])?;
    let mut labels = Vec::new();
    while let Some(row) = rows.next()? {
        labels.push(row.get::<_, String>(0)?);
    }
    Ok((total, labels))
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
        Some(row) => Ok(Some(node_from_row(row)?)),
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

/// Edges whose both ends are in `node_ids`.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn edges_among(db: &WorkspaceDb, node_ids: &[String]) -> Result<Vec<Edge>> {
    if node_ids.is_empty() {
        return Ok(Vec::new());
    }
    let sql = format!(
        "SELECT {EDGE_COLUMNS} WHERE list_contains(?::VARCHAR[], source_node_id) \
         AND list_contains(?::VARCHAR[], target_node_id) ORDER BY relation_id, id"
    );
    let list = id_list(node_ids);
    let mut stmt = db.connection().prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![list, list])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(edge_from_row(row)?);
    }
    Ok(out)
}

/// Edges touching any of `node_ids`.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn edges_touching(db: &WorkspaceDb, node_ids: &[String]) -> Result<Vec<Edge>> {
    if node_ids.is_empty() {
        return Ok(Vec::new());
    }
    let sql = format!(
        "SELECT {EDGE_COLUMNS} WHERE list_contains(?::VARCHAR[], source_node_id) \
         OR list_contains(?::VARCHAR[], target_node_id) ORDER BY relation_id, id"
    );
    let list = id_list(node_ids);
    let mut stmt = db.connection().prepare(&sql)?;
    let mut rows = stmt.query(duckdb::params![list, list])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(edge_from_row(row)?);
    }
    Ok(out)
}

/// A `DuckDB` list literal of ids, bound as one `VARCHAR[]` parameter.
pub(crate) fn id_list(ids: &[String]) -> String {
    let quoted: Vec<String> = ids
        .iter()
        .map(|id| format!("'{}'", id.replace('\'', "''")))
        .collect();
    format!("[{}]", quoted.join(","))
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
    let mut rows = stmt.query(duckdb::params![id_list(subject_ids)])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let blank_to_none = |s: String| (!s.is_empty()).then_some(s);
        out.push(Provenance {
            subject_id: row.get(0)?,
            document_id: row.get(1)?,
            chunk_id: blank_to_none(row.get(2)?),
            table_name: blank_to_none(row.get(3)?),
            row_key: blank_to_none(row.get(4)?),
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
pub fn built_with(db: &WorkspaceDb) -> Result<u32> {
    Ok(db
        .meta(MetaKey::GraphBuiltWithOntologyVersion)?
        .and_then(|v| v.parse().ok())
        .unwrap_or(0))
}

/// Record the ontology version the graph now matches.
///
/// # Errors
///
/// Returns an error if the write fails.
pub fn set_built_with(db: &WorkspaceDb, version: u32) -> Result<()> {
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
    db.set_meta(MetaKey::GraphBuiltWithOntologyVersion, "0")
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
    ontology_version: u32,
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

/// Delete the given nodes with their edges and provenance.
///
/// # Errors
///
/// Returns an error if a delete fails.
pub fn delete_nodes(db: &WorkspaceDb, ids: &[String]) -> Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    let list = id_list(ids);
    let conn = db.connection();
    conn.execute(
        "DELETE FROM _quack_provenance WHERE subject_id IN (SELECT id FROM _quack_graph_edges \
         WHERE list_contains(?::VARCHAR[], source_node_id) OR list_contains(?::VARCHAR[], target_node_id))",
        duckdb::params![list, list],
    )?;
    conn.execute(
        "DELETE FROM _quack_graph_edges WHERE list_contains(?::VARCHAR[], source_node_id) \
         OR list_contains(?::VARCHAR[], target_node_id)",
        duckdb::params![list, list],
    )?;
    conn.execute(
        "DELETE FROM _quack_provenance WHERE list_contains(?::VARCHAR[], subject_id)",
        duckdb::params![list],
    )?;
    conn.execute(
        "DELETE FROM _quack_graph_merges WHERE list_contains(?::VARCHAR[], keep_node_id) \
         OR list_contains(?::VARCHAR[], drop_node_id)",
        duckdb::params![list, list],
    )?;
    conn.execute(
        "DELETE FROM _quack_graph_nodes WHERE list_contains(?::VARCHAR[], id)",
        duckdb::params![list],
    )?;
    Ok(())
}

/// Delete the given edges with their provenance.
///
/// # Errors
///
/// Returns an error if a delete fails.
pub fn delete_edges(db: &WorkspaceDb, ids: &[String]) -> Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    let list = id_list(ids);
    let conn = db.connection();
    conn.execute(
        "DELETE FROM _quack_provenance WHERE list_contains(?::VARCHAR[], subject_id)",
        duckdb::params![list],
    )?;
    conn.execute(
        "DELETE FROM _quack_graph_edges WHERE list_contains(?::VARCHAR[], id)",
        duckdb::params![list],
    )?;
    Ok(())
}

/// What revalidation removed.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct Revalidation {
    pub dropped_nodes: u64,
    pub dropped_edges: u64,
    pub version: u32,
}

/// Bring a stale graph in line with the current ontology without a model
/// call: nodes whose class no longer exists go, and edges whose relation
/// is gone or whose endpoints no longer fit its domain and range go.
///
/// # Errors
///
/// Returns an error when there is no ontology or a write fails.
pub fn revalidate(db: &WorkspaceDb) -> Result<Revalidation> {
    let ontology = ontology_store::current(db)?
        .ok_or_else(|| Error::Ontology(String::from("no ontology to validate against")))?;
    let classes: BTreeSet<&str> = ontology.classes.iter().map(|c| c.id.as_str()).collect();
    let conn = db.connection();
    let mut stmt = conn.prepare("SELECT id, class_id FROM _quack_graph_nodes")?;
    let mut rows = stmt.query([])?;
    let mut bad_nodes = Vec::new();
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let class: String = row.get(1)?;
        if class != ontology::ROOT_CLASS && !classes.contains(class.as_str()) {
            bad_nodes.push(id);
        }
    }
    drop(rows);
    drop(stmt);
    delete_nodes(db, &bad_nodes)?;

    let mut stmt = conn.prepare(
        "SELECT e.id, e.relation_id, s.class_id, t.class_id FROM _quack_graph_edges e \
         JOIN _quack_graph_nodes s ON s.id = e.source_node_id \
         JOIN _quack_graph_nodes t ON t.id = e.target_node_id",
    )?;
    let mut rows = stmt.query([])?;
    let mut bad_edges = Vec::new();
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let relation: String = row.get(1)?;
        let source_class: String = row.get(2)?;
        let target_class: String = row.get(3)?;
        if !edge_fits(&ontology, &relation, &source_class, &target_class) {
            bad_edges.push(id);
        }
    }
    drop(rows);
    drop(stmt);
    // Edges whose endpoint vanished with a dropped node are gone already;
    // any left dangling (orphaned by an older bug) go too.
    let mut stmt = conn.prepare(
        "SELECT e.id FROM _quack_graph_edges e \
         WHERE NOT EXISTS (SELECT 1 FROM _quack_graph_nodes n WHERE n.id = e.source_node_id) \
         OR NOT EXISTS (SELECT 1 FROM _quack_graph_nodes n WHERE n.id = e.target_node_id)",
    )?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        bad_edges.push(row.get(0)?);
    }
    drop(rows);
    drop(stmt);
    bad_edges.sort();
    bad_edges.dedup();
    delete_edges(db, &bad_edges)?;
    set_built_with(db, ontology.version)?;
    Ok(Revalidation {
        dropped_nodes: u64::try_from(bad_nodes.len()).unwrap_or(u64::MAX),
        dropped_edges: u64::try_from(bad_edges.len()).unwrap_or(u64::MAX),
        version: ontology.version,
    })
}

/// Whether an edge of `relation` between the two classes is valid under
/// the ontology: the relation exists (or is `mentions`) and each endpoint's
/// class is the domain or range or a descendant of it.
#[must_use]
pub fn edge_fits(
    ontology: &Ontology,
    relation: &str,
    source_class: &str,
    target_class: &str,
) -> bool {
    if relation == ontology::MENTIONS_RELATION {
        return true;
    }
    let Some(rel) = ontology.relation(relation) else {
        return false;
    };
    class_fits(ontology, source_class, &rel.domain)
        && class_fits(ontology, target_class, &rel.range)
}

fn class_fits(ontology: &Ontology, class: &str, wanted: &str) -> bool {
    wanted == ontology::ROOT_CLASS || class == wanted || ontology.is_subclass_of(class, wanted)
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
        out.push(node_from_row(row)?);
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
        vt = db.vector_type_public()
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
        let node = node_from_row(row)?;
        let distance: f64 = row.get(5)?;
        out.push((node, distance));
    }
    Ok(out)
}
