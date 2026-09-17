//! Deterministic extraction from mapped tables: every row of a mapped
//! table is a node of the mapping's class keyed by its key column, mapped
//! columns become properties, and each foreign-key-like column becomes an
//! edge to a node of the target class. Provenance is the table and row
//! key; re-running is idempotent.

use std::collections::{BTreeMap, BTreeSet};

use super::normalize_label;
use crate::error::{Error, Result};
use crate::ontology::{Mapping, Ontology};
use crate::storage::workspace::{WorkspaceDb, quote_ident};

/// What table extraction did for one mapping.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct MappingSummary {
    pub table: String,
    pub rows: u32,
    pub nodes: u32,
    pub edges: u32,
    /// Set when the mapping was skipped: the table is no longer in the
    /// workspace (a deleted document), so nothing was extracted from it.
    pub skipped: Option<String>,
}

impl MappingSummary {
    /// Add a batch's counts to a running total.
    pub fn absorb(&mut self, batch: &Self) {
        self.rows = self.rows.saturating_add(batch.rows);
        self.nodes = self.nodes.saturating_add(batch.nodes);
        self.edges = self.edges.saturating_add(batch.edges);
        if batch.skipped.is_some() {
            self.skipped.clone_from(&batch.skipped);
        }
    }
}

/// Rows one batch reads and writes; the caller may drop the workspace
/// lock between batches.
pub const BATCH_ROWS: u32 = 5_000;

/// Build nodes and edges from every mapping in the ontology, batch by
/// batch under the caller's lock. A mapping whose table is gone is
/// skipped and reported, not fatal.
///
/// # Errors
///
/// Returns an error when a mapped column is missing or a write fails.
pub fn extract(
    db: &WorkspaceDb,
    ontology: &Ontology,
    provisional: bool,
) -> Result<Vec<MappingSummary>> {
    let mut out = Vec::with_capacity(ontology.mappings.len());
    for mapping in &ontology.mappings {
        let mut total = MappingSummary {
            table: mapping.table.clone(),
            ..MappingSummary::default()
        };
        let mut offset = 0;
        loop {
            let (batch, more) = extract_batch(db, mapping, provisional, offset)?;
            total.absorb(&batch);
            if !more {
                break;
            }
            offset = offset.saturating_add(u64::from(BATCH_ROWS));
        }
        out.push(total);
    }
    Ok(out)
}

/// One batch of a mapping: `BATCH_ROWS` rows from `offset` in key order,
/// written in one transaction under the statement timeout. Returns what
/// the batch did and whether another batch follows, so a caller holding
/// a shared lock can release it in between (issue #48).
///
/// # Errors
///
/// Returns an error when a mapped column is missing, the batch runs past
/// the query timeout, or a write fails.
pub fn extract_batch(
    db: &WorkspaceDb,
    mapping: &Mapping,
    provisional: bool,
    offset: u64,
) -> Result<(MappingSummary, bool)> {
    if offset == 0 && !db.list_tables()?.iter().any(|t| t == &mapping.table) {
        tracing::warn!(table = %mapping.table, "mapped table is not in the workspace; skipping");
        return Ok((
            MappingSummary {
                table: mapping.table.clone(),
                skipped: Some(String::from("the table is not in the workspace")),
                ..MappingSummary::default()
            },
            false,
        ));
    }
    db.under_timeout(|db| {
        let tx = db.connection().unchecked_transaction()?;
        let outcome = extract_rows(db, mapping, provisional, offset)?;
        tx.commit()?;
        Ok(outcome)
    })
}

/// What identifies a node inside a batch: the exact-merge key of the
/// graph, `(normalized_label, class_id)`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct NodeKey {
    normalized: String,
    class_id: String,
}

/// A node the batch will insert or fill in, with its id minted here.
#[derive(Debug)]
struct StagedNode {
    id: String,
    label: String,
    properties: serde_json::Map<String, serde_json::Value>,
}

/// What identifies an edge: its triple.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct EdgeKey {
    source: NodeKey,
    target: NodeKey,
    relation: String,
}

/// A row's claim on a node or an edge, for `_quack_provenance`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RowSource<K> {
    subject: K,
    row_key: String,
}

/// The batch, staged in Rust: every node, edge, and provenance row the
/// rows imply, deduplicated, before one statement per kind writes them.
#[derive(Default)]
struct Staged {
    nodes: BTreeMap<NodeKey, StagedNode>,
    node_sources: BTreeSet<RowSource<NodeKey>>,
    /// Edge -> id
    edges: BTreeMap<EdgeKey, String>,
    edge_sources: BTreeSet<RowSource<EdgeKey>>,
    summary: MappingSummary,
}

impl Staged {
    /// Stage a node, or fill property gaps on one already staged.
    /// Returns its key, or `None` for a label that normalizes to nothing.
    fn node(
        &mut self,
        label: &str,
        class_id: &str,
        properties: serde_json::Map<String, serde_json::Value>,
    ) -> Option<NodeKey> {
        let normalized = normalize_label(label);
        if normalized.is_empty() {
            return None;
        }
        let key = NodeKey {
            normalized,
            class_id: class_id.to_owned(),
        };
        let entry = self.nodes.entry(key.clone()).or_insert_with(|| StagedNode {
            id: uuid::Uuid::now_v7().to_string(),
            label: label.trim().to_owned(),
            properties: serde_json::Map::new(),
        });
        for (k, v) in properties {
            entry.properties.entry(k).or_insert(v);
        }
        Some(key)
    }

    fn claim_node(&mut self, node: &NodeKey, row_key: &str) {
        self.node_sources.insert(RowSource {
            subject: node.clone(),
            row_key: row_key.to_owned(),
        });
    }

    /// Stage an edge (one id per triple) and the row's claim on it.
    fn edge(&mut self, source: &NodeKey, target: &NodeKey, relation: &str, row_key: &str) {
        let edge = EdgeKey {
            source: source.clone(),
            target: target.clone(),
            relation: relation.to_owned(),
        };
        self.edges
            .entry(edge.clone())
            .or_insert_with(|| uuid::Uuid::now_v7().to_string());
        self.edge_sources.insert(RowSource {
            subject: edge,
            row_key: row_key.to_owned(),
        });
    }
}

/// Stage the batch's rows in Rust, then write them with one statement
/// per kind: `DuckDB` is a columnar engine, and a statement per row (the
/// previous shape) cost tens of milliseconds a row on a large table.
fn extract_rows(
    db: &WorkspaceDb,
    mapping: &Mapping,
    provisional: bool,
    offset: u64,
) -> Result<(MappingSummary, bool)> {
    let mut columns: Vec<&str> = vec![mapping.key.as_str()];
    for column in mapping.properties.keys() {
        if !columns.contains(&column.as_str()) {
            columns.push(column);
        }
    }
    for relation in &mapping.relations {
        if !columns.contains(&relation.column.as_str()) {
            columns.push(&relation.column);
        }
    }
    let select = columns
        .iter()
        .map(|c| quote_ident(c))
        .collect::<Vec<_>>()
        .join(", ");
    // One row past the batch tells whether another batch follows.
    let sql = format!(
        "SELECT {select} FROM {} WHERE {key} IS NOT NULL ORDER BY {key} LIMIT {} OFFSET {offset}",
        quote_ident(&mapping.table),
        BATCH_ROWS.saturating_add(1),
        key = quote_ident(&mapping.key),
    );
    let results = db.execute_query(&sql)?;
    let more = results.rows.len() > BATCH_ROWS as usize;
    let index_of = |name: &str| results.columns.iter().position(|c| c == name);
    let key_idx = index_of(&mapping.key)
        .ok_or_else(|| Error::Ontology(format!("key column '{}' missing", mapping.key)))?;
    let mut staged = Staged {
        summary: MappingSummary {
            table: mapping.table.clone(),
            ..MappingSummary::default()
        },
        ..Staged::default()
    };
    for row in results.rows.iter().take(BATCH_ROWS as usize) {
        let Some(key) = row.get(key_idx).map(cell_text).filter(|k| !k.is_empty()) else {
            continue;
        };
        let mut properties = serde_json::Map::new();
        for (column, property) in &mapping.properties {
            if let Some(idx) = index_of(column)
                && let Some(value) = row.get(idx)
                && !value.is_null()
            {
                properties.insert(property.clone(), value.clone());
            }
        }
        let Some(node) = staged.node(&key, &mapping.class, properties) else {
            continue;
        };
        staged.summary.rows = staged.summary.rows.saturating_add(1);
        staged.summary.nodes = staged.summary.nodes.saturating_add(1);
        staged.claim_node(&node, &key);
        for relation in &mapping.relations {
            let Some(idx) = index_of(&relation.column) else {
                continue;
            };
            let Some(target_key) = row.get(idx).map(cell_text).filter(|k| !k.is_empty()) else {
                continue;
            };
            let Some(target) =
                staged.node(&target_key, &relation.target_class, serde_json::Map::new())
            else {
                continue;
            };
            staged.claim_node(&target, &key);
            staged.edge(&node, &target, &relation.relation, &key);
            staged.summary.edges = staged.summary.edges.saturating_add(1);
        }
    }
    write_staged(db, &mapping.table, provisional, &staged)?;
    Ok((staged.summary, more))
}

/// Write a staged batch: four scratch tables filled through appenders,
/// then one statement per kind. Nodes that exist keep their id and their
/// own property values (incoming keys only fill gaps); edges that exist
/// keep their id; provenance is `INSERT OR IGNORE`.
fn write_staged(db: &WorkspaceDb, table: &str, provisional: bool, staged: &Staged) -> Result<()> {
    fill_scratch_tables(db, staged)?;
    merge_scratch_tables(db, table, provisional)
}

fn fill_scratch_tables(db: &WorkspaceDb, staged: &Staged) -> Result<()> {
    let conn = db.connection();
    conn.execute_batch(
        "CREATE OR REPLACE TABLE _quack_tmp_graph_nodes (
            id TEXT, label TEXT, normalized_label TEXT, class_id TEXT, properties TEXT);
         CREATE OR REPLACE TABLE _quack_tmp_graph_node_prov (
            normalized_label TEXT, class_id TEXT, row_key TEXT);
         CREATE OR REPLACE TABLE _quack_tmp_graph_edges (
            id TEXT, source_norm TEXT, source_class TEXT, target_norm TEXT, target_class TEXT, relation_id TEXT);
         CREATE OR REPLACE TABLE _quack_tmp_graph_edge_prov (
            source_norm TEXT, source_class TEXT, target_norm TEXT, target_class TEXT, relation_id TEXT, row_key TEXT);",
    )?;
    let mut nodes = conn.appender("_quack_tmp_graph_nodes")?;
    for (key, node) in &staged.nodes {
        nodes.append_row(duckdb::params![
            node.id,
            node.label,
            key.normalized,
            key.class_id,
            serde_json::to_string(&node.properties)?
        ])?;
    }
    nodes.flush()?;
    let mut node_prov = conn.appender("_quack_tmp_graph_node_prov")?;
    for source in &staged.node_sources {
        node_prov.append_row(duckdb::params![
            source.subject.normalized,
            source.subject.class_id,
            source.row_key
        ])?;
    }
    node_prov.flush()?;
    let mut edges = conn.appender("_quack_tmp_graph_edges")?;
    for (edge, id) in &staged.edges {
        edges.append_row(duckdb::params![
            id,
            edge.source.normalized,
            edge.source.class_id,
            edge.target.normalized,
            edge.target.class_id,
            edge.relation
        ])?;
    }
    edges.flush()?;
    let mut edge_prov = conn.appender("_quack_tmp_graph_edge_prov")?;
    for source in &staged.edge_sources {
        let edge = &source.subject;
        edge_prov.append_row(duckdb::params![
            edge.source.normalized,
            edge.source.class_id,
            edge.target.normalized,
            edge.target.class_id,
            edge.relation,
            source.row_key
        ])?;
    }
    edge_prov.flush()?;
    Ok(())
}

fn merge_scratch_tables(db: &WorkspaceDb, table: &str, provisional: bool) -> Result<()> {
    let conn = db.connection();
    // Nodes: fill in missing property keys and clear the provisional flag
    // on the ones that exist, insert the rest.
    conn.execute(
        "UPDATE _quack_graph_nodes SET \
            properties = json_merge_patch(t.properties::JSON, coalesce(_quack_graph_nodes.properties, '{}'::JSON)), \
            provisional = _quack_graph_nodes.provisional AND ? \
         FROM _quack_tmp_graph_nodes t \
         WHERE t.normalized_label = _quack_graph_nodes.normalized_label AND t.class_id = _quack_graph_nodes.class_id \
           AND (json_merge_patch(t.properties::JSON, coalesce(_quack_graph_nodes.properties, '{}'::JSON))::VARCHAR \
                  <> coalesce(_quack_graph_nodes.properties, '{}'::JSON)::VARCHAR \
                OR (_quack_graph_nodes.provisional AND NOT ?))",
        duckdb::params![provisional, provisional],
    )?;
    conn.execute(
        "INSERT INTO _quack_graph_nodes (id, label, normalized_label, class_id, properties, provisional) \
         SELECT t.id, t.label, t.normalized_label, t.class_id, t.properties::JSON, ? \
         FROM _quack_tmp_graph_nodes t \
         WHERE NOT EXISTS (SELECT 1 FROM _quack_graph_nodes n \
                           WHERE n.normalized_label = t.normalized_label AND n.class_id = t.class_id)",
        duckdb::params![provisional],
    )?;
    conn.execute(
        "INSERT OR IGNORE INTO _quack_provenance (subject_id, document_id, chunk_id, table_name, row_key, confidence) \
         SELECT DISTINCT n.id, NULL, '', ?, p.row_key, 1.0 \
         FROM _quack_tmp_graph_node_prov p \
         JOIN _quack_graph_nodes n ON n.normalized_label = p.normalized_label AND n.class_id = p.class_id",
        duckdb::params![table],
    )?;
    // Edges: an existing triple keeps its id and loses the provisional
    // flag when this evidence is not provisional; the rest are inserted.
    conn.execute(
        "UPDATE _quack_graph_edges SET provisional = false \
         FROM _quack_tmp_graph_edges t, _quack_graph_nodes s, _quack_graph_nodes g \
         WHERE s.normalized_label = t.source_norm AND s.class_id = t.source_class \
           AND g.normalized_label = t.target_norm AND g.class_id = t.target_class \
           AND _quack_graph_edges.source_node_id = s.id AND _quack_graph_edges.target_node_id = g.id \
           AND _quack_graph_edges.relation_id = t.relation_id \
           AND _quack_graph_edges.provisional AND NOT ?",
        duckdb::params![provisional],
    )?;
    conn.execute(
        "INSERT INTO _quack_graph_edges (id, source_node_id, target_node_id, relation_id, properties, provisional) \
         SELECT t.id, s.id, g.id, t.relation_id, '{}'::JSON, ? \
         FROM _quack_tmp_graph_edges t \
         JOIN _quack_graph_nodes s ON s.normalized_label = t.source_norm AND s.class_id = t.source_class \
         JOIN _quack_graph_nodes g ON g.normalized_label = t.target_norm AND g.class_id = t.target_class \
         WHERE NOT EXISTS (SELECT 1 FROM _quack_graph_edges e \
                           WHERE e.source_node_id = s.id AND e.target_node_id = g.id AND e.relation_id = t.relation_id)",
        duckdb::params![provisional],
    )?;
    conn.execute(
        "INSERT OR IGNORE INTO _quack_provenance (subject_id, document_id, chunk_id, table_name, row_key, confidence) \
         SELECT DISTINCT e.id, NULL, '', ?, p.row_key, 1.0 \
         FROM _quack_tmp_graph_edge_prov p \
         JOIN _quack_graph_nodes s ON s.normalized_label = p.source_norm AND s.class_id = p.source_class \
         JOIN _quack_graph_nodes g ON g.normalized_label = p.target_norm AND g.class_id = p.target_class \
         JOIN _quack_graph_edges e ON e.source_node_id = s.id AND e.target_node_id = g.id AND e.relation_id = p.relation_id",
        duckdb::params![table],
    )?;
    conn.execute_batch(
        "DROP TABLE _quack_tmp_graph_nodes; DROP TABLE _quack_tmp_graph_node_prov; \
         DROP TABLE _quack_tmp_graph_edges; DROP TABLE _quack_tmp_graph_edge_prov;",
    )?;
    Ok(())
}

fn cell_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.trim().to_owned(),
        other => other.to_string(),
    }
}
