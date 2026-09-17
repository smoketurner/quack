//! Traversal in plain SQL with bound parameters: neighborhood by recursive
//! CTE, shortest path by breadth-first search one hop per query, class
//! listing with subclass expansion. Entry points resolve by exact
//! normalized label, then by label-embedding similarity, then by class.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::store::{self, id_list};
use super::{GraphOptions, GraphResult, Node};
use crate::error::Result;
use crate::ontology::Ontology;
use crate::storage::workspace::WorkspaceDb;

/// How far an embedding match may be from the query to count as the entity.
const ENTRY_MAX_DISTANCE: f64 = 0.25;

/// Nodes matching `entity`: exact normalized label (any class, or one
/// class), else the nearest label embeddings within a distance, else
/// nothing.
///
/// # Errors
///
/// Returns an error if a query fails.
pub fn resolve_entry(
    db: &WorkspaceDb,
    entity: &str,
    class_id: Option<&str>,
    query_embedding: Option<&[f32]>,
) -> Result<Vec<Node>> {
    let normalized = super::normalize_label(entity);
    if normalized.is_empty() {
        return Ok(Vec::new());
    }
    let mut stmt = db.connection().prepare(
        "SELECT id, label, class_id, CAST(properties AS VARCHAR), provisional FROM _quack_graph_nodes \
         WHERE (normalized_label = ? OR list_contains(CAST(json_extract(properties, '$.aliases') AS VARCHAR[]), ?)) \
         AND (? IS NULL OR class_id = ?) ORDER BY label",
    )?;
    let mut rows = stmt.query(duckdb::params![
        normalized,
        entity.trim(),
        class_id,
        class_id
    ])?;
    let mut exact = Vec::new();
    while let Some(row) = rows.next()? {
        exact.push(store::node_from_row(row)?);
    }
    drop(rows);
    drop(stmt);
    if !exact.is_empty() {
        return Ok(exact);
    }
    if let Some(query) = query_embedding {
        let nearest = store::nearest_nodes(db, query, class_id, 3)?;
        let best: Vec<Node> = nearest
            .into_iter()
            .filter(|(_, d)| *d <= ENTRY_MAX_DISTANCE)
            .map(|(n, _)| n)
            .collect();
        if !best.is_empty() {
            return Ok(best);
        }
    }
    Ok(Vec::new())
}

/// The nodes within `hops` of `roots` (optionally along one relation),
/// with the edges among them and their provenance.
///
/// # Errors
///
/// Returns an error if a query fails.
pub fn neighborhood(
    db: &WorkspaceDb,
    roots: &[Node],
    hops: u32,
    relation: Option<&str>,
    options: &GraphOptions,
) -> Result<GraphResult> {
    if roots.is_empty() {
        return Ok(GraphResult::default());
    }
    let depth = hops.min(options.max_traversal_depth);
    let root_ids: Vec<String> = roots.iter().map(|n| n.id.clone()).collect();
    // Breadth-first, one query per frontier, never more than `max_nodes`
    // visited: a recursive CTE would enumerate every simple path out of a
    // hub before its LIMIT applied (issue #48).
    let max_visited = usize::try_from(options.max_nodes).unwrap_or(usize::MAX);
    let mut ids: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for id in &root_ids {
        if seen.insert(id.clone()) && ids.len() < max_visited {
            ids.push(id.clone());
        }
    }
    let mut frontier: Vec<String> = ids.clone();
    for _ in 0..depth {
        if frontier.is_empty() || ids.len() >= max_visited {
            break;
        }
        let mut next: Vec<String> = Vec::new();
        let mut edges = store::edges_touching(db, &frontier)?;
        if let Some(relation) = relation {
            edges.retain(|e| e.relation_id == relation);
        }
        for edge in edges {
            let there = if frontier.contains(&edge.source_node_id) {
                edge.target_node_id
            } else {
                edge.source_node_id
            };
            if !seen.insert(there.clone()) {
                continue;
            }
            ids.push(there.clone());
            next.push(there);
            if ids.len() >= max_visited {
                break;
            }
        }
        frontier = next;
    }
    let mut result = collect(db, &ids)?;
    if let Some(relation) = relation {
        result.edges.retain(|e| e.relation_id == relation);
    }
    result.roots = root_ids;
    Ok(result)
}

/// The shortest path between two nodes within `max_hops`, as the nodes and
/// edges along it; empty when none exists. Breadth-first, one query per
/// frontier, bounded by `max_nodes` visited.
///
/// # Errors
///
/// Returns an error if a query fails.
pub fn path(
    db: &WorkspaceDb,
    from: &Node,
    to: &Node,
    max_hops: u32,
    options: &GraphOptions,
) -> Result<GraphResult> {
    if from.id == to.id {
        let mut result = collect(db, std::slice::from_ref(&from.id))?;
        result.roots = vec![from.id.clone()];
        return Ok(result);
    }
    // A path spans two neighbourhoods, so it may run twice as deep.
    let limit = max_hops
        .min(options.max_traversal_depth.saturating_mul(2))
        .max(1);
    // node id -> (previous node id, edge id)
    let mut parent: BTreeMap<String, (String, String)> = BTreeMap::new();
    let mut seen: BTreeSet<String> = BTreeSet::from([from.id.clone()]);
    let mut frontier: VecDeque<String> = VecDeque::from([from.id.clone()]);
    let mut found = false;
    let max_visited = usize::try_from(options.max_nodes).unwrap_or(usize::MAX);
    for _ in 0..limit {
        if frontier.is_empty() || found {
            break;
        }
        let current: Vec<String> = frontier.drain(..).collect();
        let edges = store::edges_touching(db, &current)?;
        for edge in edges {
            let (here, there) = if current.contains(&edge.source_node_id) {
                (edge.source_node_id.clone(), edge.target_node_id.clone())
            } else {
                (edge.target_node_id.clone(), edge.source_node_id.clone())
            };
            if seen.contains(&there) {
                continue;
            }
            seen.insert(there.clone());
            parent.insert(there.clone(), (here, edge.id.clone()));
            if there == to.id {
                found = true;
                break;
            }
            if seen.len() >= max_visited {
                break;
            }
            frontier.push_back(there);
        }
    }
    if !found {
        return Ok(GraphResult::default());
    }
    let mut node_ids = vec![to.id.clone()];
    let mut edge_ids = Vec::new();
    let mut cursor = to.id.clone();
    while let Some((prev, edge)) = parent.get(&cursor) {
        edge_ids.push(edge.clone());
        node_ids.push(prev.clone());
        cursor = prev.clone();
    }
    node_ids.reverse();
    edge_ids.reverse();
    let nodes = store::nodes(db, &node_ids)?;
    let all_edges = store::edges_among(db, &node_ids)?;
    let edges: Vec<_> = all_edges
        .into_iter()
        .filter(|e| edge_ids.contains(&e.id))
        .collect();
    let subjects: Vec<String> = node_ids.iter().cloned().chain(edge_ids).collect();
    let provenance = store::provenance_of(db, &subjects)?;
    Ok(GraphResult {
        nodes,
        edges,
        provenance,
        roots: vec![from.id.clone(), to.id.clone()],
    })
}

/// Nodes of a class and its subclasses, with the edges among them.
///
/// # Errors
///
/// Returns an error if a query fails.
pub fn by_class(
    db: &WorkspaceDb,
    ontology: Option<&Ontology>,
    class_id: &str,
    limit: u32,
    options: &GraphOptions,
) -> Result<GraphResult> {
    let mut classes = vec![class_id.to_owned()];
    if let Some(ontology) = ontology {
        for class in &ontology.classes {
            if class.id != class_id && ontology.is_subclass_of(&class.id, class_id) {
                classes.push(class.id.clone());
            }
        }
    }
    let mut stmt = db.connection().prepare(
        "SELECT id FROM _quack_graph_nodes WHERE list_contains(?::VARCHAR[], class_id) ORDER BY label LIMIT ?",
    )?;
    let cap = i64::from(limit.min(options.max_nodes));
    let mut rows = stmt.query(duckdb::params![id_list(&classes), cap])?;
    let mut ids = Vec::new();
    while let Some(row) = rows.next()? {
        ids.push(row.get::<_, String>(0)?);
    }
    drop(rows);
    drop(stmt);
    collect(db, &ids)
}

/// Nodes by id with the edges among them and everything's provenance.
fn collect(db: &WorkspaceDb, ids: &[String]) -> Result<GraphResult> {
    let nodes = store::nodes(db, ids)?;
    let edges = store::edges_among(db, ids)?;
    let subjects: Vec<String> = nodes
        .iter()
        .map(|n| n.id.clone())
        .chain(edges.iter().map(|e| e.id.clone()))
        .collect();
    let provenance = store::provenance_of(db, &subjects)?;
    Ok(GraphResult {
        nodes,
        edges,
        provenance,
        roots: Vec::new(),
    })
}

/// A depth-first text rendering: each root, then its neighbours indented
/// with the relation, for the terminal and `quack graph`.
#[must_use]
pub fn render_tree(result: &GraphResult) -> String {
    if result.nodes.is_empty() {
        return String::from("No matching entities.");
    }
    let by_id: BTreeMap<&str, &Node> = result.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    let mut out = String::new();
    let mut visited: BTreeSet<&str> = BTreeSet::new();
    let roots: Vec<&str> = if result.roots.is_empty() {
        result.nodes.iter().map(|n| n.id.as_str()).collect()
    } else {
        result.roots.iter().map(String::as_str).collect()
    };
    for root in roots {
        let Some(node) = by_id.get(root) else {
            continue;
        };
        if visited.contains(root) {
            continue;
        }
        walk(result, &by_id, node, 0, &mut visited, &mut out);
    }
    for node in &result.nodes {
        if !visited.contains(node.id.as_str()) {
            walk(result, &by_id, node, 0, &mut visited, &mut out);
        }
    }
    out.push_str(&result.nodes.len().to_string());
    out.push_str(" nodes, ");
    out.push_str(&result.edges.len().to_string());
    out.push_str(" edges, ");
    out.push_str(&result.provenance.len().to_string());
    out.push_str(" sources");
    if result.nodes.iter().any(|n| n.provisional) {
        out.push_str(" (provisional: built from an unreviewed ontology)");
    }
    out.push('\n');
    out
}

fn walk<'a>(
    result: &'a GraphResult,
    by_id: &BTreeMap<&'a str, &'a Node>,
    node: &'a Node,
    depth: usize,
    visited: &mut BTreeSet<&'a str>,
    out: &mut String,
) {
    visited.insert(node.id.as_str());
    let indent = "  ".repeat(depth);
    out.push_str(&indent);
    out.push_str(&node.label);
    out.push_str(" (");
    out.push_str(&node.class_id);
    out.push_str(")\n");
    for edge in &result.edges {
        let (other, arrow) = if edge.source_node_id == node.id {
            (edge.target_node_id.as_str(), "->")
        } else if edge.target_node_id == node.id {
            (edge.source_node_id.as_str(), "<-")
        } else {
            continue;
        };
        let Some(next) = by_id.get(other) else {
            continue;
        };
        if visited.contains(other) {
            out.push_str(&indent);
            out.push_str("  ");
            out.push_str(arrow);
            out.push(' ');
            out.push_str(&edge.relation_id);
            out.push(' ');
            out.push_str(&next.label);
            out.push('\n');
            continue;
        }
        out.push_str(&indent);
        out.push_str("  ");
        out.push_str(arrow);
        out.push(' ');
        out.push_str(&edge.relation_id);
        out.push('\n');
        walk(result, by_id, next, depth.saturating_add(2), visited, out);
    }
}
