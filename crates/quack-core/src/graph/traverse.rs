//! Traversal in plain SQL with bound parameters: neighborhood by recursive
//! CTE, shortest path by breadth-first search one hop per query, class
//! listing with subclass expansion. Entry points resolve by exact
//! normalized label, then by label-embedding similarity, then by class.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use super::store::{self, EdgeScope, IdList};
use super::{GraphOptions, GraphResult, Node, NormalizedLabel, Properties};
use crate::error::Result;
use crate::ids::{EdgeId, NodeId};
use crate::ontology::Ontology;
use crate::storage::workspace::WorkspaceDb;

/// How many relations a walk follows from its entry point: at least one,
/// whatever was asked, and capped again by `[graph].max_traversal_depth`
/// when it runs. Every interface reads the caller's number through this, so
/// `--hops 0` and `hops: 0` mean the same everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(transparent)]
pub struct Hops(u32);

impl Hops {
    /// How far a neighborhood reaches when the caller does not say.
    pub const NEIGHBORHOOD: Self = Self(2);
    /// How long a path may be when the caller does not say.
    pub const PATH: Self = Self(4);

    #[must_use]
    pub const fn new(hops: u32) -> Self {
        if hops == 0 { Self(1) } else { Self(hops) }
    }

    /// The caller's number, or [`Self::NEIGHBORHOOD`].
    #[must_use]
    pub fn neighborhood(hops: Option<u32>) -> Self {
        hops.map_or(Self::NEIGHBORHOOD, Self::new)
    }

    /// The caller's number, or [`Self::PATH`].
    #[must_use]
    pub fn path(hops: Option<u32>) -> Self {
        hops.map_or(Self::PATH, Self::new)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for Hops {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// How far an embedding match may be from the query to count as the entity.
const ENTRY_MAX_DISTANCE: f64 = 0.25;

/// How many labels a lookup that matched nothing offers as alternatives.
const SUGGESTION_LIMIT: u32 = 5;

/// How alike two labels must read for one to be offered as the other's
/// correction: Jaro-Winkler, which `DuckDB` has built in, so no extension
/// is loaded. A single transposed or dropped letter scores well above
/// this; two unrelated names of the same length do not.
const SUGGESTION_SIMILARITY: f64 = 0.88;

/// How far a label embedding may be from the query and still be offered
/// as a suggestion. Looser than [`ENTRY_MAX_DISTANCE`], which decides
/// what *is* the entity, but not unbounded: in a small graph the nearest
/// node is whatever exists, and suggesting an unrelated label invites the
/// model to search again and answer about the wrong thing.
const SUGGESTION_MAX_DISTANCE: f64 = 0.5;

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
    let normalized = NormalizedLabel::new(entity);
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
        exact.push(Node::try_from(row)?);
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

/// Labels worth retrying when a lookup matched nothing: nodes whose
/// normalized label overlaps the text either way or is a near-miss of it
/// (Jaro-Winkler, so a one-letter typo still suggests the real label),
/// closest first, then label embeddings within a looser distance than an
/// entry point takes — `resolve_entry` has already rejected those as
/// matches, which does not make them useless as suggestions, though a
/// suggestion no closer than any other label would be. The embedding pass finds
/// nothing until `graph::resolve` has run with an embedding model, since
/// only that writes node embeddings; the similarity pass always works.
/// Rendered as `label (class)`.
///
/// # Errors
///
/// Returns an error if a query fails.
pub fn suggest_entities(
    db: &WorkspaceDb,
    entity: &str,
    class_id: Option<&str>,
    query_embedding: Option<&[f32]>,
) -> Result<Vec<String>> {
    let normalized = NormalizedLabel::new(entity);
    if normalized.is_empty() {
        return Ok(Vec::new());
    }
    let mut out: Vec<String> = Vec::new();
    let mut stmt = db.connection().prepare(
        "SELECT id, label, class_id, CAST(properties AS VARCHAR), provisional FROM _quack_graph_nodes \
         WHERE (contains(normalized_label, ?) OR contains(?, normalized_label) \
                OR jaro_winkler_similarity(normalized_label, ?) >= ?) \
         AND (? IS NULL OR class_id = ?) \
         ORDER BY jaro_winkler_similarity(normalized_label, ?) DESC, length(label), label LIMIT ?",
    )?;
    let mut rows = stmt.query(duckdb::params![
        normalized,
        normalized,
        normalized,
        SUGGESTION_SIMILARITY,
        class_id,
        class_id,
        normalized,
        i64::from(SUGGESTION_LIMIT)
    ])?;
    while let Some(row) = rows.next()? {
        out.push(Node::try_from(row)?.to_string());
    }
    drop(rows);
    drop(stmt);
    if let Some(query) = query_embedding {
        for (node, distance) in store::nearest_nodes(db, query, class_id, SUGGESTION_LIMIT)? {
            if distance > SUGGESTION_MAX_DISTANCE {
                continue;
            }
            let label = node.to_string();
            if !out.contains(&label) {
                out.push(label);
            }
        }
    }
    out.truncate(usize::try_from(SUGGESTION_LIMIT).unwrap_or(5));
    Ok(out)
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
    hops: Hops,
    relation: Option<&str>,
    options: &GraphOptions,
) -> Result<GraphResult> {
    if roots.is_empty() {
        return Ok(GraphResult::default());
    }
    let depth = hops.get().min(options.max_traversal_depth);
    let root_ids: Vec<NodeId> = roots.iter().map(|n| n.id.clone()).collect();
    // Breadth-first, one query per frontier, never more than `max_nodes`
    // visited: a recursive CTE would enumerate every simple path out of a
    // hub before its LIMIT applied (issue #48).
    let max_visited = usize::try_from(options.max_nodes).unwrap_or(usize::MAX);
    let mut ids: Vec<NodeId> = Vec::new();
    let mut seen: BTreeSet<NodeId> = BTreeSet::new();
    for id in &root_ids {
        if seen.insert(id.clone()) && ids.len() < max_visited {
            ids.push(id.clone());
        }
    }
    let mut frontier: Vec<NodeId> = ids.clone();
    for _ in 0..depth {
        if frontier.is_empty() || ids.len() >= max_visited {
            break;
        }
        let mut next: Vec<NodeId> = Vec::new();
        let mut edges = store::edges(db, &frontier, EdgeScope::Touching)?;
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
    // A walk that filled its budget stopped early; unlike a class listing
    // it cannot say how much it did not visit.
    result.truncated = ids.len() >= max_visited;
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
    max_hops: Hops,
    options: &GraphOptions,
) -> Result<GraphResult> {
    if from.id == to.id {
        let mut result = collect(db, std::slice::from_ref(&from.id))?;
        result.roots = vec![from.id.clone()];
        return Ok(result);
    }
    // A path spans two neighbourhoods, so it may run twice as deep.
    let limit = max_hops
        .get()
        .min(options.max_traversal_depth.saturating_mul(2))
        .max(1);
    // node id -> (previous node id, edge id)
    let mut parent: BTreeMap<NodeId, (NodeId, EdgeId)> = BTreeMap::new();
    let mut seen: BTreeSet<NodeId> = BTreeSet::from([from.id.clone()]);
    let mut frontier: VecDeque<NodeId> = VecDeque::from([from.id.clone()]);
    let mut found = false;
    let max_visited = usize::try_from(options.max_nodes).unwrap_or(usize::MAX);
    for _ in 0..limit {
        if frontier.is_empty() || found {
            break;
        }
        let current: Vec<NodeId> = frontier.drain(..).collect();
        let edges = store::edges(db, &current, EdgeScope::Touching)?;
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
    let all_edges = store::edges(db, &node_ids, EdgeScope::Among)?;
    let edges: Vec<_> = all_edges
        .into_iter()
        .filter(|e| edge_ids.contains(&e.id))
        .collect();
    let subjects: Vec<String> = node_ids
        .iter()
        .map(NodeId::to_string)
        .chain(edge_ids.iter().map(EdgeId::to_string))
        .collect();
    let provenance = store::provenance_of(db, &subjects)?;
    Ok(GraphResult {
        nodes,
        edges,
        provenance,
        roots: vec![from.id.clone(), to.id.clone()],
        ..GraphResult::default()
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
    let classes = ontology.map_or_else(
        || vec![class_id.to_owned()],
        |o| o.class_and_descendants(class_id),
    );
    // Counted before the cap applies: a listing that silently stopped at
    // `max_nodes` reads as the whole population of the class.
    let total = store::class_count(db, &classes)?;
    let mut stmt = db.connection().prepare(
        "SELECT id FROM _quack_graph_nodes WHERE list_contains(?::VARCHAR[], class_id) ORDER BY label LIMIT ?",
    )?;
    let cap = i64::from(limit.min(options.max_nodes));
    let mut rows = stmt.query(duckdb::params![IdList::new(&classes), cap])?;
    let mut ids = Vec::new();
    while let Some(row) = rows.next()? {
        ids.push(row.get::<_, NodeId>(0)?);
    }
    drop(rows);
    drop(stmt);
    let mut result = collect(db, &ids)?;
    result.truncated = total > u64::try_from(result.nodes.len()).unwrap_or(u64::MAX);
    result.total_nodes = Some(total);
    Ok(result)
}

/// Nodes by id with the edges among them and everything's provenance.
fn collect(db: &WorkspaceDb, ids: &[NodeId]) -> Result<GraphResult> {
    let nodes = store::nodes(db, ids)?;
    let edges = store::edges(db, ids, EdgeScope::Among)?;
    let subjects: Vec<String> = nodes
        .iter()
        .map(|n| n.id.to_string())
        .chain(edges.iter().map(|e| e.id.to_string()))
        .collect();
    let provenance = store::provenance_of(db, &subjects)?;
    Ok(GraphResult {
        nodes,
        edges,
        provenance,
        roots: Vec::new(),
        ..GraphResult::default()
    })
}

/// A depth-first text rendering: each root, then its neighbours indented
/// with the relation, then a summary line. The terminal, `quack graph`,
/// MCP, and the agent's graph tools all show this.
impl fmt::Display for GraphResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nodes.is_empty() {
            return f.write_str("No matching entities.");
        }
        let mut tree = TreeWriter::new(self);
        let roots: Vec<&Node> = if self.roots.is_empty() {
            self.nodes.iter().collect()
        } else {
            self.roots
                .iter()
                .filter_map(|id| tree.by_id.get(id.as_str()).copied())
                .collect()
        };
        for node in roots.into_iter().chain(&self.nodes) {
            if !tree.visited.contains(node.id.as_str()) {
                tree.node(f, node, 0)?;
            }
        }
        write!(f, "{}", self.nodes.len())?;
        if let Some(total) = self.total_nodes.filter(|_| self.truncated) {
            write!(f, " of {total} matching")?;
        }
        write!(
            f,
            " nodes, {} edges, {} sources",
            self.edges.len(),
            self.provenance.len()
        )?;
        if self.nodes.iter().any(|n| n.provisional) {
            f.write_str(" (provisional: built from an unreviewed ontology)")?;
        }
        if self.truncated {
            // Without this the reader takes the cap for the population and
            // answers "how many are there" with `max_nodes`.
            f.write_str(match self.total_nodes {
                Some(_) => {
                    " — cut off at the node limit, so this is not the whole class; count with \
                     describe_class rather than by counting these lines"
                }
                None => " — cut off at the node limit, so entities further out are missing",
            })?;
        }
        f.write_str("\n")
    }
}

/// The depth-first walk behind [`GraphResult`]'s rendering: every node
/// once, each edge under the node it leaves or enters.
struct TreeWriter<'a> {
    edges: &'a [super::Edge],
    by_id: BTreeMap<&'a str, &'a Node>,
    visited: BTreeSet<&'a str>,
}

impl<'a> TreeWriter<'a> {
    fn new(result: &'a GraphResult) -> Self {
        Self {
            edges: &result.edges,
            by_id: result.nodes.iter().map(|n| (n.id.as_str(), n)).collect(),
            visited: BTreeSet::new(),
        }
    }

    fn node(&mut self, f: &mut fmt::Formatter<'_>, node: &'a Node, depth: usize) -> fmt::Result {
        self.visited.insert(node.id.as_str());
        let indent = "  ".repeat(depth);
        writeln!(f, "{indent}{node}{}", Suffix(&node.properties))?;
        for edge in self.edges {
            let (other, arrow) = if edge.source_node_id == node.id {
                (edge.target_node_id.as_str(), "->")
            } else if edge.target_node_id == node.id {
                (edge.source_node_id.as_str(), "<-")
            } else {
                continue;
            };
            let Some(next) = self.by_id.get(other).copied() else {
                continue;
            };
            write!(
                f,
                "{indent}  {arrow} {}{}",
                edge.relation_id,
                Suffix(&edge.properties)
            )?;
            if self.visited.contains(other) {
                writeln!(f, " {}", next.label)?;
                continue;
            }
            writeln!(f)?;
            self.node(f, next, depth.saturating_add(2))?;
        }
        Ok(())
    }
}

/// Properties after a label or relation: a space and the rendering, or
/// nothing when no property has a value.
struct Suffix<'a>(&'a Properties);

impl fmt::Display for Suffix<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = self.0.to_string();
        if text.is_empty() {
            return Ok(());
        }
        write!(f, " {text}")
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn hops_are_at_least_one_with_named_defaults() {
        assert_eq!(Hops::new(0).get(), 1);
        assert_eq!(Hops::new(5).get(), 5);
        assert_eq!(Hops::neighborhood(None), Hops::NEIGHBORHOOD);
        assert_eq!(Hops::neighborhood(Some(0)).get(), 1);
        assert_eq!(Hops::path(None), Hops::PATH);
        assert_eq!(Hops::path(Some(0)).get(), 1);
        assert_eq!((Hops::NEIGHBORHOOD.get(), Hops::PATH.get()), (2, 4));
    }
    use crate::graph::Edge;

    fn node(id: &str, label: &str, properties: serde_json::Value) -> Node {
        Node {
            id: NodeId::from(String::from(id)),
            label: String::from(label),
            class_id: String::from("organization"),
            properties: Properties::from(properties),
            provisional: false,
        }
    }

    #[test]
    fn the_tree_carries_node_and_edge_properties() {
        let result = GraphResult {
            nodes: vec![
                node("a", "Acme", json!({ "founded": 1999 })),
                node("b", "Orgenics", json!({})),
            ],
            edges: vec![Edge {
                id: EdgeId::from("e"),
                source_node_id: NodeId::from("a"),
                target_node_id: NodeId::from("b"),
                relation_id: String::from("supplies"),
                weight: 1.0,
                properties: Properties::from(json!({ "since": "2020" })),
                provisional: false,
            }],
            provenance: Vec::new(),
            roots: vec![NodeId::from("a")],
            ..GraphResult::default()
        };
        let tree = result.to_string();
        assert_eq!(
            tree,
            "Acme (organization) {founded: 1999}\n  -> supplies {since: 2020}\n    \
             Orgenics (organization)\n      <- supplies {since: 2020} Acme\n2 nodes, 1 edges, 0 sources\n"
        );
    }

    #[test]
    fn an_edge_back_to_a_visited_node_names_it_without_descending() {
        let edge = |id: &str, from: &str, to: &str| Edge {
            id: EdgeId::from(String::from(id)),
            source_node_id: NodeId::from(String::from(from)),
            target_node_id: NodeId::from(String::from(to)),
            relation_id: String::from("knows"),
            weight: 1.0,
            properties: Properties::default(),
            provisional: false,
        };
        let result = GraphResult {
            nodes: vec![node("a", "A", json!({})), node("b", "B", json!({}))],
            edges: vec![edge("ab", "a", "b"), edge("ba", "b", "a")],
            ..GraphResult::default()
        };
        assert_eq!(
            result.to_string(),
            "A (organization)\n  -> knows\n    B (organization)\n      <- knows A\n      -> knows A\n  \
             <- knows B\n2 nodes, 2 edges, 0 sources\n"
        );
        assert_eq!(GraphResult::default().to_string(), "No matching entities.");
    }
}
