//! Deterministic extraction from mapped tables: every row of a mapped
//! table is a node of the mapping's class keyed by its key column, mapped
//! columns become properties, and each foreign-key-like column becomes an
//! edge to a node of the target class. Provenance is the table and row
//! key; re-running is idempotent.

use super::store::{self, NewNode, Source};
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

/// Build nodes and edges from every mapping in the ontology. A mapping
/// whose table is gone is skipped and reported, not fatal.
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
        out.push(extract_mapping(db, mapping, provisional)?);
    }
    Ok(out)
}

fn extract_mapping(
    db: &WorkspaceDb,
    mapping: &Mapping,
    provisional: bool,
) -> Result<MappingSummary> {
    if !db.list_tables()?.iter().any(|t| t == &mapping.table) {
        tracing::warn!(table = %mapping.table, "mapped table is not in the workspace; skipping");
        return Ok(MappingSummary {
            table: mapping.table.clone(),
            skipped: Some(String::from("the table is not in the workspace")),
            ..MappingSummary::default()
        });
    }
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
    let sql = format!(
        "SELECT {select} FROM {} WHERE {} IS NOT NULL",
        quote_ident(&mapping.table),
        quote_ident(&mapping.key)
    );
    let results = db.execute_query(&sql)?;
    let index_of = |name: &str| results.columns.iter().position(|c| c == name);
    let key_idx = index_of(&mapping.key)
        .ok_or_else(|| Error::Ontology(format!("key column '{}' missing", mapping.key)))?;
    let mut summary = MappingSummary {
        table: mapping.table.clone(),
        ..MappingSummary::default()
    };
    for row in &results.rows {
        let Some(key) = row.get(key_idx).map(cell_text).filter(|k| !k.is_empty()) else {
            continue;
        };
        summary.rows = summary.rows.saturating_add(1);
        let mut properties = serde_json::Map::new();
        for (column, property) in &mapping.properties {
            if let Some(idx) = index_of(column)
                && let Some(value) = row.get(idx)
                && !value.is_null()
            {
                properties.insert(property.clone(), value.clone());
            }
        }
        let node_id = store::upsert_node(
            db,
            &NewNode {
                label: key.clone(),
                class_id: mapping.class.clone(),
                properties: serde_json::Value::Object(properties),
                provisional,
            },
        )?;
        let source = Source::row(&mapping.table, &key);
        store::add_provenance(db, &node_id, &source)?;
        summary.nodes = summary.nodes.saturating_add(1);
        for relation in &mapping.relations {
            let Some(idx) = index_of(&relation.column) else {
                continue;
            };
            let Some(target_key) = row.get(idx).map(cell_text).filter(|k| !k.is_empty()) else {
                continue;
            };
            let target_id = store::upsert_node(
                db,
                &NewNode {
                    label: target_key,
                    class_id: relation.target_class.clone(),
                    properties: serde_json::json!({}),
                    provisional,
                },
            )?;
            store::add_provenance(db, &target_id, &source)?;
            let edge_id = store::upsert_edge(
                db,
                &node_id,
                &target_id,
                &relation.relation,
                &serde_json::json!({}),
                provisional,
            )?;
            store::add_provenance(db, &edge_id, &source)?;
            summary.edges = summary.edges.saturating_add(1);
        }
    }
    Ok(summary)
}

fn cell_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.trim().to_owned(),
        other => other.to_string(),
    }
}
