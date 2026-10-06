//! Read-only views over the graph (issue #406): the one sanctioned SQL
//! window into it. `graph_<class>` has a row per entity of the class or a
//! subclass, with one typed column per property; `graph_edges` has a row
//! per edge with both ends' labels. The column lists are explicit, so
//! embeddings, normalized labels, and provenance never surface. The views
//! read the live rows, so extraction, merges, and revalidation need no
//! hook; only the ontology decides which views exist, and [`ensure`] runs
//! whenever it is saved and when the workspace opens. User and agent SQL
//! may read the views and may not create, replace, or drop anything named
//! `graph_` (`WorkspaceDb::classify_user_statement`).

use std::collections::BTreeSet;

use crate::error::Result;
use crate::ontology::{Ontology, PropertyType};
use crate::storage::workspace::{StatementKind, WorkspaceDb, quote_ident};

/// Every graph view's name starts with this; user tables may not.
pub const PREFIX: &str = "graph_";

/// The view of every edge.
pub const EDGES_VIEW: &str = "graph_edges";

/// Marks the views quack made, so [`ensure`] drops only its own.
const COMMENT: &str = "quack graph view";

/// The columns every class view starts with.
const NODE_COLUMNS: [&str; 4] = ["id", "label", "class_id", "provisional"];

/// Whether a name is in the reserved `graph_` space, compared as `DuckDB`
/// compares identifiers.
#[must_use]
pub fn is_reserved(name: &str) -> bool {
    name.to_ascii_lowercase().starts_with(PREFIX)
}

/// The refusal for a statement that would change something named `graph_`.
pub const RESERVED_REFUSED: &str = "graph_ views are read-only and the graph_ prefix is reserved for \
them: read them with SELECT, and give a new table or view another name.";

/// Whether a statement that writes names anything in the `graph_` space.
/// The parser serializes only reads, so a write's names come from its
/// tokens, as for quack's internal tables: a `graph_` word inside a string
/// literal of a write is refused too.
#[must_use]
pub fn write_names_reserved(sql: &str, kind: &StatementKind) -> bool {
    *kind == StatementKind::Write
        && sql
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .any(is_reserved)
}

/// One class's view: its name and its columns, in order, with their types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassView {
    pub name: String,
    pub columns: Vec<(String, &'static str)>,
    /// The property each column after the node columns reads.
    properties: Vec<(String, PropertyType)>,
    classes: Vec<String>,
}

impl ClassView {
    /// The view of `class_id` under `ontology`: the class's own and
    /// inherited properties, and the rows of its subclasses.
    #[must_use]
    pub fn of(ontology: &Ontology, class_id: &str) -> Self {
        let mut columns: Vec<(String, &'static str)> = vec![
            (String::from("id"), "VARCHAR"),
            (String::from("label"), "VARCHAR"),
            (String::from("class_id"), "VARCHAR"),
            (String::from("provisional"), "BOOLEAN"),
        ];
        let mut properties = Vec::new();
        for id in ontology.class_properties(class_id) {
            let kind = ontology
                .property(&id)
                .map_or(PropertyType::String, |p| p.kind);
            // A property named like a node column keeps its value under a
            // suffixed name rather than hiding the node's.
            let column = if NODE_COLUMNS.contains(&id.as_str()) {
                format!("{id}_property")
            } else {
                id.clone()
            };
            columns.push((column, Self::sql_type(kind)));
            properties.push((id, kind));
        }
        Self {
            name: format!("{PREFIX}{class_id}"),
            columns,
            properties,
            classes: ontology
                .class_and_descendants(class_id)
                .into_iter()
                .map(|c| c.to_string())
                .collect(),
        }
    }

    const fn sql_type(kind: PropertyType) -> &'static str {
        match kind {
            PropertyType::String | PropertyType::Enum => "VARCHAR",
            PropertyType::Number => "DOUBLE",
            PropertyType::Date => "DATE",
            PropertyType::Boolean => "BOOLEAN",
        }
    }

    /// The `CREATE OR REPLACE VIEW` statement. Class and property ids are
    /// `snake_case` (`Ontology::validate`), and still go in as quoted
    /// identifiers and escaped literals.
    #[must_use]
    pub fn ddl(&self) -> String {
        let mut select = vec![
            String::from("id"),
            String::from("label"),
            String::from("class_id"),
            String::from("provisional"),
        ];
        for ((column, sql_type), (property, _)) in self
            .columns
            .iter()
            .skip(NODE_COLUMNS.len())
            .zip(&self.properties)
        {
            let value = format!(
                "json_extract_string(properties, {})",
                literal(&format!("$.\"{property}\""))
            );
            let typed = if *sql_type == "VARCHAR" {
                value
            } else {
                format!("TRY_CAST({value} AS {sql_type})")
            };
            select.push(format!("{typed} AS {}", quote_ident(column)));
        }
        let classes: Vec<String> = self.classes.iter().map(|c| literal(c)).collect();
        format!(
            "CREATE OR REPLACE VIEW {} AS SELECT {} FROM _quack_graph_nodes WHERE class_id IN ({})",
            quote_ident(&self.name),
            select.join(", "),
            classes.join(", ")
        )
    }
}

/// A SQL string literal.
fn literal(text: &str) -> String {
    format!("'{}'", text.replace('\'', "''"))
}

/// The edges view's statement.
fn edges_ddl() -> String {
    format!(
        "CREATE OR REPLACE VIEW {} AS \
         SELECT e.id, e.source_node_id AS source_id, s.label AS source_label, e.relation_id, \
                e.target_node_id AS target_id, t.label AS target_label, e.provisional, e.properties \
         FROM _quack_graph_edges e \
         JOIN _quack_graph_nodes s ON s.id = e.source_node_id \
         JOIN _quack_graph_nodes t ON t.id = e.target_node_id",
        quote_ident(EDGES_VIEW)
    )
}

/// The graph views quack made, by name.
///
/// # Errors
///
/// Returns an error if the catalog cannot be read.
pub fn names(db: &WorkspaceDb) -> Result<BTreeSet<String>> {
    let mut stmt = db.connection().prepare(
        "SELECT view_name FROM duckdb_views() \
         WHERE schema_name = 'main' AND NOT temporary AND comment = ?",
    )?;
    let names = stmt
        .query_map(duckdb::params![COMMENT], |row| row.get(0))?
        .collect::<duckdb::Result<BTreeSet<String>>>()?;
    Ok(names)
}

/// Make the views match `ontology`: one per class and the edges view,
/// replaced so a changed property list or subclass shows, and quack's
/// views of classes that are gone dropped. A class whose view name a user
/// table already holds (one made before the prefix was reserved) is
/// skipped with a warning.
///
/// # Errors
///
/// Returns an error if a statement fails.
pub fn ensure(db: &WorkspaceDb, ontology: &Ontology) -> Result<()> {
    let ours = names(db)?;
    let taken: BTreeSet<String> = db
        .list_tables()?
        .into_iter()
        .filter(|t| is_reserved(t) && !ours.contains(t))
        .collect();
    let mut wanted = BTreeSet::new();
    let views = ontology
        .classes
        .iter()
        .map(|c| ClassView::of(ontology, c.id.as_str()));
    let statements = views
        .map(|v| (v.name.clone(), v.ddl()))
        .chain(std::iter::once((String::from(EDGES_VIEW), edges_ddl())));
    for (name, ddl) in statements {
        if taken.contains(&name) {
            tracing::warn!(view = %name, "a user table holds this graph view's name; the view is not made");
            continue;
        }
        let conn = db.connection();
        conn.execute_batch(&ddl)?;
        conn.execute_batch(&format!(
            "COMMENT ON VIEW {} IS {}",
            quote_ident(&name),
            literal(COMMENT)
        ))?;
        wanted.insert(name);
    }
    for gone in ours.difference(&wanted) {
        db.connection()
            .execute_batch(&format!("DROP VIEW IF EXISTS {}", quote_ident(gone)))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
