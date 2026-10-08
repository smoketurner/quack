//! The ontology inside the workspace file: the live tables and the
//! versioned snapshots (design doc 5.4, 6.3).
//!
//! `save` writes a new version: a snapshot row, then the live tables
//! rewritten from it with `since_version` carried over for items that
//! already existed. `current` reads the live tables; `version` reads a
//! snapshot; `restore` saves an old snapshot as the newest version;
//! `rename` saves one with ids renamed and moves the graph with them.

use std::collections::BTreeMap;

use super::{
    Class, IdRenames, Mapping, MappingRelation, Measure, Ontology, OntologyVersion, Property,
    PropertyType, ROOT_CLASS, Relation, candidates,
};
use crate::error::{Error, Result};
use crate::graph::{Standing, store as graph_store, views};
use crate::ids::ClassId;
use crate::storage::control::ResourceKind;
use crate::storage::workspace::{StatementKind, WorkspaceDb};

/// Whether a person reviewed a version before it was saved. A graph
/// built from an auto-accepted version is provisional until someone
/// saves a reviewed one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum Acceptance {
    #[default]
    Reviewed,
    /// Written by `--auto-accept` from induced candidates nobody looked at.
    Auto,
}

text_enum!(Acceptance, "acceptance", {
    Reviewed => "reviewed",
    Auto => "auto",
});
text_enum_sql!(Acceptance);

/// Who saved a version, why, whether anyone reviewed it, and the ids it
/// renames.
#[derive(Debug, Clone, Copy)]
pub struct Revision<'a> {
    pub author: Option<&'a str>,
    pub note: Option<&'a str>,
    pub acceptance: Acceptance,
    /// Ids to rename in the ontology being saved: the save applies them
    /// and moves `since_version`, pending candidates, and the graph's
    /// nodes and edges to the new ids.
    pub renames: Option<&'a IdRenames>,
}

impl<'a> Revision<'a> {
    #[must_use]
    pub fn reviewed(author: Option<&'a str>, note: Option<&'a str>) -> Self {
        Self {
            author,
            note,
            acceptance: Acceptance::Reviewed,
            renames: None,
        }
    }

    #[must_use]
    pub fn auto(author: Option<&'a str>, note: Option<&'a str>) -> Self {
        Self {
            author,
            note,
            acceptance: Acceptance::Auto,
            renames: None,
        }
    }

    /// The same revision, renaming ids as it saves.
    #[must_use]
    pub fn renaming(mut self, renames: &'a IdRenames) -> Self {
        self.renames = Some(renames);
        self
    }
}

/// A stored version's header.
#[derive(Debug, Clone, serde::Serialize, utoipa::ToSchema)]
pub struct VersionRow {
    pub version: OntologyVersion,
    pub author: Option<String>,
    pub note: Option<String>,
    pub acceptance: Acceptance,
    pub created_at: String,
}

/// The newest version, `None` when no ontology has been saved.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn latest_version(db: &WorkspaceDb) -> Result<Option<OntologyVersion>> {
    Ok(db.connection().query_row(
        "SELECT max(version) FROM _quack_ontology_versions",
        [],
        |r| r.get(0),
    )?)
}

/// What a graph built from the newest version stands on: provisional when
/// `--auto-accept` wrote it and nobody has saved a reviewed version since.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn current_standing(db: &WorkspaceDb) -> Result<Standing> {
    Ok(
        if versions(db, 1)?
            .first()
            .is_some_and(|v| v.acceptance == Acceptance::Auto)
        {
            Standing::Provisional
        } else {
            Standing::Reviewed
        },
    )
}

/// The live ontology, or `None` when no version has been saved.
///
/// # Errors
///
/// Returns an error if a read fails or a stored row is malformed.
pub fn current(db: &WorkspaceDb) -> Result<Option<Ontology>> {
    let Some(version) = latest_version(db)? else {
        return Ok(None);
    };
    let conn = db.connection();
    let mut ontology = Ontology {
        version: Some(version),
        ..Ontology::default()
    };
    let mut stmt = conn.prepare(
        "SELECT id, parent_id, label, description, key_property FROM _quack_ontology_classes ORDER BY id",
    )?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        ontology.classes.push(Class {
            id: row.get(0)?,
            parent: row
                .get::<_, Option<ClassId>>(1)?
                .unwrap_or_else(|| ClassId::from(ROOT_CLASS)),
            label: row.get(2)?,
            description: row.get(3)?,
            key: row.get(4)?,
            properties: Vec::new(),
        });
    }
    let mut stmt = conn.prepare(
        "SELECT id, class_id, label, type, CAST(enum_values AS VARCHAR), description, unit, \
         CAST(synonyms AS VARCHAR) FROM _quack_ontology_properties ORDER BY id, class_id",
    )?;
    let mut rows = stmt.query([])?;
    let mut properties: BTreeMap<String, Property> = BTreeMap::new();
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let class_id: String = row.get(1)?;
        let label: Option<String> = row.get(2)?;
        let kind: PropertyType = row.get::<_, String>(3)?.parse()?;
        let values: Option<String> = row.get(4)?;
        let values: Vec<String> = values
            .map(|v| serde_json::from_str(&v))
            .transpose()?
            .unwrap_or_default();
        let synonyms: Option<String> = row.get(7)?;
        let synonyms: Vec<String> = synonyms
            .map(|v| serde_json::from_str(&v))
            .transpose()?
            .unwrap_or_default();
        let description: Option<String> = row.get(5)?;
        let unit: Option<String> = row.get(6)?;
        properties.entry(id.clone()).or_insert_with(|| Property {
            id: id.clone(),
            label,
            kind,
            values,
            description,
            unit,
            synonyms,
        });
        if let Some(class) = ontology
            .classes
            .iter_mut()
            .find(|c| c.id == class_id.as_str())
        {
            class.properties.push(id);
        }
    }
    ontology.properties = properties.into_values().collect();
    let mut stmt = conn.prepare(
        "SELECT id, label, description, domain_class, range_class FROM _quack_ontology_relations ORDER BY id",
    )?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        ontology.relations.push(Relation {
            id: row.get(0)?,
            label: row.get(1)?,
            description: row.get(2)?,
            domain: row.get(3)?,
            range: row.get(4)?,
        });
    }
    let mut stmt = conn.prepare(
        "SELECT table_name, class_id, key_column, CAST(property_map AS VARCHAR), CAST(relation_map AS VARCHAR) \
         FROM _quack_ontology_mappings ORDER BY table_name",
    )?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let property_map: String = row.get(3)?;
        let relation_map: String = row.get(4)?;
        let relations: Vec<MappingRelation> = serde_json::from_str(&relation_map)?;
        ontology.mappings.push(Mapping {
            table: row.get(0)?,
            class: row.get(1)?,
            key: row.get(2)?,
            properties: serde_json::from_str(&property_map)?,
            relations,
        });
    }
    ontology.measures = read_measures(conn)?;
    Ok(Some(ontology.normalized()))
}

/// The live measures.
fn read_measures(conn: &duckdb::Connection) -> Result<Vec<Measure>> {
    let mut stmt = conn.prepare(
        "SELECT id, description, table_name, expression FROM _quack_ontology_measures ORDER BY id",
    )?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(Measure {
            id: row.get(0)?,
            description: row.get(1)?,
            table: row.get(2)?,
            expression: row.get(3)?,
        });
    }
    Ok(out)
}

/// One stored version's snapshot.
///
/// # Errors
///
/// Returns an error if the query fails or the snapshot does not parse.
pub fn version(db: &WorkspaceDb, version: OntologyVersion) -> Result<Option<Ontology>> {
    let mut stmt = db.connection().prepare(
        "SELECT CAST(snapshot AS VARCHAR) FROM _quack_ontology_versions WHERE version = ?",
    )?;
    let mut rows = stmt.query(duckdb::params![version])?;
    match rows.next()? {
        Some(row) => {
            let text: String = row.get(0)?;
            let mut ontology: Ontology = serde_json::from_str(&text)?;
            ontology.version = Some(version);
            Ok(Some(ontology))
        }
        None => Ok(None),
    }
}

/// Version headers, newest first.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn versions(db: &WorkspaceDb, limit: u32) -> Result<Vec<VersionRow>> {
    let mut stmt = db.connection().prepare(
        "SELECT version, author, note, acceptance, CAST(created_at AS VARCHAR) \
         FROM _quack_ontology_versions ORDER BY version DESC LIMIT ?",
    )?;
    let mut rows = stmt.query(duckdb::params![i64::from(limit)])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(VersionRow {
            version: row.get(0)?,
            author: row.get(1)?,
            note: row.get(2)?,
            acceptance: row.get::<_, Option<Acceptance>>(3)?.unwrap_or_default(),
            created_at: row.get(4)?,
        });
    }
    Ok(out)
}

/// Validate, check mapped tables and columns against the workspace, and
/// write the ontology as the next version. Returns the stored ontology.
///
/// With `revision.renames`, the ontology is saved with those ids renamed,
/// and the same transaction moves everything keyed by the old ids:
/// `since_version`, pending candidates, and the graph's nodes and edges.
///
/// # Errors
///
/// Returns an error when the ontology is invalid, a mapping names a table
/// or column the workspace lacks, a rename's old id is not defined or its
/// new id already is (in the ontology or on graph rows left from an
/// earlier one), or a write fails.
pub fn save(db: &WorkspaceDb, ontology: &Ontology, revision: Revision<'_>) -> Result<Ontology> {
    let renamed = revision
        .renames
        .map(|renames| ontology.renamed(renames))
        .transpose()?;
    let ontology = renamed.as_ref().unwrap_or(ontology);
    ontology.validate()?;
    check_mappings(db, ontology)?;
    check_measures(db, ontology)?;
    let previous = current(db)?;
    let next = OntologyVersion::after(latest_version(db)?);
    let mut stored = ontology.normalized();
    stored.version = Some(next);
    let snapshot = serde_json::to_string(&stored)?;
    // One transaction via the RAII guard: it rolls back on drop, so a panic
    // or an `?` return between BEGIN and COMMIT cannot leave the writer's
    // one connection inside an open transaction.
    db.write_transaction(|db| {
        let conn = db.connection();
        write_version(conn, next, &stored, previous.as_ref(), &snapshot, revision)?;
        if let Some(renames) = revision.renames {
            candidates::rename_ids(db, renames)?;
            graph_store::rename_ids(db, renames, next)?;
        }
        views::ensure(db, &stored)?;
        Ok(stored)
    })
}

/// Every measure must read its table: one `SELECT` that names no internal
/// table and plans. A measure over a table the workspace no longer has is
/// kept, like a mapping, so the ontology can still be saved.
fn check_measures(db: &WorkspaceDb, ontology: &Ontology) -> Result<()> {
    if ontology.measures.is_empty() {
        return Ok(());
    }
    let tables = db.list_tables()?;
    for measure in &ontology.measures {
        if !tables.contains(&measure.table) {
            tracing::warn!(measure = %measure.id, table = %measure.table, "measure names a table the workspace does not have");
            continue;
        }
        let refuse = |why: &str| {
            Error::Ontology(format!(
                "measure '{}' is not a read of '{}': {why}",
                measure.id, measure.table
            ))
        };
        let sql = measure.check_statement();
        match db.classify_user_statement(&sql) {
            Ok(StatementKind::Read) => {}
            Ok(StatementKind::Write) => return Err(refuse("it is not one SELECT expression")),
            Ok(StatementKind::Invalid(message)) => return Err(refuse(&message)),
            Err(e) => return Err(refuse(&e.to_string())),
        }
        db.read_only(|db| db.execute_query_capped(&format!("{sql} LIMIT 0"), 0))
            .map_err(|e| refuse(&e.to_string()))?;
    }
    Ok(())
}

fn write_version(
    conn: &duckdb::Connection,
    version: OntologyVersion,
    stored: &Ontology,
    previous: Option<&Ontology>,
    snapshot: &str,
    revision: Revision<'_>,
) -> Result<()> {
    conn.execute(
        "INSERT INTO _quack_ontology_versions (version, snapshot, author, note, acceptance) \
         VALUES (?, ?, ?, ?, ?)",
        duckdb::params![
            version,
            snapshot,
            revision.author,
            revision.note,
            revision.acceptance
        ],
    )?;
    let prior_since = read_since(conn)?;
    let prior_since_property = read_since_property(conn)?;
    // A renamed id carries over what its old id had.
    let none = IdRenames::default();
    let renames = revision.renames.unwrap_or(&none);
    // An item that already existed keeps the version it first appeared in.
    let since = |existed: bool, kind: &'static str, id: &str| -> i64 {
        let kept = existed
            .then(|| prior_since.get(&(kind, id.to_owned())).copied())
            .flatten();
        i64::from(kept.unwrap_or_else(|| version.get()))
    };
    for table in [
        "_quack_ontology_classes",
        "_quack_ontology_relations",
        "_quack_ontology_properties",
        "_quack_ontology_mappings",
        "_quack_ontology_measures",
    ] {
        conn.execute(&format!("DELETE FROM {table}"), [])?;
    }
    for class in &stored.classes {
        let before = renames.class_before(class.id.as_str());
        let existed = previous.is_some_and(|p| p.class(before).is_some());
        conn.execute(
            "INSERT INTO _quack_ontology_classes (id, parent_id, label, description, key_property, since_version) \
             VALUES (?, ?, ?, ?, ?, ?)",
            duckdb::params![
                class.id,
                class.parent,
                class.label.clone().unwrap_or_else(|| class.id.to_string()),
                class.description,
                class.key,
                since(existed, "class", before)
            ],
        )?;
    }
    // Properties have a composite `(id, class_id)` key and a per-membership
    // `since_version`, so they carry over per row in their own helper.
    write_property_rows(
        conn,
        version,
        stored,
        previous,
        renames,
        &prior_since_property,
    )?;
    for relation in &stored.relations {
        let before = renames.relation_before(relation.id.as_str());
        let existed = previous.is_some_and(|p| p.relation(before).is_some());
        conn.execute(
            "INSERT INTO _quack_ontology_relations (id, label, description, domain_class, range_class, since_version) \
             VALUES (?, ?, ?, ?, ?, ?)",
            duckdb::params![
                relation.id,
                relation.label.clone().unwrap_or_else(|| relation.id.to_string()),
                relation.description,
                relation.domain,
                relation.range,
                since(existed, "relation", before)
            ],
        )?;
    }
    for mapping in &stored.mappings {
        let existed = previous.is_some_and(|p| p.mapping_for_table(&mapping.table).is_some());
        conn.execute(
            "INSERT INTO _quack_ontology_mappings (id, table_name, class_id, key_column, property_map, relation_map, since_version) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            duckdb::params![
                mapping.table,
                mapping.table,
                mapping.class,
                mapping.key,
                serde_json::to_string(&mapping.properties)?,
                serde_json::to_string(&mapping.relations)?,
                since(existed, "mapping", &mapping.table)
            ],
        )?;
    }
    write_measure_rows(conn, stored, previous, &since)?;
    Ok(())
}

/// Write the live `_quack_ontology_measures` rows, each keeping the
/// version it first appeared in (`since`).
fn write_measure_rows(
    conn: &duckdb::Connection,
    stored: &Ontology,
    previous: Option<&Ontology>,
    since: &impl Fn(bool, &'static str, &str) -> i64,
) -> Result<()> {
    for measure in &stored.measures {
        let existed = previous.is_some_and(|p| p.measures.iter().any(|m| m.id == measure.id));
        conn.execute(
            "INSERT INTO _quack_ontology_measures (id, description, table_name, expression, since_version) \
             VALUES (?, ?, ?, ?, ?)",
            duckdb::params![
                measure.id,
                measure.description,
                measure.table,
                measure.expression,
                since(existed, "measure", &measure.id)
            ],
        )?;
    }
    Ok(())
}

/// Write the live `_quack_ontology_properties` rows for `stored`, carrying
/// each `(class, property)` membership's `since_version` forward per row.
///
/// `prior_since` is the live table's per-`(class_id, id)` membership
/// `since_version` map read *before* [`write_version`] deleted the rows, so
/// each membership keeps the version it first appeared on its class.
/// Properties are the one ontology kind with a composite key
/// (`PRIMARY KEY (id, class_id)`): the same property id can first appear on
/// one class at one version and on another class at a later one, so each
/// membership row carries its own first-appearance version — never a global
/// minimum across the classes that share the property id, which would
/// silently overwrite a later membership's true first-appearance with an
/// older one. A renamed class's memberships are looked up under the id it
/// had before.
fn write_property_rows(
    conn: &duckdb::Connection,
    version: OntologyVersion,
    stored: &Ontology,
    previous: Option<&Ontology>,
    renames: &IdRenames,
    prior_since: &BTreeMap<(String, String), u32>,
) -> Result<()> {
    for class in &stored.classes {
        let before = renames.class_before(class.id.as_str());
        for property_id in &class.properties {
            let Some(property) = stored.property(property_id) else {
                continue;
            };
            let existed = previous.is_some_and(|p| {
                p.class(before)
                    .is_some_and(|c| c.properties.contains(property_id))
            });
            let kept = existed
                .then(|| {
                    prior_since
                        .get(&(before.to_owned(), property.id.clone()))
                        .copied()
                })
                .flatten();
            conn.execute(
                "INSERT INTO _quack_ontology_properties \
                 (id, class_id, label, type, enum_values, since_version, description, unit, synonyms) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                duckdb::params![
                    property.id,
                    class.id,
                    property.label.clone().unwrap_or_else(|| property.id.clone()),
                    property.kind.as_str(),
                    if property.values.is_empty() {
                        None
                    } else {
                        Some(serde_json::to_string(&property.values)?)
                    },
                    i64::from(kept.unwrap_or_else(|| version.get())),
                    property.description,
                    property.unit,
                    if property.synonyms.is_empty() {
                        None
                    } else {
                        Some(serde_json::to_string(&property.synonyms)?)
                    }
                ],
            )?;
        }
    }
    Ok(())
}

/// `since_version` of every live item, keyed by kind and id.
///
/// Properties are intentionally excluded: `_quack_ontology_properties` has a
/// composite `PRIMARY KEY (id, class_id)`, so the same property id can first
/// appear on one class at one version and on another class at a later one.
/// Each membership row carries its own first-appearance version, which a
/// `min(since_version) ... GROUP BY id` would collapse to a single global
/// minimum and lose the class dimension. Property memberships are read by
/// [`read_since_property`] instead.
fn read_since(conn: &duckdb::Connection) -> Result<BTreeMap<(&'static str, String), u32>> {
    let mut out = BTreeMap::new();
    for (kind, sql) in [
        (
            "class",
            "SELECT id, since_version FROM _quack_ontology_classes",
        ),
        (
            "relation",
            "SELECT id, since_version FROM _quack_ontology_relations",
        ),
        (
            "mapping",
            "SELECT id, since_version FROM _quack_ontology_mappings",
        ),
        (
            "measure",
            "SELECT id, since_version FROM _quack_ontology_measures",
        ),
    ] {
        let mut stmt = conn.prepare(sql)?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let since: i64 = row.get(1)?;
            out.insert((kind, id), u32::try_from(since).unwrap_or(0));
        }
    }
    Ok(out)
}

/// `since_version` of every property membership, keyed by `(class_id, id)`.
/// One row per `(class, property)` membership, each carrying the version it
/// first appeared on that class — never collapsed to a global minimum.
fn read_since_property(conn: &duckdb::Connection) -> Result<BTreeMap<(String, String), u32>> {
    let mut out = BTreeMap::new();
    let mut stmt =
        conn.prepare("SELECT class_id, id, since_version FROM _quack_ontology_properties")?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let class_id: String = row.get(0)?;
        let id: String = row.get(1)?;
        let since: i64 = row.get(2)?;
        out.insert((class_id, id), u32::try_from(since).unwrap_or(0));
    }
    Ok(out)
}

/// Every mapped column must exist in its table. A mapping whose table is
/// gone (its document was deleted) is kept and flagged by
/// `graph::store::status`, so the ontology can still be saved and the
/// mapping removed at leisure.
fn check_mappings(db: &WorkspaceDb, ontology: &Ontology) -> Result<()> {
    if ontology.mappings.is_empty() {
        return Ok(());
    }
    let tables = db.list_tables()?;
    for mapping in &ontology.mappings {
        if !tables.contains(&mapping.table) {
            tracing::warn!(table = %mapping.table, "ontology maps a table the workspace does not have");
            continue;
        }
        let columns: Vec<String> = db
            .describe_table(&mapping.table)?
            .columns
            .into_iter()
            .map(|c| c.name)
            .collect();
        let wanted = std::iter::once(&mapping.key)
            .chain(mapping.properties.keys())
            .chain(mapping.relations.iter().map(|r| &r.column));
        for column in wanted {
            if !columns.contains(column) {
                return Err(Error::Ontology(format!(
                    "mapping for '{}' uses column '{column}', which the table does not have",
                    mapping.table
                )));
            }
        }
    }
    Ok(())
}

/// Save an earlier version's snapshot as the newest version.
///
/// # Errors
///
/// Returns an error when the version does not exist or the save fails.
pub fn restore(
    db: &WorkspaceDb,
    target: OntologyVersion,
    author: Option<&str>,
) -> Result<Ontology> {
    let snapshot = version(db, target)?
        .ok_or_else(|| ResourceKind::OntologyVersion.missing(target.to_string()))?;
    save(
        db,
        &snapshot,
        Revision::reviewed(author, Some(&format!("restored version {target}"))),
    )
}

/// Rename class and relation ids as the next version: the ontology's own
/// references, pending candidates, and the graph's nodes and edges follow
/// in one transaction. The version keeps the acceptance of the one before
/// it, since renaming an id reviews nothing.
///
/// # Errors
///
/// Returns an error when there is no ontology, an old id is not defined, a
/// new id already is (in the ontology or on graph rows left from an earlier
/// one), or the save fails.
pub fn rename(db: &WorkspaceDb, renames: &IdRenames, author: Option<&str>) -> Result<Ontology> {
    let ontology =
        current(db)?.ok_or_else(|| Error::Ontology(String::from("no ontology to rename in")))?;
    let acceptance = versions(db, 1)?
        .first()
        .map(|v| v.acceptance)
        .unwrap_or_default();
    save(
        db,
        &ontology,
        Revision {
            author,
            note: Some(&format!("renamed {renames}")),
            acceptance,
            renames: Some(renames),
        },
    )
}

#[cfg(test)]
mod tests;
