//! The ontology inside the workspace file: the live tables and the
//! versioned snapshots (design doc 5.4, 6.3).
//!
//! `save` writes a new version: a snapshot row, then the live tables
//! rewritten from it with `since_version` carried over for items that
//! already existed. `current` reads the live tables; `version` reads a
//! snapshot; `restore` saves an old snapshot as the newest version.

use std::collections::BTreeMap;

use super::candidates::AUTO_ACCEPT_NOTE;
use super::{
    Class, Mapping, MappingRelation, Ontology, Property, PropertyType, ROOT_CLASS, Relation,
};
use crate::error::{Error, Record, Result};
use crate::storage::workspace::WorkspaceDb;

/// A stored version's header.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VersionRow {
    pub version: u32,
    pub author: Option<String>,
    pub note: Option<String>,
    pub created_at: String,
}

/// The newest version number, `0` when no ontology has been saved.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn latest_version(db: &WorkspaceDb) -> Result<u32> {
    let latest: Option<i64> = db.connection().query_row(
        "SELECT max(version) FROM _quack_ontology_versions",
        [],
        |r| r.get(0),
    )?;
    Ok(latest.and_then(|v| u32::try_from(v).ok()).unwrap_or(0))
}

/// Whether the newest version was written by `--auto-accept` and nobody
/// has saved a reviewed version since: a graph built from it is
/// provisional.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn current_is_auto_accepted(db: &WorkspaceDb) -> Result<bool> {
    Ok(versions(db, 1)?.first().is_some_and(|v| {
        v.note
            .as_deref()
            .is_some_and(|n| n.starts_with(AUTO_ACCEPT_NOTE))
    }))
}

/// The live ontology, or `None` when no version has been saved.
///
/// # Errors
///
/// Returns an error if a read fails or a stored row is malformed.
pub fn current(db: &WorkspaceDb) -> Result<Option<Ontology>> {
    let version = latest_version(db)?;
    if version == 0 {
        return Ok(None);
    }
    let conn = db.connection();
    let mut ontology = Ontology {
        version,
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
                .get::<_, Option<String>>(1)?
                .unwrap_or_else(|| String::from(ROOT_CLASS)),
            label: row.get(2)?,
            description: row.get(3)?,
            key: row.get(4)?,
            properties: Vec::new(),
        });
    }
    let mut stmt = conn.prepare(
        "SELECT id, class_id, label, type, CAST(enum_values AS VARCHAR) FROM _quack_ontology_properties ORDER BY id, class_id",
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
        properties.entry(id.clone()).or_insert(Property {
            id: id.clone(),
            label,
            kind,
            values,
        });
        if let Some(class) = ontology.classes.iter_mut().find(|c| c.id == class_id) {
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
    Ok(Some(ontology.normalized()))
}

/// One stored version's snapshot.
///
/// # Errors
///
/// Returns an error if the query fails or the snapshot does not parse.
pub fn version(db: &WorkspaceDb, version: u32) -> Result<Option<Ontology>> {
    let mut stmt = db.connection().prepare(
        "SELECT CAST(snapshot AS VARCHAR) FROM _quack_ontology_versions WHERE version = ?",
    )?;
    let mut rows = stmt.query(duckdb::params![i64::from(version)])?;
    match rows.next()? {
        Some(row) => {
            let text: String = row.get(0)?;
            let mut ontology: Ontology = serde_json::from_str(&text)?;
            ontology.version = version;
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
        "SELECT version, author, note, CAST(created_at AS VARCHAR) FROM _quack_ontology_versions \
         ORDER BY version DESC LIMIT ?",
    )?;
    let mut rows = stmt.query(duckdb::params![i64::from(limit)])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let version: i64 = row.get(0)?;
        out.push(VersionRow {
            version: u32::try_from(version).unwrap_or(0),
            author: row.get(1)?,
            note: row.get(2)?,
            created_at: row.get(3)?,
        });
    }
    Ok(out)
}

/// Validate, check mapped tables and columns against the workspace, and
/// write the ontology as the next version. Returns the stored ontology.
///
/// # Errors
///
/// Returns an error when the ontology is invalid, a mapping names a table
/// or column the workspace lacks, or a write fails.
pub fn save(
    db: &WorkspaceDb,
    ontology: &Ontology,
    author: Option<&str>,
    note: Option<&str>,
) -> Result<Ontology> {
    ontology.validate()?;
    check_mappings(db, ontology)?;
    let previous = current(db)?;
    let next = latest_version(db)?.saturating_add(1);
    let conn = db.connection();
    let mut stored = ontology.normalized();
    stored.version = next;
    let snapshot = serde_json::to_string(&stored)?;
    conn.execute("BEGIN", [])?;
    let outcome = write_version(conn, &stored, previous.as_ref(), &snapshot, author, note);
    match outcome {
        Ok(()) => {
            conn.execute("COMMIT", [])?;
            Ok(stored)
        }
        Err(e) => {
            drop(conn.execute("ROLLBACK", []));
            Err(e)
        }
    }
}

fn write_version(
    conn: &duckdb::Connection,
    stored: &Ontology,
    previous: Option<&Ontology>,
    snapshot: &str,
    author: Option<&str>,
    note: Option<&str>,
) -> Result<()> {
    conn.execute(
        "INSERT INTO _quack_ontology_versions (version, snapshot, author, note) VALUES (?, ?, ?, ?)",
        duckdb::params![i64::from(stored.version), snapshot, author, note],
    )?;
    let prior_since = read_since(conn)?;
    // An item that already existed keeps the version it first appeared in.
    let since = |existed: bool, kind: &'static str, id: &str| -> i64 {
        let kept = existed
            .then(|| prior_since.get(&(kind, id.to_owned())).copied())
            .flatten();
        i64::from(kept.unwrap_or(stored.version))
    };
    for table in [
        "_quack_ontology_classes",
        "_quack_ontology_relations",
        "_quack_ontology_properties",
        "_quack_ontology_mappings",
    ] {
        conn.execute(&format!("DELETE FROM {table}"), [])?;
    }
    for class in &stored.classes {
        let existed = previous.is_some_and(|p| p.class(&class.id).is_some());
        conn.execute(
            "INSERT INTO _quack_ontology_classes (id, parent_id, label, description, key_property, since_version) \
             VALUES (?, ?, ?, ?, ?, ?)",
            duckdb::params![
                class.id,
                class.parent,
                class.label.clone().unwrap_or_else(|| class.id.clone()),
                class.description,
                class.key,
                since(existed, "class", &class.id)
            ],
        )?;
        for property_id in &class.properties {
            let Some(property) = stored.property(property_id) else {
                continue;
            };
            let existed = previous.is_some_and(|p| {
                p.class(&class.id)
                    .is_some_and(|c| c.properties.contains(property_id))
            });
            conn.execute(
                "INSERT INTO _quack_ontology_properties (id, class_id, label, type, enum_values, since_version) \
                 VALUES (?, ?, ?, ?, ?, ?)",
                duckdb::params![
                    property.id,
                    class.id,
                    property.label.clone().unwrap_or_else(|| property.id.clone()),
                    property.kind.as_str(),
                    if property.values.is_empty() { None } else { Some(serde_json::to_string(&property.values)?) },
                    since(existed, "property", &property.id)
                ],
            )?;
        }
    }
    for relation in &stored.relations {
        let existed = previous.is_some_and(|p| p.relation(&relation.id).is_some());
        conn.execute(
            "INSERT INTO _quack_ontology_relations (id, label, description, domain_class, range_class, since_version) \
             VALUES (?, ?, ?, ?, ?, ?)",
            duckdb::params![
                relation.id,
                relation.label.clone().unwrap_or_else(|| relation.id.clone()),
                relation.description,
                relation.domain,
                relation.range,
                since(existed, "relation", &relation.id)
            ],
        )?;
    }
    for mapping in &stored.mappings {
        let existed = previous.is_some_and(|p| p.mappings.iter().any(|m| m.table == mapping.table));
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
    Ok(())
}

/// `since_version` of every live item, keyed by kind and id.
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
            "property",
            "SELECT DISTINCT id, min(since_version) FROM _quack_ontology_properties GROUP BY id",
        ),
        (
            "mapping",
            "SELECT id, since_version FROM _quack_ontology_mappings",
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
pub fn restore(db: &WorkspaceDb, target: u32, author: Option<&str>) -> Result<Ontology> {
    let snapshot =
        version(db, target)?.ok_or_else(|| Record::OntologyVersion.missing(target.to_string()))?;
    save(
        db,
        &snapshot,
        author,
        Some(&format!("restored version {target}")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn db() -> WorkspaceDb {
        WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()))
    }

    #[test]
    fn save_current_versions_and_restore_round_trip() {
        let db = db();
        assert!(current(&db).is_ok_and(|o| o.is_none()));
        assert_eq!(latest_version(&db).unwrap_or(99), 0);
        let v1 = save(
            &db,
            &Ontology::builtin_default(),
            Some("alice"),
            Some("default"),
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(v1.version, 1);
        let live = current(&db)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .unwrap_or_else(|| fail("no ontology"));
        assert_eq!(live.classes.len(), 7);
        assert_eq!(live.relations.len(), 5);
        assert_eq!(live.properties.len(), 5);
        assert!(live.class("person").is_some_and(
            |c| c.properties == ["title", "email"] || c.properties == ["email", "title"]
        ));

        let mut edited = live.clone();
        edited.classes.push(Class {
            id: String::from("vendor"),
            parent: String::from("organization"),
            label: Some(String::from("Vendor")),
            description: None,
            key: None,
            properties: Vec::new(),
        });
        edited.classes.retain(|c| c.id != "concept");
        let v2 = save(&db, &edited, None, None).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(v2.version, 2);
        let since: Vec<(String, i64)> = {
            let mut stmt = db.connection().prepare("SELECT id, since_version FROM _quack_ontology_classes WHERE id IN ('person', 'vendor') ORDER BY id").unwrap_or_else(|e| fail(&e.to_string()));
            let rows = stmt
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
                .unwrap_or_else(|e| fail(&e.to_string()));
            rows.flatten().collect()
        };
        assert_eq!(
            since,
            [(String::from("person"), 1), (String::from("vendor"), 2)]
        );

        let headers = versions(&db, 10).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            headers.iter().map(|h| h.version).collect::<Vec<_>>(),
            [2, 1]
        );
        assert_eq!(
            headers.last().and_then(|h| h.author.clone()).as_deref(),
            Some("alice")
        );
        let old = version(&db, 1)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .unwrap_or_else(|| fail("no v1"));
        assert!(old.class("concept").is_some() && old.class("vendor").is_none());
        let diff = v2.diff(&old);
        assert_eq!(diff.classes.added, ["vendor"]);
        assert_eq!(diff.classes.removed, ["concept"]);

        let restored = restore(&db, 1, Some("bob")).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(restored.version, 3);
        assert!(restored.class("concept").is_some() && restored.class("vendor").is_none());
        assert!(version(&db, 9).is_ok_and(|v| v.is_none()));
        assert!(restore(&db, 9, None).is_err());
    }

    #[test]
    fn a_saved_ontology_reloads_equal_and_diffs_empty() {
        let db = db();
        let saved = save(&db, &Ontology::builtin_default(), None, None)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let live = current(&db)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .unwrap_or_else(|| fail("no ontology"));
        assert_eq!(live, saved);
        assert!(live.diff(&Ontology::builtin_default()).is_empty());
        let again = save(&db, &live, None, None).unwrap_or_else(|e| fail(&e.to_string()));
        assert!(again.diff(&live).is_empty());
        let snapshot = version(&db, 1)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .unwrap_or_else(|| fail("no v1"));
        assert_eq!(snapshot, live);
    }

    #[test]
    fn mappings_must_name_real_tables_and_columns() {
        let db = db();
        assert!(
            db.execute_statement(
                "CREATE TABLE claims (claim_id TEXT, amount DOUBLE, policy_id TEXT)"
            )
            .is_ok()
        );
        let json = r#"{"classes": [{"id": "claim", "key": "claim_id", "properties": ["claim_id", "amount"]}, {"id": "policy", "key": "policy_number", "properties": ["policy_number"]}], "relations": [{"id": "filed_against", "domain": "claim", "range": "policy"}], "properties": [{"id": "claim_id", "type": "string"}, {"id": "amount", "type": "number"}, {"id": "policy_number", "type": "string"}], "mappings": [{"table": "claims", "class": "claim", "key": "claim_id", "properties": {"amount": "amount"}, "relations": [{"relation": "filed_against", "column": "policy_id", "target_class": "policy", "target_key": "policy_number"}]}]}"#;
        let ontology = Ontology::from_json(json).unwrap_or_else(|e| fail(&e.to_string()));
        let saved = save(&db, &ontology, None, None).unwrap_or_else(|e| fail(&e.to_string()));
        let live = current(&db)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .unwrap_or_else(|| fail("no ontology"));
        assert_eq!(live.mappings, saved.mappings);
        assert_eq!(live.mappings.first().map(|m| m.relations.len()), Some(1));

        // A mapping to a table the workspace no longer has (a deleted
        // document) is kept and flagged by graph status, not refused.
        let mut gone_table = ontology.clone();
        gone_table
            .mappings
            .iter_mut()
            .for_each(|m| m.table = String::from("nope"));
        assert!(save(&db, &gone_table, None, None).is_ok());
        let mut wrong_column = ontology;
        wrong_column
            .mappings
            .iter_mut()
            .for_each(|m| m.key = String::from("ghost"));
        let wrong = save(&db, &wrong_column, None, None).err();
        assert!(wrong.is_some_and(|e| e.to_string().contains("column 'ghost'")));
        assert_eq!(
            latest_version(&db).unwrap_or(0),
            2,
            "failed saves write nothing"
        );
    }
}
