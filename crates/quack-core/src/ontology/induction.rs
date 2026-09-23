//! Ontology induction from table evidence (design doc 6.5): deterministic,
//! no model calls. Each table proposes a class, each column a typed
//! property, a unique non-null column the key, and a column whose values
//! overlap another table's key a relation; the whole is also proposed as
//! a mapping. Proposals land in `_quack_ontology_candidates` for review.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{Class, Mapping, MappingRelation, Ontology, Property, PropertyType, Relation};
use crate::error::{Error, Result};
use crate::storage::workspace::{WorkspaceDb, quote_ident};

/// Tuning for table evidence.
#[derive(Debug, Clone, Copy)]
pub struct TableEvidenceOptions {
    /// Share of a column's distinct values that must appear in another
    /// table's key for a relation to be proposed.
    pub key_overlap_threshold: f64,
    /// A text column with at most this many distinct values (and at least
    /// `enum_min_rows` rows) is proposed as an enum.
    pub enum_max_values: u32,
    pub enum_min_rows: u64,
}

impl Default for TableEvidenceOptions {
    fn default() -> Self {
        Self {
            key_overlap_threshold: 0.8,
            enum_max_values: 12,
            enum_min_rows: 20,
        }
    }
}

/// What one candidate proposes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Proposal {
    Class(Class),
    Property { class: String, property: Property },
    Relation(Relation),
    Mapping(Mapping),
}

impl Proposal {
    /// The id the candidate would create.
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Class(c) => &c.id,
            Self::Property { property, .. } => &property.id,
            Self::Relation(r) => &r.id,
            Self::Mapping(m) => &m.table,
        }
    }

    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Class(_) => "class",
            Self::Property { .. } => "property",
            Self::Relation(_) => "relation",
            Self::Mapping(_) => "mapping",
        }
    }
}

/// A proposal with its evidence, before it is stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub proposal: Proposal,
    pub evidence: serde_json::Value,
    pub confidence: f64,
    /// Below the document support threshold: stored, but not shown in the
    /// main proposal (design doc 6.5).
    #[serde(default)]
    pub low_support: bool,
}

struct ColumnProfile {
    name: String,
    duckdb_type: String,
    rows: u64,
    non_null: u64,
    distinct: u64,
    samples: Vec<String>,
    /// Share of non-null values that cast to a date.
    date_share: f64,
}

struct TableProfile {
    name: String,
    rows: u64,
    columns: Vec<ColumnProfile>,
    key: Option<String>,
}

fn profile_table(db: &WorkspaceDb, table: &str) -> Result<TableProfile> {
    let described = db.describe_table(table)?;
    let conn = db.connection();
    let quoted = quote_ident(table);
    let rows: i64 = conn.query_row(&format!("SELECT count(*) FROM {quoted}"), [], |r| r.get(0))?;
    let rows = u64::try_from(rows).unwrap_or(0);
    let mut columns = Vec::new();
    for column in &described.columns {
        let q = quote_ident(&column.name);
        let (non_null, distinct, date_share): (i64, i64, Option<f64>) = conn.query_row(
            &format!(
                "SELECT count({q}), count(DISTINCT {q}), \
                 avg(CASE WHEN {q} IS NULL THEN NULL WHEN TRY_CAST({q} AS DATE) IS NOT NULL THEN 1.0 ELSE 0.0 END) \
                 FROM {quoted}"
            ),
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        let mut stmt = conn.prepare(&format!(
            "SELECT DISTINCT CAST({q} AS VARCHAR) FROM {quoted} WHERE {q} IS NOT NULL ORDER BY 1 LIMIT 3"
        ))?;
        let samples: Vec<String> = stmt
            .query_map([], |r| r.get::<_, String>(0))?
            .filter_map(std::result::Result::ok)
            .collect();
        columns.push(ColumnProfile {
            name: column.name.clone(),
            duckdb_type: column.column_type.to_ascii_uppercase(),
            rows,
            non_null: u64::try_from(non_null).unwrap_or(0),
            distinct: u64::try_from(distinct).unwrap_or(0),
            samples,
            date_share: date_share.unwrap_or(0.0),
        });
    }
    let key = columns
        .iter()
        .filter(|c| rows > 0 && c.non_null == rows && c.distinct == rows)
        .min_by_key(|c| {
            let lower = c.name.to_ascii_lowercase();
            if lower == "id" {
                0
            } else if lower.ends_with("_id") || lower.ends_with("id") {
                1
            } else {
                2
            }
        })
        .map(|c| c.name.clone());
    Ok(TableProfile {
        name: table.to_owned(),
        rows,
        columns,
        key,
    })
}

/// `snake_case` from a table or column name.
#[must_use]
pub fn snake_id(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_underscore = true;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    let trimmed = out.trim_end_matches('_').to_owned();
    if trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        format!("t_{trimmed}")
    } else if trimmed.is_empty() {
        String::from("unnamed")
    } else {
        trimmed
    }
}

/// A class name from a table name: `shipments` becomes `shipment`.
#[must_use]
pub fn class_id_for_table(table: &str) -> String {
    let id = snake_id(table);
    if let Some(stem) = id.strip_suffix("ies") {
        format!("{stem}y")
    } else if id.ends_with("ss") || id.len() < 4 {
        id
    } else if let Some(stem) = id.strip_suffix('s') {
        stem.to_owned()
    } else {
        id
    }
}

/// A relation name from a foreign-key-like column: `policy_id` becomes
/// `has_policy`.
#[must_use]
pub fn relation_id_for_column(column: &str) -> String {
    let id = snake_id(column);
    let stem = id.strip_suffix("_id").unwrap_or(&id);
    format!("has_{stem}")
}

fn property_type(profile: &ColumnProfile, options: &TableEvidenceOptions) -> PropertyType {
    let t = profile.duckdb_type.as_str();
    if t == "BOOLEAN" {
        return PropertyType::Boolean;
    }
    if t.starts_with("DATE") || t.starts_with("TIMESTAMP") {
        return PropertyType::Date;
    }
    if [
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "FLOAT",
        "DOUBLE",
    ]
    .contains(&t)
        || t.starts_with("DECIMAL")
    {
        return PropertyType::Number;
    }
    if profile.non_null > 0 && profile.date_share >= 0.9 {
        return PropertyType::Date;
    }
    if profile.rows >= options.enum_min_rows
        && profile.distinct > 1
        && profile.distinct <= u64::from(options.enum_max_values)
    {
        return PropertyType::Enum;
    }
    PropertyType::String
}

fn enum_values(db: &WorkspaceDb, table: &str, column: &str) -> Result<Vec<String>> {
    let mut stmt = db.connection().prepare(&format!(
        "SELECT DISTINCT CAST({} AS VARCHAR) FROM {} WHERE {} IS NOT NULL ORDER BY 1",
        quote_ident(column),
        quote_ident(table),
        quote_ident(column)
    ))?;
    Ok(stmt
        .query_map([], |r| r.get::<_, String>(0))?
        .filter_map(std::result::Result::ok)
        .collect())
}

fn overlap(db: &WorkspaceDb, table: &str, column: &str, other: &str, key: &str) -> Result<f64> {
    let (matched, total): (i64, i64) = db.connection().query_row(
        &format!(
            "SELECT count(DISTINCT t.c) FILTER (WHERE u.k IS NOT NULL), count(DISTINCT t.c) \
             FROM (SELECT CAST({} AS VARCHAR) AS c FROM {} WHERE {} IS NOT NULL) t \
             LEFT JOIN (SELECT DISTINCT CAST({} AS VARCHAR) AS k FROM {}) u ON u.k = t.c",
            quote_ident(column),
            quote_ident(table),
            quote_ident(column),
            quote_ident(key),
            quote_ident(other)
        ),
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    if total == 0 {
        return Ok(0.0);
    }
    #[expect(clippy::cast_precision_loss, reason = "a ratio of counts")]
    Ok(matched as f64 / total as f64)
}

/// Propose classes, properties, keys, relations, and mappings from every
/// user table. With `current`, only what the ontology lacks is proposed;
/// without one, a full draft.
///
/// # Errors
///
/// Returns an error if profiling a table fails.
pub fn propose_from_tables(
    db: &WorkspaceDb,
    current: Option<&Ontology>,
    options: &TableEvidenceOptions,
) -> Result<Vec<Candidate>> {
    let tables = db.list_tables()?;
    let mut profiles = Vec::new();
    for table in &tables {
        profiles.push(profile_table(db, table)?);
    }
    let mut candidates = Vec::new();
    for profile in &profiles {
        propose_table(db, profile, &profiles, current, options, &mut candidates)?;
    }
    Ok(candidates)
}

/// What the current ontology already has, so a proposal skips it.
struct Known<'a>(Option<&'a Ontology>);

impl Known<'_> {
    fn class(&self, id: &str) -> bool {
        self.0.is_some_and(|o| o.class(id).is_some())
    }
    fn property(&self, class: &str, id: &str) -> bool {
        self.0
            .is_some_and(|o| o.class_properties(class).contains(id))
    }
    fn relation(&self, id: &str) -> Option<&Relation> {
        self.0.and_then(|o| o.relation(id))
    }
    fn mapping(&self, table: &str) -> bool {
        self.0
            .is_some_and(|o| o.mappings.iter().any(|m| m.table == table))
    }
    /// The class a mapped table already feeds, whatever it is called
    /// (issue #55: `notable_events` maps to `storm_event`, not to a new
    /// `notable_event`).
    fn mapped_class(&self, table: &str) -> Option<String> {
        self.0.and_then(|o| {
            o.mappings
                .iter()
                .find(|m| m.table == table)
                .map(|m| m.class.clone())
        })
    }
    /// Whether the table's mapping already turns this column into a
    /// relation.
    fn mapped_relation(&self, table: &str, column: &str) -> bool {
        self.0.is_some_and(|o| {
            o.mappings
                .iter()
                .any(|m| m.table == table && m.relations.iter().any(|r| r.column == column))
        })
    }
}

/// The class a table's rows belong to: the one its mapping names, else
/// one derived from the table name.
fn class_for_table(known: &Known<'_>, table: &str) -> String {
    known
        .mapped_class(table)
        .unwrap_or_else(|| class_id_for_table(table))
}

fn propose_table(
    db: &WorkspaceDb,
    profile: &TableProfile,
    profiles: &[TableProfile],
    current: Option<&Ontology>,
    options: &TableEvidenceOptions,
    candidates: &mut Vec<Candidate>,
) -> Result<()> {
    let known = Known(current);
    let class_id = class_for_table(&known, &profile.name);
    let key_property = profile.key.as_deref().map(snake_id);
    let mut property_ids = Vec::new();
    let mut property_map = BTreeMap::new();
    let mut relations = Vec::new();

    for column in &profile.columns {
        let property_id = snake_id(&column.name);
        let kind = property_type(column, options);
        let values = if kind == PropertyType::Enum {
            enum_values(db, &profile.name, &column.name)?
        } else {
            Vec::new()
        };
        property_ids.push(property_id.clone());
        property_map.insert(column.name.clone(), property_id.clone());
        let is_key = profile.key.as_deref() == Some(column.name.as_str());
        if !known.property(&class_id, &property_id) {
            candidates.push(Candidate {
                proposal: Proposal::Property {
                    class: class_id.clone(),
                    property: Property {
                        id: property_id.clone(),
                        label: None,
                        kind,
                        values,
                    },
                },
                evidence: serde_json::json!({
                    "table": profile.name,
                    "column": column.name,
                    "duckdb_type": column.duckdb_type,
                    "rows": column.rows,
                    "non_null": column.non_null,
                    "distinct": column.distinct,
                    "samples": column.samples,
                    "is_key": is_key,
                }),
                confidence: if is_key {
                    1.0
                } else if kind == PropertyType::Enum {
                    0.8
                } else {
                    0.9
                },
                low_support: false,
            });
        }
        if !is_key && !known.mapped_relation(&profile.name, &column.name) {
            propose_relations(
                db,
                profile,
                column,
                profiles,
                &known,
                options,
                &mut relations,
                candidates,
            )?;
        }
    }

    if !known.class(&class_id) {
        candidates.push(Candidate {
            proposal: Proposal::Class(Class {
                id: class_id.clone(),
                parent: String::from(super::ROOT_CLASS),
                label: None,
                description: Some(format!("Rows of table {}", profile.name)),
                key: key_property,
                properties: property_ids,
            }),
            evidence: serde_json::json!({
                "table": profile.name,
                "rows": profile.rows,
                "columns": profile.columns.len(),
                "key_column": profile.key,
            }),
            confidence: 1.0,
            low_support: false,
        });
    }
    if let Some(key) = &profile.key
        && !known.mapping(&profile.name)
    {
        candidates.push(Candidate {
            proposal: Proposal::Mapping(Mapping {
                table: profile.name.clone(),
                class: class_id,
                key: key.clone(),
                properties: property_map,
                relations,
            }),
            evidence: serde_json::json!({ "table": profile.name, "rows": profile.rows }),
            confidence: 1.0,
            low_support: false,
        });
    }
    Ok(())
}

/// A relation for every other table whose key this column's values live in.
#[expect(
    clippy::too_many_arguments,
    reason = "one evidence pass over one column"
)]
fn propose_relations(
    db: &WorkspaceDb,
    profile: &TableProfile,
    column: &ColumnProfile,
    profiles: &[TableProfile],
    known: &Known<'_>,
    options: &TableEvidenceOptions,
    relations: &mut Vec<MappingRelation>,
    candidates: &mut Vec<Candidate>,
) -> Result<()> {
    let class_id = class_for_table(known, &profile.name);
    for other in profiles.iter().filter(|p| p.name != profile.name) {
        let Some(other_key) = other.key.as_deref() else {
            continue;
        };
        let share = overlap(db, &profile.name, &column.name, &other.name, other_key)?;
        if share < options.key_overlap_threshold || column.distinct == 0 {
            continue;
        }
        let target_class = class_for_table(known, &other.name);
        let (relation_id, defined) =
            relation_id_for(&column.name, &class_id, &target_class, known, candidates);
        relations.push(MappingRelation {
            relation: relation_id.clone(),
            column: column.name.clone(),
            target_class: target_class.clone(),
            target_key: snake_id(other_key),
        });
        if !defined {
            candidates.push(Candidate {
                proposal: Proposal::Relation(Relation {
                    id: relation_id,
                    label: None,
                    description: None,
                    domain: class_id.clone(),
                    range: target_class,
                }),
                evidence: serde_json::json!({
                    "table": profile.name,
                    "column": column.name,
                    "target_table": other.name,
                    "target_key": other_key,
                    "overlap": share,
                    "distinct": column.distinct,
                }),
                confidence: share,
                low_support: false,
            });
        }
    }
    Ok(())
}

/// The relation id for a foreign-key-like column, and whether a relation
/// with that id, domain, and range is already known or proposed.
///
/// Two tables that share a column name (`event_id` in both `fatalities` and
/// `locations`) would otherwise propose one `has_event` with two domains,
/// which no ontology can hold; the second one is qualified by its domain
/// (`location_has_event`).
fn relation_id_for(
    column: &str,
    domain: &str,
    range: &str,
    known: &Known<'_>,
    candidates: &[Candidate],
) -> (String, bool) {
    let defined = |id: &str| -> Option<Relation> {
        known.relation(id).cloned().or_else(|| {
            candidates.iter().find_map(|c| match &c.proposal {
                Proposal::Relation(r) if r.id == id => Some(r.clone()),
                Proposal::Relation(_)
                | Proposal::Class(_)
                | Proposal::Property { .. }
                | Proposal::Mapping(_) => None,
            })
        })
    };
    let plain = relation_id_for_column(column);
    match defined(&plain) {
        None => return (plain, false),
        Some(r) if r.domain == domain && r.range == range => return (plain, true),
        Some(_) => {}
    }
    let qualified = format!("{domain}_{plain}");
    let same = defined(&qualified).is_some_and(|r| r.domain == domain && r.range == range);
    (qualified, same)
}

/// Id renames from rename and merge decisions, by kind, so later proposals
/// that reference a renamed or merged item follow it.
#[derive(Default)]
struct Renames {
    classes: BTreeMap<String, String>,
    relations: BTreeMap<String, String>,
    properties: BTreeMap<String, String>,
}

impl Renames {
    fn from(accepted: &[(Proposal, Decision)]) -> Self {
        let mut out = Self::default();
        for (proposal, decision) in accepted {
            let (Decision::Rename(target) | Decision::MergeInto(target)) = decision else {
                continue;
            };
            match proposal {
                Proposal::Class(c) => {
                    out.classes.insert(c.id.clone(), target.clone());
                }
                Proposal::Relation(r) => {
                    out.relations.insert(r.id.clone(), target.clone());
                }
                Proposal::Property { property, .. } => {
                    out.properties.insert(property.id.clone(), target.clone());
                }
                Proposal::Mapping(_) => {}
            }
        }
        out
    }
    fn class(&self, id: &str) -> String {
        self.classes
            .get(id)
            .cloned()
            .unwrap_or_else(|| id.to_owned())
    }
    fn relation(&self, id: &str) -> String {
        self.relations
            .get(id)
            .cloned()
            .unwrap_or_else(|| id.to_owned())
    }
    fn property(&self, id: &str) -> String {
        self.properties
            .get(id)
            .cloned()
            .unwrap_or_else(|| id.to_owned())
    }
}

/// Apply accepted proposals to `base` in dependency order (properties,
/// classes, relations, mappings). A renamed or merged candidate is applied
/// under its new id; every reference in later proposals follows.
///
/// # Errors
///
/// Returns an error when the result does not validate.
pub fn apply(base: Option<&Ontology>, accepted: &[(Proposal, Decision)]) -> Result<Ontology> {
    let mut ontology = base.cloned().unwrap_or_default();
    let names = Renames::from(accepted);
    let merged = |d: &Decision| matches!(d, Decision::MergeInto(_));
    for (proposal, decision) in accepted {
        if let Proposal::Property { property, .. } = proposal
            && !merged(decision)
        {
            let mut property = property.clone();
            property.id = names.property(&property.id);
            if ontology.property(&property.id).is_none() {
                ontology.properties.push(property);
            }
        }
    }
    for (proposal, decision) in accepted {
        if let Proposal::Class(class) = proposal
            && !merged(decision)
        {
            apply_class(&mut ontology, class, decision, &names);
        }
    }
    for (proposal, decision) in accepted {
        if let Proposal::Property { class, property } = proposal
            && !merged(decision)
        {
            let class_id = names.class(class);
            let property_id = names.property(&property.id);
            if let Some(class) = ontology.classes.iter_mut().find(|c| c.id == class_id)
                && !class.properties.contains(&property_id)
            {
                class.properties.push(property_id);
            }
        }
    }
    for (proposal, decision) in accepted {
        if let Proposal::Relation(relation) = proposal
            && !merged(decision)
        {
            let mut relation = relation.clone();
            relation.id = names.relation(&relation.id);
            relation.domain = names.class(&relation.domain);
            relation.range = names.class(&relation.range);
            if ontology.relation(&relation.id).is_none() {
                ontology.relations.push(relation);
            }
        }
    }
    for (proposal, _) in accepted {
        if let Proposal::Mapping(mapping) = proposal {
            let mut mapping = mapping.clone();
            mapping.class = names.class(&mapping.class);
            mapping.properties = mapping
                .properties
                .iter()
                .map(|(column, p)| (column.clone(), names.property(p)))
                .collect();
            for link in &mut mapping.relations {
                link.relation = names.relation(&link.relation);
                link.target_class = names.class(&link.target_class);
                link.target_key = names.property(&link.target_key);
            }
            ontology.mappings.retain(|m| m.table != mapping.table);
            ontology.mappings.push(mapping);
        }
    }
    // A rejected or absent property drops out of the classes, keys, and
    // mappings that named it, instead of blocking the accept.
    let existing: std::collections::BTreeSet<String> =
        ontology.properties.iter().map(|p| p.id.clone()).collect();
    for class in &mut ontology.classes {
        class.properties.retain(|p| existing.contains(p));
        if class.key.as_ref().is_some_and(|k| !existing.contains(k)) {
            class.key = None;
        }
    }
    for mapping in &mut ontology.mappings {
        mapping.properties.retain(|_, p| existing.contains(p));
    }
    let ontology = ontology.normalized();
    ontology.validate().map_err(|e| {
        Error::Ontology(format!(
            "accepting these candidates would leave the ontology invalid: {e}"
        ))
    })?;
    Ok(ontology)
}

fn apply_class(ontology: &mut Ontology, class: &Class, decision: &Decision, names: &Renames) {
    let mut class = class.clone();
    class.id = names.class(&class.id);
    class.parent = names.class(&class.parent);
    if let Decision::Reparent(parent) = decision {
        class.parent.clone_from(parent);
    }
    class.key = class.key.as_deref().map(|k| names.property(k));
    class.properties = class.properties.iter().map(|p| names.property(p)).collect();
    class.properties.sort();
    class.properties.dedup();
    match ontology.classes.iter_mut().find(|c| c.id == class.id) {
        Some(existing) => {
            for p in class.properties {
                if !existing.properties.contains(&p) {
                    existing.properties.push(p);
                }
            }
            if existing.key.is_none() {
                existing.key = class.key;
            }
        }
        None => ontology.classes.push(class),
    }
}

/// What a reviewer decided for one candidate (design doc 6.5).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Decision {
    Accept,
    /// Accept under a different id.
    Rename(String),
    /// Treat as an existing class, relation, or property: nothing is
    /// created, references follow the target.
    MergeInto(String),
    /// Accept a class under another parent.
    Reparent(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn db() -> WorkspaceDb {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        for sql in [
            "CREATE TABLE policies (policy_number TEXT, holder TEXT, effective DATE, premium DOUBLE)",
            "CREATE TABLE claims (claim_id INTEGER, policy_number TEXT, amount DOUBLE, status TEXT, filed TEXT, active BOOLEAN)",
        ] {
            assert!(db.execute_statement(sql).is_ok());
        }
        for i in 0..30 {
            assert!(
                db.execute_statement(&format!(
                    "INSERT INTO policies VALUES ('P{i}', 'Holder {i}', DATE '2024-01-01', {i}.5)"
                ))
                .is_ok()
            );
            let status = ["filed", "paid", "denied"]
                .get(i % 3)
                .copied()
                .unwrap_or("filed");
            assert!(db.execute_statement(&format!("INSERT INTO claims VALUES ({i}, 'P{}', {i}.0, '{status}', '2024-02-{:02}', {})", i % 25, (i % 28).saturating_add(1), i % 2 == 0)).is_ok());
        }
        db
    }

    #[test]
    fn names_are_snake_case_singular_and_prefixed() {
        assert_eq!(snake_id("Ship Mode"), "ship_mode");
        assert_eq!(snake_id("po / so #"), "po_so");
        assert_eq!(snake_id("2024"), "t_2024");
        assert_eq!(class_id_for_table("shipments"), "shipment");
        assert_eq!(class_id_for_table("policies"), "policy");
        assert_eq!(class_id_for_table("address"), "address");
        assert_eq!(class_id_for_table("bus"), "bus");
        assert_eq!(relation_id_for_column("policy_id"), "has_policy");
        assert_eq!(relation_id_for_column("policy_number"), "has_policy_number");
    }

    #[test]
    fn shared_foreign_key_column_names_get_distinct_relations() {
        let db = db();
        assert!(
            db.execute_statement(
                "CREATE TABLE notes (note_id INTEGER, policy_number VARCHAR, body VARCHAR)"
            )
            .is_ok()
        );
        for i in 0..40_u32 {
            assert!(
                db.execute_statement(&format!(
                    "INSERT INTO notes VALUES ({i}, 'P{}', 'note {i}')",
                    i % 25
                ))
                .is_ok()
            );
        }
        let candidates = propose_from_tables(&db, None, &TableEvidenceOptions::default())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let relations: Vec<&Relation> = candidates
            .iter()
            .filter_map(|c| match &c.proposal {
                Proposal::Relation(r) => Some(r),
                Proposal::Class(_) | Proposal::Property { .. } | Proposal::Mapping(_) => None,
            })
            .collect();
        assert_eq!(relations.len(), 2, "one relation per source table");
        assert!(
            relations
                .iter()
                .any(|r| r.id == "has_policy_number" && r.domain == "claim")
        );
        assert!(
            relations
                .iter()
                .any(|r| r.id == "note_has_policy_number" && r.domain == "note")
        );
        let accepted: Vec<(Proposal, Decision)> = candidates
            .iter()
            .map(|c| (c.proposal.clone(), Decision::Accept))
            .collect();
        let ontology = apply(None, &accepted).unwrap_or_else(|e| fail(&e.to_string()));
        assert!(ontology.validate().is_ok());
    }

    #[test]
    fn tables_propose_classes_keys_typed_properties_relations_and_mappings() {
        let db = db();
        let candidates = propose_from_tables(&db, None, &TableEvidenceOptions::default())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let find = |kind: &str, id: &str| {
            candidates
                .iter()
                .find(|c| c.proposal.kind() == kind && c.proposal.id() == id)
        };
        let claim = find("class", "claim").unwrap_or_else(|| fail("no claim class"));
        assert!(
            matches!(&claim.proposal, Proposal::Class(c) if c.key.as_deref() == Some("claim_id") && c.properties.len() == 6)
        );
        let policy = find("class", "policy").unwrap_or_else(|| fail("no policy class"));
        assert!(
            matches!(&policy.proposal, Proposal::Class(c) if c.key.as_deref() == Some("policy_number"))
        );
        let status = find("property", "status").unwrap_or_else(|| fail("no status"));
        assert!(
            matches!(&status.proposal, Proposal::Property { property, .. } if property.kind == PropertyType::Enum && property.values == ["denied", "filed", "paid"])
        );
        let filed = find("property", "filed").unwrap_or_else(|| fail("no filed"));
        assert!(
            matches!(&filed.proposal, Proposal::Property { property, .. } if property.kind == PropertyType::Date),
            "text dates are dates"
        );
        assert!(
            matches!(&find("property", "amount").map(|c| &c.proposal), Some(Proposal::Property { property, .. }) if property.kind == PropertyType::Number)
        );
        assert!(
            matches!(&find("property", "active").map(|c| &c.proposal), Some(Proposal::Property { property, .. }) if property.kind == PropertyType::Boolean)
        );
        let relation = find("relation", "has_policy_number").unwrap_or_else(|| fail("no relation"));
        assert!(
            matches!(&relation.proposal, Proposal::Relation(r) if r.domain == "claim" && r.range == "policy")
        );
        assert!(relation.confidence >= 0.8);
        let mapping = find("mapping", "claims").unwrap_or_else(|| fail("no mapping"));
        assert!(
            matches!(&mapping.proposal, Proposal::Mapping(m) if m.key == "claim_id" && m.relations.len() == 1 && m.properties.len() == 6)
        );

        // Accepting everything yields a valid ontology with the mappings.
        let accepted: Vec<(Proposal, Decision)> = candidates
            .iter()
            .map(|c| (c.proposal.clone(), Decision::Accept))
            .collect();
        let ontology = apply(None, &accepted).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(ontology.classes.len(), 2);
        assert_eq!(ontology.mappings.len(), 2);
        assert!(ontology.class_properties("claim").contains("status"));
        // Extend mode proposes nothing new once everything is in.
        let again = propose_from_tables(&db, Some(&ontology), &TableEvidenceOptions::default())
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(again.is_empty(), "{}", again.len());
        // A table mapped to a class named differently from the table is
        // covered too: nothing is proposed for it (issue #55).
        let mut renamed = ontology.clone();
        for class in &mut renamed.classes {
            if class.id == "claim" {
                class.id = String::from("insurance_claim");
            }
        }
        for mapping in &mut renamed.mappings {
            if mapping.class == "claim" {
                mapping.class = String::from("insurance_claim");
            }
        }
        for relation in &mut renamed.relations {
            if relation.domain == "claim" {
                relation.domain = String::from("insurance_claim");
            }
            if relation.range == "claim" {
                relation.range = String::from("insurance_claim");
            }
        }
        let again = propose_from_tables(&db, Some(&renamed), &TableEvidenceOptions::default())
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(again.is_empty(), "{again:?}");
    }

    #[test]
    fn rename_merge_and_reparent_follow_through_references() {
        let db = db();
        let candidates = propose_from_tables(&db, None, &TableEvidenceOptions::default())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let mut base = Ontology::builtin_default();
        base.version = 1;
        let decisions: Vec<(Proposal, Decision)> = candidates
            .iter()
            .map(|c| {
                let decision = match (&c.proposal, c.proposal.id()) {
                    (Proposal::Class(_), "claim") => Decision::Reparent(String::from("event")),
                    (Proposal::Class(_), "policy") => {
                        Decision::Rename(String::from("insurance_policy"))
                    }
                    (Proposal::Relation(_), _) => Decision::Rename(String::from("filed_against")),
                    (Proposal::Property { .. }, "holder") => {
                        Decision::MergeInto(String::from("title"))
                    }
                    _ => Decision::Accept,
                };
                (c.proposal.clone(), decision)
            })
            .collect();
        let ontology = apply(Some(&base), &decisions).unwrap_or_else(|e| fail(&e.to_string()));
        assert!(ontology.class("insurance_policy").is_some() && ontology.class("policy").is_none());
        assert!(ontology.class("claim").is_some_and(|c| c.parent == "event"));
        let relation = ontology
            .relation("filed_against")
            .unwrap_or_else(|| fail("no relation"));
        assert_eq!(
            (relation.domain.as_str(), relation.range.as_str()),
            ("claim", "insurance_policy")
        );
        let mapping = ontology
            .mappings
            .iter()
            .find(|m| m.table == "claims")
            .unwrap_or_else(|| fail("no mapping"));
        assert_eq!(
            mapping
                .relations
                .first()
                .map(|r| (r.relation.as_str(), r.target_class.as_str())),
            Some(("filed_against", "insurance_policy"))
        );
        let policies = ontology
            .mappings
            .iter()
            .find(|m| m.table == "policies")
            .unwrap_or_else(|| fail("no mapping"));
        assert_eq!(
            policies.properties.get("holder").map(String::as_str),
            Some("title")
        );
        assert!(ontology.property("holder").is_none());
        assert!(
            ontology
                .class_properties("insurance_policy")
                .contains("title")
        );
        assert_eq!(ontology.classes.len(), 9);

        let bad = vec![(
            Proposal::Class(Class {
                id: String::from("x"),
                parent: String::from("ghost"),
                label: None,
                description: None,
                key: None,
                properties: Vec::new(),
            }),
            Decision::Accept,
        )];
        assert!(apply(None, &bad).is_err());
    }
}
