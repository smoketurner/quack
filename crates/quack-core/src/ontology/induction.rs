//! Ontology induction from table evidence (design doc 6.5): deterministic,
//! no model calls. Each table proposes a class, each column a typed
//! property, a unique non-null column the key, and a column whose values
//! overlap another table's key a relation; the whole is also proposed as
//! a mapping. Proposals land in `_quack_ontology_candidates` for review.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{
    Class, IdRenames, Mapping, MappingRelation, Ontology, Property, PropertyType, Relation, SnakeId,
};
use crate::error::{Error, Result};
use crate::graph;
use crate::ids::{ClassId, RelationId};
use crate::storage::profile::{ColumnKind, ColumnProfile, TableProfile};
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
            Self::Class(c) => c.id.as_str(),
            Self::Property { property, .. } => &property.id,
            Self::Relation(r) => r.id.as_str(),
            Self::Mapping(m) => &m.table,
        }
    }

    #[must_use]
    pub fn kind(&self) -> ItemKind {
        match self {
            Self::Class(_) => ItemKind::Class,
            Self::Property { .. } => ItemKind::Property,
            Self::Relation(_) => ItemKind::Relation,
            Self::Mapping(_) => ItemKind::Mapping,
        }
    }

    /// Follow renamed class and relation ids wherever the proposal names one.
    pub(crate) fn rename_ids(&mut self, renames: &IdRenames) {
        match self {
            Self::Class(class) => renames.rename_class(class),
            Self::Property { class, .. } => renames.rename_owner(class),
            Self::Relation(relation) => renames.rename_relation(relation),
            Self::Mapping(mapping) => renames.rename_mapping(mapping),
        }
    }
}

/// The kinds of item an ontology holds and a proposal adds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ItemKind {
    Class,
    Property,
    Relation,
    Mapping,
}

text_enum!(ItemKind, "item kind", {
    Class => "class",
    Property => "property",
    Relation => "relation",
    Mapping => "mapping",
});
text_enum_sql!(ItemKind);

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

/// The property type a column's type and values suggest.
fn property_type(
    column: &ColumnProfile,
    rows: u64,
    options: &TableEvidenceOptions,
) -> PropertyType {
    match column.kind() {
        ColumnKind::Boolean => return PropertyType::Boolean,
        ColumnKind::Temporal => return PropertyType::Date,
        ColumnKind::Numeric => return PropertyType::Number,
        ColumnKind::Text | ColumnKind::Other => {}
    }
    if column.non_null > 0 && column.date_share >= 0.9 {
        return PropertyType::Date;
    }
    if rows >= options.enum_min_rows
        && column.distinct > 1
        && column.distinct <= u64::from(options.enum_max_values)
    {
        return PropertyType::Enum;
    }
    PropertyType::String
}

/// A relation name from a foreign-key-like column: `policy_id` becomes
/// `has_policy`.
#[must_use]
pub fn relation_id_for_column(column: &str) -> String {
    let id = SnakeId::from_name(column).into_string();
    let stem = id.strip_suffix("_id").unwrap_or(&id);
    format!("has_{stem}")
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
        .flatten()
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
    let views = graph::views::names(db)?;
    let mut profiles = Vec::new();
    for table in db.list_tables()?.iter().filter(|t| !views.contains(*t)) {
        profiles.push(TableProfile::compute(db, table)?);
    }
    let mut pass = InductionPass {
        db,
        profiles: &profiles,
        known: Known(current),
        options,
        candidates: Vec::new(),
    };
    for profile in &profiles {
        pass.table(profile)?;
    }
    Ok(pass.candidates)
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
        self.0.is_some_and(|o| o.mapping_for_table(table).is_some())
    }
    /// The class a table's rows belong to: the one its mapping already
    /// feeds, whatever it is called (issue #55: `notable_events` maps to
    /// `storm_event`, not to a new `notable_event`), else one derived from
    /// the table name.
    fn class_for_table(&self, table: &str) -> String {
        self.0.and_then(|o| o.mapping_for_table(table)).map_or_else(
            || SnakeId::singular_from(table).into_string(),
            |m| m.class.to_string(),
        )
    }
    /// Whether the table's mapping already turns this column into a
    /// relation.
    fn mapped_relation(&self, table: &str, column: &str) -> bool {
        self.0
            .and_then(|o| o.mapping_for_table(table))
            .is_some_and(|m| m.relations.iter().any(|r| r.column == column))
    }
}

/// The id a foreign-key-like column's relation gets.
enum RelationName {
    /// Not yet known or proposed: propose it.
    New(String),
    /// Already known or proposed with this domain and range.
    Existing(String),
}

/// One induction run over the workspace's tables: the profiles, what the
/// ontology already has, and the candidates proposed so far.
struct InductionPass<'a> {
    db: &'a WorkspaceDb,
    profiles: &'a [TableProfile],
    known: Known<'a>,
    options: &'a TableEvidenceOptions,
    candidates: Vec<Candidate>,
}

impl InductionPass<'_> {
    /// Propose a table's class, its properties, the relations its columns
    /// imply, and its mapping.
    fn table(&mut self, profile: &TableProfile) -> Result<()> {
        let class_id = self.known.class_for_table(&profile.table);
        let mut property_ids = Vec::new();
        let mut property_map = BTreeMap::new();
        let mut relations = Vec::new();

        for column in &profile.columns {
            let property_id = SnakeId::from_name(&column.name).into_string();
            let kind = property_type(column, profile.row_count, self.options);
            let values = if kind == PropertyType::Enum {
                enum_values(self.db, &profile.table, &column.name)?
            } else {
                Vec::new()
            };
            property_ids.push(property_id.clone());
            property_map.insert(column.name.clone(), property_id.clone());
            let is_key = profile.key_column() == Some(column.name.as_str());
            if !self.known.property(&class_id, &property_id) {
                self.candidates.push(Candidate {
                    proposal: Proposal::Property {
                        class: class_id.clone(),
                        property: Property::new(property_id.clone(), kind, values),
                    },
                    evidence: serde_json::json!({
                        "table": profile.table,
                        "column": column.name,
                        "duckdb_type": column.duckdb_type,
                        "rows": profile.row_count,
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
            if !is_key && !self.known.mapped_relation(&profile.table, &column.name) {
                self.relations(profile, column, &mut relations)?;
            }
        }

        if !self.known.class(&class_id) {
            self.candidates.push(Candidate {
                proposal: Proposal::Class(Class {
                    id: ClassId::from(class_id.clone()),
                    parent: ClassId::from(super::ROOT_CLASS),
                    label: None,
                    description: Some(format!("Rows of table {}", profile.table)),
                    key: profile
                        .key_column()
                        .map(|key| SnakeId::from_name(key).into_string()),
                    properties: property_ids,
                }),
                evidence: serde_json::json!({
                    "table": profile.table,
                    "rows": profile.row_count,
                    "columns": profile.columns.len(),
                    "key_column": profile.key_column(),
                }),
                confidence: 1.0,
                low_support: false,
            });
        }
        if let Some(key) = profile.key_column()
            && !self.known.mapping(&profile.table)
        {
            self.candidates.push(Candidate {
                proposal: Proposal::Mapping(Mapping {
                    table: profile.table.clone(),
                    class: ClassId::from(class_id),
                    key: key.to_owned(),
                    properties: property_map,
                    relations,
                }),
                evidence: serde_json::json!({ "table": profile.table, "rows": profile.row_count }),
                confidence: 1.0,
                low_support: false,
            });
        }
        Ok(())
    }

    /// A relation for every other table whose key this column's values
    /// live in.
    fn relations(
        &mut self,
        profile: &TableProfile,
        column: &ColumnProfile,
        relations: &mut Vec<MappingRelation>,
    ) -> Result<()> {
        let class_id = self.known.class_for_table(&profile.table);
        for other in self.profiles.iter().filter(|p| p.table != profile.table) {
            let Some(other_key) = other.key_column() else {
                continue;
            };
            let share = overlap(
                self.db,
                &profile.table,
                &column.name,
                &other.table,
                other_key,
            )?;
            if share < self.options.key_overlap_threshold || column.distinct == 0 {
                continue;
            }
            let target_class = self.known.class_for_table(&other.table);
            let (relation_id, new) =
                match self.relation_name(&column.name, &class_id, &target_class) {
                    RelationName::New(id) => (id, true),
                    RelationName::Existing(id) => (id, false),
                };
            relations.push(MappingRelation {
                relation: RelationId::from(relation_id.clone()),
                column: column.name.clone(),
                target_class: ClassId::from(target_class.clone()),
                target_key: SnakeId::from_name(other_key).into_string(),
            });
            if new {
                self.candidates.push(Candidate {
                    proposal: Proposal::Relation(Relation {
                        id: RelationId::from(relation_id),
                        label: None,
                        description: None,
                        domain: ClassId::from(class_id.clone()),
                        range: ClassId::from(target_class),
                    }),
                    evidence: serde_json::json!({
                        "table": profile.table,
                        "column": column.name,
                        "target_table": other.table,
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

    /// The relation id for a foreign-key-like column, new or already
    /// known or proposed with this domain and range.
    ///
    /// Two tables that share a column name (`event_id` in both `fatalities`
    /// and `locations`) would otherwise propose one `has_event` with two
    /// domains, which no ontology can hold; the second one is qualified by
    /// its domain (`location_has_event`).
    fn relation_name(&self, column: &str, domain: &str, range: &str) -> RelationName {
        let defined = |id: &str| -> Option<Relation> {
            self.known.relation(id).cloned().or_else(|| {
                self.candidates.iter().find_map(|c| match &c.proposal {
                    Proposal::Relation(r) if r.id == id => Some(r.clone()),
                    Proposal::Relation(_)
                    | Proposal::Class(_)
                    | Proposal::Property { .. }
                    | Proposal::Mapping(_) => None,
                })
            })
        };
        let fits = |r: &Relation| r.domain == domain && r.range == range;
        let plain = relation_id_for_column(column);
        match defined(&plain) {
            None => return RelationName::New(plain),
            Some(r) if fits(&r) => return RelationName::Existing(plain),
            Some(_) => {}
        }
        let qualified = format!("{domain}_{plain}");
        if defined(&qualified).is_some_and(|r| fits(&r)) {
            RelationName::Existing(qualified)
        } else {
            RelationName::New(qualified)
        }
    }
}

/// Old property id to new. Not part of [`IdRenames`], which is what a save
/// moves the graph by, and the graph is keyed by class and relation ids only.
#[derive(Default)]
struct RenameMap(BTreeMap<String, String>);

impl RenameMap {
    /// The id `id` was renamed or merged to, or `id` itself.
    fn follow(&self, id: &str) -> String {
        self.0.get(id).cloned().unwrap_or_else(|| id.to_owned())
    }
}

/// Id renames from rename and merge decisions, so later proposals that
/// reference a renamed or merged item follow it.
#[derive(Default)]
struct Renames {
    /// Classes and relations.
    ids: IdRenames,
    properties: RenameMap,
}

impl From<&[(Proposal, Decision)]> for Renames {
    fn from(accepted: &[(Proposal, Decision)]) -> Self {
        let mut out = Self::default();
        for (proposal, decision) in accepted {
            let (Decision::Rename(target) | Decision::MergeInto(target)) = decision else {
                continue;
            };
            match proposal {
                Proposal::Class(c) => {
                    out.ids
                        .classes
                        .insert(c.id.clone(), ClassId::from(target.as_str()));
                }
                Proposal::Relation(r) => {
                    out.ids
                        .relations
                        .insert(r.id.clone(), RelationId::from(target.as_str()));
                }
                Proposal::Property { property, .. } => {
                    out.properties.0.insert(property.id.clone(), target.clone());
                }
                Proposal::Mapping(_) => {}
            }
        }
        out
    }
}

impl Renames {
    /// Add an accepted class to `ontology` under its new ids, or extend the
    /// class already there with its properties and key.
    fn apply_class(&self, ontology: &mut Ontology, class: &Class, decision: &Decision) {
        let mut class = class.clone();
        self.ids.rename_class(&mut class);
        if let Decision::Reparent(parent) = decision {
            class.parent = ClassId::from(parent.as_str());
        }
        class.key = class.key.as_deref().map(|k| self.properties.follow(k));
        class.properties = class
            .properties
            .iter()
            .map(|p| self.properties.follow(p))
            .collect();
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
            property.id = names.properties.follow(&property.id);
            if ontology.property(&property.id).is_none() {
                ontology.properties.push(property);
            }
        }
    }
    for (proposal, decision) in accepted {
        if let Proposal::Class(class) = proposal
            && !merged(decision)
        {
            names.apply_class(&mut ontology, class, decision);
        }
    }
    for (proposal, decision) in accepted {
        if let Proposal::Property { class, property } = proposal
            && !merged(decision)
        {
            let mut class_id = class.clone();
            names.ids.rename_owner(&mut class_id);
            let property_id = names.properties.follow(&property.id);
            if let Some(class) = ontology
                .classes
                .iter_mut()
                .find(|c| c.id == class_id.as_str())
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
            names.ids.rename_relation(&mut relation);
            if ontology.relation(relation.id.as_str()).is_none() {
                ontology.relations.push(relation);
            }
        }
    }
    for (proposal, _) in accepted {
        if let Proposal::Mapping(mapping) = proposal {
            let mut mapping = mapping.clone();
            names.ids.rename_mapping(&mut mapping);
            mapping.properties = mapping
                .properties
                .iter()
                .map(|(column, p)| (column.clone(), names.properties.follow(p)))
                .collect();
            for link in &mut mapping.relations {
                link.target_key = names.properties.follow(&link.target_key);
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
mod tests;
