//! The ontology: the schema of the knowledge graph (design doc 6.3).
//!
//! Classes with single inheritance from the implicit root `entity`,
//! relations with a domain and a range class, typed properties, and table
//! mappings. The workspace tables hold the live copy and every accepted
//! version as a snapshot (`store`); [`Ontology`] is that snapshot's shape
//! and the JSON form export and import move between workspaces. A file is
//! never the source of truth.

pub mod candidates;
pub mod documents;
pub mod induction;
pub mod store;

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use duckdb::types::{FromSql, FromSqlError, FromSqlResult, ToSqlOutput, ValueRef};
use serde::{Deserialize, Deserializer, Serialize};

use crate::error::{Error, Result};
use induction::ItemKind;

/// The implicit root class every class descends from.
pub const ROOT_CLASS: &str = "entity";

/// An ontology id made from a name: lowercase ASCII letters, digits, and
/// underscores.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SnakeId(String);

impl SnakeId {
    /// `name` in `snake_case`: letters and digits lowercased, every other
    /// run collapsed to one `_`, a leading digit prefixed with `t_`, and
    /// `unnamed` for a name with nothing usable in it.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
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
        Self(
            if trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                format!("t_{trimmed}")
            } else if trimmed.is_empty() {
                String::from("unnamed")
            } else {
                trimmed
            },
        )
    }

    /// The same, made singular: `shipments` becomes `shipment`, `policies`
    /// `policy`; `address` and short words stay as they are.
    #[must_use]
    pub fn singular_from(name: &str) -> Self {
        let Self(id) = Self::from_name(name);
        Self(if let Some(stem) = id.strip_suffix("ies") {
            format!("{stem}y")
        } else if id.ends_with("ss") || id.len() < 4 {
            id
        } else if let Some(stem) = id.strip_suffix('s') {
            stem.to_owned()
        } else {
            id
        })
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl std::fmt::Display for SnakeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// An id that is already `snake_case`: a lowercase letter, then lowercase
/// letters, digits, or underscores.
impl TryFrom<&str> for SnakeId {
    type Error = NotSnakeCase;

    fn try_from(id: &str) -> std::result::Result<Self, NotSnakeCase> {
        let mut chars = id.chars();
        let valid = chars.next().is_some_and(|c| c.is_ascii_lowercase())
            && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
        if valid {
            Ok(Self(id.to_owned()))
        } else {
            Err(NotSnakeCase(id.to_owned()))
        }
    }
}

/// An id [`SnakeId`] refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotSnakeCase(String);

impl NotSnakeCase {
    /// The refusal as the ontology error for an id of `kind`.
    #[must_use]
    pub fn for_item(self, kind: ItemKind) -> Error {
        Error::Ontology(format!(
            "{kind} id '{}' must be snake_case: a lowercase letter, then lowercase letters, digits, or underscores",
            self.0
        ))
    }
}

/// The implicit relation from any entity to any entity.
pub const MENTIONS_RELATION: &str = "mentions";

/// A saved ontology version: the first save is 1 and each save counts up.
/// An ontology not yet saved, and a graph never built, have none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OntologyVersion(NonZeroU32);

impl OntologyVersion {
    pub const FIRST: Self = Self(NonZeroU32::MIN);

    /// `None` for 0, which no saved version has.
    #[must_use]
    pub fn new(version: u32) -> Option<Self> {
        NonZeroU32::new(version).map(Self)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// The version a save writes after `latest`.
    #[must_use]
    pub fn after(latest: Option<Self>) -> Self {
        latest.map_or(Self::FIRST, |v| Self(v.0.saturating_add(1)))
    }

    /// The version before this one, if there is one.
    #[must_use]
    pub fn previous(self) -> Option<Self> {
        Self::new(self.get().saturating_sub(1))
    }

    /// Reads a version field that older files and hand-written JSON may
    /// give as `0` or `null` for "not saved".
    fn zero_as_none<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Option<Self>, D::Error> {
        Ok(Option::<u32>::deserialize(deserializer)?.and_then(Self::new))
    }
}

impl std::fmt::Display for OntologyVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A version typed at a prompt or in a URL.
impl std::str::FromStr for OntologyVersion {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        text.trim()
            .parse::<u32>()
            .ok()
            .and_then(Self::new)
            .ok_or_else(|| {
                Error::Ontology(format!(
                    "'{text}' is not an ontology version: versions count from 1"
                ))
            })
    }
}

impl duckdb::ToSql for OntologyVersion {
    fn to_sql(&self) -> duckdb::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(i64::from(self.get())))
    }
}

impl FromSql for OntologyVersion {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let version = i64::column_result(value)?;
        u32::try_from(version)
            .ok()
            .and_then(Self::new)
            .ok_or(FromSqlError::OutOfRange(i128::from(version)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Class {
    pub id: String,
    #[serde(default = "root_class")]
    pub parent: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The property that identifies an instance (a policy number).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub properties: Vec<String>,
}

fn root_class() -> String {
    String::from(ROOT_CLASS)
}

/// How many items a `take(limit)` left out, or `None` when it left none.
fn hidden(total: usize, limit: usize) -> Option<usize> {
    total.checked_sub(limit).filter(|rest| *rest > 0)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Relation {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub domain: String,
    pub range: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PropertyType {
    String,
    Number,
    Date,
    Enum,
    Boolean,
}

text_enum!(PropertyType, "property type", {
    String => "string",
    Number => "number",
    Date => "date",
    Enum => "enum",
    Boolean => "boolean",
});

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Property {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(rename = "type")]
    pub kind: PropertyType,
    /// Allowed values for `enum`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<String>,
}

/// One foreign-key-like column of a mapped table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MappingRelation {
    pub relation: String,
    pub column: String,
    pub target_class: String,
    pub target_key: String,
}

/// How a table's rows become nodes and edges (design doc 6.3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mapping {
    pub table: String,
    pub class: String,
    /// The column holding the class key.
    pub key: String,
    /// Column to property id.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub properties: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relations: Vec<MappingRelation>,
}

impl Mapping {
    /// The stable id of a mapping: its table.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.table
    }
}

/// The whole ontology in interchange form.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ontology {
    /// The stored version this was read from; `None` for one not yet saved.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "OntologyVersion::zero_as_none"
    )]
    pub version: Option<OntologyVersion>,
    #[serde(default)]
    pub classes: Vec<Class>,
    #[serde(default)]
    pub relations: Vec<Relation>,
    #[serde(default)]
    pub properties: Vec<Property>,
    #[serde(default)]
    pub mappings: Vec<Mapping>,
}

impl Ontology {
    /// Parse the JSON interchange form and validate it.
    ///
    /// # Errors
    ///
    /// Returns an error when the text does not parse or the ontology is invalid.
    pub fn from_json(text: &str) -> Result<Self> {
        let ontology: Self = serde_json::from_str(text)
            .map_err(|e| Error::Ontology(format!("ontology does not parse: {e}")))?;
        ontology.validate()?;
        Ok(ontology)
    }

    /// The version this was read from. The graph records which version it
    /// was built with, so it can only be built from a saved ontology.
    ///
    /// # Errors
    ///
    /// Returns an error for an ontology that was never saved.
    pub fn saved_version(&self) -> Result<OntologyVersion> {
        self.version.ok_or_else(|| {
            Error::Ontology(String::from(
                "the ontology has not been saved, so it has no version",
            ))
        })
    }

    /// The JSON interchange form.
    ///
    /// # Errors
    ///
    /// Returns an error when serialization fails.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    #[must_use]
    pub fn class(&self, id: &str) -> Option<&Class> {
        self.classes.iter().find(|c| c.id == id)
    }

    #[must_use]
    pub fn relation(&self, id: &str) -> Option<&Relation> {
        self.relations.iter().find(|r| r.id == id)
    }

    #[must_use]
    pub fn property(&self, id: &str) -> Option<&Property> {
        self.properties.iter().find(|p| p.id == id)
    }

    /// The class and its ancestors up to `entity`, nearest first.
    #[must_use]
    pub fn ancestry(&self, class_id: &str) -> Vec<String> {
        let mut chain = vec![class_id.to_owned()];
        let mut current = class_id.to_owned();
        while current != ROOT_CLASS {
            let Some(class) = self.class(&current) else {
                break;
            };
            if chain.len() > self.classes.len().saturating_add(1) {
                break;
            }
            current.clone_from(&class.parent);
            chain.push(current.clone());
        }
        chain
    }

    /// Whether `class_id` is `ancestor` or descends from it.
    #[must_use]
    pub fn is_subclass_of(&self, class_id: &str, ancestor: &str) -> bool {
        ancestor == ROOT_CLASS || self.ancestry(class_id).iter().any(|c| c == ancestor)
    }

    /// Whether an edge of `relation` from a `source_class` node to a
    /// `target_class` node is valid: the relation exists (or is `mentions`)
    /// and each end is its domain or range or descends from it.
    #[must_use]
    pub fn allows_edge(&self, relation: &str, source_class: &str, target_class: &str) -> bool {
        if relation == MENTIONS_RELATION {
            return true;
        }
        let Some(relation) = self.relation(relation) else {
            return false;
        };
        self.is_subclass_of(source_class, &relation.domain)
            && self.is_subclass_of(target_class, &relation.range)
    }

    /// The classes whose parent is this one, nearest first.
    #[must_use]
    pub fn subclasses(&self, class_id: &str) -> Vec<&Class> {
        self.classes
            .iter()
            .filter(|c| c.parent == class_id)
            .collect()
    }

    /// The relations a class can take part in, inherited ones included:
    /// those it is the domain of, and those it is the range of.
    #[must_use]
    pub fn relations_of(&self, class_id: &str) -> (Vec<&Relation>, Vec<&Relation>) {
        let mut out = (Vec::new(), Vec::new());
        for relation in &self.relations {
            if self.is_subclass_of(class_id, &relation.domain) {
                out.0.push(relation);
            }
            if self.is_subclass_of(class_id, &relation.range) {
                out.1.push(relation);
            }
        }
        out
    }

    /// The mapped table a class's nodes are built from, if any.
    #[must_use]
    pub fn mapping_for(&self, class_id: &str) -> Option<&Mapping> {
        self.mappings.iter().find(|m| m.class == class_id)
    }

    /// The mapping that builds nodes from `table`, if any.
    #[must_use]
    pub fn mapping_for_table(&self, table: &str) -> Option<&Mapping> {
        self.mappings.iter().find(|m| m.table == table)
    }

    /// Whether `id` names a class: the root, or one defined here.
    #[must_use]
    pub fn defines_class(&self, id: &str) -> bool {
        id == ROOT_CLASS || self.class(id).is_some()
    }

    /// Whether `id` names a relation: `mentions`, or one defined here.
    #[must_use]
    pub fn defines_relation(&self, id: &str) -> bool {
        id == MENTIONS_RELATION || self.relation(id).is_some()
    }

    /// Every class id, the root first.
    #[must_use]
    pub fn class_ids(&self) -> Vec<&str> {
        let mut ids = vec![ROOT_CLASS];
        ids.extend(self.classes.iter().map(|c| c.id.as_str()));
        ids
    }

    /// Every relation id, `mentions` first.
    #[must_use]
    pub fn relation_ids(&self) -> Vec<&str> {
        let mut ids = vec![MENTIONS_RELATION];
        ids.extend(self.relations.iter().map(|r| r.id.as_str()));
        ids
    }

    /// A class and every class under it: what a listing of the class
    /// covers and what its census counts.
    #[must_use]
    pub fn class_and_descendants(&self, class_id: &str) -> Vec<String> {
        let mut out = vec![class_id.to_owned()];
        for class in &self.classes {
            if class.id != class_id && self.is_subclass_of(&class.id, class_id) {
                out.push(class.id.clone());
            }
        }
        out
    }

    /// The property ids a class carries, its own and inherited.
    #[must_use]
    pub fn class_properties(&self, class_id: &str) -> BTreeSet<String> {
        self.ancestry(class_id)
            .iter()
            .filter_map(|id| self.class(id))
            .flat_map(|c| c.properties.iter().cloned())
            .collect()
    }

    /// Every structural rule the tables cannot enforce (design doc 6.3).
    ///
    /// # Errors
    ///
    /// Returns an `Ontology` error naming the first violation.
    pub fn validate(&self) -> Result<()> {
        self.validate_properties()?;
        self.validate_classes()?;
        self.validate_relations()?;
        self.validate_mappings()
    }

    fn validate_properties(&self) -> Result<()> {
        let mut seen = BTreeSet::new();
        for property in &self.properties {
            SnakeId::try_from(property.id.as_str()).map_err(|e| e.for_item(ItemKind::Property))?;
            if !seen.insert(property.id.as_str()) {
                return Err(Error::Ontology(format!(
                    "property '{}' is declared twice",
                    property.id
                )));
            }
            match property.kind {
                PropertyType::Enum if property.values.is_empty() => {
                    return Err(Error::Ontology(format!(
                        "enum property '{}' needs values",
                        property.id
                    )));
                }
                PropertyType::Enum => {}
                _ if !property.values.is_empty() => {
                    return Err(Error::Ontology(format!(
                        "property '{}' is not an enum and cannot list values",
                        property.id
                    )));
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn validate_classes(&self) -> Result<()> {
        let mut seen = BTreeSet::new();
        for class in &self.classes {
            SnakeId::try_from(class.id.as_str()).map_err(|e| e.for_item(ItemKind::Class))?;
            if class.id == ROOT_CLASS {
                return Err(Error::Ontology(format!(
                    "'{ROOT_CLASS}' is the implicit root and cannot be declared"
                )));
            }
            if !seen.insert(class.id.as_str()) {
                return Err(Error::Ontology(format!(
                    "class '{}' is declared twice",
                    class.id
                )));
            }
        }
        for class in &self.classes {
            if class.parent != ROOT_CLASS && self.class(&class.parent).is_none() {
                return Err(Error::Ontology(format!(
                    "class '{}' has unknown parent '{}'",
                    class.id, class.parent
                )));
            }
            if self.ancestry(&class.id).last().map(String::as_str) != Some(ROOT_CLASS) {
                return Err(Error::Ontology(format!(
                    "class '{}' is in an inheritance cycle",
                    class.id
                )));
            }
            for property in &class.properties {
                if self.property(property).is_none() {
                    return Err(Error::Ontology(format!(
                        "class '{}' lists unknown property '{property}'",
                        class.id
                    )));
                }
            }
            if let Some(key) = &class.key
                && !self.class_properties(&class.id).contains(key)
            {
                return Err(Error::Ontology(format!(
                    "class '{}' has key '{key}', which is not one of its properties",
                    class.id
                )));
            }
        }
        Ok(())
    }

    fn validate_relations(&self) -> Result<()> {
        let mut seen = BTreeSet::new();
        for relation in &self.relations {
            SnakeId::try_from(relation.id.as_str()).map_err(|e| e.for_item(ItemKind::Relation))?;
            if relation.id == MENTIONS_RELATION {
                return Err(Error::Ontology(format!(
                    "'{MENTIONS_RELATION}' is implicit and cannot be declared"
                )));
            }
            if !seen.insert(relation.id.as_str()) {
                return Err(Error::Ontology(format!(
                    "relation '{}' is declared twice",
                    relation.id
                )));
            }
            for (end, class) in [("domain", &relation.domain), ("range", &relation.range)] {
                if class != ROOT_CLASS && self.class(class).is_none() {
                    return Err(Error::Ontology(format!(
                        "relation '{}' has unknown {end} class '{class}'",
                        relation.id
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_mappings(&self) -> Result<()> {
        let mut seen = BTreeSet::new();
        for mapping in &self.mappings {
            if mapping.table.trim().is_empty() || mapping.key.trim().is_empty() {
                return Err(Error::Ontology(String::from(
                    "a mapping needs a table and a key column",
                )));
            }
            if !seen.insert(mapping.table.as_str()) {
                return Err(Error::Ontology(format!(
                    "table '{}' is mapped twice",
                    mapping.table
                )));
            }
            if self.class(&mapping.class).is_none() {
                return Err(Error::Ontology(format!(
                    "mapping for '{}' names unknown class '{}'",
                    mapping.table, mapping.class
                )));
            }
            let allowed = self.class_properties(&mapping.class);
            for (column, property) in &mapping.properties {
                if !allowed.contains(property) {
                    return Err(Error::Ontology(format!(
                        "mapping for '{}' maps column '{column}' to '{property}', which class '{}' does not carry",
                        mapping.table, mapping.class
                    )));
                }
            }
            for link in &mapping.relations {
                self.validate_mapping_relation(mapping, link)?;
            }
        }
        Ok(())
    }

    fn validate_mapping_relation(&self, mapping: &Mapping, link: &MappingRelation) -> Result<()> {
        let Some(relation) = self.relation(&link.relation) else {
            return Err(Error::Ontology(format!(
                "mapping for '{}' uses unknown relation '{}'",
                mapping.table, link.relation
            )));
        };
        if self.class(&link.target_class).is_none() {
            return Err(Error::Ontology(format!(
                "mapping for '{}' targets unknown class '{}'",
                mapping.table, link.target_class
            )));
        }
        if !self.is_subclass_of(&mapping.class, &relation.domain)
            || !self.is_subclass_of(&link.target_class, &relation.range)
        {
            return Err(Error::Ontology(format!(
                "mapping for '{}': relation '{}' goes {} -> {}, not {} -> {}",
                mapping.table,
                relation.id,
                relation.domain,
                relation.range,
                mapping.class,
                link.target_class
            )));
        }
        if !self
            .class_properties(&link.target_class)
            .contains(&link.target_key)
        {
            return Err(Error::Ontology(format!(
                "mapping for '{}': class '{}' has no property '{}' to match on",
                mapping.table, link.target_class, link.target_key
            )));
        }
        Ok(())
    }

    /// The whole ontology, every class, relation and mapping. The
    /// extraction prompt (6.5) needs all of it: the model may only answer
    /// with ids it has been shown.
    #[must_use]
    pub fn render_for_prompt(&self) -> String {
        self.render_capped(usize::MAX)
    }

    /// The block the system prompt carries (design doc 7.2), capped at
    /// `limit` items per section. An ontology induced from a wide
    /// workspace has a class per table and a property per column, which
    /// would crowd the guidance and the question out of a small window —
    /// the same bound the tables block has; `describe_class` has the rest.
    #[must_use]
    pub fn render_capped(&self, limit: usize) -> String {
        let mut lines = vec![
            match self.version {
                Some(version) => format!("Ontology (version {version}):"),
                None => String::from("Ontology (unsaved):"),
            },
            String::from("- classes (child: parent [key] {properties}):"),
        ];
        for class in self.classes.iter().take(limit) {
            let props = self.class_properties(&class.id);
            let key = class
                .key
                .as_deref()
                .map_or(String::new(), |k| format!(" [key {k}]"));
            let props = if props.is_empty() {
                String::new()
            } else {
                format!(
                    " {{{}}}",
                    props.iter().cloned().collect::<Vec<_>>().join(", ")
                )
            };
            lines.push(format!("  - {}: {}{key}{props}", class.id, class.parent));
        }
        if let Some(rest) = hidden(self.classes.len(), limit) {
            lines.push(format!(
                "  - ... and {rest} more classes; describe_class shows any class by id"
            ));
        }
        lines.push(String::from("- relations (id: domain -> range):"));
        for relation in self.relations.iter().take(limit) {
            lines.push(format!(
                "  - {}: {} -> {}",
                relation.id, relation.domain, relation.range
            ));
        }
        if let Some(rest) = hidden(self.relations.len(), limit) {
            lines.push(format!(
                "  - ... and {rest} more relations; describe_class lists the ones a class takes part in"
            ));
        }
        lines.push(format!(
            "  - {MENTIONS_RELATION}: {ROOT_CLASS} -> {ROOT_CLASS}"
        ));
        if !self.mappings.is_empty() {
            lines.push(String::from("- mapped tables:"));
            for mapping in self.mappings.iter().take(limit) {
                lines.push(format!(
                    "  - {} -> {} (key column {})",
                    mapping.table, mapping.class, mapping.key
                ));
            }
            if let Some(rest) = hidden(self.mappings.len(), limit) {
                lines.push(format!("  - ... and {rest} more mapped tables"));
            }
        }
        let mut out = lines.join("\n");
        out.push('\n');
        out
    }

    /// The canonical form: items sorted by id, class property lists sorted,
    /// and labels that merely repeat the id dropped. Save and load both go
    /// through it, so a round trip through the tables compares equal.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let mut out = self.clone();
        out.classes.sort_by(|a, b| a.id.cmp(&b.id));
        out.relations.sort_by(|a, b| a.id.cmp(&b.id));
        out.properties.sort_by(|a, b| a.id.cmp(&b.id));
        out.mappings.sort_by(|a, b| a.table.cmp(&b.table));
        for class in &mut out.classes {
            class.properties.sort();
            class.properties.dedup();
            if class.label.as_deref() == Some(class.id.as_str()) {
                class.label = None;
            }
        }
        for relation in &mut out.relations {
            if relation.label.as_deref() == Some(relation.id.as_str()) {
                relation.label = None;
            }
        }
        for property in &mut out.properties {
            if property.label.as_deref() == Some(property.id.as_str()) {
                property.label = None;
            }
        }
        out
    }

    /// What changed from `older` to `self`, by id, comparing canonical forms.
    #[must_use]
    pub fn diff(&self, older: &Self) -> OntologyDiff {
        let (newer, older) = (self.normalized(), older.normalized());
        let this = &newer;
        OntologyDiff {
            from: older.version,
            to: this.version,
            classes: Changes::between(&older.classes, &this.classes, |c| c.id.as_str()),
            relations: Changes::between(&older.relations, &this.relations, |r| r.id.as_str()),
            properties: Changes::between(&older.properties, &this.properties, |p| p.id.as_str()),
            mappings: Changes::between(&older.mappings, &this.mappings, Mapping::id),
        }
    }

    /// The built-in general ontology installed as version 1 (design doc 6.3).
    #[must_use]
    pub fn builtin_default() -> Self {
        let class = |id: &str, parent: &str, properties: &[&str]| Class {
            id: id.to_owned(),
            parent: parent.to_owned(),
            label: None,
            description: None,
            key: None,
            properties: properties.iter().map(|p| (*p).to_owned()).collect(),
        };
        let relation = |id: &str, domain: &str, range: &str| Relation {
            id: id.to_owned(),
            label: None,
            description: None,
            domain: domain.to_owned(),
            range: range.to_owned(),
        };
        let property = |id: &str, kind: PropertyType| Property {
            id: id.to_owned(),
            label: None,
            kind,
            values: Vec::new(),
        };
        Self {
            version: None,
            classes: vec![
                class("person", ROOT_CLASS, &["title", "email"]),
                class("organization", ROOT_CLASS, &["industry", "country"]),
                class("place", ROOT_CLASS, &["country"]),
                class("event", ROOT_CLASS, &["date"]),
                class("product", ROOT_CLASS, &[]),
                class("document", ROOT_CLASS, &["date"]),
                class("concept", ROOT_CLASS, &[]),
            ],
            relations: vec![
                relation("works_at", "person", "organization"),
                relation("located_in", ROOT_CLASS, "place"),
                relation("part_of", ROOT_CLASS, ROOT_CLASS),
                relation("produced_by", "product", "organization"),
                relation("occurred_at", "event", "place"),
            ],
            properties: vec![
                property("title", PropertyType::String),
                property("email", PropertyType::String),
                property("industry", PropertyType::String),
                property("country", PropertyType::String),
                property("date", PropertyType::Date),
            ],
            mappings: Vec::new(),
        }
    }
}

/// Ids added, removed, or changed for one kind.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Changes {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub changed: Vec<String>,
}

impl Changes {
    /// Added, removed, and changed ids between two lists of one kind.
    #[must_use]
    pub fn between<T: PartialEq>(old: &[T], new: &[T], id: impl Fn(&T) -> &str) -> Self {
        let mut out = Self::default();
        for item in new {
            match old.iter().find(|o| id(o) == id(item)) {
                None => out.added.push(id(item).to_owned()),
                Some(before) if before != item => out.changed.push(id(item).to_owned()),
                Some(_) => {}
            }
        }
        for item in old {
            if !new.iter().any(|n| id(n) == id(item)) {
                out.removed.push(id(item).to_owned());
            }
        }
        out
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

/// The difference between two versions.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct OntologyDiff {
    pub from: Option<OntologyVersion>,
    pub to: Option<OntologyVersion>,
    pub classes: Changes,
    pub relations: Changes,
    pub properties: Changes,
    pub mappings: Changes,
}

impl Ontology {
    /// A readable rendering: the class tree, then relations, properties,
    /// mappings.
    #[must_use]
    pub fn render_summary(&self) -> String {
        fn children(ontology: &Ontology, parent: &str, depth: usize, lines: &mut Vec<String>) {
            for class in ontology.classes.iter().filter(|c| c.parent == parent) {
                let key = class
                    .key
                    .as_deref()
                    .map_or(String::new(), |k| format!(" [key {k}]"));
                let props = if class.properties.is_empty() {
                    String::new()
                } else {
                    format!(" {{{}}}", class.properties.join(", "))
                };
                lines.push(format!(
                    "{}- {}{key}{props}",
                    "  ".repeat(depth.saturating_add(1)),
                    class.id
                ));
                children(ontology, &class.id, depth.saturating_add(1), lines);
            }
        }
        let mut lines = vec![
            match self.version {
                Some(version) => format!("Ontology version {version}"),
                None => String::from("Ontology (unsaved)"),
            },
            String::from("classes:"),
        ];
        children(self, ROOT_CLASS, 0, &mut lines);
        lines.push(String::from("relations:"));
        for r in &self.relations {
            lines.push(format!("  - {}: {} -> {}", r.id, r.domain, r.range));
        }
        lines.push(String::from("properties:"));
        for p in &self.properties {
            let values = if p.values.is_empty() {
                String::new()
            } else {
                format!(" [{}]", p.values.join(", "))
            };
            lines.push(format!("  - {}: {}{values}", p.id, p.kind.as_str()));
        }
        if !self.mappings.is_empty() {
            lines.push(String::from("mappings:"));
            for m in &self.mappings {
                lines.push(format!(
                    "  - {} -> {} (key {}, {} properties, {} relations)",
                    m.table,
                    m.class,
                    m.key,
                    m.properties.len(),
                    m.relations.len()
                ));
            }
        }
        let mut text = lines.join("\n");
        text.push('\n');
        text
    }
}

impl OntologyDiff {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
            && self.relations.is_empty()
            && self.properties.is_empty()
            && self.mappings.is_empty()
    }
}

impl std::fmt::Display for OntologyDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let side = |version: Option<OntologyVersion>| {
            version.map_or_else(|| String::from("unsaved"), |v| v.to_string())
        };
        writeln!(f, "version {} -> {}", side(self.from), side(self.to))?;
        if self.is_empty() {
            return writeln!(f, "  no changes");
        }
        for (kind, changes) in [
            ("classes", &self.classes),
            ("relations", &self.relations),
            ("properties", &self.properties),
            ("mappings", &self.mappings),
        ] {
            if changes.is_empty() {
                continue;
            }
            writeln!(f, "  {kind}:")?;
            for id in &changes.added {
                writeln!(f, "    + {id}")?;
            }
            for id in &changes.removed {
                writeln!(f, "    - {id}")?;
            }
            for id in &changes.changed {
                writeln!(f, "    ~ {id}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_nests_subclasses_under_parents() {
        let mut ontology = Ontology::builtin_default();
        ontology.classes.push(Class {
            id: String::from("vendor"),
            parent: String::from("organization"),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        });
        let text = ontology.render_summary();
        assert!(
            text.contains("  - organization {industry, country}\n    - vendor\n"),
            "{text}"
        );
        assert!(text.contains("  - works_at: person -> organization"));
        assert!(text.contains("  - date: date"));
    }

    #[test]
    fn versions_count_from_one() {
        assert_eq!(OntologyVersion::new(0), None);
        assert_eq!(OntologyVersion::after(None), OntologyVersion::FIRST);
        let second = OntologyVersion::after(Some(OntologyVersion::FIRST));
        assert_eq!(second.get(), 2);
        assert_eq!(second.previous(), Some(OntologyVersion::FIRST));
        assert_eq!(OntologyVersion::FIRST.previous(), None);
        assert_eq!(
            " 7 "
                .parse::<OntologyVersion>()
                .map(OntologyVersion::get)
                .ok(),
            Some(7)
        );
        for bad in ["0", "-1", "v2", ""] {
            assert!(bad.parse::<OntologyVersion>().is_err(), "{bad}");
        }
    }

    #[test]
    fn an_unsaved_version_is_absent_in_json_and_read_from_zero_or_null() {
        let json = Ontology::builtin_default().to_json().unwrap_or_default();
        assert!(!json.contains("\"version\""), "{json}");
        for version in ["0", "null"] {
            let text = format!(r#"{{"version": {version}}}"#);
            assert_eq!(
                Ontology::from_json(&text).map(|o| o.version).ok(),
                Some(None),
                "{text}"
            );
        }
        let saved = Ontology::from_json(r#"{"version": 3}"#)
            .map(|o| o.version)
            .ok();
        assert_eq!(saved, Some(OntologyVersion::new(3)));
        assert!(Ontology::from_json(r#"{"version": -1}"#).is_err());
    }

    #[test]
    fn snake_ids_are_checked_not_repaired() {
        assert_eq!(
            SnakeId::try_from("ship_mode_2").map(SnakeId::into_string),
            Ok(String::from("ship_mode_2"))
        );
        for bad in ["", "Ship", "2024", "_x", "ship mode", "ship-mode"] {
            let refused =
                SnakeId::try_from(bad).map_err(|e| e.for_item(ItemKind::Class).to_string());
            assert_eq!(
                refused,
                Err(format!(
                    "ontology error: class id '{bad}' must be snake_case: a lowercase letter, then \
                     lowercase letters, digits, or underscores"
                )),
                "{bad}"
            );
        }
    }

    fn err_of(json: &str) -> String {
        match Ontology::from_json(json) {
            Ok(_) => String::from("<ok>"),
            Err(e) => e.to_string(),
        }
    }

    const INSURANCE: &str = r#"{
  "classes": [
    { "id": "organization", "properties": ["country"] },
    { "id": "vendor", "parent": "organization" },
    { "id": "policy", "key": "policy_number", "properties": ["policy_number", "effective_date"] },
    { "id": "claim", "key": "claim_id", "properties": ["claim_id", "amount", "status"] }
  ],
  "relations": [
    { "id": "issued_by", "domain": "policy", "range": "organization" },
    { "id": "filed_against", "domain": "claim", "range": "policy" }
  ],
  "properties": [
    { "id": "country", "type": "string" },
    { "id": "policy_number", "type": "string" },
    { "id": "effective_date", "type": "date" },
    { "id": "claim_id", "type": "string" },
    { "id": "amount", "type": "number" },
    { "id": "status", "type": "enum", "values": ["filed", "paid", "denied"] }
  ],
  "mappings": [
    {
      "table": "claims", "class": "claim", "key": "claim_id",
      "properties": { "amount": "amount", "status": "status" },
      "relations": [
        { "relation": "filed_against", "column": "policy_id", "target_class": "policy", "target_key": "policy_number" }
      ]
    }
  ]
}"#;

    #[test]
    fn the_design_example_parses_and_round_trips_through_json() {
        let ontology = Ontology::from_json(INSURANCE).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(ontology.classes.len(), 4);
        assert!(ontology.is_subclass_of("vendor", "organization"));
        assert!(ontology.is_subclass_of("vendor", ROOT_CLASS));
        assert!(!ontology.is_subclass_of("policy", "organization"));
        assert!(ontology.class_properties("vendor").contains("country"));
        let json = ontology.to_json().unwrap_or_else(|e| fail(&e.to_string()));
        let again = Ontology::from_json(&json).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(again, ontology);
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[test]
    fn validation_names_each_violation() {
        let c = |body: &str| format!("{{\"classes\": [{body}]}}");
        assert!(err_of(&c(r#"{"id": "Bad-Id"}"#)).contains("snake_case"));
        assert!(err_of(&c(r#"{"id": "entity"}"#)).contains("implicit root"));
        assert!(err_of(&c(r#"{"id": "a"}, {"id": "a"}"#)).contains("declared twice"));
        assert!(err_of(&c(r#"{"id": "a", "parent": "ghost"}"#)).contains("unknown parent"));
        assert!(
            err_of(&c(
                r#"{"id": "a", "parent": "b"}, {"id": "b", "parent": "a"}"#
            ))
            .contains("cycle")
        );
        assert!(err_of(&c(r#"{"id": "a", "properties": ["nope"]}"#)).contains("unknown property"));
        assert!(err_of(r#"{"classes": [{"id": "a", "key": "k"}], "properties": [{"id": "k", "type": "string"}]}"#).contains("not one of its properties"));
        assert!(
            err_of(r#"{"properties": [{"id": "s", "type": "enum"}]}"#).contains("needs values")
        );
        assert!(
            err_of(r#"{"properties": [{"id": "s", "type": "string", "values": ["x"]}]}"#)
                .contains("cannot list values")
        );
        assert!(
            err_of(r#"{"relations": [{"id": "mentions", "domain": "entity", "range": "entity"}]}"#)
                .contains("implicit")
        );
        assert!(
            err_of(r#"{"relations": [{"id": "r", "domain": "nope", "range": "entity"}]}"#)
                .contains("unknown domain")
        );
        assert!(err_of(r#"{"classes": [{"id": "a"}], "mappings": [{"table": "t", "class": "a", "key": ""}]}"#).contains("needs a table and a key"));
        assert!(err_of(r#"{"classes": [{"id": "a"}], "mappings": [{"table": "t", "class": "ghost", "key": "id"}]}"#).contains("unknown class"));
        assert!(err_of(r#"{"classes": [{"id": "a"}], "mappings": [{"table": "t", "class": "a", "key": "id", "properties": {"c": "p"}}]}"#).contains("does not carry"));
        let wrong_direction = r#"{"classes": [{"id": "a"}, {"id": "b", "key": "k", "properties": ["k"]}], "properties": [{"id": "k", "type": "string"}], "relations": [{"id": "r", "domain": "b", "range": "a"}], "mappings": [{"table": "t", "class": "a", "key": "id", "relations": [{"relation": "r", "column": "c", "target_class": "b", "target_key": "k"}]}]}"#;
        assert!(err_of(wrong_direction).contains("goes b -> a"));
        assert!(err_of("{\"classes\": [").contains("does not parse"));
        assert!(err_of(&c(r#"{"id": "a", "colour": "red"}"#)).contains("does not parse"));
    }

    #[test]
    fn the_builtin_default_is_valid_and_renders_for_the_prompt() {
        let default = Ontology::builtin_default();
        assert!(default.validate().is_ok());
        let text = default.render_for_prompt();
        assert!(text.contains("person: entity {email, title}"), "{text}");
        assert!(text.contains("works_at: person -> organization"));
        assert!(text.contains("mentions: entity -> entity"));
        assert!(!text.contains("mapped tables"));
    }

    #[test]
    fn the_capped_rendering_counts_what_it_leaves_out() {
        let default = Ontology::builtin_default();
        let capped = default.render_capped(2);
        assert!(capped.contains("person: entity"), "{capped}");
        assert!(!capped.contains("concept: entity"), "{capped}");
        assert!(
            capped.contains("and 5 more classes; describe_class shows any class by id"),
            "{capped}"
        );
        assert!(capped.contains("and 3 more relations"), "{capped}");
        // `mentions` is implicit and always named, cap or no cap.
        assert!(capped.contains("mentions: entity -> entity"), "{capped}");
        // Uncapped, nothing is counted away.
        let full = default.render_capped(usize::MAX);
        assert_eq!(full, default.render_for_prompt());
        assert!(!full.contains("more classes"), "{full}");
    }

    #[test]
    fn relations_and_subclasses_follow_inheritance() {
        let ontology = Ontology::builtin_default();
        let (from, to) = ontology.relations_of("person");
        let from: Vec<&str> = from.iter().map(|r| r.id.as_str()).collect();
        let to: Vec<&str> = to.iter().map(|r| r.id.as_str()).collect();
        // `works_at` is the class's own; the `entity`-domain ones are inherited.
        assert!(
            from.contains(&"works_at") && from.contains(&"located_in"),
            "{from:?}"
        );
        assert!(!from.contains(&"produced_by"), "{from:?}");
        assert!(to.contains(&"part_of"), "{to:?}");
        assert_eq!(ontology.subclasses("entity").len(), ontology.classes.len());
        assert!(ontology.subclasses("person").is_empty());
        assert!(ontology.mapping_for("person").is_none());
    }

    #[test]
    fn diff_reports_added_removed_and_changed_ids() {
        let base = Ontology::from_json(INSURANCE).unwrap_or_else(|e| fail(&e.to_string()));
        let mut next = base.clone();
        next.version = OntologyVersion::new(2);
        next.classes.retain(|c| c.id != "vendor");
        next.classes.push(Class {
            id: String::from("adjuster"),
            parent: String::from("organization"),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        });
        if let Some(status) = next.properties.iter_mut().find(|p| p.id == "status") {
            status.values.push(String::from("under_review"));
        }
        let diff = next.diff(&base);
        assert_eq!(diff.classes.added, ["adjuster"]);
        assert_eq!(diff.classes.removed, ["vendor"]);
        assert_eq!(diff.properties.changed, ["status"]);
        assert!(diff.relations.is_empty() && diff.mappings.is_empty());
        let text = diff.to_string();
        assert!(
            text.contains("+ adjuster") && text.contains("- vendor") && text.contains("~ status"),
            "{text}"
        );
        assert!(base.diff(&base).is_empty());
    }
}
