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
pub mod edit;
pub mod induction;
pub mod store;

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use duckdb::types::{FromSql, FromSqlError, FromSqlResult, ToSqlOutput, ValueRef};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Deserializer, Serialize};

use crate::error::{Error, Result};
use crate::ids::{ClassId, RelationId};
use crate::storage::workspace::quote_ident;
use crate::text::OneLine;
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
    pub fn for_item(self, kind: impl std::fmt::Display) -> Error {
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
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    utoipa::ToSchema,
)]
#[serde(transparent)]
#[schema(value_type = u32, minimum = 1)]
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

/// A kind of entity: graph nodes are typed by one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Class {
    /// `snake_case`, unique among classes.
    pub id: ClassId,
    /// The class this one specializes; the implicit root `entity` when absent.
    #[serde(default = "root_class")]
    pub parent: ClassId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The property that identifies an instance (a policy number).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// Ids of the properties an instance carries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub properties: Vec<String>,
}

fn root_class() -> ClassId {
    ClassId::from(ROOT_CLASS)
}

/// How many items a `take(limit)` left out, or `None` when it left none.
fn hidden(total: usize, limit: usize) -> Option<usize> {
    total.checked_sub(limit).filter(|rest| *rest > 0)
}

/// A kind of edge, from an entity of `domain` to one of `range`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Relation {
    /// `snake_case`, unique among relations.
    pub id: RelationId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The class of the edge's source (or one of its subclasses).
    pub domain: ClassId,
    /// The class of the edge's target (or one of its subclasses).
    pub range: ClassId,
}

/// The type of a property's values.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, utoipa::ToSchema,
)]
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

/// A typed attribute that classes carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Property {
    /// Unique among properties.
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(rename = "type")]
    pub kind: PropertyType,
    /// Allowed values for `enum`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<String>,
    /// What a value means, shown beside every column mapped to it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The unit a number is in (`cents`, `USD`, `kg`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// Other words people use for it, which the table search matches.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub synonyms: Vec<String>,
}

impl Property {
    /// A property with no label, description, unit, or synonyms.
    #[must_use]
    pub fn new(id: impl Into<String>, kind: PropertyType, values: Vec<String>) -> Self {
        Self {
            id: id.into(),
            label: None,
            kind,
            values,
            description: None,
            unit: None,
            synonyms: Vec::new(),
        }
    }
}

/// A named calculation over one table: a SQL expression such as
/// `sum(amount) / 100.0`, checked as a read of that table when the
/// ontology is saved, so the agent and people compute it one way.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Measure {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The table the expression reads.
    pub table: String,
    /// What goes after `SELECT` and before `FROM table`.
    pub expression: String,
}

impl Measure {
    /// The statement that checks the expression: it must parse as one read
    /// of its table, and plan.
    #[must_use]
    pub fn check_statement(&self) -> String {
        format!(
            "SELECT {} AS measure FROM {}",
            self.expression,
            quote_ident(&self.table)
        )
    }
}

/// `- revenue = sum(amount) / 100.0: net revenue in dollars`.
impl std::fmt::Display for Measure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} = {}", self.id, OneLine(&self.expression))?;
        if let Some(description) = &self.description {
            write!(f, ": {}", OneLine(description))?;
        }
        Ok(())
    }
}

/// One foreign-key-like column of a mapped table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct MappingRelation {
    /// The relation each row's edge takes.
    pub relation: RelationId,
    /// The column holding the target's key.
    pub column: String,
    pub target_class: ClassId,
    /// The target class's key property the column's values match.
    pub target_key: String,
}

/// How a table's rows become nodes and edges (design doc 6.3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Mapping {
    /// The workspace table whose rows become nodes.
    pub table: String,
    /// The class each row's node takes.
    pub class: ClassId,
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
#[derive(
    Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema, utoipa::ToSchema,
)]
#[serde(deny_unknown_fields)]
pub struct Ontology {
    /// The stored version this was read from; `None` for one not yet saved.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "OntologyVersion::zero_as_none"
    )]
    #[schemars(with = "Option<u32>")]
    pub version: Option<OntologyVersion>,
    #[serde(default)]
    pub classes: Vec<Class>,
    #[serde(default)]
    pub relations: Vec<Relation>,
    #[serde(default)]
    pub properties: Vec<Property>,
    /// How tables' rows become nodes and edges.
    #[serde(default)]
    pub mappings: Vec<Mapping>,
    /// Named calculations over one table each.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub measures: Vec<Measure>,
}

/// The relations one class takes part in.
#[derive(Debug, Clone)]
pub struct ClassRelations<'a> {
    /// Those the class is the domain of.
    pub from: Vec<&'a Relation>,
    /// Those the class is the range of.
    pub to: Vec<&'a Relation>,
}

/// Class and relation ids to rename, each old id to its new one: what an
/// accepted candidate's rename or merge decision gives the proposals after
/// it, and what a save carries to move the graph's nodes and edges with
/// the ids (design doc 6.3).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IdRenames {
    pub classes: BTreeMap<ClassId, ClassId>,
    pub relations: BTreeMap<RelationId, RelationId>,
}

impl IdRenames {
    /// One id of `kind`, from `old` to `new`.
    ///
    /// # Errors
    ///
    /// Returns an `Ontology` error for a property or a mapping: the graph
    /// is keyed by class and relation ids only.
    pub fn one(kind: ItemKind, old: &str, new: &str) -> Result<Self> {
        let mut renames = Self::default();
        match kind {
            ItemKind::Class => {
                renames
                    .classes
                    .insert(ClassId::from(old), ClassId::from(new));
            }
            ItemKind::Relation => {
                renames
                    .relations
                    .insert(RelationId::from(old), RelationId::from(new));
            }
            ItemKind::Property | ItemKind::Mapping => {
                return Err(Error::Ontology(format!(
                    "a {kind} cannot be renamed: only a {} or a {} id can",
                    ItemKind::Class,
                    ItemKind::Relation
                )));
            }
        }
        Ok(renames)
    }

    /// The id a class had before these renames: its own unless it is the
    /// new id of one.
    pub(crate) fn class_before<'a>(&'a self, id: &'a str) -> &'a str {
        self.classes
            .iter()
            .find(|(_, new)| new.as_str() == id)
            .map_or(id, |(old, _)| old.as_str())
    }

    /// The id a relation had before these renames.
    pub(crate) fn relation_before<'a>(&'a self, id: &'a str) -> &'a str {
        self.relations
            .iter()
            .find(|(_, new)| new.as_str() == id)
            .map_or(id, |(old, _)| old.as_str())
    }

    fn move_class(&self, id: &mut ClassId) {
        if let Some(new) = self.classes.get(id.as_str()) {
            id.clone_from(new);
        }
    }

    fn move_relation(&self, id: &mut RelationId) {
        if let Some(new) = self.relations.get(id.as_str()) {
            id.clone_from(new);
        }
    }

    pub(crate) fn rename_class(&self, class: &mut Class) {
        self.move_class(&mut class.id);
        self.move_class(&mut class.parent);
    }

    pub(crate) fn rename_relation(&self, relation: &mut Relation) {
        self.move_relation(&mut relation.id);
        self.move_class(&mut relation.domain);
        self.move_class(&mut relation.range);
    }

    pub(crate) fn rename_mapping(&self, mapping: &mut Mapping) {
        self.move_class(&mut mapping.class);
        for link in &mut mapping.relations {
            self.move_relation(&mut link.relation);
            self.move_class(&mut link.target_class);
        }
    }

    /// The new id of the class named `owner`, for the places that hold a
    /// class id as plain text.
    pub(crate) fn rename_owner(&self, owner: &mut String) {
        if let Some(new) = self.classes.get(owner.as_str()) {
            new.as_str().clone_into(owner);
        }
    }

    /// Every old id must be one `ontology` defines, renamed to another
    /// id. A new id the ontology already has is left to
    /// [`Ontology::validate`], which refuses the renamed ontology for
    /// declaring it twice.
    ///
    /// # Errors
    ///
    /// Returns an `Ontology` error naming the first id that breaks this.
    pub(crate) fn check(&self, ontology: &Ontology) -> Result<()> {
        let classes = self.classes.iter().map(|(old, new)| {
            let defined = ontology.class(old.as_str()).is_some();
            (ItemKind::Class, old.as_str(), new.as_str(), defined)
        });
        let relations = self.relations.iter().map(|(old, new)| {
            let defined = ontology.relation(old.as_str()).is_some();
            (ItemKind::Relation, old.as_str(), new.as_str(), defined)
        });
        for (kind, old, new, defined) in classes.chain(relations) {
            if !defined {
                return Err(Error::Ontology(format!("no {kind} '{old}' to rename")));
            }
            if old == new {
                return Err(Error::Ontology(format!(
                    "{kind} '{old}' already has that id"
                )));
            }
        }
        Ok(())
    }
}

/// `class vendor to supplier, relation ships_to to delivers_to`.
impl std::fmt::Display for IdRenames {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let classes = self
            .classes
            .iter()
            .map(|(old, new)| format!("{} {old} to {new}", ItemKind::Class));
        let relations = self
            .relations
            .iter()
            .map(|(old, new)| format!("{} {old} to {new}", ItemKind::Relation));
        f.write_str(&classes.chain(relations).collect::<Vec<_>>().join(", "))
    }
}

impl Ontology {
    /// The JSON Schema of the interchange form, as `docs/ontology.schema.json`
    /// publishes it: what import, `PUT .../ontology`, and the ontology page
    /// accept.
    #[must_use]
    pub fn json_schema() -> Schema {
        schema_for!(Self)
    }

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
    pub fn ancestry(&self, class_id: &str) -> Vec<ClassId> {
        let mut chain = vec![ClassId::from(class_id)];
        let mut current = ClassId::from(class_id);
        while current != ROOT_CLASS {
            let Some(class) = self.class(current.as_str()) else {
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
        self.is_subclass_of(source_class, relation.domain.as_str())
            && self.is_subclass_of(target_class, relation.range.as_str())
    }

    /// The classes whose parent is this one, nearest first.
    #[must_use]
    pub fn subclasses(&self, class_id: &str) -> Vec<&Class> {
        self.classes
            .iter()
            .filter(|c| c.parent == class_id)
            .collect()
    }

    /// The relations a class can take part in, inherited ones included.
    #[must_use]
    pub fn relations_of(&self, class_id: &str) -> ClassRelations<'_> {
        let mut out = ClassRelations {
            from: Vec::new(),
            to: Vec::new(),
        };
        for relation in &self.relations {
            if self.is_subclass_of(class_id, relation.domain.as_str()) {
                out.from.push(relation);
            }
            if self.is_subclass_of(class_id, relation.range.as_str()) {
                out.to.push(relation);
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

    /// The measures defined over `table`.
    #[must_use]
    pub fn measures_on(&self, table: &str) -> Vec<&Measure> {
        self.measures.iter().filter(|m| m.table == table).collect()
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
    pub fn class_and_descendants(&self, class_id: &str) -> Vec<ClassId> {
        let mut out = vec![ClassId::from(class_id)];
        for class in &self.classes {
            if class.id != class_id && self.is_subclass_of(class.id.as_str(), class_id) {
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
            .filter_map(|id| self.class(id.as_str()))
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
        self.validate_mappings()?;
        self.validate_measures()
    }

    fn validate_measures(&self) -> Result<()> {
        let mut seen = BTreeSet::new();
        for measure in &self.measures {
            SnakeId::try_from(measure.id.as_str()).map_err(|e| e.for_item("measure"))?;
            if !seen.insert(measure.id.as_str()) {
                return Err(Error::Ontology(format!(
                    "measure '{}' is declared twice",
                    measure.id
                )));
            }
            if measure.table.trim().is_empty() || measure.expression.trim().is_empty() {
                return Err(Error::Ontology(format!(
                    "measure '{}' needs a table and an expression",
                    measure.id
                )));
            }
        }
        Ok(())
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
            if class.parent != ROOT_CLASS && self.class(class.parent.as_str()).is_none() {
                return Err(Error::Ontology(format!(
                    "class '{}' has unknown parent '{}'",
                    class.id, class.parent
                )));
            }
            if self.ancestry(class.id.as_str()).last().map(ClassId::as_str) != Some(ROOT_CLASS) {
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
                && !self.class_properties(class.id.as_str()).contains(key)
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
                if class != ROOT_CLASS && self.class(class.as_str()).is_none() {
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
            if self.class(mapping.class.as_str()).is_none() {
                return Err(Error::Ontology(format!(
                    "mapping for '{}' names unknown class '{}'",
                    mapping.table, mapping.class
                )));
            }
            let allowed = self.class_properties(mapping.class.as_str());
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
        let Some(relation) = self.relation(link.relation.as_str()) else {
            return Err(Error::Ontology(format!(
                "mapping for '{}' uses unknown relation '{}'",
                mapping.table, link.relation
            )));
        };
        if self.class(link.target_class.as_str()).is_none() {
            return Err(Error::Ontology(format!(
                "mapping for '{}' targets unknown class '{}'",
                mapping.table, link.target_class
            )));
        }
        if !self.is_subclass_of(mapping.class.as_str(), relation.domain.as_str())
            || !self.is_subclass_of(link.target_class.as_str(), relation.range.as_str())
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
            .class_properties(link.target_class.as_str())
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
            let props = self.class_properties(class.id.as_str());
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
        out.measures.sort_by(|a, b| a.id.cmp(&b.id));
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
            property.synonyms.sort();
            property.synonyms.dedup();
        }
        out
    }

    /// This ontology, unsaved and not yet validated, with `renames`
    /// applied to every place it names a class or a relation: ids,
    /// parents, domains and ranges, and table mappings.
    ///
    /// # Errors
    ///
    /// Returns an `Ontology` error when an old id is not defined or is
    /// renamed to itself.
    pub(crate) fn renamed(&self, renames: &IdRenames) -> Result<Self> {
        renames.check(self)?;
        let mut out = self.clone();
        out.version = None;
        for class in &mut out.classes {
            renames.rename_class(class);
        }
        for relation in &mut out.relations {
            renames.rename_relation(relation);
        }
        for mapping in &mut out.mappings {
            renames.rename_mapping(mapping);
        }
        Ok(out)
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
            measures: Changes::between(&older.measures, &this.measures, |m| m.id.as_str()),
        }
    }

    /// The built-in general ontology installed as version 1 (design doc 6.3).
    #[must_use]
    pub fn builtin_default() -> Self {
        let class = |id: &str, parent: &str, properties: &[&str]| Class {
            id: ClassId::from(id.to_owned()),
            parent: ClassId::from(parent.to_owned()),
            label: None,
            description: None,
            key: None,
            properties: properties.iter().map(|p| (*p).to_owned()).collect(),
        };
        let relation = |id: &str, domain: &str, range: &str| Relation {
            id: RelationId::from(id.to_owned()),
            label: None,
            description: None,
            domain: ClassId::from(domain.to_owned()),
            range: ClassId::from(range.to_owned()),
        };
        let property = |id: &str, kind: PropertyType| Property::new(id, kind, Vec::new());
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
            measures: Vec::new(),
        }
    }
}

/// Ids added, removed, or changed for one kind.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub struct OntologyDiff {
    pub from: Option<OntologyVersion>,
    pub to: Option<OntologyVersion>,
    pub classes: Changes,
    pub relations: Changes,
    pub properties: Changes,
    pub mappings: Changes,
    #[serde(default)]
    pub measures: Changes,
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
                children(ontology, class.id.as_str(), depth.saturating_add(1), lines);
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
            let unit = p
                .unit
                .as_deref()
                .map_or(String::new(), |u| format!(" [{u}]"));
            let description = p
                .description
                .as_deref()
                .map_or(String::new(), |d| format!(": {d}"));
            lines.push(format!(
                "  - {}: {}{values}{unit}{description}",
                p.id,
                p.kind.as_str()
            ));
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
        if !self.measures.is_empty() {
            lines.push(String::from("measures:"));
            for m in &self.measures {
                lines.push(format!("  - {m} (on {})", m.table));
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
            && self.measures.is_empty()
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
            ("measures", &self.measures),
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
mod tests;
