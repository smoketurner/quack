//! Graph searches as every interface asks for them: a query normalized
//! once (trimmed, blanks absent, hops defaulted), its ids checked against
//! the ontology, and its entry points resolved the same way whether the
//! CLI, the API, MCP, the web console, the terminal, or the agent asked.

use std::fmt;

use rig::embeddings::EmbeddingModel;
use serde::Serialize;

use super::traverse::{self, Hops};
use super::{GraphOptions, GraphResult};
use crate::embedding::{Embedder, Vector};
use crate::error::{Error, Result};
use crate::ontology::{Ontology, store as ontology_store};
use crate::storage::workspace::WorkspaceDb;
use crate::text::NonBlankText;

/// Text a caller gave, trimmed, and absent when blank.
fn given(text: Option<&str>) -> Option<String> {
    text.and_then(str::non_blank).map(str::to_owned)
}

/// A search of the graph: the neighborhood of an entity (optionally of one
/// class, along one relation), or every entity of a class. It serializes as
/// the audit detail of the search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GraphQuery {
    pub entity: Option<String>,
    pub class: Option<String>,
    pub relation: Option<String>,
    pub hops: Hops,
}

impl GraphQuery {
    /// The query a caller asked for, each field trimmed and a blank one
    /// absent, and hops defaulted.
    ///
    /// # Errors
    ///
    /// Returns an error when there is neither an entity nor a class.
    pub fn new(
        entity: Option<&str>,
        class: Option<&str>,
        relation: Option<&str>,
        hops: Option<u32>,
    ) -> Result<Self> {
        let query = Self {
            entity: given(entity),
            class: given(class),
            relation: given(relation),
            hops: Hops::neighborhood(hops),
        };
        if query.entity.is_none() && query.class.is_none() {
            return Err(Error::Analysis(String::from(
                "give an entity to start from, a class to list, or both",
            )));
        }
        Ok(query)
    }

    /// The entity's embedding for fuzzy entry: `None` without an entity or
    /// an embedding model.
    ///
    /// # Errors
    ///
    /// Returns the model's error rather than falling back to exact matches
    /// without saying so.
    pub async fn embedding<M: EmbeddingModel>(
        &self,
        embedder: Option<&Embedder<M>>,
    ) -> Result<Option<Vector>> {
        match (self.entity.as_deref(), embedder) {
            (Some(entity), Some(embedder)) => embedder.similarity(entity).await.map(Some),
            (Some(_) | None, None) | (None, Some(_)) => Ok(None),
        }
    }

    /// Run it: refuse class and relation ids the ontology does not define,
    /// then walk out from the entity or list the class. An entity that
    /// resolves to nothing gives an empty result; [`Self::suggestions`]
    /// has the labels to offer instead.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown id, or when a query fails.
    pub fn run(
        &self,
        db: &WorkspaceDb,
        embedding: Option<&Vector>,
        options: &GraphOptions,
    ) -> Result<GraphResult> {
        let ontology = ontology_store::current(db)?;
        if let Some(class) = self.class.as_deref() {
            OntologyId::Class(class).check(ontology.as_ref())?;
        }
        if let Some(relation) = self.relation.as_deref() {
            OntologyId::Relation(relation).check(ontology.as_ref())?;
        }
        match self.entity.as_deref() {
            Some(entity) => {
                let roots = traverse::resolve_entry(db, entity, self.class.as_deref(), embedding)?;
                traverse::neighborhood(db, &roots, self.hops, self.relation.as_deref(), options)
            }
            None => traverse::by_class(
                db,
                ontology.as_ref(),
                self.class.as_deref().unwrap_or_default(),
                options.max_nodes,
                options,
            ),
        }
    }

    /// Labels close to the entity, to offer when the search found nothing.
    ///
    /// # Errors
    ///
    /// Returns an error if the lookup fails.
    pub fn suggestions(&self, db: &WorkspaceDb, embedding: Option<&Vector>) -> Result<Vec<String>> {
        match self.entity.as_deref() {
            Some(entity) => {
                traverse::suggest_entities(db, entity, self.class.as_deref(), embedding)
            }
            None => Ok(Vec::new()),
        }
    }
}

/// The shortest chain of relations between two entities. It serializes as
/// the audit detail of the search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PathQuery {
    pub from: String,
    pub to: String,
    pub max_hops: Hops,
}

/// The embeddings of a path's two ends, for fuzzy entry.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PathEnds {
    pub from: Option<Vector>,
    pub to: Option<Vector>,
}

impl PathQuery {
    /// The path a caller asked for, both ends trimmed, and the longest
    /// path defaulted.
    ///
    /// # Errors
    ///
    /// Returns an error when either end is blank.
    pub fn new(from: &str, to: &str, max_hops: Option<u32>) -> Result<Self> {
        let (Some(from), Some(to)) = (given(Some(from)), given(Some(to))) else {
            return Err(Error::Analysis(String::from("give both entities")));
        };
        Ok(Self {
            from,
            to,
            max_hops: Hops::path(max_hops),
        })
    }

    /// Both ends' embeddings, when there is an embedding model.
    ///
    /// # Errors
    ///
    /// Returns the model's error.
    pub async fn embeddings<M: EmbeddingModel>(
        &self,
        embedder: Option<&Embedder<M>>,
    ) -> Result<PathEnds> {
        let Some(embedder) = embedder else {
            return Ok(PathEnds::default());
        };
        Ok(PathEnds {
            from: Some(embedder.similarity(&self.from).await?),
            to: Some(embedder.similarity(&self.to).await?),
        })
    }

    /// Run it: resolve both ends, then search between them. An empty
    /// result means no path within `max_hops`.
    ///
    /// # Errors
    ///
    /// An end that resolves to no entity is an [`UnknownEntity`] error
    /// naming the closest labels; a failed query is an error too.
    pub fn run(
        &self,
        db: &WorkspaceDb,
        ends: &PathEnds,
        options: &GraphOptions,
    ) -> Result<GraphResult> {
        let from = traverse::resolve_entry(db, &self.from, None, ends.from.as_ref())?;
        let to = traverse::resolve_entry(db, &self.to, None, ends.to.as_ref())?;
        match (from.first(), to.first()) {
            (Some(a), Some(b)) => traverse::path(db, a, b, self.max_hops, options),
            (None, _) => Err(UnknownEntity::find(db, &self.from, ends.from.as_ref()).into()),
            (_, None) => Err(UnknownEntity::find(db, &self.to, ends.to.as_ref()).into()),
        }
    }
}

/// A name that resolved to no entity, with the closest labels, so the
/// caller can try again with a real one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownEntity {
    pub name: String,
    pub closest: Vec<String>,
}

impl UnknownEntity {
    /// `name`, which resolved to nothing, and the labels nearest it; a
    /// failed suggestion lookup offers none.
    #[must_use]
    pub fn find(db: &WorkspaceDb, name: &str, embedding: Option<&Vector>) -> Self {
        Self {
            name: name.to_owned(),
            closest: traverse::suggest_entities(db, name, None, embedding).unwrap_or_default(),
        }
    }
}

impl fmt::Display for UnknownEntity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no entity '{}' in the knowledge graph", self.name)?;
        if !self.closest.is_empty() {
            write!(f, "; the closest labels are: {}", self.closest.join(", "))?;
        }
        Ok(())
    }
}

impl From<UnknownEntity> for Error {
    fn from(unknown: UnknownEntity) -> Self {
        Self::Analysis(unknown.to_string())
    }
}

/// Ids past this many are counted rather than listed when an unknown id is
/// refused: an induced ontology can carry a class per table.
const LISTED_IDS: usize = 40;

/// `a, b, c, ... and N more`, so a refusal names what exists without
/// pasting a whole ontology into the reply.
pub struct Listed<'a>(pub &'a [&'a str]);

impl fmt::Display for Listed<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, id) in self.0.iter().take(LISTED_IDS).enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            f.write_str(id)?;
        }
        let hidden = self.0.len().saturating_sub(LISTED_IDS);
        if hidden > 0 {
            write!(f, ", ... and {hidden} more")?;
        }
        Ok(())
    }
}

/// A class or relation id a caller named.
#[derive(Debug, Clone, Copy)]
pub enum OntologyId<'a> {
    Class(&'a str),
    Relation(&'a str),
}

impl OntologyId<'_> {
    /// Refuse an id the ontology does not define, naming the ones it does:
    /// an id that matches nothing is an error the caller can correct, never
    /// an empty result that reads as "the workspace has nothing on this".
    ///
    /// # Errors
    ///
    /// Returns an error when there is no ontology or it lacks the id.
    pub fn check(self, ontology: Option<&Ontology>) -> Result<()> {
        let Some(ontology) = ontology else {
            return Err(Error::Analysis(String::from(match self {
                Self::Class(_) => {
                    "this workspace has no ontology, so it has no classes to search by"
                }
                Self::Relation(_) => {
                    "this workspace has no ontology, so it has no relations to follow"
                }
            })));
        };
        let (kind, plural, id, defined, ids) = match self {
            Self::Class(id) => (
                "class",
                "classes",
                id,
                ontology.defines_class(id),
                ontology.class_ids(),
            ),
            Self::Relation(id) => (
                "relation",
                "relations",
                id,
                ontology.defines_relation(id),
                ontology.relation_ids(),
            ),
        };
        if defined {
            return Ok(());
        }
        Err(Error::Analysis(format!(
            "no {kind} '{id}' in the ontology; the {plural} are: {}",
            Listed(&ids)
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::{MENTIONS_RELATION, ROOT_CLASS};

    #[test]
    fn a_query_is_trimmed_defaulted_and_needs_an_entity_or_a_class() {
        let query = GraphQuery::new(Some("  Acme "), Some(""), Some(" works_at "), None);
        assert_eq!(
            query.ok(),
            Some(GraphQuery {
                entity: Some(String::from("Acme")),
                class: None,
                relation: Some(String::from("works_at")),
                hops: Hops::NEIGHBORHOOD,
            })
        );
        assert!(GraphQuery::new(Some(" "), None, None, Some(3)).is_err());
        assert_eq!(
            GraphQuery::new(None, Some("person"), None, Some(0))
                .ok()
                .map(|q| q.hops),
            Some(Hops::new(1))
        );
        let path = PathQuery::new(" Acme ", "Kenya", None);
        assert_eq!(
            path.ok(),
            Some(PathQuery {
                from: String::from("Acme"),
                to: String::from("Kenya"),
                max_hops: Hops::PATH,
            })
        );
        assert!(PathQuery::new("Acme", "  ", None).is_err());
    }

    #[test]
    fn unknown_class_and_relation_ids_are_refused_with_the_real_ones() {
        let ontology = Ontology::builtin_default();
        let class = |id| OntologyId::Class(id).check(Some(&ontology));
        let relation = |id| OntologyId::Relation(id).check(Some(&ontology));
        assert!(class("organization").is_ok());
        // The root class and `mentions` are implicit: never declared, always valid.
        assert!(class(ROOT_CLASS).is_ok());
        assert!(relation(MENTIONS_RELATION).is_ok());
        assert!(relation("works_at").is_ok());

        let err = class("organisation")
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            err.contains("no class 'organisation'") && err.contains("organization"),
            "{err}"
        );
        let err = relation("employed_by")
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            err.contains("no relation 'employed_by'") && err.contains("works_at"),
            "{err}"
        );
        assert!(OntologyId::Class("organization").check(None).is_err());
        assert!(OntologyId::Relation("works_at").check(None).is_err());
    }

    #[test]
    fn long_id_lists_are_counted_rather_than_pasted() {
        let ids: Vec<String> = (0..(LISTED_IDS + 5)).map(|i| format!("c{i:03}")).collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let text = Listed(&refs).to_string();
        assert!(text.starts_with("c000, c001"), "{text}");
        assert!(text.ends_with("... and 5 more"), "{text}");
        assert!(!text.contains("c040"), "{text}");
        assert_eq!(Listed(&["a", "b"]).to_string(), "a, b");
    }
}
