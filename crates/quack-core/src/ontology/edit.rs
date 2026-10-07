//! One change to the current ontology made without its JSON: a property's
//! meaning, or a measure added, changed, or removed. The change applies to
//! a copy, and `store::save` validates it and stores the next version.

use std::fmt;

use serde::Serialize;

use crate::error::{Error, Result};
use crate::storage::workspace::ColumnMeaning;

use super::{Measure, Ontology};

/// What the ontology page's forms change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "edit", rename_all = "snake_case")]
pub enum Edit {
    /// Set a property's description, unit, and synonyms; an empty field
    /// clears it.
    Describe {
        property: String,
        meaning: ColumnMeaning,
    },
    AddMeasure(Measure),
    /// Replace the measure with the same id.
    ChangeMeasure(Measure),
    RemoveMeasure {
        id: String,
    },
}

impl Edit {
    /// Apply the change to `ontology`.
    ///
    /// # Errors
    ///
    /// Returns an `Ontology` error when the property or measure it names is
    /// not defined, or a measure it adds already is.
    pub fn apply(self, ontology: &mut Ontology) -> Result<()> {
        match self {
            Self::Describe { property, meaning } => {
                let found = ontology
                    .properties
                    .iter_mut()
                    .find(|p| p.id == property)
                    .ok_or_else(|| Error::Ontology(format!("no property '{property}'")))?;
                found.description = meaning.description;
                found.unit = meaning.unit;
                found.synonyms = meaning.synonyms;
            }
            Self::AddMeasure(measure) => {
                if ontology.measures.iter().any(|m| m.id == measure.id) {
                    return Err(Error::Ontology(format!(
                        "a measure named '{}' already exists",
                        measure.id
                    )));
                }
                ontology.measures.push(measure);
            }
            Self::ChangeMeasure(measure) => {
                let found = ontology
                    .measures
                    .iter_mut()
                    .find(|m| m.id == measure.id)
                    .ok_or_else(|| Error::Ontology(format!("no measure '{}'", measure.id)))?;
                *found = measure;
            }
            Self::RemoveMeasure { id } => {
                let before = ontology.measures.len();
                ontology.measures.retain(|m| m.id != id);
                if ontology.measures.len() == before {
                    return Err(Error::Ontology(format!("no measure '{id}'")));
                }
            }
        }
        Ok(())
    }
}

/// `described property amount`, the version note's start.
impl fmt::Display for Edit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Describe { property, .. } => write!(f, "described property {property}"),
            Self::AddMeasure(measure) => write!(f, "added measure {}", measure.id),
            Self::ChangeMeasure(measure) => write!(f, "changed measure {}", measure.id),
            Self::RemoveMeasure { id } => write!(f, "removed measure {id}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::{Property, PropertyType};

    fn ontology() -> Ontology {
        let mut ontology = Ontology::builtin_default();
        ontology
            .properties
            .push(Property::new("amount", PropertyType::Number, Vec::new()));
        ontology
    }

    fn measure(id: &str, expression: &str) -> Measure {
        Measure {
            id: String::from(id),
            description: None,
            table: String::from("orders"),
            expression: String::from(expression),
        }
    }

    #[test]
    fn describing_a_property_sets_and_clears_its_meaning() {
        let mut ontology = ontology();
        let meaning = ColumnMeaning {
            description: Some(String::from("order total")),
            unit: Some(String::from("cents")),
            synonyms: vec![String::from("total")],
        };
        let edit = Edit::Describe {
            property: String::from("amount"),
            meaning: meaning.clone(),
        };
        assert_eq!(edit.to_string(), "described property amount");
        assert!(edit.apply(&mut ontology).is_ok());
        let amount = ontology.properties.iter().find(|p| p.id == "amount");
        assert_eq!(amount.and_then(ColumnMeaning::of), Some(meaning));

        let cleared = Edit::Describe {
            property: String::from("amount"),
            meaning: ColumnMeaning {
                description: None,
                unit: None,
                synonyms: Vec::new(),
            },
        };
        assert!(cleared.apply(&mut ontology).is_ok());
        let amount = ontology.properties.iter().find(|p| p.id == "amount");
        assert_eq!(amount.and_then(ColumnMeaning::of), None);
    }

    #[test]
    fn an_unknown_property_or_measure_is_refused_by_name() {
        let mut ontology = ontology();
        let refused =
            |edit: Edit, ontology: &mut Ontology| edit.apply(ontology).err().map(|e| e.to_string());
        let describe = Edit::Describe {
            property: String::from("nope"),
            meaning: ColumnMeaning {
                description: None,
                unit: None,
                synonyms: Vec::new(),
            },
        };
        assert!(refused(describe, &mut ontology).is_some_and(|e| e.contains("no property 'nope'")));
        assert!(
            refused(Edit::ChangeMeasure(measure("nope", "1")), &mut ontology)
                .is_some_and(|e| e.contains("no measure 'nope'"))
        );
        assert!(
            refused(
                Edit::RemoveMeasure {
                    id: String::from("nope")
                },
                &mut ontology
            )
            .is_some_and(|e| e.contains("no measure 'nope'"))
        );
    }

    #[test]
    fn measures_are_added_changed_and_removed_by_id() {
        let mut ontology = ontology();
        assert!(
            Edit::AddMeasure(measure("revenue", "sum(amount)"))
                .apply(&mut ontology)
                .is_ok()
        );
        let again = Edit::AddMeasure(measure("revenue", "count(*)")).apply(&mut ontology);
        assert!(
            again
                .err()
                .is_some_and(|e| e.to_string().contains("already exists"))
        );
        assert!(
            Edit::ChangeMeasure(measure("revenue", "sum(amount) / 100.0"))
                .apply(&mut ontology)
                .is_ok()
        );
        assert_eq!(
            ontology.measures,
            vec![measure("revenue", "sum(amount) / 100.0")]
        );
        assert!(
            Edit::RemoveMeasure {
                id: String::from("revenue")
            }
            .apply(&mut ontology)
            .is_ok()
        );
        assert!(ontology.measures.is_empty());
    }
}
