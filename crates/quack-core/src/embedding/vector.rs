//! The value types a stored embedding is made of: its width, the vector
//! itself, and the fingerprint of the profile it was made under.

use std::fmt;
use std::ops::Deref;

use duckdb::ToSql;
use duckdb::types::ToSqlOutput;
use serde::{Deserialize, Serialize};

use crate::error::Error;

/// How many numbers a vector has: a model's output width, and the width of
/// a workspace's vector columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Dimension(u32);

impl Dimension {
    #[must_use]
    pub const fn new(width: u32) -> Self {
        Self(width)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Whether `len` numbers are this many.
    #[must_use]
    pub fn fits(self, len: usize) -> bool {
        usize::try_from(self.0).is_ok_and(|width| width == len)
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A vector of `expected` numbers turned out to have `actual`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WidthMismatch {
    pub expected: Dimension,
    pub actual: usize,
}

impl WidthMismatch {
    /// The configuration error for `model` answering at the wrong width,
    /// with the setting to change.
    #[must_use]
    pub fn for_model(self, model: &str) -> Error {
        let Self { expected, actual } = self;
        Error::Config(format!(
            "{model} returned {actual}-dimensional vectors but its provider's \
             embedding_dimension is {expected}; set embedding_dimension = {actual}"
        ))
    }
}

/// An embedding vector whose width was checked when it was made.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    values: Vec<f32>,
    dimension: Dimension,
}

impl Vector {
    /// `values` as a vector of `dimension` numbers.
    ///
    /// # Errors
    ///
    /// Returns the mismatch when `values` has another length.
    pub fn new(values: Vec<f32>, dimension: Dimension) -> Result<Self, WidthMismatch> {
        if dimension.fits(values.len()) {
            Ok(Self { values, dimension })
        } else {
            Err(WidthMismatch {
                expected: dimension,
                actual: values.len(),
            })
        }
    }

    #[must_use]
    pub fn dimension(&self) -> Dimension {
        self.dimension
    }
}

impl Deref for Vector {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        &self.values
    }
}

/// The identity of an embedding profile, stored beside every vector made
/// under it: the SHA-256 of the profile's JSON.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Fingerprint(String);

impl Fingerprint {
    pub(super) fn new(hex: String) -> Self {
        Self(hex)
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl ToSql for Fingerprint {
    fn to_sql(&self) -> duckdb::Result<ToSqlOutput<'_>> {
        self.0.to_sql()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_vector_is_made_only_at_its_width() {
        let four = Dimension::new(4);
        let vector = Vector::new(vec![0.5; 4], four);
        assert_eq!(vector.as_ref().map(|v| v.len()), Ok(4));
        assert_eq!(vector.map(|v| v.dimension()), Ok(four));
        assert_eq!(
            Vector::new(vec![0.5; 3], four),
            Err(WidthMismatch {
                expected: four,
                actual: 3
            })
        );
        assert!(four.fits(4));
        assert!(!four.fits(5));
    }
}
