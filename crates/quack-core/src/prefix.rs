//! Records named by a unique prefix of their id, the way every interface
//! lets people type one: v7 ids are long, and the first few characters
//! usually tell them apart.

use crate::error::{Error, Record, Result};

/// What an id prefix names among some records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixMatch<T> {
    None,
    One(T),
    Many(Vec<T>),
}

impl<T> PrefixMatch<T> {
    /// The `items` whose `id` starts with `prefix`; one whose id is
    /// `prefix` exactly wins outright.
    pub fn of(items: impl IntoIterator<Item = T>, prefix: &str, id: impl Fn(&T) -> &str) -> Self {
        let mut matches: Vec<T> = items
            .into_iter()
            .filter(|item| id(item).starts_with(prefix))
            .collect();
        if let Some(exact) = matches.iter().position(|item| id(item) == prefix) {
            return Self::One(matches.swap_remove(exact));
        }
        match matches.len() {
            0 => Self::None,
            1 => matches.pop().map_or(Self::None, Self::One),
            _ => Self::Many(matches),
        }
    }

    /// The one match.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when nothing matched and
    /// [`Error::Ambiguous`] when several did, both naming `record`.
    pub fn one(self, record: Record, prefix: &str) -> Result<T> {
        match self {
            Self::One(item) => Ok(item),
            Self::None => Err(record.missing(prefix)),
            Self::Many(items) => Err(Error::Ambiguous {
                record,
                prefix: prefix.to_owned(),
                count: items.len(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(prefix: &str) -> PrefixMatch<&'static str> {
        PrefixMatch::of(["01a0-first", "01a1-second", "01b0-third"], prefix, |id| id)
    }

    #[test]
    fn a_prefix_names_one_several_or_none() {
        assert_eq!(find("01b"), PrefixMatch::One("01b0-third"));
        assert_eq!(
            find("01a"),
            PrefixMatch::Many(vec!["01a0-first", "01a1-second"])
        );
        assert_eq!(find("02"), PrefixMatch::None);
        assert_eq!(
            PrefixMatch::of(["ab", "abc"], "ab", |id| id),
            PrefixMatch::One("ab"),
            "an exact id wins over the longer ones it prefixes"
        );
    }

    #[test]
    fn one_names_the_record_when_it_cannot_pick() {
        assert_eq!(
            find("01b").one(Record::Document, "01b").ok(),
            Some("01b0-third")
        );
        let missing = find("02")
            .one(Record::Document, "02")
            .err()
            .map(|e| e.to_string());
        assert_eq!(missing.as_deref(), Some("document '02' does not exist"));
        let many = find("01a")
            .one(Record::Session, "01a")
            .err()
            .map(|e| e.to_string());
        assert_eq!(
            many.as_deref(),
            Some("'01a' matches 2 sessions; use more of the id")
        );
    }
}
