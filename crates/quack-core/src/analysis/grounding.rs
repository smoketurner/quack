//! Grounding: `DuckDB` provides the answers, and the chat model only writes
//! the SQL and the tool calls that fetch them (design doc 7.7). A query's
//! filter values are held to the question ([`ValueLinks`]): a value of a
//! text column the question names (`Texas`, stored as `TEXAS` in
//! `events.state`) is the one a query filtering that column must use, never
//! a neighbour the model picked (`TENNESSEE`) or the question's own spelling
//! the column does not hold (`Texas`).
//!
//! An answer is shown only when the turn fetched something: a tool ran, or
//! the answer cites a passage `always_retrieve` put in the prompt. A turn
//! that ran nothing has no answer to show, whatever the model wrote; the
//! person is told what the workspace holds instead ([`nothing_fetched`]).

use std::collections::BTreeMap;

use crate::storage::profile::TableProfile;
use crate::storage::workspace::{Filters, ValueFilter};

/// Tables named in the note before the rest are counted.
const NOTE_TABLES: usize = 10;

/// What a turn that ran no query and no search answers with, in a workspace
/// holding `tables` and `documents` ready documents: the model's text is
/// not shown, since nothing it says came from the workspace.
#[must_use]
pub(crate) fn nothing_fetched(tables: &[String], documents: u64) -> String {
    let shown: Vec<&str> = tables
        .iter()
        .take(NOTE_TABLES)
        .map(String::as_str)
        .collect();
    let more = tables.len().saturating_sub(shown.len());
    let listed_tables = (!shown.is_empty()).then(|| {
        let rest = if more > 0 {
            format!(" and {more} more")
        } else {
            String::new()
        };
        format!(
            "the table{} {}{rest}",
            plural(tables.len()),
            shown.join(", ")
        )
    });
    let counted_documents = (documents > 0).then(|| {
        let s = if documents == 1 { "" } else { "s" };
        format!("{documents} document{s}")
    });
    let listed = match (listed_tables, counted_documents) {
        (Some(tables), Some(documents)) => format!("{tables}, and {documents}"),
        (Some(tables), None) => tables,
        (None, Some(documents)) => documents,
        (None, None) => String::from("no tables and no documents"),
    };
    format!(
        "quack ran no query and no search for this question, so there is no answer to show. \
         The workspace holds {listed}. Ask about them, naming the table or the document when \
         you can."
    )
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// A value of a text column that the question names.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Link {
    table: String,
    column: String,
    /// As the column holds it: `TEXAS`.
    value: String,
    /// As the question writes it: `Texas`.
    named: String,
}

/// The question's words matched to the values the workspace's text columns
/// hold (`ColumnProfile::values`, the columns with at most 50 distinct
/// values), found once when the turn starts.
#[derive(Debug, Clone, Default)]
pub(crate) struct ValueLinks {
    links: Vec<Link>,
    /// Every value of each linked column, by table and column.
    values: BTreeMap<(String, String), Vec<String>>,
}

impl ValueLinks {
    /// The values `question` names, matched as whole words ignoring case. A
    /// value shorter than three characters or without a letter is no name.
    pub(crate) fn find(profiles: &BTreeMap<String, TableProfile>, question: &str) -> Self {
        let lower = question.to_lowercase();
        let mut found = Self::default();
        for profile in profiles.values() {
            for column in &profile.columns {
                let named: Vec<Link> = column
                    .values
                    .iter()
                    .filter(|v| v.chars().count() >= 3 && v.chars().any(char::is_alphabetic))
                    .filter_map(|value| {
                        let at = word_at(&lower, &value.to_lowercase())?;
                        let named = question
                            .get(at..at.saturating_add(value.len()))
                            .filter(|_| lower.len() == question.len())
                            .unwrap_or(value);
                        Some(Link {
                            table: profile.table.clone(),
                            column: column.name.clone(),
                            value: value.clone(),
                            named: named.to_owned(),
                        })
                    })
                    .collect();
                if !named.is_empty() {
                    found.values.insert(
                        (profile.table.clone(), column.name.clone()),
                        column.values.clone(),
                    );
                    found.links.extend(named);
                }
            }
        }
        found
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Why a read must not run as written: it compares a column the
    /// question names a value of with a value the question does not name.
    /// `None` when every such filter uses a value the question names.
    pub(crate) fn refusal(&self, filters: &Filters) -> Option<String> {
        filters
            .values
            .iter()
            .find_map(|filter| self.refuse(filter, &filters.tables))
    }

    fn refuse(&self, filter: &ValueFilter, tables: &[String]) -> Option<String> {
        let named: Vec<&Link> = self
            .links
            .iter()
            .filter(|l| {
                l.column.eq_ignore_ascii_case(&filter.column)
                    && tables.iter().any(|t| t.eq_ignore_ascii_case(&l.table))
            })
            .collect();
        let first = named.first()?;
        let same = |value: &str| {
            if filter.folded {
                value.to_lowercase() == filter.value.to_lowercase()
            } else {
                value == filter.value
            }
        };
        if named.iter().any(|l| same(&l.value)) {
            return None;
        }
        let place = format!("{}.{}", first.table, first.column);
        let holds = self
            .values
            .get(&(first.table.clone(), first.column.clone()))
            .is_some_and(|values| values.iter().any(|v| same(v)));
        let stored: Vec<String> = named
            .iter()
            .map(|l| format!("{} is '{}'", l.named, l.value))
            .collect();
        let why = if holds {
            "which the question does not name"
        } else {
            "which the column does not hold"
        };
        Some(format!(
            "Not run: this query compares {column} with '{value}', {why}. The question names \
             values of {place}: {stored}. Write the query again with those values, exactly as \
             the column holds them.",
            column = filter.column,
            value = filter.value,
            stored = stored.join(", "),
        ))
    }
}

/// Where `word` starts in `text` as a whole word: no letter or digit on
/// either side.
fn word_at(text: &str, word: &str) -> Option<usize> {
    if word.is_empty() {
        return None;
    }
    text.match_indices(word).map(|(at, _)| at).find(|&at| {
        let before = text.get(..at).and_then(|s| s.chars().next_back());
        let after = text
            .get(at.saturating_add(word.len())..)
            .and_then(|s| s.chars().next());
        !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::Dimension;
    use crate::storage::workspace::WorkspaceDb;

    /// A workspace with a storms table, profiled, and the value links of
    /// `question` in it.
    #[expect(clippy::unwrap_used, reason = "test setup")]
    fn linked(question: &str) -> (WorkspaceDb, ValueLinks) {
        let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
        db.connection()
            .execute_batch(
                "CREATE TABLE events (state VARCHAR, kind VARCHAR, damage INTEGER);
                 INSERT INTO events VALUES ('TENNESSEE', 'Tornado', 10), ('TEXAS', 'Hail', 20),
                 ('TEXAS', 'Tornado', 5), ('IOWA', 'Hail', 7);",
            )
            .unwrap();
        TableProfile::refresh(&db, "events").unwrap();
        let links = ValueLinks::find(&TableProfile::all(&db).unwrap(), question);
        (db, links)
    }

    fn refusal(db: &WorkspaceDb, links: &ValueLinks, sql: &str) -> Option<String> {
        let filters = db.statement_filters(sql).ok().flatten().unwrap_or_default();
        assert!(!filters.tables.is_empty() || sql.contains("label"), "{sql}");
        links.refusal(&filters)
    }

    /// The question's `Texas` is the column's `TEXAS`: a query on another
    /// state, or on the question's spelling, is refused and told the value.
    #[test]
    fn a_filter_must_use_the_value_the_question_names() {
        let (db, links) = linked("What is the total damage in Texas?");
        let wrong = refusal(
            &db,
            &links,
            "SELECT sum(damage) FROM events WHERE state = 'TENNESSEE'",
        )
        .unwrap_or_default();
        assert!(
            wrong.contains("'TENNESSEE', which the question does not name"),
            "{wrong}"
        );
        assert!(wrong.contains("Texas is 'TEXAS'"), "{wrong}");
        let spelled = refusal(
            &db,
            &links,
            "SELECT sum(damage) FROM events e WHERE e.state IN ('Texas')",
        )
        .unwrap_or_default();
        assert!(
            spelled.contains("which the column does not hold"),
            "{spelled}"
        );
        for sql in [
            "SELECT sum(damage) FROM events WHERE state = 'TEXAS'",
            "SELECT sum(damage) FROM events WHERE lower(state) = 'texas'",
            "SELECT sum(damage) FROM events WHERE kind = 'Hail' AND state = 'TEXAS'",
            "SELECT state, sum(damage) FROM events GROUP BY ALL",
            "SELECT 'TENNESSEE' AS label",
        ] {
            assert_eq!(refusal(&db, &links, sql), None, "{sql}");
        }
    }

    /// Short values, values in other words, and questions naming no value
    /// link nothing, so nothing is refused.
    #[test]
    fn only_whole_words_link() {
        let (db, links) = linked("Which states had texasish storms?");
        assert!(links.is_empty());
        assert_eq!(
            refusal(&db, &links, "SELECT * FROM events WHERE state = 'IOWA'"),
            None
        );
        let (_, both) = linked("Compare Texas with Iowa");
        assert_eq!(both.links.len(), 2);
    }

    #[test]
    fn the_note_names_what_the_workspace_holds() {
        let t = |names: &[&str]| -> Vec<String> { names.iter().map(|n| (*n).to_owned()).collect() };
        let both = nothing_fetched(&t(&["events", "states"]), 32);
        assert!(
            both.contains("holds the tables events, states, and 32 documents."),
            "{both}"
        );
        assert!(
            both.starts_with("quack ran no query and no search"),
            "{both}"
        );
        let tables = nothing_fetched(&t(&["events"]), 0);
        assert!(tables.contains("holds the table events. Ask"), "{tables}");
        let documents = nothing_fetched(&[], 1);
        assert!(documents.contains("holds 1 document. Ask"), "{documents}");
        let many: Vec<String> = (1..=12).map(|i| format!("t{i}")).collect();
        let capped = nothing_fetched(&many, 0);
        assert!(capped.contains("t10 and 2 more."), "{capped}");
    }
}
