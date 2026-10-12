//! Table and column names for a SQL statement being typed, from the
//! completer the web SQL page shares (`quack_core::completion`). The
//! terminal offers names alone: a person typing SQL at a prompt knows the
//! keywords, and a popup over every word would get in the way of Enter.

use quack_core::completion::{SqlCompletion, SqlItemKind, SqlVocabulary};
use quack_core::storage::workspace::{SqlName, SqlSchema};

use super::commands::{Completion, Suggestion};

impl Completion {
    /// What the popup offers for the SQL statement `line` with the cursor
    /// at character `cursor`, or `None` when the line is not a statement
    /// or no name fits there.
    pub(crate) fn for_sql(line: &str, cursor: usize, schema: &SqlSchema) -> Option<Self> {
        let found = SqlCompletion::for_sql(line, cursor, schema, &SqlVocabulary::none())?;
        let items: Vec<Suggestion> = found
            .items
            .iter()
            .map(|item| {
                let name = SqlName {
                    name: item.label.clone(),
                    sql: item.word.clone(),
                };
                let about = match item.kind {
                    SqlItemKind::Column => format!("column of {}", item.table),
                    SqlItemKind::Table => String::from("table"),
                    SqlItemKind::Keyword => String::from("keyword"),
                    SqlItemKind::Function => String::from("function"),
                };
                Suggestion::sql(&name, &about)
            })
            .collect();
        // The core counts characters; the popup replaces a byte span.
        let byte_at = |chars: usize| {
            line.char_indices()
                .nth(chars)
                .map_or(line.len(), |(at, _)| at)
        };
        Some(Self::at(byte_at(found.start), byte_at(found.end), items))
    }
}

#[cfg(test)]
mod tests {
    use quack_core::embedding::Dimension;
    use quack_core::storage::workspace::WorkspaceDb;

    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    /// The core completer decides what is offered; the terminal fills the
    /// word in at the cursor and labels each item.
    #[test]
    fn a_name_fills_in_at_the_cursor() {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        db.execute_statement("CREATE TABLE sales (region VARCHAR, revenue INTEGER)")
            .unwrap_or_else(|e| fail(&e.to_string()));
        let schema = db.sql_schema().unwrap_or_else(|e| fail(&e.to_string()));
        let line = "SELECT re FROM sales";
        let completion =
            Completion::for_sql(line, 9, &schema).unwrap_or_else(|| fail("no completion"));
        let item = completion.get(1).unwrap_or_else(|| fail("no second item"));
        assert_eq!(item.about, "column of sales");
        assert_eq!(
            completion.apply(line, item),
            (String::from("SELECT revenue FROM sales"), 14)
        );
        assert!(Completion::for_sql("hello", 5, &schema).is_none());
    }
}
