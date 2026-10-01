//! Table and column names for a SQL statement being typed. The line is
//! split into tokens by sqlparser's `DuckDB` tokenizer, so strings, quoted
//! names, and comments are read as `DuckDB` reads them; the schema comes
//! from `WorkspaceDb::sql_schema`, which never names an internal table.

use quack_core::storage::workspace::{SqlName, SqlSchema, TableColumns, looks_like_direct_sql};
use sqlparser::dialect::DuckDbDialect;
use sqlparser::keywords::Keyword;
use sqlparser::tokenizer::{Token, Tokenizer, Whitespace, Word};

use super::commands::{Completion, Suggestion};

/// The most names one popup offers.
const MAX_ITEMS: usize = 50;

/// Where the word at the cursor sits in the statement.
#[derive(Debug, PartialEq, Eq)]
enum Slot {
    /// After `FROM`, `JOIN`, `DESCRIBE`, `SUMMARIZE`, or a comma in a
    /// `FROM` list.
    Table,
    /// In an expression: the named tables' columns, and table names to
    /// qualify with.
    Column,
    /// After `name.`: that table's (or alias's) columns only.
    Qualified(String),
}

/// A statement's tokens, the word at the cursor, and what it names.
struct Typed {
    /// The tokens before the word being typed, whitespace and comments
    /// dropped.
    before: Vec<Token>,
    /// The word being typed: unquoted letters up to the cursor, or empty.
    word: String,
    /// Whether `word` is a whole keyword, which the line most likely means.
    keyword: bool,
}

impl Typed {
    /// `prefix` read up to its end, or `None` inside a string, quoted
    /// name, comment, or number, where no name belongs.
    fn read(prefix: &str) -> Option<Self> {
        let mut tokens = significant(prefix)?;
        let trailing_space = prefix.ends_with(|c: char| c.is_whitespace());
        let (word, keyword) = match tokens.last() {
            Some(Token::Word(Word {
                value,
                quote_style: None,
                keyword,
            })) if !trailing_space => (value.clone(), *keyword != Keyword::NoKeyword),
            Some(Token::Word(_) | Token::Number(..) | Token::SingleQuotedString(_))
                if !trailing_space =>
            {
                return None;
            }
            _ => (String::new(), false),
        };
        if !word.is_empty() {
            tokens.pop();
        }
        Some(Self {
            before: tokens,
            word,
            keyword,
        })
    }

    fn slot(&self) -> Option<Slot> {
        let mut back = self.before.iter().rev();
        let previous = back.next()?;
        if *previous == Token::Period {
            return match back.next() {
                Some(Token::Word(word)) => Some(Slot::Qualified(word.value.clone())),
                _ => None,
            };
        }
        if let Token::Word(word) = previous
            && opens_table(word)
        {
            return Some(Slot::Table);
        }
        let clause = self.before.iter().rev().find_map(|token| match token {
            Token::Word(word) if opens_table(word) || is_clause(word.keyword) => Some(word),
            _ => None,
        })?;
        if opens_table(clause) {
            // In a FROM list a name follows a comma; anything else there
            // is an alias being typed.
            return (*previous == Token::Comma).then_some(Slot::Table);
        }
        match previous {
            // An alias after AS, or after a name with none.
            Token::Word(word) if word.keyword == Keyword::AS => None,
            Token::Word(word) if word.keyword == Keyword::NoKeyword => None,
            _ => (!self.word.is_empty()).then_some(Slot::Column),
        }
    }
}

/// The statement's tokens without whitespace and comments, or `None` when
/// the tokenizer refuses it (an unterminated string or quoted name).
fn significant(sql: &str) -> Option<Vec<Token>> {
    let tokens = Tokenizer::new(&DuckDbDialect {}, sql).tokenize().ok()?;
    if let Some(Token::Whitespace(
        Whitespace::SingleLineComment { .. } | Whitespace::MultiLineComment(_),
    )) = tokens.last()
    {
        return None;
    }
    Some(
        tokens
            .into_iter()
            .filter(|t| !matches!(t, Token::Whitespace(_)))
            .collect(),
    )
}

/// A word after which a table name comes.
fn opens_table(word: &Word) -> bool {
    matches!(
        word.keyword,
        Keyword::FROM | Keyword::JOIN | Keyword::DESCRIBE | Keyword::UPDATE | Keyword::INTO
    ) || (word.quote_style.is_none() && word.value.eq_ignore_ascii_case("summarize"))
}

/// A keyword that starts a clause of expressions.
const fn is_clause(keyword: Keyword) -> bool {
    matches!(
        keyword,
        Keyword::SELECT
            | Keyword::WHERE
            | Keyword::BY
            | Keyword::ON
            | Keyword::HAVING
            | Keyword::QUALIFY
            | Keyword::SET
            | Keyword::USING
            | Keyword::RETURNING
    )
}

/// The tables a statement names after `FROM` or `JOIN`, each with the
/// name it goes by there (its alias, else itself).
fn named_tables<'s>(line: &str, schema: &'s SqlSchema) -> Vec<(String, &'s TableColumns)> {
    let Some(tokens) = significant(line) else {
        return Vec::new();
    };
    let mut named = Vec::new();
    let mut tokens = tokens.iter().peekable();
    while let Some(token) = tokens.next() {
        let Token::Word(word) = token else {
            continue;
        };
        if !matches!(word.keyword, Keyword::FROM | Keyword::JOIN) {
            continue;
        }
        while let Some(Token::Word(table)) = tokens.next() {
            let Some(columns) = find_table(schema, &table.value) else {
                break;
            };
            if let Some(Token::Word(w)) = tokens.peek()
                && w.keyword == Keyword::AS
            {
                tokens.next();
            }
            let alias = match tokens.peek() {
                Some(Token::Word(w))
                    if w.keyword == Keyword::NoKeyword || w.quote_style.is_some() =>
                {
                    let alias = w.value.clone();
                    tokens.next();
                    alias
                }
                _ => table.value.clone(),
            };
            named.push((alias, columns));
            if tokens.peek() == Some(&&Token::Comma) {
                tokens.next();
            } else {
                break;
            }
        }
    }
    named
}

fn find_table<'s>(schema: &'s SqlSchema, name: &str) -> Option<&'s TableColumns> {
    schema
        .tables
        .iter()
        .find(|t| t.name.name.eq_ignore_ascii_case(name))
}

impl Completion {
    /// What the popup offers for the SQL statement `line` with the cursor
    /// at character `cursor`, or `None` when the line is not a statement
    /// or no name fits there.
    pub(crate) fn for_sql(line: &str, cursor: usize, schema: &SqlSchema) -> Option<Self> {
        if !looks_like_direct_sql(line) {
            return None;
        }
        let end = line
            .char_indices()
            .nth(cursor)
            .map_or(line.len(), |(at, _)| at);
        let prefix = line.get(..end)?;
        let typed = Typed::read(prefix)?;
        let slot = typed.slot()?;
        let start = end.checked_sub(typed.word.len())?;
        let matches = |name: &SqlName| {
            name.name
                .to_lowercase()
                .starts_with(&typed.word.to_lowercase())
        };
        let mut items: Vec<Suggestion> = Vec::new();
        let tables = || {
            schema
                .tables
                .iter()
                .filter(|t| matches(&t.name))
                .map(|t| Suggestion::sql(&t.name, "table"))
        };
        match slot {
            Slot::Table => items.extend(tables()),
            Slot::Qualified(qualifier) => {
                let table = named_tables(line, schema)
                    .into_iter()
                    .find(|(alias, _)| alias.eq_ignore_ascii_case(&qualifier))
                    .map(|(_, table)| table)
                    .or_else(|| find_table(schema, &qualifier))?;
                let about = format!("column of {}", table.name.name);
                items.extend(
                    table
                        .columns
                        .iter()
                        .filter(|c| matches(c))
                        .map(|c| Suggestion::sql(c, &about)),
                );
            }
            Slot::Column => {
                for (_, table) in named_tables(line, schema) {
                    let about = format!("column of {}", table.name.name);
                    for column in table.columns.iter().filter(|c| matches(c)) {
                        if !items.iter().any(|i| i.word == column.sql) {
                            items.push(Suggestion::sql(column, &about));
                        }
                    }
                }
                items.extend(tables());
            }
        }
        // A whole keyword with no name spelled the same is the keyword.
        let exact = |item: &Suggestion| item.label.eq_ignore_ascii_case(&typed.word);
        if typed.keyword && !items.iter().any(exact) {
            return None;
        }
        // The name typed in full goes first, so Enter sends the line as is.
        items.sort_by_key(|item| !exact(item));
        items.truncate(MAX_ITEMS);
        (!items.is_empty()).then(|| Self::at(start, end, items))
    }
}

#[cfg(test)]
mod tests {
    use quack_core::storage::workspace::WorkspaceDb;

    use super::*;
    use quack_core::embedding::Dimension;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn schema() -> SqlSchema {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        for sql in [
            "CREATE TABLE sales (region VARCHAR, revenue INTEGER, sold DATE)",
            "CREATE TABLE stores (id INTEGER, \"Region\" VARCHAR)",
            "CREATE TABLE \"Order Items\" (sku VARCHAR)",
        ] {
            db.execute_statement(sql)
                .unwrap_or_else(|e| fail(&e.to_string()));
        }
        db.sql_schema().unwrap_or_else(|e| fail(&e.to_string()))
    }

    /// The words offered with the cursor at `|` (the end when there is none).
    fn offered(line: &str) -> Vec<String> {
        let cursor = line.find('|').map_or_else(
            || line.chars().count(),
            |at| line.get(..at).map_or(0, |head| head.chars().count()),
        );
        let line = line.replace('|', "");
        Completion::for_sql(&line, cursor, &schema())
            .map(|c| c.items.into_iter().map(|s| s.word).collect())
            .unwrap_or_default()
    }

    #[test]
    fn tables_complete_after_from_and_join() {
        assert_eq!(offered("SELECT * FROM sa"), ["sales"]);
        assert_eq!(
            offered("SELECT * FROM "),
            ["\"Order Items\"", "sales", "stores"]
        );
        assert_eq!(offered("SELECT * FROM sales JOIN st"), ["stores"]);
        assert_eq!(offered("SELECT * FROM sales, st"), ["stores"]);
        assert_eq!(offered("describe sa"), ["sales"]);
        assert_eq!(offered("SUMMARIZE o"), ["\"Order Items\""]);
    }

    #[test]
    fn columns_come_from_the_tables_the_statement_names() {
        assert_eq!(offered("SELECT re| FROM sales"), ["region", "revenue"]);
        assert_eq!(
            offered("SELECT * FROM sales WHERE r"),
            ["region", "revenue"]
        );
        // Both tables' columns, each once, then table names to qualify with.
        assert_eq!(
            offered("SELECT * FROM sales JOIN stores ON s"),
            ["sold", "sales", "stores"]
        );
        assert_eq!(offered("SELECT * FROM stores ORDER BY r"), ["\"Region\""]);
    }

    #[test]
    fn a_qualifier_offers_only_its_tables_columns() {
        assert_eq!(
            offered("SELECT s.| FROM sales s"),
            ["region", "revenue", "sold"]
        );
        assert_eq!(
            offered("SELECT s.re| FROM sales AS s"),
            ["region", "revenue"]
        );
        assert_eq!(offered("SELECT stores.| FROM stores"), ["id", "\"Region\""]);
        assert!(offered("SELECT x.| FROM sales s").is_empty());
    }

    #[test]
    fn nothing_is_offered_where_no_name_belongs() {
        for line in [
            "hello there",
            "SELECT * FROM sales s",
            "SELECT * FROM sales AS s",
            "SELECT region AS r",
            "SELECT * FROM sales ORDER BY revenue DESC",
            "SELECT * FROM sales WHERE region = 'n",
            "SELECT * FROM sales WHERE region = 'north'",
            "SELECT * FROM sales LIMIT 5",
            "SELECT 1 ",
            "SELECT * FROM sales -- re",
            "SELECT \"re",
        ] {
            assert!(offered(line).is_empty(), "{line}: {:?}", offered(line));
        }
    }

    #[test]
    fn a_name_typed_in_full_comes_first_and_fills_in_at_the_cursor() {
        assert_eq!(offered("SELECT * FROM sales ORDER BY region"), ["region"]);
        let schema = schema();
        let line = "SELECT re FROM sales";
        let completion =
            Completion::for_sql(line, 9, &schema).unwrap_or_else(|| fail("no completion"));
        let item = completion.get(1).unwrap_or_else(|| fail("no second item"));
        assert_eq!(
            completion.apply(line, item),
            (String::from("SELECT revenue FROM sales"), 14)
        );
    }
}
