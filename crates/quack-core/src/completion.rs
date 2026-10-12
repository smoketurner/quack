//! Completion for a SQL statement being typed, shared by the terminal and
//! the web SQL page. The line is split into tokens by sqlparser's `DuckDB`
//! tokenizer, so strings, quoted names, and comments are read as `DuckDB`
//! reads them; the schema comes from `WorkspaceDb::sql_schema`, which never
//! names an internal table; keywords and functions come from `DuckDB`'s
//! own catalog (`duckdb_keywords()`, `duckdb_functions()`), read once per
//! process.

use std::sync::{Arc, OnceLock};

use sqlparser::dialect::DuckDbDialect;
use sqlparser::keywords::Keyword;
use sqlparser::tokenizer::{Token, Tokenizer, Whitespace, Word};

use crate::error::Result;
use crate::storage::workspace::{SqlName, SqlSchema, TableColumns, looks_like_direct_sql};

/// The most items one completion offers.
const MAX_ITEMS: usize = 50;
/// The most keywords, and the most functions, offered after the names: a
/// prefix like `re` matches dozens of each, and the names come first.
const VOCABULARY_PER_KIND: usize = 12;

/// What a completion item is, which decides how each interface shows it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum SqlItemKind {
    Table,
    Column,
    Keyword,
    Function,
}

/// One thing the cursor's word can become.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, utoipa::ToSchema)]
pub struct SqlItem {
    /// What replaces the word being typed, as a statement writes it.
    pub word: String,
    /// The name as shown.
    pub label: String,
    pub kind: SqlItemKind,
    /// The table a column belongs to, else empty.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub table: String,
}

impl SqlItem {
    fn name(name: &SqlName, kind: SqlItemKind, table: &str) -> Self {
        Self {
            word: name.sql.clone(),
            label: name.name.clone(),
            kind,
            table: table.to_owned(),
        }
    }

    fn vocabulary(word: &str, kind: SqlItemKind) -> Self {
        Self {
            word: word.to_owned(),
            label: word.to_owned(),
            kind,
            table: String::new(),
        }
    }
}

/// What the cursor's word can become: `start..end` are character offsets
/// of that word in the line.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, utoipa::ToSchema)]
pub struct SqlCompletion {
    pub start: usize,
    pub end: usize,
    pub items: Vec<SqlItem>,
}

/// `DuckDB`'s keywords and function names, for an expression where a name
/// does not have to come. Read once per process from an in-memory
/// connection, since they depend on the bundled `DuckDB` alone.
#[derive(Debug, Default)]
pub struct SqlVocabulary {
    keywords: Vec<String>,
    functions: Vec<String>,
}

impl SqlVocabulary {
    /// The bundled `DuckDB`'s words, read on first use.
    ///
    /// # Errors
    ///
    /// Returns an error when the in-memory connection cannot be opened or
    /// queried, which means `DuckDB` itself is broken.
    pub fn duckdb() -> Result<Arc<Self>> {
        static WORDS: OnceLock<Arc<SqlVocabulary>> = OnceLock::new();
        if let Some(words) = WORDS.get() {
            return Ok(Arc::clone(words));
        }
        let conn = duckdb::Connection::open_in_memory()?;
        let read = |sql: &str| -> Result<Vec<String>> {
            Ok(conn
                .prepare(sql)?
                .query_map([], |row| row.get::<_, String>(0))?
                .collect::<duckdb::Result<_>>()?)
        };
        let keywords = read(
            "SELECT upper(keyword_name) FROM duckdb_keywords() \
             WHERE keyword_category IN ('reserved', 'unreserved', 'column_name', 'type_function') \
             ORDER BY 1",
        )?;
        // The functions an expression takes: scalar, aggregate, and macro,
        // not table functions (`read_csv`, which the confined connection
        // blocks anyway, belongs after FROM, where tables are offered).
        // Named ones only: operators (`+`, `||`) and `__internal` names are
        // not words a person types. Every built-in is `internal`, so that
        // column cannot filter.
        let functions = read(
            "SELECT DISTINCT function_name FROM duckdb_functions() \
             WHERE function_type IN ('scalar', 'aggregate', 'macro') \
               AND regexp_matches(function_name, '^[a-z][a-z0-9_]*$') \
             ORDER BY 1",
        )?;
        let words = Arc::new(Self {
            keywords,
            functions,
        });
        Ok(Arc::clone(WORDS.get_or_init(|| words)))
    }

    /// No words: names alone, as the terminal offers.
    #[must_use]
    pub fn none() -> Arc<Self> {
        static NONE: OnceLock<Arc<SqlVocabulary>> = OnceLock::new();
        Arc::clone(NONE.get_or_init(|| Arc::new(Self::default())))
    }

    fn matching<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = SqlItem> + 'a {
        let lower = prefix.to_lowercase();
        let keywords = self
            .keywords
            .iter()
            .filter(move |k| k.to_lowercase().starts_with(&lower))
            .map(|k| SqlItem::vocabulary(k, SqlItemKind::Keyword))
            .take(VOCABULARY_PER_KIND);
        let lower = prefix.to_lowercase();
        let functions = self
            .functions
            .iter()
            .filter(move |f| f.starts_with(&lower))
            .map(|f| SqlItem::vocabulary(f, SqlItemKind::Function))
            .take(VOCABULARY_PER_KIND);
        keywords.chain(functions)
    }
}

/// Where the word at the cursor sits in the statement.
#[derive(Debug, PartialEq, Eq)]
enum Slot {
    /// After `FROM`, `JOIN`, `DESCRIBE`, `SUMMARIZE`, or a comma in a
    /// `FROM` list.
    Table,
    /// In an expression: the named tables' columns, table names to qualify
    /// with, then keywords and functions.
    Column,
    /// After `name.`: that table's (or alias's) columns only.
    Qualified(String),
}

/// What a word leads into: a table name or a clause of expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Context {
    /// `FROM`, `JOIN`, `DESCRIBE`, `SUMMARIZE`, `UPDATE`, `INTO`.
    Tables,
    /// `SELECT`, `WHERE`, `BY`, `ON`, `HAVING`, `QUALIFY`, `SET`, `USING`,
    /// `RETURNING`.
    Expressions,
}

impl Context {
    fn of(word: &Word) -> Option<Self> {
        match word.keyword {
            Keyword::FROM | Keyword::JOIN | Keyword::DESCRIBE | Keyword::UPDATE | Keyword::INTO => {
                Some(Self::Tables)
            }
            Keyword::SELECT
            | Keyword::WHERE
            | Keyword::BY
            | Keyword::ON
            | Keyword::HAVING
            | Keyword::QUALIFY
            | Keyword::SET
            | Keyword::USING
            | Keyword::RETURNING => Some(Self::Expressions),
            // sqlparser has no SUMMARIZE keyword.
            _ if word.quote_style.is_none() && word.value.eq_ignore_ascii_case("summarize") => {
                Some(Self::Tables)
            }
            _ => None,
        }
    }
}

/// A statement's tokens without whitespace and comments.
struct Statement(Vec<Token>);

impl Statement {
    /// `sql` tokenized, or `None` when the tokenizer refuses it (an
    /// unterminated string or quoted name) or it ends inside a comment.
    fn read(sql: &str) -> Option<Self> {
        let tokens = Tokenizer::new(&DuckDbDialect {}, sql).tokenize().ok()?;
        if let Some(Token::Whitespace(
            Whitespace::SingleLineComment { .. } | Whitespace::MultiLineComment(_),
        )) = tokens.last()
        {
            return None;
        }
        Some(Self(
            tokens
                .into_iter()
                .filter(|t| !matches!(t, Token::Whitespace(_)))
                .collect(),
        ))
    }

    /// The tables the statement names after `FROM` or `JOIN`, each with the
    /// name it goes by there (its alias, else itself).
    fn named_tables<'s>(&self, schema: &'s SqlSchema) -> Vec<(String, &'s TableColumns)> {
        let mut named = Vec::new();
        let mut tokens = self.0.iter().peekable();
        while let Some(token) = tokens.next() {
            let Token::Word(word) = token else {
                continue;
            };
            if !matches!(word.keyword, Keyword::FROM | Keyword::JOIN) {
                continue;
            }
            while let Some(Token::Word(table)) = tokens.next() {
                let columns = schema.table(&table.value);
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
                if let Some(columns) = columns {
                    named.push((alias, columns));
                }
                if tokens.peek() == Some(&&Token::Comma) {
                    tokens.next();
                } else {
                    break;
                }
            }
        }
        named
    }
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
        let Statement(mut tokens) = Statement::read(prefix)?;
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
            && Context::of(word) == Some(Context::Tables)
        {
            return Some(Slot::Table);
        }
        let context = self.before.iter().rev().find_map(|token| match token {
            Token::Word(word) => Context::of(word),
            _ => None,
        })?;
        match (context, previous) {
            // In a FROM list a name follows a comma; anything else there is
            // an alias being typed.
            (Context::Tables, _) => (*previous == Token::Comma).then_some(Slot::Table),
            // An alias after AS, or after a name with none.
            (Context::Expressions, Token::Word(word))
                if word.keyword == Keyword::AS || word.keyword == Keyword::NoKeyword =>
            {
                None
            }
            // A blank word gets names only where an expression must
            // follow, so Enter still sends a finished statement.
            (Context::Expressions, _) => (!self.word.is_empty()
                || Self::expects_expression(previous))
            .then_some(Slot::Column),
        }
    }

    /// Whether an expression must come after `token`: a clause keyword, a
    /// connective, a comma, an open parenthesis, or a comparison or
    /// arithmetic operator (not `*`, which may be `SELECT *`).
    fn expects_expression(token: &Token) -> bool {
        match token {
            Token::Word(word) => matches!(
                word.keyword,
                Keyword::SELECT
                    | Keyword::WHERE
                    | Keyword::BY
                    | Keyword::ON
                    | Keyword::HAVING
                    | Keyword::QUALIFY
                    | Keyword::SET
                    | Keyword::USING
                    | Keyword::RETURNING
                    | Keyword::AND
                    | Keyword::OR
                    | Keyword::NOT
                    | Keyword::CASE
                    | Keyword::WHEN
                    | Keyword::THEN
                    | Keyword::ELSE
            ),
            Token::Comma
            | Token::LParen
            | Token::Eq
            | Token::Neq
            | Token::Lt
            | Token::Gt
            | Token::LtEq
            | Token::GtEq
            | Token::Plus
            | Token::Minus
            | Token::Div => true,
            _ => false,
        }
    }
}

impl SqlCompletion {
    /// What `line` can take at character `cursor`: names where the
    /// statement takes them, and in an expression the vocabulary's keywords
    /// and functions after a letter is typed. `None` when the line is not a
    /// statement or nothing fits there.
    #[must_use]
    pub fn for_sql(
        line: &str,
        cursor: usize,
        schema: &SqlSchema,
        vocabulary: &SqlVocabulary,
    ) -> Option<Self> {
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
        let start = cursor.checked_sub(typed.word.chars().count())?;
        // The whole line names the tables, the part after the cursor too.
        let line_statement = Statement::read(line);
        let named = |schema| {
            line_statement
                .as_ref()
                .map_or_else(Vec::new, |statement| statement.named_tables(schema))
        };
        let matches = |name: &SqlName| {
            name.name
                .to_lowercase()
                .starts_with(&typed.word.to_lowercase())
        };
        let mut items: Vec<SqlItem> = Vec::new();
        let tables = || {
            schema
                .tables
                .iter()
                .filter(|t| matches(&t.name))
                .map(|t| SqlItem::name(&t.name, SqlItemKind::Table, ""))
        };
        match slot {
            Slot::Table => items.extend(tables()),
            Slot::Qualified(qualifier) => {
                let table = named(schema)
                    .into_iter()
                    .find(|(alias, _)| alias.eq_ignore_ascii_case(&qualifier))
                    .map(|(_, table)| table)
                    .or_else(|| schema.table(&qualifier))?;
                items.extend(
                    table
                        .columns
                        .iter()
                        .filter(|c| matches(c))
                        .map(|c| SqlItem::name(c, SqlItemKind::Column, &table.name.name)),
                );
            }
            Slot::Column => {
                for (_, table) in named(schema) {
                    for column in table.columns.iter().filter(|c| matches(c)) {
                        if !items.iter().any(|i| i.word == column.sql) {
                            items.push(SqlItem::name(
                                column,
                                SqlItemKind::Column,
                                &table.name.name,
                            ));
                        }
                    }
                }
                items.extend(tables());
                // Words only once a letter is typed: a blank slot is for names.
                if !typed.word.is_empty() {
                    items.extend(vocabulary.matching(&typed.word));
                }
            }
        }
        // A whole keyword with no name spelled the same is the keyword.
        let exact = |item: &SqlItem| item.label.eq_ignore_ascii_case(&typed.word);
        let names_exact = items
            .iter()
            .any(|item| exact(item) && !matches!(item.kind, SqlItemKind::Keyword));
        if typed.keyword && !names_exact {
            return None;
        }
        // The name typed in full goes first, so Enter sends the line as is.
        items.sort_by_key(|item| !exact(item));
        items.truncate(MAX_ITEMS);
        if items.is_empty() {
            return None;
        }
        Some(Self {
            start,
            end: cursor,
            items,
        })
    }
}

#[cfg(test)]
mod tests;
