//! The columns of a labels table: the source's key, what each question
//! adds, and whether the row's text was cut. The same list names the
//! columns, writes an answer into them, and says what each means.

use duckdb::types::Value;

use super::Error;
use crate::llm::decision::{Answer, Answers, Question, QuestionName, Questions};
use crate::storage::workspace::quote_ident;
use crate::text::OneLine;

/// The column a row's cut flag lands in.
pub(super) const TRUNCATED: &str = "truncated";

/// Longest meaning a column carries.
const MEANING_CHARS: usize = 500;

/// One column the labels table adds.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LabelColumn {
    name: String,
    sql_type: &'static str,
    meaning: String,
}

impl LabelColumn {
    fn new(name: String, sql_type: &'static str, meaning: String) -> Self {
        Self {
            name,
            sql_type,
            meaning,
        }
    }
}

/// The labels table's columns, in order: the key, each question's, then
/// [`TRUNCATED`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct OutputColumns {
    key: String,
    labels: Vec<LabelColumn>,
    source: String,
}

impl OutputColumns {
    /// The columns `questions` adds beside the column `key` of `source`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ColumnClash`] when two columns are named
    /// alike, without regard to case: the key, a question's, a derived
    /// `_p`, `_confidence`, or `_level`, or `truncated`.
    pub(super) fn new(source: &str, key: &str, questions: &Questions) -> Result<Self, Error> {
        let mut labels = Vec::new();
        for (name, question) in questions.iter() {
            labels.extend(Self::of_question(name, question));
        }
        labels.push(LabelColumn::new(
            String::from(TRUNCATED),
            "BOOLEAN",
            String::from("the row's text was cut to fit the model"),
        ));
        let columns = Self {
            key: key.to_owned(),
            labels,
            source: source.to_owned(),
        };
        columns.check_distinct()?;
        Ok(columns)
    }

    fn check_distinct(&self) -> Result<(), Error> {
        let mut seen: Vec<String> = Vec::new();
        for name in self.names() {
            let lower = name.to_ascii_lowercase();
            if seen.contains(&lower) {
                return Err(Error::ColumnClash(name.to_owned()));
            }
            seen.push(lower);
        }
        Ok(())
    }

    /// The columns one question adds.
    fn of_question(name: &QuestionName, question: &Question) -> Vec<LabelColumn> {
        let instructions = question.instructions().as_str();
        match question {
            Question::Choice { criteria, .. } => {
                let options = criteria
                    .iter()
                    .map(|(label, description)| match description {
                        Some(description) => format!("{label} ({description})"),
                        None => label.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                vec![
                    LabelColumn::new(
                        name.to_string(),
                        "VARCHAR",
                        format!("{instructions} One of {options}"),
                    ),
                    LabelColumn::new(
                        format!("{name}_p"),
                        "DOUBLE",
                        format!("probability of the chosen {name}"),
                    ),
                    LabelColumn::new(
                        format!("{name}_confidence"),
                        "DOUBLE",
                        format!(
                            "how concentrated {name}'s probabilities are, 0 to 1; not the chance \
                             it is right"
                        ),
                    ),
                ]
            }
            Question::Score { criteria, .. } => {
                let levels = criteria
                    .iter()
                    .enumerate()
                    .map(|(at, description)| format!("{at} {description}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                vec![
                    LabelColumn::new(
                        name.to_string(),
                        "DOUBLE",
                        format!("{instructions} expected level, {levels}"),
                    ),
                    LabelColumn::new(
                        format!("{name}_level"),
                        "INTEGER",
                        format!("most probable level of {name}"),
                    ),
                ]
            }
            Question::Noul { .. } => vec![LabelColumn::new(
                name.to_string(),
                "DOUBLE",
                format!("probability that: {instructions}"),
            )],
        }
    }

    fn names(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.key.as_str()).chain(self.labels.iter().map(|l| l.name.as_str()))
    }

    /// Every column's name, key first.
    pub(super) fn column_names(&self) -> Vec<String> {
        self.names().map(str::to_owned).collect()
    }

    /// The label columns as `NAME TYPE` definitions for a `CREATE TABLE`
    /// that selects the key: `NULL::TYPE AS "name"`.
    pub(super) fn null_selects(&self) -> String {
        self.labels
            .iter()
            .map(|l| format!("NULL::{} AS {}", l.sql_type, quote_ident(&l.name)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// What each column means, as sentences: the key first.
    pub(super) fn meanings(&self) -> Vec<(String, String)> {
        let key = (
            self.key.clone(),
            format!("the key of {}; join on it", self.source),
        );
        std::iter::once(key)
            .chain(self.labels.iter().map(|l| {
                let line = OneLine(&l.meaning).to_string();
                (l.name.clone(), line.chars().take(MEANING_CHARS).collect())
            }))
            .collect()
    }

    /// A row of the preview: the key, then `cells` as JSON.
    pub(super) fn json_row(key: &str, cells: Vec<Value>) -> Vec<serde_json::Value> {
        std::iter::once(serde_json::Value::String(key.to_owned()))
            .chain(cells.into_iter().map(|cell| {
                match cell {
                    Value::Boolean(flag) => serde_json::Value::Bool(flag),
                    Value::Int(n) => serde_json::Value::from(n),
                    Value::Double(n) => serde_json::Number::from_f64(n)
                        .map_or(serde_json::Value::Null, serde_json::Value::Number),
                    Value::Text(text) => serde_json::Value::String(text),
                    _ => serde_json::Value::Null,
                }
            }))
            .collect()
    }

    /// The header of a preview as a screen shows it: the key, then each
    /// question's name, a choice's with `(p)` after it.
    pub(super) fn compact_header(&self, questions: &Questions) -> Vec<String> {
        std::iter::once(self.key.clone())
            .chain(questions.iter().map(|(name, question)| match question {
                Question::Choice { .. } => format!("{name} (p)"),
                Question::Noul { .. } | Question::Score { .. } => name.to_string(),
            }))
            .collect()
    }

    /// A row of the preview as a screen shows it: a choice as `label (p)`, a
    /// score as its level, a yes/no as its probability of yes, and a `*`
    /// after the key of a row whose text was cut.
    pub(super) fn compact_row(key: &str, answers: Option<&Answers>) -> Vec<String> {
        let cut = answers.is_some_and(Answers::truncated);
        let mut row = vec![if cut {
            format!("{key}*")
        } else {
            key.to_owned()
        }];
        if let Some(answers) = answers {
            row.extend(answers.iter().map(|(_, answer)| match answer {
                Answer::Choice {
                    choice,
                    probability,
                    ..
                } => format!("{choice} ({probability:.2})"),
                Answer::Noul { probability } => format!("{probability:.2}"),
                Answer::Score { score, .. } => format!("{score:.2}"),
            }));
        }
        row
    }

    /// The values to insert for `answers` after the key, in column order;
    /// `None` is a row with no text, whose labels are NULL.
    pub(super) fn cells(&self, answers: Option<&Answers>) -> Vec<Value> {
        let mut cells = Vec::with_capacity(self.labels.len());
        if let Some(answers) = answers {
            for (_, answer) in answers.iter() {
                cells.extend(Self::cells_of(answer));
            }
        } else {
            let nulls = self.labels.len().saturating_sub(1);
            cells.extend(std::iter::repeat_n(Value::Null, nulls));
        }
        cells.push(Value::Boolean(answers.is_some_and(Answers::truncated)));
        cells
    }

    fn cells_of(answer: &Answer) -> Vec<Value> {
        match answer {
            Answer::Choice {
                choice,
                probability,
                confidence,
            } => vec![
                Value::Text(choice.to_string()),
                Value::Double(*probability),
                Value::Double(*confidence),
            ],
            Answer::Noul { probability } => vec![Value::Double(*probability)],
            Answer::Score { score, level, .. } => {
                vec![Value::Double(*score), Value::Int(i32::from(*level))]
            }
        }
    }
}
