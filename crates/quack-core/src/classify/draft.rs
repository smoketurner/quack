//! The questions a decision model answers about a table's rows, and where
//! they come from: a person's sentence, drafted into questions by the chat
//! model, or a set a person already approved or edited.
//!
//! A [`LabelSet`] is what a person approves and a run records. A [`Draft`]
//! is a set not yet approved, with what a screen shows beside it. Nothing
//! here stores anything: a set is stored by the run that is started with it.

use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::llm::decision::{
    Instructions, OptionLabel, Question, QuestionName, QuestionSetError, Questions, Unique,
};
use crate::storage::profile::TableProfile;
use crate::text::{OneLine, Thousands};

/// Why a column is the key.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum KeyReason {
    /// Named like an id, with every value present and different.
    IdLike,
    /// Every value is present and different, but the name says nothing.
    #[default]
    Unique,
}

text_enum!(KeyReason, "key reason", {
    IdLike => "id_like",
    Unique => "unique",
});
text_enum_sql!(KeyReason);

impl KeyReason {
    /// The reason `name` is a key: what its spelling says.
    #[must_use]
    pub fn of(name: &str) -> Self {
        if TableProfile::is_id_name(name) {
            Self::IdLike
        } else {
            Self::Unique
        }
    }

    /// What a screen adds after the key's name, when the name does not
    /// explain itself.
    #[must_use]
    pub const fn note(self) -> Option<&'static str> {
        match self {
            Self::IdLike => None,
            Self::Unique => Some("all different; not named like an id"),
        }
    }
}

/// The kind of a question, as a person and the chat model name it.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum QuestionKind {
    /// One of 2 or more options.
    Choice,
    /// Yes or no, answered with the probability of yes.
    Noul,
    /// A level on an ordered scale, lowest first.
    Score,
}

impl QuestionKind {
    /// The kind as JSON and the web's select spell it: `choice`, `noul`,
    /// or `score`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Noul => "noul",
            Self::Score => "score",
        }
    }

    /// The kind `as_str` spells, or `None` for another word.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "choice" => Some(Self::Choice),
            "noul" => Some(Self::Noul),
            "score" => Some(Self::Score),
            _ => None,
        }
    }

    /// The word a screen uses: `choice`, `yes/no`, or `score`.
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Noul => "yes/no",
            Self::Score => "score",
        }
    }
}

/// One option of a choice question, with when it applies (empty when the
/// label says it all).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, JsonSchema)]
pub struct DraftOption {
    /// A short label.
    pub label: String,
    /// When the option applies; empty for none.
    #[serde(default)]
    pub description: String,
}

/// A question in the flat form the chat model answers and the web edits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, JsonSchema)]
pub struct DraftQuestion {
    /// The answer's column name.
    pub name: String,
    /// What kind of answer.
    #[serde(rename = "type")]
    pub kind: QuestionKind,
    /// The question about a row.
    pub instructions: String,
    /// A choice question's options; empty for the other kinds.
    #[serde(default)]
    pub options: Vec<DraftOption>,
    /// A score question's levels, lowest first; empty for the other kinds.
    #[serde(default)]
    pub levels: Vec<String>,
}

impl DraftQuestion {
    /// The question as the decision model asks it.
    ///
    /// # Errors
    ///
    /// Returns the [`QuestionSetError`] for a name, instructions, or option
    /// that breaks its rule, or a repeated option.
    pub fn into_question(self) -> Result<(QuestionName, Question), QuestionSetError> {
        let name = QuestionName::try_from(self.name)?;
        let instructions = Instructions::try_from(self.instructions)?;
        let question = match self.kind {
            QuestionKind::Choice => {
                let mut pairs = Vec::with_capacity(self.options.len());
                for option in self.options {
                    let description = Some(option.description.trim())
                        .filter(|text| !text.is_empty())
                        .map(str::to_owned);
                    pairs.push((
                        OptionLabel::try_from(option.label.trim().to_owned())?,
                        description,
                    ));
                }
                Question::Choice {
                    instructions,
                    criteria: Unique::from_pairs(pairs)
                        .map_err(|label| QuestionSetError::Duplicate(label.to_string()))?,
                }
            }
            QuestionKind::Noul => Question::Noul {
                instructions,
                criteria: None,
            },
            QuestionKind::Score => Question::Score {
                instructions,
                criteria: self.levels,
            },
        };
        Ok((name, question))
    }

    /// The flat form of a question.
    #[must_use]
    pub fn of(name: &QuestionName, question: &Question) -> Self {
        let (kind, options, levels) = match question {
            Question::Choice { criteria, .. } => (
                QuestionKind::Choice,
                criteria
                    .iter()
                    .map(|(label, description)| DraftOption {
                        label: label.to_string(),
                        description: description.clone().unwrap_or_default(),
                    })
                    .collect(),
                Vec::new(),
            ),
            Question::Noul { .. } => (QuestionKind::Noul, Vec::new(), Vec::new()),
            Question::Score { criteria, .. } => (QuestionKind::Score, Vec::new(), criteria.clone()),
        };
        Self {
            name: name.to_string(),
            kind,
            instructions: question.instructions().as_str().to_owned(),
            options,
            levels,
        }
    }

    /// `-- name (choice: a, b): instructions`, as the approval card has it.
    pub(super) fn card_line(&self) -> String {
        let options: Vec<String> = match self.kind {
            QuestionKind::Choice => self
                .options
                .iter()
                .map(|option| OneLine(&option.label).to_string())
                .collect(),
            QuestionKind::Score => self.levels.iter().map(|l| OneLine(l).to_string()).collect(),
            QuestionKind::Noul => Vec::new(),
        };
        let what = if options.is_empty() {
            self.kind.word().to_owned()
        } else {
            format!("{}: {}", self.kind.word(), options.join(", "))
        };
        format!(
            "-- {} ({what}): {}",
            OneLine(&self.name),
            OneLine(&self.instructions)
        )
    }

    /// A question from the web's four fields: the options or levels are
    /// one per line. A choice line is `label: when it applies`, split at
    /// the first `": "`, or just a label; a score line is a level taken
    /// whole; a yes/no takes none. Blank lines are skipped.
    #[must_use]
    pub fn from_lines(name: &str, kind: QuestionKind, instructions: &str, lines: &str) -> Self {
        let lines = lines.lines().map(str::trim).filter(|line| !line.is_empty());
        let (options, levels) = match kind {
            QuestionKind::Choice => (
                lines
                    .map(|line| match line.split_once(": ") {
                        Some((label, description)) => DraftOption {
                            label: label.trim().to_owned(),
                            description: description.trim().to_owned(),
                        },
                        None => DraftOption {
                            label: line.to_owned(),
                            description: String::new(),
                        },
                    })
                    .collect(),
                Vec::new(),
            ),
            QuestionKind::Score => (Vec::new(), lines.map(str::to_owned).collect()),
            QuestionKind::Noul => (Vec::new(), Vec::new()),
        };
        Self {
            name: name.trim().to_owned(),
            kind,
            instructions: instructions.trim().to_owned(),
            options,
            levels,
        }
    }

    /// The options or levels as the web's textarea holds them, one per
    /// line: the inverse of [`Self::from_lines`].
    #[must_use]
    pub fn to_lines(&self) -> String {
        match self.kind {
            QuestionKind::Choice => self
                .options
                .iter()
                .map(|option| {
                    if option.description.is_empty() {
                        option.label.clone()
                    } else {
                        format!("{}: {}", option.label, option.description)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
            QuestionKind::Score => self.levels.join("\n"),
            QuestionKind::Noul => String::new(),
        }
    }
}

/// What the chat model answers when it drafts questions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DraftAnswer {
    /// The columns whose text answers the questions.
    pub text_columns: Vec<String>,
    /// The questions.
    pub questions: Vec<DraftQuestion>,
}

impl DraftAnswer {
    /// The questions as the decision model asks them.
    ///
    /// # Errors
    ///
    /// Returns the [`QuestionSetError`] of the first question that breaks a
    /// rule, or of the set.
    pub fn questions(&self) -> Result<Questions, QuestionSetError> {
        let pairs = self
            .questions
            .iter()
            .cloned()
            .map(DraftQuestion::into_question)
            .collect::<Result<Vec<_>, _>>()?;
        Questions::new(pairs)
    }

    /// The questions of `questions`, in the flat form.
    #[must_use]
    pub fn of(text_columns: &[String], questions: &Questions) -> Self {
        Self {
            text_columns: text_columns.to_vec(),
            questions: questions
                .iter()
                .map(|(name, question)| DraftQuestion::of(name, question))
                .collect(),
        }
    }
}

/// What a person approves and a run records: which column is the key,
/// which columns the model reads, and the questions it answers about each
/// row. A request carries one back to run exactly what was drafted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LabelSet {
    /// The column that identifies a row, joining the labels to the table.
    pub key_column: String,
    /// Why it is the key; worked out again from the table when the set is
    /// checked, whatever a request says.
    #[serde(default)]
    pub key_reason: KeyReason,
    /// The columns whose text the model reads, one to eight.
    pub text_columns: Vec<String>,
    /// The questions asked about every row.
    pub questions: Questions,
    /// What the person asked for, in their words, when the questions were
    /// drafted from it.
    #[serde(default)]
    pub sentence: Option<String>,
}

impl LabelSet {
    /// The questions as the screens list them, a block of two lines each:
    /// the name, the kind, and the options or levels, then the question.
    /// `detailed` adds what each option means and numbers the levels.
    #[must_use]
    pub fn block(&self, detailed: bool) -> String {
        let width = self
            .questions
            .iter()
            .map(|(name, _)| name.as_str().chars().count())
            .max()
            .unwrap_or(0);
        let mut lines = Vec::new();
        for (name, question) in self.questions.iter() {
            let flat = DraftQuestion::of(name, question);
            lines.push(format!(
                "  {:<width$}   {:<6}   {}",
                name.as_str(),
                flat.kind.word(),
                Self::choices(&flat, detailed)
            ));
            lines.push(format!(
                "  {:<width$}   {:<6}   \"{}\"",
                "",
                "",
                OneLine(&flat.instructions)
            ));
        }
        lines.join("\n")
    }

    /// The options of a question on one line.
    fn choices(question: &DraftQuestion, detailed: bool) -> String {
        match question.kind {
            QuestionKind::Choice => question
                .options
                .iter()
                .map(|option| {
                    if detailed && !option.description.is_empty() {
                        format!(
                            "{} ({})",
                            OneLine(&option.label),
                            OneLine(&option.description)
                        )
                    } else {
                        OneLine(&option.label).to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(" \u{b7} "),
            QuestionKind::Score => question
                .levels
                .iter()
                .enumerate()
                .map(|(at, level)| {
                    if detailed {
                        format!("{at} {}", OneLine(level))
                    } else {
                        OneLine(level).to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(" \u{b7} "),
            QuestionKind::Noul => String::new(),
        }
    }

    /// The text columns in words: `subject`, `subject and body`,
    /// `a, b and c`.
    #[must_use]
    pub fn columns_in_words(&self) -> String {
        let names: Vec<String> = self
            .text_columns
            .iter()
            .map(|c| OneLine(c).to_string())
            .collect();
        match names.split_last() {
            Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
            Some((last, _)) => last.clone(),
            None => String::new(),
        }
    }
}

/// How a draft came to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DraftOrigin {
    /// The chat model drafted questions from a sentence.
    Drafted,
    /// The chat model revised the last approved questions for a sentence.
    Revised,
    /// The last approved questions, as they were.
    Reused,
    /// A set a person sent back: edited in the form, or posted.
    Given,
}

/// A set not yet approved: the table, where its labels would go, and the
/// set, with what a screen shows beside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub struct Draft {
    /// The table whose rows are labelled, by its exact name.
    pub table: String,
    /// The table the labels go to.
    pub output_table: String,
    /// The set a run would record.
    pub set: LabelSet,
    /// The table's rows when the draft was made.
    pub rows: u64,
    /// The rows the chat model looked at to draft; zero for a set that was
    /// not drafted.
    pub sample_rows: u32,
    /// How the draft came to be.
    pub origin: DraftOrigin,
    /// The chat model that drafted or revised the questions.
    pub drafted_by_model: Option<String>,
    /// When the set was last approved, when it was.
    pub approved_at: Option<String>,
    /// Columns the set reads that the table no longer has.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub columns_gone: Vec<String>,
}

impl Draft {
    /// The table's line: `support_tickets: 104,233 rows, key id, text in
    /// subject and body.`
    #[must_use]
    pub fn header(&self) -> String {
        let key = match self.set.key_reason.note() {
            Some(note) => format!("{} ({note})", OneLine(&self.set.key_column)),
            None => OneLine(&self.set.key_column).to_string(),
        };
        format!(
            "{}: {} rows, key {key}, text in {}.",
            OneLine(&self.table),
            Thousands(self.rows),
            self.set.columns_in_words()
        )
    }
}

impl fmt::Display for Draft {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.header())?;
        write!(f, "{}", self.set.block(false))
    }
}
