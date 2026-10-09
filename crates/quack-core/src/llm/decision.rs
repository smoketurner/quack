//! Decision models: classifiers that score the options of a fixed question
//! and generate no text.
//!
//! Ollama serves them on `/v1/systemone` (0.40.0 or later), a route rig has
//! no client for, so [`DecisionModel`] sends it through the provider's own
//! [`OllamaEndpoint`] and so through its [`LimitedHttp`](super::LimitedHttp):
//! every request takes the provider's permit, passes the workspace's
//! `allowed_providers` check, and uses the proxy settings.
//!
//! One request asks 1 to 64 [`Questions`] about one [`State`]. The model
//! reads at most 512 tokens per question, instructions and options
//! included, and Ollama refuses a longer state with a 400 instead of
//! cutting it. [`Asker::ask`] therefore sends a row's whole state first and
//! cuts it only after a refusal, so [`Answers::truncated`] is exact.

use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;

use rig::http_client::{self, HttpClientExt};
use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{OllamaCapability, OllamaEndpoint, OllamaRunningModels};
use crate::config::{Config, ModelRef};
use crate::error::{Error, Result};

#[cfg(test)]
pub(crate) mod fixture;
#[cfg(test)]
#[path = "../../../quack-testkit/src/decision.rs"]
pub mod stub;
#[cfg(test)]
mod tests;

/// The most questions one request carries.
pub const MAX_QUESTIONS: usize = 64;
/// The fewest options a choice or score question has.
const MIN_OPTIONS: usize = 2;
/// The most options a choice or score question has.
const MAX_OPTIONS: usize = 26;
/// The most bytes the questions of one set take on the wire, a quarter of
/// the 64 KiB request body Ollama accepts.
const MAX_QUESTION_BYTES: usize = 16 * 1024;
/// The most characters of a row's text one request carries.
pub const STATE_CHARS: usize = 4096;
/// Requests one row may spend finding the longest text the model accepts.
const FIT_ATTEMPTS: u32 = 8;
/// The search stops once the longest accepted text is this near a refused one.
const FIT_RESOLUTION: usize = 64;
/// About 200 tokens of English: what a question set must leave room for.
const ROOM_SAMPLE: &str = "The customer wrote to us on Monday about an invoice that was paid twice \
in March. Our records show two charges on the same card, one on the first and one on the fourth, \
both for the full amount of the annual plan. The customer asks for one of the payments to be \
refunded and for a corrected invoice to be sent to the accounting team. They also say that this \
is the second time a billing mistake has happened this year, that the previous one took three \
weeks to resolve, and that they will move to another provider if it is not fixed this week. \
Please review the account history, confirm the duplicate charge, and reply with a date for the \
refund.";

/// Why a set of questions cannot be asked, found when the set is built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum QuestionSetError {
    #[error("a question set needs at least one question")]
    NoQuestions,
    #[error("a question set holds at most {MAX_QUESTIONS} questions, not {0}")]
    TooManyQuestions(usize),
    #[error(
        "question '{question}' has {count} options; a question takes {MIN_OPTIONS} to {MAX_OPTIONS}"
    )]
    Options { question: String, count: usize },
    #[error("'{0}' names two questions or options (names are compared without regard to case)")]
    Duplicate(String),
    #[error(
        "'{0}' cannot name a question: it starts with a letter and holds letters, digits, and \
         underscores, at most {max} characters",
        max = QuestionName::MAX_CHARS
    )]
    BadName(String),
    #[error("the questions take {0} bytes; a request carries at most {MAX_QUESTION_BYTES}")]
    TooLarge(usize),
    #[error("the question set is larger than {max} bytes: it takes {bytes}")]
    SetTooLarge { bytes: usize, max: usize },
    #[error("the question set is not valid: {0}")]
    Unreadable(String),
    #[error("{what} must not be blank and takes at most {max} characters")]
    BadText { what: &'static str, max: usize },
}

impl QuestionSetError {
    /// `text` as a string that is not blank and has at most `max`
    /// characters.
    fn text(what: &'static str, text: String, max: usize) -> std::result::Result<String, Self> {
        if text.trim().is_empty() || text.chars().count() > max {
            return Err(Self::BadText { what, max });
        }
        Ok(text)
    }
}

/// A question's name. It becomes column names, so it starts with a letter,
/// holds letters, digits, and underscores, and is compared without regard
/// to case.
#[derive(Debug, Clone, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct QuestionName(String);

impl QuestionName {
    /// The most characters a name has.
    pub const MAX_CHARS: usize = 48;

    /// The name as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for QuestionName {
    type Error = QuestionSetError;

    fn try_from(name: String) -> std::result::Result<Self, Self::Error> {
        let mut chars = name.chars();
        let valid = chars.next().is_some_and(|c| c.is_ascii_alphabetic())
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
            && name.len() <= Self::MAX_CHARS;
        if valid {
            Ok(Self(name))
        } else {
            Err(QuestionSetError::BadName(name))
        }
    }
}

impl From<QuestionName> for String {
    fn from(name: QuestionName) -> Self {
        name.0
    }
}

impl PartialEq for QuestionName {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq_ignore_ascii_case(&other.0)
    }
}

impl fmt::Display for QuestionName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One option of a choice question: not blank, at most 64 characters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Label(String);

impl Label {
    const MAX_CHARS: usize = 64;

    /// The option as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Label {
    type Error = QuestionSetError;

    fn try_from(text: String) -> std::result::Result<Self, Self::Error> {
        QuestionSetError::text("an option", text, Self::MAX_CHARS).map(Self)
    }
}

impl From<Label> for String {
    fn from(label: Label) -> Self {
        label.0
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What a question asks: not blank, at most 1,000 characters. Only text:
/// the server's object and array forms are not offered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Instructions(String);

impl Instructions {
    const MAX_CHARS: usize = 1000;

    /// The instructions as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Instructions {
    type Error = QuestionSetError;

    fn try_from(text: String) -> std::result::Result<Self, Self::Error> {
        QuestionSetError::text("instructions", text, Self::MAX_CHARS).map(Self)
    }
}

impl From<Instructions> for String {
    fn from(instructions: Instructions) -> Self {
        instructions.0
    }
}

/// A JSON object whose members keep document order and whose keys are
/// unique.
///
/// `serde_json` runs with `preserve_order` in this build, because rig-core
/// enables it, so its `Map` keeps order too. Neither `Map` nor `indexmap`
/// refuses a repeated key: both keep the last one silently. That refusal
/// is why this type exists.
///
/// A document is read member by member and refused at the first member past
/// `MAX`, before any more is read, so a large body costs no more than a
/// small one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unique<K, V, const MAX: usize>(Vec<(K, V)>);

impl<K: PartialEq, V, const MAX: usize> Unique<K, V, MAX> {
    /// `pairs` as an object, or the first key that appears twice.
    ///
    /// # Errors
    ///
    /// Returns the repeated key.
    pub fn from_pairs(mut pairs: Vec<(K, V)>) -> std::result::Result<Self, K> {
        let repeated = pairs.iter().enumerate().find_map(|(at, (key, _))| {
            pairs
                .iter()
                .skip(at.saturating_add(1))
                .any(|(other, _)| other == key)
                .then_some(at)
        });
        match repeated {
            Some(at) => Err(pairs.remove(at).0),
            None => Ok(Self(pairs)),
        }
    }
}

impl<K, V, const MAX: usize> Unique<K, V, MAX> {
    /// The members in the order they were written.
    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.0.iter()
    }

    /// How many members there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<K: Serialize, V: Serialize, const MAX: usize> Serialize for Unique<K, V, MAX> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in &self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

impl<'de, K, V, const MAX: usize> Deserialize<'de> for Unique<K, V, MAX>
where
    K: Deserialize<'de> + PartialEq + fmt::Display,
    V: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct Members<K, V, const MAX: usize>(PhantomData<(K, V)>);

        impl<'de, K, V, const MAX: usize> Visitor<'de> for Members<K, V, MAX>
        where
            K: Deserialize<'de> + PartialEq + fmt::Display,
            V: Deserialize<'de>,
        {
            type Value = Unique<K, V, MAX>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an object")
            }

            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut pairs: Vec<(K, V)> = Vec::new();
                while let Some(key) = map.next_key::<K>()? {
                    if pairs.len() >= MAX {
                        return Err(de::Error::custom(format!("more than {MAX} members")));
                    }
                    if pairs.iter().any(|(earlier, _)| *earlier == key) {
                        return Err(de::Error::custom(format!("the key '{key}' appears twice")));
                    }
                    pairs.push((key, map.next_value::<V>()?));
                }
                Ok(Unique(pairs))
            }
        }

        deserializer.deserialize_map(Members(PhantomData))
    }
}

/// The descriptions of a true-or-false question's two answers. Either may
/// be left out.
#[derive(
    Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct NoulCriteria {
    #[serde(rename = "true", default, skip_serializing_if = "Option::is_none")]
    pub yes: Option<String>,
    #[serde(rename = "false", default, skip_serializing_if = "Option::is_none")]
    pub no: Option<String>,
}

/// One question about a state, serialized exactly as `/v1/systemone` wants
/// it.
#[derive(
    Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, schemars::JsonSchema,
)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum Question {
    /// Pick one of 2 to 26 options, each with an optional description.
    Choice {
        #[schema(value_type = String)]
        #[schemars(with = "String")]
        instructions: Instructions,
        /// The options and what each means, in order.
        #[schema(value_type = BTreeMap<String, Option<String>>)]
        #[schemars(with = "BTreeMap<String, Option<String>>")]
        criteria: Unique<Label, Option<String>, MAX_OPTIONS>,
    },
    /// True or false.
    Noul {
        #[schema(value_type = String)]
        #[schemars(with = "String")]
        instructions: Instructions,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    /// A level on an ordered rubric of 2 to 26 descriptions, lowest first.
    Score {
        #[schema(value_type = String)]
        #[schemars(with = "String")]
        instructions: Instructions,
        criteria: Vec<String>,
    },
}

impl Question {
    /// How many options or levels the question offers; zero for a
    /// true-or-false question.
    #[must_use]
    pub fn option_count(&self) -> usize {
        match self {
            Self::Choice { criteria, .. } => criteria.len(),
            Self::Score { criteria, .. } => criteria.len(),
            Self::Noul { .. } => 0,
        }
    }

    #[must_use]
    pub fn instructions(&self) -> &Instructions {
        match self {
            Self::Choice { instructions, .. }
            | Self::Noul { instructions, .. }
            | Self::Score { instructions, .. } => instructions,
        }
    }
}

/// Up to 64 uniquely named questions, in order. The limits are checked
/// once, when the set is built, so a set that exists can be asked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(try_from = "Unique<QuestionName, Question, MAX_QUESTIONS>")]
#[schemars(with = "BTreeMap<String, Question>")]
pub struct Questions(Unique<QuestionName, Question, MAX_QUESTIONS>);

/// Documented as an object of questions by name.
impl utoipa::PartialSchema for Questions {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        utoipa::openapi::schema::ObjectBuilder::new()
            .schema_type(utoipa::openapi::schema::Type::Object)
            .additional_properties(Some(<Question as utoipa::PartialSchema>::schema()))
            .into()
    }
}

impl utoipa::ToSchema for Questions {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((
            String::from("Question"),
            <Question as utoipa::PartialSchema>::schema(),
        ));
        <Question as utoipa::ToSchema>::schemas(schemas);
    }
}

impl Questions {
    /// The questions `pairs` names.
    ///
    /// # Errors
    ///
    /// Returns a [`QuestionSetError`] when the set is empty, holds more than
    /// 64 questions or a repeated name, a choice or score question has fewer
    /// than 2 or more than 26 options, or the set is over 16 KiB serialized.
    pub fn new(
        pairs: Vec<(QuestionName, Question)>,
    ) -> std::result::Result<Self, QuestionSetError> {
        let unique = Unique::from_pairs(pairs)
            .map_err(|name| QuestionSetError::Duplicate(name.to_string()))?;
        Self::try_from(unique)
    }

    /// The questions in the order they were written.
    pub fn iter(&self) -> impl Iterator<Item = &(QuestionName, Question)> {
        self.0.iter()
    }

    /// How many questions there are.
    #[must_use]
    pub fn count(&self) -> usize {
        self.0.len()
    }
}

impl TryFrom<Unique<QuestionName, Question, MAX_QUESTIONS>> for Questions {
    type Error = QuestionSetError;

    fn try_from(
        unique: Unique<QuestionName, Question, MAX_QUESTIONS>,
    ) -> std::result::Result<Self, Self::Error> {
        match unique.len() {
            0 => return Err(QuestionSetError::NoQuestions),
            count if count > MAX_QUESTIONS => {
                return Err(QuestionSetError::TooManyQuestions(count));
            }
            _ => {}
        }
        for (name, question) in unique.iter() {
            let count = question.option_count();
            let offered = matches!(question, Question::Choice { .. } | Question::Score { .. });
            if offered && !(MIN_OPTIONS..=MAX_OPTIONS).contains(&count) {
                return Err(QuestionSetError::Options {
                    question: name.to_string(),
                    count,
                });
            }
        }
        let bytes = serde_json::to_vec(&unique).map_or(usize::MAX, |bytes| bytes.len());
        if bytes > MAX_QUESTION_BYTES {
            return Err(QuestionSetError::TooLarge(bytes));
        }
        Ok(Self(unique))
    }
}

/// The text a decision model judges: a row's named text fields, in order.
///
/// Fields that are NULL or blank are dropped, and the total is capped at
/// [`STATE_CHARS`] characters as a prefix in field order. The state
/// remembers whether that cap cut anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    fields: Vec<(String, String)>,
    cut: bool,
}

impl State {
    /// The state of `fields`, each a name and the value read for it.
    pub fn new(fields: impl IntoIterator<Item = (String, Option<String>)>) -> Self {
        let mut state = Self {
            fields: Vec::new(),
            cut: false,
        };
        let mut room = STATE_CHARS;
        for (name, value) in fields {
            let Some(value) = value.filter(|v| !v.trim().is_empty()) else {
                continue;
            };
            if room == 0 {
                state.cut = true;
                break;
            }
            let chars = value.chars().count();
            if chars > room {
                let kept = Self::char_prefix(&value, room).to_owned();
                state.fields.push((name, kept));
                state.cut = true;
                break;
            }
            room = room.saturating_sub(chars);
            state.fields.push((name, value));
        }
        state
    }

    /// The smallest state a question set can be asked about.
    fn probe() -> Self {
        Self::new([(String::from("probe"), Some(String::from("-")))])
    }

    /// About 200 tokens of English, to check a set leaves room for text.
    fn room_sample() -> Self {
        Self::new([(String::from("text"), Some(String::from(ROOM_SAMPLE)))])
    }

    /// The characters across all fields.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.iter().map(|(_, v)| v.chars().count()).sum()
    }

    /// Whether every field was NULL or blank: such a row is never sent.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Whether the cap cut the row's text.
    #[must_use]
    pub fn is_cut(&self) -> bool {
        self.cut
    }

    /// The first `chars` characters across the fields, marked cut.
    fn prefix(&self, chars: usize) -> Self {
        let mut room = chars;
        let mut fields = Vec::new();
        for (name, value) in &self.fields {
            if room == 0 {
                break;
            }
            let kept = Self::char_prefix(value, room);
            room = room.saturating_sub(kept.chars().count());
            fields.push((name.clone(), kept.to_owned()));
        }
        Self { fields, cut: true }
    }

    fn char_prefix(text: &str, chars: usize) -> &str {
        text.char_indices()
            .nth(chars)
            .and_then(|(at, _)| text.get(..at))
            .unwrap_or(text)
    }
}

impl Serialize for State {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.fields.len()))?;
        for (name, value) in &self.fields {
            map.serialize_entry(name, value)?;
        }
        map.end()
    }
}

/// What the model answered to one question, checked against the question.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// The chosen option, its probability, and how concentrated the
    /// probabilities are (not the chance the choice is right).
    Choice {
        choice: Label,
        probability: f64,
        confidence: f64,
    },
    /// The probability of true.
    Noul { probability: f64 },
    /// The probability-weighted level, the most probable level (from 0),
    /// and the concentration of the probabilities.
    Score {
        score: f64,
        level: u8,
        confidence: f64,
    },
}

#[derive(Deserialize)]
struct WireAnswers {
    answers: BTreeMap<String, WireAnswer>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum WireAnswer {
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    Noul {
        noul: f64,
    },
    Score {
        score: f64,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
}

impl Answer {
    /// `wire`, the answer to the question `name`, as `question` expects it.
    fn from_wire(name: &QuestionName, question: &Question, wire: WireAnswer) -> Result<Self> {
        let wrong = |what: &str| Error::Llm(format!("ollama's answer to question '{name}' {what}"));
        match (question, wire) {
            (
                Question::Choice { criteria, .. },
                WireAnswer::Choice {
                    choice,
                    probabilities,
                    confidence,
                },
            ) => {
                let label = criteria
                    .iter()
                    .map(|(label, _)| label)
                    .find(|label| label.as_str() == choice)
                    .ok_or_else(|| wrong(&format!("chose '{choice}', which is not an option")))?;
                let probability = probabilities
                    .get(&choice)
                    .copied()
                    .ok_or_else(|| wrong("gives no probability for its choice"))?;
                Ok(Self::Choice {
                    choice: label.clone(),
                    probability,
                    confidence,
                })
            }
            (Question::Noul { .. }, WireAnswer::Noul { noul }) => {
                Ok(Self::Noul { probability: noul })
            }
            (
                Question::Score { .. },
                WireAnswer::Score {
                    score,
                    probabilities,
                    confidence,
                },
            ) => {
                let level = probabilities
                    .iter()
                    .filter_map(|(level, p)| Some((level.parse::<u8>().ok()?, *p)))
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(level, _)| level)
                    .ok_or_else(|| wrong("gives no probability for any level"))?;
                Ok(Self::Score {
                    score,
                    level,
                    confidence,
                })
            }
            (Question::Choice { .. } | Question::Noul { .. } | Question::Score { .. }, _) => {
                Err(wrong("is of another type than the question"))
            }
        }
    }
}

/// The answers to every question of a set about one state, in question
/// order.
#[derive(Debug, Clone, PartialEq)]
pub struct Answers {
    answers: Vec<(QuestionName, Answer)>,
    cut: bool,
}

impl Answers {
    fn from_wire(mut wire: WireAnswers, questions: &Questions, cut: bool) -> Result<Self> {
        let mut answers = Vec::with_capacity(questions.count());
        let mut missing = Vec::new();
        for (name, question) in questions.iter() {
            match wire.answers.remove(name.as_str()) {
                Some(answer) => {
                    answers.push((name.clone(), Answer::from_wire(name, question, answer)?));
                }
                None => missing.push(name.as_str()),
            }
        }
        if !missing.is_empty() {
            return Err(Error::Llm(format!(
                "ollama's answer lacks question{} {}",
                if missing.len() == 1 { "" } else { "s" },
                missing.join(", ")
            )));
        }
        Ok(Self { answers, cut })
    }

    /// The answers in the order of the questions.
    pub fn iter(&self) -> impl Iterator<Item = &(QuestionName, Answer)> {
        self.answers.iter()
    }

    /// Whether the text the model read is shorter than the row's text.
    #[must_use]
    pub fn truncated(&self) -> bool {
        self.cut
    }
}

/// What one row's ask came to.
#[derive(Debug, Clone, PartialEq)]
pub enum Asked {
    /// The model answered every question.
    Answered(Answers),
    /// The model refused the row's text at every length tried, while it
    /// still accepts the questions: a later run may try the row again.
    Unfit,
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    keep_alive: u32,
    state: &'a State,
    questions: &'a Questions,
}

/// The server's refusal of a request.
#[derive(Debug)]
struct Refused {
    status: http::StatusCode,
    message: String,
}

impl Refused {
    /// Ollama says a state is too long with a 400, or a 413 when the body
    /// is over 64 KiB; any other refusal is not about the state.
    fn may_be_length(&self) -> bool {
        matches!(
            self.status,
            http::StatusCode::BAD_REQUEST | http::StatusCode::PAYLOAD_TOO_LARGE
        )
    }
}

impl From<Refused> for Error {
    fn from(refused: Refused) -> Self {
        Self::DecisionRefused(refused.message)
    }
}

/// Ollama's error body.
#[derive(Deserialize)]
struct WireError {
    error: String,
}

/// How the server answered one request.
enum Reply {
    Answered(WireAnswers),
    Refused(Refused),
}

impl Reply {
    /// The reply for `status` and `body`: a 2xx is parsed, a 4xx is a
    /// refusal, and anything else is a failure of the server.
    fn from_status(status: http::StatusCode, body: &[u8]) -> Result<Self> {
        if status.is_success() {
            return serde_json::from_slice(body)
                .map(Self::Answered)
                .map_err(|e| Error::Llm(format!("ollama's answer is not a decision answer: {e}")));
        }
        let message = serde_json::from_slice::<WireError>(body).map_or_else(
            |_| String::from_utf8_lossy(body).trim().to_owned(),
            |e| e.error,
        );
        if status.is_client_error() {
            Ok(Self::Refused(Refused { status, message }))
        } else {
            Err(Error::Llm(format!("ollama answered {status}: {message}")))
        }
    }
}

/// A decision model on an Ollama provider.
#[derive(Clone)]
pub struct DecisionModel {
    endpoint: OllamaEndpoint,
    model: String,
    reference: String,
    keep_alive_seconds: u32,
    concurrency: usize,
}

impl DecisionModel {
    /// The model `[decision].model` names, or `None` when it is unset.
    ///
    /// # Errors
    ///
    /// Returns an error if the setting is invalid, the provider's
    /// credential cannot be resolved, or the workspace's provider list
    /// refuses the model.
    pub async fn from_config(config: &Config) -> Result<Option<Self>> {
        let Some(model) = config.decision_model_ref()? else {
            return Ok(None);
        };
        let key = model
            .provider
            .auth
            .credential(config, model.provider_name)
            .await?;
        Self::with_key(config, model, key.as_deref()).map(Some)
    }

    /// [`Self::from_config`], with `None` also when the workspace's provider
    /// list refuses the model: the model an interface may offer.
    ///
    /// # Errors
    ///
    /// Returns an error if the setting is invalid or the provider's
    /// credential cannot be resolved.
    pub async fn offered(config: &Config) -> Result<Option<Self>> {
        match Self::from_config(config).await {
            Err(refusal) if refusal.is_provider_refusal() => {
                tracing::debug!(error = %refusal, "the decision model is not offered in this workspace");
                Ok(None)
            }
            other => other,
        }
    }

    /// `model` with `key`, already resolved, as its bearer.
    pub(crate) fn with_key(
        config: &Config,
        model: ModelRef<'_>,
        key: Option<&str>,
    ) -> Result<Self> {
        model.permitted()?;
        Ok(Self {
            endpoint: OllamaEndpoint::new(model.provider_name, model.provider, key)?,
            model: model.model.to_owned(),
            reference: model.to_string(),
            keep_alive_seconds: config.decision.keep_alive_minutes.saturating_mul(60),
            concurrency: usize::try_from(model.provider.request_limit().get()).unwrap_or(1),
        })
    }

    /// Requests the provider lets this model have in flight at once.
    #[must_use]
    pub fn concurrency(&self) -> usize {
        self.concurrency
    }

    /// `provider/model`, as the setting writes it.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.reference
    }

    /// What the server says the model can do.
    ///
    /// # Errors
    ///
    /// Returns an error if the server does not answer or does not know the
    /// model.
    pub async fn capabilities(&self) -> Result<Vec<OllamaCapability>> {
        self.endpoint
            .show(&self.model)
            .await
            .map(|shown| shown.capabilities)
            .map_err(|e| Error::Llm(format!("ollama: {e}")))
    }

    /// The digest of the model's weights, as the server lists it: what
    /// tells `ollama pull` of a new version from the same name.
    ///
    /// # Errors
    ///
    /// Returns an error if the server does not answer or does not list the
    /// model.
    pub async fn digest(&self) -> Result<String> {
        let pulled = OllamaRunningModels::pulled(&self.endpoint)
            .await
            .map_err(|e| Error::Llm(format!("ollama: {e}")))?;
        pulled
            .find(&self.model)
            .map(|model| model.digest.clone())
            .ok_or_else(|| {
                Error::Llm(format!(
                    "ollama does not list the model '{}'; pull it with: ollama pull {}",
                    self.model, self.model
                ))
            })
    }

    /// `questions` made askable: the server accepts the set, and leaves
    /// room for a row's text. Both are checked once, here, so a refusal
    /// later is about a row.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DecisionRefused`] when the server refuses the set,
    /// or [`Error::Llm`] when it does not answer.
    pub async fn asker<'a>(&'a self, questions: &'a Questions) -> Result<Asker<'a>> {
        let asker = Asker {
            model: self,
            questions,
        };
        if let Reply::Refused(refused) = asker.post(&State::probe()).await? {
            return Err(refused.into());
        }
        if let Reply::Refused(refused) = asker.post(&State::room_sample()).await? {
            return Err(Error::DecisionRefused(format!(
                "the questions leave too little room for a row's text (about 200 tokens are \
                 needed): {}",
                refused.message
            )));
        }
        Ok(asker)
    }
}

impl fmt::Debug for DecisionModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionModel")
            .field("model", &self.reference)
            .finish_non_exhaustive()
    }
}

/// A [`DecisionModel`] and a question set it accepts.
#[derive(Debug, Clone, Copy)]
pub struct Asker<'a> {
    model: &'a DecisionModel,
    questions: &'a Questions,
}

impl Asker<'_> {
    /// One `POST /v1/systemone` about `state`.
    async fn post(&self, state: &State) -> Result<Reply> {
        let body = serde_json::to_vec(&WireRequest {
            model: &self.model.model,
            keep_alive: self.model.keep_alive_seconds,
            state,
            questions: self.questions,
        })?;
        let endpoint = &self.model.endpoint;
        let request = endpoint
            .request(http::Method::POST, "v1/systemone")
            .body(body)
            .map_err(|e| Error::Llm(e.to_string()))?;
        let sent = endpoint.http.send::<_, Vec<u8>>(request).await;
        match sent {
            Ok(response) => {
                let status = response.status();
                let bytes: Vec<u8> = response
                    .into_body()
                    .await
                    .map_err(|e| Error::Llm(format!("ollama: {e}")))?;
                Reply::from_status(status, &bytes)
            }
            Err(http_client::Error::InvalidStatusCodeWithDetails { status, body, .. }) => {
                Reply::from_status(status, body.as_bytes())
            }
            // A provider-list refusal surfaces from the request gate as an
            // instance error: hand it back typed.
            Err(http_client::Error::Instance(source)) => Err(match source.downcast::<Error>() {
                Ok(error) => *error,
                Err(source) => Error::Llm(format!("ollama: {source}")),
            }),
            Err(other) => Err(Error::Llm(format!("ollama: {other}"))),
        }
    }

    /// Ask the questions about `state`.
    ///
    /// The whole state goes first. Only a 400 or 413 can mean it is too
    /// long, and then this row alone is searched for the longest prefix the
    /// model accepts, within 8 requests. Nothing carries over to the next
    /// row: how much text fits depends on the text.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DecisionRefused`] for any other refusal, and for a
    /// set the server accepted at the start and now refuses; [`Error::Llm`]
    /// when the server fails or its answer is malformed.
    pub async fn ask(&self, state: &State) -> Result<Asked> {
        match self.post(state).await? {
            Reply::Answered(wire) => {
                return Ok(Asked::Answered(Answers::from_wire(
                    wire,
                    self.questions,
                    state.is_cut(),
                )?));
            }
            Reply::Refused(refused) if !refused.may_be_length() => return Err(refused.into()),
            Reply::Refused(_) => {}
        }
        // `fits` is the longest prefix accepted so far and `refused` the
        // shortest refused; the answer lies between them.
        let mut spent = 1_u32;
        let mut fits = 0_usize;
        let mut refused = state.len();
        let mut best = None;
        while spent < FIT_ATTEMPTS
            && refused.saturating_sub(fits) > 1
            && (best.is_none() || refused.saturating_sub(fits) > FIT_RESOLUTION)
        {
            let middle = fits.midpoint(refused);
            spent = spent.saturating_add(1);
            match self.post(&state.prefix(middle)).await? {
                Reply::Answered(wire) => {
                    best = Some(Answers::from_wire(wire, self.questions, true)?);
                    fits = middle;
                }
                Reply::Refused(r) if r.may_be_length() => refused = middle,
                Reply::Refused(r) => return Err(r.into()),
            }
        }
        if let Some(answers) = best {
            return Ok(Asked::Answered(answers));
        }
        match self.post(&State::probe()).await? {
            Reply::Refused(r) => Err(Error::DecisionRefused(format!(
                "the decision model now refuses the question set: {}",
                r.message
            ))),
            Reply::Answered(_) => Ok(Asked::Unfit),
        }
    }
}
