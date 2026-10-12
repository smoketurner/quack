//! The agent's answers through `quack -p ... -f json`, scored by the rule
//! that `DuckDB`'s rows are the record of an answer: a table question names
//! the result set one `run_sql` step must have returned, and the prose is a
//! secondary check.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::quack::{Quack, Response, progress};

/// The sentence the prompt tells the model to answer with when the
/// workspace does not hold the answer (`text_to_sql::NOT_IN_DATA`).
pub(crate) const NOT_IN_DATA: &str = "I can't tell from the data in this workspace.";

/// The longest stretch of an answer kept in the report.
const ANSWER_EXCERPT: usize = 400;
/// Result rows kept per outcome in the report.
const RESULT_ROWS_KEPT: usize = 20;

/// One question and what its answer must be. `rows` (or any of
/// `rows_any`, or the rows of `expect_sql` run by the harness, for data
/// that changes) is the result set a `run_sql` step must have returned;
/// `expect` holds prose groups each matched by one of its strings; `shape`
/// says what else the reply must be; a `follow_up` is asked on the same
/// session after it.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AgentCase {
    pub(crate) question: String,
    /// Groups the tally; a follow-up without one takes its parent's.
    #[serde(default)]
    pub(crate) kind: String,
    #[serde(default)]
    pub(crate) expect: Vec<Vec<String>>,
    /// The tools an answer should come from; any one of them will do.
    #[serde(default)]
    pub(crate) tools: Vec<String>,
    /// Whether the answer should cite a passage.
    #[serde(default)]
    pub(crate) cite: bool,
    /// Each expected row's cells all present in one actual row of its own
    /// (extra columns and extra rows are fine: a query that lists every
    /// group and an answer that picks the right one is right), numbers
    /// compared as numbers, text ignoring case.
    #[serde(default)]
    pub(crate) rows: Option<Vec<Vec<Value>>>,
    #[serde(default)]
    pub(crate) rows_any: Vec<Vec<Vec<Value>>>,
    #[serde(default)]
    pub(crate) expect_sql: Option<String>,
    #[serde(default)]
    pub(crate) shape: Shape,
    /// SQL whose result must read the same before and after the turn.
    #[serde(default)]
    pub(crate) unchanged: Option<String>,
    #[serde(default)]
    pub(crate) follow_up: Option<Box<Self>>,
}

/// What a reply must be, beyond the values it holds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Shape {
    #[default]
    Answer,
    /// The exit sentence: the data does not hold it.
    NotInData,
    /// A refused write.
    Refusal,
    /// An answer that takes a reading of an ambiguous question instead of
    /// asking one back.
    Reading,
}

#[derive(Debug, Serialize)]
pub(crate) struct AgentOutcome {
    pub(crate) question: String,
    pub(crate) kind: String,
    pub(crate) correct: bool,
    pub(crate) right_tool: Option<bool>,
    pub(crate) cited: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) rows_matched: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) exit_sentence: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) write_refused: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) unchanged: Option<bool>,
    pub(crate) think_leak: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) consistent: Option<bool>,
    pub(crate) tools: Vec<String>,
    pub(crate) seconds: f64,
    pub(crate) answer: String,
    /// The rows of the last step that returned any, for reading a miss.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) result_rows: Vec<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
}

impl AgentOutcome {
    fn judge(
        case: &AgentCase,
        kind: &str,
        answer: &Response,
        alternatives: &[Vec<Vec<Value>>],
        same: Option<bool>,
        consistent: Option<bool>,
        seconds: f64,
    ) -> Self {
        let tools: Vec<String> = answer.steps.iter().map(|s| s.tool.clone()).collect();
        let text = comparable(&answer.answer);
        let prose = case
            .expect
            .iter()
            .all(|group| group.iter().any(|want| text.contains(&comparable(want))));
        let rows_matched = (!alternatives.is_empty()).then(|| {
            answer
                .steps
                .iter()
                .filter_map(|step| step.result.as_ref())
                .any(|result| {
                    alternatives
                        .iter()
                        .any(|rows| rows_match(rows, &result.rows))
                })
        });
        let exit_sentence =
            (case.shape == Shape::NotInData).then(|| text.contains(&comparable(NOT_IN_DATA)));
        let write_refused = (case.shape == Shape::Refusal).then_some(answer.write_refused);
        let asked_back = case.shape == Shape::Reading && text.contains('?');
        Self {
            correct: prose
                && rows_matched.unwrap_or(true)
                && exit_sentence.unwrap_or(true)
                && write_refused.unwrap_or(true)
                && same.unwrap_or(true)
                && !asked_back,
            right_tool: (!case.tools.is_empty())
                .then(|| tools.iter().any(|t| case.tools.contains(t))),
            cited: case.cite.then_some(!answer.citations.is_empty()),
            rows_matched,
            exit_sentence,
            write_refused,
            unchanged: same,
            think_leak: text.contains("<think>") || text.contains("</think>"),
            consistent,
            tools,
            seconds,
            answer: answer.answer.chars().take(ANSWER_EXCERPT).collect(),
            result_rows: answer
                .steps
                .iter()
                .rev()
                .find_map(|step| step.result.as_ref())
                .map(|result| result.rows.iter().take(RESULT_ROWS_KEPT).cloned().collect())
                .unwrap_or_default(),
            error: None,
            question: case.question.clone(),
            kind: kind.to_owned(),
        }
    }

    fn failed(case: &AgentCase, kind: &str, error: &anyhow::Error, seconds: f64) -> Self {
        Self {
            correct: false,
            right_tool: (!case.tools.is_empty()).then_some(false),
            cited: case.cite.then_some(false),
            rows_matched: None,
            exit_sentence: None,
            write_refused: None,
            unchanged: None,
            think_leak: false,
            consistent: None,
            tools: Vec::new(),
            seconds,
            answer: String::new(),
            result_rows: Vec::new(),
            error: Some(format!("{error:#}")),
            question: case.question.clone(),
            kind: kind.to_owned(),
        }
    }

    fn progress_line(&self, n: usize, total: usize) -> String {
        let mark = match (&self.error, self.correct) {
            (Some(_), _) => "FAIL",
            (None, true) => "ok",
            (None, false) => "MISS",
        };
        let detail = self
            .error
            .as_deref()
            .map_or(String::new(), |e| format!(" ({e})"));
        format!(
            "  [{n}/{total}] {:>5.1}s  {mark:<4} {}: {}{detail}",
            self.seconds, self.kind, self.question
        )
    }
}

/// Text as expected values are matched against it: lowercase, every dash
/// a hyphen and every space a plain one, since a model writes `EVT-0013`
/// with a non-breaking hyphen and `Cedar Hollow` with a non-breaking space
/// as readily as not.
fn comparable(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| match c {
            '\u{2010}'..='\u{2015}' => '-',
            c if c.is_whitespace() => ' ',
            c => c,
        })
        .collect()
}

/// A result cell as it is compared: a number when it is one or reads as
/// one (commas and a dollar sign allowed), else its text in lowercase.
#[derive(Debug, Clone, PartialEq)]
enum Cell {
    Number(f64),
    Text(String),
}

impl Cell {
    fn of(value: &Value) -> Self {
        match value {
            Value::Number(n) => n
                .as_f64()
                .map_or_else(|| Self::Text(n.to_string()), Self::Number),
            Value::String(text) => text
                .trim()
                .replace([',', '$'], "")
                .parse::<f64>()
                .map_or_else(|_| Self::Text(comparable(text.trim())), Self::Number),
            Value::Bool(b) => Self::Text(b.to_string()),
            Value::Null => Self::Text(String::from("null")),
            other => Self::Text(comparable(&other.to_string())),
        }
    }

    fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Number(a), Self::Number(b)) => {
                (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0)
            }
            (Self::Text(a), Self::Text(b)) => a == b,
            _ => false,
        }
    }
}

/// Whether `actual` holds every `expected` row, in any order: each expected
/// row's cells all appear in one actual row not used by another. Extra
/// columns and extra rows pass; the prose check says whether the answer
/// picked the right row of them.
pub(crate) fn rows_match(expected: &[Vec<Value>], actual: &[Vec<Value>]) -> bool {
    if expected.len() > actual.len() {
        return false;
    }
    let actual: Vec<Vec<Cell>> = actual
        .iter()
        .map(|row| row.iter().map(Cell::of).collect())
        .collect();
    let mut used = vec![false; actual.len()];
    expected.iter().all(|row| {
        let wanted: Vec<Cell> = row.iter().map(Cell::of).collect();
        let found = actual.iter().enumerate().position(|(i, cells)| {
            !used.get(i).copied().unwrap_or(true)
                && wanted
                    .iter()
                    .all(|want| cells.iter().any(|cell| cell.same(want)))
        });
        found.is_some_and(|i| {
            if let Some(slot) = used.get_mut(i) {
                *slot = true;
            }
            true
        })
    })
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct Tally {
    pub(crate) n: usize,
    pub(crate) correct: usize,
    pub(crate) right_tool: usize,
    pub(crate) tooled: usize,
    pub(crate) cited: usize,
    pub(crate) citing: usize,
    pub(crate) think_leaks: usize,
    pub(crate) consistent: usize,
    pub(crate) repeated: usize,
    pub(crate) errors: usize,
    pub(crate) seconds: f64,
}

impl Tally {
    fn add(&mut self, o: &AgentOutcome) {
        let count = |flag: bool| usize::from(flag);
        self.n = self.n.saturating_add(1);
        self.correct = self.correct.saturating_add(count(o.correct));
        self.tooled = self.tooled.saturating_add(count(o.right_tool.is_some()));
        self.right_tool = self
            .right_tool
            .saturating_add(count(o.right_tool == Some(true)));
        self.citing = self.citing.saturating_add(count(o.cited.is_some()));
        self.cited = self.cited.saturating_add(count(o.cited == Some(true)));
        self.think_leaks = self.think_leaks.saturating_add(count(o.think_leak));
        self.repeated = self.repeated.saturating_add(count(o.consistent.is_some()));
        self.consistent = self
            .consistent
            .saturating_add(count(o.consistent == Some(true)));
        self.errors = self.errors.saturating_add(count(o.error.is_some()));
        self.seconds += o.seconds;
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct AgentReport {
    pub(crate) n: usize,
    pub(crate) all: Tally,
    pub(crate) by_kind: BTreeMap<String, Tally>,
    pub(crate) cases: Vec<AgentOutcome>,
}

impl AgentReport {
    fn of(outcomes: Vec<AgentOutcome>) -> Self {
        let mut all = Tally::default();
        let mut by_kind: BTreeMap<String, Tally> = BTreeMap::new();
        for o in &outcomes {
            all.add(o);
            by_kind.entry(o.kind.clone()).or_default().add(o);
        }
        Self {
            n: outcomes.len(),
            all,
            by_kind,
            cases: outcomes,
        }
    }

    pub(crate) fn print(&self, out: &mut impl Write, heading: &str) -> Result<()> {
        writeln!(out, "\n{heading}")?;
        writeln!(
            out,
            "{:<12} {:>8} {:>10} {:>10} {:>11} {:>8} {:>8} {:>10}",
            "kind", "correct", "right tool", "cited", "same answer", "<think>", "errors", "seconds"
        )?;
        for (kind, t) in self
            .by_kind
            .iter()
            .chain([(&String::from("all"), &self.all)])
        {
            writeln!(
                out,
                "{kind:<12} {:>8} {:>10} {:>10} {:>11} {:>8} {:>8} {:>10.1}",
                format!("{}/{}", t.correct, t.n),
                format!("{}/{}", t.right_tool, t.tooled),
                format!("{}/{}", t.cited, t.citing),
                format!("{}/{}", t.consistent, t.repeated),
                t.think_leaks,
                t.errors,
                t.seconds,
            )?;
        }
        for case in self
            .cases
            .iter()
            .filter(|c| !c.correct || c.error.is_some())
        {
            writeln!(
                out,
                "  missed: {} -> {}",
                case.question,
                case.error
                    .as_deref()
                    .unwrap_or(&case.answer)
                    .replace('\n', " ")
            )?;
        }
        Ok(())
    }
}

/// Put the questions in `path` to the agent over `workspace`, each on a
/// session of its own and a follow-up on its parent's; `repeat` askings
/// past one check the answer is the same every time.
pub(crate) fn evaluate(
    quack: &Quack,
    workspace: &str,
    path: &Path,
    repeat: u32,
) -> Result<AgentReport> {
    let cases: Vec<AgentCase> = serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
    )?;
    let total: usize = cases
        .iter()
        .map(|c| 1_usize.saturating_add(usize::from(c.follow_up.is_some())))
        .sum();
    progress(&format!(
        "Asking {total} questions ({:.0}s each at most)...",
        quack.timeout.as_secs_f64()
    ));
    let mut outcomes = Vec::with_capacity(total);
    let report = |outcome: AgentOutcome, outcomes: &mut Vec<AgentOutcome>| {
        outcomes.push(outcome);
        if let Some(last) = outcomes.last() {
            progress(&last.progress_line(outcomes.len(), total));
        }
    };
    for case in &cases {
        let (outcome, session) = scored(quack, workspace, case, &case.kind, None, repeat);
        report(outcome, &mut outcomes);
        if let Some(follow_up) = &case.follow_up {
            let kind = if follow_up.kind.is_empty() {
                &case.kind
            } else {
                &follow_up.kind
            };
            let (outcome, _) = scored(quack, workspace, follow_up, kind, session.as_deref(), 1);
            report(outcome, &mut outcomes);
        }
    }
    Ok(AgentReport::of(outcomes))
}

/// Ask `case` and score the answer; returns the session it ran on.
fn scored(
    quack: &Quack,
    workspace: &str,
    case: &AgentCase,
    kind: &str,
    session: Option<&str>,
    repeat: u32,
) -> (AgentOutcome, Option<String>) {
    let mut alternatives: Vec<Vec<Vec<Value>>> = case.rows.iter().cloned().collect();
    alternatives.extend(case.rows_any.iter().cloned());
    if let Some(sql) = &case.expect_sql {
        match quack.query(sql, workspace) {
            Ok(rows) => alternatives.push(rows),
            Err(e) => return (AgentOutcome::failed(case, kind, &e, 0.0), None),
        }
    }
    let read = |sql: Option<&String>| sql.map(|sql| quack.query(sql, workspace)).transpose();
    let before = match read(case.unchanged.as_ref()) {
        Ok(rows) => rows,
        Err(e) => return (AgentOutcome::failed(case, kind, &e, 0.0), None),
    };
    let started = std::time::Instant::now();
    let answer = match quack.ask(&case.question, workspace, session) {
        Ok(answer) => answer,
        Err(e) => {
            return (
                AgentOutcome::failed(case, kind, &e, started.elapsed().as_secs_f64()),
                None,
            );
        }
    };
    let seconds = started.elapsed().as_secs_f64();
    let same = match read(case.unchanged.as_ref()) {
        Ok(after) => before.as_ref().map(|before| Some(before) == after.as_ref()),
        Err(e) => return (AgentOutcome::failed(case, kind, &e, seconds), None),
    };
    let consistent = (repeat > 1).then(|| {
        (1..repeat).all(|_| {
            quack
                .ask(&case.question, workspace, None)
                .is_ok_and(|again| again.answer == answer.answer)
        })
    });
    (
        AgentOutcome::judge(
            case,
            kind,
            &answer,
            &alternatives,
            same,
            consistent,
            seconds,
        ),
        Some(answer.session_id.clone()),
    )
}
