//! Grounding: `DuckDB` provides the answers, and the chat model only writes
//! the SQL and the tool calls that fetch them (design doc 7.7). Every figure
//! an answer states must therefore come from this turn's searches and
//! queries (a value in a result, a passage retrieved) or from the question.
//! A figure that is none of these was made up or worked out by the model,
//! and the answer is asked for again once ([`super::hooks::Unsourced`]); if
//! the second answer still states one, the turn shows the query results
//! themselves instead of the model's sentence ([`fallback`]).
//!
//! A figure matches a source when it is the same number to the precision it
//! is written in: `1.76 million` matches `1764000`, `18%` matches `18` or
//! `0.18`, `$891,000` matches `891000`. Digits that are part of a name
//! (`EF2`, `SR-8841`, `qwen2.5`), citation markers (`[3]`), and list numbers
//! at the start of a line are not figures.

use std::fmt::Write as _;

use super::events::ToolStep;
use crate::storage::workspace::QueryResults;

/// A figure as an answer or a source writes it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Figure {
    /// As written: `$1,764,000`, `18%`, `4.2M`, `1.76 million`.
    pub(crate) text: String,
    /// The number written, before any scale.
    value: f64,
    /// Half the unit of the last digit written: how far the written number
    /// may be from the one it rounds.
    tolerance: f64,
    /// What a scale word or suffix multiplies it by (`M`, `million`).
    scale: Option<f64>,
    percent: bool,
}

impl Figure {
    /// Every number the figure can stand for, each with how far a source
    /// may be from it: as written, scaled, and as a ratio for a percentage.
    fn readings(&self) -> Vec<(f64, f64)> {
        let mut readings = vec![(self.value, self.tolerance)];
        if let Some(scale) = self.scale {
            readings.push((self.value * scale, self.tolerance * scale));
        }
        if self.percent {
            readings.push((self.value / 100.0, self.tolerance / 100.0));
        }
        readings
    }
}

/// A scale word or suffix after a number, and what it multiplies by.
fn scale_of(word: &str) -> Option<f64> {
    Some(match word.to_ascii_lowercase().as_str() {
        "k" | "thousand" => 1e3,
        "m" | "mn" | "million" | "millions" => 1e6,
        "b" | "bn" | "billion" | "billions" => 1e9,
        "t" | "trillion" | "trillions" => 1e12,
        _ => return None,
    })
}

/// The figures in `text`, in order.
pub(crate) fn figures(text: &str) -> Vec<Figure> {
    let scan = Scan(text.chars().collect());
    let mut found = Vec::new();
    let mut i = 0;
    while let Some(c) = scan.at(i) {
        let starts = c.is_ascii_digit()
            && !scan
                .at(i.wrapping_sub(1))
                .is_some_and(|p| p.is_ascii_digit() || p == '.');
        if !starts {
            i = i.saturating_add(1);
            continue;
        }
        let (end, decimals) = scan.number_end(i);
        if let Some(figure) = scan.figure(i, end, decimals) {
            found.push(figure);
        }
        i = end;
    }
    found
}

/// Text as characters, read for figures.
struct Scan(Vec<char>);

impl Scan {
    fn at(&self, i: usize) -> Option<char> {
        self.0.get(i).copied()
    }

    fn text(&self, from: usize, to: usize) -> String {
        self.0.get(from..to).unwrap_or_default().iter().collect()
    }

    /// Where the number starting at `start` ends, past digits, thousands
    /// separators, and one decimal point; and how many decimals it has.
    fn number_end(&self, start: usize) -> (usize, Option<i32>) {
        let digit = |i: usize| self.at(i).is_some_and(|d| d.is_ascii_digit());
        let mut end = start;
        let mut decimals: Option<i32> = None;
        loop {
            match self.at(end) {
                Some(d) if d.is_ascii_digit() => {
                    if let Some(n) = decimals.as_mut() {
                        *n = n.saturating_add(1);
                    }
                }
                Some(',')
                    if decimals.is_none()
                        && (1..=3).all(|k| digit(end.saturating_add(k)))
                        && !digit(end.saturating_add(4)) => {}
                Some('.') if decimals.is_none() && digit(end.saturating_add(1)) => {
                    decimals = Some(0);
                }
                _ => return (end, decimals),
            }
            end = end.saturating_add(1);
        }
    }

    /// Whether the number at `start..end` is not a figure: part of a name
    /// (a letter, `_`, or a letter-joined hyphen before it), a citation
    /// marker (`[3]`, `[^3]`, `【3】`), or a list number (nothing but spaces
    /// or a bullet before it on its line, and `.` or `)` and a space after).
    fn not_a_figure(&self, start: usize, end: usize) -> bool {
        let before = self.at(start.wrapping_sub(1));
        let after = self.at(end);
        let named = before.is_some_and(|p| p.is_alphabetic() || p == '_')
            || (before == Some('-')
                && self
                    .at(start.wrapping_sub(2))
                    .is_some_and(char::is_alphabetic));
        let marker =
            matches!(before, Some('[' | '^' | '【')) && matches!(after, Some(']' | '】' | ','));
        let listed = matches!(after, Some('.' | ')'))
            && self
                .at(end.saturating_add(1))
                .is_none_or(char::is_whitespace)
            && self
                .0
                .get(..start)
                .unwrap_or_default()
                .iter()
                .rev()
                .take_while(|p| **p != '\n')
                .all(|p| matches!(p, ' ' | '\t' | '-' | '*'));
        named || marker || listed
    }

    /// The scale after the number ending at `end`, and where the figure
    /// ends: a suffix (`4.2M`) or a word after one space (`1.76 million`).
    /// `None` when letters attached to it make it a name or an ordinal
    /// (`2nd`, `10x`, `3D`).
    fn scale_after(&self, end: usize) -> Option<(Option<f64>, usize)> {
        let word_end = |from: usize| {
            let mut w = from;
            while self.at(w).is_some_and(char::is_alphabetic) {
                w = w.saturating_add(1);
            }
            w
        };
        let attached = word_end(end);
        if attached > end {
            return scale_of(&self.text(end, attached)).map(|s| (Some(s), attached));
        }
        if self.at(end) == Some(' ') {
            let from = end.saturating_add(1);
            let to = word_end(from);
            let word = self.text(from, to);
            if word.len() > 1
                && let Some(s) = scale_of(&word)
            {
                return Some((Some(s), to));
            }
        }
        Some((None, end))
    }

    /// The figure at `start..end`, when it is one.
    fn figure(&self, start: usize, end: usize, decimals: Option<i32>) -> Option<Figure> {
        if self.not_a_figure(start, end) {
            return None;
        }
        let (scale, close) = self.scale_after(end)?;
        let percent = self.at(close) == Some('%')
            || self
                .text(close, close.saturating_add(8))
                .to_ascii_lowercase()
                .starts_with(" percent");
        let digits: String = self
            .text(start, end)
            .chars()
            .filter(|d| *d != ',')
            .collect();
        let value = digits.parse::<f64>().ok().filter(|v| v.is_finite())?;
        let tolerance = 0.5 / 10f64.powi(decimals.unwrap_or(0));
        let mut text: String = match self.at(start.wrapping_sub(1)) {
            Some(p @ ('$' | '€' | '£' | '-')) => p.to_string(),
            _ => String::new(),
        };
        text.push_str(&self.text(start, close));
        if self.at(close) == Some('%') {
            text.push('%');
        }
        Some(Figure {
            text,
            value,
            tolerance,
            scale,
            percent,
        })
    }
}

/// The numbers a turn showed the model: what an answer's figures must come
/// from.
#[derive(Debug, Default, Clone)]
pub(crate) struct Sources {
    /// Every number seen, as written and scaled, kept sorted.
    values: Vec<f64>,
}

impl Sources {
    /// Take the numbers in `text`, something the model was shown.
    pub(crate) fn saw(&mut self, text: &str) {
        for figure in figures(text) {
            for (value, _) in figure.readings() {
                self.values.push(value.abs());
            }
        }
        self.values.sort_by(f64::total_cmp);
        self.values.dedup();
    }

    /// Whether some source is `value` to within `tolerance`.
    fn holds(&self, value: f64, tolerance: f64) -> bool {
        let low = value.abs() - tolerance;
        let from = self.values.partition_point(|v| *v < low);
        self.values
            .get(from)
            .is_some_and(|v| *v <= value.abs() + tolerance)
    }

    /// The figures of `answer` no source holds, each once, as written.
    pub(crate) fn unsourced(&self, answer: &str) -> Vec<String> {
        let mut missing: Vec<String> = Vec::new();
        for figure in figures(answer) {
            let sourced = figure
                .readings()
                .into_iter()
                .any(|(value, tolerance)| self.holds(value, tolerance));
            if !sourced && !missing.contains(&figure.text) {
                missing.push(figure.text);
            }
        }
        missing
    }
}

/// What the model is told when its answer states figures no source holds.
#[must_use]
pub(crate) fn feedback(unsourced: &[String], looked: bool) -> String {
    let opening = if looked {
        ""
    } else {
        "This turn ran no query and read no document. "
    };
    format!(
        "{opening}These figures in your answer do not appear in any query result or passage from \
         this turn: {}. Every figure must come from the workspace: run a query that returns it \
         (DuckDB does the arithmetic: sums, differences, ratios, percentages, rankings), quote it \
         from a passage you read, or leave it out. Then answer again.",
        unsourced.join(", ")
    )
}

/// The answer a turn gives when its model's answer still states figures no
/// source holds: what the queries returned, which `DuckDB` computed, in place
/// of the model's sentence.
#[must_use]
pub(crate) fn fallback(unsourced: &[String], steps: &[ToolStep]) -> String {
    let mut out = format!(
        "This answer could not be confirmed from the workspace's data: {} did not come from any \
         query result or passage this turn, so the model's answer is not shown.",
        unsourced.join(", ")
    );
    // Writing to a String does not fail; were it to, what was written stands.
    if write_results(&mut out, steps).is_err() {
        tracing::warn!("could not write the query results into the answer");
    }
    out
}

/// What the turn's queries returned, after `out`.
fn write_results(out: &mut String, steps: &[ToolStep]) -> std::fmt::Result {
    let results: Vec<(&ToolStep, &QueryResults)> = steps
        .iter()
        .filter(|s| s.tool.takes_sql())
        .filter_map(|s| s.result.as_ref().map(|r| (s, r)))
        .filter(|(_, r)| !r.columns.is_empty())
        .collect();
    if results.is_empty() {
        out.push_str(
            "\n\nNo query returned rows this turn. Ask again, naming the table or the document \
             that holds the figure.",
        );
        return Ok(());
    }
    out.push_str("\n\nWhat the queries returned:");
    for (n, (step, result)) in results.iter().enumerate() {
        write!(
            out,
            "\n\nQuery {}:\n```sql\n{}\n```\n",
            n.saturating_add(1),
            step.detail.trim()
        )?;
        write_table(out, result)?;
        let shown = u64::try_from(result.rows.len()).unwrap_or(u64::MAX);
        if let Some(total) = step.rows.filter(|total| *total > shown) {
            write!(out, "\n({shown} of {total} rows shown)")?;
        }
    }
    Ok(())
}

/// `result` as a Markdown table, each cell as `DuckDB` returned it.
fn write_table(out: &mut String, result: &QueryResults) -> std::fmt::Result {
    let cell = |value: &serde_json::Value| match value {
        serde_json::Value::Null => String::from("NULL"),
        serde_json::Value::String(s) => s.replace('|', "\\|").replace('\n', " "),
        other => other.to_string(),
    };
    writeln!(out, "| {} |", result.columns.join(" | "))?;
    out.push('|');
    for _ in &result.columns {
        out.push_str(" --- |");
    }
    for row in &result.rows {
        out.push_str("\n|");
        for value in row {
            write!(out, " {} |", cell(value))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::events::ToolName;

    fn shown(text: &str) -> Vec<String> {
        figures(text).into_iter().map(|f| f.text).collect()
    }

    fn sources(texts: &[&str]) -> Sources {
        let mut sources = Sources::default();
        for text in texts {
            sources.saw(text);
        }
        sources
    }

    #[test]
    fn figures_skip_names_markers_and_list_numbers() {
        assert_eq!(
            shown(
                "1. An EF2 tornado in SR-8841 caused $850,000 [1] of damage, 4.2M in 2024.\n2) qwen2.5 said 18%."
            ),
            vec!["$850,000", "4.2M", "2024", "18%"]
        );
        assert_eq!(shown("about 1.76 million dollars"), vec!["1.76 million"]);
        assert_eq!(shown("the 2nd and 10x cases"), Vec::<String>::new());
        assert_eq!(shown("It is 25."), vec!["25"]);
    }

    /// A figure matches a source to the precision it is written in, as
    /// written, scaled, or as a ratio.
    #[test]
    fn a_figure_matches_its_source_to_its_written_precision() {
        let s = sources(&["| total |\n| 1764000 |", "ratio 0.18", "850.00K"]);
        assert!(s.unsourced("Texas lost $1,764,000.").is_empty());
        assert!(s.unsourced("about 1.76 million").is_empty());
        assert!(s.unsourced("about 1.8 million").is_empty());
        assert!(s.unsourced("18% of them").is_empty());
        assert!(s.unsourced("850,000 in damage").is_empty());
        assert_eq!(
            s.unsourced("Texas lost $2,655,000, about 2.7 million."),
            vec!["$2,655,000", "2.7 million"]
        );
    }

    #[test]
    fn the_fallback_shows_what_the_queries_returned() {
        let step = ToolStep {
            tool: ToolName::RunSql,
            detail: String::from("SELECT state, sum(d) AS total FROM events GROUP BY 1"),
            summary: String::from("2 rows"),
            rows: Some(3),
            result: Some(QueryResults {
                columns: vec![String::from("state"), String::from("total")],
                rows: vec![
                    vec![serde_json::json!("TEXAS"), serde_json::json!(1_764_000)],
                    vec![serde_json::json!("IOWA"), serde_json::json!(12)],
                ],
            }),
            duration_ms: 1,
            run: None,
        };
        let text = fallback(&[String::from("$2,655,000")], &[step]);
        assert!(text.contains("$2,655,000 did not come from"), "{text}");
        assert!(
            text.contains("| state | total |\n| --- | --- |\n| TEXAS | 1764000 |"),
            "{text}"
        );
        assert!(text.contains("(2 of 3 rows shown)"), "{text}");
        let none = fallback(&[String::from("12")], &[]);
        assert!(none.contains("No query returned rows"), "{none}");
    }
}
