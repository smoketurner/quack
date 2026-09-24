//! `quack doctor`: every check in `quack_core::doctor`, one line each, with
//! the fix under anything that needs one.

use std::io::Write;

use anyhow::Result;
use quack_core::doctor::{Report, Status};

use crate::text_or_json::TextOrJson;

pub(crate) fn write(out: &mut impl Write, report: &Report, format: TextOrJson) -> Result<()> {
    if format == TextOrJson::Json {
        writeln!(out, "{}", serde_json::to_string_pretty(report)?)?;
        return Ok(());
    }

    let width = report
        .checks
        .iter()
        .map(|c| c.area.as_str().len())
        .max()
        .unwrap_or(0);
    let indent = " ".repeat(width.saturating_add(8));
    for check in &report.checks {
        let mut summary = check.summary.lines();
        writeln!(
            out,
            "{:<4}  {:width$}  {}",
            check.status,
            check.area,
            summary.next().unwrap_or_default()
        )?;
        for line in summary {
            writeln!(out, "{indent}{line}")?;
        }
        if let Some(fix) = &check.fix {
            for (i, line) in fix.lines().enumerate() {
                let lead = if i == 0 { "→ " } else { "  " };
                writeln!(out, "{indent}{lead}{line}")?;
            }
        }
    }
    let failures = report.count(Status::Fail);
    let warnings = report.count(Status::Warn);
    writeln!(
        out,
        "\n{}",
        match (failures, warnings) {
            (0, 0) => String::from("Everything checks out."),
            (0, w) => format!("No failures; {w} to look at."),
            (f, w) => format!("{f} failed, {w} to look at."),
        }
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use quack_core::doctor::{Area, Check};

    #[test]
    #[expect(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        reason = "test: serde_json indexing yields Null rather than panicking"
    )]
    fn text_puts_the_fix_under_its_check_and_json_counts_failures() {
        let report = Report {
            checks: vec![
                Check {
                    area: Area::Config,
                    status: Status::Ok,
                    summary: String::from("loaded"),
                    fix: None,
                },
                Check {
                    area: Area::ChatModel,
                    status: Status::Fail,
                    summary: String::from("o/m: cannot reach"),
                    fix: Some(String::from("ollama serve\nsecond line")),
                },
            ],
        };
        let mut text = Vec::new();
        write(&mut text, &report, TextOrJson::Text).unwrap();
        let text = String::from_utf8(text).unwrap();
        assert!(
            text.contains("fail  chat model  o/m: cannot reach"),
            "{text}"
        );
        assert!(text.contains("→ ollama serve"));
        assert!(text.contains("  second line"));
        assert!(text.ends_with("1 failed, 0 to look at.\n"));

        let mut json = Vec::new();
        write(&mut json, &report, TextOrJson::Json).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&json).unwrap();
        assert_eq!(value["ok"], false);
        assert_eq!(value["failures"], 1);
        assert_eq!(value["checks"][1]["fix"], "ollama serve\nsecond line");
    }
}
