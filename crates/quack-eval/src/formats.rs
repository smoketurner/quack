//! Ingestion by file format: every file `formats.json` names goes through
//! `quack ingest`, and the result is checked the way a person would check
//! it, by searching for a phrase or counting a table's rows.

use std::io::Write;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::quack::{Quack, SearchMode, progress};

const WORKSPACE: &str = "formats";

/// One file and what it should become: a document names a phrase its text
/// must hold; a tabular file names its table and the rows it must have; a
/// file with nothing checkable beyond loading is `ready`.
#[derive(Debug, Deserialize)]
struct FormatCase {
    file: String,
    #[serde(default)]
    phrase: Option<String>,
    #[serde(default)]
    table: Option<String>,
    #[serde(default)]
    rows: Option<u64>,
    #[serde(default)]
    ready: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct FormatOutcome {
    pub(crate) file: String,
    pub(crate) ok: bool,
    /// What came of it: the table's rows, or whether the phrase was found.
    pub(crate) detail: String,
    pub(crate) seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct FormatsReport {
    pub(crate) n: usize,
    pub(crate) ok: usize,
    pub(crate) cases: Vec<FormatOutcome>,
}

impl FormatsReport {
    pub(crate) fn print(&self, out: &mut impl Write) -> Result<()> {
        writeln!(
            out,
            "\nIngestion by format: {}/{} files became what the manifest says",
            self.ok, self.n
        )?;
        writeln!(
            out,
            "{:<28} {:>4} {:<36} {:>8}",
            "file", "ok", "result", "seconds"
        )?;
        for case in &self.cases {
            writeln!(
                out,
                "{:<28} {:>4} {:<36} {:>8.2}{}",
                case.file,
                if case.ok { "ok" } else { "MISS" },
                case.detail,
                case.seconds,
                case.error
                    .as_deref()
                    .map_or_else(String::new, |e| format!("  error: {e}")),
            )?;
        }
        Ok(())
    }
}

pub(crate) fn evaluate(quack: &Quack, fixture: &Path) -> Result<FormatsReport> {
    let cases: Vec<FormatCase> =
        serde_json::from_str(&std::fs::read_to_string(fixture.join("formats.json"))?)?;
    progress(&format!(
        "Ingesting {} files, one per format...",
        cases.len()
    ));
    quack.create_workspace(WORKSPACE)?;
    let dir = fixture.join("formats");
    let mut outcomes = Vec::with_capacity(cases.len());
    for case in cases {
        let run = quack.ingest(&dir.join(&case.file), WORKSPACE)?;
        let (ok, detail, error) = if run.ok() {
            match check(quack, &case) {
                Ok((ok, detail)) => (ok, detail, None),
                Err(e) => (false, String::new(), Some(e.to_string())),
            }
        } else {
            (false, String::new(), Some(run.failure()))
        };
        outcomes.push(FormatOutcome {
            file: case.file,
            ok,
            detail,
            seconds: run.seconds,
            error,
        });
    }
    Ok(FormatsReport {
        n: outcomes.len(),
        ok: outcomes.iter().filter(|o| o.ok).count(),
        cases: outcomes,
    })
}

fn check(quack: &Quack, case: &FormatCase) -> Result<(bool, String)> {
    if let Some(table) = &case.table {
        let rows = quack
            .query(
                &format!(
                    "SELECT count(*) AS n FROM \"{}\"",
                    table.replace('"', "\"\"")
                ),
                WORKSPACE,
            )?
            .first()
            .and_then(|r| r.first())
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        return Ok((
            rows >= case.rows.unwrap_or(1),
            format!("table {table}: {rows} rows"),
        ));
    }
    if let Some(phrase) = &case.phrase {
        let wanted = phrase.to_lowercase();
        let hits = quack.search(phrase, WORKSPACE, SearchMode::Keyword, Some(&case.file))?;
        let found = hits
            .iter()
            .any(|hit| hit.content.to_lowercase().contains(&wanted));
        return Ok((
            found,
            format!("phrase {}", if found { "found" } else { "missing" }),
        ));
    }
    Ok((case.ready, String::from("ingested")))
}
