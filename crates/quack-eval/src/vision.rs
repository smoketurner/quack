//! The vision model reading one notice three ways: as an image, as the
//! only picture on a PDF page, and as a figure inside a DOCX. Scored by
//! the notice's words found in the ingested text; no file name holds one.

use std::io::Write;
use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use crate::quack::{Quack, SearchMode, progress};

/// Words on the notice that only reading the picture can supply.
pub(crate) const NOTICE_WORDS: &[&str] = &["notice", "220", "october", "delgado"];

const CASES: &[(&str, &str)] = &[
    ("image", "posted-sign.jpg"),
    ("scanned pdf", "records-scan.pdf"),
    ("docx figure", "shelter-memo.docx"),
];

#[derive(Debug, Serialize)]
pub(crate) struct VisionCase {
    pub(crate) name: String,
    pub(crate) found: Vec<String>,
    pub(crate) seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct VisionReport {
    pub(crate) model: String,
    pub(crate) cases: Vec<VisionCase>,
}

impl VisionReport {
    pub(crate) fn print(&self, out: &mut impl Write) -> Result<()> {
        writeln!(
            out,
            "\nVision ({} words to find, model {})",
            NOTICE_WORDS.len(),
            self.model
        )?;
        writeln!(out, "{:<16} {:>8} {:>10}", "case", "found", "seconds")?;
        for case in &self.cases {
            writeln!(
                out,
                "{:<16} {:>8} {:>10.1}{}",
                case.name,
                format!("{}/{}", case.found.len(), NOTICE_WORDS.len()),
                case.seconds,
                case.error
                    .as_deref()
                    .map_or_else(String::new, |e| format!("  error: {e}")),
            )?;
        }
        Ok(())
    }
}

pub(crate) fn evaluate(
    quack: &Quack,
    workspace: &str,
    fixture: &Path,
    model: &str,
) -> Result<VisionReport> {
    progress(&format!("Reading {} images with {model}...", CASES.len()));
    let dir = fixture.join("vision");
    let mut cases = Vec::with_capacity(CASES.len());
    for (name, file) in CASES {
        let run = quack.ingest(&dir.join(file), workspace)?;
        let (found, error) = if run.ok() {
            let mut found = Vec::new();
            for word in NOTICE_WORDS {
                let hits = quack.search(word, workspace, SearchMode::Keyword, Some(file))?;
                if hits.iter().any(|h| h.content.to_lowercase().contains(word)) {
                    found.push((*word).to_owned());
                }
            }
            (found, None)
        } else {
            (Vec::new(), Some(run.failure()))
        };
        progress(&format!(
            "  [{}/{}] {:>5.1}s  {:<4} {name}{}",
            cases.len().saturating_add(1),
            CASES.len(),
            run.seconds,
            if found.len() == NOTICE_WORDS.len() {
                "ok"
            } else {
                "MISS"
            },
            error.as_deref().map_or(String::new(), |e| format!(": {e}")),
        ));
        cases.push(VisionCase {
            name: (*name).to_owned(),
            found,
            seconds: run.seconds,
            error,
        });
    }
    Ok(VisionReport {
        model: model.to_owned(),
        cases,
    })
}
