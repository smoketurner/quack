//! Output for a person or for a program.

use std::fmt;
use std::io::Write;

use anyhow::Result;
use serde::Serialize;

/// How a command prints what it found: the text rendering, or pretty
/// JSON (`--json`, `--format json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextOrJson {
    Text,
    Json,
}

impl TextOrJson {
    /// `--json` given, or not.
    pub(crate) const fn of(json: bool) -> Self {
        if json { Self::Json } else { Self::Text }
    }

    /// `value` as its text rendering, or as one pretty JSON document.
    pub(crate) fn write<T: Serialize + fmt::Display>(
        self,
        out: &mut impl Write,
        value: &T,
    ) -> Result<()> {
        match self {
            Self::Json => writeln!(out, "{}", serde_json::to_string_pretty(value)?)?,
            Self::Text => write!(out, "{value}")?,
        }
        Ok(())
    }
}
