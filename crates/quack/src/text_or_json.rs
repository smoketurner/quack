//! Output for a person or for a program.

use std::fmt;
use std::io::Write;

use anyhow::Result;
use serde::Serialize;

/// How a command prints what it found: the text rendering, or JSON
/// (`--json`, `--format json`): one pretty document for a single value,
/// one object per line for a listing.
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

    /// A listing: `line` for each row, or `empty` when there are none; or
    /// one JSON object per row.
    pub(crate) fn write_rows<T: Serialize, W: Write>(
        self,
        out: &mut W,
        rows: &[T],
        empty: &str,
        mut line: impl FnMut(&mut W, &T) -> std::io::Result<()>,
    ) -> Result<()> {
        match self {
            Self::Json => {
                for row in rows {
                    serde_json::to_writer(&mut *out, row)?;
                    writeln!(out)?;
                }
            }
            Self::Text if rows.is_empty() => writeln!(out, "{empty}")?,
            Self::Text => {
                for row in rows {
                    line(out, row)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(format: TextOrJson, items: &[&str]) -> String {
        let mut out = Vec::new();
        let written = format.write_rows(&mut out, items, "Nothing yet.", |out, item| {
            writeln!(out, "- {item}")
        });
        assert!(written.is_ok());
        String::from_utf8_lossy(&out).into_owned()
    }

    /// Text lines or the empty note for a person; one object per line, and
    /// no note, for a program.
    #[test]
    fn listings_are_lines_for_text_and_one_object_per_row_for_json() {
        assert_eq!(rows(TextOrJson::Text, &["a", "b"]), "- a\n- b\n");
        assert_eq!(rows(TextOrJson::Text, &[]), "Nothing yet.\n");
        assert_eq!(rows(TextOrJson::Json, &["a", "b"]), "\"a\"\n\"b\"\n");
        assert_eq!(rows(TextOrJson::Json, &[]), "");
    }
}
