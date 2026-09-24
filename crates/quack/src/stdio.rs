//! A file argument where `-` means standard input or output.

use std::convert::Infallible;
use std::fmt;
use std::io::Read;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result};

/// A file to read or write, or `-` for standard input or output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StdioPath {
    Stdio,
    Path(PathBuf),
}

impl StdioPath {
    /// The whole text: standard input, or the file.
    ///
    /// # Errors
    ///
    /// Returns an error when the input cannot be read or is not UTF-8.
    pub(crate) fn read_to_string(&self) -> Result<String> {
        match self {
            Self::Stdio => {
                let mut text = String::new();
                std::io::stdin()
                    .read_to_string(&mut text)
                    .context("failed to read from stdin")?;
                Ok(text)
            }
            Self::Path(path) => std::fs::read_to_string(path)
                .with_context(|| format!("failed to read {}", path.display())),
        }
    }
}

impl FromStr for StdioPath {
    type Err = Infallible;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Ok(if text == "-" {
            Self::Stdio
        } else {
            Self::Path(PathBuf::from(text))
        })
    }
}

impl fmt::Display for StdioPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stdio => f.write_str("-"),
            Self::Path(path) => write!(f, "{}", path.display()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dash_is_stdio_and_anything_else_a_path() {
        assert_eq!("-".parse(), Ok(StdioPath::Stdio));
        assert_eq!(
            "notes.md".parse(),
            Ok(StdioPath::Path(PathBuf::from("notes.md")))
        );
        assert_eq!(StdioPath::Stdio.to_string(), "-");
        assert_eq!(StdioPath::Path(PathBuf::from("./-x")).to_string(), "./-x");
    }
}
