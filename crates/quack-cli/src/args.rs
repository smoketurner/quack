//! Arguments the command line and the terminal's slash commands share.

use std::io::Write;

use clap::ValueEnum;
use quack_core::error::Result as CoreResult;
use quack_core::storage::sessions::{ChatMode, ExportFormat};
use quack_core::storage::workspace::QueryResults;

/// The answer mode as a command-line or slash-command argument.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ModeArg {
    /// General knowledge allowed; cite when a source was used
    Chat,
    /// Every claim must come from a retrieved source
    Query,
}

/// `--sql` or `--markdown`: how `export` writes a session.
#[derive(clap::Args)]
pub struct ExportFlags {
    /// Every executed statement, each preceded by its question
    #[arg(long, conflicts_with = "markdown")]
    sql: bool,

    /// Questions, steps, and answers as Markdown (the default)
    #[arg(long)]
    markdown: bool,
}

impl ExportFlags {
    #[must_use]
    pub const fn format(&self) -> ExportFormat {
        if self.sql {
            ExportFormat::Sql
        } else {
            ExportFormat::Markdown
        }
    }
}

impl From<ModeArg> for ChatMode {
    fn from(mode: ModeArg) -> Self {
        match mode {
            ModeArg::Chat => Self::Chat,
            ModeArg::Query => Self::Query,
        }
    }
}

/// How `-q` and `saved run` print a result set: the formats of `--format`
/// without `text`, which prints an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum QueryFormat {
    /// Aligned text table
    Table,
    /// One JSON document
    Json,
    /// One JSON object per line
    Ndjson,
    /// Comma-separated values with a header row
    Csv,
    /// GitHub-flavored Markdown table
    Markdown,
}

impl QueryFormat {
    /// Without `--format`: a table on a terminal, ndjson into a pipe.
    #[must_use]
    pub fn default_for(stdout_is_tty: bool) -> Self {
        if stdout_is_tty {
            Self::Table
        } else {
            Self::Ndjson
        }
    }

    /// Print `results` in this format.
    ///
    /// # Errors
    ///
    /// Returns the error from rendering a value or writing to `out`.
    pub fn write(self, results: &QueryResults, out: &mut impl Write) -> CoreResult<()> {
        match self {
            Self::Table => results.write_table(out),
            Self::Json => results.write_json(out),
            Self::Ndjson => results.write_ndjson(out),
            Self::Csv => results.write_csv(out),
            Self::Markdown => results.write_markdown(out),
        }
    }
}
