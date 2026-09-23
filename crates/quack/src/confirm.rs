//! The one yes/no prompt the CLI puts before work that spends model calls
//! or changes the workspace. A terminal answers it; anything else is a no,
//! so a pipe or a script never approves spending by accident — `--yes`
//! does that explicitly.

use std::io::{BufRead, IsTerminal, Write};

use anyhow::Result;

/// Whether to ask before going ahead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Confirm {
    /// Ask on the terminal; without one, the answer is no.
    Ask,
    /// Go ahead: `--yes`, or a terminal-session job, which may not read stdin.
    Assume,
}

impl Confirm {
    /// `Assume` when the command's `--yes` was given.
    pub(crate) fn from_yes(yes: bool) -> Self {
        if yes { Self::Assume } else { Self::Ask }
    }

    /// Put `question` and read the answer. Without a terminal on stdin the
    /// answer is no, said on `out` with `flag` when the command has one.
    pub(crate) fn ask(
        self,
        out: &mut impl Write,
        question: &str,
        flag: Option<&str>,
    ) -> Result<bool> {
        if self == Self::Assume {
            return Ok(true);
        }
        let stdin = std::io::stdin();
        if !stdin.is_terminal() {
            match flag {
                Some(flag) => writeln!(
                    out,
                    "{question} No terminal to answer on; {flag} goes ahead."
                )?,
                None => writeln!(out, "{question} No terminal to answer on, so no.")?,
            }
            return Ok(false);
        }
        write!(out, "{question} [y/N] ")?;
        out.flush()?;
        let mut answer = String::new();
        stdin.lock().read_line(&mut answer)?;
        Ok(is_yes(&answer))
    }
}

fn is_yes(answer: &str) -> bool {
    let answer = answer.trim();
    answer.eq_ignore_ascii_case("y") || answer.eq_ignore_ascii_case("yes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_y_or_yes_in_any_case_is_a_yes() {
        for yes in ["y", "Y", "yes", "YES", " Yes\n"] {
            assert!(is_yes(yes), "{yes:?}");
        }
        for no in ["", "n", "no", "yep", "y es", "sure"] {
            assert!(!is_yes(no), "{no:?}");
        }
    }

    #[test]
    fn assume_goes_ahead_without_asking() {
        let mut out = Vec::new();
        assert!(
            Confirm::from_yes(true)
                .ask(&mut out, "Proceed?", Some("--yes"))
                .is_ok_and(|yes| yes)
        );
        assert!(out.is_empty());
        assert_eq!(Confirm::from_yes(false), Confirm::Ask);
    }
}
