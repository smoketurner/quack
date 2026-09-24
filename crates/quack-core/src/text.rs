//! Small text conventions every interface shares.

use std::fmt;

/// Text as a caller gave it for an optional field.
pub trait NonBlankText {
    /// The text trimmed, or `None` when nothing is left: a caller who sends
    /// a blank field means to leave it out.
    fn non_blank(&self) -> Option<&str>;
}

impl NonBlankText for str {
    fn non_blank(&self) -> Option<&str> {
        let trimmed = self.trim();
        (!trimmed.is_empty()).then_some(trimmed)
    }
}

/// `1 table`, `3 tables`: a count and its noun, plural past one.
pub struct Count<'a>(pub usize, pub &'a str);

impl fmt::Display for Count<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(n, noun) = self;
        if *n == 1 {
            write!(f, "1 {noun}")
        } else {
            write!(f, "{n} {noun}s")
        }
    }
}

/// A count of model tokens. Everything quack sizes before a call (the
/// history trim, pinned text, the workspace context, Ollama's window)
/// estimates at four characters per token, which errs on the side of
/// sending less; a provider's own count is `analysis::agent::TokenUsage`.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct Tokens(u32);

impl Tokens {
    const CHARS_PER_TOKEN: usize = 4;

    #[must_use]
    pub const fn new(count: u32) -> Self {
        Self(count)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The estimate for `text`.
    #[must_use]
    pub fn estimate(text: &str) -> Self {
        Self::of_chars(text.len())
    }

    /// The estimate for this many characters.
    #[must_use]
    pub fn of_chars(chars: usize) -> Self {
        Self(u32::try_from(chars.div_ceil(Self::CHARS_PER_TOKEN)).unwrap_or(u32::MAX))
    }

    /// How many characters this many tokens covers, for cutting text to a
    /// budget.
    #[must_use]
    pub fn chars(self) -> usize {
        usize::try_from(self.0)
            .unwrap_or(usize::MAX)
            .saturating_mul(Self::CHARS_PER_TOKEN)
    }

    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }
}

impl fmt::Display for Tokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_text_is_absent_and_the_rest_is_trimmed() {
        assert_eq!("  a b ".non_blank(), Some("a b"));
        assert_eq!(" \t\n".non_blank(), None);
        assert_eq!("".non_blank(), None);
    }

    #[test]
    fn counts_are_plural_except_one() {
        assert_eq!(Count(0, "table").to_string(), "0 tables");
        assert_eq!(Count(1, "table").to_string(), "1 table");
        assert_eq!(Count(2, "key").to_string(), "2 keys");
    }
}
