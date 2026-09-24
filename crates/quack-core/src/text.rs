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
