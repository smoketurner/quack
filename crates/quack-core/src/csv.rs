//! One CSV field as RFC 4180 writes it, for every CSV quack produces:
//! query results, sheet and import staging files, the audit export, and
//! the web console's download.

use std::fmt;

/// A value written as one CSV field: quoted, with embedded quotes doubled,
/// when it holds a comma, a quote, or a line break (`\n` or `\r`); bare
/// otherwise.
#[derive(Debug, Clone, Copy)]
pub struct CsvField<'a>(pub &'a str);

impl fmt::Display for CsvField<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.contains([',', '"', '\n', '\r']) {
            write!(f, "\"{}\"", self.0.replace('"', "\"\""))
        } else {
            f.write_str(self.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_are_quoted_only_when_they_need_it() {
        let written = |value: &str| CsvField(value).to_string();
        assert_eq!(written("plain"), "plain");
        assert_eq!(written(""), "");
        assert_eq!(written("a,b"), "\"a,b\"");
        assert_eq!(written("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(written("two\nlines"), "\"two\nlines\"");
        assert_eq!(written("carriage\rreturn"), "\"carriage\rreturn\"");
    }
}
