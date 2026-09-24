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

/// One CSV record: the fields as [`CsvField`]s joined by commas, without
/// the line ending.
#[derive(Debug, Clone, Copy)]
pub struct CsvRecord<'a, S>(pub &'a [S]);

impl<S: AsRef<str>> fmt::Display for CsvRecord<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, field) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(",")?;
            }
            write!(f, "{}", CsvField(field.as_ref()))?;
        }
        Ok(())
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

    #[test]
    fn records_join_quoted_fields() {
        assert_eq!(CsvRecord(&["a", "b,c", ""]).to_string(), "a,\"b,c\",");
        assert_eq!(CsvRecord::<&str>(&[]).to_string(), "");
    }
}
