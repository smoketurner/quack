//! The bound on what a compressed upload may inflate to. The upload limit
//! counts compressed bytes only, so every archive part is read through one
//! budget for the whole file.

use std::io::{self, Cursor, Read};

use crate::error::Error;

/// The decompressed bytes one file may still yield, shared by every part
/// read from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecompressionBudget {
    max_mb: u64,
    left: u64,
    spent: bool,
}

impl DecompressionBudget {
    /// A budget of `max_mb` megabytes (`[ingestion].max_decompressed_mb`).
    #[must_use]
    pub const fn megabytes(max_mb: u64) -> Self {
        Self {
            max_mb,
            left: max_mb.saturating_mul(1024 * 1024),
            spent: false,
        }
    }

    /// `inner`, with every byte it yields taken from this budget.
    pub fn reader<R: Read>(&mut self, inner: R) -> Budgeted<'_, R> {
        Budgeted {
            inner,
            budget: self,
        }
    }

    /// The refusal, once a read has asked for more than the budget held.
    #[must_use]
    pub fn refusal(&self) -> Option<Error> {
        self.spent.then(|| {
            Error::Ingestion(format!(
                "the file decompresses to more than [ingestion].max_decompressed_mb ({} MB)",
                self.max_mb
            ))
        })
    }

    /// A failed read of `part` as ingestion reports it: the refusal when
    /// the budget ran out, else the reader's own error.
    #[must_use]
    pub fn read_error(&self, part: &str, error: &io::Error) -> Error {
        self.refusal()
            .unwrap_or_else(|| Error::Ingestion(format!("cannot read {part}: {error}")))
    }

    /// Inflate every entry of a zip archive against this budget and keep
    /// nothing, so a parser that takes the whole file can be handed bytes
    /// already known to fit. Bytes that are not a zip archive inflate
    /// nothing and pass.
    ///
    /// # Errors
    ///
    /// Returns the refusal when the entries together exceed the budget.
    pub fn admit_zip(&mut self, data: &[u8]) -> Result<(), Error> {
        let Ok(mut archive) = zip::ZipArchive::new(Cursor::new(data)) else {
            return Ok(());
        };
        for index in 0..archive.len() {
            // An entry that cannot be opened or read cannot be inflated by
            // the parser either; the parser reports it.
            let Ok(entry) = archive.by_index(index) else {
                continue;
            };
            let copied = io::copy(&mut self.reader(entry), &mut io::sink());
            if copied.is_err()
                && let Some(refusal) = self.refusal()
            {
                return Err(refusal);
            }
        }
        Ok(())
    }
}

/// A reader that fails once its budget is spent.
#[derive(Debug)]
pub struct Budgeted<'b, R> {
    inner: R,
    budget: &'b mut DecompressionBudget,
}

impl<R: Read> Read for Budgeted<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let left = self.budget.left;
        // One byte past the budget tells a file that fits exactly from one
        // that does not.
        let read = (&mut self.inner).take(left.saturating_add(1)).read(buf)?;
        let taken = u64::try_from(read).unwrap_or(u64::MAX);
        if taken > left {
            self.budget.left = 0;
            self.budget.spent = true;
            return Err(io::Error::other("decompression budget spent"));
        }
        self.budget.left = left.saturating_sub(taken);
        Ok(read)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingestion::office::tests::package;

    const MEGABYTE: usize = 1024 * 1024;
    const THREE_QUARTERS: usize = 768 * 1024;

    #[test]
    fn a_read_within_the_budget_passes_and_one_byte_more_is_refused() {
        let exact = vec![b' '; MEGABYTE];
        let mut budget = DecompressionBudget::megabytes(1);
        let mut out = Vec::new();
        let read = budget.reader(exact.as_slice()).read_to_end(&mut out);
        assert_eq!(read.ok(), Some(MEGABYTE));
        assert!(budget.refusal().is_none());

        let over = vec![b' '; MEGABYTE.saturating_add(1)];
        let mut budget = DecompressionBudget::megabytes(1);
        let read = budget.reader(over.as_slice()).read_to_end(&mut Vec::new());
        assert!(read.is_err());
        let refusal = budget.refusal().map(|e| e.to_string()).unwrap_or_default();
        assert!(
            refusal.contains("[ingestion].max_decompressed_mb (1 MB)"),
            "{refusal}"
        );
    }

    #[test]
    fn readers_of_one_budget_share_it() {
        let part = vec![b' '; THREE_QUARTERS];
        let mut budget = DecompressionBudget::megabytes(1);
        assert!(
            budget
                .reader(part.as_slice())
                .read_to_end(&mut Vec::new())
                .is_ok()
        );
        let second = budget.reader(part.as_slice()).read_to_end(&mut Vec::new());
        assert!(second.is_err());
        assert!(budget.refusal().is_some());
    }

    #[test]
    fn a_read_error_that_is_not_the_budget_keeps_its_own_text() {
        let budget = DecompressionBudget::megabytes(1);
        let error = budget
            .read_error("word/document.xml", &io::Error::other("bad checksum"))
            .to_string();
        assert!(
            error.contains("cannot read word/document.xml: bad checksum"),
            "{error}"
        );
    }

    #[test]
    fn a_zip_is_admitted_by_the_sum_of_its_entries() {
        let half = " ".repeat(THREE_QUARTERS);
        let fits = package(&[("a.xml", &half)]);
        assert!(fits.len() < 16 * 1024, "{} bytes", fits.len());
        assert!(DecompressionBudget::megabytes(1).admit_zip(&fits).is_ok());

        let bomb = package(&[("a.xml", &half), ("b.xml", &half)]);
        let refused = DecompressionBudget::megabytes(1)
            .admit_zip(&bomb)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(refused.contains("max_decompressed_mb"), "{refused}");
        assert!(DecompressionBudget::megabytes(2).admit_zip(&bomb).is_ok());
    }

    #[test]
    fn bytes_that_are_not_a_zip_are_admitted() {
        let mut budget = DecompressionBudget::megabytes(0);
        assert!(budget.admit_zip(b"not an archive").is_ok());
    }
}
