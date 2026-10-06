//! A table a parser found in a document (a ruled PDF table, a DOCX or
//! ODT table, an HTML `<table>`, a Markdown pipe table): its header and
//! rows, rendered as a pipe-delimited Markdown table for one chunk of its
//! own, and as CSV when it is big enough to load as a table of the
//! workspace.

/// A rectangular table: one header row and the data rows under it, every
/// row as wide as the header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Table {
    /// A table from its rows in order, the first as the header, each cell
    /// collapsed to one line. Rows shorter than the header are padded,
    /// longer ones cut. `None` without a header and at least one data row,
    /// or when the table is one column wide: that is a list, not a grid.
    #[must_use]
    pub fn from_rows(rows: Vec<Vec<String>>) -> Option<Self> {
        let mut rows = rows.into_iter();
        let header: Vec<String> = rows.next()?.iter().map(|c| cell(c)).collect();
        let width = header.len();
        if width < 2 {
            return None;
        }
        let rows: Vec<Vec<String>> = rows
            .filter(|row| row.iter().any(|c| !c.trim().is_empty()))
            .map(|row| {
                let mut cells: Vec<String> = row.iter().take(width).map(|c| cell(c)).collect();
                cells.resize(width, String::new());
                cells
            })
            .collect();
        if rows.is_empty() {
            return None;
        }
        Some(Self { header, rows })
    }

    /// Columns.
    #[must_use]
    pub fn width(&self) -> usize {
        self.header.len()
    }

    /// The Markdown rendering: the header, the separator, one line per row.
    /// A chunker that splits it keeps the first two lines on every piece.
    #[must_use]
    pub fn render(&self) -> String {
        let separator: Vec<String> = self.header.iter().map(|_| String::from("---")).collect();
        let mut lines = vec![Self::line(&self.header), Self::line(&separator)];
        lines.extend(self.rows.iter().map(|row| Self::line(row)));
        lines.join("\n")
    }

    /// One Markdown table line.
    fn line(cells: &[String]) -> String {
        format!("| {} |", cells.join(" | "))
    }

    /// The CSV the workspace's reader loads, header first.
    ///
    /// # Errors
    ///
    /// Returns an error if a row cannot be written.
    pub fn csv(&self) -> std::io::Result<Vec<u8>> {
        let mut writer = csv::Writer::from_writer(Vec::new());
        let header: Vec<String> = self.header.iter().map(|c| c.replace("\\|", "|")).collect();
        writer.write_record(&header)?;
        for row in &self.rows {
            let cells: Vec<String> = row.iter().map(|c| c.replace("\\|", "|")).collect();
            writer.write_record(&cells)?;
        }
        writer.into_inner().map_err(csv::IntoInnerError::into_error)
    }

    /// The header and the rows of a rendered table's lines, for a chunker
    /// splitting it: the two header lines, then the row lines. `None` for
    /// text that is not a rendering of [`Self::render`].
    #[must_use]
    pub fn split_rendered(markdown: &str) -> Option<(Vec<&str>, Vec<&str>)> {
        let mut lines = markdown.lines();
        let header = lines.next()?;
        let separator = lines.next()?;
        if !header.starts_with('|') || !separator.starts_with('|') {
            return None;
        }
        Some((vec![header, separator], lines.collect()))
    }
}

/// A cell as the rendering shows it: whitespace collapsed to one line,
/// pipes escaped so they cannot end the cell.
#[must_use]
pub fn cell(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace('|', "\\|")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(rows: &[&[&str]]) -> Vec<Vec<String>> {
        rows.iter()
            .map(|r| r.iter().map(|c| String::from(*c)).collect())
            .collect()
    }

    #[test]
    fn renders_a_pipe_table_padding_short_rows_and_escaping_pipes() {
        let table = Table::from_rows(rows(&[
            &["Item", "Amount"],
            &["Flood | storm", "1,200"],
            &["Fire"],
        ]))
        .unwrap_or_else(|| Table {
            header: Vec::new(),
            rows: Vec::new(),
        });
        assert_eq!(table.width(), 2);
        assert_eq!(
            table.render(),
            "| Item | Amount |\n| --- | --- |\n| Flood \\| storm | 1,200 |\n| Fire |  |"
        );
        let rendered = table.render();
        let (header, body) = Table::split_rendered(&rendered).unwrap_or_default();
        assert_eq!(header.len(), 2);
        assert_eq!(body.len(), 2);
        let csv = String::from_utf8(table.csv().unwrap_or_default()).unwrap_or_default();
        assert_eq!(csv, "Item,Amount\nFlood | storm,\"1,200\"\nFire,\n");
    }

    #[test]
    fn a_list_or_a_header_alone_is_not_a_table() {
        assert_eq!(Table::from_rows(rows(&[&["only"], &["one"]])), None);
        assert_eq!(Table::from_rows(rows(&[&["a", "b"]])), None);
        assert_eq!(Table::from_rows(rows(&[&["a", "b"], &["", " "]])), None);
        assert_eq!(Table::split_rendered("plain text\nmore"), None);
    }
}
