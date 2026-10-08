//! Excel workbooks through `calamine` (pure Rust; the `DuckDB` `excel`
//! extension cannot be compiled into the static binary). Each sheet is
//! written as CSV under `files/` and loaded as its own table.

use std::io::Cursor;

use calamine::{Data, Reader, Sheets, XlsxError};

use super::budget::DecompressionBudget;
use crate::error::{Error, Result};

/// The most cells one sheet may write to its CSV: its non-empty rows times
/// the width of its used columns. A dense 1,048,576-row sheet 95 columns
/// wide fits; a sheet whose rows each hold one cell in column `XFD` does
/// not, and nothing wider than this is loaded.
pub const MAX_SHEET_CELLS: u64 = 100_000_000;

/// A non-empty cell at its zero-based position.
struct SheetCell {
    row: u32,
    col: u32,
    value: Data,
}

/// One sheet of a workbook as CSV bytes, with its name.
#[derive(Debug)]
pub struct SheetCsv {
    pub sheet: String,
    pub csv: Vec<u8>,
    pub rows: usize,
}

/// Every non-empty sheet as CSV, in workbook order. The first row is the
/// header (as `DuckDB`'s reader sniffs it).
///
/// `calamine` inflates a zipped workbook (XLSX, XLSM, XLSB, ODS) itself, so
/// its entries are inflated against `budget` first. An XLS file is not
/// compressed and passes untouched.
///
/// A sheet is never read as a dense grid: `calamine`'s `worksheet_range`
/// allocates the bounding box of the used cells, so a sheet holding `A1`
/// and `XFD1048576` would ask for 17 billion cells. XLSX and XLSB stream
/// their cells instead; the sparse cells are the only thing in memory, and
/// the CSV is bounded by [`MAX_SHEET_CELLS`].
///
/// # Errors
///
/// Returns an error when the bytes are not a workbook, no sheet has data,
/// the workbook inflates past `budget`, or a sheet would write more than
/// [`MAX_SHEET_CELLS`] cells.
pub fn sheets(data: &[u8], mut budget: DecompressionBudget) -> Result<Vec<SheetCsv>> {
    budget.admit_zip(data)?;
    let mut workbook = calamine::open_workbook_auto_from_rs(Cursor::new(data))
        .map_err(|e| Error::Ingestion(format!("not a spreadsheet: {e}")))?;
    let names = workbook.sheet_names();
    let mut out = Vec::new();
    for name in names {
        let cells = match sheet_cells(&mut workbook, &name) {
            Ok(cells) => cells,
            Err(e) => {
                tracing::warn!(sheet = %name, error = %e, "skipping unreadable sheet");
                continue;
            }
        };
        let Some(sheet) = SparseSheet::new(&name, cells)? else {
            tracing::info!(sheet = %name, "skipping sheet without data rows");
            continue;
        };
        out.push(sheet.to_csv(name)?);
    }
    if out.is_empty() {
        return Err(Error::Ingestion(String::from(
            "no data: every sheet is empty or has only a header row",
        )));
    }
    Ok(out)
}

/// A sheet's non-empty cells, streamed where the format allows it.
///
/// XLS parses every sheet when the workbook opens, and ODS caps a sheet at
/// 100 million cells itself; neither offers a cell reader, so both come
/// through the range `calamine` already built.
fn sheet_cells(
    workbook: &mut Sheets<Cursor<&[u8]>>,
    name: &str,
) -> std::result::Result<Vec<SheetCell>, calamine::Error> {
    let mut cells = Vec::new();
    match workbook {
        Sheets::Xlsx(xlsx) => {
            let mut reader = match xlsx.worksheet_cells_reader(name) {
                Ok(reader) => reader,
                // A chart sheet has a name but no cells.
                Err(XlsxError::NotAWorksheet(_)) => return Ok(cells),
                Err(e) => return Err(e.into()),
            };
            while let Some(cell) = reader.next_cell()? {
                let (row, col) = cell.get_position();
                let value: Data = cell.get_value().clone().into();
                if !matches!(value, Data::Empty) {
                    cells.push(SheetCell { row, col, value });
                }
            }
        }
        Sheets::Xlsb(xlsb) => {
            let mut reader = xlsb.worksheet_cells_reader(name)?;
            while let Some(cell) = reader.next_cell()? {
                let (row, col) = cell.get_position();
                let value: Data = cell.get_value().clone().into();
                if !matches!(value, Data::Empty) {
                    cells.push(SheetCell { row, col, value });
                }
            }
        }
        Sheets::Xls(_) | Sheets::Ods(_) => {
            let range = workbook.worksheet_range(name)?;
            let (row0, col0) = range.start().unwrap_or_default();
            for (row, col, value) in range.used_cells() {
                let Ok(row) = u32::try_from(row) else {
                    continue;
                };
                let Ok(col) = u32::try_from(col) else {
                    continue;
                };
                cells.push(SheetCell {
                    row: row0.saturating_add(row),
                    col: col0.saturating_add(col),
                    value: value.clone(),
                });
            }
        }
    }
    Ok(cells)
}

/// A sheet's non-empty cells in row-major order with the column span they
/// occupy; its CSV has one field per column of the span and skips rows
/// without cells.
struct SparseSheet {
    cells: Vec<SheetCell>,
    first_col: u32,
    width: usize,
    rows: usize,
}

impl SparseSheet {
    /// `None` when the sheet has fewer than two rows with cells (a table
    /// needs a header and a data row).
    fn new(name: &str, mut cells: Vec<SheetCell>) -> Result<Option<Self>> {
        cells.sort_unstable_by_key(|cell| (cell.row, cell.col));
        let (mut first_col, mut last_col) = (u32::MAX, 0u32);
        let mut rows = 0usize;
        let mut last_row = None;
        for cell in &cells {
            let (row, col) = (cell.row, cell.col);
            first_col = first_col.min(col);
            last_col = last_col.max(col);
            if last_row != Some(row) {
                rows = rows.saturating_add(1);
                last_row = Some(row);
            }
        }
        if rows < 2 {
            return Ok(None);
        }
        let width = u64::from(last_col.saturating_sub(first_col)).saturating_add(1);
        let total = u64::try_from(rows)
            .unwrap_or(u64::MAX)
            .saturating_mul(width);
        if total > MAX_SHEET_CELLS {
            return Err(Error::Ingestion(format!(
                "sheet `{name}` would load {total} cells ({rows} rows by {width} columns); \
                 the limit is {MAX_SHEET_CELLS}"
            )));
        }
        let Ok(width) = usize::try_from(width) else {
            return Err(Error::Ingestion(format!(
                "sheet `{name}` is {width} columns wide; the limit is {MAX_SHEET_CELLS} cells"
            )));
        };
        Ok(Some(Self {
            cells,
            first_col,
            width,
            rows,
        }))
    }

    fn to_csv(&self, sheet: String) -> Result<SheetCsv> {
        let mut writer = csv::Writer::from_writer(Vec::new());
        let mut fields = vec![String::new(); self.width];
        let mut current = None;
        for cell in &self.cells {
            let (row, col) = (cell.row, cell.col);
            if current.is_some_and(|r| r != row) {
                writer.write_record(&fields)?;
                fields.iter_mut().for_each(String::clear);
            }
            current = Some(row);
            let Ok(offset) = usize::try_from(col.saturating_sub(self.first_col)) else {
                continue;
            };
            if let Some(field) = fields.get_mut(offset) {
                *field = cell_text(&cell.value);
            }
        }
        if current.is_some() {
            writer.write_record(&fields)?;
        }
        let csv = writer.into_inner().map_err(|e| Error::Io(e.into_error()))?;
        Ok(SheetCsv {
            sheet,
            csv,
            rows: self.rows,
        })
    }
}

fn cell_text(cell: &Data) -> String {
    match cell {
        Data::Empty => String::new(),
        Data::String(s) | Data::DateTimeIso(s) | Data::DurationIso(s) => s.clone(),
        Data::Int(i) => i.to_string(),
        Data::Float(f) => format_float(*f),
        Data::Bool(b) => b.to_string(),
        Data::DateTime(dt) => {
            if dt.is_duration() {
                format_float(dt.as_f64())
            } else {
                excel_serial_to_iso(dt.as_f64()).unwrap_or_else(|| format_float(dt.as_f64()))
            }
        }
        Data::Error(e) => format!("#{e:?}"),
    }
}

/// Whole numbers without a trailing `.0`, so integer columns sniff as
/// integers.
fn format_float(f: f64) -> String {
    if f.fract() == 0.0 && f.abs() < 1e15 {
        format!("{f:.0}")
    } else {
        f.to_string()
    }
}

/// An Excel serial date (days since 1899-12-30, the 1900 system) as ISO
/// 8601: a date when there is no time part, else a timestamp to the
/// second.
fn excel_serial_to_iso(serial: f64) -> Option<String> {
    if !serial.is_finite() || serial < 0.0 {
        return None;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the whole-day part of a finite, bounded serial fits an i64"
    )]
    let days = serial.floor() as i64;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the fractional day, rounded, is at most 86400 seconds"
    )]
    let seconds = ((serial - serial.floor()) * 86_400.0).round() as i64;
    let epoch = jiff::civil::date(1899, 12, 30);
    let date = epoch.checked_add(jiff::Span::new().days(days)).ok()?;
    if seconds == 0 {
        return Some(date.to_string());
    }
    let datetime = date
        .to_datetime(jiff::civil::time(0, 0, 0, 0))
        .checked_add(jiff::Span::new().seconds(seconds))
        .ok()?;
    Some(datetime.strftime("%Y-%m-%d %H:%M:%S").to_string())
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;

    const BUDGET: DecompressionBudget = DecompressionBudget::megabytes(64);

    /// A one-sheet workbook whose `sheetData` is `rows`, the `<row>` elements
    /// as written, with the sheet's declared dimension `dimension`.
    #[expect(clippy::unwrap_used, reason = "test fixture")]
    fn workbook(dimension: &str, rows: &str) -> Vec<u8> {
        let sheet = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="{dimension}"/><sheetData>{rows}</sheetData></worksheet>"#
        );
        let parts: [(&str, &str); 5] = [
            (
                "[Content_Types].xml",
                r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>"#,
            ),
            (
                "_rels/.rels",
                r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#,
            ),
            (
                "xl/workbook.xml",
                r#"<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sparse" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            ("xl/worksheets/sheet1.xml", &sheet),
        ];
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = zip::ZipWriter::new(&mut cursor);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            for (name, content) in parts {
                writer.start_file(name, options).unwrap();
                writer.write_all(content.as_bytes()).unwrap();
            }
            writer.finish().unwrap();
        }
        cursor.into_inner()
    }

    fn text_cell(reference: &str, text: &str) -> String {
        format!(r#"<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>"#)
    }

    #[test]
    fn cells_render_as_csv_values() {
        assert_eq!(cell_text(&Data::Int(3)), "3");
        assert_eq!(cell_text(&Data::Float(3.0)), "3");
        assert_eq!(cell_text(&Data::Float(2.5)), "2.5");
        assert_eq!(cell_text(&Data::Bool(true)), "true");
        assert_eq!(cell_text(&Data::Empty), "");
        assert_eq!(cell_text(&Data::String(String::from("a,b"))), "a,b");
    }

    #[test]
    fn excel_serials_become_iso_dates_and_timestamps() {
        assert_eq!(excel_serial_to_iso(45_000.0).as_deref(), Some("2023-03-15"));
        assert_eq!(
            excel_serial_to_iso(45_000.5).as_deref(),
            Some("2023-03-15 12:00:00")
        );
        assert_eq!(excel_serial_to_iso(1.0).as_deref(), Some("1899-12-31"));
        assert_eq!(excel_serial_to_iso(-1.0), None);
        assert_eq!(excel_serial_to_iso(f64::NAN), None);
    }

    #[test]
    fn not_a_workbook_is_an_error() {
        assert!(sheets(b"definitely not xlsx", BUDGET).is_err());
    }

    /// The sheet's corners are `A1` and `XFD1048576`: a dense grid would be
    /// 17 billion cells. The CSV is two rows of 16,384 fields.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test")]
    fn far_apart_cells_never_build_the_grid() {
        let rows = format!(
            r#"<row r="1">{}{}</row><row r="1048576">{}{}</row>"#,
            text_cell("A1", "first"),
            text_cell("XFD1", "last"),
            text_cell("A1048576", "x"),
            text_cell("XFD1048576", "y"),
        );
        let mut sheets = sheets(&workbook("A1:XFD1048576", &rows), BUDGET).unwrap();
        assert_eq!(sheets.len(), 1);
        let sheet = sheets.pop().unwrap();
        assert_eq!(sheet.rows, 2);
        let csv = String::from_utf8(sheet.csv).unwrap();
        let mut lines = csv.lines();
        let (first, second) = (lines.next().unwrap(), lines.next().unwrap());
        assert_eq!(lines.next(), None);
        assert_eq!(first.split(',').count(), 16_384);
        assert!(first.starts_with("first,") && first.ends_with(",last"));
        assert!(second.starts_with("x,") && second.ends_with(",y"));
    }

    /// Cells out of row order land in their rows, and rows without cells are
    /// not written.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test")]
    fn cells_are_placed_by_position() {
        let rows = format!(
            r#"<row r="5">{}</row><row r="1">{}{}</row><row r="2">{}</row>"#,
            text_cell("B5", "five"),
            text_cell("B1", "b"),
            text_cell("C1", "c"),
            text_cell("C2", "two"),
        );
        let sheet = sheets(&workbook("B1:C5", &rows), BUDGET)
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(String::from_utf8(sheet.csv).unwrap(), "b,c\n,two\nfive,\n");
        assert_eq!(sheet.rows, 3);
    }

    /// One cell in column `XFD` makes every row 16,384 fields wide, so
    /// enough one-cell rows pass the limit on what a sheet may write.
    #[test]
    fn a_sheet_over_the_cell_limit_is_refused() {
        let rows: String =
            std::iter::once(format!(r#"<row r="1">{}</row>"#, text_cell("XFD1", "wide")))
                .chain((2..=6_105u32).map(|row| {
                    let cell = text_cell(&format!("A{row}"), "a");
                    format!(r#"<row r="{row}">{cell}</row>"#)
                }))
                .collect();
        let err = sheets(&workbook("A1:XFD6105", &rows), BUDGET)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            err.contains("would load 100024320 cells (6105 rows by 16384 columns)"),
            "{err}"
        );
        assert!(err.contains("the limit is 100000000"), "{err}");
    }
}
