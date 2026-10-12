//! Excel and `OpenDocument` workbooks through `calamine` (pure Rust; the
//! `DuckDB` `excel` extension cannot be compiled into the static binary).
//! Each sheet becomes a typed table: the cells keep the types the workbook
//! gives them (whole numbers, decimals, booleans, dates, timestamps), each
//! column takes the one type its cells agree on, and the rows go into the
//! table through `DuckDB`'s appender, with no CSV in between. A table found
//! inside a document takes the same path, its text cells parsed first.

use std::io::Cursor;

use calamine::{Data, Reader, Sheets, XlsxError};
use duckdb::types::{TimeUnit, Value};

use super::budget::DecompressionBudget;
use crate::error::{Error, Result};

/// The most cells one sheet may load: its non-empty rows times the width of
/// its used columns. A dense 1,048,576-row sheet 95 columns wide fits; a
/// sheet whose rows each hold one cell in column `XFD` does not, and nothing
/// wider than this is loaded.
pub const MAX_SHEET_CELLS: u64 = 100_000_000;

/// A non-empty cell at its zero-based position.
struct SheetCell {
    row: u32,
    col: u32,
    value: Data,
}

/// One cell as the table stores it. The kinds are the `DuckDB` types a
/// column can take; a column whose cells disagree becomes text.
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Null,
    Int(i64),
    Float(f64),
    Bool(bool),
    /// Days since 1970-01-01.
    Date(i32),
    /// Microseconds since 1970-01-01 00:00:00.
    Timestamp(i64),
    Text(String),
}

impl Cell {
    /// A workbook cell: typed cells stay typed, a string stays text.
    fn of_data(value: &Data) -> Self {
        match value {
            Data::Empty => Self::Null,
            Data::Int(i) => Self::Int(*i),
            Data::Float(f) => Self::of_float(*f),
            Data::Bool(b) => Self::Bool(*b),
            Data::String(s) | Data::DurationIso(s) => Self::Text(s.clone()),
            Data::DateTimeIso(s) => {
                Self::parse_temporal(s).unwrap_or_else(|| Self::Text(s.clone()))
            }
            Data::DateTime(dt) => {
                if dt.is_duration() {
                    Self::of_float(dt.as_f64())
                } else {
                    excel_serial(dt.as_f64()).unwrap_or_else(|| Self::of_float(dt.as_f64()))
                }
            }
            Data::Error(e) => Self::Text(format!("#{e:?}")),
        }
    }

    /// A cell of text (a table inside a document): a whole number, a
    /// decimal, a boolean, a date, or a timestamp when it reads as one,
    /// else the text; an empty cell is NULL.
    fn parse(text: &str) -> Self {
        let text = text.trim();
        if text.is_empty() {
            return Self::Null;
        }
        if let Ok(i) = text.parse::<i64>() {
            return Self::Int(i);
        }
        if let Ok(f) = text.parse::<f64>()
            && f.is_finite()
        {
            return Self::Float(f);
        }
        match text.to_ascii_lowercase().as_str() {
            "true" => return Self::Bool(true),
            "false" => return Self::Bool(false),
            _ => {}
        }
        Self::parse_temporal(text).unwrap_or_else(|| Self::Text(text.to_owned()))
    }

    /// A finite float that is a whole number within `i64` is an integer,
    /// as a workbook stores every number as a float.
    fn of_float(f: f64) -> Self {
        if f.is_finite() && f.fract() == 0.0 && f.abs() < 9.0e15 {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a whole number below 9e15 fits an i64 exactly"
            )]
            return Self::Int(f as i64);
        }
        Self::Float(f)
    }

    /// `2024-03-01`, `2024-03-01T12:00:00`, or `2024-03-01 12:00:00`. A
    /// date is ten characters; anything longer that parses carries a time.
    /// (jiff reads either form as the other with the time dropped or set to
    /// midnight, so the length decides.)
    fn parse_temporal(text: &str) -> Option<Self> {
        if text.len() <= 10 {
            let date = text.parse::<jiff::civil::Date>().ok()?;
            return Some(Self::Date(days_since_epoch(date)?));
        }
        let datetime = text
            .replacen(' ', "T", 1)
            .parse::<jiff::civil::DateTime>()
            .ok()?;
        Some(Self::Timestamp(micros_since_epoch(datetime)?))
    }

    fn kind(&self) -> Option<ColumnKind> {
        match self {
            Self::Null => None,
            Self::Int(_) => Some(ColumnKind::Bigint),
            Self::Float(_) => Some(ColumnKind::Double),
            Self::Bool(_) => Some(ColumnKind::Boolean),
            Self::Date(_) => Some(ColumnKind::Date),
            Self::Timestamp(_) => Some(ColumnKind::Timestamp),
            Self::Text(_) => Some(ColumnKind::Varchar),
        }
    }

    /// The cell as the appender takes it, under the column's type.
    fn value(&self, kind: ColumnKind) -> Value {
        match (self, kind) {
            (Self::Null, _) => Value::Null,
            (Self::Int(i), ColumnKind::Bigint) => Value::BigInt(*i),
            (Self::Int(i), ColumnKind::Double) =>
            {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "a column of mixed whole and decimal numbers is a double column"
                )]
                Value::Double(*i as f64)
            }
            (Self::Float(f), ColumnKind::Double) => Value::Double(*f),
            (Self::Bool(b), ColumnKind::Boolean) => Value::Boolean(*b),
            (Self::Date(d), ColumnKind::Date) => Value::Date32(*d),
            (Self::Date(d), ColumnKind::Timestamp) => Value::Timestamp(
                TimeUnit::Microsecond,
                i64::from(*d).saturating_mul(86_400_000_000),
            ),
            (Self::Timestamp(t), ColumnKind::Timestamp) => {
                Value::Timestamp(TimeUnit::Microsecond, *t)
            }
            (cell, _) => Value::Text(cell.text()),
        }
    }

    /// The cell as text, for a column whose cells disagree.
    fn text(&self) -> String {
        match self {
            Self::Null => String::new(),
            Self::Int(i) => i.to_string(),
            Self::Float(f) => f.to_string(),
            Self::Bool(b) => b.to_string(),
            Self::Date(d) => jiff::civil::date(1970, 1, 1)
                .checked_add(jiff::Span::new().days(i64::from(*d)))
                .map_or_else(|_| d.to_string(), |date| date.to_string()),
            Self::Timestamp(t) => jiff::Timestamp::from_microsecond(*t).map_or_else(
                |_| t.to_string(),
                |ts| ts.strftime("%Y-%m-%d %H:%M:%S").to_string(),
            ),
            Self::Text(s) => s.clone(),
        }
    }
}

/// The `DuckDB` type a column takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnKind {
    Bigint,
    Double,
    Boolean,
    Date,
    Timestamp,
    Varchar,
}

impl ColumnKind {
    /// The type as `CREATE TABLE` writes it.
    #[must_use]
    pub const fn sql(self) -> &'static str {
        match self {
            Self::Bigint => "BIGINT",
            Self::Double => "DOUBLE",
            Self::Boolean => "BOOLEAN",
            Self::Date => "DATE",
            Self::Timestamp => "TIMESTAMP",
            Self::Varchar => "VARCHAR",
        }
    }

    /// The one type the kinds of a column's cells agree on: whole numbers
    /// among decimals make a double column, dates among timestamps a
    /// timestamp column, anything else mixed is text; all empty is text.
    fn of_cells(cells: &[Cell]) -> Self {
        let mut kind: Option<Self> = None;
        for cell in cells {
            let Some(this) = cell.kind() else {
                continue;
            };
            kind = Some(match kind {
                None => this,
                Some(k) if k == this => k,
                Some(Self::Bigint | Self::Double)
                    if matches!(this, Self::Bigint | Self::Double) =>
                {
                    Self::Double
                }
                Some(Self::Date | Self::Timestamp)
                    if matches!(this, Self::Date | Self::Timestamp) =>
                {
                    Self::Timestamp
                }
                Some(_) => return Self::Varchar,
            });
        }
        kind.unwrap_or(Self::Varchar)
    }
}

/// One column of a sheet: its name as the header gives it, its type, and
/// one cell per data row.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedColumn {
    pub name: String,
    pub kind: ColumnKind,
    pub cells: Vec<Cell>,
}

/// One sheet of a workbook, or one table found inside a document, as a
/// typed table: its name and its columns, each with every data row.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedSheet {
    pub sheet: String,
    pub columns: Vec<TypedColumn>,
    /// Data rows, the header not counted.
    pub rows: usize,
}

impl TypedSheet {
    /// A table from a header and rows of text, each cell parsed as a
    /// number, a boolean, a date, or a timestamp when it reads as one.
    #[must_use]
    pub fn from_text(sheet: String, header: &[String], rows: &[Vec<String>]) -> Self {
        let columns = header
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let cells: Vec<Cell> = rows
                    .iter()
                    .map(|row| row.get(i).map_or(Cell::Null, |text| Cell::parse(text)))
                    .collect();
                TypedColumn {
                    name: name.clone(),
                    kind: ColumnKind::of_cells(&cells),
                    cells,
                }
            })
            .collect();
        let mut sheet = Self {
            sheet,
            columns,
            rows: rows.len(),
        };
        sheet.name_columns();
        sheet
    }

    /// The column definitions as `CREATE TABLE` takes them.
    #[must_use]
    pub fn column_sql(&self, quote: impl Fn(&str) -> String) -> String {
        self.columns
            .iter()
            .map(|c| format!("{} {}", quote(&c.name), c.kind.sql()))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Row `index` as the appender takes it.
    fn row_values(&self, index: usize) -> Vec<Value> {
        self.columns
            .iter()
            .map(|c| {
                c.cells
                    .get(index)
                    .map_or(Value::Null, |cell| cell.value(c.kind))
            })
            .collect()
    }

    /// Every row, for `WorkspaceDb::append_rows`.
    pub fn rows(&self) -> impl Iterator<Item = Vec<Value>> + '_ {
        (0..self.rows).map(|i| self.row_values(i))
    }

    /// Every column named, as the header gives it: a blank header is
    /// `column3` (its position), and a repeated one gains `_2`, `_3`, ...,
    /// since a table cannot hold two columns of one name.
    fn name_columns(&mut self) {
        let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for (i, column) in self.columns.iter_mut().enumerate() {
            let base = if column.name.trim().is_empty() {
                format!("column{}", i.saturating_add(1))
            } else {
                column.name.trim().to_owned()
            };
            let count = seen.entry(base.to_lowercase()).or_insert(0);
            *count = count.saturating_add(1);
            column.name = if *count == 1 {
                base
            } else {
                format!("{base}_{count}")
            };
        }
    }
}

/// Every non-empty sheet as a typed table, in workbook order. The first row
/// with cells is the header.
///
/// `calamine` inflates a zipped workbook (XLSX, XLSM, XLSB, ODS) itself, so
/// its entries are inflated against `budget` first. An XLS file is not
/// compressed and passes untouched.
///
/// A sheet is never read as a dense grid: `calamine`'s `worksheet_range`
/// allocates the bounding box of the used cells, so a sheet holding `A1`
/// and `XFD1048576` would ask for 17 billion cells. XLSX and XLSB stream
/// their cells instead; the sparse cells are the only thing in memory, and
/// the table is bounded by [`MAX_SHEET_CELLS`].
///
/// # Errors
///
/// Returns an error when the bytes are not a workbook, no sheet has data,
/// the workbook inflates past `budget`, or a sheet would load more than
/// [`MAX_SHEET_CELLS`] cells.
pub fn sheets(data: &[u8], mut budget: DecompressionBudget) -> Result<Vec<TypedSheet>> {
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
        out.push(sheet.typed(name));
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
/// occupy: one column per position of the span, and no row without cells.
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

    /// The sheet as a typed table: the first row with cells names the
    /// columns, each later row with cells is a data row.
    fn typed(&self, sheet: String) -> TypedSheet {
        let data_rows = self.rows.saturating_sub(1);
        let mut header = vec![String::new(); self.width];
        let mut columns: Vec<Vec<Cell>> = vec![vec![Cell::Null; data_rows]; self.width];
        let mut current: Option<u32> = None;
        // -1 while on the header row, then the data row's index.
        let mut row_index: Option<usize> = None;
        for cell in &self.cells {
            if current != Some(cell.row) {
                current = Some(cell.row);
                row_index = match row_index {
                    None => None.or(Some(usize::MAX)),
                    Some(usize::MAX) => Some(0),
                    Some(i) => Some(i.saturating_add(1)),
                };
            }
            let Ok(offset) = usize::try_from(cell.col.saturating_sub(self.first_col)) else {
                continue;
            };
            match row_index {
                Some(usize::MAX) => {
                    if let Some(name) = header.get_mut(offset) {
                        *name = Cell::of_data(&cell.value).text();
                    }
                }
                Some(i) => {
                    if let Some(slot) = columns.get_mut(offset).and_then(|c| c.get_mut(i)) {
                        *slot = Cell::of_data(&cell.value);
                    }
                }
                None => {}
            }
        }
        let columns = header
            .into_iter()
            .zip(columns)
            .map(|(name, cells)| TypedColumn {
                name,
                kind: ColumnKind::of_cells(&cells),
                cells,
            })
            .collect();
        let mut typed = TypedSheet {
            sheet,
            columns,
            rows: data_rows,
        };
        typed.name_columns();
        typed
    }
}

/// An Excel serial date (days since 1899-12-30, the 1900 system): a date
/// when there is no time part, else a timestamp to the second.
fn excel_serial(serial: f64) -> Option<Cell> {
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
        return Some(Cell::Date(days_since_epoch(date)?));
    }
    let datetime = date
        .to_datetime(jiff::civil::time(0, 0, 0, 0))
        .checked_add(jiff::Span::new().seconds(seconds))
        .ok()?;
    Some(Cell::Timestamp(micros_since_epoch(datetime)?))
}

fn days_since_epoch(date: jiff::civil::Date) -> Option<i32> {
    let days = jiff::civil::date(1970, 1, 1).until(date).ok()?.get_days();
    Some(days)
}

fn micros_since_epoch(datetime: jiff::civil::DateTime) -> Option<i64> {
    let ts = datetime.to_zoned(jiff::tz::TimeZone::UTC).ok()?.timestamp();
    Some(ts.as_microsecond())
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;

    const BUDGET: DecompressionBudget = DecompressionBudget::megabytes(64);

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

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
    fn cells_keep_their_types_and_text_parses() {
        assert_eq!(Cell::of_data(&Data::Int(3)), Cell::Int(3));
        assert_eq!(Cell::of_data(&Data::Float(3.0)), Cell::Int(3));
        assert_eq!(Cell::of_data(&Data::Float(2.5)), Cell::Float(2.5));
        assert_eq!(Cell::of_data(&Data::Bool(true)), Cell::Bool(true));
        assert_eq!(Cell::of_data(&Data::Empty), Cell::Null);
        assert_eq!(
            Cell::of_data(&Data::String(String::from("02134"))),
            Cell::Text(String::from("02134"))
        );
        assert_eq!(Cell::parse("42"), Cell::Int(42));
        assert_eq!(Cell::parse("4.5"), Cell::Float(4.5));
        assert_eq!(Cell::parse("TRUE"), Cell::Bool(true));
        assert_eq!(Cell::parse("2024-03-01"), Cell::Date(19_783));
        assert_eq!(
            Cell::parse("2024-03-01 12:00:00"),
            Cell::Timestamp(1_709_294_400_000_000)
        );
        assert_eq!(Cell::parse(" "), Cell::Null);
        assert_eq!(Cell::parse("north"), Cell::Text(String::from("north")));
    }

    #[test]
    fn excel_serials_become_dates_and_timestamps() {
        assert_eq!(excel_serial(45_000.0), Some(Cell::Date(19_431)));
        assert_eq!(
            excel_serial(45_000.5),
            Some(Cell::Timestamp(1_678_881_600_000_000))
        );
        assert_eq!(excel_serial(1.0), Some(Cell::Date(-25_568)));
        assert_eq!(excel_serial(-1.0), None);
        assert_eq!(excel_serial(f64::NAN), None);
        assert_eq!(Cell::Date(19_431).text(), "2023-03-15");
        assert_eq!(
            Cell::Timestamp(1_678_881_600_000_000).text(),
            "2023-03-15 12:00:00"
        );
    }

    /// A column takes the one type its cells agree on.
    #[test]
    fn a_column_takes_the_type_its_cells_agree_on() {
        let kind = |cells: &[Cell]| ColumnKind::of_cells(cells);
        assert_eq!(
            kind(&[Cell::Int(1), Cell::Null, Cell::Int(2)]),
            ColumnKind::Bigint
        );
        assert_eq!(kind(&[Cell::Int(1), Cell::Float(2.5)]), ColumnKind::Double);
        assert_eq!(kind(&[Cell::Bool(true)]), ColumnKind::Boolean);
        assert_eq!(kind(&[Cell::Date(1), Cell::Date(2)]), ColumnKind::Date);
        assert_eq!(
            kind(&[Cell::Date(1), Cell::Timestamp(2)]),
            ColumnKind::Timestamp
        );
        assert_eq!(
            kind(&[Cell::Int(1), Cell::Text(String::from("x"))]),
            ColumnKind::Varchar
        );
        assert_eq!(kind(&[Cell::Null, Cell::Null]), ColumnKind::Varchar);
        // Under the agreed type a cell converts; under text it renders.
        assert_eq!(Cell::Int(1).value(ColumnKind::Double), Value::Double(1.0));
        assert_eq!(
            Cell::Date(1).value(ColumnKind::Timestamp),
            Value::Timestamp(TimeUnit::Microsecond, 86_400_000_000)
        );
        assert_eq!(
            Cell::Int(7).value(ColumnKind::Varchar),
            Value::Text(String::from("7"))
        );
    }

    /// Blank and repeated headers get names a table can hold.
    #[test]
    fn headers_are_named_and_made_distinct() {
        let sheet = TypedSheet::from_text(
            String::from("t"),
            &[
                String::from("a"),
                String::new(),
                String::from("A"),
                String::from("a"),
            ],
            &[vec![
                String::from("1"),
                String::from("2"),
                String::from("3"),
                String::from("4"),
            ]],
        );
        let names: Vec<&str> = sheet.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["a", "column2", "A_2", "a_3"]);
        assert!(sheet.columns.iter().all(|c| c.kind == ColumnKind::Bigint));
        assert_eq!(sheet.rows, 1);
    }

    #[test]
    fn not_a_workbook_is_an_error() {
        assert!(sheets(b"definitely not xlsx", BUDGET).is_err());
    }

    /// The sheet's corners are `A1` and `XFD1048576`: a dense grid would be
    /// 17 billion cells. The table is 16,384 columns of one data row.
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
        assert_eq!(sheet.rows, 1);
        assert_eq!(sheet.columns.len(), 16_384);
        assert_eq!(sheet.columns.first().unwrap().name, "first");
        assert_eq!(sheet.columns.last().unwrap().name, "last");
        assert_eq!(
            sheet.columns.first().unwrap().cells,
            [Cell::Text(String::from("x"))]
        );
        let second = sheet
            .columns
            .get(1)
            .unwrap_or_else(|| fail("no second column"));
        assert_eq!(second.name, "column2");
        assert_eq!(second.cells, [Cell::Null]);
    }

    /// Cells out of row order land in their rows, and rows without cells are
    /// not counted.
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
        assert_eq!(sheet.rows, 2);
        let b = sheet.columns.first().unwrap();
        let c = sheet
            .columns
            .get(1)
            .unwrap_or_else(|| fail("no second column"));
        assert_eq!((b.name.as_str(), c.name.as_str()), ("b", "c"));
        assert_eq!(b.cells, [Cell::Null, Cell::Text(String::from("five"))]);
        assert_eq!(c.cells, [Cell::Text(String::from("two")), Cell::Null]);
    }

    /// One cell in column `XFD` makes every row 16,384 fields wide, so
    /// enough one-cell rows pass the limit on what a sheet may load.
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
