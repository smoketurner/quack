//! Excel workbooks through `calamine` (pure Rust; the `DuckDB` `excel`
//! extension cannot be compiled into the static binary). Each sheet is
//! written as CSV under `files/` and loaded as its own table.

use std::io::{Cursor, Write};

use calamine::{Data, Reader};

use crate::csv::CsvField;
use crate::error::{Error, Result};

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
/// # Errors
///
/// Returns an error when the bytes are not a workbook or no sheet has data.
pub fn sheets(data: &[u8]) -> Result<Vec<SheetCsv>> {
    let mut workbook = calamine::open_workbook_auto_from_rs(Cursor::new(data))
        .map_err(|e| Error::Ingestion(format!("not a spreadsheet: {e}")))?;
    let names = workbook.sheet_names();
    let mut out = Vec::new();
    for name in names {
        let range = match workbook.worksheet_range(&name) {
            Ok(range) => range,
            Err(e) => {
                tracing::warn!(sheet = %name, error = %e, "skipping unreadable sheet");
                continue;
            }
        };
        let mut csv = Vec::new();
        let mut rows = 0usize;
        for row in range.rows() {
            if row.iter().all(|c| matches!(c, Data::Empty)) {
                continue;
            }
            let fields: Vec<String> = row.iter().map(cell_text).collect();
            let line = fields
                .iter()
                .map(|f| CsvField(f).to_string())
                .collect::<Vec<_>>()
                .join(",");
            csv.write_all(line.as_bytes())?;
            csv.write_all(b"\n")?;
            rows = rows.saturating_add(1);
        }
        if rows < 2 {
            tracing::info!(sheet = %name, "skipping sheet without data rows");
            continue;
        }
        out.push(SheetCsv {
            sheet: name,
            csv,
            rows,
        });
    }
    if out.is_empty() {
        return Err(Error::Ingestion(String::from(
            "no data: every sheet is empty or has only a header row",
        )));
    }
    Ok(out)
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
    use super::*;

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
        assert!(sheets(b"definitely not xlsx").is_err());
    }
}
