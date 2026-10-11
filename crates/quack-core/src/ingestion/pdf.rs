//! PDF through `pdf_oxide`: each page as its typed regions, so running
//! headers, footers, page numbers, and other artifacts are left out,
//! structural headings start sections, and the tables the layout detector
//! finds become table sections of their own. The Info dictionary gives
//! the title, author, and dates.

use std::collections::BTreeMap;

use pdf_oxide::PdfDocument;
use pdf_oxide::editor::DocumentInfo;
use pdf_oxide::extractors::images::{ColorSpace, ImageData};
use pdf_oxide::layout::{TextSpan, Word};
use pdf_oxide::structure::table_extractor::Table as PdfTable;
use pdf_oxide::structured::{RegionRole, StructuredPage, StructuredRegion};

use super::parser::{DocumentMeta, Extracted, Flow, ImageFormat, PageCounts, Scan, Section};
use super::table::Table;
use crate::error::{Error, Result};

/// A PDF: the body of each page under the page's headings, its tables
/// apart, and the pages that could not be read counted.
///
/// # Errors
///
/// Returns an error when the bytes are not a PDF, it is password-protected,
/// or no page yields text.
pub fn extract(data: &[u8]) -> Result<Extracted> {
    let pdf = Pdf(PdfDocument::from_bytes(data.to_vec())
        .map_err(|e| Error::Ingestion(format!("PDF extraction failed: {e}")))?);
    let doc = &pdf.0;
    if !doc.is_authenticated() {
        return Err(Error::Ingestion(String::from(
            "the PDF is password-protected; remove the password and upload it again",
        )));
    }
    let page_count = doc
        .page_count()
        .map_err(|e| Error::Ingestion(format!("PDF extraction failed: {e}")))?;
    let (meta, title) = pdf.meta();
    let pages = Pages::read_scanning(
        page_count,
        |index| {
            let structured = doc.extract_structured(index).map_err(|e| e.to_string())?;
            // Table detection is best effort: a page whose tables cannot be
            // read still gives its text.
            let tables = doc.extract_tables(index).unwrap_or_default();
            // Words place the cells of a table drawn without rules.
            let words = doc.extract_words(index).unwrap_or_default();
            Ok(PageContent::of(&structured, &tables, &words))
        },
        |index| pdf.picture(index),
    )?;
    Ok(Extracted {
        title,
        sections: pages.sections,
        flow: Flow::Continuous,
        pages: Some(pages.counts),
        meta,
        scans: pages.scans,
    })
}

/// The smallest picture taken for a page's scan, in pixels on each side:
/// below it an image is a logo or a rule, not a page.
const SCAN_MIN_SIDE: u32 = 200;

/// What one page holds once its chrome is dropped, in reading order:
/// runs of body text, each starting at a heading or continuing the one
/// before, and the tables where they sit.
#[derive(Debug, Default)]
pub(crate) struct PageContent {
    pieces: Vec<Piece>,
}

#[derive(Debug)]
pub(crate) enum Piece {
    /// Body text starting at `started` (a heading), or under the heading
    /// in force when `None`.
    Run {
        started: Option<String>,
        text: String,
    },
    Table(Table),
}

/// A heading's font size over the body's, when the file carries no
/// structure tree to say which lines are headings.
const HEADING_SCALE: f32 = 1.15;
/// Characters a heading or a header line is at most.
const HEADING_CHARS: usize = 120;
/// The band at the top and the bottom of a page where a short line is a
/// running header, footer, or page number: a share of the page height.
const CHROME_BAND: f32 = 0.07;

/// One line of a page, assembled from its spans.
struct Line {
    /// Distance from the page's bottom edge to the line's baseline.
    y: f32,
    top: f32,
    font_size: f32,
    text: String,
}

impl Line {
    /// Spans grouped into lines: a span joins the line whose baseline is
    /// within half its font size; a line's text is its spans left to right.
    fn assemble(spans: &[&TextSpan]) -> Vec<Self> {
        let mut lines: Vec<(Vec<&TextSpan>, f32)> = Vec::new();
        for span in spans {
            let y = span.bbox.y;
            let tolerance = (span.font_size * 0.5).max(1.0);
            match lines
                .iter_mut()
                .find(|(_, line_y)| (*line_y - y).abs() <= tolerance)
            {
                Some((members, _)) => members.push(span),
                None => lines.push((vec![span], y)),
            }
        }
        lines
            .into_iter()
            .map(|(mut members, y)| {
                members.sort_by(|a, b| a.bbox.x.total_cmp(&b.bbox.x));
                let text = members
                    .iter()
                    .map(|s| s.text.trim())
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ")
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
                let font_size = members.iter().map(|s| s.font_size).fold(0.0_f32, f32::max);
                let top = members
                    .iter()
                    .map(|s| s.bbox.y + s.bbox.height)
                    .fold(0.0_f32, f32::max);
                Self {
                    y,
                    top,
                    font_size,
                    text,
                }
            })
            .collect()
    }
}

/// The gap between two words of a row, in em, past which they are two
/// cells rather than two words of one: a word space is about a quarter em.
const CELL_GAP_EM: f32 = 0.8;
/// Rows a run of aligned rows needs to be read as a table, header included.
const TABLE_MIN_ROWS: usize = 3;
/// Characters a cell of a table without rules holds at most: a longer run
/// of text is prose that happens to line up.
const CELL_MAX_CHARS: usize = 40;
/// How far apart, in points, two cells of a column may sit and still be
/// aligned on an edge or their centre.
const COLUMN_TOLERANCE: f32 = 3.0;

/// A run of a row's words that stands apart from the rest.
#[derive(Debug, Clone, PartialEq)]
struct Cell {
    left: f32,
    right: f32,
    text: String,
}

impl Cell {
    /// Whether `other` sits in the same column: aligned on the left edge,
    /// the right edge (numbers), or the centre.
    fn aligned(&self, other: &Self) -> bool {
        (self.left - other.left).abs() <= COLUMN_TOLERANCE
            || (self.right - other.right).abs() <= COLUMN_TOLERANCE
            || ((self.left + self.right) - (other.left + other.right)).abs() / 2.0
                <= COLUMN_TOLERANCE
    }

    fn short(&self) -> bool {
        self.text.chars().count() <= CELL_MAX_CHARS
    }
}

/// A row of a page's words on one baseline, split into cells where they
/// stand apart, for telling a table without rules from prose.
#[derive(Debug, Clone, PartialEq)]
struct Row {
    /// Distance from the page's bottom edge to the row's baseline.
    y: f32,
    top: f32,
    cells: Vec<Cell>,
}

impl Row {
    /// `words` as rows, top to bottom: a word joins the row whose baseline
    /// is within half its font size, and a gap of more than
    /// [`CELL_GAP_EM`] starts a new cell.
    fn of_words(words: &[&Word]) -> Vec<Self> {
        let mut grouped: Vec<(f32, Vec<&Word>)> = Vec::new();
        for word in words {
            let tolerance = (word.avg_font_size * 0.5).max(1.0);
            match grouped
                .iter_mut()
                .find(|(y, _)| (*y - word.bbox.y).abs() <= tolerance)
            {
                Some((_, members)) => members.push(word),
                None => grouped.push((word.bbox.y, vec![word])),
            }
        }
        let mut rows: Vec<Self> = grouped
            .into_iter()
            .map(|(y, mut members)| {
                members.sort_by(|a, b| a.bbox.x.total_cmp(&b.bbox.x));
                let mut cells: Vec<Cell> = Vec::new();
                for word in &members {
                    let text = word.text.trim();
                    if text.is_empty() {
                        continue;
                    }
                    let (left, right) = (word.bbox.x, word.bbox.x + word.bbox.width);
                    match cells.last_mut() {
                        Some(cell) if left - cell.right <= word.avg_font_size * CELL_GAP_EM => {
                            cell.text.push(' ');
                            cell.text.push_str(text);
                            cell.right = cell.right.max(right);
                        }
                        _ => cells.push(Cell {
                            left,
                            right,
                            text: text.to_owned(),
                        }),
                    }
                }
                let top = members
                    .iter()
                    .map(|w| w.bbox.y + w.bbox.height)
                    .fold(y, f32::max);
                Self { y, top, cells }
            })
            .collect();
        rows.sort_by(|a, b| b.y.total_cmp(&a.y));
        rows
    }

    /// Whether `next` is another row of the table this row starts: as
    /// many cells, each short and in this row's column.
    fn shares_columns(&self, next: &Self) -> bool {
        next.cells.len() == self.cells.len()
            && next.cells.iter().all(Cell::short)
            && self
                .cells
                .iter()
                .zip(&next.cells)
                .all(|(a, b)| a.aligned(b))
    }
}

/// The runs of `rows` that are tables without rules: at least
/// [`TABLE_MIN_ROWS`] rows in a row, each split into the same number (two
/// or more) of short cells, every column aligned. A line of prose is one
/// cell, its words a space apart.
fn borderless_tables(rows: &[Row]) -> Vec<std::ops::Range<usize>> {
    let mut found = Vec::new();
    let mut start = 0;
    while let Some(first) = rows.get(start) {
        let mut end = start.saturating_add(1);
        if first.cells.len() >= 2 && first.cells.iter().all(Cell::short) {
            while rows.get(end).is_some_and(|row| first.shares_columns(row)) {
                end = end.saturating_add(1);
            }
        }
        if end.saturating_sub(start) >= TABLE_MIN_ROWS {
            found.push(start..end);
            start = end;
        } else {
            start = start.saturating_add(1);
        }
    }
    found
}

/// A table drawn without rules: its rows' cells, and the area its words
/// cover, which the page's text leaves to it.
#[derive(Debug)]
struct Borderless {
    rows: Vec<Vec<String>>,
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
}

impl Borderless {
    /// The tables without rules among `rows`.
    fn find(rows: &[Row]) -> Vec<Self> {
        borderless_tables(rows)
            .into_iter()
            .filter_map(|run| {
                let rows = rows.get(run)?;
                let cells = || rows.iter().flat_map(|r| &r.cells);
                Some(Self {
                    rows: rows
                        .iter()
                        .map(|r| r.cells.iter().map(|c| c.text.clone()).collect())
                        .collect(),
                    left: cells().map(|c| c.left).fold(f32::MAX, f32::min) - 1.0,
                    right: cells().map(|c| c.right).fold(f32::MIN, f32::max) + 1.0,
                    bottom: rows.iter().map(|r| r.y).fold(f32::MAX, f32::min) - 2.0,
                    top: rows.iter().map(|r| r.top).fold(f32::MIN, f32::max) + 1.0,
                })
            })
            .collect()
    }

    /// Whether the point (`x`, `y`) lies in the table's area.
    fn holds(&self, x: f32, y: f32) -> bool {
        (self.left..=self.right).contains(&x) && (self.bottom..=self.top).contains(&y)
    }
}

/// The real tables on a page, for the spans inside them: the ruled grids
/// `pdf_oxide` finds, and the tables drawn without rules found here.
struct Grids<'a> {
    ruled: Vec<&'a PdfTable>,
    borderless: Vec<Borderless>,
}

impl Grids<'_> {
    fn contains(&self, span: &TextSpan) -> bool {
        let center = span.bbox.center();
        self.ruled.iter().any(|table| {
            table
                .bbox
                .as_ref()
                .is_some_and(|b| b.contains_point(&center))
        }) || self.borderless.iter().any(|t| t.holds(center.x, center.y))
    }
}

/// A page with its real tables, read for content: chrome regions dropped,
/// spans inside the tables left to them.
struct Page<'a> {
    page: &'a StructuredPage,
    grids: Grids<'a>,
}

impl<'a> Page<'a> {
    /// The regions that are content rather than the page's chrome.
    fn regions(&self) -> impl Iterator<Item = &'a StructuredRegion> {
        self.page.regions.iter().filter(|region| {
            !matches!(
                region.kind,
                RegionRole::Header
                    | RegionRole::Footer
                    | RegionRole::PageNumber
                    | RegionRole::Artifact
            )
        })
    }

    /// The body's font size: the one most characters outside tables are
    /// set in; 10 points on a page with none.
    fn body_font_size(&self) -> f32 {
        let mut chars_by_size: BTreeMap<u32, usize> = BTreeMap::new();
        for span in self.regions().flat_map(|r| &r.spans) {
            if self.grids.contains(span) {
                continue;
            }
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "font sizes are small positive points"
            )]
            let key = (span.font_size * 10.0).round().max(0.0) as u32;
            let chars = chars_by_size.entry(key).or_default();
            *chars = chars.saturating_add(span.text.chars().count());
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "a font size in tenths of a point"
        )]
        chars_by_size
            .iter()
            .max_by_key(|(_, chars)| **chars)
            .map_or(10.0, |(size, _)| *size as f32 / 10.0)
    }

    /// The page's lines in reading order, and the headings its structure
    /// tree names with their height on the page. Region order is trusted
    /// on a page whose columns are real; a column the detector saw in a
    /// table's cells is not one, and the page then reads top to bottom.
    fn lines(&self) -> (Vec<Line>, Vec<(f32, String)>) {
        let columnar = self.regions().any(|r| {
            r.column_index.is_some_and(|c| c >= 1)
                && !r.spans.iter().any(|s| self.grids.contains(s))
        });
        let mut lines: Vec<Line> = Vec::new();
        let mut structural: Vec<(f32, String)> = Vec::new();
        for region in self.regions() {
            if let RegionRole::StructuralHeading { .. } = region.kind {
                let heading = region.text.split_whitespace().collect::<Vec<_>>().join(" ");
                if !heading.is_empty() {
                    structural.push((region.bbox.y, heading));
                }
                continue;
            }
            let spans: Vec<&TextSpan> = region
                .spans
                .iter()
                .filter(|s| !self.grids.contains(s))
                .collect();
            let mut region_lines = Line::assemble(&spans);
            if !columnar {
                region_lines.sort_by(|a, b| b.y.total_cmp(&a.y));
            }
            lines.extend(region_lines);
        }
        if !columnar {
            lines.sort_by(|a, b| b.y.total_cmp(&a.y));
        }
        structural.sort_by(|a, b| b.0.total_cmp(&a.0));
        (lines, structural)
    }
}

impl PageContent {
    fn of(page: &StructuredPage, tables: &[PdfTable], words: &[Word]) -> Self {
        let ruled: Vec<&PdfTable> = tables.iter().filter(|t| t.is_real_grid()).collect();
        // Words outside the ruled grids, which already have their tables.
        let loose: Vec<&Word> = words
            .iter()
            .filter(|w| {
                let center = w.bbox.center();
                !ruled
                    .iter()
                    .any(|t| t.bbox.as_ref().is_some_and(|b| b.contains_point(&center)))
            })
            .collect();
        let page = Page {
            page,
            grids: Grids {
                ruled,
                borderless: Borderless::find(&Row::of_words(&loose)),
            },
        };
        let body_size = page.body_font_size();
        let (lines, mut pending) = page.lines();
        let mut out = Self::default();
        let mut current: (Option<String>, Vec<String>) = (None, Vec::new());
        // Tables sit where their top edge is, highest first.
        let mut placed: Vec<(f32, Table)> = page
            .grids
            .ruled
            .iter()
            .filter_map(|table| {
                let rows = table
                    .rows
                    .iter()
                    .map(|row| row.cells.iter().map(|c| c.text.clone()).collect())
                    .collect();
                let top = table.bbox.as_ref().map_or(0.0, |b| b.y + b.height);
                Table::from_rows(rows).map(|t| (top, t))
            })
            .collect();
        placed.extend(
            page.grids
                .borderless
                .iter()
                .filter_map(|t| Table::from_rows(t.rows.clone()).map(|table| (t.top, table))),
        );
        placed.sort_by(|a, b| b.0.total_cmp(&a.0));
        for line in lines {
            while placed.first().is_some_and(|(top, _)| *top >= line.top) {
                let (_, table) = placed.remove(0);
                out.flush(&mut current);
                out.pieces.push(Piece::Table(table));
            }
            // Structural headings above this line come first.
            while pending.first().is_some_and(|(y, _)| *y >= line.y) {
                let (_, heading) = pending.remove(0);
                out.start(&mut current, heading);
            }
            let chars = line.text.chars().count();
            let chrome = chars < HEADING_CHARS
                && (line.top < page.page.page_height * CHROME_BAND
                    || line.y > page.page.page_height * (1.0 - CHROME_BAND));
            if chrome || line.text.is_empty() {
                continue;
            }
            if chars < HEADING_CHARS && line.font_size >= body_size * HEADING_SCALE {
                out.start(&mut current, line.text);
            } else {
                current.1.push(line.text);
            }
        }
        for (_, heading) in pending {
            out.start(&mut current, heading);
        }
        out.flush(&mut current);
        for (_, table) in placed {
            out.pieces.push(Piece::Table(table));
        }
        out
    }

    /// End the run being read, when it holds anything.
    fn flush(&mut self, current: &mut (Option<String>, Vec<String>)) {
        let (started, body) = std::mem::take(current);
        if !body.is_empty() || started.is_some() {
            self.pieces.push(Piece::Run {
                started,
                text: body.join("\n"),
            });
        }
    }

    /// End the run being read and start one at `heading`.
    fn start(&mut self, current: &mut (Option<String>, Vec<String>), heading: String) {
        self.flush(current);
        *current = (Some(heading), Vec::new());
    }

    fn has_text(&self) -> bool {
        self.pieces.iter().any(|piece| match piece {
            Piece::Run { text, .. } => !text.trim().is_empty(),
            Piece::Table(_) => true,
        })
    }
}

/// A PDF's pages as sections, how the pages read, and the pictures of the
/// pages without text.
struct Pages {
    sections: Vec<Section>,
    counts: PageCounts,
    scans: Vec<Scan>,
}

impl Pages {
    /// Read `page_count` pages with `read`, carrying each heading onto the
    /// pages after it, counting the pages that fail and the pages that
    /// hold nothing.
    #[cfg(test)]
    fn read(
        page_count: usize,
        read: impl Fn(usize) -> std::result::Result<PageContent, String>,
    ) -> Result<Self> {
        Self::read_scanning(page_count, read, |_| None)
    }

    /// [`Pages::read`], taking from `picture` the image of each page that
    /// reads without text, for a vision model. A PDF of such pages alone
    /// is not refused here: whether they can be read is the vision model's
    /// to say.
    fn read_scanning(
        page_count: usize,
        read: impl Fn(usize) -> std::result::Result<PageContent, String>,
        picture: impl Fn(usize) -> Option<(Vec<u8>, ImageFormat)>,
    ) -> Result<Self> {
        let mut scans = Vec::new();
        let mut sections = Vec::new();
        let mut counts = PageCounts {
            total: u32::try_from(page_count).unwrap_or(u32::MAX),
            unreadable: 0,
            empty: 0,
            transcribed: 0,
        };
        let mut heading: Option<String> = None;
        for index in 0..page_count {
            let page = u32::try_from(index.saturating_add(1)).unwrap_or(u32::MAX);
            match read(index) {
                Ok(content) if content.has_text() => {
                    for piece in content.pieces {
                        match piece {
                            Piece::Run { started, text } => {
                                if let Some(h) = started {
                                    heading = Some(h);
                                }
                                if !text.trim().is_empty() {
                                    sections.push(
                                        Section::body(heading.clone(), text).on_page(Some(page)),
                                    );
                                }
                            }
                            Piece::Table(table) => sections.push(
                                Section::table(heading.clone(), table.render()).on_page(Some(page)),
                            ),
                        }
                    }
                }
                Ok(_) => {
                    counts.empty = counts.empty.saturating_add(1);
                    if let Some((image, format)) = picture(index) {
                        scans.push(Scan {
                            page,
                            image,
                            format,
                        });
                    }
                }
                Err(error) => {
                    tracing::warn!(page, error = %error, "skipping an unreadable PDF page");
                    counts.unreadable = counts.unreadable.saturating_add(1);
                }
            }
        }
        if sections.is_empty() && scans.is_empty() {
            if counts.unreadable > 0 {
                return Err(Error::Ingestion(format!(
                    "no readable text: {} of {page_count} pages failed to parse",
                    counts.unreadable
                )));
            }
            return Err(Error::Ingestion(String::from(
                "no extractable text: the PDF has no text layer and no page pictures to read",
            )));
        }
        Ok(Self {
            sections,
            counts,
            scans,
        })
    }
}

/// A parsed PDF.
struct Pdf(PdfDocument);

impl Pdf {
    /// The largest picture on page `index`, as a vision model takes it: an
    /// RGB or gray JPEG as stored, anything else as PNG; `None` when the page holds no
    /// picture the size of a page's text.
    fn picture(&self, index: usize) -> Option<(Vec<u8>, ImageFormat)> {
        let images = self
            .0
            .extract_images(index)
            .inspect_err(
                |e| tracing::warn!(page = index, error = %e, "could not read a PDF page's images"),
            )
            .ok()?;
        let image = images
            .into_iter()
            .filter(|i| i.width() >= SCAN_MIN_SIDE && i.height() >= SCAN_MIN_SIDE)
            .max_by_key(|i| u64::from(i.width()).saturating_mul(u64::from(i.height())))?;
        let plain = matches!(
            image.color_space(),
            ColorSpace::DeviceRGB | ColorSpace::DeviceGray
        );
        match image.data() {
            ImageData::Jpeg(bytes) if plain => Some((bytes.clone(), ImageFormat::Jpeg)),
            // A CMYK or palette JPEG, or raw pixels: converted.
            ImageData::Jpeg(_) | ImageData::Raw { .. } => match image.to_png_bytes() {
                Ok(png) => Some((png, ImageFormat::Png)),
                Err(e) => {
                    tracing::warn!(page = index, error = %e, "could not encode a PDF page's picture");
                    None
                }
            },
        }
    }

    /// The Info dictionary: the metadata, and the title apart.
    fn meta(&self) -> (DocumentMeta, Option<String>) {
        let doc = &self.0;
        let mut meta = DocumentMeta::default();
        let Some(info) = doc
            .trailer()
            .as_dict()
            .and_then(|t| t.get("Info"))
            .and_then(pdf_oxide::object::Object::as_reference)
            .and_then(|r| doc.load_object(r).ok())
            .map(|object| DocumentInfo::from_object(&object))
        else {
            return (meta, None);
        };
        DocumentMeta::set(&mut meta.author, info.author.as_deref());
        DocumentMeta::set(
            &mut meta.authored_at,
            info.creation_date
                .as_deref()
                .map(|raw| PdfDate(raw).iso())
                .as_deref(),
        );
        DocumentMeta::set(
            &mut meta.modified_at,
            info.mod_date
                .as_deref()
                .map(|raw| PdfDate(raw).iso())
                .as_deref(),
        );
        meta.extra("subject", info.subject.as_deref());
        for keyword in info
            .keywords
            .as_deref()
            .unwrap_or_default()
            .split([',', ';'])
        {
            meta.tag(keyword);
        }
        let title = info
            .title
            .map(|t| t.trim().to_owned())
            .filter(|t| !t.is_empty());
        (meta, title)
    }
}

/// A date string from a PDF's Info dictionary.
struct PdfDate<'a>(&'a str);

impl PdfDate<'_> {
    /// A PDF date (`D:YYYYMMDDHHmmSSOHH'mm'`, ISO 32000-1 section 7.9.4) as
    /// ISO 8601 text: the parts given, the zone kept when it is one.
    fn iso(&self) -> String {
        let raw = self.0;
        let digits: String = raw
            .trim()
            .trim_start_matches("D:")
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        let part = |from: usize, to: usize| digits.get(from..to);
        let (Some(year), Some(month), Some(day)) = (part(0, 4), part(4, 6), part(6, 8)) else {
            return raw.trim().to_owned();
        };
        let mut out = format!("{year}-{month}-{day}");
        if let Some(hour) = part(8, 10) {
            let minute = part(10, 12).unwrap_or("00");
            let second = part(12, 14).unwrap_or("00");
            out = format!("{out}T{hour}:{minute}:{second}");
            let rest = raw
                .trim()
                .trim_start_matches("D:")
                .get(digits.len()..)
                .unwrap_or_default();
            match rest.chars().next() {
                Some('Z') => out.push('Z'),
                Some(sign @ ('+' | '-')) => {
                    let offset: String =
                        rest.chars().skip(1).filter(char::is_ascii_digit).collect();
                    if let (Some(h), Some(m)) = (offset.get(0..2), offset.get(2..4)) {
                        out = format!("{out}{sign}{h}:{m}");
                    }
                }
                _ => {}
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdf_dates_become_iso_8601() {
        assert_eq!(
            PdfDate("D:20260105143000+01'00'").iso(),
            "2026-01-05T14:30:00+01:00"
        );
        assert_eq!(PdfDate("D:20260105143000Z").iso(), "2026-01-05T14:30:00Z");
        assert_eq!(PdfDate("D:20260105").iso(), "2026-01-05");
        assert_eq!(PdfDate("D:2026010514").iso(), "2026-01-05T14:00:00");
        assert_eq!(PdfDate("not a date").iso(), "not a date");
    }

    #[test]
    fn a_page_that_fails_to_read_is_skipped_and_counted_not_the_rest() {
        let content = |text: &str| PageContent {
            pieces: vec![Piece::Run {
                started: None,
                text: String::from(text),
            }],
        };
        let pages = Pages::read(4, |index| match index {
            1 => Err(String::from("bad font")),
            2 => Ok(content("   ")),
            _ => Ok(content(&format!("text {index}"))),
        })
        .unwrap_or_else(|_| Pages {
            sections: Vec::new(),
            scans: Vec::new(),
            counts: PageCounts {
                total: 0,
                unreadable: 0,
                empty: 0,
                transcribed: 0,
            },
        });
        assert_eq!(
            pages.counts,
            PageCounts {
                total: 4,
                unreadable: 1,
                empty: 1,
                transcribed: 0,
            }
        );
        let numbered: Vec<Option<u32>> = pages.sections.iter().map(|s| s.page).collect();
        assert_eq!(numbered, vec![Some(1), Some(4)]);
        assert_eq!(
            pages.sections.get(1).map(|s| s.text.as_str()),
            Some("text 3")
        );
    }

    #[test]
    fn headings_carry_onto_later_pages_and_tables_get_their_own_sections() {
        let pages = Pages::read(2, |index| {
            Ok(match index {
                0 => PageContent {
                    pieces: vec![
                        Piece::Run {
                            started: None,
                            text: String::from("preface"),
                        },
                        Piece::Run {
                            started: Some(String::from("Exclusions")),
                            text: String::from("flood"),
                        },
                        Piece::Table(
                            Table::from_rows(vec![
                                vec![String::from("Peril"), String::from("Covered")],
                                vec![String::from("Fire"), String::from("yes")],
                            ])
                            .unwrap_or(Table {
                                header: Vec::new(),
                                rows: Vec::new(),
                            }),
                        ),
                    ],
                },
                _ => PageContent {
                    pieces: vec![Piece::Run {
                        started: None,
                        text: String::from("still excluded"),
                    }],
                },
            })
        })
        .unwrap_or_else(|_| Pages {
            sections: Vec::new(),
            scans: Vec::new(),
            counts: PageCounts {
                total: 0,
                unreadable: 0,
                empty: 0,
                transcribed: 0,
            },
        });
        let summary: Vec<(Option<&str>, &str, Option<u32>)> = pages
            .sections
            .iter()
            .map(|s| (s.heading.as_deref(), s.kind.as_str(), s.page))
            .collect();
        assert_eq!(
            summary,
            vec![
                (None, "body", Some(1)),
                (Some("Exclusions"), "body", Some(1)),
                (Some("Exclusions"), "table", Some(1)),
                (Some("Exclusions"), "body", Some(2)),
            ]
        );
        assert!(
            pages
                .sections
                .get(2)
                .is_some_and(|s| s.text.starts_with("| Peril | Covered |"))
        );
    }

    /// A page with a picture and no text is kept for the vision model,
    /// counted without text until it is read; a PDF of only such pages is
    /// not refused here.
    #[test]
    fn a_page_without_text_hands_its_picture_on() {
        let pages = Pages::read_scanning(
            3,
            |index| {
                Ok(match index {
                    1 => PageContent {
                        pieces: vec![Piece::Run {
                            started: None,
                            text: String::from("typed page"),
                        }],
                    },
                    _ => PageContent::default(),
                })
            },
            |index| (index == 2).then(|| (vec![0xFF, 0xD8], ImageFormat::Jpeg)),
        );
        assert!(pages.is_ok(), "{:?}", pages.as_ref().err());
        let Ok(pages) = pages else { return };
        assert_eq!(pages.counts.empty, 2);
        assert_eq!(
            pages.scans,
            vec![Scan {
                page: 3,
                image: vec![0xFF, 0xD8],
                format: ImageFormat::Jpeg,
            }]
        );
        let scanned_only = Pages::read_scanning(
            2,
            |_| Ok(PageContent::default()),
            |_| Some((vec![1], ImageFormat::Png)),
        );
        assert!(scanned_only.is_ok_and(|p| p.sections.is_empty() && p.scans.len() == 2));
    }

    /// A row of `cells`, each `(left, right, text)`.
    fn row(y: f32, cells: &[(f32, f32, &str)]) -> Row {
        Row {
            y,
            top: y + 10.0,
            cells: cells
                .iter()
                .map(|(left, right, text)| Cell {
                    left: *left,
                    right: *right,
                    text: (*text).to_owned(),
                })
                .collect(),
        }
    }

    /// Rows of short cells in aligned columns are a table: labels by their
    /// left edge, numbers by their right; prose around them is not.
    #[test]
    fn aligned_rows_of_short_cells_are_a_table_without_rules() {
        let rows = vec![
            row(
                720.0,
                &[(72.0, 400.0, "Revenue by region for the third quarter.")],
            ),
            row(
                700.0,
                &[
                    (72.0, 110.0, "Region"),
                    (150.0, 166.0, "Jul"),
                    (200.0, 218.0, "Aug"),
                ],
            ),
            row(
                688.0,
                &[
                    (72.0, 100.0, "North"),
                    (148.0, 166.0, "410"),
                    (200.0, 218.0, "432"),
                ],
            ),
            row(
                676.0,
                &[
                    (72.0, 96.0, "West"),
                    (148.0, 166.0, "480"),
                    (194.0, 218.0, "1,401"),
                ],
            ),
            row(
                650.0,
                &[(72.0, 400.0, "The West fell after the warehouse closed.")],
            ),
        ];
        assert_eq!(borderless_tables(&rows), vec![1..4]);
        // Two rows are not enough to tell a table from a pair of lines.
        assert!(borderless_tables(rows.get(1..3).unwrap_or_default()).is_empty());
        // Columns that drift are not a table.
        let drift = vec![
            row(700.0, &[(72.0, 110.0, "a"), (150.0, 166.0, "b")]),
            row(688.0, &[(72.0, 110.0, "c"), (190.0, 230.0, "d")]),
            row(676.0, &[(72.0, 110.0, "e"), (260.0, 290.0, "f")]),
        ];
        assert!(borderless_tables(&drift).is_empty());
    }

    /// A table drawn as text in a grid, with no rules, becomes a table
    /// section, and its words leave the body text.
    #[test]
    fn a_table_without_rules_becomes_a_table_section() {
        let mut doc = pdf_oxide::writer::DocumentBuilder::new().title("Ops report");
        let mut page = doc
            .letter_page()
            .at(72.0, 720.0)
            .text("Revenue held in the East and fell in the West.");
        for (index, cells) in [
            ["Region", "Jul", "Aug", "Sep"],
            ["North", "410", "432", "455"],
            ["East", "520", "548", "590"],
            ["West", "480", "401", "362"],
        ]
        .iter()
        .enumerate()
        {
            let y = 680.0 - 18.0 * f32::from(u8::try_from(index).unwrap_or(0));
            for (column, cell) in cells.iter().enumerate() {
                let x = 72.0 + 60.0 * f32::from(u8::try_from(column).unwrap_or(0));
                page = page.at(x, y).text(cell);
            }
        }
        page.done();
        let bytes = doc.build().unwrap_or_default();
        let extracted = extract(&bytes);
        assert!(extracted.is_ok(), "{:?}", extracted.as_ref().err());
        let Ok(extracted) = extracted else { return };
        let is_table = |s: &&Section| s.kind == super::super::parser::SectionKind::Table;
        let tables: Vec<&Section> = extracted.sections.iter().filter(is_table).collect();
        assert_eq!(tables.len(), 1, "{:?}", extracted.sections);
        let table = tables.first().map(|s| s.text.as_str()).unwrap_or_default();
        assert!(table.starts_with("| Region | Jul | Aug | Sep |"), "{table}");
        assert!(table.contains("| West | 480 | 401 | 362 |"), "{table}");
        let body: String = extracted
            .sections
            .iter()
            .filter(|s| !is_table(s))
            .map(|s| s.text.as_str())
            .collect();
        assert!(body.contains("fell in the West"), "{body}");
        assert!(!body.contains("480"), "{body}");
    }

    #[test]
    fn a_pdf_whose_every_page_fails_or_is_blank_says_which() {
        let failed = Pages::read(3, |_| Err(String::from("bad")))
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(failed.contains("3 of 3 pages failed"), "{failed}");
        let blank = Pages::read(2, |_| Ok(PageContent::default()))
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(blank.contains("no text layer"), "{blank}");
    }
}
