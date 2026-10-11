//! PDF through `pdf_oxide`: each page as its typed regions, so running
//! headers, footers, page numbers, and other artifacts are left out,
//! structural headings start sections, and the tables the layout detector
//! finds become table sections of their own. The Info dictionary gives
//! the title, author, and dates.

use std::collections::BTreeMap;

use pdf_oxide::PdfDocument;
use pdf_oxide::editor::DocumentInfo;
use pdf_oxide::extractors::images::{ColorSpace, ImageData};
use pdf_oxide::layout::TextSpan;
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
            Ok(PageContent::of(&structured, &tables))
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

/// The real tables on a page, for the spans inside them.
struct Grids<'a>(Vec<&'a PdfTable>);

impl Grids<'_> {
    fn contains(&self, span: &TextSpan) -> bool {
        self.0.iter().any(|table| {
            table
                .bbox
                .as_ref()
                .is_some_and(|b| b.contains_point(&span.bbox.center()))
        })
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
    fn of(page: &StructuredPage, tables: &[PdfTable]) -> Self {
        let page = Page {
            page,
            grids: Grids(tables.iter().filter(|t| t.is_real_grid()).collect()),
        };
        let body_size = page.body_font_size();
        let (lines, mut pending) = page.lines();
        let mut out = Self::default();
        let mut current: (Option<String>, Vec<String>) = (None, Vec::new());
        // Tables sit where their top edge is, highest first.
        let mut placed: Vec<(f32, Table)> = page
            .grids
            .0
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
