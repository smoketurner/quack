use std::path::Path;

use pdf_oxide::PdfDocument;
use pdf_oxide::editor::DocumentInfo;

use super::{html, office};
use crate::error::{Error, Result};
use crate::okf::parse_front_matter;
use crate::text::NonBlankText;

/// Recognized file types for ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Csv,
    Parquet,
    Json,
    Xlsx,
    Pdf,
    Text,
    Markdown,
    Html,
    Docx,
    Pptx,
}

/// How a file type loads into a workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Load {
    /// One table named after the file, read by this reader.
    Table(Reader),
    /// One table per sheet.
    Workbook,
    /// Text, parsed and chunked.
    Chunks,
}

/// A `DuckDB` reader for a data file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reader {
    Csv,
    Parquet,
    Json,
}

impl Reader {
    /// The table function that reads the file.
    #[must_use]
    pub fn sql_fn(self) -> &'static str {
        match self {
            Self::Csv => "read_csv_auto",
            Self::Parquet => "read_parquet",
            Self::Json => "read_json_auto",
        }
    }

    /// The reader for bytes of no known name: Parquet by its magic, JSON
    /// when they start with `{` or `[`, else CSV (its delimiter sniffed).
    #[must_use]
    pub fn sniff(data: &[u8]) -> Self {
        if data.starts_with(b"PAR1") {
            return Self::Parquet;
        }
        match data.iter().find(|b| !b.is_ascii_whitespace()) {
            Some(b'{' | b'[') => Self::Json,
            _ => Self::Csv,
        }
    }
}

impl FileType {
    /// The type a file name's extension says, in any case; `None` for a
    /// file nothing here reads.
    #[must_use]
    pub fn of(filename: &str) -> Option<Self> {
        let ext = Path::new(filename)
            .extension()
            .and_then(|e| e.to_str())?
            .to_ascii_lowercase();
        EXTENSIONS
            .iter()
            .find(|(known, _)| *known == ext)
            .map(|(_, file_type)| *file_type)
    }

    /// How files of this type load.
    #[must_use]
    pub fn load(self) -> Load {
        match self {
            Self::Csv => Load::Table(Reader::Csv),
            Self::Parquet => Load::Table(Reader::Parquet),
            Self::Json => Load::Table(Reader::Json),
            Self::Xlsx => Load::Workbook,
            Self::Pdf | Self::Text | Self::Markdown | Self::Html | Self::Docx | Self::Pptx => {
                Load::Chunks
            }
        }
    }

    /// The extensions of the files that load as tables, in table order.
    pub fn table_extensions() -> impl Iterator<Item = &'static str> {
        EXTENSIONS
            .iter()
            .filter(|(_, file_type)| file_type.load() != Load::Chunks)
            .map(|(ext, _)| *ext)
    }

    #[must_use]
    pub fn mime_type(self) -> &'static str {
        match self {
            Self::Csv => "text/csv",
            Self::Parquet => "application/vnd.apache.parquet",
            Self::Json => "application/json",
            Self::Xlsx => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            Self::Pdf => "application/pdf",
            Self::Text => "text/plain",
            Self::Markdown => "text/markdown",
            Self::Html => "text/html",
            Self::Docx => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            Self::Pptx => {
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            }
        }
    }
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Csv => "CSV",
            Self::Parquet => "Parquet",
            Self::Json => "JSON",
            Self::Xlsx => "Excel",
            Self::Pdf => "PDF",
            Self::Text => "Text",
            Self::Markdown => "Markdown",
            Self::Html => "HTML",
            Self::Docx => "Word",
            Self::Pptx => "PowerPoint",
        };
        f.write_str(label)
    }
}

/// How a document's sections relate: real divisions of the text, or the
/// pages of one continuous text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// Sections are headings, slides, or the whole file: a chunk never
    /// crosses one.
    Sectioned,
    /// Sections are pages of one running text: chunks are windowed over
    /// the whole text and carry the page they start on, so a paragraph
    /// split by a page break stays in one chunk.
    Continuous,
}

/// What a parse yields: the document's own title when the format carries
/// one (`<title>`, Office core properties, a PDF's Info dictionary), its
/// sections and how they relate, and how many pages the parser had to
/// skip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Extracted {
    pub title: Option<String>,
    pub sections: Vec<Section>,
    pub flow: Flow,
    /// Pages whose text could not be read; only a paginated source (PDF)
    /// ever reports any, and the document keeps every other page.
    pub pages_skipped: u32,
}

impl Extracted {
    /// The title: the document's own, else the first section's heading.
    #[must_use]
    pub fn title(&self) -> Option<&str> {
        self.title
            .as_deref()
            .and_then(str::non_blank)
            .or_else(|| title_of(&self.sections))
    }
}

/// A run of text that shares one heading and one page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// Nearest preceding heading, if the source has headings.
    pub heading: Option<String>,
    /// 1-based page number for paginated sources.
    pub page: Option<u32>,
    pub text: String,
}

/// Every extension quack reads, lowercase, and the type it is read as.
const EXTENSIONS: &[(&str, FileType)] = &[
    ("csv", FileType::Csv),
    ("tsv", FileType::Csv),
    ("parquet", FileType::Parquet),
    ("pq", FileType::Parquet),
    ("json", FileType::Json),
    ("jsonl", FileType::Json),
    ("ndjson", FileType::Json),
    ("xlsx", FileType::Xlsx),
    ("xlsm", FileType::Xlsx),
    ("xls", FileType::Xlsx),
    ("ods", FileType::Xlsx),
    ("pdf", FileType::Pdf),
    ("md", FileType::Markdown),
    ("markdown", FileType::Markdown),
    ("txt", FileType::Text),
    ("text", FileType::Text),
    ("log", FileType::Text),
    ("html", FileType::Html),
    ("htm", FileType::Html),
    ("xhtml", FileType::Html),
    ("docx", FileType::Docx),
    ("pptx", FileType::Pptx),
];

/// Extract an unstructured file: PDFs one section per page, Markdown one
/// per heading, HTML one per heading with the `<title>`, DOCX one per
/// heading style with the core title, PPTX one per slide, plain text a
/// single section.
///
/// # Errors
///
/// Returns an error if the file cannot be parsed, or if it has no text at
/// all (a scanned PDF without a text layer needs OCR, which is not
/// supported).
pub fn extract(file_type: FileType, data: &[u8]) -> Result<Extracted> {
    match file_type {
        FileType::Pdf => extract_pdf(data),
        FileType::Markdown => {
            let text = utf8(data)?;
            // YAML front matter (Obsidian, Jekyll, OKF) is metadata, not
            // prose: its `title` is the document's, the rest is dropped.
            let (front, body) = parse_front_matter(&text);
            Ok(Extracted {
                title: front.get("title").map(str::to_owned),
                sections: markdown_sections(body),
                flow: Flow::Sectioned,
                pages_skipped: 0,
            })
        }
        FileType::Text => Ok(Extracted {
            title: None,
            sections: vec![Section {
                heading: None,
                page: None,
                text: utf8(data)?,
            }],
            flow: Flow::Sectioned,
            pages_skipped: 0,
        }),
        FileType::Html => html::html(&utf8(data)?),
        FileType::Docx => office::docx(data),
        FileType::Pptx => office::pptx(data),
        FileType::Csv | FileType::Parquet | FileType::Json | FileType::Xlsx => Err(
            Error::Ingestion(format!("cannot extract text from {file_type} files")),
        ),
    }
}

/// The document title a parse yields: the first section's heading when
/// the source has headings (Markdown's first `#` line, an HTML `<title>`),
/// else nothing.
fn title_of(sections: &[Section]) -> Option<&str> {
    sections
        .first()
        .and_then(|s| s.heading.as_deref())
        .and_then(str::non_blank)
}

fn utf8(data: &[u8]) -> Result<String> {
    String::from_utf8(data.to_vec()).map_err(|e| Error::Ingestion(format!("invalid UTF-8: {e}")))
}

/// A PDF, one section per page. A page the parser cannot read is skipped
/// and counted rather than ending the document there, so one bad font
/// never drops every page after it.
fn extract_pdf(data: &[u8]) -> Result<Extracted> {
    let doc = PdfDocument::from_bytes(data.to_vec())
        .map_err(|e| Error::Ingestion(format!("PDF extraction failed: {e}")))?;
    if !doc.is_authenticated() {
        return Err(Error::Ingestion(String::from(
            "the PDF is password-protected; remove the password and upload it again",
        )));
    }
    let page_count = doc
        .page_count()
        .map_err(|e| Error::Ingestion(format!("PDF extraction failed: {e}")))?;
    let (sections, pages_skipped) = extract_pdf_pages(page_count, |index| {
        doc.extract_text(index).map_err(|e| e.to_string())
    })?;
    Ok(Extracted {
        title: pdf_title(&doc),
        sections,
        flow: Flow::Continuous,
        pages_skipped,
    })
}

/// Read `page_count` pages with `read`, one section per page that has
/// text, skipping and counting the pages that fail.
///
/// # Errors
///
/// Returns an error when no page yields text: every page failed, or the
/// file has no text layer (a scanned PDF).
fn extract_pdf_pages(
    page_count: usize,
    read: impl Fn(usize) -> std::result::Result<String, String>,
) -> Result<(Vec<Section>, u32)> {
    let mut sections = Vec::new();
    let mut skipped: u32 = 0;
    for index in 0..page_count {
        let page = u32::try_from(index.saturating_add(1)).unwrap_or(u32::MAX);
        match read(index) {
            Ok(text) if !text.trim().is_empty() => sections.push(Section {
                heading: None,
                page: Some(page),
                text,
            }),
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(page, error = %error, "skipping an unreadable PDF page");
                skipped = skipped.saturating_add(1);
            }
        }
    }
    if sections.is_empty() {
        if skipped > 0 {
            return Err(Error::Ingestion(format!(
                "no readable text: {skipped} of {page_count} pages failed to parse"
            )));
        }
        return Err(Error::Ingestion(String::from(
            "no extractable text: the PDF has no text layer (scanned pages need OCR)",
        )));
    }
    Ok((sections, skipped))
}

/// The Info dictionary's `/Title`, when the file carries one.
fn pdf_title(doc: &PdfDocument) -> Option<String> {
    let info_ref = doc.trailer().as_dict()?.get("Info")?.as_reference()?;
    let info = doc.load_object(info_ref).ok()?;
    DocumentInfo::from_object(&info)
        .title
        .map(|t| t.trim().to_owned())
        .filter(|t| !t.is_empty())
}

/// Split Markdown at ATX (`# Title`) and setext (underlined) headings. Text
/// before the first heading becomes a section without one.
fn markdown_sections(text: &str) -> Vec<Section> {
    let lines: Vec<&str> = text.lines().collect();
    let mut sections = SectionBuilder::default();
    let mut i = 0;
    while i < lines.len() {
        let line = lines.get(i).copied().unwrap_or_default();
        let next = lines.get(i.saturating_add(1)).copied();
        if let Some(title) = atx_heading(line) {
            sections.heading(title);
            i = i.saturating_add(1);
            continue;
        }
        if let Some(underline) = next
            && is_setext_underline(underline)
            && !line.trim().is_empty()
            && !line.trim_start().starts_with(['-', '*', '+', '>', '|'])
        {
            sections.heading(line.trim().to_owned());
            i = i.saturating_add(2);
            continue;
        }
        sections.line(line);
        i = i.saturating_add(1);
    }
    sections.finish()
}

/// Sections built a line at a time: the lines under the current heading
/// become one section when the next heading starts, trimmed, and only when
/// they hold text.
#[derive(Debug, Default)]
pub(crate) struct SectionBuilder {
    sections: Vec<Section>,
    heading: Option<String>,
    lines: Vec<String>,
}

impl SectionBuilder {
    /// Add a line to the current section.
    pub(crate) fn line(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }

    /// End the current section and start one under `heading` (none when
    /// it is blank).
    pub(crate) fn heading(&mut self, heading: impl Into<String>) {
        self.flush();
        let heading = heading.into();
        self.heading = (!heading.trim().is_empty()).then_some(heading);
    }

    fn flush(&mut self) {
        let text = self.lines.join("\n").trim().to_owned();
        self.lines.clear();
        if !text.is_empty() {
            self.sections.push(Section {
                heading: self.heading.clone(),
                page: None,
                text,
            });
        }
    }

    /// Every section, the last one included.
    pub(crate) fn finish(mut self) -> Vec<Section> {
        self.flush();
        self.sections
    }
}

fn atx_heading(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let hashes = trimmed.chars().take_while(|c| *c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = trimmed.get(hashes..)?;
    if !rest.starts_with([' ', '\t']) {
        return None;
    }
    let title = rest.trim().trim_end_matches('#').trim();
    if title.is_empty() {
        None
    } else {
        Some(title.to_owned())
    }
}

fn is_setext_underline(line: &str) -> bool {
    let t = line.trim();
    t.len() >= 3 && (t.chars().all(|c| c == '=') || t.chars().all(|c| c == '-'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_front_matter_gives_the_title_and_is_not_chunked() {
        let extracted = extract(
            FileType::Markdown,
            b"---\ntitle: Renewal Guide\ntags: [a]\n---\n\n# Terms\n\nThirty days.\n",
        )
        .unwrap_or_else(|_| Extracted {
            title: None,
            sections: Vec::new(),
            flow: Flow::Sectioned,
            pages_skipped: 0,
        });
        assert_eq!(extracted.title(), Some("Renewal Guide"));
        assert_eq!(extracted.sections.len(), 1);
        assert!(extracted.sections.iter().all(|s| !s.text.contains("tags:")));
    }

    #[test]
    fn title_is_the_first_heading_when_there_is_one() {
        let md = markdown_sections("# Renewal terms\n\nBody.\n\n## Detail\n\nMore.\n");
        assert_eq!(title_of(&md), Some("Renewal terms"));
        let plain = vec![Section {
            heading: None,
            page: None,
            text: String::from("no headings"),
        }];
        assert_eq!(title_of(&plain), None);
        assert_eq!(title_of(&[]), None);
    }

    #[test]
    fn detects_csv() {
        assert_eq!(FileType::of("sales.csv"), Some(FileType::Csv));
        assert_eq!(FileType::of("DATA.CSV"), Some(FileType::Csv));
    }

    #[test]
    fn detects_parquet() {
        assert_eq!(FileType::of("data.parquet"), Some(FileType::Parquet));
        assert_eq!(FileType::of("data.pq"), Some(FileType::Parquet));
    }

    #[test]
    fn detects_json() {
        assert_eq!(FileType::of("config.json"), Some(FileType::Json));
        assert_eq!(FileType::of("events.jsonl"), Some(FileType::Json));
        assert_eq!(FileType::of("stream.ndjson"), Some(FileType::Json));
    }

    #[test]
    fn detects_pdf() {
        assert_eq!(FileType::of("report.pdf"), Some(FileType::Pdf));
    }

    #[test]
    fn detects_text() {
        assert_eq!(FileType::of("notes.txt"), Some(FileType::Text));
        assert_eq!(FileType::of("readme.md"), Some(FileType::Markdown));
    }

    #[test]
    fn unknown_extension() {
        assert_eq!(FileType::of("image.png"), None);
        assert_eq!(FileType::of("noext"), None);
    }

    #[test]
    fn extracts_plain_text() {
        let data = b"Hello, world!";
        let text = extract(FileType::Text, data)
            .ok()
            .and_then(|e| e.sections.into_iter().next())
            .map(|s| s.text);
        assert_eq!(text, Some(String::from("Hello, world!")));
    }

    #[test]
    fn rejects_structured_extraction() {
        let result = extract(FileType::Csv, b"a,b,c");
        assert!(result.is_err());
    }

    #[test]
    fn markdown_splits_on_atx_and_setext_headings() {
        let md = "intro line\n\n# Exclusions\n\nFlood is excluded.\n\nClaims\n------\n\nClose in 30 days.\n\n## Not a heading\ntext\n#nope\n- list\n---\n";
        let sections = markdown_sections(md);
        let summary: Vec<(Option<&str>, &str)> = sections
            .iter()
            .map(|s| (s.heading.as_deref(), s.text.as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![
                (None, "intro line"),
                (Some("Exclusions"), "Flood is excluded."),
                (Some("Claims"), "Close in 30 days."),
                (Some("Not a heading"), "text\n#nope\n- list\n---"),
            ]
        );
    }

    #[test]
    fn markdown_without_headings_is_one_section() {
        let sections = markdown_sections("just\n\ntext");
        assert_eq!(sections.len(), 1);
        assert_eq!(sections.first().and_then(|s| s.heading.as_deref()), None);
    }

    #[test]
    fn atx_heading_requires_space_and_trims_closing_hashes() {
        assert_eq!(atx_heading("## Title ##"), Some(String::from("Title")));
        assert_eq!(atx_heading("#nospace"), None);
        assert_eq!(atx_heading("####### seven"), None);
        assert_eq!(atx_heading("# "), None);
    }

    #[test]
    fn pdf_without_text_layer_is_an_error() {
        let err = extract(FileType::Pdf, b"%PDF-1.4\n%%EOF").err();
        assert!(err.is_some());
    }

    /// A PDF with `pages` pages, each carrying one line naming its number.
    fn long_pdf(pages: u32, title: &str) -> Vec<u8> {
        let mut doc = pdf_oxide::writer::DocumentBuilder::new().title(title);
        for page in 1..=pages {
            doc.letter_page()
                .at(72.0, 720.0)
                .text(&format!("Page {page} of the long report"))
                .done();
        }
        doc.build().unwrap_or_default()
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn every_page_of_a_long_pdf_is_kept_with_its_number_and_title() {
        let extracted = extract(FileType::Pdf, &long_pdf(60, "Long Report")).unwrap();
        assert_eq!(extracted.title.as_deref(), Some("Long Report"));
        assert_eq!(extracted.pages_skipped, 0);
        assert_eq!(extracted.sections.len(), 60);
        let pages: Vec<Option<u32>> = extracted.sections.iter().map(|s| s.page).collect();
        assert_eq!(pages, (1..=60).map(Some).collect::<Vec<_>>());
        let page_40 = extracted
            .sections
            .get(39)
            .map(|s| (s.text.as_str(), s.heading.as_deref()));
        assert!(
            page_40.is_some_and(|(text, heading)| text.contains("Page 40") && heading.is_none()),
            "{page_40:?}"
        );
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn a_page_that_fails_to_read_is_skipped_and_counted_not_the_rest() {
        let (sections, skipped) = extract_pdf_pages(4, |index| match index {
            1 => Err(String::from("bad font")),
            2 => Ok(String::from("   ")),
            _ => Ok(format!("text {index}")),
        })
        .unwrap();
        assert_eq!(skipped, 1);
        let pages: Vec<Option<u32>> = sections.iter().map(|s| s.page).collect();
        assert_eq!(pages, vec![Some(1), Some(4)]);
        assert_eq!(sections.get(1).map(|s| s.text.as_str()), Some("text 3"));
    }

    #[test]
    fn a_pdf_whose_every_page_fails_reports_the_count() {
        let err = extract_pdf_pages(3, |_| Err(String::from("bad"))).err();
        assert!(
            err.as_ref()
                .is_some_and(|e| e.to_string().contains("3 of 3 pages failed")),
            "{err:?}"
        );
    }

    #[test]
    fn a_pdf_whose_every_page_is_blank_is_a_scanned_document() {
        let err = extract_pdf_pages(2, |_| Ok(String::new())).err();
        assert!(
            err.as_ref()
                .is_some_and(|e| e.to_string().contains("no text layer")),
            "{err:?}"
        );
    }
}
