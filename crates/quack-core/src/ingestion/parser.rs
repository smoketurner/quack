use std::collections::BTreeMap;
use std::path::Path;

use super::budget::DecompressionBudget;
use super::{captions, code, epub, html, mail, markdown, odt, office, pdf, rtf};
use crate::error::{Error, Result};
use crate::text::NonBlankText;

/// Recognized file types for ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Csv,
    Tsv,
    Parquet,
    Json,
    Xlsx,
    Pdf,
    Text,
    Markdown,
    Html,
    Docx,
    Pptx,
    Epub,
    Odt,
    /// One RFC 5322 message.
    Eml,
    /// A mailbox of messages, `From ` separated.
    Mbox,
    /// `WebVTT` captions.
    Vtt,
    /// `SubRip` captions.
    Srt,
    /// Source code, chunked by line with line-number locators.
    Code,
    Rtf,
}

/// How a file type loads into a workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Load {
    /// One table named after the file, read by this reader.
    Table(Reader),
    /// One table per sheet.
    Workbook,
    /// Text in this format, parsed and chunked.
    Chunks(TextFormat),
}

/// A file type whose text is parsed and chunked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextFormat {
    Pdf,
    Text,
    Markdown,
    Html,
    Docx,
    Pptx,
    Epub,
    Odt,
    Eml,
    Mbox,
    Vtt,
    Srt,
    Code,
    Rtf,
}

/// A `DuckDB` reader for a data file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reader {
    /// Delimited text: the dialect is sniffed, and the separator is the
    /// one the file's type names, which a one-column sniff is checked
    /// against.
    Csv(Separator),
    Parquet,
    Json,
}

/// The field separator a delimited file's type names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Separator {
    Comma,
    Tab,
}

impl Separator {
    /// The `delim` value `read_csv` takes.
    #[must_use]
    pub const fn as_sql(self) -> &'static str {
        match self {
            Self::Comma => ",",
            Self::Tab => "\t",
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Comma => "comma",
            Self::Tab => "tab",
        }
    }
}

impl Reader {
    /// The table function that reads the file with a sniffed dialect.
    #[must_use]
    pub fn sql_fn(self) -> &'static str {
        match self {
            Self::Csv(_) => "read_csv_auto",
            Self::Parquet => "read_parquet",
            Self::Json => "read_json_auto",
        }
    }

    /// The reader's name as quack shows it to a person, never a server
    /// path. Used for the user-facing message when a file fails to parse,
    /// so the on-disk path the reader's raw error names stays in the log.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Csv(Separator::Comma) => "comma-separated values",
            Self::Csv(Separator::Tab) => "tab-separated values",
            Self::Parquet => "Parquet",
            Self::Json => "JSON",
        }
    }

    /// The reader for bytes of no known name: Parquet by its magic, JSON
    /// when they start with `{` or `[`, else comma-separated text (its
    /// dialect sniffed).
    #[must_use]
    pub fn sniff(data: &[u8]) -> Self {
        if data.starts_with(b"PAR1") {
            return Self::Parquet;
        }
        match data.iter().find(|b| !b.is_ascii_whitespace()) {
            Some(b'{' | b'[') => Self::Json,
            _ => Self::Csv(Separator::Comma),
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
            Self::Csv => Load::Table(Reader::Csv(Separator::Comma)),
            Self::Tsv => Load::Table(Reader::Csv(Separator::Tab)),
            Self::Parquet => Load::Table(Reader::Parquet),
            Self::Json => Load::Table(Reader::Json),
            Self::Xlsx => Load::Workbook,
            Self::Pdf => Load::Chunks(TextFormat::Pdf),
            Self::Text => Load::Chunks(TextFormat::Text),
            Self::Markdown => Load::Chunks(TextFormat::Markdown),
            Self::Html => Load::Chunks(TextFormat::Html),
            Self::Docx => Load::Chunks(TextFormat::Docx),
            Self::Pptx => Load::Chunks(TextFormat::Pptx),
            Self::Epub => Load::Chunks(TextFormat::Epub),
            Self::Odt => Load::Chunks(TextFormat::Odt),
            Self::Eml => Load::Chunks(TextFormat::Eml),
            Self::Mbox => Load::Chunks(TextFormat::Mbox),
            Self::Vtt => Load::Chunks(TextFormat::Vtt),
            Self::Srt => Load::Chunks(TextFormat::Srt),
            Self::Code => Load::Chunks(TextFormat::Code),
            Self::Rtf => Load::Chunks(TextFormat::Rtf),
        }
    }

    /// The extensions of the files that load as tables, in table order.
    pub fn table_extensions() -> impl Iterator<Item = &'static str> {
        EXTENSIONS
            .iter()
            .filter(|(_, file_type)| !matches!(file_type.load(), Load::Chunks(_)))
            .map(|(ext, _)| *ext)
    }

    #[must_use]
    pub fn mime_type(self) -> &'static str {
        match self {
            Self::Csv => "text/csv",
            Self::Tsv => "text/tab-separated-values",
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
            Self::Epub => "application/epub+zip",
            Self::Odt => "application/vnd.oasis.opendocument.text",
            Self::Eml => "message/rfc822",
            Self::Mbox => "application/mbox",
            Self::Vtt => "text/vtt",
            Self::Srt => "application/x-subrip",
            Self::Code => "text/x-source",
            Self::Rtf => "application/rtf",
        }
    }
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Csv => "CSV",
            Self::Tsv => "TSV",
            Self::Parquet => "Parquet",
            Self::Json => "JSON",
            Self::Xlsx => "Excel",
            Self::Pdf => "PDF",
            Self::Text => "Text",
            Self::Markdown => "Markdown",
            Self::Html => "HTML",
            Self::Docx => "Word",
            Self::Pptx => "PowerPoint",
            Self::Epub => "EPUB",
            Self::Odt => "OpenDocument text",
            Self::Eml => "Email",
            Self::Mbox => "Mailbox",
            Self::Vtt => "WebVTT captions",
            Self::Srt => "SubRip captions",
            Self::Code => "Source code",
            Self::Rtf => "Rich Text",
        };
        f.write_str(label)
    }
}

/// How a document's sections relate: real divisions of the text, or the
/// pages of one continuous text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Flow {
    /// Sections are headings, slides, or the whole file: a chunk never
    /// crosses one.
    #[default]
    Sectioned,
    /// Sections are pages of one running text: chunks are windowed over
    /// the whole text and carry the page they start on, so a paragraph
    /// split by a page break stays in one chunk.
    Continuous,
}

/// What a parse yields: the document's own title when the format carries
/// one (`<title>`, Office core properties, a PDF's Info dictionary), its
/// sections and how they relate, how its pages read, and the metadata
/// the file carries about itself.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Extracted {
    pub title: Option<String>,
    pub sections: Vec<Section>,
    pub flow: Flow,
    /// How the pages read; `None` for a source without pages (only a PDF
    /// has them).
    pub pages: Option<PageCounts>,
    pub meta: DocumentMeta,
}

/// What a file says about itself: who wrote it and when, its tags, and
/// any other named value the format carries (a PDF's subject, a mail's
/// recipients). Dates are kept as the source wrote them, trimmed; the
/// store casts what parses as a timestamp and keeps the text otherwise.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DocumentMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authored_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, String>,
}

impl DocumentMeta {
    /// Set a text field from a value that may be blank.
    pub(crate) fn set(slot: &mut Option<String>, value: Option<&str>) {
        if let Some(v) = value.map(str::trim).filter(|v| !v.is_empty()) {
            *slot = Some(v.to_owned());
        }
    }

    /// Add a tag, trimmed, once.
    pub(crate) fn tag(&mut self, tag: &str) {
        let tag = tag.trim();
        if !tag.is_empty() && !self.tags.iter().any(|t| t == tag) {
            self.tags.push(tag.to_owned());
        }
    }

    /// Keep a named value, trimmed, when it is not blank.
    pub(crate) fn extra(&mut self, key: &str, value: Option<&str>) {
        if let Some(v) = value.map(str::trim).filter(|v| !v.is_empty()) {
            self.extra.insert(key.to_owned(), v.to_owned());
        }
    }

    /// Whether nothing was found.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.author.is_none()
            && self.authored_at.is_none()
            && self.modified_at.is_none()
            && self.tags.is_empty()
            && self.extra.is_empty()
    }
}

/// How a paginated document's pages read. A page left out of the text is
/// one of two kinds, which mean different things to the person: its
/// extraction failed, or it holds no text (a scanned image).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct PageCounts {
    pub total: u32,
    /// Pages whose extraction failed.
    pub unreadable: u32,
    /// Pages that read without error and yielded no text.
    pub empty: u32,
}

impl PageCounts {
    /// What a person is told when pages are missing from the text, as
    /// `3 of 40 pages unreadable`; `None` when every page was kept.
    #[must_use]
    pub fn note(self) -> Option<String> {
        let Self {
            total,
            unreadable,
            empty,
        } = self;
        match (unreadable, empty) {
            (0, 0) => None,
            (_, 0) => Some(format!("{unreadable} of {total} pages unreadable")),
            (0, _) => Some(format!("{empty} of {total} pages without text")),
            (_, _) => Some(format!(
                "{unreadable} of {total} pages unreadable, {empty} without text"
            )),
        }
    }

    /// The note of `pages` after a comma, to end a one-line description
    /// of a document; empty when every page was kept or there are none.
    #[must_use]
    pub fn suffix(pages: Option<Self>) -> String {
        pages
            .and_then(Self::note)
            .map_or(String::new(), |note| format!(", {note}"))
    }
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

/// A run of text that shares one heading, one page, and one kind.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Section {
    /// Nearest preceding heading, if the source has headings.
    pub heading: Option<String>,
    /// 1-based page number for paginated sources.
    pub page: Option<u32>,
    pub text: String,
    pub kind: SectionKind,
    /// Where the text sits in a source without pages, as a citation says
    /// it: `line 40`, `12:04`, `chapter 3`, `message 2`.
    pub locator: Option<String>,
}

impl Section {
    /// Body text under `heading`.
    #[must_use]
    pub fn body(heading: Option<String>, text: impl Into<String>) -> Self {
        Self {
            heading,
            text: text.into(),
            ..Self::default()
        }
    }

    /// A table's Markdown rendering under `heading`.
    #[must_use]
    pub fn table(heading: Option<String>, markdown: String) -> Self {
        Self {
            heading,
            text: markdown,
            kind: SectionKind::Table,
            ..Self::default()
        }
    }

    /// A footnote, endnote, comment, or speaker note under `heading`.
    #[must_use]
    pub fn note(heading: Option<String>, text: impl Into<String>) -> Self {
        Self {
            heading,
            text: text.into(),
            kind: SectionKind::Note,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn on_page(mut self, page: Option<u32>) -> Self {
        self.page = page;
        self
    }

    #[must_use]
    pub fn at(mut self, locator: impl Into<String>) -> Self {
        self.locator = Some(locator.into());
        self
    }
}

/// What a section (and the chunks cut from it) holds: running text, a
/// table rendered as Markdown, a note (footnote, endnote, comment,
/// speaker note), or source code.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SectionKind {
    #[default]
    Body,
    Table,
    Note,
    Code,
}

text_enum!(SectionKind, "section kind", {
    Body => "body",
    Table => "table",
    Note => "note",
    Code => "code",
});
text_enum_sql!(SectionKind);

/// Every extension quack reads, lowercase, and the type it is read as.
const EXTENSIONS: &[(&str, FileType)] = &[
    ("csv", FileType::Csv),
    ("tsv", FileType::Tsv),
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
    ("epub", FileType::Epub),
    ("odt", FileType::Odt),
    ("eml", FileType::Eml),
    ("mbox", FileType::Mbox),
    ("vtt", FileType::Vtt),
    ("srt", FileType::Srt),
    ("rtf", FileType::Rtf),
    ("rs", FileType::Code),
    ("py", FileType::Code),
    ("js", FileType::Code),
    ("ts", FileType::Code),
    ("tsx", FileType::Code),
    ("jsx", FileType::Code),
    ("go", FileType::Code),
    ("java", FileType::Code),
    ("kt", FileType::Code),
    ("c", FileType::Code),
    ("h", FileType::Code),
    ("cpp", FileType::Code),
    ("hpp", FileType::Code),
    ("cs", FileType::Code),
    ("rb", FileType::Code),
    ("php", FileType::Code),
    ("swift", FileType::Code),
    ("scala", FileType::Code),
    ("sh", FileType::Code),
    ("sql", FileType::Code),
    ("toml", FileType::Code),
    ("yaml", FileType::Code),
    ("yml", FileType::Code),
];

impl TextFormat {
    /// Extract a file's text: PDFs one section per page under the page's
    /// structural headings, Markdown one per heading, HTML one per heading
    /// with the `<title>`, DOCX one per heading with the core title, PPTX
    /// one per slide, EPUB one per chapter, ODT one per heading, mail one
    /// per message, captions in timed runs, source code one section with
    /// line locators, plain text and RTF a single section. Tables in any of
    /// them are their own sections, rendered as Markdown.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be parsed, or if it has no text
    /// at all (a scanned PDF without a text layer needs OCR, which is not
    /// supported), or if a zipped format inflates past `budget`.
    pub fn extract(self, data: &[u8], budget: DecompressionBudget) -> Result<Extracted> {
        match self {
            Self::Pdf => pdf::extract(data),
            Self::Markdown => Ok(markdown::extract(&utf8(data)?)),
            Self::Text => Ok(Extracted {
                sections: vec![Section::body(None, utf8(data)?)],
                flow: Flow::Sectioned,
                ..Extracted::default()
            }),
            Self::Html => html::html(&utf8(data)?),
            Self::Docx => office::docx(data, budget),
            Self::Pptx => office::pptx(data, budget),
            Self::Epub => epub::extract(data, budget),
            Self::Odt => odt::extract(data, budget),
            Self::Eml => mail::eml(data),
            Self::Mbox => mail::mbox(data),
            Self::Vtt => captions::vtt(&utf8(data)?),
            Self::Srt => captions::srt(&utf8(data)?),
            Self::Code => Ok(code::extract(&utf8(data)?)),
            Self::Rtf => rtf::extract(&utf8(data)?),
        }
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

    /// End the current body section and add `section` whole, under the
    /// current heading when it has none.
    pub(crate) fn push(&mut self, mut section: Section) {
        self.flush();
        if section.heading.is_none() {
            section.heading.clone_from(&self.heading);
        }
        if !section.text.trim().is_empty() {
            self.sections.push(section);
        }
    }

    fn flush(&mut self) {
        let text = self.lines.join("\n").trim().to_owned();
        self.lines.clear();
        if !text.is_empty() {
            self.sections
                .push(Section::body(self.heading.clone(), text));
        }
    }

    /// Every section, the last one included.
    pub(crate) fn finish(mut self) -> Vec<Section> {
        self.flush();
        self.sections
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUDGET: DecompressionBudget = DecompressionBudget::megabytes(64);

    #[test]
    fn title_is_the_first_heading_when_there_is_one() {
        let md = markdown::sections("# Renewal terms\n\nBody.\n\n## Detail\n\nMore.\n");
        assert_eq!(title_of(&md), Some("Renewal terms"));
        let plain = vec![Section::body(None, "no headings")];
        assert_eq!(title_of(&plain), None);
        assert_eq!(title_of(&[]), None);
    }

    #[test]
    fn detects_every_extension_family() {
        assert_eq!(FileType::of("sales.csv"), Some(FileType::Csv));
        assert_eq!(FileType::of("DATA.CSV"), Some(FileType::Csv));
        assert_eq!(FileType::of("data.parquet"), Some(FileType::Parquet));
        assert_eq!(FileType::of("data.pq"), Some(FileType::Parquet));
        assert_eq!(FileType::of("config.json"), Some(FileType::Json));
        assert_eq!(FileType::of("events.jsonl"), Some(FileType::Json));
        assert_eq!(FileType::of("stream.ndjson"), Some(FileType::Json));
        assert_eq!(FileType::of("report.pdf"), Some(FileType::Pdf));
        assert_eq!(FileType::of("notes.txt"), Some(FileType::Text));
        assert_eq!(FileType::of("readme.md"), Some(FileType::Markdown));
        assert_eq!(FileType::of("book.epub"), Some(FileType::Epub));
        assert_eq!(FileType::of("letter.odt"), Some(FileType::Odt));
        assert_eq!(FileType::of("thread.eml"), Some(FileType::Eml));
        assert_eq!(FileType::of("inbox.mbox"), Some(FileType::Mbox));
        assert_eq!(FileType::of("meeting.vtt"), Some(FileType::Vtt));
        assert_eq!(FileType::of("film.srt"), Some(FileType::Srt));
        assert_eq!(FileType::of("main.rs"), Some(FileType::Code));
        assert_eq!(FileType::of("app.PY"), Some(FileType::Code));
        assert_eq!(FileType::of("memo.rtf"), Some(FileType::Rtf));
        assert_eq!(FileType::of("image.png"), None);
        assert_eq!(FileType::of("noext"), None);
        assert!(FileType::table_extensions().any(|e| e == "xlsx"));
        assert!(!FileType::table_extensions().any(|e| e == "epub"));
    }

    #[test]
    fn extracts_plain_text_and_code() {
        let text = TextFormat::Text
            .extract(b"Hello, world!", BUDGET)
            .ok()
            .and_then(|e| e.sections.into_iter().next())
            .map(|s| s.text);
        assert_eq!(text, Some(String::from("Hello, world!")));
        let code = TextFormat::Code
            .extract(b"fn main() {}\n", BUDGET)
            .ok()
            .and_then(|e| e.sections.into_iter().next());
        assert_eq!(code.as_ref().map(|s| s.kind), Some(SectionKind::Code));
        assert!(TextFormat::Text.extract(&[0xff, 0xfe], BUDGET).is_err());
    }

    #[test]
    fn pdf_without_text_layer_is_an_error() {
        let err = TextFormat::Pdf.extract(b"%PDF-1.4\n%%EOF", BUDGET).err();
        assert!(err.is_some());
    }

    /// A PDF with `pages` pages, each carrying one line naming its number.
    fn long_pdf(pages: u32, title: &str) -> Vec<u8> {
        let mut doc = pdf_oxide::writer::DocumentBuilder::new()
            .title(title)
            .author("Ada");
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
        let extracted = TextFormat::Pdf
            .extract(&long_pdf(60, "Long Report"), BUDGET)
            .unwrap();
        assert_eq!(extracted.title.as_deref(), Some("Long Report"));
        assert_eq!(extracted.meta.author.as_deref(), Some("Ada"));
        assert_eq!(
            extracted.pages,
            Some(PageCounts {
                total: 60,
                unreadable: 0,
                empty: 0
            })
        );
        assert_eq!(extracted.pages.and_then(PageCounts::note), None);
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
    fn the_page_note_names_each_kind_of_missing_page() {
        let counts = |unreadable, empty| PageCounts {
            total: 40,
            unreadable,
            empty,
        };
        assert_eq!(counts(0, 0).note(), None);
        assert_eq!(
            counts(3, 0).note().as_deref(),
            Some("3 of 40 pages unreadable")
        );
        assert_eq!(
            counts(0, 2).note().as_deref(),
            Some("2 of 40 pages without text")
        );
        assert_eq!(
            counts(3, 2).note().as_deref(),
            Some("3 of 40 pages unreadable, 2 without text")
        );
        // After a comma in a one-line description, or nothing at all.
        assert_eq!(
            PageCounts::suffix(Some(counts(3, 0))),
            ", 3 of 40 pages unreadable"
        );
        assert_eq!(PageCounts::suffix(Some(counts(0, 0))), "");
        assert_eq!(PageCounts::suffix(None), "");
    }

    #[test]
    fn document_meta_trims_dedups_and_knows_when_it_is_empty() {
        let mut meta = DocumentMeta::default();
        assert!(meta.is_empty());
        DocumentMeta::set(&mut meta.author, Some("  "));
        assert!(meta.is_empty());
        DocumentMeta::set(&mut meta.author, Some(" Ada "));
        meta.tag(" policy ");
        meta.tag("policy");
        meta.extra("subject", Some(""));
        meta.extra("subject", Some(" Renewals "));
        assert_eq!(meta.author.as_deref(), Some("Ada"));
        assert_eq!(meta.tags, ["policy"]);
        assert_eq!(
            meta.extra.get("subject").map(String::as_str),
            Some("Renewals")
        );
        assert!(!meta.is_empty());
    }
}
