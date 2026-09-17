use std::path::Path;

use crate::error::{Error, Result};
use crate::okf::parse_front_matter;

/// Recognized file types for ingestion.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    Unknown,
}

impl FileType {
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
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
            Self::Unknown => "application/octet-stream",
        }
    }

    /// Loaded as tables rather than chunked.
    #[must_use]
    pub fn is_structured(&self) -> bool {
        match self {
            Self::Csv | Self::Parquet | Self::Json | Self::Xlsx => true,
            Self::Pdf
            | Self::Text
            | Self::Markdown
            | Self::Html
            | Self::Docx
            | Self::Pptx
            | Self::Unknown => false,
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
            Self::Unknown => "Unknown",
        };
        f.write_str(label)
    }
}

/// What a parse yields: the document's own title when the format carries
/// one (`<title>`, Office core properties), and its sections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Extracted {
    pub title: Option<String>,
    pub sections: Vec<Section>,
}

impl Extracted {
    /// The title: the document's own, else the first section's heading.
    #[must_use]
    pub fn title(&self) -> Option<&str> {
        self.title
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
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

/// Detect file type from the filename extension.
#[must_use]
pub fn detect_file_type(filename: &str) -> FileType {
    let ext = Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "csv" | "tsv" => FileType::Csv,
        "parquet" | "pq" => FileType::Parquet,
        "json" | "jsonl" | "ndjson" => FileType::Json,
        "xlsx" | "xlsm" | "xls" | "ods" => FileType::Xlsx,
        "pdf" => FileType::Pdf,
        "md" | "markdown" => FileType::Markdown,
        "txt" | "text" | "log" => FileType::Text,
        "html" | "htm" | "xhtml" => FileType::Html,
        "docx" => FileType::Docx,
        "pptx" => FileType::Pptx,
        _ => FileType::Unknown,
    }
}

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
pub fn extract(file_type: &FileType, data: &[u8]) -> Result<Extracted> {
    match file_type {
        FileType::Pdf => Ok(Extracted {
            title: None,
            sections: extract_pdf_sections(data)?,
        }),
        FileType::Markdown => {
            let text = utf8(data)?;
            // YAML front matter (Obsidian, Jekyll, OKF) is metadata, not
            // prose: its `title` is the document's, the rest is dropped.
            let (front, body) = parse_front_matter(&text);
            Ok(Extracted {
                title: front.get("title").map(str::to_owned),
                sections: markdown_sections(body),
            })
        }
        FileType::Text => Ok(Extracted {
            title: None,
            sections: vec![Section {
                heading: None,
                page: None,
                text: utf8(data)?,
            }],
        }),
        FileType::Html => super::html::html(&utf8(data)?),
        FileType::Docx => super::office::docx(data),
        FileType::Pptx => super::office::pptx(data),
        FileType::Csv | FileType::Parquet | FileType::Json | FileType::Xlsx | FileType::Unknown => {
            Err(Error::Ingestion(format!(
                "cannot extract text from {file_type} files"
            )))
        }
    }
}

/// The sections of an unstructured file; see [`extract`].
///
/// # Errors
///
/// As [`extract`].
pub fn extract_sections(file_type: &FileType, data: &[u8]) -> Result<Vec<Section>> {
    extract(file_type, data).map(|e| e.sections)
}

/// The document title a parse yields: the first section's heading when
/// the source has headings (Markdown's first `#` line, an HTML `<title>`),
/// else nothing.
#[must_use]
pub fn title_of(sections: &[Section]) -> Option<&str> {
    sections
        .first()
        .and_then(|s| s.heading.as_deref())
        .map(str::trim)
        .filter(|h| !h.is_empty())
}

/// The whole text of an unstructured file, sections joined.
///
/// # Errors
///
/// Returns an error if the file cannot be parsed.
pub fn extract_text(file_type: &FileType, data: &[u8]) -> Result<String> {
    let sections = extract_sections(file_type, data)?;
    Ok(sections
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n"))
}

fn utf8(data: &[u8]) -> Result<String> {
    String::from_utf8(data.to_vec()).map_err(|e| Error::Ingestion(format!("invalid UTF-8: {e}")))
}

fn extract_pdf_sections(data: &[u8]) -> Result<Vec<Section>> {
    let pages = pdf_extract::extract_text_from_mem_by_pages(data)
        .map_err(|e| Error::Ingestion(format!("PDF extraction failed: {e}")))?;
    let sections: Vec<Section> = pages
        .into_iter()
        .enumerate()
        .filter(|(_, text)| !text.trim().is_empty())
        .map(|(i, text)| Section {
            heading: None,
            page: u32::try_from(i.saturating_add(1)).ok(),
            text,
        })
        .collect();
    if sections.is_empty() {
        return Err(Error::Ingestion(String::from(
            "no extractable text: the PDF has no text layer (scanned pages need OCR)",
        )));
    }
    Ok(sections)
}

/// Split Markdown at ATX (`# Title`) and setext (underlined) headings. Text
/// before the first heading becomes a section without one.
fn markdown_sections(text: &str) -> Vec<Section> {
    let lines: Vec<&str> = text.lines().collect();
    let mut sections: Vec<Section> = Vec::new();
    let mut heading: Option<String> = None;
    let mut buf: Vec<&str> = Vec::new();

    let flush = |heading: &Option<String>, buf: &mut Vec<&str>, out: &mut Vec<Section>| {
        let body = buf.join("\n");
        if !body.trim().is_empty() {
            out.push(Section {
                heading: heading.clone(),
                page: None,
                text: body.trim().to_owned(),
            });
        }
        buf.clear();
    };

    let mut i = 0;
    while i < lines.len() {
        let line = lines.get(i).copied().unwrap_or_default();
        let next = lines.get(i.saturating_add(1)).copied();
        if let Some(title) = atx_heading(line) {
            flush(&heading, &mut buf, &mut sections);
            heading = Some(title);
            i = i.saturating_add(1);
            continue;
        }
        if let Some(underline) = next
            && is_setext_underline(underline)
            && !line.trim().is_empty()
            && !line.trim_start().starts_with(['-', '*', '+', '>', '|'])
        {
            flush(&heading, &mut buf, &mut sections);
            heading = Some(line.trim().to_owned());
            i = i.saturating_add(2);
            continue;
        }
        buf.push(line);
        i = i.saturating_add(1);
    }
    flush(&heading, &mut buf, &mut sections);
    sections
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
            &FileType::Markdown,
            b"---\ntitle: Renewal Guide\ntags: [a]\n---\n\n# Terms\n\nThirty days.\n",
        )
        .unwrap_or_else(|_| Extracted {
            title: None,
            sections: Vec::new(),
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
        assert_eq!(detect_file_type("sales.csv"), FileType::Csv);
        assert_eq!(detect_file_type("DATA.CSV"), FileType::Csv);
    }

    #[test]
    fn detects_parquet() {
        assert_eq!(detect_file_type("data.parquet"), FileType::Parquet);
        assert_eq!(detect_file_type("data.pq"), FileType::Parquet);
    }

    #[test]
    fn detects_json() {
        assert_eq!(detect_file_type("config.json"), FileType::Json);
        assert_eq!(detect_file_type("events.jsonl"), FileType::Json);
        assert_eq!(detect_file_type("stream.ndjson"), FileType::Json);
    }

    #[test]
    fn detects_pdf() {
        assert_eq!(detect_file_type("report.pdf"), FileType::Pdf);
    }

    #[test]
    fn detects_text() {
        assert_eq!(detect_file_type("notes.txt"), FileType::Text);
        assert_eq!(detect_file_type("readme.md"), FileType::Markdown);
    }

    #[test]
    fn unknown_extension() {
        assert_eq!(detect_file_type("image.png"), FileType::Unknown);
        assert_eq!(detect_file_type("noext"), FileType::Unknown);
    }

    #[test]
    fn extracts_plain_text() {
        let data = b"Hello, world!";
        let text = extract_text(&FileType::Text, data);
        assert!(text.is_ok());
        assert_eq!(text.ok(), Some(String::from("Hello, world!")));
    }

    #[test]
    fn rejects_structured_extraction() {
        let result = extract_text(&FileType::Csv, b"a,b,c");
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
        let err = extract_sections(&FileType::Pdf, b"%PDF-1.4\n%%EOF").err();
        assert!(err.is_some());
    }
}
