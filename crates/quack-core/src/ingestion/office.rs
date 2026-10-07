//! DOCX and PPTX through `office_oxide`'s document model. Word documents
//! become sections split at headings resolved from outline levels (any
//! style name, any locale), with tables, footnotes, endnotes, and comments
//! as sections of their own kinds; presentations become one section per
//! slide, headed by the slide's title, with speaker notes as note
//! sections. The core properties give the title, author, dates, and
//! keywords.

use std::io::Cursor;

use office_oxide::Document;
use office_oxide::format::DocumentFormat;
use office_oxide::ir::{DocumentIR, Element, InlineContent, Note, Table as IrTable};

use super::budget::DecompressionBudget;
use super::parser::{DocumentMeta, Extracted, FileType, Flow, Section, SectionBuilder};
use super::table::Table;
use crate::error::{Error, Result};

/// A DOCX package as sections with headings, tables, and notes.
///
/// # Errors
///
/// Returns an error when the bytes are not a Word package, hold no text,
/// or inflate past `budget`.
pub fn docx(data: &[u8], budget: DecompressionBudget) -> Result<Extracted> {
    let package = Package::open(data, FileType::Docx, DocumentFormat::Docx, budget)?;
    let mut walker = Walker::default();
    for section in &package.0.sections {
        for element in &section.elements {
            walker.walk(element);
        }
    }
    let sections = walker.finish();
    if sections.is_empty() {
        return Err(Error::Ingestion(String::from(
            "no extractable text: the DOCX has no paragraphs",
        )));
    }
    let (meta, title) = package.meta();
    Ok(Extracted {
        title,
        sections,
        flow: Flow::Sectioned,
        pages: None,
        meta,
        blank_pages: Vec::new(),
    })
}

/// A PPTX package as one section per slide, its notes apart.
///
/// # Errors
///
/// Returns an error when the bytes are not a `PowerPoint` package, hold no
/// text, or inflate past `budget`.
pub fn pptx(data: &[u8], budget: DecompressionBudget) -> Result<Extracted> {
    let package = Package::open(data, FileType::Pptx, DocumentFormat::Pptx, budget)?;
    let mut sections = Vec::new();
    for (index, slide) in package.0.sections.iter().enumerate() {
        let number = u32::try_from(index.saturating_add(1)).unwrap_or(u32::MAX);
        let heading = slide
            .title
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_owned);
        let mut walker = Walker::default();
        if let Some(h) = &heading {
            walker.0.heading(h.clone());
        }
        for element in &slide.elements {
            walker.walk(element);
        }
        let mut body: Vec<Section> = walker
            .finish()
            .into_iter()
            // A slide's own heading is the title; its body headings read as
            // lines of the slide.
            .map(|s| Section {
                heading: heading.clone(),
                ..s
            })
            .collect();
        // A slide with only a title still carries it as text so the slide
        // is searchable and citable.
        if body.is_empty()
            && let Some(h) = &heading
        {
            body.push(Section::body(heading.clone(), h.clone()));
        }
        if let Some(notes) = slide
            .speaker_notes
            .as_deref()
            .map(str::trim)
            .filter(|n| !n.is_empty())
        {
            body.push(Section::note(heading.clone(), notes));
        }
        for section in body {
            sections.push(section.on_page(Some(number)));
        }
    }
    if sections.is_empty() {
        return Err(Error::Ingestion(String::from(
            "no extractable text: the PPTX has no text on any slide",
        )));
    }
    let first_title = sections.first().and_then(|s| s.heading.clone());
    let (meta, title) = package.meta();
    Ok(Extracted {
        title: title.or(first_title),
        sections,
        flow: Flow::Sectioned,
        pages: None,
        meta,
        blank_pages: Vec::new(),
    })
}

/// A Word or `PowerPoint` package's document model.
struct Package(DocumentIR);

impl Package {
    /// `office_oxide` inflates the zip itself, so the entries are inflated
    /// against `budget` first.
    fn open(
        data: &[u8],
        file_type: FileType,
        format: DocumentFormat,
        mut budget: DecompressionBudget,
    ) -> Result<Self> {
        budget.admit_zip(data)?;
        let document = Document::from_reader(Cursor::new(data.to_vec()), format)
            .map_err(|e| Error::Ingestion(format!("not a {file_type} file: {e}")))?;
        Ok(Self(document.to_ir()))
    }

    /// The core properties as metadata, the title apart.
    fn meta(&self) -> (DocumentMeta, Option<String>) {
        let metadata = &self.0.metadata;
        let mut meta = DocumentMeta::default();
        DocumentMeta::set(&mut meta.author, metadata.author.as_deref());
        DocumentMeta::set(&mut meta.authored_at, metadata.created.as_deref());
        DocumentMeta::set(&mut meta.modified_at, metadata.modified.as_deref());
        meta.extra("subject", metadata.subject.as_deref());
        meta.extra("description", metadata.description.as_deref());
        for keyword in &metadata.keywords {
            meta.tag(keyword);
        }
        let title = metadata
            .title
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_owned);
        (meta, title)
    }
}

/// Block elements into sections: headings start sections, tables and
/// notes are sections of their kind, the rest is lines.
#[derive(Default)]
struct Walker(SectionBuilder);

impl Walker {
    fn walk(&mut self, element: &Element) {
        match element {
            Element::Heading(heading) => self.0.heading(Inline(&heading.content).text()),
            Element::Paragraph(paragraph) => {
                let text = Inline(&paragraph.content).text();
                if !text.trim().is_empty() {
                    self.0.line(text);
                }
            }
            Element::List(list) => {
                for item in &list.items {
                    for element in &item.content {
                        self.walk(element);
                    }
                    if let Some(nested) = &item.nested {
                        self.walk(&Element::List(nested.clone()));
                    }
                }
            }
            Element::CodeBlock(code) => {
                for line in code.content.lines() {
                    self.0.line(line);
                }
            }
            Element::Table(table) => {
                if let Some(found) = Table::from_office(table) {
                    self.0.push(Section::table(None, found.render()));
                }
            }
            Element::Footnote(note) => self.note("Footnote", note),
            Element::Endnote(note) => self.note("Endnote", note),
            Element::TextBox(text_box) => {
                for element in &text_box.content {
                    self.walk(element);
                }
            }
            // Images, breaks, shapes, and any variant a newer library adds
            // (`Element` is non-exhaustive) read as nothing.
            _ => {}
        }
    }

    /// A footnote, endnote, or comment (an endnote whose marker is the
    /// comment's author) as a note section: `Footnote 3: ...`.
    fn note(&mut self, kind: &str, note: &Note) {
        let text = Self::text_of(&note.content);
        if text.trim().is_empty() {
            return;
        }
        let label = match &note.marker {
            Some(author) if kind == "Endnote" => format!("Comment by {}", author.trim()),
            Some(marker) => format!("{kind} {}", marker.trim()),
            None => format!("{kind} {}", note.id),
        };
        self.0
            .push(Section::note(None, format!("{label}: {}", text.trim())));
    }

    fn finish(self) -> Vec<Section> {
        self.0.finish()
    }

    /// Block elements as lines of text.
    fn text_of(elements: &[Element]) -> String {
        let mut walker = Self::default();
        for element in elements {
            walker.walk(element);
        }
        walker
            .finish()
            .into_iter()
            .map(|s| match s.heading {
                Some(h) => format!("{h}\n{}", s.text),
                None => s.text,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Table {
    /// An `office_oxide` table's cells as text.
    fn from_office(table: &IrTable) -> Option<Self> {
        Self::from_rows(
            table
                .rows
                .iter()
                .map(|row| {
                    row.cells
                        .iter()
                        .map(|cell| Walker::text_of(&cell.content))
                        .collect()
                })
                .collect(),
        )
    }
}

/// Inline content: spans joined, line breaks as newlines, note references
/// as their markers.
struct Inline<'a>(&'a [InlineContent]);

impl Inline<'_> {
    fn text(&self) -> String {
        let mut out = String::new();
        for inline in self.0 {
            match inline {
                InlineContent::Text(span) => out.push_str(&span.text.replace('\t', " ")),
                InlineContent::LineBreak => out.push('\n'),
                InlineContent::FootnoteRef(r) | InlineContent::EndnoteRef(r) => {
                    out.push('[');
                    match &r.marker {
                        Some(marker) => out.push_str(marker),
                        None => out.push_str(&r.note_id.to_string()),
                    }
                    out.push(']');
                }
                _ => {}
            }
        }
        out
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::io::Write;

    use super::*;

    const BUDGET: DecompressionBudget = DecompressionBudget::megabytes(64);

    /// A minimal Office package with the given parts.
    pub(crate) fn package(parts: &[(&str, &str)]) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = zip::ZipWriter::new(&mut cursor);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            for (name, content) in parts {
                writer
                    .start_file(*name, options)
                    .unwrap_or_else(|e| fail(&e.to_string()));
                writer
                    .write_all(content.as_bytes())
                    .unwrap_or_else(|e| fail(&e.to_string()));
            }
            writer.finish().unwrap_or_else(|e| fail(&e.to_string()));
        }
        cursor.into_inner()
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    const CONTENT_TYPES: &str = r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>"#;
    const RELS: &str = r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>"#;
    const DOCUMENT_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#;
    const STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style><w:style w:type="paragraph" w:styleId="Rubrik"><w:name w:val="Rubrik"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style></w:styles>"#;
    const DOCUMENT: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Renewal Playbook</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Exclusions</w:t></w:r></w:p>
<w:p><w:r><w:t xml:space="preserve">Flood is </w:t></w:r><w:r><w:t>excluded.</w:t></w:r></w:p>
<w:p><w:r><w:t>Tab</w:t><w:tab/><w:t>separated</w:t><w:br/><w:t>and broken</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Rubrik"/></w:pPr><w:r><w:t>Claims</w:t></w:r></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Peril</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Limit &amp; Co</w:t></w:r></w:p></w:tc></w:tr><w:tr><w:tc><w:p><w:r><w:t>Fire</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>1,000</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
<w:p><w:r><w:t>Close in 30 days.</w:t></w:r></w:p>
<w:p><w:r><w:t>   </w:t></w:r></w:p>
</w:body></w:document>"#;

    const CORE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>Core Title</dc:title><dc:creator>Ada</dc:creator><cp:keywords>policy, renewal</cp:keywords><dcterms:created xsi:type="dcterms:W3CDTF">2026-01-05T14:30:00Z</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">2026-02-01T09:00:00Z</dcterms:modified></cp:coreProperties>"#;

    fn docx_package(core: bool) -> Vec<u8> {
        let mut parts = vec![
            ("[Content_Types].xml", CONTENT_TYPES),
            ("_rels/.rels", RELS),
            ("word/_rels/document.xml.rels", DOCUMENT_RELS),
            ("word/styles.xml", STYLES),
            ("word/document.xml", DOCUMENT),
        ];
        if core {
            parts.push(("docProps/core.xml", CORE));
        }
        package(&parts)
    }

    #[test]
    fn docx_splits_at_outline_headings_with_tables_apart_and_reads_core_properties() {
        let extracted = docx(&docx_package(true), BUDGET).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(extracted.title.as_deref(), Some("Core Title"));
        assert_eq!(extracted.meta.author.as_deref(), Some("Ada"));
        assert_eq!(
            extracted.meta.authored_at.as_deref(),
            Some("2026-01-05T14:30:00Z")
        );
        assert_eq!(extracted.meta.tags, ["policy", "renewal"]);
        let summary: Vec<(Option<&str>, &str, &str)> = extracted
            .sections
            .iter()
            .map(|s| (s.heading.as_deref(), s.kind.as_str(), s.text.as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![
                (None, "body", "Renewal Playbook"),
                (
                    Some("Exclusions"),
                    "body",
                    "Flood is excluded.\nTab separated\nand broken"
                ),
                (
                    Some("Claims"),
                    "table",
                    "| Peril | Limit & Co |\n| --- | --- |\n| Fire | 1,000 |"
                ),
                (Some("Claims"), "body", "Close in 30 days."),
            ]
        );
    }

    #[test]
    fn the_first_heading_is_the_title_when_core_has_none() {
        let extracted = docx(&docx_package(false), BUDGET).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(extracted.title(), Some("Exclusions"));
        assert!(extracted.meta.is_empty());
    }

    #[test]
    fn docx_without_text_or_not_a_package_is_an_error() {
        let empty = package(&[
            ("[Content_Types].xml", CONTENT_TYPES),
            ("_rels/.rels", RELS),
            (
                "word/document.xml",
                r#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body/></w:document>"#,
            ),
        ]);
        assert!(docx(&empty, BUDGET).is_err());
        assert!(docx(b"not a zip", BUDGET).is_err());
    }

    #[test]
    fn docx_over_the_decompression_budget_is_refused() {
        let bytes = docx_package(false);
        let refused = docx(&bytes, DecompressionBudget::megabytes(0))
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(refused.contains("max_decompressed_mb"), "{refused}");
    }

    const PPTX_TYPES: &str = r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/><Override PartName="/ppt/slides/slide2.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/><Override PartName="/ppt/notesSlides/notesSlide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/></Types>"#;
    const PPTX_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>"#;
    const PRESENTATION: &str = r#"<?xml version="1.0" encoding="UTF-8"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst><p:sldId id="256" r:id="rId2"/><p:sldId id="257" r:id="rId3"/></p:sldIdLst></p:presentation>"#;
    const PRESENTATION_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide2.xml"/></Relationships>"#;
    const SLIDE1: &str = r#"<?xml version="1.0" encoding="UTF-8"?><p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Renewals</a:t></a:r></a:p></p:txBody></p:sp><p:sp><p:nvSpPr><p:cNvPr id="3" name="Body"/><p:cNvSpPr/><p:nvPr><p:ph type="body"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Thirty days.</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"#;
    const SLIDE1_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/></Relationships>"#;
    const NOTES1: &str = r#"<?xml version="1.0" encoding="UTF-8"?><p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:nvSpPr><p:cNvPr id="2" name="Notes"/><p:cNvSpPr/><p:nvPr><p:ph type="body"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Mention the grace period.</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>"#;
    const SLIDE2: &str = r#"<?xml version="1.0" encoding="UTF-8"?><p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Title only</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"#;

    #[test]
    fn pptx_is_one_section_per_slide_with_notes_apart() {
        let bytes = package(&[
            ("[Content_Types].xml", PPTX_TYPES),
            ("_rels/.rels", PPTX_RELS),
            ("ppt/presentation.xml", PRESENTATION),
            ("ppt/_rels/presentation.xml.rels", PRESENTATION_RELS),
            ("ppt/slides/slide1.xml", SLIDE1),
            ("ppt/slides/_rels/slide1.xml.rels", SLIDE1_RELS),
            ("ppt/notesSlides/notesSlide1.xml", NOTES1),
            ("ppt/slides/slide2.xml", SLIDE2),
        ]);
        let extracted = pptx(&bytes, BUDGET).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(extracted.title.as_deref(), Some("Renewals"));
        let summary: Vec<(Option<&str>, &str, Option<u32>, &str)> = extracted
            .sections
            .iter()
            .map(|s| {
                (
                    s.heading.as_deref(),
                    s.kind.as_str(),
                    s.page,
                    s.text.as_str(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            vec![
                (Some("Renewals"), "body", Some(1), "Thirty days."),
                (
                    Some("Renewals"),
                    "note",
                    Some(1),
                    "Mention the grace period."
                ),
                (Some("Title only"), "body", Some(2), "Title only"),
            ]
        );
    }

    #[test]
    fn pptx_without_slides_is_an_error() {
        let bytes = package(&[
            ("[Content_Types].xml", PPTX_TYPES),
            ("_rels/.rels", PPTX_RELS),
            (
                "ppt/presentation.xml",
                r#"<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>"#,
            ),
        ]);
        assert!(pptx(&bytes, BUDGET).is_err());
    }
}
