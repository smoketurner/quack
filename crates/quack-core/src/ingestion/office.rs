//! DOCX and PPTX: a zip of XML parts read with `quick-xml`. Word documents
//! become sections split at `Heading N` and `Title` paragraph styles;
//! presentations become one section per slide, headed by the slide's
//! title placeholder. The package title comes from `docProps/core.xml`.

use std::io::{Cursor, Read};

use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};

use super::parser::{Extracted, FileType, Flow, Section, SectionBuilder};
use crate::error::{Error, Result};

/// A DOCX package as sections with headings.
///
/// # Errors
///
/// Returns an error when the bytes are not a Word package or hold no text.
pub fn docx(data: &[u8]) -> Result<Extracted> {
    let mut archive = open(data, FileType::Docx)?;
    let document = part(&mut archive, "word/document.xml")?.ok_or_else(|| {
        Error::Ingestion(String::from(
            "not a Word file: word/document.xml is missing",
        ))
    })?;
    let paragraphs = word_paragraphs(&document)?;
    let core_title = part(&mut archive, "docProps/core.xml")?
        .as_deref()
        .and_then(core_title);

    let mut builder = SectionBuilder::default();
    let mut style_title: Option<String> = None;
    for paragraph in paragraphs {
        match paragraph.kind {
            ParagraphKind::Title => {
                if style_title.is_none() && !paragraph.text.trim().is_empty() {
                    style_title = Some(paragraph.text.trim().to_owned());
                }
                builder.heading(paragraph.text.trim());
            }
            ParagraphKind::Heading => builder.heading(paragraph.text.trim()),
            ParagraphKind::Body => builder.line(paragraph.text),
        }
    }
    let sections = builder.finish();
    if sections.is_empty() {
        return Err(Error::Ingestion(String::from(
            "no extractable text: the DOCX has no paragraphs",
        )));
    }
    Ok(Extracted {
        title: core_title.or(style_title),
        sections,
        flow: Flow::Sectioned,
        pages_skipped: 0,
    })
}

/// A PPTX package as one section per slide.
///
/// # Errors
///
/// Returns an error when the bytes are not a `PowerPoint` package or hold no
/// text.
pub fn pptx(data: &[u8]) -> Result<Extracted> {
    let mut archive = open(data, FileType::Pptx)?;
    let mut slide_names: Vec<(u32, String)> = Vec::new();
    for i in 0..archive.len() {
        let Ok(entry) = archive.by_index(i) else {
            continue;
        };
        let name = entry.name().to_owned();
        if let Some(number) = slide_number(&name) {
            slide_names.push((number, name));
        }
    }
    if slide_names.is_empty() {
        return Err(Error::Ingestion(String::from(
            "not a PowerPoint file: no ppt/slides/slideN.xml parts",
        )));
    }
    slide_names.sort();
    let core_title = part(&mut archive, "docProps/core.xml")?
        .as_deref()
        .and_then(core_title);

    let mut sections = Vec::new();
    for (number, name) in slide_names {
        let Some(xml) = part(&mut archive, &name)? else {
            continue;
        };
        let slide = slide_text(&xml)?;
        let text = slide.body.join("\n").trim().to_owned();
        let heading = slide.title.filter(|t| !t.is_empty());
        if text.is_empty() && heading.is_none() {
            continue;
        }
        sections.push(Section {
            heading: heading.clone(),
            page: Some(number),
            // A slide with only a title still carries it as text so the
            // slide is searchable and citable.
            text: if text.is_empty() {
                heading.unwrap_or_default()
            } else {
                text
            },
        });
    }
    if sections.is_empty() {
        return Err(Error::Ingestion(String::from(
            "no extractable text: the PPTX has no text on any slide",
        )));
    }
    let first_title = sections.first().and_then(|s| s.heading.clone());
    Ok(Extracted {
        title: core_title.or(first_title),
        sections,
        flow: Flow::Sectioned,
        pages_skipped: 0,
    })
}

/// The package's zip archive, or an error naming the type expected.
fn open(data: &[u8], file_type: FileType) -> Result<zip::ZipArchive<Cursor<&[u8]>>> {
    zip::ZipArchive::new(Cursor::new(data))
        .map_err(|e| Error::Ingestion(format!("not a {file_type} file: {e}")))
}

fn part(archive: &mut zip::ZipArchive<Cursor<&[u8]>>, name: &str) -> Result<Option<String>> {
    let mut entry = match archive.by_name(name) {
        Ok(entry) => entry,
        Err(zip::result::ZipError::FileNotFound) => return Ok(None),
        Err(e) => return Err(Error::Ingestion(format!("cannot read {name}: {e}"))),
    };
    let mut xml = String::new();
    entry
        .read_to_string(&mut xml)
        .map_err(|e| Error::Ingestion(format!("cannot read {name}: {e}")))?;
    Ok(Some(xml))
}

/// The text of an entity reference: a character reference or one of the
/// five predefined entities; anything else is dropped.
fn entity_text(reference: &quick_xml::events::BytesRef<'_>) -> Option<String> {
    if let Ok(Some(c)) = reference.resolve_char_ref() {
        return Some(c.to_string());
    }
    let name = reference.xml10_content();
    quick_xml::escape::resolve_predefined_entity(&name).map(str::to_owned)
}

fn slide_number(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("ppt/slides/slide")?;
    let digits = rest.strip_suffix(".xml")?;
    digits.parse().ok()
}

/// `dc:title` from `docProps/core.xml`, when present and non-empty.
fn core_title(xml: &str) -> Option<String> {
    let mut reader = Reader::from_str(xml);
    let mut in_title = false;
    let mut title = String::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) if e.local_name().as_ref() == "title" => in_title = true,
            Ok(Event::End(e)) if e.local_name().as_ref() == "title" => break,
            Ok(Event::Text(t)) if in_title => {
                title.push_str(&t.xml10_content());
            }
            Ok(Event::GeneralRef(r)) if in_title => {
                if let Some(text) = entity_text(&r) {
                    title.push_str(&text);
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            Ok(_) => {}
        }
    }
    let title = title.trim();
    (!title.is_empty()).then(|| title.to_owned())
}

#[derive(Debug, PartialEq, Eq)]
enum ParagraphKind {
    Title,
    Heading,
    Body,
}

impl ParagraphKind {
    /// The kind a `w:pStyle`'s `w:val` names: `Title` and
    /// `Heading1`..`Heading9` (the localized style ids Word writes in some
    /// locales start differently, so only the English ids and `Title` are
    /// recognized).
    fn of_style(e: &BytesStart<'_>) -> Self {
        let value = e
            .try_get_attribute("w:val")
            .ok()
            .flatten()
            .map(|a| a.value.into_owned())
            .unwrap_or_default();
        let lower = value.to_ascii_lowercase();
        if lower == "title" {
            Self::Title
        } else if lower.starts_with("heading") {
            Self::Heading
        } else {
            Self::Body
        }
    }
}

#[derive(Debug)]
struct Paragraph {
    kind: ParagraphKind,
    text: String,
}

/// Paragraphs of `word/document.xml` in order, each with its style class.
/// Table cells are paragraphs too, so tables read as their text.
fn word_paragraphs(xml: &str) -> Result<Vec<Paragraph>> {
    let mut reader = Reader::from_str(xml);
    let mut out = Vec::new();
    let mut current: Option<Paragraph> = None;
    let mut buf = Vec::new();
    loop {
        let event = reader
            .read_event_into(&mut buf)
            .map_err(|e| Error::Ingestion(format!("DOCX XML error: {e}")))?;
        match event {
            Event::Start(e) => match e.local_name().as_ref() {
                "p" => {
                    current = Some(Paragraph {
                        kind: ParagraphKind::Body,
                        text: String::new(),
                    });
                }
                "pStyle" => {
                    if let Some(p) = current.as_mut() {
                        p.kind = ParagraphKind::of_style(&e);
                    }
                }
                _ => {}
            },
            Event::Empty(e) => match e.local_name().as_ref() {
                "pStyle" => {
                    if let Some(p) = current.as_mut() {
                        p.kind = ParagraphKind::of_style(&e);
                    }
                }
                "tab" => {
                    if let Some(p) = current.as_mut() {
                        p.text.push(' ');
                    }
                }
                "br" | "cr" => {
                    if let Some(p) = current.as_mut() {
                        p.text.push('\n');
                    }
                }
                _ => {}
            },
            Event::Text(t) => {
                if let Some(p) = current.as_mut() {
                    p.text.push_str(&t.xml10_content());
                }
            }
            Event::GeneralRef(r) => {
                if let Some(p) = current.as_mut()
                    && let Some(text) = entity_text(&r)
                {
                    p.text.push_str(&text);
                }
            }
            Event::End(e) => {
                if e.local_name().as_ref() == "p"
                    && let Some(p) = current.take()
                    && !p.text.trim().is_empty()
                {
                    out.push(p);
                }
            }
            Event::Eof => break,
            Event::CData(_)
            | Event::Comment(_)
            | Event::Decl(_)
            | Event::PI(_)
            | Event::DocType(_) => {}
        }
        buf.clear();
    }
    Ok(out)
}

#[derive(Debug, Default)]
struct SlideText {
    title: Option<String>,
    body: Vec<String>,
}

/// Text of one slide: the title placeholder apart, the rest as paragraphs.
fn slide_text(xml: &str) -> Result<SlideText> {
    let mut reader = Reader::from_str(xml);
    let mut out = SlideText::default();
    let mut buf = Vec::new();
    let mut in_title_shape = false;
    let mut shape_depth: u32 = 0;
    let mut paragraph = String::new();
    let mut in_paragraph = false;
    loop {
        let event = reader
            .read_event_into(&mut buf)
            .map_err(|e| Error::Ingestion(format!("PPTX XML error: {e}")))?;
        match event {
            Event::Start(e) => match e.local_name().as_ref() {
                "sp" => {
                    shape_depth = shape_depth.saturating_add(1);
                    in_title_shape = false;
                }
                "p" => {
                    in_paragraph = true;
                    paragraph.clear();
                }
                _ => {}
            },
            Event::Empty(e) => {
                if e.local_name().as_ref() == "ph" && shape_depth > 0 {
                    let kind = e
                        .try_get_attribute("type")
                        .ok()
                        .flatten()
                        .map(|a| a.value.into_owned())
                        .unwrap_or_default();
                    if kind == "title" || kind == "ctrTitle" {
                        in_title_shape = true;
                    }
                }
            }
            Event::Text(t) => {
                if in_paragraph {
                    paragraph.push_str(&t.xml10_content());
                }
            }
            Event::GeneralRef(r) => {
                if in_paragraph && let Some(text) = entity_text(&r) {
                    paragraph.push_str(&text);
                }
            }
            Event::End(e) => match e.local_name().as_ref() {
                "p" => {
                    in_paragraph = false;
                    let line = paragraph.trim().to_owned();
                    if !line.is_empty() {
                        if in_title_shape {
                            match out.title.as_mut() {
                                Some(t) => {
                                    t.push(' ');
                                    t.push_str(&line);
                                }
                                None => out.title = Some(line),
                            }
                        } else {
                            out.body.push(line);
                        }
                    }
                }
                "sp" => {
                    shape_depth = shape_depth.saturating_sub(1);
                    in_title_shape = false;
                }
                _ => {}
            },
            Event::Eof => break,
            Event::CData(_)
            | Event::Comment(_)
            | Event::Decl(_)
            | Event::PI(_)
            | Event::DocType(_) => {}
        }
        buf.clear();
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::io::Write;

    use super::*;

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

    const DOCUMENT: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Renewal Playbook</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Exclusions</w:t></w:r></w:p>
<w:p><w:r><w:t xml:space="preserve">Flood is </w:t></w:r><w:r><w:t>excluded.</w:t></w:r></w:p>
<w:p><w:r><w:t>Tab</w:t><w:tab/><w:t>separated</w:t><w:br/><w:t>and broken</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Claims</w:t></w:r></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cell A</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Cell B &amp; C</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
<w:p><w:r><w:t>   </w:t></w:r></w:p>
</w:body></w:document>"#;

    const CORE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Core Title</dc:title><dc:creator>x</dc:creator></cp:coreProperties>"#;

    #[test]
    fn docx_splits_at_heading_styles_and_reads_core_title() {
        let bytes = package(&[("word/document.xml", DOCUMENT), ("docProps/core.xml", CORE)]);
        let extracted = docx(&bytes).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(extracted.title.as_deref(), Some("Core Title"));
        let headings: Vec<Option<&str>> = extracted
            .sections
            .iter()
            .map(|s| s.heading.as_deref())
            .collect();
        assert_eq!(headings, [Some("Exclusions"), Some("Claims")]);
        assert_eq!(
            extracted.sections.first().map(|s| s.text.as_str()),
            Some("Flood is excluded.\nTab separated\nand broken")
        );
        assert_eq!(
            extracted.sections.get(1).map(|s| s.text.as_str()),
            Some("Cell A\nCell B & C")
        );
    }

    #[test]
    fn docx_title_style_is_the_title_when_core_has_none() {
        let bytes = package(&[("word/document.xml", DOCUMENT)]);
        let extracted = docx(&bytes).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(extracted.title.as_deref(), Some("Renewal Playbook"));
    }

    #[test]
    fn docx_errors_are_specific() {
        assert!(docx(b"not a zip").is_err_and(|e| e.to_string().contains("not a Word file")));
        let no_part = package(&[("other.xml", "<a/>")]);
        assert!(docx(&no_part).is_err_and(|e| e.to_string().contains("word/document.xml")));
        let empty = package(&[(
            "word/document.xml",
            r#"<w:document xmlns:w="x"><w:body/></w:document>"#,
        )]);
        assert!(docx(&empty).is_err_and(|e| e.to_string().contains("no extractable text")));
    }

    fn slide(title: Option<&str>, lines: &[&str]) -> String {
        let mut xml = String::from(
            r#"<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree>"#,
        );
        if let Some(t) = title {
            xml.push_str(
                r#"<p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>"#,
            );
            xml.push_str(t);
            xml.push_str("</a:t></a:r></a:p></p:txBody></p:sp>");
        }
        xml.push_str("<p:sp><p:txBody>");
        for line in lines {
            xml.push_str("<a:p><a:r><a:t>");
            xml.push_str(line);
            xml.push_str("</a:t></a:r></a:p>");
        }
        xml.push_str("</p:txBody></p:sp></p:spTree></p:cSld></p:sld>");
        xml
    }

    #[test]
    fn pptx_is_one_section_per_slide_in_numeric_order() {
        let s1 = slide(Some("Agenda"), &["Renewals &amp; more", "Claims"]);
        let s2 = slide(None, &["No title here"]);
        let s10 = slide(Some("Only a title"), &[]);
        let bytes = package(&[
            ("ppt/slides/slide10.xml", &s10),
            ("ppt/slides/slide2.xml", &s2),
            ("ppt/slides/slide1.xml", &s1),
            ("ppt/slides/_rels/slide1.xml.rels", "<x/>"),
        ]);
        let extracted = pptx(&bytes).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(extracted.title.as_deref(), Some("Agenda"));
        let pages: Vec<Option<u32>> = extracted.sections.iter().map(|s| s.page).collect();
        assert_eq!(pages, [Some(1), Some(2), Some(10)]);
        assert_eq!(
            extracted
                .sections
                .first()
                .map(|s| (s.heading.as_deref(), s.text.as_str())),
            Some((Some("Agenda"), "Renewals & more\nClaims"))
        );
        assert_eq!(
            extracted.sections.get(1).map(|s| s.heading.as_deref()),
            Some(None)
        );
        assert_eq!(
            extracted.sections.get(2).map(|s| s.text.as_str()),
            Some("Only a title")
        );
    }

    #[test]
    fn pptx_without_slides_or_text_is_an_error() {
        assert!(pptx(b"zip? no").is_err());
        let none = package(&[("ppt/presentation.xml", "<p/>")]);
        assert!(pptx(&none).is_err_and(|e| e.to_string().contains("no ppt/slides")));
        let blank = slide(None, &["   "]);
        let empty = package(&[("ppt/slides/slide1.xml", &blank)]);
        assert!(pptx(&empty).is_err_and(|e| e.to_string().contains("no extractable text")));
    }

    #[test]
    fn core_title_ignores_blank_and_missing() {
        assert_eq!(core_title(CORE).as_deref(), Some("Core Title"));
        assert_eq!(
            core_title("<cp:coreProperties><dc:title>  </dc:title></cp:coreProperties>"),
            None
        );
        assert_eq!(core_title("<x/>"), None);
    }
}
