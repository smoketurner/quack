//! A zip package of XML parts (EPUB, ODT), read part by part against the
//! decompression budget, with the small XML reading the two formats
//! share: an element's text, its attributes, a document's structure in
//! events.

use std::io::{Cursor, Read};

use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};

use super::budget::DecompressionBudget;
use crate::error::{Error, Result};

/// A zip package: its archive, and the bytes its parts may still inflate
/// to.
pub(crate) struct Package<'a> {
    archive: zip::ZipArchive<Cursor<&'a [u8]>>,
    budget: DecompressionBudget,
}

impl<'a> Package<'a> {
    /// The package, or an error naming the kind expected.
    pub(crate) fn open(data: &'a [u8], kind: &str, budget: DecompressionBudget) -> Result<Self> {
        let archive = zip::ZipArchive::new(Cursor::new(data))
            .map_err(|e| Error::Ingestion(format!("not {kind}: {e}")))?;
        Ok(Self { archive, budget })
    }

    /// One part's text, `None` when the package has no such part.
    pub(crate) fn part(&mut self, name: &str) -> Result<Option<String>> {
        let entry = match self.archive.by_name(name) {
            Ok(entry) => entry,
            Err(zip::result::ZipError::FileNotFound) => return Ok(None),
            Err(e) => return Err(Error::Ingestion(format!("cannot read {name}: {e}"))),
        };
        let mut text = String::new();
        let read = self.budget.reader(entry).read_to_string(&mut text);
        read.map_err(|e| self.budget.read_error(name, &e))?;
        Ok(Some(text))
    }

    /// A part that must exist.
    pub(crate) fn required(&mut self, name: &str, kind: &str) -> Result<String> {
        self.part(name)?
            .ok_or_else(|| Error::Ingestion(format!("not {kind}: {name} is missing")))
    }
}

/// An attribute's value, by local name, in any namespace prefix.
pub(crate) fn attribute(element: &BytesStart<'_>, name: &str) -> Option<String> {
    element.attributes().flatten().find_map(|a| {
        (a.key.local_name().as_ref() == name).then(|| {
            a.normalized_value(quick_xml::XmlVersion::Implicit1_0)
                .ok()
                .map(std::borrow::Cow::into_owned)
        })?
    })
}

/// The text of an entity reference: a character reference or one of the
/// five predefined entities; anything else is dropped.
pub(crate) fn entity_text(reference: &quick_xml::events::BytesRef<'_>) -> Option<String> {
    if let Ok(Some(c)) = reference.resolve_char_ref() {
        return Some(c.to_string());
    }
    let name = reference.xml10_content();
    quick_xml::escape::resolve_predefined_entity(&name).map(str::to_owned)
}

/// The first `element`'s text content in `xml`, by local name.
pub(crate) fn element_text(xml: &str, element: &str) -> Option<String> {
    let mut reader = Reader::from_str(xml);
    let mut inside = false;
    let mut text = String::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) if e.local_name().as_ref() == element => inside = true,
            Ok(Event::End(e)) if inside && e.local_name().as_ref() == element => break,
            Ok(Event::Text(t)) if inside => text.push_str(&t.xml10_content()),
            Ok(Event::GeneralRef(r)) if inside => {
                if let Some(t) = entity_text(&r) {
                    text.push_str(&t);
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            Ok(_) => {}
        }
    }
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

/// Every `element`'s text content in `xml`, by local name, in order.
pub(crate) fn element_texts(xml: &str, element: &str) -> Vec<String> {
    let mut reader = Reader::from_str(xml);
    let mut inside = false;
    let mut text = String::new();
    let mut out = Vec::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) if e.local_name().as_ref() == element => {
                inside = true;
                text.clear();
            }
            Ok(Event::End(e)) if inside && e.local_name().as_ref() == element => {
                inside = false;
                let t = text.trim();
                if !t.is_empty() {
                    out.push(t.to_owned());
                }
            }
            Ok(Event::Text(t)) if inside => text.push_str(&t.xml10_content()),
            Ok(Event::GeneralRef(r)) if inside => {
                if let Some(t) = entity_text(&r) {
                    text.push_str(&t);
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            Ok(_) => {}
        }
    }
    out
}

/// The attribute values of every `element` in `xml`, by local names.
pub(crate) fn element_attributes(
    xml: &str,
    element: &str,
    names: &[&str],
) -> Vec<Vec<Option<String>>> {
    let mut reader = Reader::from_str(xml);
    let mut out = Vec::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(e) | Event::Empty(e)) if e.local_name().as_ref() == element => {
                out.push(names.iter().map(|n| attribute(&e, n)).collect());
            }
            Ok(Event::Eof) | Err(_) => break,
            Ok(_) => {}
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use std::io::Write as _;

    use super::*;

    /// A zip with the given parts, for the EPUB and ODT tests.
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

    #[test]
    fn xml_helpers_read_text_and_attributes_by_local_name() {
        let xml = r#"<r xmlns:dc="x"><dc:title>A &amp; B</dc:title><dc:subject>s1</dc:subject><dc:subject>s2</dc:subject><item id="i1" href="a.xhtml"/></r>"#;
        assert_eq!(element_text(xml, "title").as_deref(), Some("A & B"));
        assert_eq!(element_texts(xml, "subject"), ["s1", "s2"]);
        assert_eq!(
            element_attributes(xml, "item", &["id", "href", "none"]),
            vec![vec![
                Some(String::from("i1")),
                Some(String::from("a.xhtml")),
                None
            ]]
        );
        assert_eq!(element_text(xml, "missing"), None);
    }

    #[test]
    fn a_package_reads_parts_and_names_a_missing_one() {
        let bytes = package(&[("a.xml", "<a/>")]);
        let mut p = Package::open(&bytes, "a test package", DecompressionBudget::megabytes(1))
            .unwrap_or_else(|_| unreachable_package());
        assert_eq!(p.part("a.xml").unwrap_or_default().as_deref(), Some("<a/>"));
        assert_eq!(p.part("b.xml").unwrap_or_default(), None);
        let missing = p
            .required("b.xml", "a test package")
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(missing.contains("b.xml is missing"), "{missing}");
        assert!(
            Package::open(b"nope", "a test package", DecompressionBudget::megabytes(1)).is_err()
        );
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn unreachable_package() -> ! {
        panic!("the test package did not open")
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }
}
