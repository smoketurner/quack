//! EPUB: the package's spine, each XHTML item read as HTML, one run of
//! sections per chapter with `chapter N` as the locator; the title,
//! creator, date, and subjects from the OPF's Dublin Core elements.

use super::budget::DecompressionBudget;
use super::html;
use super::parser::{DocumentMeta, Extracted, Flow, Section};
use super::zipped::{Package, Xml};
use crate::error::{Error, Result};

const KIND: &str = "an EPUB";

/// An EPUB as sections, chapter by chapter.
///
/// # Errors
///
/// Returns an error when the bytes are not an EPUB, no chapter has text,
/// or the package inflates past `budget`.
pub fn extract(data: &[u8], budget: DecompressionBudget) -> Result<Extracted> {
    let mut package = Package::open(data, KIND, budget)?;
    let container = package.required("META-INF/container.xml", KIND)?;
    let opf_path = Xml(&container)
        .attributes("rootfile", &["full-path"])
        .into_iter()
        .find_map(|attrs| attrs.into_iter().next().flatten())
        .ok_or_else(|| Error::Ingestion(format!("not {KIND}: container.xml names no rootfile")))?;
    let opf = package.required(&opf_path, KIND)?;
    let opf_dir = opf_path.rsplit_once('/').map(|(dir, _)| dir);
    let manifest: Vec<(String, String)> = Xml(&opf)
        .attributes("item", &["id", "href"])
        .into_iter()
        .filter_map(|attrs| {
            let mut attrs = attrs.into_iter();
            Some((attrs.next()??, attrs.next()??))
        })
        .collect();
    let spine: Vec<String> = Xml(&opf)
        .attributes("itemref", &["idref"])
        .into_iter()
        .filter_map(|attrs| attrs.into_iter().next().flatten())
        .collect();

    let mut meta = DocumentMeta::default();
    DocumentMeta::set(&mut meta.author, Xml(&opf).text("creator").as_deref());
    DocumentMeta::set(&mut meta.authored_at, Xml(&opf).text("date").as_deref());
    meta.extra("description", Xml(&opf).text("description").as_deref());
    meta.extra("publisher", Xml(&opf).text("publisher").as_deref());
    for subject in Xml(&opf).texts("subject") {
        meta.tag(&subject);
    }
    let title = Xml(&opf).text("title");

    let mut sections = Vec::new();
    let mut chapter = 0u32;
    for idref in &spine {
        let Some((_, href)) = manifest.iter().find(|(id, _)| id == idref) else {
            continue;
        };
        let path = match opf_dir {
            Some(dir) => format!("{dir}/{href}"),
            None => href.clone(),
        };
        let Some(xhtml) = package.part(&path)? else {
            tracing::warn!(item = %path, "EPUB spine names a missing item; skipping it");
            continue;
        };
        chapter = chapter.saturating_add(1);
        let Ok(extracted) = html::html(&xhtml) else {
            continue;
        };
        let fallback = extracted.title.clone();
        for section in extracted.sections {
            let heading = section.heading.clone().or_else(|| fallback.clone());
            sections.push(Section { heading, ..section }.at(format!("chapter {chapter}")));
        }
    }
    if sections.is_empty() {
        return Err(Error::Ingestion(String::from(
            "no extractable text: no chapter of the EPUB has text",
        )));
    }
    Ok(Extracted {
        title,
        sections,
        flow: Flow::Sectioned,
        pages: None,
        meta,
    })
}

#[cfg(test)]
mod tests {
    use super::super::zipped::tests::package;
    use super::*;

    const CONTAINER: &str = r#"<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"#;
    const OPF: &str = r#"<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata><dc:title>Handbook</dc:title><dc:creator>Ada</dc:creator><dc:date>2026-01-05</dc:date><dc:subject>policy</dc:subject></metadata><manifest><item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/><item id="c2" href="ch2.xhtml" media-type="application/xhtml+xml"/><item id="gone" href="nope.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c1"/><itemref idref="gone"/><itemref idref="c2"/></spine></package>"#;

    #[test]
    fn chapters_follow_the_spine_with_their_locators_and_the_opf_metadata() {
        let bytes = package(&[
            ("mimetype", "application/epub+zip"),
            ("META-INF/container.xml", CONTAINER),
            ("OEBPS/content.opf", OPF),
            (
                "OEBPS/ch1.xhtml",
                "<html><head><title>One</title></head><body><h1>Exclusions</h1><p>Flood is excluded.</p></body></html>",
            ),
            (
                "OEBPS/ch2.xhtml",
                "<html><head><title>Two</title></head><body><p>No heading here.</p></body></html>",
            ),
        ]);
        let extracted = extract(&bytes, DecompressionBudget::megabytes(1)).unwrap_or_default();
        assert_eq!(extracted.title.as_deref(), Some("Handbook"));
        assert_eq!(extracted.meta.author.as_deref(), Some("Ada"));
        assert_eq!(extracted.meta.authored_at.as_deref(), Some("2026-01-05"));
        assert_eq!(extracted.meta.tags, ["policy"]);
        let summary: Vec<(Option<&str>, Option<&str>, &str)> = extracted
            .sections
            .iter()
            .map(|s| (s.heading.as_deref(), s.locator.as_deref(), s.text.as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![
                (Some("Exclusions"), Some("chapter 1"), "Flood is excluded."),
                (Some("Two"), Some("chapter 2"), "No heading here."),
            ]
        );
    }

    #[test]
    fn a_package_without_a_rootfile_or_text_is_refused() {
        let no_root = package(&[("META-INF/container.xml", "<container/>")]);
        assert!(extract(&no_root, DecompressionBudget::megabytes(1)).is_err());
        let empty = package(&[
            ("META-INF/container.xml", CONTAINER),
            ("OEBPS/content.opf", OPF),
            ("OEBPS/ch1.xhtml", "<html><body></body></html>"),
        ]);
        assert!(extract(&empty, DecompressionBudget::megabytes(1)).is_err());
    }
}
