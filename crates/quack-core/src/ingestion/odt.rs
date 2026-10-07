//! `OpenDocument` Text: `content.xml` read in events, `text:h` as
//! headings, `text:p` as body lines, `table:table` as table sections,
//! `text:note` bodies as note sections; the title, creator, dates, and
//! keywords from `meta.xml`.

use quick_xml::Reader;
use quick_xml::events::Event;

use super::budget::DecompressionBudget;
use super::parser::{DocumentMeta, Extracted, Flow, Section, SectionBuilder};
use super::table::Table;
use super::zipped::{Package, Xml};
use crate::error::{Error, Result};

const KIND: &str = "an OpenDocument text file";

/// An ODT as sections with headings, tables, and notes.
///
/// # Errors
///
/// Returns an error when the bytes are not an ODT, it holds no text, or
/// the package inflates past `budget`.
pub fn extract(data: &[u8], budget: DecompressionBudget) -> Result<Extracted> {
    let mut package = Package::open(data, KIND, budget)?;
    let content = package.required("content.xml", KIND)?;
    let mut meta = DocumentMeta::default();
    let mut title = None;
    if let Some(xml) = package.part("meta.xml")? {
        title = Xml(&xml).text("title");
        DocumentMeta::set(
            &mut meta.author,
            Xml(&xml)
                .text("creator")
                .or_else(|| Xml(&xml).text("initial-creator"))
                .as_deref(),
        );
        DocumentMeta::set(
            &mut meta.authored_at,
            Xml(&xml).text("creation-date").as_deref(),
        );
        DocumentMeta::set(&mut meta.modified_at, Xml(&xml).text("date").as_deref());
        meta.extra("subject", Xml(&xml).text("subject").as_deref());
        meta.extra("description", Xml(&xml).text("description").as_deref());
        for keyword in Xml(&xml).texts("keyword") {
            meta.tag(&keyword);
        }
    }
    let sections = Walk::sections(&content)?;
    if sections.is_empty() {
        return Err(Error::Ingestion(String::from(
            "no extractable text: the document has no paragraphs",
        )));
    }
    Ok(Extracted {
        title,
        sections,
        flow: Flow::Sectioned,
        pages: None,
        meta,
        blank_pages: Vec::new(),
    })
}

/// Where the reader is in `content.xml`.
#[derive(Default)]
struct Walk {
    out: SectionBuilder,
    /// The heading or paragraph being read.
    text: String,
    in_heading: bool,
    paragraphs: u32,
    /// The table being read: its rows, the row being read, the cell.
    table: Option<Vec<Vec<String>>>,
    row: Vec<String>,
    cell: String,
    /// The note body being read, when inside `text:note-body`.
    note: Option<String>,
    note_depth: u32,
    /// Inside `text:note-citation`: the marker, not text.
    in_citation: bool,
    /// Notes met inside the paragraph being read, pushed after it.
    pending_notes: Vec<String>,
}

impl Walk {
    /// `content.xml`'s headings, paragraphs, tables, and notes as sections.
    fn sections(xml: &str) -> Result<Vec<Section>> {
        let mut reader = Reader::from_str(xml);
        let mut walk = Self::default();
        let mut buf = Vec::new();
        loop {
            let event = reader
                .read_event_into(&mut buf)
                .map_err(|e| Error::Ingestion(format!("ODT XML error: {e}")))?;
            match event {
                Event::Start(e) => match e.local_name().as_ref() {
                    "h" if walk.note.is_none() && walk.table.is_none() => {
                        walk.in_heading = true;
                        walk.text.clear();
                    }
                    "p" => walk.paragraphs = walk.paragraphs.saturating_add(1),
                    "table" if walk.note.is_none() => walk.table = Some(Vec::new()),
                    "table-row" => walk.row.clear(),
                    "table-cell" => walk.cell.clear(),
                    "note-body" => {
                        walk.note_depth = walk.note_depth.saturating_add(1);
                        if walk.note.is_none() {
                            walk.note = Some(String::new());
                        }
                    }
                    "note-citation" => walk.in_citation = true,
                    _ => {}
                },
                Event::Empty(e) => match e.local_name().as_ref() {
                    "s" | "tab" => walk.push_text(" "),
                    "line-break" => walk.push_text("\n"),
                    "table-cell" => walk.row.push(String::new()),
                    _ => {}
                },
                Event::Text(t) => walk.push_text(&t.xml10_content()),
                Event::GeneralRef(r) => {
                    if let Some(t) = Xml::entity_text(&r) {
                        walk.push_text(&t);
                    }
                }
                Event::End(e) => match e.local_name().as_ref() {
                    "h" | "p" => walk.end_paragraph(),
                    "table-cell" => {
                        let cell = std::mem::take(&mut walk.cell);
                        walk.row.push(cell);
                    }
                    "table-row" => {
                        let row = std::mem::take(&mut walk.row);
                        if let Some(rows) = walk.table.as_mut() {
                            rows.push(row);
                        }
                    }
                    "table" => {
                        if let Some(rows) = walk.table.take()
                            && let Some(table) = Table::from_rows(rows)
                        {
                            walk.out.push(Section::table(None, table.render()));
                        }
                    }
                    "note-body" => {
                        walk.note_depth = walk.note_depth.saturating_sub(1);
                        if walk.note_depth == 0
                            && let Some(note) = walk.note.take()
                        {
                            let note = note.trim().to_owned();
                            if !note.is_empty() {
                                walk.pending_notes.push(note);
                            }
                        }
                    }
                    "note-citation" => walk.in_citation = false,
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
        Ok(walk.out.finish())
    }

    fn push_text(&mut self, t: &str) {
        if self.in_citation {
            return;
        }
        if let Some(note) = self.note.as_mut() {
            note.push_str(t);
        } else if self.table.is_some() {
            self.cell.push_str(t);
        } else {
            self.text.push_str(t);
        }
    }

    fn end_paragraph(&mut self) {
        if self.note.is_some() {
            if let Some(note) = self.note.as_mut() {
                note.push('\n');
            }
            return;
        }
        if self.table.is_some() {
            self.cell.push(' ');
            return;
        }
        let text = std::mem::take(&mut self.text);
        let text = text.trim();
        if self.in_heading {
            self.in_heading = false;
            self.out.heading(text);
        } else if !text.is_empty() {
            self.out.line(text);
        }
        for note in std::mem::take(&mut self.pending_notes) {
            self.out.push(Section::note(None, note));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::zipped::tests::package;
    use super::*;

    const CONTENT: &str = r#"<?xml version="1.0"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"><office:body><office:text>
<text:h text:outline-level="1">Exclusions</text:h>
<text:p>Flood is<text:s/>excluded.<text:note text:note-class="footnote"><text:note-citation>1</text:note-citation><text:note-body><text:p>See the policy.</text:p></text:note-body></text:note></text:p>
<table:table><table:table-row><table:table-cell><text:p>Peril</text:p></table:table-cell><table:table-cell><text:p>Limit</text:p></table:table-cell></table:table-row><table:table-row><table:table-cell><text:p>Fire</text:p></table:table-cell><table:table-cell><text:p>1,000</text:p></table:table-cell></table:table-row></table:table>
<text:h text:outline-level="2">Claims</text:h>
<text:p>Close in 30 days.</text:p>
</office:text></office:body></office:document-content>"#;
    const META: &str = r#"<?xml version="1.0"?><office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" xmlns:dc="http://purl.org/dc/elements/1.1/"><office:meta><dc:title>Renewal Playbook</dc:title><dc:creator>Ada</dc:creator><meta:creation-date>2026-01-05T14:30:00</meta:creation-date><dc:date>2026-02-01T09:00:00</dc:date><meta:keyword>policy</meta:keyword><meta:keyword>renewal</meta:keyword></office:meta></office:document-meta>"#;

    #[test]
    fn headings_tables_notes_and_metadata_are_read() {
        let bytes = package(&[("content.xml", CONTENT), ("meta.xml", META)]);
        let extracted = extract(&bytes, DecompressionBudget::megabytes(1)).unwrap_or_default();
        assert_eq!(extracted.title.as_deref(), Some("Renewal Playbook"));
        assert_eq!(extracted.meta.author.as_deref(), Some("Ada"));
        assert_eq!(
            extracted.meta.authored_at.as_deref(),
            Some("2026-01-05T14:30:00")
        );
        assert_eq!(
            extracted.meta.modified_at.as_deref(),
            Some("2026-02-01T09:00:00")
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
                (Some("Exclusions"), "body", "Flood is excluded."),
                (Some("Exclusions"), "note", "See the policy."),
                (
                    Some("Exclusions"),
                    "table",
                    "| Peril | Limit |\n| --- | --- |\n| Fire | 1,000 |"
                ),
                (Some("Claims"), "body", "Close in 30 days."),
            ]
        );
    }

    #[test]
    fn a_package_without_content_or_text_is_refused() {
        let no_content = package(&[("meta.xml", META)]);
        assert!(extract(&no_content, DecompressionBudget::megabytes(1)).is_err());
        let empty = package(&[("content.xml", "<office:document-content/>")]);
        assert!(extract(&empty, DecompressionBudget::megabytes(1)).is_err());
    }
}
