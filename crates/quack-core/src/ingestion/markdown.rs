//! Markdown through `pulldown-cmark`: a section per heading (a `#` inside
//! a code fence is code, not a heading), pipe tables as table sections,
//! and YAML front matter as the document's title and metadata, never as
//! prose.

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

use super::parser::{DocumentMeta, Extracted, Flow, Section, SectionBuilder};
use super::table::Table;
use crate::okf::{WithFrontMatter, parse_front_matter};

/// A Markdown file as sections, its front matter read into the metadata.
#[must_use]
pub fn extract(text: &str) -> Extracted {
    let WithFrontMatter { front, body } = parse_front_matter(text);
    let mut meta = DocumentMeta::default();
    DocumentMeta::set(
        &mut meta.author,
        front.get("author").or_else(|| front.get("authors")),
    );
    DocumentMeta::set(
        &mut meta.authored_at,
        front
            .get("date")
            .or_else(|| front.get("created"))
            .or_else(|| front.get("published")),
    );
    DocumentMeta::set(
        &mut meta.modified_at,
        front.get("modified").or_else(|| front.get("updated")),
    );
    for tag in &front.tags {
        meta.tag(tag);
    }
    for (key, value) in &front.fields {
        if !matches!(
            key.as_str(),
            "title"
                | "author"
                | "authors"
                | "date"
                | "created"
                | "published"
                | "modified"
                | "updated"
        ) {
            meta.extra(key, Some(value));
        }
    }
    Extracted {
        title: front.get("title").map(str::to_owned),
        sections: sections(body),
        flow: Flow::Sectioned,
        pages: None,
        meta,
    }
}

/// Split Markdown at its headings, with tables as sections of their own.
/// Inline formatting is dropped; the text of links, code, and emphasis is
/// kept.
#[must_use]
pub fn sections(text: &str) -> Vec<Section> {
    let mut out = SectionBuilder::default();
    let mut line = String::new();
    let mut heading: Option<String> = None;
    let mut table: Option<TableRows> = None;
    for event in Parser::new_ext(text, Options::ENABLE_TABLES | Options::ENABLE_FOOTNOTES) {
        match event {
            Event::Start(Tag::Heading { .. }) => {
                flush_line(&mut out, &mut line);
                heading = Some(String::new());
            }
            Event::End(TagEnd::Heading(_)) => {
                if let Some(h) = heading.take() {
                    out.heading(h.trim());
                }
            }
            Event::Start(Tag::Table(_)) => {
                flush_line(&mut out, &mut line);
                table = Some(TableRows::default());
            }
            Event::End(TagEnd::Table) => {
                if let Some(rows) = table.take()
                    && let Some(found) = Table::from_rows(rows.rows)
                {
                    out.push(Section::table(None, found.render()));
                }
            }
            Event::Start(Tag::TableHead | Tag::TableRow) => {
                if let Some(rows) = table.as_mut() {
                    rows.rows.push(Vec::new());
                }
            }
            Event::End(TagEnd::TableCell) => {
                if let Some(rows) = table.as_mut() {
                    rows.end_cell();
                }
            }
            Event::Start(Tag::Paragraph | Tag::Item | Tag::CodeBlock(_) | Tag::BlockQuote(_))
            | Event::End(
                TagEnd::Paragraph | TagEnd::Item | TagEnd::CodeBlock | TagEnd::BlockQuote(_),
            )
            | Event::HardBreak
            | Event::Rule => {
                if table.is_none() {
                    flush_line(&mut out, &mut line);
                }
            }
            // Inline code keeps its backticks: an identifier in code is one
            // in the text, and a phrase search for it still matches.
            Event::Code(t) => {
                let code = format!("`{t}`");
                match (&mut heading, &mut table) {
                    (Some(h), _) => h.push_str(&t),
                    (None, Some(rows)) => rows.cell.push_str(&code),
                    (None, None) => line.push_str(&code),
                }
            }
            Event::Text(t) | Event::InlineMath(t) | Event::DisplayMath(t) => {
                match (&mut heading, &mut table) {
                    (Some(h), _) => h.push_str(&t),
                    (None, Some(rows)) => rows.cell.push_str(&t),
                    (None, None) => line.push_str(&t),
                }
            }
            Event::SoftBreak => match (&mut heading, &mut table) {
                (Some(h), _) => h.push(' '),
                (None, Some(rows)) => rows.cell.push(' '),
                (None, None) => {
                    // A code block's lines arrive as one text with newlines;
                    // prose lines as text split by soft breaks.
                    flush_line(&mut out, &mut line);
                }
            },
            Event::FootnoteReference(name) => {
                if table.is_none() && heading.is_none() {
                    line.push_str("[^");
                    line.push_str(&name);
                    line.push(']');
                }
            }
            Event::Start(Tag::FootnoteDefinition(name)) => {
                flush_line(&mut out, &mut line);
                line.push_str("[^");
                line.push_str(&name);
                line.push_str("]: ");
            }
            Event::End(TagEnd::FootnoteDefinition) => flush_line(&mut out, &mut line),
            Event::Start(_)
            | Event::End(_)
            | Event::Html(_)
            | Event::InlineHtml(_)
            | Event::TaskListMarker(_) => {}
        }
    }
    flush_line(&mut out, &mut line);
    out.finish()
}

/// A table's rows as the parser hands their cells over.
#[derive(Default)]
struct TableRows {
    rows: Vec<Vec<String>>,
    cell: String,
}

impl TableRows {
    fn end_cell(&mut self) {
        let text = std::mem::take(&mut self.cell);
        if let Some(row) = self.rows.last_mut() {
            row.push(text);
        }
    }
}

fn flush_line(out: &mut SectionBuilder, line: &mut String) {
    let text = std::mem::take(line);
    // A code block's text keeps its own newlines.
    for l in text.lines() {
        if !l.trim().is_empty() {
            out.line(l);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_matter_gives_the_title_and_metadata_and_is_not_chunked() {
        let extracted = extract(
            "---\ntitle: Renewal Guide\nauthor: Ada\ndate: 2026-01-05\ntags: [policy, renewal]\nowner: claims\n---\n\n# Terms\n\nThirty days.\n",
        );
        assert_eq!(extracted.title(), Some("Renewal Guide"));
        assert_eq!(extracted.meta.author.as_deref(), Some("Ada"));
        assert_eq!(extracted.meta.authored_at.as_deref(), Some("2026-01-05"));
        assert_eq!(extracted.meta.tags, ["policy", "renewal"]);
        assert_eq!(
            extracted.meta.extra.get("owner").map(String::as_str),
            Some("claims")
        );
        assert_eq!(extracted.sections.len(), 1);
        assert!(extracted.sections.iter().all(|s| !s.text.contains("tags:")));
    }

    #[test]
    fn splits_on_atx_and_setext_headings_but_not_inside_code_fences() {
        let md = "intro line\n\n# Exclusions\n\nFlood is excluded.\n\nClaims\n------\n\nClose in 30 days.\n\n```\n# not a heading\ncode line\n```\n\n## Real\ntext\n";
        let summary: Vec<(Option<String>, String)> = sections(md)
            .into_iter()
            .map(|s| (s.heading, s.text))
            .collect();
        assert_eq!(
            summary,
            vec![
                (None, String::from("intro line")),
                (
                    Some(String::from("Exclusions")),
                    String::from("Flood is excluded.")
                ),
                (
                    Some(String::from("Claims")),
                    String::from("Close in 30 days.\n# not a heading\ncode line")
                ),
                (Some(String::from("Real")), String::from("text")),
            ]
        );
    }

    #[test]
    fn a_pipe_table_is_its_own_section_under_the_heading() {
        let md = "## Limits\n\nBefore.\n\n| Peril | Limit |\n|---|---|\n| Fire | 1,000 |\n| Flood | none |\n\nAfter.\n";
        let found = sections(md);
        let kinds: Vec<&str> = found.iter().map(|s| s.kind.as_str()).collect();
        assert_eq!(kinds, ["body", "table", "body"]);
        assert_eq!(
            found.get(1).map(|s| s.text.as_str()),
            Some("| Peril | Limit |\n| --- | --- |\n| Fire | 1,000 |\n| Flood | none |")
        );
        assert!(found.iter().all(|s| s.heading.as_deref() == Some("Limits")));
    }

    #[test]
    fn markdown_without_headings_is_one_section_and_keeps_list_text() {
        let found = sections("just\n\n- one *two* `three`\n- [four](http://x)\n\ntext");
        assert_eq!(found.len(), 1);
        assert_eq!(found.first().and_then(|s| s.heading.as_deref()), None);
        assert_eq!(
            found.first().map(|s| s.text.as_str()),
            Some("just\none two `three`\nfour\ntext")
        );
    }
}
