//! HTML through `scraper` (html5ever): the `<title>` and the page's
//! `<meta>` tags, then, walking the body in document order, one section
//! per `h1`..`h6` heading with block elements separated by newlines and
//! every `<table>` as a table section. Scripts, styles, head content, and
//! the page's chrome (`nav`, `aside`, `footer`, `form`) are skipped.

use scraper::{Html, Node, Selector};

use super::parser::{DocumentMeta, Extracted, Flow, Section, SectionBuilder};
use super::table::Table;
use crate::error::{Error, Result};

/// An HTML document as sections with headings, its `<title>`, and what
/// its `<meta>` tags say.
///
/// # Errors
///
/// Returns an error when the page holds no text.
pub fn html(text: &str) -> Result<Extracted> {
    let document = Html::parse_document(text);
    let title = Selector::parse("title")
        .ok()
        .and_then(|sel| document.select(&sel).next())
        .map(|el| el.text().collect::<String>())
        .map(|t| collapse(&t))
        .filter(|t| !t.is_empty());
    let meta = DocumentMeta::from_html(&document);

    let mut walker = Walker::default();
    let root = document.tree.root();
    walker.visit(root);
    walker.end_line();
    let sections = walker.sections.finish();
    if sections.is_empty() {
        return Err(Error::Ingestion(String::from(
            "no extractable text: the HTML has no body text",
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

#[derive(Default)]
struct Walker {
    sections: SectionBuilder,
    /// Text of the current line.
    line: String,
}

const SKIPPED: &[&str] = &[
    "script", "style", "noscript", "head", "template", "svg", "nav", "aside", "footer", "form",
];
/// Every heading level starts a section; the level itself is not kept.
const HEADINGS: &[&str] = &["h1", "h2", "h3", "h4", "h5", "h6"];
const BLOCKS: &[&str] = &[
    "p",
    "div",
    "li",
    "tr",
    "section",
    "article",
    "header",
    "main",
    "pre",
    "blockquote",
    "ul",
    "ol",
    "dl",
    "dt",
    "dd",
    "figure",
    "figcaption",
    "hr",
    "br",
    "address",
    "fieldset",
];

impl Walker {
    fn visit(&mut self, node: ego_tree::NodeRef<'_, Node>) {
        match node.value() {
            Node::Text(text) => self.line.push_str(&text.text),
            Node::Element(element) => {
                let name = element.name();
                if SKIPPED.contains(&name) {
                    return;
                }
                if HEADINGS.contains(&name) {
                    self.end_line();
                    self.sections.heading(collapse(&subtree_text(node)));
                    return;
                }
                if name == "table" {
                    self.end_line();
                    match Table::from_rows(Self::table_rows(node)) {
                        Some(table) => self.sections.push(Section::table(None, table.render())),
                        // A one-column or header-only table reads as lines.
                        None => {
                            for row in Self::table_rows(node) {
                                let line = collapse(&row.join(" "));
                                if !line.is_empty() {
                                    self.sections.line(line);
                                }
                            }
                        }
                    }
                    return;
                }
                let block = BLOCKS.contains(&name);
                if block {
                    self.end_line();
                }
                for child in node.children() {
                    self.visit(child);
                }
                if block {
                    self.end_line();
                }
            }
            Node::Document | Node::Fragment => {
                for child in node.children() {
                    self.visit(child);
                }
            }
            Node::Doctype(_) | Node::Comment(_) | Node::ProcessingInstruction(_) => {}
        }
    }

    fn end_line(&mut self) {
        let line = collapse(&self.line);
        self.line.clear();
        if !line.is_empty() {
            self.sections.line(line);
        }
    }
}

impl DocumentMeta {
    /// The `<meta>` tags that name an author, a date, keywords, or a
    /// description, under the names HTML, Dublin Core, and Open Graph use.
    fn from_html(document: &Html) -> Self {
        let mut meta = Self::default();
        let Ok(selector) = Selector::parse("meta") else {
            return meta;
        };
        for element in document.select(&selector) {
            let name = element
                .value()
                .attr("name")
                .or_else(|| element.value().attr("property"))
                .map(str::to_ascii_lowercase)
                .unwrap_or_default();
            let content = element.value().attr("content");
            match name.as_str() {
                "author" | "dc.creator" | "dcterms.creator" | "article:author" => {
                    Self::set(&mut meta.author, content);
                }
                "date"
                | "dc.date"
                | "dcterms.created"
                | "dc.date.created"
                | "article:published_time"
                | "datepublished" => Self::set(&mut meta.authored_at, content),
                "last-modified" | "dcterms.modified" | "article:modified_time" | "datemodified" => {
                    Self::set(&mut meta.modified_at, content);
                }
                "keywords" | "dc.subject" | "article:tag" => {
                    for tag in content.unwrap_or_default().split(',') {
                        meta.tag(tag);
                    }
                }
                "description" | "dc.description" | "og:description" => {
                    meta.extra("description", content);
                }
                _ => {}
            }
        }
        meta
    }
}

impl Walker {
    /// A table's rows as their cells' text, `th` and `td` alike, in order.
    fn table_rows(table: ego_tree::NodeRef<'_, Node>) -> Vec<Vec<String>> {
        let mut rows = Vec::new();
        for descendant in table.descendants() {
            let Node::Element(element) = descendant.value() else {
                continue;
            };
            if element.name() != "tr" {
                continue;
            }
            let cells: Vec<String> = descendant
                .children()
                .filter(
                    |c| matches!(c.value(), Node::Element(e) if e.name() == "td" || e.name() == "th"),
                )
                .map(|c| collapse(&subtree_text(c)))
                .collect();
            if !cells.is_empty() {
                rows.push(cells);
            }
        }
        rows
    }
}

fn subtree_text(node: ego_tree::NodeRef<'_, Node>) -> String {
    let mut out = String::new();
    for descendant in node.descendants() {
        if let Node::Text(text) = descendant.value() {
            // Mirror `Walker::visit`'s `SKIPPED` policy: drop text sitting
            // under a skipped element (script/style/noscript/template/svg
            // nested in a heading), so it never pollutes the section heading.
            let skipped = descendant
                .ancestors()
                .any(|a| matches!(a.value(), Node::Element(e) if SKIPPED.contains(&e.name())));
            if !skipped {
                out.push_str(&text.text);
            }
        }
    }
    out
}

/// Whitespace collapsed to single spaces, trimmed.
fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_follow_headings_and_the_title_is_read() {
        let page = "<html><head><title> Policy  Guide </title><script>x()</script></head><body><h1>Exclusions</h1><p>Flood is <b>excluded</b>.</p><div>Two<br>lines</div><h2>Claims</h2><ul><li>Close in 30 days.</li></ul></body></html>";
        let extracted = html(page).unwrap_or_default();
        assert_eq!(extracted.title.as_deref(), Some("Policy Guide"));
        let summary: Vec<(Option<&str>, &str)> = extracted
            .sections
            .iter()
            .map(|s| (s.heading.as_deref(), s.text.as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![
                (Some("Exclusions"), "Flood is excluded.\nTwo\nlines"),
                (Some("Claims"), "Close in 30 days."),
            ]
        );
    }

    #[test]
    fn chrome_is_skipped_and_tables_are_their_own_sections() {
        let page = "<body><nav>Home About</nav><aside>Related</aside><h1>Limits</h1><p>Before.</p><table><tr><th>Peril</th><th>Limit</th></tr><tr><td>Fire</td><td>1,000</td></tr></table><footer>Copyright</footer><form><input></form></body>";
        let extracted = html(page).unwrap_or_default();
        let summary: Vec<(&str, &str)> = extracted
            .sections
            .iter()
            .map(|s| (s.kind.as_str(), s.text.as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![
                ("body", "Before."),
                (
                    "table",
                    "| Peril | Limit |\n| --- | --- |\n| Fire | 1,000 |"
                ),
            ]
        );
        assert!(
            extracted
                .sections
                .iter()
                .all(|s| s.heading.as_deref() == Some("Limits"))
        );
    }

    #[test]
    fn meta_tags_give_author_dates_tags_and_description() {
        let page = r#"<html><head><meta name="author" content="Ada"><meta name="date" content="2026-01-05"><meta property="article:modified_time" content="2026-02-01"><meta name="keywords" content="policy, renewal"><meta name="description" content="A guide."></head><body><p>text</p></body></html>"#;
        let extracted = html(page).unwrap_or_default();
        assert_eq!(extracted.meta.author.as_deref(), Some("Ada"));
        assert_eq!(extracted.meta.authored_at.as_deref(), Some("2026-01-05"));
        assert_eq!(extracted.meta.modified_at.as_deref(), Some("2026-02-01"));
        assert_eq!(extracted.meta.tags, ["policy", "renewal"]);
        assert_eq!(
            extracted.meta.extra.get("description").map(String::as_str),
            Some("A guide.")
        );
    }

    #[test]
    fn a_page_without_text_is_an_error_and_a_heading_ignores_nested_scripts() {
        assert!(html("<html><body><script>x</script></body></html>").is_err());
        let page = "<body><h1>Title<script>bad()</script></h1><p>ok</p></body>";
        let extracted = html(page).unwrap_or_default();
        assert_eq!(
            extracted
                .sections
                .first()
                .and_then(|s| s.heading.as_deref()),
            Some("Title")
        );
    }
}
