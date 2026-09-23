//! HTML through `scraper` (html5ever): the `<title>` and, walking the body
//! in document order, one section per `h1`..`h6` heading with block
//! elements separated by newlines. Scripts, styles, and head content are
//! skipped.

use scraper::{Html, Node, Selector};

use super::parser::{Extracted, Flow, Section};
use crate::error::{Error, Result};

/// An HTML document as sections with headings and its `<title>`.
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

    let mut walker = Walker::default();
    let root = document.tree.root();
    walker.visit(root);
    walker.flush();
    if walker.sections.is_empty() {
        return Err(Error::Ingestion(String::from(
            "no extractable text: the HTML has no body text",
        )));
    }
    Ok(Extracted {
        title,
        sections: walker.sections,
        flow: Flow::Sectioned,
        pages_skipped: 0,
    })
}

#[derive(Default)]
struct Walker {
    sections: Vec<Section>,
    heading: Option<String>,
    /// Lines of the section being built.
    lines: Vec<String>,
    /// Text of the current line.
    line: String,
}

const SKIPPED: &[&str] = &["script", "style", "noscript", "head", "template", "svg"];
/// Every heading level starts a section; the level itself is not kept.
const HEADINGS: &[&str] = &["h1", "h2", "h3", "h4", "h5", "h6"];
const BLOCKS: &[&str] = &[
    "p",
    "div",
    "li",
    "tr",
    "table",
    "section",
    "article",
    "header",
    "footer",
    "nav",
    "aside",
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
    "form",
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
                    let title = collapse(&subtree_text(node));
                    self.flush();
                    self.heading = (!title.is_empty()).then_some(title);
                    return;
                }
                let block = BLOCKS.contains(&name);
                if block {
                    self.end_line();
                }
                if name == "td" || name == "th" {
                    self.line.push(' ');
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
            self.lines.push(line);
        }
    }

    fn flush(&mut self) {
        self.end_line();
        let text = self.lines.join("\n");
        self.lines.clear();
        if !text.trim().is_empty() {
            self.sections.push(Section {
                heading: self.heading.clone(),
                page: None,
                text,
            });
        }
    }
}

fn subtree_text(node: ego_tree::NodeRef<'_, Node>) -> String {
    let mut out = String::new();
    for descendant in node.descendants() {
        if let Node::Text(text) = descendant.value() {
            out.push_str(&text.text);
        }
    }
    out
}

/// Whitespace runs collapsed to one space, trimmed.
fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[test]
    fn html_yields_title_and_heading_sections() {
        let page = r"<!doctype html><html><head><title> Renewal  Guide </title>
<style>p{color:red}</style><script>var x = 1;</script></head>
<body><nav>Home | About</nav>
<p>Intro paragraph.</p>
<h1>Exclusions</h1><p>Flood is <b>excluded</b>.</p><ul><li>One</li><li>Two</li></ul>
<h2>Claims <em>process</em></h2><table><tr><th>Col</th><th>Val</th></tr><tr><td>a</td><td>1</td></tr></table>
<div>Line<br>break</div><script>ignored()</script></body></html>";
        let extracted = html(page).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(extracted.title.as_deref(), Some("Renewal Guide"));
        let got: Vec<(Option<&str>, &str)> = extracted
            .sections
            .iter()
            .map(|s| (s.heading.as_deref(), s.text.as_str()))
            .collect();
        assert_eq!(
            got,
            [
                (None, "Home | About\nIntro paragraph."),
                (Some("Exclusions"), "Flood is excluded.\nOne\nTwo"),
                (Some("Claims process"), "Col Val\na 1\nLine\nbreak"),
            ]
        );
    }

    #[test]
    fn html_without_text_is_an_error_and_tolerates_junk() {
        assert!(html("<html><head><title>t</title></head><body></body></html>").is_err());
        assert!(html("<p>unclosed <b>tags").is_ok_and(|e| e.title.is_none()));
        assert!(html("just text, no tags").is_ok_and(|e| {
            e.sections
                .first()
                .is_some_and(|s| s.text == "just text, no tags")
        }));
    }
}
