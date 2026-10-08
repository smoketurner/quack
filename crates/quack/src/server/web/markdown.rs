//! Markdown to HTML for answers. Raw HTML the model writes is escaped, so
//! nothing it says can inject markup; a link keeps its target only when it
//! is web, mail, or relative, so a prompt-injected `javascript:` link is
//! plain text; an image becomes its alt text, so an answer never makes the
//! browser fetch anything. Tables and strikethrough are on because models
//! produce them.

use pulldown_cmark::{CowStr, Event, Options, Parser, Tag, TagEnd, html};

/// Render `text` as HTML.
pub(crate) fn to_html(text: &str) -> String {
    let options = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH;
    // One entry per open link: whether it was dropped, so its end is too.
    let mut links: Vec<bool> = Vec::new();
    let events = Parser::new_ext(text, options).filter_map(|event| match event {
        Event::Html(raw) | Event::InlineHtml(raw) => {
            Some(Event::Text(CowStr::from(raw.into_string())))
        }
        Event::Start(Tag::Link { ref dest_url, .. }) => {
            let kept = Target(dest_url).is_safe();
            links.push(!kept);
            kept.then_some(event)
        }
        Event::End(TagEnd::Link) => (!links.pop().unwrap_or(false)).then_some(event),
        Event::Start(Tag::Image { .. }) => Some(Event::Start(Tag::Emphasis)),
        Event::End(TagEnd::Image) => Some(Event::End(TagEnd::Emphasis)),
        other => Some(other),
    });
    let mut out = String::with_capacity(text.len().saturating_mul(2));
    html::push_html(&mut out, events);
    out
}

/// A link's destination as the model wrote it.
struct Target<'a>(&'a str);

impl Target<'_> {
    /// Whether the browser may follow it: `http`, `https`, `mailto`, or a
    /// destination with no scheme at all (a path or a fragment).
    fn is_safe(&self) -> bool {
        let url = self.0.trim();
        let end = url.find([':', '/', '?', '#']).unwrap_or(url.len());
        let has_scheme = url.get(end..).is_some_and(|rest| rest.starts_with(':'));
        if !has_scheme {
            return true;
        }
        let scheme = url.get(..end).unwrap_or_default().to_ascii_lowercase();
        matches!(scheme.as_str(), "http" | "https" | "mailto")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_bold_and_lists_render() {
        let html =
            to_html("Here **is** it:\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n1. one\n2. two\n");
        assert!(html.contains("<strong>is</strong>"));
        assert!(
            html.contains("<table>") && html.contains("<th>a</th>") && html.contains("<td>2</td>")
        );
        assert!(html.contains("<ol>") && html.contains("<li>two</li>"));
    }

    #[test]
    fn only_web_mail_and_relative_links_survive_and_images_become_text() {
        let html = to_html(
            "[a](https://example.com) [b](/w/x) [c](#n) [d](mailto:a@b.c) \
             [e](javascript:alert(1)) [f](JavaScript:x) [g](data:text/html,x) \
             ![chart](https://evil.example/?q=secret)",
        );
        assert!(
            html.contains(r#"<a href="https://example.com">a</a>"#),
            "{html}"
        );
        assert!(html.contains(r#"<a href="/w/x">b</a>"#), "{html}");
        assert!(html.contains(r##"<a href="#n">c</a>"##), "{html}");
        assert!(html.contains(r#"<a href="mailto:a@b.c">d</a>"#), "{html}");
        assert!(
            !html.contains("javascript:") && !html.contains("JavaScript:"),
            "{html}"
        );
        assert!(!html.contains("data:text"), "{html}");
        assert!(!html.contains(">e</a>") && html.contains(" e "), "{html}");
        assert!(
            !html.contains("<img") && !html.contains("evil.example"),
            "{html}"
        );
        assert!(html.contains("<em>chart</em>"), "{html}");
    }

    #[test]
    fn raw_html_is_escaped_not_passed_through() {
        let html = to_html("hi <script>alert(1)</script> <b>x</b>");
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
        assert!(html.contains("&lt;b&gt;x&lt;/b&gt;"));
    }

    #[test]
    fn code_keeps_its_text_and_plain_text_is_a_paragraph() {
        assert_eq!(to_html("plain"), "<p>plain</p>\n");
        let html = to_html("```sql\nSELECT 1\n```");
        assert!(html.contains("<pre><code class=\"language-sql\">SELECT 1\n</code></pre>"));
    }
}
