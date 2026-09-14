//! Markdown to HTML for answers. Raw HTML the model writes is escaped, so
//! nothing it says can inject markup; tables and strikethrough are on
//! because models produce them.

use pulldown_cmark::{CowStr, Event, Options, Parser, html};

/// Render `text` as HTML.
pub(crate) fn to_html(text: &str) -> String {
    let options = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH;
    let events = Parser::new_ext(text, options).map(|event| match event {
        Event::Html(raw) | Event::InlineHtml(raw) => Event::Text(CowStr::from(raw.into_string())),
        other => other,
    });
    let mut out = String::with_capacity(text.len().saturating_mul(2));
    html::push_html(&mut out, events);
    out
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
