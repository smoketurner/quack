//! Markdown for the transcript: an answer's blocks as rows of styled spans,
//! which the transcript then wraps to its width. Tables and strikethrough
//! are on, as they are for the web UI, because models produce them.

use std::mem;

use pulldown_cmark::{Alignment, Event, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;

use crate::terminal::ui::wrap;

type Row = Vec<Span<'static>>;

/// Columns of a thematic break.
const RULE_WIDTH: usize = 24;

const CODE: Style = Style::new().fg(Color::Cyan);
const DIM: Style = Style::new().fg(Color::DarkGray);

/// `content` as rows: headings bold, list items marked and indented, code
/// verbatim, tables as aligned columns that fit `width`, and a link's
/// address after its text. A line break in the source stays a line break.
pub(crate) fn render(content: &str, width: usize) -> Vec<Row> {
    let options = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH;
    let mut renderer = Renderer {
        width,
        ..Renderer::default()
    };
    for event in Parser::new_ext(content, options) {
        renderer.event(event);
    }
    renderer.finish()
}

#[derive(Default)]
struct Renderer {
    /// Columns a row may take before the transcript wraps it.
    width: usize,
    rows: Vec<Row>,
    /// The row being written.
    current: Row,
    /// Inline styles in effect, innermost last.
    styles: Vec<Style>,
    /// Open lists, innermost last.
    lists: Vec<ListLevel>,
    /// Block quotes the rows are inside.
    quotes: usize,
    /// The text of the code block being read.
    code: Option<String>,
    table: Option<Table>,
    /// Open links and images, innermost last.
    links: Vec<Link>,
}

/// One open list.
struct ListLevel {
    /// The next item's number in an ordered list.
    next: Option<u64>,
    /// The marker of an item whose first row is not written yet.
    marker: Option<String>,
    /// Columns the current item's marker takes.
    width: usize,
}

impl ListLevel {
    fn begin_item(&mut self) {
        let marker = match self.next {
            Some(number) => {
                self.next = Some(number.saturating_add(1));
                format!("{number}. ")
            }
            None => String::from("\u{2022} "),
        };
        self.width = marker.chars().count();
        self.marker = Some(marker);
    }
}

/// An open link or image: where it points and the text shown for it.
struct Link {
    dest: String,
    text: String,
}

/// A table being read: its cells are lines of spans until every column's
/// width is known.
#[derive(Default)]
struct Table {
    alignments: Vec<Alignment>,
    header: Vec<Cell>,
    body: Vec<Vec<Cell>>,
    /// The cells of the row being read.
    cells: Vec<Cell>,
    /// The lines of the cell being read that a break has ended.
    lines: Cell,
}

/// A cell's lines: one, unless the cell holds line breaks.
type Cell = Vec<Row>;

impl Table {
    /// Columns between two cells: a bar with a space either side.
    const GAP: usize = 3;

    fn line_width(line: &[Span<'static>]) -> usize {
        line.iter().map(Span::width).sum()
    }

    /// Each column's width when nothing is wrapped: its widest line.
    fn natural_widths(&self) -> Vec<usize> {
        let mut widths: Vec<usize> = Vec::new();
        for row in self.body.iter().chain([&self.header]) {
            for (column, cell) in row.iter().enumerate() {
                let width = cell
                    .iter()
                    .map(|line| Self::line_width(line))
                    .max()
                    .unwrap_or(0);
                match widths.get_mut(column) {
                    Some(widest) => *widest = width.max(*widest),
                    None => widths.push(width),
                }
            }
        }
        widths
    }

    /// `natural` narrowed to `available` columns in all: the narrowest
    /// columns keep their width and the wide ones share what is left.
    fn fit(natural: &[usize], available: usize) -> Vec<usize> {
        let mut order: Vec<(usize, usize)> = natural.iter().copied().enumerate().collect();
        order.sort_by_key(|(_, width)| *width);
        let mut widths = natural.to_vec();
        let mut left = available;
        let mut remaining = order.len();
        for (column, width) in order {
            // At least one, so wrapping a cell always ends.
            let share = left.checked_div(remaining).unwrap_or(0).max(1);
            let width = width.min(share);
            if let Some(slot) = widths.get_mut(column) {
                *slot = width;
            }
            left = left.saturating_sub(width);
            remaining = remaining.saturating_sub(1);
        }
        widths
    }

    /// One table row as screen rows: each cell wrapped to its column and
    /// padded by the column's alignment, as many rows as the tallest cell.
    fn rows_of(&self, cells: &[Cell], widths: &[usize]) -> Vec<Row> {
        let wrapped: Vec<Vec<Row>> = widths
            .iter()
            .enumerate()
            .map(|(column, width)| {
                cells
                    .get(column)
                    .into_iter()
                    .flatten()
                    .flat_map(|line| wrap::wrap(line, *width))
                    .collect()
            })
            .collect();
        let height = wrapped.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let mut rows = Vec::new();
        for at in 0..height {
            let mut row = Row::new();
            for (column, width) in widths.iter().enumerate() {
                if column > 0 {
                    row.push(Span::styled(" \u{2502} ", DIM));
                }
                let line = wrapped
                    .get(column)
                    .and_then(|lines| lines.get(at))
                    .map_or(&[][..], Vec::as_slice);
                let pad = width.saturating_sub(Self::line_width(line));
                let before = match self.alignments.get(column) {
                    Some(Alignment::Right) => pad,
                    Some(Alignment::Center) => pad.checked_div(2).unwrap_or(0),
                    Some(Alignment::Left | Alignment::None) | None => 0,
                };
                let after = pad.saturating_sub(before);
                if before > 0 {
                    row.push(Span::raw(" ".repeat(before)));
                }
                row.extend(line.iter().cloned());
                // No padding after the last column: it would only wrap.
                if after > 0 && column.saturating_add(1) < widths.len() {
                    row.push(Span::raw(" ".repeat(after)));
                }
            }
            rows.push(row);
        }
        rows
    }

    /// The header in bold, a rule, then the body, no wider than `width`
    /// while every column can keep at least one.
    fn rows(mut self, width: usize) -> Vec<Row> {
        let natural = self.natural_widths();
        let gaps = natural.len().saturating_sub(1).saturating_mul(Self::GAP);
        let widths = Self::fit(&natural, width.saturating_sub(gaps));
        for span in self.header.iter_mut().flatten().flatten() {
            span.style = span.style.add_modifier(Modifier::BOLD);
        }
        let rule = widths
            .iter()
            .map(|width| "\u{2500}".repeat(*width))
            .collect::<Vec<_>>()
            .join("\u{2500}\u{253C}\u{2500}");
        let mut rows = self.rows_of(&self.header, &widths);
        rows.push(vec![Span::styled(rule, DIM)]);
        for cells in &self.body {
            rows.extend(self.rows_of(cells, &widths));
        }
        rows
    }
}

impl Renderer {
    fn style(&self) -> Style {
        self.styles.last().copied().unwrap_or_default()
    }

    fn push_style(&mut self, modifier: Modifier) {
        self.styles.push(self.style().add_modifier(modifier));
    }

    /// What a row starts with: a bar per block quote, then the indent of
    /// the lists it is in and the marker of an item's first row.
    fn prefix(&mut self) -> Row {
        let mut prefix = Row::new();
        if self.quotes > 0 {
            prefix.push(Span::styled("\u{2502} ".repeat(self.quotes), DIM));
        }
        if let Some((list, outer)) = self.lists.split_last_mut() {
            let indent: usize = outer.iter().map(|level| level.width).sum();
            let marker = list.marker.take().unwrap_or_else(|| " ".repeat(list.width));
            prefix.push(Span::raw(format!("{}{marker}", " ".repeat(indent))));
        }
        prefix
    }

    /// Columns [`Self::prefix`] takes.
    fn prefix_width(&self) -> usize {
        let lists: usize = self.lists.iter().map(|level| level.width).sum();
        self.quotes.saturating_mul(2).saturating_add(lists)
    }

    /// Close the row being written, if anything is on it.
    fn end_row(&mut self) {
        if self.current.is_empty() {
            return;
        }
        let mut row = self.prefix();
        row.append(&mut self.current);
        self.rows.push(row);
    }

    /// A line break: inside a table cell it starts the cell's next line.
    fn line_break(&mut self) {
        match &mut self.table {
            Some(table) => table.lines.push(mem::take(&mut self.current)),
            None => self.end_row(),
        }
    }

    /// Whether `html` is a `<br>` tag, which models write for a line
    /// break inside a table cell.
    fn is_break(html: &str) -> bool {
        let tag = html.trim_start_matches('<').trim_end_matches('>');
        tag.trim_end_matches('/').trim().eq_ignore_ascii_case("br")
    }

    /// Close the row being written and leave a blank one between blocks,
    /// except inside a list.
    fn start_block(&mut self) {
        self.end_row();
        if self.lists.is_empty() && self.rows.last().is_some_and(|row| !row.is_empty()) {
            self.rows.push(Row::new());
        }
    }

    fn text(&mut self, text: &str) {
        if let Some(code) = &mut self.code {
            code.push_str(text);
            return;
        }
        if let Some(link) = self.links.last_mut() {
            link.text.push_str(text);
        }
        self.current
            .push(Span::styled(text.to_owned(), self.style()));
    }

    fn open_link(&mut self, dest: &str) {
        self.links.push(Link {
            dest: dest.to_owned(),
            text: String::new(),
        });
        self.push_style(Modifier::UNDERLINED);
    }

    /// The address follows the text, unless the text is the address.
    fn close_link(&mut self) {
        self.styles.pop();
        let Some(link) = self.links.pop() else {
            return;
        };
        if !link.dest.is_empty() && link.dest != link.text {
            self.current
                .push(Span::styled(format!(" ({})", link.dest), DIM));
        }
    }

    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::InlineHtml(html) if Self::is_break(&html) => self.line_break(),
            Event::Text(text) | Event::InlineHtml(text) | Event::InlineMath(text) => {
                self.text(&text);
            }
            Event::Code(text) => {
                if let Some(link) = self.links.last_mut() {
                    link.text.push_str(&text);
                }
                self.current
                    .push(Span::styled(text.into_string(), self.style().patch(CODE)));
            }
            Event::Html(text) | Event::DisplayMath(text) => {
                for line in text.lines() {
                    self.text(line);
                    self.end_row();
                }
            }
            Event::FootnoteReference(name) => self.text(&format!("[^{name}]")),
            Event::SoftBreak | Event::HardBreak => self.line_break(),
            Event::Rule => {
                self.start_block();
                self.current
                    .push(Span::styled("\u{2500}".repeat(RULE_WIDTH), DIM));
                self.end_row();
            }
            Event::TaskListMarker(done) => self.text(if done { "[x] " } else { "[ ] " }),
        }
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph | Tag::HtmlBlock => self.start_block(),
            Tag::Heading { .. } => {
                self.start_block();
                self.push_style(Modifier::BOLD);
            }
            Tag::BlockQuote(_) => {
                self.start_block();
                self.quotes = self.quotes.saturating_add(1);
            }
            Tag::CodeBlock(_) => {
                self.start_block();
                self.code = Some(String::new());
            }
            Tag::List(first) => {
                if self.lists.is_empty() {
                    self.start_block();
                } else {
                    self.end_row();
                }
                self.lists.push(ListLevel {
                    next: first,
                    marker: None,
                    width: 0,
                });
            }
            Tag::Item => {
                self.end_row();
                if let Some(list) = self.lists.last_mut() {
                    list.begin_item();
                }
            }
            Tag::Table(alignments) => {
                self.start_block();
                self.table = Some(Table {
                    alignments,
                    ..Table::default()
                });
            }
            Tag::Emphasis => self.push_style(Modifier::ITALIC),
            Tag::Strong => self.push_style(Modifier::BOLD),
            Tag::Strikethrough => self.push_style(Modifier::CROSSED_OUT),
            Tag::Link { dest_url, .. } | Tag::Image { dest_url, .. } => self.open_link(&dest_url),
            Tag::TableHead
            | Tag::TableRow
            | Tag::TableCell
            | Tag::FootnoteDefinition(_)
            | Tag::DefinitionList
            | Tag::DefinitionListTitle
            | Tag::DefinitionListDefinition
            | Tag::Superscript
            | Tag::Subscript
            | Tag::MetadataBlock(_) => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph | TagEnd::HtmlBlock => self.end_row(),
            TagEnd::Heading(_) => {
                self.end_row();
                self.styles.pop();
            }
            TagEnd::BlockQuote(_) => {
                self.end_row();
                self.quotes = self.quotes.saturating_sub(1);
            }
            TagEnd::CodeBlock => {
                let code = self.code.take().unwrap_or_default();
                for line in code.lines() {
                    // A terminal cell cannot hold a tab.
                    let line = line.replace('\t', "    ");
                    self.current.push(Span::styled(format!("  {line}"), CODE));
                    self.end_row();
                }
            }
            TagEnd::List(_) => {
                self.end_row();
                self.lists.pop();
            }
            TagEnd::Item => {
                self.end_row();
                // An item with nothing in it is still its marker.
                if self.lists.last().is_some_and(|list| list.marker.is_some()) {
                    let row = self.prefix();
                    self.rows.push(row);
                }
            }
            TagEnd::TableCell => {
                let line = mem::take(&mut self.current);
                if let Some(table) = &mut self.table {
                    table.lines.push(line);
                    let cell = mem::take(&mut table.lines);
                    table.cells.push(cell);
                }
            }
            TagEnd::TableHead => {
                if let Some(table) = &mut self.table {
                    table.header = mem::take(&mut table.cells);
                }
            }
            TagEnd::TableRow => {
                if let Some(table) = &mut self.table {
                    let cells = mem::take(&mut table.cells);
                    table.body.push(cells);
                }
            }
            TagEnd::Table => {
                // The row's prefix comes out of the table's width.
                let indent = self.prefix_width();
                let width = self.width.saturating_sub(indent);
                let rows = self
                    .table
                    .take()
                    .map(|table| table.rows(width))
                    .unwrap_or_default();
                for row in rows {
                    self.current = row;
                    self.end_row();
                }
            }
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => {
                self.styles.pop();
            }
            TagEnd::Link | TagEnd::Image => self.close_link(),
            TagEnd::FootnoteDefinition
            | TagEnd::DefinitionList
            | TagEnd::DefinitionListTitle
            | TagEnd::DefinitionListDefinition
            | TagEnd::Superscript
            | TagEnd::Subscript
            | TagEnd::MetadataBlock(_) => {}
        }
    }

    fn finish(mut self) -> Vec<Row> {
        self.end_row();
        if self.rows.is_empty() {
            self.rows.push(Row::new());
        }
        self.rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(rows: &[Row]) -> Vec<String> {
        rows.iter()
            .map(|r| r.iter().map(|s| s.content.to_string()).collect())
            .collect()
    }

    fn rendered(content: &str) -> Vec<String> {
        text_of(&render(content, 80))
    }

    #[test]
    fn headings_bullets_fences_and_inline_marks_render() {
        let rows = render(
            "## Deadliest\n- **Tornado** in `Texas`, *twice*\n```sql\nSELECT 1\n\n\tFROM t\n```\n| a | b |",
            80,
        );
        assert_eq!(
            text_of(&rows),
            [
                "Deadliest",
                "",
                "\u{2022} Tornado in Texas, twice",
                "",
                "  SELECT 1",
                "  ",
                "      FROM t",
                "",
                // No delimiter row, so not a table.
                "| a | b |"
            ]
        );
        let style_of = |row: usize, span: usize| {
            rows.get(row)
                .and_then(|r| r.get(span))
                .map(|s| s.style)
                .unwrap_or_default()
        };
        assert!(style_of(0, 0).add_modifier.contains(Modifier::BOLD));
        assert!(style_of(2, 1).add_modifier.contains(Modifier::BOLD));
        assert_eq!(style_of(2, 3).fg, Some(Color::Cyan));
        assert!(style_of(2, 5).add_modifier.contains(Modifier::ITALIC));
        assert_eq!(style_of(4, 0).fg, Some(Color::Cyan));
    }

    #[test]
    fn a_table_is_columns_padded_by_their_alignment() {
        let rows = render(
            "| name | n | mid |\n|:--|--:|:-:|\n| a | 10 | x |\n| long | 2 | \u{65E5}\u{672C}\u{8A9E} |",
            80,
        );
        assert_eq!(
            text_of(&rows),
            [
                "name \u{2502}  n \u{2502}  mid",
                "\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
                "a    \u{2502} 10 \u{2502}   x",
                // Padded by columns: each of these three takes two.
                "long \u{2502}  2 \u{2502} \u{65E5}\u{672C}\u{8A9E}",
            ]
        );
        assert!(
            rows.first()
                .and_then(|r| r.first())
                .is_some_and(|s| s.style.add_modifier.contains(Modifier::BOLD))
        );
        // A row short of cells still lines up, and one cell is one column.
        assert_eq!(
            rendered("| a | b |\n|---|---|\n| only |"),
            [
                "a    \u{2502} b",
                "\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}",
                "only \u{2502} "
            ]
        );
    }

    #[test]
    fn a_table_wider_than_the_transcript_wraps_inside_its_columns() {
        let table = "| k | value |\n|---|---|\n| a | one two three four<br>five |\n| bb | x |";
        // The narrow column keeps its width; the wide one takes the rest
        // and wraps, and a `<br>` starts a new line in its cell.
        let rows = text_of(&render(table, 15));
        assert_eq!(
            rows,
            [
                "k  \u{2502} value",
                "\u{2500}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
                "a  \u{2502} one two ",
                "   \u{2502} three four",
                "   \u{2502} five",
                "bb \u{2502} x",
            ]
        );
        assert!(rows.iter().all(|row| row.chars().count() <= 15), "{rows:?}");
        // Inside a list the marker's columns come out of the table's.
        let listed = text_of(&render(
            &format!("- item\n\n  {}", table.replace('\n', "\n  ")),
            17,
        ));
        assert_eq!(
            listed.get(1).map(String::as_str),
            Some("  k  \u{2502} value")
        );
        assert!(
            listed.iter().all(|row| row.chars().count() <= 17),
            "{listed:?}"
        );
        // Too narrow for every column: each keeps one and wrapping ends.
        assert!(!render(table, 1).is_empty());
        // Outside a table a `<br>` is a line break too.
        assert_eq!(rendered("one<br/>two<BR />three"), ["one", "two", "three"]);
    }

    #[test]
    fn lists_number_nest_and_quotes_are_barred() {
        assert_eq!(
            rendered(
                "3. three\n4. four\n   - inner\n     more\n   - \n\n> quoted\n> again\n\n---\nend"
            ),
            [
                "3. three",
                "4. four",
                "   \u{2022} inner",
                "     more",
                "   \u{2022} ",
                "",
                "\u{2502} quoted",
                "\u{2502} again",
                "",
                &"\u{2500}".repeat(RULE_WIDTH),
                "",
                "end"
            ]
        );
        // A loose list keeps each item's marker on its first paragraph.
        assert_eq!(
            rendered("- one\n\n  still one\n\n- two"),
            ["\u{2022} one", "  still one", "\u{2022} two"]
        );
    }

    #[test]
    fn a_link_shows_its_address_unless_the_text_is_the_address() {
        assert_eq!(
            rendered("[the docs](https://docs.test/a) and <https://plain.test> ![alt](pic.png)"),
            ["the docs (https://docs.test/a) and https://plain.test alt (pic.png)"]
        );
    }

    #[test]
    fn source_line_breaks_citation_markers_and_raw_html_are_kept() {
        assert_eq!(
            rendered("Sources:\n[1] a.pdf\n[2] b_c_d.pdf  \n~~old~~ <b>x</b> 2 * 3 * 4"),
            [
                "Sources:",
                "[1] a.pdf",
                "[2] b_c_d.pdf",
                "old <b>x</b> 2 * 3 * 4"
            ]
        );
        assert_eq!(rendered("<div>\nraw\n</div>"), ["<div>", "raw", "</div>"]);
    }

    #[test]
    fn a_reply_still_streaming_renders_what_has_arrived() {
        assert_eq!(rendered(""), [""], "nothing yet is one empty row");
        assert_eq!(rendered("```sql\nSELECT"), ["  SELECT"]);
        assert_eq!(rendered("some **bold"), ["some **bold"]);
        assert_eq!(rendered("| a | b |\n|---|"), ["| a | b |", "|---|"]);
    }
}
