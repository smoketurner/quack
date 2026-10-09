//! Selecting transcript text with the mouse. A selection is held in
//! transcript lines, not screen rows, so scrolling does not move it.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, StyledGrapheme};

use crate::ui::wrap;

/// The columns every transcript row gives to its marker.
pub(crate) const PREFIX_WIDTH: usize = 3;

/// Whether a row begins a line of the message or continues one that was
/// too long for the width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Wrap {
    Start,
    Continued,
}

/// One drawn row of the transcript.
#[derive(Debug, Clone)]
pub(crate) struct Row {
    pub(crate) line: Line<'static>,
    pub(crate) wrap: Wrap,
}

impl Row {
    pub(crate) fn start(line: Line<'static>) -> Self {
        Self {
            line,
            wrap: Wrap::Start,
        }
    }

    /// Each grapheme with the column it starts at.
    fn cells(&self) -> impl Iterator<Item = (usize, StyledGrapheme<'_>)> {
        let mut column = 0_usize;
        self.line
            .styled_graphemes(Style::default())
            .map(move |cell| {
                let at = column;
                column = column.saturating_add(Span::raw(cell.symbol).width());
                (at, cell)
            })
    }

    fn push_selected(&self, columns: Columns, text: &mut String) {
        for (at, cell) in self.cells() {
            if columns.cover(at, cell.symbol) {
                text.push_str(cell.symbol);
            }
        }
    }

    /// Its line with the selected cells in reverse video.
    pub(crate) fn highlighted(&self, columns: Columns) -> Line<'static> {
        let cells: Vec<StyledGrapheme<'_>> = self
            .cells()
            .map(|(at, mut cell)| {
                if columns.cover(at, cell.symbol) {
                    cell.style = cell.style.add_modifier(Modifier::REVERSED);
                }
                cell
            })
            .collect();
        Line::from(wrap::regroup(&cells))
    }
}

/// The selected columns of one row, both ends included.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Columns {
    first: usize,
    last: usize,
}

impl Columns {
    /// Whether the grapheme starting at `at` is selected. One that is two
    /// columns wide is taken whole, and the marker's columns never are.
    fn cover(self, at: usize, symbol: &str) -> bool {
        let end = at.saturating_add(Span::raw(symbol).width().max(1));
        at >= PREFIX_WIDTH && at <= self.last && end > self.first
    }
}

/// A cell of the transcript: its line, and the column within the row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Position {
    pub(crate) line: usize,
    pub(crate) column: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Drag {
    Dragging,
    Done,
}

/// What the mouse has dragged over, from where the button went down to
/// where the pointer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Selection {
    anchor: Position,
    head: Position,
    drag: Drag,
}

impl Selection {
    pub(crate) fn at(position: Position) -> Self {
        Self {
            anchor: position,
            head: position,
            drag: Drag::Dragging,
        }
    }

    pub(crate) fn extend(&mut self, head: Position) {
        self.head = head;
    }

    pub(crate) fn finish(&mut self) {
        self.drag = Drag::Done;
    }

    pub(crate) fn is_dragging(&self) -> bool {
        self.drag == Drag::Dragging
    }

    /// The button went down and up on one cell.
    pub(crate) fn is_click(&self) -> bool {
        self.anchor == self.head
    }

    fn ends(&self) -> (Position, Position) {
        (self.anchor.min(self.head), self.anchor.max(self.head))
    }

    /// The columns it takes of transcript line `line`, if any.
    pub(crate) fn columns(&self, line: usize) -> Option<Columns> {
        let (start, end) = self.ends();
        if self.is_click() || line < start.line || line > end.line {
            return None;
        }
        Some(Columns {
            first: if line == start.line { start.column } else { 0 },
            last: if line == end.line {
                end.column
            } else {
                usize::MAX
            },
        })
    }

    /// The selected text as the message has it: no markers, no padding,
    /// and a wrapped line in one piece.
    pub(crate) fn text(&self, rows: &[Row]) -> String {
        let (start, _) = self.ends();
        let mut text = String::new();
        for (index, row) in rows.iter().enumerate().skip(start.line) {
            let Some(columns) = self.columns(index) else {
                break;
            };
            if index > start.line && row.wrap == Wrap::Start {
                text.truncate(text.trim_end_matches(' ').len());
                text.push('\n');
            }
            row.push_selected(columns, &mut text);
        }
        text.truncate(text.trim_end().len());
        let leading = text
            .len()
            .saturating_sub(text.trim_start_matches('\n').len());
        text.split_off(leading)
    }
}

/// Where the pointer is against the transcript's area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Edge {
    Inside,
    Above,
    Below,
}

/// A pointer placed on the transcript: the nearest cell, and whether the
/// pointer had left the area to get there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Located {
    pub(crate) position: Position,
    pub(crate) edge: Edge,
}

/// The transcript as last drawn: its text area, the line on its first row,
/// and how many lines there are.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TranscriptView {
    pub(crate) area: Rect,
    pub(crate) top: usize,
    pub(crate) lines: usize,
}

impl TranscriptView {
    /// The transcript cell nearest the screen cell, or `None` while
    /// nothing is drawn.
    pub(crate) fn locate(self, column: u16, row: u16) -> Option<Located> {
        let last_line = self.lines.checked_sub(1)?;
        let last_column = usize::from(self.area.width.checked_sub(1)?);
        if row < self.area.y {
            return Some(Located {
                position: Position {
                    line: self.top.saturating_sub(1),
                    column: 0,
                },
                edge: Edge::Above,
            });
        }
        if row >= self.area.bottom() {
            let line = self.top.saturating_add(usize::from(self.area.height));
            return Some(Located {
                position: Position {
                    line: line.min(last_line),
                    column: last_column,
                },
                edge: Edge::Below,
            });
        }
        let line = self
            .top
            .saturating_add(usize::from(row.saturating_sub(self.area.y)));
        let position = if line > last_line {
            // Below the last line of a transcript shorter than the screen.
            Position {
                line: last_line,
                column: last_column,
            }
        } else {
            Position {
                line,
                column: usize::from(column.saturating_sub(self.area.x)).min(last_column),
            }
        };
        Some(Located {
            position,
            edge: Edge::Inside,
        })
    }
}

#[cfg(test)]
mod tests {
    use proptest::collection;
    use proptest::prelude::{prop_assert, proptest};

    use super::*;

    /// `text` as one message's rows at `width`, each under a marker.
    fn rows(text: &str, width: usize) -> Vec<Row> {
        let mut rows = Vec::new();
        for line in text.lines() {
            let wrapped = wrap::wrap(&[Span::raw(line.to_owned())], width);
            for (index, spans) in wrapped.into_iter().enumerate() {
                let mut with_prefix = vec![Span::raw(" > ")];
                with_prefix.extend(spans);
                rows.push(Row {
                    line: Line::from(with_prefix),
                    wrap: if index == 0 {
                        Wrap::Start
                    } else {
                        Wrap::Continued
                    },
                });
            }
        }
        rows
    }

    fn selected(rows: &[Row], from: (usize, usize), to: (usize, usize)) -> String {
        let mut selection = Selection::at(Position {
            line: from.0,
            column: from.1,
        });
        selection.extend(Position {
            line: to.0,
            column: to.1,
        });
        selection.text(rows)
    }

    #[test]
    fn a_selection_copies_the_text_without_markers_or_padding() {
        let rows = rows("alpha beta\ngamma", 40);
        assert_eq!(selected(&rows, (0, 3), (0, 7)), "alpha", "both ends count");
        assert_eq!(selected(&rows, (0, 9), (1, 5)), "beta\ngam");
        assert_eq!(
            selected(&rows, (0, 0), (1, 80)),
            "alpha beta\ngamma",
            "the marker's columns and the space past the text add nothing"
        );
        assert_eq!(
            selected(&rows, (1, 5), (0, 9)),
            "beta\ngam",
            "a drag upward selects the same text"
        );
        assert_eq!(
            selected(&rows, (0, 4), (0, 4)),
            "",
            "a click selects nothing"
        );
    }

    #[test]
    fn a_wrapped_line_copies_in_one_piece() {
        let rows = rows("the quick brown fox jumps over\nnext", 10);
        assert!(rows.len() > 3, "the first line wrapped");
        let last = rows.len().saturating_sub(1);
        assert_eq!(
            selected(&rows, (0, 0), (last, 80)),
            "the quick brown fox jumps over\nnext"
        );
    }

    #[test]
    fn a_wide_character_under_either_end_is_taken_whole() {
        // Each takes two columns: 3-4, 5-6, 7-8.
        let rows = rows("\u{65E5}\u{672C}\u{8A9E}", 40);
        assert_eq!(selected(&rows, (0, 4), (0, 5)), "\u{65E5}\u{672C}");
        assert_eq!(selected(&rows, (0, 6), (0, 7)), "\u{672C}\u{8A9E}");
    }

    #[test]
    fn blank_lines_inside_are_kept_and_those_around_are_not() {
        let mut rows = rows("one", 40);
        rows.push(Row::start(Line::from("")));
        rows.extend(self::rows("two", 40));
        rows.push(Row::start(Line::from("")));
        assert_eq!(selected(&rows, (0, 0), (3, 80)), "one\n\ntwo");
        assert_eq!(selected(&rows, (1, 0), (3, 80)), "two");
    }

    #[test]
    fn the_highlight_reverses_only_the_selected_cells() {
        let rows = rows("alpha beta", 40);
        let mut selection = Selection::at(Position { line: 0, column: 0 });
        selection.extend(Position { line: 0, column: 7 });
        let line = selection
            .columns(0)
            .and_then(|columns| rows.first().map(|row| row.highlighted(columns)));
        let reversed: String = line
            .iter()
            .flat_map(|line| &line.spans)
            .filter(|span| span.style.add_modifier.contains(Modifier::REVERSED))
            .map(|span| span.content.as_ref())
            .collect();
        assert_eq!(reversed, "alpha");
        assert_eq!(
            line.map(|line| line.to_string()).as_deref(),
            Some(" > alpha beta")
        );
        assert_eq!(selection.columns(1), None);
    }

    #[test]
    fn a_pointer_maps_to_the_line_under_it() {
        let view = TranscriptView {
            area: Rect::new(0, 2, 79, 10),
            top: 40,
            lines: 60,
        };
        let at = |column, row| view.locate(column, row).map(|l| (l.position, l.edge));
        let position = |line, column| Position { line, column };
        assert_eq!(at(5, 2), Some((position(40, 5), Edge::Inside)));
        assert_eq!(at(5, 11), Some((position(49, 5), Edge::Inside)));
        assert_eq!(
            at(79, 3),
            Some((position(41, 78), Edge::Inside)),
            "the scrollbar's column is the row's last"
        );
        assert_eq!(at(5, 1), Some((position(39, 0), Edge::Above)));
        assert_eq!(at(5, 12), Some((position(50, 78), Edge::Below)));

        let short = TranscriptView {
            area: Rect::new(0, 2, 79, 10),
            top: 0,
            lines: 3,
        };
        assert_eq!(
            short.locate(5, 9).map(|l| l.position),
            Some(position(2, 78)),
            "below the last line is its end"
        );
        assert_eq!(
            short.locate(5, 30).map(|l| l.position),
            Some(position(2, 78))
        );
        assert_eq!(TranscriptView::default().locate(0, 0), None);
    }

    proptest! {
        #[test]
        fn any_selection_is_a_piece_of_the_text(
            words in collection::vec("[a-z\u{65E5}\u{672C}]{1,12}", 1..40),
            width in 4_usize..30,
            from in (0_usize..60, 0_usize..40),
            to in (0_usize..60, 0_usize..40),
        ) {
            let text = words.join(" ");
            let rows = rows(&text, width);
            let picked = selected(&rows, from, to);
            prop_assert!(text.contains(&picked), "{picked:?} is not in {text:?}");
        }
    }
}
