use std::f64::consts::TAU;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Painter, Shape};
use ratatui::widgets::{
    Axis, Bar, BarChart, BarGroup, Block, Borders, Chart, Dataset, GraphType, Paragraph, Widget,
};

use quack_core::analysis::chart::{ChartKind, ChartSpec};

/// Series and slice colours, in turn.
const COLORS: &[Color] = &[
    Color::Green,
    Color::Cyan,
    Color::Yellow,
    Color::Magenta,
    Color::Blue,
    Color::Red,
    Color::White,
    Color::LightGreen,
    Color::LightCyan,
    Color::LightYellow,
    Color::LightMagenta,
    Color::LightBlue,
    Color::LightRed,
    Color::Gray,
];

/// Rows a pie's panel takes at least, so its circle is not a smudge.
const PIE_MIN_HEIGHT: u16 = 12;

#[derive(Debug, Clone)]
pub(crate) struct SeriesData {
    pub(crate) name: String,
    pub(crate) values: Vec<u64>,
}

/// A line or scatter chart's points, already scaled to its axes.
#[derive(Debug, Clone)]
pub(crate) struct LineData {
    pub(crate) points: Vec<(f64, f64)>,
    x_bounds: [f64; 2],
    pub(crate) y_bounds: [f64; 2],
    x_labels: Vec<String>,
    y_labels: Vec<String>,
    pub(crate) graph_type: GraphType,
}

/// One slice of a pie, as a labelled magnitude.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Slice {
    pub(crate) name: String,
    pub(crate) value: u64,
}

/// How the terminal draws a chart.
#[derive(Debug, Clone)]
pub(crate) enum Plot {
    Bars {
        labels: Vec<String>,
        values: Vec<u64>,
    },
    Grouped {
        labels: Vec<String>,
        series: Vec<SeriesData>,
    },
    Line(LineData),
    Pie(Vec<Slice>),
}

#[derive(Debug, Clone)]
pub(crate) struct ChartData {
    pub(crate) title: String,
    pub(crate) plot: Plot,
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bar and pie widgets take integer magnitudes; negatives clamp to zero"
)]
fn to_u64(value: f64) -> u64 {
    value.max(0.0).round() as u64
}

impl ChartData {
    /// Map quack's chart spec onto the terminal renderers.
    pub(crate) fn from_spec(spec: &ChartSpec) -> Self {
        let title = spec.title.clone();
        let labels = spec.x.values.clone();
        let first = spec
            .series
            .first()
            .map(|s| s.values.as_slice())
            .unwrap_or_default();
        let plot = match spec.kind {
            ChartKind::Bar if spec.series.len() > 1 => Plot::Grouped {
                labels,
                series: spec
                    .series
                    .iter()
                    .map(|s| SeriesData {
                        name: s.name.clone(),
                        values: s.values.iter().copied().map(to_u64).collect(),
                    })
                    .collect(),
            },
            ChartKind::Bar => Plot::Bars {
                labels,
                values: first.iter().copied().map(to_u64).collect(),
            },
            ChartKind::Line => Plot::Line(LineData::of(spec, GraphType::Line)),
            ChartKind::Scatter => Plot::Line(LineData::of(spec, GraphType::Scatter)),
            ChartKind::Pie => Plot::Pie(
                labels
                    .into_iter()
                    .zip(first.iter().copied())
                    .map(|(name, value)| Slice {
                        name,
                        value: to_u64(value),
                    })
                    .collect(),
            ),
        };
        Self { title, plot }
    }

    pub(crate) fn height(&self) -> u16 {
        match &self.plot {
            Plot::Bars { values, .. } => {
                if values.is_empty() {
                    3
                } else {
                    12
                }
            }
            Plot::Grouped { series, .. } => {
                if series.is_empty() {
                    3
                } else {
                    14
                }
            }
            Plot::Line(_) => 12,
            Plot::Pie(slices) => {
                if Pie(slices).total() == 0 {
                    3
                } else {
                    // A legend row per slice, and the border.
                    let lines = u16::try_from(slices.len()).unwrap_or(u16::MAX);
                    lines.saturating_add(2).clamp(PIE_MIN_HEIGHT, 20)
                }
            }
        }
    }
}

impl LineData {
    /// The first series against its index, bounded to include zero with a
    /// tenth of the range as margin.
    fn of(spec: &ChartSpec, graph_type: GraphType) -> Self {
        let y_values = spec
            .series
            .first()
            .map(|s| s.values.as_slice())
            .unwrap_or_default();

        #[expect(
            clippy::cast_precision_loss,
            reason = "index-to-f64 for chart coordinates"
        )]
        let points: Vec<(f64, f64)> = y_values
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f64, y))
            .collect();

        #[expect(clippy::cast_precision_loss, reason = "length-to-f64 for chart bounds")]
        let x_max = points.len().saturating_sub(1) as f64;
        let y_min = y_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .min(0.0);
        let y_max = y_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0);
        let y_pad = (y_max - y_min).abs() * 0.1;

        Self {
            points,
            x_bounds: [0.0, x_max.max(1.0)],
            y_bounds: [y_min - y_pad, y_max + y_pad],
            x_labels: spec.x.values.clone(),
            y_labels: vec![
                format!("{y_min:.0}"),
                format!("{:.0}", f64::midpoint(y_min, y_max)),
                format!("{y_max:.0}"),
            ],
            graph_type,
        }
    }

    /// The chart widget: the points against the first and last x labels
    /// (or the bounds when there are none).
    fn chart(&self) -> Chart<'_> {
        let dataset = Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(self.graph_type)
            .style(Style::default().fg(Color::Cyan))
            .data(&self.points);
        let [x_min, x_max] = self.x_bounds;
        let x_labels: Vec<Line<'_>> = match (self.x_labels.first(), self.x_labels.last()) {
            (Some(first), Some(last)) => vec![first.as_str().into(), last.as_str().into()],
            _ => vec![format!("{x_min:.0}").into(), format!("{x_max:.0}").into()],
        };
        let y_labels: Vec<Line<'_>> = self.y_labels.iter().map(|s| s.as_str().into()).collect();
        Chart::new(vec![dataset])
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds(self.x_bounds)
                    .labels(x_labels),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds(self.y_bounds)
                    .labels(y_labels),
            )
    }
}

/// A pie's slices: the circle and the legend beside it.
struct Pie<'a>(&'a [Slice]);

impl Pie<'_> {
    fn total(&self) -> u64 {
        self.0.iter().map(|slice| slice.value).sum()
    }

    /// The slice's share of the whole, from 0 to 1; 0 for an empty pie.
    #[expect(
        clippy::cast_precision_loss,
        reason = "a slice's share of the total, for an angle and a percentage"
    )]
    fn share(&self, slice: &Slice) -> f64 {
        match self.total() {
            0 => 0.0,
            total => slice.value as f64 / total as f64,
        }
    }

    /// The colours in turn, except that a last slice which would wrap
    /// round to the first one's takes the next, since the two touch.
    fn color(&self, index: usize) -> Color {
        let turn = index.checked_rem(COLORS.len()).unwrap_or(0);
        let wraps_onto_first = index > 0 && turn == 0 && index.saturating_add(1) == self.0.len();
        COLORS
            .get(if wraps_onto_first { 1 } else { turn })
            .copied()
            .unwrap_or(Color::Green)
    }

    /// The slice that holds `turn`, a share of the full circle clockwise
    /// from twelve o'clock.
    fn slice_at(&self, turn: f64) -> Option<usize> {
        let mut end = 0.0;
        let mut last = None;
        for (index, slice) in self.0.iter().enumerate() {
            if slice.value == 0 {
                continue;
            }
            end += self.share(slice);
            if turn < end {
                return Some(index);
            }
            last = Some(index);
        }
        // Rounding can leave the last sliver past every end.
        last
    }

    /// One row per slice: its colour, name, value, and share. With more
    /// slices than `rows`, the last row counts the rest.
    fn legend(&self, rows: usize) -> Vec<Line<'static>> {
        let shown = if self.0.len() > rows {
            rows.saturating_sub(1)
        } else {
            self.0.len()
        };
        let mut lines: Vec<Line<'static>> = self
            .0
            .iter()
            .enumerate()
            .take(shown)
            .map(|(index, slice)| {
                Line::from(vec![
                    Span::styled(" \u{25A0} ", Style::default().fg(self.color(index))),
                    Span::styled(
                        format!(
                            "{}: {} ({:.1}%)",
                            slice.name,
                            slice.value,
                            self.share(slice) * 100.0
                        ),
                        Style::default().fg(Color::White),
                    ),
                ])
            })
            .collect();
        if shown < self.0.len() {
            lines.push(Line::styled(
                format!("   and {} more", self.0.len().saturating_sub(shown)),
                Style::default().fg(Color::DarkGray),
            ));
        }
        lines
    }
}

impl Widget for &Pie<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if self.total() == 0 {
            return;
        }
        // A half block is one column wide and half a row high, which is
        // square in a terminal's two-to-one cells.
        let diameter = area
            .width
            .saturating_sub(1)
            .min(area.height.saturating_mul(2));
        let [_, circle, legend] = Layout::horizontal([
            Constraint::Length(1),
            Constraint::Length(diameter),
            Constraint::Fill(1),
        ])
        .areas(area);
        let disc = Disc {
            pie: self,
            across: circle.width,
            down: circle.height.saturating_mul(2),
        };
        Canvas::default()
            .marker(symbols::Marker::HalfBlock)
            // One canvas unit per point, so each point is painted once.
            .x_bounds([0.0, f64::from(disc.across.saturating_sub(1))])
            .y_bounds([0.0, f64::from(disc.down.saturating_sub(1))])
            .paint(|ctx| ctx.draw(&disc))
            .render(circle, buf);
        Paragraph::new(self.legend(usize::from(legend.height))).render(legend, buf);
    }
}

/// A pie as a filled circle on a canvas of `across` by `down` points.
struct Disc<'a> {
    pie: &'a Pie<'a>,
    across: u16,
    down: u16,
}

impl Shape for Disc<'_> {
    fn draw(&self, painter: &mut Painter<'_, '_>) {
        let (across, down) = (f64::from(self.across), f64::from(self.down));
        let radius = across.min(down) / 2.0;
        let (center_x, center_y) = ((across - 1.0) / 2.0, (down - 1.0) / 2.0);
        for row in 0..self.down {
            for column in 0..self.across {
                let (x, y) = (f64::from(column), f64::from(row));
                // `up` grows towards the top, as the canvas's y does.
                let (right, up) = (x - center_x, center_y - y);
                if right.hypot(up) > radius {
                    continue;
                }
                let turn = right.atan2(up).rem_euclid(TAU) / TAU;
                if let Some(index) = self.pie.slice_at(turn)
                    && let Some((px, py)) = painter.get_point(x, down - 1.0 - y)
                {
                    painter.paint(px, py, self.pie.color(index));
                }
            }
        }
    }
}

impl Widget for &ChartData {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(Line::styled(
                format!(" {} ", self.title),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));
        let values = Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);
        match &self.plot {
            Plot::Bars {
                labels,
                values: bars,
            } => {
                let data: Vec<(&str, u64)> = labels
                    .iter()
                    .map(String::as_str)
                    .zip(bars.iter().copied())
                    .collect();
                BarChart::default()
                    .block(block)
                    .data(data.as_slice())
                    .bar_width(5)
                    .bar_gap(1)
                    .bar_style(Style::default().fg(Color::Green))
                    .value_style(values)
                    .render(area, buf);
            }
            Plot::Grouped { labels, series } => {
                let groups: Vec<BarGroup<'_>> = labels
                    .iter()
                    .enumerate()
                    .map(|(i, label)| {
                        let bars: Vec<Bar<'_>> = series
                            .iter()
                            .zip(COLORS.iter().cycle())
                            .map(|(s, &color)| {
                                Bar::new(s.values.get(i).copied().unwrap_or(0))
                                    .label(Line::from(s.name.as_str()))
                                    .style(Style::default().fg(color))
                            })
                            .collect();
                        BarGroup::new(bars).label(Line::from(label.as_str()))
                    })
                    .collect();
                BarChart::grouped(groups)
                    .block(block)
                    .bar_width(3)
                    .bar_gap(0)
                    .group_gap(2)
                    .value_style(values)
                    .render(area, buf);
            }
            Plot::Line(line) => line.chart().block(block).render(area, buf),
            Plot::Pie(slices) => {
                let inner = block.inner(area);
                block.render(area, buf);
                Pie(slices).render(inner, buf);
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use quack_core::analysis::chart::{Axis as SpecAxis, Series};

    fn spec(kind: ChartKind, labels: &[&str], series: &[(&str, &[f64])]) -> ChartSpec {
        ChartSpec {
            title: String::from("t"),
            kind,
            x: SpecAxis {
                label: String::from("x"),
                values: labels.iter().map(|s| (*s).to_owned()).collect(),
            },
            series: series
                .iter()
                .map(|(name, values)| Series {
                    name: (*name).to_owned(),
                    values: values.to_vec(),
                })
                .collect(),
        }
    }

    #[test]
    fn bar_with_one_series_is_a_bar_chart() {
        let data = ChartData::from_spec(&spec(
            ChartKind::Bar,
            &["N", "S"],
            &[("sales", &[100.0, 200.4])],
        ));
        assert!(matches!(
            &data.plot,
            Plot::Bars { labels, values } if labels == &["N", "S"] && values == &[100, 200]
        ));
        assert_eq!(data.height(), 12);
    }

    #[test]
    fn bar_with_two_series_is_grouped() {
        let data = ChartData::from_spec(&spec(
            ChartKind::Bar,
            &["Q1", "Q2"],
            &[("revenue", &[420.0, 480.0]), ("profit", &[105.0, 120.0])],
        ));
        assert!(matches!(
            &data.plot,
            Plot::Grouped { series, .. }
                if series.len() == 2 && series.last().is_some_and(|s| s.name == "profit")
        ));
    }

    #[test]
    fn line_and_scatter_share_the_line_kind() {
        let line = ChartData::from_spec(&spec(
            ChartKind::Line,
            &["a", "b", "c"],
            &[("y", &[1.0, 3.0, 2.0])],
        ));
        assert!(matches!(
            &line.plot,
            Plot::Line(data) if data.graph_type == GraphType::Line && data.points.len() == 3
        ));
        let scatter = ChartData::from_spec(&spec(ChartKind::Scatter, &["a"], &[("y", &[-5.0])]));
        assert!(matches!(
            &scatter.plot,
            Plot::Line(data) if data.graph_type == GraphType::Scatter && data.y_bounds.first().is_some_and(|b| *b < -5.0)
        ));
    }

    #[test]
    fn pie_pairs_labels_with_the_first_series() {
        let data = ChartData::from_spec(&spec(
            ChartKind::Pie,
            &["A", "B"],
            &[("share", &[60.0, 40.0])],
        ));
        assert!(matches!(
            &data.plot,
            Plot::Pie(slices) if slices == &[
                Slice { name: String::from("A"), value: 60 },
                Slice { name: String::from("B"), value: 40 },
            ]
        ));
        assert_eq!(data.height(), PIE_MIN_HEIGHT);
    }

    fn slices(values: &[u64]) -> Vec<Slice> {
        values
            .iter()
            .enumerate()
            .map(|(index, value)| Slice {
                name: format!("s{index}"),
                value: *value,
            })
            .collect()
    }

    #[test]
    fn a_turn_of_the_circle_falls_in_the_slice_that_covers_it() {
        let parts = slices(&[60, 0, 40]);
        let pie = Pie(&parts);
        assert_eq!(pie.slice_at(0.0), Some(0));
        assert_eq!(pie.slice_at(0.59), Some(0));
        // The empty slice between them takes none of the circle.
        assert_eq!(pie.slice_at(0.61), Some(2));
        assert_eq!(pie.slice_at(0.999_999), Some(2));
        // Past every end by rounding: the last slice that has any.
        assert_eq!(pie.slice_at(1.0), Some(2));
        assert_eq!(Pie(&slices(&[0, 0])).slice_at(0.5), None);
        assert_eq!(Pie(&[]).slice_at(0.5), None);
    }

    #[test]
    fn a_last_slice_never_takes_the_first_slices_colour() {
        let wrapped = slices(&vec![1; COLORS.len() + 1]);
        let pie = Pie(&wrapped);
        assert_eq!(pie.color(0), Color::Green);
        assert_ne!(pie.color(COLORS.len()), pie.color(0));
        assert_ne!(pie.color(COLORS.len()), pie.color(COLORS.len() - 1));
        // Away from the wrap the colours just come in turn.
        let longer = slices(&vec![1; COLORS.len() + 2]);
        assert_eq!(Pie(&longer).color(COLORS.len()), Color::Green);
    }

    #[test]
    fn a_pie_is_a_filled_circle_beside_its_legend() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let data = ChartData::from_spec(&spec(
            ChartKind::Pie,
            &["A", "B"],
            &[("share", &[75.0, 25.0])],
        ));
        let mut terminal = Terminal::new(TestBackend::new(60, data.height())).unwrap();
        terminal
            .draw(|frame| frame.render_widget(&data, frame.area()))
            .unwrap();
        let buffer = terminal.backend().buffer();
        // The colours a cell shows: a half block's own, and the half
        // behind it.
        let colors = |x: u16, y: u16| {
            buffer
                .cell((x, y))
                .map(|cell| (cell.symbol().to_owned(), cell.fg, cell.bg))
                .unwrap_or_default()
        };
        // Inside the border and a column's margin the circle is 20
        // columns by 10 rows. A fills
        // three quarters clockwise from the top, so B is the upper left.
        let (symbol, fg, _) = colors(16, 3);
        assert_ne!(symbol, " ");
        assert_eq!(fg, Color::Green, "upper right is A");
        let (symbol, fg, _) = colors(7, 3);
        assert_ne!(symbol, " ");
        assert_eq!(fg, Color::Cyan, "upper left is B");
        let (_, fg, _) = colors(7, 8);
        assert_eq!(fg, Color::Green, "lower left is A");
        // The corner of the square is outside the circle.
        assert_eq!(colors(2, 1).0, " ");
        let text = render_to_string(&data);
        assert!(text.contains("\u{25A0} A: 75 (75.0%)"), "{text}");
        assert!(text.contains("\u{25A0} B: 25 (25.0%)"), "{text}");
    }

    #[test]
    fn a_pie_with_nothing_in_it_and_a_crowded_legend_still_render() {
        let empty = ChartData::from_spec(&spec(ChartKind::Pie, &["A"], &[("y", &[0.0])]));
        assert_eq!(empty.height(), 3);
        render_to_string(&empty);
        // More slices than rows: the legend counts the rest.
        let many = slices(&[1; 30]);
        let lines = Pie(&many).legend(5);
        assert_eq!(lines.len(), 5);
        assert_eq!(
            lines.last().map(ToString::to_string).as_deref(),
            Some("   and 26 more")
        );
    }

    #[test]
    fn empty_series_render_without_panicking() {
        let empty = ChartData::from_spec(&spec(ChartKind::Bar, &[], &[]));
        assert_eq!(empty.height(), 3);
        render_to_string(&empty);
    }

    #[expect(clippy::unwrap_used, reason = "test helper renders into a buffer")]
    fn render_to_string(chart_data: &ChartData) -> String {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let height = chart_data.height();
        let backend = TestBackend::new(80, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| frame.render_widget(chart_data, frame.area()))
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut out = String::new();
        for y in 0..height {
            for x in 0..80 {
                out.push_str(buf.cell((x, y)).unwrap().symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn every_kind_renders_its_title() {
        for kind in [
            ChartKind::Bar,
            ChartKind::Line,
            ChartKind::Scatter,
            ChartKind::Pie,
        ] {
            let data =
                ChartData::from_spec(&spec(kind, &["a", "b", "c"], &[("y", &[1.0, 2.0, 3.0])]));
            assert!(render_to_string(&data).contains(" t "), "{kind:?}");
        }
    }
}
