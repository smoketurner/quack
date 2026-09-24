use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
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
];

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
                let lines = u16::try_from(slices.len()).unwrap_or(u16::MAX);
                lines.saturating_add(4).min(20)
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

/// Each slice as a bar of up to twenty cells, its value, and its share.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "a slice's share of the total as a percentage and a bar length"
)]
fn pie_lines(slices: &[Slice]) -> Vec<Line<'_>> {
    let total: u64 = slices.iter().map(|s| s.value).sum();
    if total == 0 {
        return Vec::new();
    }
    slices
        .iter()
        .zip(COLORS.iter().cycle())
        .map(|(slice, &color)| {
            let pct = (slice.value as f64 / total as f64) * 100.0;
            let bar = "\u{2588}".repeat((pct / 100.0 * 20.0) as usize);
            Line::from(vec![
                Span::styled(format!(" {bar:<20} "), Style::default().fg(color)),
                Span::styled(
                    format!("{}: {} ({pct:.1}%)", slice.name, slice.value),
                    Style::default().fg(Color::White),
                ),
            ])
        })
        .collect()
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
                Paragraph::new(pie_lines(slices)).render(inner, buf);
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
        assert_eq!(data.height(), 6);
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
