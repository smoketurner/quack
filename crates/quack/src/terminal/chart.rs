use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Axis, Bar, BarChart, BarGroup, Block, Borders, Chart, Dataset, GraphType};
use ratatui::{Frame, symbols};

use quack_core::analysis::chart::{ChartKind as SpecKind, ChartSpec};

#[derive(Debug, Clone)]
pub(crate) struct SeriesData {
    pub(crate) name: String,
    pub(crate) values: Vec<u64>,
}

#[derive(Debug, Clone)]
pub(crate) enum ChartKind {
    Bar {
        labels: Vec<String>,
        values: Vec<u64>,
    },
    Grouped {
        labels: Vec<String>,
        series: Vec<SeriesData>,
    },
    Line {
        points: Vec<(f64, f64)>,
        x_bounds: [f64; 2],
        y_bounds: [f64; 2],
        x_labels: Vec<String>,
        y_labels: Vec<String>,
        scatter: bool,
    },
    Pie {
        slices: Vec<(String, u64)>,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct ChartData {
    pub(crate) title: String,
    pub(crate) kind: ChartKind,
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
        let kind = match spec.kind {
            SpecKind::Bar if spec.series.len() > 1 => ChartKind::Grouped {
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
            SpecKind::Bar => ChartKind::Bar {
                labels,
                values: spec
                    .series
                    .first()
                    .map(|s| s.values.iter().copied().map(to_u64).collect())
                    .unwrap_or_default(),
            },
            SpecKind::Line | SpecKind::Scatter => line_kind(spec, spec.kind == SpecKind::Scatter),
            SpecKind::Pie => ChartKind::Pie {
                slices: labels
                    .into_iter()
                    .zip(
                        spec.series
                            .first()
                            .map(|s| s.values.clone())
                            .unwrap_or_default(),
                    )
                    .map(|(name, value)| (name, to_u64(value)))
                    .collect(),
            },
        };
        Self { title, kind }
    }

    pub(crate) fn height(&self) -> u16 {
        match &self.kind {
            ChartKind::Bar { values, .. } => {
                if values.is_empty() {
                    3
                } else {
                    12
                }
            }
            ChartKind::Grouped { series, .. } => {
                if series.is_empty() {
                    3
                } else {
                    14
                }
            }
            ChartKind::Line { .. } => 12,
            ChartKind::Pie { slices, .. } => {
                let lines = u16::try_from(slices.len()).unwrap_or(u16::MAX);
                lines.saturating_add(4).min(20)
            }
        }
    }
}

fn line_kind(spec: &ChartSpec, scatter: bool) -> ChartKind {
    let y_values: Vec<f64> = spec
        .series
        .first()
        .map(|s| s.values.clone())
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

    let y_labels = vec![
        format!("{y_min:.0}"),
        format!("{:.0}", f64::midpoint(y_min, y_max)),
        format!("{y_max:.0}"),
    ];

    ChartKind::Line {
        points,
        x_bounds: [0.0, x_max.max(1.0)],
        y_bounds: [y_min - y_pad, y_max + y_pad],
        x_labels: spec.x.values.clone(),
        y_labels,
        scatter,
    }
}

pub(crate) fn render_chart(frame: &mut Frame<'_>, area: Rect, chart_data: &ChartData) {
    let block = Block::default()
        .title(Line::styled(
            format!(" {} ", chart_data.title),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    match &chart_data.kind {
        ChartKind::Bar { labels, values } => {
            render_bar(frame, area, block, labels, values);
        }
        ChartKind::Grouped { labels, series } => {
            render_grouped_bar(frame, area, block, labels, series);
        }
        ChartKind::Line { .. } => {
            render_line(frame, area, block, &chart_data.kind);
        }
        ChartKind::Pie { slices } => {
            render_pie(frame, area, block, slices);
        }
    }
}

fn render_bar(
    frame: &mut Frame<'_>,
    area: Rect,
    block: Block<'_>,
    labels: &[String],
    values: &[u64],
) {
    let bar_data: Vec<(&str, u64)> = labels
        .iter()
        .zip(values.iter())
        .map(|(l, &v)| (l.as_str(), v))
        .collect();

    let bar_chart = BarChart::default()
        .block(block)
        .data(&bar_data)
        .bar_width(5)
        .bar_gap(1)
        .bar_style(Style::default().fg(Color::Green))
        .value_style(
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        );

    frame.render_widget(bar_chart, area);
}

#[expect(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "series index bounded by PIE_COLORS modulo; label index bounded by zip"
)]
fn render_grouped_bar(
    frame: &mut Frame<'_>,
    area: Rect,
    block: Block<'_>,
    labels: &[String],
    series: &[SeriesData],
) {
    let mut groups: Vec<BarGroup<'_>> = Vec::new();

    for (i, label) in labels.iter().enumerate() {
        let bars: Vec<Bar<'_>> = series
            .iter()
            .enumerate()
            .map(|(si, s)| {
                let value = s.values.get(i).copied().unwrap_or(0);
                Bar::new(value)
                    .label(Line::from(s.name.clone()))
                    .style(Style::default().fg(PIE_COLORS[si % PIE_COLORS.len()]))
            })
            .collect();
        groups.push(BarGroup::new(bars).label(Line::from(label.as_str())));
    }

    let bar_chart = BarChart::grouped(groups)
        .block(block)
        .bar_width(3)
        .bar_gap(0)
        .group_gap(2)
        .value_style(
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        );

    frame.render_widget(bar_chart, area);
}

fn render_line(frame: &mut Frame<'_>, area: Rect, block: Block<'_>, kind: &ChartKind) {
    let ChartKind::Line {
        points,
        x_bounds,
        y_bounds,
        x_labels,
        y_labels,
        scatter,
    } = kind
    else {
        return;
    };

    let dataset = Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(if *scatter {
            GraphType::Scatter
        } else {
            GraphType::Line
        })
        .style(Style::default().fg(Color::Cyan))
        .data(points);

    let x_axis_labels: Vec<Line<'_>> = if x_labels.is_empty() {
        vec![
            format!("{:.0}", x_bounds[0]).into(),
            format!("{:.0}", x_bounds[1]).into(),
        ]
    } else {
        let first = x_labels.first().cloned().unwrap_or_default();
        let last = x_labels.last().cloned().unwrap_or_default();
        vec![first.into(), last.into()]
    };

    let y_axis_labels: Vec<Line<'_>> = y_labels.iter().map(|s| s.as_str().into()).collect();

    let x_axis = Axis::default()
        .style(Style::default().fg(Color::DarkGray))
        .bounds(*x_bounds)
        .labels(x_axis_labels);

    let y_axis = Axis::default()
        .style(Style::default().fg(Color::DarkGray))
        .bounds(*y_bounds)
        .labels(y_axis_labels);

    let chart = Chart::new(vec![dataset])
        .block(block)
        .x_axis(x_axis)
        .y_axis(y_axis);

    frame.render_widget(chart, area);
}

const PIE_COLORS: &[Color] = &[
    Color::Green,
    Color::Cyan,
    Color::Yellow,
    Color::Magenta,
    Color::Blue,
    Color::Red,
    Color::White,
    Color::LightGreen,
];

#[expect(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    reason = "pie chart percentage calculation with bounded color index"
)]
fn render_pie(frame: &mut Frame<'_>, area: Rect, block: Block<'_>, slices: &[(String, u64)]) {
    let total: u64 = slices.iter().map(|(_, v)| v).sum();
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if total == 0 {
        return;
    }

    let mut lines: Vec<Line<'_>> = Vec::new();
    for (i, (name, value)) in slices.iter().enumerate() {
        let pct = (*value as f64 / total as f64) * 100.0;
        let color = PIE_COLORS[i % PIE_COLORS.len()];
        let bar_len = (pct / 100.0 * 20.0) as usize;
        let bar: String = "\u{2588}".repeat(bar_len);

        lines.push(Line::from(vec![
            ratatui::text::Span::styled(format!(" {bar:<20} "), Style::default().fg(color)),
            ratatui::text::Span::styled(
                format!("{name}: {value} ({pct:.1}%)"),
                Style::default().fg(Color::White),
            ),
        ]));
    }

    let paragraph = ratatui::widgets::Paragraph::new(lines);
    frame.render_widget(paragraph, inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use quack_core::analysis::chart::{Axis as SpecAxis, Series};

    fn spec(kind: SpecKind, labels: &[&str], series: &[(&str, &[f64])]) -> ChartSpec {
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
            SpecKind::Bar,
            &["N", "S"],
            &[("sales", &[100.0, 200.4])],
        ));
        assert!(matches!(
            &data.kind,
            ChartKind::Bar { labels, values } if labels == &["N", "S"] && values == &[100, 200]
        ));
        assert_eq!(data.height(), 12);
    }

    #[test]
    fn bar_with_two_series_is_grouped() {
        let data = ChartData::from_spec(&spec(
            SpecKind::Bar,
            &["Q1", "Q2"],
            &[("revenue", &[420.0, 480.0]), ("profit", &[105.0, 120.0])],
        ));
        assert!(matches!(
            &data.kind,
            ChartKind::Grouped { series, .. }
                if series.len() == 2 && series.last().is_some_and(|s| s.name == "profit")
        ));
    }

    #[test]
    fn line_and_scatter_share_the_line_kind() {
        let line = ChartData::from_spec(&spec(
            SpecKind::Line,
            &["a", "b", "c"],
            &[("y", &[1.0, 3.0, 2.0])],
        ));
        assert!(matches!(
            &line.kind,
            ChartKind::Line { scatter: false, points, .. } if points.len() == 3
        ));
        let scatter = ChartData::from_spec(&spec(SpecKind::Scatter, &["a"], &[("y", &[-5.0])]));
        assert!(matches!(
            &scatter.kind,
            ChartKind::Line { scatter: true, y_bounds, .. } if y_bounds.first().is_some_and(|b| *b < -5.0)
        ));
    }

    #[test]
    fn pie_pairs_labels_with_the_first_series() {
        let data = ChartData::from_spec(&spec(
            SpecKind::Pie,
            &["A", "B"],
            &[("share", &[60.0, 40.0])],
        ));
        assert!(matches!(
            &data.kind,
            ChartKind::Pie { slices } if slices == &[(String::from("A"), 60), (String::from("B"), 40)]
        ));
        assert_eq!(data.height(), 6);
    }

    #[test]
    fn empty_series_render_without_panicking() {
        let empty = ChartData::from_spec(&spec(SpecKind::Bar, &[], &[]));
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
            .draw(|frame| render_chart(frame, frame.area(), chart_data))
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
            SpecKind::Bar,
            SpecKind::Line,
            SpecKind::Scatter,
            SpecKind::Pie,
        ] {
            let data =
                ChartData::from_spec(&spec(kind, &["a", "b", "c"], &[("y", &[1.0, 2.0, 3.0])]));
            assert!(render_to_string(&data).contains(" t "), "{kind:?}");
        }
    }
}
