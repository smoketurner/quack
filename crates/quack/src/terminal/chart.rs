use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::Line;
use ratatui::widgets::{Axis, Bar, BarChart, BarGroup, Block, Borders, Chart, Dataset};

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

impl ChartData {
    pub(crate) fn from_echart_spec(spec: &serde_json::Value) -> Option<Self> {
        let title = spec
            .get("title")
            .and_then(|t| t.get("text"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("Chart")
            .to_owned();

        let all_series = spec.get("series")?.as_array()?;
        let first = all_series.first()?;
        let chart_type = first.get("type")?.as_str()?;

        match chart_type {
            "bar" if all_series.len() > 1 => parse_grouped_bar_chart(spec, all_series, title),
            "bar" => parse_bar_chart(spec, first, title),
            "line" => parse_line_chart(spec, first, title),
            "pie" => parse_pie_chart(first, title),
            _ => None,
        }
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

fn json_to_f64(v: &serde_json::Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "JSON numeric values converted to u64 for chart display"
)]
fn json_to_u64(v: &serde_json::Value) -> Option<u64> {
    v.as_u64().or_else(|| json_to_f64(v).map(|f| f as u64))
}

fn parse_bar_chart(
    spec: &serde_json::Value,
    series: &serde_json::Value,
    title: String,
) -> Option<ChartData> {
    let x_data = spec
        .get("xAxis")
        .and_then(|x| x.get("data"))
        .and_then(|d| d.as_array())?;
    let y_data = series.get("data").and_then(|d| d.as_array())?;

    let labels: Vec<String> = x_data
        .iter()
        .map(|v| v.as_str().map_or_else(|| v.to_string(), String::from))
        .collect();
    let values: Vec<u64> = y_data.iter().filter_map(json_to_u64).collect();

    Some(ChartData {
        title,
        kind: ChartKind::Bar { labels, values },
    })
}

fn parse_grouped_bar_chart(
    spec: &serde_json::Value,
    all_series: &[serde_json::Value],
    title: String,
) -> Option<ChartData> {
    let x_data = spec
        .get("xAxis")
        .and_then(|x| x.get("data"))
        .and_then(|d| d.as_array())?;

    let labels: Vec<String> = x_data
        .iter()
        .map(|v| v.as_str().map_or_else(|| v.to_string(), String::from))
        .collect();

    let mut series = Vec::new();
    for s in all_series {
        let name = s
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("series")
            .to_owned();
        let values: Vec<u64> = s
            .get("data")
            .and_then(|d| d.as_array())
            .map_or_else(Vec::new, |arr| arr.iter().filter_map(json_to_u64).collect());
        series.push(SeriesData { name, values });
    }

    Some(ChartData {
        title,
        kind: ChartKind::Grouped { labels, series },
    })
}

fn parse_line_chart(
    spec: &serde_json::Value,
    series: &serde_json::Value,
    title: String,
) -> Option<ChartData> {
    let y_data = series.get("data").and_then(|d| d.as_array())?;
    let y_values: Vec<f64> = y_data.iter().filter_map(json_to_f64).collect();

    if y_values.is_empty() {
        return None;
    }

    let x_data = spec
        .get("xAxis")
        .and_then(|x| x.get("data"))
        .and_then(|d| d.as_array());

    let x_labels: Vec<String> = x_data.map_or_else(Vec::new, |arr| {
        arr.iter()
            .map(|v| v.as_str().map_or_else(|| v.to_string(), String::from))
            .collect()
    });

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
    let y_max = y_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_pad = (y_max - y_min).abs() * 0.1;

    let y_labels = vec![
        format!("{:.0}", y_min),
        format!("{:.0}", f64::midpoint(y_min, y_max)),
        format!("{:.0}", y_max),
    ];

    Some(ChartData {
        title,
        kind: ChartKind::Line {
            points,
            x_bounds: [0.0, x_max],
            y_bounds: [y_min - y_pad, y_max + y_pad],
            x_labels,
            y_labels,
        },
    })
}

fn parse_pie_chart(series: &serde_json::Value, title: String) -> Option<ChartData> {
    let data = series.get("data").and_then(|d| d.as_array())?;
    let slices: Vec<(String, u64)> = data
        .iter()
        .filter_map(|item| {
            let name = item
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("?")
                .to_owned();
            let value = json_to_u64(item.get("value")?)?;
            Some((name, value))
        })
        .collect();

    Some(ChartData {
        title,
        kind: ChartKind::Pie { slices },
    })
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
    } = kind
    else {
        return;
    };

    let dataset = Dataset::default()
        .marker(symbols::Marker::Braille)
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
    use serde_json::json;

    #[test]
    #[expect(clippy::unwrap_used, clippy::panic, reason = "test assertion")]
    fn parse_bar_spec() {
        let spec = json!({
            "title": {"text": "Sales by Region"},
            "xAxis": {"data": ["North", "South", "East"]},
            "series": [{"type": "bar", "data": [100, 200, 150]}]
        });
        let chart = ChartData::from_echart_spec(&spec).unwrap();
        assert_eq!(chart.title, "Sales by Region");
        match &chart.kind {
            ChartKind::Bar { labels, values } => {
                assert_eq!(labels, &["North", "South", "East"]);
                assert_eq!(values, &[100, 200, 150]);
            }
            other => panic!("expected Bar, got {other:?}"),
        }
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test assertion")]
    fn parse_line_spec() {
        let spec = json!({
            "title": {"text": "Trend"},
            "xAxis": {"data": ["Jan", "Feb", "Mar"]},
            "series": [{"type": "line", "data": [10.0, 20.5, 15.3]}]
        });
        let chart = ChartData::from_echart_spec(&spec).unwrap();
        assert_eq!(chart.title, "Trend");
        assert!(matches!(chart.kind, ChartKind::Line { .. }));
        assert_eq!(chart.height(), 12);
    }

    #[test]
    #[expect(
        clippy::unwrap_used,
        clippy::panic,
        clippy::indexing_slicing,
        reason = "test assertion"
    )]
    fn parse_pie_spec() {
        let spec = json!({
            "title": {"text": "Market Share"},
            "series": [{"type": "pie", "data": [
                {"name": "A", "value": 60},
                {"name": "B", "value": 40}
            ]}]
        });
        let chart = ChartData::from_echart_spec(&spec).unwrap();
        assert_eq!(chart.title, "Market Share");
        match &chart.kind {
            ChartKind::Pie { slices } => {
                assert_eq!(slices.len(), 2);
                assert_eq!(slices[0], ("A".to_owned(), 60));
                assert_eq!(slices[1], ("B".to_owned(), 40));
            }
            other => panic!("expected Pie, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_type_returns_none() {
        let spec = json!({
            "series": [{"type": "radar", "data": [1, 2, 3]}]
        });
        assert!(ChartData::from_echart_spec(&spec).is_none());
    }

    #[test]
    fn missing_series_returns_none() {
        let spec = json!({"title": {"text": "Empty"}});
        assert!(ChartData::from_echart_spec(&spec).is_none());
    }

    #[test]
    fn area_chart_parsed_as_line() {
        let spec = json!({
            "title": {"text": "Area"},
            "xAxis": {"data": ["a", "b"]},
            "series": [{"type": "line", "areaStyle": {}, "data": [5, 10]}]
        });
        assert!(ChartData::from_echart_spec(&spec).is_some());
    }

    #[test]
    #[expect(
        clippy::unwrap_used,
        clippy::panic,
        clippy::indexing_slicing,
        reason = "test assertion"
    )]
    fn parse_grouped_bar_spec() {
        let spec = json!({
            "title": {"text": "Revenue vs Profit"},
            "xAxis": {"data": ["Q1", "Q2", "Q3", "Q4"]},
            "series": [
                {"type": "bar", "name": "Revenue", "data": [420, 480, 542, 607]},
                {"type": "bar", "name": "Profit", "data": [105, 120, 135, 152]}
            ]
        });
        let chart = ChartData::from_echart_spec(&spec).unwrap();
        assert_eq!(chart.title, "Revenue vs Profit");
        match &chart.kind {
            ChartKind::Grouped { labels, series } => {
                assert_eq!(labels, &["Q1", "Q2", "Q3", "Q4"]);
                assert_eq!(series.len(), 2);
                assert_eq!(series[0].name, "Revenue");
                assert_eq!(series[1].name, "Profit");
            }
            other => panic!("expected Grouped, got {other:?}"),
        }
    }

    #[test]
    #[expect(
        clippy::unwrap_used,
        clippy::print_stdout,
        reason = "visual demo test — run with --nocapture"
    )]
    fn render_all_chart_types() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let specs = [
            (
                "BAR CHART",
                json!({
                    "title": {"text": "Revenue by Region"},
                    "xAxis": {"data": ["North", "South", "East", "West"]},
                    "series": [{"type": "bar", "data": [175, 148, 162, 122]}]
                }),
            ),
            (
                "LINE CHART",
                json!({
                    "title": {"text": "Monthly Active Users"},
                    "xAxis": {"data": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]},
                    "series": [{"type": "line", "data": [12500, 13200, 14800, 15600,
                        16900, 18200, 19500, 20800, 21200, 22500, 24000, 25800]}]
                }),
            ),
            (
                "GROUPED BAR CHART (stacked)",
                json!({
                    "title": {"text": "Revenue vs Profit by Quarter"},
                    "xAxis": {"data": ["Q1", "Q2", "Q3", "Q4"]},
                    "series": [
                        {"type": "bar", "name": "Revenue", "data": [420, 480, 542, 607]},
                        {"type": "bar", "name": "Profit", "data": [105, 120, 135, 152]}
                    ]
                }),
            ),
            (
                "PIE CHART",
                json!({
                    "title": {"text": "Market Share"},
                    "series": [{"type": "pie", "data": [
                        {"name": "Product A", "value": 45},
                        {"name": "Product B", "value": 28},
                        {"name": "Product C", "value": 17},
                        {"name": "Other", "value": 10}
                    ]}]
                }),
            ),
        ];

        for (label, spec) in &specs {
            let chart_data = ChartData::from_echart_spec(spec).unwrap();
            let height = chart_data.height();
            let width = 80;

            let backend = TestBackend::new(width, height);
            let mut terminal = Terminal::new(backend).unwrap();

            terminal
                .draw(|frame| {
                    render_chart(frame, frame.area(), &chart_data);
                })
                .unwrap();

            let buf = terminal.backend().buffer().clone();
            println!("\n=== {label} ===");
            for y in 0..height {
                let mut line = String::new();
                for x in 0..width {
                    let cell = buf.cell((x, y)).unwrap();
                    line.push_str(cell.symbol());
                }
                println!("{}", line.trim_end());
            }
        }
    }
}
