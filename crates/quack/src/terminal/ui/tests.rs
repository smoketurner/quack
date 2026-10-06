use super::*;

fn text_of(rows: &[Vec<Span<'static>>]) -> Vec<String> {
    rows.iter()
        .map(|r| r.iter().map(|s| s.content.to_string()).collect())
        .collect()
}

#[test]
fn wrapping_breaks_at_spaces_and_keeps_every_character() {
    let rows = wrap::wrap(&[Span::raw("the quick brown fox jumps over")], 10);
    let lines = text_of(&rows);
    assert_eq!(lines, ["the quick ", "brown fox ", "jumps over"]);
    let long = wrap::wrap(&[Span::raw("abcdefghijkl")], 5);
    assert_eq!(text_of(&long), ["abcde", "fghij", "kl"]);
    assert_eq!(wrap::wrap(&[], 5).len(), 1, "an empty line is one row");
}

#[test]
fn wrapping_counts_columns_and_keeps_graphemes_whole() {
    // Each of these takes two columns, so two fit in four.
    let wide = wrap::wrap(&[Span::raw("\u{65E5}\u{672C}\u{8A9E}\u{65E5}\u{672C}")], 4);
    assert_eq!(
        text_of(&wide),
        ["\u{65E5}\u{672C}", "\u{8A9E}\u{65E5}", "\u{672C}"]
    );
    // A combining accent takes no column and stays with its letter.
    let accented = wrap::wrap(&[Span::raw("e\u{301}e\u{301}e\u{301}")], 2);
    assert_eq!(text_of(&accented), ["e\u{301}e\u{301}", "e\u{301}"]);
    // A grapheme wider than the row still takes one, so wrapping ends.
    let narrow = wrap::wrap(&[Span::raw("\u{65E5}\u{672C}")], 1);
    assert_eq!(text_of(&narrow), ["\u{65E5}", "\u{672C}"]);
    // A style change inside a row keeps both spans.
    let bold = Style::default().add_modifier(Modifier::BOLD);
    let styled = wrap::wrap(&[Span::raw("ab"), Span::styled("cd", bold)], 3);
    assert_eq!(text_of(&styled), ["abc", "d"]);
    assert!(
        styled
            .first()
            .and_then(|row| row.get(1))
            .is_some_and(|span| span.style == bold)
    );
}

#[test]
fn chart_fingerprint_covers_the_plot_not_just_the_title() {
    use crate::terminal::chart::ChartData;
    use quack_core::analysis::chart::{Axis as SpecAxis, ChartKind, ChartSpec, Series};

    fn spec(kind: ChartKind) -> ChartSpec {
        ChartSpec {
            title: String::from("Sales"),
            kind,
            x: SpecAxis {
                label: String::from("x"),
                values: vec![String::from("A"), String::from("B")],
            },
            series: vec![Series {
                name: String::from("s"),
                values: vec![60.0, 40.0],
            }],
        }
    }
    let width = 60usize;
    let chart_w = u16::try_from(width.saturating_sub(3)).unwrap_or(u16::MAX);
    let bar = ChartData::from_spec(&spec(ChartKind::Bar));
    let pie = ChartData::from_spec(&spec(ChartKind::Pie));
    assert_eq!(bar.title, pie.title);
    let bar_text: String = bar
        .lines(chart_w)
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    let pie_text: String = pie
        .lines(chart_w)
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(pie_text.contains('\u{25A0}'), "pie renders a legend square");
    assert!(!bar_text.contains('\u{25A0}'), "bar does not render one");
    assert_ne!(bar_text, pie_text);

    let make = |chart| Message {
        kind: MessageKind::Assistant,
        content: String::from("Here is the chart."),
        chart: Some(chart),
        detail: None,
    };
    let m_bar = make(bar.clone());
    let m_pie = make(pie.clone());
    assert_ne!(
        m_bar.fingerprint(width, false),
        m_pie.fingerprint(width, false),
        "fingerprints must differ when the plot differs",
    );

    let mut m = make(bar);
    let key_bar = m.fingerprint(width, false);
    m.chart = Some(pie);
    let key_pie = m.fingerprint(width, false);
    assert_ne!(key_bar, key_pie);
    assert_eq!(m_bar.fingerprint(width, false), key_bar);
    assert_eq!(m_pie.fingerprint(width, false), key_pie);

    let line = ChartData::from_spec(&spec(ChartKind::Line));
    let scatter = ChartData::from_spec(&spec(ChartKind::Scatter));
    assert_eq!(line.title, scatter.title);
    assert_ne!(
        make(line).fingerprint(width, false),
        make(scatter).fingerprint(width, false),
        "line and scatter differ only in graph_type",
    );
}
