use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::app::{App, AppState, MessageRole};

const SPINNER: &[&str] = &[
    "\u{280B}", "\u{2819}", "\u{2839}", "\u{2838}", "\u{283C}", "\u{2834}", "\u{2826}", "\u{2827}",
    "\u{2807}", "\u{280F}",
];

pub(crate) fn draw(frame: &mut Frame<'_>, app: &App) {
    let [header_area, messages_area, input_area, status_area] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(3),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    draw_header(frame, header_area, app);
    draw_messages(frame, messages_area, app);
    draw_input(frame, input_area, app);
    draw_status(frame, status_area, app);
}

fn draw_header(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let [title_area, sep_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(area);

    let title = Line::from(vec![
        Span::styled(
            " quack",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" v0.1.0", Style::default().fg(Color::DarkGray)),
        Span::raw("  "),
        Span::styled(&app.workspace_name, Style::default().fg(Color::Cyan)),
        Span::styled(" \u{00B7} ", Style::default().fg(Color::DarkGray)),
        Span::styled(&app.provider_display, Style::default().fg(Color::DarkGray)),
    ]);

    frame.render_widget(Paragraph::new(title), title_area);
    frame.render_widget(Paragraph::new(separator_line(sep_area.width)), sep_area);
}

fn draw_messages(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let lines = format_messages(app);
    let total_lines = lines.len();
    let visible = usize::from(area.height);
    let max_scroll = total_lines.saturating_sub(visible);
    let effective_scroll = max_scroll.saturating_sub(app.scroll_offset);
    let scroll_u16 = u16::try_from(effective_scroll).unwrap_or(u16::MAX);

    let text = Text::from(lines);
    let paragraph = Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .scroll((scroll_u16, 0));

    frame.render_widget(paragraph, area);
}

fn draw_input(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.state == AppState::Idle {
        let [prompt_area, text_area] =
            Layout::horizontal([Constraint::Length(3), Constraint::Min(1)]).areas(inner);

        let prompt = Paragraph::new(Line::from(vec![Span::styled(
            " > ",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )]));
        frame.render_widget(prompt, prompt_area);
        frame.render_widget(&app.textarea, text_area);
    } else {
        let frame_str = spinner_frame(app.tick);
        let label = match app.state {
            AppState::Thinking => "Thinking...",
            AppState::Ingesting => "Ingesting file...",
            AppState::Idle => "",
        };
        let status_line = Line::from(vec![
            Span::raw(" "),
            Span::styled(
                format!("{frame_str} {label}"),
                Style::default().fg(Color::Yellow),
            ),
        ]);
        frame.render_widget(Paragraph::new(status_line), inner);
    }
}

fn draw_status(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let scroll_indicator = if app.scroll_offset > 0 {
        format!(" \u{00B7} scroll: +{}", app.scroll_offset)
    } else {
        String::new()
    };

    let status = Line::from(vec![
        Span::styled(
            " enter",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" send", Style::default().fg(Color::DarkGray)),
        Span::styled(
            " \u{00B7} ctrl+l",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" clear", Style::default().fg(Color::DarkGray)),
        Span::styled(
            " \u{00B7} ctrl+c",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" quit", Style::default().fg(Color::DarkGray)),
        Span::styled(scroll_indicator, Style::default().fg(Color::DarkGray)),
    ]);

    frame.render_widget(Paragraph::new(status), area);
}

fn format_messages(app: &App) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    for msg in &app.messages {
        let (prefix, style) = match msg.role {
            MessageRole::User => (" > ", Style::default().fg(Color::Cyan)),
            MessageRole::Assistant => ("   ", Style::default()),
            MessageRole::System => ("   ", Style::default().fg(Color::DarkGray)),
            MessageRole::Error => ("   ", Style::default().fg(Color::Red)),
        };

        let prompt_style = if msg.role == MessageRole::User {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        } else {
            style
        };

        let mut first = true;
        for line_text in msg.content.lines() {
            let p = if first {
                first = false;
                prefix
            } else if msg.role == MessageRole::User {
                "   "
            } else {
                prefix
            };

            lines.push(Line::from(vec![
                Span::styled(p.to_owned(), prompt_style),
                Span::styled(line_text.to_owned(), style),
            ]));
        }

        lines.push(Line::from(""));
    }

    if app.state != AppState::Idle {
        let frame_str = spinner_frame(app.tick);
        let label = match app.state {
            AppState::Thinking => "Thinking...",
            AppState::Ingesting => "Ingesting file...",
            AppState::Idle => "",
        };
        lines.push(Line::from(vec![
            Span::raw("   "),
            Span::styled(
                format!("{frame_str} {label}"),
                Style::default().fg(Color::Yellow),
            ),
        ]));
    }

    lines
}

#[expect(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "SPINNER is a non-empty const array; divisor and index are always valid"
)]
fn spinner_frame(tick: usize) -> &'static str {
    SPINNER[tick % SPINNER.len()]
}

fn separator_line(width: u16) -> Line<'static> {
    let w = usize::from(width);
    Line::styled("\u{2500}".repeat(w), Style::default().fg(Color::DarkGray))
}
