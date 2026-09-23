use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

use crate::terminal::app::Message;
use crate::terminal::app::{App, MessageRole};
use crate::terminal::chart;
use crate::terminal::commands::Suggestion;
use quack_core::analysis::events;
use quack_core::jobs::{JobInfo, JobState};

/// Jobs listed above the input at most; the rest are counted.
const STRIP_JOBS: usize = 3;

/// Command popup entries shown at once; the list scrolls past them.
const POPUP_ROWS: usize = 8;

/// The widest a popup entry's label column grows before its description.
const POPUP_LABEL_WIDTH: usize = 28;

const SPINNER: &[&str] = &[
    "\u{280B}", "\u{2819}", "\u{2839}", "\u{2838}", "\u{283C}", "\u{2834}", "\u{2826}", "\u{2827}",
    "\u{2807}", "\u{280F}",
];

pub(crate) fn draw(frame: &mut Frame<'_>, app: &App) {
    let chart_height = app
        .current_chart
        .as_ref()
        .map_or(0, chart::ChartData::height);
    let strip = job_strip(app);
    let strip_height = u16::try_from(strip.len()).unwrap_or(u16::MAX);

    let [
        header_area,
        messages_area,
        chart_area,
        jobs_area,
        input_area,
        status_area,
    ] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(3),
        Constraint::Length(chart_height),
        Constraint::Length(strip_height),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    draw_header(frame, header_area, app);
    draw_messages(frame, messages_area, app);
    if let Some(chart_data) = &app.current_chart
        && chart_height > 0
    {
        chart::render_chart(frame, chart_area, chart_data);
    }
    if strip_height > 0 {
        frame.render_widget(Paragraph::new(Text::from(strip)), jobs_area);
    }
    draw_input(frame, input_area, app);
    draw_status(frame, status_area, app);
    draw_completion(frame, messages_area, jobs_area.y, app);
}

/// The command popup, drawn over the bottom of `area` so its last row sits
/// just above `bottom` (the job strip, or the input when no job runs).
fn draw_completion(frame: &mut Frame<'_>, area: Rect, bottom: u16, app: &App) {
    let Some(completion) = app.completion() else {
        return;
    };
    let lines = completion_lines(&completion.items, app.completion_selected());
    let rows = u16::try_from(lines.len()).unwrap_or(u16::MAX);
    let height = rows.saturating_add(2).min(bottom.saturating_sub(area.y));
    if height < 3 {
        return;
    }
    let width = lines
        .iter()
        .map(Line::width)
        .max()
        .and_then(|w| u16::try_from(w).ok())
        .unwrap_or(u16::MAX)
        .saturating_add(2)
        .min(area.width.saturating_sub(2));
    let popup = Rect {
        x: area.x.saturating_add(1),
        y: bottom.saturating_sub(height),
        width,
        height,
    };
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(Text::from(lines)).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        ),
        popup,
    );
}

/// The popup's visible rows: a window of [`POPUP_ROWS`] that keeps the
/// highlighted entry in view, each a label column and a dim description.
pub(crate) fn completion_lines(items: &[Suggestion], selected: usize) -> Vec<Line<'static>> {
    let selected = selected.min(items.len().saturating_sub(1));
    let first = selected.saturating_sub(POPUP_ROWS.saturating_sub(1));
    let label_width = items
        .iter()
        .map(|item| item.label.chars().count())
        .max()
        .unwrap_or(0)
        .min(POPUP_LABEL_WIDTH);
    let mut lines = Vec::new();
    for (index, item) in items.iter().enumerate().skip(first).take(POPUP_ROWS) {
        let (marker, label_style) = if index == selected {
            (
                "\u{25B8} ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            ("  ", Style::default())
        };
        lines.push(Line::from(vec![
            Span::styled(marker, label_style),
            Span::styled(format!("{:<label_width$}  ", item.label), label_style),
            Span::styled(item.about.clone(), Style::default().fg(Color::DarkGray)),
        ]));
    }
    lines
}

/// The strip above the input: one line per active job (running first),
/// with a spinner, its number, kind, label, and progress, and a count of
/// any beyond [`STRIP_JOBS`]. Empty when nothing is queued or running.
pub(crate) fn job_strip(app: &App) -> Vec<Line<'static>> {
    let mut jobs: Vec<&JobInfo> = app.active_jobs.iter().collect();
    jobs.sort_by_key(|j| (j.state != JobState::Running, j.number));
    let mut lines: Vec<Line<'static>> = jobs
        .iter()
        .take(STRIP_JOBS)
        .map(|job| {
            let (marker, style) = if job.state == JobState::Running {
                (spinner_frame(app.tick), Style::default().fg(Color::Yellow))
            } else {
                ("\u{00B7}", Style::default().fg(Color::DarkGray))
            };
            let mut detail = String::new();
            if let Some(p) = job.progress {
                detail = format!("  {p}");
            }
            if let Some(status) = job.status.as_deref() {
                detail.push_str("  ");
                detail.push_str(status);
            }
            if job.state == JobState::Queued {
                detail.push_str("  queued");
            }
            if job.cancel_requested {
                detail.push_str("  cancelling");
            }
            Line::from(vec![
                Span::styled(format!(" {marker} #{} ", job.number), style),
                Span::styled(
                    format!("{} ", job.kind),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::raw(job.label.clone()),
                Span::styled(detail, Style::default().fg(Color::DarkGray)),
            ])
        })
        .collect();
    let more = jobs.len().saturating_sub(STRIP_JOBS);
    if more > 0 {
        lines.push(Line::styled(
            format!("   and {more} more (/jobs)"),
            Style::default().fg(Color::DarkGray),
        ));
    }
    lines
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
        Span::styled(
            concat!(" v", env!("CARGO_PKG_VERSION")),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw("  "),
        Span::styled(&app.workspace_name, Style::default().fg(Color::Cyan)),
        Span::styled(" \u{00B7} ", Style::default().fg(Color::DarkGray)),
        Span::styled(&app.provider_display, Style::default().fg(Color::DarkGray)),
        Span::styled(" \u{00B7} session ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            app.session_id.chars().take(8).collect::<String>(),
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    frame.render_widget(Paragraph::new(title), title_area);
    frame.render_widget(Paragraph::new(separator_line(sep_area.width)), sep_area);
}

fn draw_messages(frame: &mut Frame<'_>, area: Rect, app: &App) {
    // Lines are wrapped here, to the real width, so the scroll range is
    // computed on what is drawn and the newest content is reachable
    // (issue #49); the paragraph itself does no wrapping.
    let lines = format_messages(app, usize::from(area.width));
    let total_lines = lines.len();
    let visible = usize::from(area.height);
    let max_scroll = total_lines.saturating_sub(visible);
    let effective_scroll = max_scroll.saturating_sub(app.scroll_offset.min(max_scroll));
    let scroll_u16 = u16::try_from(effective_scroll).unwrap_or(u16::MAX);

    let text = Text::from(lines);
    let paragraph = Paragraph::new(text).scroll((scroll_u16, 0));

    frame.render_widget(paragraph, area);
}

fn draw_input(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.awaiting_permission() {
        let status_line = Line::from(vec![
            Span::raw(" "),
            Span::styled(
                "Run this statement?",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "  [y] run   [n] refuse   [a] run and allow writes this session",
                Style::default().fg(Color::Yellow),
            ),
        ]);
        frame.render_widget(Paragraph::new(status_line), inner);
    } else {
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
            " \u{00B7} \u{2191}\u{2193}",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" history", Style::default().fg(Color::DarkGray)),
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
        Span::styled(jobs_indicator(app), Style::default().fg(Color::DarkGray)),
    ]);

    frame.render_widget(Paragraph::new(status), area);
}

/// ` · 2 running, 1 queued · /jobs`, or nothing when idle.
fn jobs_indicator(app: &App) -> String {
    let running = app
        .active_jobs
        .iter()
        .filter(|j| j.state == JobState::Running)
        .count();
    let queued = app.active_jobs.len().saturating_sub(running);
    match (running, queued) {
        (0, 0) => String::new(),
        (r, 0) => format!(" \u{00B7} {r} running \u{00B7} /jobs"),
        (r, q) => format!(" \u{00B7} {r} running, {q} queued \u{00B7} /jobs"),
    }
}

/// A message's rendered lines and the fingerprint they were rendered from.
pub(crate) type WrappedMessage = (u64, Vec<Line<'static>>);

/// Every message's lines, wrapped to `width`. Each message's lines are
/// cached on the app by a fingerprint of what it shows, so a redraw
/// re-renders only the messages that changed (the one streaming, as a
/// rule), not the Markdown and wrapping of the whole transcript.
pub(crate) fn format_messages(app: &App, width: usize) -> Vec<Line<'static>> {
    let mut cache = app.wrap_cache.borrow_mut();
    // Messages are appended, edited in place, or cleared; never removed
    // from the middle, so the cache stays aligned by index.
    cache.truncate(app.messages.len());
    cache.resize(app.messages.len(), None);
    let mut lines: Vec<Line<'static>> = Vec::new();
    for (msg, slot) in app.messages.iter().zip(cache.iter_mut()) {
        let key = fingerprint(msg, width, app.expand_steps);
        match slot {
            Some((cached, rendered)) if *cached == key => lines.extend(rendered.iter().cloned()),
            _ => {
                let rendered = message_lines(msg, width, app.expand_steps);
                lines.extend(rendered.iter().cloned());
                *slot = Some((key, rendered));
            }
        }
    }
    lines
}

/// What decides a message's rendering.
fn fingerprint(msg: &Message, width: usize, expand: bool) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    msg.role.hash(&mut hasher);
    msg.content.hash(&mut hasher);
    msg.detail.hash(&mut hasher);
    msg.chart
        .as_ref()
        .map(|c| c.title.as_str())
        .hash(&mut hasher);
    width.hash(&mut hasher);
    (expand && msg.role == MessageRole::Step).hash(&mut hasher);
    hasher.finish()
}

/// One message's lines, wrapped to `width`, with the blank line after it.
fn message_lines(msg: &Message, width: usize, expand_steps: bool) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    {
        let (prefix, style) = match msg.role {
            MessageRole::User => (" > ", Style::default().fg(Color::Cyan)),
            MessageRole::Assistant => ("   ", Style::default()),
            MessageRole::Step => ("   ", Style::default().fg(Color::Yellow)),
            MessageRole::Sql => ("   ", Style::default().fg(Color::White)),
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
        let body: Vec<Vec<Span<'static>>> = match msg.role {
            MessageRole::Assistant => markdown::render(&msg.content),
            MessageRole::Step => step_body(msg, expand_steps, style),
            _ => msg
                .content
                .lines()
                .map(|l| vec![Span::styled(l.to_owned(), style)])
                .collect(),
        };
        let mut first = true;
        for spans in body {
            let p = if first {
                first = false;
                prefix
            } else {
                "   "
            };
            for row in wrap::wrap(&spans, width.saturating_sub(p.chars().count())) {
                let mut with_prefix = vec![Span::styled(p.to_owned(), prompt_style)];
                with_prefix.extend(row);
                lines.push(Line::from(with_prefix));
            }
        }
        if let Some(chart) = &msg.chart {
            let note = vec![Span::styled(
                format!("[chart: {}; /chart shows it]", chart.title),
                Style::default().fg(Color::Magenta),
            )];
            for row in wrap::wrap(&note, width.saturating_sub(3)) {
                let mut with_prefix = vec![Span::raw("   ")];
                with_prefix.extend(row);
                lines.push(Line::from(with_prefix));
            }
        }
        lines.push(Line::from(""));
    }

    lines
}

/// A step: its header and outcome, then the detail in full when expanded
/// or its first lines with a count of the rest.
fn step_body(msg: &Message, expanded: bool, style: Style) -> Vec<Vec<Span<'static>>> {
    let mut rows: Vec<Vec<Span<'static>>> = msg
        .content
        .lines()
        .map(|l| vec![Span::styled(l.to_owned(), style)])
        .collect();
    let Some(detail) = msg.detail.as_deref().filter(|d| !d.trim().is_empty()) else {
        return rows;
    };
    let dim = Style::default().fg(Color::DarkGray);
    let (shown, more): (Vec<&str>, usize) = if expanded {
        (detail.lines().collect(), 0)
    } else {
        events::preview_detail(detail)
    };
    let at = rows.len().min(1);
    let mut detail_rows: Vec<Vec<Span<'static>>> = shown
        .into_iter()
        .map(|l| vec![Span::styled(format!("  {l}"), dim)])
        .collect();
    if more > 0 {
        detail_rows.push(vec![Span::styled(
            format!("  ({more} more lines; /steps expands)"),
            dim,
        )]);
    }
    rows.splice(at..at, detail_rows);
    rows
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

/// Wrapping of styled spans to a width, by characters, breaking at the
/// last space when one is near.
pub(crate) mod wrap {
    use ratatui::style::Style;
    use ratatui::text::Span;

    pub(crate) fn wrap(spans: &[Span<'static>], width: usize) -> Vec<Vec<Span<'static>>> {
        let width = width.max(1);
        let chars: Vec<(char, Style)> = spans
            .iter()
            .flat_map(|s| {
                let style = s.style;
                s.content
                    .chars()
                    .map(move |c| (c, style))
                    .collect::<Vec<_>>()
            })
            .collect();
        if chars.is_empty() {
            return vec![Vec::new()];
        }
        let mut rows = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let mut end = start.saturating_add(width).min(chars.len());
            if end < chars.len() {
                // Prefer the last space in the row, if it is not too early.
                if let Some(space) = chars
                    .get(start..end)
                    .and_then(|row| row.iter().rposition(|(c, _)| *c == ' '))
                    .filter(|pos| pos.saturating_mul(2) > width)
                {
                    end = start.saturating_add(space).saturating_add(1);
                }
            }
            rows.push(regroup(chars.get(start..end).unwrap_or(&[])));
            start = end;
        }
        rows
    }

    /// Consecutive characters of one style back into spans.
    fn regroup(chars: &[(char, Style)]) -> Vec<Span<'static>> {
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut current = String::new();
        let mut current_style: Option<Style> = None;
        for (c, style) in chars {
            if current_style != Some(*style) {
                if let Some(s) = current_style
                    && !current.is_empty()
                {
                    spans.push(Span::styled(std::mem::take(&mut current), s));
                }
                current_style = Some(*style);
            }
            current.push(*c);
        }
        if let Some(s) = current_style
            && !current.is_empty()
        {
            spans.push(Span::styled(current, s));
        }
        spans
    }
}

/// A light Markdown rendering for the transcript: headings bold, bullets
/// as dots, fenced code dim and verbatim, `**bold**`, `*italic*`, and
/// `` `code` `` inline. Tables pass through as their source lines.
pub(crate) mod markdown {
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::Span;

    pub(crate) fn render(content: &str) -> Vec<Vec<Span<'static>>> {
        let mut rows = Vec::new();
        let mut in_fence = false;
        for line in content.lines() {
            if line.trim_start().starts_with("```") {
                in_fence = !in_fence;
                continue;
            }
            if in_fence {
                rows.push(vec![Span::styled(
                    format!("  {line}"),
                    Style::default().fg(Color::Cyan),
                )]);
                continue;
            }
            let trimmed = line.trim_start();
            if let Some(heading) = trimmed.strip_prefix('#') {
                let text = heading.trim_start_matches('#').trim();
                rows.push(vec![Span::styled(
                    text.to_owned(),
                    Style::default().add_modifier(Modifier::BOLD),
                )]);
                continue;
            }
            let indent = line.len().saturating_sub(trimmed.len());
            let (bullet, rest) = match trimmed
                .strip_prefix("- ")
                .or_else(|| trimmed.strip_prefix("* "))
            {
                Some(rest) => ("\u{2022} ", rest),
                None => ("", trimmed),
            };
            let mut spans = Vec::new();
            if indent > 0 || !bullet.is_empty() {
                spans.push(Span::raw(format!("{}{bullet}", " ".repeat(indent))));
            }
            spans.extend(inline(rest));
            rows.push(spans);
        }
        if rows.is_empty() {
            rows.push(Vec::new());
        }
        rows
    }

    /// Inline `**bold**`, `*italic*`, and `` `code` `` runs.
    fn inline(text: &str) -> Vec<Span<'static>> {
        let mut spans = Vec::new();
        let mut plain = String::new();
        let mut rest = text;
        while !rest.is_empty() {
            if let Some(after) = rest.strip_prefix("**")
                && let Some(end) = after.find("**")
            {
                flush(&mut spans, &mut plain, Style::default());
                spans.push(Span::styled(
                    after.get(..end).unwrap_or("").to_owned(),
                    Style::default().add_modifier(Modifier::BOLD),
                ));
                rest = after.get(end.saturating_add(2)..).unwrap_or("");
                continue;
            }
            if let Some(after) = rest.strip_prefix('`')
                && let Some(end) = after.find('`')
            {
                flush(&mut spans, &mut plain, Style::default());
                spans.push(Span::styled(
                    after.get(..end).unwrap_or("").to_owned(),
                    Style::default().fg(Color::Cyan),
                ));
                rest = after.get(end.saturating_add(1)..).unwrap_or("");
                continue;
            }
            if let Some(after) = rest.strip_prefix('*')
                && !after.starts_with(' ')
                && let Some(end) = after.find('*')
                && end > 0
            {
                flush(&mut spans, &mut plain, Style::default());
                spans.push(Span::styled(
                    after.get(..end).unwrap_or("").to_owned(),
                    Style::default().add_modifier(Modifier::ITALIC),
                ));
                rest = after.get(end.saturating_add(1)..).unwrap_or("");
                continue;
            }
            let mut chars = rest.chars();
            if let Some(c) = chars.next() {
                plain.push(c);
            }
            rest = chars.as_str();
        }
        flush(&mut spans, &mut plain, Style::default());
        spans
    }

    fn flush(spans: &mut Vec<Span<'static>>, plain: &mut String, style: Style) {
        if !plain.is_empty() {
            spans.push(Span::styled(std::mem::take(plain), style));
        }
    }
}

#[cfg(test)]
mod tests {
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
    fn markdown_renders_headings_bullets_fences_and_inline_marks() {
        let rows = markdown::render(
            "## Deadliest\n- **Tornado** in `Texas`\n```sql\nSELECT 1\n```\n| a | b |",
        );
        let lines = text_of(&rows);
        assert_eq!(
            lines,
            [
                "Deadliest",
                "\u{2022} Tornado in Texas",
                "  SELECT 1",
                "| a | b |"
            ]
        );
        assert!(
            rows.first()
                .and_then(|r| r.first())
                .is_some_and(|s| s.style.add_modifier.contains(Modifier::BOLD))
        );
        assert!(
            rows.get(1)
                .and_then(|r| r.get(1))
                .is_some_and(|s| s.style.add_modifier.contains(Modifier::BOLD))
        );
    }
}
