use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

use jiff::Timestamp;
use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, Borders, Clear, HighlightSpacing, LineGauge, List, ListItem, ListState, Paragraph,
    Scrollbar, ScrollbarOrientation, ScrollbarState, StatefulWidget, Widget,
};

use crate::terminal::app::{App, Message, MessageKind, PendingWrite};
use crate::terminal::clipboard::CopyStatus;
use crate::terminal::commands::Suggestion;
use crate::terminal::markdown;
use crate::terminal::selection::{Row, TranscriptView, Wrap};
use quack_core::analysis::events::DetailPreview;
use quack_core::jobs::{JobCounts, JobInfo, JobState};

/// Jobs listed above the input at most; the rest are counted.
const STRIP_JOBS: usize = 3;

/// Command popup entries shown at once; the list scrolls past them.
const POPUP_ROWS: usize = 8;

/// The widest a popup entry's label column grows before its description.
const POPUP_LABEL_WIDTH: usize = 28;

/// The marker beside the popup's highlighted entry.
const POPUP_MARKER: &str = "\u{25B8} ";

/// Columns of the progress line in a job's strip row.
const GAUGE_WIDTH: u16 = 16;

const SPINNER: &[&str] = &[
    "\u{280B}", "\u{2819}", "\u{2839}", "\u{2838}", "\u{283C}", "\u{2834}", "\u{2826}", "\u{2827}",
    "\u{2807}", "\u{280F}",
];

/// The spinner's frame, advanced on each tick while a job is active.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Spinner(usize);

impl Spinner {
    pub(crate) fn advance(&mut self) {
        let next = self.0.saturating_add(1);
        self.0 = if next < SPINNER.len() { next } else { 0 };
    }

    fn symbol(self) -> &'static str {
        SPINNER.get(self.0).copied().unwrap_or_default()
    }
}

/// Where the transcript is scrolled to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Scroll {
    /// Following the newest line.
    #[default]
    Latest,
    /// This many lines above the newest.
    Back(usize),
    /// At the first line.
    Top,
}

impl Scroll {
    pub(crate) fn up(self, lines: usize) -> Self {
        match self {
            Self::Latest => Self::Back(lines),
            Self::Back(back) => Self::Back(back.saturating_add(lines)),
            Self::Top => Self::Top,
        }
    }

    /// `lines` further down, when the transcript scrolls at most `limit`
    /// lines back.
    pub(crate) fn down(self, lines: usize, limit: usize) -> Self {
        match self.lines_back(limit).saturating_sub(lines) {
            0 => Self::Latest,
            back => Self::Back(back),
        }
    }

    /// How many lines above the newest the view starts, at most `limit`.
    fn lines_back(self, limit: usize) -> usize {
        match self {
            Self::Latest => 0,
            Self::Back(back) => back.min(limit),
            Self::Top => limit,
        }
    }
}

/// ` · 2 running, 1 queued · /jobs`, or nothing when idle.
struct JobsIndicator(JobCounts);

impl fmt::Display for JobsIndicator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.0.running, self.0.queued) {
            (0, 0) => Ok(()),
            (running, 0) => write!(f, " \u{00B7} {running} running \u{00B7} /jobs"),
            (running, queued) => {
                write!(
                    f,
                    " \u{00B7} {running} running, {queued} queued \u{00B7} /jobs"
                )
            }
        }
    }
}

/// One job, as `/jobs` lists it and as the strip above the input shows it.
pub(crate) struct JobRow<'a>(pub(crate) &'a JobInfo);

impl JobRow<'_> {
    /// Everything on record about it: number, state, kind, label, and
    /// progress, then its latest status and how it ended.
    pub(crate) fn details(&self) -> String {
        let job = self.0;
        let progress = job.progress.map(|p| format!(" {p}")).unwrap_or_default();
        let mut text = format!(
            "Job #{} {} {} {}{progress}",
            job.number, job.state, job.kind, job.label
        );
        for extra in [job.status.as_deref(), job.outcome.as_deref()]
            .into_iter()
            .flatten()
            .filter(|extra| !extra.trim().is_empty())
        {
            text.push('\n');
            text.push_str(extra);
        }
        text
    }

    /// Its strip row: a spinner while it runs, its number, kind, and label,
    /// a gauge of its progress, its status, and `phase`, what a running
    /// turn is doing.
    fn render_strip(&self, spinner: Spinner, phase: Option<&str>, area: Rect, buf: &mut Buffer) {
        let job = self.0;
        let dim = Style::default().fg(Color::DarkGray);
        let (marker, style) = if job.state == JobState::Running {
            (spinner.symbol(), Style::default().fg(Color::Yellow))
        } else {
            ("\u{00B7}", dim)
        };
        let head = Line::from(vec![
            Span::styled(format!(" {marker} #{} ", job.number), style),
            Span::styled(format!("{} ", job.kind), dim),
            Span::raw(job.label.clone()),
        ]);
        let mut detail = String::new();
        if let Some(status) = job.status.as_deref() {
            detail.push_str("  ");
            detail.push_str(status);
        }
        if let Some(phase) = phase.filter(|_| job.state == JobState::Running) {
            detail.push_str("  ");
            detail.push_str(phase);
        }
        if job.state == JobState::Queued {
            detail.push_str("  queued");
        }
        if job.cancel_requested {
            detail.push_str("  cancelling");
        }
        let gauge = job.progress.map(|progress| {
            LineGauge::default()
                .ratio(progress.ratio())
                .label(Line::styled(format!("  {progress}"), dim))
                .filled_symbol(symbols::line::THICK.horizontal)
                .unfilled_symbol(symbols::line::NORMAL.horizontal)
                .filled_style(style)
                .unfilled_style(dim)
        });
        let gauge_width = job.progress.map_or(0, |progress| {
            // Its label, the gap `LineGauge` leaves after it, and the line.
            u16::try_from(progress.to_string().len())
                .unwrap_or(u16::MAX)
                .saturating_add(3)
                .saturating_add(GAUGE_WIDTH)
        });
        let [head_area, gauge_area, detail_area] = Layout::horizontal([
            Constraint::Length(u16::try_from(head.width()).unwrap_or(u16::MAX)),
            Constraint::Length(gauge_width),
            Constraint::Fill(1),
        ])
        .areas(area);
        head.render(head_area, buf);
        if let Some(gauge) = gauge {
            gauge.render(gauge_area, buf);
        }
        Line::styled(detail, dim).render(detail_area, buf);
    }
}

/// The first line of `text`, cut to fit a job list.
pub(crate) fn one_line(text: &str) -> String {
    const MAX: usize = 60;
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    if line.chars().count() > MAX {
        let cut: String = line.chars().take(MAX.saturating_sub(1)).collect();
        format!("{cut}\u{2026}")
    } else {
        line.to_owned()
    }
}

pub(crate) fn draw(frame: &mut Frame<'_>, app: &App) {
    let strip = JobStrip::of(app);
    let strip_height = strip.height();
    let screen = frame.area();
    let permission = app.pending_write().map(|pending| {
        pending.lines(
            usize::from(screen.width),
            usize::from(screen.height.checked_div(3).unwrap_or_default()).max(1),
        )
    });
    // The top border plus the overlay's rows, or the one-line input.
    let input_height = permission.as_ref().map_or(3, |lines| {
        u16::try_from(lines.len())
            .unwrap_or(u16::MAX)
            .saturating_add(1)
    });

    let [
        header_area,
        messages_area,
        jobs_area,
        input_area,
        status_area,
    ] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(3),
        Constraint::Length(strip_height),
        Constraint::Length(input_height),
        Constraint::Length(1),
    ])
    .areas(screen);

    draw_header(frame, header_area, app);
    draw_messages(frame, messages_area, app);
    if strip_height > 0 {
        frame.render_widget(&strip, jobs_area);
    }
    draw_input(frame, input_area, app, permission);
    draw_status(frame, status_area, app);
    draw_completion(frame, messages_area, jobs_area.y, app);
    draw_picker(frame, messages_area, jobs_area.y, app);
}

/// The `/jobs` or `/sessions` box, drawn over `area` so its last row sits
/// just above `bottom`, like the command popup.
fn draw_picker(frame: &mut Frame<'_>, area: Rect, bottom: u16, app: &App) {
    let Some(picker) = &app.picker else {
        return;
    };
    let height = picker.height().min(bottom.saturating_sub(area.y));
    let outer = Rect {
        x: area.x,
        y: bottom.saturating_sub(height),
        width: area.width,
        height,
    };
    frame.render_widget(picker, outer);
}

/// The command popup, drawn over the bottom of `area` so its last row sits
/// just above `bottom` (the job strip, or the input when no job runs).
fn draw_completion(frame: &mut Frame<'_>, area: Rect, bottom: u16, app: &App) {
    let Some(completion) = app.completion() else {
        return;
    };
    let popup = CompletionPopup::new(&completion.items, app.completion_selected());
    let height = popup
        .rows()
        .saturating_add(2)
        .min(bottom.saturating_sub(area.y));
    if height < 3 {
        return;
    }
    let outer = Rect {
        x: area.x.saturating_add(1),
        y: bottom.saturating_sub(height),
        width: popup
            .width()
            .saturating_add(2)
            .min(area.width.saturating_sub(2)),
        height,
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(outer);
    frame.render_widget(Clear, outer);
    frame.render_widget(block, outer);
    frame.render_widget(&popup, inner);
}

/// The command popup's entries, each a label column and a dim description,
/// the highlighted one marked. It shows [`POPUP_ROWS`] at once and scrolls
/// to keep the highlighted one in view.
pub(crate) struct CompletionPopup<'a> {
    items: &'a [Suggestion],
    selected: usize,
}

impl<'a> CompletionPopup<'a> {
    pub(crate) fn new(items: &'a [Suggestion], selected: usize) -> Self {
        Self {
            items,
            selected: selected.min(items.len().saturating_sub(1)),
        }
    }

    fn label_width(&self) -> usize {
        self.items
            .iter()
            .map(|item| item.label.chars().count())
            .max()
            .unwrap_or(0)
            .min(POPUP_LABEL_WIDTH)
    }

    fn entry(&self, index: usize, item: &Suggestion) -> Line<'static> {
        let label_width = self.label_width();
        let label_style = if index == self.selected {
            Self::highlight()
        } else {
            Style::default()
        };
        Line::from(vec![
            Span::styled(format!("{:<label_width$}  ", item.label), label_style),
            Span::styled(item.about.clone(), Style::default().fg(Color::DarkGray)),
        ])
    }

    fn highlight() -> Style {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    }

    fn entries(&self) -> impl Iterator<Item = Line<'static>> {
        self.items
            .iter()
            .enumerate()
            .map(|(index, item)| self.entry(index, item))
    }

    /// Columns its widest entry takes, marker included.
    fn width(&self) -> u16 {
        let widest = self.entries().map(|line| line.width()).max().unwrap_or(0);
        u16::try_from(widest.saturating_add(Span::raw(POPUP_MARKER).width())).unwrap_or(u16::MAX)
    }

    fn rows(&self) -> u16 {
        u16::try_from(self.items.len().min(POPUP_ROWS)).unwrap_or(u16::MAX)
    }
}

impl Widget for &CompletionPopup<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let list = List::new(self.entries().map(ListItem::new))
            .highlight_symbol(Line::styled(POPUP_MARKER, CompletionPopup::highlight()))
            .highlight_spacing(HighlightSpacing::Always);
        let mut state = ListState::default().with_selected(Some(self.selected));
        StatefulWidget::render(list, area, buf, &mut state);
    }
}

/// The strip above the input: one row per active job (running first),
/// with a spinner, its number, kind, label, and progress, and a count of
/// any beyond [`STRIP_JOBS`]. No rows when nothing is queued or running.
pub(crate) struct JobStrip<'a> {
    /// Each job with what its turn is doing, when it is one.
    jobs: Vec<(&'a JobInfo, Option<String>)>,
    spinner: Spinner,
}

impl<'a> JobStrip<'a> {
    pub(crate) fn of(app: &'a App) -> Self {
        let mut jobs: Vec<&JobInfo> = app.active_jobs.iter().collect();
        jobs.sort_by_key(|j| (j.state != JobState::Running, j.number));
        let now = Timestamp::now();
        Self {
            jobs: jobs
                .into_iter()
                .map(|job| {
                    let phase = app.phase_of(job).map(|p| p.note(job.started_at, now));
                    (job, phase)
                })
                .collect(),
            spinner: app.spinner,
        }
    }

    /// Jobs beyond the ones listed.
    fn more(&self) -> usize {
        self.jobs.len().saturating_sub(STRIP_JOBS)
    }

    pub(crate) fn height(&self) -> u16 {
        let listed = self.jobs.len().min(STRIP_JOBS);
        u16::try_from(listed.saturating_add(usize::from(self.more() > 0))).unwrap_or(u16::MAX)
    }
}

impl Widget for &JobStrip<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut rows = area.rows();
        for ((job, phase), row) in self.jobs.iter().take(STRIP_JOBS).zip(rows.by_ref()) {
            JobRow(job).render_strip(self.spinner, phase.as_deref(), row, buf);
        }
        let more = self.more();
        if more > 0
            && let Some(row) = rows.next()
        {
            Line::styled(
                format!("   and {more} more (/jobs)"),
                Style::default().fg(Color::DarkGray),
            )
            .render(row, buf);
        }
    }
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
            app.session_id.short().to_owned(),
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    frame.render_widget(Paragraph::new(title), title_area);
    frame.render_widget(Paragraph::new(separator_line(sep_area.width)), sep_area);
}

fn draw_messages(frame: &mut Frame<'_>, area: Rect, app: &App) {
    // The scrollbar's column is always kept, so the transcript does not
    // re-wrap when it first outgrows the screen.
    let [text_area, bar_area] =
        Layout::horizontal([Constraint::Min(1), Constraint::Length(1)]).areas(area);
    // Lines are wrapped here, to the real width, so the scroll range is
    // computed on what is drawn and the newest content is reachable
    // (issue #49); the paragraph itself does no wrapping.
    let rows = format_messages(app, usize::from(text_area.width));
    let total_lines = rows.len();
    let visible = usize::from(text_area.height);
    let max_scroll = total_lines.saturating_sub(visible);
    app.scroll_limit.set(max_scroll);
    let effective_scroll = max_scroll.saturating_sub(app.scroll.lines_back(max_scroll));
    let scroll_u16 = u16::try_from(effective_scroll).unwrap_or(u16::MAX);
    app.view.set(TranscriptView {
        area: text_area,
        top: effective_scroll,
        lines: total_lines,
    });

    let lines: Vec<Line<'static>> = rows
        .into_iter()
        .enumerate()
        .map(
            |(index, row)| match app.selection.and_then(|selection| selection.columns(index)) {
                Some(columns) => row.highlighted(columns),
                None => row.line,
            },
        )
        .collect();
    let text = Text::from(lines);
    let paragraph = Paragraph::new(text).scroll((scroll_u16, 0));

    frame.render_widget(paragraph, text_area);
    if max_scroll > 0 {
        // `ScrollbarState` counts positions, and the last one is `max_scroll`.
        let mut state = ScrollbarState::new(max_scroll.saturating_add(1))
            .position(effective_scroll)
            .viewport_content_length(visible);
        let bar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .symbols(symbols::scrollbar::VERTICAL)
            .begin_symbol(None)
            .end_symbol(None)
            .style(Style::default().fg(Color::DarkGray));
        frame.render_stateful_widget(bar, bar_area, &mut state);
    }
}

fn draw_input(
    frame: &mut Frame<'_>,
    area: Rect,
    app: &App,
    permission: Option<Vec<Line<'static>>>,
) {
    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if let Some(lines) = permission {
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
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
    if let Some(copy) = &app.copy_status {
        let color = match copy {
            CopyStatus::Copied(_) | CopyStatus::Sent(_) => Color::DarkGray,
            CopyStatus::Failed(_) => Color::Red,
        };
        let line = Line::styled(format!(" {copy}"), Style::default().fg(color));
        frame.render_widget(Paragraph::new(line), area);
        return;
    }
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
        Span::styled(
            JobsIndicator(app.job_counts()).to_string(),
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    frame.render_widget(Paragraph::new(status), area);
}

/// A message's rendered rows and the fingerprint they were rendered from.
#[derive(Debug, Clone)]
pub(crate) struct Wrapped {
    pub(crate) key: u64,
    rows: Vec<Row>,
}

impl PendingWrite<'_> {
    /// The overlay: who asks, the statement wrapped to `width` in at most
    /// `max_rows` rows, and the choices.
    pub(crate) fn lines(&self, width: usize, max_rows: usize) -> Vec<Line<'static>> {
        let mut heading = vec![
            Span::raw(" "),
            Span::styled(
                self.heading.clone(),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ];
        if self.waiting > 0 {
            heading.push(Span::styled(
                format!("  (+{} more waiting)", self.waiting),
                Style::default().fg(Color::DarkGray),
            ));
        }
        let rows: Vec<Vec<Span<'static>>> = self
            .sql
            .lines()
            .flat_map(|line| wrap::wrap(&[Span::raw(line.to_owned())], width.saturating_sub(3)))
            .collect();
        let mut lines = vec![Line::from(heading)];
        let shown = if rows.len() > max_rows {
            max_rows.saturating_sub(1)
        } else {
            rows.len()
        };
        lines.extend(rows.iter().take(shown).map(|row| {
            let mut spans = vec![Span::raw("   ")];
            spans.extend(row.iter().cloned());
            Line::from(spans)
        }));
        if shown < rows.len() {
            lines.push(Line::styled(
                format!(
                    "   \u{2026} {} more lines",
                    rows.len().saturating_sub(shown)
                ),
                Style::default().fg(Color::DarkGray),
            ));
        }
        lines.push(Line::from(vec![
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
        ]));
        lines
    }
}

/// Every message's rows, wrapped to `width`. Each message's rows are
/// cached on the app by a fingerprint of what it shows, so a redraw
/// re-renders only the messages that changed (the one streaming, as a
/// rule), not the Markdown and wrapping of the whole transcript.
pub(crate) fn format_messages(app: &App, width: usize) -> Vec<Row> {
    let mut cache = app.wrap_cache.borrow_mut();
    // Messages are appended, edited in place, or cleared; never removed
    // from the middle, so the cache stays aligned by index.
    cache.truncate(app.messages.len());
    cache.resize(app.messages.len(), None);
    let mut rows: Vec<Row> = Vec::new();
    for (msg, slot) in app.messages.iter().zip(cache.iter_mut()) {
        let key = msg.fingerprint(width, app.expand_steps);
        match slot {
            Some(cached) if cached.key == key => rows.extend(cached.rows.iter().cloned()),
            _ => {
                let rendered = msg.rows(width, app.expand_steps);
                rows.extend(rendered.iter().cloned());
                *slot = Some(Wrapped {
                    key,
                    rows: rendered,
                });
            }
        }
    }
    rows
}

impl Message {
    /// What decides its rendering.
    fn fingerprint(&self, width: usize, expand: bool) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.kind.hash(&mut hasher);
        self.content.hash(&mut hasher);
        self.detail.hash(&mut hasher);
        self.chart
            .as_ref()
            .map(|c| format!("{c:?}"))
            .hash(&mut hasher);
        width.hash(&mut hasher);
        (expand && self.kind == MessageKind::Step).hash(&mut hasher);
        hasher.finish()
    }

    /// Its rows, wrapped to `width`, with the blank line after it.
    fn rows(&self, width: usize, expand_steps: bool) -> Vec<Row> {
        let (prefix, style) = match self.kind {
            MessageKind::User => (" > ", Style::default().fg(Color::Cyan)),
            MessageKind::Assistant => ("   ", Style::default()),
            MessageKind::Step => ("   ", Style::default().fg(Color::Yellow)),
            MessageKind::Sql => ("   ", Style::default().fg(Color::White)),
            MessageKind::System => ("   ", Style::default().fg(Color::DarkGray)),
            MessageKind::Upload => (" \u{2191} ", Style::default().fg(Color::Green)),
            MessageKind::Error => ("   ", Style::default().fg(Color::Red)),
        };
        let prompt_style = if self.kind == MessageKind::User {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        } else {
            style
        };
        let body: Vec<Vec<Span<'static>>> = match self.kind {
            // Every row gets a three-column prefix below.
            MessageKind::Assistant => markdown::render(&self.content, width.saturating_sub(3)),
            MessageKind::Step => self.step_rows(expand_steps, style),
            MessageKind::User
            | MessageKind::Sql
            | MessageKind::System
            | MessageKind::Upload
            | MessageKind::Error => self
                .content
                .lines()
                .map(|l| vec![Span::styled(l.to_owned(), style)])
                .collect(),
        };
        let mut rows: Vec<Row> = Vec::new();
        let mut first = true;
        for spans in body {
            let p = if first {
                first = false;
                prefix
            } else {
                "   "
            };
            let mut wrap = Wrap::Start;
            for row in wrap::wrap(&spans, width.saturating_sub(p.chars().count())) {
                let mut with_prefix = vec![Span::styled(p.to_owned(), prompt_style)];
                with_prefix.extend(row);
                rows.push(Row {
                    line: Line::from(with_prefix),
                    wrap,
                });
                wrap = Wrap::Continued;
            }
        }
        if let Some(chart) = &self.chart {
            let chart_width = u16::try_from(width.saturating_sub(3)).unwrap_or(u16::MAX);
            for row in chart.lines(chart_width) {
                let mut with_prefix = vec![Span::raw("   ")];
                with_prefix.extend(row.spans);
                rows.push(Row::start(Line::from(with_prefix)));
            }
        }
        rows.push(Row::start(Line::from("")));
        rows
    }

    /// A step: its header and outcome, then the detail in full when
    /// expanded or its first lines with a count of the rest.
    fn step_rows(&self, expanded: bool, style: Style) -> Vec<Vec<Span<'static>>> {
        let mut rows: Vec<Vec<Span<'static>>> = self
            .content
            .lines()
            .map(|l| vec![Span::styled(l.to_owned(), style)])
            .collect();
        let Some(detail) = self.detail.as_deref().filter(|d| !d.trim().is_empty()) else {
            return rows;
        };
        let dim = Style::default().fg(Color::DarkGray);
        let DetailPreview {
            lines: shown,
            hidden: more,
        } = if expanded {
            DetailPreview::whole(detail)
        } else {
            DetailPreview::of(detail)
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
}

fn separator_line(width: u16) -> Line<'static> {
    let w = usize::from(width);
    Line::styled("\u{2500}".repeat(w), Style::default().fg(Color::DarkGray))
}

/// Wrapping of styled spans to a width in terminal columns, breaking at
/// the last space when one is near.
pub(crate) mod wrap {
    use ratatui::style::Style;
    use ratatui::text::{Span, StyledGrapheme};

    pub(crate) fn wrap(spans: &[Span<'static>], width: usize) -> Vec<Vec<Span<'static>>> {
        let width = width.max(1);
        // Graphemes, not characters: a wide one takes two columns and a
        // combining mark none, and neither may be split across rows.
        let cells: Vec<StyledGrapheme<'_>> = spans
            .iter()
            .flat_map(|s| s.styled_graphemes(Style::default()))
            .collect();
        if cells.is_empty() {
            return vec![Vec::new()];
        }
        let mut rows = Vec::new();
        let mut start = 0;
        while start < cells.len() {
            let mut end = start;
            let mut used = 0_usize;
            // The row's last space and the columns before it.
            let mut space = None;
            for (at, cell) in cells.iter().enumerate().skip(start) {
                let next = used.saturating_add(Span::raw(cell.symbol).width());
                // A grapheme wider than the row still takes one.
                if next > width && at > start {
                    break;
                }
                if cell.symbol == " " {
                    space = Some((at, used));
                }
                used = next;
                end = at.saturating_add(1);
            }
            if end < cells.len()
                // Prefer the last space in the row, if it is not too early.
                && let Some((at, _)) = space.filter(|(_, column)| column.saturating_mul(2) > width)
            {
                end = at.saturating_add(1);
            }
            rows.push(regroup(cells.get(start..end).unwrap_or(&[])));
            start = end;
        }
        rows
    }

    /// Consecutive graphemes of one style back into spans.
    pub(crate) fn regroup(cells: &[StyledGrapheme<'_>]) -> Vec<Span<'static>> {
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut current = String::new();
        let mut current_style: Option<Style> = None;
        for cell in cells {
            if current_style != Some(cell.style) {
                if let Some(s) = current_style
                    && !current.is_empty()
                {
                    spans.push(Span::styled(std::mem::take(&mut current), s));
                }
                current_style = Some(cell.style);
            }
            current.push_str(cell.symbol);
        }
        if let Some(s) = current_style
            && !current.is_empty()
        {
            spans.push(Span::styled(current, s));
        }
        spans
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
        assert_ne!(
            m_bar
                .lines(width, false)
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
            m_pie
                .lines(width, false)
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
            "rendered lines differ",
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
}
