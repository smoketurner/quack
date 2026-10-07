//! The box `/jobs` and `/sessions` open over the transcript: a table to
//! move through, with the keys that act on its highlighted row.

use quack_core::ids::SessionId;
use quack_core::jobs::{JobInfo, JobState};
use quack_core::storage::sessions::SessionRow;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, Clear, HighlightSpacing, Row, StatefulWidget, Table, TableState, Widget,
};

use crate::terminal::ui::one_line;

const DIM: Style = Style::new().fg(Color::DarkGray);

/// What the box lists.
enum Listing {
    /// Every job on record, newest first.
    Jobs(Vec<JobInfo>),
    /// Recent sessions, most recently updated first, and the one on screen.
    Sessions {
        sessions: Vec<SessionRow>,
        current: SessionId,
    },
}

/// The highlighted row.
pub(crate) enum Picked<'a> {
    Job(&'a JobInfo),
    Session(&'a SessionRow),
}

pub(crate) struct Picker {
    listing: Listing,
    selected: usize,
}

impl Picker {
    /// `jobs` in submission order; the newest is highlighted.
    pub(crate) fn jobs(mut jobs: Vec<JobInfo>) -> Self {
        jobs.reverse();
        Self {
            listing: Listing::Jobs(jobs),
            selected: 0,
        }
    }

    /// `sessions` as listed; the one on screen is highlighted.
    pub(crate) fn sessions(sessions: Vec<SessionRow>, current: SessionId) -> Self {
        let selected = sessions
            .iter()
            .position(|session| session.id == current)
            .unwrap_or(0);
        Self {
            listing: Listing::Sessions { sessions, current },
            selected,
        }
    }

    /// The queue changed: list `jobs` (in submission order) instead, the
    /// highlight staying on its job. A session list is left as it is.
    pub(crate) fn follow_jobs(&mut self, mut jobs: Vec<JobInfo>) {
        let Listing::Jobs(listed) = &mut self.listing else {
            return;
        };
        jobs.reverse();
        let highlighted = listed.get(self.selected).map(|job| job.id);
        *listed = jobs;
        self.selected = highlighted
            .and_then(|id| listed.iter().position(|job| job.id == id))
            .unwrap_or(self.selected)
            .min(listed.len().saturating_sub(1));
    }

    fn len(&self) -> usize {
        match &self.listing {
            Listing::Jobs(jobs) => jobs.len(),
            Listing::Sessions { sessions, .. } => sessions.len(),
        }
    }

    pub(crate) fn up(&mut self, rows: usize) {
        self.selected = self.selected.saturating_sub(rows);
    }

    pub(crate) fn down(&mut self, rows: usize) {
        self.selected = self
            .selected
            .saturating_add(rows)
            .min(self.len().saturating_sub(1));
    }

    pub(crate) fn first(&mut self) {
        self.selected = 0;
    }

    pub(crate) fn last(&mut self) {
        self.selected = self.len().saturating_sub(1);
    }

    pub(crate) fn picked(&self) -> Option<Picked<'_>> {
        match &self.listing {
            Listing::Jobs(jobs) => jobs.get(self.selected).map(Picked::Job),
            Listing::Sessions { sessions, .. } => sessions.get(self.selected).map(Picked::Session),
        }
    }

    /// Rows it takes when nothing cuts it short: its entries, the column
    /// headings, and the border.
    pub(crate) fn height(&self) -> u16 {
        u16::try_from(self.len().saturating_add(3)).unwrap_or(u16::MAX)
    }

    fn title(&self) -> &'static str {
        match self.listing {
            Listing::Jobs(_) => " Jobs ",
            Listing::Sessions { .. } => " Sessions ",
        }
    }

    fn keys(&self) -> &'static str {
        match self.listing {
            Listing::Jobs(_) => {
                " \u{2191}\u{2193} move \u{00B7} c cancel \u{00B7} enter details \u{00B7} esc close "
            }
            Listing::Sessions { .. } => {
                " \u{2191}\u{2193} move \u{00B7} enter resume \u{00B7} d delete \u{00B7} esc close "
            }
        }
    }

    fn table(&self) -> Table<'_> {
        match &self.listing {
            Listing::Jobs(jobs) => Table::new(
                jobs.iter().map(Self::job_row),
                [
                    Constraint::Length(4),
                    Constraint::Length(9),
                    Constraint::Length(8),
                    Constraint::Fill(1),
                    Constraint::Fill(1),
                ],
            )
            .header(Row::new(["#", "state", "kind", "label", ""]).style(DIM)),
            Listing::Sessions { sessions, current } => Table::new(
                sessions
                    .iter()
                    .map(|session| Self::session_row(session, current)),
                [
                    Constraint::Length(1),
                    Constraint::Length(8),
                    Constraint::Length(19),
                    Constraint::Length(4),
                    Constraint::Fill(1),
                ],
            )
            .header(Row::new(["", "id", "updated", "msgs", "title"]).style(DIM)),
        }
    }

    fn job_row(job: &JobInfo) -> Row<'_> {
        let state = match job.state {
            JobState::Running => Style::new().fg(Color::Yellow),
            JobState::Succeeded => Style::new().fg(Color::Green),
            JobState::Failed => Style::new().fg(Color::Red),
            JobState::Queued | JobState::Cancelled => DIM,
        };
        // How it ended, or how far it has got.
        let detail = if job.state.is_finished() {
            job.outcome.as_deref().map(one_line).unwrap_or_default()
        } else {
            let mut detail = job.progress.map(|p| p.to_string()).unwrap_or_default();
            for note in [
                job.status.as_deref(),
                job.cancel_requested.then_some("cancelling"),
            ]
            .into_iter()
            .flatten()
            {
                if !detail.is_empty() {
                    detail.push_str("  ");
                }
                detail.push_str(note);
            }
            detail
        };
        Row::new([
            Cell::new(job.number.to_string()),
            Cell::new(Span::styled(job.state.as_str(), state)),
            Cell::new(Span::styled(job.kind.as_str(), DIM)),
            Cell::new(job.label.as_str()),
            Cell::new(Span::styled(detail, DIM)),
        ])
    }

    fn session_row<'a>(session: &'a SessionRow, current: &SessionId) -> Row<'a> {
        let marker = if session.id == *current { "*" } else { "" };
        Row::new([
            Cell::new(marker),
            Cell::new(session.id.short()),
            Cell::new(Span::styled(session.updated_at.as_str(), DIM)),
            Cell::new(Span::styled(session.message_count.to_string(), DIM)),
            Cell::new(session.title.as_deref().unwrap_or("(untitled)")),
        ])
    }
}

impl Widget for &Picker {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let highlight = Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD);
        let table = self
            .table()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(DIM)
                    .title(Line::styled(self.title(), highlight))
                    .title_bottom(Line::styled(self.keys(), DIM)),
            )
            .highlight_symbol(Span::styled("\u{25B8} ", highlight))
            .highlight_spacing(HighlightSpacing::Always)
            .row_highlight_style(Style::new().add_modifier(Modifier::BOLD));
        // A new state each frame: the table scrolls just far enough to
        // keep the highlighted row in view.
        let mut state = TableState::new().with_selected(self.selected);
        Clear.render(area, buf);
        StatefulWidget::render(table, area, buf, &mut state);
    }
}
