//! Progress of a long run on stderr: a bar redrawn in place when stderr is
//! a terminal, one line per finished unit when it is a pipe.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use quack_core::progress::ChunkDone;

/// Where the progress goes. A failed write is dropped: progress is a
/// courtesy, never a reason to stop the run.
pub(crate) struct StderrProgress {
    style: Style,
    /// A bar is on the current line and has no newline after it yet.
    drawn: AtomicBool,
}

#[derive(Debug, Clone, Copy)]
enum Style {
    /// `\r`-redrawn bar, cut to the terminal's width.
    Bar { columns: usize },
    /// `12/40 done in 3 s, 1 failed; 41 s elapsed`, one line each.
    Lines,
}

impl StderrProgress {
    const CELLS: usize = 20;

    pub(crate) fn new() -> Self {
        let style = if std::io::stderr().is_terminal() {
            let columns =
                crossterm::terminal::size().map_or(80, |(columns, _)| usize::from(columns));
            Style::Bar { columns }
        } else {
            Style::Lines
        };
        Self {
            style,
            drawn: AtomicBool::new(false),
        }
    }

    pub(crate) fn report(&self, done: ChunkDone) {
        let mut err = std::io::stderr().lock();
        match self.style {
            Style::Lines => drop(writeln!(err, "{}", Self::line(done))),
            Style::Bar { columns } => {
                let finished = done.done >= done.total;
                let text: String = Self::bar(done)
                    .chars()
                    .take(columns.saturating_sub(1))
                    .collect();
                let end = if finished { "\n" } else { "" };
                drop(write!(err, "\r\x1b[2K{text}{end}"));
                drop(err.flush());
                self.drawn.store(!finished, Ordering::Relaxed);
            }
        }
    }

    fn line(done: ChunkDone) -> String {
        format!(
            "{}/{} done in {} s{}; {} s elapsed",
            done.done,
            done.total,
            done.took.as_secs(),
            Self::failed(done),
            done.elapsed.as_secs()
        )
    }

    /// `[████████░░░░░░░░░░░░] 12/40, 1 failed; 41 s elapsed, about 2 min left`.
    fn bar(done: ChunkDone) -> String {
        let filled = if done.total == 0 {
            Self::CELLS
        } else {
            let cells = u64::try_from(Self::CELLS).unwrap_or(u64::MAX);
            u64::from(done.done.min(done.total))
                .saturating_mul(cells)
                .div_euclid(u64::from(done.total))
                .try_into()
                .unwrap_or(Self::CELLS)
        };
        let cells: String = std::iter::repeat_n('█', filled)
            .chain(std::iter::repeat_n('░', Self::CELLS.saturating_sub(filled)))
            .collect();
        let left = Self::remaining(done).map_or_else(String::new, |left| {
            format!(", about {} left", Self::span(left))
        });
        format!(
            "[{cells}] {}/{}{}; {} elapsed{left}",
            done.done,
            done.total,
            Self::failed(done),
            Self::span(done.elapsed)
        )
    }

    fn failed(done: ChunkDone) -> String {
        if done.failed > 0 {
            format!(", {} failed", done.failed)
        } else {
            String::new()
        }
    }

    /// The elapsed time scaled to the units still to go; nothing before
    /// the first unit or after the last.
    fn remaining(done: ChunkDone) -> Option<Duration> {
        let left = done.total.checked_sub(done.done).filter(|left| *left > 0)?;
        if done.done == 0 {
            return None;
        }
        done.elapsed.checked_mul(left)?.checked_div(done.done)
    }

    /// `41 s`, `2 min 10 s`, `1 h 3 min`.
    fn span(duration: Duration) -> String {
        let seconds = duration.as_secs();
        let (hours, minutes, seconds) = (
            seconds.div_euclid(3600),
            seconds.rem_euclid(3600).div_euclid(60),
            seconds.rem_euclid(60),
        );
        if hours > 0 {
            format!("{hours} h {minutes} min")
        } else if minutes > 0 {
            format!("{minutes} min {seconds} s")
        } else {
            format!("{seconds} s")
        }
    }
}

impl Drop for StderrProgress {
    /// A run that stopped before its last unit leaves the bar's line open;
    /// close it so the error that follows starts on its own line.
    fn drop(&mut self) {
        if self.drawn.load(Ordering::Relaxed) {
            drop(writeln!(std::io::stderr()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(done: u32, total: u32, failed: u32, elapsed: u64) -> ChunkDone {
        ChunkDone {
            done,
            total,
            failed,
            took: Duration::from_secs(3),
            elapsed: Duration::from_secs(elapsed),
        }
    }

    #[test]
    fn bar_scales_fill_and_estimates_the_rest() {
        assert_eq!(
            StderrProgress::bar(at(12, 40, 0, 41)),
            "[██████░░░░░░░░░░░░░░] 12/40; 41 s elapsed, about 1 min 35 s left"
        );
        assert_eq!(
            StderrProgress::bar(at(0, 40, 0, 0)),
            "[░░░░░░░░░░░░░░░░░░░░] 0/40; 0 s elapsed"
        );
        assert_eq!(
            StderrProgress::bar(at(40, 40, 2, 130)),
            "[████████████████████] 40/40, 2 failed; 2 min 10 s elapsed"
        );
        assert_eq!(
            StderrProgress::bar(at(1, 3, 0, 3700)),
            "[██████░░░░░░░░░░░░░░] 1/3; 1 h 1 min elapsed, about 2 h 3 min left"
        );
    }

    #[test]
    fn pipes_keep_one_line_per_unit() {
        assert_eq!(
            StderrProgress::line(at(12, 40, 1, 41)),
            "12/40 done in 3 s, 1 failed; 41 s elapsed"
        );
    }
}
