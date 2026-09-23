//! The workspace's one writer connection, behind a queue with two tiers
//! (design doc 4.1 and 7.4).
//!
//! `DuckDB` takes one writer; every write in the process goes through the
//! [`Writer`] of its workspace, one at a time. Waiters are served in two
//! lines, first come first served within each: an interactive caller (a
//! turn recording its answer, a terminal command, a request handler) goes
//! ahead of a background one (an ingest's next batch, an extraction's next
//! write), so a person never queues behind a whole ingest. The line is
//! [`crate::priority`]'s task-local, or an explicit [`Writer::lock_at`].
//!
//! The connection is not moved onto a thread of its own: callers hand it
//! closures that borrow their data, which a thread boundary would forbid
//! without `unsafe`, and the queue gives the same order. Readers are
//! separate connections ([`crate::analysis::tools::ReaderDb`]) and never
//! wait here.

use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::priority::{Priority, current_priority};
use crate::storage::workspace::WorkspaceDb;

/// The connection was left mid-write by a panic.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("workspace lock poisoned: a write panicked while holding it")]
pub struct Poisoned;

/// A workspace connection and its two-tier line of waiters.
pub struct Writer {
    db: Mutex<WorkspaceDb>,
    line: Mutex<Line>,
}

#[derive(Default)]
struct Line {
    /// Someone holds the connection (or it is being handed on).
    held: bool,
    interactive: VecDeque<SyncSender<()>>,
    background: VecDeque<SyncSender<()>>,
}

impl Writer {
    #[must_use]
    pub fn new(db: WorkspaceDb) -> Self {
        Self {
            db: Mutex::new(db),
            line: Mutex::new(Line::default()),
        }
    }

    fn line(&self) -> MutexGuard<'_, Line> {
        // The line is plain bookkeeping; a panic elsewhere cannot leave it
        // half-updated in a way that matters.
        self.line.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Wait for the connection at the calling task's priority.
    ///
    /// # Errors
    ///
    /// [`Poisoned`] when an earlier holder panicked mid-write.
    pub fn lock(&self) -> Result<WriterGuard<'_>, Poisoned> {
        self.lock_at(current_priority())
    }

    /// Wait for the connection in `priority`'s line. Blocks the thread;
    /// async callers go through a blocking task (`analysis::tools::with_db`).
    ///
    /// # Errors
    ///
    /// [`Poisoned`] when an earlier holder panicked mid-write.
    pub fn lock_at(&self, priority: Priority) -> Result<WriterGuard<'_>, Poisoned> {
        let waiting: Option<Receiver<()>> = {
            let mut line = self.line();
            if line.held {
                let (turn, wait) = sync_channel(1);
                match priority {
                    Priority::Interactive => line.interactive.push_back(turn),
                    Priority::Background => line.background.push_back(turn),
                }
                Some(wait)
            } else {
                line.held = true;
                None
            }
        };
        if let Some(wait) = waiting {
            // The sender lives in the line until it is handed the turn; it
            // only disappears with the writer itself.
            wait.recv().map_err(|_| Poisoned)?;
        }
        let turn = Turn { writer: self };
        let db = self.db.lock().map_err(|_| Poisoned)?;
        Ok(WriterGuard { db, _turn: turn })
    }

    /// The connection if nobody holds it or waits for it, else `None`.
    #[must_use]
    pub fn try_lock(&self) -> Option<WriterGuard<'_>> {
        {
            let mut line = self.line();
            if line.held {
                return None;
            }
            line.held = true;
        }
        let turn = Turn { writer: self };
        let db = self.db.lock().ok()?;
        Some(WriterGuard { db, _turn: turn })
    }

    /// Pass the connection to the next waiter, interactive first, or free it.
    fn release(&self) {
        let mut line = self.line();
        loop {
            let next = line
                .interactive
                .pop_front()
                .or_else(|| line.background.pop_front());
            let Some(next) = next else {
                line.held = false;
                return;
            };
            // A waiter's thread that died has dropped its receiver; skip it.
            if next.send(()).is_ok() {
                return;
            }
        }
    }

    /// Waiters in each line (interactive, background), for tests.
    #[cfg(test)]
    fn waiting(&self) -> (usize, usize) {
        let line = self.line();
        (line.interactive.len(), line.background.len())
    }
}

/// The caller's turn at the connection; releasing it hands the connection on.
struct Turn<'a> {
    writer: &'a Writer,
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        self.writer.release();
    }
}

/// The connection, held. The mutex guard is declared first so it is
/// released before the turn passes the connection on.
pub struct WriterGuard<'a> {
    db: MutexGuard<'a, WorkspaceDb>,
    _turn: Turn<'a>,
}

impl Deref for WriterGuard<'_> {
    type Target = WorkspaceDb;

    fn deref(&self) -> &WorkspaceDb {
        &self.db
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn writer() -> Arc<Writer> {
        Arc::new(Writer::new(
            WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string())),
        ))
    }

    /// Wait until `n` waiters are lined up (interactive plus background).
    fn wait_for_line(writer: &Writer, n: usize) {
        for _ in 0..200 {
            let (a, b) = writer.waiting();
            if a.saturating_add(b) >= n {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        fail("the waiters never lined up");
    }

    #[test]
    fn interactive_callers_go_ahead_of_background_ones() {
        let writer = writer();
        let order = Arc::new(Mutex::new(Vec::new()));
        let held = writer.lock().unwrap_or_else(|e| fail(&e.to_string()));
        let mut threads = Vec::new();
        for (n, (name, priority)) in [
            ("background 1", Priority::Background),
            ("background 2", Priority::Background),
            ("interactive 1", Priority::Interactive),
            ("interactive 2", Priority::Interactive),
        ]
        .into_iter()
        .enumerate()
        {
            let (shared, order) = (Arc::clone(&writer), Arc::clone(&order));
            threads.push(std::thread::spawn(move || {
                let guard = shared
                    .lock_at(priority)
                    .unwrap_or_else(|e| fail(&e.to_string()));
                assert!(guard.list_tables().is_ok());
                order
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(name);
            }));
            // Line them up in this order.
            wait_for_line(&writer, n.saturating_add(1));
        }
        assert!(
            writer.try_lock().is_none(),
            "held, so try_lock waits for nobody"
        );
        drop(held);
        for thread in threads {
            assert!(thread.join().is_ok());
        }
        assert_eq!(
            *order.lock().unwrap_or_else(PoisonError::into_inner),
            vec![
                "interactive 1",
                "interactive 2",
                "background 1",
                "background 2"
            ]
        );
        assert!(writer.try_lock().is_some(), "free again");
    }

    #[test]
    fn a_panic_mid_write_poisons_but_frees_the_line() {
        let writer = writer();
        let panicking = Arc::clone(&writer);
        let joined = std::thread::spawn(move || {
            let _guard = panicking.lock().unwrap_or_else(|e| fail(&e.to_string()));
            fail("boom");
        })
        .join();
        assert!(joined.is_err());
        // The next caller is not stuck in line: it gets the poison error.
        assert!(writer.lock().is_err());
        assert!(writer.lock().is_err());
    }
}
