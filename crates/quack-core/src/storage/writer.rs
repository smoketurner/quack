//! The workspace's one writer connection, owned by a thread of its own
//! (design doc 4.1 and 7.4).
//!
//! `DuckDB` takes one writer; every write in the process goes to the
//! [`Writer`] of its workspace, an actor: a dedicated thread owns the
//! connection and runs the closures sent to it one at a time. They wait in
//! two lines, first come first served within each: an interactive caller (a
//! turn recording its answer, a terminal command, a request handler) goes
//! ahead of a background one (an ingest's next batch, an extraction's next
//! write), so a person never queues behind a whole ingest. The line is
//! [`crate::priority`]'s task-local, or explicit with [`Writer::run_at`].
//!
//! Nothing ever locks the connection: callers await [`Writer::run`] and
//! never block a runtime worker. A closure is owned (`Send + 'static`)
//! because it crosses to the writer's thread; one that panics comes back
//! as an error and the writer carries on (a transaction it left open rolls
//! back with it). Readers are separate connections
//! ([`crate::analysis::tools::ReaderDb`]) and never wait here.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{JoinHandle, ThreadId};

use tokio::sync::oneshot;

use crate::error::{Error, Result};
use crate::priority::Priority;
use crate::storage::workspace::WorkspaceDb;

/// A closure on its way to the writer's thread.
type Job = Box<dyn FnOnce(&WorkspaceDb) + Send>;

/// The two lines, and whether the writer is shutting down.
#[derive(Default)]
struct Lines {
    interactive: VecDeque<Job>,
    background: VecDeque<Job>,
    closed: bool,
}

impl Lines {
    /// Queue `job` at the back of `priority`'s line.
    fn push(&mut self, priority: Priority, job: Job) {
        match priority {
            Priority::Interactive => self.interactive.push_back(job),
            Priority::Background => self.background.push_back(job),
        }
    }

    /// The next job: interactive first, then background.
    fn pop_next(&mut self) -> Option<Job> {
        self.interactive
            .pop_front()
            .or_else(|| self.background.pop_front())
    }
}

struct Shared {
    lines: Mutex<Lines>,
    ready: Condvar,
}

impl Shared {
    fn lines(&self) -> MutexGuard<'_, Lines> {
        // Plain queues: a panic elsewhere cannot leave them inconsistent.
        self.lines.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The writer's thread: run what arrives, interactive first, until
    /// closed and drained. The connection closes (and checkpoints) when
    /// this returns.
    fn serve(&self, db: &WorkspaceDb) {
        loop {
            let job = {
                let mut lines = self.lines();
                loop {
                    if let Some(job) = lines.pop_next() {
                        break job;
                    }
                    if lines.closed {
                        return;
                    }
                    lines = self
                        .ready
                        .wait(lines)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            };
            job(db);
        }
    }
}

/// A workspace connection on its own thread, served interactive first.
pub struct Writer {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
    /// The writer's thread never joins itself (a closure that held the last
    /// handle).
    thread_id: ThreadId,
}

impl std::fmt::Debug for Writer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lines = self.shared.lines();
        f.debug_struct("Writer")
            .field("interactive", &lines.interactive.len())
            .field("background", &lines.background.len())
            .finish_non_exhaustive()
    }
}

impl Writer {
    /// Start the writer's thread with `db` as its connection.
    ///
    /// # Errors
    ///
    /// When the thread cannot be started.
    pub fn spawn(db: WorkspaceDb) -> Result<Self> {
        let shared = Arc::new(Shared {
            lines: Mutex::new(Lines::default()),
            ready: Condvar::new(),
        });
        let serving = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name(String::from("quack-writer"))
            .spawn(move || serving.serve(&db))?;
        Ok(Self {
            thread_id: thread.thread().id(),
            shared,
            thread: Some(thread),
        })
    }

    /// Run `f` on the connection at the calling task's priority and await
    /// its answer; the runtime worker is free meanwhile.
    ///
    /// # Errors
    ///
    /// `f`'s error; [`Error::WritePanicked`] when `f` panicked, or
    /// [`Error::WriterStopped`] when the writer has stopped.
    pub async fn run<T: Send + 'static>(
        &self,
        f: impl FnOnce(&WorkspaceDb) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        self.run_at(Priority::current(), f).await
    }

    /// [`Self::run`] in `priority`'s line.
    ///
    /// # Errors
    ///
    /// As [`Self::run`].
    pub async fn run_at<T: Send + 'static>(
        &self,
        priority: Priority,
        f: impl FnOnce(&WorkspaceDb) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let (answer, answered) = oneshot::channel();
        self.submit(priority, f, answer)?;
        answered.await.unwrap_or_else(|_| Err(Error::WriterStopped))
    }

    /// Queue `f`; `reply` gets its outcome on the writer's thread.
    fn submit<T: Send + 'static>(
        &self,
        priority: Priority,
        f: impl FnOnce(&WorkspaceDb) -> Result<T> + Send + 'static,
        reply: oneshot::Sender<Result<T>>,
    ) -> Result<()> {
        let job: Job = Box::new(move |db| {
            let outcome = catch_unwind(AssertUnwindSafe(|| f(db))).unwrap_or_else(|panic| {
                // A closure that opened its transaction with a raw `BEGIN`
                // (no RAII guard) leaves this one connection inside it when
                // it panics; roll it back so the next job does not start in
                // a leaked, aborted transaction. RAII callers have already
                // rolled back during unwinding, so this errors (ignored) for them.
                if let Err(e) = db.connection().execute("ROLLBACK", []) {
                    tracing::trace!(error = %e, "defensive rollback after a panicked write found no open transaction");
                }
                let what = panic
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_default();
                tracing::error!(panic = %what, "a workspace write panicked; the writer carries on");
                Err(Error::WritePanicked(what))
            });
            // A caller that stopped waiting (a dropped future) is fine.
            drop(reply.send(outcome));
        });
        let mut lines = self.shared.lines();
        if lines.closed {
            return Err(Error::WriterStopped);
        }
        lines.push(priority, job);
        drop(lines);
        self.shared.ready.notify_one();
        Ok(())
    }

    /// Closures waiting (interactive, background), for tests.
    #[cfg(test)]
    fn waiting(&self) -> (usize, usize) {
        let lines = self.shared.lines();
        (lines.interactive.len(), lines.background.len())
    }
}

impl Drop for Writer {
    /// Finish what is queued, then close the connection before returning,
    /// so a process that drops its last handle ends with the file closed.
    fn drop(&mut self) {
        self.shared.lines().closed = true;
        self.shared.ready.notify_all();
        if let Some(thread) = self.thread.take()
            && std::thread::current().id() != self.thread_id
            && thread.join().is_err()
        {
            tracing::error!("the workspace writer thread panicked");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::embedding::Dimension;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn writer() -> Arc<Writer> {
        Arc::new(
            Writer::spawn(
                WorkspaceDb::open_in_memory(Dimension::new(4))
                    .unwrap_or_else(|e| fail(&e.to_string())),
            )
            .unwrap_or_else(|e| fail(&e.to_string())),
        )
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn interactive_work_goes_ahead_of_background_work() {
        let writer = writer();
        let order = Arc::new(Mutex::new(Vec::new()));
        // Hold the writer busy so the rest line up behind it.
        let (release, hold) = std::sync::mpsc::channel::<()>();
        let busy = {
            let writer = Arc::clone(&writer);
            tokio::spawn(async move {
                writer
                    .run(move |_| {
                        hold.recv().ok();
                        Ok(())
                    })
                    .await
            })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        let mut waiting = Vec::new();
        for (name, priority) in [
            ("background 1", Priority::Background),
            ("background 2", Priority::Background),
            ("interactive 1", Priority::Interactive),
            ("interactive 2", Priority::Interactive),
        ] {
            let (shared, order) = (Arc::clone(&writer), Arc::clone(&order));
            waiting.push(tokio::spawn(async move {
                shared
                    .run_at(priority, move |db| {
                        db.list_tables()?;
                        order
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .push(name);
                        Ok(())
                    })
                    .await
            }));
            // Line them up in this order.
            for _ in 0..200 {
                let (a, b) = writer.waiting();
                if a.saturating_add(b) == waiting.len() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }
        assert!(release.send(()).is_ok());
        assert!(busy.await.is_ok_and(|r| r.is_ok()));
        for task in waiting {
            assert!(task.await.is_ok_and(|r| r.is_ok()));
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
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_panicking_closure_is_an_error_and_the_writer_carries_on() {
        let writer = writer();
        #[expect(clippy::panic, reason = "the panic under test")]
        let outcome = writer.run(|_| -> Result<()> { panic!("mid-write") }).await;
        assert!(outcome.is_err_and(
            |e| matches!(&e, Error::WritePanicked(what) if what.contains("mid-write"))
        ));
        assert!(writer.run(WorkspaceDb::list_tables).await.is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_raw_begin_that_panics_does_not_wedge_the_writer() {
        let writer = writer();
        // Job 1: open a raw `BEGIN` (no RAII guard) and panic mid-transaction.
        #[expect(clippy::panic, reason = "the panic under test")]
        let first = writer
            .run(|db| -> Result<()> {
                db.execute_with_params("BEGIN", [])?;
                db.execute_with_params("CREATE TABLE wedge (a INTEGER)", [])?;
                panic!("wedge");
            })
            .await;
        assert!(
            first
                .as_ref()
                .is_err_and(|e| matches!(e, Error::WritePanicked(what) if what.contains("wedge")))
        );
        // Job 2: a conforming `write_transaction`, which issues its own
        // `BEGIN` — it must not run inside the leaked transaction.
        let second = writer
            .run(|db| {
                db.write_transaction(|db| db.execute_with_params("CREATE TABLE t2 (a INTEGER)", []))
            })
            .await;
        // Job 3: a plain read routed through the writer.
        let tables = writer.run(WorkspaceDb::list_tables).await;
        assert!(
            second.is_ok(),
            "writer wedged after raw-BEGIN panic: {second:?}"
        );
        assert!(
            tables.is_ok(),
            "reader wedged after raw-BEGIN panic: {tables:?}"
        );
        // The panicked `CREATE TABLE wedge` rolled back; only `t2` committed.
        assert!(
            tables
                .as_ref()
                .is_ok_and(|names| names == &vec![String::from("t2")]),
            "the wedging table should have rolled back: {tables:?}"
        );
    }

    #[tokio::test]
    async fn writes_are_seen_and_dropping_the_writer_joins_its_thread() {
        let writer = writer();
        let created = writer
            .run(|db| db.execute_with_params("CREATE TABLE t (a INTEGER)", []))
            .await;
        assert!(created.is_ok());
        assert!(
            writer
                .run(WorkspaceDb::list_tables)
                .await
                .is_ok_and(|tables| tables == vec![String::from("t")])
        );
        // The last handle: the queue is empty, so this returns once the
        // thread has closed the connection.
        drop(writer);
    }
}
