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
//! Nothing ever locks the connection: async code awaits [`Writer::run`]
//! and never blocks a runtime worker; sync code outside the runtime (a
//! job's own thread, a startup step) uses [`Writer::call`]. A closure is
//! owned (`Send + 'static`) because it crosses to the writer's thread; one
//! that panics comes back as an error and the writer carries on (a
//! transaction it left open rolls back with it). Readers are separate
//! connections ([`crate::analysis::tools::ReaderDb`]) and never wait here.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{JoinHandle, ThreadId};

use tokio::sync::oneshot;

use crate::error::{Error, Result};
use crate::priority::{Priority, current_priority};
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

struct Shared {
    lines: Mutex<Lines>,
    ready: Condvar,
}

impl Shared {
    fn lines(&self) -> MutexGuard<'_, Lines> {
        // Plain queues: a panic elsewhere cannot leave them inconsistent.
        self.lines.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A workspace connection on its own thread, served interactive first.
pub struct Writer {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
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
            .spawn(move || serve(&serving, &db))?;
        Ok(Self {
            thread_id: thread.thread().id(),
            shared,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Run `f` on the connection at the calling task's priority and await
    /// its answer; the runtime worker is free meanwhile.
    ///
    /// # Errors
    ///
    /// `f`'s error; [`Error::Analysis`] when `f` panicked or the writer has
    /// stopped.
    pub async fn run<T: Send + 'static>(
        &self,
        f: impl FnOnce(&WorkspaceDb) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        self.run_at(current_priority(), f).await
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
        self.submit(priority, f, move |outcome| drop(answer.send(outcome)))?;
        answered.await.unwrap_or_else(|_| Err(stopped()))
    }

    /// Run `f` and wait for its answer, blocking this thread: for sync code
    /// off the runtime (a job's own thread, startup, shutdown). Async code
    /// uses [`Self::run`].
    ///
    /// # Errors
    ///
    /// As [`Self::run`]; also when called from the writer's own thread
    /// (from inside another closure), which would wait on itself.
    pub fn call<T: Send + 'static>(
        &self,
        f: impl FnOnce(&WorkspaceDb) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        if std::thread::current().id() == self.thread_id {
            return Err(Error::Analysis(String::from(
                "a writer closure asked the writer for more work; it would wait on itself",
            )));
        }
        let (answer, answered) = std::sync::mpsc::sync_channel(1);
        self.submit(current_priority(), f, move |outcome| {
            drop(answer.send(outcome));
        })?;
        answered.recv().unwrap_or_else(|_| Err(stopped()))
    }

    /// Queue `f`; `reply` gets its outcome on the writer's thread.
    fn submit<T: Send + 'static>(
        &self,
        priority: Priority,
        f: impl FnOnce(&WorkspaceDb) -> Result<T> + Send + 'static,
        reply: impl FnOnce(Result<T>) + Send + 'static,
    ) -> Result<()> {
        let job: Job =
            Box::new(move |db| {
                let outcome = catch_unwind(AssertUnwindSafe(|| f(db))).unwrap_or_else(|panic| {
                let what = panic
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_default();
                tracing::error!(panic = %what, "a workspace write panicked; the writer carries on");
                Err(Error::Analysis(format!("the workspace write failed: {what}")))
            });
                // A caller that stopped waiting (a dropped future) is fine.
                reply(outcome);
            });
        let mut lines = self.shared.lines();
        if lines.closed {
            return Err(stopped());
        }
        match priority {
            Priority::Interactive => lines.interactive.push_back(job),
            Priority::Background => lines.background.push_back(job),
        }
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

/// The writer's thread: run what arrives, interactive first, until closed
/// and drained. The connection closes (and checkpoints) when this returns.
fn serve(shared: &Shared, db: &WorkspaceDb) {
    loop {
        let job = {
            let mut lines = shared.lines();
            loop {
                if let Some(job) = lines
                    .interactive
                    .pop_front()
                    .or_else(|| lines.background.pop_front())
                {
                    break job;
                }
                if lines.closed {
                    return;
                }
                lines = shared
                    .ready
                    .wait(lines)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        job(db);
    }
}

impl Drop for Writer {
    /// Finish what is queued, then close the connection before returning,
    /// so a process that drops its last handle ends with the file closed.
    fn drop(&mut self) {
        self.shared.lines().closed = true;
        self.shared.ready.notify_all();
        let thread = self
            .thread
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(thread) = thread
            && std::thread::current().id() != self.thread_id
            && thread.join().is_err()
        {
            tracing::error!("the workspace writer thread panicked");
        }
    }
}

fn stopped() -> Error {
    Error::Analysis(String::from("the workspace writer has stopped"))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn writer() -> Arc<Writer> {
        Arc::new(
            Writer::spawn(WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string())))
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
        assert!(outcome.is_err_and(|e| e.to_string().contains("mid-write")));
        assert!(writer.run(WorkspaceDb::list_tables).await.is_ok());
    }

    #[test]
    fn blocking_callers_and_shutdown() {
        let writer = writer();
        writer
            .call(|db| db.execute_with_params("CREATE TABLE t (a INTEGER)", []))
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            writer
                .call(WorkspaceDb::list_tables)
                .unwrap_or_else(|e| fail(&e.to_string())),
            vec![String::from("t")]
        );
        // A closure that asks its own writer for more work is refused, not
        // deadlocked.
        let inner = Arc::clone(&writer);
        let nested = writer.call(move |_| Ok(inner.call(|_| Ok(())).is_err()));
        assert!(nested.is_ok_and(|refused| refused));
        // Dropping the last handle finishes the queue and joins the thread.
        drop(writer);
    }
}
