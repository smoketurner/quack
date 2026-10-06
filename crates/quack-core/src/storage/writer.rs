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
//!
//! [`Writer::lend`] hands the connection out as a [`Lease`] and parks the
//! thread until it comes back, so work that needs the file closed (a
//! snapshot's copy, a delete) can close it while every later closure waits.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{JoinHandle, ThreadId};

use tokio::sync::oneshot;

use crate::error::{Error, Result};
use crate::priority::Priority;
use crate::storage::workspace::WorkspaceDb;

/// A closure on its way to the writer's thread; it sees `None` once a
/// lease closed the connection.
type Closure = Box<dyn FnOnce(Option<&WorkspaceDb>) + Send>;

/// Work on its way to the writer's thread.
enum Job {
    Run(Closure),
    /// Hand the connection out until the lease comes back.
    Lend(oneshot::Sender<Lease>),
}

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
    fn serve(&self, mut db: Option<WorkspaceDb>) {
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
            match job {
                Job::Run(run) => run(db.as_ref()),
                Job::Lend(to) => db = Lease::lent(db, to),
            }
        }
    }
}

/// A writer's connection, lent out while its thread waits: whatever the
/// lease holds when it drops goes back, the same connection, a reopened
/// one, or none after [`Lease::close`], when every later closure fails
/// with [`Error::WriterStopped`]. Dropping the writer's last handle waits
/// for its lease.
pub struct Lease {
    db: Option<WorkspaceDb>,
    back: std::sync::mpsc::Sender<Option<WorkspaceDb>>,
}

impl std::fmt::Debug for Lease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lease")
            .field("open", &self.db.is_some())
            .finish_non_exhaustive()
    }
}

impl Lease {
    /// On the writer's thread: send `db` out and wait for what comes back.
    fn lent(db: Option<WorkspaceDb>, to: oneshot::Sender<Self>) -> Option<WorkspaceDb> {
        let (back, returned) = std::sync::mpsc::channel();
        match to.send(Self { db, back }) {
            Ok(()) => returned.recv().unwrap_or(None),
            // The borrower stopped waiting: keep the connection.
            Err(mut unsent) => unsent.db.take(),
        }
    }

    /// The lent connection.
    ///
    /// # Errors
    ///
    /// [`Error::WriterStopped`] when it is closed.
    pub fn db(&self) -> Result<&WorkspaceDb> {
        self.db.as_ref().ok_or(Error::WriterStopped)
    }

    /// Close the lent connection.
    pub fn close(&mut self) {
        self.db = None;
    }

    /// Give the writer `db` in place of what it lent.
    pub fn restore(&mut self, db: WorkspaceDb) {
        self.db = Some(db);
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        // The send fails only when the writer's thread is gone, and the
        // connection with nowhere to go closes here.
        drop(self.back.send(self.db.take()));
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
            .spawn(move || serving.serve(Some(db)))?;
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

    /// The connection itself, once every closure queued ahead of this call
    /// at the calling task's priority has run; later closures wait until
    /// the lease drops.
    ///
    /// # Errors
    ///
    /// [`Error::WriterStopped`] when the writer has stopped.
    pub async fn lend(&self) -> Result<Lease> {
        let (to, lent) = oneshot::channel();
        self.push(Priority::current(), Job::Lend(to))?;
        lent.await.map_err(|_| Error::WriterStopped)
    }

    /// Queue `f`; `reply` gets its outcome on the writer's thread.
    fn submit<T: Send + 'static>(
        &self,
        priority: Priority,
        f: impl FnOnce(&WorkspaceDb) -> Result<T> + Send + 'static,
        reply: oneshot::Sender<Result<T>>,
    ) -> Result<()> {
        let job = Job::Run(Box::new(move |db| {
            let Some(db) = db else {
                drop(reply.send(Err(Error::WriterStopped)));
                return;
            };
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
        }));
        self.push(priority, job)
    }

    /// Queue `job` in `priority`'s line.
    fn push(&self, priority: Priority, job: Job) -> Result<()> {
        let mut lines = self.shared.lines();
        if lines.closed {
            return Err(Error::WriterStopped);
        }
        lines.push(priority, job);
        drop(lines);
        self.shared.ready.notify_one();
        Ok(())
    }

    /// Closures waiting (interactive, background): the writer's queue
    /// depth, which the server's metrics report.
    #[must_use]
    pub fn waiting(&self) -> (usize, usize) {
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
    use crate::config::Config;
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
                        hold.recv().unwrap_or_default();
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

    /// A lease holds back every later closure; dropped unchanged it gives
    /// the same connection back, closed it stops the writer, and restored
    /// it hands over the new one.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_lease_holds_the_writer_until_it_returns_a_connection() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = Config::default();
        config.general.data_dir = dir.path().to_path_buf();
        let open = || WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        let writer = Arc::new(Writer::spawn(open()).unwrap_or_else(|e| fail(&e.to_string())));
        let created = writer
            .run(|db| db.execute_with_params("CREATE TABLE t (a INTEGER)", []))
            .await;
        assert!(created.is_ok(), "{created:?}");

        let lease = writer.lend().await.unwrap_or_else(|e| fail(&e.to_string()));
        assert!(lease.db().is_ok());
        let waiting = {
            let writer = Arc::clone(&writer);
            tokio::spawn(async move {
                writer
                    .run(|db| db.execute_with_params("INSERT INTO t VALUES (1)", []))
                    .await
            })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !waiting.is_finished(),
            "a closure ran while the connection was lent"
        );
        drop(lease);
        assert!(waiting.await.is_ok_and(|r| r.is_ok()));

        // Closed and reopened: the writer carries on with the new connection.
        let mut lease = writer.lend().await.unwrap_or_else(|e| fail(&e.to_string()));
        assert!(lease.db().is_ok_and(|db| db.checkpoint().is_ok()));
        lease.close();
        assert!(lease.db().is_err());
        lease.restore(open());
        drop(lease);
        let count = writer
            .run(|db| db.execute_query("SELECT count(*) FROM t").map(|r| r.rows))
            .await;
        assert!(
            count
                .as_ref()
                .is_ok_and(|rows| rows == &vec![vec![serde_json::json!(1)]]),
            "{count:?}"
        );

        // Closed for good: every later closure and lease is refused.
        let mut lease = writer.lend().await.unwrap_or_else(|e| fail(&e.to_string()));
        lease.close();
        drop(lease);
        assert!(matches!(
            writer.run(WorkspaceDb::list_tables).await,
            Err(Error::WriterStopped)
        ));
        let lease = writer.lend().await.unwrap_or_else(|e| fail(&e.to_string()));
        assert!(matches!(lease.db(), Err(Error::WriterStopped)));
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
