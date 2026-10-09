//! Model work a turn starts once it is recorded (a session title, a
//! summary of the turns the history window leaves out), each on its own
//! task at background priority, so the answer never waits for it.

use std::sync::LazyLock;
use std::time::Duration;

use tokio_util::task::TaskTracker;

use crate::llm::acting::Acting;
use crate::llm::egress::Egress;
use crate::priority::Priority;

/// The follow-up tasks this process started.
static PENDING: LazyLock<TaskTracker> = LazyLock::new(TaskTracker::new);

/// What a recorded turn leaves running.
pub struct AfterTurn;

impl AfterTurn {
    /// Run `work` on its own task at background priority, carrying the
    /// caller's provider scope and acting person. A process that exits
    /// after its turn waits for it through [`Self::finish`].
    pub(crate) fn spawn(work: impl Future<Output = ()> + Send + 'static) {
        let (acting, egress) = (Acting::current(), Egress::current());
        PENDING.spawn(Acting::scope(
            acting,
            Egress::scope(egress, Priority::Background.scope(work)),
        ));
    }

    /// Wait, at most `limit`, for follow-ups still running: a command that
    /// answers one question and exits (`quack -p`, `saved run --refresh`)
    /// calls this after printing, or its runtime would drop them mid-call.
    /// A server or a terminal session outlives the tasks.
    pub async fn finish(limit: Duration) {
        PENDING.close();
        if tokio::time::timeout(limit, PENDING.wait()).await.is_err() {
            tracing::warn!(
                "a session title or history summary was still being written at exit; it is dropped"
            );
        }
    }
}
