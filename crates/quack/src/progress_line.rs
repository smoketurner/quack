//! Progress of a long run, one line per finished unit on stderr.

use std::io::Write;

use quack_core::progress::ChunkDone;

/// `12/40 done in 3 s, 1 failed; 41 s elapsed` on stderr. A failed write
/// is dropped: progress is a courtesy, never a reason to stop the run.
pub(crate) fn to_stderr(done: ChunkDone) {
    let failed = if done.failed > 0 {
        format!(", {} failed", done.failed)
    } else {
        String::new()
    };
    drop(writeln!(
        std::io::stderr(),
        "{}/{} done in {} s{failed}; {} s elapsed",
        done.done,
        done.total,
        done.took.as_secs(),
        done.elapsed.as_secs()
    ));
}
