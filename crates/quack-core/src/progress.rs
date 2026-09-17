//! Progress of a per-chunk model run (ontology document evidence, graph
//! extraction), for the interface that shows it (issue #67).

use std::time::Duration;

/// One chunk finished, with the run's totals so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkDone {
    /// Chunks finished, this one included.
    pub done: u32,
    pub total: u32,
    /// Chunks whose extraction failed so far.
    pub failed: u32,
    /// How long this chunk's call took.
    pub took: Duration,
    /// Since the run started.
    pub elapsed: Duration,
}

/// Called after every chunk; `&|_| {}` when nobody is watching.
pub type Progress<'a> = &'a (dyn Fn(ChunkDone) + Sync);
