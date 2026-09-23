//! Progress of a long model run (ontology document evidence, graph
//! extraction, embeddings refresh), for the interface that shows it, and the
//! cancel token that stops it (issue #67).

use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::error::{Error, Result};

/// One unit of work finished (a chunk; for an embeddings refresh, a batch of chunks
/// or node labels), with the run's totals so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkDone {
    /// Units finished, this one included.
    pub done: u32,
    pub total: u32,
    /// Units that failed so far.
    pub failed: u32,
    /// How long this unit took.
    pub took: Duration,
    /// Since the run started.
    pub elapsed: Duration,
}

/// Called after every unit; `&|_| {}` when nobody is watching.
pub type Progress<'a> = &'a (dyn Fn(ChunkDone) + Sync);

/// What a long run reports to, and what stops it between units.
#[derive(Clone, Copy)]
pub struct RunControl<'a> {
    pub progress: Progress<'a>,
    pub cancel: Option<&'a CancellationToken>,
}

impl RunControl<'_> {
    /// Nobody watching, nothing to cancel.
    #[must_use]
    pub fn unobserved() -> RunControl<'static> {
        RunControl {
            progress: &|_| {},
            cancel: None,
        }
    }

    /// [`Error::Cancelled`] once the run has been cancelled.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when the token has fired.
    pub fn check(&self) -> Result<()> {
        if self.cancel.is_some_and(CancellationToken::is_cancelled) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}
