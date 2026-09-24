//! What graph extraction and ontology document evidence share: a model
//! that turns a passage into JSON of some shape, one bounded-concurrency
//! run over the passages with progress, an even sample of each document's
//! chunks, and counts per name.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

use futures::{Stream, StreamExt as _};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::progress::{ChunkDone, Progress};

/// Boxed future so an extractor can be a trait object.
pub type ExtractFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

/// Extraction of a `T` from one passage.
pub trait Extract<T>: Send + Sync {
    fn extract<'a>(&'a self, text: &'a str) -> ExtractFuture<'a, T>;
}

/// A model's answer read leniently: the first `{` to the last `}`, as `T`.
///
/// # Errors
///
/// Returns an error when no JSON object of that shape parses.
pub fn parse_answer<T: DeserializeOwned>(answer: &str) -> Result<T> {
    let (Some(start), Some(end)) = (answer.find('{'), answer.rfind('}')) else {
        return Err(Error::Ontology(String::from(
            "the model returned no JSON object",
        )));
    };
    serde_json::from_str(answer.get(start..=end).unwrap_or(answer))
        .map_err(|e| Error::Ontology(format!("the model's JSON does not parse: {e}")))
}

/// A passage an extraction run reads.
pub trait Passage: Sync {
    /// Its id, for logs.
    fn id(&self) -> &str;
    fn text(&self) -> &str;
}

/// One passage's extraction, and how long it took.
pub struct Extracted<'a, P, T> {
    pub passage: &'a P,
    pub outcome: Result<T>,
    pub took: Duration,
}

/// The extractions of `passages`, up to `concurrency` at a time, in the
/// passages' order.
pub fn extractions<'a, P: Passage, T: 'a>(
    extractor: &'a dyn Extract<T>,
    passages: &'a [P],
    concurrency: u32,
) -> impl Stream<Item = Extracted<'a, P, T>> + 'a {
    // Collected first: an iterator closure held across the awaits fails
    // the Send check when the run is inside a spawned task.
    let calls: Vec<_> = passages
        .iter()
        .map(|passage| async move {
            let began = Instant::now();
            let outcome = extractor.extract(passage.text()).await;
            Extracted {
                passage,
                outcome,
                took: began.elapsed(),
            }
        })
        .collect();
    futures::stream::iter(calls).buffered(usize::try_from(concurrency.max(1)).unwrap_or(1))
}

/// How far a run over `total` passages has got, reported as each finishes.
pub struct RunProgress<'p> {
    progress: Progress<'p>,
    started: Instant,
    total: u32,
    done: u32,
    failed: u32,
}

impl<'p> RunProgress<'p> {
    #[must_use]
    pub fn new(total: usize, progress: Progress<'p>) -> Self {
        Self {
            progress,
            started: Instant::now(),
            total: u32::try_from(total).unwrap_or(u32::MAX),
            done: 0,
            failed: 0,
        }
    }

    /// One more passage finished after `took`, successfully or not.
    pub fn finished(&mut self, took: Duration, succeeded: bool) {
        self.done = self.done.saturating_add(1);
        if !succeeded {
            self.failed = self.failed.saturating_add(1);
        }
        (self.progress)(ChunkDone {
            done: self.done,
            total: self.total,
            failed: self.failed,
            took,
            elapsed: self.started.elapsed(),
        });
    }

    #[must_use]
    pub const fn total(&self) -> u32 {
        self.total
    }

    #[must_use]
    pub const fn failed(&self) -> u32 {
        self.failed
    }

    /// Whether there was work and every passage of it failed.
    #[must_use]
    pub const fn all_failed(&self) -> bool {
        self.total > 0 && self.failed == self.total
    }
}

/// How many times each name was seen.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Tally(BTreeMap<String, u32>);

impl Tally {
    /// Count `name` once more.
    pub fn bump(&mut self, name: &str) {
        self.add(name, 1);
    }

    /// Count `name` `n` more times.
    pub fn add(&mut self, name: &str, n: u32) {
        let entry = self.0.entry(name.to_owned()).or_default();
        *entry = entry.saturating_add(n);
    }

    /// Add every count of `other`.
    pub fn absorb(&mut self, other: &Self) {
        for (name, n) in &other.0 {
            self.add(name, *n);
        }
    }

    #[must_use]
    pub fn get(&self, name: &str) -> u32 {
        self.0.get(name).copied().unwrap_or(0)
    }

    /// Distinct names counted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The names and their counts, in name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> {
        self.0.iter().map(|(name, n)| (name.as_str(), *n))
    }

    /// The names, in order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }

    /// The sum of every count.
    #[must_use]
    pub fn total(&self) -> u32 {
        self.0.values().fold(0, |sum, n| sum.saturating_add(*n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Deserialize)]
    struct Shape {
        n: u32,
    }

    #[test]
    fn an_answer_is_read_from_its_first_brace_to_its_last() {
        assert_eq!(
            parse_answer::<Shape>("Sure: {\"n\": 3} hope that helps").ok(),
            Some(Shape { n: 3 })
        );
        assert!(parse_answer::<Shape>("no json here").is_err());
        assert!(parse_answer::<Shape>("{\"m\": 1}").is_err());
    }

    #[test]
    fn a_tally_counts_and_merges() {
        let mut a = Tally::default();
        a.bump("vendor");
        a.bump("vendor");
        a.bump("country");
        let mut b = Tally::default();
        b.add("vendor", 3);
        a.absorb(&b);
        assert_eq!(a.get("vendor"), 5);
        assert_eq!(a.get("nothing"), 0);
        assert_eq!(a.total(), 6);
        assert_eq!(a.names().collect::<Vec<_>>(), ["country", "vendor"]);
        assert_eq!(
            serde_json::to_value(&a).ok(),
            Some(serde_json::json!({ "country": 1, "vendor": 5 }))
        );
    }
}
