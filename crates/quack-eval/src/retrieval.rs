//! Retrieval quality over the gold questions: each put to `quack search`
//! per leg, the expected chunk named by its file and a phrase it holds.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::quack::{Hit, Quack, SearchMode, progress};
use crate::share;

pub(crate) const WORKSPACE: &str = "eval";
const TOP_K: usize = 8;

#[derive(Debug, Deserialize)]
struct GoldExpected {
    filename: String,
    contains: String,
}

#[derive(Debug, Deserialize)]
struct GoldQuestion {
    question: String,
    kind: String,
    #[serde(default)]
    expected: Vec<GoldExpected>,
    /// The right answer is no result at all.
    #[serde(default)]
    expect_empty: bool,
}

impl GoldQuestion {
    /// Whether `hit` is one of the chunks the question expects.
    fn expected(&self, hit: &Hit) -> bool {
        self.expected
            .iter()
            .any(|e| hit.filename == e.filename && hit.content.contains(&e.contains))
    }

    fn recall_at(&self, ranked: &[Hit], k: usize) -> f64 {
        if self.expect_empty {
            return f64::from(u8::from(ranked.is_empty()));
        }
        if self.expected.is_empty() {
            return 0.0;
        }
        let found = ranked.iter().take(k).filter(|h| self.expected(h)).count();
        share(found, self.expected.len())
    }

    fn reciprocal_rank(&self, ranked: &[Hit]) -> f64 {
        if self.expect_empty {
            return f64::from(u8::from(ranked.is_empty()));
        }
        ranked
            .iter()
            .position(|h| self.expected(h))
            .map_or(0.0, |i| share(1, i.saturating_add(1)))
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub(crate) struct Metrics {
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_8: f64,
    pub(crate) mrr: f64,
    pub(crate) n: usize,
}

impl Metrics {
    fn of(pairs: &[(&GoldQuestion, Vec<Hit>)]) -> Self {
        let n = pairs.len();
        if n == 0 {
            return Self::default();
        }
        let mean = |f: &dyn Fn(&GoldQuestion, &[Hit]) -> f64| {
            pairs.iter().map(|(q, hits)| f(q, hits)).sum::<f64>() / share(n, 1)
        };
        Self {
            recall_at_1: mean(&|q, h| q.recall_at(h, 1)),
            recall_at_5: mean(&|q, h| q.recall_at(h, 5)),
            recall_at_8: mean(&|q, h| q.recall_at(h, 8)),
            mrr: mean(&|q, h| q.reciprocal_rank(h)),
            n,
        }
    }
}

/// The three legs; `similar` is `None` with no embedding model.
#[derive(Debug, Serialize)]
pub(crate) struct Legs {
    pub(crate) keyword: Metrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) similar: Option<Metrics>,
    pub(crate) hybrid: Metrics,
}

#[derive(Debug, Serialize)]
pub(crate) struct RetrievalReport {
    pub(crate) embedding: String,
    pub(crate) all: Legs,
    pub(crate) by_kind: BTreeMap<String, Legs>,
    pub(crate) n: usize,
}

impl RetrievalReport {
    pub(crate) fn print(&self, out: &mut impl Write) -> Result<()> {
        writeln!(
            out,
            "Retrieval quality ({} questions, embeddings from {})",
            self.n, self.embedding
        )?;
        writeln!(
            out,
            "{:<12} {:<8} {:>9} {:>9} {:>9} {:>7} {:>4}",
            "kind", "method", "recall@1", "recall@5", "recall@8", "mrr", "n"
        )?;
        let all = String::from("all");
        for (kind, legs) in std::iter::once((&all, &self.all)).chain(self.by_kind.iter()) {
            let mut rows = vec![("keyword", Some(&legs.keyword))];
            rows.push(("similar", legs.similar.as_ref()));
            rows.push(("hybrid", Some(&legs.hybrid)));
            for (name, m) in rows {
                match m {
                    Some(m) => writeln!(
                        out,
                        "{kind:<12} {name:<8} {:>9.3} {:>9.3} {:>9.3} {:>7.3} {:>4}",
                        m.recall_at_1, m.recall_at_5, m.recall_at_8, m.mrr, m.n
                    )?,
                    None => writeln!(out, "{kind:<12} {name:<8} {:>9}", "no embedding model")?,
                }
            }
        }
        Ok(())
    }
}

/// Run every gold question through each leg over the `eval` workspace,
/// which the caller has filled with the fixture's documents.
pub(crate) fn evaluate(
    quack: &Quack,
    fixture: &Path,
    embedding: &str,
    with_vectors: bool,
) -> Result<RetrievalReport> {
    let gold: Vec<GoldQuestion> = serde_json::from_str(&std::fs::read_to_string(
        fixture.join("gold_questions.json"),
    )?)?;
    progress(&format!(
        "Searching for {} gold questions, {} legs each...",
        gold.len(),
        if with_vectors { 3 } else { 2 }
    ));
    let mut keyword = Vec::with_capacity(gold.len());
    let mut similar = Vec::with_capacity(gold.len());
    let mut hybrid = Vec::with_capacity(gold.len());
    for q in &gold {
        let top = |hits: Vec<Hit>| hits.into_iter().take(TOP_K).collect::<Vec<_>>();
        keyword.push((
            q,
            top(quack.search(&q.question, WORKSPACE, SearchMode::Keyword, None)?),
        ));
        if with_vectors {
            similar.push((
                q,
                top(quack.search(&q.question, WORKSPACE, SearchMode::Vector, None)?),
            ));
        }
        hybrid.push((
            q,
            top(quack.search(&q.question, WORKSPACE, SearchMode::Hybrid, None)?),
        ));
    }
    let legs = |kind: Option<&str>| Legs {
        keyword: Metrics::of(&of_kind(&keyword, kind)),
        similar: with_vectors.then(|| Metrics::of(&of_kind(&similar, kind))),
        hybrid: Metrics::of(&of_kind(&hybrid, kind)),
    };
    let mut by_kind = BTreeMap::new();
    for q in &gold {
        if !by_kind.contains_key(&q.kind) {
            by_kind.insert(q.kind.clone(), legs(Some(&q.kind)));
        }
    }
    Ok(RetrievalReport {
        embedding: embedding.to_owned(),
        all: legs(None),
        by_kind,
        n: gold.len(),
    })
}

/// The pairs of one question kind, or all of them.
fn of_kind<'a>(
    pairs: &[(&'a GoldQuestion, Vec<Hit>)],
    kind: Option<&str>,
) -> Vec<(&'a GoldQuestion, Vec<Hit>)> {
    pairs
        .iter()
        .filter(|(q, _)| kind.is_none_or(|k| q.kind == k))
        .map(|(q, hits)| (*q, hits.clone()))
        .collect()
}
