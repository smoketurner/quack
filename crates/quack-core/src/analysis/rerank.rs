//! Reranking hook for retrieval: hybrid search over-fetches candidates,
//! a [`Reranker`] orders them by relevance to the query, and the top `k`
//! go to the model. The hook is a no-op unless `[retrieval].rerank`
//! names a provider; today that is `model`, the chat model ranking the
//! candidates listwise, so the air-gapped setup needs nothing beyond what
//! it already runs. A cross-encoder provider fits the same trait.

use std::future::Future;
use std::pin::Pin;

use crate::error::{Error, Result};
use crate::llm::stream_answer;
use crate::storage::workspace::ChunkSearchResult;

/// Boxed future so implementations can be trait objects.
pub type RankFuture<'a> = Pin<Box<dyn Future<Output = Result<Vec<usize>>> + Send + 'a>>;

/// Orders retrieval candidates by relevance to a query.
pub trait Reranker: Send + Sync {
    /// The candidate indices in relevance order, best first. Indices left
    /// out keep their fused order behind the ranked ones; indices out of
    /// range or repeated are ignored.
    fn rank<'a>(&'a self, query: &'a str, candidates: &'a [ChunkSearchResult]) -> RankFuture<'a>;

    /// A short name for the tool step summary (`reranked by model`).
    fn name(&self) -> &'static str;
}

/// Reorder `candidates` with `reranker` and keep the first `top_k`. A
/// failure in the reranker keeps the fused order: retrieval must not fail
/// because ranking did, so the error is logged and the summary says so.
pub async fn apply(
    reranker: &dyn Reranker,
    query: &str,
    candidates: Vec<ChunkSearchResult>,
    top_k: usize,
) -> (Vec<ChunkSearchResult>, RerankOutcome) {
    if candidates.len() <= 1 {
        let mut kept = candidates;
        kept.truncate(top_k);
        return (kept, RerankOutcome::Skipped);
    }
    match reranker.rank(query, &candidates).await {
        Ok(order) => {
            let mut kept = reorder(candidates, &order);
            kept.truncate(top_k);
            (kept, RerankOutcome::Reranked(reranker.name()))
        }
        Err(e) => {
            tracing::warn!(error = %e, reranker = reranker.name(), "reranking failed; keeping the fused order");
            let mut kept = candidates;
            kept.truncate(top_k);
            (kept, RerankOutcome::Failed(e.to_string()))
        }
    }
}

/// What `apply` did, for the tool step summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RerankOutcome {
    /// One candidate or none: nothing to order.
    Skipped,
    Reranked(&'static str),
    Failed(String),
}

/// `candidates` in `order`, then the rest in their original order.
fn reorder(candidates: Vec<ChunkSearchResult>, order: &[usize]) -> Vec<ChunkSearchResult> {
    let mut slots: Vec<Option<ChunkSearchResult>> = candidates.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(slots.len());
    for &i in order {
        if let Some(slot) = slots.get_mut(i)
            && let Some(chunk) = slot.take()
        {
            out.push(chunk);
        }
    }
    out.extend(slots.into_iter().flatten());
    out
}

/// Preamble for the chat model as a listwise reranker.
pub const RERANK_PROMPT: &str = "You rank passages by how well they answer a question. \
    You will get the question and numbered passages. Reply with only a JSON array of the \
    passage numbers, most relevant first, leaving out passages that do not help answer the \
    question. No prose.";

/// The user message for one ranking call: the query and numbered
/// passages, each cut to `max_chars`.
#[must_use]
pub fn rerank_request(query: &str, candidates: &[ChunkSearchResult], max_chars: usize) -> String {
    let mut text = format!("Question: {query}\n\nPassages:\n");
    for (i, chunk) in candidates.iter().enumerate() {
        let body: String = chunk.content.trim().chars().take(max_chars).collect();
        let n = i.saturating_add(1);
        text.push('[');
        text.push_str(&n.to_string());
        text.push_str("] ");
        text.push_str(&body);
        text.push_str("\n\n");
    }
    text
}

/// Parse the model's answer leniently: the first JSON array of numbers in
/// it, 1-based, mapped to 0-based indices below `len`.
///
/// # Errors
///
/// Returns an error when no array of numbers can be found.
pub fn parse_ranking(answer: &str, len: usize) -> Result<Vec<usize>> {
    let start = answer.find('[');
    let end = answer.rfind(']');
    let (Some(start), Some(end)) = (start, end) else {
        return Err(Error::Analysis(String::from(
            "the reranker returned no JSON array",
        )));
    };
    let slice = answer.get(start..=end).unwrap_or(answer);
    let numbers: Vec<serde_json::Value> = serde_json::from_str(slice)
        .map_err(|e| Error::Analysis(format!("the reranker's array does not parse: {e}")))?;
    let mut order = Vec::with_capacity(numbers.len());
    for value in numbers {
        let Some(n) = value.as_u64().and_then(|n| usize::try_from(n).ok()) else {
            continue;
        };
        if n >= 1 && n <= len {
            order.push(n.saturating_sub(1));
        }
    }
    Ok(order)
}

/// Characters of each passage shown to the ranking model.
pub const PASSAGE_CHARS: usize = 1200;

/// How long one ranking call may take.
const RERANK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// The chat model as a listwise reranker: one tool-less call with the
/// query and numbered passages, answered with the passage numbers in
/// order.
pub struct ModelReranker {
    agent: rig::agent::Agent,
}

impl ModelReranker {
    pub fn new<M>(model: M) -> Self
    where
        M: rig::completion::CompletionModel + Clone + Send + Sync + 'static,
    {
        Self {
            agent: rig::agent::AgentBuilder::new(model)
                .preamble(RERANK_PROMPT)
                .temperature(0.0)
                .build(),
        }
    }
}

impl Reranker for ModelReranker {
    fn rank<'a>(&'a self, query: &'a str, candidates: &'a [ChunkSearchResult]) -> RankFuture<'a> {
        Box::pin(async move {
            let request = rerank_request(query, candidates, PASSAGE_CHARS);
            let answer = stream_answer(&self.agent, &request, RERANK_TIMEOUT, "rerank").await?;
            parse_ranking(&answer, candidates.len())
        })
    }

    fn name(&self) -> &'static str {
        "model"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(n: u32) -> ChunkSearchResult {
        ChunkSearchResult {
            id: format!("c{n}"),
            content: format!("passage {n}"),
            document_id: String::from("d"),
            chunk_index: n,
            filename: String::from("f.md"),
            heading: None,
            page: None,
            score: 1.0,
        }
    }

    struct Reverse;
    impl Reranker for Reverse {
        fn rank<'a>(&'a self, _q: &'a str, c: &'a [ChunkSearchResult]) -> RankFuture<'a> {
            Box::pin(async move { Ok((0..c.len()).rev().collect()) })
        }
        fn name(&self) -> &'static str {
            "reverse"
        }
    }

    struct Broken;
    impl Reranker for Broken {
        fn rank<'a>(&'a self, _q: &'a str, _c: &'a [ChunkSearchResult]) -> RankFuture<'a> {
            Box::pin(async move { Err(Error::Analysis(String::from("boom"))) })
        }
        fn name(&self) -> &'static str {
            "broken"
        }
    }

    #[tokio::test]
    async fn apply_reorders_and_truncates() {
        let (kept, outcome) = apply(&Reverse, "q", vec![hit(1), hit(2), hit(3)], 2).await;
        assert_eq!(outcome, RerankOutcome::Reranked("reverse"));
        let ids: Vec<&str> = kept.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["c3", "c2"]);
    }

    #[tokio::test]
    async fn apply_keeps_the_fused_order_when_the_reranker_fails_or_has_one_candidate() {
        let (kept, outcome) = apply(&Broken, "q", vec![hit(1), hit(2)], 5).await;
        assert!(matches!(outcome, RerankOutcome::Failed(ref m) if m.contains("boom")));
        assert_eq!(kept.len(), 2);
        assert_eq!(kept.first().map(|c| c.id.as_str()), Some("c1"));
        let (kept, outcome) = apply(&Reverse, "q", vec![hit(1)], 5).await;
        assert_eq!(outcome, RerankOutcome::Skipped);
        assert_eq!(kept.len(), 1);
        let (kept, outcome) = apply(&Reverse, "q", Vec::new(), 5).await;
        assert_eq!(outcome, RerankOutcome::Skipped);
        assert!(kept.is_empty());
    }

    #[test]
    fn reorder_appends_unranked_and_ignores_bad_indices() {
        let out = reorder(vec![hit(1), hit(2), hit(3), hit(4)], &[2, 9, 2, 0]);
        let ids: Vec<&str> = out.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["c3", "c1", "c2", "c4"]);
    }

    #[test]
    fn parse_ranking_is_lenient_and_one_based() {
        assert_eq!(
            parse_ranking("Sure: [3, 1, 7, 0, 2]", 3).unwrap_or_default(),
            vec![2, 0, 1]
        );
        assert_eq!(
            parse_ranking("[]", 3).unwrap_or_default(),
            Vec::<usize>::new()
        );
        assert!(parse_ranking("no numbers here", 3).is_err());
        assert!(parse_ranking("[1, 2", 3).is_err());
    }

    #[test]
    fn request_numbers_and_truncates_passages() {
        let long = ChunkSearchResult {
            content: "x".repeat(50),
            ..hit(1)
        };
        let text = rerank_request("why?", &[long, hit(2)], 10);
        assert!(text.starts_with("Question: why?"));
        assert!(text.contains("[1] xxxxxxxxxx\n"));
        assert!(text.contains("[2] passage 2"));
    }
}
