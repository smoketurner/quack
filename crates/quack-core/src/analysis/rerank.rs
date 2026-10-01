//! Reranking hook for retrieval: hybrid search over-fetches candidates,
//! a [`Reranker`] orders them by relevance to the query, and the top `k`
//! go to the model. The hook is a no-op unless `[retrieval].rerank` names
//! one: `model`, the chat model ranking the candidates listwise, which an
//! air-gapped setup already runs; or `reranker`, a dedicated rerank model
//! (a cross-encoder) scoring each candidate through rig's `Rerank`
//! operation.

use std::future::Future;
use std::pin::Pin;

use rig::operation::RerankRequest;
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;

use crate::error::{Error, Result};
use crate::llm::{ChatModel, RerankModel, SchemaCall, Task};
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
) -> Reranked {
    if candidates.len() <= 1 {
        let mut results = candidates;
        results.truncate(top_k);
        return Reranked {
            results,
            outcome: RerankOutcome::Skipped,
        };
    }
    match reranker.rank(query, &candidates).await {
        Ok(order) => {
            let mut results = reorder(candidates, &order);
            results.truncate(top_k);
            Reranked {
                results,
                outcome: RerankOutcome::Reranked(reranker.name()),
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, reranker = reranker.name(), "reranking failed; keeping the fused order");
            let mut results = candidates;
            results.truncate(top_k);
            Reranked {
                results,
                outcome: RerankOutcome::Failed(e.to_string()),
            }
        }
    }
}

/// The candidates `apply` kept, in order, and what it did.
#[derive(Debug, Clone)]
pub struct Reranked {
    pub results: Vec<ChunkSearchResult>,
    pub outcome: RerankOutcome,
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
    You will get the question and numbered passages. Answer with the passage numbers in \
    `order`, most relevant first, leaving out passages that do not help answer the question.";

/// The chat model's ranking: passage numbers, 1-based, best first.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, JsonSchema)]
#[schemars(title = "rerank")]
pub struct RerankAnswer {
    pub order: Vec<u16>,
}

impl RerankAnswer {
    /// The order as 0-based indices below `len`; numbers out of range are
    /// dropped.
    fn indices(&self, len: usize) -> Vec<usize> {
        self.order
            .iter()
            .map(|&n| usize::from(n))
            .filter(|&n| n >= 1 && n <= len)
            .map(|n| n.saturating_sub(1))
            .collect()
    }
}

/// Characters of each passage shown to the ranking model.
pub const PASSAGE_CHARS: usize = 1200;

/// How long one ranking call may take.
const RERANK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// The chat model as a listwise reranker: one tool-less call with the
/// query and numbered passages, answered with the passage numbers in
/// order.
pub struct ModelReranker {
    call: SchemaCall<RerankAnswer>,
}

impl ModelReranker {
    #[must_use]
    pub fn new(model: ChatModel) -> Self {
        Self {
            call: SchemaCall::new(
                model,
                Task {
                    preamble: RERANK_PROMPT,
                    timeout: RERANK_TIMEOUT,
                    label: "rerank",
                },
                schema_for!(RerankAnswer),
            ),
        }
    }

    /// The user message for one ranking call: the query and numbered
    /// passages, each cut to `max_chars`.
    fn request(query: &str, candidates: &[ChunkSearchResult], max_chars: usize) -> String {
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
}

impl Reranker for ModelReranker {
    fn rank<'a>(&'a self, query: &'a str, candidates: &'a [ChunkSearchResult]) -> RankFuture<'a> {
        Box::pin(async move {
            let request = Self::request(query, candidates, PASSAGE_CHARS);
            let answer = self.call.answer(&request).await?;
            Ok(answer.indices(candidates.len()))
        })
    }

    fn name(&self) -> &'static str {
        "model"
    }
}

/// A dedicated rerank model: one `/rerank` call scores every candidate
/// against the query, and the order is the scores', best first.
pub struct ScoredReranker {
    model: RerankModel,
}

impl ScoredReranker {
    #[must_use]
    pub const fn new(model: RerankModel) -> Self {
        Self { model }
    }
}

impl Reranker for ScoredReranker {
    fn rank<'a>(&'a self, query: &'a str, candidates: &'a [ChunkSearchResult]) -> RankFuture<'a> {
        Box::pin(async move {
            let request = RerankRequest {
                query: query.to_owned(),
                documents: candidates.iter().map(|c| c.content.clone()).collect(),
            };
            #[expect(
                clippy::disallowed_methods,
                reason = "a rerank call, not an embedding; the lint guards embedding prefixes"
            )]
            let call = self.model.call(request);
            let mut response = tokio::time::timeout(RERANK_TIMEOUT, call)
                .await
                .map_err(|_| Error::Llm(String::from("the rerank model did not answer in time")))?
                .map_err(|e| Error::Llm(format!("the rerank model failed: {e}")))?;
            response
                .results
                .sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
            Ok(response.results.into_iter().map(|r| r.index).collect())
        })
    }

    fn name(&self) -> &'static str {
        "reranker"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::ids::{ChunkId, DocumentId};
    use crate::llm;

    fn hit(n: u32) -> ChunkSearchResult {
        ChunkSearchResult {
            id: ChunkId::from(format!("c{n}")),
            content: format!("passage {n}"),
            document_id: DocumentId::from("d"),
            chunk_index: n,
            filename: String::from("f.md"),
            heading: None,
            page: None,
            score: 1.0,
        }
    }

    struct Reverse;
    impl Reranker for Reverse {
        fn rank<'a>(
            &'a self,
            _query: &'a str,
            candidates: &'a [ChunkSearchResult],
        ) -> RankFuture<'a> {
            Box::pin(async move { Ok((0..candidates.len()).rev().collect()) })
        }
        fn name(&self) -> &'static str {
            "reverse"
        }
    }

    struct Broken;
    impl Reranker for Broken {
        fn rank<'a>(
            &'a self,
            _query: &'a str,
            _candidates: &'a [ChunkSearchResult],
        ) -> RankFuture<'a> {
            Box::pin(async move { Err(Error::Analysis(String::from("boom"))) })
        }
        fn name(&self) -> &'static str {
            "broken"
        }
    }

    #[tokio::test]
    async fn apply_reorders_and_truncates() {
        let Reranked {
            results: kept,
            outcome,
        } = apply(&Reverse, "q", vec![hit(1), hit(2), hit(3)], 2).await;
        assert_eq!(outcome, RerankOutcome::Reranked("reverse"));
        let ids: Vec<&str> = kept.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["c3", "c2"]);
    }

    #[tokio::test]
    async fn apply_keeps_the_fused_order_when_the_reranker_fails_or_has_one_candidate() {
        let Reranked {
            results: kept,
            outcome,
        } = apply(&Broken, "q", vec![hit(1), hit(2)], 5).await;
        assert!(matches!(outcome, RerankOutcome::Failed(ref m) if m.contains("boom")));
        assert_eq!(kept.len(), 2);
        assert_eq!(kept.first().map(|c| c.id.as_str()), Some("c1"));
        let Reranked {
            results: kept,
            outcome,
        } = apply(&Reverse, "q", vec![hit(1)], 5).await;
        assert_eq!(outcome, RerankOutcome::Skipped);
        assert_eq!(kept.len(), 1);
        let Reranked {
            results: kept,
            outcome,
        } = apply(&Reverse, "q", Vec::new(), 5).await;
        assert_eq!(outcome, RerankOutcome::Skipped);
        assert!(kept.is_empty());
    }

    /// A server that answers one request with `reply` and hands back the
    /// request it read, lowercased head and body.
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn answer_once(reply: &'static str) -> (String, tokio::task::JoinHandle<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let seen = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut read = Vec::new();
            let mut buf = [0_u8; 4096];
            while let Ok(n) = stream.read(&mut buf).await {
                read.extend_from_slice(buf.get(..n).unwrap_or_default());
                let text = String::from_utf8_lossy(&read).to_string();
                if n == 0 || text.ends_with('}') {
                    break;
                }
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                reply.len()
            );
            drop(stream.write_all(response.as_bytes()).await);
            String::from_utf8_lossy(&read).to_ascii_lowercase()
        });
        (base, seen)
    }

    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn a_rerank_model_scores_the_candidates_and_its_order_is_kept() {
        let (base, seen) = answer_once(
            r#"{"results":[{"index":0,"relevance_score":0.2},{"index":2,"relevance_score":0.9},{"index":1,"relevance_score":-1.5}]}"#,
        )
        .await;
        let config = Config::parse(&format!(
            "[retrieval]\nrerank = \"reranker\"\nrerank_model = \"tei/bge-reranker\"\n\
             [providers.tei]\ntype = \"openai\"\nbase_url = \"{base}\"\n\
             [providers.tei.headers]\nX-Team = \"quack\"\n"
        ))
        .unwrap();
        let model = config.rerank_model_ref().unwrap().unwrap();
        let reranker = ScoredReranker::new(llm::rerank_model_with(model, None).unwrap());
        let Reranked { results, outcome } =
            apply(&reranker, "refunds?", vec![hit(1), hit(2), hit(3)], 2).await;
        assert_eq!(outcome, RerankOutcome::Reranked("reranker"));
        let ids: Vec<&str> = results.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["c3", "c1"]);

        let request = seen.await.unwrap();
        assert!(request.starts_with("post /v1/rerank "), "{request}");
        assert!(request.contains("x-team: quack"), "{request}");
        assert!(
            !request.contains("authorization:"),
            "no key, no bearer: {request}"
        );
        let body = request.split("\r\n\r\n").nth(1).unwrap_or_default();
        let body: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(
            body,
            serde_json::json!({
                "model": "bge-reranker",
                "query": "refunds?",
                "documents": ["passage 1", "passage 2", "passage 3"],
            })
        );
    }

    #[test]
    fn reorder_appends_unranked_and_ignores_bad_indices() {
        let out = reorder(vec![hit(1), hit(2), hit(3), hit(4)], &[2, 9, 2, 0]);
        let ids: Vec<&str> = out.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["c3", "c1", "c2", "c4"]);
    }

    #[test]
    fn the_ranking_is_one_based_and_drops_numbers_out_of_range() {
        let answer: RerankAnswer =
            serde_json::from_str(r#"{"order": [3, 1, 7, 0, 2]}"#).unwrap_or_default();
        assert_eq!(answer.indices(3), [2, 0, 1]);
        assert!(RerankAnswer::default().indices(3).is_empty());
        assert!(serde_json::from_str::<RerankAnswer>("[1, 2]").is_err());
        let schema = serde_json::to_string(&schema_for!(RerankAnswer)).unwrap_or_default();
        assert!(schema.contains("\"order\""), "{schema}");
    }

    #[test]
    fn request_numbers_and_truncates_passages() {
        let long = ChunkSearchResult {
            content: "x".repeat(50),
            ..hit(1)
        };
        let text = ModelReranker::request("why?", &[long, hit(2)], 10);
        assert!(text.starts_with("Question: why?"));
        assert!(text.contains("[1] xxxxxxxxxx\n"));
        assert!(text.contains("[2] passage 2"));
    }
}
