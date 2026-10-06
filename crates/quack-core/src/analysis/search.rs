//! One document search for every interface (issue #396): the agent's
//! `search_documents`, REST and MCP `search`, `quack search`, the terminal's
//! `/search`, and the web Search page resolve the same request, documents,
//! entity, filter, and mode, the same way. A person may also limit a whole
//! question to certain documents (issue #399): [`DocumentScope`].

use std::fmt;

use crate::embedding::{Embedder, EmbeddingModel, Input, Vector};
use crate::error::{Error, Result};
use crate::graph;
use crate::graph::query::UnknownEntity;
use crate::ids::{ChunkId, DocumentId, NodeId};
use crate::storage::workspace::{
    ChunkScope, ChunkSearchResult, DocumentFilter, DocumentInfo, DocumentStatus, HybridLimits,
    SearchExplanation, SearchMode, WorkspaceDb,
};
use crate::text::OneLine;

use super::citations::ChunkLocation;
use super::rerank::{self, RerankOutcome, Reranked};
use super::tools::{ReaderDb, Rerank};

/// The most hits one search returns.
pub const MAX_TOP_K: u32 = 100;

/// One search of the documents, as a person or the model asked for it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DocumentSearch {
    pub query: String,
    /// Hits to return.
    pub top_k: u32,
    /// Documents to search within, each by id, id prefix, or file name;
    /// empty for every document.
    pub documents: Vec<String>,
    /// A graph entity whose source passages to search within.
    pub entity: Option<String>,
    pub filter: DocumentFilter,
    pub mode: SearchMode,
}

/// The vectors a search needs, embedded before it reaches the database.
#[derive(Debug, Clone, Default)]
pub struct SearchVectors {
    /// The query's; `None` without an embedding model or in keyword mode.
    pub query: Option<Vector>,
    /// The entity name's, for fuzzy resolution; `None` without one.
    pub entity: Option<Vector>,
}

/// A search's workings and how reranking went.
#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub explanation: SearchExplanation,
    pub rerank: RerankOutcome,
}

impl DocumentSearch {
    /// A search for `query` returning `top_k` hits, within bounds.
    ///
    /// # Errors
    ///
    /// An empty query.
    pub fn new(query: &str, top_k: u32) -> Result<Self> {
        let query = query.trim();
        if query.is_empty() {
            return Err(Error::Analysis(String::from("query must not be empty")));
        }
        Ok(Self {
            query: query.to_owned(),
            top_k: top_k.clamp(1, MAX_TOP_K),
            ..Self::default()
        })
    }

    /// What the search asked for, for a tool step or an audit row.
    #[must_use]
    pub fn describe(&self) -> String {
        let mut within = Vec::new();
        if let Some(entity) = &self.entity {
            within.push(format!("about {entity}"));
        }
        if !self.documents.is_empty() {
            within.push(format!("in {}", self.documents.join(", ")));
        }
        if !self.filter.is_empty() {
            within.push(format!(
                "where {}",
                serde_json::to_string(&self.filter).unwrap_or_default()
            ));
        }
        if self.mode != SearchMode::Hybrid {
            within.push(format!("{} only", self.mode));
        }
        if within.is_empty() {
            return self.query.clone();
        }
        format!("{} ({})", self.query, within.join(", "))
    }

    /// The query and entity vectors, embedded for someone waiting on them.
    ///
    /// # Errors
    ///
    /// The embedding model's error.
    pub async fn vectors<M: EmbeddingModel>(
        &self,
        embedder: Option<&Embedder<M>>,
    ) -> Result<SearchVectors> {
        let Some(embedder) = embedder else {
            return Ok(SearchVectors::default());
        };
        let query = match self.mode {
            SearchMode::Keyword => None,
            SearchMode::Hybrid | SearchMode::Vector => Some(
                embedder
                    .embed_interactive(&Input::Query(self.query.clone()))
                    .await?,
            ),
        };
        let entity = match &self.entity {
            Some(entity) => Some(embedder.similarity(entity).await?),
            None => None,
        };
        Ok(SearchVectors { query, entity })
    }

    /// The chunks the search may return: its documents (within `within`,
    /// a person's limit), its filter, and its entity's source passages.
    ///
    /// # Errors
    ///
    /// A document name that matches nothing, documents outside `within`,
    /// a bad filter, or an entity with no source passage.
    pub fn scope(
        &self,
        db: &WorkspaceDb,
        entity: Option<&Vector>,
        within: &DocumentScope,
    ) -> Result<ChunkScope> {
        let named = ChunkScope::for_documents(db, &self.documents)?
            .document_ids()
            .to_vec();
        let mut scope = ChunkScope::documents(within.narrow(named)?).with_filter(&self.filter)?;
        if let Some(name) = self.entity.as_deref() {
            scope = scope.and_chunks(Self::entity_chunks(db, name, entity)?);
        }
        Ok(scope)
    }

    /// The chunks an entity was extracted from. A name that resolves to
    /// nothing is an error naming the closest labels, and an entity that
    /// exists only in mapped tables says so: both beat an empty result read
    /// as "the documents do not cover this".
    pub(crate) fn entity_chunks(
        db: &WorkspaceDb,
        entity: &str,
        embedding: Option<&Vector>,
    ) -> Result<Vec<ChunkId>> {
        let nodes = graph::traverse::resolve_entry(db, entity, None, embedding)?;
        if nodes.is_empty() {
            let unknown = UnknownEntity::find(db, entity, embedding);
            return Err(if unknown.closest.is_empty() {
                Error::Analysis(format!(
                    "{unknown}; drop the entity argument to search every document"
                ))
            } else {
                unknown.into()
            });
        }
        let ids: Vec<NodeId> = nodes.iter().map(|n| n.id.clone()).collect();
        let chunks = graph::store::chunks_of_nodes(db, &ids)?;
        if chunks.is_empty() {
            return Err(Error::Analysis(format!(
                "'{entity}' is in the graph, but only from table rows, so no document passage is \
                 tied to it; search_graph has its connections, or drop the entity argument to search \
                 every document"
            )));
        }
        Ok(chunks)
    }

    /// Run the search on `db`, `limits.top_k` candidates deep, with its
    /// workings.
    ///
    /// # Errors
    ///
    /// A scope that does not resolve, or a failed leg.
    pub fn explain(
        &self,
        db: &WorkspaceDb,
        vectors: &SearchVectors,
        limits: HybridLimits,
        within: &DocumentScope,
    ) -> Result<SearchExplanation> {
        let scope = self.scope(db, vectors.entity.as_ref(), within)?;
        db.search_chunks(
            &self.query,
            vectors.query.as_ref(),
            self.mode,
            limits,
            &scope,
        )
    }

    /// Embed, search on a reader connection, and rerank: the whole search
    /// a person runs.
    ///
    /// # Errors
    ///
    /// The embedding model's error, a scope that does not resolve, or a
    /// failed leg. A failed reranker keeps the fused order instead.
    pub async fn run<M: EmbeddingModel>(
        &self,
        reader: &ReaderDb,
        embedder: Option<&Embedder<M>>,
        rerank: Option<&Rerank>,
        rrf_k: u32,
    ) -> Result<SearchOutcome> {
        let vectors = self.vectors(embedder).await?;
        let limits = HybridLimits {
            top_k: rerank.map_or(self.top_k, |r| self.top_k.max(r.candidates)),
            rrf_k,
        };
        let search = self.clone();
        let explanation = reader
            .with_db(move |db| search.explain(db, &vectors, limits, &DocumentScope::default()))
            .await?;
        Ok(SearchOutcome::rerank(explanation, rerank, &self.query, self.top_k).await)
    }
}

impl SearchOutcome {
    /// Rerank the fused hits when a reranker is set, and keep `top_k`.
    pub async fn rerank(
        mut explanation: SearchExplanation,
        rerank: Option<&Rerank>,
        query: &str,
        top_k: u32,
    ) -> Self {
        let keep = usize::try_from(top_k).unwrap_or(usize::MAX);
        let Some(rerank) = rerank else {
            explanation.fused.truncate(keep);
            return Self {
                explanation,
                rerank: RerankOutcome::Skipped,
            };
        };
        let fused = std::mem::take(&mut explanation.fused);
        let Reranked { results, outcome } =
            rerank::apply(rerank.reranker.as_ref(), query, fused, keep).await;
        explanation.fused = results;
        Self {
            explanation,
            rerank: outcome,
        }
    }

    /// The outcome as JSON: the hits, and with `explain` both legs, the
    /// phrase note, and the rerank outcome.
    #[must_use]
    pub fn to_json(&self, explain: bool) -> serde_json::Value {
        let mut value = serde_json::json!({ "chunks": self.explanation.fused });
        if explain && let serde_json::Value::Object(map) = &mut value {
            map.insert(
                String::from("explain"),
                serde_json::json!({
                    "vector": self.explanation.vector,
                    "keyword": self.explanation.keyword,
                    "phrases": self.explanation.phrases,
                    "phrase_note": self.explanation.phrase_note(),
                    "rerank": self.rerank.describe(),
                }),
            );
        }
        value
    }
}

/// How much of a search a text rendering shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchDetail {
    /// The hits, each with its rank in every leg.
    Hits,
    /// The hits, then each leg's candidates, the phrase note, and the
    /// rerank outcome.
    Workings,
}

/// Characters of a hit's text a rendering shows.
const EXCERPT_CHARS: usize = 300;

/// A search as a person reading a terminal sees it.
pub struct SearchReport<'a> {
    pub outcome: &'a SearchOutcome,
    pub detail: SearchDetail,
}

impl fmt::Display for SearchReport<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let explanation = &self.outcome.explanation;
        if explanation.fused.is_empty() {
            writeln!(f, "No passages found.")?;
        }
        for (i, hit) in explanation.fused.iter().enumerate() {
            let text: String = OneLine(hit.content.trim())
                .to_string()
                .chars()
                .take(EXCERPT_CHARS)
                .collect();
            writeln!(
                f,
                "{}. {} (chunk {}, score {:.4})\n   {}\n   {text}",
                i.saturating_add(1),
                ChunkLocation::from(hit),
                hit.chunk_index,
                hit.score,
                SearchOutcome::ranks(hit),
            )?;
        }
        if self.detail == SearchDetail::Hits {
            return Ok(());
        }
        for (name, leg) in [
            ("Vector", &explanation.vector),
            ("Keyword", &explanation.keyword),
        ] {
            writeln!(f, "\n{name} leg: {} candidates", leg.len())?;
            for (i, hit) in leg.iter().enumerate() {
                writeln!(
                    f,
                    "  #{} {} (chunk {}), {:.4}",
                    i.saturating_add(1),
                    ChunkLocation::from(hit),
                    hit.chunk_index,
                    hit.score
                )?;
            }
        }
        if let Some(note) = explanation.phrase_note() {
            writeln!(f, "\nPhrases: {note}")?;
        }
        writeln!(f, "\nRerank: {}", self.outcome.rerank.describe())
    }
}

impl SearchOutcome {
    /// The search for a person reading a terminal.
    #[must_use]
    pub fn render(&self, detail: SearchDetail) -> String {
        SearchReport {
            outcome: self,
            detail,
        }
        .to_string()
    }

    /// A hit's place in each ranking that found it.
    fn ranks(hit: &ChunkSearchResult) -> String {
        let ranks = &hit.ranks;
        let mut parts = Vec::new();
        if let (Some(rank), Some(score)) = (ranks.vector_rank, ranks.vector_score) {
            parts.push(format!("vector #{rank} ({score:.4})"));
        }
        if let (Some(rank), Some(bm25)) = (ranks.keyword_rank, ranks.bm25) {
            parts.push(format!("keyword #{rank} (bm25 {bm25:.3})"));
        }
        match (ranks.rerank_rank, ranks.rerank_score) {
            (Some(rank), Some(score)) => parts.push(format!("rerank #{rank} ({score:.4})")),
            (Some(rank), None) => parts.push(format!("rerank #{rank}")),
            (None, _) => {}
        }
        if parts.is_empty() {
            String::from("no leg ranks")
        } else {
            parts.join(", ")
        }
    }
}

/// One document a person limited a question to.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ScopedDocument {
    pub id: DocumentId,
    pub filename: String,
}

/// The documents a person limited a question to; empty for the whole
/// workspace. The model may narrow a search further within it, never
/// widen one past it.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct DocumentScope(Vec<ScopedDocument>);

impl DocumentScope {
    /// The ready documents `names` name, each by id, id prefix, or file
    /// name; the whole workspace when `names` is empty.
    ///
    /// # Errors
    ///
    /// A name that matches no document, or one that is not ready.
    pub fn resolve(db: &WorkspaceDb, names: &[String]) -> Result<Self> {
        if names.is_empty() {
            return Ok(Self::default());
        }
        let documents = db.list_documents()?;
        let mut scoped: Vec<ScopedDocument> = Vec::with_capacity(names.len());
        for name in names {
            let document = DocumentInfo::find(&documents, name)?;
            if document.status != DocumentStatus::Ready {
                return Err(Error::Analysis(format!(
                    "{} is {}, not ready, so a question cannot be limited to it",
                    OneLine(&document.filename),
                    document.status
                )));
            }
            if !scoped.iter().any(|d| d.id == document.id) {
                scoped.push(ScopedDocument {
                    id: document.id.clone(),
                    filename: document.filename.clone(),
                });
            }
        }
        Ok(Self(scoped))
    }

    /// Whether the whole workspace is in scope.
    #[must_use]
    pub fn is_everything(&self) -> bool {
        self.0.is_empty()
    }

    #[must_use]
    pub fn documents(&self) -> &[ScopedDocument] {
        &self.0
    }

    /// The documents a search of `named` (empty for every document) may
    /// cover within this scope.
    ///
    /// # Errors
    ///
    /// `named` lies wholly outside the scope: the model asked for documents
    /// the person left out.
    pub fn narrow(&self, named: Vec<DocumentId>) -> Result<Vec<DocumentId>> {
        if self.is_everything() {
            return Ok(named);
        }
        let ours: Vec<DocumentId> = self.0.iter().map(|d| d.id.clone()).collect();
        if named.is_empty() {
            return Ok(ours);
        }
        let kept: Vec<DocumentId> = named.into_iter().filter(|id| ours.contains(id)).collect();
        if kept.is_empty() {
            return Err(Error::Analysis(format!(
                "the person limited this question to {}; search within those, or leave \
                 document_ids out",
                self.names()
            )));
        }
        Ok(kept)
    }

    /// The documents' file names, comma-separated.
    #[must_use]
    pub fn names(&self) -> String {
        self.0
            .iter()
            .map(|d| OneLine(&d.filename).to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The system prompt's sentence on the scope, when there is one.
    #[must_use]
    pub fn prompt_note(&self) -> Option<String> {
        if self.is_everything() {
            return None;
        }
        let listed: Vec<String> = self
            .0
            .iter()
            .map(|d| format!("{} (id: {})", OneLine(&d.filename), d.id))
            .collect();
        Some(format!(
            "The person limited this question to these documents: {}. search_documents searches \
             only them; answer from them, and say so when they do not cover the question.",
            listed.join(", ")
        ))
    }
}

#[cfg(test)]
mod tests;
