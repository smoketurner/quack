//! The Search page (`/w/{id}/search`, issue #396): one search of the
//! documents without the model, each hit with its rank and score in the
//! vector and keyword legs and after reranking, so a person can see which
//! stage found or lost a passage. The query goes in a POST body: search text
//! is workspace content, and a URL ends up in logs.

use askama::Template;
use axum::extract::{Path, State};
use axum::response::Response;
use axum_extra::extract::Form as MultiForm;
use quack_core::analysis::citations::ChunkLocation;
use quack_core::analysis::search::SearchOutcome;
use quack_core::error::Result as CoreResult;
use quack_core::ids::WorkspaceId;
use quack_core::storage::control::{AuditAction, UserKind};
use quack_core::storage::workspace::{
    ChunkSearchResult, DocumentInfo, DocumentStatus, SearchMode, WorkspaceDb,
};
use quack_core::text::OneLine;
use serde::Deserialize;

use super::{Page, Tab, WebResult, WebUser, html};
use crate::server::api::query::SearchQuery;
use crate::server::auth::{Access, Need};
use crate::server::state::App;

/// Documents a picker offers at most, newest first; the rest are reached
/// by typing a name in the API or CLI.
pub(super) const PICKABLE_DOCUMENTS: usize = 200;

/// A ready document with text, as a document picker offers it.
pub(super) struct PickableDocument {
    pub id: String,
    pub name: String,
}

impl PickableDocument {
    /// The newest ready documents with text, up to [`PICKABLE_DOCUMENTS`].
    pub(super) fn read(db: &WorkspaceDb) -> CoreResult<Vec<Self>> {
        let (documents, _) = db.recent_documents(PICKABLE_DOCUMENTS)?;
        Ok(documents
            .iter()
            .filter(|d| d.status == DocumentStatus::Ready && d.chunk_count.unwrap_or(0) > 0)
            .map(Self::of)
            .collect())
    }

    fn of(document: &DocumentInfo) -> Self {
        Self {
            id: document.id.to_string(),
            name: OneLine(document.display_name()).to_string(),
        }
    }
}

/// The search form: the query, the documents picked, and the mode.
#[derive(Debug, Default, Deserialize)]
pub(super) struct SearchForm {
    #[serde(default)]
    query: String,
    #[serde(default)]
    documents: Vec<String>,
    #[serde(default)]
    mode: SearchMode,
    #[serde(default)]
    entity: String,
}

/// One hit, as the page's tables show it.
struct HitView {
    location: String,
    href: String,
    score: String,
    vector: String,
    keyword: String,
    rerank: String,
    excerpt: String,
}

/// Characters of a hit's text the page shows.
const EXCERPT_CHARS: usize = 400;

impl HitView {
    fn of(workspace: &WorkspaceId, hit: &ChunkSearchResult) -> Self {
        let ranks = &hit.ranks;
        let rank = |rank: Option<u32>, score: Option<f64>, digits: usize| match (rank, score) {
            (Some(rank), Some(score)) => format!("#{rank} ({score:.digits$})"),
            (Some(rank), None) => format!("#{rank}"),
            (None, _) => String::from("—"),
        };
        Self {
            location: ChunkLocation::from(hit).to_string(),
            href: format!(
                "/w/{workspace}/documents/{}/chunks/{}",
                hit.document_id, hit.chunk_index
            ),
            score: format!("{:.4}", hit.score),
            vector: rank(ranks.vector_rank, ranks.vector_score, 4),
            keyword: rank(ranks.keyword_rank, ranks.bm25, 3),
            rerank: rank(ranks.rerank_rank, ranks.rerank_score, 4),
            excerpt: hit.content.trim().chars().take(EXCERPT_CHARS).collect(),
        }
    }

    fn all(workspace: &WorkspaceId, hits: &[ChunkSearchResult]) -> Vec<Self> {
        hits.iter().map(|hit| Self::of(workspace, hit)).collect()
    }
}

/// What a search found, as the page shows it.
struct ResultView {
    hits: Vec<HitView>,
    vector: Vec<HitView>,
    keyword: Vec<HitView>,
    phrase_note: Option<String>,
    rerank: String,
}

impl ResultView {
    fn of(workspace: &WorkspaceId, outcome: &SearchOutcome) -> Self {
        let explanation = &outcome.explanation;
        Self {
            hits: HitView::all(workspace, &explanation.fused),
            vector: HitView::all(workspace, &explanation.vector),
            keyword: HitView::all(workspace, &explanation.keyword),
            phrase_note: explanation.phrase_note(),
            rerank: outcome.rerank.describe(),
        }
    }
}

#[derive(Template)]
#[template(path = "search.html")]
struct SearchPage {
    page: Page,
    documents: Vec<PickableDocument>,
    query: String,
    picked: Vec<String>,
    mode: SearchMode,
    entity: String,
    result: Option<ResultView>,
    error: Option<String>,
}

impl SearchPage {
    /// The modes the form offers, with their labels.
    const MODES: [(SearchMode, &'static str); 3] = [
        (SearchMode::Hybrid, "hybrid (vector and keyword)"),
        (SearchMode::Keyword, "keyword only"),
        (SearchMode::Vector, "vector only"),
    ];

    fn is_picked(&self, id: &str) -> bool {
        self.picked.iter().any(|p| p == id)
    }
}

pub(super) async fn search_page(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, AuditAction::Page, "search").await?;
    render(&app, &access, SearchForm::default(), None).await
}

pub(super) async fn search_run(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    MultiForm(form): MultiForm<SearchForm>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let q = SearchQuery {
        query: form.query.clone(),
        document_ids: form.documents.clone(),
        entity: Some(form.entity.clone()),
        mode: form.mode,
        explain: true,
        ..SearchQuery::default()
    };
    // A query the search refuses (an unknown document, no embedding model
    // for vector mode) is shown on the page, not as a failed page.
    let found = access.search(&app, &q).await.map_err(|e| e.message);
    render(&app, &access, form, Some(found)).await
}

async fn render(
    app: &App,
    access: &Access,
    form: SearchForm,
    found: Option<Result<SearchOutcome, String>>,
) -> WebResult<Response> {
    let workspace = access.membership.workspace.id.clone();
    let documents = app.read(&workspace, PickableDocument::read).await?;
    let (result, error) = match found {
        Some(Ok(outcome)) => (Some(ResultView::of(&workspace, &outcome)), None),
        Some(Err(e)) => (None, Some(e)),
        None => (None, None),
    };
    html(&SearchPage {
        page: Page::in_workspace(app, Tab::Search, access),
        documents,
        query: form.query,
        picked: form.documents,
        mode: form.mode,
        entity: form.entity,
        result,
        error,
    })
}
