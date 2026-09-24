//! The knowledge graph over REST: search and path (read), status, extract
//! (202, background, cost in the response), revalidate, review, and the
//! merge queue. Design doc 6.4 and 11.2.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use quack_core::embedding::{Input, Vector};
use quack_core::graph::resolve::MergeDecision;
use quack_core::graph::traverse::Hops;
use quack_core::graph::{
    ExtractSource, GraphOptions, GraphResult, extract, resolve, store as graph_store, tables,
    traverse,
};
use quack_core::llm;
use quack_core::ontology::store as ontology_store;
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use serde::Deserialize;

use crate::server::auth::{Access, Identity, Need, access};
use crate::server::error::{ApiError, ApiResult};
use crate::server::run::{BackgroundRun, GraphReport, RunKind};
use crate::server::state::{App, ExtractionSlot, with_db};
use quack_core::analysis::tools::SharedDb;
use quack_core::jobs::JobId;
use quack_core::ontology::Ontology;
use quack_core::progress::ChunkDone;

#[derive(Deserialize, Default)]
pub(crate) struct SearchQuery {
    pub entity: Option<String>,
    pub class: Option<String>,
    pub relation: Option<String>,
    pub hops: Option<u32>,
}

/// A label's embedding for fuzzy entity resolution, when a model exists.
/// The text's embedding for fuzzy entry: `None` when no embedding model
/// is configured, an error when the model fails.
pub(crate) async fn entity_embedding(app: &App, text: &str) -> ApiResult<Option<Vector>> {
    let Some(model) = llm::optional_embedding_model(&app.config).await? else {
        return Ok(None);
    };
    let input = Input::Similarity(text.to_owned());
    Ok(Some(model.embed_interactive(&input).await?))
}

pub(crate) async fn search(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Query(q): Query<SearchQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let entity = q
        .entity
        .as_deref()
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .map(str::to_owned);
    let class = q
        .class
        .as_deref()
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .map(str::to_owned);
    if entity.is_none() && class.is_none() {
        return Err(ApiError::bad_request("give entity, class, or both"));
    }
    let embedding = match &entity {
        Some(e) => entity_embedding(&app, e).await?,
        None => None,
    };
    let hops = Hops::neighborhood(q.hops);
    let relation = q.relation.clone();
    let options = app.config.graph.options();
    let detail =
        serde_json::json!({ "entity": entity, "class": class, "relation": relation, "hops": hops });
    let result = app
        .read(&id, move |db| {
            if let Some(entity) = entity {
                let roots =
                    traverse::resolve_entry(db, &entity, class.as_deref(), embedding.as_deref())?;
                return traverse::neighborhood(db, &roots, hops, relation.as_deref(), &options);
            }
            let ontology = ontology_store::current(db)?;
            traverse::by_class(
                db,
                ontology.as_ref(),
                class.as_deref().unwrap_or_default(),
                options.max_nodes,
                &options,
            )
        })
        .await?;
    access
        .audit(
            &app,
            AuditAction::Graph,
            None,
            Outcome::Allowed,
            Some(detail),
        )
        .await?;
    Ok(Json(serde_json::to_value(result)?))
}

#[derive(Deserialize)]
pub(crate) struct PathQuery {
    pub from: String,
    pub to: String,
    pub max_hops: Option<u32>,
}

pub(crate) async fn path(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Query(q): Query<PathQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    let from = q.from.trim().to_owned();
    let to = q.to.trim().to_owned();
    if from.is_empty() || to.is_empty() {
        return Err(ApiError::bad_request("from and to are both needed"));
    }
    let a = entity_embedding(&app, &from).await?;
    let b = entity_embedding(&app, &to).await?;
    let max_hops = Hops::path(q.max_hops);
    let options = app.config.graph.options();
    let detail = serde_json::json!({ "from": from, "to": to, "max_hops": max_hops });
    let (from_label, to_label) = (from.clone(), to.clone());
    let result = app
        .read(&id, move |db| {
            let from_nodes = traverse::resolve_entry(db, &from_label, None, a.as_deref())?;
            let to_nodes = traverse::resolve_entry(db, &to_label, None, b.as_deref())?;
            match (from_nodes.first(), to_nodes.first()) {
                (Some(a), Some(b)) => traverse::path(db, a, b, max_hops, &options),
                _ => Ok(GraphResult::default()),
            }
        })
        .await?;
    access
        .audit(
            &app,
            AuditAction::Graph,
            None,
            Outcome::Allowed,
            Some(detail),
        )
        .await?;
    Ok(Json(serde_json::to_value(result)?))
}

pub(crate) async fn status(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "graph_status")
        .await?;
    let status = app.read(&id, graph_store::status).await?;
    Ok(Json(serde_json::to_value(status)?))
}

#[derive(Deserialize, Default)]
pub(crate) struct ExtractRequest {
    /// `all` (default), `tables`, or `documents`.
    #[serde(default)]
    pub source: ExtractSource,
    /// Chunks to send to the model at most.
    pub sample: Option<u32>,
    #[serde(default)]
    pub reset: bool,
}

/// Build the graph. Tables run now; documents run in the background with
/// the cost (chunk count) in the 202 response and an audit row when done.
pub(crate) async fn extract(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    body: Option<Json<ExtractRequest>>,
) -> ApiResult<(StatusCode, Json<serde_json::Value>)> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let request = body.map(|b| b.0).unwrap_or_default();
    let started = start_extraction(
        &app,
        &access,
        &id,
        &ExtractionPlan {
            source: request.source,
            sample: request.sample,
            reset: request.reset,
        },
    )
    .await?;
    let code = if started.get("status").and_then(|s| s.as_str()) == Some("running") {
        StatusCode::ACCEPTED
    } else {
        StatusCode::OK
    };
    Ok((code, Json(started)))
}

/// What to extract.
pub(crate) struct ExtractionPlan {
    pub source: ExtractSource,
    pub sample: Option<u32>,
    pub reset: bool,
}

/// The extraction the API and the web page share. Returns the response
/// body: `{tables, status: "done"}` or `{tables, cost, run, status: "running"}`.
pub(crate) async fn start_extraction(
    app: &App,
    access: &Access,
    id: &str,
    plan: &ExtractionPlan,
) -> ApiResult<serde_json::Value> {
    let (sample, reset) = (plan.sample, plan.reset);
    let slot = app.begin_extraction(id).ok_or_else(|| {
        ApiError::new(
            StatusCode::CONFLICT,
            "a graph extraction is already running for this workspace",
        )
    })?;
    let db = app.workspace_db(id).await?;
    let (ontology, provisional) = app
        .read(id, |db| {
            Ok((
                ontology_store::current(db)?,
                ontology_store::current_is_auto_accepted(db)?,
            ))
        })
        .await?;
    let ontology = ontology.ok_or_else(|| ApiError::bad_request("no ontology yet"))?;
    let options = app.config.graph.options();
    let embeddings = llm::optional_embedding_model(&app.config).await?;
    if reset {
        with_db(Arc::clone(&db), graph_store::clear).await?;
    }
    let table_summaries = if plan.source.includes_tables() {
        extract_tables_in_batches(&db, &ontology, provisional).await?
    } else {
        Vec::new()
    };
    let chunks = if plan.source.includes_documents() {
        app.read(id, move |db| extract::chunks(db, sample)).await?
    } else {
        Vec::new()
    };
    if chunks.is_empty() {
        let version = ontology.version;
        let summary = resolve::resolve(&db, embeddings.as_ref(), &options).await?;
        with_db(Arc::clone(&db), move |db| {
            graph_store::set_built_with(db, version)
        })
        .await?;
        access
            .audit(
                app,
                AuditAction::GraphExtract,
                None,
                Outcome::Allowed,
                Some(serde_json::json!({ "tables": table_summaries, "resolution": summary })),
            )
            .await?;
        return Ok(
            serde_json::json!({ "tables": table_summaries, "resolution": summary, "status": "done" }),
        );
    }
    // Fail now, not in the background, when no model can be built.
    let extractor = llm::graph_extractor(&app.config, &ontology).await?;
    let cost = serde_json::json!({ "chunks": chunks.len(), "model": app.config.chat_model_ref()?.to_string() });
    let run = BackgroundRun::start(
        app,
        access,
        RunKind::Graph,
        serde_json::json!({ "tables": table_summaries, "cost": cost }),
    )
    .await?;
    let run_id = run.id().to_owned();
    let job = DocumentJob {
        db,
        chunks,
        extractor,
        ontology,
        provisional,
        embeddings,
        slot,
        options,
    }
    .run_in_background(run, Arc::clone(app));
    Ok(
        serde_json::json!({ "tables": table_summaries, "cost": cost, "run": run_id, "job": job, "status": "running" }),
    )
}

/// Table extraction one batch per lock hold, so other requests to the
/// workspace get in between batches of a large table (issue #48).
async fn extract_tables_in_batches(
    db: &SharedDb,
    ontology: &Ontology,
    provisional: bool,
) -> ApiResult<Vec<tables::MappingSummary>> {
    let mut summaries = Vec::with_capacity(ontology.mappings.len());
    for mapping in &ontology.mappings {
        let mut total = tables::MappingSummary {
            table: mapping.table.clone(),
            ..tables::MappingSummary::default()
        };
        let mut offset = 0;
        loop {
            let mapping = mapping.clone();
            let (batch, more) = with_db(Arc::clone(db), move |db| {
                tables::extract_batch(db, &mapping, provisional, offset)
            })
            .await?;
            total.absorb(&batch);
            if !more {
                break;
            }
            offset = offset.saturating_add(u64::from(tables::BATCH_ROWS));
        }
        summaries.push(total);
    }
    Ok(summaries)
}

/// Everything the background document pass needs.
struct DocumentJob {
    db: SharedDb,
    chunks: Vec<extract::ChunkText>,
    extractor: Box<dyn extract::GraphExtractor>,
    ontology: Ontology,
    provisional: bool,
    embeddings: Option<llm::Embeddings>,
    /// Freed when the pass ends.
    slot: ExtractionSlot,
    options: GraphOptions,
}

impl DocumentJob {
    /// Run the model over the chunks, resolve, and record the version, as
    /// `run`'s job with a chunk count for its progress. The extraction slot
    /// is held until the pass ends.
    fn run_in_background(self, run: BackgroundRun, app: App) -> JobId {
        let run_id = run.id().to_owned();
        run.submit(move |ctx| async move {
            let Self {
                db,
                chunks,
                extractor,
                ontology,
                provisional,
                embeddings,
                options,
                slot,
            } = self;
            let progress = |done: ChunkDone| {
                ctx.progress(done.done, done.total);
                tracing::info!(
                    run = %run_id,
                    done = done.done,
                    total = done.total,
                    failed = done.failed,
                    "graph extraction progress"
                );
            };
            let outcome = extract::run(
                &db,
                chunks,
                extractor.as_ref(),
                &ontology,
                provisional,
                app.config.analysis.extraction_concurrency,
                &progress,
            )
            .await;
            let version = ontology.version;
            let result = match outcome {
                Ok(summary) => {
                    let resolution = resolve::resolve(&db, embeddings.as_ref(), &options).await;
                    let finish = with_db(Arc::clone(&db), move |db| {
                        graph_store::set_built_with(db, version)
                    })
                    .await;
                    match (resolution, finish) {
                        (Ok(resolution), Ok(())) => Ok(GraphReport {
                            summary,
                            resolution,
                        }),
                        (Err(e), _) => Err(e.to_string()),
                        (_, Err(e)) => Err(e.message),
                    }
                }
                Err(e) => Err(e.to_string()),
            };
            drop(slot);
            result
        })
    }
}

pub(crate) async fn revalidate(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let db = app.workspace_db(&id).await?;
    let outcome = with_db(db, graph_store::revalidate)
        .await
        .map_err(|e| ApiError::bad_request(e.message))?;
    access
        .audit(
            &app,
            AuditAction::GraphRevalidate,
            None,
            Outcome::Allowed,
            Some(serde_json::to_value(&outcome)?),
        )
        .await?;
    Ok(Json(serde_json::to_value(outcome)?))
}

pub(crate) async fn review(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let db = app.workspace_db(&id).await?;
    let status = with_db(db, |db| {
        graph_store::mark_reviewed(db)?;
        graph_store::status(db)
    })
    .await?;
    access
        .audit(&app, AuditAction::GraphReview, None, Outcome::Allowed, None)
        .await?;
    Ok(Json(serde_json::to_value(status)?))
}

pub(crate) async fn merges(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "graph_merges")
        .await?;
    let pending = app.read(&id, resolve::pending).await?;
    Ok(Json(serde_json::json!({ "merges": pending })))
}

#[derive(Deserialize)]
pub(crate) struct DecideMerge {
    /// `accept` or `reject`; anything else is refused while the body is read.
    pub action: MergeDecision,
}

pub(crate) async fn decide_merge(
    State(app): State<App>,
    identity: Identity,
    Path((id, mid)): Path<(String, String)>,
    Json(body): Json<DecideMerge>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let decision = body.action;
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let merge_id = mid.clone();
    let proposal = with_db(db, move |db| {
        resolve::decide(db, &merge_id, decision, Some(&author))
    })
    .await?;
    access
        .audit(
            &app,
            AuditAction::GraphMerge,
            Some(ResourceKind::GraphMerge.id(&mid)),
            Outcome::Allowed,
            Some(serde_json::json!({ "accept": decision == MergeDecision::Accept, "keep": proposal.keep.label, "drop": proposal.drop.label })),
        )
        .await?;
    Ok(Json(serde_json::to_value(proposal)?))
}
