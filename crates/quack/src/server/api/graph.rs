//! The knowledge graph over REST: search and path (read), status, extract
//! (202, background, cost in the response), revalidate, review, and the
//! merge queue. Design doc 6.4 and 11.2.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use quack_core::graph::{
    GraphOptions, GraphResult, extract, resolve, store as graph_store, tables, traverse,
};
use quack_core::llm;
use quack_core::ontology::store as ontology_store;
use quack_core::storage::control::Outcome;
use serde::Deserialize;

use crate::server::auth::{Access, Identity, Need, access};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::{App, with_db};
use quack_core::analysis::tools::SharedDb;
use quack_core::ontology::Ontology;

#[derive(Deserialize, Default)]
pub(crate) struct SearchQuery {
    pub entity: Option<String>,
    pub class: Option<String>,
    pub relation: Option<String>,
    pub hops: Option<u32>,
}

/// A label's embedding for fuzzy entity resolution, when a model exists.
pub(crate) async fn query_embedding_for(app: &App, text: &str) -> Option<Vec<f32>> {
    let model = llm::optional_embedding_model(&app.config).await.ok()??;
    llm::embed_query(&model, text).await.ok()
}

async fn query_embedding(app: &App, text: &str) -> Option<Vec<f32>> {
    query_embedding_for(app, text).await
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
        Some(e) => query_embedding(&app, e).await,
        None => None,
    };
    let hops = q.hops.unwrap_or(2).max(1);
    let relation = q.relation.clone();
    let options = app.config.graph.options();
    let db = app.workspace_db(&id).await?;
    let detail =
        serde_json::json!({ "entity": entity, "class": class, "relation": relation, "hops": hops });
    let result = with_db(db, move |db| {
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
        .audit(&app, "graph", None, Outcome::Allowed, Some(detail))
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
    let a = query_embedding(&app, &from).await;
    let b = query_embedding(&app, &to).await;
    let max_hops = q.max_hops.unwrap_or(4).max(1);
    let options = app.config.graph.options();
    let db = app.workspace_db(&id).await?;
    let detail = serde_json::json!({ "from": from, "to": to, "max_hops": max_hops });
    let (from_label, to_label) = (from.clone(), to.clone());
    let result = with_db(db, move |db| {
        let from_nodes = traverse::resolve_entry(db, &from_label, None, a.as_deref())?;
        let to_nodes = traverse::resolve_entry(db, &to_label, None, b.as_deref())?;
        match (from_nodes.first(), to_nodes.first()) {
            (Some(a), Some(b)) => traverse::path(db, a, b, max_hops, &options),
            _ => Ok(GraphResult::default()),
        }
    })
    .await?;
    access
        .audit(&app, "graph", None, Outcome::Allowed, Some(detail))
        .await?;
    Ok(Json(serde_json::to_value(result)?))
}

pub(crate) async fn status(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, "list", "graph_status").await?;
    let db = app.workspace_db(&id).await?;
    let status = with_db(db, graph_store::status).await?;
    Ok(Json(serde_json::to_value(status)?))
}

#[derive(Deserialize, Default)]
pub(crate) struct ExtractRequest {
    /// `all` (default), `tables`, or `documents`.
    #[serde(default)]
    pub source: Option<String>,
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
    let (do_tables, do_documents) = match request.source.as_deref() {
        None | Some("all") => (true, true),
        Some("tables") => (true, false),
        Some("documents") => (false, true),
        Some(other) => return Err(ApiError::bad_request(format!("unknown source '{other}'"))),
    };
    let started = start_extraction(
        &app,
        &access,
        &id,
        &ExtractionPlan {
            tables: do_tables,
            documents: do_documents,
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
    pub tables: bool,
    pub documents: bool,
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
    let (do_tables, do_documents, sample, reset) =
        (plan.tables, plan.documents, plan.sample, plan.reset);
    let slot = app.begin_extraction(id).ok_or_else(|| {
        ApiError::new(
            StatusCode::CONFLICT,
            "a graph extraction is already running for this workspace",
        )
    })?;
    let db = app.workspace_db(id).await?;
    let (ontology, provisional) = with_db(Arc::clone(&db), |db| {
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
    let table_summaries = if do_tables {
        extract_tables_in_batches(&db, &ontology, provisional).await?
    } else {
        Vec::new()
    };
    let chunks = if do_documents {
        with_db(Arc::clone(&db), move |db| extract::chunks(db, sample)).await?
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
                "graph_extract",
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
    let run = uuid::Uuid::now_v7().to_string();
    access
        .audit(
            app,
            "graph_extract",
            Some(("graph_run", &run)),
            Outcome::Allowed,
            Some(serde_json::json!({ "tables": table_summaries, "cost": cost })),
        )
        .await?;
    spawn_document_extraction(
        Arc::clone(app),
        access.clone(),
        run.clone(),
        DocumentJob {
            db,
            chunks,
            extractor,
            ontology,
            provisional,
            embeddings,
            slot,
            options,
        },
    );
    Ok(
        serde_json::json!({ "tables": table_summaries, "cost": cost, "run": run, "status": "running" }),
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
    embeddings: Option<llm::EmbedModel>,
    /// Freed when the pass ends.
    slot: crate::server::state::ExtractionSlot,
    options: GraphOptions,
}

/// Run the model over the chunks, resolve, record the version, and audit
/// the run's end under `run_id`.
fn spawn_document_extraction(app: App, access: Access, run_id: String, job: DocumentJob) {
    tokio::spawn(async move {
        let DocumentJob {
            db,
            chunks,
            extractor,
            ontology,
            provisional,
            embeddings,
            options,
            slot,
        } = job;
        let outcome = extract::run(&db, chunks, extractor.as_ref(), &ontology, provisional).await;
        let version = ontology.version;
        let result = match outcome {
            Ok(summary) => {
                let resolution = resolve::resolve(&db, embeddings.as_ref(), &options).await;
                let finish = with_db(Arc::clone(&db), move |db| {
                    graph_store::set_built_with(db, version)
                })
                .await;
                match (resolution, finish) {
                    (Ok(resolution), Ok(())) => Ok((summary, resolution)),
                    (Err(e), _) => Err(e.to_string()),
                    (_, Err(e)) => Err(e.message),
                }
            }
            Err(e) => Err(e.to_string()),
        };
        let (outcome, detail) = match &result {
            Ok((summary, resolution)) => (
                Outcome::Allowed,
                serde_json::json!({ "finished": true, "summary": summary, "resolution": resolution }),
            ),
            Err(e) => (
                Outcome::Error,
                serde_json::json!({ "finished": true, "error": e }),
            ),
        };
        if let Err(e) = access
            .audit(
                &app,
                "graph_extract",
                Some(("graph_run", &run_id)),
                outcome,
                Some(detail),
            )
            .await
        {
            tracing::error!(error = %e.message, "audit write failed after graph extraction");
        }
        if let Err(e) = result {
            tracing::warn!(run = %run_id, error = %e, "graph extraction failed");
        }
        drop(slot);
    });
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
            "graph_revalidate",
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
        .audit(&app, "graph_review", None, Outcome::Allowed, None)
        .await?;
    Ok(Json(serde_json::to_value(status)?))
}

pub(crate) async fn merges(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, "list", "graph_merges").await?;
    let db = app.workspace_db(&id).await?;
    let pending = with_db(db, resolve::pending).await?;
    Ok(Json(serde_json::json!({ "merges": pending })))
}

#[derive(Deserialize)]
pub(crate) struct MergeDecision {
    /// `accept` or `reject`.
    pub action: String,
}

pub(crate) async fn decide_merge(
    State(app): State<App>,
    identity: Identity,
    Path((id, mid)): Path<(String, String)>,
    Json(body): Json<MergeDecision>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let accept = match body.action.as_str() {
        "accept" => true,
        "reject" => false,
        other => {
            return Err(ApiError::bad_request(format!(
                "action must be accept or reject, not '{other}'"
            )));
        }
    };
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let merge_id = mid.clone();
    let proposal = with_db(db, move |db| {
        if accept {
            resolve::accept(db, &merge_id, Some(&author))
        } else {
            resolve::reject(db, &merge_id, Some(&author))
        }
    })
    .await
    .map_err(|e| ApiError::not_found(e.message))?;
    access
        .audit(
            &app,
            "graph_merge",
            Some(("graph_merge", &mid)),
            Outcome::Allowed,
            Some(serde_json::json!({ "accept": accept, "keep": proposal.keep.label, "drop": proposal.drop.label })),
        )
        .await?;
    Ok(Json(serde_json::to_value(proposal)?))
}
