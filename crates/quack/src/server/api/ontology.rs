//! The ontology: read, replace (a new version), versions, diff, restore.
//! Members write; viewers read; every write is audited as `ontology`.

use axum::Json;
use axum::extract::{Path, Query, State};
use quack_core::ontology::induction::{Decision, propose_from_tables};
use quack_core::storage::control::Outcome;
use serde::Deserialize;

use crate::server::auth::{Access, Identity, Need, access};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::{App, with_db};
use quack_core::ontology::{Ontology, candidates, documents, store};

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let _access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let current = with_db(db, store::current).await?;
    let ontology = current.ok_or_else(|| ApiError::not_found("no ontology yet"))?;
    Ok(Json(serde_json::to_value(ontology)?))
}

/// Validate the body and store it as the next version.
pub(crate) async fn replace(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let ontology = Ontology::from_json(&body.to_string())?;
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let stored = with_db(db, move |db| {
        store::save(db, &ontology, Some(&author), Some("imported"))
    })
    .await?;
    access
        .audit(
            &app,
            "ontology",
            Some(("ontology_version", &stored.version.to_string())),
            Outcome::Allowed,
            Some(serde_json::json!({ "version": stored.version, "classes": stored.classes.len() })),
        )
        .await?;
    Ok(Json(serde_json::to_value(stored)?))
}

/// Install the built-in general ontology as version 1.
pub(crate) async fn init(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let stored = with_db(db, move |db| {
        if store::latest_version(db)? > 0 {
            return Err(quack_core::error::Error::Ontology(String::from(
                "an ontology already exists",
            )));
        }
        store::save(
            db,
            &Ontology::builtin_default(),
            Some(&author),
            Some("built-in default"),
        )
    })
    .await
    .map_err(|e| ApiError::new(axum::http::StatusCode::CONFLICT, e.message))?;
    access
        .audit(
            &app,
            "ontology",
            Some(("ontology_version", "1")),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(Json(serde_json::to_value(stored)?))
}

#[derive(Deserialize)]
pub(crate) struct VersionsQuery {
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    20
}

pub(crate) async fn versions(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Query(q): Query<VersionsQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let _access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let limit = q.limit;
    let rows = with_db(db, move |db| store::versions(db, limit)).await?;
    Ok(Json(serde_json::json!({ "versions": rows })))
}

#[derive(Deserialize)]
pub(crate) struct DiffQuery {
    /// The older version to compare against; default: the one before.
    pub against: Option<u32>,
}

pub(crate) async fn version(
    State(app): State<App>,
    identity: Identity,
    Path((id, v)): Path<(String, u32)>,
    Query(q): Query<DiffQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let _access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let against = q.against.unwrap_or(v.saturating_sub(1));
    let (snapshot, older) = with_db(db, move |db| {
        let snapshot = store::version(db, v)?;
        let older = if against == 0 {
            None
        } else {
            store::version(db, against)?
        };
        Ok((snapshot, older))
    })
    .await?;
    let snapshot = snapshot.ok_or_else(|| ApiError::not_found("no such version"))?;
    let diff = older.map(|older| snapshot.diff(&older));
    Ok(Json(
        serde_json::json!({ "ontology": snapshot, "diff": diff }),
    ))
}

pub(crate) async fn restore(
    State(app): State<App>,
    identity: Identity,
    Path((id, v)): Path<(String, u32)>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let stored = with_db(db, move |db| store::restore(db, v, Some(&author)))
        .await
        .map_err(|e| ApiError::not_found(e.message))?;
    access
        .audit(
            &app,
            "ontology",
            Some(("ontology_version", &stored.version.to_string())),
            Outcome::Allowed,
            Some(serde_json::json!({ "restored": v, "version": stored.version })),
        )
        .await?;
    Ok(Json(serde_json::to_value(stored)?))
}

#[derive(Deserialize, Default)]
pub(crate) struct ProposeRequest {
    /// `full` (default) or `extend`.
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub auto_accept: bool,
    /// Also run open extraction over a sample of the documents. This costs
    /// one model call per sampled chunk and runs in the background: the
    /// response is 202 with the estimate, and the candidates appear in the
    /// queue when the run finishes.
    #[serde(default)]
    pub documents: bool,
    /// Chunks to sample (default from config).
    pub sample: Option<u32>,
}

/// Propose from table evidence into the review queue. Deterministic and
/// fast, so it answers 200 with the count rather than 202.
pub(crate) async fn propose(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    body: Option<Json<ProposeRequest>>,
) -> ApiResult<(axum::http::StatusCode, Json<serde_json::Value>)> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let request = body.map(|b| b.0).unwrap_or_default();
    let extend = match request.mode.as_deref() {
        None | Some("full") => false,
        Some("extend") => true,
        Some(other) => return Err(ApiError::bad_request(format!("unknown mode '{other}'"))),
    };
    if request.documents {
        let started = start_document_run(&app, &access, &id, extend, request.sample).await?;
        return Ok((axum::http::StatusCode::ACCEPTED, started));
    }
    let options = app.config.ontology.table_evidence();
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let outcome = with_db(db, move |db| {
        let current = store::current(db)?;
        let base = if extend { current.as_ref() } else { None };
        let proposals = propose_from_tables(db, base.or(current.as_ref()), &options)?;
        if proposals.is_empty() {
            return Ok((0, None, None));
        }
        let run = candidates::store_run(db, &proposals)?;
        let version = if request.auto_accept {
            Some(candidates::accept_all(db, Some(&author))?.version)
        } else {
            None
        };
        Ok((proposals.len(), Some(run), version))
    })
    .await?;
    let (count, run, version) = outcome;
    access
        .audit(
            &app,
            "propose",
            run.as_deref().map(|r| ("induction_run", r)),
            Outcome::Allowed,
            Some(serde_json::json!({ "candidates": count, "auto_accept": request.auto_accept, "version": version })),
        )
        .await?;
    Ok((
        axum::http::StatusCode::OK,
        Json(serde_json::json!({ "candidates": count, "run": run, "version": version })),
    ))
}

#[derive(Deserialize, Default)]
pub(crate) struct CandidatesQuery {
    /// `pending` (default) or `low_support`.
    pub status: Option<String>,
}

pub(crate) async fn list_candidates(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Query(q): Query<CandidatesQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let _access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let rows = match q.status.as_deref() {
        None | Some("pending") => with_db(db, candidates::pending).await?,
        Some("low_support") => with_db(db, candidates::low_support).await?,
        Some(other) => {
            return Err(ApiError::bad_request(format!(
                "status must be pending or low_support, not '{other}'"
            )));
        }
    };
    Ok(Json(serde_json::json!({ "candidates": rows })))
}

#[derive(Deserialize)]
pub(crate) struct DecideRequest {
    /// `accept`, `rename`, `merge_into`, `reparent`, or `reject`.
    pub action: String,
    /// The new id for `rename`, the target for `merge_into`, or the parent for `reparent`.
    pub target: Option<String>,
}

pub(crate) async fn decide(
    State(app): State<App>,
    identity: Identity,
    Path((id, cid)): Path<(String, String)>,
    Json(body): Json<DecideRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let target = || {
        body.target
            .clone()
            .filter(|t| !t.trim().is_empty())
            .ok_or_else(|| ApiError::bad_request("this action needs a target"))
    };
    let decision = match body.action.as_str() {
        "accept" => Some(Decision::Accept),
        "rename" => Some(Decision::Rename(target()?)),
        "merge_into" => Some(Decision::MergeInto(target()?)),
        "reparent" => Some(Decision::Reparent(target()?)),
        "reject" => None,
        other => return Err(ApiError::bad_request(format!("unknown action '{other}'"))),
    };
    let candidate_id = cid.clone();
    let version = with_db(db, move |db| {
        if let Some(decision) = decision {
            let stored = candidates::accept(db, &[(candidate_id, decision)], Some(&author))?;
            return Ok(Some(stored.version));
        }
        candidates::reject(db, &[candidate_id], Some(&author))?;
        Ok(None)
    })
    .await
    .map_err(|e| ApiError::bad_request(e.message))?;
    access
        .audit(
            &app,
            "ontology",
            Some(("candidate", &cid)),
            Outcome::Allowed,
            Some(serde_json::json!({ "action": body.action, "version": version })),
        )
        .await?;
    Ok(Json(
        serde_json::json!({ "candidate": cid, "action": body.action, "version": version }),
    ))
}

/// The document pass: answer 202 with the cost, then sample, extract, and
/// queue the candidates in a background task. The end of the run is
/// audited under the same run id.
pub(crate) async fn start_document_run(
    app: &App,
    access: &Access,
    id: &str,
    extend: bool,
    sample: Option<u32>,
) -> ApiResult<Json<serde_json::Value>> {
    let mut options = app.config.ontology.document_evidence();
    if let Some(n) = sample {
        options.sample_chunks = n;
    }
    // Fail now, not in the background, when no model can be built.
    let extractor = quack_core::llm::chat_extractor(&app.config).await?;
    let embeddings = quack_core::llm::optional_embedding_model(&app.config).await?;
    let db = app.workspace_db(id).await?;
    let (cost, chunks, current) = with_db(std::sync::Arc::clone(&db), move |db| {
        let cost = documents::estimate(db, &options)?;
        let chunks = documents::sample_chunks(db, options.sample_chunks)?;
        Ok((cost, chunks, store::current(db)?))
    })
    .await?;
    if chunks.is_empty() {
        return Err(ApiError::bad_request("no ready documents to sample"));
    }
    let run = uuid::Uuid::now_v7().to_string();
    access
        .audit(
            app,
            "propose",
            Some(("induction_run", &run)),
            Outcome::Allowed,
            Some(serde_json::json!({ "documents": true, "cost": cost })),
        )
        .await?;
    let app = std::sync::Arc::clone(app);
    let access = access.clone();
    let run_id = run.clone();
    tokio::spawn(async move {
        let base = if extend { current.as_ref() } else { None };
        let outcome = documents::run(
            chunks,
            extractor.as_ref(),
            base.or(current.as_ref()),
            &options,
            embeddings.as_ref(),
        )
        .await;
        let result = match outcome {
            Ok((found, summary)) => with_db(db, move |db| {
                candidates::store_run(db, &found)?;
                Ok(summary)
            })
            .await
            .map_err(|e| e.message),
            Err(e) => Err(e.to_string()),
        };
        let (outcome, detail) = match &result {
            Ok(summary) => (
                Outcome::Allowed,
                serde_json::json!({ "finished": true, "summary": summary }),
            ),
            Err(e) => (
                Outcome::Error,
                serde_json::json!({ "finished": true, "error": e }),
            ),
        };
        if let Err(e) = access
            .audit(
                &app,
                "propose",
                Some(("induction_run", &run_id)),
                outcome,
                Some(detail),
            )
            .await
        {
            tracing::error!(error = %e.message, "audit write failed after the document pass");
        }
        if let Err(e) = result {
            tracing::warn!(run = %run_id, error = %e, "document induction failed");
        }
    });
    Ok(Json(
        serde_json::json!({ "run": run, "cost": cost, "status": "running" }),
    ))
}
