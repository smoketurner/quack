//! The ontology: read, replace (a new version), versions, diff, restore.
//! Members write; viewers read; every write is audited as `ontology`.

use axum::Json;
use axum::extract::{Path, Query, State};
use quack_core::ontology::induction::{Decision, propose_from_tables};
use quack_core::storage::control::Outcome;
use serde::Deserialize;

use crate::server::auth::{Access, Identity, Need, access};
use crate::server::error::{ApiError, ApiResult};
use crate::server::queue::when_cancelled_unstarted;
use crate::server::state::{App, with_db};
use quack_core::jobs::{JobKind, JobSpec, Lane};
use quack_core::ontology::{Ontology, candidates, documents, store};

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, "list", "ontology").await?;
    let current = app.read(&id, store::current).await?;
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
    let access = access(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, "list", "ontology_versions").await?;
    let limit = q.limit;
    let rows = app.read(&id, move |db| store::versions(db, limit)).await?;
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
    let access = access(&app, identity, &id, Need::READ).await?;
    access
        .audit(
            &app,
            "open",
            Some(("ontology_version", &v.to_string())),
            Outcome::Allowed,
            None,
        )
        .await?;
    let against = q.against.unwrap_or(v.saturating_sub(1));
    let (snapshot, older) = app
        .read(&id, move |db| {
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
    /// Propose always adds only what the current ontology lacks (a full draft
    /// when there is none). `"extend"` names that and is accepted; any other
    /// value, `"full"` included, is refused, so a caller that relied on
    /// proposing over an existing ontology from scratch finds out.
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
    if let Some(mode) = request.mode.as_deref()
        && mode != "extend"
    {
        return Err(ApiError::bad_request(format!(
            "mode '{mode}' is not supported: propose always adds only what the current ontology lacks"
        )));
    }
    if request.documents {
        let started = start_document_run(&app, &access, &id, request.sample).await?;
        return Ok((axum::http::StatusCode::ACCEPTED, started));
    }
    let options = app.config.ontology.table_evidence();
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let outcome = with_db(db, move |db| {
        let current = store::current(db)?;
        let proposals = propose_from_tables(db, current.as_ref(), &options)?;
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
    let access = access(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, "list", "ontology_candidates")
        .await?;
    let rows = match q.status.as_deref() {
        None | Some("pending") => app.read(&id, candidates::pending).await?,
        Some("low_support") => app.read(&id, candidates::low_support).await?,
        Some(other) => {
            return Err(ApiError::bad_request(format!(
                "status must be pending or low_support, not '{other}'"
            )));
        }
    };
    Ok(Json(serde_json::json!({ "candidates": rows })))
}

#[derive(Deserialize, Default)]
pub(crate) struct DecideManyRequest {
    /// Candidate ids to accept as proposed.
    #[serde(default)]
    pub accept: Vec<String>,
    /// Candidate ids to reject.
    #[serde(default)]
    pub reject: Vec<String>,
}

/// Accept and reject candidates in bulk: one new version for every
/// acceptance together (issue #55).
pub(crate) async fn decide_many(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Json(body): Json<DecideManyRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    if body.accept.is_empty() && body.reject.is_empty() {
        return Err(ApiError::bad_request(
            "give candidate ids to accept or reject",
        ));
    }
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let (accept, reject) = (body.accept.clone(), body.reject.clone());
    let (version, rejected) = with_db(db, move |db| {
        let rejected = if reject.is_empty() {
            0
        } else {
            candidates::reject(db, &reject, Some(&author))?
        };
        let version = if accept.is_empty() {
            None
        } else {
            let decisions: Vec<(String, Decision)> = accept
                .into_iter()
                .map(|id| (id, Decision::Accept))
                .collect();
            Some(candidates::accept(db, &decisions, Some(&author))?.version)
        };
        Ok((version, rejected))
    })
    .await
    .map_err(|e| ApiError::bad_request(e.message))?;
    access
        .audit(
            &app,
            "ontology",
            None,
            Outcome::Allowed,
            Some(serde_json::json!({
                "accepted": body.accept,
                "rejected": body.reject,
                "version": version,
            })),
        )
        .await?;
    Ok(Json(serde_json::json!({
        "accepted": body.accept.len(),
        "rejected": rejected,
        "version": version,
    })))
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
    sample: Option<u32>,
) -> ApiResult<Json<serde_json::Value>> {
    let mut options = app.config.ontology.document_evidence();
    if let Some(n) = sample {
        options.sample_chunks = n;
    }
    // Fail now, not in the background, when no model can be built.
    let extractor = quack_core::llm::chat_extractor(&app.config).await?;
    let embeddings = quack_core::llm::optional_embedding_model(&app.config).await?;
    let (cost, chunks, current) = app
        .read(id, move |db| {
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
    let db = app.workspace_db(id).await?;
    let spec = JobSpec::new(JobKind::Ontology, "ontology document pass")
        .workspace(id)
        .owner(Some(access.identity.user_id.clone()))
        .lane(Lane::serial(format!("ontology:{id}")));
    let (cancel_app, cancel_access, cancel_run) =
        (std::sync::Arc::clone(app), access.clone(), run.clone());
    let jobs = app.jobs.clone();
    let app = std::sync::Arc::clone(app);
    let access = access.clone();
    let run_id = run.clone();
    let job = jobs.submit(spec, move |ctx| async move {
        let progress = |done: quack_core::progress::ChunkDone| {
            ctx.progress(done.done, done.total);
            tracing::info!(
                run = %run_id,
                done = done.done,
                total = done.total,
                failed = done.failed,
                "document evidence progress"
            );
        };
        let outcome = documents::run(
            chunks,
            extractor.as_ref(),
            current.as_ref(),
            &options,
            embeddings.as_ref(),
            app.config.analysis.extraction_concurrency,
            &progress,
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
        finish_document_run(&app, &access, &run_id, result).await
    });
    when_cancelled_unstarted(&jobs, job, move || async move {
        super::graph::audit_cancelled(
            &cancel_app,
            &cancel_access,
            "propose",
            "induction_run",
            &cancel_run,
        )
        .await;
    });
    Ok(Json(
        serde_json::json!({ "run": run, "cost": cost, "job": job, "status": "running" }),
    ))
}

/// Audit the end of a document pass under its run id and turn its result
/// into the job's outcome.
async fn finish_document_run(
    app: &App,
    access: &Access,
    run_id: &str,
    result: Result<documents::RunSummary, String>,
) -> quack_core::jobs::JobResult {
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
            app,
            "propose",
            Some(("induction_run", run_id)),
            outcome,
            Some(detail),
        )
        .await
    {
        tracing::error!(error = %e.message, "audit write failed after the document pass");
    }
    match result {
        Ok(summary) => Ok(format!(
            "{} candidates from {} chunks",
            summary.candidates, summary.sampled_chunks
        )),
        Err(e) => {
            tracing::warn!(run = %run_id, error = %e, "document induction failed");
            Err(e)
        }
    }
}
