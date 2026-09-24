//! The ontology: read, replace (a new version), versions, diff, restore.
//! Members write; viewers read; every write is audited as `ontology`.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use quack_core::extraction::ExtractionRun;
use quack_core::ids::{RunId, WorkspaceId};
use quack_core::ontology::candidates::{CandidateAction, Queue};
use quack_core::ontology::induction::{Decision, propose_from_tables};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use serde::{Deserialize, Serialize};

use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::run::{BackgroundRun, RunKind};
use crate::server::state::{App, with_db};
use quack_core::llm::{self, Embeddings};
use quack_core::ontology::OntologyVersion;
use quack_core::ontology::store::Revision;
use quack_core::ontology::{Ontology, candidates, documents, store};
use quack_core::progress::{ChunkDone, RunControl};

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "ontology")
        .await?;
    let current = app.read(&id, store::current).await?;
    let ontology = current.ok_or_else(|| ApiError::not_found("no ontology yet"))?;
    Ok(Json(serde_json::to_value(ontology)?))
}

/// Validate the body and store it as the next version.
pub(crate) async fn replace(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<serde_json::Value>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let stored = access
        .replace_ontology(&app, &body.to_string(), "imported")
        .await?;
    Ok(Json(serde_json::to_value(stored)?))
}

/// Install the built-in general ontology as version 1.
pub(crate) async fn init(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let stored = access.init_ontology(&app).await?;
    Ok(Json(serde_json::to_value(stored)?))
}

/// The ontology writes the API and the web console share: each stores a
/// new version and audits it as `ontology`.
impl Access {
    /// Parse `json`, validate it, and store it as the next version; `note`
    /// says where it came from.
    pub(crate) async fn replace_ontology(
        &self,
        app: &App,
        json: &str,
        note: &'static str,
    ) -> ApiResult<Ontology> {
        let ontology = Ontology::from_json(json)?;
        let db = app.workspace_db(&self.workspace.id).await?;
        let author = self.identity.username.clone();
        let stored = with_db(db, move |db| {
            store::save(db, &ontology, Revision::reviewed(Some(&author), Some(note)))
        })
        .await?;
        self.audit(
            app,
            AuditAction::Ontology,
            Some(ResourceKind::OntologyVersion.id(&stored.saved_version()?.to_string())),
            Outcome::Allowed,
            Some(serde_json::json!({ "version": stored.version, "classes": stored.classes.len() })),
        )
        .await?;
        Ok(stored)
    }

    /// Install the built-in ontology as version 1; 409 once one exists.
    pub(crate) async fn init_ontology(&self, app: &App) -> ApiResult<Ontology> {
        let db = app.workspace_db(&self.workspace.id).await?;
        let author = self.identity.username.clone();
        let stored = with_db(db, move |db| {
            if store::latest_version(db)?.is_some() {
                return Ok(None);
            }
            store::save(
                db,
                &Ontology::builtin_default(),
                Revision::reviewed(Some(&author), Some("built-in default")),
            )
            .map(Some)
        })
        .await?
        .ok_or_else(|| ApiError::conflict("an ontology already exists"))?;
        self.audit(
            app,
            AuditAction::Ontology,
            Some(ResourceKind::OntologyVersion.id(&stored.saved_version()?.to_string())),
            Outcome::Allowed,
            None,
        )
        .await?;
        Ok(stored)
    }

    /// Store version `version` again as the newest.
    pub(crate) async fn restore_ontology(
        &self,
        app: &App,
        version: OntologyVersion,
    ) -> ApiResult<Ontology> {
        let db = app.workspace_db(&self.workspace.id).await?;
        let author = self.identity.username.clone();
        let stored = with_db(db, move |db| store::restore(db, version, Some(&author))).await?;
        self.audit(
            app,
            AuditAction::Ontology,
            Some(ResourceKind::OntologyVersion.id(&stored.saved_version()?.to_string())),
            Outcome::Allowed,
            Some(serde_json::json!({ "restored": version, "version": stored.version })),
        )
        .await?;
        Ok(stored)
    }
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
    Path(id): Path<WorkspaceId>,
    Query(q): Query<VersionsQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "ontology_versions")
        .await?;
    let limit = q.limit;
    let rows = app.read(&id, move |db| store::versions(db, limit)).await?;
    Ok(Json(serde_json::json!({ "versions": rows })))
}

#[derive(Deserialize)]
pub(crate) struct DiffQuery {
    /// The older version to compare against; default: the one before.
    pub against: Option<OntologyVersion>,
}

pub(crate) async fn version(
    State(app): State<App>,
    identity: Identity,
    Path((id, v)): Path<(WorkspaceId, OntologyVersion)>,
    Query(q): Query<DiffQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit(
            &app,
            AuditAction::Open,
            Some(ResourceKind::OntologyVersion.id(&v.to_string())),
            Outcome::Allowed,
            None,
        )
        .await?;
    let against = q.against.or_else(|| v.previous());
    let (snapshot, older) = app
        .read(&id, move |db| {
            let snapshot = store::version(db, v)?;
            let older = match against {
                Some(against) => store::version(db, against)?,
                None => None,
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
    Path((id, v)): Path<(WorkspaceId, OntologyVersion)>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let stored = access.restore_ontology(&app, v).await?;
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
    Path(id): Path<WorkspaceId>,
    body: Option<Json<ProposeRequest>>,
) -> ApiResult<(StatusCode, Json<serde_json::Value>)> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let request = body.map(|b| b.0).unwrap_or_default();
    if let Some(mode) = request.mode.as_deref()
        && mode != "extend"
    {
        return Err(ApiError::bad_request(format!(
            "mode '{mode}' is not supported: propose always adds only what the current ontology lacks"
        )));
    }
    if request.documents {
        let started = access.start_document_run(&app, request.sample).await?;
        return Ok((StatusCode::ACCEPTED, started));
    }
    let proposed = access
        .propose_from_tables(&app, request.auto_accept)
        .await?;
    Ok((StatusCode::OK, Json(serde_json::to_value(proposed)?)))
}

/// What proposing from the tables queued.
#[derive(Debug, Serialize)]
pub(crate) struct TableProposal {
    /// Candidates queued; zero when the ontology already covers the tables.
    pub candidates: usize,
    pub run: Option<RunId>,
    /// The version accepting them all made, with `auto_accept`.
    pub version: Option<OntologyVersion>,
}

/// Candidates decided one at a time.
#[derive(Debug, Serialize)]
pub(crate) struct CandidateDecided {
    pub candidate: String,
    pub action: CandidateAction,
    pub version: Option<OntologyVersion>,
}

/// Candidates decided in bulk.
#[derive(Debug, Serialize)]
pub(crate) struct CandidatesDecided {
    pub accepted: usize,
    pub rejected: usize,
    pub version: Option<OntologyVersion>,
}

/// Induction and the review queue, shared by the API and the web console.
impl Access {
    /// Queue what table evidence proposes beyond the current ontology,
    /// accepting it all at once with `auto_accept`.
    pub(crate) async fn propose_from_tables(
        &self,
        app: &App,
        auto_accept: bool,
    ) -> ApiResult<TableProposal> {
        let options = app.config.ontology.table_evidence();
        let db = app.workspace_db(&self.workspace.id).await?;
        let author = self.identity.username.clone();
        let proposed = with_db(db, move |db| {
            let current = store::current(db)?;
            let proposals = propose_from_tables(db, current.as_ref(), &options)?;
            if proposals.is_empty() {
                return Ok(TableProposal {
                    candidates: 0,
                    run: None,
                    version: None,
                });
            }
            let run = candidates::store_run(db, &proposals)?;
            let version = if auto_accept {
                candidates::accept_all(db, Some(&author))?.version
            } else {
                None
            };
            Ok(TableProposal {
                candidates: proposals.len(),
                run: Some(run),
                version,
            })
        })
        .await?;
        self.audit(
            app,
            AuditAction::Propose,
            proposed.run.as_ref().map(|r| ResourceKind::InductionRun.id(r)),
            Outcome::Allowed,
            Some(serde_json::json!({ "candidates": proposed.candidates, "auto_accept": auto_accept, "version": proposed.version })),
        )
        .await?;
        Ok(proposed)
    }

    /// Accept (as proposed or amended) or reject one candidate.
    pub(crate) async fn decide_candidate(
        &self,
        app: &App,
        candidate: &str,
        action: CandidateAction,
        target: Option<&str>,
    ) -> ApiResult<CandidateDecided> {
        let decision = action.decision(target)?;
        let db = app.workspace_db(&self.workspace.id).await?;
        let author = self.identity.username.clone();
        let candidate_id = candidate.to_owned();
        let version = with_db(db, move |db| {
            if let Some(decision) = decision {
                let stored = candidates::accept(db, &[(candidate_id, decision)], Some(&author))?;
                return Ok(stored.version);
            }
            candidates::reject(db, &[candidate_id], Some(&author))?;
            Ok(None)
        })
        .await?;
        self.audit(
            app,
            AuditAction::Ontology,
            Some(ResourceKind::Candidate.id(candidate)),
            Outcome::Allowed,
            Some(serde_json::json!({ "action": action, "version": version })),
        )
        .await?;
        Ok(CandidateDecided {
            candidate: candidate.to_owned(),
            action,
            version,
        })
    }

    /// Accept and reject candidates in bulk: one new version for every
    /// acceptance together (issue #55).
    pub(crate) async fn decide_candidates(
        &self,
        app: &App,
        accept: Vec<String>,
        reject: Vec<String>,
    ) -> ApiResult<CandidatesDecided> {
        if accept.is_empty() && reject.is_empty() {
            return Err(ApiError::bad_request(
                "choose at least one candidate to accept or reject",
            ));
        }
        let db = app.workspace_db(&self.workspace.id).await?;
        let author = self.identity.username.clone();
        let (accepting, rejecting) = (accept.clone(), reject.clone());
        let (version, rejected) = with_db(db, move |db| {
            let rejected = if rejecting.is_empty() {
                0
            } else {
                candidates::reject(db, &rejecting, Some(&author))?
            };
            let version = if accepting.is_empty() {
                None
            } else {
                let decisions: Vec<(String, Decision)> = accepting
                    .into_iter()
                    .map(|id| (id, Decision::Accept))
                    .collect();
                candidates::accept(db, &decisions, Some(&author))?.version
            };
            Ok((version, rejected))
        })
        .await?;
        self.audit(
            app,
            AuditAction::Ontology,
            None,
            Outcome::Allowed,
            Some(serde_json::json!({
                "accepted": accept,
                "rejected": reject,
                "version": version,
            })),
        )
        .await?;
        Ok(CandidatesDecided {
            accepted: accept.len(),
            rejected,
            version,
        })
    }
}

#[derive(Deserialize, Default)]
pub(crate) struct CandidatesQuery {
    /// `pending` (default) or `low_support`.
    #[serde(default)]
    pub status: Queue,
}

pub(crate) async fn list_candidates(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Query(q): Query<CandidatesQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "ontology_candidates")
        .await?;
    let queue = q.status;
    let rows = app
        .read(&id, move |db| candidates::queue(db, queue))
        .await?;
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
    Path(id): Path<WorkspaceId>,
    Json(body): Json<DecideManyRequest>,
) -> ApiResult<Json<CandidatesDecided>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    Ok(Json(
        access
            .decide_candidates(&app, body.accept, body.reject)
            .await?,
    ))
}

#[derive(Deserialize)]
pub(crate) struct DecideRequest {
    /// `accept`, `rename`, `merge_into`, `reparent`, or `reject`.
    pub action: CandidateAction,
    /// The new id for `rename`, the target for `merge_into`, or the parent for `reparent`.
    pub target: Option<String>,
}

pub(crate) async fn decide(
    State(app): State<App>,
    identity: Identity,
    Path((id, cid)): Path<(WorkspaceId, String)>,
    Json(body): Json<DecideRequest>,
) -> ApiResult<Json<CandidateDecided>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    Ok(Json(
        access
            .decide_candidate(&app, &cid, body.action, body.target.as_deref())
            .await?,
    ))
}

/// The document pass: answer 202 with the cost, then sample, extract, and
/// queue the candidates in a background task. The end of the run is
/// audited under the same run id.
impl Access {
    /// The document pass the API and the web console share: answer with
    /// the cost, then sample, extract, and queue the candidates as a
    /// background run.
    pub(crate) async fn start_document_run(
        &self,
        app: &App,
        sample: Option<u32>,
    ) -> ApiResult<Json<serde_json::Value>> {
        let (access, id) = (self, &self.workspace.id);
        let mut options = app.config.ontology.document_evidence();
        if let Some(n) = sample {
            options.sample_chunks = n;
        }
        // Fail now, not in the background, when no model can be built.
        let extractor = llm::chat_extractor(&app.config).await?;
        let embeddings = Embeddings::from_config(&app.config).await?;
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
        let db = app.workspace_db(id).await?;
        let run = BackgroundRun::start(
            app,
            access,
            RunKind::Ontology,
            serde_json::json!({ "documents": true, "cost": cost }),
        )
        .await?;
        let run_id = run.id().clone();
        let concurrency = app.config.analysis.extraction_concurrency;
        let progress_run = run_id.clone();
        let job = run.submit(move |ctx| async move {
            let run_id = progress_run;
            let progress = |done: ChunkDone| {
                ctx.progress(done.done, done.total);
                tracing::info!(
                    run = %run_id,
                    done = done.done,
                    total = done.total,
                    failed = done.failed,
                    "document evidence progress"
                );
            };
            let cancel = ctx.cancel_token();
            let control = RunControl {
                progress: &progress,
                cancel: Some(&cancel),
            };
            let outcome = documents::run(
                chunks,
                current.as_ref(),
                &options,
                embeddings.as_ref(),
                ExtractionRun {
                    extractor: extractor.as_ref(),
                    concurrency,
                    control,
                },
            )
            .await;
            match outcome {
                Ok((found, summary)) => with_db(db, move |db| {
                    candidates::store_run(db, &found)?;
                    Ok(summary)
                })
                .await
                .map_err(|e| e.message),
                Err(e) => Err(e.to_string()),
            }
        });
        Ok(Json(
            serde_json::json!({ "run": run_id, "cost": cost, "job": job, "status": "running" }),
        ))
    }
}
