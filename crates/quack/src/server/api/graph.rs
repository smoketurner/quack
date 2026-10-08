//! The knowledge graph over REST: search and path (read), status, extract
//! (202, background, cost in the response), revalidate with its preview,
//! review, and the merge queue. Design doc 6.4 and 11.2.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Response;
use quack_core::config::GraphConfig;
use quack_core::extraction::{Extract, ExtractionRun};
use quack_core::graph::export::{Destination, GraphExport, GraphFormat, ProvisionalExport};
use quack_core::graph::extract::ChunkPlan;
use quack_core::graph::resolve::{MergeDecision, MergeProposal, ResolutionSummary};
use quack_core::graph::store::{
    Asserted, Assertion, Keep, NewEdge, NewNode, NodeEdit, Revalidation,
};
use quack_core::graph::{
    Edge, ExtractSource, GraphResult, GraphStatus, Node, Properties, Standing, extract, resolve,
    store as graph_store, tables,
};
use quack_core::ids::{ClassId, EdgeId, NodeId, RelationId, RunId, WorkspaceId};
use quack_core::llm::{self, Embeddings};
use quack_core::okf;
use quack_core::ontology::store as ontology_store;
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::server::api::okf::Download;
use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::run::{BackgroundRun, GraphReport, RunKind};
use crate::server::state::{App, ExtractionSlot, with_db};
use quack_core::analysis::tools::{FindPathArgs, SearchGraphArgs, SharedDb};
use quack_core::jobs::JobId;
use quack_core::ontology::{Ontology, OntologyVersion};
use quack_core::progress::{ChunkDone, RunControl};

/// An entity's neighborhood, or a class's members.
///
/// The search and path bodies are [`SearchGraphArgs`] and [`FindPathArgs`]:
/// in the body, never the query string, since entity names are workspace
/// content and a URL ends up in logs.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/graph/search",
    tag = "graph",
    request_body = SearchGraphArgs,
    params(WorkspaceId),
    responses((status = 200, description = "The entity's neighborhood, or a class's members", body = GraphResult)),
)]
pub(crate) async fn search(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(q): Json<SearchGraphArgs>,
) -> ApiResult<Json<GraphResult>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let query = q
        .query()
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    let options = app.config.graph;
    let detail = serde_json::to_value(&query)?;
    let model = access
        .model(
            &app,
            AuditAction::Graph,
            Embeddings::from_config(&app.config).await,
        )
        .await?;
    // Run, audit the outcome, then propagate, so a failure after
    // authorization is recorded as an error rather than dropped.
    let result: ApiResult<_> = async {
        let embedding = query.embedding(model.as_ref()).await?;
        app.read(&id, move |db| query.run(db, embedding.as_ref(), &options))
            .await
    }
    .await;
    access
        .audit(
            &app,
            AuditAction::Graph,
            None,
            Outcome::of(&result),
            Some(detail),
        )
        .await?;
    Ok(Json(result?))
}

/// The shortest path between two entities.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/graph/path",
    tag = "graph",
    request_body = FindPathArgs,
    params(WorkspaceId),
    responses((status = 200, description = "The path", body = GraphResult)),
)]
pub(crate) async fn path(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(q): Json<FindPathArgs>,
) -> ApiResult<Json<GraphResult>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let query = q
        .query()
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    let options = app.config.graph;
    let detail = serde_json::to_value(&query)?;
    let model = access
        .model(
            &app,
            AuditAction::Graph,
            Embeddings::from_config(&app.config).await,
        )
        .await?;
    // Run, audit the outcome, then propagate (see `search`).
    let result: ApiResult<_> = async {
        let ends = query.embeddings(model.as_ref()).await?;
        app.read(&id, move |db| query.run(db, &ends, &options))
            .await
    }
    .await;
    access
        .audit(
            &app,
            AuditAction::Graph,
            None,
            Outcome::of(&result),
            Some(detail),
        )
        .await?;
    Ok(Json(result?))
}

/// The graph's size, and what is stale or pending.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/graph/status",
    tag = "graph",
    params(WorkspaceId),
    responses((status = 200, description = "The graph's status", body = GraphStatus)),
)]
pub(crate) async fn status(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<GraphStatus>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "graph_status")
        .await?;
    let status = app.read(&id, graph_store::status).await?;
    Ok(Json(status))
}

/// `GET .../graph/export?format=csv|graphml|jsonld&include_provisional=true`.
#[derive(Deserialize, Default, utoipa::IntoParams)]
pub(crate) struct ExportQuery {
    #[serde(default)]
    format: GraphFormat,
    /// Also export what an auto-accepted ontology version produced.
    #[serde(default)]
    include_provisional: bool,
}

/// The whole graph as an attachment (a tar of the CSV bundle, `GraphML`, or
/// JSON-LD), streamed like the OKF export and audited as `export` with
/// `{format, nodes, edges, provenance}` when it ends.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/graph/export",
    tag = "graph",
    params(WorkspaceId, ExportQuery),
    responses((status = 200, description = "The graph, as an attachment in `format`", content(
        (Vec<u8> = "application/x-tar"),
        (String = "application/graphml+xml"),
        (Object = "application/ld+json"),
    ))),
)]
pub(crate) async fn export(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Query(query): Query<ExportQuery>,
) -> ApiResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let export = GraphExport {
        format: query.format,
        provisional: ProvisionalExport::from(query.include_provisional),
    };
    let download = Download {
        filename: format!(
            "{}.graph.{}",
            okf::slug(&access.membership.workspace.name),
            export.format.extension()
        ),
        content_type: export.format.content_type(),
        format: export.format.as_str(),
    };
    download
        .stream(&app, access, id, move |db, body| {
            let summary = export.write(db, Destination::Stream(body))?;
            Ok(serde_json::to_value(summary)?)
        })
        .await
}

#[derive(Deserialize, Default, ToSchema)]
pub(crate) struct ExtractRequest {
    /// `all` (default), `tables`, or `documents`.
    #[serde(default)]
    pub source: ExtractSource,
    /// Chunks to send to the model at most.
    pub sample: Option<u32>,
    #[serde(default)]
    pub reset: bool,
    /// With `reset`: drop what people asserted too.
    #[serde(default)]
    pub all: bool,
}

/// Build the graph. Tables run now; documents run in the background with
/// the cost (chunk count) in the 202 response and an audit row when done.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/graph/extract",
    tag = "graph",
    request_body(content = Option<ExtractRequest>, description = "Optional; everything when absent"),
    params(WorkspaceId),
    responses(
        (status = 200, description = "Tables only, done", body = ExtractionStarted),
        (status = 202, description = "Documents run in the background", body = ExtractionStarted),
    ),
)]
pub(crate) async fn extract(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    body: Option<Json<ExtractRequest>>,
) -> ApiResult<(StatusCode, Json<ExtractionStarted>)> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let request = body.map(|b| b.0).unwrap_or_default();
    let started = access
        .start_extraction(
            &app,
            &ExtractionPlan {
                source: request.source,
                sample: request.sample,
                reset: request.reset.then_some(if request.all {
                    Keep::Nothing
                } else {
                    Keep::Asserted
                }),
            },
        )
        .await?;
    Ok((started.status_code(), Json(started)))
}

/// What starting an extraction did: table work finishes within the
/// request; document work goes on in the background as a run.
#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum ExtractionStarted {
    Done {
        tables: Vec<tables::MappingSummary>,
        resolution: ResolutionSummary,
    },
    Running {
        tables: Vec<tables::MappingSummary>,
        cost: ExtractionCost,
        run: RunId,
        job: JobId,
    },
}

impl ExtractionStarted {
    /// 200 when it finished, 202 when a run goes on.
    pub(crate) fn status_code(&self) -> StatusCode {
        match self {
            Self::Done { .. } => StatusCode::OK,
            Self::Running { .. } => StatusCode::ACCEPTED,
        }
    }
}

/// The model calls a document pass will make.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub(crate) struct ExtractionCost {
    pub chunks: usize,
    pub model: String,
}

/// What to extract.
pub(crate) struct ExtractionPlan {
    pub source: ExtractSource,
    pub sample: Option<u32>,
    /// Clear the graph first, keeping this much.
    pub reset: Option<Keep>,
}

impl Access {
    /// The extraction the API and the web page share: tables within the
    /// request, documents as a background run.
    pub(crate) async fn start_extraction(
        &self,
        app: &App,
        plan: &ExtractionPlan,
    ) -> ApiResult<ExtractionStarted> {
        let (access, id) = (self, &self.membership.workspace.id);
        let (sample, reset) = (plan.sample, plan.reset);
        let slot = app.begin_extraction(id).ok_or_else(|| {
            ApiError::conflict("a graph extraction is already running for this workspace")
        })?;
        let db = app.workspace_db(id).await?;
        let (ontology, standing) = app
            .read(id, |db| {
                Ok((
                    ontology_store::current(db)?,
                    ontology_store::current_standing(db)?,
                ))
            })
            .await?;
        let ontology = ontology.ok_or_else(|| ApiError::bad_request("no ontology yet"))?;
        let version = ontology.saved_version()?;
        let options = app.config.graph;
        let embeddings = access
            .model(
                app,
                AuditAction::GraphExtract,
                Embeddings::from_config(&app.config).await,
            )
            .await?;
        if let Some(keep) = reset {
            with_db(Arc::clone(&db), move |db| graph_store::clear(db, keep)).await?;
        }
        let table_summaries = if plan.source.includes_tables() {
            extract_tables_in_batches(&db, &ontology, standing).await?
        } else {
            Vec::new()
        };
        let chunks = if plan.source.includes_documents() {
            app.read(id, move |db| ChunkPlan::new(db, sample)).await?
        } else {
            ChunkPlan::Sample(Vec::new())
        };
        if chunks.is_empty() {
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
            return Ok(ExtractionStarted::Done {
                tables: table_summaries,
                resolution: summary,
            });
        }
        // Fail now, not in the background, when no model can be built.
        let extractor = access
            .model(
                app,
                AuditAction::GraphExtract,
                llm::graph_extractor(&app.config, &ontology).await,
            )
            .await?;
        let cost = ExtractionCost {
            chunks: chunks.len(),
            model: app.config.chat_model_ref()?.to_string(),
        };
        let run = BackgroundRun::start(
            app,
            access,
            RunKind::GRAPH,
            serde_json::json!({ "tables": table_summaries, "cost": cost }),
        )
        .await?;
        let run_id = run.id().clone();
        let job = DocumentJob {
            db,
            chunks,
            extractor,
            ontology,
            version,
            standing,
            embeddings,
            slot,
            options,
        }
        .run_in_background(run, Arc::clone(app));
        Ok(ExtractionStarted::Running {
            tables: table_summaries,
            cost,
            run: run_id,
            job,
        })
    }
}

/// Table extraction one batch per lock hold, so other requests to the
/// workspace get in between batches of a large table (issue #48).
async fn extract_tables_in_batches(
    db: &SharedDb,
    ontology: &Ontology,
    standing: Standing,
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
            let batch = with_db(Arc::clone(db), move |db| {
                tables::extract_batch(db, &mapping, standing, offset)
            })
            .await?;
            total.absorb(&batch.summary);
            let Some(next) = batch.next_offset else {
                break;
            };
            offset = next;
        }
        summaries.push(total);
    }
    Ok(summaries)
}

/// Everything the background document pass needs.
struct DocumentJob {
    db: SharedDb,
    chunks: ChunkPlan,
    extractor: Box<dyn Extract<extract::Extraction>>,
    ontology: Ontology,
    /// The ontology's saved version, which the graph records when the pass
    /// ends.
    version: OntologyVersion,
    standing: Standing,
    embeddings: Option<Embeddings>,
    /// Freed when the pass ends.
    slot: ExtractionSlot,
    options: GraphConfig,
}

impl DocumentJob {
    /// Run the model over the chunks, resolve, and record the version, as
    /// `run`'s job with a chunk count for its progress. The extraction slot
    /// is held until the pass ends.
    fn run_in_background(self, run: BackgroundRun, app: App) -> JobId {
        let run_id = run.id().clone();
        run.submit(move |ctx| async move {
            let Self {
                db,
                chunks,
                extractor,
                ontology,
                version,
                standing,
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
            let cancel = ctx.cancel_token();
            let control = RunControl {
                progress: &progress,
                cancel: Some(&cancel),
            };
            let outcome = extract::run(
                &db,
                &chunks,
                &ontology,
                standing,
                ExtractionRun {
                    extractor: extractor.as_ref(),
                    concurrency: app.config.analysis.extraction_concurrency,
                    control,
                },
            )
            .await;
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

/// What a revalidation would drop, per class and relation id the
/// ontology no longer defines; nothing changes.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/graph/revalidate",
    tag = "graph",
    params(WorkspaceId),
    responses((status = 200, description = "What a revalidation would drop", body = Revalidation)),
)]
pub(crate) async fn revalidation_preview(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<Revalidation>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "graph_revalidation")
        .await?;
    Ok(Json(app.read(&id, Revalidation::preview).await?))
}

/// Drop what the preview counted. The body carries the preview's totals;
/// with none, or with totals the graph no longer matches, nothing is
/// dropped and the answer is 409 with the current totals.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/graph/revalidate",
    tag = "graph",
    request_body(content = Option<DropApproval>, description = "The preview's totals"),
    params(WorkspaceId),
    responses((status = 200, description = "What was dropped", body = Revalidation)),
)]
pub(crate) async fn revalidate(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    approval: Option<Json<DropApproval>>,
) -> ApiResult<Json<Revalidation>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let approval = approval.map(|Json(approval)| approval).unwrap_or_default();
    Ok(Json(access.revalidate_graph(&app, approval).await?))
}

/// The totals the caller saw in the preview and agreed to drop: the API's
/// body and the graph page's form. Totals left out are zero, which
/// approves only a revalidation that drops nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, ToSchema)]
pub(crate) struct DropApproval {
    #[serde(default)]
    pub dropped_nodes: u64,
    #[serde(default)]
    pub dropped_edges: u64,
}

impl DropApproval {
    /// `None` when `preview` drops what was approved, else what to tell
    /// the caller.
    fn refusal(self, preview: &Revalidation) -> Option<String> {
        let Self {
            dropped_nodes,
            dropped_edges,
        } = self;
        if (preview.dropped_nodes, preview.dropped_edges) == (dropped_nodes, dropped_edges) {
            return None;
        }
        Some(format!(
            "nothing dropped: revalidating now drops {} nodes and {} edges, not the \
             {dropped_nodes} and {dropped_edges} confirmed; check the preview and confirm its \
             totals",
            preview.dropped_nodes, preview.dropped_edges
        ))
    }
}

/// Mark the provisional graph reviewed.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/graph/review",
    tag = "graph",
    params(WorkspaceId),
    responses((status = 200, description = "The graph's status", body = GraphStatus)),
)]
pub(crate) async fn review(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<GraphStatus>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    Ok(Json(access.review_graph(&app).await?))
}

/// Merge proposals waiting for a decision.
#[derive(Serialize, ToSchema)]
pub(crate) struct MergeList {
    pub merges: Vec<MergeProposal>,
}

/// Merge proposals waiting for a decision.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/graph/merges",
    tag = "graph",
    params(WorkspaceId),
    responses((status = 200, description = "The pending merges", body = MergeList)),
)]
pub(crate) async fn merges(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<MergeList>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "graph_merges")
        .await?;
    let pending = app.read(&id, resolve::pending).await?;
    Ok(Json(MergeList { merges: pending }))
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct DecideMerge {
    /// `accept` or `reject`; anything else is refused while the body is read.
    pub action: MergeDecision,
}

/// Accept or reject one merge proposal.
#[utoipa::path(
    put,
    path = "/workspaces/{id}/graph/merges/{mid}",
    tag = "graph",
    request_body = DecideMerge,
    responses((status = 200, description = "The proposal as decided", body = MergeProposal)),
)]
pub(crate) async fn decide_merge(
    State(app): State<App>,
    identity: Identity,
    Path((id, mid)): Path<(WorkspaceId, String)>,
    Json(body): Json<DecideMerge>,
) -> ApiResult<Json<MergeProposal>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let proposal = access.decide_merge(&app, &mid, body.action).await?;
    Ok(Json(proposal))
}

/// The graph writes the API and the web console share.
impl Access {
    /// Drop what the current ontology no longer allows, when that is
    /// what `approval` covers; 409 with the current totals when it is not.
    pub(crate) async fn revalidate_graph(
        &self,
        app: &App,
        approval: DropApproval,
    ) -> ApiResult<Revalidation> {
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let outcome = with_db(db, move |db| {
            if let Some(refusal) = approval.refusal(&Revalidation::preview(db)?) {
                return Ok(Err(refusal));
            }
            graph_store::revalidate(db).map(Ok)
        })
        .await
        .map_err(|e| ApiError::bad_request(e.message))?
        .map_err(ApiError::conflict)?;
        self.audit(
            app,
            AuditAction::GraphRevalidate,
            None,
            Outcome::Allowed,
            Some(serde_json::to_value(&outcome)?),
        )
        .await?;
        Ok(outcome)
    }

    /// Mark the provisional graph reviewed.
    pub(crate) async fn review_graph(&self, app: &App) -> ApiResult<GraphStatus> {
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let status = with_db(db, |db| {
            graph_store::mark_reviewed(db)?;
            graph_store::status(db)
        })
        .await?;
        self.audit(app, AuditAction::GraphReview, None, Outcome::Allowed, None)
            .await?;
        Ok(status)
    }

    /// Accept or reject one merge proposal.
    pub(crate) async fn decide_merge(
        &self,
        app: &App,
        merge: &str,
        decision: MergeDecision,
    ) -> ApiResult<MergeProposal> {
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let author = self.identity.username.clone();
        let merge_id = merge.to_owned();
        let proposal = with_db(db, move |db| {
            resolve::decide(db, &merge_id, decision, Some(&author))
        })
        .await?;
        self.audit(
            app,
            AuditAction::GraphMerge,
            Some(ResourceKind::GraphMerge.id(merge)),
            Outcome::Allowed,
            Some(serde_json::json!({ "accept": decision == MergeDecision::Accept, "keep": proposal.keep.label, "drop": proposal.drop.label })),
        )
        .await?;
        Ok(proposal)
    }
}

/// A node a person asserts over the API.
#[derive(Deserialize, ToSchema)]
pub(crate) struct CreateNode {
    pub label: String,
    pub class: ClassId,
    #[serde(default)]
    pub properties: Properties,
    pub note: Option<String>,
}

/// An edge a person asserts over the API, between nodes by id.
#[derive(Deserialize, ToSchema)]
pub(crate) struct CreateEdge {
    pub source: NodeId,
    pub target: NodeId,
    pub relation: RelationId,
    #[serde(default)]
    pub properties: Properties,
    pub note: Option<String>,
}

/// A node correction over the API: [`NodeEdit`] plus the note.
#[derive(Deserialize, ToSchema)]
pub(crate) struct UpdateNode {
    #[serde(flatten)]
    pub edit: NodeEdit,
    pub note: Option<String>,
}

/// Assert a node; one with the same label and class comes back unchanged.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/graph/nodes",
    tag = "graph",
    request_body = CreateNode,
    params(WorkspaceId),
    responses(
        (status = 201, description = "Created", body = Asserted<Node>),
        (status = 200, description = "It already existed", body = Asserted<Node>),
    ),
)]
pub(crate) async fn create_node(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<CreateNode>,
) -> ApiResult<(StatusCode, Json<Asserted<Node>>)> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let added = access.create_node(&app, body).await?;
    let status = if added.created {
        StatusCode::CREATED
    } else {
        StatusCode::OK
    };
    Ok((status, Json(added)))
}

/// Relabel, reclass, or change a node's properties.
#[utoipa::path(
    patch,
    path = "/workspaces/{id}/graph/nodes/{nid}",
    tag = "graph",
    request_body = UpdateNode,
    responses((status = 200, description = "The node as it now is", body = Node)),
)]
pub(crate) async fn update_node(
    State(app): State<App>,
    identity: Identity,
    Path((id, nid)): Path<(WorkspaceId, NodeId)>,
    Json(body): Json<UpdateNode>,
) -> ApiResult<Json<Node>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    Ok(Json(access.update_node(&app, &nid, body).await?))
}

/// Delete a node and its edges.
#[utoipa::path(
    delete,
    path = "/workspaces/{id}/graph/nodes/{nid}",
    tag = "graph",
    responses((status = 200, description = "The node deleted", body = Node)),
)]
pub(crate) async fn delete_node(
    State(app): State<App>,
    identity: Identity,
    Path((id, nid)): Path<(WorkspaceId, NodeId)>,
) -> ApiResult<Json<Node>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    Ok(Json(access.delete_node(&app, &nid).await?))
}

/// Assert an edge; one that already exists comes back unchanged.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/graph/edges",
    tag = "graph",
    request_body = CreateEdge,
    params(WorkspaceId),
    responses(
        (status = 201, description = "Created", body = Asserted<Edge>),
        (status = 200, description = "It already existed", body = Asserted<Edge>),
    ),
)]
pub(crate) async fn create_edge(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<CreateEdge>,
) -> ApiResult<(StatusCode, Json<Asserted<Edge>>)> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let added = access.create_edge(&app, body).await?;
    let status = if added.created {
        StatusCode::CREATED
    } else {
        StatusCode::OK
    };
    Ok((status, Json(added)))
}

/// Delete an edge.
#[utoipa::path(
    delete,
    path = "/workspaces/{id}/graph/edges/{eid}",
    tag = "graph",
    responses((status = 200, description = "The edge deleted", body = Edge)),
)]
pub(crate) async fn delete_edge(
    State(app): State<App>,
    identity: Identity,
    Path((id, eid)): Path<(WorkspaceId, EdgeId)>,
) -> ApiResult<Json<Edge>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    Ok(Json(access.delete_edge(&app, &eid).await?))
}

/// What a graph edit's audit row records.
struct EditDetail {
    op: &'static str,
    label: Option<String>,
    class: Option<String>,
    relation: Option<String>,
    note: Option<String>,
}

impl EditDetail {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "op": self.op,
            "label": self.label,
            "class": self.class,
            "relation": self.relation,
            "note": self.note,
        })
    }
}

/// A person's graph edits, shared by the API and the graph page: each
/// one write on the workspace, audited as `graph_edit` with the node or
/// edge as its resource. The author is the signed-in user.
impl Access {
    fn assertion(&self, note: Option<String>) -> Assertion {
        Assertion {
            author: Some(self.identity.username.clone()),
            note,
        }
    }

    /// Record a graph edit that went through.
    async fn audit_edit(
        &self,
        app: &App,
        kind: ResourceKind,
        id: &str,
        detail: EditDetail,
    ) -> ApiResult<()> {
        self.audit(
            app,
            AuditAction::GraphEdit,
            Some(kind.id(id)),
            Outcome::Allowed,
            Some(detail.json()),
        )
        .await?;
        Ok(())
    }

    pub(crate) async fn create_node(
        &self,
        app: &App,
        body: CreateNode,
    ) -> ApiResult<Asserted<Node>> {
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let assertion = self.assertion(body.note.clone());
        let node = NewNode {
            label: body.label,
            class_id: body.class,
            properties: body.properties,
            standing: Standing::Reviewed,
        };
        let added = with_db(db, move |db| {
            graph_store::create_node(db, &node, &assertion)
        })
        .await?;
        self.audit_edit(
            app,
            ResourceKind::GraphNode,
            added.subject.id.as_str(),
            EditDetail {
                op: if added.created { "create" } else { "assert" },
                label: Some(added.subject.label.clone()),
                class: Some(added.subject.class_id.to_string()),
                relation: None,
                note: body.note,
            },
        )
        .await?;
        Ok(added)
    }

    pub(crate) async fn update_node(
        &self,
        app: &App,
        id: &NodeId,
        body: UpdateNode,
    ) -> ApiResult<Node> {
        if body.edit.is_empty() {
            return Err(ApiError::bad_request(
                "nothing to change: give label, class, or properties",
            ));
        }
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let assertion = self.assertion(body.note.clone());
        let (nid, edit) = (id.clone(), body.edit);
        let node = with_db(db, move |db| {
            graph_store::update_node(db, &nid, &edit, &assertion)
        })
        .await?;
        self.audit_edit(
            app,
            ResourceKind::GraphNode,
            id.as_str(),
            EditDetail {
                op: "update",
                label: Some(node.label.clone()),
                class: Some(node.class_id.to_string()),
                relation: None,
                note: body.note,
            },
        )
        .await?;
        Ok(node)
    }

    pub(crate) async fn delete_node(&self, app: &App, id: &NodeId) -> ApiResult<Node> {
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let nid = id.clone();
        let node = with_db(db, move |db| graph_store::delete_node(db, &nid)).await?;
        self.audit_edit(
            app,
            ResourceKind::GraphNode,
            id.as_str(),
            EditDetail {
                op: "delete",
                label: Some(node.label.clone()),
                class: Some(node.class_id.to_string()),
                relation: None,
                note: None,
            },
        )
        .await?;
        Ok(node)
    }

    pub(crate) async fn create_edge(
        &self,
        app: &App,
        body: CreateEdge,
    ) -> ApiResult<Asserted<Edge>> {
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let assertion = self.assertion(body.note.clone());
        let edge = NewEdge {
            source: body.source,
            target: body.target,
            relation: body.relation,
            properties: body.properties,
        };
        let added = with_db(db, move |db| {
            graph_store::create_edge(db, &edge, &assertion)
        })
        .await?;
        self.audit_edit(
            app,
            ResourceKind::GraphEdge,
            added.subject.id.as_str(),
            EditDetail {
                op: if added.created { "create" } else { "assert" },
                label: None,
                class: None,
                relation: Some(added.subject.relation_id.to_string()),
                note: body.note,
            },
        )
        .await?;
        Ok(added)
    }

    pub(crate) async fn delete_edge(&self, app: &App, id: &EdgeId) -> ApiResult<Edge> {
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let eid = id.clone();
        let edge = with_db(db, move |db| graph_store::delete_edge(db, &eid)).await?;
        self.audit_edit(
            app,
            ResourceKind::GraphEdge,
            id.as_str(),
            EditDetail {
                op: "delete",
                label: None,
                class: None,
                relation: Some(edge.relation_id.to_string()),
                note: None,
            },
        )
        .await?;
        Ok(edge)
    }
}
