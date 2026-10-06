//! `POST .../import`: rows from Postgres, SQLite, or a data file over
//! HTTP(S) as a workspace table. Needs the write permission; audited as
//! `import` with the redacted source (the password never lands anywhere).

use axum::Json;
use axum::extract::{Path, State};
use quack_core::error::Error as CoreError;
use quack_core::ids::WorkspaceId;
use quack_core::llm::Embeddings;
use quack_core::progress::RunControl;
use quack_core::storage::control::{AuditAction, Outcome};
use quack_core::storage::profile::ColumnTypes;
use quack_core::text::NonBlankText;
use serde::Deserialize;

use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::run;
use crate::server::state::{App, ServeMode};
use quack_core::import::{self, ImportPolicy, ImportRequest, ImportSummary};
use quack_core::jobs::JobId;

#[derive(Deserialize)]
pub(crate) struct ImportBody {
    pub url: String,
    pub table: String,
    pub query: Option<String>,
    pub source_table: Option<String>,
    pub limit: Option<u64>,
    /// `COLUMN=TYPE` pairs, comma-separated, as `quack import --types`.
    pub types: Option<String>,
}

/// A blank query, source table, or types (an empty form field) is none.
impl TryFrom<ImportBody> for ImportRequest {
    type Error = ApiError;

    fn try_from(body: ImportBody) -> ApiResult<Self> {
        let given =
            |field: Option<String>| field.as_deref().and_then(str::non_blank).map(str::to_owned);
        let types: ColumnTypes = given(body.types)
            .map(|t| t.parse())
            .transpose()
            .map_err(|e: CoreError| ApiError::bad_request(e.to_string()))?
            .unwrap_or_default();
        Ok(Self {
            url: body.url.into(),
            table: body.table,
            query: given(body.query),
            source_table: given(body.source_table),
            limit: body.limit,
            types,
        })
    }
}

pub(crate) async fn import(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<ImportBody>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let imported = run_import(&app, &access, &ImportRequest::try_from(body)?).await?;
    let mut body = serde_json::to_value(&imported.summary)?;
    if let (Some(fields), Some(job)) = (body.as_object_mut(), &imported.graph_job) {
        fields.insert(String::from("graph_job"), serde_json::to_value(job)?);
    }
    Ok(Json(body))
}

/// What an import did, and the graph follow-up it queued when
/// `[graph].follow_ingest` asks for one.
pub(crate) struct Imported {
    pub summary: ImportSummary,
    pub graph_job: Option<JobId>,
}

/// The import the API and the web form share: run it, audit it either way.
pub(crate) async fn run_import(
    app: &App,
    access: &Access,
    request: &ImportRequest,
) -> ApiResult<Imported> {
    let source = request.url.redacted();
    request
        .url
        .kind()
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    let db = app.workspace_db(&access.membership.workspace.id).await?;
    let embeddings = access
        .model(
            app,
            AuditAction::Import,
            Embeddings::from_config(&app.config).await,
        )
        .await?;
    // `--local` is the owner at a keyboard; anyone else is held to
    // `[import]`: no files from the server's disk, no private hosts.
    let policy = if app.mode == ServeMode::Local {
        ImportPolicy::owner()
    } else {
        ImportPolicy::server(&app.config)
    };
    let outcome = import::Importing {
        config: &app.config,
        db: &db,
        workspace_id: access.membership.workspace.id.as_str(),
        request,
        policy,
        embedder: embeddings.as_ref(),
        control: RunControl::unobserved(),
    }
    .run()
    .await;
    let detail = serde_json::json!({
        "source": source,
        "table": request.table,
        "query": request.query,
        "source_table": request.source_table,
        "rows": outcome.as_ref().ok().map(|s| s.rows),
    });
    let audit_outcome = Outcome::of(&outcome);
    access
        .audit(app, AuditAction::Import, None, audit_outcome, Some(detail))
        .await?;
    let summary = outcome.map_err(|e| ApiError::unprocessable(e.to_string()))?;
    let graph_job = run::follow_ingest(
        app,
        access,
        db,
        embeddings,
        vec![summary.document_id.clone()],
    )
    .await?;
    Ok(Imported { summary, graph_job })
}
