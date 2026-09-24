//! `POST .../import`: rows from Postgres, SQLite, or a data file over
//! HTTP(S) as a workspace table. Needs the write permission; audited as
//! `import` with the redacted source (the password never lands anywhere).

use axum::Json;
use axum::extract::{Path, State};
use quack_core::llm;
use quack_core::storage::control::{AuditAction, Outcome};
use serde::Deserialize;

use crate::server::auth::{Access, Identity, Need, access};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::App;
use quack_core::import::{self, ImportPolicy, ImportRequest, ImportSummary};

#[derive(Deserialize)]
pub(crate) struct ImportBody {
    pub url: String,
    pub table: String,
    pub query: Option<String>,
    pub source_table: Option<String>,
    pub limit: Option<u64>,
}

/// A blank query or source table (an empty form field) is none.
impl From<ImportBody> for ImportRequest {
    fn from(body: ImportBody) -> Self {
        let given = |field: Option<String>| field.filter(|value| !value.trim().is_empty());
        Self {
            url: body.url,
            table: body.table,
            query: given(body.query),
            source_table: given(body.source_table),
            limit: body.limit,
        }
    }
}

pub(crate) async fn import(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Json(body): Json<ImportBody>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let summary = run_import(&app, &access, &ImportRequest::from(body)).await?;
    Ok(Json(serde_json::to_value(summary)?))
}

/// The import the API and the web form share: run it, audit it either way.
pub(crate) async fn run_import(
    app: &App,
    access: &Access,
    request: &ImportRequest,
) -> ApiResult<ImportSummary> {
    let source = import::redact(&request.url);
    import::source_kind(&request.url).map_err(|e| ApiError::bad_request(e.to_string()))?;
    let db = app.workspace_db(&access.workspace.id).await?;
    let embeddings = llm::optional_embedding_model(&app.config).await?;
    // `--local` is the owner at a keyboard; anyone else is held to
    // `[import]`: no files from the server's disk, no private hosts.
    let policy = if app.local {
        ImportPolicy::owner()
    } else {
        ImportPolicy::server(&app.config)
    };
    let outcome = import::import(
        &app.config,
        &db,
        &access.workspace.id,
        request,
        policy,
        embeddings.as_ref(),
        None,
    )
    .await;
    let detail = serde_json::json!({
        "source": source,
        "table": request.table,
        "query": request.query,
        "source_table": request.source_table,
        "rows": outcome.as_ref().ok().map(|s| s.rows),
    });
    let audit_outcome = if outcome.is_ok() {
        Outcome::Allowed
    } else {
        Outcome::Error
    };
    access
        .audit(app, AuditAction::Import, None, audit_outcome, Some(detail))
        .await?;
    outcome.map_err(|e| ApiError::new(axum::http::StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))
}
