//! `POST .../import`: rows from a SQLite file, a data file over HTTP(S), or
//! an S3 object as a workspace table. Needs the write permission; audited
//! as `import` with the redacted source and the headers' names (a password
//! or header value never lands anywhere).

use axum::Json;
use axum::extract::{Path, State};
use quack_core::error::Error as CoreError;
use quack_core::ids::WorkspaceId;
use quack_core::llm::Embeddings;
use quack_core::progress::RunControl;
use quack_core::storage::control::{AuditAction, Outcome};
use quack_core::storage::profile::ColumnTypes;
use quack_core::text::NonBlankText;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult, ErrorCode};
use crate::server::run::{self, BackgroundRun, RunKind};
use crate::server::state::{App, ServeMode};
use std::sync::Arc;

use axum::http::StatusCode;
use quack_core::ids::ImportId;
use quack_core::import::{
    self, ImportPolicy, ImportRequest, ImportSecrets, ImportSummary, JsonPointer, KeepSecret,
    LoadStatus, RefreshWith, SavedImport, SourceHeader,
};
use quack_core::jobs::JobId;
use quack_core::llm::oauth::KeySource;
use quack_core::progress::ChunkDone;
use quack_core::storage::control::ResourceKind;
use quack_core::storage::writer::Writer;
use quack_core::vault::Vault;

#[derive(Deserialize, ToSchema)]
pub(crate) struct ImportBody {
    /// `sqlite://...`, an `http(s)://` data file, or `s3://bucket/key`.
    pub url: String,
    /// The workspace table the rows load into.
    pub table: String,
    /// A read to run on a database source.
    pub query: Option<String>,
    /// A database source's table, read whole.
    pub source_table: Option<String>,
    /// Rows at most.
    pub limit: Option<u64>,
    /// `COLUMN=TYPE` pairs, comma-separated, as `quack import --types`.
    pub types: Option<String>,
    /// Headers an `http(s)://` download sends, one `Name: value` per line.
    pub headers: Option<String>,
    /// The server's environment variable whose token goes as
    /// `Authorization: Bearer`; needs `[import].allow_server_credentials`.
    pub bearer_env: Option<String>,
    /// Where a JSON download's rows sit (RFC 6901), as `/data/items`.
    pub json_pointer: Option<String>,
    /// Save the import under this name, so it can be refreshed.
    pub save: Option<String>,
    /// Keep the URL's password and the header values with the saved
    /// import, sealed under the vault key.
    pub store_credential: Option<bool>,
}

impl ImportBody {
    /// The name to save the import under, and whether to keep its secret.
    pub(crate) fn saving(&self) -> Option<(String, KeepSecret)> {
        let keep = if self.store_credential.unwrap_or(false) {
            KeepSecret::Sealed
        } else {
            KeepSecret::No
        };
        self.save
            .as_deref()
            .and_then(str::non_blank)
            .map(|name| (name.to_owned(), keep))
    }
}

/// A blank query, source table, or types (an empty form field) is none.
impl TryFrom<ImportBody> for ImportRequest {
    type Error = ApiError;

    fn try_from(body: ImportBody) -> ApiResult<Self> {
        let given =
            |field: Option<String>| field.as_deref().and_then(str::non_blank).map(str::to_owned);
        let bad = |e: CoreError| ApiError::bad_request(e.to_string());
        let types: ColumnTypes = given(body.types)
            .map(|t| t.parse())
            .transpose()
            .map_err(bad)?
            .unwrap_or_default();
        let mut headers = given(body.headers)
            .unwrap_or_default()
            .lines()
            .filter_map(str::non_blank)
            .map(str::parse::<SourceHeader>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(bad)?;
        headers.extend(given(body.bearer_env).map(SourceHeader::BearerEnv));
        let json_pointer = given(body.json_pointer)
            .map(|p| p.parse::<JsonPointer>())
            .transpose()
            .map_err(bad)?;
        Ok(Self {
            query: given(body.query),
            source_table: given(body.source_table),
            limit: body.limit,
            types,
            headers,
            json_pointer,
            ..Self::new(body.url, body.table)
        })
    }
}

/// Snapshot a query on a database, or a data file over HTTP(S), as a
/// workspace table.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/import",
    tag = "import",
    request_body = ImportBody,
    params(WorkspaceId),
    responses((status = 200, description = "What was imported", body = Imported)),
)]
pub(crate) async fn import(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<ImportBody>,
) -> ApiResult<Json<Imported>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let saving = body.saving();
    Ok(Json(
        run_import(&app, &access, &ImportRequest::try_from(body)?, saving).await?,
    ))
}

/// What an import did, and the graph follow-up it queued when
/// `[graph].follow_ingest` asks for one.
#[derive(Serialize, ToSchema)]
pub(crate) struct Imported {
    #[serde(flatten)]
    pub summary: ImportSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_job: Option<JobId>,
    /// The saved import, when the request asked to save it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub saved: Option<SavedImport>,
}

/// The import the API and the web form share: run it, audit it either way.
pub(crate) async fn run_import(
    app: &App,
    access: &Access,
    request: &ImportRequest,
    saving: Option<(String, KeepSecret)>,
) -> ApiResult<Imported> {
    let source = request.url.redacted();
    let kind = request
        .url
        .kind()
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    // `--local` is the owner at a keyboard; anyone else is held to
    // `[import]`: no files from the server's disk, no private hosts, and no
    // credentials of the server's own.
    let policy = if app.mode == ServeMode::Local {
        ImportPolicy::owner()
    } else {
        ImportPolicy::server(&app.config)
    };
    if let Err(refused) = request.check(kind, policy) {
        if matches!(refused, CoreError::ServerCredentials) {
            let detail = serde_json::json!({ "source": source, "table": request.table });
            access
                .audit(
                    app,
                    AuditAction::Import,
                    None,
                    Outcome::Denied,
                    Some(detail),
                )
                .await?;
            return Err(ApiError::from(refused));
        }
        return Err(ApiError::bad_request(refused.to_string()));
    }
    if let Some((name, keep)) = &saving {
        request
            .check_saveable(*keep)
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        let name = name.clone();
        app.read(&access.membership.workspace.id, move |db| {
            SavedImport::check_name(db, &name)
        })
        .await?;
    }
    let db = app.workspace_db(&access.membership.workspace.id).await?;
    let embeddings = access
        .model(
            app,
            AuditAction::Import,
            Embeddings::from_config(&app.config).await,
        )
        .await?;
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
        "headers": request.headers.iter().map(SourceHeader::name).collect::<Vec<_>>(),
        "json_pointer": request.json_pointer.as_ref().map(ToString::to_string),
        "rows": outcome.as_ref().ok().map(|s| s.rows),
    });
    let audit_outcome = Outcome::of(&outcome);
    access
        .audit(app, AuditAction::Import, None, audit_outcome, Some(detail))
        .await?;
    let summary =
        outcome.map_err(|e| ApiError::from(e).unprocessable_as(ErrorCode::ImportFailed))?;
    let saved = match saving {
        Some((name, keep)) => Some(
            access
                .save_import(app, &db, &name, request, &summary, keep)
                .await?,
        ),
        None => None,
    };
    let graph_job = run::follow_ingest(
        app,
        access,
        db,
        embeddings,
        vec![summary.document_id.clone()],
    )
    .await?;
    Ok(Imported {
        summary,
        graph_job,
        saved,
    })
}

impl Access {
    /// Where this workspace's saved imports keep their sealed secrets.
    fn secrets<'a>(&'a self, app: &'a App, vault: &'a Vault) -> ImportSecrets<'a> {
        ImportSecrets {
            control: &app.control,
            vault,
            workspace: &self.membership.workspace.id,
        }
    }

    /// Refresh `saved` as an audited background job: replaced when the
    /// source changed, with the graph follow-up after a load.
    pub(crate) async fn refresh_import(&self, app: &App, saved: SavedImport) -> ApiResult<JobId> {
        let embeddings = self
            .model(
                app,
                AuditAction::Import,
                Embeddings::from_config(&app.config).await,
            )
            .await?;
        let run = BackgroundRun::start(
            app,
            self,
            RunKind::IMPORT_REFRESH,
            serde_json::json!({ "import": saved.id, "name": saved.name, "source": saved.source }),
        )
        .await?;
        let (app_for_job, access_for_job) = (Arc::clone(app), self.clone());
        let job = run.submit(move |ctx| async move {
            let cancel = ctx.cancel_token();
            let progress = |done: ChunkDone| ctx.progress(done.done, done.total);
            let control = RunControl {
                progress: &progress,
                cancel: Some(&cancel),
            };
            let app = app_for_job;
            let workspace = access_for_job.membership.workspace.id.clone();
            let db = app
                .workspace_db(&workspace)
                .await
                .map_err(|e| e.message)?;
            let policy = if app.mode == ServeMode::Local {
                ImportPolicy::owner()
            } else {
                ImportPolicy::server(&app.config)
            };
            let vault = Vault::new(app.config.data_dir(), KeySource::Keychain);
            let summary = access_for_job
                .secrets(&app, &vault)
                .refresh(
                    &saved,
                    RefreshWith {
                        config: &app.config,
                        db: &db,
                        policy,
                        embedder: embeddings.as_ref(),
                        control,
                    },
                )
                .await
                .map_err(|e| e.to_string())?;
            if summary.status == LoadStatus::Loaded
                && let Err(e) = run::follow_ingest(
                    &app,
                    &access_for_job,
                    db,
                    embeddings,
                    vec![summary.document_id.clone()],
                )
                .await
            {
                tracing::warn!(error = %e.message, "the graph follow-up of a refresh could not be queued");
            }
            Ok(summary)
        });
        Ok(job)
    }

    /// Remove `saved` and its sealed secret, audited; its table stays.
    pub(crate) async fn remove_import(&self, app: &App, saved: &SavedImport) -> ApiResult<()> {
        let db = app.workspace_db(&self.membership.workspace.id).await?;
        let vault = Vault::new(app.config.data_dir(), KeySource::Keychain);
        self.secrets(app, &vault).remove(&db, saved).await?;
        self.audit(
            app,
            AuditAction::Delete,
            Some(ResourceKind::SavedImport.id(&saved.id)),
            Outcome::Allowed,
            Some(serde_json::json!({ "name": saved.name })),
        )
        .await?;
        Ok(())
    }

    /// Save `request`, which ran as `summary`, under `name`, by the caller.
    async fn save_import(
        &self,
        app: &App,
        db: &Writer,
        name: &str,
        request: &ImportRequest,
        summary: &ImportSummary,
        keep: KeepSecret,
    ) -> ApiResult<SavedImport> {
        let vault = Vault::new(app.config.data_dir(), KeySource::Keychain);
        Ok(self
            .secrets(app, &vault)
            .save(
                db,
                name,
                request,
                summary,
                keep,
                Some(self.identity.user_id.as_str()),
            )
            .await?)
    }

    /// The saved import named `name` (or with that id) in this workspace.
    pub(crate) async fn saved_import(&self, app: &App, name: &str) -> ApiResult<SavedImport> {
        let name = name.to_owned();
        app.read(&self.membership.workspace.id, move |db| {
            SavedImport::named(db, &name)
        })
        .await
    }
}

/// The imports saved in a workspace.
#[derive(Serialize, ToSchema)]
pub(crate) struct SavedImports {
    pub imports: Vec<SavedImport>,
}

/// Every import saved in the workspace, with how each last ran.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/imports",
    tag = "import",
    params(WorkspaceId),
    responses((status = 200, description = "The saved imports", body = SavedImports)),
)]
pub(crate) async fn list_saved(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
) -> ApiResult<Json<SavedImports>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "saved imports")
        .await?;
    let imports = app.read(&id, SavedImport::list).await?;
    Ok(Json(SavedImports { imports }))
}

/// Import and save the import under `save`, so it can be refreshed. A URL
/// password or header value needs `store_credential`.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/imports",
    tag = "import",
    request_body = ImportBody,
    params(WorkspaceId),
    responses((status = 201, description = "Imported and saved", body = Imported)),
)]
pub(crate) async fn save(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Json(body): Json<ImportBody>,
) -> ApiResult<(StatusCode, Json<Imported>)> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let Some(saving) = body.saving() else {
        return Err(ApiError::bad_request(
            "a saved import needs a name in `save`",
        ));
    };
    let imported = run_import(&app, &access, &ImportRequest::try_from(body)?, Some(saving)).await?;
    Ok((StatusCode::CREATED, Json(imported)))
}

/// A refresh running in the background.
#[derive(Serialize, ToSchema)]
pub(crate) struct RefreshStarted {
    pub job: JobId,
}

/// Run a saved import again as a background job; its table is replaced
/// only when the source changed. Audited as an import run.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/imports/{import}/refresh",
    tag = "import",
    responses((status = 202, description = "The refresh is running", body = RefreshStarted)),
)]
pub(crate) async fn refresh(
    State(app): State<App>,
    identity: Identity,
    Path((id, import)): Path<(WorkspaceId, ImportId)>,
) -> ApiResult<(StatusCode, Json<RefreshStarted>)> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let saved = access.saved_import(&app, import.as_str()).await?;
    let job = access.refresh_import(&app, saved).await?;
    Ok((StatusCode::ACCEPTED, Json(RefreshStarted { job })))
}

/// Remove a saved import and any secret sealed for it; its table stays.
#[utoipa::path(
    delete,
    path = "/workspaces/{id}/imports/{import}",
    tag = "import",
    responses((status = 204, description = "Removed")),
)]
pub(crate) async fn remove(
    State(app): State<App>,
    identity: Identity,
    Path((id, import)): Path<(WorkspaceId, ImportId)>,
) -> ApiResult<StatusCode> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let saved = access.saved_import(&app, import.as_str()).await?;
    access.remove_import(&app, &saved).await?;
    Ok(StatusCode::NO_CONTENT)
}
