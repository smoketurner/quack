//! The Documents view's commands: one page of the listing, files and
//! folders added by path (an OS drop, Add Files…, Add Folder…), each as a
//! job in the workspace's ingest lane, and a document deleted.

use std::path::PathBuf;
use std::sync::Arc;

use quack_core::ids::{DocumentId, WorkspaceId};
use quack_core::ingestion::tree::{Folder, Prune};
use quack_core::ingestion::{self, IngestOutcome, NewFile};
use quack_core::jobs::{JobInfo, JobKind, JobSpec, Lane, LaneKey};
use quack_core::llm::Embeddings;
use quack_core::llm::egress::Egress;
use quack_core::progress::{ChunkDone, RunControl};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use quack_core::storage::workspace::{
    DocumentFilter, DocumentListing, DocumentPage, DocumentSource, Shown,
};
use tauri::State;

use crate::app::{AppState, OpenWorkspace};
use crate::error::{CommandError, CommandResult};

/// One page of `workspace`'s documents: the newest first, or those after
/// `after`; replaced documents too when `all`.
#[tauri::command]
pub(crate) async fn list_documents(
    state: State<'_, AppState>,
    workspace: WorkspaceId,
    after: Option<DocumentId>,
    all: bool,
    filter: Option<DocumentFilter>,
    limit: Option<u32>,
) -> CommandResult<DocumentPage> {
    let ws = state.workspace(&workspace).await?;
    let listing = DocumentListing {
        shown: if all { Shown::All } else { Shown::Live },
        filter: filter.unwrap_or_default(),
        after,
        limit: limit
            .unwrap_or(DocumentListing::PAGE)
            .clamp(1, DocumentListing::MAX_PAGE),
    };
    let page = ws.reader.with_db(move |db| db.documents(&listing)).await?;
    state
        .auditor
        .record(
            &ws,
            AuditAction::List,
            None,
            Outcome::Allowed,
            Some(serde_json::json!({ "what": "documents", "count": page.documents.len(), "total": page.total })),
        )
        .await?;
    Ok(page)
}

/// Ingest each of `paths`: a file as a table or chunks, a folder as
/// `quack ingest DIR` does it (root recorded, unchanged files skipped).
/// One job per path, in the workspace's ingest lane; the jobs event
/// reports each.
#[tauri::command]
pub(crate) async fn add_paths(
    state: State<'_, AppState>,
    workspace: WorkspaceId,
    paths: Vec<String>,
) -> CommandResult<Vec<JobInfo>> {
    let ws = state.workspace(&workspace).await?;
    let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    if let Some(missing) = paths.iter().find(|p| !p.exists()) {
        return Err(CommandError::not_found(format!(
            "{} does not exist",
            missing.display()
        )));
    }
    state
        .auditor
        .record(
            &ws,
            AuditAction::Ingest,
            None,
            Outcome::Allowed,
            Some(serde_json::json!({ "paths": paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>() })),
        )
        .await?;
    Ok(paths
        .into_iter()
        .map(|path| submit(&state, &ws, path))
        .collect())
}

/// The job that ingests `path`.
fn submit(state: &AppState, ws: &Arc<OpenWorkspace>, path: PathBuf) -> JobInfo {
    let label = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file")
        .to_owned();
    let workspace_id = ws.row.id.clone();
    let spec = JobSpec::new(JobKind::Ingest, label.clone())
        .workspace(workspace_id.clone())
        .lane(Lane::new(
            &LaneKey::Workspace(JobKind::Ingest, workspace_id),
            state.config.server.workers_per_workspace,
        ));
    let (config, db, egress) = (state.config.clone(), Arc::clone(&ws.db), ws.egress());
    let workspace = ws.row.id.to_string();
    state.jobs.submit(spec, move |ctx| {
        Egress::scope(Some(egress), async move {
            let embedder = Embeddings::from_config(&config)
                .await
                .map_err(|e| e.to_string())?;
            let progress = |done: ChunkDone| ctx.progress(done.done, done.total);
            let cancel = ctx.cancel_token();
            let control = RunControl {
                progress: &progress,
                cancel: Some(&cancel),
            };
            if path.is_dir() {
                let report = Folder {
                    config: &config,
                    db: &db,
                    workspace_id: &workspace,
                    root: &path,
                    embedder: embedder.as_ref(),
                    control,
                    prune: Prune::Keep,
                }
                .run()
                .await
                .map_err(|e| e.to_string())?;
                let failed = report.failed();
                return Ok(format!(
                    "{} files, {failed} failed, {} unsupported",
                    report.results.len(),
                    report.unsupported.len()
                ));
            }
            let file = NewFile::at_path(&label, &path)
                .source(DocumentSource::Path)
                .control(control);
            let outcome =
                ingestion::ingest_file(&config, &db, &workspace, &file, embedder.as_ref())
                    .await
                    .map_err(|e| e.to_string())?;
            Ok(describe(&outcome))
        })
    })
}

/// The job's one-line outcome for a file.
fn describe(outcome: &IngestOutcome) -> String {
    match outcome {
        IngestOutcome::Ingested(result) => match result.tables.as_slice() {
            [] => format!("{} chunks", result.chunks_stored),
            tables => format!("tables {}", tables.join(", ")),
        },
        IngestOutcome::Duplicate(existing) => {
            format!(
                "identical to {} ({}), already in the workspace",
                existing.filename, existing.id
            )
        }
    }
}

/// Delete `document`, its chunks, its tables, and its file. `false` when
/// no such document exists.
#[tauri::command]
pub(crate) async fn delete_document(
    state: State<'_, AppState>,
    workspace: WorkspaceId,
    document: DocumentId,
) -> CommandResult<bool> {
    let ws = state.workspace(&workspace).await?;
    let id = document.clone();
    let deleted = ws.db.run(move |db| db.delete_document(&id)).await?;
    state
        .auditor
        .record(
            &ws,
            AuditAction::Delete,
            Some(ResourceKind::Document.id(&document)),
            if deleted {
                Outcome::Allowed
            } else {
                Outcome::Denied
            },
            Some(serde_json::json!({ "deleted": deleted })),
        )
        .await?;
    Ok(deleted)
}
