//! Documents: list, upload (multipart files or pasted text) into the
//! queue, status, pin, delete.

use std::sync::Arc;

use axum::extract::{Multipart, Path, State};
use axum::http::{StatusCode, header};
use axum::{Json, response::IntoResponse};
use quack_core::error::Record;
use quack_core::ingestion;
use quack_core::storage::control::Outcome;
use serde::Deserialize;

use crate::server::auth::{Access, Identity, Need, access};
use crate::server::error::{ApiError, ApiResult};
use crate::server::queue::{
    MAX_WAITING_UPLOADS, UPLOAD_RETRY_SECONDS, UploadJob, submit_upload, upload_lane,
};
use crate::server::state::{App, with_db};
use quack_core::okf::{self, Bundle};
use quack_core::ontology::candidates;
use quack_core::ontology::store as ontology_store;
use quack_core::storage::workspace::{DocumentInfo, DocumentSource, WorkspaceDb};

pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access.audit_read(&app, "list", "documents").await?;
    let docs = app.read(&id, WorkspaceDb::list_documents).await?;
    Ok(Json(serde_json::json!({ "documents": docs })))
}

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path((id, doc)): Path<(String, String)>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::READ).await?;
    access
        .audit(
            &app,
            "open",
            Some(("document", &doc)),
            Outcome::Allowed,
            None,
        )
        .await?;
    let document = app
        .read(&id, move |db| {
            db.document(&doc)?
                .ok_or_else(|| Record::Document.missing(doc.as_str()))
        })
        .await?;
    Ok(Json(serde_json::to_value(document)?))
}

#[derive(Deserialize)]
pub(crate) struct PastedText {
    pub text: String,
    pub title: Option<String>,
}

/// `multipart/form-data` with one or more `file` parts, or JSON
/// `{text, title}`. Returns 202 with the queued documents.
pub(crate) async fn upload(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    request: axum::extract::Request,
) -> ApiResult<impl IntoResponse> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    if content_type.starts_with("application/x-tar") {
        let bytes = axum::body::to_bytes(request.into_body(), usize::MAX)
            .await
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        let bundle = Bundle::from_tar(&bytes).map_err(|e| ApiError::bad_request(e.to_string()))?;
        return import_bundle(&app, &access, bundle).await;
    }
    let files = if content_type.starts_with("multipart/form-data") {
        let multipart = Multipart::from_request(request, &app)
            .await
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        multipart_files(multipart).await?
    } else {
        let bytes = axum::body::to_bytes(request.into_body(), usize::MAX)
            .await
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        let pasted: PastedText =
            serde_json::from_slice(&bytes).map_err(|e| ApiError::bad_request(e.to_string()))?;
        vec![pasted_file(&pasted.text, pasted.title.as_deref())?]
    };
    let source = if content_type.starts_with("multipart/form-data") {
        DocumentSource::Upload
    } else {
        DocumentSource::Paste
    };
    let queued = enqueue(&app, &access, source, files).await?;
    Ok((
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "documents": queued })),
    )
        .into_response())
}

/// An OKF bundle: every concept file with text is queued as a Markdown
/// document (quack's own stubs are not), the bundle's ontology snapshot
/// is restored when the workspace has none, the front matter and links go
/// to the ontology review queue, and the `index.md` body is returned as
/// `context` for the caller to apply.
async fn import_bundle(
    app: &App,
    access: &Access,
    bundle: Bundle,
) -> ApiResult<axum::response::Response> {
    let files: Vec<(String, Vec<u8>)> = bundle
        .documents()
        .map(|f| (okf::document_name(&f.path), f.content.as_bytes().to_vec()))
        .collect();
    let queued = if files.is_empty() {
        Vec::new()
    } else {
        enqueue(app, access, DocumentSource::Upload, files).await?
    };
    let db = app.workspace_db(&access.workspace.id).await?;
    let for_candidates = bundle.clone();
    let author = access.identity.username.clone();
    let (candidates, restored) = with_db(db, move |db| {
        let mut current = ontology_store::current(db)?;
        let mut restored = None;
        if current.is_none()
            && let Some(snapshot) = for_candidates.ontology()?
        {
            let saved =
                ontology_store::save(db, &snapshot, Some(&author), Some("restored from a bundle"))?;
            restored = Some(saved.version);
            current = Some(saved);
        }
        let candidates = okf::propose(&for_candidates, current.as_ref());
        if !candidates.is_empty() {
            candidates::store_run(db, &candidates)?;
        }
        Ok((candidates.len(), restored))
    })
    .await?;
    let context = bundle
        .index()
        .map(|index| okf::parse_front_matter(&index.content).1.trim().to_owned());
    Ok((
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "documents": queued,
            "candidates": candidates,
            "ontology_version": restored,
            "context": context,
        })),
    )
        .into_response())
}

/// Every part that carries a file name, as `(name, bytes)`.
pub(crate) async fn multipart_files(mut multipart: Multipart) -> ApiResult<Vec<(String, Vec<u8>)>> {
    let mut files = Vec::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::bad_request(e.to_string()))?
    {
        let Some(name) = field.file_name().map(str::to_owned) else {
            continue;
        };
        let data = field
            .bytes()
            .await
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        if name.is_empty() && data.is_empty() {
            continue;
        }
        files.push((name, data.to_vec()));
    }
    Ok(files)
}

/// A pasted text becomes a Markdown (or `.txt`) file named by its title.
pub(crate) fn pasted_file(text: &str, title: Option<&str>) -> ApiResult<(String, Vec<u8>)> {
    if text.trim().is_empty() {
        return Err(ApiError::bad_request("text must not be empty"));
    }
    let title = title
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .unwrap_or("pasted")
        .to_owned();
    let has_text_extension = std::path::Path::new(&title)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("md") || e.eq_ignore_ascii_case("txt"));
    let filename = if has_text_extension {
        title
    } else {
        format!("{title}.md")
    };
    Ok((filename, text.as_bytes().to_vec()))
}

/// Register each file with status `queued`, audit it, and hand it to the
/// workspace's upload lane. Returns `{id, filename, status}` per file; a
/// file identical to a document already in the workspace is not queued
/// and comes back as `{id, filename, status: "duplicate"}` naming the
/// existing document. A pasted text's title is its filename stem.
pub(crate) async fn enqueue(
    app: &App,
    access: &Access,
    source: DocumentSource,
    files: Vec<(String, Vec<u8>)>,
) -> ApiResult<Vec<serde_json::Value>> {
    if files.is_empty() {
        return Err(ApiError::bad_request("no file or text in the request"));
    }
    let id = access.workspace.id.clone();
    // Backpressure: every queued upload holds its bytes in memory, so a
    // workspace with a deep line of them turns more away until it drains.
    let waiting = app.jobs.lane_active(&upload_lane(&id));
    if waiting.saturating_add(files.len()) > MAX_WAITING_UPLOADS {
        return Err(ApiError::busy(
            format!("{waiting} uploads are already waiting in this workspace; try again shortly"),
            UPLOAD_RETRY_SECONDS,
        ));
    }
    let db = app.workspace_db(&id).await?;
    let mut queued = Vec::new();
    for (filename, data) in files {
        let filename = std::path::Path::new(&filename)
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_owned)
            .ok_or_else(|| ApiError::bad_request("bad file name"))?;
        let size = data.len();
        let name = filename.clone();
        let user = access.identity.user_id.clone();
        let title = match source {
            DocumentSource::Paste => std::path::Path::new(&filename)
                .file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_owned),
            DocumentSource::Upload
            | DocumentSource::Path
            | DocumentSource::Stdin
            | DocumentSource::Import => None,
        };
        let registration = with_db(Arc::clone(&db), move |db| {
            ingestion::register_document(
                db,
                &ingestion::NewFile::new(&name, &data)
                    .source(source)
                    .title(title.as_deref())
                    .ingested_by(Some(&user)),
            )
            .map(|r| (r, data))
        })
        .await?;
        let (document_id, data) = match registration {
            (ingestion::Registration::New(id), data) => (id, data),
            (ingestion::Registration::Duplicate(existing), _) => {
                access
                    .audit(
                        app,
                        "ingest",
                        Some(("document", &existing.id)),
                        Outcome::Allowed,
                        Some(serde_json::json!({
                            "filename": filename,
                            "size_bytes": size,
                            "duplicate": true,
                        })),
                    )
                    .await?;
                queued.push(serde_json::json!({
                    "id": existing.id,
                    "filename": filename,
                    "status": "duplicate",
                    "existing_filename": existing.filename,
                }));
                continue;
            }
        };
        access
            .audit(
                app,
                "ingest",
                Some(("document", &document_id)),
                Outcome::Allowed,
                Some(serde_json::json!({ "filename": filename, "size_bytes": size })),
            )
            .await?;
        let job = submit_upload(
            app,
            &id,
            Some(access.identity.user_id.clone()),
            Arc::clone(&db),
            UploadJob {
                document_id: document_id.clone(),
                filename: filename.clone(),
                data,
            },
        );
        queued.push(serde_json::json!({
            "id": document_id,
            "filename": filename,
            "status": "queued",
            "job": job,
        }));
    }
    Ok(queued)
}

#[derive(Deserialize)]
pub(crate) struct UpdateDocument {
    pub pinned: bool,
}

pub(crate) async fn update(
    State(app): State<App>,
    identity: Identity,
    Path((id, doc)): Path<(String, String)>,
    Json(body): Json<UpdateDocument>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let document = set_pinned(&app, &access, &doc, body.pinned).await?;
    Ok(Json(serde_json::to_value(document)?))
}

/// Pin or unpin, audited.
pub(crate) async fn set_pinned(
    app: &App,
    access: &Access,
    doc: &str,
    pinned: bool,
) -> ApiResult<DocumentInfo> {
    let db = app.workspace_db(&access.workspace.id).await?;
    let doc_id = doc.to_owned();
    let document = with_db(db, move |db| {
        db.set_document_pinned(&doc_id, pinned)?;
        db.document(&doc_id)?
            .ok_or_else(|| Record::Document.missing(doc_id.as_str()))
    })
    .await?;
    access
        .audit(
            app,
            "context",
            Some(("document", doc)),
            Outcome::Allowed,
            Some(serde_json::json!({ "pinned": pinned })),
        )
        .await?;
    Ok(document)
}

pub(crate) async fn remove(
    State(app): State<App>,
    identity: Identity,
    Path((id, doc)): Path<(String, String)>,
) -> ApiResult<StatusCode> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    delete_document(&app, &access, &doc).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Delete a document (and its table when it was loaded as one), audited.
/// Returns the filename.
pub(crate) async fn delete_document(app: &App, access: &Access, doc: &str) -> ApiResult<String> {
    let db = app.workspace_db(&access.workspace.id).await?;
    let doc_id = doc.to_owned();
    let filename = with_db(db, move |db| {
        let document = db
            .document(&doc_id)?
            .ok_or_else(|| Record::Document.missing(doc_id.as_str()))?;
        let table = ingestion::parser::detect_file_type(&document.filename)
            .is_structured()
            .then(|| ingestion::table_name_for(&document.filename));
        db.delete_document(&doc_id, table.as_deref())?;
        Ok(document.filename)
    })
    .await?;
    access
        .audit(
            app,
            "delete",
            Some(("document", doc)),
            Outcome::Allowed,
            Some(serde_json::json!({ "filename": filename })),
        )
        .await?;
    Ok(filename)
}

use axum::extract::FromRequest;
