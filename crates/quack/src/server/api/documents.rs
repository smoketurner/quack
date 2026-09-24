//! Documents: list, upload (multipart files or pasted text) into the
//! queue, status, pin, delete.

use std::sync::Arc;

use axum::Json;
use std::collections::HashMap;

use axum::extract::{FromRequest, Multipart, Path, State};
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use quack_core::error::Record;
use quack_core::ingestion;
use quack_core::jobs::{JobId, LaneKey};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use quack_core::text::NonBlankText;
use serde::{Deserialize, Serialize};

use crate::server::auth::{Access, Identity, Need};
use crate::server::error::{ApiError, ApiResult};
use crate::server::queue::{MAX_WAITING_UPLOADS, UPLOAD_RETRY_SECONDS, UploadJob};
use crate::server::state::{App, with_db};
use quack_core::okf::{self, Bundle};
use quack_core::ontology::store::Revision;
use quack_core::storage::workspace::{DocumentInfo, DocumentSource, WorkspaceDb};

pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit_read(&app, AuditAction::List, "documents")
        .await?;
    let docs = app.read(&id, WorkspaceDb::list_documents).await?;
    Ok(Json(serde_json::json!({ "documents": docs })))
}

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path((id, doc)): Path<(String, String)>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    access
        .audit(
            &app,
            AuditAction::Open,
            Some(ResourceKind::Document.id(&doc)),
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
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
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
        UploadForm::read(multipart).await?.files
    } else {
        let bytes = axum::body::to_bytes(request.into_body(), usize::MAX)
            .await
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        let pasted: PastedText =
            serde_json::from_slice(&bytes).map_err(|e| ApiError::bad_request(e.to_string()))?;
        vec![IncomingFile::pasted(&pasted.text, pasted.title.as_deref())?]
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
    let files: Vec<IncomingFile> = bundle
        .documents()
        .map(|f| IncomingFile {
            name: f.document_name(),
            data: f.content.as_bytes().to_vec(),
        })
        .collect();
    let queued = if files.is_empty() {
        Vec::new()
    } else {
        enqueue(app, access, DocumentSource::Upload, files).await?
    };
    let db = app.workspace_db(&access.workspace.id).await?;
    let for_candidates = bundle.clone();
    let author = access.identity.username.clone();
    let report = with_db(db, move |db| {
        for_candidates.restore_into(
            db,
            Revision::reviewed(Some(&author), Some("restored from a bundle")),
        )
    })
    .await?;
    let context = bundle
        .index()
        .map(|index| okf::parse_front_matter(&index.content).1.trim().to_owned());
    Ok((
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "documents": queued,
            "candidates": report.candidates,
            "ontology_version": report.restored,
            "context": context,
        })),
    )
        .into_response())
}

/// One file an upload carried.
pub(crate) struct IncomingFile {
    pub name: String,
    pub data: Vec<u8>,
}

impl IncomingFile {
    /// A pasted text as a Markdown (or `.txt`) file named by its title.
    pub(crate) fn pasted(text: &str, title: Option<&str>) -> ApiResult<Self> {
        if text.trim().is_empty() {
            return Err(ApiError::bad_request("text must not be empty"));
        }
        let title = title
            .and_then(str::non_blank)
            .unwrap_or("pasted")
            .to_owned();
        let has_text_extension = std::path::Path::new(&title)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("md") || e.eq_ignore_ascii_case("txt"));
        let name = if has_text_extension {
            title
        } else {
            format!("{title}.md")
        };
        Ok(Self {
            name,
            data: text.as_bytes().to_vec(),
        })
    }
}

/// A `multipart/form-data` upload: every part with a file name, and the
/// plain fields beside them (the web form's pasted `text` and `title`).
pub(crate) struct UploadForm {
    pub files: Vec<IncomingFile>,
    pub fields: HashMap<String, String>,
}

impl UploadForm {
    /// Read every part. A file part with neither a name nor bytes (a form's
    /// file input left empty) is skipped.
    pub(crate) async fn read(mut multipart: Multipart) -> ApiResult<Self> {
        let mut form = Self {
            files: Vec::new(),
            fields: HashMap::new(),
        };
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|e| ApiError::bad_request(e.to_string()))?
        {
            let field_name = field.name().unwrap_or_default().to_owned();
            if let Some(name) = field.file_name().map(str::to_owned) {
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| ApiError::bad_request(e.to_string()))?;
                if !(name.is_empty() && data.is_empty()) {
                    form.files.push(IncomingFile {
                        name,
                        data: data.to_vec(),
                    });
                }
            } else {
                let value = field
                    .text()
                    .await
                    .map_err(|e| ApiError::bad_request(e.to_string()))?;
                form.fields.insert(field_name, value);
            }
        }
        Ok(form)
    }
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
    files: Vec<IncomingFile>,
) -> ApiResult<Vec<Enqueued>> {
    if files.is_empty() {
        return Err(ApiError::bad_request("no file or text in the request"));
    }
    let id = access.workspace.id.clone();
    // Backpressure: every queued upload holds its bytes in memory, so a
    // workspace with a deep line of them turns more away until it drains.
    let waiting = app.jobs.lane_active(&LaneKey::Ingest(id.clone()));
    if waiting.saturating_add(files.len()) > MAX_WAITING_UPLOADS {
        return Err(ApiError::busy(
            format!("{waiting} uploads are already waiting in this workspace; try again shortly"),
            UPLOAD_RETRY_SECONDS,
        ));
    }
    let db = app.workspace_db(&id).await?;
    let mut queued = Vec::new();
    for IncomingFile {
        name: filename,
        data,
    } in files
    {
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
                        AuditAction::Ingest,
                        Some(ResourceKind::Document.id(&existing.id)),
                        Outcome::Allowed,
                        Some(serde_json::json!({
                            "filename": filename,
                            "size_bytes": size,
                            "duplicate": true,
                        })),
                    )
                    .await?;
                queued.push(Enqueued::Duplicate {
                    id: existing.id,
                    filename,
                    existing_filename: existing.filename,
                });
                continue;
            }
        };
        access
            .audit(
                app,
                AuditAction::Ingest,
                Some(ResourceKind::Document.id(&document_id)),
                Outcome::Allowed,
                Some(serde_json::json!({ "filename": filename, "size_bytes": size })),
            )
            .await?;
        let job = UploadJob {
            document_id: document_id.clone(),
            filename: filename.clone(),
            data,
        }
        .submit(
            app,
            &id,
            Some(access.identity.user_id.clone()),
            Arc::clone(&db),
        );
        queued.push(Enqueued::Queued {
            id: document_id,
            filename,
            job,
        });
    }
    Ok(queued)
}

/// What became of one file given to [`enqueue`].
#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum Enqueued {
    /// Registered and queued for processing.
    Queued {
        id: String,
        filename: String,
        job: JobId,
    },
    /// The same bytes are already a document; nothing was queued.
    Duplicate {
        id: String,
        filename: String,
        existing_filename: String,
    },
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
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
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
            AuditAction::Context,
            Some(ResourceKind::Document.id(doc)),
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
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
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
        db.delete_document(&doc_id)?;
        Ok(document.filename)
    })
    .await?;
    access
        .audit(
            app,
            AuditAction::Delete,
            Some(ResourceKind::Document.id(doc)),
            Outcome::Allowed,
            Some(serde_json::json!({ "filename": filename })),
        )
        .await?;
    Ok(filename)
}
