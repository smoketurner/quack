//! Documents: list, upload (multipart files or pasted text) into the
//! queue, status, pin, delete.

use std::sync::Arc;

use axum::Json;
use std::collections::HashMap;

use axum::extract::{FromRequest, Multipart, Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use quack_core::ids::{DocumentId, WorkspaceId};
use quack_core::ingestion;
use quack_core::jobs::JobId;
use quack_core::llm::Embeddings;
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};
use quack_core::text::NonBlankText;
use serde::{Deserialize, Serialize};

use crate::auth::{Access, Identity, Need};
use crate::error::{ApiError, ApiResult};
use crate::queue::UploadJob;
use crate::state::{App, with_db};
use quack_core::analysis::tools::SharedDb;
use quack_core::okf::{self, Bundle};
use quack_core::ontology::OntologyVersion;
use quack_core::ontology::store::Revision;
use quack_core::storage::workspace::{
    ChunkSearchResult, DocumentFields, DocumentFilter, DocumentInfo, DocumentListing, DocumentPage,
    DocumentSource, Pinning, Shown,
};
use utoipa::ToSchema;

/// `GET .../documents`'s filter: lists comma-separated, dates as
/// `YYYY-MM-DD`, each field given narrowing the listing.
#[derive(Debug, Default, Deserialize, utoipa::IntoParams)]
#[serde(deny_unknown_fields)]
pub(crate) struct ListFilter {
    /// File types, as extensions or MIME types.
    types: Option<String>,
    /// `upload`, `paste`, `path`, `stdin`, or `import`.
    sources: Option<String>,
    tags: Option<String>,
    /// Written on or after this date.
    since: Option<jiff::civil::Date>,
    /// Written on or before this date.
    until: Option<jiff::civil::Date>,
    author: Option<String>,
    /// The `next` of the page before; leave it out for the newest.
    after: Option<DocumentId>,
    /// Documents a page holds: 100 unless given, at most 500.
    limit: Option<u32>,
}

impl ListFilter {
    fn items(list: Option<&str>) -> Vec<String> {
        list.unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(str::to_owned)
            .collect()
    }

    /// The page it asks for.
    fn listing(&self) -> ApiResult<DocumentListing> {
        Ok(DocumentListing {
            shown: Shown::Live,
            filter: self.filter()?,
            after: self.after.clone(),
            limit: self.limit.unwrap_or(DocumentListing::PAGE),
        })
    }

    /// The filter it asks for.
    fn filter(&self) -> ApiResult<DocumentFilter> {
        let mut sources = Vec::new();
        for source in Self::items(self.sources.as_deref()) {
            sources.push(
                source
                    .parse::<DocumentSource>()
                    .map_err(|e| ApiError::bad_request(e.to_string()))?,
            );
        }
        Ok(DocumentFilter {
            types: Self::items(self.types.as_deref()),
            sources,
            tags: Self::items(self.tags.as_deref()),
            since: self.since,
            until: self.until,
            author: self.author.clone(),
        })
    }
}

/// One page of the live documents, newest first, narrowed by the filter.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/documents",
    tag = "documents",
    params(WorkspaceId, ListFilter),
    responses((status = 200, description = "One page of the documents", body = DocumentPage)),
)]
pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Query(q): Query<ListFilter>,
) -> ApiResult<Json<DocumentPage>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let listing = q.listing()?;
    access
        .audit_read(&app, AuditAction::List, "documents")
        .await?;
    let page = app.read(&id, move |db| db.documents(&listing)).await?;
    Ok(Json(page))
}

/// One document, superseded or not.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/documents/{doc}",
    tag = "documents",
    responses((status = 200, description = "The document", body = DocumentInfo)),
)]
pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path((id, doc)): Path<(WorkspaceId, DocumentId)>,
) -> ApiResult<Json<DocumentInfo>> {
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
                .ok_or_else(|| ResourceKind::Document.missing(doc.as_str()))
        })
        .await?;
    Ok(Json(document))
}

/// `?from=&limit=` on `GET .../documents/{doc}/chunks`: chunk positions
/// from `from` on, `limit` of them.
#[derive(Debug, Clone, Copy, Deserialize, utoipa::IntoParams)]
pub(crate) struct ChunkPage {
    /// The first chunk's position, from 0.
    #[serde(default)]
    pub from: u32,
    /// Chunks at most, up to 200.
    #[serde(default = "ChunkPage::default_limit")]
    #[param(default = 20, maximum = 200)]
    pub limit: u32,
}

impl ChunkPage {
    /// Chunks one request returns at most.
    pub(crate) const MAX_LIMIT: u32 = 200;

    const fn default_limit() -> u32 {
        20
    }

    /// The chunk at `position` with its neighbours, for a passage page.
    pub(crate) const fn around(position: u32) -> Self {
        Self {
            from: position.saturating_sub(1),
            limit: 3,
        }
    }
}

/// A page of one document's chunks in document order, with the
/// document itself (its `chunk_count` is the total).
pub(crate) struct Chunks {
    pub document: DocumentInfo,
    pub chunks: Vec<ChunkSearchResult>,
}

/// Read `page` of `doc`'s chunks, audited as opening the document; what
/// the REST route and the web passage page share.
pub(crate) async fn read_chunks(
    app: &App,
    access: &Access,
    doc: &DocumentId,
    page: ChunkPage,
) -> ApiResult<Chunks> {
    let limit = page.limit.clamp(1, ChunkPage::MAX_LIMIT);
    access
        .audit(
            app,
            AuditAction::Open,
            Some(ResourceKind::Document.id(doc)),
            Outcome::Allowed,
            Some(serde_json::json!({ "chunks_from": page.from, "limit": limit })),
        )
        .await?;
    let doc = doc.clone();
    app.read(&access.membership.workspace.id, move |db| {
        let document = db
            .document(&doc)?
            .ok_or_else(|| ResourceKind::Document.missing(doc.as_str()))?;
        let chunks = db.document_chunks(&doc, page.from, limit)?;
        Ok(Chunks { document, chunks })
    })
    .await
}

/// A page of one document's chunks.
#[derive(Serialize, ToSchema)]
pub(crate) struct ChunksPage {
    pub document_id: DocumentId,
    pub filename: String,
    /// Chunks in the document.
    pub total: Option<i64>,
    pub from: u32,
    pub chunks: Vec<ChunkSearchResult>,
}

/// `GET .../documents/{doc}/chunks?from=&limit=`: the document's chunks
/// from position `from`, each with its text, heading, page, and position.
#[utoipa::path(
    get,
    path = "/workspaces/{id}/documents/{doc}/chunks",
    tag = "documents",
    params(ChunkPage),
    responses((status = 200, description = "The chunks, in document order", body = ChunksPage)),
)]
pub(crate) async fn chunks(
    State(app): State<App>,
    identity: Identity,
    Path((id, doc)): Path<(WorkspaceId, DocumentId)>,
    Query(page): Query<ChunkPage>,
) -> ApiResult<Json<ChunksPage>> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let Chunks { document, chunks } = read_chunks(&app, &access, &doc, page).await?;
    Ok(Json(ChunksPage {
        document_id: document.id,
        filename: document.filename,
        total: document.chunk_count,
        from: page.from,
        chunks,
    }))
}

/// The image an image document keeps, as the response that serves it,
/// audited as opening the document.
pub(crate) async fn read_image(
    app: &App,
    access: &Access,
    doc: &DocumentId,
) -> ApiResult<axum::response::Response> {
    let wanted = doc.clone();
    let found = app
        .read(&access.membership.workspace.id, move |db| {
            let document = db
                .document(&wanted)?
                .ok_or_else(|| ResourceKind::Document.missing(wanted.as_str()))?;
            db.stored_image(&document)
                .ok_or_else(|| ResourceKind::Document.missing(format!("an image named {wanted}")))
        })
        .await;
    access
        .audit(
            app,
            AuditAction::Open,
            Some(ResourceKind::Document.id(doc)),
            if found.is_ok() {
                Outcome::Allowed
            } else {
                Outcome::Error
            },
            Some(serde_json::json!({ "image": true })),
        )
        .await?;
    let image = found?;
    let bytes = image.read().await?;
    Ok(([(header::CONTENT_TYPE, image.format().mime_type())], bytes).into_response())
}

/// `GET .../documents/{doc}/image`: the image an image document was
/// ingested from (PNG, JPEG, WebP, or GIF).
#[utoipa::path(
    get,
    path = "/workspaces/{id}/documents/{doc}/image",
    tag = "documents",
    responses((status = 200, description = "The image, as uploaded", content(
        (Vec<u8> = "image/png"),
        (Vec<u8> = "image/jpeg"),
        (Vec<u8> = "image/webp"),
        (Vec<u8> = "image/gif"),
    ))),
)]
pub(crate) async fn image(
    State(app): State<App>,
    identity: Identity,
    Path((id, doc)): Path<(WorkspaceId, DocumentId)>,
) -> ApiResult<axum::response::Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    read_image(&app, &access, &doc).await
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct PastedText {
    pub text: String,
    /// The document's title, and its file name's stem.
    pub title: Option<String>,
}

/// The `multipart/form-data` upload: one or more `file` parts.
#[derive(ToSchema)]
#[expect(
    dead_code,
    reason = "documents the multipart body; the handler reads the parts itself"
)]
pub(crate) struct UploadFiles {
    #[schema(value_type = Vec<String>, format = Binary)]
    file: Vec<Vec<u8>>,
}

/// `?replace={doc}` on `POST .../documents`: the one file in the request
/// replaces that ready document.
#[derive(Debug, Default, Deserialize, utoipa::IntoParams)]
pub(crate) struct UploadQuery {
    /// A ready document the one uploaded file takes the place of.
    pub replace: Option<DocumentId>,
}

/// What an upload queued.
#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum Uploaded {
    /// Files or pasted text.
    Files { documents: Vec<Enqueued> },
    /// An OKF bundle: its documents, the ontology candidates it queued, the
    /// ontology version it restored, and its `index.md` body to apply as
    /// the workspace context.
    Bundle {
        documents: Vec<Enqueued>,
        candidates: usize,
        ontology_version: Option<OntologyVersion>,
        context: Option<String>,
    },
}

/// `multipart/form-data` with one or more `file` parts, or JSON
/// `{text, title}`. Returns 202 with the queued documents. With
/// `?replace={doc}` the request carries one file, which takes the place
/// of that document once it is ready.
#[utoipa::path(
    post,
    path = "/workspaces/{id}/documents",
    tag = "documents",
    params(WorkspaceId, UploadQuery),
    request_body(content(
        (UploadFiles = "multipart/form-data"),
        (PastedText = "application/json"),
        (Vec<u8> = "application/x-tar"),
    ), description = "Files, pasted text, or an OKF bundle as a tar"),
    responses((status = 202, description = "Queued", body = Uploaded)),
)]
pub(crate) async fn upload(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<WorkspaceId>,
    Query(query): Query<UploadQuery>,
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
        if query.replace.is_some() {
            return Err(ApiError::bad_request(
                "replace takes one file, not a bundle",
            ));
        }
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
    let queued = enqueue(&app, &access, source, files, query.replace).await?;
    Ok((
        StatusCode::ACCEPTED,
        Json(Uploaded::Files { documents: queued }),
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
        enqueue(app, access, DocumentSource::Upload, files, None).await?
    };
    let db = app.workspace_db(&access.membership.workspace.id).await?;
    let for_candidates = bundle.clone();
    let author = access.identity.username.clone();
    let report = with_db(db, move |db| {
        for_candidates.restore_into(
            db,
            Revision::reviewed(Some(&author), Some("restored from a bundle")),
        )
    })
    .await?;
    // `restore_into` writes the workspace ontology (when it had none) and a
    // candidate run; mirror `api/ontology.rs` and audit each write so the
    // workspace audit trail records who installed the bundle.
    if let Some(version) = report.restored {
        access
            .audit(
                app,
                AuditAction::Ontology,
                Some(ResourceKind::OntologyVersion.id(&version.to_string())),
                Outcome::Allowed,
                Some(serde_json::json!({ "restored": true, "source": "bundle" })),
            )
            .await?;
    }
    if let Some(run) = report.run.as_ref() {
        access
            .audit(
                app,
                AuditAction::Propose,
                Some(ResourceKind::InductionRun.id(run)),
                Outcome::Allowed,
                Some(serde_json::json!({ "candidates": report.candidates, "source": "bundle" })),
            )
            .await?;
    }
    let context = bundle.index().map(|index| {
        okf::parse_front_matter(&index.content)
            .body
            .trim()
            .to_owned()
    });
    Ok((
        StatusCode::ACCEPTED,
        Json(Uploaded::Bundle {
            documents: queued,
            candidates: report.candidates,
            ontology_version: report.restored,
            context,
        }),
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
/// existing document. A pasted text's title is its filename stem. With
/// `replaces`, the one file takes that document's place once ready.
pub(crate) async fn enqueue(
    app: &App,
    access: &Access,
    source: DocumentSource,
    files: Vec<IncomingFile>,
    replaces: Option<DocumentId>,
) -> ApiResult<Vec<Enqueued>> {
    if files.is_empty() {
        return Err(ApiError::bad_request("no file or text in the request"));
    }
    if replaces.is_some() && files.len() != 1 {
        return Err(ApiError::bad_request("replace takes exactly one file"));
    }
    let id = access.membership.workspace.id.clone();
    // Fail now, not in the background, when no model can be built.
    let embedder = access
        .model(
            app,
            AuditAction::Ingest,
            Embeddings::from_config(&app.config).await,
        )
        .await?;
    let db = app.workspace_db(&id).await?;
    let mut queued = Vec::with_capacity(files.len());
    for file in files {
        let lane = Lane {
            app,
            access,
            db: &db,
            embedder: embedder.clone(),
            source,
            replaces: replaces.as_ref(),
        };
        queued.push(lane.enqueue_one(file).await?);
    }
    Ok(queued)
}

/// Where one upload goes: the workspace, who sends it, and what it
/// replaces.
struct Lane<'a> {
    app: &'a App,
    access: &'a Access,
    db: &'a SharedDb,
    embedder: Option<Embeddings>,
    source: DocumentSource,
    replaces: Option<&'a DocumentId>,
}

impl Lane<'_> {
    /// Register `file`, audit it, and hand it to the upload lane, or
    /// report the document that already holds its bytes.
    async fn enqueue_one(self, file: IncomingFile) -> ApiResult<Enqueued> {
        let IncomingFile {
            name: filename,
            data,
        } = file;
        let (app, access) = (self.app, self.access);
        let id = &access.membership.workspace.id;
        let filename = std::path::Path::new(&filename)
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_owned)
            .ok_or_else(|| ApiError::bad_request("bad file name"))?;
        let size = data.len();
        let name = filename.clone();
        let user = access.identity.user_id.clone();
        let title = match self.source {
            DocumentSource::Paste => std::path::Path::new(&filename)
                .file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_owned),
            DocumentSource::Upload
            | DocumentSource::Path
            | DocumentSource::Stdin
            | DocumentSource::Import
            | DocumentSource::Classify => None,
        };
        let (source, old) = (self.source, self.replaces.cloned());
        let config = app.config.clone();
        let registration = with_db(Arc::clone(self.db), move |db| {
            ingestion::register_document(
                db,
                &config,
                &ingestion::NewFile::new(&name, &data)
                    .source(source)
                    .title(title.as_deref())
                    .ingested_by(Some(user.as_str()))
                    .replaces(old.as_ref()),
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
                return Ok(Enqueued::Duplicate {
                    id: existing.id,
                    filename,
                    existing_filename: existing.filename,
                });
            }
        };
        access
            .audit(
                app,
                AuditAction::Ingest,
                Some(ResourceKind::Document.id(&document_id)),
                Outcome::Allowed,
                Some(serde_json::json!({
                    "filename": filename,
                    "size_bytes": size,
                    "replaces": self.replaces,
                })),
            )
            .await?;
        let job = UploadJob::spool(
            app,
            id,
            self.db,
            document_id.clone(),
            filename.clone(),
            &data,
        )
        .await?
        .submit(app, access, Arc::clone(self.db), self.embedder);
        Ok(Enqueued::Queued {
            id: document_id,
            filename,
            job,
        })
    }
}

/// What became of one file given to [`enqueue`].
#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum Enqueued {
    /// Registered and queued for processing.
    Queued {
        id: DocumentId,
        filename: String,
        job: JobId,
    },
    /// The same bytes are already a document; nothing was queued.
    Duplicate {
        id: DocumentId,
        filename: String,
        existing_filename: String,
    },
}

/// `PATCH .../documents/{doc}`: pin or unpin, and the fields a person may
/// set (title, author, authored date, tags); each given field is applied.
#[derive(Deserialize, ToSchema)]
pub(crate) struct UpdateDocument {
    #[serde(default)]
    pub pinned: Option<Pinning>,
    #[serde(flatten)]
    pub fields: DocumentFields,
}

/// Pin or unpin a document, or set its title, author, authored date, or tags.
#[utoipa::path(
    patch,
    path = "/workspaces/{id}/documents/{doc}",
    tag = "documents",
    request_body = UpdateDocument,
    responses((status = 200, description = "The document as it now is", body = DocumentInfo)),
)]
pub(crate) async fn update(
    State(app): State<App>,
    identity: Identity,
    Path((id, doc)): Path<(WorkspaceId, DocumentId)>,
    Json(body): Json<UpdateDocument>,
) -> ApiResult<Json<DocumentInfo>> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let pinned = match body.pinned {
        Some(pinning) => Some(set_pinned(&app, &access, &doc, pinning).await?),
        None => None,
    };
    let document = if body.fields.is_empty() {
        pinned
    } else {
        Some(set_fields(&app, &access, &doc, body.fields).await?)
    };
    document.map(Json).ok_or_else(|| {
        ApiError::bad_request("nothing to change: give pinned, title, author, authored_at, or tags")
    })
}

/// Set a document's own fields, audited with what changed.
pub(crate) async fn set_fields(
    app: &App,
    access: &Access,
    doc: &DocumentId,
    fields: DocumentFields,
) -> ApiResult<DocumentInfo> {
    let db = app.workspace_db(&access.membership.workspace.id).await?;
    let (doc_id, detail) = (doc.clone(), serde_json::to_value(FieldsDetail(&fields))?);
    let document = with_db(db, move |db| {
        db.set_document_fields(&doc_id, &fields)?;
        db.document(&doc_id)?
            .ok_or_else(|| ResourceKind::Document.missing(doc_id.as_str()))
    })
    .await?;
    access
        .audit(
            app,
            AuditAction::Context,
            Some(ResourceKind::Document.id(doc)),
            Outcome::Allowed,
            Some(detail),
        )
        .await?;
    Ok(document)
}

/// The audit detail of a fields edit: which fields were set, never the
/// values (a title or a tag is workspace content).
struct FieldsDetail<'a>(&'a DocumentFields);

impl Serialize for FieldsDetail<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let set: Vec<&str> = [
            ("title", self.0.title.is_some()),
            ("author", self.0.author.is_some()),
            ("authored_at", self.0.authored_at.is_some()),
            ("tags", self.0.tags.is_some()),
        ]
        .into_iter()
        .filter_map(|(name, given)| given.then_some(name))
        .collect();
        serde_json::json!({ "fields": set }).serialize(serializer)
    }
}

/// Pin or unpin, audited.
pub(crate) async fn set_pinned(
    app: &App,
    access: &Access,
    doc: &DocumentId,
    pinning: Pinning,
) -> ApiResult<DocumentInfo> {
    let db = app.workspace_db(&access.membership.workspace.id).await?;
    let doc_id = doc.clone();
    let document = with_db(db, move |db| {
        db.set_document_pinning(&doc_id, pinning)?;
        db.document(&doc_id)?
            .ok_or_else(|| ResourceKind::Document.missing(doc_id.as_str()))
    })
    .await?;
    access
        .audit(
            app,
            AuditAction::Context,
            Some(ResourceKind::Document.id(doc)),
            Outcome::Allowed,
            Some(serde_json::json!({ "pinned": bool::from(pinning) })),
        )
        .await?;
    Ok(document)
}

/// Delete the document, and its table when it was loaded as one.
#[utoipa::path(
    delete,
    path = "/workspaces/{id}/documents/{doc}",
    tag = "documents",
    responses((status = 204, description = "Deleted")),
)]
pub(crate) async fn remove(
    State(app): State<App>,
    identity: Identity,
    Path((id, doc)): Path<(WorkspaceId, DocumentId)>,
) -> ApiResult<StatusCode> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    delete_document(&app, &access, &doc).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Delete a document (and its table when it was loaded as one), audited.
/// Returns the filename.
pub(crate) async fn delete_document(
    app: &App,
    access: &Access,
    doc: &DocumentId,
) -> ApiResult<String> {
    let db = app.workspace_db(&access.membership.workspace.id).await?;
    let doc_id = doc.clone();
    let filename = with_db(db, move |db| {
        let document = db
            .document(&doc_id)?
            .ok_or_else(|| ResourceKind::Document.missing(doc_id.as_str()))?;
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
