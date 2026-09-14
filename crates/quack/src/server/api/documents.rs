//! Documents: list, upload (multipart files or pasted text) into the
//! queue, status, pin, delete.

use std::sync::Arc;

use axum::extract::{Multipart, Path, State};
use axum::http::{StatusCode, header};
use axum::{Json, response::IntoResponse};
use quack_core::ingestion;
use quack_core::storage::control::Outcome;
use serde::Deserialize;

use crate::server::auth::{Identity, Need, access};
use crate::server::error::{ApiError, ApiResult};
use crate::server::queue::UploadJob;
use crate::server::state::{App, with_db};

pub(crate) async fn list(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let _access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let docs = with_db(
        db,
        quack_core::storage::workspace::WorkspaceDb::list_documents,
    )
    .await?;
    Ok(Json(serde_json::json!({ "documents": docs })))
}

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path((id, doc)): Path<(String, String)>,
) -> ApiResult<Json<serde_json::Value>> {
    let _access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let found = with_db(db, move |db| db.document(&doc)).await?;
    let document = found.ok_or_else(|| ApiError::not_found("no such document"))?;
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
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    if content_type.starts_with("multipart/form-data") {
        let mut multipart = Multipart::from_request(request, &app)
            .await
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
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
            files.push((name, data.to_vec()));
        }
    } else {
        let bytes = axum::body::to_bytes(request.into_body(), usize::MAX)
            .await
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        let pasted: PastedText =
            serde_json::from_slice(&bytes).map_err(|e| ApiError::bad_request(e.to_string()))?;
        if pasted.text.trim().is_empty() {
            return Err(ApiError::bad_request("text must not be empty"));
        }
        let title = pasted
            .title
            .filter(|t| !t.trim().is_empty())
            .unwrap_or_else(|| String::from("pasted"));
        let has_text_extension = std::path::Path::new(&title)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("md") || e.eq_ignore_ascii_case("txt"));
        let filename = if has_text_extension {
            title
        } else {
            format!("{title}.md")
        };
        files.push((filename, pasted.text.into_bytes()));
    }
    if files.is_empty() {
        return Err(ApiError::bad_request("no file or text in the request"));
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
        let document_id = with_db(Arc::clone(&db), move |db| {
            ingestion::register_document(db, &name, size)
        })
        .await?;
        access
            .audit(
                &app,
                "ingest",
                Some(("document", &document_id)),
                Outcome::Allowed,
                Some(serde_json::json!({ "filename": filename, "size_bytes": size })),
            )
            .await?;
        app.queue
            .submit(
                &app.config,
                &id,
                Arc::clone(&db),
                UploadJob {
                    document_id: document_id.clone(),
                    filename: filename.clone(),
                    data,
                },
            )
            .await
            .map_err(ApiError::internal)?;
        queued.push(
            serde_json::json!({ "id": document_id, "filename": filename, "status": "queued" }),
        );
    }
    Ok((
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "documents": queued })),
    ))
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
    let db = app.workspace_db(&id).await?;
    let doc_id = doc.clone();
    let updated = with_db(db, move |db| {
        if db.document(&doc_id)?.is_none() {
            return Ok(None);
        }
        db.set_document_pinned(&doc_id, body.pinned)?;
        db.document(&doc_id)
    })
    .await?;
    let document = updated.ok_or_else(|| ApiError::not_found("no such document"))?;
    access
        .audit(
            &app,
            "context",
            Some(("document", &doc)),
            Outcome::Allowed,
            Some(serde_json::json!({ "pinned": body.pinned })),
        )
        .await?;
    Ok(Json(serde_json::to_value(document)?))
}

pub(crate) async fn remove(
    State(app): State<App>,
    identity: Identity,
    Path((id, doc)): Path<(String, String)>,
) -> ApiResult<StatusCode> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let db = app.workspace_db(&id).await?;
    let doc_id = doc.clone();
    let removed = with_db(db, move |db| {
        let Some(document) = db.document(&doc_id)? else {
            return Ok(None);
        };
        let table = ingestion::parser::detect_file_type(&document.filename)
            .is_structured()
            .then(|| ingestion::table_name_for(&document.filename));
        db.delete_document(&doc_id, table.as_deref())?;
        Ok(Some(document.filename))
    })
    .await?;
    let filename = removed.ok_or_else(|| ApiError::not_found("no such document"))?;
    access
        .audit(
            &app,
            "delete",
            Some(("document", &doc)),
            Outcome::Allowed,
            Some(serde_json::json!({ "filename": filename })),
        )
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

use axum::extract::FromRequest;
