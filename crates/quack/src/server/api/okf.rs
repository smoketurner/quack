//! The workspace as an Open Knowledge Format bundle: `GET .../okf` answers
//! a tar archive of Markdown files, audited as `export` because it moves
//! content across the boundary. Import is `POST .../documents` with a tar
//! body (see `documents::upload`).

use axum::extract::{Path, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use quack_core::storage::control::{AuditAction, Outcome, ResourceKind};

use crate::server::auth::{Access, Identity, Need};
use crate::server::error::ApiResult;
use crate::server::state::App;
use quack_core::okf;

pub(crate) async fn export(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let name = access.workspace.name.clone();
    let (bytes, files) = app
        .read(&id, move |db| {
            let bundle = okf::export(db, &name)?;
            Ok((bundle.to_tar()?, bundle.files.len()))
        })
        .await?;
    access
        .audit(
            &app,
            AuditAction::Export,
            Some(ResourceKind::Workspace.id(&id)),
            Outcome::Allowed,
            Some(serde_json::json!({ "format": "okf", "files": files })),
        )
        .await?;
    let filename = format!("{}.okf.tar", okf::slug(&access.workspace.name));
    Ok((
        [
            (header::CONTENT_TYPE, String::from("application/x-tar")),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{filename}\""),
            ),
        ],
        bytes,
    )
        .into_response())
}
