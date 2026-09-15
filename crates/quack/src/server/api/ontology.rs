//! The ontology: read, replace (a new version), versions, diff, restore.
//! Members write; viewers read; every write is audited as `ontology`.

use axum::Json;
use axum::extract::{Path, Query, State};
use quack_core::ontology::Ontology;
use quack_core::ontology::store;
use quack_core::storage::control::Outcome;
use serde::Deserialize;

use crate::server::auth::{Identity, Need, access};
use crate::server::error::{ApiError, ApiResult};
use crate::server::state::{App, with_db};

pub(crate) async fn show(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let _access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let current = with_db(db, store::current).await?;
    let ontology = current.ok_or_else(|| ApiError::not_found("no ontology yet"))?;
    Ok(Json(serde_json::to_value(ontology)?))
}

/// Validate the body and store it as the next version.
pub(crate) async fn replace(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let ontology = Ontology::from_json(&body.to_string())?;
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let stored = with_db(db, move |db| {
        store::save(db, &ontology, Some(&author), Some("imported"))
    })
    .await?;
    access
        .audit(
            &app,
            "ontology",
            Some(("ontology_version", &stored.version.to_string())),
            Outcome::Allowed,
            Some(serde_json::json!({ "version": stored.version, "classes": stored.classes.len() })),
        )
        .await?;
    Ok(Json(serde_json::to_value(stored)?))
}

/// Install the built-in general ontology as version 1.
pub(crate) async fn init(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let stored = with_db(db, move |db| {
        if store::latest_version(db)? > 0 {
            return Err(quack_core::error::Error::Ontology(String::from(
                "an ontology already exists",
            )));
        }
        store::save(
            db,
            &Ontology::builtin_default(),
            Some(&author),
            Some("built-in default"),
        )
    })
    .await
    .map_err(|e| ApiError::new(axum::http::StatusCode::CONFLICT, e.message))?;
    access
        .audit(
            &app,
            "ontology",
            Some(("ontology_version", "1")),
            Outcome::Allowed,
            None,
        )
        .await?;
    Ok(Json(serde_json::to_value(stored)?))
}

#[derive(Deserialize)]
pub(crate) struct VersionsQuery {
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    20
}

pub(crate) async fn versions(
    State(app): State<App>,
    identity: Identity,
    Path(id): Path<String>,
    Query(q): Query<VersionsQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let _access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let limit = q.limit;
    let rows = with_db(db, move |db| store::versions(db, limit)).await?;
    Ok(Json(serde_json::json!({ "versions": rows })))
}

#[derive(Deserialize)]
pub(crate) struct DiffQuery {
    /// The older version to compare against; default: the one before.
    pub against: Option<u32>,
}

pub(crate) async fn version(
    State(app): State<App>,
    identity: Identity,
    Path((id, v)): Path<(String, u32)>,
    Query(q): Query<DiffQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let _access = access(&app, identity, &id, Need::READ).await?;
    let db = app.workspace_db(&id).await?;
    let against = q.against.unwrap_or(v.saturating_sub(1));
    let (snapshot, older) = with_db(db, move |db| {
        let snapshot = store::version(db, v)?;
        let older = if against == 0 {
            None
        } else {
            store::version(db, against)?
        };
        Ok((snapshot, older))
    })
    .await?;
    let snapshot = snapshot.ok_or_else(|| ApiError::not_found("no such version"))?;
    let diff = older.map(|older| snapshot.diff(&older));
    Ok(Json(
        serde_json::json!({ "ontology": snapshot, "diff": diff }),
    ))
}

pub(crate) async fn restore(
    State(app): State<App>,
    identity: Identity,
    Path((id, v)): Path<(String, u32)>,
) -> ApiResult<Json<serde_json::Value>> {
    let access = access(&app, identity, &id, Need::WRITE).await?;
    let db = app.workspace_db(&id).await?;
    let author = access.identity.username.clone();
    let stored = with_db(db, move |db| store::restore(db, v, Some(&author)))
        .await
        .map_err(|e| ApiError::not_found(e.message))?;
    access
        .audit(
            &app,
            "ontology",
            Some(("ontology_version", &stored.version.to_string())),
            Outcome::Allowed,
            Some(serde_json::json!({ "restored": v, "version": stored.version })),
        )
        .await?;
    Ok(Json(serde_json::to_value(stored)?))
}
