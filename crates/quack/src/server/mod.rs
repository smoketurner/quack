//! `quack serve`: the REST API (and, on top of it, the web UI) as a thin
//! client of `quack-core`. Design doc sections 11.1, 11.2, and 12.

mod api;
mod auth;
mod error;
mod queue;
mod state;
#[cfg(test)]
mod tests;
mod web;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::http::{HeaderValue, Request, StatusCode};
use axum::routing::get;
use quack_core::config::Config;
use quack_core::storage::control::ControlPlane;
use tower_governor::governor::GovernorConfigBuilder;
use tower_governor::key_extractor::KeyExtractor;
use tower_governor::{GovernorError, GovernorLayer};
use tower_http::request_id::{
    MakeRequestId, PropagateRequestIdLayer, RequestId, SetRequestIdLayer,
};
use tower_http::timeout::TimeoutLayer;

use state::{App, AppState};

/// How long one request may take. Agent turns can be slow.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

/// Rate limit per caller: sustained rate and burst.
const RATE_PER_SECOND: u64 = 1;
const RATE_BURST: u32 = 120;

#[derive(Clone)]
struct RequestIdV7;

impl MakeRequestId for RequestIdV7 {
    fn make_request_id<B>(&mut self, _: &Request<B>) -> Option<RequestId> {
        HeaderValue::from_str(&uuid::Uuid::now_v7().to_string())
            .ok()
            .map(RequestId::new)
    }
}

/// Rate-limit key: the bearer token when there is one, else the peer
/// address, else one shared bucket.
#[derive(Clone)]
struct CallerKey;

impl KeyExtractor for CallerKey {
    type Key = String;

    fn extract<T>(&self, req: &Request<T>) -> Result<Self::Key, GovernorError> {
        if let Some(token) = req
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
        {
            return Ok(quack_core::storage::control::sha256_hex(token.as_bytes()));
        }
        Ok(req
            .extensions()
            .get::<axum::extract::ConnectInfo<SocketAddr>>()
            .map_or_else(|| String::from("anonymous"), |c| c.0.ip().to_string()))
    }
}

pub(crate) fn router(app: App) -> Router {
    let upload_limit = usize::try_from(app.config.ingestion.upload_max_mb)
        .unwrap_or(usize::MAX)
        .saturating_mul(1024 * 1024);
    let governor = GovernorConfigBuilder::default()
        .per_second(RATE_PER_SECOND)
        .burst_size(RATE_BURST)
        .key_extractor(CallerKey)
        .finish()
        .map(Arc::new);
    let mut api = api::router();
    if let Some(config) = governor {
        api = api.layer(GovernorLayer::new(config));
    }
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .nest("/api/v1", api)
        .merge(web::router())
        .layer(DefaultBodyLimit::max(upload_limit))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::GATEWAY_TIMEOUT,
            REQUEST_TIMEOUT,
        ))
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(SetRequestIdLayer::x_request_id(RequestIdV7))
        .with_state(app)
}

/// Bind and serve until Ctrl-C.
pub(crate) async fn serve(config: Config, bind: Option<String>, local: bool) -> anyhow::Result<()> {
    let local = local || config.server.local;
    let bind = bind.unwrap_or_else(|| config.server.bind.clone());
    let addr: SocketAddr = bind
        .parse()
        .with_context(|| format!("'{bind}' is not a socket address"))?;
    if local && !addr.ip().is_loopback() {
        anyhow::bail!(
            "--local serves without authentication and must bind a loopback address, not {addr}"
        );
    }
    let control = ControlPlane::open(&config)
        .await
        .context("failed to open control plane")?;
    if !local && control.list_users().await?.is_empty() {
        tracing::warn!(
            "no users exist; nobody can log in until `quack user add NAME --admin` runs"
        );
    }
    let app = Arc::new(AppState::new(config, control, local));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("cannot listen on {addr}"))?;
    tracing::info!(%addr, local, "quack serve listening");
    axum::serve(
        listener,
        router(app).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async {
        drop(tokio::signal::ctrl_c().await);
        tracing::info!("shutting down");
    })
    .await
    .context("server error")?;
    Ok(())
}
