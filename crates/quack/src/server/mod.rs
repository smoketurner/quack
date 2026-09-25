//! `quack serve`: the REST API (and, on top of it, the web UI) as a thin
//! client of `quack-core`. Design doc sections 11.1, 11.2, and 12.

mod api;
pub(crate) mod auth;
mod error;
mod mcp_http;
pub(crate) mod oidc;
mod queue;
mod resource;
mod run;
pub(crate) mod state;
#[cfg(test)]
mod tests;
mod web;

use std::fmt;
use std::io::Write;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::http::{HeaderValue, Request, StatusCode, header};
use axum::routing::get;
use quack_core::config::Config;
use tower_governor::governor::GovernorConfigBuilder;
use tower_governor::key_extractor::KeyExtractor;
use tower_governor::{GovernorError, GovernorLayer};
use tower_http::request_id::{
    MakeRequestId, PropagateRequestIdLayer, RequestId, SetRequestIdLayer,
};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::{DefaultOnResponse, TraceLayer};

use oidc::Oidc;
use quack_core::llm::acting::Acting;
use quack_core::llm::oauth::KeySource;
use quack_core::storage::control::{ControlPlane, sha256_hex};
use quack_core::vault::Vault;
use state::{App, AppState, ServeMode};

/// How long one request may take. Agent turns can be slow.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

/// Rate limit per caller: sustained rate and burst.
const RATE_PER_SECOND: u64 = 1;
const RATE_BURST: u32 = 120;

/// The same, for the two endpoints that check a password. The general limit
/// is sized for a browsing session and is far too loose to make password
/// guessing expensive, so the login routes carry their own (issue #73).
const LOGIN_RATE_PER_SECOND: u64 = 2;
const LOGIN_RATE_BURST: u32 = 10;

/// How often a limiter drops the per-key state that has fallen back to a
/// fresh bucket. governor holds one entry per caller until something sweeps
/// it, so without this the maps grow for the life of the process — one entry
/// per address that ever connected.
const RATE_CLEANUP_INTERVAL: Duration = Duration::from_secs(60);

/// Run `sweep` every `interval` for as long as the runtime lives.
///
/// Taking a closure rather than the limiter keeps governor's deeply generic
/// types out of a signature: the caller clones the `Arc` it already has and
/// the compiler infers the rest.
fn spawn_cleanup(interval: Duration, sweep: impl Fn() + Send + 'static) {
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        tracing::warn!("no runtime to sweep rate-limiter state on; it will not be reclaimed");
        return;
    };
    handle.spawn(async move {
        let mut tick = tokio::time::interval(interval);
        // The first tick fires immediately, when there is nothing to sweep.
        tick.tick().await;
        loop {
            tick.tick().await;
            sweep();
        }
    });
}

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
            return Ok(sha256_hex(token.as_bytes()));
        }
        Ok(req
            .extensions()
            .get::<axum::extract::ConnectInfo<SocketAddr>>()
            .map_or_else(|| String::from("anonymous"), |c| c.0.ip().to_string()))
    }
}

/// Put the login limiter in front of one route. Each call builds its own
/// bucket, so a browser hammering the form cannot spend the budget the API
/// login would have had, or the other way round.
pub(crate) fn throttled_login(
    route: axum::routing::MethodRouter<App>,
) -> axum::routing::MethodRouter<App> {
    let config = GovernorConfigBuilder::default()
        .per_second(LOGIN_RATE_PER_SECOND)
        .burst_size(LOGIN_RATE_BURST)
        .key_extractor(CallerKey)
        .finish()
        .map(Arc::new);
    let Some(config) = config else {
        // Unreachable with constants this builder accepts; an unthrottled
        // login is still better than a server that will not start.
        tracing::error!("login rate limiter could not be built; logins are not throttled");
        return route;
    };
    let limiter = Arc::clone(config.limiter());
    spawn_cleanup(RATE_CLEANUP_INTERVAL, move || limiter.retain_recent());
    route.layer(GovernorLayer::new(config))
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
    // Everything a caller can reach is rate limited, not just the API: the
    // web UI drives the same handlers, and MCP drives the agent. `/healthz`
    // stays outside, because a throttled health check reads as a dead
    // server to whatever is watching it.
    let mut limited = Router::new()
        .route("/mcp/v1/{workspace}", axum::routing::any(mcp_http::handle))
        .route(resource::METADATA_PATH, get(resource::metadata))
        .route(
            &format!("{}/{{*path}}", resource::METADATA_PATH),
            get(resource::metadata_for),
        )
        .nest("/api/v1", api::router())
        .merge(web::router());
    if let Some(config) = governor {
        let limiter = Arc::clone(config.limiter());
        spawn_cleanup(RATE_CLEANUP_INTERVAL, move || limiter.retain_recent());
        limited = limited.layer(GovernorLayer::new(config));
    }
    let limited = limited
        .layer(axum::middleware::map_response(no_store))
        // Every request gets an empty acting slot, which the identity
        // extractor fills once it knows the caller.
        .layer(axum::middleware::from_fn(acting_slot));
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .merge(limited)
        .layer(DefaultBodyLimit::max(upload_limit))
        // One span per request, carrying the id the request-id layer set
        // (it is the outer layer, so the header exists here); the response
        // event carries status and latency.
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &Request<axum::body::Body>| {
                    let request_id = request
                        .headers()
                        .get(auth::REQUEST_ID_HEADER)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("-");
                    tracing::info_span!(
                        "request",
                        method = %request.method(),
                        uri = %request.uri(),
                        request_id
                    )
                })
                .on_response(DefaultOnResponse::new().level(tracing::Level::INFO)),
        )
        .layer(TimeoutLayer::with_status_code(
            StatusCode::GATEWAY_TIMEOUT,
            REQUEST_TIMEOUT,
        ))
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(SetRequestIdLayer::x_request_id(RequestIdV7))
        .with_state(app)
}

/// Pages, API answers, downloads, and streams carry workspace content, so no
/// browser or proxy may keep a copy (a page must not come back from the Back
/// button after logout). A response that already chose a policy keeps it: the
/// static assets are public and revalidate by `ETag`. `Expires` and `Pragma`
/// cover HTTP/1.0 caches that ignore `Cache-Control`.
async fn no_store(mut response: axum::response::Response) -> axum::response::Response {
    let headers = response.headers_mut();
    if !headers.contains_key(header::CACHE_CONTROL) {
        headers.insert(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache, no-store, must-revalidate"),
        );
        headers.insert(header::EXPIRES, HeaderValue::from_static("0"));
        headers.insert(header::PRAGMA, HeaderValue::from_static("no-cache"));
    }
    response
}

/// The startup banner: what this server is and how it is configured.
struct Banner<'a> {
    config: &'a Config,
    addr: SocketAddr,
    mode: ServeMode,
    users: usize,
    workspaces: usize,
}

impl fmt::Display for Banner<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            config,
            addr,
            mode,
            users,
            workspaces,
        } = self;
        let chat = config.chat_model_ref().map_or_else(
            |_| String::from("none (set [general].chat_model)"),
            |m| m.to_string(),
        );
        let embedding = config.embedding_model_ref().ok().flatten().map_or_else(
            || String::from("none (documents stored without vectors)"),
            |m| m.to_string(),
        );
        let providers: Vec<String> = config
            .providers
            .iter()
            .map(|(name, p)| format!("{name} ({}, auth {})", p.provider_type, p.auth.mode()))
            .collect();
        let providers = if providers.is_empty() {
            String::from("none configured")
        } else {
            providers.join(", ")
        };
        let mode = match mode {
            ServeMode::Local => String::from("local: no login, one implicit owner"),
            ServeMode::Login => match &config.server.oidc {
                Some(oidc) => format!(
                    "password and token login, sign-in with {}, {users} user(s)",
                    oidc.issuer_url
                ),
                None => format!("password and token login, {users} user(s)"),
            },
        };
        write!(
            f,
            r"
     __
   <(o )___     quack {version}
    ( ._> /     knowledge engine: documents, tables, graph
     `---'

  listening      http://{addr}/
  mode           {mode}
  data dir       {data}
  workspaces     {workspaces}
  chat model     {chat}
  embeddings     {embedding}
  providers      {providers}
  uploads        up to {upload} MB, {workers} worker(s) per workspace
  api            http://{addr}/api/v1

",
            version = env!("CARGO_PKG_VERSION"),
            data = config.data_dir().display(),
            upload = config.ingestion.upload_max_mb,
            workers = config.server.workers_per_workspace,
        )
    }
}

/// Run the rest of the request with an acting slot of its own.
async fn acting_slot(
    request: Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    Acting::request(next.run(request)).await
}

/// Bind and serve until Ctrl-C.
pub(crate) async fn serve(
    config: Config,
    bind: Option<String>,
    mode: ServeMode,
) -> anyhow::Result<()> {
    let mode = if config.server.local {
        ServeMode::Local
    } else {
        mode
    };
    let bind = bind.unwrap_or_else(|| config.server.bind.clone());
    let addr: SocketAddr = bind
        .parse()
        .with_context(|| format!("'{bind}' is not a socket address"))?;
    if mode == ServeMode::Local && !addr.ip().is_loopback() {
        anyhow::bail!(
            "--local serves without authentication and must bind a loopback address, not {addr}"
        );
    }
    let control = ControlPlane::open(&config)
        .await
        .context("failed to open control plane")?;
    let users = control.list_users().await?.len();
    let workspaces = control.list_workspaces().await?.len();
    if mode == ServeMode::Login && users == 0 {
        tracing::warn!(
            "no users exist; nobody can log in until `quack user add NAME --admin` runs"
        );
    }
    let oidc = match (&config.server.oidc, mode) {
        (Some(oidc), ServeMode::Login) => Some(
            Oidc::new(
                oidc,
                Vault::new(config.data_dir(), KeySource::Keychain),
                control.clone(),
            )
            .context("failed to set up [server.oidc] sign-in")?,
        ),
        (Some(_), ServeMode::Local) => {
            tracing::warn!("[server.oidc] is ignored in local mode, which has no login");
            None
        }
        (None, _) => None,
    };
    {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        let banner = Banner {
            config: &config,
            addr,
            mode,
            users,
            workspaces,
        };
        write!(out, "{banner}")?;
        out.flush()?;
    }
    let app = Arc::new(AppState::new(config, control, mode, oidc));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("cannot listen on {addr}"))?;
    tracing::info!(%addr, ?mode, "quack serve listening");
    axum::serve(
        listener,
        router(app).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .context("server error")?;
    tracing::info!("stopped");
    Ok(())
}

/// Resolve on Ctrl-C or, on Unix, SIGTERM (what containers and systemd
/// send). In-flight requests finish; new connections are refused.
async fn shutdown_signal() {
    let ctrl_c = async {
        drop(tokio::signal::ctrl_c().await);
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(e) => {
                tracing::warn!(error = %e, "cannot listen for SIGTERM; only Ctrl-C stops the server");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => tracing::info!("received Ctrl-C; finishing in-flight requests"),
        () = terminate => tracing::info!("received SIGTERM; finishing in-flight requests"),
    }
}
