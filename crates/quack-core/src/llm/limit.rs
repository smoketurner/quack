//! Model requests are the scarce resource (design doc 4.1), so they are what
//! is limited: every rig client quack builds sends through a
//! [`LimitedHttp`], which takes one permit per request from the gate of its
//! provider and model (`[providers.NAME].max_concurrent_requests` each) and
//! holds it until the response body has been read or its stream has ended.
//!
//! Gates are process-wide: every client built for a provider, however often
//! (OAuth clients are rebuilt per call), shares them. The model is read
//! from the request body, so a chat model and an embedding model on one
//! Ollama server each get their own limit, as Ollama serves each model
//! separately. A freed permit goes to a waiting [`Priority::Interactive`]
//! request before any background one: a question is never queued behind a
//! whole ingest's embedding batches or an extraction's next chunk. The
//! priority is [`crate::priority`]'s task-local: background jobs run
//! background, and everything else, turns included, interactive.
//!
//! A turn waiting on the user's answer to a permission prompt, or running a
//! tool, holds nothing. rig's streaming loop drains a model response before
//! it runs the tool calls in it, so a tool that calls the same model never
//! waits on a permit its own turn still holds.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::Stream;
use rig::http_client::{
    self, BoxedStream, HttpClientExt, LazyBody, MultipartForm, Request, ReqwestClient, Response,
    StreamingResponse,
};
use tokio::sync::oneshot;

use super::bedrock::Signer;
use super::egress::Egress;
use crate::config::{
    BaseUrl, ProviderConfig, ProviderName, ProviderType, RequestLimit, RetryPolicy,
};
use crate::error::{self, Error};

use crate::priority::Priority;
use crate::proxy::Proxies;
use crate::telemetry;

/// Permits of one provider and model, handed to interactive waiters first.
struct Gate {
    state: Mutex<GateState>,
}

struct GateState {
    available: usize,
    interactive: VecDeque<oneshot::Sender<GatePermit>>,
    background: VecDeque<oneshot::Sender<GatePermit>>,
}

impl Gate {
    fn new(permits: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(GateState {
                available: permits,
                interactive: VecDeque::new(),
                background: VecDeque::new(),
            }),
        })
    }

    fn state(&self) -> std::sync::MutexGuard<'_, GateState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A permit now, or in turn: interactive waiters before background ones,
    /// each line first come, first served.
    async fn acquire(self: &Arc<Self>, priority: Priority) -> GatePermit {
        let receiver = {
            let mut state = self.state();
            if state.available > 0 {
                state.available = state.available.saturating_sub(1);
                return GatePermit::new(self);
            }
            let (sender, receiver) = oneshot::channel();
            match priority {
                Priority::Interactive => state.interactive.push_back(sender),
                Priority::Background => state.background.push_back(sender),
            }
            receiver
        };
        match receiver.await {
            Ok(permit) => permit,
            // Unreachable: a sender is only dropped after a send, and a
            // gate lives as long as any client holding it.
            Err(_) => GatePermit::new(self),
        }
    }

    /// A permit came back: hand it on, or return it to the pool.
    fn release(self: &Arc<Self>) {
        let mut state = self.state();
        loop {
            let next = state
                .interactive
                .pop_front()
                .or_else(|| state.background.pop_front());
            let Some(next) = next else {
                state.available = state.available.saturating_add(1);
                return;
            };
            match next.send(GatePermit::new(self)) {
                Ok(()) => return,
                // A waiter that gave up (its request was dropped): disarm
                // the returned permit so its drop does not re-enter here.
                Err(unclaimed) => unclaimed.disarm(),
            }
        }
    }

    #[cfg(test)]
    fn available(&self) -> usize {
        self.state().available
    }
}

/// One held permit; dropping it passes it on.
pub(crate) struct GatePermit(Option<Arc<Gate>>);

impl GatePermit {
    fn new(gate: &Arc<Gate>) -> Self {
        Self(Some(Arc::clone(gate)))
    }

    /// Drop without passing the permit on: the gate already counted it.
    fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for GatePermit {
    fn drop(&mut self) {
        if let Some(gate) = self.0.take() {
            gate.release();
        }
    }
}

/// A provider as its gates know it: two entries with one name and base URL
/// share their limits.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProviderKey {
    name: ProviderName,
    base_url: Option<BaseUrl>,
}

/// Which gate a request waits at: its provider, and the model its body
/// names (none for a request that names no model, as Ollama's
/// `GET api/ps`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GateKey {
    provider: ProviderKey,
    model: Option<String>,
}

impl GateKey {
    /// The gates, process-wide.
    fn registry() -> &'static Mutex<HashMap<Self, Arc<Gate>>> {
        static REGISTRY: OnceLock<Mutex<HashMap<GateKey, Arc<Gate>>>> = OnceLock::new();
        REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// The model a request names in its JSON body.
    fn model_of(body: &[u8]) -> Option<String> {
        #[derive(serde::Deserialize)]
        struct Named {
            model: Option<String>,
        }
        serde_json::from_slice::<Named>(body)
            .ok()
            .and_then(|n| n.model)
    }
}

/// A provider's gates, one per model, shared process-wide by every client
/// built for it. Every model request quack sends passes [`Self::permit`],
/// so it is also where the workspace's provider allow-list is enforced.
/// `Default` names no provider: unlimited and unchecked.
#[derive(Clone, Default)]
pub(crate) struct ProviderGates {
    /// The provider, or `None` for the default.
    provider: Option<Gated>,
}

/// The provider a set of gates belongs to.
#[derive(Clone)]
struct Gated {
    key: ProviderKey,
    kind: ProviderType,
    limit: RequestLimit,
}

/// How a request's attempts are counted and waited out: the provider's
/// `RetryPolicy`, and its name for the log and the metrics.
#[derive(Clone, Debug)]
pub(crate) struct Attempts {
    provider: String,
    policy: RetryPolicy,
}

impl Attempts {
    pub(crate) fn new(name: &ProviderName, policy: RetryPolicy) -> Self {
        Self {
            provider: name.to_string(),
            policy,
        }
    }

    /// Whether `status` is worth another attempt, and how long to wait
    /// first: `Retry-After` in seconds when the response names it, else
    /// the policy's backoff for retry `attempt`. `None` when the response
    /// stands (a success, a client error) or the attempts are used up.
    fn wait_for_status(
        &self,
        attempt: u32,
        status: http::StatusCode,
        headers: &http::HeaderMap,
    ) -> Option<Duration> {
        if attempt >= self.policy.max_retries
            || !rig::error::retryable_status(Some(status.as_u16()))
        {
            return None;
        }
        let named = headers
            .get(http::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(Duration::from_secs)
            .filter(|d| *d <= RetryPolicy::MAX_WAIT);
        Some(named.unwrap_or_else(|| self.policy.wait(attempt.saturating_add(1), jitter())))
    }

    /// Whether a failure is worth another attempt, and the wait: a failing
    /// status (rig's client turns one into an error, headers kept) as for
    /// a response, a dropped transport by the backoff.
    fn wait_for_error(&self, attempt: u32, error: &http_client::Error) -> Option<Duration> {
        if let http_client::Error::InvalidStatusCodeWithDetails {
            status, headers, ..
        } = error
        {
            return self.wait_for_status(attempt, *status, headers);
        }
        if attempt >= self.policy.max_retries || !rig::error::transient_transport(error) {
            return None;
        }
        Some(self.policy.wait(attempt.saturating_add(1), jitter()))
    }

    /// Note a retry: one `warn!` per attempt, and the counter.
    fn note(&self, model: Option<&str>, attempt: u32, wait: Duration, why: &str) {
        tracing::warn!(
            provider = %self.provider,
            model = model.unwrap_or("-"),
            attempt = attempt.saturating_add(1),
            of = self.policy.max_retries,
            wait_ms = wait.as_millis(),
            "{why}; retrying the model request"
        );
        telemetry::provider_retry(&self.provider, model);
    }
}

/// A fraction of a second's nanoseconds, so two waiters back off apart.
fn jitter() -> f64 {
    f64::from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos()),
    ) / 1_000_000_000.0
}

impl std::fmt::Debug for ProviderGates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderGates")
            .field("limit", &self.provider.as_ref().map(|gated| gated.limit))
            .finish_non_exhaustive()
    }
}

impl ProviderGates {
    /// The gates of provider `name`.
    pub(crate) fn for_provider(name: &ProviderName, provider: &ProviderConfig) -> Self {
        Self {
            provider: Some(Gated {
                key: ProviderKey {
                    name: name.clone(),
                    base_url: provider.base_url.clone(),
                },
                kind: provider.provider_type,
                limit: provider.request_limit(),
            }),
        }
    }

    /// Check the request against the calling task's [`Egress`], then wait
    /// for a permit of `model`'s gate at its priority (both read now, not
    /// when the future first runs); `None` when unlimited.
    ///
    /// # Errors
    ///
    /// The future returns [`Egress::permit`]'s refusal, and the request must
    /// not be sent.
    pub(crate) fn permit(
        &self,
        model: Option<String>,
    ) -> impl Future<Output = error::Result<Option<GatePermit>>> + Send + 'static {
        let permitted = match &self.provider {
            Some(gated) => Egress::permit(&gated.key.name, gated.kind, model.as_deref()),
            None => Ok(()),
        };
        let gate = self.gate(model);
        let priority = Priority::current();
        async move {
            permitted?;
            match gate {
                Some(gate) => Ok(Some(gate.acquire(priority).await)),
                None => Ok(None),
            }
        }
    }

    /// The gate for `model`, created with this client's limit on first use;
    /// a later config for the same provider in one process keeps the first.
    fn gate(&self, model: Option<String>) -> Option<Arc<Gate>> {
        let gated = self.provider.as_ref()?;
        let key = GateKey {
            provider: gated.key.clone(),
            model,
        };
        let limit = gated.limit;
        let mut map = GateKey::registry()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        Some(Arc::clone(map.entry(key).or_insert_with(|| {
            Gate::new(usize::try_from(limit.get()).unwrap_or(1))
        })))
    }
}

/// A reqwest client that takes a permit of its provider's limit for the
/// request's model before each request. `Default` is unlimited; quack
/// always builds one with [`LimitedHttp::for_provider`].
#[derive(Clone, Default, Debug)]
pub struct LimitedHttp {
    inner: ReqwestClient,
    gates: ProviderGates,
    /// The provider's `headers`, sent on every request beside rig's own.
    headers: http::HeaderMap,
    /// Replaces the credential rig set, once the request has its permit.
    authorize: Option<Authorize>,
    /// How a throttled or failed request is tried again; `None` sends once.
    attempts: Option<Attempts>,
}

/// How a client authorizes each request in place of rig's own header.
#[derive(Clone, Debug)]
enum Authorize {
    /// `SigV4` (Bedrock's OpenAI-compatible APIs), replacing the bearer.
    Sign(Arc<Signer>),
    /// An OAuth token as `Authorization: Bearer`, replacing the `x-api-key`
    /// rig's Anthropic client always sends, where gateways and Anthropic's
    /// own OAuth never look. Marked sensitive, so `Debug` hides it.
    Bearer(http::HeaderValue),
}

impl Authorize {
    const API_KEY: http::HeaderName = http::HeaderName::from_static("x-api-key");

    fn bearer(token: &str) -> error::Result<Self> {
        let mut value = http::HeaderValue::try_from(format!("Bearer {token}"))
            .map_err(|e| Error::Llm(format!("the OAuth token is not a header value: {e}")))?;
        value.set_sensitive(true);
        Ok(Self::Bearer(value))
    }

    /// Put the bearer in `headers`, in place of any key rig set.
    fn replace_key(value: &http::HeaderValue, headers: &mut http::HeaderMap) {
        headers.remove(Self::API_KEY);
        headers.insert(http::header::AUTHORIZATION, value.clone());
    }
}

impl LimitedHttp {
    /// The client for provider `name`, sharing its process-wide gates.
    #[must_use]
    pub fn for_provider(name: &ProviderName, provider: &ProviderConfig) -> Self {
        Self {
            // A client that cannot be built is a TLS setup failure, which
            // reqwest's default client meets the same way.
            inner: ReqwestClient::from(Proxies::from_env().client().build().unwrap_or_default()),
            gates: ProviderGates::for_provider(name, provider),
            headers: http::HeaderMap::new(),
            authorize: None,
            attempts: Some(Attempts::new(name, provider.retry)),
        }
    }

    /// The provider's name for the metrics, `-` for the default client.
    fn provider_name(&self) -> String {
        self.attempts
            .as_ref()
            .map_or_else(|| String::from("-"), |a| a.provider.clone())
    }

    /// This client, sending `headers` (a provider's `headers`) on every
    /// request beside the ones rig sets.
    #[must_use]
    pub(crate) fn with_headers(mut self, headers: http::HeaderMap) -> Self {
        self.headers = headers;
        self
    }

    /// This client, signing every request with `signer`.
    #[must_use]
    pub(crate) fn signed(mut self, signer: Arc<Signer>) -> Self {
        self.authorize = Some(Authorize::Sign(signer));
        self
    }

    /// This client, sending `token` as `Authorization: Bearer` on every
    /// request and dropping any `x-api-key`.
    ///
    /// # Errors
    ///
    /// Returns an error when `token` cannot be a header value.
    pub(crate) fn with_oauth_bearer(mut self, token: &str) -> error::Result<Self> {
        self.authorize = Some(Authorize::bearer(token)?);
        Ok(self)
    }

    /// Add the provider's `headers` to `request`, keeping any rig set.
    fn add_headers(headers: &http::HeaderMap, request: &mut http::HeaderMap) {
        for (name, value) in headers {
            if !request.contains_key(name) {
                request.insert(name.clone(), value.clone());
            }
        }
    }

    /// `request` with the provider's `headers`, authorized the way this
    /// client authorizes.
    #[expect(
        clippy::result_large_err,
        reason = "rig's HTTP error, which HttpClientExt returns; it keeps the failed response's headers"
    )]
    async fn prepare(
        authorize: Option<Authorize>,
        headers: &http::HeaderMap,
        mut request: Request<Bytes>,
    ) -> http_client::Result<Request<Bytes>> {
        Self::add_headers(headers, request.headers_mut());
        match authorize {
            Some(Authorize::Sign(signer)) => signer.sign(request).await,
            Some(Authorize::Bearer(value)) => {
                Authorize::replace_key(&value, request.headers_mut());
                Ok(request)
            }
            None => Ok(request),
        }
    }
}

/// Hold `permit` until the lazily read body has been read (or dropped).
fn body_holding<U: Send + 'static>(
    response: Response<LazyBody<U>>,
    permit: Option<GatePermit>,
) -> Response<LazyBody<U>> {
    response.map(|body| -> LazyBody<U> {
        Box::pin(async move {
            let read = body.await;
            drop(permit);
            read
        })
    })
}

/// A byte stream that releases its permit when it ends or fails, not only
/// when it is dropped, so a finished response never holds the model.
struct Holding {
    inner: BoxedStream,
    permit: Option<GatePermit>,
}

impl Stream for Holding {
    type Item = Result<Bytes, http_client::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let polled = self.inner.as_mut().poll_next(cx);
        if matches!(polled, Poll::Ready(None | Some(Err(_)))) {
            self.permit = None;
        }
        polled
    }
}

/// The parts of a request that every attempt is rebuilt from: the
/// headers and body as rig gave them, before the provider's headers and
/// credential go on (a signature is time-bound, so each attempt signs
/// anew).
struct Blueprint {
    parts: http::request::Parts,
    body: Bytes,
}

impl Blueprint {
    /// A fresh request: `Parts` is not `Clone`, so one is rebuilt from the
    /// method, URI, version, and headers.
    fn request(&self) -> Request<Bytes> {
        let mut builder = Request::builder()
            .method(self.parts.method.clone())
            .uri(self.parts.uri.clone())
            .version(self.parts.version);
        for (name, value) in &self.parts.headers {
            builder = builder.header(name, value);
        }
        builder
            .body(self.body.clone())
            .unwrap_or_else(|_| Request::new(self.body.clone()))
    }
}

impl HttpClientExt for LimitedHttp {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + Send + 'static
    where
        T: Into<Bytes> + Send,
        U: From<Bytes> + Send + 'static,
    {
        let inner = self.inner.clone();
        // Read the body now: `T` need not outlive this call, and the gate
        // is chosen by the model it names.
        let (parts, body) = req.into_parts();
        let body: Bytes = body.into();
        let model = GateKey::model_of(&body);
        let permit = self.gates.permit(model.clone());
        let authorize = self.authorize.clone();
        let headers = self.headers.clone();
        let attempts = self.attempts.clone();
        let provider = self.provider_name();
        async move {
            let waited = Instant::now();
            let permit = permit.await.map_err(http_client::Error::instance)?;
            telemetry::provider_permit_wait(&provider, model.as_deref(), waited.elapsed());
            let blueprint = Blueprint { parts, body };
            let mut attempt = 0_u32;
            loop {
                let request =
                    Self::prepare(authorize.clone(), &headers, blueprint.request()).await?;
                let started = Instant::now();
                match inner.send(request).await {
                    Ok(response) => {
                        let status = response.status();
                        telemetry::provider_request(
                            &provider,
                            model.as_deref(),
                            status.as_str(),
                            started.elapsed(),
                        );
                        let again = attempts
                            .as_ref()
                            .and_then(|a| a.wait_for_status(attempt, status, response.headers()));
                        let Some(wait) = again else {
                            return Ok(body_holding(response, permit));
                        };
                        if let Some(a) = &attempts {
                            a.note(model.as_deref(), attempt, wait, &format!("HTTP {status}"));
                        }
                        tokio::time::sleep(wait).await;
                    }
                    Err(error) => {
                        telemetry::provider_request(
                            &provider,
                            model.as_deref(),
                            "error",
                            started.elapsed(),
                        );
                        let again = attempts
                            .as_ref()
                            .and_then(|a| a.wait_for_error(attempt, &error));
                        let Some(wait) = again else {
                            return Err(error);
                        };
                        if let Some(a) = &attempts {
                            a.note(model.as_deref(), attempt, wait, &error.to_string());
                        }
                        tokio::time::sleep(wait).await;
                    }
                }
                attempt = attempt.saturating_add(1);
            }
        }
    }

    fn send_multipart<U>(
        &self,
        mut req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + Send + 'static
    where
        U: From<Bytes> + Send + 'static,
    {
        let inner = self.inner.clone();
        let permit = self.gates.permit(None);
        let authorize = self.authorize.clone();
        Self::add_headers(&self.headers, req.headers_mut());
        async move {
            match authorize {
                // SigV4 signs the body, and a multipart body is not built yet.
                Some(Authorize::Sign(_)) => {
                    return Err(http_client::Error::Instance(
                        "multipart requests cannot be SigV4-signed".into(),
                    ));
                }
                Some(Authorize::Bearer(value)) => {
                    Authorize::replace_key(&value, req.headers_mut());
                }
                None => {}
            }
            let permit = permit.await.map_err(http_client::Error::instance)?;
            let response = inner.send_multipart(req).await?;
            Ok(body_holding(response, permit))
        }
    }

    /// A streamed request is tried again only until its head arrives:
    /// once the response status is good, the stream is the model's answer
    /// and a break in it ends the turn rather than restarting it.
    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + Send
    where
        T: Into<Bytes> + Send,
    {
        let inner = self.inner.clone();
        let (parts, body) = req.into_parts();
        let body: Bytes = body.into();
        let model = GateKey::model_of(&body);
        let permit = self.gates.permit(model.clone());
        let authorize = self.authorize.clone();
        let headers = self.headers.clone();
        let attempts = self.attempts.clone();
        let provider = self.provider_name();
        async move {
            let waited = Instant::now();
            let permit = permit.await.map_err(http_client::Error::instance)?;
            telemetry::provider_permit_wait(&provider, model.as_deref(), waited.elapsed());
            let blueprint = Blueprint { parts, body };
            let mut attempt = 0_u32;
            loop {
                let request =
                    Self::prepare(authorize.clone(), &headers, blueprint.request()).await?;
                let started = Instant::now();
                match inner.send_streaming(request).await {
                    Ok(response) => {
                        let status = response.status();
                        telemetry::provider_request(
                            &provider,
                            model.as_deref(),
                            status.as_str(),
                            started.elapsed(),
                        );
                        let again = attempts
                            .as_ref()
                            .and_then(|a| a.wait_for_status(attempt, status, response.headers()));
                        let Some(wait) = again else {
                            return Ok(response.map(|stream| -> BoxedStream {
                                Box::pin(Holding {
                                    inner: stream,
                                    permit,
                                })
                            }));
                        };
                        if let Some(a) = &attempts {
                            a.note(model.as_deref(), attempt, wait, &format!("HTTP {status}"));
                        }
                        tokio::time::sleep(wait).await;
                    }
                    Err(error) => {
                        telemetry::provider_request(
                            &provider,
                            model.as_deref(),
                            "error",
                            started.elapsed(),
                        );
                        let again = attempts
                            .as_ref()
                            .and_then(|a| a.wait_for_error(attempt, &error));
                        let Some(wait) = again else {
                            return Err(error);
                        };
                        if let Some(a) = &attempts {
                            a.note(model.as_deref(), attempt, wait, &error.to_string());
                        }
                        tokio::time::sleep(wait).await;
                    }
                }
                attempt = attempt.saturating_add(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::config::{BaseUrl, ProviderType, RequestLimit};
    use crate::error::Error;

    fn name(text: &str) -> ProviderName {
        text.parse().unwrap_or_else(|e: Error| fail(&e.to_string()))
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn provider(kind: ProviderType, limit: Option<u32>, url: &str) -> ProviderConfig {
        ProviderConfig {
            base_url: Some(
                BaseUrl::try_from(url.to_owned()).unwrap_or_else(|e| fail(&e.to_string())),
            ),
            max_concurrent_requests: limit.and_then(RequestLimit::new),
            ..ProviderConfig::new(kind)
        }
    }

    /// One HTTP server on loopback answering every request after `delay`,
    /// counting how many it serves at once.
    async fn slow_server(delay: Duration) -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let addr = listener
            .local_addr()
            .unwrap_or_else(|e| fail(&e.to_string()));
        let peak = Arc::new(AtomicUsize::new(0));
        let now = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&peak);
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let (now, peak) = (Arc::clone(&now), Arc::clone(&peak));
                tokio::spawn(async move {
                    let mut buf = [0_u8; 1024];
                    drop(socket.read(&mut buf).await);
                    let current = now.fetch_add(1, Ordering::SeqCst).saturating_add(1);
                    peak.fetch_max(current, Ordering::SeqCst);
                    tokio::time::sleep(delay).await;
                    now.fetch_sub(1, Ordering::SeqCst);
                    drop(
                        socket
                            .write_all(
                                b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok",
                            )
                            .await,
                    );
                });
            }
        });
        (format!("http://{addr}/"), seen)
    }

    /// Send `body` (JSON naming a model, or empty) as work outside any
    /// workspace and read the answer.
    async fn call(client: LimitedHttp, url: String, body: &'static str) -> Bytes {
        let request = Request::post(url)
            .body(Bytes::from_static(body.as_bytes()))
            .unwrap_or_else(|e| fail(&e.to_string()));
        // The gates read the scope when `send` is called, so call it inside.
        let sent = async { client.send::<_, Bytes>(request).await };
        let response = Egress::scope(Some(Egress::NoWorkspace), sent)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        response
            .into_body()
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn requests_to_one_model_never_exceed_its_limit() {
        use std::sync::atomic::Ordering;

        let (url, peak) = slow_server(Duration::from_millis(150)).await;
        let limited = LimitedHttp::for_provider(
            &name("limit-test"),
            &provider(ProviderType::Openai, Some(2), &url),
        );
        // Another client for the same provider shares the gate.
        let again = LimitedHttp::for_provider(
            &name("limit-test"),
            &provider(ProviderType::Openai, Some(2), &url),
        );
        let mut calls = Vec::new();
        for n in 0..6 {
            let client = if n % 2 == 0 {
                limited.clone()
            } else {
                again.clone()
            };
            calls.push(tokio::spawn(call(client, url.clone(), r#"{"model":"m"}"#)));
        }
        for call in calls {
            let body = call.await.unwrap_or_else(|e| fail(&e.to_string()));
            assert_eq!(&*body, b"ok");
        }
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        let gate = limited
            .gates
            .gate(Some(String::from("m")))
            .unwrap_or_else(|| fail("no gate"));
        assert_eq!(gate.available(), 2, "every permit came back");

        // Ollama defaults to one at a time, hosted APIs to eight.
        assert_eq!(
            provider(ProviderType::Ollama, None, &url)
                .request_limit()
                .get(),
            1
        );
        assert_eq!(
            provider(ProviderType::Anthropic, None, &url)
                .request_limit()
                .get(),
            8
        );
    }

    #[tokio::test]
    async fn a_request_the_scope_refuses_is_never_sent() {
        use std::sync::atomic::Ordering;

        use crate::storage::control::AllowedProviders;

        let (url, peak) = slow_server(Duration::ZERO).await;
        let send = |egress: Option<Egress>, kind: ProviderType, body: &'static str| {
            let client = LimitedHttp::for_provider(&name("hosted"), &provider(kind, None, &url));
            let request = Request::post(url.clone())
                .body(Bytes::from_static(body.as_bytes()))
                .unwrap_or_else(|e| fail(&e.to_string()));
            async move {
                let sent = async { client.send::<_, Bytes>(request).await };
                match Egress::scope(egress, sent).await {
                    Ok(_) => String::from("sent"),
                    Err(e) => e.to_string(),
                }
            }
        };
        let only = |names: &[&str]| {
            Some(Egress::Workspace(AllowedProviders::Only(
                names.iter().map(|n| (*n).to_owned()).collect(),
            )))
        };
        let model = r#"{"model":"m"}"#;
        // Work that entered no scope sends nothing.
        let unscoped = send(None, ProviderType::Openai, model).await;
        assert!(
            unscoped.contains("outside any workspace scope"),
            "{unscoped}"
        );
        // Nor does work on a workspace whose list leaves the provider out.
        let refused = send(only(&["ollama"]), ProviderType::Openai, model).await;
        assert!(
            refused.contains("provider 'hosted' is not allowed in this workspace"),
            "{refused}"
        );
        // An allowed Ollama server is still refused a model it serves from the cloud.
        let cloud = send(
            only(&["hosted"]),
            ProviderType::Ollama,
            r#"{"model":"big:120b-cloud"}"#,
        )
        .await;
        assert!(cloud.contains("model 'big:120b-cloud'"), "{cloud}");
        assert_eq!(peak.load(Ordering::SeqCst), 0, "nothing reached the server");
        assert_eq!(
            send(only(&["hosted"]), ProviderType::Openai, model).await,
            "sent"
        );
        assert_eq!(peak.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn each_model_on_a_provider_has_its_own_limit() {
        use std::sync::atomic::Ordering;

        let (url, peak) = slow_server(Duration::from_millis(200)).await;
        let client = LimitedHttp::for_provider(
            &name("per-model"),
            &provider(ProviderType::Ollama, None, &url),
        );
        // A chat model and an embedding model: one each at a time, both at
        // once.
        let chat = tokio::spawn(call(client.clone(), url.clone(), r#"{"model":"chat"}"#));
        let embed = tokio::spawn(call(client.clone(), url.clone(), r#"{"model":"embed"}"#));
        for handle in [chat, embed] {
            assert_eq!(
                &*handle.await.unwrap_or_else(|e| fail(&e.to_string())),
                b"ok"
            );
        }
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        assert_eq!(
            GateKey::model_of(br#"{"model":"x","input":["a"]}"#).as_deref(),
            Some("x")
        );
        assert_eq!(GateKey::model_of(b""), None);
    }

    #[tokio::test]
    async fn a_freed_permit_goes_to_interactive_waiters_first() {
        let gate = Gate::new(1);
        let held = gate.acquire(Priority::Background).await;
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut waiters = Vec::new();
        for (name, priority) in [
            ("background 1", Priority::Background),
            ("background 2", Priority::Background),
            ("interactive", Priority::Interactive),
        ] {
            let (gate, order) = (Arc::clone(&gate), Arc::clone(&order));
            waiters.push(tokio::spawn(async move {
                let permit = gate.acquire(priority).await;
                order
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(name);
                drop(permit);
            }));
            // Queue them in this order.
            tokio::task::yield_now().await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        drop(held);
        for waiter in waiters {
            assert!(waiter.await.is_ok());
        }
        assert_eq!(
            *order.lock().unwrap_or_else(PoisonError::into_inner),
            vec!["interactive", "background 1", "background 2"]
        );
        assert_eq!(gate.available(), 1);

        // A waiter that gives up is skipped, not handed the permit forever.
        let held = gate.acquire(Priority::Background).await;
        let gave_up = tokio::spawn({
            let gate = Arc::clone(&gate);
            async move { gate.acquire(Priority::Interactive).await }
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        gave_up.abort();
        drop(gave_up.await);
        drop(held);
        assert_eq!(gate.available(), 1);
    }

    #[tokio::test]
    async fn an_oauth_bearer_replaces_the_api_key_header() {
        let client = LimitedHttp::for_provider(
            &name("p"),
            &provider(ProviderType::Anthropic, None, "http://127.0.0.1:9/"),
        )
        .with_oauth_bearer("tok-1")
        .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(!format!("{client:?}").contains("tok-1"));
        let request = Request::post("http://127.0.0.1:9/v1/messages")
            .header("x-api-key", "tok-1")
            .header("anthropic-version", "2023-06-01")
            .body(Bytes::from_static(b"{}"))
            .unwrap_or_else(|e| fail(&e.to_string()));
        let sent = LimitedHttp::prepare(client.authorize, &client.headers, request)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let headers = sent.headers();
        assert_eq!(
            headers
                .get(http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok()),
            Some("Bearer tok-1")
        );
        assert!(headers.get("x-api-key").is_none());
        assert!(headers.get("anthropic-version").is_some());
        assert_eq!(sent.body().as_ref(), b"{}");
    }

    /// One HTTP server on loopback answering each request with the next
    /// status in `statuses` (the last one from then on), with
    /// `Retry-After: 0` on a 429, counting the requests it served.
    async fn status_server(statuses: Vec<u16>) -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let addr = listener
            .local_addr()
            .unwrap_or_else(|e| fail(&e.to_string()));
        let served = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&served);
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let n = served.fetch_add(1, Ordering::SeqCst);
                let status = statuses
                    .get(n)
                    .or_else(|| statuses.last())
                    .copied()
                    .unwrap_or(200);
                let mut buf = [0_u8; 2048];
                drop(socket.read(&mut buf).await);
                let retry_after = if status == 429 {
                    "retry-after: 0\r\n"
                } else {
                    ""
                };
                let response = format!(
                    "HTTP/1.1 {status} X\r\ncontent-length: 2\r\n{retry_after}connection: close\r\n\r\nok"
                );
                drop(socket.write_all(response.as_bytes()).await);
            }
        });
        (format!("http://{addr}/"), seen)
    }

    /// A client of `url` that retries `retries` times with a short backoff.
    fn retrying(url: &str, retries: u32) -> LimitedHttp {
        LimitedHttp::for_provider(
            &name("retry-test"),
            &ProviderConfig {
                retry: RetryPolicy {
                    max_retries: retries,
                    backoff: Duration::from_millis(5),
                },
                ..provider(ProviderType::Openai, Some(2), url)
            },
        )
    }

    /// The status of `client`'s answer to a POST naming a model.
    async fn status_of(client: LimitedHttp, url: String) -> Result<u16, String> {
        let request = Request::post(url)
            .body(Bytes::from_static(br#"{"model":"m"}"#))
            .unwrap_or_else(|e| fail(&e.to_string()));
        let sent = async { client.send::<_, Bytes>(request).await };
        Egress::scope(Some(Egress::NoWorkspace), sent)
            .await
            .map(|r| r.status().as_u16())
            .map_err(|e| e.to_string())
    }

    /// A throttled or failing request is sent again until it succeeds or
    /// the retries run out; a client error stands at once; no retries
    /// means one attempt.
    #[tokio::test(flavor = "multi_thread")]
    async fn throttled_and_failing_requests_are_retried_within_the_policy() {
        use std::sync::atomic::Ordering;

        let (url, served) = status_server(vec![429, 503, 200]).await;
        assert_eq!(status_of(retrying(&url, 3), url.clone()).await, Ok(200));
        assert_eq!(served.load(Ordering::SeqCst), 3);

        // rig's client turns a failing status into an error; the last
        // attempt's error is what the caller gets.
        let (url, served) = status_server(vec![500, 500, 500, 500]).await;
        let failed = status_of(retrying(&url, 2), url.clone()).await;
        assert!(
            failed.as_ref().is_err_and(|e| e.contains("500")),
            "{failed:?}"
        );
        assert_eq!(served.load(Ordering::SeqCst), 3, "one try and two retries");

        let (url, served) = status_server(vec![400, 200]).await;
        let refused = status_of(retrying(&url, 3), url.clone()).await;
        assert!(
            refused.as_ref().is_err_and(|e| e.contains("400")),
            "{refused:?}"
        );
        assert_eq!(
            served.load(Ordering::SeqCst),
            1,
            "a client error is not retried"
        );

        let (url, served) = status_server(vec![429, 200]).await;
        let once = status_of(retrying(&url, 0), url.clone()).await;
        assert!(once.as_ref().is_err_and(|e| e.contains("429")), "{once:?}");
        assert_eq!(
            served.load(Ordering::SeqCst),
            1,
            "no retries means one attempt"
        );

        // A streamed request is retried the same way until its head arrives.
        let (url, served) = status_server(vec![503, 200]).await;
        let client = retrying(&url, 1);
        let request = Request::post(url)
            .body(Bytes::from_static(br#"{"model":"m"}"#))
            .unwrap_or_else(|e| fail(&e.to_string()));
        let streamed = Egress::scope(Some(Egress::NoWorkspace), async {
            client.send_streaming(request).await
        })
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(streamed.status().as_u16(), 200);
        assert_eq!(served.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn the_wait_doubles_with_jitter_and_stops_at_a_minute() {
        let policy = RetryPolicy {
            max_retries: 3,
            backoff: Duration::from_millis(500),
        };
        assert_eq!(policy.wait(1, 0.0), Duration::from_millis(500));
        assert_eq!(policy.wait(2, 0.0), Duration::from_millis(1000));
        assert_eq!(policy.wait(3, 1.0), Duration::from_millis(2500));
        assert_eq!(policy.wait(20, 0.0), RetryPolicy::MAX_WAIT);
        assert_eq!(RetryPolicy::none().wait(1, 0.5), Duration::ZERO);
        let attempts = Attempts::new(&name("p"), policy);
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::RETRY_AFTER,
            http::HeaderValue::from_static("7"),
        );
        assert_eq!(
            attempts.wait_for_status(0, http::StatusCode::TOO_MANY_REQUESTS, &headers),
            Some(Duration::from_secs(7))
        );
        assert_eq!(
            attempts.wait_for_status(3, http::StatusCode::TOO_MANY_REQUESTS, &headers),
            None
        );
        assert_eq!(
            attempts.wait_for_status(0, http::StatusCode::BAD_REQUEST, &headers),
            None
        );
    }

    #[test]
    fn a_token_that_is_no_header_value_is_refused() {
        let client = LimitedHttp::default().with_oauth_bearer("tok\n1");
        assert!(client.is_err_and(|e| e.to_string().contains("not a header value")));
    }
}
