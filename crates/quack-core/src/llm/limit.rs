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
        Some(named.unwrap_or_else(|| self.policy.next_wait(attempt.saturating_add(1))))
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
        Some(self.policy.next_wait(attempt.saturating_add(1)))
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
mod tests;
