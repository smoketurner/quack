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
//! whole ingest's embedding batches or an extraction's next chunk. A turn
//! (`run_turn`) and a query embedding run interactive; everything else is
//! background.
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

use bytes::Bytes;
use futures::Stream;
use rig::http_client::sse::BoxedStream;
use rig::http_client::{
    self, HttpClientExt, LazyBody, MultipartForm, Request, Response, StreamingResponse,
};
use tokio::sync::oneshot;

use crate::config::ProviderConfig;

/// Who is waiting for a model: a person, or background work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    /// A turn or a lookup someone is watching; served first.
    Interactive,
    /// Ingest embeddings, extraction, proposals.
    Background,
}

tokio::task_local! {
    static PRIORITY: Priority;
}

/// Run `work` with its model requests at `priority` (for everything it
/// awaits on this task).
pub async fn with_priority<F: Future>(priority: Priority, work: F) -> F::Output {
    PRIORITY.scope(priority, work).await
}

/// The priority of the task making a request: background unless scoped.
#[must_use]
pub fn current_priority() -> Priority {
    PRIORITY.try_with(|p| *p).unwrap_or(Priority::Background)
}

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
                Err(mut unclaimed) => unclaimed.armed = false,
            }
        }
    }

    #[cfg(test)]
    fn available(&self) -> usize {
        self.state().available
    }
}

/// One held permit; dropping it passes it on.
struct GatePermit {
    gate: Arc<Gate>,
    armed: bool,
}

impl GatePermit {
    fn new(gate: &Arc<Gate>) -> Self {
        Self {
            gate: Arc::clone(gate),
            armed: true,
        }
    }
}

impl Drop for GatePermit {
    fn drop(&mut self) {
        if self.armed {
            self.gate.release();
        }
    }
}

/// The gates, by provider name, base URL, and model.
fn registry() -> &'static Mutex<HashMap<String, Arc<Gate>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<Gate>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The model a request names in its JSON body (`""` when it names none, as
/// Ollama's `GET api/ps` does).
fn model_of(body: &[u8]) -> String {
    #[derive(serde::Deserialize)]
    struct Named {
        model: Option<String>,
    }
    serde_json::from_slice::<Named>(body)
        .ok()
        .and_then(|n| n.model)
        .unwrap_or_default()
}

/// A reqwest client that takes a permit of its provider's limit for the
/// request's model before each request. `Default` (required by rig's
/// provider bounds) is unlimited; quack always builds one with
/// [`LimitedHttp::for_provider`].
#[derive(Clone)]
pub struct LimitedHttp {
    inner: reqwest::Client,
    /// `name\0base_url`, or `None` for the unlimited default.
    provider: Option<Arc<str>>,
    limit: usize,
}

impl std::fmt::Debug for LimitedHttp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LimitedHttp")
            .field("limit", &self.limit)
            .finish_non_exhaustive()
    }
}

impl Default for LimitedHttp {
    fn default() -> Self {
        Self {
            inner: reqwest::Client::default(),
            provider: None,
            limit: usize::MAX,
        }
    }
}

impl LimitedHttp {
    /// The client for provider `name`, sharing its process-wide gates.
    #[must_use]
    pub fn for_provider(name: &str, provider: &ProviderConfig) -> Self {
        Self {
            inner: reqwest::Client::default(),
            provider: Some(Arc::from(format!(
                "{name}\u{0}{}",
                provider.base_url.as_deref().unwrap_or("")
            ))),
            limit: usize::try_from(provider.request_limit()).unwrap_or(1),
        }
    }

    /// The gate for `model`, created with this client's limit on first use;
    /// a later config for the same provider in one process keeps the first.
    fn gate(&self, model: &str) -> Option<Arc<Gate>> {
        let provider = self.provider.as_deref()?;
        let key = format!("{provider}\u{0}{model}");
        let mut map = registry().lock().unwrap_or_else(PoisonError::into_inner);
        Some(Arc::clone(
            map.entry(key).or_insert_with(|| Gate::new(self.limit)),
        ))
    }

    /// Wait for a permit for `model` at the calling task's priority.
    async fn permit(gate: Option<Arc<Gate>>, priority: Priority) -> Option<GatePermit> {
        match gate {
            Some(gate) => Some(gate.acquire(priority).await),
            None => None,
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
        let gate = self.gate(&model_of(&body));
        let priority = current_priority();
        async move {
            let permit = Self::permit(gate, priority).await;
            let response = inner.send(Request::from_parts(parts, body)).await?;
            Ok(body_holding(response, permit))
        }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + Send + 'static
    where
        U: From<Bytes> + Send + 'static,
    {
        let inner = self.inner.clone();
        let gate = self.gate("");
        let priority = current_priority();
        async move {
            let permit = Self::permit(gate, priority).await;
            let response = inner.send_multipart(req).await?;
            Ok(body_holding(response, permit))
        }
    }

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
        let gate = self.gate(&model_of(&body));
        let priority = current_priority();
        async move {
            let permit = Self::permit(gate, priority).await;
            let response = inner
                .send_streaming(Request::from_parts(parts, body))
                .await?;
            Ok(response.map(|stream| -> BoxedStream {
                Box::pin(Holding {
                    inner: stream,
                    permit,
                })
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::config::{AuthMode, ProviderType};

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn provider(kind: ProviderType, limit: Option<u32>, url: &str) -> ProviderConfig {
        ProviderConfig {
            provider_type: kind,
            auth: AuthMode::default(),
            base_url: Some(url.to_owned()),
            api_key_env: None,
            embedding_dimension: None,
            max_concurrent_requests: limit,
            oauth: None,
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

    /// Send `body` (JSON naming a model, or empty) and read the answer.
    async fn call(client: LimitedHttp, url: String, body: &'static str) -> Bytes {
        let request = Request::post(url)
            .body(Bytes::from_static(body.as_bytes()))
            .unwrap_or_else(|e| fail(&e.to_string()));
        let response = client
            .send::<_, Bytes>(request)
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
        let limited =
            LimitedHttp::for_provider("limit-test", &provider(ProviderType::Openai, Some(2), &url));
        // Another client for the same provider shares the gate.
        let again =
            LimitedHttp::for_provider("limit-test", &provider(ProviderType::Openai, Some(2), &url));
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
            assert_eq!(&body[..], b"ok");
        }
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        let gate = limited.gate("m").unwrap_or_else(|| fail("no gate"));
        assert_eq!(gate.available(), 2, "every permit came back");

        // Ollama defaults to one at a time, hosted APIs to eight.
        assert_eq!(
            provider(ProviderType::Ollama, None, &url).request_limit(),
            1
        );
        assert_eq!(
            provider(ProviderType::Anthropic, None, &url).request_limit(),
            8
        );
        assert_eq!(
            provider(ProviderType::Ollama, Some(0), &url).request_limit(),
            1
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn each_model_on_a_provider_has_its_own_limit() {
        use std::sync::atomic::Ordering;

        let (url, peak) = slow_server(Duration::from_millis(200)).await;
        let client =
            LimitedHttp::for_provider("per-model", &provider(ProviderType::Ollama, None, &url));
        // A chat model and an embedding model: one each at a time, both at
        // once.
        let chat = tokio::spawn(call(client.clone(), url.clone(), r#"{"model":"chat"}"#));
        let embed = tokio::spawn(call(client.clone(), url.clone(), r#"{"model":"embed"}"#));
        for handle in [chat, embed] {
            assert_eq!(
                &handle.await.unwrap_or_else(|e| fail(&e.to_string()))[..],
                b"ok"
            );
        }
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        assert_eq!(model_of(br#"{"model":"x","input":["a"]}"#), "x");
        assert_eq!(model_of(b""), "");
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

        // The priority follows the task that scoped it.
        assert_eq!(current_priority(), Priority::Background);
        let inside = with_priority(Priority::Interactive, async { current_priority() }).await;
        assert_eq!(inside, Priority::Interactive);
    }
}
