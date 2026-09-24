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

use bytes::Bytes;
use futures::Stream;
use rig::http_client::sse::BoxedStream;
use rig::http_client::{
    self, HttpClientExt, LazyBody, MultipartForm, Request, Response, StreamingResponse,
};
use tokio::sync::oneshot;

use crate::config::{BaseUrl, ProviderConfig, ProviderName, RequestLimit};

use crate::priority::Priority;

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
/// built for it. `Default` is unlimited.
#[derive(Clone, Default)]
pub(crate) struct ProviderGates {
    /// The provider and its limit, or `None` for the unlimited default.
    provider: Option<(ProviderKey, RequestLimit)>,
}

impl std::fmt::Debug for ProviderGates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderGates")
            .field("limit", &self.provider.as_ref().map(|(_, limit)| *limit))
            .finish_non_exhaustive()
    }
}

impl ProviderGates {
    /// The gates of provider `name`.
    pub(crate) fn for_provider(name: &ProviderName, provider: &ProviderConfig) -> Self {
        Self {
            provider: Some((
                ProviderKey {
                    name: name.clone(),
                    base_url: provider.base_url.clone(),
                },
                provider.request_limit(),
            )),
        }
    }

    /// Wait for a permit of `model`'s gate at the calling task's priority
    /// (read now, not when the future first runs); `None` when unlimited.
    pub(crate) fn permit(
        &self,
        model: Option<String>,
    ) -> impl Future<Output = Option<GatePermit>> + Send + 'static {
        let gate = self.gate(model);
        let priority = Priority::current();
        async move {
            match gate {
                Some(gate) => Some(gate.acquire(priority).await),
                None => None,
            }
        }
    }

    /// The gate for `model`, created with this client's limit on first use;
    /// a later config for the same provider in one process keeps the first.
    fn gate(&self, model: Option<String>) -> Option<Arc<Gate>> {
        let (provider, limit) = self.provider.as_ref()?;
        let key = GateKey {
            provider: provider.clone(),
            model,
        };
        let mut map = GateKey::registry()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        Some(Arc::clone(map.entry(key).or_insert_with(|| {
            Gate::new(usize::try_from(limit.get()).unwrap_or(1))
        })))
    }
}

/// A reqwest client that takes a permit of its provider's limit for the
/// request's model before each request. `Default` (required by rig's
/// provider bounds) is unlimited; quack always builds one with
/// [`LimitedHttp::for_provider`].
#[derive(Clone, Default, Debug)]
pub struct LimitedHttp {
    inner: reqwest::Client,
    gates: ProviderGates,
}

impl LimitedHttp {
    /// The client for provider `name`, sharing its process-wide gates.
    #[must_use]
    pub fn for_provider(name: &ProviderName, provider: &ProviderConfig) -> Self {
        Self {
            inner: reqwest::Client::default(),
            gates: ProviderGates::for_provider(name, provider),
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
        let permit = self.gates.permit(GateKey::model_of(&body));
        async move {
            let permit = permit.await;
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
        let permit = self.gates.permit(None);
        async move {
            let permit = permit.await;
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
        let permit = self.gates.permit(GateKey::model_of(&body));
        async move {
            let permit = permit.await;
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
            assert_eq!(&body[..], b"ok");
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
                &handle.await.unwrap_or_else(|e| fail(&e.to_string()))[..],
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
}
