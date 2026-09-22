//! Model requests are the scarce resource (design doc 4.1), so they are what
//! is limited: every rig client quack builds sends through a
//! [`LimitedHttp`], which takes one permit of its provider's semaphore
//! (`[providers.NAME].max_concurrent_requests`) per request and holds it
//! until the response body has been read or its stream has ended.
//!
//! The limit is per provider and process-wide: every client built for the
//! provider, however often (OAuth clients are rebuilt per call), shares one
//! semaphore. A turn waiting on the user's answer to a permission prompt, or
//! running a tool, holds nothing. rig's streaming loop drains a model
//! response before it runs the tool calls in it, so a tool that calls the
//! same provider (the query embedding, the model reranker) never waits on a
//! permit its own turn still holds.

use std::collections::HashMap;
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
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::config::ProviderConfig;

/// The semaphores, by provider name and base URL.
fn registry() -> &'static Mutex<HashMap<String, Arc<Semaphore>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<Semaphore>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The provider's semaphore, created with its limit on first use; a later
/// config for the same provider in one process keeps the first limit.
fn semaphore_for(name: &str, provider: &ProviderConfig) -> Arc<Semaphore> {
    let key = format!("{name}\u{0}{}", provider.base_url.as_deref().unwrap_or(""));
    let mut map = registry().lock().unwrap_or_else(PoisonError::into_inner);
    Arc::clone(map.entry(key).or_insert_with(|| {
        Arc::new(Semaphore::new(
            usize::try_from(provider.request_limit()).unwrap_or(1),
        ))
    }))
}

/// A reqwest client that waits for a permit of its provider's limit before
/// each request. `Default` (required by rig's provider bounds) is unlimited;
/// quack always builds one with [`LimitedHttp::for_provider`].
#[derive(Clone)]
pub struct LimitedHttp {
    inner: reqwest::Client,
    permits: Arc<Semaphore>,
}

impl std::fmt::Debug for LimitedHttp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LimitedHttp")
            .field("available", &self.permits.available_permits())
            .finish_non_exhaustive()
    }
}

impl Default for LimitedHttp {
    fn default() -> Self {
        Self {
            inner: reqwest::Client::default(),
            permits: Arc::new(Semaphore::new(Semaphore::MAX_PERMITS)),
        }
    }
}

impl LimitedHttp {
    /// The client for provider `name`, sharing its process-wide limit.
    #[must_use]
    pub fn for_provider(name: &str, provider: &ProviderConfig) -> Self {
        Self {
            inner: reqwest::Client::default(),
            permits: semaphore_for(name, provider),
        }
    }

    /// Permits free right now (for tests and diagnostics).
    #[must_use]
    pub fn available(&self) -> usize {
        self.permits.available_permits()
    }

    async fn permit(permits: Arc<Semaphore>) -> http_client::Result<OwnedSemaphorePermit> {
        permits
            .acquire_owned()
            .await
            .map_err(|e| http_client::Error::Instance(Box::new(e)))
    }
}

/// Hold `permit` until the lazily read body has been read (or dropped).
fn body_holding<U: Send + 'static>(
    response: Response<LazyBody<U>>,
    permit: OwnedSemaphorePermit,
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
/// when it is dropped, so a finished response never holds the provider.
struct Holding {
    inner: BoxedStream,
    permit: Option<OwnedSemaphorePermit>,
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
        let permits = Arc::clone(&self.permits);
        let inner = self.inner.clone();
        // Build the request before waiting: `T` need not outlive this call.
        let (parts, body) = req.into_parts();
        let body: Bytes = body.into();
        async move {
            let permit = Self::permit(permits).await?;
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
        let permits = Arc::clone(&self.permits);
        let inner = self.inner.clone();
        async move {
            let permit = Self::permit(permits).await?;
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
        let permits = Arc::clone(&self.permits);
        let inner = self.inner.clone();
        async move {
            let permit = Self::permit(permits).await?;
            let response = inner.send_streaming(req).await?;
            Ok(response.map(|stream| -> BoxedStream {
                Box::pin(Holding {
                    inner: stream,
                    permit: Some(permit),
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

    #[tokio::test(flavor = "multi_thread")]
    async fn requests_to_one_provider_never_exceed_its_limit() {
        use std::sync::atomic::Ordering;

        let (url, peak) = slow_server(Duration::from_millis(150)).await;
        let limited =
            LimitedHttp::for_provider("limit-test", &provider(ProviderType::Openai, Some(2), &url));
        // Another client for the same provider shares the limit.
        let again =
            LimitedHttp::for_provider("limit-test", &provider(ProviderType::Openai, Some(2), &url));
        let mut calls = Vec::new();
        for n in 0..6 {
            let client = if n % 2 == 0 {
                limited.clone()
            } else {
                again.clone()
            };
            let url = url.clone();
            calls.push(tokio::spawn(async move {
                let request = Request::get(url)
                    .body(Bytes::new())
                    .unwrap_or_else(|e| fail(&e.to_string()));
                let response = client
                    .send::<_, Bytes>(request)
                    .await
                    .unwrap_or_else(|e| fail(&e.to_string()));
                response
                    .into_body()
                    .await
                    .unwrap_or_else(|e| fail(&e.to_string()))
            }));
        }
        for call in calls {
            let body = call.await.unwrap_or_else(|e| fail(&e.to_string()));
            assert_eq!(&body[..], b"ok");
        }
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        assert_eq!(limited.available(), 2, "every permit came back");

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
}
