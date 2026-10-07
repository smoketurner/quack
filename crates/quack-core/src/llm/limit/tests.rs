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
        base_url: Some(BaseUrl::try_from(url.to_owned()).unwrap_or_else(|e| fail(&e.to_string()))),
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

/// A request backing off between retries holds no permit: another request
/// for the same one-permit gate is sent during the wait instead of queuing
/// behind it.
#[tokio::test(flavor = "multi_thread")]
async fn a_request_waits_out_its_backoff_without_the_permit() {
    let (url, _) = status_server(vec![503, 200]).await;
    let client = || {
        LimitedHttp::for_provider(
            &name("backoff-test"),
            &ProviderConfig {
                retry: RetryPolicy {
                    max_retries: 1,
                    backoff: Duration::from_millis(1_500),
                },
                ..provider(ProviderType::Openai, Some(1), &url)
            },
        )
    };
    let throttled = tokio::spawn(status_of(client(), url.clone()));
    tokio::time::sleep(Duration::from_millis(200)).await;
    let started = Instant::now();
    assert_eq!(status_of(client(), url.clone()).await, Ok(200));
    assert!(
        started.elapsed() < Duration::from_millis(1_000),
        "the second request waited {:?} behind the first one's backoff",
        started.elapsed()
    );
    assert_eq!(
        throttled.await.unwrap_or_else(|e| fail(&e.to_string())),
        Ok(200)
    );
}
