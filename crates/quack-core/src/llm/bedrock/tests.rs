use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use aws_smithy_runtime_api::client::orchestrator::HttpResponse;

use super::*;
use crate::config::{ProviderType, RequestLimit};
use crate::llm::egress::Egress;
use crate::proxy::{Environment, Variable};
use crate::storage::control::AllowedProviders;
use aws_types::service_config::{LoadServiceConfig, ServiceConfigKey};

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

/// An endpoint that answers every request after `delay` with a
/// streamed body, counting how many requests it holds at once.
#[derive(Debug, Default)]
struct SlowEndpoint {
    now: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

impl HttpConnector for SlowEndpoint {
    fn call(&self, _request: HttpRequest) -> HttpConnectorFuture {
        let (now, peak) = (Arc::clone(&self.now), Arc::clone(&self.peak));
        HttpConnectorFuture::new(async move {
            let current = now.fetch_add(1, Ordering::SeqCst).saturating_add(1);
            peak.fetch_max(current, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(100)).await;
            now.fetch_sub(1, Ordering::SeqCst);
            Ok(HttpResponse::new(
                200_u16.try_into().unwrap_or_else(|_| fail("status")),
                SdkBody::from("ok"),
            ))
        })
    }
}

fn request(uri: &str) -> HttpRequest {
    let request = http::Request::builder()
        .uri(uri)
        .body(SdkBody::empty())
        .unwrap_or_else(|e| fail(&e.to_string()));
    HttpRequest::try_from(request).unwrap_or_else(|e| fail(&e.to_string()))
}

/// Read a response body to its end, as the SDK does.
async fn drain(mut body: SdkBody) -> Vec<u8> {
    let mut read = Vec::new();
    while let Some(frame) = std::future::poll_fn(|cx| Pin::new(&mut body).poll_frame(cx)).await {
        if let Ok(data) = frame.unwrap_or_else(|e| fail(&e.to_string())).into_data() {
            read.extend_from_slice(&data);
        }
    }
    read
}

async fn peak_of(
    connector: &Arc<LimitedConnector>,
    endpoint: &Arc<AtomicUsize>,
    uri: &str,
) -> usize {
    endpoint.store(0, Ordering::SeqCst);
    let calls: Vec<_> = (0..5)
        .map(|_| {
            let (connector, uri) = (Arc::clone(connector), uri.to_owned());
            tokio::spawn(Egress::scope(Some(Egress::NoWorkspace), async move {
                let response = connector
                    .call(request(&uri))
                    .await
                    .unwrap_or_else(|e| fail(&format!("{e:?}")));
                drain(response.into_body()).await
            }))
        })
        .collect();
    for call in calls {
        assert_eq!(call.await.unwrap_or_else(|e| fail(&e.to_string())), b"ok");
    }
    endpoint.load(Ordering::SeqCst)
}

#[tokio::test(flavor = "multi_thread")]
async fn model_calls_wait_for_the_limit_and_credential_calls_do_not() {
    let endpoint = SlowEndpoint::default();
    let peak = Arc::clone(&endpoint.peak);
    let provider = ProviderConfig {
        max_concurrent_requests: RequestLimit::new(2),
        ..ProviderConfig::new(ProviderType::Bedrock)
    };
    let name: ProviderName = "bedrock-limit-test"
        .parse()
        .unwrap_or_else(|e: Error| fail(&e.to_string()));
    let connector = Arc::new(LimitedConnector {
        inner: SharedHttpConnector::new(endpoint),
        gates: ProviderGates::for_provider(&name, &provider),
    });
    let model = "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse";
    assert_eq!(peak_of(&connector, &peak, model).await, 2);
    // Every permit came back: the same calls again see the same limit.
    assert_eq!(peak_of(&connector, &peak, model).await, 2);
    let sso = "https://portal.sso.us-east-1.amazonaws.com/federation/credentials";
    assert_eq!(peak_of(&connector, &peak, sso).await, 5);
}

#[tokio::test]
async fn a_model_call_the_scope_refuses_never_reaches_the_endpoint() {
    let endpoint = SlowEndpoint::default();
    let peak = Arc::clone(&endpoint.peak);
    let name: ProviderName = "bedrock-egress-test"
        .parse()
        .unwrap_or_else(|e: Error| fail(&e.to_string()));
    let connector = LimitedConnector {
        inner: SharedHttpConnector::new(endpoint),
        gates: ProviderGates::for_provider(&name, &ProviderConfig::new(ProviderType::Bedrock)),
    };
    let model = "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse";
    let call = |egress: Option<Egress>| {
        let call = async { connector.call(request(model)).await };
        async move {
            match Egress::scope(egress, call).await {
                Ok(_) => String::from("sent"),
                Err(e) => DisplayErrorContext(&e).to_string(),
            }
        }
    };
    let elsewhere = AllowedProviders::Only([String::from("ollama")].into());
    let refused = call(Some(Egress::Workspace(elsewhere))).await;
    assert!(
        refused.contains("provider 'bedrock-egress-test' is not allowed"),
        "{refused}"
    );
    let unscoped = call(None).await;
    assert!(
        unscoped.contains("outside any workspace scope"),
        "{unscoped}"
    );
    assert_eq!(
        peak.load(Ordering::SeqCst),
        0,
        "nothing reached the endpoint"
    );
    assert_eq!(call(Some(Egress::NoWorkspace)).await, "sent");
}

fn static_signer(service: &'static str) -> Signer {
    Signer::new(
        SharedCredentialsProvider::new(Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            None,
            "test",
        )),
        String::from("eu-west-1"),
        service,
    )
}

#[tokio::test]
async fn requests_are_signed_for_the_endpoint_service_and_region() {
    let request = http::Request::post("https://bedrock-mantle.eu-west-1.api.aws/v1/responses")
        .header(http::header::AUTHORIZATION, "Bearer sigv4")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Bytes::from_static(br#"{"model":"openai.gpt-oss-120b"}"#))
        .unwrap_or_else(|e| fail(&e.to_string()));
    let signed = static_signer("bedrock-mantle")
        .sign(request)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let header = |name: &str| {
        signed
            .headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned()
    };
    let authorization = header("authorization");
    assert!(
        authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"),
        "{authorization}"
    );
    assert!(
        authorization.contains("/eu-west-1/bedrock-mantle/aws4_request"),
        "{authorization}"
    );
    assert!(
        authorization.contains("SignedHeaders=content-type;host;x-amz-date"),
        "{authorization}"
    );
    assert!(!authorization.contains("Bearer"));
    assert!(!header("x-amz-date").is_empty());
    // The body is what was signed, unchanged.
    assert_eq!(
        signed.body().as_ref(),
        br#"{"model":"openai.gpt-oss-120b"}"#
    );
}

fn provider_with(endpoint: BedrockEndpoint, base_url: Option<&str>) -> ProviderConfig {
    let provider_type = match endpoint {
        BedrockEndpoint::Runtime => ProviderType::Bedrock,
        BedrockEndpoint::Mantle => ProviderType::BedrockMantle,
    };
    ProviderConfig {
        base_url: base_url
            .map(|u| BaseUrl::try_from(u.to_owned()).unwrap_or_else(|e| fail(&e.to_string()))),
        ..ProviderConfig::new(provider_type)
    }
}

async fn root_of(endpoint: BedrockEndpoint, fips: bool, base_url: Option<&str>) -> Result<String> {
    let provider = provider_with(endpoint, base_url);
    let sdk = SdkConfig::builder()
        .region(Region::new("us-west-2"))
        .use_fips(fips)
        .behavior_version(BehaviorVersion::latest())
        .build();
    let name: ProviderName = "root-test"
        .parse()
        .unwrap_or_else(|e: Error| fail(&e.to_string()));
    root(&name, &provider, endpoint, &sdk, "us-west-2").await
}

#[tokio::test]
async fn each_endpoint_resolves_its_root_fips_included() {
    assert_eq!(
        root_of(BedrockEndpoint::Runtime, false, None)
            .await
            .ok()
            .as_deref(),
        Some("https://bedrock-runtime.us-west-2.amazonaws.com")
    );
    assert_eq!(
        root_of(BedrockEndpoint::Runtime, true, None)
            .await
            .ok()
            .as_deref(),
        Some("https://bedrock-runtime-fips.us-west-2.amazonaws.com")
    );
    assert_eq!(
        root_of(BedrockEndpoint::Mantle, false, None)
            .await
            .ok()
            .as_deref(),
        Some("https://bedrock-mantle.us-west-2.api.aws")
    );
    let mantle_fips = root_of(BedrockEndpoint::Mantle, true, None).await;
    assert!(mantle_fips.is_err_and(|e| e.to_string().contains("bedrock-mantle has none")));
    // A VPC endpoint is the root, as given.
    let vpce = "https://vpce-0abc.bedrock-mantle.us-west-2.vpce.amazonaws.com/";
    assert_eq!(
        root_of(BedrockEndpoint::Mantle, false, Some(vpce))
            .await
            .ok()
            .as_deref(),
        Some("https://vpce-0abc.bedrock-mantle.us-west-2.vpce.amazonaws.com")
    );
    let plain = "https://vpce-0abc.bedrock-runtime.us-west-2.vpce.amazonaws.com";
    let refused = root_of(BedrockEndpoint::Runtime, true, Some(plain)).await;
    assert!(refused.is_err_and(|e| e.to_string().contains("not one")));
    let fips_vpce = "https://vpce-0abc.bedrock-runtime-fips.us-west-2.vpce.amazonaws.com";
    assert!(
        root_of(BedrockEndpoint::Runtime, true, Some(fips_vpce))
            .await
            .is_ok()
    );
}

/// The `[services]` block the SDK would load, faked for the test: returns
/// its value for the Bedrock Runtime `endpoint_url` key and nothing else.
#[derive(Debug)]
struct BedrockRuntimeServiceConfig(Option<String>);

impl LoadServiceConfig for BedrockRuntimeServiceConfig {
    fn load_config(&self, key: ServiceConfigKey<'_>) -> Option<String> {
        if key.service_id() == "Bedrock Runtime" && key.env() == "AWS_ENDPOINT_URL" {
            self.0.clone()
        } else {
            None
        }
    }
}

#[test]
fn runtime_endpoint_override_is_none_with_nothing_set() {
    let sdk = SdkConfig::builder()
        .region(Region::new("us-west-2"))
        .behavior_version(BehaviorVersion::latest())
        .build();
    assert_eq!(runtime_endpoint_override(&sdk), None);
}

#[test]
fn runtime_endpoint_override_reads_the_service_specific_value() {
    let sdk = SdkConfig::builder()
        .region(Region::new("us-west-2"))
        .service_config(BedrockRuntimeServiceConfig(Some(String::from(
            "https://runtime-override.example/",
        ))))
        .behavior_version(BehaviorVersion::latest())
        .build();
    assert_eq!(
        runtime_endpoint_override(&sdk).as_deref(),
        Some("https://runtime-override.example/")
    );
}

#[test]
fn runtime_endpoint_override_falls_back_to_the_global_value() {
    // No service-specific value and no programmatic origin: the global one
    // is the fallback, the path `AWS_ENDPOINT_URL` takes through the loader.
    let sdk = SdkConfig::builder()
        .region(Region::new("us-west-2"))
        .endpoint_url("https://global-override.example")
        .behavior_version(BehaviorVersion::latest())
        .build();
    assert_eq!(
        runtime_endpoint_override(&sdk).as_deref(),
        Some("https://global-override.example")
    );
}

#[test]
fn runtime_endpoint_override_service_specific_wins_over_global() {
    let sdk = SdkConfig::builder()
        .region(Region::new("us-west-2"))
        .endpoint_url("https://global-override.example")
        .service_config(BedrockRuntimeServiceConfig(Some(String::from(
            "https://runtime-override.example",
        ))))
        .behavior_version(BehaviorVersion::latest())
        .build();
    assert_eq!(
        runtime_endpoint_override(&sdk).as_deref(),
        Some("https://runtime-override.example")
    );
}

#[test]
fn runtime_endpoint_override_programmatic_origin_ignores_service_specific() {
    // A `client_config` origin mirrors what the aws-config loader does for
    // `.endpoint_url(...)`: it wins and the service-specific value is
    // ignored, exactly as `Builder::from(&sdk)` does.
    let mut builder = SdkConfig::builder()
        .region(Region::new("us-west-2"))
        .endpoint_url("https://programmatic.example")
        .service_config(BedrockRuntimeServiceConfig(Some(String::from(
            "https://runtime-override.example",
        ))));
    builder.insert_origin("endpoint_url", aws_types::origin::Origin::shared_config());
    let sdk = builder.behavior_version(BehaviorVersion::latest()).build();
    assert_eq!(
        runtime_endpoint_override(&sdk).as_deref(),
        Some("https://programmatic.example")
    );
}

async fn root_of_with_sdk(endpoint: BedrockEndpoint, sdk: &SdkConfig) -> Result<String> {
    let provider = provider_with(endpoint, None);
    let name: ProviderName = "root-test"
        .parse()
        .unwrap_or_else(|e: Error| fail(&e.to_string()));
    root(&name, &provider, endpoint, sdk, "us-west-2").await
}

#[tokio::test]
async fn runtime_root_honors_the_global_endpoint_url_override() {
    let sdk = SdkConfig::builder()
        .region(Region::new("us-west-2"))
        .endpoint_url("https://global-override.example/")
        .behavior_version(BehaviorVersion::latest())
        .build();
    // A trailing slash is trimmed, so chat and embeddings get the bare root.
    assert_eq!(
        root_of_with_sdk(BedrockEndpoint::Runtime, &sdk)
            .await
            .ok()
            .as_deref(),
        Some("https://global-override.example")
    );
}

#[tokio::test]
async fn runtime_root_honors_the_service_specific_override() {
    let sdk = SdkConfig::builder()
        .region(Region::new("us-west-2"))
        .service_config(BedrockRuntimeServiceConfig(Some(String::from(
            "https://runtime-override.example",
        ))))
        .behavior_version(BehaviorVersion::latest())
        .build();
    assert_eq!(
        root_of_with_sdk(BedrockEndpoint::Runtime, &sdk)
            .await
            .ok()
            .as_deref(),
        Some("https://runtime-override.example")
    );
}

#[tokio::test]
async fn runtime_root_without_an_override_keeps_the_aws_host() {
    // The override is additive: with neither it nor `base_url`, the
    // AWS-published host and FIPS/dual-stack resolution are unchanged.
    let sdk = SdkConfig::builder()
        .region(Region::new("us-west-2"))
        .use_fips(true)
        .behavior_version(BehaviorVersion::latest())
        .build();
    assert_eq!(
        root_of_with_sdk(BedrockEndpoint::Runtime, &sdk)
            .await
            .ok()
            .as_deref(),
        Some("https://bedrock-runtime-fips.us-west-2.amazonaws.com")
    );
}

/// The override is refused under FIPS exactly as `base_url` is
/// (`each_endpoint_resolves_its_root_fips_included`): a non-FIPS AWS
/// host fails the provider, a FIPS one is the root.
#[tokio::test]
async fn runtime_root_refuses_a_non_fips_override_when_fips_is_required() {
    let with_override = |url: &str| {
        SdkConfig::builder()
            .region(Region::new("us-west-2"))
            .use_fips(true)
            .endpoint_url(url)
            .behavior_version(BehaviorVersion::latest())
            .build()
    };
    let plain = "https://vpce-0abc.bedrock-runtime.us-west-2.vpce.amazonaws.com";
    let refused = root_of_with_sdk(BedrockEndpoint::Runtime, &with_override(plain)).await;
    assert!(refused.is_err_and(|e| {
        let e = e.to_string();
        e.contains("not one") && e.contains("endpoint-URL override")
    }));
    let service_specific = SdkConfig::builder()
        .region(Region::new("us-west-2"))
        .use_fips(true)
        .service_config(BedrockRuntimeServiceConfig(Some(String::from(
            "https://bedrock-runtime.us-west-2.amazonaws.com",
        ))))
        .behavior_version(BehaviorVersion::latest())
        .build();
    let refused = root_of_with_sdk(BedrockEndpoint::Runtime, &service_specific).await;
    assert!(refused.is_err_and(|e| e.to_string().contains("not one")));
    let fips_vpce = "https://vpce-0abc.bedrock-runtime-fips.us-west-2.vpce.amazonaws.com/";
    assert_eq!(
        root_of_with_sdk(BedrockEndpoint::Runtime, &with_override(fips_vpce))
            .await
            .ok()
            .as_deref(),
        Some("https://vpce-0abc.bedrock-runtime-fips.us-west-2.vpce.amazonaws.com")
    );
    // Without FIPS required, the same non-FIPS override is the root.
    let open = SdkConfig::builder()
        .region(Region::new("us-west-2"))
        .endpoint_url(plain)
        .behavior_version(BehaviorVersion::latest())
        .build();
    assert_eq!(
        root_of_with_sdk(BedrockEndpoint::Runtime, &open)
            .await
            .ok()
            .as_deref(),
        Some(plain)
    );
}

#[tokio::test]
async fn base_url_beats_the_endpoint_url_override() {
    let sdk = SdkConfig::builder()
        .region(Region::new("us-west-2"))
        .endpoint_url("https://override.example")
        .behavior_version(BehaviorVersion::latest())
        .build();
    let provider = provider_with(
        BedrockEndpoint::Runtime,
        Some("https://vpce-0abc.bedrock-runtime.us-west-2.vpce.amazonaws.com"),
    );
    let name: ProviderName = "root-test"
        .parse()
        .unwrap_or_else(|e: Error| fail(&e.to_string()));
    let root = root(
        &name,
        &provider,
        BedrockEndpoint::Runtime,
        &sdk,
        "us-west-2",
    )
    .await;
    assert_eq!(
        root.ok().as_deref(),
        Some("https://vpce-0abc.bedrock-runtime.us-west-2.vpce.amazonaws.com")
    );
}

#[tokio::test]
async fn mantle_root_ignores_the_endpoint_url_override() {
    // The override is a bedrock-runtime concern (the SDK client); the
    // mantle endpoint has no SDK client and stays at its published host.
    let sdk = SdkConfig::builder()
        .region(Region::new("us-west-2"))
        .endpoint_url("https://override.example")
        .behavior_version(BehaviorVersion::latest())
        .build();
    assert_eq!(
        root_of_with_sdk(BedrockEndpoint::Mantle, &sdk)
            .await
            .ok()
            .as_deref(),
        Some("https://bedrock-mantle.us-west-2.api.aws")
    );
}

/// An HTTPS client that records the URI of every request and answers each
/// with a 200 and a placeholder body, so the resolved host is observable
/// without a real Bedrock response.
#[derive(Debug, Clone)]
struct RecordingHttp {
    uris: Arc<Mutex<Vec<String>>>,
}

impl HttpClient for RecordingHttp {
    fn http_connector(
        &self,
        _settings: &HttpConnectorSettings,
        _components: &RuntimeComponents,
    ) -> SharedHttpConnector {
        SharedHttpConnector::new(RecordingConnector {
            uris: Arc::clone(&self.uris),
        })
    }
}

#[derive(Debug)]
struct RecordingConnector {
    uris: Arc<Mutex<Vec<String>>>,
}

impl HttpConnector for RecordingConnector {
    fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
        let uri = request.uri().to_owned();
        self.uris
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(uri);
        HttpConnectorFuture::new(async move {
            Ok(HttpResponse::new(
                200_u16.try_into().unwrap_or_else(|_| fail("status")),
                SdkBody::from("{}"),
            ))
        })
    }
}

/// Chat (OpenAI-compatible, at `root()`) and embeddings (the Converse
/// client) send to one host: with `base_url` unset and an endpoint-URL
/// override in place, the real AWS SDK client — built the way `build()`
/// builds it — sends to the same host `root()` resolves.
#[tokio::test(flavor = "multi_thread")]
async fn chat_and_embeddings_send_to_one_host_under_an_endpoint_override() {
    let uris = Arc::new(Mutex::new(Vec::new()));
    let sdk = aws_config::defaults(BehaviorVersion::latest())
        .http_client(SharedHttpClient::new(RecordingHttp {
            uris: Arc::clone(&uris),
        }))
        .credentials_provider(SharedCredentialsProvider::new(Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            None,
            "test",
        )))
        .region(Region::new("us-west-2"))
        .endpoint_url("https://gateway.example")
        .load()
        .await;
    let provider = provider_with(BedrockEndpoint::Runtime, None);
    let name: ProviderName = "endpoint-override-e2e"
        .parse()
        .unwrap_or_else(|e: Error| fail(&e.to_string()));
    let root = root(
        &name,
        &provider,
        BedrockEndpoint::Runtime,
        &sdk,
        "us-west-2",
    )
    .await
    .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(root, "https://gateway.example");

    // The Converse / InvokeModel client, built the same way `build()` does.
    let client = aws_sdk_bedrockruntime::Client::new(&sdk);
    // The response is a placeholder; the point is the URI the orchestrator
    // resolved, recorded before the call returns.
    let _send_result = client
        .invoke_model()
        .model_id("amazon.titan-embed-text-v2:0")
        .body(aws_smithy_types::Blob::new(b"{}".to_vec()))
        .send()
        .await;
    let captured = uris.lock().unwrap_or_else(PoisonError::into_inner);
    assert!(
        !captured.is_empty(),
        "the SDK client made no request; signing or build failed: {captured:?}"
    );
    assert!(
        captured
            .iter()
            .all(|uri| uri.starts_with("https://gateway.example/")),
        "{captured:?}"
    );
    assert!(
        captured.iter().any(|uri| uri.contains("/model/")),
        "{captured:?}"
    );
}

/// The SDK client opens a tunnel through the proxy for a Bedrock
/// request, where the SDK's builder alone would connect directly.
#[tokio::test(flavor = "multi_thread")]
async fn the_sdk_client_tunnels_through_the_proxy() {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let addr = listener
        .local_addr()
        .unwrap_or_else(|e| fail(&e.to_string()));
    let seen = tokio::spawn(async move {
        let accepted = tokio::time::timeout(Duration::from_secs(10), listener.accept());
        let Ok(Ok((mut socket, _))) = accepted.await else {
            return String::new();
        };
        let mut buf = [0_u8; 2048];
        let read = socket.read(&mut buf).await.unwrap_or_default();
        drop(
            socket
                .write_all(b"HTTP/1.1 502 Bad Gateway\r\ncontent-length: 0\r\n\r\n")
                .await,
        );
        let request = String::from_utf8_lossy(buf.get(..read).unwrap_or_default());
        request.lines().next().unwrap_or_default().to_owned()
    });
    let proxies = Proxies::new(Environment {
        https: Variable::named("HTTPS_PROXY", &format!("http://{addr}")),
        ..Environment::default()
    });
    let sdk = aws_config::defaults(BehaviorVersion::latest())
        .http_client(LimitedAwsHttp::new(ProviderGates::default(), &proxies))
        .credentials_provider(SharedCredentialsProvider::new(Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            None,
            "test",
        )))
        .region(Region::new("us-west-2"))
        .endpoint_url("https://bedrock.invalid")
        .retry_config(aws_config::retry::RetryConfig::disabled())
        .load()
        .await;
    let sent = aws_sdk_bedrockruntime::Client::new(&sdk)
        .invoke_model()
        .model_id("amazon.titan-embed-text-v2:0")
        .body(aws_smithy_types::Blob::new(b"{}".to_vec()))
        .send()
        .await;
    assert!(sent.is_err(), "the stand-in proxy refuses the tunnel");
    assert_eq!(
        seen.await.unwrap_or_default(),
        "CONNECT bedrock.invalid:443 HTTP/1.1"
    );
}

#[test]
fn model_calls_are_found_by_path_and_nothing_else_is() {
    assert_eq!(
        model_of(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-opus-5-5/converse-stream"
        )
        .as_deref(),
        Some("us.anthropic.claude-opus-5-5")
    );
    assert_eq!(
        model_of(
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v2%3A0/invoke"
        )
        .as_deref(),
        Some("amazon.titan-embed-text-v2%3A0")
    );
    // An endpoint override with a path of its own.
    assert_eq!(
        model_of("https://gateway.example/bedrock/model/m/converse").as_deref(),
        Some("m")
    );
    assert_eq!(
        model_of("https://portal.sso.us-east-1.amazonaws.com/federation/credentials"),
        None
    );
    assert_eq!(model_of("https://sts.amazonaws.com/"), None);
    assert_eq!(model_of("https://x.example/model//converse"), None);
}
