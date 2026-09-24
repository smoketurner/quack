//! Amazon Bedrock: rig's Bedrock client over an AWS SDK configuration
//! loaded the way the AWS CLI loads one, so every credential source the
//! SDK knows works unchanged: `AWS_ACCESS_KEY_ID` and friends, a named
//! profile of `~/.aws/config` and `~/.aws/credentials` (`aws_profile`, or
//! `AWS_PROFILE`), IAM Identity Center (`aws sso login`), `credential_process`,
//! assumed roles, web identity (EKS), and the ECS and EC2 instance roles.
//!
//! The SDK sends through its own HTTPS client (rustls on aws-lc-rs), which
//! is wrapped here so a model call takes a permit of the provider's
//! `max_concurrent_requests` like every other provider's (design doc 4.1).
//! Only model calls wait: the SDK's own credential and SSO requests pass
//! straight through.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Mutex, OnceLock, PoisonError};
use std::task::{Context, Poll};

use aws_config::{BehaviorVersion, Region, SdkConfig};
use aws_sdk_bedrockruntime::config::ProvideCredentials;
use aws_smithy_runtime_api::client::http::{
    HttpClient, HttpConnector, HttpConnectorFuture, HttpConnectorSettings, SharedHttpClient,
    SharedHttpConnector,
};
use aws_smithy_runtime_api::client::orchestrator::HttpRequest;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_types::body::SdkBody;
use aws_smithy_types::error::display::DisplayErrorContext;
use bytes::Bytes;
use http_body::{Body, Frame, SizeHint};

use super::limit::{GatePermit, ProviderGates};
use crate::config::{BaseUrl, ProviderConfig, ProviderName};
use crate::error::{Error, Result};

/// rig's Bedrock client: completions over the Converse API, embeddings
/// over `InvokeModel` (Titan Text Embeddings V2's request shape).
pub type BedrockClient = rig::bedrock::client::Client;

/// What makes one Bedrock client different from another: two provider
/// entries that agree on all of it share a client, and so its cached
/// credentials.
#[derive(Clone, PartialEq, Eq, Hash)]
struct ClientKey {
    name: ProviderName,
    profile: Option<String>,
    region: Option<String>,
    endpoint: Option<BaseUrl>,
}

impl ClientKey {
    fn of(name: &ProviderName, provider: &ProviderConfig) -> Self {
        Self {
            name: name.clone(),
            profile: provider.auth.aws_profile().map(str::to_owned),
            region: provider.region.clone(),
            endpoint: provider.base_url.clone(),
        }
    }

    /// The clients built so far, process-wide. The SDK client caches and
    /// refreshes its credentials itself, so reusing it spares every turn
    /// a new SSO or STS round trip.
    fn registry() -> &'static Mutex<HashMap<Self, BedrockClient>> {
        static REGISTRY: OnceLock<Mutex<HashMap<ClientKey, BedrockClient>>> = OnceLock::new();
        REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
    }
}

/// The Bedrock client for provider `name`: built once per process, after
/// checking that the SDK's chain yields a region and credentials, so a
/// missing login fails here with the SDK's reason instead of at the first
/// model call.
///
/// # Errors
///
/// Returns a `Config` error when no region is configured anywhere the SDK
/// looks, or when no credential source yields credentials.
pub(crate) async fn client(
    name: &ProviderName,
    provider: &ProviderConfig,
) -> Result<BedrockClient> {
    let key = ClientKey::of(name, provider);
    if let Some(client) = ClientKey::registry()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return Ok(client.clone());
    }
    let sdk = sdk_config(name, provider).await?;
    check_credentials(name, provider, &sdk).await?;
    let mut conf = aws_sdk_bedrockruntime::config::Builder::from(&sdk);
    if let Some(endpoint) = &provider.base_url {
        // On the Bedrock client only: SSO and STS keep their own endpoints.
        conf = conf.endpoint_url(endpoint.as_str());
    }
    let client = BedrockClient::from(aws_sdk_bedrockruntime::Client::from_conf(conf.build()));
    ClientKey::registry()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key, client.clone());
    Ok(client)
}

/// The SDK configuration the AWS CLI would use for this provider: its
/// profile and region when the file names them, the SDK's defaults
/// otherwise, and the provider's request limit on model calls.
///
/// # Errors
///
/// Returns a `Config` error when no region is found.
pub(crate) async fn sdk_config(
    name: &ProviderName,
    provider: &ProviderConfig,
) -> Result<SdkConfig> {
    use aws_smithy_http_client::tls::{self, rustls_provider::CryptoMode};
    // The same module `crypto::install_default_provider` installs: FIPS on
    // Linux, where the feature is on (docs/crypto.md).
    #[cfg(target_os = "linux")]
    let mode = CryptoMode::AwsLcFips;
    #[cfg(not(target_os = "linux"))]
    let mode = CryptoMode::AwsLc;
    let mut loader = aws_config::defaults(BehaviorVersion::latest()).http_client(LimitedAwsHttp {
        inner: aws_smithy_http_client::Builder::new()
            .tls_provider(tls::Provider::Rustls(mode))
            .build_https(),
        gates: ProviderGates::for_provider(name, provider),
    });
    if let Some(profile) = provider.auth.aws_profile() {
        loader = loader.profile_name(profile);
    }
    if let Some(region) = &provider.region {
        loader = loader.region(Region::new(region.clone()));
    }
    let sdk = loader.load().await;
    if sdk.region().is_none() {
        return Err(Error::Config(format!(
            "provider '{name}' (bedrock) has no AWS region: set region = \"us-east-1\" (or \
             another) under [providers.{name}], export AWS_REGION, or give the profile a region"
        )));
    }
    Ok(sdk)
}

/// Ask the SDK's credential chain once, so a provider without credentials
/// says why (the SDK names each source it tried) and how to fix it.
async fn check_credentials(
    name: &ProviderName,
    provider: &ProviderConfig,
    sdk: &SdkConfig,
) -> Result<()> {
    let Some(credentials) = sdk.credentials_provider() else {
        return Err(no_credentials(
            name,
            provider,
            "the SDK has no credential provider",
        ));
    };
    credentials
        .provide_credentials()
        .await
        .map(drop)
        .map_err(|e| no_credentials(name, provider, &DisplayErrorContext(&e).to_string()))
}

fn no_credentials(name: &ProviderName, provider: &ProviderConfig, reason: &str) -> Error {
    let login = match provider.auth.aws_profile() {
        Some(profile) => format!("`aws sso login --profile {profile}`"),
        None => String::from("`aws sso login`"),
    };
    Error::Config(format!(
        "provider '{name}' (bedrock) found no usable AWS credentials ({reason}); sign in with \
         {login} for an IAM Identity Center profile, or configure credentials as the AWS CLI \
         reads them (`aws configure`, AWS_PROFILE, AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)"
    ))
}

/// The SDK's HTTPS client, with a permit of the provider's limit taken
/// for each model call.
#[derive(Debug, Clone)]
struct LimitedAwsHttp {
    inner: SharedHttpClient,
    gates: ProviderGates,
}

impl HttpClient for LimitedAwsHttp {
    fn http_connector(
        &self,
        settings: &HttpConnectorSettings,
        components: &RuntimeComponents,
    ) -> SharedHttpConnector {
        SharedHttpConnector::new(LimitedConnector {
            inner: self.inner.http_connector(settings, components),
            gates: self.gates.clone(),
        })
    }
}

#[derive(Debug)]
struct LimitedConnector {
    inner: SharedHttpConnector,
    gates: ProviderGates,
}

impl HttpConnector for LimitedConnector {
    fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
        let inner = self.inner.clone();
        let Some(model) = model_of(request.uri()) else {
            return inner.call(request);
        };
        let permit = self.gates.permit(Some(model));
        HttpConnectorFuture::new(async move {
            let permit = permit.await;
            let mut response = inner.call(request).await?;
            let body = std::mem::replace(response.body_mut(), SdkBody::taken());
            *response.body_mut() = SdkBody::from_body_1_x(Holding {
                inner: body,
                permit,
            });
            Ok(response)
        })
    }
}

/// The model a Bedrock runtime request is for, from its path
/// (`/model/{modelId}/converse-stream`, `/model/{modelId}/invoke`, ...),
/// still percent-encoded: it only names a gate. `None` for anything else,
/// such as the SDK's own credential requests.
fn model_of(uri: &str) -> Option<String> {
    let uri: http::Uri = uri.parse().ok()?;
    let (_, rest) = uri.path().split_once("/model/")?;
    let model = rest.split('/').next().filter(|m| !m.is_empty())?;
    Some(model.to_owned())
}

/// A response body that releases its permit when it ends or fails, not
/// only when it is dropped: a streamed answer holds the model until its
/// last event, as [`super::LimitedHttp`]'s streams do.
struct Holding {
    inner: SdkBody,
    permit: Option<GatePermit>,
}

impl Body for Holding {
    type Data = Bytes;
    type Error = aws_smithy_types::body::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<std::result::Result<Frame<Self::Data>, Self::Error>>> {
        let polled = Pin::new(&mut self.inner).poll_frame(cx);
        if matches!(polled, Poll::Ready(None | Some(Err(_)))) {
            self.permit = None;
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        Body::is_end_stream(&self.inner)
    }

    fn size_hint(&self) -> SizeHint {
        Body::size_hint(&self.inner)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use aws_smithy_runtime_api::client::orchestrator::HttpResponse;

    use super::*;
    use crate::config::{ProviderType, RequestLimit};

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
        while let Some(frame) = std::future::poll_fn(|cx| Pin::new(&mut body).poll_frame(cx)).await
        {
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
                tokio::spawn(async move {
                    let response = connector
                        .call(request(&uri))
                        .await
                        .unwrap_or_else(|e| fail(&format!("{e:?}")));
                    drain(response.into_body()).await
                })
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

    #[test]
    fn model_calls_are_found_by_path_and_nothing_else_is() {
        assert_eq!(
            model_of(
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-5/converse-stream"
            )
            .as_deref(),
            Some("us.anthropic.claude-sonnet-5")
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
}
