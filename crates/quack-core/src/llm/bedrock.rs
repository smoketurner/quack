//! Amazon Bedrock, on either of its inference endpoints (`config::bedrock`),
//! with credentials the AWS SDK loads the way the AWS CLI does, so every
//! source it knows works unchanged: `AWS_ACCESS_KEY_ID` and friends, a
//! named profile of `~/.aws/config` and `~/.aws/credentials` (`aws_profile`,
//! or `AWS_PROFILE`), IAM Identity Center (`aws sso login`),
//! `credential_process`, assumed roles, web identity (EKS), and the ECS and
//! EC2 instance roles.
//!
//! Two transports, one [`Session`] per provider:
//!
//! - `api = "converse"` (runtime only) is rig-bedrock over the AWS SDK's
//!   own client. Its HTTPS client is wrapped here so a model call takes a
//!   permit of the provider's `max_concurrent_requests` (design doc 4.1);
//!   the SDK's credential and SSO requests pass straight through.
//! - `api = "chat-completions"` and `"responses"` are rig's `OpenAI` clients
//!   over [`LimitedHttp`], which signs each request with `SigV4` for the
//!   endpoint's service (`bedrock` or `bedrock-mantle`) after it has its
//!   permit, so a queued request never carries a stale signature.
//!
//! The root both send to is `base_url` when set (a VPC endpoint, a proxy),
//! else the one AWS publishes for the region: the runtime's from the AWS
//! SDK's own resolver, which honours `use_fips_endpoint` and
//! `use_dualstack_endpoint`, the mantle's `bedrock-mantle.{region}.api.aws`.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::task::{Context, Poll};
use std::time::{Duration, SystemTime};

use aws_config::{BehaviorVersion, Region, SdkConfig};
use aws_credential_types::Credentials;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_sdk_bedrockruntime::config::endpoint::{Params, ResolveEndpoint};
use aws_smithy_runtime_api::client::http::{
    HttpClient, HttpConnector, HttpConnectorFuture, HttpConnectorSettings, SharedHttpClient,
    SharedHttpConnector,
};
use aws_smithy_runtime_api::client::identity::Identity;
use aws_smithy_runtime_api::client::orchestrator::HttpRequest;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_types::body::SdkBody;
use aws_smithy_types::error::display::DisplayErrorContext;
use bytes::Bytes;
use http_body::{Body, Frame, SizeHint};

use super::LimitedHttp;
use super::limit::{GatePermit, ProviderGates};
use crate::config::bedrock::is_fips_host;
use crate::config::{
    BaseUrl, BedrockApi, BedrockConfig, BedrockEndpoint, ProviderConfig, ProviderName,
};
use crate::error::{Error, Result};

/// rig's Bedrock client: completions over the Converse API, embeddings
/// over `InvokeModel` (Titan Text Embeddings V2's request shape).
pub type BedrockClient = rig::bedrock::client::Client;

/// A Bedrock provider, resolved: where it sends, what signs its requests,
/// and, on the runtime endpoint, the AWS SDK client for Converse and
/// embeddings. Built once per provider and process ([`session`]).
pub(crate) struct Session {
    /// The endpoint the provider's type names.
    endpoint: BedrockEndpoint,
    bedrock: BedrockConfig,
    /// The region requests are signed for.
    region: String,
    /// The endpoint's root, without a trailing `/`.
    root: String,
    signer: Arc<Signer>,
    /// `None` on the mantle endpoint, which has no SDK API.
    converse: Option<BedrockClient>,
}

impl Session {
    /// The API the chat model is called through.
    pub(crate) const fn api(&self) -> BedrockApi {
        self.bedrock.api
    }

    pub(crate) fn region(&self) -> &str {
        &self.region
    }

    /// The endpoint's root requests go to.
    pub(crate) fn root(&self) -> &str {
        &self.root
    }

    /// Where the OpenAI-compatible APIs are: the root plus `/openai/v1`
    /// (runtime) or `/v1` (mantle).
    pub(crate) fn openai_base(&self) -> String {
        format!("{}{}", self.root, self.endpoint.openai_path())
    }

    /// The limited HTTP client for provider `name`, signing every request
    /// for this endpoint's service.
    pub(crate) fn http(&self, name: &ProviderName, provider: &ProviderConfig) -> LimitedHttp {
        LimitedHttp::for_provider(name, provider).signed(Arc::clone(&self.signer))
    }

    /// The AWS SDK client for Converse and `InvokeModel`.
    ///
    /// # Errors
    ///
    /// Returns a `Config` error on the mantle endpoint, which serves
    /// neither.
    pub(crate) fn converse(&self, name: &ProviderName) -> Result<BedrockClient> {
        self.converse.clone().ok_or_else(|| {
            Error::Config(format!(
                "provider '{name}' is on the bedrock-mantle endpoint, which serves neither \
                 Converse nor InvokeModel (embeddings); use a type = \"bedrock\" provider"
            ))
        })
    }
}

impl Session {
    /// A session sending to `root`, signing with fixed example credentials.
    #[cfg(test)]
    pub(crate) fn for_test(
        endpoint: BedrockEndpoint,
        bedrock: BedrockConfig,
        root: &str,
        region: &str,
    ) -> Self {
        Self {
            endpoint,
            bedrock,
            region: region.to_owned(),
            root: root.to_owned(),
            signer: Arc::new(Signer::new(
                SharedCredentialsProvider::new(Credentials::new(
                    "AKIDEXAMPLE",
                    "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
                    None,
                    None,
                    "test",
                )),
                region.to_owned(),
                endpoint.signing_name(),
            )),
            converse: None,
        }
    }

    /// The model ids the endpoint lists (`GET {openai_base}/models`, which
    /// bedrock-mantle serves and bedrock-runtime does not), signed and
    /// under the provider's limit like any request.
    ///
    /// # Errors
    ///
    /// Returns rig's HTTP error: a non-success status, or a transport,
    /// signing, or decoding failure.
    pub(crate) async fn models(
        &self,
        name: &ProviderName,
        provider: &ProviderConfig,
    ) -> std::result::Result<Vec<String>, rig::http_client::Error> {
        use rig::http_client::HttpClientExt;

        #[derive(serde::Deserialize)]
        struct Listed {
            #[serde(default)]
            data: Vec<Entry>,
        }
        #[derive(serde::Deserialize)]
        struct Entry {
            id: String,
        }
        let request =
            http::Request::get(format!("{}/models", self.openai_base())).body(Bytes::new())?;
        let response = self
            .http(name, provider)
            .send::<_, Vec<u8>>(request)
            .await?;
        let body = response.into_body().await?;
        serde_json::from_slice::<Listed>(&body)
            .map(|listed| listed.data.into_iter().map(|e| e.id).collect())
            .map_err(|e| rig::http_client::Error::Instance(e.into()))
    }
}

/// What makes one Bedrock session different from another: two provider
/// entries that agree on all of it share one, and so its credentials.
#[derive(Clone, PartialEq, Eq, Hash)]
struct SessionKey {
    name: ProviderName,
    endpoint: Option<BedrockEndpoint>,
    profile: Option<String>,
    bedrock: Option<BedrockConfig>,
    base_url: Option<BaseUrl>,
}

impl SessionKey {
    fn of(name: &ProviderName, provider: &ProviderConfig) -> Self {
        Self {
            name: name.clone(),
            endpoint: provider.provider_type.bedrock_endpoint(),
            profile: provider.auth.aws_profile().map(str::to_owned),
            bedrock: provider.bedrock.clone(),
            base_url: provider.base_url.clone(),
        }
    }

    /// The sessions built so far, process-wide: reusing one spares every
    /// turn a new SSO or STS round trip.
    fn registry() -> &'static Mutex<HashMap<Self, Arc<Session>>> {
        static REGISTRY: OnceLock<Mutex<HashMap<SessionKey, Arc<Session>>>> = OnceLock::new();
        REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
    }
}

/// The session of Bedrock provider `name`: built once per process, after
/// checking that the SDK's chain yields a region and credentials, so a
/// missing login fails here with the SDK's reason instead of at the first
/// model call.
///
/// # Errors
///
/// Returns a `Config` error when the provider is not a Bedrock one, no
/// region is configured anywhere the SDK looks, FIPS is asked of an
/// endpoint without it, or no credential source yields credentials.
pub(crate) async fn session(
    name: &ProviderName,
    provider: &ProviderConfig,
) -> Result<Arc<Session>> {
    let key = SessionKey::of(name, provider);
    if let Some(session) = SessionKey::registry()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return Ok(Arc::clone(session));
    }
    // Boxed: loading the SDK config and resolving credentials is a future of
    // tens of kilobytes, which every caller's would otherwise carry.
    let session = Arc::new(Box::pin(build(name, provider)).await?);
    SessionKey::registry()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key, Arc::clone(&session));
    Ok(session)
}

async fn build(name: &ProviderName, provider: &ProviderConfig) -> Result<Session> {
    let (Some(endpoint), Some(bedrock)) = (
        provider.provider_type.bedrock_endpoint(),
        provider.bedrock.clone(),
    ) else {
        return Err(Error::Config(format!(
            "provider '{name}' ({}) is not a Bedrock provider",
            provider.provider_type
        )));
    };
    let sdk = sdk_config(name, provider).await;
    let region = sdk.region().map(ToString::to_string).ok_or_else(|| {
        Error::Config(format!(
            "provider '{name}' ({endpoint}) has no AWS region: set region = \"us-east-1\" (or \
                 another) under [providers.{name}], export AWS_REGION, or give the profile a region"
        ))
    })?;
    let root = root(name, provider, endpoint, &sdk, &region).await?;
    let Some(credentials) = sdk.credentials_provider() else {
        return Err(no_credentials(
            name,
            provider,
            "the SDK has no credential provider",
        ));
    };
    let signer = Arc::new(Signer::new(
        credentials,
        region.clone(),
        endpoint.signing_name(),
    ));
    signer
        .credentials()
        .await
        .map_err(|e| no_credentials(name, provider, &e))?;
    let converse = (endpoint == BedrockEndpoint::Runtime).then(|| {
        let mut conf = aws_sdk_bedrockruntime::config::Builder::from(&sdk);
        if let Some(base_url) = &provider.base_url {
            // On the Bedrock client only: SSO and STS keep their own endpoints.
            conf = conf.endpoint_url(base_url.trimmed());
        }
        BedrockClient::from(aws_sdk_bedrockruntime::Client::from_conf(conf.build()))
    });
    tracing::info!(provider = %name, %endpoint, api = %bedrock.api, %region, %root, "Bedrock provider ready");
    Ok(Session {
        endpoint,
        bedrock,
        region,
        root,
        signer,
        converse,
    })
}

/// The endpoint's root: `base_url`, else the one AWS publishes for
/// `region`, FIPS and dual-stack as the SDK's settings ask
/// (`AWS_USE_FIPS_ENDPOINT`, `use_fips_endpoint` in the profile).
async fn root(
    name: &ProviderName,
    provider: &ProviderConfig,
    endpoint: BedrockEndpoint,
    sdk: &SdkConfig,
    region: &str,
) -> Result<String> {
    let fips = sdk.use_fips().unwrap_or(false);
    if let Some(base_url) = &provider.base_url {
        if fips && is_fips_host(base_url) == Some(false) {
            return Err(Error::Config(format!(
                "provider '{name}': FIPS endpoints are required (use_fips_endpoint), but \
                 base_url {base_url} is not one; use a bedrock-runtime-fips endpoint"
            )));
        }
        return Ok(base_url.trimmed().to_owned());
    }
    match endpoint {
        BedrockEndpoint::Mantle if fips => Err(Error::Config(format!(
            "provider '{name}': FIPS endpoints are required (use_fips_endpoint), and \
             bedrock-mantle has none; use a type = \"bedrock\" provider"
        ))),
        BedrockEndpoint::Mantle => Ok(BedrockEndpoint::mantle_root(
            &crate::config::AwsRegion::try_from(region.to_owned())?,
        )),
        BedrockEndpoint::Runtime => {
            let params = Params::builder()
                .region(region)
                .use_fips(fips)
                .use_dual_stack(sdk.use_dual_stack().unwrap_or(false))
                .build()
                .map_err(|e| Error::Config(format!("provider '{name}': {e}")))?;
            let resolver = aws_sdk_bedrockruntime::config::endpoint::DefaultResolver::new();
            let endpoint = ResolveEndpoint::resolve_endpoint(&resolver, &params)
                .await
                .map_err(|e| {
                    Error::Config(format!(
                        "provider '{name}': no bedrock-runtime endpoint for {region}: {e}"
                    ))
                })?;
            Ok(endpoint.url().trim_end_matches('/').to_owned())
        }
    }
}

/// The SDK configuration the AWS CLI would use for this provider: its
/// profile and region when the file names them, the SDK's defaults
/// otherwise, and the provider's request limit on model calls.
async fn sdk_config(name: &ProviderName, provider: &ProviderConfig) -> SdkConfig {
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
    if let Some(region) = provider.bedrock.as_ref().and_then(|b| b.region.as_ref()) {
        loader = loader.region(Region::new(region.as_str().to_owned()));
    }
    loader.load().await
}

fn no_credentials(name: &ProviderName, provider: &ProviderConfig, reason: &str) -> Error {
    let login = match provider.auth.aws_profile() {
        Some(profile) => format!("`aws sso login --profile {profile}`"),
        None => String::from("`aws sso login`"),
    };
    Error::Config(format!(
        "provider '{name}' ({}) found no usable AWS credentials ({reason}); sign in with \
         {login} for an IAM Identity Center profile, or configure credentials as the AWS CLI \
         reads them (`aws configure`, AWS_PROFILE, AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)",
        provider.provider_type
    ))
}

/// Signs requests with `SigV4` for one service and region, with credentials
/// from the SDK's chain, held until five minutes before they expire and
/// fetched again by one caller while the others wait.
pub(crate) struct Signer {
    provider: SharedCredentialsProvider,
    cached: tokio::sync::Mutex<Option<Credentials>>,
    region: String,
    service: &'static str,
}

impl std::fmt::Debug for Signer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Signer")
            .field("region", &self.region)
            .field("service", &self.service)
            .finish_non_exhaustive()
    }
}

impl Signer {
    /// How long before expiry credentials are fetched again.
    const REFRESH_AHEAD: Duration = Duration::from_secs(300);

    fn new(provider: SharedCredentialsProvider, region: String, service: &'static str) -> Self {
        Self {
            provider,
            cached: tokio::sync::Mutex::new(None),
            region,
            service,
        }
    }

    /// Current credentials, the SDK's reason when there are none.
    async fn credentials(&self) -> std::result::Result<Credentials, String> {
        let mut cached = self.cached.lock().await;
        let fresh_until = SystemTime::now().checked_add(Self::REFRESH_AHEAD);
        if let Some(credentials) = cached.as_ref()
            && credentials
                .expiry()
                .is_none_or(|expiry| fresh_until.is_some_and(|until| expiry > until))
        {
            return Ok(credentials.clone());
        }
        let credentials = self
            .provider
            .provide_credentials()
            .await
            .map_err(|e| DisplayErrorContext(&e).to_string())?;
        *cached = Some(credentials.clone());
        Ok(credentials)
    }

    /// `request`, signed: any `Authorization` it carried (rig's clients
    /// always set a bearer) is replaced by the `SigV4` one.
    pub(crate) async fn sign(
        &self,
        mut request: http::Request<Bytes>,
    ) -> std::result::Result<http::Request<Bytes>, rig::http_client::Error> {
        use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
        use aws_sigv4::sign::v4;

        let failed = |e: String| rig::http_client::Error::Instance(e.into());
        request.headers_mut().remove(http::header::AUTHORIZATION);
        let identity: Identity = self.credentials().await.map_err(failed)?.into();
        let params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name(self.service)
            .time(SystemTime::now())
            .settings(SigningSettings::default())
            .build()
            .map_err(|e| failed(e.to_string()))?
            .into();
        let headers = request
            .headers()
            .iter()
            .map(|(name, value)| value.to_str().map(|value| (name.as_str(), value)))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| failed(format!("a header cannot be signed: {e}")))?;
        let uri = request.uri().to_string();
        let signable = SignableRequest::new(
            request.method().as_str(),
            uri.as_str(),
            headers.into_iter(),
            SignableBody::Bytes(request.body()),
        )
        .map_err(|e| failed(e.to_string()))?;
        let (instructions, _) = sign(signable, &params)
            .map_err(|e| failed(e.to_string()))?
            .into_parts();
        instructions.apply_to_request_http1x(&mut request);
        Ok(request)
    }
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

    async fn root_of(
        endpoint: BedrockEndpoint,
        fips: bool,
        base_url: Option<&str>,
    ) -> Result<String> {
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
