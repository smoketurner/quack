//! Amazon Bedrock, on either of its inference endpoints (`config::bedrock`),
//! with credentials the AWS SDK loads the way the AWS CLI does, so every
//! source it knows works unchanged: `AWS_ACCESS_KEY_ID` and friends, a
//! named profile of `~/.aws/config` and `~/.aws/credentials` (`aws_profile`,
//! or `AWS_PROFILE`), IAM Identity Center (`aws sso login`),
//! `credential_process`, assumed roles, web identity (EKS), and the ECS and
//! EC2 instance roles.
//!
//! Two transports, one `Session` per provider:
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
//!
//! On the runtime endpoint, when `base_url` is unset, the AWS endpoint-URL
//! overrides the SDK's own client honours apply here too — `AWS_ENDPOINT_URL`,
//! the service-specific `AWS_ENDPOINT_URL_BEDROCK_RUNTIME`, and the matching
//! `[default]` / `[bedrock runtime]` profile `endpoint_url` keys — resolved the
//! way `Builder::from(&sdk)` resolves them, so the OpenAI-compatible transports
//! (`chat-completions`, `responses`) and the Converse / `InvokeModel` client
//! never split across two hosts. With FIPS required, an override that names
//! a non-FIPS AWS host is refused, as `base_url` is. The mantle endpoint has
//! no SDK client, so it is unaffected.

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
use aws_smithy_runtime_api::client::result::ConnectorError;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_types::body::SdkBody;
use aws_smithy_types::error::display::DisplayErrorContext;
use bytes::Bytes;
use http_body::{Body, Frame, SizeHint};

use super::LimitedHttp;
use super::limit::{GatePermit, ProviderGates};
use crate::config::bedrock::is_fips_host;
use crate::config::{
    AwsRegion, BaseUrl, BedrockApi, BedrockConfig, BedrockEndpoint, ProviderConfig, ProviderName,
    RetryPolicy,
};
use crate::error::{Error, Result};
use crate::proxy::Proxies;

/// rig's Bedrock client: completions over the Converse API, embeddings
/// over `InvokeModel` (Titan Text Embeddings V2's request shape).
pub type BedrockClient = rig::bedrock::client::BedrockRuntime;

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
        // `Builder::from(&sdk)` inherits the AWS endpoint-URL overrides the SDK
        // loads (`AWS_ENDPOINT_URL`, `AWS_ENDPOINT_URL_BEDROCK_RUNTIME`, the
        // profile keys); `root()` mirrors that resolution for the
        // OpenAI-compatible transports, so chat and embeddings send to one host.
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

/// The endpoint's root: `base_url`, else — on the runtime endpoint — any AWS
/// endpoint-URL override the SDK honours (`AWS_ENDPOINT_URL`,
/// `AWS_ENDPOINT_URL_BEDROCK_RUNTIME`, or the profile equivalents), else the
/// one AWS publishes for `region`, FIPS and dual-stack as the SDK's settings
/// ask (`AWS_USE_FIPS_ENDPOINT`, `use_fips_endpoint` in the profile). With
/// FIPS required, `base_url` and the override alike are refused when they
/// name a non-FIPS AWS host ([`require_fips_host`]); `build()` resolves the
/// root before it makes the Converse client, so the refusal fails the whole
/// provider rather than one transport.
async fn root(
    name: &ProviderName,
    provider: &ProviderConfig,
    endpoint: BedrockEndpoint,
    sdk: &SdkConfig,
    region: &str,
) -> Result<String> {
    let fips = sdk.use_fips().unwrap_or(false);
    if let Some(base_url) = &provider.base_url {
        require_fips_host(name, fips, base_url, "base_url")?;
        return Ok(base_url.trimmed().to_owned());
    }
    match endpoint {
        BedrockEndpoint::Mantle if fips => Err(Error::Config(format!(
            "provider '{name}': FIPS endpoints are required (use_fips_endpoint), and \
             bedrock-mantle has none; use a type = \"bedrock\" provider"
        ))),
        BedrockEndpoint::Mantle => Ok(BedrockEndpoint::mantle_root(&AwsRegion::try_from(
            region.to_owned(),
        )?)),
        BedrockEndpoint::Runtime => {
            // An endpoint-URL override the AWS SDK honours (the same
            // `Builder::from(&sdk)` resolves for the Converse client): honor it
            // here too, so the OpenAI-compatible transport at this root and the
            // Converse / InvokeModel client cannot resolve to different hosts.
            if let Some(override_url) = runtime_endpoint_override(sdk) {
                let override_url = BaseUrl::try_from(override_url).map_err(|e| {
                    Error::Config(format!(
                        "provider '{name}': the AWS endpoint-URL override: {e}"
                    ))
                })?;
                require_fips_host(name, fips, &override_url, "the AWS endpoint-URL override")?;
                return Ok(override_url.trimmed().to_owned());
            }
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

/// Refuses an explicit root (`base_url`, or an AWS endpoint-URL override,
/// named by `source`) that is a non-FIPS AWS host while FIPS endpoints are
/// required. A host that is not an AWS one cannot be told and passes.
fn require_fips_host(name: &ProviderName, fips: bool, url: &BaseUrl, source: &str) -> Result<()> {
    if fips && is_fips_host(url) == Some(false) {
        return Err(Error::Config(format!(
            "provider '{name}': FIPS endpoints are required (use_fips_endpoint), but \
             {source} {url} is not one; use a bedrock-runtime-fips endpoint"
        )));
    }
    Ok(())
}

/// The bedrock-runtime endpoint-URL override `root()` honors — resolved the
/// same way `aws_sdk_bedrockruntime::config::Builder::from(&sdk)` (the
/// `From<&SdkConfig>` impl) resolves it — so the OpenAI-compatible transports
/// and the Converse client never diverge.
///
/// Mirrors the two branches `Builder::from(&sdk)` reads:
///
/// - When `sdk.endpoint_url()` was set programmatically (origin
///   `client_config`), the SDK ignores the service-specific value and uses
///   only the global one.
/// - Otherwise the service-specific value comes first —
///   `AWS_ENDPOINT_URL_BEDROCK_RUNTIME` or the `[bedrock runtime]` profile
///   `endpoint_url` — then the global one (`AWS_ENDPOINT_URL` or `[default]`'s
///   `endpoint_url`) as a fallback. `sdk_config()` never sets `endpoint_url`
///   programmatically, so this is the branch a quack operator hits; the first
///   is carried for parity, so a future programmatic loader keeps the two
///   transports aligned.
fn runtime_endpoint_override(sdk: &SdkConfig) -> Option<String> {
    use aws_types::service_config::ServiceConfigKey;
    if sdk.get_origin("endpoint_url").is_client_config() {
        return sdk.endpoint_url().map(str::to_owned);
    }
    let service_specific = sdk.service_config().and_then(|conf| {
        ServiceConfigKey::builder()
            .service_id("Bedrock Runtime")
            .env("AWS_ENDPOINT_URL")
            .profile("endpoint_url")
            .build()
            .ok()
            .and_then(|key| conf.load_config(key))
    });
    service_specific.or_else(|| sdk.endpoint_url().map(str::to_owned))
}

/// The SDK configuration the AWS CLI would use for this provider: its
/// profile and region when the file names them, the SDK's defaults
/// otherwise, and the provider's request limit on model calls.
async fn sdk_config(name: &ProviderName, provider: &ProviderConfig) -> SdkConfig {
    // The same retry policy as every other provider: the SDK counts
    // attempts, so one more than the retries.
    let retry = aws_config::retry::RetryConfig::standard()
        .with_max_attempts(provider.retry.max_retries.saturating_add(1))
        .with_initial_backoff(provider.retry.backoff)
        .with_max_backoff(RetryPolicy::MAX_WAIT);
    let mut loader = aws_config::defaults(BehaviorVersion::latest())
        .retry_config(retry)
        .http_client(LimitedAwsHttp::new(
            ProviderGates::for_provider(name, provider),
            Proxies::from_env(),
        ));
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
    style: SigningStyle,
}

/// How a service wants its requests signed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SigningStyle {
    /// The `SigV4` defaults most services take.
    Standard,
    /// S3's: the payload's hash in `x-amz-content-sha256`, the path as
    /// given (S3 refuses a normalized one), encoded once.
    S3,
}

impl SigningStyle {
    fn settings(self) -> aws_sigv4::http_request::SigningSettings {
        use aws_sigv4::http_request::{
            PayloadChecksumKind, PercentEncodingMode, SigningSettings, UriPathNormalizationMode,
        };
        let mut settings = SigningSettings::default();
        if self == Self::S3 {
            settings.payload_checksum_kind = PayloadChecksumKind::XAmzSha256;
            settings.uri_path_normalization_mode = UriPathNormalizationMode::Disabled;
            settings.percent_encoding_mode = PercentEncodingMode::Single;
        }
        settings
    }
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
            style: SigningStyle::Standard,
        }
    }

    /// A signer for S3 in `region`.
    pub(crate) fn s3(provider: SharedCredentialsProvider, region: String) -> Self {
        Self {
            style: SigningStyle::S3,
            ..Self::new(provider, region, "s3")
        }
    }

    /// The headers that sign a request of `method` to `uri` carrying
    /// `headers` and `body`; the caller adds them to the request it sends.
    pub(crate) async fn signature(
        &self,
        method: &str,
        uri: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> std::result::Result<Vec<(String, String)>, String> {
        use aws_sigv4::http_request::{SignableBody, SignableRequest, sign};
        use aws_sigv4::sign::v4;

        let identity: Identity = self.credentials().await?.into();
        let params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name(self.service)
            .time(SystemTime::now())
            .settings(self.style.settings())
            .build()
            .map_err(|e| e.to_string())?
            .into();
        let signable = SignableRequest::new(
            method,
            uri,
            headers.iter().copied(),
            SignableBody::Bytes(body),
        )
        .map_err(|e| e.to_string())?;
        let (instructions, _) = sign(signable, &params)
            .map_err(|e| e.to_string())?
            .into_parts();
        Ok(instructions
            .headers()
            .map(|(name, value)| (name.to_owned(), value.to_owned()))
            .collect())
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
    #[expect(
        clippy::result_large_err,
        reason = "rig's HTTP error, which HttpClientExt returns; it keeps the failed response's headers"
    )]
    pub(crate) async fn sign(
        &self,
        mut request: http::Request<Bytes>,
    ) -> std::result::Result<http::Request<Bytes>, rig::http_client::Error> {
        let failed = |e: String| rig::http_client::Error::Instance(e.into());
        request.headers_mut().remove(http::header::AUTHORIZATION);
        let headers = request
            .headers()
            .iter()
            .map(|(name, value)| value.to_str().map(|value| (name.as_str(), value)))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| failed(format!("a header cannot be signed: {e}")))?;
        let uri = request.uri().to_string();
        let signed = self
            .signature(request.method().as_str(), &uri, &headers, request.body())
            .await
            .map_err(failed)?;
        for (name, value) in signed {
            let name = http::HeaderName::try_from(name).map_err(|e| failed(e.to_string()))?;
            let value = http::HeaderValue::try_from(value).map_err(|e| failed(e.to_string()))?;
            request.headers_mut().insert(name, value);
        }
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

impl LimitedAwsHttp {
    fn new(gates: ProviderGates, proxies: &Proxies) -> Self {
        Self {
            inner: proxies.aws_client(),
            gates,
        }
    }
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
            let permit = permit.await.map_err(|e| ConnectorError::user(e.into()))?;
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
mod tests;
