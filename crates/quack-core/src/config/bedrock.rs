//! The settings only a `type = "bedrock"` provider has: which of Bedrock's
//! two inference endpoints it calls, which API on that endpoint, and the
//! region requests are signed for.
//!
//! Bedrock serves inference on two endpoints that host different models
//! and APIs (docs.aws.amazon.com/bedrock/latest/userguide/endpoints.html):
//!
//! | `endpoint` | host | `api` |
//! |---|---|---|
//! | `runtime` | `bedrock-runtime.{region}.amazonaws.com` | `converse` (default), and `chat-completions` / `responses` under `/openai/v1` |
//! | `mantle` | `bedrock-mantle.{region}.api.aws` | `responses` (default) and `chat-completions` under `/v1` |
//!
//! A provider entry names one endpoint; a model that lives on the other is
//! reached through a second entry (`bedrock/...` and `mantle/...`), the way
//! the model reference already names its provider.
//!
//! `base_url` replaces the endpoint's root, for an interface VPC endpoint
//! without private DNS (`https://vpce-….bedrock-mantle.us-east-1.vpce.amazonaws.com`)
//! or a proxy in front of Bedrock. It is checked against `endpoint` when its
//! host is an AWS one, and a region in that host is the signing region.

use serde::Deserialize;

use super::BaseUrl;
use crate::error::{Error, Result};

/// Which Bedrock inference endpoint a provider calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BedrockEndpoint {
    /// `bedrock-runtime`: the AWS SDK's Converse and `InvokeModel`, and the
    /// OpenAI-compatible APIs under `/openai/v1`; the only one with
    /// embeddings, cross-region inference profiles, and a FIPS endpoint.
    #[default]
    Runtime,
    /// `bedrock-mantle`: the OpenAI-compatible APIs under `/v1`, with the
    /// models and Responses features only it has.
    Mantle,
}

text_enum!(BedrockEndpoint, "Bedrock endpoint", {
    Runtime => "runtime",
    Mantle => "mantle",
});

impl BedrockEndpoint {
    /// The `SigV4` service name requests to it are signed for.
    #[must_use]
    pub const fn signing_name(self) -> &'static str {
        match self {
            Self::Runtime => "bedrock",
            Self::Mantle => "bedrock-mantle",
        }
    }

    /// Where the OpenAI-compatible APIs sit under the endpoint's root.
    #[must_use]
    pub const fn openai_path(self) -> &'static str {
        match self {
            Self::Runtime => "/openai/v1",
            Self::Mantle => "/v1",
        }
    }

    /// The API a provider on this endpoint uses when the file names none.
    #[must_use]
    pub const fn default_api(self) -> BedrockApi {
        match self {
            Self::Runtime => BedrockApi::Converse,
            Self::Mantle => BedrockApi::Responses,
        }
    }

    /// The first DNS label of this endpoint's hosts, VPC endpoints'
    /// included (`vpce-….bedrock-runtime-fips.us-east-1.vpce.amazonaws.com`).
    const fn host_services(self) -> &'static [&'static str] {
        match self {
            Self::Runtime => &["bedrock-runtime", "bedrock-runtime-fips"],
            Self::Mantle => &["bedrock-mantle"],
        }
    }

    /// The root URL AWS publishes for `region` (not FIPS, not dual-stack;
    /// the runtime's are resolved by the AWS SDK instead).
    #[must_use]
    pub fn mantle_root(region: &AwsRegion) -> String {
        format!("https://bedrock-mantle.{region}.api.aws")
    }
}

/// Which API a Bedrock provider's chat model is called through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BedrockApi {
    /// Bedrock's own Converse API through the AWS SDK (`runtime` only).
    Converse,
    /// OpenAI-compatible Chat Completions.
    ChatCompletions,
    /// OpenAI-compatible Responses, sent with `store: false` so Bedrock
    /// keeps no copy of the conversation.
    Responses,
}

text_enum!(BedrockApi, "Bedrock API", {
    Converse => "converse",
    ChatCompletions => "chat-completions",
    Responses => "responses",
});

impl BedrockApi {
    /// Whether `endpoint` serves this API.
    #[must_use]
    pub const fn served_by(self, endpoint: BedrockEndpoint) -> bool {
        !matches!((self, endpoint), (Self::Converse, BedrockEndpoint::Mantle))
    }
}

/// An AWS region name such as `us-east-1`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize)]
#[serde(try_from = "String")]
pub struct AwsRegion(String);

impl AwsRegion {
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Letters, then dash-separated words, ending in a number:
    /// `us-east-1`, `us-gov-west-1`, `ap-southeast-3`.
    fn looks_like(text: &str) -> bool {
        let mut parts = text.split('-');
        let first = parts.next().unwrap_or_default();
        let rest: Vec<&str> = parts.collect();
        let Some((last, middle)) = rest.split_last() else {
            return false;
        };
        first.len() >= 2
            && first.chars().all(|c| c.is_ascii_lowercase())
            && !middle.is_empty()
            && middle
                .iter()
                .all(|w| !w.is_empty() && w.chars().all(|c| c.is_ascii_lowercase()))
            && !last.is_empty()
            && last.chars().all(|c| c.is_ascii_digit())
    }
}

impl TryFrom<String> for AwsRegion {
    type Error = Error;

    fn try_from(region: String) -> Result<Self> {
        if Self::looks_like(&region) {
            Ok(Self(region))
        } else {
            Err(Error::Config(format!(
                "region \"{region}\" is not an AWS region name such as \"us-east-1\""
            )))
        }
    }
}

impl std::fmt::Display for AwsRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A Bedrock provider's endpoint, API, and region, checked against each
/// other and against `base_url`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BedrockConfig {
    pub endpoint: BedrockEndpoint,
    pub api: BedrockApi,
    /// The region requests are signed for: `region`, else the one
    /// `base_url`'s host names; unset, the AWS SDK's chain decides
    /// (`AWS_REGION`, the profile's `region`, instance metadata).
    pub region: Option<AwsRegion>,
}

impl BedrockConfig {
    /// The settings as the file writes them, with `base_url`.
    ///
    /// # Errors
    ///
    /// Returns a `Config` error for an API the endpoint does not serve, a
    /// `base_url` that points at another AWS service or endpoint or
    /// carries the API's path, or a region that disagrees with it.
    pub(super) fn new(
        endpoint: Option<BedrockEndpoint>,
        api: Option<BedrockApi>,
        region: Option<AwsRegion>,
        base_url: Option<&BaseUrl>,
    ) -> Result<Self> {
        let endpoint = endpoint.unwrap_or_default();
        let api = api.unwrap_or_else(|| endpoint.default_api());
        if !api.served_by(endpoint) {
            return Err(Error::Config(format!(
                "api = \"{api}\" is not served by endpoint = \"{endpoint}\"; bedrock-mantle serves \
                 \"responses\" and \"chat-completions\""
            )));
        }
        let mut region = region;
        if let Some(url) = base_url {
            let host = AwsHost::check(url, endpoint)?;
            match (host.and_then(|h| h.region), &region) {
                (Some(named), Some(configured)) if named != *configured => {
                    return Err(Error::Config(format!(
                        "base_url {url} is in {named} but region = \"{configured}\"; requests \
                         are signed for the region they go to"
                    )));
                }
                (Some(named), None) => region = Some(named),
                _ => {}
            }
        }
        Ok(Self {
            endpoint,
            api,
            region,
        })
    }
}

/// What an AWS hostname says: the service its first `bedrock*` label
/// names, and the region after it.
struct AwsHost {
    region: Option<AwsRegion>,
}

impl AwsHost {
    /// The domains AWS endpoints, VPC endpoints included, live under.
    const DOMAINS: [&str; 3] = [".amazonaws.com", ".amazonaws.com.cn", ".api.aws"];

    /// Check `url` as the root of `endpoint`: `None` for a host that is not
    /// an AWS one (a proxy), which is taken as given.
    fn check(url: &BaseUrl, endpoint: BedrockEndpoint) -> Result<Option<Self>> {
        let parsed = reqwest::Url::parse(url.as_str())
            .map_err(|e| Error::Config(format!("base_url {url}: {e}")))?;
        let path = parsed.path().trim_end_matches('/');
        if path.ends_with("/v1") {
            return Err(Error::Config(format!(
                "base_url {url} is the endpoint's root, without the API's path: quack adds \
                 {} itself for endpoint = \"{endpoint}\"",
                endpoint.openai_path()
            )));
        }
        let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
        if !Self::DOMAINS.iter().any(|domain| host.ends_with(domain)) {
            return Ok(None);
        }
        let mut labels = host.split('.');
        let Some(service) = labels.by_ref().find(|l| l.starts_with("bedrock")) else {
            return Err(Error::Config(format!(
                "base_url {url} is not a Bedrock endpoint"
            )));
        };
        if !endpoint.host_services().contains(&service) {
            return Err(Error::Config(format!(
                "base_url {url} is a {service} endpoint, but endpoint = \"{endpoint}\" calls {}",
                endpoint.host_services().join(" or ")
            )));
        }
        let region = labels
            .next()
            .filter(|label| AwsRegion::looks_like(label))
            .map(|label| AwsRegion(label.to_owned()));
        Ok(Some(Self { region }))
    }
}

/// Whether `url` is a FIPS endpoint by its host (`bedrock-runtime-fips`,
/// its VPC endpoints included), or `None` for a host that is not an AWS
/// one, which cannot be told.
#[must_use]
pub fn is_fips_host(url: &BaseUrl) -> Option<bool> {
    let parsed = reqwest::Url::parse(url.as_str()).ok()?;
    let host = parsed.host_str()?.to_ascii_lowercase();
    AwsHost::DOMAINS
        .iter()
        .any(|domain| host.ends_with(domain))
        .then(|| {
            host.split('.')
                .any(|l| l.starts_with("bedrock") && l.ends_with("-fips"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn url(text: &str) -> BaseUrl {
        BaseUrl::try_from(text.to_owned()).unwrap_or_else(|e| fail(&e.to_string()))
    }

    fn region(text: &str) -> AwsRegion {
        AwsRegion::try_from(text.to_owned()).unwrap_or_else(|e| fail(&e.to_string()))
    }

    fn err_of(result: Result<BedrockConfig>) -> String {
        result.err().map(|e| e.to_string()).unwrap_or_default()
    }

    #[test]
    fn each_endpoint_has_its_default_api_and_refuses_what_it_does_not_serve() {
        let runtime = BedrockConfig::new(None, None, None, None);
        assert!(
            runtime.is_ok_and(
                |c| c.endpoint == BedrockEndpoint::Runtime && c.api == BedrockApi::Converse
            )
        );
        let mantle = BedrockConfig::new(Some(BedrockEndpoint::Mantle), None, None, None);
        assert!(mantle.is_ok_and(|c| c.api == BedrockApi::Responses));
        assert!(
            err_of(BedrockConfig::new(
                Some(BedrockEndpoint::Mantle),
                Some(BedrockApi::Converse),
                None,
                None
            ))
            .contains("not served")
        );
        for api in [BedrockApi::ChatCompletions, BedrockApi::Responses] {
            for endpoint in [BedrockEndpoint::Runtime, BedrockEndpoint::Mantle] {
                assert!(BedrockConfig::new(Some(endpoint), Some(api), None, None).is_ok());
            }
        }
    }

    #[test]
    fn a_vpc_endpoint_names_its_service_and_region() {
        let vpce = url("https://vpce-0abc123-4xyz.bedrock-mantle.eu-west-1.vpce.amazonaws.com");
        let config = BedrockConfig::new(Some(BedrockEndpoint::Mantle), None, None, Some(&vpce));
        assert!(config.is_ok_and(|c| c.region == Some(region("eu-west-1"))));
        // The runtime's, FIPS or not, is not the mantle's.
        let runtime = url("https://vpce-0abc.bedrock-runtime-fips.us-east-1.vpce.amazonaws.com");
        assert!(
            BedrockConfig::new(Some(BedrockEndpoint::Runtime), None, None, Some(&runtime)).is_ok()
        );
        assert!(
            err_of(BedrockConfig::new(
                Some(BedrockEndpoint::Mantle),
                None,
                None,
                Some(&runtime)
            ))
            .contains("bedrock-runtime-fips endpoint")
        );
        // A region that disagrees with the host would sign for the wrong one.
        assert!(
            err_of(BedrockConfig::new(
                Some(BedrockEndpoint::Mantle),
                None,
                Some(region("us-east-1")),
                Some(&vpce)
            ))
            .contains("eu-west-1")
        );
        // The control plane is not an inference endpoint.
        assert!(
            err_of(BedrockConfig::new(
                None,
                None,
                None,
                Some(&url("https://bedrock.us-east-1.amazonaws.com"))
            ))
            .contains("bedrock endpoint")
        );
        assert!(
            err_of(BedrockConfig::new(
                None,
                None,
                None,
                Some(&url("https://sts.us-east-1.amazonaws.com"))
            ))
            .contains("not a Bedrock endpoint")
        );
    }

    #[test]
    fn base_url_is_the_root_and_a_proxy_is_taken_as_given() {
        assert!(
            err_of(BedrockConfig::new(
                Some(BedrockEndpoint::Mantle),
                None,
                None,
                Some(&url("https://bedrock-mantle.us-east-1.api.aws/v1"))
            ))
            .contains("quack adds /v1")
        );
        assert!(
            err_of(BedrockConfig::new(
                None,
                Some(BedrockApi::Responses),
                None,
                Some(&url(
                    "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/"
                ))
            ))
            .contains("/openai/v1")
        );
        let proxy = BedrockConfig::new(
            Some(BedrockEndpoint::Mantle),
            None,
            None,
            Some(&url("https://llm-gateway.internal.example/bedrock")),
        );
        assert!(proxy.is_ok_and(|c| c.region.is_none()));
    }

    #[test]
    fn region_names_are_checked() {
        for good in ["us-east-1", "us-gov-west-1", "ap-southeast-3", "cn-north-1"] {
            assert!(AwsRegion::looks_like(good), "{good}");
        }
        for bad in [
            "",
            "us east 1",
            "us-east",
            "useast1",
            "US-EAST-1",
            "vpce",
            "us--east-1",
        ] {
            assert!(!AwsRegion::looks_like(bad), "{bad}");
        }
        assert_eq!(
            is_fips_host(&url("https://bedrock-runtime-fips.us-east-1.amazonaws.com")),
            Some(true)
        );
        assert_eq!(
            is_fips_host(&url("https://bedrock-mantle.us-east-1.api.aws")),
            Some(false)
        );
        assert_eq!(is_fips_host(&url("https://gw.example")), None);
    }
}
