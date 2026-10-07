//! An object in Amazon S3, or in an S3-compatible store named by
//! `AWS_ENDPOINT_URL_S3` or `AWS_ENDPOINT_URL`, fetched with a `SigV4`-signed
//! GET. Credentials and the region come from the AWS SDK the way the AWS CLI
//! finds them (environment, profile, IAM Identity Center, instance roles),
//! through the same proxies as every other client.

use aws_config::{BehaviorVersion, Region, SdkConfig};
use aws_types::service_config::ServiceConfigKey;
use http::{HeaderName, HeaderValue};

use super::SourceUrl;
use crate::error::{Error, Result};
use crate::llm::bedrock::Signer;
use crate::proxy::Proxies;

/// `s3://bucket/key`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct S3Object {
    bucket: String,
    key: String,
}

/// A GET ready to send: where, and the headers that sign it.
pub(super) struct SignedGet {
    pub(super) url: reqwest::Url,
    pub(super) headers: Vec<(HeaderName, HeaderValue)>,
}

/// Where requests for a bucket go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Endpoint {
    /// AWS itself, in a region, optionally on its FIPS endpoints.
    Aws { region: String, fips: bool },
    /// An endpoint the AWS configuration names, addressed path-style.
    Custom(String),
}

impl S3Object {
    /// The bucket and key an `s3://` URL names.
    ///
    /// # Errors
    ///
    /// Returns an error when the URL has no bucket or no key.
    pub(super) fn parse(url: &SourceUrl) -> Result<Self> {
        // The key is everything after the bucket, taken literally as the AWS
        // CLI takes it: an `s3://` location is not a URL, so nothing in it is
        // percent-decoded, and the request URL encodes it once.
        let text = url.expose().trim();
        let rest = text.get("s3://".len()..).unwrap_or_default();
        let (bucket, key) = rest.split_once('/').unwrap_or((rest, ""));
        let (bucket, key) = (bucket.to_owned(), key.to_owned());
        if bucket.is_empty() || key.is_empty() {
            return Err(Error::Ingestion(String::from(
                "an S3 source is s3://BUCKET/KEY",
            )));
        }
        Ok(Self { bucket, key })
    }

    /// The object's key, whose extension says how to load it.
    pub(super) fn key(&self) -> &str {
        &self.key
    }

    /// The object's URL at `endpoint`: virtual-hosted on AWS, path-style for
    /// a bucket with a dot (its name would break the certificate's wildcard)
    /// and for a custom endpoint.
    pub(super) fn url(&self, endpoint: &Endpoint) -> Result<reqwest::Url> {
        let (base, path_style) = match endpoint {
            Endpoint::Aws { region, fips } => {
                let service = if *fips { "s3-fips" } else { "s3" };
                if self.bucket.contains('.') {
                    (format!("https://{service}.{region}.amazonaws.com"), true)
                } else {
                    (
                        format!("https://{}.{service}.{region}.amazonaws.com", self.bucket),
                        false,
                    )
                }
            }
            Endpoint::Custom(base) => (base.trim_end_matches('/').to_owned(), true),
        };
        let mut url = reqwest::Url::parse(&base)
            .map_err(|e| Error::Ingestion(format!("bad S3 endpoint {base}: {e}")))?;
        {
            let mut segments = url
                .path_segments_mut()
                .map_err(|()| Error::Ingestion(format!("bad S3 endpoint {base}")))?;
            segments.pop_if_empty();
            if path_style {
                segments.push(&self.bucket);
            }
            segments.extend(self.key.split('/'));
        }
        Ok(url)
    }

    /// A signed GET for the object. `region` overrides the configured one,
    /// for the retry S3 asks for when the bucket lives elsewhere.
    ///
    /// # Errors
    ///
    /// Returns an error when the AWS configuration names no region or finds
    /// no credentials.
    pub(super) async fn signed_get(
        &self,
        proxies: &Proxies,
        region: Option<&str>,
    ) -> Result<SignedGet> {
        let mut loader =
            aws_config::defaults(BehaviorVersion::latest()).http_client(proxies.aws_client());
        if let Some(region) = region {
            loader = loader.region(Region::new(region.to_owned()));
        }
        let sdk = loader.load().await;
        let region = sdk.region().map(ToString::to_string).ok_or_else(|| {
            Error::Ingestion(String::from(
                "S3 needs a region; set AWS_REGION or the profile's region",
            ))
        })?;
        let credentials = sdk.credentials_provider().ok_or_else(|| {
            Error::Ingestion(String::from(
                "found no AWS credentials for S3; configure them as the AWS CLI reads them",
            ))
        })?;
        let endpoint = Self::endpoint(&sdk, region.clone());
        let url = self.url(&endpoint)?;
        let signed = Signer::s3(credentials, region)
            .signature("GET", url.as_str(), &[], &[])
            .await
            .map_err(|e| Error::Ingestion(format!("cannot sign the S3 request: {e}")))?;
        let headers = signed
            .into_iter()
            .map(|(name, value)| {
                let name = HeaderName::try_from(name)
                    .map_err(|e| Error::Ingestion(format!("cannot sign the S3 request: {e}")))?;
                let mut value = HeaderValue::try_from(value)
                    .map_err(|e| Error::Ingestion(format!("cannot sign the S3 request: {e}")))?;
                value.set_sensitive(true);
                Ok((name, value))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SignedGet { url, headers })
    }

    /// The endpoint the AWS configuration names for S3: the service-specific
    /// override (`AWS_ENDPOINT_URL_S3`, the `[s3]` profile section), then the
    /// global one, else AWS in `region`, on FIPS endpoints when asked for.
    fn endpoint(sdk: &SdkConfig, region: String) -> Endpoint {
        let configured = sdk.service_config().and_then(|config| {
            ServiceConfigKey::builder()
                .service_id("S3")
                .env("AWS_ENDPOINT_URL")
                .profile("endpoint_url")
                .build()
                .ok()
                .and_then(|key| config.load_config(key))
        });
        match configured.or_else(|| sdk.endpoint_url().map(str::to_owned)) {
            Some(url) => Endpoint::Custom(url),
            None => Endpoint::Aws {
                region,
                fips: sdk.use_fips().unwrap_or(false),
            },
        }
    }
}

#[cfg(test)]
mod tests;
