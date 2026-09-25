//! OAuth 2.0 provider authentication (design doc 10.2).
//!
//! A provider with `auth = "oauth"` uses an access token from its identity
//! provider as the bearer for its endpoint. [`TokenManager`] hands out that
//! token: reusing it while more than a minute remains, renewing it under one
//! lock when it is about to expire, and otherwise failing with
//! [`Error::AuthRequired`] so the interface can name `quack auth login`.
//! The provider's [`Grant`] decides how a token is obtained: a person signs
//! in with Authorization Code and PKCE through the browser and a loopback
//! listener, or with the device-code flow where no browser can open, and a
//! refresh token renews it; or quack authenticates as itself with the
//! client-credentials grant, run again whenever the token runs out. The
//! token is kept in `control.db`, sealed by the vault (`store.rs`).

pub mod client_key;
pub(crate) mod credential;
mod key_slot;
mod keychain;
mod store;
mod token;

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use jiff::{SignedDuration, Timestamp};
use oauth2::basic::{BasicClient, BasicTokenResponse};
use oauth2::url::Url;
use oauth2::{
    AuthUrl, AuthorizationCode, ClientId, CsrfToken, DeviceAuthorizationUrl, EndpointMaybeSet,
    EndpointNotSet, EndpointSet, PkceCodeChallenge, PkceCodeVerifier, RedirectUrl, RefreshToken,
    Scope, StandardDeviceAuthorizationResponse, TokenResponse, TokenUrl,
};
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, OnceCell, RwLock};

use client_key::ClientKeys;
pub(crate) use credential::Credential;
pub(crate) use key_slot::KeySlot;
pub use key_slot::{KeyLocation, KeySource};
use store::ProviderTokens;
pub use token::CachedToken;
pub(crate) use token::Plaintext;

use crate::config::{ClientAuth, Config, Exchange, Grant, OAuthConfig, ProviderName};
use crate::error::{AuthReason, Error, Result};
use crate::ids::UserId;
use crate::llm::acting::Acting;

/// RFC 7523's grant, which Entra's On-Behalf-Of flow uses.
const JWT_BEARER: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";
/// RFC 8693's grant and its access-token type identifier.
const TOKEN_EXCHANGE: &str = "urn:ietf:params:oauth:grant-type:token-exchange";
const ACCESS_TOKEN_TYPE: &str = "urn:ietf:params:oauth:token-type:access_token";

/// An OAuth error response (RFC 6749 5.2), as far as quack reports it.
#[derive(Debug, Default, Deserialize)]
struct TokenRefusal {
    error: Option<String>,
    error_description: Option<String>,
}

/// Tokens with less than this left are refreshed before use.
const REUSE_MARGIN: SignedDuration = SignedDuration::from_secs(60);
/// How long the browser flow waits for the redirect.
const BROWSER_TIMEOUT: Duration = Duration::from_secs(300);
/// Lifetime assumed when the issuer omits `expires_in`.
const DEFAULT_LIFETIME: SignedDuration = SignedDuration::from_secs(3600);

type OAuthClient =
    BasicClient<EndpointSet, EndpointMaybeSet, EndpointNotSet, EndpointNotSet, EndpointSet>;

/// The endpoints an issuer advertises in its `OpenID` Connect discovery document.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Endpoints {
    /// The issuer identifier, which must be the configured issuer (`OpenID`
    /// Connect Discovery 4.3, RFC 8414 3.3) and which tokens carry.
    pub(crate) issuer: Option<String>,
    #[serde(rename = "authorization_endpoint")]
    pub(crate) authorization: String,
    #[serde(rename = "token_endpoint")]
    pub(crate) token: String,
    #[serde(rename = "device_authorization_endpoint")]
    device_authorization: Option<String>,
    /// Where the issuer publishes the keys its tokens are signed with.
    pub(crate) jwks_uri: Option<String>,
    /// The grants the issuer supports, when it says.
    pub(crate) grant_types_supported: Option<Vec<String>>,
    /// Whether every authorization redirect carries `iss` (RFC 9207).
    #[serde(default)]
    authorization_response_iss_parameter_supported: bool,
}

impl Endpoints {
    /// Where RFC 8414 puts an OAuth server's metadata: the well-known
    /// segment inserted between the host and the issuer's path.
    fn oauth_metadata_url(issuer: &str) -> Result<String> {
        let url = Url::parse(issuer)
            .map_err(|e| Error::Config(format!("issuer '{issuer}' is not a URL: {e}")))?;
        let path = url.path().trim_end_matches('/');
        Ok(format!(
            "{}/.well-known/oauth-authorization-server{path}",
            url.origin().ascii_serialization()
        ))
    }

    /// The redirect's `iss` against this issuer (RFC 9207): when present it
    /// must match, and an issuer that promises one must send it, so a
    /// response from another server cannot pass for this one's.
    ///
    /// # Errors
    ///
    /// Returns why the redirect cannot be from this issuer.
    pub(crate) fn check_response_issuer(
        &self,
        iss: Option<&str>,
    ) -> std::result::Result<(), String> {
        match (iss, self.issuer.as_deref()) {
            (Some(iss), Some(issuer)) if iss == issuer => Ok(()),
            (Some(iss), _) => Err(format!(
                "the redirect names issuer '{iss}', not this one (RFC 9207)"
            )),
            (None, _) if self.authorization_response_iss_parameter_supported => Err(String::from(
                "the issuer promises an iss on its redirects and this one has none (RFC 9207)",
            )),
            (None, _) => Ok(()),
        }
    }
}

/// The HTTP client OAuth requests go through: rustls with aws-lc-rs, no
/// redirects followed, so a token endpoint cannot bounce credentials
/// elsewhere.
#[derive(Debug, Clone)]
pub(crate) struct OAuthHttp(reqwest::Client);

impl OAuthHttp {
    /// # Errors
    ///
    /// Returns an error when the HTTP client cannot be built.
    pub(crate) fn new() -> Result<Self> {
        reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(30))
            .build()
            .map(Self)
            .map_err(|e| Error::Llm(format!("failed to build the OAuth HTTP client: {e}")))
    }

    /// Bridge from the `oauth2` crate's request type to this client; the
    /// crate's own client would pull in `ring`. For a client that
    /// authenticates with assertions, each request gets a newly signed one
    /// here, so a device-code poll or a retry never reuses a spent `jti`.
    pub(crate) fn sender(
        &self,
        credential: &Credential,
    ) -> impl Fn(oauth2::HttpRequest) -> HttpFuture + use<> {
        let client = self.0.clone();
        let assertion = credential.assertion().cloned();
        move |mut request| {
            let client = client.clone();
            let assertion = assertion.clone();
            Box::pin(async move {
                if let Some(assertion) = &assertion {
                    assertion
                        .append_to(request.body_mut())
                        .map_err(|e| HttpError::Assertion(e.to_string()))?;
                }
                let request = reqwest::Request::try_from(request)?;
                let response = client.execute(request).await?;
                let status = response.status();
                let headers = response.headers().clone();
                let body = response.bytes().await?.to_vec();
                let mut out = http::Response::builder().status(status).body(body)?;
                *out.headers_mut() = headers;
                Ok(out)
            })
        }
    }

    /// Read `{issuer_url}/.well-known/openid-configuration`.
    ///
    /// # Errors
    ///
    /// Returns an error when the document cannot be fetched or lacks the
    /// endpoints.
    pub(crate) async fn discover(&self, issuer_url: &str) -> Result<Endpoints> {
        let issuer = issuer_url.trim_end_matches('/');
        let url = format!("{issuer}/.well-known/openid-configuration");
        tracing::debug!(url = %url, "discovering OAuth endpoints");
        let endpoints = match self.fetch_json::<Endpoints>(&url, "OpenID discovery").await {
            Ok(endpoints) => endpoints,
            // A plain OAuth server publishes RFC 8414 metadata instead.
            Err(openid) => {
                let metadata = Endpoints::oauth_metadata_url(issuer)?;
                self.fetch_json(&metadata, "OAuth server metadata")
                    .await
                    .map_err(|_| openid)?
            }
        };
        let named = endpoints.issuer.as_deref().map(|i| i.trim_end_matches('/'));
        if named != Some(issuer) {
            return Err(Error::Llm(format!(
                "the discovery document for {issuer} names issuer {named:?}, not '{issuer}'"
            )));
        }
        Ok(endpoints)
    }

    /// POST a request built by hand (the `oauth2` crate has neither token
    /// exchange nor pushed authorization requests), authenticating the
    /// client `client_id` with `credential`: a secret in HTTP Basic with both
    /// halves form-encoded (RFC 6749 2.3.1) or in the body, or a newly
    /// signed assertion. `client_id` goes in the body unless the secret
    /// travels in the header or `form` already carries it.
    ///
    /// # Errors
    ///
    /// Returns an error when an assertion cannot be signed or the request
    /// cannot be sent or read; a refusal is a status, returned with its
    /// body.
    pub(crate) async fn post_form(
        &self,
        url: &str,
        client_id: &str,
        credential: &Credential,
        form: &[(&str, String)],
    ) -> Result<(reqwest::StatusCode, Vec<u8>)> {
        let mut basic = None;
        let body = {
            let mut body = oauth2::url::form_urlencoded::Serializer::new(String::new());
            for (key, value) in form {
                body.append_pair(key, value);
            }
            let names_client = form.iter().any(|(key, _)| *key == "client_id");
            match credential {
                Credential::Secret {
                    secret,
                    basic: true,
                } => {
                    let encode = |part: &str| {
                        oauth2::url::form_urlencoded::byte_serialize(part.as_bytes())
                            .collect::<String>()
                    };
                    basic = Some(base64::engine::general_purpose::STANDARD.encode(format!(
                        "{}:{}",
                        encode(client_id),
                        encode(secret.expose_secret())
                    )));
                }
                Credential::Secret {
                    secret,
                    basic: false,
                } => {
                    if !names_client {
                        body.append_pair("client_id", client_id);
                    }
                    body.append_pair("client_secret", secret.expose_secret());
                }
                Credential::Public => {
                    if !names_client {
                        body.append_pair("client_id", client_id);
                    }
                }
                Credential::Assertion(assertion) => {
                    if !names_client {
                        body.append_pair("client_id", client_id);
                    }
                    for (key, value) in assertion.params()? {
                        body.append_pair(key, &value);
                    }
                }
            }
            body.finish()
        };
        let mut request = self
            .0
            .post(url)
            .header(reqwest::header::ACCEPT, "application/json")
            .header(
                reqwest::header::CONTENT_TYPE,
                "application/x-www-form-urlencoded",
            );
        if let Some(credentials) = basic {
            request = request.header(
                reqwest::header::AUTHORIZATION,
                format!("Basic {credentials}"),
            );
        }
        let response = request
            .body(body)
            .send()
            .await
            .map_err(|e| Error::Llm(format!("the token request to {url} failed: {e}")))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| Error::Llm(format!("the token response from {url} was cut short: {e}")))?;
        Ok((status, bytes.to_vec()))
    }

    /// GET a JSON document; `what` names it in errors.
    ///
    /// # Errors
    ///
    /// Returns an error when the request fails or the body is not a `T`.
    pub(crate) async fn fetch_json<T: serde::de::DeserializeOwned>(
        &self,
        url: &str,
        what: &str,
    ) -> Result<T> {
        let response = self
            .0
            .get(url)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|e| Error::Llm(format!("{what} at {url} failed: {e}")))?;
        response
            .json::<T>()
            .await
            .map_err(|e| Error::Llm(format!("{what} at {url} returned no usable document: {e}")))
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum HttpError {
    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),
    #[error(transparent)]
    Http(#[from] http::Error),
    #[error("{0}")]
    Assertion(String),
}

pub(crate) type HttpFuture =
    Pin<Box<dyn Future<Output = std::result::Result<oauth2::HttpResponse, HttpError>> + Send>>;

/// What the interface must show the user during a login.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoginPrompt {
    /// Open this URL in a browser; the flow completes on the loopback redirect.
    Browser { url: String },
    /// Visit the URL and enter the code; the flow completes when the issuer
    /// reports approval.
    DeviceCode {
        verification_uri: String,
        user_code: String,
        verification_uri_complete: Option<String>,
        expires_in: Duration,
    },
}

/// Which flow `login` runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LoginFlow {
    /// The provider's `device_code` setting decides: the device-code flow
    /// when it is set, else the browser.
    #[default]
    Configured,
    /// The device-code flow for a provider that signs a person in, whatever
    /// its grant says (no browser, an SSH session). A client-credentials
    /// provider signs nobody in and ignores it.
    DeviceCode,
}

/// What `quack auth status` reports for one provider.
#[derive(Debug, Clone)]
pub struct AuthStatus {
    pub provider: String,
    /// The stored token, `None` when not logged in.
    pub token: Option<TokenStatus>,
    /// Where the vault key that seals it is.
    pub key_location: KeyLocation,
}

/// A cached token's lifetime, as `quack auth status` shows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenStatus {
    pub expires_at: Timestamp,
    pub renewal: Renewal,
}

/// What happens when a cached token expires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Renewal {
    /// A refresh token renews it silently.
    Refreshable,
    /// There is no refresh token: `quack auth login` again.
    Relogin,
    /// The client-credentials grant runs again.
    Regrant,
}

impl std::fmt::Display for Renewal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Refreshable => "refreshable",
            Self::Relogin => "no refresh token",
            Self::Regrant => "renewed by the client-credentials grant",
        })
    }
}

/// Which shared manager a provider gets: one per provider per data
/// directory, since the directory's `control.db` holds its token.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ManagerKey {
    data_dir: PathBuf,
    provider: ProviderName,
}

/// One provider's OAuth state: the config, the stored token, and the
/// in-memory one.
pub struct TokenManager {
    provider: ProviderName,
    config: OAuthConfig,
    store: ProviderTokens,
    /// The key `client_auth = "private_key_jwt"` signs with.
    client_keys: ClientKeys,
    http: OAuthHttp,
    endpoints: OnceCell<Endpoints>,
    current: RwLock<Option<CachedToken>>,
    /// One refresh or login at a time; others await it and reuse the result.
    refresh_lock: Mutex<()>,
    /// `on-behalf-of`: each person's exchanged token, in memory only, behind
    /// a lock of its own so one person's exchange never waits on another's.
    delegated: StdMutex<HashMap<UserId, Arc<Mutex<Option<CachedToken>>>>>,
}

impl std::fmt::Debug for TokenManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenManager")
            .field("provider", &self.provider)
            .field("store", &self.store)
            .finish_non_exhaustive()
    }
}

impl TokenManager {
    /// The shared manager for the provider `config` names `name`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when no such provider is configured or it
    /// does not use `auth = "oauth"`, or an error preparing the manager.
    pub fn for_provider(config: &Config, name: &str) -> Result<Arc<Self>> {
        let (name, provider) = config.providers.get_key_value(name).ok_or_else(|| {
            Error::Config(format!(
                "provider '{name}' is not configured; add [providers.{name}] with auth = \"oauth\""
            ))
        })?;
        let Some(oauth) = provider.auth.oauth() else {
            return Err(Error::Config(format!(
                "provider '{name}' does not use auth = \"oauth\""
            )));
        };
        Self::shared(config, name, oauth)
    }

    /// One manager per provider per process, shared across turns and (later)
    /// server requests so refreshes serialize.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider is not configured for OAuth or the
    /// manager cannot be built.
    pub fn shared(config: &Config, name: &ProviderName, oauth: &OAuthConfig) -> Result<Arc<Self>> {
        static MANAGERS: OnceLock<StdMutex<HashMap<ManagerKey, Arc<TokenManager>>>> =
            OnceLock::new();
        let key = ManagerKey {
            data_dir: config.data_dir().to_path_buf(),
            provider: name.clone(),
        };
        let mut managers = MANAGERS
            .get_or_init(|| StdMutex::new(HashMap::new()))
            .lock()
            .map_err(|e| Error::Llm(format!("token manager registry poisoned: {e}")))?;
        if let Some(existing) = managers.get(&key) {
            return Ok(Arc::clone(existing));
        }
        let manager = Arc::new(Self::new(config, name, oauth.clone(), KeySource::Keychain)?);
        managers.insert(key, Arc::clone(&manager));
        Ok(manager)
    }

    /// A manager for `provider`, keeping its token in the `control.db` of
    /// `config`'s data directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP client cannot be built.
    pub fn new(
        config: &Config,
        provider: &ProviderName,
        oauth: OAuthConfig,
        key_source: KeySource,
    ) -> Result<Self> {
        Ok(Self {
            provider: provider.clone(),
            config: oauth,
            store: ProviderTokens::new(config, provider, key_source),
            client_keys: ClientKeys::new(config, key_source),
            http: OAuthHttp::new()?,
            endpoints: OnceCell::new(),
            current: RwLock::new(None),
            refresh_lock: Mutex::new(()),
            delegated: StdMutex::new(HashMap::new()),
        })
    }

    #[must_use]
    pub fn provider(&self) -> &ProviderName {
        &self.provider
    }

    #[must_use]
    pub const fn grant(&self) -> Grant {
        self.config.grant
    }

    /// A bearer token with more than a minute of life left.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AuthRequired`] when there is no cached token and no
    /// refresh token, or the refresh is refused; other errors are transport
    /// or cache failures.
    pub async fn access_token(&self) -> Result<SecretString> {
        match self.config.grant {
            Grant::OnBehalfOf => self.on_behalf_of().await,
            Grant::AuthorizationCode | Grant::DeviceCode | Grant::ClientCredentials => {
                self.service_token().await
            }
        }
    }

    /// Whether the issuer lists the configured grant in
    /// `grant_types_supported`; `None` when it lists none.
    ///
    /// # Errors
    ///
    /// Returns an error when discovery fails.
    pub async fn issuer_supports_grant(&self) -> Result<Option<bool>> {
        let grant = self.config.grant_type();
        Ok(self
            .endpoints()
            .await?
            .grant_types_supported
            .as_ref()
            .map(|grants| grants.iter().any(|g| g == grant)))
    }

    /// quack's own token for this provider: the one every grant but
    /// `on-behalf-of` hands out, and that grant's actor token.
    ///
    /// # Errors
    ///
    /// As [`TokenManager::access_token`].
    pub async fn service_token(&self) -> Result<SecretString> {
        if let Some(token) = self.fresh_in_memory().await {
            return Ok(token);
        }
        let _refreshing = self.refresh_lock.lock().await;
        if let Some(token) = self.fresh_in_memory().await {
            return Ok(token);
        }
        // Prefer the stored token: another process (a login, a refresh that
        // rotated the refresh token) may have written it since this one was
        // read.
        let cached = match self.store.load().await? {
            Some(token) => Some(token),
            None => self.current.read().await.clone(),
        };
        let renewed = match (cached, self.config.grant) {
            (Some(cached), _) if cached.is_fresh(Timestamp::now(), REUSE_MARGIN) => {
                let access = cached.access_token.clone();
                *self.current.write().await = Some(cached);
                return Ok(access);
            }
            (_, Grant::ClientCredentials | Grant::OnBehalfOf) => self.client_credentials().await?,
            (None, Grant::AuthorizationCode | Grant::DeviceCode) => {
                return Err(self.auth_required(AuthReason::NoToken));
            }
            (Some(cached), Grant::AuthorizationCode | Grant::DeviceCode) => {
                let Some(refresh) = cached.refresh_token.as_ref() else {
                    return Err(self.auth_required(AuthReason::ExpiredNoRefresh));
                };
                match self.refresh(refresh).await {
                    Ok(renewed) => renewed,
                    Err(e) => {
                        let Some(stored) = self.renewed_elsewhere(&cached).await else {
                            return Err(e);
                        };
                        let access = stored.access_token.clone();
                        *self.current.write().await = Some(stored);
                        return Ok(access);
                    }
                }
            }
        };
        let access = renewed.access_token.clone();
        self.store.store(&renewed).await?;
        *self.current.write().await = Some(renewed);
        Ok(access)
    }

    /// After a refused refresh: the token another process sharing the data
    /// directory stored meanwhile, when it differs from `tried` and is fresh.
    ///
    /// The refresh lock is per process, so two processes can both refresh
    /// the same token. Where the issuer rotates refresh tokens, the one
    /// that loses presents a refresh token the winner already used and is
    /// refused, although the winner has stored a good token (#249). An
    /// issuer that answers such reuse by revoking the whole token family
    /// (Okta's and Auth0's reuse detection) leaves nothing to find here;
    /// `docs/authentication.md` covers that case.
    async fn renewed_elsewhere(&self, tried: &CachedToken) -> Option<CachedToken> {
        let stored = match self.store.load().await {
            Ok(stored) => stored?,
            Err(e) => {
                tracing::warn!(provider = %self.provider, error = %e, "re-reading the stored token after a refused refresh failed");
                return None;
            }
        };
        let changed = stored
            .refresh_token
            .as_ref()
            .map(ExposeSecret::expose_secret)
            != tried
                .refresh_token
                .as_ref()
                .map(ExposeSecret::expose_secret)
            || stored.access_token.expose_secret() != tried.access_token.expose_secret();
        let usable = changed && stored.is_fresh(Timestamp::now(), REUSE_MARGIN);
        if usable {
            tracing::info!(provider = %self.provider, "the refresh was refused, but another process has renewed the token; using it");
        }
        usable.then_some(stored)
    }

    async fn fresh_in_memory(&self) -> Option<SecretString> {
        let guard = self.current.read().await;
        guard
            .as_ref()
            .filter(|t| t.is_fresh(Timestamp::now(), REUSE_MARGIN))
            .map(|t| t.access_token.clone())
    }

    fn auth_required(&self, reason: AuthReason) -> Error {
        Error::AuthRequired {
            provider: self.provider.to_string(),
            reason,
        }
    }

    async fn refresh(&self, refresh: &SecretString) -> Result<CachedToken> {
        tracing::info!(provider = %self.provider, "refreshing the OAuth access token");
        let (client, credential) = self.client().await?;
        let http = self.http.sender(&credential);
        let response = client
            .exchange_refresh_token(&RefreshToken::new(refresh.expose_secret().to_owned()))
            .request_async(&http)
            .await
            .map_err(|e| self.auth_required(AuthReason::RefreshFailed(e.to_string())))?;
        let mut token = CachedToken::from_response(&response);
        if token.refresh_token.is_none() {
            token.refresh_token = Some(refresh.clone());
        }
        Ok(token)
    }

    /// Run a login flow and cache the result. `notify` receives what the
    /// user must do; the call returns once the issuer has granted a token.
    /// A client-credentials provider needs nobody, so this checks its
    /// credentials by requesting a token.
    ///
    /// # Errors
    ///
    /// Returns an error when discovery fails, the issuer has no device
    /// endpoint for a device-code login, the redirect port is busy, the
    /// browser flow times out or returns an error, or the exchange fails.
    pub async fn login(
        &self,
        flow: LoginFlow,
        notify: &(dyn Fn(LoginPrompt) + Sync),
    ) -> Result<CachedToken> {
        let _logging_in = self.refresh_lock.lock().await;
        let token = match (self.config.grant, flow) {
            (Grant::ClientCredentials, LoginFlow::Configured | LoginFlow::DeviceCode) => {
                self.client_credentials().await?
            }
            (Grant::DeviceCode, LoginFlow::Configured | LoginFlow::DeviceCode)
            | (Grant::AuthorizationCode, LoginFlow::DeviceCode) => {
                self.login_device_code(notify).await?
            }
            (Grant::AuthorizationCode, LoginFlow::Configured) => self.login_browser(notify).await?,
            (Grant::OnBehalfOf, LoginFlow::Configured | LoginFlow::DeviceCode) => {
                return Err(Error::Config(format!(
                    "provider '{}' acts on behalf of each person signed in to quack serve; there is no login",
                    self.provider
                )));
            }
        };
        self.store.store(&token).await?;
        *self.current.write().await = Some(token.clone());
        tracing::info!(provider = %self.provider, expires_at = %token.expires_at, "login complete");
        Ok(token)
    }

    async fn login_browser(&self, notify: &(dyn Fn(LoginPrompt) + Sync)) -> Result<CachedToken> {
        let redirect = Url::parse(&self.config.redirect_uri).map_err(|e| {
            Error::Config(format!(
                "provider '{}': redirect_uri is not a URL: {e}",
                self.provider
            ))
        })?;
        let host = redirect.host_str().unwrap_or("127.0.0.1");
        let port = redirect.port().unwrap_or(80);
        let listener = TcpListener::bind((host, port)).await.map_err(|e| {
            Error::Llm(format!(
                "cannot listen on {host}:{port} for the OAuth redirect: {e}"
            ))
        })?;

        let (client, credential) = self.client().await?;
        let verifier = PkceCodeVerifier::new(random_token()?);
        let challenge = PkceCodeChallenge::from_code_verifier_sha256(&verifier);
        let state = random_token()?;
        let (url, csrf) = client
            .authorize_url(|| CsrfToken::new(state))
            .add_scopes(self.scopes())
            .set_pkce_challenge(challenge)
            .url();
        notify(LoginPrompt::Browser {
            url: url.to_string(),
        });

        let redirected = tokio::time::timeout(
            BROWSER_TIMEOUT,
            wait_for_callback(&listener, redirect.path(), csrf.secret()),
        )
        .await
        .map_err(|_| {
            Error::Llm(format!(
                "no browser redirect arrived within {} s",
                BROWSER_TIMEOUT.as_secs()
            ))
        })??;

        self.endpoints()
            .await?
            .check_response_issuer(redirected.iss.as_deref())
            .map_err(|e| Error::Llm(format!("provider '{}': {e}", self.provider)))?;
        let http = self.http.sender(&credential);
        let response = client
            .exchange_code(AuthorizationCode::new(redirected.code))
            .set_pkce_verifier(verifier)
            .request_async(&http)
            .await
            .map_err(|e| Error::Llm(format!("code exchange failed: {e}")))?;
        Ok(CachedToken::from_response(&response))
    }

    async fn client_credentials(&self) -> Result<CachedToken> {
        tracing::info!(provider = %self.provider, "requesting a token with the client-credentials grant");
        let (client, credential) = self.client().await?;
        let http = self.http.sender(&credential);
        let response = client
            .exchange_client_credentials()
            .add_scopes(self.scopes())
            .request_async(&http)
            .await
            .map_err(|e| {
                Error::Llm(format!(
                    "provider '{}': the client-credentials grant failed: {e}",
                    self.provider
                ))
            })?;
        Ok(CachedToken::from_response(&response))
    }

    async fn login_device_code(
        &self,
        notify: &(dyn Fn(LoginPrompt) + Sync),
    ) -> Result<CachedToken> {
        let (client, credential) = self.client().await?;
        let http = self.http.sender(&credential);
        let details: StandardDeviceAuthorizationResponse = client
            .exchange_device_code()
            .map_err(|_| {
                Error::Llm(format!(
                    "provider '{}': the issuer advertises no device_authorization_endpoint; use the browser flow",
                    self.provider
                ))
            })?
            .add_scopes(self.scopes())
            .request_async(&http)
            .await
            .map_err(|e| Error::Llm(format!("device authorization failed: {e}")))?;
        notify(LoginPrompt::DeviceCode {
            verification_uri: details.verification_uri().to_string(),
            user_code: details.user_code().secret().to_owned(),
            verification_uri_complete: details
                .verification_uri_complete()
                .map(|u| u.secret().to_owned()),
            expires_in: details.expires_in(),
        });
        let response = client
            .exchange_device_access_token(&details)
            .request_async(&http, tokio::time::sleep, None)
            .await
            .map_err(|e| Error::Llm(format!("device login failed: {e}")))?;
        Ok(CachedToken::from_response(&response))
    }

    /// Whether a token is stored, and its lifetime, without any network use.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored token cannot be read.
    pub async fn status(&self) -> Result<AuthStatus> {
        let cached = self.store.load().await?;
        Ok(AuthStatus {
            provider: self.provider.to_string(),
            token: cached.map(|t| TokenStatus {
                expires_at: t.expires_at,
                renewal: match (self.config.grant, t.refresh_token.is_some()) {
                    (Grant::ClientCredentials | Grant::OnBehalfOf, _) => Renewal::Regrant,
                    (Grant::AuthorizationCode | Grant::DeviceCode, true) => Renewal::Refreshable,
                    (Grant::AuthorizationCode | Grant::DeviceCode, false) => Renewal::Relogin,
                },
            }),
            key_location: self.store.key_location().await?,
        })
    }

    /// Forget the stored token.
    ///
    /// # Errors
    ///
    /// Returns an error when it cannot be deleted.
    pub async fn logout(&self) -> Result<()> {
        let _refreshing = self.refresh_lock.lock().await;
        *self.current.write().await = None;
        self.store.clear().await
    }

    fn scopes(&self) -> Vec<Scope> {
        self.config.scopes.iter().cloned().map(Scope::new).collect()
    }

    async fn endpoints(&self) -> Result<&Endpoints> {
        self.endpoints
            .get_or_try_init(|| self.http.discover(&self.config.issuer_url))
            .await
    }

    /// The client secret from its environment variable, when one is named.
    fn client_secret(&self) -> Result<Option<String>> {
        self.config
            .client_secret_env
            .as_ref()
            .map(|var| {
                std::env::var(var).map_err(|_| {
                    Error::Config(format!(
                        "provider '{}' needs the client secret in environment variable {var}, which is not set",
                        self.provider
                    ))
                })
            })
            .transpose()
    }

    fn delegation(&self, reason: impl Into<String>) -> Error {
        Error::Delegation {
            provider: self.provider.to_string(),
            reason: reason.into(),
        }
    }

    /// `on-behalf-of`: the acting person's token for this provider, reused
    /// while fresh, else exchanged for their own token.
    async fn on_behalf_of(&self) -> Result<SecretString> {
        let Some(acting) = Acting::current() else {
            return Err(self.delegation(
                "no signed-in person is behind this request (the CLI, the terminal, or local mode)",
            ));
        };
        let entry = {
            let mut delegated = self
                .delegated
                .lock()
                .map_err(|e| Error::Llm(format!("delegated tokens poisoned: {e}")))?;
            Arc::clone(delegated.entry(acting.user().clone()).or_default())
        };
        let mut held = entry.lock().await;
        if let Some(token) = held
            .as_ref()
            .filter(|t| t.is_fresh(Timestamp::now(), REUSE_MARGIN))
        {
            return Ok(token.access_token.clone());
        }
        let subject = acting
            .subject_token()
            .await
            .map_err(|reason| self.delegation(reason))?;
        let token = self.exchange(&subject).await?;
        tracing::info!(provider = %self.provider, user = %acting.user(), "exchanged a token on behalf of a user");
        let access = token.access_token.clone();
        *held = Some(token);
        Ok(access)
    }

    /// Exchange `subject` (the person's access token for quack) for a token
    /// to this provider, in the configured wire form.
    async fn exchange(&self, subject: &SecretString) -> Result<CachedToken> {
        let endpoints = self.endpoints().await?;
        let credential = self.credential().await?;
        if credential.is_public() {
            return Err(self.delegation(
                "quack has no client credential: set client_secret_env or client_auth = \"private_key_jwt\"",
            ));
        }
        let scope = self.config.scopes.join(" ");
        let mut form: Vec<(&str, String)> = Vec::new();
        match self.config.exchange {
            Exchange::Entra => {
                form.push(("grant_type", String::from(JWT_BEARER)));
                form.push(("assertion", subject.expose_secret().to_owned()));
                form.push(("requested_token_use", String::from("on_behalf_of")));
                form.push(("scope", scope));
            }
            Exchange::TokenExchange => {
                form.push(("grant_type", String::from(TOKEN_EXCHANGE)));
                form.push(("subject_token", subject.expose_secret().to_owned()));
                form.push(("subject_token_type", String::from(ACCESS_TOKEN_TYPE)));
                form.push(("requested_token_type", String::from(ACCESS_TOKEN_TYPE)));
                if !scope.is_empty() {
                    form.push(("scope", scope));
                }
                if let Some(audience) = &self.config.audience {
                    form.push(("audience", audience.clone()));
                }
                if let Some(resource) = &self.config.resource {
                    form.push(("resource", resource.clone()));
                }
                if self.config.actor {
                    let actor = self.service_token().await?;
                    form.push(("actor_token", actor.expose_secret().to_owned()));
                    form.push(("actor_token_type", String::from(ACCESS_TOKEN_TYPE)));
                }
            }
        }
        let (status, body) = self
            .http
            .post_form(&endpoints.token, &self.config.client_id, &credential, &form)
            .await?;
        if status.is_success() {
            let response: BasicTokenResponse = serde_json::from_slice(&body).map_err(|e| {
                self.delegation(format!("the issuer's answer is not a token response: {e}"))
            })?;
            return Ok(CachedToken::from_response(&response));
        }
        let refusal: TokenRefusal = serde_json::from_slice(&body).unwrap_or_default();
        Err(self.delegation(format!(
            "the issuer refused the exchange ({status}): {}{}",
            refusal.error.as_deref().unwrap_or("no error code"),
            refusal
                .error_description
                .map_or(String::new(), |d| format!(" ({d})"))
        )))
    }

    /// How quack authenticates as this provider's client: its key's
    /// assertions, its secret, or nothing but its `client_id`.
    pub(crate) async fn credential(&self) -> Result<Credential> {
        let audience = match self.config.client_auth {
            ClientAuth::PrivateKeyJwt => self.endpoints().await?.issuer.clone(),
            ClientAuth::ClientSecretPost | ClientAuth::ClientSecretBasic => None,
        };
        Credential::of(
            credential::Registration {
                auth: self.config.client_auth,
                client_id: &self.config.client_id,
                issuer_url: &self.config.issuer_url,
                audience: audience.as_deref(),
                secret: self.client_secret()?,
            },
            &self.client_keys,
        )
        .await
    }

    /// The `oauth2` crate's client for this provider, and the credential
    /// its requests must be sent with (`OAuthHttp::sender`).
    async fn client(&self) -> Result<(OAuthClient, Credential)> {
        let credential = self.credential().await?;
        let endpoints = self.endpoints().await?.clone();
        let parse = |what: &str, url: String| {
            Url::parse(&url).map_err(|e| {
                Error::Llm(format!(
                    "provider '{}': {what} '{url}' is not a URL: {e}",
                    self.provider
                ))
            })
        };
        let device_url = endpoints
            .device_authorization
            .map(|u| parse("device_authorization_endpoint", u))
            .transpose()?
            .map(DeviceAuthorizationUrl::from_url);
        let mut client = BasicClient::new(ClientId::new(self.config.client_id.clone()))
            .set_auth_type(credential.auth_type())
            .set_auth_uri(AuthUrl::from_url(parse(
                "authorization_endpoint",
                endpoints.authorization,
            )?))
            .set_token_uri(TokenUrl::from_url(parse(
                "token_endpoint",
                endpoints.token,
            )?))
            .set_device_authorization_url_option(device_url)
            .set_redirect_uri(RedirectUrl::from_url(parse(
                "redirect_uri",
                self.config.redirect_uri.clone(),
            )?));
        if let Some(secret) = credential.client_secret() {
            client = client.set_client_secret(secret);
        }
        Ok((client, credential))
    }
}

impl CachedToken {
    /// The token a token endpoint answered with; an answer without a
    /// lifetime gets the default one.
    pub(crate) fn from_response<T: TokenResponse>(response: &T) -> Self {
        let lifetime = response
            .expires_in()
            .and_then(|d| SignedDuration::try_from(d).ok())
            .unwrap_or(DEFAULT_LIFETIME);
        Self {
            access_token: SecretString::from(response.access_token().secret().clone()),
            expires_at: Timestamp::now()
                .checked_add(lifetime)
                .unwrap_or(Timestamp::MAX),
            refresh_token: response
                .refresh_token()
                .map(|t| SecretString::from(t.secret().clone())),
        }
    }
}

/// 32 bytes of aws-lc-rs randomness as URL-safe base64: a PKCE verifier or
/// a state value.
pub(crate) fn random_token() -> Result<String> {
    let mut bytes = [0u8; 32];
    aws_lc_rs::rand::fill(&mut bytes)
        .map_err(|_| Error::Llm(String::from("random token generation failed")))?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

/// Serve the loopback redirect until a request carries the expected state,
/// returning the authorization code. Other requests (a favicon probe) get a
/// 404 and the wait continues.
/// What the browser's redirect brought back: the code, and the issuer it
/// names (RFC 9207) when it names one.
struct Redirected {
    code: String,
    iss: Option<String>,
}

async fn wait_for_callback(
    listener: &TcpListener,
    path: &str,
    expected_state: &str,
) -> Result<Redirected> {
    loop {
        let (mut stream, _) = listener.accept().await?;
        let mut buf = Vec::with_capacity(2048);
        let mut chunk = [0u8; 1024];
        loop {
            let n = stream.read(&mut chunk).await?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(chunk.get(..n).unwrap_or_default());
            if buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.len() > 16 * 1024 {
                break;
            }
        }
        let request = String::from_utf8_lossy(&buf);
        let target = request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("/");
        let Some(query) = target
            .strip_prefix(path)
            .and_then(|rest| rest.strip_prefix('?'))
        else {
            respond(&mut stream, http::StatusCode::NOT_FOUND, "Not found.").await;
            continue;
        };
        let params: HashMap<String, String> = oauth2::url::form_urlencoded::parse(query.as_bytes())
            .into_owned()
            .collect();
        if params.get("state").map(String::as_str) != Some(expected_state) {
            respond(
                &mut stream,
                http::StatusCode::BAD_REQUEST,
                "State mismatch; start the login again.",
            )
            .await;
            continue;
        }
        if let Some(error) = params.get("error") {
            let description = params
                .get("error_description")
                .map_or(String::new(), |d| format!(": {d}"));
            respond(
                &mut stream,
                http::StatusCode::OK,
                "Login failed. You can close this window.",
            )
            .await;
            return Err(Error::Llm(format!(
                "the identity provider returned {error}{description}"
            )));
        }
        let Some(code) = params.get("code") else {
            respond(&mut stream, http::StatusCode::BAD_REQUEST, "Missing code.").await;
            continue;
        };
        respond(
            &mut stream,
            http::StatusCode::OK,
            "Login complete. You can close this window and return to quack.",
        )
        .await;
        return Ok(Redirected {
            code: code.clone(),
            iss: params.get("iss").cloned(),
        });
    }
}

async fn respond(stream: &mut tokio::net::TcpStream, status: http::StatusCode, body: &str) {
    let page = format!("<!doctype html><title>quack</title><p>{body}</p>");
    let response = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{page}",
        status.as_u16(),
        status.canonical_reason().unwrap_or_default(),
        page.len()
    );
    // The browser has what it needs; a failed write changes nothing.
    drop(stream.write_all(response.as_bytes()).await);
    drop(stream.shutdown().await);
}

#[cfg(test)]
mod tests;
