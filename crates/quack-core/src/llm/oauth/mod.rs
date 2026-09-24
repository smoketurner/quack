//! OAuth 2.0 provider authentication (design doc 10.2).
//!
//! A provider with `auth = "oauth"` uses an access token from its identity
//! provider as the bearer for its endpoint. [`TokenManager`] hands out that
//! token: reusing it while more than a minute remains, refreshing it under
//! one lock when it is about to expire, and otherwise failing with
//! [`Error::AuthRequired`] so the interface can name `quack auth login`.
//! Login is Authorization Code with PKCE through the browser and a loopback
//! listener, or the device-code flow where no browser can open.

mod cache;
mod keychain;

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use jiff::{SignedDuration, Timestamp};
use oauth2::basic::BasicClient;
use oauth2::url::Url;
use oauth2::{
    AuthType, AuthUrl, AuthorizationCode, ClientId, ClientSecret, CsrfToken,
    DeviceAuthorizationUrl, EndpointMaybeSet, EndpointNotSet, EndpointSet, PkceCodeChallenge,
    PkceCodeVerifier, RedirectUrl, RefreshToken, Scope, StandardDeviceAuthorizationResponse,
    TokenResponse, TokenUrl,
};
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, OnceCell, RwLock};

pub use cache::{CachedToken, KeySource, TokenCache};

use crate::config::{OAuthConfig, ProviderName};
use crate::error::{AuthReason, Error, Result};

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
struct Endpoints {
    #[serde(rename = "authorization_endpoint")]
    authorization: String,
    #[serde(rename = "token_endpoint")]
    token: String,
    #[serde(rename = "device_authorization_endpoint")]
    device_authorization: Option<String>,
}

#[derive(Debug, thiserror::Error)]
enum HttpError {
    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),
    #[error(transparent)]
    Http(#[from] http::Error),
}

type HttpFuture =
    Pin<Box<dyn Future<Output = std::result::Result<oauth2::HttpResponse, HttpError>> + Send>>;

/// Bridge from the `oauth2` crate's request type to the rustls + aws-lc-rs
/// reqwest client the rest of the crate uses. Redirects are refused, as the
/// crate's own client does, so a token endpoint cannot bounce credentials
/// elsewhere.
fn http_client(client: reqwest::Client) -> impl Fn(oauth2::HttpRequest) -> HttpFuture {
    move |request| {
        let client = client.clone();
        Box::pin(async move {
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

/// How `login` chooses its flow.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoginOptions {
    /// Use the device-code flow even when the config does not force it
    /// (no browser, an SSH session).
    pub device_code: bool,
}

/// What `quack auth status` reports for one provider.
#[derive(Debug, Clone)]
pub struct AuthStatus {
    pub provider: String,
    pub logged_in: bool,
    pub expires_at: Option<Timestamp>,
    pub has_refresh_token: bool,
    pub key_source: KeySource,
    pub cache_path: PathBuf,
}

/// One provider's OAuth state: the config, the cache, and the in-memory token.
pub struct TokenManager {
    provider: ProviderName,
    config: OAuthConfig,
    cache: TokenCache,
    http: reqwest::Client,
    endpoints: OnceCell<Endpoints>,
    current: RwLock<Option<CachedToken>>,
    /// One refresh or login at a time; others await it and reuse the result.
    refresh_lock: Mutex<()>,
}

impl std::fmt::Debug for TokenManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenManager")
            .field("provider", &self.provider)
            .field("cache", &self.cache)
            .finish_non_exhaustive()
    }
}

impl TokenManager {
    /// A manager for `provider`, caching under `tokens_dir`.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP client cannot be built.
    pub fn new(
        tokens_dir: &Path,
        provider: &ProviderName,
        config: OAuthConfig,
        key_source: KeySource,
    ) -> Result<Self> {
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| Error::Llm(format!("failed to build the OAuth HTTP client: {e}")))?;
        Ok(Self {
            provider: provider.clone(),
            config,
            cache: TokenCache::new(tokens_dir, provider, key_source),
            http,
            endpoints: OnceCell::new(),
            current: RwLock::new(None),
            refresh_lock: Mutex::new(()),
        })
    }

    #[must_use]
    pub fn provider(&self) -> &ProviderName {
        &self.provider
    }

    /// A bearer token with more than a minute of life left.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AuthRequired`] when there is no cached token and no
    /// refresh token, or the refresh is refused; other errors are transport
    /// or cache failures.
    pub async fn access_token(&self) -> Result<SecretString> {
        if let Some(token) = self.fresh_in_memory().await {
            return Ok(token);
        }
        let _refreshing = self.refresh_lock.lock().await;
        if let Some(token) = self.fresh_in_memory().await {
            return Ok(token);
        }
        // Prefer the file: another process (a login, a refresh that rotated
        // the refresh token) may have written it since this token was read.
        let cached = match self.cache.load().await? {
            Some(token) => Some(token),
            None => self.current.read().await.clone(),
        };
        let Some(cached) = cached else {
            return Err(self.auth_required(AuthReason::NoToken));
        };
        if cached.is_fresh(Timestamp::now(), REUSE_MARGIN) {
            let access = cached.access_token.clone();
            *self.current.write().await = Some(cached);
            return Ok(access);
        }
        let Some(refresh) = cached.refresh_token.as_ref() else {
            return Err(self.auth_required(AuthReason::ExpiredNoRefresh));
        };
        let refreshed = self.refresh(refresh).await?;
        let access = refreshed.access_token.clone();
        self.cache.store(&refreshed).await?;
        *self.current.write().await = Some(refreshed);
        Ok(access)
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
        let client = self.client().await?;
        let http = http_client(self.http.clone());
        let response = client
            .exchange_refresh_token(&RefreshToken::new(refresh.expose_secret().to_owned()))
            .request_async(&http)
            .await
            .map_err(|e| self.auth_required(AuthReason::RefreshFailed(e.to_string())))?;
        let mut token = cached_from_response(&response);
        if token.refresh_token.is_none() {
            token.refresh_token = Some(refresh.clone());
        }
        Ok(token)
    }

    /// Run a login flow and cache the result. `notify` receives what the
    /// user must do; the call returns once the issuer has granted a token.
    ///
    /// # Errors
    ///
    /// Returns an error when discovery fails, the issuer has no device
    /// endpoint for a device-code login, the redirect port is busy, the
    /// browser flow times out or returns an error, or the exchange fails.
    pub async fn login(
        &self,
        options: LoginOptions,
        notify: &(dyn Fn(LoginPrompt) + Sync),
    ) -> Result<CachedToken> {
        let _logging_in = self.refresh_lock.lock().await;
        let token = if options.device_code || self.config.device_code {
            self.login_device_code(notify).await?
        } else {
            self.login_browser(notify).await?
        };
        self.cache.store(&token).await?;
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

        let client = self.client().await?;
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

        let code = tokio::time::timeout(
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

        let http = http_client(self.http.clone());
        let response = client
            .exchange_code(AuthorizationCode::new(code))
            .set_pkce_verifier(verifier)
            .request_async(&http)
            .await
            .map_err(|e| Error::Llm(format!("code exchange failed: {e}")))?;
        Ok(cached_from_response(&response))
    }

    async fn login_device_code(
        &self,
        notify: &(dyn Fn(LoginPrompt) + Sync),
    ) -> Result<CachedToken> {
        let client = self.client().await?;
        let http = http_client(self.http.clone());
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
        Ok(cached_from_response(&response))
    }

    /// Whether a token is cached, and its lifetime, without any network use.
    ///
    /// # Errors
    ///
    /// Returns an error when the cache exists but cannot be read.
    pub async fn status(&self) -> Result<AuthStatus> {
        let cached = self.cache.load().await?;
        Ok(AuthStatus {
            provider: self.provider.to_string(),
            logged_in: cached.is_some(),
            expires_at: cached.as_ref().map(|t| t.expires_at),
            has_refresh_token: cached.is_some_and(|t| t.refresh_token.is_some()),
            key_source: self.cache_key_source(),
            cache_path: self.cache.path().to_path_buf(),
        })
    }

    /// Forget the cached token and its key.
    ///
    /// # Errors
    ///
    /// Returns an error when the cache files cannot be removed.
    pub async fn logout(&self) -> Result<()> {
        let _refreshing = self.refresh_lock.lock().await;
        *self.current.write().await = None;
        self.cache.clear().await
    }

    fn cache_key_source(&self) -> KeySource {
        if self.cache.path().with_extension("key").exists() {
            KeySource::File
        } else {
            KeySource::Keychain
        }
    }

    fn scopes(&self) -> Vec<Scope> {
        self.config.scopes.iter().cloned().map(Scope::new).collect()
    }

    async fn client(&self) -> Result<OAuthClient> {
        let endpoints = self
            .endpoints
            .get_or_try_init(|| self.discover())
            .await?
            .clone();
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
            .set_auth_type(AuthType::RequestBody)
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
        if let Some(var) = &self.config.client_secret_env {
            let secret = std::env::var(var).map_err(|_| {
                Error::Config(format!(
                    "provider '{}' needs the client secret in environment variable {var}, which is not set",
                    self.provider
                ))
            })?;
            client = client.set_client_secret(ClientSecret::new(secret));
        }
        Ok(client)
    }

    async fn discover(&self) -> Result<Endpoints> {
        let url = format!(
            "{}/.well-known/openid-configuration",
            self.config.issuer_url.trim_end_matches('/')
        );
        tracing::debug!(provider = %self.provider, url = %url, "discovering OAuth endpoints");
        let response = self
            .http
            .get(&url)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|e| Error::Llm(format!("OpenID discovery at {url} failed: {e}")))?;
        response.json::<Endpoints>().await.map_err(|e| {
            Error::Llm(format!(
                "OpenID discovery at {url} returned no usable document: {e}"
            ))
        })
    }
}

fn cached_from_response<T: TokenResponse>(response: &T) -> CachedToken {
    let lifetime = response
        .expires_in()
        .and_then(|d| SignedDuration::try_from(d).ok())
        .unwrap_or(DEFAULT_LIFETIME);
    CachedToken {
        access_token: SecretString::from(response.access_token().secret().clone()),
        expires_at: Timestamp::now()
            .checked_add(lifetime)
            .unwrap_or(Timestamp::MAX),
        refresh_token: response
            .refresh_token()
            .map(|t| SecretString::from(t.secret().clone())),
    }
}

/// 32 bytes of aws-lc-rs randomness as URL-safe base64: a PKCE verifier or
/// a state value.
fn random_token() -> Result<String> {
    let mut bytes = [0u8; 32];
    aws_lc_rs::rand::fill(&mut bytes)
        .map_err(|_| Error::Llm(String::from("random token generation failed")))?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

/// Serve the loopback redirect until a request carries the expected state,
/// returning the authorization code. Other requests (a favicon probe) get a
/// 404 and the wait continues.
async fn wait_for_callback(
    listener: &TcpListener,
    path: &str,
    expected_state: &str,
) -> Result<String> {
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
            respond(&mut stream, "404 Not Found", "Not found.").await;
            continue;
        };
        let params: HashMap<String, String> = oauth2::url::form_urlencoded::parse(query.as_bytes())
            .into_owned()
            .collect();
        if params.get("state").map(String::as_str) != Some(expected_state) {
            respond(
                &mut stream,
                "400 Bad Request",
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
                "200 OK",
                "Login failed. You can close this window.",
            )
            .await;
            return Err(Error::Llm(format!(
                "the identity provider returned {error}{description}"
            )));
        }
        let Some(code) = params.get("code") else {
            respond(&mut stream, "400 Bad Request", "Missing code.").await;
            continue;
        };
        respond(
            &mut stream,
            "200 OK",
            "Login complete. You can close this window and return to quack.",
        )
        .await;
        return Ok(code.clone());
    }
}

async fn respond(stream: &mut tokio::net::TcpStream, status: &str, body: &str) {
    let page = format!("<!doctype html><title>quack</title><p>{body}</p>");
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{page}",
        page.len()
    );
    // The browser has what it needs; a failed write changes nothing.
    drop(stream.write_all(response.as_bytes()).await);
    drop(stream.shutdown().await);
}

/// One manager per provider per process, shared across turns and (later)
/// server requests so refreshes serialize.
///
/// # Errors
///
/// Returns an error when the provider is not configured for OAuth or the
/// manager cannot be built.
pub fn shared_manager(
    tokens_dir: &Path,
    name: &ProviderName,
    oauth: &OAuthConfig,
) -> Result<Arc<TokenManager>> {
    static MANAGERS: OnceLock<StdMutex<HashMap<PathBuf, Arc<TokenManager>>>> = OnceLock::new();
    let key = tokens_dir.join(name.as_str());
    let mut managers = MANAGERS
        .get_or_init(|| StdMutex::new(HashMap::new()))
        .lock()
        .map_err(|e| Error::Llm(format!("token manager registry poisoned: {e}")))?;
    if let Some(existing) = managers.get(&key) {
        return Ok(Arc::clone(existing));
    }
    let manager = Arc::new(TokenManager::new(
        tokens_dir,
        name,
        oauth.clone(),
        KeySource::Keychain,
    )?);
    managers.insert(key, Arc::clone(&manager));
    Ok(manager)
}

#[cfg(test)]
mod tests;
