//! Sign-in to `quack serve` through the organization's `OpenID` Connect
//! issuer (design doc 12): Authorization Code with PKCE, the redirect coming
//! back to the server's own callback route.
//!
//! The ID token arrives straight from the token endpoint over TLS, which
//! `OpenID` Connect Core 3.1.3.7 accepts in place of checking its signature,
//! so no JWT or JWKS library is needed: [`SignIn::finish`] checks the issuer,
//! the audience, the expiry, and the nonce this sign-in sent. The refresh
//! token is what keeps the session tied to the issuer afterwards:
//! [`SignIn::renew`] reports a refusal as [`Renewal::Revoked`].

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use jiff::{SignedDuration, Timestamp};
use oauth2::basic::{
    BasicErrorResponse, BasicErrorResponseType, BasicRevocationErrorResponse,
    BasicTokenIntrospectionResponse, BasicTokenType,
};
use oauth2::url::Url;
use oauth2::{
    AuthType, AuthUrl, AuthorizationCode, Client, ClientId, ClientSecret, CsrfToken,
    EndpointNotSet, EndpointSet, ExtraTokenFields, PkceCodeChallenge, PkceCodeVerifier,
    RedirectUrl, RefreshToken, RequestTokenError, Scope, StandardRevocableToken,
    StandardTokenResponse, TokenResponse, TokenUrl,
};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use jsonwebtoken::jwk::{Jwk, JwkSet};
use jsonwebtoken::{AlgorithmFamily, DecodingKey, Validation};
use tokio::sync::{OnceCell, RwLock};

use crate::config::OidcConfig;
use crate::error::{Error, Result};
use crate::llm::oauth::{CachedToken, Endpoints, OAuthHttp, random_token};

/// How far past `exp` an ID token is still accepted, for clock skew.
const CLOCK_LEEWAY: SignedDuration = SignedDuration::from_secs(60);

/// How long fetched signing keys are used before they are fetched again.
const KEYS_TTL: Duration = Duration::from_secs(3600);

/// How soon a token signed with an unknown key may send quack back to the
/// issuer for its keys (a rotation); sooner, it is refused without asking.
const KEYS_REFETCH: Duration = Duration::from_secs(60);

/// The `id_token` a token response carries beside the OAuth fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IdTokenField {
    #[serde(default)]
    id_token: Option<String>,
}

impl ExtraTokenFields for IdTokenField {}

type OidcTokenResponse = StandardTokenResponse<IdTokenField, BasicTokenType>;

/// A client before its endpoints are set; `new` exists only in this state.
type UnconfiguredClient = Client<
    BasicErrorResponse,
    OidcTokenResponse,
    BasicTokenIntrospectionResponse,
    StandardRevocableToken,
    BasicRevocationErrorResponse,
>;

type OidcClient = Client<
    BasicErrorResponse,
    OidcTokenResponse,
    BasicTokenIntrospectionResponse,
    StandardRevocableToken,
    BasicRevocationErrorResponse,
    EndpointSet,
    EndpointNotSet,
    EndpointNotSet,
    EndpointNotSet,
    EndpointSet,
>;

/// The issuer's name for a person: the ID token's `sub`, unique within the
/// issuer and never reassigned, unlike a username or an email address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OidcSubject(String);

impl OidcSubject {
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for OidcSubject {
    fn from(subject: &str) -> Self {
        Self(subject.to_owned())
    }
}

impl std::fmt::Display for OidcSubject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A sign-in the browser was sent off to complete: what the callback must
/// match and what the code exchange must prove.
pub struct Pending {
    /// The `state` sent to the issuer, which comes back on the redirect.
    pub state: String,
    nonce: String,
    verifier: SecretString,
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending").finish_non_exhaustive()
    }
}

/// A completed sign-in: who, and the token to keep for them.
#[derive(Debug)]
pub struct SignedIn {
    pub subject: OidcSubject,
    /// A name for a new user: `preferred_username`, else `email`, else the
    /// subject.
    pub username: String,
    pub token: CachedToken,
}

/// What renewing a signed-in user's token found.
#[derive(Debug)]
pub enum Renewal {
    /// The issuer still vouches for them: the new token.
    Renewed(CachedToken),
    /// The issuer refused the refresh token (revoked, expired, the account
    /// disabled); the session must end.
    Revoked(String),
}

/// Who a token names, in an ID token or an access token alike.
#[derive(Debug, Deserialize)]
struct Person {
    sub: Option<String>,
    preferred_username: Option<String>,
    email: Option<String>,
    /// Every other claim, for a `subject_claim` other than `sub`.
    #[serde(flatten)]
    rest: serde_json::Map<String, serde_json::Value>,
}

impl Person {
    /// The configured claim that names the person, when it is a non-empty
    /// string.
    fn subject(&self, claim: &str) -> Option<OidcSubject> {
        let value = if claim == OidcConfig::DEFAULT_SUBJECT_CLAIM {
            self.sub.as_deref()
        } else {
            self.rest.get(claim).and_then(serde_json::Value::as_str)
        };
        value
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(OidcSubject::from)
    }

    /// A name for a new user: `preferred_username`, else `email`, else the
    /// subject.
    fn username(&self, subject: &OidcSubject) -> String {
        [&self.preferred_username, &self.email]
            .into_iter()
            .flatten()
            .map(|name| name.trim())
            .find(|name| !name.is_empty())
            .unwrap_or(subject.as_str())
            .to_owned()
    }
}

/// The ID token claims quack reads.
#[derive(Debug, Deserialize)]
struct Claims {
    iss: String,
    aud: Audience,
    exp: i64,
    azp: Option<String>,
    nonce: Option<String>,
    #[serde(flatten)]
    person: Person,
}

/// The access token claims quack reads beyond the ones `jsonwebtoken`
/// validates (`iss`, `aud`, `exp`).
#[derive(Debug, Deserialize)]
struct AccessClaims {
    /// Checked by `jsonwebtoken`; kept for the token's expiry.
    exp: i64,
    /// Entra and Okta.
    scp: Option<serde_json::Value>,
    /// RFC 8693 and 9068, Auth0.
    scope: Option<serde_json::Value>,
    #[serde(flatten)]
    person: Person,
}

impl AccessClaims {
    /// Whether the token grants any scope: an access token does, an ID token
    /// (whose `aud` may be the same client id) does not.
    fn has_scope(&self) -> bool {
        [&self.scp, &self.scope]
            .into_iter()
            .flatten()
            .any(|value| match value {
                serde_json::Value::String(text) => !text.trim().is_empty(),
                serde_json::Value::Array(items) => !items.is_empty(),
                _ => false,
            })
    }
}

/// `aud` is one string or a list of them.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Audience {
    One(String),
    Many(Vec<String>),
}

impl Audience {
    fn contains(&self, client_id: &str) -> bool {
        match self {
            Self::One(aud) => aud == client_id,
            Self::Many(auds) => auds.iter().any(|aud| aud == client_id),
        }
    }

    const fn is_many(&self) -> bool {
        match self {
            Self::One(_) => false,
            Self::Many(auds) => auds.len() > 1,
        }
    }
}

impl Claims {
    /// Read the payload of a compact JWT. The signature is not checked: the
    /// token came from the token endpoint over TLS.
    fn of(id_token: &str) -> Result<Self> {
        let payload = id_token
            .split('.')
            .nth(1)
            .ok_or_else(|| sign_in_error("the ID token is not a JWT"))?;
        let bytes = URL_SAFE_NO_PAD
            .decode(payload.trim_end_matches('='))
            .map_err(|e| sign_in_error(format!("the ID token payload is not base64url: {e}")))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| sign_in_error(format!("the ID token claims are unreadable: {e}")))
    }

    /// `OpenID` Connect Core 3.1.3.7 steps 2, 3, 4, 9, and 11.
    fn check(&self, issuer: &str, client_id: &str, nonce: &str, now: Timestamp) -> Result<()> {
        if self.iss != issuer {
            return Err(sign_in_error(format!(
                "the ID token names issuer '{}', not '{issuer}'",
                self.iss
            )));
        }
        if !self.aud.contains(client_id) {
            return Err(sign_in_error(
                "the ID token is not addressed to this client",
            ));
        }
        if self.aud.is_many() && self.azp.as_deref() != Some(client_id) {
            return Err(sign_in_error(
                "the ID token has several audiences and was not issued to this client",
            ));
        }
        let expires = Timestamp::from_second(self.exp)
            .map_err(|e| sign_in_error(format!("the ID token expiry is out of range: {e}")))?;
        if expires.checked_add(CLOCK_LEEWAY).unwrap_or(Timestamp::MAX) <= now {
            return Err(sign_in_error("the ID token has expired"));
        }
        if self.nonce.as_deref() != Some(nonce) {
            return Err(sign_in_error(
                "the ID token does not carry this sign-in's nonce",
            ));
        }
        Ok(())
    }
}

fn sign_in_error(message: impl Into<String>) -> Error {
    Error::SignIn(message.into())
}

/// The server's sign-in client for one issuer.
pub struct SignIn {
    config: OidcConfig,
    http: OAuthHttp,
    endpoints: OnceCell<Endpoints>,
    /// The issuer's signing keys, for access tokens presented as bearers.
    keys: RwLock<Option<Keys>>,
}

/// The issuer's published keys and when they were fetched.
struct Keys {
    set: JwkSet,
    fetched: Instant,
}

impl Keys {
    /// The key `kid` names, or the only key when the token names none.
    fn find(&self, kid: Option<&str>) -> Option<&Jwk> {
        match kid {
            Some(kid) => self.set.find(kid),
            None => match self.set.keys.as_slice() {
                [only] => Some(only),
                _ => None,
            },
        }
    }
}

/// A verified access token: whom it names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bearer {
    pub subject: OidcSubject,
    /// A name for a new user, as a sign-in would give one.
    pub username: String,
    /// When the token stops being accepted.
    pub expires_at: Timestamp,
}

impl std::fmt::Debug for SignIn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignIn")
            .field("issuer", &self.config.issuer_url)
            .finish_non_exhaustive()
    }
}

impl SignIn {
    /// A client for `config`; the issuer is contacted on first use.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP client cannot be built.
    pub fn new(config: OidcConfig) -> Result<Self> {
        Ok(Self {
            config,
            http: OAuthHttp::new()?,
            endpoints: OnceCell::new(),
            keys: RwLock::new(None),
        })
    }

    /// Whether access tokens are accepted as bearers: `audience` is set.
    #[must_use]
    pub const fn accepts_bearers(&self) -> bool {
        self.config.audience.is_some()
    }

    /// The issuer's host, for the sign-in button.
    #[must_use]
    pub fn issuer_host(&self) -> String {
        Url::parse(&self.config.issuer_url)
            .ok()
            .and_then(|url| url.host_str().map(str::to_owned))
            .unwrap_or_else(|| self.config.issuer_url.clone())
    }

    /// The issuer's endpoints, discovered once. The document must name the
    /// configured issuer exactly (`OpenID` Connect Discovery 4.3).
    async fn endpoints(&self) -> Result<&Endpoints> {
        self.endpoints
            .get_or_try_init(|| self.http.discover(&self.config.issuer_url))
            .await
    }

    /// Check the callback's `iss` against the issuer (RFC 9207).
    ///
    /// # Errors
    ///
    /// Returns [`Error::SignIn`] when the redirect cannot be from the issuer.
    pub async fn check_response_issuer(&self, iss: Option<&str>) -> Result<()> {
        self.endpoints()
            .await?
            .check_response_issuer(iss)
            .map_err(sign_in_error)
    }

    async fn client(&self) -> Result<(OidcClient, String)> {
        let endpoints = self.endpoints().await?;
        let parse = |what: &str, url: &str| {
            Url::parse(url).map_err(|e| sign_in_error(format!("{what} '{url}' is not a URL: {e}")))
        };
        let mut client = UnconfiguredClient::new(ClientId::new(self.config.client_id.clone()))
            .set_auth_type(AuthType::RequestBody)
            .set_auth_uri(AuthUrl::from_url(parse(
                "authorization_endpoint",
                &endpoints.authorization,
            )?))
            .set_token_uri(TokenUrl::from_url(parse(
                "token_endpoint",
                &endpoints.token,
            )?))
            .set_redirect_uri(RedirectUrl::from_url(parse(
                "redirect_uri",
                &self.config.redirect_uri,
            )?));
        if let Some(var) = &self.config.client_secret_env {
            let secret = std::env::var(var).map_err(|_| {
                Error::Config(format!(
                    "[server.oidc] needs the client secret in environment variable {var}, which is not set"
                ))
            })?;
            client = client.set_client_secret(ClientSecret::new(secret));
        }
        let issuer = endpoints.issuer.clone().unwrap_or_default();
        Ok((client, issuer))
    }

    /// Start a sign-in: the URL to send the browser to, and what the
    /// callback must present.
    ///
    /// # Errors
    ///
    /// Returns an error when discovery fails or randomness is unavailable.
    pub async fn begin(&self) -> Result<(String, Pending)> {
        let (client, _) = self.client().await?;
        let verifier = PkceCodeVerifier::new(random_token()?);
        let challenge = PkceCodeChallenge::from_code_verifier_sha256(&verifier);
        let state = random_token()?;
        let nonce = random_token()?;
        let (url, _) = client
            .authorize_url(|| CsrfToken::new(state.clone()))
            .add_scopes(self.config.scopes.iter().cloned().map(Scope::new))
            .set_pkce_challenge(challenge)
            .add_extra_param("nonce", nonce.clone())
            .url();
        Ok((
            url.to_string(),
            Pending {
                state,
                nonce,
                verifier: SecretString::from(verifier.into_secret()),
            },
        ))
    }

    /// Finish a sign-in with the code the callback received.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SignIn`] when the exchange is refused or the ID token
    /// is missing or fails a check, and an error for transport failures.
    pub async fn finish(&self, code: &str, pending: Pending) -> Result<SignedIn> {
        let (client, issuer) = self.client().await?;
        let response = client
            .exchange_code(AuthorizationCode::new(code.to_owned()))
            .set_pkce_verifier(PkceCodeVerifier::new(
                pending.verifier.expose_secret().to_owned(),
            ))
            .request_async(&self.http.sender())
            .await
            .map_err(|e| sign_in_error(format!("the code exchange failed: {e}")))?;
        let id_token = response.extra_fields().id_token.as_deref().ok_or_else(|| {
            sign_in_error("the issuer returned no ID token; is \"openid\" in the scopes?")
        })?;
        let claims = Claims::of(id_token)?;
        claims.check(
            &issuer,
            &self.config.client_id,
            &pending.nonce,
            Timestamp::now(),
        )?;
        let claim = &self.config.subject_claim;
        let subject = claims
            .person
            .subject(claim)
            .ok_or_else(|| sign_in_error(format!("the ID token has no {claim} claim")))?;
        if response.refresh_token().is_none() {
            tracing::warn!(
                issuer = %issuer,
                "the issuer returned no refresh token; add \"offline_access\" to [server.oidc].scopes if it needs it"
            );
        }
        Ok(SignedIn {
            username: claims.person.username(&subject),
            subject,
            token: CachedToken::from_response(&response),
        })
    }

    /// Ask the issuer for a new token with the stored refresh token.
    ///
    /// # Errors
    ///
    /// Returns an error when the issuer cannot be reached or refuses for a
    /// reason that is not about this user (a misconfigured client); a
    /// refusal of the grant itself is [`Renewal::Revoked`].
    pub async fn renew(&self, refresh: &SecretString) -> Result<Renewal> {
        let (client, _) = self.client().await?;
        let outcome = client
            .exchange_refresh_token(&RefreshToken::new(refresh.expose_secret().to_owned()))
            .request_async(&self.http.sender())
            .await;
        match outcome {
            Ok(response) => {
                let mut token = CachedToken::from_response(&response);
                if token.refresh_token.is_none() {
                    token.refresh_token = Some(refresh.clone());
                }
                Ok(Renewal::Renewed(token))
            }
            Err(RequestTokenError::ServerResponse(refused))
                if *refused.error() == BasicErrorResponseType::InvalidGrant =>
            {
                Ok(Renewal::Revoked(refused.to_string()))
            }
            Err(e) => Err(sign_in_error(format!("renewing the sign-in failed: {e}"))),
        }
    }
}

impl SignIn {
    /// Verify an access token presented to quack as a bearer: signed by the
    /// issuer's published keys with an asymmetric algorithm, from the issuer,
    /// for `[server.oidc].audience`, unexpired, and carrying a scope.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Bearer`] when the token fails any check or no
    /// audience is configured, and an error when the issuer's documents
    /// cannot be fetched.
    pub async fn verify_bearer(&self, token: &str) -> Result<Bearer> {
        let Some(audience) = self.config.audience.as_deref() else {
            return Err(bearer_error(
                "[server.oidc].audience is unset, so no access token is accepted",
            ));
        };
        let header = jsonwebtoken::decode_header(token)
            .map_err(|e| bearer_error(format!("not a JWT: {e}")))?;
        if AlgorithmFamily::Hmac.algorithms().contains(&header.alg) {
            return Err(bearer_error(format!(
                "{:?} is a shared-secret algorithm; only the issuer's public keys are trusted",
                header.alg
            )));
        }
        let endpoints = self.endpoints().await?;
        let key = self.key(header.kid.as_deref(), endpoints).await?;
        let mut validation = Validation::new(header.alg);
        validation.set_issuer(&[endpoints
            .issuer
            .as_deref()
            .unwrap_or(&self.config.issuer_url)]);
        validation.set_audience(&[audience]);
        validation.set_required_spec_claims(&["exp", "iss", "aud"]);
        validation.leeway = CLOCK_LEEWAY.unsigned_abs().as_secs();
        let claims = jsonwebtoken::decode::<AccessClaims>(token, &key, &validation)
            .map_err(|e| bearer_error(e.to_string()))?
            .claims;
        if !claims.has_scope() {
            return Err(bearer_error(
                "the token carries no scope, so it is not an access token",
            ));
        }
        let claim = &self.config.subject_claim;
        let subject = claims
            .person
            .subject(claim)
            .ok_or_else(|| bearer_error(format!("the token has no {claim} claim")))?;
        Ok(Bearer {
            username: claims.person.username(&subject),
            subject,
            expires_at: Timestamp::from_second(claims.exp).unwrap_or(Timestamp::MIN),
        })
    }

    /// How many signing keys the issuer publishes, fetched now, for `quack
    /// doctor`.
    ///
    /// # Errors
    ///
    /// Returns an error when discovery or the key set cannot be fetched, or
    /// the issuer publishes no `jwks_uri`.
    pub async fn published_keys(&self) -> Result<usize> {
        let endpoints = self.endpoints().await?;
        let jwks_uri = endpoints
            .jwks_uri
            .as_deref()
            .ok_or_else(|| bearer_error("the issuer publishes no jwks_uri"))?;
        let set: JwkSet = self.http.fetch_json(jwks_uri, "the issuer's keys").await?;
        Ok(set.keys.len())
    }

    /// The key `kid` names: from the cache while it is fresh, else from the
    /// issuer's `jwks_uri`, fetched again for an unknown key at most once a
    /// minute so a flood of bad tokens cannot make quack flood the issuer.
    async fn key(&self, kid: Option<&str>, endpoints: &Endpoints) -> Result<DecodingKey> {
        let decode = |jwk: &Jwk| {
            DecodingKey::from_jwk(jwk)
                .map_err(|e| bearer_error(format!("the issuer's key is unusable: {e}")))
        };
        if let Some(keys) = self.keys.read().await.as_ref()
            && keys.fetched.elapsed() < KEYS_TTL
            && let Some(jwk) = keys.find(kid)
        {
            return decode(jwk);
        }
        let mut keys = self.keys.write().await;
        if keys
            .as_ref()
            .is_none_or(|k| k.fetched.elapsed() >= KEYS_REFETCH)
        {
            let jwks_uri = endpoints
                .jwks_uri
                .as_deref()
                .ok_or_else(|| bearer_error("the issuer publishes no jwks_uri"))?;
            let set: JwkSet = self.http.fetch_json(jwks_uri, "the issuer's keys").await?;
            *keys = Some(Keys {
                set,
                fetched: Instant::now(),
            });
        }
        keys.as_ref().and_then(|k| k.find(kid)).map_or_else(
            || {
                Err(bearer_error(format!(
                    "signed with a key the issuer does not publish ({})",
                    kid.unwrap_or("no kid")
                )))
            },
            decode,
        )
    }
}

fn bearer_error(message: impl Into<String>) -> Error {
    Error::Bearer(message.into())
}

mod people;
mod tokens;

pub use people::{PersonTokens, RENEW_MARGIN, Stored};
pub use tokens::UserTokens;

#[cfg(test)]
mod tests;
