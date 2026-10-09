//! quack as an OAuth 2.0 protected resource (RFC 9728): when
//! `[server.oidc].audience` is set, the API and MCP accept the issuer's
//! access tokens, `/.well-known/oauth-protected-resource` says so, and a 401
//! from either carries `WWW-Authenticate: Bearer resource_metadata=...` so a
//! client (an MCP client, say) can find the issuer and sign the user in.
//!
//! Each MCP endpoint is a resource of its own, as the MCP authorization
//! specification has it; everything else is the server's origin.

use axum::Json;
use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use quack_core::config::Config;
use serde_json::{Value, json};

use super::error::ApiError;
use super::state::{App, ServeMode};

/// The well-known path the metadata is served under (RFC 9728 section 3).
pub(crate) const METADATA_PATH: &str = "/.well-known/oauth-protected-resource";

/// Scopes every `OpenID` Connect sign-in asks for, which say nothing about
/// quack's own API.
const SIGN_IN_SCOPES: &[&str] = &["openid", "profile", "email", "offline_access"];

/// What the metadata and the challenges are built from.
#[derive(Debug, Clone)]
pub(crate) struct ProtectedResource {
    /// This server's public origin.
    origin: String,
    issuer: String,
    /// The configured scopes beyond the sign-in ones.
    scopes: Vec<String>,
}

impl ProtectedResource {
    /// The resource `quack serve` is, when it accepts the issuer's tokens.
    pub(crate) fn of(config: &Config, mode: ServeMode) -> Option<Self> {
        let oidc = config.server.oidc.as_ref()?;
        oidc.audience.as_ref()?;
        (mode == ServeMode::Login).then(|| Self {
            origin: oidc.public_url(),
            issuer: oidc.issuer_url.clone(),
            scopes: oidc
                .scopes
                .iter()
                .filter(|s| !SIGN_IN_SCOPES.contains(&s.as_str()))
                .cloned()
                .collect(),
        })
    }

    /// The resource a request path belongs to, as a path under the origin:
    /// an MCP endpoint, or the empty path for everything else.
    fn resource_path(request_path: &str) -> &str {
        let path = request_path.trim_end_matches('/');
        match path.strip_prefix("/mcp/v1/") {
            Some(workspace) if !workspace.is_empty() && !workspace.contains('/') => path,
            _ => "",
        }
    }

    /// The `WWW-Authenticate` value for a 401 on `request_path`: where the
    /// metadata is, and `invalid_token` when a credential was presented and
    /// refused.
    pub(crate) fn challenge(&self, request_path: &str, refused: bool) -> String {
        let url = format!(
            "{}{METADATA_PATH}{}",
            self.origin,
            Self::resource_path(request_path)
        );
        if refused {
            format!("Bearer resource_metadata=\"{url}\", error=\"invalid_token\"")
        } else {
            format!("Bearer resource_metadata=\"{url}\"")
        }
    }

    /// The metadata for the resource at `resource_path` (RFC 9728 section 2).
    fn document(&self, resource_path: &str) -> Value {
        let mut document = json!({
            "resource": format!("{}{resource_path}", self.origin),
            "authorization_servers": [self.issuer],
            "bearer_methods_supported": ["header"],
            "resource_name": "quack",
        });
        if !self.scopes.is_empty()
            && let Some(fields) = document.as_object_mut()
        {
            fields.insert(String::from("scopes_supported"), json!(self.scopes));
        }
        document
    }
}

fn not_served() -> ApiError {
    ApiError::not_found("no protected resource metadata: [server.oidc].audience is unset")
}

/// `GET /.well-known/oauth-protected-resource`: the server itself.
pub(crate) async fn metadata(State(app): State<App>) -> Result<Response, ApiError> {
    let resource = app.resource.as_ref().ok_or_else(not_served)?;
    Ok(Json(resource.document("")).into_response())
}

/// `GET /.well-known/oauth-protected-resource/{path}`: one resource, an MCP
/// endpoint or the API, its path inserted after the well-known segment.
pub(crate) async fn metadata_for(
    State(app): State<App>,
    Path(path): Path<String>,
) -> Result<Response, ApiError> {
    let resource = app.resource.as_ref().ok_or_else(not_served)?;
    let path = format!("/{}", path.trim_matches('/'));
    let known = path == "/api/v1" || !ProtectedResource::resource_path(&path).is_empty();
    if !known {
        return Err(ApiError::not_found("no such protected resource"));
    }
    Ok(Json(resource.document(&path)).into_response())
}
