//! `/mcp/v1/{workspace}`: the MCP server over streamable HTTP, guarded by
//! the same bearer and role checks as the REST API. Every request is
//! authorized here; the transport it is handed to belongs to the caller's
//! workspace, user, and write permission, so its audit rows carry the
//! right identity.

use axum::body::Body;
use axum::extract::{Path, Request, State};
use axum::response::{IntoResponse, Response};
use quack_core::analysis::policy::WritePolicy;
use quack_core::storage::control::Channel;
use tower::ServiceExt;

use super::auth::{Access, Identity, Need};
use super::error::ApiResult;
use super::state::App;
use crate::mcp::{Auditor, McpServer, ServerAuditor};

pub(crate) async fn handle(
    State(app): State<App>,
    mut identity: Identity,
    Path(workspace): Path<String>,
    request: Request,
) -> ApiResult<Response> {
    identity.channel = Some(Channel::Mcp);
    let access = Access::resolve(&app, identity, &workspace, Need::READ).await?;
    let can_write = access.permits(Need::WRITE);
    let key = format!(
        "{}:{}:{}",
        access.workspace.id,
        access.identity.user_id,
        if can_write { "rw" } else { "ro" }
    );
    let db = app.workspace_db(&access.workspace.id).await?;
    let reader = app.reader_db(&access.workspace.id).await?;
    let (transport, server) = app
        .mcp_transport(&key, || {
            McpServer::new(
                app.config.clone(),
                db,
                reader,
                access.workspace.clone(),
                if can_write {
                    WritePolicy::Allow
                } else {
                    WritePolicy::Deny
                },
                Some(access.identity.user_id.clone()),
                Auditor::Server(Box::new(ServerAuditor {
                    app: std::sync::Arc::clone(&app),
                    access: std::sync::Mutex::new(access.clone()),
                })),
            )
        })
        .await;
    // Every request is audited as the identity that made it, not the one
    // that first opened this transport.
    server.set_access(access);
    let response = transport
        .oneshot(request)
        .await
        .unwrap_or_else(|never| match never {});
    Ok(response.map(Body::new).into_response())
}
