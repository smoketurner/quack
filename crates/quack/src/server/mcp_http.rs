//! `/mcp/v1/{workspace}`: the MCP server over streamable HTTP, guarded by
//! the same bearer and role checks as the REST API. Every request is
//! authorized here; the transport it is handed to belongs to the caller's
//! workspace, user, and write permission, so its audit rows carry the
//! right identity.

use axum::body::Body;
use axum::extract::{Path, Request, State};
use axum::response::{IntoResponse, Response};
use quack_core::analysis::policy::WritePolicy;
use quack_core::ids::WorkspaceId;
use quack_core::llm::acting::Acting;
use quack_core::storage::control::Channel;
use tower::ServiceExt;

use super::auth::{Access, Identity, Need};
use super::error::ApiResult;
use super::state::{App, McpEntry, McpKey};
use crate::mcp::{Auditor, McpServer, McpSetup, ServerAuditor};

pub(crate) async fn handle(
    State(app): State<App>,
    mut identity: Identity,
    Path(workspace): Path<WorkspaceId>,
    request: Request,
) -> ApiResult<Response> {
    identity.channel = Some(Channel::Mcp);
    let access = Access::resolve(&app, identity, &workspace, Need::READ).await?;
    let policy = WritePolicy::Deny.allowed_if(access.permits(Need::WRITE));
    let key = McpKey {
        workspace_id: access.workspace.id.clone(),
        user_id: access.identity.user_id.clone(),
        policy,
    };
    let db = app.workspace_db(&access.workspace.id).await?;
    let reader = app.reader_db(&access.workspace.id).await?;
    let McpEntry { transport, server } = app
        .mcp_transport(key, || {
            McpServer::new(McpSetup {
                config: app.config.clone(),
                db,
                reader,
                workspace: access.workspace.clone(),
                policy,
                user_id: Some(access.identity.user_id.clone()),
                acting: Acting::current(),
                auditor: Auditor::Server(Box::new(ServerAuditor {
                    app: std::sync::Arc::clone(&app),
                    access: std::sync::Mutex::new(access.clone()),
                })),
            })
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
