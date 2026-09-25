//! `/mcp/v1/{workspace}`: the MCP server over streamable HTTP, guarded by
//! the same bearer and role checks as the REST API. Every request is
//! authorized here; the transport it is handed to belongs to the caller's
//! workspace, user, and write permission, and the request carries its own
//! `Access` to the tool call, so its audit rows carry the right token,
//! address, and request id.

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
use super::state::{App, McpKey};
use crate::mcp::{Auditor, McpCaller, McpServer, McpSetup};

pub(crate) async fn handle(
    State(app): State<App>,
    mut identity: Identity,
    Path(workspace): Path<WorkspaceId>,
    mut request: Request,
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
    let transport = app
        .mcp_transport(key, || {
            McpServer::new(McpSetup {
                config: app.config.clone(),
                db,
                reader,
                workspace: access.workspace.clone(),
                policy,
                user_id: Some(access.identity.user_id.clone()),
                auditor: Auditor::Server(std::sync::Arc::clone(&app)),
            })
        })
        .await;
    // Every call is audited as the request that made it, not the one that
    // first opened this transport (issue #245): the transport is shared by
    // every request of this user, so the caller travels with the request.
    // rmcp hands the request's parts, extensions included, to the handler.
    request.extensions_mut().insert(McpCaller {
        access,
        acting: Acting::current(),
    });
    let response = transport
        .oneshot(request)
        .await
        .unwrap_or_else(|never| match never {});
    Ok(response.map(Body::new).into_response())
}
