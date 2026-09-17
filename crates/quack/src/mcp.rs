//! The workspace as an MCP server: the same tools over stdio (`quack mcp`,
//! for Claude Code and editors) and over streamable HTTP under `quack
//! serve` (`/mcp/v1/{workspace}`, with the workspace token). Design doc
//! section 11.3.
//!
//! Tools: `query` (one agent turn), `search` (hybrid retrieval, no model),
//! `sql` (classified, writes need permission), `list_tables`,
//! `describe_table`, `list_documents`. Resources: `quack://workspace/tables`,
//! `quack://workspace/tables/{name}/schema`, `quack://workspace/documents`,
//! `quack://workspace/ontology`, `quack://workspace/context`.
//!
//! Over HTTP every call is audited through the request's `Access`, like
//! the REST API; over stdio nothing is audited, like the CLI.

use std::sync::Arc;

use quack_core::analysis::events;
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::SharedDb;
use quack_core::config::Config;
use quack_core::llm;
use quack_core::ontology::store as ontology_store;
use quack_core::storage::context;
use quack_core::storage::control::{Outcome, WorkspaceRow};
use quack_core::storage::sessions::{self, ChatMode};
use quack_core::storage::workspace::{StatementKind, WorkspaceDb};
use rmcp::ErrorData as McpError;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, ListResourceTemplatesResult, ListResourcesResult,
    PaginatedRequestParams, ReadResourceRequestParams, ReadResourceResponse, ReadResourceResult,
    Resource, ResourceContents, ResourceTemplate, ServerCapabilities, ServerConfig,
};
use rmcp::service::RequestContext;
use rmcp::{RoleServer, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::server::auth::Access;
use crate::server::state::{App, with_db};
use quack_core::graph::{GraphResult, traverse};

/// Where audit rows go: nowhere for stdio (the CLI is unaudited), or the
/// server's access log and the workspace detail table for HTTP.
pub(crate) enum Auditor {
    None,
    Server(Box<ServerAuditor>),
}

/// The server's audit path: the app for `control.db` and the caller's
/// `Access` for the identity and workspace.
pub(crate) struct ServerAuditor {
    pub app: App,
    pub access: Access,
}

impl Auditor {
    async fn record(
        &self,
        action: &str,
        resource: Option<(&str, &str)>,
        outcome: Outcome,
        detail: Option<serde_json::Value>,
    ) {
        let Self::Server(server) = self else {
            return;
        };
        if let Err(e) = server
            .access
            .audit(&server.app, action, resource, outcome, detail)
            .await
        {
            tracing::warn!(error = %e.message, action, "audit write failed");
        }
    }
}

struct Inner {
    config: Config,
    db: SharedDb,
    workspace: WorkspaceRow,
    policy: WritePolicy,
    /// The server user, for session ownership; `None` over stdio.
    user_id: Option<String>,
    auditor: Auditor,
}

/// One MCP server over one workspace. The tool router comes from the
/// `tool_router` macro (`Self::tool_router()`), which `tool_handler` wires
/// into `list_tools` and `call_tool`.
#[derive(Clone)]
pub(crate) struct McpServer {
    inner: Arc<Inner>,
}

#[derive(Deserialize, JsonSchema)]
pub(crate) struct QueryArgs {
    /// The question, in plain language; the agent searches documents, runs
    /// SQL, and answers with citations.
    pub question: String,
    /// A session to continue, from an earlier answer's `session_id`;
    /// omit to start a new one.
    pub session_id: Option<String>,
    /// `chat` (general knowledge allowed, the default for a new session)
    /// or `query` (every claim from the workspace). Given with
    /// `session_id`, it changes that session's mode.
    pub mode: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub(crate) struct SearchArgs {
    /// Words or a phrase to find in the documents.
    pub query: String,
    /// Chunks to return (default from the workspace configuration).
    pub top_k: Option<u32>,
}

#[derive(Deserialize, JsonSchema)]
pub(crate) struct SqlArgs {
    /// One `DuckDB` statement over the workspace tables.
    pub sql: String,
}

#[derive(Deserialize, JsonSchema)]
pub(crate) struct SearchGraphArgs {
    /// The entity to start from; omit to list every entity of `class`.
    pub entity: Option<String>,
    /// An ontology class id: the entry point's class, or the class to list.
    pub class: Option<String>,
    /// Follow only this relation id.
    pub relation: Option<String>,
    /// Hops out from the entity (default 2).
    pub hops: Option<u32>,
}

#[derive(Deserialize, JsonSchema)]
pub(crate) struct FindPathArgs {
    pub from: String,
    pub to: String,
    /// Longest path to consider (default 4).
    pub max_hops: Option<u32>,
}

#[derive(Deserialize, JsonSchema)]
pub(crate) struct DescribeTableArgs {
    /// A table name from `list_tables`.
    pub table: String,
}

/// The session a turn runs in, and whether this call made it.
struct ResolvedSession {
    id: String,
    created: bool,
}

fn internal(e: impl std::fmt::Display) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

fn failure(message: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(message)])
}

const RESOURCE_TABLES: &str = "quack://workspace/tables";
const RESOURCE_DOCUMENTS: &str = "quack://workspace/documents";
const RESOURCE_ONTOLOGY: &str = "quack://workspace/ontology";
const RESOURCE_CONTEXT: &str = "quack://workspace/context";
const RESOURCE_SCHEMA_TEMPLATE: &str = "quack://workspace/tables/{name}/schema";

#[tool_router]
impl McpServer {
    pub(crate) fn new(
        config: Config,
        db: SharedDb,
        workspace: WorkspaceRow,
        policy: WritePolicy,
        user_id: Option<String>,
        auditor: Auditor,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                config,
                db,
                workspace,
                policy,
                user_id,
                auditor,
            }),
        }
    }

    async fn db<T, F>(&self, f: F) -> Result<T, McpError>
    where
        T: Send + 'static,
        F: FnOnce(&WorkspaceDb) -> quack_core::error::Result<T> + Send + 'static,
    {
        with_db(Arc::clone(&self.inner.db), f)
            .await
            .map_err(|e| internal(e.message))
    }

    /// Ask the workspace agent one question. Searches the documents and
    /// runs SQL as needed; answers with numbered citations.
    #[tool(
        name = "query",
        description = "Ask the workspace's agent a question in plain language. It searches the documents, runs SQL over the tables, and answers with numbered citations and, when useful, a chart. Each call starts a new session unless `session_id` names one from an earlier answer; `mode` is `chat` or `query`."
    )]
    async fn query(
        &self,
        Parameters(args): Parameters<QueryArgs>,
    ) -> Result<CallToolResult, McpError> {
        let question = args.question.trim().to_owned();
        if question.is_empty() {
            return Ok(failure("question must not be empty"));
        }
        let mode = match args.mode.as_deref().map(str::trim) {
            None | Some("") => None,
            Some(text) => match ChatMode::parse(text) {
                Some(mode) => Some(mode),
                None => return Ok(failure("mode must be chat or query")),
            },
        };
        let session = match self.resolve_session(args.session_id, mode).await? {
            Ok(session) => session,
            Err(message) => return Ok(failure(message)),
        };
        let session_id = session.id;
        let (sink, mut events) = events::channel();
        // Nothing renders the stream here; drain it so the turn never
        // blocks on a full channel.
        let drain = tokio::spawn(async move { while events.recv().await.is_some() {} });
        let outcome = llm::run_turn(
            &self.inner.config,
            Arc::clone(&self.inner.db),
            &session_id,
            self.inner.policy,
            &question,
            sink,
            llm::CancellationToken::new(),
        )
        .await;
        drop(drain);
        let detail = serde_json::json!({ "prompt": question, "session_id": session_id });
        match outcome {
            Ok(response) => {
                self.inner
                    .auditor
                    .record(
                        "query",
                        Some(("session", &session_id)),
                        Outcome::Allowed,
                        Some(detail),
                    )
                    .await;
                let citations: Vec<serde_json::Value> = response
                    .citations
                    .iter()
                    .map(|c| {
                        serde_json::json!({
                            "n": c.n,
                            "label": c.label(),
                            "document_id": c.document_id,
                            "filename": c.filename,
                            "page": c.page,
                            "heading": c.heading,
                        })
                    })
                    .collect();
                let mut text = response.content.clone();
                if !response.citations.is_empty() {
                    text.push_str("\n\nSources:\n");
                    for c in &response.citations {
                        text.push_str("  [");
                        text.push_str(&c.n.to_string());
                        text.push_str("] ");
                        text.push_str(&c.label());
                        text.push('\n');
                    }
                }
                if response.write_refused {
                    text.push_str(
                        "\n(A mutating statement was refused: this connection cannot write.)",
                    );
                }
                let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
                result.structured_content = Some(serde_json::json!({
                    "answer": response.content,
                    "citations": citations,
                    "chart": response.chart,
                    "graph": response.graph,
                    "steps": response.steps,
                    "session_id": session_id,
                    "write_refused": response.write_refused,
                }));
                Ok(result)
            }
            Err(e) => {
                self.inner
                    .auditor
                    .record(
                        "query",
                        Some(("session", &session_id)),
                        Outcome::Error,
                        Some(detail),
                    )
                    .await;
                // A turn that never recorded a message leaves no session
                // behind; the next call starts afresh.
                if session.created {
                    let sid = session_id.clone();
                    drop(self.db(move |db| sessions::delete_if_empty(db, &sid)).await);
                }
                Ok(failure(format!("the agent turn failed: {e}")))
            }
        }
    }

    /// Hybrid retrieval over the documents, no model in the loop.
    #[tool(
        name = "search",
        description = "Find the most relevant document chunks for a query by meaning and by keyword. Returns chunks with their file, page, heading, and score; no model is called."
    )]
    async fn search(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, McpError> {
        let query = args.query.trim().to_owned();
        if query.is_empty() {
            return Ok(failure("query must not be empty"));
        }
        let top_k = args
            .top_k
            .unwrap_or(self.inner.config.retrieval.top_k)
            .clamp(1, 100);
        let rrf_k = self.inner.config.retrieval.rrf_k;
        let embedding = match llm::optional_embedding_model(&self.inner.config).await {
            Ok(Some(model)) => match llm::embed_query(&model, &query).await {
                Ok(vector) => Some(vector),
                Err(e) => return Ok(failure(format!("embedding failed: {e}"))),
            },
            Ok(None) => None,
            Err(e) => return Ok(failure(format!("embedding provider unavailable: {e}"))),
        };
        let text = query.clone();
        let hits = self
            .db(move |db| {
                let none: [String; 0] = [];
                match embedding.as_deref() {
                    Some(vector) => db.search_hybrid_chunks(&text, vector, top_k, rrf_k, &none),
                    None => db.search_keyword_chunks(&text, top_k, &none),
                }
            })
            .await?;
        self.inner
            .auditor
            .record(
                "search",
                None,
                Outcome::Allowed,
                Some(serde_json::json!({ "q": query })),
            )
            .await;
        Ok(CallToolResult::structured(
            serde_json::json!({ "chunks": hits }),
        ))
    }

    /// Run one SQL statement. Reads always run; writes need this
    /// connection's write permission.
    #[tool(
        name = "sql",
        description = "Run one DuckDB SQL statement over the workspace tables and return the rows. Reads always run; statements that modify data need write permission on this connection."
    )]
    async fn sql(&self, Parameters(args): Parameters<SqlArgs>) -> Result<CallToolResult, McpError> {
        let statement = args.sql.trim().to_owned();
        if statement.is_empty() {
            return Ok(failure("sql must not be empty"));
        }
        let sql = statement.clone();
        let kind = match self.db(move |db| db.classify_user_statement(&sql)).await {
            Ok(kind) => kind,
            Err(e) => return Ok(failure(e.message)),
        };
        let detail = serde_json::json!({ "sql": statement });
        let is_write = match kind {
            StatementKind::Read => false,
            StatementKind::Write => true,
            StatementKind::Invalid(message) => return Ok(failure(message)),
        };
        if is_write && self.inner.policy != WritePolicy::Allow {
            self.inner
                .auditor
                .record("sql", None, Outcome::Denied, Some(detail))
                .await;
            return Ok(failure(
                "this statement modifies data and this connection cannot write",
            ));
        }
        let sql = statement.clone();
        let max_rows = self.inner.config.analysis.max_query_rows;
        let result = self
            .db(move |db| db.execute_query_capped(&sql, max_rows))
            .await;
        let outcome = if result.is_ok() {
            Outcome::Allowed
        } else {
            Outcome::Error
        };
        self.inner
            .auditor
            .record("sql", None, outcome, Some(detail))
            .await;
        let capped = match result {
            Ok(capped) => capped,
            Err(e) => return Ok(failure(e.message)),
        };
        Ok(CallToolResult::structured(serde_json::json!({
            "columns": capped.results.columns,
            "rows": capped.results.rows,
            "row_count": capped.total_rows,
            "truncated": capped.truncated(),
        })))
    }

    #[tool(
        name = "list_tables",
        description = "List the tables in the workspace."
    )]
    async fn list_tables(&self) -> Result<CallToolResult, McpError> {
        let tables = self.db(WorkspaceDb::list_tables).await?;
        Ok(CallToolResult::structured(
            serde_json::json!({ "tables": tables }),
        ))
    }

    #[tool(
        name = "describe_table",
        description = "Columns, types, row count, and three sample rows of a table."
    )]
    async fn describe_table(
        &self,
        Parameters(args): Parameters<DescribeTableArgs>,
    ) -> Result<CallToolResult, McpError> {
        let name = args.table.trim().to_owned();
        match self.describe(&name).await? {
            Some(description) => {
                self.inner
                    .auditor
                    .record("open", Some(("table", &name)), Outcome::Allowed, None)
                    .await;
                Ok(CallToolResult::structured(description))
            }
            None => Ok(failure(format!("no table named '{name}'"))),
        }
    }

    #[tool(
        name = "search_graph",
        description = "Explore the knowledge graph: the entities within a few hops of a named entity (optionally along one relation), or every entity of an ontology class. Nodes and edges come with provenance to the chunk or table row they were extracted from."
    )]
    async fn search_graph(
        &self,
        Parameters(args): Parameters<SearchGraphArgs>,
    ) -> Result<CallToolResult, McpError> {
        let entity = args
            .entity
            .as_deref()
            .map(str::trim)
            .filter(|e| !e.is_empty())
            .map(str::to_owned);
        let class = args
            .class
            .as_deref()
            .map(str::trim)
            .filter(|c| !c.is_empty())
            .map(str::to_owned);
        if entity.is_none() && class.is_none() {
            return Ok(failure(
                "give an entity to start from, a class to list, or both",
            ));
        }
        let embedding = match &entity {
            Some(e) => self.embed(e).await,
            None => None,
        };
        let hops = args.hops.unwrap_or(2).max(1);
        let relation = args.relation.clone();
        let options = self.inner.config.graph.options();
        let detail = serde_json::json!({ "entity": entity, "class": class, "relation": relation, "hops": hops });
        let result = self
            .db(move |db| {
                if let Some(entity) = entity {
                    let roots = traverse::resolve_entry(
                        db,
                        &entity,
                        class.as_deref(),
                        embedding.as_deref(),
                    )?;
                    return traverse::neighborhood(db, &roots, hops, relation.as_deref(), &options);
                }
                let ontology = ontology_store::current(db)?;
                traverse::by_class(
                    db,
                    ontology.as_ref(),
                    class.as_deref().unwrap_or_default(),
                    options.max_nodes,
                    &options,
                )
            })
            .await?;
        self.inner
            .auditor
            .record("graph", None, Outcome::Allowed, Some(detail))
            .await;
        let mut out = CallToolResult::structured(serde_json::to_value(&result).map_err(internal)?);
        out.content = vec![ContentBlock::text(traverse::render_tree(&result))];
        Ok(out)
    }

    #[tool(
        name = "find_path",
        description = "The shortest chain of relations connecting two entities in the knowledge graph, with provenance for every hop."
    )]
    async fn find_path(
        &self,
        Parameters(args): Parameters<FindPathArgs>,
    ) -> Result<CallToolResult, McpError> {
        let from = args.from.trim().to_owned();
        let to = args.to.trim().to_owned();
        if from.is_empty() || to.is_empty() {
            return Ok(failure("both entities are needed"));
        }
        let a = self.embed(&from).await;
        let b = self.embed(&to).await;
        let max_hops = args.max_hops.unwrap_or(4).max(1);
        let options = self.inner.config.graph.options();
        let detail = serde_json::json!({ "from": from, "to": to, "max_hops": max_hops });
        let (from_label, to_label) = (from.clone(), to.clone());
        let result = self
            .db(move |db| {
                let from_nodes = traverse::resolve_entry(db, &from_label, None, a.as_deref())?;
                let to_nodes = traverse::resolve_entry(db, &to_label, None, b.as_deref())?;
                match (from_nodes.first(), to_nodes.first()) {
                    (Some(a), Some(b)) => traverse::path(db, a, b, max_hops, &options),
                    _ => Ok(GraphResult::default()),
                }
            })
            .await?;
        self.inner
            .auditor
            .record("graph", None, Outcome::Allowed, Some(detail))
            .await;
        if result.is_empty() {
            return Ok(failure(format!(
                "no path connects {from} and {to} within {max_hops} hops"
            )));
        }
        let mut out = CallToolResult::structured(serde_json::to_value(&result).map_err(internal)?);
        out.content = vec![ContentBlock::text(traverse::render_tree(&result))];
        Ok(out)
    }

    #[tool(
        name = "list_documents",
        description = "List the ingested documents with their status, title, and source."
    )]
    async fn list_documents(&self) -> Result<CallToolResult, McpError> {
        let documents = self.db(WorkspaceDb::list_documents).await?;
        Ok(CallToolResult::structured(
            serde_json::json!({ "documents": documents }),
        ))
    }
}

impl McpServer {
    /// A label's embedding for fuzzy entity resolution, when a model exists.
    async fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let model = llm::optional_embedding_model(&self.inner.config)
            .await
            .ok()??;
        llm::embed_query(&model, text).await.ok()
    }

    /// The session a `query` call appends to: the requested one when it
    /// exists and the caller may see it (its mode changed when asked),
    /// else a new one owned by the server user when there is one.
    async fn resolve_session(
        &self,
        requested: Option<String>,
        mode: Option<ChatMode>,
    ) -> Result<Result<ResolvedSession, String>, McpError> {
        let user = self.inner.user_id.clone();
        let sees_all = match &self.inner.auditor {
            Auditor::None => true,
            Auditor::Server(server) => server.access.sees_all_sessions(),
        };
        if let Some(id) = requested
            .map(|id| id.trim().to_owned())
            .filter(|id| !id.is_empty())
        {
            let requested_id = id.clone();
            let found = self
                .db(move |db| {
                    let Some(session) = sessions::get_session(db, &id)? else {
                        return Ok(false);
                    };
                    if !sessions::visible_to(&session, user.as_deref().unwrap_or(""), sees_all) {
                        return Ok(false);
                    }
                    if let Some(mode) = mode {
                        sessions::set_session_mode(db, &id, mode)?;
                    }
                    Ok(true)
                })
                .await?;
            return Ok(if found {
                Ok(ResolvedSession {
                    id: requested_id,
                    created: false,
                })
            } else {
                Err(String::from("that session does not exist"))
            });
        }
        let model = self
            .inner
            .config
            .chat_model_ref()
            .map(|m| m.to_string())
            .map_err(internal)?;
        let id = self
            .db(move |db| {
                sessions::create_session(db, &model, mode.unwrap_or_default(), user.as_deref())
                    .map(|s| s.id)
            })
            .await?;
        Ok(Ok(ResolvedSession { id, created: true }))
    }

    async fn describe(&self, name: &str) -> Result<Option<serde_json::Value>, McpError> {
        let table = name.to_owned();
        let described = self
            .db(move |db| {
                if !db.list_tables()?.contains(&table) {
                    return Ok(None);
                }
                db.describe_table(&table).map(Some)
            })
            .await?;
        Ok(described.map(|d| {
            let columns: Vec<serde_json::Value> = d
                .columns
                .iter()
                .map(|c| serde_json::json!({ "name": c.name, "type": c.column_type }))
                .collect();
            serde_json::json!({
                "table": d.table_name,
                "columns": columns,
                "row_count": d.row_count,
                "sample": { "columns": d.sample_rows.columns, "rows": d.sample_rows.rows },
            })
        }))
    }

    async fn resource_text(&self, uri: &str) -> Result<Option<String>, McpError> {
        if uri == RESOURCE_TABLES {
            let tables = self.db(WorkspaceDb::list_tables).await?;
            return Ok(Some(serde_json::json!({ "tables": tables }).to_string()));
        }
        if uri == RESOURCE_DOCUMENTS {
            let documents = self.db(WorkspaceDb::list_documents).await?;
            return Ok(Some(
                serde_json::json!({ "documents": documents }).to_string(),
            ));
        }
        if uri == RESOURCE_ONTOLOGY {
            let ontology = self.db(ontology_store::current).await?;
            return Ok(Some(match ontology {
                Some(ontology) => ontology.to_json().map_err(internal)?,
                None => String::from("{}"),
            }));
        }
        if uri == RESOURCE_CONTEXT {
            let current = self.db(context::current).await?;
            return Ok(Some(current.map(|c| c.content).unwrap_or_default()));
        }
        if let Some(rest) = uri.strip_prefix("quack://workspace/tables/")
            && let Some(name) = rest.strip_suffix("/schema")
        {
            return Ok(self.describe(name).await?.map(|d| d.to_string()));
        }
        Ok(None)
    }
}

#[tool_handler]
#[expect(
    clippy::unused_async_trait_impl,
    reason = "the tool_handler macro emits async methods without awaits"
)]
impl ServerHandler for McpServer {
    fn get_info(&self) -> ServerConfig {
        let instructions = format!(
            "Workspace '{}': documents, tables, and an ontology behind one agent. Use `query` \
             for questions in plain language (it searches and runs SQL for you and cites \
             sources), `search` for raw retrieval, `sql` for a statement you already know, and \
             the resources for the table list, schemas, documents, ontology, and the owner's \
             context.",
            self.inner.workspace.name
        );
        ServerConfig::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_server_info(
            Implementation::new("quack", env!("CARGO_PKG_VERSION")).with_title("quack"),
        )
        .with_instructions(instructions)
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        let mut resources = vec![
            Resource::new(RESOURCE_TABLES, "tables")
                .with_description("The workspace's tables, as JSON")
                .with_mime_type("application/json"),
            Resource::new(RESOURCE_DOCUMENTS, "documents")
                .with_description("The ingested documents with status, title, and source")
                .with_mime_type("application/json"),
            Resource::new(RESOURCE_ONTOLOGY, "ontology")
                .with_description("The ontology (classes, relations, properties, mappings) as JSON")
                .with_mime_type("application/json"),
            Resource::new(RESOURCE_CONTEXT, "context")
                .with_description(
                    "The owner's instructions and definitions for the agent, as Markdown",
                )
                .with_mime_type("text/markdown"),
        ];
        for table in self.db(WorkspaceDb::list_tables).await? {
            resources.push(
                Resource::new(
                    format!("quack://workspace/tables/{table}/schema"),
                    format!("{table} schema"),
                )
                .with_description(format!("Columns, row count, and sample rows of {table}"))
                .with_mime_type("application/json"),
            );
        }
        Ok(ListResourcesResult::with_all_items(resources))
    }

    fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListResourceTemplatesResult, McpError>> + Send + '_ {
        std::future::ready(Ok(ListResourceTemplatesResult::with_all_items(vec![
            ResourceTemplate::new(RESOURCE_SCHEMA_TEMPLATE, "table schema")
                .with_description("Columns, row count, and sample rows of one table")
                .with_mime_type("application/json"),
        ])))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, McpError> {
        let uri = request.uri;
        let Some(text) = self.resource_text(&uri).await? else {
            return Err(McpError::resource_not_found(
                format!("no resource at {uri}"),
                None,
            ));
        };
        let mime = if uri == RESOURCE_CONTEXT {
            "text/markdown"
        } else {
            "application/json"
        };
        Ok(
            ReadResourceResult::new(vec![ResourceContents::TextResourceContents {
                uri,
                mime_type: Some(String::from(mime)),
                text,
                meta: None,
            }])
            .into(),
        )
    }
}

/// `quack mcp`: serve the workspace over stdio until the client hangs up.
pub(crate) async fn serve_stdio(
    config: Config,
    db: SharedDb,
    workspace: WorkspaceRow,
    policy: WritePolicy,
) -> anyhow::Result<()> {
    let server = McpServer::new(config, db, workspace, policy, None, Auditor::None);
    let running = rmcp::serve_server(server, rmcp::transport::stdio())
        .await
        .map_err(|e| anyhow::anyhow!("MCP initialization failed: {e}"))?;
    running
        .waiting()
        .await
        .map_err(|e| anyhow::anyhow!("MCP server task failed: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use quack_core::config::Config;

    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn server(dir: &std::path::Path, policy: WritePolicy) -> McpServer {
        let mut config = Config::default();
        config.general.data_dir = dir.to_path_buf();
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        McpServer::new(
            config,
            Arc::new(Mutex::new(db)),
            WorkspaceRow {
                id: String::from("ws"),
                name: String::from("stdio"),
                classification: String::from("internal"),
                allowed_providers: None,
            },
            policy,
            None,
            Auditor::None,
        )
    }

    fn field(result: &CallToolResult, key: &str) -> serde_json::Value {
        result
            .structured_content
            .as_ref()
            .and_then(|v| v.get(key))
            .cloned()
            .unwrap_or_default()
    }

    fn error_text(result: &CallToolResult) -> String {
        assert_eq!(result.is_error, Some(true));
        result
            .content
            .iter()
            .filter_map(|c| c.as_text().map(|t| t.text.clone()))
            .collect()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn stdio_tools_gate_writes_and_serve_resources() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let read_only = server(dir.path(), WritePolicy::Deny);
        let denied = read_only
            .sql(Parameters(SqlArgs {
                sql: String::from("CREATE TABLE t AS SELECT 1 AS n"),
            }))
            .await
            .unwrap_or_else(|e| fail(&e.message));
        assert!(error_text(&denied).contains("cannot write"));
        let internal = read_only
            .sql(Parameters(SqlArgs {
                sql: String::from("SELECT * FROM _quack_documents"),
            }))
            .await
            .unwrap_or_else(|e| fail(&e.message));
        assert!(error_text(&internal).contains("internal tables"));
        let bad = read_only
            .sql(Parameters(SqlArgs {
                sql: String::from("SELEC 1"),
            }))
            .await
            .unwrap_or_else(|e| fail(&e.message));
        assert_eq!(bad.is_error, Some(true));

        let writer = server(dir.path(), WritePolicy::Allow);
        let created = writer
            .sql(Parameters(SqlArgs {
                sql: String::from("CREATE TABLE t AS SELECT 1 AS n UNION ALL SELECT 2"),
            }))
            .await
            .unwrap_or_else(|e| fail(&e.message));
        assert_eq!(created.is_error, Some(false));
        let tables = writer
            .list_tables()
            .await
            .unwrap_or_else(|e| fail(&e.message));
        assert_eq!(field(&tables, "tables"), serde_json::json!(["t"]));
        let described = writer
            .describe_table(Parameters(DescribeTableArgs {
                table: String::from("t"),
            }))
            .await
            .unwrap_or_else(|e| fail(&e.message));
        assert_eq!(field(&described, "row_count"), 2);
        assert_eq!(
            field(&described, "columns")
                .get(0)
                .and_then(|c| c.get("name")),
            Some(&serde_json::json!("n"))
        );
        let missing = writer
            .describe_table(Parameters(DescribeTableArgs {
                table: String::from("zz"),
            }))
            .await
            .unwrap_or_else(|e| fail(&e.message));
        assert!(error_text(&missing).contains("no table"));
        let documents = writer
            .list_documents()
            .await
            .unwrap_or_else(|e| fail(&e.message));
        assert_eq!(field(&documents, "documents"), serde_json::json!([]));
        let empty = writer
            .search(Parameters(SearchArgs {
                query: String::from("  "),
                top_k: None,
            }))
            .await
            .unwrap_or_else(|e| fail(&e.message));
        assert_eq!(empty.is_error, Some(true));
    }

    /// `query` with a model that cannot answer: the failure is reported,
    /// the session it made is gone, and the next call is not stuck on a
    /// deleted session id. A session the caller named survives the
    /// failure, and a session nobody made is refused.
    #[tokio::test(flavor = "multi_thread")]
    async fn query_failures_leave_no_session_and_named_sessions_are_checked() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = Config::parse(
            "[general]\nchat_model = \"o/m\"\n[providers.o]\ntype = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\n",
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
        config.general.data_dir = dir.path().to_path_buf();
        let db = WorkspaceDb::open(&config, "ws").unwrap_or_else(|e| fail(&e.to_string()));
        let db = Arc::new(Mutex::new(db));
        let server = McpServer::new(
            config,
            Arc::clone(&db),
            WorkspaceRow {
                id: String::from("ws"),
                name: String::from("stdio"),
                classification: String::from("internal"),
                allowed_providers: None,
            },
            WritePolicy::Deny,
            None,
            Auditor::None,
        );
        let ask = |session_id: Option<&str>, mode: Option<&str>| {
            Parameters(QueryArgs {
                question: String::from("how many?"),
                session_id: session_id.map(str::to_owned),
                mode: mode.map(str::to_owned),
            })
        };
        let session_count = || {
            let guard = db.lock().unwrap_or_else(|e| fail(&e.to_string()));
            sessions::list_sessions(&guard, 10)
                .unwrap_or_else(|e| fail(&e.to_string()))
                .len()
        };

        for _ in 0..2 {
            let failed = server
                .query(ask(None, None))
                .await
                .unwrap_or_else(|e| fail(&e.message));
            let text = error_text(&failed);
            assert!(text.contains("the agent turn failed"), "{text}");
            assert!(!text.contains("does not exist"), "{text}");
            assert_eq!(session_count(), 0);
        }

        let bad_mode = server
            .query(ask(None, Some("loud")))
            .await
            .unwrap_or_else(|e| fail(&e.message));
        assert!(error_text(&bad_mode).contains("chat or query"));

        let unknown = server
            .query(ask(Some("nope"), None))
            .await
            .unwrap_or_else(|e| fail(&e.message));
        assert!(error_text(&unknown).contains("does not exist"));

        let existing = {
            let guard = db.lock().unwrap_or_else(|e| fail(&e.to_string()));
            sessions::create_session(&guard, "o/m", ChatMode::Chat, None)
                .unwrap_or_else(|e| fail(&e.to_string()))
                .id
        };
        let failed = server
            .query(ask(Some(&existing), Some("query")))
            .await
            .unwrap_or_else(|e| fail(&e.message));
        assert!(error_text(&failed).contains("the agent turn failed"));
        let guard = db.lock().unwrap_or_else(|e| fail(&e.to_string()));
        let kept =
            sessions::get_session(&guard, &existing).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(kept.map(|s| s.mode), Some(ChatMode::Query));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn stdio_resources_render_tables_context_and_schemas() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let writer = server(dir.path(), WritePolicy::Allow);
        let created = writer
            .sql(Parameters(SqlArgs {
                sql: String::from("CREATE TABLE t AS SELECT 1 AS n UNION ALL SELECT 2"),
            }))
            .await
            .unwrap_or_else(|e| fail(&e.message));
        assert_eq!(created.is_error, Some(false));
        assert_eq!(
            writer
                .resource_text(RESOURCE_TABLES)
                .await
                .unwrap_or_else(|e| fail(&e.message))
                .as_deref(),
            Some("{\"tables\":[\"t\"]}")
        );
        assert_eq!(
            writer
                .resource_text(RESOURCE_CONTEXT)
                .await
                .unwrap_or_else(|e| fail(&e.message))
                .as_deref(),
            Some("")
        );
        assert!(
            writer
                .resource_text("quack://workspace/tables/t/schema")
                .await
                .unwrap_or_else(|e| fail(&e.message))
                .is_some_and(|t| t.contains("\"row_count\":2"))
        );
        assert_eq!(
            writer
                .resource_text("quack://workspace/tables/zz/schema")
                .await
                .unwrap_or_else(|e| fail(&e.message)),
            None
        );
        assert_eq!(
            writer
                .resource_text("quack://elsewhere")
                .await
                .unwrap_or_else(|e| fail(&e.message)),
            None
        );
        let info = writer.get_info();
        assert!(info.instructions.is_some_and(|i| i.contains("'stdio'")));
    }
}
