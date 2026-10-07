//! The workspace as an MCP server: the same tools over stdio (`quack mcp`,
//! for Claude Code and editors) and over streamable HTTP under `quack
//! serve` (`/mcp/v1/{workspace}`, with the workspace token). Design doc
//! section 11.3.
//!
//! Tools: `query` (one agent turn), `search` (hybrid retrieval, no model),
//! `sql` (classified, writes need permission), `list_tables`,
//! `describe_table`, `list_documents`. Resources: `quack://workspace/tables`,
//! `quack://workspace/tables/{name}/schema`, `quack://workspace/documents`,
//! `quack://workspace/ontology`, `quack://workspace/ontology/schema`,
//! `quack://workspace/context`.
//!
//! Over HTTP every call is audited through the request's `Access`, like
//! the REST API; over stdio nothing is audited, like the CLI.

use std::sync::Arc;

use axum::http::request::Parts;

use quack_core::analysis::citations::Sources;
use quack_core::analysis::events;
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::search::{DocumentSearch, SearchDetail};
use quack_core::analysis::tools::{FindPathArgs, ReaderDb, Rerank, SearchGraphArgs, SharedDb};
use quack_core::config::Config;
use quack_core::ids::{SessionId, UserId};
use quack_core::llm::acting::Acting;
use quack_core::llm::egress::Egress;
use quack_core::llm::{self, Embeddings};
use quack_core::ontology::{Ontology, store as ontology_store};
use quack_core::storage::context;
use quack_core::storage::control::{
    AuditAction, AuditResource, Outcome, ResourceKind, WorkspaceRow,
};
use quack_core::storage::profile::TableProfile;
use quack_core::storage::sessions::{self, ChatMode, SessionViewer};
use quack_core::storage::workspace::{
    DocumentFilter, SearchMode, TEMP_OBJECT_REFUSED, WorkspaceDb, creates_temp_object,
};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock, Extensions, Implementation, ListResourceTemplatesResult,
    ListResourcesResult, PaginatedRequestParams, ReadResourceRequestParams, ReadResourceResponse,
    ReadResourceResult, Resource, ResourceContents, ResourceTemplate, ServerCapabilities,
    ServerConfig,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::server::auth::Access;
use crate::server::state::{App, with_db};
use quack_core::error::{Error as CoreError, Result as CoreResult};
use quack_core::graph::query::PathQuery;

/// Where audit rows go: nowhere for stdio (the CLI is unaudited), or the
/// server's access log and the workspace detail table for HTTP.
pub(crate) enum Auditor {
    None,
    Server(App),
}

/// Who made one MCP request over HTTP: the `Access` `/mcp/v1/{workspace}`
/// resolved for that request, and whom its model requests act for. The
/// endpoint puts it in the HTTP request's extensions, and rmcp hands the
/// request's `http::request::Parts` to the tool call, so every call audits
/// and acts as the request that made it. The transport and server are
/// shared by every request of one user (issue #245): nothing per-request
/// lives on them.
#[derive(Clone)]
pub(crate) struct McpCaller {
    pub access: Access,
    pub acting: Option<Acting>,
}

/// The caller of one tool call or resource read.
enum Caller {
    /// Stdio: nobody to audit or act for.
    Unaudited,
    /// HTTP: the request's own caller, and the app to audit through.
    Audited { app: App, caller: Box<McpCaller> },
}

impl Caller {
    /// Write the audit rows; a failed write fails the call, so nothing
    /// audited proceeds unlogged.
    async fn record(
        &self,
        action: AuditAction,
        resource: Option<AuditResource<'_>>,
        outcome: Outcome,
        detail: Option<serde_json::Value>,
    ) -> Result<(), McpError> {
        let Self::Audited { app, caller } = self else {
            return Ok(());
        };
        caller
            .access
            .audit(app, action.clone(), resource, outcome, detail)
            .await
            .map_err(|e| {
                tracing::error!(error = %e.message, %action, "audit write failed");
                internal(format!("audit write failed: {}", e.message))
            })?;
        Ok(())
    }

    /// Which sessions the caller may read: every one over stdio, and the
    /// server user's over HTTP.
    fn session_viewer(&self) -> SessionViewer {
        match self {
            Self::Unaudited => SessionViewer::All,
            Self::Audited { caller, .. } => caller.access.session_viewer(),
        }
    }

    /// What cuts a `query` turn short: over HTTP, the server stopping.
    /// `quack mcp` on stdio has no stop signal (it ends with its input),
    /// so its token is never cancelled.
    fn cancel(&self) -> llm::CancellationToken {
        match self {
            Self::Unaudited => llm::CancellationToken::new(),
            Self::Audited { app, .. } => app.stopping.child_token(),
        }
    }

    /// Whom model requests are made for (over HTTP, the request's user at
    /// an on-behalf-of provider; over stdio, nobody).
    fn acting(&self) -> Option<Acting> {
        match self {
            Self::Unaudited => None,
            Self::Audited { caller, .. } => caller.acting.clone(),
        }
    }

    /// The tool result when `error` kept a model from being built, recorded
    /// as denied when the workspace's provider allow-list refused it and as
    /// an error otherwise.
    async fn unbuilt(
        &self,
        action: AuditAction,
        detail: serde_json::Value,
        error: &CoreError,
    ) -> Result<CallToolResult, McpError> {
        self.record(action, None, Outcome::of_failure(error), Some(detail))
            .await?;
        Ok(failure(error.to_string()))
    }
}

/// What an MCP server serves: the configuration, the workspace and its
/// handles, the write policy, the server user (for session ownership;
/// `None` over stdio), and where audit rows go.
pub(crate) struct McpSetup {
    pub config: Config,
    pub db: SharedDb,
    pub reader: ReaderDb,
    pub workspace: WorkspaceRow,
    pub policy: WritePolicy,
    pub user_id: Option<UserId>,
    pub auditor: Auditor,
}

/// One MCP server over one workspace. The tool router comes from the
/// `tool_router` macro (`Self::tool_router()`), which `tool_handler` wires
/// into `list_tools` and `call_tool`.
#[derive(Clone)]
pub(crate) struct McpServer {
    inner: Arc<McpSetup>,
}

#[derive(Deserialize, JsonSchema)]
pub(crate) struct QueryArgs {
    /// The question, in plain language; the agent searches documents, runs
    /// SQL, and answers with citations.
    pub question: String,
    /// A session to continue, from an earlier answer's `session_id`;
    /// omit to start a new one.
    pub session_id: Option<String>,
    /// `chat` (general knowledge allowed, the default) or `query` (every
    /// claim from the workspace), for a new session; an existing session
    /// keeps the mode it was created with.
    #[schemars(with = "Option<ChatMode>")]
    pub mode: Option<String>,
    /// Limit the question to these documents: ids from `list_documents`
    /// (prefixes accepted), exact file names, or exact titles; omit for
    /// every document.
    #[serde(default)]
    pub document_ids: Vec<String>,
}

#[derive(Default, Deserialize, JsonSchema)]
pub(crate) struct SearchArgs {
    /// Words or a phrase to find in the documents; a "quoted phrase" must
    /// appear exactly.
    pub query: String,
    /// Chunks to return (default from the workspace configuration).
    pub top_k: Option<u32>,
    /// Search only these documents: ids from `list_documents` (prefixes
    /// accepted), exact file names, or exact titles.
    #[serde(default)]
    pub document_ids: Vec<String>,
    /// Search only the passages this knowledge-graph entity was extracted
    /// from.
    pub entity: Option<String>,
    /// Search only documents of these types, sources, or tags, written in a
    /// date range, or by an author.
    #[serde(default)]
    pub filters: DocumentFilter,
    /// `hybrid` (the default), `keyword`, or `vector`.
    pub mode: Option<SearchMode>,
    /// Also return each leg's candidates with their ranks and scores, the
    /// quoted-phrase filter, and the rerank outcome.
    #[serde(default)]
    pub explain: bool,
}

#[derive(Deserialize, JsonSchema)]
pub(crate) struct SqlArgs {
    /// One `DuckDB` statement over the workspace tables.
    pub sql: String,
}

#[derive(Deserialize, JsonSchema)]
pub(crate) struct DescribeTableArgs {
    /// A table name from `list_tables`.
    pub table: String,
}

/// The session a `query` call runs in.
enum TurnSession {
    /// One the caller named, which exists and they may see.
    Existing(SessionId),
    /// A new one this call made; removed again if the turn records
    /// nothing.
    Created(SessionId),
    /// The caller named one that does not exist or they may not see.
    NotFound,
}

fn internal(e: impl std::fmt::Display) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

fn failure(message: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(message)])
}

/// A `quack://workspace/...` resource: five fixed ones and one schema per table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkspaceResource<'a> {
    Tables,
    Documents,
    Ontology,
    OntologySchema,
    Context,
    Schema(&'a str),
}

impl<'a> WorkspaceResource<'a> {
    const PREFIX: &'static str = "quack://workspace/";
    const SCHEMA_TEMPLATE: &'static str = "quack://workspace/tables/{name}/schema";
    const FIXED: [WorkspaceResource<'static>; 5] = [
        WorkspaceResource::Tables,
        WorkspaceResource::Documents,
        WorkspaceResource::Ontology,
        WorkspaceResource::OntologySchema,
        WorkspaceResource::Context,
    ];

    fn parse(uri: &'a str) -> Option<Self> {
        let path = uri.strip_prefix(Self::PREFIX)?;
        match path {
            "tables" => Some(Self::Tables),
            "documents" => Some(Self::Documents),
            "ontology" => Some(Self::Ontology),
            "ontology/schema" => Some(Self::OntologySchema),
            "context" => Some(Self::Context),
            _ => path
                .strip_prefix("tables/")?
                .strip_suffix("/schema")
                .map(Self::Schema),
        }
    }

    /// The resource as `control.db`'s access log names it: a table's
    /// schema by the template, so no table name leaves the workspace.
    fn audit_id(self) -> String {
        match self {
            Self::Schema(_) => String::from(Self::SCHEMA_TEMPLATE),
            Self::Tables
            | Self::Documents
            | Self::Ontology
            | Self::OntologySchema
            | Self::Context => self.uri(),
        }
    }

    fn uri(self) -> String {
        match self {
            Self::Schema(table) => format!("{}tables/{table}/schema", Self::PREFIX),
            Self::OntologySchema => format!("{}ontology/schema", Self::PREFIX),
            Self::Tables | Self::Documents | Self::Ontology | Self::Context => {
                format!("{}{}", Self::PREFIX, self.name())
            }
        }
    }

    fn name(self) -> String {
        match self {
            Self::Tables => String::from("tables"),
            Self::Documents => String::from("documents"),
            Self::Ontology => String::from("ontology"),
            Self::OntologySchema => String::from("ontology schema"),
            Self::Context => String::from("context"),
            Self::Schema(table) => format!("{table} schema"),
        }
    }

    fn description(self) -> String {
        match self {
            Self::Tables => String::from("The workspace's tables, as JSON"),
            Self::Documents => {
                String::from("The ingested documents with status, title, and source")
            }
            Self::Ontology => {
                String::from("The ontology (classes, relations, properties, mappings) as JSON")
            }
            Self::OntologySchema => String::from(
                "The JSON Schema of the ontology's interchange form, for writing one to import",
            ),
            Self::Context => {
                String::from("The owner's instructions and definitions for the agent, as Markdown")
            }
            Self::Schema(table) => format!("Columns, row count, and sample rows of {table}"),
        }
    }

    fn mime_type(self) -> &'static str {
        match self {
            Self::Context => "text/markdown",
            Self::Tables
            | Self::Documents
            | Self::Ontology
            | Self::OntologySchema
            | Self::Schema(_) => "application/json",
        }
    }

    fn listing(self) -> Resource {
        Resource::new(self.uri(), self.name())
            .with_description(self.description())
            .with_mime_type(self.mime_type())
    }
}

#[tool_router]
impl McpServer {
    pub(crate) fn new(setup: McpSetup) -> Self {
        Self {
            inner: Arc::new(setup),
        }
    }

    /// The caller of the request `extensions` came with. Over HTTP it must
    /// be there: a call without one fails rather than run unaudited.
    fn caller(&self, extensions: &Extensions) -> Result<Caller, McpError> {
        let Auditor::Server(app) = &self.inner.auditor else {
            return Ok(Caller::Unaudited);
        };
        let caller = extensions
            .get::<Parts>()
            .and_then(|parts| parts.extensions.get::<McpCaller>())
            .cloned()
            .ok_or_else(|| internal("the MCP request carries no authorized caller"))?;
        Ok(Caller::Audited {
            app: Arc::clone(app),
            caller: Box::new(caller),
        })
    }

    async fn db<T, F>(&self, f: F) -> Result<T, McpError>
    where
        T: Send + 'static,
        F: FnOnce(&WorkspaceDb) -> CoreResult<T> + Send + 'static,
    {
        with_db(Arc::clone(&self.inner.db), f)
            .await
            .map_err(|e| internal(e.message))
    }

    /// Like [`Self::db`], but for a tool that only reads: routes through
    /// the workspace's reader connection instead of the writer, so it
    /// never queues behind an ingest.
    async fn reader_db<T, F>(&self, f: F) -> Result<T, McpError>
    where
        T: Send + 'static,
        F: FnOnce(&WorkspaceDb) -> CoreResult<T> + Send + 'static,
    {
        self.inner.reader.with_db(f).await.map_err(internal)
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
        extensions: Extensions,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&extensions)?;
        // Boxed: an agent turn's future is large (clippy::large_futures).
        Box::pin(self.as_caller(&caller, self.ask(args, &caller))).await
    }

    /// `query`, as its caller: its model requests reach an on-behalf-of
    /// provider as the request's user, and only providers the workspace
    /// allows.
    async fn ask(&self, args: QueryArgs, caller: &Caller) -> Result<CallToolResult, McpError> {
        let question = args.question.trim().to_owned();
        if question.is_empty() {
            return Ok(failure("question must not be empty"));
        }
        let mode = match args.mode.as_deref().map(str::trim) {
            None | Some("") => None,
            Some(text) => match text.parse::<ChatMode>() {
                Ok(mode) => Some(mode),
                Err(e) => return Ok(failure(e.to_string())),
            },
        };
        let (session_id, created) =
            match self.resolve_session(caller, args.session_id, mode).await? {
                TurnSession::Existing(id) => (id, false),
                TurnSession::Created(id) => (id, true),
                TurnSession::NotFound => return Ok(failure("that session does not exist")),
            };
        let (sink, mut events) = events::channel();
        // Nothing renders the stream here; drain it so the turn never
        // blocks on a full channel.
        let drain = tokio::spawn(async move { while events.recv().await.is_some() {} });
        let outcome = llm::TurnRequest {
            db: Arc::clone(&self.inner.db),
            reader_db: self.inner.reader.clone(),
            session_id: &session_id,
            policy: self.inner.policy,
            message: &question,
            documents: &args.document_ids,
            sink,
            cancel: caller.cancel(),
        }
        .run(&self.inner.config)
        .await;
        drop(drain);
        let detail = serde_json::json!({
            "prompt": question,
            "session_id": session_id,
            "documents": args.document_ids,
        });
        match outcome {
            Ok(response) => {
                caller
                    .record(
                        AuditAction::Query,
                        Some(ResourceKind::Session.id(&session_id)),
                        Outcome::Allowed,
                        Some(detail),
                    )
                    .await?;
                let mut text = response.content.clone();
                if !response.citations.is_empty() {
                    let sources = format!("\n\n{}\n", Sources(&response.citations));
                    text.push_str(&sources);
                }
                if response.write_refused {
                    // With write access the refusal has another cause (the
                    // turn read document text first, say), which its step
                    // carries.
                    text.push_str(if self.inner.policy.allows_unasked() {
                        "\n(A mutating statement was refused; its step says why.)"
                    } else {
                        "\n(A mutating statement was refused: this connection cannot write.)"
                    });
                }
                let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
                result.structured_content = Some(serde_json::json!(response.body(&session_id)));
                Ok(result)
            }
            Err(e) => {
                caller
                    .record(
                        AuditAction::Query,
                        Some(ResourceKind::Session.id(&session_id)),
                        Outcome::of_failure(&e),
                        Some(detail),
                    )
                    .await?;
                // A turn that never recorded a message leaves no session
                // behind; the next call starts afresh.
                if created {
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
        description = "Find the most relevant document chunks for a query by meaning and by keyword, optionally within named documents, a graph entity's passages, or documents matching a filter. Returns chunks with their file, page, heading, fused score, and rank in each leg; `explain` adds both legs' candidates and the rerank outcome. No chat model is called unless the workspace reranks with it."
    )]
    async fn search(
        &self,
        Parameters(args): Parameters<SearchArgs>,
        extensions: Extensions,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&extensions)?;
        // Boxed: building the model makes the future large.
        Box::pin(self.as_caller(&caller, self.retrieve(args, &caller))).await
    }

    /// `search`, as its caller: embedding the query is a model request too.
    async fn retrieve(
        &self,
        args: SearchArgs,
        caller: &Caller,
    ) -> Result<CallToolResult, McpError> {
        let config = &self.inner.config;
        let search =
            match DocumentSearch::new(&args.query, args.top_k.unwrap_or(config.retrieval.top_k)) {
                Ok(search) => DocumentSearch {
                    documents: args.document_ids,
                    entity: args.entity.filter(|e| !e.trim().is_empty()),
                    filter: args.filters,
                    mode: args.mode.unwrap_or_default(),
                    ..search
                },
                Err(e) => return Ok(failure(e.to_string())),
            };
        let detail = serde_json::json!({ "q": search.query, "search": search.describe() });
        let model = match Embeddings::from_config(config).await {
            Ok(model) => model,
            Err(e) => return caller.unbuilt(AuditAction::Search, detail, &e).await,
        };
        let rerank = match Rerank::from_config(config).await {
            Ok(rerank) => rerank,
            Err(e) => return caller.unbuilt(AuditAction::Search, detail, &e).await,
        };
        // Run the search, audit its outcome, then answer — the way the `sql`
        // tool does, so a post-authorization failure is recorded instead of
        // dropped, while tool failures stay normal tool results.
        let result = search
            .run(
                &self.inner.reader,
                model.as_ref(),
                rerank.as_ref(),
                config.retrieval.rrf_k,
            )
            .await;
        let outcome = match &result {
            Ok(_) => Outcome::Allowed,
            Err(e) => Outcome::of_failure(e),
        };
        caller
            .record(AuditAction::Search, None, outcome, Some(detail))
            .await?;
        match result {
            Ok(found) => Ok(CallToolResult::structured(serde_json::json!(
                found.body(SearchDetail::explained(args.explain))
            ))),
            Err(e) => Ok(failure(e.to_string())),
        }
    }

    /// Run one SQL statement. Reads always run; writes need this
    /// connection's write permission.
    #[tool(
        name = "sql",
        description = "Run one DuckDB SQL statement over the workspace tables and return the rows. Reads always run; statements that modify data need write permission on this connection."
    )]
    async fn sql(
        &self,
        Parameters(args): Parameters<SqlArgs>,
        extensions: Extensions,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&extensions)?;
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
        let is_write = match kind.writes() {
            Ok(is_write) => is_write,
            Err(message) => return Ok(failure(message)),
        };
        if is_write && creates_temp_object(&statement) {
            caller
                .record(AuditAction::Sql, None, Outcome::Denied, Some(detail))
                .await?;
            return Ok(failure(TEMP_OBJECT_REFUSED));
        }
        if is_write && !self.inner.policy.allows_unasked() {
            caller
                .record(AuditAction::Sql, None, Outcome::Denied, Some(detail))
                .await?;
            return Ok(failure(
                "this statement modifies data and this connection cannot write",
            ));
        }
        let sql = statement.clone();
        let max_rows = self.inner.config.analysis.max_query_rows;
        let result = if is_write {
            let result = self
                .db(move |db| {
                    let result = db.execute_query_capped(&sql, max_rows);
                    TableProfile::after_write(db);
                    result
                })
                .await;
            // Whatever ran might have created a temp object the check
            // above did not catch (a leading comment, a multi-statement
            // batch); check the writer's catalog regardless of whether
            // the statement itself errored, since an earlier statement in
            // a batch can have already run.
            self.inner.reader.observe_write().await;
            result
        } else {
            // A read never queues behind a write: run it on the reader
            // connection instead of the writer.
            self.reader_db(move |db| db.execute_query_capped(&sql, max_rows))
                .await
        };
        let outcome = if result.is_ok() {
            Outcome::Allowed
        } else {
            Outcome::Error
        };
        caller
            .record(AuditAction::Sql, None, outcome, Some(detail))
            .await?;
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
        description = "List the names of the user tables in the workspace, as `{\"tables\": [...]}`. quack's internal tables are not listed. Call `describe_table` on a name for its columns, types, row count, and sample rows, and `sql` to query it. Takes no arguments."
    )]
    async fn list_tables(&self, extensions: Extensions) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&extensions)?;
        let tables = self.reader_db(WorkspaceDb::list_tables).await;
        caller
            .record(
                AuditAction::List,
                None,
                Outcome::of(&tables),
                Some(serde_json::json!({ "what": "tables" })),
            )
            .await?;
        let tables = tables?;
        Ok(CallToolResult::structured(
            serde_json::json!({ "tables": tables }),
        ))
    }

    #[tool(
        name = "describe_table",
        description = "A table's columns with their types and, when the owner gave them, their meaning, unit, and synonyms; its row count; the owner's note; its profile (per column: values present, distinct values, common values) with warnings such as numbers stored as text or a key that repeats; the measures defined over it; and three sample rows."
    )]
    async fn describe_table(
        &self,
        Parameters(args): Parameters<DescribeTableArgs>,
        extensions: Extensions,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&extensions)?;
        let name = args.table.trim().to_owned();
        match self.describe(&name).await? {
            Some(description) => {
                caller
                    .record(
                        AuditAction::Open,
                        None,
                        Outcome::Allowed,
                        Some(serde_json::json!({ "table": name })),
                    )
                    .await?;
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
        extensions: Extensions,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&extensions)?;
        Box::pin(self.as_caller(&caller, self.neighborhood(args, &caller))).await
    }

    /// `search_graph`, as its caller: embedding the entity's name is a
    /// model request.
    async fn neighborhood(
        &self,
        args: SearchGraphArgs,
        caller: &Caller,
    ) -> Result<CallToolResult, McpError> {
        let query = match args.query() {
            Ok(query) => query,
            Err(e) => return Ok(failure(e.to_string())),
        };
        let options = self.inner.config.graph;
        let detail = serde_json::to_value(&query).map_err(internal)?;
        let model = match Embeddings::from_config(&self.inner.config).await {
            Ok(model) => model,
            Err(e) => return caller.unbuilt(AuditAction::Graph, detail, &e).await,
        };
        // Run, audit the outcome, then answer, the way `sql` does, so a
        // failure after authorization is recorded rather than dropped.
        let result = async {
            let embedding = query.embedding(model.as_ref()).await.map_err(internal)?;
            // An unknown class or relation id names the real ones.
            self.reader_db(move |db| query.run(db, embedding.as_ref(), &options))
                .await
        }
        .await;
        caller
            .record(AuditAction::Graph, None, Outcome::of(&result), Some(detail))
            .await?;
        let result = match result {
            Ok(result) => result,
            Err(e) => return Ok(failure(e.message)),
        };
        let mut out = CallToolResult::structured(serde_json::to_value(&result).map_err(internal)?);
        out.content = vec![ContentBlock::text(result.to_string())];
        Ok(out)
    }

    #[tool(
        name = "find_path",
        description = "The shortest chain of relations connecting two entities in the knowledge graph, with provenance for every hop."
    )]
    async fn find_path(
        &self,
        Parameters(args): Parameters<FindPathArgs>,
        extensions: Extensions,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&extensions)?;
        Box::pin(self.as_caller(&caller, self.path(args, &caller))).await
    }

    /// `find_path`, as its caller: embedding each end's name is a model
    /// request.
    async fn path(&self, args: FindPathArgs, caller: &Caller) -> Result<CallToolResult, McpError> {
        let query = match args.query() {
            Ok(query) => query,
            Err(e) => return Ok(failure(e.to_string())),
        };
        let options = self.inner.config.graph;
        let detail = serde_json::to_value(&query).map_err(internal)?;
        let PathQuery { from, to, max_hops } = query.clone();
        let model = match Embeddings::from_config(&self.inner.config).await {
            Ok(model) => model,
            Err(e) => return caller.unbuilt(AuditAction::Graph, detail, &e).await,
        };
        // Run, audit the outcome, then answer (see `search_graph`).
        let result = async {
            let ends = query.embeddings(model.as_ref()).await.map_err(internal)?;
            // An end that names no entity comes back with the closest labels.
            self.reader_db(move |db| query.run(db, &ends, &options))
                .await
        }
        .await;
        caller
            .record(AuditAction::Graph, None, Outcome::of(&result), Some(detail))
            .await?;
        let result = match result {
            Ok(result) => result,
            Err(e) => return Ok(failure(e.message)),
        };
        if result.is_empty() {
            return Ok(failure(format!(
                "no path connects {from} and {to} within {max_hops} hops"
            )));
        }
        let mut out = CallToolResult::structured(serde_json::to_value(&result).map_err(internal)?);
        out.content = vec![ContentBlock::text(result.to_string())];
        Ok(out)
    }

    #[tool(
        name = "list_documents",
        description = "List the ingested documents with their status, title, and source."
    )]
    async fn list_documents(&self, extensions: Extensions) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&extensions)?;
        let documents = self.reader_db(WorkspaceDb::list_documents).await;
        caller
            .record(
                AuditAction::List,
                None,
                Outcome::of(&documents),
                Some(serde_json::json!({ "what": "documents" })),
            )
            .await?;
        let documents = documents?;
        Ok(CallToolResult::structured(
            serde_json::json!({ "documents": documents }),
        ))
    }
}

impl McpServer {
    /// Run a tool's `work` as its caller: acting for the request's user,
    /// and sending only to the model providers the workspace allows (over
    /// HTTP, as the request found the workspace; over stdio, as the command
    /// opened it). rmcp runs each call on a task of its own, so neither
    /// scope reaches it from the request.
    async fn as_caller<F: Future>(&self, caller: &Caller, work: F) -> F::Output {
        let workspace = match caller {
            Caller::Unaudited => &self.inner.workspace,
            Caller::Audited { caller, .. } => &caller.access.membership.workspace,
        };
        let egress = Egress::Workspace(workspace.allowed_providers.clone());
        Acting::scope(caller.acting(), Egress::scope(Some(egress), work)).await
    }

    /// The session a `query` call appends to: the requested one when it
    /// exists and the caller may see it, else a new one in `mode` owned
    /// by the server user when there is one.
    async fn resolve_session(
        &self,
        caller: &Caller,
        requested: Option<String>,
        mode: Option<ChatMode>,
    ) -> Result<TurnSession, McpError> {
        let user = self.inner.user_id.clone();
        let viewer = caller.session_viewer();
        if let Some(id) = requested
            .map(|id| id.trim().to_owned())
            .filter(|id| !id.is_empty())
            .map(SessionId::from)
        {
            let requested_id = id.clone();
            let found = self
                .db(move |db| {
                    let Some(session) = sessions::get_session(db, &id)? else {
                        return Ok(false);
                    };
                    if !session.visible_to(&viewer) {
                        return Ok(false);
                    }
                    Ok(true)
                })
                .await?;
            return Ok(if found {
                TurnSession::Existing(requested_id)
            } else {
                TurnSession::NotFound
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
                sessions::create_session(db, &model, mode.unwrap_or_default(), user.as_ref())
                    .map(|s| s.id)
            })
            .await?;
        Ok(TurnSession::Created(id))
    }

    async fn describe(&self, name: &str) -> Result<Option<serde_json::Value>, McpError> {
        let table = name.to_owned();
        let described = self
            .reader_db(move |db| {
                if !db.list_tables()?.contains(&table) {
                    return Ok(None);
                }
                db.describe_table(&table).map(Some)
            })
            .await?;
        Ok(described.map(|d| serde_json::json!(d.body())))
    }

    async fn resource_text(
        &self,
        resource: WorkspaceResource<'_>,
    ) -> Result<Option<String>, McpError> {
        Ok(Some(match resource {
            WorkspaceResource::Tables => {
                let tables = self.reader_db(WorkspaceDb::list_tables).await?;
                serde_json::json!({ "tables": tables }).to_string()
            }
            WorkspaceResource::Documents => {
                let documents = self.reader_db(WorkspaceDb::list_documents).await?;
                serde_json::json!({ "documents": documents }).to_string()
            }
            WorkspaceResource::Ontology => match self.reader_db(ontology_store::current).await? {
                Some(ontology) => ontology.to_json().map_err(internal)?,
                None => String::from("{}"),
            },
            WorkspaceResource::OntologySchema => {
                serde_json::to_string(&Ontology::json_schema()).map_err(internal)?
            }
            WorkspaceResource::Context => {
                let current = self.reader_db(context::current).await?;
                current.map(|c| c.content).unwrap_or_default()
            }
            WorkspaceResource::Schema(table) => {
                return Ok(self.describe(table).await?.map(|d| d.to_string()));
            }
        }))
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
        let mut resources: Vec<Resource> = WorkspaceResource::FIXED
            .into_iter()
            .map(WorkspaceResource::listing)
            .collect();
        for table in self.reader_db(WorkspaceDb::list_tables).await? {
            resources.push(WorkspaceResource::Schema(&table).listing());
        }
        Ok(ListResourcesResult::with_all_items(resources))
    }

    fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListResourceTemplatesResult, McpError>> + Send + '_ {
        std::future::ready(Ok(ListResourceTemplatesResult::with_all_items(vec![
            ResourceTemplate::new(WorkspaceResource::SCHEMA_TEMPLATE, "table schema")
                .with_description("Columns, row count, and sample rows of one table")
                .with_mime_type("application/json"),
        ])))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, McpError> {
        let caller = self.caller(&context.extensions)?;
        let uri = request.uri;
        let parsed = WorkspaceResource::parse(&uri);
        // The access log takes the resource's template, the workspace's own
        // audit detail the full URI: a table name is workspace content.
        let audit_id = parsed.map(WorkspaceResource::audit_id);
        caller
            .record(
                AuditAction::Open,
                audit_id.as_deref().map(|id| ResourceKind::Resource.id(id)),
                Outcome::Allowed,
                Some(serde_json::json!({ "uri": uri })),
            )
            .await?;
        let not_found = || McpError::resource_not_found(format!("no resource at {uri}"), None);
        let resource = parsed.ok_or_else(not_found)?;
        let text = self.resource_text(resource).await?.ok_or_else(not_found)?;
        let mime = resource.mime_type();
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
    reader: ReaderDb,
    workspace: WorkspaceRow,
    policy: WritePolicy,
) -> anyhow::Result<()> {
    let server = McpServer::new(McpSetup {
        config,
        db,
        reader,
        workspace,
        policy,
        user_id: None,
        auditor: Auditor::None,
    });
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
mod tests;
