# Quack: Data Analysis Platform - MVP Design Document

## 1. What This Is

A single Rust binary that combines a DuckDB-powered analytical engine, a document ingestion and retrieval pipeline, and an optional graph layer into one tool that runs in three modes:

- `quack tui` - terminal UI (ratatui) for interactive exploration over SSH or local terminal
- `quack serve` - web server (axum + askama + htmx) exposing a browser UI, REST API, MCP server, and OpenAI-compatible chat endpoint
- `quack query "SELECT ..." --workspace foo --format json` - non-interactive CLI for scripts and pipelines

The tool must work in air-gapped environments with no network access, using local LLMs via Ollama. In cloud deployments it connects to hosted LLM providers (OpenAI, Anthropic, Bedrock, Azure OpenAI). The hosted version adds multi-user workspaces with membership and data classification labels.

This document specifies the MVP scope. It is written for an engineering agent that will implement it.

---

## 2. Architecture Overview

```
+------------------------------------------------------------------+
|                        Binary Entrypoints                         |
|  quack tui        quack serve             quack query             |
|  (ratatui)        (axum/askama/htmx)      (CLI, pipe-friendly)   |
+------------------------------------------------------------------+
          |                |                       |
          v                v                       v
+------------------------------------------------------------------+
|                        Core Library Crate                         |
|                                                                   |
|  +------------------+  +------------------+  +-----------------+  |
|  | Ingestion        |  | Analysis Engine  |  | LLM Client      |  |
|  | - file parsing   |  | - text-to-SQL    |  | - provider trait |  |
|  | - chunking       |  | - SQL execution  |  | - OpenAI-compat |  |
|  | - embedding      |  | - RAG retrieval  |  | - Anthropic     |  |
|  | - table extract  |  | - chart specs    |  | - Ollama        |  |
|  +------------------+  +------------------+  +-----------------+  |
|                                                                   |
|  +------------------+  +------------------+  +-----------------+  |
|  | Storage          |  | Graph Layer      |  | Control Plane   |  |
|  | - DuckDB wrapper |  | - node/edge tbls |  | - workspaces    |  |
|  | - workspace iso  |  | - entity extract |  | - members/roles |  |
|  | - vector search  |  | - path queries   |  | - classification|  |
|  +------------------+  +------------------+  | - audit log     |  |
|                                              | - API tokens    |  |
|                                              +-----------------+  |
+------------------------------------------------------------------+
          |
          v
+------------------------------------------------------------------+
|  DuckDB (bundled, statically linked via duckdb-rs)               |
|  Extensions: vss (vector similarity), duckpgq (graph, optional)  |
+------------------------------------------------------------------+
```

The core library crate contains all business logic. Each entrypoint (TUI, web server, CLI) is a thin adapter that calls into the core. There is no network hop between the TUI/CLI and the engine - they run in-process.

---

## 3. Crate Structure

```
quack/
  Cargo.toml              # workspace root
  crates/
    quack-core/            # library crate - all business logic
      src/
        lib.rs
        storage/           # DuckDB wrapper, workspace isolation
          mod.rs
          workspace.rs     # create/open/attach workspace databases
          migrations.rs    # schema versioning
          queries.rs       # SeaQuery builders for all internal queries
        ingestion/         # file parsing, chunking, embedding
          mod.rs
          parser.rs        # PDF, DOCX, CSV, Parquet, Excel, plain text
          chunker.rs       # text splitting with overlap
          embedder.rs      # embedding via LLM provider
          table_extractor.rs  # structured data from tabular files
        analysis/          # query engine and agent loop
          mod.rs
          text_to_sql.rs   # schema-aware SQL generation
          rag.rs           # vector retrieval pipeline
          agent.rs         # tool-calling loop (retrieve, SQL, chart)
          chart.rs         # ECharts option spec generation
        graph/             # knowledge graph layer
          mod.rs
          schema.rs        # node/edge table definitions
          extract.rs       # entity/relationship extraction via LLM
          query.rs         # path queries, N-hop traversal
        llm/               # LLM provider abstraction
          mod.rs
          provider.rs      # trait definition
          openai_compat.rs # covers OpenAI, Ollama, vLLM, most gateways
          anthropic.rs     # native Anthropic API
          oauth.rs         # OAuth 2.0 PKCE flow + device code fallback
          token_manager.rs # token caching, refresh, concurrency
        control/           # workspace/user/auth management
          mod.rs
          workspace.rs
          auth.rs          # local mode: no auth. serve mode: tokens + OIDC
          audit.rs         # append-only log of all operations
        config.rs          # unified config (TOML file + env vars)
        error.rs           # shared error types

    quack-tui/             # binary crate - terminal interface
      src/
        main.rs
        app.rs             # application state and event loop
        panels/
          chat.rs          # chat panel with streaming
          sql_editor.rs    # SQL input with syntax hints
          results.rs       # scrollable table for query results
          graph.rs         # ASCII graph/tree view
          status.rs        # mode indicator, progress, errors

    quack-serve/           # binary crate - web server
      src/
        main.rs
        routes/
          ui.rs            # askama + htmx page handlers
          api.rs           # REST API (JSON)
          mcp.rs           # MCP server endpoint
          openai_compat.rs # /v1/chat/completions per workspace
        templates/         # askama HTML templates
        static/            # htmx.js, echarts.min.js, tailwind output (vendored)

    quack-cli/             # binary crate - non-interactive query
      src/
        main.rs            # parse args, run query, print results
```

---

## 4. Storage Design

### 4.1 Control Plane Database

A single SQLite database at `<data_dir>/control.db` (configurable) holds all metadata. The default data directory follows XDG conventions: `~/.local/share/quack/` on all platforms (or `$XDG_DATA_HOME/quack/` on Linux). Override with `QUACK_DATA_DIR`. SQLite is chosen over Postgres for MVP because it works identically in local, containerized, and air-gapped deployments with zero dependencies.

```sql
-- schema version tracking
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE workspaces (
    id TEXT PRIMARY KEY,            -- ULID
    name TEXT NOT NULL,
    classification TEXT NOT NULL DEFAULT 'internal',
    allowed_providers TEXT,         -- JSON array, NULL = all
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE members (
    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
    user_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',  -- owner | member | viewer
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (workspace_id, user_id)
);

CREATE TABLE api_tokens (
    token_hash TEXT PRIMARY KEY,    -- SHA-256 of the token
    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT
);

CREATE TABLE threads (
    id TEXT PRIMARY KEY,            -- ULID
    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
    title TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,            -- ULID
    thread_id TEXT NOT NULL REFERENCES threads(id),
    role TEXT NOT NULL,             -- user | assistant | system | tool
    content TEXT NOT NULL,
    metadata TEXT,                  -- JSON: generated SQL, chart spec, sources
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    workspace_id TEXT NOT NULL,
    user_id TEXT,
    action TEXT NOT NULL,           -- query | ingest | export | admin
    detail TEXT                     -- JSON: SQL executed, files uploaded, etc.
);
```

### 4.2 Workspace Data Database

Each workspace gets its own DuckDB file at `<data_dir>/workspaces/{workspace_id}/data.duckdb`. This provides hard isolation - no query can cross workspace boundaries.

Each workspace database contains these tables:

```sql
-- Uploaded/ingested documents metadata
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    mime_type TEXT,
    size_bytes BIGINT,
    ingested_at TIMESTAMP DEFAULT now(),
    status TEXT DEFAULT 'pending',   -- pending | processing | ready | error
    error_message TEXT
);

-- Text chunks from unstructured documents
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding FLOAT[],              -- via DuckDB vss extension
    token_count INTEGER
);

-- Index for vector similarity search
-- Created after first embedding insert:
-- CREATE INDEX chunks_embedding_idx ON chunks USING HNSW (embedding)
--   WITH (metric = 'cosine');

-- Graph nodes (created when graph features are enabled)
CREATE TABLE graph_nodes (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    type TEXT NOT NULL,              -- person | org | concept | event | etc.
    properties JSON,
    source_document_id TEXT,
    source_chunk_id TEXT,
    embedding FLOAT[]
);

-- Graph edges
CREATE TABLE graph_edges (
    id TEXT PRIMARY KEY,
    source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    relationship TEXT NOT NULL,      -- e.g. "works_at", "mentions", "caused_by"
    weight FLOAT DEFAULT 1.0,
    properties JSON,
    source_document_id TEXT,
    source_chunk_id TEXT
);

-- User-uploaded structured data lands as native DuckDB tables
-- with user-chosen names. The schema is introspected at query time.
-- Example: user uploads sales.csv -> CREATE TABLE sales AS SELECT * FROM 'sales.csv';

-- Saved chart/dashboard definitions
CREATE TABLE saved_charts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sql_query TEXT NOT NULL,
    chart_spec TEXT NOT NULL,        -- ECharts option JSON
    created_by TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);
```

### 4.3 File Storage

Uploaded files are stored on the local filesystem at `<data_dir>/workspaces/{workspace_id}/files/`. DuckDB reads tabular files directly from this path. In a future cloud deployment, this becomes an object storage prefix, but for MVP, local filesystem only.

### 4.4 Internal Query Building (SeaQuery)

All SQL queries constructed by application code use SeaQuery as the query builder. No string concatenation or `format!()` for SQL anywhere in the codebase. This applies to control plane queries against SQLite, internal queries against workspace DuckDB databases (chunk retrieval, graph traversal, schema introspection), and schema migrations.

SeaQuery's SQLite backend produces the correct quoting and parameter binding for both SQLite and DuckDB (DuckDB accepts SQLite-compatible double-quoted identifiers). Use `SqliteQueryBuilder` as the backend for both engines.

**Where SeaQuery is used vs. where raw SQL is used:**

- **SeaQuery (all application-generated queries):** CRUD on control plane tables, vector search queries, graph traversal, document/chunk inserts, audit log writes, schema introspection helpers, migration DDL.
- **Raw SQL (LLM-generated):** Queries produced by the text-to-SQL agent loop. These are user-facing analytical queries that the LLM writes against user-uploaded tables. They execute on a read-only DuckDB connection with resource limits and are never constructed by application code.

This separation is important: SeaQuery prevents injection in the queries *we* write, while the read-only connection with `memory_limit` and `timeout` constrains the queries the *LLM* writes.

**Table identifiers:** Define all internal table and column names as SeaQuery `Iden` enums:

```rust
use sea_query::Iden;

#[derive(Iden)]
pub enum Workspace {
    Table,
    Id,
    Name,
    Classification,
    AllowedProviders,
    CreatedAt,
    UpdatedAt,
}

#[derive(Iden)]
pub enum Chunk {
    Table,
    Id,
    DocumentId,
    ChunkIndex,
    Content,
    Embedding,
    TokenCount,
}

#[derive(Iden)]
pub enum GraphNode {
    Table,
    Id,
    Label,
    Type,
    Properties,
    SourceDocumentId,
    SourceChunkId,
    Embedding,
}
```

**Example - vector search query built with SeaQuery:**

```rust
use sea_query::{Expr, Query, SqliteQueryBuilder, Func};

let query = Query::select()
    .columns([Chunk::Id, Chunk::Content, Chunk::DocumentId])
    .from(Chunk::Table)
    .order_by_expr(
        Expr::cust_with_values(
            "array_cosine_distance(embedding, ?::FLOAT[])",
            [query_embedding.into()],
        ),
        Order::Asc,
    )
    .limit(top_k)
    .to_string(SqliteQueryBuilder);
```

For DuckDB-specific syntax not covered by SeaQuery's SQLite backend (e.g. `array_cosine_distance`, `DESCRIBE`, `COPY ... TO`), use `Expr::cust()` or `Expr::cust_with_values()` for parameterized custom expressions. This keeps the query structure in SeaQuery while allowing DuckDB extensions where needed.

---

## 5. LLM Provider Abstraction

### 5.1 Provider Trait

```rust
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Identifier for logging and config
    fn name(&self) -> &str;

    /// Send a chat completion request and return the full response
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;

    /// Send a chat completion request and stream tokens back
    async fn stream(&self, request: CompletionRequest) -> Result<TokenStream>;

    /// Generate embeddings for a batch of texts
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Whether this provider supports tool/function calling
    fn supports_tools(&self) -> bool;
}

pub struct CompletionRequest {
    pub messages: Vec<Message>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub system: Option<String>,
}

pub struct CompletionResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: TokenUsage,
    pub finish_reason: FinishReason,
}
```

### 5.2 MVP Implementations

For MVP, implement two providers:

1. **OpenAI-compatible** - covers OpenAI, Ollama, vLLM, LiteLLM, and most gateways. Configuration is a base URL + optional API key + model name. This single implementation handles both the cloud and air-gapped cases.

2. **Anthropic** - native Messages API with tool use. Needed because Anthropic's API differs enough from OpenAI-compat that a shim is fragile.

### 5.3 Provider Authentication

Providers support three authentication modes, configured per provider:

1. **None** - no auth header sent. Used for local Ollama and other unauthenticated endpoints.

2. **Static API key** - read from an environment variable (never stored in the config file). Sent as `Authorization: Bearer {key}`. Used for OpenAI, Anthropic, and other key-based APIs.

3. **OAuth 2.0 with PKCE** - for enterprise identity providers (Entra ID, Okta, AWS SSO). The provider authenticates via Authorization Code + PKCE flow against a token endpoint and automatically manages the token lifecycle.

#### OAuth PKCE Flow

```rust
pub struct OAuthConfig {
    pub issuer_url: String,          // e.g. https://login.microsoftonline.com/{tenant_id}/v2.0
    pub client_id: String,
    pub scopes: Vec<String>,         // e.g. ["https://cognitiveservices.azure.com/.default"]
    pub redirect_uri: String,        // default: http://localhost:19876/callback
    pub token_cache_path: Option<PathBuf>,  // persist tokens across restarts
}

pub struct TokenManager {
    config: OAuthConfig,
    current_token: RwLock<Option<CachedToken>>,
}

pub struct CachedToken {
    pub access_token: String,
    pub expires_at: Instant,
    pub refresh_token: Option<String>,
}
```

Token lifecycle:

1. On first request (or expired token with no refresh token), start the PKCE flow:
   - Generate `code_verifier` (cryptographic random, 43-128 chars) and `code_challenge` (S256 hash)
   - Open the authorization URL in the user's browser (or print it in TUI/CLI mode for manual open)
   - Start a temporary local HTTP server on `redirect_uri` to capture the callback
   - Exchange the authorization code + code_verifier for an access token at the token endpoint
2. On subsequent requests, if the token is valid (with a 60-second buffer before expiry), reuse it.
3. If the token is expired and a refresh token is available, use the refresh_token grant to get a new access token silently (no browser interaction).
4. If the refresh token is also expired or absent, restart from step 1.
5. Optionally persist the token cache to disk (encrypted with a machine-local key) so restarts don't require re-auth. In air-gapped environments where the IdP is network-local, this avoids frequent re-prompts.

The `TokenManager` is shared across all requests to the provider and handles concurrency: if multiple requests race to refresh, only one performs the refresh and the others wait.

For TUI and CLI modes where opening a browser isn't possible (e.g. SSH into a remote box), fall back to device code flow: print a URL and code for the user to enter on another device, then poll the token endpoint.

#### Provider Configuration Examples

Provider configuration in `<config_dir>/config.toml` (default: `~/.config/quack/config.toml`; override with `QUACK_CONFIG_DIR`):

```toml
# No auth - local Ollama
[providers.ollama]
type = "openai-compat"
base_url = "http://localhost:11434/v1"
model = "llama3.1:8b"
embedding_model = "nomic-embed-text"

# Static API key from env var
[providers.anthropic]
type = "anthropic"
auth = "api-key"
api_key_env = "ANTHROPIC_API_KEY"   # read from env var, never stored in config
model = "claude-sonnet-4-20250514"

# Static API key from env var
[providers.openai]
type = "openai-compat"
auth = "api-key"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "gpt-4o"
embedding_model = "text-embedding-3-small"

# OAuth PKCE - Azure OpenAI via Entra ID
[providers.azure-openai]
type = "openai-compat"
base_url = "https://{resource}.openai.azure.com/openai/deployments/{deployment}"
model = "gpt-4o"
embedding_model = "text-embedding-3-small"
auth = "oauth-pkce"

[providers.azure-openai.oauth]
issuer_url = "https://login.microsoftonline.com/{tenant_id}/v2.0"
client_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
scopes = ["https://cognitiveservices.azure.com/.default"]
# redirect_uri = "http://localhost:19876/callback"   # default, override if needed
# token_cache_path = "<data_dir>/tokens/azure.json"    # optional, encrypted at rest
```

### 5.4 Provider Selection

The active provider is determined by, in priority order:

1. Workspace `allowed_providers` list (if set, filter to only these)
2. User's explicit selection in the current session
3. First available provider in config

In single-user/local mode, all configured providers are available.

---

## 6. Ingestion Pipeline

### 6.1 File Parsing

Accept these file types for MVP:

| Type | Parser | Output |
|------|--------|--------|
| PDF | `pdf-extract` crate or call `pdftotext` | Text chunks |
| DOCX | `docx-rs` crate | Text chunks |
| Plain text, Markdown | Direct read | Text chunks |
| CSV | DuckDB `read_csv_auto()` | Native DuckDB table |
| Parquet | DuckDB `read_parquet()` | Native DuckDB table |
| Excel (.xlsx) | DuckDB `st_read()` via spatial ext, or `calamine` crate | Native DuckDB table |
| JSON/JSONL | DuckDB `read_json_auto()` | Native DuckDB table |

### 6.2 Chunking Strategy

For unstructured text documents:

- Split on paragraph boundaries (double newline)
- Target chunk size: 512 tokens
- Overlap: 64 tokens between adjacent chunks
- Preserve document structure: if a heading is detected, prepend it to the chunk as context
- Each chunk records its `document_id` and `chunk_index` for ordering

### 6.3 Embedding

- Call the configured provider's `embed()` method in batches of 64 chunks
- Store the embedding vector in the `chunks.embedding` column
- Build the HNSW index after all chunks for a document are embedded
- For MVP, embedding dimension is fixed at provider config time (e.g. 768 for nomic-embed-text, 1536 for text-embedding-3-small)

### 6.4 Structured Data Ingestion

For tabular files:

- Create a DuckDB table named after the file (sanitized): `sales_q3_2024.csv` -> table `sales_q3_2024`
- If a table with that name exists, prompt the user: replace, rename, or cancel
- Run `DESCRIBE {table}` and store the schema summary for use in text-to-SQL prompts

### 6.5 Async Processing

Ingestion runs on a background tokio task. The API/TUI returns immediately with a document ID and `status: processing`. The caller can poll status. For MVP, a simple in-memory task queue (tokio mpsc channel) is sufficient - no external queue needed.

---

## 7. Analysis Engine

### 7.1 Agent Loop

The analysis engine is a tool-calling agent. When a user asks a question, the loop is:

```
1. Build system prompt with:
   - Available tables and their schemas (from DESCRIBE/SHOW TABLES)
   - Available documents (from documents table)
   - Instructions for tool use

2. Send user question + system prompt to LLM

3. If LLM returns a tool call:
   a. Execute the tool (see tool definitions below)
   b. Append the tool result to the message history
   c. Go to step 2

4. If LLM returns a text response:
   a. Return to user
   b. If response includes a chart spec, validate and return it alongside
```

### 7.2 Tool Definitions

The agent has these tools available:

```
run_sql(query: string) -> table
  Execute a read-only SQL query against the workspace DuckDB.
  Returns up to 100 rows as a formatted table.
  DuckDB enforces: SET memory_limit = '256MB'; SET timeout = 30;

search_documents(query: string, top_k: int = 5) -> chunks
  Vector similarity search over document chunks.
  Returns the top_k most relevant chunks with their source document.

describe_table(table_name: string) -> schema
  Returns column names, types, and sample values (3 rows) for a table.

list_tables() -> names
  Returns all user tables in the workspace.

list_documents() -> documents
  Returns all ingested documents with their status.

create_chart(sql: string, chart_type: string, x: string, y: string, title: string) -> spec
  Runs the SQL query, then generates an ECharts option spec for the results.
  chart_type: bar | line | scatter | area | pie | treemap | graph
  The agent emits minimal parameters; the rendering layer wraps them
  in the app's Tailwind-derived ECharts theme (colors, fonts, spacing).
```

### 7.3 Text-to-SQL Prompt

The system prompt for SQL generation includes:

- All table schemas with column names and types
- Up to 3 sample rows per table (fetched at session start, cached)
- DuckDB-specific SQL dialect notes (e.g. `LIMIT`, `EXCLUDE`, `STRUCT`, list comprehensions)
- Instruction to always use table and column names exactly as they appear
- Instruction to explain the query in natural language before presenting results

### 7.4 RAG Retrieval

When the user's question is about document content rather than structured data:

1. Embed the user's question using the same embedding model used for ingestion
2. Query DuckDB: `SELECT id, content, document_id FROM chunks ORDER BY array_cosine_distance(embedding, ?::FLOAT[]) LIMIT ?`
3. Return chunks as context to the LLM along with the original question
4. LLM synthesizes an answer citing the source documents

The agent decides whether to use SQL tools or RAG based on the question. If tables exist and the question sounds analytical, prefer SQL. If documents exist and the question sounds like "what does the document say about...", prefer RAG. If both exist, the agent may use both in one turn.

---

## 8. Graph Layer

### 8.1 Entity Extraction

When graph features are enabled for a workspace (opt-in), entity extraction runs as a post-processing step after document ingestion:

1. For each chunk, send to LLM with a structured extraction prompt:
   ```
   Extract entities and relationships from this text.
   Return JSON: { "nodes": [{"label": "...", "type": "..."}],
                   "edges": [{"source": "...", "target": "...",
                              "relationship": "..."}] }
   ```
2. Deduplicate nodes by (label, type) - merge nodes with same normalized label
3. Insert into `graph_nodes` and `graph_edges` tables
4. Link back to source document and chunk via foreign keys

### 8.2 Graph Queries

For MVP, implement graph traversal as SQL queries rather than using DuckPGQ. Build these with SeaQuery where possible; the recursive CTE uses `Expr::cust_with_values()` since SeaQuery's CTE support is limited:

```sql
-- 1-hop neighbors (built via SeaQuery joins)
SELECT DISTINCT n2.label, n2.type, e.relationship
FROM graph_nodes n1
JOIN graph_edges e ON n1.id = e.source_node_id OR n1.id = e.target_node_id
JOIN graph_nodes n2 ON (n2.id = e.target_node_id OR n2.id = e.source_node_id)
  AND n2.id != n1.id
WHERE n1.label ILIKE ?;

-- N-hop traversal via recursive CTE (built via Expr::cust_with_values)
WITH RECURSIVE hops AS (
    SELECT id, label, type, 0 AS depth
    FROM graph_nodes WHERE label ILIKE ?
    UNION ALL
    SELECT n.id, n.label, n.type, h.depth + 1
    FROM hops h
    JOIN graph_edges e ON h.id = e.source_node_id OR h.id = e.target_node_id
    JOIN graph_nodes n ON (n.id = e.target_node_id OR n.id = e.source_node_id)
      AND n.id != h.id
    WHERE h.depth < ?  -- max depth parameter
)
SELECT DISTINCT label, type, depth FROM hops ORDER BY depth, label;
```

Add a `search_graph(entity: string, max_hops: int)` tool to the agent so it can traverse the graph when answering questions about entity relationships.

### 8.3 TUI Graph Rendering

In the TUI, render graph results as an indented tree:

```
[Person] Alice Chen
  --works_at--> [Org] Acme Corp
    --located_in--> [Place] New York
  --authored--> [Document] Q3 Report
    --mentions--> [Person] Bob Park
  --collaborates_with--> [Person] Carol Wu
```

This is a depth-first traversal of the adjacency results, rendered with Unicode box-drawing characters. No force-directed layout needed.

---

## 9. TUI Design (ratatui)

### 9.1 Layout

```
+---------------------------------------------------------------+
| quack v0.1.0 | workspace: default | provider: ollama/llama3.1 |
+---------------------------------------------------------------+
|                              |                                 |
|   Chat / Results Panel       |   Detail Panel                  |
|   (60% width)                |   (40% width)                   |
|                              |                                 |
|   > What were total sales    |   SQL:                          |
|     by region in Q3?         |   SELECT region,                |
|                              |     SUM(amount) as total        |
|   Running query...           |   FROM sales                    |
|                              |   WHERE quarter = 'Q3'          |
|   region    | total          |   GROUP BY region               |
|   ----------|--------        |   ORDER BY total DESC;          |
|   Northeast | 1,234,567      |                                 |
|   West      |   987,654      |   Chart: (would display in web) |
|   South     |   876,543      |                                 |
|   Midwest   |   654,321      |   Sources:                      |
|                              |   - sales.csv (4 cols, 12k rows)|
|                              |                                 |
+---------------------------------------------------------------+
| [Tab: Chat] [Tab: SQL] [Tab: Graph] [Tab: Tables]    F1: Help |
+---------------------------------------------------------------+
| > _                                                            |
+---------------------------------------------------------------+
```

### 9.2 Tabs

- **Chat** - conversational interface. User types a question, sees streamed response with results inline.
- **SQL** - direct SQL editor. Execute queries, see results as a table. History with up/down arrows.
- **Graph** - entity search. Type an entity name, see its neighborhood as an ASCII tree.
- **Tables** - browse workspace tables. Select a table to see schema, row count, sample rows.

### 9.3 Key Bindings

| Key | Action |
|-----|--------|
| `Tab` | Cycle between tabs |
| `Ctrl+J` / `Enter` | Send message / execute query |
| `Ctrl+C` | Cancel running operation |
| `Ctrl+Q` | Quit |
| `Ctrl+P` | Switch provider |
| `Ctrl+W` | Switch workspace |
| `Ctrl+O` | Open/import file |
| `Up/Down` | Scroll results / history |
| `F1` | Help overlay |

### 9.4 Streaming

LLM responses stream into the chat panel token by token. Implementation:

1. LLM streaming runs on a tokio task, sends tokens via an `mpsc::channel`
2. The ratatui event loop polls the channel on each tick (50ms)
3. New tokens are appended to the current message buffer
4. The terminal redraws the chat panel

DuckDB queries run on a `tokio::task::spawn_blocking` since `duckdb-rs` is synchronous. Results are sent back via another channel.

A status bar at the bottom shows the current phase: `Thinking...`, `Generating SQL...`, `Running query...`, `Streaming response...`

---

## 10. Web Server (quack serve)

### 10.1 Routes

```
GET  /                          # landing / workspace list
GET  /w/{workspace_id}          # workspace chat UI (askama + htmx)
POST /w/{workspace_id}/chat     # send message, returns SSE stream
POST /w/{workspace_id}/upload   # file upload
GET  /w/{workspace_id}/tables   # list tables
GET  /w/{workspace_id}/tables/{name}  # table detail
GET  /w/{workspace_id}/charts   # saved charts
GET  /w/{workspace_id}/charts/{id}    # render a saved chart

# REST API (JSON, token-authenticated)
POST   /api/v1/workspaces                    # create workspace
GET    /api/v1/workspaces                    # list workspaces
GET    /api/v1/workspaces/{id}               # get workspace detail
POST   /api/v1/workspaces/{id}/query         # run a question or SQL
POST   /api/v1/workspaces/{id}/upload        # ingest a file
GET    /api/v1/workspaces/{id}/documents     # list documents
GET    /api/v1/workspaces/{id}/tables        # list tables
GET    /api/v1/workspaces/{id}/graph/search  # graph entity search

# MCP server
GET  /mcp/v1/{workspace_id}     # MCP SSE endpoint (workspace-scoped token)

# OpenAI-compatible
POST /openai/v1/chat/completions  # workspace resolved from bearer token
```

### 10.2 Authentication

In local/single-user mode (`quack serve --local`), no authentication. All requests go to the default workspace.

In multi-user mode, authentication is via bearer tokens in the `Authorization` header. Tokens are workspace-scoped and created via the CLI:

```bash
quack token create --workspace my-workspace --name "my-api-key"
```

OIDC integration is deferred to post-MVP.

### 10.3 Web UI

The web UI is server-rendered HTML using askama templates, Tailwind CSS for styling, and htmx for interactivity. No JavaScript framework. Vendor these assets into the binary (embed via `include_str!` or rust-embed):

- htmx.js (~14KB gzipped) - AJAX, SSE, DOM updates
- echarts.min.js (~200KB gzipped, custom build) - chart rendering. Use the ECharts online builder to include only: bar, line, scatter, pie, treemap, graph series + grid, tooltip, legend, title, dataZoom components + canvas renderer
- tailwind.css (pre-compiled output) - utility CSS

Tailwind CSS is compiled at build time, not at runtime. The build uses the standalone Tailwind CLI binary (no Node.js or npm). Download it from GitHub releases for your platform, or let the Dockerfile handle it for CI.

```bash
# Download standalone tailwindcss binary (one-time dev setup)
curl -sLO https://github.com/tailwindlabs/tailwindcss/releases/download/v4.3.3/tailwindcss-linux-x64
chmod +x tailwindcss-linux-x64
mv tailwindcss-linux-x64 ~/.local/bin/tailwindcss

# During development (watches for changes)
tailwindcss -i styles/input.css -o static/css/output.css --watch

# For release build (via build.rs or justfile)
tailwindcss -i styles/input.css -o static/css/output.css --minify
```

The ECharts theme is derived from Tailwind's CSS custom properties so charts match the UI automatically:

```javascript
// Registered once on page load
echarts.registerTheme('quack', {
  color: [
    'var(--color-primary)',
    'var(--color-secondary)',
    // ... palette from tailwind config
  ],
  backgroundColor: 'transparent',
  textStyle: { fontFamily: 'var(--font-sans)' },
  // ... axis, tooltip, legend styles from CSS vars
});
```

Chat responses stream via Server-Sent Events. htmx's `hx-ext="sse"` handles the connection. Chart specs returned by the agent are rendered client-side by ECharts using the registered theme. The agent emits a minimal spec (series type, data, axis labels, title); the client-side JS wraps it into a full ECharts option with the theme applied, keeping the LLM's job simple and chart appearance consistent.

---

## 11. MCP Server

The MCP endpoint exposes workspace data as tools and resources that external agents (Claude Desktop, Cursor, custom agents) can use.

### 11.1 Tools Exposed

| Tool | Description |
|------|-------------|
| `query` | Run a natural language question against the workspace |
| `sql` | Execute a SQL query directly |
| `search` | Vector search over documents |
| `list_tables` | List available tables |
| `describe_table` | Get schema for a table |
| `list_documents` | List ingested documents |
| `search_graph` | Search entity graph |

### 11.2 Resources Exposed

| Resource | Description |
|----------|-------------|
| `quack://workspace/tables` | Dynamic list of tables |
| `quack://workspace/tables/{name}/schema` | Table schema |
| `quack://workspace/documents` | Dynamic list of documents |

### 11.3 Transport

MCP uses SSE (Server-Sent Events) transport since the server is already an HTTP server. The MCP endpoint is workspace-scoped: the bearer token determines which workspace the MCP client sees.

---

## 12. Configuration

All configuration lives in `<config_dir>/config.toml` with environment variable overrides. The config directory defaults to `~/.config/quack/`; override with `QUACK_CONFIG_DIR`.

```toml
[general]
# data_dir defaults to XDG data dir; override: QUACK_DATA_DIR
default_workspace = "default"

[server]
bind = "127.0.0.1:8080"           # override: QUACK_BIND
# In local mode, binds to localhost only. Set to 0.0.0.0 for network access.

[tui]
tick_rate_ms = 50
theme = "dark"                    # dark | light

[providers.ollama]
type = "openai-compat"
base_url = "http://localhost:11434/v1"
model = "llama3.1:8b"
embedding_model = "nomic-embed-text"

[ingestion]
chunk_size_tokens = 512
chunk_overlap_tokens = 64
embedding_batch_size = 64

[analysis]
max_query_rows = 100              # max rows returned from SQL tool
query_timeout_seconds = 30
memory_limit_mb = 256             # DuckDB per-query memory limit

[graph]
enabled = false                   # opt-in per workspace
max_traversal_depth = 3
```

---

## 13. Build and Distribution

### 13.1 Cargo Features

```toml
[features]
default = ["tui", "serve", "cli"]
tui = ["dep:ratatui", "dep:crossterm", "dep:tui-textarea"]
serve = ["dep:axum", "dep:askama", "dep:tower-http"]
cli = []  # no extra deps, uses core only
graph = ["dep:duckdb/duckpgq"]    # optional, adds DuckPGQ extension
```

This allows building a TUI-only binary for minimal air-gapped installs:

```bash
cargo build --release --no-default-features --features tui
```

### 13.2 Static Linking

Every release artifact is a fully static musl binary. No glibc, no dynamic linking, no shared libraries. This is non-negotiable for the air-gapped story - the binary runs on any Linux without dependency resolution.

**Global allocator:** musl's default allocator is slow under contention. Every binary crate sets mimalloc as the global allocator:

```rust
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

**Crypto backend:** All TLS and cryptographic operations go through `rustls` with `aws-lc-rs` as the crypto provider. Never pull in `ring` or `openssl`. The workspace root pins versions with `default-features = false`; each sub-crate enables only the features it needs (see section 15).

```toml
# Audit for ring leaking in transitively:
# cargo tree -i ring
# If anything appears, add a patch or swap the dep.
```

**aws-lc-rs with musl:** aws-lc-rs requires a C compiler and cmake to build its bundled AWS-LC. For musl cross-compilation, use the `musl-cross-make` toolchain or build inside an Alpine container where these are native. The `fips` feature enables the FIPS 140-3 validated module - this is a checkbox item for defense and financial buyers.

**Build command:**

```bash
# Install musl target
rustup target add x86_64-unknown-linux-musl

# Build fully static binary
CC_x86_64_unknown_linux_musl=musl-gcc \
RUSTFLAGS="-C target-feature=+crt-static" \
cargo build --release --target x86_64-unknown-linux-musl

# Verify it's static
file target/x86_64-unknown-linux-musl/release/quack
# quack: ELF 64-bit LSB executable, x86-64, statically linked, ...

ldd target/x86_64-unknown-linux-musl/release/quack
# not a dynamic executable
```

DuckDB is statically linked via the `duckdb-rs` `bundled` feature. SQLite is statically linked via the `rusqlite` `bundled` feature. The vss extension for vector search must be compiled in. All three (DuckDB, SQLite, AWS-LC) are C/C++ compiled and linked into the final musl binary.

### 13.3 Container Image

No Node.js, no npm. Tailwind CSS is downloaded as a standalone binary from GitHub releases with checksum verification. The build uses cargo-chef for dependency layer caching and produces a distroless runtime image.

```dockerfile
# syntax=docker/dockerfile:1

# CSS build stage - download and run standalone tailwindcss
FROM debian:trixie-slim AS css-builder
ARG TARGETARCH
WORKDIR /app

# Download standalone tailwindcss CLI with checksum verification
# Update checksums when bumping tailwindcss version
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && case "$TARGETARCH" in \
        amd64) \
            BINARY="tailwindcss-linux-x64" \
            CHECKSUM="<sha256-for-x64>" \
            ;; \
        arm64) \
            BINARY="tailwindcss-linux-arm64" \
            CHECKSUM="<sha256-for-arm64>" \
            ;; \
        *) \
            echo "Unsupported architecture: $TARGETARCH" && exit 1 \
            ;; \
    esac \
    && curl -sLO "https://github.com/tailwindlabs/tailwindcss/releases/download/v4.3.3/${BINARY}" \
    && echo "${CHECKSUM}  ${BINARY}" | sha256sum -c - \
    && chmod +x "${BINARY}" \
    && mv "${BINARY}" tailwindcss

COPY crates/quack-serve/static crates/quack-serve/static
COPY crates/quack-serve/tailwind.config.js crates/quack-serve/
COPY crates/quack-serve/styles crates/quack-serve/styles
COPY crates/quack-serve/templates crates/quack-serve/templates
COPY crates/quack-serve/src crates/quack-serve/src

RUN cd crates/quack-serve \
    && /app/tailwindcss -i styles/input.css -o static/css/output.css --minify

# cargo-chef base stage
FROM rust:1.80-alpine AS chef
RUN cargo install cargo-chef --locked
WORKDIR /app

# Planner stage - generate dependency recipe
FROM chef AS planner
COPY Cargo.toml Cargo.lock ./
COPY crates/quack-core/Cargo.toml crates/quack-core/
COPY crates/quack-tui/Cargo.toml crates/quack-tui/
COPY crates/quack-serve/Cargo.toml crates/quack-serve/
COPY crates/quack-cli/Cargo.toml crates/quack-cli/

# Create dummy source files so cargo metadata can resolve the workspace
RUN mkdir -p crates/quack-core/src && touch crates/quack-core/src/lib.rs \
    && mkdir -p crates/quack-tui/src && touch crates/quack-tui/src/main.rs \
    && mkdir -p crates/quack-serve/src && touch crates/quack-serve/src/main.rs \
    && mkdir -p crates/quack-cli/src && touch crates/quack-cli/src/main.rs

RUN cargo chef prepare --recipe-path recipe.json

# Rust build stage - musl for static binary
FROM chef AS builder
ARG SOURCE_DATE_EPOCH=0

# clang is required for FIPS delocator on aarch64
RUN apk add --no-cache musl-dev cmake make go perl clang linux-headers

ENV AWS_LC_FIPS_SYS_CC=clang
ENV AWS_LC_FIPS_SYS_CXX=clang++

# Cook dependencies (cached until Cargo.toml/Cargo.lock change)
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

# Restore real manifests
COPY Cargo.toml Cargo.lock ./
COPY crates/quack-core/Cargo.toml crates/quack-core/
COPY crates/quack-tui/Cargo.toml crates/quack-tui/
COPY crates/quack-serve/Cargo.toml crates/quack-serve/
COPY crates/quack-cli/Cargo.toml crates/quack-cli/

# Copy actual source code
COPY crates/quack-core/src crates/quack-core/src
COPY crates/quack-serve/src crates/quack-serve/src
COPY crates/quack-serve/templates crates/quack-serve/templates
COPY crates/quack-cli/src crates/quack-cli/src

# Copy built static assets (needed at compile time for rust-embed)
COPY --from=css-builder /app/crates/quack-serve/static crates/quack-serve/static

# Touch files with deterministic timestamp for reproducible builds
RUN touch -d "@${SOURCE_DATE_EPOCH}" crates/quack-core/src/lib.rs crates/quack-serve/src/main.rs

# Build the release binary
RUN cargo build --release --package quack-serve

# Create data directory marker
RUN mkdir -p /data && touch /data/.keep

# Runtime stage - distroless, no OS layer
FROM gcr.io/distroless/static-debian13:nonroot
WORKDIR /
LABEL org.opencontainers.image.source=https://github.com/smoketurner/quack

# Static assets are embedded via rust-embed - nothing else to copy
COPY --from=builder /app/target/release/quack-serve /quack
COPY --from=builder --chown=nonroot:nonroot /data /data

ENV QUACK_BIND=0.0.0.0:8080
ENV QUACK_DATA_DIR=/data
EXPOSE 8080
ENTRYPOINT ["/quack"]
```

Key properties of this build:
- **No Node.js/npm anywhere.** Tailwind is a standalone binary downloaded with checksum verification.
- **Multi-arch.** `TARGETARCH` is set automatically by BuildKit for amd64/arm64 builds.
- **cargo-chef** caches compiled dependencies in a separate Docker layer. Only source code changes trigger a recompile - dependency resolution is cached until `Cargo.toml` or `Cargo.lock` change.
- **SOURCE_DATE_EPOCH** ensures deterministic timestamps for reproducible builds.
- **distroless runtime.** The final image contains only the static binary and a data directory. No shell, no package manager, no glibc. If PDF parsing via `pdftotext` is needed, swap to `alpine:3.20` with `apk add poppler-utils` - but prefer a pure-Rust PDF parser (see section 6.1) to stay distroless.
- **Runs as nonroot.** The distroless `nonroot` tag runs as UID 65534.

For air-gapped: export the image as a tarball, load with `docker load`.

### 13.4 Docker Compose (air-gapped deployment)

```yaml
version: "3.8"
services:
  quack:
    image: quack:latest
    ports:
      - "8080:8080"
    volumes:
      - quack-data:/data
    environment:
      - QUACK_DATA_DIR=/data
      - QUACK_BIND=0.0.0.0:8080
    restart: unless-stopped

  # Optional: local LLM
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama-models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

volumes:
  quack-data:
  ollama-models:
```

---

## 14. MVP Scope and Boundaries

### In Scope (build this)

- [x] Single Rust binary with `tui`, `serve`, and `query` subcommands
- [x] DuckDB workspace isolation (one .duckdb file per workspace)
- [x] SQLite control plane (workspaces, threads, messages, audit log)
- [x] File ingestion: CSV, Parquet, JSON/JSONL -> DuckDB tables
- [x] File ingestion: PDF, DOCX, TXT, MD -> chunked + embedded for RAG
- [x] OpenAI-compatible LLM provider (covers Ollama + OpenAI + most gateways)
- [x] Anthropic LLM provider
- [x] Text-to-SQL agent loop with tool calling
- [x] RAG retrieval over document chunks
- [x] TUI with chat input and agent loop (quack-tui crate)
- [ ] TUI SQL editor and table browser tabs
- [ ] Web UI with chat (askama + htmx + SSE streaming)
- [ ] REST API for all operations
- [ ] MCP server (workspace-scoped)
- [x] ECharts chart generation (chart spec in analysis engine)
- [ ] Graph tables (nodes/edges) with entity extraction
- [ ] Graph traversal via recursive CTE
- [ ] ASCII graph rendering in TUI
- [x] Configuration via TOML + env vars
- [ ] Basic audit logging
- [ ] Docker image and docker-compose for deployment

### Out of Scope (defer)

- OIDC/SAML authentication (use token auth for MVP)
- DuckLake cloud storage backend (local DuckDB files only for MVP)
- Superset or external BI integration
- OpenAI-compatible chat endpoint (/v1/chat/completions)
- DuckPGQ extension integration (use plain SQL for graph)
- Dashboard builder (saved charts are v1; layout/grid is later)
- Multi-model agent routing (one provider active at a time)
- File versioning or change tracking on uploads
- WebSocket transport for MCP (SSE only)
- Kubernetes deployment manifests (docker-compose for now)
- Role-based access control beyond owner/member/viewer
- Workspace invitation flow in web UI (CLI-only for MVP)

---

## 15. Dependencies

### 15.1 Dependency Management Rules

1. **All Rust dependencies are declared in the workspace root `Cargo.toml`** under `[workspace.dependencies]` with an exact version and `default-features = false`.
2. **Sub-crates reference workspace deps** via `dep.workspace = true` and enable only the features they need.
3. **No sub-crate may introduce a dependency not declared in the workspace root.**
4. **Use `Cargo.lock` in version control** (not gitignored) for reproducible builds.
5. **`ring` and `openssl` must never appear in the dependency tree.** Audit with `cargo tree -i ring` and `cargo tree -i openssl-sys` before every release.

### 15.2 Workspace Root Cargo.toml

All versions pinned, all default features disabled. This is the single source of truth for dependency versions.

```toml
[workspace]
members = ["crates/*"]
resolver = "2"

[workspace.dependencies]
# Storage
duckdb        = { version = "=1.2.1",  default-features = false }
rusqlite      = { version = "=0.32.1", default-features = false }
sea-query     = { version = "=1.0.2",  default-features = false }

# Web server
axum          = { version = "=0.8.3",  default-features = false }
askama        = { version = "=0.13.0", default-features = false }
tower-http    = { version = "=0.6.2",  default-features = false }
tokio         = { version = "=1.44.2", default-features = false }

# TUI
ratatui       = { version = "=0.29.0", default-features = false }
crossterm     = { version = "=0.28.1", default-features = false }
tui-textarea  = { version = "=0.7.0",  default-features = false }

# HTTP / TLS / Crypto
reqwest       = { version = "=0.12.12", default-features = false }
rustls        = { version = "=0.23.23", default-features = false }
aws-lc-rs     = { version = "=1.12.0",  default-features = false }

# Serialization / Config
serde         = { version = "=1.0.217", default-features = false }
serde_json    = { version = "=1.0.138", default-features = false }
toml          = { version = "=0.8.20",  default-features = false }

# CLI
clap          = { version = "=4.5.27",  default-features = false }

# Auth
oauth2        = { version = "=5.0.0",   default-features = false }
sha2          = { version = "=0.10.8",  default-features = false }
open          = { version = "=5.3.2",   default-features = false }

# Observability / Utilities
tracing           = { version = "=0.1.41",  default-features = false }
tracing-subscriber = { version = "=0.3.19", default-features = false }
ulid              = { version = "=1.1.4",   default-features = false }
rust-embed        = { version = "=8.5.0",   default-features = false }
mimalloc          = { version = "=0.1.43",  default-features = false }
async-trait       = { version = "=0.1.83",  default-features = false }
thiserror         = { version = "=2.0.11",  default-features = false }
```

> **Note:** Version numbers above are illustrative. Pin to the latest stable version at the time of first build. The `=` prefix enforces exact match. Update versions deliberately via PR, never implicitly.

### 15.3 Sub-Crate Dependencies

Each sub-crate pulls from the workspace and enables only the features it requires.

**quack-core/Cargo.toml:**

```toml
[dependencies]
duckdb        = { workspace = true, features = ["bundled"] }
rusqlite      = { workspace = true, features = ["bundled"] }
sea-query     = { workspace = true, features = ["backend-sqlite", "derive"] }
tokio         = { workspace = true, features = ["rt-multi-thread", "macros", "sync", "time"] }
reqwest       = { workspace = true, features = ["rustls-tls", "json", "stream"] }
rustls        = { workspace = true, features = ["aws_lc_rs", "std"] }
aws-lc-rs     = { workspace = true, features = ["fips"] }
serde         = { workspace = true, features = ["derive"] }
serde_json    = { workspace = true }
toml          = { workspace = true, features = ["parse"] }
oauth2        = { workspace = true, features = ["reqwest"] }
sha2          = { workspace = true }
open          = { workspace = true }
tracing       = { workspace = true }
ulid          = { workspace = true, features = ["serde"] }
async-trait   = { workspace = true }
thiserror     = { workspace = true }
```

**quack-tui/Cargo.toml:**

```toml
[dependencies]
quack-core    = { path = "../quack-core" }
ratatui       = { workspace = true, features = ["crossterm"] }
crossterm     = { workspace = true, features = ["event-stream"] }
tui-textarea  = { workspace = true, features = ["crossterm"] }
tokio         = { workspace = true, features = ["rt-multi-thread", "macros", "sync"] }
clap          = { workspace = true, features = ["derive"] }
tracing       = { workspace = true }
tracing-subscriber = { workspace = true, features = ["fmt", "env-filter"] }
mimalloc      = { workspace = true }
```

**quack-serve/Cargo.toml:**

```toml
[dependencies]
quack-core    = { path = "../quack-core" }
axum          = { workspace = true, features = ["json", "tokio"] }
askama        = { workspace = true }
tower-http    = { workspace = true, features = ["cors", "compression-gzip", "fs"] }
tokio         = { workspace = true, features = ["rt-multi-thread", "macros", "signal"] }
rust-embed    = { workspace = true, features = ["compression"] }
clap          = { workspace = true, features = ["derive"] }
serde         = { workspace = true, features = ["derive"] }
serde_json    = { workspace = true }
tracing       = { workspace = true }
tracing-subscriber = { workspace = true, features = ["fmt", "env-filter"] }
mimalloc      = { workspace = true }
```

**quack-cli/Cargo.toml:**

```toml
[dependencies]
quack-core    = { path = "../quack-core" }
tokio         = { workspace = true, features = ["rt-multi-thread", "macros"] }
clap          = { workspace = true, features = ["derive"] }
serde_json    = { workspace = true }
tracing       = { workspace = true }
tracing-subscriber = { workspace = true, features = ["fmt", "env-filter"] }
mimalloc      = { workspace = true }
```

### 15.4 External Tools

| Tool | Purpose | When Needed | Notes |
|------|---------|-------------|-------|
| `pdftotext` (poppler-utils) | PDF text extraction | Runtime (if not using pure-Rust parser) | Prefer a pure-Rust PDF parser to keep distroless runtime |
| `tailwindcss` standalone binary | Compile utility CSS from templates | Build time only | Downloaded from GitHub releases with checksum verification; no Node.js/npm |

---

## 16. Testing Strategy

### Unit Tests

- Storage: workspace create/open/attach, schema migrations, CRUD on all control plane tables
- Ingestion: chunking logic (boundary detection, overlap, token counting)
- Analysis: SQL sanitization (verify read-only), tool dispatch, prompt construction
- LLM: provider trait mock with canned responses for agent loop testing
- Graph: entity dedup, traversal query correctness

### Integration Tests

- End-to-end: upload a CSV, ask a question, verify correct SQL is generated and results returned
- End-to-end: upload a PDF, ask about its content, verify RAG retrieval returns relevant chunks
- MCP: connect with an MCP test client, invoke tools, verify responses
- REST API: full lifecycle (create workspace, upload, query, delete)

### Manual Testing

- TUI: interactive testing for layout, key bindings, streaming behavior
- Air-gapped: build static binary, copy to a machine with no internet, run with Ollama

---

## 17. Implementation Order

Build in this order. Each phase produces a working, testable artifact.

**Phase 1: Core foundation (week 1-2)**
1. Cargo workspace structure with all crates
2. Config parsing (TOML + env vars)
3. SQLite control plane with migrations
4. DuckDB workspace creation and isolation
5. `quack query` CLI - execute raw SQL against a workspace

**Phase 2: Ingestion (week 2-3)**
6. Tabular file ingestion (CSV, Parquet, JSON -> DuckDB tables)
7. Text file parsing (PDF, DOCX, TXT -> chunks)
8. Embedding via OpenAI-compat provider
9. Vector storage and HNSW index in DuckDB vss

**Phase 3: Analysis engine (week 3-4)**
10. LLM provider trait + OpenAI-compat implementation
11. Tool-calling agent loop
12. Text-to-SQL with schema injection
13. RAG retrieval pipeline
14. Chart spec generation (ECharts option format)

**Phase 4: TUI (week 4-5)**
15. Basic ratatui app shell with tabs and input
16. Chat tab with streaming LLM responses
17. SQL editor tab with query execution and results table
18. Tables browser tab
19. File import (Ctrl+O -> file picker)

**Phase 5: Web server (week 5-6)**
20. Tailwind CSS setup (config, input.css, build script) + askama templates
21. axum server with Tailwind-styled pages
22. Chat page with htmx SSE streaming
23. File upload endpoint
24. REST API endpoints
25. ECharts rendering with Tailwind-derived theme in browser

**Phase 6: MCP + Graph (week 6-7)**
26. MCP SSE server with tool definitions
27. Graph node/edge tables
28. Entity extraction from document chunks
29. Graph traversal queries
30. ASCII graph view in TUI
31. `search_graph` tool added to agent

**Phase 7: Polish (week 7-8)**
32. Audit logging across all operations
33. Workspace membership (invite via CLI)
34. Token authentication for API/MCP
35. Docker image and docker-compose
36. Error handling, edge cases, documentation
