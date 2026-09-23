# Quack: Knowledge Engine - Design Document

## 1. What This Is

`quack` is a knowledge engine with many interfaces. The core is a single Rust library that
holds a workspace's knowledge in three forms and answers questions across all three:

| Substrate | What goes in | How it is queried |
|-----------|--------------|-------------------|
| **Documents** | PDFs, Office files, text, Markdown, uploads, pasted text | Vectorized and chunked; hybrid semantic + keyword retrieval with citations |
| **Tables** | CSV, Parquet, JSON, Excel, attached databases | DuckDB SQL written by the agent or the user, shown before it runs |
| **Knowledge graph** | Entities and relationships extracted from documents and tables, typed by an ontology the tool can propose | Traversal, neighborhood, and path questions with provenance back to source text |

Around that core sit thin interfaces that all call the same functions:

- **Web UI** - chat with a workspace in a browser; the day-one replacement for the current
  AnythingLLM deployment.
- **REST API** - the same operations over HTTP for scripts and other systems.
- **MCP server** - the workspace as tools for Claude Code, Claude Desktop, Cursor, and other
  agents, over stdio or streamable HTTP.
- **TUI** - a Claude Code-style terminal session, plus a non-interactive print mode for
  pipelines.
- **Desktop window** (`quack desktop`) - the web UI in a native Tauri window with the
  server running in-process, for laptops and offline use. Last on the roadmap, if built.

One binary, `quack`, provides every interface. There are no Cargo features to enable or
disable surfaces; every build of `quack` on every platform contains the server, the
terminal, the MCP transport, and the admin commands.

The tool works offline with local models via Ollama and with hosted providers (OpenAI,
Anthropic, Azure OpenAI via OAuth, and anything OpenAI-compatible).

This document is the contract for the engineering agent implementing it. Section 17 lists
where the code diverges today.

---

## 2. What It Replaces

The current system is an AnythingLLM deployment: a chat interface over a pgvector RAG store
where users upload documents into workspaces and ask questions about them. `quack` must
cover what people use today before adding what they cannot do today.

| AnythingLLM concept | quack equivalent | Notes |
|---------------------|------------------|-------|
| Workspace | Workspace (section 5) | Same idea: isolated documents, settings, and chat history |
| Document upload and embed | Ingestion (section 6.1) | Hybrid search adds keyword matching to vectors |
| Workspace system prompt | Workspace context (section 5.3) | Editable in the web UI; also carries data definitions |
| Chat vs Query mode | Chat modes (section 7.5) | Query mode answers only from retrieved sources |
| Citations | Citations (section 6.1) | Every answer carries chunk-level sources with page/heading |
| Threads | Sessions (section 8) | Persistent, resumable, exportable |
| Pinned documents | Pinned documents (section 6.1) | Whole document injected into context |
| Users, roles, multi-user mode | Server auth (section 12) | Admin / manager / default map to owner / member / viewer |
| API keys | API tokens (section 12) | Workspace-scoped bearer tokens |
| LLM and embedding provider settings | Providers (section 10) | Per-instance config; per-workspace `allowed_providers` |
| Agent skills (SQL connector, web search, charts) | Agent tools (section 7.3) | SQL and charts are native; web search is deferred |
| Data connectors (web, GitHub, Confluence) | Deferred (section 18) | File upload and paste on day one |

Assumptions about the current deployment that shaped this document and should be confirmed:
users log in with local accounts rather than SSO; documents are mostly PDF and Office
exports; the pgvector store does not need to be migrated in place (re-embedding into
`quack` is acceptable); a single server instance serves all users.

---

## 3. Principles

1. **The core is the product; interfaces are adapters.** `quack-core` owns every behavior.
   A feature that exists in one interface and not the others is a bug, not a roadmap item.
2. **Three substrates, one agent.** Documents, tables, and the graph are queried by one
   agent with one set of tools. The user asks a question; the agent decides whether to
   retrieve, query, traverse, or combine.
3. **The workspace is the classification boundary.** Everything classified about a
   workspace, including its documents, tables, chunks, graph, ontology, context,
   sessions, and the detail of what was done to it, lives inside the workspace's own
   storage and nowhere else. Access control is granted per workspace. Nothing outside the
   boundary holds anything that would reveal the workspace's contents.
4. **Show every action.** Every retrieval, SQL statement, and traversal is surfaced with its
   inputs, result size, and timing, in every interface, and every answer cites its sources.
5. **Ask before writing.** The agent reads freely and mutates only with approval. The
   permission decision lives in the core; each interface renders it.
6. **Structure is explicit and proposed, not invented.** The ontology is data the workspace
   owns. The tool proposes it from evidence in the corpus and the tables; a person accepts
   it; extraction is constrained by it and validated against it.
7. **Context is data too.** The workspace context (definitions, caveats, persona) is stored
   inside the boundary and edited through the interfaces; files are only an import and
   export format.

---

## 4. Architecture

```
+------------------------------------------------------------------------------+
|                                Interfaces                                     |
|  web UI (askama+htmx) | REST | MCP (stdio, HTTP) | TUI | print | desktop (planned) |
+------------------------------------------------------------------------------+
                                       |  in-process calls; no internal network hop
+------------------------------------------------------------------------------+
|                          quack-core (library crate)                           |
|                                                                               |
|  storage/      workspace.rs (DuckDB per workspace, open/create, SQL           |
|               classification, limits, hybrid search), control.rs (control.db), |
|               sessions.rs, context.rs, audit.rs, queries.rs                    |
|  ingestion/    parsers, chunking with metadata, embedding, term index          |
|  analysis/     agent loop and tools, events, policy, text_to_sql (the prompt), |
|               rerank, citations, chart, vector_index                           |
|  ontology/     model in tables, induction (propose), candidates, versions      |
|  graph/        extraction guided by ontology, resolution, traversal, store     |
|  llm/          rig providers, the turn loop, oauth (PKCE / device code)        |
|  import.rs     Postgres, SQLite, and HTTP snapshots (no ATTACH)                |
|  okf.rs        Open Knowledge Format export and import                         |
|  config.rs, crypto.rs, error.rs, progress.rs                                   |
+------------------------------------------------------------------------------+
                                       |
+------------------------------------------------------------------------------+
|  DuckDB (bundled, static, `bundled` + `json` features; no runtime extensions)  |
|  vectors: exact cosine scan; keywords: quack's own BM25 term index            |
|  SQLite (sqlx, bundled): control.db (server access control only)             |
+------------------------------------------------------------------------------+
```

Crates:

```
crates/
  quack-core/      the engine
  quack/           the one binary: `quack serve`, `quack mcp`, terminal session,
                   print mode, admin (`quack desktop` is planned, section 11.6)
    src/
      main.rs          clap surface, crypto provider install, logging
      terminal/        ratatui session
      print.rs         one-shot mode and output formats
      server/          axum router, REST, SSE, MCP over HTTP, templates, embedded assets
      mcp.rs           the MCP server, served on stdio and by server/mcp_http.rs
      admin.rs         user, token, member, and audit subcommands
      graph_cli.rs, ontology_cli.rs   the `graph` and `ontology` subcommands
```

Two crates, no Cargo features. Surfaces are subcommands, not build variants.

Every interface calls the same core entry points:

| Operation | Core | Web | REST | MCP | TUI / print |
|-----------|------|-----|------|-----|-------------|
| Ask | `llm::run_turn` (event stream) | SSE fragments | SSE or JSON | `query` tool | inline / stdout+stderr |
| Retrieve | `WorkspaceDb::search_hybrid_chunks`, `analysis::rerank` | via agent, `/search` page | `GET .../search` | `search` tool | via agent |
| SQL | `WorkspaceDb::execute_query{,_capped}` | SQL page | `POST .../sql` | `sql` tool | `/sql`, `-q` |
| Ingest | `ingestion::ingest_file` | upload | `POST .../documents` | - | `/ingest`, `quack ingest` |
| Graph | `graph::traverse::{neighborhood,path}` | graph page | `GET .../graph/*` | `search_graph` | `/graph`, `quack graph` |
| Ontology | `ontology::store::{current,save,versions,restore}`, `ontology::candidates` | ontology page | `.../ontology/*` | resource | `quack ontology` |
| Context | `storage::context::{current,set,history,combined}` | context page | `.../context` | resource | `/context` |
| Permission | `analysis::policy::WritePolicy` | prompt; `write_refused` on the answer | 200, `write_refused: true`, SSE `write_refused` | `write_refused: true` plus a sentence | y/n/a prompt / exit 3 |

The LLM layer is `rig`. `quack-core::llm` builds rig clients from config and exposes
`ChatModel` and `EmbedModel` enums so the rest of the core is provider-agnostic.

### 4.1 Work queues

Every interface is asynchronous: anything slower than a keystroke runs as a job, and the
interface that submitted it stays responsive and reports its status. `quack_core::jobs`
is the one mechanism. A submitted job is `queued` until its lane has room, runs, and ends
`succeeded`, `failed`, or `cancelled`. A job may name a *lane*: jobs that share a lane key
run at most the lane's limit at a time, strictly in submission order (a job's place in the
line is taken when it is submitted, not when its task first runs). A job with no lane
starts at once.

The priority (`crate::priority`, a Tokio task-local) is interactive unless scoped: the job
queue runs `ingest`, `import`, `ontology`, `graph`, and `export` jobs as background, and a
bridge onto another thread (the blocking pool, a job's own thread) carries it across.

The queue does not count jobs against a pool. What is scarce is the resources jobs use,
and each is limited where it is used:

| Resource | Limit | Where |
|----------|-------|-------|
| Model requests, per provider and model | `[providers.NAME].max_concurrent_requests` for each model (1 for Ollama, which serves one request per model unless `OLLAMA_NUM_PARALLEL` says more; 8 for hosted APIs), process-wide, interactive requests first | `llm::LimitedHttp`: every rig client quack builds sends through it; the model is read from the request body; a permit is held from the request until its body is read or its stream ends |
| The workspace's writer connection | one holder at a time, interactive callers first (`storage::writer::Writer`, section 7.4) | long work takes it per step, never across a model call; the terminal and async code reach it only from the blocking pool |
| Reads | the reader pool (`[analysis].reader_pool_size`) | `ReaderDb` |
| Uploads per workspace (server) | `[server].workers_per_workspace` | the `ingest:{workspace}` lane |

A turn therefore holds nothing while it waits for the user's answer to a write prompt or
runs a tool, and a quick `SELECT` never waits behind chat. Requests carry a priority
(`llm::limit::Priority`, a Tokio task-local): `run_turn` and `embed_query` run interactive,
everything else (ingest embeddings, extraction, proposals) background, and a freed permit
goes to the oldest interactive waiter before any background one, so a question never
queues behind a whole ingest. rig's streaming loop drains a
model response before it runs the tool calls in it, so a tool that calls the same
provider (the query embedding, the model reranker) never waits on a permit its own turn
still holds. `[analysis].extraction_concurrency` and `[ingestion].embedding_concurrency`
stay as the width of one run's pipeline; the provider limit caps them across runs.

| Work | Kind | Lane | Where |
|------|------|------|-------|
| An agent turn | `chat` | `session:{id}`, serial: a turn's history includes the answer before it | TUI, web chat, REST `query` |
| A typed statement | `sql` | none | TUI |
| A file or pasted text | `ingest` | `ingest:{workspace}`, `[server].workers_per_workspace` wide (server); none (TUI) | upload, `/ingest` |
| An external import | `import` | none | `/import` |
| Graph extraction | `graph` | `graph:{workspace}`, serial | graph page, REST, `/graph extract` |
| The ontology document pass | `ontology` | `ontology:{workspace}`, serial | ontology page, REST, `/ontology propose --documents` |
| Bundle and context exports | `export` | none | `/okf`, `/context export` |

A job carries a v7 id, a short number for people to type (`/cancel 3`), its kind and label,
the workspace, the submitting user, its lane, progress (`done` of `total`, which
extraction runs report per chunk), the latest status line, and its outcome (a one-line
summary or the error). Every change goes out on a broadcast channel as a snapshot: the
terminal's job strip and `/jobs`, the web console's Jobs page, and
`GET .../jobs/stream` all read that one source. Cancelling a queued job ends it without
running; a running one sees its cancel token (an agent turn is then recorded as cancelled,
as with `Esc`; a statement is interrupted through `storage::workspace::QueryCanceller`,
which only ever interrupts the connection while that job's statement holds it) and stops
at its next checkpoint, or finishes when its work has none (an ingest mid-embedding).
Quitting the terminal cancels every job and waits a few seconds for them to stop; work
without a checkpoint runs on a detached thread and never holds the process open. Work
whose end records something (an upload's document status, an extraction's closing audit
row) records it for a job cancelled while queued as well, so nothing is left `queued`.

The registry is in memory. A job's label can name a file or quote a question, which is
workspace content (section 5), so it never reaches `control.db`; a restart forgets it, and
the durable record of what a job did is the document, table, session, or audit row it
wrote. Uploads a previous process left `queued` are marked failed when the workspace is
next opened. Jobs share the workspace's one writer connection with everything else
(section 7.4); long work (graph extraction, the document pass, ingestion) takes the writer
only around each database step, never across a model call, so a question asked meanwhile
records its turn between those steps.

---

## 5. Workspaces and Storage

### 5.1 Layout

A workspace is a directory, and the directory is the unit of access control:

```
<workspace>/
  data.duckdb      # everything classified: tables, chunks, graph, ontology, context,
                   # documents registry, sessions, messages, workspace audit detail
  files/           # uploaded and ingested files
```

Nothing else about the workspace's contents exists anywhere. Back up, move, share, or
destroy the directory and you have done so to the whole workspace.

Named workspaces live under `<data_dir>/workspaces/<id>/` and are what every interface
serves. A workspace is always reached by name through the control plane
(`resolve_workspace`); there is no directory-local workspace. An earlier design had the TUI
walk up from the current directory to a `.quack/` folder the way `git` finds `.git/`; that
was never built, and nothing in the code looks for one.

### 5.2 Isolation

One DuckDB file per workspace; queries cannot cross workspaces. `ATTACH` is impossible
rather than discouraged: `WorkspaceDb::confine_to` sets `allowed_directories` to the
workspace directory alone, turns off `enable_external_access` and
`allow_persistent_secrets`, applies the `memory_limit` and `threads` caps, and then sets
`lock_configuration` before any user or agent statement runs. External data arrives through
`quack import`, which snapshots rows into an ordinary workspace table (section 6.2). A workspace's `allowed_providers`
restricts which LLM providers may see its data, so a `restricted` workspace can be pinned
to the local Ollama provider. In server mode, opening a workspace file requires membership
recorded in `control.db` (section 5.5); the file is never opened on behalf of a non-member.

### 5.3 Workspace context

The workspace context is AnythingLLM's workspace system prompt generalized: persona and
tone instructions plus the definitions the data does not carry. It is stored in
`_quack_context` inside the workspace database, loaded into the system prompt on every
turn after the schema and document blocks, capped at a token budget (default 4,000).

```markdown
# Claims workspace

Answer as a claims analyst. Prefer the policy documents over the FAQ when they disagree.

## Data
`claims.csv` is one row per line item; group by `claim_id` for per-claim figures.
`amount` is in cents.

## Definitions
- Open claim: status IN ('filed', 'under_review').
- Loss ratio: SUM(paid) / SUM(premium), by policy year.
```

Edited on the web UI context page (`member`+, audited), via `PUT .../context`, or with
`quack context edit` (opens `$EDITOR` or `$VISUAL` on a temp file and stores the result).
`quack context export|import FILE` moves it as Markdown (`-` for stdout or stdin);
`quack context history` lists versions; `/context` in the terminal shows it. A
`~/.config/quack/context.md` is an unclassified global prefix loaded before the workspace
context. The agent never writes it. Every distinct edit is a new version in
`_quack_context`; importing identical content records nothing.

### 5.4 `data.duckdb`

Internal tables are prefixed `_quack_`, hidden from the agent's table listing, and
refused outright to user and agent SQL: `classify_user_statement` returns "internal tables
are not accessible" and there is no opt-in flag. IDs are UUID v7 via
`uuid::Uuid::now_v7()`.

```sql
-- workspace metadata
CREATE TABLE _quack_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
  -- schema_version, embedding_model, embedding_dimension,
  -- graph_built_with_ontology_version, graph_drift

CREATE TABLE _quack_context (
    version    INTEGER PRIMARY KEY,
    content    TEXT NOT NULL,
    edited_by  TEXT,
    edited_at  TIMESTAMP DEFAULT now()
);

-- documents and chunks
CREATE TABLE _quack_documents (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    title         TEXT,
    mime_type     TEXT,
    size_bytes    BIGINT,
    sha256        TEXT,                    -- dedup on re-upload
    source        TEXT,                    -- upload | paste | path | stdin | import | okf
    status        TEXT DEFAULT 'pending',  -- pending | processing | ready | error
    error_message TEXT,
    pinned        BOOLEAN NOT NULL DEFAULT false,
    chunk_count   INTEGER,
    tables        JSON,                    -- tables a structured document created
    ingested_by   TEXT,
    ingested_at   TIMESTAMP DEFAULT now()
);
-- sha256, source, status and tables are nullable because older workspaces gain them
-- through ALTER TABLE ... ADD COLUMN IF NOT EXISTS on open.

CREATE TABLE _quack_chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    heading     TEXT,                       -- nearest preceding heading, if any
    page        INTEGER,                    -- for paginated sources
    token_count INTEGER,
    embedding   FLOAT[N]                    -- N fixed per workspace, recorded in _quack_meta
);
-- No vector index: search is an exact cosine scan with the core
-- array_cosine_distance function (see section 15).

CREATE TABLE _quack_terms (              -- BM25 index quack maintains at insert time
    chunk_id TEXT NOT NULL,
    term     TEXT NOT NULL,              -- Snowball-English stem of an alphanumeric run
    tf       INTEGER NOT NULL
);
CREATE INDEX _quack_terms_term_idx ON _quack_terms (term);
CREATE INDEX _quack_terms_chunk_idx ON _quack_terms (chunk_id);

-- ontology (section 6.3)
CREATE TABLE _quack_ontology_versions (
    version    INTEGER PRIMARY KEY,
    snapshot   JSON NOT NULL,               -- full ontology at this version, for diff and export
    author     TEXT,
    note       TEXT,
    created_at TIMESTAMP DEFAULT now()
);
CREATE TABLE _quack_ontology_classes (
    id           TEXT PRIMARY KEY,          -- snake_case, stable
    parent_id    TEXT,                      -- single inheritance; NULL only for 'entity'
    label        TEXT NOT NULL,
    description  TEXT,
    key_property TEXT,                      -- property used as the natural key, if any
    since_version INTEGER NOT NULL
);
CREATE TABLE _quack_ontology_relations (
    id            TEXT PRIMARY KEY,
    label         TEXT NOT NULL,
    description   TEXT,
    domain_class  TEXT NOT NULL,            -- satisfied by any subclass
    range_class   TEXT NOT NULL,
    since_version INTEGER NOT NULL
);
CREATE TABLE _quack_ontology_properties (
    id            TEXT NOT NULL,
    class_id      TEXT NOT NULL,            -- inherited by subclasses
    label         TEXT NOT NULL,
    type          TEXT NOT NULL,            -- string | number | date | enum | boolean
    enum_values   JSON,
    since_version INTEGER NOT NULL,
    PRIMARY KEY (id, class_id)
);
CREATE TABLE _quack_ontology_mappings (     -- table rows -> nodes and edges (section 6.3)
    id            TEXT PRIMARY KEY,
    table_name    TEXT NOT NULL,
    class_id      TEXT NOT NULL,
    key_column    TEXT NOT NULL,
    property_map  JSON NOT NULL,            -- {column: property_id}
    relation_map  JSON NOT NULL,            -- [{column, relation_id, target_class, target_key}]
    since_version INTEGER NOT NULL
);
CREATE TABLE _quack_ontology_candidates (   -- proposals awaiting review (section 6.5)
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,             -- class | relation | property | mapping | merge
    proposal     JSON NOT NULL,             -- the row(s) that would be created or changed
    evidence     JSON NOT NULL,             -- counts, example mentions, chunk/row ids
    confidence   FLOAT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',   -- pending | accepted | rejected | superseded
    proposed_by  TEXT NOT NULL,             -- induction run id
    decided_by   TEXT,
    decided_at   TIMESTAMP
);

-- knowledge graph (section 6.4)
CREATE TABLE _quack_graph_nodes (
    id                 TEXT PRIMARY KEY,
    label              TEXT NOT NULL,
    normalized_label   TEXT NOT NULL,       -- lowercased, whitespace-collapsed
    class_id           TEXT NOT NULL,
    properties         JSON,                -- validated against the class's properties
    embedding          FLOAT[N],            -- label + class, for fuzzy resolution
    provisional        BOOLEAN NOT NULL DEFAULT false,   -- built from an unreviewed ontology
    UNIQUE (normalized_label, class_id)
);
CREATE TABLE _quack_graph_edges (
    id                 TEXT PRIMARY KEY,
    source_node_id     TEXT NOT NULL,
    target_node_id     TEXT NOT NULL,
    relation_id        TEXT NOT NULL,
    weight             DOUBLE DEFAULT 1.0,
    properties         JSON,
    provisional        BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX _quack_graph_edges_source_idx ON _quack_graph_edges (source_node_id);
CREATE INDEX _quack_graph_edges_target_idx ON _quack_graph_edges (target_node_id);
CREATE TABLE _quack_provenance (            -- every node and edge traces to text or a row
    subject_id  TEXT NOT NULL,              -- node or edge id
    document_id TEXT,
    chunk_id    TEXT NOT NULL DEFAULT '',   -- '' rather than NULL: all three are in the key
    table_name  TEXT NOT NULL DEFAULT '',
    row_key     TEXT NOT NULL DEFAULT '',
    confidence  DOUBLE,
    PRIMARY KEY (subject_id, chunk_id, table_name, row_key)
);
CREATE TABLE _quack_graph_extracted (       -- which chunks have been through extraction
    chunk_id         TEXT PRIMARY KEY,
    ontology_version INTEGER NOT NULL,
    nodes            INTEGER NOT NULL,
    edges            INTEGER NOT NULL,
    extracted_at     TIMESTAMP DEFAULT now()
);
CREATE TABLE _quack_graph_merges (          -- entity-resolution proposals for review
    id           TEXT PRIMARY KEY,
    keep_node_id TEXT NOT NULL,
    drop_node_id TEXT NOT NULL,
    distance     DOUBLE NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',   -- pending | accepted | rejected
    decided_by   TEXT,
    decided_at   TIMESTAMP,
    UNIQUE (keep_node_id, drop_node_id)
);

-- sessions (section 8)
CREATE TABLE _quack_sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    mode       TEXT NOT NULL DEFAULT 'chat',   -- chat | query
    model      TEXT NOT NULL,
    created_by TEXT,
    shared     BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);
CREATE TABLE _quack_messages (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    role       TEXT NOT NULL,                -- user | assistant | tool
    content    TEXT NOT NULL,
    metadata   JSON,                         -- tool, sql, rows, duration_ms, citations, chart
    created_at TIMESTAMP DEFAULT now(),
    UNIQUE (session_id, seq)
);

-- workspace audit detail (section 12): what was done, inside the boundary
CREATE TABLE _quack_audit (
    id        TEXT PRIMARY KEY,             -- UUID v7, time-ordered; same id as control.db row
    timestamp TIMESTAMP DEFAULT now(),
    user_id   TEXT,
    action    TEXT NOT NULL,
    detail    JSON                          -- SQL executed, file names, context diff, ...
);
```

Opening a workspace whose recorded `embedding_dimension` differs from the configured
provider's is an error with a clear message, never a silent mismatch — unless no chunk has
an embedding yet, in which case the workspace adopts the new dimension and retypes the
`embedding` columns of `_quack_chunks` and `_quack_graph_nodes` through NULL. Session and
message writes are small and frequent; a workspace has one writer connection, so they
serialize with ingestion writes rather than running beside them, though in the writer's
interactive line, ahead of any waiting background write (section 4.1).

### 5.5 `control.db` (server only, SQLite, sea-query queries, SQL-file migrations)

At `<data_dir>/control.db`, opened by `quack serve` and the admin
subcommands. It answers one question, who may open which workspace, and holds nothing that
reveals what a workspace contains.

```sql
-- applied schema versions, written by sqlx::migrate! (docs/migrations.md)
CREATE TABLE _sqlx_migrations (
    version        BIGINT PRIMARY KEY,
    description    TEXT NOT NULL,
    installed_on   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    success        BOOLEAN NOT NULL,
    checksum       BLOB NOT NULL,        -- SHA-384 of the migration file
    execution_time BIGINT NOT NULL
);

CREATE TABLE users (
    id            TEXT PRIMARY KEY,          -- UUID v7
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT,                      -- argon2id; NULL for OIDC users
    oidc_subject  TEXT UNIQUE,
    is_admin      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE workspaces (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL UNIQUE,  -- directory under <data_dir>/workspaces/
    classification    TEXT NOT NULL DEFAULT 'internal',
    allowed_providers TEXT,                  -- JSON array, NULL = all
    created_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE members (
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role         TEXT NOT NULL DEFAULT 'member',   -- owner | member | viewer
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (workspace_id, user_id)
);

CREATE TABLE api_tokens (
    token_hash   TEXT PRIMARY KEY,           -- SHA-256
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    scopes       TEXT NOT NULL DEFAULT '["read"]',   -- read | write | admin
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at   TEXT,
    last_used_at TEXT
);

CREATE TABLE audit_log (                     -- who accessed what, when, how, and whether it was allowed
    id            TEXT PRIMARY KEY,          -- UUID v7; matches _quack_audit.id in the workspace
    timestamp     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_id       TEXT,                      -- NULL for anonymous / failed login
    token_hash    TEXT,                      -- when authenticated by API token
    workspace_id  TEXT,                      -- NULL for admin and auth actions
    action        TEXT NOT NULL,   -- login | logout | open | query | sql | search | graph | ingest | delete
                                   -- | export | import | context | ontology | propose | extract
                                   -- | session_read | share | okf | member | token | workspace | admin
    resource_type TEXT,                      -- document | table | session | ontology_version | user | token | ...
    resource_id   TEXT,                      -- opaque id or name; never content
    outcome       TEXT NOT NULL,             -- allowed | denied | error
    channel       TEXT NOT NULL,             -- web | api | mcp | tui | desktop | cli
    client_addr   TEXT,
    request_id    TEXT
);
CREATE INDEX audit_log_user_ts ON audit_log (user_id, timestamp);
CREATE INDEX audit_log_workspace_ts ON audit_log (workspace_id, timestamp);
```

`audit_log` is the access record and is required in every deployment mode except
`--local`. It is append-only: no `UPDATE` or `DELETE` path exists in the code, and the
sqlx connection for it is opened without those statements in any query. Every request
that names a workspace writes a row, including denied ones: an attempt by a non-member to
open a workspace, a `viewer` trying to write, an expired token. `resource_id` is an opaque
UUID or a table name, which is enough to answer "who read document X" without revealing
what document X says. An admin can see that a user ran a query against a workspace and
which session it belonged to; only a member of that workspace can see the query text,
which lives in `_quack_audit`. There is no retention or pruning path, which is what
append-only means here: rows are kept until an operator retires the file. `quack audit`
filters by user, workspace, action, outcome, and time range and exports as NDJSON
(`--json`) or CSV (`--csv`); `GET /api/v1/admin/audit` takes the same filters, caps `limit`
at 1000, and answers JSON only.

Workspace names themselves are treated as unclassified; if a deployment needs opaque
names, the `name` column is the directory name and a display name lives in `_quack_meta`.

### 5.6 SQL construction rules

- `control.db`: sea-query for every runtime query. Schema changes are literal SQL files
  under `crates/quack-core/migrations/`, run by `sqlx::migrate!`; a handful of `PRAGMA` and
  `sqlite_master` statements are hand-written strings.
- `data.duckdb` internal statements: parameterized via `duckdb::params!`. Table and column
  names go through `quote_ident`, the single identifier path; the only other interpolation
  is the workspace's `embedding_dimension`, a `u32` field, into `FLOAT[N]`. Vector literals
  go through `embedding_literal` and are bound, not interpolated. File paths are bound as
  parameters to `read_csv_auto(?)`. Graph traversal issues one constant query per frontier
  with bound parameters — there is no recursive CTE (section 6.4).
- Agent-generated and user-typed SQL: executed as-is through the permission layer
  (section 7.4). Never assembled by application code.

---

## 6. The Knowledge Engine

### 6.1 Documents and vectorization

**Sources.** File upload (web, REST, desktop), paste (web: a text box that becomes a
document), path (TUI, CLI), stdin (print mode), and an OKF bundle as a directory or a tar
(section 17 item 13). URL, GitHub, and Confluence connectors are deferred (section 18).

**OKF bundles are a one-way knowledge export.** `quack okf export` writes `index.md` from
the context, a schema-and-samples stub per table, a metadata stub per document, the
ontology as Markdown files plus an exact JSON snapshot (`ontology/ontology.md`), one entity
file per graph node (with its id, links that resolve to the target's file, and provenance),
and `log.md` from the ontology versions only: the audit detail never leaves the workspace,
and neither table data nor document text is in the bundle. Importing a bundle (`quack
ingest DIR`, `POST .../documents` with a tar) ingests every concept file that carries text
as a document (a foreign bundle's files; quack's own stubs are marked `generator: quack`
and skipped), restores the ontology snapshot when the workspace has none, proposes the
bundle's types and links as candidates otherwise, and offers `index.md` as the context. The
graph is rebuilt with `quack graph extract` once the tables and documents are back.

**Parsing.**

| Type | Parser | Extracted metadata |
|------|--------|--------------------|
| PDF | `pdf_oxide` | page numbers, Info title; an unreadable page is skipped and counted, never the rest of the file |
| Markdown, plain text | direct | headings (ATX and setext) |
| HTML | `scraper` (html5ever) | headings, `<title>` |
| DOCX | `zip` + `quick-xml` | headings from `Heading N` and `Title` styles, core title |
| PPTX | `zip` + `quick-xml` | one section per slide, slide title as heading, slide number as page |
| CSV, Parquet, JSON, JSONL, XLSX | DuckDB (section 6.2) | become tables, not chunks |

Scanned PDFs (no text layer) are detected and reported as `error: no extractable text`;
OCR is deferred.

**Chunking.** A fixed token window: 512-token target, 64-token overlap, stepping by the
difference. A sectioned source (Markdown, HTML, DOCX headings, PPTX slides, plain text) is
split at its section boundaries first, so a chunk never spans two sections, but within a
section the window ignores paragraph and sentence boundaries. The nearest preceding heading
is stored on the chunk and prepended to its embedding input. A PDF is one continuous text:
its pages are joined by a blank line and windowed as a whole, so a paragraph split by a page
break stays in one chunk; each chunk records the page its first token lies on, and carries
the document's Info title (else the filename stem) as its heading, since a PDF has no
heading of its own to give the embedding context. Token counts via `tiktoken`
(`cl100k_base`).

**Embedding.** Batches of `[ingestion].embedding_batch_size` (64 by default, at least one)
through the configured embedding provider, with `[ingestion].embedding_concurrency` (2)
requests in flight; each batch's vectors are written as one transaction as it returns, so
the writes overlap the requests still running. The provider sets the ceiling: an
OpenAI-compatible endpoint answers concurrent batches in parallel, while Ollama's runner
embeds one input at a time whatever the batch size or concurrency (about 14 chunks a second
for a 0.6B model on Apple silicon, measured) unless `OLLAMA_NUM_PARALLEL` is raised, and a
smaller embedding model is the other lever. Every ingest logs the chunk count, batches,
seconds, and chunks per second (`embedded chunks`), and `quack ingest` prints them. There
is no index to build on either side: vector search is an exact scan, and the term rows for a chunk are appended as
it is inserted. A full term rebuild happens only when an older workspace is opened
(schema version below 6). Re-uploading a file with the same SHA-256 is a no-op with a
message.

**Hybrid retrieval.** A query runs both an exact cosine scan over `embedding` (core
`array_cosine_distance`) and a BM25 search over the terms quack tokenized at ingest
(`_quack_terms`, scored in SQL; no DuckDB extension). Tokens are lowercased alphanumeric
runs passed through the Snowball English stemmer (`rust-stemmers`), so `renewals` meets
`renewal` on the keyword side; the query side tokenizes identically. A run joined by
`-`, `.`, `_`, `/`, or `:` with no whitespace, such as `POL-8841`, additionally indexes its
punctuation-stripped, unstemmed form (`pol8841`) alongside the split pieces (`pol`, `8841`),
so a query for that identifier ranks a chunk containing it ahead of one that merely contains
`pol` and `8841` apart (`storage::workspace::tokenize`, issue #77). A `"..."` quoted phrase
in a query is an exact adjacency requirement: since `_quack_terms` carries no term
positions, BM25 still ranks candidates by the phrase's own tokens, over-fetched, and a
post-filter keeps only the chunks whose content or heading contains the phrase as a
case-insensitive, whitespace-normalized substring; a phrase matching nothing returns no
keyword results rather than falling back to the unfiltered ranking. The term index is
rebuilt on open when a workspace predates the stemmer or the joined identifier form. Each
ranking is over-fetched to twice `top_k` (more when a phrase is present), fused with
reciprocal rank fusion (`k = 60`), and the top `k` chunks (default 8) are returned. A
reranking hook (`analysis::rerank::Reranker`) sits between fusion and the answer: off by
default (`[retrieval].rerank = "none"`), or `"model"`, which over-fetches
`rerank_candidates` (24) and has the chat model order them listwise in one tool-less call,
so an air-gapped deployment gets reranking from the model it already runs. A failed ranking
call keeps the fused order and the tool step says so. A cross-encoder provider fits the
same trait. This is the main retrieval quality improvement over the pgvector setup, where
keyword-exact questions (part numbers, policy IDs) go unanswered.

**Citations.** Every retrieved chunk carries `document_id`, `filename`, `title`, `page`,
`heading`, and its fused score. The agent cites by `[n]` markers that map to these chunks;
the core validates that every marker references a chunk retrieved in that turn, strips
those that do not (including provider "channel" markers that leak into the text), and
renumbers the survivors from 1 in order of first use. Interfaces render citations as links to the document and page.

**Pinned documents.** A pinned document's full text is injected into the system prompt
each turn rather than retrieved, bounded by its own `[retrieval].pinned_token_budget`
(default 8,000). A document that would exceed what is left is skipped with a visible
"(omitted)" line rather than truncated.

### 6.2 Tables and analytics

| Type | Mechanism | Result |
|------|-----------|--------|
| CSV, TSV, Parquet, JSON, JSONL | `read_csv_auto` / `read_parquet` / `read_json_auto` | Table in `data.duckdb` |
| Excel `.xlsx`, `.xls`, `.ods` | `calamine` (pure Rust) writes each sheet as CSV under `files/` for `read_csv_auto` | One table per data sheet: `<stem>` for one sheet, `<stem>_<sheet>` otherwise; recorded on the document row so deleting it drops them |
| stdin (print mode) | sniffed | Temporary table `stdin` |
| Postgres, SQLite | `quack import URL --table T (--from SOURCE_TABLE \| --query SQL) [--limit N]`, `POST .../import`, the Tables page form, `/import` in the terminal: sqlx runs the query on the source with every column cast to text, the rows pass through `files/<table>.csv` and `read_csv_auto`, so `DuckDB` sniffs the types and the table is a document (source `import`, title the redacted URL) that can be deleted like any other. The password in the URL is used once and never stored; audit rows carry the redacted URL. Capped by `[import].max_rows` (a file is cut to it after the load), `max_download_mb`, and `timeout_seconds`. `sqlite:` paths inside `[general].data_dir` (`control.db`, the workspace files) are refused for every caller. The CLI, the terminal, and `quack serve --local` run as the owner and reach any other source; `quack serve` with logins refuses `sqlite:` paths unless `[import].allow_local_files` is on and, unless `allow_private_hosts` is on, resolves the host first, refuses loopback, private, link-local, and metadata addresses, pins the connection to the checked addresses, and does not follow redirects. | Table in `data.duckdb`, a snapshot of the source at import time |
| CSV, Parquet, JSON, XLSX over HTTP(S) | The same command with an `http(s)://` URL: reqwest fetches the file and it goes through the usual reader under the requested table name | Table in `data.duckdb` |
| MySQL, S3 | Not yet: MySQL needs the sqlx driver enabled and its identifier quoting; S3 needs request signing (the `object_store` crate is the candidate). The scanner and httpfs extensions stay out (section 15). | — |

Table naming: sanitized file stem; on collision the web UI and TUI ask (replace, rename,
skip), the API and print mode require an explicit name. The prompt describes tables live on
every build, with no cache: the first 25 tables get columns (40 at most) and three sample
rows, tables wider than 20 columns get no samples, and the rest are listed by name alone.

Every SQL statement, whether the agent's or the user's, passes through classification and
resource limits (section 7.4).

### 6.3 Ontology

The ontology is the schema of the knowledge graph: the classes entities may have, the
relations that may connect them, and the properties each may carry. It exists so that
extraction is constrained and consistent across thousands of documents, so that "all
organizations" can include subclasses, and so that a domain team can describe their world
once instead of correcting the model per chunk.

It is stored in the `_quack_ontology_*` tables of the workspace database (section 5.4). It
is classified content: the class and relation names alone say what a workspace is about.
It travels with the workspace and is governed by the workspace's access control.

**Model.** Single inheritance from the implicit root class `entity`; relations have a
domain and a range class, each satisfied by any subclass; properties are typed
`string | number | date | enum | boolean` and inherited; every ontology implicitly contains
the `mentions` relation (`entity` to `entity`) so extraction never has to invent one. Ids
are `snake_case` and stable; a rename is a new id plus a migration of nodes and edges.

**Interchange format.** `quack ontology export` and `import`, `GET/PUT .../ontology`, and
the web editor's "download" and "save" move the ontology as JSON (the same shape as the
stored snapshot). This is how a domain pack (an insurance ontology, a legal ontology) is
shared between workspaces, diffed in review, or seeded into a new workspace. The file is
never the source of truth. The example below is that shape written as YAML for brevity.

```yaml
version: 3
classes:
  - { id: person,       parent: entity,       properties: [title, email] }
  - { id: organization, parent: entity,       properties: [industry, country] }
  - { id: vendor,       parent: organization }
  - { id: policy,       parent: entity,       key: policy_number, properties: [policy_number, effective_date] }
  - { id: claim,        parent: entity,       key: claim_id, properties: [claim_id, amount, status] }
relations:
  - { id: works_at,      domain: person, range: organization }
  - { id: issued_by,     domain: policy, range: organization }
  - { id: filed_against, domain: claim,  range: policy }
properties:
  - { id: amount,         type: number }
  - { id: effective_date, type: date }
  - { id: status,         type: enum, values: [filed, under_review, paid, denied] }
mappings:
  - table: claims
    class: claim
    key: claim_id
    properties: { amount: amount, status: status }
    relations:
      - { relation: filed_against, column: policy_id, target_class: policy, target_key: policy_number }
```

**Built-in default.** A general ontology (`person`, `organization`, `place`, `event`,
`product`, `document`, `concept`; relations `works_at`, `located_in`, `part_of`,
`produced_by`, `occurred_at`, `mentions`) is installed into a new workspace's tables as
version 1 when the graph is first enabled and no proposal has been accepted.

**Where it is used.**

1. *Extraction prompt.* Classes with parents, relations with domain and range, and property
   definitions are rendered into the extraction prompt; the model must return only ids from
   the ontology. Anything outside it is dropped and counted (section 6.5, drift).
2. *Validation.* An extracted node or edge is checked before it is written: the class
   exists, the relation exists, and its domain and range hold under inheritance.
   `revalidate` re-applies the same three checks after an ontology change. Property types
   and enum values are validated on the ontology document itself, not on instance data —
   `upsert_node` stores the properties JSON as given.
3. *Query expansion.* `search_graph(class: organization)` matches `vendor` too. The agent's
   prompt includes a compact rendering of the ontology so it can ask class-aware questions.
4. *Table mapping.* A mapping turns a table's rows into nodes and its foreign-key-like
   columns into edges without an LLM. This is the bridge between the analytics and graph
   substrates.

**Versioning.** Every accepted change writes a new row to `_quack_ontology_versions` with a
full snapshot. `_quack_meta.graph_built_with_ontology_version` records what the graph was
built with; when it lags, the graph is stale and the interfaces offer re-extract (cost
shown first) or revalidate (fast; drops nodes and edges that no longer validate). Any
version can be diffed against another or restored. Deleting a document also removes the
graph nodes and edges whose only provenance was that document or the tables it loaded,
with their provenance rows and its files under `files/`; a mapping whose table is gone
stays in the ontology (saves still succeed), extraction skips it, and `graph status`
lists it under `missing_tables`.

### 6.4 Knowledge graph

**Extraction from documents.** Runs on demand (`quack graph extract`, `POST
.../graph/extract`, the graph page). Each chunk goes to the chat model with the ontology-derived
prompt and must return JSON `{nodes: [{label, class, properties}], edges: [{source, target,
relation, properties}]}`. Parsing is deliberately lenient — the first `{` to the last `}`,
with every list defaulting to empty — because models wrap JSON in prose; a chunk that still
fails is logged and skipped, never retried in a loop. Cost (chunk count, model) is shown before extraction
starts and the operation is permission-gated.

**Extraction from tables.** Ontology mappings turn rows into nodes and edges
deterministically, with provenance `table_name` and `row_key`. Re-running is idempotent.

**Entity resolution.** Nodes are merged on `(normalized_label, class_id)`. A second pass
proposes merges for nodes of the same class whose label embeddings are within a cosine
threshold (default 0.08) and whose labels share a token, looking at each node's five
nearest neighbours; a pair within `auto_merge_threshold` (default 0.02) merges on the spot,
the rest land in `_quack_graph_merges` as pending proposals for review. Provenance rules
the pass: two nodes that both come from keyed table rows are distinct by construction and
are never paired (WEST VIRGINIA is not VIRGINIA), a pair with one keyed side is only ever
proposed with the keyed node kept, and auto-merge applies to two extracted nodes only.
Each merge runs in one transaction. Aliases are kept in `properties.aliases`.

**Extraction from documents** sends each chunk of a ready document to the chat model once:
`_quack_graph_extracted` records every chunk processed (with the ontology version and what
it yielded), so a later run sends only new chunks and `--reset` starts over; a sample of N
takes chunks spaced evenly across documents rather than the first N by ingest order; the
model's raw answer and the parse outcome are logged at debug level.

**Extraction from tables** runs in batches of 5,000 keyed rows: each batch is staged in
Rust (nodes, edges, and row provenance deduplicated, ids minted as UUID v7) and written
with one statement per kind through scratch `_quack_tmp_graph_*` tables, inside one
transaction under the statement timeout; the server releases the workspace lock between
batches and runs one extraction per workspace at a time (a second `POST .../graph/extract`
answers 409). Neighbourhood walks are breadth-first, one query per frontier, and never
visit more than `[graph].max_nodes`.

**Provenance.** Every node and edge has at least one `_quack_provenance` row. Answers from
the graph cite the source chunk or row the same way document answers cite chunks.

**Traversal.** Breadth-first in Rust over plain SQL: one constant `edges_touching` query
per frontier, with bound parameters, no DuckPGQ and no recursive CTE. The CTE this design
originally specified was removed (#48) because it enumerated every simple path out of a hub
before its `LIMIT` applied, which on a hub node never came back.

Entry point resolution: an exact match on `normalized_label` or on a value in
`properties.aliases`, with the class filter applied in the same query; failing that, the
three nearest node embeddings within a distance of 0.25, and nothing if none qualify.
Operations: `neighborhood(entity, hops, relation?)`, `path(a, b, max_hops)` (single-source
BFS with a parent map, bounded by `max_traversal_depth * 2`), `by_class(class, limit)` with
subclass expansion. Limits: `max_traversal_depth` (3) and `max_nodes` (200), the cap on
nodes visited.

**Rendering.** TUI and `quack graph` print a depth-first tree; the web UI renders an
ECharts `graph` series with class-colored nodes and an inspector showing properties and
provenance; the API returns `{nodes, edges, provenance}`.

### 6.5 Ontology induction: discovering and proposing

A workspace rarely starts with a modeled domain. The tool proposes the ontology from
evidence and a person accepts it. Proposals are rows in `_quack_ontology_candidates`;
nothing changes the live ontology until a candidate is accepted.

**Evidence from tables (deterministic, no model calls).**

- Each table proposes a class named from the table; each column proposes a property with
  its type inferred from DuckDB's column type and value profile (`enum` when distinct
  values are few and stable; `date` when the column parses as one).
- A column that is unique and non-null proposes the class key.
- A column whose values overlap heavily (default 80%) with another table's key column
  proposes a relation between the two classes, named from the column by a deterministic
  rule (`policy_id` -> `has_policy`). The whole table-evidence pass makes no model calls.
- The result is also proposed as a mapping, so accepting it makes rows into nodes at once.

**Evidence from documents (open extraction on a sample).**

1. Take a stratified sample of chunks across documents (default 200, configurable), so
   every document contributes.
2. Run extraction unconstrained: free-form entity types, free-form relation names,
   observed attributes with values. Chunks run `[analysis].extraction_concurrency` at a
   time, each call bounded by `[analysis].extraction_timeout_seconds` (a chunk that
   times out is skipped and counted), and every finished chunk is reported: a line on
   stderr in the CLI, a log line in the server. Graph extraction (6.4) runs the same way.
3. Normalize the vocabulary. Raw type and relation names are grouped by snake_case
   singular equality, and near-synonyms are clustered by embedding cosine when an embedding
   model exists (`cluster_threshold`, default 0.9). The cluster's most frequent raw name
   becomes its id; no model call is made for naming.
4. Infer structure. A relation's domain and range are the classes observed at its endpoints
   (generalized to the nearest common ancestor when mixed). Hierarchy is inferred where one
   type's mentions are consistently also labeled with a broader type (`vendor` under
   `organization`). Attributes that recur on a class propose typed properties.
5. Score. Each candidate carries occurrence count, distinct-document count, three example
   mentions with chunk ids, and a confidence from support and cluster tightness.
   Candidates below the support threshold (default 3 documents) are kept as
   `low_support` rather than shown in the main proposal.

**Cost.** Shown before the run: sample size, model, and an estimate of calls — one per
sampled chunk, so 200 for a 200-chunk sample. Chunks of 40 characters or fewer are excluded
from both the estimate and the sample. The run is permission-gated and audited.

**Review.** The web ontology page, `quack ontology review`, and `GET .../ontology/candidates`
show the proposal grouped by kind with evidence inline. Actions per candidate: accept,
rename, merge into an existing class or relation, reparent, reject. Accepting writes a new
ontology version. `PUT .../ontology/candidates/{id}` and `quack ontology accept ID...`
apply the same actions from scripts; `POST .../ontology/candidates` with `{accept: [ids],
reject: [ids]}` decides many at once, as the page's tick boxes do. The page shows fifty
candidates at a time, pending or low-support (a filter link, never hidden). Extend mode
treats a mapped table as covered: its rows belong to the class the mapping names, so no
class or mapping is proposed for it, and a column its mapping already relates proposes no
relation; new columns still propose properties.

**Modes.**

- `propose` (default): from an empty workspace, produce a full draft.
- `propose --extend`: start from the current ontology and propose only additions and
  reparents. This is also what the drift report triggers: constrained extraction counts
  every type or relation the corpus tried to express that the ontology has no place for,
  and once a count crosses a threshold the interfaces show "the corpus wants N things the
  ontology lacks; propose extensions?".
- `propose --from PACK`: seed from an imported domain pack, then extend from evidence.
- `--auto-accept`: accept the proposal and build the graph without review, for a first
  look. Everything built this way is marked `provisional` in the graph tables and the
  interfaces keep a banner up until someone reviews. Provisional graphs are excluded from
  query mode answers.

**Graph proposal follows the same loop.** After an ontology is accepted, constrained
extraction over the full corpus builds the graph, entity resolution proposes merges for
review, and drift feeds the next `propose --extend`. The loop is: propose from evidence,
review, extract, observe drift, propose again.

---

## 7. The Agent

### 7.1 Loop

```
1. Build the system prompt (7.2)
2. Send prompt + trimmed history + user message to the chat model
3. On a tool call:
   a. emit ToolStarted { tool, args }
   b. check permission (7.4); emit PermissionRequired and await the interface's answer
   c. execute; emit ToolFinished { tool, detail, summary, duration_ms }
   d. append the result to history; go to 2
4. On text: emit TextDelta as it streams; validate citations; emit
   TurnComplete { AgentResponse }; then persist the turn
0. Before 2, when the model has to be loaded first (Ollama, cold): emit Status { line }
```

The turn races a `CancellationToken`: a cancelled turn keeps the text streamed so far,
appends a note, reports `cancelled: true`, and is still recorded.

`llm::run_turn` yields these events on a channel. The web UI turns them into HTML
fragments over SSE, REST forwards them as typed SSE events or collects them into one JSON
response, MCP collects them into the tool result, the TUI renders them inline, print mode
writes them to stderr. `max_turns` 15, temperature 0.1. Before the first model call an
Ollama turn asks `GET /api/ps` whether the chat model is already in memory and, when it is
not, emits `Status` ("loading MODEL ..."), since a cold load of a 12 GB model takes seconds
during which nothing else can appear; print mode shows it on the spinner, the terminal as a
system line, SSE as a `status` event.

### 7.2 System prompt

1. Role and behavior for the mode (7.5): retrieve before answering, cite with `[n]`, run
   SQL rather than estimate, state assumptions, ask one clarifying question when the
   request is ambiguous.
2. Tool guidance, the error rule (read a `run_sql` error, fix the statement, run it
   again), and a Friendly SQL reference pinned to the bundled DuckDB version, which the
   prompt states. The reference carries only what the confined connection (7.4) can
   run: no file reads, extensions, or `SET`, and it says so, since the tables block is
   all the data there is. The guidance is one numbered procedure per substrate:
   structured data, document content, and — only when the graph tools are registered —
   how entities relate.
3. Tables block: user-facing tables and views with columns, types, row count, three sample
   rows. Bounded so one wide or narrative table cannot push the guidance and the question
   out of a small window: the first 25 tables are described, the first 40 columns listed
   (the rest counted), sample rows shown only up to 20 columns and cut at 60 characters
   per cell; `describe_table` has the rest.
4. Documents block: every document by filename, title, status and mime type, then the
   pinned documents with their full text (6.1).
5. Ontology block, whenever an ontology exists: classes with parents, relations with domain
   and range (compact), capped at 30 items per section with the rest counted — an induced
   ontology has a class per table, and `describe_class` has what the cap left out. The node
   and edge counts, and whether the graph is provisional or stale, follow only when the
   graph has content.
6. Global context prefix, then the workspace context.
7. The permission rules.

The tool guidance names the table, SQL, chart and document tools unconditionally; only the
graph tools are conditional, and mode changes no registration — query mode only drops
provisional graph results.

For Ollama every request carries `num_ctx`: the prompt's estimated tokens plus room for
tool results and the answer (a fixed 8,192-token headroom), rounded up to 8,192, capped by
`[analysis].max_context_tokens` but never below 8,192, because Ollama otherwise loads the
model with a 4,096-token window and silently truncates the front of the prompt. `num_ctx`
is a load option: a value that differs from what the model is already loaded with forces a
full reload (measured at several seconds for a 20B model), so the step is deliberately
coarse — a growing session's history crosses it at most a few times rather than at every
2,048 tokens — and every request also carries `keep_alive` (30 minutes), since nothing did
before and a gap between tool calls or turns otherwise pays the same reload once Ollama's
own default (5 minutes) lapses. Embedding requests go through quack's own `/api/embed`
client (`llm::OllamaEmbedder`) rather than rig's, which sends neither: they carry the same
`keep_alive`, so the embedding model does not lapse between the query embedding and the chat
call of one turn, and a `num_ctx` sized to a chunk (twice `[ingestion].chunk_size_tokens`,
rounded up to a power of two, never below 2,048) instead of the model's full length, which
Ollama otherwise loads it with (32k for qwen3-embedding: 5.8 GB of cache against 2.1 GB,
measured, at the same throughput). On a host where the two models at full size would not
both fit, that difference is what stops them evicting each other every turn. A turn the model derails (a call to
a tool that does not exist, the `max_turns` limit) or that fails after text streamed is
still a turn: the streamed text is kept, a parenthetical note says what happened, and the
turn is recorded; only a model that could not be reached at all is an error.

### 7.3 Tools

| Tool | Permission | Description |
|------|------------|-------------|
| `search_documents(query, top_k=8, document_ids?, entity?)` | none | Hybrid retrieval; returns chunks with citation metadata and the entities each was the source of |
| `list_documents()` | none | Registry with status and pinned flag |
| `run_sql(query)` | read: none; write: prompt | Execute SQL; result capped at `max_query_rows` with a trailer that says to narrow it in one statement, a note when the statement repeats an earlier one with only its literals changed (the one-query-per-group loop), and which tool call of `max_turns` this was |
| `describe_table(table_name)` / `list_tables()` | none | Schema and inventory |
| `describe_class(class_id)` | none | One ontology class in full, with how many entities of it the graph holds |
| `search_graph(entity?, class?, relation?, hops=2)` | none | Neighborhood or class listing with provenance |
| `find_path(from, to, max_hops=4)` | none | Shortest relation path between two entities |
| `create_chart(sql, kind, x, y, title)` | none | Runs the SQL, emits a chart spec (section 9) |

Graph tools (`search_graph`, `find_path`) register only when the graph has nodes and
`describe_class` whenever an ontology exists, since the prompt's ontology block is capped
and a class may not be in it; the rest register in every workspace, and the prompt tells
the model what the workspace holds. An `export` tool (`COPY ... TO` under `files/`) is not
built (section 17).

**The substrates cross in the tools, not only in the store.** `search_documents(entity)`
resolves the name against the graph and restricts retrieval to the chunks that entity was
extracted from, and every hit names the entities the graph already took from it, so the
model can follow one into `search_graph`. The `entity` argument is in the tool's schema
only while the graph has nodes, the same condition that registers the graph tools: a model
shown it on a workspace without a graph tries it, is refused, and spends a second round
trip and twice the tokens reaching the same answer (measured live). Graph provenance to a mapped table renders as a
predicate (`"orders" WHERE "order_id" = 'A-42'`) that `run_sql` can run, since the mapping
records the key column. An entity in the graph only from table rows says so rather than
returning nothing.

Both graph tools render each node and edge with the properties the ontology types, bounded
at eight per subject and sixty characters per value. A lookup that finds nothing says which
kind of nothing it found, so the model retries instead of reporting an empty workspace: a
class or relation id the ontology does not define is an error naming the ids that exist
(the contract `search_documents` already has for `document_ids`), a name that matches no
entity comes back with the closest labels in the graph, and a result that query mode
emptied by dropping provisional nodes says the matches exist but are unreviewed.

A result cut short by `max_nodes` says so, and a class listing carries the total it was
capped from (`GraphResult::total_nodes`, `truncated`), because the reader otherwise takes
the cap for the population of the class. Counting is the one graph question traversal
cannot answer — user SQL may not read `_quack_` tables — so `describe_class` reports the
exact count of a class and its subclasses.

### 7.4 Permissions and limits

**Classification.** Before executing `run_sql`, the statement is passed to DuckDB's own
parser via `SELECT json_serialize_sql(?)`. There are three outcomes, not two: serializes
means read, a parse error means invalid and comes back as a syntax error rather than a write,
and anything else means write. `DESCRIBE`, `SHOW`, `SUMMARIZE`, `PIVOT`, `UNPIVOT` and
`EXPLAIN` are read by an explicit allow-list, because DuckDB cannot serialize them.
`COPY`, `INSTALL`, `LOAD`, `ATTACH`, `SET` are always write. Statements that reference
`_quack_` tables are refused regardless.

**Decision by interface and role.**

| Interface | Read | Write |
|-----------|------|-------|
| TUI | run | prompt `y`/`n`/`a` showing the SQL; `a` covers the rest of the turn and the session |
| Print mode | run | refuse unless `--allow-write`; the answer completes and the exit code is 3 |
| Web / REST | run | refuse unless `allow_write: true` from a member with the write scope (a request that asks for `allow_write` without it is 403); a refusal inside the turn is not a failed request: 200 with `write_refused: true` on the response object and a `write_refused` SSE event; the web page shows a banner offering the checkbox |
| MCP | run | refuse unless `quack mcp --allow-write` set the policy at launch (stdio has no tokens); `write_refused: true` in the structured content and a sentence in the text. Over HTTP the token's `write` scope decides |
| Desktop | planned | native confirm dialog (section 11.6) |

Every interface returns the same response object (11.2): `answer`, `citations` (each with
`n`, `chunk_id`, `document_id`, `filename`, `chunk_index`, `page`, `heading`, `label`),
`queries`, `steps`, `graph`, `chart`, `write_refused`, `cancelled`, `usage`, `session_id`,
built by `AgentResponse::to_json`. `AuthRequired` is exit code 4 from every command that
reaches a provider.

`usage` is what the provider reported for the turn — `input_tokens`, `output_tokens`,
`total_tokens`, rig's aggregate over every completion request the turn made, falling back
to the sum of the per-request counts when the turn derailed before a final response. It is
`null` when the provider reported nothing, which a local model often does; zeroes there
would read as a turn that cost nothing. The same counts go on the assistant message's
metadata in `_quack_messages`, so a session export carries them. They are a record, not an
input: the history trim and Ollama's `num_ctx` still run on their own estimate, because
both are computed before the call.

**Limits.** The agent's connection runs with `SET memory_limit` and `SET threads` from
config. A statement runs on the calling thread — for the agent, a `spawn_blocking` one —
while a watchdog thread holds `Connection::interrupt_handle()` and calls `interrupt()` after
`query_timeout_seconds`; a guard disarms the watchdog when the statement returns. User SQL
has the same limits.

**Confinement.** Classification is not enough: `SELECT * FROM read_text('/etc/passwd')`
is a read. So the workspace connection is confined when it opens, before any user or
agent statement: `allowed_directories` is the workspace directory alone (ingestion reads
the originals it copied under `files/`),
`enable_external_access` is off, so file readers, replacement scans, `COPY`, `ATTACH`,
`INSTALL`, and `LOAD` fail anywhere else, `allow_persistent_secrets` is off, and
`lock_configuration` is on, so no later `SET` can widen any of it or lift the limits
above. The in-memory test database gets the same treatment with an empty allow-list.

### 7.5 Chat modes

Per session, `chat` unless set when the session is created (`--mode`, the REST `mode`
field, the web selector, the MCP `mode` argument) and changed only explicitly (`/mode`,
`PATCH .../sessions/{sid}`):

- **chat** - the agent may answer from general knowledge as well as retrieved sources; it
  must still cite when it used a source.
- **query** - the agent must ground every claim in a retrieved chunk, a query result, or a
  graph result; if retrieval returns nothing relevant it says so instead of answering.
  This is AnythingLLM's query mode and the default for classified workspaces. Provisional
  graph results are excluded in this mode.

### 7.6 Visibility

Every tool call renders as one line at start and one at finish, with up to three lines of
detail preview and a "+N more" tail, in every interface:

```
> search_documents "policy exclusions for flood"
  8 chunks, 41 ms   [1] Policy-2024.pdf p.12  [2] Policy-2024.pdf p.13  ...
> run_sql
  SELECT status, COUNT(*) FROM claims GROUP BY 1
  4 rows, 9 ms
```

The web UI shows these as a collapsible steps block above the answer with citations as
links. The TUI shows them inline with `/sql` to reopen the last query. `--verbose` in print
mode includes the full payloads.

---

## 8. Sessions

Every turn belongs to a session in `_quack_sessions` (AnythingLLM's threads). The whole
turn is written at the end, in one call: the user message, one tool message per step with
its metadata, then the assistant message. A session is a complete record and stays inside
the classification boundary.

- Resume: `session_id` on the REST request, `--continue` / `--resume` in the TUI, the
  thread list in the web UI.
- Export: `.sql` (every `run_sql` and `create_chart` statement with the question as a
  comment) or Markdown (questions, tool steps with their summaries and detail, answers, and
  a note where a chart is attached), from every interface. Export is an audited action
  because it moves content across the boundary.
- History sent to the model is trimmed to `history_token_budget` (32,000), oldest first.
  Tool messages are not replayed at all: only the user and assistant text goes back.
- Server mode: sessions carry `created_by`; members see their own, any marked `shared`, and
  any with no creator (started from the CLI or the TUI). `owner` can see all sessions in the
  workspace for audit.

---

## 9. Charts

One small spec produced by `create_chart`, rendered by ratatui in the TUI, mapped to an
ECharts option in the web UI and desktop, emitted as JSON by REST, MCP, and print mode.

```json
{
  "title": "Claims by status",
  "kind": "bar",                       // bar | line | scatter | pie
  "x": { "label": "status", "values": ["filed", "paid"] },
  "series": [ { "name": "count", "values": [120, 340] } ]
}
```

One x axis, one numeric series (the tool takes a single `y` column), at most 200 points —
more than that and the query is refused rather than sampled. A NULL x becomes the label
"NULL" and a NULL y becomes 0; a non-numeric y is an error the model is told about. The
chart's SQL always runs read-only, so charting can never prompt for a write. A chart
attaches to the assistant message that produced it and appears at that point in every
rendering.

---

## 10. LLM Layer

### 10.1 Providers

| `type` | Chat | Embeddings | Notes |
|--------|------|------------|-------|
| `ollama` | yes | yes | Offline default. `base_url` defaults to `http://localhost:11434` |
| `openai` | yes | yes | Also OpenAI-compatible endpoints via `base_url` (vLLM, LiteLLM, Azure OpenAI) |
| `anthropic` | yes | no | Native Messages API with tool use |

`[general].chat_model` and `[general].embedding_model` name `PROVIDER/MODEL` each; a
workspace's `allowed_providers` filters the choice; the session records the model it used.
Changing the embedding model for a workspace requires re-embedding and is a guided
operation, not a config edit.

### 10.2 Authentication

Each provider has an `auth` mode: `none`, `api-key` (from the env var named by
`api_key_env`), or `oauth` (Authorization Code with PKCE against an enterprise IdP, the
access token used as the bearer for the provider endpoint; how Azure OpenAI and internal
gateways are reached where static keys are forbidden).

```rust
pub struct OAuthConfig {
    pub issuer_url: String,          // https://login.microsoftonline.com/{tenant}/v2.0
    pub client_id: String,
    pub scopes: Vec<String>,         // ["https://cognitiveservices.azure.com/.default"]
    pub redirect_uri: String,        // default http://127.0.0.1:19876/callback
    pub device_code: bool,           // force device-code flow (headless, SSH, server)
    pub client_secret_env: Option<String>,   // confidential client, secret from the env
}

pub struct TokenManager {
    provider: String,
    config: OAuthConfig,
    cache: TokenCache,               // <data_dir>/tokens/<provider>.json, mode 0600
    http: reqwest::Client,
    endpoints: OnceCell<Endpoints>,  // discovered once from the issuer
    current: RwLock<Option<CachedToken>>,
    refresh_lock: Mutex<()>,         // one refresh at a time; others await it
}

pub struct CachedToken {
    pub access_token: SecretString,
    pub expires_at: jiff::Timestamp,
    pub refresh_token: Option<SecretString>,
}
```

Lifecycle: acquire via `quack auth login PROVIDER` (browser PKCE, or device code when
`device_code = true`, no browser, or `SSH_CONNECTION` is set; endpoints from
`{issuer_url}/.well-known/openid-configuration`; verifier from aws-lc-rs randomness);
reuse while more than 60 s remain; refresh silently under `refresh_lock`; restart on
refresh failure; in print, ingest, and server modes, where no flow can run, fail with exit
4 / HTTP 503 naming the command to run. Cache encrypted (AES-256-GCM, the provider name as
associated data) with a key in the OS keychain where available (macOS Keychain, the Linux
kernel keyring via `keyutils`, which is always present but in-memory, so a reboot needs a
new login; Windows Credential Manager), else a 0600 key file. `quack auth status` and
`logout`. The `oauth2` crate is used without its bundled HTTP client (that would pull
`ring`); requests go through the same rustls + aws-lc-rs `reqwest` as rig. Scopes must
include `offline_access` where the issuer needs it to return a refresh token.

The server holds one `TokenManager` per OAuth provider, shared across requests. A server
that reaches Azure OpenAI this way is a confidential client and should be registered as
such with a client secret or certificate; `client_secret_env` is honored when set.

### 10.3 Crypto

`quack_core::crypto::install_default_provider()` at the top of `main`, before any TLS use.
On Linux it installs `rustls::crypto::default_fips_provider()` — aws-lc-rs and rustls both
carry the `fips` feature there, so every distributed Linux binary runs on the FIPS-validated
AWS-LC module and the approved cipher suites; naming that function makes dropping the
feature a build error. macOS and Windows install the aws-lc-sys provider, because a FIPS
build links statically only on Linux. `--version` names the module it linked and
`log_provider()` logs it once a subscriber exists (`docs/crypto.md`).

SHA-256, AES-256-GCM and randomness come from aws-lc-rs; password hashing is the RustCrypto
`argon2` crate, salted from `getrandom`. No runtime code links OpenSSL or `ring`:
`deny.toml` bans `openssl`, `openssl-sys` and `native-tls` outright and allows `ring` only
as a build-time dependency of `libduckdb-sys`, which `make crypto-gates` re-checks with
`cargo tree -e normal`.

---

## 11. Interfaces

### 11.1 Web UI

Server-rendered askama templates, Tailwind compiled with the standalone binary (no
Node.js; the built CSS is committed so `cargo build` needs no Tailwind), htmx for
interactivity, ECharts (the full minified build, vendored; a trimmed custom build needs
Node and can replace it later), all embedded with `rust-embed`. The chat's permission
step is an "allow the agent to change tables" checkbox on the message rather than a
mid-stream dialog: an HTTP response cannot ask a question back, and a refused write
tells the user to tick it and ask again. This is the replacement for the AnythingLLM
screen people use today and must cover:

- Workspace list and switcher; workspace settings (classification label, allowed providers,
  members, API tokens); a separate context page with the editor and its version history.
- Chat: thread list, streaming answer with a collapsible steps block, citations as links
  to the document's row, charts and graph results inline, the allow-writes checkbox, a
  mode selector for new sessions, Stop, an empty state that lists what the workspace holds.
- Documents: upload (multi-file), paste text, status with progress, pin, delete.
- Tables: list with schema and sample rows, and the import form; a SQL page with result grid
  and download.
- Graph: search box, ECharts graph with class colors, node inspector with properties and
  provenance, merge review queue, provisional and stale banners.
- Ontology: class, relation, property, and mapping editors with validation inline;
  "Propose" (with cost shown) and "Propose extensions"; the candidate review queue with
  evidence, paged, with bulk accept and reject and a low-support filter; version history
  with diff and restore; import and export as JSON.
- Admin: users and the audit log viewer (the skeletal log only; the workspace-side detail is
  API-only). Tokens are managed in workspace settings, not here.

### 11.2 REST API

JSON, versioned under `/api/v1`, authenticated by a bearer token (a login session or an API
token) or the `quack_session` cookie. Every response for a question is the same object print
mode emits:

```json
{
  "answer": "...",
  "citations": [{"n": 1, "document_id": "...", "filename": "Policy-2024.pdf", "page": 12, "heading": "Exclusions", "chunk_id": "...", "chunk_index": 3, "label": "Policy-2024.pdf p.12"}],
  "queries": [{"sql": "...", "rows": 4, "duration_ms": 9}],
  "steps": [{"tool": "run_sql", "summary": "4 rows", "duration_ms": 9, "detail": "..."}],
  "graph": {"nodes": [...], "edges": [...]},
  "chart": {...},
  "write_refused": false,
  "cancelled": false,
  "usage": {"input_tokens": 1204, "output_tokens": 57, "total_tokens": 1261},
  "session_id": "..."
}
```

```
GET    /healthz                                   liveness, no auth
POST   /api/v1/auth/login                         {username,password} -> token (web session)
ANY    /mcp/v1/{id}                               MCP over streamable HTTP, same bearer (section 11.3)
GET    /api/v1/workspaces
POST   /api/v1/workspaces
GET    /api/v1/workspaces/{id}
PATCH  /api/v1/workspaces/{id}                    settings
POST   /api/v1/auth/login  POST /api/v1/auth/logout  GET /api/v1/auth/me
POST   /api/v1/workspaces/{id}/query              {prompt, session_id?, mode?, allow_write?}
POST   /api/v1/workspaces/{id}/query/stream       same, SSE agent events; closing the stream cancels the turn
POST   /api/v1/workspaces/{id}/sql                {sql}
GET    /api/v1/workspaces/{id}/search?query=&top_k=   hybrid retrieval, no LLM (the MCP `search` tool's names)
GET    /api/v1/workspaces/{id}/documents
POST   /api/v1/workspaces/{id}/documents          multipart or {text,title} -> 202 {id}
                                                  (identical bytes: status "duplicate";
                                                  a table another document owns: 409)
GET    /api/v1/workspaces/{id}/documents/{doc}    status, metadata
PATCH  /api/v1/workspaces/{id}/documents/{doc}    {pinned}
DELETE /api/v1/workspaces/{id}/documents/{doc}
GET    /api/v1/workspaces/{id}/tables[/{name}]
GET    /api/v1/workspaces/{id}/graph/search?entity=&class=&relation=&hops=
GET    /api/v1/workspaces/{id}/graph/path?from=&to=
GET    /api/v1/workspaces/{id}/graph/status
POST   /api/v1/workspaces/{id}/graph/extract       tables now; documents -> 202 with the cost, one run per workspace (409 while one runs)
POST   /api/v1/workspaces/{id}/graph/revalidate
POST   /api/v1/workspaces/{id}/graph/review        mark a provisional graph reviewed
GET    /api/v1/workspaces/{id}/graph/merges        PUT .../graph/merges/{mid} {action: accept|reject}
POST   /api/v1/workspaces/{id}/import              {url, table, query?, source_table?, limit?}
GET    /api/v1/workspaces/{id}/okf                 the bundle as a tar (import is POST .../documents with a tar)
GET    /api/v1/workspaces/{id}/ontology            current version, JSON
PUT    /api/v1/workspaces/{id}/ontology            import: validate, write a new version
POST   /api/v1/workspaces/{id}/ontology/init       the built-in default as version 1
GET    /api/v1/workspaces/{id}/ontology/versions[?limit=20] | /{v}[?against=N] for a diff
POST   /api/v1/workspaces/{id}/ontology/versions/{v}/restore
POST   /api/v1/workspaces/{id}/ontology/propose    {mode: full|extend, sample?, auto_accept?, documents?} -> 202 with documents
GET    /api/v1/workspaces/{id}/ontology/candidates[?status=low_support]
POST   /api/v1/workspaces/{id}/ontology/candidates {accept: [ids], reject: [ids]}
PUT    /api/v1/workspaces/{id}/ontology/candidates/{cid}   {action: accept|rename|merge_into|reparent|reject, ...}
GET    /api/v1/workspaces/{id}/context             current; Markdown or JSON by Accept
PUT    /api/v1/workspaces/{id}/context
GET    /api/v1/workspaces/{id}/context/versions
GET    /api/v1/workspaces/{id}/sessions[?limit=50] | /{sid}
DELETE /api/v1/workspaces/{id}/sessions/{sid}     creator or owner
PATCH  /api/v1/workspaces/{id}/sessions/{sid}     {shared} | {mode} (creator or owner; audited as share, mode)
                                                  a session's mode is set when it is created;
                                                  `mode` on a later query is ignored
GET    /api/v1/workspaces/{id}/sessions/{sid}/export?format=sql|markdown
GET    /api/v1/workspaces/{id}/audit              detail rows, members only
GET    /api/v1/workspaces/{id}/members  POST/DELETE ...   (owner)
GET    /api/v1/admin/users  POST ...  GET /api/v1/admin/audit   (admin; skeletal log)
```

Uploads, extraction, and proposals return `202` with a `job` id and run on the work queue
(section 4.1); clients poll the resource or the job. Agent turns run there too, in their
session's lane. Rate limiting per token via `tower_governor`.

```
GET    /api/v1/workspaces/{id}/jobs               queued, running, and recent jobs, newest first,
                                                  with counts and the worker total (viewer)
GET    /api/v1/workspaces/{id}/jobs/stream        SSE: `jobs` (the list) then `job` per change
GET    /api/v1/workspaces/{id}/jobs/{job}         one job
POST   /api/v1/workspaces/{id}/jobs/{job}/cancel  its submitter, or a workspace owner or admin
```

A question's text shows in a job only to whoever may read its session (its owner, or a
workspace owner or admin); other members see "a question in a private session".

### 11.3 MCP server

Same tool set over two transports (`rmcp`, the official Rust SDK): `quack mcp [-w NAME]
[--allow-write]` on stdio for Claude Code and editors (one line in `.mcp.json`, no server
needed, unaudited like the CLI), and `/mcp/v1/{workspace}` under `quack serve` over MCP's
streamable HTTP transport (the successor of the HTTP+SSE pair; responses stream as SSE),
authenticated with the same bearer as the REST API. Over HTTP every request passes
`access()`, each caller gets a transport keyed by workspace, user, and write permission
(the member role with the write scope), and every tool call is audited with channel
`mcp`. Each `query` call starts a new session (owned by the server user, `mode` `chat` or
`query`) and returns its `session_id`; passing that id back continues the session, and a
turn that fails before recording anything leaves no session behind.

Tools: `query`, `search`, `sql`, `list_tables`, `describe_table`, `list_documents`, and —
once the graph has nodes — `search_graph` and `find_path`. Every tool answers with
structured content plus text; refusals (a write without permission, an internal table, a
missing table) are tool errors the client model can read. Resources:
`quack://workspace/tables`, `.../tables/{name}/schema`, `.../documents`, `.../ontology`
(JSON), `.../context` (Markdown).

```json
{ "mcpServers": { "quack": { "command": "quack", "args": ["mcp", "-w", "logistics"] } } }
```

### 11.4 Terminal session (TUI)

A Claude Code-style single-pane transcript: streaming answers, inline steps, citations
rendered as footnotes, charts drawn with ratatui, permission prompts answered with `y`/`n`/
`a`, direct SQL when input starts with `SELECT`/`WITH`/`FROM`/`DESCRIBE`/`SHOW`/`PIVOT`/
`SUMMARIZE`; direct SQL and `/sql` pass the same gate as the agent's statements (internal
tables refused, writes ask `y`/`n`/`a`, `max_query_rows` rows shown). Slash commands:
`/help`, `/tables`, `/schema TABLE`, `/sql`, `/ingest PATH` (`/attach`), `/import`, `/docs`,
`/pin`, `/unpin`, `/delete`, `/ontology ...` and `/graph ...` (the `quack ontology` and
`quack graph` verbs, parsed by the same clap definitions, run in the background with their
output in the transcript; anything that would ask on stdin is answered yes), `/graph ENTITY`,
`/path`, `/context [import FILE | export FILE]`, `/okf DIR`, `/sessions`, `/resume`, `/new`,
`/mode`, `/share`, `/unshare`, `/export [--sql|--markdown] [FILE]`, `/jobs`, `/cancel N`,
`/chart [N]`, `/steps`, `/model`, `/workspace`, `/clear`, `/quit`.

Nothing blocks the input (section 4.1). A question, a statement, a file, an import, and an
ontology or graph verb are each submitted as a job, and the prompt takes the next line at
once: ask a follow-up while an answer streams (it is queued behind it in the session's
lane and says so), run SQL or load a file alongside. A strip above the input shows the
running and queued jobs with a spinner, number, kind, label, and progress; the status line
counts them; `/jobs` lists recent ones with their outcomes and `/cancel N` stops one.
Results land in the transcript as each job finishes. A turn's text renders only while its
session is on screen; switching sessions leaves it running and a line reports its end.
Write prompts from concurrent work queue and are answered one at a time. The session is
one async loop: a `tokio::select!` over crossterm's `EventStream`, a single channel every
job and turn reports on, the job queue's broadcast, and a spinner tick that runs only
while a job is active; every waiting message is applied before the next draw. Answers render Markdown (headings, bullets,
fences, inline marks), tool steps show a three-line preview of their detail until `/steps`
expands them (print mode folds the same way without `--verbose`), and each chart belongs
to its answer: the pane shows the latest, `/chart N` any earlier one. Lines wrap to the
terminal width before the scroll range is computed, so the end is always reachable. Typed
input is kept in `<data_dir>/terminal_history` across sessions; a relative path to an
existing file ingests it; an embedding provider is optional (keyword search without one).
Keys: `Enter` send,
`Shift+Enter` newline, `Up`/`Down` history, `PageUp`/`PageDown` and the mouse wheel scroll,
`Home`/`End` jump, `Esc` or `Ctrl+C` cancel
this session's newest turn, running or queued (recorded with whatever streamed and a
cancelled note), `Ctrl+C` with no turn quits (twice when other jobs are still running,
which stop with the session), `Ctrl+L` clear. The web chat has a Stop button and print mode cancels on
`Ctrl+C`; every interface passes a cancellation token to `run_turn`.

Works on a named workspace (`-w`), resolved through the control plane like every other
interface.

### 11.5 Print mode and CLI

```
quack -p "PROMPT" [-w NAME] [-f text|json] [--mode chat|query]
      [--allow-write] [-c | -r SESSION] [--stdin] [--verbose]
quack -q "SQL" [-w NAME] [-f table|json|ndjson|csv|markdown] [--stdin]
quack ingest FILE|DIR|- [-w NAME] [--filename N] [--title T] [--pin] [--no-embed]
quack docs [--json] [--pin ID | --unpin ID | --delete ID]
quack graph search ENTITY [--hops N] [--relation R] [--class C] | search --class C
            | path FROM TO [--max-hops N] | status | extract [--tables-only|--documents-only]
            [--sample N] [--reset] [-y] | revalidate | review | merges | merge ID.. | reject ID..
quack ontology show | init | propose [--extend] [--documents] [--from FILE] [--sample N]
              [--auto-accept] [-y] | review [--low-support]
              | accept ID... [--rename N|--merge-into ID|--reparent C] | reject ID...
              | export FILE | import FILE | versions | diff [FROM] [TO] | restore V
quack context show | edit | history | export FILE | import FILE
quack sessions [--json] [--limit N] | export SESSION [--sql|--markdown]
quack import URL --table T (--from SOURCE_TABLE | --query SQL) [--limit N]
quack okf export DIR|-
quack auth login PROVIDER [--device-code] | status [PROVIDER] | logout PROVIDER
quack config [--changed] [--json]
quack doctor [-w NAME] [--offline] [--json]
quack serve [--bind ADDR] [--local]
quack mcp [-w NAME] [--allow-write]
quack user add [--admin] | list [--json] ; quack token create|list|revoke ;
quack member add|remove|list ; quack audit [filters] [--json|--csv]      (server admin)
quack --version    version plus the AWS-LC module the binary links; -V is the bare version
```

stdin that is not a TTY is data for `-p` and `-q`: CSV, JSON, or Parquet loaded as the
temporary table `stdin` for that invocation (a pasted document is `quack ingest -`). A
pipe that delivers nothing within a second (a supervisor's inherited stdin) is skipped
with a warning rather than waited on; `--stdin` waits for it. stdout carries the answer
or result set; stderr carries steps. In `-f json` and `-f ndjson`, columns that share a
name keep every value under suffixed keys (`a`, `a_1`). Print mode streams text only on a
terminal and shows the validated answer when validation changed what streamed; a pipeline
gets the validated answer alone. Exit codes: 0 ok, 1 runtime error, 2 usage, 3 write
refused, 4 auth required; a reader that closes stdout early (`| head`) ends the command
quietly with 0.

`quack config` and `quack doctor` are the commands that do not go through `Config::load`: `config` reads the
file itself, so it describes a configuration every other command refuses rather than
failing the same way. It prints every setting this binary recognizes with the value in
force, where that value came from (built in, the file, or the environment variable that
overrides it), what the file says where that is not what is running, the keys in the file
no section recognizes with the recognized key each resembles, and which of the
environment variables the configuration reads are set — never their contents, since some
of them hold credentials. `--changed` keeps only the settings the file or the environment
has a say in; `--json` emits the whole report as one document. A rejected file exits 2
after printing the report.

`quack doctor` troubleshoots the whole setup, one line per check with the fix under
anything that needs one: the config file (rejected, unknown keys with the key each
resembles), the crypto module (a Linux build without FIPS warns), the data directory
(writable, and a warning when group or others can read it), `control.db` (opens and
migrates), the workspace (opens, embedding dimension agrees), each configured model
(credential present, plain HTTP off this machine with a credential warns, and one `GET`
of the provider's model list proves it is reachable, the key is accepted, and the model is
pulled or listed), and `[server]` (a non-loopback bind warns, `local` off loopback fails,
no users yet is noted). With no chat model it looks for a local Ollama and suggests a
`config.toml` snippet with the models that Ollama has. It creates nothing: a data
directory, control database, or workspace that does not exist yet is reported as such.
`--offline` skips the network; `--json` emits `{ok, failures, warnings, checks}`. Any
failed check exits 1.

No model is required to run quack. Without `[general].chat_model` the terminal session
opens, runs typed SQL and every slash command, and answers a question with how to set a
model up; `-q`, ingest, import, and the server's SQL and table pages work as before. A data
directory quack creates is `0700` on Unix, since it holds every workspace's content and
the OAuth token caches.

### 11.6 Desktop window (`quack desktop`)

Not built (#35, the one open gap): there is no `desktop` subcommand and no Tauri dependency
today. The design, for when it is:

`quack desktop` starts the embedded server on a random loopback port with a per-launch
bearer token and opens the web UI in a Tauri webview with that token. Nothing is
duplicated: the desktop window is the web UI plus native file dialogs, drag-drop of files
and folders, a system tray, and OS keychain access for OAuth token keys. Data lives in the
platform app-data directory as named workspaces. The local Ollama provider is
auto-detected.

It is a subcommand of the same `quack` binary, so the Tauri runtime is linked into every
build. That cost is accepted in exchange for one artifact; if the size or the platform
webview dependencies ever become a problem for the container image, the fallback is a
separate `quack-desktop` crate, not a Cargo feature. Installer bundles (`.dmg`, `.msi`,
`.AppImage`) wrap the same binary with `tauri build`. This is the last interface on the
roadmap and may never be built.

---

## 12. Server Auth, Roles, and Audit

- `quack serve --local`: no auth, loopback only, single implicit user. For a laptop that
  wants the browser.
- Otherwise, users in `control.db` with argon2id password hashes and a login form that sets
  a session cookie; API tokens (`quack token create` or the admin UI) as bearer tokens
  scoped to a workspace with `read` / `write` / `admin` scopes. OIDC login for users (the
  PKCE machinery from 10.2 pointed at the org IdP, `sub` as `oidc_subject`) is the intended
  production path and is scheduled after the token path ships.
- **A browser session is bounded at both ends** (issue #73). It dies
  `[server].session_max_age_hours` after login however much it is used, and
  `[server].session_idle_minutes` after its last request, whichever comes first; the
  expired entry is dropped on the request that finds it, which is answered 401 `session
  expired` and audited as a denied `session` action. Expiry is measured with a monotonic
  clock, so moving the system clock cannot extend a session. The cookie is `HttpOnly`,
  `SameSite=Lax`, carries a `Max-Age` matching the absolute lifetime, and carries `Secure`
  whenever the request did not arrive on loopback — so a cookie minted behind a
  TLS-terminating proxy is never sent back over a plaintext downgrade, while plain HTTP on
  a laptop keeps working.
- **Rate limiting covers everything a caller can reach**, not just the API: one
  `tower_governor` limiter over the web UI, the REST API, and MCP, keyed by bearer token
  when there is one and peer address otherwise. The two endpoints that check a password
  (`POST /login` and `POST /api/v1/auth/login`) carry a second, tighter limiter, because
  the general budget is sized for a browsing session and is far too loose to make guessing
  expensive. `/healthz` sits outside every limiter, since a throttled health check reads as
  a dead server to whatever is watching it. Each limiter's per-key state is swept once a
  minute: governor holds one entry per caller until something drops it, so an unswept
  limiter grows by one entry for every address that ever connected.
- Roles: `viewer` asks questions and searches; `member` also uploads, pins, deletes own
  uploads, grants write, edits the context and ontology, runs proposals and extraction;
  `owner` manages members and tokens and sees all sessions. `is_admin` manages users and
  all workspaces but does not thereby become a member of any; reading a workspace's
  content requires membership.
- **Audit is split at the boundary, and the access half is mandatory.** Every request
  that touches a workspace, allowed or denied, writes a row to `control.db.audit_log`
  (section 5.5): who, which workspace, which resource by opaque id, what action, the
  outcome, the channel, the client address, and when. The same UUID v7 id keys a
  `_quack_audit` row inside the workspace holding the content detail (the SQL, the file
  names, the table name, the context diff, the proposal accepted); both rows are written for
  every allowed action, listings and page views included (`list`, `page`, `open`), and a
  failed audit write fails the request. A denial writes the `control.db` row only — there is
  no workspace to write detail into when access was refused. Table names are content and never appear in
  `control.db`. Over MCP the auditor is re-pointed at each request's identity, so a
  shared transport audits the caller, not whoever opened it. An admin sees who accessed what and
  when across every workspace; a member sees what was done inside theirs. Export and
  import of context or ontology, and session export, are audited because they move content
  across the boundary. Logins and failed logins are audited with no workspace; membership
  changes carry the workspace they changed, and a denial for an expired token carries the
  workspace that token was scoped to. Successful token use is not its own row — the action
  the token performed is the row.

---

## 13. Configuration

`~/.config/quack/config.toml` (override with `QUACK_CONFIG_DIR`). Unknown keys are an
error (`deny_unknown_fields`). Config holds nothing workspace-specific; per-workspace
settings live in `_quack_meta`.

```toml
[general]
data_dir = "~/.local/share/quack"       # QUACK_DATA_DIR
chat_model = "ollama/llama3.1:8b"       # QUACK_MODEL
embedding_model = "ollama/nomic-embed-text"
default_workspace = "default"

[providers.ollama]
type = "ollama"
auth = "none"
base_url = "http://localhost:11434"
embedding_dimension = 768
# max_concurrent_requests = 1          # model requests in flight at once; default 1 for Ollama, 8 otherwise

[providers.anthropic]
type = "anthropic"
auth = "api-key"
api_key_env = "ANTHROPIC_API_KEY"

[providers.azure]
type = "openai"
auth = "oauth"
base_url = "https://{resource}.openai.azure.com/openai/deployments/{deployment}"
embedding_dimension = 1536
[providers.azure.oauth]
issuer_url = "https://login.microsoftonline.com/{tenant_id}/v2.0"
client_id = "..."
scopes = ["https://cognitiveservices.azure.com/.default", "offline_access"]
redirect_uri = "http://127.0.0.1:19876/callback"
# client_secret_env = "AZURE_CLIENT_SECRET"   # server as confidential client
# device_code = false

[retrieval]
top_k = 8
rrf_k = 60
rerank = "none"          # or "model": the chat model orders rerank_candidates listwise
rerank_candidates = 24
pinned_token_budget = 8000   # full text of pinned documents in the prompt
always_retrieve = false      # retrieve every turn, not only when the model asks

[ingestion]
chunk_size_tokens = 512
chunk_overlap_tokens = 64
embedding_batch_size = 64
embedding_concurrency = 2     # requests in flight; Ollama needs OLLAMA_NUM_PARALLEL to use more than 1
tokenizer_encoding = "cl100k_base"
upload_max_mb = 512

[context]
max_tokens = 4000

[analysis]
max_query_rows = 250
query_timeout_seconds = 30
memory_limit_mb = 256
threads = 4
max_turns = 15
history_token_budget = 32000
max_context_tokens = 32768              # Ollama num_ctx cap; each turn asks for what its prompt needs
extraction_timeout_seconds = 120        # one chunk's extraction call (ontology evidence, graph extract)
extraction_concurrency = 1              # chunks extracted at once; Ollama serves one unless OLLAMA_NUM_PARALLEL
reader_pool_size = 4                    # reader connections per workspace handle, round-robined

[import]
max_rows = 1000000
max_download_mb = 512
timeout_seconds = 300
allow_local_files = false               # quack serve with logins: sqlite: paths on the server's disk
allow_private_hosts = false             # quack serve with logins: loopback, private, link-local hosts

[graph]
max_traversal_depth = 3
max_nodes = 200
merge_threshold = 0.08                  # cosine distance under which a merge is proposed
auto_merge_threshold = 0.02             # under which it happens without review

[ontology]
propose_sample_chunks = 200
min_support_documents = 3
key_overlap_threshold = 0.8
enum_max_values = 12                    # distinct values under which a column becomes an enum

[jobs]
history = 100                           # finished jobs kept for /jobs and the Jobs page

[server]
bind = "127.0.0.1:8080"                 # QUACK_BIND
local = false
workers_per_workspace = 1               # uploads processed at once per workspace (a lane)
session_max_age_hours = 12              # a browser session dies this long after login
session_idle_minutes = 120              # ... or this long after its last request
```

Every section sets `deny_unknown_fields`, so a key that is not in this list is a startup
error rather than a silent no-op. `quack config` (section 11.5) is how an operator sees
that list from the binary itself: every recognized setting with the value in force and
where it came from, and every key in the file that is not one of them. The TUI's tick
rate is a constant, not configuration.

---

## 14. Build and Distribution

- **One binary, `quack`, no Cargo features.** Every surface is a subcommand and every
  build contains all of them. Static musl on Linux (`x86_64`, `aarch64`), native on macOS
  (`aarch64`) and Windows (`x86_64`, `aarch64`). DuckDB and SQLite are bundled and
  statically linked.
- **Released artifacts are signed and attested.** The macOS binary is signed with a
  Developer ID certificate and submitted for notarization (a bare binary cannot be stapled,
  so the ticket stays with Apple); the Windows ones are signed through Azure Artifact
  Signing, on tag builds only, because the federated credential trusts no other ref. Every
  archive, image tarball, and pushed image carries a build provenance attestation naming
  `.github/workflows/reusable-build.yml` as its builder, which is what makes the provenance
  SLSA Build Level 3 (`docs/ci-cd.md`).
- **No DuckDB extension is ever installed or loaded at runtime.** A static musl binary
  cannot `dlopen`, and the only extension features quack enables are `bundled` and `json`
  (Parquet reading is in DuckDB's core). Anything that would need another extension (`vss`,
  `fts`, `excel`, `httpfs`, the Postgres and SQLite scanners) is implemented in Rust or not
  built.
  XLSX uses a Rust reader; keyword search is quack's own BM25 index; vector search is
  an exact scan.
- **Desktop bundles:** when `quack desktop` exists, `tauri build` wraps the same binary
  into `.dmg`, `.msi`, and `.AppImage` installers. Not a separate binary.
- **mimalloc** (`secure`) as the global allocator.
- **Crypto:** rustls + aws-lc-rs, with the `fips` feature of both on Linux, where
  aws-lc-fips-sys links statically, so every distributed Linux binary and the image run on
  the FIPS-validated module (`docs/crypto.md`); `make crypto-gates` runs
  `cargo tree -i ring` and `-i openssl-sys` before the release builds anything.
- **Container image** for `quack serve`: `Dockerfile` builds from source (CSS stage with
  the standalone Tailwind binary, checksum verified; `rust:<MSRV>-alpine` with cargo-chef
  for the musl build, `cmake`, `clang`, `g++`, `perl` for DuckDB and aws-lc, and `go`
  for the FIPS module's delocate pass, which needs `AWS_LC_FIPS_SYS_CC=clang`;
  `distroless/static` `nonroot` runtime with `/quack` and `/data`, `QUACK_DATA_DIR=/data`,
  `QUACK_CONFIG_DIR=/config`, port 8080); `Dockerfile.release` builds the same runtime
  from the prebuilt musl binaries so the release job never compiles under emulation.
  The Rust image tag must track `rust-toolchain.toml`. `docker-compose.yml` runs
  `quack serve` beside Ollama with `deploy/config.toml` mounted at `/config`; the
  per-architecture image tarballs from a release are `docker load`ed on air-gapped
  hosts.
- **Dependencies** follow the workspace rules in `CLAUDE.md`. What this design added:
  `argon2`, `rmcp` for MCP, `scraper` for HTML, `zip` + `quick-xml` for DOCX and PPTX (no
  `docx-rs`), `rust-stemmers`, `tower_governor`. No YAML crate: JSON is the only ontology
  interchange form. `tauri` is still future, and only if `quack desktop` is built.

---

## 15. Scaling Constraints and Decision Points

These are the properties of the chosen storage that the team should accept explicitly,
because the system being replaced runs on Postgres.

1. **DuckDB is embedded and single-process.** One `quack serve` process owns every
   workspace file. Vertical scaling only. Concurrency within a process is narrower than
   DuckDB's MVCC would allow: a workspace has one writer connection
   (`storage::writer::Writer`, a two-tier line: interactive callers before background
   ones), so ingestion and session writes take turns; reads go to the reader pool.
   Horizontal scaling or an HA pair is not possible without moving storage to a server
   database. For a single-instance deployment this is a simplification, not a limitation.
2. **Vector search is an exact scan, not an index.** Every query computes the cosine
   distance against every stored embedding inside DuckDB; the working set is
   `chunks × dimension × 4` bytes (about 4 GB per million 1,024-wide chunks). Measured
   with `cargo bench -p quack-core --bench retrieval` (`make bench`; synthetic chunks of
   sixty words from a 2,000-word vocabulary, 1,024-dimension vectors, top 8, an in-memory
   workspace on a 10-core Apple Silicon laptop, p50 / p99 per query):

   | chunks | vector scan | BM25 leg | fused hybrid |
   |---|---|---|---|
   | 10,000 | 10 / 11 ms | 8 / 8 ms | 18 / 19 ms |
   | 100,000 | 95 / 95 ms | 36 / 37 ms | 134 / 138 ms |
   | 1,000,000 | 165 / 168 ms | 285 / 287 ms | 450 / 456 ms |

   The scan is linear to 100,000 chunks (about 1 µs per chunk) and sublinear past it
   once DuckDB spreads it across cores. Retrieval stays well under the model's own time
   at every size a workspace is likely to reach, so no index is built. The decision,
   recorded so it is not rediscovered (#32): if a workspace passes a million chunks, a
   pure-Rust HNSW crate over the same stored vectors, persisted beside `data.duckdb` and
   rebuilt from `_quack_chunks` when missing or stale, is the accepted path; the
   alternative of a custom DuckDB build with `vss` and `fts` statically linked (DuckDB's
   extension config, `DUCKDB_LIB_DIR` and `DUCKDB_STATIC`) is a C++ build pipeline per
   release target and is not pursued. BM25 is an indexed join on `_quack_terms`, but its
   cost grows with the posting lists, not the chunk count alone: at a million chunks
   (sixty million term rows) it is the larger half of a hybrid query, so a workspace that
   size wants the term index looked at before the vector scan. Because every request on
   a workspace waits behind the same connection, these are also the latencies other
   requests queue behind while a search runs.
3. **One file is the boundary, so one file is the backup unit.** Back up a workspace by
   copying its directory while the server holds no write transaction. A
   `quack workspace snapshot NAME` that does this through DuckDB's `CHECKPOINT` is still
   unwritten (section 19, step 13). There is no cross-workspace transaction and none is
   needed.
4. **Storage backend seam — not built.** The intent was that retrieval, `graph/`, and
   `ontology/` sit behind small traits so a Postgres + pgvector backend could be added
   without touching the agent or the interfaces. In the code they take `&WorkspaceDb`
   directly; the only traits are `DbHandle`, `Reranker`, `Extractor`, and `GraphExtractor`,
   none of them a storage seam. Adding another backend today means changing graph and
   ontology code.
5. **Migration from the current deployment.** Documents are re-uploaded and re-embedded
   rather than migrated from pgvector, because chunking and metadata differ. A
   `quack import anythingllm --url ... --key ...` command that pulls workspaces, documents,
   system prompts, and threads through the AnythingLLM API is the planned path; it is
   listed in section 19 after the core is stable.

---

## 16. Testing

**Unit.** `storage/` migrations and CRUD for both databases, dimension mismatch, `_quack_`
tables hidden from listing and refused to user and agent SQL, statement classification
(SELECT variants and the `DESCRIBE`/`SHOW`/`SUMMARIZE`/`PIVOT`/`UNPIVOT`/`EXPLAIN`
allow-list read; DDL, DML, COPY, SET, ATTACH write; a parse error invalid), limits, row
capping, RRF fusion on fixture rankings; `ingestion/` chunking with heading and page
metadata, SHA dedup; `analysis/` citation validation strips unknown markers and renumbers
the rest, chart spec bounds; `ontology/` inheritance, domain/range validation, property
types, mapping validation, versioning and stale detection, JSON round trip, table-evidence
induction on fixture tables (key detection, overlap relation), document-evidence
normalization on fixture extraction output (clustering, domain/range inference, hierarchy
inference, support thresholds), candidate actions; `graph/` extraction parsing (valid,
malformed, out-of-ontology dropped and counted for drift), merge on normalized label,
embedding merge proposals, provisional flagging, traversal on a fixture with a cycle, path
search; the agent loop against a mocked rig model with canned tool calls including a
refused write and a timeout, mode enforcement excluding provisional graph results,
cancellation keeping streamed text; `llm/` `TokenManager` reuse, single refresh under
concurrency, re-auth, cache round-trip against a mock IdP; `crypto` asserting the provider
is FIPS exactly on Linux.

`cargo test --workspace` currently runs 308 tests: 209 in `quack-core`'s library, 47 in the
binary (the server router and terminal harness among them), and 52 across two integration
files. `proptest` is on the dependency menu but no test uses it yet.

**Integration.** Upload PDF -> ready -> question in query mode returns an answer with a
citation on the right page; keyword-only question (a policy number) is answered via FTS;
CSV -> question produces SQL referencing a context definition; `ontology propose` on a
fixture workspace of two tables and ten documents yields the expected classes, one overlap
relation, and a mapping, and accepting them builds nodes with provenance; `--auto-accept`
marks the graph provisional and query mode ignores it; an ontology edit marks the graph
stale and revalidate drops the invalid edge; write refusal per interface (exit 3, 403, MCP
error) and success with permission; session round trip and export with audit rows in both
databases sharing an id; a denied open by a non-member and an expired token each write an
`audit_log` row with `outcome = denied`; no code path updates or deletes `audit_log` rows;
workspace isolation via CLI and API, and an admin without membership cannot read workspace
content; server token lifecycle, roles, upload queue; deleting a document takes its chunks,
its graph rows and its files with it; a cancelled turn is recorded with what streamed; an
OKF bundle exported through the API imports back as documents and candidates; MCP stdio
client lists tools and runs `query`; REST and print mode return byte-identical JSON for the
same question with a mocked model. The REST, role, audit and queue suites live in
`crates/quack/src/server/tests.rs`; the two files under `crates/quack-core/tests/` cover
ingestion and the graph.

**Manual.** Web UI end to end against Ollama including the proposal review flow; TUI
streaming and permission prompt; OAuth browser and device-code flows against a real
tenant; the air-gapped static binary, which loads no extensions at all.

**CI and local gates.** `.github/workflows/ci.yml` runs `cargo fmt --check`, clippy with
`-D warnings` over all targets and features, and `cargo test --locked --workspace` on Linux
and macOS, plus dependency review and cargo-deny, on every push and pull request. Coverage
(`make test-coverage`, `cargo llvm-cov`) and mutation testing (`make test-mutants`, the whole
workspace) are local-only and not wired into a release. There is no fuzzing.

**Evaluation.** `make eval` (`crates/quack-core/examples/eval.rs`, issue #74) is the
answer-quality counterpart to the correctness suites above: it ingests a small in-tree
storms-like fixture (`crates/quack-core/eval/`, 27 documents and three CSV tables written
for the harness, not the NOAA download) into a temporary workspace and prints recall@1/5/8
and MRR, per question kind (`identifier`, `phrase`, `semantic`) and per backend
(`search_keyword_chunks`, `search_similar_chunks`, `search_hybrid_chunks`), over a 21-question
gold set; precision and recall of `ontology::induction::propose_from_tables` against a
hand-written expected ontology; node and edge precision and recall of `graph::extract::run`
over ten hand-labelled chunks through a canned `GraphExtractor`, so the numbers measure
validation, resolution, and storage rather than a model; and how many of a fixed set of
recorded answers keep every `[n]` marker through `analysis::citations::validate`. Vector
search uses a deterministic hashing embedder (a bag-of-words projection, not a semantic one)
so the run needs no Ollama and finishes in seconds. It writes the same numbers as JSON to
`QUACK_EVAL_OUT` when set, for a before/after diff.

Five identifier questions (`SR-8841`, `SR-4437`, `SR-7765`, `SR-2214`, the `AKQ` office code)
and two phrase questions (`"flash flood emergency"`, `"wall of water"`) carry decoy documents
whose split identifier pieces or non-adjacent phrase words outscore the true chunk under
plain BM25, so the fixture actually discriminates issue #77's identifier-joining and
phrase-filter fix instead of trivially scoring 1.000 either way; one phrase question
(`"catastrophic flood damage"`) expects an empty result, since every word in it appears
somewhere but the exact phrase appears nowhere. Baseline on this fixture (before #77):
keyword recall@1 0.571 (MRR 0.762), hybrid recall@1 0.452 (MRR 0.605); by kind, keyword
recall@1/MRR is 0.444/0.722 for identifier and 0.000/0.375 for phrase, both dragged down by
the decoys; ontology induction and graph extraction both precision 1.000 / recall 1.000
against their fixtures; citation validity 8/8 recorded answers validated as expected. With
`fix/77-identifier-and-phrase-search` (#104) merged on top, the same fixture measures
keyword identifier recall@1/MRR at 0.889/0.944 and keyword phrase recall@1/MRR at 0.875/
1.000 — the identifier-joined term and the phrase substring filter recover every decoyed
question except the bare `AKQ` code, which has no hyphen for the joined-identifier term to
attach to. Those are the expected numbers once #104 merges; a prompt, chunker, stemmer, or
fusion-constant change is no longer a coin flip either way: this is the number that moves.

---

## 17. Gaps Between This Document and the Code

Every gap is a GitHub issue except where this list says otherwise; it is the map from the
design to the tracker and is updated as issues close. Ordered by risk.

1. ~~Verify against a live model~~ (#20, closed): print mode and the terminal session are
   verified with gpt-oss:20b on Ollama. Sections 7, 8, 9.
2. ~~No OAuth~~ (#25, closed): PKCE and device-code login, encrypted cache, `quack auth`;
   the server's confidential-client mode is wired (`client_secret_env`) and gets its live
   test with #26. Section 10.2.
3. ~~No server, REST API, web UI~~ (#26, closed): `quack serve` with the REST API,
   password and token auth, roles, the split audit, the upload queue, and the askama +
   htmx web UI (workspaces, chat with steps, citations, and charts, documents, tables,
   SQL, context editor, settings with members and tokens, admin users and audit). The
   graph and ontology pages arrive with #27 and #28. ~~No MCP~~ (#29, closed: `quack mcp`
   on stdio and `/mcp/v1/{workspace}` over streamable HTTP, section 11.3); **no desktop
   window** (#35). Sections 11, 12.
4. ~~No ontology or induction~~ (#27, closed): the model, validation, versions with
   diff and restore, the built-in default, JSON import and export, table and document
   evidence into the review queue (accept, rename, merge, reparent, reject, auto-accept,
   extend mode, `--from` seeding, low-support candidates), and the CLI, API, and web
   page. Two deviations from 6.5: cluster names are chosen by frequency rather than a
   model naming pass, and drift counting arrives with constrained extraction in #28.
   YAML was dropped: JSON is the only interchange form. ~~No graph~~ (#28, closed):
   `quack_core::graph` with deterministic extraction from mapped tables, constrained
   extraction from chunks through the chat model (out-of-ontology classes and relations
   counted as drift), exact merge on normalized label and class plus embedding-based
   merge proposals, provenance on every node and edge, neighborhood, path, and by-class
   traversal in plain SQL, the `search_graph` and `find_path` tools (registered only when
   the graph has nodes; query mode drops provisional results), `quack graph`, `/graph`
   and `/path` in the terminal, the REST endpoints, the MCP tools, and the web page with
   an ECharts force graph, inspector, merge queue, and provisional and stale banners.
   Three deviations from 6.4: merge proposals live in `_quack_graph_merges` (accepting
   one merges nodes, which is not an ontology change, so they stay out of the ontology
   candidate queue); "provisional" means the newest ontology version was written by
   `--auto-accept` and nobody has saved a reviewed version since; and "graph enabled"
   is simply the graph having nodes rather than a separate `_quack_meta` flag. Sections
   6.3 to 6.5.
5. ~~DOCX, HTML, PPTX, XLSX unsupported~~ (#16, closed): HTML through `scraper`, DOCX and
   PPTX through `zip` + `quick-xml`, workbooks through `calamine` as one table per sheet,
   each format carrying its own title. Sections 6.1, 6.2.
6. ~~No `ATTACH` to external databases~~ (#21, closed): `quack import`, the Rust-side
   replacement, snapshots a Postgres or SQLite query or a data file over HTTP(S) into a
   workspace table through the CSV path (`quack_core::import`). A live `ATTACH` (queries
   pushed to the source) is not offered: the scanner extensions cannot ship in the
   static binary, and a snapshot keeps the classification boundary simple, since the
   rows then live in the workspace file like any upload. MySQL and S3 are the next
   sources. Section 6.2, step 13.
7. ~~Document registry lacks `sha256` dedup, `source`, `title`~~ (#22, closed): identical
   bytes are skipped everywhere and name the existing document; `source` is `upload`,
   `paste`, `path`, or `stdin`; the title is given or parsed from the first heading;
   `chunk_count` and `ingested_by` are recorded. Section 5.4.
8. ~~Sessions have no `created_by` or sharing; print mode cannot take stdin as data~~
   (#23, closed): `created_by` since the server landed; `shared` is set by the creator or
   an owner (`PATCH .../sessions/{sid}`, the chat page's toggle), audited as `share`, and
   opens the session to every member; piped stdin is the temporary table `stdin` in `-p`
   and `-q`. Sections 8, 11.5.
9. ~~Context `edited_by` and the `_quack_audit` detail table~~ (#24, closed): the
   server records the editing user and writes the detail row under the access row's id.
   Sections 5.3, 5.4, 12.
10. ~~No release pipeline~~ (#30, closed): `.github/workflows/release.yml` runs only on a
    `v*` tag (or by hand): the gates (fmt, clippy, tests, `make crypto-gates` for the
    ring and OpenSSL runtime-tree checks, cargo deny), then
    `.github/workflows/reusable-build.yml` for everything that compiles, signs, or
    attests — reproducible static musl binaries for x86_64 and aarch64 with CycloneDX
    SBOMs through `Dockerfile.build` and `docker-bake.hcl`, native macOS arm64 and
    Windows x86_64 and aarch64 binaries, each on its own native runner and code-signed
    (Apple Developer ID with notarization, Azure Trusted Signing) when the secrets exist,
    and the `quack serve` image for amd64 and arm64 on GHCR built per architecture from
    the prebuilt binaries (`Dockerfile.release`) plus per-architecture image tarballs for
    `docker load` on air-gapped hosts. Every artifact carries a build provenance
    attestation signed under the build workflow's identity, which is SLSA Build Level 3;
    no job that builds holds `contents: write`, and a separate `publish` job writes
    `SHA256SUMS` and creates the GitHub release. `Dockerfile` builds the same image from
    source for `make image` and `docker-compose.yml`, which runs it beside Ollama with
    `deploy/config.toml`. Section 14.
11. ~~No stemming in keyword search~~ (#31, closed: Snowball English over the same
    tokenizer, schema version 6 rebuilds older term indexes on open); ~~no reranking
    hook~~ (#34, closed: `Reranker` trait, `none` or `model`); ~~large-workspace vector
    index options~~ (#32, closed as a recorded decision in section 15, item 2). Sections
    6.1, 15.
12. ~~Web UI mapping of the chart spec to ECharts~~ (#26, closed): `static/js/app.js`
    maps the spec to an ECharts option. Section 9.
13. ~~Open Knowledge Format bundles~~ (#36, closed as a deliberately one-way export that
    restores what it can): `quack okf export DIR`
    (or `-` for a tar on stdout) and `GET /api/v1/workspaces/{id}/okf` (a tar, audited as
    `export`) write `index.md` from the context, one Markdown file with YAML front matter
    per table (schema, sample rows, mapping links), class, relation, property, document,
    and graph node (properties, links per edge, provenance per chunk and row), and
    `log.md` from the ontology versions and the audit detail. `quack ingest DIR` on a
    bundle and `POST .../documents` with an `application/x-tar` body ingest every concept
    file as a Markdown document, turn front-matter types into class candidates, links
    between typed concepts into relation candidates (the relation named on a
    `- <relation>: [..](..)` line, else `<source>_links_<target>`; nothing when the
    ontology already relates the two classes or their ancestors), and
    `resource` into a document property candidate, all in the ontology review queue; the
    CLI offers `index.md` as the workspace context and the API returns it as `context`.
    `quack_core::okf`.
14. **No `quack workspace snapshot`** — the only gap here with no issue of its own. Section
    15 item 3 names it as the supported way to back a workspace up while the server runs;
    there is no `workspace` subcommand and nothing calls `CHECKPOINT`. Backing up today
    means copying the workspace directory while no write is in flight. Sections 15, 19.
15. **Work queues, first pass** (section 4.1). The terminal, the web chat, REST `query`,
    uploads, graph extraction, and the document pass run on `quack_core::jobs`, and model
    requests are limited per provider and model in `llm::LimitedHttp`, interactive first.
    Ingest and import stop on cancel, mid-embedding included. A workspace with 64 uploads
    waiting answers the next with 503 and `Retry-After: 30`; the web Jobs page follows
    `.../jobs/stream` instead of polling; the terminal re-renders only the messages that
    changed. The writer is a two-tier line (interactive first) and the terminal never
    touches it from its event loop: every command's database step runs, in the order
    typed, on a worker task, reads on the reader pool. Not yet: MCP `query` calls and
    print mode run their turn directly (one call, one answer, nothing to keep
    responsive); the web chat page shows its own turn but not a job strip (the Jobs page
    does); jobs are not persisted across restarts.

---

## 18. Scope

### In scope

- `quack-core` with the three substrates and one agent, as an event stream
- Workspace as the classification boundary: one DuckDB file plus `files/`, everything
  classified inside it; `control.db` holds access control only; audit split at the boundary
- Documents: upload, paste, path, stdin; PDF, Markdown, text, HTML, DOCX, PPTX; chunk
  metadata; hybrid retrieval; citations; pinned documents; SHA dedup; chat and query modes
- Tables: CSV/TSV/Parquet/JSON/JSONL/XLSX; snapshot imports from Postgres, SQLite, and
  http(s) data files through `quack import` (no `ATTACH`; MySQL and S3 URLs are refused)
- Ontology: stored in workspace tables with inheritance, relations with domain/range, typed
  properties, table mappings, built-in default, versioning with snapshot, diff, restore,
  and stale detection; JSON import and export
- Ontology induction: deterministic proposals from tables, sampled open extraction from
  documents with vocabulary normalization and structure inference, evidence-backed
  candidates with a review queue, extend mode driven by drift, domain-pack seeding,
  auto-accept with provisional marking
- Graph: ontology-guided extraction from documents, deterministic extraction from mapped
  tables, entity resolution with review queue, provenance, neighborhood / path / by-class
- Agent: tools, permissions per interface and role, limits, visibility, citation validation
- Workspace context stored and versioned inside the boundary, Markdown import and export
- Sessions with resume, sharing, export
- Charts: one spec, rendered everywhere
- Providers: ollama, openai (and compatible), anthropic; auth none / api-key / OAuth PKCE
  with device code, encrypted cache, confidential-client mode for the server
- Interfaces, all in one binary: web UI, REST API, MCP (stdio and streamable HTTP), TUI,
  print mode, and `quack desktop` (last, if ever)
- Open Knowledge Format bundles: `quack okf export` writes one, `quack ingest DIR` and a tar
  upload read one back as documents and ontology candidates
- Server: users with password login, tokens with scopes, roles, audit, upload queue
- Static builds, container image and compose, desktop bundles

### Deferred, in rough priority order

1. AnythingLLM import command (workspaces, documents, system prompts, threads via its API)
2. OIDC login for server users
3. Data connectors: GitHub, Confluence, SharePoint (fetching a data file over http(s)
   already ships in `quack import`)
4. Cross-encoder reranking provider
5. OCR for scanned PDFs
6. Postgres + pgvector storage backend, which now also means building the seam section 15
   item 4 describes
7. Ontology import from OWL / SKOS; a registry of domain packs
8. Web search tool for the agent
9. OpenAI-compatible `/v1/chat/completions` endpoint
10. In-process embedding models (ONNX) to drop the Ollama requirement offline
11. DuckPGQ for graph queries
12. Kubernetes manifests; WebSocket MCP transport; object-storage file backend
13. Web UI localization via fluent (pattern already documented in `docs/web-ui.md`)

---

## 19. Implementation Order

Each step leaves the tool working. The first milestone that can replace the current
deployment for document chat is the end of step 8.

1. ~~Limits and classification, real `search_documents`, crypto install.~~ Done.
2. ~~Merge binaries into `quack`; `quack-core::llm`; strict config with
   `chat_model`/`embedding_model`/`auth`.~~ Done.
3. ~~Storage consolidation: `_quack_` prefix, `_quack_meta`, dimension check,
   `control.db` reduced to access control, sessions and messages with `--continue`,
   `--resume`, `quack sessions`, and `export`.~~ Done. Context lands with step 6, audit
   detail with the server.
4. ~~Agent loop as an event stream; TUI streaming, inline steps, permission prompt, direct
   SQL; print mode with formats and exit codes.~~ Done.
5. ~~Retrieval: chunk metadata, FTS index, RRF fusion, citations with validation, pinned
   documents, chat and query modes.~~ Done. DOCX, HTML, PPTX, XLSX parsers are issue #16.
6. ~~Workspace context (stored, versioned, import/export), the prompt rewrite, and the
   chart spec with its terminal renderer.~~ Done.
7. ~~OAuth: `TokenManager`, PKCE and device code, cache, `quack auth`.~~ Done.
8. ~~`quack serve`: `control.db`, users, tokens, roles, split audit, upload queue; REST
   API; SSE; web UI (workspaces, chat with citations, documents, tables, context
   editor).~~ Done. **Milestone: document chat replacement.**
9. ~~Ontology: tables, model, validation, default, versioning, JSON interchange, prompt
   rendering, editor page, API.~~ Done (#27; YAML was dropped for JSON).
10. ~~Ontology induction: table evidence, document evidence, candidates, review queue,
    extend mode and drift counting, auto-accept with provisional marking; CLI, API, and
    web hooks.~~ Done (#27), with the deviations in section 17 item 4.
11. ~~Graph: extraction from documents and mapped tables, resolution, provenance, traversal,
    tools, TUI tree, web graph page, stale and provisional handling.~~ Done.
12. ~~MCP over stdio and SSE.~~ Done (streamable HTTP rather than SSE).
13. ~~External data import (the Rust-side replacement for `ATTACH`), XLSX via a Rust
    reader~~ Done. `workspace snapshot` is still open.
14. ~~Release engineering: musl targets, macOS, Windows, container image, compose.~~ Done,
    then extended: the build split into a reusable workflow with signing and SLSA Build
    Level 3 attestations, and FIPS AWS-LC on Linux (section 14, `docs/ci-cd.md`).
15. ~~Open Knowledge Format bundles: `quack okf export`, `quack ingest DIR`, the API
    routes.~~ Done (#36), as a one-way export that restores what it can.
16. AnythingLLM import command.
17. `quack desktop` and installer bundles, last and only if there is demand.
