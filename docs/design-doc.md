# Quack: Knowledge Engine - Design Document

## 1. What This Is

`quack` is a knowledge engine with many interfaces. Its core, one Rust library, holds a
workspace's knowledge in three forms and answers across all three:

| Substrate | What goes in | How it is queried |
|-----------|--------------|-------------------|
| **Documents** | PDFs, Office files, text, Markdown, uploads, pasted text | Vectorized and chunked; hybrid semantic + keyword retrieval with citations |
| **Tables** | CSV, Parquet, JSON, Excel, attached databases | DuckDB SQL written by the agent or the user, shown before it runs |
| **Knowledge graph** | Entities and relationships extracted from documents and tables, typed by an ontology the tool can propose | Traversal, neighborhood, and path questions with provenance back to source text |

Thin interfaces call the same core functions:

- **Web UI** - chat with a workspace in a browser; the day-one replacement for the current
  AnythingLLM deployment.
- **REST API** - the same operations over HTTP.
- **MCP server** - the workspace as tools for Claude Code, Claude Desktop, Cursor, and other
  agents, over stdio or streamable HTTP.
- **TUI** - a Claude Code-style terminal session, plus a non-interactive print mode for
  pipelines.
- **Desktop window** (`quack desktop`) - the web UI in a native Tauri window with the
  server running in-process, for laptops and offline use. Last on the roadmap, if built.

One binary, `quack`, provides every interface. There are no Cargo features: every build on
every platform contains the server, the terminal, the MCP transport, and the admin commands.

It runs offline with Ollama models, or with hosted providers (OpenAI,
Anthropic, Azure OpenAI via OAuth, and anything OpenAI-compatible).

This document is the contract for the engineering agent implementing it. Section 17 lists
where the code diverges today.

---

## 2. What It Replaces

`quack` replaces an AnythingLLM deployment: chat over a pgvector RAG store where users upload
documents into workspaces. It must cover what people use today before adding more.

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

Assumptions about the current deployment, to be confirmed:

- Users log in with local accounts, not SSO.
- Documents are mostly PDF and Office exports.
- The pgvector store need not be migrated in place; re-embedding into `quack` is acceptable.
- One server instance serves all users.

---

## 3. Principles

1. **The core is the product; interfaces are adapters.** `quack-core` owns every behavior.
   A feature in one interface and not the others is a bug, not a roadmap item.
2. **Three substrates, one agent.** One agent with one set of tools queries documents,
   tables, and the graph. The user asks; the agent decides whether to retrieve, query,
   traverse, or combine.
3. **The workspace is the classification boundary.** Everything classified about a
   workspace lives in the workspace's own storage and nowhere else: documents, tables,
   chunks, graph, ontology, context, sessions, and the detail of what was done to it.
   Access is granted per workspace. Nothing outside reveals its contents.
4. **Show every action.** Every interface surfaces each retrieval, SQL statement, and
   traversal with its inputs, result size, and timing. Every answer cites its sources.
5. **Ask before writing.** The agent reads freely and mutates only with approval. The core
   makes the permission decision; each interface renders it.
6. **Structure is explicit and proposed, not invented.** The ontology is data the workspace
   owns. The tool proposes it from the documents and tables, a person accepts it, and extraction is
   constrained by and validated against it.
7. **Context is data too.** The workspace context (definitions, caveats, persona) lives
   inside the boundary and is edited through the interfaces. Files are only an import and
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
| Ask | `llm::TurnRequest::run` (event stream) | SSE fragments | SSE or JSON | `query` tool | inline / stdout+stderr |
| Retrieve | `WorkspaceDb::search_hybrid_chunks`, `analysis::rerank` | via agent, `/search` page | `GET .../search` | `search` tool | via agent |
| SQL | `WorkspaceDb::execute_query{,_capped}` | SQL page | `POST .../sql` | `sql` tool | `/sql`, `-q` |
| Ingest | `ingestion::ingest_file` | upload | `POST .../documents` | - | `/ingest`, `quack ingest` |
| Graph | `graph::traverse::{neighborhood,path}` | graph page | `GET .../graph/*` | `search_graph` | `/graph`, `quack graph` |
| Ontology | `ontology::store::{current,save,versions,restore}`, `ontology::candidates` | ontology page | `.../ontology/*` | resource | `quack ontology` |
| Context | `storage::context::{current,set,history,combined}` | context page | `.../context` | resource | `/context` |
| Permission | `analysis::policy::WritePolicy` | prompt; `write_refused` on the answer | 200, `write_refused: true`, SSE `write_refused` | `write_refused: true` plus a sentence | y/n/a prompt / exit 3 |

The LLM layer is `rig`. `quack-core::llm` builds rig clients from config and exposes
`ChatModel` and `EmbedModel` enums, so the rest of the core is provider-agnostic.

### 4.1 Work queues

Every interface is asynchronous: anything slower than a keystroke runs as a job, and the
submitting interface stays responsive and reports its status. `quack_core::jobs` is the one
mechanism.

- **Lifecycle.** A job is `queued` until its lane has room, runs, and ends `succeeded`,
  `failed`, or `cancelled`.
- **Lanes.** Jobs sharing a *lane* key run at most the lane's limit at a time, strictly in
  submission order (a job's place is taken when submitted, not when its task first runs).
  A job with no lane starts at once.
- **Priority.** `crate::priority` (`quack_core::priority`), a Tokio task-local, is
  interactive unless scoped. The queue runs `ingest`, `import`, `ontology`, `graph`, and
  `export` jobs as background. A job is a task on the runtime, so the scope covers all the
  work it awaits. Threads outside the runtime (the writer, the file-parsing blocking pool)
  ignore it: a closure takes its writer line when it is sent.

The queue does not count jobs against a pool. Each scarce resource is limited where it is
used:

| Resource | Limit | Where |
|----------|-------|-------|
| Model requests, per provider and model | `[providers.NAME].max_concurrent_requests` for each model (1 for Ollama, which serves one request per model unless `OLLAMA_NUM_PARALLEL` says more; 8 for hosted APIs), process-wide, interactive requests first | `llm::LimitedHttp`: every rig client quack builds sends through it; it reads the model from the request body and holds a permit until the body is read or the stream ends |
| The workspace's writer connection | one thread per workspace (`storage::writer::Writer`, an actor, section 7.4) runs the closures sent to it one at a time, interactive first | callers send owned closures and await the answer (`Writer::run`), so no async worker waits on it; long work sends one step at a time, never across a model call |
| Audit detail rows (server) | one insert-only connection per workspace (`storage::audit::AuditLog`, a clone of the writer's on its own thread) | `Access::audit`: `DuckDB` commits separate connections' writes together unless they change the same rows, and a detail row is always new, so a request records its audit detail (fail-closed, issue #54) without waiting for a write in progress |
| Reads | the reader pool (`[analysis].reader_pool_size`) | `ReaderDb`: the agent's SQL and search tools, the terminal's reads, MCP, and every server handler's reads (`App::read`), each in a read-only transaction |
| Uploads per workspace (server) | `[server].workers_per_workspace` | the `ingest:{workspace}` lane |

So a turn holds nothing while it waits on a write prompt or runs a tool, and a quick
`SELECT` never waits behind chat.

Model requests follow the same priority. `TurnRequest::run` and
`Embedder::embed_interactive` are interactive; ingest embeddings, extraction, and proposals
are background. A freed permit goes to the oldest interactive waiter first, so a question
never queues behind a whole ingest. rig's streaming loop drains a model response before
running its tool calls, so a tool that calls the same provider (the query embedding, the
model reranker) never waits on a permit its own turn holds.
`[analysis].extraction_concurrency` and `[ingestion].embedding_concurrency` set the width of
one run's pipeline; the provider limit caps them across runs.

| Work | Kind | Lane | Where |
|------|------|------|-------|
| An agent turn | `chat` | `session:{id}`, serial: a turn's history includes the answer before it | TUI, web chat, REST `query` |
| A typed statement | `sql` | none | TUI |
| A file or pasted text | `ingest` | `ingest:{workspace}`, `[server].workers_per_workspace` wide (server); none (TUI) | upload, `/ingest` |
| An external import | `import` | none | `/import` |
| Graph extraction | `graph` | `graph:{workspace}`, serial | graph page, REST, `/graph extract` |
| The ontology document pass | `ontology` | `ontology:{workspace}`, serial | ontology page, REST, `/ontology propose --documents` |
| Bundle and context exports | `export` | none | `/okf`, `/context export` |

A job carries a v7 id, a short number to type (`/cancel 3`), its kind, label, workspace,
submitting user, lane, progress (`done` of `total`; extraction reports per chunk), latest
status line, and outcome (a one-line summary or the error). Each change is broadcast as a
snapshot; the terminal's job strip and `/jobs`, the web console's Jobs page, and
`GET .../jobs/stream` all read it.

Cancelling a queued job ends it without running. A running job sees its cancel token and
stops at its next checkpoint, or finishes if its work has none (an ingest mid-embedding).
A cancelled agent turn is recorded as cancelled, as with `Esc`. A statement is interrupted
through `storage::workspace::QueryCanceller`, which interrupts the connection only while that
job's statement holds it. Quitting the terminal cancels every job and waits a few seconds;
work without a checkpoint runs on a detached thread and never holds the process open. Work
whose end records something (an upload's document status, an extraction's closing audit
row) records it for a job cancelled while queued too, so nothing is left `queued`.

The registry is in memory only. A job's label can name a file or quote a question, which is
workspace content (section 5), so it never reaches `control.db`. A restart forgets it; the
durable record is the document, table, session, or audit row the job wrote. Uploads a
previous process left `queued` are marked failed when the workspace next opens. Jobs share
the one writer connection (section 7.4); because long work (graph extraction, the document
pass, ingestion) sends one step at a time, a question asked meanwhile records its turn
between steps.

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

Nothing about the workspace's contents exists elsewhere. Backing up, moving, sharing, or
destroying the directory does so to the whole workspace.

Named workspaces live under `<data_dir>/workspaces/<id>/`; every interface serves them,
always by name through the control plane (`load_workspace` in the CLI). There is no
directory-local workspace: an earlier design had the TUI walk up to a `.quack/` folder, as
`git` finds `.git/`, but it was never built and no code looks for one.

### 5.2 Isolation

One DuckDB file per workspace; queries cannot cross workspaces. `ATTACH` is impossible, not
merely discouraged: `WorkspaceDb::confine_to` sets `allowed_directories` to the workspace
directory alone, turns off `enable_external_access` and `allow_persistent_secrets`, applies
the `memory_limit` and `threads` caps, then sets `lock_configuration` before any user or
agent statement runs. External data arrives through `quack import`, which snapshots rows
into an ordinary table (section 6.2). A workspace's `allowed_providers` limits which LLM
providers see its data, so a `restricted` workspace can be pinned to local Ollama. In server
mode, opening a workspace file requires membership in `control.db` (section 5.5).

### 5.3 Workspace context

The workspace context generalizes AnythingLLM's workspace system prompt: persona and tone,
plus definitions the data does not carry. It lives in `_quack_context` and is loaded into
every turn's system prompt after the schema and document blocks, capped at a token budget
(default 4,000).

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

Edit it on the web UI context page (`member`+, audited), via `PUT .../context`, or with
`quack context edit` (opens `$EDITOR` or `$VISUAL` on a temp file and stores the result).
`quack context export|import FILE` moves it as Markdown (`-` for stdout or stdin);
`quack context history` lists versions; `/context` in the terminal shows it. An
unclassified global prefix, `~/.config/quack/context.md`, loads before it. The agent never
writes the context. Each distinct edit is a new version; importing identical content
records nothing.

### 5.4 `data.duckdb`

Internal tables are prefixed `_quack_`, hidden from the agent's table listing, and refused
to user and agent SQL: `classify_user_statement` returns "internal tables are not
accessible", with no opt-in flag. IDs are UUID v7 via `uuid::Uuid::now_v7()`.

```sql
-- workspace metadata
CREATE TABLE _quack_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
  -- schema_version, embedding_dimension (the width of the vector columns),
  -- graph_built_with_ontology_version, graph_drift

-- every embedding profile a stored vector was made under (section 6.1)
CREATE TABLE _quack_embedding_profiles (
    fingerprint TEXT PRIMARY KEY,          -- SHA-256 of the profile JSON
    profile     JSON NOT NULL,             -- {model, dimension, prompts: {query, document, similarity}}
    first_used  TIMESTAMP DEFAULT now()
);

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
    embedding   FLOAT[N],                   -- N fixed per workspace, recorded in _quack_meta
    embedding_profile TEXT                  -- fingerprint of the profile it was made under
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
    embedding_profile  TEXT,                -- fingerprint of the profile it was made under
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
    metadata   JSON,                         -- tool: ToolMeta (tool, detail, duration_ms, rows); assistant: AssistantMeta (chart, citations, write_refused, graph, usage)
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

Every stored vector carries its embedding profile's fingerprint (section 6.1). Vector
search, label matching, and merge proposals compare only current-profile vectors, so a
changed model, width, or prefix never mixes vector spaces. Other-profile vectors stay; their
chunks are found by keyword search until `quack embeddings refresh` (terminal
`/embeddings refresh`, `POST .../embeddings/refresh`, the Documents page's button) re-embeds
them. The terminal, print mode, `quack doctor`, and the Documents page report the count.

When the configured width differs from the stored one, open keeps the old columns and
vectors (a mistyped `[embedding].dimension` must not cost a workspace its embeddings). New
chunks are stored without vectors; the refresh retypes the `embedding` columns of
`_quack_chunks` and `_quack_graph_nodes` through NULL, then embeds everything. With no chunk
vector stored, open adopts the new width at once. Schema version 8 tags pre-profile vectors
with the profile they were made under: the recorded model, no prefixes.

Session and message writes are small and frequent. With one writer connection per
workspace, they serialize with ingestion writes, but in the writer's interactive line,
ahead of any waiting background write (section 4.1): the connection lives on the writer
thread, and every write is a closure sent to it.

### 5.5 `control.db` (server only, SQLite, sea-query queries, SQL-file migrations)

`<data_dir>/control.db`, opened by `quack serve` and the admin subcommands, answers one
question: who may open which workspace. It holds nothing that reveals a workspace's
contents.

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

-- a model provider's OAuth token (`quack auth login`), HPKE-sealed by the vault; section 10.2
CREATE TABLE provider_tokens (
    provider   TEXT PRIMARY KEY,           -- [providers.NAME]
    key_id     TEXT NOT NULL,
    enc        BLOB NOT NULL,
    ciphertext BLOB NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- an OAuth client's private_key_jwt signing key (P-256, PKCS#8), HPKE-sealed by the
-- vault; section 10.2
CREATE TABLE client_keys (
    -- '<issuer> <client_id>'; '<issuer>' alone until the client is registered
    name       TEXT PRIMARY KEY,
    key_id     TEXT NOT NULL,
    enc        BLOB NOT NULL,
    ciphertext BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- a client quack registered itself (RFC 7591), one per issuer; its registration access
-- token HPKE-sealed by the vault (NULL when the issuer returned none); section 10.2
CREATE TABLE client_registrations (
    name                    TEXT PRIMARY KEY,  -- the issuer, without a trailing slash
    client_id               TEXT NOT NULL,
    key_id                  TEXT,
    enc                     BLOB,
    ciphertext              BLOB,
    registration_client_uri TEXT,
    created_at              TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at              TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- a signed-in user's identity-provider token, HPKE-sealed under the server's key
-- (the vault key: the OS keychain or <data_dir>/vault.key, never here); section 12
CREATE TABLE user_tokens (
    user_id    TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    key_id     TEXT NOT NULL,              -- which server key sealed it
    enc        BLOB NOT NULL,              -- HPKE encapsulated key
    ciphertext BLOB NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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

`audit_log` is the access record, required in every mode except `--local`. It is
append-only: the code has no `UPDATE` or `DELETE` path, and its sqlx connection runs neither
statement in any query. There is no retention or pruning; rows stay until an operator
retires the file. Every request naming a workspace writes a row, denials included: a
non-member opening a workspace, a `viewer` writing, an expired token. A control-plane change
(a user, a workspace, a membership, an API token) commits in one transaction with its row, so
none stands unaudited; a sign-in and a logout write their row before the session opens or
closes. `resource_id` is an
opaque UUID or a table name: enough to answer "who read document X" without revealing its
text. An admin sees that a user queried a workspace and in which session; only members see
the query text, in `_quack_audit`.

`quack audit` filters by user, workspace, action, outcome, and time range, prints
`--format text`, `json` (NDJSON), `csv`, or `ocsf`, and follows pages itself (`--limit 0`
reads the whole log). `GET /api/v1/admin/audit` takes the same filters and `?format=ocsf`,
caps `limit` at 1000 a page, and answers JSON with a `next_cursor` for the same filter
(`null` on the last page). Pages are keyset on `(timestamp, id)`, newest first, so rows
appended mid-paging never shift what is left to read. `ocsf` renders each row as an OCSF
1.9.0 event (`quack_core::ocsf`) from `control.db` fields only: Authentication [3002] for
logins, logouts, stale sessions, and rejected bearers; API Activity [6003] for the rest.
Stored rows keep their shape.

Workspace names are unclassified. A deployment needing opaque names uses the `name` column
as the directory name and keeps a display name in `_quack_meta`.

### 5.6 SQL construction rules

- `control.db`: sea-query for every runtime query. Schema changes are literal SQL files
  under `crates/quack-core/migrations/`, run by `sqlx::migrate!`. A handful of `PRAGMA` and
  `sqlite_master` statements are hand-written strings.
- `data.duckdb` internal statements: parameterized via `duckdb::params!`. Table and column
  names go through `quote_ident`, the single identifier path. The only other interpolation
  is the workspace's `embedding_dimension`, a `u32` field, into `FLOAT[N]`. Vector literals
  go through `Vector::sql_literal` and are bound, not interpolated. File paths are bound as
  parameters to `read_csv_auto(?)`. Graph traversal issues one constant query per frontier
  with bound parameters; there is no recursive CTE (section 6.4).
- Agent-generated and user-typed SQL: executed as-is through the permission layer
  (section 7.4), never assembled by application code.

---

## 6. The Knowledge Engine

### 6.1 Documents and vectorization

**Sources.** File upload (web, REST, desktop), paste (web: a text box that becomes a
document), path (TUI, CLI), stdin (print mode), and an OKF bundle as a directory or a tar
(section 17 item 13). URL, GitHub, and Confluence connectors are deferred (section 18).

**OKF bundles are a one-way knowledge export.** Table data, document text, and audit detail
never leave the workspace. `quack okf export` writes `index.md` from the context, a
schema-and-samples stub per table, a metadata stub per document, the ontology as Markdown
files plus an exact JSON snapshot (`ontology/ontology.md`), one entity file per graph node
(its id, provenance, and links that resolve to the target's file), and `log.md` from the
ontology versions only.

The export streams each file to a directory or into the tar as it is made
(`okf::BundleSink`). One query yields the graph nodes with each entity file's path computed in
`DuckDB` (the label slug, plus an id suffix when two labels of a class share one). The API
sends the tar as written and audits the export when the stream ends. Memory holds one node
and the index, written last.

Importing a bundle (`quack ingest DIR`, `POST .../documents` with a tar) ingests every
concept file with text as a document; quack's own stubs are marked `generator: quack` and
skipped. It restores the ontology snapshot when the workspace has none, else proposes the
bundle's types and links as candidates, and offers `index.md` as the context. `quack graph
extract` rebuilds the graph once the tables and documents are back.

**Parsing.**

| Type | Parser | Extracted metadata |
|------|--------|--------------------|
| PDF | `pdf_oxide` | page numbers, Info title; an unreadable page is skipped and counted, never the rest of the file |
| Markdown, plain text | direct | headings (ATX and setext) |
| HTML | `scraper` (html5ever) | headings, `<title>` |
| DOCX | `zip` + `quick-xml` | headings from `Heading N` and `Title` styles, core title |
| PPTX | `zip` + `quick-xml` | one section per slide, slide title as heading, slide number as page |
| CSV, Parquet, JSON, JSONL, XLSX | DuckDB (section 6.2) | become tables, not chunks |

A scanned PDF (no text layer) is reported as `error: no extractable text`. OCR is deferred.

**Chunking.** A fixed token window: 512-token target, 64-token overlap, stepping by the
difference; token counts via `tiktoken` (`cl100k_base`). A sectioned source (Markdown, HTML,
DOCX headings, PPTX slides, plain text) splits at section boundaries first, so no chunk spans
two sections; within a section the window ignores paragraphs and sentences. The chunk stores
its nearest preceding heading, which the embedding model gets as the chunk's title.

A PDF is one continuous text: pages are joined by a blank line and windowed as a whole, so a
paragraph split by a page break stays in one chunk. Each chunk records the page of its first
token. Its heading is the document's Info title (else the filename stem), since a PDF has no
heading to give the embedding context.

**Embedding roles and profiles.** Every text is embedded in a role: a search query, a
document chunk, or a text compared with its own kind (entity labels, a name looked up among
them, ontology type names). Most embedding models were trained with a prefix per role and
retrieve worse without it. Ollama adds none, so `quack_core::embedding` does, per model
family, from each model card: EmbeddingGemma's
`task: search result | query: ` and `title: {title} | text: `, Qwen3-Embedding's query
instruction, nomic's `search_query: ` and `search_document: `, E5, BGE, mxbai, Snowflake
Arctic; none for all-MiniLM, BGE-M3, granite, or OpenAI's. `[embedding]` overrides each role
(section 13). A document prefix with a `{title}` slot gets the chunk's heading there (`none`
without one); otherwise the heading leads the text.

Every call goes through `Embedder`, whose `Input` names the role. Every returned `Vector` is
checked against the profile's `Dimension` (`[embedding].dimension`), so a mismatched model
fails with the fix, not a cast error. Clippy's `disallowed_methods` keeps raw embedding calls
out. The model, its width, and its prefixes are the embedding profile.

**Embedding.** Batches of `[ingestion].embedding_batch_size` (default 64, at least one) go to
the embedding provider, `[ingestion].embedding_concurrency` (2) in flight. Each batch's
vectors are written in one transaction as it returns, overlapping the requests still
running. The provider sets the ceiling. An OpenAI-compatible endpoint answers concurrent
batches in parallel. Ollama's runner embeds one input at a time regardless of batch size or
concurrency (measured: about 14 chunks a second for a 0.6B model on Apple silicon); the
levers are `OLLAMA_NUM_PARALLEL` and a smaller embedding model.

Every ingest logs chunk count, batches, seconds, and chunks per second (`embedded chunks`);
`quack ingest` prints them. No index is built: vector search is an exact scan,
and a chunk's term rows are appended on insert. A full term rebuild happens only when an
older workspace is opened (schema version below 6). Re-uploading a file with the same
SHA-256 is a no-op with a message.

**Hybrid retrieval.** A query runs an exact cosine scan over `embedding` (core
`array_cosine_distance`) and a BM25 search over the terms quack tokenized at ingest
(`_quack_terms`, scored in SQL; no DuckDB extension). This is the main gain over the pgvector setup,
which leaves keyword-exact questions (part numbers, policy IDs) unanswered.

- *Tokens:* lowercased alphanumeric runs through the Snowball English stemmer
  (`rust-stemmers`), so `renewals` meets `renewal`. Queries tokenize identically.
- *Joined identifiers:* a run joined by `-`, `.`, `_`, `/`, or `:` without whitespace, such
  as `POL-8841`, also indexes its punctuation-stripped, unstemmed form (`pol8841`) beside the
  pieces (`pol`, `8841`). A query for it ranks a chunk containing it above one with `pol`
  and `8841` apart (`storage::workspace::tokenize`, issue #77).
- *Quoted phrases:* `"..."` requires exact adjacency. `_quack_terms` has no positions, so
  BM25 ranks by the phrase's tokens, over-fetched, and a post-filter keeps chunks whose
  content or heading contains the phrase (case-insensitive, whitespace-normalized). A phrase
  matching nothing returns no keyword results, not the unfiltered ranking.
- *Rebuild:* the term index is rebuilt on open when a workspace predates the stemmer or the
  joined identifier form.
- *Fusion:* each ranking is over-fetched to twice `top_k` (more with a phrase), fused by
  reciprocal rank fusion (`k = 60`), and the top `k` chunks (default 8) returned.
- *Reranking:* `analysis::rerank::Reranker` sits between fusion and the answer, off by
  default (`[retrieval].rerank = "none"`). `"model"` over-fetches `rerank_candidates` (24)
  and has the chat model order them listwise in one tool-less call, so an air-gapped
  deployment reranks with the model it already runs. A failed ranking call keeps the fused
  order, and the tool step says so. A cross-encoder provider fits the same trait.

**Citations.** Every retrieved chunk carries `document_id`, `filename`, `title`, `page`,
`heading`, and its fused score. The agent cites with `[n]` markers mapped to these chunks.
The core strips markers that reference no chunk retrieved in that turn (including provider
"channel" markers leaked into the text) and renumbers the rest from 1 in order of first use.
Interfaces render citations as links to the document and page.

**Pinned documents.** A pinned document's full text is injected into the system prompt each
turn instead of retrieved, within `[retrieval].pinned_token_budget` (default 8,000). One
that would exceed what is left is skipped with a visible "(omitted)" line, not truncated.

### 6.2 Tables and analytics

| Type | Mechanism | Result |
|------|-----------|--------|
| CSV, TSV, Parquet, JSON, JSONL | `read_csv_auto` / `read_parquet` / `read_json_auto` | Table in `data.duckdb` |
| Excel `.xlsx`, `.xls`, `.ods` | `calamine` (pure Rust) writes each sheet as CSV under `files/` for `read_csv_auto` | One table per data sheet: `<stem>` for one sheet, `<stem>_<sheet>` otherwise; recorded on the document row so deleting it drops them |
| stdin (print mode) | sniffed | Temporary table `stdin` |
| Postgres, SQLite | `quack import URL --table T (--from SOURCE_TABLE \| --query SQL) [--limit N]`, `POST .../import`, the Tables page form, `/import` in the terminal (below) | Table in `data.duckdb`, a snapshot of the source at import time |
| CSV, Parquet, JSON, workbooks over HTTP(S) | The same command with an `http(s)://` URL naming any file `quack ingest` loads as a table (`parser::table_extensions`: CSV, TSV, Parquet, JSON, JSONL, and XLSX, XLSM, XLS, ODS workbooks): reqwest fetches the file and it goes through the usual reader under the requested table name | Table in `data.duckdb` |
| MySQL, S3 | Not yet: MySQL needs the sqlx driver enabled and its identifier quoting; S3 needs request signing (the `object_store` crate is the candidate). The scanner and httpfs extensions stay out (section 15). | — |

**Database import.** sqlx runs the query on the source with every column cast to text. Rows
pass through `files/<table>.csv` and `read_csv_auto`, so `DuckDB` sniffs the types. The table
is a document (source `import`, title the redacted URL), deletable like any other. The URL's
password is used once and never stored; audit rows carry the redacted URL. Caps:
`[import].max_rows` (a file is cut to it after the load), `max_download_mb`,
`timeout_seconds`. `sqlite:` paths inside `[general].data_dir` (`control.db`, the workspace
files) are refused for every caller. The CLI, the terminal, and `quack serve --local` run as
the owner and reach any other source. `quack serve` with logins refuses `sqlite:` paths
unless `[import].allow_local_files` is on. Unless `allow_private_hosts` is on, it resolves
the host first, refuses loopback, private, link-local, and metadata addresses, pins the
connection to the checked addresses, and does not follow redirects.

**Table naming.** The sanitized file stem. On collision the web UI and TUI ask (replace,
rename, skip); the API and print mode require an explicit name. The prompt describes tables
live on every build, with no cache (section 7.2).

Every SQL statement, the agent's or the user's, passes classification and resource limits
(section 7.4).

### 6.3 Ontology

The ontology is the knowledge graph's schema: entity classes, the relations between them,
and their properties. It keeps extraction consistent across thousands of documents, lets
"all organizations" include subclasses, and lets a domain team describe its world once
instead of correcting the model per chunk.

It lives in the workspace's `_quack_ontology_*` tables (section 5.4) under the workspace's
access control: class and relation names alone reveal what a workspace is about.

**Model.** Single inheritance from the implicit root class `entity`. Relations have a domain
and a range class, each satisfied by any subclass. Properties are typed
`string | number | date | enum | boolean` and inherited. Every ontology implicitly contains
the `mentions` relation (`entity` to `entity`), so extraction never has to invent one. Ids
are `snake_case` and stable; a rename is a new id plus a migration of nodes and edges.

**Interchange format.** JSON, the stored snapshot's shape, used by `quack ontology export`
and `import`, `GET/PUT .../ontology`, and the web editor's "download" and "save". Domain
packs (an insurance ontology, a legal ontology) are shared between workspaces, diffed in
review, or seeded into new workspaces this way. The file is never the source of truth. The
example below is that shape written as YAML for brevity.

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

**Built-in default.** When the graph is first enabled and no proposal has been accepted, a
general ontology is installed as version 1: classes `person`, `organization`, `place`,
`event`, `product`, `document`, `concept`; relations `works_at`, `located_in`, `part_of`,
`produced_by`, `occurred_at`, `mentions`.

**Where it is used.**

1. *Extraction prompt.* Classes with parents, relations with domain and range, and property
   definitions are rendered into the prompt. The model may return only ontology ids;
   anything else is dropped and counted (section 6.5, drift).
2. *Validation.* Before an extracted node or edge is written, quack checks that the class
   exists, the relation exists, and its domain and range hold under inheritance.
   `revalidate` re-applies these checks after an ontology change. Property types and enum
   values are validated on the ontology document, not on instance data: `upsert_node`
   stores the properties JSON as given.
3. *Query expansion.* `search_graph(class: organization)` matches `vendor` too. The agent's
   prompt includes a compact rendering of the ontology for class-aware questions.
4. *Table mapping.* A mapping turns a table's rows into nodes and its foreign-key-like
   columns into edges without an LLM, bridging the analytics and graph substrates.

**Versioning.** Every accepted change writes a `_quack_ontology_versions` row with a full
snapshot. `_quack_meta.graph_built_with_ontology_version` records what the graph was built
with. When it lags, the graph is stale and the interfaces offer re-extract (cost shown
first) or revalidate (fast; drops nodes and edges that no longer validate). Any version can
be diffed against another or restored. Deleting a document removes its files under `files/`
and the graph nodes and edges whose only provenance was that document or its tables, with
their provenance rows. A mapping whose table is gone stays in the ontology (saves still
succeed); extraction skips it, and `graph status` lists it under `missing_tables`.

### 6.4 Knowledge graph

**Extraction from documents.** On demand (`quack graph extract`, `POST .../graph/extract`,
the graph page), permission-gated, with cost (chunk count, model) shown first. Each chunk
goes to the chat model with the ontology-derived prompt and must return JSON `{nodes:
[{label, class, properties}], edges: [{source, target, relation, properties}]}`. Parsing is
deliberately lenient, because models wrap JSON in prose: first `{` to last `}`, every list
defaulting to empty. A chunk that still fails is logged and skipped, never retried in a
loop. Each chunk of a ready document goes to the model once: `_quack_graph_extracted`
records every chunk processed (with the ontology version and its yield), so a later run
sends only new chunks, and `--reset` starts over. A sample of N takes chunks spaced evenly
across documents, not the first N ingested. The raw answer and parse outcome are logged at
debug level.

**Extraction from tables.** Ontology mappings turn rows into nodes and edges
deterministically, with provenance `table_name` and `row_key`; re-running is idempotent. It
runs in batches of 5,000 keyed rows. Rust stages each batch (nodes, edges, and row
provenance deduplicated, ids minted as UUID v7), then writes it with one statement per kind
through scratch `_quack_tmp_graph_*` tables, in one transaction under the statement timeout.
The server releases the workspace lock between batches and runs one extraction per workspace
at a time; a second `POST .../graph/extract` answers 409.

**Entity resolution.** Nodes are merged on `(normalized_label, class_id)`. A second pass
checks each node's five nearest neighbours and proposes merging same-class nodes whose label
embeddings are within a cosine threshold (default 0.08) and whose labels share a token. A
pair within `auto_merge_threshold` (default 0.02) merges at once; the rest land in
`_quack_graph_merges` as pending proposals. Provenance rules the pass: two nodes from keyed
table rows are distinct by construction and never paired (WEST VIRGINIA is not VIRGINIA); a
pair with one keyed side is proposed only with the keyed node kept; auto-merge applies only
to two extracted nodes. Each merge is one transaction. Aliases go in `properties.aliases`.

**Provenance.** Every node and edge has at least one `_quack_provenance` row. Graph answers
cite the source chunk or row as document answers cite chunks.

**Traversal.** Breadth-first in Rust over plain SQL: one constant `store::edges` query
(`EdgeScope::Touching`) per frontier, with bound parameters, no DuckPGQ, no recursive CTE. A
walk never visits more than `[graph].max_nodes`. The CTE this design first specified was
removed (#48): it enumerated every simple path out of a hub before its `LIMIT` applied, and
never came back on a hub node.

- *Entry point:* an exact match on `normalized_label` or a `properties.aliases` value, with
  the class filter in the same query; else the three nearest node embeddings within a
  distance of 0.25; else nothing.
- *Operations:* `neighborhood(entity, hops, relation?)`, `path(a, b, max_hops)`
  (single-source BFS with a parent map, bounded by `max_traversal_depth * 2`),
  `by_class(class, limit)` with subclass expansion.
- *Limits:* `max_traversal_depth` (3) and `max_nodes` (200), the cap on nodes visited.
- *Hops:* every interface reads a hop count through `traverse::Hops`: at least 1; unset is 2
  for a neighborhood, 4 for a path. `--hops 0`, `hops: 0`, and `/graph X 0` mean the same.

**Rendering.** TUI and `quack graph` print a depth-first tree. The web UI renders an ECharts
`graph` series with class-colored nodes and an inspector for properties and provenance. The
API returns `{nodes, edges, provenance}`.

### 6.5 Ontology induction: discovering and proposing

quack proposes the ontology from evidence and a person accepts it. Proposals are rows in
`_quack_ontology_candidates`; the live ontology changes only when one is accepted.

**Evidence from tables (deterministic, no model calls).**

- Each table proposes a class named from the table. Each column proposes a property typed
  from DuckDB's column type and value profile (`enum` when distinct values are few and
  stable; `date` when the column parses as one).
- A unique, non-null column proposes the class key.
- A column whose values overlap heavily (default 80%) with another table's key proposes a
  relation between the two classes, named from the column by a deterministic rule
  (`policy_id` -> `has_policy`).
- The result is also proposed as a mapping, so accepting it turns rows into nodes at once.

**Evidence from documents (open extraction on a sample).**

1. Take a stratified sample of chunks across documents (default 200, configurable), so
   every document contributes.
2. Extract unconstrained: free-form entity types, relation names, and observed attributes
   with values. Chunks run `[analysis].extraction_concurrency` at a time, each call bounded
   by `[analysis].extraction_timeout_seconds` (a timed-out chunk is skipped and counted).
   Each finished chunk is reported: a stderr line in the CLI, a log line in the server.
   Graph extraction (6.4) runs the same way.
3. Normalize the vocabulary. Raw type and relation names are grouped by snake_case singular
   equality; near-synonyms are clustered by embedding cosine when an embedding model exists
   (`cluster_threshold`, default 0.9). The cluster's most frequent raw name becomes its id;
   naming makes no model call.
4. Infer structure. A relation's domain and range are the classes seen at its endpoints,
   generalized to the nearest common ancestor when mixed. Hierarchy is inferred where one
   type's mentions are consistently also labeled with a broader type (`vendor` under
   `organization`). Attributes that recur on a class propose typed properties.
5. Score. Each candidate carries occurrence count, distinct-document count, three example
   mentions with chunk ids, and a confidence from support and cluster tightness. Candidates
   below the support threshold (default 3 documents) are kept as `low_support`, outside the
   main proposal.

**Cost.** Shown before the run: sample size, model, and estimated calls, one per sampled
chunk (200 for a 200-chunk sample). Chunks of 40 characters or fewer are excluded from both
estimate and sample. The run is permission-gated and audited.

**Review.** The web ontology page, `quack ontology review`, and `GET
.../ontology/candidates` show the proposal grouped by kind, evidence inline. Per-candidate
actions: accept, rename, merge into an existing class or relation, reparent, reject.
Accepting writes a new ontology version. Scripts use `PUT .../ontology/candidates/{id}` and
`quack ontology accept ID...`; `POST .../ontology/candidates` with `{accept: [ids], reject:
[ids]}` decides many at once, like the page's tick boxes. The page shows fifty candidates at
a time, pending or low-support (a filter link, never hidden). A mapped table counts as
covered: its rows belong to the mapping's class, so no class or mapping is proposed for it,
and a column its mapping already relates proposes no relation. New columns still propose
properties.

**Modes.**

- `propose`: propose what the current ontology lacks, a full draft when there is none.
  Afterwards it proposes only additions and reparents, so a rerun never re-queues accepted
  items. There is deliberately no mode that ignores the current ontology: to start over, use
  the version verbs (`init`, `import`, `restore`), which keep every earlier version. Drift
  also triggers it: constrained extraction counts every type or relation the corpus tried to
  express that the ontology has no place for, and past a threshold the interfaces show "the
  corpus wants N things the ontology lacks; propose extensions?".
- `propose --from PACK`: seed from an imported domain pack, then propose what it lacks.
- `--auto-accept`: accept the proposal and build the graph without review, for a first
  look. Everything built this way is marked `provisional` in the graph tables, the
  interfaces show a banner until someone reviews, and query mode answers exclude it.

**Graph proposal follows the same loop.** After an ontology is accepted, constrained
extraction over the full corpus builds the graph, entity resolution proposes merges for
review, and drift feeds the next `propose`: propose from evidence, review, extract, observe
drift, propose again.

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

`max_turns` 15, temperature 0.1. `llm::sampling` adjusts that per model. Only Ollama's own API
is sent the temperature, because Claude and `OpenAI`'s reasoning models (GPT-5.x, GPT-6,
o-series) reject a non-default one and a gateway's model name need not say which model it is.
Claude gets `max_tokens` 64,000, since thinking counts against it. `[analysis].effort` goes out
as each API's field (`output_config.effort` for Claude, `reasoning_effort` or `reasoning.effort`
for `OpenAI` and unrecognized models on those APIs, `think` on Ollama). A level a known family
lacks is refused before the call, and so is a GPT-5.6 model on Chat Completions, because it
cannot call tools there. `temperature`, `effort`, and `background_effort` on a provider, or on
one of its `models."ID"`, override the defaults and `[analysis]` (`docs/providers.md`). The turn races a
`CancellationToken`; a cancelled turn keeps the text streamed so far, appends a note, reports `cancelled: true`, and is recorded.

`llm::TurnRequest::run` yields these events on a channel. The web UI turns them into HTML
fragments over SSE; REST forwards them as typed SSE events or collects one JSON response;
MCP collects them into the tool result; the TUI renders them inline; print mode writes them
to stderr. Before the first model call, an Ollama turn asks `GET /api/ps` whether the chat
model is loaded. If not, it emits `Status` ("loading MODEL ..."), since a cold load of a
12 GB model takes seconds with nothing else to show. Print mode shows it on the spinner, the
terminal as a system line, SSE as a `status` event.

### 7.2 System prompt

1. Role and behavior for the mode (7.5): retrieve before answering, cite with `[n]`, run
   SQL rather than estimate, state assumptions, ask one clarifying question when the
   request is ambiguous.
2. Tool guidance, the error rule (read a `run_sql` error, fix the statement, rerun it), and
   a Friendly SQL reference pinned to the bundled DuckDB version, which the prompt states.
   The reference covers only what the confined connection (7.4) can run (no file reads,
   extensions, or `SET`) and says so, since the tables block is all the data there is. The
   guidance is one numbered procedure per substrate: structured data, document content,
   and, only when the graph tools are registered, how entities relate.
3. Tables block: user-facing tables and views with columns, types, row count, and three
   sample rows. It is bounded so one wide or narrative table cannot push the guidance and
   question out of a small window: the first 25 tables are described and the rest listed
   by name; the first 40 columns are listed and the rest counted; sample rows appear only
   up to 20 columns, cut at 60 characters per cell. `describe_table` has the rest.
4. Documents block: every document by filename, title, status and mime type, then the
   pinned documents with their full text (6.1).
5. Ontology block, whenever an ontology exists: classes with parents, relations with domain
   and range (compact), capped at 30 items per section with the rest counted, since an
   induced ontology has a class per table; `describe_class` has what the cap omits. Node and
   edge counts, and whether the graph is provisional or stale, follow only when the graph
   has content.
6. Global context prefix, then the workspace context.
7. The permission rules.

The guidance always names the table, SQL, chart and document tools; only the graph tools
are conditional. Mode changes no registration; query mode only drops provisional graph
results.

**Ollama.** Every request carries `num_ctx`, because Ollama otherwise loads the model with a
4,096-token window and silently truncates the front of the prompt. It is the prompt's
estimated tokens plus a fixed 8,192-token headroom for tool results and answer, rounded up
to 8,192, capped by `[analysis].max_context_tokens`, never below 8,192. `num_ctx` is a load
option: a changed value forces a full reload (measured: several seconds for a 20B model).
The coarse step means a growing session's history crosses it a few times at most, not every
2,048 tokens. Every request also carries `keep_alive` (30 minutes), since otherwise a gap
between tool calls or turns pays the same reload once Ollama's default (5 minutes) lapses.

Embedding requests go through quack's own `/api/embed` client (`llm::OllamaEmbedder`),
because rig's sends neither option. They carry the same `keep_alive`, so the embedding model
stays loaded between a turn's query embedding and its chat call. Their `num_ctx` fits a
chunk: twice `[ingestion].chunk_size_tokens`, rounded up to a power of two, at least 2,048.
Otherwise Ollama loads the model at full length (measured: 32k for qwen3-embedding, 5.8 GB
of cache against 2.1 GB, same throughput). On a host that cannot fit both models at full
size, this stops them evicting each other every turn.

A turn the model derails (a call to a nonexistent tool, the `max_turns` limit), or that
fails after text streamed, keeps its text, gains a note saying what happened, and is
recorded. Only an unreachable model is an error.

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

`search_graph` and `find_path` register only when the graph has nodes; `describe_class`
whenever an ontology exists, since the prompt's ontology block is capped. The rest register
in every workspace, and the prompt tells the model what the workspace holds. An `export`
tool (`COPY ... TO` under `files/`) is not built (section 17).

**The substrates cross in the tools, not only in the store.** `search_documents(entity)`
resolves the name against the graph and restricts retrieval to the chunks that entity was
extracted from. Every hit names the entities the graph took from it, so the model can follow
one into `search_graph`. The `entity` argument is in the schema only while the graph has
nodes, like the graph tools: measured live, a model shown it without a graph tries it, is
refused, and spends a second round trip and twice the tokens for the same answer. Graph
provenance to a mapped table renders as a predicate `run_sql` can run
(`"orders" WHERE "order_id" = 'A-42'`), since the mapping records the key column. An entity
known only from table rows says so rather than returning nothing.

Both graph tools render each node and edge with its ontology-typed properties, at most eight
per subject and sixty characters per value. An empty lookup says why, so the model retries
instead of reporting an empty workspace. An undefined class or relation id is
an error naming the ids that exist (as `search_documents` does for `document_ids`). A name
matching no entity returns the closest labels. A result query mode emptied by dropping
provisional nodes says the matches exist but are unreviewed.

A result cut short by `max_nodes` says so, and a class listing carries the total it was
capped from (`GraphResult::total_nodes`, `truncated`); otherwise the reader takes the cap
for the class's population. Traversal cannot count, and user SQL may not read `_quack_`
tables, so `describe_class` reports the exact count of a class and its subclasses.

### 7.4 Permissions and limits

**Classification.** Before `run_sql` executes, DuckDB's own parser classifies the statement
via `SELECT json_serialize_sql(?)`, with three outcomes: it serializes (read), a parse error
(invalid, returned as a syntax error, not a write), or anything else (write). `DESCRIBE`,
`SHOW`, `SUMMARIZE`, `PIVOT`, `UNPIVOT` and `EXPLAIN` are read by an explicit allow-list,
because DuckDB cannot serialize them. `COPY`, `INSTALL`, `LOAD`, `ATTACH`, `SET` are always
write. Statements referencing `_quack_` tables are refused regardless.

**Decision by interface and role.**

| Interface | Read | Write |
|-----------|------|-------|
| TUI | run | prompt `y`/`n`/`a` showing the SQL; `a` covers the rest of the turn and the session |
| Print mode | run | refuse unless `--allow-write`; the answer completes and the exit code is 3 |
| Web / REST | run | refuse unless `allow_write: true` from a member with the write scope (a request that asks for `allow_write` without it is 403); a refusal inside the turn is not a failed request: 200 with `write_refused: true` on the response object and a `write_refused` SSE event; the web page shows a banner offering the checkbox |
| MCP | run | refuse unless `quack mcp --allow-write` set the policy at launch (stdio has no tokens); `write_refused: true` in the structured content and a sentence in the text. Over HTTP the token's `write` scope decides |
| Desktop | planned | native confirm dialog (section 11.6) |

Every interface returns one response object (11.2), built by `AgentResponse::to_json`:
`answer`, `citations` (each with `n`, `chunk_id`, `document_id`, `filename`, `chunk_index`,
`page`, `heading`, `label`), `queries`, `steps`, `graph`, `chart`, `write_refused`,
`cancelled`, `usage`, `session_id`. `AuthRequired` is exit code 4 from every command that
reaches a provider.

`usage` is the provider's report for the turn (`input_tokens`, `output_tokens`,
`total_tokens`): rig's aggregate over the turn's completion requests, or the sum of
per-request counts when the turn derailed before a final response. It is `null`, not zeroes,
when the provider reported nothing, as local models often do. The counts also go on the
assistant message's metadata in `_quack_messages`, so session exports carry them. They are a
record, not an input: the history trim and Ollama's `num_ctx` estimate before the call.

**Limits.** The agent's connection runs with `SET memory_limit` and `SET threads` from
config. A statement runs on the calling thread (for the agent, a `spawn_blocking` one) while
a watchdog thread holds `Connection::interrupt_handle()` and calls `interrupt()` after
`query_timeout_seconds`; a guard disarms it when the statement returns. User SQL has the
same limits.

**Confinement.** Classification is not enough: `SELECT * FROM read_text('/etc/passwd')` is a
read. So the workspace connection is confined at open, before any user or agent statement.
`allowed_directories` is the workspace directory alone (ingestion reads the originals it
copied under `files/`). `enable_external_access` is off, so file readers, replacement scans,
`COPY`, `ATTACH`, `INSTALL`, and `LOAD` fail anywhere else. `allow_persistent_secrets` is
off. `lock_configuration` is on, so no later `SET` can widen any of this or lift the limits
above. The in-memory test database gets the same treatment with an empty allow-list.

### 7.5 Chat modes

Per session: `chat` unless set at creation (`--mode`, the REST `mode` field, the web
selector, the MCP `mode` argument), changed only explicitly (`/mode`,
`PATCH .../sessions/{sid}`).

- **chat** - the agent may answer from general knowledge as well as retrieved sources; it
  must still cite a source it used.
- **query** - the agent must ground every claim in a retrieved chunk, a query result, or a
  graph result; if retrieval finds nothing relevant, it says so instead of answering. This
  is AnythingLLM's query mode and the default for classified workspaces. Provisional graph
  results are excluded.

### 7.6 Visibility

In every interface, each tool call renders one line at start and one at finish, with up to
three lines of detail preview and a "+N more" tail:

```
> search_documents "policy exclusions for flood"
  8 chunks, 41 ms   [1] Policy-2024.pdf p.12  [2] Policy-2024.pdf p.13  ...
> run_sql
  SELECT status, COUNT(*) FROM claims GROUP BY 1
  4 rows, 9 ms
```

The web UI shows them as a collapsible steps block above the answer, citations as links. The
TUI shows them inline, with `/sql` to reopen the last query. `--verbose` in print mode
includes full payloads.

---

## 8. Sessions

Every turn belongs to a session in `_quack_sessions` (AnythingLLM's threads). A session is a
complete record inside the classification boundary. The whole turn is written at the end in
one call: the user message, one tool message per step with its metadata, then the assistant
message.

- Resume: `session_id` on the REST request, `--continue` / `--resume` in the TUI, the thread
  list in the web UI.
- Export, from every interface: `.sql` (every `run_sql` and `create_chart` statement, with
  the question as a comment) or Markdown (questions, tool steps with summaries and detail,
  answers, and a note where a chart is attached). Export is audited because it moves content
  across the boundary.
- History sent to the model is trimmed to `history_token_budget` (32,000), oldest first.
  Tool messages are never replayed; only user and assistant text goes back.
- Server mode: sessions carry `created_by`. Members see their own, any marked `shared`, and
  any with no creator (started from the CLI or the TUI). `owner` sees all sessions in the
  workspace for audit.

---

## 9. Charts

`create_chart` produces one small spec: ratatui renders it in the TUI, the web UI and
desktop map it to an ECharts option, and REST, MCP, and print mode emit it as JSON.

```json
{
  "title": "Claims by status",
  "kind": "bar",                       // bar | line | scatter | pie
  "x": { "label": "status", "values": ["filed", "paid"] },
  "series": [ { "name": "count", "values": [120, 340] } ]
}
```

One x axis, one numeric series (the tool takes a single `y` column), at most 200 points; a
larger result refuses the query rather than sampling. A NULL x becomes the label "NULL", a
NULL y becomes 0, and a non-numeric y is an error reported to the model. The chart's SQL
always runs read-only, so charting never prompts for a write. A chart attaches to the
assistant message that produced it and appears there in every rendering.

---

## 10. LLM Layer

### 10.1 Providers

| `type` | Chat | Embeddings | Notes |
|--------|------|------------|-------|
| `ollama` | yes | yes | Offline default. `base_url` defaults to `http://localhost:11434` |
| `openai` | yes | yes | Also OpenAI-compatible endpoints via `base_url` (vLLM, LiteLLM, Azure OpenAI). `api = "responses"` (sent with `store: false`; the default without a `base_url`, since GPT-5.6 calls tools only there) or `"chat-completions"` (the default with one, since compatible servers may offer nothing else) |
| `anthropic` | yes | no | Native Messages API with tool use |
| `bedrock` | yes | yes | Amazon Bedrock's `bedrock-runtime` endpoint: `api = "converse"` (default, rig-bedrock over the AWS SDK), `"chat-completions"`, or `"responses"` (OpenAI-compatible, `/openai/v1`). Embeddings are InvokeModel in Titan Text Embeddings V2's request shape. `region`, `aws_profile`, `base_url` (VPC endpoint) optional |
| `bedrock-mantle` | yes | no | Amazon Bedrock's `bedrock-mantle` endpoint: `api = "responses"` (default) or `"chat-completions"` (`/v1`). Same `region`, `aws_profile`, `base_url` |

`[general].chat_model` and `[embedding].model` each name `PROVIDER/MODEL`. A workspace's
`allowed_providers` filters the choice; the session records the model it used. Changing the
embedding model, width, or prefixes leaves a workspace's vectors stale, not wrong: they are
not searched, and their chunks are found by keyword. `quack embeddings refresh` shows what
it will refresh, asks, and updates them in place (section 5.4).

### 10.2 Authentication

`docs/providers.md` is the operator's guide to connecting to providers and authenticating
with them; this section holds the design.

Each provider has an `auth` mode: `none`; `api-key` (from the env var named by
`api_key_env`); or `oauth`, an enterprise IdP access token as the provider's bearer, which
reaches Azure OpenAI and internal gateways where static keys are forbidden. The oauth
section's `grant` picks the flow: `authorization-code` (default; a person signs in through
the browser with PKCE), `device-code` (a person enters a code on another device), or
`client-credentials` (quack authenticates as itself with `client_id` and the secret in
`client_secret_env`, required; nobody signs in, and the grant reruns whenever the token runs
out).

**AWS.** `bedrock` and `bedrock-mantle` take only a fourth mode, `aws`, their default. The
AWS SDK (`aws-config` with `sso`) signs each request with credentials from its default chain,
as the AWS CLI finds them: `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN`;
the profile named by `aws_profile` (else `AWS_PROFILE`, else `default`) in `~/.aws/config`
and `~/.aws/credentials`, with its `source_profile`/`role_arn`, `credential_process`, and IAM
Identity Center (`aws sso login`) settings; web identity tokens (EKS); the ECS and EC2
instance roles. quack stores nothing; the SDK's caches and refreshes apply.

Bedrock has two endpoints with different models and APIs. `bedrock-runtime`: Converse and
InvokeModel through the SDK, Chat Completions and Responses under `/openai/v1`, cross-region
inference profiles, a FIPS endpoint, embeddings. `bedrock-mantle`
(`bedrock-mantle.{region}.api.aws`): Chat Completions and Responses under `/v1`, and models
and Responses features only it has, such as Responses for GPT OSS. Each is its own provider
type, `bedrock` and `bedrock-mantle` (like LiteLLM's `bedrock` and `bedrock_mantle`), and an
entry names one `api` on it. A model on the other endpoint needs a second entry sharing the
profile, picked by the model reference (`bedrock/us.anthropic.claude-opus-5-5`,
`mantle/openai.gpt-oss-120b`). Config reading refuses an API the endpoint lacks (`converse`
on `bedrock-mantle`) and embeddings on `bedrock-mantle`.

The endpoint root is `base_url` if set, else the one AWS publishes for the region: the
runtime's from the SDK's endpoint resolver (so `use_fips_endpoint` / `AWS_USE_FIPS_ENDPOINT`
and dual-stack apply), mantle's `bedrock-mantle.{region}.api.aws` (FIPS is refused: mantle
has no FIPS endpoint). `base_url` reaches a proxy or an interface VPC endpoint without
private DNS (`https://vpce-….bedrock-mantle.us-east-1.vpce.amazonaws.com`; private DNS needs
nothing). It is the root; quack adds `/v1` or `/openai/v1` per endpoint. An AWS host must
name the provider's endpoint (`bedrock-runtime`, `bedrock-runtime-fips`, or
`bedrock-mantle`), and its region is the signing region unless `region` overrides it; a
disagreeing `region` is refused, and with FIPS required so is a non-FIPS AWS host.
`base_url` replaces only the Bedrock endpoint, never SSO or STS.

Without `base_url`, the runtime root also honors the SDK client's endpoint-URL overrides
(`AWS_ENDPOINT_URL`, `AWS_ENDPOINT_URL_BEDROCK_RUNTIME`, and the `[default]` /
`[bedrock runtime]` profile `endpoint_url` keys), resolved as the bedrock-runtime `Config`
builder does, so the OpenAI-compatible transports and the Converse / `InvokeModel` client
share one host. With FIPS required, an override naming a non-FIPS AWS host is refused; the
root resolves before the SDK client is built, so this fails the whole provider. Mantle has no
SDK client and is unaffected.

The region is `region`, else `base_url`'s, else the SDK's chain (`AWS_REGION`, the profile's
`region`, instance metadata). A provider's first use builds its `llm::bedrock::Session`
(root, signer, and on the runtime the SDK client) and resolves credentials once, so a missing
or expired login fails there with the SDK's reason; the session lasts the process. Converse
and embeddings use the SDK, whose HTTPS client (rustls on aws-lc-rs, FIPS on Linux) is
wrapped so each model call (a `/model/{id}/...` path) takes a `max_concurrent_requests`
permit. The OpenAI-compatible APIs use rig's OpenAI clients over `LimitedHttp`, which
SigV4-signs each request for the endpoint's service (`bedrock`, `bedrock-mantle`) after its
permit, caching credentials until five minutes before expiry. Every Responses request sends
`store: false`, since Bedrock otherwise keeps responses 30 days, outside the workspace file
(section 5); quack replays history itself. `quack doctor` resolves the session and, on
mantle, lists models (`GET /v1/models`) to confirm the model exists.

```rust
pub struct OAuthConfig {
    pub issuer_url: String,          // https://login.microsoftonline.com/{tenant}/v2.0
    pub client_id: String,
    pub scopes: Vec<String>,         // ["https://cognitiveservices.azure.com/.default"]
    pub redirect_uri: String,        // default http://127.0.0.1:19876/callback
    pub grant: Grant,                // authorization-code (default), device-code, client-credentials
    pub client_secret_env: Option<String>,   // confidential client, secret from the env
}

pub struct TokenManager {
    provider: String,
    config: OAuthConfig,
    store: ProviderTokens,           // control.db provider_tokens, sealed by the vault
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

**Lifecycle.** `quack auth login PROVIDER` acquires the token: browser PKCE, or device code
when `grant = "device-code"`, with no browser, or under `SSH_CONNECTION`. Endpoints come from
`{issuer_url}/.well-known/openid-configuration`, else RFC 8414's
`/.well-known/oauth-authorization-server` inserted before the issuer's path; the document
must name the configured issuer. The redirect's `iss` must match it when present, and must
be present when the issuer advertises `authorization_response_iss_parameter_supported`
(RFC 9207). The verifier uses aws-lc-rs randomness. The client authenticates per `client_auth`:
secret in the body, HTTP Basic (`client_secret_basic`), or a client assertion
(`private_key_jwt`, RFC 7523 2.2, below). When discovery lists a
`pushed_authorization_request_endpoint`, the request is pushed there first (RFC 9126) and the
browser carries only its `request_uri`. The token is reused while more than 60 s remain,
refreshed silently under `refresh_lock`, and the flow restarts on refresh failure. In print,
ingest, and server modes, where no flow can run, the call fails with exit 4 / HTTP 503
naming the command to run.

The browser flow is the default because a device code can be phished: an attacker starts a
login, sends the victim the code, and receives the victim's token on approval. The browser
flow delivers the code only to the loopback listener that started the login.

`refresh_lock` covers one process. Two processes on one data directory can both refresh an
expiring token; an issuer that rotates refresh tokens refuses the second, which then
rereads the stored token and uses it if another process stored a different, unexpired one
(`TokenManager::renewed_elsewhere`). An issuer with reuse detection (Okta, Auth0) instead
revokes the whole token family, and every process needs a new login; the operator guide
tells such deployments to make model calls through one long-lived process.

A `client-credentials` provider needs no login: the first request runs the grant, a token
with 60 s or less left is replaced by rerunning it under the same lock (never a refresh
token), and a refused secret is an error naming the grant, not a login prompt. `quack auth
login` on it runs the grant once to check the credentials.

The token lives in `control.db` (`provider_tokens`, one row per provider), sealed by the
vault (section 10.3) for the `provider-token` purpose with the provider name as subject, so a
row copied under another name does not open. `control.db` opens on the first token read or
write, so a command that never reaches the provider never opens it. The vault key is in the
OS keychain where available (macOS Keychain; Windows Credential Manager; the Linux kernel
keyring via `keyutils`, always present but in-memory, so a reboot needs a new login), else
`<data_dir>/vault.key` (0600). `config::ProviderName` checks each `[providers.NAME]` key on
read: ASCII letters, digits, `_`, `-`, at most 64; no `.`, since TOML reads
`[providers.a.b]` as a table inside provider `a`. `quack auth status` and `logout`; logout
deletes the row and keeps the vault key, which other tokens use. The `oauth2` crate runs
without its bundled HTTP client (it would pull `ring`), over the same rustls + aws-lc-rs
`reqwest` as rig. Scopes must include `offline_access` where the issuer needs it to return a
refresh token.

The server shares one `TokenManager` per OAuth provider across requests. A server reaching
Azure OpenAI this way is a confidential client and should be registered as one, with a client
secret or certificate; `client_secret_env` is honored when set.

**Client authentication with a key** (`client_auth = "private_key_jwt"`, on
`[providers.NAME.oauth]` and `[server.oidc]`). Instead of a shared secret, quack signs a
client assertion on every token-endpoint and pushed-authorization-request call: an ES256 JWT
with `kid` the key's RFC 7638 thumbprint, `iss` and `sub` the `client_id`, `aud` the issuer
identifier from discovery (FAPI clients at Vouch accept only that), a UUID v7 `jti`, and a
one-minute lifetime. The issuer spends each `jti`, so a retry or device-code poll signs anew:
`OAuthHttp::sender` appends a fresh assertion to each request the `oauth2` crate sends (that
crate sees a public client, so `client_id` goes in the body, never HTTP Basic), and
hand-built requests (token exchange, PAR) add their own.

The key is P-256, made with aws-lc-rs on first use (`llm::oauth::client_key`), kept PKCS#8 in
`control.db` (`client_keys`, migration 7), sealed for the `client-key` purpose, loaded once
per process, and named `<issuer> <client_id>`, so `[server.oidc]` and a provider registered
as the same client share one key and one registered JWKS. `quack auth jwks [PROVIDER]` prints
the public key set to register; `quack auth status` shows the thumbprint and any waiting
replacement. Entra ID accepts client assertions only for uploaded certificates, named by
thumbprint in the header, so an Entra client uses a secret. A key whose vault key is gone is
replaced, with a warning to register the new one.

Rotation takes two steps. `quack auth jwks --rotate` makes a replacement under
`next <issuer> <client_id>` in `client_keys` and gives the issuer both keys (the same pair if
repeated); the old key keeps signing. `--rotate --activate` moves the replacement onto the
client's name and deletes the old key in one transaction (`ControlPlane::change_client_keys`)
and gives the issuer the new key alone (again if repeated); every running `quack serve` must
then restart. quack sends each set itself (RFC 7592, `Registrar::publish_keys`) for a client
it registered and holds a registration token for, and prints it for hand registration
otherwise.

**Registering the client** (`quack auth register`, `llm::oauth::registration`, RFC 7591 and
7592). A `[server.oidc]` or `[providers.NAME.oauth]` section with `client_auth =
"private_key_jwt"` may omit `client_id`; one registration per issuer then serves all such
sections there (Vouch's model: one client for sign-in and token exchange). The command posts
their metadata to the discovered `registration_endpoint`: the union of their grants
(`client_credentials` only for that grant or an actor token, `refresh_token` only with
`offline_access`); the sign-in callback as a `web` client or a lone loopback redirect as a
`native` one, never both; `private_key_jwt` with ES256 and the `jwks`; the union of scopes;
never `dpop_bound_access_tokens` or `tls_client_certificate_bound_access_tokens`, whose bound
tokens model APIs refuse.

The bearer is one of three. `--token-env` names a token. By default, where discovery
advertises a `registration_endpoint`, public clients (`none` in
`token_endpoint_auth_methods_supported`), and PKCE `S256` (as Vouch does), it is a signed-in
person's token, obtained through a temporary public client (native app, no secret, PKCE,
loopback redirect on a free port) that quack registers, signs the person in through without
keeping the token, and deletes (RFC 7592) only after the real registration, since deleting a
client can end its sign-ins. `--open`, after a warning and a confirmation, registers openly.
An issuer advertising less is refused without `--token-env` or `--open`, never registered
openly; quack never picks the flow by issuer name. Each temporary client's record
(`sign-in <issuer> <client_id>` in `client_registrations`, registration token sealed) stays
until the client is deleted, so an interrupted run (Ctrl-C deletes it on the way out) or a
refused delete leaves the means to delete it: the next `quack auth register` does, and
`quack doctor` names it.

The key is made under the issuer's name alone and moves to `<issuer> <client_id>` in the
transaction storing the registration in `control.db` (`client_registrations`, migration 8:
`client_id`, `registration_client_uri`, and `registration_access_token` sealed for the
`registration-token` purpose). The issuer names the record, so sections find their
`client_id` before they know it; `quack config` shows it with origin `registration`, and
`quack serve` refuses to start without it.

- The record is written only while the issuer's name still holds what the command read
  (`storage::control::Previous`); of two racing registrations, the second deletes its client
  and stops.
- `--replace` registers and keeps the new client before deleting the old one (RFC 7592
  `DELETE`), so a failure never leaves quack without a client.
- Each rotation update reads the registration back and sends all of it with the new `jwks`
  (`PUT` replaces every field), keeping a rotated registration token.
- `quack auth unregister` deletes the client, then the record and the key.
- A hand-registered client is not recorded: its `client_id` goes in the sections, its key is
  made under that name, and nothing takes the pending key over.
- `quack doctor` checks each registration is kept and, online, still readable at the issuer.
- `client_secret_env` must be unset; the key meets the confidential-client requirement of
  `client-credentials` and `on-behalf-of`.
- DPoP (RFC 9449) is not used: it would bind tokens to the key, and model APIs take bearer
  tokens only.

**On behalf of each person** (`grant = "on-behalf-of"`, `quack serve` only). Each request and
background job reaches the provider as the person who made it. quack exchanges that person's
access token for quack (their stored sign-in, section 12, renewed when due, else the
identity-provider token they presented as a bearer) for a token to this provider, kept per
person in memory.

- `exchange` picks the wire form: `token-exchange` (RFC 8693, default; Okta, Auth0, Vouch:
  `subject_token` of type `access_token`, `requested_token_type` `access_token`, `scope`,
  `audience`, and `resource` when set) or `entra` (the `jwt-bearer` grant with `assertion`
  and `requested_token_use=on_behalf_of`).
- With `token-exchange` and `actor = true` (default), quack also sends its own
  client-credentials token as `actor_token`, so the issued token names quack as actor beside
  the person. Entra's form has no actor. With `actor = false` quack never runs the
  client-credentials grant for the provider (Vouch accepts only a user's token as actor).
- The grant needs a confidential client: `client_secret_env` or
  `client_auth = "private_key_jwt"`.
- A request with no signed-in person (the CLI, the terminal, local mode), or whose person has
  no current identity-provider token, is refused with `Error::Delegation` (HTTP 403, exit 4);
  nothing reaches such a provider as quack itself.
- The acting person travels in a task-local (`llm::acting::Acting`): the server scopes an
  empty slot per request and the identity extractor fills it, the job queue carries the
  submitter's into every job, and MCP's `query` and `search` carry the transport's user,
  since they run in the MCP session's task.
- `quack auth login` refuses the grant. `quack doctor` checks quack's own (actor) token when
  one is sent and, if the issuer lists `grant_types_supported`, that it includes the
  configured grant; with `actor = false` it requests no token.

### 10.3 Crypto

`quack_core::crypto::install_default_provider()` runs at the top of `main`, before any TLS
use. On Linux it installs `rustls::crypto::default_fips_provider()`: aws-lc-rs and rustls
both carry the `fips` feature there, so every distributed Linux binary runs on the
FIPS-validated AWS-LC module and approved cipher suites, and naming that function makes
dropping the feature a build error. macOS and Windows install the aws-lc-sys provider,
because a FIPS build links statically only on Linux. `--version` names the linked module,
and `CryptoModule::log` logs it once a subscriber exists (`docs/crypto.md`).

Data at rest that must stay unreadable without the machine's key goes through
`quack_core::vault`: HPKE (RFC 9180) with DHKEM(P-256, HKDF-SHA256), HKDF-SHA256 and
AES-256-GCM, from rustls's aws-lc-rs HPKE (a FIPS-kept suite), with one key pair per data
directory in the OS keychain or a 0600 `vault.key`. Each value is sealed for a purpose (the
HPKE `info`) and a subject (the associated data) and records its key id. The caller stores
the sealed value where its classification says (signed-in users' tokens: `control.db`,
section 12).

The keychain comes first, then `vault.key`. quack writes `vault.key` only where no keychain
is usable at all (none installed, or no entry addressable, as under Docker's default seccomp
profile). A keychain that exists but refuses access is an error, never a fallback: tokens
sealed under a second key would stop opening whenever the other key answered first. quack
makes the key under an exclusive lock on `<data_dir>/vault.key.lock`, so processes starting
together on one data directory agree on one key; processes on different data directories
share the keychain entry but not the lock.

SHA-256, AES-256-GCM and randomness come from aws-lc-rs; password hashing is the RustCrypto
`argon2` crate, salted from `getrandom`. No runtime code links OpenSSL or `ring`:
`deny.toml` bans `openssl`, `openssl-sys` and `native-tls` outright and allows `ring` only as
a build-time dependency of `libduckdb-sys`, which `make crypto-gates` re-checks with
`cargo tree -e normal`.

---

## 11. Interfaces

### 11.1 Web UI

The web UI replaces today's AnythingLLM screen. Stack, all embedded with `rust-embed`:
askama templates; Tailwind from the standalone binary (no Node.js; the built CSS is
committed, so `cargo build` needs no Tailwind); htmx; ECharts (the full minified build,
vendored; a trimmed custom build needs Node and can replace it later). The chat's permission
step is an "allow the agent to change tables" checkbox on the message, since an HTTP
response cannot ask a question back mid-stream. A refused write tells the user to tick it
and ask again. The UI covers:

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
- Ontology: class, relation, property, and mapping editors with inline validation;
  "Propose" (with cost shown) and "Propose extensions"; the candidate review queue with
  evidence, paged, with bulk accept and reject and a low-support filter; version history
  with diff and restore; JSON import and export.
- Admin: users and the audit log viewer (the skeletal log only; the workspace-side detail is
  API-only). Tokens are managed in workspace settings, not here.

### 11.2 REST API

The API is JSON under `/api/v1`, authenticated by a bearer token (a login session or an API
token) or the `quack_session` cookie. Every answer to a question is the same object print
mode emits:

```json
{
  "answer": "...",
  "citations": [{"n": 1, "document_id": "...", "filename": "Policy-2024.pdf", "page": 12, "heading": "Exclusions", "chunk_id": "...", "chunk_index": 3, "label": "Policy-2024.pdf p.12"}],
  "queries": [{"sql": "...", "rows": 4, "duration_ms": 9}],
  "steps": [{"tool": "run_sql", "summary": "4 rows", "rows": 4, "duration_ms": 9, "detail": "..."}],
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
GET    /api/v1/workspaces/{id}/embeddings          current, stale, and missing vectors against the configured profile, and the plan
POST   /api/v1/workspaces/{id}/embeddings/refresh  200 when current, else 202 with the plan and the job
GET    /api/v1/workspaces/{id}/okf                 the bundle as a tar (import is POST .../documents with a tar)
GET    /api/v1/workspaces/{id}/ontology            current version, JSON
PUT    /api/v1/workspaces/{id}/ontology            import: validate, write a new version
POST   /api/v1/workspaces/{id}/ontology/init       the built-in default as version 1
GET    /api/v1/workspaces/{id}/ontology/versions[?limit=20] | /{v}[?against=N] for a diff
POST   /api/v1/workspaces/{id}/ontology/versions/{v}/restore
POST   /api/v1/workspaces/{id}/ontology/propose    {sample?, auto_accept?, documents?} -> 202 with documents
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
session's lane. Rate limiting is in section 12. Errors:

- An unknown value for a fixed-set field (`mode`, `role`, `scopes`, an audit `outcome`, a
  merge or candidate `action`, an extraction `source`) is refused while the request is
  read: 422 for a JSON body, 400 for a query string, listing the accepted values.
- 404: a missing session, document, ontology version, merge proposal, or candidate.
- 503: a provider that needs `quack auth login`, or a workspace another process holds.
- 400: a question with no chat model configured.

```
GET    /api/v1/workspaces/{id}/jobs               queued, running, and recent jobs, newest first,
                                                  with counts and the worker total (viewer)
GET    /api/v1/workspaces/{id}/jobs/stream        SSE: `jobs` (the list) then `job` per change
GET    /api/v1/workspaces/{id}/jobs/{job}         one job
POST   /api/v1/workspaces/{id}/jobs/{job}/cancel  its submitter, or a workspace owner or admin
```

A question's text shows in a job only to whoever may read its session (its owner, or a
workspace owner or admin). Other members see "a question in a private session".

### 11.3 MCP server

One tool set serves two transports through `rmcp`, the official Rust SDK:

- **stdio:** `quack mcp [-w NAME] [--allow-write]` for Claude Code and editors: one line in
  `.mcp.json`, no server, unaudited like the CLI.
- **Streamable HTTP** (the successor of MCP's HTTP+SSE pair; responses stream as SSE):
  `/mcp/v1/{workspace}` under `quack serve`, with the REST API's bearer. Every request
  passes `Access::resolve`. Each caller gets a transport keyed by workspace, user, and
  write permission (the member role with the write scope). Every tool call is audited with
  channel `mcp`.

Each `query` call starts a session (owned by the server user, `mode` `chat` or `query`)
and returns its `session_id`; passing the id back continues it. A turn that fails before
recording anything leaves no session.

Tools: `query`, `search`, `sql`, `list_tables`, `describe_table`, `list_documents`, and,
once the graph has nodes, `search_graph` and `find_path`. Each answers with structured
content plus text. Refusals (a write without permission, an internal table, a missing
table) are tool errors the client model can read. Resources:
`quack://workspace/tables`, `.../tables/{name}/schema`, `.../documents`, `.../ontology`
(JSON), `.../context` (Markdown).

```json
{ "mcpServers": { "quack": { "command": "quack", "args": ["mcp", "-w", "logistics"] } } }
```

### 11.4 Terminal session (TUI)

The terminal is a Claude Code-style single-pane transcript on a named workspace (`-w`,
resolved through the control plane like every interface). It never blocks its input. It
shows streaming answers, inline steps, citations as footnotes, and ratatui charts;
permission prompts take `y`/`n`/`a`, one at a time, and the prompt itself shows who asks,
the statement, and how many more wait, so clearing or scrolling the transcript never hides
what is being approved. Input starting with `SELECT`/`WITH`/`FROM`/
`DESCRIBE`/`SHOW`/`PIVOT`/`SUMMARIZE` is direct SQL. Direct SQL and `/sql` pass the agent's
gate: internal tables refused, writes ask `y`/`n`/`a`, `max_query_rows` rows shown.

Slash commands: `/help`, `/tables`, `/schema TABLE`, `/sql`, `/ingest PATH` (`/attach`),
`/import`, `/docs`, `/pin`, `/unpin`, `/delete`, `/ontology ...` and `/graph ...`, `/graph
ENTITY`, `/path`, `/context [import FILE | export FILE]`, `/okf DIR`, `/sessions`,
`/resume`, `/new`, `/mode`, `/share`, `/unshare`, `/export [--sql|--markdown] [FILE]`,
`/jobs`, `/cancel N`, `/chart [N]`, `/steps`, `/model`, `/workspace`, `/clear`, `/quit`.
`/ontology` and `/graph` are the `quack ontology` and `quack graph` verbs, parsed by the
same clap definitions; they run in the background, print to the transcript, and answer yes
to anything that would ask on stdin.

**Command parsing.** One clap definition (`terminal::commands::SlashCommand`) drives
dispatch, `/help`, and the completion popup. `SlashCommand::parse` reads the command and
verb words. A command taking free text (a statement, a path, an entity name, a job number)
gets the rest of the line as typed; any other splits its arguments like a shell line
(`shlex`), so `/import URL t --query "SELECT ..."` and `/export 'my file.md'` quote as in a
shell. A typed line is classified once (`commands::Input`): a command, a file to load, a
statement, or a question.

**Completion popup.** A line starting with `/` opens it above the input. It lists matching
commands, then a command's verbs (for `/ontology`, `/graph`, and `/embeddings`, the CLI's
own), fixed choices such as `/mode chat|query`, and long flags once a word starts with `-`.
Up/Down move the highlight; Tab fills it in. Enter fills it in and runs it when nothing more
may follow, or sends the line as typed when there is nothing to fill in. Esc hides the popup
until the next keystroke.

**Jobs** (section 4.1). Questions, statements, files, imports, and ontology or graph verbs
are each a job; the prompt takes the next line at once. A follow-up asked while an answer
streams queues behind it in the session's lane and says so; SQL and file loads run
alongside. A strip above the input shows running and queued jobs (spinner, number, kind,
label, progress); the status line counts them; `/jobs` lists recent ones with outcomes;
`/cancel N` stops one. Results land in the transcript as each job finishes. A turn's text
renders only while its session is on screen; switching sessions leaves it running, and a
line reports its end. Write prompts from concurrent work queue and are answered one at a
time.

**Event loop.** One `tokio::select!` over crossterm's `EventStream`, a single channel every
job and turn reports on, the job queue's broadcast, and a spinner tick that runs only while
a job is active. Every waiting message is applied before the next draw.

**Rendering.** Answers render Markdown (headings, bullets, fences, inline marks). Tool steps
show a three-line preview until `/steps` expands them (print mode folds the same way without
`--verbose`). The pane shows the latest answer's chart; `/chart N` shows an earlier one.
Lines wrap to the terminal width before the scroll range is computed, so the end is always
reachable. Typed input persists in `<data_dir>/terminal_history`. A relative path to an
existing file ingests it. An embedding provider is optional (keyword search without one).

| Key | Action |
|---|---|
| `Enter` / `Shift+Enter` | send / newline |
| `Up`/`Down` | history |
| `PageUp`/`PageDown`, mouse wheel | scroll |
| `Home`/`End` | jump |
| `Esc` or `Ctrl+C` | cancel this session's newest turn, running or queued (recorded with whatever streamed and a cancelled note) |
| `Ctrl+C` with no turn | quit (twice when other jobs still run; they stop with the session) |
| `Ctrl+L` | clear |

The web chat has a Stop button and print mode cancels on `Ctrl+C`. Every interface passes a
cancellation token in its `TurnRequest`.

### 11.5 Print mode and CLI

```
quack -p "PROMPT" [-w NAME] [-f text|json] [--mode chat|query]
      [--allow-write] [-c | -r SESSION] [--stdin] [--verbose]
quack -q "SQL" [-w NAME] [-f table|json|ndjson|csv|markdown] [--stdin]
quack ingest FILE|DIR|- [-w NAME] [--filename N] [--title T] [--pin] [--no-embed]
quack docs [--format json] [--pin ID | --unpin ID | --delete ID]
quack embeddings refresh [-w NAME] [-y]
quack graph search ENTITY [--hops N] [--relation R] [--class C] | search --class C
            | path FROM TO [--max-hops N] | status | extract [--source all|tables|documents]
            [--sample N] [--reset] [-y] | revalidate | review | merges | merge ID.. | reject ID..
quack ontology show | init | propose [--documents] [--from FILE] [--sample N]
              [--auto-accept] [-y] | review [--low-support]
              | accept ID... [--rename N|--merge-into ID|--reparent C] | reject ID...
              | export FILE | import FILE | versions | diff [FROM] [TO] | restore V
quack context show | edit | history | export FILE | import FILE
# Commands that spend model calls ask first ([y/N]) on a terminal; with no
# terminal the answer is no, and -y / --yes goes ahead.
quack sessions [--format json] [--limit N] | export SESSION [--sql|--markdown]
quack import URL --table T (--from SOURCE_TABLE | --query SQL) [--limit N]
quack okf export DIR|-
quack auth login PROVIDER [--device-code] | status [PROVIDER] | logout PROVIDER
quack auth jwks [PROVIDER] [--rotate [--activate]]
quack auth register [--issuer URL] [--device-code | --token-env VAR | --open]
                    [--name NAME] [--replace] [--print] [--yes]
quack auth unregister [--issuer URL] [--yes]
quack config [--changed] [--format json]
quack doctor [-w NAME] [--offline] [--format json]
quack serve [--bind ADDR] [--local]
quack mcp [-w NAME] [--allow-write]
quack user add [--admin] | list [--format json] ; quack token create|list|revoke ;
quack member add|remove|list ; quack audit [filters] [--format text|json|csv|ocsf]   (server admin)
quack --version    version plus the AWS-LC module the binary links; -V is the bare version
```

**Streams.** Non-TTY stdin is data for `-p` and `-q`: CSV, JSON, or Parquet loaded as the
temporary table `stdin` for that invocation (a pasted document is `quack ingest -`). A pipe
silent for a second (a supervisor's inherited stdin) is skipped with a warning; `--stdin`
waits for it. stdout carries the answer or result set; stderr carries steps. In `-f json`
and `-f ndjson`, same-named columns keep every value under suffixed keys (`a`, `a_1`).
Print mode streams text only on a terminal, and reprints the validated answer when
validation changed what streamed; a pipeline gets the validated answer alone.

**Exit codes:** 0 ok, 1 runtime error, 2 usage, 3 write refused, 4 auth required. A reader
that closes stdout early (`| head`) ends the command quietly with 0.

**`quack config`** and `quack doctor` are the only commands that skip `Config::load`.
`config` reads the file itself, so it describes even a configuration every other command
refuses. It prints every recognized setting with its value in force and origin (built in,
the file, or the overriding environment variable); what the file says where that differs
from what runs; the file's unrecognized keys, each with the recognized key it resembles; and
which environment variables the configuration reads are set, never their contents, since
some hold credentials. `--changed` keeps only settings the file or environment sets;
`--format json` emits one document. A rejected file exits 2 after the report.

**`quack doctor`** checks the whole setup, one line per check, with the fix under any that
needs one:

- config file: rejected, or unknown keys with the key each resembles;
- crypto module: a Linux build without FIPS warns;
- data directory: writable; warns when group or others can read it;
- `control.db`: opens and migrates;
- workspace: opens; embedding dimension agrees;
- each configured model: credential present; plain HTTP off this machine with a credential
  warns; one `GET` of the provider's model list proves it is reachable, the key is
  accepted, and the model is pulled or listed;
- `[server]`: a non-loopback bind warns, `local` off loopback fails, no users yet is noted.

With no chat model, it looks for a local Ollama and suggests a `config.toml` snippet with
that Ollama's models. It creates nothing: a missing data directory, control database, or
workspace is reported as missing. `--offline` skips the network; `--format json` emits
`{ok, failures, warnings, checks}`. Any failed check exits 1.

**No model is required.** Without `[general].chat_model`, the terminal opens, runs typed SQL
and every slash command, and answers a question with how to set a model up. `-q`, ingest,
import, and the server's SQL and table pages work unchanged. quack creates a data directory
as `0700` on Unix: it holds every workspace's content, `control.db`, and, where there is no
OS keychain, the vault key file.

### 11.6 Desktop window (`quack desktop`)

Not built (#35, still open): no `desktop` subcommand and no Tauri dependency exist.
It is the last interface on the roadmap and may never be built. The design:

`quack desktop` starts the embedded server on a random loopback port with a per-launch
bearer token and opens the web UI in a Tauri webview with that token. The window is the web
UI plus native file dialogs, drag-drop of files and folders, a system tray, and OS keychain
access for OAuth token keys. Data lives in the platform app-data directory as named
workspaces. quack auto-detects the local Ollama provider.

As a subcommand of the one `quack` binary, it links the Tauri runtime into every build: the
accepted cost of one artifact. If the size or the platform webview dependencies become a
problem for the container image, the fallback is a separate `quack-desktop` crate, not a
Cargo feature. `tauri build` wraps the same binary into `.dmg`, `.msi`, and `.AppImage`
installers.

---

## 12. Server Auth, Roles, and Audit

`docs/authentication.md` is the operator's guide to each way in; this section holds the design.

- `quack serve --local`: no auth, loopback only, one implicit user. For a laptop that wants
  the browser.
- Otherwise, users in `control.db` with argon2id password hashes and a login form that sets
  a session cookie; API tokens (`quack token create` or the admin UI) as bearer tokens
  scoped to a workspace with `read` / `write` / `admin` scopes.
- **Sign-in through the organization's identity provider** (`[server.oidc]`, beside the
  password form).
  - "Sign in with <issuer>" goes to `GET /auth/oidc`: Authorization Code with PKCE, a
    `state`, and a `nonce`. When discovery lists a `pushed_authorization_request_endpoint`,
    quack pushes the request first (PAR, RFC 9126) with the client's credential, so the
    browser carries only a `request_uri`. The code exchange and every renewal authenticate
    as `client_auth` says (section 10.2).
  - A pending sign-in waits in memory ten minutes (at most 10,000 at once). An `HttpOnly`
    state cookie scoped to the callback ties the return to the browser that left, so a
    callback link someone else started cannot sign this browser in.
  - `GET /auth/oidc/callback` exchanges the code and reads the ID token straight from the
    token endpoint over TLS, which OpenID Connect Core 3.1.3.7 lets skip the signature check
    (no JWT library). quack checks `iss`, `aud`, `azp`, `exp`, and the nonce, and finds the
    user by `sub` (`oidc_subject`).
  - A first sign-in creates the user with no password, admin, or memberships (just-in-time
    provisioning: nothing is visible until an owner adds them). The username is
    `preferred_username`, else `email`, else `sub`, suffixed when taken, so a sign-in never
    takes over an account by name.
  - Both outcomes are audited as `login`; both routes share the login rate limit.
- **A sign-in stays tied to the issuer.**
  - The vault (`quack_core::vault`, section 10.3) seals the user's token and refresh token
    in `control.db` (`user_tokens`, one row per user, deleted with the user) for the
    `user-token` purpose with the user id as subject, so a row copied to another user does
    not open. It uses the vault's one key rather than a keychain entry per user: the Linux
    kernel keyring's default per-user quota (200 keys, 20 KB) would cap the server at a few
    dozen users. A row sealed under a key the vault no longer has reads as no token.
  - The session records the token's renewal time; the first request after it renews under
    a per-user lock, reusing a token another session already renewed.
  - A refusal (`invalid_grant`: revoked, expired, account disabled) removes the stored
    token, ends all the user's sessions, and is audited as a denied `session`. An
    unreachable issuer is retried a minute later while the session continues.
  - Without a refresh token, a sign-in lives by quack's own session bounds. Logging out of
    the last session removes the stored token. On Linux the keychain key is in memory, so
    everyone signs in again after a reboot.
  - An on-behalf-of provider exchanges this stored token for the person (section 10.2).
- **quack is an OAuth protected resource** (RFC 9728) when `[server.oidc].audience` is set.
  Without an audience nothing is published, and a JWT is an unknown API token.
  - The API and MCP then also accept the issuer's access tokens as bearers. quack verifies a
    JWT-shaped bearer with `jsonwebtoken` (aws-lc-rs backend) against the issuer's
    `jwks_uri` keys (cached an hour; refetched for an unknown `kid` at most once a minute):
    asymmetric algorithm only, `iss` the issuer, `aud` the configured audience, `exp` with a
    minute's leeway. A token without `scp` or `scope` is refused, since an ID token can
    carry the same `aud`.
  - `subject_claim` (default `sub`) names the user. On Entra, set it to `oid` (Entra's `sub`
    differs per application), set the API's `requestedAccessTokenVersion` (formerly
    `accessTokenAcceptedVersion`) to 2 so tokens carry the v2 issuer, and set `audience` to
    the API's client ID, which is always a v2.0 token's `aud`.
  - The user is found or created with no access, as on sign-in, and the token carries that
    user's own access (`Credential::IdentityProvider`, channel `api`). A refused token is
    audited as a denied `token`.
  - `/.well-known/oauth-protected-resource` describes the server (`resource` is the origin
    of `redirect_uri`), `/.well-known/oauth-protected-resource/mcp/v1/{workspace}` each MCP
    endpoint, and `/.well-known/oauth-protected-resource/api/v1` the API, each naming the
    issuer in `authorization_servers`.
  - Every API or MCP 401 carries `WWW-Authenticate: Bearer resource_metadata="..."` for its
    resource, plus `error="invalid_token"` when a presented credential was refused, so an
    MCP client can find the issuer and sign the user in itself (MCP authorization spec).
- **A browser session is bounded at both ends** (issue #73).
  - It dies `[server].session_max_age_hours` after login regardless of use, or
    `[server].session_idle_minutes` after its last request, whichever is first. The request
    that finds it expired drops it, gets 401 `session expired`, and is audited as a denied
    `session` action. Expiry uses a monotonic clock, so changing the system clock cannot
    extend a session.
  - The cookie is `HttpOnly`, `SameSite=Lax`, with `Max-Age` equal to the absolute lifetime.
    It is `Secure` off loopback, and on loopback too when the public URL is https
    (`[server.oidc].redirect_uri`) or `[server].secure_cookies` is `always`. A cookie minted
    behind a TLS-terminating proxy, even a same-host one on loopback, thus never travels
    over a plaintext downgrade, while plain HTTP on a laptop still works. The sign-in state
    cookie follows the same rule. `X-Forwarded-Proto` is not trusted here (issue #246).
- **Rate limiting covers everything a caller can reach.**
  - One `tower_governor` limiter, keyed by peer address, covers the web UI, REST API, and
    MCP. The password endpoints (`POST /login`, `POST /api/v1/auth/login`) add a tighter
    one, keyed the same way (2 requests per second, bursting to 10); the general budget suits
    browsing and is too loose to make guessing expensive.
  - No limiter keys on what the request says about itself: keyed on the unvalidated
    `Authorization` header, each random bearer got a fresh bucket (issue #237). Everyone
    behind one address (a NAT, a same-host reverse proxy) shares one budget.
    `X-Forwarded-For` is not trusted; any client can write it.
  - `/healthz` is outside every limiter: a throttled health check reads as a dead server.
  - Per-key state is swept once a minute. governor keeps one entry per caller until
    dropped, so an unswept limiter grows by one entry per address ever seen.
- **Nothing a caller receives is cached.** Pages, API answers, downloads, and event streams
  carry workspace content. Every web UI, REST API, and MCP response carries
  `Cache-Control: no-cache, no-store, must-revalidate`, `Expires: 0`, and `Pragma: no-cache`
  unless its handler set a policy, so Back cannot restore a page after logout. Public static
  assets keep `no-cache` with an `ETag`. A new API token appears once, in the creating
  response's body, never in a URL, where history and access logs would keep it.
- **Roles:** `viewer` asks questions and searches. `member` also uploads, pins, deletes own
  uploads, grants write, edits the context and ontology, and runs proposals and extraction.
  `owner` manages members and tokens and sees all sessions. `is_admin` manages users and all
  workspaces but is not thereby a member of any; reading content requires membership.
- **Audit is split at the boundary, and the access half is mandatory.**
  - Every request touching a workspace, allowed or denied, writes a `control.db.audit_log`
    row (section 5.5): who, workspace, resource by opaque id, action, outcome, channel,
    client address, time.
  - The same UUID v7 keys a `_quack_audit` row in the workspace with the content detail (the
    SQL, file names, table name, context diff, proposal accepted). Every allowed action
    writes both rows, listings and page views included (`list`, `page`, `open`). A failed
    audit write fails the request.
  - A denial writes only the `control.db` row; there is no workspace to write detail into.
    Table names are content and never appear in `control.db`.
  - Over MCP the auditor is re-pointed at each request's identity, so a shared transport
    audits the caller, not whoever opened it.
  - An admin sees who accessed what and when in every workspace; a member sees what was done
    inside theirs.
  - Context or ontology export and import, and session export, are audited: they move
    content across the boundary.
  - Logins and failed logins carry no workspace. Membership changes carry the workspace
    changed; a denial for an expired token carries the token's workspace. Successful token
    use is not its own row; the action it performed is.

---

## 13. Configuration

Configuration lives in `~/.config/quack/config.toml` (override with `QUACK_CONFIG_DIR`). It
holds nothing workspace-specific; per-workspace settings live in `_quack_meta`. Every section
sets `deny_unknown_fields`, so a key not listed below is a startup error, not a silent no-op.
`quack config` (section 11.5) shows this list from the binary itself: every recognized
setting with its value in force and origin, and every key in the file that is not one of
them. The TUI's tick rate is a constant, not configuration.

```toml
[general]
data_dir = "~/.local/share/quack"       # QUACK_DATA_DIR
chat_model = "ollama/llama3.1:8b"       # QUACK_MODEL
default_workspace = "default"

[providers.ollama]
type = "ollama"
auth = "none"
base_url = "http://localhost:11434"
# max_concurrent_requests = 1          # model requests in flight at once; default 1 for Ollama, 8 otherwise
# temperature = false                  # send quack's temperature; default true for Ollama only
# effort = "high"                      # this provider's models, over [analysis]; also background_effort

# [providers.ollama.models."qwen3:32b"]  # one model's temperature, effort, background_effort
# temperature = false

[providers.anthropic]
type = "anthropic"
auth = "api-key"
api_key_env = "ANTHROPIC_API_KEY"

[providers.bedrock]
type = "bedrock"                       # auth = "aws" (the default): the AWS SDK's credential chain
# api = "converse"                     # runtime: converse (default), chat-completions, responses
# aws_profile = "my-sso-profile"       # else AWS_PROFILE, else default
# region = "us-east-1"                 # else base_url's, AWS_REGION, or the profile's region


[providers.mantle]
type = "bedrock-mantle"                # api = "responses" (default) or "chat-completions"
# aws_profile = "my-sso-profile"
# base_url = "https://vpce-0123456789abcdef0.bedrock-mantle.us-east-1.vpce.amazonaws.com"  # VPC endpoint without private DNS

[providers.azure]
type = "openai"
auth = "oauth"
base_url = "https://{resource}.openai.azure.com/openai/deployments/{deployment}"
# headers = { "X-Gateway-Team" = "quack" }   # every model request; any type but Bedrock converse

[providers.azure.oauth]
issuer_url = "https://login.microsoftonline.com/{tenant_id}/v2.0"
client_id = "..."
scopes = ["https://cognitiveservices.azure.com/.default", "offline_access"]
redirect_uri = "http://127.0.0.1:19876/callback"
# grant = "authorization-code"                # or "device-code", "client-credentials", "on-behalf-of"
# client_secret_env = "AZURE_CLIENT_SECRET"   # confidential client; client-credentials and on-behalf-of need it
# client_auth = "client_secret_post"          # or "client_secret_basic" (Okta's default), or "private_key_jwt"
# exchange = "entra"                          # on-behalf-of: "token-exchange" (default) or "entra"
# audience = "api://model"                    # on-behalf-of: RFC 8693 audience (Okta, Auth0)
# resource = "https://model.example.com"      # on-behalf-of: RFC 8707 resource
# actor = true                                # on-behalf-of: send quack's own token as actor_token

[embedding]
model = "ollama/nomic-embed-text"        # PROVIDER/MODEL; unset stores documents without vectors
dimension = 768                          # the width of its vectors (1024 for Bedrock's amazon.titan-embed-text-v2:0)
# input prefixes per role; unset keeps the model family's built-in one

# query_prefix = "task: search result | query: "
# document_prefix = "title: {title} | text: "    # {title}: the chunk's heading, or "none"
# similarity_prefix = "task: sentence similarity | query: "

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
# effort = "high"                       # chat turns: none, minimal, low, medium, high, xhigh, max; unset = model default
# background_effort = "low"             # graph extraction and the ontology document pass

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
secure_cookies = "auto"                 # "always": Secure cookies on loopback too (same-host TLS proxy)

[server.oidc]            # optional: "Sign in with <issuer>" beside the password form
issuer_url = "https://login.microsoftonline.com/{tenant_id}/v2.0"
client_id = "..."
redirect_uri = "https://quack.example.com/auth/oidc/callback"   # this server's URL
# client_secret_env = "QUACK_OIDC_SECRET"      # confidential client
# client_auth = "client_secret_post"           # or "client_secret_basic", or "private_key_jwt"
# scopes = ["openid", "profile", "email", "offline_access"]
# audience = "api://quack"                     # accept the issuer's access tokens (RFC 9728); Entra: the API's client ID
# subject_claim = "sub"                        # "oid" for Entra
```

---

## 14. Build and Distribution

- **One binary, `quack`, no Cargo features.** Every surface is a subcommand, and every build
  contains all of them. Static musl on Linux (`x86_64`, `aarch64`), native on macOS
  (`aarch64`) and Windows (`x86_64`, `aarch64`). DuckDB and SQLite are bundled and
  statically linked.
- **Released artifacts are signed and attested.** The macOS binary is signed with a
  Developer ID certificate and submitted for notarization (a bare binary cannot be stapled,
  so the ticket stays with Apple). Azure Artifact Signing signs the Windows binaries, on tag
  builds only, because the federated credential trusts no other ref. Every archive, image
  tarball, and pushed image carries a build provenance attestation naming
  `.github/workflows/reusable-build.yml` as its builder, which makes the provenance SLSA
  Build Level 3 (`docs/ci-cd.md`).
- **No DuckDB extension is ever installed or loaded at runtime.** A static musl binary
  cannot `dlopen`. The only extension features quack enables are `bundled` and `json`
  (Parquet reading is in DuckDB's core). Anything that would need another extension (`vss`,
  `fts`, `excel`, `httpfs`, the Postgres and SQLite scanners) is implemented in Rust or not
  built: XLSX uses a Rust reader, keyword search is quack's own BM25 index, and vector search
  is an exact scan.
- **Desktop bundles** (section 11.6) wrap the same binary; not a separate binary.
- **mimalloc** (`secure`) is the global allocator.
- **Crypto:** rustls + aws-lc-rs, with both `fips` features on Linux, where aws-lc-fips-sys
  links statically, so every distributed Linux binary and the image run on the
  FIPS-validated module (`docs/crypto.md`). `make crypto-gates` runs `cargo tree -i ring`
  and `-i openssl-sys` before the release builds anything.
- **Container image** for `quack serve`:
  - `Dockerfile` builds from source: a CSS stage with the standalone Tailwind binary
    (checksum verified); `rust:<MSRV>-alpine` with cargo-chef for the musl build, `cmake`,
    `clang`, `g++`, and `perl` for DuckDB and aws-lc, and `go` for the FIPS module's
    delocate pass, which needs `AWS_LC_FIPS_SYS_CC=clang`; a `distroless/static` `nonroot`
    runtime with `/quack` and `/data`, `QUACK_DATA_DIR=/data`, `QUACK_CONFIG_DIR=/config`,
    port 8080.
  - `Dockerfile.release` builds the same runtime from the prebuilt musl binaries, so the
    release job never compiles under emulation.
  - The Rust image tag must track `rust-toolchain.toml`.
  - `docker-compose.yml` runs `quack serve` beside Ollama with `deploy/config.toml` mounted
    at `/config`. Air-gapped hosts `docker load` the per-architecture image tarballs from a
    release.
- **Dependencies** follow the workspace rules in `CLAUDE.md`. This design added `argon2`,
  `rmcp` for MCP, `scraper` for HTML, `zip` + `quick-xml` for DOCX and PPTX (no `docx-rs`),
  `rust-stemmers`, and `tower_governor`. No YAML crate: JSON is the only ontology
  interchange form. `tauri` comes only if `quack desktop` is built.

---

## 15. Scaling Constraints and Decision Points

The system being replaced runs on Postgres, so the team should accept these storage
properties explicitly.

1. **DuckDB is embedded and single-process.** One `quack serve` process owns every workspace
   file, so scaling is vertical only; horizontal scaling or an HA (high-availability) pair
   needs storage moved to a server database. For a single instance this is a
   simplification, not a limitation. In-process concurrency is narrower than DuckDB's MVCC
   allows: a workspace has one writer connection (`storage::writer::Writer`, on its own
   thread, interactive work first), so ingestion and session writes take turns. Reads go to
   the reader pool.
2. **Vector search is an exact scan, not an index.** Every query computes cosine distance
   to every stored embedding inside DuckDB. The working set is `chunks × dimension × 4`
   bytes (about 4 GB per million 1,024-wide chunks). Measured with
   `cargo bench -p quack-core --bench retrieval` (`make bench`; synthetic sixty-word chunks
   from a 2,000-word vocabulary, 1,024-dimension vectors, top 8, an in-memory workspace on a
   10-core Apple Silicon laptop, p50 / p99 per query):

   | chunks | vector scan | BM25 leg | fused hybrid |
   |---|---|---|---|
   | 10,000 | 10 / 11 ms | 8 / 8 ms | 18 / 19 ms |
   | 100,000 | 95 / 95 ms | 36 / 37 ms | 134 / 138 ms |
   | 1,000,000 | 165 / 168 ms | 285 / 287 ms | 450 / 456 ms |

   The scan is linear to 100,000 chunks (about 1 µs per chunk) and sublinear past that as
   DuckDB spreads it across cores. Retrieval stays under the model's own time at every
   likely workspace size, so no index is built. Decision, recorded so it is not
   rediscovered (#32): past a million chunks, use a pure-Rust HNSW crate over the same
   stored vectors, persisted beside `data.duckdb` and rebuilt from `_quack_chunks` when
   missing or stale. A custom DuckDB build with `vss` and `fts` statically linked (DuckDB's
   extension config, `DUCKDB_LIB_DIR` and `DUCKDB_STATIC`) means a C++ build pipeline per
   release target and is not pursued.

   BM25 is an indexed join on `_quack_terms`, but its cost grows with the posting lists,
   not the chunk count alone. At a million chunks (sixty million term rows) it is the larger
   half of a hybrid query, so look at the term index before the vector scan at that size.
   All requests on a workspace share one connection, so other requests queue behind these
   latencies while a search runs.
3. **One file is the boundary, so one file is the backup unit.** Back up a workspace by
   copying its directory while the server holds no write transaction. A
   `quack workspace snapshot NAME` that does this through DuckDB's `CHECKPOINT` is still
   unwritten (section 19, step 13). There is no cross-workspace transaction, and none is
   needed.
4. **Storage backend seam: not built.** The intent was small traits over retrieval,
   `graph/`, and `ontology/`, so a Postgres + pgvector backend could be added without
   touching the agent or the interfaces. The code takes `&WorkspaceDb` directly; the only
   traits, `Reranker` and `extraction::Extract`, are not storage seams. Another backend
   today means changing graph and ontology code.
5. **Migration from the current deployment.** Documents are re-uploaded and re-embedded,
   not migrated from pgvector, because chunking and metadata differ. The planned path is
   `quack import anythingllm --url ... --key ...`, pulling workspaces, documents, system
   prompts, and threads through the AnythingLLM API; section 19 lists it after the core is
   stable.

---

## 16. Testing

`cargo test --workspace` runs 308 tests: 209 in `quack-core`'s library, 47 in the binary
(the server router and terminal harness among them), and 52 across two integration files.
`proptest` is on the dependency menu, but no test uses it yet.

**Unit.**

- `storage/`: migrations and CRUD for both databases, dimension mismatch, `_quack_` tables
  hidden from listing and refused to user and agent SQL, statement classification (SELECT
  variants and the `DESCRIBE`/`SHOW`/`SUMMARIZE`/`PIVOT`/`UNPIVOT`/`EXPLAIN` allow-list
  read; DDL, DML, COPY, SET, ATTACH write; a parse error invalid), limits, row capping, RRF
  fusion on fixture rankings.
- `ingestion/`: chunking with heading and page metadata, SHA dedup.
- `analysis/`: citation validation strips unknown markers and renumbers the rest; chart
  spec bounds.
- `ontology/`: inheritance, domain/range validation, property types, mapping validation,
  versioning and stale detection, JSON round trip, table-evidence induction on fixture
  tables (key detection, overlap relation), document-evidence normalization on fixture
  extraction output (clustering, domain/range inference, hierarchy inference, support
  thresholds), candidate actions.
- `graph/`: extraction parsing (valid, malformed, out-of-ontology dropped and counted for
  drift), merge on normalized label, embedding merge proposals, provisional flagging,
  traversal on a fixture with a cycle, path search.
- The agent loop against a mocked rig model with canned tool calls, including a refused
  write and a timeout; mode enforcement excluding provisional graph results; cancellation
  keeping streamed text.
- `llm/`: `TokenManager` reuse, single refresh under concurrency, re-auth, cache round-trip
  against a mock IdP.
- `crypto`: the provider is FIPS exactly on Linux.

**Integration.** The REST, role, audit, and queue suites live in
`crates/quack/src/server/tests.rs`; the two files under `crates/quack-core/tests/` cover
ingestion and the graph.

- Upload PDF -> ready -> a query-mode question returns an answer citing the right page.
- A keyword-only question (a policy number) is answered via FTS.
- CSV -> a question produces SQL referencing a context definition.
- `ontology propose` on a fixture workspace of two tables and ten documents yields the
  expected classes, one overlap relation, and a mapping; accepting them builds nodes with
  provenance. `--auto-accept` marks the graph provisional, and query mode ignores it.
- An ontology edit marks the graph stale, and revalidate drops the invalid edge.
- Write refusal per interface (exit 3, 403, MCP error), and success with permission.
- Session round trip and export, with audit rows in both databases sharing an id.
- A denied open by a non-member and an expired token each write an `audit_log` row with
  `outcome = denied`; no code path updates or deletes `audit_log` rows.
- Workspace isolation via CLI and API; an admin without membership cannot read workspace
  content.
- Server token lifecycle, roles, upload queue.
- Deleting a document takes its chunks, its graph rows, and its files with it.
- A cancelled turn is recorded with what streamed.
- An OKF (Open Knowledge Format) bundle exported through the API imports back as documents
  and candidates.
- An MCP stdio client lists tools and runs `query`.
- REST and print mode return byte-identical JSON for the same question with a mocked model.

**Manual.** Web UI end to end against Ollama, including the proposal review flow; TUI
streaming and permission prompt; OAuth browser and device-code flows against a real tenant;
the air-gapped static binary, which loads no extensions.

**CI and local gates.** On every push and pull request, `.github/workflows/ci.yml` runs
`cargo fmt --check`, clippy with `-D warnings` over all targets and features, and
`cargo test --locked --workspace` on Linux and macOS, plus dependency review and cargo-deny.
Coverage (`make test-coverage`, `cargo llvm-cov`) and mutation testing (`make test-mutants`,
the whole workspace) are local-only and not wired into a release. There is no fuzzing.

**Evaluation.** `make eval` (`crates/quack-core/examples/eval.rs`, issue #74) measures
answer quality. It ingests an in-tree storms-like fixture (`crates/quack-core/eval/`: 27
documents and three CSV tables written for the harness, not the NOAA download) into a
temporary workspace and prints:

- recall@1/5/8 and MRR (mean reciprocal rank) over a 21-question gold set, per question kind
  (`identifier`, `phrase`, `semantic`) and per backend (`search_keyword_chunks`,
  `search_similar_chunks`, `search_hybrid_chunks`);
- precision and recall of `ontology::induction::propose_from_tables` against a hand-written
  expected ontology;
- node and edge precision and recall of `graph::extract::run` over ten hand-labelled chunks
  through a canned `Extract<Extraction>`, so the numbers measure validation, resolution, and
  storage rather than a model;
- how many of a fixed set of recorded answers keep every `[n]` marker through
  `analysis::citations::validate`.

Vector search uses a deterministic hashing embedder (a bag-of-words projection, not a
semantic one), so the run needs no Ollama and takes seconds. With `QUACK_EVAL_OUT` set, it
also writes the numbers as JSON for a before/after diff.

Decoys make the fixture discriminate issue #77's identifier-joining and phrase-filter fix
rather than score 1.000 either way. Five identifier questions (`SR-8841`, `SR-4437`,
`SR-7765`, `SR-2214`, the `AKQ` office code) and two phrase questions (`"flash flood
emergency"`, `"wall of water"`) have decoy documents whose split identifier pieces or
non-adjacent phrase words outscore the true chunk under plain BM25. One phrase question
(`"catastrophic flood damage"`) expects an empty result: every word appears somewhere, the
exact phrase nowhere.

| Measure | Baseline (before #77) | With #104 (`fix/77-identifier-and-phrase-search`) |
|---|---|---|
| keyword recall@1 (MRR) | 0.571 (0.762) | |
| hybrid recall@1 (MRR) | 0.452 (0.605) | |
| keyword identifier recall@1/MRR | 0.444/0.722 | 0.889/0.944 |
| keyword phrase recall@1/MRR | 0.000/0.375 | 0.875/1.000 |
| ontology induction precision / recall | 1.000 / 1.000 | |
| graph extraction precision / recall | 1.000 / 1.000 | |
| citation validity | 8/8 recorded answers validated as expected | |

With #104, the identifier-joined term and the phrase substring filter recover every decoyed
question except the bare `AKQ` code, which has no hyphen for the joined-identifier term to
attach to. Those are the expected numbers once #104 merges. A prompt, chunker, stemmer, or
fusion-constant change is no longer a coin flip: this is the number that moves.

---

## 17. Gaps Between This Document and the Code

This list maps the design to the tracker, ordered by risk, and is updated as issues close.
Every gap is a GitHub issue unless the item says otherwise.

1. ~~Verify against a live model~~ (#20, closed): print mode and the terminal session are
   verified with gpt-oss:20b on Ollama. Sections 7, 8, 9.
2. ~~No OAuth~~ (#25, closed): PKCE and device-code login, encrypted cache, `quack auth`.
   The server's confidential-client mode is wired (`client_secret_env`) and gets its live
   test with #26. Section 10.2.
3. ~~No server, REST API, web UI~~ (#26, closed): `quack serve` with the REST API, password
   and token auth, roles, the split audit, the upload queue, and the askama + htmx web UI
   (section 11.1). The graph and ontology pages arrived with #27 and #28. ~~No MCP~~ (#29,
   closed: `quack mcp` on stdio and `/mcp/v1/{workspace}` over streamable HTTP, section
   11.3); **no desktop window** (#35). Sections 11, 12.
4. ~~No ontology or induction~~ (#27, closed): the model, validation, versions with diff and
   restore, the built-in default, JSON import and export, table and document evidence into
   the review queue (accept, rename, merge, reparent, reject, auto-accept, proposals limited
   to what the current ontology lacks, `--from` seeding, low-support candidates), and the
   CLI, API, and web page. The separate "full" and "extend" modes were merged (#123): a full
   mode that ignored the current ontology only re-queued accepted items and had never taken
   effect. Two deviations from 6.5: cluster names are chosen by frequency, not a
   model naming pass; and drift counting arrives with constrained extraction in #28. YAML
   was dropped: JSON is the only interchange form.
   ~~No graph~~ (#28, closed): `quack_core::graph` with deterministic extraction from
   mapped tables; constrained extraction from chunks through the chat model
   (out-of-ontology classes and relations counted as drift); exact merge on normalized
   label and class plus embedding-based merge proposals; provenance on every node and edge;
   neighborhood, path, and by-class traversal in plain SQL; the `search_graph` and
   `find_path` tools (registered only when the graph has nodes; query mode drops provisional
   results); `quack graph`, `/graph` and `/path` in the terminal, the REST endpoints, the
   MCP tools, and the web page with an ECharts force graph, inspector, merge queue, and
   provisional and stale banners. Three deviations from 6.4:
   - Merge proposals live in `_quack_graph_merges`. Accepting one merges nodes, which is not
     an ontology change, so they stay out of the ontology candidate queue.
   - "Provisional" means the newest ontology version was written by `--auto-accept` and
     nobody has saved a reviewed version since.
   - "Graph enabled" means the graph has nodes, not a separate `_quack_meta` flag.

   Sections 6.3 to 6.5.
5. ~~DOCX, HTML, PPTX, XLSX unsupported~~ (#16, closed): HTML through `scraper`, DOCX and
   PPTX through `zip` + `quick-xml`, workbooks through `calamine` as one table per sheet,
   each format carrying its own title. Sections 6.1, 6.2.
6. ~~No `ATTACH` to external databases~~ (#21, closed): `quack import`, the Rust-side
   replacement, snapshots a Postgres or SQLite query or a data file over HTTP(S) into a
   workspace table through the CSV path (`quack_core::import`). A live `ATTACH` (queries
   pushed to the source) is not offered: the scanner extensions cannot ship in the static
   binary, and a snapshot keeps the classification boundary simple, since the rows live in
   the workspace file like any upload. MySQL and S3 are the next sources. Section 6.2,
   step 13.
7. ~~Document registry lacks `sha256` dedup, `source`, `title`~~ (#22, closed): identical
   bytes are skipped everywhere and name the existing document; `source` is `upload`,
   `paste`, `path`, or `stdin`; the title is given or parsed from the first heading;
   `chunk_count` and `ingested_by` are recorded. Section 5.4.
8. ~~Sessions have no `created_by` or sharing; print mode cannot take stdin as data~~ (#23,
   closed): `created_by` exists since the server landed. The creator or an owner sets
   `shared` (`PATCH .../sessions/{sid}`, the chat page's toggle), audited as `share`, which
   opens the session to every member. Piped stdin is the temporary table `stdin` in `-p` and
   `-q`. Sections 8, 11.5.
9. ~~Context `edited_by` and the `_quack_audit` detail table~~ (#24, closed): the server
   records the editing user and writes the detail row under the access row's id. Sections
   5.3, 5.4, 12.
10. ~~No release pipeline~~ (#30, closed): `.github/workflows/release.yml` runs only on a
    `v*` tag (or by hand). It runs the gates (fmt, clippy, tests, `make crypto-gates` for
    the ring and OpenSSL runtime-tree checks, cargo deny), then
    `.github/workflows/reusable-build.yml` for everything that compiles, signs, or attests:
    - reproducible static musl binaries for x86_64 and aarch64 with CycloneDX SBOMs through
      `Dockerfile.build` and `docker-bake.hcl`;
    - native macOS arm64 and Windows x86_64 and aarch64 binaries, each on its own native
      runner, code-signed (Apple Developer ID with notarization, Azure Trusted Signing) when
      the secrets exist;
    - the `quack serve` image for amd64 and arm64 on GHCR, built per architecture from the
      prebuilt binaries (`Dockerfile.release`), plus per-architecture image tarballs for
      air-gapped hosts.

    Every artifact carries a build provenance attestation signed under the build workflow's
    identity (SLSA Build Level 3). No building job holds `contents: write`; a separate
    `publish` job writes `SHA256SUMS` and creates the GitHub release. `Dockerfile` builds
    the same image from source for `make image` and `docker-compose.yml`. Section 14.
11. ~~No stemming in keyword search~~ (#31, closed: Snowball English over the same
    tokenizer; schema version 6 rebuilds older term indexes on open); ~~no reranking hook~~
    (#34, closed: `Reranker` trait, `none` or `model`); ~~large-workspace vector index
    options~~ (#32, closed as a recorded decision in section 15, item 2). Sections 6.1, 15.
12. ~~Web UI mapping of the chart spec to ECharts~~ (#26, closed): `static/js/app.js` maps
    the spec to an ECharts option. Section 9.
13. ~~Open Knowledge Format bundles~~ (#36, closed as a deliberately one-way export that
    restores what it can), in `quack_core::okf`.
    - **Export:** `quack okf export DIR` (or `-` for a tar on stdout) and
      `GET /api/v1/workspaces/{id}/okf` (a tar, audited as `export`) write `index.md` from
      the context; one Markdown file with YAML front matter per table (schema, sample rows,
      mapping links), class, relation, property, document, and graph node (properties,
      links per edge, provenance per chunk and row); and `log.md` from the ontology versions;
      the audit detail stays in the workspace.
    - **Import:** `quack ingest DIR` on a bundle and `POST .../documents` with an
      `application/x-tar` body ingest every concept file as a Markdown document. They turn
      front-matter types into class candidates; links between typed concepts into relation
      candidates (the relation named on a `- <relation>: [..](..)` line, else
      `<source>_links_<target>`; nothing when the ontology already relates the two classes
      or their ancestors); and `resource` into a document property candidate, all in the
      ontology review queue. The CLI offers `index.md` as the workspace context; the API
      returns it as `context`.
14. **No `quack workspace snapshot`**, the only gap here with no issue of its own. Section 15
    item 3 names it as the supported online backup; there is no `workspace` subcommand and
    nothing calls `CHECKPOINT`. Today's backup is the directory copy section 15 item 3
    describes. Sections 15, 19.
15. **Work queues, first pass** (section 4.1).
    - Done: the terminal, the web chat, REST `query`, uploads, graph extraction, and the
      document pass run on `quack_core::jobs`. `llm::LimitedHttp` limits model requests per
      provider and model, interactive first. Ingest and import stop on cancel, mid-embedding
      included. A workspace with 64 uploads waiting answers the next with 503 and
      `Retry-After: 30`. The web Jobs page follows `.../jobs/stream` instead of polling; the
      terminal re-renders only changed messages.
    - Done: the writer is an actor (section 4.1), so no runtime worker blocks on the
      database. Ingestion, import, extraction, and the CLI commands take the `Writer` (the
      CLI spawns one too); parsing runs on the blocking pool; every job, the terminal's
      included, is a plain runtime task. The terminal's commands run their database step in
      typed order on a worker task, reads on the reader pool.
    - Not yet: MCP `query` calls and print mode run their turn directly (one call, one
      answer, nothing to keep responsive); the web chat page shows its own turn but not a
      job strip (the Jobs page does); jobs are not persisted across restarts.

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
  candidates with a review queue, repeat proposals driven by drift, domain-pack seeding,
  auto-accept with provisional marking
- Graph: ontology-guided extraction from documents, deterministic extraction from mapped
  tables, entity resolution with review queue, provenance, neighborhood / path / by-class
- Agent: tools, permissions per interface and role, limits, visibility, citation validation
- Workspace context stored and versioned inside the boundary, Markdown import and export
- Sessions with resume, sharing, export
- Charts: one spec, rendered everywhere
- Providers: ollama, openai (and compatible), anthropic, bedrock, bedrock-mantle; auth none,
  api-key, OAuth PKCE with device code, encrypted cache, and confidential-client mode for
  the server, or the AWS SDK's credential chain (profiles, SSO, instance roles) for Bedrock
- Interfaces, all in one binary: web UI, REST API, MCP (stdio and streamable HTTP), TUI,
  print mode, and `quack desktop` (last, if ever)
- Open Knowledge Format bundles: `quack okf export` writes one, `quack ingest DIR` and a tar
  upload read one back as documents and ontology candidates
- Server: users with password login or sign-in through the organization's OpenID Connect
  issuer, tokens with scopes, roles, audit, upload queue
- Static builds, container image and compose, desktop bundles

### Deferred, in rough priority order

1. AnythingLLM import command (workspaces, documents, system prompts, threads via its API)
2. Data connectors: GitHub, Confluence, SharePoint (fetching a data file over http(s)
   already ships in `quack import`)
3. Cross-encoder reranking provider
4. OCR for scanned PDFs
5. Postgres + pgvector storage backend, which now also means building the seam section 15
   item 4 describes
6. Ontology import from OWL / SKOS; a registry of domain packs
7. Web search tool for the agent
8. OpenAI-compatible `/v1/chat/completions` endpoint
9. In-process embedding models (ONNX) to drop the Ollama requirement offline
10. DuckPGQ for graph queries
11. Kubernetes manifests; WebSocket MCP transport; object-storage file backend
12. Web UI localization via fluent (pattern already documented in `docs/web-ui.md`)

---

## 19. Implementation Order

Each step leaves the tool working. Step 8 ends the first milestone that can replace the
current deployment for document chat.

1. ~~Limits and classification, real `search_documents`, crypto install.~~ Done.
2. ~~Merge binaries into `quack`; `quack-core::llm`; strict config with
   `chat_model`/`embedding_model`/`auth`.~~ Done.
3. ~~Storage consolidation: `_quack_` prefix, `_quack_meta`, dimension check,
   `control.db` reduced to access control, sessions and messages with `--continue`,
   `--resume`, `quack sessions`, and `export`.~~ Done. Context landed with step 6, audit
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
    repeat proposals and drift counting, auto-accept with provisional marking; CLI, API, and
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
