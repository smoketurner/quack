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
  agents, over stdio or SSE.
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
|  web UI (askama+htmx) | REST | MCP (stdio, SSE) | TUI | print | desktop window |
+------------------------------------------------------------------------------+
                                       |  in-process calls; no internal network hop
+------------------------------------------------------------------------------+
|                          quack-core (library crate)                           |
|                                                                               |
|  workspace/    open/create, layout, .quack/ discovery for the TUI             |
|  storage/      DuckDB per workspace (everything classified); control.db       |
|  ingestion/    parsers, chunking with metadata, embedding, hybrid index        |
|  retrieval/    vector + FTS fusion, reranking hook, citations, pinned docs     |
|  analytics/    SQL execution, classification, limits, schema introspection    |
|  ontology/     model in tables, induction (propose), validation, versions     |
|  graph/        extraction guided by ontology, entity resolution, traversal    |
|  agent/        loop as an event stream, tools, permissions, prompt, modes     |
|  llm/          rig providers; auth none / api-key / OAuth PKCE                |
|  context/      workspace context (stored), import/export as Markdown          |
|  config.rs, error.rs                                                          |
+------------------------------------------------------------------------------+
                                       |
+------------------------------------------------------------------------------+
|  DuckDB (bundled, static, core + json + parquet only; no runtime extensions)  |
|  vectors: exact cosine scan; keywords: quack's own BM25 term index            |
|  SQLite (sqlx, bundled): control.db (server access control only)             |
+------------------------------------------------------------------------------+
```

Crates:

```
crates/
  quack-core/      the engine
  quack/           the one binary: `quack serve`, `quack mcp`, `quack desktop`,
                   terminal session, print mode, admin
    src/
      main.rs          clap surface, crypto provider install
      terminal/        ratatui session
      print.rs         one-shot mode and output formats
      serve/           axum router, REST, SSE, MCP SSE transport, templates, embedded assets
      mcp_stdio.rs
      desktop.rs       `quack desktop`: embedded server + Tauri window (section 11.6)
```

Two crates, no Cargo features. Surfaces are subcommands, not build variants.

Every interface calls the same core entry points:

| Operation | Core | Web | REST | MCP | TUI / print |
|-----------|------|-----|------|-----|-------------|
| Ask | `agent::run_turn` (event stream) | SSE fragments | SSE or JSON | `query` tool | inline / stdout+stderr |
| Retrieve | `retrieval::search` | via agent, `/search` page | `GET .../search` | `search` tool | via agent |
| SQL | `analytics::execute` | SQL page | `POST .../sql` | `sql` tool | `/sql`, `-q` |
| Ingest | `ingestion::ingest` | upload | `POST .../documents` | - | `/ingest`, `quack ingest` |
| Graph | `graph::neighborhood`, `graph::path` | graph page | `GET .../graph/*` | `search_graph` | `/graph`, `quack graph` |
| Ontology | `ontology::{get,propose,accept,validate}` | ontology page | `.../ontology/*` | resource | `quack ontology` |
| Context | `context::{get,put}` | settings page | `.../context` | resource | `/context` |
| Permission | `agent::Permission` | confirm dialog | 403 + reason | tool error | y/n prompt / exit 3 |

The LLM layer is `rig`. `quack-core::llm` builds rig clients from config and exposes
`ChatModel` and `EmbedModel` enums so the rest of the core is provider-agnostic.

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

Named workspaces live under `<data_dir>/workspaces/<name>/` and are what the server, the
desktop app, and MCP serve. The TUI additionally treats any directory containing a
`.quack/` folder (same layout inside it) as a workspace, found by walking up from the
current directory like `git` finds `.git/`, so a developer can run `quack` inside a folder
of files. A `.quack/` workspace is unclassified by definition because it sits in a user's
working tree; importing one onto the server requires an owner to assign its label.

### 5.2 Isolation

One DuckDB file per workspace; queries cannot cross workspaces. External sources are
attached explicitly per session and never persisted. A workspace's `allowed_providers`
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

Edited in the web UI settings page (`member`+, audited), via `PUT .../context`, or with
`quack context edit` (opens `$EDITOR` or `$VISUAL` on a temp file and stores the result).
`quack context export|import FILE` moves it as Markdown (`-` for stdout or stdin);
`quack context history` lists versions; `/context` in the terminal shows it. A
`~/.config/quack/context.md` is an unclassified global prefix loaded before the workspace
context. The agent never writes it. Every distinct edit is a new version in
`_quack_context`; importing identical content records nothing.

### 5.4 `data.duckdb`

Internal tables are prefixed `_quack_` and hidden from the agent's table listing and from
user SQL by default (`quack -q` and the SQL page can opt in with `--internal`). IDs are
UUID v7 via `uuid::Uuid::now_v7()`.

```sql
-- workspace metadata
CREATE TABLE _quack_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
  -- schema_version, embedding_model, embedding_dimension, ontology_version,
  -- graph_built_with_ontology_version, default_mode, graph_enabled

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
    sha256        TEXT NOT NULL,           -- dedup on re-upload
    source        TEXT NOT NULL,           -- upload | paste | path | stdin
    status        TEXT NOT NULL DEFAULT 'pending',   -- pending | processing | ready | error
    error_message TEXT,
    pinned        BOOLEAN NOT NULL DEFAULT false,
    chunk_count   INTEGER,
    ingested_by   TEXT,
    ingested_at   TIMESTAMP DEFAULT now()
);

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
    term     TEXT NOT NULL,              -- lowercased alphanumeric run from content + heading
    tf       INTEGER NOT NULL
);
CREATE INDEX _quack_terms_term_idx ON _quack_terms (term);

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
    weight             FLOAT DEFAULT 1.0,
    properties         JSON,
    provisional        BOOLEAN NOT NULL DEFAULT false
);
CREATE TABLE _quack_provenance (            -- every node and edge traces to text or a row
    subject_id  TEXT NOT NULL,              -- node or edge id
    document_id TEXT,
    chunk_id    TEXT,
    table_name  TEXT,
    row_key     TEXT,
    confidence  FLOAT,
    PRIMARY KEY (subject_id, chunk_id, table_name, row_key)
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
provider's is an error with a clear message, never a silent mismatch. Session and message
writes are small and frequent; DuckDB's MVCC handles them alongside ingestion writes in
one process without a separate database.

### 5.5 `control.db` (server only, SQLite, sea-query)

At `<data_dir>/control.db`, opened by `quack serve`, the desktop app, and the admin
subcommands. It answers one question, who may open which workspace, and holds nothing that
reveals what a workspace contains.

```sql
CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);

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
                                   -- | session_read | member | token | workspace | admin
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
which lives in `_quack_audit`. Retention is configurable (`[server].audit_retention_days`,
default unlimited) and pruning is itself an audited admin action. `quack audit` and
`GET /api/v1/admin/audit` filter by user, workspace, action, outcome, and time range and
export as CSV or NDJSON.

Workspace names themselves are treated as unclassified; if a deployment needs opaque
names, the `name` column is the directory name and a display name lives in `_quack_meta`.

### 5.6 SQL construction rules

- `control.db`: sea-query only.
- `data.duckdb` internal statements: parameterized via `duckdb::params!`. Identifiers that
  must be interpolated (table names, `FLOAT[N]`) go through one `quote_ident` /
  validated-integer helper. File paths are bound as parameters to `read_csv_auto(?)`. The
  graph traversal CTE is a constant string with bound parameters.
- Agent-generated and user-typed SQL: executed as-is through the permission layer
  (section 7.4). Never assembled by application code.

---

## 6. The Knowledge Engine

### 6.1 Documents and vectorization

**Sources.** File upload (web, REST, desktop), paste (web: a text box that becomes a
document), path (TUI, CLI), stdin (print mode). URL, GitHub, and Confluence connectors are
deferred (section 18).

**Parsing.**

| Type | Parser | Extracted metadata |
|------|--------|--------------------|
| PDF | `pdf-extract` | page numbers |
| Markdown, plain text | direct | headings (ATX and setext) |
| HTML | `scraper` or `html2text` | headings, title |
| DOCX, PPTX | `docx-rs` / zip + XML | headings (from styles) |
| CSV, Parquet, JSON, JSONL, XLSX | DuckDB (section 6.2) | become tables, not chunks |

Scanned PDFs (no text layer) are detected and reported as `error: no extractable text`;
OCR is deferred.

**Chunking.** Paragraph boundaries; 512-token target; 64-token overlap; the nearest
preceding heading is stored on the chunk and prepended to its embedding input; page numbers
recorded where the source has them. Token counts via `tiktoken` (`cl100k_base`).

**Embedding.** Batches of 64 through the configured embedding provider. The HNSW index is
built after a document's chunks are stored; the FTS index is rebuilt incrementally.
Re-uploading a file with the same SHA-256 is a no-op with a message.

**Hybrid retrieval.** A query runs both an exact cosine scan over `embedding` (core
`array_cosine_distance`) and a BM25 search over the terms quack tokenized at ingest
(`_quack_terms`, scored in SQL; no DuckDB extension). Results are fused with reciprocal rank fusion
(`k = 60`) and the top `k` chunks (default 8) are returned. A reranking hook accepts an
optional cross-encoder provider later; it is a no-op at MVP. This is the main retrieval
quality improvement over the pgvector setup, where keyword-exact questions (part numbers,
policy IDs) go unanswered.

**Citations.** Every retrieved chunk carries `document_id`, `filename`, `title`, `page`,
`heading`, and its fused score. The agent cites by `[n]` markers that map to these chunks;
the core validates that every marker references a chunk retrieved in that turn and strips
those that do not. Interfaces render citations as links to the document and page.

**Pinned documents.** A pinned document's full text is injected into the system prompt
each turn (subject to the history token budget) rather than retrieved.

### 6.2 Tables and analytics

| Type | Mechanism | Result |
|------|-----------|--------|
| CSV, TSV, Parquet, JSON, JSONL | `read_csv_auto` / `read_parquet` / `read_json_auto` | Table in `data.duckdb` (server, desktop); view over the file in place (TUI in a `.quack/` directory) |
| Excel `.xlsx` | `excel` extension, `read_xlsx` | One table per sheet |
| stdin (print mode) | sniffed | Temporary table `stdin` |
| Postgres, SQLite, S3/HTTP Parquet | Deferred: the scanner and httpfs extensions cannot be compiled into the static binary (section 15); a Rust-side importer is the candidate design | — |

Table naming: sanitized file stem; on collision the web UI and TUI ask (replace, rename,
skip), the API and print mode require an explicit name. `DESCRIBE`, row count, and three
sample rows per table are cached per session for the prompt.

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
2. *Validation.* Nodes and edges are validated on insert: class exists, relation exists,
   domain and range hold under inheritance, property types check, enum values match.
3. *Query expansion.* `search_graph(class: organization)` matches `vendor` too. The agent's
   prompt includes a compact rendering of the ontology so it can ask class-aware questions.
4. *Table mapping.* A mapping turns a table's rows into nodes and its foreign-key-like
   columns into edges without an LLM. This is the bridge between the analytics and graph
   substrates.

**Versioning.** Every accepted change writes a new row to `_quack_ontology_versions` with a
full snapshot. `_quack_meta.graph_built_with_ontology_version` records what the graph was
built with; when it lags, the graph is stale and the interfaces offer re-extract (cost
shown first) or revalidate (fast; drops nodes and edges that no longer validate). Any
version can be diffed against another or restored.

### 6.4 Knowledge graph

**Extraction from documents.** Runs after embedding when the workspace has
`graph_enabled`, or on demand. Each chunk goes to the chat model with the ontology-derived
prompt and must return JSON `{nodes: [{label, class, properties}], edges: [{source, target,
relation, properties}]}`. Responses are parsed strictly; a failed chunk is logged and
skipped, never retried in a loop. Cost (chunk count, model) is shown before extraction
starts and the operation is permission-gated.

**Extraction from tables.** Ontology mappings turn rows into nodes and edges
deterministically, with provenance `table_name` and `row_key`. Re-running is idempotent.

**Entity resolution.** Nodes are merged on `(normalized_label, class_id)`. A second pass
proposes merges for nodes of the same class whose label embeddings are within a cosine
threshold (default 0.08) and whose labels share a token; proposals above a confidence
threshold merge automatically, others land in `_quack_ontology_candidates` with
`kind = 'merge'` for review. Aliases are kept in `properties.aliases`.

**Provenance.** Every node and edge has at least one `_quack_provenance` row. Answers from
the graph cite the source chunk or row the same way document answers cite chunks.

**Traversal.** Plain SQL, constant strings with bound parameters, no DuckPGQ:

```sql
WITH RECURSIVE hops AS (
    SELECT id, label, class_id, 0 AS depth, [id] AS path
    FROM _quack_graph_nodes WHERE id = ?
    UNION ALL
    SELECT n.id, n.label, n.class_id, h.depth + 1, list_append(h.path, n.id)
    FROM hops h
    JOIN _quack_graph_edges e ON h.id = e.source_node_id OR h.id = e.target_node_id
    JOIN _quack_graph_nodes n
      ON n.id = CASE WHEN e.source_node_id = h.id THEN e.target_node_id ELSE e.source_node_id END
    WHERE h.depth < ? AND NOT list_contains(h.path, n.id)
      AND (? IS NULL OR e.relation_id = ?)
)
SELECT DISTINCT id, label, class_id, depth FROM hops ORDER BY depth, label LIMIT ?;
```

Entry point resolution: exact `normalized_label` first, then embedding similarity over
node embeddings, then class filter. Operations: `neighborhood(entity, hops, relation?)`,
`path(a, b, max_hops)` (bidirectional BFS in SQL), `by_class(class, limit)` with subclass
expansion. Limits: `max_traversal_depth` (3) and `max_nodes` (200).

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
  proposes a relation between the two classes, named from the column (`policy_id` ->
  `has_policy`, refined by the model in the naming pass).
- The result is also proposed as a mapping, so accepting it makes rows into nodes at once.

**Evidence from documents (open extraction on a sample).**

1. Take a stratified sample of chunks across documents (default 200, configurable), so
   every document contributes.
2. Run extraction unconstrained: free-form entity types, free-form relation names,
   observed attributes with values.
3. Normalize the vocabulary. Types and relation names are embedded and clustered; each
   cluster is sent to the model once with its members and examples to choose a canonical
   `snake_case` id, a label, and a one-line description.
4. Infer structure. A relation's domain and range are the classes observed at its endpoints
   (generalized to the nearest common ancestor when mixed). Hierarchy is inferred where one
   type's mentions are consistently also labeled with a broader type (`vendor` under
   `organization`). Attributes that recur on a class propose typed properties.
5. Score. Each candidate carries occurrence count, distinct-document count, three example
   mentions with chunk ids, and a confidence from support and cluster tightness.
   Candidates below the support threshold (default 3 documents) are kept as
   `low_support` rather than shown in the main proposal.

**Cost.** Shown before the run: sample size, model, and an estimate of calls (sample
chunks plus one per cluster; roughly 250 requests for a 200-chunk sample). The run is
permission-gated and audited.

**Review.** The web ontology page, `quack ontology review`, and `GET .../ontology/candidates`
show the proposal grouped by kind with evidence inline. Actions per candidate: accept,
rename, merge into an existing class or relation, reparent, reject. Accepting writes a new
ontology version. `PUT .../ontology/candidates/{id}` and `quack ontology accept ID...`
apply the same actions from scripts.

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
   c. execute; emit ToolFinished { rows | chunks | nodes, duration_ms }
   d. append the result to history; go to 2
4. On text: emit TextDelta as it streams; validate citations; persist the turn;
   emit TurnComplete { citations, chart, queries }
```

`agent::run_turn` yields these events on a channel. The web UI turns them into HTML
fragments over SSE, REST forwards them as typed SSE events or collects them into one JSON
response, MCP collects them into the tool result, the TUI renders them inline, print mode
writes them to stderr. `max_turns` 10, temperature 0.1.

### 7.2 System prompt

1. Role and behavior for the mode (7.5): retrieve before answering, cite with `[n]`, run
   SQL rather than estimate, state assumptions, ask one clarifying question when the
   request is ambiguous.
2. Tool guidance and DuckDB dialect notes.
3. Tables block: user-facing tables and views with columns, types, row count, three sample
   rows.
4. Documents block: count, and titles of pinned documents with their full text.
5. Ontology block: classes with parents, relations with domain and range (compact), node
   and edge counts, and whether the graph is provisional or stale. Only when the graph is
   enabled.
6. Global context prefix, then the workspace context.
7. The permission rules.

The prompt names only tools that are registered for this workspace and mode.

### 7.3 Tools

| Tool | Permission | Description |
|------|------------|-------------|
| `search_documents(query, top_k=8, document_ids?)` | none | Hybrid retrieval; returns chunks with citation metadata |
| `list_documents()` | none | Registry with status and pinned flag |
| `run_sql(sql)` | read: none; write: prompt | Execute SQL; result capped at `max_query_rows` with a trailer |
| `describe_table(name)` / `list_tables()` | none | Schema and inventory |
| `search_graph(entity?, class?, relation?, hops=2)` | none | Neighborhood or class listing with provenance |
| `find_path(from, to, max_hops=4)` | none | Shortest relation path between two entities |
| `create_chart(sql, kind, x, y, title)` | none | Runs the SQL, emits a chart spec (section 9) |
| `export(sql, path, format)` | prompt | `COPY ... TO` into the workspace `files/` |

Graph tools register only when the workspace has `graph_enabled`; SQL tools only when it
has at least one table; `search_documents` only when it has at least one ready document.

### 7.4 Permissions and limits

**Classification.** Before executing `run_sql`, the statement is passed to DuckDB's own
parser via `SELECT json_serialize_sql(?)`. DuckDB serializes `SELECT` statements (CTEs,
`FROM`-first, `PIVOT`) and errors on everything else. Serializes means read; anything else
means write. `COPY`, `INSTALL`, `LOAD`, `ATTACH`, `SET` are always write. Statements that
reference `_quack_` tables are refused for the agent regardless.

**Decision by interface and role.**

| Interface | Read | Write |
|-----------|------|-------|
| TUI | run | prompt `Run this statement? [y/N/always]` showing the SQL |
| Print mode | run | refuse, exit 3, unless `--allow-write` |
| Web / REST, `viewer` | run | 403 |
| Web / REST, `member`+ | run | 403 unless `allow_write: true`; the web UI shows a confirm dialog and re-sends |
| MCP | run | tool error unless the token has the `write` scope |
| Desktop | run | native confirm dialog |

**Limits.** The agent's connection runs with `SET memory_limit` and `SET threads` from
config. Queries execute on a dedicated thread; the caller takes
`Connection::interrupt_handle()` first and a timer calls `interrupt()` after
`query_timeout_seconds`. User SQL has the same limits.

### 7.5 Chat modes

Per session, defaulting from the workspace's `_quack_meta.default_mode`:

- **chat** - the agent may answer from general knowledge as well as retrieved sources; it
  must still cite when it used a source.
- **query** - the agent must ground every claim in a retrieved chunk, a query result, or a
  graph result; if retrieval returns nothing relevant it says so instead of answering.
  This is AnythingLLM's query mode and the default for classified workspaces. Provisional
  graph results are excluded in this mode.

### 7.6 Visibility

Every tool call renders as one line at start and one at finish, in every interface:

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

Every turn belongs to a session in `_quack_sessions` (AnythingLLM's threads). Messages are
appended as they happen with full tool metadata, so a session is a complete record and
stays inside the classification boundary.

- Resume: `session_id` on the REST request, `--continue` / `--resume` in the TUI, the
  thread list in the web UI.
- Export: `.sql` (every executed statement with the question as a comment) or Markdown
  (questions, steps, tables, citations, answers), from every interface. Export is an
  audited action because it moves content across the boundary.
- History sent to the model is trimmed to `history_token_budget` (32,000); older tool
  payloads are replaced with one-line summaries.
- Server mode: sessions carry `created_by`; members see their own and any marked `shared`.
  `owner` can see all sessions in the workspace for audit.

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

One x axis, one or more numeric series, at most 200 points per series. A chart attaches to
the assistant message that produced it and appears at that point in every rendering.

---

## 10. LLM Layer

### 10.1 Providers

| `type` | Chat | Embeddings | Notes |
|--------|------|------------|-------|
| `ollama` | yes | yes | Offline default. `base_url` defaults to `http://localhost:11434` |
| `openai` | yes | yes | Also OpenAI-compatible endpoints via `base_url` (vLLM, LiteLLM, Azure OpenAI) |
| `anthropic` | yes | no | Native Messages API with tool use |

`[general].chat_model` and `[general].embedding_model` name `PROVIDER/MODEL` each; a
workspace's `allowed_providers` filters the choice; `--model` overrides per run in the TUI.
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
}

pub struct TokenManager {
    config: OAuthConfig,
    cache_path: PathBuf,             // <data_dir>/tokens/<provider>.json, mode 0600
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

`aws_lc_rs::default_provider().install_default()` at the top of every `main`. SHA-256,
argon2id inputs, and randomness come from aws-lc-rs. `ring` and OpenSSL never appear in
the tree.

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

- Workspace list and switcher; workspace settings (context editor with version history,
  mode default, allowed providers, members).
- Chat: thread list, streaming answer with a collapsible steps block, citations as links
  that open the document at the page, charts, permission confirm dialog, mode toggle.
- Documents: upload (drag-drop, multi-file), paste text, status with progress, pin,
  delete, re-embed.
- Tables: list with schema and sample rows; a SQL page with result grid and download.
- Graph: search box, ECharts graph with class colors, node inspector with properties and
  provenance, merge review queue, provisional and stale banners.
- Ontology: class, relation, property, and mapping editors with validation inline;
  "Propose" (with cost shown) and "Propose extensions"; the candidate review queue with
  evidence; version history with diff and restore; import and export as YAML.
- Admin: users, tokens, audit log viewer (the skeletal log; detail opens inside the
  workspace for members).

### 11.2 REST API

JSON, bearer token, versioned under `/api/v1`. Every response for a question is the same
object print mode emits:

```json
{
  "answer": "...",
  "citations": [{"n": 1, "document_id": "...", "filename": "Policy-2024.pdf", "page": 12, "heading": "Exclusions", "chunk_id": "..."}],
  "queries": [{"sql": "...", "rows": 4, "duration_ms": 9}],
  "graph": {"nodes": [...], "edges": [...]},
  "chart": {...},
  "session_id": "..."
}
```

```
POST   /api/v1/auth/login                         {username,password} -> token (web session)
GET    /api/v1/workspaces
POST   /api/v1/workspaces
GET    /api/v1/workspaces/{id}
PATCH  /api/v1/workspaces/{id}                    settings
POST   /api/v1/workspaces/{id}/query              {prompt, session_id?, mode?, allow_write?}
POST   /api/v1/workspaces/{id}/query/stream       same, SSE agent events
POST   /api/v1/workspaces/{id}/sql                {sql}
GET    /api/v1/workspaces/{id}/search?q=&k=       hybrid retrieval, no LLM
GET    /api/v1/workspaces/{id}/documents
POST   /api/v1/workspaces/{id}/documents          multipart or {text,title} -> 202 {id}
GET    /api/v1/workspaces/{id}/documents/{doc}    status, metadata
PATCH  /api/v1/workspaces/{id}/documents/{doc}    {pinned}
DELETE /api/v1/workspaces/{id}/documents/{doc}
GET    /api/v1/workspaces/{id}/tables[/{name}]
GET    /api/v1/workspaces/{id}/graph/search?entity=&class=&relation=&hops=
GET    /api/v1/workspaces/{id}/graph/path?from=&to=
POST   /api/v1/workspaces/{id}/graph/extract       -> 202, cost in response
POST   /api/v1/workspaces/{id}/graph/revalidate
GET    /api/v1/workspaces/{id}/ontology            current version; YAML or JSON by Accept
PUT    /api/v1/workspaces/{id}/ontology            import: validate, write a new version
GET    /api/v1/workspaces/{id}/ontology/versions[/{v}]
POST   /api/v1/workspaces/{id}/ontology/versions/{v}/restore
POST   /api/v1/workspaces/{id}/ontology/propose    {mode: full|extend, from?, sample?, auto_accept?} -> 202, cost
GET    /api/v1/workspaces/{id}/ontology/candidates
PUT    /api/v1/workspaces/{id}/ontology/candidates/{cid}   {action: accept|rename|merge_into|reparent|reject, ...}
GET    /api/v1/workspaces/{id}/context             current; Markdown or JSON by Accept
PUT    /api/v1/workspaces/{id}/context
GET    /api/v1/workspaces/{id}/context/versions
GET    /api/v1/workspaces/{id}/sessions[/{sid}]
GET    /api/v1/workspaces/{id}/sessions/{sid}/export?format=sql|markdown
GET    /api/v1/workspaces/{id}/audit              detail rows, members only
GET    /api/v1/workspaces/{id}/members  POST/DELETE ...   (owner)
GET    /api/v1/admin/users  POST ...  GET /api/v1/admin/audit   (admin; skeletal log)
```

Uploads, extraction, and proposals return `202` and are processed by a bounded in-process
queue (one worker per workspace); clients poll the resource. Rate limiting per token via
`tower_governor`.

### 11.3 MCP server

Same tool set over two transports: `quack mcp [--workspace NAME]` on stdio for Claude Code
and editors (one line in `.mcp.json`, no server needed), and `/mcp/v1/{workspace}` SSE
under `quack serve` with the workspace token.

Tools: `query`, `search`, `sql`, `search_graph`, `find_path`, `list_tables`,
`describe_table`, `list_documents`. Resources: `quack://workspace/tables`,
`.../tables/{name}/schema`, `.../documents`, `.../ontology`, `.../context`.

### 11.4 Terminal session (TUI)

A Claude Code-style single-pane transcript: streaming answers, inline steps, citations
rendered as footnotes, charts drawn with ratatui, permission prompts answered with `y`/`n`/
`a`, direct SQL when input starts with `SELECT`/`WITH`/`FROM`/`DESCRIBE`/`SHOW`/`PIVOT`/
`SUMMARIZE`. Slash commands: `/help`, `/tables`, `/schema`, `/sql`, `/ingest`, `/attach`,
`/docs`, `/pin`, `/graph`, `/path`, `/ontology` (`propose`, `review`, `export`, `import`),
`/mode`, `/chart`, `/export`, `/model`, `/context`, `/clear`, `/quit`. Keys: `Enter` send,
`Shift+Enter` newline, `Up`/`Down` history, `PageUp`/`PageDown` scroll, `Ctrl+C` cancel
then quit, `Ctrl+L` clear.

Works on a named workspace (`-w`) or a `.quack/` directory (section 5.1); the latter reads
local tabular files in place as views.

### 11.5 Print mode and CLI

```
quack -p "PROMPT" [-w NAME] [-f table|json|ndjson|csv|markdown] [--mode chat|query]
      [--allow-write] [--model P/M] [-c | -r SESSION]
quack -q "SQL" [-w NAME] [-f ...] [--internal]
quack ingest FILE... [-w NAME] [--as NAME] [--pin] [--no-embed] [--extract]
quack docs | tables | schema TABLE | graph ENTITY [--hops N]
quack ontology show | propose [--extend|--from PACK] [--sample N] [--auto-accept]
              | review | accept ID... | reject ID... | export FILE | import FILE
              | versions | restore V
quack context show | edit | export FILE | import FILE
quack sessions | export SESSION [--sql|--markdown]
quack auth login|status|logout PROVIDER
quack serve [--bind ADDR] [--local]
quack mcp [-w NAME]
quack user add|list ; quack token create ; quack member add|remove   (server admin)
```

stdin that is not a TTY is data (loaded as table `stdin`, or as a pasted document with
`--as-document`); stdout carries the answer or result set; stderr carries steps. Exit
codes: 0 ok, 1 runtime error, 2 usage, 3 write refused, 4 auth required.

### 11.6 Desktop window (`quack desktop`)

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
  names, the context diff, the proposal accepted). An admin sees who accessed what and
  when across every workspace; a member sees what was done inside theirs. Export and
  import of context or ontology, and session export, are audited because they move content
  across the boundary. Logins, failed logins, token use, and membership changes are audited
  with no workspace.

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

[providers.ollama]
type = "ollama"
auth = "none"
base_url = "http://localhost:11434"
embedding_dimension = 768

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
# client_secret_env = "AZURE_CLIENT_SECRET"   # server as confidential client
# device_code = false

[retrieval]
top_k = 8
rrf_k = 60

[ingestion]
chunk_size_tokens = 512
chunk_overlap_tokens = 64
embedding_batch_size = 64
tokenizer_encoding = "cl100k_base"
upload_max_mb = 512

[context]
max_tokens = 4000

[analysis]
max_query_rows = 100
query_timeout_seconds = 30
memory_limit_mb = 256
threads = 4
max_turns = 10
history_token_budget = 32000
default_mode = "chat"                   # default for new workspaces

[graph]
enabled = false                         # default for new workspaces
max_traversal_depth = 3
max_nodes = 200
merge_threshold = 0.08

[ontology]
propose_sample_chunks = 200
min_support_documents = 3
key_overlap_threshold = 0.8
drift_prompt_threshold = 25             # out-of-ontology mentions before suggesting --extend

[server]
bind = "127.0.0.1:8080"                 # QUACK_BIND
local = false
workers_per_workspace = 1
audit_retention_days = 0                # 0 = keep forever; pruning is an audited admin action

[tui]
tick_rate_ms = 50
```

---

## 14. Build and Distribution

- **One binary, `quack`, no Cargo features.** Every surface is a subcommand and every
  build contains all of them. Static musl on Linux (`x86_64`, `aarch64`), native on macOS
  and Windows. DuckDB and SQLite are bundled and statically linked.
- **No DuckDB extension is ever installed or loaded at runtime.** A static musl binary
  cannot `dlopen`, and `libduckdb-sys` can only compile in `json`, `parquet`, `icu`, and
  `autocomplete`. Anything that would need another extension (`vss`, `fts`, `excel`,
  `httpfs`, the Postgres and SQLite scanners) is implemented in Rust or not built.
  XLSX uses a Rust reader; keyword search is quack's own BM25 index; vector search is
  an exact scan.
- **Desktop bundles:** when `quack desktop` exists, `tauri build` wraps the same binary
  into `.dmg`, `.msi`, and `.AppImage` installers. Not a separate binary.
- **mimalloc** (`secure`) as the global allocator.
- **Crypto:** rustls + aws-lc-rs; `fips` as a build-time option; `cargo tree -i ring` and
  `-i openssl-sys` are release gates.
- **Container image** for `quack serve`: CSS stage with the standalone Tailwind binary
  (checksum verified), `rust:<MSRV>-alpine` with cargo-chef for the musl build (`cmake`,
  `clang`, `go` for aws-lc), `distroless/static` `nonroot` runtime with `/quack` and
  `/data`. `docker buildx bake` for `amd64` and `arm64`. The Rust image tag must track
  `rust-toolchain.toml`. A compose file runs `quack serve` beside Ollama with the model
  volume pre-populated; the image tarball is `docker load`ed on air-gapped hosts.
- **Dependencies** follow the workspace rules in `CLAUDE.md`. New entries this design
  needs, versions looked up when added: `argon2`, an MCP crate,
  `docx-rs`, `scraper` or `html2text`, `serde_yaml` or `serde_yml` for ontology
  interchange, `tauri` (only when `quack desktop` is built), `tower_governor`.

---

## 15. Scaling Constraints and Decision Points

These are the properties of the chosen storage that the team should accept explicitly,
because the system being replaced runs on Postgres.

1. **DuckDB is embedded and single-process.** One `quack serve` process owns every
   workspace file. Vertical scaling only. Concurrency within a process is fine: DuckDB's
   MVCC lets the ingestion worker, session writes, and readers share a workspace.
   Horizontal scaling or an HA pair is not possible without moving storage to a server
   database. For a single-instance deployment this is a simplification, not a limitation.
2. **Vector search is an exact scan, not an index.** Every query computes the cosine
   distance against every stored embedding inside DuckDB. That is tens of milliseconds
   for a hundred thousand 768-dimensional chunks and grows linearly; the working set is
   `chunks × dimension × 4` bytes (about 3 GB per million chunks). Past a few hundred
   thousand chunks per workspace, add an approximate index that ships inside the binary
   (a pure-Rust HNSW crate over the same stored vectors) rather than a DuckDB extension.
   BM25 is an indexed join on `_quack_terms` and stays fast far beyond that.
3. **One file is the boundary, so one file is the backup unit.** Back up a workspace by
   copying its directory while the server holds no write transaction (`quack workspace
   snapshot NAME` does this via DuckDB's `CHECKPOINT` and a copy). There is no
   cross-workspace transaction and none is needed.
4. **Storage backend seam.** `retrieval/`, `graph/`, and `ontology/` are written against
   small traits so that a Postgres + pgvector backend can be added later without touching
   the agent or the interfaces. That backend is out of scope now; the seam is in scope so
   the door stays open.
5. **Migration from the current deployment.** Documents are re-uploaded and re-embedded
   rather than migrated from pgvector, because chunking and metadata differ. A
   `quack import anythingllm --url ... --key ...` command that pulls workspaces, documents,
   system prompts, and threads through the AnythingLLM API is the planned path; it is
   listed in section 19 after the core is stable.

---

## 16. Testing

**Unit.** `workspace/` layout and discovery, `.quack/` treated as unclassified on import;
`storage/` migrations and CRUD for both databases, dimension mismatch, `_quack_` tables
hidden from listing and refused to the agent; `ingestion/` chunking with heading and page
metadata (`proptest`: no chunk over budget, concatenation covers the input), SHA dedup;
`retrieval/` RRF fusion on fixture rankings, citation validation strips unknown markers;
`analytics/` classification table (SELECT variants read; DDL, DML, COPY, SET, ATTACH
write), limits, row capping; `ontology/` inheritance, domain/range validation, property
types, mapping validation, versioning and stale detection, YAML round trip, table-evidence
induction on fixture tables (key detection, overlap relation), document-evidence
normalization on fixture extraction output (clustering, domain/range inference, hierarchy
inference, support thresholds), candidate actions; `graph/` extraction parsing (valid,
malformed, out-of-ontology dropped and counted for drift), merge on normalized label,
embedding merge proposals, provisional flagging, traversal on a fixture with a cycle, path
search; `agent/` loop against a mocked rig model with canned tool calls including a
refused write and a timeout, prompt contains only registered tools, mode enforcement
excludes provisional graph results; `llm/` `TokenManager` reuse, single refresh under
concurrency, re-auth, cache round-trip against a mock IdP.

**Integration.** Upload PDF -> ready -> question in query mode returns an answer with a
citation on the right page; keyword-only question (a policy number) is answered via FTS;
CSV -> question produces SQL referencing a context definition; `ontology propose` on a
fixture workspace of two tables and ten documents yields the expected classes, one overlap
relation, and a mapping, and accepting them builds nodes with provenance; `--auto-accept`
marks the graph provisional and query mode ignores it; an ontology edit marks the graph
stale and revalidate drops the invalid edge; write refusal per interface (exit 3, 403, MCP
error) and success with permission; session round trip and export with audit rows in both
databases sharing an id; a denied open by a non-member and an expired token each write an
`audit_log` row with `outcome = denied`; no code path updates or deletes `audit_log` rows; workspace isolation via CLI and API, and an admin without
membership cannot read workspace content; server token lifecycle, roles, upload queue;
MCP stdio client lists tools and runs `query`; REST and print mode return byte-identical
JSON for the same question with a mocked model.

**Manual.** Web UI end to end against Ollama including the proposal review flow; TUI
streaming and permission prompt; OAuth browser and device-code flows against a real
tenant; desktop app on each platform; air-gapped static binary with bundled extensions.

`cargo-mutants` on `retrieval/`, `analytics/`, `ontology/`, and `graph/` before a release.

---

## 17. Gaps Between This Document and the Code

Every gap is a GitHub issue; this list is the map from the design to the tracker and is
updated as issues close. Ordered by risk.

1. ~~Verify against a live model~~ (#20, closed): print mode and the terminal session are
   verified with gpt-oss:20b on Ollama. Sections 7, 8, 9.
2. ~~No OAuth~~ (#25, closed): PKCE and device-code login, encrypted cache, `quack auth`;
   the server's confidential-client mode is wired (`client_secret_env`) and gets its live
   test with #26. Section 10.2.
3. ~~No server, REST API, web UI~~ (#26, closed): `quack serve` with the REST API,
   password and token auth, roles, the split audit, the upload queue, and the askama +
   htmx web UI (workspaces, chat with steps, citations, and charts, documents, tables,
   SQL, context editor, settings with members and tokens, admin users and audit). The
   graph and ontology pages arrive with #27 and #28. **No MCP** (#29); **no desktop
   window** (#35). Sections 11, 12.
4. **Ontology** (#27): the model, validation, versions with diff and restore, the
   built-in default, JSON import and export, the CLI, API, and web page are in;
   induction with the candidate queue is in progress. YAML was dropped: JSON is the only
   interchange form. **No graph** (#28). Sections 6.3 to 6.5.
5. **DOCX, HTML, PPTX, XLSX unsupported** (#16; XLSX via a Rust reader). Sections 6.1, 6.2.
6. **No `ATTACH` to external databases** (#21): needs a Rust-side design now that scanner
   extensions are out. Section 6.2, step 13.
7. **Document registry lacks `sha256` dedup, `source`, `title`** (#22). Section 5.4.
8. **Sessions have no `created_by` or sharing; print mode cannot take stdin as data**
   (#23). Sections 8, 11.5.
9. ~~Context `edited_by` and the `_quack_audit` detail table~~ (#24, closed): the
   server records the editing user and writes the detail row under the access row's id.
   Sections 5.3, 5.4, 12.
10. **No release pipeline** (#30). Section 14.
11. **No stemming in keyword search** (#31, deliberately deferred); **no reranking hook**
    (#34); **large-workspace vector index options** (#32, research). Sections 6.1, 15.
12. ~~Web UI mapping of the chart spec to ECharts~~ (#26, closed): `static/js/app.js`
    maps the spec to an ECharts option. Section 9.

---

## 18. Scope

### In scope

- `quack-core` with the three substrates and one agent, as an event stream
- Workspace as the classification boundary: one DuckDB file plus `files/`, everything
  classified inside it; `control.db` holds access control only; audit split at the boundary
- Documents: upload, paste, path, stdin; PDF, Markdown, text, HTML, DOCX, PPTX; chunk
  metadata; hybrid retrieval; citations; pinned documents; SHA dedup; chat and query modes
- Tables: CSV/TSV/Parquet/JSON/JSONL/XLSX; `ATTACH` to Postgres, SQLite, MySQL, S3/HTTP
- Ontology: stored in workspace tables with inheritance, relations with domain/range, typed
  properties, table mappings, built-in default, versioning with snapshot, diff, restore,
  and stale detection; YAML/JSON import and export
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
- Interfaces, all in one binary: web UI, REST API, MCP (stdio and SSE), TUI, print mode,
  and `quack desktop` (last, if ever)
- Server: users with password login, tokens with scopes, roles, audit, upload queue
- Static builds, container image and compose, desktop bundles

### Deferred, in rough priority order

1. AnythingLLM import command (workspaces, documents, system prompts, threads via its API)
2. OIDC login for server users
3. Data connectors: URL fetch, GitHub, Confluence, SharePoint
4. Cross-encoder reranking provider
5. OCR for scanned PDFs
6. Postgres + pgvector storage backend behind the retrieval/graph/ontology seam
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
9. Ontology: tables, model, validation, default, versioning, YAML interchange, prompt
   rendering, editor page, API.
10. Ontology induction: table evidence, document evidence, candidates, review queue,
    extend mode and drift counting, auto-accept with provisional marking; CLI, API, and
    web hooks.
11. Graph: extraction from documents and mapped tables, resolution, provenance, traversal,
    tools, TUI tree, web graph page, stale and provisional handling.
12. MCP over stdio and SSE.
13. External data import (the Rust-side replacement for `ATTACH`), XLSX via a Rust reader,
    `workspace snapshot`.
14. Release engineering: musl targets, macOS, Windows, container image, compose.
15. AnythingLLM import command.
16. `quack desktop` and installer bundles, last and only if there is demand.
