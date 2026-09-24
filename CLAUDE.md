# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It also drives the `rust-agents` Claude Code plugin (conventions live in `.claude/rules/`).

## What this is

A knowledge engine with many interfaces, built in Rust. A workspace holds documents
(vectorized), tables (DuckDB), and an ontology-backed knowledge graph; one agent answers
across all three and shows every action. The core is a library; the web UI, REST API, MCP
server, terminal session, and print mode are thin clients of it, all subcommands of one
`quack` binary with no Cargo features (`quack desktop`, a Tauri window over the embedded
server, is planned: #35, not started). The
design is `docs/design-doc.md`; read it before any non-trivial change, and check its
section 17 for where the code still lags. The chosen stack:

- **Workspace** of crates under `crates/` (edition 2024, resolver 3, MSRV 1.98.1)
- **DuckDB** (via `duckdb-rs`), one file per workspace, holding everything classified
  about that workspace: user tables, chunks, graph, ontology, context, sessions, audit
  detail (all internal tables prefixed `_quack_`)
- **SQLite** (via `sqlx`) for `control.db` in server mode: users, workspaces, membership,
  tokens, and the mandatory append-only access audit log; nothing workspace-revealing
- **sea-query** for type-safe SQL generation against `control.db`; bound parameters for
  DuckDB internals
- **rig** for LLM providers (Ollama, OpenAI-compatible, Anthropic) and the agent loop
- **aws-lc-rs** as the single crypto/TLS provider (never OpenSSL or `ring`), with its and
  rustls's `fips` features on Linux, so the distributed musl binaries and the image run on
  the FIPS-validated AWS-LC module; it is the only family where the FIPS build links
  statically (`docs/crypto.md`)

## Repository layout

```
Cargo.toml            # virtual workspace: deps menu + strict lints + profiles
.clippy.toml          # clippy tuning (levels live in Cargo.toml)
.rustfmt.toml         # stable-only formatting
deny.toml             # advisories, license allow-list, OpenSSL/ring bans
rust-toolchain.toml   # pinned 1.98.1 + rustfmt + clippy
Makefile              # build / fmt / lint / test / deny
crates/               # quack-core (engine), quack (the binary) — see crates/README.md
docs/                 # design-doc.md (the product), architecture, migrations, crypto, web-ui, ci-cd
.claude/rules/        # branching, commits, continuous-improvement conventions
```

## Conventions

- **Lints are strict and inherited.** Every crate uses `[lints] workspace = true`. The
  baseline denies panics (`unwrap`/`expect`/`panic`/`todo`), panic-prone indexing/slicing,
  lossy casts, and `arithmetic_side_effects`, and warns on all of clippy `pedantic`. In
  tests, opt out narrowly: `#[expect(clippy::unwrap_used, reason = "...")]`.
- **Dependencies are pinned** to exact versions in `[workspace.dependencies]` with
  `default-features = false`. Crates opt into features explicitly. When adding a dependency,
  look up the current version and add it there, not in the member crate.
- **Errors:** `thiserror` for library crates, `anyhow` for binaries.
- **Logging:** `tracing` (`error!`/`warn!`/`info!`/`debug!`), never `println!`.
- **Date/time:** `jiff`, not `chrono` or `time`.
- **Database IDs:** UUID v7, client-generated (`uuid::Uuid::now_v7()`) — never v4.
- **Classification boundary:** anything that can reveal workspace content goes in the
  workspace DuckDB file, never in `control.db`. Every workspace-touching request writes an
  access `audit_log` row (including denials) plus a `_quack_audit` detail row.
- **Types:** newtypes over primitives, enums for state machines, `let...else` for early returns.
- **Commits:** Conventional Commits (`.claude/rules/commits-and-issues.md`). No AI/co-author
  trailers. Never push to `main` — branch and PR.

## Project rules

Enforceable invariants the compiler can't catch — read before implementing or reviewing a
data-layer, crypto, or dependency change:

- **`.claude/rules/code-standards.md`** — crypto (aws-lc-rs only), data-layer schema
  rules, and workspace hygiene, as review-gate checklists linking to `docs/`.
- **`.claude/rules/development-discipline.md`** — how agents carry out design,
  implementation, diagnostics, and agent-team hand-offs (the *how*, complementing
  `code-standards.md`'s *what*).
- `.claude/rules/branching.md`, `.claude/rules/commits-and-issues.md`,
  `.claude/rules/continuous-improvement.md` — branch/commit/CI conventions for the
  `rust-agents` flow.

## Interfaces today

```bash
cargo run --bin quack -- -q "SELECT 1" [-f table|json|ndjson|csv|markdown]   # SQL, no agent
cat x.csv | cargo run --bin quack -- -p "..."                                  # piped stdin is the temp table `stdin` (-p and -q; --stdin waits for a slow pipe)
cargo run --bin quack -- ingest sales.csv -w ws                                # file -> table(s) or chunks
#   tables: CSV/TSV, Parquet, JSON/JSONL, XLSX/XLS/ODS (one table per sheet); chunks: PDF, Markdown, text, HTML, DOCX, PPTX
cargo run --bin quack -- -p "question" -w ws [-f text|json]                    # one agent turn; steps on stderr
cargo run --bin quack -- -w ws                                                 # terminal session (needs a TTY)
cargo run --bin quack -- sessions | export ID [--sql]                          # sessions live in the workspace file
cargo run --bin quack -- ontology show|init|import|export|versions|diff|restore   # the graph schema, versioned in the workspace
cargo run --bin quack -- ontology propose [--documents] [--auto-accept] [--from FILE] | review | accept ID.. | reject ID..
cargo run --bin quack -- graph search ENTITY [--hops N] | search --class C | path A B | status | extract [-y] | revalidate | review | merges | merge ID..
cargo run --bin quack -- okf export DIR|-                                        # the workspace as an Open Knowledge Format bundle; `ingest DIR` imports one
cargo run --bin quack -- embeddings refresh [-y]                                # refresh vectors a changed embedding model, width, or prefix left stale
cargo run --bin quack -- import postgres://u:p@h/db --table t --from orders      # snapshot a Postgres/SQLite query or an http(s) data file as a table
cargo run --bin quack -- auth login|status|logout PROVIDER                      # OAuth token for an auth = "oauth" provider
cargo run --bin quack -- config [--changed] [--json]                            # every recognized setting, its value and origin, the file's unknown keys, the env vars read
cargo run --bin quack -- doctor [--offline] [--json]                            # every check with its fix: config, data dir mode, workspace, model providers (probed), bind; exit 1 on a failure
cargo run --bin quack -- user add|list ; token create|list|revoke ; member add|remove|list ; audit   # server admin
cargo run --bin quack -- serve [--bind ADDR] [--local]                          # web UI, REST API under /api/v1, MCP under /mcp/v1/{workspace}
cargo run --bin quack -- mcp [-w ws] [--allow-write]                            # MCP server on stdio for Claude Code and editors
```

Turns are recorded in `_quack_sessions` / `_quack_messages` inside the workspace DuckDB
file (`quack_core::storage::sessions`); `-c` / `-r ID` replay history to the model, trimmed
to `[analysis].history_token_budget`. Retrieval is hybrid (exact cosine scan plus quack's own
BM25 over `_quack_terms` with Snowball-stemmed tokens, reciprocal rank fusion in
`WorkspaceDb::search_hybrid_chunks`;
no DuckDB extension is ever loaded, see design doc section 14), then an optional reranker
(`analysis::rerank`, `[retrieval].rerank = "none" | "model"`; `model` over-fetches
`rerank_candidates` and has the chat model order them); citations are registered per turn
(`analysis::citations`) and validated before the answer is returned. Sessions have a mode,
`chat` or `query`; `--mode` / `/mode` set it. The workspace context (owner-written
instructions, `quack_core::storage::context`, versioned in `_quack_context`) is injected
into the system prompt after the schema and documents, capped at `[context].max_tokens`;
the agent never writes it. Charts are `analysis::chart::ChartSpec` (bar, line, scatter,
pie; 200 points max), not ECharts.

Background work is asynchronous everywhere (design doc 4.1): `quack_core::jobs::JobQueue`
runs submitted jobs with optional lanes that keep submission order (a chat session is a
serial lane, so a follow-up waits for the answer before it; a workspace's uploads, graph
extraction, and document pass have their own), cancel tokens, per-chunk progress, and a
broadcast of `JobInfo` snapshots every interface reports from. There is no job pool:
resources are limited where they are used. Every rig client is built over
`llm::LimitedHttp`, which holds one permit of the process-wide gate for the provider and
the model named in the request body (`[providers.NAME].max_concurrent_requests` each, 1
for Ollama, 8 otherwise) until the body or stream ends; a freed permit goes to interactive
requests (`TurnRequest::run`, `Embedder::embed_interactive`, via the `quack_core::priority` task-local) before
background ones. The registry is in memory only (labels can be workspace content).
The terminal is one async loop (`tokio::select!` over crossterm's `EventStream`, one
`AppMsg` channel, the job broadcast, a spinner tick) and submits every question,
statement, file, import, and ontology or graph verb as a job, so it never blocks its input:
a strip above the prompt shows active jobs, `/jobs` lists them, `/cancel N` stops one, and
write prompts from concurrent work queue up. Its commands' database steps run in the order
typed on one worker task (`App::on_db`: reads on the reader pool, writes in the writer's
interactive line), never on the loop's thread; input typed during `/new`, `/resume`, or
`/mode` waits for the switch. `SharedDb` is `Arc<storage::writer::Writer>`, an actor: one
thread per workspace owns the writer connection and runs the owned (`Send + 'static`)
closures sent to it one at a time, interactive before background by
`quack_core::priority` (a task-local, interactive unless the job queue scopes a
background job kind). Callers await `Writer::run` (or the server's `with_db`); nothing
locks the writer. Ingestion, import, extraction, `graph_cli`, and `ontology_cli` take
`&Writer` (the CLI spawns one per command, like the server and the terminal) and send
one step at a time, rendering command output into a buffer on the writer's thread
(`graph_cli::rendered`); file parsing runs on the blocking pool, so every job, the
terminal's included, is a plain task on the runtime. Tests that also read through a
`WorkspaceDb` give the pipeline a writer over its `try_clone_reader` connection.

The agent turn is an event stream (`quack_core::analysis::events`): text deltas, tool
started/finished with timing, permission requests, turn complete. Every interface consumes
it. `--allow-write` lets the agent run mutating SQL without asking; otherwise the terminal
prompts y/n/a and `-p` refuses and exits 3. Provider construction lives in `quack_core::llm`; interfaces never build rig
clients themselves. OAuth providers (`quack_core::llm::oauth`) hand out a bearer through
one shared `TokenManager` per provider: PKCE or device-code login via `quack auth`, an
AES-256-GCM cache under `<data_dir>/tokens/` keyed from the OS keychain or a 0600 key file,
silent refresh, and `Error::AuthRequired` (exit 4 from every command that reaches a provider) when no flow can run.
Every interface returns one response object, `AgentResponse::to_json` (answer, citations with
labels, queries, steps, graph, chart, `write_refused`, `cancelled`, `usage`, `session_id`); a write refused
inside a turn is `write_refused: true` (REST 200 plus a `write_refused` SSE event, MCP structured
content, print exit 3). `usage` is `AgentResponse::usage`, the provider's own
`input_tokens`/`output_tokens`/`total_tokens` for the turn taken off rig's final response
(the per-request counts summed when a turn derails first), `null` when the provider
reported none, and copied onto the assistant message's metadata in `_quack_messages`. It is
a record: the history trim and `OllamaWindow` still use the four-characters-per-token
estimate (`text::Tokens`), since both run before the call.
The ontology (`quack_core::ontology`, design doc 6.3) is classes with single inheritance
from `entity`, relations with a domain and a range, typed properties, and table mappings.
It lives in the `_quack_ontology_*` tables; `ontology::store::save` validates, checks
mapped tables and columns against the workspace, and writes a new version with a JSON
snapshot, `since_version` carried over for items that already existed. JSON is the only
interchange form (export, import, `PUT /ontology`); a file is never the source of truth.
The system prompt carries a compact rendering when an ontology exists
(`Ontology::render_capped`, 30 items per section with the rest counted; extraction still
gets `render_for_prompt` in full, since the model may only answer with ids it was shown),
and the `describe_class` tool registers alongside it: one class with its ancestry,
subclasses, typed properties, the relations it takes part in, its mapped table, and the
graph's exact count of it (`graph::store::class_census`) — the count traversal cannot give,
since a class listing stops at `max_nodes` and user SQL may not read `_quack_` tables. Induction from
tables (`ontology::induction::propose_from_tables`, no model calls) proposes a class per
table, typed properties per column (enum, date, number, boolean, string), the unique
non-null column as key, a relation where a column's values overlap another table's key,
and a mapping; proposals sit in `_quack_ontology_candidates` (`ontology::candidates`)
until accepted, renamed, merged, reparented, or rejected, and accepting writes a new
version. Document evidence (`ontology::documents`) samples chunks evenly across ready
documents, runs open extraction through `extraction::Extract` (the chat model via
`llm::chat_extractor`; tests use a canned one), normalizes type and relation names
(snake_case, singular, near-synonyms clustered by embedding cosine when an embedding model
exists), infers hierarchy from co-labelled mentions and domain and range from endpoints,
turns recurring attributes into typed properties, and marks candidates under
`[ontology].min_support_documents` as `low_support`. `quack ontology propose --documents`
shows the cost and asks first; the API's `{"documents": true}` answers 202 and runs in the
background, auditing the run's end under the same run id.
The knowledge graph (`quack_core::graph`, design doc 6.4) lives in `_quack_graph_nodes`,
`_quack_graph_edges`, `_quack_provenance`, and `_quack_graph_merges`. `graph::tables`
turns mapped rows into nodes and edges deterministically; `graph::extract` sends each
chunk its `ChunkPlan` names (every unextracted one, or an even sample chosen in SQL by
`WorkspaceDb::sample_chunk_ids`, reading text a page at a time) to the chat model (`llm::graph_extractor`, preamble from
`Ontology::extraction_prompt`) and validates the answer against the ontology, counting unknown
classes and relations as drift in `_quack_meta.graph_drift`; `graph::resolve` embeds
node labels, merges near-identical labels of one class, and queues the rest as merge
proposals; `graph::traverse` resolves an entry point (exact label, alias, then embedding)
and walks neighborhoods, shortest paths, and classes with subclass expansion, bounded by
`[graph]`. Every interface asks through `graph::query::{GraphQuery, PathQuery}`: `new`
trims and defaults the caller's fields, `run` checks class and relation ids against the
ontology and resolves the entry points, and an unresolved path end is an `UnknownEntity`
naming the closest labels. `graph::store::status` reports size, `provisional` (the newest ontology version
was auto-accepted), `stale` (`graph_built_with_ontology_version` lags), pending merges,
and drift; `revalidate` drops what the current ontology no longer allows. The agent
registers `search_graph` and `find_path` only when the graph has nodes, query mode drops
provisional results, and every response shape carries the turn's `graph` results. Their
rendering (`Display for GraphResult` in `graph::traverse`, shared with `quack graph` and the terminal)
carries each node's and edge's typed properties, bounded; a class or relation id the
ontology does not define is refused with the ids that do exist, a name that matches no
entity comes back with the closest labels (`traverse::suggest_entities`), and a result
query mode emptied by dropping provisional nodes says so rather than claiming the graph
is empty. A result cut short by `max_nodes` says so too, and a class listing carries the
total it was capped from (`GraphResult::total_nodes`, `truncated`). Provenance to a mapped
table renders as a predicate `run_sql` can run, since the mapping knows the key column. The
tool guidance in the system prompt gains a numbered graph procedure whenever those tools
are registered.

Every embedding goes through `quack_core::embedding::Embedder` as an `Input`, which names
its role: `Query` (search), `Document { title, text }` (a chunk under its heading), or
`Similarity` (entity labels and names, ontology type names). It adds the input prefixes the
model family was trained with (`presets::Family::of`, from each model card; `[embedding]`
overrides any role, `ResolvedPrompts::for_model`) and returns `Vector`s checked against the
profile's `Dimension`; `.clippy.toml` disallows rig's raw `embed_text`/`embed_texts`.
Every long run (ingestion, import, graph extraction, the ontology's document pass,
embeddings refresh) takes one `progress::RunControl`: progress per unit, and the cancel
token it checks between units and races model calls against. The model, width, and
prefixes are the `Profile`; every stored
chunk and node vector carries its fingerprint (`embedding_profile`, profiles in
`_quack_embedding_profiles`), and vector search, label matching, and merge proposals use
only the current profile's vectors. Stale or missing vectors are found by keyword until
`embedding::refresh::run` (`quack embeddings refresh`, `/embeddings refresh`, `POST .../embeddings/refresh`,
the Documents page) embeds them again; a width change keeps the old vectors until then.

Retrieval and the graph meet through `_quack_provenance`: `search_documents(entity)`
resolves the name (the same entry-point resolution `search_graph` uses), narrows the search
to the chunks that entity was extracted from (`graph::store::chunks_of_nodes` into
`storage::workspace::ChunkScope`, which bounds both the vector and the BM25 leg), and every
hit names the entities the graph took from it (`graph::store::entities_of_chunks`). An
entity that exists only in mapped table rows says so instead of returning nothing. The
`entity` argument is offered only while the graph has nodes (`SearchDocumentsTool::with_model`, `text_to_sql::Modeled`),
like the graph tools themselves. Ollama embedding requests go through `llm::OllamaEmbedder`,
not rig's client, so they carry `keep_alive` and a chunk-sized `num_ctx`.

Server access control lives in `quack_core::storage::control`: users (argon2id), workspace
membership with `Role` (viewer, member, owner), API tokens stored as SHA-256 hashes with
`Scope`s, and the append-only access `audit_log` (`AuditEntry`, `query_audit`); the content
half of each audit row is `storage::audit` (`_quack_audit`) inside the workspace under the
same UUID v7. Ingestion is `register_document` (status `queued`, or `Registration::Duplicate` when the
bytes' SHA-256 already belong to a non-failed document whose table or chunks still exist;
`Error::TableTaken` when the file's table belongs to another live document) plus `Processing::run`
(`processing` to `ready` or `error`, recording `chunk_count` and the parsed title);
`ingest_file` does both and takes a `NewFile` (name, bytes, `DocumentSource`, optional
title and uploader, and its `RunControl`).

The MCP server (`crates/quack/src/mcp.rs`, `rmcp`) exposes `query`, `search`, `sql`,
`list_tables`, `describe_table`, `list_documents` and the `quack://workspace/...` resources;
`quack mcp` serves it on stdio (unaudited, like the CLI) and `server/mcp_http.rs` serves it
at `/mcp/v1/{workspace}` behind `Access::resolve`, one transport per workspace, user, and write
permission, audited with channel `mcp`.

External data comes in through `quack_core::import` (`quack import`, `POST .../import`, the
Tables page form, `/import` in the terminal): a Postgres or SQLite query runs on the source
with every column cast to text through sqlx (`tls-rustls-aws-lc-rs`), or a CSV, Parquet,
JSON, or workbook file is fetched over HTTP(S), and the rows load through the normal
ingestion path as a document with source `import` and the redacted URL as title. No
`ATTACH`: the workspace never reaches out at query time.

`quack serve` (`crates/quack/src/server/`) is a thin axum client of core: `auth.rs` turns a
bearer (login session or API token), the session cookie, or `--local` into an `Identity`,
and `Access::resolve` resolves the workspace, checks role and token scope, and writes the denied
audit row itself, so a handler holding an `Access` is already authorized. Both login paths
go through one `auth::password_login`, and a browser session expires at
`[server].session_max_age_hours` or after `session_idle_minutes` unused, whichever is first
(`state::WebSessions`); its cookie is `HttpOnly`, `SameSite=Lax`, `Max-Age`d to the
absolute lifetime, and `Secure` unless the request came from loopback. One `tower_governor`
limiter covers the web UI, the API, and MCP, with a tighter one on the two login routes and
none on `/healthz` (design doc 12). The same routes carry `no-store` cache headers
(`server::no_store`) unless the handler set `Cache-Control` itself, as the static assets
do. Every
workspace-touching handler then records the allowed row plus its `_quack_audit` detail
through `Access::audit`. A handler's reads go through `App::read` (a reader-pool
connection in a read-only transaction, so a read never occupies the writer and a write
slipped into one is refused); its writes go to the writer through `state::with_db`; and
its `_quack_audit` detail row goes to the workspace's insert-only audit connection
(`storage::audit::AuditLog`, a writer clone on its own thread), so no request waits for a
write in progress just to record itself. `query/stream` forwards the agent event stream as SSE (`text`,
`status`, `tool_started`, `tool_finished`, `write_refused`, `complete`, `error`); uploads return 202 with a `job` id
and run on the work queue in a lane of `[server].workers_per_workspace` per workspace
(`queue.rs`), which locks the workspace only around each database step; `api/jobs.rs`
serves `GET .../jobs`, `.../jobs/stream` (SSE), `.../jobs/{job}`, and `POST .../cancel`,
and `/w/{id}/jobs` is the web console's Jobs page. The web UI (`server/web/`, `templates/`, `static/`) is askama pages over
the same `Access::resolve` checks and the API's `Access` operations; `WebUser` redirects to `/login` instead
of a 401; the built Tailwind CSS is committed (`make css-build` after template edits) and
htmx and ECharts are vendored (`docs/web-ui.md`). Tests drive the router with
`tower::ServiceExt::oneshot` and no model. Target CLI (`quack -p`, `quack serve`, `quack mcp`,
`quack ontology propose`, ...) is in design doc section 11.

## Common commands

```bash
make build     # cargo build --release
make fmt       # cargo fmt --all
make lint      # cargo clippy --workspace --all-targets --all-features -- -D warnings
make test      # cargo test --workspace --all-features
make eval      # retrieval, ontology induction, graph extraction, and citation numbers; no model needed
make deny      # cargo deny check
make bench     # criterion benchmarks: retrieval latency by workspace size, prompt assembly (BENCH_CHUNKS=1000000 for the 1M point)
make release-gates   # no ring or OpenSSL in the runtime tree, then cargo deny
make crypto-gates    # the tree check alone (what the release job runs; it checks deny separately)
make image     # the quack serve container image from source (docker buildx)
make help      # list targets
```

Run a specific test (once at least one crate exists):

```bash
cargo test -p <crate> <test_name>    # one test (name filter) in one crate
cargo test -p <crate>                # all tests in one crate
cargo test --workspace <test_name>   # name filter across the workspace
```

> Coverage and mutation testing are local-only: `make test-coverage`, `make test-mutants`.

## Releases

`.github/workflows/release.yml` runs on a `v*` tag only: gates, then everything that
compiles, signs, or attests happens inside `.github/workflows/reusable-build.yml`, which
never holds `contents: write` — that split is what makes the provenance SLSA Build Level 3,
since the Sigstore certificate names the build workflow as the builder. It produces
reproducible static musl binaries for Linux x86_64 and aarch64 (`Dockerfile.build`,
`docker-bake.hcl`) with CycloneDX SBOMs, native macOS arm64 and Windows x86_64 and aarch64
binaries (code-signed and, on macOS, notarized when the signing secrets exist; unsigned
with a warning when they do not), and the multi-arch `quack serve` image on GHCR from the
prebuilt binaries (`Dockerfile.release`) plus image tarballs for air-gapped hosts. The
`publish` job only downloads those artifacts, writes `SHA256SUMS`, and creates the release.
`docker-compose.yml` with `deploy/config.toml` runs the image beside Ollama. Cut a release
by pushing an annotated `vX.Y.Z` tag after the branch is pushed; `workflow_dispatch` builds
everything and publishes nothing.

## Where to read more

`docs/architecture.md` maps the design to the modules; `docs/migrations.md`,
`docs/crypto.md`, `docs/web-ui.md`, and `docs/ci-cd.md` cover the schema, TLS, web UI, and
release layers (the table in `README.md`). Read the relevant one before changing that layer.
