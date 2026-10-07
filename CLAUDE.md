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

- **Workspace** of crates under `crates/` (edition 2024, resolver 3, MSRV 1.99.0)
- **DuckDB** (via `duckdb-rs`), one file per workspace, holding everything classified
  about that workspace: user tables, chunks, graph, ontology, context, sessions, audit
  detail (all internal tables prefixed `_quack_`); the file records its schema version and
  the quack and DuckDB versions that wrote it, and a quack older than the file refuses it
  untouched (`Error::WorkspaceTooNew`, `docs/migrations.md`)
- **SQLite** (via `sqlx`) for `control.db` in server mode: users, workspaces, membership,
  tokens, and the mandatory append-only access audit log; nothing workspace-revealing
- **sea-query** for type-safe SQL generation against `control.db`; bound parameters for
  DuckDB internals
- **rig** for LLM providers (Ollama, OpenAI-compatible, Anthropic, Amazon Bedrock) and the agent loop
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
rust-toolchain.toml   # pinned 1.99.0 + rustfmt + clippy
Makefile              # build / fmt / lint / test / deny
crates/               # quack-core (engine), quack (the binary) — see crates/README.md
docs/                 # design-doc.md (the product), architecture, migrations, crypto, web-ui, ci-cd, authentication, providers
.claude/rules/        # code-standards and development-discipline gates; branching, commits, continuous-improvement conventions
```

## Conventions

- **Lints are strict and inherited.** Every crate uses `[lints] workspace = true`. The
  baseline denies panics (`unwrap`/`expect`/`panic`/`todo`), panic-prone indexing/slicing,
  lossy casts, and `arithmetic_side_effects`, and warns on all of clippy `pedantic`. In
  tests, opt out narrowly: `#[expect(clippy::unwrap_used, reason = "...")]`; `#[allow]` is
  denied, so every opt-out is an `#[expect]` with a reason.
- **Import, don't spell paths.** `clippy::absolute_paths` denies any `crate::` or
  `quack_core::` path longer than two segments outside a `use`: import the type, or import
  a function's parent module and call `module::function`. External crates are exempt
  through `absolute-paths-allowed-crates` in `.clippy.toml`; a new dependency named by a
  full path goes on that list.
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
cargo run --bin quack -- workspace create ws | workspace list [--format json]  # -w must name a workspace that exists (else exit 2); only [general].default_workspace is created on first use
cargo run --bin quack -- workspace rename OLD NEW | delete NAME [-y] | snapshot NAME [--to FILE] | restore FILE|- [--name N]   # storage::backup: one tar (manifest.json, data.duckdb, files/); delete is at once, the audit rows stay
cargo run --bin quack -- ingest sales.csv -w ws [--replace [ID]] [--types amount=DOUBLE]   # file -> table(s) or chunks; --replace supersedes the document with the same name (or ID) once the new one is ready; --types retypes columns strictly after the load
cargo run --bin quack -- ingest DIR -w ws [--prune]                            # every supported file under DIR (not a bundle), root and path recorded; re-run skips unchanged, replaces changed, reports gone files of that root (--prune deletes them)
#   tables: CSV/TSV, Parquet, JSON/JSONL, XLSX/XLS/ODS (one table per sheet); chunks: PDF, Markdown, text, HTML, DOCX, PPTX, EPUB, ODT, EML/MBOX, VTT/SRT, source code, RTF
#   --author/--authored/--tag set what the file says about itself; a table inside a document with >= [ingestion].table_rows_as_table rows also loads as <stem>_tableN
cargo run --bin quack -- tables [TABLE [--note TEXT] [--retype COL=TYPE]] [--format json]   # row counts, owner notes, profile warnings; a note or a strict retype (PUT .../tables/note, POST .../tables/retype, the Tables page)
cargo run --bin quack -- docs [--tag ID TAG | --untag ID TAG | --author ID NAME | --authored ID DATE]   # a document's own fields; PATCH .../documents/{doc} over REST
cargo run --bin quack -- -p "question" -w ws [-f text|json] [--documents DOC,..]   # one agent turn; steps on stderr; --documents limits it to those documents
cargo run --bin quack -- search QUERY -w ws [--in DOC..] [--keyword|--vector] [--explain] [-f text|json]   # analysis::search::DocumentSearch without the model: each hit's vector, keyword, and rerank rank
cargo run --bin quack -- -w ws                                                 # terminal session (needs a TTY)
cargo run --bin quack -- sessions | export ID [--sql]                          # sessions live in the workspace file
cargo run --bin quack -- saved list | add NAME --from-session ID | run NAME [--refresh] [--exit-code] | show NAME | remove NAME   # an answer's SQL re-run without the model; exit 5 when changed; cron schedules it
cargo run --bin quack -- ontology show|init|import|export|versions|diff|restore   # the graph schema, versioned in the workspace
cargo run --bin quack -- ontology rename class|relation OLD NEW                   # a new id as a new version; the graph's nodes and edges move with it
cargo run --bin quack -- ontology schema                                          # the interchange form's JSON Schema (docs/ontology.schema.json, kept equal by a test)
cargo run --bin quack -- ontology propose [--documents] [--auto-accept] [--from FILE] | review | accept ID.. | reject ID..
cargo run --bin quack -- graph search ENTITY [--hops N] | search --class C | path A B | status | extract [-y] [--reset [--all]] | revalidate [-y] | review | merges | merge ID..
cargo run --bin quack -- graph add node LABEL --class C [--property K=V] | add edge FROM REL TO | set NODE [--label L] [--to-class C] [--property K=V] [--unset K] | delete node NODE | delete edge ID   # a person's assertions, recorded with author and note
cargo run --bin quack -- graph export DIR|- [--format csv|graphml|jsonld] [--include-provisional]   # the whole graph (graph::export); csv to stdout is a tar
cargo run --bin quack -- okf export DIR|-                                        # the workspace as an Open Knowledge Format bundle; `ingest DIR` imports one
cargo run --bin quack -- embeddings refresh [-y]                                # refresh vectors a changed embedding model, width, or prefix left stale
cargo run --bin quack -- import sqlite:/path/src.db --table t --from orders [--types col=TYPE]   # snapshot a SQLite query, an http(s) data file (-H 'Name: value', --bearer-env VAR, --json-pointer /data), or s3://bucket/key as a table
cargo run --bin quack -- import ... --save NAME [--store-credential] | import list | refresh NAME | remove NAME   # import::SavedImport in _quack_imports; a refresh replaces the table only when the source changed; cron schedules it
cargo run --bin quack -- auth login|status|logout PROVIDER ; auth jwks [PROVIDER] [--rotate [--activate]]  # OAuth tokens; a client's public key
cargo run --bin quack -- auth register [--issuer URL] [--device-code|--token-env VAR|--open] [--replace] [--print] | unregister   # RFC 7591/7592 client registration
cargo run --bin quack -- config [--changed] [--format json]                      # every recognized setting, its value and origin, the file's unknown keys, the env vars read
cargo run --bin quack -- doctor [--offline] [--format json]                      # every check with its fix: config, data dir mode, workspace, model providers (probed), bind, vault key; exit 1 on a failure
cargo run --bin quack -- ready [URL] ; vault export-key [--to FILE] [-y]          # GET /readyz (the image's HEALTHCHECK); the vault key to a 0600 file or stdout
cargo run --bin quack -- user add|list ; token create|list|revoke ; member add|remove|list ; audit [-w ws --detail --format ocsf [--with-prompt]]   # server admin; --detail joins the workspace's own audit (OCSF ai_operation on queries)
cargo run --bin quack -- serve [--bind ADDR] [--local]                          # web UI, REST API under /api/v1 (its OpenAPI 3.1 document at /api/v1/openapi.json, rendered at /api/v1/docs), MCP under /mcp/v1/{workspace}; /readyz and /metrics (quack_core::telemetry) outside the limiter; [server].log_format = "json" for a collector; every provider request retried under [providers.NAME].max_retries / retry_backoff_ms (llm::limit::Attempts)
cargo run --bin quack -- mcp [-w ws] [--allow-write]                            # MCP server on stdio for Claude Code and editors
```

A saved question (`quack_core::saved`, design doc 8.1) is an answered turn's `run_sql`
read statements pinned under a name in `_quack_saved_questions`, visible to everyone who may
read the workspace; `quack saved run` runs them again without the model, each classified
again and under the agent's row cap and timeout, records in `_quack_saved_runs` a digest of
every result set and the row counts (never the rows; the run that produced them returns them
once), and says `changed` when any digest differs from the newest completed run that ran the
same statements. The digest (`WorkspaceDb::execute_query_digested`) is order-insensitive, the
sum of every row's SHA-256, since parallel DuckDB returns unordered rows in varying order
(`--refresh` asks the model again as a `PrintTurn`, writes denied, and pins the new answer's
statements). There is no scheduler: cron runs it, and `--exit-code` exits 5 on a change;
`-f` takes `-q`'s `QueryFormat` and default. The terminal's `/saved` verbs run as jobs
through `saved_cli` (`add` pins the session's last answer; `--refresh` and `--exit-code` are
parse errors there), and the REST routes under `.../saved` (`server/api/saved.rs`) list, save,
show, run, and remove, audited as `save`, `saved_run`, `open`, `list`, and `delete`; a run
answers directly, no job.

Turns are recorded in `_quack_sessions` / `_quack_messages` inside the workspace DuckDB
file (`quack_core::storage::sessions`); `-c` / `-r ID` replay history to the model through
rig's conversation memory (`llm::memory::History`: `SessionMemory` under `TranscriptWindow`,
rig's token window over `[analysis].history_token_budget`), and with
`[analysis].compact_history` the turns it leaves out become a chat-model summary kept in
`_quack_session_summaries`. Retrieval is hybrid (exact cosine scan plus quack's own
BM25 over `_quack_terms`, each document's chunks stemmed under the language
`WorkspaceDb::chunk_writer` detects once for it, reciprocal rank fusion in `WorkspaceDb::explain_search`;
no DuckDB extension is ever loaded, see design doc section 14), then an optional reranker
(`analysis::rerank`, `[retrieval].rerank = "none" | "model" | "reranker"`; `model` over-fetches
`rerank_candidates` and has the chat model order them, `reranker` has the dedicated rerank
model `rerank_model` score them through rig's `Rerank` at an OpenAI-compatible `/rerank`).
Each document's text is stemmed under the language `whatlang` detects at ingest
(`storage::workspace::{Language, Stemming, Analyzer}`, `_quack_documents.language`,
`[retrieval].languages = ["auto"]` or Snowball names), Chinese, Japanese, and Korean runs
become character bigrams, and a query is tokenized under every stemming in
`_quack_meta.languages`. Every hit carries its rank and score in each leg and the reranker
(`ChunkSearchResult::ranks`). One type, `analysis::search::DocumentSearch` (query, `top_k`,
documents, graph entity, `DocumentFilter`, mode `hybrid|keyword|vector`), is the search of
`search_documents`, REST and MCP `search`, `quack search`, `/search`, and the web Search
page (`/w/{id}/search`); a person can limit a question to documents (`document_ids` on REST
and MCP `query`, `-p --documents`, the chat's document picker): `DocumentScope` is resolved
when the turn starts, noted in the system prompt, recorded on the user message
(`UserMeta`), and intersected with the model's `document_ids`. Citations are registered per turn
(`analysis::citations`) and validated before the answer is returned. Sessions have a mode,
`chat` or `query`; `--mode` / `/mode` set it. The workspace context (owner-written
instructions, `quack_core::storage::context`, versioned in `_quack_context`) is injected
into the system prompt after the schema and documents, capped at `[context].max_tokens`;
the agent never writes it. The prompt states today's date (`PromptOptions::today`, the
system's local zone from jiff) right after the mode paragraph; tests pin it. Charts are `analysis::chart::ChartSpec` (bar, line, scatter,
pie; several series from several `y` columns or a `series_by` pivot, `stacked`; 200 distinct x
values and 8 series max), not ECharts. A `run_sql` or `create_chart` step keeps its first
`[analysis].step_result_rows` rows (`ToolStep::result`); `POST .../sql/export` and the SQL page's
download stream every row of a read statement (`WorkspaceDb::stream_query`).

Background work is asynchronous everywhere (design doc 4.1): `quack_core::jobs::JobQueue`
runs submitted jobs with optional lanes that keep submission order (a chat session is a
serial lane, so a follow-up waits for the answer before it; a workspace's uploads, graph
extraction, and document pass have their own), cancel tokens, per-chunk progress, and a
broadcast of `JobInfo` snapshots every interface reports from. `JobQueue::shutdown(grace)` is
the one way a process stops its jobs: a later `submit` is recorded as cancelled and never runs,
queued and running jobs are cancelled, and it waits up to the grace for them and for what
`when_ended` records. The terminal's quit calls it; `quack serve` calls it on SIGTERM or Ctrl-C
beside the HTTP drain under `[server].shutdown_grace_seconds` (20), after cancelling
`AppState::stopping` (which ends `jobs/stream` and the MCP event streams, and cancels an MCP
`query` turn, which runs under a child of it), and then
`AppState::close` drops the workspace handles so each writer checkpoints. There is no job pool:
resources are limited where they are used. Every rig HTTP client sends through
`llm::LimitedHttp` (rig's reqwest transport, plus the provider's `headers`), which holds
one permit of the process-wide gate for the provider and the model named in the request body (`[providers.NAME].max_concurrent_requests` each, 1
for Ollama, 8 otherwise) until the body or stream ends; a freed permit goes to interactive
requests (`TurnRequest::run`, `Embedder::embed_interactive`, via the `quack_core::priority` task-local) before
background ones. The same gate enforces a workspace's `allowed_providers`: `llm::egress::Egress`
is a task-local scope (`Workspace(AllowedProviders)`, or `NoWorkspace` for `quack doctor`'s probes),
entered by `Access::resolve` in the server, `OpenedWorkspace` on the command line, and
`McpServer::as_caller` per tool call, and carried into every job by `JobQueue::submit`.
`Egress::permit` is the one check, called when a model client is built from a `ModelRef` (a typed
`Error::ProviderRefused` holding the `egress::Refusal`, and `Error::is_provider_refusal` is what
the audit outcome and a turn's failure kind ask: 403 and a `denied` audit row through
`Access::model`, exit 1 from the CLI) and again by `ProviderGates::permit` for every request,
Bedrock's SDK requests included. A request with no scope is `Error::ModelRequestUnscoped`, never an
allow, so a test that makes a model request enters a scope first. Under a restricted list an Ollama
model whose id ends in `-cloud` or `:cloud` is refused. Every outbound HTTP client takes its proxy from `quack_core::proxy::Proxies`
(`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY`, read once; loopback and `169.254.0.0/16`
are never proxied): reqwest clients start from `Proxies::client` (`.clippy.toml` disallows
`reqwest::Client::builder` and `new`), and the AWS SDK connector takes `Proxies::aws`
(`docs/providers.md`). Bedrock is two provider types, one per endpoint
(`bedrock` for bedrock-runtime, `bedrock-mantle`; they host different models; `llm::bedrock`,
settings in `config::bedrock`), each with an `api` (`converse`, `bedrock` only,
through rig-bedrock and the AWS SDK, whose HTTPS client is wrapped to take the same gates for
`/model/{id}/` requests; `chat-completions` and `responses` through rig's OpenAI clients over a
`LimitedHttp` that SigV4-signs each request after its permit; Responses always sends
`store: false`). Credentials come from `aws_config::defaults` (env, `aws_profile`, SSO,
instance roles; `auth = "aws"`, the type's default); `base_url` is the endpoint root (a VPC
endpoint), checked against the type; one cached `Session` per provider. The registry is in memory only (labels can be workspace content).
The terminal is one async loop (`tokio::select!` over crossterm's `EventStream`, one
`AppMsg` channel, the job broadcast, a spinner tick) and submits every question,
statement, file, import, and ontology or graph verb as a job, so it never blocks its input:
a strip above the prompt shows active jobs, `/jobs` lists them, `/cancel N` stops one, and
write prompts from concurrent work queue up. A left-button drag over the transcript
selects text in transcript lines (`terminal::selection`), and releasing copies it without row
markers or wrap breaks to the system clipboard (`arboard`) and the terminal's (OSC 52;
`terminal::clipboard`). Its commands' database steps run in the order
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
(`graph_cli::RenderOnWriter`); file parsing runs on the blocking pool, so every job, the
terminal's included, is a plain task on the runtime. Tests that also read through a
`WorkspaceDb` give the pipeline a writer over its `try_clone_reader` connection.

The agent turn is an event stream (`quack_core::analysis::events`): text deltas, tool
started/finished with timing, permission requests, turn complete. Every interface consumes
it. Two rig hooks (`analysis::hooks`) keep a turn going: an unknown tool name is repaired or
retried with the real names (twice at most), and an empty reply is asked for once more;
`tests/agent_turn.rs` drives whole turns with rig's scripted model. The tools hold only the
workspace and its settings; a turn's recorder, write policy, refusal flag, and chart and graph
results are one `analysis::tools::Turn`, handed to every call as a runtime scope of rig's
`ToolContext` (`.tool_context(turn.context())` on the run). `--allow-write` lets the agent run mutating SQL without asking; otherwise the terminal
prompts y/n/a and `-p` refuses and exits 3. Once a turn has retrieved document or graph text
(`search_documents`, `search_graph`, `find_path`, `always_retrieve`, a `describe_class` that names
example entities; `Turn::read_documents`),
no write of that turn runs unasked, whatever was allowed: `WritePolicy::decide` asks where the
interface answers permission events (`Allow(Approver::Person)`: the terminal, a streamed turn)
and refuses where it cannot (`Allow(Approver::Nobody)`: print mode, non-streamed REST, MCP), and
the reason (`policy::Hold`) goes on the prompt, the `permission_required` event's `reason` (with
`Hold::notice`, the sentence the card shows, as `notice`), and
the refused step. Pinned documents, the workspace context, table rows, and the listing tools do
not count, so a pinned document or a table cell can still dictate a write under allow-write.
Every parser yields `parser::Section`s with a `kind` (`body`, `table`, `note`, `code`) and an
optional `locator` beside the page (`line 40`, `12:04`, `chapter 3`, `message 2`), stored on
`_quack_chunks` and rendered by every citation label (`analysis::citations::ChunkLocation`);
tables are `ingestion::table::Table` rendered as pipe Markdown, chunked by rows with the header
repeated, and loaded as document-owned workspace tables past `[ingestion].table_rows_as_table`.
`parser::DocumentMeta` (author, dates, tags, other named values) comes from each format and
lands on `_quack_documents`; `NewFile::fields` (the uploader's `DocumentFields`) wins over it.
Chunk bodies, graph source excerpts, and pinned text reach the model inside `text::Fenced`
markers, whose code is a digest of the enclosed text (so the text cannot close its own block),
after a fixed sentence that it is data; an `always_retrieve` chunk is fenced too, as the `content`
string of the JSON object rig prints, with no sentence before it; the system prompt's trust
paragraph precedes the permission rules; filenames, titles, headings, labels, and graph property
values render through `text::OneLine`. Provider construction lives in `quack_core::llm`; interfaces never build rig
clients themselves. Every chat model is wrapped in `llm::sampling::Sampled`: only Ollama's API
gets a `temperature`, Claude gets `max_tokens` 64,000, and `[analysis].effort` /
`background_effort` go out as each API's field (an unrecognized model id gets it on Chat
Completions and Responses); `temperature`, `effort`, and `background_effort` on
`[providers.NAME]` or `[providers.NAME.models."ID"]` (`config::ModelSettings`, resolved by
`Config::model_settings`: model, then provider, then `[analysis]`) override that. `type = "openai"` takes
`api = "responses"` or `"chat-completions"`; unset, `OpenAI` itself gets Responses and a `base_url`
server gets Chat Completions. OAuth providers (`quack_core::llm::oauth`) hand out a bearer through
one shared `TokenManager` per provider: PKCE or device-code login via `quack auth` (or the
client-credentials grant, which needs no login; or `on-behalf-of`, which exchanges the
requesting person's own token per request, the person carried in the `llm::acting::Acting`
task-local and taken from `oidc::SubjectTokens`), the token sealed by `quack_core::vault` in
`control.db` (`provider_tokens`; the vault key in the OS keychain or a 0600 `vault.key`, made
under a `vault.key.lock` file lock so every process agrees on one key),
silent refresh, and `Error::AuthRequired` (exit 4 from every command that reaches a provider) when no flow can run.
A client with `client_auth = "private_key_jwt"` (a provider or `[server.oidc]`) signs a new ES256
client assertion for every token-endpoint and PAR request with a vault-sealed P-256 key in
`control.db` (`client_keys`, `llm::oauth::client_key`, one per issuer and client id; `quack auth
jwks` prints it; `--rotate` stages a replacement under `next <issuer> <client_id>` and gives the
issuer both, `--rotate --activate` swaps it in within one transaction and gives the issuer the new
key alone), and a browser sign-in is pushed first (RFC 9126) whenever discovery lists a PAR endpoint.
`quack auth register` (`llm::oauth::registration`) registers one such client per issuer through
RFC 7591 for every section there that leaves `client_id` out, keeps its `client_id`, sealed
`registration_access_token`, and `registration_client_uri` in `control.db` (`client_registrations`,
named by the issuer), and those sections resolve their `client_id` from it at use; the key waits under
the issuer's name until the id exists. By default, wherever discovery advertises a
`registration_endpoint`, public clients, and PKCE `S256` (as Vouch does), it first signs the person in through a temporary public client it registers and deletes again
(`Registrar::register_signed_in`), so the registration's bearer is the person's own and Vouch
records them as the owner; widening the client to the organization stays a manual console step. Each
temporary client stays recorded (sealed) until deleted, so an interrupted run leaves it deletable
(the next run deletes it; `quack doctor` names it). An issuer that advertises less needs
`--token-env` or `--open`. RFC 7592 carries
each rotation step's key set (`Registrar::publish_keys`, a full-metadata `PUT`) and deletes the
client (`unregister`).
Every interface returns one response object, `AgentResponseBody` from `AgentResponse::body` (answer, citations with
labels and the document's `ingested_at`, queries, steps, graph, chart, `write_refused`, `cancelled`, `usage`, `duration_ms`, `session_id`); a write refused
inside a turn is `write_refused: true` (REST 200, MCP structured content, print exit 3). A streamed
web or REST turn from someone who may write asks instead: a `permission_required` SSE event, answered
by `POST .../sessions/{sid}/permissions/{request}` (`server::permissions`, held in memory, refused after
`[server].permission_timeout_seconds`, 410 and a denied row when the turn had already stopped waiting, every answer audited as `permission`). `usage` is `AgentResponse::usage`, the provider's own
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
The ontology page's forms edit a property's meaning and the measures (`ontology::edit::Edit`,
applied to the current version and saved through `store::save`).
A class or relation id changes only through `ontology::store::rename` (`quack ontology rename`,
`POST .../ontology/rename`, the ontology page's Rename form): `IdRenames` on the save's
`Revision` moves, in the save's transaction, the ontology's own references, `since_version`,
undecided candidates, and the graph's `class_id` and `relation_id`; an id that already exists
is refused, and earlier snapshots keep the old id. An import that swaps one id for another is
a removal plus an addition; `Ontology::diff` does not guess renames.
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
`WorkspaceDb::sample_chunk_ids`, reading text a page at a time) to the chat model (`llm::graph_extractor`, a `llm::SchemaCall` whose preamble is
`Ontology::extraction_prompt` and whose structured-output schema is `Ontology::extraction_schema`,
the ontology's class and relation ids enumerated) and validates the answer against the ontology, counting unknown
classes and relations as drift in `_quack_meta.graph_drift`; `graph::resolve` embeds
node labels, merges near-identical labels of one class, and queues the rest as merge
proposals; `graph::traverse` resolves an entry point (exact label, alias, then embedding)
and walks neighborhoods, shortest paths, and classes with subclass expansion, bounded by
`[graph]`. Every interface asks through `graph::query::{GraphQuery, PathQuery}`: `new`
trims and defaults the caller's fields, `run` checks class and relation ids against the
ontology and resolves the entry points, and an unresolved path end is an `UnknownEntity`
naming the closest labels. `graph::store::status` reports size, `provisional` (the newest ontology version
was auto-accepted), `stale` (`graph_built_with_ontology_version` lags), pending merges,
drift, `pending_chunks` (chunks no extraction read), and `pending_tables` (mapped tables
whose fingerprint in `_quack_graph_tables_built`, written by each mapping's last batch,
no longer matches: owning document, keyed row count, hash over the mapped columns);
`[graph].follow_ingest` (`off`, `tables`, `all`) has `graph::follow_up::after_documents`
extract a document into the graph as it becomes ready (the server queues it as an audited
graph run after an upload or import; the CLI and terminal run it after their ingest). A
person's writes go through `store::{create_node, update_node, delete_node, create_edge,
delete_edge}` (`quack graph add|set|delete`, the `.../graph/nodes` and `.../graph/edges`
routes audited as `graph_edit`, the graph page's forms), each checked against the ontology
and recorded as `Origin::Manual` provenance (`author`, `note`, `asserted_at`), rendered
everywhere as `asserted by {author}: {note}`; `graph extract --reset` keeps them unless
`--all`; `revalidate` drops what the current ontology no longer allows, and
`Revalidation::preview` counts that first (totals, per class id, per relation id) so
`quack graph revalidate` asks before dropping (`-y` skips; with nobody to ask it fails,
`Confirm::ask_to_drop`), and the graph page and `POST .../graph/revalidate` send back the
preview's totals (`api::graph::DropApproval`; 409 with the current totals when they are absent or
stale). `GET .../graph/revalidate` serves the preview, and the run deletes what the preview
counted, since both start from `Revalidation::find`. The agent
registers `search_graph` and `find_path` only when the graph has nodes, query mode drops
provisional results, and every response shape carries the turn's `graph` results. Their
rendering (`Display for GraphResult` in `graph::traverse`, shared with `quack graph` and the terminal)
carries each node's and edge's typed properties, bounded; a class or relation id the
ontology does not define is refused with the ids that do exist, a name that matches no
entity comes back with the closest labels (`traverse::suggest_entities`), and a result
query mode emptied by dropping provisional nodes says so rather than claiming the graph
is empty. A result cut short by `max_nodes` says so too, and a class listing carries the
total it was capped from (`GraphResult::total_nodes`, `truncated`). Every result carries
`status` (`GraphStatusSummary`: versions, `stale`, `provisional_nodes`, `drift_total`,
`dropped_provisional`), filled by `GraphQuery::run` and `PathQuery::run` (`store::summary`)
and counted by `without_provisional`, so every interface's `graph` says how current it is;
the prompt's graph line names the drift count. `graph::export::GraphExport` writes the whole
graph as a CSV bundle (to a directory, or a tar whose parts are staged in
`WorkspaceDb::spool_file`), GraphML (`quick-xml`), or JSON-LD, each part streamed from one
ordered statement; `GET .../graph/export?format=` streams it through `api::okf::Download`
(shared with the OKF export) and audits `export` with the counts when the stream ends. Provenance to a mapped
table renders as a predicate `run_sql` can run, since the mapping knows the key column. The
tool guidance in the system prompt gains a numbered graph procedure whenever those tools
are registered.

What the agent knows of a table (issue #403): `storage::profile::TableProfile` (per column
present and distinct counts, three common values, the share of a text column that casts to a
number or a date) is stored in `_quack_table_profiles` at load, import, and after any write that
changed the row count (`TableProfile::after_write`, called by every interface that runs a write),
and shown only while its row count matches; `profile::ColumnWarning` (all empty, half empty,
numbers or dates stored as text, a repeating key) is worked out when read, with the
`ColumnType` that fixes it when every value converts (`profile::Retype`, a strict `CAST`;
`--types` at ingest and import). Owners' notes are `profile::TableNote` (`_quack_table_notes`).
The ontology's `Property` carries `description`, `unit`, `synonyms`, and `Ontology::measures`
(a SQL expression over one table, checked as a read at save); `WorkspaceDb::describe_table`
returns all of it on `TableDescription` (`TableDescription::body`, a `TableDescriptionBody`, is the one REST, MCP, and CLI shape). Past
`table_search::DETAILED_TABLES` (25) user tables, `find_tables` registers and the prompt ranks
tables against the question after the workspace context (`analysis::table_search`: a card per
table, BM25 with `tokenize`, plus cosine over card vectors in `_quack_table_cards` refreshed at
turn start, fused by RRF). The graph is readable in SQL through `graph::views` (issue #406):
`graph_<class>` with one typed column per property and `graph_edges`, made by `views::ensure` at
every ontology save and open, marked by a comment; `table_search::user_tables` is
`list_tables` without them. A write naming `graph_` is refused (`views::write_names_reserved`),
and ingest and import refuse `graph_` and `_quack_` table names (`TableName::check_unreserved`).

Every embedding goes through `quack_core::embedding::Embedder` as an `Input`, which names
its role: `Query` (search), `Document { title, text }` (a chunk under its heading), or
`Similarity` (entity labels and names, ontology type names). It adds the input prefixes the
model family was trained with (`presets::Family::of`, from each model card; `[embedding]`
overrides any role, `ResolvedPrompts::for_model`) and returns `Vector`s checked against the
profile's `Dimension`; `.clippy.toml` disallows calling a model directly
(`embedding::EmbeddingModel::embed_texts`, rig's `DynModel::call` and `Model::call`).
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
like the graph tools themselves. `read_document(document, from, limit)` (`tools::ReadDocumentTool`,
registered in every workspace beside `search_documents`) returns a document's chunks in order
from a position, numbered through the turn's citation registry so `[n]` markers on them
validate, within `[retrieval].pinned_token_budget` (at most 50 chunks a call), and counts as
reading document text for the write rule. Every `Citation` carries `excerpt`, the chunk's
first 500 characters; the web chat links each citation to the passage page
`/w/{id}/documents/{doc}/chunks/{n}` (`templates/passage.html`, previous and next chunk linked),
and `GET .../documents/{doc}/chunks?from=&limit=` pages a document's chunks over REST, both
through `api::documents::read_chunks` (`WorkspaceDb::document_chunks`), audited as opening the
document. Ollama embedding requests go through `llm::OllamaEmbedder`,
not rig's client, so they carry `keep_alive` and a chunk-sized `num_ctx`.

Every command resolves its workspace through `ControlPlane::workspace_or_default`: `-w NAME`
must name a workspace that exists (`Error::NoWorkspaceNamed`, exit 2, nothing created), and
with no `-w` the `[general].default_workspace` is created, audited, on first use. A new
workspace's name is a `storage::control::WorkspaceName` (trimmed, non-empty, no `/`, `\`, or
`.`), the one rule `quack workspace create`, the API, the web console, and the config key share.

Server access control lives in `quack_core::storage::control`: users (argon2id), workspace
membership with `Role` (viewer, member, owner), API tokens stored as SHA-256 hashes with
`Scope`s, and the append-only access `audit_log` (`AuditEntry`, `query_audit`); a change to
users, workspaces, membership, or tokens takes its `AuditEntry` and commits the row in the
same transaction, so no change stands unaudited; the content
half of each audit row is `storage::audit` (`_quack_audit`) inside the workspace under the
same UUID v7. Ingestion is `register_document` (status `queued`, or `Registration::Duplicate` when the
bytes' SHA-256 already belong to a non-failed document whose table or chunks still exist;
`Error::TableTaken` when the file's table belongs to another live document) plus `Processing::run`
(`processing` to `ready` or `error`, recording `chunk_count`, the parsed title, and a PDF's
`parser::PageCounts`: pages in the file, pages whose extraction failed, pages without text;
`PageCounts::note` is the one wording every interface shows, as `3 of 40 pages unreadable`);
`ingest_file` does both and takes a `NewFile` (name, bytes, `DocumentSource`, optional
title and uploader, its `RunControl`, and `replaces`, the ready document this file takes
the place of: `quack ingest --replace [ID]`, `POST .../documents?replace={doc}`, the
Documents row's Replace control). The old document keeps serving, marked `superseded_by`,
until the new one is `ready`; then it becomes `DocumentStatus::Superseded` and its pin moves
over (`WorkspaceDb::finish_replacement`); a failure clears the mark (`mark_document_error`).
A table file loads over its predecessor's table. A superseded document leaves search, the
prompt, `list_documents`, and `read_document` (`WorkspaceDb::list_documents` is live rows;
`list_all_documents` has them all for `quack docs --all` and the page's `?all=true`), but keeps
its chunks so stored citations still open.

The MCP server (`crates/quack/src/mcp.rs`, `rmcp`) exposes `query`, `search`, `sql`,
`list_tables`, `describe_table`, `list_documents` and the `quack://workspace/...` resources;
`quack mcp` serves it on stdio (unaudited, like the CLI) and `server/mcp_http.rs` serves it
at `/mcp/v1/{workspace}` behind `Access::resolve`, one transport per workspace, user, and write
permission, audited with channel `mcp`. With `[server.oidc].audience` set, the API and MCP also
accept the issuer's access tokens (`SignIn::verify_bearer`, `jsonwebtoken` on aws-lc-rs), publish
`/.well-known/oauth-protected-resource` (`server::resource`), and send `WWW-Authenticate:
Bearer resource_metadata=...` on 401, so MCP clients can sign users in themselves.

External data comes in through `quack_core::import` (`quack import`, `POST .../import`, the
Tables page form, `/import` in the terminal): a query runs on a SQLite file, opened read-only
by path, with every column cast to text through sqlx, or a CSV, Parquet, JSON, or workbook
file is fetched over HTTP(S) (with `import::SourceHeader`s: given ones, or a bearer token read
from `--bearer-env` at fetch time; `JsonPointer` picks the rows of an enveloped JSON) or from S3
(`import::s3`, a GET signed by `llm::bedrock::Signer::s3` with the AWS SDK's credentials), and
the rows load through the normal ingestion path as a document with source `import` and the
redacted URL as title. S3 and `--bearer-env` use the server's own credentials, so `quack serve`
with logins refuses them (`Error::ServerCredentials`, 403) unless
`[import].allow_server_credentials`. No `ATTACH`: the workspace never reaches out at query time.

`quack serve` (`crates/quack/src/server/`) is a thin axum client of core: `auth.rs` turns a
bearer (login session or API token), the session cookie, or `--local` into an `Identity`
(with `[server.oidc]`, people can also sign in through the organization's issuer:
`quack_core::oidc::SignIn`, `server::oidc`, and `web::sign_in`; a first sign-in creates a
user with no memberships, the user's refresh token is kept HPKE-sealed in `control.db`
(`oidc::UserTokens` over `quack_core::vault`, whose HPKE key is in the keychain),
and a session whose token has run out renews it; an issuer's refusal, seen by a renewal or
an on-behalf-of exchange alike, is handled once in `SubjectTokens::refreshed`: the stored and
presented tokens go, every session ends, and one denied `session` row is written),
and `Access::resolve` resolves the workspace, checks role and token scope, and writes the denied
audit row itself, so a handler holding an `Access` is already authorized. Both login paths
go through one `auth::password_login`, and a browser session expires at
`[server].session_max_age_hours` or after `session_idle_minutes` unused, whichever is first
(`quack_core::web_sessions::WebSessions`); its cookie is `HttpOnly`, `SameSite=Lax`, `Max-Age`d to the
absolute lifetime, and `Secure` unless the request came from loopback. One `tower_governor`
limiter covers the web UI, the API, and MCP, with a tighter one on the two login routes and
none on `/healthz` (design doc 12). The same routes carry `no-store` cache headers
(`server::no_store`) unless the handler set `Cache-Control` itself, as the static assets
do. Every
workspace-touching handler then records the allowed row plus its `_quack_audit` detail
through `Access::audit`; one that changes the control plane commits the row with the change
(`Access::entry`) and writes only the detail after (`Access::record_detail`).
A handler's reads go through `App::read` (a reader-pool
connection in a read-only transaction, so a read never occupies the writer and a write
slipped into one is refused); its writes go to the writer through `state::with_db`; and
its `_quack_audit` detail row goes to the workspace's insert-only audit connection
(`storage::audit::AuditLog`, a writer clone on its own thread), so no request waits for a
write in progress just to record itself. `query/stream` forwards the agent event stream as SSE (`text`,
`status`, `tool_started`, `tool_finished`, `permission_required`, `complete`, `error`); uploads return 202 with a `job` id
and run on the work queue in a lane of `[server].workers_per_workspace` per workspace
(`queue.rs`), which locks the workspace only around each database step and keeps each queued upload's bytes on disk in the workspace's `uploads/` until its job ends; `api/jobs.rs`
serves `GET .../jobs`, `.../jobs/stream` (SSE), `.../jobs/{job}`, and `POST .../cancel`,
and `/w/{id}/jobs` is the web console's Jobs page. The web UI (`server/web/`, `templates/`, `static/`) is askama pages over
the same `Access::resolve` checks and the API's `Access` operations; `WebUser` redirects to `/login` instead
of a 401; the built Tailwind CSS is committed (`make css-build` after template edits),
htmx, ECharts, and Redoc are vendored, and the SQL page's CodeMirror editor is bundled from
`crates/quack/editor/` (`make editor-build`, bundle committed; `docs/web-ui.md`). The API's
contract is `server/api/openapi.rs`: an OpenAPI 3.1 document generated with `utoipa` from a
`#[utoipa::path]` on every handler and `ToSchema` on every request and response type (core
types included), served at `/api/v1/openapi.json` and rendered by Redoc at `/api/v1/docs`,
both beside `/healthz` (no sign-in, no audit, no limiter). A new route goes through
`api::ApiRoutes::route` and needs its annotation in `ApiDoc`'s `paths`; the tests in
`api/openapi/tests.rs` fail otherwise, and also when a method the router answers is not
documented. Parameters come from the handler's signature through utoipa's `axum_extras`:
a tuple `Path<(..)>` is named from the route template, a lone `Path<WorkspaceId>` or
`Path<UserId>` and every query struct go in `params(...)` (the ids implement `IntoParams`
for their segment), and `Query<T>` supplies the location, so no `parameter_in` is written;
a test fails when a path segment or a query field is missing. Every API error is `{"error", "code"}` with a stable `server::error::ErrorCode`
(`ErrorCode::of` maps each core error, `of_turn` each `FailureKind`); the SSE `error` event
carries the same JSON, and `error::coded_errors` gives the framework's own errors under
`/api/` (rejections, 405, 429, 504) the same body. Tests drive the router with
`tower::ServiceExt::oneshot` and no model. The full CLI (`quack -p`, `quack serve`, `quack mcp`,
`quack ontology propose`, ...) is specified in design doc section 11.

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

Run a specific test:

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
by pushing an annotated `vYYYY.M.N` tag (`v2026.10.3`) after the branch is pushed and `docs/upgrading.md` has the tag's section (the release notes lead with it, then git-cliff's grouped commits); `workflow_dispatch` builds
everything and publishes nothing.

## Where to read more

`docs/architecture.md` maps the design to the modules; `docs/migrations.md`,
`docs/crypto.md`, `docs/web-ui.md`, and `docs/ci-cd.md` cover the schema, TLS, web UI, and
release layers, `docs/authentication.md` covers signing in to quack, and `docs/providers.md`
covers connecting to and authenticating with model providers. `docs/audit.md` lists every
logged event type (a test keeps it in step with `AuditAction`), `docs/operations.md` the
erase procedure and hardware sizing, `docs/upgrading.md` each release's upgrade steps (the
version-bump PR adds the section; the release notes lead with it), `docs/compliance/` the
control matrix, and `docs/contributing/` the recipes for a new parser, agent tool, or provider
type. Read the relevant one before changing that layer.
