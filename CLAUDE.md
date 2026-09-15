# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It also drives the `rust-agents` Claude Code plugin (conventions live in `.claude/rules/`).

## What this is

A knowledge engine with many interfaces, built in Rust. A workspace holds documents
(vectorized), tables (DuckDB), and an ontology-backed knowledge graph; one agent answers
across all three and shows every action. The core is a library; the web UI, REST API, MCP
server, terminal session, print mode, and desktop window are thin clients of it, all
subcommands of one `quack` binary with no Cargo features. The
design is `docs/design-doc.md`; read it before any non-trivial change, and check its
section 17 for where the code still lags. The chosen stack:

- **Workspace** of crates under `crates/` (edition 2024, resolver 3, MSRV 1.98.0)
- **DuckDB** (via `duckdb-rs`), one file per workspace, holding everything classified
  about that workspace: user tables, chunks, graph, ontology, context, sessions, audit
  detail (all internal tables prefixed `_quack_`)
- **SQLite** (via `sqlx`) for `control.db` in server mode: users, workspaces, membership,
  tokens, and the mandatory append-only access audit log; nothing workspace-revealing
- **sea-query** for type-safe SQL generation against `control.db`; bound parameters for
  DuckDB internals
- **rig** for LLM providers (Ollama, OpenAI-compatible, Anthropic) and the agent loop
- **aws-lc-rs** as the single crypto/TLS provider (never OpenSSL or `ring`)

## Repository layout

```
Cargo.toml            # virtual workspace: deps menu + strict lints + profiles
.clippy.toml          # clippy tuning (levels live in Cargo.toml)
.rustfmt.toml         # stable-only formatting
deny.toml             # advisories, license allow-list, OpenSSL/ring bans
rust-toolchain.toml   # pinned 1.98.0 + rustfmt + clippy
Makefile              # build / fmt / lint / test / deny
crates/               # quack-core (engine), quack (the binary) — see crates/README.md
docs/                 # design-doc.md (the product) plus the stack patterns, with code
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
cargo run --bin quack -- ingest sales.csv -w ws                                # file -> table or chunks
cargo run --bin quack -- -p "question" -w ws [-f text|json]                    # one agent turn; steps on stderr
cargo run --bin quack -- -w ws                                                 # terminal session (needs a TTY)
cargo run --bin quack -- sessions | export ID [--sql]                          # sessions live in the workspace file
cargo run --bin quack -- ontology show|init|import|export|versions|diff|restore   # the graph schema, versioned in the workspace
cargo run --bin quack -- auth login|status|logout PROVIDER                      # OAuth token for an auth = "oauth" provider
cargo run --bin quack -- user add|list ; token create|list|revoke ; member add|remove|list ; audit   # server admin
cargo run --bin quack -- serve [--bind ADDR] [--local]                          # web UI and REST API under /api/v1
```

Turns are recorded in `_quack_sessions` / `_quack_messages` inside the workspace DuckDB
file (`quack_core::storage::sessions`); `-c` / `-r ID` replay history to the model, trimmed
to `[analysis].history_token_budget`. Retrieval is hybrid (exact cosine scan plus quack's own
BM25 over `_quack_terms`, reciprocal rank fusion in `WorkspaceDb::search_hybrid_chunks`;
no DuckDB extension is ever loaded, see design doc section 14); citations are registered per turn
(`analysis::citations`) and validated before the answer is returned. Sessions have a mode,
`chat` or `query`; `--mode` / `/mode` set it. The workspace context (owner-written
instructions, `quack_core::storage::context`, versioned in `_quack_context`) is injected
into the system prompt after the schema and documents, capped at `[context].max_tokens`;
the agent never writes it. Charts are `analysis::chart::ChartSpec` (bar, line, scatter,
pie; 200 points max), not ECharts.

The agent turn is an event stream (`quack_core::analysis::events`): text deltas, tool
started/finished with timing, permission requests, turn complete. Every interface consumes
it. `--allow-write` lets the agent run mutating SQL without asking; otherwise the terminal
prompts y/n/a and `-p` refuses and exits 3. Provider construction lives in `quack_core::llm`; interfaces never build rig
clients themselves. OAuth providers (`quack_core::llm::oauth`) hand out a bearer through
one shared `TokenManager` per provider: PKCE or device-code login via `quack auth`, an
AES-256-GCM cache under `<data_dir>/tokens/` keyed from the OS keychain or a 0600 key file,
silent refresh, and `Error::AuthRequired` (exit 4 in `-p` and `ingest`) when no flow can run.
The ontology (`quack_core::ontology`, design doc 6.3) is classes with single inheritance
from `entity`, relations with a domain and a range, typed properties, and table mappings.
It lives in the `_quack_ontology_*` tables; `ontology::store::save` validates, checks
mapped tables and columns against the workspace, and writes a new version with a JSON
snapshot, `since_version` carried over for items that already existed. JSON is the only
interchange form (export, import, `PUT /ontology`); a file is never the source of truth.
The system prompt carries a compact rendering when an ontology exists.
Server access control lives in `quack_core::storage::control`: users (argon2id), workspace
membership with `Role` (viewer, member, owner), API tokens stored as SHA-256 hashes with
`Scope`s, and the append-only access `audit_log` (`AuditEntry`, `query_audit`); the content
half of each audit row is `storage::audit` (`_quack_audit`) inside the workspace under the
same UUID v7. Ingestion is `register_document` (status `queued`) plus `process_document`
(`processing` to `ready` or `error`); `ingest_file` does both.

`quack serve` (`crates/quack/src/server/`) is a thin axum client of core: `auth.rs` turns a
bearer (login session or API token), the session cookie, or `--local` into an `Identity`,
and `access()` resolves the workspace, checks role and token scope, and writes the denied
audit row itself, so a handler holding an `Access` is already authorized. Every
workspace-touching handler then records the allowed row plus its `_quack_audit` detail
through `Access::audit`. `query/stream` forwards the agent event stream as SSE (`text`,
`tool_started`, `tool_finished`, `complete`, `error`); uploads return 202 and are processed
by `queue.rs`, one bounded lane per workspace, which locks the workspace only around each
database step. The web UI (`server/web/`, `templates/`, `static/`) is askama pages over
the same `access()` checks and the API's helpers; `WebUser` redirects to `/login` instead
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
make deny      # cargo deny check
make help      # list targets
```

Run a specific test (once at least one crate exists):

```bash
cargo test -p <crate> <test_name>    # one test (name filter) in one crate
cargo test -p <crate>                # all tests in one crate
cargo test --workspace <test_name>   # name filter across the workspace
```

> Coverage and mutation testing are local-only: `make test-coverage`, `make test-mutants`.

## Where to read more

The migration, query, and crypto patterns each have a doc under `docs/` (see the table in
`README.md`). Read the relevant one before implementing that layer.
