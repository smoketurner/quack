# Architecture

How the code is laid out. The product design is [design-doc.md](design-doc.md); this file
maps its sections to the crates and modules that exist.

## Workspace

A virtual Cargo workspace (`Cargo.toml` has no `[package]`) with two members under
`crates/`: `quack-core`, the library, and `quack`, the one binary. Shared settings come
from the root: `[workspace.package]` (edition 2024, the MSRV), `[workspace.dependencies]`
(every dependency pinned to an exact version with default features off; members opt in),
and `[workspace.lints]` (panic, cast, and arithmetic denies plus clippy `pedantic`). Every
member declares `[lints] workspace = true`. There are no Cargo features: every build
contains every surface.

## Layering

Every interface is a thin adapter over the core. Nothing above the core line owns
behavior; nothing below it knows about HTTP, terminals, or windows.

```
      +------------+  +------------+  +------------+  +------------+  +---------------+
      |  web UI    |  |  REST API  |  |    MCP     |  | TUI/print  |  | quack desktop |
      |  (askama,  |  |  (axum)    |  | (stdio,    |  | (ratatui,  |  | (planned, #35:|
      |   htmx)    |  |            |  |  HTTP)     |  |  clap)     |  |  Tauri window)|
      +------------+  +------------+  +------------+  +------------+  +---------------+
                        all subcommands of the single `quack` binary
             \               |               |               |                /
              +--------------+---------------+---------------+---------------+
                                             | in-process calls
                              +--------------v--------------+
                              |         quack-core          |
                              +-----------------------------+
```

## `quack-core`

| Module | Owns | Design doc |
|---|---|---|
| `config` | `config.toml` with every section (`[general]`, `[providers.*]`, `[ingestion]`, `[embedding]`, `[retrieval]`, `[context]`, `[analysis]`, `[server]`, `[ontology]`, `[graph]`, `[import]`, `[jobs]`), unknown keys rejected, `QUACK_*` overrides | 13 |
| `config::inspect` | the same file read outside `Config::load`: every recognized setting with the value in force and its origin, the file's unrecognized keys, the environment variables read (`quack config`) | 13 |
| `crypto` | installs the aws-lc-rs provider once | 14 |
| `doctor` | `quack doctor`'s checks over a `config::inspect` result: config file, crypto module, data directory mode, `control.db`, the workspace, each model's credential and a model-list probe of its provider, the server bind; creates nothing | 11.5 |
| `error` | the `thiserror` enum every layer returns | |
| `storage::control` | `control.db` (SQLite, sea-query): users, workspaces, membership, tokens, the append-only access `audit_log` | 5.5, 12 |
| `migrations/*.sql` | `control.db` schema versions | [migrations.md](migrations.md) |
| `storage::workspace` | the workspace DuckDB file: open with confinement and limits, the `_quack_` tables, statement classification, hybrid retrieval (cosine scan plus BM25 over `_quack_terms`), the document registry, query execution with faithful JSON values | 5.4, 6.1, 7.4 |
| `storage::writer` | the workspace's one writer connection as an actor: a thread of its own runs the closures sent to it, interactive before background; callers await `run` / `run_at` | 4.1, 7.4 |
| `priority` | the interactive/background task-local that the writer line and the model limiter read | 4.1 |
| `storage::sessions`, `storage::context`, `storage::audit` | conversations, the versioned workspace context, the content half of the audit (`AuditLog`: the server's insert-only audit connection) | 5.3, 8 |
| `embedding` | the role every text is embedded in (query, document, similarity), the prefixes each model family was trained with (`presets`) and their `[embedding]` overrides, the profile a vector is made under, the width check, and `refresh` | 6.1, 5.4 |
| `ingestion` | registration with SHA-256 dedup, parsers (`parser`, `html`, `office`, `xlsx`), chunking, embedding, tables from structured files, piped stdin | 6.1, 6.2 |
| `import` | rows from Postgres, SQLite, or an HTTP data file as a workspace table | 6.2 |
| `analysis` | the agent loop as an event stream (`agent`, `events`), the tools (`tools`), the system prompt (`text_to_sql`), write policy, citations, the chart spec, the reranking hook (`rerank`) | 7, 9 |
| `ontology` | the model, validation, versions (`store`), induction from tables and documents (`induction`, `documents`), the review queue (`candidates`) | 6.3, 6.5 |
| `graph` | the knowledge graph: `store`, `tables` (mapping extraction), `extract` (constrained model extraction with drift), `resolve` (merges), `traverse` | 6.4 |
| `ocsf` | access-audit rows rendered as OCSF 1.9.0 events | 12 |
| `okf` | Open Knowledge Format bundles in and out | 17 |
| `extraction` | what both extraction runs share: the `Extract` trait, lenient JSON answers, concurrent calls with per-chunk progress (`RunProgress`), even sampling across documents, name counts (`Tally`) | 6.4, 6.5 |
| `progress` | the per-chunk progress report the extraction runs make to their caller | 6.5 |
| `jobs` | the work queue every interface submits background work to: ordered lanes, cancel, progress, a broadcast of job snapshots | 4.1 |
| `llm` | rig provider construction over `limit::LimitedHttp` (each provider's process-wide request limit), `TurnRequest` (one agent turn), `OneShotAgent` (one tool-less prompt, streamed: extraction and reranking), OAuth token management (`oauth`) | 4.1, 10 |

## `quack`

| Module | Owns |
|---|---|
| `main` | the clap command tree, print mode entry, the workspace-local subcommands (`ingest`, `docs`, `sessions`, `export`, `context`, `import`, `okf`, `auth`) |
| `print` | `-p`: one turn, answer to stdout, steps to stderr, text or JSON |
| `terminal` | the interactive session (ratatui): every submission a job on the work queue, the job strip, streaming per turn, inline steps, queued permission prompts, slash commands, charts |
| `ontology_cli`, `graph_cli`, `embeddings_cli`, `admin` | `quack ontology`, `quack graph`, `quack embeddings`, and the server administration commands |
| `mcp` | the MCP server (rmcp) shared by `quack mcp` on stdio and `/mcp/v1/{workspace}` |
| `server` | `quack serve`: `auth` (identity and `Access::resolve`), `api` (REST handlers), `web` (askama pages calling the same `Access` operations as the API, [web-ui.md](web-ui.md)), `run` (the audited background runs: embeddings refresh, graph and ontology document passes), `queue` (uploads and cancel bookkeeping on the work queue), `api::jobs` (the jobs API and stream), `state`, `mcp_http` |

## The storage boundary

A workspace is one directory: `data.duckdb` plus `files/`. Everything classified about the
workspace is inside it: the sessions, the ontology, the graph, the context, and the detail
of what was done. `control.db` holds only who may open which workspace and the access audit
(who, what resource by opaque id, outcome, channel, when). When adding a table, ask which
side of the boundary it belongs on; if it can reveal workspace content, it goes in the
DuckDB file with a `_quack_` prefix (design doc sections 5 and 12).

## Build and test

```bash
make check   # cargo check --workspace --all-targets --all-features
make lint    # clippy with -D warnings
make test    # cargo test --workspace --all-features
make deny    # cargo deny check (advisories, licenses, bans)
```

New dependencies go in the root `[workspace.dependencies]` menu, pinned to the current
version, never inline in a member crate.
