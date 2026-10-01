# Architecture

This file maps the sections of [design-doc.md](design-doc.md), the product design, to the
crates and modules that exist.

## Workspace

quack is a virtual Cargo workspace (`Cargo.toml` has no `[package]`) with two members
under `crates/`: `quack-core`, the library, and `quack`, the one binary. The root sets:

- `[workspace.package]`: edition 2024, the MSRV.
- `[workspace.dependencies]`: every dependency pinned to an exact version, default features
  off; members opt in.
- `[workspace.lints]`: panic, cast, and arithmetic denies, clippy `pedantic`, and selected
  restriction and nursery lints (own-crate absolute paths, untyped `let _`, ref-counted
  `.clone()`, `#[allow]` without `#[expect]`). Every member declares `[lints] workspace = true`.

There are no Cargo features: every build contains every surface.

## Layering

Every interface is a thin adapter over the core. Nothing above the core owns behavior;
nothing in it knows about HTTP, terminals, or windows.

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
| `config::inspect` | the file read outside `Config::load` (`quack config`): each recognized setting's value in force and origin, unrecognized keys, the environment variables read | 13 |
| `crypto` | installs the aws-lc-rs provider once | 14 |
| `vault` | data at rest sealed with HPKE under one key in the OS keychain, per purpose and subject; callers store the `Sealed` value | 10.3, 12 |
| `oidc` | server sign-in through an OpenID Connect issuer (`SignIn`), bearer verification of the issuer's access tokens (`SignIn::verify_bearer`), signed-in users' tokens sealed in `control.db` (`UserTokens`), each person's own token for session renewal and on-behalf-of exchanges, and the one place an issuer's refusal of it is handled (`SubjectTokens`) | 10.2, 12 |
| `web_sessions` | `quack serve`'s browser and API-login sessions (`WebSessions`), in core so an issuer's refusal ends them where it is seen | 12 |
| `doctor` | `quack doctor`'s checks over a `config::inspect` result: config file, crypto module, data directory mode, `control.db`, the workspace, each model's credential and a model-list probe of its provider, the server bind; creates nothing | 11.5 |
| `error` | the `thiserror` enum every layer returns | |
| `storage::control` | `control.db` (SQLite, sea-query): users, workspaces, membership, tokens, the append-only access `audit_log` | 5.5, 12 |
| `migrations/*.sql` | `control.db` schema versions | [migrations.md](migrations.md) |
| `storage::workspace` | the workspace DuckDB file: open with confinement and limits, the `_quack_` tables, statement classification, hybrid retrieval (cosine scan plus BM25 over `_quack_terms`), the document registry, query execution with faithful JSON values | 5.4, 6.1, 7.4 |
| `storage::writer` | the workspace's one writer connection as an actor: its own thread runs the closures sent to it, interactive before background; callers await `run` / `run_at` | 4.1, 7.4 |
| `priority` | the interactive/background task-local that the writer line and the model limiter read | 4.1 |
| `storage::sessions`, `storage::context`, `storage::audit` | conversations, the versioned workspace context, the content half of the audit (`AuditLog`: the server's insert-only audit connection) | 5.3, 8 |
| `embedding` | each text's embedding role (query, document, similarity), each model family's trained prefixes (`presets`) and their `[embedding]` overrides, the profile a vector is made under, the width check, `refresh` | 6.1, 5.4 |
| `ingestion` | registration with SHA-256 dedup, parsers (`parser`, `html`, `office`, `xlsx`), chunking, embedding, tables from structured files, piped stdin | 6.1, 6.2 |
| `import` | rows from Postgres, SQLite, or an HTTP data file as a workspace table | 6.2 |
| `analysis` | the agent loop as an event stream (`agent`, `events`), the tools (`tools`: plain values over the workspace and its settings, the turn's state, `tools::Turn`, reaching each call as a runtime scope of rig's `ToolContext`), the system prompt (`text_to_sql`), write policy, citations, the chart spec, the reranking hook (`rerank`) | 7, 9 |
| `ontology` | the model, validation, versions (`store`), induction from tables and documents (`induction`, `documents`), the review queue (`candidates`) | 6.3, 6.5 |
| `graph` | the knowledge graph: `store`, `tables` (mapping extraction), `extract` (constrained model extraction with drift), `resolve` (merges), `traverse` | 6.4 |
| `ocsf` | access-audit rows rendered as OCSF 1.9.0 events | 12 |
| `okf` | Open Knowledge Format bundles in and out | 17 |
| `extraction` | what both extraction runs share: the `Extract` trait, lenient JSON answers, concurrent calls with per-chunk progress (`RunProgress`), even sampling across documents, name counts (`Tally`) | 6.4, 6.5 |
| `progress` | the per-chunk progress report the extraction runs make to their caller | 6.5 |
| `jobs` | the work queue every interface submits background work to: ordered lanes, cancel, progress, a broadcast of job snapshots | 4.1 |
| `llm` | rig provider construction over `limit::LimitedHttp` (each provider's process-wide request limit), `TurnRequest` (one agent turn), `SchemaCall` (one tool-less prompt whose answer a JSON schema shapes, sent as the provider's structured output and parsed whole: graph extraction against `Ontology::extraction_schema`, the ontology's document pass, the model reranker, and history summaries), OAuth token management (`oauth`), the person an on-behalf-of provider acts for (`acting`, a task-local), Amazon Bedrock over the AWS SDK's credential chain with the same limits (`bedrock`) | 4.1, 10 |

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

A table that can reveal workspace content goes in the workspace DuckDB file with a
`_quack_` prefix (design doc sections 5 and 12). A workspace is one directory:
`data.duckdb` plus `files/`. It holds everything classified about the workspace: the
sessions, the ontology, the graph, the context, and the detail of what was done.
`control.db` holds only who may open which workspace and the access audit (who, what
resource by opaque id, outcome, channel, when). Decide the side for every new table.

## Build and test

```bash
make check   # cargo check --workspace --all-targets --all-features
make lint    # clippy with -D warnings
make test    # cargo test --workspace --all-features
make deny    # cargo deny check (advisories, licenses, bans)
```

Add new dependencies to the root `[workspace.dependencies]` menu, pinned to the current
version, never inline in a member crate.
