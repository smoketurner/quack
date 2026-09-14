# Architecture

How the pieces of this project fit together. The product design is in
[design-doc.md](design-doc.md); this file covers the Cargo workspace, the crate layering,
and the lint baseline.

## Workspace

A virtual Cargo workspace (`Cargo.toml` has no `[package]`). All crates live under
`crates/` and are discovered by `members = ["crates/*"]`. Shared settings come from the
root:

- `[workspace.package]` — `version`, `edition = "2024"`, `rust-version` (MSRV), `license`.
  Inherit per crate with `edition.workspace = true`, etc.
- `[workspace.dependencies]` — the pinned dependency menu. Crates use
  `dep = { workspace = true, features = ["..."] }`.
- `[workspace.lints]` — the strict lint baseline. Crates use `[lints] workspace = true`.

`resolver = "3"` (the edition-2024 default) gives MSRV-aware dependency resolution.

## Layering

Every interface is a thin adapter over the core library. Nothing above the core line owns
behavior; nothing below it knows about HTTP, terminals, or windows.

```
      +------------+  +------------+  +------------+  +------------+  +---------------+
      |  web UI    |  |  REST API  |  |    MCP     |  | TUI/print  |  | quack desktop |
      |  (askama,  |  |  (axum)    |  | (stdio,    |  | (ratatui,  |  | (Tauri window |
      |   htmx)    |  |            |  |  SSE)      |  |  clap)     |  |  over serve)  |
      +------------+  +------------+  +------------+  +------------+  +---------------+
                        all subcommands of the single `quack` binary
             \               |               |               |                /
              +--------------+---------------+---------------+---------------+
                                             | in-process calls
                              +--------------v--------------+
                              |         quack-core          |
                              +-----------------------------+
```

**Current crates** (what builds today):

| Crate | Binary | Owns |
|---|---|---|
| `quack-core` | — | config, errors, crypto provider install, `control.db` (sqlx + sea-query), workspace DuckDB engine with statement classification and limits, ingestion, chunking, vector search, rig-based agent, tools, write policy, `llm` provider construction and `run_turn` dispatch, chart spec |
| `quack` | `quack` | interactive terminal session when run with no subcommand; `query`, `ingest`, `chat` subcommands; `--allow-write` |

`serve`, `mcp`, print mode (`-p`), admin, and later `desktop` are added to `quack` as
subcommands (design doc section 4). There are no Cargo features; every build contains
every surface.

## Core modules

`quack-core` is organized by substrate and by responsibility:

| Module | Responsibility |
|---|---|
| `workspace/` | open and create a workspace directory; `.quack/` discovery for the TUI |
| `storage/` | the workspace DuckDB file (everything classified) and `control.db` (access control and access audit) |
| `ingestion/` | parsers, chunking with heading and page metadata, embedding, index maintenance |
| `retrieval/` | vector + full-text fusion, citation metadata, pinned documents |
| `analytics/` | SQL execution, read/write classification, resource limits, schema introspection |
| `ontology/` | ontology tables, validation, versions, induction (propose and review) |
| `graph/` | ontology-guided extraction, entity resolution, provenance, traversal |
| `agent/` | the tool-calling loop as an event stream, tools, permissions, prompt, chat modes |
| `llm/` | rig provider construction; auth none / API key / OAuth PKCE with a token manager |
| `context/` | the stored workspace context; Markdown import and export |

Today's code has `storage/`, `ingestion/`, `analysis/` (agent, tools, policy, text-to-SQL,
vector index, chart), `llm/`, and `crypto`. The split above is the target; new work should
land in the target module rather than growing `analysis/`.

## The storage boundary

A workspace is one directory: `data.duckdb` plus `files/`. Everything classified about
the workspace is inside it, including the sessions, the ontology, the context, and the
detail of what was done. `control.db` holds only who may open which workspace and the
access audit (who, what resource by opaque id, outcome, channel, when). See design doc
sections 5 and 12. When adding a table, ask which side of the boundary it belongs on; if
it can reveal workspace content, it goes in the DuckDB file with a `_quack_` prefix.

## Lint inheritance

Every crate must declare:

```toml
[lints]
workspace = true
```

This applies the panic-prevention, cast, and arithmetic denies plus clippy `pedantic`
(as warnings) from the root. Without it, a crate silently escapes the baseline.

## Build & test flow

```bash
make check   # cargo check --workspace --all-targets --all-features
make lint    # clippy with -D warnings
make test    # cargo test --workspace --all-features
make deny    # cargo deny check (advisories, licenses, bans)
```

## Adding a layer

1. `cargo new --lib crates/<name>` (see `crates/README.md`).
2. Add `[lints] workspace = true` and inherit package fields.
3. Pull deps from the workspace menu; add new ones (pinned) to `[workspace.dependencies]`.
4. Read the matching `docs/` file for that layer's patterns before writing code.
