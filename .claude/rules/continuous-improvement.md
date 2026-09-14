# Continuous Improvement

Project-specific instructions for the continuous improvement cycle.
This file is read by the `rust-ci-analyst` agent and the `/rust-agents:continuous-improvement` skill.
Customize the sections below as the project grows.

## Test Configuration

Run with a temporary data directory:

```bash
QUACK_DATA_DIR=/tmp/quack-dev cargo run --bin quack -- -q "SELECT 1"
```

For debug output:

```bash
RUST_LOG=debug QUACK_DATA_DIR=/tmp/quack-dev cargo run --bin quack -- -q "SELECT 1" 2>/tmp/quack-debug.log
```

## Project Subsystems

Workspace members are auto-detected from `Cargo.toml`. Track these logical subsystems in
`coverage-status.md` as crates are added:

- **access control** — `control.db`: users, workspaces, membership, tokens, access audit
  (sea-query migrations)
- **workspace storage** — DuckDB per-workspace files: user tables, `_quack_` tables, query
  execution, read/write classification, limits
- **ingestion and retrieval** — parsers, chunking, embedding, hybrid vector + FTS search,
  citations
- **ontology and graph** — ontology tables and induction, extraction, resolution,
  traversal
- **agent** — rig-based tool loop, permissions, prompt, chat modes, event stream
- **interfaces** — CLI/print, TUI, web UI, REST, MCP, desktop
- **crypto and auth** — aws-lc-rs default provider installation, OAuth token manager

## Interfaces

- Print mode: `cargo run --bin quack -- -p "question" [-f text|json]`; SQL: `-q "..." [-f ...]`
- Ingest: `cargo run --bin quack -- ingest FILE`
- Sessions: `cargo run --bin quack -- sessions [--json]`, `export ID [--sql|--markdown]`,
  `-p ... -c` / `-r ID`
- TUI: `cargo run --bin quack -- -w <workspace>` (no arguments, needs a TTY)
- Target (design doc section 11): `quack serve`, `quack mcp`, `quack desktop`

## Critical Paths

Features prone to silent breakage — live-test before any PR that touches them:

- `control.db` migrations and the append-only `audit_log`
- sea-query query generation (correct SQLite dialect)
- Classification boundary: nothing workspace-revealing written outside the workspace file
- Agent SQL read/write classification and the terminal permission prompt (y/n/a)
- Event stream ordering: every tool call emits started then finished; `-p` steps on stderr only
- Session recording: user, one tool message per step, assistant, in that order; a failed
  first turn leaves no empty session behind
- Term index (`_quack_terms`) written on every chunk insert; no runtime DuckDB extension
  is installed or loaded anywhere (static musl cannot dlopen)
- Citation validation: `[n]` markers not registered this turn are stripped; the rest are
  renumbered from 1 and listed as sources
- Pinned documents injected in full within `[retrieval].pinned_token_budget`
- System prompt order (design doc 7.2): role and mode, tool guidance, tables, documents and
  pinned text, workspace context, permission rules; context truncated at
  `[context].max_tokens` with a visible note
- Citation validation (every `[n]` maps to a chunk retrieved in that turn)
- Embedding dimension recorded per workspace and checked on open
- DuckDB workspace isolation (each workspace gets its own database file)
- aws-lc-rs default crypto provider installed exactly once at startup

The implementation rules behind these paths are the review gates in
[`code-standards.md`](code-standards.md); the full stack patterns are the `docs/` table in
[`README.md`](../../README.md). Read them before changing the code behind any path above.

## Environment Setup

- **SQLite**: no setup; control plane DB lives at `{data_dir}/control.db`
- **DuckDB**: no setup; workspace DBs live at `{data_dir}/workspaces/{id}/data.duckdb`

## Testing Notes

- Tests use in-memory databases where possible.
- DuckDB workspace isolation is verified by creating tables in one workspace and confirming
  they don't appear in another.
