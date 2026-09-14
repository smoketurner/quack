# Continuous Improvement

Project-specific instructions for the continuous improvement cycle.
This file is read by the `rust-ci-analyst` agent and the `/rust-agents:continuous-improvement` skill.
Customize the sections below as the project grows.

## Test Configuration

Run with a temporary data directory:

```bash
QUACK_DATA_DIR=/tmp/quack-dev cargo run --bin quack -- query "SELECT 1"
```

For debug output:

```bash
RUST_LOG=debug QUACK_DATA_DIR=/tmp/quack-dev cargo run --bin quack -- query "SELECT 1" 2>/tmp/quack-debug.log
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

- CLI: `cargo run --bin quack -- query|ingest|chat ...`
- TUI: `cargo run --bin quack -- -w <workspace>` (no subcommand, needs a TTY)
- Target (design doc section 11): `quack -p`, `quack serve`, `quack mcp`, `quack desktop`

## Critical Paths

Features prone to silent breakage — live-test before any PR that touches them:

- `control.db` migrations and the append-only `audit_log`
- sea-query query generation (correct SQLite dialect)
- Classification boundary: nothing workspace-revealing written outside the workspace file
- Agent SQL read/write classification and permission prompts
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
