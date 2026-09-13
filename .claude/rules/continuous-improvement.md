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

- **control plane** — SQLite-backed workspace metadata, sea-query migrations
- **workspace engine** — DuckDB per-workspace databases, query execution
- **CLI** — clap-based command interface
- **crypto** — aws-lc-rs default provider installation

## Interfaces

- CLI: `cargo run --bin quack -- <subcommand>`

## Critical Paths

Features prone to silent breakage — live-test before any PR that touches them:

- SQLite control plane migrations
- sea-query query generation (correct SQLite dialect)
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
