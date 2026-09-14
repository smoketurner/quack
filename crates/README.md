# crates/

Your workspace members live here. The root `Cargo.toml` picks up every crate via
`members = ["crates/*"]`.

## Current crates

| Crate | Responsibility | Key deps (from the workspace menu) |
|---|---|---|
| `quack-core` | Config, errors, crypto provider, `control.db`, DuckDB workspaces, ingestion, retrieval, agent and tools, write policy, rig providers (`llm`) | `sqlx`, `duckdb`, `sea-query`, `rig`, `rustls`, `serde`, `toml`, `thiserror`, `tracing`, `uuid` |
| `quack` | The one binary: interactive terminal session (no arguments), `-p` print mode, `-q` SQL, `ingest`, `auth`, the server admin commands (`user`, `token`, `member`, `audit`), and `serve` (REST API and askama + htmx web UI, `templates/` and `static/`) | `quack-core`, `clap`, `ratatui`, `ratatui-textarea`, `crossterm`, `axum`, `askama`, `rust-embed`, `anyhow`, `tokio`, `mimalloc` |

Two crates, no Cargo features. `quack-core::llm` owns provider construction and the
per-turn dispatch (`llm::run_turn`); the binary never touches rig directly. Later
subcommands (`serve`, `mcp`, `desktop`, admin) are added to `quack`, not as new crates
(design doc section 4).

## Adding a crate

```bash
cargo new --lib crates/<name>    # or --bin for a binary
```

Then make it inherit the workspace baseline. A minimal member `Cargo.toml`:

```toml
[package]
name         = "<name>"
version.workspace      = true
edition.workspace      = true
rust-version.workspace = true
license.workspace      = true
publish      = false

[lints]
workspace = true

[dependencies]
serde     = { workspace = true, features = ["derive"] }
thiserror = { workspace = true }
```

Notes:

- Always include `[lints] workspace = true` so the strict lint baseline applies.
- Pull dependencies from the workspace menu with `{ workspace = true, features = [...] }`.
  If a crate you need isn't in `[workspace.dependencies]` yet, add it there (pinned, current
  version) rather than inline in the member.
- For binaries that open TLS connections, install the aws-lc-rs provider once at startup —
  see `docs/crypto.md`.
