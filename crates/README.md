# crates/

Your workspace members live here. The root `Cargo.toml` picks up every crate via
`members = ["crates/*"]`.

## Current crates

| Crate | Responsibility | Key deps (from the workspace menu) |
|---|---|---|
| `quack-core` | Config, error types, SQLite control plane, DuckDB workspace management | `sqlx`, `duckdb`, `sea-query`, `serde`, `toml`, `thiserror`, `tracing`, `uuid` |
| `quack-cli` | `quack` binary with `query`, `ingest`, `chat` subcommands | `quack-core`, `clap`, `anyhow`, `tokio`, `rig`, `tracing-subscriber`, `mimalloc` |
| `quack-tui` | `quack-tui` binary: ratatui chat session | `quack-core`, `ratatui`, `ratatui-textarea`, `crossterm`, `rig`, `tokio` |

Target layout (design doc section 4): `quack-cli` and `quack-tui` merge into a single
`quack` crate (terminal, print mode, `serve`, `mcp`, admin, and later `desktop`). Two
crates, no Cargo features. Provider construction moves into `quack-core::llm`.

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
