# crates/

Your workspace members live here. The root `Cargo.toml` picks up every crate via
`members = ["crates/*"]`.

## Current crates

| Crate | Responsibility | Key deps (from the workspace menu) |
|---|---|---|
| `quack-core` | Config, error types, SQLite control plane, DuckDB workspace management | `sqlx`, `duckdb`, `sea-query`, `serde`, `toml`, `thiserror`, `tracing`, `uuid` |
| `quack-cli` | `quack` binary with `query` subcommand | `quack-core`, `clap`, `anyhow`, `tokio`, `tracing-subscriber`, `mimalloc` |

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
