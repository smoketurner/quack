# quack

A data analysis platform built with Rust. Execute SQL queries against isolated DuckDB
workspaces, with a SQLite control plane managing workspace metadata.

## The stack

| Concern | Choice | Why |
|---|---|---|
| Workspace | Cargo workspace, crates under `crates/` | Edition 2024, resolver 3 |
| Control plane | SQLite | Zero-setup, fast metadata store |
| Analytical DB | DuckDB | Per-workspace analytical engine |
| Query layer | [`sea-query`](https://crates.io/crates/sea-query) | Type-safe SQL generation for control plane |
| DB driver | [`sqlx`](https://crates.io/crates/sqlx) | Async SQLite for control plane |
| Crypto / TLS | [`aws-lc-rs`](https://crates.io/crates/aws-lc-rs) | Preferred over OpenSSL and `ring` |

## Quick start

```bash
cargo run --bin quack -- query "SELECT 1 AS answer"
cargo run --bin quack -- query "SELECT * FROM generate_series(1, 5) AS t(n)" -f json
cargo run --bin quack -- query "CREATE TABLE test(id INTEGER, name VARCHAR)" -w myworkspace
```

## What's in the box

- **`Cargo.toml`** — virtual workspace with a curated, version-pinned `[workspace.dependencies]`
  menu and a strict `[workspace.lints]` baseline (clippy `pedantic` + panic/arithmetic/cast
  denies). Member crates inherit with `[lints] workspace = true`.
- **`.rustfmt.toml`, `.clippy.toml`, `deny.toml`, `rust-toolchain.toml`, `.editorconfig`** —
  formatting, lint tuning, supply-chain policy (advisories, licenses, bans — OpenSSL/`ring`
  denied), and a pinned toolchain.
- **`.github/`** — CI (fmt, clippy, `cargo test`, dependency-review, `cargo-deny`; actions
  SHA-pinned, plus `secure_workflows.yml` enforcing SHA pins) and Dependabot (cargo +
  actions, grouped, 7-day cooldown).
- **`Makefile`** — `build`, `fmt`, `lint`, `test`, `deny`, `help`.
- **`CLAUDE.md` + `.claude/rules/`** — conventions for Claude Code agents
  (branching, Conventional Commits, continuous improvement).
- **`docs/`** — the patterns, with copy-pasteable code (see below).

## Documentation

| Doc | Covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Workspace layout, recommended crate split, lint inheritance |
| [docs/migrations.md](docs/migrations.md) | SQLite migration patterns |
| [docs/sea-query.md](docs/sea-query.md) | sea-query schema, `Iden` enums, query building |
| [docs/crypto.md](docs/crypto.md) | aws-lc-rs default provider, keeping `ring`/OpenSSL out |
| [docs/ci-cd.md](docs/ci-cd.md) | CI jobs, and the deferred Docker/build/scan/release patterns |

## License

Dual-licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your
option.
