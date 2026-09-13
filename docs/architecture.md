# Architecture

How the pieces of this project fit together.

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

```
            +-----------------------------+
            |         quack-cli           |  clap binary, output formatting
            +--------------+--------------+
                           | depends on
            +--------------v--------------+
            |         quack-core          |  config, error, control plane (SQLite),
            |                             |  workspace engine (DuckDB), sea-query
            +-----------------------------+
```

- **`quack-core`** owns config parsing, error types, the SQLite control plane (workspace
  metadata, threads, audit log), and the DuckDB workspace engine. Queries against the
  control plane are built with sea-query; user SQL runs directly against DuckDB.
- **`quack-cli`** owns the clap CLI surface and output formatting (table, JSON).

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
