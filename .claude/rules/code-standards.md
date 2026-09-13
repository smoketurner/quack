# Code Standards — review gates

Non-negotiable invariants for this stack that the compiler does **not** catch. Read this
before implementing or reviewing a change. Each item links to the doc with the full rationale
and code — this file is the gate, the doc is the detail.

> The `rust-agents` plugin does **not** auto-load this file. Its agents read
> `commits-and-issues.md`, `branching.md`, and `continuous-improvement.md` — each of which
> points here — and the main session reaches it through `CLAUDE.md`. Keep those pointers
> intact so the gates below reach plugin-spawned agents doing data-layer, crypto, or
> dependency work.

## Crypto & TLS → [docs/crypto.md](../../docs/crypto.md)

- [ ] **aws-lc-rs is the only crypto provider.** No `openssl`, `openssl-sys`, `native-tls`,
      or `ring` features on any dependency (`deny.toml` bans them).
- [ ] TLS crates use their rustls **+ aws-lc-rs** features (`rustls` → `aws_lc_rs`,
      `sqlx` → `tls-rustls-aws-lc-rs`, etc.).
- [ ] Binaries install the default provider **once** at the top of `main`
      (`aws_lc_rs::default_provider().install_default()`), before any TLS use.
- [ ] After touching TLS deps: `cargo tree -i ring` and `cargo tree -i openssl-sys` return no
      match; `cargo deny check` passes.

## Data layer

- [ ] **UUID v7 primary keys**, client-generated via `uuid::Uuid::now_v7()` — not v4
      (`gen_random_uuid()`), not `SERIAL`/sequential PKs.
- [ ] Control plane queries built with sea-query — **no raw SQL in handlers**.
- [ ] DuckDB workspaces are isolated per workspace ID; user SQL executes only in DuckDB,
      never against the control plane.

## Workspace hygiene → [docs/architecture.md](../../docs/architecture.md)

- [ ] Every member crate declares `[lints] workspace = true` — no crate escapes the baseline.
- [ ] Dependencies are pinned `=x.y.z` with `default-features = false` in
      `[workspace.dependencies]`; members opt in with `{ workspace = true, features = [...] }`.
      New deps are added to the workspace menu (current version looked up), never inline.
- [ ] Panics opt out narrowly in tests only: `#[expect(clippy::unwrap_used, reason = "...")]`.
- [ ] `thiserror` for library crates, `anyhow` for binaries; `tracing` for logging, never
      `println!`/`eprintln!`.
- [ ] Date/time uses `jiff`, not `chrono` or `time`, for direct handling.

## Before opening a PR

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace
cargo deny check
```

See [branching.md](branching.md) for the full pre-PR gate and
[commits-and-issues.md](commits-and-issues.md) for commit format.
