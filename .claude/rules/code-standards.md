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
- [ ] **FIPS is Linux-only, by linkage.** `aws-lc-rs` and `rustls` carry their shared
      features in `[dependencies]` (`crates/quack-core/Cargo.toml`), and the
      `cfg(target_os = "linux")` section adds `fips` to both. aws-lc-fips-sys
      links statically only on Linux and BSD; elsewhere a FIPS build needs a shared library
      beside the binary. The Linux build needs `cmake`, `go`, and
      `AWS_LC_FIPS_SYS_CC=clang`/`CXX=clang++`; `crypto::install_default_provider` logs the
      module it installed and a test asserts the gating (`docs/crypto.md`).
- [ ] Binaries install the default provider **once** at the top of `main`
      (`aws_lc_rs::default_provider().install_default()`), before any TLS use.
- [ ] After touching TLS deps: `cargo tree -i ring` and `cargo tree -i openssl-sys` return no
      match; `cargo deny check` passes.

## Data layer

- [ ] **UUID v7 primary keys**, client-generated via `uuid::Uuid::now_v7()` — not v4
      (`gen_random_uuid()`), not `SERIAL`/sequential PKs.
- [ ] `control.db` queries built with sea-query — **no raw SQL in handlers**.
- [ ] **A shipped `control.db` migration is never edited.** Schema changes add a new
      `crates/quack-core/migrations/NNNN_*.sql`; sqlx checksums each file and refuses a
      database whose recorded checksum no longer matches. Migration SQL is literal and
      never generated from the `Iden` enums in `storage::queries`, which track the
      current schema rather than history (`docs/migrations.md`).
- [ ] **The workspace DuckDB file is the classification boundary.** Anything that can
      reveal workspace content (documents, chunks, graph, ontology, context, sessions,
      messages, audit detail) lives in that file with a `_quack_` prefix, never in
      `control.db` (design doc section 5).
- [ ] `control.db` holds only users, workspaces (name and label), membership, tokens
      (API token hashes, and HPKE-sealed OAuth tokens of signed-in users and of model
      providers, whose vault key is never in the database), and the access `audit_log`. `audit_log` is append-only: no `UPDATE`/`DELETE` path.
- [ ] Every request that touches a workspace writes an `audit_log` row, including denied
      ones, and a `_quack_audit` detail row inside the workspace under the same UUID v7.
- [ ] DuckDB internal statements use `duckdb::params!`; identifiers go through
      `quote_ident`; file paths are bound, never interpolated (design doc section 5.6).
- [ ] Agent SQL passes read/write classification (`json_serialize_sql`) and the permission
      layer, runs under `memory_limit`/`threads`/timeout, and is refused if it references
      `_quack_` tables (design doc section 7.4).
- [ ] DuckDB workspaces are isolated per workspace; user SQL executes only in DuckDB,
      never against `control.db`; no `ATTACH` between workspaces.
- [ ] **No `INSTALL` or `LOAD` of DuckDB extensions anywhere.** The static binary cannot
      load them; only what `libduckdb-sys` compiles in (`json`, `parquet`) may be used.
- [ ] **The workspace connection is confined at open** (`WorkspaceDb::confine_to`, design
      doc section 7.4): `allowed_directories` is the workspace directory only,
      `enable_external_access` and `allow_persistent_secrets` are off, and
      `lock_configuration` is on before any user or agent statement runs, and every `SET`
      the code needs goes before the lock. Every other connection to that workspace file is
      a `WorkspaceDb::try_clone_reader` clone of that confined one (`duckdb::Connection::
      try_clone`, a new connection to the same already-open `DatabaseInstance`, which
      inherits the locked configuration) — never `Connection::open` on a workspace path a
      second time. **Nothing stops you from doing that, which is why this is a rule.**
      `DuckDB`'s file lock is advisory and per-process: it refuses a second `quack`
      process, but two opens inside one process both succeed, both complete the whole
      confinement sequence, and produce two independent `DatabaseInstance`s that cannot
      see each other's commits and silently overwrite each other's writes. In-process
      single-open is therefore ours to enforce — in the server, the per-workspace
      `OnceCell` in `AppState::workspace_handle`. The opened connection then moves onto
      its workspace's writer thread (`storage::writer::Writer::spawn`) and is reached only
      by closures sent there; reader clones are made on that thread. The one other writing
      connection is the server's `storage::audit::AuditLog`, a clone that only ever
      `INSERT`s new `_quack_audit` rows (new ids never conflict, so `DuckDB` commits them
      beside a write in progress); nothing else may write through it. A reader clone never calls `confine_to`
      or `apply_resource_limits`
      again (both `SET`s fail once configuration is locked), and every statement it runs
      goes through `WorkspaceDb::read_only`, scoped to that one piece of work.

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
