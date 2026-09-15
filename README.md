# quack

A knowledge engine with many interfaces, built in Rust. A workspace holds documents
(vectorized for retrieval), tables (DuckDB analytics), and an ontology-backed knowledge
graph; one agent answers questions across all three and shows every action it takes.
The core is a library; the web UI, REST API, MCP server, terminal session, print mode,
and desktop window are thin clients of it, all subcommands of one `quack` binary.

The full design is in [docs/design-doc.md](docs/design-doc.md). The code currently
implements an earlier slice of it; section 17 of the design doc lists the gaps.

## The stack

| Concern | Choice | Why |
|---|---|---|
| Workspace | Cargo workspace, crates under `crates/` | Edition 2024, resolver 3 |
| Workspace storage | DuckDB, one file per workspace | The classification boundary: tables, chunks, graph, ontology, context, sessions, and audit detail all live inside it |
| Access control | SQLite `control.db` (server mode) | Users, membership, tokens, and the mandatory access audit log; nothing that reveals workspace content |
| Query layer | [`sea-query`](https://crates.io/crates/sea-query) | Type-safe SQL for `control.db`; DuckDB internals use bound parameters |
| DB driver | [`sqlx`](https://crates.io/crates/sqlx) | Async SQLite for `control.db` |
| LLM layer | [`rig`](https://crates.io/crates/rig) | Providers: Ollama, OpenAI-compatible, Anthropic; auth none / API key / OAuth PKCE |
| Crypto / TLS | [`aws-lc-rs`](https://crates.io/crates/aws-lc-rs) | Single provider; OpenSSL and `ring` are banned |

## Quick start (current code)

```bash
cargo run --bin quack -- -q "SELECT 1 AS answer"
cargo run --bin quack -- -q "SELECT * FROM generate_series(1, 5) AS t(n)" -f csv
cargo run --bin quack -- ingest sales.csv -w myworkspace
cargo run --bin quack -- -p "total sales by region" -w myworkspace   # answer to stdout, steps to stderr
cargo run --bin quack -- -p "total sales by region" -f json           # {answer, steps, chart}
cargo run --bin quack -- -w myworkspace                               # interactive terminal session
cargo run --bin quack -- -p "and by month?" -c                        # continue the latest session
cargo run --bin quack -- sessions                                     # list sessions
cargo run --bin quack -- export 01a0a0e1 --sql                        # replay a session's SQL
cargo run --bin quack -- ingest policy.pdf --pin                      # full text in every prompt
cargo run --bin quack -- -p "what is excluded?" --mode query          # sources only, cited [n]
cargo run --bin quack -- context edit                                 # definitions the agent follows
cargo run --bin quack -- ontology propose                             # draft the graph schema from the tables, then review
cargo run --bin quack -- auth login azure                             # OAuth sign-in for a provider
cargo run --bin quack -- user add alice --admin                       # server users, tokens, members, audit
cargo run --bin quack -- token create -w myworkspace --user alice --name ci --scopes read,write
cargo run --bin quack -- audit --outcome denied --csv
cargo run --bin quack -- serve --local                                # web UI and REST API on 127.0.0.1:8080, no login
make demo-data                                                        # load examples/logistics into workspace "logistics"
```

`examples/` holds ready-made workspaces, each with a README, a loader, its documents,
and a workspace context. [`examples/logistics`](examples/logistics/README.md) is one
public-domain supply chain dataset: USAID's delivery history as a `shipments` table
(purchase orders, vendors, manufacturing sites, products, countries, Incoterms, shipment
modes, dates, freight and insurance) with documents in the same vocabulary, so the same
entities appear in SQL, vector and keyword search, and the knowledge graph.

The workspace context is the owner's instructions for the agent: persona, what columns
mean, how metrics are defined, known data problems. It is stored and versioned inside the
workspace file (`quack context show|edit|history|export|import`), and a global prefix can
live at `~/.config/quack/context.md`.

Document search is hybrid: an exact cosine scan over stored embeddings and quack's own BM25
term index, fused with reciprocal rank fusion, so exact tokens such as policy numbers are
found. No DuckDB extension is downloaded or loaded; everything is compiled into the one
binary. Chunks carry page and
heading; answers cite `[n]` markers that are validated against what was retrieved and
listed as sources. `--mode query` makes the agent answer only from retrieved chunks and
query results.

Every turn is recorded in the workspace's own database. `-c` continues the latest session,
`-r ID` resumes one (id prefixes work), and `export` writes a session as a runnable `.sql`
file or a Markdown transcript.

`-q` prints a table on a terminal and ndjson when piped; `-f` selects table, json, ndjson, csv,
or markdown. Exit codes: 0 ok, 1 error, 2 usage, 3 write refused, 4 auth required.

A provider with `auth = "oauth"` (Azure OpenAI, gateways that forbid static keys) gets its
bearer token from `quack auth login PROVIDER`: a browser sign-in with PKCE, or a device
code over SSH, without a display, or with `--device-code`. The token is cached encrypted
under `~/.local/share/quack/tokens/`, with the key in the OS keychain (macOS Keychain,
Linux kernel keyring, Windows Credential Manager) or a 0600 key file, and refreshes itself.
`quack auth status` and `quack auth logout` inspect and forget it; `-p` and `ingest` exit 4
naming the login command when no token is usable.

`quack serve` exposes the same engine as a web UI and a REST API under `/api/v1` (design
doc sections 11.1 and 11.2): password login with a session cookie or bearer token, API
tokens scoped to one workspace with read, write, and admin scopes, viewer, member, and
owner roles, an agent `query` endpoint and its SSE `query/stream`, direct `sql`, `search`,
documents through a per-workspace upload queue, tables, the context editor, sessions with
export, members, and admin users. The web UI covers workspaces, chat with the steps
block, citations, and charts, documents (upload, paste, pin, delete), tables, SQL with
CSV download, the context editor with history, settings with members and tokens, and the
admin users and audit pages. Every request that names a workspace writes an access-audit
row in `control.db`, denied ones included, and its content detail inside the workspace.
`--local` skips authentication on a loopback address for a single user. Create the first
admin with `quack user add NAME --admin`, then log in at `http://127.0.0.1:8080/`.

`-p` and the terminal session need `[general].chat_model` and `[general].embedding_model`
set to `PROVIDER/MODEL` references in `~/.config/quack/config.toml`, with a matching
`[providers.<name>]` section for each (see the configuration section of the design doc).
Unknown keys are rejected. `QUACK_MODEL` overrides the chat model; `QUACK_DATA_DIR` keeps
test data out of `~/.local/share/quack`. A minimal offline config:

```toml
[general]
chat_model = "ollama/llama3.1:8b"
embedding_model = "ollama/nomic-embed-text"

[providers.ollama]
type = "ollama"
embedding_dimension = 768
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
- **`Dockerfile`, `Dockerfile.build`, `docker-bake.hcl`** — sample container and static-build
  files to be adapted for `quack serve` (design doc section 14).
- **`Makefile`** — `build`, `fmt`, `lint`, `test`, `deny`, `help`.
- **`CLAUDE.md` + `.claude/rules/`** — conventions for Claude Code agents
  (code standards, development discipline, branching, Conventional Commits, continuous
  improvement).
- **`docs/`** — the design and the stack patterns, with copy-pasteable code (see below).

## Documentation

| Doc | Covers |
|---|---|
| [docs/design-doc.md](docs/design-doc.md) | The product: substrates, agent, ontology and induction, interfaces, storage boundary, auth, audit, build, gaps, and implementation order |
| [docs/architecture.md](docs/architecture.md) | Cargo workspace layout, crate layering (current and target), core modules, lint inheritance |
| [docs/migrations.md](docs/migrations.md) | Schema versioning for `control.db` (sea-query) and the `_quack_` tables in workspace DuckDB files |
| [docs/sea-query.md](docs/sea-query.md) | sea-query schema, `Iden` enums, query building for `control.db` |
| [docs/crypto.md](docs/crypto.md) | aws-lc-rs default provider, keeping `ring`/OpenSSL out |
| [docs/web-ui.md](docs/web-ui.md) | axum + askama + htmx + Tailwind patterns for `quack serve` |
| [docs/ci-cd.md](docs/ci-cd.md) | CI jobs, and the Docker/build/scan/release patterns to wire up for releases |

## License

Dual-licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your
option.
