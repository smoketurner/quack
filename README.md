# quack

An air-gapped knowledge engine. A workspace holds documents (chunked and vectorized),
tables (DuckDB), and an ontology-backed knowledge graph; one agent answers questions across
all three and shows every action it takes. The core is a library; the terminal session,
print mode, web UI, REST API, and MCP server are thin clients of it, all subcommands of one
static `quack` binary that needs nothing but a local model server.

The full design is [docs/design-doc.md](docs/design-doc.md); section 17 maps what is still
open to GitHub issues.

## Install

- **Releases**: static binaries for Linux (musl, x86_64 and aarch64), macOS, and Windows,
  with `SHA256SUMS`, from the GitHub releases page.
- **Homebrew**: each release ships a `quack.rb` formula for a tap.
- **Container**: `ghcr.io/smoketurner/quack` runs `quack serve`; the release also attaches
  image tarballs for `docker load` on hosts without a registry. `docker-compose.yml` runs
  it beside Ollama ([docs/ci-cd.md](docs/ci-cd.md)).
- **From source**: `cargo build --release` with the toolchain in `rust-toolchain.toml`
  (Linux needs `cmake` and `clang`; DuckDB and aws-lc compile from source).

## Configure

`~/.config/quack/config.toml` names the models; `QUACK_CONFIG_DIR` and `QUACK_DATA_DIR`
move the config and the data directory. A minimal offline setup with Ollama:

```toml
[general]
chat_model = "ollama/gpt-oss:20b"
embedding_model = "ollama/qwen3-embedding:0.6b"

[providers.ollama]
type = "ollama"
embedding_dimension = 1024
```

Providers are Ollama, OpenAI-compatible endpoints, and Anthropic, with `auth = "none"`,
`"api-key"`, or `"oauth"` (PKCE in a browser, or a device code over SSH, cached encrypted
with the key in the OS keychain). Unknown keys are rejected. Design doc section 13 lists
every section: retrieval, reranking, context, analysis limits, server, ontology, graph,
import.

## Use it

```bash
quack ingest sales.csv -w sales                      # a file becomes a table (CSV, Parquet, JSON, XLSX) or chunks (PDF, DOCX, PPTX, HTML, Markdown, text)
quack -p "total sales by region" -w sales            # one agent turn: answer to stdout, steps to stderr
quack -p "total sales by region" -f json             # {answer, steps, citations, chart, graph}
quack -q "SELECT region, sum(total) FROM sales GROUP BY 1" -f csv   # SQL without the agent
cat orders.csv | quack -q "SELECT count(*) FROM stdin"  # piped data is the table stdin
quack -w sales                                       # the terminal session (streaming, inline steps, y/n/a on writes, /help)
quack -p "and by month?" -c                          # continue the latest session; -r ID resumes one
quack ingest policy.pdf --pin                        # full text in every prompt
quack -p "what is excluded?" --mode query            # answers only from retrieved chunks, cited [n]
quack context edit                                   # the owner's definitions the agent follows, versioned in the workspace
quack ontology propose                               # draft classes, relations, and mappings from the tables, then review
quack graph extract -y && quack graph search Kenya   # build the knowledge graph and walk it
quack import postgres://u:p@host/db --table orders --from public.orders   # snapshot a Postgres or SQLite query, or an http(s) file
quack okf export ./bundle                            # the workspace as an Open Knowledge Format bundle; `ingest DIR` imports one
quack serve --local                                  # web UI, REST API, and MCP at http://127.0.0.1:8080, no login
quack mcp -w sales                                   # MCP server on stdio, for Claude Code and editors
quack user add alice --admin                         # server users, tokens, members, and the audit log
```

`examples/storms` is a ready-made workspace that uses all of this: NOAA's 2024 Storm
Events Database as three linked tables, Census population figures imported over HTTPS,
the NWS documents that define every code in the tables, a pinned glossary, a shipped
ontology, and a knowledge graph built from the mapped rows. `make demo-data` loads it;
`examples/logistics` is a smaller supply-chain workspace.

**What every interface shares.** Retrieval is hybrid (an exact cosine scan plus quack's own
BM25 index with stemming, fused by reciprocal rank fusion, an optional reranker) and every
answer's `[n]` citations are validated against what was retrieved that turn. Writes are a
permission decision: the terminal asks, print mode refuses with exit 3 unless
`--allow-write`, the API needs the write scope. Every turn is recorded as a session inside
the workspace file, exportable as runnable SQL or a Markdown transcript. The knowledge
graph is built from mapped table rows and from documents through the chat model, with
provenance on every node and edge, and stays honest about being provisional or stale.

**The server.** `quack serve` adds password login, API tokens scoped to one workspace with
read, write, and admin scopes, viewer, member, and owner roles, a per-workspace upload
queue, session sharing, and an append-only access audit whose content detail stays inside
the workspace. The web UI covers chat, documents, tables with import, SQL, the context
editor, the ontology review queue, the graph with its merge queue, settings, and admin
pages. Design doc sections 11 and 12.

Exit codes: 0 ok, 1 error, 2 usage, 3 write refused, 4 auth required.

## Develop

```bash
make lint            # clippy with warnings denied
make test            # the whole suite
make deny            # advisories, licenses, bans
make release-gates   # no ring or OpenSSL in the runtime tree
make css-build       # after a template change (the built CSS is committed)
```

Conventions for contributors and coding agents are in [CLAUDE.md](CLAUDE.md) and
`.claude/rules/`; [CONTRIBUTING.md](CONTRIBUTING.md) has the short version.

| Doc | Covers |
|---|---|
| [docs/design-doc.md](docs/design-doc.md) | The product: substrates, agent, ontology and induction, graph, interfaces, storage boundary, auth and audit, build, gaps, implementation order |
| [docs/architecture.md](docs/architecture.md) | The crates and modules, and which design section each implements |
| [docs/migrations.md](docs/migrations.md) | Schema versioning for `control.db` and the `_quack_` tables, and why sea-query stops at the DuckDB boundary |
| [docs/crypto.md](docs/crypto.md) | aws-lc-rs as the only crypto provider and the release gate that keeps it so |
| [docs/web-ui.md](docs/web-ui.md) | axum, askama, htmx, Tailwind, and the vendored scripts behind `quack serve` |
| [docs/ci-cd.md](docs/ci-cd.md) | CI, the release workflow, the container images, and the compose deployment |

## License

Dual-licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your
option.
