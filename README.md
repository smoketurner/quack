# quack

quack answers questions about your own data on a machine with no internet connection.
You give it files. It stores tables in DuckDB, documents as searchable chunks, and the
entities in both as a knowledge graph. One agent answers across all three and shows
every step it took. It runs as one static binary with a terminal session, a print mode,
a web UI, a REST API, and an MCP server. The only other thing it needs is a local model
server such as Ollama.

The design is in [docs/design-doc.md](docs/design-doc.md).

## Install

Pick one:

- Download a static binary for Linux (x86_64, aarch64), macOS (Apple silicon), or Windows
  (x86_64, arm64) from the GitHub releases page. Each release includes `SHA256SUMS` and a
  build provenance attestation:

  ```bash
  gh attestation verify quack-<version>-<target>.tar.gz --owner smoketurner \
    --signer-workflow smoketurner/quack/.github/workflows/reusable-build.yml
  ```
- Run the container. `ghcr.io/smoketurner/quack` starts `quack serve`. Each release also
  attaches an image tarball for hosts that cannot reach a registry. `docker-compose.yml`
  runs the image next to Ollama. See [docs/ci-cd.md](docs/ci-cd.md).
- Build from source with `cargo build --release`. Linux needs `cmake` and `clang`.

## Configure

Name your models in `~/.config/quack/config.toml`. This is the smallest working setup
with Ollama:

```toml
[general]
chat_model = "ollama/gpt-oss:20b"
embedding_model = "ollama/qwen3-embedding:0.6b"

[providers.ollama]
type = "ollama"
embedding_dimension = 1024
```

quack supports Ollama, OpenAI-compatible endpoints, and Anthropic. A provider can use no
auth, an API key, or OAuth. `QUACK_CONFIG_DIR` and `QUACK_DATA_DIR` move the config and
data directories. Design doc section 13 documents every setting.

## Try it

Load the example workspace first. It takes about two minutes:

```bash
make demo-data
```

This loads NOAA's 2024 Storm Events Database: three linked tables, Census population
figures imported over HTTPS, the government documents that define every code in the
tables, an ontology, and a knowledge graph. Then ask:

```bash
quack -w storms -p "which states had the most direct deaths, and from what kind of weather?"
quack -w storms -p "chart property damage by month"
quack -w storms -p "what wind speeds define an EF3?" --mode query
quack -w storms graph path 58277 GSP
```

[examples/storms/README.md](examples/storms/README.md) walks through every feature on
this data.

## Use it

Load your own data:

```bash
quack ingest sales.csv -w sales          # CSV, Parquet, JSON, and XLSX become tables
quack ingest policy.pdf -w sales         # PDF, DOCX, PPTX, HTML, Markdown, and text become chunks
quack ingest policy.pdf --pin            # the full text goes into every prompt
quack import postgres://u:p@host/db --table orders --from public.orders   # snapshot a Postgres or SQLite query, or an https file
```

Ask questions:

```bash
quack -w sales                           # terminal session
quack -w sales -p "total sales by region" # one turn; answer to stdout, steps to stderr
quack -w sales -p "and by month?" -c     # continue the last session
quack -w sales -p "what is excluded?" --mode query   # answer only from the documents, with citations
quack -w sales -q "SELECT region, sum(total) FROM sales GROUP BY 1" -f csv   # SQL, no agent
cat orders.csv | quack -q "SELECT count(*) FROM stdin"   # piped data is the table `stdin` (--stdin waits for a slow producer)
```

Teach the agent your domain:

```bash
quack context edit                       # definitions and rules the agent follows
quack ontology propose                   # draft classes and relations from the tables, then review
quack graph extract -y                   # build the knowledge graph
quack graph search Kenya                 # walk it
```

Share the workspace:

```bash
quack serve --local                      # web UI, REST API, and MCP on http://127.0.0.1:8080
quack mcp -w sales                       # MCP over stdio for Claude Code and editors
quack okf export ./bundle                # one-way Markdown knowledge export (no data, no audit); `quack ingest DIR` restores the ontology and context
quack user add alice --admin             # users, tokens, members, and the audit log for `quack serve`
```

Exit codes: 0 ok, 1 error, 2 usage, 3 write refused, 4 auth required.

## How it works

- Retrieval combines vector search with quack's own BM25 index. No DuckDB extension is
  loaded at runtime.
- Every citation in an answer is checked against what was retrieved that turn.
- The agent asks before it writes. Print mode refuses writes unless you pass
  `--allow-write`. The API requires the write scope.
- Every turn is recorded in the workspace file. You can export a session as runnable SQL
  or a Markdown transcript.
- The knowledge graph records where every node and edge came from, and reports when it
  is provisional or out of date with the ontology.
- Everything that could reveal workspace content stays inside that workspace's file.
  The server keeps users, tokens, and an append-only access log separately.

## Develop

```bash
make lint            # clippy with warnings denied
make test            # the whole suite
make deny            # advisories, licenses, bans
make release-gates   # no ring or OpenSSL in the runtime tree
make css-build       # after a template change; the built CSS is committed
```

[CONTRIBUTING.md](CONTRIBUTING.md) has the conventions. [CLAUDE.md](CLAUDE.md) and
`.claude/rules/` have the details coding agents follow.

| Doc | Covers |
|---|---|
| [docs/design-doc.md](docs/design-doc.md) | The product and what is still open |
| [docs/architecture.md](docs/architecture.md) | The crates and modules |
| [docs/migrations.md](docs/migrations.md) | Schema versioning for `control.db` and the `_quack_` tables |
| [docs/crypto.md](docs/crypto.md) | aws-lc-rs as the only crypto provider |
| [docs/web-ui.md](docs/web-ui.md) | The web UI stack behind `quack serve` |
| [docs/ci-cd.md](docs/ci-cd.md) | CI, releases, container images, compose |

## License

Apache-2.0 or MIT, at your option.
