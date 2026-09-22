# quack

quack answers questions about your own data on a machine with no internet connection.
You give it files. It stores tables in DuckDB, documents as searchable chunks, and the
entities in both as a knowledge graph. One agent answers across all three and shows every
step it took. It runs as one static binary with a terminal session, a print mode, a web
UI, a REST API, and an MCP server, and needs only a local model server such as Ollama.

## What you can ask it

Most tools make you pick a lane: SQL over a database, or chat over a pile of PDFs. quack
puts both in one workspace, adds a graph of the entities in them, and lets one agent
choose. These run against the demo workspace below.

**Questions your tables answer.** The agent writes DuckDB SQL, shows you the statement,
runs it, and explains the rows.

```bash
quack -w storms -p "which states had the most direct deaths, and from what kind of weather?"
quack -w storms -p "chart property damage by month"     # returns a chart plus the numbers
```

**Questions your documents answer.** Retrieval is hybrid — vector similarity and a
keyword index — and every `[n]` in the answer is checked against a chunk actually
retrieved on that turn. `--mode query` forbids answering from anything else.

```bash
quack -w storms -p "what wind speeds define an EF3?" --mode query
quack -w storms -p "what is the difference between a direct and an indirect fatality?" --mode query
```

**Questions that need both at once.** This is the part a SQL tool and a document chatbot
each get half of: the tables hold the numbers, the documents say what the numbers mean.

```bash
quack -w storms -p "deaths per million residents by state, top ten"   # joins the storm table to Census figures imported over HTTPS
quack -w storms -p "how many people died in mobile homes, and do those count as direct deaths?"
```

**Questions about how things connect.** Once the graph is built the agent walks it —
neighborhoods, shortest paths, everything of a class — instead of guessing a join.

```bash
quack -w storms -p "which forecast offices recorded events that killed people in mobile homes?"
quack -w storms graph path 58277 GSP    # a fatality -> its flash flood -> the office that recorded it
```

**Follow-up questions.** Sessions live in the workspace file, resumable and exportable as
a Markdown transcript or as the SQL that ran.

```bash
quack -w storms -p "and how many of those were indirect?" -c
```

What it will not do: write to your data without asking, cite a source it did not
retrieve, reach the network while answering a question, or store anything that reveals a
workspace's contents outside that workspace's own file.

## Install

Download a static binary for Linux (x86_64, aarch64), macOS (Apple silicon), or Windows
(x86_64, arm64) from the releases page; each release carries `SHA256SUMS` and a build
provenance attestation you can check with `gh attestation verify`. Or run the container,
`ghcr.io/smoketurner/quack`, which starts `quack serve` and has a `docker-compose.yml`
that puts Ollama beside it. Or build it: `cargo build --release`, with `cmake` and a C++
compiler, plus `clang` and `go` on Linux.

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

quack supports Ollama, OpenAI-compatible endpoints, and Anthropic, with no auth, an API
key, or OAuth. `QUACK_CONFIG_DIR` and `QUACK_DATA_DIR` move the config and data directories.

## Try it

```bash
make demo-data      # about two minutes
```

That is NOAA's 2024 Storm Events Database: three linked tables, Census population figures
imported over HTTPS, the government documents defining every code in the tables, an
ontology, and a knowledge graph. Every question above works against it. [examples/](examples/)
has the walkthrough and a second workspace built from USAID supply chain shipments.

## Use it

```bash
quack ingest sales.csv -w sales          # CSV, Parquet, JSON, and XLSX become tables
quack ingest policy.pdf -w sales --pin   # PDF, DOCX, PPTX, HTML, Markdown, and text become chunks; --pin puts the full text in every prompt
quack import postgres://u:p@host/db --table orders --from public.orders   # snapshot a Postgres or SQLite query, or an https file
```

```bash
quack -w sales                           # terminal session
quack -w sales -q "SELECT region, sum(total) FROM sales GROUP BY 1" -f csv   # SQL, no agent
cat orders.csv | quack -q "SELECT count(*) FROM stdin"   # piped data is the table `stdin`
```

```bash
quack context edit                       # definitions and rules the agent follows
quack ontology propose                   # draft classes and relations from the tables, then review
quack graph extract -y                   # build the knowledge graph
```

```bash
quack serve --local                      # web UI, REST API, and MCP on http://127.0.0.1:8080
quack mcp -w sales                       # MCP over stdio for Claude Code and editors
quack user add alice --admin             # users, tokens, members, and the audit log for `quack serve`
```

```bash
quack config                             # every setting, the value in force, and where it came from
quack config --changed                   # only what config.toml and the environment set
quack doctor                             # check config, data dir, workspace, model providers, and bind address; say how to fix each
```

No configuration is needed to start: with no chat model, `quack` and `quack -q` still run
SQL, load files, and take every slash command, and a question says how to set a model up.

Exit codes: 0 ok, 1 error, 2 usage, 3 write refused, 4 auth required.

## How it works

- Retrieval combines vector search with quack's own BM25 index. No DuckDB extension is
  loaded at runtime, so the binary stays one file.
- The agent asks before it writes; print mode refuses writes without `--allow-write` and
  the API requires the write scope.
- The ontology is proposed from your tables and documents and accepted by a person;
  extraction is constrained by it, and every node and edge records where it came from.
- Everything that could reveal workspace content stays in that workspace's file. The
  server keeps users, tokens, and an append-only access log separately.

## Develop

```bash
make lint test deny   # clippy with warnings denied, the suite, advisories and licenses
make help             # everything else
```

[CONTRIBUTING.md](CONTRIBUTING.md) has the conventions and [docs/](docs/) has the rest.

## License

Apache-2.0 or MIT, at your option.
