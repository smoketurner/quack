# crates/

Two workspace members, picked up by `members = ["crates/*"]`; no Cargo features.

| Crate | Responsibility |
|---|---|
| `quack-core` | The engine: config, `control.db`, workspace DuckDB files, ingestion and import, hybrid retrieval, the agent and its tools, ontology and induction, the knowledge graph, OKF bundles, LLM providers and OAuth |
| `quack` | The one binary: print mode, `-q` SQL, the terminal session, the workspace commands, `quack serve` (REST API, web UI, MCP over HTTP), `quack mcp` on stdio, and the server administration commands |

The module map is in [`docs/architecture.md`](../docs/architecture.md). A new crate would
inherit the workspace baseline (`edition.workspace = true`, `[lints] workspace = true`) and
pull dependencies from the root `[workspace.dependencies]` menu, but the design puts every
surface into `quack` as a subcommand, so expect none.
