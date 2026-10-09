# crates/

Three workspace members, picked up by `members = ["crates/*"]`; no Cargo features.

| Crate | Responsibility |
|---|---|
| `quack-core` | The engine: config, `control.db`, workspace DuckDB files, ingestion and import, hybrid retrieval, the agent and its tools, ontology and induction, the knowledge graph, OKF bundles, LLM providers and OAuth |
| `quack-testkit` | Test support for the interface crates: a scripted Ollama server and a seeded workspace, kept out of `quack-core` so core does not depend on axum for tests |
| `quack` | The one binary: print mode, `-q` SQL, the terminal session, the workspace commands, `quack serve` (REST API, web UI, MCP over HTTP), `quack mcp` on stdio, and the server administration commands |

The module map is in [`docs/architecture.md`](../docs/architecture.md). A new crate
inherits the workspace baseline (`edition.workspace = true`, `[lints] workspace = true`) and
pulls dependencies from the root `[workspace.dependencies]` menu.
