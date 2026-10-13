# crates/

Eight workspace members, picked up by `members = ["crates/*"]`; no Cargo features.

| Crate | Responsibility |
|---|---|
| `quack-core` | The engine: config, `control.db`, workspace DuckDB files, ingestion and import, hybrid retrieval, the agent and its tools, ontology and induction, the knowledge graph, OKF bundles, LLM providers and OAuth |
| `quack-cli` | The command verbs the binary and the terminal session share: print mode, the saved, graph, ontology, tables, embeddings, and import commands, and the arguments they take |
| `quack-server` | The HTTP interfaces: `quack serve` (REST API, web UI with its templates and assets, MCP over HTTP) and the MCP server on stdio; it holds no terminal code |
| `quack-terminal` | The interactive terminal session (ratatui); it holds no server code |
| `quack-desktop` | The desktop app: a Tauri window over the web UI, served in-process by `quack-server`'s `Embedded` through the `quack://` scheme with no listener; a second binary, since Tauri owns its binary crate and links GTK on Linux, which the static `quack` cannot carry |
| `quack-testkit` | Test support for the interface crates: a scripted Ollama server and a seeded workspace, kept out of `quack-core` so core does not depend on axum for tests |
| `quack` | The one binary: print mode, `-q` SQL, the workspace commands, the server administration commands, and the `main` that wires the interface crates into subcommands |
| `quack-eval` | The evaluation harness (`make eval`): a binary that drives the built `quack` as a person would and scores its JSON; it links nothing of quack. Its fixture lives in `fixture/` |

Each crate's README says what belongs in it and where new code goes. The short rule:
- Behavior that more than one interface needs goes in `quack-core`.
- A verb for both the shell and the terminal goes in `quack-cli`.
- Anything that owns the screen goes in `quack-terminal`, or in `quack-desktop` for the app's window.
- Routes, pages, and MCP go in `quack-server`.
- Shell-only verbs go in `quack`.

The module map is in [`docs/architecture.md`](../docs/architecture.md). A new crate
inherits the workspace baseline (`edition.workspace = true`, `[lints] workspace = true`) and
pulls dependencies from the root `[workspace.dependencies]` menu.
