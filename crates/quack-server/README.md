# quack-server

The network interfaces: `quack serve` (the REST API, the web UI, and MCP over HTTP) and
`quack mcp` (the MCP server on stdio). Both are thin clients of `quack-core`.

## What it holds

- **`api/`:** the REST API under `/api/v1`. Every handler carries a `#[utoipa::path]`
  annotation, and the OpenAPI document is generated from them (`api/openapi.rs`).
- **`web/`, `templates/`, `static/`:** the web UI.
  - `web/` holds the askama pages.
  - `static/` holds the committed Tailwind CSS and the vendored htmx, ECharts, and Redoc,
    embedded in the binary.
- **`styles/` and `editor/`:** the sources for the committed CSS and for the SQL page's
  CodeMirror bundle. Rebuild them with `make css-build` and `make editor-build`, then
  commit the output (`docs/web-ui.md`).
- **`auth.rs` and `oidc/`:** sign-in, sessions, and API tokens.
- **`mcp.rs` and `mcp_http.rs`:** the MCP server, on stdio and over HTTP. It shares the
  crate with the HTTP server because each uses the other.
- **`queue.rs`, `permissions.rs`, `state.rs`:** uploads on the job queue, held write
  approvals, and the shared application state.

## What it does not hold

- **Terminal code.** This crate never depends on `quack-terminal`, and `quack-terminal`
  never depends on it, so an edit to one never recompiles the other.
- **Engine logic.** Retrieval, the agent, the graph, and storage are in `quack-core`. A
  handler checks access, calls core, and records the audit row.

## Public API

`serve`, `ServeMode`, and `serve_stdio`. The binary calls them for `quack serve` and
`quack mcp`.

## Where new code goes

- A new route, page, or MCP tool goes here.
- The behavior behind it goes in `quack-core`, so every interface can reach it.

Depends on `quack-core` only. `quack-testkit` is a dev-dependency.
