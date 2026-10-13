# quack-server

The network interface: `quack serve` (the REST API, the web UI, and `quack-mcp` over
streamable HTTP), a thin client of `quack-core`.

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
- **`mcp_http.rs`:** `/mcp/v1/{workspace}`: the bearer and role checks in front of
  `quack-mcp`'s server, one transport per workspace, user, and write permission, and the
  `McpCaller` (core's audit `Recorder`) each request carries so every call is audited as the
  request that made it.
- **`run.rs`, `permissions.rs`, `state.rs`:** the server's end of core's audited background
  runs (`jobs::runs`) and held write approvals (`analysis::permissions`), uploads queued
  through `ingestion::upload` with `Access` as their owner, and the shared application
  state.

## What it does not hold

- **Terminal code.** This crate never depends on `quack-terminal`, and `quack-terminal`
  never depends on it, so an edit to one never recompiles the other.
- **Engine logic.** Retrieval, the agent, the graph, and storage are in `quack-core`. A
  handler checks access, calls core, and records the audit row.

## Public API

`serve`, `Server`, and `ServeMode`. The binary calls them for `quack serve`; `quack mcp`
is `quack_mcp::serve_stdio`.

## Where new code goes

- A new route or page goes here; an MCP tool or resource goes in `quack-mcp`.
- The behavior behind it goes in `quack-core`, so every interface can reach it.

Depends on `quack-core` only. `quack-testkit` is a dev-dependency.
