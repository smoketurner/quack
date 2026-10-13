# quack-mcp

The workspace as an MCP server (`rmcp`): the tools `query`, `search`, `sql`, `classify`,
`list_tables`, `describe_table`, and `list_documents`, and the `quack://workspace/...`
resources, over any transport. `quack mcp` serves it on stdio (`serve_stdio`), and
`quack-server` serves it over streamable HTTP under `/mcp/v1/{workspace}`. Design doc
section 11.3.

## What it holds

- **`McpServer` and `McpSetup`:** the server over a workspace's handles (writer, reader
  pool, queue), its write policy, and the transport it serves (`Transport::Stdio`, nobody to
  audit; `Transport::Http`, every call carries an `McpCaller`).
- **`McpCaller`:** who made one request over an audited transport: a core
  `storage::audit::Recorder` (both audit halves for that person in that workspace), which
  sessions they may read, and whom their model requests act for. The HTTP handler builds one
  per request and puts it in the request's extensions; a call without one fails rather than
  run unaudited.
- **`serve_stdio`:** `quack mcp`.

## What it does not hold

- **Transports beyond stdio.** The streamable HTTP transport, its per-user sessions, and the
  bearer and role checks in front of it are `quack-server`'s (`mcp_http.rs`, `state.rs`).
- **Audit policy.** What to audit and as whom is decided by whoever builds the `McpCaller`;
  this crate only records through it.
