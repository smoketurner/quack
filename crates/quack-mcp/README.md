# quack-mcp

The workspace as an MCP server (`rmcp`): the tools `query`, `search`, `sql`, `classify`,
`list_tables`, `describe_table`, and `list_documents`, and the `quack://workspace/...`
resources, over any transport. `quack mcp` serves it on stdio (`serve_stdio`), every holder
but `quack serve` serves it on the workspace socket (`socket::SocketServer`), and
`quack-server` serves it over streamable HTTP under `/mcp/v1/{workspace}`. Design doc
section 11.3.

## What it holds

- **`McpServer` and `McpSetup`:** the server over a workspace's handles (writer, reader
  pool, queue), its write policy, the transport it serves (`Transport::Stdio` and
  `Transport::Socket`, nobody to audit; `Transport::Http`, every call carries an
  `McpCaller`), the holder's name for `initialize`, and the `Approvals` a `query` turn's
  write asks (the terminal session listens over the socket). A `query` turn is a `chat` job
  on the queue.
- **`McpCaller`:** who made one request over an audited transport: a core
  `storage::audit::Recorder` (both audit halves for that person in that workspace), which
  sessions they may read, and whom their model requests act for. The HTTP handler builds one
  per request and puts it in the request's extensions; a call without one fails rather than
  run unaudited.
- **`serve_stdio`:** `quack mcp`.
- **`socket::SocketServer`:** the accept loop over a core `forward::Listener`, one MCP
  session per connection.
- **`client::HolderClient`:** the other end: `quack -q` runs its statement through a
  holder's `sql` tool and `quack jobs` reads `quack://workspace/jobs`.

## What it does not hold

- **The HTTP transport's checks.** The streamable HTTP transport's per-user sessions and the
  bearer and role checks in front of it are `quack-server`'s (`mcp_http.rs`, `state.rs`).
- **The socket itself.** Binding, the owner and peer checks, and the path rule are
  `quack_core::forward`; which commands serve it is `quack-cli`'s `OpenedWorkspace`.
- **Audit policy.** What to audit and as whom is decided by whoever builds the `McpCaller`;
  this crate only records through it.
