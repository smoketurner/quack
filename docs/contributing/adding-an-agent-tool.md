# Adding an agent tool

A tool is a rig `Tool` the agent may call during a turn. `read_document`
(`analysis/tools.rs`, `ReadDocumentTool`) is the worked pattern for a read-only tool;
`run_sql` for one that may write and must ask.

1. **`crates/quack-core/src/analysis/events.rs`**: the variant in `ToolName`, its text in
   the `text_enum!` block (the name the model calls and the audit records), and
   `ToolName::takes_sql` when the tool's detail is a SQL statement (the terminal and the web
   chat treat it as one).
2. **`crates/quack-core/src/analysis/tools.rs`**: the `Args` struct (`Deserialize +
   JsonSchema`, one doc comment per field: the model reads them), the tool struct holding
   only the workspace handles and settings it needs, and `impl Tool`. Inside `call`, take
   the turn with `Turn::of(context)?`, open a step with `turn.recorder.start(ToolName::X,
   detail)`, and finish it with `finish`, `finish_rows`, `finish_with_result`, or `fail`.
   Any text a document or the graph supplied goes to the model inside `text::Fenced`. A
   tool that reads document text sets `turn.read_documents` so a later write is asked for.
   Register the tool's own tests in `analysis/tools/tests.rs`.
3. **`crates/quack-core/src/analysis/agent.rs`**: register it on the `AgentBuilder` in
   `run_analysis`, and only when it applies (the graph tools register only when the graph
   has nodes).
4. **`crates/quack-core/src/analysis/text_to_sql.rs`**: the prompt's tool guidance, a
   numbered step saying when to call it.
5. **The `ToolName` matches**: `storage/sessions.rs` (how the step is stored and replayed),
   `crates/quack/src/print.rs` (the step line on stderr), `server/web/mod.rs` (the chat
   page's step view), and `terminal/app.rs` (the transcript's step message). The compiler
   lists every exhaustive match the new variant breaks.
6. **`crates/quack-core/tests/agent_turn.rs`**: a whole turn through rig's scripted model
   that calls the tool and checks the step, its summary, and the response.
7. **`crates/quack/src/mcp.rs`**, when the tool is also an MCP tool: its handler, audited
   with channel `mcp`.
8. **`docs/design-doc.md`** section 7.3's tool table, and `CLAUDE.md` where it lists the
   tools.

## What bites

- `clippy::absolute_paths`: `use` the type; `crate::a::b::Type` at a call site is denied.
- No `unwrap`, `expect`, indexing, or slicing outside tests; `#[expect(..., reason)]` in
  tests only.
- Date and time through `jiff`; ids through `uuid::Uuid::now_v7()`.
- A tool never writes an audit row itself: the step it records is what the interfaces audit
  and show. A tool that mutates must go through the write gate (`SqlGate`) so the
  permission rules apply.
- Anything the tool returns to the model is workspace content: it never reaches
  `control.db`, logs, or a URL.
