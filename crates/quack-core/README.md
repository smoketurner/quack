# quack-core

The engine. Every interface (the shell commands, the terminal session, the REST API, the web
UI, and MCP) is a thin client of this library.

## What it holds

- **Storage:** config, `control.db`, and the workspace DuckDB files.
- **Data in:** ingestion and import.
- **Retrieval and the agent:** hybrid retrieval, and the agent with its tools and event
  stream.
- **Knowledge:** the ontology and the knowledge graph.
- **Providers:** model providers and OAuth.
- **Background work:** the job queue.

[`docs/architecture.md`](../../docs/architecture.md) maps its modules.

## Why it stays one crate

Its modules import each other in cycles. For example, storage uses the graph and the
ontology, and those use storage and the model providers. Splitting it would first mean
moving shared types into a base crate and reversing many call directions.

## Where new code goes

Behavior that more than one interface needs goes here, so the shell, the terminal, the API,
and MCP all reach it the same way. Code that only presents that behavior goes in an
interface crate.

It depends on no other workspace crate.
