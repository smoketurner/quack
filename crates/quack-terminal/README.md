# quack-terminal

The interactive terminal session: what `quack -w NAME` opens at a TTY. This crate is
everything that owns the screen.

## What it holds

- **The event loop** (`app.rs`): one `tokio::select!` over crossterm's events, the job
  broadcast, and a spinner tick. Every question, statement, and command runs as a job, so
  input never blocks.
- **Rendering** (`ui.rs`, `markdown.rs`, `chart.rs`, `picker.rs`): the transcript, the
  prompt, the job strip, Markdown, charts, and the session picker.
- **Selection and clipboard** (`selection.rs`, `clipboard.rs`): drag to select transcript
  text, then copy it to the system clipboard and to the terminal's (OSC 52).
- **Input parsing** (`commands.rs`, `sql.rs`): slash commands and direct SQL.

## What it does not hold

- **The slash commands' work.** `/graph`, `/ontology`, `/saved`, `/tables`, `/embeddings`,
  and `/import` parse into the action enums of [`quack-cli`](../quack-cli/README.md) and run
  their `run`. A command's logic lives there, so the shell and the terminal share it.
- **Server code.** This crate never depends on `quack-server`, and `quack-server` never
  depends on it, so an edit to one never recompiles the other.

## Public API

`run` and `SessionSetup` (the workspace, its writer and reader, the session, and the write
policy). The binary builds a `SessionSetup` and calls `run`.

## Where new code goes

- Anything that draws, reads keys, or uses ratatui or crossterm goes here.
- A new slash command parses here, but its work goes in `quack-cli` when the shell should
  offer the same verb.

Depends on `quack-core` and `quack-cli`.
