# quack-cli

Command logic that runs the same way from a shell (`quack graph search X`) and from the
terminal session (`/graph search X`). Each command writes text to a `Write` and returns. It
never draws a screen or reads keys.

## What it holds

- **The shared commands:** `saved_cli`, `graph_cli`, `ontology_cli`, `tables_cli`,
  `embeddings_cli`, and `import_cli`. Each is a clap action enum plus a `run`. The terminal
  parses a slash command into the same enum the binary parses from its arguments, so both
  interfaces run one implementation.
- **What those commands need:** `Confirm` (y/n prompts and `-y`), `TextOrJson` and
  `QueryFormat` (output formats), `ModeArg` and `ExportFlags` (shared arguments), and
  `find_session`.
- **Print mode** (`PrintTurn`, one non-interactive agent turn) and **stdin handling**
  (`StdioPath`, `NamedInput`). Only the binary calls these directly, but they live here
  because `saved run --refresh` asks the model again through a `PrintTurn`.

## Public API

- The five command modules are public, because callers name them (`saved_cli::run`,
  `ontology_cli::run`).
- The other modules are private. The crate root re-exports the items the binary and the
  terminal import.
- The workspace lint `unreachable_pub` keeps everything else crate-internal.

## Where new code goes

- A verb that should work both from the shell and as a slash command goes here.
- Code that draws, reads keys, or uses ratatui or crossterm goes in
  [`quack-terminal`](../quack-terminal/README.md).
- A verb that only makes sense from a shell goes in [`quack`](../quack/README.md).

Depends on `quack-core` only. `quack-testkit` is a dev-dependency.
