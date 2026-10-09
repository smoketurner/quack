# quack

The one binary. `main.rs` parses the arguments and hands each subcommand to the crate that
implements it. The release ships this crate's output as the single `quack` executable, with
every other crate linked in.

## What it holds

- **`main.rs`:** the clap definition, dispatch, `-q` SQL, the workspace commands, and the
  wiring into the other crates:
  - `-p` runs `quack_cli::PrintTurn`.
  - `serve` and `mcp` call `quack_server`.
  - A bare `quack` at a TTY calls `quack_terminal::run`.
- **Shell-only commands:** the ones the terminal session never offers.
  - `admin` (users, tokens, membership, audit)
  - `auth_cli` (provider OAuth and client registration)
  - `init_cli`
  - `config_cli`
  - `doctor_cli`
- **`tests/`:** integration tests that run the built binary (`CARGO_BIN_EXE_quack`).

## Where new code goes

- A verb that only makes sense from a shell goes here.
- A verb the terminal should also offer goes in [`quack-cli`](../quack-cli/README.md).
- Server and terminal code go in [`quack-server`](../quack-server/README.md) and
  [`quack-terminal`](../quack-terminal/README.md).

Depends on `quack-core`, `quack-cli`, `quack-server`, and `quack-terminal`.
