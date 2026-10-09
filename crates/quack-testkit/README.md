# quack-testkit

Test support for the interface crates. It is only ever a dev-dependency, so none of it ships
in the binary.

## What it holds

- **`ScriptedOllama`:** an Ollama stand-in on a loopback port. It answers `/api/chat` from a
  list of scripted replies (tool calls or text) and records each request, so a test can run
  a whole agent turn through an interface without a model.
- **`seed_dictating_note` and `following_the_note`:** a workspace document that tells the
  agent to drop a table, and the script that follows it. Tests use them to show that a write
  dictated by retrieved text is refused or asked about.

## Why it is a separate crate

`quack-cli` and `quack-server` both use it. Putting it in `quack-core` would turn axum into a
regular dependency of core and ship test code in the binary. Copying it into each crate would
duplicate it.

Used by: `quack-cli` and `quack-server`, as a dev-dependency.
