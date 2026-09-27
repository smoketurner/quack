# AGENTS.md

Guidance for AI coding agents lives in **[CLAUDE.md](CLAUDE.md)** — the stack, repository
layout, conventions, commands, and the review gates under `.claude/rules/`. Read it first.

## System dependencies (Linux)

Building the stack needs `cmake`, `clang`, and `go` (Linux builds the FIPS `aws-lc-rs`
module, whose delocate pass is a Go program), with `AWS_LC_FIPS_SYS_CC=clang` and
`AWS_LC_FIPS_SYS_CXX=clang++` set (`docs/crypto.md`). On Debian/Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y cmake clang golang-go
```

macOS runners and dev machines already have a suitable toolchain.
