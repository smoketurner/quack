# Contributing

## Development setup

Prerequisites:

- **Rust** — pinned in `rust-toolchain.toml`; install via [rustup](https://rustup.rs/)
- **cmake** + **clang** — build dependency of `aws-lc-rs`
- **go** (Linux only) — `aws-lc-fips-sys` runs AWS-LC's delocate pass over the generated
  assembly; Linux builds link the FIPS module (`docs/crypto.md`). Set
  `AWS_LC_FIPS_SYS_CC=clang`: delocate cannot parse gcc's output

Common commands (`make help` lists all):

```bash
make build   # cargo build --release
make fmt     # cargo fmt --all
make lint    # clippy with -D warnings
make test    # cargo test --workspace --all-features
make deny    # cargo deny check
```

## Before opening a PR

Run the gate:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace
cargo deny check
```

Then check your change against the review gates in
[`.claude/rules/code-standards.md`](.claude/rules/code-standards.md) — crypto (aws-lc-rs
only), data-layer rules, and workspace hygiene.

## Commit messages

[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/). See
[`.claude/rules/commits-and-issues.md`](.claude/rules/commits-and-issues.md) for the allowed
types and issue protocol.

## Branching

See [`.claude/rules/branching.md`](.claude/rules/branching.md). Never push to `main` — open a
PR from a feature branch.

## License

By contributing, you agree your contributions are dual-licensed under
[Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
