# Crypto and TLS: aws-lc-rs only

quack uses aws-lc-rs as the single crypto provider for every TLS connection and for
hashing. OpenSSL and `ring` are kept out: `deny.toml` bans `openssl`, `openssl-sys`,
`native-tls`, and `ring`, and every TLS-capable dependency is enabled with its aws-lc-rs
feature. One audited, FIPS-capable provider, nothing to cross-compile for the static musl
build, and no ambiguity about which backend rustls picks at runtime.

## Where it is wired

- `quack_core::crypto::install_default_provider()` installs the rustls default provider
  and is the first call in `main`, before any TLS use. A second call in one process is an
  error, which the strict lints surface instead of an `unwrap`. On Linux it names
  `rustls::crypto::default_fips_provider()`, which exists only under rustls's `fips`
  feature and returns the same provider: dropping that feature becomes a build failure
  instead of a silent return to non-FIPS key exchange.
- `crypto::log_provider()` records the module behind the installed provider — the AWS-LC
  version and, for a FIPS build, the module version. It runs from `init_logging_at`, not
  next to the install: the install happens before any tracing subscriber exists, so a log
  there would go nowhere. It logs at `info`, which `quack serve` shows by default while the
  other subcommands (default `warn`) need `RUST_LOG=info`.
- `quack --version` prints the same module on its second line
  (`crypto::provider_description()`), so a FIPS binary is identifiable without turning on
  logging; `-V` stays the bare version. The AWS-LC library version is the one that matters
  for a CVE or a certificate — aws-lc-rs exposes no runtime API for its own crate version,
  which stays in `Cargo.lock`.
- Both numbers come from `aws_lc_rs::awslc_version()` and `aws_lc_rs::fips_version()`,
  added in [aws/aws-lc-rs#1167](https://github.com/aws/aws-lc-rs/pull/1167) and present in
  the pinned 1.18.1, so the pin cannot move backwards past that release.
  `fips_version()` is `None` for every `aws-lc-sys` build, which is what the `AWS-LC FIPS`
  label keys on; it is resolved from the headers at build time, so the log line asks
  `CryptoProvider::fips()` instead for what rustls actually installed.
- Features: `reqwest` and `rig` with `rustls`, `sqlx` with `tls-rustls-aws-lc-rs` (the
  Postgres import). SHA-256 for tokens and document dedup comes from `aws_lc_rs::digest`,
  AES-256-GCM for the OAuth token cache from `aws_lc_rs::aead`.
- `aws-lc-rs` and `rustls` sit in `[dependencies]` with the features every target shares
  (`crates/quack-core/Cargo.toml`), and the `cfg(target_os = "linux")` section adds `fips`
  to both — Cargo unions the feature sets, so Linux gets FIPS and nothing else changes.
  `crates/quack` declares neither: it installs the provider through `quack_core::crypto`
  and uses no rustls API of its own.
- `rustls` carries `prefer-post-quantum`, so `X25519MLKEM768` leads the key exchange list
  instead of trailing it. It survives the FIPS build too: that hybrid sends the ML-KEM
  share first (`post_quantum_first: true`), and rustls's `fips()` for a hybrid defers to
  whichever half comes first, which is approved when the library is in FIPS mode.

## FIPS on Linux

The Linux binaries quack distributes — the static musl ones for x86_64 and aarch64, and the
container image built from them — run on the FIPS-validated AWS-LC module. `quack-core`
enables `aws-lc-rs`'s and `rustls`'s `fips` features for `cfg(target_os = "linux")`; macOS
and Windows build against `aws-lc-sys`. What that means in practice:

- **The whole crate switches, not part of it.** `aws-lc-rs` binds to `aws-lc-fips-sys`
  through `extern crate aws_lc_fips_sys as aws_lc` when the feature is on, so the direct
  `digest`/`rand`/`aead` calls and rustls's provider all land on the validated module.
  rustls's `fips` feature adds the policy half: the cipher suite and key exchange lists
  narrow to the approved ones, and `CryptoProvider::fips()` becomes true. Startup logs the
  linked AWS-LC version and, for a FIPS build, the module version; a Linux binary that
  reports a non-FIPS provider logs a warning rather than refusing to start, and
  `crypto::tests::the_provider_is_fips_on_linux_and_not_elsewhere` fails the build if the
  target gating drifts.
- **macOS is excluded because of linkage, not tooling.** `aws-lc-fips-sys` emits a static
  library for a FIPS build only on Linux and BSD, and only on x86_64 and aarch64
  (`builder/main.rs`, `impl Default for OutputLibType`); everywhere else a FIPS build
  produces a shared library. A macOS FIPS binary would therefore depend on a
  `libcrypto.dylib` that the single-file release archive cannot carry. Windows is excluded
  for a second reason as well: `aws-lc-fips-sys` has no pre-generated bindings for either
  Windows target, so it would need bindgen with libclang, and the x86_64 one an assembler.
- **The Linux build needs `cmake` and `go`, and clang specifically.** Every
  `aws-lc-fips-sys` build runs AWS-LC's `delocate` pass over generated assembly, which is a
  Go program that cannot parse what gcc emits — it fails with `parse error near WS`.
  `AWS_LC_FIPS_SYS_CC=clang` and `AWS_LC_FIPS_SYS_CXX=clang++` are set in `Dockerfile`,
  `Dockerfile.build`, and both workflows' `env:` blocks, and `go` is installed alongside
  `cmake` in each Linux builder. Ninja is not needed; the cmake crate drives make. Building
  quack from source on Linux therefore requires Go in addition to CMake and a C++ compiler.
- **The FIPS sources carry the OpenSSL license**, because AWS-LC descends from OpenSSL via
  BoringSSL, so `deny.toml` holds an `exceptions` entry for `aws-lc-fips-sys`. That is a
  license, not the banned `openssl` crate: nothing links OpenSSL, which the gate below
  re-checks.

## The gate

```bash
make release-gates   # crypto-gates, then cargo deny check
```

runs `cargo tree -i ring -e normal` and `cargo tree -i openssl-sys -e normal`, which must
print nothing, then `cargo deny check`. The `-e normal` matters: `libduckdb-sys` pulls
`ureq`, and with it `ring`, as a build-time dependency only; nothing links it into the
binary. The release workflow runs the tree half as `make crypto-gates` before building
anything, and the deny half through the pinned cargo-deny action — the runner has no
cargo-deny binary of its own.
