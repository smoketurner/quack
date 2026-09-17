# Crypto and TLS: aws-lc-rs only

quack uses aws-lc-rs as the single crypto provider for every TLS connection and for
hashing. OpenSSL and `ring` are kept out: `deny.toml` bans `openssl`, `openssl-sys`,
`native-tls`, and `ring`, and every TLS-capable dependency is enabled with its aws-lc-rs
feature. One audited, FIPS-capable provider, nothing to cross-compile for the static musl
build, and no ambiguity about which backend rustls picks at runtime.

## Where it is wired

- `quack_core::crypto::install_default_provider()` installs the rustls default provider
  and is the first call in `main`, before any TLS use. A second call in one process is an
  error, which the strict lints surface instead of an `unwrap`.
- Features: `rustls` with `aws_lc_rs`, `reqwest` and `rig` with `rustls`, `sqlx` with
  `tls-rustls-aws-lc-rs` (the Postgres import). SHA-256 for tokens and document dedup
  comes from `aws_lc_rs::digest`, AES-256-GCM for the OAuth token cache from
  `aws_lc_rs::aead`.

## The gate

```bash
make release-gates
```

runs `cargo tree -i ring -e normal` and `cargo tree -i openssl-sys -e normal`, which must
print nothing, then `cargo deny check`. The `-e normal` matters: `libduckdb-sys` pulls
`ureq`, and with it `ring`, as a build-time dependency only; nothing links it into the
binary. The release workflow runs the same target before building anything.
