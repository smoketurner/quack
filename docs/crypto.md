# Crypto and TLS: aws-lc-rs only

aws-lc-rs is quack's single crypto provider for TLS and hashing. `deny.toml` bans
`openssl`, `openssl-sys`, `native-tls`, and `ring`, and every TLS-capable dependency
enables its aws-lc-rs feature. That gives one audited, FIPS-capable provider, nothing extra
to cross-compile for the static musl build, and one certain rustls backend at runtime.

## Where it is wired

- `quack_core::crypto::install_default_provider()` installs the rustls default provider.
  It is the first call in `main`, before any TLS use. A second call is an error, which the
  strict lints surface instead of an `unwrap`. On Linux it names
  `rustls::crypto::default_fips_provider()`, which exists only under rustls's `fips`
  feature, so dropping the feature fails the build instead of silently reverting to
  non-FIPS key exchange.
- `crypto::CryptoModule::log` records the module behind the installed provider: the AWS-LC
  version and, for a FIPS build, the module version. It runs from `init_logging_at`,
  since no tracing subscriber exists at install time. It logs at `info`: `quack serve`
  shows it by default; other subcommands (default `warn`) need `RUST_LOG=info`.
- `quack --version` prints the same module on its second line (`CryptoModule`'s
  `Display`), so a FIPS binary is identifiable without logging; `-V` stays the bare
  version. The AWS-LC library version is what matters for a CVE or a certificate. The
  aws-lc-rs crate version has no runtime API; it stays in `Cargo.lock`.
- Both numbers come from `aws_lc_rs::awslc_version()` and `aws_lc_rs::fips_version()`.
  [aws/aws-lc-rs#1167](https://github.com/aws/aws-lc-rs/pull/1167) added them in time for
  the pinned 1.18.1, so the pin cannot move back past that release. The `AWS-LC FIPS`
  label keys on `fips_version()`, which is `None` for every `aws-lc-sys` build. It is
  resolved from headers at build time, so the log line asks `CryptoProvider::fips()` what
  rustls actually installed.
- Features:
  - `reqwest` with `rustls`; `rig` sends through it (its `reqwest` transport, which
    takes TLS from those features).
  - The AWS SDK behind the Bedrock provider (`aws-config` and `aws-sdk-bedrockruntime`
    with `default-https-client`) on `aws-smithy-http-client`'s `rustls-aws-lc`. Linux adds
    `rustls-aws-lc-fips`, and there `llm::bedrock` selects `CryptoMode::AwsLcFips`.
    Bedrock's OpenAI-compatible APIs go through `reqwest` like every other provider.
  - SHA-256 for tokens and document dedup comes from `aws_lc_rs::digest`; HPKE for the
    vault comes from `rustls` (below).
- **Exception: SigV4.** AWS SigV4 authenticates Bedrock requests with an HMAC-SHA256 over
  the request. The AWS SDK (Converse, embeddings) and quack's own signer
  (`llm::bedrock::Signer`, the OpenAI-compatible APIs) both compute it with `aws-sigv4`.
  That crate uses RustCrypto's `hmac` and `sha2`, so the MAC runs outside the
  FIPS-validated module even on Linux; the TLS connection carrying it runs inside. Neither
  crate is `ring` or OpenSSL, so the gates pass. A deployment that needs every primitive
  inside the validated module should not configure a Bedrock provider.
- `quack_core::vault` seals data at rest with HPKE (RFC 9180, base mode):
  DHKEM(P-256, HKDF-SHA256), HKDF-SHA256, AES-256-GCM. The suite comes from rustls's
  `crypto::aws_lc_rs::hpke` (`DH_KEM_P256_HKDF_SHA256_AES_256`), which is aws-lc-rs
  underneath and one of the suites rustls keeps under its `fips` feature. The X25519 and
  ChaCha20-Poly1305 suites are not, so the suite must not change to them.
  - One key pair per data directory, in the OS keychain (entry `vault`) or
    `<data_dir>/vault.key` (0600).
  - Each value is sealed for a `vault::Purpose` (the HPKE `info`,
    `quack vault v1 <purpose>`) and a subject (the associated data). It records the key id
    that sealed it.
  - The caller stores the `Sealed` value on the right side of the classification boundary.
    Today there are two purposes, both in `control.db`: signed-in users'
    identity-provider tokens (`user_tokens`) and model providers' OAuth tokens from
    `quack auth login` (`provider_tokens`).
- `jsonwebtoken` verifies identity-provider access tokens presented to `quack serve` (RFC
  9728, design doc 12). It is pinned with `default-features = false` and only its
  `aws_lc_rs` feature (never `rust_crypto`), so its signature checks run on the same
  aws-lc-rs, the FIPS module on Linux. With one backend it selects its provider itself.
  Its `signature` dependency is RustCrypto's trait crate and holds no algorithms.
- `crates/quack` declares neither `aws-lc-rs` nor `rustls` at runtime: it installs the
  provider through `quack_core::crypto` and uses no rustls API of its own. Its tests sign
  access tokens the way an issuer would, with `aws-lc-rs` and `jsonwebtoken` as
  dev-dependencies only.
- `rustls` carries `prefer-post-quantum`, so `X25519MLKEM768` leads the key exchange list.
  It survives the FIPS build: the hybrid sends the ML-KEM share first
  (`post_quantum_first: true`), and rustls's `fips()` for a hybrid defers to that half,
  which FIPS mode approves.

## TLS versions and post-quantum readiness

What quack negotiates, from the resolved features (`rustls` with `aws_lc_rs`,
`prefer-post-quantum`, and `std`; `reqwest` with `rustls`):

- **Outbound only.** Every TLS connection quack makes is outbound: model providers, an
  OpenID Connect issuer, an `https://` import. `quack serve` speaks plain
  HTTP. Inbound TLS belongs to the proxy in front of it (`[server].trusted_proxies` names
  it), which is also where the inbound cipher policy lives.
- **TLS 1.3 and TLS 1.2**, rustls's default protocol versions, with aws-lc-rs's default
  cipher suites (AES-GCM and ChaCha20-Poly1305 AEADs; on the FIPS build, the module's
  approved subset). Nothing older is offered.
- **Key exchange:** `X25519MLKEM768`, the hybrid of X25519 and ML-KEM-768, is offered
  first (`prefer-post-quantum`), so a server that supports it gets a post-quantum key
  exchange; otherwise X25519 or a NIST curve. The hybrid survives the FIPS build.
- **Signatures stay classical.** Certificate verification accepts ECDSA (P-256, P-384),
  RSA (PKCS#1 v1.5 and PSS), and Ed25519 through rustls-webpki; no post-quantum signature
  scheme is offered or accepted yet, since none is deployed in the web PKI. quack's own
  signatures are classical too: ES256 (P-256) client assertions for `private_key_jwt`, and
  the identity-provider access tokens it verifies carry whatever asymmetric algorithm the
  issuer uses.
- **Data at rest:** the vault seals tokens and keys with HPKE (DHKEM P-256 with HKDF-SHA256
  and AES-256-GCM), classical key agreement under a key the OS keychain holds. A
  post-quantum KEM there is a format change for a later release.

## FIPS on Linux

quack's distributed Linux binaries run on the FIPS-validated AWS-LC module: the static
musl binaries for x86_64 and aarch64 and the container image built from them. In
`crates/quack-core/Cargo.toml`, `aws-lc-rs` and `rustls` sit in `[dependencies]` with the
features every target shares, and the `cfg(target_os = "linux")` section adds `fips` to
both. Cargo unions the feature sets, so Linux gets FIPS and nothing else changes. macOS and
Windows build against `aws-lc-sys`.

- **The whole crate switches.** With the feature on, `aws-lc-rs` binds to
  `aws-lc-fips-sys` through `extern crate aws_lc_fips_sys as aws_lc`. The direct
  `digest`/`rand` calls, rustls's provider, and rustls's HPKE (the vault) all land on the
  validated module. rustls's `fips` feature adds the policy half: the cipher suite and key
  exchange lists narrow to approved ones, and `CryptoProvider::fips()` becomes true. A
  Linux binary that reports a non-FIPS provider logs a warning at startup and still runs.
  `crypto::tests::the_provider_is_fips_on_linux_and_not_elsewhere` fails the build if the
  target gating drifts.
- **Linkage, not tooling, excludes macOS.** `aws-lc-fips-sys` emits a static library for a
  FIPS build only on Linux and BSD, on x86_64 and aarch64 (`builder/main.rs`,
  `impl Default for OutputLibType`). Everywhere else a FIPS build produces a shared
  library, and the single-file release archive cannot carry a macOS `libcrypto.dylib`.
  Windows has a second blocker: `aws-lc-fips-sys` has no pre-generated bindings for either
  Windows target, so it would need bindgen with libclang, and x86_64 an assembler.
- **The Linux build needs `cmake`, `go`, and clang.** Every `aws-lc-fips-sys` build runs
  AWS-LC's `delocate` pass over generated assembly. `delocate` is a Go program that cannot
  parse gcc output; it fails with `parse error near WS`. `AWS_LC_FIPS_SYS_CC=clang` and
  `AWS_LC_FIPS_SYS_CXX=clang++` are set in `Dockerfile`, `Dockerfile.build`, and both
  workflows' `env:` blocks, and each Linux builder installs `go` beside `cmake`. The cmake
  crate drives make; Ninja is not needed. A Linux source build needs Go, CMake, and a C++
  compiler.
- **The FIPS sources carry the OpenSSL license**, because AWS-LC descends from OpenSSL via
  BoringSSL, so `deny.toml` holds an `exceptions` entry for `aws-lc-fips-sys`. That is a
  license, not the banned `openssl` crate; the gate below re-checks that nothing links
  OpenSSL.

## The gate

```bash
make release-gates   # crypto-gates, then cargo deny check
```

It runs `cargo tree -i ring -e normal` and `cargo tree -i openssl-sys -e normal`, which
must print nothing, then `cargo deny check`. The `-e normal` matters: `libduckdb-sys`
pulls `ureq`, and with it `ring`, as a build-time dependency only; nothing links it into
the binary. The release workflow runs the tree half as `make crypto-gates` before building
anything. It runs the deny half through the pinned cargo-deny action, because the runner
has no cargo-deny binary.
