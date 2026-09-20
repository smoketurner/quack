# CI/CD

## Continuous integration

- **`.github/workflows/ci.yml`** — `fmt`, `clippy` (`--locked -D warnings`), `test`
  (`cargo test --locked`, Linux + macOS), `dependency-review` (PRs), and `license-check`
  (`cargo-deny check`). Toolchain from `rust-toolchain.toml` via `rustup show`; caching via
  `Swatinem/rust-cache` (see [Build caching](#build-caching)); actions SHA-pinned;
  `permissions: {}` top-level with per-job `contents: read`.
- **`.github/workflows/secure_workflows.yml`** — fails CI if any third-party action is used
  without a full commit-SHA pin (`zgosalvez/github-actions-ensure-sha-pinned-actions`).
- **`.github/dependabot.yml`** — `cargo`, `github-actions`, and `docker` (the base-image
  tags), weekly, grouped, 7-day cooldown.

Work lands as commits on the main branch; CI runs on push. Releases are the only other
workflow, and it runs on tags alone so it never spends minutes on ordinary pushes.

### Build caching

Compiling DuckDB's C++ amalgamation (`libduckdb-sys`) takes around nine minutes and is
most of what any cold job here does, so the whole caching layout exists to keep that one
build script's output. Two constraints shape it:

- **The repository gets 10 GB of Actions cache in total.** Past that GitHub evicts
  least-recently-used entries, and it does so mid-run: a job that saves a fresh entry can
  evict the one a parallel job is about to restore. Every `Cargo.lock` change starts a new
  generation of entries, so the steady state has to leave room for two.
- **Only an exact key hit preserves the build-script output.** `rust-cache`'s restore-key
  fallback recovers the registry and some artifacts, but `libduckdb-sys` re-runs, so a
  near-miss costs the full nine minutes. An entry that survives to be hit exactly is worth
  more than a marginally better-shaped one that gets evicted.

What follows from that:

- **`CARGO_PROFILE_DEV_DEBUG: "1"`** (line tables only) roughly halves each cached
  `target/`, which is what brings the total inside the budget; test backtraces keep file
  and line numbers. `rust-cache` hashes every `CARGO_*` variable into the key, so
  `release.yml` sets it identically or its gates job cannot restore what CI saved.
- **One entry per job that compiles**: `v1-clippy-<os>` and `v1-debug-<os>`. Folding
  clippy into the test job would halve the number of entries, but the two then run in
  sequence rather than side by side — measured cold, 11m36s of clippy plus 13m08s of
  tests against about 13 minutes for the same work in parallel. Separate entries are
  affordable at line-tables-only debuginfo, so the jobs stay parallel.
- **Only pushes to `main` save** (`save-if`); pull requests restore. Tag refs can read
  caches from `main` but write their own scope, which nothing reads back — so no release
  job writes a cache, and none should.
- **`fmt` is not cached.** `cargo fmt` runs `cargo metadata --no-deps`, so it touches
  neither the registry nor `target/`.
- **Nothing caches a `--release` build.** CI builds only the dev profile, so a
  `release-<target>` key would have no writer, and funding one would evict the entries
  that keep every push fast. The macOS and Windows release builds are cold by design.

If cache pressure returns, `gh api repos/smoketurner/quack/actions/cache/usage` reports the
total and `gh cache list` the entries; stale generations can go with `gh cache delete`.

## Releases (`release.yml` + `reusable-build.yml`)

Trigger: an annotated `v*` tag pushed to the repository (for example `v0.2.0`), or
`workflow_dispatch` for a dry run that builds everything and publishes nothing.

Building and publishing are deliberately separate workflows. Everything that compiles,
signs, or attests runs in `reusable-build.yml`, called from `release.yml` through GitHub's
self-repository syntax (`uses: $/.github/workflows/reusable-build.yml`, which pins the
called workflow to the caller's commit). No job in it holds `contents: write`; the release
is created afterwards by a job that only downloads artifacts. That split is what makes the
provenance SLSA Build Level 3: the Sigstore certificate on every attestation names
`reusable-build.yml`, so a consumer can require that identity.

`release.yml`:

1. **gates** — `cargo fmt --check`, clippy, the test suite (restoring CI's `v1-check-Linux`
   cache read-only, which is why its `CARGO_*` environment has to match `ci.yml`),
   `make crypto-gates`
   (`cargo tree -i ring -e normal` and `-i openssl-sys -e normal` must be empty: aws-lc-rs
   is the only crypto provider, design doc section 14), and `cargo deny check` through the
   pinned action. The job runs `crypto-gates` rather than `release-gates` because the latter
   also shells out to `cargo deny`, which is not installed on the runner.
2. **version** — the tag without its `v`, or `0.0.0-<short sha>` for a manual run.
3. **build** — calls `reusable-build.yml`. Its `permissions:` block is the ceiling for the
   called jobs (`contents: read`, `packages: write`, `id-token: write`,
   `attestations: write`), and the signing secrets are forwarded there explicitly; each is
   declared optional, so a missing credential downgrades that platform to an unsigned
   binary with a warning instead of failing the release.
4. **publish** (tag only, `contents: write`, no checkout) — downloads the build artifacts,
   verifies each archive against the `.sha256` file written on the machine that built it,
   consolidates them into `SHA256SUMS`, and runs `gh release create --generate-notes
   --verify-tag` over the archives, SBOMs, image tarballs, and `SHA256SUMS`.

`reusable-build.yml`:

1. **binaries** — one job per target, each on a native runner, no cross-compilation
   (DuckDB and aws-lc-rs both compile C/C++ from source):

   | Target | Runner | Build |
   | --- | --- | --- |
   | `x86_64-unknown-linux-musl` | `ubuntu-latest` | `docker buildx bake ci` |
   | `aarch64-unknown-linux-musl` | `ubuntu-24.04-arm` | `docker buildx bake ci` |
   | `aarch64-apple-darwin` | `macos-latest` | `cargo build --release --locked` |
   | `x86_64-pc-windows-msvc` | `windows-latest` | `cargo build --release --locked` |
   | `aarch64-pc-windows-msvc` | `windows-11-arm` | `cargo build --release --locked` |

   The Linux builds are reproducible static musl binaries through `Dockerfile.build` and
   `docker-bake.hcl` (`rust:<MSRV>-alpine`, cargo-chef, `SOURCE_DATE_EPOCH` from the
   commit) with a CycloneDX SBOM (`cargo-cyclonedx`, installed in the base layer so a
   source change does not rebuild it) beside the binary. Within one run, cargo-chef keeps
   the cooked dependency layer off the critical path; across runs there is no layer cache,
   for the reason given under [Build caching](#build-caching). The Linux builds
   link the FIPS module, so their builders carry `go` next to `cmake` and set
   `AWS_LC_FIPS_SYS_CC=clang` — see [crypto.md](crypto.md). No runner installs an assembler:
   on Windows x86_64, rustls's `aws_lc_rs` feature turns on `aws-lc-rs/prebuilt-nasm`, so
   aws-lc-sys links its shipped objects instead. Each job then signs (see below), archives
   (`.tar.gz`, or `.zip` built with 7-Zip on Windows), writes a `.sha256`, attests build
   provenance for the archive, and attests the SBOM where there is one.
2. **image** — one job per architecture on its own native runner (`ubuntu-latest`,
   `ubuntu-24.04-arm`), so nothing is emulated: it unpacks that architecture's musl
   archive, builds `Dockerfile.release` (distroless static, nonroot, `/quack` and `/data`),
   pushes by digest on a tag, and exports a `docker load` tarball for air-gapped hosts,
   attested like the binaries.
3. **image-index** (tag only) — `docker buildx imagetools create` joins the per-architecture
   digests into `ghcr.io/smoketurner/quack:<version>` and `:latest`, then attests the index
   digest to the registry.

Code signing, all optional and all checked in one step per job that warns into the job
summary when a credential is missing:

- **macOS** — the Developer ID certificate is imported into a throwaway keychain
  (`APPLE_CERTIFICATE_P12_BASE64`, `APPLE_CERTIFICATE_PASSWORD`, `APPLE_CERTIFICATE_NAME`,
  `MACOS_KEYCHAIN_PASSWORD`), `codesign --options runtime --timestamp` signs the binary,
  and `notarytool submit --wait` notarizes it (`APPLE_ID`, `APPLE_ID_PASSWORD`,
  `APPLE_TEAM_ID`). A bare binary cannot be stapled, so the ticket stays with Apple and
  Gatekeeper checks it online.
- **Windows** — Azure Artifact Signing (formerly Trusted Signing) over GitHub OIDC
  (`TRUSTED_SIGNING_ENDPOINT`, `TRUSTED_SIGNING_ACCOUNT`, `CERTIFICATE_PROFILE`,
  `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`), signing as Smoke Turner, LLC with the certificate
  profile every smoketurner project shares. The signing tool is an x86_64 executable,
  which the arm64 runner runs under emulation. It runs on tag builds only: the federated
  identity credential (`smoketurner-infra`, `environments/azure/code_signing.tf`) trusts
  the exact subject
  `repo:smoketurner@8753311/quack@1366961169:workflow:Release:ref_type:tag`, so this
  repository's OIDC subject customization has to include the `repo`, `workflow`, and
  `ref_type` claims, and a `workflow_dispatch` run is deliberately left unsigned rather
  than failing the token exchange.

Verify a published asset or the image:

```bash
gh attestation verify quack-<version>-<target>.tar.gz --owner smoketurner \
  --signer-workflow smoketurner/quack/.github/workflows/reusable-build.yml
gh attestation verify oci://ghcr.io/smoketurner/quack:<version> --owner smoketurner \
  --signer-workflow smoketurner/quack/.github/workflows/reusable-build.yml
```

Every action is SHA-pinned. No job in `reusable-build.yml` reads or writes an Actions
cache, and `release.yml`'s gates job runs `rust-cache` restore-only (`save-if: false`), so
a release never writes a cache a later run could be poisoned by.

## Container images

- **`Dockerfile`** — the image from source, for `make image` and the compose file: a CSS
  stage rebuilds the Tailwind stylesheet with the standalone binary (checksum verified),
  cargo-chef caches the dependency build, the musl binary lands in
  `gcr.io/distroless/static-debian13:nonroot`. Environment: `QUACK_DATA_DIR=/data`,
  `QUACK_CONFIG_DIR=/config`, `QUACK_BIND=0.0.0.0:8080`; entrypoint `/quack`, default
  command `serve`, so `docker run ... quack user add alice --admin` also works.
- **`Dockerfile.release`** — the same runtime from prebuilt `dist/linux-<arch>/quack`
  binaries; no compilation, so multi-arch builds need no emulation.
- **`.dockerignore`** — a deny-by-default allowlist so the context stays small and
  cache-stable.
- Keep the `rust:<version>-alpine` tag equal to `rust-toolchain.toml`.

## Compose

`docker-compose.yml` runs `quack serve` beside `ollama/ollama`, with `deploy/config.toml`
mounted at `/config` (chat and embedding models on the `ollama` service) and named volumes
for `/data` and the models. First run:

```bash
docker compose up -d
docker compose exec ollama ollama pull gpt-oss:20b
docker compose exec ollama ollama pull qwen3-embedding:0.6b
docker compose exec quack /quack user add alice --admin
```

On an air-gapped host, `docker load < quack-image-<version>-linux-amd64.tar`, copy the
Ollama model directory into the `ollama` volume, and `docker compose up -d` with the
`image:` line pointing at the loaded tag.

## Local checks

- `actionlint` and `zizmor .github/workflows/*.yml` before committing a workflow change;
  both run clean on the shipped files. `.github/actionlint.yaml` ignores one finding:
  actionlint 1.7 does not yet know the `$/` self-repository `uses:` form that zizmor asks
  for.
- `docker buildx build --check -f Dockerfile .` (and `Dockerfile.release`,
  `Dockerfile.build`) validates the Dockerfiles without building.
- A `workflow_dispatch` run of the release workflow is the only way to exercise the macOS
  and Windows builds: CI itself builds on Linux and macOS only. Run one before tagging
  after any change to the build.
- Pin every new action to a SHA (`secure_workflows.yml` enforces it) — resolve current SHAs
  with `gh api repos/<owner>/<repo>/commits/<tag> --jq .sha`.

Two more patterns to reach for when the code calls for them:

- **Fuzzing** — once a parser takes untrusted input at scale, add a detached `fuzz/` crate
  (`cargo-fuzz` + `libfuzzer-sys`, its own empty `[workspace]`) and gitignore
  `fuzz/corpus/` and `fuzz/artifacts/`.
- **Docs site** — when `docs/` outgrows flat files, migrate to mdBook (`docs/book.toml` +
  `src/SUMMARY.md`), deploy via a GitHub Pages workflow, and gitignore `docs/book/`.
