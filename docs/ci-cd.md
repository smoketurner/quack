# CI/CD

## Continuous integration

- **`.github/workflows/ci.yml`**: `fmt`, `clippy` (`--locked -D warnings`), `test`
  (`cargo test --locked`, Linux + macOS cached, Windows uncached: a third debug cache would
  push the entries past the 10 GB Actions budget), `docs` (`cargo doc --no-deps` with
  `RUSTDOCFLAGS=-D warnings`, the same as `make doc`; `[workspace.lints.rustdoc]` denies
  broken and private intra-doc links), `dependency-review` (PRs), and `license-check`
  (`cargo-deny check`). The toolchain comes from `rust-toolchain.toml` via `rustup show`;
  `Swatinem/rust-cache` caches builds (see [Build caching](#build-caching)). Actions are
  SHA-pinned. `permissions: {}` is set at the top, with `contents: read` per job.
- **`.github/workflows/secure_workflows.yml`**: fails CI if a third-party action lacks a
  full commit-SHA pin (`zgosalvez/github-actions-ensure-sha-pinned-actions`).
- **`.github/dependabot.yml`**: `cargo`, `github-actions`, and `docker` (the base-image
  tags), weekly, grouped, 7-day cooldown.

CI runs on pushes to main. The release workflow runs on tags alone, so ordinary pushes
spend no release minutes.

### Concurrency

Every workflow declares a `concurrency` group. It cancels or queues depending on what a
cancelled run would leave behind.

- **`ci.yml` and `secure_workflows.yml` cancel.** A new commit supersedes the previous run,
  so the group is `<workflow>-<ref>` with `cancel-in-progress`. `ci.yml` has two
  exceptions:
  - It does not cancel on `refs/heads/main`. Cancelling a job skips its
    `Post Setup Rust cache` step, and `main` is the only ref that saves a cache, so
    cancelling there would starve every other run's restore.
  - It does not cancel a `merge_group` run, because a cancelled required check drops the
    entry from the queue.
- **`release.yml` cancels a run of the same tag and queues a run of another.** One group
  cannot do both, so the groups are per job:
  - `gates` and `build` share `release-<ref>` with `cancel-in-progress`. Tags are
    immutable and a release is created once, so an older run of the same tag would only
    fail at `publish`, after a full build.
  - `publish`, `publish-packages`, and `homebrew` are never cancelled by a newer run.
    After `gh release create`, cutting the run short would leave a release without its
    packages or formula. A newer run of a tag already past `build` fails at `publish`.
  - `publish-packages` (`release-packages`), `homebrew` (`release-homebrew`), and
    `reusable-build.yml`'s `image-index` (`release-image-index`) queue on constant groups,
    so two tags never race on the git pushes or on `imagetools create --tag latest`.

`ci.yml` and `secure_workflows.yml` scope `push:` to `main`, so a branch with an open pull
request runs each once, on `pull_request`.

### Merge queue

`main` sits behind a merge queue (the `main` ruleset, squash merges, `ALLGREEN` grouping).
Two settings must line up, or the queue gates nothing:

1. **`ci.yml` triggers on `merge_group`.** The queue builds each entry on its own
   `refs/heads/gh-readonly-queue/main/...` ref and dispatches `merge_group`, which is
   neither `push` nor `pull_request`. Without the trigger, no workflow runs on that ref.
2. **The ruleset lists the checks.** The queue waits only on checks the branch ruleset
   marks required; with an empty list it merges an entry the moment it is enqueued. The
   required set is `Format`, `Clippy`, `Unit Tests (linux)`, `Unit Tests (macos)`, and
   `License, Advisory & Ban Check`:

   ```bash
   gh api repos/smoketurner/quack/rulesets/23731779 --jq '.rules'   # inspect
   ```

Two jobs are **not** required. A required check that never reports deadlocks the entry for
the full `check_response_timeout_minutes`.

- **`Dependency Review`** is `if: github.event_name == 'pull_request'`, and
  `dependency-review-action` works only in a pull request context, so it skips on a merge
  group.
- **`Harden Security`** (`secure_workflows.yml`) filters on `paths: .github/workflows/**`,
  so it never starts for a merge group that touches no workflow file, and it has no
  `merge_group` trigger. Nothing is lost: combining pull requests cannot introduce an
  unpinned action the per-pull-request run missed.

Order matters when rebuilding this setup: adding the required checks before `ci.yml`'s
`merge_group` trigger is on `main` deadlocks the queue.

### Build caching

The caching layout exists to keep one build script's output: DuckDB's C++ amalgamation
(`libduckdb-sys`), about nine minutes and most of any cold job. Two constraints shape it:

- **The repository gets 10 GB of Actions cache in total.** Past that, GitHub evicts
  least-recently-used entries, even mid-run: a job saving a fresh entry can evict the one a
  parallel job is about to restore. Every `Cargo.lock` change starts a new generation of
  entries, so the steady state must leave room for two.
- **Only an exact key hit keeps the build-script output.** `rust-cache`'s restore-key
  fallback recovers the registry and some artifacts, but `libduckdb-sys` re-runs, so a
  near-miss costs nine minutes. A surviving entry beats a better-shaped evicted one.

The resulting rules:

- **`CARGO_PROFILE_DEV_DEBUG: "1"`** (line tables only) roughly halves each cached
  `target/`, which fits the total inside the budget. Test backtraces keep file and line
  numbers. `rust-cache` hashes every `CARGO_*` variable into the key, so `release.yml` sets
  it identically; otherwise its gates job cannot restore what CI saved.
- **One entry per compiling job**: `v1-clippy-<os>` and `v1-debug-<os>`. Folding clippy
  into the test job would halve the entries but serialize the work: measured cold, 11m36s
  of clippy plus 13m08s of tests, against about 13 minutes in parallel. Line-tables-only
  debuginfo makes separate entries affordable.
- **Only pushes to `main` save** (`save-if`); pull requests restore. Tag refs read
  `main`'s caches but write to their own scope, which nothing reads back, so release jobs
  write none.
- **`fmt` is not cached.** `cargo fmt` runs `cargo metadata --no-deps`, which touches
  neither the registry nor `target/`.
- **Nothing caches a `--release` build.** CI builds only the dev profile, so a
  `release-<target>` key would have no writer, and funding one would evict the entries that
  keep pushes fast. The macOS and Windows release builds are cold by design.

If cache pressure returns, `gh api repos/smoketurner/quack/actions/cache/usage` reports the
total and `gh cache list` the entries. Remove stale generations with `gh cache delete`.

## Releases (`release.yml` + `reusable-build.yml`)

Trigger: an annotated `v*` tag pushed to the repository (for example `v2026.10.3`; the scheme is `vYYYY.M.N`, the Nth release of that month), or
`workflow_dispatch` for a dry run that builds everything and publishes nothing.

Separate build and publish workflows make the provenance SLSA Build Level 3. Everything
that compiles, signs, or attests runs in `reusable-build.yml`, which holds no
`contents: write`. `release.yml` calls it through GitHub's self-repository syntax
(`uses: $/.github/workflows/reusable-build.yml`, pinned to the caller's commit); a later job
that only downloads artifacts creates the release. The Sigstore certificate on every
attestation names `reusable-build.yml`, so a consumer can require that identity.

`release.yml`:

1. **gates**: `cargo fmt --check`, clippy, the test suite, `make crypto-gates`, and
   `cargo deny check` through the pinned action. On a tag it then renders the release
   notes: this tag's section of `docs/upgrading.md` (`scripts/upgrading-section.sh`; a
   missing section fails the release) above the commits since the previous tag, grouped by
   Conventional Commit type by git-cliff (`cliff.toml`, the same output `CHANGELOG.md`
   holds), uploaded as the `release-notes` artifact.
   - The tests restore CI's `v1-check-Linux` cache read-only, which is why the job's
     `CARGO_*` environment must match `ci.yml`.
   - `make crypto-gates` requires `cargo tree -i ring -e normal` and
     `-i openssl-sys -e normal` to be empty: aws-lc-rs is the only crypto provider (design
     doc section 14). The job runs `crypto-gates`, not `release-gates`, because the latter
     also shells out to `cargo deny`, which the runner lacks.
2. **version**: the tag without its `v`, or `0.0.0-<short sha>` for a manual run.
3. **build**: calls `reusable-build.yml`. Its `permissions:` block caps the called jobs
   (`contents: read`, `packages: write`, `id-token: write`, `attestations: write`,
   `artifact-metadata: write`). The signing secrets are forwarded explicitly, each declared
   optional: a missing credential downgrades that platform to an unsigned binary with a
   warning instead of failing the release.
4. **publish** (tag only, `contents: write`, no checkout): downloads the build artifacts,
   verifies each archive against the `.sha256` written on its build machine, consolidates
   them into `SHA256SUMS`, and runs `gh release create --generate-notes --verify-tag` over
   the archives, SBOMs, packages, image tarballs, and `SHA256SUMS`. The step sets
   `GH_REPO`: with no checkout, `gh` has no git remote and fails with "not a git
   repository".
5. **publish-packages** (tag only, after publish, no `GITHUB_TOKEN` permissions):
   - Checks out `smoketurner/packages` with `PACKAGES_REPO_TOKEN` (a fine-grained token with
     contents: write on that repository alone).
   - Copies the `.deb` files into `apt/pool/main/` and each `.rpm` into `rpm/x86_64/` or
     `rpm/aarch64/`.
   - Regenerates the APT indices (`dpkg-scanpackages`, `apt-ftparchive release`) and the
     RPM `repodata/` (`createrepo_c --update`).
   - Signs `Release` (as `InRelease` and `Release.gpg`) and each `repomd.xml` with the
     packages key, and pushes `quack <version>`.

   A token push, unlike a `GITHUB_TOKEN` one, triggers that repository's
   `publish-to-s3.yml`, which serves it as packages.smoketurner.com.

`reusable-build.yml`:

1. **binaries**: one job per target, each on a native runner. Nothing cross-compiles,
   because DuckDB and aws-lc-rs both compile C/C++ from source.

   | Target | Runner | Build |
   | --- | --- | --- |
   | `x86_64-unknown-linux-musl` | `ubuntu-latest` | `docker buildx bake ci` |
   | `aarch64-unknown-linux-musl` | `ubuntu-24.04-arm` | `docker buildx bake ci` |
   | `aarch64-apple-darwin` | `macos-latest` | `cargo build --release --locked` |
   | `x86_64-pc-windows-msvc` | `windows-latest` | `cargo build --release --locked` |
   | `aarch64-pc-windows-msvc` | `windows-11-arm` | `cargo build --release --locked` |

   - The Linux builds are reproducible static musl binaries through `Dockerfile.build` and
     `docker-bake.hcl` (`rust:<MSRV>-alpine`, cargo-chef, `SOURCE_DATE_EPOCH` from the
     commit). A CycloneDX SBOM sits beside each binary (`cargo-cyclonedx`, installed in the
     base layer so a source change does not rebuild it).
   - cargo-chef keeps the cooked dependency layer off the critical path within one run.
     There is no layer cache across runs (see [Build caching](#build-caching)).
   - The Linux builds link the FIPS module, so their builders carry `go` next to `cmake`
     and set `AWS_LC_FIPS_SYS_CC=clang` (see [crypto.md](crypto.md)).
   - No runner installs an assembler. On Windows x86_64, rustls's `aws_lc_rs` feature turns
     on `aws-lc-rs/prebuilt-nasm`, so aws-lc-sys links its shipped objects.
   - Each job then signs (below), archives (`.tar.gz`, or `.zip` built with 7-Zip on
     Windows), writes a `.sha256`, attests build provenance for the archive, and attests
     the SBOM where there is one.
2. **linux-packages**: one job per architecture, both on `ubuntu-latest`, since nfpm only
   archives the prebuilt binary.
   - It unpacks that architecture's musl archive and builds `quack_<version>_<arch>.deb`
     and `quack-<version>-1.<arch>.rpm` from `packaging/nfpm.yaml` with nfpm (downloaded
     from its GitHub release, checksum pinned).
   - It signs the RPM with the packages.smoketurner.com key (`GPG_PRIVATE_KEY`,
     `GPG_PASSPHRASE`, set by smoketurner-infra's `environments/github` root) and checks
     the signature with `rpmkeys --checksig`.
   - Each package gets a `.sha256` and a build provenance attestation.
   - The key is optional on a dry run, which leaves the RPM unsigned with a warning. It is
     required on a tag, because the repository sets `gpgcheck=1`. The `.deb` is unsigned:
     APT checks the signed `Release` instead.
3. **image**: one job per architecture on its own native runner (`ubuntu-latest`,
   `ubuntu-24.04-arm`), so nothing is emulated. It unpacks that architecture's musl
   archive, builds `Dockerfile.release` (distroless static, nonroot, `/quack` and `/data`),
   pushes by digest on a tag, and exports a `docker load` tarball for air-gapped hosts,
   attested like the binaries.
4. **image-index** (tag only): `docker buildx imagetools create` joins the per-architecture
   digests into `ghcr.io/smoketurner/quack:<version>` and `:latest`, then attests the index
   digest to the registry. Pushing the attestation also writes an artifact metadata storage
   record, so this job alone carries `artifact-metadata: write`.

Code signing is optional; one step per job checks the credentials and warns into the job
summary when one is missing.

- **macOS**: imports the Developer ID certificate into a throwaway keychain
  (`APPLE_CERTIFICATE_P12_BASE64`, `APPLE_CERTIFICATE_PASSWORD`, `APPLE_CERTIFICATE_NAME`,
  `MACOS_KEYCHAIN_PASSWORD`), signs the binary with `codesign --options runtime
  --timestamp`, and notarizes it with `notarytool submit --wait` (`APPLE_ID`,
  `APPLE_ID_PASSWORD`, `APPLE_TEAM_ID`). A bare binary cannot be stapled, so the ticket
  stays with Apple and Gatekeeper checks it online.
- **Windows**: Azure Artifact Signing (formerly Trusted Signing) over GitHub OIDC
  (`TRUSTED_SIGNING_ENDPOINT`, `TRUSTED_SIGNING_ACCOUNT`, `CERTIFICATE_PROFILE`,
  `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`). It signs as Smoke Turner, LLC with the certificate
  profile every smoketurner project shares. The x86_64 signing tool runs under emulation
  on the arm64 runner. Signing runs on tags only: the federated identity credential
  (`smoketurner-infra`, `environments/azure/code_signing.tf`) trusts the exact subject
  `repo:smoketurner@8753311/quack@1366961169:workflow:Release:ref_type:tag`, so this
  repository's OIDC subject customization must include the `repo`, `workflow`, and
  `ref_type` claims. A `workflow_dispatch` run stays unsigned rather than failing the token
  exchange.

Verify a published asset or the image:

```bash
gh attestation verify quack-<version>-<target>.tar.gz --owner smoketurner \
  --signer-workflow smoketurner/quack/.github/workflows/reusable-build.yml
gh attestation verify oci://ghcr.io/smoketurner/quack:<version> --owner smoketurner \
  --signer-workflow smoketurner/quack/.github/workflows/reusable-build.yml
```

Every action is SHA-pinned. A release never writes a cache a later run could be poisoned
by: `reusable-build.yml` uses no Actions cache, and `release.yml`'s gates job runs
`rust-cache` restore-only (`save-if: false`).

## Container images

- **`Dockerfile`**: the image from source, for `make image` and the compose file. A CSS
  stage rebuilds the Tailwind stylesheet with the standalone binary (checksum verified),
  cargo-chef caches the dependency build, and the musl binary lands in
  `gcr.io/distroless/static-debian13:nonroot`. Environment: `QUACK_DATA_DIR=/data`,
  `QUACK_CONFIG_DIR=/config`, `QUACK_BIND=0.0.0.0:8080`. Entrypoint `/quack`, default
  command `serve`, so `docker run ... quack user add alice --admin` also works.
- **`Dockerfile.release`**: the same runtime from prebuilt `dist/linux-<arch>/quack`
  binaries; no compilation, so multi-arch builds need no emulation.
- **`.dockerignore`**: a deny-by-default allowlist that keeps the context small and
  cache-stable.
- Keep the `rust:<version>-alpine` tag equal to `rust-toolchain.toml`.

## Compose

`docker-compose.yml` runs `quack serve` beside `ollama/ollama`. It mounts
`deploy/config.toml` at `/config` (chat and embedding models on the `ollama` service) and
uses named volumes for `/data` and the models. `stop_grace_period` is 30 seconds, above
`[server].shutdown_grace_seconds` (20), so `docker compose stop` lets quack cancel its jobs
and checkpoint each workspace before Docker sends SIGKILL. First run:

```bash
docker compose up -d
docker compose exec ollama ollama pull gpt-oss:20b
docker compose exec ollama ollama pull qwen3-embedding:0.6b
docker compose exec quack /quack user add alice --admin
```

On an air-gapped host: `docker load < quack-image-<version>-linux-amd64.tar`, copy the
Ollama model directory into the `ollama` volume, point the `image:` line at the loaded tag,
and `docker compose up -d`.

## Local checks

- Run `actionlint` and `zizmor .github/workflows/*.yml` before committing a workflow
  change; both pass on the shipped files. `.github/actionlint.yaml` ignores one finding:
  actionlint 1.7 does not know the `$/` self-repository `uses:` form that zizmor asks for.
- `docker buildx build --check -f Dockerfile .` (and `Dockerfile.release`,
  `Dockerfile.build`) validates the Dockerfiles without building.
- CI tests on Linux, macOS, and Windows (the Windows entry builds uncached, so it is the
  slowest), but only a `workflow_dispatch` release run exercises the release builds
  themselves. Run one before tagging after any build change.
- Pin every new action to a SHA (`secure_workflows.yml` enforces it). Resolve current SHAs
  with `gh api repos/<owner>/<repo>/commits/<tag> --jq .sha`.

Two patterns to add when the code calls for them:

- **Fuzzing**: once a parser takes untrusted input at scale, add a detached `fuzz/` crate
  (`cargo-fuzz` + `libfuzzer-sys`, its own empty `[workspace]`) and gitignore
  `fuzz/corpus/` and `fuzz/artifacts/`.
- **Docs site**: when `docs/` outgrows flat files, migrate to mdBook (`docs/book.toml` +
  `src/SUMMARY.md`), deploy via a GitHub Pages workflow, and gitignore `docs/book/`.
