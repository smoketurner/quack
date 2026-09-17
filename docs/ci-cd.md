# CI/CD

## Continuous integration

- **`.github/workflows/ci.yml`** — `fmt`, `clippy` (`--locked -D warnings`), `test`
  (`cargo test --locked`, Linux + macOS), `dependency-review` (PRs), and `license-check`
  (`cargo-deny check`). Toolchain from `rust-toolchain.toml` via `rustup show`; caching via
  `Swatinem/rust-cache`; actions SHA-pinned; `permissions: {}` top-level with per-job
  `contents: read`.
- **`.github/workflows/secure_workflows.yml`** — fails CI if any third-party action is used
  without a full commit-SHA pin (`zgosalvez/github-actions-ensure-sha-pinned-actions`).
- **`.github/dependabot.yml`** — `cargo`, `github-actions`, and `docker` (the base-image
  tags), weekly, grouped, 7-day cooldown.

Work lands as commits on the main branch; CI runs on push. Releases are the only other
workflow, and it runs on tags alone so it never spends minutes on ordinary pushes.

## Releases (`release.yml`)

Trigger: an annotated `v*` tag pushed to the repository (for example `v0.2.0`), or
`workflow_dispatch` for a dry run that builds everything and publishes nothing. Jobs:

1. **gates** — `cargo fmt --check`, clippy, the test suite, `make release-gates`
   (`cargo tree -i ring -e normal` and `-i openssl-sys -e normal` must be empty: aws-lc-rs
   is the only crypto provider, design doc section 14), and `cargo deny check`.
2. **musl** — reproducible static binaries for `x86_64-unknown-linux-musl` and
   `aarch64-unknown-linux-musl`, each on a native runner (`ubuntu-latest`,
   `ubuntu-24.04-arm`) through `docker buildx bake ci` (`Dockerfile.build`,
   `docker-bake.hcl`): `rust:<MSRV>-alpine`, cargo-chef, `SOURCE_DATE_EPOCH=0`, a
   CycloneDX SBOM (`cargo-cyclonedx`) beside the binary. The GitHub Actions cache keeps the
   cooked dependency layer (DuckDB and aws-lc take most of the time).
3. **native** — `aarch64-apple-darwin` (`macos-latest`), `x86_64-apple-darwin`
   (`macos-15-intel`), `x86_64-pc-windows-msvc` (`windows-latest`) with a plain
   `cargo build --release --locked`.
4. **image** — the `quack serve` image for `linux/amd64` and `linux/arm64` from the musl
   binaries (`Dockerfile.release`: distroless static, nonroot, `/quack` and `/data`),
   pushed to `ghcr.io/smoketurner/quack:<version>` and `:latest` on a tag, plus one
   `docker save` tarball per architecture for `docker load` on air-gapped hosts.
5. **publish** — `scripts/release/package.sh` turns the binaries into
   `quack-<version>-<target>.tar.gz` (`.zip` on Windows), writes `SHA256SUMS`, and
   generates the Homebrew formula (`scripts/release/homebrew-formula.sh`, drop it into a
   tap as `Formula/quack.rb`); `gh release create --generate-notes` attaches all of it,
   the image tarballs included.

Every action is SHA-pinned; `rust-cache` runs with `lookup-only` in the release workflow so
release artifacts never write to a cache that a pull request could poison.

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

- `actionlint` and `zizmor .github/workflows` before committing a workflow change; both
  run clean on the shipped files.
- `docker buildx build --check -f Dockerfile .` (and `Dockerfile.release`,
  `Dockerfile.build`) validates the Dockerfiles without building.
- `scripts/release/package.sh 0.0.0 <binaries> <dist>` can be exercised locally with any
  `quack` binaries laid out per target.
- Pin every new action to a SHA (`secure_workflows.yml` enforces it) — resolve current SHAs
  with `gh api repos/<owner>/<repo>/commits/<tag> --jq .sha`.

Two more patterns to reach for when the code calls for them:

- **Fuzzing** — once a parser takes untrusted input at scale, add a detached `fuzz/` crate
  (`cargo-fuzz` + `libfuzzer-sys`, its own empty `[workspace]`) and gitignore
  `fuzz/corpus/` and `fuzz/artifacts/`.
- **Docs site** — when `docs/` outgrows flat files, migrate to mdBook (`docs/book.toml` +
  `src/SUMMARY.md`), deploy via a GitHub Pages workflow, and gitignore `docs/book/`.
