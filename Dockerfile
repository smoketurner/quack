# syntax=docker/dockerfile:1
#
# The `quack serve` container image, built from source: one static musl
# binary in a distroless image with only /quack and /data (design doc
# section 14). Releases build the same image from the prebuilt binaries
# with Dockerfile.release; this file is for local builds and the compose
# file.
#
# Alpine's default Rust target is musl, so a plain `cargo build --release`
# produces a fully static binary that runs on the glibc-free distroless
# image. cargo-chef caches dependency compilation (DuckDB, aws-lc) so only
# the workspace crates recompile when the source changes.

# CSS stage: the standalone tailwindcss CLI (checksum verified) rebuilds the
# minified stylesheet that rust-embed bakes into the binary at compile time.
FROM debian:trixie-slim AS css-builder

ARG TARGETARCH

WORKDIR /app

# Checksums for v4.3.3 (verified 2026-09-16):
#   tailwindcss-linux-x64:   dc61b3ac6b8c9ca874c0cc4c57b2409791a64c5540404ca5f5367360babc313a
#   tailwindcss-linux-arm64: 55fd0b241214eff3de1e8ee4f22796662f2d2e7a49bcfca7477cfd0bac398195
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && case "$TARGETARCH" in \
         amd64) \
           BINARY="tailwindcss-linux-x64" \
           CHECKSUM="dc61b3ac6b8c9ca874c0cc4c57b2409791a64c5540404ca5f5367360babc313a" \
           ;; \
         arm64) \
           BINARY="tailwindcss-linux-arm64" \
           CHECKSUM="55fd0b241214eff3de1e8ee4f22796662f2d2e7a49bcfca7477cfd0bac398195" \
           ;; \
         *) \
           echo "Unsupported architecture: $TARGETARCH" && exit 1 \
           ;; \
       esac \
    && curl -sLO "https://github.com/tailwindlabs/tailwindcss/releases/download/v4.3.3/${BINARY}" \
    && echo "${CHECKSUM}  ${BINARY}" | sha256sum -c - \
    && chmod +x "${BINARY}" \
    && mv "${BINARY}" tailwindcss

# Everything Tailwind scans for class names, plus the stylesheet input.
COPY crates/quack/static crates/quack/static
COPY crates/quack/styles crates/quack/styles
COPY crates/quack/templates crates/quack/templates
COPY crates/quack/src crates/quack/src

RUN cd crates/quack \
    && /app/tailwindcss -i styles/input.css -o static/css/output.css --minify

# cargo-chef base stage, shared by the planner and the builder.
# Keep this Rust version in sync with rust-toolchain.toml.
FROM rust:1.98.0-alpine AS chef
RUN cargo install cargo-chef --locked
WORKDIR /app

# Planner: the dependency recipe from the workspace manifests.
FROM chef AS planner
COPY Cargo.toml Cargo.lock ./
COPY crates/quack-core/Cargo.toml crates/quack-core/
COPY crates/quack/Cargo.toml crates/quack/
RUN mkdir -p crates/quack-core/src crates/quack/src \
    && touch crates/quack-core/src/lib.rs crates/quack/src/main.rs
RUN cargo chef prepare --recipe-path recipe.json

# Builder: the musl static build.
FROM chef AS builder

ARG SOURCE_DATE_EPOCH=0

# Build dependencies for DuckDB (C++) and aws-lc-rs on musl:
#   cmake/make/clang/g++: build system, C and C++ compilers, libclang for bindgen
#   linux-headers/musl-dev: musl target headers
#   perl: aws-lc's assembly generation
#   go: AWS-LC's delocate pass over the FIPS module's generated assembly
#       (Linux uses the FIPS module: quack-core's target-gated "fips" feature)
# No openssl: aws-lc-rs is the only crypto provider (deny.toml bans openssl and ring).
RUN apk add --no-cache musl-dev pkgconfig cmake make perl clang linux-headers g++ go

# delocate cannot parse gcc's assembly output, so the FIPS module builds with
# clang while everything else keeps the default toolchain.
ENV AWS_LC_FIPS_SYS_CC=clang
ENV AWS_LC_FIPS_SYS_CXX=clang++

# Cook the dependencies (cached until Cargo.toml or Cargo.lock change).
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --locked --package quack --recipe-path recipe.json

# Restore the real manifests (cook leaves stubs).
COPY Cargo.toml Cargo.lock ./
COPY crates/quack-core/Cargo.toml crates/quack-core/
COPY crates/quack/Cargo.toml crates/quack/

# The source and the compile-time assets.
COPY crates/quack-core/src crates/quack-core/src
COPY crates/quack/src crates/quack/src
COPY crates/quack/templates crates/quack/templates
COPY --from=css-builder /app/crates/quack/static crates/quack/static

# Deterministic timestamps on the entry points for reproducible builds.
RUN touch -d "@${SOURCE_DATE_EPOCH}" crates/quack-core/src/lib.rs crates/quack/src/main.rs

RUN cargo build --release --locked --package quack

# An empty data directory to copy into the distroless image.
RUN mkdir -p /data && touch /data/.keep

# Runtime: distroless static, nonroot, nothing but /quack and /data.
FROM gcr.io/distroless/static-debian13:nonroot

WORKDIR /

LABEL org.opencontainers.image.source=https://github.com/smoketurner/quack
LABEL org.opencontainers.image.description="quack: an air-gapped knowledge engine (documents, tables, graph) with a web UI, REST API, and MCP server"
LABEL org.opencontainers.image.licenses="Apache-2.0 OR MIT"

COPY --from=builder /app/target/release/quack /quack
COPY --from=builder --chown=nonroot:nonroot /data /data

# The workspace files and control.db live under /data; mount a volume there.
# Config is read from /config/config.toml when that volume is mounted.
ENV RUST_LOG=info
ENV QUACK_DATA_DIR=/data
ENV QUACK_CONFIG_DIR=/config
ENV QUACK_BIND=0.0.0.0:8080

EXPOSE 8080

ENTRYPOINT ["/quack"]
CMD ["serve"]
