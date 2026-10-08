#!/usr/bin/env bash
# Install what building and gating this repository needs, on Linux (apt) or
# macOS (Homebrew). Every step is idempotent: it skips what is already there.
# `make setup` runs this, then `prek install`. The Claude Code session hook
# calls it too.
set -euo pipefail

os=$(uname)
have() { command -v "$1" >/dev/null 2>&1; }

case "$os" in
Linux)
  if have apt-get; then
    # cmake, clang, go: aws-lc-rs builds the FIPS module on Linux and its
    # delocate pass is a Go program (docs/crypto.md). libudev, libssl,
    # pkg-config: hidapi's build. The two linters are what the prek hooks run.
    pkgs=(cmake clang golang-go libudev-dev libssl-dev pkg-config shellcheck shfmt curl)
    if ! dpkg -s "${pkgs[@]}" >/dev/null 2>&1; then
      sudo apt-get update -qq || true
      sudo apt-get install -y -qq "${pkgs[@]}"
    fi
  else
    echo "no apt-get: install cmake, clang, go, shellcheck, and shfmt yourself" >&2
  fi
  ;;
Darwin)
  if ! have brew; then
    echo "Homebrew is needed on macOS: https://brew.sh" >&2
    exit 1
  fi
  # The macOS toolchain builds aws-lc-sys, not the FIPS module, so no go.
  for formula in cmake shellcheck shfmt actionlint zizmor prek tailwindcss pnpm; do
    have "$formula" || brew install "$formula"
  done
  ;;
*)
  echo "unsupported OS: $os (Linux with apt or macOS with Homebrew)" >&2
  exit 1
  ;;
esac

# Rust: rustup fetches the toolchain rust-toolchain.toml pins on first use.
if ! have rustup && [ ! -x "$HOME/.cargo/bin/rustup" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
fi

# Tools Homebrew covered above; on Linux they come from their own sources.
if [ "$os" = "Linux" ]; then
  if ! have prek && [ ! -x "$HOME/.local/bin/prek" ]; then
    pip3 install --user --quiet prek
  fi
  if ! have actionlint; then
    "$HOME/.cargo/bin/cargo" install actionlint --locked 2>/dev/null || echo "install actionlint: https://github.com/rhysd/actionlint" >&2
  fi
  if ! have zizmor; then
    pip3 install --user --quiet zizmor
  fi
  if ! have pnpm && have npm; then
    npm install -g --silent pnpm
  fi
  # Tailwind's standalone binary, for `make css-build`. The checksum file
  # the release publishes is checked before the binary is installed.
  if ! have tailwindcss; then
    case "$(uname -m)" in
    x86_64) tw_arch=x64 ;;
    aarch64 | arm64) tw_arch=arm64 ;;
    *) tw_arch="" ;;
    esac
    if [ -n "$tw_arch" ]; then
      base="https://github.com/tailwindlabs/tailwindcss/releases/latest/download"
      tw_dir=$(mktemp -d)
      curl -fsSL --max-time 120 -o "$tw_dir/tailwindcss" "$base/tailwindcss-linux-${tw_arch}"
      curl -fsSL --max-time 60 -o "$tw_dir/sha256sums.txt" "$base/sha256sums.txt"
      (cd "$tw_dir" && grep "tailwindcss-linux-${tw_arch}\$" sha256sums.txt | sed "s/tailwindcss-linux-${tw_arch}/tailwindcss/" | sha256sum -c -)
      sudo install -m 755 "$tw_dir/tailwindcss" /usr/local/bin/tailwindcss
      rm -r "$tw_dir"
    fi
  fi
fi

cat <<'MSG'
Set these before building on Linux (docs/crypto.md); a .env file in the
repository root is read by the Makefile:
  AWS_LC_FIPS_SYS_CC=clang
  AWS_LC_FIPS_SYS_CXX=clang++
MSG
