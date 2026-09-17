#!/usr/bin/env bash
# Lay out release assets: one archive per target from the built binaries,
# SHA256SUMS over all of them, and the Homebrew formula.
#
#   scripts/release/package.sh VERSION BINARIES_DIR DIST_DIR
#
# BINARIES_DIR holds one directory per target with the binary inside
# (quack, or quack.exe for Windows). DIST_DIR receives
# quack-VERSION-<target>.tar.gz (or .zip), SHA256SUMS, and quack.rb.
set -euo pipefail

version="$1"
binaries="$2"
dist="$3"
mkdir -p "$dist"
dist="$(cd "$dist" && pwd)"

for dir in "$binaries"/*/; do
  target="$(basename "$dir")"
  case "$target" in
    *windows*)
      asset="quack-${version}-${target}.zip"
      (cd "$dir" && zip -q "$dist/$asset" quack.exe)
      ;;
    *)
      asset="quack-${version}-${target}.tar.gz"
      chmod 0755 "$dir/quack"
      tar -C "$dir" -czf "$dist/$asset" quack
      ;;
  esac
  echo "packaged $asset"
done

(cd "$dist" && shasum -a 256 quack-* > SHA256SUMS)
"$(dirname "$0")/homebrew-formula.sh" "$version" "$dist" > "$dist/quack.rb"
echo "wrote $dist/SHA256SUMS and $dist/quack.rb"
