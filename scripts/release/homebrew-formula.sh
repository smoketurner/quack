#!/usr/bin/env bash
# Print a Homebrew formula for a release: the macOS and Linux archives with
# their checksums from DIST_DIR/SHA256SUMS. Drop the output into a tap as
# Formula/quack.rb (for example smoketurner/homebrew-tap).
#
#   scripts/release/homebrew-formula.sh VERSION DIST_DIR
set -euo pipefail

version="$1"
dist="$2"
base="https://github.com/smoketurner/quack/releases/download/v${version}"

sum() {
  awk -v name="quack-${version}-$1.tar.gz" '$2 == name { print $1 }' "$dist/SHA256SUMS"
}

cat <<FORMULA
class Quack < Formula
  desc "Air-gapped knowledge engine: documents, tables, and a knowledge graph behind one agent"
  homepage "https://github.com/smoketurner/quack"
  version "${version}"
  license any_of: ["Apache-2.0", "MIT"]

  on_macos do
    on_arm do
      url "${base}/quack-${version}-aarch64-apple-darwin.tar.gz"
      sha256 "$(sum aarch64-apple-darwin)"
    end
    on_intel do
      url "${base}/quack-${version}-x86_64-apple-darwin.tar.gz"
      sha256 "$(sum x86_64-apple-darwin)"
    end
  end

  on_linux do
    on_arm do
      url "${base}/quack-${version}-aarch64-unknown-linux-musl.tar.gz"
      sha256 "$(sum aarch64-unknown-linux-musl)"
    end
    on_intel do
      url "${base}/quack-${version}-x86_64-unknown-linux-musl.tar.gz"
      sha256 "$(sum x86_64-unknown-linux-musl)"
    end
  end

  def install
    bin.install "quack"
  end

  test do
    assert_match "quack", shell_output("#{bin}/quack --help")
  end
end
FORMULA
