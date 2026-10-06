#!/usr/bin/env bash
# Seed each fuzz target's corpus from the evaluation fixtures, by extension.
# The corpus directories are gitignored; libFuzzer grows them as it runs.
set -euo pipefail
cd "$(dirname "$0")"
fixtures=../crates/quack-core/eval
seed() {
  local target=$1
  shift
  mkdir -p "corpus/${target}"
  for ext in "$@"; do
    find "${fixtures}" -type f -name "*.${ext}" -exec cp {} "corpus/${target}/" \; 2>/dev/null || true
  done
}
seed pdf pdf
seed markdown md
seed text txt md
seed html html htm
seed docx docx
seed pptx pptx
seed xlsx xlsx xlsm xlsb
seed chunker md txt
