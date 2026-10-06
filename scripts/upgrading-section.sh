#!/usr/bin/env bash
# Print one release's section of docs/upgrading.md: from its "## vTAG" heading
# to the next "## " heading. Exits 1 when the page has no section for the tag,
# which fails the release: every release documents its upgrade steps.
set -euo pipefail
tag=$1
page=${2:-docs/upgrading.md}
section=$(awk -v tag="$tag" '
  /^## / { if (found) exit; found = ($2 == tag) }
  found { print }
' "$page")
if [[ -z "$section" ]]; then
  echo "$page has no \"## $tag\" section" >&2
  exit 1
fi
printf '%s\n' "$section"
