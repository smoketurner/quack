#!/usr/bin/env bash
# Load the logistics example (see README.md in this directory) into a quack
# workspace. Idempotent: files already in the workspace are skipped.
#
#   examples/logistics/load.sh [WORKSPACE] [--reset]     default: logistics
#
# --reset deletes every document already in the workspace first.
set -euo pipefail

workspace="logistics"
reset=0
for arg in "$@"; do
  case "$arg" in
  --reset) reset=1 ;;
  *) workspace="$arg" ;;
  esac
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cache="${QUACK_EXAMPLE_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/quack/examples/logistics}"
quack="${QUACK_BIN:-}"
if [ -z "$quack" ]; then
  if [ -x target/debug/quack ]; then
    quack=target/debug/quack
  elif command -v quack >/dev/null 2>&1; then
    quack=quack
  else
    echo "quack binary not found; run 'cargo build --bin quack' or set QUACK_BIN" >&2
    exit 1
  fi
fi

mkdir -p "$cache"

fetch() {
  local name="$1" url="$2"
  if [ ! -s "$cache/$name" ]; then
    echo "downloading $name"
    curl -fsSL -o "$cache/$name.part" "$url"
    mv "$cache/$name.part" "$cache/$name"
  fi
}

fetch shipments-raw.csv "https://web.archive.org/web/2024id_/https://data.usaid.gov/api/views/a3rc-nmf6/rows.csv?accessType=DOWNLOAD"
fetch importing-into-the-united-states.pdf "https://www.cbp.gov/sites/default/files/documents/Importing%20into%20the%20U.S.pdf"

# Column names like "po / so #" and "freight cost (usd)" become SQL-friendly
# identifiers: po_so_number, freight_cost_usd. Only the header line changes.
if [ ! -s "$cache/shipments.csv" ]; then
  awk 'NR == 1 {
         $0 = tolower($0)
         gsub(/#/, " number")
         gsub(/[^a-z0-9,]+/, "_")
         gsub(/_,/, ",")
         gsub(/,_/, ",")
         sub(/^_/, "")
         sub(/_$/, "")
       }
       { print }' "$cache/shipments-raw.csv" >"$cache/shipments.csv"
fi

present="$("$quack" docs -w "$workspace" --json 2>/dev/null || true)"

if [ "$reset" = 1 ] && [ -n "$present" ]; then
  echo "removing every document in '$workspace'"
  printf '%s\n' "$present" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | while read -r id; do
    "$quack" docs -w "$workspace" --delete "$id" >/dev/null
  done
  present=""
fi

filenames="$(printf '%s\n' "$present" | sed -n 's/.*"filename":"\([^"]*\)".*/\1/p')"

ingest() {
  local path="$1" name
  name="$(basename "$path")"
  if printf '%s\n' "$filenames" | grep -qx "$name"; then
    echo "already in '$workspace': $name"
    return
  fi
  echo "ingesting $name"
  "$quack" ingest "$path" -w "$workspace"
}

ingest "$cache/shipments.csv"
ingest "$here/documents/shipments-data-dictionary.md"
ingest "$here/documents/incoterms.md"
ingest "$cache/importing-into-the-united-states.pdf"

if ! "$quack" context show -w "$workspace" 2>/dev/null | grep -q "Logistics workspace"; then
  echo "setting the workspace context"
  "$quack" context import "$here/context.md" -w "$workspace"
fi

echo
echo "Workspace '$workspace' is ready. Try:"
echo "  $quack -w $workspace -p \"which vendors shipped the most by value, and what Incoterms did they use?\""
echo "  $quack -w $workspace -p \"how late were ocean shipments to Nigeria on average?\""
echo "  $quack -w $workspace -p \"under EXW, who pays freight and insurance?\" --mode query"
echo "  $quack -w $workspace -p \"what must a commercial invoice show for customs entry?\" --mode query"
