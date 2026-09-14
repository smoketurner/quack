#!/usr/bin/env bash
# Load public, permissively licensed sample data into a quack workspace so
# there is something worth asking about. Idempotent: files already in the
# workspace are skipped. Needs network access on the first run.
#
#   scripts/demo-data.sh [WORKSPACE]      default: demo
#
# Sources:
#   Palmer penguins (CC0)            https://github.com/allisonhorst/palmerpenguins
#   Gapminder and tips (MIT)         https://github.com/plotly/datasets
#   Seattle weather (BSD-3 repo; NOAA data is public domain)
#                                    https://github.com/vega/vega-datasets
#   NIST SP 800-63B (public domain)  https://pages.nist.gov/800-63-3/
set -euo pipefail

workspace="${1:-demo}"
cache="${QUACK_DEMO_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/quack/demo}"
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

fetch penguins-raw.csv "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
# The R convention writes missing values as NA; blank them so DuckDB types
# the measurement columns as numbers.
if [ ! -s "$cache/penguins.csv" ]; then
  awk -F, 'BEGIN { OFS = "," } { for (i = 1; i <= NF; i++) if ($i == "NA") $i = ""; print }' \
    "$cache/penguins-raw.csv" >"$cache/penguins.csv"
fi
fetch gapminder.csv "https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv"
fetch tips.csv "https://raw.githubusercontent.com/plotly/datasets/master/tips.csv"
fetch seattle_weather.csv "https://raw.githubusercontent.com/vega/vega-datasets/main/data/seattle-weather.csv"
fetch nist-sp-800-63b.pdf "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf"

present="$("$quack" docs -w "$workspace" --json 2>/dev/null | sed -n 's/.*"filename":"\([^"]*\)".*/\1/p' || true)"

for name in penguins.csv gapminder.csv tips.csv seattle_weather.csv nist-sp-800-63b.pdf; do
  if printf '%s\n' "$present" | grep -qx "$name"; then
    echo "already in '$workspace': $name"
    continue
  fi
  echo "ingesting $name"
  "$quack" ingest "$cache/$name" -w "$workspace"
done

context="$cache/context.md"
cat >"$context" <<'CTX'
# Demo workspace

Tables:
- `penguins`: 344 Palmer Station penguins. `species` (Adelie, Chinstrap, Gentoo), `island`,
  bill and flipper measurements in millimetres, `body_mass_g`, `sex`, `year`. Blank means missing.
- `gapminder`: country, continent, year, life expectancy (`lifeExp`, years), population (`pop`),
  GDP per capita (`gdpPercap`, inflation-adjusted US dollars), 1952 to 2007.
- `tips`: restaurant bills. `total_bill` and `tip` in US dollars, `size` is the party size.
- `seattle_weather`: one row per day, 2012 to 2015. `precipitation` in millimetres,
  `temp_max` and `temp_min` in degrees Celsius, `wind` in metres per second, `weather`
  (drizzle, rain, sun, snow, fog).

Documents:
- NIST SP 800-63B, Digital Identity Guidelines: authentication and lifecycle management.
  Password (memorized secret) rules are in section 5.1.1.

Answer with numbers from the tables; cite the guideline when the question is about
authentication requirements.
CTX

if ! "$quack" context show -w "$workspace" 2>/dev/null | grep -q "Demo workspace"; then
  echo "setting the workspace context"
  "$quack" context import "$context" -w "$workspace"
fi

echo
echo "Workspace '$workspace' is ready. Try:"
echo "  $quack -w $workspace -p \"which penguin species is heaviest on average?\""
echo "  $quack -w $workspace -p \"how did life expectancy in Asia change from 1952 to 2007?\""
echo "  $quack -w $workspace -p \"which month in Seattle has the most rain, and how much?\""
echo "  $quack -w $workspace -p \"what does the guideline say about minimum password length?\" --mode query"
