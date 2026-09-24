#!/usr/bin/env bash
# Load the storms example (see README.md in this directory) into a quack
# workspace. Idempotent: files already in the workspace are skipped, derived
# tables are rebuilt only when missing, and the ontology and graph are written
# only when the workspace has none.
#
#   examples/storms/load.sh [WORKSPACE] [--reset]     default: storms
#
# --reset deletes every document (and the table it was loaded as) first.
# Nothing here calls a model; the README lists the model-driven steps.
set -euo pipefail

workspace="storms"
reset=0
for arg in "$@"; do
  case "$arg" in
  --reset) reset=1 ;;
  *) workspace="$arg" ;;
  esac
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cache="${QUACK_EXAMPLE_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/quack/examples/storms}"
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

# Some NOAA hosts refuse requests without a browser-like User-Agent.
agent="Mozilla/5.0 (quack example loader)"

fetch() {
  local name="$1" url="$2"
  if [ ! -s "$cache/$name" ]; then
    echo "downloading $name"
    curl -fsSL -A "$agent" -o "$cache/$name.part" "$url"
    mv "$cache/$name.part" "$cache/$name"
  fi
}

# The Storm Events files carry the date they were last regenerated in their
# names (StormEvents_details-ftp_v1.0_d2024_c20250401.csv.gz), so the loader
# reads the directory listing to find the current one for each table.
index="https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
fetch_storm_events() {
  local table="$1" name
  if [ -s "$cache/$table.csv" ]; then
    return
  fi
  if [ ! -s "$cache/index.html" ]; then
    curl -fsSL -A "$agent" -o "$cache/index.html" "$index"
  fi
  name="$(sed -n "s/.*href=\"\(StormEvents_${table}-ftp_v1.0_d2024_c[0-9]*\.csv\.gz\)\".*/\1/p" "$cache/index.html" | sort | tail -n 1)"
  if [ -z "$name" ]; then
    echo "no 2024 $table file listed at $index" >&2
    exit 1
  fi
  fetch "$name" "$index$name"
  gunzip -c "$cache/$name" >"$cache/$table.csv.part"
  mv "$cache/$table.csv.part" "$cache/$table.csv"
}

fetch_storm_events details
fetch_storm_events fatalities
fetch_storm_events locations
fetch Storm-Data-Bulk-csv-Format.pdf "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/Storm-Data-Bulk-csv-Format.pdf"
fetch nws-instruction-10-1605.pdf "https://www.weather.gov/media/directives/010_pdfs/pd01016005curr.pdf"
fetch enhanced-fujita-scale.html "https://www.spc.noaa.gov/efscale/"
fetch saffir-simpson-scale.html "https://www.nhc.noaa.gov/aboutsshws.php"

# The table takes the file's name, so the details file is loaded as `events`.
if [ ! -e "$cache/events.csv" ]; then
  ln -s details.csv "$cache/events.csv"
fi

present="$("$quack" docs -w "$workspace" --format json 2>/dev/null || true)"

sql() {
  "$quack" -q "$1" -w "$workspace" -f csv </dev/null
}

if [ "$reset" = 1 ] && [ -n "$present" ]; then
  echo "removing every document and derived table in '$workspace'"
  printf '%s\n' "$present" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | while read -r id; do
    "$quack" docs -w "$workspace" --delete "$id" >/dev/null
  done
  sql "DROP TABLE IF EXISTS notable_events" >/dev/null
  sql "DROP TABLE IF EXISTS states" >/dev/null
  present=""
fi

filenames="$(printf '%s\n' "$present" | sed -n 's/.*"filename":"\([^"]*\)".*/\1/p')"

ingest() {
  local path="$1" name
  shift
  name="$(basename "$path")"
  if printf '%s\n' "$filenames" | grep -qx "$name"; then
    echo "already in '$workspace': $name"
    return
  fi
  echo "ingesting $name"
  "$quack" ingest "$path" -w "$workspace" "$@"
}

has_table() {
  sql "SELECT count(*) FROM information_schema.tables WHERE table_name = '$1'" | tail -n 1 | grep -qx 1
}

has_column() {
  sql "SELECT count(*) FROM information_schema.columns WHERE table_name = '$1' AND column_name = '$2'" | tail -n 1 | grep -qx 1
}

ingest "$cache/events.csv"
ingest "$cache/fatalities.csv"
ingest "$cache/locations.csv"

# Dates and damage estimates are text in the file; add parsed columns once.
if ! has_column events damage_property_usd; then
  echo "adding parsed timestamp and damage columns to events"
  for column in begin_time_ts end_time_ts; do
    sql "ALTER TABLE events ADD COLUMN $column TIMESTAMP" >/dev/null
  done
  sql "UPDATE events SET begin_time_ts = strptime(BEGIN_DATE_TIME, '%d-%b-%y %H:%M:%S'), end_time_ts = strptime(END_DATE_TIME, '%d-%b-%y %H:%M:%S')" >/dev/null
  for column in damage_property_usd damage_crops_usd; do
    source="$(printf '%s' "$column" | sed 's/_usd$//' | tr '[:lower:]' '[:upper:]')"
    sql "ALTER TABLE events ADD COLUMN $column DOUBLE" >/dev/null
    sql "UPDATE events SET $column = CASE right($source, 1) WHEN 'K' THEN 1e3 WHEN 'M' THEN 1e6 WHEN 'B' THEN 1e9 END * TRY_CAST(left($source, length($source) - 1) AS DOUBLE) WHERE $source <> ''" >/dev/null
  done
fi

# Census population estimates, pulled straight over HTTP as a table, then
# shaped into one row per state with names that match events.STATE.
if ! has_table state_population; then
  echo "importing the Census population estimates"
  "$quack" import "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/totals/NST-EST2024-ALLDATA.csv" \
    --table state_population -w "$workspace"
fi
if ! has_table states; then
  echo "creating states"
  sql "CREATE TABLE states AS SELECT upper(NAME) AS name, STATE AS fips, CASE REGION WHEN '1' THEN 'Northeast' WHEN '2' THEN 'Midwest' WHEN '3' THEN 'South' WHEN '4' THEN 'West' ELSE 'Puerto Rico' END AS region, POPESTIMATE2024 AS population_2024 FROM state_population WHERE SUMLEV = '040' ORDER BY fips" >/dev/null
fi

# The graph is built from the events that matter: any casualty, an EF2 or
# stronger tornado, or property damage of a million dollars or more.
if ! has_table notable_events; then
  echo "creating notable_events"
  sql "CREATE TABLE notable_events AS SELECT EVENT_ID AS event_id, EPISODE_ID AS episode_id, EVENT_TYPE AS event_type, STATE AS state, STATE_FIPS AS state_fips, CZ_TYPE AS cz_type, CZ_NAME AS cz_name, WFO AS wfo, begin_time_ts, end_time_ts, DEATHS_DIRECT AS deaths_direct, DEATHS_INDIRECT AS deaths_indirect, INJURIES_DIRECT AS injuries_direct, INJURIES_INDIRECT AS injuries_indirect, damage_property_usd, damage_crops_usd, TOR_F_SCALE AS tor_f_scale, TOR_LENGTH AS tor_length, TOR_WIDTH AS tor_width, MAGNITUDE AS magnitude, MAGNITUDE_TYPE AS magnitude_type, FLOOD_CAUSE AS flood_cause, CATEGORY AS category, SOURCE AS source, BEGIN_LOCATION AS begin_location, BEGIN_LAT AS begin_lat, BEGIN_LON AS begin_lon, EVENT_NARRATIVE AS event_narrative FROM events WHERE DEATHS_DIRECT + DEATHS_INDIRECT + INJURIES_DIRECT + INJURIES_INDIRECT > 0 OR TOR_F_SCALE IN ('EF2', 'EF3', 'EF4', 'EF5') OR damage_property_usd >= 1e6 ORDER BY begin_time_ts" >/dev/null
fi

ingest "$here/documents/storm-events-glossary.md" --pin
ingest "$cache/Storm-Data-Bulk-csv-Format.pdf" --title "Storm Data Bulk Data Format"
ingest "$cache/nws-instruction-10-1605.pdf" --title "NWS Instruction 10-1605: Storm Data Preparation"
ingest "$cache/enhanced-fujita-scale.html" --title "The Enhanced Fujita Scale (SPC)"
ingest "$cache/saffir-simpson-scale.html" --title "Saffir-Simpson Hurricane Wind Scale (NHC)"

if ! "$quack" context show -w "$workspace" 2>/dev/null | grep -q "Storms workspace"; then
  echo "setting the workspace context"
  "$quack" context import "$here/context.md" -w "$workspace"
fi

build_graph="$reset"
if ! "$quack" ontology show -w "$workspace" 2>/dev/null | grep -q "storm_event"; then
  echo "importing the ontology"
  "$quack" ontology import "$here/ontology.json" -w "$workspace"
  build_graph=1
fi
if [ "$build_graph" = 1 ]; then
  echo "building the graph from the mapped tables"
  "$quack" graph extract --tables-only --reset -w "$workspace"
fi

echo
echo "Workspace '$workspace' is ready. Try:"
echo "  $quack -w $workspace -p \"which states had the most direct deaths, and from what kind of weather?\""
echo "  $quack -w $workspace -p \"chart property damage by month\""
echo "  $quack -w $workspace -p \"what is the difference between a direct and an indirect fatality?\" --mode query"
echo "  $quack -w $workspace graph search 'NORTH CAROLINA'"
echo "  $quack -w $workspace graph path 58277 GSP"
