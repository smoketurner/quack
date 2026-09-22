#!/usr/bin/env bash
# Load the nutrition example (see README.md in this directory) into a quack
# workspace. Idempotent: files already in the workspace are skipped, derived
# tables are rebuilt only when missing, and the ontology and graph are written
# only when the workspace has none.
#
#   examples/nutrition/load.sh [WORKSPACE] [--reset]     default: nutrition
#
# --reset deletes every document (and the table it was loaded as) first.
# Nothing here calls a model; the README lists the model-driven steps.
set -euo pipefail

workspace="nutrition"
reset=0
for arg in "$@"; do
  case "$arg" in
  --reset) reset=1 ;;
  *) workspace="$arg" ;;
  esac
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cache="${QUACK_EXAMPLE_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/quack/examples/nutrition}"
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

# Some government hosts refuse requests without a browser-like User-Agent.
agent="Mozilla/5.0 (quack example loader)"

fetch() {
  local name="$1" url="$2"
  if [ ! -s "$cache/$name" ]; then
    echo "downloading $name"
    curl -fsSL -A "$agent" -o "$cache/$name.part" "$url"
    mv "$cache/$name.part" "$cache/$name"
  fi
}

# USDA's SR Legacy release of FoodData Central: one zip of CSV files. The
# tables quack loads are unpacked into the cache under their own names.
sr_zip="FoodData_Central_sr_legacy_food_csv_2018-04.zip"
fetch "$sr_zip" "https://fdc.nal.usda.gov/fdc-datasets/$sr_zip"
for table in food nutrient food_nutrient food_category food_portion measure_unit; do
  if [ ! -s "$cache/$table.csv" ]; then
    echo "unpacking $table.csv"
    unzip -p "$cache/$sr_zip" "*/$table.csv" >"$cache/$table.csv.part"
    mv "$cache/$table.csv.part" "$cache/$table.csv"
  fi
done

fetch dietary-guidelines-2025-2030.pdf "https://cdn.realfood.gov/DGA_508.pdf"
fetch sr28-documentation.pdf "https://www.ars.usda.gov/ARSUserFiles/80400535/DATA/SR/sr28/sr28_doc.pdf"
fetch fdc-field-descriptions.pdf "https://fdc.nal.usda.gov/docs/Download_Field_Descriptions_Oct2020.pdf"
fetch fda-daily-value.html "https://www.fda.gov/food/nutrition-facts-label/daily-value-nutrition-and-supplement-facts-labels"
fetch fda-nutrition-facts-label.html "https://www.fda.gov/food/nutrition-facts-label/how-understand-and-use-nutrition-facts-label"

present="$("$quack" docs -w "$workspace" --json 2>/dev/null || true)"

sql() {
  "$quack" -q "$1" -w "$workspace" -f csv </dev/null
}

if [ "$reset" = 1 ] && [ -n "$present" ]; then
  echo "removing every document and derived table in '$workspace'"
  printf '%s\n' "$present" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | while read -r id; do
    "$quack" docs -w "$workspace" --delete "$id" >/dev/null
  done
  sql "DROP TABLE IF EXISTS nutrient_claims" >/dev/null
  sql "DROP TABLE IF EXISTS nutrition" >/dev/null
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

ingest "$cache/food.csv"
ingest "$cache/nutrient.csv"
ingest "$cache/food_nutrient.csv"
ingest "$cache/food_category.csv"
ingest "$cache/food_portion.csv"
ingest "$cache/measure_unit.csv"
ingest "$here/daily_values.csv"

# One row per food with the headline nutrients as columns, per 100 g, so
# most questions are one SELECT instead of a pivot over food_nutrient.
if ! has_table nutrition; then
  echo "creating nutrition"
  sql "CREATE TABLE nutrition AS
    SELECT f.fdc_id,
           f.description,
           c.description AS category,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1008) AS energy_kcal,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1003) AS protein_g,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1004) AS fat_g,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1258) AS saturated_fat_g,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1005) AS carbohydrate_g,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1079) AS fiber_g,
           max(n.amount) FILTER (WHERE n.nutrient_id = 2000) AS sugars_g,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1253) AS cholesterol_mg,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1093) AS sodium_mg,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1092) AS potassium_mg,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1087) AS calcium_mg,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1089) AS iron_mg,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1090) AS magnesium_mg,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1095) AS zinc_mg,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1106) AS vitamin_a_mcg,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1162) AS vitamin_c_mg,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1114) AS vitamin_d_mcg,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1109) AS vitamin_e_mg,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1185) AS vitamin_k_mcg,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1190) AS folate_mcg,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1178) AS vitamin_b12_mcg,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1051) AS water_g,
           max(n.amount) FILTER (WHERE n.nutrient_id = 1057) AS caffeine_mg
    FROM food f
    JOIN food_category c ON c.id = f.food_category_id
    LEFT JOIN food_nutrient n ON n.fdc_id = f.fdc_id
    GROUP BY ALL
    ORDER BY f.fdc_id" >/dev/null
fi

# FDA nutrient content claims, per serving: a food is an 'excellent source' of
# a nutrient when one serving has 20% or more of the Daily Value, a 'good
# source' at 10% to 19%. The serving is the food's first listed household
# portion, and only the thirteen nutrients marked in_graph in daily_values
# take part, so the graph stays the size a person can walk. This is the table
# the knowledge graph is built from.
if ! has_table nutrient_claims; then
  echo "creating nutrient_claims"
  sql "CREATE TABLE nutrient_claims AS
    WITH serving AS (
      SELECT p.fdc_id,
             p.gram_weight,
             p.amount AS portion_amount,
             coalesce(nullif(u.name, 'undetermined'), nullif(p.modifier, ''), nullif(p.portion_description, ''), 'serving') AS portion_unit
      FROM food_portion p
      LEFT JOIN measure_unit u ON u.id = p.measure_unit_id
      QUALIFY row_number() OVER (PARTITION BY p.fdc_id ORDER BY p.seq_num, p.id) = 1)
    SELECT f.description || ': ' || level || ' of ' || d.nutrient AS claim,
           f.fdc_id,
           f.description AS food,
           n.nutrient_id,
           d.nutrient,
           s.portion_amount,
           s.portion_unit,
           s.gram_weight,
           format('{} {} ({} g)', s.portion_amount, s.portion_unit, s.gram_weight) AS serving,
           round(n.amount * s.gram_weight / 100, 2) AS amount_per_serving,
           d.unit,
           d.daily_value,
           round(n.amount * s.gram_weight / d.daily_value, 1) AS percent_dv,
           level
    FROM food_nutrient n
    JOIN daily_values d ON d.nutrient_id = n.nutrient_id
    JOIN food f ON f.fdc_id = n.fdc_id
    JOIN serving s ON s.fdc_id = n.fdc_id
    CROSS JOIN LATERAL (SELECT CASE WHEN n.amount * s.gram_weight / 100 >= 0.2 * d.daily_value THEN 'excellent source' ELSE 'good source' END AS level)
    WHERE d.in_graph = 'yes' AND n.amount * s.gram_weight / 100 >= 0.1 * d.daily_value
    ORDER BY f.description, d.nutrient" >/dev/null
fi

ingest "$here/documents/nutrition-glossary.md" --pin
ingest "$cache/dietary-guidelines-2025-2030.pdf" --title "Dietary Guidelines for Americans, 2025-2030"
ingest "$cache/fda-daily-value.html" --title "Daily Value on the Nutrition and Supplement Facts Labels (FDA)"
ingest "$cache/fda-nutrition-facts-label.html" --title "How to Understand and Use the Nutrition Facts Label (FDA)"
ingest "$cache/sr28-documentation.pdf" --title "USDA National Nutrient Database for Standard Reference, Release 28: Documentation"
ingest "$cache/fdc-field-descriptions.pdf" --title "FoodData Central Download Field Descriptions"

if ! "$quack" context show -w "$workspace" 2>/dev/null | grep -q "Nutrition workspace"; then
  echo "setting the workspace context"
  "$quack" context import "$here/context.md" -w "$workspace"
fi

build_graph="$reset"
if ! "$quack" ontology show -w "$workspace" 2>/dev/null | grep -q "nutrient_claim"; then
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
echo "  $quack -w $workspace -p \"how much protein is in an egg?\""
echo "  $quack -w $workspace -p \"chart the ten fruits with the most vitamin C per 100 g\""
echo "  $quack -w $workspace -p \"what does the government say about added sugars?\" --mode query"
echo "  $quack -w $workspace graph search 'Spinach, raw' --hops 1"
echo "  $quack -w $workspace graph path 'Kale, raw' 'Vitamin K'"
