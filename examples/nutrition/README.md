# Nutrition: USDA FoodData Central and the FDA Daily Values

One public-domain dataset that exercises everything quack does: six linked tables with
real keys, amounts, and household measures for SQL and charts; the government documents
that define every code and threshold in those tables, for search and cited answers; a
hand-written ontology mapped onto the tables so the knowledge graph is built without a
model; and enough prose (the 2025-2030 Dietary Guidelines, the SR28 documentation) for
the model-driven extraction steps when a model is available. And a subject everyone can
ask about: how much protein is in an egg?

```bash
make demo-data                                      # load into the workspace named "nutrition"
examples/nutrition/load.sh WORKSPACE                # load into WORKSPACE (created if missing)
examples/nutrition/load.sh WORKSPACE --reset        # delete every document in WORKSPACE first, then load
```

Loading takes a few minutes after the downloads (about 10 MB, cached under
`~/.cache/quack/examples/nutrition/`), most of it embedding the documents. It is
idempotent: files already in the workspace are skipped, derived tables are built only
when missing, and the ontology and graph are written only once. Nothing in `load.sh`
calls a chat model.

## What is loaded

| Item | What it is | Source |
|---|---|---|
| `food` table | 7,793 foods as USDA describes them (`Egg, whole, raw, fresh`; `Spinach, raw`), keyed by `fdc_id`, each in a food category | [USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets), SR Legacy release. US government work, CC0. |
| `nutrient` table | 474 nutrients with their unit (`G`, `MG`, `UG`, `KCAL`) | same |
| `food_nutrient` table | 644,125 measurements: the amount of a nutrient in 100 g of a food, with how it was derived | same |
| `food_portion` and `measure_unit` tables | 14,449 household measures (`1 large` egg is 50 g, a `cup` of raw spinach 30 g) with their gram weight | same |
| `food_category` table | The 25 SR food groups (Dairy and Egg Products, Fruits and Fruit Juices, ...) | same |
| `daily_values` table | The FDA's Daily Values for adults, the reference behind percent Daily Value on a label, with the SR nutrient id for each and whether it is a nutrient to get enough of or to limit | [daily_values.csv](daily_values.csv), transcribed from the [FDA](https://www.fda.gov/food/nutrition-facts-label/daily-value-nutrition-and-supplement-facts-labels). Public domain. |
| `nutrition` table | One row per food with the headline nutrients as columns, per 100 g, and the category name, built with `CREATE TABLE ... AS SELECT` through `quack -q`; the table most questions need | derived |
| `nutrient_claims` table | 16,720 FDA nutrient content claims: every food whose first listed serving is a good source (10% to 19% of the Daily Value) or an excellent source (20% or more) of one of thirteen nutrients, by the FDA's own thresholds; the table the graph is built from | derived |
| `nutrition-glossary.md` | How to read the tables: per-100 g amounts, nutrient ids, portions, Daily Values, and what a claim means, pinned so it is in every prompt | [documents/](documents/) in this directory |
| Dietary Guidelines for Americans, 2025-2030 | HHS and USDA's advice on what to eat: food groups, limits on added sugars, saturated fat, and sodium, guidance by life stage | [realfood.gov](https://realfood.gov/) |
| FDA Daily Value page and Nutrition Facts label page | What a percent Daily Value means and how to read a label | [FDA](https://www.fda.gov/food/nutrition-facts-label) |
| SR28 documentation | USDA's reference for the nutrient database: every nutrient, unit, derivation code, and how values were measured | [USDA ARS](https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/methods-and-application-of-food-composition-laboratory/mafcl-site-pages/sr17-sr28/) |
| FoodData Central field descriptions | The column reference for the CSV files | [USDA](https://fdc.nal.usda.gov/data-documentation) |
| Workspace context | What the tables hold, that every amount is per 100 g, how to pick a food and convert to a serving, and when to cite which document | [context.md](context.md) |
| Ontology | Four classes (food, food category, nutrient, nutrient content claim), three relations, and mappings onto `nutrition`, `food_category`, `daily_values`, and `nutrient_claims` | [ontology.json](ontology.json) |
| Knowledge graph | Foods, their categories, the nutrients with a Daily Value, and a claim node for every good or excellent source, with a row-level source on every edge | `quack graph extract --tables-only` |

## Try it

Ask across the tables, with charts:

```bash
quack -w nutrition -p "how much protein is in an egg?"
quack -w nutrition -p "chart the ten fruits with the most vitamin C per 100 g"
quack -w nutrition -p "which cheeses have the most calcium, and what share of the daily value is 100 g?"   # joins daily_values
quack -w nutrition -p "rank the legumes by fiber per 100 g"
quack -w nutrition -p "how much caffeine is in a cup of brewed coffee?"                               # uses food_portion
```

Ask the documents, cited, in query mode:

```bash
quack -w nutrition -p "what does the government say about added sugars?" --mode query
quack -w nutrition -p "what counts as an excellent source of a nutrient on a label?" --mode query
quack -w nutrition -p "how were the values for vitamin D measured?" --mode query
```

Walk the graph:

```bash
quack -w nutrition graph search 'Spinach, raw'        # its category and every nutrient it is a good or excellent source of
quack -w nutrition graph search 'Iron' --hops 2       # every food with an iron claim
quack -w nutrition graph path 'Kale, raw' 'Vitamin K' # food -> claim -> nutrient
quack -w nutrition graph status
quack -w nutrition -p "which legumes are excellent sources of iron?"   # the agent uses search_graph
```

SQL without the agent, or with piped data:

```bash
quack -w nutrition -q "SELECT description, protein_g FROM nutrition WHERE category = 'Legumes and Legume Products' ORDER BY protein_g DESC LIMIT 10" -f markdown
quack -w nutrition -q "SELECT category, round(avg(sodium_mg)) AS sodium_mg FROM nutrition GROUP BY 1 ORDER BY 2 DESC" -f csv
cat ~/.cache/quack/examples/nutrition/food_portion.csv | quack -w nutrition -q "SELECT modifier, count(*) n FROM stdin GROUP BY 1 ORDER BY n DESC LIMIT 10"
```

Sessions, sharing, and the other interfaces:

```bash
quack -w nutrition -p "and per large egg?" -c                          # continue the last session
quack -w nutrition sessions ; quack -w nutrition export ID --markdown  # transcripts, or --sql for the statements
quack -w nutrition okf export ./nutrition-okf                          # the whole workspace as a Markdown bundle
quack serve --local                                                   # web UI: chat, tables, SQL, documents, the graph page, the ontology
quack mcp -w nutrition                                                # MCP over stdio; see .mcp.json below
```

For Claude Code or an editor, one line in `.mcp.json` gives the assistant `query`,
`search`, `sql`, `search_graph`, and `find_path` over this workspace:

```json
{ "mcpServers": { "nutrition": { "command": "quack", "args": ["mcp", "-w", "nutrition"] } } }
```

## With a model

The loader stops where chat-model calls begin. With a chat model configured, three more
steps finish the picture:

```bash
quack -w nutrition ontology propose --documents --extend --sample 40    # what the documents mention that the ontology lacks; review and accept
quack -w nutrition graph extract --documents-only --sample 100          # named entities and relations from the Dietary Guidelines and the SR28 documentation, with chunk provenance
quack -w nutrition graph merges                                         # near-duplicate labels the extractor proposed to merge
```

`[retrieval].rerank = "model"` in the config turns on reranking of retrieved chunks for
the query-mode questions above; the Dietary Guidelines are long enough for it to matter.

## Why this dataset

- **Real keys.** `fdc_id` ties foods to their nutrients and portions, `nutrient_id` ties
  measurements to the FDA's Daily Values, and `food_category_id` groups them, so
  ontology induction (`quack ontology propose`) finds the relations on its own and the
  shipped ontology maps them exactly.
- **Numbers that need documents.** Every amount is per 100 grams, a percent Daily Value
  depends on the FDA table, and "good source" is a legal threshold. The pinned glossary
  answers the common questions; the FDA pages and the SR28 documentation have the rest.
- **Numbers worth charting.** Vitamin C across fruits, fiber across legumes, sodium
  across categories, caffeine across beverages.
- **Prose for extraction.** The Dietary Guidelines name food groups, nutrients, and
  recommendations on every page, and the SR28 documentation describes methods and
  sources, so model-driven extraction has something to find.
- **Something people know.** Eggs, spinach, coffee, cheese: the questions write
  themselves.

## Adding an example

Make a directory under `examples/` with a `README.md` like this one, a `load.sh` that is
idempotent and takes `[WORKSPACE] [--reset]`, the documents it writes as real files, a
`context.md`, and an `ontology.json` when the tables have keys worth mapping. Only use
data whose license allows redistribution, and say what it is.
