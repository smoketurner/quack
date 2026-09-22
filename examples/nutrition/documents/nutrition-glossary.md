# Nutrition glossary

How to read the `food`, `nutrient`, `food_nutrient`, `food_portion`, `measure_unit`,
and `food_category` tables, which hold the SR Legacy release of USDA's FoodData Central,
the `daily_values` table from the FDA, and the `nutrition` and `nutrient_claims` tables
derived from them. The official column reference is the "FoodData Central Download Field
Descriptions" document; how each nutrient was measured, and what every derivation code
means, is in the SR28 documentation.

## Foods

SR Legacy is the final release of USDA's Standard Reference database: 7,793 foods, each
described the way a nutritionist would, most specific term last: `Egg, whole, raw,
fresh`; `Milk, whole, 3.25% milkfat, with added vitamin D`; `Spinach, raw`; `Beef, ground,
85% lean meat / 15% fat, raw`. Search descriptions with `ILIKE '%spinach%'`; the word a
person uses is usually the first word of the description, except that drinks start with
`Beverages,` (`Beverages, coffee, brewed, prepared with tap water`) and fast food with
the chain's name.

- `fdc_id` is the food's key everywhere. `description` is the name above.
- `food_category_id` joins to `food_category`, whose `description` is one of 25 groups:
  Dairy and Egg Products, Spices and Herbs, Baby Foods, Fats and Oils, Poultry Products,
  Soups, Sauces, and Gravies, Sausages and Luncheon Meats, Breakfast Cereals, Fruits and
  Fruit Juices, Pork Products, Vegetables and Vegetable Products, Nut and Seed Products,
  Beef Products, Beverages, Finfish and Shellfish Products, Legumes and Legume Products,
  Lamb, Veal, and Game Products, Baked Products, Sweets, Cereal Grains and Pasta, Fast
  Foods, Meals, Entrees, and Side Dishes, Snacks, American Indian/Alaska Native Foods,
  Restaurant Foods, and Alcoholic Beverages. `code` is the four-digit SR food group.
- "Fruit" means the `Fruits and Fruit Juices` category, which includes juices; filter
  with `description NOT ILIKE '%juice%'` for whole fruit. "Vegetables" likewise includes
  cooked, canned, and frozen forms; `raw` in the description means the raw food.

## Nutrient amounts are per 100 grams

Every amount in `food_nutrient` and every column of `nutrition` is **per 100 grams of the
food as described**, not per serving. `amount` is in the nutrient's `unit_name`: `G`
grams, `MG` milligrams, `UG` micrograms, `KCAL` kilocalories. To get a serving, multiply
by `gram_weight / 100` from `food_portion`.

- `nutrient.id` is the key; `nutrient_nbr` is the older SR nutrient number. The headline
  nutrients: Energy `1008` (kcal), Protein `1003`, Total lipid (fat) `1004`, Carbohydrate
  `1005`, Fiber `1079`, Sugars total `2000`, Cholesterol `1253`, Sodium `1093`, Potassium
  `1092`, Calcium `1087`, Iron `1089`, Magnesium `1090`, Zinc `1095`, Vitamin A RAE `1106`,
  Vitamin C `1162`, Vitamin D `1114`, Vitamin E `1109`, Vitamin K `1185`, Folate DFE
  `1190`, Vitamin B-12 `1178`, Water `1051`, Caffeine `1057`, Saturated fat `1258`.
- Energy appears twice, `1008` in kcal and `1062` in kJ. Use `1008`.
- A missing row means the nutrient was not analyzed for that food, not zero.
- `derivation_id` says how the value was obtained (`A` analytical, `AS` analytical with
  summed components, `NC` calculated from a recipe, `BF` borrowed from a similar food);
  the SR28 documentation lists every code.

`nutrition` has one row per food with the headline nutrients above as columns
(`energy_kcal`, `protein_g`, `fiber_g`, `vitamin_c_mg`, ...), still per 100 g, plus the
`category` name. Use it for most questions; use `food_nutrient` for a nutrient it does
not carry.

## Portions

`food_portion` gives household measures and their `gram_weight`: `amount` of a
`measure_unit` (`cup`, `tablespoon`, `oz`, `slice`), or, for most rows, the unnamed unit
`9999` (`undetermined`) with the measure in `modifier`: `large`, `cup`, `medium (7" to
7-7/8" long)`, `NLEA serving`. Read `modifier` first, then the unit name. A large egg is
`50 g` (`modifier = 'large'`); a cup of raw spinach is `30 g`; a medium banana is `118 g`.
`seq_num` orders a food's portions; most foods have one to eight, some none.

## Daily Values

`daily_values` is the FDA's Daily Value table for adults and children four years and
older, the reference behind the percent Daily Value on a Nutrition Facts label:
`nutrient`, `daily_value` in `unit`, and `nutrient_id` when SR Legacy measures it (biotin,
chloride, chromium, iodine, molybdenum, and added sugars have no SR column). `goal` is
`get enough` for the nutrients a diet should reach, `limit` for fat, saturated fat,
cholesterol, and sodium, and `reference` for total carbohydrate.

Percent Daily Value per 100 g is `100 * amount / daily_value`; per serving, multiply by
the portion's `gram_weight / 100`.

## Nutrient content claims

The FDA lets a label call a food a **good source** of a nutrient when one serving has 10%
to 19% of the Daily Value and an **excellent source** (or **high in**, **rich in**) at
20% or more. `nutrient_claims` applies those thresholds to the food's first listed
portion, for the thirteen nutrients marked `in_graph` in `daily_values` (calcium, fiber,
folate, iron, magnesium, potassium, vitamins A, B12, C, D, E, and K, zinc), one row per
food and nutrient: `claim` (`Spinach, raw: excellent source of Vitamin K`), `fdc_id` and
`food` (the description), `nutrient_id` and `nutrient` (the FDA name), the serving as
`portion_amount`, `portion_unit`, and `gram_weight`, `amount_per_serving` in `unit`,
`daily_value`, `percent_dv`, and `level` (`excellent source` or `good source`). A food
with no portion has no claims, and a nutrient outside the thirteen has none either;
compute those from `food_nutrient` and `daily_values` directly. It is the table the
knowledge graph is built from: a claim links a food to a nutrient, so
`search_graph('Spinach, raw')` lists what spinach is a source of and
`find_path('Kale, raw', 'Vitamin K')` goes through the claim.

## The documents

- The Dietary Guidelines for Americans, 2025-2030, is the joint HHS and USDA advice on
  what to eat: food groups, limits on added sugars, saturated fat, and sodium, and
  guidance by life stage. Cite it for recommendations.
- The FDA's Daily Value page defines the values in `daily_values` and the label rules;
  its Nutrition Facts label page explains how to read a label. Cite them for what a
  percent Daily Value means and what counts as a good source.
- The SR28 documentation and the FoodData Central field descriptions define every
  column, unit, and code. Cite them for what a number is.
