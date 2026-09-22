# Nutrition workspace

USDA's FoodData Central, SR Legacy release: 7,793 foods with the amount of every
nutrient measured per 100 grams, household portions, and food categories, plus the FDA's
Daily Values and the Dietary Guidelines for Americans.

Tables:

- `nutrition`: one row per food with the headline nutrients as columns, per 100 g
  (`energy_kcal`, `protein_g`, `fat_g`, `saturated_fat_g`, `carbohydrate_g`, `fiber_g`,
  `sugars_g`, `cholesterol_mg`, `sodium_mg`, `potassium_mg`, `calcium_mg`, `iron_mg`,
  `magnesium_mg`, `zinc_mg`, `vitamin_a_mcg`, `vitamin_c_mg`, `vitamin_d_mcg`,
  `vitamin_e_mg`, `vitamin_k_mcg`, `folate_mcg`, `vitamin_b12_mcg`, `water_g`,
  `caffeine_mg`) and the `category` name. Start here.
- `food`, `nutrient`, `food_nutrient`: the full data, 474 nutrients and 644,000
  amounts, for any nutrient `nutrition` does not carry. Join `food_nutrient.fdc_id` to
  `food.fdc_id` and `food_nutrient.nutrient_id` to `nutrient.id`.
- `food_portion` and `measure_unit`: household measures with `gram_weight`, joined by
  `fdc_id` and `measure_unit_id`.
- `food_category`: the 25 food groups, joined by `food.food_category_id`.
- `daily_values`: the FDA Daily Value per nutrient with `nutrient_id`, `unit`, and
  `goal` (`get enough`, `limit`, `reference`).
- `nutrient_claims`: every food whose first listed serving is a `good source` (10% to
  19% of the Daily Value) or `excellent source` (20% or more) of one of thirteen
  nutrients (`daily_values.in_graph = 'yes'`), with the `serving`, `amount_per_serving`,
  and `percent_dv`. The knowledge graph is built from it. For other nutrients, other
  portions, or foods without a portion, compute from `food_nutrient` and `daily_values`.

Rules:

- Every amount is per 100 g. Say so, or convert to a serving with
  `food_portion.gram_weight / 100` and name the portion used. A "large egg" is the
  `1 large` portion of `Egg, whole, raw, fresh`.
- Match foods by `description ILIKE '%word%'` and prefer the plain form (`raw`, `whole`,
  no brand, no `with added`) unless the question names one. Say which food you used.
  Drinks are described as `Beverages, coffee, brewed, ...`, `Beverages, tea, ...`. Never
  say a food is missing until an `ILIKE` search on the plain word returned nothing; the
  tables are not documents, so `search_documents` does not find foods.
- Percent Daily Value is `100 * amount / daily_values.daily_value` for the matching
  `nutrient_id`. Do not compare a per-100 g amount with a per-serving Daily Value
  without saying so.
- A missing nutrient row is not measured, not zero; `nutrition` columns are NULL then.
- "Fruit" and "vegetables" are categories that include juices, canned, and cooked forms;
  filter by `description` when the question means the whole raw food.
- Recommendations (how much of something to eat) come from the Dietary Guidelines;
  label rules (percent Daily Value, good source) from the FDA pages; measurement
  methods and codes from the SR28 documentation. Cite them. Use the tables for what is
  in a food.
- Use the knowledge graph (`search_graph`, `find_path`) for which foods are good or
  excellent sources of which nutrients; use SQL for amounts, rankings, and charts.
