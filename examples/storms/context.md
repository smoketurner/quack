# Storms workspace

NOAA's Storm Events Database for calendar year 2024: every weather event the National
Weather Service recorded in the United States and its waters, with casualties, damage
estimates, and narratives.

Tables:

- `events`: 69,801 events, one row per event type per county or zone. Keys are
  `event_id` and `episode_id`. Columns are uppercase in the file; DuckDB matches them
  case-insensitively. Loading added `begin_time_ts`, `end_time_ts` (timestamps parsed
  from the `DD-MON-YY HH:MM:SS` text), `damage_property_usd`, and `damage_crops_usd`
  (dollars parsed from `10.00K`, `1.50M`, `2.00B`). Use the parsed columns for dates
  and money; never sum the text `damage_property` column.
- `fatalities`: 1,141 people killed, one row each, with `fatality_type` (`D` direct,
  `I` indirect), age, sex, and `fatality_location`, linked by `event_id`.
- `locations`: the points along an event's path, linked by `event_id`.
- `notable_events`: events with a death, an injury, an EF2 or stronger tornado, or
  property damage of at least a million dollars. Smaller, already parsed, and the source
  of the knowledge graph.
- `states`: 2024 Census population estimates for the states, the District of Columbia,
  and Puerto Rico. Join `states.name = events.state` for per-capita questions;
  `state_population` is the raw Census file.

Rules:

- Wind `magnitude` is in knots; hail `magnitude` is inches; tornado strength is
  `tor_f_scale`, not `magnitude`. Read the glossary before writing SQL about magnitude.
- "Deaths" means `deaths_direct` unless the question says indirect or total. Say which
  you counted.
- Times are local to the event (`cz_timezone`); do not convert them.
- An episode is one storm system; an event is one hazard in one county or zone. Count
  tornadoes by events with `event_type = 'Tornado'`, and storms by `episode_id`.
- The knowledge graph links storm events to their episode, state, and forecast office,
  and each fatality to its event. Use it when the question is about what is connected
  to what; use SQL for counts and sums.

Documents: the glossary (pinned); NOAA's bulk CSV format reference for every column; NWS
Instruction 10-1605, the rules for what goes into Storm Data and how each event type is
defined and measured; the Storm Prediction Center's page on the Enhanced Fujita Scale;
and the National Hurricane Center's page on the Saffir-Simpson Hurricane Wind Scale.
Cite them for definitions, thresholds, and procedure; use the tables for what happened.
