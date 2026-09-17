# Storm Events glossary

How to read the `events`, `fatalities`, and `locations` tables, which hold NOAA's Storm
Events Database for 2024, and the `notable_events` and `states` tables derived from them.
The official column reference is the "Storm Data Bulk Data Format" document; the rules the
National Weather Service follows when it writes an entry are in NWS Instruction 10-1605.

## Episodes and events

An **episode** is one weather system as a Weather Forecast Office (WFO) experienced it:
a squall line, a winter storm, a hurricane's passage. An episode has one
`episode_narrative` and many **events**. An **event** is one kind of weather in one
county, forecast zone, or marine zone during that episode: a tornado, a hail report, a
flash flood. Every event has an `event_id` and belongs to an `episode_id`; a tornado
that crosses a county line is two events, one per county, tied together by the
`tor_other_*` columns. `event_narrative` describes that event alone.

## Where an event happened

- `cz_type` is `C` for a county or parish, `Z` for an NWS public forecast zone, and
  `M` for a marine zone. `cz_fips` is the county FIPS code or the zone number, and
  `cz_name` the county or zone name. Zone-based events (winter weather, heat, drought,
  wind) use zones; point events (tornadoes, hail, thunderstorm wind, flash floods) use
  counties.
- `state` is the full state, territory, or marine area name in capitals
  (`NORTH CAROLINA`, `GULF OF MEXICO`), and `state_fips` its numeric FIPS code.
- `wfo` is the three-letter identifier of the Weather Forecast Office that wrote the
  entry (`GSP` is Greenville-Spartanburg, `MFL` is Miami, `LKN` is Elko).
- `begin_location`, `begin_range`, and `begin_azimuth` place the start of the event
  relative to a named place: `2.5 NW ASHEVILLE` means 2.5 miles northwest of Asheville.
  `begin_lat` and `begin_lon` are its coordinates. The `locations` table repeats these
  points, one row per point, with `location_index` ordering them along the event's path.

## When an event happened

`begin_date_time` and `end_date_time` are text in the form `27-SEP-24 04:00:00`, in the
local time named by `cz_timezone` (`EST-5`, `CST-6`). Parse them with
`strptime(begin_date_time, '%d-%b-%y %H:%M:%S')`. `begin_yearmonth` is `YYYYMM` as a
number and `year`, `month_name`, `begin_day`, and `begin_time` (`HHMM` as a number) are
the same instant broken out. Loading adds a parsed `begin_time_ts` and `end_time_ts` to
`events`, so `date_trunc('month', begin_time_ts)` works directly.

## Casualties

`deaths_direct` and `injuries_direct` count people killed or hurt by the weather itself:
struck by a tree, swept away by floodwater, hit by debris. `deaths_indirect` and
`injuries_indirect` count those killed or hurt by something the weather set off: a
traffic accident on an icy road, a heart attack while shoveling snow, a fire started by
lightning. The `fatalities` table has one row per person killed, direct or indirect:

- `fatality_type` is `D` for direct and `I` for indirect.
- `fatality_location` says where the person was: `Vehicle/Towed Trailer`,
  `Permanent Home`, `Mobile/Trailer Home`, `Outside/Open Areas`, `In Water`,
  `Under Tree`, `Boating`, `Camping`, `Ball Field`, `Church`, `Business`,
  `Permanent Structure`, `Heavy Equipment/Construction`, `Other`, or `Unknown`.
- `fatality_age` and `fatality_sex` (`M`, `F`) are recorded when known.
- `event_id` links the row to its event.

## Damage

`damage_property` and `damage_crops` are estimates in US dollars written as text with a
suffix: `10.00K` is ten thousand dollars, `1.50M` is one and a half million, `2.00B`
two billion, and `0.00K` means no damage was estimated. Loading adds
`damage_property_usd` and `damage_crops_usd` to `events` with the number already
parsed; sum those, not the text columns. Estimates come from the WFO, often from
emergency managers and insurers, and are revised as claims settle.

## Magnitude

`magnitude` and `magnitude_type` mean different things by event type:

- **Hail**: `magnitude` is the largest hailstone in inches (`1.00` is an inch,
  `1.75` is golf-ball size, `2.75` is baseball size). `magnitude_type` is empty.
- **Thunderstorm Wind, High Wind, Strong Wind, and the marine wind types**: `magnitude`
  is the wind speed in knots. `magnitude_type` says how it was known: `MG` a measured
  gust, `EG` an estimated gust, `MS` a measured sustained wind, `ES` an estimated
  sustained wind. Multiply knots by 1.151 for miles per hour.
- **Tornado**: `tor_f_scale` is the Enhanced Fujita rating (`EF0` through `EF5`, or
  `EFU` for unknown, used when a tornado is confirmed but no damage indicator could be
  rated). `tor_length` is the path length in miles and `tor_width` the maximum width in
  yards. The rating comes from the damage survey, not from a wind measurement; the EF
  Scale document explains the damage indicators.
- **Hurricane (Typhoon), Tropical Storm, Storm Surge/Tide**: `category` is the
  Saffir-Simpson category (1 to 5) at the time of the event, when recorded. The
  Saffir-Simpson document explains the wind ranges.
- **Flood and Flash Flood**: `flood_cause` is one of `Heavy Rain`,
  `Heavy Rain / Tropical System`, `Heavy Rain / Snow Melt`, `Heavy Rain / Burn Area`,
  `Dam / Levee Break`, `Ice Jam`, or `Planned Dam Release`.

## Sources

`source` is who reported the event to the WFO: `Public`, `Trained Spotter`,
`Emergency Manager`, `911 Call Center`, `Law Enforcement`, `Broadcast Media`,
`Newspaper`, `Mesonet` (a state or private observing network), `ASOS` and `AWOS`
(automated airport weather stations), `CoCoRaHS` (volunteer rain gauges),
`NWS Storm Survey`, `Drought Monitor`, `Utility Company`, `Fire Department/Rescue`,
`Amateur Radio`, `Official NWS Observations`, `Social Media`, and a few others.

## Event types in 2024

Astronomical Low Tide, Avalanche, Blizzard, Coastal Flood, Cold/Wind Chill, Debris Flow,
Dense Fog, Drought, Dust Devil, Dust Storm, Excessive Heat, Extreme Cold/Wind Chill,
Flash Flood, Flood, Freezing Fog, Frost/Freeze, Funnel Cloud, Hail, Heat, Heavy Rain,
Heavy Snow, High Surf, High Wind, Hurricane (Typhoon), Ice Storm, Lake-Effect Snow,
Lakeshore Flood, Lightning, Marine Dense Fog, Marine Hail, Marine High Wind, Marine
Hurricane/Typhoon, Marine Strong Wind, Marine Thunderstorm Wind, Marine Tropical
Depression, Marine Tropical Storm, Rip Current, Seiche, Sleet, Sneakerwave, Storm
Surge/Tide, Strong Wind, Thunderstorm Wind, Tornado, Tropical Depression, Tropical Storm,
Waterspout, Wildfire, Winter Storm, Winter Weather.

`Heat` and `Excessive Heat` differ by threshold: Excessive Heat is the level at which the
WFO issues an Excessive Heat Warning. `Flood` is a river or areal flood that develops
over hours; `Flash Flood` rises within six hours of the rain. `Thunderstorm Wind`
requires a gust of 50 knots or damage; `High Wind` is non-convective wind meeting the
warning criteria; `Strong Wind` is non-convective wind below them that still did
damage or hurt someone.

## Derived tables

- `notable_events` is the subset of `events` with any death or injury, an EF2 or
  stronger tornado, or property damage of a million dollars or more, with the parsed
  timestamps and dollar amounts and without the columns that are empty for most rows.
  It is the table the knowledge graph is built from.
- `states` has one row per state, the District of Columbia, and Puerto Rico from the
  Census Bureau's 2024 population estimates: `name` in capitals to match
  `events.state`, `fips`, `region`, and `population_2024`. `state_population` is the
  Census file as imported, with the nation, regions, and divisions still in it.
